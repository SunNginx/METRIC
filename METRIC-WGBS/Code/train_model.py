#!/usr/bin/env python3
"""Train METRIC-WGBS models with a guided LDA framework.

This script trains one METRIC-WGBS model for a specified number of top markers
and latent topics. It preserves the original METRIC-WGBS workflow while using
cleaner command-line arguments and relative path support.
"""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
from typing import Dict

import gensim
import numpy as np
import pandas as pd
import scipy.sparse
from gensim import models
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score

from gensim_Utils import (
    generate_ETA_v2,
    generate_test_corpus,
    generate_train_corpus,
    setInputOutputFile_v2,
)


def cal_mean_by_mixID(mydf: pd.DataFrame) -> pd.DataFrame:
    """Average replicate mixtures with the same mixture ID."""
    mydf = mydf.copy()
    mydf["rep"] = pd.to_numeric(mydf["rep"])
    return mydf.groupby("mixID").agg({col: "mean" for col in mydf.columns[2:]})


def calc_ccc(y_true, y_pred) -> float:
    """Calculate Lin's concordance correlation coefficient."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denominator == 0:
        return np.nan
    return (2 * cov) / denominator


def evaluate_model(
    model_deconv_results_df: pd.DataFrame,
    FILES: Dict[str, str],
    parameters: Dict[str, object],
) -> pd.DataFrame:
    """Evaluate deconvolution performance on benchmark mixtures.

    The evaluation assumes that test sample names follow the mixture naming
    convention used in the synthetic benchmark and that the sample metadata
    contains ``group`` and ``actual`` columns.
    """
    n_topics = int(parameters["n_topics"])
    n_top_words = int(parameters["n_top_words"])
    prefix = f"topic{n_topics}_topword{n_top_words}"

    data = model_deconv_results_df.copy()
    required_columns = {"name", "group", "actual"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"Evaluation requires columns {sorted(required_columns)}, "
            f"but these columns are missing: {sorted(missing_columns)}"
        )

    name_parts = data["name"].astype(str).str.split("_", expand=True)
    if name_parts.shape[1] < 4:
        raise ValueError(
            "Evaluation expects sample names with at least four underscore-separated fields "
            "so that replicate and mixture IDs can be inferred."
        )

    data["rep"] = name_parts[3]
    data["mixGroup"] = name_parts[0]
    data["mixRatio"] = name_parts[0] + "_" + name_parts[2]
    data["mixID"] = name_parts[1] + "_" + name_parts[2]

    all_means = pd.DataFrame()
    for group in data["group"].unique():
        if group not in data.columns:
            continue

        group_df = data.loc[data["group"] == group, ["mixID", "rep", "actual", group]].copy()
        group_df.columns = ["mixID", "rep", "actual", group]
        group_mean = cal_mean_by_mixID(group_df)

        if all_means.empty:
            all_means = group_mean
        else:
            all_means = pd.merge(all_means, group_mean, on=["actual"], how="left")

    final_result = all_means.reset_index().sort_values(by="actual")
    actual_values = final_result["actual"]

    rows = []
    for column in final_result.columns[2:]:
        predicted_values = final_result[column]

        rho, _ = spearmanr(actual_values, predicted_values)
        pearson_corr, _ = pearsonr(actual_values, predicted_values)

        x = np.asarray(actual_values, dtype=np.float64)
        y = np.asarray(predicted_values, dtype=np.float64)
        x = x / np.sum(x) if np.sum(x) != 0 else x
        y = y / np.sum(y) if np.sum(y) != 0 else y

        rows.append(
            {
                "celltype": column,
                "R2": r2_score(actual_values, predicted_values),
                "spearman_r2": rho**2,
                "PCC": pearson_corr,
                "RMSE": np.sqrt(mean_squared_error(actual_values, predicted_values)),
                "JSD": jensenshannon(x, y) ** 2,
                "CCC": calc_ccc(actual_values, predicted_values),
            }
        )

    result_df = pd.DataFrame(rows)
    average_row = {"celltype": "Average"}
    for metric in ["R2", "spearman_r2", "PCC", "RMSE", "JSD", "CCC"]:
        average_row[metric] = result_df[metric].mean()
    result_df = pd.concat([result_df, pd.DataFrame([average_row])], ignore_index=True)

    output_file = Path(FILES["outputDir"]) / f"{prefix}_evaluate_results.csv"
    result_df.to_csv(output_file, index=False)
    return result_df


def postprocess_and_save_results(
    lda: models.LdaModel,
    train_sets: Dict[str, object],
    test_sets: Dict[str, object],
    FILES: Dict[str, str],
    parameters: Dict[str, object],
) -> pd.DataFrame:
    """Save topic matrices, marker-topic weights and deconvolution results."""
    n_topics = int(parameters["n_topics"])
    n_top_words = int(parameters["n_top_words"])
    prefix = f"topic{n_topics}_topword{n_top_words}"
    output_dir = Path(FILES["outputDir"])

    topic_train_sample = lda.get_document_topics(train_sets["corpus"], minimum_probability=0.0)
    topic_train_sample_mat = gensim.matutils.corpus2csc(topic_train_sample, num_terms=n_topics)

    scipy.sparse.save_npz(output_dir / f"{prefix}_topic_trainSample_mat.npz", topic_train_sample_mat)

    train_sample_names = train_sets["sampleGroup"]["name"]
    topic_train_sample_df = pd.DataFrame(
        topic_train_sample_mat.todense(),
        index=[f"Topic {i}" for i in range(1, topic_train_sample_mat.shape[0] + 1)],
        columns=train_sample_names,
    )
    topic_train_sample_df.to_excel(output_dir / f"{prefix}_topic_trainSample_mat.xlsx")

    train_sample_topic_df = topic_train_sample_df.T.reset_index().rename(columns={"index": "name"})
    train_sample_topic_actual = pd.merge(
        train_sample_topic_df,
        train_sets["sampleGroup"],
        on="name",
        how="right",
    )
    train_sample_topic_actual.to_excel(
        output_dir / f"{prefix}_trainSample_topic_df_actual.xlsx",
        index=True,
        header=True,
    )

    topic_marker_mat = np.asarray(lda.get_topics())
    marker_topic_mat = topic_marker_mat.T
    marker_topic_file = output_dir / f"{prefix}_marker_topic_mat.txt"
    with open(marker_topic_file, "w", encoding="utf-8") as handle:
        handle.write("marker\t" + "\t".join([f"Topic{i}" for i in range(1, n_topics + 1)]) + "\n")
        for marker, weights in zip(train_sets["DUI"].index.to_list(), marker_topic_mat):
            handle.write(marker + "\t" + "\t".join(map(str, weights)) + "\n")

    celltypes = sorted(set(train_sets["sampleGroup"]["group"]))
    celltype_topic_sum = {celltype: np.zeros(n_topics, dtype=float) for celltype in celltypes}
    celltype_counts = {celltype: 0 for celltype in celltypes}

    for sample_index, celltype in enumerate(train_sets["sampleGroup"]["group"]):
        celltype_topic_sum[celltype] += topic_train_sample_mat[:, sample_index].toarray().ravel()
        celltype_counts[celltype] += 1

    celltype_topic_mean = {
        celltype: celltype_topic_sum[celltype] / celltype_counts[celltype]
        for celltype in celltypes
    }

    topic_celltype_df = pd.DataFrame(celltype_topic_mean)
    topic_celltype_df.to_csv(output_dir / f"{prefix}_topic_celltype_mat.txt", sep="\t")
    topic_celltype_df.to_excel(output_dir / f"{prefix}_topic_celltype_mat.xlsx")

    topic_test_sample = lda.get_document_topics(test_sets["corpus"], minimum_probability=0.0)
    topic_test_sample_mat = gensim.matutils.corpus2csc(topic_test_sample, num_terms=n_topics)

    scipy.sparse.save_npz(output_dir / f"{prefix}_topic_testSample_mat.npz", topic_test_sample_mat)

    test_sample_names = test_sets["sampleGroup"]["name"]
    topic_test_sample_df = pd.DataFrame(
        topic_test_sample_mat.todense(),
        index=[f"Topic {i}" for i in range(1, topic_test_sample_mat.shape[0] + 1)],
        columns=test_sample_names,
    )
    topic_test_sample_df.to_csv(output_dir / f"{prefix}_topic_testSample_mat.txt", sep="\t")

    test_sample_topic_df = topic_test_sample_df.T.reset_index().rename(columns={"index": "name"})
    test_sample_topic_actual = pd.merge(
        test_sample_topic_df,
        test_sets["sampleGroup"],
        on="name",
        how="right",
    )
    test_sample_topic_actual.to_excel(
        output_dir / f"{prefix}_testSample_topic_group_actual_df.xlsx",
        index=True,
        header=True,
    )

    celltype_topic_df = topic_celltype_df.T
    test_sample_celltype_array = np.dot(
        topic_test_sample_mat.toarray().T,
        celltype_topic_df.iloc[:, :n_topics].T,
    )

    test_sample_celltype_df = pd.DataFrame(
        test_sample_celltype_array,
        columns=celltype_topic_df.index,
        index=test_sample_names,
    )
    test_sample_celltype_df.reset_index(inplace=True)
    test_sample_celltype_df.rename(columns={"index": "name"}, inplace=True)

    test_sample_celltype_actual = pd.merge(
        test_sample_celltype_df,
        test_sets["sampleGroup"],
        on="name",
        how="right",
    )
    test_sample_celltype_actual.to_excel(
        output_dir / f"{prefix}_testSample_celltype_unNorm_actual.xlsx",
        header=True,
        index=True,
    )

    row_sums = test_sample_celltype_array.sum(axis=1)
    row_sums[row_sums == 0] = np.nan
    test_sample_celltype_norm = np.divide(test_sample_celltype_array, row_sums[:, None])
    test_sample_celltype_norm = np.nan_to_num(test_sample_celltype_norm, nan=0.0)

    test_sample_celltype_norm_df = pd.DataFrame(
        test_sample_celltype_norm,
        columns=celltype_topic_df.index,
        index=test_sample_names,
    )
    test_sample_celltype_norm_df.to_csv(
        output_dir / f"{prefix}_testSample_celltype_frac.txt",
        sep="\t",
    )

    test_sample_celltype_norm_actual = (
        test_sample_celltype_norm_df.reset_index()
        .rename(columns={"index": "name"})
        .merge(test_sets["sampleGroup"], on="name", how="right")
    )
    test_sample_celltype_norm_actual.to_excel(
        output_dir / f"{prefix}_testSample_celltype_frac_actual.xlsx",
        header=True,
        index=True,
    )

    evaluate_model(test_sample_celltype_norm_actual, FILES, parameters)
    return test_sample_celltype_norm_actual


def train_single_model(
    parameters: Dict[str, object],
    mainPath: str | os.PathLike,
    selectID: str,
    topn: str = "250_U",
    force: bool = False,
) -> None:
    """Train one METRIC-WGBS model and save outputs."""
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    n_topics = int(parameters["n_topics"])
    n_top_words = int(parameters["n_top_words"])
    prefix = f"topic{n_topics}_topword{n_top_words}"

    print(f"\n=== Running: n_top_words={n_top_words}, n_topics={n_topics} ===")
    print(f"Start time: {datetime.datetime.now()}")

    try:
        files = setInputOutputFile_v2(
            mainPath=mainPath,
            selected_id=selectID,
            topn=topn,
            current_time=current_time,
            parameters=parameters,
        )

        topic_celltype_file = Path(files["outputDir"]) / f"{prefix}_topic_celltype_mat.txt"
        if topic_celltype_file.exists() and not force:
            topic_celltype_df_check = pd.read_csv(topic_celltype_file, sep="\t", index_col=0)
            actual_topic_num = topic_celltype_df_check.shape[0]

            print(f"[CHECK] Found topic-cell-type file: {topic_celltype_file}")
            print(f"[CHECK] Expected topics: {n_topics}; actual rows: {actual_topic_num}")

            if actual_topic_num == n_topics:
                print("[SKIP] Existing complete result found. Use --force to rerun.")
                return

            print("[RERUN] Existing result has inconsistent topic number. Rerunning.")

        train_sets = generate_train_corpus(
            file_path=files["trainSetFile"],
            marker_info_file=files["markerInfoFile"],
            sample_group_file=files["trainSampleGroupFile"],
            n_top_words=n_top_words,
            exempt_top_n=parameters.get("exempt_top_n"),
        )
        print("[INFO] Training marker counts by group:")
        print(train_sets["markerGroup"]["group"].value_counts())

        eta = generate_ETA_v2(
            train_sets["markerGroup"],
            train_sets["celltypes"],
            train_sets["markers"],
            bool(parameters["eta_norm"]),
            n_topics,
        )
        pd.DataFrame(eta).to_excel(Path(files["outputDir"]) / f"{prefix}_ETA.xlsx", index=False)

        lda = models.LdaModel(
            corpus=train_sets["corpus"],
            id2word=train_sets["dictionary"],
            num_topics=n_topics,
            random_state=int(parameters["random_state"]),
            offset=float(parameters["offset"]),
            iterations=int(parameters["iterations"]),
            passes=int(parameters["passes"]),
            alpha=parameters["alpha"],
            eval_every=int(parameters["eval_every"]),
            eta=eta,
            minimum_probability=0.0,
        )
        lda.save(files["lda_model_save_path"])
        print(f"[OK] Model saved to: {files['lda_model_save_path']}")

        feature_names = train_sets["markerGroup"]["marker"].tolist()
        test_sets = generate_test_corpus(
            file_path=files["testSetFile"],
            marker_info_file=files["markerInfoFile"],
            sample_group_file=files["testSetSampleGroupFile"],
            feature_markers=feature_names,
        )

        postprocess_and_save_results(lda, train_sets, test_sets, files, parameters)

    except Exception as exc:
        print(f"[ERROR] Training failed: {exc}")
        failed_file = Path(mainPath) / f"topic_{n_topics}_failed_placeholder.txt"
        with open(failed_file, "w", encoding="utf-8") as handle:
            handle.write("0\n")
        raise

    finally:
        print(f"End time: {datetime.datetime.now()}")
        print("======================================")


def build_default_parameters(n_top_words: int, n_topics: int) -> Dict[str, object]:
    """Create the default parameter dictionary for METRIC-WGBS."""
    return {
        "n_top_words": int(n_top_words),
        "n_topics": int(n_topics),
        "iterations": 200,
        "offset": 1,
        "passes": 20,
        "alpha": "auto",
        "eval_every": 10,
        "random_state": 0,
        "delta_means": 0.3,
        "delta_quants": 0.2,
        "delNan": False,
        "features": "DUI",
        "train_apply_diff": False,
        "train_feature_para": 100,
        "test_apply_diff": False,
        "test_feature_para": 3.5,
        "eta_marker": 100000,
        "eta_normal": 0.0001,
        "eta_norm": True,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a METRIC-WGBS guided LDA model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("n_top_words", type=int, help="Number of top markers per cell type.")
    parser.add_argument("n_topics", type=int, help="Number of latent topics.")
    parser.add_argument(
        "--main-path",
        default=str(Path(__file__).resolve().parent.parent),
        help="Project root containing input/ and results/ folders.",
    )
    parser.add_argument("--select-id", default="groupV15", help="Dataset identifier.")
    parser.add_argument("--topn", default="250_U", help="Marker-panel suffix used in input filenames.")
    parser.add_argument("--force", action="store_true", help="Force retraining even if outputs already exist.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    params = build_default_parameters(args.n_top_words, args.n_topics)
    train_single_model(
        parameters=params,
        mainPath=args.main_path,
        selectID=args.select_id,
        topn=args.topn,
        force=args.force,
    )
