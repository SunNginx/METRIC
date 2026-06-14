#!/usr/bin/env python3
"""Train and evaluate a METRIC guided LDA deconvolution model."""

from __future__ import annotations

import argparse
import datetime
import os
import traceback
from pathlib import Path

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
    """Average replicated mixture predictions by mixID."""
    mydf = mydf.copy()
    mydf["rep"] = pd.to_numeric(mydf["rep"])
    return mydf.groupby("mixID").agg({col: "mean" for col in mydf.columns[2:]})


def calc_ccc(y_true, y_pred) -> float:
    """Calculate Lin's concordance correlation coefficient (CCC)."""
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


def evaluate_model(model_deconv_results_df: pd.DataFrame, FILES: dict, parameters: dict) -> pd.DataFrame:
    """Evaluate deconvolution performance on synthetic mixture test samples.

    The sample name is expected to contain mixture metadata separated by
    underscores. The function calculates R2, Spearman R2, PCC, RMSE, JSD and
    CCC for each target cell type, plus an average row.
    """
    n_topics = parameters["n_topics"]
    n_top_words = parameters["n_top_words"]
    prefix = f"topic{n_topics}_topword{n_top_words}"

    data = model_deconv_results_df.copy()
    name_parts = data["name"].str.split("_", expand=True)
    if name_parts.shape[1] < 4:
        raise ValueError(
            "Sample names must contain at least four underscore-separated fields "
            "so that mixGroup, mixID, mixRatio and replicate can be parsed."
        )

    data["rep"] = name_parts[3]
    data["mixGroup"] = name_parts[0]
    data["mixRatio"] = name_parts[0] + "_" + name_parts[2]
    data["mixID"] = name_parts[1] + "_" + name_parts[2]

    all_means = pd.DataFrame()

    for group in data["group"].unique():
        if group not in data.columns:
            continue

        group_df = data.loc[data["group"] == group, ["mixID", "rep", "actual", group]]
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
        rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))

        x = np.asarray(actual_values, dtype=np.float64)
        y = np.asarray(predicted_values, dtype=np.float64)
        x = x / np.sum(x)
        y = y / np.sum(y)
        jsd = jensenshannon(x, y) ** 2

        rows.append(
            {
                "celltype": column,
                "R2": r2_score(actual_values, predicted_values),
                "spearman_r2": rho**2,
                "PCC": pearson_corr,
                "RMSE": rmse,
                "JSD": jsd,
                "CCC": calc_ccc(actual_values, predicted_values),
            }
        )

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        average_row = {"celltype": "Average"}
        for metric in ["R2", "spearman_r2", "PCC", "RMSE", "JSD", "CCC"]:
            average_row[metric] = result_df[metric].mean()
        result_df = pd.concat([result_df, pd.DataFrame([average_row])], ignore_index=True)

    output_path = Path(FILES["outputDir"]) / f"{prefix}_evaluate_results.csv"
    result_df.to_csv(output_path, index=False)
    return result_df


def _save_topic_sample_matrix(lda, corpus, sample_names, n_topics: int, output_path: Path):
    """Save a dense topic-by-sample matrix and return the sparse matrix and DataFrame."""
    topic_sample = lda.get_document_topics(corpus, minimum_probability=0.0)
    topic_sample_mat = gensim.matutils.corpus2csc(topic_sample, num_terms=n_topics)

    scipy.sparse.save_npz(output_path.with_suffix(".npz"), topic_sample_mat)

    topic_sample_df = pd.DataFrame(
        topic_sample_mat.todense(),
        index=[f"Topic {i}" for i in range(1, topic_sample_mat.shape[0] + 1)],
        columns=sample_names,
    )
    return topic_sample_mat, topic_sample_df


def postprocess_and_save_results(lda, train_sets: dict, test_sets: dict, FILES: dict, parameters: dict) -> None:
    """Save METRIC model outputs and evaluate test-set deconvolution.

    Outputs include topic-by-sample matrices, marker-by-topic weights,
    topic-by-cell-type summaries, normalized test-sample cell-type fractions
    and performance metrics for synthetic mixtures.
    """
    n_topics = parameters["n_topics"]
    n_top_words = parameters["n_top_words"]
    prefix = f"topic{n_topics}_topword{n_top_words}"
    output_dir = Path(FILES["outputDir"])

    # Save topic-by-training-sample matrix.
    train_sample_names = train_sets["sampleGroup"]["name"]
    topic_train_mat, topic_train_df = _save_topic_sample_matrix(
        lda,
        train_sets["corpus"],
        train_sample_names,
        n_topics,
        output_dir / f"{prefix}_topic_trainSample_mat",
    )
    topic_train_df.to_excel(output_dir / f"{prefix}_topic_trainSample_mat.xlsx", index=True, header=True)

    train_sample_topic_df = topic_train_df.T.reset_index().rename(columns={"index": "name"})
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

    # Save marker-by-topic matrix.
    topic_marker_mat = np.asarray(lda.get_topics())
    marker_topic_mat = topic_marker_mat.T
    markers = train_sets["DUI"].index.to_list()

    marker_topic_file = output_dir / f"{prefix}_marker_topic_mat.txt"
    with marker_topic_file.open("w", encoding="utf-8") as handle:
        handle.write("\t".join([f"Topic{i}" for i in range(1, n_topics + 1)]) + "\n")
        for marker, values in zip(markers, marker_topic_mat):
            handle.write(marker + "\t" + "\t".join(map(str, values)) + "\n")

    # Convert topic-by-training-sample weights to a topic-by-cell-type matrix.
    celltype_topic_sum = {}
    celltype_counts = {}
    train_groups = train_sets["sampleGroup"]["group"].reset_index(drop=True)
    celltypes = sorted(train_groups.unique())

    for celltype in celltypes:
        celltype_topic_sum[celltype] = np.zeros(n_topics, dtype=float)
        celltype_counts[celltype] = 0

    for sample_idx in range(topic_train_mat.shape[1]):
        celltype = train_groups.iloc[sample_idx]
        celltype_topic_sum[celltype] += topic_train_mat[:, sample_idx].toarray().ravel()
        celltype_counts[celltype] += 1

    celltype_topic_mean = {
        celltype: celltype_topic_sum[celltype] / celltype_counts[celltype]
        for celltype in celltypes
    }

    topic_celltype_df = pd.DataFrame(celltype_topic_mean)
    topic_celltype_df.to_csv(output_dir / f"{prefix}_topic_celltype_mat.txt", sep="\t")
    topic_celltype_df.to_excel(output_dir / f"{prefix}_topic_celltype_mat.xlsx", index=True, header=True)

    # Apply the trained model to the test set.
    test_sample_names = test_sets["sampleGroup"]["name"]
    topic_test_mat, topic_test_df = _save_topic_sample_matrix(
        lda,
        test_sets["corpus"],
        test_sample_names,
        n_topics,
        output_dir / f"{prefix}_topic_testSample_mat",
    )
    topic_test_df.to_csv(output_dir / f"{prefix}_topic_testSample_mat.txt", sep="\t", index=True, header=True)

    test_sample_topic_df = topic_test_df.T.reset_index().rename(columns={"index": "name"})
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
    test_celltype_array = np.dot(
        topic_test_mat.toarray().T,
        celltype_topic_df.iloc[:, : topic_test_mat.shape[0]].T,
    )

    test_celltype_df = pd.DataFrame(test_celltype_array, columns=celltype_topic_df.index, index=test_sample_names)
    test_celltype_actual = pd.merge(
        test_celltype_df.reset_index().rename(columns={"index": "name"}),
        test_sets["sampleGroup"],
        on="name",
        how="right",
    )
    test_celltype_actual.to_excel(
        output_dir / f"{prefix}_testSample_celltype_unNorm_actual.xlsx",
        header=True,
        index=True,
    )

    row_sums = test_celltype_array.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    test_celltype_norm = test_celltype_array / row_sums

    test_celltype_norm_df = pd.DataFrame(
        test_celltype_norm,
        columns=celltype_topic_df.index,
        index=test_sample_names,
    )
    test_celltype_norm_df.to_csv(output_dir / f"{prefix}_testSample_celltype_frac.txt", sep="\t")

    test_celltype_norm_actual = pd.merge(
        test_celltype_norm_df.reset_index().rename(columns={"index": "name"}),
        test_sets["sampleGroup"],
        on="name",
        how="right",
    )
    test_celltype_norm_actual.to_excel(
        output_dir / f"{prefix}_testSample_celltype_frac_actual.xlsx",
        header=True,
        index=True,
    )

    evaluate_model(test_celltype_norm_actual, FILES, parameters)


def train_single_model(parameters: dict, mainPath: str, selectID: str, topn: str = "250_U") -> None:
    """Train one METRIC model for a given marker number and topic number."""
    n_topics = parameters["n_topics"]
    n_top_words = parameters["n_top_words"]
    prefix = f"topic{n_topics}_topword{n_top_words}"

    print(f"\n=== Running: n_top_words={n_top_words}, n_topics={n_topics} ===")
    print(f"Start time: {datetime.datetime.now()}")

    try:
        files = setInputOutputFile_v2(
            mainPath=mainPath,
            selected_id=selectID,
            topn=topn,
            current_time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            parameters=parameters,
        )

        results_file = Path(files["outputDir"]) / f"{prefix}_evaluate_results.csv"
        topic_celltype_file = Path(files["outputDir"]) / f"{prefix}_topic_celltype_mat.txt"

        if topic_celltype_file.exists():
            try:
                topic_celltype_df = pd.read_csv(topic_celltype_file, sep="\t", index_col=0)
                actual_topic_num = topic_celltype_df.shape[0]
                print(f"[CHECK] Found topic-cell-type file: {topic_celltype_file}")
                print(f"[CHECK] Expected topics: {n_topics}, actual rows: {actual_topic_num}")

                if actual_topic_num == n_topics:
                    print("[SKIP] Existing output has the expected topic number. Skipping this run.")
                    return

                print("[RERUN] Existing output has an unexpected topic number. Re-running training.")
            except Exception as exc:
                print(f"[WARNING] Failed to inspect {topic_celltype_file}; re-running training.")
                print(exc)

        elif results_file.exists():
            print(f"[WARNING] {results_file} exists but {topic_celltype_file} is missing.")
            print("[RERUN] Output appears incomplete. Re-running training.")

        train_sets = generate_train_corpus(
            file_path=files["trainSetFile"],
            marker_info_file=files["markerInfoFile"],
            sample_group_file=files["trainSampleGroupFile"],
            n_top_words=n_top_words,
        )
        print(train_sets["markerGroup"]["group"].value_counts())

        eta = generate_ETA_v2(
            train_sets["markerGroup"],
            train_sets["celltypes"],
            train_sets["markers"],
            parameters["eta_norm"],
            n_topics,
        )
        pd.DataFrame(eta).to_excel(Path(files["outputDir"]) / f"{prefix}_ETA.xlsx")

        lda = models.LdaModel(
            corpus=train_sets["corpus"],
            id2word=train_sets["dictionary"],
            num_topics=n_topics,
            random_state=parameters["random_state"],
            offset=parameters["offset"],
            iterations=parameters["iterations"],
            passes=parameters["passes"],
            alpha=parameters["alpha"],
            eval_every=parameters["eval_every"],
            eta=eta,
        )
        lda.save(files["lda_model_save_path"])
        print(f"[OK] Model saved to: {files['lda_model_save_path']}")

        feature_names = train_sets["markerGroup"]["marker"].tolist()
        test_sets = generate_test_corpus(
            file_path=files["testSetFile"],
            marker_info_file=files["markerInfoFile"],
            sample_group_file=files["testSetSampleGroupFile"],
            feature_markers=feature_names,
            direction="U",
        )

        postprocess_and_save_results(lda, train_sets, test_sets, files, parameters)

    except Exception as exc:
        print(f"[ERROR] Training failed: {exc}")
        traceback.print_exc()

        placeholder = Path(mainPath) / f"topic_{n_topics}_failed_placeholder.txt"
        with placeholder.open("w", encoding="utf-8") as handle:
            handle.write("0\n")

    finally:
        print(f"End time: {datetime.datetime.now()}")
        print("======================================")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train one METRIC guided LDA model.")
    parser.add_argument("n_top_words", type=int, help="Number of top-ranked markers retained per group.")
    parser.add_argument("n_topics", type=int, help="Number of LDA topics.")
    parser.add_argument(
        "--main-path",
        default=os.getcwd(),
        help="Project directory containing the input/ folder. Default: current working directory.",
    )
    parser.add_argument(
        "--select-id",
        default="groupN2",
        help="Input prefix used to locate sample group and count matrix files.",
    )
    parser.add_argument(
        "--topn",
        default="250_U",
        help="Marker-set suffix used in input filenames, for example 250_U.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()

    parameters = {
        "n_top_words": args.n_top_words,
        "n_topics": args.n_topics,
        "iterations": 200,
        "offset": 1,
        "passes": 20,
        "alpha": "auto",
        "eval_every": 10,
        "random_state": 0,
        "eta_norm": True,
    }

    train_single_model(
        parameters=parameters,
        mainPath=args.main_path,
        selectID=args.select_id,
        topn=args.topn,
    )


if __name__ == "__main__":
    main()
