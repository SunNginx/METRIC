#!/usr/bin/env python3
"""Utility functions for training METRIC-WGBS models.

The functions in this module convert marker-level count-U matrices into
Gensim-compatible corpora, construct the biologically guided ETA prior, and
prepare standardized input/output paths for model training.

Expected project layout
-----------------------
<project_root>/
    input/
        groupV15.csv
        groupV15_testSetSampleGroup.csv
        groupV15_top250_U_TrainSet_countU.bed
        groupV15_top250_U_TestSet_countU.bed
        Merged_Markers_Top250_U_deldup.bed
    results/
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import gensim
import numpy as np
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from scipy.sparse import csr_matrix


MARKER_COLUMNS = [
    "chr",
    "start",
    "end",
    "startCpG",
    "endCpG",
    "group",
    "name",
    "lenCpG",
    "bp",
    "tg_mean",
    "bg_mean",
    "delta_means",
    "delta_quants",
    "delta_maxmin",
    "ttest",
    "direction",
]


def get_file_stem(filepath: str | os.PathLike) -> str:
    """Return the filename without extension."""
    return Path(filepath).stem


def _read_marker_info(marker_info_file: str | os.PathLike) -> pd.DataFrame:
    """Read the marker annotation file and standardize column names.

    Parameters
    ----------
    marker_info_file
        BED-like marker annotation table without a header.

    Returns
    -------
    pandas.DataFrame
        Marker annotation table with standardized column names.
    """
    marker_info = pd.read_csv(marker_info_file, sep="\t", header=None)
    if marker_info.shape[1] < len(MARKER_COLUMNS):
        raise ValueError(
            f"Marker file must contain at least {len(MARKER_COLUMNS)} columns, "
            f"but {marker_info.shape[1]} columns were found: {marker_info_file}"
        )

    marker_info = marker_info.iloc[:, : len(MARKER_COLUMNS)].copy()
    marker_info.columns = MARKER_COLUMNS
    marker_info["group"] = marker_info["group"].astype(str).str.capitalize()
    marker_info = marker_info.sort_values(by="group", ascending=True)
    return marker_info


def _read_sample_group(
    sample_group_file: str | os.PathLike,
    require_group: bool = True,
) -> pd.DataFrame:
    """Read sample metadata and validate required columns.

    Parameters
    ----------
    sample_group_file
        CSV file containing at least a ``name`` column. For benchmark
        evaluation, a ``group`` column is also required.
    require_group
        Whether to require the ``group`` column.

    Returns
    -------
    pandas.DataFrame
        Sample metadata sorted by group when a group column is present.
    """
    sample_group = pd.read_csv(sample_group_file, header=0)

    if "name" not in sample_group.columns:
        raise ValueError(f"Sample group file must contain a 'name' column: {sample_group_file}")

    sample_group["name"] = sample_group["name"].astype(str)

    if "group" not in sample_group.columns:
        if require_group:
            raise ValueError(f"Sample group file must contain a 'group' column: {sample_group_file}")
        sample_group["group"] = "Unknown"

    sample_group["group"] = sample_group["group"].astype(str).str.capitalize()
    sample_group = sample_group.sort_values(by="group", ascending=True).reset_index(drop=True)
    return sample_group


def _add_marker_name(data: pd.DataFrame) -> pd.DataFrame:
    """Create a genomic marker name using ``chr:start-end``."""
    required_columns = {"chr", "start", "end"}
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(f"Input count matrix is missing required columns: {sorted(missing)}")

    data = data.copy()
    data["name"] = (
        data["chr"].astype(str)
        + ":"
        + data["start"].astype(str)
        + "-"
        + data["end"].astype(str)
    )
    return data


def _validate_and_reorder_samples(DUI: pd.DataFrame, sample_group: pd.DataFrame) -> pd.DataFrame:
    """Validate sample names and reorder the matrix columns by sample metadata."""
    expected = set(sample_group["name"])
    observed = set(DUI.columns)

    if expected != observed:
        missing_in_matrix = sorted(expected - observed)
        missing_in_metadata = sorted(observed - expected)
        raise ValueError(
            "Sample names in sample metadata and the count-U matrix do not match. "
            f"Missing in matrix: {missing_in_matrix[:10]}; "
            f"missing in metadata: {missing_in_metadata[:10]}"
        )

    return DUI[sample_group["name"].tolist()]


def _select_top_n_per_group(
    df: pd.DataFrame,
    group_col: str,
    sort_col: str,
    top_n: int,
    exempt_groups: Optional[Sequence[str]] = None,
    exempt_top_n: Optional[int] = None,
) -> pd.DataFrame:
    """Select top-ranked markers within each cell type or marker group.

    Parameters
    ----------
    df
        Input marker table.
    group_col
        Column used for grouping markers.
    sort_col
        Column used for ranking markers within each group.
    top_n
        Number of markers to keep for each non-exempt group.
    exempt_groups
        Optional groups for which a different cutoff is used.
    exempt_top_n
        Number of markers to keep for exempt groups. If ``None``, all markers
        from exempt groups are retained.

    Returns
    -------
    pandas.DataFrame
        Filtered marker table.
    """
    exempt_groups = set(exempt_groups or [])

    def select_group(sub_df: pd.DataFrame) -> pd.DataFrame:
        group_value = sub_df[group_col].iloc[0]
        if group_value in exempt_groups:
            if exempt_top_n is None:
                return sub_df
            return sub_df.nlargest(exempt_top_n, sort_col)
        return sub_df.nlargest(top_n, sort_col)

    return df.groupby(group_col, group_keys=False).apply(select_group).reset_index(drop=True)


def _scale_train_matrix_global(
    matrix: pd.DataFrame,
    new_min: int = 1000,
    new_max: int = 1_000_000,
) -> pd.DataFrame:
    """Apply global min-max scaling to the training count-U matrix.

    Zero values are treated as missing observations before scaling and are
    restored to zero after scaling.
    """
    scaled = matrix.copy()
    scaled.replace(0, np.nan, inplace=True)

    min_val = scaled.min().min()
    max_val = scaled.max().max()
    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        raise ValueError("Cannot scale training matrix because all non-zero values are missing or constant.")

    scaled = ((scaled - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min
    return scaled.round(0).fillna(0).astype(int)


def _scale_test_matrix_by_sample(
    matrix: pd.DataFrame,
    new_min: int = 1000,
    new_max: int = 1_000_000,
) -> pd.DataFrame:
    """Apply per-sample min-max scaling to the test count-U matrix.

    This preserves the behavior of the original METRIC-WGBS script, where
    independent test samples were scaled column by column.
    """
    scaled = matrix.copy()
    scaled.replace(0, np.nan, inplace=True)

    def scale_column(col: pd.Series) -> pd.Series:
        min_val = col.min()
        max_val = col.max()
        if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
            return pd.Series(np.nan, index=col.index)
        return ((col - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min

    scaled = scaled.apply(scale_column, axis=0)
    return scaled.fillna(0).astype(int)


def _to_gensim_corpus(matrix: pd.DataFrame) -> tuple:
    """Convert a marker-by-sample matrix into a Gensim corpus and dictionary."""
    sparse_matrix = csr_matrix(matrix.values.astype(np.float64))
    corpus = gensim.matutils.Sparse2Corpus(sparse_matrix)
    dictionary = Dictionary([matrix.index.to_list()])
    return corpus, dictionary


def generate_train_corpus(
    file_path: str | os.PathLike,
    marker_info_file: str | os.PathLike,
    sample_group_file: str | os.PathLike,
    feature_markers: Optional[Sequence[str]] = None,
    n_top_words: int = 25,
    exempt_top_n: Optional[int] = None,
) -> Dict[str, object]:
    """Generate the training corpus for METRIC-WGBS.

    Parameters
    ----------
    file_path
        Training count-U matrix. The first columns must include ``chr``,
        ``start`` and ``end``; sample columns follow the marker metadata.
    marker_info_file
        Marker annotation file generated from differential methylation analysis.
    sample_group_file
        CSV file with sample names and cell-type labels.
    feature_markers
        Optional marker list used to force a predefined marker order.
    n_top_words
        Number of top hypomethylated DMBs selected per cell type.
    exempt_top_n
        Optional cutoff for exempt groups. The current WGBS workflow does not
        define exempt groups, so this argument has no effect unless exempt
        groups are added below.

    Returns
    -------
    dict
        Dictionary containing the Gensim corpus, dictionary, marker metadata,
        sample metadata and scaled matrices.
    """
    sample_group = _read_sample_group(sample_group_file, require_group=True)
    marker_info = _read_marker_info(marker_info_file)

    data = pd.read_csv(file_path, sep="\t", header=0)
    data = _add_marker_name(data)

    merged = pd.merge(
        marker_info[["name", "group", "delta_means", "delta_quants"]],
        data,
        how="left",
        on="name",
    )

    # WGBS model: no exempt marker groups are used by default.
    exempt_groups: List[str] = []
    merged = _select_top_n_per_group(
        merged,
        group_col="group",
        sort_col="delta_means",
        top_n=n_top_words,
        exempt_groups=exempt_groups,
        exempt_top_n=exempt_top_n,
    )

    DUI = merged.iloc[:, 9:].copy()
    DUI.index = merged["name"]
    DUI = _validate_and_reorder_samples(DUI, sample_group)

    if feature_markers is not None:
        DUI = DUI.loc[list(feature_markers)]

    marker_group = merged[["name", "group", "delta_means", "delta_quants"]].copy()
    marker_group.columns = ["marker", "group", "delta_means", "delta_quants"]
    marker_location = merged[["name", "chr", "start", "end"]].copy()

    scaled = _scale_train_matrix_global(DUI)
    celltypes = marker_info["group"].unique()
    markers = scaled.index.unique()
    corpus, dictionary = _to_gensim_corpus(scaled)

    return {
        "sampleGroup": sample_group,
        "corpus": corpus,
        "dictionary": dictionary,
        "markerGroup": marker_group,
        "celltypes": celltypes,
        "markers": markers,
        "markerLocation": marker_location,
        "DUI": DUI,
        "DUI_scaled": scaled,
    }


def generate_test_corpus(
    file_path: str | os.PathLike,
    marker_info_file: str | os.PathLike,
    sample_group_file: str | os.PathLike,
    feature_markers: Sequence[str],
    direction: str = "U",
    n_top_words: int = 25,
) -> Dict[str, object]:
    """Generate a test corpus using the marker order from the training set.

    Parameters
    ----------
    file_path
        Test count-U matrix.
    marker_info_file
        Marker annotation file.
    sample_group_file
        CSV sample metadata. It should contain ``name`` and ``group`` for
        benchmark evaluation.
    feature_markers
        Marker list from the trained model. The test matrix is reordered to
        match this list.
    direction
        Kept for backward compatibility. The current workflow uses U counts.
    n_top_words
        Kept for backward compatibility.

    Returns
    -------
    dict
        Test corpus and metadata.
    """
    sample_group = _read_sample_group(sample_group_file, require_group=True)
    marker_info = _read_marker_info(marker_info_file)

    data = pd.read_csv(file_path, sep="\t", header=0)
    data = _add_marker_name(data)

    merged = pd.merge(
        marker_info[["name", "group", "delta_means"]],
        data,
        how="left",
        on="name",
    )

    feature_markers = list(feature_markers)
    merged = merged[merged["name"].isin(feature_markers)]

    DUI = merged.iloc[:, 8:].copy()
    DUI.index = merged["name"]
    DUI = _validate_and_reorder_samples(DUI, sample_group)
    DUI = DUI.loc[feature_markers]

    marker_group = merged[["name", "group"]].copy()
    marker_group.columns = ["marker", "group"]
    marker_location = merged[["name", "chr", "start", "end"]].copy()

    scaled = _scale_test_matrix_by_sample(DUI)
    corpus, dictionary = _to_gensim_corpus(scaled)

    return {
        "sampleGroup": sample_group,
        "corpus": corpus,
        "dictionary": dictionary,
        "markerGroup": marker_group,
        "markerLocation": marker_location,
        "DUI": DUI,
        "DUI_scaled": scaled,
    }


def generate_ETA_v2(
    markerGroup: pd.DataFrame,
    celltypes: Sequence[str],
    markers: Sequence[str],
    eta_norm: bool,
    num_topics: int,
) -> np.ndarray:
    """Construct the guided ETA prior matrix for Gensim LDA.

    The first rows correspond to annotated cell types. If ``num_topics`` is
    larger than the number of annotated cell types, additional background rows
    with small prior values are appended.

    Returns
    -------
    numpy.ndarray
        ETA matrix with shape ``num_topics × num_markers``.
    """
    celltypes = list(celltypes)
    markers = list(markers)

    if num_topics < len(celltypes):
        raise ValueError(
            f"num_topics ({num_topics}) must be >= number of cell types ({len(celltypes)})."
        )

    eta = markerGroup.pivot(index="group", columns="marker", values="delta_quants").fillna(0.001)
    eta = eta[markers]
    eta = eta.loc[celltypes]

    n_extra_topics = num_topics - len(celltypes)
    if n_extra_topics > 0:
        extra_rows = pd.DataFrame(0.001, index=range(n_extra_topics), columns=eta.columns)
        eta = pd.concat([eta, extra_rows], ignore_index=True)

    if eta_norm:
        column_sums = eta.sum(axis=0).replace(0, np.nan)
        eta = eta.div(column_sums, axis=1).fillna(0.001)

    return eta.to_numpy() * 1000


def _resolve_existing_file(candidates: Iterable[Path]) -> Path:
    """Return the first existing path from a list of candidate files."""
    candidates = list(candidates)
    for path in candidates:
        if path.exists():
            return path
    candidate_text = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"None of the expected files exists:\n{candidate_text}")


def setInputOutputFile_v2(
    mainPath: str | os.PathLike,
    selected_id: str,
    topn: str | int = "250_U",
    current_time: Optional[str] = None,
    parameters: Optional[Dict[str, object]] = None,
) -> Dict[str, str]:
    """Create standardized input and output paths for one training run.

    Parameters
    ----------
    mainPath
        Project root containing ``input`` and ``results`` folders.
    selected_id
        Dataset identifier, for example ``groupV15``.
    topn
        Marker-panel suffix, for example ``250_U``.
    current_time
        Kept for compatibility with earlier scripts.
    parameters
        Parameter dictionary containing at least ``n_topics`` and
        ``n_top_words``.

    Returns
    -------
    dict
        Input, output and model paths used by the training script.
    """
    if parameters is None:
        raise ValueError("parameters must be provided.")

    n_topics = int(parameters["n_topics"])
    n_top_words = int(parameters["n_top_words"])
    topn_str = str(topn)

    work_dir = Path(mainPath).resolve()
    input_dir = work_dir / "input"

    train_sample_group_file = _resolve_existing_file([input_dir / f"{selected_id}.csv"])
    train_set_file = _resolve_existing_file([input_dir / f"{selected_id}_top{topn_str}_TrainSet_countU.bed"])
    test_set_file = _resolve_existing_file([input_dir / f"{selected_id}_top{topn_str}_TestSet_countU.bed"])
    marker_info_file = _resolve_existing_file(
        [
            input_dir / f"Merged_Markers_Top{topn_str}_deldup.bed",
            input_dir / f"Merged_Markers_top{topn_str}_deldup.bed",
        ]
    )
    test_sample_group_file = _resolve_existing_file([input_dir / f"{selected_id}_testSetSampleGroup.csv"])

    train_prefix = get_file_stem(train_set_file)
    test_prefix = get_file_stem(test_set_file)

    output_dir = work_dir / "results" / f"{train_prefix}_top{n_top_words}" / f"topic_{n_topics}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = output_dir / "ModelTrained"
    model_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "inputDir": str(input_dir),
        "markerInfoFile": str(marker_info_file),
        "trainPrefix": train_prefix,
        "trainSampleGroupFile": str(train_sample_group_file),
        "trainSetFile": str(train_set_file),
        "testPrefix": test_prefix,
        "testSetSampleGroupFile": str(test_sample_group_file),
        "testSetFile": str(test_set_file),
        "trainMatrixFile": str(output_dir / f"{train_prefix}_DUI_matrix.xlsx"),
        "trainMarkerList": str(output_dir / f"{train_prefix}_train_marker.xlsx"),
        "top_words_save_path": str(output_dir / f"{train_prefix}_predict_top_celltype_markers.xlsx"),
        "trainSet_predict_save_path": str(output_dir / f"{train_prefix}_deconv_celltype.xlsx"),
        "testMatrixFile": str(output_dir / f"{test_prefix}_DUI_matrix.xlsx"),
        "testSet_predict_save_path": str(output_dir / f"{test_prefix}_deconv_celltype.xlsx"),
        "deconvolution_record": str(output_dir / "Deconvolution_Record.txt"),
        "outputDir": str(output_dir),
        "lda_model_save_path": str(model_dir / f"{train_prefix}_lda.model"),
        "lda_model_save_dir": str(model_dir),
    }

    parameters_save_path = output_dir / f"{train_prefix}_parameters.json"
    with open(parameters_save_path, "w", encoding="utf-8") as handle:
        json.dump(parameters, handle, ensure_ascii=False, indent=4)

    files_save_path = output_dir / f"{train_prefix}_Files.json"
    with open(files_save_path, "w", encoding="utf-8") as handle:
        json.dump(files, handle, ensure_ascii=False, indent=4)

    return files
