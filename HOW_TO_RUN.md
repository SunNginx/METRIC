# How to run METRIC

This guide describes the minimal commands needed to train METRIC-RRBS and METRIC-WGBS models and to locate the main outputs.

## 1. Install dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

`wgbstools` is required only for preparing methylation marker count matrices from raw sequencing data.

## 2. Repository layout

```text
METRIC/
├── METRIC-RRBS/
│   ├── Code/
│   └── input/
├── METRIC-WGBS/
│   ├── Code/
│   └── input/
├── README.md
├── HOW_TO_RUN.md
├── METRIC_trained_model_deconvolution_guide.md
└── requirements.txt
```

Each platform-specific module contains its own code and input files.

## 3. Required input files

For each platform, the `input/` directory should contain:

```text
<select_id>.csv
<select_id>_testSetSampleGroup.csv
<select_id>_top250_U_TrainSet_countU.bed
<select_id>_top250_U_TestSet_countU.bed
Merged_Markers_<topn>_deldup.bed
```

Default dataset IDs:

| Platform | Directory | `select_id` |
|---|---|---|
| RRBS | `METRIC-RRBS/` | `groupN2` |
| WGBS | `METRIC-WGBS/` | `groupV15` |

The marker annotation filename differs slightly between platforms:

```text
METRIC-RRBS/input/Merged_Markers_top250_U_deldup.bed
METRIC-WGBS/input/Merged_Markers_Top250_U_deldup.bed
```

The optimized code supports this filename difference.

## 4. Train one model

### METRIC-RRBS

```bash
cd METRIC-RRBS/Code
python train_model.py 50 37 --main-path .. --select-id groupN2 --topn 250_U
```

### METRIC-WGBS

```bash
cd METRIC-WGBS/Code
python train_model.py 30 38 --main-path .. --select-id groupV15 --topn 250_U
```

Arguments:

| Argument | Description |
|---|---|
| `n_top_words` | Number of top hypomethylated DMBs selected per tissue or cell type |
| `n_topics` | Number of LDA topics |
| `--main-path` | Platform-specific project directory containing `input/` and `results/` |
| `--select-id` | Dataset ID used to locate input files |
| `--topn` | Marker panel suffix; default is `250_U` |
| `--force` | Re-run even if complete outputs already exist |

## 5. Run Slurm grid search

### METRIC-RRBS

```bash
cd METRIC-RRBS/Code
bash run_all.sh
```

### METRIC-WGBS

```bash
cd METRIC-WGBS/Code
bash run_all.sh
```

To customize parameter grids:

```bash
TOP_WORDS="25 30 40 50" TOPICS="34 35 36 37 38" bash run_all.sh
```

To specify a project directory:

```bash
WORKDIR=/path/to/METRIC-WGBS SELECT_ID=groupV15 TOP_WORDS="30" TOPICS="38" bash run_all.sh
```

Logs are written to:

```text
<WORKDIR>/logs/
```

## 6. Output directory

Training results are saved to:

```text
<WORKDIR>/results/<train_prefix>_top<n_top_words>/topic_<n_topics>/
```

Example:

```text
METRIC-RRBS/results/groupN2_top250_U_TrainSet_countU_top50/topic_37/
METRIC-WGBS/results/groupV15_top250_U_TrainSet_countU_top30/topic_38/
```

## 7. Main output files

| File | Description |
|---|---|
| `ModelTrained/*_lda.model` | Trained LDA model |
| `topic*_topword*_ETA.xlsx` | Guided topic prior matrix |
| `topic*_topword*_marker_topic_mat.txt` | Marker-topic distribution |
| `topic*_topword*_topic_celltype_mat.txt` | Topic-cell-type distribution |
| `topic*_topword*_topic_trainSample_mat.xlsx` | Topic distribution for training samples |
| `topic*_topword*_topic_testSample_mat.txt` | Topic distribution for test samples |
| `topic*_topword*_testSample_celltype_frac.txt` | Normalized cell-type fraction estimates |
| `topic*_topword*_evaluate_results.csv` | Performance metrics for benchmark mixtures |

## 8. Apply a trained model to new samples

Use the trained-model deconvolution notebook:

```text
METRIC_trained_model_deconvolution_name_only_fixed.ipynb
```

A separate guide is provided:

```text
METRIC_trained_model_deconvolution_guide.md
```

The independent test sample metadata file only requires one column:

```csv
name
sample1
sample2
sample3
```

## 9. Common checks

Before running METRIC, check:

```bash
ls ../input/
```

Confirm that:

1. all required input files exist;
2. sample names in metadata match the count matrix column names;
3. `n_topics` is not smaller than the number of annotated tissue or cell types;
4. the marker annotation file corresponds to the same platform and marker panel used to prepare the count matrix.
