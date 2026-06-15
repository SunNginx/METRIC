# METRIC

METRIC is a biologically guided topic-modeling framework for interpretable DNA methylome deconvolution, jointly estimating cell-type composition and marker evidence from sparse RRBS and WGBS methylation profiles.

## Overview

METRIC (**Methylation Topic modeling for Robust and Interpretable Cell-type deconvolution**) maps DNA methylation deconvolution to a guided latent Dirichlet allocation (LDA) framework:

- each methylation sample is treated as a document;
- each tissue or cell type is treated as a topic;
- each differentially methylated block (DMB) is treated as a word;
- atlas-derived methylation priors guide topics toward annotated biological identities.

METRIC supports platform-specific models for:

| Module | Sequencing platform | Default dataset ID |
|---|---|---|
| `METRIC-RRBS/` | Reduced-representation bisulfite sequencing | `groupN2` |
| `METRIC-WGBS/` | Whole-genome bisulfite sequencing | `groupV15` |

## Installation

Install Python dependencies:

```bash
pip install -r requirements.txt
```

The core Python dependencies are:

```text
gensim
numpy
pandas
scipy
scikit-learn
openpyxl
```

Raw methylation data preprocessing requires [`wgbstools`](https://github.com/nloyfer/wgbs_tools), which is not installed by `requirements.txt`.

## Quick start

### Train the RRBS model

```bash
cd METRIC-RRBS/Code
python train_model.py 50 37 --main-path .. --select-id groupN2 --topn 250_U
```

### Train the WGBS model

```bash
cd METRIC-WGBS/Code
python train_model.py 30 38 --main-path .. --select-id groupV15 --topn 250_U
```

### Run a Slurm grid search

```bash
cd METRIC-RRBS/Code
bash run_all.sh
```

or:

```bash
cd METRIC-WGBS/Code
bash run_all.sh
```

Custom parameter grids can be supplied through environment variables:

```bash
TOP_WORDS="30 50 75" TOPICS="37 38 39" bash run_all.sh
```

## Input files

Each platform-specific `input/` directory should contain:

| File | Description |
|---|---|
| `<select_id>.csv` | Training sample metadata with `name` and `group` columns |
| `<select_id>_testSetSampleGroup.csv` | Test sample metadata with `name`, `group` and, for synthetic mixtures, `actual` columns |
| `<select_id>_top250_U_TrainSet_countU.bed` | Marker-by-sample U-count matrix for training samples |
| `<select_id>_top250_U_TestSet_countU.bed` | Marker-by-sample U-count matrix for test samples |
| `Merged_Markers_*_deldup.bed` | Cell-type-specific DMB marker annotation file |

The sample names in the metadata file must exactly match the sample columns in the corresponding count matrix.

## Output files

Training results are written to:

```text
<platform>/results/<train_prefix>_top<n_top_words>/topic_<n_topics>/
```

Key files include:

| File | Description |
|---|---|
| `ModelTrained/*_lda.model` | Trained Gensim LDA model |
| `topic*_topword*_ETA.xlsx` | Guided topic prior matrix |
| `topic*_topword*_marker_topic_mat.txt` | Marker-to-topic weight matrix |
| `topic*_topword*_topic_celltype_mat.txt` | Topic-to-cell-type mapping matrix |
| `topic*_topword*_testSample_celltype_frac.txt` | Predicted cell-type fractions for test samples |
| `topic*_topword*_evaluate_results.csv` | Benchmark metrics for synthetic mixtures |

## Deconvolving new samples with a trained model

To apply a trained METRIC model to new samples, see:

[METRIC_trained_model_deconvolution_guide.md](METRIC_trained_model_deconvolution_guide.md)

In brief, new samples should be processed with `wgbstools bam2pat` and `wgbstools homog` to generate a marker-by-sample `test_countU.bed` matrix, followed by deconvolution using:

```text
METRIC_trained_model_deconvolution_name_only_fixed.ipynb
```

For independent samples, the sample metadata file only needs a `name` column.

## Reproducibility notes

- RRBS and WGBS models should be trained separately because the two platforms recover different CpG and marker regions.
- The default optimized examples are:
  - METRIC-RRBS: `n_top_words = 50`, `n_topics = 37`
  - METRIC-WGBS: `n_top_words = 30`, `n_topics = 38`

## Citation

If you use METRIC, please cite the corresponding METRIC manuscript.

## License

This project is distributed under the license provided in `LICENSE`.
