# Deconvolving new DNA methylation samples with trained METRIC models

This guide describes how to apply an existing trained METRIC model to independent bisulfite-sequencing samples.

The workflow is:

1. Convert BAM files to `.pat.gz`.
2. Extract marker-level U-read counts with `wgbstools homog`.
3. Merge per-sample `.uxm.bed.gz` files into one `test_countU.bed` matrix.
4. Prepare `test_sample_group.csv`.
5. Run `METRIC_trained_model_deconvolution.ipynb`.

## 1. Expected layout

The notebook should be placed in the same directory as `train_model.py` and `gensim_Utils.py`.

```text
METRIC-RRBS/
├── Code/
│   ├── train_model.py
│   ├── gensim_Utils.py
│   └── METRIC_trained_model_deconvolution.ipynb
├── input/
│   ├── Merged_Markers_top250_U_deldup.bed
│   ├── test_countU.bed
│   └── test_sample_group.csv
└── results/
    └── groupN2_top250_U_TrainSet_countU_top50/
        └── topic_37/
            ├── ModelTrained/
            │   └── groupN2_top250_U_TrainSet_countU_lda.model
            ├── topic37_topword50_marker_topic_mat.txt
            └── topic37_topword50_topic_celltype_mat.txt
```

For WGBS, use the corresponding `METRIC-WGBS/` directory, model folder and marker file.

The notebook uses paths relative to its own location:

```text
../input/
../results/
../results_trainedModel_deconv/
```

## 2. Convert BAM files to `.pat.gz`

Use `wgbstools bam2pat` to convert aligned BAM files into `.pat.gz` files.

Example:

```bash
wgbstools bam2pat -o pat_files/ sample.bam
```

Use the same genome build and methylation processing convention as the trained METRIC model.

## 3. Prepare the marker BED file

Use the marker file corresponding to the trained model.

Example:

```bash
MARKER_BED=../input/Merged_Markers_top250_U_deldup.bed
sort -k1,1 -k2,2n "${MARKER_BED}" > ../input/markerBlock_sorted.bed
```

## 4. Extract marker-level U-read counts with `wgbstools homog`

Run `wgbstools homog` for each `.pat.gz` file.

Example command:

```bash
wgbstools homog -b /path/to/markerBlock_sorted.bed -o /path/to/output/homog_uxm_file/  /path/to/testset/file/*.pat.gz
```

After processing all samples, the output directory should contain one file per sample:

```text
sample1.uxm.bed.gz
sample2.uxm.bed.gz
sample3.uxm.bed.gz
```

## 5. Merge `.uxm.bed.gz` files into `test_countU.bed`

Create `merge_uxm_to_countU.py`:

```python
from functools import reduce
from pathlib import Path
import pandas as pd

UXM_DIR = Path("/path/samples.uxm.bed.gz")
OUTPUT_FILE = Path("/path/test_countU.bed")

file_list = sorted(UXM_DIR.glob("*.uxm.bed.gz"))
if not file_list:
    raise FileNotFoundError(f"No .uxm.bed.gz files found in {UXM_DIR}")

dfs = []

for i, file in enumerate(file_list, start=1):
    sample_name = file.name.replace(".uxm.bed.gz", "")
    print(f"[{i}/{len(file_list)}] Reading {file.name}")

    df = pd.read_csv(file, sep="\t", header=None, usecols=range(6))
    df.rename(columns={5: sample_name}, inplace=True)
    dfs.append(df)

merged = reduce(
    lambda left, right: pd.merge(left, right, on=[0, 1, 2, 3, 4], how="outer"),
    dfs,
)

merged.rename(
    columns={
        0: "chr",
        1: "start",
        2: "end",
        3: "startCpG",
        4: "endCpG",
    },
    inplace=True,
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUTPUT_FILE, sep="\t", index=False)

print(f"[OK] Saved merged count-U matrix to: {OUTPUT_FILE}")
```

Run:

```bash
python merge_uxm_to_countU.py
```

The output should be:

```text
/path/test_countU.bed
```

## 6. Prepare `test_sample_group.csv`

For independent samples, only the `name` column is required.

```csv
name
sample1
sample2
sample3
```

The `name` values must exactly match the sample column names in `test_countU.bed`.

If known labels are available, an optional `group` column can be included:

```csv
name,group
sample1,Unknown
sample2,Unknown
sample3,Unknown
```

## 7. Configure the deconvolution notebook

Open:

```text
METRIC_trained_model_deconvolution.ipynb
```

Edit the configuration cell.

Example for METRIC-RRBS:

```python
SELECT_ID = "groupN2"
TOPN = "250_U"

N_TOP_WORDS_LIST = [50]
N_TOPICS_LIST = [37]

INPUT_DIR = PROJECT_ROOT / "input"
MODEL_RESULTS_DIR = PROJECT_ROOT / "results"

TEST_SET_FILE = INPUT_DIR / "test_countU.bed"
TEST_SAMPLE_GROUP_FILE = INPUT_DIR / "test_sample_group.csv"

OUTPUT_DIR = PROJECT_ROOT / "results_trainedModel_deconv" / "external_testset"
SKIP_MISSING_MODELS = True
```

Example for METRIC-WGBS:

```python
SELECT_ID = "groupV15"
TOPN = "250_U"

N_TOP_WORDS_LIST = [30]
N_TOPICS_LIST = [38]
```

## 8. Run deconvolution

Run all cells in the notebook.

The final combined result is saved to:

```text
../results_trainedModel_deconv/external_testset/final_trained_model_deconvolution_results.tsv
```

Per-model outputs are saved under:

```text
../results_trainedModel_deconv/external_testset/
└── <TRAIN_PREFIX>_top<n_top_words>/
    └── topic_<n_topics>/
```

Key outputs:

```text
trained_model_topic_testSample_mat.tsv
trained_model_testSample_celltype_fraction.tsv
trained_model_testSample_celltype_fraction_with_metadata.tsv
```

## 9. Minimal checklist

Before running the notebook, confirm that these files exist:

```text
../input/test_countU.bed
../input/test_sample_group.csv
../results/<TRAIN_PREFIX>_top<n_top_words>/topic_<n_topics>/ModelTrained/*.model
../results/<TRAIN_PREFIX>_top<n_top_words>/topic_<n_topics>/topic<n_topics>_topword<n_top_words>_topic_celltype_mat.txt
```

Also confirm that the marker file used by `wgbstools homog` is the same marker panel used by the trained model.
