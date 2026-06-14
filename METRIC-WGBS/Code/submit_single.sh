#!/bin/bash
#SBATCH -p cpu112c
#SBATCH -c 5

set -euo pipefail

# Train one METRIC-WGBS model.
#
# Positional arguments:
#   1. n_top_words
#   2. n_topics
#   3. project root containing input/ and results/ folders
#   4. dataset identifier
#   5. marker-panel suffix

N_TOP_WORDS="${1:?Usage: submit_single.sh <n_top_words> <n_topics> [workdir] [select_id] [topn]}"
N_TOPICS="${2:?Usage: submit_single.sh <n_top_words> <n_topics> [workdir] [select_id] [topn]}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${3:-}" ]]; then
  WORKDIR="$3"
elif [[ -d "${SCRIPT_DIR}/../input" ]]; then
  WORKDIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  WORKDIR="${SCRIPT_DIR}"
fi

SELECT_ID="${4:-groupV15}"
TOPN="${5:-250_U}"

echo "Running METRIC-WGBS with n_top_words=${N_TOP_WORDS}, n_topics=${N_TOPICS}"
echo "Project root: ${WORKDIR}"
echo "Dataset ID: ${SELECT_ID}"
echo "Marker panel: ${TOPN}"
echo "Host: $(hostname)"
date

python3 "${WORKDIR}/Code/train_model.py" \
  "${N_TOP_WORDS}" \
  "${N_TOPICS}" \
  --main-path "${WORKDIR}" \
  --select-id "${SELECT_ID}" \
  --topn "${TOPN}"

date
echo "Job done."
