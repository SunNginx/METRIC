#!/usr/bin/env bash
#SBATCH -c 1

# Run one METRIC training job.
#
# Positional arguments:
#   1: n_top_words
#   2: n_topics
#   3: project directory containing the input/ folder (optional)
#   4: input prefix, for example groupN2 or groupV15 (optional)

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: sbatch submit_single.sh <n_top_words> <n_topics> [workdir] [select_id]"
  exit 1
fi

N_TOP_WORDS="$1"
N_TOPICS="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${3:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SELECT_ID="${4:-groupN2}"

echo "Running METRIC with n_top_words=${N_TOP_WORDS}, n_topics=${N_TOPICS}, select_id=${SELECT_ID}"
echo "Workdir: ${WORKDIR}"
echo "Host: $(hostname)"
date

cd "${SCRIPT_DIR}"

python3 ${WORKDIR}/Code/train_model.py "${N_TOP_WORDS}" "${N_TOPICS}" \
  --main-path "${WORKDIR}" \
  --select-id "${SELECT_ID}" \
  --topn "250_U"

date
echo "Job done."
