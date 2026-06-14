#!/usr/bin/env bash
# Submit a grid of METRIC training jobs to Slurm.
#
# Usage examples:
#   bash run_all.sh
#   SELECT_ID=groupV15 TOPICS="34 35 36 37 38 39 40 41 42 43" bash run_all.sh
#
# Environment variables:
#   WORKDIR          Project directory containing the input/ folder.
#                   Default: parent directory of this script.
#   SELECT_ID        Input prefix, for example groupN2 or groupV15. Default: groupN2.
#   TOP_WORDS        Space-separated n_top_words values.
#   TOPICS           Space-separated n_topics values.
#   SLURM_PARTITION  Slurm partition name. Default: cpu112c.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SELECT_ID="${SELECT_ID:-groupN2}"
SLURM_PARTITION="${SLURM_PARTITION:-cpu112c}"

# TOP_WORDS="${TOP_WORDS:-25 30 40 50 60 75 100 110 115 120 125 150 175 200 225 250}"
# TOPICS="${TOPICS:-27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46}"
TOP_WORDS="${TOP_WORDS:-50}"
TOPICS="${TOPICS:-37}"

LOG_DIR="${WORKDIR}/logs"
mkdir -p "${LOG_DIR}"

for ntw in ${TOP_WORDS}; do
  for ntp in ${TOPICS}; do
    echo "Submitting: n_top_words=${ntw}, n_topics=${ntp}, select_id=${SELECT_ID}"

    sbatch \
      --partition="${SLURM_PARTITION}" \
      --job-name="METRIC_${ntw}_${ntp}" \
      --output="${LOG_DIR}/METRIC_topWord${ntw}_topic${ntp}_jobid%j.out" \
      --error="${LOG_DIR}/METRIC_topWord${ntw}_topic${ntp}_jobid%j.err" \
      "${SCRIPT_DIR}/submit_single.sh" "${ntw}" "${ntp}" "${WORKDIR}" "${SELECT_ID}"

    sleep 0.2
  done
done
