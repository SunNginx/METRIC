#!/bin/bash
set -euo pipefail

# Submit a grid of METRIC-WGBS training jobs to Slurm.
#
# Usage:
#   bash run_all.sh
#
# Optional environment variables:
#   WORKDIR       Project root containing input/ and results/ folders.
#   SELECT_ID     Dataset identifier. Default: groupV15
#   TOPN          Marker-panel suffix in input filenames. Default: 250_U
#   TOP_WORDS     Space-separated top-marker values.
#   TOPICS        Space-separated topic values.
#   SUBMIT_SCRIPT Path to submit_single.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${WORKDIR:-}" ]]; then
  if [[ -d "${SCRIPT_DIR}/../input" ]]; then
    WORKDIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
  else
    WORKDIR="${SCRIPT_DIR}"
  fi
fi

SELECT_ID="${SELECT_ID:-groupV15}"
TOPN="${TOPN:-250_U}"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-${SCRIPT_DIR}/submit_single.sh}"

# TOP_WORDS="${TOP_WORDS:-25 30 40 50 60 75 100 110 115 120 125 150 175 200 225 250}"
# TOPICS="${TOPICS:-34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53}"

TOP_WORDS="${TOP_WORDS:-30}"
TOPICS="${TOPICS:-38}"

mkdir -p "${WORKDIR}/logs"

for ntw in ${TOP_WORDS}; do
  for ntp in ${TOPICS}; do
    echo "Submitting METRIC-WGBS: n_top_words=${ntw}, n_topics=${ntp}"
    sbatch \
      --job-name="METRIC_WGBS_${ntw}_${ntp}" \
      --output="${WORKDIR}/logs/METRIC_WGBS_topWord${ntw}_topic${ntp}_jobid%j.out" \
      --error="${WORKDIR}/logs/METRIC_WGBS_topWord${ntw}_topic${ntp}_jobid%j.err" \
      --chdir="${SCRIPT_DIR}" \
      "${SUBMIT_SCRIPT}" "${ntw}" "${ntp}" "${WORKDIR}" "${SELECT_ID}" "${TOPN}"
    sleep 0.2
  done
done
