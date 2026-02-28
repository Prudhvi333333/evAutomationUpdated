#!/usr/bin/env bash
set -euo pipefail

echo "========================================"
echo "EV Research Pipeline - Run All Phases"
echo "========================================"
echo

# Resolve project root from this script's location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python"
fi

run_phase() {
  local phase_label="$1"
  local script_path="$2"

  echo "[${phase_label}] Running ${script_path}..."
  if ! "${PYTHON}" "${script_path}"; then
    echo "[ERROR] ${phase_label} failed."
    exit 1
  fi
  echo "[OK] ${phase_label} complete"
  echo
}

run_phase "PHASE 1" "pipeline/phase_1_model_check/test_models.py"
run_phase "PHASE 2" "pipeline/phase_2_ingestion/test_ingestion.py"
run_phase "PHASE 3" "pipeline/phase_3_chunking/test_chunking.py"
run_phase "PHASE 4" "pipeline/phase_4_retrieval/test_retrieval.py"
run_phase "PHASE 5" "pipeline/phase_5_full_pipeline/run_pipeline.py"

echo "========================================"
echo "ALL PHASES COMPLETED SUCCESSFULLY!"
echo "========================================"
echo "Results saved to: test_results.xlsx"
