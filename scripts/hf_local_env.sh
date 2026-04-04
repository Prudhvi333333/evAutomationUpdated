#!/usr/bin/env zsh
# Usage:
#   source scripts/hf_local_env.sh online
#   source scripts/hf_local_env.sh offline
#
# online  -> allows local model download (still private, no telemetry)
# offline -> forces fully offline local execution after model files exist

if [[ -z "${ZSH_VERSION:-}" ]]; then
  echo "Please run with zsh: source scripts/hf_local_env.sh <online|offline>"
  return 1 2>/dev/null || exit 1
fi

mode="${1:-online}"
if [[ "${mode}" != "online" && "${mode}" != "offline" ]]; then
  echo "Invalid mode: ${mode}"
  echo "Usage: source scripts/hf_local_env.sh <online|offline>"
  return 1 2>/dev/null || exit 1
fi

project_root="$(cd "$(dirname "${(%):-%N}")/.." && pwd)"

# Local-only cache root inside the project directory.
export HF_HOME="${project_root}/.hf_home"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
unset TRANSFORMERS_CACHE

# Privacy controls.
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export WANDB_DISABLED=true

# Prevent accidental uploads.
unset HF_TOKEN
unset HUGGING_FACE_HUB_TOKEN

if [[ "${mode}" == "offline" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
else
  unset HF_HUB_OFFLINE
  unset TRANSFORMERS_OFFLINE
fi

mkdir -p "${HF_HOME}" \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}"

echo "HF local environment set (${mode})"
echo "HF_HOME=${HF_HOME}"
echo "HF_HUB_DISABLE_TELEMETRY=${HF_HUB_DISABLE_TELEMETRY}"
echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}"
echo "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}"
