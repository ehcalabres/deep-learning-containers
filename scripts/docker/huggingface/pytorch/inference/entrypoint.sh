#!/usr/bin/env bash
set -eo pipefail

if [[ -f /usr/local/bin/start_cuda_compat.sh ]] && command -v nvidia-smi >/dev/null 2>&1; then
  source /usr/local/bin/start_cuda_compat.sh || true
fi

[[ -f /usr/local/bin/bash_telemetry.sh ]] && bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

# Backwards compatibility with the former Hugging Face inference toolkit envs.
if [[ -d "${HF_MODEL_ID:-}" ]]; then
  echo "WARNING: HF_MODEL_ID is a path, please use HF_MODEL_DIR for paths instead."
  export HF_MODEL_DIR="${HF_MODEL_ID}"
  unset HF_MODEL_ID
fi

if [[ -n "${HF_MODEL_DIR:-}" && -z "${MODEL_DIR:-}" ]]; then
  export MODEL_DIR="${HF_MODEL_DIR}"
fi

if [[ -n "${HF_MODEL_ID:-}" && -z "${MODEL_ID:-}" ]]; then
  export MODEL_ID="${HF_MODEL_ID}"
fi

if [[ -n "${HF_TASK:-}" && -z "${TASK:-}" ]]; then
  export TASK="${HF_TASK}"
fi

if [[ -n "${HF_REVISION:-}" && -z "${REVISION:-}" ]]; then
  export REVISION="${HF_REVISION}"
fi

if [[ -n "${HF_TRUST_REMOTE_CODE:-}" && -z "${TRUST_REMOTE_CODE:-}" ]]; then
  export TRUST_REMOTE_CODE="${HF_TRUST_REMOTE_CODE}"
fi

if [[ -n "${MODEL_DIR:-}" ]]; then
  if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "ERROR: Provided MODEL_DIR is not a valid directory" >&2
    exit 1
  fi

  if [[ -f "${MODEL_DIR}/requirements.txt" ]]; then
    echo "INFO: Installing custom dependencies from ${MODEL_DIR}/requirements.txt"
    uv pip install --python "${VIRTUAL_ENV}/bin/python" -r "${MODEL_DIR}/requirements.txt" --no-cache-dir
  fi
fi

if [[ "${1:-}" == "serve" ]]; then
  shift
fi

exec hf-serve "$@"
