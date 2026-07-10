#!/usr/bin/env bash
set -eo pipefail

[[ -f /usr/local/bin/bash_telemetry.sh ]] && bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

if [[ -f /usr/local/bin/start_cuda_compat.sh ]] && command -v nvidia-smi >/dev/null 2>&1; then
  source /usr/local/bin/start_cuda_compat.sh || true
fi

if [[ -z "${MODEL_DIR:-}" && -d /opt/ml/model && -n "$(ls -A /opt/ml/model 2>/dev/null)" ]]; then
  export MODEL_DIR=/opt/ml/model
fi

if [[ -z "${MODEL_ID:-}" && -n "${HF_MODEL_ID:-}" ]]; then
  export MODEL_ID="${HF_MODEL_ID}"
fi

if [[ -z "${TASK:-}" && -n "${HF_TASK:-}" ]]; then
  export TASK="${HF_TASK}"
fi

if [[ "${1:-}" == "serve" ]]; then
  shift
fi

ARGS=(--host "${SM_HF_SERVE_HOST:-0.0.0.0}" --port "${SM_HF_SERVE_PORT:-8080}")
PREFIX="SM_HF_SERVE_"
ARG_PREFIX="--"

while IFS='=' read -r key value; do
  case "${key}" in
    SM_HF_SERVE_HOST | SM_HF_SERVE_PORT) continue ;;
  esac

  arg_name=$(echo "${key#"${PREFIX}"}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
  lower_value=$(echo "${value}" | tr '[:upper:]' '[:lower:]')
  if [[ "${lower_value}" == "true" ]]; then
    ARGS+=("${ARG_PREFIX}${arg_name}")
  elif [[ "${lower_value}" == "false" ]]; then
    continue
  else
    ARGS+=("${ARG_PREFIX}${arg_name}")
    [[ -n "${value}" ]] && ARGS+=("${value}")
  fi
done < <(env | grep "^${PREFIX}" || true)

exec python3 /usr/local/bin/sagemaker_serve.py "${ARGS[@]}" "$@"
