#!/bin/bash
set -euo pipefail

# In SageMaker mode we run:
# - vLLM backend on localhost:8091
# - FastAPI proxy on 0.0.0.0:8080
VLLM_BACKEND_HOST="127.0.0.1"
VLLM_BACKEND_PORT="8091"
SAGEMAKER_PORT="8080"

# Check if telemetry file exists before executing
# Execute telemetry script if it exists, suppress errors
bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

# Source CUDA compat for older drivers (e.g., g5 instances)
if command -v nvidia-smi >/dev/null 2>&1 && command -v nvcc >/dev/null 2>&1; then
    source /usr/local/bin/start_cuda_compat.sh
fi

PREFIX="SM_VLLM_"
MODEL=""
ARGS=()

to_flag() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr '_' '-'
}

is_true() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

is_false() {
    case "${1,,}" in
        ""|0|false|no|off) return 0 ;;
        *) return 1 ;;
    esac
}

while IFS='=' read -r key value; do
    name="${key#"$PREFIX"}"

    if [[ "$name" == "MODEL" ]]; then
        MODEL="$value"
        continue
    fi

    if [[ "$name" == "PORT" ]]; then
        echo "Ignoring SM_VLLM_PORT; SageMaker requires port ${SAGEMAKER_PORT}" >&2
        continue
    fi

    flag="--$(to_flag "$name")"

    if is_true "$value"; then
        ARGS+=("$flag")
    elif is_false "$value"; then
        :
    else
        ARGS+=("$flag" "$value")
    fi
done < <(env | grep "^${PREFIX}" || true)

if [[ -z "$MODEL" ]]; then
    echo "Error: SM_VLLM_MODEL is required" >&2
    exit 1
fi

# Start vLLM backend on localhost and expose SageMaker-compatible proxy on 8080.
echo "Starting vLLM-omni backend on ${VLLM_BACKEND_HOST}:${VLLM_BACKEND_PORT} with model '$MODEL' and arguments: ${ARGS[*]}" >&2
vllm serve "$MODEL" --host "${VLLM_BACKEND_HOST}" --port "${VLLM_BACKEND_PORT}" "${ARGS[@]}" &
VLLM_PID=$!

cleanup() {
    echo "Received shutdown signal, stopping child processes..." >&2
    kill "${VLLM_PID}" "${PROXY_PID:-}" 2>/dev/null || true
    wait "${VLLM_PID}" "${PROXY_PID:-}" 2>/dev/null || true
}

trap cleanup SIGTERM SIGINT

echo "Starting SageMaker proxy on 0.0.0.0:${SAGEMAKER_PORT}" >&2
uvicorn sagemaker_vllm_omni_proxy:app --host 0.0.0.0 --port "${SAGEMAKER_PORT}" --app-dir /usr/local/bin &
PROXY_PID=$!

# Exit if either process exits unexpectedly.
set +e
wait -n "${VLLM_PID}" "${PROXY_PID}"
EXIT_CODE=$?
set -e

cleanup
exit "${EXIT_CODE}"
