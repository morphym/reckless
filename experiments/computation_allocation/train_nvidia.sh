#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
REQUESTED_PYTHON="${TRAIN_PYTHON:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cuda_is_runnable() {
    local candidate="$1"
    local result
    [[ -x "$candidate" ]] || return 1
    result="$("$candidate" - <<'PY' 2>/dev/null
try:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    (torch.ones(1, device="cuda") + 1).sum().item()
    torch.cuda.synchronize()
    print("cuda-ok")
except Exception:
    raise SystemExit(1)
PY
)" || return 1
    [[ "$result" == "cuda-ok" ]]
}

PYTHON=""
if [[ -n "$REQUESTED_PYTHON" ]]; then
    candidate="$(command -v "$REQUESTED_PYTHON" 2>/dev/null || true)"
    if cuda_is_runnable "$candidate"; then
        PYTHON="$candidate"
    fi
elif cuda_is_runnable "$VENV_DIR/bin/python"; then
    PYTHON="$VENV_DIR/bin/python"
else
    candidate="$(command -v python 2>/dev/null || command -v python3 2>/dev/null || true)"
    if cuda_is_runnable "$candidate"; then
        PYTHON="$candidate"
    fi
fi

if [[ -z "$PYTHON" || ! -x "$PYTHON" ]]; then
    echo "No CUDA-capable training Python was found. Run $SCRIPT_DIR/install.sh first." >&2
    exit 1
fi
if [[ ! -x "$REPO_ROOT/target/release/reckless" ]]; then
    echo "Reckless release binary not found. Run $SCRIPT_DIR/install.sh first." >&2
    exit 1
fi

mkdir -p "$REPO_ROOT/outputs/cs_online"

exec "$PYTHON" "$SCRIPT_DIR/online_train.py" \
    --engine "$REPO_ROOT/target/release/reckless" \
    --updates "${UPDATES:-10000}" \
    --episodes-per-update "${EPISODES_PER_UPDATE:-32}" \
    --workers "${WORKERS:-32}" \
    --reference-depth "${REFERENCE_DEPTH:-12}" \
    --reference-timeout "${REFERENCE_TIMEOUT:-120}" \
    --reference-attempts "${REFERENCE_ATTEMPTS:-3}" \
    --cs-depths "${CS_DEPTHS:-1,2,3,4,5,6}" \
    --minimum-budget "${MINIMUM_BUDGET:-8}" \
    --maximum-budget "${MAXIMUM_BUDGET:-64}" \
    --root-minimum-plies "${ROOT_MINIMUM_PLIES:-8}" \
    --root-maximum-plies "${ROOT_MAXIMUM_PLIES:-100}" \
    --root-temperature-start "${ROOT_TEMPERATURE_START:-500}" \
    --root-temperature-end "${ROOT_TEMPERATURE_END:-30}" \
    --controller-temperature-start "${CONTROLLER_TEMPERATURE_START:-3.0}" \
    --controller-temperature-end "${CONTROLLER_TEMPERATURE_END:-0.7}" \
    --device cuda \
    --checkpoint "$REPO_ROOT/outputs/cs_online/controller.pt" \
    --summary "$REPO_ROOT/outputs/cs_online/training.json" \
    --tensorboard-dir "${TENSORBOARD_DIR:-$REPO_ROOT/outputs/cs_online/tensorboard}" \
    "$@"
