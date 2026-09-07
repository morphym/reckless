#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Training environment not found. Run $SCRIPT_DIR/install.sh first." >&2
    exit 1
fi
if [[ ! -x "$REPO_ROOT/target/release/reckless" ]]; then
    echo "Reckless release binary not found. Run $SCRIPT_DIR/install.sh first." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$REPO_ROOT/outputs/cs_online"

exec "$VENV_DIR/bin/python" "$SCRIPT_DIR/online_train.py" \
    --engine "$REPO_ROOT/target/release/reckless" \
    --updates "${UPDATES:-10000}" \
    --episodes-per-update "${EPISODES_PER_UPDATE:-32}" \
    --workers "${WORKERS:-32}" \
    --reference-depth "${REFERENCE_DEPTH:-12}" \
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
    "$@"
