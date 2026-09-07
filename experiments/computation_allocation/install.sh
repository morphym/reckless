#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
ENABLE_CS_SEARCH="${ENABLE_CS_SEARCH:-0}"
REQUESTED_PYTHON="${TRAIN_PYTHON:-}"
if [[ -n "$REQUESTED_PYTHON" ]]; then
    ACTIVE_PYTHON="$(command -v "$REQUESTED_PYTHON" 2>/dev/null || true)"
else
    ACTIVE_PYTHON="$(command -v python 2>/dev/null || command -v python3 2>/dev/null || true)"
fi
VENV_PYTHON="$VENV_DIR/bin/python"
PYTHON=""

if [[ "$REQUIRE_CUDA" != "0" && "$REQUIRE_CUDA" != "1" ]]; then
    echo "REQUIRE_CUDA must be 0 or 1." >&2
    exit 1
fi
if [[ "$ENABLE_CS_SEARCH" != "0" && "$ENABLE_CS_SEARCH" != "1" ]]; then
    echo "ENABLE_CS_SEARCH must be 0 or 1." >&2
    exit 1
fi

for required_file in \
    online_train.py \
    reckless_uci.py \
    live_env.py \
    controller_model.py \
    controller_state.py \
    ppo.py; do
    if [[ ! -f "$SCRIPT_DIR/$required_file" ]]; then
        echo "Required experiment file is missing: $SCRIPT_DIR/$required_file" >&2
        echo "Pull the latest experiment/cs-online-training branch and rerun." >&2
        exit 1
    fi
done

torch_is_runnable() {
    local candidate="$1"
    local result
    [[ -x "$candidate" ]] || return 1
    result="$("$candidate" - "$REQUIRE_CUDA" <<'PY' 2>/dev/null
import sys

try:
    import torch

    require_cuda = sys.argv[1] == "1"
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = "cuda" if require_cuda else "cpu"
    result = (torch.ones(4, device=device) * 2).sum().item()
    if result != 8:
        raise RuntimeError("tensor operation returned an unexpected result")
    if require_cuda:
        torch.cuda.synchronize()
    print("torch-ok")
except Exception:
    raise SystemExit(1)
PY
)" || return 1
    [[ "$result" == "torch-ok" ]]
}

python_has_tensorboard() {
    "$1" - <<'PY' >/dev/null 2>&1
from torch.utils.tensorboard import SummaryWriter  # noqa: F401
PY
}

# Check existing Python environments before performing any package-manager work.
if torch_is_runnable "$VENV_PYTHON"; then
    PYTHON="$VENV_PYTHON"
    echo "Existing virtual-environment PyTorch is runnable; reusing it."
elif [[ -n "$ACTIVE_PYTHON" ]] && torch_is_runnable "$ACTIVE_PYTHON"; then
    PYTHON="$ACTIVE_PYTHON"
    echo "Active Python already has runnable PyTorch; reusing $PYTHON."
fi

install_missing_system_packages() {
    if [[ "$(uname -s)" != "Linux" ]] || ! command -v apt-get >/dev/null 2>&1; then
        return
    fi

    local packages=()
    command -v cc >/dev/null 2>&1 || packages+=(build-essential)
    command -v clang >/dev/null 2>&1 || packages+=(clang libclang-dev)
    command -v curl >/dev/null 2>&1 || packages+=(curl ca-certificates)
    if [[ -z "$PYTHON" ]]; then
        command -v python3 >/dev/null 2>&1 || packages+=(python3 python3-venv)
    fi
    if [[ -z "$PYTHON" ]] && command -v python3 >/dev/null 2>&1; then
        python3 -m venv --help >/dev/null 2>&1 || packages+=(python3-venv)
    fi

    if ((${#packages[@]} == 0)); then
        echo "System build tools are already present; skipping apt-get."
        return
    fi

    local privilege=()
    if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
        if ! command -v sudo >/dev/null 2>&1; then
            echo "sudo is required to install missing system packages: ${packages[*]}" >&2
            exit 1
        fi
        privilege=(sudo)
    fi

    echo "Installing missing system packages: ${packages[*]}"
    "${privilege[@]}" apt-get update
    "${privilege[@]}" apt-get install -y "${packages[@]}"
}

install_rust() {
    local rust_ok=0
    local version_python="${PYTHON:-$(command -v python3 2>/dev/null || true)}"
    if command -v rustc >/dev/null 2>&1 && [[ -n "$version_python" ]]; then
        if "$version_python" - "$(rustc --version | awk '{print $2}')" <<'PY'
import sys

current = tuple(int(part) for part in sys.argv[1].split(".")[:3])
raise SystemExit(0 if current >= (1, 88, 0) else 1)
PY
        then
            rust_ok=1
        fi
    fi
    if [[ "$rust_ok" == "1" ]] && command -v cargo >/dev/null 2>&1; then
        echo "Rust 1.88+ is already present; skipping Rust installation."
        return
    fi

    if ! command -v curl >/dev/null 2>&1; then
        echo "curl is required to install Rust." >&2
        exit 1
    fi
    echo "Installing a current Rust toolchain (Reckless requires Rust 1.88+)."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        RUSTUP_INIT_SKIP_PATH_CHECK=yes sh -s -- -y --profile minimal --default-toolchain stable
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
}

install_missing_system_packages
install_rust

if [[ -z "$PYTHON" ]]; then
    echo "No runnable PyTorch environment was found; creating $VENV_DIR."
    python3 -m venv "$VENV_DIR"
    PYTHON="$VENV_PYTHON"
    "$PYTHON" -m pip install --upgrade pip setuptools wheel
    if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
        "$PYTHON" -m pip install 'torch>=2.5' --index-url "$TORCH_INDEX_URL"
    else
        "$PYTHON" -m pip install 'torch>=2.5'
    fi
else
    echo "Skipping PyTorch installation."
fi

if ! torch_is_runnable "$PYTHON"; then
    echo "PyTorch cannot execute on the requested device." >&2
    echo "Set TORCH_INDEX_URL to a compatible CUDA wheel channel and rerun." >&2
    exit 1
fi

if python_has_tensorboard "$PYTHON"; then
    echo "TensorBoard is already present; skipping TensorBoard installation."
else
    echo "Installing TensorBoard into the selected Python environment."
    "$PYTHON" -m pip install 'tensorboard>=2.14'
fi

"$PYTHON" - "$REQUIRE_CUDA" <<'PY'
import sys
import torch

if sys.argv[1] == "1":
    print(f"CUDA ready: {torch.cuda.get_device_name(0)} (PyTorch {torch.__version__})")
else:
    print(f"CPU PyTorch ready: {torch.__version__}")
PY

cd "$REPO_ROOT"
ENGINE_FEATURES=()
if [[ "$ENABLE_CS_SEARCH" == "1" ]]; then
    ENGINE_FEATURES=(--features cs-search)
fi
cargo test --release "${ENGINE_FEATURES[@]}"
cargo build --release "${ENGINE_FEATURES[@]}"
cargo build --release --manifest-path experiments/computation_allocation/burn_inference/Cargo.toml
"$PYTHON" -m unittest discover -s experiments/computation_allocation/tests -v

echo
echo "Installation and compilation completed with: $PYTHON"
echo "Start large-scale training with:"
if [[ "$PYTHON" == "$VENV_PYTHON" ]]; then
    echo "  ./experiments/computation_allocation/train_nvidia.sh"
else
    echo "  TRAIN_PYTHON=$PYTHON ./experiments/computation_allocation/train_nvidia.sh"
fi
