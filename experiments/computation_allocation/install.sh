#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"

install_system_packages() {
    if [[ "$(uname -s)" != "Linux" ]]; then
        echo "Automatic system-package installation is supported on Linux only."
        echo "Install Python 3, Clang, and Rust 1.88+ before rerunning this script."
        return
    fi

    if ! command -v apt-get >/dev/null 2>&1; then
        echo "apt-get was not found; assuming the compiler and Python dependencies are already installed."
        return
    fi

    local privilege=()
    if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
        if ! command -v sudo >/dev/null 2>&1; then
            echo "sudo is required to install system packages." >&2
            exit 1
        fi
        privilege=(sudo)
    fi

    "${privilege[@]}" apt-get update
    "${privilege[@]}" apt-get install -y \
        build-essential ca-certificates clang curl libclang-dev pkg-config \
        python3 python3-pip python3-venv
}

install_rust() {
    local rust_ok=0
    if command -v rustc >/dev/null 2>&1; then
        if python3 - "$(rustc --version | awk '{print $2}')" <<'PY'
import sys

current = tuple(int(part) for part in sys.argv[1].split(".")[:3])
raise SystemExit(0 if current >= (1, 88, 0) else 1)
PY
        then
            rust_ok=1
        fi
    fi
    if [[ "$rust_ok" == "1" ]] && command -v cargo >/dev/null 2>&1; then
        return
    fi

    echo "Installing a current Rust toolchain (Reckless requires Rust 1.88+)."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        RUSTUP_INIT_SKIP_PATH_CHECK=yes sh -s -- -y --profile minimal --default-toolchain stable
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
}

install_system_packages
install_rust

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
    "$VENV_DIR/bin/python" -m pip install torch --index-url "$TORCH_INDEX_URL"
fi
"$VENV_DIR/bin/python" -m pip install -r "$SCRIPT_DIR/requirements.txt"

if [[ "$REQUIRE_CUDA" == "1" ]]; then
    "$VENV_DIR/bin/python" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    sys.exit(
        "PyTorch cannot see an NVIDIA GPU. Confirm the NVIDIA driver is installed, "
        "or rerun with REQUIRE_CUDA=0 for a CPU-only setup."
    )
print(f"CUDA ready: {torch.cuda.get_device_name(0)} (PyTorch {torch.__version__})")
PY
fi

cd "$REPO_ROOT"
cargo test --release
cargo build --release
"$VENV_DIR/bin/python" -m unittest discover -s experiments/computation_allocation/tests -v

echo
echo "Installation and compilation completed."
echo "Start large-scale training with:"
echo "  ./experiments/computation_allocation/train_nvidia.sh"
