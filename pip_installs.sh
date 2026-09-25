#!/usr/bin/env bash
# Install all project dependencies into the currently active venv.
# Usage: source venv/bin/activate && ./install.sh
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "No active virtualenv detected. Run: source venv/bin/activate" >&2
    exit 1
fi

pip install --upgrade pip

pip install \
    torch \
    torchvision \
    timm \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    Pillow \
    grad-cam \
    tqdm \
    tabulate

echo "All dependencies installed into ${VIRTUAL_ENV}"
