#!/bin/bash

set -euo pipefail

if [[ "$1" != "openblas" && "$1" != "blis" ]]; then
    echo "usage: install_chimera_backend.sh <openblas|blis>"
    exit 1
fi

BACKEND="$1"
ROOT_DIR="$(pwd)"
JOBS="${MAX_JOBS:-8}"

unset PYTHONHOME
unset PYTHONPATH
python3 -m venv venv

export VIRTUAL_ENV="${ROOT_DIR}/venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$VIRTUAL_ENV/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="/opt/OpenBLAS/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="/opt/OpenBLAS/lib/pkgconfig:/usr/local/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

pip3 install --upgrade pip setuptools wheel
pip3 install numpy pyyaml typing_extensions ninja pillow tqdm matplotlib

git clone --depth 1 --branch v2.5.1 https://github.com/pytorch/pytorch
cd "${ROOT_DIR}/pytorch"
git submodule update --init --recursive

if [[ "$BACKEND" == "openblas" ]]; then
    cd "${ROOT_DIR}"
    git clone https://github.com/OpenMathLib/OpenBLAS
    cd "${ROOT_DIR}/OpenBLAS"
    git checkout 700ea74a378cb5bf9073b4447a089a029131fb8b
    make FC=gfortran TARGET=ZEN -j "${JOBS}"
    make install

    cd "${ROOT_DIR}/pytorch"
    BUILD_TEST=0 USE_CUDA=0 USE_CUDNN=0 BLAS=OpenBLAS python3 setup.py develop
else
    cd "${ROOT_DIR}"
    git clone https://github.com/flame/blis.git
    cd "${ROOT_DIR}/blis"
    git checkout 1.0
    ./configure auto
    make -j "${JOBS}"
    make install

    cd "${ROOT_DIR}/pytorch"
    BUILD_TEST=0 USE_CUDA=0 USE_CUDNN=0 BLAS=BLIS python3 setup.py develop
fi

python3 - <<'PY'
import os
import torch
config = torch.__config__.show()
torch_file = os.path.realpath(torch.__file__)
venv = os.path.realpath(os.environ["VIRTUAL_ENV"])
source_tree = os.path.realpath(os.path.join(os.environ["VIRTUAL_ENV"], "..", "pytorch"))
print(config)
if not (torch_file.startswith(venv) or torch_file.startswith(source_tree)):
    raise SystemExit(
        f"torch imported from {torch_file}, expected under {venv} "
        f"or editable source tree {source_tree}"
    )
PY
