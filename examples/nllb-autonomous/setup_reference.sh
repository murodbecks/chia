#!/usr/bin/env bash
# Run on the dedicated NVIDIA machine. No sudo, driver changes or global installs.
set -euo pipefail
reference_root="${1:-$HOME/nllb-autonomous}"
mkdir -p "$reference_root"
python3 -m venv "$reference_root/.venv"
reference_python="$reference_root/.venv/bin/python"
"$reference_python" -m pip install --only-binary=:all: --index-url https://pypi.org/simple pip==26.2.1 setuptools==84.0.0
"$reference_python" -m pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.14.0 --report "$reference_root/torch-install.json"
"$reference_python" -m pip install --index-url https://pypi.org/simple transformers==5.17.0 numpy==2.4.4 sentencepiece==0.2.1 sacrebleu==2.6.0 --report "$reference_root/reference-install.json"
"$reference_python" -m pip check
"$reference_python" -m pip freeze > "$reference_root/requirements.lock.txt"
"$reference_python" - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA required'
print({'torch': torch.__version__, 'cuda': torch.version.cuda,
       'gpu': torch.cuda.get_device_name(0), 'bf16': torch.cuda.is_bf16_supported()})
x = torch.randn(32, 32, device='cuda')
assert torch.isfinite(x @ x).all()
PY
