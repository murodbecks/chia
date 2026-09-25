#!/usr/bin/env bash
# Run on the dedicated NVIDIA machine (or a CPU-only reference host). No sudo,
# driver changes or global installs. CUDA is preferred for the FP32 oracle and
# required for the BF16 reference study; CPU is accepted and recorded.
set -euo pipefail
reference_root="${1:-$HOME/esm2-autonomous}"
mkdir -p "$reference_root"
python3 -m venv "$reference_root/.venv"
reference_python="$reference_root/.venv/bin/python"
"$reference_python" -m pip install --only-binary=:all: --index-url https://pypi.org/simple pip==26.2.1 setuptools==84.0.0
# Keep the CUDA wheel; on a CPU-only reference host use the cpu extra index instead.
"$reference_python" -m pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.14.0 --report "$reference_root/torch-install.json"
"$reference_python" -m pip install --index-url https://pypi.org/simple transformers==5.17.0 numpy==2.4.4 --report "$reference_root/reference-install.json"
"$reference_python" -m pip check
"$reference_python" -m pip freeze > "$reference_root/requirements.lock.txt"
"$reference_python" - <<'PY'
import torch
import transformers  # noqa: F401  (import guard)
assert hasattr(transformers, 'ParakeetForTDT') or hasattr(transformers, 'ParakeetTDTForCTC'), 'transformers lacks Parakeet TDT support'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    x = torch.randn(32, 32, device='cuda')
    assert torch.isfinite(x @ x).all()
    print({'torch': torch.__version__, 'cuda': torch.version.cuda,
           'gpu': torch.cuda.get_device_name(0), 'bf16': torch.cuda.is_bf16_supported()})
else:
    x = torch.randn(32, 32)
    assert torch.isfinite(x @ x).all()
    print({'torch': torch.__version__, 'device': 'cpu (recorded; CUDA preferred)'})
PY
