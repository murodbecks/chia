# PortForge-TT: working environment

## Mac driver

```bash
conda activate chia_env
# Python 3.10.19; CHIA 1.0.1 editable here; Ray 2.54.0.
# Codex CLI is already authenticated. Do not copy its credentials to workers.
```

## Blackhole development

```bash
ssh -o ClearAllForwardings=yes tt_box
source ~/portforge-tt/tt-metal/python_env/bin/activate
export TT_METAL_HOME="$HOME/portforge-tt/tt-metal"
export TT_VISIBLE_DEVICES=0000:01:00.0
cd "$TT_METAL_HOME"
```

Use this source environment, not system Python or conda base. It contains
CHIA 1.0.1, Ray 2.54.0, TTNN 0.72.0, Torch 2.14.0+cpu, uv 0.12.13,
pytest 9.0.3, and tt-smi 5.2.0. TT-Metal is pinned to
`ba9340e3a45ac5ba51c752a49341f2def28d0514` (v0.72.0).
The separate `~/portforge-tt/.venv` is the older wheel baseline.
The project checkout on Linux is `~/portforge-tt/chia`.

Host: Ubuntu 22.04.5, four p150b cards, TT-KMD 2.10.0, firmware 19.4.2.0,
16 one-GiB hugepages. Keep shared driver/firmware/services unchanged.

## Card selection

The user authorized all four cards, including parallel experiments.
Set TT_VISIBLE_DEVICES **before launching Python** to one full PCI BDF per
worker. Empty hides every card; unset exposes all cards. Do not use CUDA
visibility variables or assume /dev indices equal TTNN logical IDs.

| Device node | PCI BDF |
|---|---|
| /dev/tenstorrent/1 | 0000:01:00.0 |
| /dev/tenstorrent/2 | 0000:41:00.0 |
| /dev/tenstorrent/3 | 0000:42:00.0 |
| /dev/tenstorrent/0 | 0000:c1:00.0 |

Allow one evaluator per card at a time. Visibility filters and per-user locks
are not security boundaries or reservations against other lab processes.
The smoke harness provides bounded execution and a per-card advisory lock.

## Checks and references

NVIDIA reference experiments use the student-lab Slurm cluster, independently
of this TT environment. See [REFERENCE.md](../portforge-tt-reference/REFERENCE.md)
for persistent environment setup, disconnect-safe batch submission, and the
interactive tmux recipe. Do not run experiments on its login node.

From the Mac, run [the separate smoke example](../portforge-tt-smoke/SMOKE.md):

```bash
python examples/portforge-tt-smoke/smoke_loop.py --backend blackhole --pci 0000:01:00.0
```

[HISTORY.md](HISTORY.md) contains installation details, package/security review,
source build commands, profiling tools, and old failure diagnostics. Consult
it when reproducing or repairing the environment; no reinstall is needed for
normal work. Remote build logs and package freezes live in ~/portforge-tt/.

The [NLLB loop](../portforge-tt-nllb/NLLB.md) uses this same Mac controller and
native TT environment. It stores immutable remote runs and a hash-verified
checkpoint in `~/portforge-nllb/`. The Slurm worker reuses
`~/portforge-reference/.venv` and its model cache. No additional packages were
needed for the TT port. CHIA's Python distribution is named `chialoops` when
checking package versions with `importlib.metadata`; `chia` is its import name.
