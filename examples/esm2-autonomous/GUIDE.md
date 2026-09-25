# ESM-2 650M autonomous port — operator guide

Status: harness prepared, hosts staged, **NOT launched**.

## Prepared (2026-09-24)

- Pinned registry `models.json`: `facebook/esm2_t33_650M_UR50D` @ `08e4846e…`
  (MIT, safetensors, full file hashes; `.bin`/`.h5` duplicates excluded).
- Frozen protein corpus `runs/data/corpus.json` (sha256 `369c475f…`): 14
  Swiss-Prot sequences (CC-BY-4.0, 68–1273 aa incl. one deterministic 1024
  truncation) + 5 synthetic edge sequences (1024/1023 boundary, 1- and
  2-residue, unknown residues). Deterministic masking (positions 5, 21, 37…).
  Gates: logits + final-hidden NRMSE ≤0.04 vs FP32 oracle, masked-position
  argmax exact, behavior NRMSE ≤0.01, determinism. Encoder-only single
  forward — the simplest of the three campaigns; trace/batching wins from
  NLLB/Chronos transfer directly.
- Tests: all CPU modules pass.
- TT host tt-blackhole-03 (10.127.30.203): driver + 4 nodes + **device
  compute verified**; runtime fd80faa3 matches; bwrap + mounts present;
  weights (2.6 GB) staged via `models.py --download` (verify receipt in
  `/home/abror/esm2-autonomous/download.log`). ⚠ Shared machine: another user
  (nandaabsar) keeps idle tmux sessions — never touch their files; card leases
  still serialize cleanly across the 4 cards.
- Reference host: `setup_reference.sh` venv (torch 2.14 cu126 +
  transformers 5.17 `EsmForMaskedLM`). Manual smoke before launch:
  `cd ~/esm2-autonomous && ESM2_REFERENCE_EXECUTION=direct HF_HOME=$HOME/esm2-autonomous/hf-cache .venv/bin/python evaluate.py reference --stage smoke --model esm2 --corpus corpus.json --out /tmp/esm2-ref-smoke`
  (upload evaluate.py, models.py, models.json, corpus.json first).

## Launch (when ready)

`PYTHONPATH=../..:. python loop.py --corpus runs/data/corpus.json --hours 6 --detach`
(agent: `ESM2_AGENT_MODEL`, default `zai-coding-plan/glm-5.3`; Claude swap via
`ESM2_AGENT_BACKEND=claude`).
