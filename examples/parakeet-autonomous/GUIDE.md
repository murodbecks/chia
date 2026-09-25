# Parakeet TDT autonomous port — operator guide

Status: harness prepared, hosts staged, **NOT launched**. One hard blocker on
the TT host (below). This harness carries forward everything hardened during
the NLLB and Chronos-2 campaigns (opencode/z.ai or Claude backends via
`PARAKEET_AGENT_BACKEND`, auto-approval by configuration, ~8-min checkpoint as
strong target with a 30-min hard cap, LESSONS.md binding rules).

## Prepared

- Pinned registry `models.json`: `nvidia/parakeet-tdt-0.6b-v3` @ `541d1f99…`
  (CC-BY-4.0 weights, safetensors, full file hashes; `.nemo`/`.gguf` variants
  excluded by design).
- Frozen audio benchmark `runs/data/corpus.json` (canonical sha256
  `62fb9c51…`): 15 LibriSpeech dev-clean clips (one per speaker, 2.2–16.4 s)
  + 4 deterministic synthetic edge clips (silence/noise/tone/0.4 s), 124 s
  audio, base64-embedded. Gates: encoder NRMSE ≤0.04 vs FP32 oracle, exact
  greedy-token match, revisit/batch consistency, full-stage WER inflation ≤2%
  vs the FP32 reference transcripts. A100 reference timings = the NVIDIA
  comparison context (matched TT-vs-TT claims stay separate).
- Trusted evaluator `evaluate.py` (transformers `ParakeetForTDT` oracle,
  `AutoProcessor` mel frontend — deterministic CPU policy), supervisor/worker
  loop (`loop.py`, `agent.py`), isolated runner, `LESSONS.md` (17 rules incl.
  autoregressive-decode overhead trap and shared-machine discipline).
- Tests: 157 CPU tests pass across 7 modules (macOS: 2 env-dependent skips).

## ⚠ TT host blocker (must clear before launch)

~~tt-blackhole-01 blocker~~ **RESOLVED by switching to tt-blackhole-04 (10.127.30.205)**: driver loaded, 4 device nodes, device compute verified, same fleet env/runtimes, internet OK, idle. tt-blackhole-03/05 also healthy (4 nodes each). The -01 issue remains for admins (message in this file).
**Needs admin action** (load driver / create device nodes; possibly
`/opt/tenstorrent/hugepages-setup.sh`). Everything else on the host is ready:
the fleet env `upstream-fd80faa3-tokenizer` and runtime
`tt-metal-fd80faa3` (same pins as tt-blackhole-00) import and run, and the
2.5 GB checkpoint downloads into
`/home/abror/parakeet-autonomous/checkpoints/hf-cache` (verified by
`models.py --download`). After the driver is up, confirm with:
`ls /dev/tenstorrent` and one runner health check via `cluster.load()` +
`prepare()` (the controller's preflight does this automatically).

## Reference host (done/verify)

`setup_reference.sh` builds `~/parakeet-autonomous/.venv` on the A100 host
(torch 2.14 cu126, transformers 5.17 with `ParakeetForTDT`, soundfile). Manual
reference smoke before launch:
`cd ~/parakeet-autonomous && PARAKEET_REFERENCE_EXECUTION=direct HF_HOME=$HOME/parakeet-autonomous/hf-cache .venv/bin/python evaluate.py reference --stage smoke --model parakeet --corpus corpus.json --out /tmp/pk-ref-smoke`
(upload evaluate.py, models.py, models.json, corpus.json first). This
validates the `ParakeetForTDT`/`AutoProcessor`/`generate` API assumptions and
the mel/lengths plumbing once, before the campaign starts.

## Readiness (all clear, 2026-09-24)

- TT host tt-blackhole-04 (10.127.30.205): driver + 4 device nodes + device
  compute verified; runtime commit fd80faa3 matches; bwrap + all mounts
  present; weights (2.4 GB) all-files hash-verified under
  `/home/abror/parakeet-autonomous/checkpoints/hf-cache`.
- Reference host: venv ready (torch 2.14 cu126, transformers 5.17
  `AutoModelForTDT`, soundfile, librosa). **Reference smoke PASSED**
  (`/tmp/pk-ref-smoke2/` on the A100 host): FP32, generate 0.162 s for the
  smoke clip, transcript correct. The evaluator was fixed to the real
  transformers API: processor gives `input_features [B,T,128]` (time-major) +
  `attention_mask [B,T]`; encoder states via `model.model.encoder(...)`;
  `generate(input_features=…, attention_mask=…, return_dict_in_generate=True)`
  → `.sequences`.
- tt-blackhole-01 (.197) remains broken for admins (kernel module not built
  for 6.8.0-138); weights are also staged there as a spare.

## Launch

From this directory:
`PYTHONPATH=../..:. python loop.py --corpus runs/data/corpus.json --hours 6 --detach`
(agent model: `PARAKEET_AGENT_MODEL`, default `zai-coding-plan/glm-5.3`;
backend switch: `PARAKEET_AGENT_BACKEND=claude` + `PARAKEET_WORKER_MODEL=…`
as used during the Chronos-2 campaign).

## Shared-machine rules (operator)

Never run anything outside `/home/abror/parakeet-autonomous`; never reset
cards; the controller serializes card leases but the host is shared — check
`w`/load before heavy runs and keep `runtime_mounts` as-is.
