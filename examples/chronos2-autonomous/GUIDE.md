# Chronos-2 autonomous port — operator guide

Status: harness prepared, NOT launched. Everything below must be checked before
`python loop.py` is started. This mirrors the NLLB campaign recipe
(`examples/nllb-autonomous`) with the Chronos-2 benchmark, contract, packaging
goal and NLLB lessons wired in.

## What is already prepared

- Pinned checkpoint registry `models.json`: `amazon/chronos-2` @
  `29ec3766…` (Apache-2.0, safetensors, 119,477,664 FP32 params; file hashes
  verified from the official HF API).
- Frozen benchmark corpus `runs/data/corpus.json` (sha256 of canonical JSON:
  `f0b08df4fe80385da1d3e9a05e3779708c4e808b2da581aa974fea9505d64ae3`):
  last 1536 hourly points of the 7 public ETTh1 series + 3 deterministic
  synthetic series (seasonal mix, arcsinh-magnitude stress, constant).
  Suites are tiny during development (`smoke` = 1 forecast; `bringup` = 8
  behavior cases incl. masks, boundary contexts 64/65, constant series) and the
  `full` gate is a bounded sample (~35 windows: contexts 64/128/512 × horizons
  16/64) with held-out futures — minutes, not hours. This is a porting
  benchmark, not a GIFT-Eval claim.
- Trusted evaluator `evaluate.py`: FP32 oracle via `chronos-forecasting`
  (`Chronos2Pipeline.predict`, univariate `[B,1,T]`, direct 21-quantile
  output), gates: quantile max-row NRMSE ≤0.04 vs oracle, cross-shape
  behavior NRMSE ≤0.01, run-to-run determinism, and (full only) weighted
  quantile-loss inflation ≤2% vs the FP32 forecasts. CPU reference is allowed
  and recorded; CUDA preferred. Reference precision study: FP32/BF16 only
  (the package does not expose FP16).
- Supervisor/worker loop (`loop.py`, `agent.py`): same two-role CHIA graph,
  tools `chronos_files/read/write/run/evaluate/status/log/submit`; tool name
  prefix, repository root env (`CHRONOS_REPOSITORY_ROOT`) and prompts are
  Chronos-adapted. Worker goal ordering: measured baseline (PyTorch CPU on the
  TT host, optional ONNX/forge) → bringup correctness → optimization against
  the recorded baseline → tt-metal-style package (`tt/ demo/ reference/ tests/
  benchmarks/ docs/`, SPDX headers) → freeze → full validation.
- **Agent runtime = opencode + z.ai GLM** (no Codex subscription needed).
  `agent.py` drives the local `opencode` CLI through CHIA's `OpenCodeLLM`:
  model `zai-coding-plan/glm-5.3` (default; override with
  `CHRONOS_AGENT_MODEL`, e.g. `…-highspeed` for cheaper turns), auth from the
  already-configured Z.AI Coding Plan login (`opencode auth list`). Auto-approval
  is structural: the generated opencode config allows the scoped Chia MCP tools
  and web search while denying native edit/bash/subagents/ask-user, so an
  unattended run has nothing that could block on an approval prompt
  (`--dangerously-skip-permissions` is NOT used — opencode 1.18.x rejects it).
  Worker turn cap: 30 min hard (`CHRONOS_WORKER_TIMEOUT_SECONDS`), with the
  ~8-minute STATE.md checkpoint as a strong target, not a hard rule; supervisor
  decisions capped at 5 min. Usage per call (input/output/reasoning/cache
  tokens, cost) is recorded by the CHIA profiler like the NLLB campaign.
- `LESSONS_NLLB.md`: 15 binding rules distilled from the NLLB campaign
  (baseline-first, tiny tests during bring-up, no card resets, no repeated
  failed experiments, log paging instead of reruns, early PR materials,
  tt-metal packaging, historical≠current evidence, etc.). TASK.md makes it
  required reading for the agent.
- Isolated TT runner (`runner.py`): bwrap sandbox, card leases, quarantine +
  preflight recovery, host-memory watchdog, phase deadlines — unchanged from
  the hardened NLLB version.

## Checklist before launch (operator)

0. **Agent CLI (this machine)** — `opencode --version` (≥1.18), `opencode auth list`
   shows the Z.AI Coding Plan credential, and `opencode models | grep glm-5.3`
   lists `zai-coding-plan/glm-5.3`. Optional one-time smoke (a few tokens):
   `opencode run --model zai-coding-plan/glm-5.3 "reply OK"`. Benchmark
   rationale and corpus provenance are documented in BENCHMARK.md.
1. **TT host (`<tt-ip>`, user `<user>`, key `~/.ssh/<tt_key>`)** —
   create `/home/<user>/chronos2-autonomous/`, install a pinned upstream
   TT-Metal runtime (`runtimes/tt-metal-<commit>`) and a matching Python env
   (`environments/upstream-<commit>/…`) exactly like the NLLB prep, then fill
   `metal`, `metal_commit`, `python`, and the two `runtime_mounts` env paths in
   `cluster.yaml`. To use another TT machine, repeat the prep there and swap
   the single `compatible_ips` entry.
2. **Weights** — on the TT host run
   `python models.py chronos2 <cache-dir> --download` (registry-pinned
   revision; verifies SHA-256) and set
   `porting.tt.weights_by_model.chronos2` to the snapshot directory containing
   `config.json` + `model.safetensors`.
3. **Reference host (`<reference-ip>`)** — run `bash setup_reference.sh`
   (venv + torch + chronos-forecasting; prints device/BF16 support). CPU works;
   CUDA preferred. Fill `python`/`hf_home`/`known_hosts` in `cluster.yaml`.
4. **Reference smoke (one manual run, before any campaign)** —
   `python evaluate.py reference --stage smoke --model chronos2 --corpus runs/data/corpus.json --out /tmp/ref-smoke`
   on the reference host. This validates the `chronos-forecasting` API
   assumptions (`predict(inputs=[B,1,T], prediction_length, batch_size,
   limit_prediction_length)` → per-series `(1, Q, H)` tensors; `dtype` kwarg).
   If the installed package version drifted, fix `load_pipeline`/
   `extract_quantiles` in `evaluate.py` once now — the agent cannot modify the
   evaluator later.
5. **Cluster sanity** — `python -c "from cluster import load; load('cluster.yaml')"`
   from this directory must pass with all EDIT-ME fields resolved.
6. **Launch** — from this directory:
   `python loop.py --corpus runs/data/corpus.json --hours 6 --detach`
   (runs appear under `runs/`, status in `runs/<ts>/status.json`; `caffeinate`
   keeps the laptop awake on macOS). Campaign budget mirrors NLLB (6–24h).
7. **Machine swap mid-campaign** — edit `cluster.yaml` (single IP + paths);
   the controller drains pending jobs, verifies the replacement runtime/weights
   via `prepare()` and revalidates references. Never hand-edit health records.

## Notes

- Costs from the NLLB campaign apply: expect the interactive supervision, not
  the worker loop, to dominate token spend; ~0.4B input tokens covered the
  whole worker/supervisor side (1,145 calls) of the NLLB fresh campaign.
- The agent never pushes to GitHub; it prepares branch/commit/PR-body artifacts
  under the run directory and the operator reviews, DCO-signs and submits
  (tt-metal PR precedent: NLLB `tenstorrent/tt-metal#57398`).
- Chronos-2 is Apache-2.0 (weights and package), so unlike NLLB (CC-BY-NC) the
  port can be upstreamed without license exceptions; keep third-party code
  attribution Apache-2.0-compatible.
