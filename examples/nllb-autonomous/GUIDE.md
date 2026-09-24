# Fresh NLLB → Blackhole loop

See `runs/active.json` for campaign pointers. Every new run freezes its harness;
existing campaigns retain their archived versions. DELIVERY.md records evidence
and remaining work. The current root harness supports all three registered sizes
and includes the host-memory watchdog.

Start with `cluster.yaml`: it contains machine addresses, SSH key **paths**, runtime
paths and resources. No SSH aliases are used. The inventory/auth sections use
CHIA's native schema and loader; `porting` supplies this example's runtime settings.
The current adapter keeps Ray/agent orchestration local and dispatches accelerator
jobs over SSH. It does not call `chia up`, provision instances or establish a
multi-host Ray network.

| File | Responsibility |
|---|---|
| `cluster.yaml` / `cluster.py` | Machine inventory, verified SSH, durable direct/Slurm jobs |
| `loop.py` | CHIA graph, scoped tools, reference and frozen final evaluation |
| `agent.py` | Codex worker and supervisor using the existing login |
| `runner.py` | TT isolation, leases, phase deadlines and quarantine |
| `evaluate.py` | FP32 CUDA oracle, precision comparisons and independent gates |
| `models.json` / `models.py` | Pinned checkpoint manifests and streaming identity verification |
| `setup_reference.sh` | Pinned private GPU environment; no sudo or driver changes |

The configured TT wrapper selects the isolated upstream build fd80faa3… and a
Transformers5.12.1/tokenizers0.22.2 environment, retaining the original
Torch2.14+cpu/NumPy1.26.4 packages. Native paths/hashes, dependency locks and
GPU/TT tokenizer comparisons are in
`runs/20260920T1945-runtime-migration/upstream-build/`. This differs from upstream’s
exact default Python environment; model coverage is recorded in DELIVERY.md.
On a replacement TT host, prepare and verify its runtime and tokenizer dependencies
before selecting its Python path; changing the address alone does not install them.

Optimization resumed in `runs/20260921T1910-decode-cache-card01/` using the actual
worker/supervisor pair. Its policy and offline preparer are in
`artifacts/decode-cache-phase/`. Start with tiny600M BF16 cache probes, then fresh
short gates and matched natural-translation timings. Existing code/tests and a
protected comparison runtime are preserved; full-corpus and new CUDA jobs are
disabled during this development phase. Check its exact controller/job handles
before inferring liveness or restarting. No new gain is established at launch.

The latest completed bounded development phase is
`runs/20260921T1520-quality-policy-v2-card41/`; inspect its saved state and exact job
handles rather than restarting it. The pair completed qualification and retained
the failed gate; the prototype is not promoted. Its worker/supervisor pair uses the
trusted asynchronous sample tool in `artifacts/quality-repair-phase-policy-v2/`.
The phase permits tiny probes, smoke/bringup, and at most two changed-production
128-pair checks. Full-corpus jobs and new CUDA reference generation are disabled.

For a subsequent phase, `artifacts/quality-repair-phase-policy-v2/prepare.py` creates a new
run and workspace offline; its `--help` documents the required run, cluster and
card arguments. Review the resulting `PREPARATION.json` and `launch-command.txt`,
verify card availability, then use that exact command. Preparation imports no
candidate passes. An existing controller state requires explicit recovery rather
than relaunch. The general root `loop.py` also supports longer campaigns, including
full evaluation and CUDA precision studies; it is **not** the bounded entry point
for the current speed policy.

For another machine, change its `compatible_ips`, matching `auth.overrides`, and
`porting` paths. For an already running campaign, edit the exact `cluster_file`
recorded in its manifest (also linked from `runs/active.json`); the root
`cluster.yaml` is the default for new launches. SSH host verification remains enabled; register/verify a new host
through your normal connection procedure first. GCloud's existing known-host file
and instance host-key alias are configured explicitly. The reference executor is
`direct` for a dedicated GPU VM, or `slurm` with `slurm.partition` for a login host.
The evaluator refuses login-node or CPU reference inference.

The general root controller rereads the cluster file between worker turns. It drains
pending jobs, verifies the replacement TT runtime/model/card health, then prepares
a fresh reference as needed. In-flight work stays pinned to its original machine;
it is not live-migrated. Source work survives a replacement. Evaluations and frozen
submissions record precision and cluster identity, so passes and timings from a
different machine/mode cannot silently authorize a new submission. Concurrent-job
capacity is selected at launch; restart to increase that capacity. Failed checks
pause/end the attempt with evidence. An ambiguous remote submission is never
blindly repeated; inspect its saved job record.
The current bounded quality phase additionally pins its environment and sample
inputs. Move that phase to another host through a fresh reviewed preparation and
source handoff; its qualification receipts do not transfer across environments.

Development: **one smoke case → eight bringup cases → targeted model/precision checks**.
The current speed policy keeps one representative full evaluation (600M BF16 has
passed) and uses a small fixed multilingual sample for larger-model qualification;
that sampled path is implemented by `nllb_qualify_sample` in the bounded phase.
Develop new general optimizations on600M with a few numerical checks and2–4
translations; run only relevant regressions after each edit. Use tiny size/precision
gates before a wider milestone sample, and reserve comprehensive validation for
the final candidate. The current optional precision experiment uses1.3B BF16 to
follow its diagnostic evidence; its sample flag also exists in the baseline.
Neither that experiment nor a smaller-model pass changes the final three-size,
two-precision target. Do not run128-pair screens after every edit.
Additional full runs are deferred. Existing
`nllb_submit` still executes its unchanged full gate; sampled evidence is reported
separately and never presented as a full pass.
The full suite retains 8,096 FLORES translation pairs plus behavior and boundary
checks. Earlier private ports are unavailable to the worker. Existing installed
runtimes, official weights and a hash-pinned input corpus are reused.

FP32 is the immutable accuracy oracle, not a restriction on TT inference. Start
TT bringup in BF16, then test selective `bfp8_b` and higher-precision exceptions.
TT block floating point is not IEEE FP8; unsupported native FP16 fails explicitly.
`nllb_evaluate(stage="smoke", precision="bf16")` and
`nllb_submit(precision="bf16")` choose and freeze a mode. CUDA diagnostics compare
FP32/FP16/BF16 using `evaluate.py reference-benchmark`; all modes retain the same
numerical and behavioral gates. Small-suite agreement is not full translation
quality. See CONTRACT.md and TT_GUIDE.md for policies and primary sources.

The shared artifact targets 600M, distilled 1.3B and 3.3B.
Separate frozen600M baseline/candidate full evaluations retain their original
source and runtime; their results are not inherited by the shared artifact.
The registry also pins sharded 3.3B; registry entries alone do not establish runtime support. Before
loading a staged checkpoint, verify its config, weights and optional shard index:
`python models.py 1.3b-distilled /path/to/checkpoint` (add `--all-files` to verify
the recorded tokenizer/generation files too). This hashes bytes without loading
pickle files. See DELIVERY.md for the remaining multi-size and upstream goals.
To populate a cache with the installed Hugging Face client and verify every pinned
file, run `python models.py 1.3b-distilled /path/to/hf-cache --download`. The command
uses public anonymous downloads and never loads the checkpoint as Python pickle.

The current contribution is staged in `artifacts/tt-metal-pr-results/`, including the
`models/experimental/nllb` tree, reproducible patch, hashes and provenance.
Its REVIEW.md records validation and remaining upstream work. Earlier exports
are historical evidence; worker edits do not update the staged package.
No CHIA credentials or weights are included.

Logs, CHIA traces, cluster snapshots, prompts, sources, jobs and verdicts live in
`runs/<id>/`; remote roots come from the YAML. `status.json` reports the controller
state. The temporary candidate workspace is recorded in `manifest.json`; every
evaluated source is saved under `jobs/`. Agents can page through their own TT
stdout/stderr with `nllb_log`; reference answers remain inaccessible.
`nllb_status(compact=True)` keeps structured results and bounds repeated log tails;
no-argument status preserves the detailed legacy response (12 recent jobs plus all pending).
Use `nllb_status(job_id="saved-job-id")` for an older gate's complete structured
result and request identity, with bounded log tails. Exact lookup reads only saved
local receipts, including pending jobs; it does not rerun jobs or refresh remote progress.
`--detach` records the PID/log and inhibits
macOS idle sleep, but cannot keep a closed or disconnected laptop running.

Newly frozen supervisors also receive a bounded excerpt of `STATE.md` and the
selected Python path. This preserves the worker's handoff after a timeout or
environment change. The notes remain advisory; saved job receipts determine
liveness and correctness. Existing frozen campaigns retain their prior harness.

The root supervisor now also receives a model/precision/stage table for the shared
artifact: all three sizes, BF16/BFP8_B, and smoke/bringup. It names the latest
matching source/environment job per cell. Missing, historical, failed, pending
and incomplete receipts cannot appear as current passes. The selected model is
the development anchor; it does not hide missing coverage for other sizes.
This table informs planning, without replacing the full-quality gates.
Newly frozen supervisors use a JSON schema and request one bounded work unit.
Unchanged short gates can run in batches of up to four or six minutes, with
pending handles and a concise receipt-based handoff saved within eight minutes.
Unexpected failures, source drift or blockage end the batch; all gates and stage
dependencies remain mandatory. Existing frozen campaigns keep their older prompts.
They preserve each reply and allow two bounded retries
for malformed JSON or oversized instructions. A valid pause is honored; provider
failures are not retried as formatting errors. This never reruns a hardware job.
New submissions also require a matching successful terminal receipt from the latest
current-source bringup. A pending or failed retry cannot fall back to an older
pass; contradictory failure flags and mismatched identities block submission.

The current root controller can attempt one real device preflight per card and
cluster identity after a recorded terminal timeout, once all TT jobs have ended.
It requires matching job/card/cluster receipts and records an exclusive submission
intent before contacting the host. A failed or ambiguous probe consumes that
attempt; memory-limit failures are not automatically recovered. The runner still
owns leases and quarantine, and no reset or manual healthy mark is performed.
Existing frozen campaigns, including `20260920T2142-shared`, predate this change.

Tests live in `tests/`: run `python -m unittest discover -s tests` here using CHIA's
Python. If it lacks NumPy, run `test_evaluate.py` separately with a NumPy-equipped
Python. Device and model results are separate from unit tests. Full validation can
outlast the worker budget and has its own limit. Full success is bounded research
acceptance, not production certification; NLLB weights are CC-BY-NC-4.0.

Newly frozen harnesses also accept `nllb_run(command, model="3.3b",
public_input_stage="bringup")` for model-specific standalone tests and profiling.
The selected checkpoint and public suite move together; the development anchor
stays unchanged, no oracle is mounted, and a run does not award an evaluation pass.
The600M configuration currently mounts a single `.bin`, while larger models mount
checkpoint directories. Use the public stage's `/input/config.json` when config
is not beside the weights; standalone CLI supports an explicit `--config` path.
The successor `20260920T232513Z-model-routing` includes this argument. Its predecessor
stopped at a verified idle boundary after recording all precision cells, including
the600M BFP8 numerical failure. Source and current BF16 baseline notes were preserved;
candidate passes are not inherited. See `model-routing-launch.json` and the boundary
receipt under `runs/20260920T1945-runtime-migration/` for the handoff.

The trusted TT runtime defaults to a 64 GiB summed-process RSS limit, configurable
with `porting.tt.host_memory_limit_bytes` (1–128 GiB). Sampling includes observed
detached descendants and can double-count shared pages; it is not an allocator
peak measurement. Memory interruption records `host_memory_limit` and quarantines
the card until a real preflight succeeds. Existing frozen campaigns do not acquire
new runtime features automatically.

Newly frozen harnesses accept `porting.tt.full_timeout_seconds` (1–24 hours,
default six hours) for the full job's outer deadline. The SSH allowance follows
that limit; phase/initialization deadlines and memory checks remain fixed.
Set this from measured workload estimates before freezing a run. Candidate tool
requests cannot override it. Existing frozen controllers retain their old limits.
