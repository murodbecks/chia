# Current autonomous campaign

Resumed 2026-09-17 at 14:13 UAE after model-capacity errors and PCIe faults.
The original workspace is retained; this is a continuation, not another fresh
experiment. Four idle cards were reset after checking device ownership and
holding all leases. Isolated device compute passed, then the selected candidate
passed smoke again (trial `6b479f5de90740a2a1fe933d0a7b9719`). Development trial
`5707fa5830b84ac48dc2eef0ac6ad4a5` was launched next; see live state for its outcome.
No development/qualification/full acceptance is implied by this status note.

Continuation PID, recovery record and active round are in `status.json`.
The original manifest PID is historical. Controller recovery and capacity-error
handling are covered by 31 passing local tests; six unchanged NumPy tests skip
locally. A bounded capacity supervisor is recorded in the recovery directory;
it never interrupts a live controller or resets hardware.

Originally started 2026-09-17. Main campaign:
`runs/20260917T040059Z/`, controller PID recorded in its manifest/launch file.
Read its `status.json` and `events.jsonl` for live state; this page is a launch
record, not a claim that the campaign has finished.

The agent starts with four instruction files and no implementation. Codex uses
the existing subscription login. CHIA schedules its actual tool/evaluation calls;
up to four isolated Blackhole jobs can run in parallel. Fresh PyTorch references
run through Slurm on the NVIDIA lab. See [OPERATIONS.md](OPERATIONS.md) for the
graph, source/evidence locations, replay command and acceptance limits.

Verified during setup:

- 24 Mac harness tests passed; the six NumPy artifact tests passed on the TT host
  (30 tests total across the two environments).
- Real device initialization/transfer and allocator memory collection passed in
  the isolated namespace. Eight DRAM banks are accounted for explicitly.
- Real Codex MCP calls worked; traversal was rejected; no host shell, image,
  apps, memory or previous-session tools were exposed in the final preflight.
- A deliberately invalid CPU matmul backend was rejected by the real TT worker.
- The tokenizer/SacreBLEU scoring path passed a labeled identical-output fixture
  on Slurm. This fixture is a harness test, not a model-port result.

The initial campaign `runs/20260917T035011Z/` was stopped after exposing a memory
collector API mismatch. It independently researched, wrote, tested and repaired
its own backend, then passed one smoke case: exact NVIDIA token match, encoder
NRMSE 0.00349, logits NRMSE 0.00434 (gate 0.04). Its reference ran on an RTX 5000
Ada, node `ws-l5-002`, Slurm job `197518`. Evidence is in trial
`e11698f885ab468f811319a4a2adf0b2` and `validation/`.

The main campaign restarts from an empty implementation after fixing the harness.
No source or optimization decisions were copied from the initial campaign. Its
harness is archived locally and installed remotely by content hash. The device
preflight verifies memory collection before the new agent starts.

Smoke success does not establish broad correctness, optimization gains or
production readiness. Development, qualification (1,024 translations), full
selected-language corpus (8,096 translations), and final transformed inputs are
separate gates. Full-corpus evaluation and optimization are still pending at
launch. Memory values are snapshots; bandwidth is not measured by the wrapper.
