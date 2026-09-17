# PortForge-TT: current work

## Goal

Build a CHIA loop that maps transformer operations/subgraphs to efficient
TTNN/TT-Metal implementations on Blackhole, using adversarial numerical tests
and repair. The submitted [hackathon proposal](../../assets/proposal.tex) and
[university proposal](../../assets/uni_proposal.tex) define scope.

Verified bring-up workload: [residual addition + RMSNorm](../portforge-tt-rmsnorm/RMSNORM.md).
Start with TTNN layout, memory placement, sharding, and fusion choices. Add
custom kernels only when needed. Compare with stock TTNN, random search,
and single-pass agents; preserve a final held-out workload split.

## Current state — September 15, 2026

- Pinned TT-Metal v0.72.0 is built; CHIA and TTNN share its isolated uv
  environment. See [SETUP.md](SETUP.md) for the working commands.
- The Mac runs CHIA/Ray and Codex using the existing subscription login;
  a trusted evaluator runs over SSH on the Linux host.
- GPT-6 Astra, the CPU smoke loop, and the complete Blackhole repair loop
  passed. Fixed evaluator checks passed on all four cards (three in parallel).
  See the [smoke evidence](../portforge-tt-smoke/SMOKE.md) for exact scope.
- The user authorized all four cards, including parallel experiments.
  NLLB correctness experiments use all four with per-card advisory locks;
  controlled timings run sequentially on one card.
- Separate and fused RMSNorm baselines passed 31 development cases on all four
  cards, including warm rechecks, nonzero padding and input integrity. A bounded
  Codex repair selected fused and passed; 15 offline regressions passed.
- Gemini remains required for the hackathon. Its provider integration and
  credentials are still pending; the smoke driver currently uses Codex only.
- Fused dispatch was faster in initial runs, with material timing variability.
  No stable kernel-speedup or adversarial-search result exists yet.
- [NLLB port](../portforge-tt-nllb/NLLB.md): full encoder, decoder and vocabulary
  projection run on one Blackhole, with pinned PyTorch FP32 references on Slurm.
  Runnable code and evidence live in `../portforge-tt-nllb/`, not this hub.
- [Numerical/batch repair](../portforge-tt-nllb/REPAIR_RESULTS.md): 600M and 1.3B
  repaired trace passes 1,024 FLORES quality pairs plus 221 auxiliary requests
  per model. Both repaired native/traced pairs pass 256 fresh quality pairs plus
  221 auxiliary requests, with identical outputs and all numerical/replay gates.
  Precision repair, source trimming and independent rows fix earlier failures;
  the latter sacrifices native batching throughput. 3.3B repair is deferred.
- [Controlled measurements](../portforge-tt-nllb/PERFORMANCE.md) compare original,
  previous optimized, repaired native and repaired trace variants. They record
  warm p50/p95, first workload requests, throughput, loading and scoped memory.
  Repaired trace gains 1.884× (600M) / 2.299× (1.3B) over matching repaired native
  across five workloads; original-baseline comparisons include batch regressions.
  Historical small-suite speedups live in [ADVANCED.md](../portforge-tt-nllb/ADVANCED.md).
- NVIDIA's FP32 reference remains faster on these short workloads. CTranslate2
  informed the research but has not been timed. No memory reduction is claimed.
- The subscription-backed Codex CHIA node consumed the CTranslate2/TT-Metal research
  brief and selected the trace candidate. Development-measurement sharing was
  explicitly authorized; no approval/security settings were changed.
- These are reproducible integration results, not production or translation-quality
  certification. SDPA and additional precision variants retain their failed runs.
- The original [failing campaign](../portforge-tt-nllb/QUALIFICATION_RESULTS.md)
  remains preserved. Repairs do not guarantee exact PyTorch FP32 translations
  or improve every chrF++ score; current reports retain those differences.

## Next tasks

Preserve fixed gates, environment and failed-run evidence.

1. Improve native batching while retaining the fixed batch/padding contract.
   Profile the old trace mapping's regressions on longer batches separately.
   Keep evaluated data as regression evidence; 96 reserved aligned FLORES IDs
   remain unused for a later validation campaign. Establish translation-quality
   noninferiority criteria before further tuning, and investigate current declines.
2. Integrate Gemini for hackathon use and replay the recorded configurations.
3. Extend controlled measurements to varied serving traffic, controlled cold
   startup, an optimized NVIDIA reference, and peak-memory/long-soak checks.
   Measure TRACE-region usage before reducing its per-bank reservation.
   Investigate longer generation and device argmax; incremental self-K/V currently
   adds overhead on short outputs.
4. Add isolated candidate execution, bounded/adversarial repair, persistent
   measurement history, equal-budget baselines and ablations.
5. Prepare the four-page report and reproducible artifact. The last checked
   deadline is September 24 AoE; verify final rules before submission.

## Experiment locations

This directory is the documentation hub. Runnable loops, tests, and artifacts
belong in sibling experiment directories:

- [Addition smoke](../portforge-tt-smoke/SMOKE.md).
- [Residual RMSNorm](../portforge-tt-rmsnorm/RMSNORM.md).
- [NVIDIA reference worker](../portforge-tt-reference/REFERENCE.md).
- [NLLB model port and CHIA workflow](../portforge-tt-nllb/NLLB.md).

## Context to load only when needed

- [HISTORY.md](HISTORY.md): full setup/research history, version rationale,
  dependency review, build commands, earlier failures, and verification logs.
- [CHIA basics](../../docs/getting-started/chia-basics.rst) and
  [architecture](../../docs/concepts/overview.rst): framework reference.
- [Smoke runbook](../portforge-tt-smoke/SMOKE.md): bring-up checks, not research code.
