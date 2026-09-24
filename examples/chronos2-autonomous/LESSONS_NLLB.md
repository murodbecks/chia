# Lessons from the NLLB campaign (binding rules)

Distilled from the NLLB-600M Blackhole campaign (fresh run Sep 20–22, 2026) and
the failed first attempt before it. Each lesson is a rule for this campaign;
violating one cost hours to days last time.

1. **Baseline before port, port before optimization.** The user's explicit
   order: measure PyTorch (and ONNX/forge if available) on the fixed suites from
   the TT host FIRST; pass correctness gates SECOND; optimize against the
   recorded baseline THIRD. An optimization claim without a preserved matched
   baseline artifact is invalid and will be rejected.
2. **Tiny tests during bring-up; full corpus only at the end.** The first
   attempt burned days running broad tests per step. Use `smoke` (1 case) and
   component checks until correct, `bringup` (8 cases) to gate, and request
   `full` only once, on frozen source. Never run full-corpus evaluations during
   development.
3. **One bounded step per turn; checkpoint STATE.md within about eight minutes.**
   Workers die on timeouts; an unsaved handoff loses the turn. Batch routine
   short gates on unchanged source (~4 gates / ~6 minutes), then reconcile
   handles and write STATE.md. The eight-minute mark is a strong target, not a
   hard rule: when a high-promise step is mid-flight, save a one-line progress
   checkpoint first and finish it — but never run past the ~30-minute runtime
   cap, and never let a turn end without a saved handoff.
4. **Never reset cards.** A card reset destroyed a day of work in attempt #1.
   Timeouts quarantine a card; only the controller's preflight recovery path
   re-admits it. You have no reset authority — report and continue other work.
5. **Do not repeat a failed experiment without a new hypothesis or evidence.**
   Re-running the same job to "see if it works now" wasted device-hours. For
   hangs, build a small timed reproducer, not another full evaluation.
6. **Truncated logs are not a reason to rerun.** Page the saved log with
   `chronos_log(job_id, offset_bytes=…, limit_bytes=…)` and continue from
   `next_offset_bytes`.
7. **Don't validate every cell early.** NLLB wasted effort qualifying all
   sizes/precisions before the anchor was solid. Here: one model, BF16 anchor
   first; BFP8_B is a separate later experiment; report unsupported cells
   honestly instead of inheriting passes.
8. **Pending jobs are polled, not duplicated.** Use `chronos_status` and do
   useful independent work while jobs run; a turn spent sleeping or re-launching
   a pending job is a wasted turn (and duplicates device load).
9. **Prepare submission materials early.** GitHub CLI auth was discovered
   missing only at packaging time last campaign. Prepare the tt-metal-style
   package, branch name, commit message and PR body as a mid-campaign milestone;
   the human operator reviews, DCO-signs and submits — you never push or open
   the PR yourself.
10. **Package in tt-metal style from the start.** Follow the layout of existing
    model demos (`tt/ demo/ reference/ tests/ benchmarks/ docs/`, README with
    measured evidence, SPDX headers). A late layout refactor invalidated
    benchmark provenance last time. User-prefixed branch names
    (`murodbecks/…`); never delete remote branches.
11. **Historical is not current.** A pass recorded for an older source, cluster
    or suite is context, never a current gate. Evidence matrices track the
    latest matching receipt; stale summary fields misled reporting before —
    keep STATE.md and REPORT.md current.
12. **Preserve baselines and rejected hypotheses.** The record of what did NOT
    work (decode-cache variants, precision policies, trace experiments) was as
    valuable as the wins. Record both with measurements.
13. **Float outputs are not tokens.** NLLB gates used exact token equality;
    here correctness is NRMSE against the FP32 oracle (≤0.04), cross-shape
    behavior consistency (≤0.01) and per-shape run-to-run determinism. Do not
    chase bitwise equality across batch shapes; do enforce it across repeated
    identical calls.
14. **Bounded scope beats a bloated workspace.** Attempt #1 died in a
    cluttered, growing workspace. Keep the workspace minimal, no recursive note
    archives, no embedded logs in STATE.md.
15. **Honest envelope reporting.** Unsupported contexts (>512), horizons (>64),
    multivariate/covariate group-attention modes and streaming are reported as
    unsupported, not implied. "Production-level" means the tt-metal package,
    tests and docs quality — not certification claims.
