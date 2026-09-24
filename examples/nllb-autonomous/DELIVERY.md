# NLLB Blackhole delivery

The optimized 600M contribution is saved as a standalone experimental TT-Metal
model. The final targeted checks and both paired benchmarks have passed, including
natural B4/B1 output parity, mixed EOS/PAD behavior and cleanup. The contribution
is now pushed to `murodbecks/tt-metal`, branch `murodbecks/nllb-blackhole-600m`,
commit `80060cfe24f5241d44f93d4bd65e217c98a842b6`. No upstream PR or merge has
been performed; [submission instructions](artifacts/layout-review-20260922/NEXT_STEPS.md)
and updated issue/PR bodies are ready.

## Current package

The fork now contains 54 files organized into `tt/`, `demo/`, `reference/`,
`tests/`, `benchmarks/` and `docs/`, with a reviewer README. Identical benchmark
requests are stored once; three model configurations retain explicit hash
binding. The refactor removes 5,707 lines while preserving the inference code's
computation ASTs and kernel strings. CHIA packaging regression passed 483 CPU
tests, 117 subtests and native CLI/B1/B2/B4/masking/EOS/cleanup checks on pinned
main `86b55b92`. These short same-TT checks do not replace independent quality
qualification. [Current submission record](artifacts/layout-review-20260922/SUBMISSION.json).

## Historical tested export

[Final package and validation](artifacts/tt-metal-nllb-submit-20260922/STATUS.md)
contains 62 files / 33 Python files under `models/experimental/nllb`:
[patch](artifacts/tt-metal-nllb-submit-20260922/nllb.patch),
[source archive](artifacts/tt-metal-nllb-submit-20260922/nllb-blackhole.tar.gz),
[usage guide](artifacts/tt-metal-nllb-submit-20260922/models/experimental/nllb/DOCS.md).
Weights, private reference answers and credentials are excluded. An isolated
TT-host contribution checkout is ready at
`~/nllb-autonomous/contributions/tt-metal-nllb-20260922`, branch
`codex/nllb-blackhole-20260922`, with all 62 files verified against this export.
That remote checkout remains an uncommitted validation snapshot. The local
`chia/tt-metal` clone contains the published fork branch, adding SPDX headers and
updated documentation/manifest. Python ASTs and embedded kernel
strings match the tested export; [submission evidence](artifacts/fork-submission-20260922/SUBMISSION.json).
No DCO sign-off or upstream PR was made.
[Checkout receipt](artifacts/tt-metal-nllb-submit-20260922/CONTRIBUTION_CHECKOUT.md).

The CHIA/Codex worker implemented the model and packed/long-output changes.
During this finishing round, coordinator agents corrected the mixed-precision
policy and benchmark provenance, integrated the export, and ran bounded
correctness/performance jobs through CHIA. This was a supervised finishing
round, not a claim of an unattended worker-only completion. This round used targeted checks rather than repeating the full
corpus. All unsuccessful policies and diagnostic failures remain archived.

## Measured results

Same 600M long B2 request: two source rows of 134 tokens, 256-token generation cap,
48/175 learned predictions including EOS. Six balanced FULL/LAST pairs per
precision; every saved first/warm/timed output exactly matches cached A100 FP32.

| TT variant | Median request latency | p95, six samples | Predictions/s | Loaded device allocation |
|---|---:|---:|---:|---:|
| BF16 FULL control | 6.45293 s | 6.46493 s | 34.56 | 1.780 GB |
| BF16 packed LAST | 3.16389 s | 3.17083 s | 70.47 | 1.780 GB |
| Mixed BFP8_B packed LAST | 2.96033 s | 2.97239 s | 75.28 | 1.534 GB |

Within BF16, the balanced speedup is **2.0396×**. Within BFP8_B, FULL is 5.63496 s,
so its balanced FULL/LAST speedup is **1.9035×**. Across the sequential precision
runs, optimized BFP8_B has **6.43% lower latency** than optimized BF16 and is
**2.1798×** faster than the BF16 FULL control. Those cross-run comparisons were
not interleaved; they are contextual observations, not additional balanced trials.
This control already includes other TT optimizations. It is not the initial
PyTorch baseline or a NVIDIA performance comparison.

Loaded device allocation falls **13.82%**, from 1,779,531,776 to 1,533,556,736 bytes.
Post-request device allocation stays constant across each run's 12 timed calls.
These snapshots are not serving peaks. Host RSS at load is higher in the BFP8_B
run: 2.052 GB versus 1.774 GB. Model load takes 6.28–6.73 s; device initialization
and loading are excluded from request timing. First FULL requests take about
56–58 s including compilation; the fixed first-call order prevents a fair
cross-mode cold-start comparison. The six-sample p95 is not a production SLA.

[BF16 benchmark](runs/20260922T1040-main-public-natural-long-b2-cardc1/coordinator-independent-benchmark-review/RESULT.md),
[BFP8_B benchmark](runs/20260922T1135-main-public-natural-long-b2-bfp8-cardc1/coordinator-independent-benchmark-review/RESULT.md).
Historical five-workload speedup was 1.5681× against an earlier optimized TT
implementation; it uses a different baseline/workload and must not be multiplied
by these ratios.

## Correctness and precision

The final export passes **363 CPU tests and 34 subtests**, with zero execution
skips/xfails. 471 tests were collected, not all run. Real-runtime testing retained
upstream conftests. The BF16 parent also passed native CLI/B1/B2/B4, fresh-JIT,
packed dispatch and cleanup checks; AST comparison proves the final changes leave
BF16 computation and benchmark timing functions unchanged.

The revised BFP8_B policy passes all **22 long numerical checks** at unchanged
NRMSE ≤0.04 (maximum 0.03826197), the natural long B2 reference, and fresh
smoke/bring-up **13/13 FP32-exact rows**. One synthetic argmax differs, also seen
in the BF16 diagnostic; token equality is separately checked on natural outputs.
For 600M, transformer weights and activations remain BF16; only the large output
projection uses BFP8_B. The original mixed policy and an attention-only BF16
revision failed long checks. Their stronger memory reductions are not claimed
for this corrected policy.

[Final CPU review](runs/20260922T1145-final-submit-cpu-card42/coordinator-main-validation/REVIEW.md),
[precision-policy evidence](artifacts/bfp8-global-policy-investigation/RESULT.json).
Earlier full evaluations covered 8,096 translations in eight directions per
precision on the donor. The revised policy has targeted qualification, not a new
full-corpus acceptance claim.

## Before upstream merge

Source attribution is complete. Authenticate GitHub CLI or use the browser
submission instructions above; agree upstream CI owner,
hardware tier, checkpoint assets and required workflows; obtain maintainer
review. [PR draft](artifacts/pr-readiness-20260922/PR_DESCRIPTION.md),
[feature-issue draft](artifacts/pr-readiness-20260922/ISSUE_DESCRIPTION.md),
[CI proposal](artifacts/pr-readiness-20260922/CI_PROPOSAL.md).

The tested runtime is pinned main `86b55b92`, with Torch 2.14; upstream's default
Torch 2.11 stack remains unqualified. The supported experimental envelope is
600M greedy B1–4, source/generation through 256; low-level forward separately
limits decoder input to 64. Larger-model quality flags, concurrent serving,
beam search, incremental decoder KV caching and peak serving memory remain open.
Checkpoints retain CC-BY-NC-4.0 terms. No production certification is claimed.
