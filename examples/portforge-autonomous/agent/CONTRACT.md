# Backend protocol v1

Implement `backend.py` in your workspace. No implementation is supplied.

```python
def create_backend(weights_path: str, config: dict, device):
    # Return an object implementing the methods below.
    ...

class Backend:
    def forward(self, input_ids, attention_mask, decoder_input_ids):
        # All inputs: CPU numpy int64 arrays [B,S] / [B,T].
        # Return {"encoder": float32 ndarray [B,S,D],
        #         "logits": float32 ndarray [B,T,V]}.
        ...

    def generate(self, input_ids, attention_mask, target_id, max_new_tokens):
        # Return numpy int64 [B,L]. Start token=2; first generated token is
        # target_id. Greedy, num_beams=1, do_sample=False. EOS=2, padding=1.
        # Count the forced language token in max_new_tokens, like HF generate.
        ...
```

The worker opens one TTNN device and passes it to the constructor. Do not open
another device or close the supplied device. Input arrays are independent copies;
do not mutate them. Return CPU-visible outputs before returning. The worker adds
device synchronization. Device-side work for each request belongs inside the
method call. Constructor work is measured separately. `forward` is a diagnostic
path; `generate` is the serving path timed end-to-end, including encode/decode,
transfers and host generation control, excluding tokenization and weight load.

An optional module-level `DEVICE_OPTIONS` dictionary can set `trace_region_size`
(0–128 MiB), `l1_small_size` (0–128 KiB) and `num_hw_cqs` (1–2); other keys are
rejected. Check the runtime's accounting: a trace-size argument may be per bank,
so it must not be reported as total card memory. TT call instrumentation and a
PyTorch dispatch guard detect common CPU fallbacks during inference. Instrumented
timing is used consistently for baseline and candidates; keep source auditable.

The weight file is a pinned official PyTorch state dict. Use
`torch.load(weights_path, map_location="cpu", weights_only=True)`. CPU weight
conversion is allowed; learned inference must execute in TT operations. Config
comes from the pinned official model. No network or reference outputs are mounted
in the execution sandbox. You can inspect weights and installed upstream source
using `run`; only your files can be changed.

Required initial envelope: greedy translation, batch 1–4, source up to 256 tokens,
generation cap up to 64, arbitrary valid language IDs. Preserve masks and padding.
The evaluation corpus uses English/French/Uzbek/Arabic/Chinese in eight directions.
This is a bounded research/serving contract, not all NLLB languages and decoding
modes. Do not claim beam search, streaming, full context, concurrent serving or
production readiness without adding and testing those features.

The controller evaluates `smoke`, `development`, `qualification`, then `full`.
Use smoke during bring-up. Development adds lengths around tile boundaries,
batch/single/reorder/padding/repeat and Unicode/empty inputs. Qualification samples
128 aligned FLORES devtest IDs (1,024 translations); full covers all 1,012 aligned
IDs in these eight directions (8,096 translations). New transformed inputs for
final validation are withheld until source submission; they are distinct from
the full corpus, rather than pretending full-corpus rows remain held out.
The corpus is public; do not retrieve it to
memorize translations. Held-out results are not used for tuning.

Frozen gates: finite, correctly shaped tensors; encoder and last-prefix logits
NRMSE <= 0.04 against FP32 CUDA; exact batch/single/reorder/padding/repeat semantics;
valid token IDs/EOS/caps; chrF++ no more than 0.5 below FP32 CUDA in each direction
on qualification/full. Report exact token agreement as well. Numerical tolerance
does not justify silent semantic regressions. The first passing development
snapshot is preserved as the TT baseline. Subsequent speedups compare the same
requests, timing boundary, repetitions and generation lengths. Full and final
evaluation are required for an accepted campaign; smoke success is not completion.

Tools return job IDs for asynchronous device work; poll them and keep building
while jobs run. Store research, tests, STATE.md and REPORT.md alongside your code.
Use `submit` only after broad tests pass and your report is complete.
