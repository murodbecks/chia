# Research map for Tenstorrent model bring-up

Read this map once; retrieve the relevant source sections as needed. Do not ingest
the whole repository into your prompt. Record the installed TT-Metal revision,
hardware architecture, runtime version and each source's date/commit before use.

## Choose the execution stack deliberately

The [Forge bring-up guide](https://docs.tenstorrent.com/tt-forge/model-bring-up-guide.html)
describes compiler frontends, operator coverage, correctness-first bring-up and
progressive optimization. It is a useful process reference, but compiler APIs and
native TTNN APIs are different. Check which stack is actually installed. This lab
provides a built native TT-Metal/TTNN runtime; introducing another stack requires
a dependency/license review and a reproducible build. Do not presume a compiler
guide's hardware capacities or sample code match this pinned Blackhole runtime.

## Read the installed upstream repository

Use the `run` tool to inspect `/opt/tt-metal`; it contains public upstream source.
Start with its documentation index, `models/README.md`, `ttnn/`, `tests/ttnn/`,
`models/experimental/`, `models/demos/` and `tech_reports/`. Use `find` and `grep`
if ripgrep is unavailable. List names before opening large files.

Topic map (directory names must be checked at the installed revision):

| Question | Source area |
|---|---|
| Blackhole capabilities, compute/memory constraints | `tech_reports/Blackhole/` |
| Tensor layout, tile padding and dimensions | `tech_reports/tensor_layouts/`, `ttnn/` tests |
| Sharding and L1/DRAM placement | `tech_reports/tensor_sharding/`, `tech_reports/memory/` |
| Precision and accumulation behavior | `tech_reports/data_formats/`, `matrix_engine/`, `Handling_Special_Value/` |
| Linear layers and bandwidth limits | `tech_reports/GEMM_FLOPS/`, `Saturating_DRAM_bandwidth/` |
| Attention and encoder/decoder models | `tech_reports/FlashAttention/`, `LLMs/`, upstream model implementations |
| Profiling, dispatch and device traces | `tech_reports/MetalProfiler/`, `PerfCounters/`, `real_time_profiler/` |
| Advanced whole-model optimization | `tech_reports/AdvancedPerformanceOptimizationsForModels/` |
| Custom kernels after profiling | `tech_reports/op_kernel_dev/`, `prog_examples/`, `tt_metal/programming_examples/` |
| Host transfer costs | `tech_reports/PCIe_bandwidth/` |
| Multiple devices, collectives and fabric | `tech_reports/TT-Distributed/`, `TT-Fabric/`, `Programming_Mesh_of_Devices/` |
| Diagnosing hangs and correctness failures | `tech_reports/Debugging/`, minimal operator tests |

The repository is the exhaustive catalogue, not this shortlist:
[TT-Metal](https://github.com/tenstorrent/tt-metal),
[technical reports](https://github.com/tenstorrent/tt-metal/tree/main/tech_reports),
[TTNN documentation](https://docs.tenstorrent.com/tt-metal/latest/ttnn/).
Check signatures, supported shapes/dtypes, examples and tests together. Validate
each uncertain behavior in a tiny device experiment before putting it in a model.

## Search before reimplementing

Search open AND closed [issues](https://github.com/tenstorrent/tt-metal/issues)
and [pull requests](https://github.com/tenstorrent/tt-metal/pulls) for the model,
architecture, related encoder-decoder families and failing operations. Search
public model repositories too. Distinguish an accepted implementation from a
draft, benchmark-only path, unsupported architecture and CPU fallback. Record
the useful file/commit, unresolved reviewer concerns and what you reused. Do not
assume a PR's existence proves numerical correctness or upstream support.

Use web search for new developments and optimization ideas on other hardware.
For NLLB, investigate the official Hugging Face implementation and model card,
Meta's NLLB/FLORES resources, and
[CTranslate2](https://github.com/OpenNMT/CTranslate2) code/tests/documentation.
CPU/CUDA ideas such as weight packing, KV reuse, quantization, batching, fusion,
length bucketing and reduced dispatch are hypotheses to test on TT, not automatic
wins. Establish their numerical and memory consequences. Preserve upstream
licenses and distinguish reused code from independently inspired tests.

## Repeatable engineering cycle

1. Define the serving contract and pinned PyTorch reference. Verify reference
   determinism and generation semantics on the allocated NVIDIA compute node.
2. Decompose the graph; make individual operators and one layer correct before
   scaling to a model. Compare tensors and logits, not merely plausible text.
3. Test masks, mixed lengths, tile boundaries, batch permutations, repeated calls,
   language tokens, EOS and cache reset. A single fluent translation is no proof.
4. Preserve the first correct TT baseline. Profile actual end-to-end workloads;
   rank compute, transfers, dispatch, memory and padding costs.
5. Change one justified design choice, measure and keep or reject it. Check the
   complete supported shape envelope after a promising change.
6. Run broad frozen evaluation after development. Publish source hashes, commands,
   hardware/runtime identity, failures and comparable distributions, not just the
   fastest sample. Document unsupported operational features explicitly.

Do not reset cards, modify shared installations, install packages, change host
permissions or touch another user's allocation. Tools acquire a single device
lease and apply timeouts. Four cards permit independent experiments; model
parallelism is a separate engineering choice requiring correctness evidence.
