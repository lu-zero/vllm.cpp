ID: ISSUE-LOCAL-01M32FVCE8XBW2PDTE48ZD8TWN
Title: Measure vllm-tt-plugin gateability on the row's P150 (qwen35 served, tokens emitted)
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: 3261
Mirror: SYNCED
Availability: FULL
Created: 2026-09-21
Updated: 2026-09-21
Closed: -

## Problem

The vllm-tt-plugin oracle record (.agents/oracles/vllm-tt-plugin.md, pin 7250ddfaa, 2026-09-21) is filed with gateable = no: nothing from the plugin has executed in this repository. The oracle-registry checker requires an owing issue named on a gateable=no record; this is it. Owed measurement: install vLLM 0.26.0 (VLLM_TARGET_DEVICE=empty) plus the plugin at the pinned HEAD inside a tt-metal environment on the row's P150, serve one registered qwen35 checkpoint, and record tokens emitted. Establishing gateability also unlocks the primary-rank denominators the Tenstorrent rows want: on-device qwen35 graph correctness at bf16 (vs our GGUF-decoded bf16 arm, per-layer like the llama.cpp bisect) and the native ~50 tok/s qwen35 rate measured side by side on the same card. Source: vllm.ai blog 2026-09-07; supported architectures include TTQwen3_5ForConditionalGeneration. NOT established by this measurement: GGUF-decode parity (the plugin serves HF weights; llama.cpp stays the GGUF decode oracle).

## Resolution

-

## Update 2026-09-22: deadlock reproduced on CURRENT tt-metal main

A fresh tt-metal main clone (`v0.80.0-dev20260922`, built from source with
the aarch64 toolchain, gcc-15-built SFPI 7.80.0, profiler disabled, and a
one-line local patch dropping an undefined `ttexalens::elf` CMake link)
hangs at the IDENTICAL spot: `qwen36/tt/gdn/weights.py:115 load_gdn_weights`
-> `ttnn.as_tensor` (qkv_proj [4096,8192] bf16 -> bfloat8_b), 23 minutes,
all EngineCore threads futex-blocked.

The deadlock is therefore deterministic across both tt-metal builds tested
(v0.79.0-dev20260911 and v0.80.0-dev20260922), lives in the tt-metal python
stack's GDN weight conversion on aarch64, and blocks both the plugin serve
and tt-metal's own demo. Full stack + environment captured in
[`.agents/oracles/vllm-tt-plugin.md`](../../oracles/vllm-tt-plugin.md) —
ready to file upstream (tenstorrent/tt-metal or vllm-tt-plugin) on request.

Build fixes this required on aarch64/gentoo (worth reporting upstream too):
`build_metal.sh` hardcodes the x86_64 toolchain (pass
`--toolchain-path cmake/aarch64-linux-clang-20-libstdcpp-toolchain.cmake`);
SFPI 7.80.0 must build with host gcc-15 (gcc-16 hits a libcody char8_t
error); the profiler WASM subproject needs `--disable-profiler`; and the
ttexalens CPM clone materializes empty (manually cloned 6f572024 into
`.cpmcache/ttexalens/a20f65d9...`).

## Update 2 (2026-09-22): SHARPEST BISECT — the defect is in the plugin's vLLM-integration context

Direct probe, no vLLM: `Qwen36ForCausalLM.initialize_vllm_model(hf_config,
mesh, max_batch_size=1, max_seq_len=2048)` on the same device/checkpoint
COMPLETES in ~4 minutes — GDN weight load, the exact conversions that
blocked in the serve. Within the vLLM engine (TTModelLoader.load_model →
the same call), it blocks indefinitely at the same line. `max_model_len`
size is irrelevant (2048 blocks identically).

Conclusion: the defect is in the plugin's vLLM INTEGRATION context —
TTWorker device/mesh configuration or the threading around load_model —
not in tt-metal's qwen36 model code and not in ttnn. vllm-tt-plugin is
the right upstream home for the report; the stack, the standalone-pass
probe (/tmp/direct_init_probe.py), and this issue are the full package.

## Update 3 (2026-09-23): in-process workaround dodges defect 1; a SECOND tt-metal livelock gates prefill

`VLLM_ENABLE_V1_MULTIPROCESSING=0` (engine in-process, no spawned
EngineCore) PROVES the weight-load deadlock is spawn-specific: the serve
loads all weights and compiles the decode trace. But the run then sat at
~110% of one core for ELEVEN HOURS in
`capture_prefill_trace_chunked` (the 2048-chunk prefill trace capture) —
no progress, the same phase where tt-metal's own traced_128 demo timed out
at 90 minutes. Two independent tt-metal aarch64 defects gate the serve:

1. Spawned-child dispatch deadlock at GDN weight load (worked around
   in-process).
2. Prefill trace-capture livelock (no workaround; the phase never
   completes on this host/build).

The gateability measurement remains blocked on defect 2; the upstream
report should cover both, with the in-process control as the isolation
between them.
