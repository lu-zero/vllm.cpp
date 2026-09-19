ID: ISSUE-LOCAL-01M2XSWCP507W0M18EXHGR5TGS
Title: TT prefill M>1: RmsNormGated gate reshape fatals MeshBuffer, SigmoidGateBf16 then OOMs
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-19
Updated: 2026-09-19
Closed: -

## Problem

At concurrency>=2 on the Qwen3.8-27B-APEX TT vehicle, the PREFILL step still throws 'MeshBuffer must be large enough to hold the tensor' (assert.hpp:104) inside RmsNormGatedKernel's act reshape ([256,6144] TILE -> {12288,128}, equal numel AND equal tile count, yet tt::tt_metal::view refuses). The fatal is silently caught upstream; the degraded step then reaches SigmoidGateBf16 whose ttnn::multiply allocates a 36 GiB output (inputs carry a poisoned [12288,786432]-class spec) and the engine dies (bank_manager.cpp:495). Repro: vllm-bench --model Qwen3.8-27B-APEX-I-Nano.gguf --num-prompts 2 --input-len 128 --output-len 32 --concurrency 2 --temperature 0 on P150, VT_TT_AFFINE_F32=1 VT_TT_NORM_PAD=1 VT_TT_PROGRAM_CACHE=1. Evidence: /tmp/m2_rng2.log (VT_TT_RNG_TRACE stage prints bracket the fatal inside CaptureSafeReshape -> ttnn::reshape), /tmp/m2_bench_fixed.log (OOM backtrace through BinaryNgDeviceOperation).

## Resolution

-
