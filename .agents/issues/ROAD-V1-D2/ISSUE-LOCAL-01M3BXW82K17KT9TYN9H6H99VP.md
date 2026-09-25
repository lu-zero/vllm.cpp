ID: ISSUE-LOCAL-01M3BXW82K17KT9TYN9H6H99VP
Title: test_qwen4_exp_layer_loop fails on main: by-name second-prompt cache movement is 0
Row: ROAD-V1-D2
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-25
Updated: 2026-09-25
Closed: 2026-09-25

## Problem

At main 636926736, test_qwen4_exp_layer_loop fails: 'by-name second-prompt logit movement: 0' — CHECK(moved > 0) with 0 of 128 indexer-cache words and 0 of 192 paged K/V words moved across two prompts (CI job 107975227373, 2026-09-25). The by-name cache path stopped moving words between prompts — a regression in the KV by-name/multi-cache work landing between 2026-09-12 and now. Blocks every open PR's build-test-cpu.

## Resolution

2026-09-25: fixed on row/FIX-QWEN4EXP-BYNAME-MOVEMENT. NOT a KV/by-name regression at all. A verified `git bisect run` (good 2415a6b22, bad 53f5af17d, build+run predicate on the by-name case) named 4756d264c (feat(MODEL-KEV), 2026-09-24) the first bad commit, and a surgical revert of only its `src/vt/cpu/cpu_ops.cpp` hunk on top of main restored the by-name movement to 31.8438 — the exact value `.agents/specs/qwen4-exp-flash-next.md` records ("0 of 128 ... 0 of 192 ... while the logits moved by 31.84"). Root cause: that commit re-pointed the shared CPU `RmsNormGatedKernel` bf16 arm from vLLM's chain (f32 end to end, ONE bf16 round at the store — `layernorm_guard.py` `layer_norm_fwd_kernel`, `RMSNormGated.forward_static`, this tree's own CUDA `RmsNormGatedRowFastKernel` `cuda_gdn.cu:2025` and fused `RmsNormGatedQuantFp8Kernel`) at the HF-transformers fallback `Qwen3_5RMSNormGated` (norm→bf16 before the weight multiply, product→bf16 before the gate), to satisfy kev's uncommitted manual Phase-6 byte-for-byte parity against the transformers-based kev reference server. The two extra roundings are an order of magnitude under the fixture's layer-3 activation ULP, so the by-name fixture's knife-edge prompt separation collapsed from 31.84 to exactly 0 logits movement, reddening the reachability CHECK at `test_qwen4_exp_layer_loop.cpp:1571` (one of the two assertions the row's MUT-REACH measured as catching a tower that ignores `token_ids`). vLLM is the reference and implements this norm on every path, so the shared kernel keeps the vLLM chain; no committed test depends on the transformers chain (all `test_kev` goldens are PointerHead/LoRA/encode-level; kev Phase-6 E2E is deferred and its spec already says "near-tie robust for bf16 paths"). The fix restores the f32-through store in `RmsNormGatedKernel` with a comment naming both chains and the anchors. Focused gate: `test_qwen4_exp_layer_loop` 598/598 (movement 31.8438); blast radius green at main+fix: `test_kev`, `test_ops_rmsnorm`(+weight_dtype+elem_dispatch), `test_qwen4_exp_qsa_block`, and all 35 qwen3_5/gdn/qwen4_exp family binaries.
