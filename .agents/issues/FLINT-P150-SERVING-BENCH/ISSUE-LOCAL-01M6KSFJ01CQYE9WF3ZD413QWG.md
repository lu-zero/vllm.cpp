ID: ISSUE-LOCAL-01M6KSFJ01CQYE9WF3ZD413QWG
Title: The first real Flint P150 serving benchmark -- vllm-bench at serving shapes on the adopted head
Row: FLINT-P150-SERVING-BENCH
State: OPEN
Kind: perf
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-04
Updated: 2026-11-04
Closed: -

## Problem

Flint on the P150 has never had a proper serving benchmark: the only number is FLINT-P150-ATTENTION stage 3's trace (32-token prompt + 6 steps, device-vs-host attention roughly neutral at 53.0 vs 55.2 s/step), taken BEFORE the pool-match fix, the NEON dot, the sliding decomposition, and the slot-serve cure landed. The numerics track is complete and adopted (device attention default-on, model gate holds at 2.198e-1/2.200e-1, allocation-order-insensitive) -- the bench runs against an adopted, gated configuration. Deliverable: vllm-bench at serving shapes (200/400 completions, concurrency 1 and 4), device-vs-host attention cells, docs/benchmarks/flint-p150-serving.md + BENCHMARKS.md row, ceiling-free with the named levers. Spec: .agents/specs/flint-p150-serving-bench.md

## Resolution

-
