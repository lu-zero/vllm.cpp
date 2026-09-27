# Spec: the dispatch-cost row — decompose the ~30 ms/op, then the levers

Row: `BACKEND-TENSTORRENT`. State: DRAFT (2026-09-27).
Issue: files against the perf-campaign row.
Follow-up to: the decompose (#3322: 1,037 ops at ~32 ms eager, ~30
ms/command trace-side; 572 matmul-class ops; kernels 1.27 s — 25x
headroom), the GEMV trace spec `tenstorrent-gemv-overhead-trace.md`
(the W1 phase-attribution pattern, committed), and the capture-economics
curve (row 1 of this pair: the fixed/len term decides how much the
dispatch matters at real lengths).
Git integration: one pull request (spec + instruments + the decomposition
+ the lever verdict), branch `row/TT-DISPATCH-COST`.

## Problem

Both paths pay a ~30 ms per-op dispatch cost: the eager cold body (1,037
ops at ~32 ms) and the trace execution (~30 ms/command). The kernels
themselves are 1.27 s/forward — the dispatch is ~25x the compute. The
W1 GEMV spec's original question (one 76.5 ms GEMV call) generalizes to
the whole step: WHERE inside one op's dispatch does the time go — launch
submission (host->device), device execution wait, or completion sync —
and which levers follow.

## Measurement (W1's pattern, per-op)

1. **Per-op phase attribution**: instrument the op dispatch path (the TT
   backend's op invocation) with the W1 timestamps — (a) host-side
   operand/descriptor construction, (b) launch submission to return,
   (c) completion wait. One knob (`VT_TT_OP_PHASES`, the
   `VT_TT_STEP_PHASES` pattern at op granularity), per-op stderr lines on
   the captured step.
2. **The op-class breakdown**: aggregate per op class (matmul-class 572,
   keepquant repair, layernorm/gdn/paged) — which class pays the most
   dispatch and where (its own phase mix).
3. **The amortization probe**: N identical ops back-to-back (the W1
   spec's probe) — does per-op cost collapse at steady state (launch
   latency, amortizable by fusion/batching) or stay flat (per-op host
   work, amortizable only by fewer ops)?
4. **The keepquant repair churn**: the decompose named ~360 x 420 MiB
   transients/step in the repair web — measure their share of the
   dispatch (staging cost) vs their compute.

## The lever verdict (binds the next rows)

- **Launch-submission dominates + amortizes** -> fusion on our side:
  the multi-op programs the W4 audit pattern enables (batched dispatch,
  fewer launches) — an our-side row.
- **Per-op host work dominates** -> the operand/descriptor
  construction path (persistent descriptors, the capture discipline
  applied to dispatch) — an our-side row.
- **tt-metal-internal launch cost** -> the tt-metal lane
  (joins tt-metal#57970's escalation path with the evidence).

## Gates

- Read-only instruments; the served-replay test stays green with knobs
  on (byte-identity, the #3327 gate).
- The per-op phases sum to the step's dispatch wall (the sum-check,
  +-10%).
- Reproducible: two legs within noise; the 4-token anchor TPOT
  re-confirms.

## Non-goals

- No optimization lands on this branch; the levers become their own
  rows. No eager-numerics change.

## Stop conditions

- The instruments perturb the measurement (the W1 stop condition).
- The op dispatch resists bracketing (report the closest
  decomposition with limits).
