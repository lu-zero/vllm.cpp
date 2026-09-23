# Spec: decompose the 76.5 ms single-GEMV call on P150

Row: `BACKEND-TENSTORRENT`. State: DRAFT (2026-09-23).
Issue: `ISSUE-LOCAL-01M37WZFEQYNWST8MEADGVGWA3`.
Git integration: one pull request (spec + instrumentation + measurement),
branch `row/TT-GEMV-OVERHEAD-TRACE`.

## Problem

The APEX-I-Nano 27B decode on the Tenstorrent P150 runs ~30–60x off the
native floor (default TPOT 1253 ms, INT8DOT 642 ms, native ~50 tok/s), and
the record localizes the gap to per-call overhead: one M=1 BFP8 GEMV call
measures **76.5 ms**. Compute at this shape should be sub-millisecond.
The W7 staging-write elimination already inverted once — counters
byte-identical, wall unchanged — so the next step is a measured
decomposition, not another elimination attempt.

## Scope

W1 — attribute, don't optimize:

1. **Phase timestamps** around the existing single-GEMV call path in the
   TT keepquant/affine dispatch: (a) host-side operand/descriptor
   construction, (b) launch submission (enqueue to return), (c) completion
   wait. Opt-in env (`VT_TT_GEMV_PHASE_TRACE=1`), zero-cost when unset,
   stderr per-call lines in the same style as the existing TT trace knobs.
2. **Amortization probe**: an opt-in microbench (`TT_GEMV_BENCH=1`, the
   `TT_GDN_BENCH` pattern) firing N=100 identical GEMV calls back-to-back,
   reporting per-call wall for the first, median and steady-state call, plus
   the state-traffic counters the GDN bench already uses.

Deliverable: a dated benchmark-record section attributing the 76.5 ms across
the three phases, with the amortization curve, on the APEX decode shape
(Q4_0/IQ-mix BFP8 arm), file mutex held, exact build/recipe recorded.

## Interpretation (binds the follow-up, not this change)

- **Steady-state per-call collapses toward compute** → launch/submission
  latency dominates; the win is amortized submission — trace capture (the
  27B capture row) or multi-op programs. This W1 quantifies how much of the
  capture row's win is available eager and how much needs capture.
- **First-call ≈ steady-state, host-construction phase dominates** → the
  cost is per-call host work capture does NOT remove (captured replay still
  builds host operands unless host-free); the GEMV path becomes the primary
  follow-up (persistent operand rings, descriptor reuse) instead of a
  capture footnote.
- **Completion wait dominates** → device serialization; queue-depth/async
  submission is the follow-up, and both prior interpretations shrink.

## Gates

- Zero-cost when unset: the hot path with the env unset is byte-identical
  (the trace knob is a static-local branch, `VT_TT_SLOT_TRACE` pattern).
- The bench and trace are read-only instruments: no numerics change, the
  e2e APEX anchor legs produce byte-identical tokens with and without the
  knobs set.
- Numbers are reproducible: two legs, same binary, file mutex held, deltas
  within noise; contention state recorded.

## Risks

- The W7 precedent: this spec's whole value is that it MEASURES before
  proposing. No elimination lands on this branch even if the attribution
  looks obvious.
- Profiler availability: our tt-metal-new build has the profiler disabled;
  the instrumentation is host-side timestamps and device-side cycle counters
  the runtime already exposes, not a profiler rebuild.

## Non-goals

- No performance change, no kernel change, no capture work. This branch only
  instruments and measures.
- The 27B capture row and the INT8DOT flip keep their own specs; W1's output
  feeds their sequencing, nothing more.

## Stop conditions

- If the instrumentation itself perturbs the measurement (trace-on legs
  change token output or wall vs trace-off beyond noise), stop and redesign.
- If the amortization probe cannot run at the decode shape without the full
  model resident, fall back to a synthetic weight-tensor leg and say so in
  the record — do not extrapolate from a different shape silently.
