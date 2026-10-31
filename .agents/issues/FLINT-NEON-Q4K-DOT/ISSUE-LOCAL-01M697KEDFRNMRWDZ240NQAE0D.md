ID: ISSUE-LOCAL-01M697KEDFRNMRWDZ240NQAE0D
Title: The paged kernel's Q4_K->Q8_K integer vec_dot runs a scalar generic port at m=1 decode -- ~55% of the routed phase
Row: FLINT-NEON-Q4K-DOT
State: OPEN
Kind: perf
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-31
Updated: 2026-10-31
Closed: -

## Problem

FLINT-SLIDING-ATTN's evidence: with pool matched and attention fixed, ~55% of the matched-pool routed-expert phase is SCALAR dots -- the Q4_K->Q8_K quantize-once-integer-dot class runs the generic scalar vec_dot at m=1 decode while the attention kernel got its NEON lane. Deliverable per spec: a NEON vec_dot_q4_K_q8_K BIT-IDENTICAL to the scalar reference (integer dot has no reordering -- divergence = wrong port, STOP), wired behind the house gating pattern, payoff measured on the FLINT-KQ-CPU-PERF harness + the keep-quant serving arm (token gate must NOT move -- bit-identity is itself the check). Spec: .agents/specs/flint-neon-q4k-dot.md

## Resolution

-
