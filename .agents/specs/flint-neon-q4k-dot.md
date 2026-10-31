# SPEC — `FLINT-NEON-Q4K-DOT` — a NEON vec_dot_q4_K_q8_K for the paged kernel's scalar m=1 decode path

- Owning row: `FLINT-NEON-Q4K-DOT`
- Kind: performance (kernel port; the named lever inside the routed-expert phase)
- Issue: created in the same flow (row-scoped local issue)

## The measured gap

FLINT-SLIDING-ATTN's evidence: with the pool matched and the attention fix
landed, ~55% of the matched-pool routed-expert phase is SCALAR dots — the
Q4_K→Q8_K integer `vec_dot` runs the generic scalar port at m=1 decode
(`cpu_quant_gemm.cpp` / the Q4_K→Q8_K quantize-once-integer-dot class),
while the attention kernel got its NEON lane. The routed loop is 33% of
the serving step; a 4-8x dot speedup on the scalar share is the largest
single remaining CPU lever.

## Deliverable

A NEON `vec_dot_q4_K_q8_K` (the standard block-dot: 8-byte q4 nibble
unpacking + `vdotq_u32`-class integer reduction per 32-value sub-block,
`Q4_K` super-block scale application — match the scalar reference's
EXACT arithmetic, including rounding and scale ordering) wired into the
Q4_K→Q8_K dispatch path for aarch64, behind the same runtime gating
pattern as the existing paged-attn NEON lanes (opt-in or auto by
`__aarch64__` + feature check, per the file's conventions).

## The contract that gates everything

**Bit-identity with the scalar `vec_dot_q4_K_q8_K` on every input.**
The integer dot has no floating-point reordering: a correct NEON port is
BIT-IDENTICAL to the scalar reference by construction, and the row's
gate is exactly that — exhaustive-ish randomized + adversarial block
tests (scale extremes, nibble patterns, zero blocks, the kernel's own
test corpus) comparing NEON vs scalar bit-for-bit. Any divergence = a
wrong port, STOP. This is the same discipline as the paged-attn NEON
lanes (their gates exist; follow the pattern in
`test_ops_paged_attn_neon.cpp` and the quant-gemm suites).

## Payoff measurement

The routed-phase micro harness (FLINT-KQ-CPU-PERF's
`tools/bench/flint_kq_moe_class_ab.cpp`, rot-guarded) re-runs with the
NEON dot: the scalar-dot share (~55% of the matched-pool routed phase)
should drop materially. Serving: the keep-quant paged arm re-measured at
16/32 threads (bench zone, idle-gated, 3-rep medians, claims file,
governor logged) vs the 7.01/6.83 records. The token gate re-pins ONLY
if tokens move — a bit-identical dot must not move them (that is itself
the check).

## Non-goals

- No new quant format. No bf16/f32 path change. No TT path. No
  batch>1-specific tuning (the m=1 decode shape is the target).

## Gates

Bit-identity suite (new, house pattern), the existing quant-gemm +
paged-attn suites, test_flint_paged, test_flint_gguf serving gate,
check-agent-record, style, trailers, preflight (named host dotnet
failure).

## Git integration

One pull request, `row/FLINT-NEON-Q4K-DOT`, spec committed first.
