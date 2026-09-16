ID: ISSUE-LOCAL-01M2NSDATJQ1YNW1PA9ZBMAAM5
Title: APEX/TT: keep-quant E=1 chunked arm emits all-zero logits at lm_head decode shapes
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-16
Updated: 2026-09-16
Closed: -

## Problem

The token-correctness gate (ISSUE-LOCAL-01M2NMSNSS6R8T0EW1WYHHVA0Z) caught: the E=1 chunked keep-quant matmul arm (MatmulBTQuantGroupedKernel, production dense path) commits all-zero logits at the 27B lm_head [1,248320] Q6_K shape (4 commits tracked=1, EnsureHost downloads them, argmax = id 0 every step, VT_DEBUG_SAMPLED tok=0 x4). Every sampled token of the 27B e2e is 0. Suspects: ttnn matmul with M in {1,2} against the chunked [chunk,K] bf16 tiles returning zeros, the partials concat, or a TTReclaimPlanes use-after-free inside the chunk loop (a reclaimed plane still aliased). The 0.8B vehicle never hit this because its bf16 head routes through MatmulBf16LogitsF32D (dense non-quant matmul), so the quant-head e2e path was never validated until now. Needs: a red unit test at [1,K]x[248320,5120] Q6_K (or scaled-down V) vs the CPU oracle, then the fix.

## Resolution

-

## CORRECTION (2026-09-16, later same day)

The nb=20 "garbage" was a TEST BUG: the discriminator allocated packed(N*2)
blocks while declaring the tensor {N, K=5120} (1280 blocks) — an out-of-bounds
read, whose drifting 1e9 values are exactly the uninit-memory signature. With
the allocation fixed, the chunked arm is CORRECT at unit scale across chunk
overrides 4 and 20, M in {1,2}, nb in {2,20} (worst_abs 1354, in envelope),
and the decode is bit-exact at rows=4/nb=20 (B=80). The keep-quant kernel and
the TTReclaimPlanes chunk loop are exonerated at unit scale.

The e2e all-zero logits stand (VT_DEBUG_SAMPLED tok=0, committed device tensor
zero). Remaining repro axes: production scale (N=248320, 76 chunks of 3276
rows, B=65520 decode, 76-way concat) or the engine context around the head
(DBuf pooled logits, ResidentWeight staging of the real lm_head pointer, the
real activation source — a zero hidden state would also produce zero logits
through a correct kernel).

## Original narrowed-repro section (SUPERSEDED by the correction above)

Unit-scale discriminator (`-tc="*lm_head decode shapes*"`):
- M=1/2, N=64, nb=2 (K=512), chunk override 4 (16 chunks): CORRECT (worst_abs 239, within envelope), stable across 5 repeated calls.
- M=1/2, N=64, nb=20 (K=5120), chunk override 4: **GARBAGE** — worst_abs 5.7e9 / 1.2e10, values drift run to run (uninitialized/reused memory signature).
- M=1, N=64, nb=20, SINGLE chunk (override 0): CORRECT magnitude (worst_abs 1031, K=5120 bf16 dot).

So the defect is the MULTI-CHUNK path at nb=20, not the K=5120 matmul itself, not repeated calls, not M. Per chunk: B = chunk_rows x nb = 80 words; the single-chunk case takes the sl_alias branch (slice returns the whole tensor) while multi-chunk slices + TTReclaimPlanes({&sl,&wf,&wbf}). Prime suspect: a plane reclaimed while still aliased in the chunk loop, or the slice of the word shadow at row offsets c0*nb.

## Engine attribution (2026-09-16, debug hooks VT_DEBUG_SAMPLED=2 / VT_TT_TRACE_DEBUG)

- The logits are EXACTLY 0.0 across all 248320 entries at every step
  ([TT-LOGITS] nz=0/248320) — the signature of a zero activation OR a zero
  weight, not numeric corruption.
- Activations entering quant matmuls are LIVE for the first ~9 calls
  (through block 2) and ZERO from block 3 on. The death point in the op
  timeline: block 3's MoE down-proj enters MatmulBTQuant and never reaches
  MatmulBTQuantGrouped — it dispatched to the int8-dot kernel (Q3_K is
  served unconditionally by int8-dot at tenstorrent_ops.cpp:2862).
- BUT forcing VT_TT_KEEPQUANT_INT8DOT=1 (all encodings through int8-dot)
  ALSO yields all-zero logits. Both arms fail in-engine; the grouped arm is
  proven correct at unit AND production scale (N=32760, chunk=3276,
  B=65520, 10-way concat: sane, worst_abs 2253).
- Slot trace on the 17x5120 hidden slot: committed 4x per forward
  (dc=1), EnsureHost downloads from the right slot — first forward reads
  live bytes, later forwards read zeros. The producers write zeros.

Next suspects, in order: (1) the word-shadow STAGING of specific weights —
if a weight's staged i32 words are zeros (staged from a stale/zeroed host
master, e.g. an mmap'd GGUF tensor never read into the staging source), every
matmul through it emits exact zeros while the kernel stays correct; (2) the
engine's weight-view plumbing for this GGUF (ResidentWeight for the
lm_head/down weights); (3) the residual path around the first dead op.
Probe: checksum the staged `words` (row 0) inside the kernels behind
VT_DEBUG_SAMPLED=2, and checksum the packed host master before staging.

## Staging + kernel exonerations (2026-09-16, continued)

- Weight staging is LIVE: all 353 stagings across the artifact's encoding mix
  (iq2_s x88, iq2_xxs x44, iq3_xxs x68, q3_K x78, q4_K x74, q6_K x1 — the
  lm_head staged live at rows=248320). No zero masters, no zero word shadows.
- The int8-dot kernel is BIT-EXACT at the production block-3 shape
  (q3_K, M=17, N=5120, K=17408): 87040/87040 exact vs the CPU vec_dot oracle
  (test "int8-dot Q3_K at the 27B down-proj shape"). Both keep-quant kernels
  are now exonerated in isolation at production shapes.

## Remaining suspect: the ServeActF32 / SigmoidGateBf16 geometry path

The death sits right after the attention block (AttnQkNormRopeGate ->
CastBf16 -> ReshapeAndCache -> TryPagedAttentionDeviceDecode ->
SigmoidGateBf16). SigmoidGateBf16 serves its operands at a [1, n] geometry
through ServeActF32 (tenstorrent_ops.cpp:7607-7653), whose eager
geometry-mismatch arm calls CaptureSafeReshape — the FREE ttnn::reshape —
on a TILE shadow whose native geometry may be [17, 87040]. This is the
exact doctrine violation NormalizeDevF32Tile's comment warns about
(free reshape of a TILE at a changed shape; the #3206 review's F2 owed
item named kSigmoidGateBf16 as a hardcoded-geometry consumer predating
the doctrine). Probe: checksum dev_attn/dev_gate inside
SigmoidGateBf16Kernel behind VT_DEBUG_SAMPLED=2; read the serve path at
:7613 against the AttnQkNormRopeGate doctrine.
