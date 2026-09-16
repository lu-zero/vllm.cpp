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

## SigmoidGateBf16 exonerated as the CAUSE (2026-09-16, probe + fix attempt)

- Probe: the gate's operands are LIVE at the first full-attention layer and
  ALREADY ZERO at the next one — the zeros arrive from upstream, the serve
  path faithfully serves them.
- A doctrine-aligned fix of the serve path's eager reshape (ROW_MAJOR
  round-trip) changed nothing — reverted as unproven. SigmoidGateBf16 is a
  victim, not the cause.
- Refined timeline: full-attention layers are RARE in this hybrid (the op
  trace shows one AttnQkNormRopeGate+SigmoidGateBf16 sequence among GDN
  blocks). The FIRST zero anywhere in the stream is the KQACT call 10
  (block 3's 17x5120 GDN in-proj activation). Between KQACT call 9 (block 2
  down-proj input, live) and call 10 sit: block 2's down-proj (grouped arm),
  its commit, the residual add, and block 3's pre-GDN RmsNorm.
- Next probes: checksum the down-proj OUTPUT commit (out stats behind the
  probe in MatmulBTQuantGroupedKernel), and find the residual-add op (no Add
  op appears in the TT op trace — it may run on device through a fused op,
  on host, or inside RmsNorm; whichever it is, it consumed zeros or
  committed zeros).

## Grouped outputs LIVE; residual add located (2026-09-16, final probes)

- All probed grouped down-proj outputs are LIVE pre-commit
  ([TT-KQOUT] nz=full on every call incl. the 17x5120 hidden-stream shapes).
  The keep-quant arm commits live data through the death window.
- The residual add has NO op of its own: it lives INSIDE RmsNormKernel —
  the TT RmsNorm takes a residual parameter (residual += x; out =
  rms_norm(residual)), reached from the model's block boundary and via
  FusedChainKernel (tenstorrent_ops.cpp:4188-4212, which dispatches
  kFusedAddRmsNormStd to the same RmsNormKernel call). That is why no Add
  appears in the TT op trace.
- Next probe: checksum x and *residual inside RmsNormKernel when residual
  is non-null (VT_DEBUG_SAMPLED=2), at the block 2->3 boundary. The
  suspects are the ttnn add at the [17,5120] non-tile-aligned M (TILE pads
  M to 32), or the residual commit/download geometry after the add.

## Residual add exonerated; corruption narrowed to rms_norm output/commit (2026-09-16)

- [TT-RESADD] probe: every residual add's operands AND sum are LIVE through
  the whole run (387 probe lines), yet the KQACT call 10 (the next RmsNorm's
  output feeding the GDN in-proj) still reads zero and the logits stay zero.
- The corruption is therefore INSIDE RmsNormKernel's device path AFTER the
  residual add: either the ttnn::rms_norm output is zero (at [17,5120]
  non-tile-aligned M?), or the CommitDevice2D of `out` records a wrong/empty
  device tensor so the consumer's EnsureHost downloads zeros.
- The same norm consumed a LIVE add output (probed), so the input side is
  clean.
- NEXT (first thing): checksum the rms_norm device output in RmsNormKernel
  before CommitDevice2D, behind VT_DEBUG_SAMPLED=2. Zero output => ttnn
  rms_norm at this shape (compare the captured 0.8B arm, which pins rms_norm
  at other shapes); live output => CommitDevice2D/slot — diff what
  CommitDevice2D stores vs what EnsureHost downloads for that slot.

## rms_norm outputs LIVE; the death is in the model plumbing between (2026-09-16)

- [TT-NORMOUT] probe: EVERY rms_norm device output is live pre-commit
  (nz~87040/87040, all norms, whole run). ttnn::rms_norm is fine at
  [17,5120].
- DECISIVE: the consumer's activation pointer (0xfff0c811fdc0 in the
  KQACT runs) is a DIFFERENT TENSOR than the norm's out slot (0xffef9011fdc0
  in the NORMOUT runs). There is an intermediate copy/view between the norm
  output and the matmul input, and THAT copy produces the zeros.
- Next: run with VT_TT_SLOT_TRACE=1 + VT_DEBUG_SAMPLED=2 and correlate the
  act tensor's pointer to its producer events (register/ensure/commit) to
  find the copy op; then read the model plumbing (qwen3_5.cpp) for how the
  norm output reaches the matmul (a DBuf/ view/ CastBf16?) and probe it.
  The drift signature (values differ run to run) says the copy reads
  unallocated/reused memory.
