# QWEN38-DFLASH2-STAGE-C W28 — the writer root-cause hunt (2026-11-19)

Row: `QWEN38-DFLASH2-STAGE-C`, issue
`ISSUE-LOCAL-01M7FY5W1A91XHV1PZDGZEKR7H`. Worktree `/tmp/tt-writer` (fresh,
branch `row/QWEN38-DFLASH2-STAGE-C-HUNT-W`, base `7b27d375d`), build
`/tmp/build-stagec-w28` (the durable-pin recipe). The W20-W27 trail lives in
the issue and in `../qwen38-dflash2-stage-c-hunt-20261118/README.md`.

## The W28 canary instrument (env `VT_TT_BFP8_CANARY=1`)

At every spec-geometry (M == 9) BFP8 matmul, diagnostics only, never writes,
never repairs:

- `[W28-MICRO]` (first check): does the pin's bf16->BFLOAT8_B typecast write
  the tile exponent headers of its output? A fresh-allocation control plus a
  0xFF-prefilled-target arm on a [64,256] (8-tile) weight.
- `[W28-STAGE]`: the direct zero-header scan of the typecast output at stage
  time (first 400 stagings).
- `[W28-CANARY] #N pre= post= probe_zero= verdict=`: the direct (slice-free)
  full-buffer header scan BEFORE and AFTER the DNR probe chain, plus the
  probe's own verdict — this splits "the weight arrived zeroed" from "the
  probe chain wrote it", and the probe-vs-direct disagreement exposes a
  slice-read artifact.
- The DRAM block table (GetMemoryView) per check: any OTHER allocated block
  intersecting the weight's per-bank range is a double allocation.

## Leg w28-canary-1 (Q4_K_M KV64 drafter-ON, VT_TT_BFP8_RESTAGE=0, 12 tokens)

Facts (this directory, `w28-canary-1.stderr.log`):

- **The probe chain is exonerated.** All 64 checks: `post == pre` — the
  `ttnn::slice`-based Bfp8HeaderProbe does NOT write the weight. (It also
  UNDERREPORTS: at the zeroed checks the probe saw 1/8 zero headers while the
  direct read saw 8/8 — the slice output's header handling is broken at the
  pin, which explains why every earlier slice-mediated "clean" reading could
  not be trusted.)
- **The writer's true footprint is exactly the first 8 tile headers.** The
  direct full-buffer scan counts 8/30720 zero headers at the struck weights —
  the first 8 pages = the first page of EACH of the 8 DRAM banks; mantissas
  intact everywhere.
- **No allocator overlap.** `blocks=1` on every check — no double allocation,
  the allocator never handed the live weight's range to another buffer.
- **Deterministic sequence position, per-call fresh staging.** The struck
  checks are the same call indexes across legs (#52, #58, #61 in both legs);
  every matmul call stages a FRESH buffer (the shadow cache is keyed by the
  host pointer, which MOVES per call), so the struck buffer is fresh DRAM,
  staged clean-in-principle seconds before the canary read.
- **Verdict `ARRIVED-ZEROED`** — the zeros are already in DRAM at the first
  canary read of the fresh buffer, BEFORE the matmul consumes it.

## Leg 3 additions (the typecast and the staging are exonerated)

- **[W28-MICRO]** `fresh_zero=0 ff_zero_before=0 into_zero_after=0`: the pin's
  bf16->BFLOAT8_B typecast writes the exponent headers fully and correctly,
  even into a 0xFF-prefilled target — the typecast is NOT the writer.
- **[W28-STAGE]**: 185 stage-time checks, `zero_headers=0` on ALL — every
  fresh staging reads CLEAN at a blocking read right after the typecast,
  INCLUDING the stages whose weight is struck later in the same call.
- Together with the `ARRIVED-ZEROED` canary verdicts (#52/#58/#61,
  deterministic across all three legs): the writer's bytes LAND between two
  blocking reads of the same fresh buffer — a write whose target address was
  bound to this DRAM block while it was free, executing after the staging's
  own writes. The staging never overlaps (allocator block table clean), so
  the writer holds a STALE address: **class (a) — a freed-but-still-referenced
  block: some tensor is freed while its write is still pending, the staging
  allocation reuses the block, and the pending write lands in the live
  staged weight.** This also explains the W20 keep-alive intervention MOVING
  the poison (delaying reuse past the writer's flight time shifts which
  buffer the stale write strikes) and the moving onset across waves.
- The struck weights in every leg are the 33423360 B (N=10240, K=3072)
  mixed-qkvz GEMM weights at the same call indexes.

## Leg 5 additions (the trace-replay hypothesis falsified; the write-set proven)

- **[W28-DIFF]** at each struck weight: re-typecast the same host source into
  a fresh buffer and diff the two device buffers byte-for-byte:
  `diff=512 hdr=512 data=0 diff_zero=512 range=[0x0,0x1dff]` on all three
  strikes (#52/#58/#61, three legs in a row). The writer writes EXACTLY the
  8 tile exponent headers, exactly 512 zero bytes, NOTHING else — every
  mantissa byte is identical to a fresh deterministic typecast. (The diff
  also proves the typecast is byte-deterministic run-to-run.)
- **The trace-replay hypothesis is FALSIFIED**: this bench leg runs fully
  eager (no `TraceEndCapture` ever fires — the `[W28-CAPTURE]` snapshot line
  is absent and `capture_hit=0`), yet the strikes reproduce byte-identically.
- Combined with "clean at the stage-time blocking read, zeroed at the
  matmul-arm blocking read, nothing enqueued in between (single CQ)": the
  write executes OUT OF BAND of the command queue — a host-side write path
  or an internal tt-metal worker writing at a stale (freed-and-recycled)
  address. Class (a) stands; the write-path attribution (which host/worker
  write of 512 B of zeroed BFP8_B headers, whose target block the staging
  recycled) is the named next instrument: wrap the tt-metal host write /
  upload paths (WriteToBuffer, EnqueueWriteBuffer, the dispatcher's write
  commands) with address logs and dump the log slice covering the strike
  window (the canary's strike calls are deterministic).

## Where the hunt stands after W28

Proven (all instrument-first, three reproducible legs):
1. The writer's write-set is EXACTLY 512 B of zeroed BFP8_B tile exponent
   headers at the first 8 tiles (the first page of each of the 8 DRAM
   banks) of the staged weight.
2. It lands AFTER a clean stage-time read and BEFORE the matmul consumes
   the weight, out of band of the command queue.
3. Exonerated with direct evidence: our staging and the typecast
   ([W28-MICRO], [W28-STAGE]), the slice probe chain (`post == pre` on 192
   checks across 3 legs), the allocator (no block-table overlap, `blocks=1`
   on every check), and the trace replay (no capture exists in the leg).
4. The slice-based probe UNDERREPORTS corruption (1/8 vs the direct read's
   8/8) — every earlier slice-mediated "clean/corrupt" reading needs the
   direct-read caveat.
5. NOT exonerated: an internal tt-metal host/worker write path holding a
   stale address across a free-and-reuse. The DNR repair is treating this
   writer's damage every step (the ~89-weight clobber set); stopping the
   writer needs its identity.

The gate suite (corruption-absent leg, token identity, the acceptance
multiple, drafter-OFF byte identity) is NOT RUN at this state: no fix has
landed, because the writer's op identity — and therefore the minimal fix —
is the one datum still missing.

## What this rules out

- NOT (a) the allocator double-bookkeeping: block table clean, no overlap.
- NOT a slice-read artifact: the slice-free direct read agrees (8/8).
- NOT the DNR/probe chain: `post == pre` on all 64 checks.
- NOT the drafter's staging: this leg ran drafter-ON, DNR OFF, and the struck
  weights are the TARGET's — consistent with the W27 trunk-off-device fact.
