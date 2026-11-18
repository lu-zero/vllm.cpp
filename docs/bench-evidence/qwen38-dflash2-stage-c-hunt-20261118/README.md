# QWEN38-DFLASH2-STAGE-C hunt continuation — the rebuilt W23 lens instrument, the corruption map, and the BFP8 re-stage repair (2026-11-18)

Row: `QWEN38-DFLASH2-STAGE-C`, issue
`ISSUE-LOCAL-01M7FY5W1A91XHV1PZDGZEKR7H`. Base: `row/QWEN38-DFLASH2-STAGE-C`
(a486aaa83, the W22-W24 consolidated hunt record), this wave's work on
`row/QWEN38-DFLASH2-STAGE-C-HUNT` in worktree `/tmp/tt-hunt`, build
`/tmp/build-stagec-w3` (durable-pin recipe: `-DVLLM_CPP_TENSTORRENT=ON
-DCMAKE_PREFIX_PATH=$HOME/Sources/tt/tt-metal-pin-durable/{lib64,share}/cmake
-DCMAKE_CXX_FLAGS="-isystem .../topk_large_indices/device"`).

## Context

The prior hunt wave (W23) was killed by a host reboot that wiped `/tmp` — its
uncommitted lens-arbitration instrument and the `/tmp/build-q6k-config.log`
build recipe were lost. This wave rebuilt the instrument from the committed
consolidated findings and re-ran the arbitration.

## The rebuilt lens instrument (committed: `run-w25-lens.sh`, `decode_bfp8_lens.py`)

`[BFP8-LENS]` (env-gated: `VT_TT_BFP8_LENS=<dir>`, `VT_TT_BFP8_LENS_MAX`,
default 64 calls) dumps, at every M == 9 BFP8-staged matmul call: the staged
weight's RAW device bytes (`tt::tt_metal::detail::ReadFromBuffer` at the pin —
the bytes as DRAM holds them), the host bf16 source, the activation and output
raw bytes, the output's metadata decode, and (twice per run) the weight's
metadata decode for decoder verification. Files live OUTSIDE the tree. The
offline decoder decodes the raw bytes under BOTH layouts — plain row-major BFP8
(the W23 reading) and the pin's BFP8_B face-packed tile order
(`tt_metal/impl/data_format/bfloat8.cpp pack_as_bfp8_tiles` /
`blockfloat_common.cpp convert_u32_to_bfp`, read at the pin) — and builds the
per-tile corruption map against the metadata decode and the host source.

Decoder facts established (and required to read every prior raw-bytes claim):
- BFP8_B tile = 32x32, 1088 B: 64-byte exponent region (16 LE u32 words; word
  `w>>4`, byte `(w>>2)&3` holds the shared biased exponent of face row
  `(w>>2)/4`... i.e. one exponent per 16-datum face row, 4 rows per exp word)
  + 1024 mantissa bytes (word `tr*128 + tc*64 + i*4 + j/4`, lane `j%4`).
- Mantissa byte = sign<<7 | 7-bit normalized fraction; the unpack
  re-normalizes (shift up to bit 6, clear the hidden bit, decrement the shared
  exponent by the shift, flush exp-zero tiles to all-zero).
- VERIFIED: the staged weight's raw bytes decode **100% identical** to the
  metadata readback under this layout (`wraw_sq040_N6144.raw` vs
  `wmeta_sq040_N6144.f32`, 1.0 exact match).

## THE CORRUPTION MAP (the failing call: the 7th mixed qkvz GEMM, N=10240)

Reproduced byte-identically: the failing call is sq=48 in the lens sequence —
the 7th mixed qkvz GEMM of the first spec step — output `c_asum=3.46266e+40,
inf=108` (the EXACT W22 signature; the z GEMM on the same h at the same call:
`21911.1`, the EXACT W22 value). `corruption-map-sq48-N10240.csv`:

- **Bad fraction: 1.47e-4 = 0.015%** — the W23 fraction, reproduced exactly.
- **Bad region: exactly 8 tiles** of 51200 — tile (0,0) through (0,7):
  logical rows 0..31, cols 0..255 of the weight.
- **What the corruption is: the 512 B of BFP8 EXPONENT HEADERS** (8 x 64 B at
  tile stride 1088) are **zero** at dump time; the mantissa bytes are intact
  (`dec_is_all_zero=True` on every bad tile; the tiles' data regions are
  ~94% non-zero bytes).
- The same zero-exp-header signature appears on `wraw_sq055_N6144.raw` and
  `wraw_sq061_N6144.raw` (the z-GEMM weights of the later steps); the earlier
  and other-step dumps (`sq030`, `sq040`, `sq049`, `sq060`) have intact exp
  headers — the clobbering is a RUNTIME writer landing on whichever buffer
  holds the staged weight's block, not a staging-time defect.
- The output garbage is confined to the first column tile (cols 0..31, 29 Infs,
  bf16 exp-31-dominated patterns — values 2^126..2^128): exactly the output
  columns the zeroed-exp weight rows feed, at exactly the magnitude a matmul
  consuming foreign exp bytes near the bf16 max produces. The activation h is
  finite and clean at the same call (max |a| = 1272).

## THE VERDICT (options (a)/(b)/(c) of the dispatched task)

**(b): a genuine partial-write corruption — a runtime writer clobbers the
first 8 tiles' 512 B of exponent headers in the staged BFP8 weight's device
buffer.** NOT (a): our staging emits the pin's own face-packed layout (the
100% raw-vs-metadata decode match proves it; using `ttnn::typecast` means the
pin's own device packer wrote the bytes). NOT (c): no metadata/actual-layout
mismatch exists at the pin. AND the freshest W23 finding resolves as a
**decoder artifact**: the raw bytes of a BFP8_B tile tensor ARE face-packed;
decoding them as plain row-major BFP8 produces 1e38-scale garbage by
construction. The W23 "h padding rows full of foreign data" and "F32-formatted
bytes in a bf16 buffer" observations are the same clobbering class seen
through the un-normalized decoder.

## THE FIX (ours, minimal — no re-architecture)

`src/vt/tenstorrent/tenstorrent_residency.cpp` `Bfp8RestageWeight` +
`src/vt/tenstorrent/tenstorrent_ops.cpp` (MatmulBTKernel, the `M == 9` arm):
at the spec geometry, when the weight is BFP8-resident, re-run the cold
staging unconditionally — `ttnn::typecast` into a FRESH allocation (the
W23-verified fact: a fresh re-stage is clean every time) and install it as the
weight's shadow. `VT_TT_BFP8_RESTAGE=0` rolls back to the unrepaired arm on
the same binary. Cost: one bf16->BFP8 device typecast per spec-geometry weight
per call (7 weights per verify step, ~ms each); the device memory footprint is
unchanged (the lever's viability is intact: ~1.55 GiB vs the 3.3 GiB bf16
staging that OOMs at KV 32 — W3/W4 record).

## Engine repro (the focused failing call, instrumented leg `w25-lens`)

- Pre-fix: sq48 `c_asum=3.46266e+40 inf=108`, stream degenerate `e171d6b1`.
- Post-fix: sq48 `c_asum=32186.3, nan=0`; **all 64 lens calls finite, 0 NaN,
  0 Inf, evolving sums** (`w25-lens.stderr.log` of the second leg); `[CONV-SPEC]`
  / `[QKVZ-IN]` instruments clean at the first call. Stream
  `913e637c...`: non-degenerate, structured tokens — with the lens
  instrumentation still attached and the W20 keep-alive instrument on (both
  perturb exact bytes per the W21 record), so identity is judged on the
  probe-free gate legs below.

## Gates

`run-w25-gate.sh` (probe-free ON legs with the keep-alive instrument OFF — it
moved the first emitted token 220->23066 in the W20 record):
- **(b) Q4_K_M KV64 drafter-OFF x3: PASS** — `d49c53d2...` byte-identical
  3/3 (`w25-off-{1,2,3}`); TPOT 6646.65 / 6648.11 / 6648.95 ms vs the
  recorded 6607 (+0.6%, the established rerun band). The repair is inert off
  the spec path.
- **(c) UD-Q6_K_XL KV8 drafter-OFF x3: PASS** — `b36bc61a...` byte-identical
  3/3; TPOT 5349.75 / 5353.04 / 5351.08 ms vs the recorded 5340.9 (+0.2%).
  drafter-ON x1: the standing W3/W4 capacity blocker, unchanged
  (`Cannot auto-fit max_model_len`).
- **(a) Q4_K_M KV64 drafter-ON x3: THE TOKEN-IDENTITY GATE FAILS — STOP, as
  the task's stop condition names.** 3/3 legs byte-identical to each other
  (`af77d5d3...` — deterministic), non-degenerate, and 22 of 24 tokens equal
  the recorded stream; the single divergence is a 3-token shift at index 7
  (the ON stream emits `1814,220,16,23066` where the OFF stream has
  `220,16,220,16` — one `[220,16,23066]` cycle early, then re-aligned).
  Mean TPOT 26975 ms (all three legs) vs the OFF 6647 — the unconditional
  re-stage pays ~18.5 s/token (7 weights x re-upload + typecast per verify
  step). THE ACCEPTANCE MULTIPLE remains UNMEASURABLE at this state: the
  identity fails and the cost is not a number a comparison can use.
- The focused w20-bfp8 isolated arms with the repair: PASS (`w25-bfp8-arms.log`)
  — cold M=9 max_rel 0.75% / M=32 0.77% (0 non-finite), warm 200/200 clean,
  unsynced-staging 0/50 bad.
- Focused: the w20-bfp8 isolated arms (spec-geometry cold / warm x200 /
  unsynced-staging x50) with the repair — `w25-bfp8-arms.log`.
- `check-agent-record.py`, `--staged` preflight detached: see the commit
  message.

The raw dumps stay OUT of the tree (`/tmp/tt-lens/dumps`, ~10 GB, wiped with
/tmp); this directory carries the decoder, the corruption map, and the
arbitration summary only.

## Where this leaves the row (the STOP state)

The degeneracy `[220, 0, 0, ...]` is DEAD: the corruption map convicted the
runtime clobbering of the staged BFP8 weight's exponent headers and the repair
removed the poison (all lens calls clean; a coherent token stream). Two facts
remain before the acceptance multiple is measurable:

1. **The residual identity divergence** (3-token shift at index 7,
   deterministic across 3 legs) — one spec step commits differently from the
   eager OFF stream. The next instrument is the verify-side acceptance trace
   at the step that spans emitted tokens 5..10 (the draft proposal vs the
   target greedy tokens, and the commit length), against the OFF stream.
2. **The repair's cost** (18.5 s/token) — the unconditional re-stage must
   become a detect-and-repair: store the first-8-tile reference bytes at
   staging, read the device head once per call (or throttled per step), and
   re-stage only the clobbered weights. The map's 512 B region is the check.
