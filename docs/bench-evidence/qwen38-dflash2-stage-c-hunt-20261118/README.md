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

---


Row: `QWEN38-DFLASH2-STAGE-C`, issue
`ISSUE-LOCAL-01M7FY5W1A91XHV1PZDGZEKR7H`. Base: the W25 commit `1e5e57d16`
(the corruption map + the v1 unconditional re-stage), this wave's work on
`row/QWEN38-DFLASH2-STAGE-C-HUNT` in worktree `/tmp/tt-hunt`, build
`/tmp/build-stagec-w3` (durable-pin recipe + topk `-isystem`; bench on
`taskset -c 32-63`, builds on `64-127`).

## The detect-and-repair implementation

`src/vt/tenstorrent/tenstorrent_residency.cpp`:

- `Bfp8WeightShadow` grows `header_ref` — the corruption's fingerprint region
  (8 tiles x 64 B of BFP8 exponent headers, 512 B) captured at COLD STAGING:
  `EnsureBfp8WeightDevice` and `Bfp8RestageWeight` both probe the fresh bytes
  after the typecast and store the reference against the weight's host
  pointer (the same pointer-keyed pattern as the shadow map itself).
- `Bfp8HeaderProbe` — `ttnn::slice` to the first 32 rows x min(cols, 256)
  (tile-aligned), a raw `ReadFromBuffer` of the ~8.7 KB slice, and extraction
  of the 8 x 64 B headers at tile stride 1088. KB-scale D2H per weight per
  check — no ~104 MB full readback, no ~104 MB re-upload.
- `Bfp8DetectRepair` — probes the weight's current headers, memcmp against
  `header_ref`; on ANY difference re-stages THAT weight (the cold path) and
  refreshes the reference. Clean weights: untouched. A `VT_TT_BFP8_DNR_TRACE=1`
  per-check line carries `us`, `repair`, `bad` (differing bytes), and the
  running totals.
- `src/vt/tenstorrent/tenstorrent_ops.cpp` — the MatmulBTKernel `M == 9` arm
  now calls `Bfp8DetectRepair` instead of the W25 unconditional re-stage.
  `VT_TT_BFP8_RESTAGE=0` still disables the whole mechanism on the same
  binary.

`src/vllm/v1/spec_decode/rejection_sampler.cpp` — the acceptance trace:
`TraceVerifyAcceptance`, env-gated `VT_SPEC_ACCEPT_TRACE=<N>` (the first N
spec steps), wired into `RejectionSampler::forward` (the arm the TT runner
takes, `kMainQueueDrain`). Per step it downloads the verify's target argmax
scratch and the full logits, and prints `[ACCEPT]` lines: per expanded row the
target argmax + logits top-5 (token ids via an index partial-sort — the first
draft of the trace printed sorted POSITIONS via a find-after-partial_sort; the
committed trace sorts indices), plus the draft proposal and the emitted
tokens.

## The per-step cost, measured (w26-on-1, Q4_K_M KV64 drafter-ON, 24 tokens)

- Clean checks: 988 checks over 9 spec steps, 1034 ms TOTAL — ~113 ms/step,
  ~1 ms per weight. KB-scale as designed.
- Repairs: 617 over 9 steps — **~55-65 repairs per verify step, ~11.8 s/step**
  (each repair is a full cold re-stage: ~104 MB bf16 H2D + typecast, ~180 ms).
- Mean TPOT 25578 ms vs the W25 blanket form's 26676 ms — the DNR form saves
  only ~4%, because **the runtime writer RE-CLOBBERS ~60 weights every verify
  step**: the same keys repair every step (`bad` 33..512 headers zeroed
  again), so detect-and-repair degenerates toward the blanket restage at this
  writer rate. The design's <10 ms/step target holds for the CHECKS (113
  ms/step) but the repair cost is set by the writer's re-clobber rate, not by
  the check. Recorded as the standing cost fact for the row: the acceptance
  multiple remains UNMEASURABLE as a win — ON is ~3.9x SLOWER than OFF
  (25578 vs 6647 ms TPOT).

## The acceptance trace — the 3-token shift at index 7 (verdict)

The ON stream `af77d5d3...` is byte-stable across 5 legs now (W25 x3 + W26
x2), 22/24 tokens equal the OFF stream `d49c53d2...`, one 3-token rotation at
index 7 (ON emits `23066,220,16` where OFF emits `220,16,23066` — same cycle,
rotated by one).

The `[ACCEPT]` trace at the first 4 spec steps answers the who-shifted
question:

- **The TARGET's verify rows are internally consistent and the accept walk
  commits exactly what the verify's argmax says.** At step 0 the draft
  proposes `[16, 23066, 220, ...]`; row 0's argmax = 16 (accept), row 1's
  argmax = 23066 conditioned on row 1's input (the accepted draft token 16),
  row 2's = 220, row 3's = 16; emitted = `16,23066,220,16` with 3 drafts
  accepted. The walk and the commit are FAITHFUL — this is not an accept-rule
  defect.
- **The shift enters in the TARGET's M=9 verify forward numerics, not in the
  repair.** At the identical conditioning (last committed token 16), the
  eager M=1 decode's next-token argmax is `220` (the OFF stream's index 5)
  while the verify's M=9 row argmax is `23066` — the top-2 logits sit ~0.3
  apart (`23066:16.77, 220:16.44`; several rows show top-2 gaps of 0.03-0.5),
  squarely inside the BFP8 lever's measured 0.75% max-rel error band. The
  draft (the DIRECT-ARGMAX arm — full-vocab argmax over the target's own
  residual taps, the same block-forward numerics class as the verify)
  proposes the verify-consistent continuation, the verify rows confirm it,
  and the lossless accept then commits a path that eager M=1 decode would
  not have taken. Because the workload's text is a `[220,16,23066]` cycle,
  the committed rotation re-aligns one element later.
- **The recorded OFF stream is NOT corrupt-tainted** — no re-record: the W25
  gates already ran drafter-OFF byte-identical to `d49c53d2` / `b36bc61a` on
  the v1-fixed tree, and the W26 OFF legs repeat that on the DNR tree (see
  Gates). Eager decode is unchanged by the repair; only the M=9 spec
  forward's near-ties move.

So the NEEDS_DECISION condition named by the dispatch ("the re-stage changes
the TARGET's numerics") is NOT met — the DNR-repaired weights equal the cold
staging by construction (the same typecast; the OFF identity gates prove the
clean path is unchanged). What the trace shows is narrower and real: **the
M=9 BFP8-batched verify forward's argmax disagrees with the M=1 eager decode
at near-tied logits**, and a token-identity gate defined against the eager
stream cannot pass over a lever whose weight encoding carries 0.75% max-rel
error. This is the row's STOP condition tripping on a numeric-parity gap of
the spec forward, and it is the coordinator's decision: a near-tie-aware
identity gate (tie-band in logit space), an f32/bf16 verify-forward arm for
the identity gate only, or acceptance of cycle-phase differences on periodic
workloads.

## Gates (run-w26-gate2.sh + the on legs of run-w26-gate.sh; 2026-11-19)

Discipline per leg: file mutex `exec 9>${GPU_LOCK:-$HOME/gpu.lock}`;
flock -x 9; luwen reset + 15 s + /dev/shm cleanup; idle gate (load1 < 16);
bench on `taskset -c 32-63`, FROM THE REPO ROOT; `INT8DOT=1`,
durable-libs-first, `TT_METAL_RUNTIME_ROOT=$HOME/Sources/tt/tt-metal-pin`.

(Tables filled from the leg logs; see this directory's `w26-*` leg files:
the token-ids JSON, the bench and stderr logs, the env notes, and the
`ps-top-*` snapshots.)

- **(a) Q4_K_M KV64 drafter-ON x3: deterministic, 22/24 tokens = `d49c53d2`,
  the 3-token rotation at index 7 (see the trace verdict above).**
  `af77d5d3...` x3 (w26-on-1 of run-w26-gate.sh + w26-on-1/-2 of this run);
  TPOT 25571.67 / 25566.88 / 25578.66 ms — **~3.9x SLOWER than OFF**. THE
  ACCEPTANCE MULTIPLE remains UNMEASURABLE as a win.
- **(b) UD-Q6_K_XL KV8 drafter-ON x3: the standing capacity blocker,
  unchanged from W25** — 3/3 exit 1, `Cannot auto-fit max_model_len: not
  enough GPU memory available to serve even a single token` (the KV-8 pool +
  the draft trunk). No TPOT.
- **(c) drafter-OFF byte identity: PASS 6/6, the fixed-tree records STAND**
  — Q4_K_M `d49c53d2...` x3 (TPOT 6648.04 / 6647.27 / 6647.23 vs the
  recorded 6607, +0.6% rerun band); UD-Q6_K_XL `b36bc61a...` x3 (TPOT
  5350.33 / 5351.69 / 5353.07 vs the recorded 5340.9, +0.2%). No re-record.

## Where this leaves the row

1. The detect-and-repair form WORKS (repairs fire on exactly the clobbered
   weights; clean weights pay ~1 ms) but cannot deliver the cost target while
   the writer re-clobbers ~60 weights per verify step — the repair cost is
   the re-stage cost at the writer's rate. The next lever on this axis is a
   DEVICE-SIDE master: keep the fresh BFP8 bytes resident and restore with a
   D2D copy (~ms) instead of a host re-stage (~180 ms) — a footprint trade
   (~2x the BFP8 pool, the 1.55 GiB -> ~3.1 GiB the W3 record bound) that
   needs the coordinator's call.
2. The identity gate's residual is the M=9-vs-M=1 near-tie parity gap named
   above — NEEDS_DECISION for the gate's definition, not a defect in the
   accept/commit path (proven faithful by the trace).
3. The acceptance multiple remains UNMEASURABLE as a win at this repair cost
   (ON 25578 ms vs OFF 6647 ms TPOT).

## W27 (2026-11-19): the tie-band gate + the D2D master restore (ISSUE-LOCAL-01M7FY5W1A91XHV1PZDGZEKR7H, the coordinator's two decisions, implemented)

### THE TIE-BAND GATE (decision 1, committed)

`tie-band-gate.py` (this directory, the committed gate harness; fed by the
W26 acceptance-trace instrument and the new eager gap instrument):
- **Threshold: `--near-tie-mnats 500`** — the ratified kNearTieMnats band
  (`tests/parity/test_commandr_paged_engine.cpp:52`). A divergence at
  position p is permissible only when the EAGER top-2 gap at p, p-1, or p+1
  (a one-position phase slip of a periodic cycle) sits below 500 mnats
  (0.5 nats).
- (a) per-step acceptance telemetry within the recorded band — the committed
  tokens per verify block parsed from the `[ACCEPT]` lines (recorded healthy
  band: 8 drafts/step, this run 8/8 across all traced legs);
- (b) the emitted stream matches the eager stream except at the permissible
  positions above;
- (c) zero NaN/Inf markers anywhere + no degenerate token ids (id <= 0
  mid-stream — the `[220, 0, 0, ...]` corruption signature) — hard fail.
- `--corrupted` (the red leg): a green gate on a corrupted run is itself a
  failure (the gate must separate corrupted from healthy).
- The eager per-position gaps come from `VT_EAGER_TOP2_TRACE=<N>`
  (`Sampler::greedy_argmax_host` prints `[EAGER-TOP2] step tok gap_mnats` for
  the first N greedy steps — the arm the drafter-OFF decode takes, so the
  gaps cover exactly the recorded OFF stream). Committed gap record:
  `w27-off-gaps-q4.stderr.log` (40 positions; the reference stream's first 24
  tokens are byte-identical to the recorded OFF stream `d49c53d2...`'s prefix).

Gate verdicts (committed `*-gate-verdict.json`):
- **W26-form healthy stream `af77d5d3...` (w26-on-1, the 5-leg-stable
  stream): PASS** — 2 mismatches (positions 7 and 10), both inside the band
  (eager gaps 118.6 and 477.2 mnats); acceptance 8/step in [7,9]. The gate
  CAN be green on a healthy lever run.
- **The W27 D2D build's stream `77688b0a...` (w27-on-1/2/3, 3/3
  deterministic): FAIL** — 13 mismatches, 4 outside the band, first at 20
  (eager gap 1107 mnats; positions 20-23 carry eager gaps 952-3236 mnats,
  far beyond near-tie). The gate correctly refuses a run whose divergence is
  NOT explainable by near-ties — the poison class.
- **The corrupted-weights RED leg (decision: gate must fail): FAIL as
  required** — `w27-on-corr`: the W20 corruption PLANTED on demand
  (`VT_TT_BFP8_PLANT_HEADERS_ZERO=1`, `Bfp8PlantHeaderZero` zeroes the 8 tile
  exponent headers of the working buffer, the corruption map's exact region)
  with the DNR mechanism disabled (`VT_TT_BFP8_RESTAGE=0`) reproduced the
  EXACT W4/W20 signature `[220, 0, 0, ...]`; the gate fails on (c) 23
  degenerate token ids and the band. **The gate separates corrupted from
  healthy at the recorded thresholds: proven both directions.**

### THE D2D MASTER RESTORE (decision 2, implemented; target NOT reached)

`Bfp8WeightShadow.master` — a second on-device BFP8 copy per admitted
weight. Policy (all measured, recorded in the code comments):
- LAZY per-CLOBBERED weight, admitted at the SECOND clobber (the writer's
  repeating footprint self-selects), cap **64 masters**
  (`kBfp8MasterCap`).
- The repair is a FRESH-ALLOCATION D2D clone
  (`ttnn::assign` with no output tensor = a pure same-dtype device copy of
  the master, ~2 ms) installed as the shadow's served tensor — the W25/W26
  fresh-allocation form is load-bearing: an in-place restore variant
  measured 755 repairs/run (vs 617 in W26) and diverged past the band.
- Weights without a master (cap-exhausted or allocation failure) keep the
  W26 host re-stage fallback; the ledger counts both transports.

Per-step ledger (w27-on-1/2/3, Q4_K_M KV64 drafter-ON, 24 tokens, 9 verify
steps each):
- repairs 806-811 per run = **~90/step** (the repeating writer footprint is
  ~89 weights, not the W26 estimate of ~55-65); master-covered ~43/step ride
  the D2D lane at **~2 ms each** (the <50 ms/weight target is met by 25x);
  ~46/step stay on the host re-stage at ~180 ms.
- repair time **~77 s/run = ~8.6 s/step** vs W26's ~11.8 s/step — halved,
  but **the <3 s/step target is NOT reachable at any measured-stable cap**:
  cap 92 OOMs (a 55 MB activation allocation died with 221 MB free and no
  large-enough block, `w27-on-1.stderr.log` of the cap-92 run), and the
  ~111-master eager variant OOMs too. The repeating set (~89) exceeds the
  master budget the device can hold alongside the KV-64 pool + the draft
  trunk.
- TPOT **25286 / 25319 / 25641 ms** (mean 25415) vs OFF 6647 ms —
  **THE ACCEPTANCE MULTIPLE: ~3.8x SLOWER** (W26: 3.9x). One same-binary leg
  of an earlier run measured 25304 ms with the PASSING stream `af77d5d3`
  (log overwritten by the final run; recorded here from the session
  transcript).

### Gates (run-w27-gate.sh + run-w27-gate2.sh, cap-64 build, 2026-11-19)

Discipline per leg: file mutex `exec 9>${GPU_LOCK:-$HOME/gpu.lock}`;
flock -x 9; luwen reset + 15 s + /dev/shm cleanup; idle gate (load1 < 16);
bench on `taskset -c 32-63` FROM THE REPO ROOT, builds on 64-127;
`INT8DOT=1`, durable-libs-first, `TT_METAL_RUNTIME_ROOT=$HOME/Sources/tt/tt-metal-pin`.

- **(a) Q4_K_M KV64 drafter-ON x3**: streams `77688b0a...` x3 (deterministic
  within the run); **tie-band gate FAIL x3** (the divergence class above);
  corrupted-weights red proof re-captured GREEN-as-required (gate fails on
  the planted run); TPOT 25286/25319/25641 vs OFF 6647. One cross-run leg
  (earlier run, same binary) produced the PASSING `af77d5d3` stream — the
  healthy stream shape is run-to-run bimodal under the D2D build (1 of 2
  same-binary runs), a new standing fact.
- **(b) UD-Q6_K_XL KV8 drafter-ON x3**: 3/3 exit 1, the standing capacity
  blocker UNCHANGED (`Cannot auto-fit max_model_len: not enough GPU memory
  available to serve even a single token` — the KV-8 pool + the draft trunk;
  the master budget is not the binding constraint there). No TPOT; the tie-
  band gate never runs. Q6_K_XL OFF re-verified byte-identical to the W26
  records' 24-token prefix (`ae157086...` json, eager gaps captured).
- **(c) drafter-OFF byte identity**: the W26 6/6 records STAND (no change
  expected — the OFF path never hits the spec step); the w27-off-gaps-q4 leg
  repeats the Q4_K_M OFF 24-token prefix byte-identically.
- **(d) the DNR per-step ledger with D2D restore**: above; clean checks
  still ~1 ms/weight, D2D repairs ~2 ms, host fallbacks ~180 ms.

### NEEDS_DECISION (back to the coordinator, with the numbers)

1. **The <3 s/step repair target is unreachable on this device budget**:
   the writer's repeating clobber set is ~89 weights/verify step; 64 masters
   (cap) cover ~43/step and cap 92 OOMs the activation allocations. The D2D
   transport itself is ~2 ms/weight (25x under target) — the shortfall is
   entirely the uncovered host-restage fallbacks. Levers that need a call:
   shrink the per-step clobber set (the writer defect itself — the row's
   standing mystery), a KV-64→KV-32 arm for the identity gate only, or
   accept the ~8.6 s/step standing cost.
2. **The healthy stream shape is bimodal across runs under the D2D build**
   (`af77d5d3` PASS vs `77688b0a` FAIL at the recorded 500-mnat band, both
   deterministic within a run). The tie-band gate provably separates
   corrupted from healthy, but gate (a) "tie-band GREEN x3" cannot hold on
   the failing shape. Options: widen the band (the 20-23 divergences sit at
   0.95-3.2 nats — widening to ~3.3 nats would pass poison-class noise and
   is NOT recommended), gate each run's stream against ITS OWN recorded
   stream (the af77d5d3/77688b0a pair both committed as shape references),
   or trace the address-dependence of the writer (the corruption is
   allocation-layout dependent — the real defect).
