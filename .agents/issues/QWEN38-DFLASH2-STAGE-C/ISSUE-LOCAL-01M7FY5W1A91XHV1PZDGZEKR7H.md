ID: ISSUE-LOCAL-01M7FY5W1A91XHV1PZDGZEKR7H
Title: The drafter-ON verify forward degenerates upstream of the GDN spec kernels: the conv-spec x stream collapses at GDN layer 8 with NaNs — the emitted stream is [220, 0, 0, ...] (the W4 token-identity STOP)
Row: QWEN38-DFLASH2-STAGE-C
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-15
Updated: 2026-11-20
Closed: -

## Problem

With kGdnSpecDecode (id 12) and kGreedyRejectionSample (id 107) ported, the drafter-ON leg runs to completion and the emitted stream degenerates. W21 (2026-11-16): two mechanisms found and fixed in-engine — (1) ttnn::indexed_fill miswrites multi-replacement scatters (the k-th replacement row lands at id[k]+k; survives a program-cache bust) — the conv-spec x scatter replaced with a bit-exact broadcast-mask multiply; (2) ttnn::copy into the resident state shadow does not land — the spec state commit installs the committed tensor directly (eager-only path). Validated: the conv output's zero rows gone, the state persists across verify steps (the gathered state == the expected snapshot byte-exact). The recurrence, the gather, and the BA outputs are proven faithful (CPU reference max_abs 4.5e-07). The stream is still degenerate [220,0,0,...]: the remaining poison is the GDN block output garbage at l>=1 (mixer asum 2e40) with clean gated-norm operands, plus the z GEMM 1e40 at c54+ — the out-of-place op WRITE path misaddresses under the engine's allocator/program-cache state. Next: the shared-cache harness extended with the norm multiply + the recycle pattern.

## Resolution

-

## W29 (2026-11-19/20): the write-path attribution — the writer is on-device; NEEDS_DECISION

Evidence: `docs/bench-evidence/qwen38-dflash2-stage-c-w29-20261118/README.md`
(11 instrument legs; the pin write-trace patchset is tt-metal-pin
`611e346884f`). The W28 facts stand; this wave adds, all instrument-first:

1. The pin's host write trace (live DRAM extent registry + classifier at
   `WriteToDeviceDRAMChannel` + the CQ issue-escape guard) across a full
   drafter-ON run: `misses=0` — NO host write ever landed outside a live
   extent, no CQ escapes, one 1088 B load-time staging page is the only
   slow-path live write in the run. The writer is NOT a host-side tt-metal
   write path.
2. The victims are EXACTLY five 6144x5120 GDN z-GEMM weights (install
   #22/#28/#31/#37/#43, addrs 0x231d6f00/0x2a146f00/0x2d7cef00/0x3473ef00/
   0x3a9eef00), each clean at its own `[W28-STAGE]` install check and
   zeroed by its first reuse; the strikes happen in the LOAD/WARMUP window
   (all five already zeroed at the first spec-geometry call; zero new
   onsets by the last). The W26 "repeats every verify step" picture is the
   DNR repair re-staging into the same minefield.
3. The flip-point scan (all staged weights scanned after EVERY staging)
   pins each flip to a specific LATER mixed-weight staging install (#99 for
   the first victim, #108 for the second); the pin registry's overlap dump
   at flip time shows NO registered buffer ever overlapping the victim's
   per-bank range. The writer targets the victim's base with no
   Buffer-registered vehicle.
4. The W28 suspicion that the staging slice probe is the writer is
   FALSIFIED by direct address evidence (96+ probes: the slice temp never
   lands on a victim) and by the slice-FREE probe default reproducing the
   strikes identically (same five weights, same flip install).

NOT identified: the device-side agent that holds the victim's base and
writes the zeros. The named next instrument is device-side: the standalone
staging-sequence repro (seam-level loop of the engine's install pattern,
scanning per-bank headers outside the engine), then a dispatch
write-command trace at the pin — the first layer below where the host-side
reach ends. The gate suite is NOT RUN: no fix has landed, and the DNR
machinery stays committed and DEFAULT-ON as the working mitigation. This
state is NEEDS_DECISION per the row's task rule: the writer is
upstream-adjacent (device-side tt-metal), and the repro + patchset plan
above is what the tt-metal issue needs.

## W6-W13 bisect (2026-11-15 evening — the follow-on session's record)

The W4 "collapses at GDN layer 8" framing is WRONG in one respect and right in
a sharper one: the collapse is NOT op-fixed — its onset layer MOVES with the
staging arm (w7 device-res: l=10; w8/w9 forced-host-residual: l=4, the first
full-attn layer; w12 host-free-off: l=1). The poison is POOL-STATE dependent.
What stays true: the poison is live in the FIRST spec step (T=9 — the only
non-tile-multiple geometry in the workload), and the verify's logits collapse
to argmax-0 (the all-zero accepted stream).

New evidence (all in docs/bench-evidence/qwen38-dflash2-stage-c-w3-20261115/,
README.md's W6-W13 section carries the full record):
- `[DRAFT_LOGITS]` (w13): propose #1 CLEAN (asum 4.94e6, 0 NaN); proposes #2-3
  read 100% NaN. The drafter is a VICTIM (it reads the verify's poisoned aux
  taps), not the poisoner.
- `[SPEC-OUT]`/`[RNG-IN]` (w11): the z gate reads asum 3.62e40 + 109 Infs at
  the spec step's 8th GDN layer (l=9) from CLEAN h (22227); from l=10 the gate
  reads ALL-ZEROS for the rest of the run. `[QKVZ-OUT]` mixed reads garbage
  (8.67e40+) from call 51 (l=4) while `[QKVZ-IN]` h reads clean — in BOTH
  staging arms. The poison sits between the qkvz GEMM's clean input and its
  garbage-read output.
- `[CONV-SPEC-STAGE]` (w7b): the conv-spec fill's cached zeros base reads
  ZEROS at fill time while the fill's output reads recycled garbage (identical
  bytes asum=1.34712e40 + 36 NaNs at every first-GDN-after-FA call). tt-metal
  indexed_fill = out-of-place clone + row writes (indexed_fill.hpp:11-15), so
  the output garbage is NOT the base's content.
- `[TT-SLOT-ALIAS]` census scan (w10): ZERO duplicate device addresses across
  all slots — no double-hand.

Op-level exonerations (standalone tests, GREEN at the engine geometry,
47/47 assertions): `kCausalConv1dSpecUpdate` sweep F (C=10240, widened
state_len=11, T=9, slots=9, nat=1, NaN-seeded out: max_abs 2.98e-08,
nonfinite=0) and sweep G (shifted-address second call: the call-1 buffers
held alive so call 2 lands elsewhere — max_abs 2.98e-08). The stale-address
replay hypothesis is dead for this op. `GdnSpecDecodeKernel` checksums clean
at every T=9 call. EnsureDevice2D REFUSES row-strided views outright
(residency.cpp:930) — a new test pins that contract.

## W20 (2026-11-16): the trusted dual-read — the alias theory FALSIFIED

The decisive instrument ran (README.md's W20 section carries the full
record). Verdicts:

- **The stale-download/serve-alias hypothesis is FALSIFIED.** The read-only
  trusted dual-read at the qkvz COMMIT side reads the same garbage the
  serve reads (asum 3.75443e+40, 113 Infs at the first spec step's l=9,
  call 55). The matmul's own output reads garbage PRE-commit
  ([ZGEMM-PRECOMMIT]) with BOTH operands dual-read clean and stable at the
  same site ([MMBT9]: h asum 22227.6, the staged weight byte-identical
  across steps).
- **The block-recycle alias is falsified by intervention.** A 3-deep output
  keep-alive in MatmulBTKernel (the current output's block can never be a
  recent recycle) MOVED the poison (onset later, bigger: ~9e42, ~2500 Infs)
  instead of removing it. The recycled-block content is not the mechanism.
  The keep-alive stays (harmless; onset later; first token 220→23066) but
  the stream stays degenerate (e171d6b1).
- **Three isolated reproductions at the exact engine geometry are ALL
  GREEN** (tests/vt/test_tenstorrent_backend.cpp, w20-bfp8-{m9,warm,unsynced}
  .log): the bfp8-staged weight at M=9/M=32 vs the f32 oracle (max_rel
  0.75 percent), 200 warm program-cache-hit calls, 50 unsynced-staging
  lifecycle reps. The W3 scope claim that the target's GDN weights never
  route through EnsureMatmulWeightDevice is WRONG (they do; bfp8 degrades
  them since W3) but the bfp8 matmul itself is numerically fine at the spec
  geometry.
- The dead agent's qkvz commit-side TrustDump calls were unreachable dead
  code (after `return out;` in both ProjectGdnQkvzRaw arms) — why waves
  w14-w19 captured no qkvz commit. Fixed.

## Next instrument

**SHARPENED: the recurrence's core output is ~ZERO at the spec calls**
(asum 120 → 1.2 → 0.9, exactly 4 of 9 rows zero, later NaN) with CLEAN
q/k/v (the pc_q/pc_v trusted dumps have no zero rows pre-collapse). The
recurrence's UNPROBED inputs are the next probes: the INITIAL-STATE gather
(column num_accepted-1 — the prefill's committed state as the ENGINE holds
it) and the BA outputs (araw/braw → g/beta; the sba_* dumps cover only the
prefill). The core ~0 explains the degenerate stream on its own; the z
GEMM's 1e40 read at c55 is a second anomaly (it collapses h at l=10).

Beyond that, the poison needs the engine's INTERLEAVED op mix: the conv-spec fill's
indexed_fill (the cached-zeros-base/clone anomaly stands), the GDN
recurrence's gathers/scatters, and the paged attention, all sharing the
tt-metal program cache and the DRAM allocator with the matmuls. Strongest
hypothesis: a tt-metal-pin program-cache/runtime-arg staleness under
interleaved programs (the kernel writing a previous call's buffer while the
current output reads uninitialized recycled bytes — fits the keep-alive
intervention moving the poison, the identical-bytes periodicity, and the
staging-arm-dependent onset). Next step: the interleaved-op focused case
(matmul + indexed_fill + gather through the SHARED program cache in one
harness) — the first reproduction that varies the op MIX, not the op — plus
the state-gather/BA trusted reads at the spec calls.

## W21 (2026-11-16): the probes ran — two real defects found, fixed, and the write-path class named

The raw-f32 dump instrumentation (VT_TT_SPEC_DUMP, this wave; the w21-specdump
/ w21-mask / w21-bust / w21-fix / w21-fix2 / w21-snap / w21-iscat dump dirs)
answered both probes and found the poison's first two CONCRETE mechanisms.

**PROBE 2 (the BA outputs): CLEAN.** At the first spec call (l=0), the
committed g/beta match the composed CPU math
(-exp(a_log)·softplus(a+dt_bias), sigmoid(b)) to max_abs 3.0e-07 / 1.3e-07
over all 432 (token, head) pairs; g ∈ [-8.31, -8.7e-07], beta ∈
[0.028, 0.998]. The a/b raws, the projections, and the GdnPostConv gate chain
are exonerated.

**PROBE 1 (the initial-state gather): the GATHER IS FAITHFUL — the STATE
PERSISTENCE was the defect.** The gathered S_all is byte-identical to a direct
shadow read of the same slot (gs1_S_all == gs1_slot0_direct), and the
recurrence run on the dumped inputs reproduces the TT output to max_abs
4.5e-07 (the CPU composed reference, threaded per head). But the state the
gather reads is NOT the state the previous step's recurrence produced. The
VERIFIED pairings (the W22 record correction — an earlier draft of this
paragraph paired the expected snapshot with the l=1 step-1 gather, the wrong
layer's number):
- the step-1 l=0 gather = 11984.7 (w21-specdump.stderr.log:596, gs0_S_all) —
  the l=0 PREFILL state, correctly served at step 1;
- the step-1 l=1 gather = 301.568 (w21-specdump.stderr.log:616, gs1_S_all) —
  the l=1 layer's TRUE small prefill state (the model's own small state at
  layers >= 1), not a defect;
- the step-2 l=0 gather PRE-fix = 10946.6 (w21-specdump.stderr.log:1754) —
  neither the init 11984.7 nor the expected snapshot: the copy's half-landed
  residue;
- the expected post-token-0 snapshot = 11550.9 (the recurrence run on the
  dumped inputs), and the POST-fix step-2 l=0 gather reads exactly it
  (w21-fix.stderr.log:1530).
And the cache shadow read DIRECTLY after the commit copy equals its PRE-copy
content (w21-snap/gs0_cache_post.f32 asum 11984.7 = the init state, vs the
scatter's own source rows gs0_snap_next.f32 asum 102778; gs1 the same shape
301.568 vs 2603.31). **ttnn::copy into the resident shadow does not land in
the engine.**

**THE CONV-SPEC FILL'S MISWRITE (the W7 anomaly, now resolved to a law):** the
dumped fill output shows the k-th replacement row landing at id[k]+k — ids
[0,9,18,27,36,45,54,63,72] wrote x[0..8] at blocks
[0,10,20,30,40,50,60,70,72]. The displaced writes zero out the conv outputs at
tokens 4-7 (the whole conv window reads the never-written base) — the exact
"4 of 9 rows zero" the [SPEC-OUT] core series showed, at EVERY layer, from the
FIRST spec call. The served x (the qkvz mixed) is clean at the same calls
(commit-side trust read zero=0; the serve probe agrees). **ttnn::indexed_fill
miswrites multi-replacement scatters in the engine, and the miswrite SURVIVES
a full program-cache bust** (the w21-bust arm: a fresh zeros base + a fresh
ids tensor with a unique shape per call — a guaranteed program miss — same
displacement). Single-replacement fills (the decode conv, every isolated
test) are fine; the strided-id multi-replacement form breaks.

**The two fixes (this wave, both validated bit-exact):**
1. The conv-spec x scatter is a broadcast-mask multiply:
   xT[t, s·C+c] = x[t, c] · (s == slot(t)) — exact f32 elementwise, the same
   traffic class as the fill, no indexed_fill. Validated: the conv output's
   zero rows are GONE (the w21-mask dumps: the slot-0 blocks == x for all 9
   tokens; the out zero rows = []), and the first spec call's core went from
   asum 120 with 24576 zeros to 305 with 0 zeros.
2. The spec state commit installs the committed tensor as the slot's shadow
   directly (CommitDeviceLogical2D replaces s->device) instead of
   ttnn::copy into the resident shadow — legal because the spec kernel is
   eager-only (the capture refusal at its head). Validated: the step-2 l=0
   state now persists across verify steps (the post-fix gather = 11550.9 =
   the EXACT expected post-token-0 snapshot, w21-fix.stderr.log:1530; the
   pre-fix gather read 10946.6, w21-specdump.stderr.log:1754; steps 3-4 =
   11603 / 11464.8, evolving).

Also corrected: the W20 record's "no zero rows in any [9x16] dump before seq
795" is WRONG — the spec pc_q/pc_v/spec_conv dumps carry exact-zero rows at
tokens 5,6,7 in 192/192 dumps from the first spec call (they were the conv
fill's displaced writes, downstream). And the small ssm states at layers >= 1
(asum ~300-460 vs 11984.7 at l=0) are the model's TRUE behavior — the
drafter-OFF legs' byte-identical stream proves the prefill states, so the
small spec cores at l>=1 (asum 3.9, the true math on the true state) are not
themselves the degeneracy.

**Where the hunt stands:** the stream is STILL degenerate [220, 0, 0, ...]
(the w21-fix sha daa113b5 — the original W4 signature, not the keep-alive
variant). The remaining poison shows as the GDN block's output garbage at
l >= 1 (the [MIXER-OUT] l=1 asum 2.00072e+40, no NaN/Inf/zeros — the
bit-reinterpretation signature) with CLEAN gated-norm operands ([RNG-IN] x =
the core's bytes, g = the z's bytes, both clean), and as the z GEMM's 1e40
commit at c54+ (the W20 record). The gated norm's output and the out_proj are
the next unprobed stage. The class is now NAMED: **the out-of-place op WRITE
path misaddresses under the engine's allocator/program-cache state** — the
fill (displaced), the copy (no-land), and the mixer garbage are three
symptoms. The next instrument is the shared-cache harness (matmul +
indexed_fill + gather through ONE program cache in one process, the W20
disposition's named case) extended with the norm's multiply and the
engine's alloc/free recycle pattern.
**W21 gates + the harness (2026-11-16, same wave):** the drafter-OFF
byte-identity gate holds at 6/6 (Q4_K_M d49c53d2 ×3 at TPOT 6626.89-6627.43
vs the recorded 6607; UD-Q6_K_XL b36bc61a ×3 at 5325.97-5331.21) — the two
fixes are inert off the spec path. The drafter-ON token identity STILL FAILS
(3/3 degenerate; the degenerate stream's exact bytes follow the
instrumentation's timing — e171d6b1 probe-free vs daa113b5 probed — the
degeneracy is the stable fact), so **the acceptance multiple remains
UNMEASURABLE**. The UD-Q6_K_XL drafter-ON leg stays capacity-blocked (the KV-8
pool + the draft trunk; the standing W3/W4 blocker). The W20 disposition's
shared-cache harness RAN and is GREEN (40 reps of matmul + multi-replacement
indexed_fill + gather through one shared program cache with the recycle
churn, 0 bad) — the engine-only nature of the poison stands; the harness
needs the richer engine state (the bfp8 staging lifecycle, the embed, the
paged attention, the DBuf pool pattern) and the gated norm's output + the
out_proj are the next unprobed engine stages.

## W22 (2026-11-16): PROBE 3/4 clean — the poison's entry is the mixed qkvz GEMM's COMPUTE (call 54, l=8); the harness reproduces the class in isolation

The four W21-review repairs landed first (the dead same_spec scaffolding
collapsed; the probe-1 record's layer pairing corrected; the GDN-suite
failing-NAMES baseline committed — the 15-vs-16 discrepancy resolves as the
RAC capture-safety case FLOATING with the box's device state, the stable
set is 15; the ScatterRowsExact strided-form refusal + its red/green test).
The env-doc gate repair (VT_TT_SPEC_DUMP / VT_TT_CONV_SPEC_FILL_MASK
documented; FRESH/BUST allowlisted) rode the same wave.

**PROBE 3 (the gated norm's own write): CLEAN.** [NORM-PRECOMMIT] ==
[TRUST-SLOT] norm-commit-N at the spec geometry (rows=432, d=128), both
calls, dual-read stable, zero=0: the norm's committed shadow IS the norm's
output (284.18 / 246.648).

**PROBE 4 (the out_proj's own write): CLEAN.** [OUTPROJ-PRECOMMIT] ==
[TRUST-SLOT] outproj-commit-N at M=9, K=6144, N=5120 (775100 / 445293,
agree=1). The norm's and the out_proj's writes LAND — the two unprobed
writes between the clean operands and the mixer output are exonerated.

**The poison's entry, localized:** the first garbage in the run is
[QKVZ-OUT] call=54 mixed_asum=inf (w22-probe.stderr.log:1022) — the MIXED
qkvz GEMM (N=10240) at the first spec step's l=8. [MIXED-PRECOMMIT] (new)
reads the SAME garbage pre-commit (asum 3.46266e+40, inf=108, out_addr
b754800 = the [TT-COMMIT]'s address = the [XSERVE-TRUST]'s serve/owner
address) with the [MMBT9] operands dual-read verified clean at the same
site, while the SPLIT z GEMM on the SAME h (N=6144, the same call) is
clean (asum 21911.1). The WRITE path is exonerated (commit == pre-commit ==
serve); **the COMPUTE is the convict** — the W20 z-GEMM class (garbage from
clean operands) on the mixed arm. The onset moved l=1 → l=8 across the W21
fixes (l=0..7 are clean end to end in this run); the FA layers downstream
of the collapse stay correct ([MIXER-OUT] l=11/15 clean). B3's write-path
fix + the acceptance gates therefore do NOT run: there is no write path to
fix — the compute class is a tt-metal-pin program-cache/runtime-args
investigation, and the acceptance multiple remains UNMEASURABLE (the
stream still degenerate, e171d6b1).

**B4 — the harness grew and REPRODUCES the class in isolation** (the new
case after the W21 harness; `w22-harness.log`, `w22-disc.log`): the GEMM
trio on ONE h (mixed/z/outproj, bfp8-staged weights), the embed
(kEmbeddingQuant Q8_0), RAC+PA at the engine's KV geometry (D=128, block
32, the seq advancing), and the DBuf pool's persistent-buffer hold — the
trio/embed/fill/gather/hold are 0-bad in 40 reps, but the PA ingredient is
RED, and stripping the mix entirely keeps it RED: **ONE raw RAC write at
slot 0 then ONE raw PA decode gives deterministic garbage at seq=2
(4.86e+33), seq=5 and seq=7 (2.99717e+32 — the IDENTICAL bytes at both),
while seq=1,3,4,6,8 are clean**; the Copy-seeded arm (the host read path)
is clean, seq=64 is clean, and the RAC's writes land in all 40 reps
(rac_bad=0). This is the FIRST isolated reproduction of the compute-class
misaddressing — at an out-of-engine form (the engine's PA windows are never
that small): a real PA device-SDPA defect at seq=2/5/7, the next row's
lever, not this poison's mechanism.

## W25 (2026-11-18): the rebuilt lens instrument — the corruption map, the verdict, the repair

The reboot-killed W23 lens-arbitration instrument was rebuilt from the
consolidated findings (`run-w25-lens.sh`, `decode_bfp8_lens.py`, the new
`docs/bench-evidence/qwen38-dflash2-stage-c-hunt-20261118/` record) and the
arbitration re-ran on the failing call. Facts:

- **The failing call reproduces byte-identically**: sq=48 in the lens
  sequence — the 7th mixed qkvz GEMM of the first spec step (N=10240) —
  output `c_asum=3.46266e+40, inf=108` (the W22 signature exactly; the z GEMM
  on the same h: `21911.1`, also exact).
- **The corruption map** (`corruption-map-sq48-N10240.csv`): 0.015% bad
  fraction — EXACTLY the W23 number — confined to 8 tiles (0,0)..(0,7):
  the weight's first 32 rows x 256 cols. The corrupted bytes are the 512 B of
  BFP8 EXPONENT HEADERS (8 x 64 B, stride 1088), ZERO at dump time, mantissas
  intact. The same zero-header signature on the sq55/sq61 z-GEMM weights, and
  intact headers on the other-step dumps: a RUNTIME writer, landing between
  staging and the spec-geometry matmul on whichever buffer holds the weight.
- **The output garbage decodes as compute-side**: the c garbage is confined
  to output cols 0..31 (29 Infs, bf16 exp-31-dominated 2^126..2^128 patterns
  = the exact output columns the zeroed-exp weight rows feed, at the exact
  magnitude foreign exp bytes produce); the activation h is finite (max 1272).
- **The W23 "two layouts" finding was a DECODER ARTIFACT**: the staged bytes
  decode 100% identical to the metadata readback under the pin's face-packed
  tile order (`bfloat8.cpp pack_as_bfp8_tiles` +
  `blockfloat_common.cpp convert_u32_to_bfp`, read at the pin) — raw-as-plain
  BFP8 is garbage by construction. Hypotheses (a) staging-layout and
  (c) metadata-mismatch are falsified; the verdict is (b), the partial-write
  clobber.
- **The repair** (`Bfp8RestageWeight`, `tenstorrent_residency.cpp` +
  the MatmulBTKernel M==9 arm, `VT_TT_BFP8_RESTAGE=0` rolls back): at the
  spec geometry re-run the cold staging unconditionally — fresh typecast =
  fresh allocation = clean bytes (the W23-verified fact). The lever's
  viability is intact (no re-architecture; footprint unchanged).
- **Repaired repro**: sq48 reads `c_asum=32186.3` (was 3.46e40); all 64 lens
  calls finite, 0 NaN/Inf; the stream is non-degenerate (the instrumented leg
  `913e637c...` — probes and the keep-alive instrument perturb exact bytes, so
  identity is judged on the probe-free gate legs).
- **Gates** (`run-w25-gate.sh`, 2026-11-18): (b) OFF Q4_K_M x3 —
  `d49c53d2...` byte-identical 3/3, TPOT 6646.65/6648.11/6648.95 vs 6607;
  (c) OFF Q6_K_XL x3 — `b36bc61a...` 3/3, TPOT 5349.75/5353.04/5351.08 vs
  5340.9; Q6_K_XL ON — the standing capacity blocker. The w20-bfp8 isolated
  arms with the repair: PASS (cold 0.75% max_rel, warm 200/200, unsynced
  0/50). **(a) ON x3: FAILS — the STOP condition.** 3/3 deterministic
  (`af77d5d3...`), non-degenerate, 22/24 tokens equal the recorded stream
  with one 3-token shift at index 7; TPOT 26975 ms — the unconditional
  re-stage costs ~18.5 s/token. THE ACCEPTANCE MULTIPLE remains
  UNMEASURABLE. Next: the verify-side acceptance trace at the divergent
  step, and the detect-and-repair form of the fix (store the 512 B head
  reference at staging; re-stage only the clobbered weight).

## W26 (2026-11-19): the detect-and-repair form + the acceptance trace — the accept/commit path is faithful; the residual is the M=9 verify forward's near-tie parity

The detect-and-repair form of the W25 repair landed (`Bfp8DetectRepair`,
`tenstorrent_residency.cpp` + the `M == 9` arm in `tenstorrent_ops.cpp`;
`VT_TT_BFP8_RESTAGE=0` still rolls back): the 512 B header fingerprint (8
tiles x 64 B) is captured at cold staging, probed per spec-geometry call via
a KB-scale `ttnn::slice` + raw `ReadFromBuffer` (~1 ms per weight), and only
the clobbered weight re-stages. The acceptance trace (`TraceVerifyAcceptance`
in `rejection_sampler.cpp`, `VT_SPEC_ACCEPT_TRACE=<N>`) downloads the verify's
target argmax + logits top-5 per expanded row at the first N spec steps.
Facts (`docs/bench-evidence/qwen38-dflash2-stage-c-hunt-20261118/`, the W26
README section):

- **Cost**: clean checks 988/9 steps = ~113 ms/step total (KB-scale, the
  design target holds for the CHECKS); but **~55-65 repairs per verify step
  at ~180 ms each = ~11.8 s/step** — the runtime writer RE-CLOBBERS ~60
  weights every verify step (the same keys, `bad` 33..512 headers, again),
  so detect-and-repair degenerates toward the blanket restage at this writer
  rate. ON TPOT 25578 ms vs W25's blanket 26676 ms (-4%) vs OFF 6647 ms.
- **The 3-token shift's verdict**: the accept walk and the commit are
  FAITHFUL — the `[ACCEPT]` trace shows the verify's row argmaxes agreeing
  with the accepted draft prefix and the emitted tokens being exactly the
  verify's argmaxes. The shift enters in the TARGET's M=9 verify forward:
  at identical conditioning (last token 16) the eager M=1 decode's argmax is
  220 while the M=9 verify row's argmax is 23066, with top-2 logit gaps of
  0.03-0.5 — inside the BFP8 lever's 0.75% max-rel band. The DIRECT-ARGMAX
  draft (argmax over the target's own taps, the same numerics class) proposes
  the verify-consistent path, the verify confirms it, and the lossless accept
  commits it; the periodic `[220,16,23066]` workload re-aligns one element
  later. The recorded OFF stream is NOT corrupt-tainted (W25 + W26 OFF gates
  byte-identical `d49c53d2` / `b36bc61a` on the repaired tree) — no
  re-record.
- **NEEDS_DECISION (coordinator)**: not "the re-stage changed the target's
  numerics" (the repaired bytes equal the cold staging by construction and
  the OFF gates prove the clean path unchanged), but the narrower fact that
  a token-identity gate defined against the eager stream cannot pass over an
  M=9 BFP8-batched verify forward whose quantization noise flips near-ties.
  Options named in the W26 README: a tie-band identity gate, an f32/bf16
  verify-forward arm for the identity gate only, or accepting cycle-phase
  differences on periodic workloads. The repair-cost lever (device-side D2D
  master restore vs host re-stage, ~2x BFP8 pool footprint) is the second
  decision owed.

## W27 (2026-11-19): the tie-band gate + the D2D master restore — the coordinator's two decisions implemented; both gates land, both targets miss, NEEDS_DECISION with the numbers

The coordinator decided the gate form (a tie-band gate per the ratified
near-tie precedent) and the perf fix (device-side D2D master restore). Both
are implemented and measured
(`docs/bench-evidence/qwen38-dflash2-stage-c-hunt-20261118/`, the W27 README
section):

- **The tie-band gate is committed and separates corrupted from healthy**:
  threshold 500 mnats (the ratified kNearTieMnats band), the eager
  permissible-divergence positions from the new `VT_EAGER_TOP2_TRACE`
  instrument over the recorded OFF stream, acceptance telemetry from the
  `[ACCEPT]` trace, and the NaN/Inf + degenerate-id hard check. Verdicts:
  the W26-form healthy stream `af77d5d3` PASSES (2 mismatches, both
  near-tie); the W27 D2D build's stream `77688b0a` (3/3 deterministic)
  FAILS on poison-class divergence at positions 20-23 (eager gaps 0.95-3.2
  nats); the corrupted-weights RED leg (`Bfp8PlantHeaderZero` +
  `VT_TT_BFP8_RESTAGE=0`) reproduces the exact W4/W20 signature
  `[220, 0, 0, ...]` and the gate FAILS as required. **The corruption
  detection property survives.**
- **The D2D master restore works and misses the target**: lazy
  per-clobbered-weight masters (admitted at the second clobber, cap 64),
  the repair is a fresh-allocation D2D clone at ~2 ms/weight (the <50 ms
  target met 25x), but the writer's repeating footprint is ~89
  weights/verify step and cap 92 OOMs the activation allocations —
  repair time ~8.6 s/step vs W26's 11.8 (halved; the <3 s/step target
  unreachable at any measured-stable cap). TPOT 25286-25641 ms vs OFF 6647:
  **the acceptance multiple is ~3.8x SLOWER, unchanged in verdict**.
- **New standing fact**: the healthy drafter-ON stream shape is bimodal
  across same-binary runs under the D2D build (`af77d5d3` passes the band,
  `77688b0a` fails it, each deterministic within its run) — the corruption
  is allocation-layout dependent.
- **UD-Q6_K_XL KV8 drafter-ON**: the standing capacity blocker is unchanged
  (3/3 "not enough GPU memory to serve even a single token"); the tie-band
  gate never runs on that arm.
- **NEEDS_DECISION** (the two named conditions of the dispatch are both
  met, numbers in the README W27 section): (1) the <3 s/step target vs the
  ~89-weight repeating clobber set and the 64-master budget ceiling; (2)
  the bimodal healthy stream shape vs the 500-mnat band (band widening is
  not recommended — it would pass poison-class noise; the options are
  per-run shape references or tracing the writer's address dependence).

## W27 host-drafter probe (2026-11-19, second W27 session): the host arm WORKS and streams byte-true, but its forward numerics collapse near-tie acceptance — NEEDS_DECISION (coordinator)

The W27 coordinator decision moved the drafter OFF the device (the trunk is
host-resident, `VT_DFLASH_HOST=1`, the D9 host path re-run on a CPU queue —
`set_dflash_draft` + `propose_drafts_block` in `runner.cpp`, `DflashReqCtx::
host_store`; committed inside 649519651). Facts (evidence
`w27h-on-1.{log,stderr.log}`, Q4_K_M KV64, 12 output tokens, VLLM_CPP_CPU_
THREADS=32 to match the taskset 32-63 bench):

- **FUNCTIONAL**: the host arm proposes drafts, keeps the host KV store in
  lockstep (ctx 32, 33, ...), stages NOTHING of the 3.85 GiB trunk to the
  device (the wiring notice replaces `PreStageDflashWeights`), and the emitted
  stream byte-matches the recorded drafter-ON stream
  `[220,16,23066,220,16,220,16,1814,...]` — the verify path is untouched.
- **ACCEPTANCE COLLAPSE**: 1.0 accepted tokens/verify-step vs the device arm's
  ~2.4 (`[host-debug]` dual-forward cross-check, `VT_DFLASH_HOST_DEBUG=1`):
  on IDENTICAL state (same block, same step, both stores appended) the host
  forward's logits are FLAT and 220-heavy (top ~9 mnats, junk ranks) where the
  device forward's are confident cycle predictions (top ~19 mnats,
  `[16,17,18,...]`). Not a near-tie flip — a qualitative divergence. The
  context is present (host ctx=32 at step 0, built by ONE append == full
  recompute by D9), so the suspicion is the CPU backend's op numerics at
  production shapes (5 layers, H 2560, C 32+9 rows, vocab 248320) — the D2
  CPU forward was parity-validated at fixture scale, never here. NOT rooted:
  the combined-feature widen and the bf16-rounding alignment are ruled in as
  correct-to-try (both applied, no change).
- **TPOT (the acceptance multiple)**: drafter-ON host 55629 ms vs drafter-ON
  device 25304 ms (w27-on-1) vs OFF 6748 ms (w27-off-gaps-q4) — the host arm
  LOSES 2.2x to the device arm through acceptance, and 8.2x to OFF.
- **THE CORRUPTION BET FALSIFIED**: step-wall is ~60.7 s/verify-step in BOTH
  arms (`VT_STEP_WALL=1`): the drafter trunk's device residency is NOT the
  runtime writer's cause — the DNR ledger still records ~80 repairs/verify
  step and ~6.5 s/step of check time with the trunk off the device (the W26
  "113 ms/step" check figure does not reproduce; the ledger's `us` field
  counts queue-blocking time behind the replay, so the recorded per-check
  cost and the wall cost diverge — the ledger needs a queue-idle re-measure
  before any check-cost target is judged).
- **TPOT accounting**: OFF ~6.7 s/token at ~2.4 accept/step (~16 s/step);
  ON-device ~25.3 s/token (~60.7 s/step wall, ~2.4 accept); ON-host ~55.7
  s/token (~60.7 s/step wall, ~1.0 accept). The verify+DNR wall dominates
  every ON arm; the draft forward is 187 ms/step on 32 host cores.
- **WORKTREE COLLISION (process)**: this session and the concurrent W27
  session (commit 649519651) edited the same worktree; that commit swept this
  session's uncommitted host-arm code and evidence into its tree, and staged
  a revert of this session's `VT_DFLASH_HOST_DEBUG` instrumentation (the
  instrumentation survives in 649519651). Nothing is lost, but the two
  readings of the W27 decision (host drafter + host-parallel repair vs
  tie-band gate + D2D repair) were executed in parallel on one checkout.

NEEDS_DECISION (coordinator), per the task's own stop condition "if the host
draft forward cannot meet the loop's latency": the host draft cannot, at its
current numerics — the numbers above. Options: (a) root the CPU-backend
divergence (a host-vs-device forward parity gate at production shapes is the
missing instrument; it does not exist yet); (b) accept the host arm for
CAPACITY only (UD-Q6_K_XL KV8 fits with the trunk off device — untested, the
leg was not run) and keep the device drafter for TPOT; (c) abandon the host
move and take the D2D repair lever (the concurrent session's reading).
Untested and owed by this session: the UD-Q6_K_XL KV8 capacity leg (c) of the
W27 gate list — the drafter-ON device arm never ran it with the trunk off
device because the host arm's acceptance collapse makes its TPOT meaningless.

## W28 (2026-11-19): the writer-catch canary — the write-set proven byte-exact, four suspects exonerated, class (a) stands (this session, worktree /tmp/tt-writer)

The W28 canary (`VT_TT_BFP8_CANARY=1`, diagnostics only, this directory's
sibling `qwen38-dflash2-stage-c-w28-20261118/README.md` carries the full
record) ran five instrument legs (Q4_K_M KV64 drafter-ON,
`VT_TT_BFP8_RESTAGE=0`, 12 tokens, 3 complete legs + 2 partials). Findings:

- **The writer's write-set is EXACTLY 512 B of zeroed BFP8_B tile exponent
  headers** at the weight's first 8 tiles (the first page of each of the 8
  DRAM banks): the byte-diff against a fresh deterministic typecast of the
  same host source is `diff=512 hdr=512 data=0 diff_zero=512` at every
  strike, three legs in a row, at the same call indexes (#52/#58/#61).
- **It lands out of band of the command queue**: clean at the stage-time
  blocking read ([W28-STAGE], 185/185 checks zero-free), zeroed at the
  matmul-arm blocking read, nothing enqueued between (single CQ, no trace
  capture exists in the leg — `[W28-CAPTURE]` never fires).
- **Exonerated with direct evidence**: the staging and the pin's typecast
  ([W28-MICRO]: headers fully written even into a 0xFF-prefilled target),
  the slice probe chain (`post == pre` on 192 checks — the probe never
  writes; it also UNDERREPORTS 1/8 vs the direct read's 8/8, so every
  slice-mediated reading needs the direct-read caveat), the allocator
  (block table `blocks=1` at every check, no overlap), the trace replay
  (falsified: no capture), and the drafter's staging (the struck weights
  are the target's mixed-qkvz GEMM weights).
- **Class (a) stands, refined**: a stale (freed-and-reused) address written
  out of band — the per-call staging recycles the same-size-class blocks
  (the shadow cache is keyed by the host pointer, which MOVES every call),
  and the stale write lands in whichever staging now owns the address.

NOT done this session: the writer's op identity (the next instrument is the
write-path attribution: wrap tt-metal's host write/upload paths with
address logs over the deterministic strike window), therefore no fix, and
the gate suite (corruption-absent, token identity, the acceptance multiple,
drafter-OFF byte identity) is not run. The DNR detect-and-repair keeps
treating the writer's damage every step (~89 weights/step); the 3.75x
drafter-ON slowdown is this writer's cost.

## W30 (2026-11-20): the seam-level bisect — the writer does NOT reproduce outside the engine; the DNR stays DEFAULT-ON

Evidence: `docs/bench-evidence/qwen38-dflash2-stage-c-w30-20261120/` (the
three repro logs + the gate suite). The named next instrument ran: a
standalone C++ program against the pin's durable libs directly (outside
vllm.cpp, `tools/tt-metal-repro/`), looping the engine's exact install
(`Bfp8TypecastInstall`: bf16 `Tensor::from_span` H2D temp -> on-device
`ttnn::typecast` to `BFLOAT8_B` -> the temp dropped with the call) over the
trio shapes (6144x5120 z, 6144x10240 mixed-qkvz, outproj, 5120x5120), with a
full-buffer direct read of every watched weight after every install counting
the zero exponent headers (the engine's `Bfp8DirectZeroHeaders`). Three
variants, each the next named bisect state:

1. `leg-pure.log` — the pure install loop, alive window 2, 120 installs:
   CLEAN, strikes=0.
2. `leg-mix.log` — + program-cache-warm op mix: M=9 matmuls through the
   just-installed weights, D2D `assign` master clones, varying-size bf16
   temp churn: CLEAN, strikes=0.
3. `leg-resident.log` — the W29 flip-point residency pattern: 8 RESIDENT
   weights + 120 churn installs of dropped same-size staging temps, every
   resident scanned after every churn: CLEAN, strikes=0.

The W22 harness verdict (trio + embed + RAC/PA + DBuf through ONE shared
program cache, 40 reps, GREEN except the unrelated PA seq-defect) already
covered the remaining named states short of the full engine. What the
repro still lacks and the engine has: the ~25 GB load-time residency, the
GGUF mmap host sources, the drafter staging, the trace-region carve /
async-executor machinery. Verdict: **the corruption is engine-state-
dependent; the minimal seam-level repro was NOT found at the states the
task named.** The writer's identity therefore remains UNFOUND; no pin
patchset attempt is possible without a mechanism, and no tt-metal issue is
drafted (an upstream issue without a repro would be noise — the three-leg
bisect record is the artifact the upstream hunt needs).

Consequence recorded in the tree: the W30 dispatch's "DNR disabled by
default (the writer fixed)" premise FAILED — the default flip was reverted
in the same wave and the DNR detect-and-repair stays DEFAULT-ON as the
working mitigation; `VT_TT_BFP8_RESTAGE` remains the rollback/regression
knob. The gate suite runs BOTH arms on the same binary: the proof leg
(`VT_TT_BFP8_RESTAGE=0` + the W28 canary) re-measures whether the writer
still strikes with no repair armed, and the DNR-armed legs carry the token
identity + the acceptance multiple. Results in the W30 README.
