# QWEN38-DFLASH2-STAGE-C W29 — the write-path attribution (2026-11-19/20)

Row: `QWEN38-DFLASH2-STAGE-C`, issue
`ISSUE-LOCAL-01M7FY5W1A91XHV1PZDGZEKR7H`. Worktree `/tmp/tt-writer` (branch
`row/QWEN38-DFLASH2-STAGE-C-HUNT-W`, base `c8d9f2db6` — the W28 canary),
engine build `/tmp/build-ttfp` (durable-headers recipe, `VLLM_CPP_TENSTORRENT=ON`),
pin instrument patchset committed at
`~/Sources/tt/tt-metal-pin` `611e346884f`. The W28 record is
`../qwen38-dflash2-stage-c-w28-20261118/README.md`; W28's proven facts are not
re-derived here.

## The instruments this wave added

- **The pin write trace** (`TT_METAL_WRITE_TRACE=1`, pin commit `611e346884f`):
  a live DRAM extent registry (buffer alloc/dealloc hooks, per-bank footprint)
  plus a classifier at `WriteToDeviceDRAMChannel` — every slow-path host DRAM
  write is classified live/miss/dead-hit, misses carry a host backtrace, and a
  CQ issue-write escape guard logs any issue target outside its region.
- **The bracket scans** (`VT_TT_BFP8_CANARY=1`): `[W29-SCAN]` reads each staged
  weight's 8 per-bank page headers DIRECTLY (8 × 2 KiB
  `ReadFromDeviceDRAMChannel` reads, no device allocation) at every staging
  install (`seq=-2xxx`), at the first and last spec-geometry canary checks, and
  `[W29-SERVE]` scans each weight at its first 300 cache-hit serves. At flip
  time the pin registry dumps every live/dead range near the struck address.
- **The slice-free header probe**: `Bfp8HeaderProbe` no longer runs
  `ttnn::slice` — it reads the 8 headers straight off each bank. The legacy
  slice body remains behind `VT_TT_BFP8_SLICE_PROBE=1` (with
  `VT_TT_BFP8_PROBE_ADDR=1` it logs the slice temp's address) for the
  confirmation legs.

## The attribution legs (`w29-attrib-1`, Q4_K_M KV64 drafter-ON,
`VT_TT_BFP8_RESTAGE=0`, 12 tokens; per-leg stderr is kept beside this
README with the leg number in the file name)

1. **Leg 1 (write trace + canary)**: the W28 strikes reproduce byte-exactly
   (#52/#58/#61, `diff=512 hdr=512 data=0 diff_zero=512 range=[0x0,0x1dff]`).
   The write trace's verdict across the WHOLE run: **`misses=0`** — no host
   slow-path DRAM write ever landed outside a live extent; **no CQ-escape**
   lines; the ONLY slow-path write into a live buffer in the entire run is one
   1088 B load-time staging page. The writer is NOT a host-side tt-metal write
   path.
2. **Leg 2/4 (bracket scans)**: `[W29-SCAN] seq=0` — at the FIRST
   spec-geometry canary check, **five weights are already zeroed**
   (0x231d6f00, 0x2a146f00, 0x2d7cef00, 0x3473ef00, 0x3a9eef00 — all five are
   the 6144×5120 GDN z-GEMM weights staged at install #22/#28/#31/#37/#43,
   each read CLEAN by `[W28-STAGE]` at its own install); `seq=63` shows the
   same five, no new onsets. The writer acts between a weight's clean
   stage-time check and its first spec-geometry reuse — inside the
   load/warmup window, NOT during the spec steps.
3. **Leg 8/10 (flip-point scans)**: scanning ALL staged weights after every
   staging install pins each weight's flip to a specific LATER staging:
   0x231d6f00 flips during install #99's window, 0x2a146f00 during #108's —
   both striking stagings are MIXED-weight (10240×5120) installs, ~77–80
   stagings after their victim. The overlap dump at flip time shows the
   struck weight's own per-bank extent (0x3fc000, LIVE) and dense
   re-handing of neighbouring segments (several uids sharing 0x235d2f00 /
   0x239d2f00), but **no registered buffer ever overlaps the struck weight's
   own per-bank range**.
4. **Leg 6b (the probe in the act)**: with the legacy slice probe armed and
   `VT_TT_BFP8_PROBE_ADDR=1`, 96+ probe invocations log
   `slice_out=0x9acxxxx ok` — the slice temp NEVER lands on the struck
   weight. The W28 suspicion that the staging slice probe is the writer is
   FALSIFIED by direct address evidence.
5. **Leg 11 (slice-free default)**: with the slice-free probe default (the
   staging slice eliminated entirely), the strikes reproduce IDENTICALLY —
   same five weights, same flip at install #99, same canary verdicts. The
   slice-free probe ships as the default (the 8-page device temp per staging
   is gone either way), and the writer is confirmed INDEPENDENT of our
   probe machinery.

## Where the hunt stands after W29 — NEEDS_DECISION

Proven this wave, all instrument-first:

1. The writer is **on-device**: no host slow-path write (misses=0 across a
   full run), no CQ issue-escape, no registered-buffer overlap with the
   victim's per-bank range. It writes 512 zero bytes (64 B at each bank's
   first page) at live staged-weight bases without any host-visible vehicle.
2. The victims are exactly five 6144×5120 GDN z-GEMM weights; each is clean
   at its own `[W28-STAGE]` install check and zeroed by its first reuse; each
   flip correlates with a specific later mixed-weight staging install.
3. Our staging machinery (typecast, slice probe, shadow cache, DNR) is
   exonerated with direct address evidence this time — the corruption is
   identical with the slice probe removed entirely.
4. The strikes happen during the load/warmup window, so the W26/W28
   "repeats every verify step" picture is the DNR repair machinery re-staging
   into the same minefield, not a repeating writer.

NOT identified: which device-side agent holds the victim's base address and
writes the zeros. The remaining instrument (named for the next session) is a
DEVICE-side one: the standalone staging-sequence repro (seam-level test that
loops the install pattern of `run-w29-attrib.sh` — bf16 H2D temps,
typecasts, frees — and scans per-bank headers after each install, outside
the engine), then a pin-side instrument of the dispatch write commands
(executing the writes on a dispatch risc with a filtered write-log), which
is the only layer below the point this wave's host-side reach ends.

**The gate suite is NOT RUN at this state**: no fix has landed, because the
writer's device-side identity is still the missing datum; the DNR detect-and-
repair stays committed and DEFAULT-ON (it remains the working mitigation —
the crutch cannot come out while the writer is alive). Per the task's
decision rule this is NEEDS_DECISION: the writer is upstream-adjacent
(device-side tt-metal), the minimal repro is the standalone staging-sequence
test named above, and the patchset plan is a dispatch-write-command trace at
the pin plus, if it convicts the dispatcher/allocator, an upstream tt-metal
issue with that repro.
