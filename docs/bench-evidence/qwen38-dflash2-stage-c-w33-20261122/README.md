# QWEN38-DFLASH2-STAGE-C W33 — the device-arg echo (2026-11-22/23)

Row: `QWEN38-DFLASH2-STAGE-C`, hunt wave W33, worktree `/tmp/tt-w31`
(branch `row/QWEN38-DFLASH2-STAGE-C-HUNT-W31`, head `bfd88f4fd` W32), engine
build `/tmp/build-stagec-w31` (unchanged — zero `src/` changes this wave, so
the CPU-target link is not owed, same shape as W31/W32). Pin
`~/Sources/tt/tt-metal-pin` head `35be99e764e` (W31+W32 instruments) plus the
uncommitted W33 instrument (`fd_mesh_command_queue.cpp`, backed up at
`/tmp/w33-pin-backup.patch`, committed this wave). Issue
`ISSUE-LOCAL-01M7FY5W1A91XHV1PZDGZEKR7H`.

## The instrument (W33, named by W32)

W29/W31/W32 measured every HOST-side address carrier clean: no host write
path, no kernel launch arg (RTA or retained CRTA), no constructed dispatch
write command ever carries a victim-window address. The remaining candidate
was a DEVICE-RESIDENT arg/descriptor base that host-side dumps cannot see
(they print host values only).

The W33 instrument (`TT_METAL_DEVECHO=1`, default off, inert — env-gated in
`tt_metal/distributed/fd_mesh_command_queue.cpp`): `enqueue_mesh_workload`
records each launch's TENSIX kernel-config ring slot (the
`dispatch_md.kernel_config_addrs[idx].addr` base, the program's per-core-type
config size, the TENSIX kernel-group core range) and the NEXT call reads back
ONE core per range (`w33-echo-11` proves a stale base is visible on any
single member: CRTAs, which carry the tensor base bindings per W31, are
multicast-identical across the group). Every 16-byte-chunk read (the UMD DMA
path at >= 32 bytes is avoided deliberately) prints one `[TT-DEVECHO]`
line with every word inside a plausible DRAM address range
(`TT_METAL_DEVECHO_LO/HI`); cores whose read threw once are skip-listed
forever. `TT_METAL_DEVECHO_EVERY=N` samples every Nth program launch
(N=256 on the recording leg).

## The repro held

`w33-base-1` (baseline, `TT_METAL_DEVECHO=0`, rebuilt pin libs): clean run,
exit 0, strikes at #52/#58/#61 deterministic (`w33-base-1.canary.log`), 131
canary lines — the rebuilt libs reproduce the W28/W31/W32 leg exactly.

`w33-echo-11` (recording leg, `EVERY=256`): reproduced the full deterministic
strike window — `#52/#58/#61 verdict=ARRIVED-ZEROED` identical to baseline,
probes through `#63 addr=0x2d04ef00 … verdict=clean` — while sampling 80
device-arg dump lines (`w33-echo-11.devecho.log`) across lseq 0..18432+
(init, staging, the strikes, post-strike probes). The leg then aborted
(exit 134, "MMIO per-op timeout: 4B load took 51210 us") AFTER the strike
window — see the stability note below.

## Verdict: the device-arg surface is clean (outcome 2, narrowed)

`w33-analyze-11.txt`: across every device-arg readback taken during and after
the deterministic strikes, ZERO words 256B-aligned or equal to a victim base
fall inside any victim per-bank window `[base, base+0x3FC000)` (extent =
canary size/8 per bank). One unaligned in-window word exists
(`0x2a412483`, slot word offset 823, constant across the run) — kernel-text
instruction encoding, the same W31 false-positive class (an arbitrary
32-bit RISC-V word can numerically land in a DRAM window). A stale
device-resident base, never re-patched, would persist across launches and be
sampled repeatedly at a 256B-aligned address; none exists on the sampled
surface.

The descriptor class therefore narrows to device-resident state OUTSIDE the
L1 kernel-config arg region — kernel-embedded compile-time constants baked
into the cached program binary at build time (the base address a
typecast/copy kernel carries in its own text). **W34 instrument (named):
compile-time constant audit of the cached typecast/copy programs** — scan the
kernel ELFs (and the launched binaries' text/data sections on device, or the
host-side built-binary cache) of every cached program active around calls
#52/#58/#61 against each victim per-bank window, plus a staging-freeze
bisect (rebuild the suspect programs with the staging-temp address
substituted) if the static scan is inconclusive.

## Residual gaps (recorded, not hand-waved)

- RTA (per-core unique args) words are sampled on ONE core per kernel group
  per dump; CRTAs (the binding carriers, W31) are fully covered via
  multicast-identical content. A per-core stale RTA that is never re-written
  would be missed on unsampled cores — but RTAs are re-staged by per-launch
  dispatch commands every launch, so "never re-patched" cannot apply to them.
- Device-resident state outside the kernel-config region (compile-time
  constants, semaphore/DFB descriptors) is NOT covered by this instrument —
  that is exactly the W34 scope.
- `w33-echo-11` aborted after the strike window (exit 134); the sampled
  surface covers init through the strikes and the following probes, which is
  the deterministic clobber region. A full-run leg is not offered as
  evidence.

## Instrument stability note (cost of this wave)

Host L1 readback on this Blackhole + luwen stack is unstable while the queue
runs; legs 1-10 died or wedged before covering the strike window:

- legs 1-5 (SIGBUS at ~74 s, engine init): UMD `read_block_reconfigure`
  host window teardown racing concurrent transfers; a throw on an
  untranslatable core (e.g. `0-0`) preceded every crash. All five ran a
  partially-reverted build (a failed edit misread as applied) — treated as
  instrument-development legs, not evidence.
- leg 6: `finish()` inside `enqueue_mesh_workload` under the API lock
  self-deadlocks (hung 30+ min at program-cache setup). Fixed by moving the
  flush before the lock; then superseded by removing `finish()` entirely.
- leg 7: full-grid sweep (~90 cores x 296 words per launch) wedged the board;
  recovery needed `rmmod`/`modprobe tenstorrent` plus `luwen reset` (the
  post-wedge KMD `reset_device` ioctl returns failure until the module
  reloads).
- leg 9: `finish()` drain wedged the pipeline mid-init (lseq ~3160).
- leg 10: aborted (ARC msg-queue timeout) at canary probe #46, short of the
  strike window.
- leg 11 (EVERY=256, no finish(), range-walk single-core probe): full strike
  window covered. This is the recording leg.

The final instrument shape (deferred single-core readback, 16-byte chunks,
skip-list, EVERY=256, no drain) is the only shape that survived the strike
window; the abort it still triggers late in the run is recorded here rather
than hidden.

## Raw artifacts

`/tmp/w33-evidence-raw/` (all legs' logs), `/tmp/tt-w33-canary/` (canary
probe dumps), `/tmp/w33-pin-backup.patch` (the pin diff as built). Pin
commit: the W33 instrument. Engine commit: this evidence directory only.
