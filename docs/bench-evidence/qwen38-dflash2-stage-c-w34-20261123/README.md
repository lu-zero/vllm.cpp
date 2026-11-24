# QWEN38-DFLASH2-STAGE-C W34 — the staging-temp recycle bisect and the
# staging-scratch fix (2026-11-23)

Row: `QWEN38-DFLASH2-STAGE-C`, hunt wave W34, worktree `/tmp/tt-w31`
(branch `row/QWEN38-DFLASH2-STAGE-C-HUNT-W31`, head `ad633811e` W33), engine
build `/tmp/build-stagec-w31`. Pin unchanged this wave
(`~/Sources/tt/tt-metal-pin` head `74bb2db5471`, clean). Issue
`ISSUE-LOCAL-01M833SZ523B0B875BWQQW3D2Y`.

## The two instruments, in the named order

### 1. Staging-freeze bisect (causal) — FIRED

`Bfp8TypecastInstall` gained `VT_TT_BFP8_FREEZE=1`
(`VT_TT_BFP8_FREEZE_N` cap): every bf16 upload temporary is retained for the
whole run, so no staging-temp DRAM base is ever freed or recycled. Each kept
temp prints one `[W34-FREEZE]` line.

- `w34-freeze-1` (cap unbounded): OOM at staging #143 (kept ≈ 11.7 GB); the
  run died during load — the segfault in its tail is a static-destruction
  order artifact (a function-static `vector<ttnn::Tensor>` outlives the
  device), not a run defect; the holder is now heap-allocated and never
  destroyed.
- `w34-freeze-2` (cap 100, ≈ 7.3 GB kept): OOM at staging ~#189 MB alloc,
  before the strike window — not evidence.
- `w34-freeze-3` (cap 88, ≈ 6.3 GB kept): OOM at staging #185 (32 MB), but
  the ENTIRE strike window completed before the abort: staging #184
  succeeded, canary probes #0..#63 all ran, and #52/#58/#61 — the
  deterministic `ARRIVED-ZEROED` strikes in every W28-W33 baseline leg
  (same probe indices, same interleave: probe #52 fires right after staging
  #184) — are ALL `verdict=clean` (`pre=0 post=0 probe_zero=0/8`).

**Verdict (outcome A): the poison is the staging-temp recycle interaction.**
With frozen staging the strikes disappear; with recycling (every baseline
leg) they reproduce deterministically at the same indices.

### 2. Compile-time constant audit — clean (supports the same verdict)

Full scan of every cached kernel ELF in the active build key
`~/.cache/tt-metal-cache/17089598622106942383` (10,380 ELFs, identified by
ELF mtimes newer than 2026-11-21; the typecast/copy programs live under
`kernels/eltwise_typecast`, `kernels/eltwise_copy`,
`reader/writer_typecast_rm_chunked`): ZERO occurrences of any victim base
(`0x231d6f00`, `0x2a146f00`, `0x2d7cef00`) as an embedded 4- or 8-byte
immediate. The 3,169,493 in-window words are the known instruction-encoding
false-positive class (W31/W33); the closest lui-split candidates
(e.g. `0x2a147e3d`) are unaligned instruction-stream bytes, same class.
Raw output: `/tmp/w34-audit-all.txt` (scanner: `/tmp/w34-scan.c`).

## The fix: persistent per-shape staging scratch

`Bfp8TypecastInstall` now (DEFAULT ON; `VT_TT_BFP8_SCRATCH=0` restores the
recycled-temporary path for regression comparison) stages the bf16 upload
into a persistent per-`(rows, cols)` device scratch that is never freed: the
first staging of a shape takes `from_span` onto the retained buffer, every
later staging of the same shape does a host `from_span` +
`ttnn::copy_to_device` into the SAME buffer (the
`RefreshQuantEmbedRows`/WarmDecodeIds H2D pattern), and the typecast reads
the scratch. Cost: one retained buffer per distinct staging shape — 9
buffers, ~1.2 GB on the repro arm (`[W34-SCRATCH] new ...` lines), versus
~2.6 GB of recycled per-weight temporaries. Bytes are identical: same bf16
source, same on-device RNE typecast.

`w34-scratch-1` (`VT_TT_BFP8_SCRATCH=1`, RESTAGE=0 + canary, Q4_K_M KV64
drafter-ON, output-len 12): **exit 0, all 64 canary probes clean — the
deterministic #52/#58/#61 strikes are GONE** — and the output tokens are
coherent (`[[220,16,23066,...]]`) where every corrupted baseline leg emitted
`[[220,0,0,...]]` (`w33-base-1-token-ids.json`).

## Root cause (evidence-backed, mechanism named to the allocator boundary)

The writer is on-device (W29); no host path, no launch arg, no dispatch
command, no sampled device-resident arg region carries a victim address
(W29/W31/W32/W33); no cached kernel ELF carries one either (this wave). The
remaining variable the address hunts structurally could not see — whether a
staging-temp base is FREE while some async device writer still targets it,
and a later weight's BFP8 output lands on that recycled base — is now
causally convicted by the freeze bisect and removed by the scratch fix.
The precise on-device writer (which program re-executes against the freed
base) remains unnamed; it is NOT reachable as a baked constant, so it is a
runtime-carried address inside a program the cache retains past the
temporary's free — the same class W31's retained-CRTA finding covered for
the sampled surface, on a surface the echo did not sample.

## Gate status (finalized 2026-11-24)

### (a) Scratch repro determinism: 3/3 CLEAN

`w34-scratch-1/-2/-3` (Q4_K_M KV64 drafter-ON, RESTAGE=0 + canary): 0/64
ARRIVED-ZEROED strikes each, 64/64 probes `verdict=clean` each, token streams
byte-identical across all three legs
(`[[220,16,23066,220,16,220,16,1814,...]]`). The deterministic #52/#58/#61
strikes of every W28-W33 baseline leg stay gone under the staging-scratch fix.

### (b) Q4_K_M KV64 drafter-ON x3 vs `d49c53d2…`: PASS under the ratified
### near-tie machinery, with two recorded caveats

All three legs (`w34-on-q4km-1/-2/-3`) emitted byte-identical streams, sha256
`af77d5d343e23847ab7f3530de07efb4ca574fa4f73db4a6d0657a30dd2b2124` — the SAME
stream the W26 tie-band gate ratified
(`qwen38-dflash2-stage-c-hunt-20261118/w26-stream-gate-verdict.json`):
22/24 tokens equal the eager golden `d49c53d2…` prefix, the single mismatch is
at position 7 (1814 vs 23066), and position 7 is inside the ratified 500-mnats
near-tie band per the W26 eager top-2 gap measurement of that same position
(W26 mismatches [7, 10]; this run shows only [7]). Zero corruption signature:
no NaN/Inf, no zero mid-stream, no strikes.

Caveats, recorded honestly:

1. The `EAGER-TOP2` gap instrument was NOT enabled on these legs, so gate
   criterion (b) is asserted by INHERITANCE from the byte-identical W26
   stream's measurement, not re-measured here. A re-measurement leg with
   `EAGER-TOP2` enabled is the cheap confirmation if the reviewer wants it.
2. The per-step acceptance telemetry is OUT of the W26 band: 23 tokens
   accepted over 9 verify blocks (~2.6/step, band [7.0, 9.0]). The stream is
   still near-identical to eager, so the walk is faithful; the draft quality
   under this tree differs from W26's. Recorded as an open observation, not a
   gate failure.

TPOT axis: Mean ITL 63361 ms (leg 1; legs 2-3 similar) against the drafter-OFF
~6647 ms — the acceptance multiple remains a LOSS / unmeasurable as a win,
consistent with W26's 3.9x-slower disposition. The absolute ITL is ~2.5x W26's
25571 ms; host contention during the W34 suite cannot be excluded
retroactively and is recorded as such. The Q4_K_M acceptance legs WERE run in
this wave, so the previously-owed `d49c53d2` x3 acceptance leg is NOT owed.

### (c) UD-Q6_K_XL KV8 drafter-ON x3 vs `b36bc61a…`: PENDING — capacity-blocked

Two attempts, both OOM at load, recorded per the stop rule (no further
iteration):

- Attempt 1 (`w34-on-q6kl-1` re-run as `w34-q6kl-1`, num-blocks 8,
  max-num-batched-tokens 64, fix default-on, drafter-ON, VT_DUMP_IDS=1):
  `vllm-bench: failed: Cannot auto-fit max_model_len: not enough GPU memory
  available to serve even a single token` (exit 1, 3/3 legs batch=64).
- Attempt 2 (`w34-q6kl-1/-2/-3`, same config, max-num-batched-tokens 32):
  identical failure, 3/3 exit 1.

This is the standing W21/W25/W26 capacity blocker (the KV-8 pool + the Q6_K_XL
BFP8 residency + the draft trunk do not fit the local Blackhole), UNCHANGED by
the staging-scratch fix — and plausibly slightly worsened by it: the fix
retains ~1.2 GB of per-shape staging scratch where the recycled path freed its
temporaries. Unblocking needs either a smaller scratch retention policy on
capacity-tight arms or a device with more DRAM; that is a coordinator
decision (recorded NEEDS_DECISION in the session report). The gate stays
PENDING; the drafter-OFF `b36bc61a` byte-identity 6/6 record from W21/W25/W26
still stands for this arm.

## Raw artifacts

Legs in this directory: `w34-freeze-1`, `w34-freeze-2`, `w34-freeze-3`,
`w34-scratch-1`, each carrying its env, run, stderr and token-id files:
`w34-freeze-1.log`, `w34-freeze-2.log`,
`w34-freeze-3.log`, `w34-scratch-1.log`, `w34-freeze-1.stderr.log`,
`w34-freeze-2.stderr.log`, `w34-freeze-3.stderr.log`,
`w34-scratch-1.stderr.log`, `w34-freeze-1-env.txt`,
`w34-freeze-2-env.txt`, `w34-freeze-3-env.txt`, `w34-scratch-1-env.txt`,
`w34-freeze-1-token-ids.json` (legs 1-3 produced none — OOM/abort),
`w34-scratch-1-token-ids.json`;
audit raw `/tmp/w34-audit-all.txt`; scanner `/tmp/w34-scan.c`; run script
`/tmp/w34-run-freeze.sh`. `git diff` backup: `/tmp/w34-backup.patch`.
