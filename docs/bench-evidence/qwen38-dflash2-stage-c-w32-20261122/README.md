# QWEN38-DFLASH2-STAGE-C W32 — the host command-construction trace (2026-11-22)

Row: `QWEN38-DFLASH2-STAGE-C`, hunt wave W32, worktree `/tmp/tt-w31`
(branch `row/QWEN38-DFLASH2-STAGE-C-HUNT-W31`, head `6ebcaf570` W31), engine
build `/tmp/build-stagec-w31` (unchanged from W31 — zero `src/` changes this
wave, so the hunt-discipline CPU-target link is not owed, same shape as W31).
Issue `ISSUE-LOCAL-01M7FY5W1A91XHV1PZDGZEKR7H`.

## The instrument (W32, named by W31)

A new env-gated host command-construction trace in the tt-metal pin
(`TT_METAL_CMD_TRACE=1`, default off, inert — one atomic load):

- one ordered `[TT-CMD] seq=… kind=bufwrite …` line at the HOST MOMENT every
  DRAM-noc-address-bearing dispatch write command is constructed:
  `add_dispatch_write_paged` (interleaved, both the alignment-prefix and the
  plain arm) and `add_dispatch_write_linear` (sharded), both in
  `tt_metal/impl/buffers/dispatch.cpp` — the txn noc address, the txn size,
  the buffer uid, and the buffer base;
- `kind=alloc` / `kind=free` lines for every DRAM buffer registration
  (`write_trace.cc` `register_dram`/`unregister_dram`, which
  `BufferImpl::allocate_impl`/`deallocate_impl` already hook), so every
  command line can be placed on the alloc/free timeline of the buffer whose
  address it carries;
- `kind=trace-begin/trace-end/trace-replay/trace-release` markers at the
  ttnn mesh-trace state transitions (`ttnn/operations/trace.cpp`).

Per-bank extent of a victim = size/8 = 0x3FC000 (pages round-robin
bank = page_index % 8, W29's probe comment). The analyzers
(`w32-analyze.py`, `w32-analyze2.py`) match every constructed command against
the three victim windows [base, base+0x3FC000), on both the txn address and
the buffer base, and reconstruct who owned each window at every seq.

## The pin patchset

Pin `~/Sources/tt/tt-metal-pin`: the W31 hunks (uncommitted there — the
README of W31 left them to the pin owner's flow) are now committed as
`8b6a60706d3` "vllm.cpp W31 launch-address trace", and this wave's
instrument rides on top as the W32 pin commit. Pin libs are copied into
`build_Release/lib64` (`libtt_metal.so`, `_ttnn.so`, `_ttnncpp.so` — ALL
THREE; the first leg ran with a stale `_ttnncpp.so` and silently lacked the
ttnn trace markers, which is why there are two legs with identical shape).

## Legs (run-w32-trace.sh; the deterministic W28 repro: Q4_K_M KV64
drafter-ON, VT_TT_BFP8_RESTAGE=0 + VT_TT_BFP8_CANARY=1, output-len 12)

- `w32-trace-1` (CMD_TRACE only; stale `_ttnncpp.so` — ttnn markers absent,
  tt_metal markers complete): strikes reproduce at EXACTLY #52/#58/#61,
  same bases 0x231d6f00 / 0x2a146f00 / 0x2d7cef00, token sha256 5076c7f5…
- `w32-trace-2` (CMD_TRACE with all three libs fresh): same three strikes,
  same bases, same token sha — 428,850 [TT-CMD] events
  (252,922 alloc / 94,951 free / 80,977 bufwrite / 0 trace events).

The raw 32 MB stderr logs are machine-local scratch at
`/tmp/w32-evidence-raw/` (W31's precedent); the committed
`w32-trace-1.canary.log` / `w32-trace-2.canary.log` files carry every
non-`[TT-CMD]` stderr line (the canary/serve/stage/step-phase subsets; the
committed subsets rewrite the engine's `#<seq>` line prefixes as
`seq<seq>` so they do not parse as issue references — the raw logs in
scratch are unmodified) and the analyzer outputs stand in for
the full match scans.

## Finding: the writer is NOT on any host dispatch-command path

Across the ENTIRE run (both legs, 80,977 constructed write commands):

1. **Zero constructed commands target any victim window.** No `bufwrite`
   command — interleaved paged or sharded linear, at ANY seq, including
   while a prior buffer owned the window — carries a txn address or buffer
   base inside [base, base+0x3FC000) of any of the three victims. The only
   occupant history of each window is (a) a big earlier staging buffer
   (per-bank 0xc80000, freed 5-12 seq before the victim's alloc) whose own
   writes carry its own base, and (b) the victim itself. W31's "stale
   command built at a cache miss embedding a staging-temp base" variant is
   therefore exonerated in BOTH sub-forms: no such command exists at
   construction time, and there is no queue-depth window in which one could
   fire late — nothing was ever constructed.
2. **No write overrun from a neighbor.** Every constructed txn's
   [addr, addr+size) end lies within its buffer's declared per-bank extent
   (0 uids with txn-end beyond alloc extent), so an adjacent-buffer spill
   into the window is also exonerated.
3. **The repro never captures a mesh trace.** Zero `trace-begin`/`replay`
   events, and the engine's own `[TT-STEP-PHASE] kind=eager-mtap` lines
   confirm the decode steps run eager (`replay_ms=0.0`). The
   trace-replay-bakes-stale-addresses variant is exonerated for this repro:
   there is no captured command stream to replay.

With W31 (no patched tensor binding, no legacy RTA, no retained CRTA carries
a victim address) and W29 (no host write path), every HOST-side surface that
could carry a victim-window address — launch args and dispatch commands
alike — is now measured clean. The writer is an ON-DEVICE agent (W31's
hypothesis 2), and the refined W32 fact narrows it: the write-set is exactly
one tile exponent header (64 B) at the victim's bank base on each of the 8
banks (512 B total, `W28-DIFF` hdr=512 data=0), which is the signature of a
BFP8 pack/copy writer's TILE-0 OUTPUT — i.e. an on-device writer whose
computed output noc address equals the victim base.

## Why the host-side instruments cannot see the device half

W31's `[TT-TA]`/`[TT-RTA]` dumps print the HOST-side retained values. A
cached program whose device-side runtime-arg (or CB/kernel-config) L1 region
was written ONCE at an earlier launch — with the staging-temp base — and
never re-patched because the host's current values "did not change" (or the
patch was skipped) would hold a stale DEVICE arg the host dumps cannot
observe. That stale-device-half is the remaining candidate.

## Named W33 instrument

Device-side runtime-arg echo: under an env gate, after every launch of the
suspect cached programs (the BFP8 typecast / copy writer programs whose
outputs are tile-header-shaped writes), read back the DEVICE-resident
runtime-arg L1 region for each placed core (the watcher/dprint-style
post-launch readback the host side cannot do) and scan those words against
the victim windows the canary records (engine passes the struck bases via an
env list, the same deterministic repro). A device arg word inside a victim
window with no host-side counterpart names the stale descriptor and the
program that owns it. Fallback arm if the readback is impractical on this
build: arm the pin's watcher on the writer kernels' noc writes.

## Status

- NOT fixed: no root-cause repair this wave; no behavior change to any
  production path (pin-only, env-gated diagnostics); the DNR stays
  DEFAULT-ON and the acceptance gate suite is not re-run.
- Pin working tree state: `8b6a60706d3` (W31 hunks) + the W32 instrument
  commit on top.
