# QWEN38-DFLASH2-STAGE-C W31 — the in-engine launch-address trace (2026-11-22)

Row: `QWEN38-DFLASH2-STAGE-C`, hunt wave W31, worktree `/tmp/tt-w31`
(branch `row/QWEN38-DFLASH2-STAGE-C-HUNT-W31`, base `161e9865a` W30), build
`/tmp/build-stagec-w31`. Issue `ISSUE-LOCAL-01M7FY5W1A91XHV1PZDGZEKR7H`.

## The instrument (tt-metal-pin patchset, on top of 611e346884f + the pin's
dirty W23 TA move / W29 write_trace, build `~/Sources/tt/tt-metal-pin/build_Release`)

- `[TT-TA]` (TT_METAL_TA_DEBUG=1, pre-existing W23 instrument): per cached-program
  launch, every patched tensor-binding address. W31 extension: unbound kernels
  (`binding_handles=EMPTY` — UpdateTensorArgs patches NOTHING, every cache hit
  reuses the miss-time addresses, the APEX zero-logits mechanism) now dump
  their retained CRTA words (64) once per kernel name — those words ARE the
  permanent stale addresses such a kernel keeps writing.
- `[TT-RTA]` (TT_METAL_RTA_DEBUG=1, new W31 instrument in
  `tt_metal/impl/kernels/kernel.cpp:807`): every legacy per-core
  `SetRuntimeArgs` value — the address class the CRTA instrument cannot see.
- Both default-off, diagnostics only, inert in production.

## Legs (run-w31-trace.sh; Q4_K_M KV64 drafter-ON, VT_TT_BFP8_RESTAGE=0 +
VT_TT_BFP8_CANARY=1, output-len 12 — the deterministic W28 repro)

- `w31-trace-1` (TA only, run `run-w31-trace.sh w31-trace-1`): strikes
  reproduce at EXACTLY #52/#58/#61
  (`verdict=ARRIVED-ZEROED` x3, pre=8 post=8, probe underreports 1/8) — the
  writer is live and deterministic under the instrument. Victim bases:
  0x231d6f00 / 0x2a146f00 / 0x2d7cef00 (identical to W28).
- `w31-trace-2` (TA + RTA, 248k stderr lines; the committed
  `w31-trace-1.canary.log` / `w31-trace-2.canary.log` carry the canary/stage/
  scan line subsets, the analyzers' full match lists are
  `w31-analyze-1.txt` / `w31-analyze-2.txt`; the raw 11/17.5 MB stderr logs
  are machine-local scratch outside the repository, path recorded in the
  wave session notes, NOT lost evidence): same three strikes, same
  bases, same token stream sha256 as leg 1 (5076c7f5...).

## Finding: the writer is NOT on any kernel launch path

Per-bank extent of a victim = size/8 = 0x3FC000 (pages round-robin
bank = page_index % 8, W29's probe comment). Scanning EVERY `[TT-TA]`,
every `[TT-RTA]`, and every EMPTY-bindings CRTA dump of both legs for an
address inside each victim window:

- the ONLY hits are the victim's OWN staging typecast (`writer/output`,
  clean at its stage check) and the victim's OWN matmul reads
  (`in1_sender_writer/in1`) — launches at the flip line and at later reuse;
- no cached unbound kernel retains a victim-window address in its CRTA;
- no legacy RTA ever carries a victim-window address (the one first-pass hit,
  `gather_writer_single_row_single_core` @ 0x2a57ef00, is OUTSIDE the true
  per-bank window — it is the next buffer at +0x14000 past the extent).

So the 512 zero-header bytes are written by an agent that never receives the
victim address through UpdateTensorArgs or SetRuntimeArgs. The launch-path
candidates are exhausted; the writer is:

1. a stale DISPATCH-level command replay (a command sequence built at a
   program-cache miss that embeds an old staging-temp base and is re-executed
   after that base is freed and recycled onto the staged weight — no current
   launch arg shows it because the addresses were baked into the command
   stream, not the program state), or
2. an on-device agent computing its write address from a stale descriptor.

This refines W28's class (a): the stale address is not handed out by our
per-call staging via a patched binding — the trace shows every patch is
correct. The next instrument (W32) is the dispatcher command-stream trace:
log, at host command CONSTRUCTION in the fast-dispatch path, every DRAM noc
address embedded in write commands, and match against the victim windows the
same way — the writer's command will carry the victim base with no
corresponding live launch.

## What did NOT reproduce

Nothing new needed exonerating: the W30 seam bisect stands (writer does not
reproduce outside the engine); the TA/RTA instruments themselves do not write
(the strikes reproduce identically across legs, and the direct probe is
slice-free).

## Status

- NOT fixed: no root-cause repair this wave; the DNR stays DEFAULT-ON and the
  gate suite is not re-run (no behavior change to production paths; the
  instruments are env-gated pin-side only, zero `src/` changes in this tree —
  the hunt-discipline CPU-target link is therefore not owed this wave).
- Pin working tree state: 611e346884f + uncommitted W23 TA move + W29
  write_trace + the W31 hunks (program_run_args.cpp CRTA dump,
  kernel.cpp RTA trace) — the pin patchset is not committed here (the pin
  checkout predates this wave dirty; committing it is left to the pin's
  owner, the same shape as W29's committed 611e346884f).
