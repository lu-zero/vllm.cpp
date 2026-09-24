# Spec: keep the keepquant repair web out of the program-cache retention

Row: `BACKEND-TENSTORRENT`. State: DRAFT (2026-09-24).
Issue: `ISSUE-LOCAL-01M3918K0WTSCZ2X1S2WZA5NHW` (this row closes it).
Follow-up to: `tenstorrent-27b-dram-growth.md` (the attribution).
Git integration: one pull request (spec + fix + gates), branch
`row/TT-KEEPQUANT-RETENTION`.

## Problem

The 27B decode retains ~950 MB of DRAM per new request served and dies at
~5 requests per process (`bank_manager.cpp:495`, the documented OOM
signature). The committed ledger
(`docs/bench-evidence/tt-27b-dram-ledger-20260924.md`) attributes the
growth: the keepquant signed-zero repair web
(`tenstorrent_keepquant.cpp:440-456`, the `repair` lambda: `gt` /
`where` / `to_layout` planes) allocates per-call TILE tensors whose
EXTENTS vary with each request's word shapes; under
`VT_TT_PROGRAM_CACHE=1` tt-metal's program/op cache pins the buffers of
every new (op, shape) pair, so each new request's shapes add ~950 MB of
cached, never-released device tensors. Our slot table is clean (census
flat); the retention owner is the cache.

## Fix design

Make the repair web's device tensor extents REQUEST-INVARIANT, so the
program cache holds ONE entry per (op, encoding) instead of one per
request:

1. **Fixed-extent planes.** Each encoding (Q4_K, Q5_K, ...) has a
   bounded word geometry: pad/slice the `value`/`mask` planes to the
   encoding's maximum tile extent (the existing per-word tables already
   bound it), run the repair at that extent, and slice the result back.
   The program cache then sees a fixed shape per encoding.
2. **Slot-table allocation where the Warm\* pattern applies.** If a
   plane can be pre-allocated once per encoding at warm-up (the
   `WarmDecodeIds`/`WarmRacIdx` precedent: allocate on first encounter
   of the shape key, refresh in place after), prefer that over
   per-call allocation entirely.
3. `TTReclaimPlanes` stays: reclamation of the incoming planes is
   orthogonal to the cache retention.

Mechanism choice (1) vs (2) per site is the implementer's call with the
code in front of them; the gates decide, not the aesthetic.

## Gates (all required)

1. **The ledger goes flat.** The DRAM-ledger leg re-run (16 requests,
   c=1, `VT_TT_ALLOC_TRACE=1`): settled free DRAM at request 8 within
   noise of request 1 — no per-request staircase. The 27B serves ≥16
   requests per process without OOM.
2. **Byte-identical outputs.** The anchor-prompt tokens and the oracle
   TF dump hash match the current default arm exactly — padding/slicing
   is arithmetic-neutral and the gates prove it.
3. **Device suite** 92/92 cases / 525,723 assertions unchanged.
4. Standard gates: `check-agent-record`, style, trailers.

## Risks

- **In-place reuse vs op semantics**: `ttnn::where` allocates its output;
  converting to an out-variant or a copy-back is where numerics could
  drift. Gate 2 is the detector; any byte-diff stops the row.
- **Fixed extents cost memory up front** (the max extent per encoding is
  resident). Bounded and small relative to the 950 MB/request it
  removes; the census states the cost.
- **tt-metal may pin per-call buffers regardless of shape** (workspace,
  not IO). If the ledger stays staircase-shaped after the shapes are
  invariant, that is the stop condition: escalate to the pinned-tt-metal
  row with the ledger — the attribution did its job either way.

## Non-goals

- No `VT_TT_PROGRAM_CACHE` default change; no tt-metal source change.
- No INT8DOT sweep here; it reopens on this row's green ledger.

## Stop conditions

- Gate 2 byte-diff that survives investigation → stop; the in-place
  conversion is not safe for this op set.
- Gate 1 unreachable by shape invariance → stop, escalate per Risks.
