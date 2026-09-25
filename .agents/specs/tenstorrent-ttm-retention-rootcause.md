# Spec: root-cause the per-prefill DRAM retention inside the pinned tt-metal

Row: `BACKEND-TENSTORRENT`. State: DRAFT (2026-09-25).
Issue: `ISSUE-LOCAL-01M3918K0WTSCZ2X1S2WZA5NHW` (this row closes it).
Follow-up to: `tenstorrent-27b-dram-growth.md` (ledger) and the refuted
first attempt `tenstorrent-keepquant-retention.md` (#3303's record).
Git integration: one pull request (spec + root-cause evidence + fix if
the cause is in our seam, else the upstream-report package), branch
`row/TT-METAL-RETENTION-ROOTCAUSE`.

## Problem

The 27B decode retains ~950 MB of DRAM per NEW request served and dies at
~5 requests per process (`bank_manager.cpp:495`). The committed ledger
(docs/bench-evidence/tt-27b-dram-ledger-20260924.md) attributes the
retention to per-prefill-event accumulation in the keepquant decode path,
retained OUTSIDE our slot table. The first fix attempt (request-invariant
plane extents) produced an IDENTICAL staircase and the decisive control —
equal retention on the same shape twice — refuting the pins-per-new-shape
program-cache theory. What remains: per-prefill-EVENT retention, owner
unknown, now attackable against the PINNED, CONFIRMED tt-metal
(`vllm-cpp-pin/20260925`: base `9161e8fdb27` + the 4-patch series,
built via the official `build_metal.sh`, confirmed by the 92/92 suite).

## Scope

Root-cause, with the fix following the cause:

1. **Instrument the pinned tt-metal directly.** The tree is OURS, built
   from source at `~/Sources/tt/tt-metal-pin` — allocator/program-cache
   logging can be added locally (temporary instrumentation, not a
   product change): log every allocation's call site class (the
   bank_manager's allocation path), and every FREE, per prefill event.
   The question the ledger cannot answer from outside: WHICH tt-metal
   structure accumulates per prefill — cached program workspaces,
   the allocator's free-block fragmentation bookkeeping, the
   `GraphCapture`/program-cache tensor registry, or something our
   decode seam holds through tt-metal (e.g. tensors captured in the
   graph and re-referenced per prefill).
2. **Candidate list from the evidence** (each to confirm or exclude
   with the instrument): (a) the program cache retaining
   per-CAPTURE-CYCLE tensor buffers under `VT_TT_PROGRAM_CACHE=1` —
   note `Qwen3_5DenseDecodeGraph`'s capture runs per batch-shape, and
   each PREFILL may trigger a capture-cycle entry; (b) the ledger's
   own hint — the retention is tagged `kq-decode/repair` and lands on
   the request's FIRST decode steps, so the repair web's per-request
   tensors may be registered with tt-metal's cache in a way our census
   cannot see; (c) allocator-level growth (free-list split) driven by
   the per-request allocation PATTERN rather than retention.
3. **Fix follows the cause.** If the owner is our seam: the smallest fix
   on our side. If tt-metal-internal: the upstream-report package (a
   minimal reproducer outside vllm.cpp + the instrumented trace) AND, if
   a local workaround is safe (e.g. an explicit cache-release call after
   each prefill, or a prefill-bounded capture cycle), the workaround with
   its tradeoffs stated.

## Gates

1. The per-request ledger goes FLAT over >= 16 requests (the original
   spec's gate 1, unchanged) — with the CAUSE named in the evidence.
2. Anchor-prompt tokens byte-identical to the current default arm.
3. Device suite 92/92 / 525,723 unchanged.
4. Standard gates; the pinned tt-metal is the build base and the weekly
   task is the record of its state.

## Risks

- Instrumenting tt-metal's allocator may perturb timing but not
  semantics; the ledger instrument is the detector for flatness.
- A tt-metal-internal owner means this row's deliverable is the
  reproducer package + any safe workaround — the fix itself becomes
  upstream's, tracked by the weekly task's surprise-report path.

## Non-goals

- No INT8DOT sweep, no capture row on this branch — they reopen when
  gate 1 goes green.
- No changes to the pinned tt-metal's series; local instrumentation is
  temporary and reverted, and any tt-metal-side change that must persist
  becomes a patch in the recorded series.

## Stop conditions

- The instrument cannot attribute the growth (allocation sites are
  uniform): stop, report, and the row escalates to tt-metal upstream
  with the instrumented trace as the package.
- Any workaround that shifts numerics: stop (gate 2 is the detector).
