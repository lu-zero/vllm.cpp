# Spec: split tenstorrent_ops.cpp into per-module translation units

Umbrella spec (owns the rowless split issue). State: DONE (2026-09-24).
Git integration: staged — one module per PR, each with this spec referenced;
each stage's PR body names which stage it lands.

## Problem

`src/vt/tenstorrent/tenstorrent_ops.cpp` is 10,445 lines / 501 KB and
grows with every keep-quant and GDN wave. It carries at least ten
distinct modules separated by banner comments. Every fix requires
reading past unrelated sections; merge pressure concentrates in one
file — the shared-file-lock failure mode at TU granularity.

## Target layout

```
src/vt/tenstorrent/
  tenstorrent_residency.cpp     # slots, SlotMutex, upload/memset, counters
  tenstorrent_weights.cpp       # weight views, embedding shadows, KQ word shadows
  tenstorrent_keepquant.cpp     # KQ decode, grouped, int8-dot kernel + dispatch (~3.2k)
  tenstorrent_paged.cpp         # PagedKv shadows, PA/RAC/DecodePos warm paths
  tenstorrent_gdn.cpp           # GDN prefill/decode/state/conv (~2.1k)
  tenstorrent_capture.cpp       # mesh-trace capture + W4d alloc trace
  tenstorrent_ops.cpp           # registration + small glue (< 2k)
  tenstorrent_internal.h        # shared internals (shadow maps, mutexes, helpers)
```

`kernels/` and `tenstorrent_device.h` untouched.

## Rules (bind every stage)

1. MECHANICAL ONLY: code moves along the existing banner seams byte-for-
   byte where possible; no behavior edits, no renames, no checker
   changes. Any edit a move forces (a static becoming internal-linked,
   a forward declaration gaining a header) is named in the stage PR
   body.
2. File-local statics shared across sections (shadow maps, mutexes,
   counters) move to `tenstorrent_internal.h` as inline functions or
   into `tenstorrent_residency.cpp` with declarations — named per
   stage.
3. Gates per stage: full backend suite unchanged on the P150 (77/77
   baseline), 0.8B vehicle battery unchanged where the stage touches
   its path, record-anchor checker repaired in the same PR (the matrix
   and specs cite many `tenstorrent_ops.cpp:NNNN` anchors — each moves
   to its new TU).
4. One stage = one PR = one branch `row/BACKEND-TENSTORRENT-SPLIT-<n>`;
   the stage list below is the landing order. A stage may be dropped or
   reordered with a spec amendment; silence is not an exception.
5. Blame/`git log -S` continuity: prefer `git mv`-style moves; where a
   function splits from its statics, note it in the PR body so future
   historians know the seam.

## Stages

1. `tenstorrent_internal.h` + `tenstorrent_residency.cpp`: extract the
   foundation (slots, mutexes, shadow-map accessors, upload/memset,
   staging counters) and convert the file-local statics they own to
   internal declarations. Largest ripple; done first so later stages
   are plain moves.
2. `tenstorrent_keepquant.cpp`: the 1942–5187 banner block.
3. `tenstorrent_gdn.cpp`: the 6577–8740 banner block.
4. `tenstorrent_paged.cpp`: 1084–1768 + 5188–6576.
5. `tenstorrent_capture.cpp`: 8741–8997 + the W4d alloc-trace section.
6. Residue: weights/embedding staging stays with residency or moves to
   `tenstorrent_weights.cpp` if residency alone is still > 2k lines;
   ops.cpp keeps registration + glue.

## Gates / stop conditions

- Per stage: suite green, anchors green, no behavior delta (the suite's
  bit-exact sweeps are the detector).
- A stage that forces a behavior edit stops and splits: the behavior
  change becomes its own unit first.
- If two stages conflict structurally (a static spans them), merge the
  stages rather than introduce a cross-TU mutable global — one more
  internal-header accessor is fine; a new shared mutable is not.

## Owed

- Stage 1: internal header + residency extraction (this spec's first
  landing).
- Stage 2: keepquant extraction.
- Stage 3: gdn extraction.
- Stage 4: paged extraction.
- Stage 5: capture extraction.
- Stage 6: residue (weights staging placement; ops.cpp residual size).
- Tracked by `ISSUE-LOCAL-01M2M2PZC0E998B88C9XKJT8JJ` (the stages above
  are the issue's exit).

## Outcome

All six stages landed 2026-09-16..2026-09-24, one module per PR, each with
the device suite holding 92/92 cases / 525,723 assertions byte-for-byte
(docs/bench-evidence/tt-split{4,5,6}-gate-20260924.md and the earlier
stage records). `tenstorrent_ops.cpp`: 10,445 lines / 501 KB at the
spec's opening -> 1,483 lines of registration (33 RegisterOp calls,
unchanged), core op kernels, and glue. Final module sizes: residency
1,711, keepquant 2,089, gdn 2,272, paged 2,398, capture 472, backend
138, device 72.

What the stages decided that the opening plan did not: stage 3's GDN
extraction left the W1/W2 decode sections behind (moved in stage 6 after
symbol evidence tied them to the kL2Norm..kGdnStateScatter chain only);
stage 6 kept weights staging inside residency rather than creating
`tenstorrent_weights.cpp`, because residency stayed under the spec's own
2k threshold; the capture module (stage 5) turned out to carry the
allocation-trace instruments that the 27B decode DRAM-exhaustion
investigation (ISSUE-LOCAL-01M3918K0WTSCZ2X1S2WZA5NHW) now needs.

Rejected: automatic three-way merges of the roadmap/matrix records --
twice during the campaign, merges landed by other agents reintroduced
collapsed record rows over repaired ones (the MODEL-LAYA and MODEL-KEV
merges), and both were repaired in the stage branches per the
keyed-record rule. The mechanical-only rule held throughout: no stage
forced a behavior edit, so no stage split.
