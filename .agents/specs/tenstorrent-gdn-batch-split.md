# Spec: decompose GDN prefill/decode batches past the tt-metal core bound

Row: `BACKEND-TENSTORRENT`. State: DRAFT (2026-09-25).
Issue: `ISSUE-LOCAL-01M3AF30AE0KR6M7JN8D9BA46B`.
Git integration: one pull request (spec + fix + gates), branch
`row/TT-GDN-BATCH-SPLIT`. NOTE: merges after #3300 (which carries the
issue file).

## Problem

On tt-metal `81f3bbf3b405`, any GDN prefill/decode with
`batch * value_heads > 110` (the P150's compute-core count) asserts at
program build: `chunk_gdn_phased_program_factory.cpp:137` requires
`BH <= ncores`. The 27B APEX decode at c>1 wants BH 480/192 and dies at
load — concurrency above 1 is unreachable on the 27B decode lane, which
blocks the M=2 batching result (+43% aggregate) and every multi-request
serving shape.

The committed analysis (ISSUE-LOCAL-01M3AF30AE0KR6M7JN8D9BA46B's
sources): BH is caller-supplied — our `GdnPrefillKernel`
(`tenstorrent_gdn.cpp:310`, batch from `q_in.shape[0]` at :629) and the
decode lane (:759) pass the whole varlen batch as dim-0, and
tt-metal's `distribute_scan` hard-requires `BH <= ncores` with no
override API. Upstream tt-metal `main` carries the identical assert.

## Fix design

Batch decomposition in OUR code, at both call sites
(`tenstorrent_gdn.cpp:310` prefill, `:759` decode):

- When `batch * hv > 110` (query the device's compute-core count, do not
  hardcode), split into sub-calls of `sub_b = max(1, 110 / hv)`
  sequences: slice q/k/v/g/beta/s0 along dim 0, call
  `chunk_gated_delta_rule` per sub-batch, concatenate the outputs and
  final states before the existing per-sequence scatter loop.
- Splitting is semantics-preserving: the op factorizes exactly over the
  flattened (batch, head) dim — the scan recurrence is per-row, final
  state is one row per (batch, head), and per-sequence initial states
  slice along dim 0.
- If `hv` ALONE exceeds the core count, the op is unusable at any batch:
  refuse by name with the numbers (the message names the missing part).

## Gates

1. The 27B decode LOADS and SERVES at c=4 and c=16 on the P150 (the
   shapes that died at load), producing the ledger's token outputs.
2. c=1 outputs are byte-identical to the pre-fix arm (the decomposition
   must not engage at c=1 — sub_b >= batch means a single call).
3. Device suite 92/92 / 525,723 unchanged.
4. Standard gates: record, style, trailers.

## Risks

- Per-sub-call program-cache entries multiply with distinct sub-shapes;
  bounded (sub_b is fixed per hv), and the retention row's ledger
  instrument is the detector if it misbehaves.
- Concat cost is linear and host-side; the M=2 aggregate win is measured
  AFTER this lands, as its own number.

## Non-goals

- No tt-metal source change; no upstream PR here (the analysis text in
  the issue is the escalation material if the fleet wants it).
- No batching-throughput claim on this branch; it restores the
  capability.

## Stop conditions

- If the decomposition cannot slice `s0` cleanly (the initial-state
  layout resists dim-0 slicing), stop and report — that changes the
  design.
- Byte-diff at c=1 → stop; the claim that c=1 does not engage the split
  is falsified.
