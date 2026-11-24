ID: ISSUE-LOCAL-01M7JYJ1WDJJ3M7E23VQCYS6KC
Title: TT paged-attention decode with the program cache DISABLED under rich suite state: value-level garbage at seq>=2 (bf16 re-read also wrong)
Row: -
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-16
Updated: 2026-11-24
Closed: -

## Problem

The TT-PA-SMALL-WINDOW W1 characterization's program-cache-disabled arm (ProgramCacheToggleForTest(false), the falsified cache-hit-run-args hypothesis's instrument) exposed a SECOND PA decode defect distinct from the fixed dtype-width one: in a fresh process the arm is green (0 garbage / 8 seqs, w3-all-arms-GREEN.log), but inside the full suite (80+ cases of allocator/program history ahead of it) it fails deterministically at seq=2..8 with VALUE-level garbage — the bf16 re-read of the output buffer shows the same 1e33-magnitude wrong values as the f32 read (w22-consume-gdn-suite2.log: 'seq=2: max_abs=1.91669e+33, bf16-reread max_abs=1.9167e+33'), so the computed values themselves are wrong, not the write width. The program cache is enabled by default in production and the engine never disables it, so this is out of the production path's reach today; it is recorded because the characterization arm that found it ships in the suite and stays red there. Owed: root-cause the cache-disabled decode's value corruption under rich state (suspects: the per-call recompile's kernel-arg/scratch reuse), fix or gate the arm by the finding.

## Resolution

-

## Triage (2026-11-24, against the W34 staging-temp recycle conviction)

The W34 freeze bisect (docs/bench-evidence/qwen38-dflash2-stage-c-w34-20261123/)
causally convicted the staging-temp recycle mechanism: a freed
`Bfp8TypecastInstall` staging temporary's DRAM base stays the target of an
async device writer, and the next allocation on that base is silently
corrupted. The persistent per-(rows, cols) staging scratch (666d89959) removes
it.

**Does the recycle mechanism explain this issue? Weaker fit than the RAC
issue — plausible but not the leading hypothesis.** Points of contact: the arm
disables the program cache, which maximizes per-call allocation/free churn
(each call recompiles and allocates fresh program/device memory), and rich
prior allocator history is required — the same context dependence as the
recycle strikes. The bf16 re-read showing the SAME 1.9e33 values as the f32
read rules out a write-width artifact, but a recycled base read as an input
(computed-from-garbage, then stored correctly) also produces identical wrong
values in both widths, so the "computed values themselves are wrong" wording
does not exclude a stale-input cause.

Against it: the issue's own record says the corruption appeared ONLY with the
cache disabled — the production path (cache enabled) is clean, and the W34
mechanism was convicted on the DEFAULT path's behavior. If staging-temp
recycle were the driver, the cache-enabled arm should be at least as exposed
(more staging, more churn), yet it is green. That asymmetry points back at the
cache-disabled arm's own difference: the per-call recompile's kernel-arg or
scratch reuse, the issue's original suspect.

**Minimal falsification/verification experiment:** in the suite context,
rerun the cache-disabled arm with the staging scratch fix default-on AND add a
freeze on the arm's per-call device allocations (retain every temporary the
recompile path allocates, the w34-freeze analogue). Prediction if recycle is
the cause: seq>=2 garbage disappears under the freeze. If the freeze does not
clear it, dump the recompile path's kernel-arg buffers before each call and
compare against the call's inputs (the W31 echo discipline): an arg buffer
carrying a stale base convicts the arg-reuse suspect directly.

No fix implemented this session; the experiment above is the next action.
