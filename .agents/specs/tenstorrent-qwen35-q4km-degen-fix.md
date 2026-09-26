# Spec: reconcile 14d7a25077's capture-safety with the eager decode numerics it broke

Row: `BACKEND-TENSTORRENT-QWEN35`. State: DRAFT (2026-09-25).
Issue: `ISSUE-LOCAL-01M3918KQ580Z3NHVRNXVF15FZ` (this row closes it).
Git integration: one pull request (spec + fix + gates), branch
`row/QWEN35-Q4KM-DEGEN-FIX`.

## Problem

The 0.8B q4km keepquant decode regressed to a degenerate period-3 loop
(`279, 6511, 314` repeating; 188/256 cells diverged from the committed
TT-lane golden). Git bisect on the pinned tt-metal named `14d7a25077`
("fix: capture-safe reshapes and zero-cache warmup for Tenstorrent
decode graph") first-bad (PR #3313's record). The committed goldens are
proven right: vllm at the capture-era commit on the confirmed pin
reproduces them byte-for-byte.

## Root-cause targets (read 14d7a25077 first)

`14d7a25077` landed TWO classes of change; determine which broke the
eager path:

1. **Zero-cache warmup**: warming caches with zeros before capture. If
   the eager/first-decode path now reads or inherits state through a
   zero-warmed cache instead of real values, the GDN recurrence freezes
   — exactly the period-3 degeneration signature.
2. **Capture-safe reshapes after NormalizeDevF32Tile** (the
   `455a2de84`-era reshape discipline): a reshape/copy semantics change
   on the tile-normalization output that alters eager numerics.

The fix reconciles the commit's capture-safety intent with the eager
path: whatever the warmup/reshape ordering must be for capture, the
EAGER decode must produce the pre-14d7a25077 stream. Read the commit's
own context (its spec/records: what defect did it fix? — the capture-era
TT_FATAL and the reshape mid-capture work) so the repair preserves that
behavior; its original gates must stay green.

## Gates

1. **The decisive gate**: the q4km TT-lane capture at current main+fix,
   against the pinned tt-metal, is BYTE-IDENTICAL to the committed
   golden (`tests/parity/goldens/qwen35_gguf_q4km/our_ids_tenstorrent_capture.i32`
   — recipe: the committed fixture + VT_DUMP_IDS bootstrap; the
   committed stream is the reference, unchanged).
2. The device suite 92/92 / 525,723 unchanged.
3. `14d7a25077`'s own purpose stays intact: the capture/replay behavior
   it repaired still works (its era's tests, the capture-lane suite
   cases) — name them in the evidence.
4. Standard gates (record, style, trailers).

## Risks

- The zero-warmup may be load-bearing for capture (the tt-metal
  !is_capturing_trace constraint) — the reconciliation may need a
  capture-vs-eager conditional rather than a revert; state the split
  explicitly in the fix.
- The 27B APEX anchor may or may not have been affected (its stream was
  never compared to a pre-14d7a25077 reference): if the fix changes 27B
  tokens too, capture before/after and report — the anchor's committed
  evidence post-dates the regression, so its reference is
  post-regression; flag any shift loudly rather than hide it.

## Non-goals

- No tt-metal changes; no golden updates (the goldens stand).
- No INT8DOT sweep here; it reopens on gate 1 green.

## Stop conditions

- If the reconciliation cannot preserve capture-safety (its original
  defect resurfaces), stop and report — the fix splits into a
  redesign of the warmup path, a bigger unit.
- If gate 1 cannot reach byte-identity without reverting capture-safety
  wholesale, stop with the evidence; the trade becomes a decision.

## Now

2026-09-26: STOPPED at the first stop condition, with the root cause
proven and the reconciliation evidence recorded in
ISSUE-LOCAL-01M3918KQ580Z3NHVRNXVF15FZ (## Resolution). The degeneration
is the `2ed5e912e4` snapshot/restore inside the bisected pair (the frozen
GDN recurrence plus the conv-restore cascade that kept the "captured" arm
silently all-eager — trace demand 0 B). Removing the restore re-exposes a
stack of capture-arm defects (the CaptureSafeReshape member-view spec
divergence, then the #3042 pointer-keyed grouped-act staging, which
structurally misses between the warmup and capture pool addresses). The
fix splits into the warmup-path redesign named above; the issue's
Resolution lists the next unit's four steps.
