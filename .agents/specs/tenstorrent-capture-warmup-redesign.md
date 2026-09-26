# Spec: redesign the captured arm's warmup/staging path

Row: `BACKEND-TENSTORRENT-QWEN35` (with the TT backend as its execution
surface). State: DRAFT (2026-09-26).
Issue: `ISSUE-LOCAL-01M3918KQ580Z3NHVRNXVF15FZ` (this row closes it).
Follow-up to: `tenstorrent-qwen35-q4km-degen-fix.md` (the stopped
reconciliation; its record carries the instrumented falsifications).
Git integration: one pull request (spec + the four waves + gates), branch
`row/TT-CAPTURE-WARMUP-REDESIGN`. This is the bigger unit the stop
condition named.

## Problem

The captured decode arm has not completed a single real capture since
the `14d7a25077`/`2ed5e912e4` pair: the shadow restore silences the
capture (the conv slot reads unserveable, the warm gate resets every
step, trace demand 0 B — the "captured" arm runs all-eager) while also
freezing the GDN recurrence (state rolled back past the warmup's commit
— the period-3 degeneration). Removing the restore re-exposes the capture
defects it papered over, in an unbounded stack: the capture-branch
reshape spec divergence, then the pointer-keyed grouped-act staging
(warmup and capture allocate activations at different pool addresses —
the lookup structurally misses). The eager numerics are intact; the
capture path's warmup/staging design is what is broken.

## The four waves (sequential, each with its own gate leg)

**W1 — the conv slot leaves the snapshot/restore.** Drop the conv slot
from any snapshot/restore so `ConvShadowServeable` stays true and the
warm gate does not reset. Gate: the decode-graph cold step reports a
non-zero trace demand (a capture actually begins) and the conv shadow
serves.

**W2 — the ssm restore becomes value-preserving.** Either copy the
warmup's committed values into the restored geometry, or serve the
warmup's commit directly (no rollback). Gate: the recurrence advances —
the red's period-3 signature is gone at the instrumented state probe
(the warmup's state update survives into the next step).

**W3 — grouped-act staging becomes device-resident in-region.** Replace
the #3042 pointer-keyed `GroupedActShadows` staging with device-resident
in-region activation serving (the GDN device-pure pattern) or persistent
staging buffers whose addresses are stable across warmup and capture.
Gate: the `grouped-quant: activation staging miss during trace capture`
refusal is gone — the capture pass finds its activations.

**W4 — the full re-audit and the decisive gates.** Re-audit every
capture-pass spec against its warmup counterpart until the trace
COMPLETES (each divergence gets its own fix in this wave; the stack is
bounded by the audit, not by hope). Then: (1) the q4km TT-lane capture
is BYTE-IDENTICAL to the committed golden; (2) the device suite 92/92 /
525,723; (3) the 27B APEX anchor re-derived (its committed evidence
embeds the degenerate all-eager reference — the new capture's TPOT and
token stream become the fresh anchor evidence, with the old numbers
explicitly superseded in the record); (4) standard gates.

## Risks

- W3 is the load-bearing design change: in-region activation serving
  must not change eager numerics (the eager path keeps the same values
  it has today — byte-identity gates it).
- The W4 audit may find more capture-pass divergences (the stack was
  "unbounded from here"); each is in-scope for this row, fixed one at a
  time, and if any one of them is a DESIGN-level impossibility (not an
  implementation gap), the row stops and records it — same discipline as
  the previous row.
- The capture/replay behavior the original pair fixed (the
  mid-capture reshape fatals) must stay fixed — W1-W4 may not regress
  the capture-safe reshape discipline; the capture-lane suite cases are
  the regression detector.

## Non-goals

- No tt-metal changes; no golden updates (the goldens stand as the
  byte-identity target).
- No INT8DOT sweep, no TPOT claims — they reopen/derive fresh after W4.

## Stop conditions

- Any wave's design-level impossibility (recorded with the instrumented
  evidence; the row stops rather than forces).
- W4's byte-identity unreachable after the trace completes: stop and
  record — at that point the capture runs and the residual divergence
  is a separate, smaller numerics row.

## Now

2026-09-26: the four waves LANDED on branch `row/TT-CAPTURE-WARMUP-REDESIGN`
(spec 6e1f80b0b; W1 a94a348ce, W2 1a5dff77d, W3 9b9bd6dfe, W4 c0ee3a41a,
suite repair 24c6ccc22). The captured decode arm COMPLETES its trace for
the first time since the `14d7a25077`/`2ed5e912e4` pair: the q4km battery
reports "device trace demand at last capture end = 436,273,152 B" (the red
printed 0 B), 16/16 prompts, and the captured token stream is
byte-identical to the all-eager stream — the capture-vs-warmup spec audit
passes at the token level. The device suite holds 92/92 / 525,723 (the
first W3 cut's 13 keep-quant capture failures — the op-level tests'
host-staged activation pattern — are repaired by the narrowed fallback
and all clear green). The byte-identity gate stopped at its spec's stop
condition: the residual 51-cell divergence is the committed golden's own
in-band deviation from the oracle (the current stream matches the oracle
at all 51 cells; the golden matches at none), superseded by justified
numeric drift accumulated since the golden's capture — recorded in
ISSUE-LOCAL-01M3918KQ580Z3NHVRNXVF15FZ and moved to
ISSUE-LOCAL-01M3ENS1FCHZ7SR5SJXQM174A2 as the separate, smaller numerics
row. The 27B APEX anchor is re-derived on the fixed tree (its old
  numbers — the degenerate all-eager reference — are superseded in the
  issue); the audit's 27B finding — the int8-dot kernel's activation
  refusal — is fixed with the same in-region serve (9b18e6ab5), and the
  27B captured arm CAPTURES for the first time since the pair, with the
  loop stream's oracle adjudication recorded as owed to the 27B's own
  row.

## Outcome

- Measured (P150, pinned tt-metal vllm-cpp-pin/20260925, all legs under
  the file mutex): red baseline dump md5 6ae141e5e4184e48071a7b3413d8df83
  (period-3 279/6511/314, 188/256 cells diverged, trace demand 0 B); the
  fixed captured battery dump md5 c403afe36f31b324ba22216136be66e2 ==
  the all-eager dump (both arms byte-identical), trace demand 436,273,152
  B, 16/16 prompts (13/16 strict-exact vs the per-prompt oracle greedy,
  all inside the 500-mnat band).
- Rejected: keeping the ssm snapshot/restore with a value-copy (the
  spec's other W2 option) — the warmup's scatter commit already lands at
  the exact geometry the capture's first read serves, so the rollback
  bought nothing and discarded the state update; serving the commit
  directly is both simpler and semantically correct. Rejected: routing
  CaptureSafeReshape through the member-view branch during capture (the
  14d7a25077 design) — the relabeled padded shape is a spec no eager
  warmup produces, which is the mid-capture program-cache miss; one free
  reshape in both passes makes the spec identical by construction and
  turns a cache miss into the audit's loud divergence detector.
  Rejected: removing the pointer-keyed activation cache outright (the
  first W3 cut) — the op-level capture tests' host-staged pattern is
  legitimate and pointer-identity-sound there; the cache survives
  NARROWED to that pattern (the engine's pool-recycled activations
  structurally miss it and hit the by-name refusal, which is the fix).
- Defaults and their values: the conv and ssm slots both stay with the
  warmup's commits (no snapshot/restore at all — the API is gone); the
  grouped-quant activation serve is primary and the host-staged cache is
  the fallback; VT_TT_GDN_STATE_PROBE ships as the row's instrumented
  state probe (env-gated, zero cost when off).
- Owed: ISSUE-LOCAL-01M3ENS1FCHZ7SR5SJXQM174A2 — the golden's 51-cell
  in-band deviation from the oracle (the byte-identity residual): the
  per-commit drift attribution and the capture pair's re-derivation and
  ratification are that row's work.

