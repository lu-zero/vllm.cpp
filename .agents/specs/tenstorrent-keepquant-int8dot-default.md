# Spec: flip `VT_TT_KEEPQUANT_INT8DOT` to default-on

Row: `BACKEND-TENSTORRENT-KEEPQUANT`. State: DRAFT (2026-09-23).
Issue: `ISSUE-LOCAL-01M37W0HP6JTNJP55S74159T02`.
Git integration: one pull request (spec, gate evidence, and the flip together),
branch `row/TT-KEEPQUANT-INT8DOT-DEFAULT`.

## Problem

The W4b lever (`VT_TT_KEEPQUANT_INT8DOT`, grouped int8-dot keep-quant decode
kernel on the Tenstorrent backend) stayed default-off since #3031 because its
e2e lane failed the ratified 500-mnat band at (5,7) 1125 mnats — measured
against the pinned llama.cpp b10451 **INCREMENTAL greedy**.

The 2026-09-22 GSQ investigation established that this denominator is invalid:
at identical causal positions the pin's own full-sequence BATCH decode
disagrees with its incremental greedy. Against the valid denominator
(teacher-forced on each arm's own generated stream, `oracle-dump` on the pinned
build), the APEX-I-Nano anchor adjudication (2026-09-23, `.agents/benchmark-record.md`)
reads:

| Arm | Max gap (mnats) | In-band (500) | Mean TPOT (real prompt) |
|---|---:|---:|---:|
| Default (W4a grouped, f32-exact) | 26.6 | 16/16 | 1253 ms |
| INT8DOT | 245.3 | 16/16 | 642 ms |

The rationale for default-off is gone, the lever halves decode TPOT, and it
shrinks captured-trace demand 33.8% (W4b evidence), which unblocks the 27B
trace-capture row that follows it. This spec gates the flip.

## Scope

- Default of `VT_TT_KEEPQUANT_INT8DOT` flips to ON on the Tenstorrent backend.
  The env var keeps its meaning: `=0` restores the W4a grouped f32-exact arm
  (a same-binary opt-out, the rollback).
- Records: `benchmark-record.md` gains the pre/post-flip sweep evidence;
  `docs/FEATURES.md` TT keep-quant row names the new default and the byte-diff
  warning; `docs/USAGE.md` documents the opt-out; the row spec gains `## Now`.
- The llama.cpp incremental-vs-batch report (draft
  `/tmp/llamacpp_report_draft.md`) is FILED upstream in the same change. The
  flip's denominator argument must not rest on an unfiled finding: if upstream
  rebuts it, the flip's evidence base is on record and re-openable.

## Gates (all must hold before the flip lands)

1. **Wider APEX sweep.** The 16-prompt anchor set is small. Extend to the
   committed sharegpt prompt set (`/tmp/gsq_prompt_sharegpt.json` lineage —
   commit the prompts used), ≥64 prompts, teacher-forced on our own stream vs
   the pinned batch decode. Gate: 100% of prompts in-band (500 mnats max gap),
   per-prompt max gap recorded. The 245.3-mnat position-3 outlier must not
   grow past the band on any prompt.
2. **Sibling keep-quant models.** The kernel serves every TT keep-quant decode,
   but the band evidence is one model. Run the existing e2e token gates for the
   other keep-quant vehicles (Qwen3.5-0.8B and the 9B vehicle) on the lever,
   against their committed gates. Gate: unchanged verdicts. A sibling that
   degrades does NOT block the flip only if its gate is a token-exact gate the
   INT8DOT arm passes; it DOES block if it fails its own committed gate — then
   the flip narrows to a per-model default and this spec is amended.
3. **Denominator on the record.** The llama.cpp report is filed (link in the
   PR body) and the batch-vs-incremental evidence is committed to
   `docs/bench-evidence/` so the flip's justification does not live in `/tmp`.
4. **Standard gates.** Full P150 backend suite unchanged (the flip changes the
   DEFAULT arm; the suite must pass with the new default, and the `=0` opt-out
   arm must reproduce the old default's outputs bit-for-bit on one APEX leg —
   proving the rollback is real). Commit style, trailers, agent-record green.

## Risks

- **Accuracy margin:** INT8DOT's max gap is 9x the default's (245.3 vs 26.6)
  with ~2x band headroom. Gate 1 exists to price this on a real prompt
  distribution, not the anchor set.
- **Byte-diff for users:** flipping the default changes every TT keep-quant
  user's token stream under unchanged flags. Declared, loud, and paired with a
  same-binary opt-out (`=0`). Existing SACRED gates that pinned the W4a arm
  are re-pointed or re-run, never deleted.
- **Denominator rebuttal:** if upstream llama.cpp defends the incremental
  greedy, the flip's justification weakens. Mitigation: gate 3 files the
  report so the disagreement is on the record, and the opt-out keeps both arms
  reachable.
- **Capture interaction:** the 27B captured arm is not yet e2e-gated with the
  lever on (capture row follows this one). The capture row's gate re-runs the
  band; it does not inherit this spec's evidence.

## Non-goals

- No kernel changes. The int8-dot kernel is as merged; this flips a default.
- No CUDA/CPU keepquant behavior change — this is TT-only.
- No 27B capture work in this change (next row; this spec only shrinks its
  trace demand).

## Stop conditions

- Any sibling model fails its own committed gate on the lever → stop, amend to
  a per-model default or abandon with the evidence recorded.
- Any gate-1 prompt exceeds the band → stop; the flip is not supported by this
  evidence. Record the sweep and keep the lever opt-in.
- A behavior edit beyond the default flip appears → split it into its own unit.

## Owed

- The 27B trace-capture row (separate spec) consumes this flip's -33.8% trace
  demand; nothing in this change gates capture.
