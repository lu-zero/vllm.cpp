ID: ISSUE-LOCAL-01M7JYHQJEANK9Z6S0SJWKXWBC
Title: TT batched-decode RAC device lane: garbage at the second slot's second KV head under rich suite state (deterministic per binary layout)
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

The W22-consume full-suite run at the merged head (e41f9fbbb: TT-PA-SMALL-WINDOW's fix merged into row/QWEN38-DFLASH2-STAGE-C) exposed a latent RAC device-lane defect: 'kTENSTORRENT batched decode RAC is capture-safe (num_slots=2)' fails its EAGER leg deterministically in the suite context (2/2 suite runs) — 'EAGER K bad u=1 h=1 e=8 got=4.45658e+28 want=0.00848389', 12/128 elements garbage, ALL of them at the second slot's second KV head — while the same case passes in isolation at both heads and passes in the suite at the pre-merge head 67d041d76 (2/2 suite runs). The garbage bits are recycled memory (the TT-PA-SMALL-WINDOW characterization's signature class), so this is a read-before-land/race or a misaddress in the RAC device lane under the host-free env (VT_TT_HOST_FREE_DECODE=1), whose trigger state shifted when the PA commit path's typecast program changed the allocator/JIT layout. NOT a regression of the PA fix (the fix cleared the W22 harness pa_bad 5/40 -> 0/40, the W22 discriminator, and the pure-decode PA case); it is a latent RAC defect newly exposed. Owed: characterize the trigger, root-cause the RAC device lane's second-slot second-head write, fix.

## Resolution

-

## Triage (2026-11-24, against the W34 staging-temp recycle conviction)

The W34 freeze bisect (docs/bench-evidence/qwen38-dflash2-stage-c-w34-20261123/)
causally convicted the deterministic BFP8 tile clobber mechanism: a
`Bfp8TypecastInstall` staging temporary is freed while an async device writer
still targets its DRAM base; the next staged weight's BFP8 output then lands on
that recycled base, and the victim is whatever allocation the allocator hands
out next. The persistent per-(rows, cols) staging scratch (666d89959) removes
the mechanism.

**Does the recycle mechanism explain this issue? Plausibly yes — the observed
signature matches on four points.** (1) The garbage is self-described as
"recycled memory ... the TT-PA-SMALL-WINDOW characterization's signature
class". (2) It is deterministic per binary layout and its trigger state
"shifted when the PA commit path's typecast program changed the allocator/JIT
layout" — exactly what changing a typecast program does under the recycle
mechanism: it reorders staging-temp allocation, so a different temporary's base
becomes the recycled victim. (3) The corruption is localized (12/128 elements,
all at the second slot's second KV head — one page/tile region) rather than
spread, which is what a mistargeted tile write produces. (4) It reproduces in
the suite only, under rich allocator history, and not in isolation — the same
rich-history dependence every W28-W33 baseline leg showed.

**Not yet explained by the mechanism:** the RAC device lane path itself is not
a `Bfp8TypecastInstall` staging path as far as the W28-W33 trace chain
established, so whether the RAC decode reads a buffer whose base was ever a
freed staging temp needs one check before the mechanism is claimed.

**Minimal falsification/verification experiment:** rerun the failing suite
case (`kTENSTORRENT batched decode RAC is capture-safe (num_slots=2)`, EAGER
leg, suite context) with the staging-scratch fix in effect (default on — this
is already the state at head 666d89959) and, as the control, with
`VT_TT_BFP8_SCRATCH=0` (the recycled-temporary path restored). Prediction if
the recycle mechanism is the cause: the case flips with the knob — clean under
the scratch, garbage under the recycle — in the same suite context. If it
stays garbage under both, the recycle mechanism is falsified for this issue
and the misaddress hypothesis (the issue's alternative) survives.

No fix implemented this session; the experiment above is the next action.
