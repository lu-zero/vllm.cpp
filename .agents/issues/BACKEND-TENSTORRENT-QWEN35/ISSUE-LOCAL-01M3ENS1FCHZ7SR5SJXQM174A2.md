ID: ISSUE-LOCAL-01M3ENS1FCHZ7SR5SJXQM174A2
Title: q4km TT-lane golden embeds superseded in-band numerics: 51 cells where the current stream matches the oracle and the committed golden does not
Row: BACKEND-TENSTORRENT-QWEN35
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-26
Updated: 2026-09-26
Closed: -

## Problem

After the capture-warmup redesign (spec tenstorrent-capture-warmup-redesign.md) the q4km TT-lane captured battery completes byte-identically to the all-eager stream (md5 c403afe36f31b324ba22216136be66e2), but 51 cells on 5 prompts (2,3,9,10,11; first divergence prompt[2] tok 2) still diverge from the committed capture golden (md5 74f003d783f94e6589e04cd3452bc8df). Adjudication: at every one of the 51 diverged cells the CURRENT stream matches the vLLM per-prompt oracle greedy and the committed golden does not (51/51 vs 0/51; 13/16 vs 8/16 prompts strict-exact); all gaps sit inside the ratified 500-mnat band (max 250 mnats). The divergence is the golden era's own in-band deviation from the oracle, superseded by justified numeric changes accumulated over ~200 commits since the golden's capture (2415a6b22, 2026-09-12): the 14d7a25077 eager tree measures 9/16 strict-exact (a mid-point; eager dump md5 ca4bd359d2a841a08de67283825c73d7), the current tree 13/16. Byte-identity to the OLD golden is therefore unreachable without a golden re-derivation, which is a non-goal of the redesign row (the goldens stand). This residual needs its own row: attribute the per-commit drift chain, then re-derive the capture pair (our_ids_tenstorrent_capture.npy/.i32 + the teacher-forced gap) under the pinned recipe and ratify the new anchor.

## Resolution

-
