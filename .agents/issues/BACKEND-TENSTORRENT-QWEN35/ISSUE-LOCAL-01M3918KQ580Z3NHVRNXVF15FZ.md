ID: ISSUE-LOCAL-01M3918KQ580Z3NHVRNXVF15FZ
Title: qwen35-gguf-q4km golden drifts on the DEFAULT arm after the tt-metal rebase
Row: BACKEND-TENSTORRENT-QWEN35
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-24
Updated: 2026-09-25
Closed: -

## Problem

The qwen35-gguf-q4km lane battery fails at prompt[0] on the committed golden — but the 2026-09-25 recapture session STOPPED before capturing: the current tt-metal tree is a DIRTY mid-experiment checkout (scratch commits 81f3bbf3b40/f3088579db0 + 25 uncommitted modifications across layernorm dispatch, program_run_args, memcpy prefetch), so the engine's tokens are measurements of an unpinned moving oracle, not of a tt-metal revision. The new capture is deterministic on the dirty build but diverges from the committed capture at 188/256 cells with max teacher-forced gap 13.125 nats (band 0.5) and degenerate loops — a numerics regression AGAINST THE DIRTY BUILD that must not be laundered into a golden. Required before any recapture: pin tt-metal to a clean recorded revision (or record the scratch modifications as a patch series), rebuild, and only re-derive goldens if the gap profile returns to the committed envelope (0.25-nat-class divergence, 0.5-nat band). Evidence: two byte-identical capture runs on the dirty build, gap re-derivation via the pinned qwen35-q4km-neartie-gap.sh recipe with the correct oracle venv and dequant artifact.

## Resolution

-
