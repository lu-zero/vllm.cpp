ID: ISSUE-LOCAL-01M3918KQ580Z3NHVRNXVF15FZ
Title: qwen35-gguf-q4km golden drifts on the DEFAULT arm after the tt-metal rebase
Row: BACKEND-TENSTORRENT-QWEN35
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-24
Updated: 2026-09-24
Closed: -

## Problem

The qwen35-gguf-q4km lane battery fails at prompt[0] tok=0 on the committed golden WITHOUT the INT8DOT lever set (engine 11751 vs committed 279; 7172 with the lever) — a stale golden against the current tt-metal build, pre-existing and independent of the lever. It must be re-captured against the current pinned tt-metal and the lane battery re-run before it can adjudicate anything about keepquant levers. Evidence: docs/bench-evidence/g2-default-20260923.log (2026-09-23, P150).

## Resolution

-
