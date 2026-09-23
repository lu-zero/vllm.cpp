ID: ISSUE-LOCAL-01M37NAZFFD8EHV653N4JWGRRJ
Title: multi-column W32 GEMV merge landed three undocumented production env vars
Row: BACKEND-CUDA-SM120
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-23
Updated: 2026-09-23
Closed: -

## Problem

Commit 1a172caf5 (multi-column grouped 32-block GEMV kernel, merged to main) reads VT_CUDA_ALLOC_STATS, VT_V4_W32_COLS and VT_V4_W32_WARPS from src/ without documenting them, so scripts/check-env-doc.py fails and main's agent-record gate is red at its own head (run 35894699062). The two W32 levers are kernel-internal tuning switches with measured defaults and a byte-identity gate; VT_CUDA_ALLOC_STATS is a zero-cost diagnostic instrument like the already-allowlisted VT_DECODE_GRAPH_STATS.

## Resolution

-
