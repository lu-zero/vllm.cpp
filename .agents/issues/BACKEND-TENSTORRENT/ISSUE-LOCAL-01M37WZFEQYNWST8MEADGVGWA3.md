ID: ISSUE-LOCAL-01M37WZFEQYNWST8MEADGVGWA3
Title: Decompose the 76.5 ms single-GEMV call on P150: staging vs launch vs compute
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-23
Updated: 2026-09-23
Closed: -

## Problem

The APEX 27B decode on P150 is 30-60x off the native floor and the record localizes the gap to per-call overhead: one M=1 BFP8 GEMV measures 76.5 ms. The W7 staging-write elimination inverted (counters byte-identical, wall unchanged), so the next step the record names is a decomposition, not another elimination attempt: attribute the 76.5 ms to host-side operand construction, launch submission, and completion wait, plus an amortization probe (N identical back-to-back calls).

## Resolution

-
