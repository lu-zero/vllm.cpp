ID: ISSUE-LOCAL-01M39392YYBDPPQXJF0B1X0AKS
Title: OPT-125m device arm throws kQkvSplit shape-contract failure on the TT backend (pre-existing on main)
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-24
Updated: 2026-09-24
Closed: -

## Problem

test_opt_paged_engine with VLLM_CPP_OPT_MODEL_DIR=/mnt/models/opt-125m-bf16-st throws 'vt: tenstorrent kQkvSplit: out shapes must be [T, *]' on the TENSTORRENT device path, at tenstorrent_ops.cpp:1968 on main (control run 2026-09-24, P150, reset-hardened) and at the identical moved site on the split-stage-4 branch (:1454) — same exception, same 21 pre-checkpoint assertions passed, so the failure predates and is independent of the paged extraction. The 92/92 device suite passes on both trees; only the OPT checkpoint-backed leg hits it. This blocks the reviewer-named checkpoint-backed backend proof (the device-gated reachability detector for RAC/PA now rides the suite instead) and the OPT device lane until root-caused.

## Resolution

-
