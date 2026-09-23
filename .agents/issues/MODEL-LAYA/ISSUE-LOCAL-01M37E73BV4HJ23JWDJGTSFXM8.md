ID: ISSUE-LOCAL-01M37E73BV4HJ23JWDJGTSFXM8
Title: MODEL-LAYA row added to model-matrix.md without regenerating the lifecycle rollup
Row: MODEL-LAYA
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-23
Updated: 2026-09-23
Closed: -

## Problem

#3263 added the Laya row to .agents/model-matrix.md but left the Architecture-support checklist rollup stale: SPIKE reads 11 with 12 SPIKE rows present and Total reads 383 with 384 rows. scripts/check-model-checklist.py fails on main (reproduced locally).

## Resolution

-
