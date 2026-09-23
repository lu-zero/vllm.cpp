ID: ISSUE-LOCAL-01M37HC2ANZT1Y3M9A8JVEENYG
Title: CUA-S1-FORMS and LAYA merges left docs/FEATURES.md without supported-arch rows, and CuaS1Forms cannot match the arch-token pattern
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

#3265 and #3263 registered CuaS1Forms and LayaModel but added no docs/FEATURES.md supported-arch rows, so scripts/check-supported-models.py fails: two registered architectures have no public row, and CuaS1Forms additionally does not match ARCH_TOKEN_RE (no For*/Model/Extractor suffix), so the checker drops it from the comparison it is supposed to gate. Red on main's CI (agent-record job, run 35882704592) and locally.

## Resolution

-
