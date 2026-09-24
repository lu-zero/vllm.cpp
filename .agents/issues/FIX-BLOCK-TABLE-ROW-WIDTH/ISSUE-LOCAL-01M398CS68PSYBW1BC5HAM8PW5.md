ID: ISSUE-LOCAL-01M398CS68PSYBW1BC5HAM8PW5
Title: test_kv_cache_interface fails at main head: MambaSpec override not reached through the base reference
Row: FIX-BLOCK-TABLE-ROW-WIDTH
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-24
Updated: 2026-09-24
Closed: -

## Problem

At main head b8a9377fd, test_kv_cache_interface fails one assertion (case 'max_num_blocks_per_req: a Mamba row holds cdiv(max_len, bs) + k', :1052): through 'const KVCacheSpec& base = *mamba_at(32, "none", 3)' the call returns 4 where the MambaSpec override (kv_cache_interface.cpp:214-228, header decl :493 with explicit override) returns 7 — while every DIRECT mamba_at(...)-> call in the same case passes. The virtual dispatch discrepancy reproduces in a fresh Debug build at main head (2026-09-24) and fails every PR's build-test-cpu that runs on the merge ref, including record-only PRs (#3292). Root-cause owed: the case landed with 9c6f4a94c (FIX-BLOCK-TABLE-ROW-WIDTH); its own merge-run build-test-cpu was cancelled, so the break may have landed with the row itself or a later commit.

## Resolution

-
