ID: ISSUE-LOCAL-01M833SZ523B0B875BWQQW3D2Y
Title: the BFP8 staging-temp recycle hosts the deterministic header clobber; persistent per-shape staging scratch fixes it
Row: QWEN38-DFLASH2-STAGE-C
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-23
Updated: 2026-11-23
Closed: -

## Problem

The deterministic BFP8 tile-0 exponent-header clobber (strikes at canary probes #52/#58/#61, Q4_K_M KV64 drafter-ON RESTAGE=0+canary repro) depends on the staging-temp RECYCLE: W34 freeze bisect (VT_TT_BFP8_FREEZE, docs/bench-evidence/qwen38-dflash2-stage-c-w34-20261123/w34-freeze-3) kept the bf16 upload temporaries alive and the entire strike window passed clean while baseline reproduced ARRIVED-ZEROED at the same probe indices; the compile-time constant audit of every cached kernel ELF in the active build key (~/.cache/tt-metal-cache/17089598622106942383) found ZERO victim-base immediates (3,169,493 in-window words are all the known instruction-encoding false-positive class). Fix shipped: Bfp8TypecastInstall stages the bf16 upload into a persistent per-(rows,cols) device scratch that is never freed (default on; VT_TT_BFP8_SCRATCH=0 restores the old path); w34-scratch-1 ran the full repro exit 0 with 0/64 strikes and coherent token output where the corrupted baseline emitted [220,0,0,...].

## Resolution

-
