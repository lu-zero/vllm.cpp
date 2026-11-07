ID: ISSUE-LOCAL-01M6VMNQ2ZDVKVZK9D7DHXQE0W
Title: DFlash2 speculative decoding for qwen3.8-27B on the P150: stage 0, the integration design study
Row: QWEN38-DFLASH2
State: OPEN
Kind: perf
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-07
Updated: 2026-11-07
Closed: -

## Problem

The trained drafter for qwen3.8-27B ALREADY EXISTS (incoai/Qwen3.8-27B-DFlash2, Apache-2.0, the exact target model — unlike kolibri1/flint where training was the blocker). The dflash2-vision bundle proves the pattern on the same model at 78-110 tok/s on 4x Blackhole; our single-P150 path serves at 0.03 tok/s (TPOT 25.2s, the rebench). Speculative decoding is architecture, not hardware: a single chip can run the 5-layer draft block + the verify over K+1 tokens + acceptance, yielding ~K tokens per verify (~8.5x on our path at the bundle's modeled acceptance). Stage 0: the integration design study — how the DFlash2 runtime maps onto OUR qwen3.8-27B TT serving path (the host-touch boundary, the verify trace over a draft block, the drafter's cost, the acceptance path, the staged plan), with file:line citations into our tree and the bundle's sources. Stages 1+ file after stage 0. Spec: .agents/specs/qwen38-dflash2.md; the mechanism study is docs/design/tt-drafter-study-20261103.md (mine/row/TT-DRAFTER-STUDY).

## Resolution

-
