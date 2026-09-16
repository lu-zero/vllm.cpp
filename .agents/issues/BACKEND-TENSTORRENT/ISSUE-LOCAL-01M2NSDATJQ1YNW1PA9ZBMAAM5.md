ID: ISSUE-LOCAL-01M2NSDATJQ1YNW1PA9ZBMAAM5
Title: APEX/TT: keep-quant E=1 chunked arm emits all-zero logits at lm_head decode shapes
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-16
Updated: 2026-09-16
Closed: -

## Problem

The token-correctness gate (ISSUE-LOCAL-01M2NMSNSS6R8T0EW1WYHHVA0Z) caught: the E=1 chunked keep-quant matmul arm (MatmulBTQuantGroupedKernel, production dense path) commits all-zero logits at the 27B lm_head [1,248320] Q6_K shape (4 commits tracked=1, EnsureHost downloads them, argmax = id 0 every step, VT_DEBUG_SAMPLED tok=0 x4). Every sampled token of the 27B e2e is 0. Suspects: ttnn matmul with M in {1,2} against the chunked [chunk,K] bf16 tiles returning zeros, the partials concat, or a TTReclaimPlanes use-after-free inside the chunk loop (a reclaimed plane still aliased). The 0.8B vehicle never hit this because its bf16 head routes through MatmulBf16LogitsF32D (dense non-quant matmul), so the quant-head e2e path was never validated until now. Needs: a red unit test at [1,K]x[248320,5120] Q6_K (or scaled-down V) vs the CPU oracle, then the fix.

## Resolution

-
