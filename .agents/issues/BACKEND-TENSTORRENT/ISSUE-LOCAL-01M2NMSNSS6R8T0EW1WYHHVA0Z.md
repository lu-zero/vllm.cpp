ID: ISSUE-LOCAL-01M2NMSNSS6R8T0EW1WYHHVA0Z
Title: APEX 27B e2e: greedy token-correctness gate vs the llama.cpp oracle
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

The APEX-I-Nano 27B now completes the e2e bench (PR #3212: exit 0, 64/64 tokens) but nothing pins that the generated tokens are CORRECT. Before any bench record or the APEX e2e owed closes, the greedy decode needs a gate against the pinned llama.cpp b10451 oracle (the registry's k-quant floor oracle) on the identical workload: same GGUF artifact, same generated-dataset prompts (BuildPrompt, seeds 0 and 1, input_len 128), greedy temp 0, 32 output tokens, no chat template. Deliverable: recorded token/text comparison, disposition, and the recipe.

## Resolution

-
