ID: ISSUE-LOCAL-01M3AHX9DQX8HNE32G80C9VGMJ
Title: Dense-arm VL reachability for the EXL3 27B (Qwen3_5 + vision tower)
Row: -
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-24
Updated: 2026-09-24
Closed: -

## Problem

The EXL3 27B checkpoint Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw (pin 19441ac87, on disk at /mnt/models/Mia-AiLab-Qwen3.8-27B-EXL3-3.5bpw) is the artifact MODEL-QWEN35-EXL3 (#2495) was built against — its TEXT arm is fully covered. The one gap is the vision tower on the dense arm: 333 bf16 model.visual.* tensors are silently filtered at load (qwen3_5_dense_weights.cpp:887) and the dense VL forward takes a caller-supplied mm_main (qwen3_5_dense.h:440-471), so no production entry point can answer an image prompt. The shared Qwen3VLVision components exist (the MoE sibling reuses them; its config geometry matches this checkpoint exactly), so this is GAP-WITH-SEAM: mirror LoadQwen3_5MoeVision into a dense variant and wire tower->merger->mm_main behind a production entry point. No new kernel work. Full gap table and config anchors in the owed spec.

## Resolution

-
