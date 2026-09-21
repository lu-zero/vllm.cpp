ID: ISSUE-LOCAL-01M3265TQ26YSVKVA1HX6E28VC
Title: vllm-cli (C ABI path) and vllm-bench (AsyncLLM path) tokenize the same prompt differently for a bare .gguf model
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-21
Updated: 2026-09-21
Closed: -

## Problem

For /mnt/models/ISTA-DASLab-Qwen3.8-27B-GSQ-RCO-GGUF/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf, vllm-cli reports prompt_tokens=51 and vllm-bench reports 65 input tokens for the IDENTICAL prompt string (32 words, '1 2 world 1 hello ...'). The cli loads through the C ABI (vllm_engine_load, src/capi/vllm_c.cpp); the bench drives AsyncLLM directly. The two entry points resolve the tokenizer differently for a bare .gguf (no tokenizer.json sidecar), so any cli-based parity or benchmark number for GGUF models is suspect. Evidence: /tmp/gsq_iq3xxs_tt_cli.log vs /tmp/gsq_iq3xxs_gate.log (same string, 51 vs 65). Discovered 2026-09-21 during the GSQ e2e parity investigation (ISSUE-LOCAL-01M32475H9MZMMVVTCDP0EK7VT), where it invalidated the cli-based CPU tie-break leg.

## Resolution

-
