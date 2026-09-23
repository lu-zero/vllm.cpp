ID: ISSUE-LOCAL-01M326DPAP4N14ZYWT1G7XEY01
Title: Pinned b10451 oracle refuses the on-disk APEX-I-Nano checkpoint (missing blk.64.ssm_conv1d.weight)
Row: BACKEND-TENSTORRENT
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-21
Updated: 2026-09-23
Closed: 2026-09-23

## Problem

oracle-dump at the pinned llama.cpp b10451 fails to load /mnt/models/mudler-qwen3.8-27B-APEX-gguf/Qwen3.8-27B-APEX-I-Nano.gguf: 'llama_model_load: error loading model: missing tensor blk.64.ssm_conv1d.weight'. The file declares qwen35.nextn_predict_layers=1 with a PARTIAL MTP head, which the pin's loader refuses. The historical APEX token-exact gate therefore ran against a file revision (or an export variant, e.g. --no-mtp) that no longer matches the checkpoint on disk — the re-quantization-under-unchanged-name hazard. Consequence: there is NO oracle denominator for APEX-I-Nano at any prompt length today, and the 'APEX passes e2e' contrast used in the GSQ parity investigation (ISSUE-LOCAL-01M32475H9MZMMVVTCDP0EK7VT evidence 5) is unverifiable. Fix direction: re-export/obtain an APEX variant the pin loads (e.g. --no-mtp), or record the sha256 of the exact gate artifact in docs/USAGE.md and restore it; then re-run the APEX token gate at both chunk-aligned (128) and boundary-crossing (63/65) prompt lengths. Discovered 2026-09-21: /tmp/apex_p0_bench.log, oracle stderr 'llama_model_load: error loading model: missing tensor blk.64.ssm_conv1d.weight'.

## Resolution

RESOLVED 2026-09-23: the oracle denominator was never broken. The pinned b10451 loads the on-disk APEX-I-Nano checkpoint (llama-completion loads and decodes it; the earlier 'missing tensor blk.64.ssm_conv1d.weight' failure was oracle-dump resolving against a mismatched libllama, not the artifact — with LD_LIBRARY_PATH pointed at the pinned build the oracle loads the file). With the denominator restored, both demanded legs were measured: the ALIGNED prompt (127 tokens, built from the verified prefix pattern, byte-verified round-trip) gives TOKEN-EXACT 16/16 vs the oracle greedy; the CROSSING prompt (65 tokens, gsq_p0) gives 16/16 in-band near-tie with max gap 26.6 mnats against the ratified 500-mnat band — the engine's 220,17 vs the oracle's 220,16 at position 1 is a genuine 0.027-nat tie. The historical APEX token-exact gate verdict stands, and the crossing-length behavior is exactly the anchor-band disposition the k-quant lever recorded. Evidence: /tmp/apex_p128_tokens.json (engine), /tmp/apex_p128_greedy.dump, /tmp/apex_ours_tf.dump, /tmp/apex_p0_greedy.dump.
