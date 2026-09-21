ID: ISSUE-LOCAL-01M32475H9MZMMVVTCDP0EK7VT
Title: GSQ-RCO Qwen3.8-27B GGUF decodes degenerate blanks on TT keep-quant e2e; fails b10451 greedy parity
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

The GSQ e2e gate (tenstorrent-gsq-keepquant.md gate 3) fails: on identical 65-token input (gsq_p0.i32, engine-tokenization verified identical to the oracle prompt), the engine produces [220,16]x16 on both GSQ-RCO files while the pinned llama.cpp b10451 oracle (oracle-dump --greedy 16, ctx=128) echoes prompt words. IQ3_XXS oracle: 23066 23066 1814 220 17 23066 220 16 220 16 23066 1814 220 17 1814 1814; IQ3_S oracle: 23066 220 16 220 16 23066 1814 220 17 1814 1814 23066 220 17 1814 220. Engine: [220,16]x16 both. 0/16 match. Prior session's gsq_iq3s.json (EXIT=0) was the same degenerate pattern - the gate was never token-green. Ruled out: prompt tokenization mismatch (engine ids == gsq_p0.i32), missing tensors (GGUF tensor names identical to APEX except the blk.64 MTP head APEX-only), env config (default env identical), metadata (only block_count 65->64, nextn_predict_layers 1->absent differ). APEX-I-Nano passes on the same build and prompts. Evidence: /tmp/gsq_iq3xxs_gate.log /tmp/gsq_iq3xxs_gate_tokens.json /tmp/gsq_iq3xxs_gate.dump /tmp/gsq_prompt_sharegpt.json. Next diagnostic: teacher-forced logits comparison at prompt positions 0..64 (oracle dump carries per-position f32 logits).

## Resolution

-

## Evidence 2 (2026-09-21, logits localization)

`VT_DUMP_LOGITS` on the TT e2e leg vs `oracle-dump` teacher-forced dump
(`gsq_p0.i32`, 65 positions, verified seq == prompt ids).

First-step logits: engine top5 = `[220, 23066, 1814, 198, 271]`, oracle
top5 = `[23066, 198, 271, 220, 1814]` — same token set, ranks flipped,
Pearson r = 0.9122. The oracle margin at the flip is ~5.3 nats
(23066 = 19.198 vs 220 = 13.849); the engine puts 220 = 18.587 above
23066 = 17.567. This is a precision-level divergence in the TT **prefill**
path — the first generated token is already wrong — not a broken graph and
not drift.

CPU tie-break: `vllm-cli --device cpu` produces first token 23066 == oracle
(the TT leg produced 220) — the defect is TT-device-specific at prefill.
Caveat: the cli leg tokenized the prompt to 51 tokens (vs 65 in
bench/oracle), so the CPU leg's post-first-token collapse is not comparable.

Artifacts: `/tmp/gsq_logits/ours_0.f32` (16x248320 f32),
`/tmp/gsq_iq3xxs_tf.dump` (65x248320 f32), `/tmp/gsq_iq3xxs_cpu_cli.log`.

## Evidence 3 (2026-09-21, keep-quant exonerated)

Bisect leg with `VT_GGUF_KEEP_QUANT=0` (all weights expand to bf16; every
quant GEMM leaves the keep-quant path): the e2e output is STILL
`[220,16]x16` on the IQ3_XXS file. The int8-dot and grouped quant kernels
are NOT the cause. The divergence lives in the shared prefill compute
(GDN chunked prefill, attention prefill, rope, norms, embedding, or lm_head
as TT-device ops) on this 64-block file.

Also verified: `qwen3_5_gguf_weights.cpp:965` reads
`nextn_predict_layers` with default 0, so the GSQ file (key absent) runs the
full 64-layer trunk — the 63-layer-trunk hypothesis is dead.

APEX-I-Nano (65 blocks + nextn=1) passes token-exact on the same build;
GSQ (64 blocks, no nextn) fails on both the keep-quant and bf16 paths.
The remaining shape difference is the file itself: block_count 64 vs 65 and
the GDN/attention pattern it implies for the trunk.

Artifacts: `/tmp/gsq_iq3xxs_bf16_tokens.json`, `/tmp/gsq_iq3xxs_bf16.log`.

## Evidence 4 (2026-09-21, chunk-boundary test, cli parity)

1. Chunk-boundary hypothesis (65 = 64+1 crosses the ubatch) tested with a
   63-token prompt (within one chunk): oracle greedy = `[220,16,23066,23066,
   1814,...]` (" 1 hello hello world 1 hello 1 1 2 world hello"); the bench
   STILL produces `[220,16]x16`. The engine matches the oracle for the first
   TWO tokens here, then collapses into the blank loop. On the 65-token
   prompt it diverged at token 1. No fixed chunk boundary; the divergence
   point moves with the prompt.
2. The blank pair `[220,16]` is a real attractor of this model (the oracle
   itself emits it early on the 63-token prompt), which is why every defect
   mode here presents as the blank loop.
3. vllm-cli on TT and vllm-cli on CPU produce IDENTICAL output
   (" hello 1 1 1 1 1 1 1") for the same prompt — but the cli path
   tokenizes that text to 51 tokens where the bench path produces 65 for
   the same string. The cli goes through the C ABI (`vllm_engine_load`);
   the bench drives AsyncLLM directly. The two entry points disagree on
   tokenization for this GGUF, which invalidates every cli-based parity
   number and is its own defect.

## Consolidated state

Confirmed: TT bench path diverges from the pinned oracle at prefill on the
GSQ files with large logit perturbations (first-step r=0.91, a 5.3-nat
margin flipped), on BOTH keep-quant and bf16 residency. Eliminated: prompt
tokenization (bench ids verified byte-equal to the oracle prompt), missing
tensors, nextn misread, quant kernels (bf16 leg), a fixed chunk boundary,
TT-vs-CPU as such (cli legs agree where their tokenization agrees).

Next step: an instrumented layer-level probe (per-layer hidden states,
TT vs CPU reference, same file) — a scoped debugging unit, not an in-flow
fix. The cli-vs-bench tokenizer discrepancy (51 vs 65) should be its own
issue and blocks any cli-based parity leg.

Artifacts: `/tmp/gsq_p64_sharegpt.json`, `/tmp/gsq_p64.i32`,
`/tmp/gsq_p64_oracle.dump`, `/tmp/gsq_p64_tokens.json`,
`/tmp/gsq_iq3xxs_tt_cli.log`, `/tmp/gsq_iq3xxs_cpu_cli.log`.

## Evidence 5 (2026-09-21, APEX contrast attempt)

APEX-I-Nano through the same bench path with the SAME 65-token prompt also
produces a blank-loop output (`[220,17,220,16,220,16,...]`) — but this is
NOT evidence of the same defect: the pinned b10451 oracle now REFUSES the
current APEX file (`llama_model_load: error loading model: missing tensor
'blk.64.ssm_conv1d.weight'`), so there is no denominator for the APEX-65
leg. The historical APEX token-exact gate ran against a file revision the
pin could load; the checkpoint on disk no longer matches it (the exact
re-quantization-under-unchanged-name hazard the protocol names). Recorded
as a side finding; APEX parity needs a repinned or re-exported checkpoint
before any 65-token claim.

Consequence for the GSQ investigation: the only file with a working oracle
denominator at non-aligned prompt lengths is the GSQ pair, and there the
divergence is real (evidence 1-4). The "APEX passes" contrast is weaker
than it looked: APEX was only ever gated at chunk-aligned prompt lengths
(128), never at 63/65.

Next step unchanged: instrumented layer-level probe with a CPU reference on
the GSQ file; additionally re-establish the APEX oracle denominator (fresh
export matching the pin, or a recorded file hash for the gate).
