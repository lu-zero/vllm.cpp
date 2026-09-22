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

## Evidence 6 (2026-09-21, the divergence is ENGINE-WIDE, not TT-specific)

RETRACTION of parts of evidence 2 and 4: the "CPU tie-break" and the
"cli-vs-bench tokenizer" claims compared different prompts (32-word string
vs the 41-word genprompt text the dataset actually carries — gsq_p0.i32 is
the byte-exact llama.cpp encoding of the 41-word text, 65 ids, verified
with llama-tokenize at the pin). The tokenizer-defect issue
(01M3265TQ26YSVKVA1HX6E28VC) is closed as falsified.

Controlled A/B, identical input: vllm-cli with the 41-word prompt,
`prompt_tokens=65` verified on both legs (ids byte-equal to the oracle's
teacher-forced sequence):
- `--device cpu` → " 1 1 1 1 1 1 1 1"
- `--device auto` (TT) → " 1 1 1 1 1 1 1 1"
IDENTICAL outputs. The oracle on the same 65 ids echoes prompt words
(first token 23066; engine 220 with a 5.3-nat oracle margin).

CONCLUSION: the forward divergence on the GSQ files is ENGINE-WIDE — CPU
and TT compute the same (wrong) logits. The defect is in the shared model
graph / weight interpretation for this file shape (64 blocks, no nextn
head), not in any Tenstorrent kernel. Evidence 2's "TT-device-specific at
prefill" and evidence 4's cli-tokenizer claims are retracted; the r=0.91
first-step logit comparison (TT bench vs oracle on identical ids) stands,
and by engine-wide agreement now describes the shared forward.

Logs: /tmp/gsq41_cpu.log, /tmp/gsq41_auto.log.

## Evidence 7 (2026-09-21, block-structure hypothesis dead; layer-bisect plan)

The GSQ and APEX files carry IDENTICAL per-block tensor names for blocks
0..63 (the full GGUF name diff is the blk.64 MTP head only, evidence 1), so
a per-file GDN/attention pattern difference cannot explain the divergence.
With evidence 6 (engine-wide) and the bf16 leg (quant kernels exonerated),
the remaining suspects are the shared qwen35 forward itself: the delta-net
chunked prefill math, rope, final norm, or a config key the engine
interprets differently from llama.cpp despite equal metadata values.

Layer-bisect plan (the next unit of work, needs code on both sides):
1. Extend `oracle-dump` (our own tool against the pinned libllama) with a
   per-layer hidden-state dump (llama.cpp's per-layer callback /
   embeddings API at b10451).
2. Add an equivalent per-layer dump to the engine's qwen35 forward (a
   VT_DEBUG_LAYER_DUMP-style env on the CPU tier is enough — the defect is
   engine-wide, so the CPU tier localizes it).
3. Compare per-layer hidden states on identical teacher-forced input
   (gsq_p0.i32); the first diverging layer names the component.

Constraints: the APEX oracle denominator must also be restored
(01M326DPAP4N14ZYWT1G7XEY01) before any APEX contrast is used again; the
historical APEX gate is unverifiable against the on-disk checkpoint.

## Evidence 8 (2026-09-21, ROOT CAUSE CLASS: the V-head reorder breaks a head-indexed consumer)

Per-layer hidden-state bisect (oracle side: a cb_eval-based layer-dump
against the pinned libllama; engine side: the existing VT_DUMP_ACT
instrument, CPU tier, identical teacher-forced input gsq_p0.i32):

- Post-embedding state MATCHES (rel err 0.16%, bf16 rounding).
- Layer 0 (a GDN layer) diverges; inside the layer, `gdn_mixed` matches the
  oracle's `linear_attn_qkv_mixed-0` exactly in the q and k blocks
  (cos 1.0) and is WRONG only in the v block (4096:10240).
- The v-block wrongness is a pure PERMUTATION: engine v head i ==
  oracle v head perm[i] with cos 1.0 for all 48 heads, where
  perm = [0,16,32,1,17,33,2,18,34,...] — the 16-strided interleave of
  `conversion/qwen.py _reorder_v_heads`.
- Our loader DELIBERATELY reorders V heads at load
  (qwen3_5_gguf_weights.cpp ReorderVRows/ReorderVCols, the inverse of the
  converter) into its own canonical order. The forward therefore has to
  apply EVERY head-indexed consumer in the engine's canonical order too.
  Something per-head does not follow: prime suspects are the per-head
  dt/beta/alpha gates (ssm_a / ssm_dt / ssm_alpha / ssm_beta, 48-wide),
  the GDN state slot layout, or the ssm_out/ffn input slices.
- Corollary: APEX carries the same reorder and would diverge the same way
  at these prompt shapes; the historical APEX gate result is consistent
  only with conditions we can no longer reconstruct (see
  01M326DPAP4N14ZYWT1G7XEY01).

Tooling added (scratch, outside the repo): /tmp/lanegate-oracle/layer-dump.cpp
+ binary — per-node manifest and per-node .f32 grab against the pinned
libllama via llama_context_params::cb_eval. Engine artifacts:
/tmp/gsq_layers_engine (VT_DUMP_ACT, CPU tier). Oracle artifacts:
/tmp/gsq_all_oracle, /tmp/gsq_layers_oracle, /tmp/gsq_l0_oracle.

NEXT: audit every head-indexed consumer of the GDN path against the
reorder (ssm_a/dt/alpha/beta indexing, state layout, ssm_out rows); the
one that reads the UN-reordered order is the defect. A red-first unit test
should pin one layer's gated output against a b10451-derived golden.

## Evidence 9 (2026-09-21, SUPERSEDES evidence 8: no forward defect — bf16 accumulation vs the f32 oracle)

Evidence 8's V-head root-cause claim was WRONG: the observed v-block
permutation is the loader's by-design ReorderVRows output, and with the
head permutation compensated EVERY layer-0 stage matches the oracle —
conv 0.22%, qkv q/k exact, gated-vs-final_output 0.39%, residual 0.8%.
Two analysis errors corrected: ggml node dumps read as reshape(ne[1],
ne[0]) (a transposed read produced three false mismatches), and the
engine's VT_DUMP_ACT pair is (branch output, residual stream), so the
layer output is hidden+res.

Correct per-layer comparison, all 64 layers: rel err grows smoothly
0.8% (layer 0) → ~1.5% (layer 12) → ~4% (layer 32) → 6-9% (layers 36+).
No discrete break, no stage-level divergence anywhere. The engine forward
is a faithful bf16 mirror of the pinned f32 oracle; the first-token flip
(the 5.3-nat margin) is accumulated bf16 activation noise on a heavily
sub-bit-quantized model whose greedy decode sits next to the
[220,16]-blank attractor.

CONSEQUENCE: the GSQ e2e gate as written (token-exact vs the f32 b10451
greedy oracle) cannot pass on the shipped bf16 arm — the same disposition
problem the spec already names for IQ1_S/IQ1_M. The f32-activation arm is
a REFUSED, owed conversion (VT_ACT_F32 aborts by design; spec
qwen38-27b-q4km-token-exactness.md, issue #2534). Options, in the row's
terms: (a) ratify the near-tie/distributional disposition for the GSQ e2e
gate and record the bf16-vs-f32 gap as the named open axis, or (b) land
the owed f32 arm and gate token-exact. A ceiling is not declared either
way.

## Evidence 10 (2026-09-22, THE GATE PASSES: gap 0.0 mnats against the batch-decode denominator; the oracle's incremental greedy is self-inconsistent at the pin)

The decisive experiment flipped the picture again. Teacher-forcing the
ENGINE's own generated stream (prompt + our 16 tokens, one batch) through
the pinned oracle and reading the oracle's argmax at each generated
position gives:

- IQ3_XXS: engine `[220,16]x16`, oracle argmax `[220,16]x16`,
  **gap 0.0 mnats at all 16 positions** (band 500).
- IQ3_S: engine `[220,16,220,17,...]`, oracle argmax identical,
  **gap 0.0 mnats at all 16 positions**.

Our engine is not merely near-tie — it is argmax-EXACT against the
oracle's full-sequence batch decode, on both files.

The residual discrepancy is INSIDE llama.cpp at the pin: position 64's
logits differ depending on what FOLLOWS in the batch — TF-65 (prompt
only) argmax 23066 max 19.198; TF-81 with our blanks argmax 220 max
20.883; TF-81 with the oracle's own greedy suffix argmax 23066 max
19.935. Same causal context, three different logit vectors. The GDN
hybrid graph's ubatch-boundary handling (splits 64+1 vs 64+17) is
non-causal-inconsistent at b10451. The "hello hello world" greedy that
motivated evidence 1-5 came from the INCREMENTAL decode path, which
disagrees with llama's own batch decode.

GATE DISPOSITION: the e2e gate (tenstorrent-gsq-keepquant gate 3/5)
passes for both GSQ-RCO files with gap 0.0 mnats against the oracle's
batch-decode logits — no near-tie waiver needed, and the bf16-vs-f32
accumulation of evidence 9 is the open axis it records, not a blocker.
The incremental-vs-batch inconsistency at the pin is a llama.cpp finding
(upstream report optional; the registry's llama-cpp pin note should carry
it once reported).

Artifacts: /tmp/gsq_ours_full.i32, /tmp/gsq_ours_tf.dump,
/tmp/gsq_iq3s_gate_tokens.json, /tmp/gsq_iq3s_ours_full.i32,
/tmp/gsq_iq3s_ours_tf.dump.
