# Spec: dense-arm VL reachability for the EXL3 27B (Qwen3_5 + vision)

Row: `MODEL-QWEN35-DENSE-VL-EXL3` (roadmap row added by this PR).
State: DRAFT (2026-09-25).
Git integration: one pull request (spec + phases 1-2 + gates), branch
`row/MODEL-QWEN35-DENSE-VL-EXL3`.

## Problem

The EXL3 27B checkpoint `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` (pin
`19441ac87`, 14.28 GiB, 2,427 tensors, on disk at
`/mnt/models/Mia-AiLab-Qwen3.8-27B-EXL3-3.5bpw`) is the artifact
`MODEL-QWEN35-EXL3` (#2495) was built against — its TEXT arm is covered
(dense attn/MLP EXL3, GDN projections incl. bf16 `in_proj_a/b`, 6-bit
lm_head, MTP 4-bit; `qwen3_5_mtp.cpp:366` names this very repo). The one
gap is the VISION TOWER on the dense arm: 333 bf16 `model.visual.*`
tensors are silently filtered at load
(`qwen3_5_dense_weights.cpp:887`), and the dense VL forward takes a
caller-supplied `mm_main` (`qwen3_5_dense.h:440-471`), so no production
entry point can answer an image prompt. Nothing lands dead: this row
wires the tower to a reachable entry point.

## Prior art (reuse, do not reinvent)

The sibling MoE arm already consumes the shared vision components:
`LoadQwen3_5MoeVision` (`qwen3_5_weights.cpp:1893-1900`) and
`Qwen3_5MoeVisionConfig` (`:1862-1881`) — whose hardcoded geometry
(1152/16/27/4304, patch 16, merge 2, no deepstack) matches this
checkpoint's `vision_config` exactly. Shared components:
`LoadQwen3VLVisionWeights`, `Qwen3VLVisionForward`. MRoPE dense math
landed (`qwen3_5.cpp:9813-10427`).

## Phases

1. **`LoadQwen3_5DenseVision`** mirroring the MoE variant; STOP
   silently filtering `model.visual.` for this architecture
   (`qwen3_5_dense_weights.cpp:887`) — unquantized-but-required.
2. **Wire tower → merger → mm_main** behind a production entry point
   (the registered `Forward` or the existing `Qwen3_5VLGenerateGreedy`
   reached from `include/vllm.h`): an image request is served, not a
   class exercised.
3. **Evidence**: re-run the text EXL3 token gate at THIS pin
   (`19441ac87`) as the row's weights documentation; port the pinned
   vLLM `qwen3_5.py` image/video tests, documenting harness
   adaptations; `docs/USAGE.md` weights section (repo + revision,
   sha256 of both shards, refused arms: none) + matrix/row records.

## Config anchors (from the checkpoint's config.json)

hidden 5120; intermediate 17408; 64 layers (48 linear + 16 full,
`full_attention_interval 4`); heads 24 / kv 4, head_dim 256;
`partial_rotary_factor 0.25`, rope_theta 1e7, mrope_section [11,11,10]
interleaved; max_position_embeddings 262144; vocab 248320;
tie_word_embeddings false; `output_gate_type swish`, attn_output_gate
true. GDN: linear 16 key / 48 value heads, key/value head dim 128, conv
dim 10240 kernel 4, `mamba_ssm_dtype float32`. Vision: depth 27,
hidden 1152, heads 16, intermediate 4304, out 5120, patch 16, merge 2,
temporal 2, `deepstack_visual_indexes []`, 2304 position embeddings.
Image token 248056, video 248057, vision start/end 248053/248054. No
MoE (dense 27B — no expert counts).

## Gates

1. An image request through the production entry point returns a text
   completion (the tower runs; the answer's quality is the ported
   tests' business, not the wiring's).
2. Text-only EXL3 behavior is byte-identical to pre-change (the tower
   load is additive; the filter removal touches only this
   architecture's load path).
3. Device suite unchanged on the P150.
4. Ported upstream VL tests pass (pinned vLLM `qwen3_5.py` image/video
   path), harness adaptations documented.

## Risks

- The VL tensor prefix (`model.language_model.*`, `model.visual.*`) vs
  the text-only loaders: the backbone prefix mapping exists
  (`qwen3_5_weights.cpp:1497-1510`); the tower prefix is new to the
  dense loader — follow the MoE variant's mapping.
- The greedy VL drivers being internal: reaching them from
  `include/vllm.h` is the reachability deliverable, and the staged-slice
  rule (name what is unreached) applies to anything short of that.

## Stop conditions

- If the tower forward diverges from the kev/MoE-arm geometry in a way
  the shared config cannot express, stop and split the config work into
  its own unit.

## Owed

- `ISSUE-LOCAL-01M3AHX9DQX8HNE32G80C9VGMJ` — the dense-arm VL reachability row itself: phases 1-2 (the
  `LoadQwen3_5DenseVision` variant + the production entry point wiring)
  are the code; phases 3-5 (token gate at the artifact pin, ported
  upstream VL tests, weights documentation) are the evidence. Until this
  row lands, image prompts on the EXL3 27B are refused by the silent
  filter this row removes.
