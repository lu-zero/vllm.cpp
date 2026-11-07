# SPEC — `QWEN38-DFLASH2` — DFlash2 speculative decoding for qwen3.8-27B on the P150: stage 0 (the integration design study)

- Owning row: `QWEN38-DFLASH2`
- Kind: umbrella row — stage 0 (the design study) delivers; stages 1+ file after stage 0's output
- Issue: created in the same flow (row-scoped local issue)

## Why this row exists

The trained drafter for qwen3.8-27B **already exists**:
`incoai/Qwen3.8-27B-DFlash2` (Apache-2.0, the exact target model —
unlike kolibri1 and flint, where training a drafter was the blocker).
The DFlash2 pattern is proven on the same model at 78–110 tok/s on 4×
Blackhole (the dflash2-vision bundle, pinned as the
`vllm-tt-plugin-qwen38` oracle). Our single-P150 qwen3.8-27B serves at
0.03 tok/s (TPOT 25.2 s, the rebench).

The gap decomposes into: hardware (4× chips vs 1), speculative
decoding (multi-token/step vs 1), and kernel maturity. **Speculative
decoding is architecture, not hardware** — a single chip can run the
draft block (5 layers ≈ small) + the verify (our decode over K+1
tokens) + acceptance, yielding ~K tokens per verify pass. At the
bundle's modeled acceptance (~6-7 of 8), our 25.2 s TPOT becomes
~25.2/6.5 ≈ 3.9 s/token ≈ 0.26 tok/s — a ~8.5x improvement on our own
path, before any kernel work.

The pattern's requirements (from TT-DRAFTER-STUDY): residual taps in
the verify trace, the draft block forward (one pass, small), the
acceptance math (greedy id-compare or exact rejection sampling), and
the bucket geometries with hysteresis. All are runtime structures —
the drafter checkpoint already exists.

## Stage 0 — the integration design study (this row's deliverable)

`docs/design/qwen38-dflash2-integration-20261102.md`: how the DFlash2
runtime maps onto **vllm.cpp's qwen3.8-27B TT serving path**, with
file:line citations into our tree and the bundle's sources (fetched to
scratch, cited, never committed). The five questions:

1. **Our qwen3.8-27B TT serving decode today**: which forward runs
   (AsyncLLM → the TT path), the host-touch boundary per step, the
   residual taps' shape, the capture/replay state (W0-style captured
   traces exist? the serving path's capture state?), file:line.
2. **The verify trace**: the target's own forward over
   `[pending, d_0..d_K-1]` at `p+1..p+K+1` — how our TT decode extends
   to a K+1-token block (the KV cache handling, the paged path vs the
   direct forward).
3. **The drafter's host cost**: the 5-layer draft forward's shape at
   kolibri1... at qwen3.8's hidden size (the drafter checkpoint's
   config — fetch incoai/Qwen3.8-27B-DFlash2's config.json), the
   residual-tap projection (fc → hidden_norm → layers), the expected
   per-step cost.
4. **The acceptance path**: greedy id-compare on-card vs host (the
   bundle's `_accept_greedy` vs the `QWEN36_DFLASH_DEVICE_LOOP` glue
   traces), the sampled rejection-sampling path, and what our code
   needs for each.
5. **The staged integration plan**: the DFlash2 study's five stages
   (trace I/O contract → trace our decode → stub drafter → real
   drafter → buckets) mapped onto OUR tree, each with its gate, risks
   named (fixed-address aliasing, accept-rate below break-even,
   bucket-flip cliffs).

## The stages after stage 0 (named, each files its own row)

- Stage 1: the trace I/O contract + our decode traced
- Stage 2: the stub drafter (the block forward, untrained weights — the
  machinery proof)
- Stage 3: the real drafter (incoai's checkpoint loaded)
- Stage 4: the buckets + the on-card accept

## The correctness contract (from the bundle's claims, verified on our path)

Greedy: token-for-token identical to the non-draft arm (the accept rule
IS the target's argmax — a correct DFlash2 implementation cannot change
greedy output). Sampled: distribution-exact (rejection sampling with
the target's own probabilities). Both gates on every stage; a greedy
divergence is a WRONG implementation, not a numerics trade — STOP.

## Non-goals

- No 4-chip mesh (single P150; the mesh patterns are studied for a
  future multi-chip box, not implemented). No CPU-path change. No
  tt-model installation (the drafter checkpoint's weights fetch from
  HF directly — Apache-2.0). No adoption of the bundle's plugin runtime
  (we implement the pattern in vllm.cpp's TT path; the bundle is the
  reference).

## The measured baseline (the rebench, `ea8566499` lineage)

TPOT 25.2 s, 0.03 tok/s output (governor performance, 3-rep medians).
The draft-forward cost and the verify-block cost are stage 0's
measurements (the study models them; the rows after measure for
real).

## Gates

check-agent-record, style, trailers, preflight (named host dotnet
failure). Device legs (the study measures the verify-block forward on
device): the standing device recipe (mutex, reset, envs, claims file,
governor logged, NO timeout guards).

## Git integration

One pull request, `row/QWEN38-DFLASH2`, spec committed first, the
study-note commit second. Stages 1+ file as follow-up rows after
stage 0's output.
