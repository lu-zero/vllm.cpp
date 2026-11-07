# Qwen3.8-27B DFlash2 on the P150 — the integration design study (stage 0)

- Row: `QWEN38-DFLASH2`, issue `ISSUE-LOCAL-01M6VMNQ2ZDVKVZK9D7DHXQE0W`.
  Spec: [`.agents/specs/qwen38-dflash2.md`](../../.agents/specs/qwen38-dflash2.md).
- Mechanism source: the `dflash2` bundle
  `mando2222/qwen3.8-27b-dflash2-vision-p300x2-q4kv-v6-thin-cp4` @
  `7855cf0d2982064350991bdb8e1ec1db82f914b3` (Apache-2.0), fetched to scratch
  (`/tmp/df2-bundle/`, not committed). The pin was re-verified against its HF
  metadata on 2026-11-07: `README.md` resolves at that revision (HTTP 200) and
  its `PROVENANCE.json` records the served `code_sha256` and the tt-metal commit
  `547841b0…`. Its deliverable, the TT-DRAFTER-STUDY
  ([`tt-drafter-study-20261103.md`](tt-drafter-study-20261103.md), on
  `mine/row/TT-DRAFTER-STUDY`), established the mechanism; this note maps it
  onto OUR tree. Bundle line numbers cite the scratch copies at the pin.
- Drafter checkpoint: `incoai/Qwen3.8-27B-DFlash2` — config.json fetched only
  (metadata; NO weights downloaded). Key shape: `DFlash2DraftModel`, 5 layers
  all `sliding_attention` (window 2048), hidden 5120, head_dim 128, 32 q-heads
  / 8 kv-heads, `intermediate_size` 17408, vocab 248320, `block_size` 8,
  `mask_token_id` 248070, `conv_group_size` 16, `conv_kernel_size` 2,
  `selector_rank` 256, `selector_top_k` 16, `target_layer_ids`
  `[5, 19, 33, 47, 61]`, `is_causal: false`, rope theta 1e7.
- Target checkpoint: `Qwen/Qwen3.8-27B` config.json (metadata only):
  `Qwen3_5ForConditionalGeneration`, 64 text layers at `full_attention_interval`
  4 — 48 `linear_attention` (GDN) + 16 `full_attention`, hidden 5120,
  `head_dim` 256.
- Citation discipline: every OUR-tree claim carries a `file:line`; every bundle
  claim cites the fetched source or the TT-DRAFTER-STUDY. This note contains NO
  product code and downloads NO weights.

## 1. Our qwen3.8-27B TT decode today (Q1)

**The serving path.** `vllm-bench --model *.gguf` runs the production AsyncLLM
path (`docs/benchmarks/tt-keepquant-27b-decode.md:23-24`, the W3 invocation the
rebench reused). The model resolves through the GGUF loader
(`src/vllm/entrypoints/model_loader.cpp:3102`: `Qwen3.8-27B-Q4_K_M.gguf`
states `qwen35.block_count = 65`) into the `Qwen3_5Model` dense arm —
`src/vllm/model_executor/models/qwen3_5.cpp` is the port of upstream
`qwen3_next` (`qwen3_5.cpp:6-8`), i.e. the GDN + sliding/full-attention
architecture `Qwen/Qwen3.8-27B`'s config declares.

**The decode driver.** Graphed decode routes through
`Qwen3_5DenseDecodeGraph` (declared
`include/vllm/model_executor/models/qwen3_5_dense.h:575`, `Impl` at
`qwen3_5.cpp:12186`, constructor `qwen3_5.cpp:12385`, `Step` at
`qwen3_5.cpp:12396`). Dispatch is weights-type templated
(`VLDecodeGraphFor`, `qwen3_5.cpp:10484-10500`): dense weights →
`Qwen3_5DenseDecodeGraph`, MoE weights → `Qwen3_5DecodeGraph`. The MoE graphed
arm is fp4-CUDA-only (`qwen3_5.cpp:10484-10495`), so on the P150 the dense
driver is the only graphed decode. **The fuzzy-match hazard is live**: the two
drivers are near-identical (`Impl` at 11486 vs 12186, both with
SizeSlot/SlotRing/capture state machines), and any DFlash2 verify work must
land in the DENSE driver (or in the shared seam) and must be checked not to
have drifted into the MoE twin.

**Keep-quant arm.** The Q4_K_M decode runs the keep-quant INT8DOT arm
(`VT_TT_KEEPQUANT_INT8DOT=1` is mandatory — the default lm_head arm OOMs;
`docs/bench-evidence/qwen38-p150-rebench-20261102/README.md:10-11`), serving
through `tenstorrent_keepquant.cpp` (the INT8DOT landing decision,
`src/vt/tenstorrent/tenstorrent_keepquant.cpp:936-945`) and the
`MatmulBTQuantGrouped` launches in the dense forward (`qwen3_5.cpp:6388`).

**Capture state.** The TT backend implements graph capture as ttnn mesh-trace
capture (`SupportsGraphCapture() = true`,
`src/vt/tenstorrent/tenstorrent_backend.cpp:96`; the state machine in
`src/vt/tenstorrent/tenstorrent_capture.cpp:21-22` — process-local single-slot
capture plus multi-graph handles, mirroring the CUDA backend's
`EndCaptureGraph` split; begin/end at `tenstorrent_capture.cpp:77-88,121`).
The dense driver captures per padded size, keyed
`DecodeGraphSlotKey{S, query_len, spec}` (`qwen3_5.cpp:11104-11110`), with the
cold → warm (capture) → replay state machine and **persistent fixed-address
host inputs** — "the captured graph's host->device copies bake these
addresses, so each size needs its own fixed-address buffers"
(`qwen3_5.cpp:12227-12234`). For the 27B specifically, `tt-27b-region-capture`
(`qwen3_5.cpp:12143`, `13008-13018`) splits the step one layer per region
because the whole-graph staging demand (~3.15 GB for 1,037 recorded commands)
never accrues and each region's trace buffer fits the 50 MiB budget the census
asserts; region scope is an env opt-in in this slice, and an over-budget
region DECLINE is sticky — the slot then serves EAGER
(`qwen3_5.cpp:12250-12253`). The capture discipline precedent is W0
(`docs/bench-evidence/flint-tt-w0-20261025/report.md`: honest
device-vs-CPU-only gate, mutex, reset, pinned substrate env) and the
kolibri1 TT records.

**The host-touch boundary per step.** On a replayed slot the host's per-step
work is: stage `token_ids`/`positions`/attention metadata into the slot's
persistent pinned buffers (zeroed per cell — the #1319 fix,
`qwen3_5.cpp:11475-11483`), then replay. On an eager slot (sticky decline, or
capture disabled via `VT_TT_DECODE_CAPTURE=0`, `qwen3_5.cpp:11496,12196`) the
host dispatches every op through the backend's op table — the eager
op-by-op dispatch the TT-DRAFTER-STUDY names as the structural difference
(study §2: "host-touched per layer per step"). **Which of the two the 27B
rebench leg actually ran is not recorded in the rebench README** (it records
JIT hits, governor, and medians, not capture state); establishing that is
stage A's first action, because the whole DFlash2 payoff model below hinges on
it.

**Residual taps.** The tap infrastructure ALREADY EXISTS as a shared seam:
`Qwen3_5AuxTaps` (`include/vllm/model_executor/models/qwen3_5.h:222-227`) —
ascending capture-after `layer_ids`, `tensor` bf16 `[T, H×taps]`, concat order
= layer_ids — produced by `ForwardDeviceMultiTap` (`qwen3_5.h:302-317`,
capture mirrors vLLM's eagle3 aux collection, `interfaces.py:1382`), carried
by the runner (`include/vllm/v1/worker/gpu/runner.h:1531`
`Qwen3_5AuxTaps spec_aux`), and held PERSISTENTLY per dense-decode slot as an
`[S, H*taps]` DBuf the captured region writes at a fixed address
(`qwen3_5.cpp:12241-12249`, the SPEC-DSPARK W8 hook). The tap layer set is a
CHECKPOINT property: the incoai DFlash2 drafter taps `[5,19,33,47,61]`
(its config `dflash_config.target_layer_ids`), and the eagle3 +1 shift
(`qwen3_5.h:212-217`: DFlash id L → capture key L+1) puts our capture keys at
`[6,20,34,48,62]`. The tree's existing comment names a DIFFERENT set
(`[1,16,31,46,61]`, `qwen3_5.h:212-214`) — that is the z-lab Qwen3.6 drafter's
set, not this checkpoint's. The runner must read the set from the drafter
config (which the W1 config builder already plumbs), never hardcode it.

**The measured baseline.** TPOT 25.23 s/token, TTFT 1203.6 s, 0.03 tok/s
output, 100% warm JIT (819/819), governor-swept 3-rep medians, schedutil
+1.0% (`docs/bench-evidence/qwen38-p150-rebench-20261102/README.md:41-47`).

## 2. The verify trace (Q2)

**What DFlash2 asks the target to do.** One verify pass runs the target's own
forward over `[pending, d_0..d_K-1]` — K+1 rows (K = `block_size` = 8, so 9
rows at the defaults) at positions `p+1..p+K+1` — and argmaxes on device into
a fixed ids buffer (`model.verify_traced`, bundle `model.py:2549`; the study
§1 item 2). The anchor+MASK structure is DRAFTER-side only: the verify sees
real candidate token ids, never a mask token (the drafter stages
`[anchor, MASK × (block-1)]`, bundle `dflash2_tp.py:1027`; our verify input is
`[x_p, d_0..d_7]`).

**How our decode extends to it.** The spec shape is already plumbed through
the dense driver, unused by the qwen3.8 serving path today:

- `DecodeGraphSlotKey{S, query_len, spec}` (`qwen3_5.cpp:11104-11110`) keys a
  captured graph by query length and spec-ness — a 9-row verify is a NEW SLOT
  KEY, not a redesign.
- The uniform-spec contract already exists: "a uniform SPEC batch carries
  `B = num_reqs * (1+k)`" (`qwen3_5.cpp:11714`, `12501`), the per-request
  arrays are sized by R not S (the #1319 lesson, `qwen3_5.cpp:11475-11483`),
  and the number of distinct spec query lengths per driver is bounded
  (default 2 — one steady-state `1+k` plus one clamped; `qwen3_5.cpp:11119-11135`,
  `VT_SPEC_GRAPH_MAX_QLENS`). DFlash2's verify is exactly the uniform `1+k`
  shape that bound was built for.
- KV: our paged path (`PagedKvCache` per full-attention layer +
  `slot_mapping`; the TT paged kernels in
  `src/vt/tenstorrent/tenstorrent_paged.cpp`) writes each of the 9 rows to its
  page slot — the paged DIRECT forward, not a new cache shape. Positions are
  the 9 absolute positions, per row, same as today's single position but
  length 9.

**The two real new pieces.**

1. **Per-layer causality over the block.** The full-attention layers must
   attend the 9-row query block BIDIRECTIONALLY (the drafter's proposal is
   block-diffused; the verify scores all 9 rows against the prefix and each
   other). In-tree precedent: `vt::DFlashBlockAttention` and the causality
   resolution — a declared top-level `is_causal` wins, else
   `dflash_config.causal`, else causal iff the DECLARED `layer_types[i]` is
   `sliding_attention` (`include/vllm/model_executor/models/qwen3_dflash.h:13-30`).
   The DRAFT side is resolved by that rule; the VERIFY side needs the same
   rule applied to the TARGET's 64 layers: `Qwen/Qwen3.8-27B` declares
   `full_attention` every 4th layer and `linear_attention` elsewhere (its
   config), and the target's verify must run the full-attention layers
   non-causally over the block while the GDN layers advance sequentially.
2. **GDN state advance and rollback.** 48 of the 64 layers are GDN
   (`linear_attention`). A K+1-token verify advances each GDN state through
   the block; a rejection of the tail must roll the state back. The bundle
   carries a per-bucket "GDN spec cfg" for exactly this
   (study §2: each `Bucket` is "a `(B_cfg, T_cfg)` verify geometry with its
   own verify trace, GDN spec cfg, and drafter layout"; the hold semantics —
   replay rows with `position = -1` that the decode SDPA and the KV write both
   skip — live at bundle `dflash2_serving.py:946` with the decode-SDPA
   contract at bundle `attention/tp.py:953`). Our `GdnStateCache`
   (`qwen3_5.cpp`, the `gdn_state` argument of `Step`,
   `qwen3_5.cpp:12396-12404`) is per-slot device state; the verify slot needs
   either a two-buffer A/B of the touched states per step (commit or restore)
   or a recompute-from-checkpoint. That is the single largest new device-side
   mechanism, and stage B owns it.

**Capture shape.** The verify trace is captured per slot key
`(S, 9, spec=true)` with the same fixed-address discipline the single-token
slot already has (persistent token/position staging, persistent logits,
persistent aux taps — `qwen3_5.cpp:12227-12249`). The bundle bakes the tap
copies into the verify trace at fixed addresses (study §4 item 1, bundle
`model.py:453-457,1780-1800`); our aux DBuf is already that
(`qwen3_5.cpp:12241-12249`).

## 3. The drafter's host cost (Q3)

**Shapes.** The incoai checkpoint: 5 layers × (attention 52.5M weights
[Q 5120×4096, K/V 5120×1024 each, O 4096×5120] + SwiGLU MLP 267.4M weights
[3 × 5120 × 17408]) ≈ 320M weights/layer ≈ **3.2 GFLOP per draft token row**
(2 FLOP/weight), ~25.6 GFLOP for one 8-row draft block — one forward, not
K steps (bundle `dflash2_tp.py:1053` `draft`, the study §1 item 1). The
residual-tap projection is `fc` from `[n, 5×5120]` → `[n, 5120]`
(bundle `dflash2.py:532-533`: "(1,n,25600) host taps -> hidden_norm(fc(taps))
[1,n,H]"), i.e. 131M weights ≈ 262 MFLOP per row — this is per ROW of the
drafter's paged context, so the `extend_context` cost grows with confirmed
history (bundle `dflash2.py:582`), amortized by the context-KV design (per-die
replicated DRAM, study §3). Add the grouped dynamic conv
(`conv_group_size` 16, kernel 2) and the selector (`rank` 256, `top_k` 16) —
both small relative to the matmuls, and both already exist as in-tree ops
with CPU references (`vt::DFlashGroupedConv`, `vt::Dflash2SelectorEdges`,
`vt::TopKValuesIndices`, `vt::Dflash2PathWalk` — the SPEC-DFLASH2 W2/W3/W4
deliverables; see `include/vllm/model_executor/models/qwen3_dflash.h:1-30`
for the drafter-model contract).

**Modeling on our path.** The draft forward is ~6% of the target's per-token
work (3.2 vs ~54 GFLOP/token for a 27B model at 2 FLOP/weight), so its cost is
NOT the question — the VERIFY block's marginal cost over a single-token step
is. Two bounds, to be decided by stage A/B measurement:

- **Dispatch-bound (the optimistic read).** Our TPOT is 25.23 s for ONE token
  — implausibly slow for pure GEMM time on a P150, and consistent with the
  eager per-op host dispatch + tt-metal op launch dominating (the sticky
  region-decline eager arm, `qwen3_5.cpp:12250-12253`; TTFT 1203.6 s for 128
  prefill tokens is 9.4 s/token, the same class). If a step's cost is ~flat in
  row count, a 9-row verify costs ≈ one step, and the DFlash2 steady state is
  (draft + verify) / accepted ≈ (1.06 + 1) × 25.2 / 6.5 ≈ **8 s/token at
  acceptance 6.5 of 8** — better than the spec's ~3.9 s estimate, which
  assumed the draft is free.
- **Compute-bound (the pessimistic read).** If the step cost scales linearly
  in rows, the verify block costs 9× one step ≈ 227 s and DFlash2 LOSES
  outright (227/6.5 ≈ 35 s/token ≈ today's 25 s). On this read the lever is
  stage A (capture), not the drafter.

The bundle's numbers support the dispatch-bound read for a captured loop: the
whole DFlash iteration runs on-card with one host touch per step and keeps
96-97% speed under a saturated host (bundle `README.md:38-40`; the loop
structure at bundle `dflash2_serving.py:898,927,942-963,978-1005`; the glue
traces that move accept/commit on-card at bundle
`dflash_devloop_run.py:294-499,660-691,732`). Its throughput claims — 78-110
tok/s DFlash vs 43 plain on 4 chips (`README.md:53-57`) — are ITS numbers on
4× Blackhole with upstream kernels, quoted as provenance, never our
denominator (`.agents/oracles/vllm-tt-plugin-qwen38.md:35-40`).

## 4. The acceptance path (Q4)

**Greedy (the first arm, and the correctness contract).** The bundle compares
ids: `_accept_greedy` (bundle `spec_decode.py:431`) compares the draft block
against the verify's argmax ids — and the argmax happens ON DEVICE
(`cfg.ids_out`, study §1 item 2), so greedy acceptance needs no logit readback;
the host reads 9 int32s per step. What our code needs:

1. a device-side argmax over the `[9, vocab]` verify logits block writing into
   a small persistent ids buffer. Precedent exists: `VLArgMaxDevice`
   (`qwen3_5.cpp:10705` region) and the sampler-on-device path that returns
   device logits WITHOUT the full-logits D2H (`ForwardDevice`,
   `qwen3_5.h:279-300`); the dense decode slot already holds the `[S, vocab]`
   f32 logits as a graph output at a fixed address
   (`qwen3_5.cpp:12228-12240`).
2. the host accept loop: prefix compare draft ids vs verify ids, commit the
   matched prefix + the bonus token, roll back KV/GDN for the rejected tail.

The correctness contract is the spec's: greedy is token-for-token identical to
the non-draft arm because the accept rule IS the target's argmax (bundle
`README.md:42-43`; spec §"The correctness contract"). A greedy divergence is a
wrong implementation — STOP, not a numerics trade.

**Sampled (the second arm).** Exact rejection sampling over the verify logits
(`SpecSampler.accept`, bundle `spec_sampling.py:280`; recovery draw
`norm(max(0, p − q))` at `spec_sampling.py:249`; one uniform per draft drawn
up front so the RNG stream is independent of where rejection lands,
`spec_sampling.py:329-331`). What our code needs: the target's per-row
probabilities over the block (a device softmax the captured trace can emit)
plus the drafter's q — this is exactly the arm the in-tree SPEC-DFLASH2 W4
work recorded as the probabilistic owed-arm (the realized-q draft-logit cache,
`CLAIM-SPEC-DFLASH2-W4` O12). Stage D records the TT sampled arm the same way:
greedy lands first, sampled follows behind its own gate.

## 5. The staged integration plan (Q5)

The TT-DRAFTER-STUDY's five stages (study §5), mapped onto OUR tree. Each
stage is independently mergeable, carries the greedy-exactness gate, and names
its risks.

- **Stage A — the trace contract, on our decode.** Enable + verify the
  region-scoped capture on the 27B dense TT driver (`tt-27b-region-capture`,
  `qwen3_5.cpp:12143,13008-13018`) under the standing device recipe: capture,
  replay, replay, assert byte-identical logits; then record which arm
  (replay vs sticky-decline eager) the rebench recipe actually serves and
  re-measure the single-token step both ways. This stage decides the
  §3 bound question before any drafter work is spent. Risks: the 50 MiB
  per-region staging budget declining regions (sticky eager, a silent
  performance cliff); fixed-address aliasing — a freed handle between capture
  and replay corrupts the trace silently, the hazard the bundle names itself
  (study §5 risk (a), bundle `spec_decode.py:976-979`).
- **Stage B — the verify block.** The `(S, 9, spec=true)` slot key with the
  existing 1+k plumbing (`qwen3_5.cpp:11104-11135,11714,12501`); per-layer
  causality resolved from the TARGET's declared layer types
  (`qwen3_dflash.h:13-30` rule applied to the 64-layer target); the GDN
  advance/rollback (§2 item 2 above) — the largest new mechanism; tap copies
  into the persistent aux DBuf (`qwen3_5.cpp:12241-12249`). Gate: the 9-row
  verify's per-row logits equal the single-token eager steps' logits row for
  row (device-vs-eager, W0 discipline), on the local P150. Risks: GDN rollback
  correctness (a wrong restore silently corrupts every later token — caught by
  the greedy-exactness gate only if the gate replays multi-block trajectories);
  query-len capture growth past the `VT_SPEC_GRAPH_MAX_QLENS` bound
  (`qwen3_5.cpp:11119-11135`) falling to eager.
- **Stage C — the stub drafter.** Wire the in-tree DFlash draft model
  (`qwen3_dflash.h:1-30`, the W1-W5 host-side machinery: config builder,
  grouped conv, selector edges, path walk) with the incoai CONFIG but
  identity/greedy-stub weights — accept rate 0, machinery proof. Gate: greedy
  output byte-identical to stage B with the stub in the loop; the tap layer
  set read from the drafter config (`[5,19,33,47,61]` → capture keys
  `[6,20,34,48,62]`), never hardcoded (§1 tap paragraph). Risks: the two
  decode drivers (11486 vs 12186) — the wiring must reach the DENSE driver and
  be proven to (the deletion test of
  [`.agents/reachability.md`](../../.agents/reachability.md)); the eagle3 +1
  shift applied twice or not at all (`qwen3_5.h:212-217`).
- **Stage D — the real drafter.** Load `incoai/Qwen3.8-27B-DFlash2` weights
  (Apache-2.0, fetched directly from HF — no tt-model install, spec
  non-goals); TT arms for the drafter's ops that lack them (grouped conv,
  selector — the CUDA arms exist, the TT arms are owed and must be recorded
  as owed, not silently eager-forever). Gate: greedy token-exact to the
  non-draft arm over a fixed multi-block prompt set (the G2-style gate
  precedent, `CLAIM-SPEC-DFLASH2-W6`), plus the acceptance measurement against
  break-even — the bundle's own break-even datum is DFlash v1 K=11 at 107.8
  tok/s vs native MTP 80.9 (`model_config.py:43`), i.e. the drafter must clear
  materially more than ~1.5 accepted drafts/iteration to beat plain decode
  (study §5 risk (b)). Risks: accept-rate below break-even on OUR kernel set
  (the bundle's ~6-7 of 8 is ITS measured claim, not ours — spec §Why); fixed
  seeds so the RNG-stream construction (`spec_sampling.py:329-331`) is
  reproducible.
- **Stage E — buckets (deferred until multi-user decode matters).**
  `(B, T)` verify geometries with per-bucket traces, hysteresis so a bucket
  never flips twice in one step, and slot HOLD (`position = -1` rows skipped
  by the decode SDPA and the KV write — bundle `dflash2_serving.py:142,237,49-53,946`,
  `attention/tp.py:953`). Our serving today is 2-prompt smoke
  (`docs/benchmarks/tt-keepquant-27b-decode.md` Limitations); the stage exists
  so the single-bucket design does not preclude it. Risks: bucket-flip
  cliffs (study §5 risk (c)); the hold contract interacting with our slot
  lifecycle differently than the bundle's.

**Stop conditions.** A greedy divergence at any stage is a wrong
implementation (spec §The correctness contract) — stop, bisect, do not tune.
Stage A finding the compute-bound read (§3) is not a failure — it re-scopes
the row onto the capture lever and the drafter stages wait.

## 6. What this stage did NOT do

- No device run: the draft-forward and verify-block costs are MODELED in §3
  from shapes and the rebench baseline, exactly as the spec's stage-0
  paragraph says ("the study models them; the rows after measure for real").
  The device recipe (mutex, reset, pinned substrate env, governor log) is
  stage A/B's obligation, and the claims file for `row/QWEN38-DFLASH2` is not
  yet written because no device leg ran.
- No weights downloaded: the drafter and target are known from their
  `config.json` metadata and their use sites in the fetched bundle sources.
- No product code, no tt-model installation, no 4-chip mesh work (spec
  non-goals).

## 7. License and attribution

The bundle and the drafter checkpoint are Apache-2.0 (bundle `README.md:3`;
the checkpoint card). Any stage that copies a bundle kernel or op file owes
the SPDX header retention, per-file attribution in the copying file, and the
PROVENANCE chain in its row spec — the TT-DRAFTER-STUDY's §6 terms, unchanged.
