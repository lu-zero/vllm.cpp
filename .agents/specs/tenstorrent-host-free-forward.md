# Tenstorrent host-free decode forward — plan

Status: **DRAFT plan, 2026-08-13.** The prerequisite for decode mesh-trace
capture (see `tenstorrent-trace-runner.md`: capture aborts on `to_vector`
readbacks inside `ForwardLayers`). This document decomposes the work into
independent rows sized for parallel claims.

## Goal

Make the per-decode-layer region of the TT forward **host-free**: zero
`to_vector` / `EnsureHost` readbacks between `BeginCapture` and
`EndCaptureGraph`. Only then can `Qwen3DenseDecodeGraph` capture/replay it
on `kTENSTORRENT` (ttnn `begin_trace_capture` prohibits any host read).

The per-layer op sequence (Qwen3-dense / Mistral, from
`dense_attn_block.h`) and its host-readback status at T=1 today:

| per-layer op | status today (T=1) | in captured region? |
|---|---|---|
| `RmsNorm` (pre-attn q-norm + residual merges) | HOST (rows<32) | yes |
| `MatmulBT` (qkv / o_proj / mlp) | device | fine |
| `QkvSplit` | **pure host** | yes |
| `RmsNorm` (qk-norm, Qwen3 only) | HOST | yes |
| `RopeNeox` / `RopeFromCache` | HOST (T·H<64) + `BuildCosSinFromPositions` host | yes |
| `ReshapeAndCache` | **pure host** | yes |
| `PagedAttention` | **pure host (host oracle)** | yes |
| `SiluAndMul` | device | fine |

Boundary ops OUTSIDE the layer loop (readbacks here are fine — they are the
capture region's input/output edges): `Embedding` (host-staged upload),
`GreedyArgmax` (host readback of the final logits).

## Three independent sub-problems (rows)

Each is independently gateable; none blocks another except the capture row,
which wants all three.

### R1 — Device-resident RmsNorm + RoPE at T=1 (threshold flip + perf)

**Problem:** the hybrid thresholds route `RmsNorm` (rows<32) and `RopeNeox`
(T·H<64) to host at T=1. The trace-runner spike measured the perf cost of
flipping them all-device: 12.5 → 10.7 tok/s (~14%, reproduces handoff §6).
Capture must recover that.

**Work:** flip the thresholds to all-device when capture is active (or
unconditionally, gated on `support_static_graph_mode()`), accept the ~1.8
tok/s eager regression, and let capture claw it back. The numerics were
already proven acceptable by `BACKEND-TENSTORRENT-RESIDUAL-GOLDEN`
(device bf16 vs CPU f32 = constant 0.0459 abs, ordinary rounding).

**Sub-blocker:** `RopeNeox`/`RopeFromCache` depend on `BuildCosSinFromPositions`,
which reads `pos` on host (line 1291) and builds cos/sin host-side. The
device RoPE apply path exists (`RopeApplyDeviceNeox`) but the cos/sin
construction is still host. Needs a device-resident cos/sin path OR a
precomputed cos/sin cache uploaded once (the `RopeCosSinCacheKernel` path
already exists for the cache mode — route through it).

**Gate:** op-level `RmsNorm`/`Rope` device parity (already measured); e2e
Qwen3/Mistral gate token-exact or near-tie vs the TT golden.

### R2 — Device-resident QkvSplit + ReshapeAndCache (small host-staged ops)

**Problem:** `QkvSplit` and `ReshapeAndCache` are pure host today — they
read every input via `EnsureHost` and `CommitHost` the output. Both are
bit-exact memcpy/stride ops that went host-staged in W0 because Alloc was
host memory. Inside a captured region they must stay on device.

**Work:** add device-resident variants using `ttnn::slice` (QkvSplit) and
the device paged-write path that already exists for paged KV
(`NotePagedKvRacWrites` / `TryDevicePagedFill` / `TryDevicePagedUpdate` —
landed with residency). The device paged-write path already keeps a ttnn KV
shadow; wire `ReshapeAndCache` to it unconditionally when capture is active.

**Gate:** op-level bit-exactness vs the host path (these are deterministic
copies — byte-identical is achievable and required); e2e gate.

### R3 — Device-resident PagedAttention decode (the big one)

**Problem:** `PagedAttention` at T=1 decode runs the **host f32 oracle**
(`PagedAttentionKernel` host path). The device path
(`TryPagedAttentionDeviceDecode`, `paged_scaled_dot_product_attention_decode`)
exists and is used when the KV shadow is current, but it still reads
`block_table`/`seq_lens`/`query_start_loc` on host (lines 1644-1646) and
reads `query` host (line 1707) before the device call. Those metadata
reads are the capture blocker.

**Work:** keep the metadata tensors device-resident across the decode step
(they are small int32 tensors; upload once per step BEFORE the captured
region, not inside it), and ensure the query entering PA is already device
(no `EnsureHost(query)`). The device SDPA decode path itself is
capture-clean (it's a single ttnn op); the work is removing the host
metadata reads around it.

**Gate:** device PA vs host oracle numerics (already measured: max_abs
~0.0009 for prefill; decode parity measured separately); e2e gate.

### R4 — Flip `support_static_graph_mode()` + wire capture (only after R1-R3)

**Problem:** the platform gate and the `Qwen3DenseDecodeGraph` wiring are
trivial once the region is host-free. This row flips the platform flag,
verifies capture no longer aborts, and measures replay tok/s vs eager.

**Gate:** capture completes (no `TT_FATAL`); replay max_abs=0 vs eager
(already the landed unit-test property); **replay warm tok/s ≥ 12.5**
(the current hybrid eager baseline) — this is the payoff that justifies
all four rows.

## Sequencing + dependencies

```
R1 (RmsNorm+RoPE device)  ─┐
R2 (QkvSplit+RAC device)  ─┼─► R4 (capture wire + measure) ──► decode tok/s win
R3 (PA decode metadata)    ─┘
```

R1, R2, R3 are independent and parallel-claimable. R4 is the integration
row that wants all three + produces the headline number. If R4's replay
tok/s does NOT beat 12.5, the whole effort is a wash — but that can only be
known after R1-R3, which is the cost of answering it.

## Gates (per row + integration)

- **Correctness:** every device-resident variant must be bit-exact or
  near-tie vs the current host path, gated by the existing TT golden pair
  (`our_ids_tenstorrent.npy` / `neartie_gap_mnats_tenstorrent.npy` for
  Qwen3-0.6B, the Mistral pair for Mistral-7B). RED-first op-level test
  before each e2e gate.
- **Capture (R4 only):** `TT_FATAL`-free capture + replay max_abs=0 +
  replay warm tok/s ≥ 12.5 (Qwen3-0.6B `vllm-cli` smoke, same harness as
  the trace-runner spike).
- **No perf regression outside capture:** the threshold flips in R1 regress
  *eager* tok/s (12.5→10.7) — that regression is acceptable ONLY because R4
  recovers it. If R4 is not landed, R1 must not ship unconditionally; it
  must gate on `support_static_graph_mode()` so non-capture runs keep the
  hybrid thresholds and the 12.5 baseline.

## Risk

- **R3 is the scope risk.** R1 and R2 are mechanical (flip + reuse existing
  device paths); R3 (device PA decode with device-resident metadata) is
  real work and the most likely place to find another host touch.
- **R4's payoff is uncertain until measured.** The whole plan exists to
  answer "does capture beat 12.5 tok/s"; if it doesn't, R1-R3 still
  delivered device-resident ops (useful for future prefill capture) but no
  decode win. That's an honest outcome, not a failure — it's the
  measurement the trace-runner spike owed and couldn't make.
