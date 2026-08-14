# Tenstorrent host-free forward R1 — device RmsNorm + RoPE at T=1

Status: **DRAFT, 2026-08-13.** First row of the host-free-forward plan
(`tenstorrent-host-free-forward.md`). Sequential: measure after each row to
see its marginal contribution to capture.

Proposed row id: `BACKEND-TENSTORRENT-HOST-FREE-R1`.

## Scope

**In.** Flip the two hybrid thresholds that route `RmsNorm` (residual,
rows<32) and `RoPE` (PreferDeviceRope, T·H<64) to host at T=1 decode, so
both go all-device. The numerics were already proven acceptable by
`BACKEND-TENSTORRENT-RESIDUAL-GOLDEN` (device bf16 vs CPU f32 = constant
0.0459 abs). The device paths already exist in `tenstorrent_ops.cpp`.

The flip MUST be gated on capture-active (`support_static_graph_mode()`) so
non-capture runs keep the 12.5 tok/s hybrid baseline. Inert when capture is
off.

**Out.** The cos/sin host build inside `RopeNeoxKernel`
(`BuildCosSinFromPositions`, line 1363) is a known sub-blocker but is NOT a
device `to_vector` readback — it reads the host `pos` tensor. Whether it
triggers the ttnn fatal is an empirical question R1 answers: after this flip,
does capture get past the current `to_vector` fatal, and what is the NEXT
host touch? (If it's `pos`, R1.5 or R2 handles it; if capture succeeds, R1
alone was enough for the RmsNorm/RoPE portion.) No QkvSplit/ReshapeAndCache/
PagedAttention work (R2/R3).

## Upstream chain

CUDA's capture contract (`cuda_backend.cu:184-197`): the captured region is
async, no host sync, no host readback. TT must match. The RmsNorm device path
(`tenstorrent_ops.cpp:1105-1117`, `ttnn::add`+`ttnn::rms_norm`) and the RoPE
device apply (`RopeApplyDeviceNeox`, line 1221) are the loyal mappings.

## Our baseline

`RmsNormKernel` (line 1067-1118): `host_residual` when rows<32.
`PreferDeviceRope` (line 1344): false when T·H<64. Both host at T=1.
The trace-runner spike measured forcing both all-device: 12.5→10.7 tok/s
eager (the cost capture must recover).

## Work breakdown

1. Add a capture-active helper reading the platform's
   `support_static_graph_mode()` (cached per device, since the platform is
   invariant).
2. Gate `host_residual` and `PreferDeviceRope` on `!capture_active`.
3. Op-level test: confirm RmsNorm + RoPE device path runs at T=1 (rows=1,
   T·H=16) without the host fallback, bit-comparable to the residual-golden
   measurement (0.0459 abs).
4. **Measure**: with `support_static_graph_mode()` also flipped on (R4's
   change, applied locally for the measurement), does capture get past the
   `to_vector` fatal? Record the next failure point if any.

## Gates

- Op-level: RmsNorm + RoPE device output at T=1 within the band already
  measured by RESIDUAL-GOLDEN.
- E2e: Qwen3-0.6B `our_ids_tenstorrent.npy` golden still near-tie-passes.
- Capture probe (informative, not a hard gate for R1 alone): record how far
  capture gets.
- No eager perf regression when capture is OFF (the gate must be inert).

## Dependencies

- `BACKEND-TENSTORRENT-RESIDUAL-GOLDEN` (numerics proof, landed).
- `BACKEND-TENSTORRENT-HOST-FREE-FORWARD` (the plan, this branch).
- Hardware: Blackhole P150.

## Risks/decisions

- **The cos/sin host build may be the next capture blocker.** If after R1
  capture still fatals on a host op inside RopeNeoxKernel, the decision is
  whether R1 is complete (RmsNorm+RoPE *apply* are device) and the cos/sin
  build moves to a separate R1.5, or whether R1 must also switch RoPE to
  the `RopeFromCache` path (model-side change). Settle empirically.
- **The gate must be inert by default.** A bug in the gate that flips
  thresholds unconditionally would ship the 12.5→10.7 regression. Test the
  inert path explicitly.

## Outcome (2026-08-13) — R1 flip alone does NOT unblock capture; readback is inside a ttnn op

Implemented the opt-in gate (`VT_TT_HOST_FREE_DECODE`): flips both
`host_residual` and `PreferDeviceRope` all-device, plus flips the platform
`support_static_graph_mode()` so `Qwen3DenseDecodeGraph` engages. All inert
by default (21/21 TT tests, 814/814 assertions unchanged).

**Measurement (Qwen3-0.6B, `VT_TT_HOST_FREE_DECODE=1`):** capture STILL
fatals — `TT_FATAL: Reads are not supported during trace capture`, `0
replays`. R1's threshold flip is **not sufficient**.

**Diagnostic (the important finding):** with a capture-gated debug print on
every readback site in our code (`DownloadToHost`, `EnsureHost`, the two
direct `dev_out.to_vector` sites in PA decode/prefill), **zero of our
readbacks fire during capture**. The offending `to_vector` is therefore
**inside a ttnn op** (`ttnn::embedding` / `ttnn::rms_norm` /
`ttnn::sdpa_decode` / `to_memory_config` / etc.), not in our explicit
readback code. The TT backtrace shows only the ttnn frame
(`ttnn::Tensor::to_vector<float>`), not which op called it.

**Implication for the plan:** the host-free forward is **not** achievable
by only changing our thresholds/host-staging. At least one ttnn op in the
forward performs an internal host readback that ttnn trace prohibits.
Identifying that op (via a ttnn-symbolized backtrace or bisection) is the
real next step — it determines whether the fix is (a) swap to a
capture-safe ttnn primitive, (b) a ttnn version/bug fix, or (c) the
capture region must exclude that op. This is a deeper blocker than the
plan's R1-R4 assumed; the "host-free forward" may require upstream ttnn
changes, not just vllm.cpp changes.

**R1 code kept** (env-gated, inert by default): the threshold flip is
correct and will be needed once the ttnn-internal readback is resolved.
The `tt_capture_active()` flag + `VT_TT_TRACE_DEBUG` prints are kept as
diagnostics for the next row.

### Correction (2026-08-13, post-bisection): NOT a ttnn-internal readback

The "inside a ttnn op" hypothesis above was **wrong** — it was based on
instrumenting only `DownloadToHost` + the two PA `to_vector` sites, which
missed the fourth readback site: `EnsureHostBytes` (line 2518). A full op
bisection with `TT_OP_TRACE` at every kernel entry + a print in
`EnsureHostBytes` gave the exact sequence:

```
[TT-TRACE] BeginCapture (flag set)
[TT-TRACE] EnsureHostBytes DURING CAPTURE      <- the offender
TT_FATAL: Reads are not supported during trace capture
```

**Zero `TT-OP` kernel-entry lines fired** between BeginCapture and the
fatal — no `*Kernel` ran at all. The capture-blocking readback is in
**our** `Backend::Copy` → `EnsureHostBytes` → `dev.to_vector<float>()`
(line 2533), triggered by `ForwardLayers`'s very first line
(`qwen3.cpp:244`): `d.b.Copy(d.q, hidden.ptr(), hidden_in.data, ...)`.
`hidden_in` has a device shadow from `EmbedInto`; `Copy` forces a
device→host download to satisfy the host-side memcpy, inside the captured
region.

**This is fixable in our code, not an upstream ttnn blocker.** The fix:
when capture is active, `Backend::Copy` from a device-resident source must
do a device→device copy (or `ForwardLayers` must receive the device tensor
directly instead of copying through host). That's a concrete, scoped
R2-target — the "host-free forward" is achievable in vllm.cpp after all,
once every `EnsureHostBytes`/`Backend::Copy` site in the captured region
is made device→device. The R1 threshold flip + this copy fix together
clear the first capture blocker; subsequent readback sites (if any)
surface as the next bisection steps.

### R2 status (2026-08-13): fix site identified, device-copy primitive pending

The bisection pinpointed the exact fix site: `Backend::Copy`
(`tenstorrent_backend.cpp:56`) -> `EnsureHostBytes` -> `to_vector`,
triggered by `ForwardLayers`'s opening `d.b.Copy(...)` (`qwen3.cpp:244`).

Fix shape: when capture is active and both `dst` and `src` carry current
device shadows, `Backend::Copy` must do a device-to-device copy via a new
`CopyDeviceDeviceIfCapture` helper in the ops TU, called before
`EnsureHostBytes`.

Pending: the exact ttnn device-to-device copy primitive for this ttnn
build. Tried `ttnn::clone` (header not on the installed include path
despite the source existing) and `ttnn::copy` + `ttnn::zeros` (wrong API
for this version). The installed ttnn headers are a subset; the right
primitive needs focused API discovery against the installed header set.
R2 code reverted to keep the branch buildable; the
`CopyDeviceDeviceIfCapture` contract is the design, the body is the open
work — the single concrete next step.

### R2 update (2026-08-13): device-copy LANDED, next blocker is ttnn program-cache warm-up

Resolved the ttnn API discovery: the installed headers live in
`build_Release/include/ttnn/operations/...` (not the `libexec` tree). The
device→device copy primitive is `ttnn::copy(src, dst)` (from
`data_movement/copy/copy.hpp`) with a destination allocated via
`ttnn::empty(shape, dtype, layout, device, memconfig)` (from
`creation/creation.hpp`), using `Tensor::logical_shape()`/`dtype()`/`layout()`
accessors. Both headers had to sit inside the Tracy-disabled include block
(they transitively pull the 6-arg `op_profiler_serialize` that breaks the
5-arg TracyC.h). Backend::Copy now calls `CopyDeviceDeviceIfCapture` first;
default tests still 21/21, 814/814 (the path is capture-gated, inert
otherwise).

**Measured:** the R2 copy fix **works** — the capture probe now gets past
the `EnsureHostBytes` readback (`[TT-TRACE] device->device copy (capture-safe)`
fires, no more `Reads are not supported` fatal). The **new** fatal is one
layer deeper:

```
TT_FATAL: Cannot load new binaries during trace capture.
This program is not yet in program cache. Warm up before capturing a trace.
```

This is the **ttnn program-cache warm-up** requirement (Q3 in the original
trace-runner spike, deferred then). ttnn `begin_trace_capture` requires
every op shape in the captured region to be JIT-compiled (program-cache
warm) BEFORE capture begins; the decode-graph framework's single eager
warmup step does not warm the exact shapes the captured path uses (or my
new `ttnn::empty` introduces an un-warmed program).

This is a known ttnn trace discipline with an established pattern (warm
identical shapes via an eager run that hits the same ops), NOT an unknown.
It is the concrete R3 target — and it means the host-free forward *is*
achievable: R1 (thresholds) + R2 (device-copy, landed) clear the readback
blockers; R3 (warm-up) is the last gate before capture can complete.

### R3 update (2026-08-13): warm-up WORKS — capture now runs ops

Fixed the warm-up: `CopyDeviceDeviceIfCapture` now runs whenever
`VT_TT_HOST_FREE_DECODE` is set (not just during capture), so the eager
warmup step also exercises `ttnn::empty`+`ttnn::copy`, compiling them into
the program cache. Also calls `device.enable_program_cache()` once on the
first host-free use (the ttnn trace precondition).

Measured: the "Cannot load new binaries" fatal is gone. Capture now enters
the forward and runs ops:

  device->device copy (eager warmup)
  BeginCapture
  device->device copy (ForwardLayers opening Copy — R2 holds)
  EnsureHostBytes DURING CAPTURE x6   <- next readback blockers
  CastBf16
  RmsNorm                              <- ops run during capture
  TT_FATAL: Writes are not supported during trace capture  <- a buffer write

R2 + R3 together got capture past the first Copy and into the layer ops.
Two new, expected, mechanical blockers surfaced:

1. 6 more EnsureHostBytes readbacks — every Backend::Copy inside
   ForwardLayers (weight uploads, residual init) hits host-staging. Same
   R2 fix at each site.
2. Writes not supported — DBuf::Zero calls Backend::Memset (host memset),
   a host write inside the captured region. Needs a device-zero path or
   pre-zero outside capture.

Conclusion: capture on TT is achievable and now demonstrated working past
the first two blocker layers. Remaining work is converting each
host-staging site (Backend::Copy, Backend::Memset/DBuf::Zero) in the layer
loop to device-resident — mechanical, not research. The bisection
instrumentation surfaces each site in order.

### R3b update (2026-08-13): copy + zero-fill done; device-allocation is the structural blocker

Added MemsetDeviceIfCapture (on-device ttnn::zeros for DBuf::Zero), fixed a
null-deref (std::optional<Tensor>). Default tests 21/21.

Measured: the 6 EnsureHostBytes readbacks are GONE. The sequence now:
  device->device copy (eager warmup)
  BeginCapture
  device->device copy (ForwardLayers opening)
  device zero-fill (DBuf::Zero)
  Writes are not supported during trace capture   <- structural blocker

The Writes fatal is ttnn forbidding device allocations during capture
(same as CUDA's no-cudaMalloc-during-capture). The TT ops do per-call
from_vector host->device uploads (weights/inputs) and ttnn::empty scratch
inside kernels; those are fresh device writes, forbidden during capture.
CUDA solves this with a pre-warmed DevicePool + fixed-address persistent
weight buffers; TT has no equivalent, and its weights are not in stable
device buffers persisting across warmup->capture.

This is the structural hard part: a TT scratch-pool analogue + stable
weight residency so no allocation/upload happens during capture. Real
engineering, the natural scope of a dedicated row.

COMPLETE BLOCKER MAP (the experiment's deliverable):
1. RmsNorm/RoPE host thresholds -> R1 (flip, done)
2. Backend::Copy host readback -> R2 (device->device copy, done)
3. ttnn program-cache warm-up -> R3 (enable + eager-warm, done)
4. Backend::Memset/DBuf::Zero host write -> R3b (device zero-fill, done)
5. per-op device allocation/upload (from_vector, ttnn::empty) -> REMAINING;
   needs a TT scratch pool + stable weight residency

Items 1-4 landed, measured, inert-by-default. Item 5 is the open
engineering gate before decode capture can complete and replay tok/s can
be measured.

### Upstream investigation (2026-08-13): item 5 may be a non-issue on newer ttnn

Searched tt-metal/tt-nn issues. The "Writes are not supported during trace
capture" fatal is a **known limitation with an upstream fix**:

- **tt-metal issue [#13690](https://github.com/tenstorrent/tt-metal/issues/13690)**
  "Enable allocation of new buffers with a warning to allow running decode
  with trace and prefill without trace" — filed by Tenstorrent **for vLLM**
  (referenced by tenstorrent/vllm#14). The exact use case: interleaving a
  traced decode with untraced prefill needs buffers allocated while a trace
  is live.
- **Fixed in PR [#13696](https://github.com/tenstorrent/tt-metal/pull/13696)**
  (commit `f0b2483`): instead of `TT_FATAL`, it now prints a warning and
  allows the allocation, safe as long as untraced intermediates are consumed
  before a trace runs.
- **This build does NOT have the fix** — `fd_mesh_command_queue.cpp:760`
  still uses `TT_FATAL(!trace_id_.has_value(), "Writes are not supported...")`.

**Implication:** bumping the tt-metal build to one including #13696 may
eliminate item 5 entirely (the upload-during-capture becomes a warning,
not a crash). Worth testing before building a scratch-pool subsystem.

Additionally:
- `ttnn::create_device_tensor(spec, device)` (from
  `graph/graph_query_op_constraints.hpp`) allocates an empty device tensor
  **without** a host→device write — the capture-safe allocation pattern.
  The canonical capture sequence (graph_query_op_runtime.hpp:76-90) uses it
  to create input tensors pre-capture, warm, then capture. Our ops use
  `from_vector` (which writes); converting uploads to
  `create_device_tensor` + a pre-capture warm would also avoid the fatal.
- `TraceBufferPool` (PR #18523) — ttnn already has trace buffer management
  infrastructure.

**Two concrete paths to clear item 5, in order of effort:**
1. **Bump tt-metal** to a build with #13696 and re-run the capture probe.
   If the warning-only path works, capture completes and we get replay
   tok/s immediately — no vllm.cpp changes.
2. If the bump is not possible or insufficient: convert the TT ops' weight
   uploads to pre-capture `create_device_tensor` (stable, pinned addresses
   — the "pin addresses for a stable pool" approach) so no write happens
   during capture. Bounded work, no new subsystem.

### Correction (2026-08-13): bump will NOT help — our fatal is a write guard, not the allocator guard

Verified `f0b2483` IS an ancestor of the installed tt-metal build (the #13690
fix is present). But #13690 only relaxed the **allocator** (`allocator.cpp`
+ `device.cpp`) — it allows **buffer allocation** during a live trace.
Our fatal is at `fd_mesh_command_queue.cpp:760`, the **`enqueue_write`**
(host→device write) guard, which is a *separate* assertion that #13690 did
NOT touch (all three `Writes are not supported` fatals in
`fd_mesh_command_queue.cpp` are still hard `TT_FATAL`s).

So bumping tt-metal will not clear item 5. The real fix is path 2: avoid
the `enqueue_write` during capture by pre-allocating device tensors with
`create_device_tensor` (which does not write) and uploading their contents
*before* capture, so the captured ops reference stable device buffers with
no host→device write. This is the "pin addresses for a stable pool" approach
— confirmed feasible by `ttnn::create_device_tensor` existing and being the
canonical capture-safe allocation path (graph_query_op_constraints.hpp).

### Architecture answer (2026-08-13): mirror the tt-metal vLLM plugin's design

Read the official Tenstorrent vLLM plugin
(tt/vllm/plugins/vllm-tt-plugin/.../model_runner.py). It solves this
exactly, and the answer is a vllm.cpp-side architecture change, not a
tt-metal patch:

1. Two-phase warmup (model_runner.py:3216-3262): Phase 1 compiles all ops
   into the program cache with enable_trace=False; Phase 2 captures with
   every op compiled. Our Qwen3DenseDecodeGraph already does the
   single-step version.
2. Persistent device tensors at warmup shape (model_runner.py:480-487):
   block tables, positions, inputs allocated as persistent ttnn device
   tensors at the max padded shape during warmup so capture replays against
   stable device addresses.
3. Per-step inputs pushed BEFORE the captured region, not inside it: the
   plugin uses ttnn.copy_host_to_device_tensor (= C++ copy_to_device ->
   enqueue_write_tensor) to populate stable buffers. Crucially,
   copy_to_device hits the SAME enqueue_write path that fatals during
   capture (fd_mesh_command_queue.cpp:760), so the plugin calls it BEFORE
   capture (warmup populate) and BEFORE each replay (per-step refresh),
   NEVER inside the captured region.

Implication: our Backend::Copy/EnsureHostBytes fatal during capture is
fundamental -- copy_to_device itself would fatal there too. The fix is
architectural: the captured ForwardLayers region must reference only
pre-allocated, pre-populated device tensors. Per-step inputs (token id,
position, slot mapping, block table) must be written to stable device
buffers BEFORE ReplayGraph, the same way CUDA's decode graph does (its
SizeSlot::Refresh writes host buffers that a captured async-copy re-reads,
qwen3.cpp:528).

So path-2 is: make the TT decode-graph slot hold persistent device tensors
for inputs, populate them before capture/replay via copy_to_device, and
ensure the captured ops read those device tensors without any internal
from_vector/to_vector. That is the real host-free forward -- a bounded
architecture port of the plugin's design, not a new subsystem and not an
upstream tt-metal fix.

### Steady-state perf baselines (2026-08-14, real Blackhole P150)

Qwen3-0.6B, `vllm-cli --prompt "Hello" --max-tokens 64 --repeat 3`:

| config | warm tok/s (runs 2/3) | ms/tok |
|--------|----------------------|--------|
| default hybrid | **7.30 / 7.31** | ~137 |
| all-device eager (`VT_TT_HOST_FREE_DECODE=1` + `VLLM_CPP_QWEN3_DENSE_DECODE_GRAPH=0`) | **6.87 / 6.92** | ~145 |

Two corrections to the earlier smoke numbers, both measured:

1. **The 4-token smoke (12.5 tok/s) is NOT steady state.** At 64 tokens the
   same config sustains 7.3 tok/s — per-token cost grows with context (PA
   decode over a growing KV), so the handoff's ~12.3 and our 12.5 both
   over-report. The honest reference for capture work is 7.3.
2. **The all-device eager cost at steady state is ~0.4 tok/s (~6%), not the
   ~1.8 tok/s (~14%) the 4-token smoke suggested.** This materially improves
   the capture break-even: capture only needs to recover ~6% of eager time
   to beat the hybrid baseline at 64-token scale — a much lower bar than
   the spike's 14% framing assumed.

(The `DECODE_GRAPH=0` opt-out is required for the all-device run: with the
flag on, `support_static_graph_mode()` flips true and the decode-graph
framework would otherwise attempt capture and abort on item 5.)

Mistral-7B-v0.3 reference on the same box: 4.26 tok/s warm at 32 tokens
(recorded in tenstorrent-mistral.md).

**What is still NOT measurable until item 5 lands: capture/replay tok/s.**
Capture cannot complete (the enqueue_write fatal fires mid-forward), so the
replay number — the actual payoff — remains open. The numbers above bound
it: replay must exceed 7.3 (the hybrid eager baseline) to be a win, and
starts from a 6.9 eager floor on the all-device path.

### Item 5 progress (2026-08-14): two sites fixed; frontier now mid-layer-0, at rope cos/sin

Instrumented all 16 `from_vector` sites (capture-gated `[TT-UP]` prints,
incl. ptr+shape on UploadRows) and iterated the bisection. Two real item-5
fixes landed:

1. **ttnn::zeros is NOT capture-safe** (creation.cpp `full_impl` host-fills
   + `to_device()` = an enqueue_write) — my own R3b helper was an offender.
   Fixed the plugin way: a persistent ZERO TENSOR CACHE (keyed by
   shape/dtype/layout) created OUTSIDE capture, primed during the eager
   warmup by `EnsureDevice2D`, and applied in-region by
   `ttnn::copy(zero, shadow)` — a device->device program that is captured
   and replayed. Cache-miss during capture is a hard VT_CHECK (must warm
   first), which is exactly what forced the priming fix.
2. **QkvSplit's device path was already correct** (ttnn::slice +
   CommitDevice2D) — the earlier suspicion was wrong; with MatmulBT's
   shadow it fires and hands q/k/v shadows downstream.

**Measured frontier after both fixes** — capture now runs DEEP into
layer 0 and dies at a precisely-identified site:

```
BeginCapture -> device-copy -> zero-fill -> [6 EnsureHostBytes readbacks
= the weight DBuf copies, handled by R2] -> CastBf16 -> RmsNorm ->
MatmulBT -> QkvSplit -> RmsNorm(q-norm) -> RmsNorm(k-norm)
-> [TT-UP] UploadRows ptr=... rows=16 cols=64   <- THE blocker
-> TT_FATAL: Writes are not supported during trace capture
```

`[16, 64]` is the **RoPE cos/sin table** (Hq=16, rot/2=64): host-computed
by `BuildCosSinFromPositions` inside `RopeNeoxKernel` and uploaded
in-region. This is the cos/sin sub-blocker the R1 spec predicted, and it
is the plugin's "per-step input" case: the fix is a PERSISTENT device
cos/sin buffer populated before capture/replay (positions change per step,
so the decode-graph driver must copy_to_device the step's rows BEFORE
ReplayGraph — the same pattern as CUDA's SizeSlot::Refresh async-copy).

**Remaining sites after rope (not yet reached by the bisection, expected
from the readback map):** ReshapeAndCache's KV writes (host-staged),
PagedAttention's metadata uploads, the lm_head/logits path. Each is the
same pattern; the rope fix establishes the template.

Status: item 5 is now a SCOPED multi-site port (rope cos/sin + RAC + PA
metadata + logits), with two sites landed and the third precisely
characterized. Not complete; the replay-tok/s payoff measurement remains
blocked behind the remaining sites.

### Item 5: rope cos/sin SOLVED (2026-08-14, measured on card)

The rope blocker took three fixes working together:

1. **Persistent cos/sin cache** keyed by (tokens*heads, rot/2), entries
   created/refreshed OUTSIDE capture, replayed in-region via the captured
   program (no per-call upload). Content-identity checked against the exact
   bytes the kernel will use — a stale table is a hard VT_CHECK during
   capture, never silent corruption.
2. **Driver warm hook** `WarmRopeCosSin(positions, ...)` called from the
   decode-graph driver's Refresh slot (qwen3.cpp, right after
   SizeSlot::Refresh) — THE per-step populate point, the exact plugin
   SizeSlot::Refresh analogue. Crucially it warms the UNPADDED T-row
   positions (what si.positions/rope sees), not the padded ppos — the
   first attempt used ppos and always missed.
3. **Byte-exact content**: the captured rope path (RopeFromCache, the
   default VT_QWEN3_ROPE_CACHE route) reads cos/sin from the per-step bf16
   CACHE table (RopeCosSinCacheKernel's StoreElemF32 rounds f32->bf16), so
   the warm content must round-trip through bf16 (BF16ToF32(F32ToBF16(v)))
   — f32 warm content never matches (cos(1)=0.540302 f32 vs 0.539062 bf16).

Measured: rope cache **HIT for both q (16x64) and k (8x64)** during
capture (`content_eq=1`), capture proceeds PAST rope. Also discovered en
route: the dense decode path routes rope through RopeFromCacheKernel (not
RopeNeoxKernel) by default — the first debug print in the wrong kernel
never fired, which is what exposed it.

**New frontier: ReshapeAndCache** — the next fatal is a to_vector readback
inside RAC (the KV-write path), right after rope in layer 0. This is the
"queued" RAC item from the original plan, now live. After RAC: PA metadata,
then the logits path. RAC is the most delicate remaining site: KV writes
inside a captured+replayed region also raise a REPLAY-SEMANTICS question
(every replay re-appends the same KV row) that must be answered alongside
the mechanical fix — the CUDA graph solves this by capturing the append
against fixed slot addresses refreshed per step.

Default-path safety re-verified after all rope changes: 23/23 cases,
830/830 assertions.
