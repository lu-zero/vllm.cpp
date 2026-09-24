// Tenstorrent op providers — the ttnn adapter layer (BACKEND-TENSTORRENT,
// .agents/specs/tenstorrent-backend.md). vllm.cpp original; no upstream
// mirror (vLLM has no Tenstorrent platform). Op table: OPT-125m's 9 ops plus
// the Qwen3-dense deltas (kRmsNorm first; kSiluAndMul / RoPE / Cast next),
// matching Metal's OPT→Qwen3 sequencing. ttnn for compute where available;
// host-staged pure data-movement / attention for the remainder (see
// HOST-STAGED OPS note below).
//
// SCOPE: F32 for the W0 path unless noted. kAdd allows rank-1 bias
// broadcast; kLayerNorm optional rank-1 weight/bias; kRmsNorm weight +
// optional residual stream; kEmbedding i32/i64 ids. Every other shape/dtype
// is a VT_CHECK failure — no CPU reference tier (UnifiedMemory()==false).
//
// HOST-STAGED OPS (kReshapeAndCache, kPagedAttention): this backend's Alloc is
// host memory (tenstorrent_backend.cpp). ReshapeAndCache is a pure contiguous /
// stride-aware page write. PagedAttention uses the CPU-oracle f32 softmax over
// the host-resident paged cache; mapping vLLM's block-table contract onto
// ttnn::sdpa_decode is deferred. kQkvSplit is hybrid: device-slice when the
#include "vt/tenstorrent/tenstorrent_internal.h"

namespace vt::tenstorrent {
namespace {
// Publish a device result as the current value of `out` WITHOUT downloading
void CommitDevice2D(Tensor& out, ttnn::Tensor dev) {
  VT_CHECK(out.rank == 2 && out.IsContiguous(),
           "tenstorrent: CommitDevice2D expects contiguous rank-2 out");
  CommitDeviceLogical2D(out, std::move(dev), static_cast<uint32_t>(out.shape[0]),
                        static_cast<uint32_t>(out.shape[1]));
}

}  // namespace

// Host wrote `out` in place — drop any device shadow.
void CommitHost(Tensor& out) {
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(out.data);
  if (s == nullptr) return;
  s->host_current = true;
  s->device_current = false;
  s->device = std::nullopt;
  s->conv_transposed = false;
  s->device_reserved = false;  // real bytes now — the upload contract applies
}

namespace {

// Device compute: keep result on device (CommitDevice2D). Host round-trip only
// when the consumer is a host-staged op (EnsureHost) or an untracked buffer.
void MatmulKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  TT_OP_TRACE("Matmul");
  VT_CHECK(a.rank == 2 && b.rank == 2 && out.rank == 2,
           "tenstorrent kMatmul: only rank-2 tensors are supported in W0");
  VT_CHECK(IsFloatDType(a.dtype) && IsFloatDType(b.dtype) &&
               (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kMatmul: float in, f32/bf16 out");
  const uint32_t M = static_cast<uint32_t>(a.shape[0]);
  const uint32_t K = static_cast<uint32_t>(a.shape[1]);
  const uint32_t N = static_cast<uint32_t>(b.shape[1]);
  VT_CHECK(b.shape[0] == K, "tenstorrent kMatmul: a/b inner dimension mismatch");
  VT_CHECK(out.shape[0] == M && out.shape[1] == N, "tenstorrent kMatmul: out shape mismatch");
  VT_CHECK(a.IsContiguous() && b.IsContiguous() && out.IsContiguous(),
           "tenstorrent kMatmul: strided (non-contiguous) tensors are not supported in W0");

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_a = EnsureDevice2D(a, device);
  ttnn::Tensor dev_b = EnsureMatmulWeightDevice(b, device);
  if (dev_b.dtype() == ttnn::DataType::BFLOAT8_B) Bfp8MatmulUse();
  ttnn::Tensor dev_c = ttnn::operations::matmul::matmul(dev_a, dev_b);
  CommitDevice2D(out, std::move(dev_c));
}

// kMatmulBT: `b` is a [N,K] row-major torch nn.Linear weight; computes
// `a @ b^T` (cpu_ops.cpp's MatmulBTKernel contract). ttnn's matmul() already
// exposes a transpose_b flag, so this is the same sequence as kMatmul with
// that flag flipped — no separate upload shape needed since `b` is uploaded
// in its native [N,K] layout and ttnn transposes on device.
void MatmulBTKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  TT_OP_TRACE("MatmulBT");
  VT_CHECK(a.rank == 2 && b.rank == 2 && out.rank == 2,
           "tenstorrent kMatmulBT: only rank-2 tensors are supported in W0");
  VT_CHECK(IsFloatDType(a.dtype) && IsFloatDType(b.dtype) &&
               (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kMatmulBT: float in, f32/bf16 out");
  const uint32_t M = static_cast<uint32_t>(a.shape[0]);
  const uint32_t K = static_cast<uint32_t>(a.shape[1]);
  const uint32_t N = static_cast<uint32_t>(b.shape[0]);
  VT_CHECK(b.shape[1] == K, "tenstorrent kMatmulBT: a/b inner dimension mismatch");
  VT_CHECK(out.shape[0] == M && out.shape[1] == N, "tenstorrent kMatmulBT: out shape mismatch");
  VT_CHECK(a.IsContiguous() && b.IsContiguous() && out.IsContiguous(),
           "tenstorrent kMatmulBT: strided (non-contiguous) tensors are not supported in W0");

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_a = EnsureDevice2D(a, device);
  ttnn::Tensor dev_b = EnsureMatmulWeightDevice(b, device);
  if (dev_b.dtype() == ttnn::DataType::BFLOAT8_B) Bfp8MatmulUse();
  ttnn::Tensor dev_c =
      ttnn::operations::matmul::matmul(dev_a, dev_b, /*transpose_a=*/false, /*transpose_b=*/true);
  CommitDevice2D(out, std::move(dev_c));
}

// kAdd: elementwise add, plus the rank-1 `b` row-broadcast form used for
// nn.Linear bias (cpu_layernorm.cpp's AddKernel contract). ttnn::add needs
// same-rank operands, so the broadcast case uploads `b` replicated into a
// [rows, d] tile rather than relying on ttnn's own broadcast rules — keeps
// this kernel's behavior pinned to the CPU reference rather than to
// whatever ttnn::add happens to support today.
void AddKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  TT_OP_TRACE("Add");
  VT_CHECK(a.rank == 2 && out.rank == 2, "tenstorrent kAdd: `a`/`out` must be rank-2 in W0");
  VT_CHECK(b.rank == 2 || b.rank == 1, "tenstorrent kAdd: `b` must be rank-1 or rank-2 in W0");
  VT_CHECK(IsFloatDType(a.dtype) && IsFloatDType(b.dtype) &&
               (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kAdd: float in, f32/bf16 out");
  VT_CHECK(a.IsContiguous() && b.IsContiguous() && out.IsContiguous(),
           "tenstorrent kAdd: strided (non-contiguous) tensors are not supported in W0");
  const uint32_t rows = static_cast<uint32_t>(a.shape[0]);
  const uint32_t d = static_cast<uint32_t>(a.shape[1]);
  VT_CHECK(out.shape[0] == rows && out.shape[1] == d, "tenstorrent kAdd: out shape mismatch");
  const bool bcast = b.rank == 1;
  VT_CHECK(bcast ? b.shape[0] == d : (b.shape[0] == rows && b.shape[1] == d),
           "tenstorrent kAdd: `b` shape mismatch");

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_a = EnsureDevice2D(a, device);
  ttnn::Tensor dev_b;
  if (bcast) {
    EnsureHost(b);
    std::vector<float> replicated(static_cast<size_t>(rows) * d);
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t c = 0; c < d; ++c)
        replicated[static_cast<size_t>(r) * d + c] = LoadElemF32(b, c);
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
      std::fprintf(stderr, "[TT-UP] AddKernel from_vector WRITE during capture\n");
    dev_b = ttnn::Tensor::from_vector<float>(replicated, TileSpecOf(rows, d), &device);
  } else {
    dev_b = EnsureDevice2D(b, device);
  }
  ttnn::Tensor dev_c = ttnn::add(dev_a, dev_b);
  CommitDevice2D(out, std::move(dev_c));
}

// kRelu: elementwise max(0, x) (cpu_layernorm.cpp's ReluKernel contract).
void ReluKernel(Queue&, Tensor& out, const Tensor& x) {
  VT_CHECK(x.rank == 2 && out.rank == 2, "tenstorrent kRelu: only rank-2 tensors are supported in W0");
  VT_CHECK(IsFloatDType(x.dtype) && (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kRelu: float in, f32/bf16 out");
  VT_CHECK(x.shape[0] == out.shape[0] && x.shape[1] == out.shape[1],
           "tenstorrent kRelu: out shape mismatch");
  VT_CHECK(x.IsContiguous() && out.IsContiguous(),
           "tenstorrent kRelu: strided (non-contiguous) tensors are not supported in W0");

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_x = EnsureDevice2D(x, device);
  ttnn::Tensor dev_y = ttnn::relu(dev_x);
  CommitDevice2D(out, std::move(dev_y));
}

}  // namespace

// ---- Persistent embedding-table shadows (ROW_MAJOR BF16 on device) --------
// The vocab table is multi-hundred MB for Qwen3; re-uploading every forward
// was a pure tax. Keyed by host table base; invalidated by MarkHostWritten /
// UnregisterHostBuffer. The EmbedTableShadow struct lives in
// tenstorrent_internal.h (split stage 5: the capture TU reads the map too).
std::mutex& EmbedTableMutex() {
  static std::mutex m;
  return m;
}
std::map<uintptr_t, EmbedTableShadow>& EmbedTableShadows() {
  static std::map<uintptr_t, EmbedTableShadow>* m = new std::map<uintptr_t, EmbedTableShadow>(); // never destroyed (#1486)
  return *m;
}
namespace {
void DropEmbedTableShadow(void* host) {
  if (host == nullptr) return;
  std::lock_guard<std::mutex> g(EmbedTableMutex());
  EmbedTableShadows().erase(reinterpret_cast<uintptr_t>(host));
}

ttnn::Tensor EnsureEmbedTableDevice(const Tensor& table, MeshDevice& device) {
  VT_CHECK(table.rank == 2 && table.IsContiguous(), "EnsureEmbedTable: rank-2 contiguous");
  const uint32_t vocab = static_cast<uint32_t>(table.shape[0]);
  const uint32_t h = static_cast<uint32_t>(table.shape[1]);
  {
    std::lock_guard<std::mutex> g(EmbedTableMutex());
    auto it = EmbedTableShadows().find(reinterpret_cast<uintptr_t>(table.data));
    if (it != EmbedTableShadows().end() && it->second.device.has_value() &&
        it->second.vocab == vocab && it->second.h == h) {
      return *it->second.device;
    }
  }
  EnsureHost(table);
  std::vector<float> host_table = ToHostF32(table);
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr, "[TT-UP] EnsureEmbedTableDevice from_vector WRITE during capture\n");
  AllocTraceSnapshot(device, "EnsureEmbedTableDevice/pre");
  ttnn::Tensor dev_table = ttnn::Tensor::from_vector<float>(
      host_table,
      SpecOf(tt::tt_metal::Shape({vocab, h}), ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR),
      &device);
  std::lock_guard<std::mutex> g(EmbedTableMutex());
  EmbedTableShadow& s = EmbedTableShadows()[reinterpret_cast<uintptr_t>(table.data)];
  s.device = dev_table;
  s.vocab = vocab;
  s.h = h;
  AllocTraceSnapshot(device, "EnsureEmbedTableDevice/post");
  return dev_table;
}

// kEmbedding: row gather `out[i,:] = table[ids[i],:]` (cpu_ops.cpp
// EmbeddingKernel contract). Two layout departures from the TILE/BFLOAT16
// linear ops, forced by ttnn::embedding's validate path:
//   1. ids upload as ROW_MAJOR UINT32 (ttnn rejects i32/i64; vt still accepts
//      kI32/kI64 at the seam and converts host-side, matching Metal/Vulkan).
//   2. table is ROW_MAJOR BFLOAT16 (cached on device after first use).
// Parameter order at the ttnn call is (ids, table) — reversed from
// vt::EmbeddingFn's (table, ids). Output is TILE so the next matmul can keep
// the activation device-resident without a host round-trip.
void EmbeddingKernel(Queue&, Tensor& out, const Tensor& table, const Tensor& ids) {
  TT_OP_TRACE("Embedding");
  VT_CHECK(table.rank == 2 && ids.rank == 1 && out.rank == 2,
           "tenstorrent kEmbedding: table rank-2, ids rank-1, out rank-2");
  VT_CHECK(IsFloatDType(table.dtype) && (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kEmbedding: float table, f32/bf16 out");
  VT_CHECK(ids.dtype == DType::kI32 || ids.dtype == DType::kI64,
           "tenstorrent kEmbedding: ids must be i32 or i64");
  VT_CHECK(table.IsContiguous() && ids.IsContiguous() && out.IsContiguous(),
           "tenstorrent kEmbedding: strided (non-contiguous) tensors are not supported");
  const uint32_t vocab = static_cast<uint32_t>(table.shape[0]);
  const uint32_t h = static_cast<uint32_t>(table.shape[1]);
  const uint32_t t = static_cast<uint32_t>(ids.shape[0]);
  VT_CHECK(out.shape[0] == t && out.shape[1] == h, "tenstorrent kEmbedding: out shape mismatch");

  EnsureHost(ids);
  std::vector<uint32_t> host_ids(t);
  if (ids.dtype == DType::kI32) {
    const int32_t* p = ids.Ptr<int32_t>();
    for (uint32_t i = 0; i < t; ++i) {
      VT_CHECK(p[i] >= 0 && static_cast<uint32_t>(p[i]) < vocab,
               "tenstorrent kEmbedding: id out of range");
      host_ids[i] = static_cast<uint32_t>(p[i]);
    }
  } else {
    const int64_t* p = ids.Ptr<int64_t>();
    for (uint32_t i = 0; i < t; ++i) {
      VT_CHECK(p[i] >= 0 && static_cast<uint64_t>(p[i]) < vocab,
               "tenstorrent kEmbedding: id out of range");
      host_ids[i] = static_cast<uint32_t>(p[i]);
    }
  }
  MeshDevice& device = SharedMeshDevice();
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr, "[TT-UP] EmbeddingKernel from_vector WRITE during capture\n");
  ttnn::Tensor dev_ids = ttnn::Tensor::from_vector<uint32_t>(
      host_ids, SpecOf(tt::tt_metal::Shape({t}), ttnn::DataType::UINT32, ttnn::Layout::ROW_MAJOR),
      &device);
  ttnn::Tensor dev_table = EnsureEmbedTableDevice(table, device);
  // TILE output → CommitDevice2D so the first residual/RMS/matmul reuses it.
  ttnn::Tensor dev_out = ttnn::embedding(dev_ids, dev_table, /*pad_token=*/std::nullopt,
                                         /*layout=*/ttnn::Layout::TILE);
  // embedding may return [t, h] or a higher-rank view; normalize to [t, h].
  if (dev_out.logical_shape().rank() != 2 ||
      dev_out.logical_shape()[0] != t || dev_out.logical_shape()[1] != h) {
    dev_out = ttnn::reshape(dev_out, ttnn::Shape({t, h}));
  }
  // HOST-FREE-DECODE: when the caller's buffer already carries a CURRENT
  // device shadow of the same shape (the decode-graph driver's PERSISTENT
  // hidden buffer), refresh that shadow IN PLACE (device->device copy) so
  // its device address never moves. A captured region reads the address
  // recorded at capture time; replacing the shadow here would leave every
  // replay reading the capture-step embedding. First call (no shadow yet)
  // commits normally.
  bool in_place = false;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(out.data);
    in_place = s != nullptr && s->device_current && s->device.has_value() &&
               s->dev_rows == t && s->dev_cols == h;
    if (in_place) {
      ttnn::copy(dev_out, *s->device);
      s->host_current = false;
    }
  }
  if (!in_place) CommitDevice2D(out, std::move(dev_out));
}

// kKeepQuantDecode: packed-block decode as a device compute chain — the packed
// stream is decoded on the device, not staged through a host decode. Contracts
// are DequantQ4_K/DequantQ5_K/DequantQ6_K/DequantQ8_0 (cpu_quant_dequant.cpp;
// llama.cpp dequantize_row_q4_K:1471, q5_K:1673, q6_K:1881, q8_0:495). Q4_K:
// per block_q4_K {f16 d; f16 dmin; u8 scales[12]; u8 qs[128]}, y[g*32+l] =
// d1*(nib) - m1 with d1 = d*sc, m1 = dmin*mm, sc/mm from GetScaleMinK4(is,
// scales), groups 0..7 over the 8 32-nibble lanes (low nibbles first, then
// high nibbles, of the same 32 bytes). Every unpack step is device work:
// bitwise masks/shifts for the nibbles, scale bytes, and f16 halves;
// reshape/permute only reorder lanes; d/dmin widen f16->f32 exactly via the
// integer bit-construction chain of vt::F16ToF32 (loader
// minimax_h3_vae_loader.cpp:47): normal/inf bits are sign | (exp+112)<<23 |
// mant<<13, subnormals ride the exact integer->f32 typecast times 2^-24,
// zero keeps its sign bit, and one i32->f32 bitcast materializes the f32.
// The one divergence from F16ToF32: a NaN scale's payload canonicalizes to
// inf across the SFPU bitcast (hardware pin) — corrupt-GGUF territory, and
// every finite pattern is bit-exact. nibbles and scale factors typecast to
// f32 exactly (values <= 255); d1/m1 and the final y = d1*nib - m1 are
// separate f32 ttnn ops (no FMA), the same IEEE order as the host under
// -ffp-contract=off.
//
// THE SHARED DECODE (KEEPQUANT W2, generalized W3): the packed-block -> f32
// chains, each returning the repaired f32 {rows, nb*elems} in ROW_MAJOR.
// KeepQuantDecode commits it to the host-visible output; the quant matmul
// (MatmulBTQuantKernel below) consumes it as the weight operand. One decode
// per encoding, two consumers, one numerics authority:
//   Q4_K 144B/256 and Q5_K 176B/256: {f16 d; f16 dmin; u8 scales[12]; ...}
//   — Q5_K inserts qh[32] between scales and qs (the 5th bit, +16) and its
//   tail is Q4_K's verbatim; Q6_K 210B/256: {u8 ql[128]; u8 qh[64]; i8
//   scales[16]; f16 d} — SIGNED 8-bit scales, 16 sub-blocks of 16, no min
//   term, q = nib6 - 32; Q8_0 34B/32: {f16 d; i8 qs[32]} — plain int8 times
//   the f16 scale. 210 and 34 are not multiples of four, so those blocks pad
//   to 53 / 9 words (2 pad bytes at the stream tail, zero-filled).
//
// STAGING (W3): the packed stream is staged as a resident i32 word shadow ONCE
// per weight (EnsureKeepQuantWords below, the EnsureMatmulWeightDevice
// persistent-shadow pattern), so the per-call decode runs entirely on-core
// from the resident words — no host repack, no from_vector upload. The -0.0f
// signed-zero repair constant is built by BIT ops from the zero-cache's +0
// (multiply canonicalizes -0 to +0 on the SFPU, so the sign bit is set below
// the float domain), because a per-call from_vector of the constant would be
// exactly the captured-graph write this wave removes.



// Upload a rank-1 affine vector as TILE BFLOAT16 [1, d], caching on the weight's
// host buffer slot so RmsNorm/LayerNorm do not re-upload every layer call.
// ttnn's TILE-gamma path requires padded height == tile_height (32); from_vector
// with TILE pads a [1,d] tensor to that.
ttnn::Tensor EnsureAffine1D(const Tensor& t, uint32_t d, MeshDevice& device) {
  VT_CHECK(t.rank == 1 && t.IsContiguous() && t.shape[0] == static_cast<int64_t>(d),
           "tenstorrent EnsureAffine1D: rank-1 [d] contiguous");
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(t.data);
    // Hit only for the BASE pointer (same interior-view hazard as
    // EnsureDevice2D — a differently-offset rank-1 view must not consume the
    // base's staged affine).
    if (s != nullptr && t.data == s->host && s->device_current &&
        s->device.has_value() && s->dev_rows == 1 && s->dev_cols == d) {
      return *s->device;
    }
  }
  EnsureHost(t);
  std::vector<float> host(d);
  for (uint32_t i = 0; i < d; ++i) host[i] = LoadElemF32(t, i);
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr, "[TT-UP] EnsureAffine1D from_vector WRITE during capture\n");
  ttnn::Tensor dev = ttnn::Tensor::from_vector<float>(
      host, SpecOf(tt::tt_metal::Shape({1, d}), ttnn::DataType::BFLOAT16, ttnn::Layout::TILE),
      &device);
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(t.data);
  if (s != nullptr && t.data == s->host) {
    s->device = dev;
    s->dev_rows = 1;
    s->dev_cols = d;
    s->device_current = true;
    s->host_current = true;
    s->device_reserved = false;  // real bytes staged — the reservation is spent
  }
  return dev;
}


// kLayerNorm: per-row mean/var over the last dim (cpu_layernorm.cpp
// LayerNormKernel / ATen native_layer_norm). Biased (1/N) variance; optional
// rank-1 weight/bias (elementwise_affine). Uses ttnn::layer_norm with the
// same TILE/BFLOAT16 upload path as the linear ops; eps comes from
// LayerNormArgs (OPT default 1e-5, not ttnn's 1e-12 default).
void LayerNormKernel(Queue&, Tensor& out, const Tensor& x, const Tensor* weight,
                     const Tensor* bias, const LayerNormArgs& args) {
  TT_OP_TRACE("LayerNorm");
  VT_CHECK(x.rank == 2 && out.rank == 2,
           "tenstorrent kLayerNorm: only rank-2 tensors are supported in this step");
  VT_CHECK(IsFloatDType(x.dtype) && (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kLayerNorm: float in, f32/bf16 out");
  VT_CHECK(x.shape[0] == out.shape[0] && x.shape[1] == out.shape[1],
           "tenstorrent kLayerNorm: out shape mismatch");
  VT_CHECK(x.IsContiguous() && out.IsContiguous(),
           "tenstorrent kLayerNorm: strided (non-contiguous) tensors are not supported");
  VT_CHECK(args.eps >= 0.0f, "tenstorrent kLayerNorm: eps must be non-negative");
  const uint32_t d = static_cast<uint32_t>(x.shape[1]);
  for (const Tensor* p : {weight, bias}) {
    if (p == nullptr) continue;
    VT_CHECK(p->rank == 1 && p->shape[0] == d,
             "tenstorrent kLayerNorm: weight/bias must be rank-1 [D]");
    VT_CHECK(IsFloatDType(p->dtype), "tenstorrent kLayerNorm: float weight/bias");
    VT_CHECK(p->IsContiguous(), "tenstorrent kLayerNorm: weight/bias must be contiguous");
  }

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_x = EnsureDevice2D(x, device);
  std::optional<ttnn::Tensor> dev_w;
  std::optional<ttnn::Tensor> dev_b;
  if (weight != nullptr) dev_w = EnsureAffine1D(*weight, d, device);
  if (bias != nullptr) dev_b = EnsureAffine1D(*bias, d, device);
  ttnn::Tensor dev_y = ttnn::layer_norm(dev_x, args.eps, dev_w, dev_b);
  CommitDevice2D(out, std::move(dev_y));
}

// kRmsNorm: per-row RMS over the last dim (cpu_ops.cpp RmsNormKernel). First
// Qwen3-dense (`Qwen3ForCausalLM`) op beyond OPT's LayerNorm set — Qwen3 uses
// RMSNorm for input/post-attn/final norms and per-head q/k norms. Weight is
// always present at the seam; optional residual is the residual stream
// (pre-norm sum written back, then normed), matching CPU residual round-trip
// for bf16 faithfulness. Gemma style (w+1) is host-only for now — Qwen3 does
// not set gemma=true.
//
// Device path: ttnn::rms_norm after residual merge (when any) and weight
// upload via the same TILE [1,D] affine helper as kLayerNorm.
void RmsNormKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& weight,
                   const RmsNormArgs& args, Tensor* residual) {
  TT_OP_TRACE("RmsNorm");
  VT_CHECK(x.rank == 2 && out.rank == 2,
           "tenstorrent kRmsNorm: only rank-2 tensors are supported in this step");
  VT_CHECK(IsFloatDType(x.dtype) && (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kRmsNorm: float in, f32/bf16 out");
  VT_CHECK(x.shape[0] == out.shape[0] && x.shape[1] == out.shape[1],
           "tenstorrent kRmsNorm: out shape mismatch");
  VT_CHECK(x.IsContiguous() && out.IsContiguous() && weight.IsContiguous(),
           "tenstorrent kRmsNorm: strided (non-contiguous) tensors are not supported");
  VT_CHECK(args.eps >= 0.0f, "tenstorrent kRmsNorm: eps must be non-negative");
  const uint32_t rows = static_cast<uint32_t>(x.shape[0]);
  const uint32_t d = static_cast<uint32_t>(x.shape[1]);
  VT_CHECK(weight.rank == 1 && weight.shape[0] == d && IsFloatDType(weight.dtype),
           "tenstorrent kRmsNorm: weight must be rank-1 float [D]");
  if (residual != nullptr) {
    VT_CHECK(residual->rank == 2 && residual->shape[0] == rows && residual->shape[1] == d,
             "tenstorrent kRmsNorm: residual shape must match x");
    VT_CHECK(IsFloatDType(residual->dtype) && residual->IsContiguous(),
             "tenstorrent kRmsNorm: residual must be contiguous float");
  }

  // Host path for gemma (w+1) and for tiny residual merges: short decode
  // (rows=1) pays more for device add+rms launches than a host loop, and was
  // a measurable e2e regression vs host residual.
  constexpr uint32_t kDeviceResidualMinRows = 32;
  // HOST-FREE-FORWARD R1: force the residual merge + RMS device path at T=1 when
  // capture is desired (ttnn trace prohibits host ops in the captured region).
  // Default ON since the R5 flip; VT_TT_HOST_FREE_DECODE=0 opts out (the
  // pre-flip default). Numerics proven by BACKEND-TENSTORRENT-RESIDUAL-GOLDEN.
  const bool host_free_decode = HostFreeDecodeEnabled();
  const bool host_residual = !host_free_decode &&
      (args.gemma || (residual != nullptr && rows < kDeviceResidualMinRows));
  if (host_residual) {
    EnsureHost(x);
    EnsureHost(weight);
    if (residual != nullptr) EnsureHost(*residual);
    for (int64_t r = 0; r < static_cast<int64_t>(rows); ++r) {
      float sumsq = 0.0f;
      for (int64_t j = 0; j < static_cast<int64_t>(d); ++j) {
        const int64_t idx = r * static_cast<int64_t>(d) + j;
        float v = LoadElemF32(x, idx);
        if (residual != nullptr) {
          v += LoadElemF32(*residual, idx);
          StoreElemF32(*residual, idx, v);
          v = LoadElemF32(*residual, idx);
        }
        sumsq += v * v;
      }
      const float inv =
          1.0f / std::sqrt(sumsq / static_cast<float>(d) + args.eps);
      for (int64_t j = 0; j < static_cast<int64_t>(d); ++j) {
        const int64_t idx = r * static_cast<int64_t>(d) + j;
        float v =
            residual != nullptr ? LoadElemF32(*residual, idx) : LoadElemF32(x, idx);
        float wj = LoadElemF32(weight, j);
        if (args.gemma) wj += 1.0f;
        StoreElemF32(out, idx, v * inv * wj);
      }
    }
    CommitHost(out);
    if (residual != nullptr) CommitHost(*residual);
    return;
  }

  // Device path: plain rms, or residual stream (x+residual → residual, then
  // rms) when rows are large enough that launches amortize.
  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_x = EnsureDevice2D(x, device);
  // Gemma (w+1) is baked host-side in the oracle's f32 order — the same
  // treatment as the fused preamble's weff — because ttnn::rms_norm applies
  // gamma raw. The baked upload is TRANSIENT and deliberately bypasses the
  // EnsureAffine1D slot cache: the cached form is the RAW weight, and a
  // non-gemma consumer of the same buffer must never read the +1 version.
  // Qwen3 never sets gemma; Qwen3.5 sets it on every norm (BACKEND-TENSTORRENT-
  // QWEN35 W2b: dropping the +1 here collapsed ambient prefill to `,`).
  ttnn::Tensor dev_w;
  if (args.gemma) {
    // Gemma (w+1) is baked host-side in the oracle's f32 order — the same
    // treatment as the fused preamble's weff — because ttnn::rms_norm applies
    // gamma raw. The baked form is cached in its own slot field, distinct
    // from EnsureAffine1D's raw BF16 form: a non-gemma consumer of the same
    // buffer must never read the +1 version. Qwen3 never sets gemma; Qwen3.5
    // sets it on every norm (BACKEND-TENSTORRENT-QWEN35 W2b: dropping the +1
    // here collapsed ambient prefill to `,`). The cache is what makes the
    // captured arm legal: the first (warmup) call stages the baked tensor,
    // and every later call — including every captured step — reuses it
    // instead of issuing the per-call host write that trace capture
    // forbids (#2812: the Qwen3.5-0.8B captured battery fatalled here,
    // fd_mesh_command_queue.cpp:760, 2/2 deterministic).
    bool have_w = false;
    {
      std::lock_guard<std::mutex> g(SlotMutex());
      BufferSlot* s = FindSlot(weight.data);
      if (s != nullptr && weight.data == s->host &&
          s->gemma_device.has_value()) {
        dev_w = *s->gemma_device;
        have_w = true;
      }
    }
    if (!have_w) {
      EnsureHost(weight);
      std::vector<float> gw(static_cast<size_t>(d));
      for (uint32_t i = 0; i < d; ++i)
        gw[static_cast<size_t>(i)] =
            LoadElemF32(weight, static_cast<int64_t>(i)) + 1.0f;
      if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
        std::fprintf(stderr,
                     "[TT-UP] RmsNorm gemma from_vector WRITE during capture\n");
      dev_w = ttnn::Tensor::from_vector<float>(
          std::move(gw),
          SpecOf(tt::tt_metal::Shape({1, d}), ttnn::DataType::FLOAT32,
                 ttnn::Layout::TILE),
          &device);
      std::lock_guard<std::mutex> g(SlotMutex());
      BufferSlot* s = FindSlot(weight.data);
      if (s != nullptr && weight.data == s->host)
        s->gemma_device = dev_w;
    }
  } else {
    dev_w = EnsureAffine1D(weight, d, device);
  }
  ttnn::Tensor to_norm = dev_x;
  if (residual != nullptr) {
    ttnn::Tensor dev_r = EnsureDevice2D(*residual, device);
    to_norm = ttnn::add(dev_x, dev_r);
    CommitDevice2D(*residual, to_norm);
  }
  ttnn::Tensor dev_y = ttnn::rms_norm(to_norm, args.eps, dev_w);
  CommitDevice2D(out, std::move(dev_y));
}

// kFusedChain: dispatch kFusedAddRmsNormStd to the same device RmsNorm path
// (residual += x; out = rms_norm(residual, weight)). Other recipes fall
// through to the CPU interpreter (host round-trip). Without this registration,
// FusedChain falls back to the CPU kernel which reads HOST memory — fatal
// when the PA output is device-resident (VT_TT_HOST_FREE_DECODE).
void FusedChainKernel(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                      Tensor* residual, const FusedRecipe& r, float eps) {
  TT_OP_TRACE("FusedChain");
  // kFusedAddRmsNormStd: step0 kAdd(residual = x + residual),
  // step1 kRmsNorm(out = rms_norm(residual, weight)).
  // This is exactly RmsNormKernel with the residual parameter.
  if (r.n == 2 &&
      r.steps[0].op == FOp::kAdd && r.steps[0].out == 2 &&
      r.steps[1].op == FOp::kRmsNorm && r.steps[1].out == 3 &&
      r.steps[1].gemma == false) {
    RmsNormKernel(q, out, x, weight, RmsNormArgs{eps, false}, residual);
    return;
  }
  // kFusedAddRmsNorm (gemma variant): same but gemma=true.
  if (r.n == 2 &&
      r.steps[0].op == FOp::kAdd && r.steps[0].out == 2 &&
      r.steps[1].op == FOp::kRmsNorm && r.steps[1].out == 3 &&
      r.steps[1].gemma == true) {
    RmsNormKernel(q, out, x, weight, RmsNormArgs{eps, true}, residual);
    return;
  }
  // Unknown recipe: fall back to host (safe outside capture).
  VT_CHECK(!tt_capture_active(),
           "tenstorrent: unknown FusedChain recipe during capture");
  EnsureHost(out);
  EnsureHost(x);
  EnsureHost(weight);
  if (residual != nullptr) EnsureHost(*residual);
  // Delegate to the CPU interpreter by calling the registered CPU op.
  auto cpu_fn = reinterpret_cast<FusedChainFn>(
      GetOpFallback(OpId::kFusedChain, DeviceType::kTENSTORRENT, "vt-tenstorrent"));
  if (cpu_fn) {
    cpu_fn(q, out, x, weight, residual, r, eps);
  } else {
    VT_CHECK(false, "tenstorrent: no FusedChain fallback available");
  }
}

// kSiluAndMul: SwiGLU gate half — out[i,j] = silu(x[i,j]) * x[i,j+d]
// with d = x.shape[1]/2 (cpu_ops.cpp SiluAndMulKernel). Second Qwen3-dense
// op beyond OPT (MLP: gate_up GEMM -> SiluAndMul -> down GEMM). Device path
// keeps the gate_up → SiluAndMul → down GEMM chain on-device: slice the
// last-dim halves, ttnn::silu(gate), ttnn::multiply by up. BF16 tile path
// (same envelope as matmul/norm); not bit-exact vs host f32.
void SiluAndMulKernel(Queue&, Tensor& out, const Tensor& x) {
  TT_OP_TRACE("SiluAndMul");
  VT_CHECK(x.rank == 2 && out.rank == 2,
           "tenstorrent kSiluAndMul: only rank-2 tensors are supported");
  VT_CHECK(IsFloatDType(x.dtype) && (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kSiluAndMul: float in, f32/bf16 out");
  VT_CHECK(x.IsContiguous() && out.IsContiguous(),
           "tenstorrent kSiluAndMul: strided (non-contiguous) tensors are not supported");
  VT_CHECK(x.shape[1] % 2 == 0, "tenstorrent kSiluAndMul: last dim must be even");
  const int64_t t = x.shape[0];
  const int64_t d = x.shape[1] / 2;
  VT_CHECK(out.shape[0] == t && out.shape[1] == d,
           "tenstorrent kSiluAndMul: out shape must be [T, D] with D = x.shape[1]/2");

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_x = EnsureDevice2D(x, device);
  const uint32_t tu = static_cast<uint32_t>(t);
  const uint32_t du = static_cast<uint32_t>(d);
  // x = [gate | up] along last dim.
  ttnn::Tensor gate = ttnn::slice(dev_x, ttsl::SmallVector<uint32_t>{0, 0},
                                  ttsl::SmallVector<uint32_t>{tu, du},
                                  ttsl::SmallVector<uint32_t>{1, 1});
  ttnn::Tensor up = ttnn::slice(dev_x, ttsl::SmallVector<uint32_t>{0, du},
                                ttsl::SmallVector<uint32_t>{tu, 2 * du},
                                ttsl::SmallVector<uint32_t>{1, 1});
  ttnn::Tensor silu_gate = ttnn::silu(gate);
  ttnn::Tensor dev_y = ttnn::multiply(silu_gate, up);
  CommitDevice2D(out, std::move(dev_y));
}

// kMoeSiluMul: silu(gate) * up with SPLIT operands (ops.cpp MoeSiluMul -> id 63;
// cpu_ops.cpp MoeSiluMulKernel, cuda_moe.cu MoeSiluMulKernel). The split sibling
// of kSiluAndMul above: the GGUF dense MLP arm reaches the DenseMlpBlock tail
// with gate/up as TWO separate [T,I] GEMM outputs (GGUF stores ff_gate/ff_up
// unmerged), so there is no merged [T,2I] operand to slice. Same ttnn math
// (ttnn::silu + ttnn::multiply) and the same capture-safe EnsureDevice2D /
// CommitDevice2D staging as kSiluAndMul, so the captured e2e treats it
// identically. The vehicle path feeds f32 gate/up, where the CPU kernel's
// RoundThrough(gate.dtype) narrowing is the identity.
void MoeSiluMulKernel(Queue&, Tensor& out, const Tensor& gate, const Tensor& up) {
  TT_OP_TRACE("MoeSiluMul");
  VT_CHECK(gate.rank == 2 && up.rank == 2 && out.rank == 2,
           "tenstorrent kMoeSiluMul: only rank-2 tensors are supported");
  VT_CHECK(IsFloatDType(gate.dtype) && IsFloatDType(up.dtype) &&
               (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kMoeSiluMul: float in, f32/bf16 out");
  VT_CHECK(gate.IsContiguous() && up.IsContiguous() && out.IsContiguous(),
           "tenstorrent kMoeSiluMul: strided (non-contiguous) tensors are not supported");
  VT_CHECK(gate.shape[0] == out.shape[0] && gate.shape[1] == out.shape[1] &&
               up.shape[0] == gate.shape[0] && up.shape[1] == gate.shape[1],
           "tenstorrent kMoeSiluMul: gate, up and out must share one [T, I] shape");

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_g = EnsureDevice2D(gate, device);
  ttnn::Tensor dev_u = EnsureDevice2D(up, device);
  ttnn::Tensor dev_y = ttnn::multiply(ttnn::silu(dev_g), dev_u);
  CommitDevice2D(out, std::move(dev_y));
}

// kCastBf16 / kCastF32: elementwise dtype convert via Load/Store (cpu_ops
// CastBf16Kernel / CastF32Kernel). Qwen3 uses these for K/V cache dtype and
// the logits / rope-cache paths. Host-staged; bit-exact for supported pairs.
// The device-pure staging below rides helpers defined later in this TU.
bool ServeDeviceShadowRaw(const Tensor& t, uint32_t rows, uint32_t cols,
                          ttnn::Tensor& out);
ttnn::Tensor NormalizeDevF32Tile(ttnn::Tensor x, uint32_t rows, uint32_t cols);

void CastBf16Kernel(Queue&, Tensor& out, const Tensor& in) {
  TT_OP_TRACE("CastBf16");
  VT_CHECK(out.dtype == DType::kBF16, "tenstorrent kCastBf16: out must be bf16");
  VT_CHECK(IsFloatDType(in.dtype), "tenstorrent kCastBf16: in must be float");
  VT_CHECK(out.Numel() == in.Numel(), "tenstorrent kCastBf16: numel mismatch");
  VT_CHECK(out.IsContiguous() && in.IsContiguous(),
           "tenstorrent kCastBf16: contiguous required");
  // Captured arm: the producer (fused preamble / GEMM) left `in`
  // device-authoritative, so the cast must run on the device shadow — a host
  // EnsureHost here would enqueue exactly the readback the trace forbids.
  // The device typecast is the same single RNE round the host StoreElemF32
  // makes (the SigmoidGateBf16 doctrine), so both arms stay byte-identical.
  const uint32_t n = static_cast<uint32_t>(in.Numel());
  ttnn::Tensor dev_in;
  if (ServeDeviceShadowRaw(in, 1, n, dev_in)) {
    ttnn::Tensor dev = NormalizeDevF32Tile(std::move(dev_in), 1, n);
    // Commit the SERVED geometry (#2282): a rank-2 shadow keeps its native
    // logical shape, and the record must name the geometry actually stored.
    const auto dls = dev.logical_shape();
    CommitDeviceLogical2D(out, ttnn::typecast(dev, ttnn::DataType::BFLOAT16),
                          dls[dls.rank() - 2], dls[dls.rank() - 1]);
    return;
  }
  {
    // The host fallback below is legal mid-capture ONLY while the bytes are
    // already host-current (EnsureHost would no-op). An untracked tensor is
    // host-only by construction. Anything else would need exactly the
    // readback a trace capture forbids, so refuse it loudly instead.
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* cs = FindSlot(in.data);
    VT_CHECK(!tt_capture_active() || cs == nullptr || cs->host_current,
             "tenstorrent kCastBf16: input arrived without a servable device "
             "shadow during trace capture and is not host-current — the "
             "readback the host path would need is what capture forbids");
  }
  EnsureHost(in);
  const int64_t ne = out.Numel();
  for (int64_t i = 0; i < ne; ++i) StoreElemF32(out, i, LoadElemF32(in, i));
  CommitHost(out);
}

void CastF32Kernel(Queue&, Tensor& out, const Tensor& in) {
  TT_OP_TRACE("CastF32");
  VT_CHECK(out.dtype == DType::kF32, "tenstorrent kCastF32: out must be f32");
  VT_CHECK(IsFloatDType(in.dtype), "tenstorrent kCastF32: in must be float");
  VT_CHECK(out.Numel() == in.Numel(), "tenstorrent kCastF32: numel mismatch");
  VT_CHECK(out.IsContiguous() && in.IsContiguous(),
           "tenstorrent kCastF32: contiguous required");
  // Captured arm, identical to kCastBf16 above: the producer (the lm_head
  // GEMM) left `in` device-authoritative, so the cast must run on the device
  // shadow — a host EnsureHost here would enqueue exactly the readback the
  // trace forbids. NormalizeDevF32Tile yields f32 TILE, which is the target
  // dtype, so no typecast remains; the ROW_MAJOR round-trip is a pure copy.
  const uint32_t nf = static_cast<uint32_t>(in.Numel());
  ttnn::Tensor dev_in;
  if (ServeDeviceShadowRaw(in, 1, nf, dev_in)) {
    ttnn::Tensor dev = NormalizeDevF32Tile(std::move(dev_in), 1, nf);
    // SERVED geometry, as in kCastBf16 above.
    const auto dls = dev.logical_shape();
    CommitDeviceLogical2D(out, std::move(dev), dls[dls.rank() - 2],
                          dls[dls.rank() - 1]);
    return;
  }
  {
    // Same host-current discipline as kCastBf16 above.
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* cs = FindSlot(in.data);
    VT_CHECK(!tt_capture_active() || cs == nullptr || cs->host_current,
             "tenstorrent kCastF32: input arrived without a servable device "
             "shadow during trace capture and is not host-current — the "
             "readback the host path would need is what capture forbids");
  }
  EnsureHost(in);
  const int64_t n = out.Numel();
  for (int64_t i = 0; i < n; ++i) StoreElemF32(out, i, LoadElemF32(in, i));
  CommitHost(out);
}

// kSigmoidGateBf16: out[i] = F32ToBF16(attn[i] * sigmoid(gate[i])) — the
// full-attention o_proj gate (cpu_ops.cpp SigmoidGateBf16Kernel; wrapper
// ops.cpp:4136-4153). The gate must NOT be rounded before the sigmoid
// (ops.cpp:4140: "sigmoid input must not be rounded"), so both operands ride
// FLOAT32 tile shadows and the product runs in f32; the single RNE round to
// bf16 happens in ttnn::typecast (the device fp32->fp16b cast: +0x7FFF+lsb
// then mask, ckernel_sfpu_typecast.h:246-262 — bit-identical to F32ToBF16;
// ttnn::to_dtype is HOST-ONLY at this pin, tensor_ops.cpp:533). The
// only TT-vs-CPU delta is the SFPU f32 sigmoid (accurate exp +
// reciprocal_iter<2> vs std::exp) — a few f32 ULP that can flip at most one
// bf16 rounding; the doctest envelope is one bf16 ULP.
// (UploadTensor / DeviceRows / ServePostConvAB / CachedTile / ServeActF32
// forward declarations removed by the stage-3 split: their definitions
// gained external linkage and tenstorrent_internal.h declarations, and a
// declaration left inside this anonymous namespace would name a distinct,
// never-defined internal entity that shadows them.)
ttnn::Tensor CachedRepeatIdx(uint64_t t, uint64_t heads, uint64_t half,
                             MeshDevice& device);  // defined below (rope index)
bool ServeDeviceWindow(const Tensor& t, uint32_t rows, uint32_t cols,
                       ttnn::Tensor& out);  // defined below (window serve)
bool ServeDeviceShadowRaw(const Tensor& t, uint32_t rows, uint32_t cols,
                          ttnn::Tensor& out);  // defined below (exact serve)
ttnn::Tensor NormalizeDevF32Tile(ttnn::Tensor x, uint32_t rows,
                                 uint32_t cols);  // defined below (serve norm)
void SigmoidGateBf16Kernel(Queue&, Tensor& out, const Tensor& attn,
                           const Tensor& gate) {
  TT_OP_TRACE("SigmoidGateBf16");
  VT_CHECK(out.dtype == DType::kBF16,
           "tenstorrent kSigmoidGateBf16: out must be bf16");
  VT_CHECK((attn.dtype == DType::kF32 || attn.dtype == DType::kBF16) &&
               gate.dtype == DType::kF32,
           "tenstorrent kSigmoidGateBf16: attn must be f32/bf16, gate f32");
  VT_CHECK(out.Numel() == attn.Numel() && out.Numel() == gate.Numel(),
           "tenstorrent kSigmoidGateBf16: out/attn/gate same element count");
  VT_CHECK(out.IsContiguous() && attn.IsContiguous() && gate.IsContiguous(),
           "tenstorrent kSigmoidGateBf16: contiguous required");
  const uint32_t n = static_cast<uint32_t>(out.Numel());
  MeshDevice& device = SharedMeshDevice();
  // UploadTensor FLOAT32 (not the bf16 EnsureDevice2D shadow): bf16 attn
  // upcasts exactly, and f32 attn keeps its full mantissa — a bf16 shadow
  // would pre-round the attn operand and widen the envelope (the f32 arm of
  // the doctest exists to catch exactly that).
  // Capture staging: in the captured arm attn arrives device-resident (the
  // PA sdpa commit) and the gate rides the same device-authoritative commit
  // path, so ServeActF32 serves those f32 TILE shadows and refuses under
  // capture when neither a shadow nor host bytes are readable — a ToHostF32
  // here would enqueue exactly the readback the trace forbids. Eager keeps
  // the host staging (ServeActF32's fallback is byte-identical to it).
  ttnn::Tensor dev_attn = ServeActF32(attn, 1, n, "kSigmoidGateBf16 attn", device);
  ttnn::Tensor dev_gate = ServeActF32(gate, 1, n, "kSigmoidGateBf16 gate", device);
  ttnn::Tensor sig = ttnn::sigmoid(dev_gate);
  ttnn::Tensor prod = ttnn::multiply(dev_attn, sig);
  ttnn::Tensor dev_y = ttnn::typecast(prod, ttnn::DataType::BFLOAT16);
  CommitDeviceLogical2D(out, std::move(dev_y), 1, n);
}

// Content-checked serve of the fused preamble's cos|sin table. The host bytes
// of `cos_sin` are fresh (the in-region RopeCosSinCacheKernel refill ran
// before this op), so equality against the warmed table proves the persistent
// device tensor holds THIS step's content. Returns false on miss.
static bool LookupAttnCSTable(const Tensor& cos_sin, uint32_t t, uint32_t rot,
                              ttnn::Tensor& out) {
  EnsureHost(cos_sin);
  const int64_t n = static_cast<int64_t>(t) * rot;
  std::vector<float> host(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    host[static_cast<size_t>(i)] = LoadElemF32(cos_sin, i);
  std::lock_guard<std::mutex> g(AttnCSMutex());
  auto it = AttnCSCache().find(AttnCSKey(t, rot));
  if (tt_capture_active() && std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr,
                 "[TT-TRACE] attn-cs lookup key=%ux%u found=%d eq=%d "
                 "want_first=%f have_first=%f\n",
                 t, rot, it != AttnCSCache().end() ? 1 : 0,
                 it != AttnCSCache().end() && it->second.cs_host == host ? 1 : 0,
                 host.empty() ? -1.0f : host.front(),
                 (it == AttnCSCache().end() || it->second.cs_host.empty())
                     ? -1.0f
                     : it->second.cs_host.front());
  if (it == AttnCSCache().end() || it->second.cs_host != host) return false;
  out = it->second.cs;
  return true;
}

// kAttnQkNormRopeGate: fused full-attention preamble = split q|gate +
// per-head gemma qk-RMSNorm + partial NeoX RoPE-from-cos_sin + exact gate
// passthrough, in ONE launch (cpu_ops.cpp AttnQkNormRopeGateKernel:1216-1270;
// wrapper ops.cpp:1638-1689; production call qwen3_5.cpp:5206-5224, gemma=true,
// rot < Dh). qgate/kf arrive as merged-QKV strided views (rows
// inner-contiguous); the split into per-head [T*H, Dh] rows happens in the
// HOST gather, and each leg is uploaded already in its final logical shape
// (no device slice/reshape: measured on the P150, first green attempt, a
// column slice + ttnn::reshape chain returned wrong data for the qgate legs
// while the full-width kf leg stayed correct — only a fresh contiguous
// upload is a safe reshape input at this pin). Norm+rope ride FLOAT32 tiles
// end to end (the SigmoidGateBf16 doctrine): rsqrt, the weight mix and the
// cos/sin rotation never round before the single output typecast, so an f32
// out sits within reduction-order ULPs of the scalar-f32 oracle and a bf16
// out is its exact RNE round. The gate leg is a plain device copy — no
// arithmetic touches it, so the f32 passthrough is bit-exact (the sigmoid
// input must not be rounded, ops.cpp:1660-1662/4140). The gemma weight w+1
// is computed host-side in f32, the same add GemmaNormElem does
// (cpu_ops.cpp:1204-1208).
void AttnQkNormRopeGateKernel(Queue&, Tensor& q_out, Tensor& k_out,
                              Tensor& gate_out, const Tensor& qgate,
                              const Tensor& kf, const Tensor& q_norm,
                              const Tensor& k_norm, const Tensor& cos_sin,
                              const RmsNormArgs& na, const RopeArgs& ra) {
  TT_OP_TRACE("AttnQkNormRopeGate");
  VT_CHECK((q_out.dtype == DType::kF32 || q_out.dtype == DType::kBF16) &&
               k_out.dtype == q_out.dtype &&
               (gate_out.dtype == q_out.dtype ||
                (q_out.dtype == DType::kBF16 && gate_out.dtype == DType::kF32)),
           "tenstorrent kAttnQkNormRopeGate: q/k/gate out f32 or bf16 "
           "(gate f32 allowed with bf16 q/k)");
  VT_CHECK(IsFloatDType(qgate.dtype) && kf.dtype == qgate.dtype,
           "tenstorrent kAttnQkNormRopeGate: qgate/kf float, same dtype");
  VT_CHECK(q_out.IsContiguous() && k_out.IsContiguous() &&
               gate_out.IsContiguous() && qgate.stride[1] == 1 &&
               qgate.stride[0] >= qgate.shape[1] && kf.stride[1] == 1 &&
               kf.stride[0] >= kf.shape[1] && q_norm.IsContiguous() &&
               k_norm.IsContiguous() && cos_sin.IsContiguous(),
           "tenstorrent kAttnQkNormRopeGate: contiguous required "
           "(qgate/kf row views excepted)");
  const int64_t t = q_out.shape[0], hq = q_out.shape[1], dh = q_out.shape[2];
  const int64_t hkv = k_out.shape[1];
  const int64_t rot = ra.rotary_dim, half = rot / 2;
  const int64_t qrow = qgate.shape[1], krow = kf.shape[1];
  VT_CHECK(qrow == hq * 2 * dh && krow == hkv * dh,
           "tenstorrent kAttnQkNormRopeGate: qgate [T, Hq*2*Dh], kf [T, Hkv*Dh]");

  MeshDevice& device = SharedMeshDevice();
  // Host-gather the strided merged-QKV rows straight into per-head [T*H, Dh]
  // legs (the q|gate split is the host half of this fused op). The FLOAT32
  // upload also upcasts a bf16 input exactly (LoadElemF32), so every later
  // leg is dtype-uniform.
  // ---- device-pure staging ------------------------------------------------
  // qgate is the merged q|gate view: one [q(dh)|gate(dh)] block per head.
  // Serving the legs is pure data movement on the committed shadow — a
  // ROW_MAJOR materialize (the safe reshape input at this pin), the
  // [t*hq, 2*dh] head-row metadata view, then the two TILE column slices
  // (the dev_conv slice precedent). kf normalizes straight to its rows.
  // Every fallback below stays in the eager arm: an EnsureHost on a
  // committed activation is exactly the readback the trace refuses, and
  // the gathered re-uploads are capture-time writes.
  ttnn::Tensor dev_q, dev_gate, dev_k;
  {
    const uint32_t tw = static_cast<uint32_t>(t);
    const uint32_t qw = static_cast<uint32_t>(qgate.shape[1]);
    const uint32_t kw = static_cast<uint32_t>(kf.shape[1]);
    ttnn::Tensor qg_shadow, kf_shadow;
    const bool qg_ok = ServeDeviceShadowRaw(qgate, tw, qw, qg_shadow) ||
                       ServeDeviceWindow(qgate, tw, qw, qg_shadow);
    const bool kf_ok = ServeDeviceShadowRaw(kf, tw, kw, kf_shadow) ||
                       ServeDeviceWindow(kf, tw, kw, kf_shadow);
    if (qg_ok && kf_ok) {
      auto rm_f32 = [](ttnn::Tensor x, uint32_t r, uint32_t c) {
        x = ttnn::to_layout(x, ttnn::Layout::ROW_MAJOR);
        const auto ls = x.logical_shape();
        if (ls.rank() != 2 || ls[0] != r || ls[1] != c)
          x = ttnn::reshape(x, ttnn::Shape({r, c}));
        if (x.dtype() != ttnn::DataType::FLOAT32)
          x = ttnn::typecast(x, ttnn::DataType::FLOAT32);
        return x;
      };
      const uint32_t rows_q = static_cast<uint32_t>(t * hq);
      ttnn::Tensor qg = rm_f32(std::move(qg_shadow), tw, qw);
      qg = ttnn::reshape(qg, ttnn::Shape({rows_q, static_cast<uint32_t>(2 * dh)}));
      qg = ttnn::to_layout(qg, ttnn::Layout::TILE);
      dev_q = ttnn::slice(qg, ttsl::SmallVector<uint32_t>{0, 0},
                          ttsl::SmallVector<uint32_t>{rows_q,
                                                      static_cast<uint32_t>(dh)},
                          ttsl::SmallVector<uint32_t>{1, 1});
      dev_gate = ttnn::slice(qg, ttsl::SmallVector<uint32_t>{0,
                                                             static_cast<uint32_t>(dh)},
                             ttsl::SmallVector<uint32_t>{rows_q,
                                                         static_cast<uint32_t>(2 * dh)},
                             ttsl::SmallVector<uint32_t>{1, 1});
      dev_k = NormalizeDevF32Tile(std::move(kf_shadow),
                                  static_cast<uint32_t>(t * hkv),
                                  static_cast<uint32_t>(dh));
      // NormalizeDevF32Tile preserves the shadow's native [t, hkv*dh] geometry
      // (PR #3206 L1 fix). Reshape to the intended [t*hkv, dh] — safe on TILE
      // (view_device guard at tensor_ops.cpp:398: layout != ROW_MAJOR
      // short-circuits before the page_size rebuild that corrupts ROW_MAJOR).
      {
        const auto k_target = ttnn::Shape({static_cast<uint32_t>(t * hkv),
                                            static_cast<uint32_t>(dh)});
        if (dev_k.logical_shape() != k_target)
          dev_k = CaptureSafeReshape(dev_k, k_target);
      }
    } else {
      VT_CHECK(!tt_capture_active(),
               "tenstorrent kAttnQkNormRopeGate: qgate/kf arrived without a "
               "servable device shadow during trace capture — the producer "
               "must commit device-side before the captured region");
      EnsureHost(qgate);
      EnsureHost(kf);
      std::vector<float> qh(static_cast<size_t>(t * hq * dh)),
          gh(qh.size()), kh(static_cast<size_t>(t * hkv * dh));
      for (int64_t i = 0; i < t; ++i) {
        for (int64_t h = 0; h < hq; ++h) {
          const int64_t base = i * qgate.stride[0] + h * 2 * dh;
          const size_t dst = static_cast<size_t>((i * hq + h) * dh);
          for (int64_t j = 0; j < dh; ++j) {
            qh[dst + j] = LoadElemF32(qgate, base + j);
            gh[dst + j] = LoadElemF32(qgate, base + dh + j);
          }
        }
        for (int64_t h = 0; h < hkv; ++h) {
          const int64_t base = i * kf.stride[0] + h * dh;
          const size_t dst = static_cast<size_t>((i * hkv + h) * dh);
          for (int64_t j = 0; j < dh; ++j)
            kh[dst + j] = LoadElemF32(kf, base + j);
        }
      }
      dev_q = UploadTensor(std::move(qh),
                           ttnn::Shape({static_cast<uint32_t>(t * hq),
                                        static_cast<uint32_t>(dh)}),
                           ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
      dev_gate = UploadTensor(std::move(gh),
                              ttnn::Shape({static_cast<uint32_t>(t * hq),
                                           static_cast<uint32_t>(dh)}),
                              ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
      dev_k = UploadTensor(std::move(kh),
                           ttnn::Shape({static_cast<uint32_t>(t * hkv),
                                        static_cast<uint32_t>(dh)}),
                           ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
    }
  }
  // Gemma effective weight (host f32 add, oracle-identical): immutable
  // model constants, so the tiles ride the content-checked cache (the warm
  // step builds them; an in-region rebuild is refused by name).
  EnsureHost(q_norm);
  EnsureHost(k_norm);
  auto weff = [&](const Tensor& w) {
    std::vector<float> v(static_cast<size_t>(dh));
    for (int64_t j = 0; j < dh; ++j) {
      const float wj = LoadElemF32(w, j);
      v[static_cast<size_t>(j)] = na.gemma ? wj + 1.0f : wj;
    }
    return v;
  };
  const uint32_t du = static_cast<uint32_t>(dh);
  ttnn::Tensor dev_wq = CachedTile(q_norm.data, du, na.gemma ? 1 : 0, 0,
                                   [&] { return weff(q_norm); },
                                   ttnn::Shape({1, du}), device);
  ttnn::Tensor dev_wk = CachedTile(k_norm.data, du, na.gemma ? 1 : 0, 0,
                                   [&] { return weff(k_norm); },
                                   ttnn::Shape({1, du}), device);
  // cos_sin: the driver stages the step's table device-side in the captured
  // arm; the eager arm falls back to the host bytes. Both arms then share
  // the same device expansion: the [cos|sin] halves are TILE column slices,
  // and the per-(token,head) repeat is a dim-0 gather under the cached
  // full-width u32 index (the GdnDecode head-map precedent) — pure data
  // movement, bit-exact against the host expansion it replaces.
  ttnn::Tensor dev_cs;
  {
    ttnn::Tensor cs_shadow;
    if (ServeDeviceShadowRaw(cos_sin, static_cast<uint32_t>(t),
                             static_cast<uint32_t>(rot), cs_shadow) ||
        ServeDeviceWindow(cos_sin, static_cast<uint32_t>(t),
                          static_cast<uint32_t>(rot), cs_shadow)) {
      dev_cs = NormalizeDevF32Tile(std::move(cs_shadow),
                                   static_cast<uint32_t>(t),
                                   static_cast<uint32_t>(rot));
    } else {
      // Capture: serve the persistent per-step table the driver's
      // WarmAttnCosSin refreshes OUTSIDE capture (WarmRopeCosSin pattern).
      ttnn::Tensor cs_cached;
      if (LookupAttnCSTable(cos_sin, static_cast<uint32_t>(t),
                            static_cast<uint32_t>(rot), cs_cached)) {
        dev_cs = std::move(cs_cached);  // already f32 TILE [t, rot]
      } else {
        VT_CHECK(!tt_capture_active(),
                 "tenstorrent kAttnQkNormRopeGate: cos_sin arrived without a "
                 "servable device shadow during trace capture — the "
                 "decode-graph driver must call WarmAttnCosSin for the step's "
                 "positions BEFORE BeginCapture (the WarmRopeCosSin pattern)");
        EnsureHost(cos_sin);
        std::vector<float> cs(static_cast<size_t>(t) * rot);
        for (int64_t i = 0; i < t; ++i)
          for (int64_t j = 0; j < rot; ++j)
            cs[static_cast<size_t>(i) * rot + j] = LoadElemF32(cos_sin, i * rot + j);
        dev_cs = UploadTensor(std::move(cs),
                              ttnn::Shape({static_cast<uint32_t>(t),
                                           static_cast<uint32_t>(rot)}),
                              ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
      }
    }
  }
  const uint32_t halfu = static_cast<uint32_t>(half);
  const uint32_t rotu = static_cast<uint32_t>(rot);
  const uint32_t tu = static_cast<uint32_t>(t);
  ttnn::Tensor dev_cosh = ttnn::slice(
      dev_cs, ttsl::SmallVector<uint32_t>{0, 0},
      ttsl::SmallVector<uint32_t>{tu, halfu}, ttsl::SmallVector<uint32_t>{1, 1});
  ttnn::Tensor dev_sinh = ttnn::slice(
      dev_cs, ttsl::SmallVector<uint32_t>{0, halfu},
      ttsl::SmallVector<uint32_t>{tu, rotu}, ttsl::SmallVector<uint32_t>{1, 1});
  // normed = x * rsqrt(mean(x^2)+eps) * (w+gemma); NeoX half-split rotation of
  // the leading rot cols; tail [rot, Dh) passes through NORMED (not rotated).
  auto norm_rope = [&](ttnn::Tensor x, const ttnn::Tensor& dev_w,
                       const ttnn::Tensor& dev_cos,
                       const ttnn::Tensor& dev_sin, int64_t nrows) {
    const uint32_t nr = static_cast<uint32_t>(nrows);
    ttnn::Tensor sq = ttnn::multiply(x, x);
    ttnn::Tensor s = ttnn::sum(sq, ttsl::SmallVector<int>{1}, true);
    ttnn::Tensor denom =
        ttnn::add(ttnn::multiply(s, 1.0f / static_cast<float>(dh)), na.eps);
    ttnn::Tensor inv = ttnn::rsqrt(denom);
    ttnn::Tensor normed = ttnn::multiply(ttnn::multiply(x, inv), dev_w);
    ttnn::Tensor x1 = ttnn::slice(normed, ttsl::SmallVector<uint32_t>{0, 0},
                                  ttsl::SmallVector<uint32_t>{nr, halfu},
                                  ttsl::SmallVector<uint32_t>{1, 1});
    ttnn::Tensor x2 = ttnn::slice(normed,
                                   ttsl::SmallVector<uint32_t>{0, halfu},
                                   ttsl::SmallVector<uint32_t>{nr, rotu},
                                   ttsl::SmallVector<uint32_t>{1, 1});
    ttnn::Tensor o1 = ttnn::subtract(ttnn::multiply(x1, dev_cos),
                                     ttnn::multiply(x2, dev_sin));
    ttnn::Tensor o2 = ttnn::add(ttnn::multiply(x1, dev_sin),
                                ttnn::multiply(x2, dev_cos));
    ttnn::Tensor out =
        ttnn::concat(std::vector<ttnn::Tensor>{o1, o2}, /*dim=*/1);
    if (rot < dh) {
      ttnn::Tensor tail =
          ttnn::slice(normed, ttsl::SmallVector<uint32_t>{0, rotu},
                      ttsl::SmallVector<uint32_t>{nr, du},
                      ttsl::SmallVector<uint32_t>{1, 1});
      out = ttnn::concat(std::vector<ttnn::Tensor>{out, tail}, /*dim=*/1);
    }
    return out;
  };
  // Gate leg FIRST (plain device copy of its own upload — no math, so the
  // passthrough stays bit-exact), then the q/k norm+rope legs.
  if (gate_out.dtype == DType::kBF16)
    dev_gate = ttnn::typecast(dev_gate, ttnn::DataType::BFLOAT16);
  CommitDeviceLogical2D(gate_out, std::move(dev_gate),
                        static_cast<uint32_t>(t * hq),
                        static_cast<uint32_t>(dh));
  ttnn::Tensor cos_q = ttnn::gather(dev_cosh, /*dim=*/0,
                                    CachedRepeatIdx(t, hq, half, device),
                                    /*sparse_grad=*/false, std::nullopt);
  ttnn::Tensor sin_q = ttnn::gather(dev_sinh, /*dim=*/0,
                                    CachedRepeatIdx(t, hq, half, device),
                                    /*sparse_grad=*/false, std::nullopt);
  ttnn::Tensor q_dev = norm_rope(std::move(dev_q), dev_wq, cos_q, sin_q,
                                 t * hq);
  if (q_out.dtype == DType::kBF16)
    q_dev = ttnn::typecast(q_dev, ttnn::DataType::BFLOAT16);
  CommitDeviceLogical2D(q_out, std::move(q_dev), static_cast<uint32_t>(t * hq),
                        static_cast<uint32_t>(dh));
  ttnn::Tensor cos_k = ttnn::gather(dev_cosh, /*dim=*/0,
                                    CachedRepeatIdx(t, hkv, half, device),
                                    /*sparse_grad=*/false, std::nullopt);
  ttnn::Tensor sin_k = ttnn::gather(dev_sinh, /*dim=*/0,
                                    CachedRepeatIdx(t, hkv, half, device),
                                    /*sparse_grad=*/false, std::nullopt);
  ttnn::Tensor k_dev =
      norm_rope(std::move(dev_k), dev_wk, cos_k, sin_k, t * hkv);
  if (k_out.dtype == DType::kBF16)
    k_dev = ttnn::typecast(k_dev, ttnn::DataType::BFLOAT16);
  CommitDeviceLogical2D(k_out, std::move(k_dev),
                        static_cast<uint32_t>(t * hkv),
                        static_cast<uint32_t>(dh));
}

// Llama-3 frequency rescale (cpu_ops Llama3ScaleFreq); no-op when scaling_factor
// is unset. Kept so Qwen3 / Llama rope paths share one host implementation.
inline double Llama3ScaleFreq(double freq, const RopeArgs& a) {
  const double sf = static_cast<double>(a.llama3_scaling_factor);
  if (!(sf > 0.0)) return freq;
  constexpr double kTwoPi = 6.283185307179586476925286766559;
  const double lo = static_cast<double>(a.llama3_low_freq_factor);
  const double hi = static_cast<double>(a.llama3_high_freq_factor);
  const double omax = static_cast<double>(a.llama3_orig_max_position);
  const double low_freq_wavelen = omax / lo;
  const double high_freq_wavelen = omax / hi;
  const double wave_len = kTwoPi / freq;
  double smooth = 0.0;
  if (lo != hi) smooth = (omax / wave_len - lo) / (hi - lo);
  if (wave_len < high_freq_wavelen) return freq;
  if (wave_len > low_freq_wavelen) return freq / sf;
  return (1.0 - smooth) * freq / sf + smooth * freq;
}

// Expand per-token cos|sin [T, half] to per-(token,head) [T*H, half] for a
// device NeoX apply over the flat [T*H, D] view of qs/ks.
void ExpandCosSinPerHead(const float* cos_t, const float* sin_t, int64_t tokens,
                         int64_t heads, int64_t half, std::vector<float>& cos_exp,
                         std::vector<float>& sin_exp) {
  cos_exp.resize(static_cast<size_t>(tokens * heads * half));
  sin_exp.resize(static_cast<size_t>(tokens * heads * half));
  for (int64_t t = 0; t < tokens; ++t) {
    for (int64_t h = 0; h < heads; ++h) {
      const size_t dst = static_cast<size_t>((t * heads + h) * half);
      const size_t src = static_cast<size_t>(t * half);
      std::memcpy(cos_exp.data() + dst, cos_t + src, static_cast<size_t>(half) * sizeof(float));
      std::memcpy(sin_exp.data() + dst, sin_t + src, static_cast<size_t>(half) * sizeof(float));
    }
  }
}

// Device NeoX apply: view [T,H,D] as [T*H,D], rotate leading `rot` cols via
// slice + mul/sub/add + concat. Reuses EnsureDevice2D so a prior RmsNorm on the
// [T*H,D] view leaves the shadow resident (no re-upload). BF16 tile path.
void RopeApplyDeviceNeox(Tensor& x3, const float* cos_t, const float* sin_t,
                         int64_t tokens, int64_t heads, int64_t d, int64_t rot,
                         MeshDevice& device) {
  VT_CHECK(x3.rank == 3 && x3.IsContiguous() && x3.shape[0] == tokens &&
               x3.shape[1] == heads && x3.shape[2] == d,
           "tenstorrent device rope: rank-3 contiguous [T,H,D]");
  VT_CHECK(rot > 0 && (rot % 2) == 0 && rot <= d, "tenstorrent device rope: rotary_dim");
  const int64_t half = rot / 2;
  const int64_t th = tokens * heads;
  Tensor x_mat = x3.View({th, d});
  ttnn::Tensor dev_x = EnsureDevice2D(x_mat, device);

  std::vector<float> cos_exp, sin_exp;
  ExpandCosSinPerHead(cos_t, sin_t, tokens, heads, half, cos_exp, sin_exp);
  const uint32_t thu = static_cast<uint32_t>(th);
  const uint32_t halfu = static_cast<uint32_t>(half);
  const uint32_t rotu = static_cast<uint32_t>(rot);
  const uint32_t du = static_cast<uint32_t>(d);
  // ITEM 5: persistent cos/sin — build outside capture, copy in-region.
  const std::string rk = RopeCSKey(thu, halfu);
  ttnn::Tensor dev_cos, dev_sin;
  bool cache_hit = false;
  {
    std::lock_guard<std::mutex> g(RopeCSMutex());
    auto it = RopeCSCache().find(rk);
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
      std::fprintf(stderr,
                   "[TT-TRACE] rope lookup key=%s found=%d content_eq=%d "
                   "(want first=%f n=%zu)\n",
                   rk.c_str(), it != RopeCSCache().end(),
                   it != RopeCSCache().end() && it->second.cos_host == cos_exp,
                   cos_exp.empty() ? -1.0f : cos_exp.front(), cos_exp.size());
    if (it != RopeCSCache().end() && it->second.cos_host == cos_exp) {
      dev_cos = it->second.cos;
      dev_sin = it->second.sin;
      cache_hit = true;
    }
  }
  if (!cache_hit) {
    VT_CHECK(!tt_capture_active(),
             "tenstorrent: rope cos/sin cache miss during capture — the "
             "table changed (positions moved); the decode-graph driver must "
             "call WarmRopeCosSin for the step's positions BEFORE BeginCapture "
             "(the SizeSlot::Refresh pattern)");
    dev_cos = UploadRows(cos_exp.data(), thu, halfu, device);
    dev_sin = UploadRows(sin_exp.data(), thu, halfu, device);
    std::lock_guard<std::mutex> g(RopeCSMutex());
    RopeCSEntry e;
    e.cos = dev_cos;
    e.sin = dev_sin;
    e.cos_host = cos_exp;
    RopeCSCache()[rk] = std::move(e);
  }
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr, "[TT-TRACE] rope cos/sin cache %s during capture "
                 "(key th=%u half=%u first=%f)\n",
                 cache_hit ? "HIT" : "MISS", thu, halfu,
                 cos_exp.empty() ? -1.0f : cos_exp.front());

  // x1 = x[..., :half], x2h = x[..., half:rot]  (NeoX half-split)
  ttnn::Tensor x1 = ttnn::slice(dev_x, ttsl::SmallVector<uint32_t>{0, 0},
                                ttsl::SmallVector<uint32_t>{thu, halfu},
                                ttsl::SmallVector<uint32_t>{1, 1});
  ttnn::Tensor x2h = ttnn::slice(dev_x, ttsl::SmallVector<uint32_t>{0, halfu},
                                 ttsl::SmallVector<uint32_t>{thu, rotu},
                                 ttsl::SmallVector<uint32_t>{1, 1});
  ttnn::Tensor o1 = ttnn::subtract(ttnn::multiply(x1, dev_cos), ttnn::multiply(x2h, dev_sin));
  ttnn::Tensor o2 = ttnn::add(ttnn::multiply(x1, dev_sin), ttnn::multiply(x2h, dev_cos));
  ttnn::Tensor rotated = ttnn::concat(std::vector<ttnn::Tensor>{o1, o2}, /*dim=*/1);
  ttnn::Tensor out_dev;
  if (rot == d) {
    out_dev = std::move(rotated);
  } else {
    ttnn::Tensor tail = ttnn::slice(dev_x, ttsl::SmallVector<uint32_t>{0, rotu},
                                    ttsl::SmallVector<uint32_t>{thu, du},
                                    ttsl::SmallVector<uint32_t>{1, 1});
    out_dev = ttnn::concat(std::vector<ttnn::Tensor>{rotated, tail}, /*dim=*/1);
  }
  CommitDevice2D(x_mat, std::move(out_dev));
}

// Gather per-token cos|sin from a [P, rot] cache via rank-1 positions.
void GatherCosSinRows(const Tensor& cache, const Tensor& positions, int64_t tokens,
                      int rot, std::vector<float>& cos_t, std::vector<float>& sin_t) {
  EnsureHost(cache);
  EnsureHost(positions);
  const int64_t half = rot / 2;
  cos_t.resize(static_cast<size_t>(tokens * half));
  sin_t.resize(static_cast<size_t>(tokens * half));
  for (int64_t t = 0; t < tokens; ++t) {
    const int64_t position = positions.dtype == DType::kI32
                                 ? static_cast<int64_t>(positions.Ptr<int32_t>()[t])
                                 : positions.Ptr<int64_t>()[t];
    VT_CHECK(position >= 0 && position < cache.shape[0],
             "tenstorrent rope: position outside cache");
    const int64_t cache_off = position * rot;
    for (int64_t i = 0; i < half; ++i) {
      cos_t[static_cast<size_t>(t * half + i)] = LoadElemF32(cache, cache_off + i);
      sin_t[static_cast<size_t>(t * half + i)] =
          LoadElemF32(cache, cache_off + half + i);
    }
  }
}

// Build per-token cos|sin from RopeNeox frequencies (double angles, f32 store).
void BuildCosSinFromPositions(const Tensor& pos, int64_t tokens, int rot, double base,
                              const RopeArgs& args, std::vector<float>& cos_t,
                              std::vector<float>& sin_t) {
  EnsureHost(pos);
  const int64_t half = rot / 2;
  cos_t.resize(static_cast<size_t>(tokens * half));
  sin_t.resize(static_cast<size_t>(tokens * half));
  for (int64_t t = 0; t < tokens; ++t) {
    const int64_t p =
        pos.dtype == DType::kI32 ? pos.Ptr<int32_t>()[t] : pos.Ptr<int64_t>()[t];
    for (int64_t i = 0; i < half; ++i) {
      double freq = std::pow(base, -2.0 * static_cast<double>(i) / static_cast<double>(rot));
      freq = Llama3ScaleFreq(freq, args);
      const double angle = static_cast<double>(p) * freq;
      cos_t[static_cast<size_t>(t * half + i)] = static_cast<float>(std::cos(angle));
      sin_t[static_cast<size_t>(t * half + i)] = static_cast<float>(std::sin(angle));
    }
  }
}

// Host NeoX/GPT-J apply from a precomputed cos|sin table. Fast path for short
// decode: many tiny device launches (slice/mul/concat × q/k) lose to this.
void RopeApplyHost(Tensor& qs, Tensor* ks, const float* cos_t, const float* sin_t,
                   int64_t tokens, int64_t hq, int64_t hk, int64_t d, int rot,
                   bool is_neox) {
  EnsureHost(qs);
  if (ks != nullptr) EnsureHost(*ks);
  const int64_t half = rot / 2;
  auto apply_one = [&](Tensor& x, int64_t heads) {
    for (int64_t token = 0; token < tokens; ++token) {
      for (int64_t pair = 0; pair < half; ++pair) {
        const float c = cos_t[static_cast<size_t>(token * half + pair)];
        const float s = sin_t[static_cast<size_t>(token * half + pair)];
        const int64_t first = is_neox ? pair : pair * 2;
        const int64_t second = is_neox ? pair + half : pair * 2 + 1;
        for (int64_t head = 0; head < heads; ++head) {
          const int64_t off = (token * heads + head) * d;
          const float xv = LoadElemF32(x, off + first);
          const float yv = LoadElemF32(x, off + second);
          StoreElemF32(x, off + first, xv * c - yv * s);
          StoreElemF32(x, off + second, xv * s + yv * c);
        }
      }
    }
  };
  apply_one(qs, hq);
  if (ks != nullptr) apply_one(*ks, hk);
  CommitHost(qs);
  if (ks != nullptr) CommitHost(*ks);
}

// Prefer device apply only when T*H amortizes the slice/mul/concat launches.
// Short Qwen3 decode (T=1,H=16) is host-faster even when Q is already on device
// (measured regression when always-device-for-resident was forced).

inline bool PreferDeviceRope(int64_t tokens, int64_t heads) {
  // HOST-FREE-FORWARD R1: force device RoPE at T=1 for capture (see RmsNorm note).
  if (HostFreeDecodeEnabled()) return true;
  return tokens * heads >= 64;
}

// kRopeNeox: Qwen3-dense RoPE. Device NeoX for large [T*H]; host for short decode.
void RopeNeoxKernel(Queue&, Tensor& qs, Tensor& ks, const Tensor& pos, const RopeArgs& args) {
  TT_OP_TRACE("RopeNeox");
  VT_CHECK(qs.rank == 3 && ks.rank == 3, "tenstorrent kRopeNeox: qs/ks rank-3");
  VT_CHECK(IsFloatDType(qs.dtype) && qs.dtype == ks.dtype,
           "tenstorrent kRopeNeox: qs/ks float same dtype");
  VT_CHECK(pos.rank == 1 && (pos.dtype == DType::kI32 || pos.dtype == DType::kI64),
           "tenstorrent kRopeNeox: positions rank-1 i32/i64");
  VT_CHECK(qs.IsContiguous() && ks.IsContiguous() && pos.IsContiguous(),
           "tenstorrent kRopeNeox: contiguous required");
  VT_CHECK(args.rotary_dim > 0 && (args.rotary_dim % 2) == 0 &&
               args.rotary_dim <= qs.shape[2],
           "tenstorrent kRopeNeox: rotary_dim must be even and <= head_dim");
  const int64_t t = qs.shape[0], hq = qs.shape[1], hk = ks.shape[1], d = qs.shape[2];
  VT_CHECK(ks.shape[0] == t && ks.shape[2] == d, "tenstorrent kRopeNeox: ks shape");
  VT_CHECK(pos.shape[0] == t, "tenstorrent kRopeNeox: positions length");

  std::vector<float> cos_t, sin_t;
  BuildCosSinFromPositions(pos, t, args.rotary_dim, static_cast<double>(args.base), args, cos_t,
                           sin_t);
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr, "[TT-TRACE] rope kernel pos0=%d t=%lld hq=%lld cos_first=%f\n",
                 (int)(pos.dtype == DType::kI32 ? pos.Ptr<int32_t>()[0]
                                                : static_cast<int32_t>(pos.Ptr<int64_t>()[0])),
                 (long long)t, (long long)hq,
                 cos_t.empty() ? -1.0f : cos_t.front());
  if (PreferDeviceRope(t, hq)) {
    MeshDevice& device = SharedMeshDevice();
    RopeApplyDeviceNeox(qs, cos_t.data(), sin_t.data(), t, hq, d, args.rotary_dim, device);
    RopeApplyDeviceNeox(ks, cos_t.data(), sin_t.data(), t, hk, d, args.rotary_dim, device);
  } else {
    RopeApplyHost(qs, &ks, cos_t.data(), sin_t.data(), t, hq, hk, d, args.rotary_dim,
                  /*is_neox=*/true);
  }
}

// kRopeCosSinCache: per-step cos|sin table [T, rot] (cpu_ops RopeCosSinCacheKernel).
// Stays host — table is small and built once per step; apply is device.
void RopeCosSinCacheKernel(Queue&, Tensor& cos_sin, const Tensor& positions,
                           const RopeArgs& args) {
  VT_CHECK(cos_sin.rank == 2 && cos_sin.dtype == DType::kF32 && cos_sin.IsContiguous(),
           "tenstorrent kRopeCosSinCache: cos_sin contiguous f32 [T,rot]");
  VT_CHECK(positions.rank == 1 &&
               (positions.dtype == DType::kI32 || positions.dtype == DType::kI64) &&
               positions.IsContiguous(),
           "tenstorrent kRopeCosSinCache: positions rank-1 i32/i64");
  VT_CHECK(args.rotary_dim > 0 && (args.rotary_dim % 2) == 0,
           "tenstorrent kRopeCosSinCache: rotary_dim even > 0");
  const int64_t t = cos_sin.shape[0];
  const int rot = args.rotary_dim;
  VT_CHECK(cos_sin.shape[1] == rot && positions.shape[0] == t,
           "tenstorrent kRopeCosSinCache: shape mismatch");
  EnsureHost(positions);
  const int64_t half = rot / 2;
  const double base = static_cast<double>(args.base);
  for (int64_t i = 0; i < t; ++i) {
    const int64_t p = positions.dtype == DType::kI32 ? positions.Ptr<int32_t>()[i]
                                                     : positions.Ptr<int64_t>()[i];
    for (int64_t pair = 0; pair < half; ++pair) {
      double freq =
          std::pow(base, -2.0 * static_cast<double>(pair) / static_cast<double>(rot));
      freq = Llama3ScaleFreq(freq, args);
      const double angle = static_cast<double>(p) * freq;
      StoreElemF32(cos_sin, i * rot + pair, static_cast<float>(std::cos(angle)));
      StoreElemF32(cos_sin, i * rot + half + pair, static_cast<float>(std::sin(angle)));
    }
  }
  CommitHost(cos_sin);
}

// kRopeFromCache: apply precomputed cos|sin (cpu_ops RopeFromCacheKernel).
// Rank-1 positions only (Qwen3-dense); mrope deferred. DEFAULT Qwen3 path
// (VT_QWEN3_ROPE_CACHE). Device NeoX when T*H is large; host for short decode
// and GPT-J interleave.
void RopeFromCacheKernel(Queue&, Tensor& qs, Tensor* ks, const Tensor& positions,
                         const Tensor& cache, const RopeArgs& args) {
  VT_CHECK(qs.rank == 3 && IsFloatDType(qs.dtype) && qs.IsContiguous(),
           "tenstorrent kRopeFromCache: qs rank-3 contiguous float");
  VT_CHECK(positions.rank == 1 &&
               (positions.dtype == DType::kI32 || positions.dtype == DType::kI64) &&
               positions.IsContiguous(),
           "tenstorrent kRopeFromCache: rank-1 positions only (no mrope yet)");
  VT_CHECK(cache.rank == 2 && IsFloatDType(cache.dtype) && cache.IsContiguous(),
           "tenstorrent kRopeFromCache: cache rank-2 contiguous float");
  VT_CHECK(args.rotary_dim > 0 && (args.rotary_dim % 2) == 0 &&
               args.rotary_dim <= qs.shape[2],
           "tenstorrent kRopeFromCache: rotary_dim");
  const int64_t tokens = qs.shape[0];
  const int64_t hq = qs.shape[1];
  const int64_t d = qs.shape[2];
  const int64_t hk = ks == nullptr ? 0 : ks->shape[1];
  if (ks != nullptr) {
    VT_CHECK(ks->rank == 3 && ks->dtype == qs.dtype && ks->IsContiguous(),
             "tenstorrent kRopeFromCache: ks must match qs");
    VT_CHECK(ks->shape[0] == tokens && ks->shape[2] == d,
             "tenstorrent kRopeFromCache: ks shape");
  }
  VT_CHECK(positions.shape[0] == tokens, "tenstorrent kRopeFromCache: positions length");

  std::vector<float> cos_t, sin_t;
  GatherCosSinRows(cache, positions, tokens, args.rotary_dim, cos_t, sin_t);

  if (args.is_neox_style && PreferDeviceRope(tokens, hq)) {
    MeshDevice& device = SharedMeshDevice();
    RopeApplyDeviceNeox(qs, cos_t.data(), sin_t.data(), tokens, hq, d, args.rotary_dim, device);
    if (ks != nullptr) {
      RopeApplyDeviceNeox(*ks, cos_t.data(), sin_t.data(), tokens, hk, d, args.rotary_dim,
                          device);
    }
    return;
  }
  RopeApplyHost(qs, ks, cos_t.data(), sin_t.data(), tokens, hq, hk, d, args.rotary_dim,
                args.is_neox_style);
}

// kQkvSplit: column split of merged [T, q+k+v] into q/k/v (cpu_ops QkvSplitKernel).
//
// Device path when qkv already has a resident shadow (post MatmulBT): slice the
// last dim on-device and CommitDevice2D each shard so qk-RmsNorm can reshape-
// reuse without download+reupload. Host path (bit-exact memcpy) when qkv is
// host-only — unit tests and weight-load style callers.
void QkvSplitKernel(Queue&, Tensor& q_out, Tensor& k_out, Tensor& v_out, const Tensor& qkv) {
  TT_OP_TRACE("QkvSplit");
  VT_CHECK(qkv.rank == 2 && IsFloatDType(qkv.dtype),
           "tenstorrent kQkvSplit: rank-2 float qkv required");
  VT_CHECK(q_out.dtype == qkv.dtype && k_out.dtype == qkv.dtype && v_out.dtype == qkv.dtype,
           "tenstorrent kQkvSplit: q/k/v out must match qkv dtype");
  VT_CHECK(q_out.IsContiguous() && k_out.IsContiguous() && v_out.IsContiguous() &&
               qkv.IsContiguous(),
           "tenstorrent kQkvSplit: contiguous required");
  const int64_t t = qkv.shape[0];
  const int64_t q_dim = q_out.Numel() / t;
  const int64_t k_dim = k_out.Numel() / t;
  const int64_t v_dim = v_out.Numel() / t;
  const int64_t total = q_dim + k_dim + v_dim;
  VT_CHECK(qkv.shape[1] == total, "tenstorrent kQkvSplit: inner dim mismatch");
  VT_CHECK(q_out.rank == 2 && k_out.rank == 2 && v_out.rank == 2 &&
               q_out.shape[0] == t && k_out.shape[0] == t && v_out.shape[0] == t &&
               q_out.shape[1] == q_dim && k_out.shape[1] == k_dim && v_out.shape[1] == v_dim,
           "tenstorrent kQkvSplit: out shapes must be [T, *]");

  const uint32_t tu = static_cast<uint32_t>(t);
  const uint32_t total_u = static_cast<uint32_t>(total);
  const uint32_t qd = static_cast<uint32_t>(q_dim);
  const uint32_t kd = static_cast<uint32_t>(k_dim);
  const uint32_t vd = static_cast<uint32_t>(v_dim);

  if (DeviceShadowExact(qkv, tu, total_u)) {
    MeshDevice& device = SharedMeshDevice();
    ttnn::Tensor dev = EnsureDevice2D(qkv, device);
    ttnn::Tensor dq = ttnn::slice(dev, ttsl::SmallVector<uint32_t>{0, 0},
                                  ttsl::SmallVector<uint32_t>{tu, qd},
                                  ttsl::SmallVector<uint32_t>{1, 1});
    ttnn::Tensor dk = ttnn::slice(dev, ttsl::SmallVector<uint32_t>{0, qd},
                                  ttsl::SmallVector<uint32_t>{tu, qd + kd},
                                  ttsl::SmallVector<uint32_t>{1, 1});
    ttnn::Tensor dv = ttnn::slice(dev, ttsl::SmallVector<uint32_t>{0, qd + kd},
                                  ttsl::SmallVector<uint32_t>{tu, qd + kd + vd},
                                  ttsl::SmallVector<uint32_t>{1, 1});
    CommitDevice2D(q_out, std::move(dq));
    CommitDevice2D(k_out, std::move(dk));
    CommitDevice2D(v_out, std::move(dv));
    return;
  }

  EnsureHost(qkv);
  const size_t esz = SizeOf(qkv.dtype);
  const auto* src = static_cast<const uint8_t*>(qkv.data);
  auto* qdst = static_cast<uint8_t*>(q_out.data);
  auto* kdst = static_cast<uint8_t*>(k_out.data);
  auto* vdst = static_cast<uint8_t*>(v_out.data);
  for (int64_t i = 0; i < t; ++i) {
    const uint8_t* row = src + static_cast<size_t>(i * total) * esz;
    std::memcpy(qdst + static_cast<size_t>(i * q_dim) * esz, row,
                static_cast<size_t>(q_dim) * esz);
    std::memcpy(kdst + static_cast<size_t>(i * k_dim) * esz, row + static_cast<size_t>(q_dim) * esz,
                static_cast<size_t>(k_dim) * esz);
    std::memcpy(vdst + static_cast<size_t>(i * v_dim) * esz,
                row + static_cast<size_t>(q_dim + k_dim) * esz, static_cast<size_t>(v_dim) * esz);
  }
  CommitHost(q_out);
  CommitHost(k_out);
  CommitHost(v_out);
}

// kGreedyArgmax: per-row lowest-index max of f32 logits (cpu_sample.cpp).
// OPT's lm_head produces F32 logits; host-staged, bit-exact with CPU.
void GreedyArgmaxKernel(Queue&, Tensor& token_ids, const Tensor& logits) {
  VT_CHECK(logits.rank == 2 && logits.dtype == DType::kF32 && logits.IsContiguous(),
           "tenstorrent kGreedyArgmax: logits must be contiguous f32 [N,V]");
  VT_CHECK(token_ids.rank == 1 && token_ids.dtype == DType::kI64 && token_ids.IsContiguous() &&
               token_ids.shape[0] == logits.shape[0],
           "tenstorrent kGreedyArgmax: token_ids must be i64 [N]");
  EnsureHost(logits);
  const int64_t n = logits.shape[0], v = logits.shape[1];
  const float* lp = logits.Ptr<float>();
  int64_t* out = token_ids.Ptr<int64_t>();
  for (int64_t i = 0; i < n; ++i) {
    const float* row = lp + i * v;
    int64_t best = 0;
    float best_v = row[0];
    for (int64_t j = 1; j < v; ++j) {
      if (row[j] > best_v) {
        best_v = row[j];
        best = j;
      }
    }
    out[i] = best;
  }
  CommitHost(token_ids);
}

// ---- GDN prefill op set (BACKEND-TENSTORRENT-GDN W1) -------------------------
// kL2Norm / kRmsNormGated / kCausalConv1dFwd / kGdnPrefill: the op chain the
// Qwen3.5-family GDN layer issues in prefill. The CPU f32 arm (cpu_ops.cpp
// GdnPrefillKernel / CausalConv1dFwdKernel / L2NormKernel / RmsNormGatedKernel)
// is the correctness oracle; the tt-metal substrate is the implementation.
// Device-composed: kL2Norm (square → row-sum → rsqrt → scale, bf16 tiles),
// kRmsNormGated (ttnn::rms_norm + silu/sigmoid gate eltwise), kGdnPrefill
// (ttnn::transformer::chunk_gated_delta_rule behind a varlen→dense adapter).
// Host-staged in W1: kCausalConv1dFwd (the varlen window build + rolling
// conv_state writeback is pure data movement at these shapes; the conv-state
// device shadow that would make a composed path win is W2's decode work).

}  // namespace

// Rank-flexible rows view of a contiguous float tensor as [rows, cols] TILE
// bf16 on device: reuses a resident same-numel shadow without a host
// round-trip (the EnsureDevice2D residency win, reached from rank-3 GDN
// shapes [T,H,D]).
ttnn::Tensor DeviceRows(const Tensor& t, uint32_t rows, uint32_t cols,
                        MeshDevice& device) {
  if (t.rank == 2 && t.IsContiguous()) return EnsureDevice2D(t, device);
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(t.data);
    if (s != nullptr && s->device_current && s->device.has_value() &&
        static_cast<uint64_t>(s->dev_rows) * s->dev_cols ==
            static_cast<uint64_t>(rows) * cols) {
      // Same re-layout discipline as EnsureDevice2D's exact hit: round-trip
      // ROW_MAJOR so a non-tile-aligned row count cannot reinterpret the
      // owner's padded tile rows.
      ttnn::Tensor reshaped = ttnn::to_layout(
          ttnn::reshape(ttnn::to_layout(*s->device, ttnn::Layout::ROW_MAJOR),
                        ttnn::Shape({rows, cols})),
          s->device->layout());
      s->device = reshaped;
      s->dev_rows = rows;
      s->dev_cols = cols;
      return reshaped;
    }
  }
  EnsureHost(t);
  const auto host = ToHostF32(t);
  return UploadRows(host.data(), rows, cols, device);
}

// Upload an arbitrary-rank host f32 buffer as a device tensor of `dtype` /
// `layout` (from_vector handles tile padding of logical dims).
ttnn::Tensor UploadTensor(std::vector<float> host, const ttnn::Shape& shape,
                          ttnn::DataType dtype, ttnn::Layout layout,
                          MeshDevice& device) {
  return ttnn::Tensor::from_vector<float>(
      std::move(host),
      tt::tt_metal::TensorSpec(tt::tt_metal::Shape(shape),
                               tt::tt_metal::TensorLayout(
                                   dtype, tt::tt_metal::PageConfig(layout),
                                   tt::tt_metal::MemoryConfig{})),
      &device);
}

namespace {

// kL2Norm: y = x * rsqrt(sum(x^2) + eps) over the last dim (cpu_ops.cpp
// L2NormKernel; gdn-semantics.md §4). GDN callers run it on q/k [T,H,D] rows.
// Device path: square → row sum → +eps → rsqrt → broadcast multiply, bf16
// tiles (same envelope as kRmsNorm).
void L2NormKernel(Queue&, Tensor& out, const Tensor& x, const L2NormArgs& args) {
  TT_OP_TRACE("L2Norm");
  VT_CHECK(x.rank == 2 || x.rank == 3,
           "tenstorrent kL2Norm: rank 2 or 3 required");
  VT_CHECK(out.rank == x.rank, "tenstorrent kL2Norm: out rank must match x");
  VT_CHECK(IsFloatDType(x.dtype) &&
               (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kL2Norm: float in, f32/bf16 out");
  VT_CHECK(x.IsContiguous() && out.IsContiguous(),
           "tenstorrent kL2Norm: contiguous required");
  const uint32_t d = static_cast<uint32_t>(x.shape[x.rank - 1]);
  const uint32_t rows = static_cast<uint32_t>(x.Numel() / d);

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_x = DeviceRows(x, rows, d, device);
  ttnn::Tensor sq = ttnn::multiply(dev_x, dev_x);
  ttnn::Tensor s = ttnn::sum(sq, ttsl::SmallVector<int>{1}, true);
  ttnn::Tensor denom = ttnn::add(s, args.eps);
  ttnn::Tensor inv = ttnn::rsqrt(denom);
  ttnn::Tensor dev_y = ttnn::multiply(dev_x, inv);
  CommitDeviceLogical2D(out, std::move(dev_y), rows, d);
}

// kRmsNormGated: out = x * rsqrt(mean(x^2)+eps) * w * act(gate) with
// norm_before_gate=True semantics baked in (cpu_ops.cpp RmsNormGatedKernel;
// gdn-semantics.md §5). Device path reuses the kRmsNorm machinery
// (ttnn::rms_norm with the [1,D] affine upload) + a silu/sigmoid gate pass.
// The gate may be a padded-row rank-3 view of the merged qkvz z-slice: its
// rows are gathered honoring the token stride and uploaded compactly.
void RmsNormGatedKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& gate,
                        const Tensor& weight, const RmsNormGatedArgs& args) {
  TT_OP_TRACE("RmsNormGated");
  VT_CHECK((x.rank == 2 || x.rank == 3) && gate.rank == x.rank &&
               out.rank == x.rank && weight.rank == 1,
           "tenstorrent kRmsNormGated: x/gate/out rank-2 or rank-3, weight rank-1");
  VT_CHECK(IsFloatDType(x.dtype) && IsFloatDType(gate.dtype) &&
               IsFloatDType(weight.dtype) &&
               (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kRmsNormGated: float in, f32/bf16 out");
  VT_CHECK(x.IsContiguous() && weight.IsContiguous() && out.IsContiguous(),
           "tenstorrent kRmsNormGated: x/out/weight contiguous required");
  const uint32_t d = static_cast<uint32_t>(x.shape[x.rank - 1]);
  const uint32_t rows = static_cast<uint32_t>(x.Numel() / d);
  VT_CHECK(weight.shape[0] == d,
           "tenstorrent kRmsNormGated: weight size mismatch");

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_x = DeviceRows(x, rows, d, device);
  ttnn::Tensor dev_w = EnsureAffine1D(weight, d, device);
  // The gate rides its device shadow in the captured arm (a committed
  // projection output, or a packed slice of one via ServeDeviceWindow).
  // EnsureHost on a committed gate is exactly the readback the trace
  // refuses, and the gathered re-upload below is a capture-time write —
  // both stay in the eager arm only.
  ttnn::Tensor dev_g;
  if (ServeDeviceShadowRaw(gate, rows, d, dev_g) ||
      ServeDeviceWindow(gate, rows, d, dev_g)) {
    dev_g = NormalizeDevF32Tile(std::move(dev_g), rows, d);
  } else {
    VT_CHECK(!tt_capture_active(),
             "tenstorrent kRmsNormGated: gate arrived without a servable "
             "device shadow during trace capture — the producer must commit "
             "device-side before the captured region");
    EnsureHost(gate);
    const int64_t gate_group = gate.rank == 3 ? gate.shape[1] : 1;
    const int64_t gate_outer = gate.stride[0];
    std::vector<float> gh(static_cast<size_t>(rows) * d);
    for (uint32_t i = 0; i < rows; ++i) {
      const int64_t gbase =
          (i / gate_group) * gate_outer + (i % gate_group) * d;
      for (uint32_t j = 0; j < d; ++j)
        gh[static_cast<size_t>(i) * d + j] = LoadElemF32(gate, gbase + j);
    }
    dev_g = UploadRows(gh.data(), rows, d, device);
  }
  ttnn::Tensor act =
      args.sigmoid_gate ? ttnn::sigmoid(dev_g) : ttnn::silu(dev_g);
  // Reshape act to [rows, d] to match rms_norm(dev_x) output. The gate
  // shadow's native geometry (e.g. [256, 6144] for APEX's gated-norm
  // activation) is preserved by NormalizeDevF32Tile (the L1 overflow fix:
  // a free reshape to [1, n] overflows L1, and view_device on ROW_MAJOR
  // with a changed last dim corrupts data). act is TILE layout here
  // (NormalizeDevF32Tile's final to_layout(TILE), preserved by the
  // elementwise sigmoid/silu), so CaptureSafeReshape's member view_device
  // takes the TILE safe path (tensor_ops.cpp:398: layout != ROW_MAJOR
  // short-circuits before the page_size rebuild that corrupts ROW_MAJOR).
  // Same numel (256*6144 == 12288*128), same tile count (1536), so the
  // reshape is a pure metadata view — no bytes move, no device program.
  {
    const auto target_shape = ttnn::Shape({rows, d});
    if (act.logical_shape() != target_shape) {
      act = CaptureSafeReshape(act, target_shape);
    }
  }
  if (std::getenv("VT_TT_SLOT_TRACE") != nullptr) {
    std::fprintf(stderr,
                 "[TT-RNG] rows=%u d=%u x=%s g=%s act=%s\n", rows, d,
                 DevShapeStr(dev_x).c_str(), DevShapeStr(dev_g).c_str(),
                 DevShapeStr(act).c_str());
    std::fflush(stderr);
  }
  ttnn::Tensor dev_y = ttnn::multiply(ttnn::rms_norm(dev_x, args.eps, dev_w), act);
  CommitDeviceLogical2D(out, std::move(dev_y), rows, d);
}

// kCausalConv1dFwd: depthwise causal conv over time with the rolling
// conv_state writeback (cpu_ops.cpp CausalConv1dFwdKernel; gdn-semantics.md
// §2). W1 HOST-STAGED: this backend's Alloc is host memory
// (tenstorrent_backend.cpp), so the scalar port below runs the exact oracle
// instruction order on the host bytes — outputs read the OLD state row
// (buffered), the new row carries the last K-1 RAW x tokens (left-shifted
// from the old state when T < K-1). A composed slice/concat+MAC path pays a
// full [T*K,C] window materialization to build what this loop streams; the
// conv-state device shadow that would flip that trade is W2's decode work.
void CausalConv1dFwdKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& w,
                           const Tensor* bias, Tensor& conv_state,
                           const Tensor& qsl, const Tensor& his,
                           const CausalConv1dArgs& args) {
  TT_OP_TRACE("CausalConv1dFwd");
  const int64_t total = x.shape[0], c_dim = x.shape[1], k = w.shape[1];
  const int64_t width = k - 1;
  const int64_t n = conv_state.shape[0];
  const int64_t x_rs = x.stride[0];
  EnsureHost(x);
  EnsureHost(w);
  if (bias != nullptr) EnsureHost(*bias);
  EnsureHost(conv_state);
  EnsureHost(qsl);
  EnsureHost(his);
  const int32_t* qslp = qsl.Ptr<int32_t>();
  VT_CHECK(qslp[0] == 0 && qslp[n] == total,
           "tenstorrent causal_conv1d_fwd: bad query_start_loc bounds");
  for (int64_t s = 0; s < n; ++s) {
    VT_CHECK(qslp[s + 1] >= qslp[s] && qslp[s] >= 0,
             "tenstorrent causal_conv1d_fwd: query_start_loc not monotonic");
  }
  // Per (sequence, channel): mirrors the oracle's row-chunked decomposition —
  // disjoint out columns / conv_state rows, so the loop order is free.
  std::vector<float> old_row(static_cast<size_t>(width));
  for (int64_t s = 0; s < n; ++s) {
    const bool init = his.dtype == DType::kI8 ? his.Ptr<int8_t>()[s] != 0
                                              : his.Ptr<int32_t>()[s] != 0;
    const int64_t begin = qslp[s], t_len = qslp[s + 1] - begin;
    for (int64_t c = 0; c < c_dim; ++c) {
      float* srow = conv_state.Ptr<float>() + (s * c_dim + c) * width;
      for (int64_t j = 0; j < width; ++j)
        old_row[static_cast<size_t>(j)] = srow[j];
      const float b = bias != nullptr ? LoadElemF32(*bias, c) : 0.0f;
      for (int64_t t = 0; t < t_len; ++t) {
        float acc = b;
        for (int64_t j = 0; j < k; ++j) {
          const int64_t ti = t - (k - 1 - j);  // token index of window[j]
          float v = 0.0f;
          if (ti >= 0) {
            v = LoadElemF32(x, (begin + ti) * x_rs + c);
          } else if (init) {
            v = old_row[static_cast<size_t>(width + ti)];  // state col (K-1)+(t-i)
          }
          acc += LoadElemF32(w, c * k + j) * v;
        }
        const float y = args.silu_activation
                            ? acc / (1.0f + std::exp(-acc))
                            : acc;
        StoreElemF32(out, (begin + t) * c_dim + c, y);
      }
      for (int64_t j = 0; j < width; ++j) {
        const int64_t tj = t_len - width + j;  // new state col j holds token tj
        float v = 0.0f;
        if (tj >= 0) {
          v = LoadElemF32(x, (begin + tj) * x_rs + c);
        } else if (init) {
          v = old_row[static_cast<size_t>(width + tj)];  // shifted old state
        }
        srow[j] = v;
      }
    }
  }
  CommitHost(out);
  CommitHost(conv_state);
}

// ==== W2: the decode set =====================================================
// kCausalConv1dUpdate / kGdnDecode (both state_idx forms) / kGdnStateGather /
// kGdnStateScatter. The decode state and the conv state live in DEVICE
// shadows keyed by the host pointer — the same BufferSlot residency every
// other op uses (the PagedKvShadow discipline, but the GDN caches keep the
// CALLER's layout on device so EnsureHostBytes stays byte-correct): a host
// write via Backend::Copy drops the shadow (MarkHostWritten), a host read
// downloads it once (EnsureHostBytes), and a decode step updates it in place
// with ZERO PCIe state traffic. Indexed forms gather/scatter rows on-device
// through exact 0/1 one-hot matmuls (0*x + v == v in f32), so the full cache
// never moves either.

}  // namespace

// Upload a small u32 index vector as a device tensor (ROW_MAJOR — the
// indexed_fill batch_id contract; gather takes TILE, reshaped at the call).
ttnn::Tensor UploadIdxU32(std::vector<uint32_t> host, const ttnn::Shape& shape,
                          ttnn::Layout layout, MeshDevice& device) {
  return ttnn::Tensor::from_vector<uint32_t>(
      std::move(host),
      tt::tt_metal::TensorSpec(tt::tt_metal::Shape(shape),
                               tt::tt_metal::TensorLayout(
                                   ttnn::DataType::UINT32,
                                   tt::tt_metal::PageConfig(layout),
                                   tt::tt_metal::MemoryConfig{})),
      &device);
}

// EXACT row gather from a [slots, cols] cache: ttnn::gather is pure data
// movement, so the bytes land bit-identical (the one-hot matmul alternative
// rounds its inputs to tf32 on this hardware — measured 2.4e-4 — and breaks
// the conv/cache-I/O 1e-4 envelopes). idx<0 = NULL row: gathered as a ZERO
// row (index clamped to 0, then masked) — the decode/conv NULL semantics.
ttnn::Tensor GatherRowsExact(const ttnn::Tensor& cache2d,
                             const std::vector<int32_t>& idx, int64_t cols,
                             MeshDevice& device) {
  const int64_t rows = static_cast<int64_t>(idx.size());
  const ttnn::Layout lay = cache2d.layout();
  // gather demands TILE on both tensors; the layout conversion is a pure
  // copy (f32 bytes unchanged).
  ttnn::Tensor tin = lay == ttnn::Layout::TILE
                         ? cache2d
                         : ttnn::to_layout(cache2d, ttnn::Layout::TILE);
  bool has_null = false;
  std::vector<uint32_t> flat(static_cast<size_t>(rows) * cols);
  for (int64_t r = 0; r < rows; ++r) {
    const int32_t ix = idx[static_cast<size_t>(r)];
    if (ix < 0) has_null = true;
    const uint32_t u = ix < 0 ? 0u : static_cast<uint32_t>(ix);
    for (int64_t c = 0; c < cols; ++c)
      flat[static_cast<size_t>(r) * cols + c] = u;
  }
  ttnn::Tensor index = UploadIdxU32(
      std::move(flat),
      ttnn::Shape({static_cast<uint32_t>(rows), static_cast<uint32_t>(cols)}),
      ttnn::Layout::TILE, device);
  ttnn::Tensor out = ttnn::gather(tin, /*dim=*/0, index,
                                  /*sparse_grad=*/false, std::nullopt);
  if (has_null) {
    std::vector<float> valid(static_cast<size_t>(rows));
    for (int64_t r = 0; r < rows; ++r)
      valid[static_cast<size_t>(r)] = idx[static_cast<size_t>(r)] >= 0 ? 1.0f : 0.0f;
    out = ttnn::multiply(
        out, UploadTensor(std::move(valid),
                          ttnn::Shape({static_cast<uint32_t>(rows), 1}),
                          ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device));
  }
  return lay == ttnn::Layout::TILE ? out : ttnn::to_layout(out, lay);
}

// EXACT scatter of rows2d [rows*factor, cols/factor] into cache2d
// [slots*factor, cols/factor] (both shadows are born split by SplitFactor;
// the LOGICAL geometry both kernels serve is [rows, cols]) via indexed_fill
// (torch.index_copy_ semantics): pure data movement, unnamed slots keep
// their bytes, and a slot named by SEVERAL rows keeps the LAST row — the
// CPU oracle's sequential loop order. idx<0 = NULL row: writes nothing (its
// row is compacted away before the fill). Each live slot s expands into its
// `factor` block ids s*F+j, so source block-row (e, j) lands in slot
// idx[e]'s j-th block; relative entry order is preserved inside every block
// group, so per-(slot, block) last-of-duplicates wins exactly like the
// whole-row form.
// The scatter core on a PREBUILT [rows*factor] u32 block-id tensor — the
// in-region (captured) form, where the ids were warmed outside capture and
// the per-call upload would be the enqueue_write the trace refuses. `rows2d`
// may be either layout; the ROW_MAJOR staging view keeps the last dim
// (metadata-only rank change — see the geometry note below).
ttnn::Tensor ScatterRowsDevice(const ttnn::Tensor& cache2d,
                               const ttnn::Tensor& bid_dev,
                               const ttnn::Tensor& rows2d, int64_t slots,
                               int64_t cols, int64_t factor) {
  const uint32_t blk = static_cast<uint32_t>(cols / factor);
  const uint32_t unb = static_cast<uint32_t>(bid_dev.logical_shape()[0]);
  const ttnn::Tensor& src = rows2d;
  const ttnn::Layout lay = cache2d.layout();
  ttnn::Tensor cache_rm =
      lay == ttnn::Layout::ROW_MAJOR ? cache2d : ttnn::to_layout(cache2d, ttnn::Layout::ROW_MAJOR);
  ttnn::Tensor src_rm =
      lay == ttnn::Layout::ROW_MAJOR ? src : ttnn::to_layout(src, ttnn::Layout::ROW_MAJOR);
  ttnn::Tensor out4 = ttnn::indexed_fill(
      bid_dev,
      ttnn::experimental::view(
          cache_rm, ttnn::Shape({static_cast<uint32_t>(slots * factor), 1, 1, blk})),
      ttnn::experimental::view(src_rm, ttnn::Shape({unb, 1, 1, blk})),
      std::nullopt, /*dim=*/0);
  ttnn::Tensor out2 = ttnn::experimental::view(
      out4, ttnn::Shape({static_cast<uint32_t>(slots * factor), blk}));
  return lay == ttnn::Layout::ROW_MAJOR ? out2 : ttnn::to_layout(out2, lay);
}

ttnn::Tensor ScatterRowsExact(const ttnn::Tensor& cache2d,
                              const std::vector<int32_t>& idx,
                              const ttnn::Tensor& rows2d, int64_t slots,
                              int64_t cols, int64_t factor, MeshDevice& device) {
  const uint32_t blk = static_cast<uint32_t>(cols / factor);
  std::vector<uint32_t> bid;
  bid.reserve(idx.size());
  for (int32_t ix : idx)
    if (ix >= 0) bid.push_back(static_cast<uint32_t>(ix));
  if (bid.empty()) return cache2d;  // all NULL: cache untouched
  ttnn::Tensor src = rows2d;
  if (rows2d.layout() != cache2d.layout())
    src = ttnn::to_layout(rows2d, cache2d.layout());  // pure copy, f32 exact
  if (bid.size() != idx.size()) {
    // Compact the live rows through the same exact gather: row_of[e] is the
    // ORIGINAL row of the e-th live entry — expanded to its `factor`
    // block-rows, which sit contiguous and in order.
    std::vector<int32_t> row_of;
    row_of.reserve(bid.size());
    for (int64_t r = 0; r < static_cast<int64_t>(idx.size()); ++r)
      if (idx[static_cast<size_t>(r)] >= 0) row_of.push_back(static_cast<int32_t>(r));
    std::vector<int32_t> row_of_exp;
    row_of_exp.reserve(row_of.size() * static_cast<size_t>(factor));
    for (int32_t r : row_of)
      for (int64_t j = 0; j < factor; ++j)
        row_of_exp.push_back(static_cast<int32_t>(r * factor + j));
    src = GatherRowsExact(src, row_of_exp, blk, device);
  }
  // Expand each live slot into its `factor` block ids before the upload.
  std::vector<uint32_t> blocks;
  blocks.reserve(bid.size() * static_cast<size_t>(factor));
  for (uint32_t s : bid)
    for (int64_t j = 0; j < factor; ++j)
      blocks.push_back(s * static_cast<uint32_t>(factor) + static_cast<uint32_t>(j));
  const uint32_t unb = static_cast<uint32_t>(blocks.size());
  ttnn::Tensor bid_dev = UploadIdxU32(
      std::move(blocks), ttnn::Shape({unb}), ttnn::Layout::ROW_MAJOR, device);
  // indexed_fill wants rank-4 on dim 0, and the rank change must stay a
  // ZERO-COPY view, never ttnn::reshape: a TILE rank-4 reshape of
  // [rows, cols] pads the trailing 1-dims to the 32-wide tile (physical
  // x1024 — the Qwen3.5 GDN ssm cache asked for 32 GiB and OOM'd the P150),
  // and a ROW_MAJOR reshape launches a data-movement program whose circular
  // buffers scale with the tensor (4.3 MB > 1.5 MB L1 for the 4 GiB cache) —
  // both found by the W0 sweep (BACKEND-TENSTORRENT-QWEN35, runs 2-5). So:
  // convert to ROW_MAJOR once (to_layout — a pure copy, bytes unchanged),
  // then view rank-4. The views keep the LAST dim (the block width), which
  // is the metadata-only case that preserves flat order; a last-dim-changing
  // view would scramble the bank interleave (see SplitFactor above). The
  // staging per launch is 2 x blk x elem_size, inside L1 by construction.
  // Semantics are unchanged: pure data movement, unnamed slots keep their
  // bytes, last-of-duplicates wins (the oracle's loop order).
  return ScatterRowsDevice(cache2d, bid_dev, src, slots, cols, factor);
}

namespace {

// The conv-side entry: scatter/gather ids, the scatter base, the roll masks.
// Refreshes eagerly when the idx content changed; refuses to refresh during
// capture (a content change in-region means a slot change the recapture
// cadence owns — the eager boundary lane re-warms after the graph reset).
GdnIdxCacheEntry GdnConvIdxEntry(const std::vector<int32_t>& idxv,
                                 uint32_t slots, uint32_t C,
                                 MeshDevice& device) {
  const uint32_t R = slots * C;
  const std::array<uint64_t, 3> key{slots, C, idxv.size()};
  std::lock_guard<std::mutex> g(GdnIdxCacheMutex());
  GdnIdxCacheEntry& e = GdnIdxCacheMap()[key];
  if (e.warmed_conv && e.idx == idxv) return e;  // copy under the lock
  VT_CHECK(!tt_capture_active(),
           "tenstorrent causal_conv1d_update: state-slot indices changed "
           "during trace capture — decode slots must be stable across a "
           "sequence's steps; the recapture cadence owns a slot change "
           "(reset the graph, then re-warm on the eager boundary step)");
  GdnIdxCacheEntry n;
  n.idx = idxv;
  n.warmed_conv = true;
  std::vector<uint32_t> ids;
  ids.reserve(idxv.size());
  for (int32_t ix : idxv) ids.push_back(static_cast<uint32_t>(ix));
  n.idx_scatter = UploadIdxU32(std::move(ids), ttnn::Shape({static_cast<uint32_t>(idxv.size())}),
                               ttnn::Layout::ROW_MAJOR, device);
  std::vector<uint32_t> rows(static_cast<size_t>(idxv.size()) * C);
  for (size_t i = 0; i < idxv.size(); ++i)
    for (uint32_t c = 0; c < C; ++c)
      rows[i * C + c] = static_cast<uint32_t>(idxv[i]);
  n.idx_rows = UploadIdxU32(std::move(rows), ttnn::Shape({static_cast<uint32_t>(idxv.size()), C}),
                            ttnn::Layout::TILE, device);
  n.zeros_rc = ttnn::zeros(tt::tt_metal::Shape({slots, C}),
                           ttnn::DataType::FLOAT32, ttnn::Layout::ROW_MAJOR,
                           std::ref(device));
  std::vector<float> mv(R, 0.0f), mk(R, 1.0f);
  for (int32_t s : idxv)
    if (s >= 0)
      for (uint32_t c = 0; c < C; ++c) {
        mv[static_cast<size_t>(s) * C + c] = 1.0f;
        mk[static_cast<size_t>(s) * C + c] = 0.0f;
      }
  n.mv = UploadTensor(std::move(mv), ttnn::Shape({1, R}),
                      ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
  n.mk = UploadTensor(std::move(mk), ttnn::Shape({1, R}),
                      ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
  e = std::move(n);
  return e;
}

// Serve the tensor's current device shadow without touching host bytes — the
// capture-pure read (an EnsureHost here would enqueue the readback the trace
// refuses). Returns false when no current shadow exists; the caller then
// takes its eager staging path, and a capture-time miss is refused there.
// `rows`/`cols` are the caller's logical 2D geometry; the shadow must match
// that volume (the recorded commit geometry is the producer's logical shape,
// and a mismatch means a shape the device path has no evidence for).
bool ServeDeviceShadowRaw(const Tensor& t, uint32_t rows, uint32_t cols,
                          ttnn::Tensor& out) {
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(t.data);
  if (s == nullptr || !s->device_current || !s->device.has_value()) return false;
  // Device-authoritative only: the last tracked write must have been a
  // COMMIT (host bytes known-stale), not a plain stage. A both-current slot
  // says nothing about the device bytes matching the buffer's latest value —
  // a raw backend Copy (the op tests' per-step input refresh; any
  // non-tracked producer) bypasses slot tracking, and serving such a shadow
  // reads the previous tenant's bytes.
  if (s->host_current) return false;
  if (static_cast<uint64_t>(s->dev_rows) * s->dev_cols !=
      static_cast<uint64_t>(rows) * cols)
    return false;
  if (s->device->dtype() != ttnn::DataType::FLOAT32 &&
      s->device->dtype() != ttnn::DataType::BFLOAT16)
    return false;  // the device path covers the production activation dtypes
  out = *s->device;
  return true;
}

}  // namespace

// Serve the activation as an f32 TILE device tensor for the decode step: the
// current device shadow when the producer committed one (the production
// driver commits its outputs device-side in every phase — the traffic
// counters prove no per-token round-trip), reshaped/typecast in-region when
// the shadow's split or dtype differs (both pure device ops, exact bytes;
// TILE is what every consumer at this pin demands — a non-TILE shadow is
// converted, a pure copy); else the eager arm's existing host staging, which
// a capture-time arrival refuses by name (an EnsureHost here would enqueue
// exactly the readback the trace forbids).
ttnn::Tensor ServeActF32(const Tensor& t, uint32_t rows, uint32_t cols,
                         const char* what, MeshDevice& device) {
  ttnn::Tensor raw;
  if (ServeDeviceShadowRaw(t, rows, cols, raw)) {
    const auto ls = raw.logical_shape();
    if (ls.rank() != 2 || ls[0] != rows || ls[1] != cols) {
      if (tt_capture_active() && ls.rank() == 2 &&
          ls[0] * ls[1] == static_cast<uint32_t>(rows) * cols) {
        // CaptureSafeReshape: member reshape (pure view, no device program)
        raw = CaptureSafeReshape(raw, ttnn::Shape({rows, cols}));
      } else {
        raw = CaptureSafeReshape(raw, ttnn::Shape({rows, cols}));
      }
    }
    if (raw.dtype() != ttnn::DataType::FLOAT32)
      raw = ttnn::typecast(raw, ttnn::DataType::FLOAT32);
    if (raw.layout() != ttnn::Layout::TILE)
      raw = ttnn::to_layout(raw, ttnn::Layout::TILE);
    return raw;
  }
  // The exact-shadow serve demands owner-numel equality, so an interior
  // window of a packed producer output (the q/k expand row views at a batch
  // the owner padded wider) falls through here. ServeDeviceWindow splits the
  // interior pointer offset into the owner's row/column geometry and rides a
  // device column slice — the same serve ServePostConvAB takes. Without it
  // the host fallback ran EnsureHost on the OWNER slot and downloaded the
  // whole plane against the window's numel (ISSUE-LOCAL-01M2E5F69CMWDERKXG32YY9P8N:
  // EnsureHost 1x16x128 vs dev 1024x128, vllm-bench chunked decode).
  if (ServeDeviceWindow(t, rows, cols, raw)) return raw;
  VT_CHECK(!tt_capture_active(),
           std::string("tenstorrent gdn_decode: ") + what +
               " arrived without a current device shadow during trace "
               "capture — the captured arm reads device-resident "
               "activations only");
  EnsureHost(t);
  std::vector<float> h(static_cast<size_t>(rows) * cols);
  for (uint64_t i = 0; i < static_cast<uint64_t>(rows) * cols; ++i)
    h[static_cast<size_t>(i)] = LoadElemF32(t, static_cast<int64_t>(i));
  return UploadTensor(std::move(h), ttnn::Shape({rows, cols}),
                      ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
}

namespace {

// Both serve paths normalize through here. The ROW_MAJOR round-trip is the
// load-bearing step: a committed shadow can carry any logical shape the
// producer chose, a TILE reshape of such a tensor is not a safe reshape
// input at this pin (the AttnQkNormRopeGate doctrine), and a column slice
// of a TILE owner keeps the owner's tile padding — every one of those feeds
// a downstream elementwise op a misaligned physical layout (the prefill
// "Invalid subtile broadcast type" fatal). Materializing in ROW_MAJOR gives
// a fresh contiguous buffer; the elementwise chain on it is shape-agnostic,
// and the final to_layout(TILE) rebuilds the padding from the tensor's own
// (served) geometry. A rank-2 shadow keeps its native logical shape — the
// callers commit that served geometry (see NormalizeDevF32Tile's comment).
ttnn::Tensor NormalizeDevF32Tile(ttnn::Tensor x, uint32_t rows, uint32_t cols) {
  x = ttnn::to_layout(x, ttnn::Layout::ROW_MAJOR);
  const auto ls = x.logical_shape();
  if (ls.rank() != 2 || ls[0] != rows || ls[1] != cols) {
    // ISSUE-LOCAL-01M2K8WWA0X09VJF55NTS16VTV: a producer shadow natively
    // shaped [T, W] (the APEX gated-norm activation [128, 6144]) must not
    // take the free ttnn::reshape to (1, n) — reshape_rm's statically
    // allocated staging CBs scale with the row width, and the 3,145,728 B
    // f32 row is 2x the L1 cap (fatal at program allocation). The
    // metadata view cannot serve it either: view_device on ROW_MAJOR with
    // a CHANGED last dim rebuilds the buffer page mapping and the data
    // comes back wrong (the focused cast test's bit-exact memcmp caught
    // it) — which is why upstream routes this case through reshape_rm.
    // The elementwise ops below (typecast, to_layout(TILE)) are
    // shape-agnostic on the contiguous buffer, so a rank-2 shadow runs at
    // its native geometry and the CALLER commits the served geometry (the
    // #2282 served-geometry doctrine). The free reshape stays for the
    // non-rank-2 cases the elementwise chain cannot carry as-is.
    if (ls.rank() != 2) x = ttnn::reshape(x, ttnn::Shape({rows, cols}));
  }
  if (x.dtype() != ttnn::DataType::FLOAT32)
    x = ttnn::typecast(x, ttnn::DataType::FLOAT32);
  return ttnn::to_layout(x, ttnn::Layout::TILE);
}

// Serve an interior window of a device-authoritative OWNER slot. araw/braw
// are interior column slices of ONE packed producer output (merged-BA,
// VT_GDN_MERGED_BA default-on since #1168; qwen3_5.cpp ProjectGdnBA hands
// strided row views into the BA matmul result), and the gated-norm gate leg
// can be the z slice of a fused projection. FindSlot resolves an interior
// pointer to the owner slot; the element offset splits into a row and a
// column of the owner's logical geometry, and the window rides a device
// column slice — the same slice form GdnPostConvKernel already runs on
// dev_conv. A stride that does not walk the owner's rows exactly, a window
// past the owner's bounds, or a byte offset that is not a whole number of
// device elements is not served (the caller refuses under capture). Both
// this and the exact ServeDeviceShadowRaw path normalize to an f32 TILE
// [rows, cols] via `normalize` so a served shadow never changes the
// arithmetic the eager host staging defined.
bool ServeDeviceWindow(const Tensor& t, uint32_t rows, uint32_t cols,
                       ttnn::Tensor& out) {
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(t.data);
  if (s == nullptr || !s->device_current || s->host_current || !s->device)
    return false;
  if (s->device->dtype() != ttnn::DataType::FLOAT32 &&
      s->device->dtype() != ttnn::DataType::BFLOAT16)
    return false;
  const int64_t elem_bytes =
      s->device->dtype() == ttnn::DataType::FLOAT32 ? 4 : 2;
  // The window's own geometry: rank-2 views stride [dev_cols, 1] by
  // construction; higher-rank views are accepted only when CONTIGUOUS
  // row-major (every stride the product of the inner extents), which makes
  // them a plain flattened [rows*cols] span — the v/g/beta expands of the
  // decode step arrive as rank-3 [1, bh, ud] views of the packed producer
  // output (the vllm-bench chunked path; ISSUE-LOCAL-01M2E5F69CMWDERKXG32YY9P8N).
  // A non-contiguous multi-rank view is not served.
  const int64_t dev_cols = static_cast<int64_t>(s->dev_cols);
  if (t.rank == 2) {
    if (t.stride[0] != dev_cols || t.stride[1] != 1) return false;
  } else {
    if (t.shape[t.rank - 1] != cols) return false;
    int64_t expect = 1;
    for (int i = t.rank - 1; i >= 0; --i) {
      if (t.shape[i] != 0 && t.stride[i] != expect) return false;
      expect *= t.shape[i];
    }
    if (rows * cols != expect) return false;
  }
  const int64_t delta = static_cast<const char*>(t.data) -
                        static_cast<const char*>(s->host);
  if (delta < 0 || delta % elem_bytes != 0) return false;
  const int64_t off = delta / elem_bytes;
  // The slot's registered allocation must COVER the whole window (the slot
  // map is never erased, #1486 — a freed tensor's range can host a later,
  // unrelated allocation; the vllm-bench prefill buffers hit exactly that).
  if (delta + rows * cols * elem_bytes >
      static_cast<int64_t>(s->bytes))
    return false;
  const int64_t off_row = off / dev_cols, off_col = off % dev_cols;
  if (off_row + rows > s->dev_rows || off_col + cols > dev_cols)
    return false;
  ttnn::Tensor win = ttnn::slice(
      *s->device, ttsl::SmallVector<uint32_t>{static_cast<uint32_t>(off_row),
                                              static_cast<uint32_t>(off_col)},
      ttsl::SmallVector<uint32_t>{static_cast<uint32_t>(off_row + rows),
                                  static_cast<uint32_t>(off_col + cols)},
      ttsl::SmallVector<uint32_t>{1, 1});
  if (std::getenv("VT_TT_SLOT_TRACE") != nullptr) {
    std::fprintf(stderr, "[TT-WINSERVE] serve %ux%u off=%" PRId64
                         " owner=%s\n",
                 rows, cols, off, DevShapeStr(*s->device).c_str());
    std::fflush(stderr);
  }
  out = NormalizeDevF32Tile(std::move(win), rows, cols);
  return true;
}

}  // namespace

// Serve the kGdnPostConv a/b operand: exact shadow first, then the packed
// owner window. Returns false when neither exists (the caller then stages
// from host in the eager arm and refuses under capture).
bool ServePostConvAB(const Tensor& t, uint32_t rows, uint32_t cols,
                     ttnn::Tensor& out) {
  ttnn::Tensor raw;
  if (ServeDeviceShadowRaw(t, rows, cols, raw)) {
    out = NormalizeDevF32Tile(std::move(raw), rows, cols);
    return true;
  }
  return ServeDeviceWindow(t, rows, cols, out);
}

ttnn::Tensor CachedTile(const void* owner, uint64_t g0, uint64_t g1, uint64_t g2,
                        const std::function<std::vector<float>()>& build,
                        const ttnn::Shape& shape, MeshDevice& device) {
  const std::array<uint64_t, 4> key{reinterpret_cast<uintptr_t>(owner), g0, g1, g2};
  std::lock_guard<std::mutex> g(ConvTileCacheMutex());
  auto it = ConvTileCacheMap().find(key);
  if (it != ConvTileCacheMap().end()) {
    // The content identity below is the point of `host` — the key is the
    // OWNER POINTER, and a test allocator hands the next case the same
    // address with different weight bytes (the rope cos/sin discipline).
    // Rebuild the host tile every call (the eager arm always did) and
    // re-upload only when the bytes moved; in-region they are the model
    // constants the warm step baked — match, no upload, no PCIe.
    std::vector<float> now = build();
    if (now == it->second.host) return it->second.dev;
    VT_CHECK(!tt_capture_active(),
             "tenstorrent constant tile: content changed during trace "
             "capture — the warm step baked different bytes than the "
             "captured region reads");
    it->second.host = std::move(now);
    it->second.dev = UploadTensor(it->second.host, shape,
                                  ttnn::DataType::FLOAT32, ttnn::Layout::TILE,
                                  device);
    return it->second.dev;
  }
  VT_CHECK(!tt_capture_active(),
           "tenstorrent constant tile not warmed — the cold step must build "
           "it before the captured region");
  ConvTileEntry e;
  e.host = build();
  e.dev = UploadTensor(e.host, shape, ttnn::DataType::FLOAT32,
                       ttnn::Layout::TILE, device);
  ConvTileCacheMap().emplace(key, std::move(e));
  return ConvTileCacheMap()[key].dev;
}

namespace {

// The per-(token,head) repeat index for the rope cos|sin expansion:
// rows[(i*H + h)*half + j] = i. Content is a pure function of the key, so
// no host bytes are kept; the key's (0, 3, ...) prefix cannot collide with
// the scratch rows ((0, 1/2, ...)) or the head map (batch >= 1).
ttnn::Tensor CachedRepeatIdx(uint64_t t, uint64_t heads, uint64_t half,
                             MeshDevice& device) {
  const std::array<uint64_t, 4> key{0, 3, t * heads, half};
  std::lock_guard<std::mutex> g(ConvTileCacheMutex());
  auto it = ConvTileCacheMap().find(key);
  if (it != ConvTileCacheMap().end()) return it->second.dev;
  VT_CHECK(!tt_capture_active(),
           "tenstorrent attn rope: repeat index not warmed — the cold step "
           "must build it before the captured region");
  const uint32_t rows_n = static_cast<uint32_t>(t * heads);
  std::vector<uint32_t> rows(static_cast<size_t>(rows_n) * half);
  for (uint64_t r = 0; r < t * heads; ++r)
    for (uint64_t j = 0; j < half; ++j)
      rows[static_cast<size_t>(r) * half + j] =
          static_cast<uint32_t>(r / heads);
  ConvTileEntry e;
  e.dev = UploadIdxU32(
      std::move(rows),
      ttnn::Shape({rows_n, static_cast<uint32_t>(half)}),
      ttnn::Layout::TILE, device);
  ConvTileCacheMap().emplace(key, std::move(e));
  return ConvTileCacheMap()[key].dev;
}

namespace {

ttnn::Tensor CachedAccBase(uint32_t R, MeshDevice& device) {
  const std::array<uint64_t, 4> key{0, 2, R, 0};
  std::lock_guard<std::mutex> g(ConvTileCacheMutex());
  auto it = ConvTileCacheMap().find(key);
  if (it != ConvTileCacheMap().end()) return it->second.dev;
  VT_CHECK(!tt_capture_active(),
           "tenstorrent causal_conv1d_update: accumulation base not warmed — "
           "the cold step must build it before the captured region");
  ConvTileEntry e;
  e.dev = ttnn::zeros(tt::tt_metal::Shape({1, R}), ttnn::DataType::FLOAT32,
                      ttnn::Layout::TILE, std::ref(device));
  ConvTileCacheMap().emplace(key, std::move(e));
  return ConvTileCacheMap()[key].dev;
}

}  // namespace

}  // namespace

// Read the tiny i32 index metadata on the host (NOT state traffic) and refuse
// out-of-range slots by name — the CPU oracle dereferences blindly, the TT
// kernel owns a one-hot and must not silently read a wrong row. The decode
// ops allow idx<0 (the NULL slot), gather/scatter refuse it like the oracle.
std::vector<int32_t> ReadIdxHost(const Tensor& idx, int64_t slots, const char* op,
                                 bool allow_null) {
  EnsureHost(idx);
  const int32_t* p = idx.Ptr<int32_t>();
  std::vector<int32_t> v(p, p + idx.shape[0]);
  for (int32_t ix : v)
    VT_CHECK(ix < slots && (allow_null || ix >= 0),
             std::string("tenstorrent ") + op +
                 ": state index out of range");
  return v;
}

namespace {

// ---- Conv-state transposed shadow (BACKEND-TENSTORRENT-GDN W2) --------------
// ttnn movement exactness at this pin is dim-0 only (W2 micro-probe evidence):
// slice/concat/gather/indexed_fill on ROWS are byte-exact, while any
// last-dim (width) slice/concat at sub-tile offsets is broken or tf32-rounds.
// The conv roll shifts the window along the width axis, so the shadow is
// stored TRANSPOSED: [sl+1, slots*C] time-major (row j = timestep j of every
// (slot, channel); row sl = scratch for the incoming x). Every per-step op is
// then row-based and byte-exact: x lands via indexed_fill, the window+scratch
// come out via one gather with CONSTANT indices, the MAC is exact multiply +
// sequential single-row adds (bit-identical to the CPU oracle's accumulation
// order), and the roll is one gather whose index is the same rotation every
// step (rows [1..width-1, scratch, tail..., scratch]).
//
// Constant gather indices are cached per geometry; uploads cross PCIe only
// for x / w / bias / tiny per-step masks — never the state.

struct ConvGatherIdxCache {
  std::mutex mu;
  // key: (kind, R = slots*C, width, sl) -> u32 index tensor [rows, R] TILE.
  // `kind` separates the mac and roll lists — they share every geometry
  // field whenever state_len == width (rows == width+1 == sl+1), and the
  // roll list differs from the mac list exactly then.
  std::map<std::array<uint32_t, 4>, ttnn::Tensor> cache;
};

ConvGatherIdxCache& ConvGatherIdxCacheInstance() {
  static ConvGatherIdxCache* inst = new ConvGatherIdxCache(); // never destroyed (#1486)
  return *inst;
}

ttnn::Tensor ConvGatherIdx(const std::vector<uint32_t>& rows_of_src, uint32_t R,
                           uint32_t width, uint32_t sl, MeshDevice& device,
                           uint32_t kind) {
  ConvGatherIdxCache& c = ConvGatherIdxCacheInstance();
  std::lock_guard<std::mutex> g(c.mu);
  const std::array<uint32_t, 4> key{kind, R, width, sl};
  auto it = c.cache.find(key);
  if (it != c.cache.end()) return it->second;
  // The captured region serves the entry the cold step warmed; a miss
  // in-region would upload (enqueue_write) and the trace refuses it.
  VT_CHECK(!tt_capture_active(),
           "tenstorrent: conv gather-index cache miss during capture — the "
           "cold step must warm the constant index first");
  // The full [rows, R] broadcast index gather demands (same as
  // GatherRowsExact); both callers pass the list that is constant for
  // (kind, geometry), so first-writer-wins stores the right tensor.
  std::vector<uint32_t> flat(rows_of_src.size() * R);
  for (size_t r = 0; r < rows_of_src.size(); ++r)
    for (uint32_t j = 0; j < R; ++j)
      flat[r * R + j] = rows_of_src[r];
  ttnn::Tensor idx = UploadIdxU32(
      std::move(flat), ttnn::Shape({static_cast<uint32_t>(rows_of_src.size()), R}),
      ttnn::Layout::TILE, device);
  // First-caller-wins under the same geometry key: both callers pass the
  // constant list for that geometry, so the stored tensor is correct.
  if (!c.cache.emplace(key, idx).second) {
    idx = c.cache[key];
  }
  return idx;
}

// Ensure the conv-state shadow in TRANSPOSED [sl+1, R] TILE f32 form. Uploads
// (counted) only when missing/stale/wrong geometry. The scratch row is built
// on device (zeros + row concat — exact), so exactly `state bytes` cross h2d.
ttnn::Tensor EnsureConvStateTransposed(const Tensor& t, uint32_t slots,
                                       uint32_t Cc, uint32_t sl,
                                       MeshDevice& device) {
  const uint32_t R = slots * Cc;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(t.data);
    if (s != nullptr && s->device_current && s->device.has_value() &&
        s->conv_transposed && s->conv_slots == slots && s->conv_c == Cc &&
        s->conv_sl == sl && s->device->dtype() == ttnn::DataType::FLOAT32 &&
        s->device->layout() == ttnn::Layout::TILE &&
        s->device->logical_shape()[0] == sl + 1 && s->device->logical_shape()[1] == R) {
      return *s->device;
    }
  }
  // (Reachable only on the cold arm — the fast path above serves the warmed
  // shadow in-region. A capture-time miss must refuse BEFORE the EnsureHost:
  // across a prefill role transition the slot is device-current host-stale,
  // and the download would be the trace-refused read that aborts the process
  // instead of naming the condition. The driver's ConvShadowServeable query
  // keeps captures off this path; this CHECK is the backstop.)
  VT_CHECK(!tt_capture_active(),
           "tenstorrent: conv-state transposed shadow not warmed during "
           "capture — the cold step must establish it first");
  EnsureHost(const_cast<Tensor&>(t));  // host truth (oracle [slots, C, sl] order)
  const int64_t numel = static_cast<int64_t>(slots) * Cc * sl;
  VT_CHECK(t.Numel() == numel, "tenstorrent: conv state numel mismatch");
  std::vector<float> host(static_cast<size_t>(numel));
  for (int64_t i = 0; i < numel; ++i) host[static_cast<size_t>(i)] = LoadElemF32(t, i);
  std::vector<float> trans(static_cast<size_t>(numel));
  for (uint32_t s = 0; s < slots; ++s)
    for (uint32_t c = 0; c < Cc; ++c)
      for (uint32_t j = 0; j < sl; ++j)
        trans[static_cast<size_t>(j) * R + s * Cc + c] =
            host[static_cast<size_t>((s * Cc + c) * sl + j)];
  GdnStateH2dBytes().fetch_add(static_cast<uint64_t>(numel) * sizeof(float),
                               std::memory_order_relaxed);
  ttnn::Tensor body = UploadTensor(
      std::move(trans), ttnn::Shape({sl, R}), ttnn::DataType::FLOAT32,
      ttnn::Layout::TILE, device);
  ttnn::Tensor scratch = ttnn::zeros(tt::tt_metal::Shape({1, R}),
                                     ttnn::DataType::FLOAT32, ttnn::Layout::TILE,
                                     std::ref(device));
  ttnn::Tensor dev = ttnn::concat(std::vector<ttnn::Tensor>{body, scratch}, 0);
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(t.data);
    if (s != nullptr) {
      s->device = dev;
      s->dev_rows = sl + 1;
      s->dev_cols = R;
      s->device_current = true;
      s->host_current = true;
      s->device_reserved = false;  // real bytes staged — reservation spent
      s->conv_transposed = true;
      s->conv_slots = slots;
      s->conv_c = Cc;
      s->conv_sl = sl;
    }
  }
  return dev;
}

// Publish a new transposed shadow as the conv state's current value.
void CommitConvTransposed(Tensor& state, ttnn::Tensor dev, uint32_t slots,
                          uint32_t Cc, uint32_t sl) {
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(state.data);
  if (s == nullptr) {
    // Untracked buffer: materialize the transposition on host directly.
    std::vector<float> v = dev.to_vector<float>();
    // W3 #2201: the readback is a real device→host download of the whole
    // [sl+1, R] shadow — count it like every other GDN state download.
    GdnStateD2hBytes().fetch_add(static_cast<uint64_t>(v.size()) * sizeof(float),
                                 std::memory_order_relaxed);
    const uint32_t R = slots * Cc;
    for (uint32_t sc = 0; sc < slots * Cc; ++sc)
      for (uint32_t j = 0; j < sl; ++j)
        StoreElemF32(state, static_cast<int64_t>(sc) * sl + j,
                     v[static_cast<size_t>(j) * R + sc]);
    return;
  }
  s->device = std::move(dev);
  s->dev_rows = sl + 1;
  s->dev_cols = slots * Cc;
  s->device_current = true;
  s->host_current = false;
  s->conv_transposed = true;
  s->conv_slots = slots;
  s->conv_c = Cc;
  s->conv_sl = sl;
  s->device_reserved = false;  // real bytes committed — the reservation is spent
}

// kCausalConv1dUpdate (§3, seqlen==1 read-old-then-roll; cpu_ops.cpp
// CausalConv1dUpdateKernel at :1341). Transposed-shadow device path: every op
// is a proven-exact primitive (micro-probe table in the W2 evidence): x is
// written into the scratch row by indexed_fill, the [window | x] MAC source
// is one constant-index gather, the MAC multiplies exactly and accumulates by
// sequential single-row adds in the ORACLE's order (bias, taps 0..width-1,
// x tap last) so out is bit-identical up to silu, and the roll is one
// constant-index gather (shift rows, scratch in at width-1, tail identity).
// The indexed form masks with exact 0/1 multiplies: NULL rows (idx<0) keep
// both the out row's previous bytes and the cache row untouched (ops.h NULL
// skip), and non-batch cache columns never move. The state never crosses
// PCIe per step: only x / w / bias / the tiny 0/1 masks upload.
void CausalConv1dUpdateKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& w,
                              const Tensor* bias, Tensor& conv_state,
                              const Tensor* conv_state_indices,
                              const CausalConv1dArgs& args) {
  TT_OP_TRACE("CausalConv1dUpdate");
  GdnDecodeSteps().fetch_add(1, std::memory_order_relaxed);
  const int64_t batch = x.shape[0], c_dim = x.shape[1], k = w.shape[1];
  const int64_t width = k - 1;
  const int64_t state_len = conv_state.shape[2];
  const int64_t slots = conv_state.shape[0];
  const int64_t x_rs = x.stride[0];
  if (batch == 0) return;  // empty batch: no-op, cache as-is
  std::vector<int32_t> idxv;
  const bool indexed = conv_state_indices != nullptr;
  if (indexed) {
    idxv = ReadIdxHost(*conv_state_indices, slots, "causal_conv1d_update", true);
  } else {
    idxv.resize(static_cast<size_t>(batch));
    for (int64_t b = 0; b < batch; ++b) idxv[static_cast<size_t>(b)] = static_cast<int32_t>(b);
  }

  bool has_null = false;
  for (int32_t s : idxv)
    if (s < 0) has_null = true;
  // The captured arm is the pure-decode indexed path: every batch row names a
  // live slot, so the NULL-row arms (keep the previous out bytes, skip the
  // state row) never engage in-region. NULL handling stays on the eager path
  // (pinned by the op-level oracle suite); a capture with a NULL row is a
  // refused shape, not a silent skip.
  VT_CHECK(!has_null || !tt_capture_active(),
           "tenstorrent causal_conv1d_update: NULL state rows during trace "
           "capture — the captured arm is the pure-decode indexed path");

  MeshDevice& device = SharedMeshDevice();
  const uint32_t ub = static_cast<uint32_t>(batch), uc = static_cast<uint32_t>(c_dim),
                  uw = static_cast<uint32_t>(width), usl = static_cast<uint32_t>(state_len),
                  uslots = static_cast<uint32_t>(slots), R = uslots * uc,
                  utaps = uw + 1;

  // Weights are immutable across steps: the [taps, R] tap tile and the [1, R]
  // bias tile are step-constant caches built once on the cold step and served
  // in-region (a capture-time upload is an enqueue_write the trace refuses).
  // The host weight reads run only inside the cold-step builders.
  ttnn::Tensor Wt = CachedTile(
      w.data, utaps, R, 0,
      [&w, utaps, R, uslots, uc, k] {
        EnsureHost(w);
        std::vector<float> wh(static_cast<size_t>(utaps) * R);
        for (uint32_t j = 0; j < utaps; ++j)
          for (uint32_t s = 0; s < uslots; ++s)
            for (uint32_t c = 0; c < uc; ++c)
              wh[static_cast<size_t>(j) * R + s * uc + c] =
                  LoadElemF32(w, static_cast<int64_t>(c) * k + j);
        return wh;
      },
      ttnn::Shape({utaps, R}), device);
  ttnn::Tensor biasT;
  if (bias != nullptr) {
    biasT = CachedTile(
        bias->data, R, 0, 0,
        [bias, R, uslots, uc] {
          EnsureHost(*bias);
          std::vector<float> bh(static_cast<size_t>(R));
          for (uint32_t s = 0; s < uslots; ++s)
            for (uint32_t c = 0; c < uc; ++c)
              bh[static_cast<size_t>(s) * uc + c] = LoadElemF32(*bias, c);
          return bh;
        },
        ttnn::Shape({1, R}), device);
  }

  ttnn::Tensor T = EnsureConvStateTransposed(conv_state, uslots, uc, usl, device);

  // x into the scratch row sl: indexed_fill (exact row write). Non-batch
  // columns are zero — they are masked out of the roll and never read.
  // Device arm: the producer commits x device-side, so the shadow serves
  // in-region and the scatter is one dim-0 indexed_fill into the warmed zeros
  // base (duplicates last-writer-wins, exactly the host loop). The eager
  // fallback covers shapes with no current shadow (first use, op-level tests).
  ttnn::Tensor xT;
  GdnIdxCacheEntry ice;
  const bool ice_ok = indexed && !has_null;
  if (ice_ok) ice = GdnConvIdxEntry(idxv, uslots, uc, device);
  {
    ttnn::Tensor x_raw;
    if (ServeDeviceShadowRaw(x, ub, uc, x_raw)) {
      ttnn::Tensor x_f32 = x_raw.dtype() == ttnn::DataType::FLOAT32
                               ? x_raw
                               : ttnn::typecast(x_raw, ttnn::DataType::FLOAT32);
      if (ice_ok) {
        // The rank-4 indexed_fill rides on ROW_MAJOR copies with the LAST
        // dim preserved — the exact ScatterRowsExact discipline. A rank-4
        // view on a TILE tensor re-pads the trailing 1-dims (the W0-sweep
        // x1024 physical blowup), and a view's span check fires on the
        // producer's tight slice buffers — found by the first driver run of
        // this path (MeshBuffer fatal from CausalConv1dUpdateKernel).
        auto to_rm = [](ttnn::Tensor t) {
          return t.layout() == ttnn::Layout::ROW_MAJOR
                     ? t
                     : ttnn::to_layout(t, ttnn::Layout::ROW_MAJOR);
        };
        ttnn::Tensor filled = ttnn::indexed_fill(
            ice.idx_scatter,
            ttnn::experimental::view(to_rm(ice.zeros_rc),
                                     ttnn::Shape({uslots, 1, 1, uc})),
            ttnn::experimental::view(to_rm(x_f32),
                                     ttnn::Shape({ub, 1, 1, uc})),
            std::nullopt, /*dim=*/0);
        xT = ttnn::to_layout(ttnn::reshape(std::move(filled),
                                           ttnn::Shape({1, R})),
                             ttnn::Layout::TILE);
      } else {
        // Non-indexed (compact) form: the batch rows ARE the cache rows.
        xT = ttnn::reshape(x_f32, ttnn::Shape({1, R}));
      }
    } else {
      VT_CHECK(!tt_capture_active(),
               "tenstorrent causal_conv1d_update: x arrived without a current "
               "device shadow during trace capture — the producer must commit "
               "device-side before the captured region");
      EnsureHost(x);
      std::vector<float> xrow(static_cast<size_t>(R), 0.0f);
      for (int64_t b = 0; b < batch; ++b) {
        const int32_t s = idxv[static_cast<size_t>(b)];
        if (s < 0) continue;  // NULL row: no x lands anywhere
        for (int64_t c = 0; c < c_dim; ++c)
          xrow[static_cast<size_t>(static_cast<int64_t>(s) * c_dim + c)] =
              LoadElemF32(x, b * x_rs + c);
      }
      xT = UploadTensor(std::move(xrow), ttnn::Shape({1, R}),
                        ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
    }
  }
  // The scratch row is the LAST state row (index usl), and the fill never
  // reads its old content — "write x's row into the scratch row" is therefore
  // exactly concat(history rows, xT), rank-2 TILE end to end. The rank-4
  // indexed_fill this replaces was the W4d 27B wall twice over: a rank-4 view
  // on the TILE state re-pads the trailing 1-dims to 32x32 per element (the
  // x1024 blowup — ~1.6 GB per decode step at R = 102400), and the ROW_MAJOR
  // rescue dies inside tt-metal's all-RM indexed_fill at op scale (conv
  // oracle max_abs 0.17 under every view/reshape variant, against a bit-clean
  // baseline). concat touches only {usl+1, R} planes — 13 MB at 27B — and
  // reinterprets nothing.
  if (xT.dtype() != T.dtype()) xT = ttnn::typecast(std::move(xT), T.dtype());
  T = usl == 0
          ? xT
          : ttnn::concat(
                std::vector<ttnn::Tensor>{
                    ttnn::slice(T, ttsl::SmallVector<uint32_t>{0u, 0u},
                                ttsl::SmallVector<uint32_t>{
                                    static_cast<uint32_t>(usl),
                                    static_cast<uint32_t>(R)},
                                ttsl::SmallVector<uint32_t>{1u, 1u}),
                    xT},
                /*dim=*/0);

  // MAC source: rows [0..width-1, scratch sl] — constant per geometry.
  std::vector<uint32_t> mac_rows;
  mac_rows.reserve(utaps);
  for (uint32_t j = 0; j < uw; ++j) mac_rows.push_back(j);
  mac_rows.push_back(usl);
  ttnn::Tensor Msrc = ttnn::gather(
      T, /*dim=*/0, ConvGatherIdx(mac_rows, R, uw, usl, device, /*kind=*/0),
      /*sparse_grad=*/false, std::nullopt);

  // Tap weights as [taps, R] (row j, column s*C+c = w[c, j]) — the step-
  // constant Wt tile warmed above; no per-step build or upload.
  ttnn::Tensor prod = ttnn::multiply(Msrc, Wt);  // exact (one ulp)

  // Sequential accumulation in the oracle's order: bias first, taps 0..width.
  // Both bases are step-constant caches (bias tile / warmed zeros).
  ttnn::Tensor acc = bias != nullptr ? biasT : CachedAccBase(R, device);
  for (uint32_t j = 0; j < utaps; ++j) {
    acc = ttnn::add(
        acc, ttnn::slice(prod, ttsl::SmallVector<uint32_t>{j, 0},
                         ttsl::SmallVector<uint32_t>{j + 1, R},
                         ttsl::SmallVector<uint32_t>{1, 1}));
  }
  ttnn::Tensor y = args.silu_activation ? ttnn::silu(acc) : acc;

  if (indexed && !has_null) {
    // Every batch row names a live slot: select each out row straight from
    // silu(y) viewed [slots, C] by the warmed broadcast row ids (one exact
    // dim-0 gather — the index's last dim spans the input's, the
    // ConvGatherIdx shape rule). No previous-bytes read: with no NULL rows
    // the oracle's prev branch never engages, so the fresh out buffer needs
    // nothing staged and the whole select stays device-side in-region.
    ttnn::Tensor yv = ttnn::reshape(y, ttnn::Shape({uslots, uc}));
    ttnn::Tensor ysel = ttnn::gather(yv, /*dim=*/0, ice.idx_rows,
                                     /*sparse_grad=*/false, std::nullopt);
    CommitDeviceLogical2D(out, std::move(ysel), ub, uc);
  } else if (indexed) {
    // NULL rows (idx<0) keep the out row's PREVIOUS bytes (the oracle skips
    // them entirely). Select each out element from one two-block row source:
    // block 0 = silu(y)^T [R, 1], block 1 = prev^T [R, 1] (prev[b, c] stored
    // at row b*C+c, independent of the slot map — no aliasing with named
    // slots). One per-step gather index; never cached (idx varies).
    EnsureHost(out);
    std::vector<float> prev(static_cast<size_t>(R), 0.0f);
    for (int64_t b = 0; b < batch; ++b)
      for (int64_t c = 0; c < c_dim; ++c)
        prev[static_cast<size_t>(b * c_dim + c)] = LoadElemF32(out, b * c_dim + c);
    ttnn::Tensor prevT = UploadTensor(std::move(prev), ttnn::Shape({1, R}),
                                      ttnn::DataType::FLOAT32,
                                      ttnn::Layout::TILE, device);
    ttnn::Tensor src = ttnn::concat(
        std::vector<ttnn::Tensor>{ttnn::transpose(y, 0, 1),
                                  ttnn::transpose(prevT, 0, 1)},
        0);  // [2R, 1]
    std::vector<uint32_t> orows(static_cast<size_t>(ub) * uc);
    for (int64_t b = 0; b < batch; ++b)
      for (int64_t c = 0; c < c_dim; ++c) {
        const int32_t s = idxv[static_cast<size_t>(b)];
        orows[static_cast<size_t>(b * c_dim + c)] =
            s >= 0 ? static_cast<uint32_t>(s) * uc + static_cast<uint32_t>(c)
                   : R + static_cast<uint32_t>(b * c_dim + c);
      }
    ttnn::Tensor oidx = UploadIdxU32(
        std::move(orows), ttnn::Shape({ub * uc, 1}), ttnn::Layout::TILE, device);
    ttnn::Tensor ysel = ttnn::gather(src, /*dim=*/0, oidx,
                                     /*sparse_grad=*/false, std::nullopt);
    ttnn::Tensor yout = ttnn::reshape(ysel, ttnn::Shape({ub, uc}));
    CommitDeviceLogical2D(out, std::move(yout), ub, uc);
  } else {
    // Compact form: acc's columns ARE the [B, C] out order.
    CommitDeviceLogical2D(out, ttnn::reshape(y, ttnn::Shape({ub, uc})), ub, uc);
  }

  // Roll: one constant-index gather — rows [1..width-1], scratch in at
  // width-1, tail rows identity, scratch row stays last.
  std::vector<uint32_t> roll_rows;
  roll_rows.reserve(usl + 1);
  for (uint32_t j = 1; j < uw; ++j) roll_rows.push_back(j);
  roll_rows.push_back(usl);
  for (uint32_t j = uw; j < usl; ++j) roll_rows.push_back(j);
  roll_rows.push_back(usl);
  ttnn::Tensor rolled = ttnn::gather(
      T, /*dim=*/0, ConvGatherIdx(roll_rows, R, uw, usl, device, /*kind=*/1),
      /*sparse_grad=*/false, std::nullopt);
  if (indexed && !has_null) {
    // Non-batch columns must not move (oracle leaves their rows untouched):
    // exact 0/1 blend back to the pre-step shadow — the warmed masks, no
    // per-step build or upload.
    rolled = ttnn::add(ttnn::multiply(rolled, ice.mv),
                        ttnn::multiply(T, ice.mk));
  } else if (indexed) {
    // NULL rows also pin their slot's row: exact 0/1 blend, built inline
    // (eager-only shape — NULL rows never capture).
    std::vector<float> mv(static_cast<size_t>(R), 0.0f),
        mk(static_cast<size_t>(R), 1.0f);
    for (int32_t s : idxv)
      if (s >= 0)
        for (uint32_t c = 0; c < uc; ++c) {
          mv[static_cast<size_t>(static_cast<int64_t>(s) * uc + c)] = 1.0f;
          mk[static_cast<size_t>(static_cast<int64_t>(s) * uc + c)] = 0.0f;
        }
    auto up1 = [&](std::vector<float>&& v) {
      return UploadTensor(std::move(v), ttnn::Shape({1, R}),
                          ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
    };
    rolled = ttnn::add(ttnn::multiply(rolled, up1(std::move(mv))),
                        ttnn::multiply(T, up1(std::move(mk))));
  }
  // bf16 STORAGE semantics (SupportsCompressedConvState, cuda_backend.cu:119):
  // a bf16 conv_state is "read/written in f32 registers" — every value that
  // enters the cache rounds through bf16 at the store boundary, so the next
  // step's window sees the STORED bits, not the unrounded f32 tap. The device
  // shadow is f32, so honor that boundary here: round the committed shadow
  // through bf16 on device (RNE, zero PCIe — the shadow stays resident). With
  // bf16-representable inputs (the production activation dtype) this is a
  // no-op; with f32-mantissa taps it is what keeps TT on the CUDA contract —
  // pinned by the bf16-state arm in tests/vt/test_tenstorrent_backend.cpp
  // (steps 1+: out max_rel 2-6.5% without this round).
  if (conv_state.dtype == DType::kBF16) {
    rolled = ttnn::typecast(ttnn::typecast(rolled, ttnn::DataType::BFLOAT16),
                            ttnn::DataType::FLOAT32);
  }
  CommitConvTransposed(conv_state, std::move(rolled), uslots, uc, usl);
}

struct Registrar {
  Registrar() {
    if (!DeviceAvailable()) return;
    RegisterOp(OpId::kMatmul, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulKernel)));
    RegisterOp(OpId::kMatmulBT, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulBTKernel)));
    RegisterOp(OpId::kAdd, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<AddFn>(&AddKernel)));
    RegisterOp(OpId::kRelu, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<ReluFn>(&ReluKernel)));
    RegisterOp(OpId::kEmbedding, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<EmbeddingFn>(&EmbeddingKernel)));
    RegisterOp(OpId::kKeepQuantDecode, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<KeepQuantDecodeFn>(&KeepQuantDecodeKernel)));
    RegisterOp(OpId::kMatmulBTQuant, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulBTQuantKernel)));
    RegisterOp(OpId::kMatmulBTQuantGrouped, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<MatmulBTQuantGroupedFn>(
                   &MatmulBTQuantGroupedKernel)));
    RegisterOp(OpId::kLayerNorm, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<LayerNormFn>(&LayerNormKernel)));
    RegisterOp(OpId::kRmsNorm, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<RmsNormFn>(&RmsNormKernel)));
    RegisterOp(OpId::kSiluAndMul, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<SiluAndMulFn>(&SiluAndMulKernel)));
    RegisterOp(OpId::kMoeSiluMul, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<MoeSiluMulFn>(&MoeSiluMulKernel)));
    RegisterOp(OpId::kCastBf16, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<CastBf16Fn>(&CastBf16Kernel)));
    RegisterOp(OpId::kCastF32, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<CastF32Fn>(&CastF32Kernel)));
    RegisterOp(OpId::kRopeNeox, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<RopeFn>(&RopeNeoxKernel)));
    RegisterOp(OpId::kRopeCosSinCache, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<RopeCosSinCacheFn>(&RopeCosSinCacheKernel)));
    RegisterOp(OpId::kRopeFromCache, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<RopeFromCacheFn>(&RopeFromCacheKernel)));
    RegisterOp(OpId::kQkvSplit, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<QkvSplitFn>(&QkvSplitKernel)));
    RegisterOp(OpId::kReshapeAndCache, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<ReshapeAndCacheFn>(&ReshapeAndCacheKernel)));
    RegisterOp(OpId::kPagedAttention, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<PagedAttentionFn>(&PagedAttentionKernel)));
    RegisterOp(OpId::kGreedyArgmax, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<GreedyArgmaxFn>(&GreedyArgmaxKernel)));
    RegisterOp(OpId::kFusedChain, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<FusedChainFn>(&FusedChainKernel)));
    RegisterOp(OpId::kL2Norm, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<L2NormFn>(&L2NormKernel)));
    RegisterOp(OpId::kRmsNormGated, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<RmsNormGatedFn>(&RmsNormGatedKernel)));
    RegisterOp(OpId::kCausalConv1dFwd, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<CausalConv1dFwdFn>(&CausalConv1dFwdKernel)));
  RegisterOp(OpId::kGdnPrefill, DeviceType::kTENSTORRENT,
             reinterpret_cast<void*>(static_cast<GdnPrefillFn>(&GdnPrefillKernel)));
    RegisterOp(OpId::kCausalConv1dUpdate, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<CausalConv1dUpdateFn>(&CausalConv1dUpdateKernel)));
    RegisterOp(OpId::kGdnDecode, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<GdnDecodeFn>(&GdnDecodeKernel)));
    RegisterOp(OpId::kGdnStateGather, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<GdnStateGatherFn>(&GdnStateGatherKernel)));
    RegisterOp(OpId::kGdnStateScatter, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(
                   static_cast<GdnStateScatterFn>(&GdnStateScatterKernel)));
    RegisterOp(OpId::kGdnPostConv, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<GdnPostConvFn>(&GdnPostConvKernel)));
    RegisterOp(OpId::kSigmoidGateBf16, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<SigmoidGateBf16Fn>(&SigmoidGateBf16Kernel)));
    RegisterOp(OpId::kAttnQkNormRopeGate, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(
                   static_cast<AttnQkNormRopeGateFn>(&AttnQkNormRopeGateKernel)));
  }
} registrar;

}  // namespace

std::atomic<int64_t>& LastTraceBytes() {
  static std::atomic<int64_t> v{0};
  return v;
}
int64_t LastTraceBytesForTest() {
  return LastTraceBytes().load(std::memory_order_relaxed);
}

// W4a wave-3b-1 residency-policy probes — the contract lives in
// tenstorrent_device.h. Each reads its map under its own mutex; a nullptr or
// absent host reads false. Defined OUTSIDE the anonymous namespace (the
// wave-3a hook pattern): a -Werror=unused-function TU would reject an
// anonymous-namespace helper no other file-local code calls.
bool EmbedTableShadowPresentForTest(const void* host) {
  if (host == nullptr) return false;
  std::lock_guard<std::mutex> g(EmbedTableMutex());
  return EmbedTableShadows().find(reinterpret_cast<uintptr_t>(host)) !=
         EmbedTableShadows().end();
}

// ---- Called from TenstorrentBackend::Alloc/Free/Copy (no ttnn in that TU). ----

void RegisterHostBuffer(void* host, size_t bytes) {
  if (host == nullptr) return;
  std::lock_guard<std::mutex> g(SlotMutex());
  if (std::getenv("VT_TT_SLOT_TRACE") != nullptr)
    std::fprintf(stderr, "[TT-SLOT] register %p bytes=%zu\n", host, bytes);
  BufferSlot s;
  s.host = host;
  s.bytes = bytes;
  s.host_current = true;
  s.device_current = false;
  Slots()[reinterpret_cast<uintptr_t>(host)] = std::move(s);
}

void UnregisterHostBuffer(void* host) {
  if (host == nullptr) return;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    if (std::getenv("VT_TT_SLOT_TRACE") != nullptr)
      std::fprintf(stderr, "[TT-SLOT] unregister %p\n", host);
    Slots().erase(reinterpret_cast<uintptr_t>(host));
  }
  DropPagedKvShadow(host);
  DropEmbedTableShadow(host);
  DropDecodedWeightShadow(host);
  DropKeepQuantWordShadow(host);
  DropGroupedActShadow(host);
}

void MarkHostWritten(void* host) {
  if (host == nullptr) return;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(host);
    if (s != nullptr) {
      s->host_current = true;
      s->device_current = false;
      s->device = std::nullopt;
      s->conv_transposed = false;
      s->device_reserved = false;  // real bytes now — the upload contract applies
    }
  }
  // Weight tables may be rewritten in place during load — drop embed cache.
  DropEmbedTableShadow(host);
  // ... and the PACKED word shadow: a weight re-staged in place must not keep
  // serving the pre-rewrite words (same in-place-load hazard as the embed twin).
  DropKeepQuantWordShadow(host);
}

void MarkScratchAcquired(void* host) {
  if (host == nullptr) return;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(host);
    if (s != nullptr) {
      // The state MarkHostWritten registers, plus the W7 reservation: the
      // bytes on BOTH sides belong to the previous tenant. host_current stays
      // true so host reads behave exactly as before (stale bytes served);
      // only the first device stage is gated.
      s->host_current = true;
      s->device_current = false;
      s->device = std::nullopt;
      s->conv_transposed = false;
      s->device_reserved = true;
    }
  }
  // Same drop MarkHostWritten performed for every caller — a pool block must
  // never serve a stale embed-table staging either.
  DropEmbedTableShadow(host);
}

// TRUSTED dump (BACKEND-TENSTORRENT-QWEN35 W2c measurement reset): whole
// tensors only, typed header, DUAL-READ verified. Two independent
// Synchronize+Copy passes must agree byte-for-byte or the file records
// verified=0 (a mismatch means residency state changed under us and the
// payload must not be trusted). File layout: magic 'TDMP', dtype u32,
// rank u32, dims[4] u32, numel u64, verified u8, payload.
void TrustDump(Queue& q, const char* dir, const char* name, const Tensor& t) {
  static std::atomic<uint64_t> seq{0};
  // Defense-in-depth refresh (review finding F2, #1715): Backend::Copy below
  // already runs EnsureHostBytes on the source before its memcpy, so this
  // entry call is redundant today; it is kept explicit because TrustDump's
  // contract ("the payload reflects current truth") must not depend on a
  // copy-path implementation detail elsewhere.
  EnsureHostBytes(t.data);
  const size_t bytes = static_cast<size_t>(t.Numel()) * vt::SizeOf(t.dtype);
  std::vector<uint8_t> a(bytes), b(bytes);
  Backend& tb = GetBackend(DeviceType::kTENSTORRENT);
  tb.Synchronize(q);
  tb.Copy(q, a.data(), t.data, bytes);
  tb.Synchronize(q);
  tb.Copy(q, b.data(), t.data, bytes);
  tb.Synchronize(q);
  const bool verified = (a == b);
  uint32_t dims[4] = {0, 0, 0, 0};
  for (int i = 0; i < t.rank && i < 4; ++i)
    dims[i] = static_cast<uint32_t>(t.shape[i]);
  char hdr[48];
  const uint32_t magic = 0x504D4454;  // 'TDMP'
  const uint32_t dt = static_cast<uint32_t>(t.dtype);
  const uint32_t rk = static_cast<uint32_t>(t.rank);
  const uint64_t numel = static_cast<uint64_t>(t.Numel());
  const uint8_t ver = verified ? 1 : 0;
  std::memcpy(hdr, &magic, 4);
  std::memcpy(hdr + 4, &dt, 4);
  std::memcpy(hdr + 8, &rk, 4);
  std::memcpy(hdr + 12, dims, 16);
  std::memcpy(hdr + 28, &numel, 8);
  std::memcpy(hdr + 36, &ver, 1);
  const uint64_t sid = seq.fetch_add(1, std::memory_order_relaxed);
  std::FILE* f = std::fopen(
      (std::string(dir) + "/" + std::to_string(sid) + "_" + name + ".tdmp").c_str(),
      "wb");
  if (f != nullptr) {
    std::fwrite(hdr, 1, 40, f);
    if (verified) std::fwrite(a.data(), 1, bytes, f);
    std::fclose(f);
  }
}


std::vector<float> DebugDeviceReadbackF32(Queue& q, const Tensor& t) {
  (void)q;
  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev = EnsureDevice2D(t, device);
  std::vector<float> v = dev.to_vector<float>();
  return v;
}

void EnsureHostBytes(void* host) {
  if (host == nullptr) return;
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr, "[TT-TRACE] EnsureHostBytes DURING CAPTURE\n");
  ttnn::Tensor dev;
  size_t bytes = 0;
  void* base = nullptr;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(host);
    if (s == nullptr || s->host_current) return;
    VT_CHECK(s->device_current && s->device.has_value(),
             "tenstorrent: EnsureHostBytes with no current device data");
    dev = *s->device;
    bytes = s->bytes;
    base = s->host;
  }
  std::vector<float> result = dev.to_vector<float>();
  const size_t n = result.size();
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(host);
    if (s == nullptr || s->host_current) return;
    if (s->conv_transposed) {
      // Time-major shadow -> oracle [slots, C, sl] byte order.
      const uint32_t slots = s->conv_slots, Cc = s->conv_c, sl = s->conv_sl;
      const uint32_t R = slots * Cc;
      const int64_t numel = static_cast<int64_t>(slots) * Cc * sl;
      if (bytes >= static_cast<size_t>(numel) * sizeof(float)) {
        auto* dst = static_cast<float*>(base);
        for (int64_t i = 0; i < numel; ++i) {
          const int64_t j = i % sl, rc = i / sl;
          dst[i] = result[static_cast<size_t>(j * R + rc)];
        }
      } else if (bytes >= static_cast<size_t>(numel) * sizeof(uint16_t)) {
        auto* dst = static_cast<uint16_t*>(base);
        for (int64_t i = 0; i < numel; ++i) {
          const int64_t j = i % sl, rc = i / sl;
          dst[i] = F32ToBF16(result[static_cast<size_t>(j * R + rc)]);
        }
      } else {
        VT_CHECK(false, "tenstorrent: EnsureHostBytes host buffer too small");
      }
      s->host_current = true;
      return;
    }
    // Device results are f32 via to_vector. Host Alloc is typically
    // numel*sizeof(float) (tests/f32 path) or numel*2 (bf16 activations).
    if (bytes >= n * sizeof(float)) {
      std::memcpy(base, result.data(), n * sizeof(float));
    } else if (bytes >= n * sizeof(uint16_t)) {
      auto* dst = static_cast<uint16_t*>(base);
      for (size_t i = 0; i < n; ++i) dst[i] = F32ToBF16(result[i]);
    } else {
      VT_CHECK(false, "tenstorrent: EnsureHostBytes host buffer too small");
    }
    s->host_current = true;
  }
}

// ITEM 5: persistent zero tensors, created OUTSIDE capture (ttnn::zeros
// host-fills + to_device()s = an enqueue_write, illegal during trace capture).
// EnsureDevice2D primes the cache during the eager warmup so the captured
// res.Zero finds its entry and replays a warm device->device ttnn::copy.

// HOST-FREE-FORWARD R2: device->device copy when capturing, so Backend::Copy
// does not to_vector inside the captured region. Both dst and src must carry a
// current device shadow of equal byte size; dst's shadow becomes a copy of src.
// The byte volume a shadow's LOGICAL shape holds, or 0 for a dtype this
// contract does not cover (conservative: the D2D copy then declines and the
// host path runs).
static size_t ShadowLogicalBytes(const ttnn::Tensor& t) {
  const auto dt = t.dtype();
  size_t esz = 0;
  if (dt == ttnn::DataType::FLOAT32)
    esz = 4;
  else if (dt == ttnn::DataType::BFLOAT16)
    esz = 2;
  if (esz == 0) return 0;
  uint64_t vol = 1;
  const auto ds = t.logical_shape();
  for (uint32_t i = 0; i < ds.rank(); ++i) vol *= ds[i];
  return static_cast<size_t>(vol) * esz;
}

 bool CopyDeviceDeviceIfCapture(void* dst, const void* src, size_t bytes) {
  // Run the device->device copy when EITHER capturing OR in host-free-decode
  // mode (the env opt-in). The latter is essential so the EAGER warmup step
  // (which the decode-graph framework runs BEFORE capture) also exercises
  // ttnn::empty+ttnn::copy, compiling those programs into the cache so the
  // subsequent capture doesn't hit "Cannot load new binaries during trace
  // capture." Read LIVE, not cached in a static: the inertness-guard case in
  // test_tenstorrent_backend unsets the env mid-process and must observe the
  // decline, and a suite run under an ambient flag must not pin the armed
  // behavior for cases that unset it.
  const bool host_free = HostFreeDecodeEnabled();
  if (!tt_capture_active() && !host_free) return false;
  static bool once = [&] {
    // Enable program cache once on the first host-free path use — ttnn trace
    // requires every captured op to be program-cache-warm.
    MeshDevice& device = SharedMeshDevice();
    device.enable_program_cache();
    return true;
  }();
  (void)once;
  ttnn::Tensor src_dev;
  size_t copy_bytes = 0;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(const_cast<void*>(src));
    BufferSlot* d = FindSlot(dst);
    if (s == nullptr || !s->device_current || !s->device.has_value()) return false;
    if (d == nullptr) return false;
    // The Copy contract is `bytes`, NOT slot capacity. The registered
    // s->bytes / d->bytes are the HOST blocks' pool capacities, and the
    // best-fit lend (#1922) hands a DBuf a block from a LARGER size class:
    // Qwen3-4B #1625 capture staged s.hidden ([1,2560] bf16 = 5120 B) in a
    // 10240-B block lent from the prefill K/V class while the working copy
    // landed on an exactly-5120-B block. Capacity equality declined, the
    // host-path fallthrough issued a D2H read mid-capture, and tt-metal
    // TT_FATALs ("Reads are not supported during trace capture"). The content
    // contract is the SRC SHADOW's logical volume — the tensor the clone
    // reproduces — which must hold exactly the bytes this Copy names.
    if (ShadowLogicalBytes(*s->device) != bytes) return false;
    src_dev = *s->device;
    copy_bytes = bytes;
  }
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr) {
    void* bt[8];
    const int nbt = ::backtrace(bt, 8);
    char** sym = ::backtrace_symbols(bt, nbt);
    std::fprintf(stderr,
                 "[TT-TRACE] device->device copy (capture-safe) dst=%p src=%p "
                 "bytes=%zu frames=%d\n",
                 dst, src, copy_bytes, nbt);
    for (int i = 1; sym != nullptr && i < nbt; ++i)
      std::fprintf(stderr, "[TT-TRACE]   bt[%d]=%p %s\n", i, bt[i], sym[i]);
    std::fflush(stderr);
    std::free(sym);
  }
  MeshDevice& device = SharedMeshDevice();
  // Allocate a destination device tensor matching src's shape/dtype/layout,
  // then copy. No host readback.
  ttnn::Tensor cloned = ttnn::empty(src_dev.logical_shape(), src_dev.dtype(),
                                    src_dev.layout(), &device,
                                    src_dev.memory_config());
  cloned = ttnn::copy(src_dev, cloned);
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* d = FindSlot(dst);
    if (d == nullptr) return false;
    // The shadow just installed is SRC-shaped, and equal byte size does not
    // mean equal geometry: the record must name the served geometry, or a
    // later stage at dst's recorded exact geometry would hit the fast path
    // and be handed a wrongly-shaped tensor. The capture lane mirrors the
    // eager device-resident arm (#2294).
    const auto ds = src_dev.logical_shape();
    d->device = std::move(cloned);
    d->dev_rows = static_cast<uint32_t>(ds[0]);
    d->dev_cols = static_cast<uint32_t>(ds[1]);
    d->device_current = true;
    d->host_current = false;
    d->device_reserved = false;  // real bytes installed — reservation spent
  }
  StagingAvoidedDeviceCopy().fetch_add(1, std::memory_order_relaxed);
  return true;
}

// HOST-FREE-FORWARD R3: on-device fill (for DBuf::Zero -> Backend::Memset)
// when host-free decode is active, so no host write happens inside capture.
// Reinterprets the buffer as a 2D [rows, cols] f32 tensor matching the
// existing device shadow's numel (zeros is the only value the forward uses).
bool MemsetDeviceIfCapture(void* p, int value, size_t bytes) {
  // Live read for the same reason as CopyDeviceDeviceIfCapture above.
  const bool host_free = HostFreeDecodeEnabled();
  if (!tt_capture_active() && !host_free) return false;
  if (value != 0) return false;  // only zero-fill is handled on-device
  // Need an existing shadow to know shape/dtype; or allocate from the slot.
  std::optional<ttnn::Tensor> dev;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(p);
    if (s != nullptr && s->device_current && s->device.has_value()) {
      dev = *s->device;
    }
  }
  std::optional<ttnn::Tensor> fresh;
  if (!dev.has_value()) {
    // HOST-FREE-FORWARD R4 (#1105): a brand-new buffer (registered at Alloc,
    // no device tensor yet) still takes the device lane. res.Zero at the top
    // of the captured layer region must leave the slot device-resident, or
    // the first EnsureDevice2D(*residual) restages from the recycled slot's
    // persistent buffer — an enqueue_write, which trace capture fatals on
    // (fd_mesh_command_queue.cpp:760).
    // bf16-only, the same polarity as the W7 reservation arm: the geometry
    // is derived from the registered byte size, which is dtype-unambiguous
    // only for 2-byte elements.
    // CAPTURE-ONLY: an eager fresh-slot zero keeps the host fallback. The
    // byte size does not name a dtype (the f32 KV masters share these pool
    // blocks), so serving one eagerly would install a wrongly-typed shadow;
    // inside the capture the write is banned and the buffer is scratch whose
    // every consumer reads on device, which is what makes the guess safe.
    // The capture-time zero still finds its tensor: the cold step's
    // EnsureDevice2D restage primed the zero at this exact spec
    // (ZeroCachePrime) and the copy program is warm from the eager copy
    // lane — ZeroCacheGet refuses a capture-time miss by design.
    if (!tt_capture_active()) {
      // Prime the zero-cache AND warm the copy program for this spec during
      // eager warmup: the capture-time lane runs ttnn::copy(zero_src, *fresh)
      // whose CopyDeviceOperation hash is shape-specific, so a copy never
      // executed during warmup is not in the program cache and trace capture
      // fatals on the missing binary. Also prime ZeroCacheGet for the
      // [1, cols] bf16 TILE spec — a fresh-slot memset whose geometry never
      // staged (the 27B bench: a 20480-B res.Zero → [1,10240] bf16 TILE)
      // would miss mid-capture.
      if (bytes > 0 && (bytes % 2) == 0) {
        uint32_t cols = static_cast<uint32_t>(bytes / 2);
        MeshDevice& md = SharedMeshDevice();
        md.enable_program_cache();
        auto shape = ttnn::Shape({1u, cols});
        ZeroCachePrime(shape, ttnn::DataType::BFLOAT16,
                       ttnn::Layout::TILE, md);
        ttnn::Tensor zero_src = ZeroCacheGet(
            shape, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, md);
        ttnn::Tensor tmp = ttnn::empty(shape, ttnn::DataType::BFLOAT16,
                                       ttnn::Layout::TILE, &md,
                                       ttnn::MemoryConfig{});
        ttnn::copy(zero_src, tmp);
      }
      return false;
    }
    MeshDevice& device_fresh = SharedMeshDevice();
    device_fresh.enable_program_cache();
    uint32_t cols = 0;
    {
      std::lock_guard<std::mutex> g(SlotMutex());
      BufferSlot* s = FindSlot(p);
      if (s == nullptr) return false;  // untracked buffer: host memset
      // The MEMSET's byte count names the geometry, not the slot's
      // registered capacity: the #1922 best-fit lend hands a DBuf a block
      // from a larger class (Qwen3-4B #1625: a fresh [1,2560] bf16 res got
      // a 10240-B block), and cols derived from capacity staged a
      // [1,5120] zero whose key no warm step primed — a zero-cache miss
      // VT_CHECK-fails mid-capture. Requested bytes are the contract, the
      // same shadow-volume rule the D2D copy lane above now applies.
      if (bytes == 0 || (bytes % 2) != 0) return false;
      cols = static_cast<uint32_t>(bytes / 2);
      if (s->persistent.has_value() && s->persist_rows == 1 &&
          s->persist_cols == cols) {
        // Recycled staging: zero the persistent buffer IN PLACE — the device
        // address stays stable across steps, which is what lets the captured
        // zero-copy replay against the same buffer.
        fresh = *s->persistent;
      }
    }
    if (!fresh.has_value()) {
      fresh = ttnn::empty(ttnn::Shape({1u, cols}), ttnn::DataType::BFLOAT16,
                          ttnn::Layout::TILE, &device_fresh,
                          ttnn::MemoryConfig{});
      std::lock_guard<std::mutex> g(SlotMutex());
      if (BufferSlot* s = FindSlot(p)) {
        // W5 semantics: the allocation becomes the slot's persistent buffer,
        // so the next zero reuses the same device address.
        s->persistent = fresh;
        s->persist_rows = 1;
        s->persist_cols = cols;
      }
    }
    ttnn::Tensor zero_src = ZeroCacheGet(*fresh, device_fresh);
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
      std::fprintf(stderr, "[TT-TRACE] device zero-fill (fresh slot %p cols=%u)\n",
                   p, cols);
    ttnn::Tensor z = ttnn::copy(zero_src, *fresh);
    (void)z;
    {
      std::lock_guard<std::mutex> g(SlotMutex());
      BufferSlot* s = FindSlot(p);
      if (s == nullptr) return false;
      s->device = *fresh;
      s->dev_rows = 1;
      s->dev_cols = cols;
      s->device_current = true;
      s->host_current = false;
      s->conv_transposed = false;
      s->device_reserved = false;  // real zeros installed — reservation spent
    }
    return true;
  }
  MeshDevice& device = SharedMeshDevice();
  const ttnn::Tensor& shadow = *dev;
  // ITEM 5: ttnn::zeros/full is NOT capture-safe — full_impl host-fills and
  // to_device()s (creation.cpp:52-71), i.e. an enqueue_write that ttnn trace
  // fatals on. The plugin pattern instead: keep PERSISTENT zero tensors
  // (created outside capture, at warmup) and ttnn::copy one onto the target —
  // a device->device program that is captured/replayed like any other warm op.
  ttnn::Tensor zero_src = ZeroCacheGet(shadow, device);
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr, "[TT-TRACE] device zero-fill (capture-safe)\n");
  // Copy the persistent zero onto the shadow IN PLACE (keeps the shadow's
  // device address stable — the whole point of persistent buffers).
  ttnn::Tensor z = ttnn::copy(zero_src, shadow);
  (void)z;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(p);
    if (s == nullptr) return false;
    s->device_current = true;
    s->host_current = false;
    s->device_reserved = false;  // real zeros installed — reservation spent
  }
  return true;
}

// BACKEND-TENSTORRENT-QWEN35 W7 (#2282): the EAGER zero-fill. Outside capture
// (the capture lane is MemsetDeviceIfCapture above), a FULL-slot memset(0) of
// a device-resident slot fills the shadow on-device and KEEPS it: the caller
// still memsets the host bytes, so both sides hold zeros and the next device
// read serves the shadow instead of restaging the zeros. Zero is the only
// value handled — memset writes byte patterns, and only the all-zero pattern
// is a valid value in every dtype the shadows use.
bool MemsetDeviceFill(void* p, int value, size_t bytes) {
  if (value != 0) return false;
  if (tt_capture_active()) return false;  // capture-unsafe refusals keep semantics
  std::optional<ttnn::Tensor> dev;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(p);
    if (s == nullptr || !s->device_current || !s->device.has_value()) return false;
    if (s->bytes != bytes) return false;  // a partial memset keeps the host path
    dev = *s->device;
  }
  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor zero_src = ZeroCacheGet(*dev, device);
  // In place into the slot's owning shadow — shared TensorAttributes carry the
  // write, the shadow's device address stays stable.
  ttnn::Tensor z = ttnn::copy(zero_src, *dev);
  (void)z;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(p);
    if (s == nullptr) return false;
    s->device_current = true;  // shadow KEPT; the caller zeroes the host bytes
    s->host_current = true;
    s->device_reserved = false;  // real zeros installed — reservation spent
  }
  StagingAvoidedMemset().fetch_add(1, std::memory_order_relaxed);
  return true;
}

// BACKEND-TENSTORRENT-QWEN35 W7 (#2282): the EAGER device->device copy.
// Outside capture (the capture lane is CopyDeviceDeviceIfCapture above), a
// copy whose SOURCE shadow is current and whose destination already owns a
// persistent buffer goes device->device: the host path would download the
// source, memcpy, drop the destination's shadow, and re-upload the same bytes
// on the next device read. A destination WITHOUT a persistent buffer is a
// host reader's buffer (d2h readback, dump) and keeps the host path. Base
// pointers only: FindSlot resolves interior views to the base slot, and a
// sub-range copy is not served by a whole-shadow copy.
bool CopyDeviceDeviceIfResident(void* dst, const void* src, size_t bytes) {
  if (tt_capture_active()) return false;
  ttnn::Tensor src_dev;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(const_cast<void*>(src));
    BufferSlot* d = FindSlot(dst);
    if (s == nullptr || !s->device_current || !s->device.has_value()) return false;
    if (d == nullptr || s->host != src || d->host != dst) return false;
    // Same shadow-volume contract as the capture lane above: slot bytes are
    // HOST BLOCK CAPACITY (the #1922 best-fit lend pads it), not the content
    // the shadow holds. The clone must reproduce exactly `bytes`.
    if (d->bytes < bytes) return false;
    if (ShadowLogicalBytes(*s->device) != bytes) return false;
    if (!d->persistent.has_value()) return false;
    src_dev = *s->device;
  }
  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor cloned = ttnn::empty(src_dev.logical_shape(), src_dev.dtype(),
                                    src_dev.layout(), &device,
                                    src_dev.memory_config());
  cloned = ttnn::copy(src_dev, cloned);
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* d = FindSlot(dst);
    if (d == nullptr) return false;
    // The shadow just installed is SRC-shaped, and equal byte size does not
    // mean equal geometry: the record must name the served geometry, or a
    // later stage at dst's recorded exact geometry would hit the fast path
    // and be handed a wrongly-shaped tensor.
    const auto ds = src_dev.logical_shape();
    d->device = std::move(cloned);
    d->dev_rows = static_cast<uint32_t>(ds[0]);
    d->dev_cols = static_cast<uint32_t>(ds[1]);
    d->device_current = true;
    d->host_current = false;
    d->device_reserved = false;  // real bytes installed — reservation spent
  }
  StagingAvoidedDeviceCopy().fetch_add(1, std::memory_order_relaxed);
  return true;
}


// ITEM 5 (rope): driver-side warm hook. The decode-graph driver calls this
// for the step's (padded) positions BEFORE BeginCapture — the exact
// SizeSlot::Refresh slot in qwen3.cpp — so the persistent cos/sin tensors
// are populated outside capture and the captured rope cache-HITs on content.
// hq/hk select the expanded layouts to warm; base/args must match RopeNeox.
void WarmRopeCosSin(const int32_t* positions, int64_t tokens, int64_t hq,
                    int64_t hk, int64_t rot, double base) {
  if (!HostFreeDecodeEnabled()) return;
  MeshDevice& device = SharedMeshDevice();
  std::vector<float> cos_t, sin_t;
  Tensor pos = Tensor::Contiguous(const_cast<int32_t*>(positions), DType::kI32,
                                  Device{DeviceType::kTENSTORRENT, 0}, {tokens});
  const RopeArgs no_scale{};  // plain rope only on the warm path
  BuildCosSinFromPositions(pos, tokens, rot, base, no_scale, cos_t, sin_t);
  // Byte-exact with what the captured rope reads: the per-step cos|sin CACHE
  // stores f32-built values into a BF16 tensor (RopeCosSinCacheKernel's
  // StoreElemF32 rounds), and the rope-side gather reads them back. Round the
  // warm content through the same bf16 round-trip so the content-HIT
  // comparison is exact.
  for (auto& v : cos_t) v = BF16ToF32(F32ToBF16(v));
  for (auto& v : sin_t) v = BF16ToF32(F32ToBF16(v));
  auto warm_one = [&](int64_t heads) {
    std::vector<float> ce, se;
    ExpandCosSinPerHead(cos_t.data(), sin_t.data(), tokens, heads, rot / 2, ce, se);
    const uint32_t thu = static_cast<uint32_t>(tokens * heads);
    const uint32_t halfu = static_cast<uint32_t>(rot / 2);
    std::lock_guard<std::mutex> g(RopeCSMutex());
    auto& c = RopeCSCache();
    const std::string k = RopeCSKey(thu, halfu);
    auto it = c.find(k);
    if (it == c.end()) {
      RopeCSEntry e;
      e.cos = UploadRows(ce.data(), thu, halfu, device);
      e.sin = UploadRows(se.data(), thu, halfu, device);
      e.cos_host = ce;
      c[k] = std::move(e);
    } else if (it->second.cos_host != ce) {
      // In-place CONTENT refresh of the SAME device tensors: a captured rope
      // op reads the address recorded at capture time, so replacing the
      // tensors here would leave every replay reading the capture-step
      // cos/sin (stale positions). The host tensors are built with the
      // identical bf16 TILE spec so copy_to_device writes byte-matching
      // data. Legal here: the driver calls this outside capture.
      // VT_TT_NO_ROPE_REFRESH: stall bisection only — skip the per-step H2D
      // copies AFTER the first capture (stale cos/sin on replays, numerically
      // wrong, mechanics test only).
      if (ReplayRegimeBisectSkip("VT_TT_NO_ROPE_REFRESH")) return;
      ttnn::Tensor cos_h = ttnn::Tensor::from_vector<float>(
          ce, TileSpecOf(thu, halfu), nullptr);
      ttnn::Tensor sin_h = ttnn::Tensor::from_vector<float>(
          se, TileSpecOf(thu, halfu), nullptr);
      ttnn::copy_to_device(cos_h, it->second.cos);
      ttnn::copy_to_device(sin_h, it->second.sin);
      it->second.cos_host = ce;
    }
  };
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr, "[TT-TRACE] WarmRopeCosSin tokens=%lld hq=%lld hk=%lld"
                 " rot=%lld first_pos=%d cos_first=%f\n",
                 (long long)tokens, (long long)hq, (long long)hk,
                 (long long)rot, (int)positions[0],
                 cos_t.empty() ? -1.0f : cos_t.front());
  warm_one(hq);
  warm_one(hk);
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr) {
    std::lock_guard<std::mutex> g(RopeCSMutex());
    for (auto& [k, e] : RopeCSCache())
      std::fprintf(stderr, "[TT-TRACE] warm stored key=%s first=%f n=%zu\n",
                   k.c_str(), e.cos_host.empty() ? -1.0f : e.cos_host.front(),
                   e.cos_host.size());
  }
}

// kAttnQkNormRopeGate's per-step cos|sin table: the dense decode-graph driver
// calls this EVERY step (capture + replay), outside capture, with the same
// positions vector the in-region RopeCosSinCacheKernel refill reads, so the
// captured fused preamble serves a table matching this step (WarmRopeCosSin's
// populate-outside/content-HIT-inside pattern; the SizeSlot::Refresh shape).
// Byte-exactness: RopeCosSinCacheKernel stores plain f32 (no bf16 round-trip
// needed — the DBuf is kF32), and BuildCosSinFromPositions carries the same
// double-pow + Llama3ScaleFreq math the kernel inlines.
void WarmAttnCosSin(const int32_t* positions, int64_t tokens, int64_t rot,
                      double base) {
  if (!HostFreeDecodeEnabled()) return;
  if (rot <= 0 || (rot % 2) != 0 || tokens < 1) return;
  Tensor pos = Tensor::Contiguous(const_cast<int32_t*>(positions), DType::kI32,
                                    Device{DeviceType::kTENSTORRENT, 0}, {tokens});
  std::vector<float> cos_t, sin_t;
  BuildCosSinFromPositions(pos, tokens, static_cast<int>(rot), base, RopeArgs{},
                             cos_t, sin_t);
  const int64_t half = rot / 2;
  std::vector<float> table(static_cast<size_t>(tokens) * rot);
  for (int64_t i = 0; i < tokens; ++i) {
    for (int64_t j = 0; j < half; ++j) {
      table[static_cast<size_t>(i) * rot + j] =
          cos_t[static_cast<size_t>(i) * half + j];
      table[static_cast<size_t>(i) * rot + half + j] =
          sin_t[static_cast<size_t>(i) * half + j];
    }
  }
  const uint32_t tu = static_cast<uint32_t>(tokens);
  const uint32_t rotu = static_cast<uint32_t>(rot);
  const std::string dbg_key = AttnCSKey(tu, rotu);
  std::lock_guard<std::mutex> g(AttnCSMutex());
  auto& c = AttnCSCache();
  const std::string k = dbg_key;
  auto it = c.find(k);
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr,
                 "[TT-TRACE] WarmAttnCosSin tokens=%lld rot=%lld key=%s "
                 "found=%d first=%f\n",
                 (long long)tokens, (long long)rot, k.c_str(),
                 it != c.end() ? 1 : 0,
                 table.empty() ? -1.0f : table.front());
  if (it == c.end()) {
    AttnCSEntry e;
    e.cs_host = table;
    e.cs = UploadTensor(std::move(table), ttnn::Shape({tu, rotu}),
                          ttnn::DataType::FLOAT32, ttnn::Layout::TILE,
                          SharedMeshDevice());
    c[k] = std::move(e);
  } else if (it->second.cs_host != table) {
    // In-place content refresh of the SAME device tensor: a captured op reads
    // the address recorded at capture time, so replacing the tensor would
    // strand every replay on the capture-step positions. Legal here — the
    // driver calls this outside capture.
    // Same f32 TILE spec the entry was uploaded with (TileSpecOf is the BF16
    // rope-cache spec — a dtype mismatch here is a TT_FATAL in copy_to_device).
    ttnn::Tensor h = ttnn::Tensor::from_vector<float>(
        table,
        SpecOf(tt::tt_metal::Shape({tu, rotu}), ttnn::DataType::FLOAT32,
               ttnn::Layout::TILE),
        nullptr);
    ttnn::copy_to_device(h, it->second.cs);
    it->second.cs_host = std::move(table);
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
      std::fprintf(stderr, "[TT-TRACE] WarmAttnCosSin REFRESHED key=%s\n",
                   k.c_str());
  }
}

// Serveability query for the transposed conv-state shadow: mirrors the
// EnsureConvStateTransposed fast path READ-ONLY (no download, no upload), so
// the decode driver can gate capture/replay on it outside capture. A
// prefill-bearing step's ssm/cache role transition (GdnStateGather) clears
// the shadow and replaces the slot's device tensor; decode must then run the
// step eagerly — which rebuilds the shadow — before any graph may capture or
// replay against the buffer again.
bool ConvShadowServeable(const void* conv_state_data, int64_t slots,
                         int64_t conv_dim, int64_t state_len) {
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(const_cast<void*>(conv_state_data));
  if (s == nullptr) return false;
  const uint32_t uslots = static_cast<uint32_t>(slots);
  const uint32_t uc = static_cast<uint32_t>(conv_dim);
  const uint32_t usl = static_cast<uint32_t>(state_len);
  return s->device_current && s->device.has_value() && s->conv_transposed &&
         s->conv_slots == uslots && s->conv_c == uc && s->conv_sl == usl &&
         s->device->dtype() == ttnn::DataType::FLOAT32 &&
         s->device->layout() == ttnn::Layout::TILE &&
         s->device->logical_shape()[0] == usl + 1 &&
         s->device->logical_shape()[1] == uslots * uc;
}

// ---- BACKEND-TENSTORRENT-QWEN35 W4 (#2107): staging counters ----
// Readers for the tenstorrent_device.h API; the increment sites live with the
// staging paths in the anonymous namespace above.
StagingStats GetStagingStats() {
  StagingStats s;
  s.uploads_bulk_bf16 = StagingBulkUploads().load(std::memory_order_relaxed);
  s.staged_bulk_bf16_bytes = StagingBulkBytes().load(std::memory_order_relaxed);
  s.staged_f32_elems = StagingF32Elems().load(std::memory_order_relaxed);
  s.uploads_persistent_bf16 =
      StagingPersistentWrites().load(std::memory_order_relaxed);
  s.uploads_persistent_allocs =
      StagingPersistentAllocs().load(std::memory_order_relaxed);
  s.staged_persistent_bf16_bytes =
      StagingPersistentBytes().load(std::memory_order_relaxed);
  s.stages_avoided_reservation =
      StagingAvoidedReservation().load(std::memory_order_relaxed);
  s.stages_avoided_device_memset =
      StagingAvoidedMemset().load(std::memory_order_relaxed);
  s.stages_avoided_device_copy =
      StagingAvoidedDeviceCopy().load(std::memory_order_relaxed);
  return s;
}

void ResetStagingStats() {
  StagingBulkUploads().store(0, std::memory_order_relaxed);
  StagingBulkBytes().store(0, std::memory_order_relaxed);
  StagingF32Elems().store(0, std::memory_order_relaxed);
  StagingPersistentWrites().store(0, std::memory_order_relaxed);
  StagingPersistentAllocs().store(0, std::memory_order_relaxed);
  StagingPersistentBytes().store(0, std::memory_order_relaxed);
  StagingAvoidedReservation().store(0, std::memory_order_relaxed);
  StagingAvoidedMemset().store(0, std::memory_order_relaxed);
  StagingAvoidedDeviceCopy().store(0, std::memory_order_relaxed);
}

}  // namespace vt::tenstorrent
