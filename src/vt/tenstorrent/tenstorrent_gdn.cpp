// Tenstorrent GDN op kernels (BACKEND-TENSTORRENT-SPLIT stage 3).
// Definitions moved verbatim from tenstorrent_ops.cpp; the shared counters,
// idx/conv-tile caches and cross-TU declarations live in
// tenstorrent_internal.h.
#include "vt/tenstorrent/tenstorrent_internal.h"

namespace vt::tenstorrent {

// kGdnPostConv: fused GDN post-conv prep = GdnConvSplit + per-head L2Norm(q/k)
// + GdnGBeta in one pass (cpu_ops.cpp GdnPostConvKernel:3472-3516; wrapper
// ops.cpp:4255-4287; the fla fused_recurrent_gated_delta_rule prefill preamble).
// q/k: conv slices [0,key_dim)/[key_dim,2*key_dim) reshape to [T*Hk, Dk] rows
// and run the kL2Norm device math (square -> sum -> +eps -> rsqrt -> mul) in
// bf16 tiles — the row's calibrated l2norm envelope. v: plain slice copy.
// g/beta: FLOAT32 tiles end to end (softplus threshold-20, exp(a_log) and
// sigmoid are f32 in the oracle); araw/braw are row-strided views, gathered
// on host exactly like the kRmsNormGated gate view.
void GdnPostConvKernel(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& v_out,
                       Tensor& g_out, Tensor& beta_out, const Tensor& conv,
                       const Tensor& araw, const Tensor& braw,
                       const Tensor& a_log, const Tensor& dt_bias,
                       const L2NormArgs& args) {
  TT_OP_TRACE("GdnPostConv");
  VT_CHECK((q_out.dtype == DType::kF32 || q_out.dtype == DType::kBF16) &&
               k_out.dtype == q_out.dtype && v_out.dtype == q_out.dtype,
           "tenstorrent kGdnPostConv: q_out/k_out/v_out must be f32 or bf16, same dtype");
  VT_CHECK(conv.dtype == DType::kF32 || conv.dtype == DType::kBF16,
           "tenstorrent kGdnPostConv: conv must be f32 or bf16");
  VT_CHECK(g_out.dtype == DType::kF32 && beta_out.dtype == DType::kF32 &&
               (araw.dtype == DType::kF32 || araw.dtype == DType::kBF16) &&
               braw.dtype == araw.dtype && a_log.dtype == DType::kF32 &&
               dt_bias.dtype == DType::kF32,
           "tenstorrent kGdnPostConv: g/beta/a_log/dt_bias f32; a/b share f32 or bf16");
  VT_CHECK(q_out.IsContiguous() && k_out.IsContiguous() && v_out.IsContiguous() &&
               g_out.IsContiguous() && beta_out.IsContiguous() &&
               conv.IsContiguous() && araw.stride[1] == 1 && braw.stride[1] == 1 &&
               a_log.IsContiguous() && dt_bias.IsContiguous(),
           "tenstorrent kGdnPostConv: contiguous required (a/b row views excepted)");
  const uint32_t t = static_cast<uint32_t>(conv.shape[0]);
  const uint32_t hk = static_cast<uint32_t>(q_out.shape[1]);
  const uint32_t dk = static_cast<uint32_t>(q_out.shape[2]);
  const uint32_t hv = static_cast<uint32_t>(v_out.shape[1]);
  const uint32_t dv = static_cast<uint32_t>(v_out.shape[2]);
  const uint32_t key_dim = hk * dk, value_dim = hv * dv;
  const uint32_t conv_dim = 2 * key_dim + value_dim;
  VT_CHECK(conv.shape[0] == q_out.shape[0] && conv.shape[1] == conv_dim &&
               g_out.shape[0] == t && g_out.shape[1] == hv,
           "tenstorrent kGdnPostConv: shape mismatch");

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_conv = DeviceRows(conv, t, conv_dim, device);
  ttnn::Tensor q2 = ttnn::slice(dev_conv, ttsl::SmallVector<uint32_t>{0, 0},
                                ttsl::SmallVector<uint32_t>{t, key_dim},
                                ttsl::SmallVector<uint32_t>{1, 1});
  ttnn::Tensor k2 = ttnn::slice(dev_conv, ttsl::SmallVector<uint32_t>{0, key_dim},
                                ttsl::SmallVector<uint32_t>{t, 2 * key_dim},
                                ttsl::SmallVector<uint32_t>{1, 1});
  ttnn::Tensor v2 = ttnn::slice(dev_conv, ttsl::SmallVector<uint32_t>{0, 2 * key_dim},
                                ttsl::SmallVector<uint32_t>{t, conv_dim},
                                ttsl::SmallVector<uint32_t>{1, 1});
  // Heads are laid contiguously along the sliced cols, so [t, hk*dk] ->
  // [t*hk, dk] is a pure logical re-view; each row is one head's Dk vector.
  auto l2 = [&](const ttnn::Tensor& cols) {
    ttnn::Tensor rows =
        ttnn::reshape(cols, ttnn::Shape({t * hk, dk}));
    ttnn::Tensor sq = ttnn::multiply(rows, rows);
    ttnn::Tensor s = ttnn::sum(sq, ttsl::SmallVector<int>{1}, true);
    ttnn::Tensor denom = ttnn::add(s, args.eps);
    ttnn::Tensor inv = ttnn::rsqrt(denom);
    return ttnn::multiply(rows, inv);
  };
  CommitDeviceLogical2D(q_out, l2(q2), t * hk, dk);
  CommitDeviceLogical2D(k_out, l2(k2), t * hk, dk);
  CommitDeviceLogical2D(v_out, ttnn::reshape(v2, ttnn::Shape({t * hv, dv})), t * hv, dv);

  // g/beta in f32 (the row's "f32 intermediates" doctrine: softplus(x) with
  // threshold 20, exp(a_log) and sigmoid must not round their inputs).
  // a_log/dt_bias are model constants — host bytes stay current for the
  // model's lifetime, so EnsureHost is a no-op. a/b are per-step producer
  // outputs: device-authoritative in the captured arm, so they are served
  // from their current shadows (an EnsureHost here enqueues exactly the
  // readback the trace refuses); the eager arm keeps the host-staging
  // fallback for shapes with no committed shadow.
  EnsureHost(a_log);
  EnsureHost(dt_bias);
  ttnn::Tensor dev_a, dev_b;
  if (!(ServePostConvAB(araw, t, hv, dev_a) && ServePostConvAB(braw, t, hv, dev_b))) {
    VT_CHECK(!tt_capture_active(),
             "tenstorrent kGdnPostConv: a/b arrived without a servable "
             "device shadow during trace capture — the producer must commit "
             "device-side before the captured region");
    EnsureHost(araw);
    EnsureHost(braw);
  if (const char* td = std::getenv("VT_DUMP_TRUST")) {
    static std::atomic<int> gb_seq{0};
    if (gb_seq.fetch_add(1, std::memory_order_relaxed) == 0) {
      // Raw HOST bytes of the a/b windows exactly as LoadElemF32 sees them,
      // after EnsureHost. Compare against ba_a_win_dev/ba_b_win_dev.
      std::FILE* fp = std::fopen(
          (std::string(td) + "/0_ab_host_raw.f32").c_str(), "wb");
      if (fp) {
        const float* ap = static_cast<const float*>(
            static_cast<const void*>(araw.data));
        const float* bp = static_cast<const float*>(
            static_cast<const void*>(braw.data));
        for (uint32_t i = 0; i < t; ++i) {
          std::fwrite(ap + i * araw.stride[0], 4, hv, fp);
          std::fwrite(bp + i * braw.stride[0], 4, hv, fp);
        }
        std::fclose(fp);
      }
    }
  }
  std::vector<float> a(static_cast<size_t>(t) * hv), b(a.size());
  for (uint32_t i = 0; i < t; ++i)
    for (uint32_t h = 0; h < hv; ++h) {
      const int64_t aidx = i * araw.stride[0] + h;
      const int64_t bidx = i * braw.stride[0] + h;
      a[static_cast<size_t>(i) * hv + h] = LoadElemF32(araw, aidx);
      b[static_cast<size_t>(i) * hv + h] = LoadElemF32(braw, bidx);
    }
  dev_a = UploadTensor(std::move(a), ttnn::Shape({t, hv}),
                       ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
  dev_b = UploadTensor(std::move(b), ttnn::Shape({t, hv}),
                       ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
  }
  // a_log/dt_bias are immutable model constants — cached host-content tiles
  // (the conv weight discipline): the warm step uploads them once and the
  // captured region serves the resident tile. The per-call UploadTensor this
  // replaces is a write_shard, which the trace refuses — the captured arm
  // fatalled exactly here once a/b stopped EnsureHost-ing first.
  ttnn::Tensor dev_al = CachedTile(
      a_log.data, hv, 0, 0,
      [&] {
        std::vector<float> v(hv);
        for (uint32_t h = 0; h < hv; ++h) v[h] = LoadElemF32(a_log, h);
        return v;
      },
      ttnn::Shape({1, hv}), device);
  ttnn::Tensor dev_dt = CachedTile(
      dt_bias.data, hv, 0, 0,
      [&] {
        std::vector<float> v(hv);
        for (uint32_t h = 0; h < hv; ++h) v[h] = LoadElemF32(dt_bias, h);
        return v;
      },
      ttnn::Shape({1, hv}), device);

  // x = araw + dt_bias (row-vector broadcast); sp = relu(x) +
  // log1p(exp(-|x|)) — the oracle's log1p(exp(x)) written in a form that is
  // stable for every x (the exp argument is always <= 0). For x > 20 the
  // log1p term sits below x's half-ULP, so the sum rounds to x EXACTLY and
  // reproduces the oracle's threshold branch. Used instead of the one-shot
  // SFPU softplus poly, whose ~1e-6 ABSOLUTE fit error becomes ~1e-3
  // RELATIVE on small softplus values (measured on the P150, first green
  // run: g max_rel 4.4e-4 over |g|~0.04). g = -exp(a_log)*sp.
  ttnn::Tensor x = ttnn::add(dev_a, dev_dt);
  ttnn::Tensor sp = ttnn::add(
      ttnn::relu(x), ttnn::log1p(ttnn::exp(ttnn::neg(ttnn::abs(x)))));
  ttnn::Tensor neg_ea = ttnn::multiply(ttnn::exp(dev_al), -1.0f);
  ttnn::Tensor g = ttnn::multiply(neg_ea, sp);
  ttnn::Tensor beta = ttnn::sigmoid(dev_b);
  CommitDeviceLogical2D(g_out, std::move(g), t, hv);
  CommitDeviceLogical2D(beta_out, std::move(beta), t, hv);
  if (const char* td = std::getenv("VT_DUMP_TRUST")) {
    // Kernel-side trusted captures at the COMMIT SITE: same tensor objects,
    // same TU — removes every cross-function identity assumption.
    static std::atomic<int> gpc_seq{0};
    const int call = gpc_seq.fetch_add(1, std::memory_order_relaxed);
    if (call == 0) {
      TrustDump(q, td, "k_conv_in", conv);
      TrustDump(q, td, "k_q", q_out);
      TrustDump(q, td, "k_k", k_out);
      TrustDump(q, td, "k_v", v_out);
      TrustDump(q, td, "k_g", g_out);
      TrustDump(q, td, "k_beta", beta_out);
      // Intermediates of the g chain (device truth via to_vector).
      auto tdv = [&](const char* n, const ttnn::Tensor& t) {
        auto v = t.to_vector<float>();
        std::FILE* f = std::fopen(
            (std::string(td) + "/" + std::to_string(call) + "_" + n + ".f32")
                .c_str(), "wb");
        if (f) { std::fwrite(v.data(), 4, v.size(), f); std::fclose(f); }
      };
      tdv("k_dev_a", dev_a);
      tdv("k_dev_b", dev_b);
      tdv("k_dev_al", dev_al);
      tdv("k_dev_dt", dev_dt);
      tdv("k_x", x);
      tdv("k_sp", sp);
      tdv("k_neg_ea", neg_ea);
    }
  }
}

// kGdnPrefill: the chunked gated-delta-rule scan over a varlen batch, behind
// the tt-metal fused kernel (ttnn::transformer::chunk_gated_delta_rule — one
// Tensix core per (B·HV) head, recurrent state on-core, fp32 state / HiFi4).
// Adapter duties (spec "Design"):
//   - varlen [T,...] + query_start_loc -> ONE dense padded [N,L,...] batch:
//     per-sequence initial states ride the batch's [B,HV,K,V] initial_state,
//     so N sequences cost one op call, not N. L is padded to the 64-token
//     chunk multiple so the op's own time-pad path stays idle; padded rows
//     carry zero q/k/v/g/beta, and g=0/beta=0 is an IDENTITY state update
//     (exp(0)=1, v'=0), so empty sequences and short tails leave the state
//     exactly where the oracle leaves it.
//   - q/k arrive PRE-normalized and GdnArgs::scale multiplies q (the op folds
//     scale into q itself), so use_qk_l2norm stays FALSE (the op fatal-errors
//     on true; FLA scale semantics match the CPU GdnHeadTokenStep).
//   - state layout: ours [N,Hv,Dv,Dk], the op's [B,HV,K,V] — one transpose of
//     the trailing dims on upload and download, inside the adapter.
// Host-staged upload/download in W1 (the decode shadow that keeps the state
// resident is W2's); outputs are written to the host bytes and committed.
void GdnPrefillKernel(Queue&, Tensor& out, const Tensor& q_in, const Tensor& k_in,
                      const Tensor& v_in, const Tensor& g, const Tensor& beta,
                      Tensor& state, const Tensor& qsl, const GdnArgs& args) {
  TT_OP_TRACE("GdnPrefill");
  const int64_t n = state.shape[0], hv = state.shape[1], dv = state.shape[2],
                dk = state.shape[3];
  const int64_t hk = q_in.shape[1];
  const int64_t total = q_in.shape[0];
  // chunk_gated_delta_rule device constraints (validated TILE dims): refuse
  // by name rather than silently reshape into a wrong answer (spec Risk #4).
  VT_CHECK(dk % 32 == 0 && dv % 32 == 0,
           "tenstorrent gdn_prefill: tt-metal chunk_gated_delta_rule needs "
           "Dk/Dv multiples of 32 (tile), got Dk=" +
               std::to_string(dk) + " Dv=" + std::to_string(dv));
  EnsureHost(qsl);
  const int32_t* qslp = qsl.Ptr<int32_t>();
  VT_CHECK(qslp[0] == 0 && qslp[n] == total,
           "tenstorrent gdn_prefill: bad query_start_loc bounds");
  int64_t max_len = 0;
  for (int64_t s = 0; s < n; ++s) {
    VT_CHECK(qslp[s + 1] >= qslp[s],
             "tenstorrent gdn_prefill: query_start_loc not monotonic");
    max_len = std::max(max_len, static_cast<int64_t>(qslp[s + 1] - qslp[s]));
  }
  if (total == 0) return;  // all sequences empty: no out rows, state as-is

  constexpr int64_t kChunk = 64;
  const int64_t len = ((max_len + kChunk - 1) / kChunk) * kChunk;

  EnsureHost(q_in);
  EnsureHost(k_in);
  EnsureHost(v_in);
  EnsureHost(g);
  EnsureHost(beta);
  EnsureHost(state);
  EnsureHost(out);

  // Dense padded batch [N, len, ...]: zero-filled tails are identity updates.
  const size_t qk_elems = static_cast<size_t>(n) * len * hk * dk;
  const size_t v_elems = static_cast<size_t>(n) * len * hv * dv;
  std::vector<float> qp(qk_elems, 0.0f), kp(qk_elems, 0.0f), vp(v_elems, 0.0f);
  std::vector<float> gp(static_cast<size_t>(n) * len * hv, 0.0f);
  std::vector<float> bp(static_cast<size_t>(n) * len * hv, 0.0f);
  // initial state [N,Hv,Dk,Dv] — the trailing-dims transpose of ours.
  std::vector<float> s0(static_cast<size_t>(n) * hv * dk * dv, 0.0f);
  for (int64_t s = 0; s < n; ++s) {
    const int64_t t0 = qslp[s], t_len = qslp[s + 1] - t0;
    for (int64_t t = 0; t < t_len; ++t) {
      const int64_t src = t0 + t, dst = s * len + t;
      for (int64_t e = 0; e < hk * dk; ++e) {
        qp[static_cast<size_t>(dst * hk * dk + e)] =
            LoadElemF32(q_in, src * hk * dk + e);
        kp[static_cast<size_t>(dst * hk * dk + e)] =
            LoadElemF32(k_in, src * hk * dk + e);
      }
      for (int64_t e = 0; e < hv * dv; ++e)
        vp[static_cast<size_t>(dst * hv * dv + e)] =
            LoadElemF32(v_in, src * hv * dv + e);
      for (int64_t e = 0; e < hv; ++e) {
        gp[static_cast<size_t>(dst * hv + e)] = LoadElemF32(g, src * hv + e);
        bp[static_cast<size_t>(dst * hv + e)] = LoadElemF32(beta, src * hv + e);
      }
    }
    for (int64_t h = 0; h < hv; ++h) {
      const float* ours =
          state.Ptr<float>() + ((s * hv + h) * dv) * dk;  // [Dv,Dk] rows
      float* theirs = &s0[static_cast<size_t>((s * hv + h) * dk) * dv];  // [Dk,Dv]
      for (int64_t i = 0; i < dv; ++i)
        for (int64_t j = 0; j < dk; ++j)
          theirs[static_cast<size_t>(j * dv + i)] =
              ours[static_cast<size_t>(i * dk + j)];
    }
  }

  MeshDevice& device = SharedMeshDevice();
  const uint32_t un = static_cast<uint32_t>(n), ul = static_cast<uint32_t>(len),
                 uhk = static_cast<uint32_t>(hk), uhv = static_cast<uint32_t>(hv),
                 udk = static_cast<uint32_t>(dk), udv = static_cast<uint32_t>(dv);
  ttnn::Tensor dev_q =
      UploadTensor(std::move(qp), ttnn::Shape({un, ul, uhk, udk}),
                   ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, device);
  ttnn::Tensor dev_k =
      UploadTensor(std::move(kp), ttnn::Shape({un, ul, uhk, udk}),
                   ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, device);
  ttnn::Tensor dev_v =
      UploadTensor(std::move(vp), ttnn::Shape({un, ul, uhv, udv}),
                   ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, device);
  ttnn::Tensor dev_g =
      UploadTensor(std::move(gp), ttnn::Shape({un, ul, uhv}),
                   ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
  ttnn::Tensor dev_b =
      UploadTensor(std::move(bp), ttnn::Shape({un, ul, uhv}),
                   ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
  ttnn::Tensor dev_s0 =
      UploadTensor(std::move(s0), ttnn::Shape({un, uhv, udk, udv}),
                   ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
  auto [o, final_state] = ttnn::transformer::chunk_gated_delta_rule(
      dev_q, dev_k, dev_v, dev_g, dev_b, args.scale, dev_s0,
      /*output_final_state=*/true, static_cast<uint32_t>(kChunk),
      /*use_qk_l2norm=*/false,  // caller pre-normalized q/k
      /*output_head_major=*/false);
  VT_CHECK(final_state.has_value(),
           "tenstorrent gdn_prefill: chunk_gated_delta_rule returned no final state");
  const std::vector<float> ov = o.to_vector<float>();           // [N,len,Hv,Dv]
  const std::vector<float> fv = final_state->to_vector<float>();  // [N,Hv,Dk,Dv]

  // Scatter token-major outputs back into the packed varlen rows.
  for (int64_t s = 0; s < n; ++s) {
    const int64_t t_len = qslp[s + 1] - qslp[s];
    for (int64_t t = 0; t < t_len; ++t)
      for (int64_t e = 0; e < hv * dv; ++e)
        StoreElemF32(out, (qslp[s] + t) * hv * dv + e,
                     ov[static_cast<size_t>(((s * len + t) * hv) * dv + e)]);
  }
  // Final state: transpose the trailing dims back into [N,Hv,Dv,Dk].
  for (int64_t s = 0; s < n; ++s)
    for (int64_t h = 0; h < hv; ++h) {
      const float* theirs =
          &fv[static_cast<size_t>((s * hv + h) * dk) * dv];  // [Dk,Dv]
      float* ours = state.Ptr<float>() + ((s * hv + h) * dv) * dk;  // [Dv,Dk]
      for (int64_t i = 0; i < dv; ++i)
        for (int64_t j = 0; j < dk; ++j)
          ours[static_cast<size_t>(i * dk + j)] =
              theirs[static_cast<size_t>(j * dv + i)];
    }
  CommitHost(out);
  CommitHost(state);
}

namespace {

// The shadow-row split factor for a wide cache: indexed_fill's generic
// interleaved path stages TWO full pages of the LAST dim through its
// dataflow buffer (indexed_fill_program_factory.cpp: page_size =
// padded_shape[-1] * elem_size, data DFB num_entries = 2), so a row wider
// than kStageBytes/(2*esz) elems overflows L1 — the Qwen3.5 GDN ssm_state
// row Hv*Dk*Dv = 16*128*128 = 262144 f32 elems stages 2 x 1 MB and threw
// exactly that ("grow to 2208704 B beyond max L1 size of 1572864 B", first
// W2a e2e bootstrap, /tmp/w2a_e2e_bootstrap.log). The fix is NOT a
// last-dim-changing view: tt::tt_metal::view moves no bytes, but device
// pages are the INTERLEAVE UNIT (buffer.cpp Buffer::page_address:
// bank_offset = aligned_page_size * (page_index / num_banks)) — reinterpreting
// [rows, cols] as [rows*nb, cb] changes every page index's bank mapping and
// scrambles the flat order (measured in W2a: one block landed correctly,
// three came back scrambled). Instead the SHADOW IS BORN SPLIT:
// EnsureGdnCacheDevice uploads the SAME flat host bytes as
// [rows*F, cols/F], so every later op sees narrow rows natively and the
// fill stays ONE launch inside the budget. F is the smallest divisor of
// cols with 2*(cols/F)*esz <= 1 MiB; F == 1 (cols at or under the budget —
// every small-cache model) keeps the W1-verified form byte-identical.
int64_t SplitFactor(int64_t cols, int64_t esz) {
  constexpr int64_t kStageBytes = 1 << 20;
  int64_t f = std::max<int64_t>(1, (2 * cols * esz + kStageBytes - 1) / kStageBytes);
  while (cols % f != 0) ++f;
  return f;
}

// Ensure the f32 device shadow of a GDN state/conv CACHE tensor, viewed as
// [rows, cols]. Uploads (counted) only when the shadow is missing, stale
// (host wrote), or the wrong shape/dtype. The device tensor is FLOAT32 TILE
// regardless of the host cache dtype (bf16 caches round on upload/download —
// EnsureHostBytes already does the f32→bf16 rounding on the way out).
ttnn::Tensor EnsureGdnCacheDevice(const Tensor& t, int64_t rows, int64_t cols,
                                  MeshDevice& device,
                                  ttnn::Layout layout = ttnn::Layout::TILE) {
  // The shadow is stored SPLIT: [rows*F, cols/F] with F = SplitFactor(cols)
  // (see the helper above — wide rows must not become indexed_fill pages).
  // dev_rows/dev_cols stay LOGICAL; the flat bytes are identical either way,
  // so every download and every volume check is unaffected.
  const uint32_t sf = static_cast<uint32_t>(SplitFactor(cols, /*esz=*/4));
  const uint32_t ublk = static_cast<uint32_t>(cols / sf);
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(t.data);
    if (s != nullptr && s->conv_transposed) {
      // W3 #2201, scoped to the corrupting case by the HIGH review finding:
      // ONE conv-state buffer legitimately reaches this path under two views —
      // the decode step commits it time-major ([sl+1, R], conv_transposed,
      // host-stale) and a later prefill-bearing step gathers it in the
      // ssm/cache view at a DIFFERING volume (qwen3_5.cpp GdnStateGather on
      // state.conv_state). That transition must fall through to the slow path
      // below: EnsureHost transposes the shadow back into the caller's order
      // and the refresh re-uploads it in this role's geometry (pinned by the
      // decode-to-prefill alternation test). What must never happen is the
      // fast path serving — or, at equal volume, reshaping — the transposed
      // shadow under this role's shape: same numel, different geometry is a
      // silent wrong-geometry serve. Refuse exactly that, naming both
      // geometries.
      VT_CHECK(static_cast<uint64_t>(s->dev_rows) * s->dev_cols !=
                   static_cast<uint64_t>(rows) * cols,
               "tenstorrent: GDN cache role mismatch: host pointer already "
               "holds a conv_transposed shadow [" + std::to_string(s->dev_rows) +
                   "x" + std::to_string(s->dev_cols) +
                   "]; refusing to serve or reshape it at equal volume as [" +
                   std::to_string(rows) + "x" + std::to_string(cols) + "]");
    }
    if (s != nullptr && s->device_current && s->device.has_value() &&
        s->device->dtype() == ttnn::DataType::FLOAT32 &&
        s->device->layout() == layout &&
        static_cast<uint64_t>(s->dev_rows) * s->dev_cols ==
            static_cast<uint64_t>(rows) * cols) {
      if (s->dev_rows == rows && s->dev_cols == cols) return *s->device;
      // Same buffer re-served at new logical dims: reshape to the NEW dims'
      // split geometry (equal volume). Always use the member reshape (pure
      // view) to produce the same padded shape in both eager and capture,
      // so downstream ops (untilize in ScatterRowsDevice) see the same input
      // and hit the program cache.
      const auto old_padded = s->device->padded_shape();
      ttnn::Tensor reshaped = s->device->reshape(
          ttnn::Shape({static_cast<uint32_t>(rows) * sf, ublk}), old_padded);
      s->device = reshaped;
      s->dev_rows = static_cast<uint32_t>(rows);
      s->dev_cols = static_cast<uint32_t>(cols);
      return reshaped;
    }
    // W3 #2201: the refresh below calls EnsureHost, which downloads a
    // device-current host-stale shadow (wrong dtype/layout for this role —
    // a foreign commit over the same buffer) before the re-upload. Count
    // that d2h traffic; the download itself is the shadow's logical volume.
    if (s != nullptr && !s->host_current) {
      GdnStateD2hBytes().fetch_add(
          static_cast<uint64_t>(s->dev_rows) * s->dev_cols * sizeof(float),
          std::memory_order_relaxed);
    }
  }
  // (Cold arm only — the fast path above serves the warmed f32 shadow
  // in-region. Refuse a capture-time miss BEFORE the EnsureHost for the same
  // reason as the conv shadow: across a prefill role transition the slot is
  // device-current host-stale, and downloading during capture is the
  // trace-refused read that aborts the process instead of naming the
  // condition.)
  VT_CHECK(!tt_capture_active(),
           "tenstorrent: GDN cache shadow not warmed during capture — the "
           "cold step must establish it first");
  EnsureHost(t);  // host truth (downloads a stale foreign shadow if any)
  const int64_t n = t.Numel();
  std::vector<float> host(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) host[static_cast<size_t>(i)] = LoadElemF32(t, i);
  GdnStateH2dBytes().fetch_add(static_cast<uint64_t>(n) * sizeof(float),
                               std::memory_order_relaxed);
  ttnn::Tensor dev = UploadTensor(std::move(host),
                                  ttnn::Shape({static_cast<uint32_t>(rows) * sf,
                                               ublk}),
                                  ttnn::DataType::FLOAT32, layout, device);
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(t.data);
  if (s != nullptr) {
    s->device = dev;
    s->dev_rows = static_cast<uint32_t>(rows);
    s->dev_cols = static_cast<uint32_t>(cols);
    s->device_current = true;
    s->host_current = true;
    s->device_reserved = false;  // real bytes staged — the reservation is spent
    // The re-upload replaces the transposed shadow with this role's logical
    // [rows, cols] split layout — the slot is no longer transposed, and a
    // stale flag would make the equal-volume refusal above fire on the next
    // ordinary serve (BufferSlot: every shadow-replacing commit restates the
    // layout).
    s->conv_transposed = false;
  }
  return dev;
}

// The [rows, slots] one-hot f32 matrix for `idx` (idx<0 = NULL row → zero
// row): G@X gathers rows — each row independently reads its own slot, so
// duplicates are fine. Only the DECODE kernel still uses it (a [batch, slots]
// matrix is far smaller than GatherRowsExact's [batch, row-width] index on
// the wide state rows); its matmul rounds the gathered state to tf32, which
// sits inside the decode step's own tf32 envelope. Exact paths use
// GatherRowsExact/ScatterRowsExact above.
//
// `factor` adapts the matrix to the SPLIT shadow geometry
// (EnsureGdnCacheDevice): the result is [rows*factor, slots*factor] with
// gmat[(b*F+j)][(s*F+j')] = 1 iff s == idx[b] and j == j', so each gathered
// block-row lands under its own batch row. Same values, same tf32 envelope.
ttnn::Tensor UploadOneHot(const std::vector<int32_t>& idx, int64_t slots,
                          MeshDevice& device, int64_t factor = 1) {
  const int64_t rows = static_cast<int64_t>(idx.size());
  const int64_t scols = slots * factor;
  std::vector<float> gmat(static_cast<size_t>(rows * factor) *
                              static_cast<size_t>(scols),
                          0.0f);
  for (int64_t r = 0; r < rows; ++r) {
    const int32_t ix = idx[static_cast<size_t>(r)];
    if (ix < 0) continue;  // NULL row: gathers zeros
    for (int64_t j = 0; j < factor; ++j)
      gmat[static_cast<size_t>(r * factor + j) * static_cast<size_t>(scols) +
           static_cast<size_t>(ix * factor + j)] = 1.0f;
  }
  return UploadTensor(std::move(gmat),
                      ttnn::Shape({static_cast<uint32_t>(rows * factor),
                                   static_cast<uint32_t>(scols)}),
                      ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
}

// The gdn_decode-side entry: the ssm one-hot gather and the split-block
// scatter ids, in the same content-keyed discipline.
GdnIdxCacheEntry GdnSsmIdxEntry(const std::vector<int32_t>& idxv,
                                uint32_t slots, uint32_t sf,
                                MeshDevice& device) {
  const std::array<uint64_t, 3> key{slots, sf, idxv.size()};
  std::lock_guard<std::mutex> g(GdnIdxCacheMutex());
  GdnIdxCacheEntry& e = GdnIdxCacheMap()[key];
  if (e.warmed_gdn && e.idx == idxv) return e;
  VT_CHECK(!tt_capture_active(),
           "tenstorrent gdn_decode: state-slot indices changed during trace "
           "capture — decode slots must be stable across a sequence's steps; "
           "the recapture cadence owns a slot change (reset the graph, then "
           "re-warm on the eager boundary step)");
  GdnIdxCacheEntry n;
  n.idx = idxv;
  n.warmed_gdn = true;
  n.oh = UploadOneHot(idxv, slots, device, sf);
  std::vector<uint32_t> blocks;
  blocks.reserve(idxv.size() * static_cast<size_t>(sf));
  for (int32_t ix : idxv)
    for (uint32_t j = 0; j < sf; ++j)
      blocks.push_back(static_cast<uint32_t>(ix) * sf + j);
  n.scatter_bid = UploadIdxU32(
      std::move(blocks), ttnn::Shape({static_cast<uint32_t>(idxv.size()) * sf}),
      ttnn::Layout::ROW_MAJOR, device);
  e = std::move(n);
  return e;
}

// The GQA head-map gather index for gdn_decode: rows[(b*Hv + h)*Dk + j] =
// b*Hk + h/ratio — the full-width u32 TILE form the pin's dim-0 gather takes
// (the same shape rule as GatherRowsExact's index and the conv out-select's
// idx_rows). Step-constant per (batch, hv, hk, dk), so the conv-tile cache
// discipline applies; the content is a pure function of the key, so no host
// bytes are kept, and the upload belongs to the eager cold step.
ttnn::Tensor CachedHeadMapIdx(uint64_t batch, uint64_t hv, uint64_t hk,
                              uint64_t dk, MeshDevice& device) {
  const std::array<uint64_t, 4> key{batch, hv, hk, dk};
  std::lock_guard<std::mutex> g(ConvTileCacheMutex());
  auto it = ConvTileCacheMap().find(key);
  if (it != ConvTileCacheMap().end()) return it->second.dev;
  VT_CHECK(!tt_capture_active(),
           "tenstorrent gdn_decode: head-map gather index not warmed — the "
           "cold step must build it before the captured region");
  const uint32_t rows_n = static_cast<uint32_t>(batch * hv);
  const uint64_t ratio = hv / hk;
  std::vector<uint32_t> rows(static_cast<size_t>(rows_n) * dk);
  for (uint64_t b = 0; b < batch; ++b)
    for (uint64_t h = 0; h < hv; ++h)
      for (uint64_t j = 0; j < dk; ++j)
        rows[static_cast<size_t>((b * hv + h) * dk + j)] =
            static_cast<uint32_t>(b * hk + h / ratio);
  ConvTileEntry e;
  e.dev = UploadIdxU32(std::move(rows),
                       ttnn::Shape({rows_n, static_cast<uint32_t>(dk)}),
                       ttnn::Layout::TILE, device);
  ConvTileCacheMap().emplace(key, std::move(e));
  return ConvTileCacheMap()[key].dev;
}

// The rank-1 gated-delta step for a gathered batch, state [B*Hv, Dv, Dk] f32
// on device: S *= exp(g); dot = S@k; v' = (v-dot)*beta; S += v'k^T;
// o = S@(q*scale). Returns {o [B*Hv, Dv, 1], S_new [B*Hv, Dv, Dk]} — the
// state stays on the card; only q/k/v/g/beta cross PCIe per step.
struct DecodeStepResult {
  ttnn::Tensor o;
  ttnn::Tensor state;
};

DecodeStepResult GdnDecodeStepComposed(const ttnn::Tensor& S, const ttnn::Tensor& dev_q,
                                       const ttnn::Tensor& dev_k, const ttnn::Tensor& dev_v,
                                       const ttnn::Tensor& dev_g, const ttnn::Tensor& dev_b,
                                       float scale, uint32_t bh, uint32_t dk, uint32_t dv) {
  // During capture, use the free ttnn::reshape: the program IS cached from
  // the eager warmup (GdnDecodeStepComposed runs the same reshapes in both
  // passes). create_program_artifacts only runs on cache miss, so the cache
  // hit path doesn't call to_device.
  ttnn::Tensor decay = ttnn::exp(ttnn::reshape(dev_g, ttnn::Shape({bh, 1, 1})));
  ttnn::Tensor Sd = ttnn::multiply(S, decay);
  ttnn::Tensor krow = ttnn::reshape(dev_k, ttnn::Shape({bh, 1, dk}));
  // f32-safe reductions: ttnn::matmul on this pin TRUNCATES f32 operands to
  // tf32 in its MACs on Blackhole, and the ComputeConfig does NOT lift it —
  // HiFi4 + fp32_dest_acc_en + math_approx off leaves the result bit-identical
  // to the plain call (probed on the wide-range fidelity arm's distribution:
  // matmul max_abs 0.53 vs a double reference, broadcast multiply + ttnn::sum
  // 4e-5). The truncated dot error lands in the small-magnitude elements of
  // the delta-rule correction (v - S@k) and compounds across the GDN layers —
  // pinned by the wide-range arm in tests/vt/test_tenstorrent_backend.cpp.
  // The eltwise multiply is exact f32 SFPU and ttnn::sum accumulates f32, so
  // the three GEMMs route through broadcast multiply + sum (the outer product
  // has no reduction at all — one exact product per element):
  ttnn::Tensor dot = ttnn::sum(ttnn::multiply(Sd, krow), ttsl::SmallVector<int>{2},
                               /*keep_dim=*/true);
  ttnn::Tensor vcol = ttnn::reshape(dev_v, ttnn::Shape({bh, dv, 1}));
  ttnn::Tensor beta = ttnn::reshape(dev_b, ttnn::Shape({bh, 1, 1}));
  ttnn::Tensor vp = ttnn::multiply(ttnn::subtract(vcol, dot), beta);
  ttnn::Tensor S2 = ttnn::add(Sd, ttnn::multiply(vp, krow));
  ttnn::Tensor qs = ttnn::multiply(dev_q, scale);
  ttnn::Tensor o = ttnn::sum(
      ttnn::multiply(S2, ttnn::reshape(qs, ttnn::Shape({bh, 1, dk}))),
      ttsl::SmallVector<int>{2}, /*keep_dim=*/true);
  return {std::move(o), std::move(S2)};
}

}  // namespace

// kGdnDecode (§7; cpu_ops.cpp GdnDecodeKernel). Both forms: compact
// state [B,Hv,Dv,Dk] (row == token) and the indexed FULL cache (slot per
// token, NULL idx<0 ⇒ out row zeroed, slot untouched). State lives in the
// device shadow across steps — the traffic counters prove no per-token
// round-trip. Two adapter choices, spec Design: the composed rank-1 step
// (default) and one T=1 chunk_gated_delta_rule call per step
// (VT_TT_GDN_DECODE=chunked diagnostic; the op pads T to its chunk multiple
// internally, so it pays the full-chunk cost for one token).
void GdnDecodeKernel(Queue&, Tensor& out, const Tensor& q_in, const Tensor& k,
                     const Tensor& v, const Tensor& g, const Tensor& beta,
                     Tensor& state, const Tensor* state_idx, const GdnArgs& args) {
  TT_OP_TRACE("GdnDecode");
  GdnDecodeSteps().fetch_add(1, std::memory_order_relaxed);
  const int64_t batch = q_in.shape[0];
  const int64_t hv = state.shape[1], dv = state.shape[2], dk = state.shape[3];
  const int64_t hk = q_in.shape[1];
  const int64_t ratio = hv / hk;
  const int64_t slots = state.shape[0];
  if (batch == 0) return;  // empty batch: no-op, state as-is
  bool has_null = false;
  std::vector<int32_t> idxv;
  if (state_idx != nullptr) {
    idxv = ReadIdxHost(*state_idx, slots, "gdn_decode", true);
    for (int32_t ix : idxv)
      if (ix < 0) has_null = true;
  }
  // NULL rows zero their out row and skip the state update — supported on
  // the eager arm through the host-staged validity mask. The captured arm
  // refuses them by name: the gated captured battery runs unpadded batches
  // (PadToCaptureSize(1) == 1), so the padded-capture arm stays a named owed
  // item rather than a silent divergence from the oracle's skip semantics.
  VT_CHECK(!has_null || !tt_capture_active(),
           "tenstorrent gdn_decode: NULL state rows (idx<0) under trace "
           "capture are the padded-capture arm this row owes — refusing "
           "rather than diverging from the oracle's skip semantics");

  MeshDevice& device = SharedMeshDevice();
  const uint32_t ub = static_cast<uint32_t>(batch), uhv = static_cast<uint32_t>(hv),
                  udk = static_cast<uint32_t>(dk), udv = static_cast<uint32_t>(dv),
                  bh = static_cast<uint32_t>(batch * hv);
  // Device-pure staging: the projections commit their outputs device-side in
  // every phase, so the activations are served from their current shadows
  // (ServeActF32; the eager arm keeps its host-staging fallback). q/k expand
  // to Hv heads — the hk head each value head maps to, exactly the CPU
  // GdnHeadTokenStep head map — through the proven dim-0 gather (pure data
  // movement, bit-identical, unlike the one-hot matmul's tf32 rounding); the
  // full-width u32 index is step-constant per geometry and warmed outside
  // capture, and ratio==1 needs no movement at all.
  ttnn::Tensor q2d = ServeActF32(q_in, static_cast<uint32_t>(batch * hk), udk,
                                 "q", device);
  ttnn::Tensor k2d =
      ServeActF32(k, static_cast<uint32_t>(batch * hk), udk, "k", device);
  ttnn::Tensor dev_q, dev_k;
  if (ratio == 1) {
    dev_q = std::move(q2d);
    dev_k = std::move(k2d);
  } else {
    ttnn::Tensor head_map =
        CachedHeadMapIdx(ub, uhv, static_cast<uint32_t>(hk), udk, device);
    dev_q =
        ttnn::gather(q2d, /*dim=*/0, head_map, /*sparse_grad=*/false, std::nullopt);
    dev_k =
        ttnn::gather(k2d, /*dim=*/0, head_map, /*sparse_grad=*/false, std::nullopt);
  }
  ttnn::Tensor dev_v = ServeActF32(v, bh, udv, "v", device);
  ttnn::Tensor dev_g = ServeActF32(g, bh, 1, "g", device);
  ttnn::Tensor dev_b = ServeActF32(beta, bh, 1, "beta", device);

  // The shadow is born split ([slots*F, blk], F = SplitFactor of the state
  // row) — see EnsureGdnCacheDevice / SplitFactor.
  const uint32_t sf =
      static_cast<uint32_t>(SplitFactor(hv * dv * dk, /*esz=*/4));
  ttnn::Tensor cache2d =
      EnsureGdnCacheDevice(state, slots, hv * dv * dk, device);
  std::optional<ttnn::Tensor> oh;
  GdnIdxCacheEntry sentry;
  ttnn::Tensor S;  // [B*Hv, Dv, Dk] — the batch's state rows, on device
  if (state_idx != nullptr) {
    // The one-hot gather matrix and the split-block scatter ids are warmed
    // outside capture (content-keyed: a slot change mid-capture is the
    // recapture cadence's, and the VT_CHECK inside names it).
    sentry = GdnSsmIdxEntry(idxv, slots, sf, device);
    oh = sentry.oh;
    S = CaptureSafeReshape(ttnn::matmul(*oh, cache2d),
                      ttnn::Shape({bh, udv, udk}));
  } else {
    S = CaptureSafeReshape(cache2d, ttnn::Shape({bh, udv, udk}));
  }

  const char* mode = std::getenv("VT_TT_GDN_DECODE");
  const bool chunked = mode != nullptr && std::string_view(mode) == "chunked";
  VT_CHECK(!chunked || !tt_capture_active(),
           "tenstorrent gdn_decode: the VT_TT_GDN_DECODE=chunked diagnostic "
           "stages its inputs host-side and cannot run inside a trace "
           "capture");
  ttnn::Tensor o;      // [B*Hv, Dv, 1] (composed) or [B, 1, Hv, Dv] (chunked)
  ttnn::Tensor S_new;  // [B*Hv, Dv, Dk] (composed) or [B, Hv, Dk, Dv] (chunked)
  if (chunked) {
    // One T=1 fused call on the whole batch (the alternative adapter, kept as
    // a diagnostic per the spec: pad/permute duties mirror GdnPrefillKernel).
    VT_CHECK(dk % 32 == 0 && dv % 32 == 0,
             "tenstorrent gdn_decode: the chunked adapter (VT_TT_GDN_DECODE="
             "chunked) needs Dk/Dv multiples of 32");
    ttnn::Tensor s_b =
        ttnn::reshape(S, ttnn::Shape({ub, uhv, udv, udk}));
    ttnn::Tensor s0 = ttnn::permute(s_b, ttsl::SmallVector<int64_t>{0, 1, 3, 2});
    // q/k [B,1,Hk,Dk] from the compact staged rows; v [B,1,Hv,Dv].
    std::vector<float> qc(static_cast<size_t>(batch) * hk * dk),
        kc(static_cast<size_t>(batch) * hk * dk), vc(static_cast<size_t>(batch) * hv * dv),
        gc(static_cast<size_t>(batch) * hv), bc(static_cast<size_t>(batch) * hv);
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t h = 0; h < hv; ++h) {
        for (int64_t j = 0; j < dk; ++j) {
          qc[static_cast<size_t>((b * hk + h / ratio) * dk + j)] =
              LoadElemF32(q_in, (b * hk + h / ratio) * dk + j);
          kc[static_cast<size_t>((b * hk + h / ratio) * dk + j)] =
              LoadElemF32(k, (b * hk + h / ratio) * dk + j);
        }
        for (int64_t j = 0; j < dv; ++j)
          vc[static_cast<size_t>((b * hv + h) * dv + j)] =
              LoadElemF32(v, (b * hv + h) * dv + j);
        gc[static_cast<size_t>(b * hv + h)] = LoadElemF32(g, b * hv + h);
        bc[static_cast<size_t>(b * hv + h)] = LoadElemF32(beta, b * hv + h);
      }
    }
    ttnn::Tensor dq = UploadTensor(
        std::move(qc),
        ttnn::Shape({ub, 1, static_cast<uint32_t>(hk), udk}),
        ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
    ttnn::Tensor dk4 = UploadTensor(
        std::move(kc),
        ttnn::Shape({ub, 1, static_cast<uint32_t>(hk), udk}),
        ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
    ttnn::Tensor dv4 = UploadTensor(std::move(vc),
                                    ttnn::Shape({ub, 1, uhv, udv}),
                                    ttnn::DataType::FLOAT32, ttnn::Layout::TILE,
                                    device);
    ttnn::Tensor dg = UploadTensor(std::move(gc), ttnn::Shape({ub, 1, uhv}),
                                   ttnn::DataType::FLOAT32, ttnn::Layout::TILE,
                                   device);
    ttnn::Tensor db = UploadTensor(std::move(bc), ttnn::Shape({ub, 1, uhv}),
                                   ttnn::DataType::FLOAT32, ttnn::Layout::TILE,
                                   device);
    auto [oc, fs] = ttnn::transformer::chunk_gated_delta_rule(
        dq, dk4, dv4, dg, db, args.scale, s0,
        /*output_final_state=*/true, /*chunk_size=*/64,
        /*use_qk_l2norm=*/false, /*output_head_major=*/false);
    VT_CHECK(fs.has_value(),
             "tenstorrent gdn_decode: chunk_gated_delta_rule returned no final state");
    o = std::move(oc);
    S_new = ttnn::permute(*fs, ttsl::SmallVector<int64_t>{0, 1, 3, 2});
    o = ttnn::reshape(o, ttnn::Shape({bh, udv, 1}));  // [B,1,Hv,Dv] -> [B*Hv,Dv,1]
    S_new = ttnn::reshape(S_new, ttnn::Shape({bh, udv, udk}));
  } else {
    DecodeStepResult r =
        GdnDecodeStepComposed(S, dev_q, dev_k, dev_v, dev_g, dev_b, args.scale, bh,
                              udk, udv);
    o = std::move(r.o);
    S_new = std::move(r.state);
  }

  if (has_null) {
    // NULL rows: the oracle zeroes the out row and skips the state update.
    std::vector<float> valid(static_cast<size_t>(bh));
    for (int64_t b = 0; b < batch; ++b) {
      const float live = idxv[static_cast<size_t>(b)] >= 0 ? 1.0f : 0.0f;
      for (int64_t h = 0; h < hv; ++h) valid[static_cast<size_t>(b * hv + h)] = live;
    }
    o = ttnn::multiply(o, UploadTensor(std::move(valid), ttnn::Shape({bh, 1, 1}),
                                       ttnn::DataType::FLOAT32, ttnn::Layout::TILE,
                                       device));
  }

  CommitDeviceLogical2D(
      out,
      CaptureSafeReshape(o, ttnn::Shape({ub, uhv, udv})),
      static_cast<uint32_t>(batch * hv), udv);

  // rows2d in the SPLIT shadow geometry ([B*F, blk]) — same flat bytes as
  // [B, Hv*Dv*Dk]; the regroup across the head boundary is one exact
  // data-movement program with per-tile circular buffers (only when F > 1).
  ttnn::Tensor rows2d = CaptureSafeReshape(
      S_new, ttnn::Shape({static_cast<uint32_t>(ub * sf),
                          static_cast<uint32_t>((uhv * udv * udk) / sf)}));
  // bf16 STORAGE semantics (SupportsCompressedGdnState, cuda_backend.cu): a
  // bf16 ssm_state is "read/written in f32 registers" — the state a step
  // STORES rounds through bf16, and the next step reads those bits. The
  // device shadow is f32, so round it in place on commit (RNE typecast
  // round-trip, zero PCIe). With a persistent f32 shadow instead, a
  // full-mantissa state value would re-enter unrounded and drift from the
  // CUDA contract within a few steps — pinned by the bf16-state arm of the
  // kGdnDecode oracle test (per-step outs must match the f32 path with
  // host-side bf16 round-trips bit-for-bit).
  if (state.dtype == DType::kBF16) {
    rows2d = ttnn::typecast(ttnn::typecast(rows2d, ttnn::DataType::BFLOAT16),
                            ttnn::DataType::FLOAT32);
  }
  if (oh.has_value()) {
    // Exact scatter (last-writer-wins on duplicate slots, NULL rows write
    // nothing) — the state bytes never round through tf32 on the way back.
    // In-region the block ids come from the warmed entry (the per-call
    // upload would be the enqueue_write the trace refuses; the entry's
    // content match plus the NULL refusal above make it the all-live form
    // ScatterRowsExact would build); eager keeps the per-call path, NULL
    // compaction included.
    // Use ScatterRowsDevice for BOTH eager and capture: the committed state
    // must have the same properties in both passes, so EnsureGdnCacheDevice
    // takes the same branch in the next layer. The eager warmup warms the
    // untilize/indexed_fill programs that capture needs.
    ttnn::Tensor newc =
        ScatterRowsDevice(cache2d, sentry.scatter_bid, rows2d, slots,
                            hv * dv * dk, sf);
    CommitDeviceLogical2D(state, std::move(newc), static_cast<uint32_t>(slots),
                          static_cast<uint32_t>(hv * dv * dk));
  } else {
    CommitDeviceLogical2D(state, std::move(rows2d), ub,
                          static_cast<uint32_t>(hv * dv * dk));
  }
}

// kGdnStateGather (cpu_ops.cpp GdnStateGatherKernel): the indexed
// persistent-cache boundary for GDN mixed prefill. Exact-dim caches gather
// through the one-hot matmul on the device shadow (the cache never fully
// downloads); a WIDENED cache row (spec taps) takes the host path on the
// shadowed bytes — a spec-decode-only arm, counted as d2h traffic.
void GdnStateGatherKernel(Queue&, Tensor& working, const Tensor& cache,
                          const Tensor& state_idx, const Tensor* has_initial_state) {
  TT_OP_TRACE("GdnStateGather");
  const int64_t rows = state_idx.shape[0];
  if (rows == 0) return;
  const int64_t slots = cache.shape[0];
  const int64_t work_inner = working.shape[working.rank - 1];
  const int64_t cache_inner = cache.shape[cache.rank - 1];
  const int64_t work_row = working.Numel() / rows;
  const int64_t mid = work_row / work_inner;
  const int64_t cache_row = mid * cache_inner;
  const std::vector<int32_t> idxv = ReadIdxHost(state_idx, slots, "gdn_state_gather", false);
  bool has_his = false;
  std::vector<float> keepv(static_cast<size_t>(rows), 1.0f);
  if (has_initial_state != nullptr) {
    EnsureHost(*has_initial_state);
    has_his = true;
    for (int64_t r = 0; r < rows; ++r) {
      const bool live = has_initial_state->dtype == DType::kI8
                            ? has_initial_state->Ptr<int8_t>()[r] != 0
                            : has_initial_state->Ptr<int32_t>()[r] != 0;
      keepv[static_cast<size_t>(r)] = live ? 1.0f : 0.0f;
    }
  }

  if (cache_inner != work_inner) {
    // Widened row: per-channel LEADING work_inner columns with the physical
    // stride — host path on the shadowed bytes (spec-decode arm).
    EnsureHost(cache);
    GdnStateD2hBytes().fetch_add(static_cast<uint64_t>(cache.Numel()) * sizeof(float),
                                 std::memory_order_relaxed);
    for (int64_t r = 0; r < rows; ++r)
      for (int64_t m = 0; m < mid; ++m) {
        const float* src = cache.Ptr<float>() +
                           static_cast<int64_t>(idxv[static_cast<size_t>(r)]) * cache_row +
                           m * cache_inner;
        float* dst = working.Ptr<float>() + r * work_row + m * work_inner;
        const float mul = has_his ? keepv[static_cast<size_t>(r)] : 1.0f;
        for (int64_t e = 0; e < work_inner; ++e) dst[e] = mul * src[e];
      }
    CommitHost(working);
    return;
  }

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor cache2d = EnsureGdnCacheDevice(cache, slots, cache_row, device);
  // Split-shadow geometry: expand the (NULL-free) row indices to their
  // block-rows — contiguous, in order — and the has_initial_state mask the
  // same way, so every gathered block-row keeps its own keep factor.
  const uint32_t sgf =
      static_cast<uint32_t>(SplitFactor(cache_row, /*esz=*/4));
  std::vector<int32_t> idx_blk;
  const std::vector<int32_t>* idxp = &idxv;
  if (sgf > 1) {
    idx_blk.reserve(idxv.size() * static_cast<size_t>(sgf));
    for (int32_t ix : idxv)
      for (uint32_t j = 0; j < sgf; ++j)
        idx_blk.push_back(ix * static_cast<int32_t>(sgf) +
                          static_cast<int32_t>(j));  // BLOCK-row id, not slot id
    idxp = &idx_blk;
    std::vector<float> keep_exp;
    keep_exp.reserve(keepv.size() * static_cast<size_t>(sgf));
    for (float kv : keepv) keep_exp.insert(keep_exp.end(), sgf, kv);
    keepv.swap(keep_exp);
  }
  ttnn::Tensor gathered = GatherRowsExact(cache2d, *idxp,
                                          cache_row / sgf, device);  // exact, no NULLs
  if (has_his) {
    gathered = ttnn::multiply(
        gathered,
        UploadTensor(std::move(keepv),
                     ttnn::Shape({static_cast<uint32_t>(rows) * sgf, 1}),
                     ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device));
  }
  CommitDeviceLogical2D(working, std::move(gathered), static_cast<uint32_t>(rows),
                        static_cast<uint32_t>(work_row));
}

// kGdnStateScatter (cpu_ops.cpp GdnStateScatterKernel): the inverse indexed
// store into the persistent cache. Exact-dim caches scatter through the
// exact indexed_fill on the device shadow; unnamed rows keep their bytes and
// the LAST row wins a duplicate slot (the oracle's loop order). A widened row
// takes the host path (spec-decode arm, counted).
void GdnStateScatterKernel(Queue&, Tensor& cache, const Tensor& working,
                           const Tensor& state_idx) {
  TT_OP_TRACE("GdnStateScatter");
  const int64_t rows = state_idx.shape[0];
  if (rows == 0) return;
  const int64_t slots = cache.shape[0];
  const int64_t work_inner = working.shape[working.rank - 1];
  const int64_t cache_inner = cache.shape[cache.rank - 1];
  const int64_t work_row = working.Numel() / rows;
  const int64_t mid = work_row / work_inner;
  const int64_t cache_row = mid * cache_inner;
  const std::vector<int32_t> idxv = ReadIdxHost(state_idx, slots, "gdn_state_scatter", false);

  if (cache_inner != work_inner) {
    EnsureHost(working);
    EnsureHost(cache);
    GdnStateD2hBytes().fetch_add(static_cast<uint64_t>(cache.Numel()) * sizeof(float),
                                 std::memory_order_relaxed);
    for (int64_t r = 0; r < rows; ++r)
      for (int64_t m = 0; m < mid; ++m) {
        const float* src = working.Ptr<float>() + r * work_row + m * work_inner;
        float* dst = cache.Ptr<float>() +
                     static_cast<int64_t>(idxv[static_cast<size_t>(r)]) * cache_row +
                     m * cache_inner;
        for (int64_t e = 0; e < work_inner; ++e) dst[e] = src[e];
      }
    CommitHost(cache);
    return;
  }

  MeshDevice& device = SharedMeshDevice();
  const uint32_t ssf =
      static_cast<uint32_t>(SplitFactor(cache_row, /*esz=*/4));
  ttnn::Tensor cache2d = EnsureGdnCacheDevice(cache, slots, cache_row, device);
  // working rows: reuse a resident f32 shadow when its element count and
  // dims match, else upload (and leave it unregistered — transient rows).
  // The upload carries the SPLIT shadow geometry ([rows*F, blk]); the flat
  // host bytes are unchanged.
  bool rows_resident = false;
  ttnn::Tensor rows2d;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(working.data);
    if (s != nullptr && s->device_current && s->device.has_value() &&
        s->device->dtype() == ttnn::DataType::FLOAT32 &&
        s->dev_rows == rows && s->dev_cols == work_row) {
      rows2d = *s->device;
      rows_resident = true;
    }
  }
  if (!rows_resident) {
    EnsureHost(working);
    std::vector<float> wh(static_cast<size_t>(working.Numel()));
    for (int64_t i = 0; i < working.Numel(); ++i)
      wh[static_cast<size_t>(i)] = LoadElemF32(working, i);
    rows2d = UploadTensor(
        std::move(wh),
        ttnn::Shape({static_cast<uint32_t>(rows) * ssf,
                     static_cast<uint32_t>(work_row / ssf)}),
        ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
  }
  ttnn::Tensor newc = ScatterRowsExact(cache2d, idxv, rows2d, slots,
                                       cache_row, ssf, device);
  CommitDeviceLogical2D(cache, std::move(newc), static_cast<uint32_t>(slots),
                        static_cast<uint32_t>(cache_row));
}

namespace {

struct GdnShadowData {
  std::optional<ttnn::Tensor> device;
  uint32_t dev_rows = 0, dev_cols = 0;
  bool device_current = false, conv_transposed = false;
};

}  // namespace

std::vector<GdnStateShadowSnapshot> SnapshotGdnStateShadows(
    const std::vector<const void*>& ptrs) {
  std::vector<GdnStateShadowSnapshot> out;
  out.reserve(ptrs.size());
  std::lock_guard<std::mutex> g(SlotMutex());
  for (const void* p : ptrs) {
    GdnShadowData data;
    BufferSlot* s = FindSlot(const_cast<void*>(p));
    if (s != nullptr) {
      data.device = s->device;
      data.dev_rows = s->dev_rows;
      data.dev_cols = s->dev_cols;
      data.device_current = s->device_current;
      data.conv_transposed = s->conv_transposed;
    }
    GdnStateShadowSnapshot snap;
    static_assert(sizeof(GdnShadowData) <= sizeof(snap.storage),
                  "GdnStateShadowSnapshot storage too small");
    new (snap.storage) GdnShadowData(std::move(data));
    out.push_back(snap);
  }
  return out;
}

void RestoreGdnStateShadows(
    const std::vector<const void*>& ptrs,
    const std::vector<GdnStateShadowSnapshot>& snapshots) {
  std::lock_guard<std::mutex> g(SlotMutex());
  for (size_t i = 0; i < ptrs.size() && i < snapshots.size(); ++i) {
    auto* data = reinterpret_cast<GdnShadowData*>(
        const_cast<char*>(snapshots[i].storage));
    BufferSlot* s = FindSlot(const_cast<void*>(ptrs[i]));
    if (s == nullptr) continue;
    s->device = data->device;
    s->dev_rows = data->dev_rows;
    s->dev_cols = data->dev_cols;
    s->device_current = data->device_current;
    s->conv_transposed = data->conv_transposed;
  }
}

// ---- BACKEND-TENSTORRENT-GDN W2: GDN state/conv-shadow traffic counters ----
// Readers for the tenstorrent_device.h API; the counters themselves live in
// tenstorrent_internal.h.
GdnShadowTraffic GetGdnShadowTraffic() {
  GdnShadowTraffic t;
  t.state_h2d_bytes = GdnStateH2dBytes().load(std::memory_order_relaxed);
  t.state_d2h_bytes = GdnStateD2hBytes().load(std::memory_order_relaxed);
  t.decode_steps = GdnDecodeSteps().load(std::memory_order_relaxed);
  return t;
}

void ResetGdnShadowTraffic() {
  GdnStateH2dBytes().store(0, std::memory_order_relaxed);
  GdnStateD2hBytes().store(0, std::memory_order_relaxed);
  GdnDecodeSteps().store(0, std::memory_order_relaxed);
}

}  // namespace vt::tenstorrent
