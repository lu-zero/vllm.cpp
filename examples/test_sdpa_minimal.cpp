// Minimal C++ test: call sdpa_decode directly to check if 2-head bug reproduces
// in isolation (without the surrounding model forward ops).
//
// Build: cmake --build build --target vllm-cli (reuses the vllm.cpp build)
// Run: ./build/examples/test_sdpa_minimal
//
// This creates Q/KV/page_table/cur_pos with the exact same parameters as the
// model forward, calls sdpa_decode, and checks all 16 output heads.

#include "vt/backend.h"
// Disable Tracy for ttnn include chain (TracyC.h has 5-arg ___tracy_alloc_srcloc,
// ttnn headers expect 6-arg version).
#ifdef TRACY_ENABLE
#undef TRACY_ENABLE
#define VT_RESTORE_TRACY_ENABLE 1
#endif
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/operations/core/core.hpp>  // to_layout, to_device
#ifdef VT_RESTORE_TRACY_ENABLE
#define TRACY_ENABLE 1
#undef VT_RESTORE_TRACY_ENABLE
#endif

// Forward declare from tenstorrent_device.cpp
namespace tt::tt_metal::distributed { class MeshDevice; }
namespace vt::tenstorrent {
bool DeviceAvailable();
tt::tt_metal::distributed::MeshDevice& SharedMeshDevice();
}
#include <ttnn/operations/transformer/sdpa_decode/sdpa_decode.hpp>
#include <ttnn/operations/transformer/sdpa_config.hpp>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/work_split.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

using namespace tt::tt_metal;

int main() {
  setenv("TT_METAL_HOME", "/home/lu_zero/Sources/tt/tt-metal", 1);
  setenv("TT_METAL_RUNTIME_ROOT", "/home/lu_zero/Sources/tt/tt-metal", 1);

  auto& device = vt::tenstorrent::SharedMeshDevice();
  if (!vt::tenstorrent::DeviceAvailable()) {
    fprintf(stderr, "No Tenstorrent device available\n");
    return 1;
  }

  // Match Qwen3-0.6B dims
  const uint32_t B = 1;
  const uint32_t num_q_heads = 16;
  const uint32_t num_kv_heads = 8;
  const uint32_t head_dim = 128;
  const uint32_t block_size = 32;
  const uint32_t max_blocks = 2;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // Create Q: [1, B, num_q_heads, head_dim] with random data in ALL heads
  std::vector<float> q_host(B * num_q_heads * head_dim);
  srand(42);
  for (auto& v : q_host) v = static_cast<float>(rand()) / 2147483647.0f * 2.0f - 1.0f;

  // Create K/V: [max_blocks, num_kv_heads, block_size, head_dim]
  // Block 0 = zeros, Block 1 has data at offset 0 (matching our RAC write)
  std::vector<float> k_host(max_blocks * num_kv_heads * block_size * head_dim, 0.0f);
  std::vector<float> v_host(max_blocks * num_kv_heads * block_size * head_dim, 0.0f);
  for (uint32_t h = 0; h < num_kv_heads; h++) {
    for (uint32_t e = 0; e < head_dim; e++) {
      size_t off = (1 * num_kv_heads * block_size + h * block_size + 0) * head_dim + e;
      k_host[off] = static_cast<float>(rand()) / 2147483647.0f * 2.0f - 1.0f;
      v_host[off] = static_cast<float>(rand()) / 2147483647.0f * 2.0f - 1.0f;
    }
  }

  // page_table: [B, 2] = [[1, 0]] (virtual block 0 → physical block 1)
  std::vector<int32_t> pt_host = {1, 0};
  // cur_pos: [B] = [0]
  std::vector<int32_t> pos_host = {0};

  // Helper to create TILE device tensors
  auto make_rm_int = [&](const std::vector<int32_t>& data, const ttnn::Shape& shape) {
    return ttnn::Tensor::from_vector<int32_t>(data,
        tt::tt_metal::TensorSpec(shape, tt::tt_metal::TensorLayout(
            ttnn::DataType::INT32,
            tt::tt_metal::PageConfig(ttnn::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig{})),
        &device);
  };

  // Create tensors via from_vector ROW_MAJOR → to_device → to_layout(TILE)
  // (matching as_tensor's path)
  auto make_via_rm = [&](const std::vector<float>& data, const ttnn::Shape& shape) {
    ttnn::Tensor rm = ttnn::Tensor::from_vector<float>(data,
        tt::tt_metal::TensorSpec(shape, tt::tt_metal::TensorLayout(
            ttnn::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(ttnn::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig{})),
        nullptr);
    ttnn::Tensor dev = rm.to_device(&device, tt::tt_metal::MemoryConfig{});
    return ttnn::to_layout(dev, ttnn::Layout::TILE);
  };

  fprintf(stderr, "Creating tensors...\n");
  ttnn::Tensor dev_q = make_via_rm(q_host, ttnn::Shape({1u, B, num_q_heads, head_dim}));
  ttnn::Tensor dev_k = make_via_rm(k_host, ttnn::Shape({max_blocks, num_kv_heads, block_size, head_dim}));
  ttnn::Tensor dev_v = make_via_rm(v_host, ttnn::Shape({max_blocks, num_kv_heads, block_size, head_dim}));
  ttnn::Tensor dev_pt = make_rm_int(pt_host, ttnn::Shape({B, 2u}));
  ttnn::Tensor dev_pos = make_rm_int(pos_host, ttnn::Shape({B}));

  fprintf(stderr, "Q: logical=[%u,%u,%u,%u] padded=[%u,%u,%u,%u]\n",
          (unsigned)dev_q.logical_shape()[0], (unsigned)dev_q.logical_shape()[1],
          (unsigned)dev_q.logical_shape()[2], (unsigned)dev_q.logical_shape()[3],
          (unsigned)dev_q.padded_shape()[0], (unsigned)dev_q.padded_shape()[1],
          (unsigned)dev_q.padded_shape()[2], (unsigned)dev_q.padded_shape()[3]);
  fprintf(stderr, "K: logical=[%u,%u,%u,%u]\n",
          (unsigned)dev_k.logical_shape()[0], (unsigned)dev_k.logical_shape()[1],
          (unsigned)dev_k.logical_shape()[2], (unsigned)dev_k.logical_shape()[3]);
  fprintf(stderr, "PT: logical=[%u,%u]  POS: logical=[%u]\n",
          (unsigned)dev_pt.logical_shape()[0], (unsigned)dev_pt.logical_shape()[1],
          (unsigned)dev_pos.logical_shape()[0]);

  // Call sdpa_decode with explicit config (matching our model forward)
  fprintf(stderr, "Calling sdpa_decode (with explicit config)...\n");
  auto grid = device.compute_with_storage_grid_size();
  ttnn::operations::transformer::SDPAProgramConfig prog{
      grid, std::nullopt, /*q_chunk_size=*/32, /*k_chunk_size=*/32,
      /*exp_approx_mode=*/false, /*max_cores_per_head_batch=*/16};
  ttnn::DeviceComputeKernelConfig ck_cfg{
      tt::tt_metal::MathFidelity::HiFi4, false, false, false};

  ttnn::Tensor dev_out = ttnn::transformer::paged_scaled_dot_product_attention_decode(
      dev_q, dev_k, dev_v, dev_pt,
      /*is_causal=*/true, /*attn_mask=*/std::nullopt,
      /*cur_pos_tensor=*/dev_pos, /*attention_sink=*/std::nullopt,
      /*scale=*/scale, /*sliding_window_size=*/std::nullopt,
      /*memory_config=*/std::nullopt,
      /*program_config=*/prog, /*compute_kernel_config=*/ck_cfg,
      /*paged_cache_geometry=*/std::nullopt, /*cache_position_modulo=*/std::nullopt);

  auto out_vec = dev_out.to_vector<float>();
  fprintf(stderr, "Output size=%zu logical=[%u,%u,%u,%u]\n",
          out_vec.size(),
          (unsigned)dev_out.logical_shape()[0], (unsigned)dev_out.logical_shape()[1],
          (unsigned)dev_out.logical_shape()[2], (unsigned)dev_out.logical_shape()[3]);

  int non_zero = 0;
  for (uint32_t h = 0; h < num_q_heads; h++) {
    size_t off = static_cast<size_t>(h) * head_dim;
    float maxval = 0;
    for (size_t i = off; i < off + head_dim && i < out_vec.size(); i++)
      maxval = std::max(maxval, std::abs(out_vec[i]));
    if (maxval > 0.001f) non_zero++;
    if (h % 2 == 0)
      fprintf(stderr, "  head%u: max=%.4f %s\n", h, maxval, maxval > 0.001f ? "OK" : "ZERO");
  }
  fprintf(stderr, "Non-zero heads: %d/%u\n", non_zero, num_q_heads);

  if (non_zero == num_q_heads) {
    fprintf(stderr, "PASS: all heads have data\n");
    return 0;
  } else {
    fprintf(stderr, "FAIL: only %d of %u heads have data\n", non_zero, num_q_heads);
    return 1;
  }
}
