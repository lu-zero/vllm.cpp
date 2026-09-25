// repro_scatter_web_leak.cpp — minimal reproducer for the per-request DRAM
// retention named by the TT-LEDGER instrument on 2026-09-25 (vllm.cpp row
// TT-METAL-RETENTION-ROOTCAUSE, spec tenstorrent-ttm-retention-rootcause.md).
//
// Mirrors src/vt/tenstorrent/tenstorrent_gdn.cpp ScatterRowsExact ->
// ScatterRowsDevice's op web (tenstorrent_gdn.cpp:1502-1503, :1476-1488):
// per call, ttnn::to_layout conversions + ttnn::indexed_fill +
// ttnn::experimental::view rank-4 views, ALL as function locals that die at
// scope exit (RAII), the same lifetime discipline the engine's web has.
// The 27B leg leaks one (16MiB + 3.75MiB) to_layout-output pair per GDN
// layer (48 layers: dense blocks 0,1,2 mod 4) on each request's FIRST decode
// step = ~948MiB/request (the committed ledger's -949.3MB/request), retained
// outside every vllm.cpp cache with the program cache FROZEN (0 inserts in
// request 3), OOM at bank_manager.cpp:495 after ~5 requests.
//
// Each loop iteration plays one "request": the only per-request input is a
// fresh slot index (the indexed_fill batch ids), exactly like the engine's
// per-request GDN state slot. Settled free DRAM is printed per iteration. If
// it steps down per iteration outside vllm.cpp, the retention holder is
// tt-metal/ttnn-internal.

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/tensor_ops.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/core.hpp>
#include <ttnn/device.hpp>
#include <ttnn/operations/core/to_layout/to_layout_op.hpp>
#include <ttnn/operations/experimental/reshape/view.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/memory_reporter.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt_stl/span.hpp>
#include <tt_stl/overloaded.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

// ttnn::indexed_fill is outside the installed include set at this pin; the
// declaration mirrors the source-tree signature 1:1 (the engine does the
// same, tenstorrent_internal.h:160; the symbol is exported by _ttnncpp.so).
namespace ttnn {
Tensor indexed_fill(const Tensor& batch_id, const Tensor& input_tensor_a,
                    const Tensor& input_tensor_b,
                    const std::optional<tt::tt_metal::MemoryConfig>& memory_config =
                        std::nullopt,
                    int64_t dim = 0);
}  // namespace ttnn

namespace {

tt::tt_metal::TensorSpec SpecOf(tt::tt_metal::Shape shape, ttnn::DataType dtype,
                                ttnn::Layout layout) {
  return tt::tt_metal::TensorSpec(
      std::move(shape),
      tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(layout),
                                 tt::tt_metal::MemoryConfig{}));
}

int64_t FreeDram(ttnn::MeshDevice& dev) {
  const auto view = tt::tt_metal::detail::GetMemoryView(
      &dev, tt::tt_metal::BufferType::DRAM);
  return static_cast<int64_t>(view.num_banks) *
         static_cast<int64_t>(view.total_bytes_free_per_bank);
}

}  // namespace

int main(int argc, char** argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 6;
  auto dev_ptr = ttnn::open_mesh_device(/*device_id=*/0,
                                        /*l1_small_size=*/DEFAULT_L1_SMALL_SIZE,
                                        /*trace_region_size=*/0);
  ttnn::MeshDevice& dev = *dev_ptr;
  dev.enable_program_cache();

  // Geometry in the leaked classes' size range: cache 16MiB, rows 3.75MiB
  // (f32 TILE), split factor 8 like the engine's SplitFactor discipline.
  const int64_t slots = 64;
  const int64_t cols = 32768;  // f32 elements per slot row
  const int64_t factor = 8;
  const uint32_t blk = static_cast<uint32_t>(cols / factor);
  const tt::tt_metal::TensorSpec cache_spec = SpecOf(
      tt::tt_metal::Shape({static_cast<uint32_t>(slots * factor), blk}),
      ttnn::DataType::FLOAT32, ttnn::Layout::TILE);
  const tt::tt_metal::TensorSpec rows_spec = SpecOf(
      tt::tt_metal::Shape({static_cast<uint32_t>(factor), blk}),
      ttnn::DataType::FLOAT32, ttnn::Layout::TILE);
  std::vector<float> zeros_cache(
      static_cast<size_t>(slots * factor) * blk, 0.0f);
  std::vector<float> zeros_rows(static_cast<size_t>(factor) * blk, 0.0f);
  ttnn::Tensor cache2d = ttnn::Tensor::from_vector<float>(
      std::vector<float>(zeros_cache), cache_spec, &dev);
  ttnn::Tensor rows2d = ttnn::Tensor::from_vector<float>(
      std::vector<float>(zeros_rows), rows_spec, &dev);
  printf("start: free=%lld MiB\n", (long long)(FreeDram(dev) >> 20));
  int64_t prev = FreeDram(dev);
  for (int r = 0; r < iters; ++r) {
    // The engine's step-0 branch: the request's fresh working-state rows have
    // NO resident device shadow (tenstorrent_gdn.cpp:1054-1064), so the scatter
    // uploads them via UploadTensor -> from_vector(TILE spec) — whose staging
    // buffers (the ROW_MAJOR intermediate + the tilize output) are transient
    // and unregistered. Mirror that per "request".
    ttnn::Tensor fresh_rows = ttnn::Tensor::from_vector<float>(
        std::vector<float>(zeros_rows), rows_spec, &dev);
    // The only other per-request input: a fresh slot index.
    std::vector<uint32_t> blocks;
    blocks.reserve(factor);
    for (int64_t j = 0; j < factor; ++j)
      blocks.push_back(static_cast<uint32_t>(r * factor + j));
    ttnn::Tensor bid_dev = ttnn::Tensor::from_vector<uint32_t>(
        std::move(blocks),
        SpecOf(tt::tt_metal::Shape({static_cast<uint32_t>(factor)}),
               ttnn::DataType::UINT32, ttnn::Layout::ROW_MAJOR),
        &dev);
    // The ScatterRowsDevice web, verbatim order and shapes.
    ttnn::Tensor src = ttnn::to_layout(fresh_rows, cache2d.layout());
    ttnn::Tensor cache_rm =
        ttnn::to_layout(cache2d, ttnn::Layout::ROW_MAJOR);
    ttnn::Tensor src_rm = ttnn::to_layout(src, ttnn::Layout::ROW_MAJOR);
    const uint32_t unb = static_cast<uint32_t>(bid_dev.logical_shape()[0]);
    ttnn::Tensor out4 = ttnn::indexed_fill(
        bid_dev,
        ttnn::experimental::view(
            cache_rm,
            ttnn::Shape{static_cast<uint32_t>(slots * factor), 1, 1, blk}),
        ttnn::experimental::view(src_rm, ttnn::Shape{unb, 1, 1, blk}),
        std::nullopt, /*dim=*/0);
    ttnn::Tensor out2 = ttnn::experimental::view(
        out4, ttnn::Shape{static_cast<uint32_t>(slots * factor), blk});
    ttnn::Tensor newc = ttnn::to_layout(out2, ttnn::Layout::TILE);
    (void)newc;
    // Every web local dies here, exactly like the engine's scope exit.
    dev.mesh_command_queue().finish();
    const int64_t free = FreeDram(dev);
    printf("iter %d: free=%lld MiB (delta %+lld MiB)\n", r,
           (long long)(free >> 20), (long long)((free - prev) >> 20));
    prev = free;
  }
  return 0;
}
