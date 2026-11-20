// THE W30 SEAM-LEVEL BFP8 STAGING REPRO (issue ISSUE-LOCAL-01M7FY5W1A91XHV1PZDGZEKR7H).
//
// Mirrors the engine's exact per-call BFP8 weight install
// (vt::tenstorrent Bfp8TypecastInstall, tenstorrent_residency.cpp): a bf16
// H2D temp via Tensor::from_span + an ON-DEVICE typecast to BFLOAT8_B, the
// temp dropped with the call, the BFP8 output kept only inside a small
// alive window (the engine's shadow map drops dead keys per call, so the
// same-size-class DRAM blocks recycle every install).
//
// After EVERY install it direct-reads every ALIVE weight and counts the
// BFP8_B tile exponent headers that are ZERO (the W25-W28 corruption map:
// 512 B at the first page of each of the 8 DRAM banks). A zero count > 0 on
// a previously clean weight IS the writer, caught outside vllm.cpp.
//
// Build (aarch64 host, the pin's durable libs):
//   See build.sh next to this file.
// Run: the usual device discipline (file mutex, luwen reset, idle gate).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <ttnn/device.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/types.hpp>
#include <ttnn/operations/copy/typecast/typecast.hpp>
#include <ttnn/operations/core/core.hpp>


namespace {

tt::tt_metal::TensorSpec TileSpec(ttnn::DataType dt, uint32_t rows, uint32_t cols) {
  return tt::tt_metal::TensorSpec(
      tt::tt_metal::Shape({rows, cols}),
      tt::tt_metal::TensorLayout(dt, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                                 tt::tt_metal::MemoryConfig{}));
}

constexpr size_t kHdrBytes = 64;    // one BFP8_B tile's exponent region
constexpr size_t kTileStride = 1088;  // BFP8_B tile = 64 B hdr + 1024 B data

// Direct full readback + zero exponent-header count (the engine's
// Bfp8DirectZeroHeaders). Returns <zero pages, first zero page index>.
std::pair<size_t, size_t> ZeroHeaders(const ttnn::Tensor& dev_b) {
  tt::tt_metal::Buffer* buf = dev_b.mesh_buffer().get_reference_buffer();
  std::vector<uint8_t> raw(buf->size());
  tt::tt_metal::detail::ReadFromBuffer(*buf, raw.data());
  const uint64_t page = buf->page_size();
  const uint64_t pages = buf->num_pages();
  size_t zero = 0, first = SIZE_MAX;
  for (uint64_t p = 0; p < pages && (p + 1) * page <= raw.size(); ++p) {
    const uint8_t* h = raw.data() + p * page;
    bool z = true;
    for (uint64_t i = 0; i < kHdrBytes && i < page; ++i)
      if (h[i] != 0) { z = false; break; }
    if (z) { ++zero; if (first == SIZE_MAX) first = p; }
  }
  return {zero, first};
}

struct Install {  // one alive staged weight
  ttnn::Tensor dev;
  uint32_t rows, cols;
  size_t last_clean_scan = 0;  // install index of the last zero-free scan
  int installed;
  const char* tag;
};

struct Shape { uint32_t r, c; const char* tag; };

}  // namespace

int main(int argc, char** argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 120;
  const size_t alive_max = argc > 2 ? std::atoi(argv[2]) : 2;

  // The engine's device open: 1x1 mesh, dynamic trace region.
  auto device = ttnn::open_mesh_device(
      /*device_id=*/0, /*l1_small_size=*/DEFAULT_L1_SMALL_SIZE,
      /*trace_region_size=*/0);
  device->enable_program_cache();
  std::fprintf(stderr, "[repro] device open\n");

  // The trio: the struck 6144x5120 GDN z-GEMM weight and the mixed-qkvz
  // shapes that interleave with it in the engine's staging sequence.
  const Shape shapes[] = {
      {6144, 5120, "z"},
      {6144, 10240, "qkvz-mixed"},
      {6144, 5120, "outproj"},
      {5120, 5120, "outproj2"},
  };

  std::mt19937 rng(0xC0FFEE);
  std::vector<Install> alive;
  size_t strikes = 0;

  for (int i = 0; i < iters; ++i) {
    const Shape& s = shapes[i % 4];
    const size_t n = static_cast<size_t>(s.r) * s.c;
    std::vector<bfloat16> host(n);
    for (size_t j = 0; j < n; ++j)
      host[j] = bfloat16(0.03125f * static_cast<float>(rng() % 4096) - 64.0f);
    ttnn::Tensor bf16_dev = ttnn::Tensor::from_span(
        ttsl::Span<const bfloat16>(host.data(), n), TileSpec(ttnn::DataType::BFLOAT16, s.r, s.c),
        device.get());
    ttnn::Tensor out = ttnn::typecast(bf16_dev, ttnn::DataType::BFLOAT8_B);
    // `bf16_dev` drops here (the engine's temp). Fold it to keep only `out`.
    auto tmp = std::move(bf16_dev);
    tt::tt_metal::Buffer* ob = out.mesh_buffer().get_reference_buffer();
    Install inst{std::move(out), s.r, s.c, static_cast<size_t>(i), i, s.tag};
    const auto [z0, first0] = ZeroHeaders(inst.dev);
    if (z0 > 0)
      std::fprintf(stderr,
                   "[STAGE] #%d %ux%u(%s) addr=0x%llx DIRTY-AT-STAGE zero=%zu\n",
                   i, s.r, s.c, s.tag, (unsigned long long)ob->address(), z0);
    alive.push_back(std::move(inst));
    if (alive.size() > alive_max) alive.erase(alive.begin());  // the recycle

    // The per-install scan of every ALIVE weight.
    for (auto& a : alive) {
      const auto [z, first] = ZeroHeaders(a.dev);
      tt::tt_metal::Buffer* b = a.dev.mesh_buffer().get_reference_buffer();
      if (z > 0 && a.last_clean_scan < static_cast<size_t>(i)) {
        ++strikes;
        std::fprintf(stderr,
                     "[STRIKE] scan=%d victim=%ux%u(%s) addr=0x%llx installed=%d "
                     "last_clean=%zu zero_pages=%zu first_zero_page=%zu size=%llu page=%llu\n",
                     i, a.rows, a.cols, a.tag, (unsigned long long)b->address(),
                     a.installed, a.last_clean_scan,
                     z, first, (unsigned long long)b->size(), (unsigned long long)b->page_size());
        a.last_clean_scan = i;  // report once per onset
      } else if (z == 0) {
        a.last_clean_scan = i;
      }
    }
    if (i % 10 == 0) std::fprintf(stderr, "[repro] install %d alive=%zu strikes=%zu\n", i,
                                  alive.size(), strikes);
  }
  std::fprintf(stderr, "[repro] DONE iters=%d strikes=%zu\n", iters, strikes);
  return strikes == 0 ? 0 : 1;
}
