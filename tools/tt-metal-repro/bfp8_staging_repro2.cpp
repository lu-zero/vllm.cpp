// THE W30 SEAM-LEVEL BFP8 STAGING REPRO, VARIANT 2: the pure install loop of
// bfp8_staging_repro.cpp PLUS the engine's interleaved state — a matmul at the
// spec geometry (M=9) through the JUST-installed weight, a program-cache-warmed
// op mix (assign D2D masters, zeros temps, bf16 temps of varying sizes for
// allocator churn). Bisect step toward the engine state when the pure loop is
// clean (issue ISSUE-LOCAL-01M7FY5W1A91XHV1PZDGZEKR7H).
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
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/creation/creation.hpp>
#include <ttnn/operations/data_movement/copy/copy.hpp>

namespace {
tt::tt_metal::TensorSpec TileSpec(ttnn::DataType dt, uint32_t rows, uint32_t cols) {
  return tt::tt_metal::TensorSpec(
      tt::tt_metal::Shape({rows, cols}),
      tt::tt_metal::TensorLayout(dt, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                                 tt::tt_metal::MemoryConfig{}));
}
constexpr size_t kHdrBytes = 64;
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
struct Shape { uint32_t r, c; const char* tag; };
struct Install {
  ttnn::Tensor dev; uint32_t rows, cols; size_t last_clean; int installed; const char* tag;
};
}  // namespace

int main(int argc, char** argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 120;
  const size_t alive_max = argc > 2 ? std::atoi(argv[2]) : 3;
  auto device = ttnn::open_mesh_device(0, DEFAULT_L1_SMALL_SIZE, 0);
  device->enable_program_cache();
  std::fprintf(stderr, "[repro2] device open\n");
  const Shape shapes[] = {
      {6144, 5120, "z"}, {6144, 10240, "qkvz-mixed"},
      {6144, 5120, "outproj"}, {5120, 5120, "outproj2"},
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
    alive.push_back(Install{std::move(out), s.r, s.c, static_cast<size_t>(i), i, s.tag});
    if (alive.size() > alive_max) alive.erase(alive.begin());

    // The engine's interleaved op mix at the spec geometry.
    if (i % 2 == 0) {
      // M=9 matmul through the freshest install (the verify forward's arm).
      const uint32_t K = alive.back().rows;
      const size_t an = 9u * K;
      std::vector<bfloat16> ah(an);
      for (size_t j = 0; j < an; ++j)
        ah[j] = bfloat16(0.01f * static_cast<float>(rng() % 100));
      ttnn::Tensor a_dev = ttnn::Tensor::from_span(
          ttsl::Span<const bfloat16>(ah.data(), ah.size()),
          TileSpec(ttnn::DataType::BFLOAT16, 9, K), device.get());
      (void)ttnn::matmul(a_dev, alive.back().dev);
    }
    if (i % 5 == 0) {
      // The D2D master clone (the engine's lazy masters).
      ttnn::Tensor m = ttnn::assign(alive.back().dev, alive.back().dev.memory_config(),
                                    alive.back().dev.dtype(), std::nullopt);
      // hold briefly then drop — churn
      const auto [zm, f0] = ZeroHeaders(m);
      (void)zm; (void)f0;
    }
    if (i % 7 == 0) {
      // Varying-size bf16 temp allocations (the embed / KV churn).
      const uint32_t rr = 32u * (1 + (i % 5)), cc = 256u * (1 + (i % 3));
      ttnn::Tensor t = ttnn::zeros(tt::tt_metal::Shape({rr, cc}), ttnn::DataType::BFLOAT16,
                                   tt::tt_metal::Layout::TILE, std::ref(*device));
    }
    for (auto& a : alive) {
      const auto [z, first] = ZeroHeaders(a.dev);
      tt::tt_metal::Buffer* b = a.dev.mesh_buffer().get_reference_buffer();
      if (z > 0 && a.last_clean < static_cast<size_t>(i)) {
        ++strikes;
        std::fprintf(stderr,
                     "[STRIKE] scan=%d victim=%ux%u(%s) addr=0x%llx installed=%d last_clean=%zu "
                     "zero_pages=%zu first_zero_page=%zu size=%llu page=%llu\n",
                     i, a.rows, a.cols, a.tag, (unsigned long long)b->address(), a.installed,
                     a.last_clean, z, first, (unsigned long long)b->size(),
                     (unsigned long long)b->page_size());
        a.last_clean = i;
      } else if (z == 0) {
        a.last_clean = i;
      }
    }
    if (i % 10 == 0)
      std::fprintf(stderr, "[repro2] install %d alive=%zu strikes=%zu\n", i, alive.size(), strikes);
  }
  std::fprintf(stderr, "[repro2] DONE iters=%d strikes=%zu\n", iters, strikes);
  return strikes == 0 ? 0 : 1;
}
