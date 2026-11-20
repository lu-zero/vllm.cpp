// THE W30 SEAM-LEVEL BFP8 STAGING REPRO, VARIANT 3: the engine's RESIDENCY
// PATTERN. The flip-point scan (the W29 record) pins each corruption onset to
// a LATER staging install — the victim stays RESIDENT for ~77 installs of
// churn before it flips. Variant 1's alive window (2) can never see that.
// This variant: phase 1 installs a persistent resident set (like the engine's
// ~46 target weights); phase 2 runs the per-call staging churn — bf16 H2D
// temp + on-device typecast to BFLOAT8_B, both dropped with the call, the
// same-size-class blocks recycling every install — and scans EVERY resident
// weight after EVERY churn install.
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
struct Resident {
  ttnn::Tensor dev; uint32_t rows, cols; size_t last_clean; int installed; const char* tag;
};
}  // namespace

int main(int argc, char** argv) {
  const int residents = argc > 1 ? std::atoi(argv[1]) : 8;
  const int churn = argc > 2 ? std::atoi(argv[2]) : 120;
  auto device = ttnn::open_mesh_device(0, DEFAULT_L1_SMALL_SIZE, 0);
  device->enable_program_cache();
  std::fprintf(stderr, "[repro3] device open\n");
  const Shape shapes[] = {
      {6144, 5120, "z"}, {6144, 10240, "qkvz-mixed"},
      {6144, 5120, "outproj"}, {5120, 5120, "outproj2"},
  };
  std::mt19937 rng(0xC0FFEE);
  std::vector<Resident> res;
  size_t strikes = 0;

  auto install = [&](uint32_t r, uint32_t c) {
    const size_t n = static_cast<size_t>(r) * c;
    std::vector<bfloat16> host(n);
    for (size_t j = 0; j < n; ++j)
      host[j] = bfloat16(0.03125f * static_cast<float>(rng() % 4096) - 64.0f);
    ttnn::Tensor bf16_dev = ttnn::Tensor::from_span(
        ttsl::Span<const bfloat16>(host.data(), n), TileSpec(ttnn::DataType::BFLOAT16, r, c),
        device.get());
    ttnn::Tensor out = ttnn::typecast(bf16_dev, ttnn::DataType::BFLOAT8_B);
    return out;  // bf16 temp drops with the call
  };

  // Phase 1: the resident set.
  for (int i = 0; i < residents; ++i) {
    const Shape& s = shapes[i % 4];
    res.push_back(Resident{install(s.r, s.c), s.r, s.c, 0, i, s.tag});
    tt::tt_metal::Buffer* b = res.back().dev.mesh_buffer().get_reference_buffer();
    std::fprintf(stderr, "[repro3] resident %d %ux%u(%s) addr=0x%llx\n", i, s.r, s.c, s.tag,
                 (unsigned long long)b->address());
  }
  // Scan the residents clean.
  for (auto& a : res) {
    const auto [z, first] = ZeroHeaders(a.dev);
    if (z > 0)
      std::fprintf(stderr, "[STRIKE] phase1 resident(%s) installed=%d zero=%zu first=%zu\n",
                   a.tag, a.installed, z, first);
  }

  // Phase 2: the per-call staging churn (allocate -> typecast -> drop both).
  for (int i = 0; i < churn; ++i) {
    const Shape& s = shapes[i % 4];
    { ttnn::Tensor dead = install(s.r, s.c); }  // dropped with the scope
    for (auto& a : res) {
      const auto [z, first] = ZeroHeaders(a.dev);
      tt::tt_metal::Buffer* b = a.dev.mesh_buffer().get_reference_buffer();
      if (z > 0 && a.last_clean < static_cast<size_t>(i)) {
        ++strikes;
        std::fprintf(stderr,
                     "[STRIKE] churn=%d victim=%ux%u(%s) addr=0x%llx installed=%d last_clean=%zu "
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
      std::fprintf(stderr, "[repro3] churn %d strikes=%zu\n", i, strikes);
  }
  std::fprintf(stderr, "[repro3] DONE residents=%d churn=%d strikes=%zu\n", residents, churn,
               strikes);
  return strikes == 0 ? 0 : 1;
}
