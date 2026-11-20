#!/usr/bin/env bash
# Build the W30 seam-level BFP8 staging repro against the pin's durable libs.
set -eu
DUR="$HOME/Sources/tt/tt-metal-pin-durable"
OUT="${1:-/tmp/repro-w30}"
mkdir -p "$OUT"
taskset -c 64-127 /usr/bin/c++ \
  -DDISABLE_NAMESPACE_STATIC_ASSERT -DFMT_HEADER_ONLY=1 -DSPDLOG_COMPILED_LIB \
  -DSPDLOG_FMT_EXTERNAL -DTRACY_ENABLE -DTRACY_IMPORTS -DTRACY_TIMER_FALLBACK \
  -isystem "$DUR/include" \
  -isystem "$DUR/libexec/tt-metalium/ttnn" \
  -isystem "$DUR/include/tracy" \
  -isystem "$DUR/libexec/tt-metalium/tt_metal/hostdevcommon/api" \
  -isystem "$DUR/libexec/tt-metalium" \
  -isystem "$DUR/include/metalium-thirdparty" \
  -O2 -DNDEBUG -std=c++20 -fPIC -ffp-contract=off -Wall -Wextra \
  -o "$OUT/bfp8_staging_repro" \
  "$(dirname "$0")/bfp8_staging_repro.cpp" \
  -L"$DUR/lib64" -Wl,-rpath,"$DUR/lib64" \
  "$DUR/lib64/_ttnncpp.so" "$DUR/lib64/libtt_metal.so" -ldl
echo "built $OUT/bfp8_staging_repro"
