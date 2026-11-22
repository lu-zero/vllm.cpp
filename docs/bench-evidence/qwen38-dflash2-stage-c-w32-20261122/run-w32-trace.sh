#!/usr/bin/env bash
# QWEN38-DFLASH2-STAGE-C W32 diagnostic: the host command-construction trace.
# The W31 fact: the writer carries the victim address on NO kernel launch path
# (no patched binding, no legacy RTA, no retained CRTA). The W32 instrument
# (tt-metal-pin patchset, TT_METAL_CMD_TRACE=1) prints one ordered [TT-CMD]
# line at the host moment every DRAM-noc-address-bearing dispatch command is
# constructed (buffer write txns: interleaved paged + sharded linear in
# tt_metal/impl/buffers/dispatch.cpp) and at every ttnn mesh-trace
# capture/replay/release transition (ttnn/operations/trace.cpp), plus
# alloc/free events for every DRAM buffer (write_trace register_dram). The
# repro is the deterministic W28 canary leg; the canary's own stderr lines
# interleave with [TT-CMD] lines, so a victim-window write command constructed
# BEFORE the victim's alloc (i.e. while the staging-temp owned the address)
# and replayed later is caught by address + seq ordering.
set -u
EV="$(cd "$(dirname "$0")" && pwd)"
BENCH=/tmp/build-stagec-w31/examples/vllm-bench
REPO=/tmp/tt-w31
Q4KM=/mnt/models/unsloth-qwen3.8-27B-gguf/Qwen3.8-27B-Q4_K_M.gguf
DRAFT=/mnt/models/qwen38-dflash2-drafter
SPEC='{"method":"dflash","model":"'"$DRAFT"'","num_speculative_tokens":8}'
export LD_LIBRARY_PATH="$HOME/Sources/tt/tt-metal-pin/build_Release/lib64:$HOME/Sources/tt/tt-metal-pin-durable/lib64"
export TT_METAL_RUNTIME_ROOT="$HOME/Sources/tt/tt-metal-pin"
export TT_METAL_HOME="$HOME/Sources/tt/tt-metal-pin"
export VT_TT_KEEPQUANT_INT8DOT=1 INT8DOT=1

name="$1"
exec 9>"${GPU_LOCK:-$HOME/gpu.lock}"; flock -x 9
"$HOME/Sources/tt/luwen/target/release/reset" >/dev/null 2>&1; sleep 15
rm -f /dev/shm/TT_UMD_LOCK.* /dev/shm/tt_device_*_memory 2>/dev/null
while [ "$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)" -ge 16 ]; do sleep 60; done
{
  echo "date: $(date -Is)"
  echo "head: $(git -C "$REPO" rev-parse HEAD)"
  echo "pin: $(git -C "$HOME/Sources/tt/tt-metal-pin" rev-parse HEAD)"
  echo "bench: $BENCH"
  echo "model: $Q4KM"
  echo "model sha256: $(sha256sum "$Q4KM" | cut -d' ' -f1)"
  echo "extra-env: VLLM_CPP_CUDAGRAPH=0 VT_TT_WEIGHT_RESIDENCY=bfp8 VT_DFLASH_DRAFT=1 VT_TT_GEMM_OUT_KEEPALIVE=0 VT_TT_BFP8_RESTAGE=0 VT_TT_BFP8_CANARY=1 VT_TT_BFP8_CANARY_DIR=/tmp/tt-w32-canary TT_METAL_CMD_TRACE=1 TT_METAL_WRITE_TRACE=1"
  echo "output-len: 12"
} > "$EV/$name-env.txt"
( cd "$REPO" && env VLLM_CPP_CUDAGRAPH=0 VT_TT_WEIGHT_RESIDENCY=bfp8 \
    VT_DFLASH_DRAFT=1 VT_TT_GEMM_OUT_KEEPALIVE=0 VT_TT_BFP8_RESTAGE=0 \
    VT_TT_BFP8_CANARY=1 VT_TT_BFP8_CANARY_DIR=/tmp/tt-w32-canary \
    TT_METAL_CMD_TRACE=1 \
    taskset -c 32-63 "$BENCH" \
    --model "$Q4KM" --speculative-config "$SPEC" \
    --num-prompts 1 --input-len 32 --output-len 12 --concurrency 1 \
    --num-blocks 64 --max-num-batched-tokens 64 --seed 0 \
    --output-token-ids "$EV/$name-token-ids.json" \
    > "$EV/$name.log" 2> "$EV/$name.stderr.log" )
echo "=== $name done exit=$? sha256=$(sha256sum "$EV/$name-token-ids.json" 2>/dev/null | cut -d' ' -f1) ==="
