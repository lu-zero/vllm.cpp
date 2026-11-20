#!/usr/bin/env bash
# QWEN38-DFLASH2-STAGE-C W29 diagnostic: the write-path attribution leg.
# The W28 canary ON plus the pin's TT_METAL_WRITE_TRACE (live DRAM extent
# tracking + host write classification at WriteToDeviceDRAMChannel and the
# CQ issue path): whoever writes the 512 zero header bytes at the struck
# weight's address either shows up as a host write into a dead/unknown
# range ([TT-WTRACE][MISS] + backtrace) or is on-device (no host write in
# the strike window). DNR OFF (VT_TT_BFP8_RESTAGE=0).
set -u
EV="$(cd "$(dirname "$0")" && pwd)"
BENCH=/tmp/build-ttfp/examples/vllm-bench
REPO=/tmp/tt-writer
Q4KM=/mnt/models/unsloth-qwen3.8-27B-gguf/Qwen3.8-27B-Q4_K_M.gguf
DRAFT=/mnt/models/qwen38-dflash2-drafter
SPEC='{"method":"dflash","model":"'"$DRAFT"'","num_speculative_tokens":8}'
export LD_LIBRARY_PATH="$HOME/Sources/tt/tt-metal-pin/build_Release/lib64:/tmp/umdtrial-install/lib64:/tmp/umdtrial-install/libexec/tt-metalium"
export TT_METAL_RUNTIME_ROOT="$HOME/Sources/tt/tt-metal-pin"
export TT_METAL_HOME="$HOME/Sources/tt/tt-metal-pin"
export VT_TT_KEEPQUANT_INT8DOT=1 INT8DOT=1

name="w29-attrib-1"
echo "=== $name start $(date -Is) ==="
exec 9>"${GPU_LOCK:-$HOME/gpu.lock}"; flock -x 9
"$HOME/Sources/tt/luwen/target/release/reset" >/dev/null 2>&1; sleep 15
rm -f /dev/shm/TT_UMD_LOCK.* /dev/shm/tt_device_*_memory 2>/dev/null
while [ "$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)" -ge 16 ]; do sleep 60; done
{
  echo "date: $(date -Is)"
  echo "head: $(git -C "$REPO" rev-parse HEAD)"
  echo "bench: $BENCH"
  echo "model: $Q4KM"
  echo "model sha256: $(sha256sum "$Q4KM" | cut -d' ' -f1)"
  echo "num-blocks: 64"
  echo "output-len: 12"
  echo "spec: 1"
} > "$EV/$name-env.txt"
ps aux --sort=-%cpu | head -12 > "$EV/ps-top-$name.txt"
rm -rf /tmp/tt-w29-canary; mkdir -p /tmp/tt-w29-canary
( cd "$REPO" && env VLLM_CPP_CUDAGRAPH=0 VT_TT_WEIGHT_RESIDENCY=bfp8 \
    VT_DFLASH_DRAFT=1 VT_TT_GEMM_OUT_KEEPALIVE=0 VT_TT_BFP8_RESTAGE=0 \
    VT_TT_BFP8_CANARY=1 VT_TT_BFP8_CANARY_DIR=/tmp/tt-w29-canary \
    TT_METAL_WRITE_TRACE=1 \
    taskset -c 32-63 "$BENCH" \
    --model "$Q4KM" --speculative-config "$SPEC" \
    --num-prompts 1 --input-len 32 --output-len 12 --concurrency 1 \
    --num-blocks 64 --max-num-batched-tokens 64 --seed 0 \
    --output-token-ids "$EV/$name-token-ids.json" \
    > "$EV/$name.log" 2> "$EV/$name.stderr.log" )
echo "=== $name end $(date -Is) exit=$? sha256=$(sha256sum "$EV/$name-token-ids.json" 2>/dev/null | cut -d' ' -f1) ==="
