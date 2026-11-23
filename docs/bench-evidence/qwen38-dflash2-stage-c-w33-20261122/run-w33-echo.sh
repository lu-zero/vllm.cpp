#!/usr/bin/env bash
# QWEN38-DFLASH2-STAGE-C W33 diagnostic: the device-arg echo.
# The W29/W31/W32 facts: the staged-BFP8 clobber writes one BFP8 tile-0
# exponent header per DRAM bank (strikes at calls #52/#58/#61, victim bases
# 0x231d6f00 / 0x2a146f00 / 0x2d7cef00) and NO host-side address carrier —
# no host write path, no kernel launch arg (RTA or retained CRTA), no
# constructed dispatch write command — ever holds a victim-window address.
# The remaining candidate is a DEVICE-RESIDENT arg/descriptor base that the
# host-side dumps cannot see (they print host values only). The W33
# instrument (tt-metal-pin, TT_METAL_DEVECHO=1) reads back each worker core's
# L1 kernel-config (RTA/CRTA) region at the launch's ring slot and prints
# [TT-DEVECHO] lines with every word in a plausible DRAM address range, so a
# stale never-re-patched device base appears as a victim-window address no
# host trace contains. Reads are deferred to the next enqueue (see the pin
# comment). TT_METAL_TA_DEBUG=1 stays on so the host-patched binding
# addresses interleave for correlation. The repro is the deterministic W28
# canary leg (VT_TT_BFP8_RESTAGE=0 + VT_TT_BFP8_CANARY=1, Q4_K_M KV64
# drafter-ON, output-len 12): strikes at #52/#58/#61.
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
  echo "extra-env: VLLM_CPP_CUDAGRAPH=0 VT_TT_WEIGHT_RESIDENCY=bfp8 VT_DFLASH_DRAFT=1 VT_TT_GEMM_OUT_KEEPALIVE=0 VT_TT_BFP8_RESTAGE=0 VT_TT_BFP8_CANARY=1 VT_TT_BFP8_CANARY_DIR=/tmp/tt-w33-canary TT_METAL_DEVECHO=1 TT_METAL_DEVECHO_EVERY=256 TT_METAL_TA_DEBUG=1"
  echo "output-len: 12"
} > "$EV/$name-env.txt"
( cd "$REPO" && env VLLM_CPP_CUDAGRAPH=0 VT_TT_WEIGHT_RESIDENCY=bfp8 \
    VT_DFLASH_DRAFT=1 VT_TT_GEMM_OUT_KEEPALIVE=0 VT_TT_BFP8_RESTAGE=0 \
    VT_TT_BFP8_CANARY=1 VT_TT_BFP8_CANARY_DIR=/tmp/tt-w33-canary \
    TT_METAL_DEVECHO=1 TT_METAL_DEVECHO_EVERY=256 TT_METAL_TA_DEBUG=1 \
    taskset -c 32-63 "$BENCH" \
    --model "$Q4KM" --speculative-config "$SPEC" \
    --num-prompts 1 --input-len 32 --output-len 12 --concurrency 1 \
    --num-blocks 64 --max-num-batched-tokens 64 --seed 0 \
    --output-token-ids "$EV/$name-token-ids.json" \
    > "$EV/$name.log" 2> "$EV/$name.stderr.log" )
echo "=== $name done exit=$? sha256=$(sha256sum "$EV/$name-token-ids.json" 2>/dev/null | cut -d' ' -f1) ==="
