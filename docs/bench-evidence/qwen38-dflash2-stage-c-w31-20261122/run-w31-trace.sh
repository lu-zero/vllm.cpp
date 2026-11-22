#!/usr/bin/env bash
# QWEN38-DFLASH2-STAGE-C W31 diagnostic: the in-engine launch-address trace.
# The W29/W30 facts: the writer is on-device, no host write path, no seam
# repro. The W31 instrument (tt-metal-pin patchset, TT_METAL_TA_DEBUG=1,
# build build_instr) prints, at EVERY cached-program launch, the DRAM address
# patched into each tensor binding ([TT-TA] kernel= param= addr=) and, once
# per unbound kernel, the CRTA words a never-patched kernel retains (its
# permanent stale addresses). The engine repro is the W28 canary leg
# (VT_TT_BFP8_RESTAGE=0 + VT_TT_BFP8_CANARY=1, Q4_K_M KV64 drafter-ON,
# output-len 12): strikes are deterministic at #52/#58/#61 and the struck
# buffer's base address is printed by the canary itself, so the TA stream
# can be matched against the victim windows offline:
#   grep -n "W28-CANARY.*verdict=ARRIVED-ZEROED"  -> flip positions
#   the [TT-TA] addr= values near/preceding the flip that equal the victim
#   base belong to (a) the victim's own matmul and (b) any EARLIER program
#   holding the same address — the stale writer, caught by address.
# Leg 2 adds TT_METAL_RTA_DEBUG=1: every legacy SetRuntimeArgs value
# ([TT-RTA] kernel= core= args:...) — the launch-address trace (leg 1,
# w31-trace-1) showed NO patched binding ever lands in the victim window
# near the flip, so the writer must carry its address in a legacy RTA or a
# retained CRTA beyond the first 16 words (the dump is widened to 64).
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
  echo "pin: $(git -C "$HOME/Sources/tt/tt-metal-pin" rev-parse HEAD) + dirty TA dump"
  echo "bench: $BENCH"
  echo "model: $Q4KM"
  echo "model sha256: $(sha256sum "$Q4KM" | cut -d' ' -f1)"
  echo "extra-env: VLLM_CPP_CUDAGRAPH=0 VT_TT_WEIGHT_RESIDENCY=bfp8 VT_DFLASH_DRAFT=1 VT_TT_GEMM_OUT_KEEPALIVE=0 VT_TT_BFP8_RESTAGE=0 VT_TT_BFP8_CANARY=1 VT_TT_BFP8_CANARY_DIR=/tmp/tt-w31-canary TT_METAL_TA_DEBUG=1"
  echo "output-len: 12"
} > "$EV/$name-env.txt"
( cd "$REPO" && env VLLM_CPP_CUDAGRAPH=0 VT_TT_WEIGHT_RESIDENCY=bfp8 \
    VT_DFLASH_DRAFT=1 VT_TT_GEMM_OUT_KEEPALIVE=0 VT_TT_BFP8_RESTAGE=0 \
    VT_TT_BFP8_CANARY=1 VT_TT_BFP8_CANARY_DIR=/tmp/tt-w31-canary \
    TT_METAL_TA_DEBUG=1 TT_METAL_RTA_DEBUG=1 \
    taskset -c 32-63 "$BENCH" \
    --model "$Q4KM" --speculative-config "$SPEC" \
    --num-prompts 1 --input-len 32 --output-len 12 --concurrency 1 \
    --num-blocks 64 --max-num-batched-tokens 64 --seed 0 \
    --output-token-ids "$EV/$name-token-ids.json" \
    > "$EV/$name.log" 2> "$EV/$name.stderr.log" )
echo "=== $name done exit=$? sha256=$(sha256sum "$EV/$name-token-ids.json" 2>/dev/null | cut -d' ' -f1) ==="
