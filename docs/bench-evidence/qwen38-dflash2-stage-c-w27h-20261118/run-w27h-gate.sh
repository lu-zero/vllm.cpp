#!/usr/bin/env bash
# QWEN38-DFLASH2-STAGE-C W27h gate: the host-drafter-arm legs. Same leg
# discipline as run-w27-gate2.sh (mutex exec 9>$GPU_LOCK; luwen reset; 15 s;
# /dev/shm cleanup; idle gate; bench on cpus 32-63).
#   (1) w27h-off-gaps-full  Q4_K_M drafter-OFF, 200 tok, VT_EAGER_TOP2_TRACE=220
#       — the eager reference stream + per-position gaps for the tie-band gate,
#       and the drafter-OFF byte-identity check vs the W26 OFF records.
#   (2) w27h-on-full        Q4_K_M drafter-ON-HOST, 200 tok, accept+step traces
#       — THE ACCEPTANCE MULTIPLE leg and the tie-band ON stream.
#   (3) w27h-corr           planted corruption, DNR disabled, host arm ON, 24 tok
#       — the red proof: tie-band gate MUST fail.
#   (4) w27h-q6kl           UD-Q6_K_XL drafter-ON-HOST, 24 tok — the capacity
#       verdict with the draft off-device (the W26 blocker was its 3.85 GiB).
set -u
EV="$(cd "$(dirname "$0")" && pwd)"
BENCH=/tmp/build-stagec-w3/examples/vllm-bench
REPO=/tmp/tt-hunt
Q4KM=/mnt/models/unsloth-qwen3.8-27B-gguf/Qwen3.8-27B-Q4_K_M.gguf
Q6KL=/mnt/models/unsloth-qwen3.8-27B-gguf-v2/Qwen3.8-27B-UD-Q6_K_XL.gguf
DRAFT=/mnt/models/qwen38-dflash2-drafter
SPEC='{"method":"dflash","model":"'"$DRAFT"'","num_speculative_tokens":8}'
export LD_LIBRARY_PATH="$HOME/Sources/tt/tt-metal-pin-durable/lib64:/tmp/umdtrial-install/lib64:/tmp/umdtrial-install/libexec/tt-metalium"
export TT_METAL_RUNTIME_ROOT="$HOME/Sources/tt/tt-metal-pin"
export TT_METAL_HOME="$HOME/Sources/tt/tt-metal-pin"
export VT_TT_KEEPQUANT_INT8DOT=1 INT8DOT=1

leg() {  # $1 tag, $2 model, $3 blocks, $4 extra env, $5 output-len, $6 spec(1/0)
  local name="$1"
  echo "=== $name start $(date -Is) ==="
  exec 9>"${GPU_LOCK:-$HOME/gpu.lock}"; flock -x 9
  "$HOME/Sources/tt/luwen/target/release/reset" >/dev/null 2>&1; sleep 15
  rm -f /dev/shm/TT_UMD_LOCK.* /dev/shm/tt_device_*_memory 2>/dev/null
  while [ "$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)" -ge 16 ]; do sleep 60; done
  {
    echo "date: $(date -Is)"
    echo "head: $(git -C "$REPO" rev-parse HEAD)"
    echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
    echo "load1: $(cut -d' ' -f1 /proc/loadavg)"
    echo "bench: $BENCH"
    echo "model: $2"
    echo "model sha256: $(sha256sum "$2" | cut -d' ' -f1)"
    echo "num-blocks: $3"
    echo "extra-env: $4"
    echo "output-len: $5"
    echo "spec: $6"
  } > "$EV/$name-env.txt"
  ps aux --sort=-%cpu | head -12 > "$EV/ps-top-$name.txt"
  local spec_args="" env_extra="$4"
  [ "$6" = "1" ] && spec_args="--speculative-config $SPEC"
  ( cd "$REPO" && env $env_extra taskset -c 32-63 "$BENCH" \
      --model "$2" $spec_args \
      --num-prompts 1 --input-len 32 --output-len "$5" --concurrency 1 \
      --num-blocks "$3" --max-num-batched-tokens 64 --seed 0 \
      --output-token-ids "$EV/$name-token-ids.json" \
      > "$EV/$name.log" 2> "$EV/$name.stderr.log" )
  local rc=$?
  echo "=== $name end $(date -Is) exit=$rc sha256=$(sha256sum "$EV/$name-token-ids.json" 2>/dev/null | cut -d' ' -f1) ==="
}

OFF_ENV="VT_TT_GEMM_OUT_KEEPALIVE=0"
ONH_ENV="VLLM_CPP_CUDAGRAPH=0 VT_TT_WEIGHT_RESIDENCY=bfp8 VT_DFLASH_DRAFT=1 VT_DFLASH_HOST=1 VT_TT_GEMM_OUT_KEEPALIVE=0 VT_TT_BFP8_DNR_TRACE=1 VT_STEP_WALL=1 VT_SPEC_TRACE=1"
ONH_TRACE_ENV="$ONH_ENV VT_SPEC_ACCEPT_TRACE=220"
GAPS_ENV="$OFF_ENV VT_EAGER_TOP2_TRACE=220"
CORR_ENV="$ONH_TRACE_ENV VT_TT_BFP8_RESTAGE=0 VT_TT_BFP8_PLANT_HEADERS_ZERO=1"

leg w27h-off-gaps-full "$Q4KM" 64 "$GAPS_ENV"     200 0
leg w27h-corr          "$Q4KM" 64 "$CORR_ENV"      24 1
leg w27h-q6kl          "$Q6KL"  8 "$ONH_TRACE_ENV" 24 1
leg w27h-on-full       "$Q4KM" 64 "$ONH_TRACE_ENV" 200 1
echo "=== ALL LEGS DONE $(date -Is) ==="
