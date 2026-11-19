#!/usr/bin/env bash
# QWEN38-DFLASH2-STAGE-C W27 gate, second run: the legs the terminated first
# run did not reach. Same leg discipline as run-w27-gate.sh.
#   (R) the corrupted-weights RED leg (planted corruption, DNR disabled) —
#       the tie-band gate MUST FAIL on it.
#   (a) Q4_K_M KV64 drafter-ON x2 more (with run 1's w27-on-1: the 3 reps).
#   (b) UD-Q6_K_XL KV8 drafter-ON x3 + its eager gap leg.
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
    echo "worktree: dirty (W27 tie-band gate + D2D master, uncommitted)"
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
ON_ENV="VLLM_CPP_CUDAGRAPH=0 VT_TT_WEIGHT_RESIDENCY=bfp8 VT_DFLASH_DRAFT=1 VT_TT_GEMM_OUT_KEEPALIVE=0 VT_TT_BFP8_DNR_TRACE=1"
ON_TRACE_ENV="$ON_ENV VT_SPEC_ACCEPT_TRACE=24"
GAPS_ENV="$OFF_ENV VT_EAGER_TOP2_TRACE=40"
CORR_ENV="$ON_ENV VT_SPEC_ACCEPT_TRACE=24 VT_TT_BFP8_RESTAGE=0 VT_TT_BFP8_PLANT_HEADERS_ZERO=1"

leg w27-on-corr       "$Q4KM" 64 "$CORR_ENV"      24 1
leg w27-on-2          "$Q4KM" 64 "$ON_ENV"        24 1
leg w27-on-3          "$Q4KM" 64 "$ON_TRACE_ENV"  24 1
leg w27-q6kl-off-gaps "$Q6KL" 8  "$GAPS_ENV"      24 0
leg w27-q6kl-on-1     "$Q6KL" 8  "$ON_TRACE_ENV"  24 1
leg w27-q6kl-on-2     "$Q6KL" 8  "$ON_ENV"        24 1
leg w27-q6kl-on-3     "$Q6KL" 8  "$ON_ENV"        24 1
echo "=== ALL LEGS DONE $(date -Is) ==="
