#!/usr/bin/env bash
# QWEN38-DFLASH2-STAGE-C W26 gate: the BFP8 DETECT-AND-REPAIR form of the W25
# repair (Bfp8DetectRepair — re-stage only the weight whose 512 B exponent-
# header fingerprint clobbered; clean weights pay one KB-scale D2H probe).
#   (a) Q4_K_M KV64 drafter-ON x3: token identity vs the recorded d49c53d2...
#       + TPOT (the acceptance multiple). Leg 1 carries the acceptance trace
#       (VT_SPEC_ACCEPT_TRACE=4: the verify's target argmax + logits top-5 at
#       the first 4 spec steps) and the DNR cost trace (VT_TT_BFP8_DNR_TRACE).
#   (b) UD-Q6_K_XL KV8 drafter-ON x3: TPOT vs 5340.9.
#   (c) drafter-OFF x3 both arms: byte identity vs d49c53d2 / b36bc61a (the
#       fixed-tree streams; re-record only if the trace shows the old records
#       corrupt-tainted).
# Discipline per leg: file mutex, luwen reset + 15 s + /dev/shm cleanup,
# idle gate, taskset 32-63 bench / 64-127 builds, FROM THE REPO ROOT,
# INT8DOT=1, durable-libs-first, TT_METAL_RUNTIME_ROOT at the pin.
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
    echo "worktree: dirty (W26 BFP8 detect-and-repair, uncommitted)"
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
ON_TRACE_ENV="$ON_ENV VT_SPEC_ACCEPT_TRACE=4"

# Leg 1 carries the acceptance trace (the divergent-step instrument).
leg w26-on-1  "$Q4KM" 64 "$ON_TRACE_ENV" 24 1
leg w26-on-2  "$Q4KM" 64 "$ON_ENV"  24 1
leg w26-on-3  "$Q4KM" 64 "$ON_ENV"  24 1
leg w26-off-1 "$Q4KM" 64 "$OFF_ENV" 200 0
leg w26-off-2 "$Q4KM" 64 "$OFF_ENV" 200 0
leg w26-off-3 "$Q4KM" 64 "$OFF_ENV" 200 0
leg w26-q6kl-off-1 "$Q6KL" 8 "$OFF_ENV" 200 0
leg w26-q6kl-off-2 "$Q6KL" 8 "$OFF_ENV" 200 0
leg w26-q6kl-off-3 "$Q6KL" 8 "$OFF_ENV" 200 0
leg w26-q6kl-on-1  "$Q6KL" 8 "$ON_ENV"  64 1
leg w26-q6kl-on-2  "$Q6KL" 8 "$ON_ENV"  64 1
leg w26-q6kl-on-3  "$Q6KL" 8 "$ON_ENV"  64 1
echo "=== ALL LEGS DONE $(date -Is) ==="
