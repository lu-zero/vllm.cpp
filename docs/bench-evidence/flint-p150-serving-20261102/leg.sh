#!/bin/bash
# One vllm-bench leg on the P150, full device recipe. Args: NAME CONC OUTLEN REPS DEVICE_ATTN(1|0)
set -x
NAME=$1; CONC=$2; OUTLEN=$3; REPS=$4; DATTN=${5:-1}
exec 9>$HOME/gpu.lock
flock -x 9
/home/lu_zero/Sources/tt/luwen/target/release/reset
sleep 15
rm -f /dev/shm/TT_UMD_LOCK.* /dev/shm/tt_device_*_memory
echo "governor-before: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
export LD_LIBRARY_PATH=/tmp/umdtrial-install/lib64:/tmp/umdtrial-install/libexec/tt-metalium
export TT_METAL_RUNTIME_ROOT=/tmp/umdtrial-install/libexec/tt-metalium
export TT_METAL_HOME=/home/lu_zero/Sources/tt/tt-metal-pin
export VT_FLINT_KEEP_QUANT=1
export VLLM_CPP_CPU_THREADS=32
export VT_FLINT_TT_DEVICE_ATTN=$DATTN
cd /tmp/p150bench-logs
for rep in $(seq 1 "$REPS"); do
  while true; do
    LOAD=$(cut -d' ' -f1 /proc/loadavg)
    BUSY=$(ps aux --sort=-%cpu | awk 'NR>1 && $3>50 {print $11}' | head -1)
    if awk -v l="$LOAD" 'BEGIN{exit !(l<16)}' && [ -z "$BUSY" ]; then break; fi
    echo "idle-gate: load=$LOAD busy=$BUSY waiting"; sleep 30
  done
  echo "=== REP $rep $NAME conc=$CONC outlen=$OUTLEN device_attn=$DATTN load=$(cut -d' ' -f1 /proc/loadavg) ==="
  RSS_BEFORE=$(ps -o rss= -p $$)
  /usr/bin/time -v taskset -c 32-63 /tmp/build-p150bench/examples/vllm-bench \
    --model /mnt/models/flint-GGUF/flint-Q4_K_M.gguf \
    --num-prompts "$CONC" --input-len 32 --output-len "$OUTLEN" \
    --concurrency "$CONC" --seed 0 --temperature 0 --ignore-eos \
    2>&1 | tee "$NAME-rep$rep.log"
  echo "governor-after: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
done
