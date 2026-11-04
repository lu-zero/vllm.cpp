# Flint on the Flint P150, the first serving benchmark

The RecursiveSI Flint Q4_K_M k-quant container served on the LOCAL Flint P150
(blackhole) through the full AsyncLLM stack with the TT device attention
adopted default: `VT_FLINT_KEEP_QUANT=1` packs the container, the linear
layers run the keep-quant dispatch, attention runs on the device. Measured
2026-11-04, commit `f11c7f305`; container sha256
`20c89707f762af73f0840f7b2c2911193617e60cac2ae30a96cb2cda7c15927e`; firmware
bundle 19.15.0; governor `performance`; idle-gated (1-min load < 16 and no
process above 50% CPU before every rep); 3-rep medians. Full recipe, per-rep
raw numbers, and the contamination notes:
[bench-evidence/flint-p150-serving-20261102](../bench-evidence/flint-p150-serving-20261102/README.md).

Correctness context: the device arm is the ADOPTED default — the single-layer
probe is 64/64 per-head slices, the dense-MLP staging defect is cured, and
the model-level gate holds (host 2.198e-1 / device 2.200e-1, allocation-order-
insensitive) per `FLINT-P150-ATTENTION` / `FLINT-TT-STAGE2-GATE`. This row is
a measurement on that adopted configuration, not an experiment.

## Results — output tok/s, 3-rep medians

| concurrency | window | device attn (default) | host attn (`VT_FLINT_TT_DEVICE_ATTN=0`) | ms/step dev / host |
|---:|---:|---:|---:|---|
| 1 | 200 | **6.83** | **9.32** | 140.6 / 100.3 |
| 1 | 400 | **6.37** | **5.92** | 154.2 / 164.6 |
| 4 | 200 | **11.56** | **11.61** | 334.3 / 332.6 |
| 4 | 400 | **11.45** | **11.39** | 344.4 / 345.4 |

Peak RSS 46.3 GiB every leg. Input 32 tokens, greedy, ignore-eos, pool
matched to the `taskset -c 32-63` budget (`VLLM_CPP_CPU_THREADS=32`).

Recipe:

```sh
VT_FLINT_KEEP_QUANT=1 VLLM_CPP_CPU_THREADS=32 taskset -c 32-63 \
  build/examples/vllm-bench --model /mnt/models/flint-GGUF/flint-Q4_K_M.gguf \
  --num-prompts <conc> --input-len 32 --output-len <200|400> \
  --concurrency <conc> --seed 0 --temperature 0 --ignore-eos
```

under the standing P150 device recipe (gpu.lock mutex, luwen reset, pinned
tt-metalium substrate) — the same recipe as the `tt-*` P150 rows.

## Findings

- **The TT serving path works.** The device arm serves end to end through
  the bench on every leg — the spec's stop condition (a refusal at load or
  forward) did not fire, so no capability gap was filed.
- **The stage-3 s/step figure was stale by ~90x.** The only prior number
  (53.0 vs 55.2 s/step, taken before every perf fix) suggested ~9 s/step;
  the real batch-1 serving step at the current head is ~0.1 s/step.
- **Device-vs-host attention: a wash at serving concurrency, a batch-1
  cost.** At concurrency 4 the arms are within 1% ms/step of each other in
  both windows. At batch 1 the device arm costs ~40% more ms/step (140.6 vs
  100.3 at t200). The default stays the device arm; the batch-1 overhead is
  a named, unadopted lever.
- **The step is MoE-dominated.** Swapping the entire attention substrate
  moves the batch-4 step by <1% and the batch-1 step by 40%; the routed
  expert phase is the wall on both arms.
- **No external denominator.** llama.cpp refuses flint at stock, so there
  is no floor; the number is recorded against the step profile, ceiling-free,
  with the named levers: the batch-1 device-attention overhead, the residual
  1.77x bisect row, the batch-1 quantized follow-ups, and the
  history-dependence finding. No ceiling is declared.
