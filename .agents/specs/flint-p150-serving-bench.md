# SPEC — `FLINT-P150-SERVING-BENCH` — the first real Flint P150 serving benchmark: vllm-bench at serving shapes on the current head

- Owning row: `FLINT-P150-SERVING-BENCH`
- Kind: benchmark (measurement; the first serving-shape number for Flint on the P150)
- Issue: created in the same flow (row-scoped local issue)

## Why this row exists

Flint on the P150 has never had a proper serving benchmark. The only
number is FLINT-P150-ATTENTION stage 3's trace — a 32-token prompt + 6
decode steps, device-vs-host attention roughly neutral (53.0 vs 55.2
s/step) — taken BEFORE the pool-match fix, the NEON dot, the sliding
decomposition, and the slot-serve cure landed. The step is MoE-dominated
and the model is 2.8B-active on device; any tok/s claim before a real
serving run would be an invention.

The numerics track is complete and adopted: the single-layer probe is
64/64 per-head slices (both window modes), the dense-MLP staging defect
is cured (fd15425a1), the model-level gate holds (host 2.198e-1 /
device 2.200e-1, allocation-order-insensitive), and device attention is
the DEFAULT. The bench runs against an adopted, gated configuration —
this is a measurement, not an experiment.

## Deliverable

1. **Serving throughput**: vllm-bench (the house serving bench) on
   flint-Q4_K_M.gguf at serving shapes — 200 and 400 completion
   requests, concurrency 1 and 4, input 32 tokens — 3-rep medians,
   idle-gated, claims file, governor logged, device recipe per the
   standing discipline (mutex, reset, envs, substrate unmodified).
2. **Device vs host attention**: the same shapes with
   `VT_FLINT_TT_DEVICE_ATTN=0` (opt-out) vs default — the stage-3
   neutral finding re-tested at serving shapes with everything landed.
3. **The record**: docs/benchmarks/flint-p150-serving.md (new public
   benchmark ID, the number, the recipe, the artifact pins) +
   docs/BENCHMARKS.md index row + docs/bench-evidence/
   flint-p150-serving-20261102/ raw logs. Ceiling-free: the number is
   recorded against the MoE-dominated step profile, with the named
   levers (the residual 1.77x bisect row; the batch-1 quantized
   follow-ups; the history-dependence finding) — none adopted.

## The recipe

- VT_FLINT_KEEP_QUANT=1, pool matched to the taskset (USAGE matched-pool
  section), taskset -c 32-63, VT_FLINT_TT_DEVICE_ATTN default (=1) for
  the device arm; =0 for the host-attention comparison cell.
- Idle gate: 1-min load < 16 + ps aux --sort=-%cpu before every measured
  leg. Claims-file RUNNING line for the row, closed DONE at end.
- Governor: log as found.
- Device recipe (mandatory): mutex exec 9>$HOME/gpu.lock; flock -x 9;
  reset ~/Sources/tt/luwen/target/release/reset + sleep 15 + rm -f
  /dev/shm/TT_UMD_LOCK.* /dev/shm/tt_device_*_memory; env
  LD_LIBRARY_PATH=/tmp/umdtrial-install/lib64:/tmp/umdtrial-install/libexec/tt-metalium
  TT_METAL_RUNTIME_ROOT=/tmp/umdtrial-install/libexec/tt-metalium
  TT_METAL_HOME=/home/lu_zero/Sources/tt/tt-metal-pin. Substrate
  unmodified. NO timeout guards anywhere.

## Gates

check-agent-record, check-benchmark-index, check-env-doc, style,
trailers, preflight (named host dotnet failure); test_flint_paged
208/208; the single-layer probe; the W0 model gate at the adopted
default (green per #3542's gate case).

## Stop conditions

- vllm-bench cannot run the flint TT arm on device (throws at load or
  forward) → record the refusal, STOP: that is a capability gap for the
  TT arm's serving path, filed as its own row — the benchmark cannot
  invent the path.
- The number is so slow it cannot complete 200 completions in a
  reasonable session → record the partial number with its completion
  count and the ms/step extrapolation (the stage-3 numbers suggest
  ~9 s/step at batch 1 — ~0.11 tok/s — so a 200-completion run is
  ~30 minutes; that is FINE and recordable with the per-leg gates).
  Slow is a result, not a failure.

## Git integration

One pull request, `row/FLINT-P150-SERVING-BENCH`, spec committed first.
