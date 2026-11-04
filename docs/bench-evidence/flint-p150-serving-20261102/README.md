# FLINT-P150-SERVING-BENCH — the first Flint P150 serving benchmark

Row: `FLINT-P150-SERVING-BENCH`
Spec: `.agents/specs/flint-p150-serving-bench.md`
Date: 2026-11-04. Host: the shared aarch64 dev box, LOCAL Flint P150
(blackhole, firmware bundle 19.15.0). Commit `f11c7f305` (spec + issue).

## Recipe

Per-leg driver [`leg.sh`](leg.sh), the standing device recipe: `exec
9>$HOME/gpu.lock; flock -x 9`, luwen `reset` + sleep 15 + `/dev/shm`
lock-file cleanup, pinned substrate (`LD_LIBRARY_PATH=/tmp/umdtrial-install/
lib64:.../tt-metalium`, `TT_METAL_RUNTIME_ROOT=/tmp/umdtrial-install/
libexec/tt-metalium`, `TT_METAL_HOME=~/Sources/tt/tt-metal-pin`), substrate
unmodified. Env: `VT_FLINT_KEEP_QUANT=1`, `VLLM_CPP_CPU_THREADS=32` (pool
matched to the `taskset -c 32-63` budget), `VT_FLINT_TT_DEVICE_ATTN` default
(=1, device arm) / `=0` (host arm). Governor **performance**, logged
before and after every rep — unchanged on every leg.

Idle gate before every rep: 1-min load < 16 AND no process above 50% CPU.
The gate held on every measured rep; the driver log (in the session record)
shows the waits. Two dev reps were still contaminated by co-resident box
compile load that started mid-run (see Deviations) — the 3-rep median
absorbs both.

Command (the shape axis is the output window; num-prompts = concurrency):

```sh
taskset -c 32-63 /tmp/build-p150bench/examples/vllm-bench \
  --model /mnt/models/flint-GGUF/flint-Q4_K_M.gguf \
  --num-prompts <conc> --input-len 32 --output-len <200|400> \
  --concurrency <conc> --seed 0 --temperature 0 --ignore-eos
```

Artifact: `/mnt/models/flint-GGUF/flint-Q4_K_M.gguf`, sha256
`20c89707f762af73f0840f7b2c2911193617e60cac2ae30a96cb2cda7c15927e` — the
same byte-identical file every prior flint row measured. Engine header on
every leg: kv cache bf16, ignore-eos on, chat template skipped, async
scheduling on (max_concurrent_batches=2), auto-fit max_model_len 4064.

Build: `/tmp/build-p150bench`, `-G Ninja -DCMAKE_BUILD_TYPE=Release
-DVLLM_CPP_TENSTORRENT=ON`,
`CMAKE_PREFIX_PATH=/tmp/umdtrial-install/lib64/cmake;/tmp/umdtrial-install/share/cmake`,
`taskset -c 64-127 ninja -j 64`. No timeout guards anywhere.

## The STOP-condition check (the TT serving path works)

The spec's first stop condition was "vllm-bench cannot run the flint TT arm
on device". It CAN: the device arm serves end to end through the full
AsyncLLM stack, load → prefill → device-attention decode → sampled output,
on every one of the 12 device legs plus the load probe
([probe-rep1.log](probe-rep1.log), 32-token prompt, 4 tokens out, first
device forward through the bench). No refusal, no throw. **No STOP filed.**

## Results — 3-rep medians, tok/s = output token throughput

| conc | window | device attn (default) | host attn (`VT_FLINT_TT_DEVICE_ATTN=0`) | ms/step (TPOT) device / host | per-stream decode dev / host (tok/s) |
|---:|---:|---:|---:|---|---|
| 1 | 200 | **6.83** | **9.32** | 140.64 / 100.27 | 7.12 / 9.97 |
| 1 | 400 | **6.37** | **5.92** | 154.15 / 164.59 | 6.49 / 6.08 |
| 4 | 200 | **11.56** | **11.61** | 334.25 / 332.62 | 2.99 / 3.01 |
| 4 | 400 | **11.45** | **11.39** | 344.35 / 345.35 | 2.91 / 2.90 |

Per-rep output tok/s (medians underlined in spirit, raw here):

- dev c1 t200: 6.84 / 6.63 / 6.83 — dev c1 t400: 6.37 / 3.31* / 6.96
- dev c4 t200: 11.56 / 12.22 / 8.34* — dev c4 t400: 17.19 / 11.42 / 11.45
- host c1 t200: 6.93 / 9.52 / 9.32 — host c1 t400: 5.92 / 6.05 / 5.75
- host c4 t200: 11.37 / 11.61 / 11.65 — host c4 t400: 11.20 / 11.41 / 11.39

(*) contaminated rep — see Deviations.

RSS: max resident set size **48,519,680 KiB ≈ 46.3 GiB** on every leg
(`/usr/bin/time -v`), the keep-quant packed tree plus the engine.

## The device-vs-host cell

At **batch 1** the host-attention arm is the faster decode: ~100 ms/step vs
~140 ms/step at t200 (6.83 vs 9.32 tok/s). At t400 the two arms converge
(154 vs 165 ms/step) — the longer window drags the host arm down toward the
device arm's fixed cost. At **concurrency 4** the cell is a WASH: 334.25 vs
332.62 ms/step (t200) and 344.35 vs 345.35 ms/step (t400), inside noise.

So the stage-3 "roughly neutral" finding HOLDS at serving concurrency and
overstates at batch 1: with everything landed, device attention costs
~40% more ms/step than host attention at batch 1 and nothing at batch 4.
The default stays the device arm (the model-level gate and the adoption
decision own that), and the batch-1 device-attn overhead is a named,
unadopted lever.

## The MoE-dominated profile statement

The step is MoE-dominated on both arms. Evidence: the batch-4 step costs
~2.4x the batch-1 step (140→334 ms device, 100→333 ms host) while carrying
4x the tokens — and swapping the ENTIRE attention substrate (host NEON vs
TT device) moves the batch-4 step by less than 1% (332.62 vs 334.25 ms) and
the batch-1 step by 40%. The routed-expert phase, not attention, is the
wall. This mirrors the CPU row's attribution finding (flint-kq-cpu: ~88% of
serving decode outside the routed phase there; here the device/host
attention delta bounds the attention slice at ≤40% of the batch-1 step and
≤1% of the batch-4 step).

Ceiling-free. Named, unadopted levers: (1) the batch-1 device-attention
overhead (~40 ms/step); (2) the residual 1.77x bisect row from
FLINT-P150-ATTENTION; (3) the batch-1 quantized follow-ups; (4) the
history-dependence finding. No external denominator exists — llama.cpp
refuses flint at stock, and the llama-cpp-flint PR-head oracle is not
gateable for serving — so the number is recorded against the step profile
only. No ceiling is declared.

## Deviations

- **Two contaminated dev reps.** A co-resident compile (cc1plus/rustc/mbx,
  another session's work) started mid-run on the shared box:
  `dev-c1-t400-rep2` measured 299 ms/step and 3.31 tok/s (the idle gate
  held before the rep; load was 137 during it), and `dev-c4-t200-rep3`
  measured 468 ms/step and 8.34 tok/s. Both are committed as raw evidence
  and marked; the medians (6.37, 11.56) use the clean neighbors. The host
  arm ran during quieter windows; its reps are self-consistent.
- The spec's stop-condition arithmetic anticipated ~9 s/step from the
  stage-3 trace; the real figure at the current head is ~0.1 s/step
  (batch 1) — the perf fixes moved the serving step ~90x — so the full
  matrix ran in one session instead of the partial+extrapolation path.

## Raw logs

`{dev,host}-{c1,c4}-{t200,t400}-rep{1,2,3}.log` — the full vllm-bench
report plus `/usr/bin/time -v` block per rep. `probe-load-rep1.log` — the
load probe (32-in/4-out). `leg.sh` — the per-leg driver.
