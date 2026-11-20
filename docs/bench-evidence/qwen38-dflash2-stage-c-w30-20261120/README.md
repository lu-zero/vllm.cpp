# QWEN38-DFLASH2-STAGE-C — W30 (2026-11-20): the seam-level minimal reproduction, the pin verdict, and the corruption-absent proof

Issue: `ISSUE-LOCAL-01M7FY5W1A91XHV1PZDGZEKR7H` (row `QWEN38-DFLASH2-STAGE-C`).
Wave: the W28/W29-proven facts (512 zero bytes at the 8 tile exponent headers,
on-device writer, host paths exonerated) driven to the named next instrument:
the seam-level repro outside vllm.cpp, the bisect toward the engine state, and
the pin patchset decision.

## The repro program

`tools/tt-metal-repro/bfp8_staging_repro{,2,3}.cpp` + `build.sh` (aarch64 host,
the pin's durable libs, `taskset -c 64-127` build / `32-63` run). Each variant
mirrors the engine's exact install
(`Bfp8TypecastInstall`, `tenstorrent_residency.cpp`: bf16 `Tensor::from_span`
H2D temp -> on-device `ttnn::typecast` to `BFLOAT8_B` -> the temp dropped with
the call) over the trio shapes (6144x5120 z, 6144x10240 mixed-qkvz, 6144x5120
outproj, 5120x5120) and direct-reads the full BFP8 buffer of every watched
weight after every install, counting zero exponent headers (the engine's
`Bfp8DirectZeroHeaders`).

| Leg | Program | State | Result |
|---|---|---|---|
| `leg-pure.log` | repro v1 | 120 installs, alive window 2, no other ops | **clean, strikes=0** |
| `leg-mix.log` | repro v2 | v1 + M=9 matmuls through the installs + D2D `assign` masters + varying-size bf16 temps, program cache warm | **clean, strikes=0** |
| `leg-resident.log` | repro v3 | 8 RESIDENT weights + 120 churn installs of dropped staging temps (the W29 flip-point pattern: the victim resident ~77 installs before it flips) | (below) |

## Verdict

**NOT isolated — engine-dependent.** All three seam-level variants are clean:
the pure install loop (120 installs), the program-cache-warm op-mix loop
(M=9 matmuls + D2D masters + temp churn), and the resident-plus-churn
pattern (8 residents x 120 churn installs — the W29 flip-point residency).
The corruption needs the engine's fuller load/warmup state (the ~25 GB
residency, the GGUF mmap sources, the drafter staging, the trace-region /
async-executor machinery); the W22 harness had already cleared the
embed/RAC/PA/DBuf states short of that. The writer's identity stays
UNFOUND, no pin patchset is possible without a mechanism, and no upstream
tt-metal issue is drafted — an issue without a repro is noise; the
three-leg bisect record is the artifact.

Consequence: the dispatch's "DNR disabled by default (the writer fixed)"
premise FAILED; the default flip was reverted in the same wave and the DNR
detect-and-repair stays DEFAULT-ON as the working mitigation
(`VT_TT_BFP8_RESTAGE` remains the rollback/regression knob). The gate suite
runs BOTH arms on the same binary.

## The gate suite

`run-w30-gate.sh`: (a) Q4_K_M KV64 drafter-ON x3 with the DNR machinery
DEFAULT-OFF (leg 1 carries the W28 canary — the corruption-absent proof),
token identity vs `d49c53d2...` + TPOT; (b) UD-Q6_K_XL KV8 drafter-ON x3 vs
`b36bc61a...` + TPOT; (c) drafter-OFF x3 both arms, byte identity. Discipline
per leg: file mutex, luwen reset + 15 s + shm cleanup, idle gate, INT8DOT=1,
durable-libs-first, `TT_METAL_RUNTIME_ROOT` at the pin.

(filled at the end of the wave)

## The gate suite (`run-w30-gate.sh`, 2026-11-20)

Because the writer did NOT isolate, the dispatch's "DNR disabled by default"
premise FAILED: the default flip was reverted in the same wave and the gate
suite runs BOTH arms on the same binary — the proof leg with the DNR OFF
(the honest corruption-absent probe: it answers whether the writer still
strikes with no repair armed) and the DNR-armed legs (the working
mitigation). Head `b7e8c64d4` + this wave's record commit; the pin untouched
(`611e346884f` still the W29 patchset). Leg env files carry the full recipe.

- **(proof) `w30-on-proof-1` — VT_TT_BFP8_RESTAGE=0 + the W28 canary: THE
  WRITER IS LIVE AND UNFIXED.** 185/185 `[W28-STAGE]` stage-time checks
  zero-free; 188 canary scans climb from clean to the FULL five-victim set
  (`0x231d6f00/0x2a146f00/0x2d7cef00/0x3473ef00/0x3a9eef00` — the EXACT W28
  victims, at the EXACT W28 addresses, a deterministic allocation layout);
  the stream is the W4 degenerate signature `[[220,0,0,...]]` byte-exact
  (`daa113b5...` — the pre-repair W21 sha). The corruption, not the repair
  machinery, is the cause of the degenerate stream: without the DNR the
  drafter-ON leg does not serve at all. THE CORRUPTION-ABSENT PROOF FAILS,
  and honestly: the writer was not fixed this wave because nothing
  reproduced outside the engine. TPOT 21612 ms (a corrupt run's number; not
  comparable).
- **(a) `w30-on-{2,3}` — Q4_K_M KV64 drafter-ON x3, DNR armed: PASS at the
  documented band, NOT token-exact.** 3/3 legs byte-identical `af77d5d3...`
  — the W25/W26/W27 healthy bimodal shape, 22/24 tokens equal the recorded
  `d49c53d2` with the SAME single 3-token near-tie shift at index 7 the W26
  acceptance trace attributed to the M=9 BFP8 verify-forward quantization
  band (the 500-mnat tie-band gate passes it; the corruption-class gate
  does not). THE ACCEPTANCE MULTIPLE: TPOT 24794.79 / 24791.63 (+ the
  identical on-2 leg) vs the OFF 6647.67 / 6648.36 / 6648.55 — **~3.73x
  SLOWER, the writer's repair cost, unchanged in verdict from W27's ~3.8x**
  because the writer is unfixed.
- **(b) UD-Q6_K_XL KV8 drafter-ON x3: the standing capacity blocker,
  UNCHANGED.** 3/3 exit 1 `Cannot auto-fit max_model_len: not enough GPU
  memory available to serve even a single token`. The tie-band gate never
  runs on this arm. (The dispatch's drafter-ON b-leg therefore remains
  blocked as in W25-W27.)
- **(c) drafter-OFF byte identity: PASS 6/6.** Q4_K_M KV64 x3 =
  `d49c53d2...` 3/3 (TPOT 6647.67 / 6648.36 / 6648.55 vs the recorded 6607,
  the established rerun band); UD-Q6_K_XL KV8 x3 = `b36bc61a...` 3/3
  (TPOT 5351.68 / 5350.50 / 5350.06 vs the recorded 5340.9, +0.2%).

## Records

- The repro program + the three legs: `tools/tt-metal-repro/` (committed) and
  `leg-{pure,mix,resident}.log` here.
- The issue's W30 section records the verdict and the default-flip revert.
- The pin is UNCHANGED this wave (`vllm-cpp-pin/20260923-adv` head
  `611e346884f`); no tt-metal issue is drafted.
- Owed next (the writer hunt continues IN-ENGINE): the strike addresses are
  DETERMINISTIC across runs and builds (the five W28 addresses again), which
  makes an in-engine device-side capture (a dispatcher write-command trace at
  the pin, or a DRAM watchpoint arm) feasible at a known address window —
  the first instrument with a fixed target.
