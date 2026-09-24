# TT split stage 5 — capture extraction, P150 gate session (2026-09-24)

Branch `row/BACKEND-TENSTORRENT-SPLIT-5` @ `94307da7f` (+ anchor `688371c40`,
rot repair `b0c9d4d2c`), stacked on the merged stage-4 main (`8ee16c33e`).
Host: personal Tenstorrent Blackhole P150a workstation; suite under
`flock -x $HOME/gpu.lock`, `luwen reset` (+15 s) retried up to 3x; build
`/tmp/build-split5-tt` (Release, `-DVLLM_CPP_TENSTORRENT=ON`, `-j16`).

## Verdicts

| Leg | Verdict |
|---|---|
| Device suite `test_tenstorrent_backend` | **92/92 cases, 525,723/525,723 assertions** — identical to the stage-3/stage-4 bar |
| Fresh review (mutations M1-M3) | PASS — see below |
| Record gates | `check-agent-record` ANCHOR-ROT=0, style, trailers OK on all three commits |

Review mutation evidence (operator-completed legs included):

- **M1** (`TraceBeginCapture` body inert): suite RED with the causal
  signature `TraceEndCapture without Begin` at the two capture-lane cases
  (`test_tenstorrent_backend.cpp:1784`, `:5143`).
- **M2** (decl deleted): `tenstorrent_backend.cpp:116` compile error
  `'TraceBeginCapture' was not declared` — the production ABI call site
  binds through the declared seam (verified twice, independent runs).
- **M3** (capture.cpp removed from the OBJECT lib): link failures from six
  production seams (`tenstorrent_backend.cpp:116-121` ABI overrides,
  `qwen3.cpp:954` WarmDecodeIds, `model_loader.cpp` alloc-trace trio,
  `platforms/tenstorrent.cpp:50`, `qwen3_5.cpp` keep-quant staging) — the
  moved TU is live and multiply-reached, no orphan.
- **Alloc-growth cross-check** (adjacent to
  `ISSUE-LOCAL-01M3918K0WTSCZ2X1S2WZA5NHW`): `EmbedDeviceIdsInto` runs
  exactly twice per capture cycle (dummy run + captured pass,
  `qwen3.cpp:1013/:1047`), never per replay; 2 fresh allocations per cycle,
  1 held steady-state per distinct `n`; `WarmDecodeIds` allocates once per
  new decode-batch size and refreshes in place after
  (`tenstorrent_capture.cpp:213-222`). No new growth site in the move.

Logs: `/tmp/split5_suite.log`, `/tmp/split5_build.log`.
