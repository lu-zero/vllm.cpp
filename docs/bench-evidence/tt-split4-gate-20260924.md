# TT split stage 4 — paged extraction, P150 gate session (2026-09-24)

Branch `row/BACKEND-TENSTORRENT-SPLIT-4` @ `dbee528e0` (+ record `72f110728`),
host: personal Tenstorrent Blackhole P150a workstation, `origin/main` `b24f8094c`.
Every device leg under `flock -x $HOME/gpu.lock`, `luwen reset` (+15 s) before each run,
reset retried up to 3x on the observed 101 flake. Build:
`/tmp/build-split4-tt` (`-DVLLM_CPP_TENSTORRENT=ON`, tt-metal `build_Release`,
GCC 16, Release).

## Legs and verdicts

| Leg | Verdict |
|---|---|
| Full device suite `test_tenstorrent_backend` | **92/92 cases, 525,723/525,723 assertions** — identical to the stage-3 bar |
| Checkpoint-backed `test_opt_paged_engine` (`VLLM_CPP_OPT_MODEL_DIR=/mnt/models/opt-125m-bf16-st`) | **blocked by a pre-existing main defect**: `kQkvSplit: out shapes must be [T, *]` — control run at `b24f8094c` (own worktree `/tmp/row-main-ctrl`) throws the identical exception at the unmoved site (`ops.cpp:1968` on main vs `:1454` on this branch, same 21 pre-checkpoint assertions passed on both). Filed `ISSUE-LOCAL-01M39392YYBDPPQXJF0B1X0AKS`; the extraction is exonerated |
| Reachability mutation (both `RegisterOp(OpId::kReshapeAndCache/PagedAttention, …)` statements commented, lines 2793–2796) | **RED as required**: mutated build green, suite **87/92, 5 failed** — exactly `:1141` `OpRegistered(kReshapeAndCache)` and `:1216/:1419/:1536/:1664` `OpRegistered(kPagedAttention)`; restore → byte-for-byte clean tree (`git status` 0 lines), rebuild green, suite **92/92 green** |

Logs: `/tmp/split4_gate.log`, `/tmp/split4_gate3.log` (first mutation attempt:
regex mismatch — `mutated_registrations: 0`, suite green on the unmutated
binary, i.e. the harness baseline re-confirmed), `/tmp/split4_gate4.log`
(the proven red), `/tmp/mut_suite.log`, `/tmp/restore_suite.log`,
`/tmp/split4_gate_opt.log`, `/tmp/main_ctrl_opt.log`.

## Interpretation

The suite is the device-executing reachability detector for the moved
kernels: it carries dedicated RAC/PA device cases, and losing the two
registrations reds exactly those cases while nothing else moves. The OPT
checkpoint leg cannot serve as that detector for anyone until the
pre-existing `kQkvSplit` defect is fixed, and that leg is owed by the
issue rather than by this row.
