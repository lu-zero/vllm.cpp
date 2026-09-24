# TT split stage 6 — residue extraction, P150 gate session (2026-09-24)

Branch `row/BACKEND-TENSTORRENT-SPLIT-6` @ `4f20c7ffd`, on merged main
`1e53767c4` (stages 1-5 landed). Host: personal Tenstorrent Blackhole P150a
workstation; suite under `flock -x $HOME/gpu.lock`, `luwen reset` (+15 s)
retried up to 3x; build `/tmp/build-split6-tt` (Release,
`-DVLLM_CPP_TENSTORRENT=ON`, `-j16` — link-capped per the host RAM
discipline).

## Verdicts

| Leg | Verdict |
|---|---|
| Device suite `test_tenstorrent_backend` | **92/92 cases, 525,723/525,723 assertions** — identical to the stage-3/4/5 bar |
| Fresh review (mutations M1-M3) | PASS, see below |
| Record gates | `check-agent-record` ANCHOR-ROT=0, style, trailers OK |

Review mutation evidence:

- **M1** (internal.h `L2NormKernel` decl deleted):
  `tenstorrent_ops.cpp:1454: error: 'L2NormKernel' was not declared` — the
  Registrar's address-taking binds the moved gdn.cpp definition.
- **M2** (internal.h `EnsureAffine1D` decl deleted):
  `tenstorrent_ops.cpp:328,329,465` (embed/layernorm staging paths) and
  `tenstorrent_gdn.cpp:1170` all fail to resolve — ops.cpp callers tie to
  the residency definition.
- **M3** (verbatim accounting): 2,097 removed lines, 2,082 byte-identical
  reappearances; the 15 non-matching lines are exactly the named seams
  (4 namespace pairs, the `inline` drop on `Llama3ScaleFreq`, 6
  decl-dedup lines). ops.cpp diff: 0 insertions, pure deletions.
- Registrar `RegisterOp` count: 33 before = 33 after.
- Build: fresh configure + `ninja -j 16`, 2,142/2,142 targets green.

Logs: `/tmp/split6_build.log`, `/tmp/split6_suite.log`.

## Final shape (split complete)

`tenstorrent_ops.cpp` 1,483 lines (registration + core op kernels + glue);
`residency` 1,711 / `keepquant` 2,089 / `gdn` 2,272 / `paged` 2,398 /
`capture` 472 / `backend` 138 / `device` 72. The spec's umbrella issue
`ISSUE-LOCAL-01M2M2PZC0E998B88C9XKJT8JJ` exits when this lands.
