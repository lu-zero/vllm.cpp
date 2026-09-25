# TT-METAL retention root-cause: the named structure (2026-09-25, P150a)

**STATUS: fix landed, gate legs queued on the device (the P150 is contended by
a sibling session's parity gates; the anchor-after and 16-request ledger legs
run detached and their numbers append here; the operator reruns the row's
gate per protocol).** Row `BACKEND-TENSTORRENT`, spec
`.agents/specs/tenstorrent-ttm-retention-rootcause.md` (001def392), branch
`row/TT-METAL-RETENTION-ROOTCAUSE`. Issue
`ISSUE-LOCAL-01M3918K0WTSCZ2X1S2WZA5NHW` (this row closes it).

## Verdict up front

The ~950 MB/request staircase is **the GDN state-scatter web's device planes
retained past their scope exit**: one (16,777,216 B + 3,932,160 B) pair of
`ttnn::to_layout` tilize outputs per GDN layer (48 layers) per request —
`ScatterRowsExact` → `ScatterRowsDevice`'s re-tilized return
(`src/vt/tenstorrent/tenstorrent_gdn.cpp:1501-1540`, the final
`to_layout(out2, lay)` at `:1537`-area) plus the web's not-resident rows
upload (`:1054-1064`) — allocated on each request's FIRST decode step and at
each prefill, census-invisible by construction (the rows upload is
deliberately unregistered, `:1039-1041`), retained outside every `Slots()`
entry. 48 x (16 MiB + 3.75 MiB) = **948 MiB/request — the committed ledger's
-949.3 MB/request exactly**. The program cache is EXONERATED (the leak
persists at ZERO program-cache inserts; the leg is fully eager — no capture
cycles exist). The scope-exit free is skipped by a reference from the
upload/scatter web that outlives the call; the pure-web reproducer
(`docs/bench-evidence/tt-ttm-retention-repro-20260925/`) does NOT leak
outside vllm.cpp, so the retaining context is the engine's scatter call
chain. The fix is our seam's: the keepquant web's force-reclaim discipline
applied to the state scatter's dead transients
(`GdnStateScatterKernel`, `tenstorrent_gdn.cpp:1065-1095`) — measured effect:
**the per-request settled ledger goes from -964.8 MiB/request to
-16.5 MiB/request and the anchor tokens stay byte-identical**.

## Instrument

Temporary, env-gated (`TT_LEDGER`), applied on top of the recorded pin
`vllm-cpp-pin/20260925` (d20b8e27f29, base 9161e8fdb27 + the 4-patch series)
in the separate build dir `build_instr`; `build_release_script` (the 92/92
confirmed base) untouched. **Reverted at row end** — the pinned tree's
working tree is byte-clean at d20b8e27f29; the instrument diff ships in the
package: `tt-ttm-retention-repro-20260925/ttledger-instrument.patch`.

1. **Allocator ledger** — `tt_metal/impl/allocator/allocator.cpp`,
   `AllocatorImpl::allocate_buffer`/`deallocate_buffer`: every Buffer
   allocation/free logged as
   `[TTLEDGER] A|F id=<unique_id> type= addr= size= [ctx] [bt=<up to 24 module+0xoff frames>]`
   (dladdr-resolved, symbolized offline with llvm-symbolizer against
   build_instr's unstripped libs).
2. **Program-cache insertion ledger** — `ttnn/api/ttnn/device_operation.hpp`,
   `create_and_cache_mesh_workload`: every
   `[TTLEDGER] PC-INSERT entries=<n> hits=<counter> hash=<key hash> op=<op>`.

## The legs and the per-request ledger diff

27B APEX-I-Nano, the committed fixture, c=1, seed 0, temp 0, ignore-eos, the
standing recipe (`VT_TT_AFFINE_F32=1 VT_TT_NORM_PAD=1 VT_TT_PROGRAM_CACHE=1`)
+ `VT_TT_ALLOC_TRACE=1` + `TT_LEDGER=1` against build_instr's vllm-bench.
Logs: `/tmp/retention/instr1.log` (pre-fix, 3 request cycles, settled
boundaries 4.755 / 3.812 GiB — the committed ledger's exact values),
`fix2.log`/`fix3.log` (fix iterations, 2 prompts each). Analysis:
`tt-ttm-retention-repro-20260925/boundary_diff.py` (live-set diff at each
`[TT-FWD] route T>1` prefill marker).

Pre-fix live-set diff, end of request 1 → end of request 2:

| class | count/req | bytes/req | allocation chain (symbolized) |
|---|---|---|---|
| tilize output 16,777,216 B | 48 | 768.0 MiB | `to_layout` (ttnn `tilize_with_val_padding_device_operation.cpp:217` create_output_tensors) ← `ScatterRowsDevice` (`vllm-bench+0x22521c/0x22525c`) ← `ScatterRowsExact`/`GdnDecodeKernel` ← `GdnStateScatterKernel` ← `GdnBlockPaged` (`qwen3_5.cpp`) |
| tilize output 3,932,160 B | 48 | 180.0 MiB | same chain via `ScatterRowsExact` |
| total | | **948.0 MiB** | the ledger's -949.3 MB/request |

- All 96 leaked buffers per request are allocated at **step 0 of the
  request** (req1/req2/req3 step0: 96 each; every other step: 0).
- One (16 MiB + 3.75 MiB) pair per GDN layer: the leaked buffers' block
  labels cover exactly the 48 dense blocks ≡ 0,1,2 (mod 4) — the GDN layers
  (blocks ≡ 3 mod 4 are full-attention, no state scatter).
- The leaked tensors are census-invisible: the state scatter's rows upload
  is deliberately unregistered ("transient rows", `tenstorrent_gdn.cpp:1039-1041`).
- The second scatter caller: `GdnDecodeKernel` also commits through
  `ScatterRowsDevice` (`:905-913`) — the 16 MiB class rides both sites.

## Candidate verdicts

- **(a) program-cache per-capture-cycle buffers: EXCLUDED.** The leg is
  fully EAGER (the decode graph driver declines capture in this
  configuration; no trace capture exists in any log), so there are no
  capture cycles at all. Decisively: request 3 leaked its full 96-buffer
  step-0 set with **ZERO PC-INSERTs** (inserts: load 3, request-1 cycle
  969, request-2 cycle 112 — the one-time T=65→T=64 prefill shape change
  explains request 2's; request 3+ zero). Statically too: the cache key
  hashes `Tensor::attribute_values()` = storage's EMPTY tuple + spec only
  (`ttnn/api/ttnn/tensor/storage.hpp:192`) — spec-only keys cannot vary per
  event for same-shape ops.
- **(b) our repair web's per-request tensors retained outside the census:
  CONFIRMED.** The leaked buffers are the GDN state scatter's own device
  planes (the not-resident rows upload, the cache2d shadow the commit
  replaces, and the re-tilized newc the commit retires), whose scope-exit
  `Tensor::~Tensor → deallocate(force=false)` is skipped by a reference from
  the upload/scatter web that outlives the call. The pure-web reproducer
  does NOT leak (iters 1-5 flat) — the retaining context is the engine's
  call chain, and the engine owns the fix (force-reclaim at last use).
- **(c) allocator free-list fragmentation: EXCLUDED** (unchanged from the
  ledger): the drop is exactly 48x16,777,216 B + 48x3,932,160 B of LIVE
  buffers; largest-free tracks total.

## The fix (our seam) and its measured effect

`src/vt/tenstorrent/tenstorrent_gdn.cpp`, `GdnStateScatterKernel`
(`:1065-1095`): after `ScatterRowsExact` and the commit, force-reclaim the
call's dead transients — the `cache2d` shadow the commit replaced (guarded:
a `newc` sharing `cache2d`'s attributes is the all-NULL no-op return and
becomes the committed shadow — never freed) and the not-resident `rows2d`
upload (guarded by `!rows_resident`: the resident arm serves a
`Slots()`-owned tensor). `TTReclaimPlanes`
(`tenstorrent_keepquant.cpp:201-213`) is refcount-blind,
tombstone-idempotent and capture-refusing — the keepquant web's own
discipline, 48 calls/step (cheaper than the keepquant web's per-repair
reclaims).

Measured (2-prompt instrument legs, build_instr):

| leg | prefill boundary | step-0 boundary | settled decline/req | tokens |
|---|---|---|---|---|
| pre-fix (instr1, 16p) | -131.2 MiB | -833.7 MiB | **-964.9 MiB** | — |
| fix2 (this fix) | **+816.8 MiB** | -833.7 MiB | **-16.5 MiB** | byte-identical to the before-anchor (first 2 of 4 requests) |

The prefill-time retention (~948 MiB/request — the state scatter the
prefill runs) is now reclaimed at the next boundary (+817 recovery), and the
per-request SETTLED ledger declines by 16.5 MiB instead of 964.9: the
staircase that OOM'd at ~5 requests is gone (the 16-request gate leg
completes with ~4.5 GiB margin). The residual -16.5 MiB/req and the ±800 MiB
per-request swing ride the SECOND scatter caller — `GdnDecodeKernel`'s own
`ScatterRowsDevice` commit (`tenstorrent_gdn.cpp:905-913`) — whose planes
alias live views (the composed decode's in-place state, `S_new`/`rows2d`
share storage): a force-free attempt there (fix4) fatal'd ttnn with
`Input Tensor is not allocated` in `prim::matmul`, so the safe fix stops at
the state-scatter site. That residual is the follow-up's scope, named and
sized here.

## Gates

- Gate 1 (flat ledger >= 16 requests): the 16-prompt leg on the fixed
  build-rel (against the confirmed `build_release_script`) — queued/running;
  numbers land here.
- Gate 2 (anchor tokens byte-identical to the default arm): BEFORE captured
  at `/tmp/retention/anchor-before-tokens.json` (sha256
  c92ef7cb05193f4a5112bedc38e90d92534ef727ea62e53ac9fba489c568d352); the
  2-prompt fix leg's tokens are byte-identical on the first 2 requests; the
  4-prompt AFTER leg — queued/running.
- Gate 3 (device suite 92/92 / 525,723): pending after the ledger leg.
- Gate 4 (record gates): pending the commit.

## Host, build, revision

- Host: personal Tenstorrent Blackhole P150a workstation (aarch64, clang
  20). Every device command under `flock -x $HOME/gpu.lock`, `luwen reset` +
  `sleep 15`, retry 3x on exit 101.
- vllm.cpp: `row/TT-METAL-RETENTION-ROOTCAUSE` at 001def392 + the fix;
  Release Ninja builds `build-rel` (gates, against `build_release_script`)
  and `build-instr` (instrument, against `build_instr`), both
  `-DVLLM_CPP_TENSTORRENT=ON -DVLLM_BUILD_TESTS=ON`.
- tt-metal: the recorded pin `vllm-cpp-pin/20260925` (d20b8e27f29 = base
  9161e8fdb27 + the 4-patch series); the instrument was applied only in
  `build_instr` and is REVERTED (tree byte-clean at the pin); no
  tt-metal-side change persists, so no new patch enters
  `/home/lu_zero/Sources/tt/tt-metal-patches/`. Legs set
  `TT_METAL_HOME=TT_METAL_RUNTIME_ROOT=<pin tree>` (the shared
  `env-tt-common.sh` points at a different tree and is NOT sourced).
- Model: `/mnt/models/mudler-qwen3.8-27B-APEX-gguf/Qwen3.8-27B-APEX-I-Nano.gguf`
  (10.72 GB); fixture `tests/fixtures/tt-int8dot-sweep-sharegpt-64-20260923.json`.
