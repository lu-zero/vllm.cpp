# TT-KEEPQUANT-INT8DOT-DEFAULT: gate evidence — sweep status, siblings, opt-out identity (2026-09-23/24, P150)

**STATUS: DRAFT for operator review.** Row `BACKEND-TENSTORRENT-KEEPQUANT`,
spec `.agents/specs/tenstorrent-keepquant-int8dot-default.md`, branch
`row/TT-KEEPQUANT-INT8DOT-DEFAULT` (spec commit `14a813a74`, no product code
changed). Gate verdicts up front:

| Gate | Verdict |
|---|---|
| 1 (wide APEX band sweep, ≥64 prompts) | **NOT MEASURED this session** — device-side blocker, see §G1 |
| 2 (sibling keep-quant gates, both arms) | **FAIL** — pre-existing, NOT lever-caused; flip track stops per spec |
| 4 (`=0` opt-out identity) | **PASS** — byte-identical to the default arm |
| gate 3 (denominator filed) | not in this change's scope (spec: filed with the flip PR) |

Per the spec's stop conditions the flip must NOT land on this evidence.

## Host, build, device protocol

- Host: personal Tenstorrent Blackhole P150a workstation (aarch64, GCC 16).
- Every device command under `flock -x $HOME/gpu.lock`; per hold:
  `~/Sources/tt/luwen/target/release/reset && sleep 15`, then
  `source ~/Sources/tt/env-tt-common.sh`; tt-metal cache cleared between arms.
- Tree: this branch at `14a813a74`. Build `/tmp/row-int8dot-flip/build-tt`
  (Release, Ninja, `-DVLLM_CPP_BUILD_TESTS=ON -DVLLM_CPP_TENSTORRENT=ON
  -DCMAKE_PREFIX_PATH=<tt-metal build_Release/{lib64,share}/cmake>`); TT
  linkage verified (`nm vllm-bench | grep -c tenstorrent` = 398).
- Model: `/mnt/models/mudler-qwen3.8-27B-APEX-gguf/Qwen3.8-27B-APEX-I-Nano.gguf`
  (MTP head skip loud in every log, matching the oracle's ignore).
- Oracle: pinned llama.cpp b10451 CPU batch decode via
  `/tmp/lanegate-oracle/oracle-dump` (VTLGDUMP raw-f32 per-position logits),
  `LD_LIBRARY_PATH=/tmp/llamacpp-b10451/build/bin`, threads=8, ctx=128.
  Tokenizer verified byte-exact against `/tmp/gsq_p0.i32` via
  `llama-tokenize --ids --no-bos` before use.
- Gap semantics (as the 2026-09-23 anchor adjudication): teacher-force (prompt
  ids + OUR 16 generated ids) on the pinned batch decode; gap per generated
  token = max(0, argmax_logp − our_logp) in mnats; band 500.

## DEVIATION 1 — the TT-OFF build trap (found and worked around)

A default configure leaves `VLLM_CPP_TENSTORRENT:AUTO`, which resolves **OFF**
(`CMakeLists.txt:317-323`); the engine then silently lands models on CPU. The
log line `device placement: --fit resolved NO placement: the device memory
budget is UNKNOWN` appears in CPU runs too, so it is NOT proof of TT
execution. The first sweep pass of this session ran such a CPU build; its logs
were discarded (`f2`-prefix artifacts are the TT ones). Both
`/tmp/row-gsq-e2e/build` (the Make build several 2026-09-23 investigation
scripts point at) and its `vllm-bench` carry ZERO tenstorrent symbols — any
"P150 TT" number cited from that binary must be re-verified against a TT-ON
build before use. This includes the 2026-09-23 anchor adjudication logs
(`/tmp/int8dot_band.log` shows the same UNKNOWN-budget line).

## GATE 1 — wide APEX band sweep: NOT MEASURED (device blocker)

Setup completed: prompt fixture committed at
`tests/fixtures/tt-int8dot-sweep-sharegpt-64-20260923.json` — 64 prompts from
the committed sharegpt lineage RNG (standard MT19937-64 twist, seed 0, words
{hello, world, 1, 2}, grown to ≥64 tokens); prompt 0 reproduces the committed
`gsq_p0.i32` anchor byte-identically (verified through the pinned
tokenizer). Per-prompt `.i32` files and the whole pipeline (bench → token
ids → seq → oracle-dump → mnat gaps) were exercised end to end.

The TT device run failed on this host/tt-metal revision, on THIS tree with no
product changes, in three shapes (all under the mutex, all `TT_FATAL: Out of
Memory` at `bank_manager.cpp:495`, ~268 MB DRAM allocation against a ~63 MB
largest-free-block, in the decode path `Qwen3_5DenseDecodeGraph::Step` →
`DecodeKeepQuantWordsF32` → `ttnn::where`):

1. one 64-request process: OOM after ~60 requests (~79 min, ~1.3 min/req);
2. 16-request processes: OOM after ~14 requests;
3. 8-request processes: OOM after ~7 requests;
4. single-request processes: complete, but the second and later processes
   settle at ~20 min each (the first carries ~25 min of JIT compile; the
   persistent tt-metal cache does not amortize across processes at this
   revision). 128 such runs ≈ 40 h — not executable in this session.

Consequence: the ≥64-prompt band measurement for either arm did not happen.
What DID measure, on the anchor prompt (this TT build, greedy 16, seed 0,
ignore-eos, concurrency 4, one fresh device hold per leg):

| Leg | Stream (16 tokens) | Oracle TF sha256 | Max gap (mnats) | Cells off argmax |
|---|---|---|---:|---|
| default (lever unset) | `220,17,220,16`×7… | `37db5908…534e9` | **245** | (1: 27) (3: 245) (5: 3) |
| lever `=1` | `220,17,220,17,220,16`… | `38c80844…aa456` | **449** | (1: 27) (5: 449) (7: 141) |

Both arms in-band on the anchor, but the lever's margin shrank to 51 mnats on
this build/tt-metal revision (the prior adjudication recorded 245.3). This
single-prompt number is NOT the gate-1 verdict; it is the warning that prices
the risk the wide sweep exists to measure.

## GATE 2 — sibling keep-quant gates: FAIL (pre-existing, not lever-caused)

Ran the committed gates in `tests/parity/test_qwen35_paged_engine.cpp` (TT
build; logs confirm `device type 6 (TENSTORRENT)`), one lock hold per arm:

- `qwen35-gguf-q4km` (0.8B Q4_K_M), lever UNSET: **FAIL** — anchor drift
  prompt[0] tok=0, engine 11751 vs committed 279.
- same test, lever `=1`: **FAIL** — drift prompt[0] tok=0, engine 7172 vs 279.
  The default arm failing under the same build proves the committed pair is
  stale against this tree's tt-metal rebase (commit `2415a6b22` already
  recorded one such ULP-level re-roll on 2026-09-12); the lever arm's
  different drift value is the expected arm difference, not the cause.
- `qwen38-gguf-q4km-27b` (27B Q4_K_M), BOTH arms: **test binary dies** with
  the same tt-metal DRAM OOM as gate 1. The 27B gate is unreachable on this
  build/tt-metal revision regardless of the lever.
- `qwen35-gguf-q4km-lanegate` (int8-dot lane battery): the lane pair was never
  captured for the 0.8B Q4KM vehicle — loud skip (red-first by design).

Spec stop condition triggered: "any sibling model fails its own committed gate
on the lever → stop". The failures are pre-existing on the default arm, so the
follow-up is: re-capture the 0.8B golden pair on this tt-metal revision and
root-cause the 27B OOM, then re-run this gate.

DEVIATION vs the task text: no committed Q2_K or 9B keep-quant gate exists
in-tree — the keep-quant sibling gates are 0.8B Q4_K_M and 27B Q4_K_M (the 9B
checkpoint on this box is HF safetensors). The "Q2K/Q4K + 9B vehicle" wording
could not be executed as written; flagged for spec amendment.

## GATE 4 — `=0` opt-out identity: PASS

Anchor prompt (`gsq_p0.i32`, 65 tokens), 16 greedy tokens, concurrency 4,
seed 0, one fresh device hold per leg, identical flags across legs:

| Leg | Token stream | Oracle TF dump sha256 |
|---|---|---|
| default (lever unset) | `220,17,220,16,220,16,…` | `37db59084984bbbf1881a3e33fb54e3ecaa1a716ebfd7e85f25f1175030534e9` |
| `VT_TT_KEEPQUANT_INT8DOT=0` | `220,17,220,16,220,16,…` | `37db59084984bbbf1881a3e33fb54e3ecaa1a716ebfd7e85f25f1175030534e9` |
| `VT_TT_KEEPQUANT_INT8DOT=1` | `220,17,220,17,220,16,…` | `38c808444c3ea1b4029d10b60371ea81be47b83aec0b68bf42e19e406afaa456` |

The `=0` opt-out reproduces the default arm byte-for-bit — same 16-token
stream AND byte-identical full-logit oracle TF dump (hash equal). The `=1`
lever differs (the flip's declared byte-diff for TT keep-quant users), and its
stream stays coherent. The rollback the spec requires is real, on this build.

Provenance caveat: the historical `/tmp/apex_ours_tf.dump` (2026-09-23 anchor
session) teacher-forces the oracle-side near-tie stream rather than the engine
stream, and that session's bench binary is of doubtful provenance (see the
TT-OFF trap), so identity is anchored to this build's default arm, not a
cross-session hash.

## Artifacts

- `tt-int8dot-anchor-triple-hashes-20260924.txt` — the sha256 table above.
- `tt-int8dot-anchor-gaps-20260924.csv` — per-cell anchor gaps (both arms).
- `g2-default-20260923.log`, `g2-int8dot-20260923.log` — the sibling gate legs.
- `tt-int8dot-flip-record-draft-20260924.md` — DRAFT benchmark-record entries.
- Working directory of the session: `/tmp/tt_sweep/` (scripts
  `run_gate1.sh`, `run_final2.sh`, `gap_analyze.py`; discarded CPU-pass logs
  `band_arm{A,B}.log` of the first pass).
