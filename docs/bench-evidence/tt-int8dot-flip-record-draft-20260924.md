# DRAFT benchmark-record entries — TT-KEEPQUANT-INT8DOT-DEFAULT gate work

> DRAFT for operator review; to be merged into `.agents/benchmark-record.md`
> (row `BACKEND-TENSTORRENT-KEEPQUANT`) after fresh review of the evidence
> commit. Full detail: `docs/bench-evidence/tt-int8dot-flip-gates-20260923.md`.

## BACKEND-TENSTORRENT-KEEPQUANT: INT8DOT flip gates — siblings RED (pre-existing), opt-out identity PASS, wide sweep BLOCKED by a device DRAM fault (2026-09-23/24, P150)

Gate work for the `VT_TT_KEEPQUANT_INT8DOT` default-flip spec
(`tenstorrent-keepquant-int8dot-default`, branch `row/TT-KEEPQUANT-INT8DOT-DEFAULT`
at `14a813a74`, no product code changed). TT-ON build verified by symbol
count; every device command under the file mutex with reset + env per hold.

**Gate 2 — FAIL, and NOT lever-caused.** The committed keep-quant vehicle
gates (`tests/parity/test_qwen35_paged_engine.cpp`, device type 6 confirmed in
both logs) fail their committed goldens on the DEFAULT arm before the lever
enters: `qwen35-gguf-q4km` anchor drifts at prompt[0] tok=0 (engine 11751 vs
committed 279) with the lever UNSET, and differently (7172) with `=1` — the
pair is stale against this tree's tt-metal rebase (`2415a6b22` recorded the
previous re-roll). `qwen38-gguf-q4km-27b` dies with a tt-metal DRAM OOM
(~268 MB vs ~63 MB largest-free-block, in `Qwen3_5DenseDecodeGraph::Step` →
`DecodeKeepQuantWordsF32` → `ttnn::where`) on BOTH arms. The 0.8B lane battery
loudly skips: no lane pair was ever captured for that vehicle. Spec stop
condition triggered — the flip does not land on this evidence; the follow-ups
are a golden re-capture on this tt-metal revision and the 27B OOM root cause.

**Gate 4 — PASS.** Anchor triple (65-token `gsq_p0` prompt, 16 greedy,
identical flags, one fresh device hold per leg): `VT_TT_KEEPQUANT_INT8DOT=0`
reproduces the default arm BYTE-FOR-BIT — same token stream and
byte-identical full-logit oracle TF dump (sha256 `37db5908…534e9` for both);
`=1` differs (stream and hash `38c80844…aa456`) — the flip's declared
byte-diff, rollback real. Anchor gaps against the pinned batch decode on this
build: default max **245 mnats** (band 500), lever max **449 mnats** — the
lever's anchor margin is 51 mnats on this tt-metal revision (the prior
adjudication recorded 245.3), which prices exactly the risk the wide sweep
exists to measure.

**Gate 1 — NOT MEASURED; device blocker on record.** The ≥64-prompt sweep is
fully staged (committed fixture
`tests/fixtures/tt-int8dot-sweep-sharegpt-64-20260923.json`, prompt 0
byte-reproducing the `gsq_p0` anchor; verified tokenizer and gap pipeline),
but the 27B APEX decode OOMs on this host/tt-metal revision after ~7 requests
in one process (64-, 16-, and 8-request processes all OOM; single-request
processes complete but settle at ~20 min each because the persistent tt-metal
cache does not amortize across processes), so the sweep is not executable in
one session. The blocker reproduces with NO product change on this branch.

**BUILD TRAP (row-wide):** `VLLM_CPP_TENSTORRENT:AUTO` resolves OFF
(`CMakeLists.txt:317-323`) and the engine silently lands models on CPU; the
`device memory budget is UNKNOWN` placement line appears in CPU runs too, so
it is not proof of TT execution (detect via `nm <binary> | grep -c
tenstorrent`). `/tmp/row-gsq-e2e` — the Make build several 2026-09-23
investigation scripts point at — is a TT-OFF build: any "P150" number cited
from that binary needs re-verification against a TT-ON build, including the
2026-09-23 anchor adjudication logs (`/tmp/int8dot_band.log` carries the same
UNKNOWN-budget line). A correct TT configure pins `-DVLLM_CPP_TENSTORRENT=ON`
plus `-DCMAKE_PREFIX_PATH` at the tt-metal build tree's
`build_Release/{lib64,share}/cmake`.

Also recorded: the flip gate's tasking said the keep-quant siblings are
"Qwen3.5-0.8B Q2K/Q4K and the 9B vehicle"; the tree's committed keep-quant
sibling gates are 0.8B Q4_K_M and 27B Q4_K_M — no Q2_K or 9B keep-quant gate
or golden exists (the 9B checkpoint on the box is HF safetensors).

Evidence: `docs/bench-evidence/tt-int8dot-flip-gates-20260923.md`,
`tt-int8dot-anchor-triple-hashes-20260924.txt`,
`tt-int8dot-anchor-gaps-20260924.csv`, `g2-default-20260923.log`,
`g2-int8dot-20260923.log`.
