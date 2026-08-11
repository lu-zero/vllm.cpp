# Tenstorrent Mistral allowlist + device-aware gate — spike

Status: **DRAFT, 2026-08-11.** Owes the RED mutation + on-card run before the
row leaves `SPIKE`. Two environmental prerequisites (7B checkpoint + vLLM
oracle) are required for the e2e gate and are being staged in parallel.

Proposed row id: `BACKEND-TENSTORRENT-MISTRAL` (child of `BACKEND-TENSTORRENT`).

## Scope

**In.** Allowlist `MistralForCausalLM` on `DeviceType::kTENSTORRENT` and add
the device-aware gate wiring so the Mistral-7B-v0.3 SACRED greedy gate
(`tests/parity/test_mistral_paged_engine.cpp`) runs on Blackhole against a
**Tenstorrent-appropriate** golden pair, not the CUDA one.

Mistral reuses the Qwen3-dense forward verbatim
(`src/vllm/model_executor/models/mistral_registry.cpp:7-11`:
`MistralModel == Qwen3DenseModel` with qk-norm skipped + plain rope theta 1e6
+ untied lm_head). Every op the forward dispatches is **already registered**
on `kTENSTORRENT` (the Qwen3 op set: `kEmbedding`, `kMatmulBT`, `kRmsNorm`,
`kRopeNeox`, `kReshapeAndCache`, `kPagedAttention`, `kSiluAndMul`,
`kGreedyArgmax`). The single new op Mistral adds vs Qwen3-0.6B is an
**untied** `lm_head` → `kMatmul` (Qwen3-0.6B ties its lm_head, so it only
uses `kMatmulBT`); `kMatmul` is already registered. **No new op, no new
kernel, no model code change** — the change is the platform allowlist line +
the test's device-awareness.

**Out.** No MoE, no quantized Mistral, no SWA-enabled variant (Mistral-7B-v0.3
has `sliding_window: null` → full attention), no Ministral/Mistral-Large, no
speed/perf axis (a separate increment). Metal allowlisting of Mistral is NOT
in scope (Metal's allowlist is `OPTForCausalLM` + `Qwen3ForCausalLM` only;
this row is Tenstorrent-only).

## Upstream chain

**No upstream vLLM equivalent** (no vLLM Tenstorrent platform). The loyal
contract is `MistralForCausalLM(LlamaForCausalLM)` in vLLM
(`vllm/model_executor/models/mistral.py`), already faithfully ported to this
tree's shared dense machinery. This row adds a device leg to an
already-ported model — it does not deviate from anything vLLM defines.

Reference anchors:
- `src/vllm/platforms/tenstorrent.cpp:52-54` — the allowlist under edit.
- `src/vllm/model_executor/models/mistral_registry.cpp:1-11` — the forward
  is Qwen3-dense verbatim with qk-norm skipped.
- `tests/parity/test_qwen3_paged_engine.cpp:221-296` — the device-aware gate
  pattern to mirror (the metal/tenstorrent branch).
- `tests/parity/test_mistral_paged_engine.cpp:101-185` — the gate to make
  device-aware.
- `tests/parity/goldens/mistral_greedy_7b/` — the CUDA golden pair already
  committed (`greedy_ids.npy`, `our_ids.npy`, `neartie_gap_mnats.npy`).

## Our baseline

**Landed on `origin/main`:** the full Mistral model (registry + weights +
forward), the SACRED gate (`test_mistral_paged_engine.cpp`), and the CUDA
golden pair. The gate is **not device-aware** — it has no
metal/tenstorrent branch, so on Blackhole it would compare TT output against
the CUDA `our_ids.npy` anchor. That is the wrong comparison: different bf16
decoders resolve genuine near-ties differently (the Qwen3 gate's file header
documents this at length — e.g. Metal p0 tok5 France 9625 vs Italy 15344).
The CUDA golden must NOT be reused for a TT run; a TT-specific golden pair is
required.

**Honest gaps:** (1) Mistral is not in the TT allowlist; (2) the Mistral
gate lacks the device-aware branch; (3) no TT golden pair exists; (4) the
7B checkpoint + vLLM oracle are not on this box (being staged).

## Port map

| Surface | Change |
|---|---|
| `src/vllm/platforms/tenstorrent.cpp:53` | Add `\|\| architecture == "MistralForCausalLM"` to `supports_model_architecture` |
| `tests/parity/test_mistral_paged_engine.cpp` | Mirror `test_qwen3_paged_engine.cpp:221-296`: read `run_dev`, set `tenstorrent = run_dev == kTENSTORRENT`, `device_golden = tenstorrent`, op-registration proof for the Mistral op set, device-appropriate golden selection (`our_ids_tenstorrent.npy` / `neartie_gap_mnats_tenstorrent.npy`) with the existing BOOTSTRAP dump path. |
| `tests/parity/goldens/mistral_greedy_7b/` | NEW `our_ids_tenstorrent.npy` + `neartie_gap_mnats_tenstorrent.npy` (captured via `VT_DUMP_IDS=1` on Blackhole → `scripts/qwen3-neartie-gap.py` teacher-force against the vLLM oracle). |

The Mistral op set for the registration proof (untied lm_head → includes
`kMatmul`):
`kEmbedding, kMatmulBT, kRmsNorm, kRopeNeox, kReshapeAndCache,
kPagedAttention, kSiluAndMul, kMatmul (lm_head), kGreedyArgmax`.

## Tests to port

None upstream (no vLLM Tenstorrent). The device-aware surgery mirrors the
already-landed Qwen3 device-aware gate 1:1.

## Gates

**Correctness gate:** `test_mistral_paged_engine.cpp` on real Blackhole,
gated against the TT-specific golden pair, near-tie-robust (≤500 milli-nats
per position), with strict-exact reported. Op-registration proof
(selections > 0, declines == 0) must confirm Mistral actually ran on the TT
provider, not CPU fallback.

**Hardware:** real Blackhole (P150). CPU-only CI compiles + skips.

**Not claimed:** no perf axis. Mistral-7B cold JIT on this P150 will be slow
(longer than Qwen3-0.6B's ~30 min cold — bigger model, more kernel shapes);
the gate run needs a long uninterrupted window with a warm cache.

## Dependencies

- `BACKEND-TENSTORRENT` (parent) — `ACTIVE` on `origin/main`.
- `MODEL-TEXT-mistral-mistral-for-causal-lm` — the model itself, `ACTIVE`
  on main (CUDA-gated). This row extends its device surface to TT.
- **Checkpoint:** `models--mistralai--Mistral-7B-v0.3` (~14 GB) in the HF
  hub cache. NOT on this box; download authorized by the developer for this
  row.
- **vLLM oracle:** a vLLM 0.25.0 install (pin `555967922`) for
  `scripts/qwen3-neartie-gap.py` teacher-forcing. NOT on this box; being
  staged by the developer.
- Toolchain: same tt-metal + Blackhole as the rest of the TT suite.

## Work breakdown

1. **Allowlist** (`tenstorrent.cpp`) — one line. Gate: the platform
   `supports_model_architecture("MistralForCausalLM")` unit assertion in
   `test_tenstorrent_backend.cpp` (already checks OPT + Qwen3; extend to
   Mistral).
2. **Device-aware gate wiring** (`test_mistral_paged_engine.cpp`) — mirror
   the Qwen3 branch. Gate: compiles; runs on CPU and SKIPs loudly (no card
   / no checkpoint).
3. **Bootstrap TT golden** — on Blackhole with the 7B checkpoint:
   `VT_DUMP_IDS=1 ./test_mistral_paged_engine` → `our_ids.i32`; convert to
   `our_ids_tenstorrent.npy`; run `qwen3-neartie-gap.py` against the vLLM
   oracle → `neartie_gap_mnats_tenstorrent.npy`.
4. **Run the gate** — `test_mistral_paged_engine` on Blackhole against the
   TT golden pair; record strict-exact / near-tie / fail counts + worst gap.

Steps 1-2 LANDED (verified on Blackhole: allowlist assertion 814/814; the
gate loads the 7B checkpoint, runs on device type 6, and 20/21 assertions
pass — the one failure is the intended "TT golden absent" REQUIRE_MESSAGE).
Steps 3-4 wait on a persistent shell + the dgx oracle.

### Resume recipe (steps 3-4, needs a persistent shell + dgx access)

The 7B cold JIT is long (likely >1 h, worse than Qwen3-0.6B's ~30 min) and
this session's tool environment reaps background jobs, so the bootstrap +
gate must run from a persistent shell. The vLLM oracle runs on the dgx
(x86+CUDA, per `qwen3-neartie-gap.py`'s header "Run on dgx"), NOT this
AArch64 box.

**Step 3a — bootstrap the TT golden** (on the Blackhole box, persistent shell):
```sh
cd /home/lu_zero/Sources/vllmcpp-tenstorrent
export TT_METAL_HOME=/home/lu_zero/Sources/tt/tt-metal
export TT_METAL_RUNTIME_ROOT=/home/lu_zero/Sources/tt/tt-metal
export LD_LIBRARY_PATH="build:$TT_METAL_HOME/build_Release/lib:$TT_METAL_HOME/build_Release/lib64:${LD_LIBRARY_PATH}"
# Clear any stale lock from a prior aborted run (only if no test_mistral process is alive):
# rm -f /dev/shm/TT_UMD_LOCK.CHIP_IN_USE_0_PCIe
VT_DUMP_IDS=1 ./build/tests/test_mistral_paged_engine
# -> tests/parity/goldens/mistral_greedy_7b/our_ids_tenstorrent.i32
```

**Step 3b — convert + teacher-force the gap** (the `.i32` → `.npy` conversion
then `qwen3-neartie-gap.py` on the dgx with the oracle venv):
```sh
# On the Blackhole box: i32 -> npy (N=16, T from greedy_ids.npy shape)
python3 -c "import numpy as np,sys; \
  a=np.fromfile('tests/parity/goldens/mistral_greedy_7b/our_ids_tenstorrent.i32',dtype=np.int32); \
  g=np.load('tests/parity/goldens/mistral_greedy_7b/greedy_ids.npy'); \
  np.save('tests/parity/goldens/mistral_greedy_7b/our_ids_tenstorrent.npy', a.reshape(g.shape))"

# On the dgx (vLLM oracle venv, pin 555967922):
PATH=$HOME/venvs/vllm-oracle/bin:$PATH ~/venvs/vllm-oracle/bin/python \
  scripts/qwen3-neartie-gap.py --model mistralai/Mistral-7B-v0.3 \
    --golden-dir tests/parity/goldens/mistral_greedy_7b \
    --ids-npy our_ids_tenstorrent.npy --gap-npy neartie_gap_mnats_tenstorrent.npy
```
(Check `qwen3-neartie-gap.py --help` for the exact `--ids-npy`/`--gap-npy`
flag names; the script as written reads `our_ids.i32` directly — adjust the
bootstrap filename or the script's `--our-ids` argument to the `_tenstorrent`
suffix so the CUDA `our_ids.npy` is not overwritten.)

**Step 4 — run the gate** (back on Blackhole, persistent shell):
```sh
./build/tests/test_mistral_paged_engine
# Expect: strict-exact + near-tie-only >= 16/16, fail == 0, op selections > 0
# on kTENSTORRENT. Commit the two new .npy goldens + record the counts here.
```

## Outcome (2026-08-11)

**Code landed + on-card verified; e2e golden capture pending a persistent
shell + dgx oracle.** The gate's own output confirms the wiring is correct:
on the Blackhole card it selects device type 6, loads the 7B checkpoint
(`caa1feb0e54d415e2df31207e5f4e273e33509b1`, now downloaded), and the
op-registration proof passes (20/21 assertions; the 1 failure is the
intended "TT golden pair absent" message). The checkpoint download
(mistralai/Mistral-7B-v0.3, Apache-2.0, ~14 GB) completed. What did NOT run:
the `VT_DUMP_IDS=1` bootstrap (7B cold JIT is too long for this session's
foreground tool calls and the env reaps background jobs) and the
`qwen3-neartie-gap.py` teacher-force (vLLM oracle belongs on the dgx, not
this AArch64 box).

## Risks/decisions

- **Cold-JIT wall-time.** Mistral-7B has 32 layers, hidden 4096, GQA 32/8,
  head_dim 128 — many more distinct kernel shapes than Qwen3-0.6B. The
  bootstrap + gate runs need a long, uninterrupted window (each aborted run
  risks the `CHIP_IN_USE_0_PCIe` robust-mutex self-deadlock seen on the
  residual-golden row; resume recipe: clear the stale lock only when no
  `test_mistral_paged` process is alive).
- **Untied lm_head correctness.** The one Mistral-specific op vs Qwen3-0.6B
  is the standalone `kMatmul` lm_head (untied). `kMatmul` is registered and
  tested, but the Mistral gate is the first e2e exercise of the TT untied
  lm_head path — watch the first divergent position if the gate fails.
- **Device-golden parity philosophy.** A TT-specific golden is mandatory,
  not a weakening: it holds TT to the SAME near-tie-robust bar as CUDA
  (≤500 milli-nats vs vLLM's own teacher-forced argmax given the TT prefix),
  just with a TT-appropriate anchor. Reusing the CUDA anchor would fail
  spuriously at genuine bf16 near-ties where TT and CUDA legitimately differ.
