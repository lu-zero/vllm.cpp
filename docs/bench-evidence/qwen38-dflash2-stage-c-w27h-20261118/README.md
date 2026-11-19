# QWEN38-DFLASH2-STAGE-C W27h — the host-drafter arm gate (2026-11-19)

The W27 host arm (`VT_DFLASH_HOST=1` + `VT_DFLASH_DRAFT=1`: the DFlash2 draft
block forward on the CPU thread pool over host-resident weights, the trunk
never staged to the device) captured at FULL length on the committed form.
Commit `4dca33bc1` (the stripped, reviewable implementation; the probe
scaffolding `VT_DFLASH_HOST_DEBUG` removed). The probe leg of the earlier
session, `w27h-on-1` (12 tokens), is here too.

Issue: `ISSUE-LOCAL-01M7FY5W1A91XHV1PZDGZEKR7H`. Harness: `run-w27h-gate.sh`
(this directory), the tie-band gate `tie-band-gate.py` committed in
`../qwen38-dflash2-stage-c-hunt-20261118/` (500 mnats, the eager top-2-gap
transform). Leg discipline: mutex exec 9>$GPU_LOCK; luwen reset + 15 s +
/dev/shm cleanup; idle gate; bench taskset 32-63; INT8DOT=1;
TT_METAL_RUNTIME_ROOT at the pin; VT_TT_WEIGHT_RESIDENCY=bfp8.

## Legs

| leg | arm | len | exit | stream sha256 | verdict |
|---|---|---|---|---|---|
| `w27h-off-gaps-full` | OFF (+VT_EAGER_TOP2_TRACE=220) | 200 | 0 | `d49c53d2…` | **byte-identical** to the W26 OFF records (`w26-off-1/2/3`, same hash) at FULL 200 length — the drafter-OFF surface is untouched by the host arm; the W26 6/6 record stands (the Q6 OFF half rests on the unchanged code path: every edit sits inside a `dflash_host_arm_`/`host_debug` branch) |
| `w27h-on-full` | ON-HOST | 200 | 0 | `fa01bf2f…` | tie-band **PASS** (2 mismatches, both inside the band; zero NaN/Inf) — but acceptance **1.03 drafts/step** (the bonus token only; drafts proposed and rejected every step) vs the device arm's 3.0 |
| `w27h-corr` | ON-HOST, planted corruption, DNR off | 24 | 0 | `daa113b5…` | tie-band **FAIL** (red proof: 23 degenerate ids, 4 out-of-band mismatches, NaN markers) — the gate SEPARATES corrupted from healthy on host-arm streams |
| `w27h-q6kl` | ON-HOST, UD-Q6_K_XL KV8 | — | 1 | — | **no capacity verdict: the leg refuses at auto-fit before draft wiring** ("not enough GPU memory to serve even a single token"), identically on a fresh luwen reset. NOTE: all three W27 device-drafter legs (`w27-q6kl-on-1/2/3`) failed with the SAME refusal — the KV8 blocker is the TARGET's bfp8 footprint + the auto-fit accounting at `--num-blocks 8`, not the draft trunk; taking the draft off-device does not unblock it |

## The acceptance multiple — THE HONEST NO-WIN (NEEDS_DECISION)

| arm | Mean TPOT | vs OFF 6647 ms | acceptance |
|---|---|---|---|
| drafter-OFF (recorded) | 6647 ms | 1.0x | — |
| drafter-ON device (W27) | 25415 ms | 3.8x | 3.0 drafts/step |
| **drafter-ON host (W27h, FULL 200)** | **59522.82 ms** | **8.96x** | **1.03 drafts/step** |

The host arm does not beat the OFF baseline — it is 2.3x WORSE than the
on-device drafter it was meant to replace, because the host forward's
numerics reject nearly every draft (the probe's flat-logits finding
reproduces on the committed form). The verify stays lossless, so the
emitted stream remains tie-band-correct; only acceptance dies, and with
k=8 that is pure overhead: 61 s/step to emit 1.03 tokens.

## The per-step budget (w27h-on-full, 194 traced steps, median)

| seam | per step |
|---|---|
| taps D2H + bf16 convert + host context-KV appends (`pre`) | 15.5 ms |
| **host draft block forward** (CPU thread pool, host-resident weights) | **413 ms** |
| sample + tokens H2D | 1.3 ms |
| BFP8-DNR repair ledger (master_fail=5399 across the run) | ~6.6 s |
| verify + other (the residual wall) | ~54 s |
| **step wall** | **61.0 s** |

The draft forward is 0.7% of the step. The step is dominated by the same
verify+DNR wall every drafter-ON arm carries (W26 25.4 s, W27h 61 s) — the
host-arm legs' wall is INFLATED relative to W26's, and the DNR master-fail
storm (5399 failures; the master-restore path re-staging from host while
the trunk is host-resident) is the named, unrooted suspect. Never a
ceiling: the next traceable lever is a queue-idle re-measure of the DNR
ledger (owed from the W27 record) and a host-arm leg with the DNR master
path instrumented, before any check-cost target is judged.

## What survives, what is owed

- The host arm is FUNCTIONAL and token-faithful (tie-band PASS), and the
  corrupted red proof still fails the gate. Those are real.
- The acceptance collapse is NOT rooted: the host-vs-device forward parity
  gate at production shapes does not exist yet. Owed.
- The UD-Q6_K_XL KV8 capacity question is answered NEGATIVELY for both
  arms: the refusal precedes the draft. The W26 "the draft's 3.85 GiB"
  attribution does not reproduce as the binding term. Owed: an auto-fit
  decomposition at this geometry (target bfp8 residency vs pool sizing).
- The OFF 200-token byte-identity at `d49c53d2…` and OFF TPOT 6652.90 ms
  (+0.1% vs the recorded 6647) re-pin the reference on this head.
