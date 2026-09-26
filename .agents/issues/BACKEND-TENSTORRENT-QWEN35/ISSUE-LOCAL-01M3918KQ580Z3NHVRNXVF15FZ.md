ID: ISSUE-LOCAL-01M3918KQ580Z3NHVRNXVF15FZ
Title: qwen35-gguf-q4km golden drifts on the DEFAULT arm after the tt-metal rebase
Row: BACKEND-TENSTORRENT-QWEN35
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-24
Updated: 2026-09-26
Closed: -

## Problem

Fix row: [tenstorrent-qwen35-q4km-degen-fix](../../specs/tenstorrent-qwen35-q4km-degen-fix.md) (2026-09-25, STOPPED at the warmup-path redesign its own stop condition named).
Redesign row: [tenstorrent-capture-warmup-redesign](../../specs/tenstorrent-capture-warmup-redesign.md) (2026-09-26, the four waves).

**CULPRIT PINNED 2026-09-25 (git bisect, pinned tt-metal):** the degenerate stream entered with `14d7a25077` "fix: capture-safe reshapes and zero-cache warmup for Tenstorrent decode graph" (`2ed5e912e4`, its adjacent follow-up, inherits it). Bisect: good `46dbc7c1e` (09-13) -> bad `455a2de84` -> narrowed to the two-commit pair -> boundary leg at `14d7a25077` DIFFERS, so the first-bad is that commit itself. The committed goldens stand: vllm@2415a6b22 on the confirmed pin reproduces the committed stream byte-for-byte; the gated-norm fix (#3311) did not cure it (post-fix main still degenerate). The fix must reconcile the commit's capture-safety intent with the eager/first-decode numerics it broke (the zero-cache warmup path is the prime mechanism candidate for the period-3 degeneration).

The qwen35-gguf-q4km lane battery fails at prompt[0] on the committed golden — but the 2026-09-25 recapture session STOPPED before capturing: the current tt-metal tree is a DIRTY mid-experiment checkout (scratch commits 81f3bbf3b40/f3088579db0 + 25 uncommitted modifications across layernorm dispatch, program_run_args, memcpy prefetch), so the engine's tokens are measurements of an unpinned moving oracle, not of a tt-metal revision. The new capture is deterministic on the dirty build but diverges from the committed capture at 188/256 cells with max teacher-forced gap 13.125 nats (band 0.5) and degenerate loops — a numerics regression AGAINST THE DIRTY BUILD that must not be laundered into a golden. Required before any recapture: pin tt-metal to a clean recorded revision (or record the scratch modifications as a patch series), rebuild, and only re-derive goldens if the gap profile returns to the committed envelope (0.25-nat-class divergence, 0.5-nat band). Evidence: two byte-identical capture runs on the dirty build, gap re-derivation via the pinned qwen35-q4km-neartie-gap.sh recipe with the correct oracle venv and dequant artifact.

## Resolution

- 2026-09-26 (row tenstorrent-qwen35-q4km-degen-fix, spec commit 3e6268e28,
  pinned tt-metal vllm-cpp-pin/20260925, local Blackhole under the file
  mutex): ROOT CAUSE PROVEN, and the row STOPPED at its spec's stop
  condition — the reconciliation cannot preserve capture-safety without a
  warmup-path redesign, a bigger unit. The evidence:

  **Root cause (the red, reproduced and dissected).** The degeneration is
  the WARMUP MACHINERY of the bisected pair — `2ed5e912e4`'s GDN-shadow
  snapshot/restore (qwen3_5.cpp:12590-12606 at 3e6268e28), the 12-minutes-
  later follow-up inside 14d7a25077's window — NOT the reshape discipline.
  The restore rolls the ssm/conv state slots' device shadows back to their
  pre-warmup tensors after every decode-graph cold step. Two effects, both
  measured: (a) the warmup step's GDN state update is discarded, so every
  later step recomputes from the post-prefill state — the frozen recurrence
  (period-3 loop `279,6511,314`, first divergence prompt[0] tok 4, 188/256
  cells diverged, dump md5 6ae141e5e4184e48071a7b3413d8df83 vs the committed
  74f003d783f94e6589e04cd3452bc8df); (b) the restored conv_state slot reads
  `ConvShadowServeable()==false`, so the Step's conv_shadow_stale gate
  (qwen3_5.cpp:12292-12293) resets `s.warm` on EVERY step and the
  "captured" arm silently runs all-eager — the red leg's own gate printed
  "device trace demand at last capture end = 0 B": the capture never
  happened. The eager numerics are intact (the red's first 4 tokens match
  the golden; the op-level oracle suite is green), so 14d7a25077's
  CaptureSafeReshape class is exonerated for the red.

  **The reconciliation attempts (why the row stops).** Removing the
  restore re-exposes the capture-arm defects it was papering over, in a
  stack:

  1. First fatal: `ScatterRowsDevice`'s untilize misses the program cache
     mid-capture ("Cannot load new binaries during trace capture",
     backtrace untilize <- to_layout <- ScatterRowsDevice <-
     GdnDecodeKernel). Instrumented cause: at capture,
     `CaptureSafeReshape`'s member-view branch (tenstorrent_internal.h)
     relabels instead of reshaping — the capture's rows2d is the ORIGINAL
     [16,128,128]-padded tensor (the double-failure `return t` fallback)
     while the eager warmup produced [2,131072] tile-padded, so the first
     consumer's spec differs from every warmed program. Routing both
     passes through the free reshape (cache-hit; a miss fatals loudly)
     cleared this.
  2. Next refusal: "grouped-quant: activation staging miss during trace
     capture" (tenstorrent_keepquant.cpp:1126). Instrumented cause: the
     #3042 from_span staging caches the MLP activation device tensor
     keyed by the HOST POINTER; the engine's decode-graph warmup and
     capture steps allocate their activation scratch at DIFFERENT pool
     addresses (warmup 0xaaac978e6ac0/0xaaacaf7cd400 vs capture
     0xaaacafa4c880/0xaaacb18f8080), so the capture lookup structurally
     misses. Serving the warmup's cached tensor would read the WRONG
     step's values; the correct design is device-resident in-region
     activation serving (the GDN device-pure pattern) or persistent
     staging buffers — the warmup-path redesign this row's spec names as
     the stop condition's bigger unit. The pointer-identity assumption
     only holds in the op-level capture tests (same buffer staged then
     captured), which is why the suite stays green while the engine's
     captured arm cannot run.

  **Consequence for the fix shape.** The 0.8B captured arm has not
  completed a single capture since 14d7a25077: the conv-restore cascade
  kept it eagerly frozen, and every un-papering step surfaces the next
  capture-arm divergence (the reshape spec above, then the grouped-act
  staging, depth unbounded from here without the redesign). Byte-identity
  to the committed golden is unreachable inside this row's scope.

  **27B APEX anchor verdict (before-leg, unfixed tree, 4 prompts c=1,
  tt-int8dot-sweep-sharegpt-64-20260923.json, --output-token-ids).** The
  27B arm is AFFECTED TOO: its current stream degenerates the same way
  (period-2 loops, e.g. req0 `[220,17,220,17,...]`, req2 `[220,16,...]`;
  0.02 tok/s output — the all-eager cascade). Its committed evidence
  post-dates the regression, so it embeds a degenerate reference; any fix
  MUST re-derive it. Anchor tokens preserved at
  /tmp/q4km-red/anchor_before_ids.json (session evidence; re-capture from
  the worktree before relying on them).

  **Next unit (the split the spec names).** Redesign the captured arm's
  warmup/staging path: (1) drop the conv slot from any snapshot/restore
  so `ConvShadowServeable` stays true and the capture actually runs;
  (2) make the ssm restore value-preserving (copy the warmup's committed
  values into the restored geometry) or serve the warmup's commit
  directly; (3) replace the pointer-keyed grouped-act staging with
  device-resident in-region serving; (4) re-audit every capture-pass
  spec against its warmup counterpart until the trace completes, then
  gate byte-identity. The red dump, the instrumented logs, and the anchor
  tokens live under /tmp/q4km-red/ (this session).

- 2026-09-26 (row tenstorrent-capture-warmup-redesign, branch
  row/TT-CAPTURE-WARMUP-REDESIGN, commits 6e1f80b0b spec + a94a348ce W1 +
  1a5dff77d W2 + 9b9bd6dfe W3 + c0ee3a41a W4 + 24c6ccc22 suite repair;
  pinned tt-metal vllm-cpp-pin/20260925, P150, local Blackhole under the
  file mutex): THE FOUR WAVES LANDED AND THE CAPTURE COMPLETES. The red
  baseline reproduced first (dump md5 6ae141e5e4184e48071a7b3413d8df83,
  period-3 279/6511/314 from prompt[0] tok 4, 188/256 cells vs the
  committed golden, "device trace demand at last capture end = 0 B").
  W1: the conv slot left the snapshot/restore — the capture BEGINS
  (in-capture [TT-OP] entries through GdnDecode; the conv shadow serves;
  the warm branch is entered at all). W2: the ssm restore is gone (the
  commit is served directly — same split geometry, correct values) and
  the VT_TT_GDN_STATE_PROBE instrument landed; the gate leg (all eager)
  shows the recurrence ADVANCING (270 forwards x 16 layers, every layer's
  state checksum distinct at every step; period-3 GONE: 13/16 prompts
  strict-exact vs the oracle where the red was 0/16). W3: the grouped-quant
  activation serves device-resident in-region (ServeDeviceShadowRaw/
  Window + the geometry reinterpretation); the suite repair narrowed the
  pointer-keyed cache to the HOST-STAGED fallback only. W4:
  CaptureSafeReshape runs the free reshape in both passes. Decisive
  gates: the captured battery completes with "device trace demand at last
  capture end = 436,273,152 B" (the red printed 0 B) and its token stream
  is BYTE-IDENTICAL to the all-eager stream (md5
  c403afe36f31b324ba22216136be66e2 both arms); the device suite is at
  the bar (92/92 / 525,723, the red run's 13 keep-quant capture failures
  all clear with the narrowed fallback); the 27B APEX anchor is
  re-derived (see the anchor note below).

  **The byte-identity residual (this row's stop condition, recorded).**
  The captured stream diverges from the committed golden
  74f003d783f94e6589e04cd3452bc8df at 51 cells on prompts 2,3,9,10,11
  (first: prompt[2] tok 2 — the golden's own max-gap cell). Adjudication:
  at EVERY one of the 51 cells the CURRENT stream matches the vLLM
  per-prompt oracle greedy and the committed golden does not (51/51 vs
  0/51; 13/16 vs 8/16 prompts strict-exact); all gaps are inside the
  ratified 500-mnat band (max 250 mnats). The divergence is the golden
  era's own in-band deviation, superseded by justified numeric changes
  accumulated since the golden's capture (2415a6b22, 2026-09-12): the
  14d7a25077 eager tree measures 9/16 strict-exact (eager dump md5
  ca4bd359d2a841a08de67283825c73d7, a mid-point), the current tree 13/16.
  Byte-identity to the OLD golden is unreachable without a golden
  re-derivation (a non-goal of this row: the goldens stand), so per the
  spec's stop condition the residual moves to
  ISSUE-LOCAL-01M3ENS1FCHZ7SR5SJXQM174A2 as the separate, smaller
  numerics row: attribute the per-commit drift chain, then re-derive and
  ratify the capture pair under the pinned recipe.

  **The 27B APEX anchor (re-derived; the old numbers superseded by this
  measurement).** The committed evidence embedded the degenerate all-eager
  reference (0.02 tok/s output, mean TPOT 34225.60 ms, period-2 loops
  e.g. [220,17,220,17,...] / [220,16,...] — the stop record's before-leg,
  /tmp/q4km-red/anchor_before.log). The re-derivation on the fixed tree
  (4 prompts, c=1, seed 0, ignore-eos, the committed fixture,
  --output-token-ids): the 27B captured arm CAPTURES for the first time
  since the pair ("[DenseDecodeGraph] captured Qwen3.5 dense decode graph
  for padded size S=1" + replays, VT_DECODE_GRAPH_STATS), after the W4
  audit's 27B finding — the int8-dot kernel's activation refusal ("bf16
  activation staging during trace capture") — got the same in-region
  serve as the grouped kernel (the APEX anchor leg fatalled on it before
  the serve; the debug print [TT-I8DOT] records the miss state under
  VT_TT_TRACE_DEBUG). Measured: output throughput 0.02 tok/s, mean TPOT
  35041.16 ms (median 34962.10, P99 35327.76) — the same per-token class
  as the before-leg on this host; token stream
  [[220,17]x8, [220,17/16 mixed], [220,17/16], [220,16]x8] — still the
  loop SIGNATURE, but a DIFFERENT stream from the before-leg (req1 moved
  from [220,17,23066]* to [220,17,220,16]*: the numerics moved with the
  waves). The old numbers are superseded by this measurement; the 27B
  stream's adjudication (loop vs the model's true output under the pinned
  llama.cpp oracle) is owed — the re-derivation does NOT ratify the loop
  stream as correct, and the 27B anchor's oracle leg belongs to the 27B's
  own row.
