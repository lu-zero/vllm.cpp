# BACKEND-TENSTORRENT-GDN-DEVICE-PURE — device-resident GDN decode for trace capture

Row: `BACKEND-TENSTORRENT-GDN-DEVICE-PURE` · Issue: #2907 · Owed from #2812
(`.agents/specs/tenstorrent-host-free-forward.md` `## Owed`). Lifecycle:
`ACTIVE` at the spec commit; implementation follows in this same PR (one-PR
shape is the recorded row-claim answer, 2026-09-04).

## Scope

Decode path only (T=1) inside the `Qwen3_5DenseDecodeGraph` Step. Make the
decode-side GDN ops device-resident so a tt-metal trace capture admits them,
unblocking the Qwen3.5-0.8B captured arm — the last capture-blocked family
with a committed eager pair.

IN:
- `CausalConv1dUpdateKernel` (`src/vt/tenstorrent/tenstorrent_ops.cpp:5013`)
  and `GdnDecodeKernel` (`tenstorrent_ops.cpp:5225`): remove the per-call
  host orchestration — `EnsureHost` on x/q/k/v/g/beta, `ReadIdxHost` on the
  state index, per-call host vector builds, per-call `UploadTensor`
  (`:5225` region, five-plus sites) — and replace it with device-resident
  inputs, device-side indexed state update, and no host reads or writes
  inside the captured region.
- Whatever must restage per step moves under the recapture cadence's
  re-priming instead of inside the captured span.
- The q35 harness (tests/parity/test_qwen35_paged_engine.cpp, pair selection
  ~:233) grows the captured-arm selection and the committed capture pair;
  the #2812 loud-skip cells run for real.
- Admission of the Qwen3.5 architecture to `DecodeCaptureDefaultArch`
  (tenstorrent_device.h:85 region) once the pair gate passes.

OUT: prefill capture (`kGdnPrefill` stays as landed), MoE decode graphs
(`Qwen3_5DecodeGraph`), recapture-policy changes, Llama/InternLM2 pairs,
performance floors (see Stop conditions).

Prerequisites (landed in the base): #2812's two staging-class fixes
(`ebce75456` gemma slot + weight-view shadows) — the fatal that remains is
structural, repro 2/2 (`pairs-q35-run1.out`, `pairs-q35-repro1.out`):
"Writes are not supported during trace capture"
(`fd_mesh_command_queue.cpp:760`).

## Upstream anchors

- vLLM is the behavior mirror for GDN semantics; the TT decode kernel
  already ports it and the committed eager pair adjudicates it. Ported
  sites are cited in the GDN parent row's records; this row changes
  orchestration, not recurrence math — any math edit must cite the upstream
  `file:line` it mirrors and re-run the backend op-level oracle suite
  (`test_tenstorrent_backend.cpp:1749-3340`, CPU f32 arm).
- Pair oracle: `transformers` via `scripts/qwen3-neartie-gap-transformers.py`
  (secondary oracle, `~/Sources/tt/tt-metal/python_env/bin/python`), the
  Mistral-precedent treatment; vLLM has no TT backend.

## Design (shape; the implementer refines with traced evidence)

1. Inputs resident: the Step's GDN operands must arrive as device tensors
   from the preceding captured ops, not as host tensors the kernel stages
   per call. Where the engine hands a host tensor, the graph's producer op
   commits device-side and the kernel consumes the shadow.
2. Indexed state without host: `ReadIdxHost` addresses the state slab by a
   host-side slot index. Decode slots are stable across a sequence's steps,
   so the captured graph can bake the addressing and let the recapture
   cadence re-prime on slot change; the device-side alternative is
   indirection through `kGdnStateGather`/`kGdnStateScatter` on a device
   index tensor. Measure the recapture cost before choosing; if per-step
   recapture destroys the win, indirection wins by default.
3. Conv-state coherence: the conv update keeps its two views
   (hidden-state view, ring view) coherent on device — see the conv-state
   two-views record in session memory before touching the layout.
4. Outputs commit device-side; token readback stays outside the captured
   span, as the dense path already does.

## Tests (red-first)

- RED now: the q35 opt-in battery's capture-blocked cells skip loudly
  (#2812 skip at the `test_qwen35_paged_engine.cpp:504` case); flipping them
  to run fatals deterministically (`fd_mesh_command_queue.cpp:760`). The
  wave's first green is those cells running captured.
- Byte-identity: captured dump ×2 with a card reset between
  (`VT_DUMP_IDS=1`), byte-equal — the Mistral precedent.
- Pair: teacher-forced via the transformers oracle; every cell inside the
  committed eager pair's band (or the 500-mnat Mistral band if the eager
  band is looser); worst cell recorded in the pair commit message.
- Admission: Qwen3.5 joins `DecodeCaptureDefaultArch`; ambient adjudicates
  the CAPTURED arm; `VT_TT_DECODE_CAPTURE=0` keeps the eager arm against
  the eager pair. Harness selection keys on `DecodeCaptureEnabled()` (the
  q3/mistral pattern), never on the env being set.
- Mutations: tamper one capture-pair cell → ambient reds at the anchor;
  delete the arch line → ambient reds (the #2812 mutation pattern).

## Gates

- Full battery on the head: 06b/q35/mistral ambient+opt-out, q35 opt-in with
  NO skips left (all cells run captured), 4B skip #2811 unchanged, backend
  suite 52/52; preflight rc=0; fresh review with mutation proof; the
  operator reruns the battery itself.
- Perf is report-only: captured-vs-eager tok/s on 0.8B re-measured on the
  landed tree (the 2026-09-04 eager wall is 0.177 tok/s). No ratio floor is
  claimed or gated.

## Evidence

Session logs under `/tmp/r1625-issues/` (`pairs-q35-*.out` fatal repros;
this row's legs prefixed `gdn-`). Committed goldens carry the model
snapshot revision in the pair manifest.

## Risks

- Recapture-per-step cost: if slot changes force recapture every step, the
  captured arm loses to eager — the design's clause 2 decides on measurement.
- A third structural staging class may surface once the two kernels are
  device-pure (unknown unknowns behind the fatal).
- The conv-state two-views layout is fragile; a layout change touches the
  prefill path's state handoff even though prefill capture is out of scope.

## Stop conditions

- The captured arm still fatals after both kernels are device-pure: name the
  next blocker with `file:line`, record it, stop.
- Pair outside the band after two repair attempts: `NEEDS_DECISION`.
- Never declare a same-architecture performance ceiling; an apparent limit
  is an unresolved difference — keep the gap open and name the next
  traceable hypothesis.

## Owed

- Qwen3.5 MoE-family captured arms (out of scope here; `Qwen3_5DecodeGraph`).
- Llama/InternLM2 captured pairs: demoted behind #1627 multi-request
  evidence (2026-09-04 bench: Mistral capture = 1.00x at c=1/7B, 0.6B =
  2.16x; their payoff is contingent on concurrency).
- #1625's multi-request captured throughput claim (async lane) is untouched.
