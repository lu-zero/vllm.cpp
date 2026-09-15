# Spec: QUANT-GGUF-IQ-TENSTORRENT wave 3 — Q3_K on the P150

Row: `QUANT-GGUF-IQ-TENSTORRENT`
Issue: `ISSUE-LOCAL-01M2CNC22SYAKJN3YBGNVGSG56`
State: ACTIVE (2026-09-15)
Git integration: one PR for spec and implementation (developer preference,
recorded for this row).
Base: `origin/main` (waves 1-2 landed: PR #3193, PR #3200; records in #3202).

## Scope

Serve `DType::kQ3_K` matmul weights through the TT int8-dot keep-quant
decode, closing the last refusable arm in the APEX-I-Nano census (78 Q3_K
tensors). Q3_K is NOT the codebook family: it is a min-term K-quant with
per-32 sub-block scales and a high-bit mask, so this wave ports a
structurally different dot — the last wave that the row's per-encoding
census demands. W4a grouped E=1 arms stay owed and out of scope.

## Upstream anchors

- CPU oracle (bit-exact target): `VecDotQ3_KQ8_K`,
  `src/vt/cpu/cpu_quant_dot.cpp:202` (quants.c:566,
  `ggml_vec_dot_q3_K_q8_K_generic`).
- Block: `BlockQ3_K`, `src/vt/cpu/cpu_quant_blocks.h:90` — 110 B =
  `{u8 hmask[32]; u8 qs[64]; u8 scales[12]; u16 d;}`. 256 elems per
  block; pairs with q8_K per `cpu_quant_traits.cpp` (`kQ3_K` case :67).
- Reference structure (statement-for-statement port source):
  - Quants: two bits per element from `qs` (`&3`, `>>2&3`, `>>4&3`,
    `>>6&3` over four 32-lane passes per 128), then subtract 4 where the
    `hmask` bit for that pass is CLEAR (`m` shifts 1..8 over the two
    128-loops). Result `aux8[256]` ∈ {-4,-3,-2,-1,0,1,2,3}.
  - Scales: 12 bytes memcpy into `auxs[4]` u32, then the kmask1/kmask2
    splice (`0x03030303`/`0x0f0f0f0f`) producing 32 int8 scales used as
    `(scales[j] - 32)`.
  - Dot: 16 lanes of `q8*a` per j-step, `aux32[8] += (scales[j]-32)*aux16`
    over `kQK_K/16` j; `d = x.d * y.d` (f16 x f32); `sums[8] += d*aux32`;
    final `sumf = Σ sums[8]`.
- Waves 1-2 pattern (the template): device dot in
  `keepquant_kernel_code.h` + `KeepQuantWordsPerBlock` + `enc_sel`/kernel
  arm + unconditional dispatch + `DeviceKeepQuantSupported` + pin tests —
  landed in #3193/#3200.
- Gate artifact: the APEX-I-Nano pin (wave-1 spec ## Upstream anchors);
  78 Q3_K tensors become device-decoded, making every quantized census
  arm in the artifact on-core decodable.

## Design

Mirror waves 1-2 point-for-point; the structural differences are local
to the dot:

1. NO new tables file. Q3_K uses only the inline `kmask1`/`kmask2`
   constants — no codebook lookup. This wave is smaller than wave 2 in
   device text.
2. `keepquant_kernel_code.h`: `kq_vec_dot_q3_k_q8_K` — ported
   statement-for-statement, including the `aux8[256]` local array, the
   hmask clear-bit subtraction, the `auxs[4]` scale splice, the
   `(scales[j] - 32)` bias, the 8-lane `aux32`/`sums` accumulation, and
   the final two-stage fold. No divisions. The existing q8_K activation
   quantizer is reused unchanged.
3. Staging: `KeepQuantWordsPerBlock(kQ3_K)`: 110 B → 128 B zero-padded →
   32 words (same sub-word tail treatment as IQ3_XXS/IQ2). The kernel
   reads `hmask`, `qs`, `scales`, `d` at their fixed offsets inside the
   padded 128 B stride.
4. Kernel dispatch: `enc_sel` 7 = Q3_K (4/5/6 are taken); kernel arm
   `enc==7`; `qb_pad` falls into the `nb*292` arm.
5. Route: `MatmulBTQuantKernel` routes kQ3_K to the int8-dot arm
   regardless of env, extending the existing waves-1-2 guard.
   `DeviceKeepQuantSupported` kTENSTORRENT adds kQ3_K;
   `test_gguf_keep_quant.cpp` pin widens (and the negative
   `route(kQ3_K) == kExpandBf16` check flips to `kKeepQuant`). The APEX
   refusal message narrows to the base registered set (no named missing
   arm); `docs/USAGE.md` artifact row drops the Q3_K refused-arm note.

## Risks

- The `auxs[4]` scale splice (`kmask2`/`kmask1` with the `tmp` shuffle)
  is the trickiest expression in the port — the mutation review MUST
  flip one shift (e.g. `tmp >> 4` → `tmp >> 5` in the `auxs[2]` line)
  and expect red.
- The hmask polarity is a second named trap: the subtraction applies
  where the bit is CLEAR (`? 0 : 4`), not where set — mutation MUST
  flip the conditional once.
- `aux8[256]` + `aux16[8]` + `sums[8]`/`aux32[8]` locals are the largest
  local-array footprint of the seven dots. Q6_K's device arm already
  carries comparable locals, but if the brisc kernel exceeds its stack
  budget the fallback is restructuring into per-128 passes, NOT a
  residency redesign (stop condition below). Verify via the linked
  brisc.elf, as wave 2 did.
- f16 `x.d` times f32 `y[i].d` association (`d` formed before the
  `sums[l] +=` fold) is upstream's; preserve it exactly for
  bit-reproducibility.

## Tests

- Red-first op-level: new case in the keep-quant op suite
  (Q3_K×q8_K) vs `VecDotQ3_KQ8_K`; RED = route refusal today (named in
  the registered-set message). Sweep-shape pattern of waves 1-2.
- Default-path leg: extend the env-unset pattern to kQ3_K (the
  dispatch-only mutation must go red).
- Route pin: kQ3_K admitted on kTENSTORRENT; the existing negative
  kQ3_K check flips (red before the predicate widens).
- Full backend suite stays green (75 + new cases).

## Gates

1. Op-level oracle: device dot bit-exact vs CPU across the sweep.
2. Route pin green.
3. Backend suite green on the P150 (device reset + GPU mutex, as waves
   1-2).
4. E2E: UNCHANGED — the APEX e2e generation gate stays recorded
   OOM/owed (Q4_K grouped repair-plane residency, keep-quant W4
   territory); wave 3 does not own it. No new device-run bench claim.

## Owed

- W4a grouped E=1 arms for IQ3_XXS/IQ2_XXS/IQ2_S/Q3_K.
- The APEX e2e generation gate (keep-quant W4 residency redesign).

## Stop conditions

- Same as waves 1-2: soft-float drift that cannot be closed → stop and
  escalate; no residency redesign in this wave. A device stack-budget
  refusal that survives the per-128 restructuring fallback is an
  escalate, not a redesign.
