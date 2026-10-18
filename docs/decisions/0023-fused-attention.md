# 0023: Fused attention (staged fusion, bit-identical)

- Status: accepted (2026-08-03 — decided in the main session under the maintainer's approval to
  continue the performance work); partly superseded by [0067](0067-autoregressive-attention-vocabulary.md)
  decision 1 (GQA: the exact H match of decision 4, 2026-08-17), by the 2026-08-11 addendum (the
  mask input of decision 4), by [0110](0110-practical-tier-numerics-contract.md) decision 2 and
  [0124](0124-attention-row-block-budget.md) decision 9 (decision 2's ban covers only the existing
  key), and by the later family ADRs (the target list of decision 5). Decision 1's rejection of
  full flash fusion was re-graded to SHOULD on 2026-10-18.
- Date: 2026-08-03
- Scope: the `attention` op (the 50th contract op), `packages/runtime/src/kernels/attention.ts`,
  shared fragments in `gemm.ts`, and per-target SDPA preservation in the exporter
- Related: [0022](0022-gemm-register-blocking.md) (GEMM skeleton) / [0016](0016-anima-chain-export.md)
  (safe-softmax guard removal)

## Context

In one DiT step (ADR 0021 timing, 1024 px, i8 weights), bmm 22.2% + softmax 2.7% + attention-related
strided / mul ≈ 27.0% of GPU time could be fused. Memory: S and P were alive together at
1,073.7 MB each (1024 px, one in the DiT and one in the VAE), the main share of VRAM during
denoising. Design: [research/2026-08-03-attention-design.md](../research/2026-08-03-attention-design.md).

## Decision

1. **Staged fusion (option B).** One `attention(q, k, v)` node is **3 dispatches**: ① QK GEMM —
   materialise `S = (q·scale) @ (k·scale)ᵀ`, reading k `[N, D]` transposed in place like linear (the
   old `permute(kᵀ)` disappears); ② row statistics — `m = amax S` and `inv = 1/Σexp(S − m)`, cut
   verbatim from passes 1–2 of the softmax kernel; ③ PV GEMM — evaluate `exp(S − m)·inv`
   per component while filling the A tile, **without materialising P**.
   The default path SHOULD stay with staged fusion rather than full flash fusion (option A).
   The prototype measured flash fusion slower at the same model and resolution (+12.5% step time at
   1024 px, +3.1% at 512 px; 5,498 vs 10,754 GFLOP/s), and with D = 128 and 32 KB shared memory
   BR = 32 / BC = 16 was nearly the only configuration (design §4.3.1). A flash path enters as a
   separate key with its own evidence. **(re-graded 2026-10-18 to SHOULD: the reason to keep the
   default staged — bit identity, decision 2 — remains, but the rejection numbers come from one
   measurement on a different machine without subgroups and do not forbid re-measuring a flash path
   as an opt-in route; ADR 0124 decision 9 leaves that route to its own ADR, and Dawn offers
   subgroups and packed dot products — ADR 0127 Context.)**
2. **Bit identity is the core of the design (MUST).** The output is bit-identical to the
   decomposed path (`mul → permute → mul → bmm → softmax → expand → bmm`). It rests on four
   points: (a) ① reduces in K-ascending order with tile 16 (the bmm skeleton); (b) the scale is
   applied to **both** q and k while filling tiles; (c) ② uses the same 256-wide tree reduction,
   scan order, identity and `inv = 1.0/Σ` as softmax passes 1–2; (d) ③'s A element uses the same
   expression and order as softmax pass 3. Breaking any one changes the rounding sequence.
   Changing ② to online softmax is forbidden; it would need a separate key and re-derived evidence,
   like option A. **(superseded by ADR 0110 decision 2 on 2026-09-26 and ADR 0124 decision 9 on
   2026-10-11: the ban covers changing the existing key only; a separate key that does not materialise S is
   admitted through the contract classes of ADR 0110.)**
3. **The `scale` attribute is a half scale** (required, no default): `√scale_factor`, applied to
   both q and k (as torch `_scaled_dot_product_attention_math` does). A full scale applied once
   after the dot product changes the rounding sequence and violates the contract. The exporter
   computes `f32(math.sqrt(scale_factor))` from SDPA's `scale` (default `1/√D`); a pytest pins the
   D = 128 oracle value `0.2973017692565918` taken from the real DiT IR.
4. **Narrow contract.** Arity 3; rank-4 head-first (`q[B,H,M,D]`, `k[B,H,N,D]`, `v[B,H,N,D]` →
   `[B,H,M,D]`); uniform f32; **D equal for all three** (torch allows a different head_dim for v,
   which would let a D mix-up pass the shape check); **B and H checked separately** (the kernel
   folds B·H into one batch axis, so a swap does not show in the values); N = 0 rejected. Mask,
   causal, dropout and GQA get no field.
   - GQA: the exact H match of q / k / v **(superseded by ADR 0067 decision 1 on 2026-08-17:
     `H % Hkv == 0` and `H ≥ Hkv ≥ 1` are accepted as a divisible broadcast; `r = H / Hkv` is
     derived, no field. The exact B match, equal Hkv for k / v, equal D and the N = 0 rejection
     stay).**
   - Mask: **(superseded by the 2026-08-11 addendum: an optional fourth input `mask`.)**
5. **The exporter preserves SDPA per target.** `export_module(…, preserved=…)` keeps SDPA only for
   the targets that ask for it (`TARGET_PRESERVED` in `tools/export-recipes/anima/export.py`). It is
   **not** added to the default `PRESERVED_OP_PREFIXES` (`tools/exporter/src/karume/convert.py`):
   globally, the text encoder (causal mask folded with −inf) would fail loudly in `_h_attention`
   (`tools/exporter/src/karume/aten_handlers.py`), and the safe-softmax guard removal of ADR 0016
   would no longer apply. `_h_attention` rejects every unsupported form explicitly (mask outside
   the addendum's contract, `is_causal`, dropout ≠ 0, rank ≠ 4, D mismatch, symbolic D, non-finite
   scale). As decided, only Anima's transformer and VAE decoder preserved SDPA. **(superseded from
   2026-08-11 on by the family ADRs — 0067, 0118, 0121 and the 2026-08-11 addendum: Wan, Gemma 4,
   EmbeddingGemma, MiniCPM5 and Irodori recipes also opt in through
   `PRESERVED_OP_PREFIXES_WITH_ATTENTION`. The principle stays: never in the default list, opt-in
   per target.)** SBV2 cannot be fused (its softmax output is consumed twice).
6. **No separate clean-up of identity copies.** Preserving SDPA removes the identity expands (224),
   identity permute pairs, the scale mul and the kᵀ permute from the fused graphs at the IR level.
   What remains in non-fused targets is negligible.

## Verification

- Permanent bit-identity check: `packages/runtime/tests/gpu_attention_parity_test.ts` runs the
  decomposed node sequence and the `attention` op on a real GPU and asserts identical Uint32 bits
  on five shapes, with a guard that fails on constant output.
- On 2026-08-03 the demo PNG sha256 matched before and after (512 px / f16 weights, 1024 px / i8
  weights); no tolerance changed. Fault injection: 10 cases, each caught by two or more tests.
- Golden `fused_attention` uses distinct lengths for B / H / M / N / D; the existing
  `attention_block` golden is a hand-written decomposition that never goes through SDPA and is kept
  as cover for bmm → softmax → bmm.

## Detection limits

- A B / H swap does not show in kernel values; only the per-axis contract check and the golden's
  five distinct axes catch it.
- Ragged M / N do not occur on real model paths (the DiT uses multiples of 64); unit tests are the
  only detector.
- In the row-wise grid-stride family, mutating the stride to a constant 1 is not detected by the
  degeneration harness (every row is still visited); skipping rows is detected.
- Inputs with large negative logits that vary within a row cannot meet `GOLDEN_TOLERANCE`
  (atol 1e-6): the f32 dot-product error at |S| ≈ 190 is about 1e-5 and becomes a relative error
  through `exp`. This is the input's conditioning, not the fusion; the parity test pins equality
  with the decomposed path.

## Consequences

- Measured on 2026-08-03: 512 px DiT GPU 968 → 929 ms per step and 3,137 → 2,633 dispatches;
  attention transients at 1024 px halved (2,147.5 MB → 1,073.7 MB, P not materialised). Step time
  varies ±20% with GPU clock (heat); compare per-key ratios within one session.
- Implementation: `packages/runtime/src/kernels/attention.ts`, `gemm.ts` (shared fragments),
  `packages/runtime/src/runtime/recipe-builders/attention.ts`, `_h_attention` in the exporter.

## Addendum (2026-08-11) — additive mask input (revises decision 4)

An optional fourth input `mask` brings EmbeddingGemma's banded bidirectional attention onto the
fused path. Acceptance is narrow:

- f32, rank 4, shape exactly `[1, 1, M, N]`, additive (`S' = S + mask`, broadcast over B·H). Bool,
  `[B, 1, M, N]` and `[1, H, M, N]` fail loudly; a run-time mask revises the contract together with
  its index arithmetic.
- The contract uses `ContractBase.maxArity` (arity 3..4 — one optional slot, not variadic).
- Mask × `attentionCompute: "a8"` fails loudly (① QK in a8 has no epilogue); it is not silently
  dropped to f32.

Bit identity holds: ① adds `S' = fl(fl(Σqk) + mask[m·N + n])` once in its store epilogue, the same
rounding position and count as the decomposed `bmm` → `add`. ②, ③ and the mask-free ① are
unchanged (snapshots). Masked parity (five shapes, band mask with −inf, two fault injections) is a
permanent check for the f32 path only; s16 / c16 × mask have no permanent check
(`docs/limitations.md`).

The exporter's `_h_attention` accepts this mask. transformers (Gemma 3) passes a bool band mask;
`_additive_attn_mask` in `normalize.py` turns it into `where(mask, 0, −inf)` — the same op and
constants as torch's own bool-to-additive decomposition, byte-identical to the decomposed path's
mask constant (tested). Preservation stays opt-in per target. Measured on EmbeddingGemma-300m:
nodes 1,681 → 1,273, wall time neutral; the value is memory, the opened mask contract and fewer
nodes (research/2026-08-11-skinny-m-geometry.md §3).
