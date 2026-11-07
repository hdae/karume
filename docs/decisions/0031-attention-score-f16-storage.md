# 0031: f16 storage of the attention score S (`attentionScoreStorage`, `pack2x16float`)

- Status: accepted (2026-08-04; visual check accepted by the maintainer); partly superseded by
  [0058](0058-numerics-opt-in-contract.md) (decision 1 with [0128](0128-default-preset-original-weights.md):
  runtime default vs distributed default preset; decision 3's detector, addendum 2026-10-10)
- Date: 2026-08-04
- Related: [0006](0006-quantization.md) (its "activation f16: not adopted" is overridden for the
  storage of S only — decision 7) / [0018](0018-f16-weight-execution.md) (the `unpack2x16float`
  precedent; its "free" does not carry over — decision 2) / [0023](0023-fused-attention.md) (fused
  attention, the writer and readers of S) / [0028](0028-f16-compute-variants.md) (`:c16` compute
  variants; the reading of the storage axis is decision 6) / [0030](0030-attention-a8-execution.md)
  (attention a8; its decision 1, no `shader-f16`, is untouched) / design recon:
  [2026-08-04-intermediate-f16-design.md](../research/2026-08-04-intermediate-f16-design.md) (the
  first ruling on f16 intermediate buffers, the bandwidth model, the discovery of the write-side
  read-modify-write hazard of `pack2x16float`)
- Checks (as of 2026-10-18): `packages/runtime/tests/gpu_attention_s16_test.ts` and the sha256
  regression checks of `packages/e2e/tests/e2e_anima_test.ts` (its default-preset cases use s16).
  The original e2e check (`packages/runtime/tests/e2e_anima_s16_test.ts`) was deleted on 2026-08-15;
  its tolerances and floor are historical.

## Context

The recon of f16 intermediate buffers measured that, in 1024 px turbo w8a8 + attention a8, **S (the
attention score) alone takes 1,024.0 MiB of the 1,224.5 MiB DiT transient memory (83.6%)**. The
expected time gain was small (bandwidth-bound kernels are only 21.48% of DiT GPU time). The
maintainer ruled on 2026-08-04 to start small: the storage of S only, as the first phase; full f16
intermediates wait for measurements and are re-prioritised against VAE tiling. The Q0 simulation
passed at 1024 px (the extra degradation of S in f16 is at noise level: `w8a8-qkpv-s16` 1.043 × vs
the control `w8a8-qkpv` 1.066 ×, S alone 1.015 ×).

## Decision

1. **Opt-in `SessionOptions.attentionScoreStorage?: "f32" | "f16"`**, default `"f32"`. It is
   **a second axis orthogonal to `attentionCompute`**: S is an intermediate buffer, so "at which
   precision to compute" and "at which precision to store" are chosen separately. The intended pair
   is `a8 × s16`. (Superseded by ADR 0058 decision 2 on 2026-08-14 and ADR 0128 on 2026-10-17: the
   runtime default — every knob omitted — is the reference path and resolves to `"f32"`
   (`packages/runtime/src/runtime/session-options-check.ts:123`); the distributed default presets of
   anima and wan, `f16+dit8-a8-attn8-s16`, declare `attentionScoreStorage: "f16"`.)
2. **The encoding is `pack2x16float`** (two elements per word in `array<u32>`, core WGSL). It does
   not request `shader-f16`, so it combines with a8 and **ADR 0030 decision 1 stays untouched**. The
   mechanism is that of the f16 weights of ADR 0018, but "free" does not carry over: weights are
   read-only, while **S has a writer**. Packing two elements into a u32 turns the partial write of
   the scalar path (`if (ocol + k < dims.n)`) into a non-atomic read-modify-write of a shared word.
   **s16 is therefore v4-only** (one thread writes four consecutive columns, exactly two words,
   exclusively), and generation fails loudly on `!v4` (`assertScoreStorageSupported` in
   `packages/runtime/src/kernels/score-storage.ts`, which owns the parts).
3. **Eligibility is `D % 4 == 0 && N % 4 == 0`**; an ineligible shape falls back to f32 storage
   silently (as linear's `k % 4` and ADR 0030 decision 5). With this definition every f32 / a8
   writer and reader is v4 whenever the shape qualifies, so **no scalar s16 variant exists**. The
   three kernels (① QK write, ② row statistics, ③ PV read) switch **together**; mixed storage forms
   are structurally impossible. The measured shapes (D = 64 / 128, N = 512 / 4096) all qualify.
   (Superseded by ADR 0058, addendum 2026-10-10: the route is decided by the pure function
   `attentionRoute` (`packages/runtime/src/runtime/attention-route.ts:56`), and the series checks
   test full eligibility as a ratio — `assertAttentionSeatsComplete`,
   `packages/runtime/tests/helpers/pipeline-census.ts` — instead of a key count of one or more.)
4. **The numeric contract is atol = 0.** The only rounding is the `pack2x16float` store (the
   `unpack` read is exact), so "**the s16 variant ≡ the f32 variant with S rounded on the host by
   `Math.f16round`**" holds by definition. This is a stronger contract than the f16 tile compute of
   ADR 0028, where rounding happens inside the dot product. Bit-identical on 5 shapes; the
   `pack2x16float` rounding matched `Math.f16round` on all 22 tie values (round to nearest even).
   The a8 ③PV qP check (±1 step, mismatch rate < 2%) was re-derived with rounded S: 0 / 27,200
   mismatches.
5. **The key gets a third suffix `:s16` at its end** (no clash with the weight storage suffix
   `:wf16` or the compute suffix `:c16`; the order is fixed in one key builder, `scoreKeyPart`).
   f32 is the empty string, so existing keys and WGSL snapshots stay byte-identical.
6. **Combining with `attentionCompute: "f16"` (`:c16`) fails loudly.** `:c16` holds S as
   `array<f16>`, a different storage form; silently choosing either would hide from the diagnostics
   which contract the rounding sequence follows. ADR 0028's "no product exists" follows from q / k /
   v all being activations; **S is an intermediate buffer and can have a storage axis** — this ADR
   is the reference for that reading.
7. **ADR 0006's "activation f16: not adopted" is overridden for the storage of S only.** The
   skeleton "quantize the storage only, compute in f32" stays: S is stored narrower, and the
   reductions and `exp` stay f32. Intermediate buffers other than S (strided / elementwise /
   `quantize_rows` inputs) are not started; the first-phase measurement (time about −2%) is the
   input for re-prioritising them against VAE tiling. They are parked in the
   [backlog](../backlog.md) ("parked" section, full f16 intermediates).

## Measurements (2026-08-04, 1024 px turbo, `--dit i8`, linear and attention a8, A/B/A/B)

DiT transient memory 1,224.5 → 712.5 MiB (−512.0 MiB, exactly the design value of halving S); DiT
GPU time −2.0%; dispatches per step unchanged (3,311, no conversion dispatch). Row statistics gained
1.13–1.39 ×, PV nothing (the a8 integer GEMM is instruction-bound). The default-path sha256 checks
were unchanged. Full record: commit `99a3c649`, this file.

## Detection limits

1. **Swapping even and odd elements in the scalar read `score_at` of ② row statistics cannot be
   detected numerically.** Eligibility `N % 4 == 0` makes every row start at an even index, so the
   swap only permutes pairs inside a row (amax is permutation-invariant; Σexp matched to the last
   ulp in measurement). The "row-relative index" trap of ADR 0018 is neutralised by the eligibility
   condition. The working detector is one WGSL snapshot.
2. **u32-packed storage with a writer forces "one thread, two elements, even boundary" on every
   kernel** (decision 2) — the main cost of extending f16 storage to the elementwise / strided
   families. S was safe because it has one writer and is v4.
3. In the production form (w8a8 + attention a8 + s16) no numeric floor fits (the s16 effect is
   buried in the earlier quantization noise, as in ADR 0030); the later an opt-in sits in the stack,
   the less a numeric detector can see it.

## Consequences

- The 1024 px DiT transient memory drops to 712.5 MiB; halving S is now usable with a8 (closing the
  gap noted in ADR 0030's consequences). The 1024 px peak then sits in the VAE decode (transient
  1,536 MiB), which [ADR 0033](0033-vae-fixed-tile-decode.md) addresses.
- The time gain is small (about −2%); full f16 intermediates are judged on this measurement.
- Images differ from f32 / w8a8 + a8 (trajectory divergence), but the **visual check was accepted by
  the maintainer** (2026-08-04, "I hardly feel any change"; four before/after images at 512 and
  1024 px), consistent with Q0's "s16 adds noise-level degradation".
- The "non-GPU fixed cost" measured here (695–704 ms per step) was later attributed by
  [ADR 0032](0032-deserialize-submit-and-optin-timing.md) to timing overhead and CPU/GPU
  serialisation, and removed there.

## Addendum 2026-11-07: `pack2x16float` rounds differently between backends

Decision 4 rests on `pack2x16float` rounding to nearest with ties to even, which it does on Vulkan on the RTX 3080 Ti. WGSL leaves
that rounding open, and on D3D12 on the RTX 5070 Ti (Chrome 156) it rounds toward zero: the
`score-f16-store` check mismatched in 1,616 words, and rounding toward zero predicts the count and the first failing word of both of
its cases ([research](../research/2026-11-07-gemma4-f16-kv.md) §6).

**The difference is recorded and accepted.** The contract of decision 4 now reads: the output equals the f32 variant with S rounded on
the host **in this backend's rounding**. The check passes on either rounding, on every value alike, and records which, and any other
word fails, a mix of the two included.

**Why not round in integers.** That store passes the check in Chrome and gives every backend the same S, at 3–5% of the attention route
on both machines measured (research §6). It would not give the same output across backends, because the reference values are rows per
environment for other reasons as well (ADR [0106](0106-device-keyed-references.md)). What it removes is the bias toward zero of at
most one f16 step, on an S computed from int8 inputs in the shipped presets. Keeping the store also leaves Chrome's numbers where
0.17.0 has them, and the geometry tables measured on the kernels that run.

**Not measured.** The effect of rounding toward zero on a final image, and which rounding Metal and other Vulkan drivers do. The check
covers finite normal f16 values only: subnormals may flush and an out-of-range value makes its word indeterminate, as before.

**Reconsider when** an integer store costs no measurable time, or a use needs the same S on every backend. The full-layer KV took the
integer store because it costs nothing measurable there (ADR [0143](0143-f16-full-state-storage.md)).
