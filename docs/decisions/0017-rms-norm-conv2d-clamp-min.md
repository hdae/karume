# 0017: Op vocabulary extension — rms_norm / conv2d / clamp_min

- Status: accepted (2026-08-02); partly superseded by [0020](0020-nan-propagation-bitwise.md) (NaN of
  `clamp_min`, 2026-08-03) and [0118](0118-wan21-video-generation.md) (conv3d rejection, 2026-10-02)
- Date: 2026-08-02
- Related: [../research/2026-08-02-anima-recon.md](../research/2026-08-02-anima-recon.md)

## Context

Measured demand: 269 RMSNorms (Qwen3 113 / conditioner 43 / DiT 113, QK norms included), 37 conv2d in
the VAE decoder (32 T=1 slices of CausalConv3d plus 5 plain), and 30 `clamp(min=eps)` in channel L2
normalization; the prototype's final IR converges on the same three ops (recon §3). TS contract,
Python contract, WGSL kernel and CPU reference are extended **together in one step** (ADR 0015).

## Decision

- **`rms_norm`**: arity 2 (x, weight), attrs `{eps}` required, normalized over the last dimension
  only, weight rank 1 of the last-dimension length, f32: `y = x · rsqrt(mean(x², last dim) + eps) ·
  weight`. A row-reduce kernel of the layer_norm family (tree reduction; one pass of the sum of
  squares, so layer_norm's two-pass variance issue does not arise). Two supply routes: `aten.rms_norm`
  (diffusers `nn.RMSNorm`) is preserved; the hand-written decomposition (Qwen3 / DiT) is folded by
  `_fold_rms_norm` (ADR 0016). A form without weight, if it appears, gets synthesized ones (arity 2).
- **`conv2d`**: input [B, Cin, H, W], weight **[Cout, Cin/groups, Kh, Kw]**, bias [Cout] (arity 3; a
  missing bias becomes the exporter's zero bias). attrs `{stride, padding, dilation, groups}` are
  **required, no defaults filled in** (as conv1d in ADR 0015; H and W components). The output length
  includes dilation; the contract checks `Cin % groups == 0`, `Cout % groups == 0`, `stride >= 1`.
  Spatial padding only appears folded into conv2d's padding (the patch layer ensures it; IR pad stays
  single-axis). Tests MUST include **Kh ≠ Kw, H / W-asymmetric stride and padding, and Cin ≠ Cout,
  both ≥ 2**: a swapped weight layout has the same element count and passes shape checks.
- **`clamp_min`**: elementwise, attrs `{min}` required, f32. Making clamp's attrs optional is rejected
  (it breaks "no defaults for declared attrs, all keys matched with `Object.hasOwn`", ADR 0012);
  filling the missing side with ±finite max is what the MUST of `_h_clamp` forbids by name. NaN: the
  planned select form did not propagate NaN on the GPU (measured 2026-08-02); ADR 0020 replaced it
  with a bitwise NaN check.
- Every new kernel **writes its whole output (full-write, ADR 0004 ⑤)** and gets one case in the
  degenerate-shape harness (`packages/runtime/tests/gpu_gridstride_test.ts`) in the same step (MUST).

## Alternatives considered

- mean.dim plus rsqrt (RMSNorm decomposed): five times the dispatches at 269 sites and three
  intermediate buffers. Rejected (as in ADR 0016).
- A conv3d kernel: CausalConv3d(T=1) equals conv2d (prototype, f64 max diff 1.78e-15) and the time
  slice shrinks the weights 2.8 times. Rejected for lack of demand (superseded by ADR 0118 on
  2026-10-02: Wan2.1's video VAE needs it; `conv3d` is in the vocabulary and in `WEIGHT_SLOTS`).
- A dedicated upsample_nearest kernel: reshape → expand → reshape expresses it exactly (done in the
  patch layer). Rejected.

## Addendum 1 (2026-09-11) — RMS normalization of width ≤ 128 skips the empty upper half (K-28)

Anima's QK normalization has a last dimension of 128 but used 256 threads; the upper 128 passed a sum
of squares of 0. Based on the kernel-level and whole-generation acceptance in
[the 2026-09-10 research](../research/2026-09-10-codex-mtp-optimization.md) (section on Anima RMS
normalization, 2026-09-11), **the 128-thread variant is selected only for 1 ≤ dim ≤ 128**; other
widths keep the 256 variant.

- Only the first `+0` stage is skipped; the squares, the lower-half addition tree, the reciprocal
  square root, the product with weight and the barrier aligning row grid-stride with scratch lifetime
  are unchanged.
- The 256 variant's WGSL and key are byte-identical to before; the 128 variant is a separate key from
  the same template (`RMS_NORM_128_KEY`, `packages/runtime/src/kernels/rms-norm.ts`). No public API, IR
  or option; no GPU-touching initialization or mutable module global.
- 128 threads for every width is rejected: above 128 the terms summed per thread change, so the
  addition order differs. 64 threads would need separate verification.
- Fusing the preceding permute was prototyped; the extra gain is small, so it is not productized.

Checks: u32 equality with the 256 variant, CPU reference under the canonical op tolerance, edge widths,
degenerate shapes, full-write and non-finite propagation; Anima latents of every step / PNG and Qwen's
full logits were compared. The NaN payload is outside the contract; its position is checked. A re-run
on the M2 machine was open on 2026-09-11.

## Addendum 2 (2026-10-31) — the prefetch form of the 256-wide kernel (v2, perf-ledger K-99)

A lone row of the 256-wide kernel was latency-bound in its output loop: after the reduction each
thread re-read its x and weight words behind a runtime trip count (`while (o < dim)`). Where the row
is a multiple of 256 elements up to 4096 (`RMS_NORM_PREFETCH_MAX_DIM`), the kernel and the RMS → add
fusion of ADR [0099](0099-rms-norm-add-fusion.md) now take a **prefetch form**: the weight words are
loaded into registers once per workgroup before the row loop, the row's x words (and the add's
residual words) before its reduction, and the output loop runs a constant `dim / 256` trips.

- **Numerics unchanged by construction**: the reduction is the same text (`rmsNormInvWgsl`), and the
  value is the same expression (`rmsNormValueWgsl`, `x · inv · weight` left to right) with the
  registers as its operands; the add keeps its XOR rounding barrier. Only the source of each operand
  changes.
- **Key scheme**: the trip count is in the WGSL, so it is in the key —
  `rms_norm:v2:f32:lastdim:wg256:trip<dim / 256>` and
  `rms_norm_add:v2:rms_norm:v2:f32:lastdim:wg256:trip<n>:<order>:xor-round`
  (`rmsNormWorkgroupKey` / `rmsNormAddKey(order, dim)` in `packages/runtime/src/kernels/rms-norm.ts`).
  Every other width of the 256-wide kernel keeps v1's text and key (`RMS_NORM_KEY`); the 128-wide
  form (addendum 1) and the subgroup form (ADR [0100](0100-rms-subgroup-reduction.md)) are unchanged.
  The add has no v1 path left: its rule accepts only 256 / 1536 / 2560, all inside the range, and the
  generator throws on any other width. The v1 add WGSL is kept test-only in
  `packages/runtime/tests/fixtures/wgsl-baseline/`.
- **Where it was measured faster** (RTX 3080 Ti, Node + Dawn, dependent chain, cold and warm,
  interleaved rounds — `outputs/diag/qat-0151-2026-10-31/rms-loop/`, untracked): at every multiple of
  256 from 256 to 4096 and every row count measured (1 / 2 / 4 / 8 / 35 / 768 / 8,192, and 2,048 /
  32,768 at 256 and 1280). M = 1: 2.65 → 2.31 µs at 256, 5.09 → 2.80 at 1536, 9.93 → 3.80 at 4096;
  the add 5.23 → 3.02 at 1536. At large M the gain shrinks to a few per cent at 256 and stays
  13–22 % at 1536–4096. The only ratio above 1 was one warm run at 256 × 8,192 rows (×1.001); the
  three other runs of that shape were ×0.98. Wider rows are unmeasured and keep v1.
- **The `rmsNormRope` rule (ADR [0040](0040-fusion-pass.md))** keeps its kernel, which shares v1's
  reduction and value text; the plain path it stands for now runs v2 at the Wan widths 1536 / 3072,
  which writes v1's words, so the fused and plain paths still agree (`gpu_rms_norm_rope_fusion_test.ts`
  compares them in Sessions).

Checks: `gpu_rms_norm_prefetch_test.ts` compares v2 with v1 as u32 words at every width of the range,
rows 1 / 4 / 8 / 33 and 768, natural and on three workgroups (grid-stride), on adversarial rows (zero
rows, ±inf, NaN payloads, subnormals only, overflowing and underflowing sums of squares, the mean at
eps, raw random words) with special weights and residuals; reordering the value, dropping the add's
barrier or a wrong prefetch stride each fail it. The degenerate-shape harness carries a v2 case. Every
family lane's sha256 rows are unchanged. End to end (Node + Dawn, Gemma 4 E2B `i4-fast`, greedy, v1
substituted back): GPU busy 5.79 → 5.42 ms per token regular, 5.99 → 5.64 QAT; the text is identical.
Not checked on D3D12 or Metal.

## Addendum 3 (2026-11-13) — a rounding barrier on `x · inv` in the prefetch form (v3)

On Chrome / Metal (Apple M2) the prefetch form did not write v1's words: with x and the weight already in registers
before `inv` exists, Metal computes `x · inv · weight` as `(x · weight) · inv`. Recomputing both orders in f32
reproduces the M2 kernel-check record exactly (known-issues, the Metal section). The same text therefore does not
fix the order where the operands are known before `inv`.

- The prefetch form's value is `bitcast<f32>(bitcast<u32>(xv[t] * inv) ^ params.rounding_mask) * wv[t]`. The
  rounding mask is the fourth word of `rmsNormParams` (already 0, now read by the plain kernel too), so `x · inv` is
  rounded before the weight multiplies it on every backend. The add forms keep their outer barrier around this value.
  v1 and the `rmsNormRope` kernel load their operands after `inv` and keep their text.
- Keys: `rms_norm:v3:f32:lastdim:wg256:trip<n>` and `rms_norm_add:v3:rms_norm:v3:…:<order>:xor-round`.
- On Node + Dawn (RTX 3080 Ti, Vulkan) v3 writes v1's words on the adversarial check, and the sha256 rows of the
  core, gemma4, gemma4-qat-mobile, wan and moss lanes are unchanged (the other lanes were not re-run). The RMS → add
  test that the barrier executes now checks both XORs under a weight that is not 1 (with a weight of 1 they cancel). The dependent-chain timing against v2 is ×0.998–1.006 (dims 256 / 1536 / 2560 / 4096, rows 1
  and 768, cold and warm; `outputs/diag/rms-barrier-2026-11-13/`, untracked). On Chrome / Metal (Apple M2 and M5) the
  prefetch check and the `rmsNormRope` twin now pass (`outputs/bench-browser/kernel-check_2026-11-13T21-22-18.json` /
  `…T20-59-22.json`, untracked).
