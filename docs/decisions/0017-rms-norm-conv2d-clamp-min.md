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
