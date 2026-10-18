# 0015: conv family extension — groups / dilation / conv_transpose1d / zero-bias synthesis

- Status: accepted (2026-08-02); the `leaky_relu` select-form rationale overridden by [0020](0020-nan-propagation-bitwise.md)
- Date: 2026-08-02
- Related: [recon](../research/2026-08-02-sbv2-chain-recon.md) / [0012](0012-attrs-and-fused-ops.md)

## Context

Measured SBV2 demand: sdp's DDSConv (depthwise, groups = 192, dilation 1 / 3 / 9), dec's ResBlock1
(dilation 1 / 3 / 5, 30 convs), five ConvTranspose1d and the bias-free `conv_post`.

## Decision

- **conv1d gains the attrs `groups` / `dilation`.** The old value "no field means fixed at 1, so no path
  can silently run a value other than 1" carries over as "a present field MUST be declared; every key is
  matched with `Object.hasOwn`; a mismatch fails loudly" (no default filling for declared attrs — the
  ADR 0012 rule). The output-length formula includes dilation, and `Cin % groups == 0` /
  `Cout % groups == 0` are contract checks. The TS contract, the Python contract, the WGSL kernel and
  the CPU reference MUST be widened **together in the same change** (none of the four may go first).
- **A bias-free conv gets arity 3 through the exporter's zero-bias synthesis**; kernels and contracts
  have no arity branch (proven in the prototype).
- **New op `conv_transpose1d`:**
  - Weight layout **[Cin, Cout, K]** (transposed relative to conv1d's [Cout, Cin, K]). A swap can still
    match the element count and pass the shape check, so tests use **asymmetric channel counts**.
  - The saved form may omit trailing defaults (a 4-argument form exists); the handler pads the
    arguments with None to full length before filling defaults.
  - `stride >= 1` is a MUST contract check: stride 0 never advances the kernel loop and **hangs the
    GPU** instead of raising.
  - The output length L·u is checked together with `pad = (k − u) // 2` (SBV2 satisfies it). Other forms
    fail loudly until a need appears.
- **`leaky_relu` is a new op with a required `negative_slope` attr** (here because it is dec-only and
  lands with the conv family):
  - dec **mixes** slope 0.1 (ups / ResBlock) and 0.01 (the last stage, torch's default by omission). A
    default would make one of them silently wrong, so the attr is required with no default filling.
  - It is preserved (not decomposed): the decomposed form (gt_scalar + mul + where) grows intermediate
    buffers 1.5–2× and breaks the memory-estimate premise.
  - The WGSL **select form** was chosen because WGSL `max` does not guarantee NaN propagation. ADR
    0020 overrode that rationale (compilers fold select into max); NaN propagates because **x is in both branches**.

## Remaining vocabulary additions (no ADR)

where / clamp / ge · le · gt (Scalar and Tensor) / bitwise_and / sum over bool (→ i32) / log1p / cumsum
(last axis) / f32 for expand are ordinary measurement-based contract-table extensions under ADR 0009 /
0012. `eq(x, 0)` is normalized to `bitwise_not(cast(x, bool))` with no new op (it follows from the cast
rule "x → bool is x ≠ 0"; `_eq_zero_to_not_bool`, `tools/exporter/src/karume/normalize.py`).

## Alternatives considered

- A separate depthwise op: duplicates the conv1d contract and kernel and blocks the general groups form
  (SBV2 needs depthwise only, but the contract can be general). Rejected.
- conv_transpose1d rewritten as conv1d (zero insertion plus a normal conv): intermediates grow u-fold
  and hit the dec memory estimate (naive sum 6.97 MB × Ty). Rejected.
