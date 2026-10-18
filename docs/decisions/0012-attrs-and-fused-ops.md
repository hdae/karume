# 0012: attrs vocabulary and the fused-op contract (preserved list of 9 ops)

- Status: accepted (2026-08-02); the attrs contract is current. The preserved list is now 12 (0015
  `leaky_relu`, 0017 `rms_norm`, 0118 `conv3d`; ADR 0023's fused attention is only in
  `PRESERVED_OP_PREFIXES_WITH_ATTENTION`, `tools/exporter/src/karume/convert.py`).
- Date: 2026-08-02
- Related: [0007](0007-op-vocabulary.md) / [op-vocabulary.md](../op-vocabulary.md) (current list) /
  [recon](../research/2026-08-02-deberta-front-recon.md)

## Decision

- **M0's "every op has empty attrs" is removed.** The contract tables (TS `packages/runtime/src/ops.ts`,
  a facade over `src/ops/`; Python `tools/exporter/src/karume/ops.py`) carry an attrs schema per op
  (allowed keys, types, value ranges). Unknown keys and out-of-contract values fail loudly. The contract
  tables define attrs meaning; ir-v2.md keeps only the frame (a per-op contract check).
- **The exporter's preserved list (decomposition suppressed) grows from gelu to the 9 ops of ADR 0007**:
  linear / layer_norm / softmax / gelu / conv1d / conv2d / conv_transpose1d / embedding / masked_fill.
  Kernels only for ops in measured graphs (conv_transpose1d and conv2d came later, ADR 0015 / 0017).
- New fused and structural ops of P2 and their main attrs (measured value ranges are narrowed by the
  kernel contract, that is the runtime capability — the rule of ADR 0007):
  - `linear` (with or without bias) / `layer_norm` (normalized_shape, eps) / `softmax` (dim)
  - `embedding` (`padding_idx` has no effect in forward; the contract accepts it and states why)
  - `masked_fill` (value: f32 scalar) / `conv1d` (stride, padding; B = 1 etc. are capability) /
    `bmm` (batched matmul) / `gather` (dim)
  - layout ops (ADR 0011): `reshape` / `permute` (dims) / `expand` / `slice`
  - `cast` (to) / `bitwise_not` (ADR 0009) / `sym_prefix_slice` (ADR 0010)
- The `masked_fill` fill value −3.4028234663852886e+38 (the smallest finite f32) does not violate the
  rejection of non-finite values (ir-v2.md); a test pins that it is **ulp-unchanged through a JSON
  round trip** (`tools/exporter/tests/test_aten_handlers.py`).

## Consequences

- "Adding an op moves the TS contract table, the Python contract table and the golden as one set" now
  includes the attrs schema. The exporter's `Emitted` carries attrs.
- softmax accepts only dim = last axis (only −1 was measured; a general dim waits for a measured need).
- native_layer_norm (3 outputs) is not brought into the IR; preserving layer_norm keeps it single-output.
  Multi-output nodes exist since ADR [0068](0068-decode-exit-multi-output.md) (decisions 1 and 3, `topk`).
