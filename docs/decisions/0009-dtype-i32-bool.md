# 0009: Execution of semantic i32 / bool and i64 normalization at the exporter boundary

- Status: accepted (2026-08-02); "do not add `where`" withdrawn by [0015](0015-conv-family-extension.md).
- Date: 2026-08-02
- Related: [../research/2026-08-02-deberta-front-recon.md](../research/2026-08-02-deberta-front-recon.md)
  (DeBERTa front: i64 inputs; the bool mask path depends on input values and cannot be folded).

## Decision

- **Execution of semantic i32 / bool is enabled.** Both are already in the IR's declared vocabulary
  ([ir-v2.md](../ir-v2.md)); what changes is execution: input transfer (`RUNTIME_SUPPORT.io`), the
  dtype sets of the op contracts, and the element type as a parameter of elementwise codegen. **The
  element type is part of the WGSL canonical key** (the codegen determinism invariant is unchanged).
- **bool is stored on the GPU as u32 0 / 1** (storage buffers have no 1-bit type; proven in the prototype).
- **The IR has no i64.** torch's default integer i64 is **normalized to i32 at the exporter
  boundary**, always with a range check (|x| > 2^31 − 1 fails loudly). The conversion into a world
  without 64-bit integers happens in that one place (`tools/exporter/src/karume/convert.py`).
- New ops: **cast** (between f32 / i32 / bool; f32 → i32 truncates as torch does, stated in the
  contract) and **bitwise_not** (bool negation — needed to invert attention masks). `where` was not
  added at the time (frozen allowlist, ADR 0007); withdrawn by ADR 0015, "remaining vocabulary
  additions": a measured graph needed it, and it is implemented as `WHERE_OP`.
- **The public `Tensor` becomes a dtype-discriminated union** (a partial revision of ADR 0008):
  `data` is f32 = `Float32Array` / i32 = `Int32Array` / bool = **`Uint32Array`** (not `Uint8Array`;
  the same 0 / 1 as the GPU storage), symmetric for inputs and outputs.

## Alternatives considered

1. **Add i64 to the IR** (the prototype's way) — WebGPU has no 64-bit integer buffers, so a
   conversion to i32 happens at transfer anyway; the vocabulary grows without adding expressiveness.
2. **Carry indices as f32** — silent precision loss above 2^24 (against fail loudly);
   the contract table loses checkability.
3. **i32 normalization at the boundary (chosen)** — minimal vocabulary, checkable, one conversion
   point. DeBERTa's real ranges (T ≤ 512, vocab 22012) fit easily in i32.

## Consequences

- The TS (`packages/runtime/src/ops/contracts.ts`) and Python (`tools/exporter/src/karume/ops.py`)
  contract tables are extended as one set, each op declaring its allowed dtypes (f32-only ops stay
  f32 — no blanket enabling).
- Readback and upload dispatch on dtype (4-byte elements for every type). Each new i32 / bool
  operation adds its CPU reference and golden at the same time (ADR 0005).
