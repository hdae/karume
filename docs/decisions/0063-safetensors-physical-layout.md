# 0063: safetensors physical layout — no gaps, element alignment, fixed write order

- Status: accepted (2026-08-15 — records a ruling already implemented on both sides); partly
  superseded by [0108](0108-container-format.md) (the writer's fixed order and the rule for new storage
  dtypes, effective 2026-09-24); the reader rules stay in force for auxiliary assets.
- Date: 2026-08-15
- Related: [0003](0003-ir-v1.md) (container format) / [0018](0018-f16-weight-execution.md) (f16
  storage — origin of the odd/even problem) / [0019](0019-i8-weight-execution.md) (i8 storage) /
  [0069](0069-packed-w4-storage.md) (packed 4-bit — revised together with this ADR) /
  implementation: reader `packages/runtime/src/format/safetensors.ts`, Python copy of its rules
  `assert_reader_layout` in `tools/exporter/src/karume/verify.py`

## Context

- The safetensors specification fixes neither the order nor the alignment of tensors. The Karume
  reader maps TypedArray views directly onto the file buffer, so **the absolute offset of each tensor
  must be aligned to its element size**. An F16 tensor with an odd element count (byte length ≡ 2 mod
  4) followed by F32 / I32 moves the offset off a multiple of 4.
- The HF `safe_open` **does read** such files, so upstream tools do not detect the violation. The
  detector has to be our own.

## Decision

- **Reader (runtime).** The reader requires that the tensors cover the data section without gaps and
  that every tensor is aligned to its element size. A violation fails loudly with `SafetensorsError`
  (gap, alignment, exact match of the declared byte length).
- **Writer (exporter).** The write order is fixed as **`F32 → I32 → I4 → even-element F16 →
  odd-element F16 → I8`** (`emit.py`, without `safetensors.torch.save_file`): descending by alignment
  unit, so odd-element F16 and I8 tensors sit at the end and break no later alignment. `I4` requires a
  4-byte-aligned start (bound as `array<u32>` — ADR 0069 decision 2) and its section is always a
  multiple of 8 bytes, so it joins the F32 / I32 group (ADR 0069 addendum 2). (Superseded by ADR 0108
  decision 7 on 2026-09-22, effective 2026-09-24 when step 3d `9e905d9f` removed the old reader: the
  container writer bakes 64-byte block alignment and block lengths that are multiples of 4. `emit.py`
  now writes containers; auxiliary assets are single-tensor safetensors written with
  `safetensors.numpy.save` (`table_payload` in `tools/exporter/src/karume/dist.py`), where order does
  not matter.)
- **Self-check after writing.** `verify.assert_reader_layout` applies a copy of the reader rules, so a
  file that was written but cannot be read never reaches a distribution. Today it runs on the input
  of the migration CLI (`karume migrate`, the only caller that accepts the dialect dtypes `I4` / `I2`)
  and right after a recipe writes an asset safetensors (e.g. `tools/export-recipes/irodori/dacvae`).
- Zero padding at GPU residency (ADR 0018 / 0019) is **inside the buffer**; file bytes are not padded.

## Consequences

- The reader rules and the writer rules change **as a pair** (a cross-package invariant: a written
  file that breaks the reader rules fails `assert_reader_layout`).
- A new storage dtype is first decided **as a revision of this ADR** (superseded by ADR 0108 on
  2026-09-22: storage dtypes are decided on the container — bf16, for example, by
  [0131](0131-bf16-weight-execution.md); the safetensors reader accepts no quantized dialect dtype).
- The asset safetensors section of `docs/limitations.md` is a summary that points to this ADR.

## Addendum (2026-09-22) — ADR 0108 replaces the weight layout contract

ADR 0108 decision 7 replaces the layout and write-order contract for weights; it took effect on
2026-09-24 (step 3d, `9e905d9f`). The safetensors reader stays for auxiliary assets, with the
coverage and alignment checks, and rejects `I4` / `I2`.
