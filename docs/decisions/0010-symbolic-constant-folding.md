# 0010: Folding symbol-dependent constants at Tmax and the symbolic prefix slice

- Status: accepted (2026-08-02)
- Date: 2026-08-02
- Related: [../research/2026-08-02-deberta-front-recon.md](../research/2026-08-02-deberta-front-recon.md)
  (the relative-position bucket table is the source of 14 of the 27 unsupported op kinds).

## Decision

DeBERTa's relative-position bucket table (the T-dependent subtree of arange / sign / log / ceil /
clamp / comparisons / bitwise_and …) is **evaluated by the exporter at Tmax (the model's maximum
sequence length), baked into an i32 constant, and cut out at run time with a symbolic prefix slice**.

- IR v1 gains **semantic i32 initializers and the storage dtype `i32` (raw int32)**, an explicit
  exception to "the storage vocabulary encodes f32 values" (storage moved on 2026-09-24, addendum).
- The IR op **`sym_prefix_slice`** is added: attrs `{ sym, slices: [{dim, coeff, offset}] }`, reusing
  the dimension language `coeff·sym+offset` of [ir-v2.md](../ir-v2.md).
- A subtree may be folded only if evaluation at Tmax commutes with the prefix. **Being on an allowlist
  is not enough**: the exporter **evaluates at two different T and compares the prefixes**; a
  mismatch fails loudly (proven in the prototype). Revised on 2026-08-13 and 2026-10-10 (addenda).

## Alternatives considered

1. **Run the integer arithmetic on the GPU** — needs i32 elementwise, comparisons, sign, ceil, log
   and clamp, makes three open ledger items (WGSL NaN comparison / sign(NaN) / max not propagating
   NaN) load-bearing, and keeps a bug class where **a 1-ulp difference at a bucket boundary shifts a
   gather index by one**.
2. **Precompute on the host as an extra input** — the lightest, but breaks "one model file is
   self-contained" and the thin public surface (ADR [0008](0008-public-api.md)).
3. **Fold at Tmax + prefix slice (chosen).**

## Rationale for the choice

1. **The bucket-boundary bug class is removed by construction**: torch computes the table itself.
2. Unsupported ops drop from 27 kinds to about 13; the WGSL NaN items stay off this path.
3. The prototype proved DeBERTa end-to-end with the same scheme.
4. Unreleased: the incompatible IR v1 revision needs no shim or migration (ADR [0003](0003-ir-v1.md)).

## Consequences

- Tmax is a model attribute (512 for DeBERTa); a longer run-time sequence fails loudly when binding.
- A subtree that does not commute with the prefix is not folded; it is listed as unsupported.

## Addendum (2026-08-13) — commutativity is enforced in two stages

The claim that the two-point check (Tmax / Tmax − 1) guarantees commutativity was unsound: when
`scalar_tensor(T)` lifts the symbol into tensor **data** (for example `(T − Tmax)(T − (Tmax − 1))`,
zero at both points), the check matches by chance and other T give silent wrong values (reproduced).
Now: (1) the symbol is rejected at its **consumption position** — `SYMBOL_EXTENT_ARGS` allowlists only
extent arguments (lengths, shapes); a subtree reaching a value position is not folded and falls back
to normal lowering (export fails outside the vocabulary); (2) the two-point check stays as a safety
net, comparing **bit-exactly** instead of with `torch.equal` (`-0.0 == 0.0` no longer passes —
`_bitwise_equal` in `tools/exporter/src/karume/convert.py`).

## Addendum (2026-09-10) — dynamic `view` extents

A dynamic `view` extent does not guarantee prefix commutativity either (`arange(2*T)` laid out as
`[2, T]` moves the second row's start with T). A view with a symbolic dimension is folded only when it
adds or removes unit axes; otherwise the input prefix is cut first and the runtime `reshape` runs it.

## Addendum (2026-09-24) — storage `i32` moved to the codec ledger

ADR [0108](0108-container-format.md) decision 17 moved storage to the container's binding table; raw
int32 is the codec `i32` of `CODEC_LEDGER` (`packages/runtime/src/format/container/codecs.ts`).

## Addendum (2026-10-10) — positions on a symbolic axis are excluded by structure

Referring to a **position** on a symbolic axis turns T into element values (`arange(T)[tensor([-1])]`
is T − 1; `arange(T)[-2:]` is `[T − 2, T − 1]`), and a later comparison may match at both points by
chance. `_classify_foldable` therefore excludes, by structure and without reading index values:
`index` with an index tensor on a symbolic axis; `slice` with a negative start on a symbolic axis;
`cat` along a symbolic axis; `repeat` of a symbolic axis two or more times; `bmm` contracting over a
symbolic axis; and `index` with a bool / uint8 mask on any axis (a mask of rank ≥ 2 spans several
axes and the output length depends on its values). A non-negative `slice` start counts from the
front and is still folded. Excluded subtrees fall back to normal lowering. Neither addendum adds
evaluation points as the remedy.
