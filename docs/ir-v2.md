# IR v2 specification

Karume's graph format. It makes ADR [0003](decisions/0003-ir-v1.md) (v1) and
ADR [0108](decisions/0108-container-format.md) (v2 — the storage declaration moved out to the container's binding table)
concrete. An incompatible revision that builds on the proven structure of the prototype IR (v0) and adds
versioning and capability declarations. It defines the settled scope; extensions are made by revising this
document. The **physical format** that carries the graph (`krm` / `krg`, block / part, the codec ledger) is
specified canonically in [container-v1](container-v1.md); this document specifies only the graph JSON.

Revision history (during 0.x there is neither dual reading nor a shim — old formats are converted with the migration CLI ([container-v1](container-v1.md) §12).
Grounds: the Consequences of ADR [0108](decisions/0108-container-format.md) and decision 18 (version branching); the revision procedure is in ADR 0003):

- M0: first edition.
- M1-P2: semantic dtypes i32 / bool (ADR [0009](decisions/0009-dtype-i32-bool.md)),
  layout ops (ADR [0011](decisions/0011-layout-strategy.md)), the attrs vocabulary and fused ops
  (ADR [0012](decisions/0012-attrs-and-fused-ops.md)), **i32 initializers / storage `i32` /
  `sym_prefix_slice`** (ADR [0010](decisions/0010-symbolic-constant-folding.md)).
- M1-P3 (wave 3): a group of 10 math ops — `where` / `clamp` / `ge_scalar` / `le_scalar` /
  `gt_scalar` / `ge` / `bitwise_and` / `log1p` / `cumsum` / `leaky_relu` (required by sdp's spline and
  dec. The design ruling on `leaky_relu` is in ADR
  [0015](decisions/0015-conv-family-extension.md)). Alongside, **bool input to `sum` → i32**
  and **f32 for `expand`** were allowed, and an **output dtype mapping** column (slot 0 → output, identity by
  default) was added to the contract table.
- M1-P3 (waves 4 and 5): the second layout group `slice` / `cat` (**the only variable-arity op in IR v1**) / `pad` /
  `flip` (ADR [0014](decisions/0014-layout-ops-full-write.md)). Extension of the conv family —
  `dilation` / `groups` added to the attrs of `conv1d` (**required, no default filled in**), and
  `conv_transpose1d` newly added (ADR [0015](decisions/0015-conv-family-extension.md)).
- M1-P4 (wave 1): added `rms_norm` / `conv2d` / `clamp_min` (ADR
  [0017](decisions/0017-rms-norm-conv2d-clamp-min.md)). `conv2d` was an op on the preserved list
  without a kernel; it showed up in measurement in Anima's VAE decoder, so it **became executable**.
  A new value shape enters attrs: **two components `[H, W]`** (`stride` / `padding` / `dilation` of
  `conv2d`).
- perf-a: added the fused `attention` (ADR [0023](decisions/0023-fused-attention.md)).
  **Arity 3 or 4, rank-4 head-first, and the attrs `scale` is a half scale** (`√scale_factor`, applied to
  both q and k). The optional fourth input is an **additive mask** (f32, exactly `[1,1,M,N]`, broadcast
  to B·H — the 2026-08-11 revision, ADR 0023 addendum). A bool mask / `[B,1,M,N]` and the like, causal /
  dropout are not in the vocabulary and all fail loudly at the exporter boundary (GQA / MQA are accepted as an
  evenly-dividing broadcast by the 2026-08-17 revision (ADR 0067 decision 1) — the `attention` entry below). SDPA preservation is
  **per target**, so existing IR in decomposed form stays valid as it is.
- Model expansion (2026-08-13): added `upsample_bilinear2d` as **layer 1** (step 3 of the decision procedure
  of ADR 0043 — an atom inside Core ATen. The provisional practice of that time was resolved by the entry-door model of 2026-08-14
  (ADR [0059](decisions/0059-op-vocabulary-entry-doors.md), now the Core ATen layer)). attrs has only `output_size`, and
  **there is no field for anything other than `align_corners = True`**. No effect on existing IR (only a new op name
  is added).
- Model expansion (2026-08-13): added `deform_conv2d` as **layer 1'** (ADR
  [0055](decisions/0055-deform-conv2d.md) — an atom outside Core ATen, requested by the BiRefNet family).
  **Arity 5** (x / weight / offset / mask / bias) — the largest among fixed arities
  (previously 3 for `where` / the conv family and 3 to 4 for `attention`).
  attrs has only `padding`; `stride` / `dilation` / `groups` / `offset_groups` have no
  field (= fixed at 1). No effect on existing IR.
- Model expansion (2026-08-14): added `gru_scan` / `gru_scan_reverse` as **layer 2** (now the extended composite layer —
  ADR 0059) (ADR
  [0056](decisions/0056-gru-scan.md) — composite, but the semantic information "the time axis is decided at run time" is lost in
  the decomposed form. Requested by the two-layer BiGRU of vowel-detector). **Arity fixed at 4, empty attrs**;
  the op handles **only the hidden-side recurrence** (the input-side GEMM is the caller's `linear`). The scan direction is carried
  not by attrs but by the **op name** (the same approach as `gelu` / `gelu_tanh`). No effect on existing IR.
- Model expansion (2026-10-31): added `gated_delta_scan` as the **extended composite layer** (ADR
  [0135](decisions/0135-gated-delta-rule-scan.md), accepted — the recurrence of a Gated DeltaNet layer, requested by the
  MOSS-Image text encoder; upstream's Python loops specialise the length axis in export). **Arity fixed at 5, empty attrs**;
  the value form only (zero initial state, no state output). No effect on existing IR.
- autoregressive (2026-08-17): added `argmax` as the **Core ATen layer** (ADR
  [0068](decisions/0068-decode-exit-multi-output.md) decision 2 — the exit of greedy decoding.
  Measured as core by `torch.Tag.core`, so only a ledger NOTE is required (ADR 0059), but decision 2 sits in
  the same ADR as allowing multiple outputs). **Empty attrs, arity 1, input f32 → output i32** (indices), and
  **rank preservation** (the last dimension collapsed to 1) is held by the shape rule. No effect on existing IR.
- autoregressive (2026-08-17): added `topk` as the **Core ATen layer** (ADR
  [0068](decisions/0068-decode-exit-multi-output.md) decision 3 — the exit of top-k sampling.
  Measured as core by `torch.Tag.core`, and not in the core decompositions either). **The first occupant of
  node-level multiple outputs**: IR with two `outs` (values f32 + indices i32) exists for the first time
  — the schema has allowed `outs` of length 1 or more from the start, so **the specification is unrevised** (ADR 0068
  decision 1. Allowing zero is a separate change). **attrs `k` is required, arity 1**, with the accepted domain
  `1 ≤ k ≤ last dimension`. No effect on existing IR.
- autoregressive (2026-08-17): added the **optional top-level section `states{}`** (ADR
  [0066](decisions/0066-generation-context-state-slots.md) decision 2 — named slots that hold the mutable state of
  one generation). **A graph without the section is unaffected** (no effect on existing IR; the exporter
  does not emit it yet). **There is no field yet for a node to reference a slot (the `states` field / `state_append`)** —
  this revision brings in only the declaration and its checks (the "State slots" section below).
- autoregressive (2026-08-18): added the **optional `states` field of a node**, the effect op
  `state_append` and the optional attrs `window` (ADR
  [0067](decisions/0067-autoregressive-attention-vocabulary.md) decisions 4 / 5 / 5b). At the same time three points
  loosen / tighten: ① **length 0 is allowed** for `outs` (an effect op that defines no value — allowed only for an op whose contract
  declares it, and the point of enforcement is the contract table, not the parser) ② symbols get **two binding points**
  (input shapes ∪ states shapes — ADR 0066 addendum item 7. A states-only symbol appearing in a value shape
  fails loudly) ③ **reference completeness** (a declared slot MUST be referenced by one or more nodes).
  For a graph without the `states` field (= every existing model) the accepted set does not move by a single byte.
- MTP (2026-09-08): added **three borrowing declarations** (ADR
  [0096](decisions/0096-speculative-decoding.md) stage 2 — the form in which the drafter reads the target's KV and weights):
  ① `states[].external` (optional, `true` only — the backing data lives in the lending context and is not allocated by itself)
  ② the optional attrs **`readonly`** of `attention` (`true` only — the states form narrows **ins to q alone**
  and becomes a past-only form that does not take the current step's k/v) ③ `initializers[].shared` (a borrowing declaration written
  in place of `tensor` — its bytes are not carried in the distribution). For all three, **a graph without the field is unaffected**, and
  there is no effect on existing IR.
- **v2** (2026-09-22, ADR [0108](decisions/0108-container-format.md) decisions 11 / 17):
  **an incompatible revision**. ① `initializers[].storage` and `initializers[].tensor` are removed from the IR, and the
  storage declaration moves to the container's binding table (`initializers[name]` is `{}` or `{ "shared": true }`)
  ② the initializer name becomes **the key of the backing data** (the exporter assigns FQNs / `const.<hash>`)
  ③ scales are unified to the rank-2 group form and `rowAxis` is put in the declaration (on the binding-table side —
  [container-v1](container-v1.md) §6.1) ④ the rules of **canonical serialization** are defined (the section below — the condition for
  judging the identity of a `krg` by content hash). `version` is `2`. No dual reading with v1 (migrating old
  assets is the CLI of container-v1 §12).
- Video generation (2026-10-02): added `conv3d` as the **extended composite layer** (ADR
  [0118](decisions/0118-wan21-video-generation.md) decision 1 — outside Core ATen, requested by Wan2.1's
  causal 3D VAE). **unbatched** (x is the rank-4 `[Cin,T,H,W]`), and a new value shape enters attrs: **three
  components `[T, H, W]`** (`stride` / `padding` / `dilation`). **Rank 5** appears for the first time in a weight
  initializer (the rank limit of 4 for the strided family applies to values, not to initializers).
  No effect on existing IR.
- Model expansion (2026-10-27): added `group_norm` as the **Core ATen layer** (`aten.native_group_norm` — a
  ledger NOTE in [op-vocabulary.md](op-vocabulary.md); the design is ADR [0138](decisions/0138-moss-image-dit-vae.md)
  decision 7, requested by MOSS-Image's untiled FLUX.2 decoder). **Arity fixed at 3**, attrs `num_groups` / `eps`;
  the exporter preserves `aten.group_norm`, so the default preserved list has 13 ops. No effect on existing IR
  (a GroupNorm failed before as an unsupported op).

## Container

- The graph JSON is carried as `graphs[<graph name>]` in the **graph descriptor** of the container (`krm` / `krg`)
  ([container-v1](container-v1.md) §2.1). One container can hold multiple graphs. The backing data (weights and constants) lives in
  blocks of the same container, and the binding table (§5) holds the correspondence with the graph.
- The JSON must not contain NaN / Infinity literals (the exporter writes with the equivalent of `allow_nan=False`;
  the parser fails loudly when it detects one).
- Duplicate keys within the same object are **accepted, last one wins** (the same as [container-v1](container-v1.md) §0 —
  not writing duplicates is the writer's responsibility).

## Graph JSON

```jsonc
{
  "format": "karume-ir",
  "version": 2,
  "requires": { "ops": ["matmul"] },
  "symbols": ["T"],
  "inputs": [{ "name": "x", "dtype": "f32", "shape": ["T", 128] }],
  "outputs": ["h"],
  "initializers": {
    "enc.w": {}
  },
  "values": {
    "enc.w": { "dtype": "f32", "shape": [128, 64] },
    "h": { "dtype": "f32", "shape": ["T", 64] }
  },
  "nodes": [{ "op": "matmul", "ins": ["x", "enc.w"], "outs": ["h"], "attrs": {} }]
}
```

- `format` is the fixed string `"karume-ir"`, and `version` is the integer `2`. A mismatch fails loudly.
- **An unknown top-level key fails loudly** (there is no forward-compatibility channel;
  if one is needed, it is introduced by revising this document). The 9 keys above are **all required**, and the only optional section is
  **`states`** (the "State slots" section below).
- `requires.ops` must equal the set of op names actually used in nodes (the parser
  checks it). The runtime matches it against its own support table and fails loudly **listing** the unsupported ops.

## Canonical serialization

The graph JSON must be **unique as a byte sequence** (re-exporting the same graph yields a byte-identical graph
descriptor — ADR 0108 decision 4, container-v1 §13.5; the condition for judging the identity of a `krg` by content
hash). Writers (the Python exporter, the migration CLI, the TS test helpers) serialize by the following rules, and readers
check in tests that `parse → canonicalize → serialize` matches the input byte sequence.

- **UTF-8, no whitespace** (no space after `,` and `:`, no line breaks).
- **Object key order**: objects of a fixed schema (top level, elements of `inputs[]`,
  values of `values{}`, values of `states{}`, nodes) use **the order in which keys appear in this document's examples** (top level is `format` /
  `version` / `requires` / `symbols` / `inputs` / `outputs` / `initializers` / `values` /
  `states` / `nodes`; nodes are `op` / `ins` / `outs` / `attrs` / `states`; values of `states{}` are
  `dtype` / `shape` / `external`). **Maps keyed by name** (`initializers` / `values` /
  `states` / a node's `attrs` / a node's `states`) use **code point order of the keys** (not the order of UTF-16
  units — the order changes for non-BMP characters. Names are all ASCII in measurement, so the two agree).
- **Arrays stay in declaration order** (`inputs` / `outputs` / `nodes` / `ins` / `outs` / `shape` carry meaning in
  their order). Sets whose order carries no meaning (`requires.ops` / `symbols`) are sorted in **code point order**.
- **The spelling of numbers is fixed to ECMAScript's `Number::toString`** (ECMA-262 §6.1.6.1.20). Integer values
  carry neither a decimal point nor an exponent (`1`, `10000` — not `10000.0`), non-integers use the shortest round-trip digits,
  and exponent notation is used only when `|x| < 1e-6` or `|x| ≥ 1e21` (`1e-7`, `1e+21` — no zero padding of the
  exponent; `0.000001` is not written with an exponent). `-0` is written as `0`. NaN / Infinity cannot be written
  (rejected at the value level — the section above). On the JS side `JSON.stringify` is exactly this spelling, so **the Python side
  implements it** (`repr(1e-6)` = `1e-06` and `repr(10000.0)` = `10000.0` are not canonical).
- **String escaping** is JSON's minimal form (escape only `"` / `\` / the control characters U+0000 to U+001F;
  `\b` `\f` `\n` `\r` `\t` use the short forms, the rest `\u00XX` with lowercase hex. Non-ASCII is raw UTF-8).
  It is the common subset of JS `JSON.stringify` and Python `json.dumps(ensure_ascii=False)`, and the two agree.
- **Optional sections** (`states` / a node's `states` / `states[].external`) are **not written when empty or false**
  (the absence of the field is itself the declaration — always emitting them would give the same graph two spellings).

Whether a reader **rejects non-canonical JSON** containing whitespace or `\u` escapes belongs to the reader's contract
(TS `parseIrDeclaration` accepts it as JSON; canonical form is protected by the writer-side rules and round-trip tests).

## Values and types

- **Semantic dtypes** (`inputs[].dtype` / `values[].dtype`): `"f32" | "i32" | "bool"`.
  Computation is always done in these types (f16/bf16/i8 do not exist as semantic dtypes).
  Run-time storage is i32 = 32 bit, bool = **u32 0 / 1** (because storage buffers have no 1-bit type
  — ADR [0009](decisions/0009-dtype-i32-bool.md)). Elements are 4 bytes for all three types.
- **The semantic dtypes an op can execute differ per op** (not one uniform set). The canonical accepted set is the contract table
  (`packages/runtime/src/ops.ts` / `karume/ops.py`), and the runtime fails loudly listing what is unsupported.
  Graph inputs can be transferred for all three types. torch's default integer i64 is **normalized to i32
  at the exporter boundary** (out-of-range values fail loudly) — there is no i64 in the IR.
- **Storage (codec) is not written in the IR**. The storage form of an initializer's backing data is held by the container's **binding table**
  ([container-v1](container-v1.md) §5 — `binding[<graph name>][<initializer name>].encoding`),
  and the IR holds only the semantic dtype and the declared shape ("one architecture, one quantization scheme, one graph" —
  ADR [0108](decisions/0108-container-format.md) decision 17). The codec vocabulary is the ledger (container-v1
  §6.3): `f32` / `f16` / `bf16` / `i32` and the five quantized ones `int8-sym` / `int4-sym-g` / `int4-sym-g-f16` /
  `int2-off` / `ternary`. Everything but `i32` is **an encoding of semantic f32**; only `i32` is **raw int32** (the target that
  symbol-dependent constants are baked into — ADR [0010](decisions/0010-symbolic-constant-folding.md). The explicit exception to
  "the storage vocabulary is an encoding of f32"). **The runtime can execute every codec in the ledger** but `int4-sym-g-f16`, which only `expert_linear` reads (ADR [0145](decisions/0145-gemma4-26b-moe-resident.md) decision 3) (until 2026-10-17 bf16 was
  "valid as a declaration, fails loudly on execution", but with ADR [0131](decisions/0131-bf16-weight-execution.md)
  it joined the same path as f16). The ledger holds only the storage format; the canonical source for whether it can be executed
  is the runtime's `RUNTIME_SUPPORT.storage` (`packages/runtime/src/ops/contracts.ts` — the set of decode paths)
  (container-v1 §6.2, ADR 0108 addendum item 12).
- **Merge layer** (container-v1 §13.3): the graph and the binding table are **merged before the weights are fetched**, and
  the storage rules are applied to the merged representation (it lives in one place, the runtime's `format/` — the Python side mirrors the
  same rules; they are not implemented twice):
  - A quantized codec **requires** `encoding.scale` (no default of 1.0 is filled in — this blocks a forgotten entry from silently
    running as "weights at 1/127 of their pre-quantization value", ADR 0019 / 0069)
  - The `groupSize` of `int4-sym-g` MUST be **a power of 2 and at least 16**, and MUST divide the row length (`numel / shape[rowAxis]`)
    (the constraint of not creating fractional groups guarantees byte alignment at row and group boundaries —
    ADR 0069 decision 2). `int8-sym` / `int2-off` / `ternary` are per-channel (`groupSize` = row length)
  - The scale shape is unified to the **rank-2 group form `[shape[rowAxis], row length / groupSize]`**
    (container-v1 §6.1 — the keepdim broadcast form used by v1's i8 is retired). `rowAxis` appears in the
    declaration (1 only for the weight `[Cin,Cout,K]` of `conv_transpose1d`, 0 otherwise)
  - Semantic and storage pairs are **only the following two** (any cross pairing fails loudly — this blocks the silent wrong value of an i32 declaration
    being read as an f16 bit pattern): semantic `f32` × `f32` / `f16` / `bf16` / the four quantized ones,
    semantic `i32` × `i32`
- **Eligibility and the two paths** (ADR [0018](decisions/0018-f16-weight-execution.md) /
  [0019](decisions/0019-i8-weight-execution.md) / [0069](decisions/0069-packed-w4-storage.md)):
  the semantics are f32 throughout ("quantized storage only, computation in f32" — ADR 0006), and an initializer with compressed storage
  splits into two paths by **eligibility**.
  - **Eligible** (the initializer is consumed **only in the weight slot** of `linear` / `conv1d` / `conv2d` / `conv3d` /
    `conv_transpose1d` / `embedding`): it stays GPU-resident as the compressed payload, and dequantization happens in the kernel.
    The eligible ops per codec are decided not by the ledger but by the runtime's eligibility check (canonical: `eligibleCompressedInitializers` /
    `i4EligibleInitializers` / `i2EligibleInitializers` in `packages/runtime/src/runtime/plan.ts` and
    `planWeightResidency` in `runtime/weight-residency.ts` — container-v1 §6.2, ADR 0108 addendum item 12).
    `int4-sym-g` is linear / embedding / conv1d (`groups == 1`); `int2-off` / `ternary` are linear / embedding.
  - **Ineligible** (bias / norm weights / other ops / consumption mixing the weight slot with other slots /
    zero consumers / an initializer listed in `graph.outputs`): **expanded to f32 on the CPU at load time**.
    Correctness holds, but the VRAM saving is zero; only the delivery size shrinks. Bias not being eligible is
    ADR 0006's "bias is always f32" rule itself (bias is not included in the low-precision eligibility check).
  - The breakdown is available from `Session.diagnostics().storage` (GPU-resident compressed bytes / CPU-expanded bytes)
    — a permanent diagnostic so that "compression requested but 0 MB eligible" does not go unnoticed (ADR 0006).
- **`f16`**: when GPU-resident, two elements per word are unpacked with `unpack2x16float` (the value at index `i` is
  `unpack2x16float(w[i / 2])[i % 2]`). VRAM is ≈ 1/2 of f32. The trailing padding for an odd element count is
  **baked into the block by the writer** (container-v1 §4.1 — there is no alignment branch on the reader side).
- **`int8-sym`** (ADR 0019): **per-channel symmetric int8** (no zero-point, `q ∈ [−127, 127]`).
  When GPU-resident it is bound as `array<u32>`, packed as **four elements per word (u32) in little-endian order**,
  and unpacked with `unpack4xI8` (word = `w[i / 4]`, lane = `i % 4`, derived from the **flat index**).
  **The scale is applied "per element"** (dequantization on read — `out = Σ x·(q·s) + bias`). The form that applies it outside
  the reduction (`(Σ x·q)·s`) has fewer multiplications but loses bit equality with CPU expansion, so it is not used. Thanks to this form
  **the eligible path (GPU) and the ineligible path (CPU expansion) produce bit-identical values** (the f32
  rounding of `q·s` happens once on both sides). VRAM is ≈ 1/4 of f32 (+ under 1% of scale overhead). The diagnostic's
  `residentCompressedBytes` **also adds the bytes of the scale buffer**.
- **`int4-sym-g`** (ADR 0069 decisions 2 / 3): symmetric quantization with row-wise groups, held as packed 4 bit
  (`q ∈ [−7, 7]`, `u = q + 8`, element `2i` = the low nibble within a byte). The declared shape **stays the logical
  shape**, and the payload is `numel / 2` bytes (a declaration with an odd element count fails loudly because the total bits
  do not land on a byte boundary). The decode kernel binds it as `array<u32>`.
- **`int4-sym-g-f16`** (ADR [0145](decisions/0145-gemma4-26b-moe-resident.md) decision 3): the bytes of `int4-sym-g` with
  one f16 step per group (two per u32 word, step `2i` in the low half), the step table padded to a whole word. Read by
  `expert_linear` only; every other consumer refuses it.
- **`int2-off`** (ADR [0097](decisions/0097-gemma4-qat-integration.md#addendum-1-2026-09-11--int2-storage-and-execution-contract)):
  the logical shape is a rank-2 `[N,K]` of positive integers, with K a multiple of 16. Four elements per byte are packed from the low 2 bits
  as `u = q + 2`, holding the integers `[−2, 1]`. Restoration is the per-element `fround(q·scale)`. linear computes in f32
  only, and a8 / f16 are explicitly rejected. **`ternary`** is an alias codec that claims the value range `{−1, 0, +1}`;
  its packing, restoration and kernels are identical to `int2-off` (container-v1 §6.3).
- **The initializer name is the key of the backing data**. v1's `initializers[].tensor` (the safetensors tensor
  key) is gone, and the binding table points at the backing data by `(graph name, initializer name)` (container-v1 §5).
  The value of `initializers[name]` takes only two forms, **`{}`** (has backing data) or **`{ "shared": true }`** (below);
  any other key fails loudly as an unknown key. The exporter names them by **the parameter FQN**
  (`model.layers.0.mlp.gate_proj.weight`) and **`const.<first 16 hex of sha256>`** (constants — deduplicated by
  content). Importing upstream checkpoints (ADR 0108 decision 19) and resolving LoRA targets use
  this name as the key. The declared shape / semantic dtype and the backing data must agree when the bindings are reconciled
  (checked at load time; both missing and surplus entries are rejected with a full listing).
- **Shared initializers (`shared`)** (ADR [0096](decisions/0096-speculative-decoding.md) stage 2 §1.3):
  writing `{ "shared": true }` makes a declaration whose bytes are not in the container (the backing data is the weight the lending Session
  has already put on the GPU).

  ```jsonc
  "initializers": {
    "model.lm_head.weight": { "shared": true }
  }
  ```

  - **The name MUST be the same as the initializer name in the lending graph**. The borrower receives the backing data from
    `SessionOptions.sharedWeights` under this name, and the models side looks up the lender's
    `exportWeight` by the same name (v1's `shared.tensor` is not needed — because the name is the key).
  - The dtype / shape declaration in `values{}` is **required** as before. Storage (codec / scale / group) is
    **not written** — the lender's resident weight is canonical, and copying it would grow the same fact in two places.
  - It is **outside** the set reconciled against the binding table (it is not in the binding table — container-v1 §5). The provenance of the
    backing data is examined by the load-side checks (same device, declared shape, semantic dtype, consuming slot, channel axis — the lender's
    codec must be executable by the borrower's consuming op). The semantic dtype MUST match the lender's declaration (f32 / i32 have the
    same resident format, and the slot cannot tell them apart).
- **Declaration completeness**: every value (inputs, initializers, every node output) is declared in exactly one place
  — inputs in `inputs[]`, and **initializers, intermediate values and outputs in `values{}`**. At run time every node is
  checked against the declared shape/dtype.
- Initializer declarations use **numeric dimensions only** (no symbolic dimensions). The semantic dtype is **f32 / i32**
  (pairing with storage follows the merge-layer rules above). bool initializers are not in the vocabulary (none in measurement, and 1-byte storage
  does not fit the transfer that assumes 4 bytes).
- Non-finite numbers are rejected not only as literals but **at the value level** (`1e999` is syntactically valid JSON but
  rounds to Infinity, so it is rejected by a reviver at parse time). The rules on the descriptor side (binding table, block index) are held by
  container-v1 §0 — there too, unknown keys fail loudly.

## Shapes and the dimension language

- A shape element is either "a non-negative integer" or "a dimension expression string".
- The canonical grammar of dimension expressions: **a linear expression `coeff·sym + offset`, one symbol per dimension**.
  The string forms are `"T"` / `"8T"` / `"T+8"` / `"8T+2"` (coeff is prefixed when 2 or more,
  offset is suffixed with `+` when 1 or more. `"1T"` and `"+0"` are rejected as non-canonical. Negative coefficients,
  negative offsets and multiple symbols are rejected).
- A symbol name is `[A-Za-z_][A-Za-z0-9_]*`. It must be listed in `symbols` and **appear in a dimension position
  of at least one input shape or states shape** (the binding is taken directly from the dimension position of the shape;
  it is not back-computed from element counts). The rule that splits binding points in two and the constraint that a states-only symbol
  cannot be written in `values{}` are canonical in the "State slots (`states{}`)" section below — only bindings from inputs
  take part in resolving ordinary values. The appearance **may be a derived form** — from the actual size of `"2T"` or `"T+8"`
  the solution is uniquely determined by `(size − offset) / coeff`, and a size that does not divide fails loudly
  ([ADR 0057](decisions/0057-derived-dim-binding.md). The vowel-detection CRNN can declare its length axis only as `2T`
  because of the stride 2 of its first conv).
- The grammar has a single canonical source: the conformance case table `packages/runtime/tests/fixtures/dim-grammar.json` (three sections: valid / invalid /
  binding evaluation). The TS implementation and the Python implementation (from M1) are checked against the same table.

## State slots (`states{}`)

The declaration of **named slots** that hold the mutable state of one generation (such as the autoregressive KV)
(ADR [0066](decisions/0066-generation-context-state-slots.md) decision 2). **An optional section**; for
a graph without it (= every existing model) the accepted set does not change at all.

```jsonc
{
  // … (the other top-level sections are the same as above)
  "symbols": ["T"],
  "states": {
    "layer0.k": { "dtype": "f32", "shape": [1, 2, 512, 128] },
    "layer0.v": { "dtype": "f32", "shape": [1, 2, 512, 128] }
  }
}
```

- The form is `name → { dtype, shape }`, two keys + **an optional `external`** (unknown keys fail loudly).
  An empty object and omitting the section mean the same (= zero slots). A slot name is **not the empty string** (if a name that the
  referencing fields do not accept could be declared, the slot would be unreferenceable in principle).
- **`external`** (ADR [0096](decisions/0096-speculative-decoding.md) stage 2 §1.1, **`true` only**,
  absence of the field = allocated by itself): a slot whose backing data is held by **the lending context**. `false` cannot be written
  (no two spellings of the same meaning as the absence of the field). There are four rules, and breaking any of them does not raise an exception but
  produces a different value — "reading an uninitialized past" or "overwriting KV the lender has committed":
  - **zero** `state_append` (only the lender can write to a borrowed slot)
  - the only readers are **attention with `attrs.readonly`** (the form that adds the current step's k/v assumes it reads rows it
    wrote itself, so in a slot with no writer it always reads uninitialized rows)
  - the slots that a `readonly` reader references are **external only** (the counterpart of the above)
  - if even one slot is external, **every slot is external** (a borrowing context has no slots of its own)
- `shape` is **a concrete shape of fixed rank including capacity**: rank is **1..4** ("fixed rank
  (rank ≤ 4)" of ADR 0066 decision 2), and numeric dimensions are **positive integers** (unlike the non-negative ones of `values` — a slot of capacity 0 cannot
  hold backing data). Dimension expressions use the same dimension language as values, and symbols already declared in `symbols` can be used. **Symbols
  have two binding points** (ADR 0066 addendum item 7): the dimension positions of input shapes (as before — only these take part
  in resolving value shapes), and **symbols that appear in states shapes are bound by
  `createGenerationContext(spec.bindings)`** (values decided when the context is created, such as the KV capacity).
  The bindability check is "appears somewhere in input shapes **∪** states shapes", and **a states-only symbol
  (a symbol that appears only in states) appearing in a declared shape in `values{}` fails loudly** — only bindings from inputs take
  part in resolving ordinary values, so if one appeared it would certainly be unbindable at run time (it is rejected at
  declaration time). The same division holds at run time: states-only symbols are **not required** in the bindings of `run` / `enqueue`,
  **and passing one fails loudly** (the binding point is the single place `createGenerationContext` —
  no double bookkeeping against the capacity the context holds).
- `dtype` is **`f32` only**. `f16` is **only reserved** (ADR 0066 addendum item 5 — lowering the precision of state
  storage changes the numeric contract, so it needs an opt-in in the style of ADR [0058](decisions/0058-numerics-opt-in-contract.md)),
  and when declared it fails loudly as "**unsupported**" (a different diagnostic from "outside the vocabulary").
  Anything else (`i32` / `bool` / unknown names) is outside the vocabulary.
- **A slot is not a value**: it has no declaration in `inputs` / `values` and is not referenced from `ins` / `outs` either
  (not making state an ordinary graph input / output is the MUST of ADR 0066 decision 1). Allocation, lifetime and
  logical length of the backing data are held by the GenerationContext.
- The namespace is separate from values, but **a slot with the same name as a value name (inputs / initializers / node outputs / declarations in `values`)
  is rejected** — the same name in separate namespaces only makes it impossible to detect "a value name written in a slot-name field / or the reverse"
  and adds no expressiveness (the same style as rule ① by which the binding table does not let a scale's block collide with
  another initializer's backing data (container-v1 §5)).
- **There is no "layers × uniform KV" assumption**: a slot is a general-purpose container that knows nothing about "KV"; sliding layers and full layers are
  separate slots with different capacities, and KV-sharing layers are expressed by **referencing the same slot name** (no fields for layer count or
  layout are created).
- **Reference completeness**: a declared slot MUST be **referenced from the `states` field of at least one node**
  (the same hole as an orphaned declaration in `values` — a slot no one reads is only allocated by the GenerationContext and
  silently eats capacity). Slots with only readers or only writers are both valid (KV-sharing layers
  have no `state_append` — ADR 0067 decision 5).
- References from nodes are in the "State-referencing nodes" section below (the states form of `attention` / `state_append`).
  The physical form of a slot (`[B, Hkv, C, D]`) and `window ≤ C` are held there.

## Nodes

- SSA, single assignment: each value name is defined in **exactly one place** among inputs / initializers / the `outs` of some
  node. `ins` references only defined names. `outputs` is a subset of defined
  names. The number of `outs` is **decided by the op contract** (**multiple outputs** are the two of `topk` (values f32 +
  indices i32, ADR [0068](decisions/0068-decode-exit-multi-output.md) decision 3), **zero** is
  `state_append` (an effect op that defines no value, ADR
  [0067](decisions/0067-autoregressive-attention-vocabulary.md) decision 5), and all others have a single output).
  **Zero is allowed only for an op whose contract declares an effect** — the parser gives the count no meaning, and the point of enforcement is
  matching the output count in the contract table (writing `outs: []` for another op fails on the count mismatch).
- `nodes` are stored in **topological order** (the parser checks it; a forward reference fails loudly).
- `attrs` are checked against the per-op contract table. Unknown attrs and values outside the contract fail loudly
  (no approximate execution). **The only optional attrs are `window` and `readonly`** (both
  exclusive to the states form — "State-referencing nodes" below). They are fields whose absence is itself a different declaration (no `window` =
  the whole context / no `readonly` = the form that also reads the current step's k/v); all other attrs are as before:
  "every declared key is required, no default filled in".
- **The `states` field** (optional, "State-referencing nodes" below): a field separate from `ins` / `outs` that references state
  slots by name. The keys are fixed by the op contract, and the values are slot names declared in `states{}`
  (a reference to an undeclared slot fails loudly).

## State-referencing nodes (`attention` states form / `state_append`)

The contract of nodes that read and write state slots (ADR
[0067](decisions/0067-autoregressive-attention-vocabulary.md) decisions 4 / 5 / 5b).

```jsonc
{
  "nodes": [
    {
      "op": "attention",
      "ins": ["q", "k_cur", "v_cur"],
      "outs": ["o"],
      "attrs": { "scale": 0.088, "window": 512 },
      "states": { "k": "layer0.k", "v": "layer0.v" }
    },
    {
      "op": "state_append",
      "ins": ["k_cur"],
      "outs": [],
      "attrs": { "window": 512 },
      "states": { "slot": "layer0.k" }
    },
    {
      "op": "state_append",
      "ins": ["v_cur"],
      "outs": [],
      "attrs": { "window": 512 },
      "states": { "slot": "layer0.v" }
    }
  ]
}
```

- **The presence of the field determines the form**: `attention` without a `states` field is the existing contract unchanged. When the field
  is `{ k, v }` (**exactly these two keys**, and **the same slot name cannot be written** for k and v) it becomes
  the states form; the k / v in `ins` are **the current step's chunk** (`[B,Hkv,M,D]`, M shared with q),
  and the past is held by the slot. The states form **takes no mask input** (causal is fixed — expressed by a predicate,
  with no mask tensor materialized).
- `state_append` (f32, **arity 1**, **zero outputs**, the `states` field is exactly `{ slot }`,
  no required attrs) — a single-purpose effect op that writes the current step's k / v into the slot. The input is
  `[B,Hkv,M,D]`. A KV-sharing layer is expressed simply by "this node not being there" (the same expressiveness as ORT's kv_empty
  obtained through the absence of the op).
- **The physical form of a slot is fixed at `[B, Hkv, C, D]`** (C = capacity). **B / Hkv / D must match**
  the `ins` of the referencing node, and the k / v slots of `attention` must **have the same shape as each other**.
- **readonly form** (ADR [0096](decisions/0096-speculative-decoding.md) stage 2 §1.2): when a states-form
  attention declares the optional attrs **`readonly`** (`true` only), `ins` becomes **exactly q
  alone** (a past-only form that does not take the current step's k / v).

  ```jsonc
  {
    "op": "attention",
    "ins": ["q"],
    "outs": ["o"],
    "attrs": { "scale": 1.0, "window": 512, "readonly": true },
    "states": { "k": "l13.k", "v": "l13.v" }
  }
  ```

  - q is `[B, H, M, D]` with **M = 1 MUST** (only the single row at logical position P−1 is read — with M > 1 the
    second and later rows would read the same column range, a silent wrong value). Against the slot `[B, Hkv, C, D]`,
    **B / D match**, `H % Hkv == 0` and `H ≥ Hkv`, and for sliding `window ≤ C`. The output is
    `[B, H, 1, D]` (the same shape as q).
  - Semantics: attention over the columns `[column_base, P)` (`column_base` = `P − min(P, W)` for sliding /
    `0` for full). There are **no** `ins` for the current step, so the number of live columns is `min(P, W)` / `P`.
    The numeric contract of softmax (half scale, −inf identity, ② statistics) is the same as the states form, and **P = 0
    (zero columns) is an empty row → output 0**.
  - `readonly` can be declared **only on a node with a states field**, and it references **only external slots**
    (the "State slots" section above). It cannot be written on `state_append`.
- The optional attrs **`window`** (a positive integer; absence of the field = the whole context) = the width of the sliding window.
  **`window ≤ C`** MUST. **Presence and value MUST match across every node that touches the same slot**
  (the logical col → physical row mapping uses the same formula for reads and writes — a different formula on the read side alone would be a silent misread).
  `window` **can be declared only on a node with a states field** (written on the existing form it would become "an attr that is accepted but
  read by no one").
- **The order is the `nodes` array order** (decision 5b): state references do not create data edges between tensors, so
  topological order does not decide it. ① at most one `state_append` per slot ② if there is an append, it is
  **the last node that touches that slot** (no reader can be placed after the append). The exporter emits them
  in the order "all readers → append".
- The logical lengths (`pastLength` / `queryLength`) appear in neither attrs nor shapes — they are run-time scalars that change every step,
  carried by a mutable uniform owned by the GenerationContext (ADR 0066 decision 3, addendum item 4).
  `pastLength + queryLength ≤ C` for a full slot is a run-time check (on the context side).

## Op set (the implementation's contract table is canonical; this is only a list)

> There are two implementations (TS `packages/runtime/src/ops.ts` and the exporter's `karume/ops.py` +
> `karume/shapes.py`). The canonical source for both holding the same contract is the conformance case table
> `packages/runtime/tests/fixtures/op-contracts.json`: the tests on both sides match the full set of op names / arity / slot dtypes /
> attrs key sets and value ranges / output shape rules (including rank limits) against the same table
> (the same relation as the dimension grammar and `dim-grammar.json`).

- unary elementwise: `neg, abs, exp, log, log1p, sqrt, sin, tanh, sigmoid, relu, gelu, gelu_tanh`
  (f32) / `bitwise_not` (bool)
  - `gelu_tanh` is torch's `gelu(approximate="tanh")`. Its numbers differ from the erf form `gelu`, so it is carried as
    **a separate op** (a contract with empty attrs has no field for the approximation kind — ADR
    [0043](decisions/0043-op-addition-layers.md) first application; current door = an attr variant in the
    Core ATen layer (ADR 0059))
  - `sin` is **the only trigonometric function** in the vocabulary (Core ATen layer (formerly layer 1)). Constant tables such as RoPE are
    folded into initializers by the exporter, so the vocabulary needs only the form that takes run-time values
    (first appeared in DACVAE's Snake activation). `cos` is not added until it shows up in measurement
  - `clamp` (f32, attrs `min` / `max`) — both ends are **required** finite f32 with `min <= max`
    (an inversion is the direction in which WGSL's `clamp` is undefined)
  - `clamp_min` (f32, attrs `min`) — **one-sided clamp** (ADR
    [0017](decisions/0017-rms-norm-conv2d-clamp-min.md)). It has no field for the upper bound (structurally ruling out the approximation
    "fill the missing side with ± the largest finite value"). Making the attrs of the existing `clamp` optional to
    cover it is not adopted because it would break "no defaults are filled in for declared attrs" (ADR 0012)
  - `leaky_relu` (f32, attrs `negative_slope`) — **no default is filled in** (ADR
    [0015](decisions/0015-conv-family-extension.md). Measurement shows a mix of 0.1 and 0.01, and relying on a default
    would make one of them silently wrong)
  - `ge_scalar` / `le_scalar` / `gt_scalar` (f32 → **bool**, attrs `value`) — comparison with a finite f32
- binary elementwise: `add, div` (f32) / `mul, sub` (f32, i32) / `ge` (f32 → **bool**) /
  `bitwise_and` (bool) — all with torch-style right-aligned broadcast
- `where` (**ternary**, no attrs) — `out = cond ? a : b`. dtype per slot
  (cond: bool / a, b: f32 → output f32), with **right-aligned broadcast across all three** (the condition broadcasts by the same
  rule as the values). The output has the same type as **the value side**, not the condition
- `cumsum` (f32, attrs `dim`) — the prefix sum over the last dimension `out[…, j] = Σ_{i ≤ j} x[…, i]`.
  **Only the last dimension** is accepted (`dim` in non-negative notation — the same narrowing as softmax). An axis of length 0 passes through
- `cast` (between f32 / i32 / bool; attrs `to` is the target)
- `matmul` (rank-2 × rank-2, f32)
- `bmm` (batched matmul of **rank-3 × rank-3** `[B,M,K] × [B,K,N] → [B,M,N]`, f32, no attrs)
  — the batch dimensions match exactly (neither broadcast nor stride 0). rank-2 is `matmul`'s job, and the two are not
  combined
- `gather` (**last dimension fixed**, no attrs) — `out[..., j] = src[..., index[..., j]]`.
  src and index have the same rank, the leading dimensions match exactly, and only the last dimension is free. **The only op whose
  dtype differs per input slot** (src: f32 / index: i32 → output f32). The index range `0 <= index < D`
  depends on run-time data, so the shape contract does not check it (out-of-range handling is the implementation contract —
  the GPU contaminates only the affected elements with NaN / the CPU reference throws)
- reduce (**one axis**, attrs `dim`, no keepdim): `sum` (f32 → f32 / **bool → i32**, the true count) /
  `amax, amin` (f32) — `dim` is a **required** non-negative axis number (no default is filled in. If omission were allowed,
  IR meant to reduce the channel axis would silently run as a different computation that collapses the last dimension).
  The last dimension runs on the row kernel and others on the axis variant, but **the reduction order matches exactly between the two variants**
  (design and symbolic verification in research/2026-08-04-vae-axis-reduce-recon.md §5.2)
- `argmax` (f32 → **i32**, **no attrs** — ADR
  [0068](decisions/0068-decode-exit-multi-output.md) decision 2) — argmax over the last dimension; the output
  **preserves rank** (a fixed form with the last dimension collapsed to 1). A contract separate from the reduce family: the axis is fixed at the last dimension
  (no `dim` field), there is no `keepdim` field either (forms that lower the rank are not in the vocabulary), and a last dimension of length 0
  is rejected. The fixed behaviour follows torch on all three points: **ties break to the smallest index** / **NaN is the maximum**
  (the smallest index if several) / **an all −inf row gives index 0** (the identity of the row max MUST be −inf — with a finite
  sentinel the sentinel index leaks into the output). llama.cpp disagrees even within one repository, GPU side = largest index, CPU sampler =
  smallest index, and without writing this down greedy reproducibility splits on implementation differences
- `topk` (f32 → **two outputs, values f32 + indices i32**, attrs `k` — ADR
  [0068](decisions/0068-decode-exit-multi-output.md) decision 3) — top-k over the last dimension; both outputs
  are `[…, k]` (rank preserved). **The only multi-output op**. `k` is **a plan-time constant** (attrs required =
  static-k. torch's schema is SymInt, but a k decided at run time does not fit the static-shape premise), and
  the accepted domain is **`1 ≤ k ≤ last dimension`** — k=0 (which torch accepts), k > last dimension and symbolic k
  all fail loudly. The axis is fixed at the last dimension (no `dim` field), and there are no `largest` / `sorted` fields either
  (only one form, **the largest side sorted in descending order**, is in the vocabulary). The fixed behaviour is **ties = smallest index** /
  **NaN is the maximum** (the smallest index if several) / an all −inf row also yields k entries from the smallest index. **The value column is bit-identical
  to torch**, while **the index column is the side where karume specified what torch leaves unspecified** (torch does not
  guarantee the order of equal elements, and contradicts itself with `topk([5,5,5,5],1)` = 2 against `argmax` = 0 —
  measured 2026-08-17. In karume, k=1 matches argmax). The implementation is "lane-local top-k →
  tournament merge" without going through an argsort of the whole vocabulary, and **the implementation limit on k** is decided statically by the
  device limit of workgroup storage (with the WebGPU default of 16384 bytes, **k ≤ 63**. Exceeding it does not degrade but raises
  `CodegenError` carrying the limit)
- Layout (ADR [0011](decisions/0011-layout-strategy.md)) — all unary:
  - `reshape` (f32 / i32 / bool) — **the declared shape of the output is the target shape**. The contract is only a matching element count,
    and the element order does not change. No attrs (holding the target shape in attrs would duplicate the declaration)
  - `permute` (f32, attrs `dims`) — `dims[d]` = the input dimension number taken by output dimension d.
    **Only a bijection of non-negative integers** (negative axis notation is normalized at the exporter boundary)
  - `expand` (f32 / i32 / bool) — right-aligned broadcast to the declared shape of the output. **Only dimensions of length 1
    can be expanded** (replicating length n → m is not in the vocabulary). No attrs
- Second layout group (ADR [0014](decisions/0014-layout-ops-full-write.md)) — only forms in which every kernel
  **writes every byte of the output** (full-write) enter the vocabulary:
  - `slice` (f32, attrs `dim` / `start` / `end`) — a slice over **a static axis and static range**
    `x[…, start:end, …]`. All three are non-negative integers (negative axis and negative index notation are normalized at the exporter boundary),
    `start ≤ end ≤ the input's axis length`. **The target axis is static at the declaration level** (slicing a symbolic axis is
    `sym_prefix_slice`'s job — no duplication). Execution reuses the strided read-copy family, and
    the only variable is one params word for the offset
  - `cat` (f32, attrs `dim`) — **the only variable-arity op** (2 or more inputs); dimensions other than the concatenation axis
    match across all inputs, and the output's axis length is the sum of the inputs' axis lengths. The concatenation axis may be symbolic when it is (a constant) or (a linear expression in **the same**
    symbol) and the sum fits the dimension language `coeff·sym+offset`
    (`S`+1519 → `S+1519`, `S`+`S` → `2S` — ADR
    [0046](decisions/0046-cat-symbolic-axis.md)). A mix of different symbols has no representation. Execution is the
    strided **write**-copy family, writing each input into a sub-region of the output (all inputs together cover the whole output)
  - `pad` (f32, attrs `left` / `right`) — **constant zero padding of the last dimension**. Widths are non-negative integers (cropping by
    negative widths is not in the vocabulary), and there is no field for the fill value (only 0 in measurement). One dedicated kernel
    copies the in-range part and **also writes 0 out of range** (it does not assume a zero-initialized buffer)
  - `flip` (f32, attrs `dim`) — index reversal along a static axis (shape is identity). The target axis is static at the declaration level
- **Symbolic prefix slice** (ADR [0010](decisions/0010-symbolic-constant-folding.md)):
  - `sym_prefix_slice` (f32 / i32, attrs `sym` / `slices`) — a unary op that cuts the **prefix** of the bound length out of
    **a constant that the exporter baked by actually evaluating** the symbolic-T-dependent subtree **at Tmax**.
    `slices` is `[{ dim, coeff, offset }, …]`, and the output length of axis `dim` is `coeff·sym+offset`, the same as the dimension
    language (`coeff ≥ 1` / `offset ≥ 0` / no duplicate `dim` / no empty array).
    Axes not in `slices` stay as in the input.
    - **The input is a static shape without symbols** (= the Tmax shape; in practice an initializer baked by the exporter).
      A symbolic dimension on the input side would break the "Tmax shape" premise and change the read stride on every run.
    - `sym` must be declared in `symbols` (without a binding the prefix length is not determined).
    - The output shape is **computed** from attrs and the bindings and checked against the declaration (not "the declaration is the
      target shape" as with reshape / expand). If `coeff·sym+offset` exceeds the input's dimension, it fails loudly.
    - Execution is the same strided materializing copy as the layout ops (start position 0, contiguous strides on the input side)
- Fused ops (ADR [0012](decisions/0012-attrs-and-fused-ops.md) / ADR
  [0015](decisions/0015-conv-family-extension.md) / ADR
  [0017](decisions/0017-rms-norm-conv2d-clamp-min.md)) — high-level ops the exporter stops decomposing and carries as
  a single node. The preserved ops of ADR [0007](decisions/0007-op-vocabulary.md) became **11** with
  `leaky_relu` in M1-P3 and `conv2d` / `rms_norm` in M1-P4 (ADR 0017), **12** with `conv3d` of the extended composite layer (ADR 0118),
  and **13** with `group_norm` of the Core ATen layer (ADR 0138 decision 7);
  all of them
  **have kernels** (`conv2d` showed up in measurement in Anima's VAE decoder). Only `rms_norm` has
  **two supply routes**: ① `aten.rms_norm` from diffusers' `nn.RMSNorm` becomes one node as is via the preserved
  list ② the hand-written decomposed form of Qwen3 / DiT cannot be folded by preservation, so
  the exporter's folding pass (`_fold_rms_norm`) composes it into one node (ADR 0016 / 0017).
  **The fused `attention` (perf-a) is not in the default preserved list (13 ops)**: applying SDPA preservation globally would
  also pick up forms with masks outside the contract and make them impossible to export, so it is **an opt-in per target**
  (ADR 0023 addendum):
  - `linear` (f32, no attrs, **arity fixed at 3**) — `x[…,in] × W[out,in] + b[out]`.
    The weight stays in the transposed layout `[out, in]`. A form without bias is not in the vocabulary
  - `layer_norm` (f32, attrs `normalized_shape` / `eps`, **arity fixed at 3**) — last dimension only
    (`normalized_shape` has length 1), affine always present. The variance is **the population variance (correction = 0)**,
    and `eps` is a finite positive number
  - `group_norm` (f32, attrs `num_groups` / `eps`, **arity fixed at 3** — ADR [0138](decisions/0138-moss-image-dit-vae.md)
    decision 7) — `x[N,C,…]` (rank ≥ 2; **C static**, the trailing axes may be symbolic, no axis of length 0) /
    `weight[C]` / `bias[C]` → the shape of x. The C channels form `num_groups` groups of `C/num_groups` consecutive
    channels; each group (its channels × every trailing position of one batch item) is normalised with **the population
    variance** (`y = (x − mean)·rsqrt(var + eps)`), then the per-channel affine `y·weight[c] + bias[c]`. `num_groups` is a
    positive integer that **divides C** (a shape-rule check), and `eps` is a finite positive number (the normalisation ops'
    rule). The affine is always present: for a module without affine the exporter composes ones / zeros (the bias-only form
    fails loudly). The statistics are two passes — the mean, then the sum of squared deviations — over a reduction tree
    fixed by the shape (the one-pass `E[x²] − E[x]²` cancels over groups of millions of elements)
  - `attention` (f32, attrs `scale`, **arity 3 or 4**) — `out = softmax_lastdim((q·scale) @
    (k·scale)ᵀ + mask) @ v` (ADR [0023](decisions/0023-fused-attention.md)). The inputs are
    **rank-4 head-first** (`q[B,H,M,D]` / `k[B,Hkv,N,D]` / `v[B,Hkv,N,D]`, contiguous) and the output is
    `[B,H,M,D]`. **H and Hkv are an evenly-dividing broadcast** (`H % Hkv == 0` and `H ≥ Hkv ≥ 1` — GQA / MQA,
    ADR [0067](decisions/0067-autoregressive-attention-vocabulary.md) decision 1.
    `r = H / Hkv` is a derived value with no attrs field. B matches exactly, and Hkv also matches exactly between k/v).
    **D is the same for all three** (allowing a different length on the v side alone would let "IR with D mixed up" slip
    through the shape check). **`scale` is a half scale** = `√scale_factor`, applied to **both** q and k
    (the same form as torch's `_scaled_dot_product_attention_math`. Changing it to a form applied once after the dot product
    changes the rounding sequence and loses bit identity with the decomposed path).
    The optional fourth input is an **additive mask** (f32, rank-4, shape exactly `[1,1,M,N]`),
    broadcast to every batch of B·H. `[B,1,M,N]` / `[1,H,M,N]` / bool / rank≠4 are not accepted, and
    causal / dropout are not in the vocabulary. A mask whose row is all −inf is **a valid input with output 0** (ADR
    [0044](decisions/0044-runtime-attention-mask.md) addendum 2026-08-31 — the fused-attention part of decision 3
    was withdrawn). An all −inf row is a contract violation (NaN contamination) only for the plain
    `softmax` of the decomposed path.
    With the optional **`states` field** it becomes the autoregressive states form (the
    "State-referencing nodes" section above — a contract extension under the same op name, where the presence of the field determines the form).
    When the states form further declares the optional attrs **`readonly`**, it becomes **arity 1** (the q-only
    past-only form — ADR [0096](decisions/0096-speculative-decoding.md) stage 2 §1.2)
  - `state_append` (f32, **arity 1, zero outputs**, optional attrs `window`, `states` field
    `{ slot }` required — ADR
    [0067](decisions/0067-autoregressive-attention-vocabulary.md) decision 5) — an effect op that writes the current step's k / v into
    a state slot (the "State-referencing nodes" section above). **The only op that defines no value**;
    there is no path to emit it from torch (aten has no counterpart, and only the decode graph scripts emit it)
  - `rms_norm` (f32, attrs `eps`, **arity fixed at 2**) —
    `y = x · rsqrt(mean(x², last dimension) + eps) · weight`. **No bias, and no mean subtraction**
    (these two points are the difference from layer_norm). The canonical normalized length is **the length of weight**, and there is
    no `normalized_shape` field (the same fact is not held twice, in attrs and in an input).
    `weight` is rank 1 with the last dimension's length, and `eps` is a finite positive number. For a form without weight the exporter
    composes ones and normalizes it to arity 2
  - `softmax` (f32, attrs `dim`) — **last dimension only** (`dim` in non-negative notation). Computed as **safe-softmax**
    (subtracting the row maximum)
  - `safe_softmax` (f32, attrs `dim`) — **the same contract** as `softmax` + "**a row whose max is −inf is
    all 0**" (ADR [0044](decisions/0044-runtime-attention-mask.md) — extended composite layer
    (formerly layer 2)). The same semantics as the safe-softmax guard that torch's SDPA
    decomposition puts over `softmax`; it appears only in forms where the mask is a run-time value and the guard cannot be proven
    inactive. Values of rows that have finite elements are **bit-identical** to `softmax`
  - `embedding` (attrs `padding_idx`) — a row gather `weight f32[V,H] × index i32[…] → f32[…,H]`.
    **The dtype differs per input slot** (weight: f32 / index: i32 → output f32).
    `padding_idx` is **accepted but has no effect on forward** (a gradient-side field). Out-of-range indices are handled
    as in `gather` (the GPU contaminates only the affected rows with NaN / the CPU reference throws)
  - `masked_fill` (attrs `value`) — `out = mask ? value : x`. dtype per slot
    (x: f32 / mask: bool → output f32), and **the output always has the same shape as x** (only mask is read with right-aligned
    broadcast). `value` is a finite f32 scalar
  - `conv1d` (f32, attrs `stride` / `padding` / `dilation` / `groups`, **arity fixed at 3**) —
    `x[B,Cin,L] * W[Cout,Cin/groups,K] + b[Cout]`. The output length is
    `floor((L + 2·padding − dilation·(K−1) − 1) / stride) + 1`. **All four attrs are required, and
    no default is filled in** (if omission were allowed, depthwise IR would silently run as an ordinary convolution).
    The contract is that `groups` divides both `Cin` / `Cout` and that the weight's second axis is `Cin/groups`
  - `conv2d` (f32, attrs `stride` / `padding` / `dilation` / `groups`, **arity fixed at 3**) —
    `x[B,Cin,H,W] * W[Cout,Cin/groups,Kh,Kw] + b[Cout]`. **The three spatial attrs are arrays of length 2,
    `[H, W]`** (scalar notation is not accepted — it would give two IRs for the same convolution. Normalization is the
    exporter boundary's job). Only `groups` is a scalar. The output length for each of H / W is
    `floor((L + 2·padding − dilation·(K−1) − 1) / stride) + 1`. The contract is that `groups` divides both
    `Cin` / `Cout`, that the weight's second axis is `Cin/groups`, and **the order `Kh` then `Kw`**
    (as with conv1d, all four attrs are required and no default is filled in)
  - `conv_transpose1d` (f32, attrs `stride` / `padding`, **arity fixed at 3**) —
    `x[B,Cin,L] ⊛ᵀ W[Cin,Cout,K] + b[Cout]`. **The weight is `[Cin, Cout, K]`, transposed relative to conv1d**.
    `stride ≥ 1` MUST hold (0 divides by zero in the kernel and hangs). **Only** forms whose output length is exactly
    `L·stride` are accepted, that is, forms where `2·padding == K − stride` holds; the general form
    `(L−1)·stride − 2·padding + K` fails loudly (widened by revising this document if demand appears).
    `output_padding` / `dilation` / `groups` have no field in attrs = fixed at 0 / 1 / 1

  A conv without bias (one in measurement, dec's `conv_post`) is normalized to arity 3 by **the exporter composing a zero bias initializer**
  — there is no arity branch in the IR, the contract or the kernels.

- **3D convolution** (extended composite layer — ADR [0118](decisions/0118-wan21-video-generation.md) decision 1.
  The exporter preserves `aten.conv3d`, which is outside Core ATen):
  - `conv3d` (f32, attrs `stride` / `padding` / `dilation` / `groups`, **arity fixed at 3**) —
    `x[Cin,T,H,W] * W[Cout,Cin/groups,Kt,Kh,Kw] + b[Cout] → [Cout,Tout,Hout,Wout]`. x is
    **unbatched rank 4 only** (batched rank 5 is not in the contract — the surrounding cat / slice hit the strided family's
    rank limit of 4. It fails loudly at the exporter boundary). **The three spatial attrs are arrays of length 3,
    `[T, H, W]`** (scalar notation and two components are not accepted — the same rule as conv2d's `[H, W]`), and
    only `groups` is a scalar. All four are required and no default is filled in. The output length for each of T / H / W is
    `floor((L + 2·padding − dilation·(K−1) − 1) / stride) + 1` (K = 0 and forms whose window does not reach the input
    fail loudly). Padding is **symmetric zeros** per axis; causal (one-sided) temporal padding is not in the vocabulary
    (the recipe expresses it by cat-ing the previous chunk's cache onto the start of the time axis). The contract is that `groups` divides both
    `Cin` / `Cout`, that the weight's second axis is `Cin/groups`, and **the order `Kt, Kh, Kw`**.
    The semantics are general in groups (the CPU reference holds all of it), but **the GPU's implemented subset is groups == 1**
    (implicit GEMM), and groups > 1 fails loudly when the Session is built (ADR 0064 axis A). The weight is
    eligible for f16 / i8 (weight slot 1, per-channel scale along axis 0)

- **Deformable convolution** (an atom of the extended atomic layer (formerly layer 1') — `torchvision::deform_conv2d`. ADR
  [0055](decisions/0055-deform-conv2d.md); the doors are defined in
  [0059](decisions/0059-op-vocabulary-entry-doors.md)):
  - `deform_conv2d` (f32, attrs `padding`, **arity fixed at 5**) —
    DCNv2 with `x[B,Cin,H,W]` / `W[Cout,Cin,Kh,Kw]` / `offset[B,2·Kh·Kw,Hout,Wout]` /
    `mask[B,Kh·Kw,Hout,Wout]` / `b[Cout]` → `[B,Cout,Hout,Wout]`.
    The output space is `Hout = H + 2·padH − (Kh−1)` / `Wout = W + 2·padW − (Kw−1)`, and **it is the contract that** the two spatial axes of `offset` /
    `mask` **match it** (the canonical output shape is on the x + weight + attrs side).
    The sample coordinates are `y = (oy − padH) + kh + off_y` / `x = (ox − padW) + kw + off_x`, and
    **the channel order of offset is nested `(kh, kw)`, innermost even = y / odd = x**.
    `mask` (the modulator) is applied **after bilinear interpolation**, and out of bounds is **zero-filled, not border
    clamped** (if the centre is outside `(−1, in)` the whole tap is 0; even inside, only an out-of-range corner is 0).
    `stride` / `dilation` / `groups` / `offset_groups` **have no field** in attrs = fixed at 1, and
    `mask` is required as a slot = **DCNv2 only** (DCNv1 with `use_mask=False` has no representation
    = fails loudly at the exporter boundary). **NaN in offset is not dropped to 0 but propagates to the output**
    (treated separately from the out-of-range 0 — ADR 0055 decision 5)

- **Spatial resample** (an atom of the Core ATen layer (formerly layer 1) — `aten.upsample_bilinear2d.vec`):
  - `upsample_bilinear2d` (f32, attrs `output_size`, **arity fixed at 1**) —
    bilinear interpolation `x[B,C,H,W] → [B,C,Hout,Wout]`. **`align_corners = True` only**;
    `align_corners` / `mode` / `scale_factor` **have no field** in attrs (`False`, nearest /
    bicubic / area / antialias have no representation = fail loudly at the exporter boundary. If demand for `False`
    appears, it is added as a separate op by the same approach as `gelu` / `gelu_tanh`). `output_size` is an array of length 2,
    `[Hout, Wout]`, as in conv2d; required, no default filled in.
    The source coordinate is `((in−1)/(out−1)) · output index` (an axis of output length 1 has factor 0), the taps are
    `index0 = trunc(source coordinate)` and `index1 = index0 + (index0 < in−1 ? 1 : 0)`, the weights are
    `λ1 = source coordinate − index0` / `λ0 = 1 − λ1`, and **the expression tree nests H outside and W inside**
    (the same order as torch's `aten/src/ATen/native/UpSample.h`). The factor may be non-integer, and
    **downscaling (`Hout < H`) is the same op** (reading only two taps is the same specification as torch, and is different from
    `area`). A spatial axis of length 0 is not accepted

- **Fixed activation quantization** (ADR [0097](decisions/0097-gemma4-qat-integration.md#addendum-2-2026-09-11--fixed-srq-op-contract)):
  - `static_quantize` (f32, arity 1, attrs `scale`) — the shape does not change. scale is non-negative, finite and an exact f32 value.
    It performs f32 division → integer rounding to nearest even → saturation to `[-128,127]` → f32 multiplication.
    scale=0 is an identity that keeps all bits. It keeps the sign of signed zeros and the sign and payload of NaN,
    and NaN with scale > 0 is quieted. ±Inf saturates, and output overflow is ±Inf.
    It does not depend on the GPU's division precision or on flushing of denormals; boundaries and the output table are read as integers.

- **RNN scan** (a composite of the extended composite layer (formerly layer 2) — ADR [0056](decisions/0056-gru-scan.md)):
  - `gru_scan` / `gru_scan_reverse` (f32, **empty attrs**, **arity fixed at 4**) —
    the GRU hidden-side scan `gi[T,N,3H]` / `h0[N,H]` / `W_hh[3H,H]` / `b_hh[3H]` → `y[T,N,H]`.
    **The time axis T may be symbolic** (the reason this op exists — the decomposed form is unrolled T times and loses the symbol).
    The op handles only the hidden-side recurrence; **the input-side GEMM (`x·W_ihᵀ + b_ih`) is prepared by the caller with the existing
    `linear`** (`gi` is its result). The formula for one step is
    `gh = W_hh·h + b_hh` / `r = σ(gh_r + gi_r)` / `z = σ(gh_z + gi_z)` /
    `n = tanh(gi_n + gh_n·r)` / `h' = (h − n)·z + n`, and **this order of operations and grouping is the contract**
    (the mathematically equivalent `(1 − z)·n + z·h` is a different rounding sequence — 44,345 of 100,000 f32 elements
    differ in bits). The gate order within 3H is **r / z / n**, `b_hh` is **last** (`(Σ W·h) + b`), and the
    reset gate applies to **the hidden-side product including b_hh**.
    `gru_scan_reverse` differs **only in scan order**, and **its output is in forward time order** (no `flip` in between —
    because `flip` does not accept a symbolic axis).
    Multi-layer / bidirectional / `has_biases=False` / `batch_first` / `dropout` **have no field** in attrs
    (layers and directions are expressed **by placing nodes in sequence** — `aten.gru`'s `Tensor[16]` does not fit the IR).
    The output is `y` only, and **`h_n` is not returned** (the IR's single-output premise). `h0` is an input slot and is
    required (a zero `h0` becomes an initializer through the exporter's constant folding).
    The hidden width is **`H ≤ 256`** (one lane = one hidden unit — exceeding it raises `CodegenError`)
  - `gated_delta_scan` (f32, **empty attrs**, **arity fixed at 5** — ADR [0135](decisions/0135-gated-delta-rule-scan.md),
    the value form) — the gated delta rule recurrence `q[B,T,Hk,dk]` / `k[B,T,Hk,dk]` / `v[B,T,Hv,dv]` / `g[B,T,Hv]` /
    `beta[B,T,Hv]` → `o[B,T,Hv,dv]`. **The length axis T may be symbolic.** q / k arrive **already l2-normalised**
    (q not scaled); g is the **log decay**. Per batch element, value head h and token t, from `S_0 = 0`
    (`S ∈ ℝ^{dk×dv}`): `S' = exp(g_t)·S_{t−1}` / `δ = β_t·(v_t − S'ᵀk_t)` / `S_t = S' + k_t δᵀ` /
    `o_t = S_tᵀ(q_t·dk^-½)` (the read is after the update — token t is included). **The head broadcast is derived
    from the shapes**: `Hv % Hk == 0` with `Hv ≥ Hk ≥ 1`, value head h reads key head `⌊h/(Hv/Hk)⌋`
    (= `repeat_interleave` on the head axis, the same rule as attention's GQA); the q scale `dk^-½` is derived from dk.
    B, T, dk and dv are at least 1. The initial state, the final state, the l2-normalisation, the scale and the head repeat
    **have no field** (a carried state is the states form, not in the vocabulary yet). The semantics hold for any head
    dims; **the GPU's implemented subset is `dk = dv = 128`**, and other head dims fail loudly when the Session is built.
    `g > 0` is not rejected but its result is unspecified on the GPU (docs/limitations.md)

The output dtype is decided by the mapping "**input dtype of slot 0 → output dtype**" (identity by default). The mapping is
**a column per output slot**, and the column's length is the op's output count (ADR 0068 decision 1). The only non-identity cases are the four comparisons
(→ bool), `sum` with bool input (→ i32), `where` (bool → f32), `argmax` (→ i32 indices) and
slot 1 of `topk` (→ i32 indices; slot 0's values are identity), and only `cast` is the exception decided by attrs.to.

The accepted set per dtype (including the accepted set **per input slot** and the derivation of the output dtype) and the value ranges of attrs
(including `cast`'s rounding convention, `permute`'s `dims` and each attr of the fused ops) are canonical in the contract table.

The order of extension follows the implementation order in [op-vocabulary.md](op-vocabulary.md).
