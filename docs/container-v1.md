# Karume container v1 specification

The concrete form of ADR [0108](decisions/0108-container-format.md). The canonical source for the
**physical format, descriptor schema, binding table and codec ledger** of `krm` (model) and `krg`
(graph).

What this document does not hold:

- **The graph representation of IR v2 itself** → [ir-v2.md](ir-v2.md) is canonical (the differences
  from v1 are in §13; the canonical serialization rules are also in the "Canonical serialization"
  section of ir-v2.md)
- **manifest `karume/5`** → ADR [0109](decisions/0109-manifest-v5-container.md) is canonical
- **op contracts** → the implementation's contract table (`packages/runtime/src/ops/contracts.ts`)
  is canonical
- **Execution defaults** (`session` / `gpuFeatures` / `requiredLimits` / `label` / `description`) →
  owned by the manifest (ADR 0038 §3). The descriptor does not hold them

Revision history (during 0.x there is neither dual reading nor a shim — the old format is converted
by the migration CLI (§12). Rationale: the Consequences of ADR [0108](decisions/0108-container-format.md) and decision 18 (version branching); the revision procedure: ADR 0003):

- v1 (2026-09-22): first edition. Makes decisions 1–20 of ADR 0108 concrete.
- v1 correction (2026-09-22, found while implementing step 1): ① move `capabilities.codecs` to
  **the model descriptor's `codecs`** (the graph descriptor MUST be byte-identical between `krm` and
  `krg`, so it cannot hold a field that depends on the binding table) ② drop the borrowed form
  `{ "shared": true }` from the binding table (only the IR side holds the shared declaration, and the
  key set of the binding table equals exactly "the initializers that are neither shared nor
  const-supplied") ③ `rowAxis` / `groupSize` / `scale` are written only for quantized codecs
  (writing them for a non-quantized codec fails loudly) ④ correct the packing of `int8-sym` to
  1 element / 1 byte / alignment 4 (4 / 4 / 1 would introduce a constraint `numel % 4 == 0` that v1
  does not have) ⑤ fix the order of descriptor arrays for writer determinism (`const.blocks` MUST be
  in ascending offset order, `blocks` in ascending (part, offset) order, `constants` in code-point
  order of (graph, initializer); the spelling of numbers and maps is the same as ir-v2.md's canonical
  serialization).
- v1 correction 2 (2026-09-22, rulings of step 2 — ADR 0109 / ADR 0108 addendum 2): ① split the
  digest table of §7 by whether the source is verified (via HF only once, in the fetch layer; block
  digests only for unverified sources) ② add to §8 the shape on the manifest side
  (`container.descriptor` = the expected values of the 2 documents, `container.parts` = all parts
  including part 0, a zero-length part is a 0-byte file, no field for a shared `krg`) ③ add to §11
  the fetch unit of step 2 (part; the seek / scan branch) ④ add to §12 the whole-repository mode of
  the migration CLI (writes `karume/5`) ⑤ add `length` to `assets` in §2.2 (the logical length of
  the payload; the block length MUST be that rounded up to a multiple of 4), and align the PLE role
  names with the implementation's `ple-values` / `ple-scales` / `ple-index`.
- v1 correction 3 (2026-09-24, implementing step 3 — ADR 0108 addenda 4 / 5, ADR 0109
  addendum 1): ① add the **graph-name rules** to §2.1 (container graph name = component name = key
  of `weights` MUST) ② the default part length of §4.2 stays at 256 MiB; the review of the block
  limit of §4.1 moves to step 6 ③ count RAM in §11 as 3 terms (the caller's holdings, the item being
  processed, fetch retention), add the copy of the carried-over scale, and make the fence once per
  part, governing staging only ④ bring §12 to the current CLI surface (2 modes,
  `provenance.writer` not written by default, `--part-bytes`, `--cross-repo`) ⑤ point references to
  removed implementation at their current locations.
- v1 correction 4 (2026-09-25, reconciled with the implementation): ① correct §1's "pad after the
  model descriptor up to 64 B alignment, and that is the end of part 0" to "part 0 = header + exactly
  the 2 documents (no padding)". It contradicted the MUST "no padding before a zero-length part" of
  §3 / §8. The 3 implementations (the runtime's `derivePartOffsets`, the exporter's mirror, the hub's
  part-0 length check) already wrote by the corrected rule ② change the part 0 / part 1 cells of
  the table in §8 to the same meaning ③ add `[padding]` to the formula of §9, and state that padding
  is not copied but derived again.
- v1 correction 5 (2026-10-10, reconciled with the implementation): ① state in §0 that duplicate
  JSON keys are accepted, last one wins (ADR 0108 addendum 11) ② state in §9 that only a `krm` is an
  extraction source, and that the reader checks the fetched length of the const region and
  `parts[0].sha256` ③ apply the part-length ceiling of §10 also to the `const.length` of a `krg` (the
  reader rejects it in the descriptor check — the spec table already included parts 0 / 1).
- v1 correction 6 (2026-10-11, ruling): state in §6.1 that the reader's merge layer rejects
  quantized weights with zero rows (`shape[rowAxis] = 0`) (real models have no use for them — an
  empty weight is not created at all, initializer included).
- v1 extension 7 (2026-11-03, ADR [0108](decisions/0108-container-format.md) addendum 15): ① a
  quantized binding's scale that one block cannot hold is split on the data's pieces —
  `encoding.scale.pieces` (§6.1), one scale block per data piece over the same rows, all in the part
  of piece 1 (rule ③ of §5) ② the writer checks scale blocks against the block limit when it plans
  the layout, before any byte is written ③ §11 counts a split scale as the joined scale it rides
  with piece 1. A strict superset: a container with no split scale is spelled and read exactly as
  before.

## 0. Notation and common rules

- All multi-byte integers are **little-endian**. `u32` / `u64` are unsigned.
- JSON is UTF-8. **`NaN` / `Infinity` literals are forbidden** (the writer writes with the
  equivalent of `allow_nan=False`, and the reader fails loudly when it detects one). Inherited from
  IR v1.
- **Unknown keys fail loudly** (there is no forward-compatibility channel — inherited from IR v1).
  The keys this document enumerates are all there are, and only fields stated as optional may be
  omitted.
- **Duplicate keys within one object are accepted, last one wins** (the reader sees only the later
  value, as `JSON.parse` / Python `json` do by default — the earlier value does not even reach the
  checks). Not writing duplicates is the writer's responsibility, and canonical serialization cannot
  write duplicates. This matches the manifest's handling (ADR 0038).
- A "`sha256`" field is always a string of **64 lowercase hex characters**.
- A "block id" is a string of 1–64 characters (`[A-Za-z0-9._:-]+`).
- Shape elements are non-negative integers or the canonical notation `coeff·sym+offset` (identical
  to IR v1's dimension language).
- Anything unsupported, undeclared, missing or surplus is **rejected with a full enumeration** (no
  silent approximation).
- The JSON key `__proto__` **can be neither written nor read** (the reader rejects it with the same
  check as non-finite numbers and depth, and the writer rejects it in canonical serialization). It is
  the name that, assigned to a plain `{}`, turns into a [[Prototype]] set and silently drops the
  declaration; in v2, where a tensor key becomes the initializer name as it is, it could in theory be
  spelled, so both sides block it.
- `"…"` in this document's JSON examples is **an elision on the page** (the real value is 64
  lowercase hex characters, per the rule of §0). The examples show the shape and the values
  themselves are not normative — the field tables of each section are normative.

## 1. Header

Fixed length of **24 bytes**.

| offset | length | field                         | value                                                  |
| -----: | -----: | ----------------------------- | ------------------------------------------------------ |
|      0 |      4 | magic                         | the 4 ASCII characters `KRMC` (`krm`) / `KRGC` (`krg`) |
|      4 |      4 | `u32` container version       | `1`                                                    |
|      8 |      8 | `u64` graph descriptor length | at least 1, at most the limit of §10                   |
|     16 |      8 | `u64` model descriptor length | at least 1 in `krm`; **`0` MUST in `krg`**             |

- **Only the magic** carries the kind (`krm` / `krg`). The descriptor has no `kind` field (the same
  fact is not held in 2 places).
- A magic of `KRGC` with a non-zero model descriptor length, or the reverse, fails loudly.
- The graph descriptor follows right after the header, and the model descriptor right after it,
  **packed** (there is no padding between the two).
- **Part 0 is exactly the header + the 2 documents** and contains no padding (file length of part 0
  in split form = 24 + graph descriptor length + model descriptor length). Padding (0x00) between
  parts in single form is determined by the layout rule of §8 — padding goes only right before the
  next non-zero-length part, so that it starts 64 B aligned (example: with part 0 of 100 B, part 1
  of 0 B and part 2 of 30 B, the part offsets are 0 / 100 / 128).

## 2. Descriptor (2 documents)

The descriptor is split into 2 documents, the **graph descriptor** and the **model descriptor**, so
that a `krg` can be extracted by byte copy (§9).

### 2.1 Graph descriptor

```jsonc
{
  "format": "karume-container",
  "version": 1,
  "capabilities": {
    "ops": ["matmul", "linear", "..."],
    "features": []
  },
  "graphs": {
    "transformer": {/* IR v2 graph (§13) */}
  },
  "const": {
    "length": 30736384,
    "blocks": [
      { "id": "const.rope.inv_freq", "offset": 0, "length": 512, "sha256": "…" }
    ],
    "constants": [
      {
        "graph": "transformer",
        "initializer": "rope_inv_freq",
        "block": "const.rope.inv_freq",
        "encoding": {/* §6.1 */}
      }
    ]
  }
}
```

| field                   | type     | required | limits and rules                                                                                                                                                    |
| ----------------------- | -------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`                | string   | required | fixed `"karume-container"`. A mismatch fails loudly                                                                                                                 |
| `version`               | integer  | required | fixed `1`                                                                                                                                                           |
| `capabilities.ops`      | string[] | required | **exactly equal** to the union of `requires.ops` over all `graphs`. The reader checks it against its own support table and rejects, enumerating the unsupported ops |
| `capabilities.features` | string[] | required | future extension point. The first edition accepts only an empty array (non-empty fails loudly)                                                                      |
| `graphs`                | object   | required | at least 1, **at most 64**. Keys are graph names (1–64 characters, `[A-Za-z0-9._-]+`). Values are IR v2 graphs                                                      |
| `const.length`          | u64      | required | byte length of the const region (§3). `0` when there is no const                                                                                                    |
| `const.blocks`          | array    | required | index of the const blocks (§3). Empty array when there is no const                                                                                                  |
| `const.constants`       | array    | required | the **binding table** that binds const blocks to initializers (§3). Empty array when there is no const                                                              |

**Graph-name rules** (they apply to every path that writes a distribution — the single export path,
`karume dist`, `karume migrate`):

- **Container graph name = component name in the distribution = key of `weights` in
  `karume.json`** MUST. The runtime looks graphs up by name (`prepareContainer(opened, <key of
  weights>)`), so a container whose spellings diverge gets installed together with its manifest and
  first fails at the user's `createSession` with "the container has no such graph" (the rationale is
  recorded in ADR [0109](decisions/0109-manifest-v5-container.md) addendum 1).
- **The name of the directory it is placed in is not a rule.** It can differ from the component name
  (in families that place the container directly under the series, the directory name is the series
  name — the key of siglip2 is `vision`; the key of irodori's `caption-proj` directory is
  `caption_proj`). Writers state the component name as a constant and do not derive it from the
  directory (the single export path `karume.pipeline` makes `graph_name` required, with no default).
- It is checked in 2 places: before assembling, `karume dist` checks that the set of graph names of
  the actual containers equals the keys of `weights` of that preset
  (`tools/exporter/src/karume/dist.py` — a mismatch is a `DistError`). The merge on the models side
  (`packages/models/src/hub/components.ts`) looks graphs up by the keys of `weights`, and when one is
  missing, `prepareContainer` fails, enumerating the graphs that exist.

Elements of `const.blocks[]`:

| field    | type   | required | rule                                                                           |
| -------- | ------ | -------- | ------------------------------------------------------------------------------ |
| `id`     | string | required | block id (§0). MUST be **disjoint** from the model descriptor's `blocks[].id`  |
| `offset` | u64    | required | offset **relative to the start of the const region**. MUST be a multiple of 64 |
| `length` | u64    | required | MUST be a multiple of 4; at most the block limit of §10                        |
| `sha256` | string | required | sha256 of this block's bytes (padding included)                                |

`const.blocks` is in ascending `offset` order; entries do not overlap and do not exceed
`const.length`.

Elements of `const.constants[]`:

| field         | type   | required | rule                                                                     |
| ------------- | ------ | -------- | ------------------------------------------------------------------------ |
| `graph`       | string | required | one of the keys of `graphs`                                              |
| `initializer` | string | required | an initializer name of that graph. `(graph, initializer)` MUST be unique |
| `block`       | string | required | points to a `const.blocks[].id`                                          |
| `encoding`    | object | required | §6.1                                                                     |

**The `const` binding table lives in the graph descriptor rather than the model descriptor because
it is the only means by which a `krg` alone can supply `const.*`** (a `krg` has no model descriptor
— §9).
The graph descriptor holds `const.length` for the same reason (the only means by which a `krg` knows
the size of part 1).
CPU spike ① spelled these `constRegion.constants[]` / `constRegion.length` — this document folds
them into the `const` object of the graph descriptor.

### 2.2 Model descriptor

Does not exist in a `krg` (the model descriptor length in the header is 0).

```jsonc
{
  "format": "karume-model",
  "version": 1,
  "codecs": ["int4-sym-g"],
  "parts": [
    { "index": 1, "length": 30801920, "sha256": "…" },
    { "index": 2, "length": 268435456, "sha256": "…" }
  ],
  "blocks": [
    { "id": "w.0", "part": 2, "offset": 0, "length": 33554432, "sha256": "…", "role": "weight" },
    { "id": "s.0", "part": 2, "offset": 33554432, "length": 6144, "sha256": "…", "role": "scale" }
  ],
  "binding": {
    "transformer": {
      "model.layers.0.mlp.gate_proj.weight": {
        "block": "w.0",
        "encoding": {
          "codec": "int4-sym-g",
          "packing": { "blockElements": 8, "blockBytes": 4, "alignBytes": 4 },
          "rowAxis": 0,
          "groupSize": 32,
          "scale": { "block": "s.0", "dtype": "f32" }
        }
      }
    }
  },
  "assets": {
    "ple.values.0": { "block": "a.0", "role": "ple-values", "length": 33546240 },
    "ple_index": { "block": "a.1", "role": "ple-index", "length": 165 }
  },
  "provenance": {
    "license": "gemma",
    "notice": "NOTICE",
    "upstreamRevision": "…"
  }
}
```

| field        | type     | required | limits and rules                                                                                                                                                                                                           |
| ------------ | -------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`     | string   | required | fixed `"karume-model"`                                                                                                                                                                                                     |
| `version`    | integer  | required | fixed `1`                                                                                                                                                                                                                  |
| `codecs`     | string[] | required | **exactly equal** to the set of codec registered names the binding table uses. Checked against the ledger **before a single byte of weights is fetched** (the reader also checks each `encoding.codec` against the ledger) |
| `parts`      | array    | required | **indices 1 and above only**; at least 1, **at most 1024** (§10). §4.2                                                                                                                                                     |
| `blocks`     | array    | required | 0 or more, **at most 65,536** (§10). §4.1                                                                                                                                                                                  |
| `binding`    | object   | required | graph name → initializer name → supply (§5). The keys are a subset of the keys of `graphs`                                                                                                                                 |
| `assets`     | object   | required | asset name → `{ block, role }`. An empty object when there are none                                                                                                                                                        |
| `provenance` | object   | required | §2.3                                                                                                                                                                                                                       |

Elements of `parts[]`:

| field    | type   | required | rule                                                                                                                 |
| -------- | ------ | -------- | -------------------------------------------------------------------------------------------------------------------- |
| `index`  | u32    | required | **1-based** sequence (`index = array index + 1` MUST). **Part 0 is not listed**                                      |
| `length` | u64    | required | byte length of the part. Part 1 is its actual length; parts 2 onward are at most one of the part-length set of §10   |
| `sha256` | string | required | sha256 of the whole part's bytes. **For checking at publication and repacking**; not used for runtime integrity (§7) |

**Why `parts` starts at index 1**: part 0 contains the model descriptor itself, so writing part 0's
sha256 into the model descriptor would be **self-referential** (the hash changes the moment it is
written). Hence **the integrity of part 0 is covered by the outer expected hash + length** (① of
§7). The FileRefs of manifest `karume/5` hold `size` / `sha256` for **all parts including part 0**,
so integrity as seen from outside has no hole.

**Part 1 (the const region) is always declared, with length 0 even when there is no const at all**
(§3). The part index "1 = const region" then stays constant, and the reader does not have to recount
the layout.

Elements of `blocks[]`:

| field    | type   | required | rule                                                                                                        |
| -------- | ------ | -------- | ----------------------------------------------------------------------------------------------------------- |
| `id`     | string | required | block id (§0). MUST be disjoint from `const.blocks[].id`; duplicates forbidden                              |
| `part`   | u32    | required | **2 or above** (part 0 is the descriptor, part 1 is for const only)                                         |
| `offset` | u64    | required | offset from the start of that part. MUST be a multiple of 64                                                |
| `length` | u64    | required | MUST be a multiple of 4; at most the block limit of §10                                                     |
| `sha256` | string | required | sha256 of the block's bytes (padding included)                                                              |
| `role`   | string | required | one of `"weight"` / `"scale"` / `"zero-point"` / `"asset"` (`"const"` appears only on the const index side) |

Values of `assets`:

| field    | type   | required | rule                                                                                                                                                                                                                                                                                              |
| -------- | ------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `block`  | string | required | points to a `blocks[].id`. The `role` of that block MUST be `"asset"`                                                                                                                                                                                                                             |
| `role`   | string | required | **name of the interpreter on the models side** (`ple-values` / `ple-scales` / `ple-index` / `rope-base` / `style-vectors` …). **The runtime does not interpret it** — the descriptor holds only "name → block"                                                                                    |
| `length` | u64    | required | number of payload bytes (logical length). Assets have no shape, so they declare it themselves. The `length` of that block MUST be **this value rounded up to a multiple of 4** (the amount of padding is closed by the declaration too, so the consumer does not strip trailing 0x00 by guessing) |

### 2.3 `provenance`

| field              | type   | required | rule                                                                                             |
| ------------------ | ------ | -------- | ------------------------------------------------------------------------------------------------ |
| `license`          | string | required | license **identifier**. The text is not included (the text is at the repository root — ADR 0071) |
| `notice`           | string | optional | reference to a NOTICE file (path fragment). The text is not included                             |
| `upstreamRevision` | string | optional | revision of the upstream checkpoint                                                              |
| `writer`           | string | optional | identification of the writer (the `karume` version, etc.)                                        |

`provenance` is **not carried over to a `krg`** (§9). Its purpose is that the origin can be traced
when a single `krm` is taken away as one file, and a `krg` is a "compute contract", not the origin
of a distributed artifact.

## 3. Const region

- The const region is **owned by the graph descriptor**. The index is `const.blocks` and the binding
  to initializers is `const.constants` (both in §2.1); **offsets are relative to the start of the
  const region**.
- Its contents are **weight-independent constants**: values baked at the constant-folding frontier
  and config-derived values (RoPE `inv_freq`, etc.). **Weight-derived scales are not const** (a
  scale is a block on the model descriptor side).
- In split form, **part 1 is the const region itself** (start of part 1 = start of the const
  region). In single form, the `const.length` bytes right after part 0 (after 64 B alignment) are
  the const region.
- When there is no const, `const.length = 0`, `const.blocks = []`, `const.constants = []`, and in
  split form **part 1 is placed with length 0** (so that the meaning of part numbers does not depend
  on the layout).
- **No padding is inserted before a zero-length part** MUST. Inserting it would make a `krg`
  extracted from a `krm` with empty const and a `krg` written directly for the same graph differ in
  bytes (breaking the byte identity of §9).

Const is not necessarily small. Measured (aggregate over the 447 mirror files): 0.08 % of the total
for gemma4 e2b i4, 29.3 MB (2.5 %) for birefnet 2048 f32, 2.1 MB (**29.7 %**) for sbv2 F1 front
i4. Hence only part 0 is fetched at preparation, and **const is fetched when it is needed** (§11).

**When a single const block exceeds the block limit (§10), the writer stops with fail loudly.**
Const has no piece-splitting mechanism (only the weight binding table can express a piece sequence,
and a `krg` does not have it — §9), so a const above the limit **cannot be represented**. The largest
today is birefnet 2048's f32 constant of 29.3 MB, leaving only about 2.7 MB of margin to 32 MiB.
For the same reason a const's scale cannot be split (a split scale follows the data's pieces —
rule ③ of §5): a const whose scale exceeds the limit also stops the writer, and the reader rejects
`scale.pieces` on a `const.constants` entry.

**Open** (to be ruled when a real need appears; no escape hatch is built now): ① allow piece
sequences for const too (= a minimal weight binding table comes back to `krg`) ② give const alone
its own block limit ③ treat large constants as assets (`assets`) and fetch them lazily.

## 4. Blocks and parts

### 4.1 Block

**A block is an independent unit of fetch, verification and release.**

- **A block does not span parts** MUST.
- **Block length ≤ 32 MiB = 33,554,432 B** (§10, an independent constant). **"At most", not "less
  than".**
- **Block starts are 64 B aligned** (∴ offsets are automatically multiples of 4), and **block
  lengths are multiples of 4** MUST.
- Trailing padding is **baked by the writer**. Padding bytes are **fixed at 0x00** MUST.
  - The byte length of the payload is determined by the codec and the declared shape (§6). Calling it
    `payloadBytes`, `payloadBytes ≤ length` and `length − payloadBytes < 4` MUST.
  - `sha256` is taken over all `length` bytes, padding included. This **defines the bit identity of
    repacking including the padding**.
- Within one part, blocks do not overlap and do not exceed the part length. Gaps between blocks are
  allowed (alignment padding), but those gaps MUST also be 0x00.
- **The hashes that cover padding work at 2 levels**:

| Position of the padding        | Covering hash                        |
| ------------------------------ | ------------------------------------ |
| End of a block                 | **that block's `sha256`** (in scope) |
| Between blocks / between parts | **only that part's `sha256`**        |

- The reader can digest each block independently and, once it is uploaded to the GPU, release it
  independently.

There are 3 grounds for making the block limit **at most** 32 MiB (measured in CPU spikes ① / ②,
2026-09-22):

1. **Splitting does not raise the total digest cost.** Measuring 256 MiB as the median of 5 rounds,
   a single digest takes 277.4 ms / **923 MiB/s**, while per-block digests of 32 MiB × 8 (passing a
   `subarray`) take 272.9 ms / **938 MiB/s** (ratio 0.984). A "verify per block" design needs no
   excuse on the digest side.
2. **The largest weights of real assets sit exactly on the limit.** The 2048×8192 f16 weights of the
   anima transformer are exactly 33,554,432 B, and **56** of them have this size. With the limit as
   "at most", **0** are piece-split; with "less than", the same **56** would be split. The
   distribution of block lengths is 32 MiB×56 / 24 MiB×1 / 8 MiB×169 / 4 MiB×56 / 3 MiB×84 /
   2 MiB×1 / 1 MiB×85 / 272 KiB×1.
3. **Const blocks cannot be piece-split** (§3), so birefnet 2048's f32 constant of **29.3 MB** has
   to fit. There is only about 2.7 MB of margin to 32 MiB.

Implementation rules:

- **Pass a `subarray`** to digest (inserting a `slice` copy drops it from 943 to 641 MiB/s).
- Concurrent digests help (16 MiB × 16 with `Promise.all` reaches **3,702 MiB/s** = 69.2 ms; 2.07×
  sequential 16 MiB digests and 4.0× a single 256 MiB digest). A single digest cannot get this.
- Browsers are unmeasured. Chrome copies the whole digest input into Blink internals (recorded at
  `BYTE_BUDGET` in `packages/hub/src/fetch.ts`), so blocking should help in browsers most of all
  (**speculation** — the amount copied is capped by the block length).

**Future review (lowering the limit to 16 MiB)**: in measurements that alternate sizes, 16 MiB =
1,784 MiB/s against 32 MiB = 938 MiB/s, so on speed alone 16 MiB wins. Lowering it has 2 side
effects — ① **the const limit breaks first** (birefnet's 29.3 MB no longer fits, and one of the 3
open options of §3 becomes necessary) ② **the block count and the descriptor double**. Switching is
one constant. The RAM-peak measurement of step 3 measured only the part length, not the block limit,
so the review is carried over to step 6 (the round for new bit widths and Range fetches).

**Withdrawn ground**: the draft used as the ground for the block limit a cliff of "1,865 MiB/s up to
exactly 32 MiB, halving to 948 MiB/s from 33 MiB", but re-measurement **did not reproduce it**. Those
numbers are an artifact of the measuring method, appearing only when sizes are measured
consecutively in ascending order (measuring 31.0 → 31.5 → 31.94 → 31.999 → 32.0 MiB in a row makes
32 MiB give 1,840 MiB/s). Alternating with 16 MiB gives the table below, and **the cliff is below
32 MiB**.

| block length            | throughput of the alternating measurement |
| ----------------------- | ----------------------------------------: |
| **16 MiB**              |                           **1,784 MiB/s** |
| **32 MiB**              |                             **938 MiB/s** |
| 40 MiB                  |                                 959 MiB/s |
| 64 MiB                  |                                 933 MiB/s |
| 256 MiB (single digest) |                                 923 MiB/s |

The cause is not identified (**speculation**: 32 MiB equals the upper bound of glibc's mmap
threshold, and beyond this boundary every allocation runs mmap / munmap). **A digest measurement
that varies sizes reaches the opposite conclusion unless the sizes are alternated.**

### 4.2 Part

**A part is the unit of delivery** (not the host-RAM buffer — that is the block).

- part 0 = `[header][graph descriptor][model descriptor]`, part 1 = the const region, part 2
  onward = weight / asset blocks.
- **The writer chooses the part length**: one of `{256, 512, 768, 1024} MiB` (default
  **256 MiB**). Parts 0 / 1 have their actual length and are not constrained by this set (but they
  obey the "part-length ceiling" of §10). Note that **part 0 is not listed in the model descriptor's
  `parts`** (it would be self-referential — §2.2).
- **An asset that needs range reads is placed alone in a dedicated part** MUST (1 block = 1 part).
  The `ple-values` / `ple-scales` block sequences of PLE are such assets (per token, values
  8,960 B + scale 140 B = 9,100 B, read in 2 reads — ADR 0109 decision 4, ADR 0108 addendum 3
  item 2). Depending on the source, a range read is a scan (proportional to the offset; the default
  path on Deno and Node — `defaultReadStrategy` in `packages/hub/src/sources/hf.ts`), so sharing a
  part would turn it into "a walk from the start of the part" and lose the benefit of range reads.
- **Both** the manifest and the descriptor declare the length of each part. The reader can estimate
  host RAM before fetching a single byte (ADR 0089's approach — how far this closes differs by source
  type, see §11).
- **The default is 256 MiB** (ADR 0108 addendum 5 item 1 — measured in step 3e). Seek-type sources
  (the browser's Blob, local range reads) read per block, so the host-RAM peak does not depend on the
  part length. Scan-type sources (via HF — the fetch layer's strategy stream) read a part at once and
  cut it into blocks, and the hub's holding slot keeps the part, so the peak grows roughly with the
  part length. Raising the default gains only a smaller number of parts, and range-read assets are in
  dedicated parts, so even the number barely drops. The fetch surface is ADR 0109 decision 7.
- Part count ≤ 1024 (inherits the value of the old `MAX_SHARDS` — the constant is `MAX_PARTS`:
  Python canonical source `tools/exporter/src/karume/container.py`, hub
  `packages/hub/src/manifest.ts`, runtime `packages/runtime/src/format/container/limits.ts`).

**The old `SHARD_BYTE_LIMIT` = 256 MiB** (the upper bound of the shard length of the old shard
distribution) — where its 3 grounds went:

| #   | Old ground                                                                                      | Goes to                                                                         |
| --- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| R1  | host-RAM peak = constant + one largest shard                                                    | **moves to the block limit**                                                    |
| R2  | file and request counts within the hub's 4-way concurrency and the reader's limit of 1024 files | **stays with the part**                                                         |
| R3  | well below Chromium's single-ArrayBuffer limit of 2,145,386,496 B                               | **moves to the block limit** (a whole part is no longer put in one ArrayBuffer) |

## 5. Binding table and rules ①–⑤

A binding is **identified by `(graph name, initializer name)`**. Its value is **the one supply
form** (borrowing is not listed in the binding table — below):

**Supply form**

```jsonc
{
  "block": "<block id>",
  "encoding": {/* §6.1 */},
  "pieces": [{ "block": "<block id>", "rows": [0, 4096] }]
}
```

| field      | type   | required                           | rule                                                   |
| ---------- | ------ | ---------------------------------- | ------------------------------------------------------ |
| `block`    | string | required when there is no `pieces` | block id of the single body. `role` MUST be `"weight"` |
| `encoding` | object | required                           | §6.1                                                   |
| `pieces`   | array  | optional                           | 2 or more. **Mutually exclusive with `block`**         |

An element of `pieces[]` is `{ block, rows: [begin, end) }`. `rows` is a half-open interval over the
**leading dimension**.

**Borrowing (shared) is not listed in the binding table.** Borrowing from another Session (currently
`sharedWeights` — `sharedWeights` in `packages/runtime/src/runtime/session-types.ts`) is held only by
the IR-side `{ "shared": true }` declaration ([ir-v2.md](ir-v2.md), "Shared initializers" — the
borrower's name = the lender's initializer name) and lies **outside the set the binding table is
checked against**. Writing it into the binding table fails loudly as a surplus.

### Rules

- **① 1 block ≤ 1 binding** (1 body, 1 initializer). Two bindings pointing to the same block id
  fail loudly. One initializer **used by several nodes** is allowed (that is a different matter).
  Scale / zero-point blocks likewise belong to only one binding (each piece of a split scale too).
- **② The length and offset of every block are multiples of 4.** Offsets are covered by the 64 B
  alignment rule (§4.1). For lengths, this **generalises** the old rule "the block length of a
  non-final piece is a multiple of 4". The writer bakes trailing padding (§4.1), so the reader needs
  no padding branch.
  - **This rule applies to all blocks, not only weight blocks** — const and assets included.
    Anything whose length is not a multiple of 4 (bool tables, odd-length f16, etc.) is **declared
    with its padding**, and **the consumer derives the logical length from the declared shape** (it
    does not use the block length as the logical length).
  - **Padding may be applied only to a whole single block and to the last piece of a piece
    sequence.** **It must not be applied to a middle piece** (doing so overwrites the start of the
    next piece — as the note on `tailAligned` in `session-build.ts` says). Hence **the byte length of
    a middle piece MUST be a multiple of 4 to begin with**; the writer cuts at a row step rounded to
    `4/gcd(rowBytes, 4)`, and fails loudly when it cannot round.
- **③ The companion scale block is in the same part as the body.** For a piece sequence, **in the
  same part as piece 1**. Zero-point blocks follow the same rule.
  - **A scale that one block cannot hold is split on the data's pieces** (ADR 0108 addendum 15):
    `encoding.scale.pieces` in place of `scale.block` (§6.1), **one scale block per data piece**,
    scale piece k covering exactly the rows of data piece k along `rowAxis` (`rowAxis` is 0 for
    every piece sequence — rule ④). Its length is `rows × (rowLength / groupSize) × 4` bytes — for
    `int4-sym-g` with `groupSize` 32 that is `rows × rowLength / 8`. **Every scale piece is in the
    same part as data piece 1** (Session construction reads the whole scale with piece 1).
  - The writer splits a scale **only when it exceeds the block limit** (the same rule that splits
    the data); a scale that fits MUST be one `scale.block`. When no split fits — the data is one
    block, or a scale piece would still exceed the limit, or piece 1 and the whole scale exceed one
    part — the writer fails loudly while planning the layout, before any byte is written.
  - The reader rejects, with the path of the field: `pieces` with fewer than 2 entries or together
    with `block` (or neither); an empty row range, a gap or an overlap (the pieces continue from
    row 0 — checked when parsing); a piece block whose role is not `"scale"`, that is bound twice,
    that is outside the part of data piece 1, or that exceeds the block limit (§4.1); and in the
    merge layer, a split scale beside data in one block, a piece count other than the data's, rows
    other than the data piece's, or a piece length other than its rows' bytes (a middle piece has
    no padding; the last one may have less than 4). The total then equals the scale's payload
    length (coverage + per-piece lengths), so it has no check of its own.
- **④ Pieces cover the row range without gaps, and their parts are non-decreasing in index order**
  (MUST — the construction side does its bookkeeping on the assumption that piece 1 is in the first
  part. Reverse order is rejected by `openContainer` — 2026-09-25). `rows` in ascending order,
  `begin[0] = 0`, `end[k] = begin[k+1]`, `end[last] = shape[0]` MUST.
  **An initializer with `encoding.rowAxis != 0` may not be piece-split** (the row range of the scale
  does not correspond to the row range of the pieces — today's exporter also forbids row splitting of
  `conv_transpose1d`).
- **⑤ The declared shape / dtype and the actual data match exactly.** Missing and surplus items are
  **rejected with a full enumeration**.
  - The key set of `binding` MUST **exactly equal** the set of that graph's `initializers` that are
    not `shared`.
  - Blocks in `blocks` / `const.blocks` that no binding / `assets` references are rejected as
    **surplus**.
  - The payload byte length is determined by `encoding` and the declared shape (§6). Rejected when
    the measured length differs.

## 6. `encoding` and the codec ledger

### 6.1 Fields of `encoding`

```jsonc
{
  "codec": "int4-sym-g",
  "packing": { "blockElements": 8, "blockBytes": 4, "alignBytes": 4 },
  "rowAxis": 0,
  "groupSize": 32,
  "scale": { "block": "s.0", "dtype": "f32" },
  "zeroPoint": { "block": "z.0" }
}
```

A split scale (rule ③ of §5 — the scale of a piece sequence that one block cannot hold) spells
`scale` with the piece element of §5 instead of `block`:

```jsonc
"scale": {
  "pieces": [
    { "block": "s0017", "rows": [0, 17476] },
    { "block": "s0018", "rows": [17476, 34952] }
    // … one per data piece, the same rows as the data's pieces
  ],
  "dtype": "f32"
}
```

| field                   | type   | required  | rules and limits                                                                                                                           |
| ----------------------- | ------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `codec`                 | string | required  | registered name in the ledger (§6.3). **A name not in the ledger is rejected before weights are fetched**                                  |
| `packing.blockElements` | u32    | required  | at least 1. The **number of elements** one packing block carries                                                                           |
| `packing.blockBytes`    | u32    | required  | at least 1. The **number of bytes** of one packing block                                                                                   |
| `packing.alignBytes`    | u32    | required  | one of `1 / 2 / 4 / 8 / 16 / 64`. Alignment requirement of the payload start                                                               |
| `rowAxis`               | u32    | quantized | `0` or `1`. **The row axis**. Written only for codecs whose ledger `scale` is required (writing it for a non-quantized codec fails loudly) |
| `groupSize`             | u32    | quantized | at least 1, MUST divide the row length. For a per-channel codec MUST **equal** the row length                                              |
| `scale.block`           | string | quantized | block id of the scale. `role` MUST be `"scale"`. **Exclusive with `scale.pieces`**; a scale that fits one block MUST be written this way   |
| `scale.pieces`          | array  | optional  | a split scale (rule ③ of §5): 2 or more `{ block, rows }`, scale piece k with the rows of data piece k; each `role` MUST be `"scale"`      |
| `scale.dtype`           | string | quantized | the dtype the codec implies (the ledger entry's `scaleDtype`: `"f32"`, and `"f16"` for `int4-sym-g-f16` only — another value fails loudly) |
| `zeroPoint.block`       | string | optional  | may be written only for codecs whose ledger `zeroPoint.allowed` is true (false for all 4 in the first edition ⇒ always omitted)            |

`zeroPoint` has no split form: no ledger codec allows a zero-point, so none is defined until one does
(it would take the same `pieces` spelling as `scale`).

**The bit count is not part of the declaration.** It is a value derived from
`packing.blockElements` / `packing.blockBytes` (`bits = blockBytes · 8 / blockElements`). With
`bits` in the declaration, a non-integer bpw could not be expressed and the payload byte length would
not be determined either.

**Computing the payload byte length** (a generalisation of the exact match `numel × bits / 8` of the
old distribution — implemented by `payloadBytes` in
`packages/runtime/src/format/container/codecs.ts`):

```text
numel % blockElements == 0                   ← MUST
payloadBytes = numel / blockElements * blockBytes
```

**`packing` is a copy of the ledger, and for the reader it is there to be checked.** The reader
**checks that the declared `packing` equals the values of the ledger entry, and fails loudly on a
mismatch** (it is placed not as an independently updated field holding a derivable fact, but as a
checkpoint for a cross-package invariant — the same approach as the hub's `manifest.ts` holding
`MAX_PARTS` / `MAX_DESCRIPTOR_BYTES` at the same values as the Python canonical source
(`container.py`) and naming them as checkpoints). This is the check that keeps "assets written with a
different version of the ledger" from being accepted silently.

**Exposing `rowAxis` in the declaration** is the substantive improvement of this version. In IR v1,
the channel axis of i8's per-channel scale was not written in the declaration and **was derived from
the consuming op** (`weightChannelAxes` in `packages/runtime/src/runtime/plan.ts` — it now remains as
a check against the declaration, failing when the consuming op disagrees — §13.2). With `rowAxis` in
the declaration, **the meaning of the scale is closed by the declaration alone**.

**The shape of the scale is unified to the rank-2 group form**:

```text
scaleShape = [ shape[rowAxis], rowLength / groupSize ]
rowLength  = numel / shape[rowAxis]
```

**Degenerate form with zero elements** (`in_features = 0`, etc., row length 0): the per-channel
`groupSize` cannot satisfy the MUST of at least 1, so it is **1**, and the group count is **1**
(there is one per-channel scale per row, which matches `[rows, 1]` of the old distribution). The
`rowAxis` of the group codec (`int4-sym-g`) is **0** only (the decode kernel and `decodeI4` treat
the leading dimension as rows).

**Zero rows** (`shape[rowAxis] = 0` — not a single quantized row): such quantized weights are **not
accepted**. The reader fails loudly in the merge layer (`format/container/bind.ts`), whatever the
supply source (real models have no use for them — an empty weight is not created at all, initializer
included).

The "keepdim broadcast form of the same rank as the weight" that only i8 used in the old
distribution is retired (only the migration CLI reads it — §12). Taking i8 per-channel as
"row = the dimension `rowAxis`, group length = row length" gives `[shape[rowAxis], 1]`, which is the
same shape i2 already uses (`[N,1]`).

### 6.2 Codec ledger entries

The ledger is **immutable data in the repository**. Nothing registers at import time (that would
violate the cross-cutting invariant "zero side effects in every module"). The ledger holds **only the
storage format**; CPU decoding and whether it can be executed are held by other owners (ADR 0108
addendum 12 — the "one entry bundles the 3" of decision 12 is withdrawn).

```text
registered name → {
  layout,                                                // kind of decode path (f32 / f16 / bf16 / i32 / i8 / i4 / i2)
  packing:   { blockElements, blockBytes, alignBytes },  // canonical source of the declared byte length and alignment
  scale:     "required" | "forbidden",                   // whether it is a quantized codec (whether scale / rowAxis / groupSize are needed)
  grouping:  "channel" | "group" | undefined,            // per-channel or group quantization (none for non-quantized)
  zeroPoint: "allowed" | "forbidden",                    // forbidden for all 4 in the first edition
  rowAxes,                                               // set of rowAxis values that may be declared (empty for non-quantized)
}
```

The canonical source is `CODEC_LEDGER` in `packages/runtime/src/format/container/codecs.ts`, and the
Python mirror is `CODEC_LEDGER` in `tools/exporter/src/karume/container.py` (a test checks that the
two agree — on every field except `zeroPoint`. Python's `CodecEntry` has no `zeroPoint` field, and
the Python reader rejects `encoding.zeroPoint` whatever the codec). The scale dtype is a per-codec field
(`scaleDtype` / `scale_dtype` — `f32` for every quantized codec but `int4-sym-g-f16`, whose steps
are `f16`; the same test checks it), within the accepted set `SCALE_DTYPES`. The value set, pack order and reconstruction formula (the table of §6.3) are the
codec's spec, not ledger fields — CPU decoding and WGSL implement them.

**The 3 axes are reported separately, and each axis has its own owner**:

| Axis                                            | What it decides                                        | Owner (canonical source)                                                                                                                         |
| ----------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| ① readable as a declaration                     | accept / reject                                        | the ledger (`codecs.ts`) and the declaration rules (`descriptor.ts` / `memory.ts` / `bind.ts` — "Canonical locations" in §6.3 below)             |
| ② executable                                    | execute / reject, enumerating the missing capabilities | `RUNTIME_SUPPORT.storage` (the set of decode paths) in `packages/runtime/src/ops/contracts.ts` and `assertRuntimeSupport` in `ops/support.ts` |
| ③ which ops can keep it compressed and resident | packed residency / CPU decoding at load time           | the predicates in `packages/runtime/src/runtime/plan.ts` and `planWeightResidency` in `runtime/weight-residency.ts`                              |

The predicates of axis ③ are 3, by decode path: the common `eligibleCompressedInitializers`
(consumed by weight slots only), and `i4EligibleInitializers` / `i2EligibleInitializers`, which
narrow it by intersection.

Until 2026-10-17 (ADR 0131), `bf16` satisfied only ① (the declaration is valid because it is in the
ledger; execution fails loudly because it is not in `RUNTIME_SUPPORT.storage`) — it now has ①②③,
like f16. `i2` has ①②③, but ③ only for linear / embedding.
**The path of "accepting a new codec as a declaration first" works** (bf16 is the real example: its
execution path landed 2 months after the declaration).

Decoding is implemented in 2 places, separately from the axes.

- **CPU decoding**: `decodeF16` (`f16.ts`), `decodeBf16` (`bf16.ts`), `decodeI8` (`i8.ts`),
  `decodeI4` (`i4.ts`) and `decodeI2` (`i2.ts`) under `packages/runtime/src/format/`, one per
  decode path. They are called from `packages/runtime/src/runtime/session-build.ts` (for weights
  that axis ③ rules ineligible).
- **GPU decoding (WGSL)**: the generator shared by the 5 fused kernels,
  `packages/runtime/src/kernels/weight-storage.ts`, and the dedicated kernels (`linear-gemv.ts` /
  `linear-i8a8.ts`, etc.) hold the read for each storage form.

**Dependency direction MUST**: the format layer (ledger, declaration rules, CPU decoding) knows
nothing about execution. `format` does not import `ops/` / `runtime/` / `kernels/`; the execution
side reads the ledger's `layout` and branches.

The canonical pack order is the exporter's (`pack_int4` / `pack_int2` in
`tools/exporter/src/karume/emit.py`), and CPU decoding (the MUST at the top of
`packages/runtime/src/format/i4.ts` — decoding lives only in this one place) and WGSL hold its
mirror.

### 6.3 First-edition codec ledger (4 codecs)

All values are **transferred verbatim** from the current implementation (not a single new value is
created).

**The registered names (`int8-sym` / `int4-sym-g` / `int2-off` / `ternary`) are settled spellings
(ruling of 2026-09-22).** They are names this document chose, not names that come from upstream.
The registered names in the ledger **are baked into assets** (changing them later would need another
migration), so they will not be changed from now on.

| Item                                          | `int8-sym`                                                       | `int4-sym-g`                                                                                                                   | `int2-off`                                                                   | `ternary`                           |
| --------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- | ----------------------------------- |
| Old `storage.dtype`                           | `i8`                                                             | `i4`                                                                                                                           | `i2`                                                                         | (new)                               |
| Set of quantized values `q`                   | `[−127, +127]` (**−128 is not used**, 255 levels)                | `[−7, +7]` (**−8 is not used**, 15 levels)                                                                                     | `[−2, +1]` (all 4 levels)                                                    | `{−1, 0, +1}` (3 levels)            |
| Stored code `u`                               | `u = q` (signed 8-bit)                                           | `u = q + 8` (`[1,15]`, 0 unused)                                                                                               | `u = q + 2` (`[0,3]`)                                                        | `u = q + 2` (`[1,3]`, **0 unused**) |
| Value kind                                    | `int-sym`                                                        | `int-sym`                                                                                                                      | `int-offset`                                                                 | `int-sym`                           |
| Storage offset                                | 0                                                                | 8                                                                                                                              | 2                                                                            | 2                                   |
| `packing.blockElements`                       | 1                                                                | 8                                                                                                                              | 16                                                                           | 16                                  |
| `packing.blockBytes`                          | 1                                                                | 4                                                                                                                              | 4                                                                            | 4                                   |
| `packing.alignBytes`                          | 4                                                                | 4                                                                                                                              | 4                                                                            | 4                                   |
| Derived bits / element                        | 8                                                                | 4                                                                                                                              | 2                                                                            | 2                                   |
| Pack order (one u32 word)                     | 4 elements, `unpack4xI8(w[i>>2])[i&3]`                           | 8 elements; within a byte, element `2i` = **low** nibble / `2i+1` = **high**                                                   | 16 elements, **from the low 2 bits** in order, `(w[i>>4] >> ((i&15)*2)) & 3` | **exactly identical to `int2-off`** |
| Scale dtype (ledger-wide `SCALE_DTYPES`)      | `f32`                                                            | `f32`                                                                                                                          | `f32`                                                                        | `f32`                               |
| `groupSize`                                   | row length (per-channel)                                         | **power of 2 and ≥ 16** (default 32)                                                                                           | row length (per-channel)                                                     | row length (per-channel)            |
| `rowAxis`                                     | 0 (**1 only for `conv_transpose1d`**)                            | 0                                                                                                                              | 0                                                                            | 0                                   |
| `zeroPoint`                                   | `forbidden`                                                      | `forbidden` (the reservation that reads `offset 8` as "the default when zero_point is omitted" is kept by ADR 0069 decision 3) | `forbidden`                                                                  | `forbidden`                         |
| Additional conditions on the declared shape   | none                                                             | row length `% groupSize == 0`, `numel` even                                                                                    | positive rank 2 `[N,K]`, **`K % 16 == 0`**                                   | same as `int2-off`                  |
| **Reconstruction formula (CPU)**              | `fround(q · s)`                                                  | `fround((u − 8) · s)`                                                                                                          | `fround((u − 2) · s)`                                                        | `fround((u − 2) · s)`               |
| Ops that can stay compressed and resident (③) | linear / embedding / conv1d / conv2d / conv3d / conv_transpose1d | linear / embedding / conv1d (`groups == 1`)                                                                                    | linear / embedding                                                           | same as `int2-off`                  |
| GPU decoding (WGSL)                           | yes                                                              | yes                                                                                                                            | yes (`linearCompute` is `f32` only)                                          | **shares `int2-off`'s**             |

**`int4-sym-g-f16`** (ADR [0145](decisions/0145-gemma4-26b-moe-resident.md) decision 3) is `int4-sym-g`
in every row above but two: its layout is `i4f16` (a decode path of its own) and its scale dtype is
`f16` — one f16 step per group, two per u32 word (step `2i` in the low half, `2i+1` in the high
half, as `unpack2x16float` reads them), the payload padded to a whole word (an odd step count takes
one zero step). Only `expert_linear` keeps it compressed; every other consumer refuses it.

Canonical locations:

- `int8-sym`: value range in `INT8_MAX` (`= 127`, "−128 is not used") in
  `tools/exporter/src/karume/quantize.py`; scale = `clamp(amax / 127, f32 tiny)` in `channel_scale`
  in `quantize.py`; `q = round(w/scale).clamp(−127, 127)` in `quantize_to_int8` in `quantize.py`;
  CPU decoding in `decodeI8` in `packages/runtime/src/format/i8.ts`.
- `int4-sym-g`: value range in `INT4_MAX` (`= 7`) in `quantize.py`; default group length 32 in
  `DEFAULT_GROUP_SIZE` in `quantize.py`; scale = `clamp(amax / 7, f32 tiny)` in `group_scale` in
  `quantize.py`; `q = round(w/scale).clamp(−7, 7)` in `quantize_to_int4` in `quantize.py`; canonical
  pack order in `pack_int4` in `tools/exporter/src/karume/emit.py`; offset constant in `INT4_OFFSET`
  (`= 8`) in `emit.py`; group scale shape in `groupScaleShape` in
  `packages/runtime/src/format/container/codecs.ts` (one function shared by the merge layer, the
  residency planner, construction and CPU decoding); CPU decoding in `decodeI4` in
  `packages/runtime/src/format/i4.ts`.
- `int2-off`: shape condition in `isI2Shape` in `packages/runtime/src/format/i2.ts`; CPU decoding in
  `decodeI2` in `i2.ts`; packing in `pack_int2` in `emit.py` (rejects values outside `[-2,1]`); the
  scale is F32 `[N,1]`, no groups (ADR 0097 addendum 1).
- The declaration rules are split across 2 places. The presence of scale / `groupSize` is checked by
  the descriptor reader `packages/runtime/src/format/container/descriptor.ts` (`parseEncoding`) and
  by the in-memory container, whose input shape differs,
  `packages/runtime/src/format/container/memory.ts`. The range of `groupSize`, row-length
  divisibility, the declared shapes of the i2 family and the scale length are checked by the merge
  layer `packages/runtime/src/format/container/bind.ts` (`bindDeclarations`). The Python mirror is
  `bind_graphs` in `tools/exporter/src/karume/verify.py`, whose `_plan_supply` also checks presence.

**Why `ternary` is a separately named codec** (why it is kept apart although not one value differs
from `int2-off`):

- **The runtime adds 0 lines.** `{−1, 0, +1}` is a **subset** of the range `[−2, +1]` of
  `int2-off`, so packing, the reconstruction formula, WGSL and CPU decoding all reuse those of
  `int2-off` as they are. The only addition is **one absmean quantizer** on the exporter side (**not
  implemented** — step 6 of ADR 0108. The current exporter cannot produce `ternary` assets).
- It is still kept apart **so that assets can be identified**. The reverse direction (reading i2
  assets as ternary) does not hold: counting the 2-bit codes of the first 4 MiB of the I2 tensors of
  gemma4-qat e2b, `q = −2` appears **5.8–7.4 %** of the time (`model.lm_head.weight` 7.447 % /
  `layers.15.mlp.down_proj.weight` 5.766 % / `layers.15.mlp.gate_proj.weight` 7.230 %).
  Silently reinterpreting them would turn around 6 % of all elements into different values.
- Additional condition of a `ternary` declaration: every code of the payload is in `{1, 2, 3}` (an
  occurrence of code 0 fails loudly). This is the checkable content of the claim "it is ternary".

### 6.4 Reserving odd bit widths — bit-plane decomposition

For 3 / 5 / 6 / 7 bits, **only the form of the ledger entry is reserved**. Implementation starts with
whichever width has demand.

Scheme: **put 32 elements into `bits` u32 words as "planes"** (bit `j` of word `b` = bit `b` of
element `j`).

- Storage is **exactly `bits` per element** (discarded bits are 0, so even odd widths do not
  inflate).
- The flat index is always divisible by a power of 2 (32 elements per plane), so both invariants
  hold: "divide the in-word position out of the flat index with shifts" and "a row start lands on a
  word boundary".
- `packing` can be written as `{ blockElements: 32, blockBytes: 4 * bits, alignBytes: 4 }`.
- Unpack cost (**speculation** — unmeasured): per element, `bits` operations `(w_b >> j) & 1` and
  `bits − 1` OR / shift ≈ **3·bits instructions** (9 instructions at 3 bits; about 2× the 4–5
  instructions of the current i4).

**Packings not adopted** (reasons in rejected options 4 / 5 of ADR 0108):

- "Pack n into a u32 word and discard the remainder" — the effective bits inflate (+6.7 % at 3 / 5 /
  6 bits, +14.3 % at 7 bits), and the elements per word are no longer a power of 2, so the index
  computation of the group scale changes from shifts to multiplication and division.
- base-243 (GGUF TQ1_0's 5 trits / byte) — it shrinks **only −15.6 %** relative to 2-bit packing,
  while the 5-element granularity is not a power of 2, which breaks 2 invariants. Row lengths would
  need a `% 5` divisibility condition, and **none** of gemma4's real shapes (1536 / 2048 / 6144 /
  12288) is divisible by 5.

**Lower bit widths are not a reason for speed.** Decode on the RTX 3080 Ti (Chrome, 2026-09-19) is
not bandwidth-bound (measured bandwidth utilisation 11.5 / 25.5 % — recomputed from the table's
columns, 12.1 / 27.5 %; the B570 (the development machine before the swap) and the M2 are unmeasured.
[ADR 0108](decisions/0108-container-format.md) addendum 7); what helps is **memory** (gemma4 e2b
from i4 g32 to ternary 2-bit: storage −40 %; f16 scales alone: −10 %; group 32 → 128: −15 %).
Low-bit codecs presuppose packed int8 activations (ADR 0105).

## 7. Integrity — three separate hash roles

**The descriptor cannot prove its own correctness.** Hence **it is verified first with the outer
expected hash + length**:

1. The reader receives **the expected sha256 and the expected byte length of each of the 2
   documents** from outside (`container.descriptor` of manifest `karume/5`, or the pin that the
   caller of `openContainer` passes as the second argument `expect` (`DescriptorExpectation`))
   (ADR 0109 decision 3 — a different fact from the sha256 of the part 0 file).
2. Fetch part 0 (or the start of the single form), read the header and get the lengths of the 2
   documents.
3. **Check the bytes of the 2 documents against the expected values before parsing the JSON.** Do
   not parse before the check.
4. After parsing, pass all declaration checks of §2 / §5 (**before fetching a single byte of
   weights**).

Runtime integrity is covered by **the sha256 of the descriptor and of the blocks**. However, **block
digests are applied only to unverified sources** (ADR 0108 addendum 2 item 3 / ADR 0109
decision 7):

| Path                                                                      | Number of digests                                                                                      |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| cold (first fetch via HF)                                                 | **once**, by the fetch layer while streaming the whole part (bakes the recorded hash). 0 block digests |
| warm (cache hit)                                                          | **0** — only a string comparison of the recorded hash                                                  |
| unverified source (`openContainer({ kind: "bytes" })`, a local directory) | **a single digest per block** (at most 32 MiB, so it falls on the fast side of §4.1)                   |

A `BlockSource` declares whether it is "verified", and `readBlock` digests only for sources that do
not. Not running digests on warm inherits the current discipline (the NOTE at `BYTE_BUDGET` in
`packages/hub/src/fetch.ts` — no GB-scale digest on a cache hit). **"0" refers to weight blocks and
asset payloads**; step 3 above (checking the 2 documents against the expected values before parsing)
applies on every open, cold or warm (the descriptor is at most 32 MiB, a few hundred KB in real
assets). The fetch layer's sequential sha256 over the whole part on cold is a pure-TS implementation,
so it does not show up in the count of `crypto.subtle.digest` (`tools/ram-peak`, removed in 0.15).
Block sha256 remains as the contract for "verifying only what has arrived" with the Range fetches of
step 6.

**The sha256 of a whole file** (`parts[].sha256`) is separated out **for checking at publication and
repacking**. It is not used at runtime.

**Three separate hash roles** (the same field must not serve more than one):

| Role                                  | What it refers to                                     | When it is decided                       |
| ------------------------------------- | ----------------------------------------------------- | ---------------------------------------- |
| ① **expected hash of a fetched item** | "the bytes about to be fetched should be these"       | **before** fetching; held by the outside |
| ② **derived cache key**               | "input digest + recipe version + conversion settings" | decided **before** generation            |
| ③ **content hash of a derived item**  | "the bytes of the generated result were these"        | computed **after** generation            |

Combining these 3 in one field would make it impossible to express importing upstream weights (a
shape that needs the content hash of something that does not exist yet) and caching the derived
items of load-time quantization.

## 8. Single form and split form

They are **the same byte order**; the only difference is "whether there are cuts".

|                   | Single form                                                                            | Split form                                    |
| ----------------- | -------------------------------------------------------------------------------------- | --------------------------------------------- |
| part 0            | from the start of the file. Header + exactly the 2 documents (no padding)              | a separate file                               |
| part 1 (const)    | right after part 0 (if its length is not 0, after padding with 0x00 to 64 B alignment) | a separate file                               |
| part 2 onward     | right after part 1, concatenated with **64 B alignment**                               | each a separate file                          |
| `parts[].length`  | length of the logical part (interval length after concatenation)                       | file length                                   |
| FileRefs to fetch | 1                                                                                      | 1 descriptor + N parts (held by the manifest) |

- **`parts` lists only indices 1 and above** (writing part 0's sha256 into itself is
  self-referential — §2.2). The integrity of part 0 is covered by **the outer expected hash +
  length** (§7). `container.parts` of manifest `karume/5` holds the FileRefs (`size` / `sha256`) of
  **all parts including part 0**, in index order (ADR 0109 decision 3).
- In split form a zero-length part (part 1 with empty const) is also **written as a 0-byte file**.
  The manifest holds its FileRef with `size: 0`, and the hub does not fetch it (ADR 0109
  decision 3).
- **No padding is inserted before a zero-length part** MUST. Absolute part offsets are derived from
  "declared part lengths + the 64 B alignment rule", so giving a zero-length part padding would make
  the result extracted from a `krm` with empty const differ from a directly written `krg` (§9).
- **Official HF distributions are split form only.**
- **File names** (the convention of the exporter / migration CLI — the manifest points to them with
  FileRefs): single form `<stem>.krm`, split form `<stem>-NNNNN-of-NNNNN.krm` (counting from part 0,
  5-digit zero-padded, the same spelling convention as the old shards), `krg` `<stem>.krg`.
- Placing a single form on HF is not itself forbidden, as a "download asset". **The restriction is
  explained by fetch capability, not by location** (the limit of `openContainer({ kind: "bytes" })`
  in §10).
- A single-form `krm` **contains** a `krg` (it can be extracted as is — §9).
- A split-form `krm` may contain its graph, or may hold in its descriptor only a **content reference
  to `graph`** (`{ sha256, size }`) that points to an external shared `krg`. Where the reference is
  fetched from is **given by the manifest as a FileRef** (the descriptor has no notion of a repo).
  `karume/5` has no field for this — it is added in the step that needs it (step 5) (ADR 0109
  decision 5).

## 9. Byte-copy extraction of `krg`

```text
krg = [header'][graph descriptor (byte copy from krm)][padding][const region (byte copy from krm)]
```

- **Not a single byte of offsets needs rewriting**, because const block offsets are "relative to the
  start of the const region" (§3).
- **Only the header** changes: the magic becomes `KRGC`, the model descriptor length becomes `0`, and
  the graph descriptor length stays as it is. Padding is not copied; it is derived again by the layout
  rule of §8 from the length of the `krg`'s own "header + graph descriptor" (its value is 0x00, so
  there is nothing to copy; with empty const there is no padding either).
- **Only a `krm`** is an extraction source. Before extracting, the reader checks the fetched length of
  the const region against `const.length`, and its content against the model descriptor's
  `parts[0].sha256`. With a `krg` as the source, no document holds the hash of the whole const
  region and it cannot be checked, so the reader rejects it with fail loudly.
- Hence "a `krg` extracted from a `krm`" and "the same graph written as a `krg` from the start" are
  **byte-identical**, and the identity of a `krg` can be decided by its **content hash**.
- What a `krg` **does not hold**: **the weight binding table** (the weight requirements are derived
  from the declared shape / dtype of `graphs[].values`), `provenance`, weight blocks, scale blocks.
  Hence **a `krg` alone cannot build a Session** (`createSessionFromContainer` fails loudly with "no
  weight supply"; graphs with only const supplies and shared declarations are the exception and can be
  built).
- What a `krg` **holds**: `const.constants` (the binding table from const blocks to initializers) and
  `const.length`. Both are inside the graph descriptor and are **the only means by which a `krg`
  alone can supply `const.*`** (this keeps the `encoding` of a const from becoming unreadable from
  anywhere).
- The assets a `krg` can carry are only **weight-independent** ones (tables attached to the const
  region, such as `rope_base`).
- A `krg` is defined as "**holds no weight blocks**", not as "weight-independent". QAT graphs bake
  activation scales into the attr of `static_quantize` (ADR 0097 addendum 2), so even a `krg` can hold
  numbers derived from weights.

## 10. Limits

**Each is held as an independent constant** (none is derived from another).

| Target                                                |                              Limit | Rationale                                                                                                                                                                                                                             |
| ----------------------------------------------------- | ---------------------------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Graph descriptor length                               |                             32 MiB | the measured IR is at most 610,143 B per graph, 9.0 MiB for the 47 deduplicated graphs. Keeps the cost of fetching part 0 in the same order of magnitude as the block limit (**provisional value**)                                   |
| Model descriptor length                               |                             32 MiB | same as above (**provisional value**)                                                                                                                                                                                                 |
| JSON nesting depth                                    |                                 64 | closes the recursion depth of parsing by declaration                                                                                                                                                                                  |
| **Block length**                                      |  **at most 33,554,432 B (32 MiB)** | the 3 grounds of §4.1 (no cost to splitting / the 56 largest weights are exactly this value / const cannot be piece-split)                                                                                                            |
| Block count (one container)                           |                             65,536 | total of the const index and the model index (**provisional value**)                                                                                                                                                                  |
| Part length (part 2 onward)                           | one of `{256, 512, 768, 1024} MiB` | §4.2                                                                                                                                                                                                                                  |
| Part-length ceiling (all parts, parts 0 / 1 included) |         1,073,741,824 B (1024 MiB) | applies the same value as the maximum of the set for parts 2 onward to parts 0 / 1 as well — keeps the formula that estimates buffer sizes from the declaration to a single one                                                       |
| Part count                                            |                               1024 | inherits the value of the old `MAX_SHARDS` (`MAX_PARTS` — §4.2)                                                                                                                                                                       |
| Bytes of one FileRef                                  |                             16 GiB | the current `MAX_FILE_BYTES` (`packages/hub/src/manifest.ts`)                                                                                                                                                                         |
| Number of `graphs`                                    |                                 64 | the largest in real assets is around 4 graphs, for sbv2 (**provisional value**)                                                                                                                                                       |
| **Total size for `openContainer({ kind: "bytes" })`** |                **2,145,386,496 B** | `MAX_SINGLE_CONTAINER_BYTES` (`packages/runtime/src/format/container/limits.ts`). Chromium's single-ArrayBuffer limit (the measurement is recorded in `packages/hub/src/fetch.ts`). **A limit that remains only on this entry point** |

**About the provisional values**: 64 `graphs`, 65,536 blocks and a descriptor length of 32 MiB are
**provisional values set with margin** from measurements (IR at most 610,143 B per graph, 9.0 MiB
for the 47 deduplicated graphs, around 4 graphs for sbv2), not constraints imposed by real assets.
**They shipped as the constants of `limits.ts` without being tightened further.**

**Why the limits are not derived from each other**: the old spec had no notion of a block, and put
the RAM peak, the ArrayBuffer ceiling and the delivery granularity all on the single
`SHARD_BYTE_LIMIT` (ADR 0090 decision 2). With everything on one, when one ground moves, the other 2
move along with it.

**The escape hatch for a const above the block limit is open** (§3 — ① allow piece sequences for
const ② give const alone its own limit ③ treat large constants as assets and fetch them lazily).
Until a real need appears, **the writer stops with fail loudly**.

## 11. Memory contract

**The browser RAM peak is kept to constant + the overlap of fetch units (blocks).**

- The overlap is limited by **both count and total bytes** (e.g. "1 in transfer + 1 being received
  and verified"). Throttling by count alone makes the peak "concurrent count × the largest file at
  that moment" (measured: anima turbo i4 pre-allocated its first 4 files at once, 2.503 GiB in total
  — `BYTE_BUDGET` in `packages/hub/src/fetch.ts`).
- **Decoding is a separate unit of processing.** 1 bit → f32 grows 32× and i2 → f32 16×, so the
  overlap of fetches and the overlap of decoding are not counted in the same budget.
- Only **part 0** is fetched at preparation. Const (part 1) is fetched when it is needed.
- **The fetch unit is the part** (ADR 0109 decision 7 — HTTP Range comes in step 6): on cold, the
  fetch layer streams and verifies the whole part and drops it into the cache (the part is not put on
  the heap). Range reads go per block for seek-type sources (the browser's Blob, local range reads),
  and for scan-type sources (via HF — the fetch layer's strategy stream) the whole part is read once
  and cut into blocks. The slices of the scan type are views of the part buffer held by the hub's
  holding slot, with no copy (`packages/hub/src/container.ts`).
- **Session construction reads one item, uploads it and lets it go** (the batch loop of
  `packages/runtime/src/runtime/session-build.ts`). An item is one block; a whole initializer and
  piece 1 also bring along the scale blocks that ride with them. A split scale (rule ③ of §5) rides
  as **one joined scale**: the reader reads its pieces one at a time, each verified like a block,
  into one buffer of the whole scale's length (`readBlock` of its joined id — ADR 0108
  addendum 15), so the item at piece 1 holds piece 1 + the whole scale, and the join briefly holds
  one scale piece on top. `queue.writeBuffer` copies the bytes
  at the time of the call, so the reference to the item ends within that iteration (it is not held
  until the fence — "GPU transfer" below).
- **The supply reads at most 2 blocks ahead of the item being uploaded, across part boundaries**
  (`READ_AHEAD_BLOCKS` in `packages/runtime/src/runtime/executor.ts` — ADR 0108 addendum 14), so
  the next part is read while the current part's fence is awaited. A source sees up to 3 reads at
  once.

**How it is counted**: host RAM derived from weights during Session construction is counted as the
sum of the following 3 terms. Staging (what `writeBuffer` accumulates) is a separate axis, governed
by "GPU transfer" below.

| Term                       | Seek type                                                                                                                                                                                                                                                               | Scan type                                                                                                                                                           |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ① the caller's holdings    | 0 for `krm`. For the in-memory container (`openMemoryContainer`), the `bytes` and companion scales the caller passed as whole supplies (the container holds them without copying). If the `read()` of `pieces` goes through a source, ③ follows the type of that source | same as the seek type                                                                                                                                               |
| ② the item being processed | 1 block + the riding scale + under decoded residency its f32 decoded result (inversely proportional to the storage bit width — 8× the block for i4, 16× for i2) + for a piece sequence under decoded residency, the copy of the carried-over scale                      | the f32 decoded result under decoded residency + the copy of the carried-over scale (the block and the riding scale are views of ③'s buffer, so they are not added) |
| ③ fetch retention          | the 2 blocks read ahead (each a tight view of its range — at most 2 × 32 MiB for a `krm`; a read ahead can be a joined split scale, whose length is the whole scale)                                                                                                       | 1 hub holding slot (the part length) + buffers waiting for GC; while the read-ahead crosses a part boundary, the parts its blocks fall in (up to 3)                 |
| Closed by the declaration  | yes                                                                                                                                                                                                                                                                     | no (the number of buffers in ③ waiting for GC)                                                                                                                      |

- **The copy of the carried-over scale**: in a piece sequence under decoded residency (the weight is
  decoded to f32 on the CPU and then uploaded), the body of the companion scale exists only in the
  part of piece 1 (rule ③ of §5), so at piece 1 the whole scale is copied as values and carried to
  the last piece of the sequence (`carriedScales` in `session-build.ts`). Holding it as a view would
  keep the scale's buffer (the scale's block for the seek type, the part's holding slot for the scan
  type) alive until the end of the sequence. In exchange, **one extra full copy of the scale lives for
  the duration of the piece sequence**. At piece 1, the view of the riding scale and the copy live at
  the same time (2 scales). In a `krm`, a scale in one block is inside the block limit (32 MiB); a
  split scale is not — the riding scale is the joined scale (120 MiB for Gemma 4 12B's i4 g32
  embedding), and so is the copy. Compressed residency (the weight stays packed and resident) takes
  no copy — the scale is written to the GPU as is at piece 1.
- The estimate **closes by the declaration alone** for the seek type: from `parts[].length` and
  `blocks[].length`, the "maximum concurrent host RAM" can be computed before fetching a single byte
  (ADR 0089's approach). For the scan type, buffers waiting for GC pile up on the largest part, and
  their number does not close from the declaration. What the declaration gives is a lower bound
  (1 holding slot = the largest part length); in the measurement of step 3e the maximum external was
  about 1.5–4.4 times the largest part
  ([research record](research/2026-09-24-part-length-ram-peak.md)).
- "Part" has 2 meanings here (the hub's part and the in-memory container's part). Both are units of
  the fence and set the upper bound of staging. Only the hub's part matters as delivery granularity,
  and what matters for RAM is ③ of the scan type. The host-RAM peak does not depend on how parts are
  split (where staging lives is implementation-dependent, and it was measured only on wgpu / Deno).

### GPU transfer

- **`mappedAtCreation` is not used.** A mapped buffer cannot be released until all blocks have
  arrived and it is unmapped, which runs against the goal of "fetch, verify and release per block".
  `queue.writeBuffer(buffer, dstOffset, …)` is called per block.
- The `writeBuffer` staging of wgpu (Deno) is **not released until the submit completes**
  (measured: after `createBuffer` of 2 GiB 2,311 MiB → after `writeBuffer` 4,359 MiB → 2,306 MiB
  after `queue.submit([])` + waiting for completion —
  `docs/research/2026-08-08-vram-oom-misreport.md:72-78`). Hence a fence of "empty submit +
  `onSubmittedWorkDone`" (ADR 0070 decision 3) is raised **once per part**. The fence governs only
  staging; CPU-side bytes may be let go as soon as `writeBuffer` returns (WebGPU spec — the copy is
  made at the time of the call. Addendum to ADR 0108 decision 9, addendum to ADR 0070 decision 3.
  Pinned on real hardware by `packages/runtime/tests/gpu_write_buffer_copy_test.ts`).
- **errorScope granularity is the block, fence (empty submit + `onSubmittedWorkDone`) granularity is
  the part** (measured in ADR 0108 decision 9: push/pop costs 1.81 µs per call, a fence 13.0 ms per
  call. With only the pop per block and one fence per 256 MiB, it is 0.77×, faster than the current
  equivalent).
- Buffers written with `writeBuffer` are **outside the arena's pool** (because `writeBuffer`
  overtakes earlier unsubmitted encodes in queue order — the MUST at the top of
  `packages/runtime/src/gpu/arena.ts` and `allocHostWritten`). This property does not change with
  blocking.
- **flush-before-destroy** as before.

## 12. Migration CLI contract

Reading the old format lives **only in this CLI**. The readers of the new packages read only
`karume/5` and the new container, and **dual reading is not implemented**.

- **Input**: old single-form safetensors / old shard sequences / old manifests (`karume/4`).
- **Output**: new `krm` (and also `krg` when `--graph` is passed in per-component mode), plus, in
  whole-repository mode, a new manifest (`karume/5`).
- **Old inputs are kept** (the CLI neither deletes nor rewrites its inputs).
- Implemented in Python (`tools/exporter/src/karume/migrate.py`). Reading of old forms lives only in
  the migration CLI (`migrate.py` and `legacy.py`). `legacy.py` reads safetensors shard sequences
  and piece keys; `migrate.py` reads the old `karume/4` manifest and the PLE sidecar / index. The 3
  steps write → re-read and verify → replace go through **the same single path** as the single export
  path (`karume.publish.publish_container`). What the migration CLI adds is only the reading of old
  forms and the checks of assumptions that exist only in the old forms (the dtype and shape of old
  tensors, the dtype and shape of scales, missing / surplus tensors, the output directory being
  empty).
- **Options common to both modes**: `--out <dir>`, `--license <identifier>` (required — no default
  that would falsify the provenance), `--notice <reference>` / `--upstream-revision <revision>` (the
  optional fields of `provenance` — §2.3), `--writer <identification>`,
  `--part-bytes {256,512,768,1024}` (part length in MiB, default 256 — §4.2. Values outside the set
  are rejected).
- **`provenance.writer` is not written by default** (it goes into the container only when `--writer`
  is given explicitly). The tool version is held in one place, the `generator` field of
  `karume.json`. Baking the version into the container would make the model descriptors of a migrated
  container and of a container written directly by a recipe differ forever, by the version.
- **Per-component mode** (`karume migrate <representative path | old single form>… --out <dir>
  --license <identifier> [--graph-name <name>] [--single] [--graph]`): works per component (one
  graph), and neither reads nor writes a manifest. Instead of a representative path, an actual local
  file (`…-00001-of-00002.safetensors`) may be passed. The graph name defaults to the parent directory
  name and can be given explicitly with `--graph-name` (only when there is one positional argument).
  Per the rule of §2.1 it MUST be the key of weights, so where the parent directory name differs from
  the component name, `--graph-name` states the component name. `--single` writes a single-form `krm`
  (split form by default). `--graph` also writes a `krg`. Shard sequences are assembled from the
  representative path, so a shard sequence that crosses directories (sbv2's `shared/front` /
  `shared/voice` have shard 2 in the speaker directory) cannot be assembled — use whole-repository
  mode.
- **Whole-repository mode** (`karume migrate --manifest <karume.json> --out <dir> --license
  <identifier> [--cross-repo <owner/name>=<converted directory>@<40-character revision>]…`): reads the
  old `karume/4`, converts every (model, component, dtype) to `krm`, copies `assets`, and writes the
  `karume.json` of `karume/5` (ADR 0109 decision 8). Shard sequences are assembled from the
  declarations of the old manifest, so sequences that cross directories are resolved here. The graph
  name is the `weights` key itself (§2.1). Sequences of cross-repository references are not
  converted; `container` is copied from the `karume/5` of the converted directory that `--cross-repo`
  points to. At start it checks that the output directory is empty (or absent), assembles the whole
  repository in a staging directory (`karume.artifacts.staged_publication`), and installs it only for
  a run in which all conversions, the copying of regular files and the manifest check have passed (a
  run that fails part-way leaves nothing in the output directory).
- **Paths in the old manifest**: read with the same grammar as the hub's path check (ADR 0038 §2,
  0041 §6 — rejects empty segments, segments starting with a dot and disallowed characters). The write
  destination joins the parent directory of the old path onto the output directory, so letting `../`
  or absolute paths through would write outside the output directory. A declared file that is a
  symlink pointing outside the input repository is rejected as well.
- **Options that cannot be combined across modes** (each fails loudly): `--manifest` together with
  positional arguments. In whole-repository mode, `--graph-name` / `--graph` (the key decides the
  graph name, and `karume/5` has no field for a shared `krg` — ADR 0109 decision 5) and `--single`
  (`container.parts` MUST have at least the 2 elements part 0 + part 1, and official HF distributions
  are split form only — ADR 0109 decision 3, §8). In per-component mode, `--cross-repo`.
- The self-check (invariant 5) compares **the payload part** (the sha256 of a whole block includes
  padding, so it does not match between old and new). Output is written to `.partial` and replaced
  only after the check passes (a failed run leaves nothing).
- Before baking, it checks that the **shape** of the old scale (keepdim / group form) equals the shape
  determined by `rowAxis` / `groupSize` (for square weights, per-column `[1,N]` and per-channel
  `[N,1]` cannot be told apart by byte count).
- Before baking, it checks the **dtype and shape** of old tensors against the IR declaration, by the
  old writer's rules (storage dtype → safetensors dtype is one-to-one, the shape stays the logical
  shape, scales are F32 — `karume.legacy.LEGACY_STORED_DTYPE`). For the PLE sidecar, `values` is
  `[rows, layers, dim]` and `scales` is F32 `[rows, layers]`. Looking only at the byte length would
  silently bake a different type of the same length, or a different shape with only the same element
  count, into the new container.

Invariants:

1. **Raw bytes identical**: the **payload bytes** (`numel × bits / 8` bytes) of each initializer's
   body / scale do not change by a single byte.
   - **Trailing padding is newly baked** (§4.1), so "raw bytes identical" is said of the payload.
     The sha256 of the whole block (padding included) does not match between old and new.
2. **The IR is re-serialized from v1 to v2** (renames and canonical serialization come in, so
   verbatim identity of the IR is not guaranteed).
3. **Codecs map to the existing 3**: `i8 → int8-sym` / `i4 → int4-sym-g` / `i2 → int2-off`. **No
   conversion to `ternary`** (even when the range is a subset, the claim of being ternary is the
   quantizer side's to make).
4. **Deterministic**: the same input produces **byte-identical output** (how parts are packed, how
   blocks are cut and the JSON key order are all determined by the input).
5. **Self-check**: the CLI itself re-reads the written container and checks that the sha256 of each
   block's **payload part** and the sha256 of the body taken from the old shards **match per
   initializer** before installing it (it does not produce something that was written but cannot be
   read — the same approach as `verify.assert_reader_layout`).
6. **Piece-key spellings disappear.** The old `<name>#00002-of-00003` becomes
   `binding[].pieces[].rows`. Physical layout no longer leaks into the namespace. Only the migration
   CLI's reading of old forms (`legacy.py`) interprets the old spelling.

## 13. Differences in IR v2

The differences from IR v1 are **4 points** (canonical in [ir-v2.md](ir-v2.md) — here only the
correspondence with the binding table).

### 13.1 Move `initializers[].storage` out to the binding table, and make the initializer name the key of the body

```jsonc
// IR v1
"initializers": {
  "p_enc_w": { "tensor": "enc.w", "storage": { "dtype": "i4", "scale": "enc.w_scale", "group_size": 32 } }
}

// IR v2 — the name becomes the v1 tensor key (FQN / const.<hash>) itself
"initializers": {
  "enc.w": {}
}
```

| Old                                                  | New location                                                                                                                               |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `initializers[].tensor` (safetensors tensor key)     | becomes **the initializer name itself** (the binding table is looked up by `(graph name, initializer name)`; block ids are internal names) |
| `storage.dtype`                                      | `binding[].encoding.codec`                                                                                                                 |
| `storage.scale` (key of the scale tensor)            | `binding[].encoding.scale.block`                                                                                                           |
| `storage.group_size`                                 | `binding[].encoding.groupSize`                                                                                                             |
| `initializers[].shared` (`{ tensor: <lender key> }`) | `{ "shared": true }` — **borrower's name = lender's initializer name** MUST (the name is the key)                                          |

**Why the name is the key**: in v1 the initializer name was the torch.export placeholder name
(`p_model_layers_0_mlp_gate_proj_weight`), and the upstream key (FQN) was held separately by the
`tensor` field. Removing the `tensor` field would make importing upstream checkpoints (decision 19)
and resolving LoRA targets (step 5) lose their key. Making the name the FQN adds not a single table
(measured: the names and keys of the 77 mirror graphs are all ASCII, at most 87 characters, with 0
collisions). The migration CLI renames by the `tensor` field of v1 (together with the references from
`values` and the nodes).

The aim is "**1 architecture, 1 quantization scheme, 1 graph**". The graph becomes independent of
PTQ presets (storage dtype / group / session knobs) and of swapping the weights of the same config
(fine-tunes). Measured support: in the assets surveyed, the graph difference between PTQ presets was
**only** `initializers[*].storage` (measured on irodori dit f32/f16/i8/i4, anima f16/i8, sbv2 voice).

**QAT stays a separate graph** (it has 485 `static_quantize` nodes and 403 baked activation scales,
and differs from the regular presets even in `requires.ops` / `values` / `outputs` — the graph itself
is part of the scheme).

### 13.2 Unify scales to the rank-2 group form and expose `rowAxis` in the declaration

As in §6.1. The i8 keepdim broadcast form is retired, and only i8 of `conv_transpose1d` declares
`rowAxis: 1`. **Deriving** the axis from the consuming op stops, but `weightChannelAxes` (`plan.ts`)
remains **as a check against the declaration** — when the declaration and the consuming op's axis
disagree, the GPU-resident path would apply the scale to another axis, a silent wrong value
(implemented in `planWeightResidency`).

### 13.3 Merge layer — the rules live in one place

Moving them out must be **reversible**. Before weights are fetched, **the graph + the selected
bindings are merged**, and on the merged representation the following run:

- the full set of storage rules — scale required for quantized codecs / `groupSize` required for
  `int4-sym-g`, a power of 2 ≥ 16 / no groups for `int2-off` / row-length divisibility / payload
  length checked against block length / the piece checks (coverage of the leading dimension, no
  padding in middle pieces, no splitting when `rowAxis != 0`) / the split-scale checks (beside a
  piece sequence only, one scale piece per data piece over the same rows, per-piece lengths)
- the checks, residency plan and estimate of `prepareContainer` (the stage after `mergedGraph`, which
  maps the merged representation to the execution graph)

**The storage rules are split across 2 places inside the runtime's `format/container/`.** The
presence of scale / `groupSize` is checked by the descriptor reader `descriptor.ts` (`parseEncoding`).
The in-memory container (`openMemoryContainer`) has a different input shape, so `memory.ts` checks
the same presence. Value ranges, row-length divisibility, the declared shapes of the i2 family, scale
length, payload length and the piece checks are checked by `bind.ts` — `bindDeclarations`, which
does not depend on the 2 documents, and the outer wrapper `bindGraphs`, which calls it from the 2
documents. The in-memory container also goes through the same `bindDeclarations`. There is one kind
of IR declaration reader, `parseIrDeclaration` / `parseIrDeclarationValue`
(`packages/runtime/src/format/ir.ts`), and host-built graphs also go through
`parseIrDeclarationValue` before they are passed to the in-memory container (the doc in
`packages/runtime/mod.ts`).

**The Python side also passes the merged representation to the existing checks, through its own
mirror of the merge rules** (`bind_graphs` in `tools/exporter/src/karume/verify.py` mirrors TS
`bindGraphs`; `container.py` the codec rules): Python is not present at run time, so the exporter
implements the rules a second time, and cross-language fixtures keep the two in agreement (ADR
[0108](decisions/0108-container-format.md) decision 17).

### 13.4 What does not change

Semantic dtypes (`f32` / `i32` / `bool`), shapes and the dimension language, `states` slots, the form
of `nodes`, the contract of `requires.ops`, the op set, and "unknown top-level keys fail loudly".

### 13.5 Acceptance

**The main acceptance of step 1** consists of 3 checks.

1. **A decompose → merge round trip** reproduces the original declaration, initializer order,
   residency plan and estimate (CPU).
2. **On all 77 graphs, the old and new eligibility predicates give the same result**
   (`eligibleCompressedInitializers` / `i2EligibleInitializers` / `i4EligibleInitializers` in
   `plan.ts`). The predicates are not folded into the ledger (ADR 0108 addendum 12).
3. **Re-exporting the same graph yields a byte-identical graph descriptor.** Since the identity of a
   `krg` is decided by content hash (§9), serialization has to be deterministic down to
   `graphs[name]`. **The IR v2 serialization rules were settled in the "Canonical serialization"
   section of [ir-v2.md](ir-v2.md)** (2026-09-22 — keys in schema order, name-keyed maps in
   code-point order, numbers spelled by ECMAScript `Number::toString`, no whitespace).
   The whole graph descriptor (the key order of `graphs`, the `const` index, the sets of
   `capabilities`) follows the same rules.
   CPU spike ① sorted only the graph names, and the IR body kept the serialization order of the
   Python side — in that state, byte identity of an extracted `krg` and an independently generated
   `krg` held only "when written by the same writer".
