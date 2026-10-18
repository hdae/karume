# 0109: Manifest `karume/5` — the entry point of the container distribution

- Status: accepted (2026-09-22 — the maintainer approved the recommended options 1-1 to 1-12 of the
  step 2 plan of the container-format work, and irodori-v4.1-small as the one repository to
  re-upload); implemented in `@karume/hub`, the distribution format of 0.13.0 and 0.14.0. The
  concurrent-fetch budget at the end of decision 7 was superseded by addendum 2 (2026-09-25).
- Date: 2026-09-22
- Related: [0108](0108-container-format.md) (makes 0108 decision 20 concrete and corrects part of
  it — 0108 addendum 2) / [0038](0038-manifest-v1.md) (version bump rule, FileRef, cross-repository
  pairs) / [0041](0041-manifest-v2.md) / [0071](0071-manifest-v3-shards.md) and
  [0090](0090-shard-spec-v3-tensor-pieces.md) (shards and tensor pieces — retired) /
  [0075](0075-quant-presentation.md) (unchanged) / [0080](0080-hub-fetch-cache-050.md),
  [0086](0086-distribution-source.md), [0089](0089-memory-limits-preflight.md) (inherited) /
  [0085](0085-ple-host-gather.md) (PLE moves into the container) /
  [0128](0128-default-preset-original-weights.md) decision 6 (`defaultQuant` unchanged). Canonical
  spec: [container-v1](../container-v1.md) (physical format) and this ADR (manifest shape).

## Context

`karume/4` had three layers: "component × dtype → shard list (first = graph shard) + extras",
"preset = complete mapping component → dtype label + execution options", and model-level assets.
With the distribution moving from the safetensors dialect to `krm` (ADR 0108), the shard list is
replaced by **the container's entry point** (FileRefs of the descriptor and the parts). ADR 0108
decision 20 put that entry point at `quants[].container`, but an inventory of the 11 real mirrors
(2026-09-22, totals over `models/*/karume.json`) showed:

- Unique component × dtype files total **65.9 GiB**. A preset is "weight mapping + execution
  options", not a weight unit: gemma4's three presets (`i4` / `i4-gemvpar` / `i4-fast`) **share the
  weights** and differ only in execution options; anima's 5 models × 6 presets point at the same
  text_encoder (1.1 GB).
- One physical container per preset would be **197 GiB (×2.99)** (anima ×3.87, sbv2 ×4.66, gemma4
  ×3.00, irodori ×1.22).
- `extras` held **one kind**: anima's `rope_base` (65,968 B). Cross-repository references:
  anima-extra's text_encoder / vae_decoder and two tokenizer files. PLE: 9 shards (2.3 GB) + 1
  index in gemma4's `assets`.
- `openContainer` (step 1) takes a `BlockSource` (part count, part lengths,
  `read(part, offset, length)`) and checks **each of the two documents** against an expected
  length + sha256 before parsing (`DescriptorExpectation`, `packages/runtime/src/format/container/open.ts`).
- The fetch layer (`@hdae/fetch-cache` 0.8.0) has no HTTP Range. Cold reads stream the whole file,
  verify sha256 and record the hash; warm reads compare the recorded hash (ADR 0080). Range reads
  exist only on the warmed cache: scan (cost proportional to the offset) on Deno and Node, seek in
  browsers (`packages/hub/src/sources/hf.ts:244`).

## Decision

### 1. The format is `karume/5`; `karume/4` is not read

The major version is bumped by ADR 0038 decision 1; an old major fails as an unsupported format. No
dual reading (ADR 0108 decision 18, alternative 3).

### 2. Container granularity is **component × dtype**; presets stay mappings

`weights.<component>.<dtype>` holds `{ container }`. `quants[].weights` (complete mapping
component → dtype label) and `session` / `gpuFeatures` / `requiredLimits` / `label` /
`description` stay as in `karume/4`. ADR 0108 decision 20's `quants[].container` is **corrected to
this shape** (0108 addendum 2).

- Reason: one file per preset is ×2.99; the step 1 writer and the migration CLI already wrote this
  granularity. Several graphs per container **remains a format capability**, unused in `karume/5`.
- The dtype label is vocabulary for selection and display; storage is canonical in the
  descriptor's `encoding.codec` (ADR 0071 decision 3 inherited). The hub checks only that every
  component of a preset has a container.

### 3. The `container` field — descriptor expectations + part FileRefs

```json
"container": {
  "descriptor": {
    "graph": { "length": 183422, "sha256": "…" },
    "model": { "length": 40961, "sha256": "…" }
  },
  "parts": [
    { "path": "v4.1-small/dit/model.i8-00001-of-00003.krm", "size": 224407, "sha256": "…" },
    { "path": "v4.1-small/dit/model.i8-00002-of-00003.krm", "size": 0, "sha256": "e3b0c442…" },
    { "path": "v4.1-small/dit/model.i8-00003-of-00003.krm", "size": 268435456, "sha256": "…" }
  ]
}
```

- **`descriptor` gives length + sha256 for each of the two documents** (not the part 0 file's
  sha256). The reader contract (`DescriptorExpectation`) stays, and the `graph` document's sha256
  serves as `krg` identity (ADR 0108 decision 4).
- **`parts` lists every part including part 0** in index order (container-v1 §8). The first is
  part 0 = the descriptor file, `size ≥ 24` (header length); at least two entries (part 0 + part 1).
- **A zero-length part (part 1 with an empty const region) is written as a 0-byte file, and
  `size: 0` is allowed only in `container.parts`** (sha256 = that of the empty string). The hub does
  not fetch it (`BlockSource.partLength` answers 0, so no read happens). The general FileRef rule
  `size > 0` stays (FileRef parsing in `packages/hub/src/manifest.ts`).
- Limits are **kept independently**: part count ≤ 1024, one part ≤ 1024 MiB (container-v1 §10;
  `MAX_PARTS` / `MAX_PART_BYTES` in `packages/hub/src/manifest.ts`, Python counterparts in
  `tools/exporter/src/karume/container.py`). `MAX_SHARD_BYTES` and its exemptions retire.
- **Cross-repository references are per container**: parsing checks that every element of `parts`
  has the same `repo` / `revision` pair (or none, all local). One-sided or mixed references fail
  loudly.
- The file name convention (`<stem>-NNNNN-of-NNNNN.krm` — container-v1 §8) is the writer's; the hub
  does not check it (as with `karume/4` shards).

### 4. `assets` stays; PLE and extras move into the container

- **Model-level `assets` (preset-independent FileRefs) stay** — tokenizers, symbols, style vectors
  and speaker embeddings are shared by several containers and referenced across repositories;
  folding them in would duplicate them.
- **The PLE sidecar becomes assets of the model container**: `values` and `scales` are re-cut
  separately into blocks of at most 32 MiB **on row multiples**, as block lists with roles
  `ple-values` / `ple-scales` (asset names `ple.values.<k>` / `ple.scales.<k>`) plus an index with
  role `ple-index` (schema 3 — row bytes and token range per block). Range-read blocks are **one
  block per part** (container-v1 §4.2); whole-read assets (index, `rope_base`) may share a part with
  each other, never with weight blocks. The reader translates token → (block, row offset).
- Assets have no shape, so **`assets[].length` (payload bytes) is declared** (block length = rounded
  up to a multiple of 4). Consumers do not guess and strip trailing 0x00.
- **`extras` retires**; `rope_base` becomes a container asset (role `rope-base`, 66 KB × 2).

### 5. No shared-`krg` field in `karume/5`

Each `krm` embeds its graph descriptor (77 graph JSONs total 14.2 MB — duplication is cheap). The
sha256 of the descriptor's `graph` document gives identity, so a field that distributes a shared
`krg` (decision 20's `graph: FileRef`) is **added when needed (LoRA, step 5)**.

### 6. The hub checks only what declarations alone decide

Parsing checks counts, ceilings, the `size: 0` rule, the length of part 0 (= 24 + both document
lengths), uniform cross-repository references and descriptor length (≤ 32 MiB); part lengths have no
multiple-of-64 rule (64 B alignment is a block-offset rule). **Descriptor / parts consistency, the
block index and the codec ledger are checked by `openContainer`**
(`packages/runtime/src/format/container/open.ts`), not by the hub — one check point, not two.

Resolution returns **a structured selection** (component → container reference, asset name →
FileRef): `resolveSelection`, with the flat FileRef list for prefetch, inventory and progress derived
from it by `selectionRefs` (`packages/hub/src/resolve.ts`). The `karume/4` fetch-key convention
(`<weights>[i]` / `<weights>.<extra>`, and the index reconstruction on the models side) retires.

### 7. Fetching

The hub has one fetch surface that returns a `BlockSource` per container.

- **Cold**: phase 1 (`prefetchFile` — stream the whole file, verify sha256, record the hash) warms
  the part, then range reads open. The part is not held in RAM.
- **Warm**: recorded-hash comparison only (ADR 0080); zero digests.
- **Branch on range-read cost**: seek (browser Blob, local positional reads) reads per block; scan
  (Deno and Node via HF — `packages/hub/src/sources/hf.ts:244`) reads the part once and cuts it into
  blocks (reading the blocks of one part in order would make the skipping quadratic).
- **"Verified" is carried along**: bytes whose whole file the fetch layer verified (via HF) are
  declared verified by the `BlockSource`, and `openContainer`'s `readBlock` digests blocks **only
  for unverified sources** (the `{ kind: "bytes" }` input of `openContainer`, local directories —
  the hub does not check whole-file sha256 there, ADR 0086 decision 2). This avoids double digests
  on cold reads (warming + block). Container-v1 §7 was corrected to this shape.
- **HTTP Range stays at step 6.** The trigger to bring it forward is "the RAM-peak harness shows a
  cold peak above 'part length + overlap'". The fetch layer (`@hdae/fetch-cache`) is not changed.
- Progress is counted per FileRef (= part). The concurrent-fetch budget "count 2, total = largest
  selected part × 2" (replacing `BYTE_BUDGET` = 1.5 GiB) **was superseded by addendum 2 on
  2026-09-25** (4 streamed warm-ups, at most one held part; container-v1 §11).

### 8. Step 2 writer — the whole-repository mode of the migration CLI

History (2026-09-22): `karume migrate --manifest <karume.json> --out <dir>` converted `karume/4`
repositories to `krm` + `karume/5`; recipes and `karume dist` write `krm` directly since step 3
(ADR 0108 addendum 4, item 1).

### 9. Coexistence

History (2026-09-22): from step 2b the hub read only `karume/5`; local mirrors were migrated in 2d;
the remaining pins moved at the release (addendum 1, item 2).

### 10. Relation to component replacement

ADR 0108 decision 19's `fromPretrained(source, { components: { <role>: ComponentSource } })` takes
this ADR's `container` field as is — a `ComponentSource` is "(model, preset, component) of another
`karume/5` repository" (the type is `{ source, model?, quant? }` — ADR 0108 addendum 3, item 5).
**Before any weight byte is fetched**, admission checks, listing every failure: ① the `graph`
document's sha256 equals the base's expectation, ② no missing / surplus bindings (computed by the
merge in `openContainer` from the descriptor alone), ③ the preset's execution options agree with the
codecs.

## Consequences

- **Breaking change** (major bump; hub public surface: the resolution return type and the new fetch
  surface; the `StreamedAsset` shard surface retired in step 3d). The CHANGELOG lists it under
  Breaking.
- Retired contracts: the meaning of shard order (ADR 0070 decision 3, 0081 decision 1), tensor
  pieces (0090), and the Python ↔ TS `SHARD_BYTE_LIMIT` correspondence.
- The model card (`modelcard.py`) and the upload script walk `*.krm`. Acceptance: ADR 0108 step 2.

## Addendum 1 (2026-09-24) — graph-name rule and timing of the pin migration

The manifest-side points of step 3 (3a–3d). The container side is ADR
[0108](0108-container-format.md) addendum 4; the rule text is [container-v1](../container-v1.md)
§2.1.

1. **Container graph name = component name = the `weights` key of `karume.json`** MUST (adds to
   decision 2).
   - **Why a rule**: readers look up the graph by name (`prepareContainer(opened, <weights key>)`).
     The manifest points at one single-graph container per component × dtype, so the weights key is
     the only name leading from the manifest to the graph. Another spelling would make migrated
     mirrors and re-exported series differ while both still parse, and the user's load would be the
     first to fail ("graph not in container").
   - **Directory names are not part of the rule** (siglip2's series directory has key `vision`,
     irodori's `caption-proj` key `caption_proj`, deberta's `full-24layer` key `text_encoder`). The
     export path requires `graph_name` without a default; recipes name the component by a constant.
   - **Check points**: `karume dist` compares container graph names with the preset's weights keys
     before assembly (`DistError`); the AST check `tools/export-recipes/tests/test_graph_names.py`
     pins that recipes use a constant that maps to a weights key of the distribution plan (and owns
     the exceptional families).
2. **Correction of decision 9 — the remaining pins move at the release, not in step 3** (only
   irodori-v4.1-small pointed at `karume/5` before; the other 9 repositories ran only from old
   packages until the release). The order of
   [release-runbook](../release-runbook.md) §0 (version bump → rebuild → upload → pin → JSR publish)
   is a non-commutative MUST: the manifest's `generator` field copies the package version, so an
   upload before the bump publishes a distribution naming an old version. A re-upload deletes old
   files the new manifest no longer references from the same repository and pins the `main` after
   the deletion (HF history is kept — ADR 0108 decision 18). The procedure lives in the runbook.
3. History: decision 8's direct `krm` writing landed in steps 3a / 3b, and the `StreamedAsset` shard
   surface was removed in step 3d.

## Addendum 2 (2026-09-25) — decision 7's concurrent-fetch budget and the real name of `fromContainer(bytes)`

The end of decision 7 (step 2 default = count 2, total = largest part × 2) was not adopted. From
step 2f the implementation is as follows; counting rules are canonical in
[container-v1](../container-v1.md) §11.

1. **Warming (phase 1) streams 4 files** (`CONCURRENCY` in `packages/hub/src/fetch.ts` —
   `prefetchAssets`). The body flows straight into the cache with no pre-allocated receive buffer,
   so the count does not determine RAM.
2. **Scan-type range reads hold one part in a holding slot** (`packages/hub/src/container.ts`);
   seek-type sources hold only block views. This is stricter than decision 7's "count 2".
3. **`BYTE_BUDGET` (1.5 GiB) remains only on the whole-file surface `fetchAssets`** (manifest
   `assets`; it allocates a file's receive buffer before receiving). Container parts do not pass it.
   The value is kept (the maintainer's ruling of 2026-09-25). The measurement cited in `fetch.ts`
   (2.503 GiB for the first 4 anima turbo i4 files) dates from `karume/4` shards; the remaining basis
   is keeping one file's pre-allocation plus the verification copy below the single-ArrayBuffer
   limit.
4. History: `fromContainer(bytes)` is the `{ kind: "bytes" }` input of `openContainer`
   (`ContainerInput` in `packages/runtime/src/format/container/open.ts`); the decisions above use
   the real name.
