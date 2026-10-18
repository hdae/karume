# 0041: Manifest v2 (`karume/2`) — several models per repository and a cleaned-up vocabulary

- Status: accepted (2026-08-08 — structure, vocabulary and the decision against shards ruled by the
  maintainer; the size limits of decision 7 were proposed values and are confirmed as implemented,
  2026-10-18); partly superseded: the format by [0071](0071-manifest-v3-shards.md) (2026-08-20,
  `karume/3`, shard field per dtype entry, which also replaces decision 4's "no shards"),
  [0075](0075-quant-presentation.md) (2026-08-25, `karume/4`) and
  [0109](0109-manifest-v5-container.md) (2026-09-22, `karume/5`, a weights entry is the container
  entry `{ container }`; decision 1 of 0109 now carries "read only the current format"); the graph
  location of decision 4 by [0108](0108-container-format.md) (2026-09-22, effective 2026-09-24; the
  container descriptor). The vocabulary (`models` / `defaultModel` / `quants` / `dtype` / `weights`
  / `assets`), the path rules, the absence of a shared pool, the size limits and the
  independent-copy rule are in force (`packages/hub/src/manifest.ts`).
- Date: 2026-08-08
- Related: ADR [0038](0038-manifest-v1.md) (manifest v1 — replaced by this ADR) /
  [0039](0039-sbv2-distribution.md) (SBV2 distribution — this ADR answers its decision 7 on multiple
  speakers) / [0037](0037-karume-monorepo.md) (parent decision on the distribution form) /
  measurements: [research/2026-08-08-xet-split-probe.md](../research/2026-08-08-xet-split-probe.md)

## Context

Three real needs for different weights of the same architecture came together: (1) publishing the
four official SBV2 JVNV models (F1/F2/M1/M2) on Hugging Face; (2) FN1 to FN10 duplicate the 319 MB
text encoder per model; (3) Anima has third-party derivatives. In v1 one manifest was one Hugging Face
repository was one architecture plus one set of weights; there was no model axis (the repository name
was the only identifier).

A repository groups a loose unit — a family, a series of models or one author (maintainer's ruling:
the range one README and one licence statement can cover, for example `karume-sbv2-jvnv` with four
JVNV models and shared assets). Third-party derivatives naturally get their own family repositories.

Two measured premises: **splitting by layer does not improve downloads** (per-stream speed drops
monotonically and each request adds a fixed TTFB of about 0.67 s; parallelism comes from HTTP Range
without splitting, and four parallel Range requests were fastest in every configuration —
research/2026-08-08-xet-split-probe.md). And only one manifest (`hdae/anima-turbo`) had been published,
so keeping a v1 parser had no real value (maintainer's ruling: "v1 can be dropped").

## Decision

### 1. One root manifest (`karume/2`); no v1 parser

The manifest stays one file, `karume.json`, at the repository root (fixed name), with
`format: "karume/2"`. **hub reads only v2** (no two-format parsing); `hdae/anima-turbo` was
re-uploaded as v2, and old clients (JSR v0.1.0) fail loudly with "unsupported format". (Superseded
by ADR 0109 decision 1 on 2026-09-22: hub reads only the current format `karume/5`; its basis is no
longer "one unreleased distribution", since 0.14.0 is published and the models package pins
revisions — ADR [0073](0073-models-source-pin.md).)

Sub-manifests (`<model>/karume.json`) are not adopted: the manifest location and the path base would
diverge, and models could not be listed by machine.

### 2. Top level: `models` plus `defaultModel` (required)

```json
{
  "format": "karume/2",
  "generator": "…",
  "defaultModel": "jvnv-F1",
  "models": {
    "jvnv-F1": {
      "pipeline": "sbv2/1",
      "weights": {
        "front": { "f16": { "path": "jvnv-F1/front/model.f16.safetensors", "size": 0, "sha256": "…" } },
        "text_encoder": { "i8": { "path": "shared/text_encoder/model.i8.safetensors", "…": 0 } }
      },
      "assets": {
        "tokenizer": { "path": "shared/tokenizer/deberta-tokenizer.json", "…": 0 },
        "style_vectors": { "path": "jvnv-F1/styles/style_vectors.safetensors", "…": 0 }
      },
      "quants": {
        "w8": { "weights": { "front": "i8", "voice": "i8", "text_encoder": "i8" }, "session": {} }
      },
      "defaultQuant": "w8",
      "pipelineConfig": { "…": "…" }
    }
  }
}
```

The example is the form of 2026-08-08: model IDs became `F1`/`F2`/`M1`/`M2` (addendum 2026-08-09),
quant names follow ADR [0074](0074-quant-seat-naming.md), and a weights entry is `{ container }` in
`karume/5` (ADR 0109).

- **`defaultModel` is required.** If a single-model repository could omit it, adding a second model
  would break the implicit default of existing users. `fromPretrained` without a model uses
  `defaultModel`.
- **`pipeline` is per model**, so a family repository can mix architectures.
- A model entry is essentially a v1 manifest without `format` / `generator`.

### 3. Vocabulary: `quants` / `dtype`, `weights` / `assets`

- **`presets` → `quants`** (`defaultPreset` → `defaultQuant`): the choice is in practice a quantization
  / precision mode, which matches how users pick from a GGUF quant table.
- **variant → `dtype`**: since v1 a label maps 1:1 to the storage dtype vocabulary (ADR 0038 §2).
  Separate words for the mode (quant) and the storage form (dtype) remove "a quant selects a variant".
  The label vocabulary (`f16` / `i8` / …) is unchanged.
- **components split into `weights` and `assets`**: v1 mixed `{file}` and `{variants}` components.
  **weights are tensor containers that always carry a dtype key** (an i8-only text encoder is
  `{ "i8": … }`), **assets are unconditional files independent of the quant** (tokenizer / symbols /
  style_vectors / speaker_embeddings …). Two-form parsing disappears, and the check that a quant's
  weights map points only at weights follows from the types.

### 4. No shards; the graph stays in metadata

(Superseded: ADR 0071 introduced a shard field per dtype entry on 2026-08-20, ADR 0108 moved the
graph into the container descriptor and ADR 0109 made the container the weights entry, both on
2026-09-22.) The original decision: splitting by layer was measured to hurt downloads, so there are
no shards, and the graph JSON stays in the safetensors header `__metadata__.karume_ir` (0.085% of
334 MB, no reason to move it). Faster downloads come from HTTP Range parallelism in the fetch layer
(cold 3.1×, warm 84 MB/s, independent of the manifest).

### 5. No shared-pool mechanism (sharing by identical paths)

Several models' weights / assets simply point at the same `shared/…` path. Fetching and caching
deduplicate per URL; a dedicated indirection (a `"shared": …` field) would only duplicate derivable
structure. The manifest grows by a few KB, harmless against the 1 MiB limit.

### 6. Paths are relative to the repository root (as in v1); the allowlist check stays

The manifest is always one file at the root, so the v1 path semantics (root-relative, `SEGMENT_RE`
check) hold as they are.

### 7. Size limits (DoS guard)

The whole manifest ≤ 1 MiB and `pipelineConfig` ≤ 256 KiB per model stay. **`models` ≤ 32**, and per
model **weights ≤ 32, assets ≤ 32, quants ≤ 32**. These values were proposed here and are confirmed as
implemented (`MAX_MANIFEST_BYTES`, `MAX_MODELS`, `MAX_WEIGHTS`, `MAX_ASSETS`, `MAX_QUANTS`,
`MAX_PIPELINE_CONFIG_BYTES` in `packages/hub/src/manifest.ts`); the largest real table is far below
them. Structure depth ≤ 64 was added on 2026-08-12 (addendum).

### 8. Public API shape

```ts
using p = await Sbv2Pipeline.fromPretrained("hdae/karume-sbv2-jvnv", {
  model: "M1",
  quant: "i8",
});
// no model = defaultModel / no quant = defaultQuant
// an unknown model / quant fails loudly and lists what is available (listing became possible in v2)
```

`HubRepoRef` is unchanged (no subpath axis). Selection takes `(manifest, { model, quant })` — today
`resolveSelection` in `packages/hub/src/resolve.ts`.

### 9. Local conventions stay

In `models/`, one directory is still one Hugging Face repository (only its inside gains levels). The
undeclared-file check of `verify_dist` covers per-model subtrees plus `shared/` plus the root
`karume.json` / `README.md`. The exporter gained a `--model` axis and family assembly (a shared file
with the same sha256 is placed once; `tools/export-recipes/dist.py`).

## Consequences

- One breaking change covered the single manifest, the vocabulary and the weights / assets split (JSR
  0.2.0, the three packages in lockstep). The only migration cost was re-uploading anima-turbo.
- Models can be listed by machine, which error messages use (`available`).

## Addendum (2026-08-08) — follow-ups

- Implemented in the order hub parser → exporter → models → re-upload. The follow-ups listed out of
  scope (HTTP Range parallel fetch, two-phase `fetchAssets`, verified-cache marker, family licence
  statement) moved to their own tracks; Range parallel fetch landed with ADR
  [0080](0080-hub-fetch-cache-050.md).

## Addendum (2026-08-09) — rulings at implementation

- **Repository names carry the `karume-` prefix** instead of a Hugging Face organisation:
  `hdae/anima-turbo` → `hdae/karume-anima-turbo` (retired on 2026-09-01; the official Turbo checkpoint
  is now a model of `hdae/karume-anima`, `packages/models/src/anima/config.ts`); SBV2 is `karume-sbv2-jvnv` (published, model IDs
  F1/F2/M1/M2 — the repository name carries the family namespace, defaultModel F1) and `karume-sbv2-fn`
  (not published; no permanent local mirror since 2026-08-30, `docs/assets-layout.md`). Series names
  (`outputs/series/sbv2-<model>-*`) do not take the prefix.
- **A distribution is always made of independent copies; hard links MUST NOT be used.** The series
  writers overwrite files by truncation, so a linked distribution would change silently on re-export and
  the manifest sha256 would no longer match the files (`verify_dist` does not recompute sha256, so the
  mismatch stays silent). A distribution is a self-contained snapshot (`tools/exporter/src/karume/dist.py`,
  `shutil.copyfile`). Writing `models/` directly from the export step is not adopted: it would lose the
  merge of N export outputs into one repository and the rearrangement without re-export.
- **The defaultModel of a family assembly is the first `--model`** (no dedicated flag).

## Addendum (2026-08-12) — structure depth in the size limits

The exhaustive review (H1-W1) found that a deeply nested manifest within 1 MiB overflows the stack in
the recursive forbidden-key walk and leaks a bare `RangeError` out of the `HubError` contract. Decision
7 adds **structure depth ≤ 64** (`MAX_MANIFEST_DEPTH`); exceeding it fails loudly with
`ManifestFormatError`. Real manifests are a few levels deep; 64 is a DoS guard.

## Addendum (2026-10-10) — size of the diagnostics that run before the count checks

The diagnostic walk that collects the available labels for errors (`available`) runs **before** the
table count checks. To keep decision 7's guard effective, the walk stays linear in the input and
truncates each list (`models` / `weights` / `quants`) at that table's limit (32). When it truncates, the
error message says so. The limit is also the largest legitimate count, so a manifest that passes the
checks always gets a complete `available`.
