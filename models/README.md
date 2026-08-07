# models/ — index of the distribution forms (nothing but this README is tracked in git)

**`models/` holds only the distribution form** — 1 directory = 1 HF repo, ready to upload as-is.
Right now that's `anima-turbo/` (text-to-image) and `sbv2-FN4/` (text-to-speech). The exporter's raw
output (the source series directories those are assembled from) lives under `outputs/series/` at the
repo root, not here — keeping the assembled, upload-ready tree separate from the raw assets it's
built from. Hand-placed real weights go to a third root, **`inputs/<family>/<name>/`** (they are not
generated, so they are not under `outputs/` either). Demo and benchmark **outputs also go to
`outputs/`** (see below), so nothing but distribution form ever mixes into this tree. The source of
truth for all three spellings is `tools/exporter/karume/paths.py` (`DIST_ROOT` / `SERIES_ROOT` /
`INPUTS_ROOT` / `OUTPUTS_ROOT`).

## Distribution form — `anima-turbo/`

`karume.json` plus files named per the convention in ADR 0038 §2. Only files declared by the
manifest go here — the E2E fixtures (`io.*.safetensors`) present in the series directories are
**not included**. The two metadata files at the top level, `karume.json` and the model card
`README.md` (both written by `karume dist`), are the only exceptions to that check.

```
cd tools/exporter && uv run karume dist
```

Assembly is idempotent (rerunning relinks). Since it's the same filesystem, the actual files are
**hard links** — the series and the distribution form don't double up disk usage.

| Path                                 | Source                                                                                 |
| ------------------------------------ | -------------------------------------------------------------------------------------- |
| `karume.json`                        | **Derived from the actual files** at assembly time (hand-writing forbidden — ADR 0038) |
| `text_encoder/model.safetensors`     | `outputs/series/anima-f16/text_encoder/`                                               |
| `text_conditioner/model.safetensors` | `outputs/series/anima-f16/text_conditioner/`                                           |
| `transformer/model.f16.safetensors`  | `outputs/series/anima-turbo-f16-dyn/transformer/`                                      |
| `transformer/model.i8.safetensors`   | `outputs/series/anima-turbo-i8-dyn/transformer/`                                       |
| `transformer/rope_base.safetensors`  | The two series above (**verified byte-identical, then unified into one**)              |
| `vae_decoder/model.safetensors`      | `outputs/series/anima-f16/vae_decoder/`                                                |
| `tokenizer/qwen2-tokenizer.json`     | `outputs/series/anima-demo/text/`                                                      |
| `tokenizer_2/t5-tokenizer.json`      | `outputs/series/anima-demo/text/`                                                      |

If `rope_base` diverges between series, assembly **stops** (silently picking one would let the
unselected series' preset run with a rope table of different geometry — loading and execution
would both succeed while only the image comes out broken).

## Distribution form — `sbv2-FN4/`

11 files, 504MiB (the manifest is fixed by ADR 0039). The same assembly rules as above apply:
`karume.json` is derived from the actual files, placement is by hard link, and rerunning is
idempotent.

```
cd tools/exporter && uv run karume dist --pipeline sbv2
```

Assembly spans three roots, so all of the following have to be in place first: the real weights
under `inputs/sbv2/FN4/`, the series `outputs/series/sbv2-FN4-f16/` and
`outputs/series/sbv2-FN4-i8/`, the text encoder series `outputs/series/deberta-i8/full-24layer/`,
and the host assets `outputs/sbv2-demo/{symbols.json,deberta-tokenizer.json}`.

| Path                                      | Source                                                              |
| ----------------------------------------- | ------------------------------------------------------------------- |
| `karume.json`                             | **Derived from the actual files** at assembly time (ADR 0038)       |
| `README.md`                               | Model card rendered from the manifest (`karume.modelcard`)          |
| `text_encoder/model.i8.safetensors`       | `outputs/series/deberta-i8/full-24layer/`                           |
| `front/model.f16.safetensors`             | `outputs/series/sbv2-FN4-f16/front/`                                |
| `front/model.i8.safetensors`              | `outputs/series/sbv2-FN4-i8/front/`                                 |
| `voice/model.f16.safetensors`             | `outputs/series/sbv2-FN4-f16/voice/`                                |
| `voice/model.i8.safetensors`              | `outputs/series/sbv2-FN4-i8/voice/`                                 |
| `tokenizer/deberta-tokenizer.json`        | `outputs/sbv2-demo/`                                                |
| `text/symbols.json`                       | `outputs/sbv2-demo/`                                                |
| `styles/style_vectors.safetensors`        | **Converted** from `inputs/sbv2/FN4/style_vectors.npy`              |
| `speakers/speaker_embeddings.safetensors` | **Converted** from `emb_g.weight` of the ckpt in `inputs/sbv2/FN4/` |

Only the 3 graphs that inference needs are distributed: `front`, `voice` (flow + dec fused) and
`text_encoder`. The `dp` / `flow` / `dec` targets in the series exist for the golden E2E only, so
shipping them would be dead weight (ADR 0039 §1). The names that index the rows of `style_vectors` /
`speaker_embeddings` live in `pipelineConfig` of the manifest — the three are one set, and their row
counts are checked against `inputs/sbv2/FN4/config.json` **before anything is placed**, because a
row that slips loads and runs fine and only comes out as a different voice.

## `outputs/` (repo root, not tracked in git)

Where demo and benchmark outputs live (PNG, WAV, dumps, and visual-gate record images), plus the
exporter's raw output: **`outputs/series/`** — the series directories the distribution forms above
are assembled from. The source of truth for regeneration commands is
[tools/exporter/README.md](../tools/exporter/README.md). E2E **SKIP**s a series missing its assets
(partial absence is FAIL), so tests for a deleted series won't run again until re-emitted.

| Directory                                 | Contents                                                                           |
| ----------------------------------------- | ---------------------------------------------------------------------------------- |
| `outputs/series/anima-f16/`               | text_encoder / text_conditioner / VAE decoder (f16) + io reference                 |
| `outputs/series/anima-turbo-f16-dyn/`     | turbo DiT f16 **S-shape** (resolution-independent) + raw rope table + io reference |
| `outputs/series/anima-turbo-i8-dyn/`      | turbo DiT i8 **S-shape** + raw rope table + io reference                           |
| `outputs/series/anima-demo/text/`         | Tokenizer tables (Qwen2 BPE / T5 Unigram)                                          |
| `outputs/series/sbv2-FN4/`                | SBV2's 5 emit targets (dp / front / flow / dec / voice) in f32 + io reference      |
| `outputs/series/sbv2-FN4-f16/`            | the same 5 targets, f16 storage + io reference                                     |
| `outputs/series/sbv2-FN4-i8/`             | the same 5 targets, i8 storage + io reference                                      |
| `outputs/series/deberta/full-24layer/`    | Japanese DeBERTa (f32) + io reference                                              |
| `outputs/series/deberta-i8/full-24layer/` | Japanese DeBERTa (i8) + io reference — the `text_encoder` of `sbv2-FN4/`           |
| `outputs/sbv2-demo/`                      | host assets of the voice demo — **not a series**, hence directly under `outputs/`  |

Only the single S-shape is distributed (fixed shapes are not distributed — ADR 0038 §4). The SBV2
series carry all 5 targets because the golden E2E compares every one of them; only 2 of them reach
`sbv2-FN4/`.

`models/` holds only the assembled distribution form; `outputs/` holds everything the exporter and
demos produce along the way, distributed or not — the split is designed so that deleting `outputs/`
entirely doesn't change `models/`'s regeneration procedure (rerun `karume dist` once the series are
re-emitted).
