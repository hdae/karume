# models/ — index of local assets (nothing but this README is tracked in git)

**`models/` holds only the distribution form** — 1 directory = 1 HF repo, ready to upload as-is.
Right now that's `anima-turbo/`. The exporter's raw output (the source series directories
`anima-turbo/` is assembled from) lives under `outputs/series/` at the repo root, not here — keeping
the assembled, upload-ready tree separate from the raw assets it's built from. Demo and benchmark
**outputs also go to `outputs/`** (see below), so nothing but distribution form ever mixes into this
tree.

## Distribution form — `anima-turbo/`

`karume.json` plus files named per the convention in ADR 0038 §2. Only files declared by the
manifest go here — the E2E fixtures (`io.*.safetensors`) present in the series directories are
**not included**.

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

## `outputs/` (repo root, not tracked in git)

Where demo and benchmark outputs live (PNG, WAV, dumps, and visual-gate record images), plus the
exporter's raw output: **`outputs/series/`** — the series directories `anima-turbo/` above is
assembled from. The source of truth for regeneration commands is
[tools/exporter/README.md](../tools/exporter/README.md). E2E **SKIP**s a series missing its assets
(partial absence is FAIL), so tests for a deleted series won't run again until re-emitted.

| Directory                             | Contents                                                                           |
| ------------------------------------- | ---------------------------------------------------------------------------------- |
| `outputs/series/anima-f16/`           | text_encoder / text_conditioner / VAE decoder (f16) + io reference                 |
| `outputs/series/anima-turbo-f16-dyn/` | turbo DiT f16 **S-shape** (resolution-independent) + raw rope table + io reference |
| `outputs/series/anima-turbo-i8-dyn/`  | turbo DiT i8 **S-shape** + raw rope table + io reference                           |
| `outputs/series/anima-demo/text/`     | Tokenizer tables (Qwen2 BPE / T5 Unigram)                                          |

Only the single S-shape is distributed (fixed shapes are not distributed — ADR 0038 §4).

`models/` holds only the assembled distribution form; `outputs/` holds everything the exporter and
demos produce along the way, distributed or not — the split is designed so that deleting `outputs/`
entirely doesn't change `models/`'s regeneration procedure (rerun `karume dist` once the series are
re-emitted).
