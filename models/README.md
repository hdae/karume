# models/ — index of local assets (nothing but this README is tracked in git)

Split into two layers: the **distribution form** (`anima-turbo/` — a single repo that can be
uploaded to HF as-is) and its **source series directories** (the raw output the exporter
produces). Demo and benchmark **outputs go to `outputs/` at the repo root, not `models/`** (don't
mix assets and outputs in the same tree).

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
| `text_encoder/model.safetensors`     | `anima-f16/text_encoder/`                                                              |
| `text_conditioner/model.safetensors` | `anima-f16/text_conditioner/`                                                          |
| `transformer/model.f16.safetensors`  | `anima-turbo-f16-dyn/transformer/`                                                     |
| `transformer/model.i8.safetensors`   | `anima-turbo-i8-dyn/transformer/`                                                      |
| `transformer/rope_base.safetensors`  | The two series above (**verified byte-identical, then unified into one**)              |
| `vae_decoder/model.safetensors`      | `anima-f16/vae_decoder/`                                                               |
| `tokenizer/qwen2-tokenizer.json`     | `anima-demo/text/`                                                                     |
| `tokenizer_2/t5-tokenizer.json`      | `anima-demo/text/`                                                                     |

If `rope_base` diverges between series, assembly **stops** (silently picking one would let the
unselected series' preset run with a rope table of different geometry — loading and execution
would both succeed while only the image comes out broken).

## Series directories (assembly source — do not delete)

The source of truth for regeneration commands is
[tools/exporter/README.md](../tools/exporter/README.md). E2E **SKIP**s a series missing its
assets (partial absence is FAIL), so tests for a deleted series won't run again until re-emitted.

| Directory              | Contents                                                                           |
| ---------------------- | ---------------------------------------------------------------------------------- |
| `anima-f16/`           | text_encoder / text_conditioner / VAE decoder (f16) + io reference                 |
| `anima-turbo-f16-dyn/` | turbo DiT f16 **S-shape** (resolution-independent) + raw rope table + io reference |
| `anima-turbo-i8-dyn/`  | turbo DiT i8 **S-shape** + raw rope table + io reference                           |
| `anima-demo/text/`     | Tokenizer tables (Qwen2 BPE / T5 Unigram)                                          |

Only the single S-shape is distributed (fixed shapes are not distributed — ADR 0038 §4).

## `outputs/` (repo root, not tracked in git)

Where demo and benchmark outputs live. PNG, WAV, dumps, and visual-gate record images go here.
`models/` holds only "input (assets)"; `outputs/` holds "output" — the split is designed so that
deleting one entirely doesn't change the other's regeneration procedure.
