# 0154: Anima keeps its upstream bf16 weights, and its presets are named from the bf16 base

- Status: accepted (2026-11-18 — the maintainer's decision; the export recipe carries it, and the TypeScript side, the
  GPU checks, the rebuilt mirrors and the publication follow)
- Date: 2026-11-18
- Related: [0074](0074-quant-seat-naming.md) (the preset-name grammar; decision 6 and Consequences: no compatibility
  shim) / [0131](0131-bf16-weight-execution.md) (bf16 storage, Anima's original-weights preset `bf16`) /
  [0128](0128-default-preset-original-weights.md) decision 4 (original weights are the upstream storage) /
  [0075](0075-quant-presentation.md) (`label` / `description`) / [0087](0087-anima-official-extra-repos.md) (the official
  and extra repositories) / [0149](0149-irodori-rare-token-split.md) decision 5 (a build that takes the published
  containers instead of the series) / [0147](0147-numerics-and-kernel-adoption.md) decision 3 (2) (the review of a
  preset before publication) / [0073](0073-models-source-pin.md) (revision pins)

## Context

Every upstream component of Anima is BF16: the text encoder, the text conditioner and the VAE of the base checkpoint
(310, 118 and 194 tensors, headers read), and the transformer of each of the five official checkpoints on Hugging Face
(685 BF16 tensors each, headers read on 2026-11-18). The two community fine-tunes of the extra repository are not on
disk; the headers of their upstream files, read the same day, hold 685 BF16 tensors each, and in their f16 containers
99.65–99.96 % of the 16-bit words have the low three mantissa bits zero (a true F16 file: 13 %).

The distribution nevertheless stored the text stack (text encoder, text conditioner, VAE decoder) in f16 for six of its
seven presets, and the transformer in f16 for two of them. Only the preset `bf16` (ADR 0131 decision 3) kept the upstream
bits. f16 storage of a BF16 value is a rounding step Karume adds: f16 has more mantissa bits but fewer exponent bits, so
values in f16's subnormal range move. The presets were named by ADR 0074 decision 6 from that f16 base:

| Preset                  | text stack | transformer | session                          |
| ----------------------- | ---------- | ----------- | -------------------------------- |
| `f16`                   | f16        | f16         | —                                |
| `f16+dit8`              | f16        | i8          | —                                |
| `f16+dit8-a8`           | f16        | i8          | linear a8                        |
| `f16+dit8-a8-attn8`     | f16        | i8          | linear and attention a8          |
| `f16+dit8-a8-attn8-s16` | f16        | i8          | as above, f16 attention scores   |
| `f16-c16`               | f16        | f16         | linear and attention f16 compute |
| `bf16`                  | bf16       | bf16        | —                                |

The default was `f16+dit8-a8-attn8-s16`. The recipe also carries `f16+dit4` / `f16+dit4-attn8-s16`, which no model ships
(the visual-evaluation assembly puts them back locally).

The f16 files are 20.64 GiB of `karume-anima` and 7.79 GiB of `karume-anima-extra`. The published revision of
`karume-anima` already holds every bf16 text-stack file and every bf16 and i8 transformer file, and the series the
distributions were built from are no longer on disk.

## Decision

1. **Every float-stored component of the seven models is bf16, the upstream bits.** The f16 containers leave both
   repositories. The transformer keeps its i8 series beside the bf16 one.
2. **The preset names follow ADR 0074 as written**: `<storage>` is the base storage of every role, now `bf16`.

   | Before                        | After                          |
   | ----------------------------- | ------------------------------ |
   | `f16+dit8`                    | `bf16+dit8`                    |
   | `f16+dit8-a8`                 | `bf16+dit8-a8`                 |
   | `f16+dit8-a8-attn8`           | `bf16+dit8-a8-attn8`           |
   | `f16+dit8-a8-attn8-s16` (default) | `bf16+dit8-a8-attn8-s16` (default) |
   | `f16-c16`                     | `bf16-c16`                     |
   | `f16+dit4` / `f16+dit4-attn8-s16` (not shipped) | `bf16+dit4` / `bf16+dit4-attn8-s16` |
   | `f16`                         | dropped                        |
   | `bf16`                        | `bf16`                         |

   The default stays the same selection under its new name: the i8 transformer with int8 linear and attention
   activations and f16 attention scores, now with the bf16 text stack.
3. **`f16` is dropped.** With a bf16 transformer it is the preset `bf16`: the same weights and an empty session. A
   second name for the same selection is the flaw ADR 0074 removed ("names that contradict the content").
4. **`f16-c16` becomes `bf16-c16`**: the original weights, with the transformer computed in f16 (the `-c16` knob reaches
   the transformer only). It keeps the shader-f16 speed option without keeping f16 files.
5. **No alias and no compatibility shim** for the old names (ADR 0074 decision 6). A client pinned to an older revision
   keeps reading the old manifest and its f16 files; a caller that follows `main` and names an old preset gets the
   unknown-preset error, which lists the available names.
6. **The distributions are rebuilt from the published containers, not re-exported.** The recipe takes every container of
   the seven models, and the two tokenizer files, from the local mirrors of the two repositories, and pins each by the
   sha256 of its part 0 (of the file, for a tokenizer) as published — part 0's descriptor declares the sha256 of every
   other part. Before a byte is written, the build hashes every part 0 and tokenizer file against its pin, and every
   other part of a container the extra repository borrows from the official one (the text encoder and the VAE decoder):
   a borrowed container is only referenced, so nothing compares it later. The other parts of a container a repository
   stores are compared with part 0's declaration as the assembly places them in its staging directory, which a mismatch
   discards before anything is put in place. The published bytes are pinned instead of re-exported, as for Irodori
   (ADR 0149 decision 5); a stored part is hashed once, as it is placed. Re-exporting a model's containers is a
   deliberate change to that table.
7. **A single-file checkpoint whose tensors are F16 stays loadable under a float preset.** Its f16 plan is derived from
   the distributed bf16 container, not from an f16 container kept for that purpose.

## Alternatives

- **Keep the f16 presets beside `bf16`.** Not chosen: they round the text stack and the VAE for no gain on a BF16
  upstream, they keep 28.4 GiB of files in the two repositories, and they keep the re-export of the f16 containers that
  predate the corrected f16 rounding rule.
- **Keep `f16` as a second name of `bf16`.** Not chosen: two names for one selection (decision 3), and the hub's cache
  inventory cannot tell two selections with the same files apart (evicting one evicts both — `inventory.ts`).
- **Keep `f16-c16` as an all-f16 preset.** f16 compute rounds the transformer's weights to f16 tiles anyway, so the
  transformer would lose little. Not chosen: the text stack and the VAE would still be rounded to f16 although the
  preset asks for no quantization, and the f16 files and their re-export would stay. Dropping the preset loses the
  shader-f16 speed option (its measurement on other devices is still open — perf-ledger K-68). A notation for a mixed
  float base (bf16 text stack, f16 transformer) would need a new grammar: `+dit16` cannot tell f16 from bf16, and ADR 0131
  rejected a 16-bit float mix for the same reason.
- **Re-export the seven models from upstream.** Not chosen: it needs the upstream files of five models (network and the
  maintainer's approval) and gives nothing the published containers do not already hold — their bf16 containers are the
  upstream bits, and an i8 export on another machine may round differently.
- **An alias table in the hub for renamed presets.** Not chosen: the hub has no such mechanism, and ADR 0074 decided
  against a shim for the same reason as here — a period where both spellings work misleads more than one break that the
  pin absorbs.

## Consequences

- **Breaking, at the next publication of the two repositories**: every preset name of the seven models changes and
  `f16` is gone. A caller that omits `quant` gets `bf16+dit8-a8-attn8-s16`, whose text stack is now bf16, so its images
  change; the sha256 reference rows of Anima move with it (11 of the 13 rows; the two `bf16` rows stay).
- The f16 files leave `main` of both Hugging Face repositories when they are uploaded; earlier revisions keep them, so
  the pinned clients up to 0.22.0 keep working.
- `bf16-c16` runs bf16 storage with f16 compute, which has no GPU numerics check yet; it is checked before publication,
  together with the e2e lanes, the A/B bands and the fusion counts. The renamed default is reviewed by a person before
  publication (ADR 0147 decision 3 (2)), and so is the new notice of both repositories, which no longer names an f16
  series.
- The item that re-exports the f16 containers made before the corrected f16 rounding rule loses its Anima part.
- The unpublished 2.9B Anima distribution takes these names at its first publication, so it is renamed only once.
- Families whose upstream is F32 (Irodori, Wan, Lucida) are not touched: whether bf16 storage avoids a loss there is not
  known. MOSS-Image and the Gemma 4 E2B drafter have BF16 upstreams and ship only as int8. bf16 presets for any of the
  five are probably not added: they run well quantized, and bf16 weights of their size are too heavy for a browser.
