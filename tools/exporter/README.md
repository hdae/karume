# Karume exporter

Python tooling that lowers `torch.export`-ed models into Karume's **IR v1**
([../../docs/ir-v1.md](../../docs/ir-v1.md)). Managed with uv; CPU-only torch (no GPU required).

The distribution form is a single safetensors file — tensors (weights and constants) plus the graph
JSON under the `__metadata__` key `karume_ir`.

## Setup

```sh
uv sync            # run in tools/exporter/ (pulls CPU torch from the pytorch-cpu index)
```

`torchvision` is a **base** dependency, not an optional group: `torchvision::deform_conv2d` (the
layer-1' op of ADR [0055](../../docs/decisions/0055-deform-conv2d.md)) is registered by importing
the package, so both the handler key (`torch.ops.torchvision.deform_conv2d.default`) and the tiny
golden that covers the op need it. Making it conditional would split environments into "rejects
`deform_conv2d` as an unknown op" and "accepts it".

`karume/custom_ops.py` registers this project's own `karume::` operators (`gru_scan` /
`gru_scan_reverse` — ADR [0056](../../docs/decisions/0056-gru-scan.md)) with
`torch.library.custom_op`, so **importing it has a process-wide side effect**. `convert.py` imports
it at the top for the same reason it imports `torchvision`: the handler key
(`torch.ops.karume.gru_scan.default`) cannot be written otherwise. Hiding the body behind
`register_fake` is what keeps the time axis symbolic — tracing never enters the Python loop, so the
op survives `run_decompositions` as a single node instead of unrolling T times.

## CLI (`karume`)

The subcommands installed by `[project.scripts]`. **The CLI does not interpret arguments** —
everything after the subcommand name is passed straight to the body (`karume <subcommand> --help`
prints the usage of the body's own parser). This shape keeps a copy of the exclusivity rules
(`--verify` × `--target` etc.) out of the CLI; dispatch is a lazy import.

| Subcommand                     | Wrapped body                                                                    | Former invocation                 |
| ------------------------------ | ------------------------------------------------------------------------------- | --------------------------------- |
| `karume export`                | script `export_anima.py` (the 4 emit targets of ADR 0016)                       | `python export_anima.py`          |
| `karume export-sbv2`           | script `export_sbv2.py` (the 5 emit targets of ADR 0013)                        | `python export_sbv2.py`           |
| `karume export-embeddinggemma` | script `export_embeddinggemma.py` (the sentence-embedding series)               | `python export_embeddinggemma.py` |
| `karume export-irodori`        | script `export_irodori.py` (the 6 text-side graphs of the TTS chain)            | `python export_irodori.py`        |
| `karume export-dacvae`         | script `export_dacvae.py` (the 2 DACVAE codec graphs)                           | `python export_dacvae.py`         |
| `karume export-deberta`        | script `export_deberta.py` (the real-weight DeBERTa-v2 series)                  | `python export_deberta.py`        |
| `karume export-siglip2`        | script `export_siglip2.py` (the SigLIP2 vision tower series)                    | `python export_siglip2.py`        |
| `karume export-birefnet`       | script `export_birefnet.py` (the BiRefNet_HR background-removal series)         | `python export_birefnet.py`       |
| `karume export-depth-anything` | script `export_depth_anything.py` (the Depth Anything V2 relative-depth series) | `python export_depth_anything.py` |
| `karume export-vowel-detector` | script `export_vowel_detector.py` (the vowel-detector CRNN, one graph)          | `python export_vowel_detector.py` |
| `karume dist`                  | `karume.dist` (assembles the distribution form; arguments fully compatible)     | `python -m karume.dist`           |
| `karume verify`                | `karume.verify` (validates the distribution form against every IR v1 rule)      | (new)                             |

Which script runs is spelled in the **subcommand name**, never in a flag: `karume export --pipeline
sbv2` would mean the CLI reads one argument of its own, and the no-copy rule above does not survive
being true "except for one flag". Bare `karume export` stays Anima's spelling.

```sh
uv run karume dist --series ../../outputs/series
uv run karume dist --pipeline sbv2 --card-profile fn     # model FN4 -> models/karume-sbv2-FN4/
uv run karume dist --pipeline sbv2 --card-profile jvnv --model F1 --model F2 \
    --out ../../models/karume-sbv2-jvnv
uv run karume verify ../../models/karume-anima-turbo/anima-turbo/transformer/model.f16.safetensors
```

The scripts are **outside** the package, so they are not in the wheel — every `karume export…`
subcommand only runs in a repository working tree (when absent they spell out where the script
belongs and fail loudly). The bodies that live in the package (`dist` / `verify`) have no such
restriction. Scripts that need extra dependencies keep taking them from `uv run --with …`, exactly
as in the recipes further down.

`--model` names the model to assemble (it moves the series it reads, the subtree it writes and the
key it declares in the manifest). Repeating it assembles **one repository holding several models**
(ADR 0041): the first one given becomes `defaultModel`, and files two models produce byte for byte
identically are placed **once** under `shared/`, referenced by the same path from both. A family
repository's name cannot be derived from the model list, so `--out` is required there.

The layout inside a distribution is uniform — `<model>/…` subtrees plus `shared/`, with only
`karume.json` and `README.md` at the root — and a single-model repository follows the same rule
(otherwise adding a second model would move every existing path).

`karume dist` writes a **model card `README.md`** after assembly and `verify_dist`
(`karume.modelcard` — including the ADR 0037 §3 frontmatter), from a template per pipeline. The
model list, the numbers, the file list, the quant table and the style / speaker tables are derived
mechanically from the manifest; the only constants a template carries are the facts the manifest
does not record (base model, license, provenance of the fused LoRA). `README.md` is a metadata file
on par with `karume.json`, so it is exempt from the undeclared-file check.

**Attribution is a separate axis from the template**: `--card-profile` picks which upstream family
the card credits (source repository, source directories and version, license terms, citations).
SBV2 ships two profiles — `fn` and `jvnv` — and naming one is **required** there, because a silent
default would keep the previous family's attribution on the next family's repository, where every
table and snippet still reads correctly and only the credit is wrong. Anima has a single profile,
so the flag may be omitted; the moment a second one exists, the same rule starts demanding it.

## Verification commands (all of them, after any change)

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Regenerating the golden fixtures

```sh
uv run python -m karume.goldens          # default output: ../../packages/runtime/tests/fixtures/golden/
uv run python -m karume.goldens --out /tmp/golden   # change the output directory
```

The seed is fixed (`karume.goldens.SEED`), so the same environment produces **byte-identical** files
(`tests/test_goldens.py::TestDeterminism` regenerates and compares). Generation fails when the
models do not, in aggregate, cover every op in the contract table (EMITTABLE_OPS) — add an op, add a
golden, is the implementation contract (ADR 0005).

### Golden layout

```
packages/runtime/tests/fixtures/golden/<model>/model.safetensors   weights/constants + __metadata__.karume_ir
packages/runtime/tests/fixtures/golden/<model>/io.safetensors      input tensors and expected outputs from torch CPU
```

Tensor key naming convention in `io.safetensors`:

| Key            | Content                                                       |
| -------------- | ------------------------------------------------------------- |
| `input.<name>` | `<name>` is the **graph input name** (`graph.inputs[].name`). |
| `output.<i>`   | `<i>` is the **position in `graph.outputs`** (0-based).       |

Symbolic dimension bindings are not stored separately — they come from the dimension positions of
the input shapes (the same binding rule as IR v1: for a dimension that appears with coefficient 1
and offset 0, such as `"T"`, its actual length is the bound value). The current goldens bake every
symbolic dimension to `GOLDEN_T`.

The storage dtype in `io.safetensors` is the **concrete representation of the semantic dtype**
(boundary normalization of ADR 0009): f32 → `F32` / i32 (including torch i64) → `I32` / bool → `U32`
as 0 / 1. Out-of-range i64 fails loudly (`convert.normalize_boundary_tensor`).

Current models and coverage:

| Model                      | Symbolic dim | IR ops exercised                                                                                                                                                       |
| -------------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `unary_chain`              | none         | neg, abs, sqrt, log, exp                                                                                                                                               |
| `activations`              | none         | tanh, sigmoid, relu, gelu, gelu_tanh, sin                                                                                                                              |
| `broadcast_binary`         | `T`          | add, sub, mul, div (right-aligned broadcast + lifted constants)                                                                                                        |
| `mlp`                      | `T`          | matmul, add, relu (rank-2 MLP through weight initializers)                                                                                                             |
| `row_reduce`               | `T`          | sum, amax, amin                                                                                                                                                        |
| `mask_chain`               | `T`          | mul(i32), cast, bitwise_not (has a bool output)                                                                                                                        |
| `int_cast`                 | `T`          | cast(f32→i32 truncating), sub(i32), mul(i32) (i32 output)                                                                                                              |
| `layout_chain`             | `T`          | permute(3-cycle), reshape ×3 (chain of aliases + coefficient dim 4T)                                                                                                   |
| `expand_mask`              | `T`          | expand(bool / i32), cast, mul, bitwise_not                                                                                                                             |
| `batch_matmul`             | `T`          | bmm(B/M/K/N all distinct lengths), permute (rank-3 batched matmul)                                                                                                     |
| `gather_last_dim`          | `T`          | gather(last dim / indices from an i32 input), sum                                                                                                                      |
| `attention_block`          | `T`          | linear ×4, softmax ×2, layer_norm, bmm, permute, reshape, add                                                                                                          |
| `fused_attention`          | none         | attention (one node preserved from SDPA; B/H/M/N/D all distinct, last query row has logits around −190)                                                                |
| `embedding_lookup`         | `T`          | embedding(padding_idx=0 is inactive in forward), sum                                                                                                                   |
| `masked_scores`            | `T`          | masked_fill(−3.4e38 broadcast / 0 same-shape), softmax, cast, bitwise_not                                                                                              |
| `runtime_masked_attention` | none         | safe_softmax, where(bool graph input → 0/−inf), add, bmm, mul, permute, reshape, expand — the ADR 0044 chain, with one query row fully masked                          |
| `conv_block`               | `T`          | conv1d(kernel 3 / stride 1 / padding 1), permute                                                                                                                       |
| `dilated_conv`             | `T`          | conv1d(depthwise g=C, dilation 1/3/9 / intermediate groups / residual), leaky_relu, add                                                                                |
| `conv_transpose`           | `T`          | conv_transpose1d(up 2 and up 8 / asymmetric channels), conv1d(no bias), tanh                                                                                           |
| `symbolic_table`           | `T`          | sym_prefix_slice(i32 on 2 axes / f32 on 1 axis), gather, add (Tmax folding)                                                                                            |
| `scalar_operands`          | `T`          | add, sub, mul, div, cast (scalar promotion + reversed `1 − mask` used as a weight)                                                                                     |
| `spline_pieces`            | `T`          | ge_scalar, le_scalar, gt_scalar, ge, bitwise_and, cumsum, sum(bool→i32), clamp, exp, log1p, where, reshape                                                             |
| `coupling_split`           | `T`          | slice(split decomposition + slicing after pad), cat, flip(axis length 3), pad, tanh, mul                                                                               |
| `decoder_tail`             | `T`          | leaky_relu(slope 0.1 and the default 0.01), expand(f32), conv1d, tanh, mul                                                                                             |
| `deform_conv2d_block`      | none         | deform_conv2d (DCNv2 — offsets reaching outside the input plane, modulator in [0,2], k=3×2 with asymmetric padding and a k=1 branch with no bias)                      |
| `gru_scan_block`           | `T`          | gru_scan / gru_scan_reverse (both directions over a symbolic time axis; the forward branch starts from a folded zero `h0`, the reverse one from a graph input), linear |
| `bilinear_resize`          | none         | upsample_bilinear2d (non-integer upscale 4×5→7×9, shrink 4×5→2×3, and a height-1 input whose H scale is 0)                                                             |
| `i8_weights`               | `T`          | **i8 storage** for linear, conv1d, conv2d, conv_transpose1d, embedding (all 5 `WEIGHT_SLOTS` ops), tanh                                                                |

The second output of `attention_block` is a **softmax over large negative values (−205..−180)**, the
regime where a naive form (a softmax that does not subtract amax) has `exp` collapse to 0 in f32 and
yields `0/0 = NaN`. This golden turns red the moment safe-softmax comes off. The mask of
`masked_scores` makes one row **fully masked**, so it also exercises the softmax of a row whose
elements are all −3.4e38 (a uniform distribution).

`dilated_conv` / `conv_transpose` cover the conv-family extension of ADR 0015. The former exercises
depthwise (`groups = C`), intermediate groups (`1 < g < C`) and dilation 1/3/9 in a single path; the
latter makes the **channel counts asymmetric at every stage** (5 → 3 → 2 → 1) — reading the
`conv_transpose1d` weight `[Cin, Cout, K]` as `[Cout, Cin, K]` still gives a matching element count
when Cin == Cout, so both the shape check and the golden would let it pass (recon §4). The final
`Conv1d(..., bias=False)` is the only golden that goes through the exporter's **zero-bias
synthesis** (normalization to arity 3).

`spline_pieces` runs the numerical ops of wave 3 through one path **in the same order as the sdp
spline** (in-interval test → cumsum over interval boundaries → searchsorted-free `sum(x[…,None] >=
bl)` = a row sum over bools → the where for leaving the interval and the log1p of softplus). The
input must be a sequence that **straddles** both ±TAIL and the clamp bounds (if every element lands
on one side, only one branch of the where is taken). `coupling_split` is shaped so that the order
split → transform one side only → cat → flip shows up in the values, with **6 channels** (flip over
an axis of length 3 — reversing 2ch makes off-by-one errors cancel symmetrically). `decoder_tail`
mixes **two leaky_relu slopes** (0.1 and torch's default 0.01 with the positional argument omitted)
into one graph, so a design that does not carry the slope in attrs is caught by the golden instead
of silently getting one of them wrong.

The point of `symbolic_table` is baking at **T (= 6) < Tmax (= 24)** — the mistake of building the
read stride from the post-binding shape only agrees when T = Tmax, so only a golden with a shorter
actual length is a detector. The `1 − mask` of `scalar_operands` is non-commutative, and putting the
constant on the right (a sign flip that turns it into `mask − 1`) shows up in the values here — for
which the **result must flow downstream as a value** (with mask ∈ {0,1}, `(1 − mask) · mask` is
identically 0 in either order, giving a vacuously true expectation). The generator refuses to write
out an output that "has more than one element yet every element is the same value"
(`_assert_not_trivial`).

`i8_weights` is the **only compressed-storage golden** (`GoldenSpec.weight_dtype = "i8"` — at
generation time `fake_quant_int8` is applied **before both** the export and the expectation
capture). Two points matter:

- **Neither the row length nor the total element count of a weight is a multiple of 4** (`[7,5]` /
  `[3,5]` / `[5,3,3]` / `[5,2,3]` / `[3,2,3,1]`). i8 packs 4 elements into one u32, so the mistake
  of "deriving word and lane from the row-relative index" **only agrees by accident when the row
  length is a multiple of 4** (isomorphic to the f16 parity trap — ADR 0019).
- **One row of the embedding is all zeros** (a channel with `amax == 0`). If the lower clamp on the
  scale comes off, it becomes `0/0 = NaN`, and this single row is what turns the golden red.

`conv_transpose1d` is in the mix because it is the only op whose per-channel axis is **1** (the
transposed `[Cin,Cout,K]` layout) — mixing up the axis table does not show up in the values for the
other 4 ops.

## Real-weight DeBERTa export and E2E (M1-P2 wave 5)

Where the tiny goldens take on "op contract coverage", `export_deberta.py` takes on **numerical
agreement on real weights and real token sequences**. The target is the very BERT the SBV2 text
front uses (HF `ku-nlp/deberta-v2-large-japanese-char-wwm`).

```sh
# 1. generate (transformers fetches the weights and tokenizer from HF; ~1.3GB download on the first run)
cd tools/exporter
uv run --with 'transformers==5.14.1' python export_deberta.py            # 2 layers + 24 layers
uv run --with 'transformers==5.14.1' python export_deberta.py --layers 2 # 2 layers only (development)

# 1b. i8 series (ADR 0019 storage + ADR 0025 w8a8 mirror goldens)
uv run --with 'transformers==5.14.1' python export_deberta.py --dtype i8 --act-quant

# 2. real-GPU comparison (every case SKIPs when the assets are absent)
cd ../.. && deno test -A packages/runtime/tests/e2e_deberta_test.ts packages/runtime/tests/e2e_deberta_w8a8_test.ts
```

- **transformers is pinned to 5.14.1** (recon §6-5 — the graph shape changes when the modeling code
  changes). It is not added to `pyproject.toml` / `uv.lock`; `--with` brings it in temporarily.
- The outputs go to **`outputs/series/deberta/<variant>/`** (kept out of commits by the `outputs/`
  entry in the top-level `.gitignore` — the 24-layer weights are 1.3GB). `--dtype i8` is a
  **separate series** `outputs/series/deberta-i8/<variant>/`.
- **`sbv2-22layer` is what the SBV2 distribution ships.** SBV2 only ever reads
  `hidden_states[-3]` (= index 22 = the output of layer 21), so the last two layers are dead weight;
  the truncated model's final output is **bit-identical** to the 24-layer model's `hidden_states[-3]`
  (measured — see `docs/research/2026-08-11-deberta-size-recon.md`). That variant also emits **only
  that one tensor** instead of the whole `hidden_states` tuple, because the runtime reads every
  `graph.output` back on every run. The other variants keep the full tuple so the per-layer error
  growth stays readable off the goldens (ADR 0026).

```
outputs/series/deberta/dev-2layer/model.safetensors      2 layers (130 nodes / 208MB)
outputs/series/deberta/dev-2layer/io.<case>.safetensors
outputs/series/deberta/full-24layer/model.safetensors    24 layers (1230 nodes / 1.32GB / 25 outputs)
outputs/series/deberta/full-24layer/io.<case>.safetensors

outputs/series/deberta-i8/sbv2-22layer/model.safetensors      22 layers in i8 storage (1130 nodes /
                                                              294.5MB / 1 output) — shipped
outputs/series/deberta-i8/full-24layer/model.safetensors      24 layers in i8 storage (319MB)
outputs/series/deberta-i8/<variant>/io.<case>.safetensors       w8 goldens (activations in f32)
outputs/series/deberta-i8/<variant>/io-i8a8.<case>.safetensors  w8a8 mirror (--act-quant)
```

The io tensor key naming is the same as the tiny goldens (`input.<graph input name>` /
`output.<position>`). The only difference is that a single model has **several io files, one per
case**. There are 4 cases:

| Case     | Content                             | T  |
| -------- | ----------------------------------- | -- |
| `case0`  | short sentence                      | 11 |
| `case1`  | longer                              | 26 |
| `case2`  | long sentence with symbols          | 35 |
| `padded` | `case0` + `[PAD]`×5 (0 in the mask) | 16 |

The wrapper is `forward(input_ids, attention_mask, c2p_pos, p2c_pos) -> hidden_states` (a tuple of
all layers, or the single last one for the shipped variant). Because every layer can be compared,
**how the error grows with depth** can be read directly off the goldens (which puts the tolerances on
a measured footing). `padded` is the only case that mixes in `attention_mask=0` and therefore
exercises the mask path (mul → cast → bitwise_not → masked_fill, plus the zero fill on the conv
path).

`c2p_pos` / `p2c_pos` are the disentangled-attention gather indices. They depend only on T, so the
exporter used to constant-fold them into two `[1,512,512]` i32 tensors (2MiB of dead weight); they
are now **graph inputs** built on the host (`karume/patch_deberta.py` is the reference
implementation, mirrored by `packages/models/src/sbv2/text/rel-pos-tables.ts`). The mirror is pinned
byte-for-byte by `packages/models/tests/sbv2_rel_pos_parity_test.ts`.

On the Deno side: `packages/runtime/tests/e2e_deberta_test.ts` (one case = one test). **If not a
single asset is present, everything SKIPs** (the generation command is printed in the warning); this
is independent of the ADR 0005 "all-SKIP is an explicit FAIL" gate
(`packages/runtime/tests/gpu_gate_test.ts`) — the gate only looks at whether a GPU adapter exists,
and the tiny-golden real-GPU tests run even without the assets. When the assets are **partially**
present (only one variant / a missing case) it is a FAIL, not a SKIP. The tolerance is a dedicated
value matched to the error accumulation of 24 layers and is separate from the tiny goldens'
`GOLDEN_TOLERANCE` (the derivation is authoritative in the `DEBERTA_TOLERANCE` comment in
`packages/runtime/tests/e2e_deberta_test.ts`). **The f32 and i8 series run through the same
structure, with the tolerance alone derived from measurements per series** (no reuse across series).

The `io-i8a8.<case>` files written by `--act-quant` are the **w8a8** (`linearCompute: "i8a8"`)
mirror, used by `packages/runtime/tests/e2e_deberta_w8a8_test.ts`. The regular `io.<case>` MUST be
taken **without the hook** (taking it with the hook still applied would contaminate the w8-side E2E
expectations with activation quantization). The prefix is kept apart from `io.` so that the
Deno-side enumeration of regular cases (startsWith `io.`) does not pick up the mirror. The w8a8 E2E
is **not a numerical parity net** (activation quantization is discontinuous, so after a few layers
the GPU and torch become "different samples of the same distribution") — the design of its detection
power is authoritative in the comment at the top of that test.

## Real-weight SBV2 export and E2E (M1-P3 wave 1: dp / wave 6: front / wave 7: flow, dec, voice)

Real weights for the acoustic chain. All 5 emit targets of ADR
[0013](../../docs/decisions/0013-sbv2-chain-export.md) are **in place** (a green `voice` E2E = the
whole SBV2 chain holds):

| Target  | Contents                                       | sym_max | Notes                                                                                 |
| ------- | ---------------------------------------------- | ------- | ------------------------------------------------------------------------------------- |
| `dp`    | DurationPredictor alone                        | 512 (P) | Wave 1. Needs neither the patch layer nor vocabulary growth (the end-to-end scaffold) |
| `front` | enc_p + dp + sdp(reverse) fused into one graph | 512 (P) | Wave 6. **Patch layer required** (see "Patch layer" below)                            |
| `flow`  | reverse of TransformerCouplingBlock            | 4096(T) | Wave 7. Relative-position tables promoted to **graph inputs**                         |
| `dec`   | HiFi-GAN Generator                             | 4096(T) | Wave 7. No patches; `remove_weight_norm` is the only preprocessing                    |
| `voice` | flow + dec fused into one graph                | 4096(T) | Wave 7. **Once this passes, the whole chain is in place**                             |

### Dependencies

```sh
cd tools/exporter
uv sync --group sbv2   # style-bert-vits2==2.5.0 / huggingface-hub
```

- **`style-bert-vits2` is pinned with `==`.** The exporters of the later waves are built on
  monkeypatching package internals (class attributes of the modeling code), so a minor update that
  renames a replacement target or changes the shape of a forward silently changes the graph itself.
  The reason is also left as a comment on that line in `pyproject.toml`.
- It is **not** part of the default `uv sync` (so that the base deps alone keep the tiny goldens and
  pytest running). `transformers==5.14.1` comes in as a transitive dependency of this group.
- **Watch the build dependencies**: the transitive dependency `pyopenjtalk-dict` has no wheel and is
  built from an sdist, so it **requires a C / C++ compiler plus cmake and make**. Without them, `uv
  sync --group sbv2` fails with a cmake error (the dp export itself never uses this package, but
  dependency resolution cannot be skipped).

### Obtaining the weights

**The real weights are not part of the repository.** The default speaker is `FN4` of HF
`rufflet17/voice_models`, whose `FN/FN4/` directory carries `FN4.safetensors` / `config.json` /
`style_vectors.npy` / `style_settings.json` (the same content is also packed in `zip/FN_sbv2.zip`,
but the unpacked directory can be fetched file by file, so the zip is unnecessary). The publisher
declares the model free to modify. Place the 3 files the exporter reads under `inputs/sbv2/FN4/` —
the default `--model-dir` (`style_settings.json` is not read):

```
inputs/sbv2/FN4/config.json         HyperParameters (`version` drives the JP-Extra decision)
inputs/sbv2/FN4/FN4.safetensors     ckpt (**exactly one** `*.safetensors` directly under this dir)
inputs/sbv2/FN4/style_vectors.npy   style vectors (used for the front's style_vec)
```

`FN4` is a Style-Bert-VITS2 JP-Extra model (`version: 2.6.1-JP-Extra`) with `n_speakers: 1`, a
`sampling_rate` of 44100, and 4 styles (`data.style2id` = Neutral / high / low / NSFW).

The ckpt is required to be the **unique** match of `<model-dir>/*.safetensors` (with several, which
one was read would change silently), and the glob is non-recursive. Nothing generated is written
next to it: the exports go to a **separate root**, `outputs/series/`. Another speaker or another
model therefore needs nothing but `--model-dir` pointing at its directory — the series name is
derived from that directory's name (`inputs/sbv2/FN4/` → `outputs/series/sbv2-FN4/`), so two
speakers cannot silently overwrite each other's assets.

### Generation and comparison

```sh
# 1. generate (about 40 seconds for all 5, including loading the 251MB-class weights)
cd tools/exporter
uv run --group sbv2 python export_sbv2.py                # all targets
uv run --group sbv2 python export_sbv2.py --target front # one target only
uv run --group sbv2 python export_sbv2.py --dtype f16    # f16 series → outputs/series/sbv2-FN4-f16/
uv run --group sbv2 python export_sbv2.py --dtype i8     # i8 series  → outputs/series/sbv2-FN4-i8/

# 2. eager equivalence against the reference implementation (**one target per process**; see "Patch layer" below)
uv run --group sbv2 python export_sbv2.py --verify front
uv run --group sbv2 python export_sbv2.py --verify flow
uv run --group sbv2 python export_sbv2.py --verify dec    # before/after remove_weight_norm
uv run --group sbv2 python export_sbv2.py --verify voice

# 3. real-GPU comparison (every case SKIPs when the assets are absent)
cd ../.. && deno test -A packages/runtime/tests/e2e_sbv2_test.ts packages/models/tests/sbv2_relattn_parity_test.ts
```

```
outputs/series/sbv2-FN4/dp/model.safetensors       IR   17 nodes /  12 initializers /   1.78MB
outputs/series/sbv2-FN4/front/model.safetensors    IR  911 nodes / 263 initializers /  33.4MB (2.1MB of baked tables)
outputs/series/sbv2-FN4/flow/model.safetensors     IR 1589 nodes / 458 initializers / 158.9MB (0.15MB of baked tables)
outputs/series/sbv2-FN4/dec/model.safetensors      IR  246 nodes / 197 initializers /  58.7MB
outputs/series/sbv2-FN4/voice/model.safetensors    IR 1836 nodes / 655 initializers / 217.6MB
outputs/series/sbv2-FN4/<target>/io.<case>.safetensors  inputs and expected torch CPU outputs
```

#### Storage dtype series (`--dtype f16` / `--dtype i8` — ADR 0018 / 0019)

`--dtype f16` / `--dtype i8` each write to a **separate series**,
`outputs/series/sbv2-FN4-f16/<target>/` / `outputs/series/sbv2-FN4-i8/<target>/` (keeping them next
to the f32 series would silently apply the f32 tolerance of the existing E2E to compressed assets).
The rounding (fake-quant) uses the shared `quantize.round_weights_to_f16` /
`quantize.fake_quant_int8` and is applied to the modules of each exported target **after
`remove_weight_norm` and the patches, and before the reference and golden capture**. i8 is **not a
vehicle for w8a8** (in all 5 SBV2 targets conv1d is 86–90% and linear is effectively 0 GFLOP — ADR
0025 decision ⑤) — the aim is asset size and load time, with the computation staying f32 (w8a32).

| Target  | f32 storage | f16 storage | Ratio | i8 storage | Ratio | Eligible (compressed resident) |
| ------- | ----------- | ----------- | ----: | ---------- | ----: | ------------------------------ |
| `dp`    | 1.78MB      | 0.90MB      | 50.4% | 0.46MB     | 25.7% | 4 tensors / 0.44MB             |
| `front` | 33.39MB     | 17.96MB     | 53.8% | 10.33MB    | 30.9% | 80 tensors / 7.72MB            |
| `flow`  | 158.86MB    | 79.92MB     | 50.3% | 40.65MB    | 25.6% | 156 tensors / 39.47MB          |
| `dec`   | 58.71MB     | 29.43MB     | 50.1% | 14.84MB    | 25.3% | 98 tensors / 14.64MB           |
| `voice` | 217.60MB    | 109.37MB    | 50.3% | 55.53MB    | 25.5% | 254 tensors / 54.11MB          |

(The eligible counts are the same across all 3 series. The byte figures are from the i8 series — the
set of compression candidates is determined solely by the `WEIGHT_SLOTS` weight slots and does not
depend on the storage dtype.) The ratio is higher for `front` alone because the baked
relative-position tables (2.1MB of i32 / f32 constants) are ineligible for weight slots and stay
f32. The totals are f32 470.34MB → f16 237.57MB (50.5%) → **i8 121.81MB (25.9%)**, and the i8
companion scales are 505,576 B (0.42% of the compressed bytes).

MUST: `--dtype` is **emit-only** (like `--sym-max`). The CLI rejects combining it with `--verify` —
verification is an eager comparison that does not look at the storage format, and for dec / voice
the rounding can only be applied **after** `remove_weight_norm` while the reference is taken
**before** the removal, so combining them would compare "the rounded side vs the unrounded side" and
break the `bit_exact` claim.

The dp wrapper is `forward(h, x_mask, g) -> logw`. The plain `DurationPredictor.forward` has `g` as
`Optional` and therefore a branch, so it is made mandatory to remove the branch, and the input names
are aligned with the names used in the recon (IR input names come straight from the forward argument
names). The dynamic axis is `Dim("P", min=2, max=512)` — **sym_max is per target: 512 for the front
family (P = number of phonemes) and 4096 for flow/dec/voice (T = number of frames)**. Nothing
enforces it mechanically and a wrong value stays silent, so the default lives in the script and
`--sym-max` only has to be spelled out when deviating from it (ADR 0013).

The front wrapper is `forward(x, x_mask, tone, language, bert, style_vec, g, z_noise) -> (logw_sdp,
logw_dp, m_p, logs_p)`. `x_mask` is an **external input** (the original implementation derives it
internally from `x_lengths`, but using a length "as a value" does not fit into the convolution), and
the randomness of the sdp reverse is likewise promoted to the external input `z_noise` (multiplying
by `noise_scale` is done on the host — runtime knobs are not baked into the graph). Mixing in
`sdp_ratio` and turning the result into durations is host-side as well.

The flow / voice wrapper is `forward(z_p, y_mask, g, idx_k, valid) -> z / audio`, and dec is the
plain `Generator.forward(x, g) -> audio [1,1,512T]` (no wrapper). `idx_k` / `valid` are the `(T,T)`
tables of relative-position attention and, **unlike front, are graph inputs** (see "Patch layer"
below).

The golden cases are **5 shared by every target** (so that the Deno side can check equality across
targets from a single table). The dp `h`, the front `x` / `tone` / `bert`, and the flow / dec `z_p`
/ `x` are randn with a fixed seed per length, `g` is the **real speaker embedding weight (`emb_g`)**
and `style_vec` is taken from the **real asset (`style_vectors.npy`)** (with synthetic randomness
the value ranges would not correspond to production and the tolerances would have no footing).

| Case     | Length | Content                                                                          |
| -------- | ------ | -------------------------------------------------------------------------------- |
| `p2`     | 2      | lower bound (the smallest value that avoids `torch.export`'s 0/1 specialization) |
| `p37`    | 37     | short                                                                            |
| `p203`   | 203    | medium                                                                           |
| `p512`   | 512    | exactly the declared upper bound of the front family                             |
| `padded` | 16     | last 5 columns masked to 0 (the only detector for the mask path of front / flow) |

- The `p<n>` in the case names comes from **P (number of phonemes)** in the front family, but in the
  flow family it means **T (number of frames)**. The names are left as they are, prioritizing one
  shared table across all targets (i.e. no target may skip a case).
- **No golden reaches the declared upper bound 4096 of the flow family.** The tables and attention
  scores are O(T²) and the dec output is 512·T, so at T=4096 a single io case would be 134MB with
  2.1e6 output points — not worth it as a golden asset nor as a real-GPU test. Coverage near the
  bound is instead shifted onto "no implementation depends on the declared upper bound", checked
  through the spread of lengths (2 → 512, a factor of 256).

The `padded` input **has values in the padded columns too** — if the mask multiplication works, the
tail of the output is exactly 0, and if it comes off, values leak. Without this shape it is not a
detector for the mask path (`tests/test_export_sbv2.py` goes as far as pinning down that "replacing
the mask with all ones makes the tail non-zero", closing off vacuous truth).

There are 2 asserts pinned at load time:

- **The relative-position attention window size == 4** (for both `enc_p` and `flow`). The index
  tables of the gather patch bake in `clamp(rel + 4, 0, 8)`, so a different window size would
  **silently read an embedding of a different width** (the element count still matches, so it is the
  "silent wrong value" class rather than a shape error). Worse, the goldens would be generated with
  the same mistake, so a numerical comparison would pass as well. Failing the moment the ckpt is
  swapped is the minimal remedy, so it lives in the loader (ADR 0013).
- **No weight_norm-derived parameters remain in `enc_p` / `sdp` / `dp` / `flow`.** If any remain,
  `weight` is no longer the effective weight and writing it out as-is produces a different model
  (measured 0 occurrences, but this is the side that does nothing on the assumption they are absent,
  so it is pinned with fail loudly). **`dec` is not on this list** — it ships with weight_norm
  active (95 modules / 190 parameters), and removal is done by `ensure_dec_plain` immediately before
  export.

`ensure_dec_plain` (idempotent) runs dec through `remove_weight_norm` and then applies the asserts
above. There is an ordering constraint that **weight rounding (`--dtype f16` / `--dtype i8`) is
applied to the effective weights after the removal** (rounding before the removal would round
`weight_g` / `weight_v`, leaving the effective weights off the rounding lattice; with i8 the
discarded elements feed into amax and shift the whole per-channel scale), and it is written as a
MUST in the docstrings of `ensure_dec_plain` and `_fake_quant` (ADR 0013 / 0018 / 0019).

On the Deno side: `packages/runtime/tests/e2e_sbv2_test.ts` (one case = one test) and
`packages/models/tests/sbv2_relattn_parity_test.ts` (byte equality of the tables). Same two-stage
structure as DeBERTa: **if `outputs/series/sbv2-FN4/` contains not a single target directory,
everything SKIPs** (this is the environment where only the raw weights are in place and export has
not been run yet), and when they are **partially** present (a missing target / a missing case) it is
a FAIL. It is **parameterized by series (f32 / f16 / i8)**, and the tolerances are **derived from
measurements per series × target** (no reuse across series — re-deriving one would silently move the
other):

| Target  | f32 atol | f32 rtol | Dominant check | f32 measured maxAbs | f16 atol | f16 rtol | f16 measured maxAbs | i8 atol | i8 rtol | i8 measured maxAbs |
| ------- | -------- | -------- | -------------- | ------------------- | -------- | -------- | ------------------- | ------- | ------- | ------------------ |
| `dp`    | 1e-6     | 1e-5     | rtol           | 2.62e-6             | 1e-6     | 1e-5     | 3.58e-6             | 1e-6    | 2e-5    | 2.38e-6            |
| `front` | 1e-4     | 1e-5     | atol           | 1.75e-5             | 3e-4     | 1e-5     | 3.62e-5             | 2e-4    | 1e-5    | 2.37e-5            |
| `flow`  | 2e-5     | 1e-6     | atol           | 2.74e-6             | 2e-5     | 1e-6     | 2.15e-6             | 2e-5    | 1e-6    | 2.62e-6            |
| `dec`   | 3e-5     | 1e-6     | atol (alone)   | 4.00e-6             | 2e-5     | 1e-6     | 2.37e-6             | 5e-5    | 1e-6    | 6.07e-6            |
| `voice` | 1e-5     | 1e-6     | atol (alone)   | 1.60e-6             | 1.5e-5   | 1e-6     | 1.71e-6             | 1.5e-5  | 1e-6    | 1.74e-6            |

In the f16 series the only significant increase is the front's `logw_sdp` (1.75e-5 → 3.62e-5). The
other 4 targets stay in the same order of magnitude as the f32 series, which corroborates that the
rounded weights also went into the goldens (i.e. the quantization error does not enter the
difference) — had the rounding been forgotten, the difference would have turned into the weights'
relative 5e-4 class and shown up 3 orders higher. The i8 series has the same structure, with all 5
targets landing in the same order of magnitude as the f32 series (front at 1.4× and dec at 1.5×
being the largest — a forgotten rounding would push the per-channel quantization error of the 4e-3
class up by an order of magnitude). Which check dominates is the same per target in every series (it
is determined by the shape of the value range, so the storage dtype does not move it).

rtol cannot be the dominant check for flow / dec / voice because the outputs (the latent z and the
waveform) span a range that crosses 0, where the smallest non-zero `|ref|` drops to the 1e-8 class —
the relative error diverges there (dec's measured maxRel reaches 0.44, but the absolute error of
that element is of the 1e-8 class). The derivations are authoritative in the `SBV2_*_TOLERANCE`
comments in the same file. **The final `tanh` of dec is WGSL-implementation dependent and does not
match torch bit for bit**, so this comparison can in principle only hold with a tolerance.

On the pytest side: `tests/test_export_sbv2.py` (the script's contract) and
`tests/test_patch_sbv2.py` (unit tests for the patch layer). Building the golden inputs and the CLI
exclusivity need no real weights and always run; the export body runs **only in an environment where
the real weights and the `sbv2` group are both present** (otherwise SKIP).

### Patch layer (`karume/patch_sbv2.py`)

front and flow / voice cannot be exported unmodified (dec does not need it). The patches work by
replacing attributes of already-imported classes (monkeypatch) plus wrappers; the `style_bert_vits2`
package itself is left untouched.

| Patch                                      | What it does                                          | Why it is needed                                                                                                                                 |
| ------------------------------------------ | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Branch-free, non-destructive spline        | boolean-mask indexing → clamp + where, in-place → cat | Extracting only the in-interval elements makes the **element count data-dependent**, giving `GuardOnDataDependentSymNode`                        |
| Gather form of relative-position attention | rel⇄abs shift → index table + last-dim gather         | The shift creates **quadratic shape expressions** `2P−1` / `2P²` / `P(2P−1)` (not rescued even by an affine extension of the dimension language) |
| Folding the FFN pad into the convolution   | explicit `F.pad` → `conv1d(padding=…)`                | Reduces `constant_pad_nd`. Exactly equivalent **only for an odd kernel and non-causal** (pinned by an assert)                                    |

#### Two schemes for the relative-position tables (ADR 0013)

The index tables (`idx_k` / `valid` / `idx_v`) are determined by `i` and `j` alone and are
**independent of length**, so the exporter's constant folding can lower them to baked tables +
`sym_prefix_slice`. The baked volume is O(sym_max²), however, so **the scheme differs per target**:

| Family        | Key side `idx_k` / `valid` | Value side `idx_v` | Baked volume    |
| ------------- | -------------------------- | ------------------ | --------------- |
| front (P≤512) | baked `(512,512)`          | baked              | 2.1MB measured  |
| flow / voice  | **graph inputs**           | baked `(4096,9)`   | 0.15MB measured |

Baking them for flow would come to `(4096,4096)` × 2 tables = **134MB**, hence the promotion to
inputs. The formulas are authoritative in `patch_sbv2.build_relattn_tables`, and **the in-graph
construction for front calls the same function** (writing the formula in two places would silently
produce two different tables the moment only one is fixed). The host-side mirror is
`packages/models/src/sbv2/relattn-tables.ts` (SBV2-specific, so it is not placed under
`packages/runtime/src/` — the models package owns the model-side knowledge), and **byte equality is
pinned by `packages/models/tests/sbv2_relattn_parity_test.ts` against the goldens' real data**. A mismatch
in window size 4 is of the silent-wrong-value class rather than a shape error, so the Python side
has `_assert_window_size` (at ckpt load) and the TS side has the parity test comparing against **the
width `2w+1` of the `idx_v` baked into the container**; gates on both sides exist because with only
one side there remains a path where the host and the goldens share the same mistake and slip
through.

#### The flow / voice wrappers

- `FlowReverse` applies `[TransformerCouplingLayer(mean_only) + Flip] × 4` in reversed order. The
  coupling reverse replaces `torch.split` with explicit slices (96/96) and **folds away the
  multiplication by `exp(−logs)`** (with `mean_only=True`, `logs = zeros_like(m)` and `exp(−0.0)` is
  exactly 1.0 under IEEE 754, so it is bit-identical — the standard trick for keeping `zeros_like`
  out of the IR).
- `Sbv2Voice` is `FlowReverse` → `z * y_mask` → `dec`. Same order as the tail of the reference
  infer, without carrying in the `max_len` slice (always `None` = identity on the inference path).

#### Gates against process contamination

- **`--verify` and emit cannot be combined in the same process** (the CLI rejects it with
  `parser.error`). The patches replace class attributes **process-wide** and `remove_weight_norm`
  destructively folds the weights, so if emit runs first the "pre-patch reference" can no longer be
  taken and the equivalence check becomes **vacuously true, a false PASS**. The verification itself
  follows the order "fix the references for every case → mutate → compare", and if that order is
  broken it raises a `RuntimeError` right before the reference capture (ADR 0013) — front / flow /
  voice check `patch_sbv2.patches_applied()`, while **dec checks the opposite**, that
  weight_norm-derived parameters are still present (its contamination source is the removal; voice
  has both gates).
- **`--verify` takes exactly one target.** Giving the CLI a pairwise exclusivity table ("MHA-patch
  targets are mutually exclusive / dec's removal conflicts only with voice / rounding conflicts with
  everything") would turn every hole in the table into a false PASS. With a form that takes a single
  value, **contaminating combinations structurally do not exist**.

#### Measured equivalence

| `--verify` | Cases | worst maxdiff                      | Notes                                                         |
| ---------- | ----- | ---------------------------------- | ------------------------------------------------------------- |
| `front`    | 9     | 2.02e-5 @P=512                     | `P ≤ 5` is bit-identical                                      |
| `flow`     | 10    | 1.43e-6 @T=512                     | `T ≤ 5` is bit-identical                                      |
| `dec`      | 10    | **0 (bit-identical in all cases)** | before/after remove_weight_norm                               |
| `voice`    | 10    | 1.25e-6 @T=203                     | compared against unpatched flow + dec with weight_norm active |

- The differences come from BLAS ordering as the value-side reduction length changes, and stay in
  the same order of magnitude as the real-GPU golden error (front 1.75e-5 / flow 2.74e-6 / voice
  1.60e-6).
- **dec being bit-identical in all cases closes an open item from recon §6** (until then there were
  only measurements for the single case `z=(1,192,50)`, recorded with the note that f32
  reproducibility of the effective weight `g·v/‖v‖` is not a spec guarantee). Since it requires
  `torch.equal`, even a `0.0` / `-0.0` mix-up would not pass.

## Real-weight Anima export

Real weights for the image generation side. `export_anima.py` writes out the 4 emit targets of ADR
[0016](../../docs/decisions/0016-anima-chain-export.md). The full emit (`outputs/series/anima/`) and the
Deno-side E2E (`packages/runtime/tests/e2e_anima_test.ts`) were completed in M1-P4.

### Dependencies and obtaining the weights

```bash
uv sync --group anima   # accelerate / diffusers==0.39.0 / torchvision / transformers==5.14.1
```

The weights are `circlestone-labs/Anima-Base-v1.0-Diffusers` on the HF Hub (downloaded automatically
on the first run; 5.3GB). `diffusers` is pinned with `==` because `patch_anima` replaces the
forwards of `QwenImageRMS_norm` / `QwenImageResample` / `QwenImageUpsample` /
`QwenImageAttentionBlock` at the class-attribute level and carries wrappers that transcribe the
forwards of `AnimaTextConditioner` / `CosmosTransformer3DModel` line by line (a minor update would
silently change the graph shape or the premises of eager equivalence).

### Generation and comparison

```bash
# emit (IR + golden io into <out>/<target>/)
uv run --group anima python export_anima.py --out /path/to/out
uv run --group anima python export_anima.py --target vae_decoder --out /path/to/out
uv run --group anima python export_anima.py --target transformer --num-layers 2 --out ...

# eager equivalence across the patches (**one target per process** — the CLI rejects combining them)
uv run --group anima python export_anima.py --verify text_encoder
uv run --group anima python export_anima.py --verify vae_decoder

# fuse a LoRA before emitting (applies to transformer / text_conditioner)
uv run --group anima python export_anima.py --target transformer --lora turbo.safetensors

# S form (one symbol for the token length), an additional series — transformer only; the default out gets -dyn
uv run --group anima python export_anima.py --dtype f16 --dit-graph dyn --lora turbo.safetensors \
  --out ../../outputs/series/anima-turbo-f16-dyn
uv run --group anima python export_anima.py --dtype f16 --dit-graph dyn --verify transformer \
  --lora turbo.safetensors
```

- **`--verify` and `--target` cannot be combined in the same process.** The VAE patches replace
  class attributes process-wide, so if the emit side applies them first the "pre-patch reference" is
  contaminated and the equivalence check becomes vacuously true (the difference is always 0 = a
  breakage where green is no longer evidence — the discipline of ADR 0013).
- **`--lora` is fused before `--num-layers`.** The other way around, the LoRA belonging to the
  truncated layers would be silently discarded as "no target", and the leftover check of `fuse_lora`
  would stop working on the reduced model.

### `--dit-graph dyn` (the S form of the DiT — #21 wave T2)

An **additional series** that moves the graph entrance **after** the patchify and promotes the rope
cos / sin tables to graph inputs (the static series does not move by a single byte). The entrance is
`tokens [1,S,68]` plus two rope tables `[1,1,S,128]`, and the exit is `[1,S,64]` before the
unpatchify. **Zero resolution-dependent bakes remain** (the static form has 3: the padding channel
and the rope tables). The design rationale is in
[dynres-vae-tiling](../../docs/research/2026-08-03-dynres-vae-tiling.md) §2.2.

- **transformer only.** The other 3 targets do not depend on the resolution and share the static
  series (the CLI rejects specifying them). `--resolution` **has no effect and is therefore
  rejected** too — the golden resolutions are fixed at `DIT_DYN_RESOLUTIONS = (512, 1024)` and the
  graph itself carries no resolution.
- Alongside `model.safetensors` / `io.*`, the series directory holds **`rope_base.safetensors`**
  (64KiB). It is the **per-axis base table** the host (`examples/anima/host/dit-tokens.ts`) uses to
  assemble the rope tables, cut out of the `model.rope` output. **Why it is needed**: torch's f32
  trigonometric functions can be 1 ulp off from the correctly rounded value (measured: over 8,192
  position × frequency combinations, cos 472 and sin 231 cases), and JS's `Math.cos` cannot
  reproduce them. The static graph has torch's values baked in, so bit identity is only achievable
  by permuting the base table. The base table is **independent of the resolution**, and its number
  of rows (= the length of the upstream `seq = arange(max(max_size))`) is the model-side upper bound
  (for Anima, 128 = latent 256 = the equivalent of 2048px).
- `--verify transformer --dit-graph dyn` compares "host patchify → S form → host unpatchify" against
  the **pre-patch diffusers path**. Measured (with the turbo LoRA fused and f16 rounding applied):
  **`bit_exact=True` / maxdiff 0.000e+00 in both cases** (S=1,024 and S=4,096).
- The main real-GPU gate is `packages/runtime/tests/e2e_anima_dyn_test.ts` (S form ≡ static graph,
  exact Uint32 equality).

### Measurements (as of wave 2; DiT / Qwen3 with `--num-layers 2`, the rest full)

| Target             | IR nodes | model.safetensors | symbols       | max rank |
| ------------------ | -------- | ----------------- | ------------- | -------- |
| `text_encoder`     | 131      | 749.8MB           | `T`           | 4        |
| `text_conditioner` | 613      | 539.1MB           | `Tsrc` `Ttgt` | 4        |
| `transformer`      | 316      | 629.4MB           | (static)      | 4        |
| `vae_decoder`      | 455      | 101.3MB           | (static)      | 4        |

**Unsupported aten ops: 0 kinds in all 4** (at the start of wave 2 the union was 19 kinds — recon
§2). `text_conditioner` / `vae_decoder` are not truncated, so this table is already the full depth
for them; `text_encoder` was separately measured at 0 kinds for the full 28 layers as well (2467 FX
nodes after normalization). The full 28 layers of the DiT are 7.29GiB in f32, so they are measured
in wave 3 — the layer count does not change the **kinds** of unsupported ops (the same block
repeated).

### Measurements (wave 3, **full depth**; all 4 into `outputs/series/anima/`)

Measured with each target in a **separate process** (so that host RAM peaks do not overlap; listing
several `--target`s in one process also works, but the DiT peak would stack on top of the other 3
staying resident).

| Target             | IR nodes | initializers | model.safetensors | emit time | peak RAM  |
| ------------------ | -------- | ------------ | ----------------- | --------- | --------- |
| `text_encoder`     | 1769     | 317          | 2,386,195,204 B   | 17.5s     | 3,794MiB  |
| `text_conditioner` | 613      | 122          | 539,060,388 B     | 5.7s      | 1,152MiB  |
| `transformer`      | 3904     | 579          | 7,827,646,080 B   | 36.5s     | 11,593MiB |
| `vae_decoder`      | 455      | 108          | 101,279,604 B     | 10.0s     | 1,546MiB  |

`model.safetensors` totals 10,854,181,276 B, and `outputs/series/anima/` as a whole, including
`io.<case>.safetensors`, is 10,868,931,292 B (10.12GiB). **`outputs/` is under `.gitignore`** (never
committed). The op vocabulary is **23 kinds** across the 4 targets (`add bmm cat clamp clamp_min
conv2d div embedding expand gelu layer_norm linear mul neg permute reshape rms_norm sigmoid slice
softmax sqrt sum sym_prefix_slice`), with **zero new ops** (the 3 kinds added in wave 1 sufficed).

**Regeneration determinism**: `vae_decoder` was emitted a second time into a different directory and
all 3 files — `model.safetensors` / `io.case0` / `io.case1` — were confirmed to have **matching
sha256** (the randomness is derived from `SEED` and does not depend on the global seed). The 3 large
targets are unconfirmed for reasons of emit time and disk space — the randomness path is the same
implementation in all 4, so it is stated explicitly that the confirmation covers `vae_decoder` only.

### Measurements (wave 3; the `--verify` eager equivalence is unchanged at full depth)

The Deno-side real-GPU comparison results are authoritative in the tolerance comments of
`packages/runtime/tests/e2e_anima_test.ts`. In summary:

| Target             | Real-GPU golden          | Notes                                                               |
| ------------------ | ------------------------ | ------------------------------------------------------------------- |
| `text_encoder`     | maxAbs 5.22e-4 (3 cases) | 130,816 folded -inf values with **0 NaN**                           |
| `text_conditioner` | maxAbs 2.23e-6 (2 cases) | Tsrc/Ttgt varied separately to kill mix-ups                         |
| `transformer`      | **not run**              | the 7,465MiB of weights exceed this machine's GPU limit of 7,280MiB |
| `vae_decoder`      | maxAbs 1.01e-5 (2 cases) | same order as the 9.34e-6 of `--verify`                             |

`transformer` being unrun is **not a graph problem**: what is emitted with `--num-layers 20`
(5,613MiB of weights) runs to completion on a real GPU and agrees with torch (maxAbs 9.63e-5).
Details in `docs/known-issues.md`.

Eager equivalence across the patches (`--verify`):

| Target             | Cases | worst maxdiff         | Notes                                                    |
| ------------------ | ----- | --------------------- | -------------------------------------------------------- |
| `text_encoder`     | 3     | **0 (bit-identical)** | all-ones mask dropped                                    |
| `text_conditioner` | 2     | **0 (bit-identical)** | two all-ones masks dropped                               |
| `transformer`      | 2     | **0 (bit-identical)** | timestep promotion / padding turned into a zero constant |
| `vae_decoder`      | 2     | 9.34e-6               | conv3d → conv2d reduction order difference (below)       |

The VAE alone is not bit-identical because CausalConv3d(T=1) → conv2d becomes **the same number of
additions performed in a different order**. Equivalence per rewrite is pinned by
`tests/test_patch_anima.py` at `< 1e-14` in f64 (conv3d↔conv2d / channel L2↔`F.normalize` / rank-4
form of RMS_norm). The two nearest-exact resamples, which only move data, are **bit-identical** in
f32. The amplification factor of the whole decoder is about 5.4e3 as measured in f64, so f32
rounding (1.19e-7) growing into the 9e-6 class is consistent.

> NOTE: unlike the f64 unit checks above, running `--verify` in f64 does not drive the residual to 0
> (measured 9.1e-8). The cause is that `QwenImageUpsample.forward` in the original implementation
> drops to f32 with `x.float()` — on the f32 path (= the path that gets exported) `.float()` is a
> no-op, so it is bit-identical there.

### f16 storage emit (`--dtype f16` — ADR 0018)

`--dtype f16` takes the references and goldens **after rounding the weights to f16-representable
values** (fake-quant) and stores **only the eligible weight slots** in f16. The output goes to
`outputs/series/anima-f16/`, separate from the f32 series (the default for an omitted `--out` switches with
`--dtype`).

```sh
uv run --group anima python export_anima.py --dtype f16                    # all 4 at once
uv run --group anima python export_anima.py --dtype f16 --target transformer
uv run --group anima python anima_pipeline.py --dtype f16                  # fixture
```

**MUST: the rounding happens before the reference and golden capture** (ADR 0006). `_fake_quant` is
applied right after each builder assembles the model (and **after** any `--lora` fusion). Moving it
later would leave only the reference computed on the original weights, making the E2E difference a
mixture of "quantization error + implementation error" — which only ever loosens the tolerance, i.e.
a breakage that **stays green while losing detection power**.

**Eligibility is the AND of 2 conditions** (`karume/emit.py`):

1. That initializer is consumed **only** by `WEIGHT_SLOTS` (the weights of `linear` / `conv1d` /
   `conv2d` / `conv_transpose1d` = slot 1, `embedding` = slot 0). This mirrors the runtime side
   `packages/runtime/src/runtime/plan.ts`, and any divergence is caught from both TS and Python by
   the conformance table (`weight_slot` in `packages/runtime/tests/fixtures/op-contracts.json`).
   **Bias is never included** — the root-cause fix for the prototype's f16 demotion bug (an f32 bias
   constant dragging the weight along with it, leaving 0MB eligible).
2. The f32 → f16 → f32 round trip is **bit-identical**. Anything eligible that does not match fails
   loudly (either the rounding was not applied, the order is wrong, or a folded constant flowed into
   a weight slot).

Specifying f16 while 0 tensors are eligible also fails with `EmitError` (the writer-side counterpart
of ADR 0006's "never let 0MB eligible stay silent").

**safetensors ordering** (`docs/limitations.md`): Karume's reader requires the data section to be
covered "without gaps and aligned to the element size", so placing an F32 / I32 tensor immediately
after an **F16 with an odd element count** (byte length ≡ 2 mod 4) makes loading fail on an
alignment violation. Ordering is the exporter's responsibility, so it does not use `save_file` but
decides the order and writes the file itself (never entrusting the order to the implementation
detail of an external library):

    F32 (name ascending) → I32 (name ascending) → even-count F16 → **odd-count F16 (last)**

Everything before the odd F16 group has a length that is a multiple of 4, so the cumulative offset
stays a multiple of 4, and odd F16 tensors among themselves only need 2-byte alignment. Right after
writing, `verify_model` runs a **check that transcribes Karume's reader rules**
(`assert_reader_layout`) — HF's `safe_open` **can still read** files with alignment violations, so
going through it alone would not detect the problem (the fault injection in `tests/test_emit.py`
demonstrates this). For a file that contains no f16 / i8 at all, this ordering is **byte-identical**
to the output of `save_file` (confirmed on the 24 f32 tiny goldens — the f32-series assets do not
move by a single byte when the writer is swapped; only the 25th, `i8_weights`, uses compressed
storage).

#### Measurements (2026-08-03, `--dtype f16` at full depth, separate processes)

| Target             | model.safetensors | vs f32 | Eligible (f16 storage)  | Ineligible (f32 storage) | emit time | peak RAM  |
| ------------------ | ----------------- | ------ | ----------------------- | ------------------------ | --------- | --------- |
| `text_encoder`     | 1,194,225,580 B   | 50.0%  | 197 tensors / 1,192.0MB | 120 tensors / 1.86MB     | 19.0s     | 4,292MiB  |
| `text_conditioner` | 269,838,164 B     | 50.1%  | 62 tensors / 269.2MB    | 60 tensors / 0.48MB      | 6.2s      | 1,360MiB  |
| `transformer`      | 3,914,867,592 B   | 50.0%  | 454 tensors / 3,912.8MB | 125 tensors / 1.22MB     | 42.2s     | 11,807MiB |
| `vae_decoder`      | 50,732,956 B      | 50.1%  | 37 tensors / 50.5MB     | 71 tensors / 0.075MB     | 10.4s     | n/a       |

`outputs/series/anima-f16/` as a whole is 5,444,414,308 B (5.07GiB; the f32 series is 10.12GiB). The rounded
weights amount to 5.96e8 elements for text_encoder, 1.35e8 for text_conditioner, 19.56e8 for
transformer and 1.27e8 for the vae. **The ineligible bytes are under 0.5% for every target**
(biases, norm weights, folded constants), so the constraint "ineligible means zero VRAM reduction"
does no real harm for Anima.

Measured `Session.diagnostics().storage` (Deno side) — `residentCompressedBytes` is the number of
bytes that stayed compressed on the GPU, `hostExpandedBytes` is the part of the f16 declarations
that was expanded to f32 at load time:

| Target             | resident   | hostExpanded |
| ------------------ | ---------- | ------------ |
| `text_encoder`     | 1,136.8MiB | 0.0MiB       |
| `text_conditioner` | 256.8MiB   | 0.0MiB       |
| `transformer`      | 3,731.5MiB | 0.0MiB       |
| `vae_decoder`      | 48.2MiB    | 0.0MiB       |

`hostExpanded` being 0 everywhere is **by design**: the exporter only **declares** f16 for what
passed the eligibility check, so the runtime's "declared f16 but ineligible → expand on the CPU"
path is never entered (that path itself is reachable with hand-written IR, and
`packages/runtime/tests/gpu_f16_weights_test.ts` pins it in isolation).

**This is what put the full 28-layer DiT on a real GPU**: the 7,465MiB of f32 exceeded the GPUBuffer
ceiling of 7,280MiB (`docs/known-issues.md`) and could not be loaded, whereas f16 is 3,731.5MiB,
half the ceiling. The real-GPU golden and the stage-② measurements of the end-to-end chain are
authoritative in the tolerance comments of `packages/runtime/tests/e2e_anima_test.ts` (DiT golden
maxAbs 6.68e-5, stage ② raw DiT output 3.03e-5, stage ③ end-to-end 6.41e-6).

`anima_pipeline.py --dtype f16` fake-quants **all 4** components before taking the references (if
even one were left unrounded, that stage alone would be the numbers of a different model). The
output is `outputs/series/anima-pipeline-f16/` (21 tensors, 9.4MB), measured at 44s and 13.5s per DiT step.

#### Measurement: Anima's weights are already BF16, so f16 rounding is near-identity (2026-08-03)

In HF's `circlestone-labs/Anima-Base-v1.0-Diffusers`, **all 4 components are `BF16` in the
safetensors** (`torch_dtype` in `text_encoder/config.json` is `bfloat16` as well). BF16's 8-bit
mantissa fits inside f16's 11 bits, so promoting to f32 and rounding to f16 only moves values that
**fall into f16's subnormal range**:

| Component          | Elements moved by rounding    | Max change |
| ------------------ | ----------------------------- | ---------- |
| `vae`              | 163,271 / 1.27e8 = **0.129%** | 2.98e-8    |
| `text_conditioner` | 16,718 / 1.35e8 = **0.0124%** | 1.59e-4    |

**This fact constrains how the f16 series may be read**: the f16 and f32 series having errors of the
same order was not because "quantization is harmless" but because f16 was applied to a model that
was BF16 to begin with — for f32-trained weights the story is different.

**Fault injection results (important — not what was expected)**: actually injecting "move the
fake-quant after the reference capture" gave the following.

| Injection                                                         | Result                                                                                                                                |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| ① remove `_fake_quant` from the builder                           | **emit fails loudly** (`EmitError: initializer 'p_vae_decoder_conv_in_weight' … fake-quant が未適用か、参照採取より後に掛かっている`) |
| ② ① plus removing the eligibility check (round-trip bit equality) | the assets are written but **the E2E stays green** (vae_decoder maxAbs 6.85e-6 / text_encoder 4.40e-4 — both inside the tolerance)    |

② failing to turn red is a consequence of the BF16 fact above: the difference in torch's own output
across the rounding is itself small (measured: 2.64e-3 for text_encoder t024 = 3.8e-5 relative to a
range of 68.7, and 1.72e-5 for vae_decoder). In other words, **for Anima the E2E is not a detector
for a forgotten rounding** — the only thing detecting it is the emit-side eligibility check. Do not
treat this gate as redundant and remove it.

> NOTE (2026-08-03, measured against SBV2 f16 as a control): this "invisible to the E2E" property is
> **a property of the model (its distribution dtype)**, not of f16 storage in general. For SBV2,
> whose ckpt is genuinely f32, the same injection turns **both** the emit gate and the E2E red
> (exceeding atol by more than 31× — ADR 0027).

### i8 storage emit (`--dtype i8` — ADR 0019)

`--dtype i8` fake-quants with **per-channel symmetric int8** (no zero point) before taking the
references and goldens, and stores the eligible weight slots as i8 plus companion scales. The output
goes to `outputs/series/anima-i8/`.

```sh
uv run --group anima python export_anima.py --dtype i8              # transformer only
uv run --group anima python export_anima.py --verify transformer --dtype i8
uv run --group anima python anima_pipeline.py --dtype i8            # fixture
```

**MUST: `--dtype i8` is transformer-only** (`DTYPE_TARGETS` — the CLI rejects other targets even
when spelled out with `--target` / `--verify`). Two reasons:

1. **Series design** (ADR 0019): the DiT's −1.87GiB is the dominant term, while text / cond / VAE
   share `outputs/series/anima-f16/`. Prototype measurements put i8 for the VAE two orders lower.
2. **The VAE cannot satisfy the rounding order constraint**: `patch_anima` replaces the CausalConv3d
   weights with **the last slice along the time axis**, which happens when the patches are applied
   (= after the reference capture). f16's element-wise rounding commutes with slicing and is
   therefore harmless, but i8's per-channel scale **counts even the discarded elements into amax**
   (a shifted scale moves the value of every element).

**Definition of the quantization** (`karume/quantize.py`):

- `scale = clamp(amax / 127, f32 tiny)` per output channel, and `q = clamp(round(w/scale), ±127)`.
  **−128 is not used** — the element with the largest absolute value lands on `q = ±127` and is
  restored exactly by `q·scale`, which makes the fake-quant **idempotent** (bit-invariant under
  reapplication). The lower clamp avoids `0/0` for all-zero channels.
- The channel axis comes from a table keyed by module type (`QUANT_CHANNEL_AXES`). Only
  `ConvTranspose1d` has the transposed weight layout `[Cin, Cout, K]` and therefore axis **1**; the
  other 4 use 0. Against the op-name-keyed mirror (`ops.WEIGHT_CHANNEL_AXES` / TS-side
  `packages/runtime/src/ops.ts`), the conformance table (`channel_axis` in
  `packages/runtime/tests/fixtures/op-contracts.json`) cross-checks from both sides.
- Matching is by **FQN** (`<module>.weight`) — `id(tensor)` is not used (ADR 0006). `convert.py`
  uses the FQN verbatim as the safetensors tensor key, so it meshes with the emit side in the same
  namespace. **That is why `--dtype i8` applies the rounding to the exported wrapper (`AnimaDit`)**
  (applying it to the inner `model` would make the FQN prefixes disagree).
- 0 targets fails loudly with `QuantizeError` (never silently allowing "`--dtype i8` was given, yet
  what got written is effectively f32").

**Eligibility is the same AND of 2 conditions as f16** (`eligible_compressed_initializers` is
shared). The second one, "inverse-transform bit equality", becomes `torch.equal(q8.to(f32) · scale,
t)` for i8. **The scale written out is exactly the one the fake-quant used** (never recomputed).

**Companion scales**: an F32 tensor named `karume.scale.<weight key>` goes into the same file, and
the IR declares that key explicitly via `storage.scale` (**mandatory for `i8`** — defaulting it to
1.0 would turn a forgotten declaration into "a weight dequantized with 1.0 on every channel", which
would load and run just fine). Name collisions with real tensors are checked before writing.

**Ordering**: I8 has an element size of 1 and therefore no alignment constraint, but it does produce
**arbitrary byte lengths**, so it goes after the existing F16 rules = **last**. Placing it earlier
would push the following absolute offsets off the multiple of their element size (the fault
injection in `test_emit.py` demonstrates this — HF's `safe_open` can still read them).

    F32 (name ascending) → I32 → even-count F16 → odd-count F16 → **I8 (last)**

#### Measurements (2026-08-03, `--dtype i8` at the full 28 layers, separate process)

| Metric                          | Measured                                                 |
| ------------------------------- | -------------------------------------------------------- |
| `model.safetensors`             | 1,963,762,200 B (**1,872.8MiB**)                         |
| vs f32 / vs f16                 | **25.09%** / **50.16%** (f32 7,465.0MiB, f16 3,733.5MiB) |
| Eligible (i8 storage)           | 454 tensors / 1,956.4MB (19.56e8 elements)               |
| Companion scales                | 5.19MB = **0.265%** of the eligible bytes                |
| Ineligible (f32 storage)        | 125 tensors / 1.22MB                                     |
| emit time / peak RSS            | 44.3s / 11,593MiB                                        |
| `--verify transformer`, 2 cases | maxdiff 0.000e+00, **bit-identical in every case**       |

`Session.diagnostics().storage` (Deno side) reports `residentCompressedBytes` **1,961,579,776 B
(1,870.7MiB)** and `hostExpandedBytes` **0**. The former matches the exporter's `compressed_bytes +
scale_bytes` (1,956,388,864 + 5,190,912) **byte for byte** — a measurement that "adding the scales
into the diagnostics" means the same thing in the exporter and the runtime.

The scale overhead is **0.265%**, smaller than the 0.4–0.9% estimated in ADR 0019, because the DiT's
Linear weights are `[Cout, Cin]` with Cin as large as 1024–4096 (the scales only need Cout entries).

`anima_pipeline.py --dtype i8` rounds **the DiT in i8 and the other 3 components in f16** before
taking the references (`COMPONENT_DTYPES`). This is to keep a one-to-one correspondence with the
asset series (`outputs/series/anima-i8/transformer` plus the other 3 from `outputs/series/anima-f16/`); making
everything i8 would make the text-path references the numbers of a different model than the assets
actually being executed. The output is `outputs/series/anima-pipeline-i8/` (21 tensors, 9,873,808 B),
measured at 14.1s / 14.4s per DiT step.

The real-GPU E2E measurements are authoritative in the tolerance comments of
`packages/runtime/tests/e2e_anima_i8_test.ts` (DiT golden maxAbs 8.59e-5 / stage ② raw DiT output
5.34e-5 / stage ② latents 1.19e-6). **The f16-series values are not reused** — i8 rounding produces
different weights, so even the implementation error stemming from reduction order is a different
quantity.

#### Fault injection results (2026-08-03, pytest + tiny golden regeneration)

Breaking exactly one thing and then running `uv run pytest tests/` and `python -m karume.goldens`
(the harness always restores, and the post-restore baseline of 1,602 passed was re-confirmed).

| Injection                                                         | Result                                                                                                                                          |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| ① broken rounding order (fake-quant after the export)             | pytest **208 errors** / golden generation stops with `EmitError: … per-channel scale が無い`                                                    |
| ② recomputing the scale (axis fixed to 0)                         | pytest **2 failed** / golden generation stops with `EmitError: … 逆変換してもビット一致しない` (the failure is `up.weight` = transposed axis 1) |
| ③ recomputing the scale (**correct axis**)                        | pytest **2 failed** (the negative tests detect that the gate itself is gone). **The goldens regenerate byte-identically**                       |
| ④ letting −128 in (`scale = amax/128`, −128 on the negative side) | pytest **5 failed** (±127 / idempotence / scale definition / golden determinism / scale fixed point)                                            |
| ⑤ broken write order (I8 moved into the leading group)            | pytest **3 failed** / golden generation stops with `ContainerError: … 絶対 offset 4175 が F32 の要素サイズ 4 に整列していない`                  |

**③ is the important one (not what was expected)**: "re-deriving the scale from an
already-fake-quantized weight on the correct axis" **changes nothing in the data**. Because `q` is
closed to ±127, the element with the largest absolute value necessarily lands on `q = ±127`, and
`amax(|q·s|)/127 = fl(fl(127·s)/127) = s` is a **fixed point** in f32 (zero counterexamples over
8.9e7 random samples). So ADR 0019's "no recomputation" is a discipline the inverse-transform gate
cannot detect, and the reason to keep it lies elsewhere: when "the weight at emit time is not the
effective weight / the axis is wrong / the formula is wrong", the scale silently becomes a different
one (② is exactly that case). This fixed-point property is pinned by
`test_emit.py::test_recomputing_the_scale_from_a_quantized_weight_is_a_fixed_point`.

### Patch layer (`karume/patch_anima.py`)

Unlike `patch_sbv2`, the goal is **not exportability but IR quality** (ADR 0016 / recon §5). All 4
targets export even unpatched, but as-is the vocabulary would need conv3d / `linalg_vector_norm` /
`upsample_nearest2d`, the rank would go up to 8, and the timestep embedding would be baked into the
graph.

| Subject     | Rewrite                                                                                                                              |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Qwen3       | do not pass `attention_mask` (an all-ones mask ⇒ the causal mask alone, with an identical additive bias)                             |
| Conditioner | do not pass the two all-ones masks. The 512 padding and the output mask multiplication are host-side                                 |
| DiT         | promote the timestep embedding to a graph input / turn `padding_mask` into a zero-constant channel / move to rank 4                  |
| VAE decoder | CausalConv3d(T=1) → conv2d / nearest-exact ×2 → reshape+expand / channel L2 → last-dim sum + `clamp_min` / `feat_cache` fails loudly |
| Shared      | demote RoPE's `inv_freq` from a buffer to a plain attribute (making it a leaf for constant folding)                                  |

The `--verify` references are taken **before the patches** (with `vae_patches_applied()` as the gate
for the VAE). The reference is `vae.decode` itself — one comparison measures both "conv3d ⇒ conv2d
equivalence" and "at T=1 the feat_cache does not affect the result" at the same time (checking one
of them through a separate path would leave room for the other's premise to break silently).

### LoRA (`karume/lora.py`)

ΔW=(B@A)·scale is **computed in f32** and added in place to the original dtype. The IR does not
change by a single node and no runtime implementation is required (ADR 0016). Name conversion calls
`_convert_non_diffusers_anima_lora_to_diffusers` bundled with diffusers (no hand-maintained mapping
table). There are 4 fail-loudly conditions: `lora_B` being all zeros for the whole file / 0 targets
matched / one of `lora_A`, `lora_B` missing / a shape mismatch. A per-component ΔW=0 can
legitimately happen, so that is reported through `FuseReport.is_noop` instead (not raised).

## Anima host pipeline reference fixture (`anima_pipeline.py`)

Only 4 graphs go into the IR; everything outside them — tokenization / sigma schedule / timestep
embedding table / CFG / Euler update / latent denormalization / 512 padding — is host code. This
script pins **the authoritative numbers for that part** into a single fixture. The authority is the
4 blocks of `modular_pipelines/anima/` in diffusers 0.39, taken through the **plain diffusers path
without the patch layer** (patch-layer equivalence is measured separately by `export_anima.py
--verify` — keeping the verification nets independent; going through the patches here as well would
put a patch bug into both the reference and the subject under test in the same shape, letting it
pass with a difference of 0).

```sh
uv run --group anima python anima_pipeline.py                    # into outputs/series/anima-pipeline/
uv run --group anima python anima_pipeline.py --steps 32 --ref-steps 2
uv run --group anima python anima_pipeline.py --resolution 1344x768 …   # non-square (#23)
```

- The outputs are `outputs/series/anima-pipeline/pipeline.safetensors` (21 tensors, 9.4MB) and
  `pipeline.json` (prompt, step count, shift, CFG coefficient, and the role and shape of every
  tensor).
- **Not placed directly under the distribution tree `models/anima-turbo/`** — that one holds exactly
  the files the manifest declares and is uploaded to HF as-is, and an undeclared file stops
  `verify_dist` (the same reason `outputs/sbv2-demo/` is kept separate).
- The prompt is a single fixed English string (danbooru-style tags). **The negative prompt is not
  the empty string** — an empty T5 id sequence has length 1 and falls outside the conditioner's
  accepted set `Dim("Ttgt", min=2)`.
- `latents_init` uses the fixed `SEED = 20260802` (independent of the global seed).
- **`--resolution` takes `WxH`** (square may be abbreviated; the spelling is the same as the demo's
  `--resolution`, and `karume/resolution.py` is authoritative). For non-square, the **axis order**
  of `latents_init [1,16,H/8,W/8]` and `padding_mask [1,1,H,W]` is the one pitfall, since swapping
  them still gives a matching element count. The metadata carries both the spelling (`resolution`)
  and `width` / `height` — **readers should use the latter** (`resolution` stays an int for square,
  for compatibility with tests that read existing fixtures).
- The raw DiT outputs (`noise_cond_*` / `noise_uncond_*`) are kept as well. Without them the Deno
  side **cannot parity-check the CFG and Euler host glue in isolation** and can only see it mixed
  with the DiT's error. On top of that, the σ step is only `sigmas[1] − sigmas[0] = −1.064e-2`, so
  the DiT error is diluted roughly 100× by the Euler update (a detector that only looks at the
  latents is structurally blunt).
- `image` is the return value of `vae.decode` itself. `AutoencoderKLQwenImage._decode` applies a
  final `clamp(-1, 1)` (this clamp is what `AnimaVaeDecoder` bakes in — it does not come from
  postprocessing), so the clamp sits at the same position on the fixture side and the IR side.

Measured (`--steps 32 --ref-steps 2`, 512px): **44.0s, peak RAM 12,918MiB** (4 DiT runs on CPU f32 =
2 steps × cond/uncond; 13.5s per step). On the text side, qwen 29 and t5 30 tokens (negative 13 /
21).

The Deno-side host glue implementation (`sigmaSchedule` / `cfgEulerStep` / `denormalizeLatents` /
`padSequence` in `packages/runtime/tests/e2e_anima_test.ts`) is **bit-identical to this fixture in
all 4** (rounding to f32 one operation at a time with `Math.fround`).

### Fusing the Turbo LoRA and the turbo reference fixture

Passing a few-step distilled Turbo LoRA to `--lora` (e.g.
`models/anima-turbo-lora-v0.2.safetensors`; the real weights are not in the repository — place them
by hand) fuses it into the weights before the export. This LoRA has been **measured to have an
all-zero (noop) `lora_B` on the text_conditioner side**, so **emitting the transformer target alone
is enough** (the other 3 targets share the existing `outputs/series/anima-f16/`):

```sh
uv run --group anima python export_anima.py --dtype f16 --target transformer \
  --lora ../../models/anima-turbo-lora-v0.2.safetensors --out ../../outputs/series/anima-turbo-f16
```

`--verify transformer --lora <path>` becomes an eager equivalence check against the post-LoRA
weights (`_apply_lora` takes effect before the fake-quant and the reference capture, so both sides
see the same LoRA-applied model with no extra code changes; measured: bit-identical in every case).

Reference fixture for turbo operation (steps=10 / CFG=1):

```sh
uv run --group anima python anima_pipeline.py --dtype f16 --steps 10 --ref-steps 10 \
  --guidance-scale 1.0 --lora ../../models/anima-turbo-lora-v0.2.safetensors \
  --out ../../outputs/series/anima-pipeline-turbo-f16
```

`--guidance-scale 1.0` **skips the DiT call for the uncond branch entirely** (the same shape as a
production turbo deployment), so the output fixture **has no `noise_uncond_stepNNNN` keys**. Note
that the key set differs from the base-series fixtures with guidance_scale != 1.0 (the host glue
treats the missing keys as the "do not compute uncond" branch).

## VAE decode fixed-tiling reference fixture (`anima_tiling.py`)

The **authoritative host-side numbers** for `--vae-tiling` in `examples/anima` (splitting the VAE
decode into fixed latent 64×64 tiles). The VAE decoder graph is **structurally invariant with
respect to resolution** (the `model.safetensors` for 512px and for 1024px agree exactly, down to the
node sequence and the weight bytes — recon `docs/research/2026-08-03-dynres-vae-tiling.md` §1.2), so
the 512px assets can be used directly as the tile decoder. **Nothing is added to the IR or the
runtime**; only cutting, blending and pasting live on the host (`examples/anima/host/tiling.ts`).

```sh
uv run --group anima python anima_tiling.py     # into outputs/series/anima-tiling-f16-1024/
uv run --group anima python anima_tiling.py --resolution 1344x768 \
  --latents ../../outputs/series/anima-pipeline-turbo-f16-1344x768/pipeline.safetensors
```

- The input latent is borrowed from `latents_denorm` in
  `outputs/series/anima-pipeline-turbo-f16-1024/pipeline.safetensors` (already denormalized = exactly the
  VAE decoder input). **Generate that first** (otherwise it fails naming the missing file). A real
  pipeline latent is used rather than randn because how the seams show depends on the actual values.
- The outputs are `tiling.safetensors` (`latents_denorm` / `image_tiled`, 13.0MB) and `tiling.json`
  (**the tile geometry** = per-axis start positions, stride and blend width, plus observations of
  the difference against a non-tiled decode). The Deno side
  (`packages/runtime/tests/e2e_anima_tiling_test.ts`) checks **against this geometry metadata** as
  well as the numbers — numbers alone cannot rule out "a different geometry that also lands inside
  the tolerance".
- **`vae.enable_tiling()` is not used.** The upstream `tiled_decode` scans with `range(0, H,
  stride)`, which makes the last tile shorter — something a fixed-shape tile decoder cannot digest.
  The scan is changed to "evenly spaced placement that snaps the last tile's start to `extent −
  tile`" (the deliberate deviation announced in recon §4.2). The script fails loudly if
  `vae.use_tiling` is true.
- **The blend formulas are transcribed verbatim from upstream** (`blend_v` / `blend_h`). Equivalence
  is pinned by `tests/test_anima_tiling.py` as **bit equality against the real methods** — once the
  scan is our own, this is the only place the formulas' isomorphism can be guaranteed.
- The weights are fake-quantized to the same dtype as the asset series before the references are
  taken (ADR 0006 — the same discipline as `anima_pipeline.py`). The default `--dtype f16`
  corresponds to the `outputs/series/anima-f16/vae_decoder` the TS side opens.

- **`--resolution` takes `WxH`** (non-square is #23). The geometry is built from the shape of the
  input latent, so this argument only gates "is the borrowed latent at the intended resolution" and
  determines the default `--out` name. 1344×768 (latent 96×168) is **2×3 = 6 tiles, stride 32/52,
  blend 256/96px**.

Measured (1024px, CPU f32, 9 tiles): the difference against a non-tiled decode is **maxAbs 5.07e-2 /
mean 9.82e-4**. Tiling is an **approximation** that confines the receptive field of the decoder's
attention to within a tile (the same approximation as upstream `tiled_decode`), so **this difference
being non-zero is correct**. The implementation error is a separate matter: real GPU vs torch is
maxAbs 1.642e-5 (the tolerance derivation in `packages/runtime/tests/e2e_anima_tiling_test.ts`). The
same observation for 1344×768 (6 tiles) is **maxAbs 9.97e-2**, with real GPU vs torch at **8.02e-6**
(`packages/runtime/tests/e2e_anima_nonsquare_test.ts`).

## Non-square rope table reference fixture (`anima_rope.py`)

The host for the S-form DiT (`--dit-graph dyn`) assembles the rope cos / sin by **permuting the
per-axis base tables**. **On square resolutions an h ↔ w mix-up in that permutation is in principle
undetectable** — Anima's `rope_scale` has the same value for h and w (`[1.0, 4.0, 4.0]`), so `cos_h`
and `cos_w` agree byte for byte and, with H'=W', the tables themselves hold the same values
(detection limit 1 of ADR 0034). **On non-square resolutions the positional mix-up splits them
apart**, so the upstream `model.rope` tables are baked for 4 geometries and compared against the TS
side's reconstruction with exact Uint32 equality.

```sh
uv run --group anima python anima_rope.py       # into outputs/series/anima-rope-nonsquare/ (a few seconds)
```

- The geometries are the fixed 4 of `GEOMETRIES` (**16:9 and 3:4, both orientations** = 1344×768 /
  768×1344 / 1152×896 / 896×1152, all with S=4,032). **Both orientations of a pair must be
  included** — with only one, "an implementation that swapped h and w" would agree with the table of
  the other geometry. The script fails if a square resolution creeps in.
- **Not a single byte of weights is read.** `CosmosRotaryPosEmbed` has neither parameters nor
  buffers and is pure computation, so the model is built from the config on the `meta` device (no
  7.3GiB load required).
- The outputs are `rope.safetensors` (`cos_<WxH>` / `sin_<WxH>` per geometry, each `[1,1,S,128]`,
  16.5MB) and `rope.json` (latent dimensions, token grid, S, number of rows in the base tables).
- The Python-side mirror (reconstruction from the base tables ≡ the upstream tables) is pinned by
  `tests/test_anima_rope.py` using a **synthetic rope without real weights** (a configuration where
  h and w of `rope_scale` differ so that a mix-up shows up in the numbers).

## Prompt layer of the image demo (`anima_demo.py`)

The script that produces the **runtime assets** required by `examples/anima/text/` (the Deno
implementation of prompt string → token id sequence) and the **parity fixture** for it. It does not
touch the model graphs.

```sh
# assets (2 files into outputs/series/anima-demo/text/) + fixture (packages/runtime/tests/fixtures/anima-text/)
cd tools/exporter
uv run --group anima python anima_demo.py
# always format afterwards (the committed form is what the formatter produces — verify's fmt --check covers fixtures too)
cd ../.. && deno fmt packages/runtime/tests/fixtures/anima-text/parity.json
```

**A single run always produces both** (if they are not built from the same table, the runtime
assets and the fixture would age apart into "the tests are green but only the demo emits a
different id sequence"). Measured at **about 3 minutes** (dominated by the exhaustive checks
below).

| Output                                                   | Size (measured)                | Content                                                                                                 |
| -------------------------------------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------- |
| `outputs/series/anima-demo/text/qwen2-tokenizer.json`    | 3,514,619 B                    | vocabulary 151,643 / merges 151,387 / character class tables / NFC segmentation table / 26 added tokens |
| `outputs/series/anima-demo/text/t5-tokenizer.json`       | 1,093,419 B                    | vocabulary 32,100 + scores / normalization table / 103 added tokens                                     |
| `packages/runtime/tests/fixtures/anima-text/parity.json` | 474KB (formatted, git-tracked) | reference id sequences for 28 cases + 251 NFC pairs + a **subset** of the vocabularies                  |

- The runtime assets (4.6MB in total) live under **`outputs/` = `.gitignore`**. They keep only the
  information execution needs out of the raw `tokenizer.json` files (13.8MB in total), so that
  **licensed material is not carried in the repository**.
- **MUST: do not place them directly under the distribution tree `models/anima-turbo/`** — that one
  holds exactly the files the manifest declares and is uploaded to HF as-is (the same reason
  `outputs/series/anima-pipeline/` is kept separate).
- The fixture only holds a **subset** of the vocabularies (Qwen2 218 tokens / 375 merges / T5 125
  tokens), so every case can be reproduced without committing the 151k / 32k entries. The
  normalization table and the character class tables are **the folding output itself** (= the
  subject under test), so they are included in full.

### Why the tables are baked (`karume/anima_text.py`)

**Unicode decisions are neither reimplemented in TS nor delegated to standard APIs** (the same
discipline as `sbv2_demo.clean_text_ranges`). The authority for those decisions is the Unicode
tables on the Rust side (`tokenizers` / its regex engine / `unicode-normalization`), and the moment
they diverge from the JS engine's ICU version the pre-token boundaries or the normalization result
change and **the id sequence alone becomes a different thing, with no exception and no warning**.
Every code point is evaluated from Python and folded into closed-interval tables and mapping tables,
leaving TS with nothing but a binary search.

There are 5 kinds of baked tables:

| Table                                  | How it is baked (querying the authority)                                          | Measured                      |
| -------------------------------------- | --------------------------------------------------------------------------------- | ----------------------------- |
| character classes `\p{L}` `\p{N}` `\s` | feed one character at a time into `Split(Regex(…), behavior="removed")`           | 677 / 144 / 10 intervals      |
| `(?i:)` case equivalence               | `Split(Regex("(?i:s)"), …)` for the 8 suffix characters                           | 9 pairs                       |
| `NFC` segmentation                     | code points where `normalizers.NFC` and plain NFC diverge, via 8 context probes   | 123 cp / 43 intervals         |
| `Precompiled` normalization            | DARTS decoding of the charsmap + minimization to **only the rules that can fire** | 5,512 rules                   |
| cluster boundaries (3 tables)          | **probes** exploiting "pushing to 6 bytes or more makes the rule not fire"        | extend / breakAfter / prepend |

**`NFC` is not covered by the standard API either** (measured 2026-08-03): the authoritative
`unicode-normalization` has old Unicode tables and diverges from `String.prototype.normalize("NFC")`
/ `unicodedata` on **123 code points** (120 cp treat the combining class as 0 and do not reorder; 3
cp lack a newer canonical composition). The folding is "segment the string at the diverging code
points, run plain NFC on each segment only, and concatenate (the segmenting code points themselves
do not take part in normalization)". The real damage is a silent id-sequence mismatch, and because
`PROMPT_CASES` contained not one of the affected characters, it slipped through the 28-case gate
(measured on 1,200 randomized prompts: 6 mismatches for Qwen2 → 0 after the fix).

**`(?i:)` is not the ASCII equivalent** (measured 2026-08-03): Rust's `(?i:)` is Unicode simple case
folding, and among the 8 suffix characters (s t r e v m l d) **U+017F (ſ) is equated with `s`**.
`.lower()` / `toLowerCase()` / flipping ASCII case are none of them authoritative, and the boundary
in `it'ſs` becomes `'ſs` rather than `'ſ` + `s` (the fixture case `apostrophe_fold` is exactly this
boundary).

**`Precompiled` is neither NFKC nor longest-match**: it is whole-cluster replacement at the grapheme
cluster level with **shortest prefix wins**, so `A`+U+0301+U+0301 → `Á` (the third character
silently disappears) and `A`+U+0302+U+0301 → `Â` (the 2cp prefix wins over the 3cp rule `Ấ`).
Clusters of 6 or more UTF-8 bytes never enter the whole-replacement path, which is why Hangul,
regional indicators and emoji ZWJ sequences need no boundary rules implemented.

### Three stages of verification (nothing is emitted if any one of them comes off)

| Stage                                         | Where                                            | Scale (measured)                                                                                                                                                          |
| --------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ① equivalence of the folding                  | `anima_demo.py` (the emit gate)                  | all 1,112,064 cp + 5,512 rule keys + 200,000 randomized (fixed seed) / the pre-token scan and the NFC segmentation each cover **all cp × 11 contexts = 12,232,704 cases** |
| ② reference implementation vs `AutoTokenizer` | `anima_demo.py` + `tests/test_anima_demo.py`     | 28 cases (`padding="longest"` / `max_length=512` / `truncation=True` — the same call as `anima_pipeline.encode_text`) + **2,000 randomized prompts** (fixed seed)         |
| ③ TS implementation vs the fixture            | `packages/runtime/tests/anima_tokenizer_test.ts` | 28 cases × 2 tokenizers + 251 NFC pairs + property tests + a cross-check against the pipeline reference                                                                   |

- ① **only runs on regeneration** (it takes minutes). The pytest side (②) redoes the same comparison
  against the committed fixture, so **a change in the upstream tokenizer.json is noticed even
  without regenerating**.
- The pre-token scan uses 11 context probes. `'{}` alone as a single character is **not enough** —
  whether or not the contraction alternative fires, the result is the same single fragment, and a
  case-insensitivity mix-up slips through (this hole was actually hit in practice). NFC segmentation
  has the same shape, keeping the detection probes (8 contexts) and the verification probes (11
  contexts) **separate**.
- The 2,000 randomized cases of ② are a permanent gate that mechanically exercises "the combinations
  a hand-written script did not think of" (the engine difference in NFC went through a gap between
  the 28 cases and split the id sequence). They are drawn with a fixed seed from an alphabet mixing
  emoji, combining characters, segmenting code points and various kinds of whitespace.
- ③ **runs without the real assets** (the fixture contains a subset of the vocabularies). In an
  environment that has the assets, reproduction against the real 151k / 32k vocabularies and a
  cross-check against the `qwen_input_ids` / `t5_input_ids` (the id sequences captured by torch) of
  `outputs/series/anima-pipeline{,-f16,-i8,-turbo-f16}/` run in addition. The latter goes through a **path
  separate from the fixture generator**, so it catches the case where the fixture itself is wrong.
- The structural assumptions about the upstream `tokenizer.json` (`normalizer.type` / the head of
  `pre_tokenizer` / ByteLevel's `add_prefix_space` / the special tokens added by the post_processor
  / the flags of the added tokens / T5's `byte_fallback`) are checked by
  `anima_demo.check_upstream_shape` on every emit. ①–③ can only catch **what they hit**, so the
  assumptions themselves are pinned structurally.

## Asset prep and torch reference for the voice demo (`sbv2_demo.py`)

The script that provides the **host-side assets** required by `examples/sbv2/` (the Deno demo from
real text to WAV) and takes on the **numerical parity** of its output. It does not touch the model
graphs (neither the emit path nor the goldens change).

```sh
# ① runtime assets (3 files into outputs/sbv2-demo/)
uv run --group sbv2 python sbv2_demo.py assets

# ② run the demo (from the repository root) → out.wav and dump.safetensors
cd ../.. && deno task demo:sbv2 --text "こんにちは、これはテストです。" && cd tools/exporter

# ③ torch reference (rerun the same chain on the dump's discrete inputs and random sequence) → reference.wav + numbers
uv run --group sbv2 python sbv2_demo.py reference --dump ../../outputs/demo/sbv2-dump/dump.safetensors

# ④ official infer (the pyopenjtalk path) → official.wav (for listening comparisons of the accent)
uv run --group sbv2 python sbv2_demo.py official --text "こんにちは、これはテストです。"
```

### What `assets` produces

| File                     | Content                                                                                     |
| ------------------------ | ------------------------------------------------------------------------------------------- |
| `symbols.json`           | the JP-Extra id rules, model constants and knob defaults (**all read from the real thing**) |
| `deberta-tokenizer.json` | the DeBERTa character tokenizer's vocabulary, special ids and `_clean_text` decision table  |
| `assets.safetensors`     | `style_vec` `[1,256]` and the speaker embedding `g` `[1,512,1]`                             |

**MUST: never transcribe constants by hand.** The symbol table, the tone base
(`LANGUAGE_TONE_START_MAP["JP"]`), the language id (`LANGUAGE_ID_MAP["JP"]`) and the add_blank
insertion value are read from `style_bert_vits2`. The multilingual version and JP-Extra look the
same, yet a divergence **keeps the shapes matching and only breaks the sound** (the silent
wrong-value class). Only the insertion value is written as a literal in the source, so
`blank_id_from_source` extracts it from the source of `infer.get_text` with a regular expression for
all 3 series and additionally confirms that the value is the same across them.

Baking the `_clean_text` decision table has the same motivation: moving the `unicodedata.category`
classification into TS would turn ICU version differences into a silent mismatch — so every code
point is evaluated in Python and folded into closed intervals.

> NOTE: **`language` is not all zeros even for JP-Extra.** `infer.get_text` goes through
> `cleaned_text_to_sequence(..., JP)`, so the real phoneme positions are 1 and only the add_blank
> insertion positions are 0. `export_sbv2.make_language` being all zeros is a choice made for
> _synthetic golden inputs_ (any value makes the golden valid); it is not the inference rule.

### What `reference` claims

It runs the `patch_sbv2` modules (`Sbv2Front` / `Sbv2Voice`) together with **the same host glue as
the demo** on torch CPU and compares against the Karume waveform recorded in the dump. So what is
being measured is "the same computation graph run on a real GPU vs run on torch CPU"; equivalence
against the original unpatched implementation is held separately by `export_sbv2.py --verify` (the
layers are not mixed). Two gates are applied along with it:

- **Tokenization parity** — the dump's `bertText` is fed to the Python tokenizer and required to
  match the dump's `input_ids` exactly, **before** the waveform comparison. A divergence here
  distributes the BERT features to different phonemes and stays silent in the shape of "sound comes
  out, but distorted".
- **Integer equality of `w_ceil`** — durations use `ceil`, so if the front output sits just above a
  threshold, a 1e-5 GPU/CPU difference shifts a frame. Positions that disagree are reported together
  with the `w` values (in a form that lets the reader judge flake versus implementation difference).

### Why `official` is a separate subcommand

The `patch_sbv2` patches replace class attributes **process-wide**, and `reference` applies them.
What `official` claims is "the sound through the original implementation's g2p (pyopenjtalk) and the
original implementation's attention / spline", so co-hosting it in the same process would silently
put it on the patched path. Since argparse subparsers only allow one choice per process, **one
subcommand per process** holds structurally (the same rationale as `--verify` not carrying a
pairwise exclusivity table).

The 3 wav files (`out.wav` / `reference.wav` / `official.wav`) are written with the same PCM16
conversion rule (clip → `floor(x·32767 + 0.5)`). Python's built-in `round` is banker's rounding, so
without aligning that, an implementation difference would creep into the listening comparison.

## Usage (from a script)

```python
from karume import export_to_file

graph = export_to_file(module, (x,), "model.safetensors", dynamic_shapes=({0: dim},))
```

`export_to_file` runs export → normalize → convert → write → **verify**. It is the gate that keeps a
file that was written but cannot be read by the runtime from being left behind as a distributable,
so the path is never branched.

## Module structure

| Module        | Role                                                                                                                      |
| ------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `dims`        | dimension language `coeff·sym+offset`. The grammar is authoritative in `packages/runtime/tests/fixtures/dim-grammar.json` |
| `ir`          | IR v1 graph representation and JSON serialization (`allow_nan=False`)                                                     |
| `ops`         | op contract table (the counterpart of TS-side `packages/runtime/src/ops.ts`)                                              |
| `shapes`      | output shape rules (the counterpart of TS-side `computeOutputShape`; declared shapes are compared on every node)          |
| `convert`     | ExportedProgram → IR graph (constant folding, aten mapping table, CSE)                                                    |
| `normalize`   | FX equivalence rewrites that do not grow the vocabulary (pass registration)                                               |
| `emit`        | writing to safetensors                                                                                                    |
| `verify`      | all IR v1 rules + distribution-form comparison + runtime capability comparison                                            |
| `pipeline`    | `export_module` / `export_to_file`, the above laid out as one straight path                                               |
| `goldens`     | definition and generation of the tiny golden fixtures                                                                     |
| `patch_sbv2`  | model-specific — the monkeypatch layer and wrappers that **make SBV2 exportable**                                         |
| `patch_anima` | model-specific — the monkeypatch layer and wrappers that raise **Anima's IR quality** (a different motivation)            |
| `lora`        | fusing LoRA weights before the export (the IR does not change by a single node — ADR 0016)                                |
| `resolution`  | the resolution spelling `WxH` (shared by the 2 reference scripts. **The accepted set is authoritative on the demo side**) |

Only `patch_sbv2` / `patch_anima` are model-specific, and the `style_bert_vits2` / `diffusers`
imports are confined **inside functions** (so `import` succeeds even where the packages are absent =
tests of other modules are not dragged along by the dependency groups).

**The two patch layers have different motivations** (ADR 0016): `patch_sbv2` targets "exportability
itself" (without the branch-free form, `torch.export` fails on a data-dependent guard).
`patch_anima` is a quality layer on top of the fact that **all 4 targets export even unpatched** —
it exists for 3 things: not growing the op set, staying within rank ≤ 4, and not baking runtime
knobs into the graph.

### Contract synchronization with the TS side

`ops` and `shapes` are **a different implementation of the same contract** as the TS side
(`packages/runtime/src/ops.ts`), and their agreement is guaranteed not by human discipline but by
the conformance case table `packages/runtime/tests/fixtures/op-contracts.json` (the same relation as
between `dims` and `packages/runtime/tests/fixtures/dim-grammar.json`). The tests on both sides
(`tests/test_ops_conformance.py` / `packages/runtime/tests/ops_conformance_test.ts`) compare **the
table derived from their own implementation** against this file, so moving only one side turns both
red. What it carries: the full set of op names / arity / slot dtypes / the attrs key set / the value
ranges of attrs / output shape rules (including the rank ceiling of the strided-copy family).

`shapes` does **not** take the declared shapes attached by torch's meta as authoritative — it
computes shapes independently from the contract rules and compares them on every node, failing the
export on a mismatch (this runs both at the exit of `convert` and in `verify_model`). Only the
decisions that need bindings (zero-length axes, exceeding Tmax) are held by the runtime-side layer.

## Supported scope (as of perf-a)

**The op count and the contracts are authoritative in
`packages/runtime/tests/fixtures/op-contracts.json`** (both TS-side `packages/runtime/src/ops.ts`
and Python-side `ops.py` compare against it). What follows is a copy, so on any disagreement the
conformance table is the correct one.

- **The semantic dtypes are f32 / i32 / bool** (ADR 0009). torch's i64 is normalized to i32 at the
  exporter boundary (out of range fails loudly). **The storage dtypes are f32 and i32** (i32 is raw
  int32 — the explicit exception of ADR 0010). An initializer's semantic dtype is f32 or i32, and
  the semantic/storage pairs are only `f32 × {f32,f16,bf16,i8}` and `i32 × i32` (the cross products
  fail loudly).
- There are **57** IR ops (ADR 0017 added `rms_norm` / `conv2d` / `clamp_min`, ADR 0023 added
  `attention`, `gelu_tanh` was added for EmbeddingGemma, `sin` for the Snake activation,
  `safe_softmax` for runtime attention masks — ADR 0044 — `upsample_bilinear2d` for the
  segmentation / depth family, `deform_conv2d` for the BiRefNet family — ADR 0055 — and
  `gru_scan` / `gru_scan_reverse` for recurrent models — ADR 0056):
  - unary `neg abs exp log log1p sqrt sin tanh sigmoid relu gelu gelu_tanh` (f32) / `bitwise_not`
    (bool) / unary with attrs `clamp` / `clamp_min` / `leaky_relu` (f32). `sin` is the **only**
    trigonometric op: constant tables (RoPE) are still folded away at export time, so only the
    runtime-valued form needs a kernel, and `cos` is not added for symmetry alone
  - scalar comparison `ge_scalar le_scalar gt_scalar` (f32 → bool)
  - binary `add div` (f32) / `mul sub` (f32, i32) / `ge` (f32 → bool) / `bitwise_and` (bool) and the
    ternary `where` (cond is bool) — all with torch's right-aligned broadcast
  - `cast` (among f32 / i32 / bool) / `matmul` (rank-2) / `bmm` (rank-3) / `gather` (fixed to the
    last dim) / reduce `sum amax amin` (**1 axis**, the `dim` attr is mandatory, no keepdim; `sum`
    also accepts bool input → i32) / `cumsum` (last dim)
  - layout (ADR 0011 / 0014): `reshape` / `permute` / `expand` (f32 unlocked as well) / `slice` /
    `cat` (**the only variadic-arity op in IR v1**) / `pad` / `flip`
  - symbolic prefix slice (ADR 0010): `sym_prefix_slice`
  - fused ops (ADR 0012 / 0015 / 0017 / 0023): `linear` / `layer_norm` / **`rms_norm`** /
    `softmax` / **`safe_softmax`** / **`attention`** / `embedding` / `masked_fill` / `conv1d` /
    **`conv2d`** / `conv_transpose1d`. `safe_softmax` is `softmax` plus “a row whose max is −inf
    is written as all zeros”, i.e. the semantics of the safe-softmax guard that torch's SDPA
    decomposition wraps around `softmax` (ADR 0044)
  - spatial resample (a layer-1 atom): **`upsample_bilinear2d`** — `x[B,C,H,W] →
    [B,C,Hout,Wout]`, arity 1, attrs `output_size` only. **`align_corners=True` only**: there is no
    field for `align_corners` / `mode` / `scale_factor`, so half-pixel alignment, nearest / bicubic /
    area / antialias and factor-based sizing all fail loudly at the exporter boundary. Shrinking
    goes through the same op (torch does too — reading only 2 taps is the specified behaviour, and
    is not `area`)
  - deformable convolution (a layer-1' atom, ADR 0055): **`deform_conv2d`** — DCNv2 with arity 5
    (`x[B,Cin,H,W]` / `W[Cout,Cin,Kh,Kw]` / `offset[B,2·Kh·Kw,Hout,Wout]` /
    `mask[B,Kh·Kw,Hout,Wout]` / `b[Cout]`), attrs `padding` only. The offset channels are nested as
    `(kh, kw)` with the innermost pair being **even = y, odd = x**; the modulator is applied
    **after** the bilinear interpolation; out-of-range samples are **zero-filled, not clamped**.
    There is no field for `stride` / `dilation` / `groups` / `offset_groups` (all fixed to 1) and
    the mask is a mandatory slot, so DCNv1 (`use_mask=False`) fails loudly at the exporter boundary.
    A NaN offset propagates to the output instead of collapsing to a zero contribution
  - recurrent scan (layer-2 molecules, ADR 0056): **`gru_scan`** / **`gru_scan_reverse`** — the
    hidden-side scan of a GRU with arity 4 (`gi[T,N,3H]` / `h0[N,H]` / `W_hh[3H,H]` / `b_hh[3H]`)
    and **no attrs**, producing `y[T,N,H]`. The **input-side GEMM is not part of the op**: the
    caller prepares `gi` with the existing `linear`, which keeps its f16 / i8 storage eligibility.
    The time axis may stay symbolic — that is the whole point, since decomposing `aten.gru`
    unrolls it T times and forces `torch.export` to specialize the length. One step is
    `gh = W_hh·h + b_hh` / `r = σ(gh_r + gi_r)` / `z = σ(gh_z + gi_z)` / `n = tanh(gi_n + gh_n·r)` /
    `h' = (h − n)·z + n`, and **that exact operand order and grouping is the contract** (the
    algebraically equivalent `(1 − z)·n + z·h` is a different rounding sequence). The reverse op
    only reverses the **scan order** — its output is still in forward time order, because `flip`
    does not accept a symbolic axis. There is no field for stacking, bidirectionality,
    `has_biases=False`, `batch_first` or `dropout` (layers and directions are expressed by placing
    several nodes), the op returns `y` only (no `h_n`), and the hidden width is capped at 256
- **29 ops carry attrs** (`sum.dim` / `amax.dim` / `amin.dim` / `attention.scale` /
  `clamp.{min,max}` / `clamp_min.min` / `rms_norm.eps` / `conv2d.{stride,padding,dilation,groups}` /
  `leaky_relu.negative_slope` / `ge_scalar.value` / `le_scalar.value` / `gt_scalar.value` /
  `cumsum.dim` / `cast.to` / `permute.dims` / `slice.{dim,start,end}` / `cat.dim` /
  `pad.{left,right}` / `flip.dim` / `sym_prefix_slice.{sym,slices}` /
  `layer_norm.{normalized_shape,eps}` / `softmax.dim` / `safe_softmax.dim` /
  `embedding.padding_idx` /
  `masked_fill.value` / `conv1d.{stride,padding,dilation,groups}` /
  `conv_transpose1d.{stride,padding}` / `upsample_bilinear2d.output_size` /
  `deform_conv2d.padding`). Every declared key is
  mandatory, and undeclared keys or
  out-of-range values fail loudly (ADR 0012). **Defaults are never filled in** — being able to omit
  `dilation` / `groups` on `conv1d` would silently turn a depthwise IR into an ordinary convolution
  (ADR 0015). `gelu(approximate="tanh")` has no field to record it either, so it is carried by its
  own op (`gelu_tanh`) rather than by an attr (never silently approximating with a different
  formula).
- The default set of decomposition stops (preserved) is **11 ops** (`PRESERVED_OP_PREFIXES` — the 9
  ops of ADR 0007 plus `leaky_relu` from ADR 0015 and `rms_norm` from ADR 0017): linear / layer_norm
  / rms_norm / softmax / gelu / leaky_relu / conv1d / conv2d / conv_transpose1d / embedding /
  masked_fill.
  - **The 12th, `scaled_dot_product_attention`, is not in the default set** (ADR 0023). SDPA can
    express mask / causal / GQA through its arguments, and adding it to the default would make
    Anima's text_encoder (a causal mask with −inf folded in) hit the fail loudly of `_h_attention`
    and become **unexportable**. Enabling it is a **per-target opt-in** via
    `export_module(…, preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION)`
    (`export_anima.TARGET_PRESERVED` — currently only transformer and vae_decoder).
  - **`rms_norm` arrives through 2 routes** (ADR 0017): the `aten.rms_norm` coming from diffusers'
    `nn.RMSNorm` survives preservation, while the hand-written decomposed form (Qwen3 / DiT) cannot
    be folded by preservation and is handled by `normalize._fold_rms_norm`.
- The narrowness of the fused ops' kernel contracts lives on the runtime capability side, not in the
  IR (ADR 0007). What the exporter boundary rejects is only **the axes for which no representation
  exists**: a multi-axis `normalized_shape`, a softmax on anything but the last dim, a non-finite
  `masked_fill` fill value, a clamp with only an upper bound (`clamp_max` is not in the vocabulary),
  a conv_transpose1d whose `output_padding` / `groups` / `dilation` are not the defaults, and a
  conv_transpose1d with `2·padding ≠ K − stride` (a form whose output length is not `L·stride`).
  - `groups` / `dilation` on `conv1d` / `conv2d` **are accepted** (ADR 0015 / 0017).
  - **Optional slots are synthesized to fix the arity** (`Emitted.synth_consts`) — a conv / linear
    without bias gets a zero bias, a layer_norm without affine gets ones/zeros, and an rms_norm
    without weight gets ones. This keeps arity branching out of the kernels and the contracts (`+0`
    / `×1` are exact identities — ADR 0015 / 0016). Measured weight: 698 of Anima's 711 linears have
    no bias, and all 85 layer_norms of the DiT have no affine.
  - A `clamp(min=eps)` with only a lower bound lowers to a **separate op** (`clamp_min`) (ADR 0017).
    Filling the missing side with ±the largest finite value would amount to "silently executing a
    form that has no representation as a different form", so it is not adopted.
- Unsupported aten ops are **enumerated in full** before failing (never stopping at the first one).

### Constant folding (Tmax folding and two-point evaluation — ADR 0010)

Subtrees that depend only on constants and shape symbols are **actually evaluated at each symbol's
upper bound Tmax**, baked into an initializer, and sliced from the front at runtime with
`sym_prefix_slice`. The relative-position bucket tables (`arange / sign / log / ceil / clamp /
comparison / where`) disappear wholesale through this path, structurally eliminating the bug class
where a 1ulp difference at a bucket boundary becomes a 1-off gather index.

- **Tmax comes from `ExportedProgram.range_constraints`** (= exactly the constraints that the
  `Dim(min=…, max=…)` of `dynamic_shapes` registered with torch). It is not received from the caller
  as a separate argument — that would be dual bookkeeping with `dynamic_shapes`, and on a
  disagreement "a constant baked shorter than the declaration, out of range at runtime" would pass
  silently. An unset `Dim(max=…)` (`int_oo`) fails loudly.
- **Eligibility is measured by two-point evaluation, not by an allowlist**
  (`_check_prefix_commutes`). The first point is Tmax and the second is **Tmax − 1** (which must be
  at least 2 to avoid 0/1 specialization; the range where `Tmax − 1 < max(lower bound, 2)` fails
  loudly as "the check would become vacuously true"). The result of the second evaluation is
  byte-compared against the prefix (of length `coeff·sym+offset`) of the constant baked at Tmax, and
  **if they disagree it is not folded but rejected**. Forms that use T "as a value", such as
  `arange(T)/T` or `full((T,), T)`, can be built entirely out of allowlisted ops, so only this
  measurement can stop them.
- The dtype of a baked constant is f32 or i32 (i64 goes to i32 through boundary normalization; out
  of range fails loudly). bool constants are rejected because there is no initializer vocabulary for
  them.
- `expand` / `repeat` are not on the allowlist (folding them would materialize B·H times over) —
  they stop at the frontier and are handled by a strided copy at runtime.

### Normalization passes (`normalize.py`)

They run in registration order. Only equivalence rewrites that grow neither the IR vocabulary nor
the attrs are placed here.

| Pass                       | Rewrite                                                                | Firing condition (untouched otherwise)                                              |
| -------------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `_drop_metadata_asserts`   | remove `_assert_tensor_metadata`                                       | always                                                                              |
| `_fold_rms_norm`           | `x·rsqrt(mean(x²)+eps)·w` → `rms_norm`                                 | weight is rank-1 of the last-dim length, eps a finite positive                      |
| `_drop_safe_softmax_guard` | remove the safe-softmax guard of SDPA, or rewrite it to `safe_softmax` | removed **only when inactivity can be proven**, otherwise rewritten (ADR 0044)      |
| `_lower_unit_expand`       | `unsqueeze→expand→view` → expand of rank ≤ 3                           | rank > `STRIDED_RANK`, the replicated axes static                                   |
| `_lower_split_unbind`      | width-1 slice+squeeze of a last-dim split → last-dim slice             | rank > `STRIDED_RANK`, the split contiguous and static                              |
| `_lower_reshape_permute`   | rank ≥ 5 reshape→permute→reshape → a rank-4 transpose sequence         | rank > `STRIDED_RANK`, endpoints of rank ≤ 4                                        |
| `_collect_dead_code`       | one DCE in the middle (**the position is meaningful**)                 | always                                                                              |
| `_pow2_to_mul`             | `pow(x, 2)` → `mul(x, x)`                                              | the exponent is exactly 2                                                           |
| `_drop_identity_repeat`    | `repeat(x, [1,…,1])` → `x`                                             | **all arguments 1 and their count = rank** (raising the rank is a different matter) |
| `_drop_identity_add`       | `add(x, 0)` → `x`                                                      | a Python scalar 0 and the dtype unchanged                                           |
| `_promote_scalar_operands` | `add/sub/mul/div(tensor, scalar)` → binary op + constant               | dtype unchanged, rank ≥ 1, `alpha=1`, a finite scalar                               |
| `_eq_zero_to_not_bool`     | `eq(x, 0)` → `bitwise_not(cast(x, bool))`                              | the right side is exactly 0 and the output is bool                                  |
| `_select_to_squeeze`       | `select.int(axis of length 1, 0)` → `squeeze`                          | **statically** of length 1 (symbolic axes are excluded)                             |
| `_split_to_slices`         | `split_with_sizes` + `getitem` → a sequence of `slice`                 | the only consumers are getitem, the split axis static                               |

- **Three orderings are load-bearing**:
  - `_fold_rms_norm` comes **before** `_pow2_to_mul` / `_promote_scalar_operands` (ADR 0016). Once
    eps is promoted to a rank-1 constant, the scalar match comes off and it can no longer be folded.
  - `_drop_identity_add` comes before `_promote_scalar_operands`. In the reverse order, `+0` becomes
    a rank-1 constant and is no longer a candidate for identity removal.
  - `_collect_dead_code` comes **immediately after** the 3 rank-lowering passes. Without deleting
    the folded source patterns here (pow/mean/rsqrt, the guard's eq/any/logical_not), later passes
    would rewrite dead subtrees and inflate the statistics alone (the IR does not change, so it
    stays green and unnoticed).
- **The 3 rank-lowering passes only fire for `rank > STRIDED_RANK`** (the safety line of ADR 0016).
  They do not misfire on existing graphs (SBV2 / DeBERTa are all rank ≤ 4) — they create no new
  constants and take "a rank the strided kernel cannot execute" as the firing condition itself.
  Measured: with this wave's additions, **the IR of the 22 tiny goldens and of SBV2 / DeBERTa is
  unchanged byte for byte**.
- **Guard removal comes with a proof** (ADR 0016): only when either ① the dependency cone contains
  no -inf source, or ② the -inf comes from a single additive mask and evaluation at two symbolic
  lengths (5 / 9) leaves a finite element in every row. If neither holds, `NotImplementedError`
  (removing it would let NaN flow downstream).
- Promoted constants are inserted as `aten.full.default([1], value, dtype=…)`. That op is on the
  folding allowlist, so if the consumer is a folding candidate the constant is folded along with it,
  and otherwise it becomes a rank-1 initializer (an i32 initializer for i32) absorbed by broadcast.
- **The constant is placed on the side the original scalar was on.** `sub` / `div` are
  non-commutative, and the measurements include forms where the scalar is on the left, such as
  `sub.Tensor(1, mask)` (`1 − attention_mask`).
- Forms that involve type promotion (`div(i64, 128)` returns f32) and rank-0 tensors (`[] × [1] →
  [1]` raises the rank) are **not rewritten** — they go to folding or to the enumeration of
  unsupported ops.

## Not there yet (later phases)

- A Python interpretation oracle for the IR (numerical comparison). Numerical verification is
  handled by the Deno-side E2E.
- **bf16 storage** (it is in the IR vocabulary but has no execution path — ADR 0006). f16 is
  implemented in ADR 0018 and i8 + per-channel scale in ADR 0019. w4 (group quantization) is
  confirmed as not adopted.
- **Mixed storage** (mixing i8 and f16 within one target). `_apply_weight_dtype` applies a single
  `weight_dtype` to everything.
- **A real-GPU comparison of Anima's `transformer` in f32 storage at the full 28 layers** (the
  7,465MiB of weights exceed this machine's GPU buffer limit of 7,280MiB and cannot be loaded —
  `docs/known-issues.md`). The real-GPU E2E does run for f16 storage (3,733.5MiB) and i8 storage
  (1,872.8MiB).
- Host implementations of the runtime pipelines (SBV2: text → durations → assembling y_mask → voice
  / Anima: tokenization → scheduler → CFG → denormalization). The layer that connects the emitted
  targets is outside the exporter's scope (for Anima there is a TS implementation for testing in
  `packages/runtime/tests/e2e_anima_test.ts`). (The **TS port of Anima's tokenizer** is complete, in
  `anima_demo.py` + `examples/anima/text/`. `packages/runtime/tests/e2e_anima_test.ts` keeps using
  the fixture's `input_ids` — what it measures is the NN's numbers, and tokenization parity is held
  separately by `packages/runtime/tests/anima_tokenizer_test.ts`.)

## Real-weight Irodori-TTS export and E2E (waves 1–4)

Six text-side graphs plus the DACVAE codec pair, with host-side goldens. The scripts have a
**required regeneration order** (later ones read earlier outputs):

```sh
# 0. one-time inputs: inputs/irodori/{v4-small,dacvae-32dim,Irodori-TTS,dacvae-src}/
uv run python convert_dacvae.py                                       # 1. codec pth → safetensors
uv run --with 'transformers==5.14.1' python export_irodori.py         # 2. six graphs + io goldens
uv run --with 'transformers==5.14.1' python irodori_tokenizer.py      # 3. tokenizer asset + goldens + parity fixture (deno fmt it)
uv run --with 'transformers==5.14.1' python irodori_pipeline.py       # 4. full-loop latent goldens
uv run --with descript-audiotools --with einops \
    --with 'transformers==5.14.1' python dacvae_host.py               # 5. host preprocessing goldens
uv run --with descript-audiotools --with einops python export_dacvae.py  # 6. codec graphs + io goldens
uv run karume dist --pipeline irodori                                 # 7. distribution (8 graphs + tokenizer)
```

The distribution carries an **f16 weight-storage series** next to f32 (ADR 0050), so step 7 needs
both series. Regenerate the f16 side with the same three scripts (order caveats are identical; the
codec / full-loop inputs stay on the f32 series on purpose — inputs are dtype-neutral):

```sh
uv run --with 'transformers==5.14.1' python export_irodori.py --dtype f16     # 2'. six graphs
uv run --with 'transformers==5.14.1' python irodori_pipeline.py --dtype f16   # 4'. full-loop goldens
uv run --with descript-audiotools --with einops python export_dacvae.py --dtype f16  # 6'. codec
```

Order caveats measured in practice: step 2 reads step 5's real latent for the speaker cases
(`SPEAKER_REAL_CASES`), and step 6 reads step 4's `z` for the decoder cases — so a **full** rebuild
from scratch runs 2 once more after 5 (2 → 3 → 4 → 5 → 2 → 6 → 7). Incremental regeneration of a
single script is safe as long as its inputs above exist. Design records: ADR 0044 / 0046 / 0047
(graphs), 0048 (host port), 0049 (codec integration).

## Vowel-detector CRNN export (one graph, symbolic length)

A small CRNN that turns 10 ms speech features into 8-class lip-sync logits: two Conv1d layers
(the first with stride 2, which halves the time axis), a 2-layer bidirectional GRU with hidden
size 128, and a linear head — 664,744 parameters in total. Feature extraction (80 log-mel bins
plus 3 DSP dimensions) and post-processing (log-softmax, penalised Viterbi, short-segment merge,
`.lab`) both stay on the host; only the network is a graph.

```sh
# one-time input: inputs/vowel-detector/crnn_epoch3.pt (upstream training checkpoint)
uv run karume export-vowel-detector             # → outputs/series/vowel-detector-crnn-epoch3/
uv run karume export-vowel-detector --verify    # gru_scan rewrite vs nn.GRU (bit-exact, 5 lengths)
```

No dependency group is needed — `torch` is a base dependency, and the 20-line model definition is
**transcribed verbatim** into the script instead of importing the upstream `vowel_detector`
package (whose import chain pulls in pyopenjtalk and librosa for G2P and feature extraction, none
of which the export touches). `load_state_dict(strict=True)` is what keeps the transcription
honest; `tests/test_export_vowel_detector.py` pins the 22 parameter names and shapes so a drifting
transcription fails without the real weights present.

### The length is symbolic — the GRU is rewritten into scan nodes

`aten.gru.input` survives `torch.export` as a single node, but `run_decompositions` unrolls it
**along time**, which specialises the graph on T — asking for a dynamic `Dim("T")` fails with
`Specializations unexpectedly required (T)`, and T10 = 200 produced 8,434 nodes. So the script
rewrites `nn.GRU` into `karume::gru_scan` / `gru_scan_reverse` calls before tracing
(`karume/patch_vowel_detector.py`, ADR 0056): the input-side GEMM stays a plain `linear` over the
whole time axis, and each layer and direction becomes **one scan node**, joined by `cat`.
Everything else is the upstream `forward`.

The symbol is declared on the **output** grid, as `2*Dim("T")` on the input axis. A bare `Dim("T")`
would make the first Conv1d (kernel 5, stride 2, padding 2) produce `((T−1)//2)+1` — a floor
division, which the dimension language (`coeff·sym+offset`) cannot express. With `2*Dim("T")` the
conv output is exactly `T`. The consequence is a runtime contract: **the number of 10 ms input
frames is always even** (callers drop the odd tail frame, which the 20 ms output grid has no room
for anyway). Binding `T` from the derived `2T` axis is ADR 0057.

| what                | before (T10 = 200, unrolled) | now (symbolic)  |
| ------------------- | ---------------------------- | --------------- |
| IR nodes            | 8,434                        | **18**          |
| depends on length   | yes (38 nodes per frame)     | **no**          |
| `model.safetensors` | 3,892,256 B                  | **2,668,608 B** |
| export time         | 8.1 s                        | **0.6 s**       |
| initializers        | 23                           | 23              |

(Measured 2026-08-14, torch 2.13.0+cpu, on the real checkpoint. The 23 initializers are the 22
checkpoint tensors plus one folded constant — the zero initial hidden state. The weights are
2,658,976 B, so the graph JSON is now under 10 KB instead of growing by ~5.27 KB per input frame.)

The rewrite is **bit-exact against `nn.GRU`**, which is the whole basis for keeping the golden
values: `--verify` compares the two eager paths at five lengths, and every emit compares them again
per golden case before writing the expected outputs (see the gates below).

**Right zero-padding a short utterance into a longer graph does not work** — which is why no part
of this pipeline pads any more. The backward GRU carries state home from the padding: padding a
true length of 137 frames into a 500-frame graph measures a max abs diff of 5.91 and an argmax
agreement of 0.971, with the error concentrated at the tail (0.138 at the head, 5.915 at the end).
A unidirectional model would only see the conv window edge (5.1e-03).

### Gates that run on every emit

- **Weight conversion**: every one of the 22 checkpoint tensors is read back out of the emitted
  container with an independent reader and compared **byte for byte** (`assert_checkpoint_bytes`);
  any initializer that is neither a checkpoint tensor nor a folded `const.` constant fails loudly,
  so a weight cannot come back under another name.
- **Sanity** (orderings only, no thresholds): the silence-like case has the highest mean P(pau)
  of the four, and the voiced-like case the highest mean vowel mass — plus all four outputs must
  differ from one another.
- **Rewrite equivalence**: the expected outputs written into `io.<case>.safetensors` are taken from
  the **reference path** (`nn.GRU` itself), and the rewritten path is compared against them with
  `torch.equal` for every case. Taking the expected values from the rewritten path instead would
  let an exporter bug and a runtime bug agree with each other and both stay green.
- **`--verify`**: the same comparison at five lengths (T10 = 4 / 6 / 18 / 274 plus `--length`),
  measured 0.0 on the real checkpoint. Length-dependent mistakes (the backward scan's boundary, the
  order of the per-layer `cat`) do not all show up at a single length.

### Distribution form (`karume dist --pipeline vowel-detector`)

```sh
uv run karume export-vowel-detector                 # one graph, any length
cp <upstream>/assets/feature_config.json ../../inputs/vowel-detector/
uv run karume dist --pipeline vowel-detector        # → models/karume-vowel-detector/
```

The repository ships **one graph** (`crnn`, 2,668,608 B) plus the mel filterbank as an `assets`
entry, because feature extraction happens off-graph and cannot be reproduced without the exact
80 × 257 matrix the model was trained on. **2,759,461 B in total** — the four length buckets it
replaces were 34,088,454 B (−91.9%), since each bucket was a whole container, weights included.

The clip runs at its own length: no padding, no bucket, and therefore no length-dependent numbers.
That also removes a discrepancy the bucketed form had no gate for — the padded path disagreed with
the exact-length path on 3 of the 4 real utterances (a 20 ms boundary shift, a 40 ms `pau`
appearing at the tail, a 40 ms `pau` appearing mid-utterance), because the backward GRU reads state
back out of the padding. `packages/models/tests/e2e_vowel_detector_lab_test.ts` now pins the two
paths to be byte-identical.

`pipelineConfig` comes from two independent places and they are checked against each other before
anything is placed: the feature contract (`sampleRate` / `featureDim` / `classes`) is read verbatim
from the upstream `feature_config.json`, while `maxFrames` is the symbolic upper bound the export
script baked (`SYM_MAX`, in 20 ms frames, doubled). The IR carries symbol _names_ but no ranges, so
the limit exists only in the manifest — the assembly gate checks instead that the graph really is
symbolic (`2T` in, `T` out); a graph baked for one fixed length would otherwise assemble fine and
fail in the user's hands for every clip but one.
