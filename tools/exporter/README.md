# Karume exporter (core)

Python tooling that lowers `torch.export`-ed models into Karume's **IR v1**
([../../docs/ir-v1.md](../../docs/ir-v1.md)). Managed with uv; CPU-only torch (no GPU required).

The distribution form is a single safetensors file — tensors (weights and constants) plus the graph
JSON under the `__metadata__` key `karume_ir`.

The PyPI distribution `karume` is the **generic exporter core only**: dims / ir / ops / shapes /
convert / aten_handlers / normalize / quantize / act_quant / emit / verify / pipeline / goldens /
golden_models, plus the generic dist engine and the generic model-card renderer. Model-specific recipes — patch layers, export
scripts, reference pipelines, dist recipes, card templates and their provenance — live outside the
wheel in [`../export-recipes/`](../export-recipes/README.md), and the dependency direction is
**recipe → core only** (ADR
[0065](../../docs/decisions/0065-exporter-core-recipe-split.md), enforced by
`tests/test_architecture_boundary.py`).

## Setup

```sh
uv sync            # run in tools/exporter/ (pulls CPU torch from the pytorch-cpu index)
```

`torchvision` is a **base** dependency, not an optional group: `torchvision::deform_conv2d` (the
layer-1' op of ADR [0055](../../docs/decisions/0055-deform-conv2d.md)) is registered by importing
the package, so both the handler key (`torch.ops.torchvision.deform_conv2d.default`) and the tiny
golden that covers the op need it. Making it conditional would split environments into "rejects
`deform_conv2d` as an unknown op" and "accepts it".

`src/karume/custom_ops.py` registers this project's own `karume::` operators (`gru_scan` /
`gru_scan_reverse` — ADR [0056](../../docs/decisions/0056-gru-scan.md)) with
`torch.library.custom_op`, so **importing it has a process-wide side effect**. `aten_handlers.py`
imports it at the top for the same reason it imports `torchvision`: the handler key
(`torch.ops.karume.gru_scan.default`) cannot be written otherwise. Hiding the body behind
`register_fake` is what keeps the time axis symbolic — tracing never enters the Python loop, so the
op survives `run_decompositions` as a single node instead of unrolling T times.

## Usage (from a script)

```python
from karume import export_to_file

graph = export_to_file(module, (x,), "model.safetensors", dynamic_shapes=({0: dim},))
```

`export_to_file` runs export → normalize → convert → write → **verify**. It is the gate that keeps a
file that was written but cannot be read by the runtime from being left behind as a distributable,
so the path is never branched.

## CLI (`karume`)

The subcommands installed by `[project.scripts]`. **The CLI does not interpret arguments** —
everything after the subcommand name is passed straight to the body (`karume <subcommand> --help`
prints the usage of the body's own parser). This shape keeps a copy of the argument rules out of
the CLI — for example `karume dist --card-profile`, which is required only when the chosen pipeline
offers more than one attribution profile, a rule the body derives from the registry it is handed.
Dispatch is a lazy import.

| Subcommand      | Wrapped body                                                                            |
| --------------- | --------------------------------------------------------------------------------------- |
| `karume dist`   | `karume.dist` (assembles the distribution form from the pipeline registry it is handed) |
| `karume verify` | `karume.verify` (validates the distribution form against every IR v1 rule)              |

There is no `export-*` among them: every export script left the wheel for
`tools/export-recipes/<family>/`, and so did the loader that used to read a script by path (ADR
0065). `karume dist` runs on the registry it is given — `karume.dist.PIPELINES`, **empty** in the
wheel — so assembling a family goes through the repository driver
[`tools/export-recipes/dist.py`](../export-recipes/README.md), which injects the family pipelines
and the repository's own spellings for `--series` and `--out`. The seat stays here because the
assembly engine itself is core (ADR 0065 decision 1).

```sh
uv run karume verify ../../models/karume-anima-turbo/anima-turbo/transformer/model.f16-00001-of-00004.safetensors
```

### `karume dist` — the assembly engine

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
uv run python -m karume.goldens --out ../../packages/runtime/tests/fixtures/golden
uv run python -m karume.goldens --out /tmp/golden      # any other output directory
```

**`--out` is mandatory.** Which repository's `packages/runtime/tests/fixtures/` the goldens land in
is repo topology, not knowledge a generic exporter carries (ADR 0065); the path above is the
spelling for a run from `tools/exporter/`.

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
| `argmax_pick`              | `T`          | argmax (greedy exit: rank-preserving over the last dim, i32 output), linear, cat — `cat([t, t])` forces the minimum-index tie-break                                    |
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
| `rms_norm_block`           | `T`          | rms_norm ×3 (hand-written `x·rsqrt(mean(x²)+eps)·w` folded into one node + `nn.RMSNorm` with and without affine), layer_norm(no affine), linear(no bias)               |
| `conv2d_block`             | none         | conv2d ×3 (Kh≠Kw / asymmetric stride, padding and dilation / groups 3 / one branch with no bias), sum(channel axis), sqrt, clamp_min, div, mul, reshape                |
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

`rms_norm_block` covers the **two supply routes** of `rms_norm` (ADR 0017) in one graph: the
hand-written `x · rsqrt(mean(x²) + eps) · weight`, which only becomes one node because
`normalize._fold_rms_norm` matches it, and `nn.RMSNorm`, which survives as `aten.rms_norm` because
it is preserved. The affine-free `nn.RMSNorm` and the affine-free `LayerNorm` are the goldens for
the **synthesized optional slots** (ones / ones+zeros — ADR 0016), and the `bias=False` linear is
the second zero-bias synthesis after `conv_transpose`. All three eps values are **different**
(1e-6 / 1e-5 / 1e-7): with one shared eps, dropping eps from attrs and falling back to a default
does not show up in the values at all.

`conv2d_block` makes every axis of the 2-D weight layout distinguishable, because reading
`[Cout,Cin,Kh,Kw]` with H and W swapped still gives a matching element count and passes the shape
check (the same failure mode as `conv_transpose`): **Kh ≠ Kw, stride / padding / dilation
asymmetric between H and W, Cin ≠ Cout and both above 1**, with the input plane H ≠ W as well. It
also carries the only **axis reduction** (`sum` with `dim=1`, the channel L2 of the Anima host
mirror) among the goldens — every other `sum` here is over the last dimension — and the
`example_inputs` zero out one channel column so that the `clamp_min` floor is what keeps the
division off `0/0`.

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

## Compressed weight storage (f16 / i8 / i4 — ADR 0018 / 0019 / 0069)

Storage compression is emit-side and model-independent: which initializers may be stored
compressed, how the fake-quant is defined, and in what order the tensors are written. The
per-model measurements these rules were derived against stay with the recipes (for example
[`../export-recipes/anima/README.md`](../export-recipes/anima/README.md)).

### Eligibility and write order

**Eligibility is the AND of 2 conditions** (`src/karume/emit.py`):

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
to the output of `save_file` (confirmed on the 29 f32 tiny goldens — the f32-series assets do not
move by a single byte when the writer is swapped; only `i8_weights`, the 30th, uses compressed
storage).

### Per-channel int8 (`--dtype i8`)

**Definition of the quantization** (`src/karume/quantize.py`):

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
  namespace.
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

### Group-wise int4 (`weight_dtype="i4"` — ADR 0069)

**Eligibility is narrower than f16 / i8**: on top of the AND of 2 conditions above, an i4 initializer
must be consumed **only by the weight slot of `linear`** (`linear_weight_initializers`). The
execution path starts at linear, so an initializer that is also consumed by another weight slot
(`embedding` / the conv family) cannot be stored as i4. With the **default** `weight_dtype="i4"` such
a weight silently stays f32 — the same landing pad as an i8-ineligible weight, and the counterpart of
the runtime's `eligible ∩ linearOnly`; without it an ordinary LLM (linear + embedding) could not be
exported at all. An **explicit** i4 on a non-linear weight fails loudly instead (see below).

**Definition of the quantization** (`src/karume/quantize.py`, `fake_quant_int4`): symmetric int4
along the K (input) axis, per group of `group_size` elements —
`scale = clamp(amax_group / 7, f32 tiny)` and `q = clamp(round(w/scale), ±7)`. **−8 is not used**, so
the largest-magnitude element of a group lands on `q = ±7` and is restored exactly, which makes the
fake-quant **idempotent**. The target is `nn.Linear.weight` by default (bias and norm weights are
never touched); `op_types` opts in to the wider set the i8 path uses (`QUANT_MODULE_TYPES` — conv
family and embedding), where a group runs along the flattened receptive field of one output channel.
That widening is for **measuring** quantization error: only the linear entries of the returned ledger
can be handed to `write_model`. `group_size` defaults to **32** and must be a **power of two ≥ 16**;
the quantized axis has to be divisible by it. 0 targets fails loudly with `QuantizeError`.

**Packing order** (`emit.pack_int4` is authoritative, and `tests/test_emit.py` pins it by byte
value): two elements that are **adjacent in flat index** share one byte, element `2i` in the **low**
nibble and `2i+1` in the high nibble, each stored as the offset-8 unsigned nibble `u = q + 8`. This
is deliberately **not** llama.cpp's Q4_0 split-half layout — mixing the two up produces a container
whose shapes and types still match, so only this rule and its byte-level test stand between the two.
An odd element count fails loudly (the last element would stick out by half a byte).

**Companion scales**: like i8, an F32 tensor named `karume.scale.<weight key>` goes into the same
file and is declared through `storage.scale`, but the shape is the **group form** — same rank as the
weight with the last dimension replaced by the group count (`[…, K/group_size]`) — and
`storage.group_size` is declared alongside it. The scale is the one the fake-quant used, verbatim.
The inverse-transform check runs on the **stored bytes** (`dequantize_int4(unpack_int4(packed))`),
because a mistaken pack order is otherwise a silent wrong-value bug.

**Ordering**: an I4 data section is always a multiple of 8 bytes, so it belongs to the 4-byte-aligned
group and goes with F32 / I32 (ADR 0069 addendum 2):

    F32 (name ascending) → I32 → **I4** → even-count F16 → odd-count F16 → I8 (last)

### Mixed storage (`weight_dtype_overrides` — ADR 0069 addendum 4)

`export_to_file` / `write_model` take `weight_dtype_overrides` (**tensor key (FQN) → storage dtype**),
which takes precedence over the single default `weight_dtype`. This is what an LLM needs: "embedding
i8, linear i4" (first used by Gemma 4 E2B — `../export-recipes/gemma4/`). Pass the merged i8 + i4
ledgers in one `weight_scales` mapping (the key space is the FQN, so they never collide). On the
fake-quant side, `fake_quant_int8` / `fake_quant_int4` take an `include` predicate over module FQNs
so that each weight is rounded exactly once — rounding one weight through both would leave the scale
ledger disagreeing with the actual values.

Where the default `weight_dtype` **silently leaves ineligible weights as f32**, an explicit override
**fails loudly whenever it cannot be honoured** — the caller wrote the intent one tensor at a time,
so there is no room for silently choosing another storage. The 4 branches:

| Situation                          | Result                                                             |
| ---------------------------------- | ------------------------------------------------------------------ |
| the key is no initializer's tensor | `EmitError` (a typo would silently drop the requested compression) |
| the initializer is not eligible    | `EmitError` (consumed outside a weight slot)                       |
| the tensor is not f32              | `EmitError` (compressed storage takes f32 values only)             |
| `"i4"` on a non-linear weight      | `EmitError` (the i4 eligibility above)                             |

An explicit `"f32"` is the opposite direction: it **exempts** one tensor from a compressed default.
That is mandatory for the RoPE position tables of the decode series, which land in an `embedding`
weight slot and would otherwise be rounded by an i8 default — unlike weight rounding, angle error
there accumulates along the position axis.

## Module structure

| Module          | Role                                                                                                                          |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `dims`          | dimension language `coeff·sym+offset`. The grammar is authoritative in `packages/runtime/tests/fixtures/dim-grammar.json`     |
| `ir`            | IR v1 graph representation and JSON serialization (`allow_nan=False`)                                                         |
| `ops`           | op contract table (the counterpart of TS-side `packages/runtime/src/ops.ts`)                                                  |
| `shapes`        | output shape rules (the counterpart of TS-side `computeOutputShape`; declared shapes are compared on every node)              |
| `convert`       | ExportedProgram → IR graph engine (graph traversal, constant folding, CSE — dispatches to the handler table)                  |
| `aten_handlers` | the aten op → IR mapping table (per-op handlers + fused ops; the growth point when a model family is added)                   |
| `normalize`     | FX equivalence rewrites that do not grow the vocabulary (pass registration)                                                   |
| `emit`          | writing to safetensors                                                                                                        |
| `verify`        | all IR v1 rules + distribution-form comparison + runtime capability comparison                                                |
| `pipeline`      | `export_module` / `export_to_file`, the above laid out as one straight path                                                   |
| `goldens`       | the golden spec table and generation driver                                                                                   |
| `golden_models` | the tiny golden fixtures themselves — `nn.Module` definitions and input generators                                            |
| `quantize`      | weight fake-quant for the storage dtypes (f16 rounding / per-channel symmetric i8)                                            |
| `act_quant`     | per-token symmetric i8 fake-quant for activations (the torch mirror of the w8a8 execution path)                               |
| `extents`       | identity of dimension lengths — the one place symbolic (`SymInt`) lengths are compared without a guard                        |
| `rope`          | model-independent export check that RoPE frequency buffers were lifted out to constant-folding leaves                         |
| `custom_ops`    | the `karume::` operators (`gru_scan` / `gru_scan_reverse`) registered with `torch.library`                                    |
| `dist`          | generic assembly engine: series directories → one distribution directory (ADR 0041 / 0052; the pipeline registry is injected) |
| `modelcard`     | generic model-card rendering — pure functions deriving the card from the manifest (ADR 0037 §3 frontmatter)                   |
| `cli`           | the `karume` entry point (`dist` / `verify`, lazy dispatch)                                                                   |

No module here is model-specific: the wheel carries no `patch_*`, no export script and no family
name table. That is a machine gate, not a convention — `tests/test_architecture_boundary.py` fails
the moment core imports a recipe (ADR 0065 decision 3).

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
    (`anima.export.TARGET_PRESERVED` in the recipes — currently only transformer and vae_decoder).
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
  implemented in ADR 0018, i8 + per-channel scale in ADR 0019, and i4 + group scale in ADR 0069.
- Host implementations of the runtime pipelines (SBV2: text → durations → assembling y_mask → voice
  / Anima: tokenization → scheduler → CFG → denormalization). The layer that connects the emitted
  targets is outside the exporter's scope (for Anima the host implementation lives in
  `packages/models/src/anima/pipeline.ts`). (The **TS port of Anima's tokenizer** is complete, in
  `tools/export-recipes/anima/demo.py` + `packages/models/src/anima/text/`, and tokenization parity is held separately by
  `packages/models/tests/anima_tokenizer_test.ts`.)
