# Weights from a single file

What `fromSingleFile` of `@karume/models` does now: a pipeline built from a weights file the caller holds — an upstream
release or a fine-tune, in an upstream layout — and a distribution, with no export step (the counterpart of diffusers'
`from_single_file`). The file is read with [the safetensors reader](safetensors.md), placed with the distribution's
[name map](convert.md#the-name-map) and converted while each Session reads it, against the plans of the
distribution's containers ([the conversion](convert.md)). Anima has it: `anima.Pipeline.fromSingleFile`. It is item 7
of the ADR [0152](../decisions/0152-maintained-specs-and-declared-numeric-requirements.md) list in the
[backlog](../backlog.md).

## What it takes

`Pipeline.fromSingleFile(file, ref, options = {})` resolves to a `Pipeline`, the same as one `fromPretrained` builds.

- `file` (`SingleFileSource`): a `Blob` (a browser `File`) or a ranged reader `{ size, read(offset, length) }`. Its
  format is told from its first 16 bytes; only safetensors is read (an 8-byte length and the header's `{` at byte 8).
  The pipeline reads `file` for as long as it lives: the Session of a component from the file reads its weights from
  it when it is built — the text conditioner's at every generate, the transformer's at every generate in the default
  per-stage residency and, in `"transformer"` residency, whenever the resident transformer is built or rebuilt — so
  the file must stay readable until `dispose` (with `retainConverted`, until each such Session has been built once).
- `ref`: the distribution, as `fromPretrained` takes it (required). Its manifest must carry a name map (`nameMap` of
  `karume/6`).
- `options` (`FromSingleFileOptions`): those of `fromPretrained`, with the same meaning, and `retainConverted`
  ([below](#keeping-the-converted-weights)). `model` is the model of the manifest the file derives from — the graphs,
  the plans and `pipelineConfig` come from it —, `quant` the preset, `components` the replacements. `fromPretrained`
  does not take `retainConverted`.

## Which component comes from where

A family declares the components it takes from a file, each with whether the file must hold it and how the container
its tensors are converted against is chosen; the weights label of its float container; and the 16-bit float codec
each float dtype is stored as. A component it does not declare comes from the distribution. The rule: **a preset
decides what is quantized; what it does not quantize keeps the precision of its source** — a float component from the
file is converted against the plan of the model's float container, with that plan's 16-bit float weights stored as
the file's dtype ([a plan with another 16-bit float codec](convert.md#a-plan-with-another-16-bit-float-codec)): a bit
copy, never a rounding to the container's type. Anima's float container is `bf16` (every float weight of its
distribution is bf16, the upstream's bits); `BF16` is stored as `bf16` and `F16` as `f16`; `F32` has none (no Anima
container stores f32 weights). No container of another label is read for a float component, whether the distribution
carries one or not. Anima's components:

| Component | From | Converted against |
| --- | --- | --- |
| `transformer` | The file (required) | Under a quant that stores it as int8: that quant's transformer container, whatever the file's float dtype (a mix of them included). Under a quant that stores it as a float (as `bf16`: `bf16`, `bf16-c16`): that quant's own `bf16` transformer, its 16-bit weights stored as the file's dtype (BF16 as bf16, F16 as f16), run with the quant's Session options |
| `text_conditioner` | The file when the file holds one, else the distribution | When from the file: the model's `bf16` text conditioner, its 16-bit weights stored as the file's dtype, whatever the quant — no preset quantizes the text conditioner |
| `text_encoder`, `vae_decoder` | The distribution | — (the selected quant's, as `fromPretrained`) |

- **The dtype rule.** Where the storage follows the dtype, the dtype is that of the component's tensors in the file:
  one dtype for the whole component. A component whose tensors mix dtypes is refused naming each, and a dtype with no
  16-bit float codec is refused with what to do (below). Which quant stores a component quantized is read from its
  weights label: a label that is not the family's float label is a quantized storage.
- A component named in `options.components` comes from that replacement, never from the file, whatever the file
  holds; its tensors in the file are not read.
- The file "holds" a component when the name map's layout that recognises the file places at least one key in it.
- When no component is left to come from the file, the call is refused: it is a `fromPretrained` call.
- A component converted from the file runs the graph of the container whose plan it is converted against. For the
  transformer that is the selected quant's own container under every quant, so the quant's Session options, declared
  for that container, apply as they are; with its 16-bit weights stored as f16 instead of bf16 the graph does not
  change (the plan changes in that codec alone).
- The text conditioner from the file runs the graph of the model's `bf16` text conditioner under every preset, an F16
  file's as well. Its Session gets the options every text stage gets (none of the preset's knobs); f16 and bf16
  storage under f32 compute need no GPU feature. Under a preset whose own text conditioner is the `bf16` one, a BF16
  file of the distribution's weights runs what `fromPretrained` runs; under a preset that stores it otherwise, the
  images can differ from `fromPretrained`'s. The exporter rounds only what a checkpoint carries, for f16 and bf16
  alike (`round_weights_to_f16` / `round_weights_to_bf16` in `tools/exporter/src/karume/quantize.py`; the module's
  non-persistent buffers stay f32), so an f16 export of the text conditioner carries the bf16 export's graph
  descriptor, and an F16 file's text conditioner runs the graph such an export has.
- What this opens for the transformer, with the runtime's part in it:

  | File | Quant | Storage × compute | |
  | --- | --- | --- | --- |
  | Any float dtype | An int8 quant | int8, the quant's compute | As before: the quant's own container |
  | BF16 | A quant that stores it as bf16 under f32 compute (`bf16`) | bf16 × f32 | The quant's own pair |
  | F16 | The same | f16 × f32 | The quant's container with its 16-bit weights stored as f16: f16 storage under f32 compute |
  | F16 | A quant that stores it as bf16 under f16 linear and attention compute (`bf16-c16`) | f16 × f16 linear and attention | f16 storage under f16 compute, which the f16-compute checks cover |
  | F16 | A float quant with an explicit `linearCompute: "f16"` (as `bf16` with it) | f16 × f16 linear; attention in the quant's attention compute, or in an explicit `attentionCompute` (under `bf16`: f32) | The same: an explicit `linearCompute` sets the linear layers' compute alone |
  | BF16 | A quant under f16 linear compute (`bf16-c16`), or a float quant with an explicit `linearCompute: "f16"` | bf16 × f16 linear | Refused: the runtime builds it (its f16-compute GEMM reads a bf16 weight like an f32 one and rounds it into the f16 tile), but no GPU check covers its numbers (the f16-compute checks cover f32 and f16 storage) |

  f16 attention compute reads activations only, so it does not depend on the weights' storage.

## The order

1. The manifest and the selection, as `fromPretrained` (`loadManifest`, `resolveSelection`).
2. The name map: the manifest's `nameMap`, fetched through the hub like any file reference and read with
   `parseNameMap`. It is not part of the load's progress stream.
3. The file: its format from its first bytes, then its header (`openSafetensors`: two reads, no tensor byte).
4. The placement (`placeTensors`), the components that come from the file, their tensor lists and the containers they
   are converted against, with the 16-bit float codec of a float storage (the dtype rule and the family's refusals:
   manifest data only).
5. The shared loader, as for `fromPretrained`, with each component from the file served by a converting source:
   part 0 of every component's container is fetched; for a component from the file, its container's two documents are
   checked against the manifest, its plan is read — for a float storage with its 16-bit float weights stored as the
   file's dtype —, its plan checks run (below), and the converting source is built (with its own checks: a tensor
   missing for an initializer, a shape that differs). Then the family's admission, on the same opener; then the weight
   parts and the assets.
6. The build, as `fromPretrained`.

The abort signal is checked at the same kind of boundaries as in `fromPretrained`: the fetches, after the name map,
after the file's header, between the components in the loader, admission and the build.

## The refusals

Everything that needs no tensor byte is refused before a weight part is fetched, with one message each:

| Refused | Where | The message |
| --- | --- | --- |
| A distribution without a name map | 2 | The distribution carries no name map |
| A name map that is not a `karume-name-map/1` document | 2 | `parseNameMap`'s, every problem together |
| A file that is not safetensors | 3 | What was seen: a GGUF file, a zip archive (a `torch.save` checkpoint), a Python pickle, a file too short, or its first bytes |
| A safetensors file the reader refuses | 3 | The reader's |
| A file no layout of the map recognises (or more than one) | 4 | `placeTensors`' |
| A key no component of the layout takes (`unplaced`) | 4 | Every such key. Nothing of a file is dropped silently; a key a rule of the map excludes is the map's own statement and is not refused |
| Tensors of a component the family does not take from a file (and the caller does not replace) | 4 | The component |
| A file that holds no component the family requires (Anima: the transformer) | 4 | The component and the layout |
| A tensor of a component from the file that is not `F32`, `F16` or `BF16` | 4 | Every such tensor and its dtype |
| A component stored as the float type of its tensors (Anima: the text conditioner; the transformer under a quant that stores it as a float) whose tensors mix dtypes | 4 | Each dtype with its tensor count and one of its file keys, and what to do |
| Such a component whose dtype has no 16-bit float codec (Anima: `F32`) | 4 | The dtype, the family's float storages, and what to do |
| A model without the float container whose plan such a component is converted against (Anima: no `bf16` text conditioner, whatever the file's dtype — an `f16` one is not read instead) | 4 | The component, the label, the file's dtype, and what to do |
| A combination the family does not run (Anima: a BF16 transformer under f16 linear compute — a quant's, as `bf16-c16`'s, or an explicit `linearCompute: "f16"`) | 4 | One message: the quant, where its linear compute comes from, and the file's dtype |
| No component left to come from the file | 4 | What the replacements take |
| A tensor placed in a component from the file that its plan does not use | 5 | Every such tensor, with its initializer name |
| A plan that stores a codec the conversion does not encode (Anima: none of its containers) | 5 | One message for the component: the label and the quant, the codec, the encoded codecs and, for a component its quant decides, the quants under which the component's container works |
| A tensor whose dtype is not the one an initializer's float codec stores as it is (`F16` for `f16`, `BF16` for `bf16`) | 5 | The component, the first such initializer, its file key and its dtype, and what to do |
| A tensor missing for an initializer of the plan, a shape that differs | 5 | The converting source's, every problem together |
| What the family's admission refuses | 5 | The family's, as for `fromPretrained` |

- What to do, in every refusal that says it: `components` to take the distribution's, for a component that may come
  from the distribution (Anima's text conditioner); a quant that stores it quantized, which converts from any float
  dtype, with those quants named, for a component its quant decides (Anima's transformer: the int8 quants).
- f16 and bf16 storage never round (the conversion stores a value of the type as it is and refuses another), so an
  initializer stored as one takes tensors of that dtype only; without the last plan refusal such a file would fail at
  the first read of a Session. For a float storage it holds by construction (the plan's 16-bit float weights are
  stored as the file's dtype); it catches an f16 or bf16 weight in the container of a quant that stores the component
  quantized.
- To name the quants that work, each quant's container for the component is settled as above (a quant refused there
  does not work), and the part 0 of every such container other than the failing one is fetched (on that refusal only,
  outside the progress stream) and its plan read.

## What is fetched and what is converted when

| What | Fetched | Counted in the progress |
| --- | --- | --- |
| The manifest | Yes | No (as for `fromPretrained`) |
| The name map | Yes, before the file is read | No |
| Part 0 of each component's container (for a component from the file: of the container it is converted against) | Yes | Yes, from the start |
| The const part and the asset parts of a component from the file | Yes | Yes, added to the total once its plan has been read, before any of them is fetched |
| The parts a converting source builds (`convertedParts`) | No | No |
| The container of the selected quant for a component that comes from the file against another container (Anima's text conditioner under a quant whose own is not the `bf16` one) | No | No |
| Every part of a component from the distribution or a replacement, and the assets | Yes | Yes |

- The load's progress is one stream, as for `fromPretrained`; its total grows once, by the kept parts of each
  component from the file, between the descriptors and the weight parts.
- A component from the file is converted when a Session of it is built: each tensor is read from the file and encoded
  when a block of it is read. By default the conversion keeps the payloads of the last tensor only, so every Session
  build reads and converts the component's tensors again; with `retainConverted`, only the first one does.
- Nothing is written: the converted container exists only while it is read (or, with `retainConverted`, in memory).

## Keeping the converted weights

`retainConverted: true` (default `false`) keeps the converted weights of every component that comes from the file in
host memory once a Session has read them: the shared loader passes `retainEncoded` to the converting source of each
such component ([the conversion](convert.md#what-is-served)).

| | Without (default) | With `retainConverted` |
| --- | --- | --- |
| A Session build of a component from the file | Reads its tensors from the file and converts them | The first reads and converts; every later one is served from memory and reads nothing of the file |
| Host memory held between generations | None for the converted weights | Their converted size: Anima's transformer about 1.8 GiB as int8 (the default quant), 3.6 GiB as bf16 or f16; plus the text conditioner's when it comes from the file |
| GPU memory | Unchanged | Unchanged |

- What a Session build reads is the same bytes either way, so the images do not change.
- The kept payloads are held by the converting source, which the component holds, which the pipeline holds. They are
  released with the pipeline, by garbage collection once the caller drops it; `dispose` does not drop them (it does
  not reach the converting sources).
- In the default per-stage residency the transformer's Session is built at every generate, so this is where the option
  pays: the second generation's transformer stage reads nothing of the file. `residency: "transformer"` is the other
  way to avoid the rebuild: it keeps the built Session on the GPU, at GPU memory, and it is rebuilt — from the file
  without this option — only after an eviction.
- A value that is not a boolean is refused with `ModelInputError` before anything is fetched.
