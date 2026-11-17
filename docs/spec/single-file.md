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
its tensors are converted against is chosen, and which weights label holds which float dtype as it is. A component it
does not declare comes from the distribution. The rule: **a preset decides what is quantized; what it does not
quantize keeps the precision of its source** — a float component from the file is converted against the container
whose storage is the file's dtype, a bit copy. Anima's float labels are `BF16` → `bf16` and `F16` → `f16`; `F32` has
none (no Anima container stores f32 weights). Anima's components:

| Component | From | Converted against |
| --- | --- | --- |
| `transformer` | The file (required) | Under a quant that stores it as int8: that quant's transformer container, whatever the file's float dtype (a mix of them included). Under a quant that stores it as a float (`f16`, `f16-c16`, `bf16`): the model's transformer container of the file's dtype (BF16 → `bf16`, F16 → `f16`), run with the quant's Session options |
| `text_conditioner` | The file when the file holds one, else the distribution | When from the file: the model's text conditioner container of the file's dtype (BF16 → `bf16`, F16 → `f16`), whatever the quant — no preset quantizes the text conditioner |
| `text_encoder`, `vae_decoder` | The distribution | — (the selected quant's, as `fromPretrained`) |

- **The dtype rule.** Where the container is chosen by the dtype, the dtype is that of the component's tensors in the
  file: one dtype for the whole component. A component whose tensors mix dtypes is refused naming each, and a dtype
  with no float label is refused with what to do (below). Which quant stores a component quantized is read from its
  weights label: a label that is not one of the family's float labels is a quantized storage.
- A component named in `options.components` comes from that replacement, never from the file, whatever the file
  holds; its tensors in the file are not read.
- The file "holds" a component when the name map's layout that recognises the file places at least one key in it.
- When no component is left to come from the file, the call is refused: it is a `fromPretrained` call.
- A component converted from the file runs the graph of the container it is converted against. The transformer's graph
  is the same for every storage of it in a distribution — in every Anima manifest of `karume-anima` and
  `karume-anima-extra` the `f16`, `i8` and `bf16` transformer containers carry the same graph descriptor (one sha256) —
  so the quant's Session options, declared for the quant's own container, apply to the container of the file's dtype
  unchanged. The shared code does not assume it: when the container of the file's dtype is not the quant's own, their
  graph descriptor hashes in the manifest are compared and a difference is refused.
- The text conditioner's graphs differ between its `f16` and `bf16` containers — in their constant tables only (the
  f16 export also rounds the module's non-persistent buffers, the bf16 one leaves them f32 —
  ADR [0131](../decisions/0131-bf16-weight-execution.md) decision 2) — and the text conditioner from the file runs the
  graph of the container of its dtype under every preset. Its Session gets the options every text stage gets (none of
  the preset's knobs); f16 and bf16 storage under f32 compute need no GPU feature. Under a preset whose own text
  conditioner is f16, a BF16 file's images can differ from `fromPretrained`'s.
- What this opens for the transformer, with the runtime's part in it:

  | File | Quant | Storage × compute | |
  | --- | --- | --- | --- |
  | Any float dtype | An int8 quant | int8, the quant's compute | As before: the quant's own container |
  | BF16 | `bf16`, `f16` | bf16 × f32 | The `bf16` preset's own pair |
  | F16 | `f16`, `bf16` | f16 × f32 | The `f16` preset's own pair |
  | F16 | `f16-c16` | f16 × f16 linear and attention | The `f16-c16` preset's own pair |
  | BF16 | `f16-c16`, or any quant with an explicit `linearCompute: "f16"` | bf16 × f16 linear | Refused: the runtime builds it (its f16-compute GEMM reads a bf16 weight like an f32 one and rounds it into the f16 tile), but no distributed preset runs it and no GPU check covers its numbers (the f16-compute checks cover f32 and f16 storage) |

  f16 attention compute reads activations only, so it does not depend on the weights' storage.

## The order

1. The manifest and the selection, as `fromPretrained` (`loadManifest`, `resolveSelection`).
2. The name map: the manifest's `nameMap`, fetched through the hub like any file reference and read with
   `parseNameMap`. It is not part of the load's progress stream.
3. The file: its format from its first bytes, then its header (`openSafetensors`: two reads, no tensor byte).
4. The placement (`placeTensors`), the components that come from the file, their tensor lists and the containers they
   are converted against (the dtype rule, the graph comparison and the family's refusals: manifest data only).
5. The shared loader, as for `fromPretrained`, with each component from the file served by a converting source:
   part 0 of every component's container is fetched; for a component from the file, its container's two documents are
   checked against the manifest, its plan checks run (below), and the converting source is built (with its own checks:
   a tensor missing for an initializer, a shape that differs). Then the family's admission, on the same opener; then
   the weight parts and the assets.
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
| Such a component whose dtype has no float label (Anima: `F32`) | 4 | The dtype, the family's float storages, and what to do |
| A model without the container a component is stored as (Anima: no `f16` or `bf16` text conditioner for the file's dtype) | 4 | The component, the label, the file's dtype, and what to do |
| A container of the file's dtype whose graph is not the quant's own container's (the manifest's graph descriptor sha256) | 4 | The quant, both labels and both hashes, and what to do |
| A combination the family does not run (Anima: a BF16 transformer under f16 linear compute — `f16-c16`, or an explicit `linearCompute: "f16"`) | 4 | One message: the quant, where its linear compute comes from, and the file's dtype |
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
  the first read of a Session. With the container chosen by the file's dtype it holds by construction; it still
  catches a manifest whose label holds another float codec.
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
| The container of the selected quant for a component that comes from the file against another container (Anima's text conditioner, and its transformer under a float quant whose label is not the file's dtype) | No | No |
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
