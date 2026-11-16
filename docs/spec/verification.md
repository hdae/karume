# Verification: how the runtime checks itself

What is built now. Reasons are in ADR [0152](../decisions/0152-maintained-specs-and-declared-numeric-requirements.md)
(decision 5); work that is not built is in the [backlog](../backlog.md). The numeric levels the checks compare are in
[numerics.md](numerics.md).

## The regression corpus

The corpus runs the graphs of real models inside the runtime's own tests (`packages/runtime/tests/gpu_corpus_test.ts`,
core lane, real GPU). An entry is a real model's graph cut to a few layers, run on weights generated from a seed by a
driver that knows only the declaration. It reads no real weights, no distribution mirror, and imports neither
`packages/models` nor `packages/hub`.

### An entry

An entry (`packages/runtime/tests/corpus/entries.ts`) is:

- **A fixture** under `packages/runtime/tests/fixtures/corpus/`: a graph container (`<name>.krg` — the declaration and
  its const region) of at most 2 MiB, and a storage table (`<name>.storage.json` — the codec, group size and row axis
  the distribution stores each model-side weight in, which a graph container does not carry).
- **A seed** for the weights and the inputs.
- **A plan**: the symbol that is the rows of a step, the values of the other input symbols, the states-only symbol that
  sizes the state slots with its value per run, and the runs — each a list of steps given by their rows. On a graph
  with state slots the steps of a run follow one another (a prefill, then appended steps); on a graph without, they
  are independent runs of the graph at those rows.
- **Settings**: the Session settings the entry is checked under, each with a name and its kind. A setting of kind
  *level* is a level alone (`exact`, `reorder`); a setting of kind *approximation* names an approximation. Every
  entry has `exact`.

Widths are the real model's: the kernels a level selects are listed for real shapes, so a narrower graph would select
none of them. What an entry cuts is depth. It keeps layers that together dispatch every kernel key the full graph
dispatches under the entry's settings, the layers those read values or state slots from, and the layer that computes
the graph's outputs.

| Entry | Graph | Kept layers | Settings |
| --- | --- | --- | --- |
| `gemma4-qat-e2b-cut4` | Gemma 4 QAT E2B, `model`; weights `int4-sym-g` group 32 | 13, 14, 15, 34 of 35 | `exact`, `reorder` |
| `gemma4-qat-mobile-e2b-cut3` | Gemma 4 QAT E2B of the mobile line, `model`; weights `int4-sym-g` (groups 512, 2048, 4096), `int2-off`, `int8-sym` and f32; a `static_quantize` around every linear | 13, 14, 34 of 35 | `exact`, `reorder`, `packed` (`reorder` with `packedStaticQuantize`), `integer` (`packed` with `integerGemv`) |
| `irodori-v4-small-dit-cut2` | Irodori v4-small, `dit`; no state slots; weights `int8-sym`; attention built from `bmm` and `safe_softmax` | blocks 0, 11 of 12 | `exact`, `a8` (`linearCompute: "a8"`) |
| `wan2.2-ti2v-5b-transformer-cut2` | Wan2.2 TI2V-5B, `transformer`; no state slots; weights `int8-sym`; the `attention` op | blocks 28, 29 of 30 | `exact`, `a8-attn8-s16` (`linearCompute` and `attentionCompute` `"a8"`, `attentionScoreStorage: "f16"`), `a8-attn8-online` (the same with `attentionScoreStorage: "none"`) |

- The two Gemma entries have two runs: a prefill of 8 rows, then 4 steps of 1 row; a prefill of 64 rows, then 4 steps
  of 1 row.
- The two entries without state slots have one run of two steps: 48 and 513 rows (Irodori), 16 and 513 rows (Wan2.2).
  The rows are chosen so that the two steps dispatch the largest set of kernel keys two values can reach; kernel keys
  that only other row counts select are not run. `reorder` dispatches the kernels of `exact` on these two graphs and
  is not a setting of theirs.

### Generated weights

- Every model-side initializer is generated from (seed, initializer name) and the declaration, in its stored codec
  with its scales (`packages/runtime/tests/corpus/weights.ts`). The same initializer gets the same bytes in the full
  graph and in a cut of it, on every machine.
- The role comes from the op and input slot that read the initializer:

| Role | Read by | Values |
| --- | --- | --- |
| linear weight | `linear` slot 1 (also when `embedding` slot 0 reads the same table) | levels drawn uniformly, every scale 1/(sqrt(fan-in) · d), d the standard deviation of the levels: `int4-sym-g` levels [-7, 7]; `int2-off` all four codes, levels [-2, 1]; `int8-sym` levels [-127, 127]; f32 U(-1, 1) |
| norm weight | `rms_norm` slot 1, `layer_norm` slot 1 | 1 + U(-0.1, 0.1) |
| operand | `mul` slot 1 | U(-1, 1) |
| bias | `linear` slot 2, `layer_norm` slot 2 | U(-0.1, 0.1) |
| additive operand | `add` slot 0 or 1 | U(-0.1, 0.1) |

- An initializer read by anything else, by consumers that ask for different rules, or stored in a codec without a rule
  fails with its name and what reads it. There is no fallback rule.
- Initializers of the const region are read from the fixture and never generated.
- The generator's bytes are pinned: `corpus_weights_test.ts` holds the sha256 of a small tensor per codec and role, so
  a change to the generator is told apart from a change in the runtime.

### The driver

- The inputs of a step are drawn from what reads them (`packages/runtime/tests/corpus/driver.ts`): an i32 input read
  as an `embedding` index is drawn below the table's row count; an f32 input is U(-1, 1); a bool input is all ones.
  Any other i32 input fails with its name.
- Dimension expressions are evaluated with the runtime's dimension grammar.
- A graph with state slots runs each run through one generation context: the first step is the prefill, later steps
  append. A graph without state slots runs each step directly.

### What is checked

Per entry and per run, with Sessions opened one after another on the same generated weights:

| Check | Under | Compared | Stored value |
| --- | --- | --- | --- |
| Fusion identity | every setting of kind *level* | The setting against the same setting with every fusion rule disabled: every output word of every step, as u32. The fused side's rule counters are above 0 at every step and the plain side's are 0. | none |
| Determinism | every setting | Two Sessions of the setting: every output word, as u32. | none |
| The comparison can fail | `exact` | A Session whose weights differ by one quantization level in one element gives other output words. | none |
| Every setting exercises something | every pair of settings | No two settings dispatch the same sets of kernel keys at every step. | none |
| Outputs are finite and vary | every setting | No NaN or infinity; not every value equal. | none |
| Output rows | every setting | The sha256 of every step's output words (kept as a safetensors artifact). | a row per environment |
| Census rows | every setting | The sha256 of a text listing every step's kernel keys with their dispatch counts and the fusion counters that are not 0. | a row per environment |

- A setting of kind *approximation* is not compared with another plan: the packed activations and the integer GEMV
  are forms of the linear→static_quantize fusion and do not run with the rule disabled.
- The element that moves one level is named by the entry. In the mobile entry it is in the tied table: a level of a
  layer weight there moves its linear's output by less than the quantization step that follows.

- The rows are the sha256 rows of the reference mechanism (`packages/runtime/tests/fixtures/references/corpus.json`):
  one row per environment key, a missing row is an explicit skip and fails the reference guard, and a row is never
  replaced by a tolerance.

### What is not checked

- **The distance of a setting from `exact`.** It is written to the results per step (largest absolute and relative
  difference) and never asserted. One quantization level in one weight moves the outputs less than a change of
  summation order does, so no tolerance at the graph outputs separates a defect from a reordering. Where a
  `static_quantize` follows, a change of summation order either leaves a step's outputs bit-identical or moves them
  far (a rounding flips).
- **The op band of a `reorder` strategy on the corpus's activations.** A strategy's band is checked by its own kernel
  check, on its own inputs.
- **That a fixture still equals the model it was cut from.** No test reads a distribution mirror.

### Adding an entry

- `flock /tmp/karume-gpu.lock node packages/runtime/tests/corpus/author.ts <entry name>` reads a local distribution
  mirror, cuts the declaration to the entry's layers and writes the fixture. The same mirror and entry give the same
  bytes.
- The cut (`packages/runtime/tests/corpus/cut.ts`) takes layer membership from the `<segment>.<n>.` of initializer
  names, the segment being the one the entry names (`layers`, `blocks`). A node belongs to the layer of its own
  weights, else to the latest layer among its producers; a node without weights whose outputs only one later layer
  reads, and whose other inputs that layer reads anyway, belongs to that layer (a norm without weights at the head of
  a block). The cut rewires the residual value across each removed range and removes what nothing reads any more,
  graph inputs included. It fails when the boundary is not a single residual value, when a kept layer reads a value or
  a state slot of a removed layer, or when a graph output is computed in a removed layer.
- The tool refuses to write unless the cut dispatches the same set of kernel keys as the full graph, step by step, at
  both levels, on generated weights.
- The entry's rows for a machine are created with `KARUME_REFERENCE=write vp test packages/runtime/tests/gpu_corpus_test.ts`.
