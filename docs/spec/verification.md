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
  sizes the state slots with its value per run, and the runs — each a list of steps given by their rows.

Widths are the real model's: the kernels a level selects are listed for real shapes, so a narrower graph would select
none of them. What an entry cuts is depth. It keeps one layer of each kind, together with the layers those read values
or state slots from, and the layer that computes the graph's outputs.

| Entry | Graph | Kept layers | Runs |
| --- | --- | --- | --- |
| `gemma4-qat-e2b-cut4` | Gemma 4 QAT E2B, `model`, weights `int4-sym-g` group 32 | 13, 14, 15, 34 of 35 | prefill of 8 rows, then 4 steps of 1 row; prefill of 64 rows, then 4 steps of 1 row |

### Generated weights

- Every model-side initializer is generated from (seed, initializer name) and the declaration, in its stored codec
  with its scales (`packages/runtime/tests/corpus/weights.ts`). The same initializer gets the same bytes in the full
  graph and in a cut of it, on every machine.
- The role comes from the op and input slot that read the initializer:

| Role | Read by | Values |
| --- | --- | --- |
| linear weight | `linear` slot 1 (also when `embedding` slot 0 reads the same table) | standard deviation 1/sqrt(fan-in); in `int4-sym-g`, levels uniform in [-7, 7] and every group scale 1/(sqrt(fan-in) · sqrt(56/3)) |
| norm weight | `rms_norm` slot 1 | 1 + U(-0.1, 0.1) |
| operand | `mul` slot 1 | U(-1, 1) |

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

Per entry and per run, with six Sessions opened one after another on the same generated weights:

| Check | Compared | Stored value |
| --- | --- | --- |
| Fusion identity | `exact` against `exact` with every fusion rule disabled: every output word of every step, as u32. The fused side's rule counters are above 0 at every step and the plain side's are 0. | none |
| Determinism | Two Sessions of the same settings, at `exact` and at `reorder`: every output word, as u32. | none |
| The comparison can fail | A Session whose weights differ by one quantization level in one element gives other output words. | none |
| `reorder` selects something | At least one step dispatches another set of kernel keys at `reorder` than at `exact`. | none |
| Outputs are finite and vary | No NaN or infinity; not every value equal. | none |
| Output rows | Per level, the sha256 of every step's output words (kept as a safetensors artifact). | a row per environment |
| Census rows | Per level, the sha256 of a text listing every step's kernel keys with their dispatch counts and the fusion counters that are not 0. | a row per environment |

- The rows are the sha256 rows of the reference mechanism (`packages/runtime/tests/fixtures/references/corpus.json`):
  one row per environment key, a missing row is an explicit skip and fails the reference guard, and a row is never
  replaced by a tolerance.

### What is not checked

- **The distance between `exact` and `reorder`.** It is written to the results per step (largest absolute and relative
  difference) and never asserted. One quantization level in one weight moves the outputs less than a change of
  summation order does, so no tolerance at the graph outputs separates a defect from a reordering.
- **The op band of a `reorder` strategy on the corpus's activations.** A strategy's band is checked by its own kernel
  check, on its own inputs.
- **That a fixture still equals the model it was cut from.** No test reads a distribution mirror.

### Adding an entry

- `flock /tmp/karume-gpu.lock node packages/runtime/tests/corpus/author.ts <entry name>` reads a local distribution
  mirror, cuts the declaration to the entry's layers and writes the fixture. The same mirror and entry give the same
  bytes.
- The cut (`packages/runtime/tests/corpus/cut.ts`) takes layer membership from the `layers.<n>.` segment of
  initializer names, rewires the residual value across each removed range, and removes what nothing reads any more. It
  fails when the boundary is not a single residual value, when a kept layer reads a value or a state slot of a removed
  layer, or when a graph output is computed in a removed layer.
- The tool refuses to write unless the cut dispatches the same set of kernel keys as the full graph, step by step, at
  both levels, on generated weights.
- The entry's rows for a machine are created with `KARUME_REFERENCE=write vp test packages/runtime/tests/gpu_corpus_test.ts`.
