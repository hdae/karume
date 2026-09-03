# opbench — static op census

A CLI that reads a karume asset and writes **one row per IR node**, with every symbolic dimension
resolved to a concrete number. No GPU, no weight bytes: it only reads the safetensors _header_ of
each component's first shard, where the whole IR graph lives (`__metadata__.karume_ir`).

This is stage 1 of the op-microbenchmark harness. It answers two questions that were previously
answered by extrapolation:

- **What shapes does a real run actually dispatch?** The census is the shape corpus for single-op
  benchmarks, so a benchmark measures the geometry the model really uses.
- **How many times does each shape occur in one run?** That count is the _census weight_. A
  single-op speedup only matters in proportion to its weight, so every candidate's payoff can be
  computed instead of guessed.

## Usage

```
deno run -A tools/opbench/main.ts census --source <dir> --out <dir> [options]
```

| Option                                | Meaning                                                                                                                                    |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `--source <dir>`                      | A distribution mirror (a directory with `karume.json`) or a series output directory under `outputs/series/`                                |
| `--out <dir>`                         | Where `census.jsonl` and `summary.json` are written                                                                                        |
| `--model <name>`                      | Distribution form only. Defaults to the manifest's `defaultModel`                                                                          |
| `--quant <name>`                      | Distribution form: the manifest quant (defaults to `defaultQuant`). Series form: the storage-dtype group (`model.i8-*.safetensors` → `i8`) |
| `--family <name>`                     | Overrides the family inferred from the manifest pipeline id or the series directory name                                                   |
| `--scenario <name>=<SYM>:<value>[,…]` | Symbol bindings. May be repeated; each one produces a separate `scenario` value in the output. Defaults to the family's built-in scenarios |

One invocation = one asset = one `census.jsonl` + one `summary.json`, following the
`tools/ram-peak/measure.ts` convention of one configuration per process.

Examples:

```sh
# Distribution mirror, default scenarios (gemma4 = decode and prefill)
deno run -A tools/opbench/main.ts census \
  --source models/karume-gemma4-e2b \
  --out outputs/bench/karume-gemma4-e2b/2026-09-03_op-census

# Series output (no manifest — components are discovered from the directory tree)
deno run -A tools/opbench/main.ts census \
  --source outputs/series/birefnet-hr-1024 \
  --out outputs/bench/birefnet-hr-1024/2026-09-03_op-census

# Explicit binding: a longer full-attention capacity
deno run -A tools/opbench/main.ts census \
  --source models/karume-gemma4-e2b --scenario long=M:1,C:8192 --out <dir>
```

## Scenario bindings

A static IR keeps its symbolic dimensions (`M` and `C` for gemma4, `S` for the Anima transformer,
`T` / `P` for the speech families). Concrete numbers only exist for a _particular run_, so the
census needs a scenario: a map from symbol to value.

A binding key is either `SYM` (applies to every component) or `<component>.SYM` (applies to that
component only). The qualified form is necessary because the same symbol name can mean different
things in different components — in Irodori, `T` is a token count in `backbone` but a frame count in
`codec_encoder`.

Every family has a built-in default scenario set with its provenance recorded in
`summary.json`. **A symbol that a component's shapes use but the scenario does not bind is a hard
error.** Emitting rows with unresolved symbols would silently break the meaning of the census
weight, so the tool refuses instead.

Bindings that a component declares but never uses in any shape (for example `C` when the asset was
exported with a baked capacity) are reported per scenario as `unused_bindings`, so a scenario that
does not actually reach the graph is visible rather than silent.

## Output

### `census.jsonl` — one JSON object per line, one line per IR node

| Field                                                      | Meaning                                                                                                                                                                                  |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `family`, `model`, `quant`, `component`, `component_dtype` | Which asset and component the node came from                                                                                                                                             |
| `scenario`, `bindings`, `binding_source`                   | The scenario that produced the numbers, its effective bindings for this component, and whether they came from the built-in table (`default`) or `--scenario` (`cli`)                     |
| `node_index`                                               | Position in `graph.nodes` (also the identity used by `producers` / `consumers`)                                                                                                          |
| `op`, `attrs`                                              | The op and its attributes, verbatim from the IR                                                                                                                                          |
| `in_shapes`, `out_shapes`                                  | Fully resolved shapes. `in_shapes` follows `ins`; `out_shapes` follows output slot order                                                                                                 |
| `in_dtypes`, `out_dtypes`                                  | Semantic dtypes (`f32` / `i32` / `bool`) — computation is always in the semantic type                                                                                                    |
| `storage`                                                  | Aligned with `ins`. `null` for a non-initializer input; otherwise the initializer's storage: safetensors key, storage dtype, and for group quantization the `group_size` and `scale` key |
| `state_shapes`                                             | Resolved shapes of the state slots this node touches (empty for nodes that touch none)                                                                                                   |
| `fused_by`                                                 | The fusion rule that absorbed this node, or `null` if it runs as a plain node                                                                                                            |
| `aliases_input`                                            | The node produces zero dispatches because its output aliases its input (`reshape` always, identity `expand` when no axis is replicated)                                                  |
| `producers`, `consumers`                                   | Adjacent node indices, for reconstructing the dataflow around a candidate                                                                                                                |

### `summary.json`

Per scenario:

- `by_op` — node count and total output element count per op. The element count is what makes the
  strided-materialization question (`permute` / `expand` / `slice`) answerable in real numbers
  rather than by extrapolation.
- `by_storage` — node count per storage signature of the node's initializer inputs (`f32+i4g32`,
  `f32+i8`, `none`, …). Group length is part of the signature because it changes which kernel runs.
- `by_fusion` — `hits` is the number of fusion _steps_ (the number the standing asset gate in
  `packages/runtime/tests/assets_fusion_counts_test.ts` fixes), `absorbed` is the number of _nodes_
  those steps swallowed, `plain` is everything left over, and `aliased` is the subset of `plain`
  that costs zero dispatches.
- `weights` — the census weight table: identical `(component, op, in_shapes, out_shapes, dtypes,
  storage, fusion)` tuples collapsed into one row with a `count`, sorted by count. This is the input
  a single-op benchmark iterates over.

## Notes

- Fusion is planned against the **WebGPU core default limits** (128 MiB storage binding, 65535
  workgroups per dimension), not the limits of whatever machine runs the tool. A census must not
  change with the machine that produced it; for a machine's real behaviour, `lastRunFusions` at
  runtime is the source of truth.
- Generated files belong under `outputs/bench/<model>/<YYYY-MM-DD>_<purpose>/`, which is untracked
  (see `docs/assets-layout.md`). Numbers that are meant to last go into `docs/research/`, and
  adoption decisions into `docs/perf-ledger.md`.
