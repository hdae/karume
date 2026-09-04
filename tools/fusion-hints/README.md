# fusion-hints

Enumerate **fusion candidates** in shipped karume assets: contiguous windows of plain nodes that
`planFusions` did not fold, filtered down to the ones that already satisfy every eligibility
condition the fusion pass applies to all rules. GPU-free — only the safetensors header of each
component's first shard is read (the IR travels in `__metadata__.karume_ir`).

This is stage 1 of the semi-automatic fusion discovery pipeline. It answers "where could a fusion
rule be written?", not "should it be?".

## What it is not

The output is a list of **candidates, not a design**. A candidate only means: these adjacent nodes
form a chain whose intermediate values stay inside the window, are f32, touch no state slot, and
have single-output contracts. It says nothing about whether folding them preserves rounding
positions, whether a private kernel can be written for them, or whether the GPU time is worth it.
The acceptance set of ADR 0040 (decision 2) is unchanged by this tool, and widening it stays a
deliberate, per-rule decision.

Ranking by GPU time is out of scope: mapping an IR node to a kernel key needs a `GpuContext`, so
candidates are cross-checked by hand against the dispatch census an e2e run already prints.

## Usage

```
deno run -A tools/fusion-hints/main.ts enumerate --source <dir> [options]
```

`--source` is either a distribution mirror (a directory holding `karume.json`) or a series output
directory (every subdirectory holding `model*.safetensors` is one component, and its shard sequence
is resolved to the first shard).

| Option                                | Meaning                                                                                                                                                            |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--out <dir>`                         | Write `candidates.jsonl` and `candidates.md` there. Without it the Markdown goes to stdout.                                                                        |
| `--max-window <n>`                    | Longest contiguous window to consider (default 9, minimum 2).                                                                                                      |
| `--top <n>`                           | Rows per graph in the Markdown table (default 10). The jsonl always holds every row.                                                                               |
| `--model <name>`                      | Distribution form only: which model in the manifest (default: `defaultModel`).                                                                                     |
| `--quant <name>`                      | Distribution form: the manifest quant (defaults to `defaultQuant`). Series form: the storage-dtype group (`model.i8-*.safetensors` → `i8`)                         |
| `--family <name>`                     | The family, which selects the default scenario set. Required for a series output whose directory name the tool cannot map to a family (see below)                  |
| `--scenario <name>=<SYM>:<value>[,…]` | Symbol bindings. May be repeated; each one produces a separate report. Defaults to the family's built-in scenarios                                                 |
| `--no-fusion`                         | Enumerate against a plan with fusion switched off — every rule the pass would fold shows up as a candidate. Used to check the enumerator against known hit counts. |

```
# distribution mirror, current plan, the family's default scenarios
deno run -A tools/fusion-hints/main.ts enumerate --source models/karume-anima \
    --out outputs/bench/karume-anima/2026-09-03_fusion-hints

# decode series, one explicit scenario, fusion switched off (reproduces the known hit counts)
deno run -A tools/fusion-hints/main.ts enumerate --source outputs/series/gemma4-e2b-decode \
    --scenario decode=M:1,C:640 --no-fusion
```

## Scenario bindings

A static IR keeps its symbolic dimensions (`M` and `C` for gemma4, `S` for the Anima transformer,
`T` / `P` for the speech families). Concrete numbers only exist for a _particular run_, so the
enumerator needs a scenario: a map from symbol to value. Candidate counts depend on it — Gemma 4
decode folds 15 RoPE chains at `M=1` and none at `M=32` — so the binding is always stated in the
output.

A binding key is either `SYM` (applies to every component) or `<component>.SYM` (applies to that
component only). The qualified form is necessary because the same symbol name can mean different
things in different components — in Irodori, `T` is a token count in `backbone` but a frame count in
`codec_encoder`.

Every family has a built-in default scenario set with its provenance recorded in the report. Two
things are **hard errors**, not skips: a symbol the component's shapes use that the scenario does not
bind, and a qualified key naming a symbol the component does not declare (a misspelling on either
side of the dot). Enumerating against a half-bound plan, or dropping a binding that was meant to
apply, would silently produce a candidate table for a shape no run ever has, so the tool refuses
instead of falling back to a default value. A scenario name may not be repeated: the census
`summary.json` is joined to this report by that name.

The family selects which default scenario set applies, and it is inferred from the manifest's
pipeline id (distribution form) or from the series directory name. Directory names that do not start
with a known family — `outputs/series/embeddinggemma-300m`, for instance — cannot be inferred, so
`--family` is required there. The value is not checked against the known families: it is only a
lookup key for the default scenario table, and an unknown one fails when that lookup happens (an
explicit `--scenario` makes the family irrelevant).

The scenario table (`tools/_shared/scenario.ts`) is shared with `tools/opbench`, so a candidate
table and a census of the same asset are taken under the same bindings and can be read side by side.
The binding checks above live there as well, so both tools reject the same mistake with the same
message.

## Output

`candidates.md` — one section per scenario (its bindings and their provenance), and inside it one
subsection per graph: node count, the symbol values that reached that component, the rule hit counts
of the current plan, and the top candidate chains.

`candidates.jsonl` — one JSON record per line, tagged by `kind`. Every field name is `snake_case`,
as in the census rows:

- `graph` — the asset identity (`family`, `model`, `quant`, `component`), the scenario
  (`scenario`, `bindings`, `binding_source` — spelled as in the census `summary.json`), the graph's
  `path` and `node_count`, `symbols` (what this component was actually bound to), the `FusionCounts`
  of the current plan, `max_window`, and `window_count` (the total candidate windows).
- `candidate` — one op-name chain: `ops`, `count`, `maximal`, `window_sizes`, and an `example`
  (`node_index` is a step-order index; it equals the IR node index only with `--no-fusion` — use
  `output_name` to locate the chain).
- `skipped` — a file that could not be read, with the reason. Nothing is dropped silently. An
  unbound symbol is _not_ a skip: it aborts the run.

Reading the table:

- **maximal** — a window of length 7 also produces its length 2…6 prefixes, so counts alone
  over-report. A row's `maximal` counts the occurrences where no longer window was accepted from the
  same head node. Rows are ordered by it, so the top of the table is already the longest chains
  rather than their prefixes.
- **window_sizes** larger than `ops.length` mean the window carries an in-window passthrough (a node
  the chain reads but cannot fold, such as the shared cos/sin table slice of RoPE). The passthrough
  is excluded from the chain, exactly as the existing rules declare it.

## `inductor` — does Inductor fuse the chains karume does not?

```
deno run -A tools/fusion-hints/main.ts inductor --out <dir> --candidates <candidates.jsonl> [...]
```

Runs `inductor_probe.py` in the CUDA venv (`--venv`, `KARUME_CUDA_VENV`, default
`~/workspace/karume-cuda-venv`). The probe exports the exporter's golden models (the 31 tiny modules
that cover the op contract) plus a handful of chain modules written for the chains that top the
candidate tables (post-norm residual `rms_norm,add`, `linear,rms_norm,add`, gated `gelu_tanh,mul`,
the RoPE half-rotation, `bmm,softmax,bmm`, adaLN modulation), and asks Inductor — via a patched
`Scheduler.codegen` — which nodes it would put into one kernel. Node names are joined to the IR the
exporter produces for the same module, so every fused group is reported as a sequence of IR ops.

`comparison.jsonl` then labels every candidate chain from the given `candidates.jsonl` files:
`fused` (a probe model shows Inductor putting that op sequence into one kernel — the witness names
the model and group), `split` (all ops were observed but never in one kernel), `unobserved` (an op
in the chain occurs in no probe model — the verdict needs a model that has it), `trivial` (the chain
is only reshapes). This is a structural check: the probe models are tiny, so Inductor's shape-dependent
decisions (tiling, reduction splits) are out of scope until a real asset's ExportedProgram is fed in.

## Where the judgment lives

The tool contains no eligibility judgment of its own. `enumerateUnfusedWindows` lives in
`packages/runtime/src/runtime/fusion.ts` next to the fusion rules and calls the same private
predicates the pass applies to every rule, so a change to the acceptance set moves the candidate
table with it (ADR 0040 decision 1: one judgment point). This module only decides how to bind the
symbols of a graph and how to count what comes back.

Which graphs to read is not decided here either: `tools/_shared/assets.ts` resolves an asset
directory to one first shard per component and reads the IR out of its safetensors header, shared
with `tools/opbench` so both tools see the same components — under the same names, which is what
makes a `<component>.SYM` binding mean the same thing in both. Storage dtype follows the quant table
there; candidates do not depend on it (a chain is decided by the node sequence alone), but resolving
through one path keeps the two tables pointing at the same shard. The device limits the plan is
built against (`CORE_LIMITS`) come from there too, so a candidate table and a census can be read
side by side.

Because of that the tool imports `packages/runtime/src/…` directly rather than the published
`mod.ts` surface: the fusion pass is internal on purpose, and this is in-repo tooling, not a
consumer (ADR 0008, addendum of 2026-09-03).

## Checked by

`enumerate_test.ts`, next to the code — the aggregation on a synthetic IR, the scenario vocabulary
(a `<component>.SYM` binding reaching one component only, an unbound symbol aborting, a qualified
key naming an undeclared symbol aborting, the removed `--bind` / `--default-symbol` flags and any
unknown `--flag` failing with a reason, a repeated scenario name rejected), plus the real-asset gate:
the default
scenarios run end to end, with fusion switched off the known hit counts reappear as candidates, and
with the current plan the folded chains are gone. Expected values come from
`packages/runtime/tests/assets_fusion_counts_test.ts`; a source whose assets are absent is skipped
by name, not silently.

The enumerator itself (which contiguous windows are eligible) is checked on synthetic graphs in
`packages/runtime/tests/runtime_fusion_hints_test.ts`, next to `fusion.ts`.
