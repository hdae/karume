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
directory (the tool walks the tree and picks the first shard of every component).

| Option                     | Meaning                                                                                                                                                            |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--out <dir>`              | Write `candidates.jsonl` and `candidates.md` there. Without it the Markdown goes to stdout.                                                                        |
| `--max-window <n>`         | Longest contiguous window to consider (default 9, minimum 2).                                                                                                      |
| `--top <n>`                | Rows per graph in the Markdown table (default 10). The jsonl always holds every row.                                                                               |
| `--model <name>`           | Distribution mirrors only: which model in the manifest (default: `defaultModel`).                                                                                  |
| `--bind <symbol>=<value>`  | Bind one shape symbol. Repeatable.                                                                                                                                 |
| `--default-symbol <value>` | Bind every symbol left unbound by `--bind`.                                                                                                                        |
| `--no-fusion`              | Enumerate against a plan with fusion switched off — every rule the pass would fold shows up as a candidate. Used to check the enumerator against known hit counts. |

Symbols are never bound implicitly: a graph with an unbound symbol is reported and skipped. Hit
counts depend on them (Gemma 4 decode folds 15 RoPE chains at `M=1` and none at `M=32`), so the
binding is always stated in the output.

```
# distribution mirror, current plan (what is still unfused)
deno run -A tools/fusion-hints/main.ts enumerate --source models/karume-anima \
    --bind S=4096 --bind T=64 --bind Tsrc=64 --bind Ttgt=512 \
    --out outputs/bench/karume-anima/2026-09-03_fusion-hints

# decode series, fusion switched off (reproduces the known hit counts)
deno run -A tools/fusion-hints/main.ts enumerate --source outputs/series/gemma4-e2b-decode \
    --bind M=1 --bind C=640 --no-fusion
```

## Output

`candidates.md` — one section per graph: node count, symbol bindings, the rule hit counts of the
current plan, and the top candidate chains.

`candidates.jsonl` — one JSON record per line, tagged by `kind`:

- `graph` — the graph's path, node count, bindings, `FusionCounts` of the current plan, total
  candidate windows.
- `candidate` — one op-name chain: `ops`, `count`, `maximal`, `windowSizes`, and an `example`
  (step-order index; equals the IR node index only with `--no-fusion` — use `outputName` to locate
  the chain).
- `skipped` — a file that could not be read, with the reason. Nothing is dropped silently.

Reading the table:

- **maximal** — a window of length 7 also produces its length 2…6 prefixes, so counts alone
  over-report. A row's `maximal` counts the occurrences where no longer window was accepted from the
  same head node. Rows are ordered by it, so the top of the table is already the longest chains
  rather than their prefixes.
- **windowSizes** larger than `ops.length` mean the window carries an in-window passthrough (a node
  the chain reads but cannot fold, such as the shared cos/sin table slice of RoPE). The passthrough
  is excluded from the chain, exactly as the existing rules declare it.

## Where the judgment lives

The tool contains no eligibility judgment of its own. `enumerateUnfusedWindows` lives in
`packages/runtime/src/runtime/fusion.ts` next to the fusion rules and calls the same private
predicates the pass applies to every rule, so a change to the acceptance set moves the candidate
table with it (ADR 0040 decision 1: one judgment point). This module only decides which graphs to
read, how to bind their symbols, and how to count what comes back.

Because of that the tool imports `packages/runtime/src/…` directly rather than the published
`mod.ts` surface: the fusion pass is internal on purpose, and this is in-repo tooling, not a
consumer (ADR 0008, addendum of 2026-09-03).

## Checked by

`enumerate_test.ts`, next to the code — the aggregation on a synthetic IR, plus the real-asset
gate: with fusion switched off the known hit counts reappear as candidates, and with the current
plan the folded chains are gone. Expected values come from
`packages/runtime/tests/assets_fusion_counts_test.ts`; a source whose assets are absent is skipped
by name, not silently.

The enumerator itself (which contiguous windows are eligible) is checked on synthetic graphs in
`packages/runtime/tests/runtime_fusion_hints_test.ts`, next to `fusion.ts`.
