# @karume/sweep-app-support

Private workspace package (not published): what the workbench (`apps/workbench`, its `gemv-sweep` and
`geometry-sweep` entries) and the benchmark app (`apps/benchmark`) share of the two sweeps, in one implementation, so a
record of either sweep is built, merged, summarised and placed the same way whichever app made it. Nothing published
may import it.

| Subpath | What it holds |
| --- | --- |
| `./parameters` | The typed parameters of a run: the six kinds of a declaration, `parseParameters` (the one place strings become typed values) and the accessors (`choiceValue`, `integerValue`, …), `ParameterError` |
| `./record` | The record's envelope (`karume-workbench/1`): its types, `buildRecord`, `recordRun` (times a run and wraps how it settled), `recordText` (the saved bytes), `utcStamp`, `errorText`, the entry id rule, `Verdict` and `EntryResult` |
| `./record-path` | Where a record lands: `outputs/workbench/<host>-<adapter name>/<entry id>_<stamp>.json` (`adapterDirectory`, `recordPath`) |
| `./environment` | The record's environment, read from an adapter the host requests itself (`readEnvironment`), and Dawn's backend (`dawnBackend`) |
| `./per-case-sweep` | One sweep call per case with idle time before every case but the first, its cancel and device-loss rules (`runPerCase`), the live view, the verdict order and the merge's checks |
| `./gemv-sweep/{settings,run,record,summary}` | The GEMV sweep: its entry's parameters and plan, the per-case run, the merged `karume-gemv-sweep/4` record, the summary and verdict |
| `./geometry-sweep/{settings,run,record,summary,k88}` | The geometry sweep: the same for `karume-geometry-sweep/4`, and the K-88 summary |
| `./node` | Node only: the checkout (`readCheckout`) and the exclusive write of a record (`writeRecord`, `writeNewFile`) |

What only one app uses stays in that app (the workbench's entries, shell, batches and dev server; the benchmark's
page, estimate and bundle). The modules are tested through the two apps' tests (`pnpm --filter @karume/workbench test`,
`pnpm --filter @karume/benchmark test`), without a GPU. The specs: [workbench](../../docs/spec/workbench.md),
[benchmark](../../docs/spec/benchmark.md).
