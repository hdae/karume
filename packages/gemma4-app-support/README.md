# @karume/gemma4-app-support

Private workspace package (not published): the Gemma 4 helpers that the playground's chat page
(`apps/playground`, `/gemma4`) and the workbench's Gemma entries (`apps/workbench`,
`gemma4-session-ab` and `gemma4-bench`) both use, in one implementation, so the two apps parse the
same text forms and record stops and timings the same way. Nothing published may import it.

| Subpath | What it holds |
| --- | --- |
| `./settings` | The three Gemma 4 mounts (`GEMMA_MOUNTS`, `isGemmaMount`) and the entry point each is built with (`entryPointOf`), `ModelKey`, `isQatModel`, a model's validated `pipelineConfig` (`modelConfig`), which model takes the exact expert cache (`hasExpertCache`), and the expert cache's text forms: the slots (`parseExpertCacheSlots`, `EXPERT_CACHE_SLOTS`, `FITTED_EXPERT_SLOTS`), the host tier (`parseExpertHostTier`) and the self-speculation depth's range (`EXPERT_SPECULATION_DEPTH`). |
| `./stop-record` | `stopRecordOf`: how a generation stopped, flattened for JSON (`StopRecord`). |
| `./turn-timing` | `turnTiming`: a generation's prefill, time to first token and decode from its `performance.now()` marks (`TurnMarks`, `TurnTiming`). The decode ends at the last decode run when the marks carry it (`lastRunAt`, a pipeline built with `onRunDiagnostics` — the chat page's diagnostics), else at the stream's end (the workbench). |

What only one app uses stays in that app (the chat page's forms and model choices, the workbench's
`<mount>/<model>` parsing and pipeline builder). Pure modules; the tests (`tests/`) run in the core
lane and need no GPU and no mirror.
