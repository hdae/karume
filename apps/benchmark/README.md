# Karume benchmark

A small browser page for outside collaborators: it runs the GEMV sweep's and the geometry sweep's `explore` sets on
the collaborator's own GPU (`@karume/runtime/tune`), shows how far it got and how long it has left, and leaves one
JSON file to send to the maintainer. It needs no workbench, no local model mirror and no dev-server endpoint: the
built site runs from any static host. The app is private (not published). What it does is specified in
[docs/spec/benchmark.md](../../docs/spec/benchmark.md).

```sh
export PATH=$HOME/.local/share/vite-plus/bin:$PATH   # Node 24 + pnpm from Vite+
pnpm install
pnpm --filter @karume/benchmark dev                  # http://127.0.0.1:8792
pnpm --filter @karume/benchmark build                # a static site in apps/benchmark/dist
```

A short trial narrows the run by URL: `?cases=<id>,<id>` (case ids of either sweep's table; a sweep none of them
belongs to does not run) and `?idleSeconds=<s>` (default 0: no idle time between cases), for example
`http://127.0.0.1:8792/?cases=i8-n256-k1536,i8a8-conv3d-cin256-cout256-k333-in6x128x128&idleSeconds=0`. A run of the
defaults (both sweeps, every case) takes about 1 h 7 min.

## The file a collaborator sends

`karume-benchmark_<host>-<adapter name>_<UTC time>.json` (`_incomplete` when a cancel, a device loss or an error
stopped the run) holds the run's workbench records exactly as the workbench writes them, wrapped with what the page
knows of the run. Unpack it into `outputs/workbench/` and hand the records to the table generator:

```sh
node apps/benchmark/unpack.ts ~/Downloads/karume-benchmark_browser-amd-…_2026-11-27T10-00-00.json
node tools/geometry-profiles/generate.ts <profile id> --records=outputs/workbench/browser-amd-…/geometry-sweep_….json \
  --gemv-records=outputs/workbench/browser-amd-…/gemv-sweep_….json --dry-run
```

`unpack.ts` writes nothing when a record's file exists already (its name is the record's own time, so unpacking the
same file twice is refused, not duplicated).

## Layout

| Path | What it holds |
| --- | --- |
| `src/plan.ts` | What a run measures: the two sweeps planned by the shared settings, the URL's trial, each case's kind |
| `src/controller.ts` | The run: the sweeps one after another, a record each, the bundle, the download on completion |
| `src/progress.ts` | The two progress bars, the time left and the patterns, folded from the run's events |
| `src/estimate.ts` | The estimate of the time left and its priors |
| `src/patterns.ts` | The optimisable patterns found, by the derivations' own threshold and predicates |
| `src/bundle.ts` | The file: its form, its name, its reader and its records as workbench files |
| `src/notes.ts`, `src/visibility.ts`, `src/wake-lock.ts` | The words before and after a run and the start rule; how long the page was hidden; the screen wake lock |
| `src/main.ts`, `src/dom.ts`, `src/style.css` | The page (the only DOM code) |
| `unpack.ts` | The Node command that writes a file's records where the workbench would have |

The per-case loop, the record envelope and its path rule, and the sweeps' plans, runs, merged records and summaries are
[`@karume/sweep-app-support`](../../packages/sweep-app-support/README.md), shared with the workbench.

## Develop

```sh
pnpm --filter @karume/benchmark test        # vp test: plan, estimate, progress, patterns, notes, wake lock, bundle and unpack, the run on fake sweeps (no GPU)
pnpm --filter @karume/benchmark typecheck
```
