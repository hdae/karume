# Karume workbench

A bare browser app for measurements and checks on a real GPU. One file is one **entry**; the page
lists the entries, shows the selected entry's parameters, runs it on a click on **Run**, and keeps the
result as a JSON **record**. The same entry modules also run under Node + Dawn (`run.ts`). The app is
private (not published).

| Entry | What it does |
| --- | --- |
| `kernel-check` | The kernel identity checks of `@karume/kernel-checks` — the cases and criteria of the runtime's node GPU tests — on this host's backend (D3D12 in Chrome on Windows, Metal on macOS, Dawn under Node). Parameter `checks`: the checks to run (default: all twelve, the release check). The verdict rules are in `src/kernel-check/verdict.ts`, the checks in [the package README](../../packages/kernel-checks/README.md). |

## Run in a browser

```sh
export PATH=$HOME/.local/share/vite-plus/bin:$PATH   # Node 24 + pnpm from Vite+
pnpm install
pnpm --filter @karume/workbench dev
```

The dev server listens on **http://127.0.0.1:8791** (loopback only; the playground uses 8790). From
another machine, open it through an SSH tunnel as described in
[the playground README](../playground/README.md#from-another-machine-ssh-tunnel), with port 8791.
Every response carries COOP / COEP, so the page is cross-origin isolated.

The URL holds the selected entry, its parameters and the declared backend
(`?entry=kernel-check&checks=gemv-integer,gemv-planes&backend=d3d12`), so a link reproduces a run
setup; nothing runs before a click on Run. WebGPU does not tell the backend: type it in the
**backend (optional)** field, or leave it empty and the record has none. Cancel stops the run at the
entry's next check point; a cancelled or failed run still makes a record.

Under the dev server the page POSTs each record to the server, which writes it under `outputs/` and
shows the path. **Download record** and **Copy record** give the same text. A production build
(`pnpm --filter @karume/workbench build`, served by `vp preview`) writes nothing: download the record.

## Run under Node

```sh
flock /tmp/karume-gpu.lock node apps/workbench/run.ts kernel-check
flock /tmp/karume-gpu.lock node apps/workbench/run.ts kernel-check --checks=gemv-integer,gemv-planes
node apps/workbench/run.ts --list
```

Parameters are `--name=value` (a `choices` list is comma-separated). The registry and the entries
load through Vite's module runner, so the run sees the same modules as the page. Dawn's toggles come
from `KARUME_DAWN_FLAGS` as in the tests; a `backend=<name>` toggle selects the backend and is
recorded, otherwise the record names Dawn's default for the platform (Vulkan on Linux). Progress goes
to stderr; the outcome, the summary and the record's path go to stdout. Exit code: 0 when the run is
done and its verdict is pass, incomplete or none; 1 for fail, error or cancelled (Ctrl-C cancels); 2
for a command line that names no entry or a parameter that does not fit.

## Records

A record lands at `outputs/workbench/<host>-<adapter name>/<entry id>_<UTC stamp>.json` (outside git):
`node-nvidia-geforce-rtx-3080-ti`, `browser-nvidia-geforce-rtx-5070-ti`. The adapter name is the
description in a browser and the device under Node (Dawn puts the driver version in the description),
else vendor-architecture. A second record in the same second gets `-1`, `-2`, …; no file is ever
replaced. The rules are in [docs/spec/workbench.md](../../docs/spec/workbench.md).

```text
format       "karume-workbench/1"
entry        { id, parameters }              the typed parameter values of the run
recordedAt   when the run ended (ISO 8601)
seconds      wall time of the entry's run
outcome      "done" | "error" | "cancelled"   (error: the stack in `error`)
environment  host ("browser" | "node"), userAgent, crossOriginIsolated (browser), backend
             ({ name, source: "declared" | "dawn-flag" | "dawn-default" }, when known), the adapter
             (vendor, architecture, device, description), its features and limits, WGSL language
             features — read from an adapter the host requests itself
checkout     { commit, dirty, runtimeVersion }   (dev server: at run start; build: at build time)
verdict      "pass" | "fail" | "incomplete"   (entries that conclude something)
summary      the lines the page shows first
data         the entry's own record body
```

## Add an entry

Add one file `src/entries/<id>.ts` (kebab-case) whose default export is an `Entry`
(`src/entry.ts`): `id` (the file name), `title`, a one- or two-sentence `description`, the declared
`parameters` (`boolean`, `integer` with min / max, `choice`, `choices`) and `run`. `run` gets the
typed parameters, `gpuOptions` to spread into `acquireGpu` (it acquires and destroys its own GPU
context), an `AbortSignal` and `progress(text)`, and resolves to `{ verdict?, summary, data }`. An
entry holds no UI code and no Node API: the shell and `run.ts` render and drive every entry the same
way. The registry (`src/entries.ts`) picks the file up; nothing else lists entries.

## Develop

```sh
pnpm --filter @karume/workbench test        # vp test: parameters, registry, record, write-back, the kernel-check entry (no GPU)
pnpm --filter @karume/workbench typecheck
pnpm --filter @karume/workbench build
```
