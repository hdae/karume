# The workbench

What `apps/workbench` does now: the browser app (and the Node command) that runs the repository's checks and
measurements on a GPU and leaves one JSON record per run. It is the app ADR
[0152](../decisions/0152-maintained-specs-and-declared-numeric-requirements.md) decision 9 calls "the harness"; the
maintainer named it the workbench, because "harness" already names the test harness (`packages/test-support`) and the
sweep's measurement core (`packages/runtime/src/tune/harness.ts`).

The workbench is private (not published) and is not part of the lanes: its tests run with
`pnpm --filter @karume/workbench test` and need no GPU.

## An entry

- One file `apps/workbench/src/entries/<id>.ts` is one entry. Its default export declares an `id` (kebab-case, equal
  to the file name), a `title`, a `description`, its `parameters` and `run`. Nothing else lists the entries: the
  registry collects the files, and an id that is not its file name is an error at start.
- An entry holds no UI code. The browser shell and the Node command render and drive every entry the same way, and the
  same module runs in both.
- Parameters are declared, of four kinds: `boolean`, `integer` (with a minimum and a maximum), `choice` (one of a
  list) and `choices` (a subset of a list, written as a comma-separated list). Values arrive as strings — a URL query,
  `--name=value` on the command line, the form — and one function turns them into typed values. A missing name takes
  its default; a name the entry does not declare and a value that does not fit are errors that name the parameter.
- `run` receives the typed parameters, the GPU options of the host (`{}` in a browser, Dawn's `GPU` under Node), an
  abort signal and a function that reports one status line. An entry acquires and releases its own GPU context.
- `run` resolves to an optional `verdict` (`pass`, `fail` or `incomplete`), a `summary` (a few lines a person reads
  first) and `data` (the entry's own record body).

### The entries

| Entry | What it runs | Inputs |
| --- | --- | --- |
| `kernel-check` | The twelve kernel identity checks of `@karume/kernel-checks` — the cases and criteria of the runtime's node GPU tests — on the host's backend. Parameter `checks`: which of the twelve (default: all). `pass` needs every check run and passed; a check the device cannot run is not available, never a pass. | Synthetic (built in memory); no distribution |

## The record

Every run leaves one record, format `karume-workbench/1`:

| Field | Content |
| --- | --- |
| `format` | `karume-workbench/1` |
| `entry` | `id` and the typed `parameters` the run used |
| `recordedAt`, `seconds` | When the run ended (ISO 8601) and the wall time of `run` |
| `outcome` | `done`; `error` (`run` threw: `error` holds the stack or message); `cancelled` (the signal was aborted before `run` settled — what the entry returned by then is kept) |
| `environment` | `host` (`browser` or `node`), `userAgent`, `crossOriginIsolated` (browser only), `backend` (optional), `adapter` (vendor, architecture, device, description), the adapter's `features` (sorted), every numeric adapter `limits` value and `wgslLanguageFeatures` (sorted) |
| `checkout` | `commit`, `dirty` and `runtimeVersion` (the version of `packages/runtime/package.json`) |
| `verdict`, `summary`, `data` | What the entry returned |

- The environment is read from an adapter the host requests itself, not from the entry.
- WebGPU does not tell the backend. In a browser a person may declare it (the shell's field or `backend=` in the
  URL; `source: "declared"`), and an undeclared backend leaves the field out. Under Node it is the `backend=<name>`
  token of the Dawn flags (`source: "dawn-flag"`), else Dawn's default for the platform (`source: "dawn-default"`).
- The checkout is read when the run starts: by the dev server at that moment for a browser run, from git for a Node
  run. A production build carries the checkout its build read.
- The saved text is the record as JSON with one-space indentation and a final newline. The download, the copy, the
  dev server's file and the Node command's file are that same text.

## Where records go

- `outputs/workbench/<host>-<adapter name>/<entry id>_<UTC time>.json`, outside git. `<host>` is `browser` or `node`.
  The adapter name follows the rule of the tests' environment key: the adapter's `description` in a browser and its
  `device` under Node (Dawn puts the driver version in the description, and the directory does not change with a
  driver update), in lower case with every run of characters other than `a`–`z` and `0`–`9` as one hyphen (at most 80
  characters); an empty name falls back to vendor and architecture, then to `unknown-adapter`. Examples:
  `node-nvidia-geforce-rtx-3080-ti`, `browser-nvidia-geforce-rtx-5070-ti`.
- The time is the writer's clock (`YYYY-MM-DDTHH-MM-SS`); a taken name gets `-1`, `-2`, …, and an existing file is never
  replaced.
- A browser run under the dev server posts its record, and the dev server writes it; the page says where. A machine
  that reaches the dev server through an SSH tunnel therefore leaves its records on the development machine.
- A production build writes nothing: the page offers the download and says the record is not saved.
- A cancelled run and a failed run leave a record like any other.

## The dev server

`vp dev` in `apps/workbench` listens on `127.0.0.1:8791` only.

| Request | Answer |
| --- | --- |
| `POST /workbench/results` | Writes the body as a record and answers `201` with `{ "path": … }` (repository-relative). `405` for another method; `415` unless `Content-Type` is `application/json`; `403` unless the `Origin` header is the server's own origin; `413` for a body over 64 MiB (not read past the limit); `422` unless the body is JSON with the format above, a kebab-case `entry.id`, an `environment.host` of `browser` or `node` and an `environment.adapter` object. The bytes written are the body, unchanged. |
| `GET /workbench/checkout` | The checkout at the moment of the request. |

- Nothing the client sends becomes part of a path except through the two rules above (the kebab-case entry id and the
  adapter directory).
- The preview server and a production build have neither endpoint.
- Every response of the dev and preview servers carries `Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: require-corp`, so the page is cross-origin isolated.

## The shell

- A list of the entries; the selected entry's parameter form, generated from its declaration; an optional backend
  field; Run and Cancel; one status line.
- After a run: the verdict (or the outcome of a run that did not finish), the summary, where the record was saved, the
  buttons "Download record" and "Copy record", and the record's text (its first 200,000 characters).
- The selected entry, the parameter values and the declared backend are in the URL. Nothing runs without a click on
  Run. One run at a time.
- Without WebGPU or without an adapter the page says so in place of Run.

## The Node command

`node apps/workbench/run.ts <entry id> [--name=value …]` runs an entry on Dawn and writes its record through the same
path rule; `--list` prints the entries. It uses the GPU, so it runs under the GPU lock
(`flock /tmp/karume-gpu.lock …`).

| Exit code | When |
| --- | --- |
| 0 | The run is done and its verdict is not `fail` |
| 1 | The verdict is `fail`, `run` threw, or the run was cancelled (SIGINT) |
| 2 | The command line names no entry, an unknown entry, or a parameter that does not fit |

Progress goes to stderr; the outcome, the summary and the record's path go to stdout.

## What it does not do

- It collects nobody else's results: a record stays where it was written.
- It reads no model distribution. An entry that needs one is not built yet.
- It keeps no history and compares no two records.
