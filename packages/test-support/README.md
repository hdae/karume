# @karume/test-support

Private workspace package: the Node test harness for the Karume monorepo (Vitest on Node 24 with
Dawn as the WebGPU implementation), plus the codemods used while porting the Deno test suite.

## Modules

| Import | Contents |
| --- | --- |
| `@karume/test-support/setup` | Vitest `setupFiles` entry. Creates one Dawn `GPU` per worker process, installs it as `navigator.gpu`, and installs the WebGPU namespaces / error classes (`GPUBufferUsage`, `GPUMapMode`, `GPUOutOfMemoryError`, ...) as globals, as Deno and browsers have them. `KARUME_DAWN_FLAGS` passes Dawn toggles (e.g. `disable-dawn-features=timestamp_quantization`). |
| `@karume/test-support/fs` | Deno-shaped file helpers on `node:fs`: `readFile`, `readTextFile`, `writeFile`, `writeTextFile` (+ `Sync` forms), `mkdir`, `remove`, `makeTempDir`, `stat`, `readDir`, `symlink`, `readLinkSync`, `realPathSync`, `readRange(path, offset, length)`, and the error predicates `isNotFound` / `isADirectory` / `isFilesystemLoop`. Byte results always own their `ArrayBuffer`. |
| `@karume/test-support/test-assets` | The pin of the private test-asset dataset (`TEST_ASSETS` — the single source of truth, ADR 0139), its local root `outputs/test-assets/` (`TEST_ASSETS_ROOT`), `goldenDir(series)` / `corpusDir(kind)`, and `readFetchedRevision()` (the revision the last completed `node tools/test-assets/fetch.ts` recorded). |
| `@karume/test-support/webgpu` | `platformObject<T>(fields)` — supplies the nominal `__brand` that `@webgpu/types` puts on platform interfaces, for hand-built fakes. |

## Writing tests

Tests use Vitest's own registration — `import { describe, it, test, beforeAll, afterAll } from
"vite-plus/test"`, with `test.skipIf(cond)` / `describe.skipIf(cond)` for conditions known at
collection time and `context.skip()` for conditions known only at run time — and `@std/assert`
assertions. Sequential sub-steps are `it`s inside a `describe`: Vitest runs them in order and a
failing `it` does not stop the following ones. Runs that exercise the GPU are wrapped in
`flock /tmp/karume-gpu.lock …` so concurrent sessions do not share the device; CPU-only files do
not need the lock.

The codemods under `codemod/` are the record of the Deno port: `port-deno-test.mjs`,
`move-e2e.mjs`, `check-ported.mjs` and `deno-baseline.mjs` did the phase-1 port, and
`port-shim.mjs` replaced the Deno-shaped registration shim with Vitest's forms in phase 2.

## Known differences between Deno and Node runs

- Environment key: `node-<adapter device>` (Dawn puts the driver version into `description`), e.g.
  `node-nvidia-geforce-rtx-3080-ti` next to Deno's `deno-nvidia-geforce-rtx-3080-ti`.
- `results.json` records `runtime.name = "node"`, Node / V8 / TypeScript versions, and Node's
  `process.platform` / `process.arch` spellings (`linux` / `x64`).
- Dawn exposes `subgroups` and `packed_4x8_integer_dot_product` on this machine, so cases that were
  explicit SKIPs under Deno's wgpu run here.
- No per-test resource / op sanitizer exists in Vitest (Deno's `sanitizeResources` / `sanitizeOps`).
