# @karume/test-support

Private workspace package: the Node test harness for the Karume monorepo (Vitest on Node 24 with
Dawn as the WebGPU implementation), plus the codemods used while porting the Deno test suite.

## Modules

| Import | Contents |
| --- | --- |
| `@karume/test-support/setup` | Vitest `setupFiles` entry. Creates one Dawn `GPU` per worker process, installs it as `navigator.gpu`, and installs the WebGPU namespaces / error classes (`GPUBufferUsage`, `GPUMapMode`, `GPUOutOfMemoryError`, ...) as globals, as Deno and browsers have them. `KARUME_DAWN_FLAGS` passes Dawn toggles (e.g. `disable-dawn-features=timestamp_quantization`). |
| `@karume/test-support/test` | Deno-shaped registration on Vitest: `test(name, fn)` / `test({ name, ignore, only, fn })`, `describe` / `it` (string and object forms), `beforeAll` / `afterAll` / `beforeEach` / `afterEach`, and a `TestContext` whose `t.step(...)` runs sub-steps sequentially inside the test (a failed step records its error, later steps still run, the test fails afterwards). Deno's sanitizer options are accepted and ignored. |
| `@karume/test-support/fs` | Deno-shaped file helpers on `node:fs`: `readFile`, `readTextFile`, `writeFile`, `writeTextFile` (+ `Sync` forms), `mkdir`, `remove`, `makeTempDir`, `stat`, `readDir`, `symlink`, `readLinkSync`, `realPathSync`, `readRange(path, offset, length)`, and the error predicates `isNotFound` / `isADirectory` / `isFilesystemLoop`. Byte results always own their `ArrayBuffer`. |
| `@karume/test-support/webgpu` | `platformObject<T>(fields)` — supplies the nominal `__brand` that `@webgpu/types` puts on platform interfaces, for hand-built fakes. |

## Porting a Deno test file (phase 1 rules)

Phase 1 is a **behaviour-preserving port**: the test keeps its assertions, its structure, its
names and its Japanese text. Do not fix, prune, or restructure tests in this phase.

1. Run the codemod: `node packages/test-support/codemod/port-deno-test.mjs --write <files>`.
   It rewrites `Deno.test` → `test`, `@std/testing/bdd` → `@karume/test-support/test`,
   `Deno.env.get` → `process.env`, the file APIs → `@karume/test-support/fs`,
   `x instanceof Deno.errors.NotFound` → `isNotFound(x)`, `Deno.exit` / `Deno.memoryUsage` /
   `Deno.build.*` → `process.*`, and adds the imports. Lines it cannot map are printed as `REMAINING`.
2. Port the `REMAINING` lines by hand with the same shape:
   - `Deno.open` + `seek` + `read` loops → `readRange(path, offset, length)`;
   - `Deno.Command` → `node:child_process` (`execFile` promisified; a missing binary is `isNotFound`);
   - `Deno.version.*` / `Deno.build.*` → `process.versions.*` / `process.platform` / `process.arch`;
   - `Deno.serve` → `node:http`;
   - hand-built platform objects that fail with "Property '__brand' is missing" → `platformObject<T>({...})`.
3. Type-check: `node packages/test-support/codemod/check-ported.mjs <files>` must print 0 errors.
   (The gate hides errors in files that still reference Deno; everything else must be clean.)
4. Run: `vp test <files>`. Wrap runs that exercise the GPU (family lanes, runtime GPU tests) in
   `flock /tmp/karume-gpu.lock …` so concurrent sessions do not share the device; CPU-only files
   (the Vitest setup only creates the Dawn instance) do not need the lock. Compare the Vitest summary with the Deno baseline for the same files
   (`node packages/test-support/codemod/deno-baseline.mjs <files>` prints the Deno counts from
   `outputs/misc/0.15-baseline-2026-10-16/deno-junit/*.xml`): the number of passed tests and of
   skipped (= Deno "ignored") tests must match. Sub-steps are not counted by Vitest; a mismatch in
   top-level counts is a porting defect until proven otherwise.
5. Report every judgement call (anything beyond the mechanical rules) in the result; do not widen
   the change to other files.

## Known differences between Deno and Node runs

- Environment key: `node-<adapter device>` (Dawn puts the driver version into `description`), e.g.
  `node-nvidia-geforce-rtx-3080-ti` next to Deno's `deno-nvidia-geforce-rtx-3080-ti`.
- `results.json` records `runtime.name = "node"`, Node / V8 / TypeScript versions, and Node's
  `process.platform` / `process.arch` spellings (`linux` / `x64`).
- Dawn exposes `subgroups` and `packed_4x8_integer_dot_product` on this machine, so cases that were
  explicit SKIPs under Deno's wgpu run here.
- No per-test resource / op sanitizer exists in Vitest (Deno's `sanitizeResources` / `sanitizeOps`).
