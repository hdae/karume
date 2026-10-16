# 0127: Development toolchain: Node 24 + Dawn with Vite+, Deno as a secondary runtime

- Status: proposed (2026-10-16; to be accepted by the maintainer after review)
- Date: 2026-10-16
- Related: ADR [0001](0001-scope-and-non-goals.md) (scope and non-goals — this ADR revises the
  non-goal "environments without `navigator.gpu`, such as bare Node") /
  [0002](0002-web-standard-api-only.md) (runtime depends on Web standard APIs only — kept unchanged;
  its lint rule moves from a `deno lint` plugin to an Oxlint JS plugin) /
  [0005](0005-verification.md) (verification strategy — the real-GPU stage moves from `deno test` to
  Vitest on Node + Dawn; the lanes move from `deno.json` tasks to Vitest projects) /
  [0008](0008-public-api.md) (thin public surface — the runtime-specific code stays in opt-in
  subpaths) / [0021](0021-gpu-timing-diagnostics.md) (GPU timing — Dawn quantises timestamps by
  default) / [0037](0037-karume-monorepo.md) (the monorepo was a Deno workspace; it becomes a pnpm
  workspace) / [0038](0038-manifest-v1.md) §5 and [0086](0086-distribution-source.md) (hub: the
  common layer stays Web-standard; local file systems are reached through `DirectoryAdapter`
  subpaths) / [0065](0065-exporter-core-recipe-split.md) (the Python exporter and recipes are not
  affected) / [0106](0106-device-keyed-references.md) (environment keys — gains the `node-` runtime
  and a different name source on Dawn) / [0115](0115-geometry-profiles.md) (the geometry sweep takes
  the same explicit WebGPU instance as `acquireGpu`).

## Context

### Why move now

- **Deno's future.** Deno joined Cloudflare on 2026-10-09. Development of the Deno runtime ends after
  one year of maintenance (around 2027-10). JSR continues as a registry. Karume's whole development
  loop (tests, lint, format, type check, task runner, lanes, CI, dependency age guard) depended on
  the Deno CLI, so it would have to move within that year in any case.
- **A viable replacement exists.** Vite+ 1.0 shipped on 2026-09-28. It bundles one toolchain behind
  the `vp` command: Vitest (tests), Oxlint (lint, type-aware), Oxfmt (format), tsdown / Rolldown
  (packaging) and TypeScript 7 (type check), on a pnpm workspace. The npm package `webgpu` (0.6.2)
  is Dawn, the WebGPU implementation that Chrome uses, built as a Node addon.
- **The runtime already runs there.** A spike on 2026-10-15 (Node 24, `webgpu` 0.6.2, RTX 3080 Ti,
  Vulkan backend) installed `create([])` as `navigator.gpu` and imported `@karume/runtime` from source
  **unchanged**:
  - `acquireGpu()` and `acquireGpu({ gpuTiming: true })` succeeded;
  - all 45 elementwise cases (unary / binary / dtype / math groups) matched the CPU reference;
  - the feature-dependent kernels ran green on Dawn's WGSL compiler (Tint): the shader-f16 linear
    variant matched the oracle bit for bit, the subgroup RMSNorm variant passed (that check is an
    explicit SKIP under Deno, whose wgpu lacks the feature), and the packed-dot-product (`dp4a`)
    int8 / int4 kernels matched the emulated path and the TS reference at atol = 0;
  - no WGSL that compiles on Deno's compiler (naga) and fails on Tint was found in that range.
- **Two differences need design**, both outside the published code:
  1. Dawn's `GPUAdapterInfo.description` carries the **driver version**
     (`NVIDIA: 615.71.09 615.71.9.0`), not the GPU name. ADR 0106 decision 2 builds the environment
     key from `description` first and forbids driver versions in the key.
  2. Dawn reports timestamps in nanoseconds but **quantised to 65,536 ns** by default. The toggle
     `disable-dawn-features=timestamp_quantization` brings the step down to 1,024 ns. Short passes
     measure as 0 under the default.

### The size of the move

The published sources hardly touch Deno: the five `Deno.` occurrences in package `src/` are all in
comments, and the only real dependency is the opt-in subpath `@karume/hub/deno`
(`packages/hub/deno.ts`). The cost is in the development side: 434 `*_test.ts` files with about
3,200 `Deno.test` registrations and 692 `t.step` calls, 4,192 lines mentioning `Deno.` across
tests / tools / examples, about 180 relative imports from tools / examples into package internals,
and four checks built on Deno subcommands (`deno doc`, `deno info`, `deno bundle`, `Deno.serve`).

## Decision

1. **Development, tests and CI run on Node 24 with Dawn (npm `webgpu`) as the WebGPU implementation,
   on the Vite+ toolchain (`vp`: Vitest, Oxlint, Oxfmt, tsdown / Rolldown, TypeScript 7) over a pnpm
   workspace.**
   Rationale: the Deno CLI has a bounded future (above), and Vite+ replaces every role the Deno CLI
   played with one pinned toolchain. Dawn is the same implementation family as Chrome's WebGPU, so
   the development runtime and the main target browser now share a WGSL compiler (Tint), and the
   test suite gains the subgroup and packed-dot-product paths that wgpu could not run. The workspace
   is `packages/*` and `tools/*` (`pnpm-workspace.yaml`); versions are centralised with
   `catalog:` (`catalogMode: prefer`); the root declares `engines.node >= 24.0.0` and pins pnpm
   through `devEngines.packageManager` (12.14.0, the version `vp` ships).

2. **The published packages keep their runtime contract: Web standard APIs only (ADR 0002); the only
   runtime-specific code are the opt-in subpaths `@karume/hub/deno` and `@karume/hub/node`; the
   runtime accepts an explicit WebGPU instance through `AcquireGpuOptions.gpu`.**
   Rationale: moving the development host is not a reason to tie the libraries to it. The browser
   remains the primary target, and Node applications choose their own WebGPU implementation.
   - `@karume/runtime` and `@karume/models` MUST NOT depend on `node:` modules, Deno APIs or the
     `webgpu` package. `webgpu` is a development dependency of the test harness only.
   - `@karume/hub/node` (`nodeDirectory`, on `node:fs/promises`) is the Node counterpart of
     `@karume/hub/deno` (`denoDirectory`). Each subpath is the only place in hub that touches its
     host's APIs, holds only "how to read local bytes", and MUST behave the same as the other from
     the common layer's point of view (ADR 0086). Browser code MUST NOT import either.
   - `acquireGpu({ gpu })`, `readAdapterLimits({ gpu })` and `runGeometrySweep({ gpu })` request the
     adapter from the given `GPU` and fall back to `navigator.gpu` when it is omitted, so browsers and
     Deno are unaffected and a Node application can pass Dawn (`create([])`) without defining
     `navigator.gpu` itself (test: `packages/runtime/tests/gpu_injection_test.ts`).
   - The existing invariant still holds: package code MUST NOT read WebGPU globals
     (`GPUBufferUsage`, …) at module scope or rely on them at all — the runtime reads them through
     `src/gpu/webgpu-constants.ts`. Under Node these globals exist only if someone installs them;
     the Vitest setup file installs Dawn's copies for the tests, which is a test-harness convenience,
     not a contract.

3. **Deno becomes a secondary runtime: the packages MUST remain importable from Deno, and publishing
   to JSR continues, but the development suite is not maintained for Deno after 0.15.**
   Rationale: JSR outlives the Deno runtime and existing users import from it; dropping Deno
   consumers at the same time as the toolchain move would make 0.15 a breaking change for reasons
   unrelated to the library. Keeping two full suites alive in one tree buys nothing, because a ported
   test no longer runs on Deno.
   - The Deno check is `tools/published-smoke` run on Deno against the **published** packages (its
     own `deno.json` import map points the bare specifiers at `jsr:@karume/*@<version>`): every
     published repository's manifest resolves through the published `@karume/hub`, and one family
     (sbv2) is constructed with `fromPretrained` when a GPU is present. The same `main.ts` runs on
     Node as the primary check.
   - Numerics on Deno's WebGPU (wgpu) are no longer verified per release beyond that smoke run.
     Behaviour on wgpu is best effort from 0.15 on.

4. **The release-age guard (pnpm `minimumReleaseAge`, 7 days) applies to dependencies; the Vite+
   toolchain's own packages are exempt; the `webgpu` package's install script is refused.**
   Rationale: the guard carries over the intent of Deno's `minimumDependencyAge` (do not install a
   version that has been on the registry for less than a week).
   - `minimumReleaseAge: 10080` (minutes) in `pnpm-workspace.yaml`.
   - `minimumReleaseAgeExclude` lists only the packages that make up Vite+ (`vite-plus`,
     `@voidzero-dev/*`, Vitest, Oxlint and its type-aware backend, Oxfmt, the Oxc and Yuku parsers).
     Their versions are fixed by the installed `vp`, which is an explicit choice of the maintainer;
     the age guard adds no protection there and would only block installs for a week after each
     Vite+ release. Ordinary dependencies MUST NOT be added to the exclusion list.
   - `allowBuilds: { webgpu: false }` is a **refusal**, not an approval. The prebuilt `dawn.node`
     ships in the tarball; the postinstall script only strips the macOS quarantine attribute, so
     Linux installs work without it. No build script is approved in the workspace.

5. **The Deno-era measurement tools and CLI examples (`tools/*`, `examples/*`) are not ported; they
   will be rebuilt later as browser applications in the style of the 0.14.0 `tools/gpu-lab` page.
   The process tools are ported: the lint rule, `published-smoke`, `verify-diff` and the release
   scripts.**
   Rationale: the measurement tools and examples are Deno CLIs that import package internals by
   relative path (about 180 imports, many outside the public surface). Porting them one to one
   would carry that coupling into the new tree. Their purpose — measuring on the real target and
   demonstrating the public API — is better served by browser applications built on the public
   surface; the gpu-lab page already showed that a browser application works for the
   geometry-profile loop and for Wan.
   - Ported as workspace packages: `tools/lint` (the `karume/typed-array-buffer` rule of ADR 0002,
     now an Oxlint JS plugin), `tools/published-smoke`, `tools/verify-diff`, `tools/release`.
   - Not affected: the Python exporter and recipes (`tools/exporter`, `tools/export-recipes`, uv —
     ADR 0065) and the other non-Deno utilities (Python and shell scripts).
   - Every Deno program under `tools/` and `examples/` is removed from the development tree during
     the port, `tools/gpu-lab` included (its page is served by a Deno server and drives package
     internals). They stay available in the history and on `main` at 0.14.0.

6. **ADR 0001's non-goal "environments without `navigator.gpu` (bare Node, etc.)" is revised: Node
   is a supported host for applications and the development runtime.**
   Rationale: with decision 2, a Node application that supplies a WebGPU implementation is a
   first-class host, and the project's own verification now runs there. What stays out of scope is
   an environment with **no** WebGPU implementation at all: there is still no WASM or CPU fallback
   (ADR 0001 non-goal unchanged). ADR 0001's scope sentence ("runs on both Deno and the browser")
   reads, from 0.15 on, as "runs in the browser and on Node, and stays importable from Deno".

## Consequences

### For users

- Node applications can use the packages: pass Dawn (or another implementation) through
  `acquireGpu({ gpu })` and read local distributions with `@karume/hub/node`.
- Deno users keep importing from JSR. Their path is smoke-tested on each release, not covered by
  the full suite.
- Browser users see no change from this ADR.

### For the repository

- `pnpm-workspace.yaml`, the root `package.json`, `vite.config.ts` and a `package.json` per package
  replace the Deno workspace configuration. Each published package's `package.json` declares its
  `exports` with the same entry files as today (`mod.ts`, `tune.ts`, the family subpaths,
  `deno.ts` / `node.ts`).
- The test harness is the private package `@karume/test-support`: the Vitest setup file creates one
  Dawn `GPU` per worker and installs it as `navigator.gpu` (or leaves it undefined on a machine
  without a WebGPU backend), a Deno-shaped registration shim (`test`, `describe`, `t.step`), file
  helpers on `node:fs`, and the porting codemods. `KARUME_DAWN_FLAGS` passes Dawn toggles.
- Real-weight end-to-end tests move to `packages/e2e/tests/`; the lanes become Vitest projects
  generated from one table (`packages/test-support/src/lanes.ts`) and run with
  `vp test --project <lane>`. `verify_lanes_test` checks coverage, disjointness and non-empty lanes
  against that table instead of parsing `deno.json` task strings.
- The checks that used Deno subcommands are replaced: the public-surface check (`deno doc` →
  the TypeScript 7 API), the dependency-graph check (`deno info` → oxc-parser), the hub barrel
  bundle check (`deno bundle` → the Vite+ build), and the local distribution server
  (`Deno.serve` → `node:http`).
- Type checking is TypeScript 7 through `vp check`, with `@webgpu/types` and `@types/node`.

### For verification (amends ADR 0005 and ADR 0106)

- The real-GPU stage of ADR 0005 runs under Vitest on Node + Dawn. Its rules are unchanged:
  all-SKIP is a FAIL unless the matching opt-out is set (`KARUME_ALLOW_NO_GPU`,
  `KARUME_ALLOW_NO_ASSETS`, `KARUME_ALLOW_NO_DISTRIBUTION`, …), and a release needs a green
  real-GPU run.
- GPU tests run one file at a time in one forked worker (`pool: "forks"`,
  `fileParallelism: false`), because they share one device and the sha256 rows assume one run at a
  time. `--expose-gc` is passed for the memory tests; no per-test timeout is set.
- **Environment keys gain the `node-` runtime.** ADR 0106 decision 2 is amended: on Node the key's
  base is `GPUAdapterInfo.device` (e.g. `node-nvidia-geforce-rtx-3080-ti`), because Dawn's
  `description` is the driver version and the key MUST NOT contain one. Deno and Chrome keep their
  rule. A Node machine therefore starts without reference rows; the rows are written only after the
  Node outputs have been compared with the frozen Deno outputs (below).
- `results.json` records `runtime.name = "node"` and Node's version / platform spellings.
- CI runs on Node (`.github/workflows/ci.yml`): pnpm install with a frozen lockfile and declined
  build scripts, `vp check` (lint + type check), and the core lane with the GPU / asset /
  distribution opt-outs, since CI runners have no GPU, weights or mirrors. Publishing dry-runs
  return once the registry question (Open points) is settled.
- Vitest has no counterpart of Deno's resource / op sanitizers; leaks those caught are no longer
  detected automatically.
- GPU timing (ADR 0021) on Dawn is quantised to 65,536 ns unless the toggle is passed. Measurements
  that need finer steps MUST disable the quantisation explicitly (`KARUME_DAWN_FLAGS`), and records
  MUST say which setting was used.

### Explicitly left open

See Open points. This ADR does not decide the numerics profile, the default weight preset, the
test reorganisation beyond the port, or the language of sources and documents; those are recorded
separately.

## Staging

Each phase ends only when its output matches the previous phase's output (phases 1 and 2) or
differs only where intended (phase 3).

1. **Phase 1 — port, behaviour unchanged.** Work happens in a worktree outside the repository on
   branch `wip/0.15`; `main` stays at 0.14.0.
   - Before: the release run of 0.14.0 (full verify on Deno, 36 lanes, every case passed) is frozen
     as the baseline, with a checksum list.
   - Skeleton: pnpm workspace, catalog, release-age guard, package manifests, `@karume/test-support`,
     `@karume/hub/node`, `AcquireGpuOptions.gpu`.
   - Tests: mechanical port with the codemod (assertions, names and structure kept); a ported file
     passes when its passed / skipped counts match the Deno baseline for the same file.
   - Lanes, Deno-tool checks, process tools, CI.
   - Per lane: compare the Node outputs with the frozen Deno outputs within a band, then write the
     `node-*` reference rows.
   - The switch to Oxfmt is the last commit of the phase.
2. **Phase 2 — cleanup, behaviour unchanged.** Replace the Deno-shaped shim with native Vitest
   forms, choose the Oxlint rule set (the default rules report 331 warnings and 0 errors today; the
   port does not change code to satisfy them), translate comments and documents, reorganise the
   test split.
3. **Phase 3 — behaviour changes and the 0.15.0 release.** The behaviour changes decided elsewhere,
   the release runbook rewritten for the Node toolchain, the registry decision, and the first
   `published-smoke` run on both Node and Deno against 0.15.0.

## Open points

- **Registry for 0.15.0: JSR only, or JSR and npm.** Not decided; it is the first open point of the
  0.15.0 release. Facts that bear on it: the npm names are unused (`@karume/runtime` returns E404);
  Node applications can already install from JSR through its npm compatibility endpoint
  (`published-smoke` does this). Caveat, from Node's documented behaviour rather than a Karume
  measurement: Node does not strip types from files under `node_modules`, so an npm release of the
  `.ts` sources as they are would not load; npm publishing implies a build step (tsdown).
- **JSR publishing configuration.** `packages/hub/deno.json` still lists only `./deno` in its JSR
  `exports`; `./node` must be added before publishing. Which file carries the JSR configuration once
  the Deno workspace files leave the tree (`deno.json` or `jsr.json`) is not decided.
- **When `deno.json` / `deno.lock` leave the development tree**, and what happens to the existing
  `deno-*` reference rows in the fixtures (kept as history or pruned).
- **Whether a Deno import check runs before publishing** (in CI or the runbook), in addition to the
  post-release `published-smoke` run.
- **When Deno support ends altogether**, given the end of Deno runtime development around 2027-10.
- **Default Dawn timestamp setting in the test harness** (quantised default, or the 1,024 ns toggle).
- **Final form of GPU serialisation in Vitest** (the current `forks` + `fileParallelism: false`
  setting works; it has not been reviewed as the long-term setting) and a replacement for the
  Deno sanitizers.
- **Which removed tools and examples come back as browser applications, and in what order**, and
  what replaces the gpu-lab page as their common shell (dev server, shared asset loading).
  ADR 0005's manual real-browser acceptance stage, which used the examples harness, needs a new
  home in the same step.
- **Maintenance of the release-age exclusion list** when Vite+ adds or renames its packages.
