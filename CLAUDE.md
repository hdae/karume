# Karume — project conventions (one screen)

Monorepo of a general-purpose WebGPU NN inference stack (development on Node 24 + Dawn; the primary target is the browser;
Deno is secondary (ADR 0127); pure TS + WGSL; zero runtime dependencies). JSR `@karume/runtime` (IR execution) /
`@karume/hub` (HF revision resolve, download, cache, model / quant resolution) / `@karume/models` (pipelines, tokenizers) +
PyPI `karume` (tools/exporter — torch.export → container `krm` (IR v2), managed by uv). pnpm workspace + Vite+ (`vp`). The
design's source of truth is [docs/decisions/](docs/decisions/). The current focus and pitfalls are in
[.claude/ACTIVE_DESIGN.md](.claude/ACTIVE_DESIGN.md) (required reading for review and planning tasks).

## Terms

- "The prototype" = the earlier experimental project used as a reference. **Do not write its proper name in any artifact
  (code, comments and commit messages included)**.
- The Japanese terms of the conversation are in [docs/GLOSSARY.ja.md](docs/GLOSSARY.ja.md) (chosen from meaning, not by
  literal translation).

## Language (ADR 0130)

- **The repository is English**: code, identifiers, comments and JSDoc, Python docstrings, WGSL comments,
  error messages and diagnostics, commit messages, ADRs, everything under `docs/`, READMEs, the CHANGELOG
  and generated model cards. **Commit messages are all-English Conventional Commits**
  (`<type>(<scope>): <description>`, body explains why) — this overrides any personal default that writes
  the description in Japanese. Do not add new Japanese text anywhere else.
- **Japanese appears in three places only**: the conversation with the maintainer; [docs/GLOSSARY.ja.md](docs/GLOSSARY.ja.md)
  (English term → the Japanese used in conversation — read it first when the conversation is in Japanese and
  use its terms); and material that is Japanese by nature (inputs of the Japanese TTS, tokenizer and generation
  tests, and their fixtures). **The language gate** (`packages/runtime/tests/verify_language_test.ts`, core
  lane) fails on Japanese outside the allow-list `tools/lang/language-allow-list.ts` — one entry per file or
  area with its reason; the areas still to translate (`packages/*/tests`, `tools/export-recipes`) are marked
  `until: "0.15.x"`. New Japanese needs an entry with a reason; a translated area loses its entry (a stale
  entry is red too). From the shell: `node tools/lang/language-gate.ts`; counts: `node tools/lang/scan-japanese.ts`.
- Terms: English follows ordinary industry usage (no in-house coinages); the Japanese terms of the
  conversation are chosen from meaning, not by literal translation (ADR 0130 decisions 4 and 5).
- Checks: `node tools/lang/check-doc-links.ts` (Markdown links and `DECIDED:` pointers resolve) and
  `node tools/lang/code-skeleton.ts --compare <ref>` (a comment-translation commit changed no code).

## Layout

- `packages/runtime/` — the public API is **only the thin surface** of mod.ts (ADR 0008). `src/` is the body
  (gpu / codegen / kernels / format / ops / reference / runtime / tune). When it is used from Node, pass a WebGPU
  implementation (Dawn) to `acquireGpu({ gpu })`
- `packages/hub/` — manifest resolution, fetch, cache (the spec's source of truth is ADR 0038 §4 + 0041 +
  0075 (display names of quants) + 0109 (manifest is `karume/5`) + 0080 (fetch layer) + 0086 (source abstraction)).
  The only runtime-specific subpaths are `hub/node` (`nodeDirectory`) and `hub/deno` (`denoDirectory`)
- `packages/models/` — **both a barrel (mod.ts) and per-family subpath exports**
- `packages/e2e/` (private) — real-weight e2e (local assets + GPU). Run per-family lanes before a release (ADR 0129)
- `packages/test-support/` (private) — the test harness (Dawn injection, fs helpers, the lane table, the port codemod).
  The porting rules are in the README of the same directory
- `tools/exporter/` — PyPI `karume` = **the general-purpose core only** (src layout; the boundary is a machine gate —
  ADR 0065) / `tools/export-recipes/` — per-model recipes (outside the wheel; shared venv of the uv workspace;
  launched with `python -m <family>.<mod>`; dist is `export-recipes/dist.py`).
  `tools/{lang,lint,published-smoke,verify-diff,release}` are process tools (workspace packages).
  The measurement tools and CLI examples (`tools/*` / `examples/*` of 0.14.0) have been removed and are rebuilt as
  browser apps (ADR 0127)

## Verification commands (all of them after a change)

- `vp check` (= fmt (Oxfmt) + lint (Oxlint, type-aware) + type check (TypeScript 7)) and `vp test`
  (every lane = full verify). GPU tests run on a real GPU (Dawn); an environment without an adapter SKIPs explicitly
  (a release requires green — ADR 0005)
- **Lane runs**: when a change is confined to one family, run only that family (`vp test --project models:<family>`).
  Larger changes to the core (runtime / hub / shared layers), cross-cutting changes and pre-release runs use the full
  `vp test`. Core only: `vp test --project core` (includes the three all-SKIP guards). The lane table is
  `packages/test-support/src/lanes.ts`; `verify_lanes_test.ts` fails on coverage gaps (ADR 0129)
- **Upstream parity** (cases that compare with torch / diffusers goldens within a tolerance) is opt-in:
  `KARUME_UPSTREAM_PARITY=1`. The default lanes SKIP them explicitly. Parity is established at implementation time, and
  later changes are caught by the sha256 regression checks (ruling 2026-10-17; the flag is in
  `packages/e2e/tests/helpers/upstream-parity.ts`)
- **Cases that use the real network** (the one `node_cache_storage_test.ts` that fetches small assets from
  huggingface.co) are also opt-in: `KARUME_NETWORK=1`. The default lanes SKIP them explicitly
- **Families not run by default** (minicpm5 / siglip2 — maintained but not run every time; `LOW_PRIORITY_FAMILIES` in the
  lane table): `KARUME_ALL_FAMILIES=1 vp test --project models:siglip2`. Full verify and the release decision run the
  default set
- One file: `vp test <path>`. Runs that use the GPU are serialised with other sessions through
  `flock /tmp/karume-gpu.lock`. To run without the GPU, set `KARUME_NO_DAWN=1` (the same configuration as CI — the
  all-SKIP guards are opened explicitly with `KARUME_ALLOW_NO_GPU=1` and the like)
- **sha256 reference values are rows per environment** (ADR 0106): on Node the environment key is
  `node-<adapter device>` (e.g. `node-nvidia-geforce-rtx-3080-ti`). On a machine without a row the case SKIPs
  explicitly + the reference guard is red. `KARUME_REFERENCE=write vp test --project <lane>` creates your machine's rows
  (existing rows are not touched; re-recording is `rewrite`). Results and artifacts go to
  `outputs/verify/<environment key>/<date>_<series>/` (safe to delete).
  **Turning the sha256 regression check into a tolerance is forbidden** (once it is loosened, the check loses its meaning as the
  port's check)
- exporter: `uv run pytest` (**in both tools/exporter and tools/export-recipes** — ADR 0065)
- On a machine without the distribution mirrors under `models/`, `KARUME_ALLOW_NO_DISTRIBUTION=1` turns the guard off
  (the real-asset e2e rooted at the distributions and the fusion hit-count checks do not run)
- Create git worktrees **outside the repository** (e.g. `~/workspace/karume-wt-<name>`) — inside the repository,
  lint / test pick up the worktree's files by mistake (for the same reason, the app-created `.claude/worktrees/` is not used)

## Cross-cutting invariants

- **Zero side effects in every module** (no top-level registration, no global mutable state, no execution at import) —
  MUST, the condition for tree-shaking through the barrel (check: `hub_barrel_surface_test`)
- WebGPU globals (`GPUBufferUsage` and so on) are not read at module scope — the runtime goes through
  `src/gpu/webgpu-constants.ts`; bare references are forbidden — MUST (import safety. Node has none of these globals;
  only the test harness installs a copy from Dawn)
- Codegen determinism: same key → byte-identical WGSL (pinned by snapshots)
- Elementwise / row reductions assume grid-stride. requiredLimits are stated explicitly down to the compute limits +
  `pushErrorScope('validation')` is always in place
- Unsupported or unexpected input fails loudly (no silent approximation). Symbol binding uses `Object.hasOwn`
- Flush before destroy. The buffer pool invariants are ADR 0004
- Runtime dependencies are Web standard APIs only (dependency packages are limited to ones built only on Web standard
  APIs — e.g. `@hdae/fetch-cache`. ADR 0002; the check is `dependency_graph_test`; only `hub/node.ts` may import
  `node:`). Relative imports carry `.ts`. WGSL is written in template
  literals. TypedArrays state `<ArrayBuffer>` explicitly (lint rule `karume/typed-array-buffer` —
  `tools/lint/oxlint-plugin.ts`; exceptions are in the exception table of ADR 0002)
- Upstream parity is reached at the level of the numbers (or their intent), and upstream bugs are not copied. When the
  upstream has a problem, provide a workaround as an option (the default if it is better). When parity is impractical,
  show correctness another way (ADR 0005 addendum 2026-10-11)

## Where docs go

- `docs/decisions/` — ADRs (MADR-lite). The target of inline `DECIDED:`
- `docs/backlog.md` — **the source of truth for the order of phases, work items and their status** (now / next / later / release / parked)
- `docs/ir-v2.md` — IR format spec / `docs/op-vocabulary.md` — op vocabulary ledger
- `docs/glossary.md` — term → meaning → source (the meaning lives at the source; add a row when a new term enters docs) /
  `docs/GLOSSARY.ja.md` — the Japanese terms of the conversation
- `docs/quantization.md` — index of quantization methods (storage dtypes / execution knobs / quant presets / exporter-side
  methods and their status. Numbers and semantics are not duplicated; it points to the source of truth)
- `docs/assets-layout.md` — conventions for the three local asset roots (models / outputs / inputs — all outside git)
- `docs/release-runbook.md` — the standing procedure for releases and HF publishing (writing the pins, countermeasures against fragmentation, publish order)
- `docs/research/` — investigation and measurement records (**point-in-time snapshots** — one opening line states their nature)
- `docs/perf-ledger.md` — ledger of performance candidates: filing, adoption decisions, kill criteria (the numbers'
  source of truth is research; the overall order of phases is the backlog)
- `docs/known-issues.md` / `docs/limitations.md` — open bugs / by-design constraints
