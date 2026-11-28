> A point-in-time measurement record (2026-11-28): what re-runs the verification cache causes on the commits that
> touch `packages/runtime`, the designs measured against them, and the checks of the code layer's syntax-tree identity
> that was built from them. Every number below was computed from the untracked records under `outputs/verify-cache/`
> and the git history; the scripts and their outputs are under `outputs/misc/c3-verify-cache-recon-2026-11-28/`. Nothing
> here is a design source of truth or a decision ([spec/verification.md](../spec/verification.md#a-modules-identity)
> says what is built), and it is not edited to follow later events.

# The verification cache's code layer: by bytes or by syntax tree (2026-11-28)

The backlog item "Verification, second part" proposed that a kernel change re-run only the test files that compile
that kernel. Before building it, the history was replayed to see where the re-run time of commits that touch the
runtime comes from.

## The replay

- **Inputs.** The latest full record of every lane on the development machine (environment `891188c8ce93`,
  2026-11-28): 584 test files, 6,376 s when run one after another. Each record gives the file's import closure, its
  data inputs, its kernels and its duration.
- **Commits.** The last 100 commits that touched `packages/runtime` (`4c80a5c8`, 2026-11-17, to `ff2498c5`). Each is
  compared with its parent only. A file re-runs when the lockfile or `vite.config.ts` changed, when one of its data
  inputs changed, or when a module of its closure changed under the design's rule.
- **Simplification.** The closures are the latest records', not each commit's own.

| Design | File runs | Seconds | Seconds per commit |
| --- | --- | --- | --- |
| A: every module by its bytes (the key until now) | 21,300 | 303,675 | 3,037 |
| A2: comment-only changes ignored | 16,207 | 229,289 | 2,293 |
| B: A2, and a kernel module's change re-runs only the GPU files that dispatch its kernels | 15,900 | 223,647 | 2,236 |
| C: the dispatch programs derived without a device (ceiling, every derivation assumed equal) | — | — | 835 |
| U: every runtime change narrowed perfectly on GPU files (no design does better) | 10,070 | 48,392 | 484 |

- **The 67 commits that touched `packages/runtime/src`.** A 4,141 s, A2 3,182 s, B 3,097 s and C at most 1,006 s per
  commit.
- **Commits that re-ran more than 6,000 s.** 41 under A, 31 under A2, 28 under B.

Where the A2 re-run time comes from (a file run that several changed modules hit is split evenly among them):

| Changed modules | Seconds |
| --- | --- |
| `kernels/geometry-profile.ts` and `kernels/geometry-profiles/*` | 117,543 |
| the other `kernels/*` | 34,983 |
| `runtime/src/runtime` | 19,214 |
| `runtime/src/gpu` | 12,678 |
| `packages/models` | 11,925 |
| `runtime/src/tune` | 11,111 |

## What the measurements say

- **B saves 56 s per commit.** It saves time on three commits only: `a6010247`, `9b7138d2` and `91fa0d10`. A kernel
  module holds more than the WGSL it writes:
  - `91fa0d10` rewrote `linearGemvParallelRowsFits`, a selection predicate in `linear-gemv.ts`.
  - `9b7138d2` made `expert-linear.ts` choose the grouped form only for measured run lengths.

  A change of what a module selects can dispatch a kernel the old record never held. B made no wrong reuse on these
  three commits, but only because the files it reused dispatch no linear or expert kernel. A module-to-kernel map is
  therefore unsound. It would be sound only after about 58 kernel modules were split into WGSL text and selection.
- **C is the real lever for selection changes, but not yet.** It needs the Session's construction recorded and every
  op's dispatch program derived without a device. Most of the recipe builders are to be rewritten into per-op dispatch
  programs by the backlog item on kernel selection; C is measured again with this replay after that.
- **A2 saves 744 s per commit (24 %) and is sound by construction.** Comment and documentation commits
  (`7cbf159d`, `9b34bd96`, `8709179e`, `71c52fae`, `cd1d6416`, `73164a83`, `bda29b7f`) each re-ran 6,264 to 6,346 s
  under A and re-run 4 to 5 s under A2.

The maintainer's ruling of 2026-11-28 built A2 as the code layer's syntax-tree identity, left B unbuilt, and put C
after the per-op dispatch programs.

## The built identity, replayed

- **Rule.** `moduleIdentityOf` (`packages/test-support/src/verify-cache.ts`) is applied to both sides of every changed
  file. Byte identity is kept for modules imported with a query and for every module of the two barrel tests. The
  models barrel test's closure includes the barrel it bundles.
- **Data layer.** A closure module that the test read through `fs` is in its data layer. These pairs were taken from a
  fresh record run of the core lane; no family-lane file reads a closure module.

| Rule | File runs | Seconds | Seconds per commit |
| --- | --- | --- | --- |
| A (byte identity) | 21,300 | 303,675 | 3,037 |
| Built, without the exceptions | 16,209 | 229,290 | 2,293 |
| Built, with the exceptions | 16,265 | 229,300 | 2,293 |

- **What the exceptions cost.** 56 file runs and 10 s over the 100 commits.
- **The 67 runtime-source commits.** 4,141 s under A and 3,182 s under the built rule.
- **Commits that re-ran more than 6,000 s.** 41 under A and 31 under the built rule.
- **Difference from A2.** The built rule and A2 differ by one or two file runs on 5 commits.

### Soundness on the history

The 100 commits changed 727 JavaScript or TypeScript files whose bytes differ between parent and commit. For 117 of
them the identity did not change. For each of the 117, every comment (from the lexer's comment list, not the tree) and
every whitespace character was removed from both texts, and the two texts were compared. All 117 were equal: those
commits changed only comments and whitespace in those files. There is no exception.

A whitespace change inside a literal changes the tree, so a pair equal after stripping and equal in the tree changed
nothing but comments and whitespace between tokens.

### The serialisation

- **What it is.** The tree is the oxc-parser 0.153 program, parsed with `range: false` and `preserveParens: true`,
  written as JSON. The properties `start`, `end`, `range` and `loc` are dropped. A bigint value is written as text
  (`1n`), and a regular expression value as its source (`/a+/g`).
- **Which keys it emits.** Over the repository's 1,589 JavaScript and TypeScript files, the parser emits 2,288,252
  nodes of 132 types. Each node carries exactly one `start` and one `end`; no other positional property appears.
  After the drop, the only property with a numeric value is a numeric literal's `value`.
- **Shift check.** Every one of the 1,589 files gave the same text twice. It also gave the same text when moved down by
  a line comment, a block comment and blank lines inserted at its top (after the first line when that line is a
  hashbang).

## Which tests read sources as text

A record run of the core lane (no GPU) lists, per test, the files it read. Nine tests read a module of their own
closure:

| Test | Closure modules read |
| --- | --- |
| `packages/runtime/tests/verify_language_test.ts` (language gate) | 12 |
| `tools/lang/tests/repository_links_test.ts` (links and `SPEC:` / `DECIDED:` pointers) | 10 |
| `packages/runtime/tests/verify_test_imports_test.ts` | 9 |
| `packages/models/tests/external_fusion_rule_test.ts` | 4 |
| `tools/geometry-profiles/tests/regeneration_test.ts` | 2 |
| `packages/runtime/tests/e2e_public_api_test.ts` | 1 (itself) |
| `packages/test-support/tests/verify_cache_test.ts` | 1 (itself) |
| `tools/geometry-profiles/tests/printer_test.ts` | 5 |
| `tools/geometry-profiles/tests/command_test.ts` | 1 |

The last two start processes, so their records are never reused.

- **Readers outside the closure.** `dependency_graph_test.ts` and `session_gpu_features_test.ts` read sources that are
  not in their closure. Those sources were always in their data layer.
- **Bundlers.** The two barrel tests bundle sources with Rolldown, which reads them natively. No fs read records
  that: the models barrel test's earlier record held ten modules and no data input.
- **Tests that always run.** The public-surface tests and `check_doc_links_test.ts` start processes, so they always
  run.
- **Tests that read no files.** `code_skeleton_test.ts`, `language_gate_test.ts`, `scan_japanese_test.ts` and
  `fixture_notes_test.ts` check texts written in the test itself.

## Fault injections

- **Method.** Ten reuse runs of the core lane (`KARUME_NO_DAWN=1`), each from the same baseline records, with one edit
  each.
- **What ran in every run.** The GPU guard `gpu_gate_test.ts` fails in every run: no adapter, and
  `KARUME_ALLOW_NO_GPU` is unset because an opt-out makes every record unreusable. Every run reused 260 to 263 files.

| Edit | Target | Target | Others re-run by a layer |
| --- | --- | --- | --- |
| A Japanese comment in `tools/lang/scan-japanese.ts` | language gate | re-ran, failed (2 lines with Japanese, allowed 1) | the import check, the link check |
| `// SPEC: docs/spec/no-such-spec.md#nothing` in `tools/lang/check-doc-links.ts` | link check | re-ran, failed (missing file) | the language gate, the import check |
| A comment in `packages/test-support/src/module-graph.ts` | import check | re-ran, passed | the language gate, the link check, both barrel tests |
| A comment in `packages/candidate-rules/src/gelu-mul.ts` | external fusion rule test | re-ran, passed | the language gate, the import check, the link check |
| A comment in `e2e_public_api_test.ts` | itself | re-ran, passed | the cache's guard, the language gate, the import check, the link check |
| A comment in the generated `geometry-profiles/apple-m5.ts` | regeneration check | re-ran, passed | the models barrel test, the dependency graph, the language gate, the import check, the link check (and `printer_test.ts`, never reused) |
| A comment in `verify_cache_test.ts` | itself | re-ran, passed | the language gate, the import check, the link check |
| A comment in `packages/hub/src/cache.ts` | hub barrel test | re-ran, passed | the models barrel test, the dependency graph, the language gate, the import check, the link check |
| `/* @__PURE__ */` removed from `SOFT_TOKEN_BUDGETS` in `packages/models/src/gemma/vision.ts` | models barrel test | re-ran, failed (module-scope code survives tree-shaking) | the dependency graph, the language gate, the import check, the link check |
| A WGSL comment in the `?raw` baseline `attention_state_pv_par.wgsl` | kernel checks | re-ran, passed | `gpu_attention_online_tiled_test.ts`, the language gate |

## A comment-only commit

The same baseline was taken through a scratch commit that added one comment line at the top of
`packages/runtime/src/kernels/linear-gemv.ts`, and the lane was run again in `reuse` mode:

- **Result.** 261 files were reused (their records ran 102.8 s) and 130 ran.
- **The 174 files whose closure holds `linear-gemv.ts`.** 167 were reused. Of the other seven, six start a process.
  The seventh is the models barrel test, which keeps byte identity.
- **What the edit re-ran.**
  - Through the data layer: the four files that read the module (language gate, import check, link check, dependency
    graph).
  - Through the code layer: the models barrel test.
- **Everything else that ran.** Files without a record (every case skipped without a GPU, so no record is written)
  and records that are not reusable.

## Not covered

- **Line numbers and function text.** A test that depends on a line number in a stack trace or on
  `Function.prototype.toString` would not re-run on a comment change. No test of the workspace does either today
  (searched for `.stack` and function `toString` in test files).
- **Family lanes.** The closure-read pairs were observed on the core lane only. The family lanes' files read no
  JavaScript or TypeScript source in the full record, except through processes, whose records are never reused.
