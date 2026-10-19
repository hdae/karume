# 0130: Project language — English in the repository, Japanese in conversation

- Status: accepted (2026-10-18 — the maintainer's rulings at the plan check of phase 2 step 2-9 fill the
  open points; see the addendum)
- Date: 2026-10-16
- Related: [0037](0037-karume-monorepo.md) (monorepo, brand and distribution — the published
  packages whose documentation this ADR covers) / [0075](0075-quant-presentation.md) (quant
  `label` / `description` are English, "the same convention as the distribution README and model
  card" — unchanged, now a special case of decision 1) / [0107](0107-model-input-error.md) (callers
  branch on error types, not on message text — the reason translating messages does not break the
  error contract) / [0005](0005-verification.md) (verification strategy — tests that pin message
  substrings are translated with the messages, decision 6) / [0008](0008-public-api.md) (thin
  public surface — identifier renames stay with the ADR that owns the API, decision 8).
- Amends: the "言語" (language) section of the project `CLAUDE.md`, which today requires English
  only for READMEs and model cards and keeps code comments, docstrings, `docs/`,
  `.claude/ACTIVE_DESIGN.md` and the description part of commit messages in Japanese. No earlier
  ADR decides the repository language as a whole; this ADR is the first record of it.

## Context

### What the repository looks like today

Karume was written with a split language rule. The project `CLAUDE.md` ("言語" section) makes
every README English (repository root, `models/`, `tools/exporter/`, future `packages/*/`) and the
generated model card (`README.md` written by `karume dist`); everything else is Japanese: code
comments and JSDoc, Python docstrings, `docs/` including the ADRs, `.claude/ACTIVE_DESIGN.md`, and
the description part of Conventional Commits. Two exceptions keep Japanese inside English prose:
verbatim quotes of error text that the code actually throws, and Japanese TTS demo inputs.

The size of the Japanese material (counted on `wip/0.15` on 2026-10-16; line-based `rg` counts, so
multi-line messages are under-counted):

| Material                                                                                      | Count                                     |
| --------------------------------------------------------------------------------------------- | ----------------------------------------- |
| TypeScript files under `packages/*/src` containing Japanese                                   | 297 of 303                                |
| `throw new` lines in `packages/{runtime,hub,models}/src`                                      | 1,806 (runtime 863 / hub 33 / models 910) |
| ... of which carry Japanese text on the same line                                             | 811 (runtime 353 / hub 8 / models 450)    |
| Python files under `tools/exporter/src` and `tools/export-recipes` containing Japanese        | 335                                       |
| ADR files / total lines in `docs/decisions/`                                                  | 125 / 24,371                              |
| ADRs fully superseded (0038, 0043, 0071, 0072) / partly superseded (0081, 0090)               | 4 / 2                                     |
| WGSL snapshot fixtures (`packages/runtime/tests/fixtures/wgsl/`) containing Japanese comments | 228 of 258                                |
| `DECIDED:` pointers to ADR numbers in `packages/**/*.ts`                                      | 53                                        |

The test suite pins error messages: `assertThrows` / `assertRejects` are called about 3,200 times in
`*_test.ts`, and many of those calls assert a Japanese substring of the message. Changing a message
therefore fails its test unless both move together.

### Why change now

The 0.15 migration (Deno to Node + Dawn with the Vite+ toolchain) already rewrites every test file
mechanically and adds new packages (`@karume/test-support`, the Node entry of `@karume/hub`). The
commits on `wip/0.15` since `cc22c086` are already written entirely in English. The maintainer
decided on 2026-10-15 to make the whole repository English as part of the same migration, and to
keep Japanese for the conversation with the maintainer.

The plan records the decision itself; the rationale below is stated by this ADR for the maintainer
to confirm:

- The packages are published (JSR, PyPI, Hugging Face). Readers of the source, the error messages
  and the design records are not assumed to read Japanese; today only the READMEs are reachable for
  them.
- A user who hits an error sees a Japanese message from an English-documented library. Error text is
  the most visible prose the library emits.
- Splitting the language by file type forces every contributor (human or agent) to switch language
  between a README and the code it documents, and lets in-house Japanese coinages leak into English
  prose through literal translation.

### What the conversation side needs

The maintainer works in Japanese. Over the project's life a set of in-house Japanese terms grew up
(for example 席 "seat" for a quant preset, 門 "gate" for a test check, 系列 "series" for exporter
output) that do not read as ordinary Japanese and do not map to ordinary English terms.
`docs/GLOSSARY.ja.md` (draft of 2026-10-15, added in `f7c62d9d`) maps each English term used in code
and docs to the Japanese used in conversation, and lists the in-house phrasings that are retired. Its
rule is to choose the Japanese term from the concept's meaning, not by translating the English word.

## Decision

1. **The repository language is English.** Source code, identifiers, code comments and JSDoc,
   Python docstrings, WGSL comments (see Open points), error messages, log and diagnostic text,
   commit messages, ADRs, everything under `docs/`, READMEs, the CHANGELOG and generated model cards
   MUST be written in English. Rationale: everything in the repository is read by people who use
   the published packages; one language removes the per-file switch and makes every artefact
   reachable for them.

2. **Commit messages are all-English Conventional Commits.** The format stays
   `<type>(<scope>): <description>` with a body that explains why; the description and the body are
   English. This replaces the current convention of an English type with a Japanese description, and
   it overrides any personal default that writes the description in Japanese. Past commits are not
   rewritten. Rationale: the history is part of the repository and is read the same way as the code.

3. **Japanese is used in three places only:** (a) the conversation with the maintainer (reviews,
   plans, reports, decision requests); (b) `docs/GLOSSARY.ja.md`, which maps an English term to its
   meaning and to the Japanese used in conversation; (c) material that is Japanese by nature —
   inputs for Japanese TTS and Japanese tokenizer / generation tests, Japanese fixtures, and
   user-facing strings that are Japanese today until they are translated (decision 6); (d) the research
   snapshots under `docs/research/` written before 2026-10-18, which stay Japanese behind an English header
   line — research written after that date is English (ruling of 2026-10-18). Rationale:
   the maintainer's working language is Japanese; translating a Japanese test input would change
   what the test exercises.

4. **Japanese terms are chosen from meaning, not by translation.** A Japanese term for the
   conversation MUST describe what the concept is in ordinary Japanese (a katakana word that is
   understood as is, or plain Japanese), and is recorded in `docs/GLOSSARY.ja.md` with the phrasings
   it replaces. One concept SHOULD have one term. Examples already agreed: "series" (exporter output
   directory) is 中間成果物, not シリーズ; "spike" is 試作; "kill criterion" is 打ち切りの基準; the
   work units are 計画 → フェーズ → ステップ (plan → phase → step). Rationale: literal translation is
   how the retired in-house vocabulary arose.

5. **English terms follow ordinary industry usage; no in-house coinages.** English prose MUST use
   the term the industry uses for the concept (for example "preset" for a named bundle of storage
   dtype and execution options, "check" for one test assertion target, "fallback" for dropping to
   the reference variant). A term that exists only in this project MUST NOT be introduced by
   translating a Japanese coinage; when no industry term exists, the new term is defined in
   `docs/glossary.md` at its first use. Rationale: English readers have no access to the Japanese
   history behind a coinage.

6. **Existing Japanese material is translated in phase 2 of the migration, not during the port.**
   - ADRs are first triaged for force, then translated. For every decision still in force the triage
     asks whether the constraint that produced it still holds: a decision that was provisional under a
     constraint that is gone is re-graded by the maintainer (MUST to SHOULD or MAY, or marked superseded
     by the later fact) before it is translated, and the translation records the new grade with its
     date. "Excessive" is about force, not volume (ruling of 2026-10-18). Fully superseded or withdrawn
     ADRs are excluded from translation and marked as such (an English status line; the number is
     kept); ADRs that are still in force (including the partly superseded ones) are translated, with
     the history-only narrative of their addenda compressed to the decision and its date.
   - Code comments and JSDoc are rewritten, not translated verbatim: each comment is checked against
     the code it describes, and a stale or wrong statement is corrected or removed rather than
     carried into English.
   - Error messages MUST be translated in the same commit as the tests that assert them. A test that
     pins a Japanese substring MUST pin an English substring of comparable specificity afterwards;
     translation MUST NOT be used to weaken an assertion (for example by dropping the message check).
     Rationale: reorganising before translating avoids translating 24,000 lines of which a part is
     no longer in force; a verbatim translation of a stale comment makes the stale claim look current;
     moving a message without its assertion turns the suite red or, worse, invites loosening it.

7. **Transitional rule for phase 1 (the port).** New code, new comments, new documents and commit
   messages written during the port are English. Japanese text that the port does not otherwise touch
   stays as it is until phase 2; ported tests keep their Japanese text, including message
   assertions (as `packages/test-support/README.md` requires for the behaviour-preserving port). A
   file MAY mix English and Japanese during the transition. Rationale: the port and the clean-up are
   separate phases by the maintainer's decision ("do not port and fix at the same time"); translating
   while porting would make a behavioural regression and a wording change indistinguishable in review.

8. **Identifiers are covered by the rule, but renames follow their owning ADR.** New identifiers
   are English and use industry terms (decision 5). Renaming an existing public identifier that
   embodies an in-house coinage is a public API change and is decided by the ADR that owns that API,
   not by this ADR. Rationale: ADR 0008 keeps the public surface deliberate; a language rule is not a
   reason to break callers silently.

## Consequences

### For users

- From 0.15.0, error messages, JSDoc and every document are English. The CHANGELOG lists the
  change of message language. Error types and their meaning do not change; callers that branch on
  types (ADR 0107) are unaffected. Callers that match message text break, as they could already on
  any wording change.
- Quant preset `label` / `description` (ADR 0075) and generated model cards were already English;
  nothing changes there.

### For the repository

- The "言語" section of the project `CLAUDE.md` is rewritten in phase 2: repository English,
  conversation Japanese, the commit convention of decision 2, and the transitional rule removed once
  phase 2 completes. Until then the transitional rule of decision 7 governs. (Rewritten at step 2-9a,
  2026-10-18.)
- `tools/lang/` holds the three checks of phase 2: the Japanese-text scan (progress and the basis of the
  gate), the Markdown link and `DECIDED:` pointer check, and the code-skeleton check that proves a
  comment-translation commit changed no code.
- `docs/glossary.md` becomes an English glossary (term → meaning summary → pointer to the canonical
  source). Its "Japanese to use" column moves to `docs/GLOSSARY.ja.md`, which stays the only Japanese
  document in `docs/`. Meaning lives in the canonical source (ADR or doc); both glossaries remain
  indexes and MUST NOT carry a second definition.
- ADR numbers are referenced from code (`DECIDED:` pointers, 53 in `packages/**/*.ts`) and from
  other ADRs; the reorganisation has to keep those pointers resolvable (see Open points).
- Translating WGSL comments changes the generated WGSL text: the codegen snapshots under
  `packages/runtime/tests/fixtures/wgsl/` are regenerated in the same commit as the comment change.
  The determinism invariant (same codegen key → byte-identical WGSL) holds within a version, not
  across the translation. WGSL comments are blankspace for the compiler, so output artefacts are
  expected not to move; the sha256 regression checks confirm it.
- The Python exporter (`tools/exporter`, `tools/export-recipes`) follows the same rule; its
  docstrings and error messages are translated in phase 2 with the same fact check and the same
  test discipline (`uv run pytest` in both directories).

### For verification

- Phase 2 is behaviour-preserving apart from message text: every lane MUST stay green with the same
  test counts as at the end of phase 1, and the sha256 regression checks MUST NOT move.
- A message translation is reviewed as a pair (message, assertion); a diff that changes one without
  the other is a defect.
- Whether the rule is enforced by an automated check after phase 2 is open (below).

### Explicitly left open

See "Open points".

## Staging

| Phase                        | Language work                                                                                                                                                         | Done when                                                                 |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Phase 1 — port (in progress) | Decision 7: new material English, untouched Japanese stays, ported tests keep Japanese text                                                                           | Every lane runs on Node with the Deno baseline counts                     |
| Phase 2 — step 2-9a (2026-10-18) | Rulings recorded here; `CLAUDE.md` language section rewritten; the three checks in `tools/lang/` | Baseline taken: 170,828 lines with Japanese text in tracked files |
| 2-9b (done 2026-10-18) | ADRs: force triage over every ADR → the maintainer's rulings → translation in numeric batches of five, each with an independent review | Every ADR in force is English; excluded ADRs carry an English status line; links and `DECIDED:` pointers resolve |
| 2-9c (done 2026-10-18) | `docs/*.md` and `docs/glossary.md` (English; the Japanese column moves to `GLOSSARY.ja.md`); `CLAUDE.md`, `AGENTS.md`, `.claude/ACTIVE_DESIGN.md` as a whole; CHANGELOG remnants; the English header line on research snapshots | No Japanese under `docs/` outside `GLOSSARY.ja.md` and `research/` |
| 2-9d (done 2026-10-18) | Comments and JSDoc of `packages/*/src` rewritten with a fact check; WGSL comments translated and the snapshots regenerated in the same commit | Code skeleton unchanged per file; lanes green; sha256 rows unchanged |
| 2-9e (done 2026-10-19) | Error messages with their assertions, module by module; verbatim quotes in docs in the same commit | All lanes green with unchanged test counts |
| 2-9f (done 2026-10-19) | Python exporter (`tools/exporter`, sources and tests) | AST unchanged apart from docstrings; `uv run pytest` green |
| 2-9g (done 2026-10-19) | Remaining tools and the hand-written fixture notes | The scan shows no Japanese there apart from the allow-list |
| 2-9h | Language gate (`verify_language_test.ts`, core lane) with an allow-list of the files that keep Japanese by nature or by this staging | Gate green; allow-list recorded here |
| 0.15.x (after the release) | Comments in `packages/*/tests` (about 37,000 lines), `tools/export-recipes` (about 30,000 lines) and, with the recipes, the notes of the fixtures the recipes generate (710 strings in 14 files) | Allow-list entries removed as each area is translated |
| Phase 3 — behaviour change | None from this ADR | — |

## Open points

All resolved by the maintainer's rulings of 2026-10-18 (step 2-9 plan check):

- **WGSL comments — resolved.** Translated; the snapshots are regenerated in the same commit (2-9d).
  249 of 279 fixtures carry Japanese comments on 2026-10-18.
- **Order of ADR translation — resolved.** The force triage runs over every ADR first; translation
  then goes in numeric order in batches of five, so an interrupted run resumes at a batch boundary.
- **Reviewer of translations — resolved.** One independent review leg per batch (medium effort; high
  for records about numerics or GPU code) with a closed checklist — no decision lost, status line
  present, links and `DECIDED:` pointers resolve, terms per the glossaries — and the maintainer's
  session samples one ADR per batch plus every section a `DECIDED:` pointer names.
- **Criteria and form of the ADR reorganisation — resolved.** "Excessive" means a decision whose
  force outlived the constraint that produced it (decision 6; re-graded by the maintainer), not the
  volume of text. Excluded ADRs stay in `docs/decisions/` untranslated with an English status line;
  numbers are kept.
- **Snapshot documents — resolved.** Kept Japanese behind an English header line (decision 3 (d)).
- **Agent instruction files — resolved.** Rewritten in English as a whole (2-9c).
- **Enforcement — resolved.** A test in the core lane (2-9h) fails on Japanese text outside an
  allow-list; the allow-list is recorded in this ADR.
- **Japanese user-facing strings — resolved.** The error messages and the three `console` messages of
  `packages/*/src` (2026-10-18 scan); translated at 2-9e.
- **Exact wording of the new `CLAUDE.md` language section — resolved.** Written at step 2-9a; phase 2
  of the language work started on 2026-10-18.

## Addendum (2026-10-18) — rulings at the plan check of step 2-9

- **Scope for 0.15.0.** The material that readers of the published packages and of the design record
  see goes first: ADRs in force, `docs/`, the sources of `packages/*/src` (comments, JSDoc, WGSL
  comments, messages), the Python exporter, the tools and the agent instruction files — about 66,000
  of the 170,828 lines with Japanese text counted on 2026-10-18. Comments in `packages/*/tests`,
  `tools/export-recipes` and the research snapshots follow in 0.15.x (research stays Japanese by
  decision 3 (d)); until then the gate's allow-list names them.
- **Unpushed history.** The 27 commits on `wip/0.15` whose subject is Japanese (2026-10-17 and 18)
  are not reworded: decision 2 does not rewrite history, and the documents cite their hashes.
- **Residue after 2-9c (2026-10-18).** Japanese that stays under `docs/` outside `GLOSSARY.ja.md` and
  `research/`: anchors into Japanese research headings (perf-ledger 23 lines, backlog 3, ACTIVE_DESIGN 2);
  verbatim quotes of error messages, exporter labels and script output that the code still prints
  (limitations 4, known-issues 2, op-vocabulary 2, release-runbook 1); test-name filters inside recorded
  commands (known-issues 3, release-runbook 1); one quoted commit subject (perf-ledger). The quotes and
  filters go when 2-9d / 2-9e translate their sources; the anchors stay on the allow-list with research.
  The `*-results.json` files under research (10) carry Japanese labels and stay with research.
- **Residue after 2-9d (2026-10-18).** The comments, JSDoc and WGSL comments of `packages/*/src` and
  of the public entry points are English; the code skeleton of every file is unchanged
  (`tools/lang/code-skeleton.ts`, which now carries a WGSL comment across `${…}`), the 249 WGSL
  fixtures were regenerated with `KARUME_WGSL_SNAPSHOT=write` (comment lines only), and the kernel
  fingerprint of the two embedded geometry profiles was re-derived because it hashed the generated
  WGSL with its comments (ADR 0117 decision 4; since the addendum of 2026-10-19 it hashes the WGSL
  without comments, so comment edits no longer move it). What stays Japanese there is string literals —
  error messages, diagnostics, WGSL header fragments such as `gemmGeometryNote`, test-name labels
  — about 3,150 lines, the subject of 2-9e; a few comments keep a Japanese example in a code span.
  Three test assertions that matched Japanese comment text (`codegen_wgsl_test.ts`,
  `linear_gemv_rows_test.ts`) were re-keyed to the English text.
- **Residue after 2-9e (2026-10-19).** The string literals that reach a human — error messages,
  warnings, diagnostics, labels, the notes written into WGSL headers — of `packages/*/src` and
  `packages/test-support/src` are English: 3,103 literals in 244 files over three iterations
  (runtime core; kernels / tune / gpu / codegen / reference / hub; models). The proof per file is the
  strings mode of `tools/lang/code-skeleton.ts` (`--strings`): the skeleton with every literal lifted
  out is unchanged, and every changed literal held Japanese before, except full-width parentheses
  turned ASCII and the three object keys of `assertConvFlatExtents` (Japanese identifiers that
  reached the message) made quoted English keys. The notes interpolated into WGSL headers changed
  171 snapshots (comment lines only; the kernel fingerprint does not count comments). The assertions
  of 180 test files were re-keyed to the same fragments of the English messages at the same strength,
  one to the interpolated binding limit (`estimate_test.ts`), and the verbatim quotes in
  `docs/known-issues.md` and `docs/limitations.md` carry the English messages. What stays Japanese
  under `packages/*/src` is data by nature — 10 lines in 4 files: the Wan default negative prompt
  (`wan/text-stage.ts`, Chinese, upstream's), the irodori normalisation tables
  (`irodori/text/normalize.ts`), and tokenizer / detokenizer examples in comments
  (`sbv2/text/bert-text.ts`, `text/detokenizer.ts`) — the allow-list of 2-9h. The glyphs `·` and
  `〜` (the row-range spelling of `tune/derive.ts`, also in the embedded tables' comments) are not
  Japanese to the scanner. Test comments that quote old messages verbatim (`*_pipeline_test.ts`
  headers) are left for the test-comment step. The Python exporter (2-9f) and `tools/*` (2-9g) still
  print Japanese.
- **Residue after 2-9f (2026-10-19).** The Python exporter (`tools/exporter`: 31 modules, 39 test
  files, `pyproject.toml`) is English — docstrings, comments, messages, CLI help and the tests'
  own labels. The proof per file is `tools/lang/py_skeleton.py`: the AST without docstrings and
  string constants is unchanged (1,221 constants changed in the sources, 130 in the tests; the only
  constants without Japanese before are full-width parentheses turned ASCII in `dist.py`'s help).
  The assertions of the exporter's tests and of the recipes' tests that matched the old messages
  (`match=` is a regular expression — the English fragments are regex-safe) are re-keyed at the
  same strength; both suites keep their counts (exporter 3,644 passed / 1 skipped, recipes 4,483
  passed / 196 skipped). Facts the originals carried and the worktree contradicted were corrected
  on the way (pointers into superseded ADRs now also name the current decision; the retired I4
  dialect note; `computeOutputShape` lives in `ops/shapes.ts`). What stays Japanese under
  `tools/exporter` is three fixture payloads in the tests (a LICENSE.md body, a stray file, an
  interrupted staging file) — the allow-list of 2-9h. `tools/release/hf-upload.zsh` still prints
  `### FAILED 公開前の門` (quoted by `docs/release-runbook.md`); it belongs to 2-9g, as does the rest
  of `tools/*`. `tools/export-recipes` still prints Japanese (0.15.x). The `RUF001-003` ignore in
  `tools/exporter/pyproject.toml` (full-width punctuation in Japanese prose) loses its reason once
  2-9g ends; 2-9h decides whether to drop it.
- **Residue after 2-9g (2026-10-19).** The remaining tools (`tools/lint`, `tools/published-smoke`,
  `tools/verify-diff`, `tools/release`, `tools/diag`, `tools/llm-baseline`, the comments of
  `tools/.gitignore` and `tools/pyproject.toml`) and the hand-written fixture notes
  (`op-contracts.json`, `dim-grammar.json`, `quant-presentation.json`, `gemma4-qat-rope-golden.json`,
  the comment of `fixtures/golden/.gitignore`) are English. Proofs: the TypeScript and Python
  skeletons for the tools, `zsh -n` plus the diff for the shell scripts, and
  `tools/lang/fixture-notes.ts` for the fixtures (only the strings under `_doc` / `why` / `note`
  changed; keys, order, numbers and data strings identical). The tools' own tests are re-keyed at
  the same strength. The notes of the fixtures that `tools/export-recipes` generates (710 strings in
  14 files — `wan-text/parity.json`, `umt5/tokenizer-cases.json`, `birefnet-matte/matte.json`,
  the gemma, anima and irodori parity files, …) are written by the recipes' scripts, so they move
  with the recipes to 0.15.x (ruling of 2026-10-19). The allow-list of 2-9h, as it stands after
  2-9g: the input data of `tools/lang`'s own tests and the `JAPANESE` character classes; the three
  fixture payloads of `tools/exporter/tests`; the 10 data lines of `packages/*/src`; the data
  strings of the fixtures (vocabularies, sample texts, expected values); the generated fixtures'
  notes (until 0.15.x); `tools/export-recipes`; `packages/*/tests`; `docs/research` and
  `docs/GLOSSARY.ja.md`; the excluded ADRs.
- **Baseline (tracked files, line-based, 2026-10-18).** `docs/decisions` 19,541 lines in 129 files;
  `docs/` otherwise 5,651; `docs/research` 23,964; `packages/runtime/src` 16,512; `packages/models/src`
  11,911; `packages/hub/src` 1,433; tests of the three packages and e2e 37,113; test fixtures 11,964;
  `tools/exporter` 6,680 and its tests 3,806; `tools/export-recipes` 30,414; the other tools 703.
