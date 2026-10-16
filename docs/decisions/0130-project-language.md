# 0130: Project language — English in the repository, Japanese in conversation

- Status: proposed (2026-10-16; to be accepted by the maintainer after review)
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
   user-facing strings that are Japanese today until they are translated (decision 6). Rationale:
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
   - ADRs are first reorganised, then translated. Fully superseded or withdrawn ADRs, and content
     judged excessive in the reorganisation, are excluded from translation and marked as such;
     ADRs that are still in force (including the partly superseded ones) are translated.
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
  phase 2 completes. Until then the transitional rule of decision 7 governs.
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
| Phase 2 — clean-up, step 2-1 | Reorganise ADRs (exclude and mark superseded / withdrawn / excessive), then translate the rest; `docs/glossary.md` to English; `CLAUDE.md` language section rewritten | Every ADR in force is English; excluded ADRs carry an English status line |
| Phase 2 — clean-up, step 2-2 | Rewrite comments and JSDoc with a fact check; translate error messages together with their assertions; translate the remaining `docs/`                                | No Japanese outside the places of decision 3                              |
| Phase 3 — behaviour change   | None from this ADR                                                                                                                                                    | —                                                                         |

## Open points

- **WGSL comments.** They follow the same rule unless the maintainer objects. The cost is the
  snapshot regeneration described above (228 of 258 fixtures carry Japanese comments today).
- **Order of ADR translation.** Not decided (for example: ADRs referenced by `DECIDED:` pointers
  first, or numeric order, or by package).
- **Reviewer of translations.** Not decided who reviews translated ADRs, comments and messages, and
  whether an independent review leg is required per batch.
- **Criteria and form of the ADR reorganisation.** What counts as "excessive" content, whether
  excluded ADRs stay in `docs/decisions/` untranslated with an English status line or are removed
  (git keeps them), and whether ADR numbers are kept. Keeping numbers is the low-risk default because
  of the `DECIDED:` pointers.
- **Snapshot documents.** Whether time-stamped research records in `docs/research/` (171 entries)
  are translated or kept as Japanese snapshots with an English header.
- **Agent instruction files.** Whether `CLAUDE.md`, `AGENTS.md` and `.claude/ACTIVE_DESIGN.md` are
  rewritten in English as a whole or only in their language section; decision 1 reads as "as a
  whole", but only the language section is named in the plan.
- **Enforcement.** Whether a lint check (Japanese characters outside an allow-list of paths such as
  `docs/GLOSSARY.ja.md` and Japanese test inputs) is added once phase 2 completes.
- **Japanese user-facing strings.** The list of strings that remain Japanese until translated
  (decision 3c) is not yet enumerated.
- **Exact wording of the new `CLAUDE.md` language section** and the date phase 2 starts.
