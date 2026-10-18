# @karume/lang-tools

Tools for the repository language rule ([ADR 0130](../../docs/decisions/0130-project-language.md):
English in the repository, Japanese in the conversation). All three run with Node 24 directly
(`node <file>.ts`) from the repository root.

| Command | What it checks |
| --- | --- |
| `node tools/lang/scan-japanese.ts [--list] [prefix...]` | Lines that carry Japanese text, per area (tracked files only, binaries skipped). `--list` prints every file with its count. |
| `node tools/lang/check-doc-links.ts` | Relative Markdown links in tracked `*.md` files and `DECIDED:` pointers in tracked `*.ts` files resolve to an existing file and, with a `#fragment`, to a heading with that slug. Exit 1 when anything is broken. |
| `node tools/lang/code-skeleton.ts --compare <git-ref> [--head <git-ref>] [file...]` | The changed TypeScript files differ from `<git-ref>` only in comments and whitespace — language comments and the `//` / `/* */` comments inside template literals (where the WGSL lives). Exit 1 when a file's code changed. Without `--compare`, prints the skeleton of the given file. |

The scan is the progress measure of phase 2 and the basis of the language gate; the link check
catches renamed files and translated headings; the skeleton check is the proof that a
comment-translation commit changed no code (ADR 0130 decision 6).
