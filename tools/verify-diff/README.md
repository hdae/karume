# verify-diff — compare verify results across machines

A read-only CLI that lines up the verify results recorded on several machines and points at the cases
where they disagree. It reads `outputs/verify/<environment key>/<date>_<family>/results.json` (the
result seats of ADR 0106) and writes nothing.

Results from a second machine are copied **by hand** into `outputs/verify/<environment key>/` (the
results stay untracked, so there is no sync mechanism). If they are collected somewhere else, point
`--root` at that directory.

## Usage

Run from the repository root (Node 24 runs the TypeScript sources directly):

```
node tools/verify-diff/main.ts [options]
```

The package script `verify-diff` runs the same entry point: `pnpm -s --filter @karume/verify-diff verify-diff [options]`
(no `--` separator — pnpm would pass it on as an argument). It runs in `tools/verify-diff/`, so a
relative `--root` resolves against that directory.

Examples:

```
node tools/verify-diff/main.ts
node tools/verify-diff/main.ts --family anima --family golden
node tools/verify-diff/main.ts --date 2026-09-21
node tools/verify-diff/main.ts --root ~/verify-from-another-box --json
```

| Option              | Default                              | Meaning                                                    |
| ------------------- | ------------------------------------ | ---------------------------------------------------------- |
| `--root <dir>`      | this repository's `outputs/verify/`  | Root directory of the result seats                         |
| `--family <name>`   | every family found under the root    | Only this family (repeatable)                              |
| `--date YYYY-MM-DD` | the latest date per environment      | Only the seats of this date (**UTC**)                      |
| `--json` (switch)   | off (Markdown)                       | Print the matrix, differences and warnings as JSON instead |

Options take the form `--key value` (no `--key=value`). Unknown options, a missing value, and a
repeated `--root` / `--date` fail loudly.

For every environment the tool takes **one** seat per family: the latest date, or the `--date` one.
Seat dates are UTC (the date part of the run's start time), so `--date` is a UTC date as well.

## Output

Per family: the selected seat of each environment (date and checkout), a status matrix
(case × environment), warnings, the differences (status mismatch, differing output sha256, cases
present on only some environments), and per-case tables of the recorded tolerance measurements and
self A/B comparisons.

## Exit code

- `0` — the results were read, **even when they differ**. This is a pointer, not a gate: bit-identical
  output across devices is not guaranteed (`docs/limitations.md`), so a differing sha256 is listed as a
  difference only.
- `1` — a usage error, a missing root, a seat directory without `results.json`, a document that is not
  JSON, or a document that does not match the results schema. The tool never prints "no differences"
  for results it could not read.

## Tests

```
vp test tools/verify-diff/tests
```

`tests/diff_test.ts` covers the pure diff logic and the argument parser; `tests/main_test.ts` runs the
CLI as a child process (`node main.ts`) against seats synthesised in a temporary directory. Neither
touches the GPU or `outputs/`.
