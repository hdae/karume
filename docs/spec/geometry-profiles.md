# Geometry profile tables

How the per-adapter geometry profile tables under `packages/runtime/src/kernels/geometry-profiles/` are made now. A
table is a static answer for one adapter: which tile geometry each op family uses. The runtime selects a table by the
adapter's match and measures nothing at run time.

## Where a table comes from

- A table changes only through the generator, `node tools/geometry-profiles/generate.ts`, run on records of the
  workbench's `geometry-sweep` entry (and, for the online attention's key axis, of its `attention-bench` entry —
  [workbench.md](workbench.md#the-entries)). The table files are not edited by hand.
- The generator derives with `deriveGeometryProfile` of `@karume/runtime/tune`. Whether a record is usable — its
  format version, its adapter, its case sets, a sweep that was cut short — is that function's decision, and its
  refusal is the generator's error.
- The generator always works as **the registered table as base plus new records**: the op families the given records
  cover are derived from them, and every other family is carried over from the registered table unchanged. With a
  base, a covered family is derived from every case of its current case set, so a sweep of only some of a family's
  ops is refused.
- A new adapter's first table has no base and needs its adapter match on the command line.
- A whole table is never re-derived from all the records it ever came from, and a record of an older format is never
  rewritten to pass. The records cited by a family of an older format stay cited until that family is swept again.

## A record

- A workbench record (`karume-workbench/1`), unwrapped to its `data`, or a bare runtime record
  (`karume-geometry-sweep/*`, `karume-attention-bench/*`) written before the workbench existed. A workbench record of
  another entry and a workbench record without `data` are errors that name the file.
- A record path lies inside the repository. The table's provenance cites it repository-relative, with the sha256 of
  the file's bytes as stored (for a workbench record, of the whole file).
- The records live outside git (`outputs/`).

## The file

- From its `import type` line on, a table file is the profile object as a JavaScript literal passed through the
  repository's formatter. An unchanged profile gives the same bytes.
- The header of a generated table says: the adapter match; the op families derived in this generation and, per record,
  its path, sha256 and date and how it was made (browser or Node, the backend and its source, whether timestamps were
  quantised, the checkout); the families carried over and the last commit that touched the table file, where their
  adoption lines are; the adoption rule with the threshold used; the verdict lines of the derivation. It holds no
  date of generation and no absolute path: the same inputs give the same bytes.
- The headers of the three tables registered before the generator existed (`apple-metal-3`, `nvidia-ampere`,
  `nvidia-blackwell`) are older prose. Each is replaced when its table is regenerated.

## The GEMV choices of linear

- A profile can also hold the GEMV choices of linear (`linearGemv`): the variant of the sequential kernel and its
  per-shape exceptions, the thread target that decides the row-block height, and the rows of the shapes that take the
  parallel GEMV with their lane count and their packed and staged flags.
- The generator does not derive them and no registered table carries them. A table without them runs with the default
  profile's, the values measured on the RTX 3080 Ti: the Session fills them in once, where it selects its profile, so
  the linear selection, the fusion pass and the recipe builder all read the same choices. The memory estimate, which
  has no adapter, reads the default profile's.
- `SessionOptions.linearGemvRowsThreadTarget`, when given, replaces the profile's thread target.
- The printer writes the field of a table that carries it, and the saved-table reader (`parseGeometryProfileJson`)
  reads it.

## The command

`node tools/geometry-profiles/generate.ts <profile id> --records=<path>[,…] [--attention-bench=<path>[,…]]
[--vendor=<v> [--architecture=<a> [--description=<d>]]] [--min-speedup=<x>] [--dry-run]`

- A registered id uses its table as base and its table's match; giving a match is an error. An id that is not
  registered needs `--vendor`.
- `--dry-run` writes nothing and prints the derived profile, the verdict lines and every field that differs from the
  registered table.
- A new table is written but not registered: the command prints the two edits registration needs (the array of
  `BUILTIN_GEOMETRY_PROFILES` and the list `geometry_profile_test.ts` compares it with).
- Exit code 0 on success, 2 for a command line it cannot act on, 1 for anything else.

## What is checked

- Every registered table's text from the `import type` line on equals what the generator writes for that profile
  (core lane). A hand edit of a table's object fails this.
- On a machine that has the records: for `nvidia-ampere` and `nvidia-blackwell`, the records their `i8a8-linear`
  family cites, with the registered table as base, derive an object equal to the registered table, provenance
  included. Without the records the case is skipped and says which files are missing.
- `apple-metal-3` has no such case: every record it cites has a format the derivation no longer accepts.
- The default profile's GEMV choices equal, row for row, the values the GEMV kernel module held before they moved into
  the profile, and a profile with other choices changes the selection and the pipeline key (core lane).
