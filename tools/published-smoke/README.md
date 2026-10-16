# published-smoke — smoke test against the published JSR packages

Runs the post-release check from `docs/release-runbook.md` §5 against the
**published** `@karume/*` packages: every entry of `KARUME_SOURCES` (the published
repository table exported by the `@karume/models` barrel — the union of every
family's `*_SOURCES`) must resolve its manifest through the published
`@karume/hub`, and one family (sbv2) is constructed with `fromPretrained` end to
end.

The same `main.ts` runs on Node (primary) and on Deno (secondary — the Deno run
checks that the published packages remain importable from Deno).

```sh
# Node (from the repository root)
node tools/published-smoke/main.ts                   # manifests for every entry + sbv2 fromPretrained (needs a GPU)
node tools/published-smoke/main.ts --manifests-only  # no GPU: manifest resolution only

# Deno (from tools/published-smoke)
deno run -A main.ts
deno run -A main.ts --manifests-only
```

The same commands are available as package scripts in this directory:
`pnpm smoke [--manifests-only]` and `pnpm smoke:deno [--manifests-only]`.

The version under test is read from `packages/models/package.json` (the three
packages are released in lockstep). The run fails if `@karume/hub` or
`@karume/models` resolves to any other version. On Node, the GPU path installs
Dawn (npm `webgpu`) as `navigator.gpu`, which the published `fromPretrained`
reads.

## Where the published packages come from

The packages are imported by bare specifier (`@karume/hub`, `@karume/models`,
`@karume/models/sbv2`). This directory's own configuration points those
specifiers at the registry, never at the workspace members:

| Runtime | Configuration | Entry |
| --- | --- | --- |
| Node | `package.json` dependencies | `"@karume/hub": "jsr:^<version>"` (pnpm installs it from npm.jsr.io) |
| Deno | `deno.json` import map | `"@karume/hub": "jsr:@karume/hub@<version>"` (exact version) |

After each release, bump both files to the released version, then run
`pnpm install` so the Node side picks it up. Keep the Deno entries exact: Deno
reports an import-map entry without resolving its range, so `^<version>` does
not match the expected version and the run fails.

## Why this directory has its own `deno.json`

Inside the workspace, `jsr:@karume/hub@<version>` resolves to the local
workspace member, not to the registry, so a script under the root config would
never exercise the published artifact. This tool therefore runs with its own
`deno.json` (picked up when Deno starts in this directory, or passed with
`--config tools/published-smoke/deno.json`), which is deliberately **not** a
workspace member. Deno prints a one-line warning that it ignores the parent
workspace config, which is expected. That config also excludes `jsr:@karume/*`
from the minimum-dependency-age check (the packages are ours and are checked
right after publishing) and disables the lockfile so no stray `deno.lock` is
written here.
