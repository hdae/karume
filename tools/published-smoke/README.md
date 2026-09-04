# published-smoke — smoke test against the published JSR packages

Runs the post-release check from `docs/release-runbook.md` §5 against the
**published** `@karume/*` packages: every entry of `KARUME_SOURCES` (the published
repository table exported by the `@karume/models` barrel — the union of every
family's `*_SOURCES`) must resolve its manifest through the published
`@karume/hub`, and one family (sbv2) is constructed with `fromPretrained` end to
end.

```sh
deno task smoke:published                   # manifests for every entry + sbv2 fromPretrained (needs a GPU)
deno task smoke:published --manifests-only  # no GPU: manifest resolution only
```

The version under test is read from `packages/models/deno.json` (the three
packages are released in lockstep).

## Why this directory has its own `deno.json`

Inside the workspace, `jsr:@karume/hub@<version>` resolves to the local
workspace member, not to the registry, so a script under the root config would
never exercise the published artifact. This tool therefore runs with
`--config tools/published-smoke/deno.json`, which is deliberately **not** a
workspace member; Deno prints a one-line warning that it ignores the parent
workspace config, which is expected. That config also excludes `jsr:@karume/*`
from the minimum-dependency-age check (the packages are ours and are checked
right after publishing) and disables the lockfile so no stray `deno.lock` is
written here.
