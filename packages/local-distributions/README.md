# @karume/local-distributions

Private workspace package (not published): the distribution mirrors under `models/` (outside git —
[docs/assets-layout.md](../../docs/assets-layout.md)) as the two browser apps read them. The
playground (`apps/playground`) and the workbench (`apps/workbench`) import it; nothing published
may.

| Subpath | Runs in | What it holds |
| --- | --- | --- |
| `./plugin` | Node (Vite config) | `distributionsPlugin`: the dev and preview servers' `/models/index.json`, `/models/<mount>/…` and the HF-compatible `/hf/…` route, with COOP / COEP on every response, and — for an app that asks for it with `inputs` (the workbench) — the files of a directory of the repository at `/inputs/<path>`, read only, by ranges; `distributionMounts` (the table's directories resolved) and `realRoot` (whether a mirror is there), which the Node side shares. |
| `./mounts` | anywhere | `MOUNTS`, the mount table (`/models/<mount>/` → a directory under `models/`, and the public repo a mount is also served under on `/hf/`), and its `MountName` type. |
| `./browser` | a page (Web APIs only) | Reading a mount from a page: `fetchDistributionIndex`, `distributionSource` (the direct route), `hfRepoRef` (the cached route), `pretrainedRef`, `crossRepoMounts` (one selection), `crossRepoHolders` (a whole manifest) and `openMountRoute` (a mount on the route a `SourceKind` names — `cached` or `direct` — with its directory and its manifest's bytes; the twin of `openMountDirectory`). Every URL is resolved against the `base` the page passes. |
| `./node` | Node | `openMountDirectory`: a mount as a `nodeDirectory` source with its directory and its manifest's bytes, cross-repo references mapped by `crossRepoHolders`. It reads the mount table and the opened mirror only — not the `/hf/` serving table, which reads every mirror's manifest — so a broken mirror of another mount does not stop a Node run. |

Its tests (`tests/`) run in the core lane and need no GPU and no mirror.
