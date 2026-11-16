# @karume/local-distributions

Private workspace package (not published): the distribution mirrors under `models/` (outside git —
[docs/assets-layout.md](../../docs/assets-layout.md)) as the two browser apps read them. The
playground (`apps/playground`) and the workbench (`apps/workbench`) import it; nothing published
may.

| Subpath | Runs in | What it holds |
| --- | --- | --- |
| `./plugin` | Node (Vite config) | `distributionsPlugin`: the dev and preview servers' `/models/index.json`, `/models/<mount>/…` and the HF-compatible `/hf/…` route, with COOP / COEP on every response; `distributionIndex` and `distributionMounts`, which the Node side shares. |
| `./mounts` | anywhere | `MOUNTS`, the mount table (`/models/<mount>/` → a directory under `models/`, and the public repo a mount is also served under on `/hf/`), and its `MountName` type. |
| `./browser` | a page (Web APIs only) | Reading a mount from a page: `fetchDistributionIndex`, `distributionSource` (the direct route), `hfRepoRef` (the cached route), `pretrainedRef`, `crossRepoMounts` (one selection) and `crossRepoHolders` (a whole manifest). Every URL is resolved against the `base` the page passes. |
| `./node` | Node | `localDistributionIndex` (the facts of `/models/index.json`, read from the directories) and `openMountDirectory` (a mount as a `nodeDirectory` source, cross-repo references mapped by `crossRepoHolders`). |

Its tests (`tests/`) run in the core lane and need no GPU and no mirror.
