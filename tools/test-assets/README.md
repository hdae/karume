# test-assets — fetch the pinned private test-asset dataset

The real-weight tests read the files that no distribution carries — recipe outputs (torch / diffusers references,
inputs captured from upstream) and the inputs this project froze by hand — from the local root
`outputs/test-assets/`. That root is a cache of the private Hugging Face dataset `hdae/karume-test-assets`, fetched at
the revision pinned in `packages/test-support/src/test-assets.ts` (`TEST_ASSETS`). The design is
[ADR 0139](../../docs/decisions/0139-tests-on-distributions-and-test-assets.md).

The dataset root and the local root have the same layout, one folder per kind of asset:

| Path                     | Contents                                                                                       |
| ------------------------ | ---------------------------------------------------------------------------------------------- |
| `goldens/<series>/<path>` | recipe outputs, at the relative path they had under `outputs/series/<series>/` (never containers) |
| `corpus/anima/`          | the four 1024² Anima images (inputs of the real-image checks and of the vision goldens)        |
| `corpus/wan-i2v/`        | the three image-to-video condition images                                                      |
| `.revision`              | written by the fetch only (the revision it downloaded) — not part of the dataset               |

## Fetch

Run from the repository root. The dataset is private, so `hf` must be logged in with a read token that includes it
(the everyday read token):

```
node tools/test-assets/fetch.ts
```

It runs `hf download hdae/karume-test-assets --repo-type dataset --revision <pin> --local-dir outputs/test-assets
--exclude .revision` and writes `outputs/test-assets/.revision` only after `hf` exited 0. A stale `.revision` is removed first, so a failed or
interrupted fetch leaves none; run the command again to resume (`hf` skips files it already has). The assets gate
(`packages/e2e/tests/e2e_assets_gate_test.ts`) is red while `.revision` is absent or differs from the pin
(`KARUME_ALLOW_NO_ASSETS=1` opts out on a machine without assets).

Partial fetch — only the goldens of one series (repeat `--include` for more):

```
node tools/test-assets/fetch.ts --include 'goldens/birefnet-hr-1024-f16/**'
node tools/test-assets/fetch.ts --include 'goldens/lucida-1024/**' --include 'corpus/anima/**'
```

A partial fetch records the revision too: the gate checks the revision, and each test still skips explicitly or fails
when a file it reads is absent.

`hf` is the Hugging Face CLI (huggingface_hub 1.x) on PATH; `KARUME_HF_BIN=<path>` names another executable. The command
fails loudly when `hf` is missing, when it exits non-zero, and when the pin is still the placeholder revision.

## Upload the assets of a new model

1. Stage the files under `outputs/test-assets/` at the dataset's layout: each golden at
   `goldens/<series>/<path>`, the same relative path it has under `outputs/series/<series>/` (no containers, nothing
   the distribution already carries — ADR 0139 decision 4); a new kind of asset gets its own top-level folder.
2. Switch to the write token (`hf auth switch --token-name "Karume Release"`) and upload:

   ```
   hf upload-large-folder hdae/karume-test-assets outputs/test-assets --repo-type dataset --exclude .revision
   ```

   `.revision` is this machine's record of its last fetch, not an asset (the fetch never downloads one either).

3. Put the new commit SHA of the dataset into `TEST_ASSETS.revision` (`packages/test-support/src/test-assets.ts`) **in
   the same commit as the tests that read the new assets**, then run the fetch so `.revision` matches the pin.
4. Switch back to the read token (`hf auth switch --token-name "Karume Gated Read"`; check with `hf auth list`).

## Tests

`tools/test-assets/tests/fetch_test.ts` runs the fetch against a fake `hf` in a temporary directory: the arguments it
receives, `.revision` written on success, not written on failure, and a stale `.revision` removed before the download.
