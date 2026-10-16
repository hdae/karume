# @karume/e2e

Private workspace package: the real-weight end-to-end tests of the Karume packages. Every test here
needs local model assets (`models/` distribution mirrors, `outputs/series/` exporter output,
`inputs/`) and a GPU; the all-skip gates (`KARUME_ALLOW_NO_ASSETS`, `KARUME_ALLOW_NO_DISTRIBUTION`,
`KARUME_ALLOW_NO_GPU`) make a machine without them fail loudly instead of passing empty.

The tests are grouped into lanes by model family (`packages/test-support/src/lanes.ts`); a lane is
run with

```bash
vp test --project models:wan-ti2v
```

and the full suite with `vp test`. Runs that use the GPU should be serialised with
`flock /tmp/karume-gpu.lock …` when another session may own the device. The sha256 reference rows
are keyed by environment (ADR 0106): a machine without rows runs the sha-gated cases as explicit
skips and the reference gate turns red until `KARUME_REFERENCE=write vp test --project <lane>`
creates its rows.

The files were moved here from `packages/runtime/tests` (12) and `packages/models/tests` (49) in
the 0.15 migration (ADR 0129); their helpers still live next to the package they test
(`packages/runtime/tests/helpers`, `packages/models/tests/helpers`).
