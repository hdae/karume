# 0152: Maintained specs, declared numeric requirement levels, and verification that follows what changed

- Status: accepted (2026-11-15 — the maintainer's rulings in the discussion of 2026-11-15. Nothing is built; the work
  items are in [backlog](../backlog.md), section "later", "The move to maintained specs (ADR 0152)")
- Date: 2026-11-15
- Related: [0147](0147-numerics-and-kernel-adoption.md) (levels, evidence classes, who decides — partly replaced, see
  "What this changes in earlier ADRs") / [0106](0106-device-keyed-references.md) (sha256 rows per environment) /
  [0005](0005-verification.md) and [0129](0129-test-layout-and-lanes.md) (lanes, the release run) /
  [0128](0128-default-preset-original-weights.md) (the default preset per model) / [0040](0040-fusion-pass.md) and
  [0148](0148-external-fusion-rules.md) (fusion rules) / [0098](0098-linear-gemv-parallel.md) (measured shape tables) /
  [0115](0115-geometry-profiles.md) and [0117](0117-app-geometry-tuning.md) (profiles, no tuning at run time) /
  [0001](0001-scope-and-non-goals.md), [0016](0016-anima-chain-export.md), [0037](0037-karume-monorepo.md),
  [0006](0006-quantization.md), [0019](0019-i8-weight-execution.md) and [0108](0108-container-format.md) (where weights
  come from) / [0130](0130-project-language.md) (language)

## Context

The project moves from exploratory development to maintaining an established design. A survey of the repository on
2026-11-14 found one cause behind most of the maintenance cost: **bit equality between two implementations** was used
beyond its role as evidence. It defined the default path, froze the sha256 rows, pinned the WGSL text byte for byte and
decided when a geometry profile expires, so one changed kernel line moved checks, snapshots, profiles and documents.

Measured on 2026-11-14:

- A full verify takes 6,754 s. The 82 real-weight GPU e2e files are 93.8% of it; the two Wan lanes are 53%.
- Selecting tests through the import graph does not narrow a kernel change: every real-weight e2e file imports 68–69 of
  the 69 kernel and codegen modules, so one kernel edit selects 99.7% of the time.
- No commit has re-recorded a sha256 row because a kernel or numerics change moved it. About 59 commits cite "rows
  unchanged" as the evidence of a bit-identical change. The artifact a row was written from is not kept with the row.
- There are 151 ADRs. For several of them the rule in force is in an addendum, and the process rules ask for ADR text
  with every kernel, fusion or numerics change. None of the code changes ADR 0147 decided is built.
- Models' end-to-end tests are what vouches for the runtime's kernels, so every added model adds to the runtime's cost.

## Decision

### 1. Documents

- `docs/spec/` holds what the system does now and is the source of truth. It is written in the present tense, covers
  built behaviour only, and is edited in place in the commit that changes the code. It has no addenda.
- `docs/decisions/` holds a decision and its reasons at one point in time. ADR files keep their paths and numbers. An ADR
  is not the pointer for a current rule. A new ADR is written only for a choice among alternatives; refining a rule
  edits the spec. A replaced clause is listed once, in the ADR that replaces it — no status lines across older ADRs.
- A decision that is not built yet is a backlog item and is deleted from the backlog when built.
- Code points at a current rule with `SPEC: docs/spec/<file>.md#<section>`. `DECIDED:` stays as a pointer to reasons.
  Existing `DECIDED:` lines and "ADR nnnn" mentions are not converted in bulk; new ones are not added.
- New spec files are written only where the rule is spread over several ADRs. `docs/architecture/` is not created ahead
  of need. A model family's own rules live in a README next to its code.
- Error messages and diagnostics do not cite an ADR number or clause; the message states the cause and what to do.

Why: the reader of a rule should not have to replay its history, and a rule change should cost one edit.

### 2. Numeric requirement levels, declared by the model

A model's preset declares how far each computation may move from the reference kernel:

| Level | Meaning |
| --- | --- |
| `exact` | the reference kernel's bits on this device |
| `reorder` | the summation order may differ; the error stays inside the op's band against the f64 reference |
| a named approximation | an approximation the preset lists by name (int8 activations, f16 compute, …) |

- The declaration has two scopes — the whole model, then an op kind — and the narrower one wins. An explicit option
  overrides the preset. There is no per-node scope.
- With no declaration the runtime runs `exact`. A preset is what relaxes it.
- Within the declared level the runtime may pick the fastest eligible strategy per device. An approximation is never
  picked unless it is named; "any approximation" means listing them.
- A fusion that spans several ops is applied only when it meets the strictest level among them.
- Equality of bits across devices or backends is not required at any level.
- `exact` is neither "accurate" nor "the same on every GPU": it is "the same as the reference kernel here". A `reorder`
  strategy may be closer to f64 than the reference kernel.

Why: the model knows which computations are sensitive; the runtime knows which kernels are fast on a device. Freezing
the default path kept more accurate and faster kernels out of every default.

### 3. Reference kernels and evidence

- Each op keeps one reference kernel whose bits are frozen. Better kernels are added as `reorder` strategies. Whether
  the reference GEMM reduction is replaced once by a more accurate one is decided after measuring (open).
- The forms of evidence of ADR 0147 decision 2 stay, spelled out: **mirror** (bit-identical to a TS mirror or to the
  untransformed plan), **composition** (bit-identical to host-rounded input plus an existing kernel), **reorder** (the
  four-point check against the op band), **approximation** (a wider band from an error model). They are evidence a
  strategy carries, not the level a model asks for.
- A device probe at Session construction is used only for a capability the adapter's report cannot settle (as the
  subgroup-width requirement is). The existing dp4a probe stays. The Session canary of ADR 0147 decision 9 is not built.

### 4. sha256 rows

- Every preset has rows. A row is stored with the fingerprint of the set of kernels the case compiled and with the
  artifact it was written from.

| Kernel set | sha256 | Result |
| --- | --- | --- |
| same | same | pass |
| same | differs | fail (a change in the model's code, or non-determinism) |
| differs | differs | warning that reports the distance from the stored artifact |

- An all-`exact` configuration always falls in the first two rows, because reference kernels do not change. That
  configuration is how a model's port is checked.

### 5. The runtime checks itself

- The runtime keeps a regression corpus: graphs of real models with synthetic weights at reduced shapes, on which a
  selected strategy is compared with `exact` inside the op band. It needs neither real weights nor `packages/models`.
- A runtime defect that a model's test finds gets a reproduction in the corpus or in an op-level test.

Why: graph rewrites (fusion, strategy selection) fail on real graphs in ways op-level tests do not show, so real graph
structure is needed — but not real weights, and not the models' pipelines.

### 6. Verification cache

- The unit is one test file. A record is reused only when every case passed, and reuse is reported as its own status,
  never as a pass.
- The key has three layers: environment (environment key, adapter features and limits, Node and Dawn versions, the
  `KARUME_*` values), data (manifest hashes of the mirrors read, the test-asset pin, fixtures read) and code.
- The driver version is not in the key. It is stored in the record, and a difference is reported as a warning.
- For a model's e2e file the code layer is the models-side import closure plus the fingerprint of the WGSL it compiles.
  The runtime's host code (executor, buffer pool) is left out of that key, once the regression corpus of decision 5
  exists. Until then it stays in.
- A release may use cached results. An uncached full run is required only when the environment changed (Dawn or Node
  version, driver string).
- Stages: record what each file compiles and reads while still running everything, and count wrong predictions; then
  cache for changes outside the runtime; then narrow kernel changes to the files that compile them.

### 7. Case sizes

- Wan's rows are re-recorded at fewer frames (the default lane already runs two steps). Cases compared with upstream
  goldens keep their shape.
- A model cut after a few layers is a fallback for a model that cannot be loaded whole, and is not extended to other
  families.

### 8. Kernel structure

- An op spec object gathers an op's meaning, shape rule, reference implementation and band. A per-op strategy registry
  replaces the selection chains; the recipe builder, the fusion pass, the estimate, the census and the verification
  cache read the same registry. Both are refactors that move no pipeline key and no row; `linear` and attention first.
  Hand-written kernels enter the registry as they are.
- WGSL byte snapshots are kept for reference kernels only. Other strategies are covered by property tests.
- A shape that was not measured stays on the reference kernel. Shape tables move out of kernel source into tables
  generated per adapter. Nothing is tuned at run time.
- Generating kernels beyond this is not a goal now; the structure only stops blocking it.

### 9. The harness replaces the Playground

- A bare browser app: one file is one entry, one Run button, results as JSON. Locally the dev server writes results
  back under `outputs/`; a public deployment offers a download only. Benchmarks run on synthetic tensors, so a run
  needs no model download. Collecting other people's results is not built.
- The Playground's measurement and check pages are removed when the harness covers them. A model's page stays until
  that model has an example app.

### 10. Weights from outside the exporter

- The TS side may build a container from a graph plus a user's safetensors file, and quantize at load (RTN to `int8` /
  `int4`, `f16`, `bf16`). The test is that the same upstream weights converted in TS give the block sha256 of the
  distributed container. Nothing is checked by sha256 at run time. GPTQ and QAT are not offered at load.
- Whether a converted container is kept is the application's business; the runtime and `models` do not manage it.
- LoRA: merge into the upstream weights before quantization. Merging into quantized weights and a run-time low-rank
  branch wait for a need.
- GGUF is deferred.
- A fine-tune's output has no golden. That is stated as a limitation, not covered by a tolerance.
- Evidence that the test can pass: re-quantizing Anima's bf16 base with the exporter's formula reproduced the
  distributed int8 blocks on a sample of 76 tensors (Python;
  [research 2026-11-15](../research/2026-11-15-quantized-weight-delta.md)).

## What this changes in earlier ADRs

- **0147**: decisions 5, 6 and 9 admitted a variant without an option only when it was u32-identical to the default
  (fusions, geometry); under `reorder` a strategy is admitted on the op band instead. Decision 9's Session canary is not
  built. The removal of the three boolean fusion keys (decision 5) goes ahead. Derived applicability (decision 7) and
  the device switches (decision 8) are built as part of the strategy registry.
- **0106**: rows gain the kernel-set fingerprint and the stored artifact; a mismatch with a different kernel set is a
  warning. A row is still never replaced by a tolerance.
- **0005 / 0129**: "a release needs one green full run" becomes "an uncached full run when the environment changed".
- **0001, 0016, 0037**: `torch.export` is no longer the only way weights reach a Session, LoRA is no longer export-time
  only, and reading upstream safetensors on the TS side is in scope for weights (model definitions still come from the
  exporter's graphs).
- **0006 / 0019**: the exporter's fake-quant model stays the definition of a quantized weight; the TS encoder is a
  second implementation held to it by block sha256.

## Consequences

- Faster and more accurate kernels can run by default wherever a preset allows `reorder`, without moving any `exact`
  row.
- A kernel change re-runs the model tests that compile that kernel, not all of them.
- Between uncached full runs, a runtime defect that leaves the WGSL unchanged (buffer handling, for instance) is caught
  by the runtime's own tests and corpus, not by the models' e2e. This is accepted.
- The default preset of a model is no longer vouched for by frozen bits; its rows warn and report a distance when the
  kernel set changes.
- Session options that name one kernel become unnecessary under `reorder`. This breaks the public API.
- Encoders exist in Python and in TS and must agree byte for byte.

## Open

- Replacing the reference GEMM reduction once (measure accuracy against f64, speed and upstream parity first).
- Whether the WGSL a file compiles can be derived without a device for every pipeline (the estimate path does it for
  session plans).
- Whether the regression corpus works for graphs with dynamic shapes.
- A full-model case for Gemma 4 26B through the expert cache, next to the cut model.
- The harness's UI stack, and where per-GPU evidence is kept.
