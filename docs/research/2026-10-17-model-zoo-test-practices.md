# How diffusers and transformers test a large model zoo

> Snapshot of 2026-10-17: read from diffusers `b800f828`, transformers `d4804f9d` and transformers-ci `bd97403f` (links are pinned to those commits), plus CI run data from the GitHub Actions API on the same day. Upstream changes often; re-check before relying on any detail.

Purpose: input for deciding how Karume maintains model families and exporter recipes after the 0.15 test triage ([2026-10-16-test-triage.md](2026-10-16-test-triage.md)).
Every claim below cites a primary source (repository file at a pinned commit, or a CI run).
Inferences are marked **(inference)**.

## Summary

| | diffusers | transformers |
| --- | --- | --- |
| Per-model fast tests | one config class + shared tester mixins, tiny random components built in the test | one `ModelTester` (tiny config) + `ModelTest` with shared mixins |
| Tiny Hub models | `hf-internal-testing/tiny-random-*` for tokenizers / text encoders | `hf-internal-testing/tiny-random-{Class}`, regenerated daily, pinned by commit sha in a JSON summary |
| Real-weight tests | `@slow` / `@nightly`, gated on `RUN_SLOW` / `RUN_NIGHTLY` | `@slow`, gated on `RUN_SLOW` |
| Where real weights run | nightly (all pipelines with any downloads); on push to main only fast tests on GPU, for the high-usage subset | daily on every model folder, single- and multi-GPU; on push only for 27 "important" models |
| Expected values | hardcoded slices; `Expectations` keyed by device type + compute-capability major | same `Expectations` mechanism, plus exact generated text |
| Deprecation | `DeprecatedPipelineMixin` → tests auto-skipped, code frozen | `deprecate_models.py` moves code to `deprecated/` and **deletes the tests** |
| Daily GPU budget (one run) | 121 jobs, 10.5 job-hours, ~1 h 16 min wall | 1241 jobs, 122.1 job-hours, ~4 h 36 min wall |

## 1. Where per-model tests live

### diffusers

- Pipeline tests are pytest classes that combine one per-pipeline config class with shared mixins. `FluxPipelineTesterConfig` declares `pipeline_class`, `get_dummy_components()` and `get_dummy_inputs()`. `TestFluxPipeline(FluxPipelineTesterConfig, PipelineTesterMixin)` inherits the common tests, and further classes add memory / LoRA / cache mixins ([D-flux] L49–L242).
- The config base declares only the contract. The mixin provides common tests: save/load, batch consistency, dtype/device moves, callbacks, variants ([D-common] L98–L111, L296–L303).
- The dummy components are real diffusers classes at tiny size (one transformer layer, 8×8 output, 2 inference steps). Tokenizers and the T5 config come from `hf-internal-testing/tiny-random-t5` and `tiny-random-clip` ([D-flux] L56–L136).
- Contributor rule: build dummies from the **real classes** at tiny config, not hand-written mocks. A mock copied from what the pipeline reads today "can only confirm the pipeline against itself". A tiny Hub repo must mirror the real checkpoint's layout ([D-testing-md] L9–L10, L71–L72).
- Several fast tests compare against the pipeline's own output (save → load → same output within `atol=5e-4`), not against fixed numbers ([D-common] L305–L329).
- Model-level tests have their own mixin set under `tests/models/testing_utils/` ([D-tree-models]).

### transformers

- Each model's test file has a `ModelTester` that builds a tiny config and dummy inputs. A `ModelTest(unittest.TestCase + mixins)` inherits the generated tests ([T-testing-md] L231–L234).
- Decoder LLMs need almost no code. `LlamaModelTester` only sets `base_model_class`, and the other classes are inferred by naming convention ([T-llama] L47–L61, [T-causal] L50–L64).
- Default tiny text config: `hidden_size=32`, `num_hidden_layers=2`, `vocab_size=99`, `seq_length=7` ([T-tu] L244–L264).
- `tests/test_modeling_common.py` is 6,725 lines and defines 120 `test_*` methods that every model inherits ([T-common]).
- Tiny Hub models are named `hf-internal-testing/tiny-random-{ModelClassName}`. They are generated from `ModelTester.get_config()` by `utils/create_dummy_models.py` and recorded in `tests/utils/tiny_model_summary.json` ([T-testing-md] L397–L415). The summary has 647 entries, each pinned to a Hub commit `sha` ([T-tiny-json]). A daily workflow regenerates them ([T-check-tiny] L9).
- Docs: tiny models are a "last resort" for integration tests. Use them only when the smallest real checkpoint exceeds ~24 GB VRAM ([T-testing-md] L399).

## 2. Real-weight tests and how CI tiers them

### Markers

- diffusers: `@slow` skips unless `RUN_SLOW` is set, and `@nightly` skips unless `RUN_NIGHTLY` is set ([D-tu] L304–L341). `@require_big_accelerator` needs ≥ `BIG_GPU_MEMORY` GB, default 40 ([D-tu] L76, L650–L676). Feature markers (`lora`, `quantization`, `compile`, `big_accelerator`, …) let CI select subsets with `-m` ([D-conftest] L34–L54).
- transformers: `@slow` skips unless `RUN_SLOW` is set. `@tooslow` is an unconditional skip for tests "in the process of being fixed": "No test should stay tagged as tooslow" ([T-tu] L320–L328, L427–L451). In the snapshot, 501 of 520 `test_modeling_*.py` files contain `@slow`, and `@tooslow` appears 17 times (rg count).
- `is_flaky(max_attempts=5)` retries a test in both repos ([D-tu] L1310, [T-tu] L2993–L3010). transformers asks contributors to prefer extending the CI's rerun-on-pattern list over `is_flaky` ([T-tu] L2997–L3000). Usage counts: 87 `@is_flaky` in transformers tests, and 7 files in diffusers (rg count).

### CI tiers — diffusers

| Trigger | What runs | Hardware | Source |
| --- | --- | --- | --- |
| PR | fast tests on CPU (pipelines, models/schedulers, examples) | CPU runners | [D-pr] L94–L165 |
| PR touching core loading / test-mixin files | mixin tests per pipeline module on GPU, max 8 parallel | `aws-g4dn-2xlarge` | [D-pr-gpu] L4–L17, L129–L165 |
| Push to main | fast tests on GPU (`RUN_SLOW` unset), for pipelines with ≥ 50,000 Hub downloads plus 9 always-tested modules | `aws-g4dn-2xlarge` | [D-push] L3–L20, [D-matrix] L13–L24 |
| Nightly (cron 00:00) | `RUN_SLOW` + `RUN_NIGHTLY`, cutoff 0 = every pipeline with any downloads; plus models, compile, quantization backends, minimum torch | g4dn; big-GPU tests on `aws-g6e-xlarge-plus` with `-m big_accelerator` | [D-nightly] L3–L17, L65–L73, L233–L273 |

- The GPU test matrix is chosen by **Hub download counts** of the pipeline class ([D-matrix] L45–L63). Usage decides what is tested on GPU per push.
- Releases are gated: "maintainers first check that the integration is complete and that release tests pass" ([D-contrib] L350).

### CI tiers — transformers

| Trigger | What runs | Source |
| --- | --- | --- |
| PR | non-`@slow` tests selected by `utils/tests_fetcher.py` from the import graph. If ≥ 15 models are impacted, everything runs | [T-fetcher] L15–L30, L74; [T-prchecks] L122–L124 |
| PR, on demand | a member comments `run-slow: <models>` → slow tests for those models on Nvidia GPUs | [T-comment-ci] L31; [T-prchecks] L157 |
| Push to main | slow tests on GPU only for the 27 `IMPORTANT_MODELS` touched by the diff, or all 27 if a core file (`modeling_utils.py`, `cache_utils.py`, …) changed | [T-push-important]; [T-important] |
| Daily (cron 02:17) | `RUN_SLOW=yes` on **every** `tests/models/*` folder, each on single-GPU (`aws-g5-4xlarge-cache`) and multi-GPU (`aws-g5-12xlarge-cache`) machines, split into 3 slices to stay under GitHub's 256-job matrix cap | [T-sched] L6, L62; [TC-daily] L75, L93, L130–L139 |

- After a daily run, failing tests that are new are bisected automatically with `git bisect run`. Each test is rerun up to 4 times to classify it as flaky ([T-bad-commit] L35–L44, L155–L163).
- Additional scheduled callers exist for AMD MI250 / MI300 / MI355, Intel Gaudi / Gaudi3 and flash-attn ([T-workflows]).

## 3. Expected values: format, tolerance, pinned device/dtype

- **Hardcoded slices.** Both repos store a small slice of the output literally in the test. Examples: the first and last 8 values of a Wan video ([D-wan] L125–L141), or `logits[0, 0, :15]` of Llama-2-7B ([T-llama] L125–L135).
- **Fast-test slices are tied to CPU.** "Run on CPU: the expected slice below is CPU-specific", compared with `atol=1e-3` ([D-wan] L126, L141). diffusers' new-model parity check is also CPU/float32 with `max_diff < 1e-3` ([D-skill] "Model parity test").
- **Device-keyed expectations.** Both repos have an `Expectations` dict keyed by device type plus compute capability. diffusers matches on the major version; the best match is scored: same type 8 points, cuda↔rocm 4, same major 2, default 1 ([D-tu] L1766–L1802). transformers also accepts `(major, minor)` keys such as `("rocm", (9, 5))` ([T-tu] L3679, [T-testing-md] L378, [T-llama] L74). Real-weight tests carry per-device entries, e.g. `("cuda", None)` and `("xpu", 3)` for FLUX.1-schnell ([D-flux] L406–L412), and `cuda` 7/8, `rocm`, `xpu` for Llama ([T-llama] L105–L135). `Expectations(` appears in 192 transformers model test files and in 20 diffusers test files (rg count).
- **Reference hardware is stated.** "Transformers CI runs slow tests on an NVIDIA A10. Numerical results can vary slightly across GPU generations" ([T-testing-md] L378).
- **Tolerances for real weights are loose and metric-based.** diffusers: cosine distance of the slice `< 1e-4` in bf16 ([D-flux] L396–L416). transformers: `allclose(atol=1e-2, rtol=1e-2)` on bf16/fp16 logits ([T-llama] L93–L135), or exact string equality on greedy text ([T-llama] L67–L91).
- **Determinism.** Generation tests use `do_sample=False`, and MoE models force the eager expert path so router noise cannot flip experts ([T-testing-md] L360). diffusers' `enable_full_determinism()` sets deterministic algorithms, disables TF32 and sets `CUBLAS_WORKSPACE_CONFIG` ([D-tu] L1438–L1456). The nightly GPU jobs also set `CUBLAS_WORKSPACE_CONFIG` ([D-nightly] L98). Generators are CPU-seeded ([D-flux] L375).
- **Heavy inputs are fixtures on the Hub.** The FLUX integration test loads precomputed prompt embeddings from the `diffusers/test-slices` dataset, so the test skips the text encoders ([D-flux] L375–L398).

## 4. What a new model / pipeline / conversion script must ship

- **diffusers**
  - Two test layers are required: pipeline-level tests, plus model-level tests if a new model is introduced ([D-testing-md] L5).
  - Integration / slow tests and LoRA tests are **not** added in the initial PR. They come later, after discussion with maintainers ([D-testing-md] L5, L13).
  - Conversion goes in a standalone `scripts/convert_<model>_to_diffusers.py`. It must round-trip through `save_pretrained` / `from_pretrained` ([D-skill] "Weight / Checkpoint Conversion").
  - The parity test against the reference repo is required during integration but is **not shipped**, because it imports the reference code ([D-skill] "Model parity test").
  - `scripts/` is covered by lint only (`check_dirs` in the Makefile) ([D-make] L6, L45).
  - The PR template asks "Did you write any new necessary tests?" and whether the author belongs to the model's team ([D-pr-template]).
- **transformers**
  - The guide asks for integration tests first. Run them with `RUN_SLOW=1 pytest …IntegrationTests`, and include a meaningful end-to-end example ([T-add-model] L489–L495, L578–L582).
  - Conversion must initialize all weights and print unused checkpoint keys ([T-add-model] L345–L353).
  - The 338 in-tree `convert_*.py` scripts are excluded from coverage ([T-pyproject] L1–L5) and from the test fetcher's dependency graph ([T-fetcher] L725, L787). Some models do unit-test a conversion function, e.g. `pp_doclayout_v4` ([T-ppdoc] L149–L169), but this is the exception.

## 5. Deprecated and low-priority models

- **diffusers.** A pipeline that inherits `DeprecatedPipelineMixin` warns on instantiation and records `_last_supported_version` ([D-pu] L161–L180). The shared test fixture skips it ([D-common] L235–L240). The deprecated folder README: very low usage "as measured by model downloads, issues and PRs … we will stop testing the pipelines and will not accept any changes" ([D-dep-readme]).
- **diffusers, observed side effect.** The nightly matrix does not filter deprecated pipelines (cutoff 0 with download-based selection — [D-matrix], [D-nightly] L17). In run 38987307407 the `alt_diffusion` job ended with "no tests ran" and exit code 5, so it showed as failed ([D-run-log]). **(inference)** Some of that run's 66 failed jobs are of this kind rather than regressions. I checked only one such log.
- **transformers.** `utils/models_to_deprecate.py` proposes candidates with < 5,000 downloads and no commit since a date, default one year ([T-to-deprecate] L305–L322). `utils/deprecate_models.py` adds a "maintenance mode only, we don't accept any new PRs" tip to the docs and moves the code to `models/deprecated/`. It also **`git rm`s `tests/models/<model>`** ([T-deprecate] L44–L55, L145–L166, L350–L352). In this snapshot `models/deprecated/__init__.py` lists no models ([T-dep-init]). **(inference)** Earlier deprecated models were removed outright in a major release, but I did not confirm this from git history (shallow clone).
- **Priority tiers without deprecation.** Both repos run a small named set more often: diffusers' 9 `ALWAYS_TEST_PIPELINE_MODULES` plus the ≥ 50k-download set on every push ([D-matrix] L13–L24), and transformers' 27 `IMPORTANT_MODELS` on push ([T-important]). Everything else gets real-weight coverage only in the daily / nightly run.

## 6. CI scale (one recent run each, GitHub Actions API, 2026-10-17)

- **diffusers nightly, run 38987307407:** 121 jobs (104 per-pipeline jobs), 55 success / 66 failure. Summed job time 10.5 h, wall 00:38 → 01:54 UTC. The last 5 nightly runs all concluded `failure` ([D-runs]). One failure was an OOM on a 14.74 GiB GPU in the CogVideoX integration test ([D-run-log2]).
- **transformers Nvidia daily CI, run 38995454912:** 1241 jobs with start/end times (1102 of them `run_models_gpu`, ≈ one per model folder × 2 machine types). 882 success / 315 failure / 44 skipped. Summed 122.1 job-hours, wall 02:34 → 07:10 UTC ([T-runs]).
- Runner GPUs per AWS instance families: g4dn = NVIDIA T4 (16 GB), g5 = A10G (24 GB), g6e = L40S (48 GB) ([AWS]).
- **(inference)** Both daily runs are red by default. Failures are triaged by diffing against the previous run (transformers' new-failure bisect, diffusers' consolidated report — [D-nightly] L470–L500), not by demanding an all-green run.

## Applicability to Karume

Karume facts: per-family test lanes · real-weight e2e in `packages/e2e` · sha256 regression check with per-environment rows (ADR 0106) · exporter recipes in Python.

- **Fits:** the two-layer split (tiny/structural tests every run, real weights in a separate tier) already matches lanes + `packages/e2e`.
- **Fits:** device-keyed expectations are the same idea as ADR 0106's environment rows. Upstream keys more coarsely (device type + compute-capability major) and accepts tolerance; Karume keeps bit-exact sha per environment by design.
- **Fits:** an explicit "maintained, not run by default" tier is normal upstream (push subset vs. daily all). It needs a scheduled run that still covers it. Otherwise it decays like diffusers' deprecated pipelines.
- **Fits:** deprecation = freeze code + stop testing (diffusers) or delete tests (transformers). Both are explicit, tooling-driven decisions, not silent skips.
- **Fits (inference):** for recipes, the diffusers pattern applies: parity check against the reference at integration time, not shipped; lint-only afterwards. The thing to keep testing is the produced artifact (Karume's e2e), not the conversion script.
- **Doesn't fit:** usage-based selection (Hub download counts) — Karume has no comparable telemetry.
- **Doesn't fit:** upstream's scale (hundreds of GPU job-hours per day, standing red runs triaged by diff). A single-machine project cannot absorb a chronically red tier.
- **Doesn't fit:** tiny random-weight Hub repos as fixtures. Karume's sha gate needs real assets, and tiny models only exercise the code path.

## Addendum: direction for Karume (maintainer, 2026-10-17)

Proposed by the maintainer after reading this note. It is a direction, not yet a decision record; the open points below need a ruling before CLAUDE.md / ADR 0129 change.

- **Priority set.** Keep a short, explicitly listed set of the most important families and run it first. The list changes as the focus moves. This is the analogue of transformers' `IMPORTANT_MODELS` and diffusers' always-tested modules (§5). Upstream keeps the list as a file in the repo ([T-important], [D-matrix] L13–L24). Karume's analogue would be a list next to `LOW_PRIORITY_FAMILIES` in the lane table.
- **Full run.** At Karume's scale, running every family on every change is unnecessary. A full run weekly, or at each release, is enough. Upstream runs everything daily (§6), but its daily budget is 10–120 GPU job-hours on a fleet. Karume has one machine.
- **Why a periodic full run is still needed.** A tier that never runs decays without anyone noticing. diffusers' deprecated pipelines are the example: their nightly jobs fail with "no tests ran" and nobody acts on it (§5).
- **Cost of the longer interval (inference).** A weekly run makes the window for finding a regression up to a week of commits. Karume's sha256 gate is bit-exact and repeatable on one machine, so `git bisect run` on the failing lane gives a clean answer. Upstream needs repeated runs per commit to rule out flakiness ([T-bad-commit] L155–L163).

Open points:

1. **Release gate and the low-priority families.** The current rule says the full verify and the release gate run on the default set, which excludes minicpm5 / siglip2 (CLAUDE.md, verification section). "All families at each release" would add them to the release gate. Decide which applies.
2. **Core changes.** The current rule runs the full suite after larger core changes. The transformers analogue is core change → all priority models, with the full suite left to the scheduled run ([T-push-important]). Adopting it would narrow the current rule.
3. **Who triggers the weekly run.** Karume has no scheduled GPU CI. The weekly full run would be a manual step (or a local scheduled task) on the development machine.

## Sources

[D-flux]: https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/tests/pipelines/flux/test_pipeline_flux.py
[D-common]: https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/tests/pipelines/testing_utils/common.py
[D-wan]: https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/tests/pipelines/wan/test_wan.py
[D-tu]: https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/tests/testing_utils.py
[D-conftest]: https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/tests/conftest.py
[D-tree-models]: https://github.com/huggingface/diffusers/tree/b800f828f3951c4d206e154411bcbfa693cd322e/tests/models/testing_utils
[D-pu]: https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/src/diffusers/pipelines/pipeline_utils.py
[D-dep-readme]: https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/src/diffusers/pipelines/deprecated/README.md
[D-pr]: https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/.github/workflows/pr_tests.yml
[D-pr-gpu]: https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/.github/workflows/pr_tests_gpu.yml
[D-push]: https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/.github/workflows/push_tests.yml
[D-nightly]: https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/.github/workflows/nightly_tests.yml
[D-matrix]: https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/utils/fetch_torch_cuda_pipeline_test_matrix.py
[D-skill]: https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/.ai/skills/model-integration/SKILL.md
[D-testing-md]: https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/.ai/references/testing.md
[D-pr-template]: https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/.github/PULL_REQUEST_TEMPLATE.md
[D-contrib]: https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/docs/source/en/conceptual/contribution.md
[D-make]: https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/Makefile
[D-runs]: https://github.com/huggingface/diffusers/actions/workflows/nightly_tests.yml
[D-run-log]: https://github.com/huggingface/diffusers/actions/runs/38987307407/job/117261961017
[D-run-log2]: https://github.com/huggingface/diffusers/actions/runs/38987307407
[T-llama]: https://github.com/huggingface/transformers/blob/d4804f9d76b8c071b993d813e9e87067b780cbee/tests/models/llama/test_modeling_llama.py
[T-causal]: https://github.com/huggingface/transformers/blob/d4804f9d76b8c071b993d813e9e87067b780cbee/tests/causal_lm_tester.py
[T-common]: https://github.com/huggingface/transformers/blob/d4804f9d76b8c071b993d813e9e87067b780cbee/tests/test_modeling_common.py
[T-tu]: https://github.com/huggingface/transformers/blob/d4804f9d76b8c071b993d813e9e87067b780cbee/src/transformers/testing_utils.py
[T-testing-md]: https://github.com/huggingface/transformers/blob/d4804f9d76b8c071b993d813e9e87067b780cbee/docs/source/en/testing.md
[T-prchecks]: https://github.com/huggingface/transformers/blob/d4804f9d76b8c071b993d813e9e87067b780cbee/docs/source/en/pr_checks.md
[T-add-model]: https://github.com/huggingface/transformers/blob/d4804f9d76b8c071b993d813e9e87067b780cbee/docs/source/en/add_new_model.md
[T-tiny-json]: https://github.com/huggingface/transformers/blob/d4804f9d76b8c071b993d813e9e87067b780cbee/tests/utils/tiny_model_summary.json
[T-check-tiny]: https://github.com/huggingface/transformers/blob/d4804f9d76b8c071b993d813e9e87067b780cbee/.github/workflows/check_tiny_models.yml
[T-fetcher]: https://github.com/huggingface/transformers/blob/d4804f9d76b8c071b993d813e9e87067b780cbee/utils/tests_fetcher.py
[T-comment-ci]: https://github.com/huggingface/transformers/blob/d4804f9d76b8c071b993d813e9e87067b780cbee/.github/workflows/self-comment-ci.yml
[T-push-important]: https://github.com/huggingface/transformers/blob/d4804f9d76b8c071b993d813e9e87067b780cbee/.github/workflows/push-important-models.yml
[T-important]: https://github.com/huggingface/transformers/blob/d4804f9d76b8c071b993d813e9e87067b780cbee/utils/important_files.py
[T-sched]: https://github.com/huggingface/transformers/blob/d4804f9d76b8c071b993d813e9e87067b780cbee/.github/workflows/self-scheduled-caller.yml
[T-workflows]: https://github.com/huggingface/transformers/tree/d4804f9d76b8c071b993d813e9e87067b780cbee/.github/workflows
[T-bad-commit]: https://github.com/huggingface/transformers/blob/d4804f9d76b8c071b993d813e9e87067b780cbee/utils/check_bad_commit.py
[T-pyproject]: https://github.com/huggingface/transformers/blob/d4804f9d76b8c071b993d813e9e87067b780cbee/pyproject.toml
[T-ppdoc]: https://github.com/huggingface/transformers/blob/d4804f9d76b8c071b993d813e9e87067b780cbee/tests/models/pp_doclayout_v4/test_modeling_pp_doclayout_v4.py
[T-to-deprecate]: https://github.com/huggingface/transformers/blob/d4804f9d76b8c071b993d813e9e87067b780cbee/utils/models_to_deprecate.py
[T-deprecate]: https://github.com/huggingface/transformers/blob/d4804f9d76b8c071b993d813e9e87067b780cbee/utils/deprecate_models.py
[T-dep-init]: https://github.com/huggingface/transformers/blob/d4804f9d76b8c071b993d813e9e87067b780cbee/src/transformers/models/deprecated/__init__.py
[T-runs]: https://github.com/huggingface/transformers/actions/runs/38995454912
[TC-daily]: https://github.com/huggingface/transformers-ci/blob/bd97403f18371f92422241cad56bfbd993cbce34/.github/workflows/daily-ci_reusable.yml
[AWS]: https://aws.amazon.com/ec2/instance-types/

- diffusers: [test_pipeline_flux.py][D-flux] · [pipeline testing_utils/common.py][D-common] · [test_wan.py][D-wan] · [tests/testing_utils.py][D-tu] · [conftest.py][D-conftest] · [model testing_utils/][D-tree-models] · [pipeline_utils.py][D-pu] · [deprecated/README.md][D-dep-readme] · [pr_tests.yml][D-pr] · [pr_tests_gpu.yml][D-pr-gpu] · [push_tests.yml][D-push] · [nightly_tests.yml][D-nightly] · [fetch_torch_cuda_pipeline_test_matrix.py][D-matrix] · [model-integration SKILL.md][D-skill] · [.ai/references/testing.md][D-testing-md] · [PR template][D-pr-template] · [contribution.md][D-contrib] · [Makefile][D-make] · [nightly runs][D-runs] · [alt_diffusion job log][D-run-log] · [nightly run 38987307407][D-run-log2]
- transformers: [test_modeling_llama.py][T-llama] · [causal_lm_tester.py][T-causal] · [test_modeling_common.py][T-common] · [testing_utils.py][T-tu] · [testing.md][T-testing-md] · [pr_checks.md][T-prchecks] · [add_new_model.md][T-add-model] · [tiny_model_summary.json][T-tiny-json] · [check_tiny_models.yml][T-check-tiny] · [tests_fetcher.py][T-fetcher] · [self-comment-ci.yml][T-comment-ci] · [push-important-models.yml][T-push-important] · [important_files.py][T-important] · [self-scheduled-caller.yml][T-sched] · [workflows/][T-workflows] · [check_bad_commit.py][T-bad-commit] · [pyproject.toml][T-pyproject] · [test_modeling_pp_doclayout_v4.py][T-ppdoc] · [models_to_deprecate.py][T-to-deprecate] · [deprecate_models.py][T-deprecate] · [models/deprecated/\_\_init\_\_.py][T-dep-init] · [daily run 38995454912][T-runs]
- transformers-ci: [daily-ci_reusable.yml][TC-daily]
- AWS: [EC2 instance types][AWS]
