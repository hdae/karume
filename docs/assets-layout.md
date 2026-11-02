# Where assets live (models / outputs / inputs)

Conventions for the three local asset roots (the four-way split of outputs is the ruling of 2026-08-30; `outputs/release/` is a fifth directory that
grew later out of publishing work; `outputs/verify/` is a sixth that appeared on 2026-09-20 as the directory for verification results; `outputs/diag/` is a seventh, written down
on 2026-10-15 as the directory for investigation records).
The canonical spelling is `tools/export-recipes/_shared/paths.py` (`DIST_ROOT` / `SERIES_ROOT` /
`EXAMPLES_ROOT` / `BENCH_ROOT` / `VERIFY_ROOT` / `MISC_ROOT` / `INPUTS_ROOT` / `OUTPUTS_ROOT`).
**None of the three roots is tracked by git** (everything in them is either a regenerable artifact or real weights placed by hand; the repository holds
only how to make them).

| Root                                                  | Contents                                                                                                                                                                                       | Example                                                        |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `models/`                                             | **distributions only** (1 directory = 1 HF repository; can be uploaded as is)                                                                                                                 | `models/karume-anima/` / `models/karume-sbv2-jvnv/`            |
| `outputs/series/`                                     | the exporter's series output (containers + golden fixtures `io.*`) — an implementation-time intermediate since ADR 0139: the default lanes read `models/` and `outputs/test-assets/`; only `KARUME_SERIES=1` checks read it | `outputs/series/sbv2-F1-f16/`                                  |
| `outputs/test-assets/`                                | the fetched cache of the private dataset `hdae/karume-test-assets` at the pinned revision: `goldens/<series>/…` (recipe outputs) and `corpus/<kind>/…` (hand-frozen inputs) — ADR 0139 (**safe to delete**; `node tools/test-assets/fetch.ts` restores it) | `outputs/test-assets/goldens/lucida-1024/`                     |
| `outputs/examples/<model>/`                           | default output directory of the example scripts (`<model>` = the basename of `--source`; the CLI examples were removed in 0.15 — `bb35fe2a`)                                                   | `outputs/examples/karume-sbv2-jvnv/*.wav`                      |
| `outputs/bench/<model>/<date>_<purpose>/`             | e2e dumps, benchmarks, visual evaluation (**safe to delete** — the successor of the old `demo/`)                                                                                               | `outputs/bench/karume-anima/2026-08-30_e2e-mismatch/`          |
| `outputs/verify/<environment key>/<date>_<series>/`   | verification results and artifacts (**safe to delete** — `results.json` + PNG / WAV)                                                                                                           | `outputs/verify/deno-intel-graphics-bmg-g21/2026-09-20_anima/` |
| `outputs/misc/<name>/`                                | host assets (**deleting them means re-fetching and re-emitting**)                                                                                                                             | `outputs/misc/sbv2-demo/` / `outputs/misc/corpus/`             |
| `outputs/diag/<topic>-<date>/`                        | investigation records (scripts, probes, raw data — **includes things that cannot be regenerated**; scripts that are run repeatedly do not go here — see the bullet below)                     | `outputs/diag/deberta-w8a8-2026-10-07/`                        |
| `outputs/release/`                                    | the workbench for publishing work (**two kinds, by content** — see the bullet below)                                                                                                           | `outputs/release/upload-karume-siglip2.log`                    |
| `inputs/<family>/<name>/`                             | real weights placed by hand (ckpt, config — not generated)                                                                                                                                     | `inputs/sbv2/F1/`                                              |
| `inputs/anima/civitai-<versionId>/`                   | Civitai intake (weights + `civitai.json` — ADR 0088)                                                                                                                                           | `inputs/anima/civitai-2983680/`                                |
| `inputs/umt5/<name>/`                                 | intake of a compatible text encoder (only the upstream files as they are + `config.json`, the machine-owned `intake.json` and the human-written `license-review.md` — ADR 0122 decision 5)     | `inputs/umt5/community-ft/`                                    |
| `outputs/misc/local-dist/<name>/`                     | local experimental mirror (from an intake; the same form as a distribution, but not published — rejected by real path under `models/`; ADR 0122 decisions 5 / 6)                              | `outputs/misc/local-dist/community-ft/`                        |

- Series output also holds **host-side assets** besides containers (compile products that have no graph) —
  a tokenizer is `<series name>-tokenizer/tokenizer.json` (for example
  `outputs/series/gemma4-e2b-tokenizer/`), umT5 is `wan2.1-umt5-tokenizer/tokenizer.json` (the vocabulary and the promptClean preprocessing tables in one file — ADR 0119 step 10a), the standalone umT5 (the asset `tokenizer` of `karume-umt5-xxl`) is `umt5-xxl-tokenizer/tokenizer.json` (T5 tables only, no preprocessing tables; format `karume-umt5-tokenizer/1` — `python -m wan.umt5_xxl_tokenizer`), and anima's demo tables are `anima-demo/text/`. They are not mixed into an export series
  directory (the undeclared-file check of `dist` would catch them).
- Series names around Wan: the umT5 encoder is `umt5-xxl-i8-dyn` (named after its origin, the original `google/umt5-xxl` — ADR
  [0122](decisions/0122-umt5-upstream-and-compatible-encoders.md) decision 4), an intake is `umt5-xxl-<name>-i8-dyn`, the Wan2.2 TI2V-5B DiT is
  `wan2.2-ti2v-5b-i8-dyn/transformer` (ADR [0121](decisions/0121-wan22-ti2v-5b.md) stage 1), and the VAE (decoder first / next and the I2V encoder pre / attn / post) is
  `wan2.2-ti2v-5b-f16-dyn` (it holds no DiT — ADR 0121 decision 7; the encoder is stage 9a — `python -m wan.export_vae_encoder`).
- Series names of MOSS-Image (`main`; its distribution is a local mirror `models/karume-moss-image/`, built by
  `dist.py --pipeline moss-image` and not published — ADR [0138](decisions/0138-moss-image-dit-vae.md)): the DiT graphs
  `dit` / `dit_skip22` / `dit_text` are `moss-image-sft-i8-dyn` (assets `rope_base` and `training_grid`), the VAE-only series
  is `moss-image-sft-f32-dyn` (named by its f32 storage; it holds no DiT — ADR 0138 decision 8), the precomputed text
  embeddings are the host asset `moss-image-sft-text-embeds/text_embeds.safetensors`, and the layer-by-layer goldens are
  `moss-image-sft-i8-dyn-reference`. Its records go under `outputs/bench/karume-moss-image/`.
- The PLE of the gemma4 product series (the index `ple_index` and the `ple.values.<k>` / `ple.scales.<k>` blocks) goes
  into the part list as an **asset** of the product container `krm` (ADR
  [0109](decisions/0109-manifest-v5-container.md) decision 4), so the series directory holds no separate file
  for it. Unlike the tokenizer it is not an independent component but **a member of the same container** as the product graph, so
  a "new graph + old PLE" pair cannot be made. `ple.probe.safetensors` directly under the series (the dequantization reference) is
  an e2e fixture like `io.*` and is not distributed.
- `<date>_<purpose>` in `bench/` is the run date YYYY-MM-DD + a short identifying slug (`e2e-mismatch` /
  `eval-images` / `quant-sim` etc.). The convention exists to prevent mixing up files, and machines (tests, scripts) also
  write in this form.
- The first level of `verify/` is the **environment key** (`<runtime>-<adapter name slug>` — for example
  `deno-intel-graphics-bmg-g21`; `packages/runtime/tests/helpers/environment.ts` builds it from
  `GPUAdapterInfo`), and the second level is `<date>_<series>` (the run date YYYY-MM-DD, as in `bench/`).
  It holds `results.json` (the outcome of each case + the identity of the environment + the checkout) and the artifacts (PNG / WAV;
  kept on matching runs too). The writers are the e2e tests, and the spelling pairs with `VERIFY_ROOT` on the exporter side
  (there is no Python writer at present, but that is the canonical spelling when a script that reads the results is added).
  Running the same series twice on the same day keeps the last run. The canonical source is ADR
  [0106](decisions/0106-device-keyed-references.md).
- `outputs/misc/corpus/` was a **frozen copy of test inputs** (superseded 2026-11-02: the four images are `corpus/anima/` of the test assets, and the three
  `inputs/wan-i2v/*.png` are `corpus/wan-i2v/` — ADR 0139; the tests read only the fetched copies) (4 real images = the inputs of the real-image checks and the golden generation of depth-anything / birefnet /
  siglip2; the real audio `vowel-*.wav` likewise). The canonical generators were
  `examples/anima/eval-images.ts` (removed in 0.15) / `examples/irodori/eval-audio.ts` (removed in 0.15) (they wrote to bench), and the adopted ones are
  **copied here by hand** — keeping the generation target separate from the frozen copy prevents a benchmark rerun from silently
  overwriting the corpus.
- Fetching the Japanese dictionary (`*.jtd`) under `outputs/misc/yomi/`: unpack `naist-jdic.jtd.gz` from the HF dataset `hdae/yomi-dict`
  and place it there (without it, the corrected-dictionary tests of models are SKIPped).
- `outputs/release/` **holds things that may be deleted next to things that cannot**, so do not delete it in bulk.
  The `upload-<repo>.log` (the run log of an upload) written by the script `tools/release/hf-upload.zsh` is on
  the **safe to delete** side, like `bench/`. Release-note drafts, archived copies of upstream cards and investigation notes placed by hand in the same directory
  **cannot be regenerated** (the same nature as `misc/`). When discarding, look file by file.
  The script holds this directory's spelling as a **hand copy** (zsh cannot read the constants of `paths.py`, so there is no `RELEASE_ROOT`),
  so a change has to fix two places at once: the script and this table.
- `outputs/diag/` is the directory for **investigation records** (investigation scripts, probes, raw data, logs), which ADRs and known-issues point to as
  evidence. A new investigation goes together under `<topic>-<date>/` (older scripts placed flat are not moved, to keep the paths the records point to).
  When discarding, look file by file (the same nature as `misc/`). **Scripts that are run repeatedly do not go here** — scripts run as part of a procedure go to
  the tracked `tools/diag/` (for example: the GPU busy check `gpu-busy.zsh`; the derivation of DeBERTa's collapse ceiling `deberta-collapse.ts` (removed in 0.15)).
- The old `outputs/series-archive/` (where ruled-on series were set aside) was **retired** by the clean-up ruling of 2026-08-30
  (the revival lever for base i4 is still carried by the `*-i4-dyn` series under `outputs/series/` —
  the NOTE in `anima/distribution.py`).

## Assembly (series → distribution)

```sh
cd tools/export-recipes
uv run python dist.py --model anima-turbo-v1.1 --model anima-v1.0 \
    --model anima-aesthetic-v1.1 --model anima-turbo-v1.0 \
    --model anima-aesthetic-v1.0 \
    --out ../../models/karume-anima                  # the five official variants (default = Turbo v1.1 — ADR 0087)
# Repositories that reference another one across repositories (the recipe declares which roles — ADR 0038 §7 addendum 2026-10-28)
# are built after the referenced mirror under models/, with one revision source: --resolve-refs hf (the referenced repository's
# main, every borrowed file checked on the Hub — the publication build), --ref-revision <name>=<SHA>, or --allow-placeholder-ref
# (a development mirror while the referenced repository is unpublished). Procedure: the release-runbook section on
# repositories with cross-repository references.
uv run python dist.py --pipeline anima-extra --model anima-wai-v1.0 --model anima-copycat-20260610 \
    --out ../../models/karume-anima-extra --resolve-refs hf
                                                     # → models/karume-anima-extra/ (fine-tunes; the text stack is referenced in karume-anima)
uv run python dist.py --pipeline umt5                # → models/karume-umt5-xxl/ (umT5-XXL encoder; i8 — ADR 0119)
uv run python dist.py --pipeline wan --allow-placeholder-ref
                                                     # → models/karume-wan2.1/ (text_encoder is referenced in karume-umt5-xxl; not published)
uv run python dist.py --pipeline wan-ti2v --resolve-refs hf
                                                     # → models/karume-wan2.2/ (Wan2.2 TI2V 5B; text_encoder is referenced in karume-umt5-xxl — ADR 0121)
uv run python dist.py --pipeline umt5 --intake ../../inputs/umt5/<name> [--allow-undeclared-license]
                                                     # → outputs/misc/local-dist/<name>/ (local experimental mirror; repository name umt5-xxl-<name>-local — ADR 0122)
uv run python dist.py --pipeline irodori             # → models/karume-irodori-v4-small/
uv run python dist.py --pipeline irodori \
    --model v4.1-small --resolve-refs hf             # → models/karume-irodori-v4.1-small/ (references 26 roles in karume-irodori-v4-small)
uv run python dist.py --pipeline sbv2 \
    --model F1 --model F2 --model M1 --model M2 --out ../../models/karume-sbv2-jvnv   # the pipeline declares the family-level repository name (karume-sbv2-jvnv) (2026-09-25)
uv run python dist.py --pipeline sbv2-fn \
    --model FN1 --model FN2 --model FN3 --model FN4 --model FN5 --model FN6 \
    --model FN7 --model FN8 --model FN9 --model FN10 --out ../../models/karume-sbv2-fn
uv run python dist.py --pipeline gemma4              # → models/karume-gemma4/ (about 4.0GiB)
uv run python dist.py --pipeline siglip2 \
    --model base --model so400m \
    --out ../../models/karume-siglip2                # one repository, two models (default base — ADR 0092 decision 8)
uv run python dist.py --pipeline birefnet \
    --model 1024 --model 2048 \
    --out ../../models/karume-birefnet-hr            # one repository, two models (model name = resolution; default 1024 — ADR 0092 decision 9)
uv run python dist.py --pipeline lucida \
    --model 1024 --model 2048 \
    --out ../../models/karume-lucida                 # a derivative = a separate repository; the same two models
uv run python dist.py --pipeline depth-anything      # → models/karume-depth-anything-v2/
uv run python dist.py --pipeline vowel-detector      # → models/karume-vowel-detector/
```

- **Not yet published for the first time: `vowel-detector`.** Of the three Wan repositories, `karume-umt5-xxl` and `karume-wan2.2` were first published with 0.14.0, and `karume-wan2.1` is not published (ADR [0118](decisions/0118-wan21-video-generation.md) addendum 2026-10-15)
  (siglip2 and depth-anything were first published with 0.9.0, birefnet / lucida on 2026-09-05 in the two-model form 1024² + 2048²).
  The two Wan repositories reference `karume-umt5-xxl` across repositories, so umT5 is published first (procedure: [release-runbook](release-runbook.md) "First publication of a family" — the runbook describes
  the 0.14.0 round of umT5 and Wan2.2).
  `karume-sbv2-fn` is a separate case whose mirror is not kept permanently (the bullet below). The canonical source for repository names and the rules for sharing a repository is ADR
  [0092](decisions/0092-distribution-repos-and-sources.md) decisions 1 / 2 / 8 / 9.
- Depending on the upstream license (siglip2 = Apache-2.0 / the birefnet line = MIT / depth-anything small =
  Apache-2.0), `LICENSE.md` / `NOTICE.md` are bundled at the repository root (decision 7 — a standing rule that applies
  regardless of publication state).

- `karume-gemma4` folds **three series** (the product container of `gemma4-e2b-product` (the PLE is an asset of the container),
  the MTP drafter container of `gemma4-e2b-drafter`, and the compiled assets of `gemma4-e2b-tokenizer`) into
  one repository. Only the tokenizer goes into the manifest's `assets` (details in
  `tools/export-recipes/gemma4/README.md`). The upstream is Apache 2.0, so
  `LICENSE.md` / `NOTICE.md` go at the repository root (the legal-text slot of `karume.dist`). The repository name is
  `karume-gemma4` by the one-repository-per-family rule (a repository that E4B / 12B share — ADR
  [0092](decisions/0092-distribution-repos-and-sources.md) decision 1)
  (renamed from `karume-gemma4-e2b` in 0.9.0 — HF redirects the old name). **The recipe's
  `repo_name` = the output directory name of the command above = the repo name in the card's Usage** (the latter two are
  derived from the former), so a rename takes one constant in the recipe. Where the published revision lives: the models table
  `gemma.SOURCES["gemma4"]` (the same ADR, decision 3 — not copied into docs).

- The mirror of `karume-sbv2-fn` is **not kept permanently** (ruling of 2026-08-30 — the e2e checks take `karume-sbv2-jvnv`, whose license statement is
  in order, as canonical). The command above is a record of how to regenerate it; whenever the series (exported from `inputs/sbv2/FN*`
  → `outputs/series/`) are present, it can be rebuilt at any time.

- The canonical spec is ADR [0041](decisions/0041-manifest-v2.md) (layout inside a repository = per-model
  subtrees + `shared/`; **placement is always an independent copy** — the reason hard links are forbidden is also in an addendum of that ADR) +
  ADR [0075](decisions/0075-quant-presentation.md) (quant `label` / `description`,
  `requiredLimits`, the cross-repository `repo` / `revision` of file references) + ADR
  [0109](decisions/0109-manifest-v5-container.md) (`weights.<component>.<dtype>.container`).
  **The current format is `karume/5`** (hub parses a single form = reads nothing but `karume/5` —
  `packages/hub/src/manifest.ts`).
  `karume.json` is derived from the actual files (hand-writing forbidden — ADR 0038).
- Assembly is idempotent (a rerun replaces). `verify_dist` goes as far as matching the declarations against the actual files and the undeclared-file check,
  and the model card `README.md` is generated mechanically from the verified manifest (attribution via
  `--card-profile` — see the exporter README).
- Deleting a series does not break a distribution (independent copies). Conversely, `dist.py` can regenerate a distribution from the series
  at any time. The core's own `karume dist` is designed to fail with an empty acceptance set (ADR 0065) — a family is
  always assembled by this repository's driver `dist.py`.

## Local-first in the CLIs

The Deno-era CLI examples this section describes were removed in 0.15 (`bb35fe2a` — to be rebuilt as browser applications); it records their behaviour as of 0.14.0.
When `--source` is omitted, `demo:anima` and `demo:irodori` prefer the standard mirrors above
(`models/karume-anima` / `models/karume-irodori-v4.1-small`).
Only when the mirror itself is missing do they fetch the verified published revision in the family's `SOURCES`.
If an existing directory has no `karume.json`, it is an error, as an incomplete distribution.
An explicit `--source` takes top priority, and the actual source is printed at startup.
Gemma 4 defaults to local as before; fetching from the public repository is requested with `--repo`.
SBV2 and vowel-detector name the distribution to use explicitly with `--source`.

The experimental CLIs for additional LLMs are MiniCPM5 (`examples/minicpm5/README.md` (removed in 0.15)) /
Qwen3 (`examples/qwen3/README.md` (removed in 0.15)). No public distribution exists yet, so they read the converted
series under `outputs/series/` and the official `tokenizer.json` under `inputs/` directly. They look for existing series in the order GPTQ i4 → i8 → RTN i4 → f16 → f32,
and when there are several experiments of the same kind they ask for `--source`. They do no new fetching and no re-conversion.

## Publishing (upload to HF)

**MUST: when uploading model files to HF, use `tools/.venv/bin/hf` (huggingface_hub 1.27 /
hf_xet 1.6.0) and always set the four env variables below in the same shell.**

```sh
export HF_XET_DEDUPLICATION_MIN_N_CHUNKS_PER_RANGE=1000000
export HF_XET_DEDUPLICATION_MIN_N_CHUNKS_PER_RANGE_HYSTERESIS_FACTOR=1.0
export HF_XET_DEDUPLICATION_NRANGES_IN_STREAMING_FRAGMENTATION_ESTIMATOR=1
export HF_XET_DEDUPLICATION_GLOBAL_DEDUP_QUERY_ENABLED=false
```

The script `tools/release/hf-upload.zsh upload <repo>` does these env variables, setting aside the shard-cache, running the venv's hf and
printing the fragmentation table right after, in one go ([release-runbook](release-runbook.md) §2).
Without them, Xet's chunk-level dedup works too well, **the reconstruction becomes fragmented and downloads become several times slower**.
The fourth one stops the chunk query against the whole CAS (global dedup) — without it, the upload hits xorbs of
other repositories you never uploaded, and **even a first upload gets fragmented** (measured with siglip2 on 2026-09-04).
Mechanism and measurements:
[research/2026-08-09-xet-fragmentation.md](research/2026-08-09-xet-fragmentation.md).

- **Always verify** after uploading — for every krm (and the asset safetensors), count the terms of the reconstruction and
  check that `MiB/range` does not fall below 10 (procedure in §9 of the same document; a few sample files
  miss the skew between files). When healthy, it comes close to 1 xorb = 1 term.
- Before an upload, set aside `~/.cache/huggingface/xet/*/shard-cache` **every time** (even on a first upload,
  shards fetched by a global-dedup hit remain, and the next upload dedup-hits them).

**Which env variables take effect depends on the hf_xet version.**

- The fourth, `GLOBAL_DEDUP_QUERY_ENABLED`, **does not exist in hf_xet 1.4.3 (nix's hf) and exists in 1.6.0
  (the hf in tools/.venv)**. That it took effect can be confirmed by the line
  `global_dedup_query_enabled = false (user set)` in the hf_xet log. With 1.4.3, exporting an env variable that does not exist
  does nothing, so do not upload with nix's hf.
- **The canonical source for the recovery procedure after fragmentation and for what is forbidden on repositories with a published pin is
  [release-runbook](release-runbook.md) §2 "Fragmentation check right after the upload"**. What this section holds as canonical
  is only the four env variables and the hf_xet version condition; it does not duplicate whether recovery is possible, the two-commit method or how deleting a repository is handled.
