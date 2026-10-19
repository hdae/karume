# Release runbook (JSR + HF model publication)

> Nature: **the canonical source for the standing procedure**. A checklist to follow from the top at every release and
> model publication. The canonical sources for mechanisms and numbers live elsewhere (fragmentation =
> [research/2026-08-09-xet-fragmentation.md](research/2026-08-09-xet-fragmentation.md);
> env = the publication section of [assets-layout.md](assets-layout.md); pin =
> ADR [0073](decisions/0073-models-source-pin.md); publish mechanism =
> `.github/workflows/publish.yml`). When the procedure disagrees with reality, **fix this document first**.

## 0. Ordering invariant

**version bump → rebuild of the distributions → HF upload → baking the pins → JSR publish**, in this order
(non-commutative, MUST).

- **Rebuild after the bump**: the manifest's `generator` field copies the package version (`karume/<version>`),
  so a distribution built before the bump lands on HF still claiming the old version. Local rebuilds during
  development may keep the old label (the e2e tests read the bytes, not `generator`), but **the one build that
  is published is rebuilt after the bump**. The order of the sections below groups the work; it is not a timeline —
  **only the lockstep bump commit of §4 and the venv sync right after it come before §1 to §3** (the rest of §4 =
  moving the CHANGELOG into the version section / push / Release comes last).
- The pin SHA does not exist until the upload is done (ADR 0073 — the models mapping table
  `<FAMILY>_SOURCES` is fixed to "the latest commit at publication time". The table shape is ADR
  [0092](decisions/0092-distribution-repos-and-sources.md) decision 3).
- When the manifest format goes up, old JSR clients cannot read the new repositories (hub parses a single format —
  ADR 0041/0071). Putting the JSR publish last means the release goes out in the state "assets that the new hub can read are
  already on HF".

### Publication order for repositories with cross-repository references (MUST)

A cross-repository component reference (ADR [0038](decisions/0038-manifest-v1.md) §7 addendum) **bakes in the
commit SHA of the referenced repository**, so it cannot be built until the referenced repository has been published. There are two pairs today —
the pair in which extra (fine-tuned models) references the text stack of the official repository (ADR 0087 — the same shape as the former
turbo → anima direction), and the pair in which `karume-wan2.2` references the text_encoder of `karume-umt5-xxl` (ADR
[0119](decisions/0119-wan-umt5-gpu-text-encoder.md) addendum "design of step 10d" and ADR
[0121](decisions/0121-wan22-ti2v-5b.md) decision 9 — the procedure is the umT5 and Wan2.2 items of "First publication of a family" below).
`karume-wan2.1` has the same reference but is not published on HF (the maintainer's ruling 2026-10-15). For the anima pair:

1. **Upload `karume-anima` first** (with the fragmentation countermeasures of §2; the 5 official variants (`anima-turbo-v1.1` (default) /
   `anima-v1.0` / `anima-aesthetic-v1.1` / `anima-turbo-v1.0` / `anima-aesthetic-v1.0`) — ADR 0087)
2. **Record the commit SHA of its main** (taken the same way as in §3)
3. Pass that SHA and **build `karume-anima-extra` with the cross-repository reference** — the 5 options of
   `tools/export-recipes/dist.py` (`--ref-repo` / `--ref-revision` / `--ref-dist` / `--ref-model` / `--ref-role`) come
   **either all together or not at all**; a partial set fails.
   **extra holds 2 models, so build the one repository in one run with `--model anima-wai-v1.0 --model anima-copycat-20260610`**
   (`--model` repeated = family assembly; the first one is `defaultModel`).
   `--ref-role` is one of the 6 `text_encoder` / `vae_decoder` / `text_encoder_bf16` / `vae_decoder_bf16` / `tokenizer` /
   `tokenizer_2` (the role name is the artifact key, and the bf16 text stack is a separate role — bf16 was extended to
   every model on 2026-10-17. `text_conditioner` is not referenced across repositories because both extra models have their own — `own_text_conditioner=True`
   in `distribution.py`; every model has its own bf16).
   **The repo name in the card's Usage example is derived from the pipeline's declaration (`Pipeline.repo_name`); the
   directory name of `--out` is not looked at** (the real damage of 0.5.0, where the staging name with `-release` was copied into the public card, disappeared
   structurally with this derivation). For extra the declared `karume-anima-extra` appears as is, so `--repo` is not needed.
   `--repo` is the way to override the declaration, and its argument has the form `OWNER/NAME` (e.g. `hdae/karume-anima-extra` — passing
   only the name fails the form check). With `--repo`, `--out` is required too. For a series that bundles several models into one repository and whose
   per-model declarations disagree, `--repo` is required, and without it the build fails before assembly — today that is the 4 JVNV voices of `sbv2`
   (the declaration is `karume-sbv2-<speaker>` per speaker, so `--repo hdae/karume-sbv2-jvnv` — confirmed with `resolve_repo` on
   2026-09-24).
   Put `--out` at `models/<repo>` (the same name as the repository) — the script of §2 uploads `models/<repo>` as is
4. **Upload `karume-anima-extra`**
   - [ ] After generating the extra mirror, **bring back the cross-repository real-asset check** — put the entries back into
         `CROSS_REPO_MIRRORS` of `packages/e2e/tests/e2e_anima_test.ts` and `MIRRORS` of
         `packages/e2e/tests/e2e_*_fusion_counts_test.ts`, and freeze the fusion counts and reference sha of the extra variants anew
         (the check that was emptied once when the official repository became self-contained — the table and the distribution mechanism are kept)

If the referenced repository is uploaded again later, its SHA changes, and the extra manifest **keeps pointing at the old revision**
(the bytes are protected by the double pin, so nothing wrong is served, but the two repositories' contents become different generations) —
**when the referenced repository is uploaded again, rebuild and re-upload extra too** (the same for Wan — when `karume-umt5-xxl` is uploaded again,
rebuild and re-upload `karume-wan2.2` too).

**The placeholder-SHA check**: a development mirror for the time the referenced repository is unpublished is built by passing a placeholder SHA (40 zeros) to
`--ref-revision` and passing `--allow-placeholder-ref` **explicitly** (`PLACEHOLDER_REVISION` in `tools/export-recipes/dist.py`
— the local source is resolved by the `crossRepo` mapping and does not look at the revision). The placeholder SHA satisfies the form of 40 hex
digits, so it passes hub's parse and `verify_dist` too — the publication rebuild does **not** pass
`--allow-placeholder-ref`. Without it the driver fails before writing a single byte, so the build cannot happen until the real SHA of the
referenced repository's main is passed. The rebuild after the bump of §4 is always done (do not upload the development mirror as is —
right before uploading, confirm that `rg -c '"revision": "0{40}"' models/<repo>/karume.json` prints nothing (no placeholder SHA
remains)).

NOTE (for the next release only): the old `hdae/karume-anima-turbo` was retired by ADR 0087 — do not upload it again.
How to handle the published repository (deprecation notice, replacing the README, etc.) is the maintainer's ruling at upload time.

## 1. Pre-release checks

- [ ] `vp check && vp test` green (full verify = every lane with the GPU tests — a SKIP on an environment without an adapter is not allowed for the release decision;
      ADR 0005). **Conditions for the release machine**: real hardware whose adapter
      **enumerates** `shader-f16` and `timestamp-query` (a software adapter (lavapipe etc.) computes f16 in f32,
      so it is kept apart from the real-hardware lane), with the real-weight series (`outputs/series/`) in place.
      Passing verify on a machine missing the enumeration, assets, reference values or distribution mirrors needs the env variables that declare that intent
      (`KARUME_ALLOW_NO_SHADER_F16` / `KARUME_ALLOW_NO_TIMESTAMP_QUERY` /
      `KARUME_ALLOW_NO_ASSETS` / `KARUME_ALLOW_NO_REFERENCE` / `KARUME_ALLOW_NO_DISTRIBUTION`), but
      **a green run in an environment with any of them set is not used for the release decision** (by default `gpu_gate_test.ts` /
      `e2e_assets_gate_test.ts` / the reference guard of each series / `distribution_gate_test.ts` FAIL it)
- [ ] **No recording mode is set**: `KARUME_REFERENCE` (`write` / `rewrite`) and
      `KARUME_SURFACE=write` rewrite the expected values from measurements; they are not checks. In particular the reference guard's green condition is
      "green even with 0 rows in a recording mode" (`packages/runtime/tests/helpers/reference.ts`), so
      leaving `KARUME_REFERENCE` on the release machine makes **every sha256 check green as
      `written` without comparing anything**. Only a green run with both unset is used for the release decision
- [ ] In a release that changes the WGSL or the selection condition of the subgroup variants (`rmsNormReduce: "subgroup32"` / `linearGemvReduce: "parallel-subgroup32"`),
      run the round-trip comparisons of Chrome's `/check.html` (GEMV) and of RMS.
      In `vp test` this real-run check is always SKIP, and there is no check that turns the SKIP red (the
      "The real-execution checks of the subgroup variants do not run automatically" section of limitations — the procedure is `tools/llm-speed/browser/README.md` of 0.14.0 (main).
      Use that until the browser tools are rebuilt — ADR 0127)
- [ ] exporter: `uv run --no-sync pytest` + `uv run --no-sync ruff check` +
      `uv run --no-sync ruff format --check` green
      **in both tools/exporter and tools/export-recipes** (CI runs lint and format as
      separate steps — a format failure slipped through once because only check was looked at, 2026-08-21)
- [ ] If there was a change that needs the distributions regenerated (manifest format, quant presets, card, default quant),
      regenerate the target repositories under `models/` with dist, and each family's check (WAV / PNG / verify_dist) is green.
      **The publication rebuild comes after the bump commit of §4** (the order of §0 — `generator` copies the package
      version). The canonical source for the local e2e SBV2 check is the public mirror `karume-sbv2-jvnv` (switched on 2026-08-30 from
      the private fn — the fn mirror is not kept around; when needed it is regenerated from
      `inputs/sbv2/FN*` with the dist command of assets-layout)
- [ ] License check: each family's provenance record
      (`tools/export-recipes/<family>/THIRD_PARTY_NOTICES.md`) and the
      attribution in the generated card of the distribution repository match the actual upstream (a per-revision human check — release gate. Canonical source =
      the release section of backlog. Only `LICENSE.md` / `NOTICE.md` may sit at the top of a distribution repository —
      `LEGAL_PATHS` of `verify_dist`)
- [ ] git: everything committed; the maintainer pushes

## 2. HF upload (fragmentation countermeasures — MUST)

**Background**: uploading without the env variables makes Xet's chunk dedup work too well, the reconstruction fragments, and downloads become
5 to 6 times slower. The dedup includes **global dedup** (a chunk lookup across the whole CAS — it also hits xorbs
of others that you did not upload), and unless it is stopped, **even a first upload** fragments.

- [ ] **Pin the `hf` to use**: `tools/.venv/bin/hf` (huggingface_hub 1.27 / hf_xet 1.6.0).
      **Do not use nix's `hf` (hf_xet 1.4.3)** — the knob that stops global dedup (the fourth below) does not exist
      in that version, and there is no way to stop the fragmentation

```sh
export HF_XET_DEDUPLICATION_MIN_N_CHUNKS_PER_RANGE=1000000
export HF_XET_DEDUPLICATION_MIN_N_CHUNKS_PER_RANGE_HYSTERESIS_FACTOR=1.0
export HF_XET_DEDUPLICATION_NRANGES_IN_STREAMING_FRAGMENTATION_ESTIMATOR=1
export HF_XET_DEDUPLICATION_GLOBAL_DEDUP_QUERY_ENABLED=false
```

- [ ] Export the 4 env variables above **in the same shell** and then run `hf upload`
      (canonical source: the publication section of [assets-layout.md](assets-layout.md)). **The script
      `tools/release/hf-upload.zsh upload <repo>` does these env variables, moving the shard-cache aside, the venv's hf and
      the fragmentation table right after, all in one run** — type it by hand only when the script cannot be used. The table also shows
      `### FAILED` rows (a part whose xet hash or reconstruction could not be read — the script exits non-zero. Fix the cause and then
      upload again) and `### SKIP` rows (small files stored outside xet) (2026-09-25). That the fourth
      was read is confirmed by the line `global_dedup_query_enabled = false (user set)` in the hf_xet log
      (the script copies this line and the number of CAS lookups into its log).
      **NOTE (hf_xet version difference)**: the fourth, `GLOBAL_DEDUP_QUERY_ENABLED`,
      **does not exist in 1.4.3 and exists in 1.6.0** (confirmed on 2026-09-04 from the list of env names in the binary).
      The form of setting `HF_XET_MIN_SPACING_BETWEEN_GLOBAL_DEDUP_QUERIES`, named as the successor in 1.4.3, to a huge value
      **does not work** (measured 2026-08-29). Furthermore, in 1.4.3 **the in-repository dedup when the repository's own history holds identical
      chunks cannot be stopped by any knob** (measured the same day on anima's text_encoder.
      It did not reproduce in 0.8.0 after the shard split — the results are in the 0.8.0 section of
      [the archived, processed sections](research/2026-09-22-backlog-archive-0.5.0-to-0.12.0.md)).
- [ ] **Move `~/.cache/huggingface/xet/*/shard-cache` aside every time** (not only for re-uploads but
      **for first uploads too**). Shards fetched from the server by a global-dedup hit stay here, and
      the next upload picks them up and inherits the fragmentation
- [ ] **Switch to the write token**: `tools/.venv/bin/hf auth switch --token-name "Karume Release"` —
      with the default read token (Karume Gated Read) the LFS batch returns 403
      (measured 2026-08-21). **When the upload is done, switch back to the read token**:
      `tools/.venv/bin/hf auth switch --token-name "Karume Gated Read"` → check the selected token with `tools/.venv/bin/hf auth list`.
      NOTE: in practice this switch-back is not kept (as of 2026-09-24, Karume Release had stayed selected since the
      upload of 09-23). With the write token in everyday use, a mistake even in read-only work
      becomes a write to HF as is
- [ ] **The script is also the machine check** (it does not rely on a human looking): `upload` exits non-zero as soon as
      moving the shard-cache aside fails, and after the upload it checks `global_dedup_query_enabled = false` and
      0 CAS lookups and exits non-zero on a mismatch. `check` fails non-zero when `models/<repo>` has
      not a single `.krm` / `.safetensors` (it does not print an empty table as "checked, no problem")
- [ ] **The pre-publication licence check (licence marker — ADR [0122](decisions/0122-umt5-upstream-and-compatible-encoders.md) decision 6)**:
      before moving the shard-cache aside and before starting hf, `upload` reads, with `tools/release/container_license.ts` (needs Node),
      `provenance.license` of every container (part 0 of `.krm`) in `models/<repo>`, and if any of these is present — an undeclared-licence marker
      (`NOASSERTION`, and values such as `unknown` that do not identify the redistribution terms — case-insensitive), an unreadable part 0, or a
      directory without a single container — it prints `### FAILED pre-publication check` and exits non-zero without uploading a single byte.
      It closes on the containers' content rather than on the location, so a local experimental mirror copied by hand under `models/`
      (from intake — assets-layout) is stopped too. **An `hf upload` that bypasses the script does not pass this check** — when typing it by hand,
      run `node tools/release/container_license.ts models/<repo>` first and confirm the exit code is
      0
- [ ] Upload: `tools/release/hf-upload.zsh upload <repo>` (inside, it is
      `tools/.venv/bin/hf upload hdae/<repo> models/<repo> . --repo-type model` —
      in `models/`, 1 directory = 1 HF repository — assets-layout. Extra arguments are passed to hf as they are)
- [ ] **When the distribution changed (`karume/4` → `karume/5` etc.), delete the old-format files that the new manifest does not reference
      in the same upload**: `tools/release/hf-upload.zsh upload <repo> --delete '*.safetensors'`.
      `hf upload` does not delete files that are not present locally, so without it the old-format files stay on main.
      The glob crosses `/` and also matches nested paths. Paths re-uploaded in the same call are excluded from the deletion, so
      asset safetensors such as sbv2's `speakers/` / `styles/` stay. The deletion goes into the first commit of the upload,
      and a large repository is split into several commits (`(part N)` — the commit-split number of huggingface_hub, not the same thing as
      a container part). All of this is the behaviour of the implementation of `upload_folder` in huggingface_hub 1.27, not a guarantee of the spec
      — check again after upgrading
- [ ] **Right before uploading, compare HF's main with the local mirror and add the paths that exist only on HF to `--delete`**:
      take the difference between the `path` list of `type: "file"` entries of
      `curl -sS "https://huggingface.co/api/models/hdae/<repo>/tree/main?recursive=true"` and the list of files under `models/<repo>`. `'*.safetensors'` alone leaves
      old-format files other than safetensors behind (e.g. the old PLE index `e2b/ple/ple.json` of `karume-gemma4`). `--delete` can be
      repeated — e.g. `tools/release/hf-upload.zsh upload karume-gemma4 --delete '*.safetensors'
      --delete 'e2b/ple/*'`. NOTE (speculation): the tree API may return a large listing in pages (huggingface_hub's
      `list_repo_tree` follows the same endpoint through the `Link` header), so one curl may miss part of the listing
      — for repositories with many files, use `HfApi().list_repo_files("hdae/<repo>")` through `tools/.venv/bin/python -c`.
      HF's default file `.gitattributes` shows up in the difference but is not deleted. After the upload, do the same
      comparison once more and confirm that the difference is only `.gitattributes`
- [ ] **Bake the pin on main after the deletion**: take the SHA of §3 from main after all commits of the upload are done
      (the `### main sha` line of the script's log). Splitting the deletion into a separate round moves main, and the pin has to be repointed twice
      (example: irodori-v4.1-small deleted 65 old safetensors afterwards and repointed the pin — `1c952948`).
      Files of old revisions stay in HF's history, and published older package versions keep reading from the old pin
- [ ] **Renaming a repository (only in the release where it applies)**: rename with `tools/.venv/bin/hf repos move <old> <new>`. The old name
      redirects to the new name with **HTTP 307** for both API and resolve, so published references do not break, but
      **confirm that the redirect is alive by actually hitting it** —
      it is fine if `curl -sS -o /dev/null -w '%{http_code} %{redirect_url}' "https://huggingface.co/api/models/<old>"`
      returns `307` and the URL of the new name.
      After the rename, the model card and `karume.json` are rebuilt and uploaded again (the repo name in the card's Usage
      would otherwise stay the old name), so **the revision moves** — update the mapping table of §3 with the new name + the new SHA

### Fragmentation check right after the upload (MUST)

**Tabulate every krm / safetensors** (a sample of 2 or 3 representatives is not enough — on 2026-09-04 siglip2 had only 5 of
the 7 so400m shards fragmented, which a sample misses depending on luck).
`tools/release/hf-upload.zsh upload` prints the table at the end. To look at a published repository later, use
`tools/release/hf-upload.zsh check <repo>`. The content of the table is exactly the reproduction procedure of research §9:

```sh
curl -sS -I "https://huggingface.co/<repo>/resolve/main/<path>" | grep -i x-xet-hash
curl -sS "https://huggingface.co/api/models/<repo>/xet-read-token/main"   # -> casUrl / accessToken
curl -sS -H "Authorization: Bearer <accessToken>" "<casUrl>/v1/reconstructions/<x-xet-hash>"
```

- [ ] Count `terms[]`; **MiB per range ≥ 10 as a guideline** (when healthy, close to 1 xorb = 1 term).
      The table prints `ranges=` next to `terms=` (the total number of fetch ranges in the response's `fetch_info` = the number of ranges actually fetched).
      A term that points at the same range many times, like an all-zero chunk, raises only terms, so ranges tells apart whether many terms come from repeated
      references or from fragmented fetch ranges (the guideline and the verdict stay on terms).
      Far below it means fragmentation — **with hf_xet 1.6.0 + the 4 env variables above it can be recovered**.
      Procedure = move the shard-cache aside → **delete and re-create the repository** → upload the same bytes again. CAS lookups
      become 0 and xorbs are written anew, so it returns to the healthy form (2026-09-04, siglip2 recovered to 46 to 61 MiB/term
      — the 2026-09-04 addendum of the [research](research/2026-08-09-xet-fragmentation.md)).
      The two-commit method of delete → re-upload within the same repository **does not work** (measured 2026-09-05 — the same bytes are not transferred and keep
      referencing the original xorbs). In a repository holding 2 resolutions of the same checkpoint, even delete → re-create left one shard
      on the 2048 side unrecovered ([research 2026-08-09, 2026-09-05 addendum](research/2026-08-09-xet-fragmentation.md)).
      **MUST NOT: do not use delete → re-create on a repository with a published pin** (ADR 0108 decision 18 — the HF repository's history is kept, and
      published older package versions keep reading from the pin of the old revision). The view that deleting the repository makes the old pins 404
      is **speculation** (not confirmed on the real service). When a repository with a published pin fragments, do not delete it;
      stop and ask for a ruling.
      **With hf_xet 1.4.3 there is no way to recover** (a one-way ratchet — re-uploading the same bytes is skipped by the hf CLI together with
      the transfer, and the two-commit method also misfires. Measured 2026-08-29). Keep the observed values in the record of §5

## 3. Baking the pins (writing into the mapping tables — ADR 0073 / 0092)

They live in the per-family mapping tables `<FAMILY>_SOURCES` of `packages/models` (key = the basename of the HF repository name without
`karume-`) and in `KARUME_SOURCES`, into which the barrel folds every family (ADR
[0092](decisions/0092-distribution-repos-and-sources.md) decision 3).

- [ ] Get the SHA (40 hex) of main of each published repository:
      the `sha` field of `curl -sS "https://huggingface.co/api/models/<owner>/<repo>/revision/main"`
- [ ] **Write it into the `revision` of the matching key** for each uploaded repository
      (e.g. `IRODORI_SOURCES["irodori-v4.1-small"].revision`). The key follows mechanically from the repository name,
      so the repository name determines where to write
- [ ] **Confirm that `KARUME_SOURCES` is complete**: every repository uploaded this time is in the table.
      The invariants (`"karume-" + key === repo basename`, owner `hdae`, `revision` is 40 hex) are
      checked by tests, so `vp test --project core` fails on them
- [ ] **Create the entry of a newly published repository at this point** (inheriting the reason of ADR 0073 decision 1 — placing it before publication
      grows a key on the public surface that can only 404). The procedure is just adding 1 entry to `<FAMILY>_SOURCES` in each family's
      `config.ts`; the barrel re-exports per table, so nothing else needs to follow
- [ ] Updating the pins is **a duty at every bump** (the mapping table = "the sources **verified** by this package version" —
      inheriting the maintenance duty of the ADR 0073 addendum of 2026-08-25). In addition to the connectivity check below, when presets or the default quant moved,
      pass the behaviour tests too before claiming "verified"
- [ ] Connectivity: `fromPretrained` with the written SHA passes against the real URL (with a SHA, no revision-resolution
      request happens = offline start-up is possible — ADR 0038). **The connectivity check across every entry of the table is
      `node tools/published-smoke/main.ts` of §5** — it walks the whole `KARUME_SOURCES` with the published `@karume/hub`,
      so it can only be run after the JSR publish (the order of §0)
- [ ] **The values of the mapping tables are the only place where the published revisions live**: check with the equivalent of `rg '\b[0-9a-f]{8,40}\b' docs
      .claude/ACTIVE_DESIGN.md` that no SHA was copied into the docs (copies go stale at every
      rebuild — records are spelled with keys like `ANIMA_SOURCES["anima"]`. To prevent a repeat of the 2026-08-24 / 08-29
      mismatch in two releases in a row)
- [ ] `vp check && vp test` → commit

### First publication of a family (creating the mapping table itself — the most recent case: 0.14.0's umT5 = `karume-umt5-xxl` and Wan2.2 = `karume-wan2.2`)

A family without a public repository has no mapping table (ADR 0073 decision 1 — the same as `vowel-detector`). In the release that
creates the table, in addition to "add 1 entry" above, put the checks that bind the table's completeness in the same commit (forgetting one of them
makes that check fail, or goes green while the table is not on the public surface). In 0.14.0, 2 families were published for the first time at once:
umT5 (TS entry `Umt5Encoder` — `@karume/models/umt5`) got `UMT5_SOURCES`, and Wan got `WAN_SOURCES`.

Wan2.1 (`karume-wan2.1`) is not published on HF (the maintainer's ruling 2026-10-15 — ADR
[0118](decisions/0118-wan21-video-generation.md) addendum (2026-10-15)). The code stays, and the distribution is assembled locally
with `dist.py --pipeline wan` (the same 5 `--ref-*` options as Wan2.2 below — before umT5 is published, the placeholder SHA of §0 and
`--allow-placeholder-ref`; after it is published, the SHA of its main. `--out` is the default `models/karume-wan2.1`). Do not add `"wan2.1"`
to `WAN_SOURCES`, and do not upload `karume-wan2.1`.

- [ ] **Check before publication (Wan2.2)**: the 50-step run (`KARUME_WAN_TI2V_FULL_PIPELINE=1 vp test
      packages/e2e/tests/e2e_wan_ti2v_pipeline_test.ts -t "50 ステップ"` —
      the precomputed path, 1280×704, 33 frames, one each with the reference preset and the practical preset. About 35 minutes on an RTX 3080 Ti) is green, and
      the sha256 environment rows match (the default lane runs only 2 steps). The mirror `models/karume-wan2.2` is needed
- [ ] **Build the umT5 series from the provenance moved to the original** (ADR [0122](decisions/0122-umt5-upstream-and-compatible-encoders.md)
      decisions 1 / 4, addendum "results of step a"): the containers of `karume-umt5-xxl` claim the commit of the original `google/umt5-xxl` as upstream, and the series is
      `outputs/series/umt5-xxl-i8-dyn`. Do not upload a mirror built from the old series `wan2.1-umt5-i8-dyn` (which claims Wan's `text_encoder`).
      On a machine that rewrites the series, first fetch the original's shards at the pinned revision:
      `uv run --group wan python -m wan.sources --umt5 xxl --fetch` (about 29.8 GB, into HF's default cache — the shards 1 to 3 that the encoder needs
      are checked against the sha256 of the table before reading). NOTE: the original's index (`pytorch_model.bin.index.json`) and
      `config.json` are not pinned by sha256 (filed in backlog)
- [ ] **Build the umT5 tokenizer asset** (ADR 0122 addendum (2026-10-15)): `karume-umt5-xxl` has the asset
      `tokenizer` (format `karume-umt5-tokenizer/1`) of the model `xxl`. On a machine without the series location `outputs/series/umt5-xxl-tokenizer/tokenizer.json`,
      run `uv run --group wan --inexact python -m wan.umt5_xxl_tokenizer` in `tools/export-recipes` (`--fetch` if the pinned
      tokenizer files are not in the HF cache). The same run also writes the git-tracked fixture
      `packages/models/tests/fixtures/umt5/tokenizer-cases.json`, so run
      `vp fmt packages/models/tests/fixtures/umt5/` afterwards. If the file is missing or its format or provenance pin does not match, the next
      `dist.py --pipeline umt5` fails (`wan.umt5_distribution.assert_umt5_tokenizer`)
- [ ] **Upload the umT5 repository first (the cross-repository reference order of §0)**: in `tools/export-recipes`,
      `uv run python dist.py --pipeline umt5` (`--out` is the default `models/karume-umt5-xxl`) → with the script of §2,
      `tools/release/hf-upload.zsh upload karume-umt5-xxl` → record the SHA of its main the same way as in §3
- [ ] **Rebuild Wan2.2 with that real SHA**: `uv run python dist.py --pipeline wan-ti2v --ref-repo hdae/karume-umt5-xxl
      --ref-revision <SHA of main of karume-umt5-xxl> --ref-dist ../../models/karume-umt5-xxl --ref-model xxl
      --ref-role text_encoder` (do **not** pass `--allow-placeholder-ref` — do not upload with the development mirror's placeholder SHA.
      `--out` is the default `models/karume-wan2.2`) → after confirming that every text_encoder part in `karume.json` points
      at that SHA, `tools/release/hf-upload.zsh upload karume-wan2.2`
- [ ] Add the tables (key = the basename of the repository name without `karume-`, value = `repo` + the SHA of main):
      add `UMT5_SOURCES` (key `"umt5-xxl"` → `hdae/karume-umt5-xxl`) to `packages/models/src/umt5/config.ts` and
      export it from `umt5.ts` and `mod.ts`; add `WAN_SOURCES` (key `"wan2.2"` →
      `hdae/karume-wan2.2` — read by `WanTi2vPipeline`) to `packages/models/src/wan/config.ts` and export it from `wan.ts` and `mod.ts`. NOTE: the TS
      `UMT5_SOURCES` is the mapping table of the public distribution repositories, not the same thing as the recipe's `wan.sources.UMT5_SOURCES` (the table of the original's upstream pins)
- [ ] Fold both tables into `KARUME_SOURCES` in `packages/models/src/sources.ts`
- [ ] Add 2 rows to `FAMILIES` in `packages/models/tests/sources_test.ts` (`load` is `Umt5Encoder.fromPretrained` and
      `WanTi2vPipeline.fromPretrained`) and update the named counts of families and repositories (7 families, 10 repositories → 9 families, 12 repositories)
- [ ] Add both tables to the named family table (the list of `*_SOURCES`) in `packages/models/tests/models_barrel_surface_test.ts`
- [ ] Add `("umt5", ["xxl"], None, "karume-umt5-xxl")` and `("wan-ti2v", ["ti2v-5b"], None, "karume-wan2.2")` to
      `RELEASE_REPOSITORIES` in `tools/export-recipes/tests/test_dist_driver.py`
      (the check that requires the same set of repositories as the TS tables — it reads the `repo: "hdae/…",` lines of `packages/models/src/*/config.ts`)
- [ ] Pass the hint for an omitted ref (the third argument of `toManifestSource`): `UMT5_SOURCES["umt5-xxl"] (@karume/models/umt5)` to
      `Umt5Encoder.fromPretrained` (`packages/models/src/umt5/encoder.ts`), and `WAN_SOURCES["wan2.2"] (@karume/models/wan)` to
      `WanTi2vPipeline.fromPretrained`. Wan's call shares the single `loadWanFromPretrained` of
      `packages/models/src/wan/family.ts` with `WanPipeline` (2.1), so
      shape it so that 2.1 gets no hint (the table has no 2.1 key)
- [ ] Fix the docs that say "there is no table yet" or "unpublished, no pin constant either" (pick them up with `rg -n 'UMT5_SOURCES|WAN_SOURCES|未公開|not
      published' packages/models examples/wan`). Keep the statements that Wan2.1 is unpublished
- [ ] In the note of `DISTRIBUTIONS` in `packages/runtime/tests/distribution_gate_test.ts`, move `karume-umt5-xxl` and
      `karume-wan2.2` to "published" (the mirrors themselves were on the all-skip guard before publication. `karume-wan2.1`
      stays unpublished and the e2e tests use it as their root, so it stays on the list)
- [ ] Rewrite the public-surface snapshot (`KARUME_SURFACE=write`) and confirm that the difference is one line each of `UMT5_SOURCES` and `WAN_SOURCES`
      (the barrel and `./umt5` / `./wan`)

## 4. JSR publish

Mechanism: **publishing a GitHub Release fires `publish.yml`**, and `deno publish` at the workspace root
publishes the 3 packages (hub / runtime → models) together in dependency order
(already-published versions are skipped idempotently; OIDC, tokenless).

NOTE: the package README / LICENSE are part of what is published (`packages/*/README.md` and
`packages/*/LICENSE` — confirmed on 2026-09-20 that they appear for all 3 packages in the output of `deno publish --dry-run --allow-dirty`.
The `publish` of each `deno.json` is only `exclude: ["tests/"]`; no explicit `include` is
needed).

- [ ] Lockstep bump commit: the `version` of the 3 JSR packages + exporter, the `^`
      dependencies such as models → hub, the specifiers in `deno.lock`, and **regenerating `tools/uv.lock` (`uv lock` — MUST. CI
      checks freshness with `uv run --locked`, so missing it turns both the exporter and recipes jobs red.
      Example: `e15e271` of 0.4.3)**, all in one commit (precedents: `d65535c` / `7d97dd4`).
      `CHANGELOG.md` is not part of this commit (the item below — the bump is cut before §1 to §3, so
      changes between the rebuild and the upload also keep going into `[Unreleased]`)
- [ ] **Sync the venv right after the bump (MUST)**: `(cd tools && uv sync --all-groups)`. The manifest's
      `generator` field copies neither pyproject nor uv.lock but **the version of the dist-info of `karume` installed in the venv**
      (`karume.dist.generator_tag()` = `importlib.metadata.version`), so continuing with `uv run` under `--no-sync` after only
      `uv lock` builds with the old version. Machine check = the `<version>` in the last line
      `[dist] <out> — karume/<version> / ...` of each rebuilt `dist.py` is the version after the bump
- [ ] **In a separate commit right before the release**, move `[Unreleased]` of `CHANGELOG.md` into the section of that version, and put the
      date (the same `YYYY-MM-DD` as `git log -1 --format=%cs` of that commit) in the heading and a link definition for the version.
      Place it after the pin commit of §3 and before the push (the substance of "breaking changes are stated in the
      CHANGELOG" of ADR [0008](decisions/0008-public-api.md) — if the commit that the Release is cut from has no version section, it disagrees with the content of the published tag)
- [ ] push (the maintainer) → confirm CI is green (`ci.yml` does not run the publish dry-run today — the publishing dry-runs return once the 0.15.0 registry decision (JSR / npm) is made)
- [ ] Create a GitHub Release from the main commit with green CI → published (fires)
- [ ] Confirm the new version of the 3 packages on JSR

PyPI `karume` (tools/exporter) is **unreleased**. Add the procedure to this section when publication starts.

## 5. After the release

- [ ] Docs sync: ACTIVE_DESIGN's Now, backlog (the now heading and the progress of the release section),
      the release record, the project memory. **The release is not "done" until this item is done** —
      treat this one place as the only check for syncing the ledgers (in 0.9.0 only backlog was updated and
      ACTIVE_DESIGN stayed at "the maintainer publishes". The second miss of the same shape)
- [ ] Connectivity from the published packages: `node tools/published-smoke/main.ts` (tools/published-smoke — the published versions.
      Import from Deno is checked with `deno run -A tools/published-smoke/main.ts` (ADR 0127). Write the published version into
      `tools/published-smoke/package.json` and `deno.json` and run `pnpm install` before running it.
      With `@karume/hub` it resolves the manifest of every entry of `KARUME_SOURCES` and takes sbv2 through
      `fromPretrained`. On a machine without a
      GPU, `--manifests-only`). `jsr:@karume/*` under the workspace resolves to the local members,
      so always run the connectivity check with this task (it pulls the registry with its own deno.json).
      **Freshly published dependencies get caught by `minimumDependencyAge`** — `jsr:@karume/*` and `jsr:@hdae/fetch-cache`, which hub pulls in,
      are listed in the `exclude` of `tools/published-smoke/deno.json` (the example of 0.11.0, where fetch-cache 0.7.0, published one day earlier,
      was stopped — 2026-09-06). When a new `@hdae/*` dependency is added,
      add it here too
- [ ] Record one line of the fragmentation check results (§2) in research or backlog
