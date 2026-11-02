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
  (each family's `SOURCES`) is fixed to "the latest commit at publication time". The table shape is ADR
  [0092](decisions/0092-distribution-repos-and-sources.md) decision 3).
- When the manifest format goes up, old JSR clients cannot read the new repositories (hub parses a single format —
  ADR 0041/0071). Putting the JSR publish last means the release goes out in the state "assets that the new hub can read are
  already on HF".

### Publication order for repositories with cross-repository references (MUST)

A cross-repository component reference (ADR [0038](decisions/0038-manifest-v1.md) §7 addendum) **bakes in the
commit SHA of the referenced repository**, so it cannot be built until the referenced repository has been published. Which roles a
repository borrows is the recipe's declaration (`Pipeline.references` — ADR 0038 §7 addendum 2026-10-28); the build supplies only
the commit. **The publication build passes `--resolve-refs hf`** (`tools/export-recipes/dist.py`): it reads the referenced
repository's main from the HF API and, before writing anything, requires every borrowed file to be in that commit with the bytes of
the local mirror under `models/` — a stale SHA, a referenced mirror rebuilt but not uploaded, or a renamed path fails the build
instead of a user's fetch. `--ref-revision <name>=<SHA>` pins it explicitly without that check. There are three pairs today:

| Referencing repository | Referenced repository | Borrowed roles (the recipe's declaration) |
| --- | --- | --- |
| `karume-anima-extra` (fine-tuned models — ADR 0087, the same shape as the former turbo → anima direction) | `karume-anima` (model `anima-v1.0`) | `text_encoder` / `vae_decoder` / `text_encoder_bf16` / `vae_decoder_bf16` / `tokenizer` / `tokenizer_2` (`ANIMA_EXTRA_SHARED_ROLES`) |
| `karume-wan2.2` (ADR [0119](decisions/0119-wan-umt5-gpu-text-encoder.md) addendum "design of step 10d", ADR [0121](decisions/0121-wan22-ti2v-5b.md) decision 9) | `karume-umt5-xxl` (model `xxl`) | `text_encoder` (`WAN_TEXT_ENCODER_REFERENCE`) |
| `karume-irodori-v4.1-small` | `karume-irodori-v4-small` (model `v4-small`) | 26 roles = 90 files, 5,637 MiB: every graph role in f32 / f16 / i8 except `duration`, plus `dit_context_i4` and `tokenizer` (`IRODORI_V41_SHARED_ROLES`) |

`karume-wan2.1` has the same reference as Wan2.2 but is not published on HF (the maintainer's ruling 2026-10-15). The Wan procedure
is the umT5 and Wan2.2 items of "First publication of a family" below. For the anima pair:

1. **Upload `karume-anima` first** (with the fragmentation countermeasures of §2; the 5 official variants (`anima-turbo-v1.1` (default) /
   `anima-v1.0` / `anima-aesthetic-v1.1` / `anima-turbo-v1.0` / `anima-aesthetic-v1.0`) — ADR 0087)
2. (No SHA is recorded by hand: `--resolve-refs hf` reads the commit SHA of its main and checks the borrowed files in it — see the note under 3.)
3. **Build `karume-anima-extra` with the cross-repository reference**: in `tools/export-recipes`,
   `uv run python dist.py --pipeline anima-extra --model anima-wai-v1.0 --model anima-copycat-20260610 --out ../../models/karume-anima-extra --resolve-refs hf`.
   **extra holds 2 models, so the one repository is built in one run** (`--model` repeated = family assembly; the first one is
   `defaultModel`). The borrowed roles are the table's (the role name is the artifact key, and the bf16 text stack is a separate role — bf16
   was extended to every model on 2026-10-17. `text_conditioner` is not referenced across repositories because both extra models have
   their own — `own_text_conditioner=True` in `distribution.py`; every model has its own bf16).
   **0.15.0**: the published main of `karume-anima` (the `anima.SOURCES["anima"]` pin) has no bf16 text stack, so
   `--resolve-refs hf` refuses extra until `karume-anima` has been uploaded again (10 of the 22 borrowed files are not in that commit —
   checked 2026-10-28). **Build extra with `--resolve-refs hf`, never with `--ref-revision`**: an explicit SHA is not checked against
   the Hub, so `--ref-revision karume-anima=<that pin>` would build a distribution that references 10 files its commit does not hold.
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

For the irodori pair (v4.1-small references v4-small — upstream v4.1 retrained only the duration predictor; v4-small stays
self-contained):

1. **`karume-irodori-v4-small` first.** When its bytes and manifest are unchanged since its published pin
   (`irodori.SOURCES["irodori-v4-small"]`), it need not be uploaded again — v4.1-small references that commit. When it is rebuilt and
   uploaded (a new generator tag or card text), upload it first and let the next step read its new main.
2. **Build `karume-irodori-v4.1-small`**: in `tools/export-recipes`,
   `uv run python dist.py --pipeline irodori --model v4.1-small --resolve-refs hf` (it reads the mirror `models/karume-irodori-v4-small/`).
   The result holds 16 files: the 12 parts of `duration` f32 / f16 / i8 and `dit` i4, the manifest, the card and the 2 legal texts.
3. **Upload it with `--delete` for the 90 borrowed paths** — `hf upload` does not delete files that are absent locally, and the
   self-contained 0.14.0 upload holds them on main: `tools/release/hf-upload.zsh upload karume-irodori-v4.1-small
   --delete 'v4.1-small/backbone/*' --delete 'v4.1-small/text_proj/*' --delete 'v4.1-small/caption_proj/*'
   --delete 'v4.1-small/speaker/*' --delete 'v4.1-small/dit_context/*' --delete 'v4.1-small/codec_decoder/*'
   --delete 'v4.1-small/codec_encoder/*' --delete 'v4.1-small/tokenizer/*' --delete 'v4.1-small/dit/model.f32-*'
   --delete 'v4.1-small/dit/model.f16-*' --delete 'v4.1-small/dit/model.i8-*'`, then compare HF's main with the mirror as in §2
   (the 12 kept parts are byte-identical to what is already there). The old revisions keep the full files, so pins of 0.14.0 and earlier
   keep working.
4. **Pins**: update `irodori.SOURCES["irodori-v4.1-small"]` (and `"irodori-v4-small"` only when step 1 uploaded it).

If the referenced repository is uploaded again later, its SHA changes, and the referencing manifest **keeps pointing at the old revision**
(the bytes are protected by the double pin, so nothing wrong is served, but the two repositories' contents become different generations) —
**when the referenced repository is uploaded again, rebuild and re-upload the referencing one too** (extra for `karume-anima`,
`karume-wan2.2` for `karume-umt5-xxl`, `karume-irodori-v4.1-small` for `karume-irodori-v4-small`). **0.15.0: `karume-wan2.2` must be
re-uploaded in any case — its manifest gained the `f16+dit8-a8-attn8-online`, `i8` and `i8-a8-attn8-online-conv8` presets
(ADRs 0132 / 0133), the default moved to `i8-a8-attn8-online-conv8`, and it gained the `vae_decoder_first_32` / `vae_decoder_next_32`
containers (ADR 0134) — check the new containers in the fragmentation check of §2 too.**

**The placeholder-SHA check**: a development mirror for the time the referenced repository is unpublished is built with
`--allow-placeholder-ref`, which pins every reference at a placeholder SHA (40 zeros — `PLACEHOLDER_REVISION` in
`tools/export-recipes/dist.py`; the local source is resolved by the `crossRepo` mapping and does not look at the revision). The placeholder
SHA satisfies the form of 40 hex digits, so it passes hub's parse and `verify_dist` too — the publication rebuild does **not** pass
`--allow-placeholder-ref`, and `--ref-revision <name>=<40 zeros>` is refused. A build that declares references and gets no revision source
fails before writing a single byte. The rebuild after the bump of §4 is always done (do not upload the development mirror as is —
right before uploading, confirm that `rg -c '"revision": "0{40}"' models/<repo>/karume.json` prints nothing (no placeholder SHA
remains)).

NOTE (for the next release only): the old `hdae/karume-anima-turbo` was retired by ADR 0087 — do not upload it again.
How to handle the published repository (deprecation notice, replacing the README, etc.) is the maintainer's ruling at upload time.

## 1. Pre-release checks

NOTE (Node version): run every `node` / `vp` / `pnpm` command of this runbook inside the repository, where `.node-version` pins Node
to the 24 line (the same line as CI's `node-version: 24`). Without it vp resolves `engines.node` (`>=24.0.0`) to the newest release,
and Node 26 does not start on the maintainer's machine (its binary needs `libatomic.so.1`). `tools/release/hf-upload.zsh` calls `node`
after changing to the repository root, so the pin applies to it too.

- [ ] `vp check && vp test` green (full verify = every lane with the GPU tests — a SKIP on an environment without an adapter is not allowed for the release decision;
      ADR 0005). **Conditions for the release machine**: real hardware whose adapter
      **enumerates** `shader-f16` and `timestamp-query` (a software adapter (lavapipe etc.) computes f16 in f32,
      so it is kept apart from the real-hardware lane), with the distribution mirrors (`models/`) and a complete fetch of the
      pinned test assets (`node tools/test-assets/fetch.ts` → `outputs/test-assets/` — ADR 0139) in place.
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
- [ ] In a release that changes the WGSL or the selection condition of the subgroup variants (`rmsNormReduce: "subgroup32"` / `linearGemvReduce: "parallel-subgroup32"`)
      or a bit-identical states-attention rewrite (the folded ①′ / unrolled ③′ — perf-ledger K-55 / K-91),
      run the playground's `/kernel-check` page (**Run all**) in Chrome on the D3D12 machine (and on Metal when one is at hand) and keep its record
      ([apps/playground/README.md](../apps/playground/README.md#the-kernel-check-page) "The kernel-check page").
      The node lanes run the same checks only on Dawn / Vulkan, and a device without the subgroup features SKIPs the subgroup ones; there is no check
      that turns that SKIP red (the "The real-execution checks of the subgroup variants do not run automatically" section of limitations).
      "not available" on the page is not a PASS, and neither is a page where a check was not run (the overall verdict then reads INCOMPLETE).
      Until K-95 lands, the states-attention check FAILs on D3D12 in the e4b-full classes only (shown as "kept for K-95"; the product does not fold them);
      its verdict "outside the K-95 classes" (`productVerdict` in the record) must read PASS — any failure there is a regression on that backend.
      Before 0.15 the procedure was Chrome's `/check.html` (GEMV) and the RMS round-trip comparison (`tools/llm-speed/browser/README.md` of 0.14.0 — ADR 0127)
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
- [ ] **Use the project account's token**: `tools/.venv/bin/hf auth switch --token-name "Karume Release"` (the
      `hdae` account, read and write) — check the selected token with `tools/.venv/bin/hf auth list`. It is the
      everyday token (maintainer, 2026-11-02). `Karume Gated Read` belongs to a different, personal account and is
      used **only** to fetch gated upstream models; it cannot be granted access to `hdae` repositories (with it the
      LFS batch returns 403 — measured 2026-08-21). Switch to it only for such a fetch and back to `Karume Release`
      afterwards. NOTE: with a write token in everyday use, a mistake in read-only work can become a write to HF
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

They live in the per-family mapping tables of `packages/models` — `SOURCES` of each subpath, `<namespace>.SOURCES` through the barrel (key = the basename of the HF repository name without
`karume-`) and in `KARUME_SOURCES`, into which the barrel folds every family (ADR
[0092](decisions/0092-distribution-repos-and-sources.md) decision 3).

- [ ] Get the SHA (40 hex) of main of each published repository:
      the `sha` field of `curl -sS "https://huggingface.co/api/models/<owner>/<repo>/revision/main"`
- [ ] **Write it into the `revision` of the matching key** for each uploaded repository
      (e.g. `irodori.SOURCES["irodori-v4.1-small"].revision`). The key follows mechanically from the repository name,
      so the repository name determines where to write
- [ ] **Confirm that `KARUME_SOURCES` is complete**: every repository uploaded this time is in the table.
      The invariants (`"karume-" + key === repo basename`, owner `hdae`, `revision` is 40 hex) are
      checked by tests, so `vp test --project core` fails on them
- [ ] **Create the entry of a newly published repository at this point** (inheriting the reason of ADR 0073 decision 1 — placing it before publication
      grows a key on the public surface that can only 404). The procedure is just adding 1 entry to the `SOURCES` table in each family's
      `config.ts`; the subpath exports the table and the barrel's namespace is that subpath, so nothing else needs to follow
- [ ] Updating the pins is **a duty at every bump** (the mapping table = "the sources **verified** by this package version" —
      inheriting the maintenance duty of the ADR 0073 addendum of 2026-08-25). In addition to the connectivity check below, when presets or the default quant moved,
      pass the behaviour tests too before claiming "verified"
- [ ] Connectivity: `fromPretrained` with the written SHA passes against the real URL (with a SHA, no revision-resolution
      request happens = offline start-up is possible — ADR 0038). **The connectivity check across every entry of the table is
      `node tools/published-smoke/main.ts` of §5** — it walks the whole `KARUME_SOURCES` with the published `@karume/hub`,
      so it can only be run after the JSR publish (the order of §0)
- [ ] **The values of the mapping tables are the only place where the published revisions live**: check with the equivalent of `rg '\b[0-9a-f]{8,40}\b' docs
      .claude/ACTIVE_DESIGN.md` that no SHA was copied into the docs (copies go stale at every
      rebuild — records are spelled with keys like `anima.SOURCES["anima"]`. To prevent a repeat of the 2026-08-24 / 08-29
      mismatch in two releases in a row)
- [ ] `vp check && vp test` → commit

### First publication of a family (creating the mapping table itself — the most recent case: 0.14.0's umT5 = `karume-umt5-xxl` and Wan2.2 = `karume-wan2.2`)

A family without a public repository has no mapping table (ADR 0073 decision 1 — the same as `vowel-detector`). In the release that
creates the table, in addition to "add 1 entry" above, put the checks that bind the table's completeness in the same commit (forgetting one of them
makes that check fail, or goes green while the table is not on the public surface). In 0.14.0, 2 families were published for the first time at once:
umT5 (TS entry `Umt5Encoder` — `@karume/models/umt5`) got `UMT5_SOURCES`, and Wan got `WAN_SOURCES` (0.15.0 spells
them `umt5.Encoder`, `umt5.SOURCES` and `wan.SOURCES` — ADR 0136).

Wan2.1 (`karume-wan2.1`) is not published on HF (the maintainer's ruling 2026-10-15 — ADR
[0118](decisions/0118-wan21-video-generation.md) addendum (2026-10-15)). The code stays, and the distribution is assembled locally
with `dist.py --pipeline wan` (the same umT5 reference as Wan2.2 below — before umT5 is published, `--allow-placeholder-ref` (the
placeholder of §0); after it is published, `--resolve-refs hf`. `--out` is the default `models/karume-wan2.1`). Do not add `"wan2.1"`
to `wan.SOURCES`, and do not upload `karume-wan2.1`.

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
- [ ] **Rebuild Wan2.2 against umT5's published main**: `uv run python dist.py --pipeline wan-ti2v --resolve-refs hf` (it pins the
      reference at the main of `karume-umt5-xxl` and checks every borrowed part there; do **not** pass `--allow-placeholder-ref` — do not
      upload with the development mirror's placeholder SHA. `--out` is the default `models/karume-wan2.2`) → after confirming that every
      text_encoder part in `karume.json` points at the SHA recorded above, `tools/release/hf-upload.zsh upload karume-wan2.2`
- [ ] Add the tables (key = the basename of the repository name without `karume-`, value = `repo` + the SHA of main):
      add `umt5.SOURCES` (key `"umt5-xxl"` → `hdae/karume-umt5-xxl`) to `packages/models/src/umt5/config.ts` and
      export it from `umt5.ts` (the barrel's `umt5` namespace follows); add `wan.SOURCES` (key `"wan2.2"` →
      `hdae/karume-wan2.2` — read by `wan.Ti2vPipeline`) to `packages/models/src/wan/config.ts` and export it from `wan.ts`. NOTE: the TS
      `umt5.SOURCES` is the mapping table of the public distribution repositories, not the same thing as the recipe's `wan.sources.UMT5_SOURCES` (the table of the original's upstream pins)
- [ ] Fold both tables into `KARUME_SOURCES` in `packages/models/src/sources.ts`
- [ ] Add 2 rows to `FAMILIES` in `packages/models/tests/sources_test.ts` (`load` is `umt5.Encoder.fromPretrained` and
      `wan.Ti2vPipeline.fromPretrained`) and update the named counts of families and repositories (7 families, 10 repositories → 9 families, 12 repositories)
- [ ] Add both subpaths to the table of source-table names (subpath → `SOURCES`) in `packages/models/tests/models_barrel_surface_test.ts`
- [ ] Add `("umt5", ["xxl"], None, "karume-umt5-xxl")` and `("wan-ti2v", ["ti2v-5b"], None, "karume-wan2.2")` to
      `RELEASE_REPOSITORIES` in `tools/export-recipes/tests/test_dist_driver.py`
      (the check that requires the same set of repositories as the TS tables — it reads the `repo: "hdae/…",` lines of `packages/models/src/*/config.ts`)
- [ ] Pass the hint for an omitted ref (the third argument of `toManifestSource`): `SOURCES["umt5-xxl"] (@karume/models/umt5)` to
      `umt5.Encoder.fromPretrained` (`packages/models/src/umt5/encoder.ts`), and `SOURCES["wan2.2"] (@karume/models/wan)` to
      `wan.Ti2vPipeline.fromPretrained`. Wan's call shares the single `loadWanFromPretrained` of
      `packages/models/src/wan/family.ts` with `wan.Pipeline` (2.1), so
      shape it so that 2.1 gets no hint (the table has no 2.1 key)
- [ ] Fix the docs that say "there is no table yet" or "unpublished, no pin constant either" (pick them up with `rg -n 'SOURCES|未公開|not
      published' packages/models`). Keep the statements that Wan2.1 is unpublished
- [ ] In the note of `DISTRIBUTIONS` in `packages/runtime/tests/distribution_gate_test.ts`, move `karume-umt5-xxl` and
      `karume-wan2.2` to "published" (the mirrors themselves were on the all-skip guard before publication. `karume-wan2.1`
      stays unpublished and the e2e tests use it as their root, so it stays on the list)
- [ ] Rewrite the public-surface snapshot (`KARUME_SURFACE=write`) and confirm that the difference is one `SOURCES` line in each of `./umt5` and `./wan`
      (the barrel's `.` entry lists the namespaces only, so it does not change)

## 4. JSR publish

Mechanism: **publishing a GitHub Release fires `publish.yml`**, and `deno publish` at the workspace root
publishes the 3 packages (hub / runtime → models) together in dependency order
(already-published versions are skipped idempotently; OIDC, tokenless).

NOTE: the package README / LICENSE are part of what is published (`packages/*/README.md` and
`packages/*/LICENSE` — confirmed on 2026-09-20 that they appear for all 3 packages in the output of `deno publish --dry-run --allow-dirty`;
re-confirmed on 2026-10-28 with `"nodeModulesDir": "none"` in the root `deno.json`, on a clean `git archive` copy.
The `publish` of each `deno.json` is only `exclude: ["tests/"]`; no explicit `include` is
needed).

- [ ] Lockstep bump commit: the `version` of the 3 JSR packages in both their `deno.json` (what JSR publishes) and their
      `package.json` (`tools/published-smoke` reads the expected version from `packages/models/package.json`), the root `package.json`,
      the exporter, the `^` dependencies such as models → hub in `packages/models/deno.json`, the specifiers in `deno.lock` (run
      `deno install` at the repository root after editing the `deno.json` files — with `"nodeModulesDir": "none"` it rewrites only
      `deno.lock`), and **regenerating `tools/uv.lock` (`uv lock` — MUST. CI
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
- [ ] push (the maintainer) → confirm CI is green, **including the `publish-dry-run` job** (`deno publish --dry-run` with the Deno version and
      directory of `publish.yml` — ADR [0127](decisions/0127-node-dawn-toolchain.md) addendum of 2026-10-28); do not cut the Release while it is red
- [ ] Create a GitHub Release from the main commit with green CI → published (fires)
- [ ] Confirm the new version of the 3 packages on JSR
- [ ] **0.15.0 only — card-only re-upload of the published repositories not rebuilt in this release** (ADR
      [0136](decisions/0136-models-public-naming.md)): their Usage snippets import the 0.14 names from `jsr:@karume/models` without a
      version, so they stop compiling once 0.15.0 is the latest. Regenerate the cards with this release's recipes and upload
      `README.md` alone (manifests, containers and the `SOURCES` pins do not move), right after the JSR publish and in the same session
      (the backlog release section holds the item)

PyPI `karume` (tools/exporter) is **unreleased**. Add the procedure to this section when publication starts.

## 5. After the release

- [ ] Docs sync: ACTIVE_DESIGN's Now, backlog (the now heading and the progress of the release section),
      the release record, the project memory. **The release is not "done" until this item is done** —
      treat this one place as the only check for syncing the ledgers (in 0.9.0 only backlog was updated and
      ACTIVE_DESIGN stayed at "the maintainer publishes". The second miss of the same shape)
- [ ] Connectivity from the published packages (tools/published-smoke — the published versions, ADR 0127). Before running it, write the
      published version into `tools/published-smoke/package.json` (`"jsr:^<version>"`) and `tools/published-smoke/deno.json` (the exact
      `jsr:@karume/*@<version>` — a `^` range there fails the version check), run `pnpm install`, and commit both files with the updated
      `pnpm-lock.yaml` (CI installs with `--frozen-lockfile`). The expected version is read from `packages/models/package.json`; a package
      that resolves to any other version fails the run. Node: `node tools/published-smoke/main.ts` from the repository root. Deno (the
      import check): `deno run -A main.ts` **inside `tools/published-smoke`** — under the root workspace `@karume/*` resolves to the local
      members, so the Deno run starts in that directory to read its own `deno.json` (the one-line "ignoring the parent workspace" warning
      is expected). With `@karume/hub` it resolves the manifest of every entry of `KARUME_SOURCES` and takes sbv2 through
      `fromPretrained` (on a machine without a GPU, `--manifests-only`).
      **Freshly published dependencies get caught by the minimum release age** — on Deno, `jsr:@karume/*` and `jsr:@hdae/fetch-cache`, which
      hub pulls in, are listed in the `exclude` of `minimumDependencyAge` in `tools/published-smoke/deno.json` (the example of 0.11.0, where
      fetch-cache 0.7.0, published one day earlier, was stopped — 2026-09-06); when a new `@hdae/*` dependency is added, add it there too.
      On Node, `minimumReleaseAgeExclude` of `pnpm-workspace.yaml` already covers `@jsr/karume__*` and `@jsr/hdae__*`
- [ ] Record one line of the fragmentation check results (§2) in research or backlog
