# 0122: umT5 upstream moves to google/umt5-xxl; a path to build and swap in compatible text encoders

- Status: accepted (2026-10-04, the maintainer's rulings on the explicit opt-in for undeclared
  licences and on keeping component replacement as it is); decisions 3 and 8 partly superseded by
  this ADR's addendum of 2026-10-15 (the umT5 repository now carries its own tokenizer asset);
  decision 4's "the modules stay under `wan/`" and the `examples/wan` part of decision 7.2 withdrawn
  2026-10-18
- Date: 2026-10-04
- Related: research [2026-10-04-umt5-upstream-provenance](../research/2026-10-04-umt5-upstream-provenance.md)
  ("provenance study §n" — weights, tokenizer and container effects, measured) /
  [2026-10-04-component-swap-recon](../research/2026-10-04-component-swap-recon.md) ("swap study §n" —
  the state of component replacement and its gaps) / the read-out of the community fine-tune of umT5
  (`.claude/reviews/2026-10-04_umt5-variant-recon/`, untracked; third-party names live only there) /
  ADR [0119](0119-wan-umt5-gpu-text-encoder.md) (the umT5 GPU path, the separate repository
  `karume-umt5-xxl`, cross-repository references — this ADR replaces the provenance statement of its
  "step 10d design" addendum, part D) / [0108](0108-container-format.md) decision 19 (component
  replacement) / [0109](0109-manifest-v5-container.md) decisions 3 and 10 (double pin of
  cross-repository references; admission of a replacement) / [0038](0038-manifest-v1.md) §7
  (cross-repository references) / [0086](0086-distribution-source.md) (source handles, explicit
  cross-repository mapping) / [0088](0088-civitai-air-intake.md) (intake of third-party weights with a
  provenance JSON — the precedent) / [0092](0092-distribution-repos-and-sources.md) decisions 1, 2, 6
  and 7 (repository granularity and naming, licence-encumbered derivatives, bundled legal texts) /
  [0065](0065-exporter-core-recipe-split.md) (exporter core and recipe boundary) /
  [0106](0106-device-keyed-references.md) (per-environment sha256 rows) /
  [0121](0121-wan22-ti2v-5b.md) (Wan2.2 TI2V-5B, which also references the umT5 container)

## Context

### Requests

The maintainer asked for three things:

1. **Make the upstream the original.** `karume-umt5-xxl` was built from the `text_encoder` of Wan2.1
   Diffusers, so its model card named "a component of the Wan repository" as its origin. umT5-XXL is
   Google's general encoder; export it from `google/umt5-xxl`.
2. **Design the replacement of a dependent model** (for example Wan's text encoder) with a compatible
   one.
3. **Try converting a community fine-tune of umT5** published on HF and run it as Wan2.1's text
   encoder.

### Terms first used here

| Term                                      | Meaning                                                                                                                                         |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| original (`google/umt5-xxl`)              | Google's HF repository: the full `UMT5ForConditionalGeneration` (encoder, decoder, `lm_head`) as six pickle `.bin` shards                       |
| pickle `.bin` shards                      | `pytorch_model-0000N-of-0000M.bin` plus the index `pytorch_model.bin.index.json`; unlike safetensors, loading can run arbitrary code             |
| tied alias                                | one weight under two names; in umT5 `shared.weight` and `encoder.embed_tokens.weight`                                                           |
| compatible text encoder                   | an umT5 with the original's configuration (the graph-relevant fields of `config.json`) and different weights, such as a fine-tune               |
| intake                                    | fetching third-party weights at a pinned revision, checking sha256, storing the files unchanged under `inputs/` with a provenance record (as ADR 0088) |
| provenance record (`intake.json`)         | JSON written by the intake script from API values: repository, revision, files, sha256, declared licence and base, source dtype, config origin  |
| local experimental mirror                 | a distribution-shaped directory that is never published; it lives outside `models/`                                                            |
| undeclared licence                        | the HF repository names no licence (no `cardData`, no README); the redistribution terms are unknown                                              |

### Findings (numbers from the studies)

- **The weights are identical.** The 243 encoder tensors of the original at commit
  `66cb9e7e85526fe440a945569e42c72fb6cbc0ad` (`apache-2.0`) match the 242 tensors of Wan2.1
  Diffusers' `text_encoder` bit for bit in f32 (`encoder.embed_tokens.weight` shares storage with
  `shared.weight`). Wan's f32 is real f32, not an upcast of bf16 (provenance study §2). Quantization
  and the traced graph are identical too (§4.1).
- **The obstacles were the recipe's reader**: the original ships only pickle `.bin` (safetensors exist
  only in two unmerged bot PRs whose shard 1 differs in sha256 for an unknown reason); the index
  lists both tied aliases; `UMT5EncoderModel.from_pretrained` requires all six shards (about 51.9 GB)
  while the encoder needs shards 1–3 (29,837,688,579 B).
- **The tokenizer gives the same ids but different bytes**: no id mismatch over 2,034 inputs, but the
  original `tokenizer.json` differs in the last digits of 65,654 scores (§3).
- **Wan2.2 TI2V-5B uses the same umT5**: its bf16 is the round-to-nearest-even (RNE) rounding of
  Wan2.1's f32 for all 242 tensors (§6).
- **Component replacement already exists** (swap study): eight families' `fromPretrained` accept
  `components: { <role>: { source, model?, quant? } }`, handled in one place
  (`packages/models/src/hub/components.ts`). A replacement is accepted when its graph descriptor
  sha256 equals the original manifest's declaration, bindings have no missing or surplus entries, and
  the family check passes. The graph descriptor does not depend on the weights. Only containers are
  taken from the replacement; tokenizer and other assets come from the original manifest and nobody
  checks they match. Wan's family check (`umt5Contract`, now in
  `packages/models/src/wan/text-stage.ts`) accepts only i8 and f32 storage.
- **The graph is not the whole contract**: changing `relative_attention_max_distance` from 128 to 64
  produced a byte-identical container. The bucket table is a graph input whose configuration the TS
  host fixes (`UMT5_RELATIVE_POSITION = { numBuckets: 32, maxDistance: 128 }` and the effective length
  512 in `packages/models/src/umt5/relative-position.ts`). Component replacement neither checks nor
  records the provenance, licence or pipeline name of the replacement.
- **The community fine-tune**: one bf16 safetensors file (11,361,851,256 B) with the 242 HF-style keys
  of Wan's `text_encoder`, no `config.json`, no index. Only 80 tensors (blocks 16–23) differ; the
  other 162 equal the RNE rounding of Wan's f32. The fp8 variant is a scale-1 cast (relative RMS
  error of q 4.9%) and unusable as input. Its embedded `spiece.model` matches the original's sha256
  (fact); that the bf16 variant used the same tokenizer is an inference. **It declares neither licence
  nor base.** i8 quantized from bf16 differs in bytes from i8 quantized from f32 even in unchanged
  layers.

### Constraints

- Unsupported or unexpected input fails loudly; sha256 regression checks are never turned into
  tolerances (ADR 0106).
- The recipe's MUST "the starting point is F32" (`Checkpoint` in `tools/export-recipes/wan/umt5_export.py`):
  another dtype changes the rounding start and therefore the i8 bytes.
- The distribution check closes on the container's provenance
  (`assert_upstream_provenance` in `tools/export-recipes/_shared/upstream.py`), not on a model-name
  table or on a path alone.
- ADR 0092 decision 6: assets whose redistribution cannot be backed in writing (unknown training data,
  uncensored derivatives) go to a separate library and HF repository built on the exporter core,
  never into this repository's models or tables. The fine-tune is such an asset; this ADR's intake
  stops at a local experimental mirror and does not replace that route.
- Third-party repository names (adult ones in particular) are not written into tracked files.

## Decision

### 1. umT5's upstream is `google/umt5-xxl`

Protects: the distribution's provenance matches its real origin, and Wan2.1 and Wan2.2 refer to it in
the same words.

- The upstream of the umT5 container is the original at commit `66cb9e7e…` (`apache-2.0`), held in
  its own table `UMT5_SOURCES` in `tools/export-recipes/wan/sources.py` (model `xxl`). Container
  provenance, the distribution check, the card and NOTICE read this row. Wan's table `SOURCES` stays
  for Wan's DiT, VAE and tokenizer.
- Card and NOTICE: `base_model: google/umt5-xxl`; attribution names "the encoder weights
  (`shared.weight` and `encoder.*`) of google/umt5-xxl at commit …" without a subfolder and without
  the decoder and `lm_head`; the relation line states the bit identity with Wan2.1 Diffusers'
  `text_encoder` and the RNE relation to Wan2.2's bf16 (confirmed 2026-10-04). The modification
  notice says only the encoder was converted. `tools/export-recipes/wan/THIRD_PARTY_NOTICES.md` names
  the same upstream.
- Why: the values are the same, so changing the provenance table meets request 1, and the original is
  correct for both consumers.

### 2. Reading the original — pin, shards from the index, mmap `.bin` reads, tied aliases

Protects: a moved upstream is never read silently; unpickling never runs arbitrary code; resident
memory (RSS) stays bounded while reading about 30 GB of shards.

- **Pin** the main commit by its 40-digit SHA (no branch names).
- **Derive the shards from the index**: fetch `config.json` and `pytorch_model.bin.index.json`, derive
  the shards holding `shared` and `encoder.*` from `weight_map`, fetch only those. Shard file names
  are not written by hand.
- **sha256 before unpickling**: the row holds the 64-digit LFS sha256 of each needed shard. Each
  shard is checked before `torch.load`; any mismatch, or a derived shard set that differs from the
  table, fails without opening anything.
- **Reader**: `Checkpoint` gains the `.bin` shard form, read with
  `torch.load(path, map_location="cpu", mmap=True, weights_only=True)`, one tensor and one row block
  at a time; the F32 MUST still applies. A non-zip `.bin` fails with a named error. The safetensors
  reader stays (decision 5 uses it).
- **Only the pinned original opens `.bin`**: the `.bin` form is opened only from a table row that
  carries sha256, never because a directory contains an index. Intake (decision 5) accepts
  safetensors only. `weights_only=True` is the second defence; the sha256 pin is the first.
- **Tied aliases** (`checkpoint_keys`): when several candidates exist, they are accepted only if they
  share storage or are bit-identical by row blocks, and `shared.weight` (transformers' tied target) is
  used. Any difference, or no candidate, fails loudly.
- The unmerged bot PRs' safetensors are not used. The graph is still built from
  `UMT5Config.from_pretrained` on the meta device, never from `UMT5EncoderModel.from_pretrained`.
- Implemented 2026-10-04 (`c657a3dd`): each read opens and closes the shard (holding shards open put
  file-backed pages in RSS: quantization peak 17.62 GiB, against 6.14 GiB when reopening at 10–19 ms
  per open). The sha256 check runs on the descriptor held at construction and reads use the same
  descriptor, so swapping the path after the check still reads the checked content; an in-place
  rewrite of the same inode cannot be blocked while reading by mmap (documented in the docstring).
  Arguments that do not apply to a subcommand are rejected when given.

### 3. Tokenizer asset, check-mask, precomputed embeddings and bucket fixture keep Wan as upstream

Protects: the bytes of Wan's tokenizer asset and preprocessing tables do not change; each component's
provenance statement is correct and consistent.

- These keep using the Wan snapshot: the tokenizer asset (`wan/umt5_tokenizer.py`, format
  `karume-wan-umt5-tokenizer/1`, which bundles Wan's preprocessing tables), check-mask, the
  precomputed embedding asset (`wan/text_embeds.py`), the host bucket fixture's config
  (`wan/umt5_host_fixture.py`) and the golden cases' id sequences.
- **The writer has two axes**: `--upstream` (umT5: weights, config and provenance for `prepare` /
  `write`, config and unquantized weights for `reference`, the distribution check and attribution)
  and `--model` (Wan's text side: id sequences of the `reference` cases, `check-mask` /
  `compare-mask`, tokenizer asset, precomputed embeddings, host fixture). `reference` mixes both and
  records which value it used in the golden metadata.
- Why: building the Wan asset from the original changes only score digits; running check-mask on the
  original needs all six shards.
- Cards: Wan's card and NOTICE describe umT5 as "the original encoder (bit-identical in f32 to this
  checkpoint's `text_encoder`)"; the tokenizer is "converted from this checkpoint's `tokenizer/`".
- (Superseded by this ADR's addendum on 2026-10-15: `karume-umt5-xxl` now also carries a tokenizer
  asset of its own, built from the original. Wan's asset stays with Wan with unchanged bytes.)

### 4. The series is renamed after its provenance

Protects: a series name does not misstate where its content comes from.

- Series `wan2.1-umt5-i8-dyn` became `umt5-xxl-i8-dyn` (`<name>-<storage>-dyn`). The tokenizer series
  `wan2.1-umt5-tokenizer`, the repository `karume-umt5-xxl`, model `xxl`, pipeline `umt5-encoder/1`
  and component `text_encoder` keep their names.
- Withdrawn 2026-10-18: "the recipe modules stay under `wan/`". Its own revival condition (the
  tokenizer asset moves to the umT5 repository) was met by the 2026-10-15 addendum. The writers
  (`umt5_export`, `umt5_reference`, `umt5_patch`, `umt5_distribution`, `umt5_intake`,
  `umt5_xxl_tokenizer`) are still under `tools/export-recipes/wan/`; a split into
  `tools/export-recipes/umt5/` (ADR 0065 decision 2, one directory per family) is open for a new
  decision.

### 5. Building a compatible text encoder — intake script and provenance record

Protects: third-party provenance (which repository, which revision, under which terms) is
machine-traceable; the provenance check stays; no third-party repository name enters tracked code;
the rounding start is never misstated (the table path keeps "the starting point is F32"; the intake
path states its source dtype).

- **Intake script** (`tools/export-recipes/wan/umt5_intake.py`) takes
  `--repo <owner/name> --revision <40 digits> --file <name> --name <name>` and:
  1. reads the HF API values (LFS sha256 at that revision, licence and base from `cardData`);
  2. fetches the one file at the pinned revision and checks sha256 (`.part`, renamed on a match, as
     ADR 0088 decision 4); safetensors only — `.bin` is rejected;
  3. reads the header: the expected key set is the 242 keys of Wan's `text_encoder`; a file with both
     tied aliases (the original's 243) follows decision 2; all tensors must share one dtype, BF16 or
     F32. FP8, mixed BF16/F32 and legacy key spellings (`blocks.N.attn.q`) fail loudly;
  4. stores the file **unchanged** under `inputs/umt5/<name>/` (no F32 copy), so its sha256 and name
     still match the upstream;
  5. stores `config.json`: an upstream config is compared with the original's on the graph-relevant
     fields and the bucket configuration and fails loudly on a difference; without one, the
     original's config is copied and `intake.json` records "configuration inferred (copy of the
     original)" — the comparison is then empty, and `relative_attention_max_distance` cannot be
     derived from weight shapes (a limitation of decision 7);
  6. writes `intake.json` (machine-owned; format `karume-umt5-intake/2`). It does not judge the
     licence (ADR 0088 decision 5); the agent's review goes into `license-review.md` beside it.
- **Writer**: `umt5_export write --intake inputs/umt5/<name>` writes series
  `umt5-xxl-<name>-i8-dyn`, with provenance from `intake.json`. `Checkpoint` accepts only the dtype
  `intake.json` declares and widens BF16 to F32 at read time (lossless), so the container is
  byte-identical to one written from an F32 copy of the same values. The table path still accepts F32
  only. The mirror's card states that the upstream weights are bfloat16 and were widened losslessly
  before quantization (container provenance has no dtype field — container-v1 §2.3).
- **Distribution check**: `assert_umt5_encoder` reads `intake.json` for a model not in the code table
  and compares the container's provenance with it.
- **No third-party component in `karume-umt5-xxl`**: third-party models never take that name; their
  default output is outside `models/` (`outputs/misc/local-dist/<name>/`). An intake-derived model
  fails before writing when the output resolves (symlinks and `..` resolved) under `models/`,
  **whether or not it declares a licence**.
- **Root files of a third-party mirror** derive from `intake.json` (`intake_root_files`), never the
  original's Apache `LICENSE.md` and NOTICE: a declared licence gives its verbatim text (an identifier
  without text in `_shared/licenses/` fails); an undeclared one gives no `LICENSE.md` and a statement
  at the top of `NOTICE.md` and the card. `NOTICE.md` names the upstream from `intake.json` and the
  changes made here, and never says "No retraining" or names Wan or Google as upstream.
- **No publication route here**: publishing a declared third-party component needs a ruling on repository
  naming and card attribution (ADR 0088: each new model is its own ruling).
- Implemented 2026-10-04 (`ca38d638`, `6cdad993`). Review fixes: reads go through a descriptor checked
  against the record; `intake.json` records the sha256 of `config.json` and checks it at entry; the
  intake directory accepts only an allow-list (weights, `config.json`, `intake.json`,
  `license-review.md`) and rejects symlinks.

### 6. Weights with an undeclared licence — an explicit opt-in for local experimental mirrors only

Protects: weights with unknown redistribution terms never reach the publication route; local trials
(request 3) still work. The maintainer ruled on 2026-10-04 to provide this opt-in.

- When `intake.json` declares no licence, the writer and the distribution build fail by default. With
  `--allow-undeclared-license` they write the marker `NOASSERTION` (SPDX) into the container's licence
  field and may build a local experimental mirror — the same "open only when explicit" form as
  `--allow-placeholder-ref` (`tools/export-recipes/dist.py`). Licence values that cannot identify terms
  (no declaration, `unknown`, `none`, an `other` without `license_name`) are recorded as undeclared;
  a named `other` passes as declared.
- **Three guards on the publication route** (1 and 3 close on container content, as
  `_shared/upstream.py` requires, not on location):
  1. the distribution build refuses a container with the marker without the flag, before writing a
     byte;
  2. an intake-derived model fails when its output is under `models/` (decision 5);
  3. **pre-upload guard**: `tools/release/hf-upload.zsh upload` runs `tools/release/container_license.ts`
     (Node) over `models/<repo>`, reads `provenance.license` from every container's part 0, and
     uploads nothing when a marker (any case) is present or the directory has no container.
  - Remaining gap: running `hf upload` directly bypasses every guard; the container marker and the
    card's opening statement keep the status visible.
- The card of an experimental mirror opens with "local experiment, upstream declares no licence, do
  not redistribute".
- Difference from ADR 0088 decision 5 (intake neither judges nor forces confirmation): intake still
  only records; the flag applies to the writer and the distribution build, and what it tests is the
  mechanical fact "the licence field is empty", not an interpretation of terms.
- Why: the alternative is to falsify the licence field or to block request 3. The marker stays in the
  container; it carries only "the upstream declared no licence", since provenance holds a revision but
  no repository name.

### 7. Replacement stays with the current `components` mechanism (option A)

Protects: compatibility is decided by one content hash (the graph descriptor sha256), with no second
derivable field; the manifest format (`karume/5`) does not change. The maintainer ruled on 2026-10-04
to proceed with the current mechanism.

- The mechanism stays. Added:
  1. the route for building third-party compatible components (decisions 5 and 6);
  2. the explanation: intake and conversion steps in `tools/export-recipes/wan/README.md` and a
     paragraph in the Usage of the `karume-umt5-xxl` card (`_umt5_usage` in `umt5_distribution.py`).
     It states that `source` is pinned as `{ repo, revision }` (a string reads as `{ repo }` and
     follows `main` — `packages/models/src/hub/repo-ref.ts`) and that replacement has no effect with
     `textEncoder: "precomputed"`. Withdrawn 2026-10-18: the `examples/wan` README and example, removed
     with `examples/` in `bb35fe2a` (2026-10-16); a new home depends on ADR 0127's open point on
     rebuilding the examples as browser applications;
  3. tests that exercise replacement without a GPU (Wan's positive path, replacing a
     cross-repository slot, an i4 replacement failing in `umt5Contract` before weights are fetched —
     landed in the step d addendum);
  4. `docs/limitations.md` ("Wan2.1: umT5 swap … what it does not check"), each item harm first:
     **prompts silently misread** (a component trained with another tokenizer of the same vocabulary
     size is accepted; nobody checks the tokenizer); **output silently degraded** (a different bucket
     configuration is accepted, since bucket configuration and effective length are host constants
     outside the graph descriptor; with a config-less upstream the intake comparison is empty);
     **weights of unknown licence run without warning** (replacement neither checks nor records
     provenance; the runtime does not know provenance — ADR 0108 decision 19);
  5. `docs/known-issues.md`: **replacing with a larger component fails only after the weights are
     fetched** — the replacement preset's `session` / `gpuFeatures` / `requiredLimits` are ignored and
     the original preset's values decide admission, so `fromPretrained` would succeed and the first
     `generate()` (where Wan's text stage builds its Session) would fail (inferred; no instance yet). ADR 0108 decision 19 asks
     admission to check this; it is not by design. Recipe-built components write the vocabulary
     embedding as i8 and are not affected; a component with an f32 vocabulary embedding passes
     `umt5Contract` and can fail this way.
- Why: the case at hand (the fine-tune) has the original's key set and shapes and, by inference, its
  tokenizer, so it passes the current acceptance conditions. The silent tokenizer error needs "same
  vocabulary size, other tokenizer", which has no instance yet.
- As of 2026-10-18 option C's revival condition (three or more sharing pairs, or a family other than
  Wan consuming umT5) is at a boundary: Wan2.2 also references `karume-umt5-xxl`, umT5 is a family of
  its own with `Umt5Encoder` (`packages/models/mod.ts`), and Wan2.1 is not published on HF. How to
  count is the maintainer's call.

### 8. Relation to Wan2.2 TI2V-5B (ADR 0121)

Protects: Wan2.1 and Wan2.2 share one umT5 container with one provenance statement.

- Wan2.2 also references `karume-umt5-xxl` across repositories and does not rebuild umT5 from Wan2.2
  Diffusers' bf16 (the rounding start would no longer be f32, giving a second, byte-different copy of
  the same encoder).
- Wan2.2's card and NOTICE describe umT5 as the original encoder (Wan2.2's own bf16 `text_encoder`
  is its RNE rounding). Wan2.2's distribution carries its own tokenizer asset with the preprocessing
  tables. (Superseded by this ADR's addendum on 2026-10-15: "this ADR adds no asset to the umT5
  repository" — `karume-umt5-xxl` now has its own tokenizer asset.)
- Publication order (umT5 first, then the Wan distributions with its real SHA — release-runbook §0)
  was followed at 0.14.0.

### 9. Steps

Steps a (baseline, then the switch of reader and provenance — decisions 1–4), b (intake, `--intake`,
JSON check, undeclared-licence opt-in, pre-upload guard — decisions 5–6), c (convert the fine-tune and
swap it into Wan2.1 on the GPU) and d (explanation, tests, limitations — decision 7) were all
completed on 2026-10-04; the addenda record the results.

## Alternatives not taken

- **Keep Wan2.1 as upstream**: no work, but request 1 remains, and Wan2.2 would use an encoder that
  calls itself a Wan2.1 component.
- **Use Wan2.2 Diffusers' bf16 (11.4 GB) as upstream**: lighter to fetch, but the rounding start is no
  longer f32 and the i8 bytes change, against the F32 MUST.
- **Read the bot PRs' safetensors**: not on the original's main, and the two PRs disagree on shard 1.
- **`UMT5EncoderModel.from_pretrained`**: needs all six shards (about 51.9 GB).
- **Build the tokenizer asset from the original**: same ids, only the bytes change (decision 3).
- **Move the umT5 recipe to `tools/export-recipes/umt5/` now**: rejected at the time by decision 4;
  that reason was withdrawn on 2026-10-18.
- **Hard-code third-party provenance in recipe code** (the form used for anima fine-tunes): puts
  third-party names into tracked code at every experiment and cannot hold ADR 0092 decision 6
  derivatives. A published third-party component decides on a code table within its own ruling.
- **State provenance through CLI flags** (`--license …`): the check would compare stated with written
  values, unlinked to the real upstream.
- **Store an F32 upcast copy (about 22.7 GB) under `inputs/` and keep `Checkpoint` F32-only**: meets
  the F32 MUST only in form (the rounding start is still bf16 and the container is byte-identical to
  decision 5's read-time widening), while storing a generated file among hand-placed weights, breaking
  the sha256 link to the upstream LFS value, and erasing the source dtype.
- **Let `Checkpoint` accept any dtype without a declaration**: an unexpected dtype would pass
  silently.
- **Use the fp8 variant**: lossy scale-1 cast.
- **Host the third-party component as a second model in `karume-umt5-xxl`**: mixes provenance and
  licences in one repository (ADR 0092 decision 1).
- **Put local experimental mirrors under `models/`**: that is the directory the upload script
  publishes; it would leave the pre-upload guard as the only barrier.
- **Never convert weights with an undeclared licence**: blocks request 3 (the maintainer chose the
  opt-in).
- **Option B — the manifest names role contracts and their assets**: removes the silent tokenizer
  error, but raises the manifest major, adds a second identifier not derivable from the graph sha256,
  and requires splitting Wan's bundled tokenizer asset. Revival: a compatible component with the same
  vocabulary size and another tokenizer appears (then rule only on declaring role-owned assets).
- **Option C — dependent models as first-class component repositories; replacement re-targets the
  cross-repository reference**: replaces assets along with the weights, but rebuilds every
  distribution and changes hub and models; only two sharing pairs existed (Wan → umT5, anima-extra →
  anima). Revival: three or more sharing pairs, or a family other than Wan consuming umT5 (see
  decision 7 for the 2026-10-18 count).
- **Bake the bucket configuration into the graph descriptor**: would reject a differently configured
  umT5 but changes the descriptor and forces new sha rows and fixtures; for a config-less upstream the
  baked value would also be a copy of the original's, so it would catch nothing. Revival: a compatible
  component with a different bucket configuration appears.
- **Check provenance and licence at replacement time**: the runtime does not know provenance (ADR
  0108 decision 19); recorded as a limitation.

## Consequences

- `karume-umt5-xxl` names the original as upstream. The weight parts did not change; part 0, the two
  manifest values that point at it, the card and NOTICE changed (verified at step a). For
  `karume-wan2.1` the same two cross-repository values and its card and NOTICE changed — this now
  concerns only the local distribution, since Wan2.1 is not published on HF (addendum 2026-10-15).
- The recipe reads the original's shards 1–3 (about 29.8 GB, sha256-checked on every `write` /
  `reference`); Wan's f32 `text_encoder` (about 22.7 GB) is still needed for the tokenizer,
  check-mask and precomputed embeddings.
- The F32 MUST holds on the table path; only the intake path accepts the declared dtype (BF16 or F32)
  and widens at read time.
- The upload script reads container licences before uploading; existing distributions carry no
  marker, so the release procedure is otherwise unchanged.
- The public TS surface did not change for replacement; it gained explanation and tests.
- Risks that remain: `weights_only=True` loading was observed without warnings on torch 2.13.0 only;
  graph descriptor identity across exporter / torch / transformers versions is unmeasured (a mismatch
  fails safe). The original's index and `config.json` are not pinned by sha256; a mismatching index
  fails with a plain `KeyError` (no path substitutes values).

## Open points

- Publishing a third-party compatible component that declares a licence (repository naming, card
  attribution, whether it enters a code table) is ruled on when requested. ADR 0092 decision 6
  derivatives are never published from this repository.
- Graph descriptor identity across exporter / torch / transformers versions (swap study §2.2).

## Addendum (2026-10-04) — step a: the switch changed no weight byte

Baseline at `1fb74d10` (`write --check` green, golden reproduced byte for byte, dist rebuild identical);
after the switch the weight parts 00002–00026 matched 25 of 25, part 00001 differed only in
`provenance.upstreamRevision`, and the 48 golden tensors matched bit for bit. Evidence:
`outputs/bench/karume-umt5-xxl/2026-10-04_stage-a/`; reader details under decision 2.

## Addendum (2026-10-04) — step b and the CPU part of step c: the fine-tune has the original's graph

Step b landed as decisions 5 and 6 describe (`NOASSERTION`, the Node `container_license.ts`). The
fine-tune (local name `community-ft`) was taken in (BF16, undeclared licence, inferred configuration);
its graph descriptor sha256 equals `karume-umt5-xxl`'s (`59ca0ccf…68ee`), and the pre-upload guard
rejects its mirror. Evidence: `outputs/bench/karume-umt5-xxl/2026-10-04_stage-c/`.

## Addendum (2026-10-04) — step d: replacing a cross-repository slot needs no `crossRepo` mapping

Fact: a Wan distribution given no `crossRepo` mapping passes admission and fetching once
`text_encoder` is replaced; without the replacement it fails with "cross-repository source missing".
The replaced slot takes the replacement's manifest as its origin (`packages/models/src/hub/components.ts`),
and a cross-repository origin is resolved only per reference (`sourceForRef` → `originFor` in
`packages/hub/src/source.ts`). Host test: `packages/models/tests/wan_components_test.ts` (the replaced
role comes only from the replacement and the original cross-repository part is fetched 0 bytes; the
mapping-free case for local and HTTP replacements; an i4 replacement fails in `umt5Contract` before
any weight part; a one-character change of the graph sha256 is rejected in admission).

## Addendum (2026-10-04) — step c on the GPU and the swap e2e

The swapped Wan2.1 passed admission and generated without non-finite values; the maintainer's visual
review (2026-10-05) found no large differences or collapse, which closed the quality question for i8
quantized from bf16. The opt-in e2e is `packages/e2e/tests/e2e_wan_swap_test.ts`
(`KARUME_WAN_SWAP_TEXT_ENCODER=<mirror directory>`), with a GPU-free control in the default lane.
Material: `outputs/verify/<environment key>/2026-10-04_wan-visual-gpu-text-community-ft/`.

## Addendum (2026-10-15) — umT5 usable without Wan (supersedes the "no asset in the umT5 repository" of decisions 3 and 8)

- The maintainer ruled (2026-10-15) that umT5 is usable independently of Wan, shipped in 0.14.0 (the
  first publication of `karume-umt5-xxl`). The same day's ruling keeps Wan2.1 off HF, so
  `karume-wan2.2` is the only published distribution that uses the umT5 repository.
- `karume-umt5-xxl` carries the model asset `tokenizer` (format `karume-umt5-tokenizer/1`): only the
  T5 table (vocabulary, scores, added vocabulary, whitespace set, maximum length 512), without Wan's
  preprocessing tables. Its source is the original's `tokenizer.json` and `tokenizer_config.json` at
  the pinned commit, both sha256-pinned in `wan.sources.UMT5_SOURCES` and checked before reading. The
  table builder is shared with Wan's asset (`t5_table_fields` in `wan/umt5_tokenizer.py`); the writer
  is `wan/umt5_xxl_tokenizer.py`, the series `outputs/series/umt5-xxl-tokenizer/tokenizer.json`.
  Wan2.2 keeps its own asset with unchanged bytes.
- The original's backend has the same form as Wan's `tokenizer/` (the score difference is a last-ulp
  notation); accepted inputs give Wan's id sequences (50 fixtures, 218 Wan parity cases, 4,563 sweep
  inputs, no mismatch, through transformers 5.14.1 and `tokenizer.json`).
- **Acceptance set without preprocessing**: Wan's rejection rules (ADR 0119 decisions 1 and 4) plus
  **whitespace other than U+0020 and a trailing U+0020**, because without preprocessing transformers
  5.14.1 and `tokenizer.json` disagree on those forms (`"a\tb"`: `[289, 748, 1]` vs
  `[289, 3, 485, 1]`; `"a "`: `[289, 1]` vs `[289, 273, 1]`). No trimming or whitespace folding is
  added (it would be a second preprocessing). On Wan's path preprocessing folds whitespace first, so
  the rejection never fires there.
- **TS entry**: subpath `@karume/models/umt5` (also in the barrel), `Umt5Encoder` with
  `fromPretrained(ref)` / `fromAssets` / `tokenize(text)` /
  `encode(text, { signal }) → { inputIds, hiddenStates: [L, 4096] f32 row-major, length, hiddenSize }`
  / `dispose`. The Session is resident from construction to `dispose` (i8 weights about 5.3 GiB);
  Wan's text stage still builds and tears down its Session per stage. Admission of pipeline
  `umt5-encoder/1` checks the tokenizer asset declaration before fetching weight parts. Intake-derived
  mirrors (no tokenizer) remain usable only through Wan's component replacement.
- **One shared implementation**: the umT5 host code (bucket table, Session I/O, graph contract check,
  tokenizer acceptance set) lives in `packages/models/src/umt5/`, and Wan calls it (Wan's tokenizer =
  preprocessing → `Umt5Tokenizer`), so the rejection rules cannot drift apart.
- The published source pin is `UMT5_SOURCES["umt5-xxl"]` in `packages/models/src/umt5/config.ts`
  (`hdae/karume-umt5-xxl`), added at the 0.14.0 publication (ADR 0073 decision 1; release-runbook
  "first publication of a family").
- Verification: the GPU e2e of `Umt5Encoder` (same goldens and band as Wan's umT5 port check, all 10
  within the band, `tokenize` ids equal to the golden; RTX 3080 Ti, 2026-10-15) now lives merged into
  `packages/e2e/tests/e2e_wan_umt5_test.ts`; the tokenizer parity test is
  `packages/models/tests/umt5_tokenizer_test.ts` (17 failures under fault injection on the whitespace
  rule).
