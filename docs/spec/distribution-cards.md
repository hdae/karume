# Distribution repositories: the model card, NOTICE.md and LICENSE.md

What the distribution recipes write at the root of a distribution repository now: the model card (`README.md`), whose
front matter is the metadata the Hub reads, and the two legal texts `NOTICE.md` and `LICENSE.md`. The writers are the
recipes' card builders and root files (`tools/export-recipes/<family>/`), the generic card pieces of
`karume.modelcard` and the root-file rules of `karume.dist` (`tools/exporter/`); the driver
`tools/export-recipes/dist.py` composes them. Nineteen repositories are published from its seventeen pipelines
(`RELEASE_REPOSITORIES` in `tools/export-recipes/tests/test_dist_driver.py`).

A rule below holds for every published repository. Where families write a part differently, the difference is stated
as it is, naming the repositories. What has been decided and is not built yet is in the [backlog](../backlog.md)
item "The model cards and notices across repositories".

## Files at the repository root

- Every repository's root holds `karume.json` (the manifest), `README.md`, `LICENSE.md` and `NOTICE.md`. The three
  Anima repositories also hold `name-map.json` ([the name map](convert.md#the-name-map)). Everything else sits under a
  model's directory or `shared/`.
- The manifest and the card are the meta files (`META_PATHS` of `karume.dist`). The two legal texts are the only other
  files a pipeline may put at the root (`LEGAL_PATHS`, passed as `Pipeline.root_files`), and every pipeline of the
  driver passes both.
- The card is drawn by the pipeline's card renderer (`Pipeline.card_profiles`; every pipeline has one) from the
  manifest being written. The legal texts are fixed per pipeline, except where the notice names what only some
  models carry: then they follow the models built (`ROOT_FILES` of the driver) — Gemma 4's (`karume-gemma4`, whose
  drafters differ per model), SigLIP2's (a checkpoint per model) and the Irodori MIT line's, whose three repositories
  (`karume-irodori-v4-small`, `karume-irodori-v4.1-small`, `karume-irodori-v4.1-small-mf`) are one pipeline: each
  carries the same `LICENSE.md` and a `NOTICE.md` naming its own checkpoint and the repositories it references.
- The texts are written in American English with the ordinary terms: a quant preset is a "preset", never a "seat";
  "license", "normalization", "color", "meter". A quotation of an upstream keeps the upstream's words.
- Rendering is deterministic: the same manifest gives the same bytes. Sizes, the model list, the quant tables and the
  dtype labels are read from the manifest, never written by hand; what the manifest does not hold (upstream, license,
  provenance) is a constant of the recipe.

## The model card

### Front matter

- Drawn by `karume.modelcard.frontmatter`, in this key order: `library_name: karume`, `pipeline_tag`, `base_model`,
  `base_model_relation` (when written), `license`, `license_name` and `license_link` (when written), `tags`.
- `base_model` holds HF repository IDs, a scalar for one and a sequence for several: every upstream whose bytes the
  repository runs, stored here or fetched from another Karume repository — the checkpoints it converts and the
  separate upstream repositories of its components (a text backbone or encoder, a codec, a drafter, a VAE's origins,
  the repository a tokenizer or shared components come from). A work that is not on the Hub (Anima's community
  fine-tunes on Civitai) has no ID to list. What each repository lists:

  | Repositories | `base_model` lists |
  | --- | --- |
  | `karume-anima` | `circlestone-labs/Anima` (the four single-file variants) and `circlestone-labs/Anima-Base-v1.0-Diffusers` (the base and the shared text encoder, VAE and tokenizers) |
  | `karume-anima-extra` | `circlestone-labs/Anima-Base-v1.0-Diffusers` (the shared components, fetched from `karume-anima`) |
  | `karume-anima-2.9b` | `Gazingstars123/Anima-2.9B` and `circlestone-labs/Anima-Base-v1.0-Diffusers` (the shared components, fetched) |
  | the four Irodori repositories | the checkpoint, its text backbone (or encoder) and the codec, whether stored here or fetched |
  | `karume-wan2.2` | the checkpoint and `google/umt5-xxl` (the text encoder, fetched from `karume-umt5-xxl`); not `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, whose tokenizer files are this checkpoint's and whose encoder is the umT5-XXL one (the attribution names it as the source of the precomputed embeddings and the tokenizer table) |
  | `karume-moss-image` | the checkpoint, the text encoder's release and both VAE origins, `black-forest-labs/FLUX.2-klein-4B` (the latent statistics) included |
  | every other repository | the checkpoints it converts and the separate upstream repositories of what it stores |

- `base_model_relation` is `quantized` or left out. birefnet derives it from the default preset (`quantized` when its
  download stores values rounded from the upstream checkpoint), the regular Gemma 4 card from every stored dtype
  (`quantized` when one is not the checkpoint's bf16); siglip2, depth-anything and embeddinggemma2 leave it out; the
  other cards write `quantized`. Left out today: `karume-birefnet-hr`, `karume-lucida`, `karume-depth-anything-v2`,
  `karume-siglip2`, `karume-embeddinggemma-2`.
- `license` is an HF license identifier: `apache-2.0` (nine repositories), `mit` (`karume-birefnet-hr`,
  `karume-lucida` and the Irodori MIT line), `cc-by-sa-4.0` (`karume-sbv2-jvnv`), `gemma` (`karume-irodori-v4-large`),
  `other` (the three Anima repositories).
- `license_name` and `license_link` are written with `license: other`: the Anima repositories name
  `circlestone-labs-non-commercial-license` and link the license file at a pinned commit of `circlestone-labs/Anima`.
  Beside an SPDX `license`, `license_link` is written only as the link the upstream card itself writes, and
  `license_name` never: the three Gemma 4 repositories and `karume-embeddinggemma-2` write
  `license_link: https://ai.google.dev/gemma/docs/gemma_4_license` beside `license: apache-2.0`, the page Google's
  cards link (`_shared.upstream.GEMMA_LICENSE_PAGE`, the one constant every one of these recipes writes). The card's
  text names the license and links the license text; it does not link the page again.
- `tags` contains `webgpu`. It begins with the `pipeline_tag` value. Where `webgpu` stands among the family's tags is
  not fixed. No card has a `karume` tag (`library_name` carries it).

### Title and the non-endorsement sentence

- The card's one level-1 heading is `# <title> — Karume`.
- The first paragraph under it is the non-endorsement sentence, the same words on every card
  (`karume.modelcard.UNOFFICIAL_CONVERSION_NOTICE`, the house sentence in the code): "This is an unofficial
  conversion of {upstreams} for Karume. It is not affiliated with or endorsed by {authors}." Each upstream and each
  author is listed once, in order.
- The upstreams are the `base_model` IDs and each author is the account's publisher as `_shared.upstream.UPSTREAM_AUTHORS`
  spells it (`hf_upstreams`): the sentence follows the front matter. A redistributed work that is not on the Hub comes
  first, by its title and author — on `karume-anima-extra`, its two fine-tunes.
- Wherever else a card or a notice names the publisher of an HF upstream (the Anima origins, the MOSS-Image text
  encoder, the SBV2 notice), it spells it as `UPSTREAM_AUTHORS` does.

### Sections and their order

The level-2 sections come in this order:

1. `## What is this`.
2. The attribution: `## Base weights and attribution`. On the three Anima repositories it lists each model as a bullet
   (title, author, source and pin, the permissions its source page lists), then the license terms and the notices
   the CircleStone and NVIDIA licenses require in their own words.
3. `## Models` — the model table of `karume.modelcard.models`.
4. `## Usage`.
5. The family's own sections, where it has any: `## Long context` (`karume-gemma4-qat`); `## Prompts`, `## Accepted
   inputs`, `## Determinism and verification` and `## Resources` (`karume-wan2.2`).
6. One section per model, in manifest order: `## Model: <name>` (`karume.modelcard.model_sections`). Each opens with
   `### Quants`; the family's other level-3 sections follow.

### What is this

- Its first paragraph says what is distributed, converted into Karume's container format, "(a `.krm` part sequence
  whose first part carries the graph and model descriptors)".
- The runtimes it names are one sentence, `karume.modelcard.RUNTIME_SENTENCE` ("Runs in the browser, in Node (with a
  WebGPU implementation passed in) and in Deno."), written as a line of its own in the first paragraph, on every
  card.
- A bullet "Not readable by <library> (it's a different container with an embedded graph); the reader is a pipeline
  that implements `<pipeline>`." — on every card.
- A bullet "Exporter used for the conversion: `<generator>`. The distribution manifest is `karume.json`
  (`<format>`)." The format is spelled once, as `karume.migrate.card_format_statement` spells it, so the upgrade of a
  repository rewrites it in place.

### Naming and pinning an upstream

- Wherever a card names an HF repository — an upstream, or a Karume repository it fetches components from or is
  used by — it is a link with its repository ID as plain link text:
  `[<owner>/<name>](https://huggingface.co/<owner>/<name>)` (`karume.modelcard.hf_link`), the quant tables'
  cross-repository note included; the Anima origins link a Civitai source by its address as link text
  (`[civitai.com/models/…](…)`). No card or notice writes a URL without a link (no bare or angle-bracket URL), except
  the NVIDIA notice of the Anima texts, whose wording was ruled with its parenthesized addresses. The Gemma 4 cards
  still name some repositories in code font.
- A card pins every upstream it stores, and every upstream of a component it fetches from another Karume repository,
  at the full commit the bytes were taken from: `` [<owner>/<name>](…) at commit `<40 hex digits>` ``
  (`karume.modelcard.pinned`). The commits are the ones the recipes and their `THIRD_PARTY_NOTICES.md` ledgers
  record (a fetch record, the commit a checkpoint's own config names for its text encoder, a pinned source table).
  A Civitai source, which has no commits, is pinned by its model version ("(version `<id>`)"). Every `base_model`
  is pinned somewhere in the card, except on `karume-gemma4` and `karume-gemma4-qat`, which pin no upstream.
- A commit of another Karume repository (the manifest holds it in full) is written as its first 16 hex digits and
  "…", in the quant tables' note "Some components are fetched from … at commit … — those bytes are identical to this
  model's own, so they are not stored here a second time." (`karume.modelcard.quants`).

### License statements

- Every card names `LICENSE.md` and `NOTICE.md`.
- A card names every license it states by its HF identifier in bold: "licensed **apache-2.0**", "licensed **mit**",
  "licensed **cc-by-sa-4.0**", "licensed **gemma**", and for a custom license (`license: other`) its `license_name`,
  "licensed **circlestone-labs-non-commercial-license**". It refers to a text in `LICENSE.md` as a verbatim copy,
  not by the text's title. Two kinds of words keep a license's own name: a clause the card cites ("per Apache 2.0
  §4(b)", "license §2(e)") and a notice the card shows in a license's own words (Anima's). A further upstream under
  the same license is "licensed **apache-2.0** under the same terms" (the Gemma 4 drafters).
- Where the weights' license is the primary Apache text of `LICENSE.md`, "licensed **apache-2.0**" is followed by
  `karume.modelcard.APACHE_LICENSE_REFERENCE`: "(as of retrieval; [full text](https://www.apache.org/licenses/LICENSE-2.0)
  — a verbatim copy is in `LICENSE.md`)" (`karume-depth-anything-v2`, `karume-embeddinggemma-2`, the three Gemma 4
  repositories, `karume-moss-image`, `karume-siglip2`, `karume-umt5-xxl`, `karume-wan2.2`). No other card links the
  text.
- Every card carries one "**Changes made here**" bullet in the attribution section summarising the changes, headed
  by `karume.modelcard.changes_made_here`: "- **Changes made here** (listed in full in `NOTICE.md`, per <clause>):",
  the clause being the one of the license that asks for the notice of the changes — "Apache 2.0 §4(b)" on the Apache
  cards, "CC BY-SA 4.0 §3(a)(1)(B)" on `karume-sbv2-jvnv` — and none where the card cites none (the MIT, Gemma and
  CircleStone cards). `karume-gemma4` and `karume-gemma4-qat` still write "(also listed in `NOTICE.md`…)". The Anima
  cards' license paragraph says that `NOTICE.md` holds the attribution notices and the list of modifications.
- A sentence an upstream license requires in its own words is kept as it is, beside the non-endorsement sentence,
  which does not stand in for it: the CircleStone attribution and rights-grant notices and the NVIDIA notice (Anima),
  the Black Forest Labs non-endorsement (MOSS-Image).

### What the upstream says

- Every card ends its attribution section with one bullet, `karume.modelcard.upstream_cards`, sending the reader to
  the upstream model cards for what the model can and cannot do (capabilities, limitations, intended use, evaluations,
  safety guidance), which the repository does not restate. It links the card of every redistributed upstream: the
  `base_model` repositories, preceded on `karume-anima-extra` by the Civitai pages of its fine-tunes. The Gemma 4
  cards write the bullet in their own words: `karume-gemma4` and `karume-gemma4-qat` link the default model's upstream
  card only.
- What an upstream states and the recipe cannot measure — a statement of license or use policy, the training data,
  what the model is and what it is for — is quoted verbatim, never reworded, by `karume.modelcard.quotation`: a lead
  naming the quoted card, linked to its `README.md` at the commit the recipe pinned (`UpstreamQuote.source`), and the
  headings the text stands under (the card's title included, without emphasis or a leading emoji; none for a text
  before the first heading); a blank line; then the paragraphs as one blockquote. A text that follows a quotation is
  set apart by a blank line. A statement whose source at the pinned commit is not on the Hub card stays in the
  recipe's words (docs/backlog.md lists them).

  | Repository | Card quoted | What it states |
  | --- | --- | --- |
  | `karume-anima-2.9b` | `Gazingstars123/Anima-2.9B` | what the model is and what was trained ("What is this") |
  | `karume-birefnet-hr` | `ZhengPeng7/BiRefNet_HR` | the training resolution and the training set |
  | `karume-lucida` | `egeorcun/lucida` | what the model is ("What is this"), its training data, what the variant left out is for |
  | `karume-depth-anything-v2` | `depth-anything/Depth-Anything-V2-Small-hf` | how Depth Anything V2 was trained |
  | `karume-embeddinggemma-2` | `google/embeddinggemma-2` | the numerical precision to run it in, the use policy |
  | `karume-irodori-v4-small`, `karume-irodori-v4.1-small` | `Aratako/Semantic-DACVAE-Japanese-32dim` | where the codec comes from |
  | `karume-irodori-v4.1-small-mf` | `Aratako/Irodori-TTS-v4.1-Small-MF`, the codec's card | how it was distilled, where the codec comes from |
  | `karume-irodori-v4-large` | `Aratako/Irodori-TTS-v4-Large`, the codec's card | the Gemma terms and the ethical restrictions, where the codec comes from |
  | `karume-moss-image` | `black-forest-labs/FLUX.2-small-decoder`, `black-forest-labs/FLUX.2-klein-4B` | the pointers to Black Forest Labs' usage policies |
  | `karume-sbv2-jvnv` | `litagin/style_bert_vits2_jvnv` | that the corpus's license carries over to the voices (in Japanese, followed by an English translation labeled as Karume's) |

  Unless the table says otherwise, the quotation stands in the attribution section.
- Each family lists its quotations in `UPSTREAM_QUOTES` of its `card.py`. The driver's test checks each against the
  upstream card at its commit wherever a copy is on the machine (the HF cache, or an input directory whose download
  record or `REVISION.txt` names the commit), and checks that every blockquote of a card is one of them.
- A notice does not repeat a quotation. Where an upstream's statement is the basis of what the notice says, the notice
  keeps one line stating that basis and refers to the card ("`README.md` quotes it"): the voices' license
  (`karume-sbv2-jvnv`), the Gemma terms of the checkpoint (`karume-irodori-v4-large`), the use policy
  (`karume-embeddinggemma-2`). `karume-anima-2.9b`'s notice refers to the upstream card for how the model was made.

### Presets and the quant tables

- The model table and every quant table come from `karume.modelcard.models` and `quants`: one model table, one
  quant table per model. A preset's "What it is" cell is its `label` and `description` from the manifest; sizes are the
  download of that preset.
- A preset's description says what the preset is: its storage, its knobs and what follows from them, such as the
  download. It states nothing the preset measured.
- What a preset measured (its speed, the quality heard or seen) is a note under its model's quant table:
  `quants(measured=…)` writes `karume.modelcard.measurements` after the table's notes (on `karume-sbv2-jvnv`,
  after the notes on how the stored weights were rounded). The notes open with `MEASUREMENTS_LEAD`; each bullet
  (`Measurement`) names the preset and the result, then in parentheses what was measured when it was not this
  model, the device (with the runtime where it was recorded) and the date. A note on a preset the model does not
  ship is left out. `model_measurements` picks the notes of a one-model manifest's model from a recipe table, and
  fails on a manifest of several models that the table names.

  | Recipe table | Repositories | What it holds |
  | --- | --- | --- |
  | `irodori.card.IRODORI_MEASUREMENTS` | `karume-irodori-v4-small`, `karume-irodori-v4.1-small`, `karume-irodori-v4.1-small-mf` | v4 Small's listening check of `f16` and wall time of `i8-a8`, which v4.1-small carries naming v4 Small as what was measured; the MeanFlow student's listening of `i8` and `i8-a8`. None for v4-Large |
  | `sbv2.card.SBV2_MEASUREMENTS` | `karume-sbv2-jvnv`, under every voice | the simulated audio difference of `f16+bert8`, the text encoder stage of `i8-a8`, the listening checks of `i8+bert4` and `i4` and the warm synthesis of `i4`, each but the text encoder's naming another voice as what was measured |
  | `WanCard.measurements` (`wan.card`) | `karume-wan2.2` | the transformer pass of the online preset and the VAE decode of the conv8 preset |

  The other cards write no measurement notes. The Gemma 4 repositories' descriptions keep their measured words
  until their stage.

### Usage

- `## Usage` opens with a ```` ```ts ```` block that imports from `jsr:@karume/models/<family>` and calls
  `fromPretrained` with the object form `{ repo: "<owner>/<name>", … }` and the commented revision pin
  `karume.modelcard.REVISION_PIN_COMMENT` right after the `repo` line, once per card. The repository is the one the
  card is uploaded to (`resolve_repo`).
- Every card writes the call through `karume.modelcard.from_pretrained`, declared with `await using` or `using` as
  the family's dispose is, with the model and the quant (and the family's other knobs) as commented options. The
  three Gemma 4 cards share one usage section (`gemma4.card.gemma4_usage`): the regular and q4_0 QAT lines import
  from `jsr:@karume/models/gemma`, the mobile QAT line from `jsr:@karume/models/gemma4-qat-mobile`.

## NOTICE.md

- Sixteen notices open with `# NOTICE`, then a paragraph naming what is redistributed and its license by the license's
  full title ("the Apache License, Version 2.0", "the MIT License", "Creative Commons Attribution-ShareAlike 4.0
  International", "the Gemma Terms of Use"), with parts taken from other releases as bullets or paragraphs; then
  "The following changes were made:" and a
  bullet list; then a closing paragraph beginning "No retraining and no fine-tuning were performed." that says the
  original checkpoint is not distributed here, on one line (`karume.modelcard.notice_closing`).
- Every license a notice names is named by its full title, never by a short name ("Apache 2.0", "CC BY-SA 4.0",
  "(MIT)").
- Statements a license or an upstream requires in its own words stand before the change list or after the closing
  paragraph, verbatim: the Gemma `NOTICE` line first in `karume-irodori-v4-large`, the Black Forest Labs
  non-endorsement in `karume-moss-image`, the share-alike statement after the closing in `karume-sbv2-jvnv`, the
  line on the upstream use policy after the closing in `karume-embeddinggemma-2`.
- The three Anima notices take the same structure, keeping every statement the CircleStone and NVIDIA licenses
  require: after the opening paragraph, the rights-grant statement of CircleStone §3(a) and the Attribution Notice of
  §3(b) verbatim, followed at once by the statement of §3(d)(i) that the model was modified ("As required by the
  license, this Attribution Notice also states that …") and "The following changes were made:" with the list; after
  the closing paragraph, the non-endorsement statement of §3(d)(iii) and the NVIDIA notice (§3.1 / §3.2). The license
  prescribes the words of the Attribution Notice and that the statement of the modification is included in it, not
  a layout, so the notices have no headings of their own.
- A notice names each upstream — every `base_model` of its card — by its repository ID in code font
  (`` `<owner>/<name>` ``), without a link, and a Karume repository it references the same way
  (`` `hdae/<name>` ``); it does not point at `README.md` for an upstream's name. A Civitai fine-tune, which has no
  repository ID, is named by its title and its page address in code font (`karume-anima-extra`).

## LICENSE.md

- `LICENSE.md` is built by `_shared.licenses.license_markdown`. It begins at its first byte with the primary license
  text inside a ```` ```text ```` fence (`fenced`), the text inside byte for byte the file it is read from
  (`_shared/licenses` or the family's own license files); a text is never reformatted, and one without a final line
  break gets the line break that ends its last line before the closing fence (the CircleStone text). The three Gemma 4
  repositories are the exception: their `LICENSE.md` is the Apache text alone, unfenced.

  | Primary text | Repositories |
  | --- | --- |
  | Apache License 2.0 (`_shared/licenses/apache_license_2_0.txt`) | `karume-depth-anything-v2`, `karume-embeddinggemma-2`, the three Gemma 4 repositories, `karume-moss-image`, `karume-siglip2`, `karume-umt5-xxl`, `karume-wan2.2` |
  | MIT (`_shared/licenses/mit.txt` with the copyright lines, closest holder first) | `karume-birefnet-hr`, `karume-lucida`, the Irodori MIT line |
  | CC BY-SA 4.0 legal code (`_shared/licenses/cc_by_sa_4_0.txt`) | `karume-sbv2-jvnv` |
  | CircleStone Labs Non-Commercial License v1.2 (`anima/circlestone_license.txt`) | the three Anima repositories |
  | Gemma Terms of Use (`irodori/licenses/gemma_terms_of_use.txt`) | `karume-irodori-v4-large` |

  The MIT copyright lines are the only inserted text (`mit_license`).
- A further text (`FurtherLicense`) follows a `---` line, a `## <license> — <scope>` heading and a paragraph saying
  which components it covers, then the text fenced the same way: the NVIDIA Open Model License Agreement (Anima), the
  Apache License 2.0 for the codec (the Irodori MIT line), and on `karume-irodori-v4-large` the Gemma Prohibited Use
  Policy (part of the Gemma Terms of Use, which incorporate it), then the MIT License and the Apache License 2.0 for
  the codec.

## Shared wording

One constant, imported where it is written:

- In `karume.modelcard`: the non-endorsement sentence, the runtime sentence, the upstream-card bullet, the link and
  pin of an upstream (`hf_link`, `pinned`), the Apache text's address and the words after an Apache license
  (`APACHE_LICENSE_TEXT_URL`, `APACHE_LICENSE_REFERENCE`), the quotation of an upstream card (`UpstreamQuote`,
  `quotation`), the revision-pin comment, the closing paragraph of `NOTICE.md` (`NOTICE_CLOSING`,
  `notice_closing`), the notes under the quant tables (the measurement notes among them:
  `Measurement`, `measurements`, `MEASUREMENTS_LEAD`) and the cross-repository note.
- In `_shared`: the license texts and the layout of `LICENSE.md` (`_shared/licenses`), the publishers' spelling
  (`_shared.upstream.UPSTREAM_AUTHORS`) and the Gemma license page (`_shared.upstream.GEMMA_LICENSE_PAGE`).
- Within a family, for its card and its notice: Anima's attribution, rights-grant and NVIDIA notices; MOSS-Image's
  Black Forest Labs non-endorsement (`moss.card.MOSS_BFL_NON_ENDORSEMENT`). Each quotation is one `UpstreamQuote` of
  its family's card module.

Everything else above (the "What is this" sentences among it) is written per family.
