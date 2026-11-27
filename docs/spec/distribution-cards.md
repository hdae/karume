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
  manifest being written. The legal texts are fixed per pipeline, except Gemma 4's (`karume-gemma4`), which follow
  the models built (`ROOT_FILES` of the driver). The three repositories of the Irodori MIT line
  (`karume-irodori-v4-small`, `karume-irodori-v4.1-small`, `karume-irodori-v4.1-small-mf`) are one pipeline, so they
  carry the same `NOTICE.md` and the same `LICENSE.md`.
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
  cards link (`_shared.upstream.GEMMA_LICENSE_PAGE`; the Gemma 4 recipes still spell it as their own constant).
- `tags` contains `webgpu`. It begins with the `pipeline_tag` value, except on `karume-gemma4-qat-mobile`
  (`webgpu`, `gemma`, `text-generation`). Where `webgpu` stands among the family's tags is not fixed. No card has a
  `karume` tag (`library_name` carries it).

### Title and the non-endorsement sentence

- The card's one level-1 heading is `# <title> — Karume`.
- The first paragraph under it is the non-endorsement sentence, the same words on every card
  (`karume.modelcard.UNOFFICIAL_CONVERSION_NOTICE`, the house sentence in the code): "This is an unofficial
  conversion of {upstreams} for Karume. It is not affiliated with or endorsed by {authors}." Each upstream and each
  author is listed once, in order.
- The upstreams are the `base_model` IDs and each author is the account's publisher as `_shared.upstream.UPSTREAM_AUTHORS`
  spells it (`hf_upstreams`): the sentence follows the front matter. A redistributed work that is not on the Hub comes
  first, by its title and author — on `karume-anima-extra`, its two fine-tunes.

### Sections and their order

The level-2 sections come in this order:

1. `## What is this` — except on `karume-gemma4-qat-mobile`, which has four unheaded lines in its place.
2. The attribution: `## Base weights and attribution`, or, on the three Anima repositories, `## Models and their
   origins` (one `### `<model>` — <title>` block per model) followed by `## License`.
3. `## Models` — the model table of `karume.modelcard.models`.
4. `## Usage` — except on `karume-gemma4-qat-mobile`.
5. The family's own sections, where it has any: `## Long context` (`karume-gemma4-qat`); `## Prompts`, `## Accepted
   inputs`, `## Determinism and verification` and `## Resources` (`karume-wan2.2`).
6. One section per model, in manifest order: `## Model: <name>` (`karume.modelcard.model_sections`), or `## <name>`
   on `karume-gemma4-qat-mobile`. Each opens with `### Quants`; the family's other level-3 sections follow.

### What is this

- Its first paragraph says what is distributed, converted into Karume's container format, "(a `.krm` part sequence
  whose first part carries the graph and model descriptors)".
- The runtimes it names are one sentence, `karume.modelcard.RUNTIME_SENTENCE` ("Runs in the browser, in Node (with a
  WebGPU implementation passed in) and in Deno."), written as a line of its own in the first paragraph, on every card
  but the three Gemma 4 ones: `karume-gemma4` and `karume-gemma4-qat` say "Runs as-is in the browser and in Deno",
  and `karume-gemma4-qat-mobile` names no runtime.
- A bullet "Not readable by <library> (it's a different container with an embedded graph); the reader is a pipeline
  that implements `<pipeline>`." — on every card with this section except `karume-umt5-xxl`'s.
- A bullet "Exporter used for the conversion: `<generator>`. The distribution manifest is `karume.json`
  (`<format>`)." The format is spelled once, as `karume.migrate.card_format_statement` spells it, so the upgrade of a
  repository rewrites it in place. `karume-gemma4-qat-mobile`'s card does not state the format.

### Naming and pinning an upstream

- In the attribution section an HF upstream is a link with its repository ID as plain link text:
  `[<owner>/<name>](https://huggingface.co/<owner>/<name>)` (`karume.modelcard.hf_link`). The Anima origins link a
  model's source the same way, and a Civitai source by its address as link text (`[civitai.com/models/…](…)`). A
  Karume repository named as the source of fetched components is a link with the ID in code font: the quant tables'
  note, and the `karume-wan2.2` and `karume-umt5-xxl` cards' mentions of each other.
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

- The card names `LICENSE.md` and `NOTICE.md`, except `karume-sbv2-jvnv`'s, which names neither.
- A card names every license it states by its HF identifier in bold: "licensed **apache-2.0**", "licensed **mit**",
  "licensed **cc-by-sa-4.0**", "licensed **gemma**", and for a custom license (`license: other`) its `license_name`,
  "licensed **circlestone-labs-non-commercial-license**". It refers to a text in `LICENSE.md` as a verbatim copy,
  not by the text's title. Two kinds of words keep a license's own name: a clause the card cites ("per Apache 2.0
  §4(b)", "license §2(e)") and a notice the card shows in a license's own words (Anima's). The three Gemma 4 cards
  are the exception: they write "licensed **Apache 2.0**", with the license page and the text linked.
- Sixteen cards (all but the three Anima ones) carry a "**Changes made here**" bullet in the attribution section
  summarising the changes. Its pointer differs: "(also listed in `NOTICE.md`…)" on eleven cards, "(listed in full in
  `NOTICE.md`…)" on `karume-gemma4-qat-mobile`, `karume-moss-image`, `karume-umt5-xxl` and `karume-wan2.2`, none on
  `karume-sbv2-jvnv`. Every Apache 2.0 card adds "per Apache 2.0 §4(b)". The Anima cards' `## License` paragraph
  says that `NOTICE.md` holds the attribution notices and the list of modifications.
- A sentence an upstream license requires in its own words is kept as it is, beside the non-endorsement sentence,
  which does not stand in for it: the CircleStone attribution and rights-grant notices and the NVIDIA notice (Anima),
  the Black Forest Labs non-endorsement (MOSS-Image).

### What the upstream says

- Four cards end the attribution section with a bullet sending the reader to the upstream model card for what the
  model can and cannot do (capabilities, limitations, intended use, evaluations, safety guidance), which the
  repository does not restate: `karume-gemma4`, `karume-gemma4-qat`, `karume-gemma4-qat-mobile`,
  `karume-embeddinggemma-2`. `karume-gemma4` and `karume-gemma4-qat` link the default model's upstream card.
- `karume-birefnet-hr` and `karume-depth-anything-v2` send the reader to the upstream card for the training data. The
  other cards link the upstream in the attribution section only.
- `karume-anima-2.9b` quotes the upstream card's description verbatim, and its `NOTICE.md` refers to the upstream card
  for it. `karume-embeddinggemma-2` quotes the upstream's use-policy sentence verbatim, in the card and in `NOTICE.md`
  (one constant). Other upstream statements are written in the recipe's own words.

### Presets and the quant tables

- The model table and every quant table come from `karume.modelcard.models` and `quants`: one model table, one
  quant table per model. A preset's "What it is" cell is its `label` and `description` from the manifest; sizes are the
  download of that preset.
- What a preset description says about speed or quality is the recipe's text in the manifest; it follows no rule
  of this spec.

### Usage

- `## Usage` opens with a ```` ```ts ```` block that imports from `jsr:@karume/models/<family>` and calls
  `fromPretrained` with the object form `{ repo: "<owner>/<name>", … }` and the commented revision pin
  `karume.modelcard.REVISION_PIN_COMMENT` right after the `repo` line, once per card. The repository is the one the
  card is uploaded to (`resolve_repo`).
- Seventeen cards write the call through `karume.modelcard.from_pretrained`, declared with `await using` or `using`
  as the family's dispose is. `karume-umt5-xxl` writes it by hand (`const encoder = …`, with `await
  encoder.dispose()`). `karume-gemma4-qat-mobile` has no usage section.

## NOTICE.md

- Sixteen notices open with `# NOTICE`, then a paragraph naming what is redistributed and its license by the license's
  full title ("the Apache License, Version 2.0", "the MIT License", "Creative Commons Attribution-ShareAlike 4.0
  International", "the Gemma Terms of Use"), with parts taken from other releases as bullets or paragraphs; then
  "The following changes were made:" and a
  bullet list; then a closing paragraph beginning "No retraining and no fine-tuning were performed." that says the
  original checkpoint is not distributed here.
- Every license a notice names is named by its full title, never by a short name ("Apache 2.0", "CC BY-SA 4.0",
  "(MIT)").
- Statements a license or an upstream requires in its own words stand before the change list or after the closing
  paragraph, verbatim: the Gemma `NOTICE` line first in `karume-irodori-v4-large`, the Black Forest Labs
  non-endorsement in `karume-moss-image`, the share-alike statement after the closing in `karume-sbv2-jvnv`, the
  upstream use-policy quotation after the closing in `karume-embeddinggemma-2`.
- The three Anima notices open with `# Notice` and follow the CircleStone license's layout: `## Attribution Notice`
  (the notice, then the modifications after "modified as follows:"), `## Not an official product`, `## NVIDIA Open
  Model License`, `## License`. They have no closing paragraph.
- How a notice names the upstream differs: its repository ID in code font (`karume-birefnet-hr`, `karume-lucida`,
  `karume-embeddinggemma-2`, the three Gemma 4 repositories, `karume-umt5-xxl`); "the … checkpoint listed in
  `README.md`" (`karume-depth-anything-v2`, `karume-siglip2`, `karume-moss-image`, `karume-wan2.2`, the four Irodori
  repositories); an angle-bracket URL (`karume-sbv2-jvnv`); prose names and bare URLs (the three Anima repositories).

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

- In `karume.modelcard`: the non-endorsement sentence, the runtime sentence, the revision-pin comment, the notes under
  the quant tables and the cross-repository note.
- In `_shared`: the license texts and the layout of `LICENSE.md` (`_shared/licenses`), the publishers' spelling
  (`_shared.upstream.UPSTREAM_AUTHORS`) and the Gemma license page (`_shared.upstream.GEMMA_LICENSE_PAGE`).
- Within a family, for its card and its notice: Anima's attribution, rights-grant and NVIDIA notices; EmbeddingGemma 2's
  use-policy sentence.

Everything else above (the Apache text URL, the Gemma 4 license page, the notices' closing paragraph, the "What is
this" sentences, the upstream-card bullet) is written per family.
