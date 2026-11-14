# Irodori-TTS export recipe (with the DACVAE codec)

**Outside the wheel** — this recipe is repo-only (ADR
[0065](../../../docs/decisions/0065-exporter-core-recipe-split.md)) and the authoritative model card
is the exemplary `README.md` that `dist` generates into the distribution directory.

Upstream provenance: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) — the codec
(`irodori/dacvae/`) is a **different upstream repository under a different license**, kept in one
directory with the family it ships with. Scripts are started as modules from
`tools/export-recipes/` (`uv run --with 'transformers==5.14.1' python -m irodori.export …`); the
generic emit, verify and dist
contracts live in [`../../exporter/README.md`](../../exporter/README.md).

## Real-weight Irodori-TTS export and E2E (waves 1–4)

Seven text-side graphs plus the DACVAE codec pair, with host-side goldens. The numbering below names
the steps used throughout this file; the scripts read each other's outputs, so a **clean tree** has
to run them in the order given at the end of this section, not in the listed one:

```sh
# 0. one-time inputs: inputs/irodori/{v4-small,v4.1-small,dacvae-32dim,Irodori-TTS,dacvae-src}/
uv run python -m irodori.dacvae.convert                                  # 1. codec pth → safetensors
uv run --with 'transformers==5.14.1' python -m irodori.export            # 2. seven graphs + io goldens
uv run --with 'transformers==5.14.1' python -m irodori.tokenizer_ref     # 3. tokenizer asset + goldens + parity fixture (vp fmt it)
uv run --with 'transformers==5.14.1' python -m irodori.pipeline_ref      # 4. full-loop latent goldens
uv run --with descript-audiotools --with einops \
    --with 'transformers==5.14.1' python -m irodori.dacvae.host          # 5. host preprocessing goldens
uv run --with descript-audiotools --with einops python -m irodori.dacvae.export  # 6. codec graphs + io goldens
uv run python dist.py --pipeline irodori                                 # 7. distribution (9 graphs + tokenizer)
```

Step 7 does **not** place the containers of the published models from these series. A fresh export
cannot reproduce the published bytes — the published non-DiT containers predate the
`provenance.notice` field, and the GPTQ i4 rounding differs between machines — so v4-small and
v4.1-small take every graph role as the **published container** from the local mirror of the
repository that holds it (`IRODORI_PUBLISHED`), and the build fails unless each one is the published
bytes (`IRODORI_PUBLISHED_PART0` pins part 0, whose descriptor pins the other parts). Only the
tokenizer asset comes from the series. A reassembly therefore changes non-container files only (the
tokenizer's rare-token split table, the card, the manifest's `generator`); a mirror that fails the
check is restored from the Hub, and a deliberate re-export of a model's containers is a change to
those two tables.

The distribution carries **f16 and i8 weight-storage series** next to f32 (ADR 0050), plus an i4
series for the DiT's two graphs (below), so step 7 needs f32 / f16 / i8 (all nine roles) and i4
(`dit` and `dit-context`).
Regenerate the f16 and i8 sides with the same three scripts (order caveats are identical; the
codec / full-loop inputs stay on the f32 series on purpose — inputs are dtype-neutral):

```sh
uv run --with 'transformers==5.14.1' python -m irodori.export --dtype f16     # 2'. seven graphs
uv run --with 'transformers==5.14.1' python -m irodori.pipeline_ref --dtype f16   # 4'. full-loop goldens
uv run --with descript-audiotools --with einops python -m irodori.dacvae.export --dtype f16  # 6'. codec
uv run --with 'transformers==5.14.1' python -m irodori.export --dtype i8      # 2'''. seven graphs
uv run --with 'transformers==5.14.1' python -m irodori.pipeline_ref --dtype i8    # 4'''. full-loop goldens
uv run --with descript-audiotools --with einops python -m irodori.dacvae.export --dtype i8  # 6'''. codec
```

The `i8+dit4` quant seat adds an **i4 series for the DiT only** — its two graphs, `dit` and
`dit-context` (the conditioning keys / values, computed once per generation — ADR 0114); the other
seven roles share the i8 bytes, so nothing else is re-exported for it. Only the DiT block weights
outside the adaLN modulation are stored as i4 (168 linears: the 72 conditioning key / value
projections in `dit-context`, the other 96 in `dit`); the 144 adaLN linears (`attention_adaln`,
`mlp_adaln`) and the five linears outside the blocks (`in_proj`, `out_proj`, `cond_module.{0,2,4}`)
are stored as i8 in the `dit` container — both exclusions come from listening judgements. The
rounding runs once on the whole DiT and the two containers are written from that one result, with
the same `calib_provenance.json` next to each. Step 7 requires this series; the
rounding is GPTQ-calibrated by default and takes about 20–25 minutes of CPU time (twelve
calibration cases × the full reference loop — 1,447 s for v4-small and 1,183 s for v4.1-small,
measured 2026-09-05):

```sh
uv run --with 'transformers==5.14.1' python -m irodori.export --dtype i4   # 2''. dit + dit-context, calibrated
uv run --with 'transformers==5.14.1' python -m irodori.pipeline_ref --dtype i4  # 4''. full-loop goldens
```

Step 4'' **reads the two containers step 2'' wrote** (it is the only series whose goldens are baked
from the shipped bytes rather than from a second calibration run), so it must follow 2''. Step 2''
captures the calibration activations through upstream's DiT block and follows its `context_plan`
argument, so it needs an upstream clone (`inputs/irodori/Irodori-TTS/`) at `89f9d8f` or later.

`--no-calib` swaps the calibration for plain RTN. It exists for smoke runs only: the storage form is
byte-identical either way, so `dist` — and step 4'' — refuse the result by reading the
`calib_provenance.json` the export writes next to the container.

Order caveats measured in practice: step 2 reads step 5's real latent for the speaker cases
(`SPEAKER_REAL_CASES`) and exits when it is missing, steps 5 and 6 read step 4's `z`, and both of
them load the codec step 1 converts. A **full** rebuild from scratch therefore runs
**1 → 4 → 5 → 2 → 3 → 6 → 7**, not the listing order above (that one stops on the first script).
The i4 series is the one place where the dependency runs the other way (step 4'' reads what step 2''
wrote, above). Incremental regeneration of a single script is safe as long as its inputs above
exist. Design records: ADR 0044 / 0046 / 0047 / 0114 (graphs), 0048 (host port), 0049 (codec
integration).

## Another model of the same architecture (v4.1-small)

One distribution repository per model. The text-side scripts (steps 2 / 3 / 4 and the dtype
variants) all take `--model-dir inputs/irodori/<model>/` and derive the series names from that
directory name, so exporting e.g. `v4.1-small` is the same chain with `--model-dir
inputs/irodori/v4.1-small` — plus `--model v4.1-small` on step 7 (the output directory
`models/karume-irodori-v4.1-small/` follows). The codec series (`dacvae-32dim*`) is shared across
models and its inputs are model-independent, so steps 1 / 5 / 6 are **not** re-run when they
already exist. `card.py`'s `IRODORI_UPSTREAMS` must know the model name (upstream repository +
display name), or the card render — and with it step 7 — fails loudly.

v4.1-small does not store what it shares with v4-small: upstream v4.1 retrained only the duration
predictor, so 26 of its roles (90 files, 5,637 MiB — every graph role in f32 / f16 / i8 except
`duration`, plus `dit_context` i4 and the tokenizer) are byte-identical to v4-small's, and the
recipe declares them as a cross-repository reference to `karume-irodori-v4-small`
(`irodori_references` — ADR 0038 §7 addendum 2026-10-28). `duration_*` and `dit_i4` stay in
v4.1-small's own repository: references are per container, and the i4 DiT is calibrated through
each model's own reference loop, so its bytes differ. Step 7 for v4.1-small therefore reads the
two mirrors (its own published containers from `models/karume-irodori-v4.1-small/`, the shared ones
from `models/karume-irodori-v4-small/`), needs only the tokenizer step (3) of its own series, and
needs a revision source — `--resolve-refs hf` for the published v4-small (the build checks every
borrowed file on the Hub), or `--allow-placeholder-ref` for a development mirror:

```sh
uv run --with 'transformers==5.14.1' python -m irodori.tokenizer_ref \
    --model-dir ../../inputs/irodori/v4.1-small --fixture-out ../../outputs/misc/parity-v4.1.json  # 3
uv run python dist.py --pipeline irodori --model v4.1-small --resolve-refs hf                     # 7
```

v4-small stays self-contained. A local reader of v4.1-small (`nodeDirectory`) needs both mirrors
and a `crossRepo` mapping of `hdae/karume-irodori-v4-small` to the v4-small one.

## A MeanFlow student (v4.1-small-mf — ADR 0150)

Upstream's `Aratako/Irodori-TTS-v4.1-Small-MF` (place it in `inputs/irodori/v4.1-small-mf/`) is v4.1-small distilled
with MeanFlow: its checkpoint config says `flow_parameterization: "meanflow"`, its DiT has one more branch
(`delta_cond_module`, embedding the length of the sampling interval) and it samples in 4 steps without CFG
(`irodori_tts/meanflow.py`). Everything keys off that config value, never off the directory name: `export.py` writes a
step graph `dit` with the extra input `delta_embed` (after `t_embed`) and golden cases with an interval each,
`pipeline_ref.py` runs the MeanFlow loop and compares it with upstream's `sample_euler_meanflow`, `calib.py` captures
the i4 calibration from that loop (12 cases × 4 steps — about 2.5 minutes of CPU), and `distribution.py` declares the
MeanFlow `pipelineConfig` (no CFG knob).

Only the DiT pair is new, so only it is exported — the other seven graphs are the published v4-small / v4.1-small
containers, which the distribution borrows across repositories (`IRODORI_PUBLISHED`: `duration_*` from v4.1-small, the
rest from v4-small), and the tokenizer asset is referenced at v4-small (`IRODORI_TOKENIZER_HOLDERS` — the build checks
that the student's own asset, rare-token split table included, is byte-identical to v4-small's). Step 7 reads those
containers from the two mirrors (`models/karume-irodori-v4-small/`, `models/karume-irodori-v4.1-small/`) and checks that
the student's checkpoint tensors behind every borrowed graph, and its configs, equal the owner's (`inputs/irodori/v4-small/`
and `inputs/irodori/v4.1-small/` must be present):

```sh
D=../../inputs/irodori/v4.1-small-mf
for dtype in f32 f16 i8; do
  uv run --with 'transformers==5.14.1' python -m irodori.export --model-dir $D --dtype $dtype \
      --target dit --target dit-context                                    # 2 / 2' / 2'''
done
uv run --with 'transformers==5.14.1' python -m irodori.export --model-dir $D --dtype i4        # 2''
uv run --with 'transformers==5.14.1' python -m irodori.tokenizer_ref --model-dir $D \
    --fixture-out ../../outputs/misc/parity-mf.json                        # 3 (the fixture equals v4-small's)
for dtype in f32 f16 i8 i4; do
  uv run --with 'transformers==5.14.1' python -m irodori.pipeline_ref --model-dir $D --dtype $dtype  # 4 / 4' / 4'' / 4'''
done
uv run python dist.py --pipeline irodori --model v4.1-small-mf --resolve-refs hf             # 7
```

The export writes no `--target` other than the DiT pair for this model (the reference computations still run in
full). A local reader (`nodeDirectory`) needs the two borrowed-from mirrors too, mapped through `crossRepo`.

## The graphs of v4-Large (step 2)

Upstream's `Aratako/Irodori-TTS-v4-Large` (place it in `inputs/irodori/v4-large/`) keeps the tensor names and the
structure of the DiT, speaker encoder, duration predictor and projectors — only their dimensions change, and the
export reads every one of them from the checkpoint's own config. What is new is the text backbone: T5Gemma2's text
encoder (`google/t5gemma-2-1b-1b`, 26 layers, `[1,T,1152]`), built from the config embedded in the checkpoint (no
Hub access — that repository is gated). The export recognizes it by that config's `model_type` (the same test
upstream makes), never by the directory name, and writes the same seven graphs to `outputs/series/irodori-v4-large/`:

```sh
uv run --with 'transformers==5.14.1' python -m irodori.export --model-dir ../../inputs/irodori/v4-large  # 2
```

- The `backbone` graph takes `input_ids [1,T]` only (T up to 512). Its sliding layers mask a bidirectional window of
  256 to the left and 257 to the right (it cuts from T = 257 on), built from the model's own window function and
  folded into one Tmax constant; the full layers take no mask. Besides the usual checks the export measures that the
  window reaches the output (T 40 / 256 / 257 / 300 / 512 against full attention in the sliding layers), that the
  wrapper matches upstream bit for bit at those lengths, and that the folded constant is transformers' own mask
  (`window_evidence` / `window_equivalence_max_abs` / `window_form` in the printed summary).
- The token embedding stays in the graph (`embedding` over the f32 `[262144, 1152]` table, 1,152 MiB); its
  end-of-image row is selected with `where` (patch 5 of `patch.py`).
- The golden cases are the backbone's own table (`T5GEMMA2_GOLDEN_CASES`): the tokenizer splits the texts differently,
  and the caption side adds `caption-window` (T 388), `caption-long` cut to the bound by upstream's rule (T 512) and
  `caption-eoi` (the end-of-image id).
- Measured on 2026-11-14 (CPU): 134.6 s wall, 30.0 GiB peak RSS for the f32 series (13.2 GB checkpoint). Step 2 still
  reads the codec's real latent (`dacvae-32dim/host`, step 5), which is model-independent.
- Only the f32 series exists so far (the graphs here, the tokenizer asset and the full-loop goldens — the two
  sections at the end of this file). The f16 / i8 / i4 series and the distribution of this model are later steps.

## Rare-token split (ADR 0149)

Upstream (`irodori_tts/token_split.py`, on by default) encodes the **text** — never the caption — of
models whose `text_tokenizer_repo` is `sbintuitions/modernbert-ja-310m` without the rare long tokens
of that vocabulary, which the model tends to misread. Step 3 does not restate that rule: it runs
upstream's `RareTokenSplitter` on the real tokenizer and writes the result into the tokenizer asset
as one key written last (`rareTokenSplit` — the disabled ids, their replaced score and the forced id
splits), so the asset without it stays byte-identical. Before writing, it checks that a tokenizer
rebuilt from `tokenizer.json` with only that table reproduces the splitter, and it adds a split-on /
split-off golden (`encode.split` of the parity fixture) taken through upstream's own entry. v4-small,
v4.1-small and v4.1-small-mf share the tokenizer, so the table is the same for all three; the
manifests of v4.1-small and v4.1-small-mf pin v4-small's tokenizer bytes by their cross-repository
reference, so the table reaches them only when they are assembled against the v4-small revision
that carries it (`docs/release-runbook.md` §0 has the upload order).

Step 3 imports `token_split.py` from `--source-dir`, so the clone must include upstream commit
`82c9980b` or later.

## The tokenizer of v4-Large (step 3)

Upstream's `Aratako/Irodori-TTS-v4-Large` (place it in `inputs/irodori/v4-large/`) ships the
SentencePiece-style BPE tokenizer of `google/t5gemma-2-1b-1b` in its own `tokenizer/` directory
(the small models use a Unigram one). Step 3 picks the path from that `tokenizer.json` itself
(`model.type`), never from the directory name, and reads nothing from the Hub:

```sh
uv run --with 'transformers==5.14.1' python -m irodori.tokenizer_ref --model-dir ../../inputs/irodori/v4-large
vp fmt packages/models/tests/fixtures/irodori-text/v4-large-parity.json    # from the repository root
```

- The asset `outputs/series/irodori-v4-large/tokenizer/tokenizer.json` is the shared Gemma format
  (`karume-gemma-tokenizer/1`, compiled by `_shared/gemma_tokenizer.py`), not the Unigram one; the
  goldens next to it keep their names. Its `spec.postProcessor` is `bos` (T5Gemma 2 prepends
  `<bos>` only — a shape gemma4 and EmbeddingGemma do not have).
- The parity fixture is `packages/models/tests/fixtures/irodori-text/<weight directory>-parity.json`,
  so the small models' `parity.json` is left alone. It holds the asset subset, the encode / batch /
  caption goldens and the BOS / pad ids; the normalize golden and the NFKC table stay in
  `parity.json` (they do not depend on the tokenizer).
- No rare-token split: upstream applies it only to `sbintuitions/modernbert-ja-310m` and only to a
  Unigram tokenizer.
- The pad id is the added token `<pad>` (the Gemma format has no `padId` field); the step checks
  that it and the BOS agree with the checkpoint config and with upstream's wrapper.
- The caption cases add one caption longer than `max_text_len` (256) and one that `max_caption_len`
  (512) truncates.

## The full-loop goldens of v4-Large (step 4)

v4-Large is an RF checkpoint (`flow_parameterization: "rf_velocity"`), so step 4 runs the same Euler + CFG loop (40
steps) and compares it with upstream's `sample_euler_rf_cfg` exactly as for v4-small; every dimension, length cap and
token id comes from the checkpoint's own configs. At f32 it reads no file of step 2 (it builds the same graph
wrappers from the checkpoint), and writes `outputs/series/irodori-v4-large/pipeline/`:

```sh
uv run --with 'transformers==5.14.1' python -m irodori.pipeline_ref --model-dir ../../inputs/irodori/v4-large  # 4
```

- The case table follows the text backbone, picked by the embedded config as in step 2 (`pipeline_cases`): the
  small models keep `full` and `no-ref`, and a T5Gemma2 backbone adds `caption-window` — no reference and a 459-token
  caption, so the backbone runs where its sliding window cuts the sequence (from T = 257 on). Before writing, the step
  checks with transformers' own mask that some case really is cut, and records the names in `meta.json`
  (`windowCutCases`, a key written for T5Gemma2 only).
- The host side tokenizes with the bundled `tokenizer/tokenizer.json` as for the small models; the caption sequence
  is checked against upstream's own tokenizer wrapper, as before. There is no rare-token split for this model.
- Measured on 2026-11-14 (CPU, f32): every case passes the first-stage tolerance (1e-3) — largest difference of the
  final z from upstream 1.96e-5 (`full`), 1.90e-5 (`no-ref`) and 8.13e-5 (`caption-window`); 543 s wall, 30.0 GiB
  peak RSS.
- Only the f32 goldens exist so far; the f16 / i8 / i4 series follow their graphs.
