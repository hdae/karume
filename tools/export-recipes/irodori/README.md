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

Six text-side graphs plus the DACVAE codec pair, with host-side goldens. The scripts have a
**required regeneration order** (later ones read earlier outputs):

```sh
# 0. one-time inputs: inputs/irodori/{v4-small,v4.1-small,dacvae-32dim,Irodori-TTS,dacvae-src}/
uv run python -m irodori.dacvae.convert                                  # 1. codec pth → safetensors
uv run --with 'transformers==5.14.1' python -m irodori.export            # 2. six graphs + io goldens
uv run --with 'transformers==5.14.1' python -m irodori.tokenizer_ref     # 3. tokenizer asset + goldens + parity fixture (deno fmt it)
uv run --with 'transformers==5.14.1' python -m irodori.pipeline_ref      # 4. full-loop latent goldens
uv run --with descript-audiotools --with einops \
    --with 'transformers==5.14.1' python -m irodori.dacvae.host          # 5. host preprocessing goldens
uv run --with descript-audiotools --with einops python -m irodori.dacvae.export  # 6. codec graphs + io goldens
uv run python dist.py --pipeline irodori                                 # 7. distribution (8 graphs + tokenizer)
```

The distribution carries an **f16 weight-storage series** next to f32 (ADR 0050), so step 7 needs
both series. Regenerate the f16 side with the same three scripts (order caveats are identical; the
codec / full-loop inputs stay on the f32 series on purpose — inputs are dtype-neutral):

```sh
uv run --with 'transformers==5.14.1' python -m irodori.export --dtype f16     # 2'. six graphs
uv run --with 'transformers==5.14.1' python -m irodori.pipeline_ref --dtype f16   # 4'. full-loop goldens
uv run --with descript-audiotools --with einops python -m irodori.dacvae.export --dtype f16  # 6'. codec
```

The `i8+dit4` quant seat adds an **i4 series for `dit` only** — the other seven roles share the i8 bytes,
so nothing else is re-exported for it. Only the DiT block weights outside the adaLN modulation are
stored as i4 (168 linears); the 144 adaLN linears (`attention_adaln`, `mlp_adaln`) and the five
linears outside the blocks (`in_proj`, `out_proj`, `cond_module.{0,2,4}`) are stored as i8 in the
same container — both exclusions come from listening judgements. Step 7 requires this series; the
rounding is GPTQ-calibrated by default and takes hours of CPU time (twelve calibration cases × the
full reference loop):

```sh
uv run --with 'transformers==5.14.1' python -m irodori.export --dtype i4   # 2''. dit only, calibrated
uv run --with 'transformers==5.14.1' python -m irodori.pipeline_ref --dtype i4  # 4''. full-loop goldens
```

Step 4'' **reads the container step 2'' wrote** (it is the only series whose goldens are baked from
the shipped bytes rather than from a second calibration run), so it must follow 2''.

`--no-calib` swaps the calibration for plain RTN. It exists for smoke runs only: the storage form is
byte-identical either way, so `dist` — and step 4'' — refuse the result by reading the
`calib_provenance.json` the export writes next to the container.

Order caveats measured in practice: step 2 reads step 5's real latent for the speaker cases
(`SPEAKER_REAL_CASES`), and step 6 reads step 4's `z` for the decoder cases — so a **full** rebuild
from scratch runs 2 once more after 5 (2 → 3 → 4 → 5 → 2 → 6 → 7). Incremental regeneration of a
single script is safe as long as its inputs above exist. Design records: ADR 0044 / 0046 / 0047
(graphs), 0048 (host port), 0049 (codec integration).

## Another model of the same architecture (v4.1-small)

One distribution repository per model. The text-side scripts (steps 2 / 3 / 4 and the dtype
variants) all take `--model-dir inputs/irodori/<model>/` and derive the series names from that
directory name, so exporting e.g. `v4.1-small` is the same chain with `--model-dir
inputs/irodori/v4.1-small` — plus `--model v4.1-small` on step 7 (the output directory
`models/karume-irodori-v4.1-small/` follows). The codec series (`dacvae-32dim*`) is shared across
models and its inputs are model-independent, so steps 1 / 5 / 6 are **not** re-run when they
already exist. `card.py`'s `IRODORI_UPSTREAMS` must know the model name (upstream repository +
display name), or the card render — and with it step 7 — fails loudly.
