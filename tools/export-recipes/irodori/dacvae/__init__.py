"""DACVAE（Irodori チェーンの波形 ↔ latent コーデック）の export recipe。

**上流は Irodori-TTS とは別のリポ・別の重み**（`facebookresearch/dacvae` の実装 +
`Aratako/Semantic-DACVAE-Japanese-32dim` の重み）なので、由来を 1 ディレクトリに固めて
provenance を読み取れる形にする（ADR 0065 決定 2 — provenance は family 単位でコードと同居）。

`irodori/` の下に居るのは 2 つの理由から:

1. **コード結合が実在する** — {@link irodori.dacvae.host} は `irodori.export` を、
   `irodori.measure_quant` は {@link irodori.dacvae.export} を import する（配布形も
   テキスト〜音声を 1 リポで完走させるために同梱する — `irodori.distribution`）。
2. **上流パッケージ名が `dacvae`** — `tools/export-recipes/dacvae/` を作ると、
   export-recipes ルートが sys.path に載っている状態で上流の `dacvae` を覆い隠し、
   `from dacvae.model.dacvae import DACVAE`（{@link irodori.dacvae.export} の実装取り出し）が
   自分自身を掴む。ネストしていれば `irodori.dacvae` と `dacvae` は別のキーで衝突しない。

配布 recipe（`Pipeline`）は持たない — コーデック 2 グラフは Irodori 配布形の
`codec_decoder` / `codec_encoder` 席として配られるので、表は `irodori.distribution` が持つ。

MUST: **再輸出しない**（`from irodori.dacvae import X` で上流 import が芋づるに入る形に
しない）。呼び出し側はサブモジュールを名指しで import すること。

台本の起動は export-recipes ルートから（例: `uv run python -m irodori.dacvae.export`）。
"""
