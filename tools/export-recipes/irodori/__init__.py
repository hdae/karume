"""Irodori-TTS（text-to-speech）の export recipe（ADR 0065 決定 2 — wheel 外・リポ専用）。

ここに入るのは**汎用 exporter core に載せられないもの**だけ: 上流モデル由来のパッチ層
（{@link irodori.patch}）・export 台本・参照 pipeline / tokenizer の参照実装・quant 計測・
配布 recipe（{@link irodori.distribution}）とカードテンプレート（{@link irodori.card}）。
依存方向は **recipe → core の一方向だけ**（`tools/exporter/tests/test_architecture_boundary.py`
が機械で守る）。

コーデック（DACVAE）は {@link irodori.dacvae} サブパッケージ — **別の上流リポ・別ライセンス**
なので provenance ごと 1 ディレクトリに固めるが、`measure_quant` / `dacvae.host` が
`irodori.export` と相互に import する（配布形も 1 リポへ同梱する）ので独立 recipe にはしない。
トップレベルに置けない事情もある: 上流パッケージ名が `dacvae` そのもので、
`tools/export-recipes/dacvae/` は sys.path 上でそれを覆い隠す。

MUST: **再輸出しない**（`from irodori import X` で family の中身が芋づるに import される形に
しない）。台本はどれも重い上流 import を持つので、`import irodori` が transformers /
`irodori_tts` を引き込む形になると、配布 recipe を触るだけの経路まで巻き添えになる。
呼び出し側はサブモジュールを名指しで import すること。

台本の起動は export-recipes ルートから（例: `uv run python -m irodori.export --dtype i8`）。
"""
