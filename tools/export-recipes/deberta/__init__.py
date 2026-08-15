"""DeBERTa-v2（日本語文字 BERT / SBV2 の text encoder）の export recipe（ADR 0065 決定 2）。

ここに入るのは**汎用 exporter core に載せられないもの**だけ: 上流モデル由来のパッチ層
（{@link deberta.patch}）と export 台本（{@link deberta.export}）。配布 recipe は持たない —
この系列は SBV2 配布形の `text_encoder` 席として配られるので、配布の表は
`sbv2.distribution` 側が持つ（あちらは系列の出力を **path で**拾うだけで、この package を
import しない）。

依存方向は **recipe → core の一方向だけ**（`tools/exporter/tests/test_architecture_boundary.py`
が機械で守る）。

MUST: **再輸出しない**（`from deberta import X` で transformers が芋づるに入る形にしない）。
呼び出し側はサブモジュールを名指しで import すること。

台本の起動は export-recipes ルートから（例: `uv run python -m deberta.export --dtype i8`）。
"""
