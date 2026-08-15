"""母音検出 CRNN（音声 → リップシンク用の母音系列）の export recipe（ADR 0065 決定 2）。

ここに入るのは**汎用 exporter core に載せられないもの**だけ: 上流モデル由来のパッチ層
（{@link vowel_detector.patch}）・export 台本・配布 recipe
（{@link vowel_detector.distribution}）とカードテンプレート（{@link vowel_detector.card}）。
依存方向は **recipe → core の一方向だけ**（`tools/exporter/tests/test_architecture_boundary.py`
が機械で守る）。

パッケージ名がハイフンでなく `vowel_detector` なのは Python の識別子制約 — pipeline 名
（`--pipeline vowel-detector`）と系列名・リポ名の綴りは従来どおりハイフンで、そちらは
{@link vowel_detector.distribution} が持つ。

MUST: **再輸出しない**（`from vowel_detector import X` で family の中身が芋づるに import
される形にしない）。呼び出し側はサブモジュールを名指しで import すること。

台本の起動は export-recipes ルートから（例: `uv run python -m vowel_detector.export`）。
"""
