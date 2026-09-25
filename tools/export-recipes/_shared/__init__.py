"""複数 family で共有するが core（PyPI `karume`）へ昇格できない補助（ADR 0065 決定 2）。

ここに置くのは「モデル別 recipe が共通で要るが、汎用 exporter の責務ではないもの」—
リポジトリの置き場の綴り（{@link _shared.paths}）・decode 系列の台本が共有する門
（{@link _shared.decode_series}）・i4 系列の校正条件の判定（{@link _shared.calib_provenance}）・
据えたコンテナの格納のままの読み戻し（{@link _shared.container_read}）・上流 checkpoint の
出所の導出と突合（{@link _shared.upstream}）・Gemma 系 SPM-BPE
トークナイザの compile（{@link _shared.gemma_tokenizer}）・配布リポへ同梱するライセンス原文
（{@link _shared.licenses}）。
core へ昇格できる（= repo topology にもモデル台本の運用にも依存しない）と分かったものは
`karume` 側へ出す。
"""
