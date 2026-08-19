"""複数 family で共有するが core（PyPI `karume`）へ昇格できない補助（ADR 0065 決定 2）。

ここに置くのは「モデル別 recipe が共通で要るが、汎用 exporter の責務ではないもの」—
リポジトリの置き場の綴り（{@link _shared.paths}）と、decode 系列の台本が共有する門
（{@link _shared.decode_series}）。core へ昇格できる（= repo topology にもモデル台本の
運用にも依存しない）と分かったものは `karume` 側へ出す。
"""
