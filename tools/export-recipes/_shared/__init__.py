"""複数 family で共有するが core（PyPI `karume`）へ昇格できない補助（ADR 0065 決定 2）。

ここに置くのは「モデル別 recipe が共通で要るが、汎用 exporter の責務ではないもの」— 今は
リポジトリの置き場の綴り（{@link _shared.paths}）だけ。core へ昇格できる（= repo topology に
依存しない）と分かったものは `karume` 側へ出す。
"""
