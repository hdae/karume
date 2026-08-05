"""リポジトリ内の置き場 — 「配布形」と「生成物」を分ける 1 箇所。

DECIDED: `models/` に置くのは **HF へそのまま上げられる配布形だけ**（1 ディレクトリ =
1 HF リポ — ADR 0037 §3）。エクスポータが吐く系列（IR + io フィクスチャ）はモデルの
実行にも配布にも要らない中間生成物なので `outputs/` 側に置く。

系列の置き場は**書き手（台本）と読み手（`karume.dist`）の共有知識**なので、綴りは
ここ 1 箇所だけが持つ。
"""

from __future__ import annotations

from pathlib import Path

#: リポジトリのルート（karume/paths.py → karume → tools/exporter → tools → repo）。
REPO_ROOT = Path(__file__).resolve().parents[3]

#: 配布形の親（`<DIST_ROOT>/<配布名>/` が 1 つの HF リポになる）。
DIST_ROOT = REPO_ROOT / "models"

#: エクスポータの系列出力の親。**`models/` には置かない**（上の DECIDED）。
SERIES_ROOT = REPO_ROOT / "outputs" / "series"
