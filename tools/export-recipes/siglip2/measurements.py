"""SigLIP2 の export で実測した数（**torch を読まない側**に置く）。

置き場を patch 層から分けているのは import 連鎖のため。実測幅を語る綴りは配布側
（`siglip2.card` のモデルカード / `siglip2.distribution` の改変告知）が引くが、それらは
`dist.py` ドライバの import 連鎖に乗っている。数を `siglip2.patch` が持つと、配布側の import
連鎖に patch 層（torch 依存）が入る。NOTE: 現状は exporter core の `karume/__init__` が
`karume.convert` を eager に引くため、`import dist` だけでも torch は読まれる（2026-09-04 実測 —
backlog 起票）。ここで達成しているのは「patch 層を配布経路の連鎖から外す」と「実測幅の綴りを
1 か所にする」の 2 点で、torch の起動コストが消えるのは core が遅延化された日から。
ここは定数と文字列化だけの葉モジュールで、
`gemma4.distribution` が `SYM_MAX` を torch を読まない側へ写しているのと同じ向き
（あちらは写しで同値をテストが見るが、こちらは**正本ごと**こちらに置くので写しにならない）。

MUST: 実測幅を語る綴りはこのファイルの 1 つだけ。モデルカードと `NOTICE.md`（改変告知）が
同じ数を名乗るので、2 箇所に持つと片方だけ動いた日に「告知だけが古い数を主張する」形が
黙って作れる — 散文としては妥当なままなので、配ってからでないと誰も気づけない。
"""

from __future__ import annotations

#: MAP head の q/k/v 明示化（`siglip2.patch` の段 ③）が持ち込む pooler_output の差の実測幅
#: （下限・上限）。出どころは `siglip2/export.py --verify` の golden 4 ケース × 2 系列。
MAP_HEAD_MAXDIFF: tuple[float, float] = (7.75e-07, 2.38e-06)


#: 差を割るスケール = pooled ベクトルの L2 ノルムの実測幅（下限・上限）。同じ `--verify` の
#: 実走から採る対で、**この数が無いと maxdiff の大小が読めない**（相対 ~1e-7 の根拠）。
MAP_HEAD_VECTOR_NORM: tuple[float, float] = (12.7, 13.1)


def map_head_diff_text() -> str:
    """{@link MAP_HEAD_MAXDIFF} を散文へ埋める綴り（`"7.75e-07 to 2.38e-06"`）。

    書式を関数にしているのは、モデルカードと `NOTICE.md` が**同じ文字列**を名乗るため
    （`f"{low:g} to {high:g}"` を各所で書くと、書式だけが片側で動いても誰も落ちない）。
    """
    low, high = MAP_HEAD_MAXDIFF
    return f"{low:g} to {high:g}"


def map_head_norm_text() -> str:
    """{@link MAP_HEAD_VECTOR_NORM} を配布テキストへ埋める綴り（`"around 13"`）。

    内部は幅（対）で持ち、配布文では丸めたスカラで語る — 幅のまま出すと「差が無視できる」と
    いう主張に要らない精度が付く。{@link map_head_diff_text} と同じ 2 段（定数 + 文字列化）に
    してあるので、カードと `NOTICE.md` は同じ 1 語を引く。
    """
    low, high = MAP_HEAD_VECTOR_NORM
    return f"around {round((low + high) / 2):g}"
