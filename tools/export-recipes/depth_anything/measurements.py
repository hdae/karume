"""Depth Anything V2 の export で実測した数（**torch を読まない側**に置く）。

置き場を patch 層から分けている理由は `siglip2.measurements` と同じ — 実測幅を語る綴りは
配布側（`depth_anything.card` のモデルカード / `depth_anything.distribution` の改変告知）が
引くが、それらは `dist.py` ドライバの import 連鎖に乗っている。数を `depth_anything.patch` が
持つと、配布側の import 連鎖に patch 層（torch 依存）が入る（torch 自体は現状 core の
`karume/__init__` が eager に引くので、ここで消えるのは patch 層の依存だけ — siglip2 と同じ）。

MUST: 実測幅を語る綴りはこのファイルの 1 つだけ。以前は patch 層の docstring（`1.4e-06`）と
カードの定数（`1.4e-6`）に分かれており、**同じ実測値が 2 つの綴りで並んでいた** — 片方だけが
動いても散文としては妥当なままなので、配ってからでないと誰も気づけない。
"""

from __future__ import annotations

#: DPT reassemble の `ConvTranspose2d` → 1×1 conv + pixel shuffle 差し替え
#: （`depth_anything.patch` の段 ③）が持ち込む深度の差の実測上限。縮約順序の違いで最下位
#: ビットが動くぶんで、深度の値の RMS はおよそ 1.0。出どころは
#: `depth_anything/export.py --verify`（合成 4 ケースの最大 — 2026-09-05 実測: checker 1.67e-06 /
#: disc 1.55e-06 / noise 1.67e-06 / ramp 6.08e-06。以前の綴り 1.4e-06 は 1 ケースの値で、
#: 4 ケースの上限ではなかった）。`verify_patches` はこの値を**門**として持ち、再実測が
#: 超えれば落ちる（W-P6-3）。
CONVT_MAXDIFF: float = 6.1e-06


def convt_diff_text() -> str:
    """{@link CONVT_MAXDIFF} を散文へ埋める綴り（`"6.1e-06"`）。

    書式を関数にしているのは、モデルカードと `NOTICE.md` が**同じ文字列**を名乗るため。
    """
    return f"{CONVT_MAXDIFF:g}"
