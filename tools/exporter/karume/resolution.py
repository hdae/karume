"""解像度の綴り（`WxH` と正方の略記 `N`）— 参照台本が共有する 1 箇所（#23）。

デモ（`examples/anima/main.ts`）の `--resolution` と同じ綴りを参照台本でも使う。同じ文字列で
フィクスチャとデモが指せないと、「1344x768 の参照で 768x1344 のデモを検証した」形の取り違えが
名前の上で見えなくなる。

MUST: **受理集合の正本はデモ側**（`examples/anima/host/resolution.ts`）。ここが見るのは
参照フィクスチャが構造として成立する条件（正の整数・{@link GRANULARITY} の倍数）だけで、
資産側の下限（各辺 ≥ 512 = VAE タイル latent 64）も `Dim("S")` の上限も見ない — 参照台本は
デモが受けない小さい形で回すこともある（単体テストの合成ケース）。両側に同じ受理集合を
書くと、片方だけ動かしたときに「参照は採れるがデモが受けない」形が静かに増える。
"""

from __future__ import annotations

#: 解像度の刻み（= 空間圧縮 8 × patch 2）。latent の各辺が patch 2 で割り切れる最小の単位で、
#: 外すと patchify が組めない（ホスト側 `latentSides` が落とす）。
GRANULARITY = 16


def parse_resolution(text: str) -> tuple[int, int]:
    """`"1344x768"` → `(1344, 768)`、`"512"` → `(512, 512)`（正方の略記）。

    MUST: 略記と `WxH` 以外の綴りを黙って受けない。`1344*768` や `1344,768` を
    「数値でない」で落とすのは入口の綴り検査であって、寸法の妥当性はここでは見ない。
    """
    parts = text.lower().split("x")
    if len(parts) == 1:
        parts = parts * 2
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f"解像度 {text!r} が WxH でも正方の略記でもない（例: 1344x768 / 512）")
    width, height = (int(part) for part in parts)
    for side, label in ((width, "幅"), (height, "高さ")):
        if side <= 0 or side % GRANULARITY != 0:
            raise ValueError(f"解像度 {text!r} の{label} {side} が {GRANULARITY} の正の倍数でない")
    return width, height


def format_resolution(width: int, height: int) -> str:
    """`(1344, 768)` → `"1344x768"`、正方は略記（`(512, 512)` → `"512"`）。

    {@link parse_resolution} の逆で、往復が恒等になる綴りを返す（ディレクトリ名に載るので、
    正方の既存フィクスチャのパスが 1 文字も変わらないことが要る）。
    """
    return str(width) if width == height else f"{width}x{height}"


def resolution_meta(width: int, height: int) -> dict[str, int | str]:
    """フィクスチャ JSON の解像度欄（寸法の正本は `width` / `height`）。

    MUST: 正方の run では `resolution` を**これまでどおり int の一辺**で出す。既存の
    フィクスチャ（`models/anima-tiling-f16-1024/tiling.json`）を読むテストがこの欄を数として
    使っており、型を動かすと再生成したときだけ黙って壊れる。非正方には一辺が存在しないので
    綴り（`"1344x768"`）を入れる — 読み手は `width` / `height` を見ること。
    """
    return {
        "resolution": width if width == height else format_resolution(width, height),
        "width": width,
        "height": height,
    }
