"""examples/wan の画像デコーダ（decode-image.ts）のテストの fixture と期待値を書き出す。

実行時には不要。期待値は「Pillow で開いて convert("RGB") した RGB8」— 上流 Wan2.2 の generate.py
が条件画像を読む形（`Image.open(...).convert("RGB")`）と同じ。画像は奇数寸法で、画素ごと・
チャネルごとに値が違う（チャネルの取り違え・アルファの捨て忘れ・行の詰め方の誤りでテストが
落ちるように）。

Pillow は tools/export-recipes の uv 環境の wan グループにある（pillow を `==` で pin —
版を変えたら作り直す）:

    cd tools/export-recipes && uv run --group wan python ../../examples/wan/emit-image-fixtures.py \
        --out ../../examples/wan/fixtures

出力先に fixture が既にあれば上書きせずに落ちる（作り直すときは消してから回し、差分を確かめて
から採用する）。
"""

import argparse
import io
import json
import random
import struct
import zlib
from importlib.metadata import version
from pathlib import Path

from PIL import Image, JpegImagePlugin, features

#: PNG の fixture の寸法（奇数 × 奇数 — 行の端の詰め物と 4 bit の索引の半端なバイトを通す）。
PNG_SIZE = (7, 5)
#: JPEG の fixture の寸法（奇数 × 奇数で、4:2:0 の MCU〈16×16〉を複数またぎ、右端と下端が
#: MCU の途中で切れる）。
JPEG_SIZE = (37, 21)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

#: Adam7 の 7 パス（x0, y0, dx, dy — PNG 仕様 8.2）。
ADAM7_PASSES = [
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
]


def channel_values(index: int) -> tuple[int, int, int, int]:
    """画素 index の (R, G, B, A)。チャネルごとに別の数列で、隣の画素とも値が違う。"""
    return (
        (index * 7 + 1) % 256,
        (index * 13 + 50) % 256,
        (index * 29 + 100) % 256,
        (index * 37 + 5) % 256,
    )


def pixel_image(mode: str) -> Image.Image:
    """PNG_SIZE の画像を mode で作る（L / LA は R / A の数列を使う）。"""
    width, height = PNG_SIZE
    image = Image.new(mode, PNG_SIZE)
    values = [channel_values(index) for index in range(width * height)]
    if mode == "RGB":
        image.putdata([value[:3] for value in values])
    elif mode == "RGBA":
        image.putdata(values)
    elif mode == "L":
        image.putdata([value[0] for value in values])
    elif mode == "LA":
        image.putdata([(value[0], value[3]) for value in values])
    else:
        raise ValueError(mode)
    return image


def palette_image(colors: int) -> Image.Image:
    """索引 i の色が channel_values(i) の RGB のパレット画像。

    Pillow は 16 色以下なら 4 bit 以下で書く。
    """
    width, height = PNG_SIZE
    image = Image.new("P", PNG_SIZE)
    image.putpalette([channel for index in range(colors) for channel in channel_values(index)[:3]])
    # 索引は画素の並びと別の順（索引 = 画素位置のままだと、パレットを引かずに索引をそのまま返す
    # 誤りが偶然の一致で隠れうる — ここでは値の数列が違うので隠れないが、並びも崩しておく）。
    image.putdata([(index * 11 + 3) % colors for index in range(width * height)])
    return image


def jpeg_source() -> Image.Image:
    """JPEG の元画像: R は横・G は縦・B は斜めのグラデーション。

    チャネルの取り違えが大きな差になる。色の境界は鋭くしない: jpeg-js は 4:2:0 の色差を最近傍で
    広げ、libjpeg-turbo は補間するので、鋭い色の境界では差が大きく出る（B を市松にすると最大 56）。
    テストはデコーダの配線を見るので、補間の流儀の差が上限を決めない画像にする。
    """
    width, height = JPEG_SIZE
    image = Image.new("RGB", JPEG_SIZE)
    image.putdata(
        [
            (
                round(255 * x / (width - 1)),
                round(255 * y / (height - 1)),
                round(255 * (x + y) / (width + height - 2)),
            )
            for y in range(height)
            for x in range(width)
        ]
    )
    return image


def cmyk_source() -> Image.Image:
    """CMYK の JPEG の元画像（4 版とも乱数 — K が 0 だと CMYK → RGB の式の K の項を試せない）。"""
    width, height = JPEG_SIZE
    rng = random.Random(9)
    image = Image.new("CMYK", JPEG_SIZE)
    image.putdata([tuple(rng.randrange(256) for _ in range(4)) for _ in range(width * height)])
    return image


def png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def adam7_png(
    width: int, height: int, depth: int, color_type: int, pixels, pack, palette: bytes | None
) -> bytes:
    """Adam7 インターレースの PNG を手で組む（Pillow の PNG 書き出しはインターレースを書かない）。

    pixels(x, y) は画素の値、pack(values) はパスの 1 行の値の列をバイト列へ詰める。
    フィルタは全行 None。
    """
    raw = bytearray()
    for x0, y0, dx, dy in ADAM7_PASSES:
        xs = range(x0, width, dx)
        ys = range(y0, height, dy)
        if len(xs) == 0 or len(ys) == 0:
            continue
        for y in ys:
            raw.append(0)
            raw += pack([pixels(x, y) for x in xs])
    header = struct.pack(">IIBBBBB", width, height, depth, color_type, 0, 0, 1)
    return (
        PNG_SIGNATURE
        + png_chunk(b"IHDR", header)
        + (b"" if palette is None else png_chunk(b"PLTE", palette))
        + png_chunk(b"IDAT", zlib.compress(bytes(raw)))
        + png_chunk(b"IEND", b"")
    )


def palette_png(palette_colors: int, indices: list[int], with_palette: bool = True) -> bytes:
    """8 bit のパレットの PNG を手で組む（CRC は正しい — 壊れ方は中身だけ）。

    Pillow の書き出しはパレットの外の索引も PLTE 無しのパレット画像も書かないので手で組む。
    """
    width, height = PNG_SIZE
    raw = b"".join(b"\x00" + bytes(indices[y * width : (y + 1) * width]) for y in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 3, 0, 0, 0)
    palette = bytes(
        channel for index in range(palette_colors) for channel in channel_values(index)[:3]
    )
    return (
        PNG_SIGNATURE
        + png_chunk(b"IHDR", header)
        + (png_chunk(b"PLTE", palette) if with_palette else b"")
        + png_chunk(b"IDAT", zlib.compress(raw))
        + png_chunk(b"IEND", b"")
    )


def jpeg_markers(data: bytes) -> list[tuple[int, bytes]]:
    """SOI の後のマーカーの区画を最初の SOS まで（(マーカー, 本体) の列）。"""
    assert data[:2] == b"\xff\xd8"
    markers = []
    at = 2
    while True:
        assert data[at] == 0xFF
        marker = data[at + 1]
        if marker == 0xDA:
            return markers
        length = struct.unpack(">H", data[at + 2 : at + 4])[0]
        markers.append((marker, data[at + 4 : at + 2 + length]))
        at += 2 + length


def rgb_jpeg_variants() -> dict[str, bytes]:
    """3 成分の RGB 符号の JPEG（色の変換をしない）と、その判定の境目の形。

    Pillow の keep_rgb=True は JFIF を書かず、Adobe の APP14（transform = 0）と成分 ID 'R' 'G' 'B'
    で書く。そこから APP14 を抜く（成分 ID だけで RGB と分かる形）・JFIF の APP0 を足す（JFIF が
    優先して YCbCr とみなされる形）の 2 つを作る。
    """
    adobe = encode(jpeg_source(), "JPEG", quality=90, keep_rgb=True)
    assert adobe[2:4] == b"\xff\xee", "keep_rgb の JPEG の先頭の区画が APP14 でない"
    app14_end = 4 + struct.unpack(">H", adobe[4:6])[0]
    jfif = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    return {
        "rgb-adobe.jpg": adobe,
        "rgb-ids.jpg": adobe[:2] + adobe[app14_end:],
        "jfif-adobe0.jpg": adobe[:2] + jfif + adobe[2:],
    }


def pack_rgb8(values) -> bytes:
    return bytes(channel for value in values for channel in value)


def pack_index4(values) -> bytes:
    padded = [*values, 0] if len(values) % 2 else values
    return bytes((padded[at] << 4) | padded[at + 1] for at in range(0, len(padded), 2))


def encode(image: Image.Image, fmt: str, **params) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, fmt, **params)
    return buffer.getvalue()


def png_header(data: bytes) -> tuple[int, int, int]:
    """IHDR の (ビット深度, 色の種類, インターレース)。"""
    assert data[:8] == PNG_SIGNATURE and data[12:16] == b"IHDR"
    return data[24], data[25], data[28]


def build() -> dict[str, tuple[bytes, str]]:
    """fixture の名前 → (バイト列, 期待: "exact" | "jpeg" | "reject")。"""
    width, height = PNG_SIZE
    fixtures: dict[str, tuple[bytes, str]] = {
        "rgb.png": (encode(pixel_image("RGB"), "PNG"), "exact"),
        "rgba.png": (encode(pixel_image("RGBA"), "PNG"), "exact"),
        "l.png": (encode(pixel_image("L"), "PNG"), "exact"),
        "la.png": (encode(pixel_image("LA"), "PNG"), "exact"),
        # tRNS 付きの 8 bit パレット（パレットの各色にアルファが付く — 捨てて RGB を引く）。
        "p8.png": (
            encode(palette_image(width * height), "PNG", transparency=bytes(range(0, 200, 10))),
            "exact",
        ),
        "p4.png": (encode(palette_image(16), "PNG"), "exact"),
        "adam7-rgb.png": (
            adam7_png(
                width,
                height,
                8,
                2,
                lambda x, y: channel_values(y * width + x)[:3],
                pack_rgb8,
                None,
            ),
            "exact",
        ),
        # Adam7 の 8 bit 未満のパレット — fast-png の Adam7 の経路は 8 bit 未満の行の詰め方を
        # 扱えないので拒否する。
        "adam7-p4.png": (
            adam7_png(
                width,
                height,
                4,
                3,
                lambda x, y: (y * width + x) % 16,
                pack_index4,
                bytes(channel for index in range(16) for channel in channel_values(index)[:3]),
            ),
            "reject",
        ),
        "l16.png": (encode(pixel_image("L").convert("I;16"), "PNG"), "reject"),
        "l1.png": (encode(pixel_image("L").convert("1"), "PNG"), "reject"),
        # 壊れたパレット（CRC は正しい）: PLTE が 2 色なのに索引 0〜5 を使う・
        # 色の種類 3 で PLTE が無い。
        "p8-bad-index.png": (
            palette_png(2, [index % 6 for index in range(width * height)]),
            "reject",
        ),
        "p8-no-plte.png": (
            palette_png(2, [index % 2 for index in range(width * height)], with_palette=False),
            "reject",
        ),
        "baseline-420.jpg": (encode(jpeg_source(), "JPEG", quality=90, subsampling=2), "jpeg"),
        "progressive.jpg": (encode(jpeg_source(), "JPEG", quality=90, progressive=True), "jpeg"),
        "gray.jpg": (encode(jpeg_source().convert("L"), "JPEG", quality=90), "jpeg"),
        # Pillow は CMYK を Adobe の APP14 付き（反転）で書く。
        "cmyk.jpg": (encode(cmyk_source(), "JPEG", quality=90), "jpeg"),
        # EXIF の向き = 6（90° 回転して見せる指定）。デコーダは向きを適用しない（寸法が
        # 入れ替わらない）。
        "exif-orientation-6.jpg": (
            encode(jpeg_source(), "JPEG", quality=90, exif=exif_orientation(6)),
            "jpeg",
        ),
        **{name: (data, "jpeg") for name, data in rgb_jpeg_variants().items()},
        "gif.gif": (encode(pixel_image("RGB"), "GIF"), "reject"),
    }
    if features.check("webp"):
        fixtures["webp.webp"] = (encode(pixel_image("RGB"), "WEBP", lossless=True), "reject")
    else:
        raise RuntimeError("この Pillow は WebP を書けない — WebP の拒否の fixture を作れない")
    checks = {
        "p8.png": (8, 3, 0),
        "p4.png": (4, 3, 0),
        "adam7-rgb.png": (8, 2, 1),
        "adam7-p4.png": (4, 3, 1),
        "l16.png": (16, 0, 0),
        "l1.png": (1, 0, 0),
    }
    for name, expected in checks.items():
        actual = png_header(fixtures[name][0])
        assert actual == expected, (
            f"{name} の IHDR (深度, 色の種類, インターレース) が {actual}（期待 {expected}）"
        )
    for name in ["baseline-420.jpg", "progressive.jpg"]:
        with Image.open(io.BytesIO(fixtures[name][0])) as opened:
            # get_sampling: 2 = 4:2:0。progressive は SOF2（Pillow の info["progressive"]）。
            assert JpegImagePlugin.get_sampling(opened) == 2, f"{name} が 4:2:0 でない"
            assert bool(opened.info.get("progressive")) == (name == "progressive.jpg"), (
                f"{name} の progressive が違う"
            )
    with Image.open(io.BytesIO(fixtures["exif-orientation-6.jpg"][0])) as opened:
        assert opened.getexif().get(0x0112) == 6, "EXIF の向きが書けていない"
    with Image.open(io.BytesIO(fixtures["cmyk.jpg"][0])) as opened:
        assert opened.mode == "CMYK" and opened.info.get("adobe") is not None, (
            "CMYK（Adobe）の JPEG になっていない"
        )
    # RGB 符号の 3 形: マーカーの組み合わせと、Pillow（libjpeg-turbo）の色空間の判定を確かめる。
    # 判定は元画像との差で見る（RGB とみなせば差は量子化の範囲、YCbCr とみなせば色が崩れる）。
    source = jpeg_source().tobytes()
    for name, (has_jfif, adobe_transform, rgb) in {
        "rgb-adobe.jpg": (False, 0, True),
        "rgb-ids.jpg": (False, None, True),
        "jfif-adobe0.jpg": (True, 0, False),
    }.items():
        markers = jpeg_markers(fixtures[name][0])
        jfif = any(m == 0xE0 and body[:5] == b"JFIF\x00" for m, body in markers)
        adobe = [body[11] for m, body in markers if m == 0xEE and body[:5] == b"Adobe"]
        sof = next(body for m, body in markers if m == 0xC0)
        ids = [sof[6 + 3 * index] for index in range(sof[5])]
        actual = (jfif, adobe[0] if adobe else None, ids)
        assert actual == (has_jfif, adobe_transform, [82, 71, 66]), (
            f"{name} のマーカーが想定と違う: JFIF {jfif}・Adobe {adobe}・成分 ID {ids}"
        )
        with Image.open(io.BytesIO(fixtures[name][0])) as opened:
            decoded = opened.convert("RGB").tobytes()
        worst = max(abs(a - b) for a, b in zip(decoded, source, strict=True))
        assert (worst < 32) == rgb, (
            f"{name} の Pillow の色空間の判定が想定と違う（元画像との最大差 {worst}）"
        )
    return fixtures


def exif_orientation(value: int) -> bytes:
    exif = Image.Exif()
    exif[0x0112] = value
    return exif.tobytes()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fixtures = build()
    expected_path = args.out / "expected.json"
    targets = [args.out / name for name in fixtures] + [expected_path]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"出力先に fixture があります（消してから回す）: {existing}")
    args.out.mkdir(parents=True, exist_ok=True)
    cases = {}
    for name, (data, kind) in fixtures.items():
        path = args.out / name
        path.write_bytes(data)
        case: dict[str, object] = {"kind": kind}
        if kind != "reject":
            # 期待値は書き出したファイルを開き直して取る（JPEG はファイルの復号が正本）。
            with Image.open(path) as opened:
                rgb = opened.convert("RGB")
            case |= {"width": rgb.width, "height": rgb.height, "rgb": rgb.tobytes().hex()}
        cases[name] = case
    expected = {
        "generator": "examples/wan/emit-image-fixtures.py",
        "pillow": version("pillow"),
        "cases": cases,
    }
    expected_path.write_text(
        json.dumps(expected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"{len(fixtures)} fixtures + expected.json → {args.out}")


if __name__ == "__main__":
    main()
