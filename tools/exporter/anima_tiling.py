"""VAE decode の**固定タイル化**の参照フィクスチャを作る（#21 波 T1）。

Karume の VAE decoder のグラフは**解像度に対して構造不変**なので（512px 用と 1024px 用の
`model.safetensors` はノード列・重みバイトまで完全一致 —
`docs/research/2026-08-03-dynres-vae-tiling.md` §1.2）、512px 用資産を latent 64×64 の
**タイル decoder** としてそのまま使い、切り出し / ブレンド / 貼り付けだけをホストで行える。
その「ホスト側の数の正」がここで、TS 実装（`examples/anima/host/tiling.ts`）は
`packages/runtime/tests/e2e_anima_tiling_test.ts` でこのフィクスチャと突き合わせる。

MUST: `vae.enable_tiling()` を**呼ばない**。上流（`autoencoder_kl_qwenimage.tiled_decode`）は
`range(0, H, stride)` で走査するので最後のタイルが短くなり、固定形のタイル decoder では
食えない。走査は「最後のタイルの開始位置を `extent − tile` にスナップする等間隔配置」へ
変えてある（recon §4.2 が予告した意図的逸脱）— したがって幾何は**自前実装**でなければ
TS 側と一致しない。**ブレンドの式**は上流の `blend_v` / `blend_h` の逐語移植で、同値は
`tests/test_anima_tiling.py` が本物のメソッドとの突合で固定する。

MUST: 重みは資産系列と同じ dtype へ fake-quant してから参照を採る（ADR 0006 —
`anima_pipeline.py` と同じ規律）。`--dtype f16` が既定なのは、TS 側が
`models/anima-f16/vae_decoder`（= f16 系列）をタイル decoder として開くから。

出力（既定 `<repo>/models/anima-tiling-f16-1024/`）:

    tiling.safetensors   latents_denorm（タイル decode の入力）/ image_tiled（出力）
    tiling.json          タイル幾何（開始位置列・stride・ブレンド幅）とメタ

    uv run --group anima python anima_tiling.py
    uv run --group anima python anima_tiling.py --dtype f16 --resolution 1024
    uv run --group anima python anima_tiling.py --resolution 1344x768 \
        --latents ../../models/anima-pipeline-turbo-f16-1344x768/pipeline.safetensors

`--resolution` は **WxH**（正方は略記できる）。非正方では VAE の静的資産が無いので、
タイル decode が唯一の経路になる（#23 — デモ側も `--vae-tiling` を要求する）。

入力 latent は既定で `models/anima-pipeline-turbo-f16-1024/pipeline.safetensors` の
`latents_denorm`（逆正規化済み = VAE decoder の入力そのもの）を借りる。randn ではなく実
パイプラインの latent を使うのは、継ぎ目の出方が値の中身に依るため。同じフィクスチャの
`image`（**非タイル**の torch decode）があれば、タイル化による差もメタに記録する
（観測であって門ではない — タイル化は近似なので差は 0 にならない）。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from karume.quantize import round_weights_to_f16
from karume.resolution import format_resolution, parse_resolution, resolution_meta

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"
#: 入力 latent を借りる参照フィクスチャ（`anima_pipeline.py --dtype f16 --resolution 1024`）。
DEFAULT_LATENTS = REPO_ROOT / "models" / "anima-pipeline-turbo-f16-1024" / "pipeline.safetensors"
#: タイル decoder の latent 幅（= 512px 用資産の入力 `[1,16,64,64]`）。
TILE_LATENT = 64
#: 隣り合うタイルが latent で重なる最小幅（= sample 64px — TS 側 `MIN_TILE_OVERLAP_LATENT`）。
MIN_OVERLAP_LATENT = 8
SPATIAL_COMPRESSION = 8


@dataclass(frozen=True)
class TileAxis:
    """latent の 1 軸ぶんのタイル配置（TS 側 `TileAxis` と同じ意味論）。"""

    extent: int
    tile: int
    stride: int
    starts: tuple[int, ...]

    def blend(self, scale: int) -> int:
        """ブレンド幅（**sample 空間** — 上流の `blend_height` / `blend_width` と同じ単位）。"""
        return (self.tile - self.stride) * scale


@dataclass(frozen=True)
class TileGeometry:
    scale: int
    channels: int
    rows: TileAxis
    cols: TileAxis

    @property
    def tiles(self) -> int:
        return len(self.rows.starts) * len(self.cols.starts)


def plan_tile_axis(extent: int, tile: int, min_overlap: int) -> TileAxis:
    """1 軸ぶんの等間隔スナップ配置（TS 側 `planTileAxis` と同じ規則）。

    「重なり `tile − stride` が `min_overlap` 以上」を満たす**最小のタイル本数**を選び、
    本数から stride を割り出す。本数から決めるので `(extent − tile) % stride == 0` は
    構成上つねに成立する。
    """
    if tile < 1:
        raise ValueError(f"タイル幅 {tile} が 1 未満")
    if extent < tile:
        raise ValueError(f"latent 全長 {extent} がタイル幅 {tile} より小さい")
    if not 0 <= min_overlap < tile:
        raise ValueError(f"最小の重なり {min_overlap} が [0, {tile}) の外")
    span = extent - tile
    if span == 0:
        # 縮退: 1 枚。stride = tile とすると重なり 0 → ブレンド無し・貼り付けは素の写し。
        return TileAxis(extent=extent, tile=tile, stride=tile, starts=(0,))
    for count in range(2, span + 2):
        if span % (count - 1) != 0:
            continue
        stride = span // (count - 1)
        if tile - stride < min_overlap:
            continue
        return TileAxis(
            extent=extent, tile=tile, stride=stride, starts=tuple(i * stride for i in range(count))
        )
    raise ValueError(f"latent {extent} をタイル {tile}（最小の重なり {min_overlap}）で覆えない")


def plan_tiling(
    latent_shape: tuple[int, ...],
    tile: int = TILE_LATENT,
    min_overlap: int = MIN_OVERLAP_LATENT,
    scale: int = SPATIAL_COMPRESSION,
) -> TileGeometry:
    """`[1,C,H,W]` の latent に対するタイル幾何。"""
    if len(latent_shape) != 4 or latent_shape[0] != 1:
        raise ValueError(f"latent の形 {list(latent_shape)} が [1,C,H,W] でない")
    return TileGeometry(
        scale=scale,
        channels=latent_shape[1],
        rows=plan_tile_axis(latent_shape[2], tile, min_overlap),
        cols=plan_tile_axis(latent_shape[3], tile, min_overlap),
    )


def blend_v(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    """上のタイルとの線形ランプ合成（`AutoencoderKLQwenImage.blend_v` の逐語移植・in-place）。"""
    blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
    for y in range(blend_extent):
        b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[
            :, :, :, y, :
        ] * (y / blend_extent)
    return b


def blend_h(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    """左のタイルとの線形ランプ合成（`AutoencoderKLQwenImage.blend_h` の逐語移植・in-place）。"""
    blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
    for x in range(blend_extent):
        b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[
            :, :, :, :, x
        ] * (x / blend_extent)
    return b


def tiled_decode(decode, z: torch.Tensor, geometry: TileGeometry) -> torch.Tensor:
    """rank5 latent `[1,C,1,H,W]` をタイルごとに decode してブレンド合成する。

    `decode` は「rank5 のタイル latent → rank5 の画像タイル」。実行経路（TS）は 1 本の
    session を使い回し、ここでは同じ `vae.decode` を繰り返し呼ぶ。

    MUST: ブレンドは **縦（上のタイル）→ 横（左のタイル）** の順・タイル配列は in-place に
    書き換える（上流 `tiled_decode` と同じ — 隣に効くのは**ブレンド済み**のタイル）。
    MUST: 貼り付けは「タイル i の担当領域 = `[starts[i], starts[i+1])`、最後だけ末端まで」の
    領域割り当て（上流の「stride 幅へ切り詰め + 全体 crop」の等間隔スナップ版）。素朴に
    stride 幅で切り詰めると `n·stride = extent − 重なり` にしかならず末端が欠ける。
    """
    rows_axis, cols_axis = geometry.rows, geometry.cols
    scale = geometry.scale
    tiles: list[list[torch.Tensor]] = []
    for top in rows_axis.starts:
        row: list[torch.Tensor] = []
        for left in cols_axis.starts:
            tile = z[:, :, :, top : top + rows_axis.tile, left : left + cols_axis.tile]
            row.append(decode(tile.contiguous()))
        tiles.append(row)

    blend_rows = rows_axis.blend(scale)
    blend_cols = cols_axis.blend(scale)
    for i, row in enumerate(tiles):
        for j, tile in enumerate(row):
            if i > 0:
                tile = blend_v(tiles[i - 1][j], tile, blend_rows)
            if j > 0:
                tile = blend_h(row[j - 1], tile, blend_cols)
            row[j] = tile

    sample = tiles[0][0]
    out = sample.new_zeros(
        sample.shape[0],
        sample.shape[1],
        sample.shape[2],
        rows_axis.extent * scale,
        cols_axis.extent * scale,
    )
    for i, top in enumerate(rows_axis.starts):
        span_rows = (
            rows_axis.starts[i + 1] if i + 1 < len(rows_axis.starts) else rows_axis.extent
        ) - top
        for j, left in enumerate(cols_axis.starts):
            span_cols = (
                cols_axis.starts[j + 1] if j + 1 < len(cols_axis.starts) else cols_axis.extent
            ) - left
            out[
                :,
                :,
                :,
                top * scale : (top + span_rows) * scale,
                left * scale : (left + span_cols) * scale,
            ] = tiles[i][j][:, :, :, : span_rows * scale, : span_cols * scale]
    return out


def geometry_meta(geometry: TileGeometry) -> dict[str, Any]:
    """幾何を JSON へ落とす（TS 側はこの表と自分の計算を突き合わせる — 数値だけに頼らない）。"""

    def axis(value: TileAxis) -> dict[str, Any]:
        return {
            "extent": value.extent,
            "tile": value.tile,
            "stride": value.stride,
            "starts": list(value.starts),
            "blend_sample": value.blend(geometry.scale),
        }

    return {
        "scale": geometry.scale,
        "channels": geometry.channels,
        "rows": axis(geometry.rows),
        "cols": axis(geometry.cols),
        "tiles": geometry.tiles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--out", type=Path, default=None, help="既定 models/anima-tiling-<dtype>-<解像度>/"
    )
    parser.add_argument("--dtype", choices=("f32", "f16"), default="f16")
    parser.add_argument(
        "--latents",
        type=Path,
        default=DEFAULT_LATENTS,
        help="入力 latent を借りる pipeline.safetensors（キー latents_denorm）",
    )
    parser.add_argument(
        "--resolution",
        default="1024",
        help="WxH（例 1344x768）。正方は略記できる — 綴りはデモの --resolution と同じ",
    )
    args = parser.parse_args()
    try:
        width, height = parse_resolution(args.resolution)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    from safetensors.torch import load_file, save_file

    if not args.latents.is_file():
        raise SystemExit(
            f"入力 latent {args.latents} が無い — 先に `anima_pipeline.py --dtype f16 "
            f"--resolution {args.resolution} …` でパイプライン参照を採る"
            "（examples/anima/README.md の資産表）"
        )
    source = load_file(str(args.latents))
    latents = source["latents_denorm"].to(torch.float32)
    # MUST: 軸の順は `[..., H, W]`。非正方で入れ替えると要素数が合ったまま転置された latent を
    # decode することになり、参照だけが別の絵になる（正方では検出不能な取り違え）。
    expected = [height // SPATIAL_COMPRESSION, width // SPATIAL_COMPRESSION]
    if list(latents.shape[2:]) != expected:
        raise SystemExit(
            f"入力 latent の形 {list(latents.shape)} が --resolution {args.resolution}"
            f"（latent {expected[0]}×{expected[1]}）と合わない"
        )
    geometry = plan_tiling(tuple(latents.shape))
    print(
        f"[geometry] {geometry.rows.starts} × {geometry.cols.starts} / tile {TILE_LATENT} / "
        f"stride {geometry.rows.stride},{geometry.cols.stride} / "
        f"blend(sample) {geometry.rows.blend(geometry.scale)},{geometry.cols.blend(geometry.scale)}"
        f" / {geometry.tiles} タイル",
        flush=True,
    )

    from diffusers import AutoencoderKLQwenImage

    vae = AutoencoderKLQwenImage.from_pretrained(args.repo, subfolder="vae")
    vae.to(torch.float32).eval()
    # MUST: 上流のタイル化は使わない（走査形が違う — モジュール doc）。既定 False だが、
    # 将来 config 側で既定が変わったときに黙って別の幾何で参照を採るのを防ぐ。
    if vae.use_tiling:
        raise SystemExit("vae.use_tiling が True — 上流のタイル化は走査形が違うので使わない")
    if args.dtype == "f16":
        report = round_weights_to_f16(vae)
        print(f"[fake-quant] vae (f16): {report.describe()}", flush=True)

    decoded = 0

    def decode(tile: torch.Tensor) -> torch.Tensor:
        nonlocal decoded
        with torch.no_grad():
            out = vae.decode(tile, return_dict=False)[0]
        decoded += 1
        print(f"[decode] タイル {decoded}/{geometry.tiles} {list(out.shape)}", flush=True)
        return out

    image = tiled_decode(decode, latents.unsqueeze(2), geometry)[:, :, 0].contiguous()
    tensors = {"latents_denorm": latents.contiguous(), "image_tiled": image}

    meta: dict[str, Any] = {
        "repo": args.repo,
        "dtype": args.dtype,
        **resolution_meta(width, height),
        "latents_source": str(args.latents),
        "min_overlap_latent": MIN_OVERLAP_LATENT,
        "geometry": geometry_meta(geometry),
        "image_range": [float(image.min()), float(image.max())],
        "tensors": {name: list(value.shape) for name, value in tensors.items()},
    }
    # 観測（門ではない）: 同じフィクスチャに非タイルの torch decode があれば差を記録する。
    # タイル化は attention の受容野をタイル内に閉じる近似なので、差は 0 にならない。
    full = source.get("image")
    if full is not None and full.shape == image.shape:
        difference = (image - full.to(torch.float32)).abs()
        meta["vs_full_decode"] = {
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
        }
        print(f"[observe] 非タイル decode との差 max {difference.max():.4e}", flush=True)

    args.out = (
        args.out
        or REPO_ROOT / "models" / f"anima-tiling-{args.dtype}-{format_resolution(width, height)}"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out / "tiling.safetensors"))
    (args.out / "tiling.json").write_text(
        json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"fixture OK: {len(tensors)} tensors → {args.out}", flush=True)


if __name__ == "__main__":
    main()
