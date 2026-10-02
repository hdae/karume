"""Wan2.1 の動画 VAE の**タイル decode**（幾何の Python 側の正と参照フィクスチャ — ADR 0118 段 5）。

chunk グラフ（`wan/vae_patch.py`）の空間は潜在 `t×t` の固定タイル（資産の入力形・既定 32）。
全画面（832×480 = 潜在 60×104）はホストがタイルに切って decode し、重なりをブレンドして
貼り合わせる（常時タイル — ADR 0118 決定 2・ADR 0033 / 0038 §4 の動画版）。ここはその
**幾何とブレンドの Python 側の正**と、GPU の照合
（`packages/models/tests/e2e_wan_vae_tiles_test.ts`）が読む参照フィクスチャの台本。
TS 側は `packages/models/src/wan/vae-tiles.ts`。

## 幾何 = 丸め等間隔スナップ配置（上流からの意図的な逸脱）

上流 `AutoencoderKLWan.tiled_decode` は `range(0, H, stride)` で走査するので最後のタイルが
短くなり、固定形の chunk グラフでは食えない。開始位置は 0 と `extent − tile` の間を**丸めて
等分**する（Anima と同じ規則 — ADR 0033 追記 P-3・`anima/tiling.py` の `plan_tile_axis`）。
本数は「隣り合う対の重なりが {@link MIN_OVERLAP_LATENT} 以上」を満たす最小。832×480
（潜在 60×104・タイル 32）は行 0 / 14 / 28 × 列 0 / 24 / 48 / 72 の 12 枚（重なりは行 18 潜在・
列 8 潜在）。

## ブレンド・貼り付け・クランプ

- ブレンドは上流の `blend_v` / `blend_h` の**逐語**（{@link blend_v}）。縦（上のタイル）→
  横（左のタイル）の順・in-place・sample 空間（上流 `tiled_decode` と同じ — 隣に効くのは
  ブレンド済みのタイル）。全フレーム・全チャネルに同じ式が掛かる（上流の `b[:, :, :, y, :]` が
  フレーム軸を丸ごと取る）。
- 貼り付けは**領域割り当て**（タイル i の担当 = `[starts[i], starts[i+1])`・最後だけ末端まで）
  — 上流の「stride 幅へ切り詰め + 全体 crop」のスナップ版（ADR 0033 決定 3。stride 幅で
  切り詰めると末端が欠ける）。
- `clamp(-1, 1)` は貼り付けの後（上流と同じ位置）。参照はクランプ**前**を書く（クランプは
  飽和した要素の差を隠す — 段 4 と同じ判断）。

## タイルが外・chunk が内

タイルごとに cache を作り直す（上流 `tiled_decode` の `clear_cache` と同じ — chunk が外だと
タイルの枚数ぶんの cache を同時に持つ。決定 2）。post-quant の 1×1×1 conv もタイル ×
フレームごとに掛ける（上流 `tiled_decode` の形のまま。chunk グラフもグラフの中でフレームごとに
掛ける）。

## フィクスチャ（系列の根の `vae_tiles.<case>.safetensors`）

- `band`（帯を決める）: seed 固定の潜在 `[16,9,60,104]`（832×480・33 フレーム）のタイル参照
  `frames` と、**非タイル**の参照 `frames_full`（上流の非タイル `_decode` の chunk ループ・
  クランプ前 — タイル化の近似の差の観測用。門ではない）。
- `accept`（受け入れを判定する）: 別の seed・**縦長** `[16,3,104,60]`（480×832・9 フレーム）。
  タイルの位置も chunk 境界も band と違う。行と列の取り違えは縦横が違う形でしか値に出ない
  （正方では対合）。
- メタに幾何（開始位置・ブレンド幅）を書く — TS の計画と突き合わせる（ADR 0033 追記 9a の
  二重凍結の上に、フィクスチャでの突き合わせを足す）。

重みは f16 表現可能値へ丸めてから参照を採る（ADR 0006 — `export_vae.load_vae(round_f16=True)`）。

    uv run --group wan --inexact python -m wan.vae_tiling                # band + accept
    uv run --group wan --inexact python -m wan.vae_tiling --case accept  # 1 本だけ

MUST: diffusers は関数の中でだけ触る（`wan` グループは既定の sync に入らない —
`tests/test_optional_group_imports.py`）。
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import save_file

from _shared.paths import SERIES_ROOT
from wan import export_vae, vae_patch

if TYPE_CHECKING:
    from diffusers import AutoencoderKLWan

#: 隣り合うタイルが潜在で重なる最小幅（= 64 px）。上流の既定のブレンド幅
#: `tile_sample_min − tile_sample_stride` = 256 − 192 = 64 px と同じ（TS 側
#: `WAN_VAE_MIN_TILE_OVERLAP`）。
MIN_OVERLAP_LATENT = 8

#: 参照フィクスチャのファイル名（系列の根に置く — `vae_tiles.<case>.safetensors`）。
FIXTURE_PREFIX = "vae_tiles."
FIXTURE_SUFFIX = ".safetensors"


@dataclass(frozen=True)
class TileAxis:
    """潜在の 1 軸ぶんのタイル配置（TS 側 `WanVaeTileAxis` と同じ意味論）。

    MUST: 開始位置の差（間隔）を欄として持たない — 丸め等間隔なので対ごとに 1 潜在まで動く。
    要るところで `starts` から引く（{@link blend_at}・{@link region}）。
    """

    extent: int
    tile: int
    starts: tuple[int, ...]

    def blend_at(self, scale: int, index: int) -> int:
        """対 `(index − 1, index)` のブレンド幅（**sample 空間** — 上流と同じ単位）。"""
        if not 1 <= index < len(self.starts):
            raise ValueError(
                f"ブレンド対 {index} が範囲外（開始位置 {len(self.starts)} 本 = "
                f"対 {len(self.starts) - 1} 組）"
            )
        return (self.tile - (self.starts[index] - self.starts[index - 1])) * scale

    def blends(self, scale: int) -> list[int]:
        """全対のブレンド幅（先頭のタイルには対が無いので `本数 − 1` 個）。"""
        return [self.blend_at(scale, index) for index in range(1, len(self.starts))]

    def region(self, index: int) -> int:
        """タイル `index` の担当領域の長さ（潜在）= 次のタイルの開始まで・最後だけ末端まで。"""
        end = self.starts[index + 1] if index + 1 < len(self.starts) else self.extent
        return end - self.starts[index]


@dataclass(frozen=True)
class TilePlan:
    """潜在 `[C, F, H, W]` の 2 軸ぶんのタイル配置と、潜在 ↔ sample の縮尺。"""

    scale: int
    rows: TileAxis
    cols: TileAxis

    @property
    def tiles(self) -> int:
        return len(self.rows.starts) * len(self.cols.starts)

    def meta(self) -> dict[str, str]:
        """safetensors のメタ（文字列だけ）へ落とした幾何 — TS の計画と突き合わせる。"""

        def joined(values: Sequence[int]) -> str:
            return ",".join(str(value) for value in values)

        return {
            "tile": str(self.rows.tile),
            "scale": str(self.scale),
            "rows_starts": joined(self.rows.starts),
            "cols_starts": joined(self.cols.starts),
            "rows_blend": joined(self.rows.blends(self.scale)),
            "cols_blend": joined(self.cols.blends(self.scale)),
        }


def plan_tile_axis(extent: int, tile: int, min_overlap: int = MIN_OVERLAP_LATENT) -> TileAxis:
    """1 軸ぶんの丸め等間隔スナップ配置（TS 側 `planWanVaeTileAxis` と同じ規則）。

    本数は「重なりが `min_overlap` 以上」を満たす最小値 `ceil(span / (tile − min_overlap)) + 1`
    （`span = extent − tile`）、開始位置は `round(i · span / (本数 − 1))`（0.5 は切り上げ）。
    """
    if tile < 1:
        raise ValueError(f"タイル幅 {tile} が 1 未満")
    if extent < tile:
        raise ValueError(f"潜在の全長 {extent} がタイル幅 {tile} より小さい")
    if not 0 <= min_overlap < tile:
        raise ValueError(f"最小の重なり {min_overlap} が [0, {tile}) の外")
    span = extent - tile
    if span == 0:
        # 縮退: 1 枚。対が 1 つも無いのでブレンド無し・貼り付けは素の写し。
        return TileAxis(extent=extent, tile=tile, starts=(0,))
    count = -(-span // (tile - min_overlap)) + 1
    # MUST: 組み込みの `round` を使わない — Python は偶数丸め・TS の `Math.round` は 0.5
    # 切り上げで、ちょうど半分になる対だけ開始位置が黙って 1 潜在ずれる。整数式
    # floor((2·i·span + (本数 − 1)) / (2·(本数 − 1))) は 0.5 切り上げそのもの。
    starts = tuple((2 * index * span + (count - 1)) // (2 * (count - 1)) for index in range(count))
    # 重なりの下限は本数の式から導けるが、導出は丸めの誤差評価に依っていて目で追えない。破れたら
    # 継ぎ目がランプで隠れなくなる（絵にしか出ない沈黙誤り）ので、構造で落とす。
    for index in range(1, count):
        overlap = tile - (starts[index] - starts[index - 1])
        if overlap < min_overlap:
            raise ValueError(
                f"タイル {index - 1}/{index} の重なり {overlap} が下限 {min_overlap} 未満"
                f"（潜在 {extent} / タイル {tile} / 開始位置 {starts}）"
            )
    return TileAxis(extent=extent, tile=tile, starts=starts)


def plan_tiles(
    height: int,
    width: int,
    tile: int,
    scale: int = export_vae.SPATIAL_SCALE,
    min_overlap: int = MIN_OVERLAP_LATENT,
) -> TilePlan:
    """潜在の空間 `height × width` に対するタイル計画（軸ごとに独立）。"""
    return TilePlan(
        scale=scale,
        rows=plan_tile_axis(height, tile, min_overlap),
        cols=plan_tile_axis(width, tile, min_overlap),
    )


# Third-party code notice. `blend_v`, `blend_h`, `decode_tiles` and `assemble_tiles` below are
# adapted from `AutoencoderKLWan.blend_v`, `AutoencoderKLWan.blend_h` and
# `AutoencoderKLWan.tiled_decode` in
# huggingface/diffusers
# (`src/diffusers/models/autoencoders/autoencoder_kl_wan.py`, `diffusers==0.39.0`).
# `blend_v` / `blend_h` are verbatim (as module functions instead of methods). `decode_tiles` keeps
# the inner chunk loop verbatim and walks the snapped tile starts instead of
# `range(0, H, stride)`; `assemble_tiles` keeps the blend order and replaces the stride crop and the
# final crop by the region assignment; the clamp and the unpatchify branch are left out.
# License: Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Upstream
# copyright notice, copied verbatim from the header of that file:
#
#   Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
def blend_v(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    """上のタイルとの線形ランプ合成（上流 `blend_v` の逐語・in-place）。"""
    blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
    for y in range(blend_extent):
        b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[
            :, :, :, y, :
        ] * (y / blend_extent)
    return b


def blend_h(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    """左のタイルとの線形ランプ合成（上流 `blend_h` の逐語・in-place）。"""
    blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
    for x in range(blend_extent):
        b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[
            :, :, :, :, x
        ] * (x / blend_extent)
    return b


def decode_tiles(
    vae: AutoencoderKLWan, latents: torch.Tensor, plan: TilePlan
) -> list[list[torch.Tensor]]:
    """タイルごとに chunk ループで decode する（**タイルが外・chunk が内**・クランプ前）。

    `latents` は逆正規化済みの `[1, 16, F, H, W]`。戻りは行優先のタイル `[1, 3, 1 + 4(F−1), s, s]`
    （s = タイル辺 × 縮尺）。中身は上流 `tiled_decode` の内側のループの逐語で、走査だけが
    スナップ配置（モジュール doc）。
    """
    if vae.config.patch_size is not None:
        raise vae_patch.UnsupportedVaeError("patch_size 付きの VAE は未対応")
    _, _, num_frames, height, width = latents.shape
    if (height, width) != (plan.rows.extent, plan.cols.extent):
        raise ValueError(
            f"潜在の空間 {height}×{width} が計画 {plan.rows.extent}×{plan.cols.extent} と違う"
        )
    tile_height, tile_width = plan.rows.tile, plan.cols.tile
    rows: list[list[torch.Tensor]] = []
    for i in plan.rows.starts:
        row: list[torch.Tensor] = []
        for j in plan.cols.starts:
            vae.clear_cache()
            frames: list[torch.Tensor] = []
            for k in range(num_frames):
                vae._conv_idx = [0]
                tile = latents[:, :, k : k + 1, i : i + tile_height, j : j + tile_width]
                tile = vae.post_quant_conv(tile)
                decoded = vae.decoder(
                    tile, feat_cache=vae._feat_map, feat_idx=vae._conv_idx, first_chunk=(k == 0)
                )
                frames.append(decoded)
            row.append(torch.cat(frames, dim=2))
        rows.append(row)
    vae.clear_cache()
    return rows


def assemble_tiles(tiles: list[list[torch.Tensor]], plan: TilePlan) -> torch.Tensor:
    """decode 済みのタイル（行優先）をブレンドして `[1, 3, F', H·s, W·s]` に貼り合わせる。

    MUST: ブレンドは縦（上）→ 横（左）の順で、タイルを **in-place** に書き換える（上流と同じ —
    隣に効くのはブレンド済みのタイル。角の 4 枚が重なる領域で係数の順が変わる）。
    MUST: ブレンド幅は**対ごと**に引く（丸め等間隔なので対で 1 潜在まで違う）。
    MUST: 貼り付けは領域割り当て（{@link TileAxis.region}）。
    """
    rows_axis, cols_axis = plan.rows, plan.cols
    if len(tiles) != len(rows_axis.starts) or any(
        len(row) != len(cols_axis.starts) for row in tiles
    ):
        raise ValueError(
            f"タイル {[len(row) for row in tiles]} が計画の "
            f"{len(rows_axis.starts)}×{len(cols_axis.starts)} と違う"
        )
    scale = plan.scale
    for i, row in enumerate(tiles):
        for j, tile in enumerate(row):
            # blend the above tile and the left tile to the current tile
            if i > 0:
                tile = blend_v(tiles[i - 1][j], tile, rows_axis.blend_at(scale, i))
            if j > 0:
                tile = blend_h(row[j - 1], tile, cols_axis.blend_at(scale, j))
            row[j] = tile

    sample = tiles[0][0]
    out = sample.new_zeros(*sample.shape[:3], rows_axis.extent * scale, cols_axis.extent * scale)
    for i, top in enumerate(rows_axis.starts):
        span_rows = rows_axis.region(i) * scale
        for j, left in enumerate(cols_axis.starts):
            span_cols = cols_axis.region(j) * scale
            out[
                :,
                :,
                :,
                top * scale : top * scale + span_rows,
                left * scale : left * scale + span_cols,
            ] = tiles[i][j][:, :, :, :span_rows, :span_cols]
    return out


def tiled_decode_unclamped(
    vae: AutoencoderKLWan, latents: torch.Tensor, plan: TilePlan
) -> torch.Tensor:
    """タイル decode の参照（クランプ前）= {@link decode_tiles} → {@link assemble_tiles}。"""
    return assemble_tiles(decode_tiles(vae, latents, plan), plan)


# ---- 参照フィクスチャ ----------------------------------------------------------------


@dataclass(frozen=True)
class FixtureCase:
    """タイル decode のフィクスチャ 1 本（潜在の seed・chunk 数・空間）。"""

    name: str
    seed: int
    chunks: int
    #: 潜在の空間（縦 × 横）。
    height: int
    width: int
    #: `band` = 帯を決めるケース・`accept` = 受け入れを判定するケース（ADR 0118 追記 2026-10-02）。
    role: str
    #: 非タイルの参照も採る（タイル化の近似の差の観測 — 門ではない）。
    full: bool


#: 帯の決定用と受入れ用で、潜在（seed）・chunk 境界（chunk 数）・タイルの位置（縦横）を全て変える。
FIXTURE_CASES = (
    FixtureCase("band", seed=20261012, chunks=9, height=60, width=104, role="band", full=True),
    FixtureCase("accept", seed=20261013, chunks=3, height=104, width=60, role="accept", full=False),
)


def seeded_latents(vae: AutoencoderKLWan, case: FixtureCase) -> torch.Tensor:
    """固定 seed の乱数潜在を逆正規化した `[1, 16, F, H, W]`（上流 `WanPipeline` の decode 直前の
    値域 — `export_vae.fixture_latents` と同じ式）。"""
    generator = torch.Generator().manual_seed(case.seed)
    channels = int(vae.config.z_dim)
    mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(1, channels, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, dtype=torch.float32).view(1, channels, 1, 1, 1)
    noise = torch.randn(1, channels, case.chunks, case.height, case.width, generator=generator)
    return noise * std + mean


def _peak_rss_gib() -> float:
    """このプロセスの RSS の最大（GiB — Linux の `ru_maxrss` は KiB）。"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)


def _difference(got: torch.Tensor, want: torch.Tensor) -> dict[str, float]:
    difference = (got - want).abs()
    want_max = float(want.abs().max())
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "reference_abs_max": want_max,
        "ratio": float(difference.max()) / want_max,
    }


def write_fixture(
    vae: AutoencoderKLWan, case: FixtureCase, tile: int, out_root: Path
) -> dict[str, Any]:
    """1 ケースを書く（潜在 + タイル参照・band は非タイル参照も）。要約を返す。"""
    plan = plan_tiles(case.height, case.width, tile)
    latents = seeded_latents(vae, case)
    started = time.perf_counter()
    with torch.no_grad():
        frames = tiled_decode_unclamped(vae, latents, plan)[0]
    summary: dict[str, Any] = {
        "case": case.name,
        "role": case.role,
        "latents": list(latents.shape[1:]),
        "frames": list(frames.shape),
        "tiles": plan.tiles,
        "rows_starts": list(plan.rows.starts),
        "cols_starts": list(plan.cols.starts),
        "rows_blend": plan.rows.blends(plan.scale),
        "cols_blend": plan.cols.blends(plan.scale),
        "abs_max": float(frames.abs().max()),
        "tiled_seconds": round(time.perf_counter() - started, 1),
    }
    tensors = {"latents": latents[0].contiguous(), "frames": frames.contiguous()}
    metadata = {
        "seed": str(case.seed),
        "chunks": str(case.chunks),
        "role": case.role,
        "weights": "f16-rounded",
        "reference": "snapped-tile decode with upstream blend_v / blend_h before clamp (CPU f32)",
        **plan.meta(),
    }
    if case.full:
        started = time.perf_counter()
        with torch.no_grad():
            full = vae_patch.reference_decode_unclamped(vae, latents)[0]
        observed = {
            "unclamped": _difference(frames, full),
            "clamped": _difference(frames.clamp(-1.0, 1.0), full.clamp(-1.0, 1.0)),
            "full_seconds": round(time.perf_counter() - started, 1),
        }
        summary["vs_full"] = observed
        tensors["frames_full"] = full.contiguous()
        metadata["frames_full"] = "upstream non-tiled _decode chunk loop before clamp (CPU f32)"
    summary["peak_rss_gib"] = round(_peak_rss_gib(), 2)

    path = out_root / f"{FIXTURE_PREFIX}{case.name}{FIXTURE_SUFFIX}"
    staging = path.with_name(path.name + ".staging")
    out_root.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(staging), metadata=metadata)
    staging.replace(path)
    summary["path"] = str(path)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--tile",
        type=int,
        default=export_vae.DEFAULT_TILE,
        help="潜在タイルの辺（既定 32 — export した chunk グラフの入力形と揃える）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=SERIES_ROOT / export_vae.SERIES_NAME,
        help="系列の根（既定は決定 7 の系列）",
    )
    names = [case.name for case in FIXTURE_CASES]
    parser.add_argument(
        "--case", action="append", choices=names, default=None, help="書くケース（既定は全部）"
    )
    args = parser.parse_args(argv)
    if args.tile <= 0:
        parser.error(f"--tile は正の整数（{args.tile}）")

    vae = export_vae.load_vae(round_f16=True)
    if vae.use_tiling:
        # MUST: 上流のタイル化は走査形が違う（モジュール doc）。参照は自前の幾何でだけ採る。
        raise SystemExit("vae.use_tiling が True — 上流のタイル化は使わない")
    selected = [case for case in FIXTURE_CASES if args.case is None or case.name in args.case]
    summary = {
        "series": str(args.out),
        "fixtures": [write_fixture(vae, case, args.tile, args.out) for case in selected],
    }
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
