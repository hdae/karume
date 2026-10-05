"""Wan の動画 VAE の**タイル decode**（幾何の Python 側の正と参照フィクスチャ — ADR 0118 / 0121）。

chunk グラフ（`wan/vae_patch.py`）の空間は潜在 `t×t` の固定タイル（資産の入力形・既定は系列の
表 `export_vae.VAE_SERIES` — 2.1 は 32・2.2 は 16）。
全画面（832×480 = 潜在 60×104）はホストがタイルに切って decode し、重なりをブレンドして
貼り合わせる（常時タイル — ADR 0118 決定 2・ADR 0033 / 0038 §4 の動画版）。ここはその
**幾何とブレンドの Python 側の正**と、GPU の照合
（2.1 は `packages/models/tests/e2e_wan_vae_tiles_test.ts`・2.2 は
`packages/models/tests/e2e_wan_ti2v_vae_tiles_test.ts`）が読む参照フィクスチャの台本。
TS 側は `packages/models/src/wan/vae-tiles.ts`。

## 幾何 = 丸め等間隔スナップ配置（上流からの意図的な逸脱）

上流 `AutoencoderKLWan.tiled_decode` は `range(0, H, stride)` で走査するので最後のタイルが
短くなり、固定形の chunk グラフでは食えない。開始位置は 0 と `extent − tile` の間を**丸めて
等分**する（Anima と同じ規則 — ADR 0033 追記 P-3・`anima/tiling.py` の `plan_tile_axis`）。
本数は「隣り合う対の重なりが下限以上」を満たす最小。下限の正本は出力の {@link MIN_OVERLAP_PX}
で、潜在へは空間の圧縮（グラフの比 × unpatchify の倍率）で割って導く（{@link min_overlap_latent}
— ADR 0121 決定 6）。Wan2.1 は {@link WAN21_MIN_OVERLAP_LATENT} = 8。832×480（潜在 60×104・
タイル 32）は行 0 / 14 / 28 × 列 0 / 24 / 48 / 72 の 12 枚（重なりは行 18 潜在・列 8 潜在）。

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
- Wan2.2（上流 config の `patch_size` = 2）は、ブレンドと貼り付けを chunk グラフの出口の
  **patchify 空間**（12 ch）で行い、unpatchify → クランプはその後（上流 `tiled_decode` と同じ順 —
  ブレンド幅も patchify 空間の画素で、上流の 256/2 − 192/2 = 32 と単位が一致する）。参照は
  unpatchify とクランプの前を書く（{@link unpatchify_frames} はテストと `frames_rgb` のためだけ）。

## タイルが外・chunk が内

タイルごとに cache を作り直す（上流 `tiled_decode` の `clear_cache` と同じ — chunk が外だと
タイルの枚数ぶんの cache を同時に持つ。決定 2）。post-quant の 1×1×1 conv もタイル ×
フレームごとに掛ける（上流 `tiled_decode` の形のまま。chunk グラフもグラフの中でフレームごとに
掛ける）。

## フィクスチャ（系列の根の `vae_tiles.<case>.safetensors`）

Wan2.1:

- `band`（帯を決める）: seed 固定の潜在 `[16,9,60,104]`（832×480・33 フレーム）のタイル参照
  `frames` と、**非タイル**の参照 `frames_full`（上流の非タイル `_decode` の chunk ループ・
  クランプ前 — タイル化の近似の差の観測用。門ではない）。
- `accept`（受け入れを判定する）: 別の seed・**縦長** `[16,3,104,60]`（480×832・9 フレーム）。
  タイルの位置も chunk 境界も band と違う。行と列の取り違えは縦横が違う形でしか値に出ない
  （正方では対合）。
- メタに幾何（開始位置・ブレンド幅）を書く — TS の計画と突き合わせる（ADR 0033 追記 9a の
  二重凍結の上に、フィクスチャでの突き合わせを足す）。

Wan2.2（`--model ti2v-5b` — {@link TI2V_FIXTURE_CASES}・タイル 16・重なり潜在 4）は
`band`（832×480×81）・`accept`（480×832×9・非タイルの参照つき）・`wide`（1280×704×5 — 対ごとに
ブレンド幅が違う寸法）の 3 本。`frames` は patchify 空間・クランプ前で、メタに `patch_size` を足す。
`accept` だけ、上流の unpatchify → クランプを当てた RGB `frames_rgb [3, F, H, W]` も書く（GPU の
VAE 段の末尾〈ホストの unpatchify〉を上流の関数そのものと照合するため）。2.1 のフィクスチャは
テンソルもメタも 2.2 の経路を足す前と同じ式から出る。

重みは f16 表現可能値へ丸めてから参照を採る（ADR 0006 — `export_vae.load_vae(round_f16=True)`）。

    uv run --group wan --inexact python -m wan.vae_tiling                # band + accept
    uv run --group wan --inexact python -m wan.vae_tiling --case accept  # 1 本だけ
    uv run --group wan --inexact python -m wan.vae_tiling --model ti2v-5b  # Wan2.2

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
from wan.sources import DEFAULT_MODEL

if TYPE_CHECKING:
    from diffusers import AutoencoderKLWan

#: 隣り合うタイルの出力画素での最小の重なり。上流の既定のブレンド幅
#: `tile_sample_min − tile_sample_stride` = 256 − 192 = 64 px と同じ。潜在の重なりはここから
#: {@link min_overlap_latent} で導く（世代ごとに潜在の値を持たない — ADR 0121 決定 6）。
MIN_OVERLAP_PX = 64
#: Wan2.1 の VAE の unpatchify の倍率（上流 `config.patch_size` は None = 1 — unpatchify しない）。
WAN21_VAE_PATCH_SIZE = 1


def _is_positive_int(value: object) -> bool:
    # bool は int の部分型だが、TS 側の `Number.isInteger` は真偽値を整数と見ない — 揃えて拒む。
    # 型として int を要求するので TS より厳しい（TS は 8.0 を通すが、ここで通すと float が流れる）。
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def min_overlap_latent(scale: int, patch_size: int) -> int:
    """潜在の重なりの下限 = {@link MIN_OVERLAP_PX} ÷ 空間の圧縮（ADR 0121 決定 6）。

    空間の圧縮 = グラフの入出力の空間比 `scale` × ホストの unpatchify の倍率 `patch_size`。
    圧縮をグラフの比だけで取ると、unpatchify のある世代で重なりが倍になる（2.2 の 8 × 2 = 16 が
    8 になり潜在 8 = 128 px）ので、2 つを別の引数で受ける。

    MUST: 割り切れない圧縮は丸めずに落とす（丸めると重なりが 64 px から黙ってずれる）。
    """
    if not (_is_positive_int(scale) and _is_positive_int(patch_size)):
        raise ValueError(
            f"空間の圧縮の因子が正の整数でない（グラフの比 {scale!r}・unpatchify {patch_size!r}）"
        )
    compression = scale * patch_size
    if MIN_OVERLAP_PX % compression:
        raise ValueError(
            f"重なり {MIN_OVERLAP_PX} px が空間の圧縮 {compression}"
            f"（グラフの比 {scale} × unpatchify {patch_size}）で割り切れない"
        )
    return MIN_OVERLAP_PX // compression


#: Wan2.1 の重なり（潜在）= 64 ÷ (`export_vae.SPATIAL_SCALE` 8 × 1) = 8。Python 側には開いた
#: グラフが無いので、グラフの比は recipe の宣言値（export する chunk グラフのフレームの辺
#: `tile * SPATIAL_SCALE` と同じ値）を使う。
WAN21_MIN_OVERLAP_LATENT = min_overlap_latent(export_vae.SPATIAL_SCALE, WAN21_VAE_PATCH_SIZE)

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


def plan_tile_axis(extent: int, tile: int, min_overlap: int) -> TileAxis:
    """1 軸ぶんの丸め等間隔スナップ配置（TS 側 `planWanVaeTileAxis` と同じ規則）。

    本数は「重なりが `min_overlap` 以上」を満たす最小値 `ceil(span / (tile − min_overlap)) + 1`
    （`span = extent − tile`）、開始位置は `round(i · span / (本数 − 1))`（0.5 は切り上げ）。

    MUST: `min_overlap` に既定値を置かない — 世代で値が違う（{@link min_overlap_latent}）ので、
    渡し忘れが 2.1 の値で黙って通ると別の世代の計画が静かにずれる。
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
    min_overlap: int,
    scale: int = export_vae.SPATIAL_SCALE,
) -> TilePlan:
    """潜在の空間 `height × width` に対するタイル計画（軸ごとに独立・`min_overlap` は潜在）。"""
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
# final crop by the region assignment; clamp and unpatchify are left to the consumer after
# assembly.
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

    `latents` は逆正規化済みの潜在 `[1, z, F, H, W]`。戻りは行優先のタイル
    `[1, C, 1 + 4(F−1), s, s]`（s = タイル辺 × 縮尺・C は出口の空間のチャネル — 2.1 は画素の 3、
    2.2 は patchify 空間の 12）。中身は上流 `tiled_decode` の内側のループの逐語で、走査だけが
    スナップ配置（モジュール doc）。

    MUST: 世代は chunk グラフの書き直しが受けるものだけ（`vae_patch.assert_supported` —
    参照は chunk グラフと照合するためにある）。
    """
    vae_patch.assert_supported(vae)
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
    """decode 済みのタイル（行優先）をブレンドして `[1, C, F', H·s, W·s]` に貼り合わせる。

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


def unpatchify_frames(frames: torch.Tensor, patch_size: int | None) -> torch.Tensor:
    """出口の空間 `[1, C·p², F, h, w]` → 画素 `[1, C, F, h·p, w·p]`（上流の `unpatchify`）。

    `patch_size` が None（2.1 — unpatchify しない世代）なら `frames` をそのまま返す。それ以外は
    上流の関数そのものを呼ぶ（自前の写しにすると、参照の素性の主張が自分との比較になる）。
    フィクスチャの `frames` には掛けない（参照は unpatchify の前を書く — モジュール doc）。
    """
    if patch_size is None:
        return frames
    from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

    return unpatchify(frames, patch_size=patch_size)


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
    #: 上流の unpatchify → クランプを当てた RGB `frames_rgb` も書く（patchify する世代だけ —
    #: GPU の VAE 段の末尾を上流の関数そのものと照合する）。
    rgb: bool = False


#: 帯の決定用と受入れ用で、潜在（seed）・chunk 境界（chunk 数）・タイルの位置（縦横）を全て変える。
FIXTURE_CASES = (
    FixtureCase("band", seed=20261012, chunks=9, height=60, width=104, role="band", full=True),
    FixtureCase("accept", seed=20261013, chunks=3, height=104, width=60, role="accept", full=False),
)

#: Wan2.2 TI2V-5B のケース（タイル 16・潜在 48 ch）。2.1 と同じ組み立てで、seed は 2.1 とも段 4 の
#: chunk 列（`export_vae.TI2V_FIXTURE_CASES`）とも別:
#:
#: - `band`: 832×480×81（潜在 30×52・21 chunk）— 帯を決める 1 本（ADR 0121 の検収表の段 5）。
#: - `accept`: 縦長 480×832×9（潜在 52×30・3 chunk）— タイルの位置も chunk 境界も band と違う。
#:   非タイルの参照（観測）と RGB の `frames_rgb` を持つ。
#: - `wide`: 1280×704×5（潜在 44×80・2 chunk）— 対ごとにブレンド幅が違う（行 56 / 48 / 56・
#:   列 40 / 48 / 40 / 40 / 48 / 40）のはこの寸法だけ（832×480 は対ごとに一様）。
TI2V_FIXTURE_CASES = (
    FixtureCase("band", seed=20261051, chunks=21, height=30, width=52, role="band", full=False),
    FixtureCase(
        "accept", seed=20261052, chunks=3, height=52, width=30, role="accept", full=True, rgb=True
    ),
    FixtureCase("wide", seed=20261053, chunks=2, height=44, width=80, role="accept", full=False),
)

#: モデル名（`export_vae.VAE_SERIES` のキー）→ ケースの表。
FIXTURE_CASES_BY_MODEL: dict[str, tuple[FixtureCase, ...]] = {
    "t2v-1.3b": FIXTURE_CASES,
    "ti2v-5b": TI2V_FIXTURE_CASES,
}

#: フィクスチャのメタ `reference`（参照の素性）。2.1 の値は 2.2 の経路を足す前と同じ文字列。
FIXTURE_REFERENCE = "snapped-tile decode with upstream blend_v / blend_h before clamp (CPU f32)"
FIXTURE_REFERENCE_IN_PATCHIFY_SPACE = (
    "snapped-tile decode with upstream blend_v / blend_h in patchify space"
    " before unpatchify and clamp (CPU f32)"
)


def seeded_latents(vae: AutoencoderKLWan, case: FixtureCase) -> torch.Tensor:
    """固定 seed の乱数潜在を逆正規化した `[1, z, F, H, W]`（上流 `WanPipeline` の decode 直前の
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
    vae: AutoencoderKLWan,
    case: FixtureCase,
    tile: int,
    out_root: Path,
    *,
    min_overlap: int,
    patch_size: int | None,
) -> dict[str, Any]:
    """1 ケースを書く（潜在 + タイル参照・`full` は非タイル参照も・`rgb` は RGB も）。要約を返す。

    MUST: `min_overlap`（潜在）と `patch_size` に既定値を置かない — 渡し忘れが 2.1 の値で黙って
    通ると、別の世代のフィクスチャが静かにずれる。`patch_size` は上流 config と一致しなければ
    落とす（メタの `patch_size` と `reference` が実物と食い違うのを防ぐ）。
    """
    if vae.config.patch_size != patch_size:
        raise ValueError(
            f"patch_size {patch_size!r} が上流 config の {vae.config.patch_size!r} と違う"
        )
    if case.rgb and patch_size is None:
        raise ValueError(
            f"ケース {case.name} の frames_rgb は patchify する世代だけ（patch_size が None）"
        )
    plan = plan_tiles(case.height, case.width, tile, min_overlap)
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
        "reference": (
            FIXTURE_REFERENCE if patch_size is None else FIXTURE_REFERENCE_IN_PATCHIFY_SPACE
        ),
        **plan.meta(),
    }
    if patch_size is not None:
        metadata["patch_size"] = str(patch_size)
    if case.rgb:
        # 上流 `_decode` の末尾と同じ順（unpatchify → clamp）。unpatchify は置換なので、patchify
        # 空間の帯がそのまま使える。
        rgb = unpatchify_frames(frames[None], patch_size)[0].clamp(-1.0, 1.0)
        summary["frames_rgb"] = list(rgb.shape)
        tensors["frames_rgb"] = rgb.contiguous()
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
        metadata["frames_full"] = (
            "upstream non-tiled _decode chunk loop before clamp (CPU f32)"
            if patch_size is None
            else "upstream non-tiled _decode chunk loop before unpatchify and clamp (CPU f32)"
        )
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
        "--model",
        choices=sorted(export_vae.VAE_SERIES),
        default=DEFAULT_MODEL,
        help=f"モデル（系列・タイル・ケースは表から — 既定 {DEFAULT_MODEL}）",
    )
    tiles = "・".join(
        f"{model} は {series.tile}" for model, series in export_vae.VAE_SERIES.items()
    )
    parser.add_argument(
        "--tile",
        type=int,
        default=None,
        help=f"潜在タイルの辺（既定は表 — {tiles}。export した chunk グラフの入力形と揃える）",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="系列の根（既定はモデルの系列 — 決定 7 の表の値）"
    )
    # choices は全モデルの和集合（モデルごとに表が違う — `wide` は 2.2 だけ）。選んだモデルの表に
    # 無い名前は parse の後で落とす。
    names = list(
        dict.fromkeys(case.name for cases in FIXTURE_CASES_BY_MODEL.values() for case in cases)
    )
    parser.add_argument(
        "--case", action="append", choices=names, default=None, help="書くケース（既定は全部）"
    )
    args = parser.parse_args(argv)
    series = export_vae.VAE_SERIES[args.model]
    cases = FIXTURE_CASES_BY_MODEL[args.model]
    unknown = sorted(set(args.case or ()) - {case.name for case in cases})
    if unknown:
        parser.error(
            f"--case {unknown} は --model {args.model} の表に無い"
            f"（{[case.name for case in cases]}）"
        )
    tile = series.tile if args.tile is None else args.tile
    if tile <= 0:
        parser.error(f"--tile は正の整数（{tile}）")
    out = args.out if args.out is not None else SERIES_ROOT / series.series
    min_overlap = min_overlap_latent(export_vae.SPATIAL_SCALE, series.patch_size or 1)
    if tile <= min_overlap:
        # 計画（`plan_tile_axis`）でも落ちるが、それは VAE を読んだ後 — 読む前に落とす。
        parser.error(f"--tile {tile} が重なりの下限 {min_overlap}（潜在）以下")

    vae = export_vae.load_vae(args.model, round_f16=True)
    export_vae.assert_series_config(vae, series)
    if vae.use_tiling:
        # MUST: 上流のタイル化は走査形が違う（モジュール doc）。参照は自前の幾何でだけ採る。
        raise SystemExit("vae.use_tiling が True — 上流のタイル化は使わない")
    selected = [case for case in cases if args.case is None or case.name in args.case]
    summary = {
        "series": str(out),
        "fixtures": [
            write_fixture(
                vae, case, tile, out, min_overlap=min_overlap, patch_size=series.patch_size
            )
            for case in selected
        ],
    }
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
