"""Depth Anything V2（単一画像の相対深度推定）を torch.export → IR v1 まで通すパッチ層。

対象は transformers 移植版（`depth-anything/Depth-Anything-V2-Small-hf` の
`DepthAnythingForDepthEstimation` = DINOv2 backbone + DPT neck / head）。差し替えるのは
transformers のクラス属性と、モデル内の `nn.ConvTranspose2d` インスタンスだけで、
重みは 1 要素も書き換えない。

差し替えは**全て第 0 層**（新しい op を足さない）で、性格は 2 群に分かれる:

- **ビット同一（演算を 1 つも増減させない）**
  ① `DepthAnythingFeatureFusionStage.forward` の**最終段だけ** `size=None` になり、
     融合層が `F.interpolate(..., scale_factor=2, align_corners=True)` を呼ぶ
     （`aten.upsample_bilinear2d.vec` の `scale_factors` 指定形 = 変換段が受理しない —
     出力長の丸め規約が 2 通りになるため。`aten_handlers._h_upsample_bilinear2d`）。静的形なので
     出力寸法を先に計算して `size=` へ倒す。**align_corners=True では倍率が使われない**
     （aten の `area_pixel_compute_scale` は `(in−1)/(out−1)` を作る）ので、値はビット同一。
  ② `Dinov2Embeddings.interpolate_pos_encoding` を**恒等 or 落とす**へ。原実装は
     `num_patches == num_positions and height == width` のとき位置埋め込みをそのまま返し、
     外れると `aten.upsample_bicubic2d`（語彙に無い）を出す。事前学習解像度（518² =
     37×37 パッチ）に固定すれば前者しか通らないが、**前提が外れたときに黙って bicubic へ
     落ちる**形なので、条件を反転して fail loudly にする（値は原実装と同一）。
- **ビット同一ではない（式は同値・丸めだけ違う）— 実測値は下記**
  ③ DPT reassemble の `nn.ConvTranspose2d`（kernel == stride == 4 / 2・padding 0）を
     {@link SubPixelUpsample}（1×1 conv + pixel shuffle 相当の rank-4 並べ替え 2 本）へ。
     karume の conv_transpose は 1d だけ・しかも `2·padding == K − stride` 形限定なので、
     この形は**分解でしか通らない**。kernel == stride のとき転置畳み込みは重なりが無く、
     出力の各位置が 1 本の内積になるので、重みを `[Cin, Cout, r, r] → [Cout·r·r, Cin, 1, 1]`
     へ並べ替えた 1×1 conv と pixel shuffle の合成に**厳密に**一致する（添字の同値は
     `depth_anything/tests/test_patch.py` が整数値データの `torch.equal` で固定する）。
     f32 の実データでは Cin 方向の**縮約順序**が ATen の転置畳み込みと違うぶん最下位ビットが
     動く（実測 max 1.4e-06 / 値の RMS ≈ 1.0）。

入口は 2 群に対応して分かれている（{@link apply_layout_patches} / {@link apply_module_patches}
・両方当てるのが {@link apply}）— 一括でしか当てられないと ①② のビット同一を単体で
実測できない（`birefnet.patch` と同じ規律）。

MUST: どのパッチも fallback を持たない。前提（kernel == stride / 事前学習解像度）を外した
モデルは黙って別の数値経路へ落ちるのではなく、その場で落ちる。

NOTE: ①② はクラス属性のプロセス全域差し替えなので、「パッチ前の参照」を採れるのは
1 プロセスにつき 1 回だけ（{@link patches_applied} が門）。
"""

from __future__ import annotations

import sys
from typing import Any

import torch
from torch import nn

#: 期待する深度の型（`metric` は head が sigmoid × max_depth になり、グラフも意味も別物）。
EXPECTED_DEPTH_ESTIMATION_TYPE = "relative"

#: 期待する backbone（DPT neck の入力は `Dinov2Backbone` の feature_maps 前提）。
EXPECTED_BACKBONE_TYPE = "dinov2"

#: 最終融合段の倍率（原実装の `modifier = {"scale_factor": 2}`）。
LAST_FUSION_SCALE = 2

#: 差し替えを当てたか（パッチ前の参照を採る側の恒真化を防ぐ門）。
_APPLIED = False


def patches_applied() -> bool:
    """このプロセスでクラス属性の差し替えを当てたか。"""
    return _APPLIED


def assert_supported(model: nn.Module) -> None:
    """差し替え版が前提にしている構成を検査する。

    見るのは**外れても shape エラーにならずに別の経路へ落ちる**ものだけ
    （`birefnet.patch.assert_supported` と同じ流儀）。
    """
    config = model.config
    if config.depth_estimation_type != EXPECTED_DEPTH_ESTIMATION_TYPE:
        raise ValueError(
            f"depth_estimation_type が {config.depth_estimation_type!r}"
            f"（{EXPECTED_DEPTH_ESTIMATION_TYPE!r} 以外は head が別経路）— 差し替え版の対象外"
        )
    backbone_config = getattr(config, "backbone_config", None)
    if backbone_config is None or backbone_config.model_type != EXPECTED_BACKBONE_TYPE:
        raise ValueError(f"backbone が {EXPECTED_BACKBONE_TYPE!r} でない — 差し替え版の対象外")
    if not backbone_config.apply_layernorm:
        raise ValueError(
            "apply_layernorm=False — feature_maps に最終 LayerNorm が掛からない"
            "（DPT neck が受け取る値が黙って変わる）"
        )
    if backbone_config.reshape_hidden_states:
        raise ValueError(
            "reshape_hidden_states=True — feature_maps が既に [B,C,H,W] で"
            " reassemble の cls 除去と reshape が二重に掛かる"
        )


def pretrained_resolution(model: nn.Module) -> int:
    """位置埋め込みの補間が**起きない**唯一の入力解像度（= backbone の `image_size`）。

    ② の門が受けるのはこの 1 点だけ（`num_patches == num_positions and height == width`）。
    """
    image_size = model.config.backbone_config.image_size
    if isinstance(image_size, (list, tuple)):
        raise ValueError(f"image_size={image_size!r} は差し替え版が持たない経路（正方のみ）")
    return int(image_size)


# ---- ①最終融合段の倍率指定を size 指定へ -----------------------------------


def _feature_fusion_stage_forward(
    self: nn.Module, hidden_states: Any, size: Any = None
) -> list[torch.Tensor]:
    """`DepthAnythingFeatureFusionStage.forward` の同値実装（最終段も `size=` で呼ぶ）。

    元実装との差は 1 点だけで、値は変わらない: 最終段の `size=None`（= 融合層が
    `scale_factor=2` を使う）を、入力の空間寸法から計算した `(2H, 2W)` へ倒す。
    `align_corners=True` の座標式は入出力の寸法だけで決まるので、倍率を渡す形と**ビット同一**
    （その土台は `depth_anything/tests/test_patch.py` が直に実測する）。

    NOTE: 融合層のもう 1 つの resize 枝（残差の形が違うときの `align_corners=False` —
    IR 語彙に無い変種）はこの段では**構造的に死んでいる**ので、ガードを置かない: 段 `i` の
    出力は `hidden_states[i+1]` の寸法へ揃えられ、その `hidden_states[i+1]` がそのまま段 `i+1`
    の残差になる（チャネルは neck の conv が全段 `fusion_hidden_size` に揃える）。万一これが
    崩れても、変換段の `_h_upsample_bilinear2d` が `align_corners=False` を落とす。
    """
    hidden_states = hidden_states[::-1]

    fused_hidden_states: list[torch.Tensor] = []
    fused_hidden_state: torch.Tensor | None = None

    for index, (hidden_state, layer) in enumerate(zip(hidden_states, self.layers, strict=True)):
        if index != len(hidden_states) - 1:
            target_size = tuple(int(dim) for dim in hidden_states[index + 1].shape[2:])
        else:
            source = hidden_state if fused_hidden_state is None else fused_hidden_state
            target_size = tuple(int(dim) * LAST_FUSION_SCALE for dim in source.shape[2:])

        if fused_hidden_state is None:
            fused_hidden_state = layer(hidden_state, size=target_size)
        else:
            fused_hidden_state = layer(fused_hidden_state, hidden_state, size=target_size)

        fused_hidden_states.append(fused_hidden_state)

    return fused_hidden_states


# ---- ②位置埋め込みの補間（恒等 or 落とす） ---------------------------------


def _interpolate_pos_encoding(
    self: nn.Module, embeddings: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """`Dinov2Embeddings.interpolate_pos_encoding` の同値実装（補間が要る形は落とす）。

    元実装は同じ条件で `self.position_embeddings` をそのまま返す枝を持つ（ビット同一）。
    外れた場合に元実装が出す `aten.upsample_bicubic2d` は語彙に無く、そのまま export すると
    変換段まで気づけないので、ここで落とす。
    """
    num_patches = int(embeddings.shape[1]) - 1
    num_positions = int(self.position_embeddings.shape[1]) - 1
    if num_patches != num_positions or height != width:
        raise NotImplementedError(
            f"パッチ数 {num_patches} が位置埋め込みの {num_positions} と違う"
            f"（入力 {height}×{width}）— 位置埋め込みの bicubic 補間は語彙に無い。"
            " 事前学習解像度の正方入力だけが差し替え版の対象"
        )
    return self.position_embeddings


# ---- ③conv_transpose2d（kernel == stride）→ 1×1 conv + pixel shuffle -------


class SubPixelUpsample(nn.Module):
    """`nn.ConvTranspose2d(Cin, Cout, kernel_size=r, stride=r, padding=0)` の同値形。

    kernel == stride のとき出力窓は重ならないので、転置畳み込みは

        y[b, co, i·r + ki, j·r + kj] = Σ_ci x[b, ci, i, j]·W[ci, co, ki, kj] + bias[co]

    という「1 本の内積 + 並べ替え」に分解できる。`[Cout·r·r, Cin, 1, 1]` へ並べ替えた重みの
    1×1 conv が内積を、pixel shuffle 相当の並べ替えが添字を担う。bias は空間に依存しない
    ので、`(co, ki, kj)` へ複製して conv 側へ畳み込める（`+0` ではなく**同じ値**を足す）。

    pixel shuffle をそのまま書くと rank-6 の view + permute になり、IR の strided 表現の
    rank 上限（4）を超える。ここでは **rank-4 の並べ替え 2 本**（W 方向 → H 方向）に割る:

        [B, C·r·r, H, W] → [B·C·r, r, H, W] → permute(0,2,3,1) → [B·C, r, H, W·r]
                         → permute(0,2,1,3) → [B, C, H·r, W·r]

    MUST: `output_padding` / `groups` / `dilation` / `padding` が既定でない
    `ConvTranspose2d` は受けない（分解が成り立たない）。
    """

    def __init__(self, source: nn.ConvTranspose2d) -> None:
        super().__init__()
        strides = set(source.stride)
        if len(strides) != 1 or set(source.kernel_size) != strides:
            raise NotImplementedError(
                f"kernel_size={source.kernel_size} と stride={source.stride} が"
                "「H/W 共通の同じ値」でない ConvTranspose2d は差し替え対象外"
                "（重なる窓は 1×1 conv へ分解できない）"
            )
        (stride,) = strides
        for what, value in (
            ("padding", source.padding),
            ("output_padding", source.output_padding),
            ("dilation", tuple(dim - 1 for dim in source.dilation)),
        ):
            if set(value) != {0}:
                raise NotImplementedError(f"{what}={value} の ConvTranspose2d は差し替え対象外")
        if source.groups != 1:
            raise NotImplementedError(f"groups={source.groups} の ConvTranspose2d は差し替え対象外")
        if source.bias is None:
            raise NotImplementedError("bias 無しの ConvTranspose2d は差し替え対象外")

        weight = source.weight.detach()  # [Cin, Cout, r, r]
        in_channels, out_channels = weight.shape[0], weight.shape[1]
        self.scale = int(stride)
        self.out_channels = int(out_channels)
        self.register_buffer(
            "weight",
            weight.permute(1, 2, 3, 0)
            .reshape(out_channels * stride * stride, in_channels, 1, 1)
            .contiguous(),
        )
        self.register_buffer(
            "bias",
            source.bias.detach()
            .reshape(out_channels, 1)
            .expand(out_channels, stride * stride)
            .reshape(-1)
            .contiguous(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale, channels = self.scale, self.out_channels
        projected = nn.functional.conv2d(x, self.weight, self.bias)
        batch, _, height, width = projected.shape
        columns = projected.reshape(batch * channels * scale, scale, height, width)
        columns = columns.permute(0, 2, 3, 1)
        columns = columns.reshape(batch * channels, scale, height, width * scale)
        rows = columns.permute(0, 2, 1, 3)
        return rows.reshape(batch, channels, height * scale, width * scale)


def _replace_transposed_convs(model: nn.Module) -> int:
    """`nn.ConvTranspose2d` を {@link SubPixelUpsample} へ差し替える（インスタンス走査）。"""
    replaced = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.ConvTranspose2d):
                setattr(parent, name, SubPixelUpsample(child))
                replaced += 1
    return replaced


# ---- 適用 ------------------------------------------------------------------


def apply_layout_patches(model: nn.Module) -> dict[str, int]:
    """ビット同一の差し替え ①② を当てる（transformers のクラス属性 2 本）。

    この段だけならパッチ前 eager と**ビット同一**で、`depth_anything.export.py --verify` の
    1 点目がその主張を実測する。③ と分けて公開しているのはそのため。
    """
    global _APPLIED
    assert_supported(model)
    neck = sys.modules[type(model.neck).__module__]
    embeddings = sys.modules[type(model.backbone.embeddings).__module__]
    neck.DepthAnythingFeatureFusionStage.forward = _feature_fusion_stage_forward
    embeddings.Dinov2Embeddings.interpolate_pos_encoding = _interpolate_pos_encoding
    _APPLIED = True
    return {"fusion_stage": 1, "interpolate_pos_encoding": 1}


def apply_module_patches(model: nn.Module) -> dict[str, int]:
    """丸めが動く差し替え ③ を当てる（`ConvTranspose2d` のインスタンス走査）。"""
    return {"conv_transpose2d": _replace_transposed_convs(model)}


def apply(model: nn.Module) -> dict[str, int]:
    """差し替えを全て当てる（export 経路 — ①②③）。戻り値は差し替えた数。"""
    return {**apply_layout_patches(model), **apply_module_patches(model)}
