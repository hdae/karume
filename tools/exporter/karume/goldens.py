"""tiny golden fixtures の生成（後段の Deno E2E テストがこれを読む）。

固定 seed の小モデルを torch.export → IR v1 コンテナ化し、torch CPU での期待出力を
同じディレクトリに置く。契約表の全 op を全モデル合計で必ず被覆する（COVERAGE 検査）。

    uv run python -m karume.goldens

レイアウトとテンソルキー命名は README.md「golden レイアウト」に明記。
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import nn
from torch.export import Dim

from karume.convert import (
    PRESERVED_OP_PREFIXES,
    PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    normalize_boundary_tensor,
)
from karume.ir import IrGraph
from karume.ops import EMITTABLE_OPS
from karume.pipeline import export_to_file
from karume.quantize import fake_quant_int8

#: 生成物の既定の置き場（リポジトリ直下 packages/runtime/tests/fixtures/golden/）。
REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_ROOT = REPO_ROOT / "packages" / "runtime" / "tests" / "fixtures" / "golden"

MODEL_FILE = "model.safetensors"
IO_FILE = "io.safetensors"
#: io.safetensors のキー規約。入力はグラフ入力名、出力は graph.outputs の位置で引く。
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."

#: 乱数は全てここから引く（torch.Generator を明示 — グローバル seed に依存しない）。
SEED = 20260802
#: 記号次元 T の golden 実長。torch.export の 0/1 特殊化を避けるため 2 以上。
GOLDEN_T = 6
_DYNAMIC_T = Dim("T", min=2, max=64)
#: Tmax 畳み込みを踏む golden 用の小さい上限（焼いた定数は Tmax² で膨らむ — fixture を
#: 16KiB 未満に保つ）。T = GOLDEN_T はこれより十分小さく、prefix が本当に短くなる。
_SMALL_DYNAMIC_T = Dim("T", min=2, max=24)


def _rng() -> torch.Generator:
    return torch.Generator().manual_seed(SEED)


def _uniform(generator: torch.Generator, *shape: int, low: float, high: float) -> torch.Tensor:
    return torch.rand(shape, generator=generator) * (high - low) + low


def _mask(*shape: int, zero_at: int) -> torch.Tensor:
    """1 本だけ 0 を含む i64 の 0/1 マスク（torch 既定の整数 dtype = int64）。"""
    mask = torch.ones(shape, dtype=torch.int64).reshape(-1)
    mask[zero_at] = 0
    return mask.reshape(shape)


def _band_mask(length: int, *, blocked_row: int) -> torch.Tensor:
    """attention 用の i64 マスク `[1,1,T,T]`（1 = 残す / 0 = マスク）。

    対角の近傍だけを残す帯マスクにしたうえで、`blocked_row` の行は**全て 0**（全マスク）に
    する。全マスク行は masked_fill で全要素が -3.4e38 になり、safe-softmax なら一様分布、
    素朴形なら NaN になる — limitations.md の「identity は ±F32_MAX」との相互作用を踏む形。
    """
    rows = torch.arange(length).reshape(length, 1)
    cols = torch.arange(length).reshape(1, length)
    mask = ((rows - cols).abs() <= 1).to(torch.int64)
    mask[blocked_row] = 0
    return mask.reshape(1, 1, length, length)


class UnaryChain(nn.Module):
    """neg / abs / sqrt / log / exp（log と sqrt の定義域は入力生成側で保証する）。"""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        magnitude = torch.abs(torch.neg(x))
        logged = torch.log(torch.sqrt(magnitude))
        return logged, torch.exp(logged)


class Activations(nn.Module):
    """tanh / sigmoid / relu / gelu / gelu_tanh / sin（gelu 2 種は分解を止めて 1 ノードで運ぶ）。

    `sin` は活性ではないが、**実行時値**を取る単項の f32 elementwise という点で同型なので
    ここに相乗りする（定数入力の `sin` は FOLDABLE_OPS が畳んでノードを残さない — 発行経路を
    踏ませるには placeholder 由来の値を通す必要がある）。
    """

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (
            torch.tanh(x),
            torch.sigmoid(x),
            torch.relu(x),
            nn.functional.gelu(x),
            nn.functional.gelu(x, approximate="tanh"),
            torch.sin(x),
        )


class BroadcastBinary(nn.Module):
    """add / sub / mul / div を torch 右詰め broadcast で踏む（rank2 × rank1）。

    `offset` は Parameter でも buffer でもない素の属性 — torch.export が lifted tensor
    constant として持ち上げる経路（initializer 化）を通す。
    """

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        self.offset = _uniform(generator, 1, 4, low=-0.5, high=0.5)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return x + y, x - self.offset, x * y, x / y


class Mlp(nn.Module):
    """rank-2 matmul の 2 層 MLP。重みは全て initializer 経由で運ぶ。"""

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(8, 16, generator=generator) * 0.3)
        self.b1 = nn.Parameter(torch.randn(16, generator=generator) * 0.1)
        self.w2 = nn.Parameter(torch.randn(16, 4, generator=generator) * 0.3)
        self.b2 = nn.Parameter(torch.randn(4, generator=generator) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(x @ self.w1 + self.b1)
        return hidden @ self.w2 + self.b2


class RowReduce(nn.Module):
    """最終次元の行 reduce（keepdim 無し）— sum / amax / amin。"""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return x.sum(dim=-1), x.amax(dim=-1), x.amin(dim=-1)


class MaskChain(nn.Module):
    """DeBERTa front の mask 経路を縮めたもの（recon §3-8）。

    mask 外積 mul(i64) → 真偽化 cast → bitwise_not → 重み化 cast → f32 の重み掛け。
    外積を unsqueeze 無しで踏むため、列/行の mask を [T,1] / [1,T] で受け取る。
    bool 出力を 1 本持たせて非 f32 の readback 経路も golden に載せる。
    """

    def forward(
        self, mask_col: torch.Tensor, mask_row: torch.Tensor, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair = mask_col * mask_row
        keep = pair.to(torch.bool)
        drop = torch.bitwise_not(keep)
        return scores * keep.to(torch.float32), drop


class IntCast(nn.Module):
    """f32 → 整数の切り捨て規約と i32 の sub / mul、および i32 出力の readback。

    `x.to(torch.int64)` は torch の truncate（0 方向切り捨て）。IR では i32 として
    宣言される（境界正規化 — ADR 0009）。
    """

    def forward(self, x: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        truncated = x.to(torch.int64)
        scaled = (truncated - k) * k
        return scaled, scaled.to(torch.float32)


class LayoutChain(nn.Module):
    """permute → clone → view → unsqueeze → squeeze の鎖（ADR 0011 / recon §2）。

    実測の attention head 整形（permute [0,2,1,3] の直後に contiguous 化 clone → view）と
    同じ並び。`reshape(3, -1)` が非連続な permute 結果に掛かるので torch は clone を挟む —
    その clone は IR では恒等（値は常に連続）になる。

    MUST: 並べ替えは**巡回長 3**（[1,2,0]）を使う。実測に出る形（[0,2,1,3] / [0,2,1]）は
    全て対合で逆置換が自分自身になるため、stride 表を逆置換で組む誤りを torch 突合でも
    検出できない（対合形そのものは packages/runtime/tests/gpu_ops_test.ts の op 単位ケースが持つ）。

    2 本目の出力は permute 出力の**別名の連鎖**そのもの（reshape 3 段）で、
    「グラフ出力がエイリアスでも readback が正しい」を golden 側から踏む。
    """

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        heads = x.permute(1, 2, 0)  # [T,3,4] -> [3,4,T]
        merged = heads.reshape(3, -1)  # -> [3,4T]（clone + view）
        wide = merged.unsqueeze(0)  # -> [1,3,4T]
        return heads, torch.squeeze(wide, dim=[0])


class ExpandMask(nn.Module):
    """bool マスクと i32 添字の expand（recon §2 の実測 5 本と同型）。

    conv 経路の bool マスク `[1,T,1] → [1,T,C]` と、gather 添字の `[1,T,T] → [16,T,T]` を
    縮めた形。expand の出力を i32 のまま 1 本読み戻して、非 f32 の strided コピーを
    readback まで通す。
    """

    def forward(
        self, mask: torch.Tensor, index: torch.Tensor, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keep = mask.to(torch.bool)  # bool[T,1]
        wide = keep.expand(-1, 3)  # bool[T,3]
        spread = index.expand(3, -1)  # i64[3,T]（IR では i32）
        return scores * spread.to(torch.float32), torch.bitwise_not(wide), spread


class BatchMatmul(nn.Module):
    """rank-3 のバッチ matmul（recon §2 の実測 8 本と同型）。

    MUST: **B / M / K / N を全て違う長さ**にする（ACTIVE_DESIGN の Pitfalls）。バッチ stride を
    隣の次元の積で組む誤り（M·K と K·N と M·N の取り違え）は、2 軸が同じ長さの形では
    数値に出ない。ここは B=2 / M=T=6 / K=3 / N=5 で、行列 1 枚の要素数も 18 / 15 / 30 と
    全て違う。

    2 本目の出力は同じ被乗数を転置した相手と掛けたもので、K と N を入れ替えた読み方
    （b の stride を n ではなく k で組む誤り）を別の形からも踏む。
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.bmm(x, y), torch.bmm(y.permute(0, 2, 1), x.permute(0, 2, 1))


class GatherLastDim(nn.Module):
    """最終次元の gather（実測は dim=-1 / src f32[16,T,512] / index i32[16,T,T]）。

    index は **i32 のグラフ入力**として渡す（torch 既定の i64 が境界で i32 に正規化される —
    ADR 0009）。整数 initializer は IR v1 の語彙に無いので、定数として畳まない形にすること
    自体が契約の要求。

    2 本目の出力は引いた値を使う下流演算で、gather の結果が値として正しく流れることを
    golden に載せる。
    """

    def forward(self, src: torch.Tensor, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        picked = torch.gather(src, -1, index)
        return picked, picked.sum(dim=-1)


def _fill_param(generator: torch.Generator, tensor: torch.Tensor, scale: float) -> None:
    """パラメータを固定 seed の生成器で上書きする。

    MUST: `nn.Linear` / `nn.Conv1d` / `nn.Embedding` の既定初期化は**グローバル RNG**を使う。
    上書きしないと golden がバイト単位で再現せず、再生成突合（TestDeterminism）が落ちる。
    """
    with torch.no_grad():
        tensor.copy_(torch.randn(tensor.shape, generator=generator) * scale)


class AttentionBlock(nn.Module):
    """linear → bmm → softmax → bmm → linear → 残差 + layer_norm（recon §2 の並びの縮小形）。

    実測の attention と同じ順序で、head 分割の `view → permute[0,2,1,3] → reshape` まで含む。
    融合 op の linear（bias 常時あり）/ softmax（dim=-1）/ layer_norm（[H] + affine）を
    1 本のグラフで踏む。

    2 本目の出力は **大きい負値の softmax**（素朴形なら f32 で `exp` が 0 に潰れて 0/0 = NaN
    になる領域）。safe-softmax の amax 減算が外れた瞬間に golden 突合が赤くなる形を、golden
    側に意図的に仕込んである（ADR 0012 の反例集）。
    """

    HIDDEN = 8
    HEADS = 2
    HEAD_DIM = 4

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        self.query = nn.Linear(self.HIDDEN, self.HIDDEN)
        self.key = nn.Linear(self.HIDDEN, self.HIDDEN)
        self.value = nn.Linear(self.HIDDEN, self.HIDDEN)
        self.dense = nn.Linear(self.HIDDEN, self.HIDDEN)
        # 実測の layer_norm_eps（1e-07）をそのまま使う
        self.norm = nn.LayerNorm(self.HIDDEN, eps=1e-07)
        for projection in (self.query, self.key, self.value, self.dense):
            _fill_param(generator, projection.weight, 0.3)
            _fill_param(generator, projection.bias, 0.1)
        _fill_param(generator, self.norm.weight, 0.2)
        _fill_param(generator, self.norm.bias, 0.1)

    def _split_heads(self, x: torch.Tensor, length: Any) -> torch.Tensor:
        return (
            x.view(1, length, self.HEADS, self.HEAD_DIM)
            .permute(0, 2, 1, 3)
            .reshape(self.HEADS, length, self.HEAD_DIM)
        )

    def forward(self, x: torch.Tensor, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = x.shape[1]
        query = self._split_heads(self.query(x), length)
        key = self._split_heads(self.key(x), length)
        value = self._split_heads(self.value(x), length)
        scores = torch.bmm(query, key.permute(0, 2, 1))
        context = torch.bmm(torch.softmax(scores, dim=-1), value)
        merged = (
            context.view(1, self.HEADS, length, self.HEAD_DIM)
            .permute(0, 2, 1, 3)
            .reshape(1, length, self.HIDDEN)
        )
        return self.norm(self.dense(merged) + x), torch.softmax(logits, dim=-1)


class FusedAttention(nn.Module):
    """SDPA を保存して融合 `attention` 1 ノードにする golden（ADR 0023）。

    MUST: **B / H / M / N / D を全て違う長さ**にする。カーネルは B と H を 1 本のバッチ軸へ
    畳むので、B=1（実測形）や B==H では取り違えが値に出ない（設計 recon §4.6 の検出限界 ①）。
    NOTE: 形は**静的**。DiT / VAE の attention は解像度固定で記号次元を持たないので、実測に
    無い形（記号 M / N）を golden で先取りしない。
    """

    BATCH = 2
    HEADS = 3
    QUERIES = 5
    KEYS = 7
    DEPTH = 4

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)


def _fused_attention_inputs(generator: torch.Generator) -> tuple[torch.Tensor, ...]:
    """`FusedAttention` の入力。**最後の query 行だけ logit を一様に −195 へ落とす**。

    行統計（②）の amax 減算が外れると `exp(−195)` が f32 で 0 に潰れ、分母 0 → 0/0 = NaN に
    なる領域を golden から必ず踏ませる（`masked_scores` / `attention_block` が softmax 単体で
    やっているのと同じ仕掛けを、融合経路の内側に置く）。

    MUST: その行の logit は **n によらず同一のビット列**にする（`q = [−c,0,0,0]` × `k[…,0]` が
    n に依らない定数）。同一なら `S − amax` は厳密に 0 になり、重みは厳密に `1/N` へ落ちる。
    **不揃いにすると golden が tolerance を超える** — |S| ≈ 190 の内積は f32 で ≈1e-5 の
    絶対誤差を持ち、それが `exp(S − amax)` でそのまま**相対** 1e-5 の重み誤差になる
    （実測: 行内の logit が 12.7 ばらつく形で maxAbs 4.7e-6 / maxRel 9.6e-4 — GOLDEN_TOLERANCE
    の atol 1e-6 を超える）。これは融合の誤差ではなく**入力の条件数**で、分解経路でも同じ値に
    なる（融合と分解のビット同一は packages/runtime/tests/gpu_attention_parity_test.ts が
    別途固定している）。
    """
    shape = FusedAttention
    query = _uniform(
        generator, shape.BATCH, shape.HEADS, shape.QUERIES, shape.DEPTH, low=-1.0, high=1.0
    )
    key = _uniform(generator, shape.BATCH, shape.HEADS, shape.KEYS, shape.DEPTH, low=-1.0, high=1.0)
    value = _uniform(
        generator, shape.BATCH, shape.HEADS, shape.KEYS, shape.DEPTH, low=-1.0, high=1.0
    )
    # 第 0 チャネルだけ n に依らない定数にする。半スケール √(1/√D) の 2 乗 = 1/√D = 0.5 が
    # 内積に掛かるので、q = [−60,0,0,0] の行は S = 0.5 · (−60 · 6.5) = −195（全 n で同値）。
    key[:, :, :, 0] = 6.5
    query[:, :, -1, :] = 0.0
    query[:, :, -1, 0] = -60.0
    return (query, key, value)


class EmbeddingLookup(nn.Module):
    """embedding（行 gather）— weight f32[V,H] × 添字 i32[…]。

    MUST: `padding_idx` の行を**非ゼロで上書きしてから** export する。`nn.Embedding` は
    padding_idx の行を 0 で初期化するので、素のままだと「padding 行を 0 で潰す」誤実装と
    torch の出力が一致してしまい、恒真な golden になる（ADR 0012 の「受理して不活性」を
    実際に検証できない）。
    """

    VOCAB = 5
    HIDDEN = 3

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        self.table = nn.Embedding(self.VOCAB, self.HIDDEN, padding_idx=0)
        _fill_param(generator, self.table.weight, 1.0)

    def forward(self, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = self.table(index)
        return rows, rows.sum(dim=-1)


class MaskedScores(nn.Module):
    """masked_fill の実測 2 形（recon §2）— attention の -3.4e38 埋めと conv 経路の 0 埋め。

    - `scores f32[1,2,T,T]` に `mask bool[1,1,T,T]` を**右詰め broadcast**で当てる。
    - 埋め値は f32 の最小有限値（JSON 往復で ulp が動けばここが赤くなる）。
    - 下流の softmax は、全要素がマスクされた行で一様分布になる（limitations.md の
      「amax/amin の identity は ±F32_MAX」との相互作用を padded ケースで踏む）。
    """

    NEG_INF = -3.4028234663852886e38

    def forward(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        hidden: torch.Tensor,
        gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        drop = torch.bitwise_not(mask.to(torch.bool))
        weights = torch.softmax(scores.masked_fill(drop, self.NEG_INF), dim=-1)
        return weights, hidden.masked_fill(gate.to(torch.bool), 0.0)


class ConvBlock(nn.Module):
    """conv1d の実測形（kernel 3 / stride 1 / padding 1 / groups 1 / dilation 1 / bias あり）。

    実測（recon §2）と同じく `[B,T,C] → permute → conv1d → permute` の往復を含める。
    padding を 0 扱いにする誤りは、出力長ではなく**両端の値**に出る（出力長は attrs から
    先に決まるため）。
    """

    CHANNELS = 4

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        self.conv = nn.Conv1d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1)
        _fill_param(generator, self.conv.weight, 0.3)
        _fill_param(generator, self.conv.bias, 0.1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        moved = self.conv(x.permute(0, 2, 1))
        return moved, moved.permute(0, 2, 1)


class SymbolicTable(nn.Module):
    """記号長テーブル（ADR 0010）— T 依存の部分木を Tmax で焼き、sym_prefix_slice で運ぶ。

    - 相対位置 `arange(T)` の外積 → clamp → **gather 添字（i32 initializer）**。2 軸とも
      記号なので slices が 2 本になる。
    - `arange(T)` の f32 スケール（f32 の sym_prefix_slice、slices 1 本）。

    焼いた定数は Tmax（= `_DYNAMIC_T` の上限）の全長で、golden の実長 T はそれより短い。
    したがって **読み出し stride を束縛後の shape から組む誤りはここで値に出る**
    （T = Tmax でしか一致しないため、実長の短い golden が検出器になる）。
    """

    BUCKETS = 5

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = x.shape[0]
        positions = torch.arange(length)
        buckets = torch.clamp(
            positions.unsqueeze(1) - positions.unsqueeze(0) + 2, 0, self.BUCKETS - 1
        )
        scale = torch.arange(length).to(torch.float32) * 0.5
        return torch.gather(x, -1, buckets), x + scale.unsqueeze(-1)


class ScalarOperands(nn.Module):
    """スカラ被演算子の定数昇格（recon §4-4）— f32 の 4 形と、逆順スカラの i32 形。

    `1 - mask` は非可換な逆順形（実測の conv 経路と同じ）。定数を右に置く誤りは
    `mask - 1` になって符号が反転し、下流の重み掛けで**値が変わる**。

    MUST: `1 - mask` の結果を**値として下流に流す**（重みとして x に掛ける）。mask ∈ {0,1}
    では `(1 - mask) * mask` が逆順でも恒等 0 で、誤りが値に出ない恒真な golden になる。
    """

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scaled = (x + 2.5 - 0.5) * 3.0 / 2.0
        return scaled, (1 - mask).to(torch.float32) * x


class SplinePieces(nn.Module):
    """sdp の rational-quadratic spline の骨格（recon §2）を縮めたもの。

    区間内判定 `(x >= -b) & (x <= b)`（ge_scalar / le_scalar / bitwise_and）→ 区間境界の
    `cumsum` → searchsorted_free の `sum(x[…,None] >= bl, dim=-1)`（ge の Tensor 形と
    **bool の行 sum → i32**）→ 区間外復帰の `where` と softplus 分解形の `log1p` まで、
    波3 で足した数理 op をひと続きの経路で踏む。

    MUST: 入力は区間境界（±TAIL）と clamp の両端を跨ぐ値にする。全要素が区間内（または外）に
    収まると where の分岐が片側しか踏まれず、取り違えが値に出ない。
    """

    TAIL = 1.0
    CLAMP_MIN = -0.75
    CLAMP_MAX = 0.5

    def forward(self, inputs: torch.Tensor, widths: torch.Tensor) -> tuple[torch.Tensor, ...]:
        inside = (inputs >= -self.TAIL) & (inputs <= self.TAIL)
        cumwidths = torch.cumsum(widths, dim=-1)
        # searchsorted_free: 各要素が何本の境界以上かを数える（bool の行 sum = i32 のカウント）
        bins = torch.sum(inputs.unsqueeze(-1) >= cumwidths, dim=-1)
        clamped = torch.clamp(inputs, self.CLAMP_MIN, self.CLAMP_MAX)
        softplus = torch.log1p(torch.exp(clamped))
        return (
            torch.where(inside, softplus, inputs),
            bins,
            inside,
            clamped > 0.25,
            cumwidths,
        )


class DecoderTail(nn.Module):
    """dec の最終段（recon §2）— leaky_relu の **slope 2 種**と f32 の expand。

    ups / ResBlock は `LRELU_SLOPE=0.1`、最終段は torch 既定の 0.01（位置引数ごと省略）。
    attrs に slope を持たせないと片方が黙って誤る形（ADR 0015）を golden で固定する。
    `gain` の expand は相対位置埋め込みの 4D 化（enc_p / flow）と同型の **f32 expand**。
    """

    CHANNELS = 4
    OUT_CHANNELS = 3
    SLOPE = 0.1

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        self.conv = nn.Conv1d(self.CHANNELS, self.OUT_CHANNELS, kernel_size=3, padding=1)
        _fill_param(generator, self.conv.weight, 0.3)
        _fill_param(generator, self.conv.bias, 0.1)

    def forward(self, x: torch.Tensor, gain: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        wide = gain.expand(-1, -1, x.shape[2])
        hidden = nn.functional.leaky_relu(x * wide, self.SLOPE)
        # 位置引数を省略した形（torch 既定 0.01）— エクスポータが境界で attrs に明示化する
        return hidden, torch.tanh(nn.functional.leaky_relu(self.conv(hidden)))


class CouplingSplit(nn.Module):
    """coupling reverse の split → 変換 → cat → flip 経路（recon §2）を縮めたもの。

    ① `torch.split(x, [half]*2, 1)` — normalize の split→slice パスで slice 2 本に開く
    ② 片側だけを変換して `torch.cat([x0, x1], 1)` で戻す（**入力の順序が値に出る**形）
    ③ 最後に `Flip`（チャネル軸の反転）

    MUST: cat の 2 本を**別の値域**にする（片側が変換済み・片側が素）。同じ値域だと
    書き出し位置の取り違えが値に出ない。
    MUST: チャネル数は 6（= 3 + 3）にして flip の軸長を 3 以上にする。2ch の反転は
    off-by-one が対称に消えるので検出器にならない（sdp の実測 2ch は
    packages/runtime/tests/gpu_ops_test.ts の op 単位ケースが別途持つ）。

    2 本目の出力は**最終次元の pad + slice**（相対位置注意の value 側 `F.pad(p_attn, [w,w])`
    と、spline の `bin_locations[..., :-1]` の縮小形）。pad の埋め値が 0 でない誤りと、
    slice の start ずれがそのまま値に出る。
    """

    WINDOW = 2

    def forward(self, x: torch.Tensor, attn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        first, second = torch.split(x, [3, 3], 1)
        transformed = torch.tanh(first) * 2.0
        joined = torch.cat([transformed, second], 1)
        flipped = torch.flip(joined, [1])
        padded = nn.functional.pad(attn, [self.WINDOW, self.WINDOW])
        return flipped, padded[..., 1:] * 3.0


class DilatedConvStack(nn.Module):
    """conv1d の groups / dilation（ADR 0015）— sdp の DDSConv と dec の ResBlock1 の縮小形。

    ① **depthwise dilated**（`groups = C`・k 5・dilation 1/3/9・padding = d·(k−1)/2）を 3 段。
       実測は sdp の DDSConv（`dilation = kernel_size**i`）そのもの。groups を無視して
       通常畳み込みで実行する誤りは、**別チャネルの値が混ざる**形で必ず値に出る。
    ② **pointwise 1x1**（Cin 6 → Cout 4 の非対称）で DDSConv の `conv_1x1` を踏む。
       k=1 は `d·(K−1) = 0` なので dilation 一般形が従来式と一致する境界でもある。
    ③ **中間の groups**（1 < g < C: Cin 4 / Cout 6 / g 2）— 重みの第 2 軸が Cin/groups = 2 に
       なる形。グループ帯の先頭オフセットを落とす誤りは depthwise では消えるので、
       この段が検出器になる。
    ④ **ResBlock1 の残差**（k 3・dilation 3 と 5・padding = d）— `x + conv(leaky_relu(x))`。

    MUST: 各段のチャネル数を揃えない。B/Cin/Cout/K が同じ長さだと添字の取り違えが値に出ない。
    """

    DEPTH_CHANNELS = 6
    DEPTH_KERNEL = 5
    DEPTH_DILATIONS = (1, 3, 9)
    MID_CHANNELS = 4
    GROUPED_CHANNELS = 6
    GROUPS = 2
    RES_KERNEL = 3
    RES_DILATIONS = (3, 5)
    SLOPE = 0.1

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        self.depthwise = nn.ModuleList(
            nn.Conv1d(
                self.DEPTH_CHANNELS,
                self.DEPTH_CHANNELS,
                kernel_size=self.DEPTH_KERNEL,
                padding=dilation * (self.DEPTH_KERNEL - 1) // 2,
                dilation=dilation,
                groups=self.DEPTH_CHANNELS,
            )
            for dilation in self.DEPTH_DILATIONS
        )
        self.pointwise = nn.Conv1d(self.DEPTH_CHANNELS, self.MID_CHANNELS, kernel_size=1)
        self.grouped = nn.Conv1d(
            self.MID_CHANNELS,
            self.GROUPED_CHANNELS,
            kernel_size=3,
            padding=1,
            groups=self.GROUPS,
        )
        self.res = nn.ModuleList(
            nn.Conv1d(
                self.GROUPED_CHANNELS,
                self.GROUPED_CHANNELS,
                kernel_size=self.RES_KERNEL,
                padding=dilation,
                dilation=dilation,
            )
            for dilation in self.RES_DILATIONS
        )
        for conv in [*self.depthwise, self.pointwise, self.grouped, *self.res]:
            _fill_param(generator, conv.weight, 0.4)
            _fill_param(generator, conv.bias, 0.1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = x
        for conv in self.depthwise:
            hidden = nn.functional.leaky_relu(conv(hidden), self.SLOPE)
        depthwise_out = hidden
        hidden = self.grouped(self.pointwise(hidden))
        for conv in self.res:
            hidden = hidden + conv(nn.functional.leaky_relu(hidden, self.SLOPE))
        return depthwise_out, hidden


class ConvTransposeStack(nn.Module):
    """conv_transpose1d（ADR 0015）— dec の ups と bias 無し conv_post の縮小形。

    - ups の実測は up_rates [8,8,2,2,2] / up_k [16,16,8,2,2]。ここは **up 2（k 2 / pad 0）**と
      **up 8（k 16 / pad 4）**の 2 種を 1 本ずつ持つ（`2·pad == k − u` が両方で成立する形）。
    - **チャネル数は全段で非対称**（5 → 3 → 2 → 1）。重み `[Cin, Cout, K]` を
      `[Cout, Cin, K]` と読む取り違えは、Cin == Cout の形では要素数が合ってしまい
      shape 検査も golden も素通りする（recon §4 / ADR 0015）。
    - 最終段は **bias 無しの conv1d**（`Conv1d(2, 1, 7, padding=3, bias=False)` — 実測の
      conv_post と同型）。エクスポータのゼロ bias 合成がここを貫通する。
    """

    IN_CHANNELS = 5
    MID_CHANNELS = 3
    UP_CHANNELS = 2
    POST_KERNEL = 7
    SLOPE = 0.1

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        # (k, stride, padding) は 2·padding == k − stride を満たす実測の 2 形
        self.up_small = nn.ConvTranspose1d(
            self.IN_CHANNELS, self.MID_CHANNELS, kernel_size=2, stride=2, padding=0
        )
        self.up_large = nn.ConvTranspose1d(
            self.MID_CHANNELS, self.UP_CHANNELS, kernel_size=16, stride=8, padding=4
        )
        self.post = nn.Conv1d(
            self.UP_CHANNELS, 1, kernel_size=self.POST_KERNEL, padding=3, bias=False
        )
        for conv in (self.up_small, self.up_large):
            _fill_param(generator, conv.weight, 0.4)
            _fill_param(generator, conv.bias, 0.1)
        _fill_param(generator, self.post.weight, 0.4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = nn.functional.leaky_relu(self.up_small(x), self.SLOPE)
        hidden = nn.functional.leaky_relu(self.up_large(hidden), self.SLOPE)
        return hidden, torch.tanh(self.post(hidden))


class RmsNormBlock(nn.Module):
    """RMSNorm の **供給ルート 2 系統**（ADR 0017）と、省略スロットの合成（ADR 0016）。

    - **手書き分解形** `x · rsqrt(mean(x²)+eps) · weight` — `normalize._fold_rms_norm` が
      1 ノードへ畳む経路（Qwen3 / DiT がこの形）。畳めないと mean / rsqrt が未対応 op で落ちる。
    - **`nn.RMSNorm`** — `aten.rms_norm` を PRESERVED で残す経路（diffusers 由来）。
      affine あり / 無しの両方を置き、無し側で **ones 合成**（アリティ 2 への正規化）を踏む。
    - **bias 無し linear** と **affine 無し layer_norm** — ゼロ bias / ones・zeros 合成。

    MUST: eps を 3 本とも**別の値**にする。attrs に載せず既定へ落とす誤りは、全部同じ eps の
    形では値に出ない（layer_norm の実測 1e-7 と rms の 1e-6 が同居する Anima の形と同型）。
    MUST: weight は 1 から離した値にする（`ones` のままだと ones 合成の取り違えが値に出ない）。
    """

    HIDDEN = 8
    FOLD_EPS = 1e-6
    RMS_EPS = 1e-5
    NORM_EPS = 1e-7

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        self.weight = nn.Parameter(_uniform(generator, self.HIDDEN, low=0.4, high=1.6))
        self.affine = nn.RMSNorm(self.HIDDEN, eps=self.RMS_EPS)
        self.plain = nn.RMSNorm(self.HIDDEN, eps=self.RMS_EPS, elementwise_affine=False)
        self.dense = nn.Linear(self.HIDDEN, self.HIDDEN, bias=False)
        self.layer_norm = nn.LayerNorm(self.HIDDEN, eps=self.NORM_EPS, elementwise_affine=False)
        _fill_param(generator, self.affine.weight, 0.5)
        _fill_param(generator, self.dense.weight, 0.3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        squared = x.pow(2).mean(-1, keepdim=True)
        folded = x * torch.rsqrt(squared + self.FOLD_EPS) * self.weight
        projected = self.dense(folded)
        return self.affine(projected), self.plain(projected), self.layer_norm(projected)


class Conv2dBlock(nn.Module):
    """conv2d（ADR 0017）と、チャネル L2 正規化の `clamp_min`。

    MUST（ADR 0017 のテスト要件 — 重みレイアウト取り違えは要素数が合って shape 検査を素通り
    する）: **Kh≠Kw / stride・padding の H/W 非対称 / Cin・Cout とも 2 以上で互いに異なる**を
    全て踏む。加えて groups（1 < g < C）と dilation の非対称形、bias 無し（ゼロ bias 合成）も
    1 本ずつ持つ。入力も H≠W にして高さと幅の取り違えを値に出す。

    MUST: L2 の入力に**ノルムがちょうど 0 になる位置**を作る（`example_inputs` が 1 点を
    ゼロにする）。`clamp_min` の床が外れると 0/0 = NaN になり、golden 突合がそこで赤くなる —
    床が効いていることを恒真でない形で固定できる唯一の点。

    NOTE: チャネル L2 は `patch_anima._l2_normalize_channels` の鏡像で、**permute 無しの
    軸 sum**（attrs `dim=1`）。実 GPU golden で**軸 reduce 変種**を torch 参照つきで踏む
    唯一の経路でもある（他の golden の `sum` は全て最終次元 = 行カーネル）。
    """

    EPS = 1e-12

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        # Kh≠Kw / stride (1,2) / padding (1,0) / Cin 4 → Cout 6
        self.wide = nn.Conv2d(4, 6, kernel_size=(3, 1), stride=(1, 2), padding=(1, 0))
        # groups 3（Cin/groups = 2）・bias 無し → ゼロ bias 合成
        self.grouped = nn.Conv2d(6, 3, kernel_size=(1, 3), padding=(0, 1), groups=3, bias=False)
        # dilation の H/W 非対称（(2,1)）と padding (2,1)
        self.dilated = nn.Conv2d(3, 2, kernel_size=(2, 2), padding=(2, 1), dilation=(2, 1))
        for conv in (self.wide, self.dilated):
            _fill_param(generator, conv.weight, 0.4)
            _fill_param(generator, conv.bias, 0.1)
        _fill_param(generator, self.grouped.weight, 0.4)

    def forward(self, x: torch.Tensor, channels: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hidden = self.dilated(self.grouped(self.wide(x)))
        norm = torch.sqrt(torch.sum(channels * channels, dim=1)).clamp(min=self.EPS).unsqueeze(1)
        return hidden, channels / norm


class Int8Weights(nn.Module):
    """i8 格納（per-channel scale）で **`WEIGHT_SLOTS` の全 5 op** を踏む golden（ADR 0019）。

    MUST: 重みの**行長も総要素数も 4 の倍数にしない**。i8 は 4 要素を 1 u32 へ詰めるので、
    「語とレーンを行内相対添字から割り出す」誤りは**行長が 4 の倍数のときだけ偶然一致する**
    （f16 の偶奇と同型の罠 — ADR 0019）。4 の倍数だけで固めると、この誤りが golden を素通り
    する。実測形: embedding `[7,5]` / linear `[3,5]` / conv1d `[5,3,3]` /
    conv_transpose1d `[5,2,3]` / conv2d `[3,2,3,1]` — 全て要素数も出力チャネルあたりの
    ブロック長も 4 の倍数でない。

    MUST: embedding の 1 行を**全ゼロ**にする（`amax == 0` のチャネル）。scale の下限 clamp が
    外れると `0/0 = NaN` になり、この 1 行だけが golden を赤くする — 下限が効いていることを
    恒真でない形で固定できる唯一の点。

    MUST: conv_transpose1d を混ぜる（重みが `[Cin,Cout,K]` の転置レイアウトで per-channel 軸が
    **1**）。軸表の取り違えは他 4 op では値に出ない。
    """

    VOCAB = 7
    EMBED = 5
    HIDDEN = 3
    SIGNAL_IN = 3
    SIGNAL_MID = 5
    SIGNAL_OUT = 2
    UP_STRIDE = 3
    IMAGE_IN = 2
    IMAGE_OUT = 3

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        self.table = nn.Embedding(self.VOCAB, self.EMBED)
        self.dense = nn.Linear(self.EMBED, self.HIDDEN)
        self.conv = nn.Conv1d(self.SIGNAL_IN, self.SIGNAL_MID, kernel_size=3, padding=1)
        # 2·padding == k − stride（ランタイムが受理する出力長 L·stride の形 — ADR 0015）
        self.up = nn.ConvTranspose1d(
            self.SIGNAL_MID, self.SIGNAL_OUT, kernel_size=self.UP_STRIDE, stride=self.UP_STRIDE
        )
        self.image = nn.Conv2d(self.IMAGE_IN, self.IMAGE_OUT, kernel_size=(3, 1), padding=(1, 0))
        for module in (self.dense, self.conv, self.up, self.image):
            _fill_param(generator, module.weight, 0.5)
            _fill_param(generator, module.bias, 0.2)
        _fill_param(generator, self.table.weight, 1.0)
        with torch.no_grad():
            self.table.weight[0].zero_()

    def forward(
        self, index: torch.Tensor, signal: torch.Tensor, image: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        return (
            self.dense(self.table(index)),
            self.up(torch.tanh(self.conv(signal))),
            self.image(image),
        )


def _zeroed_column(generator: torch.Generator, *shape: int, at: tuple[int, int]) -> torch.Tensor:
    """1 つの空間位置だけ全チャネル 0 の `[1,C,H,W]`（`clamp_min` の床を踏ませる）。"""
    tensor = _uniform(generator, *shape, low=-1.5, high=1.5)
    tensor[:, :, at[0], at[1]] = 0.0
    return tensor


@dataclass(frozen=True)
class GoldenSpec:
    name: str
    build: Callable[[torch.Generator], nn.Module]
    example_inputs: Callable[[torch.Generator], tuple[torch.Tensor, ...]]
    dynamic_shapes: Any = None
    symbol_names: Sequence[str] = field(default=("T",))
    #: 重みの格納 dtype。`"i8"` のとき `generate_golden` が **export と参照採取の前に**
    #: fake-quant を掛ける（ADR 0006 / 0019）。
    weight_dtype: str = "f32"
    #: 分解を止める高位 op の接頭辞集合（ADR 0023）。SDPA を保存して融合 `attention` を
    #: 出す golden だけが既定を差し替える — 既定へ足すと mask 付き SDPA まで保存対象になる。
    preserved: Sequence[str] = field(default=PRESERVED_OP_PREFIXES)


GOLDEN_SPECS: tuple[GoldenSpec, ...] = (
    GoldenSpec(
        name="unary_chain",
        build=lambda _: UnaryChain(),
        # sqrt / log の定義域を外さないよう、0 から離した正値だけを与える。
        example_inputs=lambda g: (_uniform(g, 3, 4, low=0.5, high=2.0),),
    ),
    GoldenSpec(
        name="activations",
        build=lambda _: Activations(),
        example_inputs=lambda g: (_uniform(g, 2, 5, low=-2.0, high=2.0),),
    ),
    GoldenSpec(
        name="broadcast_binary",
        build=BroadcastBinary,
        # 除数 y は 0 を跨がせない（div の golden が発散しないように）。
        example_inputs=lambda g: (
            _uniform(g, GOLDEN_T, 4, low=-1.0, high=1.0),
            _uniform(g, 4, low=0.5, high=1.5),
        ),
        dynamic_shapes=({0: _DYNAMIC_T}, None),
    ),
    GoldenSpec(
        name="mlp",
        build=Mlp,
        example_inputs=lambda g: (_uniform(g, GOLDEN_T, 8, low=-1.0, high=1.0),),
        dynamic_shapes=({0: _DYNAMIC_T},),
    ),
    GoldenSpec(
        name="row_reduce",
        build=lambda _: RowReduce(),
        example_inputs=lambda g: (_uniform(g, GOLDEN_T, 5, low=-2.0, high=2.0),),
        dynamic_shapes=({0: _DYNAMIC_T},),
    ),
    GoldenSpec(
        name="mask_chain",
        build=lambda _: MaskChain(),
        # mask は torch 既定の i64（境界で i32 へ正規化される）。0 を必ず含めて
        # bitwise_not / 真偽化の両方の分岐を踏む。
        example_inputs=lambda g: (
            _mask(GOLDEN_T, 1, zero_at=2),
            _mask(1, GOLDEN_T, zero_at=5),
            _uniform(g, GOLDEN_T, GOLDEN_T, low=-2.0, high=2.0),
        ),
        dynamic_shapes=({0: _DYNAMIC_T}, {1: _DYNAMIC_T}, {0: _DYNAMIC_T, 1: _DYNAMIC_T}),
    ),
    GoldenSpec(
        name="int_cast",
        build=lambda _: IntCast(),
        # 切り捨てが round / floor と区別できる値（±.5 / ±.7 を跨ぐ）を与える。
        example_inputs=lambda g: (
            _uniform(g, GOLDEN_T, 3, low=-3.0, high=3.0),
            torch.arange(GOLDEN_T * 3, dtype=torch.int64).reshape(GOLDEN_T, 3) % 4 - 1,
        ),
        dynamic_shapes=({0: _DYNAMIC_T}, {0: _DYNAMIC_T}),
    ),
    GoldenSpec(
        name="layout_chain",
        build=lambda _: LayoutChain(),
        example_inputs=lambda g: (_uniform(g, GOLDEN_T, 3, 4, low=-1.0, high=1.0),),
        dynamic_shapes=({0: _DYNAMIC_T},),
    ),
    GoldenSpec(
        name="expand_mask",
        # mask / index は torch 既定の i64（境界で i32 へ正規化される）。mask は 0 を必ず
        # 含めて bitwise_not の両分岐を踏む。
        example_inputs=lambda g: (
            _mask(GOLDEN_T, 1, zero_at=2),
            torch.arange(GOLDEN_T, dtype=torch.int64).reshape(1, GOLDEN_T) - 2,
            _uniform(g, 3, GOLDEN_T, low=-2.0, high=2.0),
        ),
        build=lambda _: ExpandMask(),
        dynamic_shapes=({0: _DYNAMIC_T}, {1: _DYNAMIC_T}, {1: _DYNAMIC_T}),
    ),
    GoldenSpec(
        name="batch_matmul",
        build=lambda _: BatchMatmul(),
        # B=2 / M=T / K=3 / N=5 — 全て違う長さ（軸の取り違えが数値に出る形）
        example_inputs=lambda g: (
            _uniform(g, 2, GOLDEN_T, 3, low=-1.0, high=1.0),
            _uniform(g, 2, 3, 5, low=-1.0, high=1.0),
        ),
        dynamic_shapes=({1: _DYNAMIC_T}, None),
    ),
    GoldenSpec(
        name="gather_last_dim",
        build=lambda _: GatherLastDim(),
        # index は torch 既定の i64（境界で i32 へ正規化される）。3i mod 4 で src の
        # 最終次元 4 を全部踏み、恒等でも単調でもない列にする（添字 1 ずれが必ず出る）。
        example_inputs=lambda g: (
            _uniform(g, 2, GOLDEN_T, 4, low=-2.0, high=2.0),
            (torch.arange(2 * GOLDEN_T * 3, dtype=torch.int64) * 3 % 4).reshape(2, GOLDEN_T, 3),
        ),
        dynamic_shapes=({1: _DYNAMIC_T}, {1: _DYNAMIC_T}),
    ),
    GoldenSpec(
        name="attention_block",
        build=AttentionBlock,
        # 2 本目の logits は素朴 softmax なら f32 で exp が 0 に潰れる領域（-205..-180）
        example_inputs=lambda g: (
            _uniform(g, 1, GOLDEN_T, AttentionBlock.HIDDEN, low=-1.0, high=1.0),
            _uniform(g, 3, 5, low=-205.0, high=-180.0),
        ),
        dynamic_shapes=({1: _DYNAMIC_T}, None),
    ),
    GoldenSpec(
        name="fused_attention",
        build=lambda _: FusedAttention(),
        example_inputs=_fused_attention_inputs,
        # SDPA を保存するのはこの golden だけ（既定の 11 op に足すと mask 付き SDPA を持つ
        # グラフが export 不能になる — ADR 0023）。
        preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    ),
    GoldenSpec(
        name="embedding_lookup",
        build=EmbeddingLookup,
        # 添字は torch 既定の i64（境界で i32 へ正規化される）。padding_idx=0 を必ず引く列で、
        # 語彙 5 行を一巡する。
        example_inputs=lambda _: (
            (torch.arange(GOLDEN_T, dtype=torch.int64) % EmbeddingLookup.VOCAB).reshape(
                1, GOLDEN_T
            ),
        ),
        dynamic_shapes=({1: _DYNAMIC_T},),
    ),
    GoldenSpec(
        name="masked_scores",
        build=lambda _: MaskedScores(),
        # mask は torch 既定の i64（境界で bool へ cast される）。**1 行を全マスク**にして、
        # 全要素が -3.4e38 になった行の softmax（一様分布）まで踏む。
        example_inputs=lambda g: (
            _uniform(g, 1, 2, GOLDEN_T, GOLDEN_T, low=-2.0, high=2.0),
            _band_mask(GOLDEN_T, blocked_row=2),
            _uniform(g, 1, GOLDEN_T, 4, low=-1.0, high=1.0),
            _mask(1, GOLDEN_T, 4, zero_at=5),
        ),
        dynamic_shapes=(
            {2: _DYNAMIC_T, 3: _DYNAMIC_T},
            {2: _DYNAMIC_T, 3: _DYNAMIC_T},
            {1: _DYNAMIC_T},
            {1: _DYNAMIC_T},
        ),
    ),
    GoldenSpec(
        name="conv_block",
        build=ConvBlock,
        example_inputs=lambda g: (
            _uniform(g, 1, GOLDEN_T, ConvBlock.CHANNELS, low=-1.0, high=1.0),
        ),
        dynamic_shapes=({1: _DYNAMIC_T},),
    ),
    GoldenSpec(
        name="symbolic_table",
        build=lambda _: SymbolicTable(),
        example_inputs=lambda g: (
            _uniform(g, GOLDEN_T, SymbolicTable.BUCKETS, low=-1.0, high=1.0),
        ),
        dynamic_shapes=({0: _SMALL_DYNAMIC_T},),
    ),
    GoldenSpec(
        name="scalar_operands",
        build=lambda _: ScalarOperands(),
        # mask は torch 既定の i64（境界で i32 へ正規化される）。0 を含めて `1 - mask` の
        # 両方の値を踏む。
        example_inputs=lambda g: (
            _uniform(g, GOLDEN_T, 4, low=-1.0, high=1.0),
            _mask(GOLDEN_T, 4, zero_at=7),
        ),
        dynamic_shapes=({0: _DYNAMIC_T}, {0: _DYNAMIC_T}),
    ),
    GoldenSpec(
        name="spline_pieces",
        build=lambda _: SplinePieces(),
        # ±TAIL（1.0）と clamp の両端（-0.75 / 0.5）をどちらも跨ぐ列。境界ちょうど（-1.0）も
        # 1 点入れて ge と gt の分かれ目を踏む。
        example_inputs=lambda g: (
            torch.tensor([-2.0, -1.0, -0.4, 0.3, 1.2, 2.0]),
            _uniform(g, GOLDEN_T, 4, low=0.2, high=0.8),
        ),
        dynamic_shapes=({0: _DYNAMIC_T}, {0: _DYNAMIC_T}),
    ),
    GoldenSpec(
        name="coupling_split",
        build=lambda _: CouplingSplit(),
        # x[1,6,T]（チャネル軸を静的に保ったまま T は記号）と attn[2,4]（最終次元を pad）
        example_inputs=lambda g: (
            _uniform(g, 1, 6, GOLDEN_T, low=-1.5, high=1.5),
            _uniform(g, 2, 4, low=-1.0, high=1.0),
        ),
        dynamic_shapes=({2: _DYNAMIC_T}, None),
    ),
    GoldenSpec(
        name="dilated_conv",
        build=DilatedConvStack,
        example_inputs=lambda g: (
            _uniform(g, 1, DilatedConvStack.DEPTH_CHANNELS, GOLDEN_T, low=-1.5, high=1.5),
        ),
        dynamic_shapes=({2: _DYNAMIC_T},),
    ),
    GoldenSpec(
        name="conv_transpose",
        build=ConvTransposeStack,
        example_inputs=lambda g: (
            _uniform(g, 1, ConvTransposeStack.IN_CHANNELS, GOLDEN_T, low=-1.5, high=1.5),
        ),
        dynamic_shapes=({2: _DYNAMIC_T},),
    ),
    GoldenSpec(
        name="decoder_tail",
        build=DecoderTail,
        example_inputs=lambda g: (
            _uniform(g, 1, DecoderTail.CHANNELS, GOLDEN_T, low=-1.5, high=1.5),
            _uniform(g, 1, DecoderTail.CHANNELS, 1, low=0.5, high=1.5),
        ),
        dynamic_shapes=({2: _DYNAMIC_T}, None),
    ),
    GoldenSpec(
        name="rms_norm_block",
        build=RmsNormBlock,
        # 0 を跨ぐ値域（rms の分母は二乗和なので符号に依らないが、weight 掛けの符号は出る）
        example_inputs=lambda g: (
            _uniform(g, 1, GOLDEN_T, RmsNormBlock.HIDDEN, low=-1.5, high=1.5),
        ),
        dynamic_shapes=({1: _DYNAMIC_T},),
    ),
    GoldenSpec(
        name="conv2d_block",
        build=Conv2dBlock,
        # x は H≠W（6×5）。channels は 1 点だけノルム 0 にして clamp_min の床を踏む。
        example_inputs=lambda g: (
            _uniform(g, 1, 4, 6, 5, low=-1.5, high=1.5),
            _zeroed_column(g, 1, 4, 3, 5, at=(1, 2)),
        ),
    ),
    GoldenSpec(
        name="i8_weights",
        build=Int8Weights,
        # index は torch 既定の i64（境界で i32 へ正規化される）。**0 を必ず含める** —
        # 全ゼロ行（scale の下限 clamp）を引くのはこのケースだけ。
        example_inputs=lambda g: (
            torch.arange(GOLDEN_T, dtype=torch.int64) % Int8Weights.VOCAB,
            _uniform(g, 1, Int8Weights.SIGNAL_IN, GOLDEN_T, low=-1.5, high=1.5),
            _uniform(g, 1, Int8Weights.IMAGE_IN, 5, 3, low=-1.5, high=1.5),
        ),
        dynamic_shapes=({0: _DYNAMIC_T}, {2: _DYNAMIC_T}, None),
        weight_dtype="i8",
    ),
)


def _assert_not_trivial(where: str, tensor: torch.Tensor) -> None:
    """golden の出力が**自明値でない**ことを生成時に固定する。

    2 要素以上あるのに全要素が同じ値の出力は、演算の誤りが値に出ない恒真な期待値
    （実例: mask ∈ {0,1} に対する `(1 - mask) * mask` は逆順 sub でも恒等 0）。突合が
    緑でも何も検証していないので、fixture を書き出す前に落とす。
    """
    if tensor.numel() > 1 and tensor.unique().numel() == 1:
        raise AssertionError(
            f"{where}: 全要素が同じ値（{tensor.flatten()[0].item()}）の自明な期待値 —"
            " 演算の誤りが値に出ない golden になっている"
        )


def generate_golden(spec: GoldenSpec, root: Path) -> IrGraph:
    """1 モデル分の model.safetensors と io.safetensors を書き、グラフを返す。"""
    generator = _rng()
    module = spec.build(generator).eval()
    args = spec.example_inputs(generator)
    # MUST: fake-quant は export と**下の参照採取の両方より前**（ADR 0006）。後ろへ動かすと
    # 期待値だけが元の重みで計算され、E2E の差が量子化誤差と実装誤差の合成になる。
    scales = fake_quant_int8(module).scales if spec.weight_dtype == "i8" else None
    out_dir = root / spec.name
    graph = export_to_file(
        module,
        args,
        out_dir / MODEL_FILE,
        dynamic_shapes=spec.dynamic_shapes,
        symbol_names=spec.symbol_names,
        weight_dtype=spec.weight_dtype,
        weight_scales=scales,
        preserved=spec.preserved,
    )

    with torch.no_grad():
        result = module(*args)
    outputs = result if isinstance(result, tuple) else (result,)
    if len(outputs) != len(graph.outputs):
        raise AssertionError(
            f"{spec.name}: torch 出力 {len(outputs)} 本 ≠ IR 出力 {len(graph.outputs)} 本"
        )
    for index, tensor in enumerate(outputs):
        _assert_not_trivial(f"{spec.name} の出力 {index}", tensor)
    # MUST: io も IR の意味論 dtype の実表現へ落とす（i64 → I32 / bool → U32 の 0/1）。
    # ランタイムが受け取る形と揃っていないと、Deno 側 E2E が golden を読めない。
    io_tensors = {
        f"{INPUT_PREFIX}{declared.name}": normalize_boundary_tensor(
            tensor.detach(), f"{spec.name} の入力 '{declared.name}'"
        )
        for declared, tensor in zip(graph.inputs, args, strict=True)
    }
    for index, tensor in enumerate(outputs):
        io_tensors[f"{OUTPUT_PREFIX}{index}"] = normalize_boundary_tensor(
            tensor.detach(), f"{spec.name} の出力 {index}"
        )
    save_file(io_tensors, str(out_dir / IO_FILE))
    return graph


def generate_all(root: Path = GOLDEN_ROOT) -> dict[str, IrGraph]:
    """全 golden を生成し、契約表の op 被覆を検査する。

    被覆検査を生成側に置くのは、op を 1 個足すたびに golden を足す実装契約
    （ADR 0005）を機械で担保するため — モデルを削って穴を空けた瞬間に落ちる。
    """
    graphs = {spec.name: generate_golden(spec, root) for spec in GOLDEN_SPECS}
    covered = {op for graph in graphs.values() for op in graph.required_ops}
    uncovered = sorted(EMITTABLE_OPS - covered)
    if uncovered:
        raise AssertionError(f"golden がカバーしない op: {uncovered}")
    return graphs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=GOLDEN_ROOT)
    args = parser.parse_args()
    for name, graph in generate_all(args.out).items():
        print(f"{name}: ops={graph.required_ops} symbols={graph.symbols}")
    print(f"wrote {len(GOLDEN_SPECS)} models to {args.out}")


if __name__ == "__main__":
    main()
