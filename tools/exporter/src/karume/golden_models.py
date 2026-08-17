"""golden fixture の中身 — tiny な nn.Module 群と、その重み初期化・入力生成ヘルパ。

台帳（{@link karume.goldens.GOLDEN_SPECS}）と生成ドライバは {@link karume.goldens}。
import は goldens → golden_models の一方向のみ。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.ops import deform_conv2d


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


class ArgmaxPick(nn.Module):
    """greedy デコードの出口（ADR 0068 決定 2）— 最終次元 = 語彙軸の argmax・**rank 保存**。

    1 本目は実形（`linear` = lm_head 相当の logits に argmax を掛ける・位置軸は記号 T）。
    2 本目は **同値タイ**への argmax で、`cat([t, t], dim=-1)` が対称位置に必ず同値を作るので
    「最小 index」が偶然ではなく**構造で**要求される — 最大 index を返す実装なら答えが
    `argmax(t) + n` になり、golden 突合がその場で赤くなる（llama.cpp が GPU / CPU で
    食い違っていた軸 — 調査 §2）。

    MUST: タイの入力は**グラフ入力**から作る（定数だと FOLDABLE_OPS が畳んでノードが消える）。
    """

    HIDDEN = 5
    VOCAB = 7

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        self.head = nn.Linear(self.HIDDEN, self.VOCAB)
        _fill_param(generator, self.head.weight, 0.7)
        _fill_param(generator, self.head.bias, 0.3)

    def forward(self, x: torch.Tensor, tied: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.head(x)
        doubled = torch.cat([tied, tied], dim=-1)
        return (
            torch.argmax(logits, dim=-1, keepdim=True),
            torch.argmax(doubled, dim=-1, keepdim=True),
        )


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


class RuntimeMaskedAttention(nn.Module):
    """実行時 bool マスク付き SDPA（ADR 0044）— ガードが `safe_softmax` として焼かれる形。

    ADR 0044 の連鎖を丸ごと踏む: **bool のグラフ入力**（ADR 0009 — u32 0/1 で IO 可）→
    torch の SDPA 分解が出す `where(mask, 0, -inf)` → スコアへ `add` → `softmax` +
    safe-softmax ガード → `_drop_safe_softmax_guard` の実値証明が実行時入力で立たないので
    `safe_softmax` 1 ノードへ構成的置換。

    MUST: マスクに**全 False の行**を含める（`_masked_row_mask`）。その行は加算後に全要素
    -inf になり、torch は 0 を返す — safe_softmax が空行を 0 で書けなければ 0/0 = NaN に
    なって突合が赤くなる。ここが本 golden の存在理由。
    NOTE: SDPA は保存しない（既定の `PRESERVED_OP_PREFIXES` のまま）。融合 attention の
    mask 契約は静的 `[1,1,M,N]` の加算型 f32 だけで、実行時マスクは分解経路で実行する
    （ADR 0044 決定 1）。
    """

    HEADS = 2
    QUERIES = 4
    DEPTH = 3

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask)


def _runtime_masked_attention_inputs(generator: torch.Generator) -> tuple[torch.Tensor, ...]:
    """`RuntimeMaskedAttention` の入力（q / k / v と **bool のマスク**）。

    マスクは `[1,1,M,M]` で True = 残す。残す側を市松にして「全 True の行」だけにならない
    ようにし、行 2 は**全 False**（torch のガードが発火する行）にする。
    """
    shape = RuntimeMaskedAttention
    length = shape.QUERIES
    qkv = tuple(
        _uniform(generator, 1, shape.HEADS, length, shape.DEPTH, low=-1.0, high=1.0)
        for _ in range(3)
    )
    rows = torch.arange(length).reshape(length, 1)
    cols = torch.arange(length).reshape(1, length)
    mask = ((rows + cols) % 3) != 2
    mask[2] = False
    return (*qkv, mask.reshape(1, 1, length, length))


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

    NOTE: チャネル L2 は `anima.patch._l2_normalize_channels` の鏡像で、**permute 無しの
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


class DeformConvBlock(nn.Module):
    """deform_conv2d（DCNv2・第 1' 層 — ADR 0055）の torch 突合。

    新規原子には対になる既存経路が無いので、**torchvision 出力を焼いたこの golden が主門**に
    なる（もう 1 本の門は「offset 全 0・mask 全 1 → conv2d とビット一致」で、そちらは実 GPU
    テスト側が持つ）。踏むべき形:

    MUST: **Kh ≠ Kw / Cin ≠ Cout / padding の H ≠ W**。offset のチャネル並び（偶数 = y /
    奇数 = x）の取り違えは、正方カーネル・対称 padding では値が一致しうる。
    MUST: **offset が入力平面の外を指す**範囲（±2.5）。ゼロ埋めの 2 段（中心が範囲外なら
    タップ全体 0 / 内側でも範囲外の隅はその隅だけ 0）を両方踏ませる — border clamp 実装は
    ここでしか赤くならない。
    MUST: offset は**グラフ入力**にする（Parameter にすると initializer に落ちて「実行時値の
    offset」という本 op の前提を踏まない）。
    MUST: mask の値域は **[0, 2]**（BiRefNet の `2·sigmoid(...)`）。[0,1] に絞ると
    「mask を掛け忘れても大差ない」形になりうるし、補間の**前**に掛ける誤りとも切り分かない。
    NOTE: `point` 分岐は k=1（ASPP の `aspp1`）で **bias 無し** — torchvision の Python ラッパが
    `aten.full([Cout], 0)` を挿し、第 0 層の定数畳み込みが initializer にする経路を踏む。
    """

    IN_CHANNELS = 3
    WIDE_CHANNELS = 5
    POINT_CHANNELS = 4

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        # Kh=3 / Kw=2 / padding (1,0) — Cin 3 → Cout 5
        self.weight = nn.Parameter(
            _uniform(generator, self.WIDE_CHANNELS, self.IN_CHANNELS, 3, 2, low=-1.0, high=1.0)
        )
        self.bias = nn.Parameter(_uniform(generator, self.WIDE_CHANNELS, low=-0.5, high=0.5))
        # k=1 / padding 0 / bias 無し
        self.point = nn.Parameter(
            _uniform(generator, self.POINT_CHANNELS, self.IN_CHANNELS, 1, 1, low=-1.0, high=1.0)
        )

    def forward(
        self,
        x: torch.Tensor,
        offset: torch.Tensor,
        mask: torch.Tensor,
        point_offset: torch.Tensor,
        point_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        wide = deform_conv2d(
            input=x, offset=offset, weight=self.weight, bias=self.bias, padding=(1, 0), mask=mask
        )
        point = deform_conv2d(
            input=x, offset=point_offset, weight=self.point, padding=(0, 0), mask=point_mask
        )
        return wide, point


class GruScanBlock(nn.Module):
    """gru_scan / gru_scan_reverse（第 2 層 — ADR 0056）の torch 突合。

    入力側 GEMM は素の `nn.Linear`（= IR の `linear`）で、op が受け持つのは隠れ側の逐次だけ。
    eager の期待値は `karume/custom_ops.py` の本体が出す（`nn.GRU` の単方向 1 層と
    `torch.equal` でビット一致することは同モジュールの docstring の実測）。踏むべき形:

    MUST: **時間軸を記号にする**。この op の存在理由が「T を記号のまま通す」ことなので、
    静的 T の golden では分解形との差が出ない（`aten.gru.input` は T 回展開されて specialize
    される）。
    MUST: **2 方向を両方出す**。逆方向は「走査順だけが逆で、書き出しは順方向の時間順」で、
    出力側 `flip` を挟む誤り形はここでしか赤くならない。
    MUST: **h0 を 2 通り踏む** — 順方向は `torch.zeros`（第 0 層の定数畳み込みで initializer に
    落ちる経路）、逆方向は**グラフ入力**（実行時値の初期状態）。ゼロ h0 だけだと「h0 を
    読み飛ばすカーネル」が緑のまま通る。
    MUST: 隠れ側の重みは小さめ（縮約 H 本ぶんでゲートが飽和すると、更新式の丸め差が値に
    出ず突合が恒真化する）。
    """

    IN_FEATURES = 4
    HIDDEN = 5

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        self.project = nn.Linear(self.IN_FEATURES, 3 * self.HIDDEN)
        _fill_param(generator, self.project.weight, 0.4)
        _fill_param(generator, self.project.bias, 0.2)
        self.weight_hh = nn.Parameter(
            _uniform(generator, 3 * self.HIDDEN, self.HIDDEN, low=-0.5, high=0.5)
        )
        self.bias_hh = nn.Parameter(_uniform(generator, 3 * self.HIDDEN, low=-0.3, high=0.3))

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, ...]:
        gates = self.project(x)
        zeros = torch.zeros(x.shape[1], self.HIDDEN)
        return (
            torch.ops.karume.gru_scan(gates, zeros, self.weight_hh, self.bias_hh),
            torch.ops.karume.gru_scan_reverse(gates, state, self.weight_hh, self.bias_hh),
        )


class BilinearResize(nn.Module):
    """upsample_bilinear2d（第 1 層・align_corners=True 専業）の torch 突合。

    新規原子には対になる既存経路（分解形 / 別カーネル）が無いので、**torch 出力を焼いた
    この golden が主門**になる。踏むべき形:

    MUST: **非整数倍**（4×5 → 7×9 は scale 3/6 と 4/8 で、出力位置ごとに重みが違う）。
    整数倍だけで固めると「重み表が周期 2 の決め打ち」でも通ってしまう。
    MUST: **縮小**（4×5 → 2×3）。実測（BiRefNet の `forward_enc` / `cxt`）に 8 本ある形で、
    拡大と同じ式・同じ 2 タップで通ることの固定。
    MUST: **H ≠ W かつ Hout ≠ Wout**。正方形だと H と W の取り違えが値に出ない。
    MUST: **入力の高さ 1**（`narrow`）。align_corners の scale は `(in−1)/(out−1)` なので
    in = 1 の軸は scale 0 = 行の複製になる（末尾特例 `index1 = index0` が全出力で発火する
    唯一の形）。同じテンソルの幅は縮小させて、2 軸で別の分岐を同時に踏ませる。
    """

    def forward(self, x: torch.Tensor, narrow: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (
            nn.functional.interpolate(x, size=(7, 9), mode="bilinear", align_corners=True),
            nn.functional.interpolate(x, size=(2, 3), mode="bilinear", align_corners=True),
            nn.functional.interpolate(narrow, size=(3, 4), mode="bilinear", align_corners=True),
        )


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
