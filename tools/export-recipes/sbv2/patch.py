"""SBV2（Style-Bert-VITS2 JP-Extra）を torch.export 可能にするパッチ層（ADR 0013）。

対象は **front**（enc_p + dp + sdp reverse の融合 1 グラフ）と、後半チェーンの
**flow**（TransformerCouplingBlock の reverse）/ **voice**（flow + dec の融合）。パッチは
import 済みクラスの属性差し替え（monkeypatch）とラッパで行い、`style_bert_vits2` パッケージ
本体には触れない（**dec** はパッチ不要 — `remove_weight_norm` だけが前処理）。

パッチは 3 種 + ラッパ 2 本:

1. **spline の分岐フリー・非破壊化** — `transforms.py` の boolean-mask indexing / テンソル値
   依存の Python 分岐 / in-place を除いた同値実装。原形は「区間内の要素だけを 1 次元へ抽出」
   するため要素数がデータ依存になり、`torch.export` が GuardOnDataDependentSymNode で落ちる。
2. **相対位置注意の gather 化** — rel⇄abs シフトが作る `2P−1` / `2P²` / `P(2P−1)` の二次
   shape 式（front 側で 144 箇所）を、`P` 非依存の添字表 + 最終次元 gather に置換する。
   二次式は次元言語（`coeff·sym+offset`）のアフィン拡張でも救えないので、**動的 P を成立
   させる構造的前提**であって最適化ではない。
3. **FFN の明示 pad → conv の padding 引数** — `constant_pad_nd` を畳んで消す（奇数 kernel
   かつ非 causal のときだけ厳密に等価 — 下の `_patched_ffn_forward` を参照）。

MUST: パッチ後のモジュールはパッチ前と **eager 同値**であること。同値でない変更をここに
置いてはならない（`sbv2.export --verify` が実重み・複数 P で実測する）。

MUST: パッチはクラス属性の**プロセス全域**差し替えなので、「パッチ前の参照」を採れるのは
1 プロセスにつき 1 回だけ。適用済みかどうかは {@link patches_applied} が答え、順序違反は
呼び出し側（`sbv2.export`）が fail loudly で拒否する（恒真化 = 偽 PASS の遮断）。
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional

#: spline の既定値（`style_bert_vits2.models.transforms` と同じ値）。差し替え先の関数
#: シグネチャに既定値として現れるので、原実装と食い違うと呼び出し側の省略時に別の spline
#: になる（沈黙誤値）。
DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3

#: パッチ適用済みフラグ。プロセス全域差し替えの副作用を可視化するためだけに持つ
#: （パッチ後に「パッチ前の参照」を採ると同値検証が恒真化する — ADR 0013）。
_APPLIED = False


def patches_applied() -> bool:
    """このプロセスで既にパッチを当てたか。

    参照値を採る側（同値検証）が「まだ当てていない」ことを assert するための門。
    """
    return _APPLIED


# ---- ① spline の分岐フリー・非破壊化 --------------------------------------


def _constant_column(t: torch.Tensor, value: float) -> torch.Tensor:
    """`t[..., :1]` と同形の定数列を **既存の語彙だけで**作る。

    `full_like` / `zeros_like` は IR の語彙に無い（定数生成 op を足すと、実行時 shape に
    依存する定数という新しい概念が IR に入る）。`x * 0.0 + value` は t が有限値である限り
    厳密に `value` になる — spline へ来る値は clamp 済みで有限なので前提は満たされる。
    """
    return t[..., :1] * 0.0 + value


def searchsorted_free(
    bin_locations: torch.Tensor, inputs: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """原実装 `transforms.searchsorted` の in-place 破壊を除いた同値版。

    原実装は `bin_locations[..., -1] += eps` で**引数を破壊**してから比較する。非破壊化
    （末尾 1 列だけ eps を足した列を cat で組み直す）が同値なのは、破壊された最終要素が
    後段の gather から**参照されないため** — 返り値は `sum(...) - 1` で高々 `num_bins - 1`
    にしかならず、gather の添字は常に最終要素の 1 つ手前までしか届かない。この理由を
    運ばないと「同値でない変更」に見える。
    """
    last = bin_locations[..., -1:] + eps
    shifted = torch.cat([bin_locations[..., :-1], last], dim=-1)
    return torch.sum(inputs[..., None] >= shifted, dim=-1) - 1


def rational_quadratic_spline_free(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool = False,
    left: float = 0.0,
    right: float = 1.0,
    bottom: float = 0.0,
    top: float = 1.0,
    min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float = DEFAULT_MIN_DERIVATIVE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`transforms.rational_quadratic_spline` の分岐フリー・非破壊版（数値同値）。

    原実装からの差分は 3 点だけ:

    - **定義域検査の削除** — `if torch.min(inputs) < left ...: raise` はテンソルの値に
      依存した Python 分岐で、export 時に data-dependent guard になる。呼び出し元
      （`unconstrained_rqs_free`）が定義域へ clamp 済みの値しか渡さないので検査は恒真。
      **安全弁を外した以上、この関数を clamp 抜きで呼ばせてはならない**（差し替え関数
      `piecewise_free` の `tails is None` 分岐が RuntimeError なのはこのため）。
    - **累積境界の組み立て** — `pad` してから `cumwidths[..., 0] = left` / `[..., -1] = right`
      と slice 代入する形を、cat による構築に置換（in-place の除去）。値は同一。
    - **searchsorted の非破壊化**（{@link searchsorted_free}）。
    """
    num_bins = unnormalized_widths.shape[-1]
    # 形状だけで決まる前提（テンソルの値に依らない）ので Python 分岐のまま残せる。
    if min_bin_width * num_bins > 1.0:
        raise ValueError(f"min_bin_width {min_bin_width} が bin 数 {num_bins} に対して大きすぎる")
    if min_bin_height * num_bins > 1.0:
        raise ValueError(f"min_bin_height {min_bin_height} が bin 数 {num_bins} に対して大きすぎる")

    widths = functional.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    scaled_widths = (right - left) * torch.cumsum(widths, dim=-1) + left
    # 原実装の「先頭に 0 を pad → スケール → 両端を left / right で上書き」と同じ列。
    # 末尾の 1 要素は捨てる（累積和の最終値は上書きされる側）。
    cumwidths = torch.cat(
        [
            _constant_column(scaled_widths, left),
            scaled_widths[..., :-1],
            _constant_column(scaled_widths, right),
        ],
        dim=-1,
    )
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + functional.softplus(unnormalized_derivatives)

    heights = functional.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    scaled_heights = (top - bottom) * torch.cumsum(heights, dim=-1) + bottom
    cumheights = torch.cat(
        [
            _constant_column(scaled_heights, bottom),
            scaled_heights[..., :-1],
            _constant_column(scaled_heights, top),
        ],
        dim=-1,
    )
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    locations = cumheights if inverse else cumwidths
    bin_idx = searchsorted_free(locations, inputs)[..., None]

    # 原実装の `gather(...)[..., 0]` は `aten.select.int`（軸を 1 本落とす添字アクセス）に
    # なる。IR には select が無く、長さ 1 の軸を落とすこの形は要素順を変えない純粋な
    # メタ操作なので `squeeze(-1)`（= reshape）で書く。値もレイアウトも同一。
    input_cumwidths = cumwidths.gather(-1, bin_idx).squeeze(-1)
    input_bin_widths = widths.gather(-1, bin_idx).squeeze(-1)
    input_cumheights = cumheights.gather(-1, bin_idx).squeeze(-1)
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx).squeeze(-1)
    input_derivatives = derivatives.gather(-1, bin_idx).squeeze(-1)
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx).squeeze(-1)
    input_heights = heights.gather(-1, bin_idx).squeeze(-1)

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)
        # 原実装の `assert (discriminant >= 0).all()` もテンソル値依存の Python 分岐なので
        # 落とす（定義域内では数学的に非負 — 単調な rational-quadratic の逆解）。
        discriminant = b.pow(2) - 4 * a * c
        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths
        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, -logabsdet

    theta = (inputs - input_cumwidths) / input_bin_widths
    theta_one_minus_theta = theta * (1 - theta)
    numerator = input_heights * (
        input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
    )
    denominator = input_delta + (
        (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
    )
    outputs = input_cumheights + numerator / denominator
    derivative_numerator = input_delta.pow(2) * (
        input_derivatives_plus_one * theta.pow(2)
        + 2 * input_delta * theta_one_minus_theta
        + input_derivatives * (1 - theta).pow(2)
    )
    logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
    return outputs, logabsdet


def unconstrained_rqs_free(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool = False,
    tails: str = "linear",
    tail_bound: float = 1.0,
    min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float = DEFAULT_MIN_DERIVATIVE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """boolean-mask indexing を clamp + where に置換した同値版。

    原実装は `inputs[inside_interval_mask]` で区間内の要素だけを 1 次元へ抽出するため、
    **要素数がテンソルの値で決まる**（export できない）。全要素を定義域へ clamp して
    spline に通し、区間外は where で入力そのものへ戻す — 区間内の要素は clamp が恒等
    （既に定義域内）なので選ばれる値は原実装と厳密に一致し、区間外の要素は spline の
    結果が捨てられるので clamp が何を返そうと影響しない。
    """
    if tails != "linear":
        raise RuntimeError(f"{tails} tails are not implemented.")

    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)

    # 原実装の「両端に pad → 端の 2 要素を constant で上書き」と同じ列（in-place の除去）。
    constant = float(np.log(np.exp(1 - min_derivative) - 1))
    tail_column = _constant_column(unnormalized_derivatives, constant)
    padded_derivatives = torch.cat([tail_column, unnormalized_derivatives, tail_column], dim=-1)

    spline_outputs, spline_logabsdet = rational_quadratic_spline_free(
        inputs=torch.clamp(inputs, -tail_bound, tail_bound),
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=padded_derivatives,
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )
    outputs = torch.where(inside, spline_outputs, inputs)
    # 区間外の logabsdet は 0（原実装の `torch.zeros_like` 初期値そのもの）。定数 0 は
    # `x * 0.0` で作る — zeros_like を IR に持ち込まないための定型手筋。
    logabsdet = torch.where(inside, spline_logabsdet, spline_logabsdet * 0.0)
    return outputs, logabsdet


def apply_spline_patch() -> None:
    """spline 実装を分岐フリー版へ差し替える。

    MUST: 差し替え先は **`style_bert_vits2.models.modules`** — `modules.py` は
    `from ...transforms import piecewise_rational_quadratic_transform` で関数オブジェクトを
    自分の名前空間へ束縛済みなので、`transforms` 側を差し替えても呼び出しには効かない
    （差し替えたつもりで原実装が走り、export だけが落ちる形になる）。
    """
    from style_bert_vits2.models import modules

    def piecewise_free(
        inputs,
        unnormalized_widths,
        unnormalized_heights,
        unnormalized_derivatives,
        inverse=False,
        tails=None,
        tail_bound=1.0,
        min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE,
    ):
        if tails is None:
            # MUST: 握り潰さない。`rational_quadratic_spline_free` は原実装の定義域検査
            # （InputOutsideDomain）を削除しているので、tails=None はクランプ無しでその
            # spline へ直行する経路 = **安全弁だけが外れた形**になる。SBV2 の呼び出し元
            # （ConvFlow）は tails="linear" 固定で到達不能だが、到達したら落とす。
            raise RuntimeError("tails=None は未対応（定義域クランプ経由でのみ spline を呼ぶ）")
        return unconstrained_rqs_free(
            inputs,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=inverse,
            tails=tails,
            tail_bound=tail_bound,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
        )

    modules.piecewise_rational_quadratic_transform = piecewise_free


# ---- ② 相対位置注意の gather 化 -------------------------------------------


def build_relattn_tables(
    length: int,
    window_size: int = 4,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """gather 化注意の `(T, T)` 表 `(idx_k, valid)` を実長で生成する（Python 側の正本）。

    front（P ≤ 512）は同じ式を **in-graph** で組んでエクスポータの定数畳み込みに
    Pmax 焼き込み + `sym_prefix_slice` を作らせるが、flow は Ty が桁違いで焼き込みが
    O(Tmax²)（sym_max=4096 で 134MB）になるため**グラフ入力へ昇格**する（ADR 0013）。
    昇格すると表の値を作る責務がホスト側へ移るので、ここが golden の供給元、
    `packages/models/src/sbv2/relattn-tables.ts` が TS 側の鏡像で、両者のバイト一致は
    `packages/models/tests/sbv2_relattn_parity_test.ts` が golden の実データで固定する。

    MUST: front の in-graph 構築も**この関数を呼ぶ**（{@link mha_gather_forward} の
    `relattn_tables is None` 経路）。式を 2 箇所に書くと、片方だけ直したとき front
    （焼き込み）と flow（入力）で別の表になり、どちらも shape は合うので黙って別の
    モデルになる（沈黙誤値クラス）。`device` / `dtype` はトレース中の in-graph 構築が
    余計な `_to_copy` を生まないための引数で、ホスト供給側は既定（CPU / f32）を使う。
    """
    positions = torch.arange(length, device=device)
    rel = positions.unsqueeze(0) - positions.unsqueeze(1)
    idx_k = torch.clamp(rel + window_size, 0, 2 * window_size)
    valid = (torch.abs(rel) <= window_size).to(dtype)
    return idx_k, valid


def mha_gather_forward(
    self,
    x: torch.Tensor,
    c: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    relattn_tables: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """窓付き相対位置注意の gather 版 `MultiHeadAttention.forward`（原実装と同値）。

    原実装は相対 logits `(b,h,P,2P−1)` を pad / view のシフトで絶対位置 `(b,h,P,P)` へ
    写す。この経路が `2P−1` / `2P²` / `P(2P−1)` という**二次の shape 式**を作るため、
    動的 P では export が成立しない。窓幅 `2w+1` の外が恒等的に 0 であることを使って、
    同じ値を添字表と gather で組む:

    - key 側: `scores_local[i,j] = (q_i·Ek)[w + (j−i)] · [|j−i| ≤ w]`
      → 添字表 `idx_k[i,j] = clamp(w + j − i, 0, 2w)` の gather × 0/1 マスク乗算。
      clamp した添字は窓外で**別の埋め込みを読む**が、valid マスクの乗算で必ず 0 になる
      （gather の範囲外規約に頼らない形 — ADR 0013）。
    - value 側: `out_i += Σ_j p[i,j]·Ev[w + (j−i)]`（窓外は 0）
      → p を `[w,w]` ゼロパディングして `idx_v[i,c] = i + c` を gather。パディング済みの
      長さ `P+2w` に対し `i+c` は常に範囲内なので、窓外は厳密に 0 を読む（clamp 不要）。

    表は `i` と `j`（と定数 w）だけで決まり **P に依存しない**ので、エクスポータの定数
    畳み込みが Pmax で焼き込み + `sym_prefix_slice` に落とす（2 点評価検査も通る）。

    `relattn_tables=(idx_k, valid)` を渡すと key 側の `(T, T)` 表だけが**外部供給**へ
    切り替わる（flow 経路 — {@link FlowReverse} がグラフ入力をここまでスレッディングする）。
    None なら in-graph 構築（front 経路 — 焼き込み適格）。値は両経路で同一
    （{@link build_relattn_tables} と同じ式）。**value 側の `idx_v` は `(T, 2w+1)` と
    小さく、焼き込んでも Tmax=4096 で 150KB にしかならない**ので flow でも in-graph の
    ままにする（グラフ入力を増やすほどホスト側の鏡像実装の責務が増える）。

    MUST: 埋め込み側は **4D に expand して 4D×4D** で掛ける。2D のまま matmul すると
    `(b·h·P, kc)` の view に分解され、係数付きシンボル次元 `2P` が生えて次元言語の
    アフィン拡張が要るように見えてしまう（実際には要らない — recon §4）。
    """
    if self.window_size is None:
        raise ValueError("gather 化は窓付き相対位置注意が対象（window_size=None）")
    if not self.heads_share:
        raise ValueError(
            "heads_share=False は未対応 — 埋め込みの head 軸（長さ 1）を expand で"
            " 全 head に配る形が成立しなくなる"
        )
    if self.block_length is not None or self.proximal_bias:
        raise ValueError(
            f"未対応の注意構成（block_length={self.block_length} /"
            f" proximal_bias={self.proximal_bias}）"
        )

    q = self.conv_q(x)
    k = self.conv_k(c)
    v = self.conv_v(c)

    b, d, t_s = k.size()
    t_t = q.size(2)
    if t_s != t_t:
        raise ValueError("gather 化は self-attention（正方の相対位置表）が対象")
    h = self.n_heads
    kc = self.k_channels
    w = self.window_size
    q = q.view(b, h, kc, t_t).transpose(2, 3)
    k = k.view(b, h, kc, t_s).transpose(2, 3)
    v = v.view(b, h, kc, t_s).transpose(2, 3)

    scaled_q = q / math.sqrt(kc)
    scores = torch.matmul(scaled_q, k.transpose(-2, -1))

    # rel[i,j] = j − i（原実装の rel→abs 写像 `x_final[i,j] = x[i, j−i+P−1]` と同じ向き）。
    positions = torch.arange(t_t, device=x.device)
    if relattn_tables is None:
        idx_k, valid = build_relattn_tables(t_t, w, device=x.device, dtype=scores.dtype)
    else:
        idx_k, valid = relattn_tables

    # emb_rel_k は [1, 2w+1, kc]（heads_share なので head 軸は長さ 1）。head 軸をそのまま
    # broadcast 元にして [1, 1, kc, 2w+1] を作る（`emb[0]` の select を経由しない）。
    ek = self.emb_rel_k.permute(0, 2, 1).unsqueeze(0)
    rel_local = torch.matmul(scaled_q, ek.expand(b, h, kc, 2 * w + 1))
    scores_local = torch.gather(
        rel_local, -1, idx_k.unsqueeze(0).unsqueeze(0).expand(b, h, t_t, t_t)
    ) * valid.unsqueeze(0).unsqueeze(0)
    scores = scores + scores_local

    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask == 0, -1e4)
    p_attn = functional.softmax(scores, dim=-1)
    p_attn = self.drop(p_attn)
    output = torch.matmul(p_attn, v)

    p_pad = functional.pad(p_attn, [w, w])
    window = torch.arange(2 * w + 1, device=x.device)
    idx_v = positions.unsqueeze(1) + window.unsqueeze(0)
    p_local = torch.gather(p_pad, -1, idx_v.unsqueeze(0).unsqueeze(0).expand(b, h, t_t, 2 * w + 1))
    ev = self.emb_rel_v.unsqueeze(0)
    output = output + torch.matmul(p_local, ev.expand(b, h, 2 * w + 1, kc))

    output = output.transpose(2, 3).contiguous().view(b, d, t_t)
    return self.conv_o(output)


def apply_gather_relattn_patch() -> None:
    """`MultiHeadAttention.forward` を gather 版へ差し替える。

    原実装は `x, self.attn = self.attention(...)` で**モジュールへ副作用代入**しており、
    export ではこれ自体が警告になる。差し替えで副作用ごと消える（`self.attn` は推論経路の
    誰も読まない）。
    """
    from style_bert_vits2.models.attentions import MultiHeadAttention

    MultiHeadAttention.forward = mha_gather_forward


# ---- ③ FFN の明示 pad → conv の padding ------------------------------------


def _patched_ffn_forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
    """`FFN.forward` から明示 `F.pad` を消し、conv の padding 引数へ畳んだ版。

    MUST: 等価なのは **奇数 kernel かつ非 causal** のときだけ。原実装の `_same_padding` は
    左 `(k−1)//2` / 右 `k//2` の非対称パディングで、偶数 kernel では左右が 1 ずれる
    （conv1d の padding 引数は左右対称しか表せない）。causal 側は左だけ `k−1` 詰めるので
    そもそも別物。どちらも黙って通すと出力が 1 サンプルずれた別のモデルになるので落とす。
    """
    if self.causal:
        raise ValueError("causal FFN はこのパッチの対象外（左右非対称パディング）")
    if self.kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size={self.kernel_size} は偶数 — 明示 pad と conv の padding は等価でない"
        )
    padding = (self.kernel_size - 1) // 2
    x = functional.conv1d(x * x_mask, self.conv_1.weight, self.conv_1.bias, padding=padding)
    x = x * torch.sigmoid(1.702 * x) if self.activation == "gelu" else torch.relu(x)
    x = self.drop(x)
    x = functional.conv1d(x * x_mask, self.conv_2.weight, self.conv_2.bias, padding=padding)
    return x * x_mask


def apply_ffn_conv_padding_patch() -> None:
    """`FFN.forward` を pad 畳み込み版へ差し替える。"""
    from style_bert_vits2.models.attentions import FFN

    FFN.forward = _patched_ffn_forward


def apply_all_patches() -> None:
    """front の export に必要な全パッチを当てる（冪等）。

    MUST: 「パッチ前の参照」を採る処理より**後**に呼ぶこと。クラス属性のプロセス全域
    差し替えなので、先に当てると以後の参照値がパッチ後の値になり、同値検証が恒真化して
    偽 PASS する（ADR 0013）。
    """
    global _APPLIED
    apply_spline_patch()
    apply_gather_relattn_patch()
    apply_ffn_conv_padding_patch()
    _APPLIED = True


# ---- ④ 乱数の外出し + 融合 front ラッパ ------------------------------------


class SdpReverseNoiseIn(nn.Module):
    """`StochasticDurationPredictor` の reverse 経路 — 乱数を外部入力へ昇格した版。

    原 `forward(..., reverse=True)` と等価で、差分は `torch.randn(B,2,P) * noise_scale` を
    外部入力 `z_noise` で受ける 1 点のみ（`noise_scale` の乗算はホスト側 — 実行時ノブを
    グラフに焼かない）。`torch.detach` は eager の推論経路で恒等なので写さない。
    """

    def __init__(self, sdp: nn.Module) -> None:
        super().__init__()
        self.sdp = sdp

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor,
        z_noise: torch.Tensor,
    ) -> torch.Tensor:
        sdp = self.sdp
        x = sdp.pre(x)
        x = x + sdp.cond(g)
        x = sdp.convs(x, x_mask)
        x = sdp.proj(x) * x_mask

        # 原実装どおり: reversed 順から末尾 2 本目（無用な vflow）を除く。
        flows = list(reversed(sdp.flows))
        flows = [*flows[:-2], flows[-1]]
        z = z_noise
        for flow in flows:
            z = flow(z, x_mask, g=x, reverse=True)
        # 原実装は `torch.split(z, [1, 1], 1)` の第 1 要素。分割は取り出し口 1 本で足りる。
        return z[:, :1]


class Sbv2Front(nn.Module):
    """enc_p + dp + sdp(reverse) の融合 front グラフ（ADR 0013 の emit ターゲット）。

    - `x_mask` は外部入力（原 `TextEncoder.forward` の `sequence_mask` 生成はホスト側へ）。
      長さから mask を組む部分はグラフに入れても定数畳み込みの対象にならず、`x_lengths` を
      値として使う形（記号を値にする経路）になるため。
    - `TextEncoder.forward` はここへ写した — 原形は x_mask を内部生成し `torch.split` で
      m/logs を割るので、入力の形と分割の形を両方ここで固定する（値は原実装と同一）。
    - 出力は `(logw_sdp, logw_dp, m_p, logs_p)`。sdp_ratio の混合・durations 化はホスト側。
    """

    def __init__(self, net_g: nn.Module) -> None:
        super().__init__()
        self.enc_p = net_g.enc_p
        self.sdp_rev = SdpReverseNoiseIn(net_g.sdp)
        self.dp = net_g.dp

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        tone: torch.Tensor,
        language: torch.Tensor,
        bert: torch.Tensor,
        style_vec: torch.Tensor,
        g: torch.Tensor,
        z_noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        enc = self.enc_p
        bert_emb = enc.bert_proj(bert).transpose(1, 2)
        style_emb = enc.style_proj(style_vec.unsqueeze(1))
        h = (
            enc.emb(x) + enc.tone_emb(tone) + enc.language_emb(language) + bert_emb + style_emb
        ) * math.sqrt(enc.hidden_channels)
        h = torch.transpose(h, 1, -1)
        h = enc.encoder(h * x_mask, x_mask, g=g)
        stats = enc.proj(h) * x_mask
        channels = enc.out_channels
        m_p, logs_p = stats[:, :channels], stats[:, channels:]

        logw_sdp = self.sdp_rev(h, x_mask, g, z_noise)
        logw_dp = self.dp(h, x_mask, g=g)
        return logw_sdp, logw_dp, m_p, logs_p


# ---- ⑤ flow reverse（表を入力へ昇格）と voice 融合 ---------------------------


def _encoder_forward(
    enc: nn.Module,
    x: torch.Tensor,
    x_mask: torch.Tensor,
    g: torch.Tensor,
    tables: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """`attentions.Encoder.forward` の写し + 相対位置表のスレッディング。

    front は `enc_p.encoder(...)` を**そのまま**呼ぶ（表は in-graph 構築 = 焼き込み適格）。
    flow は表をグラフ入力から注意層まで運ぶ必要があるが、`Encoder.forward` の引数には
    表を載せる口が無い。**Encoder.forward もクラス属性差し替えで表を受ける形にする**案は
    front 経路（表なし）と分岐を共有することになり、「front が誤って flow の表を掴む」
    経路を作ってしまうので採らない — 写しを 1 本置いて呼び分ける方が経路が交わらない。

    MUST: 原実装（`attentions.py` の `Encoder.forward`）と行対応を保つこと。`isflow`
    構成の条件付け（`cond_layer_idx` 層で `spk_emb_linear(g)` を足す）を含む。`drop` は
    eval では恒等だが、原実装との突合を行単位で追えるように残す。
    """
    attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
    x = x * x_mask
    for i in range(enc.n_layers):
        if i == enc.cond_layer_idx and g is not None:
            gl = enc.spk_emb_linear(g.transpose(1, 2))
            gl = gl.transpose(1, 2)
            x = x + gl
            x = x * x_mask
        y = mha_gather_forward(enc.attn_layers[i], x, x, attn_mask, tables)
        y = enc.drop(y)
        x = enc.norm_layers_1[i](x + y)

        y = enc.ffn_layers[i](x, x_mask)
        y = enc.drop(y)
        x = enc.norm_layers_2[i](x + y)
    x = x * x_mask
    return x


def _coupling_reverse(
    layer: nn.Module,
    x: torch.Tensor,
    x_mask: torch.Tensor,
    g: torch.Tensor,
    tables: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """`modules.TransformerCouplingLayer.forward(reverse=True)` の写し（数値同値）。

    原実装からの差分は 2 点:

    - `torch.split(x, [half] * 2, 1)` → 明示スライス 2 本（`split_with_sizes` → `getitem`
      は IR に無い形で、正規化で slice へ落とす経路も持たない — front の `stats` 分割と
      同じ流儀）。
    - **`exp(-logs)` の乗算を畳む**。`mean_only=True` なので原実装の `logs` は
      `torch.zeros_like(m)`、`exp(-0.0)` は IEEE 754 で**厳密に 1.0**、`t * 1.0 == t` は
      NaN/Inf を含めてビット一致するので `(x1 − m) * exp(−logs) * x_mask` は
      `(x1 − m) * x_mask` と同じ数になる。`zeros_like` / `full_like` を IR に持ち込まない
      ための定型手筋（recon §4）で、近似ではない。
    """
    if not layer.mean_only:
        raise ValueError(
            "mean_only=False の coupling は未対応 — logs が 0 でなくなり exp(−logs) を畳めない"
        )
    half = layer.half_channels
    x0, x1 = x[:, :half], x[:, half:]
    h = layer.pre(x0) * x_mask
    h = _encoder_forward(layer.enc, h, x_mask, g, tables)
    m = layer.post(h) * x_mask
    x1 = (x1 - m) * x_mask
    return torch.cat([x0, x1], 1)


class FlowReverse(nn.Module):
    """`TransformerCouplingBlock` の reverse 経路 — 相対位置表を入力へ昇格した版。

    `forward(z_p, y_mask, g, idx_k, valid) -> z`。原
    `TransformerCouplingBlock.forward(..., reverse=True)` と等価で、差分は

    - 表 `idx_k` `(T, T)` i64 / `valid` `(T, T)` f32 を**グラフ入力**で受ける
      （焼き込みは sym_max=4096 で 134MB — ADR 0013）。値はホストが
      {@link build_relattn_tables} の鏡像で作る。
    - `Flip` は `torch.flip(x, [1])`（`modules.Flip.forward` の reverse 分岐は logdet を
      返さないだけで値は同じ）。
    - coupling は {@link _coupling_reverse}（split → slice、`exp(−logs)=1` の畳み込み）。

    flows のループは Python 側で展開されるので、IR は完全に静的な直列グラフになる。
    """

    def __init__(self, net_g: nn.Module) -> None:
        super().__init__()
        self.flow = net_g.flow

    def forward(
        self,
        z_p: torch.Tensor,
        y_mask: torch.Tensor,
        g: torch.Tensor,
        idx_k: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        from style_bert_vits2.models import modules

        x = z_p
        for flow in reversed(self.flow.flows):
            if isinstance(flow, modules.Flip):
                x = torch.flip(x, [1])
            else:
                x = _coupling_reverse(flow, x, y_mask, g, (idx_k, valid))
        return x


class Sbv2Voice(nn.Module):
    """flow reverse + dec の融合グラフ（ADR 0013 の emit ターゲット `voice`）。

    `forward(z_p, y_mask, g, idx_k, valid) -> audio [1, 1, 512T]`。参照 infer の末尾
    （`models_jp_extra.py` の `z = flow(z_p, y_mask, g, reverse=True)` →
    `o = dec((z * y_mask)[:, :, :max_len], g=g)`）と同順。`max_len` は推論経路で常に
    `None`（＝恒等スライス）なのでグラフには持ち込まない。

    MUST: `dec` は **`remove_weight_norm` 済み**であること（`sbv2.export.ensure_dec_plain`）。
    weight_norm が残っていると `dec.weight` は実効重みではなく、そのまま IR へ書けば
    別のモデルになる。将来の f16/i8 丸めは remove **後**の実効重みに当てる（ADR 0013）。
    """

    def __init__(self, net_g: nn.Module) -> None:
        super().__init__()
        self.flow_rev = FlowReverse(net_g)
        self.dec = net_g.dec

    def forward(
        self,
        z_p: torch.Tensor,
        y_mask: torch.Tensor,
        g: torch.Tensor,
        idx_k: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        z = self.flow_rev(z_p, y_mask, g, idx_k, valid)
        return self.dec(z * y_mask, g=g)
