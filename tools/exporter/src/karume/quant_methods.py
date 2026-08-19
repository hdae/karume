"""**測定専用**の重み丸め方式（FP4 / NF4 / MXFP4 / k-means codebook）— 格納経路は持たない。

`quantize.py` に載る形（f16 / i8 / i4）は全て emit とランタイムの実装を持つ「**書ける格納
形**」だが、ここに載るのは書けない方式で、狙いは「その方式なら品質がどこまで戻るか」を
GPU コード 0 行で先に答えること（ADR 0069 決定 6 の Phase 0 sweep と同じ思想 — ADR 0058
決定 5 の分担で言えば「品質 = 助言的数値」の数値側）。別モジュールに置くのは、`quantize.py`
の「載っている形 = 出荷できる格納形」という読み方を壊さないため。

MUST: emit へ渡す口を作らない。戻り値は計数だけで scale / codebook の台帳を返さない
（返すと「測定用の丸め」が格納経路へ流れる道ができる — 適格判定は消費 op で決まるので、
emit 側は受け取った時点では正誤を判別できない）。丸めた重みそのものは in-place で
書き換わるので、劣化の測定はそこから採る。

MUST: **同一入力 → ビット同一出力**。k-means の初期化まで決定的（分位点 + 固定反復）に
してあるのは、乱数 seed 依存だと「方式の差」と「seed の差」が分離できなくなるため。

MUST: 丸めは**参照・golden の採取より前**（`quantize` モジュール docstring と同文）。

対象選択は `quantize.iter_quant_targets` の**共有**で、既定は {@link
quantize.fake_quant_int4} と同じ linear 限定・`op_types` で i8 と同じ 5 op 種まで広げられる。
写した別実装にしないのは、測った対象と出荷した対象が黙って割れるのを防ぐため。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from .quantize import (
    DEFAULT_GROUP_SIZE,
    QuantizeError,
    channel_rows,
    group_size_of,
    grouped_view,
    iter_quant_targets,
    restore_channel_rows,
)

#: FP4 (e2m1) の値集合 — 符号 1 / 指数 2 / 仮数 1 bit（subnormal 込みで 0.5 刻みから 6.0 まで）。
#: 16 符号のうち ±0 が同値なので**相異なる値は 15**（i4 の ±7 が 15 準位なのと同じ）。
#: MUST: 昇順（丸めが隣接準位の中点による二分探索 — {@link round_to_levels}）。
FP4_E2M1_LEVELS: tuple[float, ...] = (
    -6.0,
    -4.0,
    -3.0,
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
)

#: NF4（4bit NormalFloat）の 16 準位。出典 = Dettmers et al., "QLoRA: Efficient Finetuning of
#: Quantized LLMs"（arXiv:2305.14314）§3 の NF4 と、その実装 bitsandbytes の
#: `create_normal_map()` が出す表。標準正規の分位点を ±1 へ正規化した非対称格子で、**0.0 と
#: ±1.0 を厳密に含む**（0 が表に無いと疎な重みが復元できず、±1 が無いと group の amax 要素が
#: 復元されない）。FP4 と違い正負で刻みが違う。
NF4_LEVELS: tuple[float, ...] = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)

#: MXFP4 の要素形式（= FP4 e2m1）の最大指数。OCP Microscaling Formats (MX) v1.0 の変換手順は
#: block の共有 scale を `2^(floor(log2(amax)) − emax_elem)` の 2 のべきで採る。e2m1 の最大値
#: 6.0 = 1.5×2² なので `emax_elem = 2`。
MXFP4_ELEMENT_EMAX = 2

#: k-means codebook の準位数（4bit = 16 centroid）。
DEFAULT_CODEBOOK_LEVELS = 16

#: Lloyd 反復の固定回数。**収束判定で止めない** — 判定の閾値が入力依存の任意性になるうえ、
#: 「反復が足りない」と「方式が弱い」が混ざる。1 次元・16 centroid では十分に落ち着く回数。
DEFAULT_CODEBOOK_ITERATIONS = 16

#: k-means の表の粒度。`shared` は**層をまたいで 1 枚**の表を、group(g32) の absmax で
#: 正規化した値空間に張る（表のコストが層数に比例しない代わりに scale が要る形）。
KMeansGranularity = Literal["per_tensor", "per_channel", "shared"]

KMEANS_GRANULARITIES: tuple[KMeansGranularity, ...] = ("per_tensor", "per_channel", "shared")


@dataclass(frozen=True)
class MethodReport:
    """丸めた方式と本数・要素数（「方式を指定したのに 0 本」を沈黙させないための計数）。

    scale / codebook の台帳を持たないのはモジュール docstring の MUST（emit へ渡す口を
    作らない）。層に当たった表そのものは丸め済み重みの `torch.unique` から引ける。
    """

    method: str
    modules: int
    elements: int

    def describe(self) -> str:
        return f"{self.method}: modules {self.modules} / {self.elements:,} elements"


def levels_tensor(levels: Sequence[float]) -> torch.Tensor:
    """準位表を F32 テンソルにする（昇順であることの検査つき）。

    表が昇順でないと {@link round_to_levels} の二分探索が黙って別の準位を返す（例外は
    出ず、値だけが最近傍でなくなる）ので、表を作る側の 1 度だけここで確かめる。
    """
    table = torch.tensor(levels, dtype=torch.float32)
    if table.dim() != 1 or table.numel() < 2:
        raise QuantizeError(
            f"準位表は 2 個以上の 1 次元でなければならない（実測 {list(table.shape)}）"
        )
    if not bool(torch.all(table[1:] > table[:-1])):
        raise QuantizeError("準位表が昇順でない（最近傍丸めが二分探索なので昇順 MUST）")
    return table


def round_to_levels(values: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """`values` の各要素を昇順の準位表 `levels` の**最近傍**へ丸める（形は保つ）。

    隣接準位の中点ちょうどは**下側**（−∞ 方向）へ落ちる（`torch.bucketize` の既定
    `right=False` = `boundaries[i-1] < x ≤ boundaries[i]`）。偶数丸めでも 0 から遠ざける丸めでも
    ないが、境界に厳密一致する値は実重みでは測度 0 に近く、方式間の比較で効く量ではない —
    ここで向きを固定するのは決定性のため（実測: MXFP4 で `amax/scale = 5.0` がこの枝を踏む）。
    """
    midpoints = (levels[1:] + levels[:-1]) / 2
    return levels[torch.bucketize(values, midpoints)]


def round_rows_to_codebook(values: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """行ごとに別の表を持つ版の {@link round_to_levels}（`values[R,N]` × `codebook[R,L]`）。

    per-channel の表を Python ループで当てると層あたり数千回の往復になるので、
    `searchsorted` の batched 形（境界も行ごと）で 1 発にする。
    """
    midpoints = (codebook[:, 1:] + codebook[:, :-1]) / 2
    index = torch.searchsorted(midpoints.contiguous(), values.contiguous())
    return torch.gather(codebook, 1, index)


def group_absmax_scale(
    rows: torch.Tensor, group_size: int, max_level: float, where: str = "重み"
) -> torch.Tensor:
    """`scale = clamp(amax_group / max_level, f32 tiny)` を group 形 `[チャネル, group 数]` で返す。

    `quantize.group_scale`（i4 の `amax/7`）の準位表版で、`max_level` は表の最大絶対値
    （FP4 = 6.0 / NF4 = 1.0）。group の amax 要素が表の両端へ乗るので、`quantize` 側と同じ
    理由で丸めが冪等になる。下限 clamp は全ゼロ group（`amax == 0`）の 0 除算避け。
    """
    grouped = grouped_view(rows, group_size, where)
    amax = grouped.abs().amax(dim=-1)
    return torch.clamp(amax / max_level, min=torch.finfo(torch.float32).tiny)


def group_power_of_two_scale(
    rows: torch.Tensor, group_size: int, where: str = "重み"
) -> torch.Tensor:
    """OCP MX 流の**2 のべき**の group scale = `2^(floor(log2(amax)) − MXFP4_ELEMENT_EMAX)`。

    指数は `frexp`（`x = m·2^e`, `m ∈ [0.5,1)` なので `floor(log2 x) = e − 1`）から採る —
    `log2` は 2 のべきちょうどで 1ulp 下振れしうるが、`frexp` は指数を厳密に返す。

    2 のべきに切り下げるので `amax/scale ∈ [4, 8)` に入り、表の外側（6 < |x| < 8）は
    {@link round_to_levels} が両端の 6.0 へ落とす（= MX の飽和）。absmax 版と違って amax 要素
    そのものは復元されない代わりに、scale の乗除算が**厳密**になる（丸めは真に冪等）。
    全ゼロ group は `frexp` が `e = 0` を返すので `scale = 2^-3` に落ち、値は 0 のまま。
    """
    grouped = grouped_view(rows, group_size, where)
    amax = grouped.abs().amax(dim=-1)
    _, exponent = torch.frexp(amax)
    shared = exponent - 1 - MXFP4_ELEMENT_EMAX
    return torch.ldexp(torch.ones_like(amax), shared)


def round_groups_to_levels(
    rows: torch.Tensor, levels: torch.Tensor, scale: torch.Tensor, where: str = "重み"
) -> torch.Tensor:
    """group ごとに `scale` で正規化して `levels` の最近傍へ丸め、`scale` を掛け戻す。

    MUST: group 長の源は**渡された scale の形**だけにする（`quantize.group_size_of` —
    別引数で受けると「丸めに使った group」と「scale を採った group」が独立に動ける。
    ADR 0069 決定 3 と同じ穴）。
    """
    grouped = grouped_view(rows, group_size_of(rows, scale), where)
    rounded = round_to_levels(grouped / scale.unsqueeze(-1), levels)
    return (rounded * scale.unsqueeze(-1)).reshape(rows.shape)


def _round_model_in_place(
    model: nn.Module,
    method: str,
    round_rows: Callable[[torch.Tensor, str], torch.Tensor],
    op_types: tuple[type[nn.Module], ...],
    include: Callable[[str], bool] | None,
) -> MethodReport:
    """全対象の重みを `round_rows(rows, where) -> rows` で丸めて in-place 書き戻す（共有の胴体）。

    書き戻しを {@link quantize.restore_channel_rows} 経由にするのは `ConvTranspose1d` のため
    — 平坦形が重みの view にならない（`movedim` で非連続 → `reshape` がコピー）ので、
    平坦形へ書いても元の重みは 1 要素も変わらない。

    「対象 0 本」は fail loudly（`quantize` 側と同じ常設診断 — 方式を指定したのに実質 f32 の
    ままだった、を沈黙させない）。
    """
    modules = 0
    elements = 0
    with torch.no_grad():
        for fqn, weight, axis in iter_quant_targets(model, op_types, include):
            rows = channel_rows(weight, axis)
            weight.copy_(restore_channel_rows(round_rows(rows, f"'{fqn}'"), weight, axis))
            modules += 1
            elements += weight.numel()
    if not modules:
        raise QuantizeError(
            f"{method}: 丸められる重みが 1 本も無い"
            f"（対象の型: {', '.join(cls.__name__ for cls in op_types)}）"
        )
    return MethodReport(method=method, modules=modules, elements=elements)


def fake_quant_fp4(
    model: nn.Module,
    group_size: int = DEFAULT_GROUP_SIZE,
    include: Callable[[str], bool] | None = None,
    op_types: tuple[type[nn.Module], ...] = (nn.Linear,),
) -> MethodReport:
    """FP4 (e2m1) の固定表 × group(既定 g32) absmax scale で丸める。

    i4 との違いは**格子の疎密**だけ（同じ 15 準位・同じ group scale で、i4 は等間隔・FP4 は
    指数的に疎）。0 付近が密なので、重み分布が正規に近いほど FP4 が有利になりうる、という
    仮説を測るための列。
    """
    table = levels_tensor(FP4_E2M1_LEVELS)
    top = float(table.abs().max())

    def rounder(rows: torch.Tensor, where: str) -> torch.Tensor:
        return round_groups_to_levels(
            rows, table, group_absmax_scale(rows, group_size, top, where), where
        )

    return _round_model_in_place(model, "fp4", rounder, op_types, include)


def fake_quant_nf4(
    model: nn.Module,
    group_size: int = DEFAULT_GROUP_SIZE,
    include: Callable[[str], bool] | None = None,
    op_types: tuple[type[nn.Module], ...] = (nn.Linear,),
) -> MethodReport:
    """NF4 の固定表（{@link NF4_LEVELS}）× group(既定 g32) absmax scale で丸める。

    表が「正規分布の分位点」なので、group 内の重みが正規に近ければ準位の利用が均等になる
    （= 情報理論的に最適に近い）という主張の測定列。非対称なのは**内側の準位だけ**で、
    両端は ±1.0 を厳密に含む（{@link NF4_LEVELS}）ため group の amax 要素は正負とも
    復元される。
    """
    table = levels_tensor(NF4_LEVELS)
    top = float(table.abs().max())

    def rounder(rows: torch.Tensor, where: str) -> torch.Tensor:
        return round_groups_to_levels(
            rows, table, group_absmax_scale(rows, group_size, top, where), where
        )

    return _round_model_in_place(model, "nf4", rounder, op_types, include)


def fake_quant_mxfp4(
    model: nn.Module,
    group_size: int = DEFAULT_GROUP_SIZE,
    include: Callable[[str], bool] | None = None,
    op_types: tuple[type[nn.Module], ...] = (nn.Linear,),
) -> MethodReport:
    """MXFP4（FP4 表 × block の**2 のべき** scale）で丸める — OCP Microscaling の流儀。

    absmax 版（{@link fake_quant_fp4}）との差は scale の表現だけで、scale が E8M0（指数 1
    バイト）に載るぶん格納が軽い代わりに、group の amax が 2 のべきへ切り下げられて最大 1
    段ぶんの余白が捨てられる。既定の block 長は本リポの既定 group（32）で、MX 仕様の
    block = 32 と一致する。
    """
    table = levels_tensor(FP4_E2M1_LEVELS)

    def rounder(rows: torch.Tensor, where: str) -> torch.Tensor:
        return round_groups_to_levels(
            rows, table, group_power_of_two_scale(rows, group_size, where), where
        )

    return _round_model_in_place(model, "mxfp4", rounder, op_types, include)


def fit_codebook(values: torch.Tensor, levels: int, iterations: int) -> torch.Tensor:
    """1 次元 k-means（Lloyd 法）を行ごとに当て、昇順の centroid 表 `[R, levels]` を返す。

    MUST: **決定的**（同一入力 → ビット同一の表）。初期化は乱数播種ではなく**分位点**
    （ソート済み値の `(i+0.5)/levels` 位置）で、空クラスタは centroid を据え置く
    （再播種すると「どこへ播くか」が新しい任意性になる）。反復は固定回数
    （{@link DEFAULT_CODEBOOK_ITERATIONS}）。

    1 次元なので最近傍は「中点による二分探索」で厳密に求まる（centroid 総当たりの距離計算は
    要らない）。centroid の更新和は **f64** で積む — f32 の逐次加算だと 1 クラスタが数百万
    要素になる層で桁落ちが効き、平均が入力の並び順に依存しはじめる。
    """
    if values.dim() != 2:
        raise QuantizeError(f"k-means の入力は `[R, N]` の 2 次元（実測 {list(values.shape)}）")
    if values.shape[1] < levels:
        raise QuantizeError(
            f"k-means の準位数 {levels} に対して要素が {values.shape[1]} 個しか無い"
            "（分位点初期化が同じ値の centroid を並べてしまう）"
        )
    sorted_values, _ = torch.sort(values, dim=-1)
    count = sorted_values.shape[1]
    position = ((torch.arange(levels, dtype=torch.float64) + 0.5) / levels * count).long()
    centroids = sorted_values.index_select(1, position.clamp(max=count - 1))
    wide = values.to(torch.float64)
    ones = torch.ones_like(wide)
    for _ in range(iterations):
        index = torch.searchsorted(
            ((centroids[:, 1:] + centroids[:, :-1]) / 2).contiguous(), values.contiguous()
        )
        totals = torch.zeros(centroids.shape, dtype=torch.float64).scatter_add_(1, index, wide)
        counts = torch.zeros(centroids.shape, dtype=torch.float64).scatter_add_(1, index, ones)
        means = (totals / counts.clamp(min=1.0)).to(torch.float32)
        centroids = torch.where(counts.to(torch.float32) > 0, means, centroids)
        # 空クラスタを据え置くと（据え置いた値が隣の平均を追い越して）順序が崩れうる。
        # 崩れた表は二分探索の前提を壊すので、毎反復で並べ直す（16 要素なので安い）。
        centroids, _ = torch.sort(centroids, dim=-1)
    return centroids


def fake_quant_kmeans(
    model: nn.Module,
    granularity: KMeansGranularity,
    group_size: int = DEFAULT_GROUP_SIZE,
    include: Callable[[str], bool] | None = None,
    op_types: tuple[type[nn.Module], ...] = (nn.Linear,),
    levels: int = DEFAULT_CODEBOOK_LEVELS,
    iterations: int = DEFAULT_CODEBOOK_ITERATIONS,
    fit_stride: int = 1,
) -> MethodReport:
    """16 centroid の k-means codebook で丸める（表の粒度 3 種）。

    - `per_tensor` … 層ごとに 1 枚。表のコストは層あたり `levels` 個で、scale は要らない
    - `per_channel` … 出力チャネル（{@link quantize.channel_rows} の行）ごとに 1 枚。
      表のコストはチャネル数に比例するが、チャネル間のスケール差を表が吸収する
    - `shared` … **層をまたいで 1 枚**。group(既定 g32) absmax で正規化した値空間に張るので、
      スケール差は scale が、形は共有表が担う（= 表のコストがモデル全体で `levels` 個）

    `group_size` が効くのは `shared` だけ（他の 2 つに scale は無い）。

    MUST: 同一入力 → ビット同一出力（{@link fit_codebook} が決定的・seed に依存しない）。

    NOTE: `shared` は表を当てる前に**全対象の正規化値を 1 本へ連結**するので、対象の重みと
    同サイズの f32 一時領域（+ {@link fit_codebook} の f64 作業領域）が要り、1B 級では実
    メモリを超える。`fit_stride` はその逃げ道 — **`shared` の表の fit だけ**を等間隔
    `flat[::fit_stride]`（モジュールごと）の部分標本で採る（**適用は常に全量**）。乱数を
    使わないので決定性 MUST はそのまま成り立つ。MUST: 使ったら測定側の出力へ明記する —
    部分標本の表と全量の表は別物になりうるので、数値を読む側が区別できること。
    """
    if granularity not in KMEANS_GRANULARITIES:
        raise QuantizeError(
            f"k-means の粒度 '{granularity}' は未対応（{', '.join(KMEANS_GRANULARITIES)}）"
        )
    if fit_stride < 1:
        raise QuantizeError(f"fit_stride は 1 以上（実測 {fit_stride}）")
    if fit_stride != 1 and granularity != "shared":
        raise QuantizeError(
            f"fit_stride は shared 専用（'{granularity}' の fit は対象と同サイズの連結を"
            "作らないので、標本化する理由が無い — 黙って無視もしない）"
        )
    method = f"kmeans:{granularity}"
    if granularity != "shared":
        per_channel = granularity == "per_channel"

        def rounder(rows: torch.Tensor, _where: str) -> torch.Tensor:
            flat = rows if per_channel else rows.reshape(1, -1)
            codebook = fit_codebook(flat, levels, iterations)
            return round_rows_to_codebook(flat, codebook).reshape(rows.shape)

        return _round_model_in_place(model, method, rounder, op_types, include)

    # `shared` は 2 パス — 1 パス目で全対象の正規化値を集めて表を 1 枚張り、2 パス目で当てる。
    # 対象選択は両パスとも同じ `iter_quant_targets`（1 パス目だけ広い、が起きない形）。
    normalized: list[torch.Tensor] = []
    with torch.no_grad():
        for fqn, weight, axis in iter_quant_targets(model, op_types, include):
            rows = channel_rows(weight, axis)
            scale = group_absmax_scale(rows, group_size, 1.0, f"'{fqn}'")
            grouped = grouped_view(rows, group_size_of(rows, scale), f"'{fqn}'")
            flat = (grouped / scale.unsqueeze(-1)).reshape(-1)
            # 標本のみ保持する場合は clone で母体を切り離す（view のままだと全量が生き残り、
            # fit_stride の目的である一時領域の削減が黙って無効になる）。
            normalized.append(flat[::fit_stride].clone() if fit_stride > 1 else flat)
    if not normalized:
        raise QuantizeError(
            f"{method}: 丸められる重みが 1 本も無い"
            f"（対象の型: {', '.join(cls.__name__ for cls in op_types)}）"
        )
    codebook = fit_codebook(torch.cat(normalized).reshape(1, -1), levels, iterations)
    table = codebook.reshape(-1)

    def apply_shared(rows: torch.Tensor, where: str) -> torch.Tensor:
        return round_groups_to_levels(
            rows, table, group_absmax_scale(rows, group_size, 1.0, where), where
        )

    return _round_model_in_place(model, method, apply_shared, op_types, include)
