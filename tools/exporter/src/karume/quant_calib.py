"""**校正データを使う丸め**（GPTQ / AWQ）— `quantize` の格納グリッドに載る第 2 段。

`quantize.fake_quant_int4` は各重みを単独で（= 入力の分布を見ずに）最近傍へ丸める。ここに
載るのは「その層に実際に流れる活性」を校正データから見て、**同じ格納グリッドの中で**丸め先を
選び直す方式で、格子そのものは 1 段目と 1 バイトも変えない。狙いは「格納形を変えずに品質を
どこまで戻せるか」。

- **GPTQ**（`method="gptq"`）… 列を 1 本ずつ丸め、丸め誤差を Hessian `H = Σ XᵀX` の逆行列で
  **未処理の列へ配り直す**。格納グリッドが `rtn` のときは**出荷可能** — 格子は
  {@link quantize.quantize_to_int4} と厳密に同じ（`scale = group amax / 7`・`q ∈ [−7,+7]`・
  −8 不使用）で、scale 台帳は {@link quantize.Int4Report} の器でそのまま emit へ渡せる。
- **AWQ**（`method="awq"`）… per-in-channel の等価倍率 `s` を活性の大きさから決め、
  `W' = W·diag(s)` を丸めて `W_eff = Q(W')/s` を書き戻す **fake-quant**。fold の理想形の
  上限を測るための列で、**そのままでは格納できない**（下の MUST）。
- **併用**（`method="awq+gptq"`）… `s` を決めてから `W'` に GPTQ を掛ける。実効入力が `x/s` に
  なるので Hessian も `H' = diag(1/s)·H·diag(1/s)` へ変換して使う。

MUST: **emit へ流せるのは `method="gptq"` × `grid="rtn"` だけ**。これは意図的な設計で、
他の組はレポートに scale 台帳を持たない:

- `nf4` / `kmeans_shared` グリッドは `quant_methods` と同じ理由で台帳を返さない（格納経路と
  ランタイム実装が無い方式へ「emit へ渡す口」を作らない — `quant_methods` モジュール
  docstring）。
- AWQ は `W_eff = Q(W')/s` を書き戻すが、格納したいのは `Q(W')` と `s` の組で、`W_eff` を
  そのまま group absmax で量子化し直すと **`s` は group 内一定の定数として absmax scale と
  相殺され、RTN と同値に戻る**（`s` が担っていた情報が消える）。出荷するには `s` を
  隣接演算へ **fold** するか、`s` を **companion テンソル**として格納する実装が要る。

MUST: **同一入力 → ビット同一出力**。乱数は 1 つも使わない。Hessian の蓄積と Cholesky
solver は **f64**（`quant_methods.fit_codebook` の f64 精算と同じ理由 — 数百万要素の逐次和は
f32 だと並び順に依存しはじめる）。一方で**グリッドの scale 決定と丸めそのものは f32** で
行う — f64 で割ってから f32 へ落とすと二重丸めで `scale` が 1ulp 動きうるので、
{@link quantize.fake_quant_int4} との**ビット同一**（下のオラクル）が成立しなくなる。

MUST: 丸めは**参照・golden の採取より前**（`quantize` モジュール docstring と同文）。

NOTE（オラクル）: `H` が対角（例 `λI`）のとき補償項 `U[j, j+1:]` が厳密に 0 になるので、
`gptq` × `rtn` は {@link quantize.fake_quant_int4} と**ビット同一**になる。これは実装の性質
ではなく数学的帰結で、`tests/test_quant_calib.py` が門として固定する。

NOTE（act-order を実装しない理由）: 参照実装の GPTQ は `diag(H)` の降順で in 軸を並べ替えて
から丸める（act-order / desc_act）と品質が上がるが、本リポの格納は **in 軸の連続 32 要素を
1 group** として scale を張る（ADR 0069 決定 2・端数 group 禁止）ので、並べ替えると格納の
group 整列が壊れる。復元するには「並べ替え順」を companion テンソルとして配る実装が要り、
それ無しでは**出荷できない**。測定だけできても出口が無い列は作らない。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn

from .quant_methods import (
    DEFAULT_CODEBOOK_ITERATIONS,
    DEFAULT_CODEBOOK_LEVELS,
    NF4_LEVELS,
    fit_codebook,
    group_absmax_scale,
    levels_tensor,
    round_groups_to_levels,
    round_to_levels,
)
from .quantize import (
    DEFAULT_GROUP_SIZE,
    INT4_MAX,
    Int4Report,
    QuantizeError,
    channel_rows,
    dequantize_int4,
    group_scale,
    grouped_view,
    iter_quant_targets,
    quantize_to_int4,
    restore_channel_rows,
)

#: GPTQ の列ブロック長（既定）。ブロックは「補償をまとめて当てる遅延幅」でしかなく、
#: 数学的な結果はブロック長に依らない（浮動小数の加算順だけが変わる）。
DEFAULT_GPTQ_BLOCK = 128

#: damping 係数 — `λ = GPTQ_DAMPING × mean(diag(H))` を対角へ加える。**定数固定**にするのは、
#: 可変にすると「方式の差」と「damping の差」が分離できなくなるから（`quant_methods` の
#: Lloyd 反復を固定回数にしてあるのと同じ理由）。
GPTQ_DAMPING = 0.01

#: AWQ の α 格子の分割数（`α ∈ {0, 1/16, …, 1}` の 17 点）。
AWQ_ALPHA_STEPS = 16

#: AWQ の `act_amax` 下限。0 のチャネル（校正データで 1 度も動かなかった入力）を放置すると
#: `s_j = 0` になり、幾何平均 1 への正規化が壊れる（`log 0`）。
AWQ_ACT_EPS = 1e-5

#: AWQ の目的関数に使う入力サンプルの最大行数（**先頭から**採る決定的な間引き）。
DEFAULT_AWQ_SAMPLES = 1024

#: 格納グリッドの種類。`rtn` だけが出荷経路を持つ（モジュール docstring の MUST）。
GridKind = Literal["rtn", "nf4", "kmeans_shared"]

GRID_KINDS: tuple[GridKind, ...] = ("rtn", "nf4", "kmeans_shared")

#: 校正付き丸めの方式。
CalibMethod = Literal["gptq", "awq", "awq+gptq"]

CALIB_METHODS: tuple[CalibMethod, ...] = ("gptq", "awq", "awq+gptq")

#: 先頭 stage へ渡す 1 バッチ（位置引数と keyword 引数の組）。
StageBatch = tuple[tuple[object, ...], Mapping[str, object]]

#: 実行順の stage 1 つ（**モデル内 FQN の接頭辞**とモジュールの組）。接頭辞を受けるのは、
#: scale 台帳のキーを {@link quantize.Int4Report} と同じ**モデル内 FQN** の空間に揃えるため
#: （stage 単体で `named_modules` を舐めると `0.weight` のような局所名しか採れない）。
StageSpec = tuple[str, nn.Module]


@dataclass(frozen=True)
class GridSpec:
    """丸め先の格納グリッド（`quantize` / `quant_methods` の格子をそのまま指す指定）。

    - `rtn` … {@link quantize.quantize_to_int4} の格子（group amax / 7・`q ∈ [−7,+7]`）。
      **唯一の出荷可能グリッド**で、scale 台帳を返す
    - `nf4` … {@link quant_methods.NF4_LEVELS} の固定表 × group absmax scale
    - `kmeans_shared` … **層ごとに 1 枚**の k-means 表 × group absmax scale。表は
      {@link quant_methods.fit_codebook} を**元の重み**（丸め前）へ当てて fit する

    NOTE: `kmeans_shared` の "shared" は {@link quant_methods.fake_quant_kmeans} の同名粒度と
    同じ意味論（group absmax で正規化した値空間に 1 枚の表を張る）だが、表の**射程は層内**に
    閉じている — GPTQ は stage を 1 つずつ進める形なので、全層を跨ぐ表を張るには量子化の前に
    全対象を舐める別パスが要り、stage 逐次という設計の要（H を 1 stage 分しか持たない）と
    噛み合わない。

    `fit_stride` は `kmeans_shared` の表を `flat[::fit_stride]` の部分標本から fit する逃げ道
    （適用は常に全量・乱数を使わないので決定性 MUST は保つ）。MUST: 使ったら測定側の出力へ
    明記する — {@link CalibReport.describe} が拾う。
    """

    kind: GridKind = "rtn"
    group_size: int = DEFAULT_GROUP_SIZE
    levels: int = DEFAULT_CODEBOOK_LEVELS
    iterations: int = DEFAULT_CODEBOOK_ITERATIONS
    fit_stride: int = 1

    def __post_init__(self) -> None:
        if self.kind not in GRID_KINDS:
            raise QuantizeError(f"格納グリッド '{self.kind}' は未対応（{', '.join(GRID_KINDS)}）")
        if self.group_size < 1:
            raise QuantizeError(f"group_size は 1 以上（実測 {self.group_size}）")
        if self.levels < 2:
            raise QuantizeError(f"levels は 2 以上（実測 {self.levels}）")
        if self.iterations < 1:
            raise QuantizeError(f"iterations は 1 以上（実測 {self.iterations}）")
        if self.fit_stride < 1:
            raise QuantizeError(f"fit_stride は 1 以上（実測 {self.fit_stride}）")
        if self.fit_stride != 1 and self.kind != "kmeans_shared":
            raise QuantizeError(
                f"fit_stride は kmeans_shared 専用（'{self.kind}' に fit する表は無い"
                " — 黙って無視もしない）"
            )

    @property
    def shippable(self) -> bool:
        """このグリッドの scale 台帳を emit へ渡せるか（モジュール docstring の MUST）。"""
        return self.kind == "rtn"


#: 既定のグリッド（i4 g32 の RTN 格子 = 唯一の出荷経路）。引数既定を**呼び出し式にしない**
#: ために名前で持つ（毎回作っても同値な frozen dataclass なので共有して差し支えない）。
DEFAULT_GRID = GridSpec()


@dataclass(frozen=True)
class _BoundGrid:
    """1 つの層へ束ねたグリッド（`kmeans_shared` の表だけが層ごとに変わる）。

    MUST: scale 決定も丸めも **f32** で閉じる（モジュール docstring — f64 経由の二重丸めで
    `scale` が 1ulp 動くと `fake_quant_int4` とのビット同一が崩れる）。
    """

    spec: GridSpec
    where: str
    #: 準位表（`rtn` は表を持たず {@link quantize.quantize_to_int4} の格子を直に使う）。
    table: torch.Tensor | None
    #: 表の最大絶対値（group absmax 正規化の分母）。
    max_level: float

    def scale_of(self, block: torch.Tensor) -> torch.Tensor:
        """1 group ぶんの列（`[チャネル, group_size]` F32）の**現在値**から scale を決める。

        GPTQ は列を進むたびに残りの列を書き換えるので、group scale は「その group の列に
        **到達した時点**の値」から採る。丸める前の元の重みから採ると、補償で動いた値が
        scale の担当範囲から外れうる。
        """
        if self.table is None:
            return group_scale(block, self.spec.group_size, self.where).squeeze(-1)
        return group_absmax_scale(block, self.spec.group_size, self.max_level, self.where).squeeze(
            -1
        )

    def round_column(self, values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """1 列（`[チャネル]` F32）を `scale`（`[チャネル]` F32）のグリッドへ丸める。

        `rtn` は {@link quantize.quantize_to_int4} / {@link quantize.dequantize_int4} を
        **そのまま**通す（列を `[チャネル, 1]` の 1 要素 group とみなす形）— 格納の式と
        丸めの式を別実装にしないための遠回りで、要素ごとの演算は層一括版と同一。
        """
        if self.table is None:
            column = values.unsqueeze(-1)
            group = scale.unsqueeze(-1)
            return dequantize_int4(quantize_to_int4(column, group), group).squeeze(-1)
        return round_to_levels(values / scale, self.table) * scale

    def round_rows(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """層全体を素で丸める（GPTQ を通さない fake-quant）— AWQ の α 探索が使う。

        `rtn` の経路は {@link quantize.fake_quant_int4} の 1 層ぶんと**ビット同一**。
        """
        if self.table is None:
            scale = group_scale(rows, self.spec.group_size, self.where)
            return dequantize_int4(quantize_to_int4(rows, scale), scale), scale
        scale = group_absmax_scale(rows, self.spec.group_size, self.max_level, self.where)
        return round_groups_to_levels(rows, self.table, scale, self.where), None


def _bind_grid(spec: GridSpec, fit_source: torch.Tensor, where: str) -> _BoundGrid:
    """グリッド指定を 1 層へ束ねる（`fit_source` は `kmeans_shared` の表を採る元の重み）。

    GPTQ では `fit_source` に**丸める前**の重みを渡す — 列の補償で動いた値から表を採り直すと
    表がブロックの進み方に依存してしまう。AWQ の α 探索では `W' = W·diag(s)` を渡す
    （実際に丸められるのが `W'` なので、表もそこから採るのが素直な fake-quant）。
    """
    if spec.kind == "rtn":
        return _BoundGrid(spec=spec, where=where, table=None, max_level=float(INT4_MAX))
    if spec.kind == "nf4":
        table = levels_tensor(NF4_LEVELS)
        return _BoundGrid(spec=spec, where=where, table=table, max_level=float(table.abs().max()))
    scale = group_absmax_scale(fit_source, spec.group_size, 1.0, where)
    grouped = grouped_view(fit_source, spec.group_size, where)
    flat = (grouped / scale.unsqueeze(-1)).reshape(1, -1)
    sampled = flat[:, :: spec.fit_stride] if spec.fit_stride > 1 else flat
    table = fit_codebook(sampled, spec.levels, spec.iterations).reshape(-1)
    return _BoundGrid(spec=spec, where=where, table=table, max_level=1.0)


# ---- GPTQ（層単位）----------------------------------------------------------


def _inverse_cholesky(hessian: torch.Tensor, where: str) -> torch.Tensor:
    """damping 済み `H` の逆行列の上三角 Cholesky 因子 `U`（`H⁻¹ = Uᵀ U`）を F64 で返す。

    `λ = GPTQ_DAMPING × mean(diag(H))` を対角へ加えるので、校正データで 1 度も動かなかった
    入力チャネル（対角 0）が混ざっていても正定値性は保たれる。**H が丸ごと 0** のときだけ
    `λ = 0` で特異になるので fail loudly — 「校正 forward に載っていない層を量子化した」を
    沈黙させない。
    """
    wide = hessian.to(torch.float64)
    average = float(torch.diagonal(wide).mean())
    # NaN も落とすので `<= 0` ではなく `not > 0` で書く（NaN は全比較が偽になる）。
    if not average > 0.0:
        raise QuantizeError(
            f"{where}: Hessian の対角平均が {average:.4g}（正でない）— 校正入力が 1 行も"
            "流れていないか全ゼロで、damping λ = 0 では H が特異になる"
        )
    damped = wide.clone()
    damped.diagonal().add_(GPTQ_DAMPING * average)
    try:
        factor = torch.linalg.cholesky(damped)
        return torch.linalg.cholesky(torch.cholesky_inverse(factor), upper=True)
    except RuntimeError as error:  # torch の LinAlgError は RuntimeError の派生
        raise QuantizeError(f"{where}: Hessian の Cholesky 分解に失敗した（{error}）") from error


def gptq_quantize_rows(
    rows: torch.Tensor,
    hessian: torch.Tensor,
    spec: GridSpec = DEFAULT_GRID,
    where: str = "重み",
    block_size: int = DEFAULT_GPTQ_BLOCK,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """`[チャネル, in]` に畳んだ重みを GPTQ で丸め、`(丸め済み rows, scale 台帳)` を返す。

    `hessian` は `H = Σ XᵀX`（`X` = この層への入力 `[tokens, in]`）の `[in, in]`。台帳は
    `spec.kind == "rtn"` のときだけ `[チャネル, group 数]` の F32 で、それ以外は `None`
    （モジュール docstring の MUST）。

    手順は標準の列ブロック法:

    1. `U = cholesky(H⁻¹, upper)` を採る（`H` は damping 済み — {@link _inverse_cholesky}）
    2. 列 `j` を**現在値**からグリッドへ丸め、`err = (w_j − dq_j) / U[j,j]` を作る
    3. ブロック内の残り列を `W[:, j+1:] −= err ⊗ U[j, j+1:]` で補償する
    4. ブロック終端で、ブロック外の残り全列を `Err @ U[block, 残り]` で一括更新する

    MUST: 台帳は**量子化に使った scale をそのまま**返す（`quantize_to_int8` /
    `quantize_to_int4` と同文の再計算禁止だが、GPTQ ではさらに強い意味を持つ）— RTN では
    group の amax 要素が必ず `|q| = 7` に乗るので引き直しても同じ scale が出るが、GPTQ は
    補償で amax 要素を押し下げうる。押し下がった group では `|q|` が 7 に届かず、丸め済みの
    重みから引き直すと 1ulp どころか**別の scale** になって格納値が復元されなくなる
    （実測: 全 group で起きるわけではないので、テストは条件を満たす group を含む模型で門を
    張っている — `tests/test_quant_calib.py`）。

    NOTE: `block_size` は `group_size` の倍数へ丸められる（group が 2 つのブロックへ跨ると
    「group の列に到達した時点の現在値」が採れなくなるため）。結果は数学的にはブロック長に
    依らない。
    """
    if rows.dim() != 2:
        raise QuantizeError(
            f"{where}: GPTQ の入力は `[チャネル, in]` の 2 次元（実測 {list(rows.shape)}）"
        )
    in_axis = int(rows.shape[1])
    if hessian.dim() != 2 or list(hessian.shape) != [in_axis, in_axis]:
        raise QuantizeError(
            f"{where}: Hessian は `[in, in]` = [{in_axis}, {in_axis}]（実測 {list(hessian.shape)}）"
        )
    # 端数 group の fail loudly は格納側と同じ 1 本（ADR 0069 決定 2）。
    groups = int(grouped_view(rows, spec.group_size, where).shape[-2])
    grid = _bind_grid(spec, rows, where)
    upper = _inverse_cholesky(hessian, where)
    block = spec.group_size * max(1, block_size // spec.group_size)

    work = rows.to(torch.float64)
    rounded = torch.empty(rows.shape, dtype=torch.float32)
    ledger = (
        torch.empty((int(rows.shape[0]), groups), dtype=torch.float32) if spec.shippable else None
    )
    for start in range(0, in_axis, block):
        stop = min(start + block, in_axis)
        span = work[:, start:stop].clone()
        residuals = torch.zeros_like(span)
        for head in range(0, stop - start, spec.group_size):
            scale = grid.scale_of(span[:, head : head + spec.group_size].to(torch.float32))
            if ledger is not None:
                ledger[:, (start + head) // spec.group_size] = scale
            for offset in range(head, head + spec.group_size):
                column = start + offset
                quantized = grid.round_column(span[:, offset].to(torch.float32), scale)
                rounded[:, column] = quantized
                residual = (span[:, offset] - quantized.to(torch.float64)) / upper[column, column]
                residuals[:, offset] = residual
                tail = upper[column, column + 1 : stop]
                if tail.numel():
                    span[:, offset + 1 :] -= residual.unsqueeze(1) * tail
        if stop < in_axis:
            work[:, stop:] -= residuals @ upper[start:stop, stop:]
    return rounded, ledger


# ---- AWQ（層単位）----------------------------------------------------------


@dataclass(frozen=True)
class AwqSearch:
    """AWQ の α 探索の結果（層ごと）。"""

    #: 採用した α（`{0, 1/16, …, 1}` の 17 点から）。
    alpha: float
    #: その α での目的関数値 `‖X·W_effᵀ − X·Wᵀ‖²`（F64 で積んだ値）。
    error: float
    #: per-in-channel の等価倍率 `s`（幾何平均 1・`[in]` F64）。
    channel_scale: torch.Tensor


def awq_search_scale(
    rows: torch.Tensor,
    act_amax: torch.Tensor,
    samples: torch.Tensor,
    spec: GridSpec = DEFAULT_GRID,
    where: str = "重み",
) -> AwqSearch:
    """活性の大きさから per-in-channel の等価倍率 `s` を探す（AWQ の α 掃引）。

    `s_j = clamp(act_amax_j, AWQ_ACT_EPS)^α` を**幾何平均 1** へ正規化し、
    `W' = W·diag(s)` を `spec` のグリッドで fake-quant して `W_eff = Q(W')/s` を作る。
    目的関数は捕捉した入力サンプル `X` に対する `‖X·W_effᵀ − X·Wᵀ‖²`（F64）で、最小の α を
    採る。同値は**小さい α** が勝つ（決定性 MUST — `<` の狭義比較）。

    幾何平均 1 の正規化が要るのは、`s` 全体の定数倍が group absmax scale に吸収されて
    目的関数に効かないから（正規化しないと α の意味が「向き」だけになる）。

    MUST: 返る `s` は**格納の一部**であって、`W_eff` を単独で格納しても復元されない
    （モジュール docstring — group 内一定の `s` は absmax scale と相殺して RTN と同値に戻る）。
    """
    if rows.dim() != 2:
        raise QuantizeError(
            f"{where}: AWQ の入力は `[チャネル, in]` の 2 次元（実測 {list(rows.shape)}）"
        )
    in_axis = int(rows.shape[1])
    if act_amax.dim() != 1 or int(act_amax.shape[0]) != in_axis:
        raise QuantizeError(
            f"{where}: act_amax は `[in]` = [{in_axis}]（実測 {list(act_amax.shape)}）"
        )
    if samples.dim() != 2 or int(samples.shape[1]) != in_axis:
        raise QuantizeError(
            f"{where}: 入力サンプルは `[tokens, in]`（in = {in_axis}・実測 {list(samples.shape)}）"
        )
    if not int(samples.shape[0]):
        raise QuantizeError(f"{where}: 入力サンプルが 0 行 — α の目的関数が全て 0 になる")
    grouped_view(rows, spec.group_size, where)  # 端数 group を α ループの前に落とす

    amax = torch.clamp(act_amax.to(torch.float64), min=AWQ_ACT_EPS)
    inputs = samples.to(torch.float64)
    wide = rows.to(torch.float64)
    reference = inputs @ wide.T
    best_alpha = 0.0
    best_error = math.inf
    best_channel = torch.ones(in_axis, dtype=torch.float64)
    for step in range(AWQ_ALPHA_STEPS + 1):
        alpha = step / AWQ_ALPHA_STEPS
        channel = amax.pow(alpha)
        channel = channel / torch.exp(torch.log(channel).mean())
        scaled = (wide * channel).to(torch.float32)
        quantized, _ = _bind_grid(spec, scaled, where).round_rows(scaled)
        effective = (quantized.to(torch.float64) / channel).to(torch.float32)
        error = float((inputs @ effective.to(torch.float64).T - reference).pow(2).sum())
        if error < best_error:
            best_alpha, best_error, best_channel = alpha, error, channel
    return AwqSearch(alpha=best_alpha, error=best_error, channel_scale=best_channel)


# ---- stage 逐次の駆動 -------------------------------------------------------


@dataclass(frozen=True)
class LayerCalibReport:
    """1 層ぶんの校正結果（数値の読み手が層を特定できる粒度）。"""

    #: モデル内 FQN（`<module>.weight`）。
    fqn: str
    #: 何番目の stage で処理したか（0 始まり）。
    stage: int
    #: この層の統計に積んだ入力の行数（tokens）。
    tokens: int
    #: AWQ が採った α（GPTQ 単独では `None`）。
    alpha: float | None
    #: AWQ の目的関数値（GPTQ 単独では `None`）。
    error: float | None


@dataclass(frozen=True)
class CalibReport:
    """校正付き丸めの計数と台帳（`Int8Report` / `Int4Report` と同じ流儀の器）。"""

    method: str
    grid: str
    group_size: int
    stages: int
    elements: int
    fit_stride: int
    layers: tuple[LayerCalibReport, ...]
    #: 出荷できる scale 台帳 — `method="gptq"` × `grid="rtn"` のときだけ非 `None`。
    int4: Int4Report | None

    @property
    def modules(self) -> int:
        return len(self.layers)

    def describe(self) -> str:
        text = (
            f"{self.method}/{self.grid}: stages {self.stages} / modules {self.modules}"
            f" / {self.elements:,} elements / group {self.group_size}"
        )
        if self.fit_stride > 1:
            text += f" / fit_stride {self.fit_stride}（表は部分標本で fit）"
        alphas = [layer.alpha for layer in self.layers if layer.alpha is not None]
        if alphas:
            text += f" / AWQ α {min(alphas):.4g}〜{max(alphas):.4g}"
        text += (
            " / scale 台帳あり（emit へ渡せる）"
            if self.int4 is not None
            else " / scale 台帳なし（格納には fold か companion が要る）"
        )
        return text


@dataclass
class _LayerStats:
    """1 層ぶんの校正統計（H は 1 stage 分しか同時に持たない — メモリ設計の要）。"""

    in_features: int
    tokens: int = 0
    hessian: torch.Tensor | None = None
    act_amax: torch.Tensor | None = None
    samples: list[torch.Tensor] = field(default_factory=list)
    sampled: int = 0


def first_tensor_output(output: object) -> torch.Tensor:
    """stage 出力から次 stage の入力を選ぶ既定（Tensor はそのまま・tuple / list は先頭）。

    transformer ブロックは `(hidden, present_kv, …)` の tuple を返すことが多いので既定を
    先頭要素にする。別の位置を使う stage は `select_output` で差し替える。
    """
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple | list) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise QuantizeError(
        f"stage の出力（{type(output).__name__}）から次 stage の入力を選べない"
        "（`select_output` で選び方を渡す）"
    )


def _owner_name(fqn: str) -> str:
    """`iter_quant_targets` の FQN（`<module>.weight`）から所有モジュール名を戻す。"""
    return "" if fqn == "weight" else fqn[: -len(".weight")]


def _qualify(prefix: str, local: str) -> str:
    """stage 内の局所 FQN をモデル内 FQN へ持ち上げる。"""
    return f"{prefix}.{local}" if prefix else local


def _make_pre_hook(
    stats: _LayerStats, needs_hessian: bool, needs_act: bool, sample_limit: int
) -> Callable[[nn.Module, tuple[object, ...]], None]:
    """入力を素通ししつつ `H` / 活性統計 / 入力サンプルを積む forward_pre_hook。

    活性を書き換えないのが `act_quant.attach_act_quant` との違い — こちらは観測だけで、
    校正 forward が出す数は元のモデルと 1bit も変わらない。
    """

    def pre(_module: nn.Module, args: tuple[object, ...]) -> None:
        if not args or not isinstance(args[0], torch.Tensor):
            raise QuantizeError(
                "校正対象の linear が位置引数の Tensor で呼ばれていない"
                "（入力を観測できないので H が採れない）"
            )
        given = args[0].detach()
        if int(given.shape[-1]) != stats.in_features:
            raise QuantizeError(
                f"校正入力の最終次元 {int(given.shape[-1])} が in_features "
                f"{stats.in_features} と違う"
            )
        plain = given.reshape(-1, stats.in_features)
        wide = plain.to(torch.float64)
        stats.tokens += int(plain.shape[0])
        if needs_hessian:
            gram = wide.T @ wide
            stats.hessian = gram if stats.hessian is None else stats.hessian + gram
        if needs_act:
            amax = wide.abs().amax(dim=0)
            stats.act_amax = amax if stats.act_amax is None else torch.maximum(stats.act_amax, amax)
            room = sample_limit - stats.sampled
            if room > 0:
                taken = plain[:room].to(torch.float32).clone()
                stats.samples.append(taken)
                stats.sampled += int(taken.shape[0])
        return None

    return pre


def calibrate_stages(
    stages: Sequence[StageSpec],
    batches: Sequence[StageBatch],
    method: CalibMethod = "gptq",
    spec: GridSpec = DEFAULT_GRID,
    include: Callable[[str], bool] | None = None,
    block_size: int = DEFAULT_GPTQ_BLOCK,
    samples: int = DEFAULT_AWQ_SAMPLES,
    select_output: Callable[[object], torch.Tensor] = first_tensor_output,
    advance_kwargs: Callable[[int, Mapping[str, object]], Mapping[str, object]] | None = None,
) -> CalibReport:
    """stage を 1 つずつ進めながら校正付き丸めを当てる（**校正データを 2 周しない**駆動）。

    `stages` は実行順の `(モデル内 FQN 接頭辞, モジュール)` の列、`batches` は**先頭 stage**
    への `(位置引数, keyword 引数)` の列。stage `i` について:

    1. 対象 linear へ観測フックを掛け、全バッチを forward して `H` / 活性統計を積む
    2. その stage の対象を量子化する（`H` はここで捨てる — **同時に 1 stage 分**しか持たない）
    3. **量子化後の重みで** stage を forward し直し、次 stage の入力を作る

    3 が誤差伝播込みの標準 GPTQ の形 — 前段の劣化を後段が見た状態で丸めることになる。

    stage 間の受け渡しは「選んだ出力を次 stage の**唯一の位置引数**にする」形（出力の選び方は
    `select_output`）。keyword 引数は stage 間で不変とみなすのが既定で、変える必要がある場合は
    `advance_kwargs(次 stage の添字, いまの kwargs) -> kwargs` を渡す。

    対象は `nn.Linear`（とその派生）限定 — `H = Σ XᵀX` が「入力の最終次元 = in 軸」という
    linear の形に閉じている（conv は im2col が要る）。列挙は
    {@link quantize.iter_quant_targets} の共有で、`include` は**モジュール FQN**（stage 内の
    局所名）の述語。

    1 本も丸められなかった場合は fail loudly（`quantize` / `quant_methods` と同じ常設診断）。

    NOTE（メモリ）: 同時に生きるのは「現 stage の入力バッチ全部 + 次 stage の入力バッチ全部
    + 現 stage の `H`（対象 linear ごとに `in²` の F64）」。`in = 4096` の linear なら
    `H` 1 本で 128MiB なので、stage あたりの対象本数が効く。
    """
    if method not in CALIB_METHODS:
        raise QuantizeError(f"方式 '{method}' は未対応（{', '.join(CALIB_METHODS)}）")
    if not stages:
        raise QuantizeError("stage が 1 つも無い（実行順の stage 列を渡す）")
    if not batches:
        raise QuantizeError("校正バッチが 1 つも無い")
    if samples < 1:
        raise QuantizeError(f"samples は 1 以上（実測 {samples}）")
    needs_hessian = method in ("gptq", "awq+gptq")
    needs_act = method in ("awq", "awq+gptq")

    current: list[StageBatch] = [(tuple(args), dict(kwargs)) for args, kwargs in batches]
    reports: list[LayerCalibReport] = []
    scales: dict[str, torch.Tensor] = {}
    elements = 0

    with torch.no_grad():
        for index, (prefix, stage) in enumerate(stages):
            targets = list(iter_quant_targets(stage, (nn.Linear,), include))
            stats = _observe_stage(stage, targets, current, needs_hessian, needs_act, samples)
            for local, weight, axis in targets:
                fqn = _qualify(prefix, local)
                report, ledger = _quantize_layer(
                    stats[local], weight, axis, index, fqn, method, spec, block_size
                )
                reports.append(report)
                if ledger is not None:
                    scales[fqn] = ledger
                elements += weight.numel()
            stats.clear()
            if index + 1 < len(stages):
                current = [
                    (
                        (select_output(stage(*args, **kwargs)).detach(),),
                        dict(advance_kwargs(index + 1, kwargs) if advance_kwargs else kwargs),
                    )
                    for args, kwargs in current
                ]

    if not reports:
        raise QuantizeError(
            "校正付き丸めを指定したが対象の nn.Linear が 1 本も無い"
            "（stage の綴りか include を確かめる）"
        )
    return CalibReport(
        method=method,
        grid=spec.kind,
        group_size=spec.group_size,
        stages=len(stages),
        elements=elements,
        fit_stride=spec.fit_stride,
        layers=tuple(reports),
        int4=(
            Int4Report(
                scales=scales, group_size=spec.group_size, modules=len(scales), elements=elements
            )
            if scales
            else None
        ),
    )


def _observe_stage(
    stage: nn.Module,
    targets: Sequence[tuple[str, torch.Tensor, int]],
    batches: Sequence[StageBatch],
    needs_hessian: bool,
    needs_act: bool,
    sample_limit: int,
) -> dict[str, _LayerStats]:
    """フックを掛けて全バッチを 1 周し、対象ごとの校正統計を採って**必ず外す**。"""
    lookup = dict(stage.named_modules())
    stats: dict[str, _LayerStats] = {}
    handles = []
    for local, weight, _axis in targets:
        layer = _LayerStats(in_features=int(weight.shape[-1]))
        stats[local] = layer
        handles.append(
            lookup[_owner_name(local)].register_forward_pre_hook(
                _make_pre_hook(layer, needs_hessian, needs_act, sample_limit), with_kwargs=False
            )
        )
    try:
        for args, kwargs in batches:
            stage(*args, **kwargs)
    finally:
        for handle in handles:
            handle.remove()
    return stats


def _quantize_layer(
    stats: _LayerStats,
    weight: torch.Tensor,
    axis: int,
    stage: int,
    fqn: str,
    method: CalibMethod,
    spec: GridSpec,
    block_size: int,
) -> tuple[LayerCalibReport, torch.Tensor | None]:
    """1 層を丸めて in-place 書き戻し、`(層レポート, scale 台帳 or None)` を返す。

    AWQ 系は `W' = W·diag(s)` を丸めてから `W_eff = Q(W')/s` を書き戻すので、返る台帳は
    常に `None`（`W_eff` の group amax は `Q(W')` の scale と対応しない — モジュール
    docstring の fold / companion）。`awq+gptq` は実効入力が `x/s` になるぶん Hessian も
    `H' = diag(1/s)·H·diag(1/s)` へ変換して渡す。
    """
    where = f"'{fqn}'"
    if not stats.tokens:
        raise QuantizeError(
            f"{where}: 校正入力が 1 行も流れていない（stage {stage} の forward に載っていない）"
        )
    rows = channel_rows(weight, axis)
    alpha: float | None = None
    error: float | None = None
    ledger: torch.Tensor | None = None

    if method == "gptq":
        rounded, ledger = gptq_quantize_rows(rows, stats.hessian, spec, where, block_size)
    else:
        search = awq_search_scale(rows, stats.act_amax, torch.cat(stats.samples), spec, where)
        alpha, error = search.alpha, search.error
        channel = search.channel_scale
        scaled = (rows.to(torch.float64) * channel).to(torch.float32)
        if method == "awq+gptq":
            inverse = 1.0 / channel
            transformed = stats.hessian * inverse.unsqueeze(0) * inverse.unsqueeze(1)
            quantized, _ = gptq_quantize_rows(scaled, transformed, spec, where, block_size)
        else:
            quantized, _ = _bind_grid(spec, scaled, where).round_rows(scaled)
        rounded = (quantized.to(torch.float64) / channel).to(torch.float32)

    weight.copy_(restore_channel_rows(rounded, weight, axis))
    return (
        LayerCalibReport(fqn=fqn, stage=stage, tokens=stats.tokens, alpha=alpha, error=error),
        ledger,
    )
