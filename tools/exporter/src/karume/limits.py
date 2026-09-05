"""quant が要求する device limit（manifest の optional `requiredLimits`）の導出。

読み手は hub — 宣言があれば**取得の前に**「このデバイスでは動かない」を出せる（宣言の無い
quant は GB 級を落としてから `createSession` で初めて落ちる）。綴りの正本は
`packages/hub/src/manifest.ts` の `REQUIRED_LIMIT_NAMES`（runtime の `REQUIRED_LIMIT_KEYS` と
1 対 1）で、そのうちここが焼くのは**サイズ系 2 つだけ**（{@link WEBGPU_DEFAULT_LIMITS}）:

- `maxStorageBufferBindingSize` — storage buffer 1 本を binding として渡せる上限
- `maxBufferSize` — buffer 1 本の確保上限

workgroup 系（`maxComputeWorkgroupStorageSize` / `maxComputeInvocationsPerWorkgroup` /
`maxComputeWorkgroupSize*`）は**焼かない**: 要求値を決めるのはカーネル設計（タイル形・
`@workgroup_size` の選び・行ブロック化）であって配布物のバイト列ではなく、同じ資産でも
実行時のノブ（数値変種の opt-in）で必要量が変わる。export 時の定数として宣言すると「古い
カーネル設計の下限を名乗り続ける配布物」になり、宣言と実装が独立に動く二重管理になる。

MUST: 宣言が保証するのは**常駐分**（格納テンソルと state スロット）が既定内に収まることだけで、
実行中の**中間テンソル（ノード出力）は含まない**（2026-09-04 裁定）。「欄が無い = 既定スペックの
デバイスで動く」は常駐分についての主張で、中間まで名乗る宣言ではない。融合後の実需要は device が
実際に granted した limit に依存する（行ブロック化は許された binding 幅で刻みが決まる）ので配布形
からは原理的に決まらず、融合**前**のノード出力の最大を焼くと「行ブロックで走るデバイス」を取得の
前に誤って拒否する — 下の「誤拒否を作らない」MUST と同じ失敗形。中間が上限を超える形は Session
構築時の実行時検査が受ける。

MUST: 需要は**常駐前提の寸法**だけから採る — デグレード経路（f16 / i8 / i4 を f32 へ展開して
実行する形）の展開後サイズは含めない。展開ワーストを要求に書くと、本来はそのまま動く
デバイスを取得の**前に**誤って拒否する（宣言は満たせないが実行はできる、が最も損な形）。
展開時の実寸は runtime 側の実行時検査（バッファ確保時の limit 検査）が守る。

MUST: 焼くのは**保証既定を超える席だけ**。「`requiredLimits` が無い = 既定スペックの
デバイスで動く」という意味論を保つためで、既定以内の値を書くと「宣言があるのに何も
制約していない」欄が全配布物に増える（境界＝ちょうど既定値も焼かない）。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from karume.dims import DimError, DimExpr, eval_dim, try_parse_dim
from karume.shards import parse_piece_key


class LimitsError(ValueError):
    """requiredLimits の導出が成立しない（記号束縛の欠落・壊れた宣言）。"""


#: WebGPU 仕様の**保証既定値**（supported limits の default — アダプタに要求しなくても得られる
#: 下限）。これを超える需要だけが `requiredLimits` になる。
#:
#: MUST: 並びがそのまま manifest のキー順になる（辞書順で固定 — 焼き直しで欄の並びが揺れない）。
WEBGPU_DEFAULT_LIMITS: Mapping[str, int] = {
    "maxBufferSize": 268_435_456,
    "maxStorageBufferBindingSize": 134_217_728,
}

#: state スロットの dtype → 1 要素のバイト数。語彙の正本は `karume.verify.STATE_DTYPES`
#: （f16 は席だけの予約で、実行経路があるのは f32 だけ）— ここは寸法を数えるための対応表なので、
#: 語彙の写しではなく**知っている dtype だけ**を持ち、未知は fail loudly で落とす。
STATE_DTYPE_BYTES: Mapping[str, int] = {"f32": 4}

#: state shape の記号次元（KV 容量）の束縛値を引く `pipelineConfig` の席。
#:
#: MUST: 既定容量（`capacity`）ではなく**配布形が許す最大容量**（`maxPosition`）を読む。
#: 容量は実行時ノブ（ホストが `createGenerationContext` で選ぶ）なので、既定値で焼くと
#: 「既定より大きい容量を選んだ瞬間に、宣言を満たすデバイスで `createSession` が落ちる」
#: という最も損な形になる。上限で焼けば宣言は常に十分側へ倒れる。
STATE_CAPACITY_KEY = "maxPosition"


def required_limits(demand: int) -> dict[str, int]:
    """常駐 1 バッファの最大バイト数 → 焼く `requiredLimits`（既定超えの席だけ・厳密比較）。

    binding 需要と buffer 需要が同じ値なのは、重みテンソルも state スロットも**1 バッファ =
    1 binding**で置かれるため（分割して束ねる形は無い）。既定の違いから、需要が
    128MiB〜256MiB の帯では `maxStorageBufferBindingSize` だけが焼かれる — このとき
    `maxBufferSize` は仕様既定 256MiB に落ちるので、`maxStorageBufferBindingSize ≤
    maxBufferSize`（requestDevice の前提）は自動で保たれる。
    """
    if demand < 0:
        raise LimitsError(f"需要 {demand} バイトが負")
    return {name: demand for name, default in WEBGPU_DEFAULT_LIMITS.items() if demand > default}


def max_tensor_payload(header: Mapping[str, Any], where: str) -> int:
    """safetensors ヘッダの**最大テンソル payload バイト数**（テンソル 0 本なら 0）。

    数えるのは `data_offsets` の差 = **格納形そのままのバイト数**（f16 / i8 / i4 は圧縮寸法の
    まま）。1 テンソルが 1 つの storage buffer になるので、この値がそのまま需要になる。

    MUST: 分割テンソル（`<親名>#NNNNN-of-NNNNN` — ADR 0090 決定 1）は**親へ畳んで
    合算**する。GPU 側は分割を知らず親 1 本ぶんのバッファを確保するので、断片の最大を採ると
    `requiredLimits` が過小に焼かれ、「宣言は満たすのに `createSession` で落ちる」という最も
    損な形になる。呼び手はコンポーネント全 shard のヘッダを 1 枚へ畳んで渡す
    （`karume.dist.component_demand_bytes` — 断片は shard を跨いで散る）。

    グラフ shard（ADR 0081 の「`karume_ir` だけ・データ節は空」）は 0 を返す。
    """
    totals: dict[str, int] = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        offsets = spec.get("data_offsets") if isinstance(spec, Mapping) else None
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise LimitsError(f"{where}: テンソル '{name}' の data_offsets が 2 要素でない")
        begin, end = offsets
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (begin, end)):
            raise LimitsError(f"{where}: テンソル '{name}' の data_offsets が整数でない")
        if not 0 <= begin <= end:
            raise LimitsError(
                f"{where}: テンソル '{name}' の data_offsets [{begin}, {end}] が昇順の非負でない"
            )
        parsed = parse_piece_key(name)
        owner = name if parsed is None else parsed[0]
        totals[owner] = totals.get(owner, 0) + (end - begin)
    return max(totals.values(), default=0)


def state_bindings(
    states: Mapping[str, Any], pipeline_config: Mapping[str, Any], where: str
) -> dict[str, int]:
    """state shape の記号次元 → 束縛値（`pipelineConfig` の位置上限の席から引く）。

    束縛点は `createGenerationContext(spec.bindings)`（ADR 0066 追記 7）で、そこへ渡る容量は
    ホストが選ぶ実行時ノブ。配布形が持っているその**上限**が `pipelineConfig` の
    {@link STATE_CAPACITY_KEY} 席で、requiredLimits はこの上限で焼く（席の docstring）。

    MUST: 束縛が取れなければ fail loudly — 黙って state を勘定から外すと「重みだけを見た
    小さい値」が焼かれ、**宣言があるのに足りない**という最も損な形になる（宣言が無い方が
    まだ実行時に落ちる）。記号が 2 本以上あるのも同じ扱い: どれが容量記号かを配布形からは
    決められない（states 形の記号は容量 1 本 — ADR 0066 決定 2）。
    """
    found: set[str] = set()
    for slot in states.values():
        for dim in _slot_shape(slot, where):
            if isinstance(dim, str):
                found.add(_parse_symbol(dim, where).sym)
    symbols = sorted(found)
    if not symbols:
        return {}
    if len(symbols) > 1:
        raise LimitsError(
            f"{where}: state shape の記号が {symbols} の複数ある"
            f" — 束縛できるのは pipelineConfig['{STATE_CAPACITY_KEY}'] が決める容量 1 本だけ"
        )
    capacity = pipeline_config.get(STATE_CAPACITY_KEY)
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise LimitsError(
            f"{where}: state shape の記号 '{symbols[0]}' を束縛する"
            f" pipelineConfig['{STATE_CAPACITY_KEY}'] が正整数でない（{capacity!r}）"
        )
    return {symbols[0]: capacity}


def max_state_slot_bytes(
    graph: Mapping[str, Any], pipeline_config: Mapping[str, Any], where: str
) -> int:
    """グラフの state スロット（ADR 0066 決定 2）**1 本の最大**バイト数（states 無しなら 0）。

    スロットは `createGenerationContext` が容量ぶん丸ごと確保する常駐バッファ（1 スロット =
    1 バッファ = 1 binding）なので、記号を**配布形が許す最大容量**で束縛した寸法がそのまま
    需要になる。KV 容量の大きい系列では**最大テンソルより state の方が大きい**（どちらも
    1 バッファのまま — 重みは shard を跨いで配れるが、GPU 側で 1 本に戻る）。
    """
    states = graph.get("states")
    if states is None:
        return 0
    if not isinstance(states, dict):
        raise LimitsError(f"{where}: graph.states がオブジェクトでない")
    bindings = state_bindings(states, pipeline_config, where)
    largest = 0
    for name, slot in states.items():
        at = f"{where}: state スロット '{name}'"
        dtype = slot.get("dtype") if isinstance(slot, Mapping) else None
        item_bytes = STATE_DTYPE_BYTES.get(dtype) if isinstance(dtype, str) else None
        if item_bytes is None:
            raise LimitsError(
                f"{at} の dtype {dtype!r} は寸法を数えられない"
                f"（知っているのは {sorted(STATE_DTYPE_BYTES)}）"
            )
        elements = 1
        for dim in _slot_shape(slot, where):
            elements *= _dim_size(dim, bindings, at)
        largest = max(largest, elements * item_bytes)
    return largest


def _slot_shape(slot: Any, where: str) -> list[Any]:
    """state スロット宣言の `shape`（形だけを見る — 値域の門は `karume.verify` が持つ）。"""
    shape = slot.get("shape") if isinstance(slot, Mapping) else None
    if not isinstance(shape, list) or not shape:
        raise LimitsError(f"{where}: state スロットの shape が非空の配列でない（{shape!r}）")
    return shape


def _parse_symbol(dim: str, where: str) -> DimExpr:
    """記号次元 `coeff·sym+offset` を分解する（非正準は fail loudly）。"""
    expr = try_parse_dim(dim)
    if expr is None:
        raise LimitsError(f"{where}: state shape の次元式 '{dim}' が正準文法に適合しない")
    return expr


def _dim_size(dim: Any, bindings: Mapping[str, int], where: str) -> int:
    """次元 1 つを具体値にする（数値はそのまま・記号は束縛表を当てる）。"""
    if isinstance(dim, bool) or not isinstance(dim, int | str):
        raise LimitsError(f"{where}: shape の次元 {dim!r} が整数でも記号でもない")
    if isinstance(dim, int):
        if dim < 1:
            raise LimitsError(f"{where}: shape の次元 {dim} が正整数でない")
        return dim
    try:
        return eval_dim(_parse_symbol(dim, where), bindings)
    except DimError as cause:
        raise LimitsError(f"{where}: 次元 '{dim}' を束縛できない: {cause}") from cause
