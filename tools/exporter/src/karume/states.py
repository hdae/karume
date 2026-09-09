"""torch.export 済みグラフを **states 形**（ADR 0067 決定 4 / 5 / 5b）へ書き換える手術。

torch.export は KV state を trace できない。recipe は「chunk 局所の causal self-attention」
（attention が mask を第 4 入力に取る従来形）を export し、その IR をここで states 形
（過去分は state スロット・今 step の k/v は `ins`）へ書き換える。**モデル非依存の機構**なので
汎用 core に置く（Gemma 4 も MiniCPM5 も同じ手術を通る）。

手術は 3 操作:

1. **読み替え** — 対象 attention の mask 入力を落とし、`states` 欄 `{k, v}`（sliding なら
   attrs `window` も）を付ける。causal は述語計算になるので mask tensor は要らなくなる。
2. **書き込みの挿入** — スロット 1 本につき `state_append` を 1 本、**そのスロットの最後の
   読者の直後**へ置く（ADR 0067 決定 5b の発行規約 = 全読者 → append）。
3. **刈り込み** — `outputs` と `state_append` から到達しないノード・宣言を落とす。mask の
   Tmax² 定数と `sym_prefix_slice`（ADR 0010 の定数畳み込みの結果）はここで死ぬ — 残すと
   「誰も読まない Tmax² の f32」が配布物に居座る。

MUST: **純関数**（入力 graph を変異させない）。手術前後の 2 本を同じ検証に掛けられないと、
壊れた IR の出どころ（export か手術か）を切り分けられない。

NOTE: 結果の正しさの門は verify（states 検査・順序検査・shape 検査）で、本モジュールは
**それを通る形を組み立てるだけ**。同じ規則をここへ写さない（二重管理にしない）。
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from karume.ir import IrDim, IrGraph, IrNode, IrState
from karume.ops import ATTENTION_OP, STATE_APPEND_OP
from karume.shapes import declared_shape

#: state スロットの dtype（ADR 0066 決定 2）。語彙の正本は `verify.STATE_DTYPES` で、f16 は
#: 席だけの予約（追記 5）— 手術が作れるのは現に実行できる f32 スロットだけ。
_SLOT_DTYPE = "f32"


class StatesFormError(ValueError):
    """states 形への手術が成立しない（対象の取り違え・配線ミス・指定の誤り）。"""


@dataclass(frozen=True)
class StateAttentionSpec:
    """states 形へ書き換える attention 1 本の指定。

    `output` は対象ノードの `outs[0]` — SSA 単一代入なので、値名がそのままノードの一意識別子。

    `window` は sliding の窓幅（`None` = 全 context — ADR 0067 決定 4 の省略可能 attrs）で、
    `capacity` はスロット容量の**数値上書き**（`None` なら plan の容量記号を使う）。容量は
    `createGenerationContext` が決める値なので記号が既定で、export 時定数に焼くのは
    「容量が実行時に選べない」ことを承知した形（ADR 0066 決定 3）。
    """

    output: str
    k_slot: str
    v_slot: str
    window: int | None = None
    capacity: int | None = None


@dataclass(frozen=True)
class StatesPlan:
    """1 グラフぶんの手術指定（容量記号 + 対象 attention の並び）。"""

    capacity_symbol: str
    attentions: tuple[StateAttentionSpec, ...]


@dataclass
class _Slot:
    """登録中の state スロット。

    `source` は `state_append` の入力名（= 読者が k / v として受けている値）、`last_reader` は
    **nodes 配列添字**で最後にこのスロットへ触れる読者。KV 共有層（同じスロットを複数の
    attention が読む形）は 2 度目以降の登録で既存と突き合わせる。
    """

    shape: list[IrDim]
    window: int | None
    source: str
    last_reader: int


def to_states_form(graph: IrGraph, plan: StatesPlan) -> IrGraph:
    """従来形 attention のグラフを states 形へ書き換えた**新しいグラフ**を返す。

    入力 `graph` は読むだけ（MUST — モジュール docstring）。返るグラフは states 節 +
    `state_append` を持ち、mask の定数畳み込み残骸が刈られている。
    """
    nodes = list(graph.nodes)
    slots: dict[str, _Slot] = {}
    for spec in plan.attentions:
        where = f"attention '{spec.output}'"
        index = _target_index(nodes, spec.output, where)
        node = nodes[index]
        _assert_convertible(node, where)
        window = _window(spec.window, where)
        capacity: IrDim = plan.capacity_symbol if spec.capacity is None else spec.capacity
        # k / v の順で登録する（同着の append 並びがこの順になる — _with_appends）。
        for slot_name, source in ((spec.k_slot, node.ins[1]), (spec.v_slot, node.ins[2])):
            candidate = _Slot(
                shape=_slot_shape(graph, source, capacity, where),
                window=window,
                source=source,
                last_reader=index,
            )
            _register(slots, slot_name, candidate, where)
        attrs = dict(node.attrs)
        if window is not None:
            attrs["window"] = window
        nodes[index] = IrNode(
            op=node.op,
            # mask（第 4 入力）を落として q / k / v の 3 本にする。
            ins=list(node.ins[:3]),
            outs=list(node.outs),
            attrs=attrs,
            states={"k": spec.k_slot, "v": spec.v_slot},
        )
    return _prune(graph, _with_appends(nodes, slots), slots, _symbols(graph, plan, slots))


@dataclass(frozen=True)
class ExternalAttentionSpec:
    """**external states 形**（ADR 0096 段 2）へ書き換える attention 1 本の指定。

    `output` は対象ノードの `outs[0]`（SSA 単一代入なのでノードの一意識別子）。`k_input` /
    `v_input` は trace 用に置いた K / V の**グラフ入力名**で、手術はこの 2 本を落とす
    （借り手は今 step の k/v を持たない — 過去だけを読む）。

    MUST: スロットの形は `kv_heads` / `head_dim` / `capacity` を**指定で受ける**（k/v 入力の
    宣言 shape からは導かない）。外部スロットの実体は貸し手 context のもので、借り手の宣言は
    「貸し手と一致していること」だけが正しさなので、trace 用の使い捨て placeholder の形に
    引きずられてはならない（placeholder の T を小さくした日に宣言が黙って動く）。
    """

    output: str
    k_slot: str
    v_slot: str
    k_input: str
    v_input: str
    kv_heads: int
    head_dim: int
    #: スロット容量（数値、または plan の記号）。
    capacity: IrDim
    window: int | None = None


@dataclass(frozen=True)
class ExternalStatesPlan:
    """1 グラフぶんの external 手術指定（容量記号 + 対象 attention の並び）。"""

    capacity_symbol: str
    attentions: tuple[ExternalAttentionSpec, ...]


def to_external_states_form(graph: IrGraph, plan: ExternalStatesPlan) -> IrGraph:
    """従来形 attention のグラフを **external states 形**へ書き換えた新しいグラフを返す。

    `to_states_form` との差は 3 点だけで、規律（純関数・`_prune` の根・検査は verify）は同じ:

    1. ノードは `ins=[q]` の **readonly 形**（`attrs.readonly = True`）になる。今 step の k / v も
       mask も持たない — 借り手は貸し手のスロットの列 `[column_base, P)` だけを読む
    2. `state_append` を**挿さない**（実体は貸し手のもの）。スロット宣言は `external: True`
    3. 落とした k / v のグラフ入力は `_prune` の孤児入力拒否から外す（意図して落とす 2 本）

    MUST: 入力 `graph` は読むだけ（純関数 — モジュール docstring）。
    MUST: mask 入力（第 4 入力）は従来どおり落とす。借り手の q は 1 行で、bidirectional な
    「全列を見る」形なので mask tensor に情報が無い（列の絞りは window と論理長が持つ）。
    """
    nodes = list(graph.nodes)
    slots: dict[str, _Slot] = {}
    dropped: set[str] = set()
    for spec in plan.attentions:
        where = f"attention '{spec.output}'"
        index = _target_index(nodes, spec.output, where)
        node = nodes[index]
        _assert_convertible(node, where)
        window = _window(spec.window, where)
        # 添字は K = 1 / V = 2（`_assert_dropped_input` が個別に照合する — 集合で見ると
        # K / V の取り違えが通る）。
        for slot_name, source, kv_index in (
            (spec.k_slot, spec.k_input, 1),
            (spec.v_slot, spec.v_input, 2),
        ):
            _assert_dropped_input(graph, node, source, kv_index, where)
            dropped.add(source)
            candidate = _Slot(
                shape=[1, spec.kv_heads, spec.capacity, spec.head_dim],
                window=window,
                source=source,
                last_reader=index,
            )
            _register(slots, slot_name, candidate, where)
        attrs = dict(node.attrs)
        attrs["readonly"] = True
        if window is not None:
            attrs["window"] = window
        nodes[index] = IrNode(
            op=node.op,
            # q だけを残す（今 step の k / v と mask を落とす）。
            ins=[node.ins[0]],
            outs=list(node.outs),
            attrs=attrs,
            states={"k": spec.k_slot, "v": spec.v_slot},
        )
    return _prune(
        graph,
        nodes,
        slots,
        _symbols(graph, plan, slots),
        external=True,
        droppable_inputs=dropped,
    )


def _assert_dropped_input(
    graph: IrGraph, node: IrNode, source: str, index: int, where: str
) -> None:
    """落とす k / v が**そのノードがその位置で実際に読んでいるグラフ入力**であることを見る。

    `index` は期待する `ins` の添字（K = 1 / V = 2）。MUST: 添字ごとに**個別**に照合する
    （集合 `{ins[1], ins[2]}` で見ると K に元 V・V に元 K を渡した取り違えが通り、同形なら
    スロット登録まで成立して沈黙誤値になる — 借り手は貸し手の K スロットから V を読む形で走る）。

    MUST: グラフ入力であることも見る。入力でない名前の指定は「計算途中の値を入力扱いで落とす」
    形で、`_prune` が読者ごと消すので出力の値だけが静かに変わる。
    """
    role = {1: "k", 2: "v"}[index]
    if source != node.ins[index]:
        raise StatesFormError(
            f"{where}: {role} に指定した '{source}' がこのノードの {role}"
            f"（ins[{index}] = '{node.ins[index]}'）でない"
            "（k は ins[1] / v は ins[2] ちょうど）"
        )
    if not any(spec.name == source for spec in graph.inputs):
        raise StatesFormError(
            f"{where}: 落とす k / v '{source}' がグラフ入力でない"
            "（external 手術が落とせるのは trace 用に置いた入力だけ）"
        )


def _target_index(nodes: Sequence[IrNode], output: str, where: str) -> int:
    """`outs[0]` が `output` のノードの添字。"""
    for index, node in enumerate(nodes):
        if node.outs and node.outs[0] == output:
            return index
    raise StatesFormError(f"{where}: この値を定義するノードがグラフに無い")


def _assert_convertible(node: IrNode, where: str) -> None:
    """対象は**従来形の attention**（mask 込み 4 本・states 欄なし）ちょうど。

    MUST: 3 点とも見る。op 違いは指定の取り違え、states 欄ありは二重手術（今 step の k/v を
    2 度書く形になる）、`ins` 3 本は「mask を持たない export」（causal が別の形で表されている
    = 手術の前提が崩れている）。

    MUST: states 欄の検査を arity より**先**に置く。手術済みノードは `ins` が 3 本なので、
    順序が逆だと二重手術が「mask を持たない export」と同じ診断に化ける。
    """
    if node.op != ATTENTION_OP:
        raise StatesFormError(f"{where}: op が '{node.op}'（手術できるのは '{ATTENTION_OP}' だけ）")
    if node.states:
        raise StatesFormError(f"{where}: すでに states 形（{node.states}）")
    if len(node.ins) != 4:
        raise StatesFormError(
            f"{where}: 入力が {len(node.ins)} 本（mask 込みの 4 本 = q / k / v / mask のみ）"
        )


def _window(window: int | None, where: str) -> int | None:
    """sliding の窓幅（`None` = 全 context）。

    MUST: bool を弾く。Python の bool は int の派生なので、`True` が窓幅 1 として黙って通る
    （契約層の `_assert_integer_attr` と同じ規律）。
    """
    if window is None:
        return None
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise StatesFormError(f"{where}: window {window!r} が正整数でない")
    return window


def _slot_shape(graph: IrGraph, source: str, capacity: IrDim, where: str) -> list[IrDim]:
    """今 step の k / v の宣言 `[B,Hkv,M,D]` → スロットの容量込み具体形 `[B,Hkv,C,D]`。

    MUST: B / Hkv / D は**宣言から導く**（引数で受け取らない）。手術が独自に持つと、
    スロットと ins の形一致（ADR 0067 決定 4 の②）が指定ミスで崩れうる。
    """
    shape = declared_shape(graph, source)
    if len(shape) != 4:
        raise StatesFormError(
            f"{where}: '{source}' の宣言 shape {shape} が [B,Hkv,M,D] の rank-4 でない"
        )
    return [shape[0], shape[1], capacity, shape[3]]


def _register(slots: dict[str, _Slot], name: str, candidate: _Slot, where: str) -> None:
    """スロットを登録する（KV 共有層は同じ名前へ複数の読者が来る — ADR 0067 決定 4）。

    MUST: 2 度目以降は導出 shape（= B / Hkv / D / 容量）・`window`・append の入力名の
    **完全一致**を要求する。共有層は所有層の k/v 値テンソルをそのまま `ins` へ配線する規約
    なので、食い違いは配線ミスそのもの — 通すと 1 本のスロットへ別テンソルを書く形になる。
    """
    slot = slots.get(name)
    if slot is None:
        slots[name] = candidate
        return
    if slot.shape != candidate.shape:
        raise StatesFormError(
            f"{where}: スロット '{name}' の導出 shape が読者ごとに違う"
            f"（{slot.shape} / {candidate.shape}）"
        )
    if slot.window != candidate.window:
        raise StatesFormError(
            f"{where}: スロット '{name}' の window が読者ごとに違う"
            f"（{slot.window} / {candidate.window}）"
        )
    if slot.source != candidate.source:
        raise StatesFormError(
            f"{where}: スロット '{name}' へ書く値が読者ごとに違う"
            f"（'{slot.source}' / '{candidate.source}'）"
            " — KV 共有層は所有層の k/v 値テンソルをそのまま ins へ配線する"
        )
    slot.last_reader = max(slot.last_reader, candidate.last_reader)


def _with_appends(nodes: Sequence[IrNode], slots: Mapping[str, _Slot]) -> list[IrNode]:
    """`state_append` を**そのスロットの最後の読者の直後**へ挿す（ADR 0067 決定 5b の①）。

    同じ位置へ複数本が来たときは**スロットの登録順**（spec の並び × k → v）で並べる —
    state effect の順序は nodes 配列順が契約そのものなので、決め方を dict の気分に預けない。
    """
    pending: dict[int, list[IrNode]] = {}
    for name, slot in slots.items():
        pending.setdefault(slot.last_reader, []).append(_append_node(name, slot))
    out: list[IrNode] = []
    for index, node in enumerate(nodes):
        out.append(node)
        out.extend(pending.get(index, ()))
    return out


def _append_node(name: str, slot: _Slot) -> IrNode:
    """1 スロットぶんの `state_append`（出力 0 本の effect op — ADR 0067 決定 5）。

    MUST: sliding スロットには読者と同じ `window` を宣言する。論理 col → 物理 row の写像は
    読み書き同式 MUST（ADR 0067 決定 4）で、片側だけ落とすと沈黙誤読になる（宣言の食い違いは
    verify._assert_state_order が落とす）。
    """
    return IrNode(
        op=STATE_APPEND_OP,
        ins=[slot.source],
        outs=[],
        attrs={} if slot.window is None else {"window": slot.window},
        states={"slot": name},
    )


def _symbols(
    graph: IrGraph, plan: StatesPlan | ExternalStatesPlan, slots: Mapping[str, _Slot]
) -> list[str]:
    """容量記号は**いずれかのスロットが使うときだけ** `symbols` へ足す。

    MUST: 衝突は fail loudly。既存記号との衝突は「入力から束縛される記号を容量としても使う」
    形（値 shape が context 生成時の容量で解けてしまう — ADR 0066 追記 7 の分担が壊れる）で、
    値名との衝突は記号の欄へテンソル名を渡した取り違え。
    """
    symbol = plan.capacity_symbol
    if not any(slot.shape[2] == symbol for slot in slots.values()):
        return list(graph.symbols)
    if symbol in graph.symbols:
        raise StatesFormError(
            f"容量記号 '{symbol}' が graph.symbols に既にある（入力由来の束縛と二重になる）"
        )
    if symbol in graph.values or any(spec.name == symbol for spec in graph.inputs):
        raise StatesFormError(f"容量記号 '{symbol}' が値名と同名（記号の欄へ値名を渡していないか）")
    return [*graph.symbols, symbol]


def _prune(
    graph: IrGraph,
    nodes: Sequence[IrNode],
    slots: Mapping[str, _Slot],
    symbols: list[str],
    *,
    external: bool = False,
    droppable_inputs: Collection[str] = (),
) -> IrGraph:
    """生きた値から到達しないノード・initializer・`values` 宣言を落とす。

    根は `outputs` と**出力を持たないノード**（値を定義しない effect op = `state_append` —
    ADR 0067 決定 5）。effect op を根に入れないとデータ辺だけの到達可能性で丸ごと消える。
    nodes はトポロジカル順（verify._check_definitions が前方参照を拒否する）なので、逆順の
    1 走査で live が確定し、残ったノードの**相対順はそのまま**保たれる。

    MUST: 到達不能になった `inputs` は fail loudly。手術が入力を孤児化するのは配線ミス
    （mask をグラフ入力から作っている export など）で、通すと「呼び手は今までどおり値を渡すのに
    どのノードも読まない」グラフになる — 検査も実行も通るので、誰も気づかない。

    `droppable_inputs` は **意図して落とす入力**（external 手術の k / v placeholder — ADR 0096
    段 2）。孤児拒否から外すと同時に `inputs` 宣言そのものからも消す。MUST: 落とした後も誰かが
    読んでいる形は fail loudly — 宣言だけ消すと「未定義の名前を読むノード」が残る。
    """
    live_values = set(graph.outputs)
    live = [False] * len(nodes)
    for index in reversed(range(len(nodes))):
        node = nodes[index]
        if node.outs and not any(out in live_values for out in node.outs):
            continue
        live[index] = True
        live_values.update(node.ins)
    kept = [node for index, node in enumerate(nodes) if live[index]]

    dropped = set(droppable_inputs)
    still_read = sorted(name for name in dropped if name in live_values)
    if still_read:
        raise StatesFormError(
            f"落とすはずの入力 {still_read} を手術後もどれかのノードが読んでいる"
            "（宣言だけ消すと未定義参照になる）"
        )
    orphans = [
        spec.name
        for spec in graph.inputs
        if spec.name not in live_values and spec.name not in dropped
    ]
    if orphans:
        raise StatesFormError(
            f"手術で入力 {orphans} が到達不能になった"
            "（mask を入力から作っている形は states 形へ落とせない）"
        )

    defined = {out for node in kept for out in node.outs}
    initializers = {
        name: initializer for name, initializer in graph.initializers.items() if name in live_values
    }
    return IrGraph(
        symbols=symbols,
        inputs=[spec for spec in graph.inputs if spec.name not in dropped],
        outputs=list(graph.outputs),
        initializers=initializers,
        values={
            name: value
            for name, value in graph.values.items()
            if name in defined or name in initializers
        },
        states={
            name: IrState(dtype=_SLOT_DTYPE, shape=slot.shape, external=external)
            for name, slot in slots.items()
        },
        nodes=kept,
    )
