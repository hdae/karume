"""op 契約の適合表テスト（Python 側）。

正本は実装ではなく `packages/runtime/tests/fixtures/op-contracts.json`（リポジトリ直下）で、TS 実装
（packages/runtime/tests/ops_conformance_test.ts）は**同じ表**に対して同じことを確かめる。表をコピーせず
読み込むのは、コピーが増えた瞬間に「片側だけ通る契約」が生まれるため（test_dims.py と
同じ規律）。

見るのは両実装で沈黙のうちに割れうる契約面だけ: op 名の全集合 / アリティ /
スロット dtype / attrs キー集合 / attrs の値域 / 出力数と出力 slot 別の dtype 写像
（ADR 0068 決定 1）/ 出力 shape 規則（rank 上限を含む）/ 低精度格納の適格スロット（ADR 0018）。
"""

from __future__ import annotations

import pytest
from conftest import OP_CONTRACT_TABLE, OP_CONTRACT_TABLE_PATH

from karume.dims import eval_dim, parse_dim
from karume.ops import (
    OP_CONTRACTS,
    STRIDED_RANK,
    WEIGHT_CHANNEL_AXES,
    WEIGHT_SLOTS,
    OpContractError,
    PerSlotDtypes,
    UniformDtypes,
    resolve_op_contract,
)
from karume.shapes import compute_output_shape

OPS = OP_CONTRACT_TABLE["ops"]
ATTR_VALUES = OP_CONTRACT_TABLE["attr_values"]
SHAPES = OP_CONTRACT_TABLE["shapes"]

ATTR_KEYS = {entry["op"]: entry["attrs"] for entry in OPS}

#: 出力次元を入力から伝播させず **attrs から作る** op。束縛を代入しても記号のまま出るのが
#: 正しい（prefix 長 coeff·sym+offset — ADR 0010）ので、下の代入テストだけ扱いが違う。
#: 束縛を当てる層は TS 側（plan.ts）で、エクスポータ側は束縛を持たない。
DIM_CREATING_OPS = {"sym_prefix_slice"}


def _shape_id(case) -> str:
    ins = " ".join(str(shape) for shape in case["ins"])
    return f"{case['op']}({ins}){'!' if case.get('throws') else ''}"


def _resolve(shape, bindings) -> list[int]:
    """次元言語を束縛で数値へ落とす（記号のままの計算と結論が一致することの確認用）。"""
    return [dim if isinstance(dim, int) else eval_dim(parse_dim(dim), bindings) for dim in shape]


def _shape_of(shape, bindings):
    """bindings=None なら記号のまま、あれば数値へ解決してから渡す。"""
    return shape if bindings is None else _resolve(shape, bindings)


def _compute(case, bindings=None):
    declared = case.get("declared")
    return compute_output_shape(
        resolve_op_contract(case["op"]),
        [_shape_of(shape, bindings) for shape in case["ins"]],
        "t",
        declared=None if declared is None else _shape_of(declared, bindings),
        attrs=case.get("attrs"),
    )


class TestFixtureTable:
    def test_the_shared_case_table_exists_and_has_all_sections(self):
        assert OP_CONTRACT_TABLE_PATH.exists(), f"適合ケース表が無い: {OP_CONTRACT_TABLE_PATH}"
        assert {"strided_rank_max", "ops", "attr_values", "shapes"} <= set(OP_CONTRACT_TABLE)

    def test_the_op_set_matches_the_contract_table(self):
        assert sorted(entry["op"] for entry in OPS) == sorted(OP_CONTRACTS)

    def test_no_op_is_declared_twice(self):
        # 重複すると片方の宣言が黙って無視され、突合が緩む。
        assert len({entry["op"] for entry in OPS}) == len(OPS)

    def test_the_strided_rank_limit_matches_the_implementation(self):
        assert OP_CONTRACT_TABLE["strided_rank_max"] == STRIDED_RANK


class TestDeclaredContract:
    """ops 節: アリティ / スロット dtype / attrs キー集合が契約表と一致する。"""

    @pytest.mark.parametrize("entry", OPS, ids=lambda e: e["op"])
    def test_arity_matches(self, entry):
        assert resolve_op_contract(entry["op"]).arity == entry["arity"]

    @pytest.mark.parametrize("entry", OPS, ids=lambda e: e["op"])
    def test_variadic_matches(self, entry):
        """可変アリティは「入力何本まで受理するか」という契約面そのもの。

        片側だけ可変にすると、エクスポータが書ける本数とランタイムが受理する本数が割れる。
        """
        assert resolve_op_contract(entry["op"]).variadic == entry.get("variadic", False)

    @pytest.mark.parametrize("entry", OPS, ids=lambda e: e["op"])
    def test_max_arity_matches(self, entry):
        """省略可能な末尾入力の上限も契約面（片側だけ広げると mask 付き IR が割れる）。"""
        assert resolve_op_contract(entry["op"]).max_arity == entry.get("max_arity")

    @pytest.mark.parametrize("entry", OPS, ids=lambda e: e["op"])
    def test_attr_keys_match(self, entry):
        assert sorted(resolve_op_contract(entry["op"]).attr_keys) == sorted(entry["attrs"])

    @pytest.mark.parametrize("entry", OPS, ids=lambda e: e["op"])
    def test_output_dtypes_match(self, entry):
        """出力 slot 別の dtype 写像の列（省略時は出力 1 本・スロット 0 の受理集合上の恒等）。

        比較（f32 → bool）・bool の sum（→ i32）・where（bool → f32）が両側で揃っていない
        と、「エクスポータが書けるのにランタイムが別 TypedArray として読む」が生える。
        列の長さ = 契約が宣言する出力数（ADR 0068 決定 1）も同じ表で突き合わせる。
        cast だけは出力が attrs.to で決まるのでこの欄を持たない。
        """
        contract = resolve_op_contract(entry["op"])
        if entry["op"] == "cast":
            assert "out_dtypes" not in entry
            return
        slots = contract.slot_dtypes
        domain = slots.accept if isinstance(slots, UniformDtypes) else slots.slots[0]
        declared = entry.get("out_dtypes", [{}])

        assert contract.output_dtypes == tuple(
            {dtype: slot.get(dtype, dtype) for dtype in domain} for slot in declared
        )
        assert contract.output_count == len(declared)

    @pytest.mark.parametrize("entry", OPS, ids=lambda e: e["op"])
    def test_weight_slot_matches(self, entry):
        """低精度格納の適格スロット（ADR 0018）。

        割れると「エクスポータが f16 で書いた重みをランタイムが適格外と見なして CPU 展開する」
        （VRAM 削減が黙って消える）か、その逆の「f32 golden と対応しない丸め」になる。
        どちらも例外にならないので表の側から突き合わせる。
        """
        assert WEIGHT_SLOTS.get(entry["op"]) == entry.get("weight_slot")

    def test_the_weight_slot_table_has_no_extra_entries(self):
        # 実装側にだけある適格スロット（= 表に載っていない降格経路）も落とす。
        assert len([entry for entry in OPS if "weight_slot" in entry]) == len(WEIGHT_SLOTS)

    @pytest.mark.parametrize("entry", OPS, ids=lambda e: e["op"])
    def test_channel_axis_matches(self, entry):
        """i8 の per-channel scale が乗る軸（ADR 0019）。

        割れると「エクスポータが軸 0 で作った scale をカーネルが軸 1 として引く」形になり、
        **例外は出ずに値だけが壊れる**（conv_transpose1d の [Cin,Cout,K] は Cin == Cout の
        とき shape 検査も通る）。重みスロットと過不足なく同時に載ることも同じ表で見る。
        """
        assert WEIGHT_CHANNEL_AXES.get(entry["op"]) == entry.get("channel_axis")
        assert ("channel_axis" in entry) == ("weight_slot" in entry), entry["op"]

    def test_the_channel_axis_table_has_no_extra_entries(self):
        assert len([entry for entry in OPS if "channel_axis" in entry]) == len(WEIGHT_CHANNEL_AXES)

    @pytest.mark.parametrize("entry", OPS, ids=lambda e: e["op"])
    def test_slot_dtypes_match(self, entry):
        slots = resolve_op_contract(entry["op"]).slot_dtypes
        declared = entry["dtypes"]
        if isinstance(slots, UniformDtypes):
            assert declared["kind"] == "uniform", entry["op"]
            assert sorted(slots.accept) == sorted(declared["accept"])
            return
        assert isinstance(slots, PerSlotDtypes)
        assert declared["kind"] == "perSlot", entry["op"]
        assert [sorted(accept) for accept in slots.slots] == [
            sorted(accept) for accept in declared["slots"]
        ]


class TestAttrValues:
    """attr_values 節: 値域は検査関数でしか表せないので、実値を通して判定を突き合わせる。"""

    @pytest.mark.parametrize("entry", ATTR_VALUES, ids=lambda e: f"{e['op']}.{e['attr']}")
    def test_the_attr_is_declared_in_the_ops_section(self, entry):
        assert entry["attr"] in ATTR_KEYS[entry["op"]]

    @pytest.mark.parametrize("entry", ATTR_VALUES, ids=lambda e: f"{e['op']}.{e['attr']}")
    def test_accepted_values_pass_the_schema(self, entry):
        check = resolve_op_contract(entry["op"]).attrs[entry["attr"]]
        for value in entry["accept"]:
            check(value, f"{entry['op']}.{entry['attr']}")

    @pytest.mark.parametrize("entry", ATTR_VALUES, ids=lambda e: f"{e['op']}.{e['attr']}")
    def test_rejected_values_fail_loudly(self, entry):
        check = resolve_op_contract(entry["op"]).attrs[entry["attr"]]
        for value in entry["reject"]:
            with pytest.raises(OpContractError):
                check(value, f"{entry['op']}.{entry['attr']}")

    def test_every_declared_attr_has_a_value_range_case(self):
        """値域が表に無い attr を残さない（載せ忘れると値域が無検証のまま残る）。"""
        covered = {f"{entry['op']}.{entry['attr']}" for entry in ATTR_VALUES}

        missing = [
            f"{op}.{attr}"
            for op, attrs in ATTR_KEYS.items()
            for attr in attrs
            if f"{op}.{attr}" not in covered
        ]

        assert missing == []


class TestOutputShapes:
    """shapes 節: 宣言 shape（記号を含む）から計算した出力が表どおりになる。"""

    @pytest.mark.parametrize("case", SHAPES, ids=_shape_id)
    def test_shapes_follow_the_table(self, case):
        if case.get("throws"):
            with pytest.raises(OpContractError):
                _compute(case)
            return

        # 表の outs は出力 slot 別の列（ADR 0068 決定 1）。compute_output_shape も列を返す
        # ので、列どうしで突き合わせる（本数が割れた時点でここが落ちる）。
        assert _compute(case) == case["outs"]

    @pytest.mark.parametrize("case", SHAPES, ids=_shape_id)
    def test_the_numeric_reading_agrees_with_the_symbolic_one(self, case):
        """束縛を当てた数値 shape でも同じ結論になる（TS 側はこちらの定義域で同じ表を回す）。

        記号のままの計算と、束縛を当ててからの計算が食い違うと「export では通るのに
        実行時 shape で落ちる」が復活する — 代入で閉じることをここで固定する。
        """
        bindings = case.get("bindings", {})
        if case.get("throws"):
            with pytest.raises(OpContractError):
                _compute(case, bindings)
            return
        if case["op"] in DIM_CREATING_OPS:
            # 代入で閉じ**ない**ことそのものを固定する（入力を数値にしても attrs 由来の
            # 記号次元が出る = 束縛の層はここではない）。素通しにすると層の分担が緩む。
            assert _compute(case, bindings) == _compute(case)
            return

        assert _compute(case, bindings) == [_resolve(shape, bindings) for shape in case["outs"]]

    def test_every_op_is_exercised_by_an_accepting_case(self):
        """shape 規則が表で踏まれていない op を残さない。"""
        accepted = {case["op"] for case in SHAPES if not case.get("throws")}

        assert accepted == {entry["op"] for entry in OPS}
