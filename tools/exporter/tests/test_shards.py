"""配布コンテナの shard 分割規則（ADR 0090 — shard 仕様 v3・テンソル分割）。

ここが見るのは**規則そのもの**（名前・詰め方・分割・検査）だけで、実際に書けるかは
`test_emit.py`、宣言に載るかは `test_dist.py` が持つ。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import ClassVar

import pytest

from karume.shards import (
    MAX_SHARDS,
    SHARD_BYTE_LIMIT,
    SHARD_DATA_CAPACITY,
    SHARD_HEADER_ALLOWANCE,
    Piece,
    ShardError,
    assert_co_shard,
    assert_shard_partition,
    pack_shards,
    parse_piece_key,
    piece_key,
    resolve_shards,
    shard_name,
    shard_path,
    shard_siblings,
)

#: 対の写像は**対称**で持つ（weight → scale と scale → weight）。
PAIR = {"w": "karume.scale.w", "karume.scale.w": "w"}

MIB = 1024 * 1024

#: 詰め方の検算で使う容量（実配布の容量ではなく、MiB 単位の暗算が効く丸い値）。
CAPACITY = 1024 * MIB


def uniform(total_mib: int) -> tuple[list[str], dict[str, int]]:
    """1MiB のテンソル `total_mib` 本（GiB 規模の分割を実データ無しで踏むための合成）。"""
    order = [f"t{index:05d}" for index in range(total_mib)]
    return order, dict.fromkeys(order, MIB)


def _member_bytes(member: str | Piece, sizes: dict[str, int]) -> int:
    """member 1 つのバイト数（合成は 1 行 1 バイトなので piece は行数そのもの）。"""
    return member.end - member.begin if isinstance(member, Piece) else sizes[member]


def shard_mib(groups: list[tuple[str, ...]], sizes: dict[str, int]) -> list[int]:
    """**weight shard** ごとのデータ節を MiB で。

    先頭のグラフ shard は常に空（読み手契約 1）なので、詰め方の検算からは落とす — 落とさないと
    どの検算も先頭の `0` を引きずる。空であること自体は {@link TestTheGraphShard} が見る。
    """
    return [sum(sizes[name] for name in group) // MIB for group in groups[1:]]


class TestTheByteLimit:
    """上限は固定定数（配布形の不変条件であって呼び手の好みではない）。"""

    def test_it_is_256_mebibytes(self) -> None:
        """hub の `MAX_SHARD_BYTES`（`2 ** 28`）と同値 — 両側が同じ値で同じ量を測る。"""
        assert SHARD_BYTE_LIMIT == 268_435_456

    def test_the_writer_capacity_leaves_a_header_allowance_under_the_limit(self) -> None:
        """上限は**ファイル長**なので、書き手はヘッダぶんを空けてデータ節を詰める。

        余裕を数えるのは読み手ではなく書き手側 — 足りなかった回はファイル長の門
        （`karume.verify.assert_shard_byte_limits`）が落とす。
        """
        assert SHARD_HEADER_ALLOWANCE == 1024 * 1024
        assert SHARD_DATA_CAPACITY == SHARD_BYTE_LIMIT - SHARD_HEADER_ALLOWANCE
        assert SHARD_DATA_CAPACITY < SHARD_BYTE_LIMIT

    def test_the_shard_count_ceiling_matches_the_reader(self) -> None:
        """1 dtype エントリの shard 本数の上限は hub の `MAX_SHARDS` と**同値**。

        焼く側が先に落とすための値なので（`karume.dist` はこれを import して manifest 検査に
        使う）、片方だけが動くと「書けるが読めない」または「読めるが焼けない」配布形が生まれる。
        綴りの正本は `packages/hub/src/manifest.ts` の `MAX_SHARDS`。
        """
        assert MAX_SHARDS == 1024

    def test_it_leaves_room_under_the_single_arraybuffer_ceiling(self) -> None:
        """Chromium の単一 `ArrayBuffer` 上限（docs/limitations）を割るのが必要条件。

        測るのがファイル長になったので、天井と比べるのは shard 1 本の**全長**そのもの。
        """
        ceiling = 2_145_386_496

        assert ceiling - SHARD_BYTE_LIMIT >= 1_000_000_000


class TestTheGraphShard:
    """先頭は必ず**グラフ shard**（テンソル 0 本の空 shard — 読み手契約 1・ADR 0081）。"""

    def test_the_first_shard_is_always_empty(self) -> None:
        """総量が容量に遠く及ばなくても、実重みは weight shard 側にしか載らない。"""
        groups = pack_shards(["a", "b"], {"a": 4, "b": 4}, {}, {}, capacity=100)

        assert groups == [(), ("a", "b")]

    def test_a_graph_without_tensors_is_the_graph_shard_alone(self) -> None:
        """テンソルが 1 本も無ければ weight shard は開かない（空ファイルを作らない）。"""
        assert pack_shards([], {}, {}, {}, capacity=8) == [()]

    def test_no_weight_shard_is_ever_empty(self) -> None:
        """空の weight shard は「読んでも何も確定しない 1 往復」でしかない。"""
        order, sizes = uniform(2500)

        groups = pack_shards(order, sizes, {}, {})

        assert all(group for group in groups[1:])


class TestShardNames:
    """連番は `<拡張子の前>-NNNNN-of-NNNNN<拡張子>`（HF の慣行に似せた形・index.json は無い）。"""

    def test_it_numbers_the_stem_and_keeps_the_suffix(self) -> None:
        assert shard_name("model.safetensors", 1, 3) == "model-00001-of-00003.safetensors"

    def test_it_only_renames_the_last_path_segment(self) -> None:
        assert (
            shard_name("front/model.safetensors", 2, 2) == "front/model-00002-of-00002.safetensors"
        )

    def test_it_keeps_only_the_last_suffix(self) -> None:
        """`model.i8.safetensors` のような多段の綴りでも拡張子は 1 つだけ動かさない。"""
        assert shard_name("model.i8.safetensors", 1, 2) == "model.i8-00001-of-00002.safetensors"

    def test_the_path_form_stays_in_the_same_directory(self, tmp_path: Path) -> None:
        assert shard_path(tmp_path / "model.safetensors", 3, 4) == (
            tmp_path / "model-00003-of-00004.safetensors"
        )

    @pytest.mark.parametrize(("index", "total"), [(0, 2), (3, 2), (1, MAX_SHARDS + 1)])
    def test_it_refuses_a_number_outside_the_range(self, index: int, total: int) -> None:
        with pytest.raises(ShardError, match="連番"):
            shard_name("model.safetensors", index, total)


class TestSequentialPacking:
    """順序は変えず、対は原子、容量を跨がない（読み手契約 2〜4）。"""

    def test_everything_fits_in_one_weight_shard_under_the_limit(self) -> None:
        """総量が容量以下なら weight shard は 1 本（グラフ shard との 2 ファイル）。"""
        groups = pack_shards(["a", "b", "c"], {"a": 4, "b": 4, "c": 4}, {}, {}, capacity=100)

        assert groups == [(), ("a", "b", "c")]

    def test_it_opens_a_new_shard_before_the_limit_would_break(self) -> None:
        groups = pack_shards(["a", "b", "c"], {"a": 4, "b": 4, "c": 4}, {}, {}, capacity=8)

        assert groups == [(), ("a", "b"), ("c",)]

    def test_a_tensor_that_exactly_fills_the_limit_stays_in_place(self) -> None:
        """境界は「超えたら」であって「届いたら」ではない（`used + size > capacity`）。"""
        groups = pack_shards(["a", "b"], {"a": 8, "b": 1}, {}, {}, capacity=8)

        assert groups == [(), ("a",), ("b",)]

    def test_it_keeps_the_given_order_inside_the_shards(self) -> None:
        """並べ替えない — 詰める順は書き手が決めた書き出し順そのもの（ADR 0063）。"""
        groups = pack_shards(["c", "a", "b"], {"a": 1, "b": 1, "c": 1}, {}, {}, capacity=100)

        assert groups == [(), ("c", "a", "b")]

    def test_the_same_input_always_splits_the_same_way(self) -> None:
        """決定的（同入力 → 同分割）— 再 dist で sha256 が揺れないことの前提。"""
        order = ["a", "w", "b", "karume.scale.w", "c"]
        sizes = {"a": 3, "w": 5, "b": 3, "karume.scale.w": 2, "c": 4}

        first = pack_shards(order, sizes, {}, PAIR, capacity=8)
        second = pack_shards(order, sizes, {}, PAIR, capacity=8)

        assert first == second


class TestTheMinimalShardCount:
    """本数は「容量の下での最小連続分割数」（書き手ポリシー — ADR 0081 決定 4）。

    実寸を踏むテストは書けないので、1MiB のテンソルを並べて総量だけ合わせ、容量は暗算の効く
    {@link CAPACITY}（1GiB）を明示する。
    """

    def test_the_count_is_the_ceiling_of_the_total_over_the_limit(self) -> None:
        """容量の 3.2 倍 → 4 本（`ceil(3.2)`）。均すので 3 本 + 端数にはならない。"""
        order, sizes = uniform(3277)

        groups = pack_shards(order, sizes, {}, {}, capacity=CAPACITY)

        assert shard_mib(groups, sizes) == [820, 819, 819, 819]

    def test_an_asset_just_over_the_limit_opens_a_second_shard(self) -> None:
        """容量の 1.11 倍は 2 本 — 単一ファイルの席はもう無い（ADR 0081）。"""
        order, sizes = uniform(1137)

        groups = pack_shards(order, sizes, {}, {}, capacity=CAPACITY)

        assert shard_mib(groups, sizes) == [569, 568]

    def test_an_asset_at_the_limit_keeps_a_single_weight_shard(self) -> None:
        """ちょうど容量ぶんは 1 本のまま（境界は「超えたら」）。"""
        order, sizes = uniform(1024)

        groups = pack_shards(order, sizes, {}, {}, capacity=CAPACITY)

        assert groups == [(), tuple(order)]

    def test_one_byte_over_the_limit_opens_the_second_shard(self) -> None:
        """+1 バイトで 2 本（X1 TG-2 相当の境界 — 容量は「以下」で判定する）。"""
        groups = pack_shards(["a", "b"], {"a": CAPACITY, "b": 1}, {}, {}, capacity=CAPACITY)

        assert groups == [(), ("a",), ("b",)]

    def test_the_count_can_exceed_the_ceiling_when_the_order_is_lumpy(self) -> None:
        """容量 0.6 倍の**対**が 3 つ（総量 1.8 倍）は 2 本に詰め替えられない。

        隣接 2 つで容量の 1.2 倍になるのでどの 2 つも同居できない。対は割れないので piece にも
        ならない（1 対 = 1 単位 — 分割は単位の中の重みにだけ掛かる）。並べ替えれば減らせるが、
        宣言順を動かすのは読み手契約の側の変更なので買わない（ADR 0081 決定 4 の NOTE）。
        """
        pairs = {f"w{index}": f"karume.scale.w{index}" for index in range(3)}
        companions = {**pairs, **{scale: name for name, scale in pairs.items()}}
        order = [name for pair in pairs.items() for name in reversed(pair)]
        sizes: dict[str, int] = dict.fromkeys(pairs, 613 * MIB)
        sizes.update(dict.fromkeys(pairs.values(), MIB))

        groups = pack_shards(order, sizes, {}, companions, capacity=CAPACITY)

        assert [len(group) for group in groups] == [0, 2, 2, 2]
        assert sum(sizes.values()) < 2 * CAPACITY


class TestBalancing:
    """本数を固定したら**均す** — 端数 shard は尾部スラックではなく均しで消す。"""

    def test_the_shards_come_out_within_one_unit_of_each_other(self) -> None:
        """容量の 2.6 倍 → 3 本がほぼ等分（旧規則の `[1024, 1024, 614]` を置き換える形）。"""
        order, sizes = uniform(2662)

        groups = pack_shards(order, sizes, {}, {}, capacity=CAPACITY)

        assert shard_mib(groups, sizes) == [888, 887, 887]

    def test_no_shard_ever_exceeds_the_limit_while_balancing(self) -> None:
        """均しは容量を緩めない（目標に届いても、容量を跨ぐ単位は次の shard へ送る）。"""
        order, sizes = uniform(2662)

        groups = pack_shards(order, sizes, {}, {}, capacity=CAPACITY)

        assert all(sum(sizes[name] for name in group) <= CAPACITY for group in groups)

    def test_it_refuses_to_cut_where_the_rest_would_not_fit(self) -> None:
        """suffix 実行可能性ガード — 目標に届いても、残りが残り shard に収まらない位置では
        cut を打たない。

        `[4, 1, 5, 5, 1]`（容量 5）は 4 本。1 本目は目標 4 に**最初の単位で届く**が、そこで
        切ると残り `[1, 5, 5, 1]` が 3 本に収まらない（4 本要る）ので、`1` まで詰めてから切る。
        ガードを外すと最後の `1` が行き場を失う（= どの shard にも載らない）。
        """
        order = ["a", "b", "c", "d", "e"]
        sizes = {"a": 4, "b": 1, "c": 5, "d": 5, "e": 1}

        groups = pack_shards(order, sizes, {}, {}, capacity=5)

        assert groups == [(), ("a", "b"), ("c",), ("d",), ("e",)]

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_the_rules_hold_for_arbitrary_size_sequences(self, seed: int) -> None:
        """乱択の並びに対する不変条件（分割・容量・非空・最小本数・決定性）。

        規則は「順序を変えない連続分割」なので、乱数で作れるのは**並びだけ**。期待値を
        別実装で綴ると規則の写経になるので、確かめるのは規則そのものが述べている性質に絞る。
        """
        rng = random.Random(seed)
        capacity = rng.randint(5, 40)
        order = [f"t{index:03d}" for index in range(rng.randint(1, 60))]
        sizes = {name: rng.randint(1, capacity) for name in order}

        groups = pack_shards(order, sizes, {}, {}, capacity=capacity)

        assert groups[0] == ()
        assert [name for group in groups for name in group] == order
        assert all(group for group in groups[1:])
        assert all(sum(sizes[name] for name in group) <= capacity for group in groups)
        assert groups == pack_shards(order, sizes, {}, {}, capacity=capacity)
        # 最小本数: 貪欲（入るだけ詰める）が連続分割の最小そのもの。
        greedy, used = 1, 0
        for name in order:
            if used + sizes[name] > capacity:
                greedy, used = greedy + 1, 0
            used += sizes[name]
        assert len(groups) - 1 == greedy

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_the_rules_hold_when_oversized_tensors_are_mixed_in(self, seed: int) -> None:
        """同じ不変条件を**容量超えのテンソルを混ぜた**並びで（piece が現れる経路）。

        合成は 1 行 1 バイト（`shape = [バイト長]`）— どんな行数でも 4 バイト整列の刻みが
        取れるので、分割の可否ではなく**詰め方**だけを試せる。見るのは規則が述べている性質:
        全 shard が容量以下・weight shard は非空・決定的・分割として整合（piece が index 順に
        連続 shard へ 1 本ずつ・行を隙間なく覆う = `assert_shard_partition`）・**宣言順が
        保たれる**（piece は親の位置に固まって並ぶ）。
        """
        rng = random.Random(seed)
        capacity = rng.randint(8, 40)
        order = [f"t{index:03d}" for index in range(rng.randint(1, 60))]
        sizes = {
            name: (
                rng.randint(capacity + 1, 4 * capacity)
                if rng.randrange(6) == 0
                else rng.randint(1, capacity)
            )
            for name in order
        }
        shapes = {name: (size,) for name, size in sizes.items()}

        groups = pack_shards(order, sizes, shapes, {}, capacity=capacity)

        assert groups[0] == ()
        assert all(group for group in groups[1:])
        assert groups == pack_shards(order, sizes, shapes, {}, capacity=capacity)
        assert_shard_partition(groups, order, shapes)
        for group in groups:
            # 1 行 1 バイトなので、piece の行数がそのままバイト数。
            assert sum(_member_bytes(member, sizes) for member in group) <= capacity
        flat = [
            member.name if isinstance(member, Piece) else member
            for group in groups
            for member in group
        ]
        runs = [name for index, name in enumerate(flat) if index == 0 or flat[index - 1] != name]
        assert runs == order


class TestCoShard:
    """weight と companion scale は同一 shard MUST（ADR 0070 決定 1）。"""

    def test_the_partner_is_pulled_into_the_same_shard(self) -> None:
        """順序上あとに居る相方を引き寄せる（scale は F32 群で weight より前に来る）。"""
        groups = pack_shards(
            ["karume.scale.w", "b", "w"],
            {"karume.scale.w": 2, "b": 3, "w": 5},
            {},
            PAIR,
            capacity=8,
        )

        assert groups == [(), ("karume.scale.w", "w"), ("b",)]

    def test_a_pulled_partner_is_not_written_twice(self) -> None:
        """引き寄せた相方は自分の順番が来ても再登場しない（重複は宣言完全性の破れ）。"""
        groups = pack_shards(
            ["karume.scale.w", "w"], {"karume.scale.w": 2, "w": 5}, {}, PAIR, capacity=100
        )

        assert groups == [(), ("karume.scale.w", "w")]

    def test_a_scale_that_alone_fills_the_capacity_fails_loudly(self) -> None:
        """scale は割らない（読み手契約 4）— piece 1 と同居する余地が無ければ fail loudly。

        重みは piece へ割れるが、相方が容量を丸ごと埋めていると piece 1 の置き場が 1 バイトも
        残らない。黙って容量を破るか scale を割るしかなくなるので、ここで落とす。
        """
        with pytest.raises(ShardError, match="companion scale は割らない"):
            pack_shards(
                ["karume.scale.w", "w"],
                {"karume.scale.w": 8, "w": 9},
                {"karume.scale.w": (8,), "w": (9,)},
                PAIR,
                capacity=8,
            )

    def test_the_check_catches_a_split_pair_that_the_rule_would_never_make(self) -> None:
        """規則と検査は別物 — 手で割り付けた（= 別実装が書いた）列を受け止める。"""
        with pytest.raises(ShardError, match="同一 shard MUST"):
            assert_co_shard([("w",), ("karume.scale.w",)], PAIR)

    def test_the_check_passes_when_the_pair_shares_a_shard(self) -> None:
        assert_co_shard([("w", "karume.scale.w"), ("other",)], PAIR)

    def test_the_scale_belongs_with_the_first_piece_of_a_split_weight(self) -> None:
        """分割された重みの所属は **piece 1** の shard（co-shard の piece 版）。"""
        first = Piece(name="w", index=1, count=2, begin=0, end=4)
        second = Piece(name="w", index=2, count=2, begin=4, end=8)

        assert_co_shard([("karume.scale.w", first), (second,)], PAIR)

    def test_a_scale_parked_with_a_later_piece_fails_loudly(self) -> None:
        """piece 2 と同居する scale は「重みの先頭を読む時点で scale が無い」形になる。"""
        first = Piece(name="w", index=1, count=2, begin=0, end=4)
        second = Piece(name="w", index=2, count=2, begin=4, end=8)

        with pytest.raises(ShardError, match="同一 shard MUST"):
            assert_co_shard([(first,), ("karume.scale.w", second)], PAIR)


class TestPartition:
    """全 shard の和 = 元の全テンソル（欠け・重複・余剰なし）。"""

    def test_the_packed_groups_cover_every_tensor_exactly_once(self) -> None:
        order = ["karume.scale.w", "b", "w", "c"]
        sizes = dict.fromkeys(order, 3)
        groups = pack_shards(order, sizes, {}, PAIR, capacity=6)

        assert_shard_partition(groups, order, {})
        assert sorted(name for group in groups for name in group) == sorted(order)

    def test_a_duplicated_tensor_fails_loudly(self) -> None:
        with pytest.raises(ShardError, match="重複"):
            assert_shard_partition([("a", "b"), ("b",)], ["a", "b"], {})

    def test_a_missing_tensor_fails_loudly(self) -> None:
        with pytest.raises(ShardError, match=r"欠け \['b'\]"):
            assert_shard_partition([("a",)], ["a", "b"], {})

    def test_a_surplus_tensor_fails_loudly(self) -> None:
        with pytest.raises(ShardError, match=r"余剰 \['c'\]"):
            assert_shard_partition([("a", "c")], ["a"], {})


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


class TestResolvingTheShardsOnDisk:
    """組み立ての入口は「代表 path → 実在する shard 列」の 1 本（分割の有無は現物が決める）。"""

    def test_an_unsplit_component_resolves_to_itself(self, tmp_path: Path) -> None:
        path = _touch(tmp_path / "model.safetensors")

        assert resolve_shards(path) == (path,)

    def test_a_missing_component_resolves_to_itself(self, tmp_path: Path) -> None:
        """不在の診断は呼び手の既存の門が出す（綴りを 2 つに割らない）。"""
        path = tmp_path / "model.safetensors"

        assert resolve_shards(path) == (path,)

    def test_a_split_component_resolves_in_shard_order(self, tmp_path: Path) -> None:
        """並びは番号順（先頭 = グラフ shard）— ディレクトリの列挙順に依存しない。"""
        third = _touch(tmp_path / "model-00003-of-00003.safetensors")
        first = _touch(tmp_path / "model-00001-of-00003.safetensors")
        second = _touch(tmp_path / "model-00002-of-00003.safetensors")

        assert resolve_shards(tmp_path / "model.safetensors") == (first, second, third)

    def test_it_ignores_another_components_shards(self, tmp_path: Path) -> None:
        """stem が違うファイルは拾わない（系列には `io.*.safetensors` も同居する）。"""
        _touch(tmp_path / "io.input-00001-of-00001.safetensors")
        path = _touch(tmp_path / "model.safetensors")

        assert resolve_shards(path) == (path,)

    def test_the_single_file_and_the_sequence_together_fail_loudly(self, tmp_path: Path) -> None:
        """前回の書き出しの残骸 — どちらを配るかが一意に決まらない。"""
        _touch(tmp_path / "model.safetensors")
        _touch(tmp_path / "model-00001-of-00002.safetensors")
        _touch(tmp_path / "model-00002-of-00002.safetensors")

        with pytest.raises(ShardError, match="同居"):
            resolve_shards(tmp_path / "model.safetensors")

    def test_a_gap_in_the_sequence_fails_loudly(self, tmp_path: Path) -> None:
        _touch(tmp_path / "model-00001-of-00003.safetensors")
        _touch(tmp_path / "model-00003-of-00003.safetensors")

        with pytest.raises(ShardError, match=r"欠け \[2\]"):
            resolve_shards(tmp_path / "model.safetensors")

    def test_disagreeing_totals_fail_loudly(self, tmp_path: Path) -> None:
        """分割数が変わった再 export の残骸（`-of-` が食い違う）。"""
        _touch(tmp_path / "model-00001-of-00002.safetensors")
        _touch(tmp_path / "model-00002-of-00003.safetensors")

        with pytest.raises(ShardError, match="総数"):
            resolve_shards(tmp_path / "model.safetensors")

    def test_the_siblings_include_every_output_name_of_the_component(self, tmp_path: Path) -> None:
        """後片付けは**壊れた残骸ほど拾えなければ困る**ので、名前の形だけで拾う。"""
        single = _touch(tmp_path / "model.safetensors")
        stale = _touch(tmp_path / "model-00002-of-00007.safetensors")
        _touch(tmp_path / "other.safetensors")

        assert set(shard_siblings(tmp_path / "model.safetensors")) == {single, stale}


class TestPieceKeys:
    """piece キーの綴りは読み手契約（TS ランタイムの `parsePieceKey` と 1 文字も違わない）。"""

    def test_the_key_is_the_parent_name_with_a_five_digit_run(self) -> None:
        assert piece_key("enc.w", 1, 12) == "enc.w#00001-of-00012"

    @pytest.mark.parametrize(("index", "count"), [(1, 2), (7, 7), (3, 99999)])
    def test_it_round_trips(self, index: int, count: int) -> None:
        assert parse_piece_key(piece_key("enc.w", index, count)) == ("enc.w", index, count)

    @pytest.mark.parametrize(
        "key",
        [
            "enc.w",
            "enc.w#1-of-3",
            "enc.w#00001of00003",
            "enc.w#00001-of-3",
            "#00001-of-00003",
        ],
    )
    def test_a_key_that_is_not_a_piece_parses_to_nothing(self, key: str) -> None:
        """形が違えば piece ではない（= 普通のテンソル名）。"""
        assert parse_piece_key(key) is None

    @pytest.mark.parametrize("key", ["enc.w#00003-of-00002", "enc.w#00000-of-00002"])
    def test_an_out_of_range_run_is_not_a_piece_either(self, key: str) -> None:
        """域外の綴りは**キー自体の誤り**なので piece と解釈しない。

        通すと違反の帰属が「piece 列の並び」へ移るが、直すべきはそのキーそのもの — 読み手側は
        余剰テンソル（どの initializer からも参照されない）として落とす。
        """
        assert parse_piece_key(key) is None

    def test_a_single_piece_run_is_refused_by_the_writer(self) -> None:
        with pytest.raises(ShardError, match="2 未満"):
            piece_key("enc.w", 1, 1)

    def test_the_piece_carries_its_own_key(self) -> None:
        assert Piece(name="enc.w", index=2, count=3, begin=4, end=8).key == "enc.w#00002-of-00003"


class TestSplittingOversizedTensors:
    """容量に収まらないテンソルは**先頭次元（行）**で割る（ADR 0090 決定 1）。

    合成は「行バイト長 × 行数」で綴る — 分割の可否も刻みも行バイト長で決まるので、ここでは
    dtype も実データも要らない。
    """

    def test_it_becomes_a_run_of_pieces_on_consecutive_shards(self) -> None:
        """40 バイト（4 バイト × 10 行）を容量 16 へ → 4 行ずつの piece が 3 本。"""
        groups = pack_shards(["w"], {"w": 40}, {"w": (10, 1)}, {}, capacity=16)

        assert groups == [
            (),
            (Piece(name="w", index=1, count=3, begin=0, end=4),),
            (Piece(name="w", index=2, count=3, begin=4, end=8),),
            (Piece(name="w", index=3, count=3, begin=8, end=10),),
        ]

    def test_the_pieces_cover_every_row_exactly_once(self) -> None:
        groups = pack_shards(["w"], {"w": 40}, {"w": (10, 1)}, {}, capacity=16)

        assert_shard_partition(groups, ["w"], {"w": (10, 1)})

    def test_a_tensor_that_fits_the_capacity_is_never_split(self) -> None:
        """容量以下は丸ごとのまま — 小さいテンソルが shard 境界で piece に化けない。"""
        groups = pack_shards(["a", "w"], {"a": 8, "w": 8}, {"w": (2, 4)}, {}, capacity=8)

        assert groups == [(), ("a",), ("w",)]

    def test_the_scale_rides_with_the_first_piece_and_the_rest_follows(self) -> None:
        """鎖の引き寄せ — scale が並びの先頭に居ても piece が別々の shard へ散らない。

        `[scale + ブロック 1], [ブロック 2], …` を連続した単位列として置くので、piece は
        scale の位置から連続 shard へ並ぶ。ブロック 1 だけを引き寄せる素朴な規則だと、残りの
        ブロックが末尾群（重みの dtype 群）に取り残されて piece が飛び飛びの shard に載る。
        """
        order = ["karume.scale.w", "b", "w"]
        sizes = {"karume.scale.w": 8, "b": 4, "w": 32}
        shapes = {"karume.scale.w": (8, 1), "b": (4,), "w": (8, 4)}

        groups = pack_shards(order, sizes, shapes, PAIR, capacity=16)

        assert groups[1][0] == "karume.scale.w"
        # 同じ shard に落ちた連続ブロックは 1 piece へ畳まれるので、本数は「shard を跨いだ
        # 回数」ちょうど（刻みの細かさは読み手に漏れない）。
        pieces = [member for group in groups for member in group if isinstance(member, Piece)]
        assert [piece.index for piece in pieces] == [1, 2, 3]
        # piece 1 の shard には相方の scale が、piece n の shard には後続の `b` が同居する
        # （読み手契約 5 が明示的に許している形）。
        assert [len(group) for group in groups] == [0, 2, 1, 2]
        assert groups[-1][-1] == "b"
        assert_shard_partition(groups, order, shapes)
        assert_co_shard(groups, PAIR)

    @pytest.mark.parametrize(("row_bytes", "rows"), [(1, 40), (2, 20), (3, 16), (6, 10)])
    def test_every_piece_but_the_last_is_a_multiple_of_four_bytes(
        self, row_bytes: int, rows: int
    ) -> None:
        """末尾以外の piece は 4 の倍数長 MUST（読み手がオフセット書きする — 読み手契約 5）。

        行バイト長が奇数 / ≡2 mod 4 のときは 1 行刻みでは整列しないので、ブロックの行数が
        4 行 / 2 行へ切り上がる。末尾だけは端数のままでよい。
        """
        nbytes = row_bytes * rows

        groups = pack_shards(["w"], {"w": nbytes}, {"w": (rows, 1)}, {}, capacity=13)

        pieces = [member for group in groups for member in group if isinstance(member, Piece)]
        assert len(pieces) >= 2
        assert all((piece.end - piece.begin) * row_bytes % 4 == 0 for piece in pieces[:-1])
        assert sum(piece.end - piece.begin for piece in pieces) == rows

    def test_a_row_that_does_not_fit_the_capacity_fails_loudly(self) -> None:
        """これ以上細かい粒度が無いので、黙って容量を破らない。"""
        with pytest.raises(ShardError, match="これ以上細かく割れない"):
            pack_shards(["w"], {"w": 40}, {"w": (2, 5)}, {}, capacity=16)

    def test_an_odd_row_length_that_cannot_reach_four_byte_alignment_fails_loudly(self) -> None:
        """行バイト長 5 は 4 行で 20 バイト — 容量 16 には 1 ブロックも入らない。"""
        with pytest.raises(ShardError, match="4 バイト整列には 4 行"):
            pack_shards(["w"], {"w": 50}, {"w": (10, 1)}, {}, capacity=16)

    def test_a_rank_zero_tensor_cannot_be_split(self) -> None:
        with pytest.raises(ShardError, match="rank 0"):
            pack_shards(["w"], {"w": 40}, {"w": ()}, {}, capacity=16)

    def test_a_row_length_that_is_not_a_whole_number_of_bytes_fails_loudly(self) -> None:
        """`40 バイト / 3 行` は 1 行が整数バイトでない — 行では割れない。"""
        with pytest.raises(ShardError, match="割り切れない"):
            pack_shards(["w"], {"w": 40}, {"w": (3, 1)}, {}, capacity=16)

    def test_a_missing_shape_fails_loudly(self) -> None:
        """分割が要るのに shape が渡っていない = 呼び手の配線ミス（黙って詰め込まない）。"""
        with pytest.raises(ShardError, match="shape が分からない"):
            pack_shards(["w"], {"w": 40}, {}, {}, capacity=16)


class TestThePartitionOfPieces:
    """piece 列の整合（読み手契約 5）は**検査**の側でも見る（規則の写経ではなく門）。"""

    SHAPES: ClassVar[dict[str, tuple[int, ...]]] = {"w": (8, 1)}

    def run(self, *placed: tuple[int, Piece]) -> list[tuple[str | Piece, ...]]:
        """`(shard 添字, piece)` から手組みの割り付けを作る（規則を通さない列）。"""
        groups: list[list[str | Piece]] = [[] for _ in range(1 + max(index for index, _ in placed))]
        for index, piece in placed:
            groups[index].append(piece)
        return [tuple(group) for group in groups]

    def test_a_well_formed_run_passes(self) -> None:
        groups = self.run(
            (1, Piece(name="w", index=1, count=2, begin=0, end=4)),
            (2, Piece(name="w", index=2, count=2, begin=4, end=8)),
        )

        assert_shard_partition(groups, ["w"], self.SHAPES)

    def test_a_missing_piece_fails_loudly(self) -> None:
        groups = self.run((1, Piece(name="w", index=1, count=2, begin=0, end=4)))

        with pytest.raises(ShardError, match="piece が 1 本で宣言の総数 2"):
            assert_shard_partition(groups, ["w"], self.SHAPES)

    def test_a_gap_between_the_shards_fails_loudly(self) -> None:
        groups = self.run(
            (1, Piece(name="w", index=1, count=2, begin=0, end=4)),
            (3, Piece(name="w", index=2, count=2, begin=4, end=8)),
        )

        with pytest.raises(ShardError, match="連続する shard に 1 本ずつ"):
            assert_shard_partition(groups, ["w"], self.SHAPES)

    def test_two_pieces_of_one_parent_in_the_same_shard_fail_loudly(self) -> None:
        groups = self.run(
            (1, Piece(name="w", index=1, count=2, begin=0, end=4)),
            (1, Piece(name="w", index=2, count=2, begin=4, end=8)),
        )

        with pytest.raises(ShardError, match="連続する shard に 1 本ずつ"):
            assert_shard_partition(groups, ["w"], self.SHAPES)

    def test_a_hole_in_the_rows_fails_loudly(self) -> None:
        """行の被覆は隙間なし MUST — 抜けた行は「読めたのに値が別物」になる。"""
        groups = self.run(
            (1, Piece(name="w", index=1, count=2, begin=0, end=3)),
            (2, Piece(name="w", index=2, count=2, begin=4, end=8)),
        )

        with pytest.raises(ShardError, match="隙間なく覆う"):
            assert_shard_partition(groups, ["w"], self.SHAPES)

    def test_a_run_that_stops_short_of_the_last_row_fails_loudly(self) -> None:
        groups = self.run(
            (1, Piece(name="w", index=1, count=2, begin=0, end=4)),
            (2, Piece(name="w", index=2, count=2, begin=4, end=7)),
        )

        with pytest.raises(ShardError, match="行 7 までしか覆っていない"):
            assert_shard_partition(groups, ["w"], self.SHAPES)

    def test_mixing_a_whole_tensor_with_pieces_fails_loudly(self) -> None:
        groups: list[tuple[str | Piece, ...]] = [
            (),
            ("w",),
            (Piece(name="w", index=1, count=2, begin=0, end=4),),
            (Piece(name="w", index=2, count=2, begin=4, end=8),),
        ]

        with pytest.raises(ShardError, match="どちらか一方"):
            assert_shard_partition(groups, ["w"], self.SHAPES)

    def test_an_empty_piece_fails_loudly(self) -> None:
        groups = self.run(
            (1, Piece(name="w", index=1, count=2, begin=0, end=0)),
            (2, Piece(name="w", index=2, count=2, begin=0, end=8)),
        )

        with pytest.raises(ShardError, match="1 行以上"):
            assert_shard_partition(groups, ["w"], self.SHAPES)
