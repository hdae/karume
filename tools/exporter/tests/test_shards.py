"""配布コンテナの shard 分割規則（ADR 0070 決定 1 / ADR 0071 決定 4 の解除）。

ここが見るのは**規則そのもの**（名前・詰め方・検査）だけで、実際に書けるかは
`test_emit.py`、宣言に載るかは `test_dist.py` が持つ。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from karume.shards import (
    MAX_SHARDS,
    SHARD_BYTE_LIMIT,
    SHARD_TAIL_LIMIT,
    ShardError,
    assert_co_shard,
    assert_shard_partition,
    pack_shards,
    resolve_shards,
    shard_name,
    shard_path,
    shard_siblings,
)

#: 対の写像は**対称**で持つ（weight → scale と scale → weight）。
PAIR = {"w": "karume.scale.w", "karume.scale.w": "w"}

MIB = 1024 * 1024


def uniform(total_mib: int) -> tuple[list[str], dict[str, int]]:
    """1MiB のテンソル `total_mib` 本（GiB 規模の分割を実データ無しで踏むための合成）。"""
    order = [f"t{index:05d}" for index in range(total_mib)]
    return order, dict.fromkeys(order, MIB)


def shard_mib(groups: list[tuple[str, ...]], sizes: dict[str, int]) -> list[int]:
    """shard ごとのデータ節を MiB で（`1024` = 1GiB）。"""
    return [sum(sizes[name] for name in group) // MIB for group in groups]


class TestTheByteLimit:
    """上限は固定定数（配布形の不変条件であって呼び手の好みではない）。"""

    def test_it_is_one_gibibyte(self) -> None:
        assert SHARD_BYTE_LIMIT == 1_073_741_824

    def test_it_leaves_room_under_the_single_arraybuffer_ceiling(self) -> None:
        """Chromium の単一 `ArrayBuffer` 上限（docs/limitations）を割るのが必要条件。

        ヘッダは上限に数えない（`karume.shards` のモジュール doc）ので、天井との差が
        ヘッダぶんの余裕そのもの — ここが縮むと「データ節は上限内なのにファイルが載らない」
        形が生まれる。余裕は 1GB 級（グラフ JSON は実測で数 MB 級）。
        """
        ceiling = 2_145_386_496

        assert ceiling - SHARD_BYTE_LIMIT >= 1_000_000_000


class TestTheTailLimit:
    """最後の shard にだけ許す上限（端数 shard を作らないための尾部スラック）。"""

    def test_it_is_the_same_1_5_gibibytes_as_the_hub_ram_budget(self) -> None:
        """hub の同時 RAM 予算（`fetch.ts` の in-flight バイト予算）と**同値** — 予算を広げない。"""
        assert SHARD_TAIL_LIMIT == 1_610_612_736

    def test_it_is_wider_than_the_cut_limit(self) -> None:
        """尾部が切り口より狭いと「畳んだ結果が上限超え」になる（規則が成り立つ必要条件）。"""
        assert SHARD_TAIL_LIMIT > SHARD_BYTE_LIMIT

    def test_it_still_leaves_room_under_the_single_arraybuffer_ceiling(self) -> None:
        """壁の約 75%。最大の shard がこの値になるので、天井を割るのはここでも必要条件。"""
        ceiling = 2_145_386_496

        assert ceiling - SHARD_TAIL_LIMIT >= 500_000_000


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
    """順序は変えず、対は原子、上限を超える手前で次を開く。

    尾部スラックを外した素のカット規則を見るので、`tail_limit` は `limit` と同値で渡す
    （スラックそのものは {@link TestTheTailSlack}）。
    """

    def test_everything_stays_in_one_shard_under_the_limit(self) -> None:
        """総量が上限以下なら 1 本のまま（= 単一ファイル配布の形が変わらない）。"""
        groups = pack_shards(["a", "b", "c"], {"a": 4, "b": 4, "c": 4}, {}, limit=100)

        assert groups == [("a", "b", "c")]

    def test_it_opens_a_new_shard_at_the_byte_where_the_limit_would_break(self) -> None:
        groups = pack_shards(["a", "b", "c"], {"a": 4, "b": 4, "c": 4}, {}, limit=8, tail_limit=8)

        assert groups == [("a", "b"), ("c",)]

    def test_a_tensor_that_exactly_fills_the_limit_stays_in_place(self) -> None:
        """境界は「超えたら」であって「届いたら」ではない（`used + size > limit`）。"""
        groups = pack_shards(["a", "b"], {"a": 8, "b": 1}, {}, limit=8, tail_limit=8)

        assert groups == [("a",), ("b",)]

    def test_it_keeps_the_given_order_inside_the_shards(self) -> None:
        """並べ替えない — 詰める順は書き手が決めた書き出し順そのもの（ADR 0063）。"""
        groups = pack_shards(["c", "a", "b"], {"a": 1, "b": 1, "c": 1}, {}, limit=100)

        assert groups == [("c", "a", "b")]

    def test_the_same_input_always_splits_the_same_way(self) -> None:
        """決定的（同入力 → 同分割）— 再 dist で sha256 が揺れないことの前提。"""
        order = ["a", "w", "b", "karume.scale.w", "c"]
        sizes = {"a": 3, "w": 5, "b": 3, "karume.scale.w": 2, "c": 4}

        first = pack_shards(order, sizes, PAIR, limit=8, tail_limit=8)
        second = pack_shards(order, sizes, PAIR, limit=8, tail_limit=8)

        assert first == second

    def test_it_returns_one_empty_shard_for_a_graph_without_tensors(self) -> None:
        """テンソルが 1 本も無くてもグラフ shard は在る（`karume_ir` を載せる器）。"""
        assert pack_shards([], {}, {}, limit=8) == [()]


class TestTheTailSlack:
    """最後の shard は {@link SHARD_TAIL_LIMIT} まで許して端数を作らない（規則 4）。

    実 GiB を踏むテストは書けないので、1MiB のテンソルを並べて総量だけ実寸に合わせる
    （上限は既定 = 実際に配布で使う 1GiB / 1.5GiB のまま）。
    """

    def test_it_folds_the_tail_into_the_last_shard_instead_of_a_stub(self) -> None:
        """3.2GiB → `[1GiB, 1GiB, 1.2GiB]`（ユーザー裁定 2026-08-29 の検算 1）。"""
        order, sizes = uniform(3277)

        groups = pack_shards(order, sizes, {})

        assert shard_mib(groups, sizes) == [1024, 1024, 1229]

    def test_a_tail_still_over_the_slack_is_cut_as_before(self) -> None:
        """2.6GiB → `[1GiB, 1GiB, 0.6GiB]`（検算 2）。

        端数 0.6GiB は残る — 1.6GiB を最後の shard にすると尾部上限を破るので、これが
        正しい帰結（判定点での未閉は 2.6 / 1.6GiB でどちらも 1.5GiB 超）。
        """
        order, sizes = uniform(2662)

        groups = pack_shards(order, sizes, {})

        assert shard_mib(groups, sizes) == [1024, 1024, 614]

    def test_an_asset_under_the_slack_is_never_split(self) -> None:
        """1.11GiB（text_encoder 相当）は単一のまま — 1GiB 超でも分割ゼロ。"""
        order, sizes = uniform(1137)

        groups = pack_shards(order, sizes, {})

        assert shard_mib(groups, sizes) == [1137]

    def test_a_tail_that_was_already_whole_is_untouched(self) -> None:
        """1.96GiB → `[1GiB, 0.96GiB]` は従来と同一（スラック無しの割り付けと一致する）。"""
        order, sizes = uniform(2007)

        groups = pack_shards(order, sizes, {})

        assert shard_mib(groups, sizes) == [1024, 983]
        assert groups == pack_shards(order, sizes, {}, tail_limit=SHARD_BYTE_LIMIT)

    def test_an_asset_under_the_cut_limit_keeps_its_single_shard(self) -> None:
        """総量が 1GiB 以下ならカット判定自体が起きない（既存資産の再 dist がバイト不変）。"""
        order, sizes = uniform(1024)

        assert pack_shards(order, sizes, {}) == [tuple(order)]

    def test_the_folded_tail_keeps_a_pair_together(self) -> None:
        """畳んだ尾部でも対は原子（畳みを「新しい shard を開く」で書くと対を割りうる）。"""
        order = ["a", "karume.scale.w", "b", "w"]
        sizes = {"a": 5, "karume.scale.w": 2, "b": 4, "w": 3}

        folded = pack_shards(order, sizes, PAIR, limit=8, tail_limit=12)

        assert folded == [("a",), ("karume.scale.w", "w", "b")]
        assert_co_shard(folded, PAIR)
        # スラック無しなら端数の `b` が 3 本目に落ちる（= 畳みが効いていることの対照）。
        assert pack_shards(order, sizes, PAIR, limit=8, tail_limit=8) == [
            ("a",),
            ("karume.scale.w", "w"),
            ("b",),
        ]

    def test_a_tail_limit_under_the_cut_limit_fails_loudly(self) -> None:
        """尾部が切り口より狭い割り付けは、畳んだ瞬間に自分の上限を破る（規則が壊れる）。"""
        with pytest.raises(ShardError, match="尾部の上限"):
            pack_shards(["a"], {"a": 1}, {}, limit=8, tail_limit=7)


class TestCoShard:
    """weight と companion scale は同一 shard MUST（ADR 0070 決定 1）。"""

    def test_the_partner_is_pulled_into_the_same_shard(self) -> None:
        """順序上あとに居る相方を引き寄せる（scale は F32 群で weight より前に来る）。"""
        groups = pack_shards(
            ["karume.scale.w", "b", "w"],
            {"karume.scale.w": 2, "b": 3, "w": 5},
            PAIR,
            limit=8,
            tail_limit=8,
        )

        assert groups == [("karume.scale.w", "w"), ("b",)]

    def test_a_pulled_partner_is_not_written_twice(self) -> None:
        """引き寄せた相方は自分の順番が来ても再登場しない（重複は宣言完全性の破れ）。"""
        groups = pack_shards(
            ["karume.scale.w", "w"], {"karume.scale.w": 2, "w": 5}, PAIR, limit=100
        )

        assert groups == [("karume.scale.w", "w")]

    def test_a_pair_that_alone_exceeds_the_limit_fails_loudly(self) -> None:
        """1 対は分割できないので、次の shard へ送っても同じ — 黙って上限を破らない。"""
        with pytest.raises(ShardError, match="1 対は分割できない"):
            pack_shards(["karume.scale.w", "w"], {"karume.scale.w": 2, "w": 9}, PAIR, limit=8)

    def test_the_check_catches_a_split_pair_that_the_rule_would_never_make(self) -> None:
        """規則と検査は別物 — 手で割り付けた（= 別実装が書いた）列を受け止める。"""
        with pytest.raises(ShardError, match="同一 shard MUST"):
            assert_co_shard([("w",), ("karume.scale.w",)], PAIR)

    def test_the_check_passes_when_the_pair_shares_a_shard(self) -> None:
        assert_co_shard([("w", "karume.scale.w"), ("other",)], PAIR)


class TestPartition:
    """全 shard の和 = 元の全テンソル（欠け・重複・余剰なし）。"""

    def test_the_packed_groups_cover_every_tensor_exactly_once(self) -> None:
        order = ["karume.scale.w", "b", "w", "c"]
        groups = pack_shards(order, dict.fromkeys(order, 3), PAIR, limit=6, tail_limit=6)

        assert_shard_partition(groups, order)
        assert sorted(name for group in groups for name in group) == sorted(order)

    def test_a_duplicated_tensor_fails_loudly(self) -> None:
        with pytest.raises(ShardError, match="重複"):
            assert_shard_partition([("a", "b"), ("b",)], ["a", "b"])

    def test_a_missing_tensor_fails_loudly(self) -> None:
        with pytest.raises(ShardError, match=r"欠け \['b'\]"):
            assert_shard_partition([("a",)], ["a", "b"])

    def test_a_surplus_tensor_fails_loudly(self) -> None:
        with pytest.raises(ShardError, match=r"余剰 \['c'\]"):
            assert_shard_partition([("a", "c")], ["a"])


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
