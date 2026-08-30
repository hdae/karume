"""容器の詰め替え（`karume.repack`）— 旧規則の配布形を shard 仕様 v2 へ移す。

被験体の**旧レイアウト**は手組みする: 旧 writer はもう無い（v2 は常時分割で、fat グラフ
shard も単一ファイルも書けない — ADR 0081）ので、器の低レベル面
（`emit.write_container`）だけを借りて「どのテンソルがどのファイルに居るか」をここが決める。
中身そのものは {@link ir_fixtures.ir_container} が v2 の書き手で作った**本物**なので、
「詰め替えたら v2 の書き手が書いたのと同じバイト列になる」を突き合わせられる。
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from ir_fixtures import ir_container

from karume import repack
from karume.emit import ContainerEntry, container_order, write_container
from karume.ir import IR_METADATA_KEY
from karume.shards import pack_shards, shard_name, shard_path
from karume.verify import ContainerError, verify_shards

#: コンポーネントの代表 path のファイル名（系列の綴りと同じ）。
COMPONENT = "model.safetensors"

#: 合成資産（f32 で 660 バイト）を weight shard 2 本へ割る上限 — 詰め替えの前後で本数が
#: 変わる形（= 前回の連番が残骸になる形）を踏むための差し込み。
SPLIT_LIMIT = 512

#: 名前 → (safetensors dtype, 論理 shape, 生バイト)。
Payload = dict[str, tuple[str, tuple[int, ...], bytes]]


def parse_container(blob: bytes) -> tuple[dict[str, str], Payload]:
    """safetensors のバイト列を `(__metadata__, 名前 → 宣言と生バイト)` へ開く。"""
    length = struct.unpack("<Q", blob[:8])[0]
    header = json.loads(blob[8 : 8 + length])
    start = 8 + length
    found: Payload = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        begin, end = spec["data_offsets"]
        found[name] = (spec["dtype"], tuple(spec["shape"]), blob[start + begin : start + end])
    return dict(header.get("__metadata__", {})), found


def source_material(shards: Sequence[bytes]) -> tuple[dict[str, str], Payload]:
    """v2 の合成コンテナ（shard 列）を 1 つの素材へ畳む（メタデータは先頭 shard のもの）。"""
    metadata, tensors = parse_container(shards[0])
    for blob in shards[1:]:
        _, found = parse_container(blob)
        tensors.update(found)
    return metadata, tensors


def write_legacy(
    path: Path, metadata: Mapping[str, str], tensors: Payload, groups: Sequence[Sequence[str]]
) -> list[Path]:
    """旧レイアウトを手組みする（1 群なら単一ファイル・複数群なら fat グラフ shard + 続き）。

    並びは群の中だけ v2 と同じ規約（`container_order`）で満たす — 旧規則でもリーダの整列規則は
    同じなので、被験体は「読める旧配布形」でなければ意味が無い。
    """
    total = len(groups)
    written: list[Path] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    for index, group in enumerate(groups, start=1):
        entries = container_order(
            ContainerEntry(
                name=name,
                dtype=tensors[name][0],
                shape=tensors[name][1],
                nbytes=len(tensors[name][2]),
            )
            for name in group
        )
        target = path if total == 1 else shard_path(path, index, total)
        write_container(
            target, entries, dict(metadata) if index == 1 else {}, lambda e: [tensors[e.name][2]]
        )
        written.append(target)
    return written


def legacy_single(directory: Path, shards: Sequence[bytes]) -> Path:
    """旧・単一ファイル配布形（`karume_ir` と全テンソルが 1 つの器に同居）。"""
    metadata, tensors = source_material(shards)
    write_legacy(directory / COMPONENT, metadata, tensors, [sorted(tensors)])
    return directory / COMPONENT


def legacy_tail_slack(directory: Path, shards: Sequence[bytes]) -> Path:
    """旧・尾部スラック形（fat グラフ shard + 残りの weight shard 1 本）。"""
    metadata, tensors = source_material(shards)
    names = sorted(tensors)
    cut = len(names) // 2
    # 先頭群が空だと「fat グラフ shard」ではなく v2 のグラフ shard になり、被験体が旧形で
    # なくなる（合成資産のテンソル本数が減った日に黙って別のものを試さない）。
    assert cut >= 1, f"fat グラフ shard に載せるテンソルが無い: {names}"
    write_legacy(directory / COMPONENT, metadata, tensors, [names[:cut], names[cut:]])
    return directory / COMPONENT


def v2_component(directory: Path, shards: Sequence[bytes]) -> Path:
    """v2 の書き手が書いたままの連番を置く（詰め替えの冪等性の被験体）。"""
    directory.mkdir(parents=True, exist_ok=True)
    for index, blob in enumerate(shards, start=1):
        (directory / shard_name(COMPONENT, index, len(shards))).write_bytes(blob)
    return directory / COMPONENT


def listing(directory: Path) -> dict[str, str]:
    """ディレクトリの現物（ファイル名 → sha256）— 触っていないことの観測点。"""
    return {
        entry.name: hashlib.sha256(entry.read_bytes()).hexdigest()
        for entry in sorted(directory.iterdir())
        if entry.is_file()
    }


def tensors_of(paths: Sequence[Path]) -> Payload:
    """shard 列の全テンソル（名前 → 宣言と生バイト）。"""
    found: Payload = {}
    for path in paths:
        _, part = parse_container(path.read_bytes())
        found.update(part)
    return found


def metadata_of(path: Path) -> dict[str, str]:
    return parse_container(path.read_bytes())[0]


class TestRepackingALegacySingleFile:
    """単一ファイル配布形 → グラフ shard + weight shard 列。"""

    def test_the_component_becomes_a_numbered_sequence(self, tmp_path):
        series = tmp_path / "series"
        path = legacy_single(series, ir_container())

        published = repack.repack_component(path)

        assert [entry.name for entry in published] == [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]
        assert sorted(listing(series)) == [entry.name for entry in published]

    def test_the_old_single_file_is_gone(self, tmp_path):
        """旧ファイルが残ると `resolve_shards` が「単一と連番の同居」で落ちる。"""
        series = tmp_path / "series"
        path = legacy_single(series, ir_container())

        repack.repack_component(path)

        assert not path.exists()

    def test_every_tensor_keeps_its_bytes(self, tmp_path):
        series = tmp_path / "series"
        path = legacy_single(series, ir_container(storage="i8"))
        before = tensors_of([path])

        published = repack.repack_component(path)

        assert tensors_of(published) == before

    def test_the_graph_metadata_stays_verbatim(self, tmp_path):
        """`karume_ir` は文字列のまま運ぶ（parse → 再 serialize しない）。"""
        series = tmp_path / "series"
        path = legacy_single(series, ir_container(storage="i8"))
        before = metadata_of(path)

        published = repack.repack_component(path)

        assert metadata_of(published[0]) == before
        assert IR_METADATA_KEY in before

    def test_the_graph_shard_carries_no_tensor(self, tmp_path):
        """v2 の読み手契約 1 — 先頭は `karume_ir` だけを載せる器。"""
        series = tmp_path / "series"
        path = legacy_single(series, ir_container())

        published = repack.repack_component(path)

        assert tensors_of(published[:1]) == {}
        assert metadata_of(published[1]) == {}

    def test_the_result_passes_the_full_verification(self, tmp_path):
        series = tmp_path / "series"
        path = legacy_single(series, ir_container(storage="i8"))

        published = repack.repack_component(path)

        verify_shards(published)  # 例外が出なければ合格

    def test_the_result_is_byte_identical_to_what_the_writer_would_have_written(self, tmp_path):
        """詰め替えの出力 = 同じ資産を v2 の書き手が書いたバイト列（器の綴りが 1 本道）。"""
        series = tmp_path / "series"
        shards = ir_container(storage="i8")
        path = legacy_single(series, shards)

        published = repack.repack_component(path)

        assert [entry.read_bytes() for entry in published] == list(shards)

    def test_the_neighbours_are_left_alone(self, tmp_path):
        """同居する io フィクスチャ（`io.*.safetensors`）には触れない。"""
        series = tmp_path / "series"
        path = legacy_single(series, ir_container())
        neighbour = series / "io.noise.safetensors"
        neighbour.write_bytes(b"not a shard of this component")

        repack.repack_component(path)

        assert neighbour.read_bytes() == b"not a shard of this component"


class TestRepackingALegacyTailSlackPair:
    """fat グラフ shard + 尾部 shard → v2（本数が変わる回）。"""

    def test_the_previous_numbering_does_not_survive(self, tmp_path):
        series = tmp_path / "series"
        path = legacy_tail_slack(series, ir_container())
        before = sorted(listing(series))
        assert before == [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]

        published = repack.repack_component(path, _shard_byte_limit=SPLIT_LIMIT)

        assert [entry.name for entry in published] == [
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ]
        assert sorted(listing(series)) == [entry.name for entry in published]

    def test_every_tensor_keeps_its_bytes(self, tmp_path):
        series = tmp_path / "series"
        path = legacy_tail_slack(series, ir_container())
        before = tensors_of(list(series.iterdir()))

        published = repack.repack_component(path, _shard_byte_limit=SPLIT_LIMIT)

        assert tensors_of(published) == before
        verify_shards(published)


class TestIdempotence:
    def test_repacking_a_v2_component_changes_nothing(self, tmp_path):
        series = tmp_path / "series"
        path = v2_component(series, ir_container(storage="i8"))
        before = listing(series)

        published = repack.repack_component(path)

        assert len(published) == 2
        assert listing(series) == before


class TestWritingElsewhere:
    def test_the_out_directory_gets_the_sequence_and_the_source_is_untouched(self, tmp_path):
        series = tmp_path / "series"
        path = legacy_single(series, ir_container())
        before = listing(series)

        published = repack.repack_component(path, tmp_path / "out")

        assert [entry.parent for entry in published] == [tmp_path / "out"] * len(published)
        assert listing(series) == before
        verify_shards(published)

    def test_two_inputs_that_land_on_the_same_name_fail_loudly(self, tmp_path):
        """`--out` は宛先を 1 ディレクトリへ畳むので、同名のコンポーネントは共存できない。"""
        first = legacy_single(tmp_path / "a", ir_container(mark="a"))
        second = legacy_single(tmp_path / "b", ir_container(mark="b"))

        with pytest.raises(repack.RepackError, match="宛先が同じ"):
            repack.main([str(first), str(second), "--out", str(tmp_path / "out")])


class TestFaultInjection:
    """門が本当に見ているか（詰め替えは「バイトを変えない」ことだけが価値）。"""

    def test_a_write_that_drops_a_tensor_is_caught_after_the_write(self, tmp_path, monkeypatch):
        """故障注入: 1 本欠けた割り付けで書き切らせる（前段の分割検査を外して門へ届かせる）。

        書き出し**後**の検証（`verify_shards` → メタデータ → バイト写像）が受け止め、手元の
        旧配布形は 1 バイトも変わらない。
        """
        series = tmp_path / "series"
        path = legacy_single(series, ir_container())
        before = listing(series)

        def drop_the_last(order, payload_bytes, companions, limit):
            groups = pack_shards(order, payload_bytes, companions, limit)
            return [tuple(name for name in group if name != order[-1]) for group in groups]

        monkeypatch.setattr(repack, "pack_shards", drop_the_last)
        monkeypatch.setattr(repack, "assert_shard_partition", lambda *args: None)

        with pytest.raises(ContainerError, match="ファイルに無い"):
            repack.repack_component(path)

        assert listing(series) == before
