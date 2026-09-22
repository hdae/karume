"""旧配布形 → コンテナ形式の移行（`karume.migrate`）— docs/container-v1.md §12。

被験体は合成の**正当な**コンポーネント（{@link ir_fixtures.ir_container} が現行の書き手で焼く
shard 列）で、旧単一形は器の低レベル面（`emit.write_container`）だけを借りて手組みする
（旧 writer はもう無い — ADR 0081）。突き合わせの正本は**入力の safetensors から直に読んだ
生バイト**で、移行器の自己検査とは別経路で取る（自己検査が恒真でも落ちる形にしておく）。

グラフ単体の規則（改名・正準直列化・block の詰め方）は `test_container.py` の担当。ここが見るのは
移行そのもの — 束縛の導出・scale の形の門・payload の同一性・据え替えの規律・CLI。
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest
from ir_fixtures import ir_container

from karume import migrate
from karume.container import ContainerFormatError, Encoding, Provenance, read_container
from karume.emit import ContainerEntry, container_order, write_container
from karume.ir import (
    IR_METADATA_KEY,
    IrGraph,
    IrInitializer,
    IrInput,
    IrNode,
    IrShared,
    IrStorage,
    IrValue,
)
from karume.migrate import MigrateError, container_bindings, migrate_component
from karume.repack import SourceTensor
from karume.shards import shard_path
from karume.verify import bind_graphs, parse_ir_graph

#: コンポーネントの代表 path のファイル名（格納形のタグ込み — 系列の綴りと同じ）。
COMPONENT = "model.i4.safetensors"

#: 移行の呼び手が渡す出所（`--license` は必須・`writer` は CLI が既定を埋める）。
PROVENANCE = Provenance(license="apache-2.0", writer="karume/test")

#: 合成資産（数 KiB）で part またぎと piece 分割を踏むために下げた寸法。
SMALL_PART_BYTES = 512
SMALL_BLOCK_BYTES = 256


def read_safetensors(blob: bytes) -> tuple[dict[str, str], dict[str, bytes]]:
    """safetensors のバイト列を `(__metadata__, テンソルキー → 生バイト)` へ開く。"""
    length = struct.unpack("<Q", blob[:8])[0]
    header = json.loads(blob[8 : 8 + length])
    start = 8 + length
    tensors = {
        name: blob[start + spec["data_offsets"][0] : start + spec["data_offsets"][1]]
        for name, spec in header.items()
        if name != "__metadata__"
    }
    return dict(header.get("__metadata__", {})), tensors


def source_material(shards: list[bytes]) -> tuple[dict[str, str], dict[str, bytes]]:
    """shard 列を 1 つの素材へ畳む（メタデータは先頭 shard のもの）。"""
    metadata, tensors = read_safetensors(shards[0])
    for blob in shards[1:]:
        tensors.update(read_safetensors(blob)[1])
    return metadata, tensors


def stage_series(directory: Path, shards: list[bytes]) -> Path:
    """現行形（連番の shard 列）を置き、コンポーネントの代表 path を返す。"""
    directory.mkdir(parents=True, exist_ok=True)
    for index, blob in enumerate(shards, start=1):
        shard_path(directory / COMPONENT, index, len(shards)).write_bytes(blob)
    return directory / COMPONENT


def stage_legacy_single(directory: Path, shards: list[bytes]) -> Path:
    """旧・単一ファイル配布形（`karume_ir` と全テンソルが 1 つの器に同居）。"""
    directory.mkdir(parents=True, exist_ok=True)
    metadata, tensors = source_material(shards)
    shapes = {}
    for blob in shards:
        length = struct.unpack("<Q", blob[:8])[0]
        for name, spec in json.loads(blob[8 : 8 + length]).items():
            if name != "__metadata__":
                shapes[name] = (spec["dtype"], tuple(spec["shape"]))
    entries = container_order(
        ContainerEntry(name=name, dtype=shapes[name][0], shape=shapes[name][1], nbytes=len(raw))
        for name, raw in tensors.items()
    )
    target = directory / COMPONENT
    write_container(target, entries, metadata, lambda entry: [tensors[entry.name]])
    return target


def migrated_payloads(paths: list[Path]) -> dict[str, bytes]:
    """書いたコンテナから **initializer ごとの payload** を取り直す（詰め物を除いた実バイト）。

    取り直しは合流層の供給計画（`verify.bind_graphs`）越しなので、piece 列も scale も
    「宣言がそう言っている場所」から読む — 移行器の自己検査とは別の経路で同じ答えに着く。
    """
    read = read_container(paths)
    bound = next(iter(bind_graphs(read.graph, read.model).values()))
    found: dict[str, bytes] = {}
    for name, supply in bound.supplies.items():
        found[name] = b"".join(
            read.block(block.id)[: block.payload_bytes] for block in supply.blocks
        )
        if supply.scale is not None:
            assert supply.encoding.scale_block is not None
            found[f"scale:{name}"] = read.block(supply.scale.id)[: supply.scale.payload_bytes]
    return found


def scale_keys(graph: IrGraph) -> dict[str, str]:
    """initializer のテンソルキー → scale のテンソルキー（突合の相手を旧宣言から引く）。"""
    return {
        init.tensor: init.storage.scale
        for init in graph.initializers.values()
        if init.tensor is not None and init.storage.scale is not None
    }


@pytest.fixture
def component(tmp_path: Path) -> tuple[Path, dict[str, str], dict[str, bytes]]:
    """i4 混成（i4 / i8 / f32 が同居する実物と同じ形）の現行配布形を置く。"""
    shards = ir_container(mark="mig", storage="i4")
    metadata, tensors = source_material(shards)
    return stage_series(tmp_path / "src", shards), metadata, tensors


class TestTheMigratedContainer:
    def test_every_payload_survives_byte_for_byte(
        self, component: tuple[Path, dict[str, str], dict[str, bytes]], tmp_path: Path
    ) -> None:
        """§12 の不変条件 1 — 実体も scale も 1 バイトも変わらない（詰め物は新たに焼かれる）。"""
        path, metadata, tensors = component
        result = migrate_component(path, tmp_path / "out", provenance=PROVENANCE)
        graph = parse_ir_graph(metadata[IR_METADATA_KEY])
        scales = scale_keys(graph)

        found = migrated_payloads(list(result.parts))
        expected = {key: tensors[key] for key in scales}
        expected.update({f"scale:{key}": tensors[scale] for key, scale in scales.items()})
        expected.update(
            {
                init.tensor: tensors[init.tensor]
                for init in graph.initializers.values()
                if init.tensor is not None
            }
        )
        assert found == expected
        # 自己検査が数えた本数（実体 + scale）と、突き合わせた本数が一致する。
        assert result.payloads == len(expected)
        assert result.initializers == len(graph.initializers)

    def test_the_old_distribution_is_read_only(
        self, component: tuple[Path, dict[str, str], dict[str, bytes]], tmp_path: Path
    ) -> None:
        """§12 —「旧入力は保持する」（CLI は入力を消さない・書き換えない）。"""
        path, _, _ = component
        before = {entry.name: entry.read_bytes() for entry in sorted(path.parent.iterdir())}
        migrate_component(path, tmp_path / "out", provenance=PROVENANCE)

        assert {entry.name: entry.read_bytes() for entry in sorted(path.parent.iterdir())} == before

    def test_the_same_input_produces_the_same_bytes(
        self, component: tuple[Path, dict[str, str], dict[str, bytes]], tmp_path: Path
    ) -> None:
        """§12 の不変条件 4 — 決定的（part への詰め方も JSON のキー順も入力から決まる）。"""
        path, _, _ = component
        first = migrate_component(path, tmp_path / "a", provenance=PROVENANCE)
        second = migrate_component(path, tmp_path / "b", provenance=PROVENANCE)

        assert [part.read_bytes() for part in first.parts] == [
            part.read_bytes() for part in second.parts
        ]

    def test_an_old_single_file_form_migrates_to_the_same_container(self, tmp_path: Path) -> None:
        """旧規則の配布形（単一ファイル）も受ける — 現行の門は入力に掛けない。"""
        shards = ir_container(mark="mig", storage="i4")
        series = migrate_component(
            stage_series(tmp_path / "series", shards),
            tmp_path / "a",
            provenance=PROVENANCE,
            graph_name="depth",
        )
        single = migrate_component(
            stage_legacy_single(tmp_path / "legacy", shards),
            tmp_path / "b",
            provenance=PROVENANCE,
            graph_name="depth",
        )

        assert [part.read_bytes() for part in single.parts] == [
            part.read_bytes() for part in series.parts
        ]

    def test_a_numbered_input_path_is_folded_to_the_component(
        self, component: tuple[Path, dict[str, str], dict[str, bytes]], tmp_path: Path
    ) -> None:
        """手元の現物（`-00001-of-00002`）を指しても、1 本だけを移す形にならない。"""
        path, _, _ = component
        first = sorted(path.parent.iterdir())[0]
        whole = migrate_component(path, tmp_path / "a", provenance=PROVENANCE)
        folded = migrate_component(first, tmp_path / "b", provenance=PROVENANCE)

        assert [part.read_bytes() for part in folded.parts] == [
            part.read_bytes() for part in whole.parts
        ]

    def test_the_graph_name_defaults_to_the_component_directory(
        self, component: tuple[Path, dict[str, str], dict[str, bytes]], tmp_path: Path
    ) -> None:
        path, _, _ = component
        result = migrate_component(path, tmp_path / "out", provenance=PROVENANCE)
        read = read_container(list(result.parts))

        assert list(read.graph.graphs) == [path.parent.name]

    def test_a_directory_name_outside_the_container_vocabulary_asks_for_an_explicit_name(
        self, tmp_path: Path
    ) -> None:
        """既定は暗黙（親ディレクトリ名）なので、外れた回は**何を渡すか**まで綴って止まる。"""
        shards = ir_container(mark="mig", storage="f32")
        path = stage_series(tmp_path / "text encoder", shards)

        with pytest.raises(MigrateError, match="`--graph-name` で明示する"):
            migrate_component(path, tmp_path / "out", provenance=PROVENANCE)

    def test_the_split_form_is_named_like_a_shard_series(
        self, component: tuple[Path, dict[str, str], dict[str, bytes]], tmp_path: Path
    ) -> None:
        path, _, _ = component
        result = migrate_component(
            path,
            tmp_path / "out",
            provenance=PROVENANCE,
            _part_bytes=SMALL_PART_BYTES,
            _block_bytes=SMALL_BLOCK_BYTES,
        )
        total = len(result.parts)

        assert total >= 3
        assert [part.name for part in result.parts] == [
            f"model.i4-{index + 1:05d}-of-{total:05d}.krm" for index in range(total)
        ]

    def test_the_single_form_is_one_file(
        self, component: tuple[Path, dict[str, str], dict[str, bytes]], tmp_path: Path
    ) -> None:
        path, _, _ = component
        result = migrate_component(path, tmp_path / "out", provenance=PROVENANCE, single=True)

        assert [part.name for part in result.parts] == ["model.i4.krm"]
        assert read_container(list(result.parts)).header.kind == "model"

    def test_the_graph_container_carries_the_same_graph_description(
        self, component: tuple[Path, dict[str, str], dict[str, bytes]], tmp_path: Path
    ) -> None:
        """`krg` は `krm` からのバイトコピー抽出（§9 — グラフ記述はバイト単位で同一）。"""
        path, _, _ = component
        result = migrate_component(path, tmp_path / "out", provenance=PROVENANCE, write_graph=True)
        assert result.graph is not None
        model = read_container(list(result.parts))
        graph = read_container([result.graph])

        assert result.graph.name == "model.i4.krg"
        assert graph.header.kind == "graph"
        assert graph.model is None
        assert graph.graph_descriptor_bytes == model.graph_descriptor_bytes

    def test_the_container_is_never_staged_over_a_previous_output(
        self, component: tuple[Path, dict[str, str], dict[str, bytes]], tmp_path: Path
    ) -> None:
        """出力先の残骸は**消さずに止まる**（どのバイト列を配るかが一意に決まらない）。"""
        path, _, _ = component
        migrate_component(path, tmp_path / "out", provenance=PROVENANCE)

        with pytest.raises(MigrateError, match="前回の成果物が残っている"):
            migrate_component(path, tmp_path / "out", provenance=PROVENANCE)

    def test_a_failed_migration_leaves_nothing_behind(
        self, component: tuple[Path, dict[str, str], dict[str, bytes]], tmp_path: Path
    ) -> None:
        """検査を通してから据える — 落ちた回は一時ファイルごと消える（本番名は生まれない）。"""
        path, _, _ = component
        out = tmp_path / "out"
        out.mkdir()

        with pytest.raises(ContainerFormatError):
            # 1 block と companion scale の一群が入らない part 長（配置が決まらない）。
            migrate_component(path, out, provenance=PROVENANCE, _part_bytes=64)

        assert sorted(out.iterdir()) == []


class TestTheDerivedBindings:
    """束縛（codec / rowAxis / groupSize）は旧宣言と**消費側 op** から決まる。"""

    @staticmethod
    def _graph(storage: IrStorage, shape: list[int], op: str = "linear") -> IrGraph:
        return IrGraph(
            inputs=[IrInput(name="x", dtype="f32", shape=[1, shape[-1]])],
            outputs=["y"],
            initializers={"p_w": IrInitializer(tensor="w", storage=storage)},
            values={"p_w": IrValue(dtype="f32", shape=shape), "y": IrValue(dtype="f32", shape=[1])},
            nodes=[IrNode(op=op, ins=["x", "p_w"], outs=["y"], attrs={})],
        )

    @pytest.mark.parametrize(
        ("dtype", "codec"),
        [("f32", "f32"), ("f16", "f16"), ("bf16", "bf16"), ("i32", "i32")],
    )
    def test_an_unquantized_storage_maps_to_its_codec(self, dtype: str, codec: str) -> None:
        bindings = container_bindings(self._graph(IrStorage(dtype=dtype), [4, 32]))

        assert bindings == {"w": Encoding(codec)}

    def test_a_per_channel_storage_declares_the_row_length_as_its_group_size(self) -> None:
        bindings = container_bindings(self._graph(IrStorage(dtype="i8", scale="w_scale"), [4, 32]))

        assert bindings == {
            "w": Encoding("int8-sym", group_size=32, row_axis=0, scale_key="w_scale")
        }

    def test_a_conv_transpose1d_weight_declares_row_axis_1(self) -> None:
        """`[Cin,Cout,K]` の転置レイアウトだけが軸 1（`emit.weight_channel_axes` の鏡像）。"""
        bindings = container_bindings(
            self._graph(IrStorage(dtype="i8", scale="w_scale"), [4, 3, 8], op="conv_transpose1d")
        )

        # 行長 = numel / shape[1] = 96 / 3。
        assert bindings == {
            "w": Encoding("int8-sym", group_size=32, row_axis=1, scale_key="w_scale")
        }

    def test_a_group_storage_keeps_its_declared_group_size(self) -> None:
        bindings = container_bindings(
            self._graph(IrStorage(dtype="i4", scale="w_scale", group_size=16), [4, 32])
        )

        assert bindings == {
            "w": Encoding("int4-sym-g", group_size=16, row_axis=0, scale_key="w_scale")
        }

    def test_an_i2_storage_maps_to_int2_off_and_never_to_ternary(self) -> None:
        """三値であるという主張は量子化器の側がする（§6.3 — 旧 i2 には値域外のコードが出る）。"""
        bindings = container_bindings(self._graph(IrStorage(dtype="i2", scale="w_scale"), [4, 32]))

        assert bindings["w"].codec == "int2-off"

    def test_a_shared_initializer_has_no_binding(self) -> None:
        graph = self._graph(IrStorage(dtype="f16"), [4, 32])
        graph.initializers["p_s"] = IrInitializer(
            shared=IrShared(tensor="lend.weight"), storage=IrStorage(dtype="f16")
        )
        graph.values["p_s"] = IrValue(dtype="f32", shape=[4, 32])

        assert set(container_bindings(graph)) == {"w"}

    def test_a_group_storage_consumed_on_axis_1_fails_loudly(self) -> None:
        """旧 group scale は先頭次元を行として焼かれている — 軸 1 の消費とは両立しない。"""
        graph = self._graph(
            IrStorage(dtype="i4", scale="w_scale", group_size=16), [4, 3, 8], "conv_transpose1d"
        )

        with pytest.raises(MigrateError, match="写せる形が無い"):
            container_bindings(graph)

    def test_a_quantized_storage_without_a_scale_fails_loudly(self) -> None:
        with pytest.raises(MigrateError, match="scale の宣言が無い"):
            container_bindings(self._graph(IrStorage(dtype="i8"), [4, 32]))


class TestTheScaleLayoutGate:
    """旧 scale の**形**が新しい宣言（`rowAxis` / `groupSize`）と一致することを焼く前に見る。"""

    @staticmethod
    def _sources(weight: tuple[int, ...], scale: tuple[int, ...]) -> dict[str, SourceTensor]:
        return {
            "w": SourceTensor(
                entry=ContainerEntry(name="w", dtype="I8", shape=weight, nbytes=64), segments=()
            ),
            "w_scale": SourceTensor(
                entry=ContainerEntry(name="w_scale", dtype="F32", shape=scale, nbytes=4 * scale[0]),
                segments=(),
            ),
        }

    def _graph(self) -> IrGraph:
        return TestTheDerivedBindings._graph(IrStorage(dtype="i8", scale="w_scale"), [8, 8])

    def test_a_keepdim_scale_on_the_row_axis_passes(self) -> None:
        graph = self._graph()

        migrate._assert_scale_layouts(
            graph, container_bindings(graph), self._sources((8, 8), (8, 1))
        )

    def test_a_per_column_scale_is_refused_even_though_the_byte_count_agrees(self) -> None:
        """正方の重みではバイト数が一致してしまう — 形まで見ないとチャネルが入れ替わる。"""
        graph = self._graph()

        with pytest.raises(MigrateError, match=r"scale 'w_scale' の形 \[1, 8\]"):
            migrate._assert_scale_layouts(
                graph, container_bindings(graph), self._sources((8, 8), (1, 8))
            )


class TestTheCli:
    def test_it_writes_the_container_and_prints_one_line(
        self,
        component: tuple[Path, dict[str, str], dict[str, bytes]],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        path, _, _ = component
        migrate.main([str(path), "--out", str(tmp_path / "out"), "--license", "apache-2.0"])
        printed = capsys.readouterr().out

        # part 0（2 文書）+ part 1（const 領域・この合成では長さ 0）+ 重みの part 1 本。
        assert (tmp_path / "out" / "model.i4-00001-of-00003.krm").is_file()
        assert "parts 3" in printed and "payloads" in printed

    def test_the_writer_defaults_to_the_generator_tag(
        self, component: tuple[Path, dict[str, str], dict[str, bytes]], tmp_path: Path
    ) -> None:
        from karume.dist import generator_tag

        path, _, _ = component
        migrate.main([str(path), "--out", str(tmp_path / "out"), "--license", "mit"])
        parts = sorted((tmp_path / "out").glob("*.krm"))
        read = read_container(parts)

        assert read.model is not None
        assert read.model.provenance.license == "mit"
        assert read.model.provenance.writer == generator_tag()

    @pytest.mark.parametrize("missing", [["--license", "mit"], ["--out", "/tmp/out"]])
    def test_it_requires_both_the_output_directory_and_the_license(
        self, missing: list[str]
    ) -> None:
        with pytest.raises(SystemExit) as raised:
            migrate.main(["a/model.safetensors", *missing])

        assert raised.value.code == 2

    def test_a_single_graph_name_cannot_cover_several_components(self, tmp_path: Path) -> None:
        with pytest.raises(MigrateError, match="1 コンポーネントにだけ"):
            migrate.main(
                [
                    "a/model.safetensors",
                    "b/model.safetensors",
                    "--out",
                    str(tmp_path),
                    "--license",
                    "mit",
                    "--graph-name",
                    "shared",
                ]
            )


class TestTheSelfCheck:
    """自己検査が恒真でないこと（故障注入）— 書いたバイトが旧と違えば落ちる。"""

    def test_a_flipped_payload_byte_is_caught(
        self,
        component: tuple[Path, dict[str, str], dict[str, bytes]],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path, _, _ = component
        original = migrate._SourcePayloads.__getitem__

        def corrupt(self: migrate._SourcePayloads, key: str) -> bytes:
            raw = bytearray(original(self, key))
            if key.endswith("weight"):
                raw[0] ^= 0xFF
            return bytes(raw)

        monkeypatch.setattr(migrate._SourcePayloads, "__getitem__", corrupt)

        with pytest.raises(MigrateError, match="payload の sha256 が旧配布形と違う"):
            migrate_component(path, tmp_path / "out", provenance=PROVENANCE)


def test_the_digest_helper_reads_the_whole_payload(tmp_path: Path) -> None:
    """突合の土台（旧実体の sha256）が本当に全バイトを読む — 1 バイトでも欠ければ別の値。"""
    blob = bytes(range(256)) * 8
    target = tmp_path / "raw.bin"
    target.write_bytes(blob)
    source = SourceTensor(
        entry=ContainerEntry(name="w", dtype="F32", shape=(len(blob) // 4,), nbytes=len(blob)),
        segments=((target, 0, len(blob)),),
    )

    assert migrate._source_digest(source) == hashlib.sha256(blob).hexdigest()
