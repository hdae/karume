"""旧配布形 → コンテナ形式の移行（`karume.migrate`）— docs/container-v1.md §12。

被験体は合成の**正当な**旧配布形（{@link legacy_writer.legacy_shards} が旧規約で焼く shard 列 —
書き手は製品側にもう無いのでテストの中だけに置く）。突き合わせの正本は**入力の safetensors から
直に読んだ生バイト**で、移行器の自己検査とは別経路で取る（自己検査が恒真でも落ちる形にしておく）。

グラフ単体の規則（改名・正準直列化・block の詰め方）は `test_container.py` の担当。ここが見るのは
移行そのもの — 束縛の導出・scale の形の門・payload の同一性・据え替えの規律・CLI。
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest
from ir_fixtures import fixture_spec
from legacy_writer import (
    PLE_INDEX_FILE,
    PLE_SHARD_FILE,
    ROPE_BASE_NAME,
    Entry,
    legacy_ple_sidecar,
    legacy_rope_base,
    legacy_shards,
    order,
    stage_shards,
    write_safetensors,
)

from karume import migrate, publish
from karume.container import (
    PART_LENGTH_CHOICES,
    AssetInput,
    ContainerFormatError,
    Encoding,
    Provenance,
    numbered_path,
    read_container,
)
from karume.emit import stored_model
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
from karume.legacy import (
    SourceTensor,
    StoredEntry,
    payload_chunks,
    read_component,
    resolve_shards,
)
from karume.migrate import EXTRA_ASSETS, FileRef, MigrateError, migrate_component
from karume.ple import PLE_INDEX_ASSET, ple_assets
from karume.publish import publish_container
from karume.verify import bind_graphs, parse_ir_graph

#: コンポーネントの代表 path のファイル名（格納形のタグ込み — 系列の綴りと同じ）。
COMPONENT = "model.i4.safetensors"

#: 旧 `extras` の `rope_base` が畳まれる容器の資産名と役割（`karume.migrate.EXTRA_ASSETS`）。
ROPE_BASE_ASSET, ROPE_BASE_ROLE = EXTRA_ASSETS[ROPE_BASE_NAME]

#: PLE を切るときの block 上限（合成の小さな表を**複数 block**へ割る席 — 容器側の block 上限
#: とは別の軸で、資産 1 本ぶんの大きさを決める）。
PLE_BLOCK_BYTES = 64

#: 移行の呼び手が渡す出所（`--license` は必須・`writer` は明示したときだけ載る）。
PROVENANCE = Provenance(license="apache-2.0", writer="karume/test")

#: 合成資産（数 KiB）で part またぎと piece 分割を踏むために下げた寸法。
SMALL_PART_BYTES = 512
SMALL_BLOCK_BYTES = 256

#: 1 block と companion scale の一群が入らない part 長（配置が決まらない回を作る）。
UNPLACEABLE_PART_BYTES = 64

MIB = 1024 * 1024


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
        numbered_path(directory / COMPONENT, index, len(shards)).write_bytes(blob)
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
    entries = order(
        [
            Entry(name=name, dtype=shapes[name][0], shape=shapes[name][1], payload=raw)
            for name, raw in tensors.items()
        ]
    )
    target = directory / COMPONENT
    target.write_bytes(write_safetensors(entries, metadata))
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
def small_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    """書き手の選択集合（256 MiB〜）に**合成資産の寸法**を足す。

    数 KiB の合成資産は集合の part 長では part を割らないので、part またぎを踏む回だけ集合を
    広げる。集合の検査そのもの（集合外は落ちる）は `TestThePartLength` が本物の集合で見る。
    """
    monkeypatch.setattr(
        migrate,
        "PART_LENGTH_CHOICES",
        (*PART_LENGTH_CHOICES, SMALL_PART_BYTES, UNPLACEABLE_PART_BYTES),
    )


@pytest.fixture
def written_part_bytes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """書き手（`write_model_container`）に届いた part 長を呼ばれた順に拾う（書き出しは素通し）。"""
    seen: list[int] = []
    original = publish.write_model_container

    def spy(*args, **kwargs):
        seen.append(kwargs["part_bytes"])
        return original(*args, **kwargs)

    monkeypatch.setattr(publish, "write_model_container", spy)
    return seen


@pytest.fixture
def component(tmp_path: Path) -> tuple[Path, dict[str, str], dict[str, bytes]]:
    """i4 混成（i4 / i8 / f32 が同居する実物と同じ形）の現行配布形を置く。"""
    shards = legacy_shards(mark="mig", storage="i4")
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
        shards = legacy_shards(mark="mig", storage="i4")
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
        shards = legacy_shards(mark="mig", storage="f32")
        path = stage_series(tmp_path / "text encoder", shards)

        with pytest.raises(MigrateError, match="`--graph-name` で明示する"):
            migrate_component(path, tmp_path / "out", provenance=PROVENANCE)

    def test_the_split_form_is_named_like_a_shard_series(
        self,
        component: tuple[Path, dict[str, str], dict[str, bytes]],
        tmp_path: Path,
        small_parts: None,
    ) -> None:
        path, _, _ = component
        result = migrate_component(
            path,
            tmp_path / "out",
            provenance=PROVENANCE,
            part_bytes=SMALL_PART_BYTES,
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
        self,
        component: tuple[Path, dict[str, str], dict[str, bytes]],
        tmp_path: Path,
        small_parts: None,
    ) -> None:
        """検査を通してから据える — 落ちた回は一時ファイルごと消える（本番名は生まれない）。"""
        path, _, _ = component
        out = tmp_path / "out"
        out.mkdir()

        with pytest.raises(ContainerFormatError):
            # 1 block と companion scale の一群が入らない part 長（配置が決まらない）。
            migrate_component(path, out, provenance=PROVENANCE, part_bytes=UNPLACEABLE_PART_BYTES)

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
        bindings = migrate._bindings(self._graph(IrStorage(dtype=dtype), [4, 32]))

        assert bindings == {"w": Encoding(codec)}

    def test_a_per_channel_storage_declares_the_row_length_as_its_group_size(self) -> None:
        bindings = migrate._bindings(self._graph(IrStorage(dtype="i8", scale="w_scale"), [4, 32]))

        assert bindings == {
            "w": Encoding("int8-sym", group_size=32, row_axis=0, scale_key="w_scale")
        }

    def test_a_conv_transpose1d_weight_declares_row_axis_1(self) -> None:
        """`[Cin,Cout,K]` の転置レイアウトだけが軸 1（`emit.weight_channel_axes` の鏡像）。"""
        bindings = migrate._bindings(
            self._graph(IrStorage(dtype="i8", scale="w_scale"), [4, 3, 8], op="conv_transpose1d")
        )

        # 行長 = numel / shape[1] = 96 / 3。
        assert bindings == {
            "w": Encoding("int8-sym", group_size=32, row_axis=1, scale_key="w_scale")
        }

    def test_a_group_storage_keeps_its_declared_group_size(self) -> None:
        bindings = migrate._bindings(
            self._graph(IrStorage(dtype="i4", scale="w_scale", group_size=16), [4, 32])
        )

        assert bindings == {
            "w": Encoding("int4-sym-g", group_size=16, row_axis=0, scale_key="w_scale")
        }

    def test_an_i2_storage_maps_to_int2_off_and_never_to_ternary(self) -> None:
        """三値であるという主張は量子化器の側がする（§6.3 — 旧 i2 には値域外のコードが出る）。"""
        bindings = migrate._bindings(self._graph(IrStorage(dtype="i2", scale="w_scale"), [4, 32]))

        assert bindings["w"].codec == "int2-off"

    def test_a_shared_initializer_has_no_binding(self) -> None:
        graph = self._graph(IrStorage(dtype="f16"), [4, 32])
        graph.initializers["p_s"] = IrInitializer(
            shared=IrShared(tensor="lend.weight"), storage=IrStorage(dtype="f16")
        )
        graph.values["p_s"] = IrValue(dtype="f32", shape=[4, 32])

        assert set(migrate._bindings(graph)) == {"w"}

    def test_a_group_storage_consumed_on_axis_1_fails_loudly(self) -> None:
        """旧 group scale は先頭次元を行として焼かれている — 軸 1 の消費とは両立しない。"""
        graph = self._graph(
            IrStorage(dtype="i4", scale="w_scale", group_size=16), [4, 3, 8], "conv_transpose1d"
        )

        with pytest.raises(MigrateError, match="写せる形が無い"):
            migrate._bindings(graph)

    def test_a_quantized_storage_without_a_scale_fails_loudly(self) -> None:
        with pytest.raises(MigrateError, match="scale の宣言が無い"):
            migrate._bindings(self._graph(IrStorage(dtype="i8"), [4, 32]))


class TestTheScaleLayoutGate:
    """旧 scale の**形**が新しい宣言（`rowAxis` / `groupSize`）と一致することを焼く前に見る。"""

    @staticmethod
    def _sources(weight: tuple[int, ...], scale: tuple[int, ...]) -> dict[str, SourceTensor]:
        return {
            "w": SourceTensor(
                entry=StoredEntry(name="w", dtype="I8", shape=weight, nbytes=64), segments=()
            ),
            "w_scale": SourceTensor(
                entry=StoredEntry(name="w_scale", dtype="F32", shape=scale, nbytes=4 * scale[0]),
                segments=(),
            ),
        }

    def _graph(self) -> IrGraph:
        return TestTheDerivedBindings._graph(IrStorage(dtype="i8", scale="w_scale"), [8, 8])

    def test_a_keepdim_scale_on_the_row_axis_passes(self) -> None:
        graph = self._graph()

        migrate._assert_scale_layouts(
            graph, migrate._bindings(graph), self._sources((8, 8), (8, 1))
        )

    def test_a_per_column_scale_is_refused_even_though_the_byte_count_agrees(self) -> None:
        """正方の重みではバイト数が一致してしまう — 形まで見ないとチャネルが入れ替わる。"""
        graph = self._graph()

        with pytest.raises(MigrateError, match=r"scale 'w_scale' の形 \[1, 8\]"):
            migrate._assert_scale_layouts(
                graph, migrate._bindings(graph), self._sources((8, 8), (1, 8))
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

    def test_the_writer_is_absent_unless_the_caller_names_one(
        self, component: tuple[Path, dict[str, str], dict[str, bytes]], tmp_path: Path
    ) -> None:
        """MUST: 既定では `writer` を**書かない**（生成器タグを埋めない）。

        埋めると、移行した容器と recipe が直接 export した容器（`PROVENANCE` はどれも
        `writer` を綴らない）が永久に別バイトになる — 実測で、ミラー 7 リポの part 0 が
        `,"writer":"karume/0.12.0"` の 25 バイトぶんだけ焼き直しと食い違っていた。
        ツールの版は `karume.json` の `generator` 欄が 1 箇所で持つ。
        """
        path, _, _ = component
        migrate.main([str(path), "--out", str(tmp_path / "out"), "--license", "mit"])
        read = read_container(sorted((tmp_path / "out").glob("*.krm")))

        assert read.model is not None
        assert read.model.provenance.license == "mit"
        assert read.model.provenance.writer is None
        assert "writer" not in read.model.provenance.to_document()

    def test_an_explicit_writer_is_still_written(
        self, component: tuple[Path, dict[str, str], dict[str, bytes]], tmp_path: Path
    ) -> None:
        """対（恒真でない）: 明示すれば載る — 落ちたのは「既定」であって席ではない。"""
        path, _, _ = component
        migrate.main(
            [str(path), "--out", str(tmp_path / "out"), "--license", "mit", "--writer", "acme/1"]
        )
        read = read_container(sorted((tmp_path / "out").glob("*.krm")))

        assert read.model is not None
        assert read.model.provenance.writer == "acme/1"

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


class TestThePartLength:
    """part 長は書き手の選択集合 {256, 512, 768, 1024} MiB の 1 つ（container-v1 §4.2）。"""

    def test_the_default_is_256_mib(
        self,
        component: tuple[Path, dict[str, str], dict[str, bytes]],
        tmp_path: Path,
        written_part_bytes: list[int],
    ) -> None:
        path, _, _ = component
        migrate.main([str(path), "--out", str(tmp_path / "cli"), "--license", "mit"])
        migrate_component(path, tmp_path / "api", provenance=PROVENANCE)

        assert written_part_bytes == [256 * MIB, 256 * MIB]

    def test_the_cli_hands_the_chosen_mib_to_the_writer_as_bytes(
        self,
        component: tuple[Path, dict[str, str], dict[str, bytes]],
        tmp_path: Path,
        written_part_bytes: list[int],
    ) -> None:
        path, _, _ = component
        out = tmp_path / "out"
        migrate.main([str(path), "--out", str(out), "--license", "mit", "--part-bytes", "512"])

        assert written_part_bytes == [512 * MIB]
        assert read_container(sorted(out.glob("*.krm"))).header.kind == "model"

    @pytest.mark.parametrize("part_bytes", [300 * MIB, SMALL_PART_BYTES])
    def test_a_length_outside_the_choices_fails_loudly_before_writing(
        self,
        component: tuple[Path, dict[str, str], dict[str, bytes]],
        tmp_path: Path,
        part_bytes: int,
    ) -> None:
        """天井（1024 MiB）の内側でも集合外なら止まる — 書き手の天井検査とは別の門。"""
        path, _, _ = component
        out = tmp_path / "out"

        with pytest.raises(MigrateError, match=r"選択集合 \{256, 512, 768, 1024\} MiB の外"):
            migrate_component(path, out, provenance=PROVENANCE, part_bytes=part_bytes)

        assert not out.exists()

    def test_the_cli_refuses_a_length_outside_the_choices(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as raised:
            migrate.main(
                [
                    "a/model.safetensors",
                    "--out",
                    str(tmp_path),
                    "--license",
                    "mit",
                    "--part-bytes",
                    "300",
                ]
            )

        assert raised.value.code == 2


class TestTheSelfCheck:
    """自己検査が恒真でないこと（故障注入）— 書いたバイトが旧と違えば据えない。"""

    def test_a_flipped_byte_in_a_written_part_is_caught(
        self,
        component: tuple[Path, dict[str, str], dict[str, bytes]],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """書いた直後に 1 バイト化けた回は据わらない（block の sha256 は書き手が宣言する）。"""
        path, _, _ = component
        original = publish.write_model_container

        def flip(*args, **kwargs):
            written = original(*args, **kwargs)
            raw = bytearray(written[-1].read_bytes())
            raw[-1] ^= 0xFF
            written[-1].write_bytes(bytes(raw))
            return written

        monkeypatch.setattr(publish, "write_model_container", flip)
        out = tmp_path / "out"

        with pytest.raises(ContainerFormatError, match="sha256 が宣言と違う"):
            migrate_component(path, out, provenance=PROVENANCE)

        assert not out.exists() or list(out.iterdir()) == []

    def test_the_payload_check_reads_the_old_shards(
        self, component: tuple[Path, dict[str, str], dict[str, bytes]]
    ) -> None:
        """突合の相手は**旧 shard から読み直したバイト**（新しい容器の写しではない）。

        ここが恒真化すると、移行が「自分で書いたものを自分で確かめる」だけになる。
        """
        path, _, tensors = component
        _, stored = read_component(list(resolve_shards(path)))
        payloads = migrate._SourcePayloads(stored)

        assert {key: bytes(payloads[key]) for key in stored} == tensors


def test_the_payload_reader_reads_the_whole_payload(tmp_path: Path) -> None:
    """突合の土台（旧実体の読み出し）が本当に全バイトを流す — 1 バイトでも欠ければ別の値。"""
    blob = bytes(range(256)) * 8
    target = tmp_path / "raw.bin"
    target.write_bytes(blob)
    source = SourceTensor(
        entry=StoredEntry(name="w", dtype="F32", shape=(len(blob) // 4,), nbytes=len(blob)),
        segments=((target, 0, len(blob)),),
    )

    digest = hashlib.sha256()
    for chunk in payload_chunks(source):
        digest.update(chunk)

    assert digest.hexdigest() == hashlib.sha256(blob).hexdigest()


class TestTheMigratedContainerMatchesADirectWrite:
    """**同一性の門** — 同じ素材から出た容器は、経路が違ってもバイト同一。

    ①旧 shard 形へ焼いてから `karume migrate` で移した容器と、②`stored_model` →
    `publish_container` で直接書いた容器が、part 列のバイト列まで一致する。ここが割れると
    「移行済みのミラー」と「再 export した系列」が同じ重みなのに別の資産になり、pin と
    sha256 の対応が段をまたぐたびに切れる（container-v1 §12 の不変条件 4「決定的」の実効面）。

    束縛表も配置も JSON のキー順も入力から決まるので、割れる余地があるのは**導出が 2 経路に
    分かれている場所**だけ — 実際に分かれていた頃（`container_bindings` の写しが移行側にあった
    頃）は「宣言 i4 / 実体 i8」が作れた。
    """

    @staticmethod
    def _material(storage: str):
        return fixture_spec("ident", storage, (), ([1],), ("weight",), None)

    def _migrated(self, tmp_path: Path, storage: str) -> list[bytes]:
        """旧 shard 形へ焼いてから移す（①）。"""
        shards = legacy_shards(mark="ident", storage=storage, groups=2)
        source = stage_shards(tmp_path / "old" / "enc", f"model.{storage}.safetensors", shards)
        out = tmp_path / "migrated"
        result = migrate_component(source, out, provenance=PROVENANCE, graph_name="enc")
        return [path.read_bytes() for path in result.parts]

    def _direct(self, tmp_path: Path, storage: str) -> list[bytes]:
        """同じ素材から直接書く（②）。"""
        graph, tensors, scales, overrides = self._material(storage)
        stored = stored_model(
            graph,
            tensors,
            weight_dtype=storage,
            weight_scales=scales,
            weight_dtype_overrides=overrides,
        )
        result = publish_container(
            tmp_path / "direct" / f"model.{storage}.krm",
            stored.graph,
            stored.tensors,
            stored.bindings,
            graph_name="enc",
            provenance=PROVENANCE,
        )
        return [path.read_bytes() for path in result.parts]

    @pytest.mark.parametrize("storage", ["f32", "f16", "i8", "i4"])
    def test_both_routes_write_the_same_parts(self, tmp_path: Path, storage: str) -> None:
        migrated = self._migrated(tmp_path, storage)
        direct = self._direct(tmp_path, storage)

        assert [len(part) for part in migrated] == [len(part) for part in direct]
        assert migrated == direct

    def _asset_material(self, tmp_path: Path, kind: str):
        """旧 sidecar を置き、①移行が組む資産と ②直接書く側が組む資産の対を返す。

        ①は移行 CLI の資産経路（`migrate._extra_assets` / `migrate._ple_assets` — 旧形の読みと
        索引の畳み込み）、②は書き手が現に使う口（`AssetInput` / `karume.ple.ple_assets`）。
        **どちらも「正しい容器」を作る**ので、物理配置（専用 part・末尾詰め物・`assets` 節の
        正準直列化）のずれを見る検出器はこの門しか無い。
        """
        repo = tmp_path / "old"
        where = "ident"
        if kind == "rope_base":
            payload = legacy_rope_base(repo)
            ref = FileRef(f"{ROPE_BASE_NAME}.safetensors", len(payload), "0" * 64)
            migrated = migrate._extra_assets(repo, ((ROPE_BASE_NAME, ref),), where)
            direct = {
                ROPE_BASE_ASSET: AssetInput(ROPE_BASE_ROLE, len(payload), payload),
            }
            return migrated, direct
        ple = legacy_ple_sidecar(repo)
        if kind == "ple-sidecar":
            # 系列ディレクトリを移す経路（`migrate_series` が通る口 — 在処は索引が名乗る）。
            migrated = migrate.ple_sidecar_assets(
                repo / PLE_INDEX_FILE, where=where, block_bytes=PLE_BLOCK_BYTES
            )
        else:
            refs = tuple(
                (name, FileRef(name, (repo / name).stat().st_size, "0" * 64))
                for name in (PLE_INDEX_FILE, PLE_SHARD_FILE)
            )
            index = migrate.read_ple_index(repo / PLE_INDEX_FILE, where)
            fold = migrate._PleFold("enc", ((PLE_INDEX_ASSET, refs[0][1]), refs[1]), index)
            migrated = migrate._ple_assets(repo, fold, PLE_BLOCK_BYTES, where)
        direct = ple_assets(
            storage=ple.storage,
            tokens=ple.tokens,
            layers=ple.layers,
            dim=ple.dim,
            embed_scale=ple.embed_scale,
            read_values=lambda begin, end: ple.payloads["values"][begin:end],
            read_scales=lambda begin, end: ple.payloads["scales"][begin:end],
            block_bytes=PLE_BLOCK_BYTES,
        )
        return migrated, direct

    @pytest.mark.parametrize("kind", ["rope_base", "ple", "ple-sidecar"])
    def test_both_routes_place_the_same_assets(self, tmp_path: Path, kind: str) -> None:
        """資産を同梱した容器も両経路でバイト同一（契約の「重み + scale + const + 資産 1 本」）。

        `rope_base` は通常 part に載る資産、PLE は **1 block = 1 part**（`dedicated_part`）の
        資産で、物理配置の規則が別 — どちらも踏む。PLE は在処の引き方が 2 つあるので
        （旧 manifest の `assets` / 索引が名乗るファイル名）、**両方**を直接書きと突き合わせる
        — 資産の block id は呼び手が渡した dict の順で決まる（`_plan_asset_parts` は並べ直さない）
        ので、経路ごとに並びが割れると「同じ値・別バイトの容器」ができる。
        """
        migrated_assets, direct_assets = self._asset_material(tmp_path, kind)
        graph, tensors, scales, _ = self._material("f32")
        stored = stored_model(graph, tensors, weight_dtype="f32", weight_scales=scales)
        left = publish_container(
            tmp_path / "left" / "model.f32.krm",
            stored.graph,
            stored.tensors,
            stored.bindings,
            graph_name="enc",
            provenance=PROVENANCE,
            assets=migrated_assets,
        )
        right = publish_container(
            tmp_path / "right" / "model.f32.krm",
            stored.graph,
            stored.tensors,
            stored.bindings,
            graph_name="enc",
            provenance=PROVENANCE,
            assets=direct_assets,
        )

        assert [path.read_bytes() for path in left.parts] == [
            path.read_bytes() for path in right.parts
        ]
        # 資産が本当に載っている（空の `assets` どうしを比べていない）ことの対。
        assert sorted(migrated_assets) == sorted(direct_assets)
        assert migrated_assets

    def test_the_asset_gate_is_not_vacuous(self, tmp_path: Path) -> None:
        """恒真化の門 — 資産の payload が 1 バイト違えば part 列は動く。"""
        migrated_assets, _ = self._asset_material(tmp_path, "rope_base")
        graph, tensors, scales, _ = self._material("f32")
        stored = stored_model(graph, tensors, weight_dtype="f32", weight_scales=scales)
        original = publish_container(
            tmp_path / "a" / "model.f32.krm",
            stored.graph,
            stored.tensors,
            stored.bindings,
            graph_name="enc",
            provenance=PROVENANCE,
            assets=migrated_assets,
        )
        payload = bytearray(migrated_assets[ROPE_BASE_ASSET].payload())
        payload[-1] ^= 0xFF
        flipped = publish_container(
            tmp_path / "b" / "model.f32.krm",
            stored.graph,
            stored.tensors,
            stored.bindings,
            graph_name="enc",
            provenance=PROVENANCE,
            assets={ROPE_BASE_ASSET: AssetInput(ROPE_BASE_ROLE, len(payload), bytes(payload))},
        )

        assert [path.read_bytes() for path in original.parts] != [
            path.read_bytes() for path in flipped.parts
        ]

    def test_the_gate_is_not_vacuous(self, tmp_path: Path) -> None:
        """恒真化の門 — 出所が 1 文字違えば part 0 のバイト列は動く。

        比較が「どちらの経路も同じ関数を呼んだ」ではなく**実バイト**を見ていることの対。
        """
        graph, tensors, scales, _ = self._material("f32")
        stored = stored_model(graph, tensors, weight_dtype="f32", weight_scales=scales)
        other = publish_container(
            tmp_path / "other" / "model.f32.krm",
            stored.graph,
            stored.tensors,
            stored.bindings,
            graph_name="enc",
            provenance=Provenance(license="mit", writer=PROVENANCE.writer),
        )

        assert [path.read_bytes() for path in other.parts] != self._migrated(tmp_path, "f32")
