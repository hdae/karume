"""`wan/umt5_export.py`（meta で trace し、重みを checkpoint から 1 本ずつ i8 にする形）の約束事
（ADR 0119 段 10b — 決定 5 / 6）。

固定するのは、壊れても例外が出ず**容器のバイトだけが静かにずれる**側:

- 重みを持たない（meta）上流で trace したグラフが、実重みの export と JSON で同一
- 行の塊ごとに作った i8（`fixed_weights`）で書いた容器が、素直な形（f32 で `fake_quant_i8` →
  `weight_dtype="i8"` + 表の F32 明示）で書いた容器と**全 part のバイトで一致**する — 塊の境目が
  行の途中に来ない・tied な語彙埋め込みを `shared.weight` から読む・表を丸めない、の全部をここで縛る
- 格納の内訳: linear と語彙埋め込みは i8、相対位置の表と RMSNorm は f32
- checkpoint のキーが決まらない・F32 でない、は fail loudly
- 正式な書き口（{@link wan.umt5_export.write_container}）が上の素直な形と同じバイトを書き、既存の
  容器との照合（{@link wan.umt5_export.assert_same_container}）が 1 バイトの違いも part で名指しする

乱数初期化の小さな umT5（`wan.umt5_probe.TINY_CONFIG`）を `save_pretrained` した checkpoint で回す —
実重みは読まない。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file

from karume.pipeline import publish_model
from wan import umt5_export as ue
from wan import umt5_patch as up
from wan import umt5_probe as probe

LAYERS = int(probe.TINY_CONFIG["num_layers"])

#: 小模型の linear の本数（層ごとに q / k / v / o と wi_0 / wi_1 / wo）。
LINEARS_PER_LAYER = 7

#: 塊の行数（語彙 384・d_ff 160・d_model 64 のどれも割り切らない — 半端な塊を必ず踏む）。
CHUNK_ROWS = 7


@pytest.fixture(scope="module")
def model() -> Any:
    pytest.importorskip("transformers")
    return probe.tiny_model()


@pytest.fixture(scope="module")
def checkpoint_dir(model: Any, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """上流の `save_pretrained` の形（config + 単一の safetensors — tied な重みは片方だけ）。"""
    directory = tmp_path_factory.mktemp("umt5-tiny")
    model.save_pretrained(directory)
    return directory


@pytest.fixture(scope="module")
def prepared(checkpoint_dir: Path) -> ue.Umt5Export:
    return ue.prepare(checkpoint_dir, chunk_rows=CHUNK_ROWS)


def write(path: Path, graph: Any, tensors: Any, **storage: Any) -> list[bytes]:
    """容器を書いて全 part のバイトを返す（出所とグラフ名は正式な書き口と同じ — 差はバイトの
    中身だけに出る）。"""
    publish_model(
        path / ue.MODEL_FILE,
        graph,
        dict(tensors),
        provenance=ue.provenance(),
        graph_name=ue.GRAPH_NAME,
        **storage,
    )
    return read_parts(path)


def read_parts(path: Path) -> list[bytes]:
    return [part.read_bytes() for part in sorted(path.glob("*.krm"))]


class TestCheckpoint:
    def test_the_tied_embedding_is_read_from_the_saved_name(self, checkpoint_dir, model):
        """`encoder.embed_tokens.weight` は checkpoint に `shared.weight` としてしか無い。"""
        checkpoint = ue.Checkpoint(checkpoint_dir)

        mapping = ue.checkpoint_keys(model, ["encoder.embed_tokens.weight"], checkpoint.names())

        assert "encoder.embed_tokens.weight" not in checkpoint.names()
        assert mapping == {"encoder.embed_tokens.weight": "shared.weight"}

    def test_a_missing_key_fails_loudly(self, checkpoint_dir, model):
        checkpoint = ue.Checkpoint(checkpoint_dir)
        table = "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"

        with pytest.raises(ue.Umt5ExportError, match="1 つに決まらない"):
            ue.checkpoint_keys(model, [table], checkpoint.names() - {table})

    def test_a_non_f32_checkpoint_is_rejected(self, checkpoint_dir, tmp_path):
        """丸めの出発点が F32 でない checkpoint は読まない（bf16 から i8 にすると別の値になる）。"""
        tensors = load_file(str(checkpoint_dir / ue.CHECKPOINT_SINGLE))
        save_file(
            {key: value.to(torch.bfloat16) for key, value in tensors.items()},
            str(tmp_path / ue.CHECKPOINT_SINGLE),
        )

        with pytest.raises(ue.Umt5ExportError, match="F32 だけ"):
            ue.Checkpoint(tmp_path).read_rows("shared.weight", 0, 1)


class TestMetaTrace:
    def test_the_graph_is_the_real_weight_export(self, prepared, model):
        """meta の重みで辿ったグラフ = 実重み（f32 で丸める前）の export のグラフ。"""
        wrapper = up.Umt5EncoderTokens(model)
        graph, _ = ue.trace(wrapper)

        assert prepared.graph.to_json() == graph.to_json()

    def test_only_the_weights_stay_meta(self, prepared):
        """meta のまま残るのは i8 の対象だけで、残りの重みは checkpoint の実体。"""
        meta = sorted(key for key, value in prepared.tensors.items() if value.is_meta)

        assert meta == sorted(prepared.fixed)
        assert all(not prepared.tensors[key].is_meta for key in prepared.plain)


class TestStorage:
    def test_linear_and_embedding_are_i8_tables_and_norms_f32(self, prepared):
        kinds = sorted({ue.storage_kind(key) for key in prepared.fixed})
        plain = sorted({ue.storage_kind(key) for key in prepared.plain})

        assert len(prepared.fixed) == LAYERS * LINEARS_PER_LAYER + 1
        assert kinds == ["embed_tokens", "linear"]
        assert plain == ["norm", up.RELATIVE_BIAS_ATTRIBUTE]
        assert sum(ue.storage_kind(key) == "norm" for key in prepared.plain) == 2 * LAYERS + 1

    def test_the_container_matches_the_whole_tensor_path_byte_for_byte(
        self, prepared, model, tmp_path
    ):
        """正式な書き口（行の塊ごとの i8）が、素直な形（f32 で丸め → i8 + 表の F32 明示）と
        全 part のバイトで一致する。"""
        whole = probe.export_probe(model)

        expected = write(
            tmp_path / "whole",
            whole.graph,
            whole.tensors,
            weight_dtype="i8",
            weight_scales=whole.scales,
            weight_dtype_overrides=whole.overrides,
        )
        (tmp_path / "rows").mkdir()
        ue.write_container(prepared, tmp_path / "rows" / ue.MODEL_FILE)
        actual = read_parts(tmp_path / "rows")

        assert len(actual) == len(expected) > 0
        assert actual == expected

    def test_a_shifted_chunk_breaks_the_match(self, prepared, model):
        """対（非恒真）: 塊の scale を 1 行ずらすと packed が素直な形と割れる。"""
        whole = probe.export_probe(model)
        key = "encoder.embed_tokens.weight"
        fixed = prepared.fixed[key]
        weight = whole.tensors[key]

        restored = fixed.packed.to(torch.float32) * fixed.scale
        shifted = fixed.packed.to(torch.float32) * torch.roll(fixed.scale, 1, dims=0)

        assert torch.equal(restored, weight)
        assert not torch.equal(shifted, weight)


class TestTheSeriesCheck:
    """`write --check` の照合（書いた容器と既存の容器を part 列の sha256 で突き合わせる）。"""

    @pytest.fixture
    def pair(self, prepared, tmp_path) -> tuple[Path, Path]:
        written, existing = tmp_path / "written", tmp_path / "existing"
        for place in (written, existing):
            place.mkdir()
            ue.write_container(prepared, place / ue.MODEL_FILE)
        return written / ue.MODEL_FILE, existing / ue.MODEL_FILE

    def test_the_same_material_writes_the_same_container(self, pair):
        written, existing = pair

        ue.assert_same_container(written, existing)
        assert len(ue.part_digests(written)) > 1

    def test_one_flipped_byte_is_named_by_its_part(self, pair):
        """故障注入: 既存の最後の part の 1 バイトを反転すると、その添字で落ちる。"""
        written, existing = pair
        parts = sorted(existing.parent.glob("*.krm"))
        last = bytearray(parts[-1].read_bytes())
        last[-1] ^= 0xFF
        parts[-1].write_bytes(bytes(last))

        with pytest.raises(ue.Umt5ExportError, match=rf"part \[{len(parts) - 1}\]"):
            ue.assert_same_container(written, existing)

    def test_another_part_count_is_refused(self, pair, tmp_path):
        """part の本数が違えば、中身を比べる前に落ちる（別の分割で書いた容器）。"""
        written, _ = pair
        single = tmp_path / "single"
        single.mkdir()
        (single / "model-00001-of-00002.krm").write_bytes(b"0")
        (single / "model-00002-of-00002.krm").write_bytes(b"1")

        with pytest.raises(ue.Umt5ExportError, match="part の本数"):
            ue.assert_same_container(written, single / ue.MODEL_FILE)


class TestQuantizeRows:
    @pytest.mark.parametrize("chunk_rows", [1, CHUNK_ROWS, 10_000])
    def test_the_chunk_size_does_not_change_the_values(self, checkpoint_dir, chunk_rows):
        """塊の大きさ（1 行ずつ・半端・全体を 1 塊）に依らず同じ packed と scale。"""
        checkpoint = ue.Checkpoint(checkpoint_dir)
        reference = ue.quantize_rows(checkpoint, "shared.weight", chunk_rows=10_000)

        actual = ue.quantize_rows(checkpoint, "shared.weight", chunk_rows=chunk_rows)

        assert torch.equal(actual.packed, reference.packed)
        assert torch.equal(actual.scale, reference.scale)


class TestMemoryMonitor:
    def test_a_stage_sees_what_it_allocates(self):
        """段の中で確保して触った 256 MiB が、その段の山（VmHWM）に出る。"""
        with ue.MemoryMonitor(interval=0.05) as monitor:
            with monitor.stage("idle") as idle:
                pass
            with monitor.stage("allocate") as allocate:
                block = torch.ones(64 << 20, dtype=torch.float32)
                del block

        assert [record.name for record in monitor.records] == ["idle", "allocate"]
        assert allocate.hwm - idle.hwm > 0.2
        assert allocate.seconds >= 0
