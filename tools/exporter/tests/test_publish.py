"""配布形を据える 3 段（`karume.publish`）— 書く → 読み直して検証 → 据え替え。

ここが固定するのは**据え替えの規律**そのもの（原子性・後始末・読み直し検証の検出力）で、
容器のバイト列の規則は `test_container.py`、export の一本道は `test_pipeline.py` の担当。

故障注入は「書き出しの途中で落とす」「据え替えの rename を落とす」「渡した束縛だけを壊す」の
3 系統。どれも**呼び出しが書いていない現物**（前回の成果物）が巻き添えにならないことまで見る。
"""

from __future__ import annotations

import os
from collections.abc import Buffer, Callable, Iterator, Mapping
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from container_fixture import split_assets
from ir_fixtures import FIXTURE_PROVENANCE, fixture_spec

from karume import publish
from karume.container import AssetInput, ContainerFormatError, ReadContainer
from karume.emit import FixedQuantizedWeight, StoredModel, stored_model
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue
from karume.publish import PublishError, publish_container

#: 前回の成果物の目印（`krg` の中身は読まないので、バイト列は何でもよい）。
SENTINEL = b"previous"

GRAPH_NAME = "publish"


def material(storage: str = "i8") -> StoredModel:
    """合成の部品 1 つを `publish_container` へ渡せる 3 点へ落とす。"""
    graph, tensors, scales, overrides = fixture_spec("publish", storage)
    return stored_model(
        graph,
        tensors,
        weight_dtype=storage,
        weight_scales=scales,
        weight_dtype_overrides=overrides,
    )


def publish_material(
    directory: Path, stored: StoredModel, *, graph_path: Path | None = None
) -> publish.PublishResult:
    return publish_container(
        directory / "model.krm",
        stored.graph,
        stored.tensors,
        stored.bindings,
        graph_name=GRAPH_NAME,
        provenance=FIXTURE_PROVENANCE,
        graph_path=graph_path,
    )


class _Exploding(Mapping[str, Buffer]):
    """引かれた瞬間に落ちるテンソルの口（書き出しの途中で落ちる形を作る）。"""

    def __init__(self, inner: Mapping[str, Buffer]) -> None:
        self._inner = inner

    def __getitem__(self, key: str) -> Buffer:
        raise OSError(f"実体を読めない: {key}")

    def __iter__(self) -> Iterator[str]:
        return iter(self._inner)

    def __len__(self) -> int:
        return len(self._inner)


class TestTheGraphFileIsPlacedAtomically:
    """`krg` も `krm` と同じ 3 段（一時名 → `os.replace`）を通る。

    最終名へ直接書くと、途中で落ちた回に**切り詰められた `krg`** が最終名に残る（プロセスの
    強制終了では例外経路の後始末すら走らない）。
    """

    def test_a_failed_swap_leaves_the_previous_graph_file_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        graph_path = tmp_path / "model.krg"
        graph_path.write_bytes(SENTINEL)
        real = os.replace

        def failing(src: object, dst: object) -> None:
            if Path(str(dst)) == graph_path:
                raise OSError("据え替えに失敗した")
            real(src, dst)  # type: ignore[arg-type]

        monkeypatch.setattr(publish.os, "replace", failing)

        with pytest.raises(OSError, match="据え替えに失敗した"):
            publish_material(tmp_path, material(), graph_path=graph_path)

        # 最終名へ直接書いていれば、ここで前回のバイト列は既に上書きされている。
        assert graph_path.read_bytes() == SENTINEL
        # 一時名（`.partial`）の残骸も残さない。
        assert sorted(entry.name for entry in tmp_path.iterdir()) == ["model.krg"]

    def test_a_successful_publish_replaces_the_graph_file(self, tmp_path: Path) -> None:
        """対照 — 通った回は新しい `krg` が据わる（上の門が「常に消さない」ではない）。"""
        graph_path = tmp_path / "model.krg"
        graph_path.write_bytes(SENTINEL)

        result = publish_material(tmp_path, material(), graph_path=graph_path)

        assert result.graph == graph_path
        assert graph_path.read_bytes() != SENTINEL


class TestTheCleanupTouchesOnlyWhatItWrote:
    """例外経路が消すのは**この呼び出しが据えた現物**だけ。

    `graph_path` を無条件に消すと、書き出しが `krg` を抜く前に落ちた回に前回の `krg` が
    道連れになる — 自分が触っていないファイルを消すのは後始末ではなく破壊。
    """

    def test_a_failure_before_the_graph_is_extracted_keeps_the_previous_graph_file(
        self, tmp_path: Path
    ) -> None:
        graph_path = tmp_path / "model.krg"
        graph_path.write_bytes(SENTINEL)
        stored = material()

        with pytest.raises(OSError, match="実体を読めない"):
            publish_container(
                tmp_path / "model.krm",
                stored.graph,
                _Exploding(stored.tensors),
                stored.bindings,
                graph_name=GRAPH_NAME,
                provenance=FIXTURE_PROVENANCE,
                graph_path=graph_path,
            )

        assert graph_path.read_bytes() == SENTINEL
        assert sorted(entry.name for entry in tmp_path.iterdir()) == ["model.krg"]


class TestTheScaleAgreement:
    """渡した束縛と書いた容器の供給が **scale の有無**で食い違ったら落とす。

    片方だけ `None` を素通しすると、「宣言 i4 / 実体 i8」の相方（scale を持つはずの席が
    scale 無しで据わる）がこの門を抜ける。被験体は**渡す束縛表**だけを壊した組で、書いた容器
    そのものは正しい — 読み直し検証以外に検出器が無い形。
    """

    def _quantized_key(self, stored: StoredModel) -> str:
        keys = sorted(
            key for key, encoding in stored.bindings.items() if encoding.scale_key is not None
        )
        assert keys, "i8 の合成に量子化席が 1 つも無い"
        return keys[0]

    def test_a_binding_without_the_scale_key_fails_loudly(self, tmp_path: Path) -> None:
        stored = material("i8")
        key = self._quantized_key(stored)
        read_back, bound = _published(tmp_path, stored)
        doctored = {**stored.bindings, key: replace(stored.bindings[key], scale_key=None)}

        with pytest.raises(PublishError, match="scale"):
            publish._assert_payloads_match(read_back, bound, doctored, stored.tensors)

    def test_the_matching_binding_passes(self, tmp_path: Path) -> None:
        """対照 — 束縛が合っていれば通る（上の門が「常に落ちる」ではない）。"""
        stored = material("i8")
        read_back, bound = _published(tmp_path, stored)

        checked = publish._assert_payloads_match(read_back, bound, stored.bindings, stored.tensors)

        assert checked > len(bound.supplies)


def _published(tmp_path: Path, stored: StoredModel):
    """合成の部品を据え、読み直した容器と合流結果を返す（門の単体試験の土台）。"""
    result = publish_material(tmp_path, stored)
    read_back = publish.read_container(list(result.parts))
    return read_back, publish.bind_graphs(read_back.graph, read_back.model)[GRAPH_NAME]


class TestThePayloadDigest:
    """検証 1 の本体: 書いた payload の sha256 を、渡した実体と突き合わせる。

    被験体は**渡した実体**だけを 1 バイト壊した組（書いた容器は正しい）— 書き手が 2 度引く
    `Mapping` から違うバイト列が来た形と同じで、読み直し検証以外に検出器が無い。
    """

    @staticmethod
    def _flipped(raw: Buffer) -> bytes:
        flipped = bytearray(memoryview(raw).cast("B"))
        flipped[0] ^= 0xFF
        return bytes(flipped)

    def test_a_weight_that_differs_from_the_written_payload_fails_loudly(
        self, tmp_path: Path
    ) -> None:
        stored = material("i8")
        read_back, bound = _published(tmp_path, stored)
        key = sorted(bound.supplies)[0]
        doctored = {**stored.tensors, key: self._flipped(stored.tensors[key])}

        with pytest.raises(
            PublishError, match=rf"initializer '{key}': payload の sha256 が渡した実体と違う"
        ):
            publish._assert_payloads_match(read_back, bound, stored.bindings, doctored)

    def test_a_scale_that_differs_from_the_written_payload_fails_loudly(
        self, tmp_path: Path
    ) -> None:
        stored = material("i8")
        read_back, bound = _published(tmp_path, stored)
        name = next(name for name, supply in sorted(bound.supplies.items()) if supply.scale)
        scale_key = stored.bindings[name].scale_key
        assert scale_key is not None
        doctored = {**stored.tensors, scale_key: self._flipped(stored.tensors[scale_key])}

        with pytest.raises(PublishError, match=r"の scale .*payload の sha256 が渡した実体と違う"):
            publish._assert_payloads_match(read_back, bound, stored.bindings, doctored)


class TestTheAssetAgreement:
    """検証 2: 書いた資産を、渡した宣言（役割・論理長）と payload に突き合わせる。

    `rope_base` は 37 バイト（0x00 の詰め物 3 バイトが要る奇数長）なので、詰め物の分岐も踏める。
    """

    @staticmethod
    def _published_with_assets(tmp_path: Path) -> ReadContainer:
        stored = material()
        result = publish_container(
            tmp_path / "model.krm",
            stored.graph,
            stored.tensors,
            stored.bindings,
            graph_name=GRAPH_NAME,
            provenance=FIXTURE_PROVENANCE,
            assets=split_assets(),
        )
        return publish.read_container(list(result.parts))

    def test_the_passed_assets_agree_with_the_written_ones(self, tmp_path: Path) -> None:
        """対照 — 正しい資産なら突き合わせた本数が返る（下の門が「常に落ちる」ではない）。"""
        read_back = self._published_with_assets(tmp_path)

        assert publish._assert_assets_match(read_back, split_assets()) == 2

    @pytest.mark.parametrize(
        ("doctor", "message"),
        [
            (
                lambda assets: {"absent": assets["rope_base"]},
                "資産 'absent' が書いた容器の宣言に無い",
            ),
            (
                lambda assets: {"rope_base": assets["rope_base"]._replace(role="ple-index")},
                "資産 'rope_base': 役割が 'rope-base'",
            ),
            (
                lambda assets: {"rope_base": assets["rope_base"]._replace(length=36)},
                "資産 'rope_base': 宣言の論理長が 37",
            ),
            (
                lambda assets: {
                    "rope_base": assets["rope_base"]._replace(
                        payload=b"\xff" + bytes(assets["rope_base"].payload)[1:]
                    )
                },
                "資産 'rope_base': payload が渡したバイト列と違う",
            ),
            (
                # 書き手へ渡した回と長さが変わった呼び出し（引かれるたびに同じバイト列 MUST）。
                lambda assets: {
                    "rope_base": assets["rope_base"]._replace(
                        payload=lambda: bytes(assets["rope_base"].payload) + b"\x00"
                    )
                },
                "資産 'rope_base': 引き直した実体が 38 バイト",
            ),
        ],
        ids=["undeclared", "role", "length", "payload", "repulled-length"],
    )
    def test_a_disagreeing_asset_fails_loudly(
        self,
        tmp_path: Path,
        doctor: Callable[[dict[str, AssetInput]], dict[str, AssetInput]],
        message: str,
    ) -> None:
        read_back = self._published_with_assets(tmp_path)

        with pytest.raises(PublishError, match=message):
            publish._assert_assets_match(read_back, doctor(split_assets()))

    def test_a_tail_pad_that_is_not_zero_fails_loudly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        read_back = self._published_with_assets(tmp_path)
        assert read_back.model is not None
        pad_block = read_back.model.assets["rope_base"].block
        real = read_back.block

        def doctored(block_id: str) -> bytes:
            raw = real(block_id)
            return raw[:-1] + b"\x01" if block_id == pad_block else raw

        monkeypatch.setattr(read_back, "block", doctored)

        with pytest.raises(PublishError, match="資産 'rope_base': 末尾の詰め物が 0x00 でない"):
            publish._assert_assets_match(read_back, split_assets())


def ternary_material(packed: bytes) -> StoredModel:
    """`ternary` を宣言した 2 行 × 16 要素（1 行 4 バイト）の部品。

    三値の量子化器はまだ無い（段 6）ので、i2 の固定重みを通し、束縛の codec だけを `ternary` へ
    差し替える（値の詰め方と scale は `int2-off` と同じ — container-v1 §6.3）。
    """
    graph = IrGraph(
        symbols=[],
        inputs=[IrInput(name="ids", dtype="i32", shape=[2])],
        outputs=["y"],
        initializers={"w": IrInitializer(tensor="w", storage=IrStorage(dtype="f32"))},
        values={"w": IrValue(dtype="f32", shape=[2, 16]), "y": IrValue(dtype="f32", shape=[2, 16])},
        nodes=[IrNode(op="embedding", ins=["w", "ids"], outs=["y"], attrs={"padding_idx": -1})],
    )
    fixed = FixedQuantizedWeight(
        dtype="i2",
        packed=torch.frombuffer(bytearray(packed), dtype=torch.uint8).reshape(2, 4),
        scale=torch.tensor([[0.375], [1.25]]),
    )
    stored = stored_model(
        graph,
        {"w": torch.empty(2, 16, dtype=torch.float32, device="meta")},
        fixed_weights={"w": fixed},
    )
    bindings = {
        key: replace(encoding, codec="ternary") if encoding.codec == "int2-off" else encoding
        for key, encoding in stored.bindings.items()
    }
    return replace(stored, bindings=bindings)


#: 全 2 bit コードが {1, 2, 3} の 2 行ぶん（0x6D = 01 10 11 01 など）。
TERNARY_PACKED = bytes([0x6D, 0xB6, 0xDB, 0x79] * 2)


class TestTheTernaryCodes:
    """検証 4: `ternary` の payload にコード 0 があれば据えない（container-v1 §6.3）。

    TS の読み手はロード時に同じ検査で落とすので、ここが無いと「書けて据わったが、利用者の
    `openContainer` で初めて落ちる」容器が配布形になる。
    """

    def test_a_payload_whose_codes_are_all_in_1_to_3_is_placed(self, tmp_path: Path) -> None:
        result = publish_material(tmp_path, ternary_material(TERNARY_PACKED))

        assert result.parts
        assert all(path.is_file() for path in result.parts)

    def test_a_payload_with_a_code_0_is_refused_and_nothing_is_placed(self, tmp_path: Path) -> None:
        packed = bytearray(TERNARY_PACKED)
        packed[5] = 0b00_10_10_10

        with pytest.raises(
            ContainerFormatError,
            match=r"initializer 'w': ternary の payload にコード 0.*（バイト 5）",
        ):
            publish_material(tmp_path, ternary_material(bytes(packed)))

        assert list(tmp_path.iterdir()) == []
