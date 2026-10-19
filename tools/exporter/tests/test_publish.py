"""The three steps that put a distribution in place (`karume.publish`) — write → read back and
verify → swap into place.

What this file pins is **the discipline of the swap** itself (atomicity, cleanup, the detection
power of the read-back verification); the byte rules of the container belong to
`test_container.py`, the end-to-end export path to `test_pipeline.py`.

Fault injection comes in three kinds: "fail partway through writing", "fail the rename of the
swap" and "break only the passed binding". Each also checks that **files the call did not write**
(the previous artifacts) are not taken down with it.
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

#: The marker of the previous artifact (the content of the `krg` is not read, so any bytes do).
SENTINEL = b"previous"

GRAPH_NAME = "publish"


def material(storage: str = "i8") -> StoredModel:
    """Turns one synthetic component into the three pieces `publish_container` accepts."""
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
    """A tensor source that fails the moment it is read (makes writing fail partway through)."""

    def __init__(self, inner: Mapping[str, Buffer]) -> None:
        self._inner = inner

    def __getitem__(self, key: str) -> Buffer:
        raise OSError(f"cannot read the data: {key}")

    def __iter__(self) -> Iterator[str]:
        return iter(self._inner)

    def __len__(self) -> int:
        return len(self._inner)


class TestTheGraphFileIsPlacedAtomically:
    """The `krg` goes through the same three steps as the `krm` (temporary name → `os.replace`).

    Writing straight to the final name leaves **a truncated `krg`** under the final name when a run
    fails partway (a forced process kill does not even run the cleanup of the exception path).
    """

    def test_a_failed_swap_leaves_the_previous_graph_file_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        graph_path = tmp_path / "model.krg"
        graph_path.write_bytes(SENTINEL)
        real = os.replace

        def failing(src: object, dst: object) -> None:
            if Path(str(dst)) == graph_path:
                raise OSError("the swap failed")
            real(src, dst)  # type: ignore[arg-type]

        monkeypatch.setattr(publish.os, "replace", failing)

        with pytest.raises(OSError, match="the swap failed"):
            publish_material(tmp_path, material(), graph_path=graph_path)

        # Had it written straight to the final name, the previous bytes would be overwritten by now.
        assert graph_path.read_bytes() == SENTINEL
        # No leftover temporary name (`.partial`) either.
        assert sorted(entry.name for entry in tmp_path.iterdir()) == ["model.krg"]

    def test_a_successful_publish_replaces_the_graph_file(self, tmp_path: Path) -> None:
        """Counterpart — a run that passes puts the new `krg` in place (the check above is not
        "never removes")."""
        graph_path = tmp_path / "model.krg"
        graph_path.write_bytes(SENTINEL)

        result = publish_material(tmp_path, material(), graph_path=graph_path)

        assert result.graph == graph_path
        assert graph_path.read_bytes() != SENTINEL


class TestTheCleanupTouchesOnlyWhatItWrote:
    """The exception path removes only **the files this call put in place**.

    Removing `graph_path` unconditionally would take the previous `krg` down with it when writing
    fails before the `krg` is extracted — removing a file you did not touch is destruction, not
    cleanup.
    """

    def test_a_failure_before_the_graph_is_extracted_keeps_the_previous_graph_file(
        self, tmp_path: Path
    ) -> None:
        graph_path = tmp_path / "model.krg"
        graph_path.write_bytes(SENTINEL)
        stored = material()

        with pytest.raises(OSError, match="cannot read the data"):
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
    """Fails when the passed binding and the supply of the written container disagree on **whether
    there is a scale**.

    Passing `None` on one side only would let the counterpart of "declared i4 / actual i8" (a slot
    that should carry a scale put in place without one) through this check. The subject is a set
    where only **the binding table passed in** is broken, while the written container itself is
    correct — a form with no detector other than the read-back verification.
    """

    def _quantized_key(self, stored: StoredModel) -> str:
        keys = sorted(
            key for key, encoding in stored.bindings.items() if encoding.scale_key is not None
        )
        assert keys, "the i8 synthetic model has no quantized slot"
        return keys[0]

    def test_a_binding_without_the_scale_key_fails_loudly(self, tmp_path: Path) -> None:
        stored = material("i8")
        key = self._quantized_key(stored)
        read_back, bound = _published(tmp_path, stored)
        doctored = {**stored.bindings, key: replace(stored.bindings[key], scale_key=None)}

        with pytest.raises(PublishError, match="scale"):
            publish._assert_payloads_match(read_back, bound, doctored, stored.tensors)

    def test_the_matching_binding_passes(self, tmp_path: Path) -> None:
        """Counterpart — a matching binding passes (the check above is not "always fails")."""
        stored = material("i8")
        read_back, bound = _published(tmp_path, stored)

        checked = publish._assert_payloads_match(read_back, bound, stored.bindings, stored.tensors)

        assert checked > len(bound.supplies)


def _published(tmp_path: Path, stored: StoredModel):
    """Puts the synthetic component in place and returns the read-back container and the merge
    result (the base of the unit tests of the checks)."""
    result = publish_material(tmp_path, stored)
    read_back = publish.read_container(list(result.parts))
    return read_back, publish.bind_graphs(read_back.graph, read_back.model)[GRAPH_NAME]


class TestThePayloadDigest:
    """The core of verification 1: compares the sha256 of the written payload with the passed data.

    The subject is a set where only **the passed data** is broken by one byte (the written
    container is correct) — the same as different bytes coming from a `Mapping` the writer reads
    twice, with no detector other than the read-back verification.
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
            PublishError,
            match=rf"initializer '{key}': the payload sha256 differs from the passed data",
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

        with pytest.raises(
            PublishError, match=r", scale .*: the payload sha256 differs from the passed data"
        ):
            publish._assert_payloads_match(read_back, bound, stored.bindings, doctored)


class TestTheAssetAgreement:
    """Verification 2: compares the written assets with the passed declaration (role, logical
    length) and payload.

    `rope_base` is 37 bytes (an odd length that needs 3 bytes of 0x00 padding), so the padding
    branch is exercised too.
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
        """Counterpart — correct assets return the number compared (the checks below are not "always
        fail")."""
        read_back = self._published_with_assets(tmp_path)

        assert publish._assert_assets_match(read_back, split_assets()) == 2

    @pytest.mark.parametrize(
        ("doctor", "message"),
        [
            (
                lambda assets: {"absent": assets["rope_base"]},
                "Asset 'absent' is not in the written container's declaration",
            ),
            (
                lambda assets: {"rope_base": assets["rope_base"]._replace(role="ple-index")},
                "Asset 'rope_base': role is 'rope-base'",
            ),
            (
                lambda assets: {"rope_base": assets["rope_base"]._replace(length=36)},
                "Asset 'rope_base': declared logical length is 37",
            ),
            (
                lambda assets: {
                    "rope_base": assets["rope_base"]._replace(
                        payload=b"\xff" + bytes(assets["rope_base"].payload)[1:]
                    )
                },
                "Asset 'rope_base': payload differs from the passed bytes",
            ),
            (
                # A call whose length differs from the one passed to the writer (each read MUST
                # return the same bytes).
                lambda assets: {
                    "rope_base": assets["rope_base"]._replace(
                        payload=lambda: bytes(assets["rope_base"].payload) + b"\x00"
                    )
                },
                "Asset 'rope_base': the re-fetched data is 38 bytes",
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

        with pytest.raises(PublishError, match="Asset 'rope_base': trailing padding is not 0x00"):
            publish._assert_assets_match(read_back, split_assets())


def ternary_material(packed: bytes) -> StoredModel:
    """A component of 2 rows × 16 elements (4 bytes per row) that declares `ternary`.

    There is no ternary quantizer yet (step 6 of ADR 0108), so fixed i2 weights go through and
    only the codec of the binding is swapped to `ternary` (the packing and the scale are the same
    as `int2-off` — container-v1 §6.3).
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


#: Two rows whose 2-bit codes are all in {1, 2, 3} (0x6D = 01 10 11 01 and so on).
TERNARY_PACKED = bytes([0x6D, 0xB6, 0xDB, 0x79] * 2)


class TestTheTernaryCodes:
    """Verification 4: a `ternary` payload with a code 0 is not put in place (container-v1 §6.3).

    The TS reader fails on the same check at load time, so without this a container that "was
    written and put in place but fails only in the user's `openContainer`" would become a
    distribution.
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
            match=r"initializer 'w': the ternary payload has code 0.*\(byte 5\)",
        ):
            publish_material(tmp_path, ternary_material(bytes(packed)))

        assert list(tmp_path.iterdir()) == []
