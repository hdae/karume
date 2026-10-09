"""`wan/lanczos_fixture.py` の約束事 — git 管理の LANCZOS の fixture が今の Pillow の出力で
あること。

TS 側の LANCZOS は fixture と uint8 で完全一致を門にする。fixture の値は Pillow の版で決まるので、
今の Pillow で作り直した組と、git にある組（ケース表・テンソル）が一致することをここで見る — 版を
上げて値が変わったのに fixture を作り直し忘れた状態を赤にする。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from wan import lanczos_fixture

pytest.importorskip("PIL")


@pytest.fixture(scope="module")
def committed() -> tuple[dict, dict[str, np.ndarray]]:
    from safetensors.numpy import load_file

    directory = lanczos_fixture.FIXTURE_DIR
    document = json.loads((directory / lanczos_fixture.FIXTURE_JSON).read_text(encoding="utf-8"))
    return document, load_file(directory / lanczos_fixture.FIXTURE_TENSORS)


class TestTheCommittedFixture:
    def test_it_is_what_todays_pillow_produces(self, committed):
        document, tensors = committed

        rebuilt_document, rebuilt_tensors = lanczos_fixture.build_fixture()

        assert document["source"] == rebuilt_document["source"]
        assert document["cases"] == rebuilt_document["cases"]
        assert sorted(tensors) == sorted(rebuilt_tensors)
        mismatched = [
            name
            for name, tensor in rebuilt_tensors.items()
            if not np.array_equal(tensors[name], tensor)
        ]
        assert mismatched == []

    def test_every_case_has_an_input_and_a_resized_tensor_of_its_size(self, committed):
        document, tensors = committed

        for case in document["cases"]:
            name = case["name"]
            assert tensors[f"{name}.input"].shape == (case["height"], case["width"], 3)
            assert tensors[f"{name}.resized"].shape == (case["outHeight"], case["outWidth"], 3)
        assert len(tensors) == 2 * len(document["cases"])

    def test_it_stays_small(self):
        directory = lanczos_fixture.FIXTURE_DIR
        total = sum(
            (directory / name).stat().st_size
            for name in (lanczos_fixture.FIXTURE_JSON, lanczos_fixture.FIXTURE_TENSORS)
        )

        assert total < lanczos_fixture.MAX_TOTAL_BYTES


class TestTheGeometry:
    """ケースが掃くはずの経路が表に残っていること（ケースを削って経路が抜ける形を赤にする）。"""

    def test_the_sweep_covers_the_skipped_axes_and_both_directions(self):
        cases = lanczos_fixture.build_cases()
        shapes = [
            (case.image.shape[0], case.image.shape[1], case.out_height, case.out_width)
            for case in cases
        ]

        assert any(h == oh and w != ow for h, w, oh, ow in shapes)
        assert any(w == ow and h != oh for h, w, oh, ow in shapes)
        assert any(h == oh and w == ow for h, w, oh, ow in shapes)
        assert any(oh > h and ow > w for h, w, oh, ow in shapes)
        assert any(oh < h and ow < w for h, w, oh, ow in shapes)
        assert any(ow < w and oh > h for h, w, oh, ow in shapes)

    def test_the_production_coefficient_tables_are_present(self):
        names = {case.name for case in lanczos_fixture.build_cases()}

        assert {
            "row-832-1280",
            "row-832-2219",
            "row-832-277",
            "column-480-738",
            "column-480-1280",
            "column-480-160",
            "column-480-704",
        } <= names
