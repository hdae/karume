"""`birefnet/matte_fixture.py` の約束事 — git 管理の α マットの後処理の golden が、今のライブラリで
系列ごとの上流の関数（量子化だけを四捨五入へ差し替えたもの）を呼んだ結果であること。

TS 側の後処理（`matte.ts` の 2 つの手順）は fixture と突き合わせて門にする。fixture の値は
torch / torchvision / Pillow / numpy の版で決まるので、git にある入力から今の版で作り直した組と、
git にある組（入力表・ケース表・テンソル）が一致することをここで見る — 版を上げて値が変わったのに
fixture を作り直し忘れた状態を赤にする。

ただし lucida（torch の CPU の f32 の σ と resize）の最終 bit は CPU の命令セットでも変わるので、
そのバイト一致は参照環境専用（docs/limitations.md「sha256 参照門は参照環境専用」の CPU 側の原則）。
JSON と hr（Pillow の経路）は x86 の 3 つの命令セット（AVX-512 / AVX2 / SIMD なし）で一致を実測
（2026-10-15）したので、どの環境でもバイト一致を要求する。
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from birefnet import matte_fixture

pytest.importorskip("PIL")


@pytest.fixture(scope="module")
def committed() -> tuple[dict, dict[str, np.ndarray]]:
    from safetensors.numpy import load_file

    directory = matte_fixture.FIXTURE_DIR
    document = json.loads((directory / matte_fixture.FIXTURE_JSON).read_text(encoding="utf-8"))
    return document, load_file(directory / matte_fixture.FIXTURE_TENSORS)


@pytest.fixture(scope="module")
def rebuilt() -> tuple[dict, dict[str, np.ndarray]]:
    """git にある入力から今のライブラリで作り直した組。"""
    return matte_fixture.build_fixture(matte_fixture.committed_real_inputs())


def _mismatched(
    committed: dict[str, np.ndarray], rebuilt: dict[str, np.ndarray], *, lucida: bool
) -> list[str]:
    return [
        name
        for name, tensor in rebuilt.items()
        if name.endswith(".lucida") == lucida
        and (committed[name].dtype != tensor.dtype or not np.array_equal(committed[name], tensor))
    ]


class TestTheCommittedFixture:
    def test_its_document_and_hr_are_what_todays_libraries_produce(self, committed, rebuilt):
        document, tensors = committed
        rebuilt_document, rebuilt_tensors = rebuilt

        assert document == rebuilt_document
        assert sorted(tensors) == sorted(rebuilt_tensors)
        assert _mismatched(tensors, rebuilt_tensors, lucida=False) == []

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="lucida の torch CPU 出力は最終 bit が CPU の命令セットで変わる — "
        "バイト一致は参照環境専用",
    )
    def test_its_lucida_is_what_todays_torch_produces(self, committed, rebuilt):
        _, tensors = committed
        _, rebuilt_tensors = rebuilt

        assert sorted(tensors) == sorted(rebuilt_tensors)
        assert _mismatched(tensors, rebuilt_tensors, lucida=True) == []

    def test_it_records_the_versions_of_the_libraries_that_decide_the_values(self, committed):
        document, _ = committed

        assert document["source"]["versions"] == matte_fixture.library_versions()
        assert set(document["source"]["versions"]) == {"torch", "torchvision", "pillow", "numpy"}

    def test_every_input_and_case_has_tensors_of_its_size_and_dtype(self, committed):
        document, tensors = committed

        for row in document["inputs"]:
            logits = tensors[f"{row['name']}.logits"]
            assert (logits.dtype, logits.shape) == (np.float32, (row["height"], row["width"]))
        for case in document["cases"]:
            size = (case["outHeight"], case["outWidth"])
            assert (tensors[f"{case['name']}.hr"].dtype, tensors[f"{case['name']}.hr"].shape) == (
                np.uint8,
                size,
            )
            lucida = tensors[f"{case['name']}.lucida"]
            assert (lucida.dtype, lucida.shape) == (np.float32, size)
        assert len(tensors) == len(document["inputs"]) + 2 * len(document["cases"])

    def test_the_real_inputs_carry_their_provenance(self, committed):
        document, _ = committed
        real = [row for row in document["inputs"] if row["provenance"] is not None]

        crops = {row["name"] for row in real if row["provenance"]["kind"] == "crop"}
        assert crops == {
            f"real-{case}-{image.removeprefix('photo-')}"
            for _series, case in matte_fixture.REAL_SERIES
            for image in matte_fixture.REAL_IMAGES
        }
        for row in real:
            provenance = row["provenance"]
            assert provenance["file"].startswith("outputs/verify/")
            assert len(provenance["sha256"]) == 64
            assert (provenance["width"], provenance["height"]) == (row["width"], row["height"])
            assert all(isinstance(provenance[key], int) for key in ("x", "y"))

    def test_the_lucida_epsilon_keeps_its_margin_over_the_measured_difference(self, committed):
        document, _ = committed

        assert document["lucida"]["epsilon"] == matte_fixture.LUCIDA_EPSILON
        assert 0 < document["lucida"]["measuredMaxDelta"] <= matte_fixture.LUCIDA_EPSILON / 2

    def test_it_stays_small(self):
        directory = matte_fixture.FIXTURE_DIR
        total = sum(
            (directory / name).stat().st_size
            for name in (matte_fixture.FIXTURE_JSON, matte_fixture.FIXTURE_TENSORS)
        )

        assert total < matte_fixture.MAX_TOTAL_BYTES


class TestTheRealSeries:
    def test_each_series_is_the_default_seat_of_its_checkpoint(self):
        """実画像の切り出しの元は、checkpoint ごとに 1 本ずつ、配布の既定席の 1024² の系列。

        退役した席（HR の f32）や既定でない席の logit から fixture を採らない — 系列名の綴りを
        `birefnet_series_name` で逆に辿った席を、`birefnet_default_quant` と突き合わせる。
        """
        from birefnet.card import BIREFNET_CHECKPOINTS
        from birefnet.distribution import birefnet_default_quant

        resolved = [
            matte_fixture.real_series_seat(series) for series, _case in matte_fixture.REAL_SERIES
        ]

        assert sorted(checkpoint for checkpoint, _model, _dtype in resolved) == sorted(
            BIREFNET_CHECKPOINTS
        )
        for checkpoint, model, dtype in resolved:
            assert model == "1024"
            assert dtype == birefnet_default_quant(checkpoint)


@pytest.fixture(scope="module")
def shapes() -> list[tuple[int, int, int, int]]:
    """ケースごとの（入力の高さ・幅・出力の高さ・幅）。"""
    inputs, cases = matte_fixture.build_cases(matte_fixture.committed_real_inputs())
    by_name = {entry.name: entry.logits.shape for entry in inputs}
    return [(*by_name[case.input], case.out_height, case.out_width) for case in cases]


class TestTheGeometry:
    """ケースが掃くはずの経路が表に残っていること（ケースを削って経路が抜ける形を赤にする）。"""

    def test_it_never_targets_the_graph_resolution(self, shapes):
        assert all((oh, ow) != (1024, 1024) for _, _, oh, ow in shapes)

    def test_it_covers_up_down_mixed_single_axis_and_identity(self, shapes):
        assert any(oh > h and ow > w for h, w, oh, ow in shapes)
        assert any(oh < h and ow < w for h, w, oh, ow in shapes)
        assert any(ow < w and oh > h for h, w, oh, ow in shapes)
        assert any(h == oh and w != ow for h, w, oh, ow in shapes)
        assert any(w == ow and h != oh for h, w, oh, ow in shapes)
        assert any(h == oh and w == ow and h > 1 and w > 1 for h, w, oh, ow in shapes)

    def test_the_production_coefficient_tables_are_present(self):
        _, cases = matte_fixture.build_cases(matte_fixture.committed_real_inputs())
        names = {case.name for case in cases}

        for series in ("birefnet-hr", "lucida"):
            assert {
                f"real-{series}-portrait-row.down-832",
                f"real-{series}-portrait-row.up-2048",
                f"real-{series}-portrait-column.down-480",
                f"real-{series}-portrait-column.up-2048",
            } <= names


class TestTheUpstreamProcedures:
    def test_hr_rounds_half_up_where_the_upstream_truncates(self):
        logits = np.asarray([[7.0]], dtype=np.float32)  # 255σ(7) = 254.77

        assert matte_fixture.hr_alpha(logits, 1, 1).tolist() == [[255]]
        assert matte_fixture.hr_alpha(logits, 1, 1, quantize=matte_fixture.truncate).tolist() == [
            [254]
        ]

    def test_hr_with_truncation_is_the_upstream_call_chain_itself(self):
        """量子化を切り捨てに戻すと、上流の `ToPILImage()(pred).resize(size)` とビット一致する
        （差し替えたのが量子化だけであることの裏付け）。

        σ の精度は写さない（ADR 0125 決定 5 — 上流の半精度の σ ではなく f64 の σ で比べる）ので、
        裏付けの範囲は Pillow の段と `ToPILImage` の切り捨てまで。上流の半精度の sigmoid との
        一致はここでは主張しない。"""
        import torch
        from torchvision import transforms

        logits = matte_fixture.committed_real_inputs()[0].logits
        sigma = matte_fixture.sigmoid_f64(logits)

        upstream = transforms.ToPILImage()(torch.from_numpy(sigma)).resize((97, 83))

        np.testing.assert_array_equal(
            matte_fixture.hr_alpha(logits, 97, 83, quantize=matte_fixture.truncate),
            np.asarray(upstream, dtype=np.uint8),
        )

    def test_hr_rejects_samples_whose_rounding_direction_is_not_pinned(self):
        # logit 0 → 255σ = 127.5 ちょうど（k + 0.5 — 丸めの向きが exp の実装に預けられる境目）。
        with pytest.raises(ValueError, match=r"k \+ 0\.5"):
            matte_fixture.hr_alpha(np.asarray([[1.0, 0.0]], dtype=np.float32), 3, 1)

    def test_lucida_refuses_the_width_one_input_that_breaks_torch(self):
        with pytest.raises(ValueError, match="幅 1"):
            matte_fixture.lucida_scaled(np.zeros((8, 1), dtype=np.float32), 1, 5)

    def test_lucida_keeps_the_unrounded_value_in_f32(self):
        scaled = matte_fixture.lucida_scaled(np.asarray([[0.0, 2.0]], dtype=np.float32), 2, 1)

        assert scaled.dtype == np.float32
        np.testing.assert_allclose(scaled[0, 0], 127.5)
