"""The whole-input reference fixture (`embeddinggemma2/multimodal_reference.py`) and its helpers.

The fixture is written by an opt-in upstream run; what is pinned here is that the tracked files
hold what the next stage relies on, read without torch or transformers:

- exactly the script's cases (names, content parts, prompt names) and images (names, sizes, the
  bytes `images.py` draws)
- each case's ids run `<bos>` … `<eos>`, each image is one `<|image>` + n × `<|image|>` +
  `<image|>` block at the stated start, in the parts' order, with the image's soft-token count, and
  `tokens` is the length
- each vector is a unit vector of width 768, and Karume's composed graphs stayed within the
  script's bound of it
- the helpers find the soft blocks and refuse a run outside `<|image>` … `<image|>`, and build
  upstream's dict inputs in the parts' order
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from safetensors.numpy import load_file

from embeddinggemma2 import multimodal_reference as reference
from embeddinggemma2.images import IMAGES

BOS, EOS = 2, 1


@pytest.fixture(scope="module")
def fixture() -> tuple[dict, dict[str, np.ndarray]]:
    document = json.loads(reference.FIXTURE_JSON.read_text(encoding="utf-8"))
    tensors = load_file(str(reference.FIXTURE_TENSORS))
    return document, tensors


class TestFixture:
    def test_it_holds_the_scripts_cases(self, fixture) -> None:
        document, _ = fixture

        assert [case["name"] for case in document["cases"]] == [c.name for c in reference.CASES]
        for case, expected in zip(document["cases"], reference.CASES, strict=True):
            assert case["parts"] == [{"type": kind, kind: value} for kind, value in expected.parts]
            assert case.get("prompt") == expected.prompt

    def test_the_writer_writes_the_tracked_json_as_it_stands(self) -> None:
        """The repository's formatter leaves the writer's output as it is (no drift on the next
        regeneration)."""
        text = reference.FIXTURE_JSON.read_text(encoding="utf-8")

        assert reference.fixture_json(json.loads(text)) == text
        assert "e-0" not in text

    def test_it_holds_the_images_bytes_as_images_py_draws_them(self, fixture) -> None:
        document, tensors = fixture

        assert [image["name"] for image in document["images"]] == list(IMAGES)
        for entry in document["images"]:
            image = IMAGES[entry["name"]]
            assert (entry["width"], entry["height"]) == (image.width, image.height)
            assert np.array_equal(tensors[f"image.{image.name}"], image.rgb8())

    def test_every_tensor_is_described(self, fixture) -> None:
        document, tensors = fixture

        expected = {f"image.{name}" for name in IMAGES} | {
            f"{case['name']}.{field}"
            for case in document["cases"]
            for field in ("input_ids", "embedding")
        }
        assert set(tensors) == expected
        assert document["tensors"] == reference.FIXTURE_TENSORS.name

    def test_each_case_runs_bos_to_eos_with_one_block_per_image(self, fixture) -> None:
        document, tensors = fixture
        tokens = document["imageTokens"]
        counts = {image["name"]: image["softTokens"] for image in document["images"]}
        for case in document["cases"]:
            ids = tensors[f"{case['name']}.input_ids"].tolist()
            images = [part["image"] for part in case["parts"] if part["type"] == "image"]

            assert ids[0] == BOS and ids[-1] == EOS, case["name"]
            assert case["tokens"] == len(ids)
            blocks = reference.image_blocks(ids, tokens["begin"], tokens["soft"], tokens["end"])
            assert [
                (block["image"], block["start"], block["softTokens"])
                for block in case["imageBlocks"]
            ] == [(name, start, count) for name, (start, count) in zip(images, blocks, strict=True)]
            assert [count for _, count in blocks] == [counts[name] for name in images]
            assert ids.count(tokens["begin"]) == ids.count(tokens["end"]) == len(images)

    def test_each_vector_is_a_unit_vector_and_the_composition_stayed_within_the_bound(
        self, fixture
    ) -> None:
        document, tensors = fixture
        for case in document["cases"]:
            vector = tensors[f"{case['name']}.embedding"].astype(np.float64)

            assert vector.shape == (768,)
            assert abs(np.linalg.norm(vector) - 1) < 1e-5
            assert case["composition"]["maxAbs"] <= reference.COMPOSITION_TOLERANCE
            assert case["composition"]["cosine"] > 0.9999999

    def test_the_text_and_image_orders_differ(self, fixture) -> None:
        """Text then image and image then text are separate inputs upstream (the key order)."""
        document, _ = fixture
        starts = {case["name"]: case["imageBlocks"][0]["start"] for case in document["cases"]}

        assert starts["image-only"] == starts["image-then-text"] == 2
        assert starts["text-then-image"] > 2
        assert starts["query-text-image"] > starts["text-then-image"] - 1


class TestImageBlocks:
    def test_it_finds_each_run_of_soft_ids(self) -> None:
        ids = [2, 9, 7, 7, 8, 9, 7, 8, 1]

        assert reference.image_blocks(ids, begin=9, soft=7, end=8) == [(2, 2), (6, 1)]

    @pytest.mark.parametrize("ids", [[2, 7, 7, 8, 1], [2, 9, 7, 7, 1], [2, 9, 7, 7]])
    def test_a_run_outside_begin_and_end_is_refused(self, ids: list[int]) -> None:
        with pytest.raises(reference.MultimodalReferenceError, match="not between"):
            reference.image_blocks(ids, begin=9, soft=7, end=8)


class TestUpstreamInput:
    def test_the_parts_become_one_dict_in_their_order(self) -> None:
        pytest.importorskip("PIL")
        case = reference.Case("x", (("image", "noise-tall"), ("text", "t")), "why")

        built = reference.upstream_input(case)

        assert list(built) == ["image", "text"]
        assert built["text"] == "t"
        assert built["image"].size == (48, 112)

    def test_two_images_become_one_list(self) -> None:
        pytest.importorskip("PIL")
        case = reference.Case("x", (("image", "shapes-wide"), ("image", "stripes-edge")), "why")

        built = reference.upstream_input(case)

        assert list(built) == ["image"]
        assert [image.size for image in built["image"]] == [(128, 64), (1190, 17)]

    def test_a_repeated_text_part_is_refused(self) -> None:
        case = reference.Case("x", (("text", "a"), ("image", "noise-tall"), ("text", "b")), "why")

        with pytest.raises(reference.MultimodalReferenceError, match="no dict form"):
            reference.upstream_input(case)
