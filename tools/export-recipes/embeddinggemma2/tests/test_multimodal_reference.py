"""The whole-input reference fixture (`embeddinggemma2/multimodal_reference.py`) and its helpers.

The fixture is written by an opt-in upstream run; what is pinned here is that the tracked files
hold what the next stage relies on, read without torch or transformers:

- exactly the script's cases (names, content parts, prompt names), images (names, sizes, the
  bytes `images.py` draws) and audios (names, the samples `waveforms.py` draws, their frames)
- each case's ids run `<bos>` … `<eos>`, each image is one `<|image>` + n × `<|image|>` +
  `<image|>` block and each audio one `<|audio>` + m × `<|audio|>` + `<audio|>` block at the
  stated start, in the parts' order, with its soft-token count, and `tokens` is the length
- each vector is a unit vector of width 768, and Karume's composed graphs stayed within the
  script's bound of it
- the image cases' list holds images alone (the image pipeline reads it as it is)
- the id cases hold ids alone, run `<bos>` … `<eos>` with one block per image, and cover every
  boundary they are named for
- the helpers find the soft blocks and refuse a run outside its begin and end ids, and build
  upstream's dict inputs and its one-message inputs in the parts' order
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from safetensors.numpy import load_file

from embeddinggemma2 import multimodal_reference as reference
from embeddinggemma2.images import IMAGES
from embeddinggemma2.waveforms import WAVEFORMS, soft_tokens, valid_frames

BOS, EOS = 2, 1


@pytest.fixture(scope="module")
def fixture() -> tuple[dict, dict[str, np.ndarray]]:
    document = json.loads(reference.FIXTURE_JSON.read_text(encoding="utf-8"))
    tensors = load_file(str(reference.FIXTURE_TENSORS))
    return document, tensors


class TestFixture:
    def test_it_holds_the_scripts_cases(self, fixture) -> None:
        document, _ = fixture

        for key, cases in (
            ("cases", reference.CASES),
            ("audioCases", reference.AUDIO_CASES),
            ("idCases", reference.ID_CASES),
        ):
            assert [case["name"] for case in document[key]] == [c.name for c in cases]
            for case, expected in zip(document[key], cases, strict=True):
                parts = [{"type": kind, kind: value} for kind, value in expected.parts]
                assert case["parts"] == parts
                assert case.get("prompt") == expected.prompt

    def test_the_image_cases_hold_images_alone(self, fixture) -> None:
        """The image pipeline's checks read every case of `cases`; an audio there would break
        them, so the audio cases are a list of their own."""
        document, _ = fixture

        for case in document["cases"]:
            assert {part["type"] for part in case["parts"]} <= {"text", "image"}, case["name"]
            assert "audioBlocks" not in case
        for case in document["audioCases"]:
            assert "audio" in {part["type"] for part in case["parts"]}, case["name"]

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

    def test_it_holds_the_audios_samples_as_waveforms_py_draws_them(self, fixture) -> None:
        document, tensors = fixture

        assert [audio["name"] for audio in document["audios"]] == list(reference.FIXTURE_WAVEFORMS)
        for entry in document["audios"]:
            waveform = WAVEFORMS[entry["name"]]
            assert (entry["samples"], entry["sampleRate"]) == (waveform.samples, 16_000)
            assert np.array_equal(tensors[f"audio.{waveform.name}"], waveform.f32())
            assert entry["validFrames"] == valid_frames(waveform.samples)
            assert entry["softTokens"] == soft_tokens(entry["validFrames"])
            assert entry["extractorFrames"] >= entry["validFrames"]
            features = tensors[f"audio.{waveform.name}.features"]
            assert features.dtype == np.float32
            assert features.shape == (entry["validFrames"], 128)
            # Log-mel values: the floor's log(1e-3) at the least, finite.
            assert bool(np.isfinite(features).all())
            assert float(features.min()) >= float(np.log(np.float32(1e-3))) - 1e-5

    def test_the_mask_edge_is_among_the_audios(self, fixture) -> None:
        """Stage 2's frame layout is checked on an audio whose frame count is 2 mod 4."""
        document, _ = fixture

        assert {audio["validFrames"] % 4 for audio in document["audios"]} >= {2}

    def test_every_tensor_is_described(self, fixture) -> None:
        document, tensors = fixture

        expected = (
            {f"image.{name}" for name in IMAGES}
            | {f"audio.{audio['name']}" for audio in document["audios"]}
            | {f"audio.{audio['name']}.features" for audio in document["audios"]}
            | {
                f"{case['name']}.{field}"
                for case in (*document["cases"], *document["audioCases"])
                for field in ("input_ids", "embedding")
            }
            | {f"{case['name']}.input_ids" for case in document["idCases"]}
        )
        assert set(tensors) == expected
        assert document["tensors"] == reference.FIXTURE_TENSORS.name

    def test_each_case_runs_bos_to_eos_with_one_block_per_image(self, fixture) -> None:
        document, tensors = fixture
        tokens = document["imageTokens"]
        counts = {image["name"]: image["softTokens"] for image in document["images"]}
        for case in (*document["cases"], *document["idCases"]):
            ids = tensors[f"{case['name']}.input_ids"].tolist()
            images = [part["image"] for part in case["parts"] if part["type"] == "image"]

            assert ids[0] == BOS and ids[-1] == EOS, case["name"]
            assert case["tokens"] == len(ids)
            blocks = reference.soft_blocks(ids, tokens["begin"], tokens["soft"], tokens["end"])
            assert [
                (block["image"], block["start"], block["softTokens"])
                for block in case["imageBlocks"]
            ] == [(name, start, count) for name, (start, count) in zip(images, blocks, strict=True)]
            assert [count for _, count in blocks] == [counts[name] for name in images]
            assert ids.count(tokens["begin"]) == ids.count(tokens["end"]) == len(images)

    def test_each_audio_case_has_one_block_per_audio_and_image(self, fixture) -> None:
        document, tensors = fixture
        image_tokens, audio_tokens = document["imageTokens"], document["audioTokens"]
        images = {image["name"]: image["softTokens"] for image in document["images"]}
        audios = {audio["name"]: audio["softTokens"] for audio in document["audios"]}
        for case in document["audioCases"]:
            ids = tensors[f"{case['name']}.input_ids"].tolist()
            image_names = [part["image"] for part in case["parts"] if part["type"] == "image"]
            audio_names = [part["audio"] for part in case["parts"] if part["type"] == "audio"]

            assert ids[0] == BOS and ids[-1] == EOS, case["name"]
            assert case["tokens"] == len(ids)
            for key, names, counts, tokens in (
                ("imageBlocks", image_names, images, image_tokens),
                ("audioBlocks", audio_names, audios, audio_tokens),
            ):
                blocks = reference.soft_blocks(ids, tokens["begin"], tokens["soft"], tokens["end"])
                kind = key.removesuffix("Blocks")
                assert [
                    (block[kind], block["start"], block["softTokens"]) for block in case[key]
                ] == [
                    (name, start, count) for name, (start, count) in zip(names, blocks, strict=True)
                ]
                assert [count for _, count in blocks] == [counts[name] for name in names]
                assert ids.count(tokens["begin"]) == ids.count(tokens["end"]) == len(names)

    def test_each_vector_is_a_unit_vector_and_the_composition_stayed_within_the_bound(
        self, fixture
    ) -> None:
        document, tensors = fixture
        for case in (*document["cases"], *document["audioCases"]):
            vector = tensors[f"{case['name']}.embedding"].astype(np.float64)

            assert vector.shape == (768,)
            assert abs(np.linalg.norm(vector) - 1) < 1e-5
            assert case["composition"]["maxAbs"] <= reference.COMPOSITION_TOLERANCE
            assert case["composition"]["cosine"] > 0.9999999

    def test_the_audio_after_an_image_or_a_text_starts_after_them(self, fixture) -> None:
        document, _ = fixture
        starts = {case["name"]: case["audioBlocks"][0]["start"] for case in document["audioCases"]}

        assert starts["audio-only"] == 2
        assert starts["text-then-audio"] > 2
        assert starts["query-audio"] > 2
        # <bos>, <|image>, 256 soft tokens, <image|>, then <|audio>.
        assert starts["image-then-audio"] == 2 + 256 + 2

    def test_the_text_and_image_orders_differ(self, fixture) -> None:
        """Text then image and image then text are separate inputs upstream (the key order)."""
        document, _ = fixture
        starts = {case["name"]: case["imageBlocks"][0]["start"] for case in document["cases"]}

        assert starts["image-only"] == starts["image-then-text"] == 2
        assert starts["text-then-image"] > 2
        assert starts["query-text-image"] > starts["text-then-image"] - 1

    def test_the_id_cases_hold_no_vector_and_no_composition(self, fixture) -> None:
        document, tensors = fixture

        for case in document["idCases"]:
            assert f"{case['name']}.embedding" not in tensors
            assert "composition" not in case
            assert {part["type"] for part in case["parts"]} <= {"text", "image"}, case["name"]

    def test_the_id_cases_cover_the_boundaries_they_are_named_for(self, fixture) -> None:
        """Each boundary is in its case's parts, so a case edited into another input fails here."""
        document, _ = fixture
        parts = {case["name"]: case["parts"] for case in document["idCases"]}
        prompts = {case["name"]: case.get("prompt") for case in document["idCases"]}
        texts = {
            name: [p["text"] for p in found if p["type"] == "text"] for name, found in parts.items()
        }
        kinds = {name: [p["type"] for p in found] for name, found in parts.items()}

        assert kinds["trailing-space-before-image"] == ["text", "image"]
        assert texts["trailing-space-before-image"][0].endswith(" ")
        assert kinds["leading-space-after-image"] == ["image", "text"]
        assert texts["leading-space-after-image"][0].startswith(" ")
        assert kinds["empty-text-between-images"] == ["image", "text", "image"]
        assert texts["empty-text-between-images"] == [""]
        before, after = texts["newline-and-tab-at-boundaries"]
        assert before.endswith("\n") and after.startswith("\n")
        assert before.startswith("a\t") and after.endswith("\t")
        assert all(
            any("\u3040" <= char <= "\u30ff" for char in text)
            for text in texts["japanese-around-image"]
        )
        assert kinds["prompt-before-image-first"][0] == "image"
        assert prompts["prompt-before-image-first"] is not None
        assert kinds["prompt-image-text-image"] == ["image", "text", "image"]
        assert prompts["prompt-image-text-image"] is not None


class TestSoftBlocks:
    def test_it_finds_each_run_of_soft_ids(self) -> None:
        ids = [2, 9, 7, 7, 8, 9, 7, 8, 1]

        assert reference.soft_blocks(ids, begin=9, soft=7, end=8) == [(2, 2), (6, 1)]

    @pytest.mark.parametrize("ids", [[2, 7, 7, 8, 1], [2, 9, 7, 7, 1], [2, 9, 7, 7]])
    def test_a_run_outside_begin_and_end_is_refused(self, ids: list[int]) -> None:
        with pytest.raises(reference.MultimodalReferenceError, match="not between"):
            reference.soft_blocks(ids, begin=9, soft=7, end=8)


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

    def test_an_audio_becomes_its_samples_with_their_rate(self) -> None:
        pytest.importorskip("PIL")
        case = reference.Case("x", (("text", "t"), ("audio", "edge")), "why")

        built = reference.upstream_input(case)

        assert list(built) == ["text", "audio"]
        assert built["audio"]["sampling_rate"] == 16_000
        assert np.array_equal(built["audio"]["array"], WAVEFORMS["edge"].f32())

    def test_the_message_form_is_one_user_message_of_the_parts_in_order(self) -> None:
        pytest.importorskip("PIL")
        case = reference.Case(
            "x", (("image", "noise-tall"), ("text", ""), ("image", "stripes-edge")), "why"
        )

        (message,) = reference.upstream_message(case)

        assert message["role"] == "user"
        assert [item["type"] for item in message["content"]] == ["image", "text", "image"]
        assert message["content"][1]["text"] == ""
        assert [message["content"][index]["image"].size for index in (0, 2)] == [
            (48, 112),
            (1190, 17),
        ]

    def test_a_repeated_text_part_is_refused(self) -> None:
        case = reference.Case("x", (("text", "a"), ("image", "noise-tall"), ("text", "b")), "why")

        with pytest.raises(reference.MultimodalReferenceError, match="no dict form"):
            reference.upstream_input(case)
