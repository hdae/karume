"""`moss/text_embeds.py` — the asset's form and the token count of the empty negative (T₋ = 10).

The text encoder is not run here (bf16, about 14 GiB resident). The T₋ = 10 check renders the
pinned chat template and tokenizes with the pinned tokenizer (SKIP when not cached); an asset that
was generated is checked for its shapes and token counts (SKIP when absent). Generation:
`uv run --group moss --inexact python -m moss.text_embeds`.
"""

from __future__ import annotations

import json

import pytest
import torch

from _shared.paths import SERIES_ROOT
from moss import pipeline_ref, prompts, text_embeds

TEXT_EMBEDS = SERIES_ROOT / text_embeds.SERIES_NAME / text_embeds.ASSET_NAME

#: The chat template's rendering of an empty user turn (ADR NNNN, Context).
EMPTY_TURN = "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<think>\n"

FAKE_TOKENS = {"fridge-note": 5, "guide-whiteboard": 7, "guide-fridge-note": 4, "negative": 10}

#: The classes `model_index.json` names at the pinned revision.
CLASSES = {"tokenizer": "Qwen2Tokenizer", "text_encoder": "Qwen3_5Model"}


def _fake_asset() -> tuple[dict[str, torch.Tensor], dict]:
    generator = torch.Generator().manual_seed(7)
    embeds = {
        name: torch.randn(count, pipeline_ref.TEXT_DIM, generator=generator)
        for name, count in FAKE_TOKENS.items()
    }
    ids = {name: list(range(count)) for name, count in FAKE_TOKENS.items()}
    ids["negative"] = list(text_embeds.EMPTY_PROMPT_IDS)
    metadata = text_embeds.asset_metadata(
        prompts.FIXED_PROMPTS, ids, classes=CLASSES, run={"versions": {"torch": "test"}}
    )
    return embeds, metadata


class TestAssetForm:
    def test_it_round_trips_with_one_metadata_key(self, tmp_path):
        embeds, metadata = _fake_asset()
        path = tmp_path / text_embeds.ASSET_NAME

        text_embeds.write_asset(path, embeds, metadata)
        tensors, read = text_embeds.read_asset(path)

        assert read == metadata
        assert all(torch.equal(tensors[name], embeds[name]) for name in embeds)

    def test_the_same_inputs_give_the_same_bytes(self, tmp_path):
        embeds, metadata = _fake_asset()
        first, second = tmp_path / "a.safetensors", tmp_path / "b.safetensors"

        text_embeds.write_asset(first, embeds, metadata)
        text_embeds.write_asset(second, embeds, metadata)

        assert first.read_bytes() == second.read_bytes()

    def test_rows_that_disagree_with_the_token_count_are_refused(self, tmp_path):
        embeds, metadata = _fake_asset()
        embeds["fridge-note"] = embeds["fridge-note"][:-1]

        with pytest.raises(text_embeds.TextEmbedsError, match="rows"):
            text_embeds.write_asset(tmp_path / "x.safetensors", embeds, metadata)

    def test_a_non_f32_tensor_is_refused(self, tmp_path):
        embeds, metadata = _fake_asset()
        embeds["negative"] = embeds["negative"].to(torch.bfloat16)

        with pytest.raises(text_embeds.TextEmbedsError):
            text_embeds.write_asset(tmp_path / "x.safetensors", embeds, metadata)

    def test_the_class_names_are_the_ones_given(self):
        _, metadata = _fake_asset()

        assert metadata["text_encoder"]["class"] == CLASSES["text_encoder"]
        assert metadata["tokenizer"]["class"] == CLASSES["tokenizer"]

    def test_no_rows_are_refused(self, tmp_path):
        embeds, metadata = _fake_asset()
        embeds["fridge-note"] = torch.zeros(0, pipeline_ref.TEXT_DIM)
        entry = next(e for e in metadata["prompts"] if e["name"] == "fridge-note")
        entry["tokens"], entry["token_ids"] = 0, []

        with pytest.raises(text_embeds.TextEmbedsError, match="1 ≤ T < 4096"):
            text_embeds.write_asset(tmp_path / "x.safetensors", embeds, metadata)

    def test_t_4096_is_refused_as_possibly_truncated(self, tmp_path):
        """Upstream truncates at 4096 tokens: T = 4096 looks the same as a longer prompt cut
        short."""
        embeds, metadata = _fake_asset()
        rows = pipeline_ref.MAX_SEQUENCE_LENGTH
        embeds["fridge-note"] = torch.zeros(rows, pipeline_ref.TEXT_DIM)
        entry = next(e for e in metadata["prompts"] if e["name"] == "fridge-note")
        entry["tokens"], entry["token_ids"] = rows, list(range(rows))

        with pytest.raises(text_embeds.TextEmbedsError, match="1 ≤ T < 4096"):
            text_embeds.write_asset(tmp_path / "x.safetensors", embeds, metadata)


class TestTextClasses:
    def test_they_are_read_from_the_pinned_model_index(self, moss_snapshot):
        assert text_embeds.text_classes(moss_snapshot) == CLASSES

    def test_a_class_from_another_library_is_refused(self, tmp_path):
        index = {"tokenizer": ["transformers", "Qwen2Tokenizer"], "text_encoder": ["x", "Model"]}
        (tmp_path / "model_index.json").write_text(json.dumps(index), encoding="utf-8")

        with pytest.raises(text_embeds.TextEmbedsError, match="text_encoder"):
            text_embeds.text_classes(tmp_path)


class TestKernelPath:
    def test_it_names_the_cpu_capability_and_the_bf16_gemm_flags(self):
        path = text_embeds.kernel_path()

        assert path["cpu_capability"] == torch.backends.cpu.get_cpu_capability()
        assert set(path["bf16_gemm"]) == {
            "mkldnn_enabled",
            "mkldnn_bf16_supported",
            "avx512_bf16",
            "amx_tile",
        }
        assert all(isinstance(flag, bool) for flag in path["bf16_gemm"].values())


class TestEmptyNegative:
    def test_other_ids_for_the_empty_prompt_are_refused(self):
        negative = prompts.prompt_by_name("negative")
        ids = [*text_embeds.EMPTY_PROMPT_IDS[:-1], 0]

        with pytest.raises(text_embeds.TextEmbedsError, match="empty prompt"):
            text_embeds.check_tokens(negative, torch.zeros(len(ids), pipeline_ref.TEXT_DIM), ids)

    def test_the_pinned_template_and_tokenizer_give_ten_tokens(self, moss_text_snapshot):
        """T₋ = 10: upstream's chat template on '' and the pinned tokenizer (ADR NNNN, Context)."""
        transformers = pytest.importorskip("transformers")
        tokenizer = transformers.Qwen2Tokenizer.from_pretrained(moss_text_snapshot / "tokenizer")

        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        ids = tokenizer(rendered).input_ids

        assert rendered == EMPTY_TURN
        assert tuple(ids) == text_embeds.EMPTY_PROMPT_IDS


class TestGeneratedAsset:
    @pytest.fixture
    def asset(self):
        if not TEXT_EMBEDS.is_file():
            pytest.skip(f"{TEXT_EMBEDS} has not been generated")
        return text_embeds.read_asset(TEXT_EMBEDS)

    def test_every_prompt_has_its_token_count_of_rows(self, asset):
        tensors, metadata = asset

        for entry in metadata["prompts"]:
            tensor = tensors[entry["name"]]
            assert tensor.shape == (entry["tokens"], pipeline_ref.TEXT_DIM)
            assert tensor.dtype == torch.float32
            assert entry["prompt"] == prompts.prompt_by_name(entry["name"]).text

    def test_it_agreed_across_thread_counts(self, asset):
        _, metadata = asset

        assert metadata["run"]["threads_agree"] is True
        assert metadata["run"]["thread_counts"][0] == 1 and metadata["run"]["thread_counts"][1] > 1
        assert "cpu_capability" in metadata["run"] and "bf16_gemm" in metadata["run"]

    def test_the_negative_is_the_ten_template_tokens(self, asset):
        tensors, metadata = asset
        negative = [e for e in metadata["prompts"] if e["role"] == prompts.NEGATIVE]

        assert [e["token_ids"] for e in negative] == [list(text_embeds.EMPTY_PROMPT_IDS)]
        assert tensors["negative"].shape[0] == 10
