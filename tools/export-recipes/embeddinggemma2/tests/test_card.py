"""Model card rendering of the EmbeddingGemma 2 distribution (`embeddinggemma2.card`).

The real `karume.json` is not used: the manifest here carries values **other** than the real ones
(another token bound, other dimensions, other prompts and placeholder ids), so a hand-written
number in the template shows as a mismatch. Besides the derivation from the manifest, it pins what
the card must say plainly (text tower only — or, with the vision tower, text and images with its
budget and its one bf16 storage —, the token limit against upstream's context, what each storage
holds, the int8 form's measured distance with its source and only beside an `i8` preset,
the prompts, the Matryoshka dimensions, upstream's precision statement, the licence) and the
manifests it refuses to describe.
"""

from __future__ import annotations

import copy
import re
from typing import Any

import pytest

from _shared.paths import REPO_ROOT
from embeddinggemma2.card import (
    EMBEDDINGGEMMA2_I8_SCREENING,
    EMBEDDINGGEMMA2_I8_SCREENING_SOURCE,
    EMBEDDINGGEMMA2_SUPPORTED_PIPELINE,
    EMBEDDINGGEMMA2_UPSTREAM_CONTEXT_TOKENS,
    render_embeddinggemma2_model_card,
)
from karume.migrate import card_format_statement

#: The repository the usage snippet names (the dist passes it from the pipeline's declaration).
REPO = "hdae/fake-embeddings"

#: Prompts with a prefix holding a pipe (the real prefixes do), two names sharing one prefix.
PROMPTS = {
    "query": "task: search | query: ",
    "Retrieval-query": "task: search | query: ",
    "document": "title: none | text: ",
}


def _manifest() -> dict[str, Any]:
    """A minimal manifest of the contract with fake values (none of them the real ones)."""
    container = {
        "container": {
            "descriptor": {
                "graph": {"length": 40, "sha256": "0" * 64},
                "model": {"length": 24, "sha256": "1" * 64},
            },
            "parts": [
                {
                    "path": "text/model/model.bf16-00001-of-00002.krm",
                    "size": 11,
                    "sha256": "a" * 64,
                },
                {
                    "path": "text/model/model.bf16-00002-of-00002.krm",
                    "size": 13,
                    "sha256": "b" * 64,
                },
            ],
        }
    }
    return {
        "format": "karume/6",
        "generator": "karume/9.9.9",
        "defaultModel": "text",
        "models": {
            "text": {
                "pipeline": EMBEDDINGGEMMA2_SUPPORTED_PIPELINE,
                "weights": {"model": {"bf16": container}},
                "assets": {
                    "tokenizer": {
                        "path": "text/tokenizer/tokenizer.json",
                        "size": 7,
                        "sha256": "c" * 64,
                    }
                },
                "quants": {
                    "bf16": {
                        "weights": {"model": "bf16"},
                        "session": {},
                        "label": "Original weights (bf16)",
                        "description": "The checkpoint's own bf16 weights with f32 compute.",
                    }
                },
                "defaultQuant": "bf16",
                "pipelineConfig": {
                    "prompts": dict(PROMPTS),
                    "dimensions": [96, 48, 24],
                    "maxTokens": 1234,
                    "placeholderTokens": {"image": 71, "audio": 72, "video": 73},
                },
            }
        },
    }


def _with_i8(manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """The manifest with an `i8` preset beside `bf16` (its own container, the default unchanged)."""
    manifest = copy.deepcopy(manifest or _manifest())
    model = manifest["models"]["text"]
    container = copy.deepcopy(model["weights"]["model"]["bf16"])
    for index, part in enumerate(container["container"]["parts"]):
        part["path"] = part["path"].replace(".bf16-", ".i8-")
        part["size"] = 5 + index
    model["weights"]["model"]["i8"] = container
    model["quants"]["i8"] = {
        "weights": {"model": "i8"},
        "session": {},
        "label": "Half size (int8)",
        "description": "The token table and every linear layer stored as int8.",
    }
    return manifest


def _render(manifest: dict[str, Any] | None = None) -> str:
    return render_embeddinggemma2_model_card(manifest or _manifest(), REPO)


class TestWhatTheCardSays:
    def test_it_says_text_tower_only(self) -> None:
        card = _render()
        assert "the **text tower** of `google/embeddinggemma-2`" in card
        assert "cannot embed images, video or audio" in card

    def test_it_states_the_token_limit_from_the_manifest_against_upstreams_context(self) -> None:
        card = _render()
        assert "at most 1,234 tokens including `<bos>` and `<eos>` for now" in card
        assert f"{EMBEDDINGGEMMA2_UPSTREAM_CONTEXT_TOKENS:,} tokens" in card
        assert "refused, never" in card

    def test_a_limit_at_upstreams_context_is_stated_as_the_whole_context(self) -> None:
        manifest = _manifest()
        config = manifest["models"]["text"]["pipelineConfig"]
        config["maxTokens"] = EMBEDDINGGEMMA2_UPSTREAM_CONTEXT_TOKENS
        card = _render(manifest)

        assert (
            f"at most {EMBEDDINGGEMMA2_UPSTREAM_CONTEXT_TOKENS:,} tokens including `<bos>` and"
            " `<eos>` (the\n  upstream model's whole context)"
        ) in card
        assert "for now" not in card
        assert "refused, never truncated" in card

    def test_it_states_the_band_among_the_changes(self) -> None:
        assert "band declared on the attention" in _render()

    def test_it_states_the_bf16_storage(self) -> None:
        card = _render()
        assert "- `bf16`: the checkpoint's weights unchanged (bf16)." in card
        assert "| `bf16` (default) | **Original weights (bf16)**" in card
        assert "the original checkpoint files are not\ndistributed here" in card

    def test_a_bf16_only_manifest_states_no_int8_form(self) -> None:
        card = _render()
        assert "int8" not in card
        assert "quantized" not in card

    def test_it_states_what_each_storage_holds_beside_an_i8_preset(self) -> None:
        card = _render(_with_i8())
        assert "- `bf16`: the checkpoint's weights unchanged (bf16)." in card
        assert (
            "- `i8`: the token table and every linear layer rounded to int8, one f32 scale per row;"
        ) in card
        assert "| `bf16` (default) | **Original weights (bf16)**" in card
        assert "| `i8` | **Half size (int8)**" in card
        assert "unchanged in `bf16`, and an int8-quantized copy (`i8`) was\n  added." in card
        assert "computes in f32 over the stored bf16 or int8 weights" in card

    def test_it_states_the_int8_forms_measured_distance_with_its_source(self) -> None:
        card = _render(_with_i8())
        assert "### How far the int8 weights are from the original ones" in card
        assert "One measurement, not a benchmark" in card
        assert "0.999396 at the lowest and\n  0.999842 on average" in card
        assert "0.74% of the 3,800 (query, document, document) orders are flipped" in card
        assert (
            "Source: `docs/research/2026-11-22-embeddinggemma2-quant-screen.md` in the Karume"
            " repository."
        ) in card

    def test_the_measured_numbers_are_the_i8_all_rows_of_the_source(self) -> None:
        """Each number of the card's screening text is a cell of the `i8-all` rows of the research
        record it cites (a frozen snapshot), or a count that record states."""
        record = (REPO_ROOT / EMBEDDINGGEMMA2_I8_SCREENING_SOURCE).read_text(encoding="utf-8")
        rows: dict[int, list[str]] = {}
        sections = re.findall(r"^### Width (\d+)\n(.*?)(?=^### |\Z)", record, re.M | re.S)
        for width, section in sections:
            row = next(line for line in section.splitlines() if line.startswith("| i8-all "))
            rows[int(width)] = [cell.strip() for cell in row.strip("|").split("|")]
        assert sorted(rows) == [128, 256, 512, 768]
        # Columns: config, cos min, cos mean, …, inversion (the ninth cell).
        text = "\n".join(EMBEDDINGGEMMA2_I8_SCREENING)
        assert f"is {rows[768][1]} at the lowest and\n  {rows[768][2]} on average" in text
        assert f"and {rows[768][9]} of the 3,800 (query, document, document)" in text
        assert (
            f"(lowest cosine {rows[512][1]} / {rows[256][1]} / {rows[128][1]} at\n"
            "  512 / 256 / 128 values)"
        ) in text
        cells = {rows[width][index] for width in rows for index in (1, 2, 9)}
        assert set(re.findall(r"\d+\.\d+%?", text)) <= cells
        assert "**Texts (58)**: the 10 golden cases" in record
        assert "(T 12 to 8,192;" in record
        assert "the 48 texts of the first EmbeddingGemma's" in record
        assert "10 of the 48 in Japanese" in record
        assert "the 20 × 190 = 3,800 (query, document" in record

    def test_the_measured_distance_needs_an_i8_preset(self) -> None:
        assert "How far the int8 weights are" not in _render()

    def test_an_i8_preset_beside_the_default_keeps_the_relation_out(self) -> None:
        """The default download is still the checkpoint's values."""
        assert "base_model_relation" not in _render(_with_i8())

    def test_the_storage_list_does_not_run_into_the_attribution_list(self) -> None:
        """Two adjacent `-` lists would render as one list in CommonMark."""
        for manifest in (_manifest(), _with_i8()):
            lines = _render(manifest).splitlines()
            storages = [
                index
                for index, line in enumerate(lines)
                if line.startswith(("- `bf16`: ", "- `i8`: "))
            ]
            following = next(line for line in lines[storages[-1] + 1 :] if line.strip())
            assert not following.startswith("- "), following

    def test_it_lists_every_prompt_grouped_by_prefix_with_the_pipe_escaped(self) -> None:
        card = _render()
        assert "| `task: search \\| query: ` | `query`, `Retrieval-query` |" in card
        assert "| `title: none \\| text: ` | `document` |" in card

    def test_it_states_the_dimensions_from_the_manifest(self) -> None:
        card = _render()
        assert "a unit vector of 96 values" in card
        assert "a shorter vector of 48 / 24 values" in card
        assert "dimension: 24 }); // available: 96 / 48 / 24" in card

    def test_it_states_the_placeholder_ids_from_the_manifest(self) -> None:
        assert "(image 71, audio 72, video 73)" in _render()

    def test_it_carries_upstreams_precision_statement(self) -> None:
        card = _render()
        assert "bfloat16 or float32, never float16" in card
        assert "computes in f32 over the stored bf16 weights" in card

    def test_it_names_the_apache_license(self) -> None:
        card = _render()
        assert "license: apache-2.0" in card
        assert "licensed **Apache 2.0**" in card
        assert "a verbatim copy is in `LICENSE.md`" in card

    def test_it_advertises_the_familys_entry_point_and_the_repository(self) -> None:
        card = _render()
        assert 'import { Pipeline } from "jsr:@karume/models/embeddinggemma";' in card
        assert f'repo: "{REPO}"' in card
        assert 'pipeline.embed("how do I reset my password", { prompt: "query" })' in card

    def test_it_states_the_manifest_format_once(self) -> None:
        assert _render().count(card_format_statement("karume/6")) == 1

    def test_it_carries_no_quantized_relation(self) -> None:
        """The weights keep the checkpoint's storage — none of the Hub's relations applies."""
        assert "base_model_relation" not in _render()


class TestWhatTheCardRefuses:
    def test_another_contract_is_refused(self) -> None:
        manifest = _manifest()
        manifest["models"]["text"]["pipeline"] = "siglip2/1"
        with pytest.raises(ValueError):
            _render(manifest)

    def test_a_model_missing_from_the_attribution_table_is_refused(self) -> None:
        manifest = _manifest()
        manifest["models"]["image"] = copy.deepcopy(manifest["models"]["text"])
        with pytest.raises(ValueError, match="model 'image' has no upstream"):
            _render(manifest)

    def test_weights_in_another_storage_are_refused(self) -> None:
        manifest = _manifest()
        weights = manifest["models"]["text"]["weights"]["model"]
        weights["f16"] = weights.pop("bf16")
        with pytest.raises(ValueError, match=r"stored as \['f16'\]"):
            _render(manifest)

    def test_weights_in_an_undescribed_storage_beside_bf16_are_refused(self) -> None:
        manifest = _manifest()
        weights = manifest["models"]["text"]["weights"]["model"]
        weights["f16"] = copy.deepcopy(weights["bf16"])
        with pytest.raises(ValueError, match=r"stored as \['bf16', 'f16'\]"):
            _render(manifest)

    def test_weights_without_the_checkpoints_storage_are_refused(self) -> None:
        manifest = _with_i8()
        model = manifest["models"]["text"]
        del model["weights"]["model"]["bf16"]
        del model["quants"]["bf16"]
        model["defaultQuant"] = "i8"
        with pytest.raises(ValueError, match=r"stored as \['i8'\]"):
            _render(manifest)

    def test_an_int8_default_is_refused(self) -> None:
        """The card carries no quantized relation on the ground that the default is bf16."""
        manifest = _with_i8()
        manifest["models"]["text"]["defaultQuant"] = "i8"
        with pytest.raises(ValueError, match="the default quant 'i8' stores the text tower as i8"):
            _render(manifest)

    def test_a_limit_above_upstreams_context_is_refused(self) -> None:
        manifest = _manifest()
        manifest["models"]["text"]["pipelineConfig"]["maxTokens"] = (
            EMBEDDINGGEMMA2_UPSTREAM_CONTEXT_TOKENS + 1
        )
        with pytest.raises(ValueError, match="above the upstream model's context"):
            _render(manifest)

    def test_a_preset_with_a_session_is_refused(self) -> None:
        """The card says every preset computes in f32 — a session option could make it false."""
        manifest = _manifest()
        manifest["models"]["text"]["quants"]["bf16"]["session"] = {"numerics": {"level": "reorder"}}
        with pytest.raises(ValueError, match="declares a session"):
            _render(manifest)


def _with_vision(manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """The manifest with the vision tower: its one bf16 container, mapped by every preset, and
    `pipelineConfig.vision` (a budget and ids other than the real ones)."""
    manifest = copy.deepcopy(manifest or _manifest())
    model = manifest["models"]["text"]
    container = copy.deepcopy(model["weights"]["model"]["bf16"])
    for index, part in enumerate(container["container"]["parts"]):
        part["path"] = part["path"].replace("text/model/", "text/vision/")
        part["size"] = 17 + index
    model["weights"]["vision"] = {"bf16": container}
    for quant in model["quants"].values():
        quant["weights"]["vision"] = "bf16"
    model["pipelineConfig"]["vision"] = {
        "softTokenBudget": 70,
        "imageTokenId": 71,
        "beginImageTokenId": 74,
        "endImageTokenId": 75,
        "hiddenSize": 48,
    }
    return manifest


class TestTheVisionTower:
    def test_a_manifest_with_the_tower_says_text_and_images(self) -> None:
        card = _render(_with_vision())
        assert "# EmbeddingGemma 2 Text and Image Embedding — Karume" in card
        assert "the **text and vision towers** of\n`google/embeddinggemma-2`" in card
        assert "Built with `vision: true`, the pipeline also fetches the" in card
        assert "cannot embed images" not in card
        assert "neither video nor audio can be embedded" in card

    def test_it_states_the_budget_and_a_square_images_count_from_the_manifest(self) -> None:
        card = _render(_with_vision())
        assert "becomes up to\n  70 soft tokens (a square one 64)" in card
        assert "each image counts its soft tokens (at most 70) and its two markers" in card
        assert "at most 1,234 per input" in card

    def test_it_states_that_every_preset_ships_the_towers_bf16_values(self) -> None:
        for manifest in (_with_vision(), _with_vision(_with_i8())):
            card = _render(manifest)
            assert "The vision tower (`vision`) ships in every preset as the" in card
            assert "No rounded form of the tower was measured, so no preset rounds it." in card
            assert "built without `vision: true` does not fetch it" in card
            assert "its weights keep the checkpoint's bf16 values unchanged in every preset" in card

    def test_the_usage_shows_the_vision_option_and_an_image_part(self) -> None:
        card = _render(_with_vision())
        assert "  // vision: true, // also fetches the vision tower, for image parts" in card
        assert '// const picture = await pipeline.embed([{ type: "image", image }]);' in card
        assert "image part on a pipeline built without `vision: true` is refused" in card

    def test_a_text_only_manifest_shows_no_vision_option(self) -> None:
        card = _render()
        assert "vision: true" not in card
        assert "# EmbeddingGemma 2 Text Embedding — Karume" in card

    def test_the_tower_paragraph_keeps_the_two_lists_apart(self) -> None:
        lines = _render(_with_vision(_with_i8())).splitlines()
        last = max(index for index, line in enumerate(lines) if line.startswith("- `i8`: "))
        following = next(line for line in lines[last + 1 :] if line.strip())
        assert not following.startswith("- "), following

    def test_a_rounded_tower_is_refused(self) -> None:
        manifest = _with_vision(_with_i8())
        model = manifest["models"]["text"]
        model["weights"]["vision"]["i8"] = copy.deepcopy(model["weights"]["vision"]["bf16"])
        model["quants"]["i8"]["weights"]["vision"] = "i8"
        with pytest.raises(ValueError, match="every preset ships its bf16 values"):
            _render(manifest)

    def test_weights_without_the_declaration_and_the_other_way_round_are_refused(self) -> None:
        manifest = _with_vision()
        del manifest["models"]["text"]["pipelineConfig"]["vision"]
        with pytest.raises(ValueError, match=r"pipelineConfig\.vision is not declared"):
            _render(manifest)
        manifest = _with_vision()
        del manifest["models"]["text"]["weights"]["vision"]
        for quant in manifest["models"]["text"]["quants"].values():
            del quant["weights"]["vision"]
        with pytest.raises(ValueError, match="the vision weights are not shipped"):
            _render(manifest)
