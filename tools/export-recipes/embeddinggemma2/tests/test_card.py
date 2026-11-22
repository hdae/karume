"""Model card rendering of the EmbeddingGemma 2 distribution (`embeddinggemma2.card`).

The real `karume.json` is not used: the manifest here carries values **other** than the real ones
(another token bound, other dimensions, other prompts and placeholder ids), so a hand-written
number in the template shows as a mismatch. Besides the derivation from the manifest, it pins what
the card must say plainly (text tower only, the token limit against upstream's context, bf16 only,
the prompts, the Matryoshka dimensions, upstream's precision statement, the licence) and the
manifests it refuses to describe.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from embeddinggemma2.card import (
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
        assert "its weights shipped unchanged (bf16)" in card
        assert "| `bf16` (default) | **Original weights (bf16)**" in card

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
