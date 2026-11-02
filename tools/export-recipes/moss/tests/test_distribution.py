"""The MOSS-Image distribution recipe (`moss.distribution`, `moss.card` — ADR 0138 addendum of
2026-11-01).

What is pinned here:

- **The spellings are the writers'**: series, graph, asset and tokenizer names, and the text
  tower's source (the distribution repeats them because it must not import torch).
- **The presets satisfy ADR 0128 C1–C3** on their own (the distribution gate checks the mirror),
  and the default is the practical preset.
- **`pipelineConfig` is upstream's** request defaults, read from the pinned snapshot (skipped
  without it).
- **The gates refuse** a tokenizer of another revision, format or bound, a borrowing graph that
  holds weights of its own, and a container without the assets its reader needs (synthetic
  containers); the borrower gate also on the real containers of the distribution mirror
  `models/karume-moss-image/` (ADR 0139 — skipped without the mirror).
- **The plan over the real series** (an implementation-time check, opt-in `KARUME_SERIES=1` —
  ADR 0139 decision 6; skipped without `outputs/series/`): every role at its output path, the
  presets completed, the default.

Faults injected 2026-11-01 (each turned its case red): the tokenizer's revision check removed, the
borrower's byte check removed, the asset check reduced to a presence check, the default switched to
the reference preset, `guidance` 4.0 in `MOSS_PIPELINE_CONFIG`.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from container_series import write_component
from ir_fixtures import ir_container
from real_assets import SERIES_CHECKS, mirror_container, requires_series

from _shared.paths import SERIES_ROOT
from karume.container import CODEC_LEDGER, AssetInput
from karume.dist import (
    DistError,
    assert_quant_presentation,
    complete_quant_weights,
    component_parts,
)
from moss import distribution as dist
from moss import sources
from moss.card import render_moss_model_card


def _series_present() -> bool:
    """Whether the real series this distribution reads are on this machine (a container is its
    numbered parts — the representative path itself is not a file)."""
    found = dist.moss_sources(SERIES_ROOT)
    try:
        for container in found.containers.values():
            component_parts(container)
    except (DistError, OSError, ValueError):
        return False
    return found.tokenizer.is_file()


# MUST: `SERIES_CHECKS and …` — the default lane does not touch `outputs/series/` (ADR 0139
# decision 6).
_SERIES_PRESENT = SERIES_CHECKS and _series_present()


def _completed() -> Mapping[str, Any]:
    return complete_quant_weights(dist.MOSS_WEIGHTS, dist.MOSS_QUANTS)


class TestTheSpellingsAreTheWriters:
    def test_the_series_names(self) -> None:
        from moss import export_dit, export_vae
        from qwen3_5 import checkpoint, tokenizer
        from qwen3_5 import export as qwen_export

        assert dist.MOSS_DIT_SERIES == export_dit.SERIES_NAME
        assert dist.MOSS_VAE_SERIES == export_vae.SERIES_NAME
        assert qwen_export.series_dir(checkpoint.MOSS_IMAGE_SFT, "i8").name == dist.MOSS_TEXT_SERIES
        assert dist.MOSS_TOKENIZER_SERIES == tokenizer.SERIES_NAME
        assert dist.MOSS_TOKENIZER_FILE == tokenizer.ASSET_FILE
        assert dist.MOSS_MODEL_FILE == export_dit.MODEL_FILE == qwen_export.MODEL_FILE

    def test_the_graph_names(self) -> None:
        from moss import export_dit, vae_patch
        from qwen3_5 import export as qwen_export

        assert set(dist.MOSS_DIT_ROLES) == set(export_dit.GRAPHS)
        assert dist.MOSS_VAE_ROLES == vae_patch.GRAPHS
        assert dist.MOSS_TEXT_ENCODER_ROLE == qwen_export.GRAPH_NAME == qwen_export.COMPONENT_DIR
        assert dist.MOSS_LENDERS == {export_dit.DIT_SKIP22: export_dit.DIT}

    def test_the_container_assets(self) -> None:
        from moss import dit_patch, export_dit, export_vae

        assert dist.MOSS_CONTAINER_ASSETS == {
            export_dit.DIT: {
                export_dit.ROPE_BASE_ASSET: export_dit.ROPE_BASE_ROLE,
                export_dit.TRAINING_GRID_ASSET: export_dit.TRAINING_GRID_ROLE,
            },
            export_vae.VAE_DECODER_PRE: {export_vae.BATCH_NORM_ASSET: export_vae.BATCH_NORM_ROLE},
        }
        assert dist.MOSS_DIT_TEXT_INPUT in dit_patch.DIT_TEXT_INPUTS

    def test_the_tokenizer_asset(self) -> None:
        from qwen3_5 import tokenizer

        assert dist.MOSS_TOKENIZER_FORMAT == tokenizer.ASSET_FORMAT
        assert dist.MOSS_MAX_TEXT_TOKENS == tokenizer.MAX_LENGTH
        assert dist.MOSS_TOKENIZER_SUBFOLDER == tokenizer.SOURCE_SUBFOLDER
        assert (sources.SOURCE.repo, sources.SOURCE.revision) == (
            tokenizer.SOURCE_REPO,
            tokenizer.SOURCE_REVISION,
        )

    def test_the_text_tower_comes_from_the_pinned_checkpoint(self) -> None:
        """The provenance gate checks every container against `moss.sources.SOURCE`, the text
        tower's too — the text recipe pins the same checkpoint."""
        from qwen3_5 import checkpoint

        source = checkpoint.MOSS_IMAGE_SFT
        assert (source.repo, source.revision, source.license) == (
            sources.SOURCE.repo,
            sources.SOURCE.revision,
            sources.SOURCE.license,
        )


class TestThePresets:
    def test_the_default_is_the_practical_preset(self) -> None:
        assert dist.MOSS_DEFAULT_QUANT == "f32+dit8-a8-attn8-online"
        assert dist.MOSS_DEFAULT_QUANT in dist.MOSS_QUANTS

    def test_the_default_declares_no_gpu_feature(self) -> None:
        """ADR 0128 C1: the default runs on baseline WebGPU."""
        assert "gpuFeatures" not in dist.MOSS_QUANTS[dist.MOSS_DEFAULT_QUANT]

    def test_the_activation_quantized_preset_has_its_sibling(self) -> None:
        """ADR 0128 C2 (all three knobs — addendum 2026-10-29, amendment 5)."""
        knobs = ("linearCompute", "attentionCompute", "convCompute")
        completed = _completed()
        for name, quant in completed.items():
            if not any(quant["session"].get(knob) == "a8" for knob in knobs):
                continue
            siblings = [
                other
                for other, candidate in completed.items()
                if other != name
                and candidate["weights"] == quant["weights"]
                and not any(candidate["session"].get(knob) == "a8" for knob in knobs)
            ]
            assert siblings, name

    def test_every_preset_is_labelled_within_the_limits(self) -> None:
        """ADR 0128 C3 and ADR 0075's field limits."""
        for name, quant in dist.MOSS_QUANTS.items():
            assert quant["label"].strip() and quant["description"].strip(), name
            assert_quant_presentation(name, quant)

    def test_both_presets_take_every_component_on_one_label(self) -> None:
        for quant in _completed().values():
            assert quant["weights"] == dict(dist.MOSS_STORAGE_LABELS)

    def test_the_forbidden_storage_names_every_compressed_layout_but_the_required_one(
        self,
    ) -> None:
        compressed = {entry.layout for entry in CODEC_LEDGER.values()} - {"f32", "i32"}
        for role, required in dist.MOSS_STORAGE_REQUIREMENTS.items():
            assert set(dist.MOSS_STORAGE_FORBIDDEN[role]) == compressed - {required}, role


def _call_defaults(source: str) -> dict[str, Any]:
    """The keyword defaults of `MOSSImagePipeline.__call__` (read from the source, not imported)."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef) and node.name == "MOSSImagePipeline":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__call__":
                    arguments = item.args
                    return {
                        argument.arg: ast.literal_eval(default)
                        for argument, default in zip(
                            arguments.kwonlyargs, arguments.kw_defaults, strict=True
                        )
                        if default is not None
                    }
    raise AssertionError("MOSSImagePipeline.__call__ is not in the pinned pipeline code")


class TestThePipelineConfig:
    def test_the_defaults_are_upstreams(self, moss_snapshot: Path) -> None:
        defaults = _call_defaults(sources.verify_pipeline_code(moss_snapshot).decode("utf-8"))
        scheduler = json.loads(
            (moss_snapshot / "scheduler" / "scheduler_config.json").read_text(encoding="utf-8")
        )
        assert {
            "scheduler": {"shift": scheduler["shift"]},
            "defaults": {
                "width": defaults["width"],
                "height": defaults["height"],
                "steps": defaults["num_inference_steps"],
                "guidance": defaults["guidance_scale"],
                "negativePrompt": defaults["negative_prompt"],
            },
        } == dist.MOSS_PIPELINE_CONFIG


def _tokenizer_asset(**override: Any) -> dict[str, Any]:
    asset: dict[str, Any] = {
        "format": dist.MOSS_TOKENIZER_FORMAT,
        "source": {
            "repo": sources.SOURCE.repo,
            "revision": sources.SOURCE.revision,
            "subfolder": dist.MOSS_TOKENIZER_SUBFOLDER,
            "files": {},
        },
        "maxLength": dist.MOSS_MAX_TEXT_TOKENS,
    }
    asset.update(override)
    return asset


class TestTheTokenizerGate:
    def _write(self, tmp_path: Path, asset: Mapping[str, Any]) -> Path:
        path = tmp_path / "tokenizer.json"
        path.write_text(json.dumps(asset), encoding="utf-8")
        return path

    def test_it_accepts_the_writers_asset(self, tmp_path: Path) -> None:
        dist.assert_tokenizer(self._write(tmp_path, _tokenizer_asset()))

    def test_it_refuses_another_format(self, tmp_path: Path) -> None:
        with pytest.raises(DistError, match="format"):
            dist.assert_tokenizer(self._write(tmp_path, _tokenizer_asset(format="karume-x/1")))

    def test_it_refuses_another_revision(self, tmp_path: Path) -> None:
        source = {**_tokenizer_asset()["source"], "revision": "0" * 40}
        with pytest.raises(DistError, match="not the pinned"):
            dist.assert_tokenizer(self._write(tmp_path, _tokenizer_asset(source=source)))

    @pytest.mark.parametrize("bound", [4095, 4097, True, "4096"])
    def test_it_refuses_another_bound(self, tmp_path: Path, bound: Any) -> None:
        with pytest.raises(DistError, match="maxLength"):
            dist.assert_tokenizer(self._write(tmp_path, _tokenizer_asset(maxLength=bound)))


class TestTheContainerGates:
    def test_a_borrower_with_weights_of_its_own_is_refused(self, tmp_path: Path) -> None:
        lender = tmp_path / "dit" / "model.krm"
        borrower = tmp_path / "dit_skip22" / "model.krm"
        write_component(lender, ir_container(named="dit"))
        write_component(borrower, ir_container(named="dit_skip22"))
        with pytest.raises(DistError, match="outside its descriptors"):
            dist.assert_borrows_every_weight(borrower, lender)

    def test_the_dit_without_its_host_tables_is_refused(self, tmp_path: Path) -> None:
        container = tmp_path / "dit" / "model.krm"
        write_component(container, ir_container(named="dit"))
        with pytest.raises(DistError, match="declares the assets"):
            dist.assert_container_assets("dit", container)

    def test_an_asset_under_another_role_is_refused(self, tmp_path: Path) -> None:
        payload = b"\0" * 8
        container = tmp_path / "vae_decoder_pre" / "model.krm"
        write_component(
            container,
            ir_container(
                named="vae_decoder_pre",
                assets={"latent_batch_norm": AssetInput("rope-base", len(payload), payload)},
            ),
        )
        with pytest.raises(DistError, match="declares the assets"):
            dist.assert_container_assets("vae_decoder_pre", container)

    def test_the_declared_assets_pass(self, tmp_path: Path) -> None:
        payload = b"\0" * 8
        container = tmp_path / "vae_decoder_pre" / "model.krm"
        write_component(
            container,
            ir_container(
                named="vae_decoder_pre",
                assets={
                    "latent_batch_norm": AssetInput("latent-batch-norm", len(payload), payload)
                },
            ),
        )
        dist.assert_container_assets("vae_decoder_pre", container)


class TestTheMirrorContainers:
    def test_a_graph_with_weights_of_its_own_is_not_a_borrower(self) -> None:
        """The borrower gate on real containers: the mirror's `dit_text` holds its own weights."""

        def container(role: str) -> Path:
            return mirror_container(
                dist.MOSS_REPO_NAME, dist.MOSS_MODEL, role, dist.MOSS_STORAGE_LABELS[role]
            )

        with pytest.raises(DistError, match="outside its descriptors"):
            dist.assert_borrows_every_weight(
                container(dist.MOSS_DIT_TEXT_ROLE), container(dist.MOSS_DIT_ROLE)
            )


@pytest.mark.skipif(not _SERIES_PRESENT, reason="the MOSS-Image series are not in outputs/series/")
@requires_series
class TestThePlanOverTheSeries:
    def test_it_plans_every_role_at_its_output_path(self) -> None:
        plan = dist.moss_dist_plan(SERIES_ROOT, dist.MOSS_MODEL)
        assert plan.pipeline == "moss-image/1"
        assert {role: artifact.rel_path for role, artifact in plan.artifacts.items()} == dict(
            dist.MOSS_OUTPUT_PATHS
        )
        assert plan.default_quant == dist.MOSS_DEFAULT_QUANT
        assert plan.quants == _completed()
        assert plan.pipeline_config == dist.MOSS_PIPELINE_CONFIG

    def test_it_refuses_another_model(self) -> None:
        with pytest.raises(DistError, match="no model 'base'"):
            dist.moss_dist_plan(SERIES_ROOT, "base")


class TestTheCard:
    def _manifest(self) -> dict[str, Any]:
        return {
            "format": "karume/5",
            "generator": "karume/0.15.0",
            "defaultModel": dist.MOSS_MODEL,
            "models": {
                dist.MOSS_MODEL: {
                    "pipeline": dist.MOSS_PIPELINE,
                    "weights": {
                        role: {
                            label: {
                                "container": {
                                    "descriptor": {
                                        "graph": {"length": 16, "sha256": "b" * 64},
                                        "model": {"length": 8, "sha256": "c" * 64},
                                    },
                                    "parts": [
                                        {
                                            "path": f"sft/{role}/model.{label}-{part}.krm",
                                            "size": size,
                                            "sha256": "a" * 64,
                                        }
                                        for part, size in (
                                            ("00001-of-00002", 64),
                                            ("00002-of-00002", 1 << 20),
                                        )
                                    ],
                                }
                            }
                        }
                        for role, label in dist.MOSS_STORAGE_LABELS.items()
                    },
                    "assets": {},
                    "quants": _completed(),
                    "defaultQuant": dist.MOSS_DEFAULT_QUANT,
                    "pipelineConfig": dist.MOSS_PIPELINE_CONFIG,
                }
            },
        }

    def test_the_usage_imports_the_subpath_and_names_the_defaults(self) -> None:
        card = render_moss_model_card(self._manifest(), "hdae/karume-moss-image")
        assert 'from "jsr:@karume/models/moss"' in card
        assert f'// quant: "{dist.MOSS_DEFAULT_QUANT}", // default' in card
        assert '// negativePrompt: "",' in card
        assert "negative prompt: the empty prompt" in card
        assert "`` " not in card

    def test_it_names_the_upstream_and_the_vae_origins(self) -> None:
        card = render_moss_model_card(self._manifest(), "hdae/karume-moss-image")
        assert f"`{sources.SOURCE.revision}`" in card
        for origin in sources.VAE_ORIGINS:
            assert f"`{origin.source.revision}`" in card

    def test_the_front_matter_lists_every_upstream_whose_weights_it_carries(self) -> None:
        """The Hub's model tree: the DiT, the text tower and the small decoder are redistributed;
        FLUX.2 [klein] 4B supplies only the latent statistics and is not a base model."""
        card = render_moss_model_card(self._manifest(), "hdae/karume-moss-image")
        front_matter = card.split("---")[1]
        sequence = front_matter.split("base_model:\n")[1].split("\nbase_model_relation:")[0]
        base_models = [line.removeprefix("  - ") for line in sequence.splitlines()]
        assert base_models == [
            sources.SOURCE.repo,
            sources.TEXT_ENCODER_ORIGIN.repo,
            "black-forest-labs/FLUX.2-small-decoder",
        ]
        assert "black-forest-labs/FLUX.2-klein-4B" not in front_matter

    def test_the_vae_bullet_passes_the_usage_policy_pointer_on_as_information(self) -> None:
        card = render_moss_model_card(self._manifest(), "hdae/karume-moss-image")
        vae_bullet = card.split("- **VAE decoder**")[1].split("\n- ")[0]
        assert "BFL Usage Policy" in vae_bullet
        assert "not a term of the Apache License, Version 2.0" in vae_bullet
        for origin in sources.VAE_ORIGINS:
            assert (
                f"https://huggingface.co/{origin.source.repo}/blob/{origin.source.revision}"
                "/README.md"
            ) in vae_bullet

    def test_only_black_forest_labs_is_named_as_not_endorsing(self) -> None:
        card = render_moss_model_card(self._manifest(), "hdae/karume-moss-image")
        for text in (card, dist.MOSS_NOTICE_MARKDOWN):
            parts = [bullet for block in text.split("\n\n") for bullet in block.split("\n- ")]
            (sentence,) = [part for part in parts if "endorsed" in part]
            assert "Black Forest Labs" in sentence
            assert "FLUX" in sentence
            assert "OpenMOSS" not in sentence
            assert "Qwen" not in sentence

    def test_the_notice_names_what_was_changed(self) -> None:
        notice = dist.MOSS_NOTICE_MARKDOWN
        for fact in ("int8 transformer", "f32 VAE decoder", "int8 text encoder", "FLUX.2"):
            assert fact in notice
