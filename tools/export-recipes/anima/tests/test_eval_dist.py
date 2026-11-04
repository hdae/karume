"""The rules of the build used only for visual evaluation (`anima/eval_dist.py`).

This guards one thing only — **not mixing up the distribution and the evaluation**. The i4
sensitivity-experiment variant has the same storage form, count, `verify_dist` and header checks
as the default (a mix of i4 and i8), so the only way to tell them apart from the assets is the
`method` of `calib_provenance.json`. Hence **both directions** are needed:

- the evaluation entry **accepts only the variant's record** (plugging in a series baked under the
  shipping conditions fails)
- the shipping entry **accepts only the default record** (plugging in the variant fails)

With only one of them, "which rounding method fills the i4 slot" can no longer be read from the
location.

The build engine itself (placement, sharing, manifest, verification) is covered by
`test_distribution.py`, so this file goes only as far as the plan (the side that writes not a
single byte).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from container_series import write_component
from ir_fixtures import ir_container

from anima import eval_dist
from anima.card import NVIDIA_NOTICE
from anima.distribution import (
    ADALN_I8_CALIB_METHOD,
    ANIMA_BASE_MODEL_NAME,
    ANIMA_TURBO_MODEL_NAME,
    CALIB_PROVENANCE_FILE,
    CALIB_SHIPPABLE_METHOD,
    OFFICIAL_PIPELINE,
    ROPE_BASE_ASSET,
    ROPE_BASE_ROLE,
    anima_model,
    anima_sources,
    assert_calib_provenance,
)
from karume.container import AssetInput
from karume.dist import DistError

#: The base model's visual-evaluation target (a model whose i4 slot is out of the distribution —
#: the ruling of 2026-08-24).
MODEL = ANIMA_BASE_MODEL_NAME


def _fake_safetensors(dtype: str, payload: bytes) -> bytes:
    """The smallest safetensors that passes the storage-dtype check (8-byte length + header JSON +
    data section)."""
    header: dict[str, Any] = {
        "w": {"dtype": dtype, "shape": [len(payload)], "data_offsets": [0, len(payload)]}
    }
    encoded = json.dumps(header).encode("utf-8")
    return len(encoded).to_bytes(8, "little") + encoded + payload


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _calib_record(method: str, model: str) -> bytes:
    """The record of calibration conditions that `anima/export.py` leaves in an i4 series (the same
    shape as the real one).

    step / CFG are **per model** (`anima.calib.calib_conditions` derives them from
    `pipeline_config`), so the stand-in takes them from the same single place — copying them
    would make it no longer independent of the check of "which model's conditions it was baked
    with".
    """
    defaults = anima_model(model).pipeline_config["defaults"]
    return json.dumps(
        {
            "method": method,
            "group_size": 32,
            "grid": "rtn",
            "prompts": 4,
            "resolution": 512,
            "steps": int(defaults["steps"]),
            "guidance": float(defaults["guidanceScale"]),
            "text_dtype": "f16",
        }
    ).encode("utf-8")


def _build_series(series_dir: Path, *, method: str = ADALN_I8_CALIB_METHOD) -> Path:
    """Build the series of one base model (the f16 / i8 / bf16 shipping series + **the variant's
    i4 series**) from fake assets.

    The weights slots are **valid containers** (the build opens its inputs and checks them against
    every rule of the declaration — `karume.dist.assert_weight_components_verified`). The rope
    base table is an **asset** of the transformer's container (ADR 0109 decision 4), so it goes
    into the same container.
    """
    sources = anima_sources(series_dir, MODEL)
    # One text stack per storage label the base model declares (f16 + the bf16 of the faithful
    # preset).
    for storage, base in sources.base.items():
        write_component(
            base / "text_encoder" / "model.krm",
            ir_container(mark=f"te-{storage}", named="text_encoder", storage=storage),
        )
        write_component(
            sources.text_conditioner[storage] / "text_conditioner" / "model.krm",
            ir_container(mark=f"tc-{storage}", named="text_conditioner", storage=storage),
        )
        write_component(
            base / "vae_decoder" / "model.krm",
            ir_container(mark=f"vae-{storage}", named="vae_decoder", storage=storage),
        )
    _write(sources.tokenizers / "qwen2-tokenizer.json", b'{"qwen2": true}')
    _write(sources.tokenizers / "t5-tokenizer.json", b'{"t5": true}')
    rope = AssetInput(ROPE_BASE_ROLE, 4, b"rope")
    variant = series_dir / eval_dist.EVAL_SERIES.format(model=MODEL)
    for storage, series in {**sources.transformer, "i4": variant}.items():
        write_component(
            series / "transformer" / "model.krm",
            ir_container(
                mark=f"dit-{storage}",
                # The fake series is **named by its component** too (graph name = weights key,
                # container-v1 §2.1).
                named="transformer",
                storage=storage,
                assets={ROPE_BASE_ASSET: rope},
            ),
        )
    _write(variant / "transformer" / CALIB_PROVENANCE_FILE, _calib_record(method, MODEL))
    return variant


class TestEvalPlan:
    def test_it_restores_the_i4_seat_without_touching_the_shipping_table(self, tmp_path):
        """MUST: the slot is restored by replacing the spec — not one line of `ANIMA_MODELS`
        moves."""
        _build_series(tmp_path / "series")

        plan = eval_dist.eval_plan(tmp_path / "series", MODEL)

        assert "i4" not in anima_model(MODEL).storages, "the distribution's accepted set moved"
        assert set(plan.weights["transformer"]) == {"f16", "i8", "bf16", "i4"}
        assert "f16+dit4" in plan.quants

    def test_the_i4_seat_reads_the_variant_series(self, tmp_path):
        """MUST: it does not read `*-i4-dyn` baked under the shipping conditions (no other rounding
        method gets into the same slot)."""
        variant = _build_series(tmp_path / "series")

        plan = eval_dist.eval_plan(tmp_path / "series", MODEL)

        assert plan.artifacts["transformer_i4"].source == variant / "transformer" / "model.krm"
        assert eval_dist.EVAL_SERIES.format(model=MODEL) != f"{MODEL}-i4-dyn"

    def test_a_series_calibrated_for_shipping_is_refused(self, tmp_path):
        """One direction of the mix-up: plugging a series baked with the default rounding into the
        visual-evaluation slot fails."""
        _build_series(tmp_path / "series", method=CALIB_SHIPPABLE_METHOD)

        with pytest.raises(
            DistError, match="not built with a rounding method allowed for distribution"
        ):
            eval_dist.eval_plan(tmp_path / "series", MODEL)

    def test_a_series_without_calibration_is_refused(self, tmp_path):
        """No visual ruling is taken from a `--no-calib` smoke output (its quality baseline is a
        different one)."""
        _build_series(tmp_path / "series", method="rtn-adaln8")

        with pytest.raises(DistError):
            eval_dist.eval_plan(tmp_path / "series", MODEL)

    def test_an_unknown_model_is_refused(self, tmp_path):
        """An unknown name fails with the choices listed (accepted set = ANIMA_MODELS)."""
        with pytest.raises(DistError, match="is not a visual-evaluation target"):
            eval_dist.eval_plan(tmp_path / "series", "anima-nonexistent")

    def test_a_model_with_a_shipped_i4_seat_is_refused(self, tmp_path, monkeypatch):
        """A model whose i4 slot is back in the distribution is built by the shipping path — this
        entry does not accept it.

        The ruling of 2026-09-01 removed the shipping i4 slot from every model (the slot of the
        old fused turbo is not carried over either), so no real model reaches this branch — a
        spec with an injected i4 slot guards the mechanism.
        """
        spec = replace(anima_model(ANIMA_TURBO_MODEL_NAME), storages=("f16", "i8", "i4"))
        monkeypatch.setattr(eval_dist, "anima_model", lambda name: spec)
        with pytest.raises(DistError, match="already has an i4 preset in the distribution"):
            eval_dist.eval_plan(tmp_path / "series", ANIMA_TURBO_MODEL_NAME)

    def test_the_default_output_never_lands_in_the_distribution_root(self):
        """MUST: `models/` is a place for the distribution layout only (the DECIDED of
        `_shared.paths`)."""
        out = eval_dist.default_out_dir(eval_dist.PIPELINE, [MODEL])

        # Default out = `<EVAL_ROOT>/<repo name>/<date>_eval/` (the dated slot of bench —
        # assets-layout).
        assert out.parent.parent == eval_dist.EVAL_ROOT
        assert "models" not in out.parts

    def test_more_than_one_model_needs_an_explicit_output(self):
        with pytest.raises(DistError, match="--out"):
            eval_dist.default_out_dir(eval_dist.PIPELINE, [MODEL, "anima-wai-v1.0"])


class TestEndToEnd:
    """Check one full round from the CLI (plan → materialize → card → `verify_dist`).

    Only this class goes as far as materializing, because of the cost asymmetry — baking the
    variant's i4 series takes ~3h (measured), so a flaw on the build side (the card renderer, the
    default output location, the manifest's shape) must not be discovered **only after baking**.
    """

    def test_the_cli_assembles_a_loadable_distribution(self, tmp_path):
        _build_series(tmp_path / "series")
        out_dir = tmp_path / "eval" / "karume-anima-v1.0-adaln8"

        eval_dist.main(["--series", str(tmp_path / "series"), "--out", str(out_dir)])

        manifest = json.loads((out_dir / "karume.json").read_text(encoding="utf-8"))
        assert manifest["defaultModel"] == MODEL
        quants = manifest["models"][MODEL]["quants"]
        assert "f16+dit4" in quants
        assert quants["f16+dit4"]["weights"]["transformer"] == "i4"
        # The visual-evaluation entry (`--source` of `examples/anima/main.ts`) needs just the
        # manifest.
        assert (out_dir / "README.md").is_file()

    def test_the_notice_states_the_i4_series_it_carries(self, tmp_path):
        """The modification notice matches the contents — the added i4 series of the restored slot
        is stated alongside the shipping int8 line."""
        _build_series(tmp_path / "series")
        out_dir = tmp_path / "eval" / "karume-anima-v1.0-adaln8"

        eval_dist.main(["--series", str(tmp_path / "series"), "--out", str(out_dir)])

        notice = " ".join((out_dir / "NOTICE.md").read_text(encoding="utf-8").split())
        assert "An int8-quantized series of the transformer was added" in notice
        assert "A bf16 series was added alongside the f16 one." in notice
        assert "int4-quantized series of the transformer was added" in notice
        assert "adaLN linears and the linears outside the blocks stay int8" in notice

    def test_it_ships_the_same_legal_texts_as_the_distribution(self, tmp_path):
        """The evaluation build carries the shipping `LICENSE.md` (CircleStone + NVIDIA Open
        Model License Agreement) and the NVIDIA part of the notice, like the distribution."""
        _build_series(tmp_path / "series")
        out_dir = tmp_path / "eval" / "karume-anima-v1.0-adaln8"

        eval_dist.main(["--series", str(tmp_path / "series"), "--out", str(out_dir)])

        shipped = (out_dir / "LICENSE.md").read_text(encoding="utf-8")
        assert shipped == OFFICIAL_PIPELINE.root_files["LICENSE.md"]
        assert NVIDIA_NOTICE in (out_dir / "NOTICE.md").read_text(encoding="utf-8")


class TestShippingRefusesTheVariant:
    """The other direction of the mix-up: the variant's record fails **the shipping** build.

    The ruling of 2026-09-01 removed the shipping i4 slot from every model, so no real spec
    reaches this check — a spec and series with an injected i4 slot guard the mechanism. Whichever
    model the slot returns to in the future, the same single implementation applies (the decision
    is in one place, `spec.calib_method`).
    """

    def _i4_seat(self, tmp_path):
        model = ANIMA_TURBO_MODEL_NAME
        spec = replace(anima_model(model), storages=(*anima_model(model).storages, "i4"))
        sources = anima_sources(tmp_path / "series", model)
        sources = replace(
            sources,
            transformer={
                **sources.transformer,
                "i4": tmp_path / "series" / f"{model}-i4-dyn",
            },
        )
        return model, sources, spec

    def test_a_variant_record_fails_the_shipping_gate(self, tmp_path):
        model, sources, spec = self._i4_seat(tmp_path)
        _write(
            sources.transformer["i4"] / "transformer" / CALIB_PROVENANCE_FILE,
            _calib_record(ADALN_I8_CALIB_METHOD, model),
        )

        with pytest.raises(
            DistError, match="not built with a rounding method allowed for distribution"
        ):
            assert_calib_provenance(sources, spec)

    def test_the_shipping_gate_still_accepts_the_default_record(self, tmp_path):
        model, sources, spec = self._i4_seat(tmp_path)
        _write(
            sources.transformer["i4"] / "transformer" / CALIB_PROVENANCE_FILE,
            _calib_record(CALIB_SHIPPABLE_METHOD, model),
        )

        assert_calib_provenance(sources, spec)
