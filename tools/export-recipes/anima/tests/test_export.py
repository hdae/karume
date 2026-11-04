"""Script-level rules of `anima/export.py` (the part that needs no real weights).

emit / verify, which need real weights, are manual (see the README). What is pinned here is only
the discipline that, when broken, gives a **false PASS**:

- the CLI mechanically refuses `--verify` and `--target` together in the same process
- the reference is not taken **after** the VAE patches are applied (the guard against a vacuous
  check)
- no gap opens between the target table and the builders / LoRA prefixes
"""

from __future__ import annotations

import argparse
import hashlib
import json

import pytest
import torch
from torch import nn

from anima import export as export_anima
from anima import patch as patch_anima
from anima.distribution import (
    ADALN_I8_CALIB_METHOD,
    ADALN_I8_TAG,
    ANIMA_BASE_MODEL_NAME,
    ANIMA_TURBO_MODEL_NAME,
    CALIB_PROVENANCE_FILE,
    CALIB_SHIPPABLE_DEVICE,
    CALIB_SHIPPABLE_METHOD,
    LORA_PROVENANCE_FILE,
)
from karume.quantize import DEFAULT_GROUP_SIZE, QuantizeError


class _DitLike(nn.Module):
    """A miniature of the DiT — holds only the shapes that affect the i4 eligibility check.

    The in axis 68 of `patch_embed` (= 17 channels × patch 2×2) is not divisible by g32. It has
    the same shape as the only unaligned linear of the real DiT (`CosmosPatchEmbed.proj`): it
    drops out of the i4 targets and goes to the i8 side. `attn` is aligned and eligible; `norm` is
    a type with no i4 execution path.
    """

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(68, 8)
        self.attn = nn.Linear(64, 8)
        self.norm = nn.LayerNorm(8)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.norm(self.attn(tokens))


class TestTargets:
    def test_every_target_has_a_builder(self):
        assert set(export_anima.BUILDERS) == set(export_anima.TARGETS)

    def test_lora_prefixes_only_name_real_targets(self):
        """The LoRA targets are only the DiT and the conditioner (recon §1) — a misspelling silently
        hits nothing."""
        assert set(export_anima.LORA_PREFIXES) <= set(export_anima.TARGETS)
        assert set(export_anima.LORA_PREFIXES) == {"transformer", "text_conditioner"}

    def test_emit_refuses_a_target_outside_the_table(self, tmp_path):
        """MUST: the target name becomes the container's graph name (= the weights key) as is.

        Accepting a spelling outside the table bakes "a container the runtime cannot look up".
        Without the check nothing fails until the `KeyError` of `BUILDERS[target]`, and it slips
        through the day `BUILDERS` grows an entry outside the table (the spelling check looks only
        at `TARGETS`).
        """
        with pytest.raises(AssertionError, match="unknown target"):
            export_anima.emit_target("vae_encoder", argparse.Namespace(), tmp_path / "out")

        assert not (tmp_path / "out").exists()


class TestLoraProvenance:
    """The attribution of a baked LoRA is kept in the series (the one fact that cannot be recovered
    from the fused weights)."""

    @staticmethod
    def _args(lora, **overrides):
        return argparse.Namespace(lora=lora, lora_scale=1.0, **overrides)

    def test_it_records_the_digest_of_the_file_that_was_fused(self, tmp_path):
        lora = tmp_path / "anima-turbo-lora-v0.2.safetensors"
        lora.write_bytes(b"lora-bytes")
        out_dir = tmp_path / "transformer"
        out_dir.mkdir()

        written = export_anima._write_lora_provenance(self._args(lora), "transformer", out_dir)

        assert written == LORA_PROVENANCE_FILE
        record = json.loads((out_dir / LORA_PROVENANCE_FILE).read_text(encoding="utf-8"))
        assert record == {
            "file": lora.name,
            "sha256": hashlib.sha256(b"lora-bytes").hexdigest(),
        }

    def test_it_writes_nothing_for_a_target_the_lora_never_touched(self, tmp_path):
        """A record in a series that was not baked would be the false claim "this model was baked
        too"."""
        lora = tmp_path / "lora.safetensors"
        lora.write_bytes(b"lora-bytes")
        out_dir = tmp_path / "vae_decoder"
        out_dir.mkdir()

        assert export_anima._write_lora_provenance(self._args(lora), "vae_decoder", out_dir) is None
        assert list(out_dir.iterdir()) == []

    def test_it_writes_nothing_without_the_flag(self, tmp_path):
        out_dir = tmp_path / "transformer"
        out_dir.mkdir()

        assert export_anima._write_lora_provenance(self._args(None), "transformer", out_dir) is None
        assert list(out_dir.iterdir()) == []

    def test_re_exporting_without_the_flag_removes_the_stale_record(self, tmp_path):
        """The record exists if and only if "the current weights were baked" — if the withdrawal
        slips through, the README prints a lie."""
        lora = tmp_path / "lora.safetensors"
        lora.write_bytes(b"lora-bytes")
        out_dir = tmp_path / "transformer"
        out_dir.mkdir()
        export_anima._write_lora_provenance(self._args(lora), "transformer", out_dir)
        assert (out_dir / LORA_PROVENANCE_FILE).exists()

        assert export_anima._write_lora_provenance(self._args(None), "transformer", out_dir) is None

        assert not (out_dir / LORA_PROVENANCE_FILE).exists()


class TestCalibProvenance:
    """The i4 rounding conditions are kept in the series (calibration changes not one byte of the
    storage form, so it cannot be read from the assets)."""

    @staticmethod
    def _args(
        dtype,
        *,
        no_calib=False,
        calib_prompts=4,
        model=ANIMA_TURBO_MODEL_NAME,
        adaln_i8=False,
        calib_device=CALIB_SHIPPABLE_DEVICE,
    ):
        return argparse.Namespace(
            dtype=dtype,
            no_calib=no_calib,
            calib_prompts=calib_prompts,
            model=model,
            i4_adaln_i8=adaln_i8,
            calib_device=calib_device,
        )

    def test_it_records_the_shippable_method_and_the_corpus(self, tmp_path):
        out_dir = tmp_path / "transformer"
        out_dir.mkdir()

        written = export_anima._write_calib_provenance(self._args("i4"), "transformer", out_dir)

        assert written == CALIB_PROVENANCE_FILE
        record = json.loads((out_dir / CALIB_PROVENANCE_FILE).read_text(encoding="utf-8"))
        assert record["method"] == CALIB_SHIPPABLE_METHOD
        assert (record["prompts"], record["resolution"], record["steps"]) == (4, 512, 8)
        assert record["guidance"] == 1.0

    def test_it_records_the_conditions_of_the_model_that_was_calibrated(self, tmp_path):
        """The record holds **that model's** conditions (step / CFG) — turbo's values must not end
        up in the base model's series.

        The conditions cannot be recovered from the assets, so if this went back to fixed turbo
        values, nobody could later tell "the base model was calibrated at 8 steps, CFG 1" from
        "only the record is stale".
        """
        out_dir = tmp_path / "transformer"
        out_dir.mkdir()

        export_anima._write_calib_provenance(
            self._args("i4", model=ANIMA_BASE_MODEL_NAME), "transformer", out_dir
        )

        record = json.loads((out_dir / CALIB_PROVENANCE_FILE).read_text(encoding="utf-8"))
        assert (record["steps"], record["guidance"]) == (20, 4.0)

    def test_no_calib_is_recorded_rather_than_left_absent(self, tmp_path):
        """Absence could also read as "an old export" — `rtn` is written so that it can be refused
        by name."""
        out_dir = tmp_path / "transformer"
        out_dir.mkdir()

        # With `--no-calib` the CLI does not take `--model` (there is nothing to take conditions
        # from) — exercise that the record can be written that way too, with `model=None`.
        export_anima._write_calib_provenance(
            self._args("i4", no_calib=True, model=None), "transformer", out_dir
        )

        record = json.loads((out_dir / CALIB_PROVENANCE_FILE).read_text(encoding="utf-8"))
        assert record["method"] == "rtn"
        assert record["method"] != CALIB_SHIPPABLE_METHOD, "it would pass as shippable"
        assert (record["prompts"], record["resolution"], record["steps"]) == (0, 0, 0)
        assert record["guidance"] == 0

    def test_the_sensitivity_variant_is_recorded_as_its_own_method(self, tmp_path):
        """MUST: the variant is recorded with **a spelling outside** the distribution's accepted
        set.

        Even with the same rounding method `gptq`, a different "which weights it was applied to"
        is a different rounding, and not one byte of the storage form tells them apart (the mix
        of i4 and i8 is the same shape as the default). A spelling the method-match check
        (`distribution.assert_calib_provenance`) can refuse by name is the only safeguard that
        keeps a variant baked for visual evaluation out of the shipping path.
        """
        out_dir = tmp_path / "transformer"
        out_dir.mkdir()

        export_anima._write_calib_provenance(
            self._args("i4", adaln_i8=True), "transformer", out_dir
        )

        record = json.loads((out_dir / CALIB_PROVENANCE_FILE).read_text(encoding="utf-8"))
        assert record["method"] == ADALN_I8_CALIB_METHOD
        assert record["method"] != CALIB_SHIPPABLE_METHOD, "it would pass as shippable"
        # The conditions (budget, step, CFG) are the same as the default — only the split
        # changed.
        assert (record["prompts"], record["resolution"], record["steps"]) == (4, 512, 8)

    def test_the_variant_and_no_calib_compose_in_the_method(self, tmp_path):
        """The two opt-outs are independent — neither hides the other (`rtn-adaln8` is not shippable
        either)."""
        out_dir = tmp_path / "transformer"
        out_dir.mkdir()

        export_anima._write_calib_provenance(
            self._args("i4", no_calib=True, model=None, adaln_i8=True), "transformer", out_dir
        )

        record = json.loads((out_dir / CALIB_PROVENANCE_FILE).read_text(encoding="utf-8"))
        assert record["method"] == f"rtn-{ADALN_I8_TAG}"
        assert record["method"] not in (CALIB_SHIPPABLE_METHOD, ADALN_I8_CALIB_METHOD)

    def test_the_shippable_device_is_recorded_without_touching_the_method(self, tmp_path):
        """The default (CPU) record keeps its spelling — adding the `device` field does not break
        existing series."""
        out_dir = tmp_path / "transformer"
        out_dir.mkdir()

        export_anima._write_calib_provenance(self._args("i4"), "transformer", out_dir)

        record = json.loads((out_dir / CALIB_PROVENANCE_FILE).read_text(encoding="utf-8"))
        assert record["device"] == CALIB_SHIPPABLE_DEVICE
        assert record["method"] == CALIB_SHIPPABLE_METHOD

    def test_another_calibration_device_is_recorded_as_its_own_method(self, tmp_path):
        """MUST: i4 baked on cuda is recorded with **a spelling outside** the distribution's
        accepted set.

        GPTQ's rounding solution changes with the device, yet not one byte of the storage form
        tells them apart (the same kind of hole as `--i4-adaln-i8`). The method-match check looks
        only at `method`, so adding just a `device` field is not enough — the spelling has to
        name it.
        """
        out_dir = tmp_path / "transformer"
        out_dir.mkdir()

        export_anima._write_calib_provenance(
            self._args("i4", calib_device="cuda"), "transformer", out_dir
        )

        record = json.loads((out_dir / CALIB_PROVENANCE_FILE).read_text(encoding="utf-8"))
        assert record["device"] == "cuda"
        assert record["method"] == f"{CALIB_SHIPPABLE_METHOD}-cuda"
        assert record["method"] != CALIB_SHIPPABLE_METHOD, "it would pass as shippable"

    def test_the_device_and_the_variant_compose_in_the_method(self, tmp_path):
        """The two deviations are independent — neither hides the other."""
        out_dir = tmp_path / "transformer"
        out_dir.mkdir()

        export_anima._write_calib_provenance(
            self._args("i4", adaln_i8=True, calib_device="cuda"), "transformer", out_dir
        )

        record = json.loads((out_dir / CALIB_PROVENANCE_FILE).read_text(encoding="utf-8"))
        assert record["method"] == f"{ADALN_I8_CALIB_METHOD}-cuda"
        assert record["method"] not in (CALIB_SHIPPABLE_METHOD, ADALN_I8_CALIB_METHOD)

    def test_it_writes_nothing_for_a_series_that_is_not_i4(self, tmp_path):
        out_dir = tmp_path / "transformer"
        out_dir.mkdir()

        assert (
            export_anima._write_calib_provenance(self._args("i8"), "transformer", out_dir) is None
        )
        assert list(out_dir.iterdir()) == []

    def test_re_exporting_at_another_dtype_removes_the_stale_record(self, tmp_path):
        """The record exists if and only if "the current weights were rounded with calibration".

        If it remained when the series is re-taken at another dtype, an untrue claim would
        survive.
        """
        out_dir = tmp_path / "transformer"
        out_dir.mkdir()
        export_anima._write_calib_provenance(self._args("i4"), "transformer", out_dir)
        assert (out_dir / CALIB_PROVENANCE_FILE).exists()

        assert (
            export_anima._write_calib_provenance(self._args("i8"), "transformer", out_dir) is None
        )

        assert not (out_dir / CALIB_PROVENANCE_FILE).exists()


class TestFakeQuant:
    """The rounding of `--dtype f16` (ADR 0006) — the default f32 behavior must not move by a single
    bit."""

    def test_f32_leaves_the_weights_untouched(self):
        model = nn.Linear(4, 3)
        before = model.weight.clone()

        export_anima._fake_quant(argparse.Namespace(dtype="f32"), model, "t")

        assert torch.equal(model.weight, before)

    def test_f16_rounds_the_weights_to_representable_values(self):
        model = nn.Linear(4, 3)

        export_anima._fake_quant(argparse.Namespace(dtype="f16"), model, "t")

        assert torch.equal(model.weight, model.weight.to(torch.float16).to(torch.float32))

    def test_bf16_rounds_the_weights_to_representable_values(self):
        """The rounding of the faithful preset (ADR 0131 decision 2 — the rounding that is a no-op
        when upstream is BF16)."""
        model = nn.Linear(4, 3)
        before = model.weight.clone()

        scales, overrides = export_anima._fake_quant(argparse.Namespace(dtype="bf16"), model, "t")

        assert (scales, overrides) == ({}, {})
        assert torch.equal(model.weight, before.to(torch.bfloat16).to(torch.float32))

    def test_i8_rounds_to_per_channel_representable_values(self):
        model = nn.Linear(4, 3)

        scales, _ = export_anima._fake_quant(argparse.Namespace(dtype="i8"), model, "t")

        scale = scales["weight"]
        assert list(scale.shape) == [3, 1]
        assert torch.equal(torch.round(model.weight / scale) * scale, model.weight)

    def test_the_default_output_root_is_a_separate_series_per_dtype(self):
        """MUST: a compressed series gets its own directory (ADR 0018 / 0019) — sharing one would
        lose the f32 safety net."""
        roots = export_anima.DEFAULT_OUT_ROOTS

        assert set(roots) == set(export_anima.WEIGHT_DTYPES) == {"f32", "f16", "bf16", "i8", "i4"}
        assert len(set(roots.values())) == len(roots)
        assert roots["bf16"].name == "anima-bf16"


class TestFakeQuantI4:
    """The mix of `--dtype i4` (eligible = i4 group32, the rest = i8 per-channel — ADR 0069).

    Both "eligible but not rounded" and "not eligible but i4 requested" get through loading and
    break by **only the size silently going back / failing at run time**, so the agreement between
    the rounded set and the storage spec is enforced here.
    """

    @staticmethod
    def _quantize(model: nn.Module):
        """The plain RTN path (`--no-calib`) — the calibrated wiring is covered by
        `test_calib.py`."""
        return export_anima._fake_quant(
            argparse.Namespace(dtype="i4", no_calib=True, i4_adaln_i8=False), model, "t"
        )

    def test_the_series_is_stored_as_i8_with_the_eligible_weights_overridden(self):
        """The i4 execution path is limited to eligible slots — the series defaults to i8, and only
        the eligible ones go to i4, one by one."""
        base = export_anima.BASE_WEIGHT_DTYPES

        assert set(base) == set(export_anima.WEIGHT_DTYPES)
        assert base["i4"] == "i8"
        assert [dtype for dtype, stored in base.items() if dtype != stored] == ["i4"]

    def test_it_rounds_an_aligned_linear_to_group_representable_values(self):
        model = _DitLike()

        scales, _ = self._quantize(model)

        scale = scales["attn.weight"]
        assert list(scale.shape) == [8, 64 // DEFAULT_GROUP_SIZE]
        spread = scale.repeat_interleave(DEFAULT_GROUP_SIZE, dim=-1)
        assert torch.equal(torch.round(model.attn.weight / spread) * spread, model.attn.weight)

    def test_the_unaligned_linear_falls_back_to_i8_per_channel(self):
        """A partial group cannot be stored (ADR 0069 decision 2) — it drops out of the targets and
        the i8 side rounds it."""
        model = _DitLike()

        scales, overrides = self._quantize(model)

        scale = scales["patch_embed.weight"]
        assert list(scale.shape) == [8, 1]
        assert torch.equal(
            torch.round(model.patch_embed.weight / scale) * scale, model.patch_embed.weight
        )
        assert "patch_embed.weight" not in overrides

    def test_every_quantizable_weight_lands_in_exactly_one_ledger(self):
        """A weight in neither predicate **silently stays f32** (the values are right; only the size
        goes back)."""
        model = _DitLike()

        scales, overrides = self._quantize(model)

        assert sorted(scales) == ["attn.weight", "patch_embed.weight"]
        assert overrides == {"attn.weight": "i4"}

    def test_the_exclusion_follows_the_axis_length_and_not_the_name(self):
        """The reason for the exclusion is **the length of the quantization axis**, not the name (no
        hard-coded names).

        If the resolution / patch_size / input channels move, divisibility moves too, so an
        exclusion by name becomes a lie the moment upstream changes — the same `patch_embed`
        becomes eligible once its axis is aligned.
        """
        assert export_anima._i4_module_names(_DitLike()) == frozenset({"attn"})

        aligned = _DitLike()
        aligned.patch_embed = nn.Linear(64, 8)

        assert export_anima._i4_module_names(aligned) == frozenset({"attn", "patch_embed"})

    def test_letting_an_unaligned_weight_into_the_i4_set_fails_loudly(self, monkeypatch):
        """If the eligibility check breaks, the rounding fails — confirms that "falls back to i8"
        above is not vacuous."""
        monkeypatch.setattr(
            export_anima, "_i4_module_names", lambda model: frozenset({"attn", "patch_embed"})
        )

        with pytest.raises(QuantizeError, match="is not divisible"):
            self._quantize(_DitLike())

    def test_a_name_the_rounding_never_reaches_is_reported(self, monkeypatch):
        """The check for a miss — if the rounded set and the storage spec drift apart, the i4 slot
        stays green with i8 weights mixed in."""
        monkeypatch.setattr(
            export_anima, "_i4_module_names", lambda model: frozenset({"attn", "norm"})
        )

        with pytest.raises(AssertionError, match="stored as i4 but"):
            self._quantize(_DitLike())


class TestDtypeTargets:
    """MUST: the i8 series is the transformer only (ADR 0019) — pinned in both the table and the
    CLI."""

    def test_every_dtype_declares_its_targets(self):
        assert set(export_anima.DTYPE_TARGETS) == set(export_anima.WEIGHT_DTYPES)
        for dtype, targets in export_anima.DTYPE_TARGETS.items():
            assert set(targets) <= set(export_anima.TARGETS), dtype

    def test_i8_covers_the_transformer_only(self):
        assert export_anima.DTYPE_TARGETS["i8"] == ("transformer",)

    def test_i4_covers_the_transformer_only(self):
        """The i4 targets are the same as i8's (the DiT dominates; text / cond / VAE share the f16
        series)."""
        assert export_anima.DTYPE_TARGETS["i4"] == ("transformer",)

    def test_bf16_covers_all_four_targets(self):
        """The faithful preset is bf16 for all 4 roles (ADR 0131 decision 3 — upstream is BF16 for
        all 4 components)."""
        assert export_anima.DTYPE_TARGETS["bf16"] == export_anima.TARGETS
        assert export_anima.BASE_WEIGHT_DTYPES["bf16"] == "bf16"

    @pytest.mark.parametrize("flag", ["--target", "--verify"])
    def test_an_i8_target_outside_the_table_is_refused(self, monkeypatch, flag):
        """Only narrowing the default would let an explicit request through (an asset that was
        supposed to be excluded would silently appear)."""
        monkeypatch.setattr("sys.argv", ["export_anima.py", "--dtype", "i8", flag, "vae_decoder"])

        with pytest.raises(SystemExit):
            export_anima.main()

    @pytest.mark.parametrize("flag", ["--target", "--verify"])
    def test_an_i4_target_outside_the_table_is_refused(self, monkeypatch, flag):
        monkeypatch.setattr("sys.argv", ["export_anima.py", "--dtype", "i4", flag, "vae_decoder"])

        with pytest.raises(SystemExit):
            export_anima.main()


class TestCliExclusion:
    def test_verify_and_target_cannot_share_a_process(self, monkeypatch):
        """MUST: after emit has applied the patches, "the unpatched reference" cannot be taken
        (ADR 0013 / 0016)."""
        monkeypatch.setattr(
            "sys.argv",
            ["export_anima.py", "--verify", "vae_decoder", "--target", "vae_decoder"],
        )

        with pytest.raises(SystemExit):
            export_anima.main()


class TestIneffectiveKnobs:
    """Knobs that have no effect (`--lora` / `--lora-scale` / `--num-layers`) are not silently
    accepted."""

    @staticmethod
    def _emitted(monkeypatch, argv: list[str]) -> list[str]:
        emitted: list[str] = []

        def stub(target, args, out_dir):
            emitted.append(target)
            return {"target": target}

        monkeypatch.setattr("sys.argv", ["export.py", *argv])
        monkeypatch.setattr(export_anima, "emit_target", stub)
        export_anima.main()
        return emitted

    @pytest.mark.parametrize(
        ("argv", "knob"),
        [
            (["--target", "text_encoder", "--lora", "x.safetensors"], "--lora"),
            (["--verify", "vae_decoder", "--lora", "x.safetensors"], "--lora"),
            (["--lora-scale", "2"], "--lora-scale"),
            (["--target", "vae_decoder", "--num-layers", "2"], "--num-layers"),
            (["--target", "text_conditioner", "--num-layers", "2"], "--num-layers"),
        ],
    )
    def test_a_knob_no_requested_target_uses_is_refused(self, monkeypatch, capsys, argv, knob):
        """A series that was meant to have the LoRA baked in must not appear without a record (the
        same as `--dit-graph dyn`)."""
        monkeypatch.setattr("sys.argv", ["export.py", *argv])

        with pytest.raises(SystemExit) as raised:
            export_anima.main()

        assert raised.value.code == 2
        assert knob in capsys.readouterr().err

    def test_lora_on_a_target_it_fuses_into_is_accepted(self, monkeypatch):
        emitted = self._emitted(monkeypatch, ["--target", "transformer", "--lora", "x"])

        assert emitted == ["transformer"]

    def test_the_default_targets_count_as_the_targets_that_run(self, monkeypatch):
        """Without `--target` the dtype's default set runs — the decision is by the set that runs,
        not by whether a target was given."""
        emitted = self._emitted(
            monkeypatch,
            ["--dtype", "i8", "--lora", "x", "--lora-scale", "0.5", "--num-layers", "2"],
        )

        assert emitted == ["transformer"]


class TestOrderGuard:
    def test_taking_a_reference_after_the_vae_patch_is_rejected(self, monkeypatch):
        """Guard against a vacuous check — a "reference" taken after the patches always differs by
        0, so green proves nothing."""
        monkeypatch.setattr(patch_anima, "vae_patches_applied", lambda: True)

        with pytest.raises(RuntimeError, match="pass vacuously"):
            export_anima._assert_vae_unpatched("test")

    def test_it_passes_before_the_patch(self, monkeypatch):
        monkeypatch.setattr(patch_anima, "vae_patches_applied", lambda: False)

        export_anima._assert_vae_unpatched("test")


class TestIoNaming:
    def test_graph_input_names_must_match_the_declared_order(self, tmp_path):
        """IR input names = forward's argument names. Writing by position while they disagree makes
        a golden point to another input."""

        class Two(nn.Module):
            def forward(self, first, second):
                return first + second

        component = export_anima.Component(
            module=Two(),
            dynamic_shapes=None,
            input_names=("first", "third"),
            cases=(("case0", (torch.zeros(2), torch.ones(2))),),
            reference=None,
        )
        graph = export_anima.export_to_file(
            Two(),
            component.cases[0][1],
            tmp_path / export_anima.MODEL_FILE,
            provenance=export_anima.PROVENANCE,
            graph_name="tiny",
        )

        with pytest.raises(AssertionError, match="graph inputs differ from the declaration"):
            export_anima._write_io(component, graph, tmp_path)


class TestDitGraph:
    """The CLI discipline of `--dit-graph dyn` (the additional S-shaped series — #21 iteration
    T2)."""

    def test_the_dyn_series_lands_in_its_own_directory(self, monkeypatch):
        """MUST: the additional series goes to **its own directory** (overwriting the static series
        would remove the premise of iteration T2)."""
        captured: dict[str, object] = {}

        def stub(target, args, out_dir):
            captured[target] = out_dir
            return {"target": target}

        monkeypatch.setattr("sys.argv", ["export_anima.py", "--dtype", "f16", "--dit-graph", "dyn"])
        monkeypatch.setattr(export_anima, "emit_target", stub)

        export_anima.main()

        assert set(captured) == {"transformer"}
        assert captured["transformer"].parent.name == (
            f"{export_anima.DEFAULT_OUT_ROOTS['f16'].name}{export_anima.DYN_SUFFIX}"
        )

    @pytest.mark.parametrize("flag", ["--target", "--verify"])
    def test_a_non_transformer_target_is_refused(self, monkeypatch, flag):
        """The S shape is for the transformer only (the other 3 targets do not depend on the
        resolution, so they stay shared)."""
        monkeypatch.setattr(
            "sys.argv", ["export_anima.py", "--dit-graph", "dyn", flag, "vae_decoder"]
        )

        with pytest.raises(SystemExit):
            export_anima.main()

    def test_resolution_is_refused_because_it_does_not_apply(self, monkeypatch):
        """A knob with no effect is not silently accepted — the S-shaped graph holds no resolution
        at all."""
        monkeypatch.setattr(
            "sys.argv", ["export_anima.py", "--dit-graph", "dyn", "--resolution", "1024"]
        )

        with pytest.raises(SystemExit):
            export_anima.main()

    def test_the_golden_cases_use_two_different_token_lengths(self):
        """What the 2-point evaluation holds: the two resolutions differ (two equal S values would
        not show a binding hole in the numbers)."""
        assert len(set(export_anima.DIT_DYN_RESOLUTIONS)) == len(export_anima.DIT_DYN_RESOLUTIONS)
        assert export_anima.DIT_DYN_RESOLUTIONS[0] == export_anima.RESOLUTION


class TestAdalnI8Cli:
    """The CLI-side rules of `--i4-adaln-i8` (the sensitivity-experiment variant): off by default,
    i4 only, its own series."""

    @staticmethod
    def _run(monkeypatch, argv: list[str]) -> argparse.Namespace:
        captured: dict[str, argparse.Namespace] = {}

        def stub(target, args, out_dir):
            captured[target] = args
            return {"target": target}

        monkeypatch.setattr("sys.argv", ["export.py", *argv])
        monkeypatch.setattr(export_anima, "emit_target", stub)
        export_anima.main()
        return captured["transformer"]

    def test_the_variant_is_off_by_default(self, monkeypatch):
        args = self._run(monkeypatch, ["--dtype", "i4", "--model", ANIMA_TURBO_MODEL_NAME])

        assert args.i4_adaln_i8 is False

    def test_the_variant_writes_to_its_own_series_directory(self, monkeypatch):
        """MUST: the visual-evaluation variant **does not overwrite** the i4 series baked under the
        shipping conditions (a separate location even with the default out)."""
        plain = self._run(
            monkeypatch, ["--dtype", "i4", "--dit-graph", "dyn", "--model", ANIMA_TURBO_MODEL_NAME]
        )
        variant = self._run(
            monkeypatch,
            [
                "--dtype",
                "i4",
                "--dit-graph",
                "dyn",
                "--i4-adaln-i8",
                "--model",
                ANIMA_TURBO_MODEL_NAME,
            ],
        )

        assert plain.out != variant.out
        assert variant.out.name == f"{plain.out.name.removesuffix('-dyn')}-{ADALN_I8_TAG}-dyn"

    @pytest.mark.parametrize("dtype", ["f32", "f16", "i8"])
    def test_the_variant_is_refused_outside_i4(self, monkeypatch, capsys, dtype):
        """A knob with no effect is not silently accepted (outside i4 there is no i4 / i8 split at
        all)."""
        monkeypatch.setattr("sys.argv", ["export.py", "--dtype", dtype, "--i4-adaln-i8"])

        with pytest.raises(SystemExit):
            export_anima.main()

        assert "--i4-adaln-i8" in capsys.readouterr().err


class TestCalibDeviceCli:
    """The CLI-side rules of `--calib-device` (the device calibration runs on): cpu by default,
    only when calibrating, checked to exist."""

    def test_the_default_is_the_shippable_device(self, monkeypatch):
        args = TestAdalnI8Cli._run(
            monkeypatch, ["--dtype", "i4", "--model", ANIMA_TURBO_MODEL_NAME]
        )

        assert args.calib_device == CALIB_SHIPPABLE_DEVICE

    def test_another_device_writes_to_its_own_series_directory(self, monkeypatch):
        """MUST: a cuda-made series **does not overwrite** the i4 series baked under the shipping
        conditions (a separate location even with the default out).

        The rounding solution changes with the device, yet the storage form does not differ by a
        byte, so sharing the same path would silently turn "the shipping-condition series" into
        a cuda-made one (provenance stops the distribution, but the asset is lost).
        """
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        plain = TestAdalnI8Cli._run(
            monkeypatch, ["--dtype", "i4", "--dit-graph", "dyn", "--model", ANIMA_TURBO_MODEL_NAME]
        )
        moved = TestAdalnI8Cli._run(
            monkeypatch,
            [
                "--dtype",
                "i4",
                "--dit-graph",
                "dyn",
                "--calib-device",
                "cuda",
                "--model",
                ANIMA_TURBO_MODEL_NAME,
            ],
        )

        assert plain.out != moved.out
        assert moved.out.name == f"{plain.out.name.removesuffix('-dyn')}-cuda-dyn"

    def test_a_device_torch_cannot_see_is_refused_before_any_work(self, monkeypatch, capsys):
        """MUST: an unusable CUDA fails **before calibration starts** (there is no branch that
        silently falls back to CPU).

        The shared venv's torch is the CPU build, so this path is exercised as is. In an
        environment with a real GPU, `is_available()` is true and the CLI goes through (that can
        only be checked on real hardware).
        """
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(
            "sys.argv",
            [
                "export.py",
                "--dtype",
                "i4",
                "--model",
                ANIMA_TURBO_MODEL_NAME,
                "--calib-device",
                "cuda",
            ],
        )

        with pytest.raises(SystemExit):
            export_anima.main()

        assert "CUDA" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "argv",
        [
            ["--dtype", "f16"],
            ["--dtype", "i4", "--no-calib"],
        ],
    )
    def test_it_is_refused_where_no_calibration_runs(self, monkeypatch, capsys, argv):
        """A knob with no effect is not silently accepted (a path without calibration has no use for
        the GPU at all)."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr("sys.argv", ["export.py", *argv, "--calib-device", "cuda"])

        with pytest.raises(SystemExit):
            export_anima.main()

        assert "--calib-device" in capsys.readouterr().err


class TestProvenance:
    """The container's `provenance.license` is an identifier (container-v1 §2.3) — HF's `other` is
    not baked in."""

    def test_the_container_names_the_license_identifier_not_other(self) -> None:
        """With `other`, the license cannot be identified from a krm taken elsewhere (W-RC3-6)."""
        from anima.card import ANIMA_METADATA

        assert ANIMA_METADATA.license == "other"
        assert export_anima.PROVENANCE.license == ANIMA_METADATA.license_name
        assert export_anima.PROVENANCE.license == "circlestone-labs-non-commercial-license"
