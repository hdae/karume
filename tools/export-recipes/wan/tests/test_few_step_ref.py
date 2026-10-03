"""`wan/few_step_ref.py` の約束事 — ケースの独立・逆正規化・書いた参照の形。

参照そのもの（DiT の forward 4 回 + タイル decode・1 ケース約 25 分）は回さない。生成済みの
参照が系列の根にあれば形とメタを見る（無ければ SKIP）。生成は
`uv run --group wan --inexact python -m wan.few_step_ref`。
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from _shared.paths import SERIES_ROOT
from wan import export_vae, few_step_ref, pipeline_ref, prompts, text_embeds, vae_tiling

SERIES = SERIES_ROOT / export_vae.SERIES_NAME
EMBEDS = SERIES_ROOT / text_embeds.SERIES_NAME / text_embeds.ASSET_NAME


class TestCases:
    def test_two_band_cases_and_one_accept_case_differ_in_seed_and_prompt(self):
        """決定用 2 本 + 受入れ 1 本（決定 8 の形）。受入れは決定用と別の seed・プロンプト。"""
        cases = few_step_ref.FIXTURE_CASES
        roles = [case.role for case in cases]

        assert sorted(roles) == ["accept", "band", "band"]
        assert len({case.seed for case in cases}) == len(cases)
        assert len({case.prompt for case in cases}) == len(cases)
        assert len({case.name for case in cases}) == len(cases)

    def test_the_accept_seed_was_never_seen_by_the_band(self):
        """前の受入れの seed 20261031 は帯の決定に影響した可能性があるので使わない。"""
        assert 20261031 not in {case.seed for case in few_step_ref.FIXTURE_CASES}

    def test_every_case_uses_a_positive_fixed_prompt(self):
        for case in few_step_ref.FIXTURE_CASES:
            assert text_embeds.prompt_by_name(case.prompt).role == prompts.POSITIVE


class TestDenormalize:
    def test_it_inverts_the_upstream_latent_normalization(self):
        """平均と標準偏差の取り違え（`·std + mean` の逆や `mean`/`std` の入れ替え）を掴む。"""
        generator = torch.Generator().manual_seed(0)
        mean = torch.randn(16, generator=generator).tolist()
        std = (torch.rand(16, generator=generator) + 0.5).tolist()
        pipeline = SimpleNamespace(
            vae=SimpleNamespace(
                dtype=torch.float32,
                config=SimpleNamespace(z_dim=16, latents_mean=mean, latents_std=std),
            )
        )
        raw = torch.randn(1, 16, 2, 3, 4, generator=generator)
        view = (1, 16, 1, 1, 1)
        normalized = (raw - torch.tensor(mean).view(view)) / torch.tensor(std).view(view)

        restored = few_step_ref.denormalize_latents(pipeline, normalized)

        assert torch.allclose(restored, raw, atol=1e-5, rtol=0)


@pytest.fixture(scope="module", params=[case.name for case in few_step_ref.FIXTURE_CASES])
def written(request):
    from safetensors import safe_open

    path = SERIES / f"{few_step_ref.FIXTURE_PREFIX}{request.param}{few_step_ref.FIXTURE_SUFFIX}"
    if not path.is_file():
        pytest.skip(f"{path} が無い — `python -m wan.few_step_ref` で作る")
    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata()
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}  # noqa: SIM118
    return request.param, tensors, metadata


class TestWrittenReference:
    def test_the_tensors_have_the_documented_shapes(self, written):
        _, tensors, _ = written
        latent = few_step_ref.LATENT_SHAPE
        frames, height, width = 4 * (latent[1] - 1) + 1, 8 * latent[2], 8 * latent[3]

        expected = {"latents_init", "frames"} | {
            f"{kind}.{index}"
            for kind in ("noise_cond", "noise_uncond", "latents")
            for index in range(few_step_ref.STEPS)
        }
        assert set(tensors) == expected
        for name, tensor in tensors.items():
            assert tensor.dtype == torch.float32
            assert bool(tensor.isfinite().all()), name
            if name != "frames":
                assert tensor.shape == latent, name
        assert tensors["frames"].shape == (3, frames, height, width)

    def test_the_metadata_matches_the_schedule_and_the_tile_plan(self, written):
        name, _, metadata = written
        case = next(case for case in few_step_ref.FIXTURE_CASES if case.name == name)
        plan = vae_tiling.plan_tiles(
            few_step_ref.LATENT_SHAPE[2], few_step_ref.LATENT_SHAPE[3], export_vae.DEFAULT_TILE
        )

        assert json.loads(metadata["timesteps"]) == [999, 750]
        assert metadata["seed"] == str(case.seed)
        assert metadata["prompt"] == case.prompt
        assert metadata["flow_shift"] == "3.0"
        assert metadata["guidance_scale"] == "5.0"
        for key, value in plan.meta().items():
            assert metadata[key] == value, key

    def test_it_was_made_from_the_current_text_embeds(self, written):
        """埋め込みを作り直したら参照も作り直す（sha256 で縛る）。"""
        if not EMBEDS.is_file():
            pytest.skip(f"{EMBEDS} が無い")
        _, _, metadata = written

        assert metadata["text_embeds_sha256"] == hashlib.sha256(EMBEDS.read_bytes()).hexdigest()

    def test_the_noise_is_the_seeded_torch_randn(self, written):
        """初期ノイズは torch CPU の `randn`（seed 固定）— TS が fixture から読む値の出所。"""
        name, tensors, _ = written
        case = next(case for case in few_step_ref.FIXTURE_CASES if case.name == name)
        generator = torch.Generator().manual_seed(case.seed)

        expected = torch.randn(1, *few_step_ref.LATENT_SHAPE, generator=generator)[0]

        assert torch.equal(tensors["latents_init"], expected)


class TestRunCase:
    """`run_case` の記録の振り分けと attention の固定（DiT・decode・scheduler は差し替え — 重みは
    読まない）。

    差し替えた DiT は「受けた文脈の先頭の値」を出力に書くので、記録の行き先が値で分かる。
    """

    CASE = few_step_ref.FIXTURE_CASES[0]
    COND, UNCOND = 1.0, 2.0

    class _Dit(torch.nn.Module):
        def forward(self, *, hidden_states, timestep, encoder_hidden_states, return_dict):
            return (torch.full_like(hidden_states, float(encoder_hidden_states[0, 0, 0])),)

    @pytest.fixture
    def pipeline(self, monkeypatch):
        monkeypatch.setattr(few_step_ref, "LATENT_SHAPE", (16, 1, 2, 2))
        monkeypatch.setattr(few_step_ref, "denormalize_latents", lambda _pipeline, latents: latents)
        monkeypatch.setattr(few_step_ref.vae_tiling, "plan_tiles", lambda *_args: None)
        monkeypatch.setattr(
            few_step_ref.vae_tiling,
            "tiled_decode_unclamped",
            lambda _vae, _latents, _plan: torch.zeros(1, 3, 1, 16, 16),
        )
        return SimpleNamespace(
            transformer=self._Dit(),
            scheduler=SimpleNamespace(
                timesteps=torch.tensor([999, 750]), sigmas=torch.tensor([1.0, 0.5, 0.0])
            ),
            vae=None,
        )

    def _embeds(self) -> dict[str, torch.Tensor]:
        return {
            self.CASE.prompt: torch.full((3, pipeline_ref.TEXT_DIM), self.COND),
            "negative": torch.full((4, pipeline_ref.TEXT_DIM), self.UNCOND),
        }

    def _upstream(self, monkeypatch, order) -> None:
        """上流の denoise ループの代役（`order` が 1 ステップの中の文脈の呼び順を決める）。"""

        def run(pipeline, *, prompt_embeds, negative_prompt_embeds, latents, **kwargs):
            contexts = {
                "cond": pipeline_ref.pad_text_embeds(prompt_embeds),
                "uncond": pipeline_ref.pad_text_embeds(negative_prompt_embeds),
                "other": torch.zeros(1, pipeline_ref.MAX_SEQUENCE_LENGTH, pipeline_ref.TEXT_DIM),
            }
            for index, timestep in enumerate(pipeline.scheduler.timesteps):
                for label in order:
                    pipeline.transformer(
                        hidden_states=latents,
                        timestep=timestep.expand(1),
                        encoder_hidden_states=contexts[label],
                        return_dict=False,
                    )
                latents = latents + 1
                kwargs["callback_on_step_end"](pipeline, index, timestep, {"latents": latents})
            return latents

        monkeypatch.setattr(pipeline_ref, "run", run)

    @pytest.mark.parametrize(
        "order", [("cond", "uncond"), ("uncond", "cond")], ids=["upstream", "swapped"]
    )
    def test_the_outputs_are_labelled_by_their_text_context(self, pipeline, monkeypatch, order):
        """上流の呼び順（cond → uncond）が入れ替わっても、cond の出力は cond の欄へ入る。"""
        self._upstream(monkeypatch, order)

        tensors, _ = few_step_ref.run_case(pipeline, self.CASE, self._embeds(), tile=2)

        for index in range(few_step_ref.STEPS):
            assert bool((tensors[f"noise_cond.{index}"] == self.COND).all()), index
            assert bool((tensors[f"noise_uncond.{index}"] == self.UNCOND).all()), index

    def test_a_context_that_is_neither_embedding_is_refused(self, pipeline, monkeypatch):
        self._upstream(monkeypatch, ("cond", "other"))

        with pytest.raises(AssertionError, match="振り分けを決められない"):
            few_step_ref.run_case(pipeline, self.CASE, self._embeds(), tile=2)
