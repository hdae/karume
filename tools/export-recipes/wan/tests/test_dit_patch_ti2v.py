"""`wan/dit_patch.py` の Wan2.2 TI2V の形（ADR 0121 決定 3 / 4・段 1）の約束事。

固定するのは、壊れても例外が出ず**数だけが静かにずれる**側:

- I2V 対応のグラフ（`WanDitTokensTi2v` — 時刻入力 2 本・条件マスク・`where`）の T2V の使い方が、
  上流の 1 次元 timestep の forward と**ビット一致**すること（2.1 と同じ形の計算 — 決定 3 の
  「なぜ」）
- I2V の使い方が、決定 4 の参照ラッパ（時刻の MLP を値ごとに M = 1 で回し、トークンごとの表を上流の
  ブロックへ渡す）と**ビット一致**すること（故障注入: 時刻入力 2 本の取り違え・マスクの 1 トークンの
  ずれが割れる）
- 参照ラッパは、全トークンが同じ時刻（T2V）のとき上流の 1 次元 timestep の経路とビット一致すること
  （f32 / f64・in-memory と層逐次の両方）
- 層逐次の書き手（`wan.dit_reference.LayerwiseDit`）の参照ラッパの経路が、モデル全体の参照と
  ビット一致すること（f64 は全体 `double()` + `float64_forward` の値）
- 条件マスクのホスト関数（先頭の潜在フレームのトークンが真）

合成モデル（`test_dit_patch.TINY_DIT` — 乱数初期化の小さな `WanTransformer3DModel`）だけで回す。
5B の実重みの eager 同値は `python -m wan.ti2v_export_dit write`（golden の書き出しの門 —
runbook）。
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from wan import dit_patch, dit_reference
from wan.tests.test_dit_patch import TINY_DIT, TINY_LATENT

#: 合成の patch（TINY_DIT の patch_size）。
PATCH = (1, 2, 2)


def _tiny(seed: int = 20261004) -> nn.Module:
    transformer_wan = pytest.importorskip("diffusers.models.transformers.transformer_wan")
    torch.manual_seed(seed)
    return transformer_wan.WanTransformer3DModel(**TINY_DIT).to(torch.float32).eval()


def _inputs(seed: int = 11) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    latents = torch.randn(1, TINY_DIT["in_channels"], *TINY_LATENT, generator=generator)
    timestep = torch.tensor([750], dtype=torch.int64)
    embeds = torch.randn(1, 5, TINY_DIT["text_dim"], generator=generator)
    return latents, timestep, embeds


def _graph_inputs(
    model: nn.Module,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    embeds: torch.Tensor,
    condition_timestep: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return (
        dit_patch.dit_patchify(latents, PATCH),
        dit_patch.dit_timesteps_proj(model, timestep),
        embeds,
        *dit_patch.dit_rope_tables(model.rope, TINY_LATENT),
        dit_patch.dit_timesteps_proj(model, condition_timestep),
        mask,
    )


def _trunk(model: nn.Module, latents: torch.Tensor, inputs: tuple[torch.Tensor, ...]):
    """patch 埋め込みを上流の conv3d の出力にした経路（グラフの本体だけを比べる — 2.1 と同じ）。"""
    wrapper = dit_patch.WanDitTokensTi2v(model)
    with torch.no_grad():
        hidden = model.patch_embedding(latents).flatten(2).transpose(1, 2)
        tokens = wrapper.forward_hidden(hidden, *inputs[1:])
    return dit_patch.dit_unpatchify(tokens, TINY_LATENT, PATCH)


def _i2v_mask() -> torch.Tensor:
    return dit_patch.dit_condition_mask(TINY_LATENT, PATCH, conditioned=True)


ZERO = torch.tensor([0], dtype=torch.int64)


class TestTheConditionMask:
    def test_i2v_marks_exactly_the_tokens_of_the_first_latent_frame(self) -> None:
        """トークン添字 `(f·H' + h)·W' + w` — 先頭の潜在フレームの 15 トークンが先頭に並ぶ。"""
        mask = _i2v_mask()

        assert mask.dtype == torch.bool
        assert list(mask.shape) == [1, 2 * 3 * 5, 1]
        assert mask[0, :15, 0].all()
        assert not mask[0, 15:, 0].any()

    def test_t2v_marks_nothing(self) -> None:
        mask = dit_patch.dit_condition_mask(TINY_LATENT, PATCH, conditioned=False)

        assert list(mask.shape) == [1, 30, 1]
        assert not mask.any()

    def test_a_temporal_patch_other_than_one_fails_loudly(self) -> None:
        with pytest.raises(NotImplementedError, match="時間方向の patch 2"):
            dit_patch.dit_condition_mask((4, 6, 10), (2, 2, 2), conditioned=True)


class TestTheTi2vGraphForm:
    def test_t2v_matches_the_upstream_one_dimensional_timestep_forward_bit_exactly(self) -> None:
        """マスクが全て偽・条件側の時刻 = 生成側 → 上流の `[B,6,C]` の経路とビット一致。"""
        model = _tiny()
        latents, timestep, embeds = _inputs()
        no_mask = dit_patch.dit_condition_mask(TINY_LATENT, PATCH, conditioned=False)
        inputs = _graph_inputs(model, latents, timestep, embeds, timestep, no_mask)
        with torch.no_grad():
            expected = dit_patch.reference_dit(model, latents, timestep, embeds)

        assert torch.equal(_trunk(model, latents, inputs), expected)

    def test_i2v_matches_the_reference_wrapper_bit_exactly(self) -> None:
        model = _tiny()
        latents, timestep, embeds = _inputs()
        mask = _i2v_mask()
        inputs = _graph_inputs(model, latents, timestep, embeds, ZERO, mask)
        condition = dit_patch.TimestepCondition(ZERO, mask[..., 0])
        with torch.no_grad():
            expected = dit_patch.reference_dit(model, latents, timestep, embeds, condition)
            one_dimensional = dit_patch.reference_dit(model, latents, timestep, embeds)

        assert torch.equal(_trunk(model, latents, inputs), expected)
        # 恒真の門でないこと: 条件フレームの t = 0 は 1 次元の経路（全トークン t = 750）と値が違う。
        assert not torch.equal(expected, one_dimensional)

    @pytest.mark.parametrize("fault", ["swapped-times", "shifted-mask"])
    def test_a_wiring_fault_breaks_the_i2v_identity(self, fault: str) -> None:
        """故障注入: 時刻入力 2 本の取り違え・条件マスクを 1 トークンずらす → 参照ラッパと割れる
        （段 2 の r 門の故障注入と同じ 2 件 — I2V の形でしか効かない）。"""
        model = _tiny()
        latents, timestep, embeds = _inputs()
        mask = _i2v_mask()
        condition = dit_patch.TimestepCondition(ZERO, mask[..., 0])
        if fault == "swapped-times":
            inputs = _graph_inputs(model, latents, ZERO, embeds, timestep, mask)
        else:
            inputs = _graph_inputs(model, latents, timestep, embeds, ZERO, mask.roll(1, dims=1))
        with torch.no_grad():
            expected = dit_patch.reference_dit(model, latents, timestep, embeds, condition)

        assert not torch.equal(_trunk(model, latents, inputs), expected)

    def test_the_two_wrappers_share_the_patch_embedding_and_parts(self) -> None:
        """TI2V のラッパは 2.1 のラッパと同じ部品を参照する（重みの丸めが両方と上流に届く）。"""
        model = _tiny()
        wrapper = dit_patch.WanDitTokensTi2v(model)

        assert wrapper.patch_embedding.weight.data_ptr() == model.patch_embedding.weight.data_ptr()
        assert wrapper.condition_embedder is model.condition_embedder
        assert wrapper.blocks is model.blocks


class TestTheReferenceWrapper:
    @pytest.mark.parametrize("variant", ["no-condition-token", "same-time"])
    def test_one_time_for_every_token_is_the_one_dimensional_path_in_f32(
        self, variant: str
    ) -> None:
        """全トークンが同じ時刻なら、参照ラッパ = 上流の 1 次元 timestep の経路（ビット一致）。

        マスクが全て偽の形と、マスクは真でも条件側の時刻が生成側と同じ形の両方。
        """
        model = _tiny()
        latents, timestep, embeds = _inputs()
        mask = _i2v_mask()[..., 0]
        condition = (
            dit_patch.TimestepCondition(timestep, torch.zeros_like(mask))
            if variant == "no-condition-token"
            else dit_patch.TimestepCondition(timestep.clone(), mask)
        )
        with torch.no_grad():
            expected = dit_patch.reference_dit(model, latents, timestep, embeds)
            got = dit_patch.reference_dit(model, latents, timestep, embeds, condition)

        assert torch.equal(got, expected)

    def test_one_time_for_every_token_is_the_one_dimensional_path_in_f64(self) -> None:
        model = copy.deepcopy(_tiny()).double()
        latents, timestep, embeds = _inputs()
        condition = dit_patch.TimestepCondition(timestep, torch.zeros_like(_i2v_mask()[..., 0]))
        with torch.no_grad(), dit_patch.float64_forward(model):
            expected = dit_patch.reference_dit(model, latents.double(), timestep, embeds.double())
            got = dit_patch.reference_dit(
                model, latents.double(), timestep, embeds.double(), condition
            )

        assert got.dtype == torch.float64
        assert torch.equal(got, expected)

    def test_the_original_condition_embedder_is_restored(self) -> None:
        model = _tiny()
        original = model.condition_embedder
        latents, timestep, embeds = _inputs()
        condition = dit_patch.TimestepCondition(ZERO, _i2v_mask()[..., 0])
        with torch.no_grad():
            dit_patch.reference_dit(model, latents, timestep, embeds, condition)

        assert model.condition_embedder is original

    def test_a_per_token_timestep_out_of_line_with_the_mask_fails_loudly(self) -> None:
        """上流が渡す平坦の timestep が `where(mask, 条件側, 生成側)` と違えば止まる（上流の平坦化の
        順が変わったら、トークンと変調の対応が黙ってずれる）。"""
        model = _tiny()
        mask = _i2v_mask()[..., 0]
        embedder = dit_patch._TokenwiseTimeEmbedder(
            model.condition_embedder, torch.tensor([750]), dit_patch.TimestepCondition(ZERO, mask)
        )
        wrong = torch.full((mask.shape[1],), 750, dtype=torch.int64)

        with pytest.raises(AssertionError, match="条件マスクと一致しない"):
            embedder(wrong, _inputs()[2], None, mask.shape[1])

    def test_the_condition_timestep_must_be_one_value(self) -> None:
        model = _tiny()
        latents, timestep, embeds = _inputs()
        mask = _i2v_mask()[..., 0]

        with pytest.raises(ValueError, match="生成側・条件側とも"):
            dit_patch.reference_dit(
                model,
                latents,
                timestep,
                embeds,
                dit_patch.TimestepCondition(torch.zeros(1, mask.shape[1], dtype=torch.int64), mask),
            )

    def test_the_diffusers_per_token_path_runs_the_time_mlp_on_every_token(self) -> None:
        """観測の相手（diffusers の 2 次元 timestep — M = S の linear）
        は参照ラッパと別の経路を通る。

        時刻の MLP の linear の入力の行数で見る: 参照ラッパは 1 行ずつ 2 回、2 次元の経路は S 行。
        """
        model = _tiny()
        latents, timestep, embeds = _inputs()
        mask = _i2v_mask()[..., 0]
        rows: list[int] = []
        handle = model.condition_embedder.time_embedder.linear_1.register_forward_hook(
            lambda _m, args, _o: rows.append(args[0].numel() // int(args[0].shape[-1]))
        )
        try:
            with torch.no_grad():
                dit_patch.reference_dit(
                    model, latents, timestep, embeds, dit_patch.TimestepCondition(ZERO, mask)
                )
                dit_patch.reference_dit(model, latents, torch.where(mask, ZERO, timestep), embeds)
        finally:
            handle.remove()

        assert rows == [1, 1, mask.shape[1]]


class TestTheLayerwiseTokenwisePath:
    """層逐次の参照ラッパの経路 = モデル全体の参照（5B の f64 / f32 の golden の作り方）。"""

    def test_f32_and_f64_are_bit_identical_to_the_whole_model(self) -> None:
        model = _tiny()
        latents, timestep, embeds = _inputs()
        condition = dit_patch.TimestepCondition(ZERO, _i2v_mask()[..., 0])
        double = copy.deepcopy(model).double()
        with torch.no_grad():
            expected32 = dit_patch.reference_dit(model, latents, timestep, embeds, condition)
            with dit_patch.float64_forward(double):
                expected64 = dit_patch.reference_dit(
                    double, latents.double(), timestep, embeds.double(), condition
                )
        writer = dit_reference.LayerwiseDit(model.config)
        source = dit_reference.ModuleWeights(model)

        got32 = writer.forward(
            source, torch.float32, latents, timestep, embeds, condition=condition
        )
        got64 = writer.forward(
            source, torch.float64, latents, timestep, embeds, condition=condition
        )

        assert torch.equal(got32.output, expected32)
        assert torch.equal(got64.output, expected64)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["f32", "f64"])
    def test_one_time_for_every_token_is_the_one_dimensional_layerwise_path(
        self, dtype: torch.dtype
    ) -> None:
        """T2V（全トークンが同じ t）で参照ラッパの経路 = 1 次元 timestep の経路（段 1 の検収）。"""
        model = _tiny()
        latents, timestep, embeds = _inputs()
        condition = dit_patch.TimestepCondition(timestep, torch.zeros_like(_i2v_mask()[..., 0]))
        writer = dit_reference.LayerwiseDit(model.config)
        source = dit_reference.ModuleWeights(model)

        expected = writer.forward(source, dtype, latents, timestep, embeds)
        got = writer.forward(source, dtype, latents, timestep, embeds, condition=condition)

        assert torch.equal(got.output, expected.output)

    def test_the_patched_graph_runs_on_the_layerwise_weights(self) -> None:
        """パッチ後のグラフの eager を層逐次の重みで回す口（`run` + `install_processors=False`）=
        in-memory のモデルで回したパッチ後のグラフ（5B の eager 同値の形）。"""
        model = _tiny()
        latents, timestep, embeds = _inputs()
        mask = _i2v_mask()
        inputs = _graph_inputs(model, latents, timestep, embeds, ZERO, mask)
        with torch.no_grad():
            expected = dit_patch.WanDitTokensTi2v(copy.deepcopy(model))(*inputs)
        writer = dit_reference.LayerwiseDit(model.config)

        got, _, _ = writer.run(
            dit_reference.ModuleWeights(model),
            torch.float32,
            lambda outer: dit_patch.WanDitTokensTi2v(outer, install_processors=False)(*inputs),
            prepare_block=lambda block: dit_patch.install_real_pair_processors([block]),
        )

        assert torch.equal(got, expected)

    @pytest.mark.parametrize("installed", [True, False], ids=["prepared", "not-prepared"])
    def test_the_layerwise_patched_graph_runs_the_real_pair_rope(
        self, installed: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """層逐次のパッチ後の eager はグラフと同じ実数形の RoPE を通る（ブロックごとに q / k の
        2 回）。

        値では見分けられない: 上流の processor も `RealPairRope`（中身は上流と同じ形の表）を
        tuple として受け、ビット一致する値を出す。だから `prepare_block` を落とすと、グラフの経路を
        回したつもりで上流の経路を回すことになる — 呼び出しの回数で縛る。
        """
        model = _tiny()
        latents, timestep, embeds = _inputs()
        inputs = _graph_inputs(model, latents, timestep, embeds, ZERO, _i2v_mask())
        writer = dit_reference.LayerwiseDit(model.config)
        calls: list[int] = []
        real = dit_patch.real_pair_rotary

        def counting(*args: torch.Tensor) -> torch.Tensor:
            calls.append(1)
            return real(*args)

        monkeypatch.setattr(dit_patch, "real_pair_rotary", counting)
        writer.run(
            dit_reference.ModuleWeights(model),
            torch.float32,
            lambda outer: dit_patch.WanDitTokensTi2v(outer, install_processors=False)(*inputs),
            prepare_block=(
                (lambda block: dit_patch.install_real_pair_processors([block]))
                if installed
                else None
            ),
        )

        assert len(calls) == (2 * writer.layers if installed else 0)

    @pytest.mark.parametrize("hook", ["collect_blocks", "on_block", "on_output"])
    def test_block_hooks_that_cannot_reach_the_patched_graph_fail_loudly(self, hook: str) -> None:
        """ブロックごとの口はブロックの forward hook で発火する — 部品を直に呼ぶ TI2V のグラフには
        届かないので、黙って空の記録を返さずに止まる。"""
        model = _tiny()
        latents, timestep, embeds = _inputs()
        inputs = _graph_inputs(model, latents, timestep, embeds, ZERO, _i2v_mask())
        writer = dit_reference.LayerwiseDit(model.config)
        seen: list[int] = []
        option: dict[str, object] = {
            "collect_blocks": {"collect_blocks": True},
            "on_block": {"on_block": lambda index, _seconds: seen.append(index)},
            "on_output": {"on_output": lambda index, _output: seen.append(index)},
        }[hook]

        with pytest.raises(dit_reference.DitReferenceError, match="forward が呼ばれていない"):
            writer.run(
                dit_reference.ModuleWeights(model),
                torch.float32,
                lambda outer: dit_patch.WanDitTokensTi2v(outer, install_processors=False)(*inputs),
                prepare_block=lambda block: dit_patch.install_real_pair_processors([block]),
                **option,  # type: ignore[arg-type]
            )

        assert seen == []

    def test_a_single_block_run_is_checked_at_the_end(self) -> None:
        """1 ブロックだけ回すと「次のブロックを求める時点」が来ない — 終わりの検査が止める。"""
        model = _tiny()
        latents, timestep, embeds = _inputs()
        inputs = _graph_inputs(model, latents, timestep, embeds, ZERO, _i2v_mask())
        writer = dit_reference.LayerwiseDit(model.config)

        with pytest.raises(dit_reference.DitReferenceError, match="ブロック 0 の forward"):
            writer.run(
                dit_reference.ModuleWeights(model),
                torch.float32,
                lambda outer: dit_patch.WanDitTokensTi2v(outer, install_processors=False)(*inputs),
                prepare_block=lambda block: dit_patch.install_real_pair_processors([block]),
                on_output=lambda _index, _output: None,
                blocks=1,
            )

    def test_the_upstream_reference_wrapper_still_feeds_every_block_hook(self) -> None:
        """上流のブロックの forward を通る参照ラッパの経路では、口が全ブロックで発火する。"""
        model = _tiny()
        latents, timestep, embeds = _inputs()
        condition = dit_patch.TimestepCondition(ZERO, _i2v_mask()[..., 0])
        writer = dit_reference.LayerwiseDit(model.config)
        timed: list[int] = []
        outputs: list[int] = []

        got = writer.forward(
            dit_reference.ModuleWeights(model),
            torch.float32,
            latents,
            timestep,
            embeds,
            condition=condition,
            collect_blocks=True,
            on_block=lambda index, _seconds: timed.append(index),
            on_output=lambda index, _output: outputs.append(index),
        )

        assert timed == outputs == list(range(writer.layers))
        assert len(got.blocks) == writer.layers
