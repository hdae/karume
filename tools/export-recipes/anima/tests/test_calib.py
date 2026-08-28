"""校正付き i4（GPTQ）の結線の約束事（実重み不要分）。

実重みの校正は手動（README / `anima.export --dtype i4`）。ここで固定するのは、壊れると
**偽 PASS** になる側だけ:

- stage 逐次の block ループがラッパ 1 回の forward とビット一致すること（ずれた経路で
  丸めても数値は普通に出る）
- i4 適格が「block 内 = 校正」「block 外 = 素の RTN」へ**過不足なく排他に**割れること
- 感度実験変種（`--i4-adaln-i8`）が adaLN + block 外を i8 へ役割で割ること、そして
  **既定 OFF が集合も経路も値も従来と 1 ビットも変わらない**こと
- scale 台帳のキーがラッパの FQN 空間（= safetensors のテンソルキー）に居ること
- 校正が**実際に別の丸め**を産むこと（素通りしたら格納形が同じなので資産からは読めない）
- 校正入力とグラフで block の呼ばれ方が同じこと
- 校正入力を**配布条件と同じ状態の重み**から採ること（block 外は丸めた後・block 内は丸める前）
- `--no-calib` が opt-out として本当に校正を回さないこと

模型は tiny な骨格で、**実物と同じ FQN**（`model.transformer_blocks.<i>` /
`model.time_embed.t_embedder.linear_1` / `model.norm_out.linear_1` / `model.proj_out`）と
**同じ呼び出しの形**（block は hidden を位置引数で受けて hidden を返す・`patch_embed.proj` の
量子化軸は g32 非整除）を写す。名前まで写すのは、実物向けの定数
（{@link anima.export.NON_STAGE_I4_WEIGHTS}）を差し替えずに門を試すため。

模型は実物と同じく **block ループを 2 本**持つ（{@link TinyRawDit} = 素のモデルの forward /
{@link TinyWrapper} = export するラッパ）。1 本にまとめると、捕捉元とグラフが同じコードパスに
なって {@link anima.calib.assert_calib_batches_match_graph} が守っている継ぎ目そのものが
テストから消える（実物では捕捉は素のモデル・丸めるのはラッパ）。
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
from dataclasses import replace

import pytest
import torch
from torch import nn

from anima import calib, pipeline_ref
from anima import export as export_anima
from anima.distribution import (
    ANIMA_BASE_MODEL_NAME,
    ANIMA_MODELS,
    ANIMA_TURBO_MODEL_NAME,
    BASE_MODELS,
    CALIB_SHIPPABLE_DEVICE,
)
from karume.quantize import fake_quant_int4, fake_quant_int8

#: 量子化軸（= group 長）。i4 は端数 group を作らない（ADR 0069 決定 2）。
HIDDEN = 32
#: patchify 入口の入力チャネル（実 DiT と同じ 17ch × 2×2 = 68 — g32 非整除で i4 適格外）。
PATCH_IN = 68
#: トークン数（校正の hidden は `[1, TOKENS, HIDDEN]`）。
TOKENS = 4
#: 捕捉テストの latent 辺と、それを産む解像度（実物の 512px は 64² トークンで、ここでは
#: 8² で足りる — 見るのは分岐の数え方であって活性の分布ではない）。
CAPTURE_LATENT = 8
CAPTURE_RESOLUTION = CAPTURE_LATENT * pipeline_ref.SPATIAL_COMPRESSION


class TinyAdaLayerNormZero(nn.Module):
    """block の `norm1` / `norm2` / `norm3`（`CosmosAdaLayerNormZero`）の席。

    実物と同じく `linear_1` / `linear_2` の **2 本**を持ち、入力は hidden ではなく
    `embedded_timestep`（rank2）— 感度実験変種（`--i4-adaln-i8`）が i4 から外すのはこの綴りで、
    GPTQ が見る `H = Σ XᵀX` の標本も block あたり 1 本ぶんしか積まれない側。
    """

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(HIDDEN, elementwise_affine=False)
        self.linear_1 = nn.Linear(HIDDEN, HIDDEN)
        self.linear_2 = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, hidden_states: torch.Tensor, embedded_timestep: torch.Tensor):
        scale = self.linear_2(torch.nn.functional.silu(self.linear_1(embedded_timestep)))
        return self.norm(hidden_states) * (1 + scale.unsqueeze(1))


class TinyBlock(nn.Module):
    """`CosmosTransformerBlock` の**呼び出しの形**だけを写した block。

    位置引数 8 本（hidden + 付随 7 本）で呼ばれ hidden を返す — 実物と同じなので、stage は
    包まずにそのまま並べられる（`anima.calib.dit_stages`）。

    modulation（`norm1` / `norm2` / `norm3`）まで写すのは、実物向けのセグメント定数
    （{@link anima.calib.ADALN_SEGMENTS}）を差し替えずに変種の割り方を試すため — block 内の
    linear は adaLN 6 本 + それ以外 2 本になる。
    """

    def __init__(self) -> None:
        super().__init__()
        self.norm1 = TinyAdaLayerNormZero()
        self.attn = nn.Linear(HIDDEN, HIDDEN)
        self.norm2 = TinyAdaLayerNormZero()
        self.norm3 = TinyAdaLayerNormZero()
        self.ff = nn.Linear(HIDDEN, HIDDEN)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        temb: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        extra_pos_emb: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        controlnet_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended = torch.tanh(
            self.attn(self.norm1(hidden_states, embedded_timestep))
            + self.norm2(encoder_hidden_states, embedded_timestep).mean(dim=1, keepdim=True)
        )
        return hidden_states + self.ff(self.norm3(attended, embedded_timestep)) * temb.unsqueeze(1)


class TinyTimestepEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(HIDDEN, HIDDEN)
        self.linear_2 = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, timesteps_proj: torch.Tensor) -> torch.Tensor:
        return self.linear_2(torch.nn.functional.silu(self.linear_1(timesteps_proj)))


class TinyEmbedding(nn.Module):
    """`CosmosEmbedding` の席（`t_embedder` が temb・`norm` が embedded_timestep）。"""

    def __init__(self) -> None:
        super().__init__()
        self.t_embedder = TinyTimestepEmbedding()
        self.norm = nn.LayerNorm(HIDDEN, elementwise_affine=False)


class TinyAdaLayerNorm(nn.Module):
    """`CosmosAdaLayerNorm`（= `norm_out`）の席。"""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(HIDDEN, elementwise_affine=False)
        self.linear_1 = nn.Linear(HIDDEN, HIDDEN)
        self.linear_2 = nn.Linear(HIDDEN, 2 * HIDDEN)

    def forward(
        self, hidden_states: torch.Tensor, embedded_timestep: torch.Tensor, temb: torch.Tensor
    ) -> torch.Tensor:
        shift, scale = self.linear_2(self.linear_1(embedded_timestep)).chunk(2, dim=-1)
        return self.norm(hidden_states) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TinyPatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(PATCH_IN, HIDDEN)


class TinyRawDit(nn.Module):
    """`CosmosTransformer3DModel` の骨格と**素の forward**（校正入力の捕捉元）。

    実物の捕捉は参照 denoise が素のモデルを回して採る（`export._round_i4_calibrated` が
    `wrapper.model` を渡す）ので、block ループはラッパとは**別実装**で持つ — 上流がここの
    引数を増減させた日に、ラッパとの食い違いが門に掛かることをテストでも踏めるようにする。
    """

    def __init__(self, blocks: int) -> None:
        super().__init__()
        self.patch_embed = TinyPatchEmbed()
        self.time_embed = TinyEmbedding()
        self.transformer_blocks = nn.ModuleList(TinyBlock() for _ in range(blocks))
        self.norm_out = TinyAdaLayerNorm()
        self.proj_out = nn.Linear(HIDDEN, HIDDEN)

    def forward(
        self,
        tokens: torch.Tensor,
        timesteps_proj: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.patch_embed.proj(tokens)
        temb = self.time_embed.t_embedder(timesteps_proj)
        embedded_timestep = self.time_embed.norm(timesteps_proj)
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                embedded_timestep,
                temb,
                None,
                None,
                None,
                None,
            )
        hidden_states = self.norm_out(hidden_states, embedded_timestep, temb)
        return self.proj_out(hidden_states)


class DriftingRawDit(TinyRawDit):
    """上流が block の付随引数を 1 本減らした日の素のモデル（ラッパは 8 本のまま）。

    実物でこれが起きると、校正だけが `controlnet_residual` を見ない条件で丸め先を選び、
    数値は普通に出る（golden も緑のまま通る）。
    """

    def forward(
        self,
        tokens: torch.Tensor,
        timesteps_proj: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.patch_embed.proj(tokens)
        temb = self.time_embed.t_embedder(timesteps_proj)
        embedded_timestep = self.time_embed.norm(timesteps_proj)
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states, encoder_hidden_states, embedded_timestep, temb, None, None, None
            )
        hidden_states = self.norm_out(hidden_states, embedded_timestep, temb)
        return self.proj_out(hidden_states)


class TinyWrapper(nn.Module):
    """`patch.AnimaDit` の骨格（patchify → block ループ → norm_out → proj_out）。"""

    def __init__(self, model: TinyRawDit) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        tokens: torch.Tensor,
        timesteps_proj: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        model = self.model
        hidden_states = model.patch_embed.proj(tokens)
        temb = model.time_embed.t_embedder(timesteps_proj)
        embedded_timestep = model.time_embed.norm(timesteps_proj)
        for block in model.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                embedded_timestep,
                temb,
                None,
                None,
                None,
                None,
            )
        hidden_states = model.norm_out(hidden_states, embedded_timestep, temb)
        return model.proj_out(hidden_states)


class DenoiseDit(nn.Module):
    """参照 denoise から**本物のまま**呼ばれる縮図（`CosmosTransformer3DModel` の呼ばれ方）。

    {@link TinyRawDit} と別に要るのは、{@link anima.calib.capture_stage_batches} が
    `pipeline_ref.reference_steps` を通して DiT を回すため — 入口は keyword 5 本、latent は
    rank5 `[1, LATENT_CHANNELS, 1, h, w]` で、出力も同じ形（Euler 更新 `x + Δσ·pred` が成立
    する形）である必要がある。

    MUST: `reference_steps` を写しに差し替えない。CFG の分岐（`skip_uncond`）を写すと、
    捕捉バッチ数の検算門が**テスト側の写し**を検算するだけになり、上流の分岐が変わった日に
    テストも一緒にずれる。
    """

    def __init__(self, blocks: int = 2, seed: int = 11) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.patch_embed = nn.Linear(pipeline_ref.LATENT_CHANNELS, HIDDEN)
        self.time_embed = TinyEmbedding()
        self.transformer_blocks = nn.ModuleList(TinyBlock() for _ in range(blocks))
        self.norm_out = TinyAdaLayerNorm()
        self.proj_out = nn.Linear(HIDDEN, pipeline_ref.LATENT_CHANNELS)
        self.eval()

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        padding_mask: torch.Tensor,
        return_dict: bool = True,
    ):
        batch, channels, frames, height, width = hidden_states.shape
        tokens = hidden_states.reshape(batch, channels, height * width).transpose(1, 2)
        hidden = self.patch_embed(tokens)
        temb = self.time_embed.t_embedder(timestep.reshape(1, 1).expand(batch, HIDDEN))
        embedded_timestep = self.time_embed.norm(temb)
        for block in self.transformer_blocks:
            hidden = block(
                hidden, encoder_hidden_states, embedded_timestep, temb, None, None, None, None
            )
        hidden = self.norm_out(hidden, embedded_timestep, temb)
        out = self.proj_out(hidden).transpose(1, 2).reshape(batch, channels, frames, height, width)
        return out if return_dict else (out,)


def embed_for(text: str) -> torch.Tensor:
    """プロンプト → テキスト条件（決定的・文字列ごとに別のテンソル）。

    cond と uncond が**別の埋め込み**であることを捕捉の中身から言うための足場 — 同じ値を
    返す stub にすると、「uncond に cond をそのまま流す」壊れ方がテストから見えなくなる。
    """
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    return torch.randn(1, TOKENS, HIDDEN, generator=torch.Generator().manual_seed(seed))


def make_wrapper(blocks: int = 3, seed: int = 0, dit: type[TinyRawDit] = TinyRawDit) -> TinyWrapper:
    torch.manual_seed(seed)
    wrapper = TinyWrapper(dit(blocks))
    wrapper.eval()
    return wrapper


def make_probes(count: int = 3, seed: int = 7) -> tuple[tuple[torch.Tensor, ...], ...]:
    """ラッパの位置引数（実物では 1 本目が probe・残りは参照 denoise が作る条件に相当）。"""
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        (
            torch.randn(1, TOKENS, PATCH_IN, generator=generator),
            torch.randn(1, HIDDEN, generator=generator),
            torch.randn(1, TOKENS, HIDDEN, generator=generator),
        )
        for _ in range(count)
    )


def capture_batches(raw: nn.Module, probes) -> tuple:
    """**素のモデル**を回して先頭 block への `(args, kwargs)` を採る。

    実物では `calib.capture_stage_batches` が参照 denoise で同じことをする（駆動されるのは
    `wrapper.model` = 素の `CosmosTransformer3DModel`）— ここでラッパを回してしまうと捕捉元と
    グラフが同一コードパスになり、付随引数一致門が自明に通る入力しか作れない。
    """
    blocks = raw.transformer_blocks
    names = tuple(inspect.signature(blocks[0].forward).parameters)
    batches: list = []

    def catch(_module, args, kwargs):
        batches.append(((args[0].detach(),), dict(zip(names[1:], args[1:], strict=False)) | kwargs))

    handle = blocks[0].register_forward_pre_hook(catch, with_kwargs=True)
    try:
        with torch.no_grad():
            for probe in probes:
                raw(*probe)
    finally:
        handle.remove()
    return tuple(batches)


def i4_args(
    *,
    no_calib: bool = False,
    model: str = ANIMA_TURBO_MODEL_NAME,
    adaln_i8: bool = False,
    calib_device: str = CALIB_SHIPPABLE_DEVICE,
) -> argparse.Namespace:
    return argparse.Namespace(
        dtype="i4",
        no_calib=no_calib,
        calib_prompts=2,
        repo="unused",
        model=model,
        i4_adaln_i8=adaln_i8,
        calib_device=calib_device,
    )


@pytest.fixture
def stub_capture(monkeypatch):
    """`capture_stage_batches` を差し替える（実物はテキスト前段 + 参照 denoise を回す）。

    差し替え先は**素のモデル**（`wrapper.model`）を実際に走らせて採るので、`(args, kwargs)` の
    形も捕捉元も実物と同じまま。返すのは `(呼び出しごとのプロンプト本数, 捕捉時点の重み)` で、
    後者は「どの重みが丸まった状態で校正入力を採ったか」を呼び出し側が主張するための写し
    （バッチ自体は install 時に採るが、スタブは**校正が呼ぶその瞬間**に写す）。
    """
    calls: list[int] = []
    at_capture: dict[str, torch.Tensor] = {}

    def install(wrapper: nn.Module, probes=None) -> tuple[list[int], dict[str, torch.Tensor]]:
        batches = capture_batches(wrapper.model, probes if probes is not None else make_probes())

        def stub(model, prompts, **kwargs):
            calls.append(len(prompts))
            at_capture.update({name: p.detach().clone() for name, p in wrapper.named_parameters()})
            return batches

        monkeypatch.setattr(calib, "capture_stage_batches", stub)
        return calls, at_capture

    return install


@pytest.fixture
def stub_text_stack(monkeypatch):
    """テキスト前段（Qwen3 + conditioner）**だけ**を差し替える（参照 denoise は本物を回す）。

    戻り値は `encode_prompt` が受け取ったプロンプトの並び — uncond 側の埋め込みが本当に
    negative prompt から作られているかを、呼ばれ方の側からも見るため。
    """
    seen: list[str] = []

    def load_text_stack(repo: str, dtype: str) -> str:
        return f"stack:{repo}:{dtype}"

    def encode_prompt(_stack: str, _max_len: int, text: str) -> dict[str, torch.Tensor]:
        seen.append(text)
        return {"encoder_hidden_states": embed_for(text)}

    monkeypatch.setattr(calib.pipeline_ref, "load_text_stack", load_text_stack)
    monkeypatch.setattr(calib.pipeline_ref, "encode_prompt", encode_prompt)
    return seen


def capture(model: nn.Module, prompts, conditions: calib.CalibConditions):
    """縮図の DiT に対する {@link anima.calib.capture_stage_batches}（解像度だけ縮める）。"""
    return calib.capture_stage_batches(
        model, prompts, repo="unused", conditions=conditions, resolution=CAPTURE_RESOLUTION
    )


def quantize(wrapper: nn.Module, args: argparse.Namespace | None = None):
    return export_anima._fake_quant(
        args if args is not None else i4_args(),
        wrapper,
        "transformer",
        calib_probe=make_probes(count=1, seed=3)[0],
    )


class TestCalibConditions:
    """校正条件は配布形の既定から**導く**（波 J-4 ② — 2026-08-23）。

    turbo 前提のモジュール定数（8 step / CFG 1）だった頃は、素版の i4 席がこの 1 点で
    塞がっていた。導出へ移した以上、見るべきは「turbo が 1 ビットも動いていないこと」と
    「素版が自分の既定で回ること」の 2 つ。
    """

    def test_turbo_derives_what_the_module_constants_used_to_hold(self) -> None:
        """MUST: turbo の校正条件は据え置き（丸め結果が動くと配布済みの資産と食い違う）。"""
        conditions = calib.calib_conditions(ANIMA_TURBO_MODEL_NAME)

        assert (conditions.steps, conditions.guidance) == (8, 1.0)
        assert conditions.branches == 1, "CFG=1 は uncond 分岐を計算しない"

    def test_the_plain_models_derive_their_cfg_conditions(self) -> None:
        """素版 3 モデルは多 step + CFG — 1 step が forward 2 本になる。"""
        for model in BASE_MODELS:
            conditions = calib.calib_conditions(model)

            assert (conditions.steps, conditions.guidance) == (20, 4.0), model
            assert conditions.branches == 2, model
            assert conditions.negative_prompt, model

    def test_every_model_reads_its_own_pipeline_config(self) -> None:
        """MUST: 校正条件と配布既定を独立に持たない（写しがあれば片方だけ古くなる）。"""
        for model, spec in ANIMA_MODELS.items():
            defaults = spec.pipeline_config["defaults"]
            conditions = calib.calib_conditions(model)

            assert conditions.steps == defaults["steps"], model
            assert conditions.guidance == defaults["guidanceScale"], model
            assert conditions.negative_prompt == defaults["negativePrompt"], model

    def test_it_follows_a_change_to_the_shipped_defaults(self, monkeypatch) -> None:
        """故障注入: 配布既定を動かすと校正条件も動く（定数へ戻したら緑にならない）。"""
        spec = ANIMA_MODELS[ANIMA_TURBO_MODEL_NAME]
        moved = replace(
            spec,
            pipeline_config={
                "defaults": {**spec.pipeline_config["defaults"], "steps": 12, "guidanceScale": 2.5}
            },
        )
        monkeypatch.setitem(ANIMA_MODELS, ANIMA_TURBO_MODEL_NAME, moved)

        conditions = calib.calib_conditions(ANIMA_TURBO_MODEL_NAME)

        assert (conditions.steps, conditions.guidance, conditions.branches) == (12, 2.5, 2)


class TestCaptureBranches:
    """CFG > 1 の捕捉 — 配布実行時に DiT を通る活性は cond / uncond の**両方**。

    cond に絞ると「実際に流れる活性の半分」で丸め先を選ぶことになり、しかも本数の帳尻は
    合うので資産からも診断行からも読めない。分岐を作るのは `pipeline_ref.reference_steps`
    （本物を回す）で、ここが見るのはその結果を捕捉と検算門がどう数えるか。
    """

    def test_cfg_one_captures_one_batch_per_step(self, stub_text_stack) -> None:
        conditions = calib.CalibConditions(steps=3, guidance=1.0, negative_prompt="neg")

        batches = capture(DenoiseDit(), ("a", "b"), conditions)

        assert len(batches) == 2 * 3
        assert stub_text_stack == ["a", "b", "neg"], "uncond 用の埋め込みは常に用意する"

    def test_cfg_above_one_captures_both_branches(self, stub_text_stack) -> None:
        conditions = calib.CalibConditions(steps=3, guidance=4.0, negative_prompt="neg")

        batches = capture(DenoiseDit(), ("a", "b"), conditions)

        assert len(batches) == 2 * 3 * 2

    def test_the_uncond_branch_carries_the_negative_prompt(self, stub_text_stack) -> None:
        """MUST: uncond へ cond と同じ埋め込みを流さない。

        流すと `uncond == cond` になって CFG が恒等に退化し、**本数だけ 2 倍**の H に同じ
        活性が積まれる。バッチ数の検算門は通り、丸めも完走し、記録にも CFG 4 と書かれる。
        """
        conditions = calib.CalibConditions(steps=2, guidance=4.0, negative_prompt="neg")

        batches = capture(DenoiseDit(), ("a",), conditions)

        embeds = [kwargs["encoder_hidden_states"] for _args, kwargs in batches]
        assert sum(torch.equal(embed, embed_for("a")) for embed in embeds) == 2
        assert sum(torch.equal(embed, embed_for("neg")) for embed in embeds) == 2

    def test_the_turbo_conditions_capture_the_same_batches_as_before(self, stub_text_stack) -> None:
        """導出後も turbo は プロンプト × 8 step × 1 分岐（定数だった頃と同じ本数）。"""
        batches = capture(DenoiseDit(), ("a", "b"), calib.calib_conditions(ANIMA_TURBO_MODEL_NAME))

        assert len(batches) == 2 * 8

    def test_a_branch_count_that_disagrees_with_the_capture_fails_loudly(
        self, monkeypatch, stub_text_stack
    ) -> None:
        """検算門: 想定の分岐数と実際の forward 本数が食い違ったら止まる。

        壊れ方は「CFG > 1 でも uncond を回さなくなる」— H に積まれる標本が半分になるだけで、
        丸めは普通に完走し格納形も本数も変わらない。
        """
        conditions = calib.CalibConditions(steps=2, guidance=4.0, negative_prompt="neg")
        genuine = calib.pipeline_ref.reference_steps
        monkeypatch.setattr(
            calib.pipeline_ref, "reference_steps", lambda *args: genuine(*args[:-1], 1.0)
        )

        with pytest.raises(AssertionError, match="校正バッチが"):
            capture(DenoiseDit(), ("a",), conditions)


class TestModelSelection:
    """`--model` は校正条件の引き先（既定を置かない — 黙って turbo の条件で回さない）。"""

    def test_the_calibration_refuses_to_run_without_a_named_model(self, monkeypatch, capsys):
        """MUST: 既定を置くと `--model` 忘れが「turbo の条件で校正した素版」を静かに産む。"""
        monkeypatch.setattr("sys.argv", ["export.py", "--dtype", "i4"])

        with pytest.raises(SystemExit):
            export_anima.main()

        assert "--model が要る" in capsys.readouterr().err

    def test_the_knob_is_refused_where_no_calibration_runs(self, monkeypatch, capsys):
        """効かないノブを黙って受けない（`--no-calib` に引くべき条件は無い）。"""
        monkeypatch.setattr(
            "sys.argv",
            ["export.py", "--dtype", "i4", "--no-calib", "--model", ANIMA_TURBO_MODEL_NAME],
        )

        with pytest.raises(SystemExit):
            export_anima.main()

        assert "--model は" in capsys.readouterr().err

    def test_the_export_path_hands_the_models_conditions_to_the_capture(self, monkeypatch):
        """MUST: 捕捉が受け取る条件は `--model` のもの（turbo 固定へ戻ったら緑にならない）。"""
        wrapper = make_wrapper()
        batches = capture_batches(wrapper.model, make_probes())
        seen: dict = {}

        def stub(model, prompts, **kwargs):
            seen.update(kwargs)
            return batches

        monkeypatch.setattr(calib, "capture_stage_batches", stub)

        quantize(wrapper, i4_args(model=ANIMA_BASE_MODEL_NAME))

        assert seen["conditions"] == calib.calib_conditions(ANIMA_BASE_MODEL_NAME)


class TestStageSplit:
    def test_the_stage_chain_reproduces_the_block_loop_bit_exactly(self):
        """MUST: 分解が本物の経路と 1bit も違わない（違えば別経路の GPTQ を出荷する）。"""
        wrapper = make_wrapper()

        batch = calib.assert_stage_split(
            wrapper, make_probes(count=1)[0], calib.dit_stages(wrapper)
        )

        assert set(batch[1]) == {
            "encoder_hidden_states",
            "embedded_timestep",
            "temb",
            "image_rotary_emb",
            "extra_pos_emb",
            "attention_mask",
            "controlnet_residual",
        }

    def test_a_dropped_stage_is_caught(self):
        """段を 1 つ落とすと最終 hidden が変わる（門が素通りしないことの実測）。"""
        wrapper = make_wrapper()
        stages = calib.dit_stages(wrapper)

        with pytest.raises(AssertionError, match="ビット一致しない"):
            calib.assert_stage_split(wrapper, make_probes(count=1)[0], stages[:-1])

    def test_a_reordered_stage_chain_is_caught(self):
        """順序が入れ替わっても形は合う（値だけが静かに変わる壊れ方）。"""
        wrapper = make_wrapper()
        stages = calib.dit_stages(wrapper)
        swapped = (stages[1], stages[0], *stages[2:])

        with pytest.raises(AssertionError, match="ビット一致しない"):
            calib.assert_stage_split(wrapper, make_probes(count=1)[0], swapped)

    def test_the_stage_prefix_lands_in_the_wrapper_fqn_space(self):
        """接頭辞つきの局所 FQN がラッパの実 FQN と一致する（台帳キーの空間が決まる場所）。"""
        wrapper = make_wrapper(blocks=2)

        names = calib.stage_linear_names(calib.dit_stages(wrapper))

        assert names == {
            f"model.transformer_blocks.{index}.{child}"
            for index in range(2)
            for child in (
                "attn",
                "ff",
                *(f"norm{at}.linear_{part}" for at in (1, 2, 3) for part in (1, 2)),
            )
        }
        assert names <= {name for name, _module in wrapper.named_modules()}


class TestBatchesMatchGraph:
    def test_the_raw_model_and_the_wrapper_call_the_block_alike(self):
        """素のモデル（捕捉元）とラッパ（丸める対象）が block を同じ顔ぶれで呼ぶ。

        2 経路が**同じ関数の別実装**であること（出力のビット一致）も一緒に見る — ここが
        崩れていると、下の drift 注入が「本物の食い違い」を写していない。
        """
        wrapper = make_wrapper()
        probe = make_probes(count=1)[0]
        stages = calib.dit_stages(wrapper)
        graph_batch = calib.assert_stage_split(wrapper, probe, stages)

        calib.assert_calib_batches_match_graph(
            graph_batch, capture_batches(wrapper.model, make_probes())
        )

        with torch.no_grad():
            assert torch.equal(wrapper(*probe), wrapper.model(*probe))

    def test_a_batch_with_a_different_keyword_set_is_caught(self):
        """参照 denoise とラッパで block の呼ばれ方が違うと、別条件で選んだ丸め先を出荷する。"""
        wrapper = make_wrapper()
        stages = calib.dit_stages(wrapper)
        graph_batch = calib.assert_stage_split(wrapper, make_probes(count=1)[0], stages)
        batches = capture_batches(wrapper.model, make_probes(count=1))
        args, kwargs = batches[0]
        drifted = ((args, {**kwargs, "img_context": None}),)

        with pytest.raises(AssertionError, match="block の呼び方が違う"):
            calib.assert_calib_batches_match_graph(graph_batch, drifted)

    def test_the_export_path_runs_the_gate(self, stub_capture):
        """MUST: 門が `_round_i4_calibrated` に本当に嵌まっていること（関数単体では足りない）。

        壊れ方は「素のモデルが block の付随引数を 1 本落とす」— 数値は普通に出るので、門が
        外れていると別条件で選んだ丸め先がそのまま出荷される。
        """
        wrapper = make_wrapper(dit=DriftingRawDit)
        stub_capture(wrapper)

        with pytest.raises(AssertionError, match="block の呼び方が違う"):
            quantize(wrapper)


class TestCalibratedI4:
    def test_the_ledger_keys_live_in_the_wrapper_fqn_space(self, stub_capture):
        """MUST: 台帳のキー = safetensors のテンソルキー（emit の突合はここで決まる）。"""
        wrapper = make_wrapper()
        stub_capture(wrapper)

        scales, overrides = quantize(wrapper)

        parameters = dict(wrapper.named_parameters())
        assert set(overrides) <= set(parameters), "i4 席のキーがラッパの FQN 空間に無い"
        assert set(scales) <= set(parameters)
        assert set(overrides) == {
            f"{name}.weight" for name in export_anima._i4_module_names(wrapper)
        }

    def test_the_eligible_set_is_split_exclusively_between_the_two_paths(self):
        """block 内（校正）と block 外（RTN）で**過不足なく排他**（`quantize.py` の混成 MUST）。"""
        wrapper = make_wrapper()
        stage_names = calib.stage_linear_names(calib.dit_stages(wrapper))
        i4_names = export_anima._i4_module_names(wrapper)

        assert i4_names - stage_names == export_anima.NON_STAGE_I4_WEIGHTS
        assert stage_names <= i4_names, "block 内の linear が i4 適格から漏れている"
        # patchify 入口は量子化軸が g32 非整除 — i4 側に入らない（i8 の担当のまま）
        assert "model.patch_embed.proj" not in i4_names

    def test_calibration_produces_a_different_rounding_from_plain_rtn(self, stub_capture):
        """校正が素通りしたら格納形が同じなので資産からは読めない — 値差で実測する。"""
        calibrated = make_wrapper()
        plain = make_wrapper()
        stub_capture(calibrated)

        quantize(calibrated)
        quantize(plain, i4_args(no_calib=True))

        after = dict(calibrated.named_parameters())
        differing = sorted(
            name
            for name, weight in plain.named_parameters()
            if not torch.equal(weight, after[name])
        )
        assert differing, "GPTQ が RTN と同じ値を出した（校正が効いていない）"
        assert all(name.startswith("model.transformer_blocks.") for name in differing)

    def test_the_capture_sees_the_shipped_condition(self, stub_capture):
        """MUST: block 外は丸めた**後**・block 内は丸める**前**に校正入力を採る。

        後半（block 内が素のまま）が「丸めた重みが作った活性から同じ重みの丸め先を選ぶ
        循環」を、前半（block 外が丸まっている）が「f32 の周辺で選んだ丸め先を i4 の周辺と
        組んで配る」を、それぞれ捕まえる。
        """
        wrapper = make_wrapper()
        before = {name: p.detach().clone() for name, p in wrapper.named_parameters()}
        _calls, at_capture = stub_capture(wrapper)

        quantize(wrapper)

        for name in export_anima.NON_STAGE_I4_WEIGHTS:
            key = f"{name}.weight"
            assert not torch.equal(before[key], at_capture[key]), f"{key} が捕捉時に素のまま"
        for key in (name for name in before if name.startswith("model.transformer_blocks.")):
            assert torch.equal(before[key], at_capture[key]), f"{key} が捕捉前に丸まっている"

    def test_the_capture_sees_the_i8_side_already_rounded(self, stub_capture):
        """MUST: i8 側（patchify 入口）も捕捉より**前**に丸まっていること。

        `patch_embed.proj` は量子化軸が g32 非整除で i4 に載らない唯一の linear（実 DiT でも
        1 本）だが、**block 0 の入力そのものを作る**。捕捉より後に丸めると「f32 の patchify を
        通った活性」で選んだ丸め先を i8 の patchify と組んで配ることになり、block 外の i4 を
        先に丸める理由（順序 MUST ②）がこの 1 本にだけ効いていない状態になる。
        """
        wrapper = make_wrapper()
        before = {name: p.detach().clone() for name, p in wrapper.named_parameters()}
        _calls, at_capture = stub_capture(wrapper)

        quantize(wrapper)

        key = "model.patch_embed.proj.weight"
        assert key in before, "i8 側の重みが縮図に無い（テストが空振りしている）"
        assert not torch.equal(before[key], at_capture[key]), f"{key} が捕捉時に素のまま"

    def test_the_out_of_block_weights_are_rounded_by_the_plain_path_in_both_modes(
        self, stub_capture
    ):
        """block 外はどちらの経路でも素の RTN（丸め値はビット一致）。"""
        calibrated = make_wrapper()
        plain = make_wrapper()
        stub_capture(calibrated)

        quantize(calibrated)
        quantize(plain, i4_args(no_calib=True))

        after = dict(calibrated.named_parameters())
        for name in export_anima.NON_STAGE_I4_WEIGHTS:
            key = f"{name}.weight"
            assert torch.equal(dict(plain.named_parameters())[key], after[key]), key

    def test_an_unexpected_non_stage_eligible_fails_loudly(self, monkeypatch, stub_capture):
        """block 外の適格が宣言と違う = 上流の構成が動いた合図（黙って分類を変えない）。"""
        wrapper = make_wrapper()
        stub_capture(wrapper)
        monkeypatch.setattr(export_anima, "NON_STAGE_I4_WEIGHTS", frozenset())

        with pytest.raises(AssertionError, match="block の外の i4 適格"):
            quantize(wrapper)

    def test_a_block_linear_outside_the_i4_set_fails_loudly(self, monkeypatch, stub_capture):
        """block 内に非適格が混じると「走査の本数 = 丸めた本数」の門が張れない。"""
        wrapper = make_wrapper()
        stub_capture(wrapper)
        full = export_anima._i4_module_names(wrapper)
        dropped = frozenset(full - {"model.transformer_blocks.0.attn"})
        monkeypatch.setattr(export_anima, "_i4_module_names", lambda model: dropped)

        with pytest.raises(AssertionError, match="i4 適格でない"):
            quantize(wrapper)

    def test_a_ledger_that_overlaps_the_plain_path_fails_loudly(self, monkeypatch, stub_capture):
        """同じ重みを 2 経路で丸めると値だけが静かに狂う（格納形も本数も正しく見える）。"""
        wrapper = make_wrapper()
        stub_capture(wrapper)
        genuine = calib.calibrate_i4
        outside = f"{sorted(export_anima.NON_STAGE_I4_WEIGHTS)[0]}.weight"

        def overlapping(rig, include=None):
            report, ledger = genuine(rig, include)
            merged = dict(ledger.scales) | {outside: torch.ones(HIDDEN, 1)}
            return report, replace(ledger, scales=merged)

        monkeypatch.setattr(calib, "calibrate_i4", overlapping)

        with pytest.raises(AssertionError, match="二重丸め"):
            quantize(wrapper)

    def test_a_report_short_of_the_scan_fails_loudly(self, monkeypatch, stub_capture):
        """丸め漏れは品質を**良い側**に見せる（素通りを数字から読めない）。"""
        wrapper = make_wrapper()
        stub_capture(wrapper)
        genuine = calib.calibrate_i4

        def short(rig, include=None):
            report, ledger = genuine(rig, include)
            return replace(report, layers=report.layers[:-1]), ledger

        monkeypatch.setattr(calib, "calibrate_i4", short)

        with pytest.raises(AssertionError, match="走査の"):
            quantize(wrapper)

    def test_zero_calibration_prompts_fail_loudly(self):
        """MUST: 素の RTN へ黙って落ちる分岐は持たない。"""
        wrapper = make_wrapper()

        with pytest.raises(AssertionError, match="校正プロンプトが 1 件も無い"):
            capture(wrapper.model, (), calib.calib_conditions(ANIMA_TURBO_MODEL_NAME))


class TestOptOut:
    def test_the_i4_series_is_calibrated_by_default(self, monkeypatch):
        """MUST: CLI の `--dtype i4` はフラグ無しで校正付き（`--no-calib` だけが opt-out）。"""
        captured: dict[str, argparse.Namespace] = {}

        def stub(target, args, out_dir):
            captured[target] = args
            return {"target": target}

        monkeypatch.setattr(
            "sys.argv", ["export.py", "--dtype", "i4", "--model", ANIMA_TURBO_MODEL_NAME]
        )
        monkeypatch.setattr(export_anima, "emit_target", stub)

        export_anima.main()

        assert captured["transformer"].no_calib is False
        assert captured["transformer"].calib_prompts == export_anima.DEFAULT_CALIB_PROMPTS

    def test_no_calib_never_touches_the_calibration_path(self, stub_capture):
        """opt-out は「校正が痩せる」ではなく**回らない**こと（費用も品質も別物になる）。"""
        wrapper = make_wrapper()
        calls, _at_capture = stub_capture(wrapper)

        quantize(wrapper, i4_args(no_calib=True))

        assert calls == []

    def test_the_calibration_knobs_are_refused_outside_i4(self, monkeypatch):
        """効かないノブを黙って受けない（f16 に校正の経路は 1 本も無い）。"""
        monkeypatch.setattr("sys.argv", ["export.py", "--dtype", "f16", "--no-calib"])

        with pytest.raises(SystemExit):
            export_anima.main()

    def test_a_resolution_off_the_calibration_condition_is_refused(self, monkeypatch, capsys):
        """MUST: 512px で選んだ丸め先を別解像度のグラフへ焼かない（数値は普通に出る壊れ方）。"""
        monkeypatch.setattr(
            "sys.argv",
            [
                "export.py",
                "--dtype",
                "i4",
                "--model",
                ANIMA_TURBO_MODEL_NAME,
                "--resolution",
                "1024",
            ],
        )

        with pytest.raises(SystemExit):
            export_anima.main()

        assert f"{calib.CALIB_RESOLUTION}px 固定" in capsys.readouterr().err

    def test_no_calib_leaves_the_resolution_free(self, monkeypatch):
        """縛りは校正経路のもの — 素の RTN（`--no-calib`）は解像度に何も要求しない。"""
        captured: dict[str, argparse.Namespace] = {}

        def stub(target, args, out_dir):
            captured[target] = args
            return {"target": target}

        monkeypatch.setattr(
            "sys.argv", ["export.py", "--dtype", "i4", "--no-calib", "--resolution", "1024"]
        )
        monkeypatch.setattr(export_anima, "emit_target", stub)

        export_anima.main()

        assert captured["transformer"].resolution == 1024


class TestAdalnSegments:
    """adaLN の判定は**セグメント一致**（`.` で割った要素）— 部分文字列でも接尾辞でもない。"""

    def test_it_matches_a_segment_and_not_a_substring(self):
        assert calib.is_adaln("norm1.linear_1")
        assert calib.is_adaln("norm3.linear_2")
        # 部分文字列一致だと `norm1x` のような綴りを巻き込み、接尾辞一致だと adaLN 配下の子を
        # 取りこぼす — どちらも「除外したつもりの本数」が黙ってずれる。
        assert not calib.is_adaln("norm1x.linear_1")
        assert not calib.is_adaln("norm_out.linear_1")
        assert not calib.is_adaln("attn")

    def test_it_answers_alike_for_stage_local_and_wrapper_fqns(self):
        """MUST: 校正の `include`（局所名）と export の集合分割（ラッパ内 FQN）が同じ 1 実装。"""
        assert calib.is_adaln("norm2.linear_1")
        assert calib.is_adaln("model.transformer_blocks.7.norm2.linear_1")

    def test_it_reports_which_spellings_were_seen(self):
        """片側だけの改名を捕まえる網 — 「1 本も無い」だけを見る門は素通りする。"""
        names = {"model.transformer_blocks.0.norm1.linear_1", "model.transformer_blocks.0.attn"}

        assert calib.adaln_segments_seen(names) == frozenset({"norm1"})
        assert calib.adaln_segments_seen(()) == frozenset()


class TestAdalnI8Variant:
    """`--i4-adaln-i8`（既定 OFF）= adaLN + block 外を i8 格納へ回す感度実験変種。

    既定を 1 ビットも動かさないこと（集合・経路・値）と、変種が**本当に**役割で割れている
    ことの両方を見る。格納形は変種でも「i4 と i8 が混ざった系列」のままなので、資産からは
    どちらで焼いたか読めない — 割り方の綻びは全部ここで捕まえるしかない。
    """

    @staticmethod
    def _split(wrapper: nn.Module) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
        """`(適格, block 内 adaLN, block 内 adaLN 以外)`。"""
        eligible = export_anima._i4_module_names(wrapper)
        stage_names = calib.stage_linear_names(calib.dit_stages(wrapper))
        adaln = frozenset(name for name in stage_names if calib.is_adaln(name))
        return eligible, adaln, stage_names - adaln

    def test_the_default_stores_every_eligible_weight_as_i4(self, stub_capture):
        """既定 OFF の格納集合 = 適格の全量（変種の配管が既定の割り方へ触っていない）。"""
        wrapper = make_wrapper()
        stub_capture(wrapper)
        eligible, _adaln, _rest = self._split(wrapper)

        _scales, overrides = quantize(wrapper)

        assert set(overrides) == {f"{name}.weight" for name in eligible}

    def test_the_default_hands_no_include_predicate_to_the_calibration(self, stub_capture):
        """MUST: 既定は述語を**持たない**（`None`）— 「全部」を綴り直した述語は走査の定義が
        動いた日に片側だけ追随して黙ってずれる。"""
        wrapper = make_wrapper()
        stub_capture(wrapper)
        genuine = calib.calibrate_i4
        seen: list[object] = []

        def spy(rig, include=None):
            seen.append(include)
            return genuine(rig, include)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(calib, "calibrate_i4", spy)
            quantize(wrapper)

        assert seen == [None]

    def test_the_default_matches_an_independently_expressed_plain_rounding(self):
        """既定 OFF（`--no-calib`）が「適格全部を素の RTN → 残りを i8」と**ビット一致**する。

        変種の配管は集合の引き算を 1 つ挟むだけで、既定では引く側が空 — それが値の側でも
        成り立っていることを、台本を通さない直接の丸めと突き合わせて実測する
        （格納形は変種でも同じなので、ずれたら資産からは読めない）。
        """
        through_export = make_wrapper()
        direct = make_wrapper()
        eligible = export_anima._i4_module_names(direct)

        quantize(through_export, i4_args(no_calib=True))
        fake_quant_int4(
            direct,
            include=lambda name: name in eligible,
            op_types=export_anima.I4_MODULE_TYPES,
        )
        fake_quant_int8(direct, include=lambda name: name not in eligible)

        after = dict(through_export.named_parameters())
        assert {name for name, _ in direct.named_parameters()} == set(after)
        for name, weight in direct.named_parameters():
            assert torch.equal(weight, after[name]), name

    def test_the_variant_keeps_only_the_non_adaln_block_linears_in_i4(self, stub_capture):
        wrapper = make_wrapper()
        stub_capture(wrapper)
        _eligible, _adaln, rest = self._split(wrapper)

        _scales, overrides = quantize(wrapper, i4_args(adaln_i8=True))

        assert set(overrides) == {f"{name}.weight" for name in rest}
        assert all(dtype == "i4" for dtype in overrides.values())

    def test_the_excluded_count_is_the_adaln_rows_plus_the_out_of_block_five(self):
        """勘定: block 数 × (セグメント 3 × linear 2) + block 外 {@link NON_STAGE_I4_WEIGHTS}。

        実物（anima-v1.0 = 28 block）ではこれが 28 × 6 + 5 = **173 本**（adaLN 168 + 5）。
        本数を焼き込まないのは `--num-layers` の縮小 export でも同じ式で通るため。
        """
        blocks = 4
        wrapper = make_wrapper(blocks=blocks)

        excluded = export_anima._adaln_i8_names(wrapper)

        per_block = len(calib.ADALN_SEGMENTS) * 2
        assert len(excluded) == blocks * per_block + len(export_anima.NON_STAGE_I4_WEIGHTS)
        assert excluded >= export_anima.NON_STAGE_I4_WEIGHTS
        assert len(excluded - export_anima.NON_STAGE_I4_WEIGHTS) == blocks * per_block

    def test_the_variant_rounds_the_excluded_weights_as_i8(self, stub_capture):
        """MUST: 外した側は i8 で丸まる（i4 からも i8 からも漏れて f32 のまま、を潰す）。"""
        wrapper = make_wrapper()
        before = {name: p.detach().clone() for name, p in wrapper.named_parameters()}
        stub_capture(wrapper)

        scales, overrides = quantize(wrapper, i4_args(adaln_i8=True))

        after = dict(wrapper.named_parameters())
        for name in export_anima._adaln_i8_names(wrapper):
            key = f"{name}.weight"
            assert key in scales, f"{key} の scale が台帳に無い"
            assert key not in overrides, f"{key} が i4 の格納指定を持っている"
            # per-channel の scale は行ごと 1 本（group scale は行 × group 数）。
            assert list(scales[key].shape) == [after[key].shape[0], 1]
            assert not torch.equal(before[key], after[key]), f"{key} が丸まっていない"

    def test_the_calibration_rounds_only_the_non_adaln_block_linears(self, stub_capture):
        """GPTQ に載る集合が「block 内 − adaLN」と過不足なく一致する（述語と格納集合の同一性）。"""
        wrapper = make_wrapper()
        stub_capture(wrapper)
        _eligible, adaln, rest = self._split(wrapper)
        genuine = calib.calibrate_i4
        rounded: list[frozenset[str]] = []

        def spy(rig, include=None):
            report, ledger = genuine(rig, include)
            rounded.append(frozenset(layer.fqn for layer in report.layers))
            return report, ledger

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(calib, "calibrate_i4", spy)
            quantize(wrapper, i4_args(adaln_i8=True))

        assert rounded == [frozenset(f"{name}.weight" for name in rest)]
        assert adaln, "縮図に adaLN が無い（テストが空振りしている）"

    def test_the_capture_sees_the_adaln_side_already_rounded(self, stub_capture):
        """MUST: 変種では adaLN も**捕捉より前**に i8 で丸まっていること。

        配布実行時に modulation を作るのは i8 の adaLN なので、後に回すと「f32 の
        modulation を通った活性」で選んだ丸め先を i8 の modulation と組んで配ることになる
        （block 外を先に丸める理由と同じ — 順序 MUST ②）。
        """
        wrapper = make_wrapper()
        before = {name: p.detach().clone() for name, p in wrapper.named_parameters()}
        _calls, at_capture = stub_capture(wrapper)
        _eligible, adaln, rest = self._split(wrapper)

        quantize(wrapper, i4_args(adaln_i8=True))

        for name in adaln:
            key = f"{name}.weight"
            assert not torch.equal(before[key], at_capture[key]), f"{key} が捕捉時に素のまま"
        for name in rest:
            key = f"{name}.weight"
            assert torch.equal(before[key], at_capture[key]), f"{key} が捕捉前に丸まっている"

    def test_the_plain_rtn_mode_splits_the_same_way(self):
        """`--no-calib` でも割り方は同じ（変種は格納の割りであって校正の有無とは直交）。"""
        wrapper = make_wrapper()
        _eligible, _adaln, rest = self._split(wrapper)

        _scales, overrides = quantize(wrapper, i4_args(no_calib=True, adaln_i8=True))

        assert set(overrides) == {f"{name}.weight" for name in rest}

    def test_a_renamed_adaln_segment_fails_loudly(self, monkeypatch, stub_capture):
        """MUST: 綴りが 1 つでも見つからなければ落ちる — 片側だけの改名は本数の門を素通りする。"""
        wrapper = make_wrapper()
        stub_capture(wrapper)
        monkeypatch.setattr(calib, "ADALN_SEGMENTS", calib.ADALN_SEGMENTS | {"norm4"})

        with pytest.raises(AssertionError, match="adaLN の linear が 1 本も無い綴り"):
            quantize(wrapper, i4_args(adaln_i8=True))

    def test_an_exclusion_outside_the_eligible_set_fails_loudly(self, monkeypatch, stub_capture):
        """除外の綴りが適格に無い = 上流の構成が動いた合図（黙って引き算を空振りさせない）。"""
        wrapper = make_wrapper()
        stub_capture(wrapper)
        monkeypatch.setattr(
            export_anima, "_adaln_i8_names", lambda model: frozenset({"model.nowhere"})
        )

        with pytest.raises(AssertionError, match="が i4 適格に無い"):
            quantize(wrapper, i4_args(adaln_i8=True))

    def test_the_out_of_block_weights_leave_the_plain_i4_path_entirely(self, stub_capture):
        """変種では素の RTN i4 の経路が**1 本も丸めない**（block 外は全部 i8 へ移る）。"""
        wrapper = make_wrapper()
        stub_capture(wrapper)
        calls: list[frozenset[str]] = []

        def spy(model, names, target, label):
            calls.append(frozenset(names))
            return export_anima._round_i4_plain(model, names, target, label)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(export_anima, "_round_i4_plain", spy)
            quantize(wrapper, i4_args(adaln_i8=True))

        assert calls == []
