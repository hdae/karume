"""i4 系列の**校正付き丸め**（GPTQ × RTN 格子）を回すリグ（perf-ledger Q-6）。

`karume.quant_calib.calibrate_stages` は「stage を 1 つずつ進めながら、その層に実際に流れる
活性から丸め先を選び直す」駆動で、格納グリッドは {@link karume.quantize.quantize_to_int4} と
**1 バイトも変わらない**（`scale = group amax / 7`・`q ∈ [−7,+7]`）。変わるのは丸め値と
scale 台帳の中身だけで、`emit` 側の格納経路も配布形のバイト構造も一切動かない。

ここが用意するのは core の駆動が要求する 3 つ:

1. **stage 列** — `DebertaV2Encoder.forward` のループを写した {@link EncoderStage} の並び。
   `(モデル内 FQN 接頭辞, モジュール)` の組で、接頭辞は **export する wrapper の FQN 空間**
   （{@link STAGE_PREFIX}）— scale 台帳のキーが safetensors のテンソルキーと同じ空間に
   居ないと emit 側の突合が空振りする（`deberta.export._fake_quant` の FQN 規律 MUST）。
2. **先頭 stage への入力** — 埋め込みと 4 次元 mask / 相対位置の添字表 / `rel_embeddings` を
   自前で組み直さず、**先頭層の呼び出しそのもの**を forward_pre_hook で捕まえる
   （{@link capture_stage_batches}）。
3. **stage 分解一致門** — 捕まえた入力を stage 列で逐次 forward した最終 hidden が、フック
   なしの wrapper forward の対応出力と**ビット一致**すること（{@link assert_stage_split}）。

MUST: 3 は丸める前に実測する。{@link EncoderStage} は上流ループの写しなので、transformers 側の
`DebertaV2Encoder.forward` が変わると黙ってずれる — ずれた側で丸めると「別の経路の GPTQ」を
出荷することになり、しかも数値は普通に出る（golden も緑のまま通る）。

NOTE（`sbv2.measure_quant` のリグと共有しない理由）: あちらが駆動するのは**素の**
`DebertaV2Model` だが、export は `deberta.patch.apply_external_rel_pos_patch()` を当てた後の
wrapper を駆動するので、stage の実シグネチャ（`relative_pos` がタプルで来る）が同一である
保証が無い。写しを 1 本にまとめると、片方の前提が動いたときにもう片方が黙って追随する。
各リグが自前の一致門で独立に自己検証する形にしてある。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from karume.quant_calib import (
    CalibReport,
    GridSpec,
    StageBatch,
    StageSpec,
    calibrate_stages,
)
from karume.quantize import DEFAULT_GROUP_SIZE, Int4Report, iter_quant_targets

#: stage 列のモデル内 FQN 接頭辞。`HiddenStatesWrapper.model` が `DebertaV2Model` なので、
#: 層の FQN は `model.encoder.layer.<層番号>.…` になる（{@link EncoderStage} が子の名前を
#: 層番号にしてあるので、接頭辞はここまでで足りる）。
STAGE_PREFIX = "model.encoder.layer"

#: {@link EncoderStage.forward} が ConvLayer 用に受け取る keyword 名。`DebertaV2Encoder` は
#: 層へ渡す 4 次元 mask とは別に、**素の 2 次元 mask** を ConvLayer へ渡す。
INPUT_MASK_KWARG = "input_mask"

#: 校正付き丸めの方式。GPTQ 以外は scale 台帳を返さない（`karume.quant_calib` の MUST）。
CALIB_METHOD = "gptq"

#: 丸め先の格納グリッド。**`rtn` × g32 = 既存 i4 系列と同じ格子**（唯一の出荷経路）。
CALIB_GRID = GridSpec(kind="rtn", group_size=DEFAULT_GROUP_SIZE)


class _FirstStageReached(Exception):  # noqa: N818 — 異常ではなく打ち切りの合図なので Error と呼ばない
    """先頭 stage の入力が揃った合図（校正 forward を打ち切るための番兵）。"""


class EncoderStage(nn.Module):
    """encoder layer 1 枚を「hidden を位置引数で受ける」形へ包む stage ラッパ。

    包む理由は 2 つあり、どちらも `DebertaV2Encoder.forward`（transformers 5.14.1）の
    呼び出しの形に由来する:

    1. 層へ mask を**位置引数**で渡す（`layer_module(next_kv, attention_mask, …)`）。
       `calibrate_stages` は次 stage へ「選んだ出力を**唯一の位置引数**」として渡す駆動なので、
       mask は keyword で運ぶ形に直さないと 2 段目以降で落ちる。
    2. **先頭層だけ**は出力に ConvLayer が乗る（ループ内の `i == 0` — stage の入力 hidden を
       残差として混ぜ直す）。ここを落とすと 2 段目以降が本物と違う hidden を見ることになり、
       校正が別の活性から丸め先を選ぶ。

    層 kwargs（`relative_pos` / `rel_embeddings` / `query_states` / `output_attentions`）は
    ループの**前**に 1 度だけ作られ全層で同一なので、`calibrate_stages` の既定（stage 間で
    kwargs 不変）にそのまま乗る。

    子モジュールの名前を**層番号**にしてあるのは、stage 内の局所 FQN
    （`0.attention.self.query_proj.weight`）へ {@link STAGE_PREFIX} を足すだけでモデル内 FQN へ
    戻すため。ConvLayer も子として登録されるが `nn.Linear` を 1 本も持たない（`Conv1d` +
    `LayerNorm`）ので、走査（`iter_quant_targets` の `nn.Linear` 限定）の対象集合は動かない。
    """

    def __init__(self, index: int, layer: nn.Module, conv: nn.Module | None) -> None:
        super().__init__()
        self.child = str(index)
        self.add_module(self.child, layer)
        self.conv = conv

    def forward(
        self, hidden: torch.Tensor, *, input_mask: torch.Tensor, **layer_kwargs: Any
    ) -> torch.Tensor:
        output = getattr(self, self.child)(hidden, **layer_kwargs)[0]
        if self.conv is None:
            return output
        return self.conv(hidden, output, input_mask)


def _encoder(wrapper: nn.Module) -> nn.Module:
    """wrapper の中の `DebertaV2Encoder` を取り出す（構成が想定と違えば fail loudly）。"""
    encoder = wrapper.model.encoder
    if encoder.conv is None:
        raise AssertionError(
            "DeBERTa encoder に ConvLayer が無い（先頭層の残差混合が消える構成 —"
            " 模型の構成が台本の想定と違う）"
        )
    return encoder


def encoder_stages(wrapper: nn.Module) -> tuple[StageSpec, ...]:
    """実行順の encoder layer を `(モデル内 FQN 接頭辞, stage)` で返す。

    **export するグラフに載る全層**を並べる（`sbv2.measure_quant` のリグが末尾の死に層を
    落とすのと違う点）— こちらは切り詰め済みの模型をそのまま出荷するので、載っている層は
    全部が配布形の重みだから。
    """
    encoder = _encoder(wrapper)
    return tuple(
        (STAGE_PREFIX, EncoderStage(index, layer, encoder.conv if index == 0 else None))
        for index, layer in enumerate(encoder.layer)
    )


def stage_linear_names(stages: Sequence[StageSpec]) -> frozenset[str]:
    """stage 列が抱える `nn.Linear` のモデル内 FQN（`.weight` を落とした**モジュール名**）。

    対象選択は core の `iter_quant_targets` の共有 — 「校正が丸める対象」と「呼び出し側が
    stage 内と数える対象」を写した別実装にすると、どちらにも入らない重みが黙って i8 へ落ちる。
    """
    return frozenset(
        f"{prefix}.{local}".removesuffix(".weight")
        for prefix, stage in stages
        for local, _weight, _axis in iter_quant_targets(stage, (nn.Linear,))
    )


def capture_stage_batches(
    wrapper: nn.Module, inputs: Sequence[Sequence[torch.Tensor]]
) -> tuple[StageBatch, ...]:
    """先頭 stage への hidden と付随引数を forward_pre_hook で捕まえる（Catcher）。

    `inputs` は wrapper の **位置引数そのもの**（`deberta.export.INPUT_ORDER` の並び）。捕まえる
    のは「embeddings を通った後の hidden と、encoder が組んだ 4 次元 mask / 相対位置 /
    `rel_embeddings`」で、自前で組み直すと transformers 側と黙って割れる。

    ConvLayer が要る 2 次元 mask は層へは渡らないので、**ConvLayer の呼び出し**からもう 1 本
    捕まえる。番兵で打ち切るのは ConvLayer 側 — そこまでで走るのは先頭層 1 枚だけ。

    MUST: 揃わずに forward が完走したら fail loudly — stage の綴りが模型の構成と食い違って
    いる合図で、黙って進むと「校正入力ゼロ」の診断が core 側で出るだけになる。
    """
    encoder = _encoder(wrapper)
    captured: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    masks: list[torch.Tensor] = []

    def catch_layer(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        captured.append((args, dict(kwargs)))

    def catch_conv(_module: nn.Module, args: tuple[Any, ...]) -> None:
        masks.append(args[2])
        raise _FirstStageReached

    handles = [
        encoder.layer[0].register_forward_pre_hook(catch_layer, with_kwargs=True),
        encoder.conv.register_forward_pre_hook(catch_conv),
    ]
    batches: list[StageBatch] = []
    try:
        for index, args in enumerate(inputs):
            captured.clear()
            masks.clear()
            try:
                with torch.no_grad():
                    wrapper(*args)
            except _FirstStageReached:
                pass
            if len(captured) != 1 or len(masks) != 1:
                raise AssertionError(
                    f"校正入力 {index} で先頭 stage の入力が揃わなかった"
                    f"（層 {len(captured)} 件 / ConvLayer {len(masks)} 件）"
                    "— stage の綴りが模型の構成と食い違っている"
                )
            layer_args, layer_kwargs = captured[0]
            if len(layer_args) != 2:
                raise AssertionError(
                    f"先頭層が位置引数 {len(layer_args)} 個で呼ばれた（hidden と mask の 2 個が"
                    "想定 — transformers 側の呼び出しの形が変わっている）"
                )
            hidden, attention_mask = layer_args
            batches.append(
                (
                    (hidden.detach(),),
                    {"attention_mask": attention_mask, INPUT_MASK_KWARG: masks[0], **layer_kwargs},
                )
            )
    finally:
        for handle in handles:
            handle.remove()
    return tuple(batches)


def assert_stage_split(
    wrapper: nn.Module,
    probe: Sequence[torch.Tensor],
    batch: StageBatch,
    stages: Sequence[StageSpec],
) -> None:
    """stage 列の逐次 forward が wrapper の最終 hidden と**ビット一致**することを見る。

    `probe` は wrapper の位置引数、`batch` は**同じ入力**で捕まえた先頭 stage への入力。
    wrapper の出力は最終層の hidden で終わる（`single_output` の有無に依らず末尾がそれ）ので、
    突合先は `outputs[-1]`。

    MUST: 丸める前に実測し、近似 tolerance ではなく `torch.equal` で見る — 同じモジュールを
    同じ順で呼んでいる以上、一致しないなら分解が本物と違う経路を通っている。
    """
    args, kwargs = batch
    with torch.no_grad():
        reference = wrapper(*probe)[-1]
        hidden = args[0]
        for _prefix, stage in stages:
            hidden = stage(hidden, **kwargs)
    if not torch.equal(hidden, reference):
        raise AssertionError(
            "stage 分解の最終 hidden が wrapper の出力とビット一致しない"
            f"（最大絶対差 {float((hidden - reference).abs().max()):.4e}）"
            " — EncoderStage の写しが DebertaV2Encoder.forward とずれている"
        )


@dataclass(frozen=True)
class CalibRig:
    """校正 1 回ぶんの足場（stage 列と先頭 stage への入力）。"""

    stages: tuple[StageSpec, ...]
    batches: tuple[StageBatch, ...]

    @property
    def tokens(self) -> int:
        """校正に積んだ hidden の総トークン数（縮小実行を数値の横に残すための診断）。"""
        return sum(int(args[0].shape[-2]) for args, _kwargs in self.batches)


def build_rig(
    wrapper: nn.Module, stages: Sequence[StageSpec], inputs: Sequence[Sequence[torch.Tensor]]
) -> CalibRig:
    """校正の足場を組む（Catcher → stage 分解一致門）。

    MUST: 校正入力が 0 件なら fail loudly — 黙って素の RTN へ落ちる分岐は作らない
    （「校正付きのつもりで校正なしを配った」が資産からは読めない）。
    """
    if not inputs:
        raise AssertionError("校正入力が 1 件も無い（校正付き i4 は入力ゼロでは成立しない）")
    batches = capture_stage_batches(wrapper, inputs)
    assert_stage_split(wrapper, inputs[0], batches[0], stages)
    return CalibRig(stages=tuple(stages), batches=batches)


def calibrate_i4(rig: CalibRig, include: Callable[[str], bool]) -> tuple[CalibReport, Int4Report]:
    """stage 逐次の GPTQ を当て、`(レポート, scale 台帳)` を返す（丸めは in-place）。

    `include` は **stage 内の局所モジュール FQN**（`0.attention.self.query_proj`）の述語
    （core の `iter_quant_targets` の契約）。

    MUST: 台帳の無いレポートは fail loudly — 出荷経路を持つのは `gptq` × `rtn` だけで
    （`karume.quant_calib` の MUST）、台帳が無いまま進むと i4 席に scale の無い重みが載る。
    """
    report = calibrate_stages(
        rig.stages, rig.batches, method=CALIB_METHOD, spec=CALIB_GRID, include=include
    )
    ledger = report.int4
    if ledger is None:
        raise AssertionError(
            f"校正が scale 台帳を返さなかった（{report.describe()}）"
            " — 出荷できるのは gptq × rtn だけ（karume.quant_calib の MUST）"
        )
    return report, ledger
