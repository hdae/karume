"""anima の i4 系列の**校正付き丸め**（GPTQ × RTN 格子）を回すリグ（perf-ledger Q-6）。

`karume.quant_calib.calibrate_stages` は「stage を 1 つずつ進めながら、その層に実際に流れる
活性から丸め先を選び直す」駆動で、格納グリッドは {@link karume.quantize.quantize_to_int4} と
**1 バイトも変わらない**（`scale = group amax / 7`・`q ∈ [−7,+7]`）。変わるのは丸め値と
scale 台帳の中身だけで、`emit` 側の格納経路も配布形のバイト構造も一切動かない。

ここが用意するのは core の駆動が要求する 3 つと、anima 固有の門 1 つ:

1. **stage 列** — DiT の `transformer_blocks` をそのまま並べたもの（{@link dit_stages}）。
   `CosmosTransformerBlock.forward` が「hidden を位置引数・残りを keyword で受けて hidden を
   返す」形なので**包まない**（包み直すと写しが上流とずれうる）。接頭辞は
   **export するラッパの FQN 空間**（{@link STAGE_PREFIX} = `model.transformer_blocks`）—
   計測リグ（`anima.measure_quant`）が素の `CosmosTransformer3DModel` を駆動して
   `transformer_blocks.<i>` を使うのに対し、export は `patch.AnimaDit` を通すので `model.` が
   1 段挟まる。ここを外すと scale 台帳のキーが safetensors のテンソルキーと空振りする。
2. **先頭 stage への入力** — {@link capture_stage_batches} が**参照 denoise を実際に回して**
   先頭 block への `(args, kwargs)` を forward_pre_hook で捕まえる。自前で組み直さないのは
   deberta / 計測リグと同文で、拡散モデルでは加えて「活性が sigma で動く」ので**step を
   横断して**採る（1 step だけ見ると後半 step の分布が校正から漏れる）。
3. **stage 分解一致門** — {@link assert_stage_split}。deberta の同名の門は「`EncoderStage` と
   いう写しが上流のループとずれていないか」を見るが、anima は block をそのまま stage にして
   いるのでその形では degenerate になる。代わりに「ラッパの block ループを stage 逐次で
   回した結果が、ラッパを 1 回 forward したときの `norm_out` への入力とビット一致する」を
   見る（丸めを 1 本も当てる前に実測する MUST）。
4. **校正入力とグラフの付随引数一致門** — {@link assert_calib_batches_match_graph}。捕捉は素の
   `CosmosTransformer3DModel` の forward から採るのに、丸めるのはラッパ（`patch.AnimaDit`）の
   重み。両者が block へ渡す付随引数の顔ぶれが違うと、「配布グラフとは別の条件で選んだ
   丸め先」を出荷することになる（数値は普通に出るので golden も緑のまま通る）。見るのは
   **keyword 名の集合だけ**で、値・shape・位置引数は突き合わせない（「経路が同じ」までは
   主張しない — 顔ぶれのずれだけを捕まえる門）。

NOTE（`anima.measure_quant` のリグと共有しない理由）: あちらは 3 グリッド × 品質計測のための
足場で、対象も駆動も「素のモデル」に閉じている。export 側はラッパの FQN 空間・配布条件
（stage 外を先に丸める）・出荷可能な台帳の取り出しが要る。写しを 1 本にまとめると、片方の
前提が動いたときにもう片方が黙って追随する — 各リグが自前の一致門で独立に自己検証する形に
してある。
"""

from __future__ import annotations

import gc
import inspect
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from karume.quant_calib import (
    CalibReport,
    GridSpec,
    StageBatch,
    StageSpec,
    calibrate_stages,
)
from karume.quantize import DEFAULT_GROUP_SIZE, Int4Report, iter_quant_targets

from . import pipeline_ref
from .distribution import anima_model

#: stage 列のモデル内 FQN 接頭辞。`patch.AnimaDit.model` / `patch.AnimaDitTokens.model` が
#: `CosmosTransformer3DModel` なので、block の FQN は `model.transformer_blocks.<番号>.…`。
STAGE_PREFIX = "model.transformer_blocks"

#: 校正付き丸めの方式。GPTQ 以外は scale 台帳を返さない（`karume.quant_calib` の MUST）。
CALIB_METHOD = "gptq"

#: adaLN（modulation の shift / scale / gate を作る層）を指す block 内 FQN セグメント。
#:
#: `CosmosTransformerBlock` は block ごとに `norm1` / `norm2` / `norm3` の
#: `CosmosAdaLayerNormZero` を持ち、各々が `linear_1` / `linear_2` を抱える
#: （anima-v1.0 = 28 block × 3 × 2 = **168 本**。実 checkpoint のヘッダ走査で確認）。
#: 感度実験変種（`anima.export` の `--i4-adaln-i8`）はこの 168 本を i4 から外して i8 で丸める
#: — irodori の w4 席で同型の構成が聴感を回復させた実績があり（research
#: `2026-08-24-gptq-expansion-quality.md` §1）、素版 i4 の視認裁定（同 §5 = 配布スキップ）の
#: 改善候補として名指しされた側。
ADALN_SEGMENTS = frozenset({"norm1", "norm2", "norm3"})

#: 丸め先の格納グリッド。**`rtn` × g32 = 既存 i4 系列と同じ格子**（唯一の出荷経路）。
CALIB_GRID = GridSpec(kind="rtn", group_size=DEFAULT_GROUP_SIZE)

#: 校正の解像度（px・正方）。波 J-2 の実測条件と同じ（`docs/research/2026-08-20-…` §6）で、
#: **`--resolution` には追随しない** — 品質裁定（f32 とほぼ同一）が採られた条件を固定して
#: おかないと、解像度を動かすたびに根拠の採り直しになる。追随しない代わりに **CLI が食い違いを
#: 拒否する**（`anima.export.main` — 別解像度のグラフへ 512px で選んだ丸め先を焼かないため）。
#: 1 プロンプトあたりのトークン数 = GPTQ の `H = Σ XᵀX` の標本数なので、上げれば校正は効くが
#: export の CPU 時間が線形に伸びる（本数と並ぶ品質の上振れ軸 — `calib_prompts` の NOTE）。
CALIB_RESOLUTION = 512

#: 校正 1 バッチあたりの丸め所要（秒・CPU）。**同じ recipe の配布 export 実測**（512px・
#: 4 プロンプト × 8 step = 32 バッチで 2,287 秒 —
#: `docs/research/2026-08-21-anima-i4-seat-speed.md` §6）をバッチ数で割った**線形外挿の係数**で、
#: 開始時に見積りを 1 行出すためだけに使う（判断の根拠にはしない — 実機と stage 数が変われば
#: 当然ずれる）。波 J-2 の別条件実測（10 バッチで 933 秒 = 93 秒/バッチ）より小さいが、
#: そちらは 1 プロンプト × 10 step の測定で条件が違う。
CALIB_SECONDS_PER_BATCH = 72

#: トークナイザの `max_length`（パイプライン既定 — `pipeline_ref` の `--max-sequence-length`）。
CALIB_MAX_SEQUENCE_LENGTH = 512

#: 校正入力を作るときのテキスト前段の格納 dtype。**配布条件に合わせる** — 配布形の
#: `text_encoder` / `text_conditioner` は F16（`anima.distribution.STORAGE_REQUIREMENTS`）で、
#: 校正は「配布で実際に流れる活性」から丸め先を選ぶ機構だから、前段の丸めも揃える
#: （block 外を先に RTN で丸めるのと同じ理由）。波 J-2 の計測は f32 前段だったが、前段の
#: f16 / f32 差が H = Σ XᵀX の統計に出るほどの大きさではないと見ている（**この見立ての実測は
#: research に記録していない** — 条件を戻す / 広げる判断が出たら測り直す）。
CALIB_TEXT_DTYPE = "f16"


@dataclass(frozen=True)
class CalibConditions:
    """参照 denoise の条件のうち**モデルごとに違う**もの（{@link calib_conditions} が導く）。

    定数で持たないのは、turbo（8 step・CFG 1）と素の base 系（20 step・CFG 4）で校正が見る
    sigma 列も分岐の顔ぶれも違うから — 片方を焼き込むと、もう片方は「配布実行時には通らない
    条件」で選んだ丸め先を出荷することになる（格納形も本数も正しいままなので資産からは
    読めない）。解像度・テキスト前段の dtype・トークン長はモデルに依らないので、ここには
    載せずモジュール定数のまま。
    """

    #: 参照 denoise の step 数（配布形の `defaults.steps`）。
    steps: int
    #: CFG 係数（配布形の `defaults.guidanceScale`）。
    guidance: float
    #: uncond 分岐へ通すプロンプト（配布形の `defaults.negativePrompt`）。`guidance == 1.0` では
    #: 1 度も使われない（{@link pipeline_ref.reference_steps} が uncond 分岐を計算しない）。
    negative_prompt: str

    @property
    def branches(self) -> int:
        """1 step で DiT を通る forward の本数（= 1 step が積む校正バッチ数）。

        MUST: 述語は {@link pipeline_ref.reference_steps} の `skip_uncond` と**同じ形**
        （`== 1.0`）。ずれると捕捉バッチ数の検算門が本物の分岐数と食い違い、正しい捕捉を
        落とす / 誤った捕捉を通すのどちらにも倒れる。
        """
        return 1 if self.guidance == 1.0 else 2


def calib_conditions(model: str) -> CalibConditions:
    """配布形の既定（`anima.distribution` の `pipeline_config`）から校正条件を**導く**。

    MUST: 写しを別に持たない。校正条件と配布既定を独立に更新できる形にすると、既定 step を
    動かした日に校正だけが古い sigma 列で回り、その食い違いはどこにも出ない（格納形は 1 バイトも
    変わらず、絵が少し眠くなるだけ）。turbo はここから 8 step / CFG 1 が出て、モジュール定数
    だった頃と 1 ビットも変わらない。
    """
    defaults = anima_model(model).pipeline_config["defaults"]
    return CalibConditions(
        steps=int(defaults["steps"]),
        guidance=float(defaults["guidanceScale"]),
        negative_prompt=str(defaults["negativePrompt"]),
    )


def dit_stages(wrapper: nn.Module) -> tuple[StageSpec, ...]:
    """実行順の DiT block を `(モデル内 FQN 接頭辞, block)` で返す。

    接頭辞に block 番号まで入れてあるので、stage 内の局所 FQN（`attn1.to_q.weight`）は
    そのままラッパ内 FQN（`model.transformer_blocks.0.attn1.to_q.weight`）へ戻る。
    """
    return tuple(
        (f"{STAGE_PREFIX}.{index}", block)
        for index, block in enumerate(wrapper.model.transformer_blocks)
    )


def stage_linear_names(stages: Sequence[StageSpec]) -> frozenset[str]:
    """stage 列が抱える `nn.Linear` のラッパ内 FQN（`.weight` を落とした**モジュール名**）。

    対象選択は core の `iter_quant_targets` の共有 — 「校正が丸める対象」と「呼び出し側が
    stage 内と数える対象」を写した別実装にすると、どちらにも入らない重みが黙って i8 へ落ちる。
    """
    return frozenset(
        f"{prefix}.{local}".removesuffix(".weight")
        for prefix, stage in stages
        for local, _weight, _axis in iter_quant_targets(stage, (nn.Linear,))
    )


def is_adaln(name: str) -> bool:
    """モジュール FQN が adaLN 配下か（{@link ADALN_SEGMENTS} を**セグメントとして**含むか）。

    判定を `.` で割った要素の一致で採るのは、部分文字列一致だと `norm_out` の綴りに引っ張られる
    ような取り違えを招き、逆に接尾辞一致だと adaLN 配下の子（`norm1.linear_1`）を取りこぼす
    ため。stage 内の局所 FQN（`norm1.linear_1`）とラッパ内 FQN
    （`model.transformer_blocks.0.norm1.linear_1`）のどちらにも同じ答えを返すので、校正の
    `include` と export 側の集合分割が 1 実装で決まる（写すと i4 に丸めた集合と i4 で格納する
    集合が割れる）。

    NOTE: 判定は **block の中でだけ**使う。DiT には block の外にも `norm_out`
    （`CosmosAdaLayerNorm` = 同じ modulation の役割）が居るが、あちらは
    {@link anima.export.NON_STAGE_I4_WEIGHTS} が名指しで持つ側で、セグメントの綴りも違う。
    """
    return not ADALN_SEGMENTS.isdisjoint(name.split("."))


def adaln_segments_seen(names: Iterable[str]) -> frozenset[str]:
    """`names` の中に**セグメントとして**現れた {@link ADALN_SEGMENTS} の綴り。

    {@link is_adaln} が「どれか 1 つでも当たったか」を返すのに対し、こちらは**どれが当たったか**
    を返す。両方を数える理由は、i8 へ戻す 168 本が 3 セグメントの**和**で決まるから — 上流が
    一部（例: `norm3`）だけを改名すると、`is_adaln` は残る綴りで真を返し続けるので「adaLN が
    1 本も無い」だけを見る門は素通りし、本数の門も同じ分類器から両辺を作るので自己整合した
    まま緑になる（= 除外したはずの 56 本だけが黙って i4 へ戻る）。
    """
    seen: set[str] = set()
    for name in names:
        seen.update(ADALN_SEGMENTS.intersection(name.split(".")))
    return frozenset(seen)


def _on_cpu(value: Any) -> Any:
    """捕捉した付随引数を CPU へ落とす（Tensor と**素の** tuple / list の中身だけ）。

    組の中まで見るのは、block が rope を `(cos, sin)` の組で受けるから — 素通しにすると
    `--calib-device cuda` で校正バッチの一部だけが VRAM に残り続ける。戻す側（stage 実行直前の
    デバイス移動）は core の `karume.quant_calib` が同じ形で持つ。CPU 経路では `Tensor.cpu()` が
    self を返すので、既定の捕捉は 1bit も変わらない。
    """
    if isinstance(value, torch.Tensor):
        return value.cpu()
    if type(value) is tuple:
        return tuple(_on_cpu(item) for item in value)
    if type(value) is list:
        return [_on_cpu(item) for item in value]
    return value


def _block_kwargs(
    names: Sequence[str], args: Sequence[Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """block への位置引数を**シグネチャの名前**で keyword へ畳む（hidden は除く）。

    名前を写した並びで持たないのは、上流が引数の順序を変えたときに綴りが黙って入れ替わる
    のを避けるため（`anima.measure_quant.capture_stage_batches` と同じ手）。
    """
    return {**dict(zip(names[1:], args[1:], strict=False)), **kwargs}


def assert_stage_split(
    wrapper: nn.Module, probe: Sequence[torch.Tensor], stages: Sequence[StageSpec]
) -> StageBatch:
    """stage 逐次の block ループが**ラッパ 1 回の forward とビット一致**することを見る。

    ラッパの出力は unpatchify（静的形）や `proj_out`（S 形）まで進んでいて block 列の出力
    そのものではないので、突合先は **`norm_out` への入力**を forward_pre_hook で採る。
    これで「ラッパが block をどう呼んでいるか」と「校正が block をどう呼ぶか」が同じである
    ことが、丸める前に実測で固定される。

    MUST: 近似 tolerance ではなく `torch.equal` で見る — 同じモジュールを同じ順で呼んで
    いる以上、一致しないなら分解が本物と違う経路を通っている。

    戻り値はラッパが先頭 block へ渡した `(args, kwargs)`（{@link
    assert_calib_batches_match_graph} の基準）。
    """
    model = wrapper.model
    blocks = model.transformer_blocks
    names = tuple(inspect.signature(blocks[0].forward).parameters)
    captured: list[StageBatch] = []
    tail: list[torch.Tensor] = []

    def catch_block(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        captured.append(((args[0].detach(),), _block_kwargs(names, args, kwargs)))

    def catch_tail(_module: nn.Module, args: tuple[Any, ...]) -> None:
        tail.append(args[0].detach())

    handles = [
        blocks[0].register_forward_pre_hook(catch_block, with_kwargs=True),
        model.norm_out.register_forward_pre_hook(catch_tail),
    ]
    try:
        with torch.no_grad():
            wrapper(*probe)
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != 1 or len(tail) != 1:
        raise AssertionError(
            f"ラッパ 1 回の forward で先頭 block {len(captured)} 回 / norm_out {len(tail)} 回"
            "（どちらも 1 回が想定 — ラッパの綴りが台本の想定と食い違っている）"
        )
    args, kwargs = captured[0]
    with torch.no_grad():
        hidden = args[0]
        for _prefix, stage in stages:
            hidden = stage(hidden, **kwargs)
    if not torch.equal(hidden, tail[0]):
        raise AssertionError(
            "stage 分解の最終 hidden がラッパの block ループの出力とビット一致しない"
            f"（最大絶対差 {float((hidden - tail[0]).abs().max()):.4e}）"
            " — stage 列がラッパの経路とずれている"
        )
    return captured[0]


def capture_stage_batches(
    model: nn.Module,
    prompts: Sequence[str],
    *,
    repo: str,
    conditions: CalibConditions,
    resolution: int = CALIB_RESOLUTION,
) -> tuple[StageBatch, ...]:
    """校正プロンプトごとに参照 denoise を回し、先頭 block への入力を step 横断で捕まえる。

    経路は参照フィクスチャ（`anima.pipeline_ref`）の使い回し — テキスト前段
    （{@link pipeline_ref.load_text_stack} / {@link pipeline_ref.encode_prompt}）と sigma 列と
    `reference_steps` をそのまま通す。写しを作らないのは「校正で見る活性」と「配布で流れる
    活性」を 1 本の実装のままにするため。`timesteps_proj_table` だけは使わない — あれは DiT を
    **もう 1 体ロードする**入口で、ここには既に丸める対象の実体がある（timestep 埋め込みは
    `reference_steps` が素の forward の中で組む）。

    MUST: **CFG > 1 では cond / uncond の両分岐を採る**。素の base 系は CFG 運用なので、配布
    実行時に DiT を通る活性は cond だけではなく negative prompt 側も同数ある — cond に絞ると
    「実際に流れる活性の半分」で丸め先を選ぶことになる。`reference_steps` は
    `guidance != 1.0` のとき 1 step で forward を 2 回回すので、先頭 block の
    forward_pre_hook が**両方をそのまま**採る（分岐をここで組み立てない）。検算門も
    {@link CalibConditions.branches} 倍で数える。

    決定的であること MUST: seed 固定（`pipeline_ref.SEED`）・乱数はここの `latents_init`
    1 本きり（CFG 分岐は乱数を持たない）。初期ノイズは**全プロンプト共通**（波 J-2 の条件を
    そのまま広げた形 — 散らす軸はプロンプトと解像度に閉じる）。

    MUST: 呼び出し側は**block 内の丸めを 1 本も当てる前**に呼ぶ（当てた後だと、丸めた重みが
    作った活性から同じ重みの丸め先を選ぶ循環になる）。

    NOTE（デバイス）: 参照 denoise は `model` が居るデバイスで回す（`--calib-device cuda` の
    感度実験経路では DiT が GPU に居る）。**テキスト前段は CPU のまま**動かす — Qwen3 +
    conditioner と f32 の DiT は同じ VRAM に同居できないので、GPU へ運ぶのは条件が出来上がった
    後の `embeds` / `negative` / `latents_init` だけ。初期ノイズは **CPU で生成してから移す**
    （`torch.randn` の乱数列はデバイスごとに別物なので、生成をデバイス側に寄せると CPU 経路と
    ビットが割れる）。捕捉したバッチは常に CPU へ落として持つ（校正バッチの置き場は CPU 一択 —
    stage 実行時の運搬は `karume.quant_calib` の JIT 移動が担う）。
    """
    if not prompts:
        raise AssertionError("校正プロンプトが 1 件も無い（校正付き i4 は入力ゼロでは成立しない）")
    stack = pipeline_ref.load_text_stack(repo, CALIB_TEXT_DTYPE)
    embeds = [
        pipeline_ref.encode_prompt(stack, CALIB_MAX_SEQUENCE_LENGTH, prompt)[
            "encoder_hidden_states"
        ]
        for prompt in prompts
    ]
    # uncond 分岐のプロンプトは CFG=1 では 1 度も使われないが、分岐を作らず常に用意する —
    # 「使う / 使わない」の判断は `reference_steps` の MUST 1 箇所に閉じる（2 箇所で持つと、
    # 片方だけが CFG の扱いを変えた日に cond と同じ埋め込みで uncond を回すことになり、
    # バッチ本数だけが正しいまま H に同じ活性が 2 倍積まれる）。
    negative = pipeline_ref.encode_prompt(
        stack, CALIB_MAX_SEQUENCE_LENGTH, conditions.negative_prompt
    )["encoder_hidden_states"]
    # テキスト前段（Qwen3 + conditioner）は条件を作り終えたら手放す — DiT の常駐と重ねない。
    del stack
    gc.collect()

    sigmas = pipeline_ref.sigma_schedule(conditions.steps, pipeline_ref.SHIFT)
    latent = resolution // pipeline_ref.SPATIAL_COMPRESSION
    latents_init = torch.randn(
        (1, pipeline_ref.LATENT_CHANNELS, 1, latent, latent),
        generator=torch.Generator().manual_seed(pipeline_ref.SEED),
        dtype=torch.float32,
    )
    device = next(model.parameters()).device
    embeds = [embed.to(device) for embed in embeds]
    negative = negative.to(device)
    latents_init = latents_init.to(device)
    blocks = model.transformer_blocks
    names = tuple(inspect.signature(blocks[0].forward).parameters)
    batches: list[StageBatch] = []

    def catch(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        # 捕捉は CPU へ落として溜める（バッチ数 × hidden を VRAM に常駐させない）。CPU 経路では
        # `Tensor.cpu()` が self を返すので、既定の捕捉は 1bit も変わらない。
        captured = _block_kwargs(names, args, kwargs)
        batches.append(
            (
                (args[0].detach().cpu(),),
                {name: _on_cpu(value) for name, value in captured.items()},
            )
        )

    handle = blocks[0].register_forward_pre_hook(catch, with_kwargs=True)
    try:
        for index, embed in enumerate(embeds):
            print(f"[calib] プロンプト {index + 1}/{len(embeds)} の参照 denoise", flush=True)
            pipeline_ref.reference_steps(
                model,
                latents_init,
                embed,
                negative,
                sigmas,
                (resolution, resolution),
                conditions.steps,
                conditions.guidance,
            )
    finally:
        handle.remove()
    expected = len(prompts) * conditions.steps * conditions.branches
    if len(batches) != expected:
        raise AssertionError(
            f"校正バッチが {len(batches)} 本で、プロンプト {len(prompts)}"
            f" × step {conditions.steps} × 分岐 {conditions.branches}"
            f"（CFG {conditions.guidance}）= {expected} 本と違う"
            "（DiT の block ループが台本の想定と食い違っている）"
        )
    return tuple(batches)


def assert_calib_batches_match_graph(
    graph_batch: StageBatch, batches: Sequence[StageBatch]
) -> None:
    """校正入力が**グラフと同じ顔ぶれの付随引数**で block へ入ることを見る。

    捕捉は素の `CosmosTransformer3DModel` の forward から採るのに、丸めるのは export する
    ラッパの重み。両者が block へ渡す keyword の集合が違えば、「配布グラフとは別の条件で
    選んだ丸め先」を出荷することになる — しかも数値は普通に出る（golden も緑のまま通る）。

    見るのは **keyword 名の集合**だけ。値・shape・位置引数は突き合わせないので、同じ顔ぶれの
    まま中身が違う入力（例: 別の sigma で採った temb）は素通りする — この門が主張するのは
    「引数の顔ぶれが同じ」までで、「経路が同じ」ではない。

    MUST: fail loudly。ここが黙ると、上流が block の引数を増やした日に校正だけがそれを見る
    （あるいは見ない）状態が資産から読めない形で入り込む。
    """
    wanted = set(graph_batch[1])
    for index, (_args, kwargs) in enumerate(batches):
        if set(kwargs) != wanted:
            raise AssertionError(
                f"校正バッチ {index} の block 付随引数 {sorted(kwargs)} が"
                f" グラフの {sorted(wanted)} と違う — 参照 denoise とラッパで block の呼び方が違う"
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


def _announce_stages(stages: Sequence[StageSpec]) -> list[RemovableHandle]:
    """各 stage の**初回** forward で進捗を 1 行出す hook を張る（戻り値は外す用の handle）。

    `karume.quant_calib` は 1 行も印字しないので、既定運用の校正は開始行から最終診断行まで
    時間単位で無出力になり、ハングと区別がつかない。core は境界（ADR 0065）なので触らず、
    呼び出し側から覗く。初回だけに絞るのは、stage が「観測 × バッチ数 + 前進 × バッチ数」回
    呼ばれるため — 出る行数は stage 数と同じで、待ち時間がその粒度で割れる。
    """
    total = len(stages)

    def announce(index: int):
        seen = False

        def hook(_module: nn.Module, _args: tuple[Any, ...]) -> None:
            nonlocal seen
            if seen:
                return
            seen = True
            print(f"[calib] stage {index + 1}/{total}", flush=True)

        return hook

    return [
        stage.register_forward_pre_hook(announce(index))
        for index, (_prefix, stage) in enumerate(stages)
    ]


def calibrate_i4(
    rig: CalibRig, include: Callable[[str], bool] | None = None
) -> tuple[CalibReport, Int4Report]:
    """stage 逐次の GPTQ を当て、`(レポート, scale 台帳)` を返す（丸めは in-place）。

    `include` は **stage 内の局所モジュール FQN** の述語で、core の `calibrate_stages` へ
    そのまま渡る（`None` = stage 内の `nn.Linear` 全部 = 既定の配布経路）。感度実験変種
    （`anima.export` の `--i4-adaln-i8`）だけが述語を渡し、adaLN 168 本を GPTQ から外して
    i8 側へ回す。

    MUST: 渡した述語が選ぶ集合と i4 で格納する集合は**同一**であること — ずれると「走査の
    本数 = 丸めた本数」の門が意味を失い、丸めていない重みに i4 の格納指定が付く。実測は
    {@link assert_calib_covers_scan}（呼び出し側が i4 格納の集合を渡して突き合わせる）。

    MUST: 台帳の無いレポートは fail loudly — 出荷経路を持つのは `gptq` × `rtn` だけで
    （`karume.quant_calib` の MUST）、台帳が無いまま進むと i4 席に scale の無い重みが載る。
    """
    # デバイスは stage から読む（core が計算デバイスを導くのと同じ源 — 印字だけ別の綴りから
    # 作ると、実際に回った場所と診断行が食い違いうる）。見積りの係数は CPU 実測なので、
    # `device=cuda` の行では桁が合わない（そちらは感度実験専用の経路）。
    device = next(rig.stages[0][1].parameters()).device
    print(
        f"[calib] GPTQ 校正を開始（device={device.type}）"
        f" — stage {len(rig.stages)} × バッチ {len(rig.batches)} 本"
        f"（見積り約 {len(rig.batches) * CALIB_SECONDS_PER_BATCH / 60:.0f} 分"
        " — 波 J-2 実測からの線形外挿）",
        flush=True,
    )
    handles = _announce_stages(rig.stages)
    try:
        report = calibrate_stages(
            rig.stages, rig.batches, method=CALIB_METHOD, spec=CALIB_GRID, include=include
        )
    finally:
        for handle in handles:
            handle.remove()
    ledger = report.int4
    if ledger is None:
        raise AssertionError(
            f"校正が scale 台帳を返さなかった（{report.describe()}）"
            " — 出荷できるのは gptq × rtn だけ（karume.quant_calib の MUST）"
        )
    return report, ledger


def assert_calib_covers_scan(report: CalibReport, scan: frozenset[str]) -> None:
    """校正が丸めた層が **i4 で格納する block 内の集合**と過不足なく一致することを見る。

    `scan` は `.weight` を落としたモジュール名の集合（{@link stage_linear_names} の形。既定は
    stage の走査そのもの・感度実験変種では adaLN を除いた側 = {@link calibrate_i4} の
    `include` が選ぶ集合と同一）。

    MUST: fail loudly。丸め漏れは品質を**良い側**へ動かすので、素通りを数字から読めない。
    しかも漏れた重みはその後 i8 側にも i4 側にも入らないまま、格納指定だけが i4 を要求する
    形になる。

    捕まえられるのは「**校正の実走が走査からずれた**」形（core が対象を落とした / 別の
    `stages` を渡した / 台帳の取り出しが層を落とした）まで。走査そのもの（`iter_quant_targets`
    の対象型・stage 分解）が変わった場合は**両辺が一緒に動くのでこの門は黙る** — そちらは
    `export.py` 側の「走査 == i4 適格集合」の突合（グラフ由来の独立な源）が受け持つ。
    """
    rounded = {layer.fqn for layer in report.layers}
    wanted = {f"{name}.weight" for name in scan}
    missing = sorted(wanted - rounded)
    extra = sorted(rounded - wanted)
    if missing or extra or report.modules != len(wanted):
        raise AssertionError(
            f"校正が丸めた {report.modules} 本が走査の {len(wanted)} 本と一致しない"
            f"（丸め漏れ {missing[:3]} / 走査に無い {extra[:3]}）"
        )
