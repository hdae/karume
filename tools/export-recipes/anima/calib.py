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
4. **校正入力とグラフの経路一致門** — {@link assert_calib_batches_match_graph}。捕捉は素の
   `CosmosTransformer3DModel` の forward から採るのに、丸めるのはラッパ（`patch.AnimaDit`）の
   重み。両者が block へ渡す付随引数の顔ぶれが違うと、「配布グラフとは別の条件で選んだ
   丸め先」を出荷することになる（数値は普通に出るので golden も緑のまま通る）。

NOTE（`anima.measure_quant` のリグと共有しない理由）: あちらは 3 グリッド × 品質計測のための
足場で、対象も駆動も「素のモデル」に閉じている。export 側はラッパの FQN 空間・配布条件
（stage 外を先に丸める）・出荷可能な台帳の取り出しが要る。写しを 1 本にまとめると、片方の
前提が動いたときにもう片方が黙って追随する — 各リグが自前の一致門で独立に自己検証する形に
してある。
"""

from __future__ import annotations

import gc
import inspect
from collections.abc import Sequence
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

#: stage 列のモデル内 FQN 接頭辞。`patch.AnimaDit.model` / `patch.AnimaDitTokens.model` が
#: `CosmosTransformer3DModel` なので、block の FQN は `model.transformer_blocks.<番号>.…`。
STAGE_PREFIX = "model.transformer_blocks"

#: 校正付き丸めの方式。GPTQ 以外は scale 台帳を返さない（`karume.quant_calib` の MUST）。
CALIB_METHOD = "gptq"

#: 丸め先の格納グリッド。**`rtn` × g32 = 既存 i4 系列と同じ格子**（唯一の出荷経路）。
CALIB_GRID = GridSpec(kind="rtn", group_size=DEFAULT_GROUP_SIZE)

#: 校正の参照 denoise の step 数。**配布形の既定**（`models/karume-anima-turbo/karume.json` の
#: `defaults.steps`）に合わせる — 校正で見る sigma 列を実運用の列そのものにするため。
CALIB_STEPS = 8

#: 校正の解像度（px・正方）。波 J-2 の実測条件と同じ（`docs/research/2026-08-20-…` §6）で、
#: **`--resolution` には追随しない** — 品質裁定（f32 とほぼ同一）が採られた条件を固定して
#: おかないと、解像度を動かすたびに根拠の採り直しになる。追随しない代わりに **CLI が食い違いを
#: 拒否する**（`anima.export.main` — 別解像度のグラフへ 512px で選んだ丸め先を焼かないため）。
#: 1 プロンプトあたりのトークン数 = GPTQ の `H = Σ XᵀX` の標本数なので、上げれば校正は効くが
#: export の CPU 時間が線形に伸びる（本数と並ぶ品質の上振れ軸 — `calib_prompts` の NOTE）。
CALIB_RESOLUTION = 512

#: CFG 係数。Turbo LoRA は CFG=1 運用（`reference_steps` が uncond 分岐を計算しない）—
#: 配布条件と揃えることが決定性（乱数ゼロ・分岐なし）にもそのまま効く。
CALIB_GUIDANCE = 1.0

#: 校正 1 バッチあたりの丸め所要（秒・CPU）。波 J-2 の実測（512px・1 プロンプト × 10 step =
#: 10 バッチで gptq 933 秒 — `docs/research/2026-08-20-gptq-awq-calibrated-rounding.md` の
#: anima 節）をバッチ数で割った**線形外挿の係数**で、開始時に見積りを 1 行出すためだけに使う
#: （判断の根拠にはしない — 実機と stage 数が変われば当然ずれる）。
CALIB_SECONDS_PER_BATCH = 93

#: トークナイザの `max_length`（パイプライン既定 — `pipeline_ref` の `--max-sequence-length`）。
CALIB_MAX_SEQUENCE_LENGTH = 512

#: 校正入力を作るときのテキスト前段の格納 dtype。**配布条件に合わせる** — 配布形の
#: `text_encoder` / `text_conditioner` は F16（`anima.distribution.STORAGE_REQUIREMENTS`）で、
#: 校正は「配布で実際に流れる活性」から丸め先を選ぶ機構だから、前段の丸めも揃える
#: （block 外を先に RTN で丸めるのと同じ理由）。波 J-2 の計測は f32 前段だったが、f16 と f32 の
#: 差は DiT 出力で相対 RMS 3e-4 級（2026-08-21 の golden 突合）で、H = Σ XᵀX の統計には出ない。
CALIB_TEXT_DTYPE = "f16"


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
    resolution: int = CALIB_RESOLUTION,
    steps: int = CALIB_STEPS,
) -> tuple[StageBatch, ...]:
    """校正プロンプトごとに参照 denoise を回し、先頭 block への入力を step 横断で捕まえる。

    経路は参照フィクスチャ（`anima.pipeline_ref`）の使い回し — テキスト前段
    （{@link pipeline_ref.load_text_stack} / {@link pipeline_ref.encode_prompt}）と sigma 列と
    `reference_steps` をそのまま通す。写しを作らないのは「校正で見る活性」と「配布で流れる
    活性」を 1 本の実装のままにするため。`timesteps_proj_table` だけは使わない — あれは DiT を
    **もう 1 体ロードする**入口で、ここには既に丸める対象の実体がある（timestep 埋め込みは
    `reference_steps` が素の forward の中で組む）。

    決定的であること MUST: seed 固定（`pipeline_ref.SEED`）・CFG 1（uncond 分岐なし）・
    乱数はここの `latents_init` 1 本きり。初期ノイズは**全プロンプト共通**（波 J-2 の条件を
    そのまま広げた形 — 散らす軸はプロンプトと解像度に閉じる）。

    MUST: 呼び出し側は**block 内の丸めを 1 本も当てる前**に呼ぶ（当てた後だと、丸めた重みが
    作った活性から同じ重みの丸め先を選ぶ循環になる）。
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
    # テキスト前段（Qwen3 + conditioner）は条件を作り終えたら手放す — DiT の常駐と重ねない。
    del stack
    gc.collect()

    sigmas = pipeline_ref.sigma_schedule(steps, pipeline_ref.SHIFT)
    latent = resolution // pipeline_ref.SPATIAL_COMPRESSION
    latents_init = torch.randn(
        (1, pipeline_ref.LATENT_CHANNELS, 1, latent, latent),
        generator=torch.Generator().manual_seed(pipeline_ref.SEED),
        dtype=torch.float32,
    )
    blocks = model.transformer_blocks
    names = tuple(inspect.signature(blocks[0].forward).parameters)
    batches: list[StageBatch] = []

    def catch(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        batches.append(((args[0].detach(),), _block_kwargs(names, args, kwargs)))

    handle = blocks[0].register_forward_pre_hook(catch, with_kwargs=True)
    try:
        for index, embed in enumerate(embeds):
            print(f"[calib] プロンプト {index + 1}/{len(embeds)} の参照 denoise", flush=True)
            pipeline_ref.reference_steps(
                model,
                latents_init,
                embed,
                embed,
                sigmas,
                (resolution, resolution),
                steps,
                CALIB_GUIDANCE,
            )
    finally:
        handle.remove()
    expected = len(prompts) * steps
    if len(batches) != expected:
        raise AssertionError(
            f"校正バッチが {len(batches)} 本で、プロンプト {len(prompts)} × step {steps}"
            f" = {expected} 本と違う（DiT の block ループが台本の想定と食い違っている）"
        )
    return tuple(batches)


def assert_calib_batches_match_graph(
    graph_batch: StageBatch, batches: Sequence[StageBatch]
) -> None:
    """校正入力が**グラフと同じ顔ぶれの付随引数**で block へ入ることを見る。

    捕捉は素の `CosmosTransformer3DModel` の forward から採るのに、丸めるのは export する
    ラッパの重み。両者が block へ渡す keyword の集合が違えば、「配布グラフとは別の条件で
    選んだ丸め先」を出荷することになる — しかも数値は普通に出る（golden も緑のまま通る）。

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


def calibrate_i4(rig: CalibRig) -> tuple[CalibReport, Int4Report]:
    """stage 逐次の GPTQ を当て、`(レポート, scale 台帳)` を返す（丸めは in-place）。

    `include` を渡さないのは、**stage 内の `nn.Linear` は 1 本残らず i4 適格**であることを
    呼び出し側が先に門で確かめているから（適格でない 1 本だけ外す形にすると「走査の本数 =
    丸めた本数」の門が張れなくなる — `anima.measure_quant.scan_calib_targets` と同じ判断）。

    MUST: 台帳の無いレポートは fail loudly — 出荷経路を持つのは `gptq` × `rtn` だけで
    （`karume.quant_calib` の MUST）、台帳が無いまま進むと i4 席に scale の無い重みが載る。
    """
    print(
        f"[calib] GPTQ 校正を開始 — stage {len(rig.stages)} × バッチ {len(rig.batches)} 本"
        f"（見積り約 {len(rig.batches) * CALIB_SECONDS_PER_BATCH / 60:.0f} 分"
        " — 波 J-2 実測からの線形外挿）",
        flush=True,
    )
    handles = _announce_stages(rig.stages)
    try:
        report = calibrate_stages(rig.stages, rig.batches, method=CALIB_METHOD, spec=CALIB_GRID)
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
    """校正が丸めた層が stage の走査と**過不足なく**一致することを見る。

    `scan` は `.weight` を落としたモジュール名の集合（{@link stage_linear_names} の形）。

    MUST: fail loudly。stage の綴りや対象型が変わって block の一部が校正に載らなくなっても、
    丸め漏れのぶん品質は**良い側**に出る（素通りを数字から読めない）。しかも漏れた重みは
    その後 i8 側にも i4 側にも入らないまま格納指定だけが i4 を要求する形になる。
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
