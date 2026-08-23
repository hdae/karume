"""Irodori の i4 系列の**校正付き丸め**（GPTQ × RTN 格子）を回すリグ（perf-ledger Q-2 / Q-3）。

`karume.quant_calib.calibrate_stages` は「stage を 1 つずつ進めながら、その層に実際に流れる
活性から丸め先を選び直す」駆動で、格納グリッドは {@link karume.quantize.quantize_to_int4} と
**1 バイトも変わらない**（`scale = group amax / 7`・`q ∈ [−7,+7]`）。変わるのは丸め値と
scale 台帳の中身だけで、emit 側の格納経路も配布形のバイト構造も一切動かない。

ここが用意するのは core の駆動が要求する 3 つと、Irodori 固有の門 2 つ:

1. **stage 列** — DiT の `blocks` をそのまま並べたもの（{@link dit_stages}）。上流の
   `DiffusionBlock.forward` が「hidden を第 1 位置引数で受けて hidden を返す」形なので
   **包まない**。接頭辞 {@link STAGE_PREFIX} は `blocks` で、これは
   **配布ラッパ（`irodori.export.DitGraph`）の FQN 空間でもある** — ラッパは DiT の部分木を
   `self.blocks` / `self.in_proj` / `self.out_proj` / `self.cond_module` と**同じ属性名**で
   抱える（`irodori.export.TARGET_SCALE_SOURCES` の `dit` 行が張り替え無しなのはこのため）。
   ここが外れると scale 台帳のキーが safetensors のテンソルキーと空振りする。
2. **先頭 stage への入力** — {@link capture_stage_batches} が**参照 denoise を実際に回して**
   先頭 block への `(args, kwargs)` を forward_pre_hook で捕まえる。自前で組み直さないのは
   計測リグ（`irodori.measure_quant`）と同文で、拡散モデルでは加えて「活性が t で動く」ので
   **step を横断して**採る。
3. **stage 分解一致門** — {@link assert_stage_split}。「stage 逐次で回した結果が、同じ
   forward で `out_norm` へ入った hidden とビット一致する」を見る（丸めを 1 本も当てる前に
   実測する MUST）。
4. **過不足一致門** — {@link assert_calib_covers_scan}。校正が丸めた層 = stage の走査。

MUST（駆動が**素の DiT** である理由 — この recipe 固有の制約）: `irodori.export` の丸めは
「`load_*` の直後・パッチ前の参照より前」に置かれ（`irodori.export.fake_quant` の順序 MUST）、
一方で配布ラッパ `DitGraph` は**パッチ後でしか動かない**（実数形 RoPE 表を渡すため —
`irodori.patch` のモジュール docstring）。両方は同時に満たせないので、校正の参照ループは
**上流の正本経路**（`irodori_tts.rf.sample_euler_rf_cfg` が素の `TextToLatentRFDiT` を回す形）
で採る。素の経路と `DitGraph` が同じ値を出すことは、**同じ export 実行の中で**
`irodori.export` の eager 同値門（`EAGER_EQUIV_ATOL = 0`）が全 golden ケースについて実測
する — 「stage 逐次 ≡ 素の block ループ」（ここの門）と合わせて、stage 分解が配布グラフの
経路と一致することが 2 本の実測で閉じる。

MUST: 参照ループは `use_context_kv_cache=False` で回す（{@link capture_stage_batches}）—
上流の既定（True）は条件 K/V を**block ごとに別のキャッシュ**で渡すので、1 つの kwargs を
全 stage で使い回す core の駆動と噛み合わず、しかも stage 分解一致門より前に静かに壊れる。
False は配布グラフの綴り（毎 forward 内で `project_context_kv` を計算 — ADR 0047 決定 3）
とも一致する。

NOTE（`irodori.measure_quant` のリグと共有しない理由）: あちらは 3 グリッド × 品質計測のための
足場で、駆動は**パッチ後の `DitGraph`**（`DitBlockStage` という写しを持つ）に閉じている。
export 側はパッチ前の素の経路・配布条件（stage 外を先に丸める）・出荷可能な台帳の取り出しが
要る。写しを 1 本にまとめると、片方の前提が動いたときにもう片方が黙って追随する — 各リグが
自前の一致門で独立に自己検証する形にしてある。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

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
from karume.quantize import DEFAULT_GROUP_SIZE, Int4Report, channel_rows, iter_quant_targets

from . import export as ex
from . import pipeline_ref as ip

#: stage 列の FQN 接頭辞（`TextToLatentRFDiT.blocks` — 配布ラッパの属性名と同じ）。
STAGE_PREFIX = "blocks"

#: 校正付き丸めの方式。GPTQ 以外は scale 台帳を返さない（`karume.quant_calib` の MUST）。
CALIB_METHOD = "gptq"

#: 丸め先の格納グリッド。**`rtn` × g32 = 既存 i4 格子そのもの**（唯一の出荷経路）で、
#: `irodori.measure_quant.CALIB_CONFIGS` の `gptq-rtn` と同じ指定。
CALIB_GRID = GridSpec(kind="rtn", group_size=DEFAULT_GROUP_SIZE)

#: 校正で捕捉する step 数の既定（1 ケースあたり）。**波 J-2 の計測既定と同じ**
#: （`irodori.measure_quant` の `--calib-steps` 既定 = 参照ループ全長）— 品質裁定が採られた
#: 条件を下回らせないため。`--calib-steps` で下げられるのは smoke 用で、使った値は
#: `calib_provenance.json` に残る。
CALIB_STEPS = ip.NUM_STEPS

#: `DiffusionBlock.forward` の引数のうち**batch 軸を持つ**もの（cond 側 = 行 0 を切り出す）。
#: 上流の independent CFG は cond と uncond を**1 forward へ束ねて**回す（`x_t` を
#: `cfg_batch_mult` 本 cat する）ので、捕捉したバッチの行 0 だけが cond 側になる。
BATCHED_BLOCK_ARGS = frozenset(
    {
        "x",
        "cond_embed",
        "text_state",
        "text_mask",
        "speaker_state",
        "speaker_mask",
        "caption_state",
        "caption_mask",
    }
)

#: 同 forward 引数のうち batch 軸を持たないもの（そのまま次 stage へ渡る）。
#: `context_kv` は {@link capture_stage_batches} の MUST により常に `None`。
UNBATCHED_BLOCK_ARGS = frozenset({"freqs_cis", "self_mask", "context_kv"})

#: hidden を運ぶ引数名（core の駆動は hidden を**位置引数**で渡すので kwargs から抜く）。
HIDDEN_ARG = "x"


class _CalibStepsReached(Exception):  # noqa: N818 — 異常ではなく打ち切りの合図
    """必要な step 数ぶん捕まえた合図（参照ループを途中で畳むための番兵）。"""


@dataclass(frozen=True)
class CalibContext:
    """校正が要る「重み以外」の一切（`irodori.export.export_series` が組んで渡す）。

    トークナイザと条件の作り方は `irodori.pipeline_ref` の関数をそのまま呼ぶ — 写しを作ると
    「校正で見る活性」と「評価で流れる活性」が別の前処理で作られはじめる。
    """

    model_dir: Path
    source: Any
    config: Any
    text_config: Mapping[str, Any]
    model_config: Mapping[str, Any]

    @property
    def text_cap(self) -> int:
        return int(self.model_config["max_text_len"])

    @property
    def caption_cap(self) -> int:
        return int(self.model_config["max_caption_len"])


class CapturedRun(NamedTuple):
    """1 ケースぶんの捕捉。

    `batches` は cond 側（行 0）へ切り出した校正入力、`graph_batch` は**先頭 forward の生**
    （batch 軸そのまま）、`block_loop_output` は**同じ forward** で `out_norm` へ入った
    hidden。後ろの 2 つが {@link assert_stage_split} の突合の両辺で、同じ forward から採る
    ので「別条件どうしを比べている」余地が無い。
    """

    batches: tuple[StageBatch, ...]
    graph_batch: StageBatch
    block_loop_output: torch.Tensor


def dit_stages(dit: nn.Module) -> tuple[StageSpec, ...]:
    """実行順の DiT block を `(モデル内 FQN 接頭辞, block)` で返す。

    接頭辞に block 番号まで入れてあるので、stage 内の局所 FQN（`attention.wq.weight`）は
    そのままモデル内 FQN（`blocks.0.attention.wq.weight`）へ戻る。
    """
    return tuple((f"{STAGE_PREFIX}.{index}", block) for index, block in enumerate(dit.blocks))


def stage_linear_names(stages: Sequence[StageSpec]) -> frozenset[str]:
    """stage 列が抱える `nn.Linear` の FQN（`.weight` を落とした**モジュール名**）。

    対象選択は core の `iter_quant_targets` の共有 — 「校正が丸める対象」と「呼び出し側が
    stage 内と数える対象」を写した別実装にすると、どちらにも入らない重みが黙って f32 のまま
    残る。
    """
    return frozenset(
        f"{prefix}.{local}".removesuffix(".weight")
        for prefix, stage in stages
        for local, _weight, _axis in iter_quant_targets(stage, (nn.Linear,))
    )


def dit_i4_names(dit: nn.Module, sym_max: int) -> frozenset[str]:
    """DiT の i4 適格（= 配布グラフに載る `nn.Linear` 全部）のモジュール FQN。

    適格は 2 条件の積で決まる:

    1. **配布グラフに載る重みであること** — 判定は `irodori.export.DitGraph` の
       `named_parameters` との交差で採る。素の `TextToLatentRFDiT` は backbone / projector /
       speaker / duration の**コピーを内側に持つ**（`irodori.pipeline_ref` の NOTE）ので、
       名前で絞らないと他役割の linear まで i4 に化ける。名指しの一覧
       （`blocks` / `in_proj` / `out_proj` / `cond_module`）を持たないのは、上流が DiT の
       構成を変えた日に一覧だけが古いまま通るのを避けるため。
    2. **量子化軸が group 長で割り切れること** — i4 は端数 group を作らない MUST
       （ADR 0069 決定 2）。実重み（12 block）では linear 317 本すべてが g32 整除だが
       （`docs/research/2026-08-12-irodori-quant-recon.md` の k ヒストグラム）、**前提にせず
       fail loudly** する — 校正は stage を丸ごと駆動するので 1 本だけ外す逃げ道が無く、
       外れた 1 本を黙って i8 へ落とすと「走査の本数 = 丸めた本数」の門が張れなくなる。

    MUST: 適格 0 本は fail loudly（`--dtype i4` を指定したのに i4 が 1 本も無い、を沈黙させ
    ない — ADR 0006 と同じ規律）。
    """
    owned = {
        name.removesuffix(".weight")
        for name, _parameter in ex.DitGraph(dit, sym_max).named_parameters()
    }
    names: set[str] = set()
    for fqn, weight, axis in iter_quant_targets(dit, (nn.Linear,)):
        name = fqn.removesuffix(".weight")
        if name not in owned:
            continue
        length = int(channel_rows(weight, axis).shape[-1])
        if length % DEFAULT_GROUP_SIZE:
            raise AssertionError(
                f"{fqn}: 量子化軸 {length} が g{DEFAULT_GROUP_SIZE} で割り切れない"
                "（校正は stage 単位で駆動するので 1 本だけ外す逃げ道が無い）"
            )
        names.add(name)
    if not names:
        raise AssertionError(
            "DiT に i4 適格な linear が 1 本も無い（DitGraph の構成が台本の想定と食い違っている）"
        )
    return frozenset(names)


def _split_block_kwargs(kwargs: Mapping[str, Any], batch_rows: int) -> StageBatch:
    """block の呼び出し引数を `(hidden 1 本, 残りの kwargs)` へ割り、cond 側の行を切り出す。

    MUST: 引数の顔ぶれを実測で固定する（fail loudly）— 上流が `DiffusionBlock.forward` の
    引数を増減させると、batch 軸の有無の判定が黙って外れて「別条件で選んだ丸め先」を出荷する
    ことになる（数値は普通に出るので golden も緑のまま通る）。

    `batch_rows` は cond 側として残す行数（**常に 1** — 上流の independent CFG は
    `[cond, uncond…]` を 1 forward へ束ね、cond は行 0）。
    """
    wanted = BATCHED_BLOCK_ARGS | UNBATCHED_BLOCK_ARGS
    if set(kwargs) != wanted:
        raise AssertionError(
            f"block の呼び出し引数が {sorted(kwargs)} で、想定の {sorted(wanted)} と違う"
            "（上流 DiffusionBlock.forward の綴りが動いている）"
        )
    if kwargs["context_kv"] is not None:
        raise AssertionError(
            "block へ条件 K/V キャッシュが渡っている"
            "（参照ループは use_context_kv_cache=False で回す MUST — block ごとに別の"
            "キャッシュを 1 つの kwargs で使い回すことになる）"
        )
    sliced = {
        name: (value[:batch_rows] if name in BATCHED_BLOCK_ARGS and value is not None else value)
        for name, value in kwargs.items()
    }
    hidden = sliced.pop(HIDDEN_ARG)
    return ((hidden.detach(),), {name: _detached(value) for name, value in sliced.items()})


def _detached(value: Any) -> Any:
    return value.detach() if isinstance(value, torch.Tensor) else value


def _reference_latent(case: ip.PipelineCase, config: Any) -> torch.Tensor:
    """参照 latent（`irodori.pipeline_ref.run_case` の ③ と同じ作り方）。"""
    if case.reference is None:
        return torch.zeros((1, int(config.speaker_patch_size), int(config.latent_dim)))
    generator = torch.Generator().manual_seed(case.reference.seed)
    return torch.randn(1, case.reference.frames, int(config.latent_dim), generator=generator)


def _load_tokenizers(context: CalibContext) -> tuple[Any, Any]:
    """text 側と caption 側のトークナイザ（`irodori.pipeline_ref` と同じ出どころ）。"""
    from tokenizers import Tokenizer

    return (
        Tokenizer.from_file(str(context.model_dir / ex.TOKENIZER_FILE)),
        ip.upstream_caption_tokenizer(
            context.model_dir, bool(context.config.caption_add_bos_resolved)
        ),
    )


def _capture_case(
    dit: nn.Module,
    case: ip.PipelineCase,
    context: CalibContext,
    tokenizers: tuple[Any, Any],
    steps: int,
    label: str,
) -> CapturedRun:
    """1 ケースの参照 denoise を回し、**step ごとに 1 バッチ**（cond 側）捕まえる。

    上流の independent CFG は 1 step = **1 forward**（cond と uncond を batch へ束ねる）なので、
    先頭 block の呼び出し回数がそのまま step 数になる。行 0 だけを採るのが「cond 側 1 forward
    のみ」の意味で、波 J-2 の計測（`irodori.measure_quant.capture_case_batches`）と同じ
    捕捉セマンティクス。

    `steps` を捕まえ切ったら番兵で参照ループを畳む — 上流ループは残りの step も回すので、
    畳まないと縮小実行が成立しない。
    """
    from irodori_tts.rf import sample_euler_rf_cfg

    tokenizer, caption_tokenizer = tokenizers
    bos_id = int(context.text_config["bos_token_id"])
    pad_id = int(context.text_config["pad_token_id"])
    text_ids = ip._packed_ids(
        tokenizer, case.text, bos_id, context.text_cap, context.source.normalize_text
    )
    caption_padded, caption_mask = ip.upstream_caption_condition(
        caption_tokenizer, case.caption, context.caption_cap
    )
    reference_latent = _reference_latent(case, context.config)
    has_speaker = case.reference is not None
    reference_mask = torch.full(reference_latent.shape[:2], has_speaker, dtype=torch.bool)
    sequence = ip.upstream_sequence_length(
        dit,
        text_ids,
        caption_padded,
        caption_mask,
        reference_latent,
        has_speaker,
        context.text_config,
        context.model_config,
        context.config,
    )
    text_padded, text_mask = ip._right_pad_ids(text_ids, context.text_cap, pad_id)

    batches: list[StageBatch] = []
    graph_batch: list[StageBatch] = []
    tail: list[torch.Tensor] = []

    def on_block(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if args:
            raise AssertionError(
                f"先頭 block が位置引数 {len(args)} 本で呼ばれた"
                "（上流 forward_with_encoded_conditions は全て keyword で渡す）"
            )
        if len(batches) >= steps:
            raise _CalibStepsReached
        if not graph_batch:
            graph_batch.append(
                (
                    (kwargs[HIDDEN_ARG].detach(),),
                    {
                        name: _detached(value)
                        for name, value in kwargs.items()
                        if name != HIDDEN_ARG
                    },
                )
            )
        batches.append(_split_block_kwargs(kwargs, 1))

    def on_tail(_module: nn.Module, args: tuple[Any, ...]) -> None:
        if not tail:
            tail.append(args[0].detach())

    handles = [
        dit.blocks[0].register_forward_pre_hook(on_block, with_kwargs=True),
        dit.out_norm.register_forward_pre_hook(on_tail),
    ]
    try:
        with torch.no_grad():
            sample_euler_rf_cfg(
                model=dit,
                text_input_ids=text_padded,
                text_mask=text_mask,
                ref_latent=reference_latent,
                ref_mask=reference_mask,
                sequence_length=sequence,
                caption_input_ids=caption_padded,
                caption_mask=caption_mask,
                num_steps=ip.NUM_STEPS,
                cfg_scale_text=ip.CFG_SCALES["text"],
                cfg_scale_caption=ip.CFG_SCALES["caption"],
                cfg_scale_speaker=ip.CFG_SCALES["speaker"] if has_speaker else 0.0,
                cfg_guidance_mode="independent",
                cfg_min_t=ip.CFG_MIN_T,
                cfg_max_t=ip.CFG_MAX_T,
                seed=case.seed,
                use_context_kv_cache=False,
            )
    except _CalibStepsReached:
        pass
    finally:
        for handle in handles:
            handle.remove()
    if len(batches) != steps or not tail:
        raise AssertionError(
            f"{case.name}: 校正入力を {len(batches)} step しか捕まえられなかった"
            f"（要求 {steps} step / block ループの尾 {len(tail)} 本）"
            " — 上流 sample_euler_rf_cfg の綴りが台本の想定と食い違っている"
        )
    print(f"[{label}] {case.name}: S={sequence} / {len(batches)} step を捕捉", flush=True)
    return CapturedRun(tuple(batches), graph_batch[0], tail[0])


def capture_stage_batches(
    dit: nn.Module,
    cases: Sequence[ip.PipelineCase],
    context: CalibContext,
    *,
    steps: int = CALIB_STEPS,
    label: str = "calib",
) -> tuple[CapturedRun, ...]:
    """校正ケースごとに参照 denoise を回して先頭 block への入力を集める。

    MUST: 呼び出し側は**block 内の丸めを 1 本も当てる前**に呼ぶ（当てた後だと、丸めた重みが
    作った活性から同じ重みの丸め先を選ぶ循環になる）。逆に **block の外は先に丸めておく**
    — 配布実行時に block へ入るのは i4 の `in_proj` が作った hidden と i4 の `cond_module` が
    作った条件埋め込みで、後に回すと「f32 の周辺で選んだ丸め先」を i4 の周辺と組んで配る
    ことになる（`irodori.export._round_i4_calibrated` の順序 MUST）。
    """
    if not cases:
        raise AssertionError("校正ケースが 1 件も無い（校正付き i4 は入力ゼロでは成立しない）")
    if steps < 1:
        raise AssertionError(f"捕捉 step 数は 1 以上（実測 {steps}）")
    tokenizers = _load_tokenizers(context)
    return tuple(_capture_case(dit, case, context, tokenizers, steps, label) for case in cases)


def assert_stage_split(stages: Sequence[StageSpec], run: CapturedRun) -> None:
    """stage 逐次の block ループが**素の DiT 1 回の forward とビット一致**することを見る。

    突合先は `out_norm` への入力（block ループの出力そのもの）で、両辺とも
    {@link CapturedRun} が**同じ forward** から採ったもの。

    MUST: 近似 tolerance ではなく `torch.equal` で見る — 同じモジュールを同じ順で呼んで
    いる以上、一致しないなら分解が本物と違う経路を通っている。

    MUST: 丸めを 1 本も当てる前に呼ぶ。ずれた分解で丸めると「別の経路の GPTQ」を出荷する
    ことになり、しかも数値は普通に出る（golden も緑のまま通る）。
    """
    args, kwargs = run.graph_batch
    with torch.no_grad():
        hidden = args[0]
        for _prefix, stage in stages:
            hidden = stage(hidden, **kwargs)
    if not torch.equal(hidden, run.block_loop_output):
        raise AssertionError(
            "stage 分解の最終 hidden が素の DiT の block ループの出力とビット一致しない"
            f"（最大絶対差 {float((hidden - run.block_loop_output).abs().max()):.4e}）"
            " — stage 列が上流の経路とずれている"
        )


def _announce_stages(stages: Sequence[StageSpec]) -> list[RemovableHandle]:
    """各 stage の**初回** forward で進捗を 1 行出す hook を張る（戻り値は外す用の handle）。

    `karume.quant_calib` は 1 行も印字しないので、既定運用の校正は開始行から最終診断行まで
    時間単位で無出力になり、ハングと区別がつかない。core は境界（ADR 0065）なので触らず、
    呼び出し側から覗く（`anima.calib._announce_stages` と同じ手）。
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
    stages: Sequence[StageSpec], batches: Sequence[StageBatch]
) -> tuple[CalibReport, Int4Report]:
    """stage 逐次の GPTQ を当て、`(レポート, scale 台帳)` を返す（丸めは in-place）。

    `include` を渡さないのは、**stage 内の `nn.Linear` は 1 本残らず i4 適格**であることを
    呼び出し側が先に門で確かめているから（適格でない 1 本だけ外す形にすると「走査の本数 =
    丸めた本数」の門が張れなくなる — {@link dit_i4_names} と同じ判断）。

    MUST: 台帳の無いレポートは fail loudly — 出荷経路を持つのは `gptq` × `rtn` だけで
    （`karume.quant_calib` の MUST）、台帳が無いまま進むと i4 席に scale の無い重みが載る。
    """
    tokens = sum(int(args[0].shape[1]) for args, _kwargs in batches)
    print(
        f"[calib] GPTQ 校正を開始 — stage {len(stages)} 段 × バッチ {len(batches)} 本"
        f"・hidden 合計 {tokens:,} token",
        flush=True,
    )
    handles = _announce_stages(stages)
    try:
        report = calibrate_stages(stages, batches, method=CALIB_METHOD, spec=CALIB_GRID)
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
