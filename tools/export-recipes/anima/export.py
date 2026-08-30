"""Anima の 4 コンポーネントを IR v1 + golden io へ書き出す台本（ADR 0016 の emit ターゲット）。

起点は diffusers 版 `circlestone-labs/Anima-Base-v1.0-Diffusers`（recon は
docs/research/2026-08-02-anima-recon.md）。素の diffusers モジュールは rank5〜8・feat_cache・
ホストへ出すべき埋め込みを含むので、キュレーションは `anima.patch`
（パッチ層 + ラッパ）が担う。

ターゲットは 4 本（ADR 0016 の全量。融合ターゲットは作らない — コンポーネント間にホスト処理が
挟まるため構造的に不可能）:

- **`text_encoder`** — Qwen3 の `last_hidden_state`。動的 `T`（プロンプト長 ≤ 512）。
- **`text_conditioner`** — `Tsrc`（Qwen3 出力長）/ `Ttgt`（T5 id 列長）の**独立 2 シンボル**。
  ケースは両方の長さを変えて取り、片方の束縛でもう片方を代用する取り違えを数で暴く。
- **`transformer`** — CosmosDiT の 1 denoise step。既定は**解像度固定の静的グラフ**（512px）。
  `--dit-graph dyn` で**トークン長 1 シンボル `S` の追加系列**（#21 波 T2）。
- **`vae_decoder`** — T=1 の画像 decode。**静的**（既定 512px）。

    uv sync --all-groups                                        # tools/ で 1 回
    uv run python -m anima.export --out /path/to/out
    uv run python -m anima.export --target vae_decoder --out /path/to/out
    uv run python -m anima.export --verify vae_decoder
    uv run python -m anima.export --dtype f16   # → outputs/series/anima-f16/
    uv run python -m anima.export --dtype i8    # → …/anima-i8/（DiT のみ）
    uv run python -m anima.export --dtype i4    # → …/anima-i4/（DiT のみ・混成）
    uv run python -m anima.export --dtype f16 --dit-graph dyn  # → …-f16-dyn/

MUST: `--dit-graph dyn` は **transformer 専用の追加系列**で、静的系列を置き換えない
（既存資産・E2E・tolerance を 1 つも動かさないのが波 T2 の前提）。patchify /
unpatchify / rope 表の構築はホストへ出るので、系列ディレクトリには `model.safetensors` と
golden io に加えて **`rope_base.safetensors`**（ホストが rope 表を組むための軸別素表）が並ぶ。

MUST: `--dtype i8` は **transformer 専用**（ADR 0019 の系列設計）。DiT の −1.87GiB が支配項で、
text / cond / VAE は `outputs/series/anima-f16/` を共有する。加えて VAE は「CausalConv3d の時間方向
スライス」をパッチ適用時に行うため、丸めを先に当てると per-channel scale が**捨てられる要素の
amax**まで数えた値になる（f16 の要素ごとの丸めと違い、scale は全要素の値を動かす）。他ターゲット
への `--dtype i8` は CLI が機械的に拒否する。

MUST: `--dtype i4` も **transformer 専用**（i8 と同じ理由・同じ表 — {@link DTYPE_TARGETS}）で、
中身は**混成**（{@link I4_MODULE_TYPES} の適格な重み = i4 group32・残り = i8 per-channel）。
i4 の実行経路は linear / embedding / conv1d の重みスロット限定（ADR 0069 決定 5 と追補）なので、
単一 dtype の i4 系列は原理的に作れない — 系列の**既定**格納は i8 で（{@link BASE_WEIGHT_DTYPES}）、
適格な 1 本ずつを `weight_dtype_overrides` で i4 へ振る（deberta / gemma4 の混成と同形）。

DiT block 内の linear は **GPTQ 校正付きで丸める**（既定 — perf-ledger Q-6 / `anima.calib`）。
格納形は 1 バイトも変わらない（格子は RTN i4 g32 のまま）で、変わるのは丸め値と scale 台帳の
中身だけ。校正入力は {@link anima.calib_prompts.CALIB_PROMPTS} の先頭
`--calib-prompts` 本（既定 {@link DEFAULT_CALIB_PROMPTS} 本）を**参照 denoise**（512²・seed 固定・
step 数と CFG は `--model` の配布既定から導く — {@link anima.calib.calib_conditions}）へ通して
先頭 block の入力を step 横断で捕まえたもの（CFG > 1 では cond / uncond の両分岐）。block の
**外**に居る i4 適格（{@link NON_STAGE_I4_WEIGHTS}）は校正の駆動が届かないので**先に**素の RTN で
丸める（配布実行時の条件へ校正入力を合わせる）。校正の失敗は fail loudly で、素の RTN へ黙って
落ちる分岐は持たない — 明示の `--no-calib` だけが opt-out（配布資産には使わない）。

`--i4-adaln-i8`（既定 OFF）は**量子化感度の実験変種**で、block 内 adaLN（`norm1` / `norm2` /
`norm3` の `linear_1` / `linear_2` = anima-v1.0 で 168 本）と block 外の 5 本を i8 格納へ回し、
残る block 内 linear だけを GPTQ i4 にする（{@link _adaln_i8_names}）。素版 3 モデルの i4 が
視認裁定で配布スキップになった（research `2026-08-24-gptq-expansion-quality.md` §5）ときの
改善候補で、irodori の w4 席では同型の構成が聴感を回復させている（同 §1 の R3）。系列名にも
`calib_provenance.json` の `method` にも `-adaln8` が付き、**配布の一致検査に落ちる**
（視認評価専用 — 組み立ては `python -m anima.eval_dist`）。

`--calib-device`（既定 `cpu`）は校正の**捕捉 + GPTQ だけ**を別デバイスへ出すノブ。`cuda` は
感度実験の回転を上げるための経路で、**丸め解が CPU と変わる**（f64 縮約の順序も linalg の実装も
違う）ため配布には使わない — `calib_provenance.json` の `method` にデバイス名が付き
（`gptq-cuda`）、`-adaln8` と同じく配布の一致検査に落ちる。stage 分解一致門・block 外の RTN・
i8 の丸めは CPU のまま（デバイス差の交絡を丸めの側だけに閉じる）。

MUST: `--dtype f16` は**重みを f16 表現可能値へ丸めてから**（fake-quant — ADR 0006）参照と
golden を採り、適格な重みスロットだけを f16 で格納する（ADR 0018）。丸めは各 builder が
モデルを組んだ直後に掛かる — 参照より後ろへ動かすと「参照だけ元の重み」になり、E2E の差が
量子化誤差と実装誤差の合成になって tolerance の意味が消える。出力先は f32 系列と**別**
（既定 `outputs/series/anima-f16/`）。

MUST: `--verify` と emit は**同一プロセスで併用できない**（CLI が機械的に拒否する）。VAE の
パッチはクラス属性のプロセス全域差し替えなので、emit 側が先にパッチを当てると「パッチ前の
参照」が採れなくなり、同値検証が恒真化して偽 PASS する（ADR 0013 / 0016）。

MUST: `--verify` は**ターゲットを 1 つだけ取る**。複数を並べられる形にすると対ごとの排他表を
CLI が持つことになり、表の穴がそのまま偽 PASS になる（export_sbv2 の先例）。

MUST: `--lora` は **DiT の層切り詰め（`--num-layers`）より前**に焼き込む。後にすると切った層に
対応する LoRA が「対象が無い」まま黙って捨てられ、取りこぼし検査（`fuse_lora` の fail loudly）が
縮小モデルでは効かなくなる。

出力レイアウト（tiny golden / SBV2 と同じ規約）:

    <out>/<target>/model.safetensors      重み・定数 + __metadata__.karume_ir
    <out>/<target>/io.<case>.safetensors  入力と torch CPU での期待出力
    <out>/<target>/lora_provenance.json   焼き込んだ LoRA の帰属（`--lora` を焼いた対象のみ）
    <out>/<target>/calib_provenance.json  i4 の丸め条件（`--dtype i4` の transformer のみ）
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import nn
from torch.export import Dim

from _shared.paths import SERIES_ROOT
from anima.distribution import (
    ADALN_I8_TAG,
    ANIMA_MODELS,
    CALIB_DEVICES,
    CALIB_PROVENANCE_FILE,
    CALIB_SHIPPABLE_DEVICE,
    LORA_PROVENANCE_FILE,
)
from karume.artifacts import staged_publication
from karume.convert import (
    PRESERVED_OP_PREFIXES,
    PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    normalize_boundary_tensor,
)
from karume.dist import sha256_file
from karume.emit import storage_breakdown
from karume.ir import IrGraph
from karume.pipeline import export_to_file
from karume.quantize import (
    DEFAULT_GROUP_SIZE,
    Int8Report,
    channel_rows,
    fake_quant_int4,
    fake_quant_int8,
    iter_quant_targets,
    round_weights_to_f16,
)
from karume.shards import resolve_shards

from . import calib, patch
from .calib_prompts import CALIB_PROMPTS, DEFAULT_CALIB_PROMPTS, calibration_prompts

REPO_ROOT = Path(__file__).resolve().parents[3]
#: この recipe が扱う格納 dtype（irodori / deberta と同じく **recipe 固有の集合**）。
#: core の `karume.emit.WEIGHT_DTYPES` から引かない — core が書ける集合（i4 追加 —
#: ADR 0069）と anima が系列・検収を持つ集合は別の判断で、結合すると core 側の拡張が
#: そのままこの CLI の受理に化ける。
WEIGHT_DTYPES: tuple[str, ...] = ("f32", "f16", "i8", "i4")

#: i4 group32 で丸める**モジュール型**（= i4 の実行経路を持つ op と対 — `karume.emit` の
#: `I4_WEIGHT_OPS`）。DiT に実在する量子化可能型は `nn.Linear` だけ（patchify も
#: `CosmosPatchEmbed.proj` = `nn.Linear`・埋め込み表も conv も無い — `anima/measure_quant.py` の
#: `W4_OP_TYPES` と同じ事実）だが、型で書くのは名前で書かないため。`nn.Embedding` を併記して
#: あるのは、上流が表引きを持ち込んでも**適格判定の一般形のまま**拾うため（表引きされない
#: `nn.Embedding` が現れた場合は emit の明示指定の門が「適格でない」でその FQN を挙げて落ちる
#: ので、deberta の `NON_LOOKUP_EMBEDDINGS` のような名指しの除外は先回りで置かない）。
#: conv 系は入れない — DiT に conv は実在しないため。core 側では conv1d も i4 適格
#: （`groups == 1` ∧ 行長整除 — ADR 0069 追記 7）だが、ここに `nn.Conv1d` を足す実需が無い。
I4_MODULE_TYPES: tuple[type[nn.Module], ...] = (nn.Linear, nn.Embedding)

#: DiT block の**外**に居る i4 適格の重み（ラッパ内 FQN）。校正付き丸めは stage 逐次の駆動
#: なので block の中しか丸められず、外の適格は素の RTN i4 が担う（{@link _round_i4_calibrated}）。
#:
#: 名前で宣言するのは「黙った分類替え」を作らないため — 実測（実重み・28 block）で block 外の
#: i4 適格はこの 5 本（`patch_embed.proj` は量子化軸 68 が g32 非整除で適格外・rope は重み無し）。
#: 上流の構成が変わって適格が増減したら、{@link _round_i4_calibrated} の門がその FQN を挙げて
#: 落ちる（黙って RTN 側へ流す = 校正が痩せる、も黙って i8 へ落とす = サイズだけ戻る、も
#: 数字からは読めない）。
NON_STAGE_I4_WEIGHTS: frozenset[str] = frozenset(
    {
        "model.time_embed.t_embedder.linear_1",
        "model.time_embed.t_embedder.linear_2",
        "model.norm_out.linear_1",
        "model.norm_out.linear_2",
        "model.proj_out",
    }
)

#: 系列名 → `export_to_file` へ渡す**既定**の格納 dtype。i4 系列だけ既定が i8 で、適格な重みは
#: 1 本単位の `weight_dtype_overrides` で i4 へ振る（deberta の `BASE_WEIGHT_DTYPES` と同形）。
BASE_WEIGHT_DTYPES: Mapping[str, str] = {"f32": "f32", "f16": "f16", "i8": "i8", "i4": "i8"}

#: 実重みの取得元（HF Hub。ローカルキャッシュ済み — recon §7）。
DEFAULT_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"
#: 生成物の既定の置き場（格納 dtype 別の**系列**）。ターゲット名のサブディレクトリを 1 段掘る。
#: 親は `SERIES_ROOT`（= outputs/series/）— models/ は配布形だけの場所（_shared.paths）。
#:
#: MUST: f16 系列は**別ディレクトリ**（ADR 0018）。f32 系列（`anima/`）は
#: 「量子化なしの実装誤差」を測る網として独立に残り、f16 系列がその上へ量子化の実装誤差を
#: 上乗せで検証する。同じ場所に上書きすると片方の網が消える。
DEFAULT_OUT_ROOTS = {
    "f32": SERIES_ROOT / "anima",
    "f16": SERIES_ROOT / "anima-f16",
    "i8": SERIES_ROOT / "anima-i8",
    "i4": SERIES_ROOT / "anima-i4",
}

TARGET_TEXT_ENCODER = "text_encoder"
TARGET_TEXT_CONDITIONER = "text_conditioner"
TARGET_TRANSFORMER = "transformer"
TARGET_VAE_DECODER = "vae_decoder"
TARGETS = (
    TARGET_TEXT_ENCODER,
    TARGET_TEXT_CONDITIONER,
    TARGET_TRANSFORMER,
    TARGET_VAE_DECODER,
)

#: 格納 dtype ごとに emit するターゲット（ADR 0019 — i8 / i4 系列は DiT のみ）。
#: MUST: 既定を絞るだけでなく `--target` の明示も拒否する（モジュール docstring の理由）。
DTYPE_TARGETS = {
    "f32": TARGETS,
    "f16": TARGETS,
    "i8": (TARGET_TRANSFORMER,),
    "i4": (TARGET_TRANSFORMER,),
}

#: DiT のグラフ形。`dyn` = トークン長 1 シンボル `S` の**追加系列**（#21 波 T2）。
DIT_GRAPHS = ("static", "dyn")

#: `--dit-graph dyn` の系列ディレクトリ接尾辞（既定 out を静的系列と分けるため）。
DYN_SUFFIX = "-dyn"

MODEL_FILE = "model.safetensors"
#: ホストが rope 表を組むための軸別素表（`--dit-graph dyn` のときだけ書く）。
ROPE_BASE_FILE = "rope_base.safetensors"
IO_PREFIX = "io."
IO_SUFFIX = ".safetensors"
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."

#: 記号次元の上限。text 系は 2 本とも 512（パイプラインの `max_length`）。
SYM_MAX = 512

#: DiT が受け取る `encoder_hidden_states` の行数。**`SYM_MAX` とは別の量**で、由来は
#: `AnimaTextConditioner.config.min_sequence_length`（conditioner の出力をホスト側で
#: ここまでゼロ詰めする — ADR 0016 / anima.pipeline_ref.MIN_SEQUENCE_LENGTH と同じ値）。
#: 記号次元の上限に相乗りさせると、`--sym-max` を動かした瞬間に DiT の**静的な**グラフ形が
#: 黙って変わる（そちらは記号ではないので、束縛では吸収されない）。
MIN_SEQUENCE_LENGTH = 512

#: 生成解像度（px）。DiT / VAE は解像度固定の静的グラフなので、値がグラフ形そのものになる。
RESOLUTION = 512
#: VAE の空間圧縮率（latent 解像度 = 解像度 / これ）。
SPATIAL_COMPRESSION = 8

#: S 形（`--dit-graph dyn`）の golden ケースが取る解像度。**2 点評価**（DeBERTa T /
#: SBV2 P と同じ流儀）で、実運用の 2 解像度をそのまま踏む。先頭が export の例示入力。
DIT_DYN_RESOLUTIONS = (512, 1024)
#: `Dim("S")` の上限。**グラフ側にはトークン長の制約が無い**（rope 表を入力へ出したので
#: S 依存の焼き込み定数がゼロ）ため、ここは「素表の行数 128 × 128 = 2048px 相当」という
#: モデル側の天井をそのまま宣言する。上限を持たせるのは ADR 0010 の定数畳み込み要件。
DIT_SYM_MAX = 16384
#: 同下限。0 / 1 特殊化を避ける線（text 系の `Dim("T", min=2)` と同じ）。
DIT_SYM_MIN = 2

#: 乱数はここから派生させる（グローバル seed に依存しない — 再生成でバイト一致させる）。
SEED = 20260802


def _generator(salt: int) -> torch.Generator:
    return torch.Generator().manual_seed(SEED + salt)


@dataclass(frozen=True)
class Component:
    """1 ターゲットのキュレーション済み export 材料。

    `cases` は golden のケース名と入力タプル（`input_names` と同じ位置）。`reference` は
    **パッチ前** eager の出力で、`--verify` のときだけ採る（VAE パッチはクラス属性の
    プロセス全域差し替えなので、参照はパッチ適用前に採らなければ恒真化する — ADR 0013）。
    """

    module: nn.Module
    dynamic_shapes: Any
    input_names: tuple[str, ...]
    cases: tuple[tuple[str, tuple[torch.Tensor, ...]], ...]
    reference: tuple[tuple[torch.Tensor, ...], ...] | None
    symbol_names: tuple[str, ...] = ("T",)
    #: i8 / i4 の scale 台帳（FQN → scale）。キーは **`module` から見た FQN** で、
    #: safetensors のテンソルキーと同じ空間になる（`--dtype i8` / `i4` 以外では空）。
    weight_scales: Mapping[str, torch.Tensor] = field(default_factory=dict)
    #: 1 本単位の格納 dtype 指定（テンソルキー → dtype）。混成 i4 系列だけが埋める
    #: （既定の {@link BASE_WEIGHT_DTYPES} に優先する — `karume.emit._plan_weight_dtype`）。
    weight_dtype_overrides: Mapping[str, str] = field(default_factory=dict)
    #: `--verify` の突合に入る前にグラフ出力へ掛ける後段（**ホストへ出した段**の逐語再現）。
    #: S 形 DiT は unpatchify がホストなので、参照（パッチ前 diffusers の latent）と比べるには
    #: ここを通す必要がある。第 2 引数は**ケース番号** — ケースごとに解像度が違う系列では
    #: 後段の形もケースで変わるので、1 本の恒等な関数では足りない。既定は恒等。
    verify_adapter: Callable[[torch.Tensor, int], torch.Tensor] | None = None
    #: グラフの外でホストが使う素表（`<out>/rope_base.safetensors` へ書く。既定では空）。
    host_tables: Mapping[str, torch.Tensor] = field(default_factory=dict)

    @property
    def example(self) -> tuple[torch.Tensor, ...]:
        """export の例示入力（**最初のケース**）。dynamic_shapes は位置で対応する。"""
        return self.cases[0][1]


def _eval_cases(
    run: Callable[..., torch.Tensor], cases: Sequence[tuple[str, tuple[torch.Tensor, ...]]]
) -> tuple[tuple[torch.Tensor, ...], ...]:
    with torch.no_grad():
        return tuple((run(*inputs),) for _, inputs in cases)


# ---- ① Qwen3 テキストエンコーダ --------------------------------------------


def build_text_encoder(args: argparse.Namespace, verify: bool) -> Component:
    from transformers import Qwen3Model

    model = Qwen3Model.from_pretrained(
        args.repo, subfolder="text_encoder", dtype=torch.float32, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    if args.num_layers is not None:
        model.layers = model.layers[: args.num_layers]
    model.eval()
    _fake_quant(args, model, TARGET_TEXT_ENCODER)
    generator = _generator(0)
    # 長さは下限（2 — 0/1 特殊化の回避線）から上限 512 の間に散らす。Tmax ちょうどの
    # ケースを 1 本置いて、宣言上限に依存した実装（プランを Tmax で組む等）を踏む。
    cases = tuple(
        (
            f"t{length:03d}",
            (
                torch.randint(
                    5, model.config.vocab_size, (1, length), generator=generator, dtype=torch.long
                ),
            ),
        )
        for length in (5, 24, args.sym_max)
    )
    reference = None
    if verify:
        # 参照は全 1 マスクを渡した原経路（ラッパはマスクを渡さない）。
        reference = _eval_cases(
            lambda ids: (
                model(
                    input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
                ).last_hidden_state
            ),
            cases,
        )
    length = Dim("T", min=2, max=args.sym_max)
    return Component(
        module=patch.AnimaTextEncoder(model),
        dynamic_shapes=({1: length},),
        input_names=("input_ids",),
        cases=cases,
        reference=reference,
    )


# ---- ② テキストコンディショナ ----------------------------------------------


def build_text_conditioner(args: argparse.Namespace, verify: bool) -> Component:
    from diffusers import AnimaTextConditioner

    model = AnimaTextConditioner.from_pretrained(args.repo, subfolder="text_conditioner")
    model.to(torch.float32).eval()
    _apply_lora(args, model, TARGET_TEXT_CONDITIONER)
    _fake_quant(args, model, TARGET_TEXT_CONDITIONER)
    generator = _generator(1)
    # MUST: Tsrc と Ttgt を**ケースごとに別々の値**にする。同じ長さを配ると「片方の束縛で
    # もう片方を代用する」取り違えが数に出ない（ADR 0016 の 2 シンボル取り違え検出）。
    cases = tuple(
        (
            f"case{index}",
            (
                torch.randn(1, source, model.config.source_dim, generator=generator),
                torch.randint(0, model.config.target_vocab_size, (1, target), generator=generator),
            ),
        )
        for index, (source, target) in enumerate(((24, 16), (48, 23)))
    )
    reference = None
    if verify:
        reference = _eval_cases(
            lambda source, target: patch.reference_conditioner(model, source, target), cases
        )
    source_length = Dim("Tsrc", min=2, max=args.sym_max)
    target_length = Dim("Ttgt", min=2, max=args.sym_max)
    return Component(
        module=patch.AnimaConditioner(model),
        dynamic_shapes=({1: source_length}, {1: target_length}),
        input_names=("source_hidden_states", "target_input_ids"),
        cases=cases,
        reference=reference,
        symbol_names=("Tsrc", "Ttgt"),
    )


# ---- ③ CosmosDiT -----------------------------------------------------------

#: 校正の stage 分解一致門（`anima.calib.assert_stage_split`）へ流すラッパ入力の乱数 salt。
#: golden ケースの乱数列（salt 2）とは**別**にする — probe は資産に 1 バイトも出ない診断用の
#: 入力なので、混ぜると golden の値が probe の有無（= `--dtype`）で動く。
PROBE_SALT = 5

#: probe の timestep（0〜1 の sigma スケール — 値そのものに意味は無い。門は同じ入力で
#: 2 経路を比べるだけなので、golden の 2 点とは独立でよい）。
PROBE_TIMESTEP = 0.5


def _dit_probe(model: nn.Module, latent: int) -> tuple[torch.Tensor, ...]:
    """静的形ラッパ（{@link patch.AnimaDit}）の probe 入力（`--dtype i4` のときだけ使う）。"""
    generator = _generator(PROBE_SALT)
    return (
        torch.randn(1, model.config.in_channels, latent, latent, generator=generator),
        patch.dit_timesteps_proj(model, torch.full((1,), PROBE_TIMESTEP)),
        torch.randn(1, MIN_SEQUENCE_LENGTH, model.config.text_embed_dim, generator=generator),
    )


def _dit_tokens_probe(model: nn.Module, patch_size: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
    """S 形ラッパ（{@link patch.AnimaDitTokens}）の probe 入力。

    解像度は S 形 golden の**小さい方**（= 校正の解像度でもある）— 門は block 列の経路一致を
    見るだけなので 1 点で足り、大きい方を選ぶと門だけで数分伸びる。
    """
    generator = _generator(PROBE_SALT)
    side = DIT_DYN_RESOLUTIONS[0] // SPATIAL_COMPRESSION
    latents = torch.randn(1, model.config.in_channels, side, side, generator=generator)
    return (
        patch.dit_patchify(latents, patch_size),
        patch.dit_timesteps_proj(model, torch.full((1,), PROBE_TIMESTEP)),
        torch.randn(1, MIN_SEQUENCE_LENGTH, model.config.text_embed_dim, generator=generator),
        *patch.dit_rope_tables(model, side, side),
    )


def build_transformer(args: argparse.Namespace, verify: bool) -> Component:
    from diffusers import CosmosTransformer3DModel

    model = CosmosTransformer3DModel.from_pretrained(args.repo, subfolder="transformer")
    model.to(torch.float32).eval()
    # MUST: 層を切る**前**に焼き込む（縮小モデルでも全層ぶんの取りこぼし検査を効かせる）。
    _apply_lora(args, model, TARGET_TRANSFORMER)
    if args.num_layers is not None:
        model.transformer_blocks = model.transformer_blocks[: args.num_layers]
    if args.dit_graph == "dyn":
        return _build_transformer_tokens(args, verify, model)
    latent = args.resolution // SPATIAL_COMPRESSION
    # MUST: 丸めは **export するラッパ経由**で当てる（i8 の scale 台帳のキーを
    # `torch.export` が見る FQN と同じ空間に揃えるため — `_fake_quant` の docstring）。
    # ラッパは同じ Parameter を参照するので、下の参照採取（生の `model`）にも同じ丸めが効く。
    # 丸めは切り詰めの後で足りる（切った層は export にも参照にも現れない）。LoRA と違い
    # 「全層ぶんの取りこぼしを検査する」性質が無いので、順序の MUST は無い。
    module = patch.AnimaDit(model, latent, latent)
    scales, dtype_overrides = _fake_quant(
        args, module, TARGET_TRANSFORMER, calib_probe=_dit_probe(model, latent)
    )
    generator = _generator(2)
    # timestep は 0〜1 の連続値（FlowMatch の sigma スケール）。2 点とも別の値にして、
    # timestep 埋め込みがグラフ入力として本当に効いていることを golden の数で踏む。
    raw = [
        (
            torch.randn(1, model.config.in_channels, latent, latent, generator=generator),
            torch.full((1,), step),
            torch.randn(1, MIN_SEQUENCE_LENGTH, model.config.text_embed_dim, generator=generator),
        )
        for step in (0.7, 0.15)
    ]
    reference = None
    if verify:
        with torch.no_grad():
            reference = tuple((patch.reference_dit(model, *inputs),) for inputs in raw)
    cases = tuple(
        (
            f"t{int(step.item() * 1000):04d}",
            (latents, patch.dit_timesteps_proj(model, step), encoder_hidden_states),
        )
        for latents, step, encoder_hidden_states in raw
    )
    return Component(
        module=module,
        dynamic_shapes=None,
        input_names=("latents", "timesteps_proj", "encoder_hidden_states"),
        cases=cases,
        reference=reference,
        weight_scales=scales,
        weight_dtype_overrides=dtype_overrides,
    )


def _build_transformer_tokens(
    args: argparse.Namespace, verify: bool, model: nn.Module
) -> Component:
    """S 形（トークン長 1 シンボル）の DiT — **追加系列**（#21 波 T2）。

    静的形との差は入口と出口だけ（`anima.patch.AnimaDitTokens` の doc）。ここで決めるのは
    ケースの取り方 3 点:

    - **2 点評価は解像度 × timestep の両方を変える**（`DIT_DYN_RESOLUTIONS`）。同じ S を
      2 本並べると「S の束縛が効いていない」実装が数に出ない（conditioner の 2 シンボルで
      同じ長さを配らない規律と同型）。
    - 例示入力（= `cases[0]`）は**小さい方**にする。`torch.export` はトレースで 1 回
      forward を回すので、大きい方を先頭にすると export だけで数分伸びる。
    - `--verify` の参照は**パッチ前 diffusers の latent**（`reference_dit`）で、突合は
      「ホスト patchify → S 形 → ホスト unpatchify」の合成に対して行う（`verify_adapter`）。
      静的ラッパを相手にすると、静的ラッパ自身の同値が別プロセスでしか採れないぶん
      検証が 1 段浅くなる。
    """
    patch_size = tuple(int(size) for size in model.config.patch_size)
    module = patch.AnimaDitTokens(model)
    scales, dtype_overrides = _fake_quant(
        args,
        module,
        TARGET_TRANSFORMER,
        calib_probe=_dit_tokens_probe(model, patch_size),
    )
    generator = _generator(2)
    latents = [
        (
            resolution // SPATIAL_COMPRESSION,
            torch.randn(
                1,
                model.config.in_channels,
                resolution // SPATIAL_COMPRESSION,
                resolution // SPATIAL_COMPRESSION,
                generator=generator,
            ),
            torch.full((1,), step),
            torch.randn(1, MIN_SEQUENCE_LENGTH, model.config.text_embed_dim, generator=generator),
        )
        for resolution, step in zip(DIT_DYN_RESOLUTIONS, (0.7, 0.15), strict=True)
    ]
    reference = None
    if verify:
        with torch.no_grad():
            reference = tuple(
                (patch.reference_dit(model, latent, step, embeds),)
                for _, latent, step, embeds in latents
            )
    cases = tuple(
        (
            f"s{(side // patch_size[1]) * (side // patch_size[2]):05d}"
            f"t{int(step.item() * 1000):04d}",
            (
                patch.dit_patchify(latent, patch_size),
                patch.dit_timesteps_proj(model, step),
                embeds,
                *patch.dit_rope_tables(model, side, side),
            ),
        )
        for side, latent, step, embeds in latents
    )
    tokens = Dim("S", min=DIT_SYM_MIN, max=DIT_SYM_MAX)
    sides = [side for side, *_ in latents]
    return Component(
        module=module,
        # rope 表も同じ `S` で宣言する（別シンボルにすると「表と本体の長さがずれた」形が
        # 受理されてしまい、沈黙誤値になる）。
        dynamic_shapes=({1: tokens}, None, None, {2: tokens}, {2: tokens}),
        input_names=("tokens", "timesteps_proj", "encoder_hidden_states", "rope_cos", "rope_sin"),
        cases=cases,
        reference=reference,
        symbol_names=("S",),
        weight_scales=scales,
        weight_dtype_overrides=dtype_overrides,
        verify_adapter=lambda output, index: patch.dit_unpatchify(
            output, sides[index], sides[index], patch_size
        ),
        host_tables=patch.dit_rope_base_tables(model),
    )


# ---- ④ VAE decoder ---------------------------------------------------------


def build_vae_decoder(args: argparse.Namespace, verify: bool) -> Component:
    from diffusers import AutoencoderKLQwenImage

    vae = AutoencoderKLQwenImage.from_pretrained(args.repo, subfolder="vae")
    vae.to(torch.float32).eval()
    _fake_quant(args, vae, TARGET_VAE_DECODER)
    latent = args.resolution // SPATIAL_COMPRESSION
    generator = _generator(3)
    cases = tuple(
        (f"case{index}", (torch.randn(1, vae.config.z_dim, latent, latent, generator=generator),))
        for index in range(2)
    )
    reference = None
    if verify:
        # MUST: パッチはクラス属性のプロセス全域差し替え — 参照はパッチ前に採る。
        _assert_vae_unpatched("vae_decoder の参照採取")
        reference = _eval_cases(lambda latents: patch.reference_vae_decode(vae, latents), cases)
    patch.apply_vae_decoder_patch(vae)
    return Component(
        module=patch.AnimaVaeDecoder(vae),
        dynamic_shapes=None,
        input_names=("latents",),
        cases=cases,
        reference=reference,
    )


BUILDERS: dict[str, Callable[[argparse.Namespace, bool], Component]] = {
    TARGET_TEXT_ENCODER: build_text_encoder,
    TARGET_TEXT_CONDITIONER: build_text_conditioner,
    TARGET_TRANSFORMER: build_transformer,
    TARGET_VAE_DECODER: build_vae_decoder,
}

#: ターゲット別の「分解を止める高位 op」（ADR 0023）。
#:
#: **DiT（transformer）と VAE decoder だけ** SDPA を保存して融合 `attention` 1 ノードにする。
#: 両者の attention は実測で全て **マスク無し・非因果・dropout 0・GQA 無し・rank-4**（融合の
#: 設計 recon §1）。
#:
#: MUST: `text_encoder` は従来どおり**分解**する — −inf 折り込みの因果マスクを持つので
#: `_h_attention` の fail loudly に当たり、保存すると export できなくなる。ADR 0016 の
#: safe-softmax ガード除去パスもこの分解経路に掛かっている。
#: NOTE: `text_conditioner` はマスク無しで融合可能だが、GPU 時間が 0.10ms/18.7ms（測定限界
#: 以下）なので v1 のスコープ外（設計 recon §1.3）。広げるときはここに 1 行足すだけ。
TARGET_PRESERVED: dict[str, tuple[str, ...]] = {
    TARGET_TEXT_ENCODER: PRESERVED_OP_PREFIXES,
    TARGET_TEXT_CONDITIONER: PRESERVED_OP_PREFIXES,
    TARGET_TRANSFORMER: PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    TARGET_VAE_DECODER: PRESERVED_OP_PREFIXES_WITH_ATTENTION,
}

#: `--lora` が効くターゲット（recon §1 — 蒸留 LoRA の対象は DiT と conditioner のみ）。
LORA_PREFIXES = {
    TARGET_TRANSFORMER: "transformer",
    TARGET_TEXT_CONDITIONER: "text_conditioner",
}


def _assert_vae_unpatched(where: str) -> None:
    """参照採取が VAE パッチ適用**前**であることを固定する（恒真化の門 — ADR 0013）。

    パッチ後に採った「参照」はパッチ後の値そのものになり、同値検証が恒真化して偽 PASS する。
    **差が常に 0 になる**方向の壊れ方なので、検証が緑であること自体は何の証拠にもならない。
    """
    if patch.vae_patches_applied():
        raise RuntimeError(
            f"{where}: VAE パッチ適用後に参照値を採ろうとした — 同値検証が恒真化する"
            "（順序は「全ケースの参照を確定 → パッチ適用 → 比較」）"
        )


def _i4_module_names(model: nn.Module) -> frozenset[str]:
    """i4 group32 で丸めるモジュールの FQN 集合（混成 i8+i4 の**排他割り**の唯一の源）。

    適格は 2 条件の積: ① {@link I4_MODULE_TYPES} であること（emit の i4 適格 = linear /
    embedding の重みスロット限定 — ADR 0069 決定 5 と追補。外れたテンソルへ i4 を明示指定すると
    emit が fail loudly する）② 量子化軸が group 長で割り切れること（i4 は端数 group を作らない
    MUST — 同決定 2。外れた重みは**構成ごと落とすのではなく対象から外す**ので i8 側へ落ちる）。

    DiT で ② に落ちるのは patchify の入口 1 本だけ（`patch_embed.proj` の in 軸 = 17ch × 2×2 =
    68 は g32 で割り切れない）だが、**名前で外さない** MUST — 解像度・patch_size・入力チャネルが
    動けば整除も動くので、名指しの除外は上流が変わった瞬間に嘘になる。

    対象列挙を core（`iter_quant_targets`）に通すのは、丸めが見る集合とここが数える集合を
    1 本の実装のままにするため。i8 側の述語を「この集合に居ない」で書くのも同じ理由で、
    2 つの述語を別々の綴りから作ると、どちらにも入らない重みが**黙って f32 のまま残る**
    （値は正しいままサイズだけが戻るので、数字を見ない限り気づけない）。
    """
    return frozenset(
        fqn.removesuffix(".weight")
        for fqn, weight, axis in iter_quant_targets(model, I4_MODULE_TYPES)
        if channel_rows(weight, axis).shape[-1] % DEFAULT_GROUP_SIZE == 0
    )


def _adaln_i8_names(model: nn.Module) -> frozenset[str]:
    """感度実験変種（`--i4-adaln-i8`）が i4 から外して **i8 で丸める**重みの FQN 集合。

    中身は 2 群の和で、どちらも「modulation を作る側」= 量子化感度が高いと目された役割:

    1. **block 内の adaLN**（`norm1` / `norm2` / `norm3` の `linear_1` / `linear_2` —
       anima-v1.0 で 28 block × 6 = 168 本）。判定はセグメント一致
       （{@link anima.calib.is_adaln}）で、名指しの一覧は持たない。
    2. **block の外の 5 本**（{@link NON_STAGE_I4_WEIGHTS}）。うち 4 本は modulation の
       系譜そのもの（`time_embed.t_embedder` が全 adaLN へ渡る temb を作り、`norm_out` は
       最終段の `CosmosAdaLayerNorm`）で、残る `proj_out` は校正の駆動が stage 単位である
       都合で GPTQ に載らず素の RTN i4 で丸まっていた側 — irodori の w4 席では**この 2 群を
       まとめて i8 へ戻した構成**（research `2026-08-24-gptq-expansion-quality.md` §1 の R3）が
       採用されており、片方だけを動かした構成はどこでも検証されていない。

    MUST: adaLN の綴りが 1 つでも見つからなければ fail loudly（{@link
    anima.calib.adaln_segments_seen}）— 上流が一部だけを改名すると、除外したはずの本数だけが
    黙って i4 へ戻り、格納形も本数の門も自己整合したまま緑になる。
    """
    stage_names = calib.stage_linear_names(calib.dit_stages(model))
    missing = sorted(calib.ADALN_SEGMENTS - calib.adaln_segments_seen(stage_names))
    if missing:
        raise AssertionError(
            f"DiT block 内に adaLN の linear が 1 本も無い綴り: {missing}"
            f"（探した綴り: {sorted(calib.ADALN_SEGMENTS)}）— 上流が modulation の属性名を"
            "変えている"
        )
    return frozenset(name for name in stage_names if calib.is_adaln(name)) | NON_STAGE_I4_WEIGHTS


def _round_i4_plain(
    model: nn.Module, names: frozenset[str], target: str, label: str
) -> Mapping[str, torch.Tensor]:
    """名指しの集合を素の RTN i4 g32 で丸める（校正を通さない 1 段目の格子そのもの）。

    `label` は診断行の主語（助詞まで込み — 「〜を i4 group へ丸めた」に嵌まる形）。
    """
    report = fake_quant_int4(model, include=lambda name: name in names, op_types=I4_MODULE_TYPES)
    print(
        f"[fake-quant] {target}: {label} i4 group へ丸めた（RTN） — {report.describe()}",
        flush=True,
    )
    return report.scales


def _round_i4_calibrated(
    args: argparse.Namespace,
    wrapper: nn.Module,
    target: str,
    i4_names: frozenset[str],
    excluded: frozenset[str],
    probe: Sequence[torch.Tensor],
    round_int8: Callable[[], Int8Report],
) -> tuple[Mapping[str, torch.Tensor], Int8Report]:
    """i4 で格納する重みを「block 外 = 素の RTN」「block 内 = GPTQ」へ排他に割って丸める。

    `i4_names` は**格納 i4 の全量**、`excluded` は i4 適格でありながら i8 へ回す側
    （既定は空・感度実験変種では {@link _adaln_i8_names} の 173 本）。適格の全量は 2 つの和で、
    下の 2 つの門はそちら（= グラフが決める事実）に対して掛かる — 変種で動くのは割り方だけで、
    「上流の構成が変わっていないか」を見る網は 1 つも緩めない。

    順序（① → ② → ③ → ④。**②〜④ は MUST・① は SHOULD**）:

    1. **stage 分解一致門を丸める前に通す**（`anima.calib.assert_stage_split`）— ずれた分解で
       丸めると「別の経路の GPTQ」を出荷することになり、しかも数値は普通に出る。SHOULD どまり
       なのは、この門が読み取り専用で**位置を後ろへ動かしても出荷バイトが変わらない**ため
       （先頭に置く得は、数時間の校正を回す前に落ちること）。
    2. **block 外を先に丸める** — 既定はここが RTN i4（配布実行時に block へ入るのは i4 の
       `time_embed` が作った temb で、step をまたぐ latent も i4 の `norm_out` / `proj_out` を
       通った値）。後に回すと「f32 の周辺を通った活性」で選んだ丸め先を、丸めた周辺と組んで
       配ることになる。**i8 側も同じ理由で校正入力の捕捉より前に丸める**
       （{@link _fake_quant_i4} の `round_int8` をここで呼ぶ）— 既定では `patch_embed.proj`
       1 本（patchify 入口 = block 0 の入力そのもの）、変種ではそこへ除外した 173 本が加わり、
       block 内の adaLN もこの時点で i8 になる（配布実行時に modulation を作るのは i8 の
       adaLN だから、その状態で校正入力を採る）。
    3. 校正入力の生成（参照 denoise の捕捉）と付随引数一致門。
    4. block 内の linear を GPTQ × RTN 格子で丸める。**除外は core の `include` で排他に**
       （`anima.calib.is_adaln` — 丸めた集合と i4 格納の集合を 1 実装で決める）。

    MUST: block 外の適格は {@link NON_STAGE_I4_WEIGHTS} と一致し、block 内の linear は 1 本
    残らず i4 適格であること — どちらも外れたら上流の構成が変わっている。

    NOTE（デバイス）: `--calib-device` が動かすのは ③④（捕捉 + GPTQ）だけで、①② は CPU のまま
    回す（デバイス差の交絡を丸め方式の側だけに閉じ込める）。戻りは try/finally で必ず CPU —
    emit も golden も CPU 経路で、GPU に残った重みは後続の診断を別の形で壊す。既定 `cpu` では
    `Module.to` も `Tensor.to` も no-op なので、配布経路のバイトは 1 つも動かない。
    """
    stages = calib.dit_stages(wrapper)
    stage_names = calib.stage_linear_names(stages)
    eligible = i4_names | excluded
    unaligned = sorted(stage_names - eligible)
    if unaligned:
        raise AssertionError(
            f"DiT block 内の linear {unaligned[:3]} が i4 適格でない（量子化軸が g"
            f"{DEFAULT_GROUP_SIZE} 非整除）— 校正は stage を丸ごと駆動するので 1 本だけ外す"
            "逃げ道が無い"
        )
    non_stage = eligible - stage_names
    if non_stage != NON_STAGE_I4_WEIGHTS:
        raise AssertionError(
            f"DiT block の外の i4 適格が {sorted(non_stage)} で、宣言"
            f"（NON_STAGE_I4_WEIGHTS = {sorted(NON_STAGE_I4_WEIGHTS)}）と違う"
            " — 上流の構成が変わって i4 の割り方が動いている"
        )
    graph_batch = calib.assert_stage_split(wrapper, probe, stages)
    plain_names = non_stage - excluded
    plain = _round_i4_plain(wrapper, plain_names, target, "block 外の適格を") if plain_names else {}
    int8 = round_int8()
    prompts = calibration_prompts(args.calib_prompts)
    conditions = calib.calib_conditions(args.model)
    # デバイスの継ぎ目は**ここ**（i8 丸めの直後）— 上の ①②（stage 分解一致門・block 外 RTN・
    # i8 丸め）は CPU のまま回す。GPTQ の丸め解はデバイスで変わるので、デバイス差が混ざる範囲を
    # 「捕捉 + GPTQ」だけに隔離しておかないと、視認 A/B が何の差を見ているのか言えなくなる。
    wrapper.to(args.calib_device)
    try:
        batches = calib.capture_stage_batches(
            wrapper.model, prompts, repo=args.repo, conditions=conditions
        )
        calib.assert_calib_batches_match_graph(graph_batch, batches)
        rig = calib.CalibRig(stages=stages, batches=batches)
        # MUST: 述語が選ぶ集合 = i4 で格納する block 内の集合。既定（除外なし）は `None` を渡して
        # **述語そのものを持たない** — 「全部」を綴り直した述語は、走査の定義が動いた日に片側だけ
        # 追随して黙ってずれる。
        stage_i4 = stage_names - excluded
        report, ledger = calib.calibrate_i4(
            rig, None if not excluded else lambda local: not calib.is_adaln(local)
        )
    finally:
        # MUST: emit へ渡すのは CPU の重み（グラフ採取も golden も CPU 経路）。例外で抜けた
        # ときも戻す — 失敗したまま GPU に残ると、後続の診断が「重みが無い」形で化ける。
        wrapper.to("cpu")
        if args.calib_device == "cuda":
            torch.cuda.empty_cache()
    calib.assert_calib_covers_scan(report, stage_i4)
    # 台帳は丸めが走ったデバイスに載って返る — 格納側（emit）は CPU のテンソルを前提にするので
    # ここで回収する（`plain` は CPU 産のまま）。
    calibrated = {fqn: scale.cpu() for fqn, scale in ledger.scales.items()}
    # MUST: 2 経路は互いに素（重なれば同じ重みを 2 度丸めたことになり、値だけが静かに狂う）。
    overlap = sorted(set(plain) & set(calibrated))
    if overlap:
        raise AssertionError(f"i4 の 2 経路が同じ重みを丸めている（二重丸め）: {overlap[:3]}")
    scope = (
        "DiT block の linear"
        if not excluded
        else f"DiT block の adaLN 以外 {len(stage_i4)} 本"
        f"（adaLN {len(excluded - NON_STAGE_I4_WEIGHTS)} 本 + block 外"
        f" {len(NON_STAGE_I4_WEIGHTS)} 本は i8 格納 — --i4-adaln-i8）"
    )
    print(
        f"[fake-quant] {target}: {scope}を GPTQ 校正付きで丸めた"
        f" — {report.describe()} / 校正プロンプト {len(prompts)} 本・バッチ {len(batches)} 本"
        f"・{rig.tokens:,} トークン（{args.model}: {conditions.steps} step"
        f"・CFG {conditions.guidance:g}・分岐 {conditions.branches}）",
        flush=True,
    )
    return {**plain, **calibrated}, int8


def _fake_quant_i4(
    args: argparse.Namespace,
    model: nn.Module,
    target: str,
    probe: Sequence[torch.Tensor] | None,
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, str]]:
    """混成 i4 系列の丸め（格納 i4 = 適格 − 除外・残り = i8 per-channel）。

    2 つの述語は {@link _i4_module_names} の適格から**排他に**割る（`quantize.py` の混成
    MUST）。返す override は「i4 の scale 台帳のキー全部を i4 に振る」写像で、emit 側は明示
    指定を満たせなければ fail loudly する（系列の既定格納は i8 なので、i8 へ回した側は
    override に載せない = 既定のまま）。

    `--i4-adaln-i8`（既定 OFF）は適格から {@link _adaln_i8_names} を**引いて** i4 の集合を
    決める感度実験変種で、引いた側は i8 の述語（`name not in i4_names`）が自動で拾う。
    既定では除外が空なので、この関数が通る経路も丸める集合も従来と 1 バイトも変わらない。

    適格の丸めは既定で校正付き（{@link _round_i4_calibrated}）。`--no-calib` のときだけ
    格納 i4 の全量を素の RTN で丸める — **配布資産には使わない**テスト / smoke 用の opt-out
    で、「校正付きのつもりで校正なしを配った」が資産から読めないので診断行で明示する。
    """
    eligible = _i4_module_names(model)
    # MUST: 除外は**適格を計算した後**に引く（適格判定より前に名前で削ると、上流の構成が
    # 変わって適格から落ちた重みと「意図して外した重み」が区別できなくなる）。
    excluded = _adaln_i8_names(model) if args.i4_adaln_i8 else frozenset()
    outside = sorted(excluded - eligible)
    if outside:
        raise AssertionError(
            f"i8 へ回す指定 {outside[:3]} が i4 適格に無い（適格 {len(eligible)} 本 /"
            f" 除外 {len(excluded)} 本）— 上流の構成が変わって除外の綴りが空振りしている"
        )
    i4_names = eligible - excluded

    def round_int8() -> Int8Report:
        return fake_quant_int8(model, include=lambda name: name not in i4_names)

    if args.no_calib:
        print(
            f"[fake-quant] {target}: --no-calib — 校正なしの素の RTN"
            "（配布資産にしないこと・品質は perf-ledger Q-6 の基線より下）",
            flush=True,
        )
        int4_scales = _round_i4_plain(model, i4_names, target, "格納 i4 の重みを")
        int8 = round_int8()
    else:
        if probe is None:
            raise AssertionError(
                f"{target}: 校正付き i4 には stage 分解一致門の probe が要る"
                "（builder が渡していない — 校正なしへ黙って落ちる分岐は持たない）"
            )
        # 校正経路では i8 の丸めを**校正入力の捕捉より前**に差し込む（順序 MUST ② — 呼ぶ位置は
        # {@link _round_i4_calibrated} が持つ）。適格判定の門より前に呼ぶと、適格の綻びより先に
        # 「i8 の対象が 1 本も無い」で落ちて診断が入れ替わる。
        int4_scales, int8 = _round_i4_calibrated(
            args, model, target, i4_names, excluded, probe, round_int8
        )
    print(f"[fake-quant] {target}: 残りは i8 per-channel — {int8.describe()}", flush=True)
    # MUST: 丸めた集合 = 格納集合（override のキー）。ずれるのは「i4 と数えたのに丸まって
    # いない」形で、i4 席に i8 の重みが混ざったまま緑になる（サイズだけが静かに戻る）。
    expected = {f"{name}.weight" for name in i4_names}
    if set(int4_scales) != expected:
        raise AssertionError(
            f"格納 i4 の {len(i4_names)} 本に対し丸めたのは {len(int4_scales)} 本"
            f"（過不足: {sorted(set(int4_scales) ^ expected)[:3]}）"
        )
    # MUST: 除外した側は i8 で丸まっていること。i8 の述語は `i4_names` の否定なので構造上は
    # 落ちないが、ここが空振りすると「i4 からも i8 からも漏れて f32 のまま」— 値は正しく
    # サイズだけが戻る壊れ方で、格納指定にも本数の門にも出ない。
    unrounded = sorted(f"{name}.weight" for name in excluded if f"{name}.weight" not in int8.scales)
    if unrounded:
        raise AssertionError(
            f"i4 から外した {len(excluded)} 本のうち {unrounded[:3]} が i8 でも丸まっていない"
            "（i4 側にも i8 側にも入らず f32 のまま残っている）"
        )
    return {**int8.scales, **int4_scales}, dict.fromkeys(int4_scales, "i4")


def _fake_quant(
    args: argparse.Namespace,
    model: nn.Module,
    target: str,
    *,
    calib_probe: Sequence[torch.Tensor] | None = None,
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, str]]:
    """格納 dtype の表現可能値へ重みを丸め、scale 台帳と 1 本単位の格納指定を返す（ADR 0006）。

    MUST: 各 builder で **`reference` の採取より前**・`--lora` の焼き込みより**後**に呼ぶ。
    前後を逆にすると①参照だけが元の重みで計算されて E2E の差に量子化誤差が混ざる
    ②焼き込んだ ΔW が丸めを外して格納時の再丸めが golden との対応を壊す。

    MUST（i8 / i4 のみ）: **export する `nn.Module` そのもの**に当てる。scale 台帳のキーは
    ここで見た FQN で、safetensors のテンソルキー（= `torch.export` が見る FQN）と同じで
    なければ emit 側の突合が空振りする（`id()` 突合は禁止 — ADR 0006）。

    `calib_probe` は i4 の校正付き丸めが stage 分解一致門に流すラッパの入力（golden ケースの
    乱数列を動かさないよう、builder が**別 salt**で組んだもの）。校正するのに欠けていたら
    fail loudly — 「probe が無いから校正なし」で黙って落ちる分岐は作らない。
    """
    if args.dtype == "f32":
        return {}, {}
    if args.dtype == "i8":
        report = fake_quant_int8(model)
        print(f"[fake-quant] {target}: i8 per-channel へ丸めた — {report.describe()}", flush=True)
        return report.scales, {}
    if args.dtype == "i4":
        return _fake_quant_i4(args, model, target, calib_probe)
    rounded = round_weights_to_f16(model)
    print(f"[fake-quant] {target}: f16 表現可能値へ丸めた — {rounded.describe()}", flush=True)
    return {}, {}


def _write_lora_provenance(args: argparse.Namespace, target: str, out_dir: Path) -> str | None:
    """焼き込んだ LoRA の帰属（ファイル名 + sha256）を系列へ書き、ファイル名を返す。

    融合後の重みからは「どの LoRA を焼いたか」が復元できないので、**焼いた側が事実を書き残す**
    のが唯一の道。配布 README が印字する帰属（`anima/card.py` の `LORA_SHA256`）は
    `anima/distribution.py` の `assert_lora_provenance` がこの記録と突き合わせる。

    書くのは実際に焼いたターゲットだけ（`LORA_PREFIXES` に無いターゲットの系列に置くと
    「このモデルにも焼いた」という事実でない主張になる）。配布形には入らない —
    `anima/distribution.py` の配置表が許可した役割だけが出力へ渡る。

    MUST: 焼いていない側で**古い記録を消す**（全域関数）。`--lora` 付きで採った系列へ `--lora`
    無しで採り直したとき、重みだけ素に戻って記録が前回のまま生き残ると、
    `assert_lora_provenance` は「記録が在る × sha 一致」しか見ないので「取り下げ」を素通しして
    配布 README に嘘の帰属を印字させる。記録の存在が「今の `model.safetensors` に焼いた」と
    同値であることが、この機構の唯一の拠り所。NOTE: `emit_target` が作業席ごと据え替えるように
    なった今、CLI 経路では席が毎回まっさらなのでここは実質空振りする — それでも全域関数のまま
    残すのは、直接呼ぶ書き手（テスト・将来の別入口）にとって不変条件が変わらないため。
    """
    if args.lora is None or target not in LORA_PREFIXES:
        (out_dir / LORA_PROVENANCE_FILE).unlink(missing_ok=True)
        return None
    record = {"file": args.lora.name, "sha256": sha256_file(args.lora)}
    (out_dir / LORA_PROVENANCE_FILE).write_text(
        json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return LORA_PROVENANCE_FILE


def _write_calib_provenance(args: argparse.Namespace, target: str, out_dir: Path) -> str | None:
    """i4 系列の丸め条件（方式・格子・コーパス・解像度・step・CFG・デバイス）を系列へ書き、
    ファイル名を返す。

    校正の有無は**格納形を 1 バイトも変えない**ので、資産からは復元できない — `--lora` と同じ形で
    「書き出した側が事実を書き残す」しかない。`anima/distribution.py` の
    `assert_calib_provenance` が組み立て時にこの記録と突き合わせ、`--no-calib` の生成物が配布へ
    紛れるのを止める。

    MUST: i4 以外では**古い記録を消す**（全域関数 — `_write_lora_provenance` と同じ理由・
    据え替えで空振りする点も同じ）。i4 で採った系列を別 dtype で採り直したとき記録だけが前回の
    まま生き残ると、「校正付き」という事実でない主張が残る。
    """
    if args.dtype != "i4" or target != TARGET_TRANSFORMER:
        (out_dir / CALIB_PROVENANCE_FILE).unlink(missing_ok=True)
        return None
    # `--no-calib` でも**書く**（消すのではなく `rtn` と記録する）— 不在は「古い export」とも
    # 読めてしまい、組み立て側が「校正なしを配ろうとした」を名指しで拒否できない。
    #
    # `guidance` は校正条件をモデル別化した 2026-08-23（波 J-4 ②）に足した欄。読み手
    # （`distribution.assert_calib_provenance`）が見るのは `method` だけなので、欄の追加は
    # 既にある turbo 系列の記録（追加前の形）の受理を 1 つも変えない — 既存資産へ再 export を
    # 要求しない形に留める MUST（記録は資産と違って作り直しの費用が丸め時間そのもの）。
    conditions = None if args.no_calib else calib.calib_conditions(args.model)
    # 感度実験変種は**方式の綴りごと**変える（`gptq` → `gptq-adaln8`）。丸め方式が同じでも
    # 「どの重みに当てたか」が違えば別の丸め方で、格納形からは 1 バイトも判別できない —
    # 配布側の一致検査（`distribution.CALIB_SHIPPABLE_METHOD`）がそのまま名指しで拒否する
    # 綴りにしておくのが、視認用に焼いた変種を配布経路から締め出す唯一の手（{@link
    # ADALN_I8_TAG}）。
    method = "rtn" if args.no_calib else calib.CALIB_METHOD
    if args.i4_adaln_i8:
        method = f"{method}-{ADALN_I8_TAG}"
    # 校正デバイスも**方式の綴りごと**変える（`gptq` → `gptq-cuda`）。GPTQ の丸め解は
    # デバイスで変わる（f64 縮約の順序も linalg の実装も違う）ので、cuda で焼いた i4 は
    # 「配布条件で焼いた i4」ではない — しかも格納形からは 1 バイトも判別できない。
    # `device` 欄は人と後の突き合わせのための事実で、配布を止めるのは綴りの側（{@link
    # anima.distribution.CALIB_SHIPPABLE_DEVICE}）。
    if args.calib_device != CALIB_SHIPPABLE_DEVICE:
        method = f"{method}-{args.calib_device}"
    record: dict[str, object] = {
        "method": method,
        "group_size": calib.CALIB_GRID.group_size,
        "grid": calib.CALIB_GRID.kind,
        "prompts": 0 if args.no_calib else args.calib_prompts,
        "resolution": 0 if args.no_calib else calib.CALIB_RESOLUTION,
        "steps": 0 if conditions is None else conditions.steps,
        "guidance": 0 if conditions is None else conditions.guidance,
        "text_dtype": calib.CALIB_TEXT_DTYPE,
        "device": args.calib_device,
    }
    (out_dir / CALIB_PROVENANCE_FILE).write_text(
        json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return CALIB_PROVENANCE_FILE


def _apply_lora(args: argparse.Namespace, model: nn.Module, target: str) -> None:
    if args.lora is None:
        return
    from .lora import fuse_lora, load_lora_state_dict

    state = load_lora_state_dict(args.lora)
    report = fuse_lora(model, state, LORA_PREFIXES[target], args.lora_scale)
    print(f"[lora] {target}: {report.describe()}", flush=True)


# ---- emit / golden ---------------------------------------------------------


def _write_io(component: Component, graph: IrGraph, out_dir: Path) -> list[str]:
    """各ケースの入力と torch CPU 期待出力を `io.<case>.safetensors` へ書く。

    MUST: グラフ入力の宣言名がラッパの引数名と一致していることを先に固定する（IR の入力名は
    forward の引数名がそのまま出る）。ずれたまま位置で書くと、golden のキーだけが黙って
    別の入力を指す。
    """
    declared = [entry.name for entry in graph.inputs]
    if declared != list(component.input_names):
        raise AssertionError(
            f"グラフ入力名が宣言と不一致: {declared} vs {list(component.input_names)}"
        )
    written: list[str] = []
    for name, inputs in component.cases:
        with torch.no_grad():
            result = component.module(*inputs)
        outputs = (result,) if isinstance(result, torch.Tensor) else tuple(result)
        if len(outputs) != len(graph.outputs):
            raise AssertionError(
                f"{name}: eager 出力 {len(outputs)} 本が IR 出力 {len(graph.outputs)} 本と違う"
            )
        # MUST: io も IR の意味論 dtype の実表現へ落とす（ADR 0009 の境界正規化）。
        tensors = {
            f"{INPUT_PREFIX}{key}": normalize_boundary_tensor(value, f"{name} の入力 '{key}'")
            for key, value in zip(component.input_names, inputs, strict=True)
        }
        for index, value in enumerate(outputs):
            tensors[f"{OUTPUT_PREFIX}{index}"] = normalize_boundary_tensor(
                value.detach().contiguous(), f"{name} の出力 {index}"
            )
        path = out_dir / f"{IO_PREFIX}{name}{IO_SUFFIX}"
        save_file(tensors, str(path))
        written.append(path.name)
    return written


def emit_target(target: str, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    """1 ターゲットの IR コンテナと golden io を書き、要約を返す。

    MUST: 生成物は作業席へ書き、**全ての門**（出力本数・境界正規化）を通してから据える。門より
    前に final へ置くと、落ちた実走が「検収門を通れる資産」を残す — io golden は同じ壊れた
    コンポーネントから採るので互いに整合し、TS 側の突合は**緑になる**（「いつ公開してよいか」の
    綴りは {@link _shared.decode_series._publish}・据え替えと後片付けの規律は core の原語
    {@link karume.artifacts.staged_publication}）。据える単位が**ターゲットのディレクトリ丸ごと**
    なので、容器と出所記録（LoRA / 校正）が食い違った組も作れない。
    """
    started = time.perf_counter()
    component = BUILDERS[target](args, False)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with staged_publication(out_dir) as staged:
        # ディレクトリの席は書き手が作る（原語は席を作らない — path しか渡さない）。
        staged.mkdir()
        graph = export_to_file(
            component.module,
            component.example,
            staged / MODEL_FILE,
            dynamic_shapes=component.dynamic_shapes,
            symbol_names=component.symbol_names,
            weight_dtype=BASE_WEIGHT_DTYPES[args.dtype],
            weight_scales=component.weight_scales,
            weight_dtype_overrides=component.weight_dtype_overrides,
            preserved=TARGET_PRESERVED[target],
        )
        written = _write_io(component, graph, staged)
        if component.host_tables:
            # ホスト素表は IR コンテナの外に置く（グラフが使わないテンソルを model.safetensors へ
            # 混ぜると、initializer とテンソルキーの 1:1 検査〈verify_model〉が壊れる）。
            save_file(dict(component.host_tables), str(staged / ROPE_BASE_FILE))
            written.append(ROPE_BASE_FILE)
        provenance = _write_lora_provenance(args, target, staged)
        if provenance is not None:
            written.append(provenance)
        calib_record = _write_calib_provenance(args, target, staged)
        if calib_record is not None:
            written.append(calib_record)
    breakdown = storage_breakdown(graph)
    return {
        "target": target,
        "dir": str(out_dir),
        "dtype": args.dtype,
        "dit_graph": args.dit_graph if target == TARGET_TRANSFORMER else "static",
        # 校正付き丸めの条件（i4 以外・`--no-calib` は 0）— この要約は stdout にしか出ないので
        # **人が読むため**のもの。機械の突き合わせは `calib_provenance.json`
        # （{@link _write_calib_provenance} → `distribution.assert_calib_provenance`）が持つ。
        "calib_prompts": 0 if args.dtype != "i4" or args.no_calib else args.calib_prompts,
        "calib_resolution": 0 if args.dtype != "i4" or args.no_calib else calib.CALIB_RESOLUTION,
        # 役割別の格納割り（感度実験変種）。数時間の export の記録が要約 1 枚で読めるように
        # 出す — 機械の突き合わせは `calib_provenance.json` の `method` が持つ。
        "i4_adaln_i8": args.dtype == "i4" and args.i4_adaln_i8,
        "nodes": len(graph.nodes),
        "outputs": len(graph.outputs),
        "initializers": len(graph.initializers),
        # 適格 = 圧縮のまま GPU 常駐する重み / 適格外 = ロード時に f32 展開（ADR 0006 の診断）。
        "compressed_tensors": breakdown.compressed_tensors,
        "compressed_bytes": breakdown.compressed_bytes,
        "plain_tensors": breakdown.plain_tensors,
        "plain_bytes": breakdown.plain_bytes,
        # i8 の companion scale（ランタイムの residentCompressedBytes も同じものを足す）。
        "scale_bytes": breakdown.scale_bytes,
        "model_bytes": sum(p.stat().st_size for p in resolve_shards(out_dir / MODEL_FILE)),
        "ops": sorted(graph.required_ops),
        "symbols": list(graph.symbols),
        "io": written,
        "case_shapes": {
            name: [list(tensor.shape) for tensor in inputs] for name, inputs in component.cases
        },
        "seconds": round(time.perf_counter() - started, 1),
    }


# ---- パッチ前後の eager 同値 -----------------------------------------------


def verify_target(target: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    """パッチ層適用前後の eager 同値を実重みで実測する（ADR 0016）。

    `bit_exact` は「差 0」より強い主張（`0.0 == -0.0` は差 0 だがビットは違う）で、Qwen3 /
    conditioner のマスク落としのように**ビット一致が主張の中身**である検証で意味を持つ。
    """
    component = BUILDERS[target](args, True)
    if component.reference is None:
        raise AssertionError(f"{target}: --verify なのに参照が採られていない")
    report: list[dict[str, Any]] = []
    for index, ((name, inputs), expected) in enumerate(
        zip(component.cases, component.reference, strict=True)
    ):
        with torch.no_grad():
            result = component.module(*inputs)
            if component.verify_adapter is not None:
                result = component.verify_adapter(result, index)
        outputs = (result,) if isinstance(result, torch.Tensor) else tuple(result)
        report.append(
            {
                "case": name,
                "shapes": [list(tensor.shape) for tensor in inputs],
                "maxdiff": max(
                    float((got - want).abs().max())
                    for got, want in zip(outputs, expected, strict=True)
                ),
                "bit_exact": all(
                    torch.equal(got, want) for got, want in zip(outputs, expected, strict=True)
                ),
            }
        )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="出力先（既定は --dtype ごとの系列 — outputs/series/anima{,-f16,-i8,-i4}/・ADR 0018）",
    )
    parser.add_argument(
        "--dtype",
        choices=WEIGHT_DTYPES,
        default="f32",
        help=(
            "重みの格納 dtype（f16 / i8 / i4 は fake-quant してから適格スロットだけ圧縮格納する。"
            "i4 は混成で、適格な重みが group32 の i4・残りは i8 — ADR 0069。"
            "i4 の DiT block 内 linear は GPTQ 校正付きで丸める（perf-ledger Q-6）。"
            "i8 / i4 は transformer 専用 — ADR 0019）"
        ),
    )
    parser.add_argument(
        "--target",
        action="append",
        choices=TARGETS,
        default=None,
        help=f"emit するターゲット（繰り返し可・既定は全て: {', '.join(TARGETS)}）",
    )
    parser.add_argument(
        "--verify",
        choices=TARGETS,
        default=None,
        help="パッチ前後の eager 同値を検証する（emit はしない・1 プロセス 1 ターゲット）",
    )
    parser.add_argument(
        "--dit-graph",
        choices=DIT_GRAPHS,
        default=DIT_GRAPHS[0],
        help=(
            "DiT のグラフ形（dyn = トークン長 1 シンボル S の追加系列・transformer 専用。"
            "既定 out に -dyn が付く — #21 波 T2）"
        ),
    )
    parser.add_argument("--sym-max", type=int, default=SYM_MAX, help="text 系の記号次元の上限")
    parser.add_argument("--resolution", type=int, default=RESOLUTION, help="生成解像度（px）")
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="DiT / Qwen3 の層切り詰め（開発イテレーション用 — LoRA は切り詰め前に焼く）",
    )
    parser.add_argument(
        "--lora",
        type=Path,
        default=None,
        help="export 前に重みへ焼き込む LoRA（transformer / text_conditioner に効く）",
    )
    parser.add_argument("--lora-scale", type=float, default=1.0, help="LoRA の倍率")
    parser.add_argument(
        "--calib-prompts",
        type=int,
        default=DEFAULT_CALIB_PROMPTS,
        help=f"i4 の GPTQ 校正に使うプロンプト本数（先頭 N 本・既定 {DEFAULT_CALIB_PROMPTS}・"
        f"上限 {len(CALIB_PROMPTS)} — 本数は品質の上振れ軸で、export の CPU 時間に線形に効く）",
    )
    parser.add_argument(
        "--no-calib",
        action="store_true",
        help="i4 を校正なしの素の RTN で丸める（テスト / smoke 用 — 配布資産にしないこと）",
    )
    parser.add_argument(
        "--calib-device",
        choices=CALIB_DEVICES,
        default=CALIB_SHIPPABLE_DEVICE,
        help="GPTQ 校正（捕捉 + 丸め）を回すデバイス"
        f"（既定 {CALIB_SHIPPABLE_DEVICE}）。cuda は感度実験用 — 丸め解が CPU と変わるので"
        "配布経路には使わない（calib_provenance の method にデバイス名が付き、"
        "配布の一致検査に落ちる）",
    )
    parser.add_argument(
        "--i4-adaln-i8",
        action="store_true",
        help="i4 の量子化感度実験変種: block 内 adaLN（norm1/2/3 の linear_1/2）と block 外の"
        f" {len(NON_STAGE_I4_WEIGHTS)} 本を i8 格納へ回し、残りだけを i4 にする"
        f"（既定 OFF・系列名と calib_provenance の method に -{ADALN_I8_TAG} が付き、"
        "配布の一致検査に落ちる = 視認評価専用）",
    )
    parser.add_argument(
        "--model",
        choices=tuple(ANIMA_MODELS),
        default=None,
        help="校正条件（step 数 / CFG / negative prompt）を引く配布モデル名"
        "（--dtype i4 の校正で必須 — 既定値は置かない）",
    )
    args = parser.parse_args(argv)
    if args.out is None:
        # MUST: 変種は既定 out でも**別ディレクトリ**（配布条件で焼いた i4 系列を、視認用の
        # 変種が同じ path へ上書きしない）。校正デバイスも同じ穴 — cuda で焼いた i4 は丸め解が
        # 別物なのに、既定 out を共有すると配布条件の系列を黙って潰す。接尾辞の順は
        # `-adaln8` → `-cuda` → `-dyn` で、grep したときグラフ形が末尾に揃う。
        root = DEFAULT_OUT_ROOTS[args.dtype]
        name = root.name + (f"-{ADALN_I8_TAG}" if args.i4_adaln_i8 else "")
        if args.calib_device != CALIB_SHIPPABLE_DEVICE:
            name += f"-{args.calib_device}"
        args.out = root.with_name(name + (DYN_SUFFIX if args.dit_graph == "dyn" else ""))

    # MUST: 同一プロセスでの併用を機械的に拒否する。VAE パッチはプロセス全域の差し替えなので、
    # emit 側が先に当てると「パッチ前の参照」が汚染され、同値検証が恒真化して偽 PASS する
    # （ADR 0013 / 0016）。順序を main の中で気をつける規律にはしない。
    if args.verify is not None and args.target is not None:
        parser.error(
            "--verify と --target は同一プロセスで併用できない"
            "（VAE パッチのプロセス全域差し替えで参照が汚染され、同値検証が恒真化する）"
        )

    # MUST: 格納 dtype が扱えないターゲットは**明示指定でも**拒否する（既定を絞るだけだと
    # `--target vae_decoder --dtype i8` が通ってしまい、ADR 0019 が排除した形の資産が
    # 黙って生える）。`--verify` も同じ表で見る（検証だけ通る dtype を作らない）。
    allowed = DTYPE_TARGETS[args.dtype]
    requested = [*(args.target or []), *([args.verify] if args.verify is not None else [])]
    outside = sorted({name for name in requested if name not in allowed})
    if outside:
        parser.error(
            f"--dtype {args.dtype} が扱えるターゲットは {', '.join(allowed)} だけ"
            f"（指定: {', '.join(outside)} — ADR 0019 の系列設計）"
        )

    # MUST: 校正のノブは i4 だけに効く。他系列で受けると「校正を外したつもりの f16 資産」の
    # ような読めない指定が通る（f16 / i8 / f32 に校正の経路は 1 本も無い）。
    if args.dtype != "i4" and (args.no_calib or args.calib_prompts != DEFAULT_CALIB_PROMPTS):
        parser.error(
            f"--calib-prompts / --no-calib は --dtype i4 だけに効く（指定は {args.dtype}）"
            " — 校正付き丸めは i4 系列の経路（perf-ledger Q-6）"
        )

    # MUST: 役割別の格納割りも i4 だけに効く（他系列に i4 / i8 の割り方は 1 通りも無い）。
    # 受けてしまうと「adaLN を i8 にしたつもりの f16 資産」という読めない指定が通る。
    if args.dtype != "i4" and args.i4_adaln_i8:
        parser.error(f"--i4-adaln-i8 は --dtype i4 だけに効く（指定は {args.dtype}）")

    # MUST: 校正条件は**モデル別**に名指しさせる（既定を置かない）。step 数と CFG は配布形の
    # 既定から導く（`calib.calib_conditions`）ので、`--model` を忘れた素版の i4 が turbo の
    # 条件（8 step・CFG 1）で黙って校正される形にはできない — 出来上がる資産は格納形も本数も
    # 正しく、`verify_dist` もヘッダ検査も素通りし、`calib_provenance.json` にはその条件が
    # 事実として書かれる（記録が嘘をつくと突き合わせの機構ごと無意味になる）。
    calibrating = args.dtype == "i4" and not args.no_calib
    if calibrating and args.model is None:
        parser.error(
            f"--dtype i4 の校正は --model が要る（選択肢: {', '.join(ANIMA_MODELS)}）"
            " — 校正の step 数 / CFG は配布形の既定から引く（anima.calib.calib_conditions）"
        )
    if args.model is not None and not calibrating:
        knob = f"--dtype {args.dtype}" + ("・--no-calib" if args.no_calib else "")
        parser.error(f"--model は --dtype i4 の校正条件を引くためだけのノブ（指定は {knob}）")

    # MUST: 校正デバイスも校正するときだけ効くノブ（`--calib-prompts` / `--no-calib` と同文）。
    # 受けてしまうと「GPU で焼いたつもりの f16 資産」という読めない指定が通る。
    if args.calib_device != CALIB_SHIPPABLE_DEVICE and not calibrating:
        knob = f"--dtype {args.dtype}" + ("・--no-calib" if args.no_calib else "")
        parser.error(f"--calib-device は --dtype i4 の校正だけに効く（指定は {knob}）")

    # MUST: 使えないデバイスは**校正を始める前**に落とす。校正は数十分〜数時間の経路で、
    # 途中の `Module.to` で落ちると block 外 RTN と i8 丸めまで済んだ状態を捨てることになる。
    if args.calib_device == "cuda" and not torch.cuda.is_available():
        parser.error(
            "--calib-device cuda を指定したが torch から CUDA が見えない"
            "（CPU 版 torch / ドライバ未検出 — CPU へ黙って落ちる分岐は持たない）"
        )

    # MUST: 効かないノブを黙って受けない。S 形は transformer 専用（他 3 ターゲットは解像度に
    # 依らないので共有）で、解像度はケース表 DIT_DYN_RESOLUTIONS が決める（`--resolution` は
    # 1 本も効かない）。受けてしまうと「512 のつもりの資産が実は無関係に生えた」形になる。
    if args.dit_graph == "dyn":
        others = sorted({name for name in requested if name != TARGET_TRANSFORMER})
        if others:
            parser.error(
                f"--dit-graph dyn は {TARGET_TRANSFORMER} 専用（指定: {', '.join(others)}）"
            )
        if args.resolution != RESOLUTION:
            parser.error(
                "--dit-graph dyn では --resolution が効かない"
                f"（golden の解像度は {DIT_DYN_RESOLUTIONS} 固定 — グラフは解像度を持たない）"
            )
        if args.target is None and args.verify is None:
            args.target = [TARGET_TRANSFORMER]

    # MUST: 校正の解像度はグラフの解像度と揃っていること。校正入力は
    # `calib.CALIB_RESOLUTION` 固定（品質裁定が採られた条件を動かさないための設計 — あちらの
    # NOTE）なので、`--resolution` を振ると「別の解像度で選んだ丸め先」を焼き込んだグラフが
    # 黙って生える（活性のトークン数が変わるだけで数値は普通に出る）。dyn は `--resolution` を
    # 上で既に弾いているので、ここに掛かるのは静的形だけ。
    if args.dtype == "i4" and not args.no_calib and args.resolution != calib.CALIB_RESOLUTION:
        parser.error(
            f"--dtype i4 の校正は {calib.CALIB_RESOLUTION}px 固定（指定は {args.resolution}px）"
            " — 品質裁定が採られた条件を動かすと根拠の採り直しになる（calib.CALIB_RESOLUTION）"
        )

    if args.verify is not None:
        report = verify_target(args.verify, args)
        for entry in report:
            print(
                f"{args.verify} {entry['case']}: shapes={entry['shapes']}"
                f" maxdiff={entry['maxdiff']:.3e} bit_exact={entry['bit_exact']}"
            )
        print(f"worst maxdiff = {max(entry['maxdiff'] for entry in report):.3e}")
        print(f"bit_exact all = {all(entry['bit_exact'] for entry in report)}")
        return

    targets = list(dict.fromkeys(args.target if args.target is not None else allowed))
    summaries = [emit_target(target, args, args.out / target) for target in targets]
    print(json.dumps(summaries, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
