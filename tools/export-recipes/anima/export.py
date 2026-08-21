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
丸めは**素の RTN**（校正なし）— 格納形は丸め方式に依らないので、速度実測はこれで足りる。

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
from anima.distribution import LORA_PROVENANCE_FILE
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
    channel_rows,
    fake_quant_int4,
    fake_quant_int8,
    iter_quant_targets,
    round_weights_to_f16,
)

from . import patch

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
#: conv 系は i4 の展開経路が無い（conv1d は `groups == 1` 限定）ので入れない。
I4_MODULE_TYPES: tuple[type[nn.Module], ...] = (nn.Linear, nn.Embedding)

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
    scales, dtype_overrides = _fake_quant(args, module, TARGET_TRANSFORMER)
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
    scales, dtype_overrides = _fake_quant(args, module, TARGET_TRANSFORMER)
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


def _fake_quant_i4(
    model: nn.Module, target: str
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, str]]:
    """混成 i4 系列の丸め（適格 = 素の RTN i4 g32・残り = i8 per-channel）。

    2 つの述語は {@link _i4_module_names} から**排他に**割る（`quantize.py` の混成 MUST）。
    返す override は「i4 の scale 台帳のキー全部を i4 に振る」写像で、emit 側は明示指定を
    満たせなければ fail loudly する。

    GPTQ（校正付き丸め）は結線しない — 格納形もカーネルも丸め値に依らないので、この段の
    速度実測には素の RTN で足りる（品質を採るときに別途決める）。
    """
    i4_names = _i4_module_names(model)
    report = fake_quant_int4(model, include=lambda name: name in i4_names, op_types=I4_MODULE_TYPES)
    print(
        f"[fake-quant] {target}: 適格な重みを i4 group へ丸めた（RTN） — {report.describe()}",
        flush=True,
    )
    # MUST: 丸めた集合 = 格納集合（override のキー）。ずれるのは「適格と数えたのに丸まって
    # いない」形で、i4 席に i8 の重みが混ざったまま緑になる（サイズだけが静かに戻る）。
    expected = {f"{name}.weight" for name in i4_names}
    if set(report.scales) != expected:
        raise AssertionError(
            f"i4 適格 {len(i4_names)} 本に対し丸めたのは {len(report.scales)} 本"
            f"（過不足: {sorted(set(report.scales) ^ expected)[:3]}）"
        )
    int8 = fake_quant_int8(model, include=lambda name: name not in i4_names)
    print(f"[fake-quant] {target}: 残りは i8 per-channel — {int8.describe()}", flush=True)
    return {**int8.scales, **report.scales}, dict.fromkeys(report.scales, "i4")


def _fake_quant(
    args: argparse.Namespace, model: nn.Module, target: str
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, str]]:
    """格納 dtype の表現可能値へ重みを丸め、scale 台帳と 1 本単位の格納指定を返す（ADR 0006）。

    MUST: 各 builder で **`reference` の採取より前**・`--lora` の焼き込みより**後**に呼ぶ。
    前後を逆にすると①参照だけが元の重みで計算されて E2E の差に量子化誤差が混ざる
    ②焼き込んだ ΔW が丸めを外して格納時の再丸めが golden との対応を壊す。

    MUST（i8 / i4 のみ）: **export する `nn.Module` そのもの**に当てる。scale 台帳のキーは
    ここで見た FQN で、safetensors のテンソルキー（= `torch.export` が見る FQN）と同じで
    なければ emit 側の突合が空振りする（`id()` 突合は禁止 — ADR 0006）。
    """
    if args.dtype == "f32":
        return {}, {}
    if args.dtype == "i8":
        report = fake_quant_int8(model)
        print(f"[fake-quant] {target}: i8 per-channel へ丸めた — {report.describe()}", flush=True)
        return report.scales, {}
    if args.dtype == "i4":
        return _fake_quant_i4(model, target)
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

    MUST: 焼いていない側で**古い記録を消す**（全域関数）。`emit_target` は系列ディレクトリを
    掃除しないので、`--lora` 付きで採った系列へ `--lora` 無しで採り直すと、重みだけ素に戻って
    記録が前回のまま生き残る。`assert_lora_provenance` は「記録が在る × sha 一致」しか見ない
    ので、その状態は「取り下げ」を素通しして配布 README に嘘の帰属を印字させる。記録の存在が
    「今の `model.safetensors` に焼いた」と同値であることが、この機構の唯一の拠り所。
    """
    if args.lora is None or target not in LORA_PREFIXES:
        (out_dir / LORA_PROVENANCE_FILE).unlink(missing_ok=True)
        return None
    record = {"file": args.lora.name, "sha256": sha256_file(args.lora)}
    (out_dir / LORA_PROVENANCE_FILE).write_text(
        json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return LORA_PROVENANCE_FILE


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
    """1 ターゲットの IR コンテナと golden io を書き、要約を返す。"""
    started = time.perf_counter()
    component = BUILDERS[target](args, False)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph = export_to_file(
        component.module,
        component.example,
        out_dir / MODEL_FILE,
        dynamic_shapes=component.dynamic_shapes,
        symbol_names=component.symbol_names,
        weight_dtype=BASE_WEIGHT_DTYPES[args.dtype],
        weight_scales=component.weight_scales,
        weight_dtype_overrides=component.weight_dtype_overrides,
        preserved=TARGET_PRESERVED[target],
    )
    written = _write_io(component, graph, out_dir)
    if component.host_tables:
        # ホスト素表は IR コンテナの外に置く（グラフが使わないテンソルを model.safetensors へ
        # 混ぜると、initializer とテンソルキーの 1:1 検査〈verify_model〉が壊れる）。
        save_file(dict(component.host_tables), str(out_dir / ROPE_BASE_FILE))
        written.append(ROPE_BASE_FILE)
    provenance = _write_lora_provenance(args, target, out_dir)
    if provenance is not None:
        written.append(provenance)
    breakdown = storage_breakdown(graph)
    return {
        "target": target,
        "dir": str(out_dir),
        "dtype": args.dtype,
        "dit_graph": args.dit_graph if target == TARGET_TRANSFORMER else "static",
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
        "model_bytes": (out_dir / MODEL_FILE).stat().st_size,
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
    args = parser.parse_args(argv)
    if args.out is None:
        root = DEFAULT_OUT_ROOTS[args.dtype]
        args.out = root.with_name(f"{root.name}{DYN_SUFFIX}") if args.dit_graph == "dyn" else root

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
