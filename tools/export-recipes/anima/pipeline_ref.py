"""Anima の**ホスト側**パイプラインの参照フィクスチャを作る（M1-P4 波 3 / ADR 0016 の段②）。

IR に載るのは `anima/export.py` が書き出す 4 グラフだけで、その外側 — トークナイズ /
スケジューラ（sigmas）/ timestep 埋め込み表 / CFG / Euler 更新 / latent 逆正規化 / 512
パディング — は全てホストコードになる。**そのホスト側の「数の正」がここ**で、Deno 側の
通しチェーン E2E（`packages/runtime/tests/e2e_anima_test.ts`）は TS で書いたグルーを
このフィクスチャの対応値と突き合わせてから使う。

正本は diffusers 0.39 の `modular_pipelines/anima/`（encoders.py / before_denoise.py /
denoise.py / decoders.py）。この台本はその 4 ブロックの逐語的な書き下しで、**パッチ層を
通さない素の diffusers 経路**で参照を採る（`anima.patch` の同値は
`anima/export.py --verify` が別に測る — 検証網を独立に保つ。パッチ層をここでも通すと、
パッチのバグが参照とテスト対象の両方に同じ形で乗り、差 0 のまま素通りする）。

出力（既定 `<repo>/outputs/series/anima-pipeline/`、`--dtype f16` ならその `-f16` 版）:

    pipeline.safetensors   各段のテンソル（下の `_TENSOR_ROLES` が用途の索引）
    pipeline.json          プロンプト・step 数・shift・CFG 係数・LoRA（`lora` / `lora_scale`）
                           ・shape 一覧

MUST: 出力先を `models/anima-turbo/`（配布形）直下にしない。あちらは manifest が宣言した
ファイルだけを並べて**そのまま HF へ上げる**木で、宣言外のファイルが混ざると `verify_dist` が
止まる（音声デモで `outputs/sbv2-demo/` を分けたのと同じ理由）。

MUST: 圧縮系列のフィクスチャは **4 コンポーネントとも** fake-quant してから採る（ADR 0006）。
1 つでも素の重みのまま残すと、その段だけ参照が別のモデルの数になり、通しチェーンの差が
「量子化誤差 + 実装誤差」の合成になる。丸めは各モデルのロード直後に掛ける。

MUST: `--dtype i8` は **DiT だけ i8・他 3 つは f16**（`COMPONENT_DTYPES`）。資産系列が
`anima-i8/transformer` + `anima-f16/{text_encoder,text_conditioner,vae_decoder}`
という構成（ADR 0019）なので、フィクスチャの丸めもその構成と 1 対 1 に対応させる。全部を
i8 にすると text 経路の参照だけが実行される資産と別のモデルの数になる。

    uv run python -m anima.pipeline_ref
    uv run python -m anima.pipeline_ref --steps 32 --ref-steps 2
    uv run python -m anima.pipeline_ref --dtype f16
    uv run python -m anima.pipeline_ref --dtype i8
    uv run python -m anima.pipeline_ref --dtype f16 --steps 10 --ref-steps 10 \
        --guidance-scale 1.0 --lora ../../inputs/anima/anima-turbo-lora-v0.2.safetensors \
        --out ../../outputs/series/anima-pipeline-turbo-f16

`--act-quant` は **w8a8**（`SessionOptions.linearCompute: "a8"`）の鏡像で、DiT の適格
`nn.Linear` の入力を per-token i8 へ fake-quant してから参照を採る（数値仕様の正本は
`karume.act_quant`）。重み側の i8 と対なので `--dtype i8` との併用を強制する:

    uv run python -m anima.pipeline_ref --dtype i8 --act-quant --steps 10 \
        --ref-steps 10 --guidance-scale 1.0 \
        --lora ../../inputs/anima/anima-turbo-lora-v0.2.safetensors \
        --out ../../outputs/series/anima-pipeline-turbo-i8a8

`--resolution` は **WxH**（正方は略記できる）。非正方の参照は `--dit-graph dyn` の DiT と
`--vae-tiling` の VAE を突き合わせる先で（#23）、綴りはデモの `--resolution` と同じ:

    uv run python -m anima.pipeline_ref --dtype f16 --steps 10 --ref-steps 2 \
        --guidance-scale 1.0 --resolution 1344x768 \
        --lora ../../inputs/anima/anima-turbo-lora-v0.2.safetensors \
        --out ../../outputs/series/anima-pipeline-turbo-f16-1344x768

DiT の forward は既定で 4 回（2 step × cond/uncond）、CFG=1 の turbo 系列では uncond を
畳むので `--ref-steps` と同数（上の例なら 10 回）。CPU f32 なので数分かかる。ピーク RAM は
DiT の 7.29GiB 常駐 + 活性で 12GiB 級（recon §7）。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _shared.paths import SERIES_ROOT
from karume.act_quant import attach_act_quant, detach_act_quant
from karume.convert import normalize_boundary_tensor
from karume.quantize import fake_quant_int8, round_weights_to_f16

from .resolution import parse_resolution, resolution_meta

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"
#: この台本が参照を採れる系列。**`export.py` の `WEIGHT_DTYPES` から引かない** MUST — あちらは
#: 「資産として書ける格納 dtype」の集合で、i4 のように**参照経路を持たない**（下の 3 表に席が
#: 無い）系列が入る。結合すると argparse の受理だけが広がり、`component_dtype` / `DEFAULT_OUTS`
#: の素の KeyError まで落ちない — text_encoder のロードまで数分走った末に診断文言ゼロで死ぬ形で、
#: fail loudly（理由を挙げて落ちる）を満たさない。席を足すときは下の 3 表と同時に足す。
REF_DTYPES: tuple[str, ...] = ("f32", "f16", "i8")
#: 生成物の既定の置き場（格納 dtype 別）。**配布形 `models/` とは別の系列側**（上の MUST）。
DEFAULT_OUTS = {
    "f32": SERIES_ROOT / "anima-pipeline",
    "f16": SERIES_ROOT / "anima-pipeline-f16",
    "i8": SERIES_ROOT / "anima-pipeline-i8",
}

#: 系列 → コンポーネントごとの実効格納 dtype（資産系列との 1 対 1 対応 — 上の MUST）。
#: キーは `_fake_quant` に渡すラベルで、i8 系列だけが DiT と他 3 つで分かれる。
COMPONENT_DTYPES = {
    "f32": {},
    "f16": {},
    "i8": {"transformer": "i8"},
}
#: `COMPONENT_DTYPES` に載らないコンポーネントの既定（f32 系列は丸めない）。
_FALLBACK_DTYPE = {"f32": "f32", "f16": "f16", "i8": "f16"}


def component_dtype(dtype: str, label: str) -> str:
    """コンポーネント 1 つの実効格納 dtype（系列の構成をここ 1 箇所で決める）。"""
    return COMPONENT_DTYPES[dtype].get(label, _FALLBACK_DTYPE[dtype])


def _fake_quant(dtype: str, model: torch.nn.Module, label: str) -> None:
    """系列に応じて重みを格納表現可能値へ丸める（ADR 0006 — 参照採取より前 MUST）。"""
    effective = component_dtype(dtype, label)
    if effective == "f32":
        return
    report = fake_quant_int8(model) if effective == "i8" else round_weights_to_f16(model)
    print(f"[fake-quant] {label} ({effective}): {report.describe()}", flush=True)


def _apply_lora(path: Path | None, scale: float, model: torch.nn.Module, prefix: str) -> None:
    """turbo 等の蒸留 LoRA を export 前に焼き込む（`anima/export.py` と同じ規律）。

    MUST: `_fake_quant` より前に呼ぶ（ADR 0016 と同じ順序制約 — 丸め後に焼くと ΔW が丸めの
    格子を外れ、格納時の再丸めが参照と食い違う）。text_conditioner 側は実測で lora_B が
    全ゼロ（noop）と確認済みなので、このパイプラインでは transformer にのみ適用する。
    """
    if path is None:
        return
    from .lora import fuse_lora, load_lora_state_dict

    state = load_lora_state_dict(path)
    report = fuse_lora(model, state, prefix, scale)
    print(f"[lora] {prefix}: {report.describe()}", flush=True)


#: Anima はアニメ画像特化（danbooru 系タグ）。固定 1 本・英語。
PROMPT = (
    "1girl, solo, long hair, blue eyes, school uniform, cherry blossoms, "
    "outdoors, smile, upper body, masterpiece, best quality"
)
#: MUST: **空文字列にしてはいけない** — 空プロンプトの T5 id 列は長さ 1 になり、conditioner の
#: 受理集合（`Dim("Ttgt", min=2)`）から外れて export 済みグラフに食わせられなくなる。
#: パイプライン既定は `""` だが、モデルカードの推奨も CFG 4〜5 + ネガティブ指定。
NEGATIVE_PROMPT = "low quality, worst quality, blurry, bad anatomy, jpeg artifacts"

NUM_TRAIN_TIMESTEPS = 1000  # scheduler_config.json
SHIFT = 3.0  # 〃（use_dynamic_shifting=false なので静的 shift）
GUIDANCE_SCALE = 4.0  # ClassifierFreeGuidance の既定（encoders.py / denoise.py）
MIN_SEQUENCE_LENGTH = 512  # AnimaTextConditioner.config.min_sequence_length
SPATIAL_COMPRESSION = 8
LATENT_CHANNELS = 16

#: `latents_init` の seed。**固定**（再生成でバイト一致させる — golden の規約と同じ）。
SEED = 20260802

#: フィクスチャのテンソルの用途索引（JSON メタにそのまま載る。Deno 側の読み手向け）。
_TENSOR_ROLES: dict[str, str] = {
    "sigmas": "FlowMatchEuler の sigma 列（末尾に終端 0）。TS 実装のパリティ対象",
    "timesteps_proj": "全 step 分の timestep 埋め込み表 [steps, 2048]。DiT の入力 2 本目",
    "latents_mean": "VAE の per-channel latent 平均（逆正規化の定数）",
    "latents_std": "VAE の per-channel latent 標準偏差（逆正規化の定数）",
    "latents_init": "初期ノイズ [1,16,h,w]（seed 固定）。チェーンの起点",
    "qwen_input_ids": "Qwen2 BPE の id 列。text_encoder の入力",
    "t5_input_ids": "T5 Unigram の id 列。text_conditioner の入力 2 本目",
    "qwen_hidden_states": "text_encoder の出力（全 1 マスク乗算後 = 恒等）",
    "encoder_hidden_states": "conditioner 出力を 512 へゼロパディングしたもの。DiT の入力 3 本目",
    "noise_cond_stepNNNN": "step NNNN の DiT 出力（cond 側）。CFG グルーのパリティ対象",
    "noise_uncond_stepNNNN": "step NNNN の DiT 出力（uncond 側）。同上。"
    "guidance_scale=1.0 の fixture ではキー自体が存在しない（uncond 分岐を計算しない）",
    "latents_stepNNNN": "step NNNN の Euler 更新後 latent",
    "latents_denorm": "最終 latent の逆正規化後。VAE decoder の入力",
    "image": "VAE decode 出力 [1,3,H,W]（`_decode` の clamp(-1,1) 込み = IR 側と同じ位置）",
}


def sigma_schedule(steps: int, shift: float) -> np.ndarray:
    """`AnimaSetTimestepsStep` + `FlowMatchEulerDiscreteScheduler.set_timesteps` の再現。

    `sigmas=linspace(1, 1/N, N)` を渡す経路なので `_sigma_to_t` は通らず、静的 shift の
    `shift·s / (1 + (shift−1)·s)` だけが効く。末尾に終端 0 を 1 つ足すのは `step()` が
    `sigmas[i+1]` を読むため。

    MUST: f32 化の位置を scheduler と揃える（`astype` を後ろへ動かすと f64 で計算して
    最終桁が変わり、TS 側の `Math.fround` 逐次実装とビット一致しなくなる）。
    """
    sigmas = np.linspace(1.0, 1.0 / steps, steps).astype(np.float32)
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    return np.concatenate([sigmas, np.zeros(1, dtype=np.float32)])


@dataclass(frozen=True)
class TextStack:
    """テキスト前段のロード済み一式（トークナイザ 2 本 + Qwen3 + conditioner）。

    まとめて持つのは「1 回ロードして複数プロンプトを通す」呼び出し（`anima.calib` の校正入力）
    のため — プロンプトごとに `from_pretrained` を叩くと Qwen3 のロードがその回数だけ走る。
    """

    tokenizer: Any
    t5_tokenizer: Any
    encoder: torch.nn.Module
    conditioner: torch.nn.Module


def load_text_stack(repo: str, dtype: str) -> TextStack:
    """テキスト前段をロードして系列の格納 dtype へ丸める（{@link encode_prompt} の前提）。

    MUST: 丸めはここ（参照を採る前）— 後ろへ動かすと参照だけが元の重みで計算される。
    """
    from diffusers import AnimaTextConditioner
    from transformers import AutoTokenizer, Qwen3Model

    tokenizer = AutoTokenizer.from_pretrained(repo, subfolder="tokenizer")
    t5_tokenizer = AutoTokenizer.from_pretrained(repo, subfolder="t5_tokenizer")
    encoder = Qwen3Model.from_pretrained(
        repo, subfolder="text_encoder", dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    conditioner = AnimaTextConditioner.from_pretrained(repo, subfolder="text_conditioner")
    conditioner.to(torch.float32).eval()
    _fake_quant(dtype, encoder, "text_encoder")
    _fake_quant(dtype, conditioner, "text_conditioner")
    return TextStack(
        tokenizer=tokenizer, t5_tokenizer=t5_tokenizer, encoder=encoder, conditioner=conditioner
    )


def encode_prompt(stack: TextStack, max_len: int, text: str) -> dict[str, torch.Tensor]:
    """1 プロンプトを `AnimaTextEncoderStep` + `AnimaTextConditioningStep` で通す。

    トークナイザは repo 同梱の `tokenizer.json`（Qwen2 BPE / T5 Unigram）。`padding="longest"`・
    単一プロンプトなのでマスクは**全 1** になり、`prompt_embeds * mask` も conditioner の
    マスク乗算もどちらも恒等 — ラッパ（`anima.patch`）がマスクを持たない根拠がここ。

    キーは接頭辞なし（`neg_` のような役割の綴りは呼び出し側が付ける）。
    """
    qwen = stack.tokenizer(
        [text], padding="longest", max_length=max_len, truncation=True, return_tensors="pt"
    )
    t5 = stack.t5_tokenizer(
        [text], padding="longest", max_length=max_len, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        hidden = stack.encoder(
            input_ids=qwen.input_ids,
            attention_mask=qwen.attention_mask,
            output_hidden_states=False,
        ).last_hidden_state
        hidden = hidden * qwen.attention_mask.to(hidden).unsqueeze(-1)
        embeds = stack.conditioner(
            source_hidden_states=hidden,
            target_input_ids=t5.input_ids,
            target_attention_mask=t5.attention_mask,
            source_attention_mask=qwen.attention_mask,
        )
    if embeds.shape[1] != MIN_SEQUENCE_LENGTH:
        raise ValueError(
            f"conditioner 出力長 {embeds.shape[1]} が {MIN_SEQUENCE_LENGTH} でない"
            f"（T5 id 列 {t5.input_ids.shape[1]} が長すぎる）— DiT の入力形と合わない"
        )
    # MUST: id 列は IR の意味論 dtype（i32）へ落として書く — 変換点は 1 箇所（ADR 0009）。
    return {
        "qwen_input_ids": normalize_boundary_tensor(qwen.input_ids, "qwen id 列"),
        "t5_input_ids": normalize_boundary_tensor(t5.input_ids, "t5 id 列"),
        "qwen_hidden_states": hidden.contiguous(),
        "encoder_hidden_states": embeds.contiguous(),
    }


def encode_text(repo: str, max_len: int, dtype: str) -> dict[str, torch.Tensor]:
    """固定プロンプト 2 本（正 / ネガティブ）をフィクスチャのキー空間へ落とす。"""
    stack = load_text_stack(repo, dtype)
    out: dict[str, torch.Tensor] = {}
    for tag, text in (("", PROMPT), ("neg_", NEGATIVE_PROMPT)):
        for key, value in encode_prompt(stack, max_len, text).items():
            out[f"{tag}{key}"] = value
    return out


def timesteps_proj_table(
    repo: str, sigmas: np.ndarray, dtype: str, lora: Path | None, lora_scale: float
) -> tuple[torch.Tensor, torch.nn.Module]:
    """全 step 分の timestep 埋め込み表 `[steps, 2048]` と、ロード済み DiT を返す。

    表は `AnimaDit` の入力 2 本目そのもの（`anima.patch` が timestep 埋め込みをグラフ入力へ
    昇格させた先）。DiT を一緒に返すのは、直後の参照 denoise で同じインスタンスを使い回して
    7.29GiB のロードを 1 回で済ませるため。

    `AnimaLoopBeforeDenoiser` の `timestep = timesteps[i] / num_train_timesteps` をそのまま
    踏む（`timesteps = sigmas · num_train_timesteps` なので値としては sigma に戻る）。

    NOTE（実測・当初「f32 の往復で最終桁が変わる」と書いたのは**誤りだったので訂正**）:
    steps=32 / shift=3 の 32 本とも `f32(f32(s·1000)/1000) == s` で、往復は 1 ビットも
    変えない。順序を写してあるのは上流の式と 1 対 1 で追えるようにするためで、丸めの実測に
    基づく制約ではない。
    """
    from diffusers import CosmosTransformer3DModel

    model = CosmosTransformer3DModel.from_pretrained(repo, subfolder="transformer")
    model.to(torch.float32).eval()
    # MUST: LoRA は丸めより前に焼く（anima/export.py と同じ順序制約 — ADR 0016）。
    _apply_lora(lora, lora_scale, model, "transformer")
    # MUST: 表を作る**前**に丸める。time_proj は素の正弦波で重みを持たないが、順序を
    # 「参照より前」に統一しておかないと、将来 time_embed 側に重みが増えた時に黙ってずれる。
    _fake_quant(dtype, model, "transformer")
    rows = []
    for sigma in sigmas[:-1]:
        t = torch.tensor([sigma * NUM_TRAIN_TIMESTEPS], dtype=torch.float32)
        with torch.no_grad():
            rows.append(model.time_embed.time_proj(t / NUM_TRAIN_TIMESTEPS).to(torch.float32))
    return torch.cat(rows, dim=0), model


def reference_steps(
    model: torch.nn.Module,
    latents: torch.Tensor,
    embeds: torch.Tensor,
    negative_embeds: torch.Tensor,
    sigmas: np.ndarray,
    size: tuple[int, int],
    ref_steps: int,
    guidance: float,
) -> dict[str, torch.Tensor]:
    """`AnimaDenoiseStep` を `ref_steps` 回。CFG は `uncond + scale·(cond − uncond)`。

    latent は素の経路では `(B,C,1,h,w)`。`padding_mask` はピクセル解像度のゼロ
    （パイプライン既定）で、Karume 側は `anima.patch.AnimaDit` がゼロ定数チャネルへ畳む。

    cond / uncond の生の DiT 出力も残す — これが無いと Deno 側で CFG と Euler の
    ホストグルーを**単体でパリティ検査できず**、DiT の誤差と混ざった形でしか見られない。

    MUST: `guidance == 1.0` のとき uncond 分岐を**計算しない**（Turbo LoRA は CFG=1 で運用
    する設計 — recon L3）。`uncond + 1.0·(cond − uncond)` は数学的には `cond` に退化するが、
    素朴に両方計算すると①不要な DiT 呼び出しが 1 回増える②浮動小数の丸め順序で `cond` と
    ビット一致しなくなる。実運用の「単一呼び出し」を忠実に映すため、この場合は
    `noise_uncond_stepNNNN` キー自体を書かない — fixture のキー構成が「uncond 分岐が消えた」
    ことを直接示す形になる。
    """
    # MUST: `padding_mask` は**ピクセル解像度**（latent ではない）で、軸は `[1,1,H,W]`。
    # 非正方では W と H の取り違えが shape 検査を素通りしない側の唯一の場所なので、
    # `size` を `(width, height)` で受けてここで入れ替える（呼び出し側は WxH の綴り順）。
    width, height = size
    padding_mask = latents.new_zeros(1, 1, height, width)
    skip_uncond = guidance == 1.0
    out: dict[str, torch.Tensor] = {}
    x = latents
    for index in range(ref_steps):
        started = time.perf_counter()
        timestep = torch.tensor(
            [sigmas[index] * NUM_TRAIN_TIMESTEPS / NUM_TRAIN_TIMESTEPS], dtype=torch.float32
        )
        branches = (embeds,) if skip_uncond else (embeds, negative_embeds)
        preds = []
        for encoder_hidden_states in branches:
            with torch.no_grad():
                preds.append(
                    model(
                        hidden_states=x,
                        timestep=timestep,
                        encoder_hidden_states=encoder_hidden_states,
                        padding_mask=padding_mask,
                        return_dict=False,
                    )[0]
                )
        tag = f"step{index + 1:04d}"
        if skip_uncond:
            (cond,) = preds
            noise_pred = cond
        else:
            cond, uncond = preds
            noise_pred = uncond + guidance * (cond - uncond)
            out[f"noise_uncond_{tag}"] = uncond.squeeze(2).contiguous()
        x = x + float(sigmas[index + 1] - sigmas[index]) * noise_pred
        out[f"noise_cond_{tag}"] = cond.squeeze(2).contiguous()
        out[f"latents_{tag}"] = x.squeeze(2).contiguous()
        print(
            f"[denoise] step {index + 1}/{ref_steps} sigma={sigmas[index]:.4f}"
            f" ({time.perf_counter() - started:.1f}s)",
            flush=True,
        )
    return out


def denormalize_latents(vae: torch.nn.Module, latents4: torch.Tensor) -> torch.Tensor:
    """`AnimaVaeDecoderStep` の逆正規化（rank4 → rank5）。

    MUST: decoders.py は std の**逆数を作って割る**。`latents * std` に直すと最終桁が
    変わる（TS 側の実装も同じ順序で書く）。
    """
    z = latents4.unsqueeze(2)
    mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
    return z / inv_std + mean


def decode_latents(repo: str, latents4: torch.Tensor, dtype: str) -> dict[str, torch.Tensor]:
    """`AnimaVaeDecoderStep` の再現（逆正規化 → decode）。**パッチ前**の素の経路で採る。

    `image` は `vae.decode` の戻り値そのもの。`AutoencoderKLQwenImage._decode` は最後に
    `torch.clamp(-1, 1)` を掛けており（`AnimaVaeDecoder` が焼き込んでいるのはこの clamp で、
    postprocess 由来ではない）、**フィクスチャ側も IR 側も同じ位置で clamp 済み**になる —
    突合にホスト側の clamp 鏡像は要らない。
    """
    from diffusers import AutoencoderKLQwenImage

    vae = AutoencoderKLQwenImage.from_pretrained(repo, subfolder="vae")
    vae.to(torch.float32).eval()
    _fake_quant(dtype, vae, "vae")
    z = denormalize_latents(vae, latents4)
    with torch.no_grad():
        image = vae.decode(z, return_dict=False)[0][:, :, 0]
    return {
        "latents_mean": torch.tensor(vae.config.latents_mean),
        "latents_std": torch.tensor(vae.config.latents_std),
        "latents_denorm": z.squeeze(2).contiguous(),
        "image": image.contiguous(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="出力先（既定は --dtype ごとに outputs/series/anima-pipeline{,-f16,-i8}/）",
    )
    parser.add_argument(
        "--dtype",
        choices=REF_DTYPES,
        default="f32",
        help="圧縮系列は 4 コンポーネントとも fake-quant してから参照を採る（i8 は DiT のみ i8）",
    )
    parser.add_argument("--steps", type=int, default=32, help="sigma スケジュールの総 step 数")
    parser.add_argument(
        "--ref-steps", type=int, default=2, help="torch で実行して参照 latent を残す step 数"
    )
    parser.add_argument(
        "--resolution",
        default="512",
        help="WxH（例 1344x768）。正方は略記できる（512 = 512x512）"
        "— 綴りはデモの --resolution と同じ",
    )
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=GUIDANCE_SCALE,
        help="CFG のガイダンス係数（1.0 で uncond 分岐を計算しない — Turbo LoRA の運用値）",
    )
    parser.add_argument(
        "--lora",
        type=Path,
        default=None,
        help="export 前に DiT へ焼き込む LoRA（text_conditioner 側は既知の noop なので適用しない）",
    )
    parser.add_argument("--lora-scale", type=float, default=1.0, help="LoRA の倍率")
    parser.add_argument(
        "--act-quant",
        action="store_true",
        help="DiT の適格 linear の入力を per-token i8 へ fake-quant する"
        "（ランタイムの linearCompute:'a8' の鏡像 — 重み側は --dtype i8 と併用する）",
    )
    args = parser.parse_args()
    try:
        width, height = parse_resolution(args.resolution)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.act_quant and args.dtype != "i8":
        raise SystemExit(
            f"--act-quant は --dtype i8 と併用する（指定は {args.dtype}）— "
            "活性 i8 は i8 常駐重みの linear にしか効かない（設計 §4.1）"
        )
    if args.out is None:
        args.out = DEFAULT_OUTS[args.dtype]

    if args.ref_steps > args.steps:
        raise SystemExit(f"--ref-steps({args.ref_steps})が --steps({args.steps})を超えている")

    sigmas = sigma_schedule(args.steps, SHIFT)
    tensors: dict[str, torch.Tensor] = {"sigmas": torch.from_numpy(sigmas)}
    print(f"[sigmas] {sigmas[0]:.4f} … {sigmas[-2]:.4f} → 0 ({args.steps} steps)", flush=True)

    tensors.update(encode_text(args.repo, args.max_sequence_length, args.dtype))
    print(
        f"[text] qwen={list(tensors['qwen_input_ids'].shape)}"
        f" t5={list(tensors['t5_input_ids'].shape)}"
        f" neg qwen={list(tensors['neg_qwen_input_ids'].shape)}"
        f" neg t5={list(tensors['neg_t5_input_ids'].shape)}",
        flush=True,
    )

    # `AnimaPrepareLatentsStep.prepare_latents` と同じ shape・同じ dtype で引く。
    # MUST: 軸の順は `[..., H, W]`（綴りの WxH とは逆）。非正方でここを入れ替えると
    # 要素数は合ったまま latent が転置され、参照だけが別の絵になる。
    latents = torch.randn(
        (1, LATENT_CHANNELS, 1, height // SPATIAL_COMPRESSION, width // SPATIAL_COMPRESSION),
        generator=torch.Generator().manual_seed(SEED),
        dtype=torch.float32,
    )
    tensors["latents_init"] = latents.squeeze(2).contiguous()

    proj, model = timesteps_proj_table(args.repo, sigmas, args.dtype, args.lora, args.lora_scale)
    tensors["timesteps_proj"] = proj
    # w8a8（`SessionOptions.linearCompute: "a8"`）の鏡像。**DiT だけ**に掛ける — ランタイム
    # 側も transformer の Session にしかノブを立てないので、text / VAE まで掛けると参照だけが
    # 別のモデルの数になる（`_fake_quant` の COMPONENT_DTYPES と同じ規律）。
    act_handles: list[object] = []
    if args.act_quant:
        act_handles, attached = attach_act_quant(model)
        # MUST: 本数を出す。0 本のまま参照を採ると「w8a8 のつもりで w8 の数を採った」ことに
        # 気づけない（ADR 0006 の診断常設と同じ流儀）。
        print(f"[act-quant] transformer: 適格 linear {attached} 本に per-token i8 を適用")
        if attached == 0:
            raise SystemExit("--act-quant を指定したが適格 linear が 0 本（適格判定の破れ）")
    try:
        tensors.update(
            reference_steps(
                model,
                latents,
                tensors["encoder_hidden_states"],
                tensors["neg_encoder_hidden_states"],
                sigmas,
                (width, height),
                args.ref_steps,
                args.guidance_scale,
            )
        )
    finally:
        detach_act_quant(act_handles)
    del model

    tensors.update(
        decode_latents(args.repo, tensors[f"latents_step{args.ref_steps:04d}"], args.dtype)
    )
    image = tensors["image"]
    print(f"[decode] image={list(image.shape)} range=[{image.min():.4f}, {image.max():.4f}]")

    from safetensors.torch import save_file

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out / "pipeline.safetensors"))
    meta: dict[str, Any] = {
        "repo": args.repo,
        "dtype": args.dtype,
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "steps": args.steps,
        "ref_steps": args.ref_steps,
        **resolution_meta(width, height),
        "seed": SEED,
        "shift": SHIFT,
        "guidance_scale": args.guidance_scale,
        "act_quant": args.act_quant,
        "lora": str(args.lora) if args.lora is not None else None,
        "lora_scale": args.lora_scale if args.lora is not None else None,
        "num_train_timesteps": NUM_TRAIN_TIMESTEPS,
        "min_sequence_length": MIN_SEQUENCE_LENGTH,
        "image_range": [float(image.min()), float(image.max())],
        "roles": _TENSOR_ROLES,
        "tensors": {name: list(value.shape) for name, value in tensors.items()},
    }
    (args.out / "pipeline.json").write_text(
        json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"fixture OK: {len(tensors)} tensors → {args.out}", flush=True)


if __name__ == "__main__":
    main()
