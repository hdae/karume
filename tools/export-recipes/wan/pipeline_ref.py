"""Wan2.1 の参照パイプライン（diffusers の `WanPipeline` を CPU f32 で回す — ADR 0118 決定 4 / 5）。

数の正本は diffusers 0.39.0 の `WanPipeline`（`pipeline_wan.py`）。ここはそれを**素のまま**組んで
回す共通部で、DiT（段 2 / 3）と VAE（段 4 / 5）の参照出力を作る台本はこの上に乗る。パッチ層は
通さない（`anima/pipeline_ref.py` と同じ規律 — パッチを参照側にも通すと、パッチのバグが参照と
テスト対象の両方に同じ形で乗り、差 0 のまま素通りする）。

組み方:

- **CPU f32**（DiT も VAE も `torch_dtype=torch.float32`）。上流の bf16 autocast は使わない
  （WebGPU に bf16 が無い — 決定 3）。
- **`text_encoder=None` / `tokenizer=None`**。umT5-XXL（f32 で約 22.7 GB）は読まない — 開発機の
  ホスト RAM 31 GiB に DiT と同居させない（決定 4）。テキスト文脈は `prompt_embeds` /
  `negative_prompt_embeds` に**有効長の埋め込み `[L_valid, 4096]`** を渡し、ここで上流と同じ形
  （有効長の後ろをゼロで埋めた `[1, 512, 4096]` — `_get_t5_prompt_embeds`）にして注入する。
- **UniPC は scheduler の config をそのまま**使う（`flow_shift` 3.0 — 決定 5。参照を作る側でだけ
  値を書き換えると、`from_pretrained` だけで同じ参照を作れる性質が消える）。
- 初期ノイズは外から注入する（`latents=` — 決定 5。torch CPU の `randn` で作る側は呼び手）。

MUST: MPS が見える機では組まない（{@link assert_no_mps}）。diffusers の `WanRotaryPosEmbed` は
MPS が使える機でだけ RoPE の表を float32 で作り、他は float64 で作る（`transformer_wan.py:375`）。
参照と素表が機に依って変わらないよう、float64 の側に固定する（決定 3）。

MUST: diffusers / transformers は関数の中で import する（`wan` グループは既定の sync に入らない —
`tests/test_optional_group_imports.py`）。

    uv run --group wan python -m wan.pipeline_ref --smoke   # 読み込みと 1 ステップの疎通
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import TYPE_CHECKING, Any

import torch

from wan.sources import DEFAULT_MODEL, local_snapshot

if TYPE_CHECKING:
    from diffusers import WanPipeline

#: 上流のテキスト文脈の長さ（`WanPipeline.__call__` の `max_sequence_length` の既定）。
MAX_SEQUENCE_LENGTH = 512

#: umT5-XXL の隠れ次元（transformer の config の `text_dim`）。
TEXT_DIM = 4096

#: VAE の時間 / 空間の縮小率（vae の config の `scale_factor_temporal` / `scale_factor_spatial`）。
TEMPORAL_SCALE = 4
SPATIAL_SCALE = 8

#: 参照出力を作る設定（決定 5）。shift は scheduler の config の値を使うのでここには置かない。
GUIDANCE_SCALE = 5.0
NUM_INFERENCE_STEPS = 50


class MpsVisibleError(RuntimeError):
    """MPS が見える機で参照を組もうとした（RoPE の表が float32 になる — 決定 3）。"""


def assert_no_mps() -> None:
    """RoPE の表が float64 で作られる条件（MPS 不在）を確かめる。"""
    if torch.backends.mps.is_available():
        raise MpsVisibleError(
            "MPS が見える — diffusers の WanRotaryPosEmbed が RoPE の表を float32 で作り、"
            "参照が float64 の機と変わる（ADR 0118 決定 3）。MPS の無い機で回す"
        )


def pad_text_embeds(embeds: torch.Tensor) -> torch.Tensor:
    """有効長の埋め込み `[L_valid, 4096]` → 上流と同じ `[1, 512, 4096]`（後ろをゼロで埋める）。

    上流の `_get_t5_prompt_embeds` は umT5 の出力を有効長で切ってから `new_zeros` で 512 まで
    埋める。ここも同じ順（切った値 → ゼロの連結）で作る。f32 以外は fail loudly（参照に無い
    丸めを足さない — 決定 4）。
    """
    if embeds.dtype != torch.float32:
        raise ValueError(f"埋め込みは f32 で渡す（{embeds.dtype}）")
    if embeds.dim() != 2 or embeds.shape[1] != TEXT_DIM:
        raise ValueError(f"埋め込みは [L_valid, {TEXT_DIM}] で渡す（{tuple(embeds.shape)}）")
    length = embeds.shape[0]
    if not 0 < length <= MAX_SEQUENCE_LENGTH:
        raise ValueError(f"有効長 {length} が 1〜{MAX_SEQUENCE_LENGTH} の外")
    padding = embeds.new_zeros(MAX_SEQUENCE_LENGTH - length, TEXT_DIM)
    return torch.cat([embeds, padding]).unsqueeze(0)


def video_size(latents: torch.Tensor) -> tuple[int, int, int]:
    """潜在 `[1, 16, F, H, W]` → 上流の `(num_frames, height, width)`（`4(F−1)+1`, `8H`, `8W`）。"""
    if latents.dim() != 5 or latents.shape[0] != 1 or latents.shape[1] != 16:
        raise ValueError(f"潜在は [1, 16, F, H, W] で渡す（{tuple(latents.shape)}）")
    frames, height, width = latents.shape[2:]
    return TEMPORAL_SCALE * (frames - 1) + 1, SPATIAL_SCALE * height, SPATIAL_SCALE * width


def load_pipeline(model: str = DEFAULT_MODEL) -> WanPipeline:
    """pin した revision の DiT・VAE・scheduler で `WanPipeline` を CPU f32 に組む（umT5 無し）。"""
    assert_no_mps()
    from diffusers import (
        AutoencoderKLWan,
        UniPCMultistepScheduler,
        WanPipeline,
        WanTransformer3DModel,
    )

    snapshot = local_snapshot(model)
    transformer = WanTransformer3DModel.from_pretrained(
        snapshot, subfolder="transformer", torch_dtype=torch.float32
    )
    vae = AutoencoderKLWan.from_pretrained(snapshot, subfolder="vae", torch_dtype=torch.float32)
    scheduler = UniPCMultistepScheduler.from_pretrained(snapshot, subfolder="scheduler")
    pipeline = WanPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=vae,
        scheduler=scheduler,
        transformer=transformer,
    )
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def run(
    pipeline: WanPipeline,
    *,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    latents: torch.Tensor,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
    guidance_scale: float = GUIDANCE_SCALE,
    output_type: str = "latent",
    **call_kwargs: Any,
) -> torch.Tensor:
    """埋め込みとノイズを注入して `WanPipeline.__call__` を回し、出力（既定は最終の潜在）を返す。

    `negative_prompt_embeds` を必須にするのは、省くと上流が空文字を umT5 で埋め込もうとして
    `text_encoder=None` で落ちるため（決定 4 は公式の negative を明示で渡す）。動画の大きさは
    潜在の形から導く（別に渡すと潜在と食い違う形を作れてしまう）。`call_kwargs` は
    `callback_on_step_end` などをそのまま上流へ渡す。
    """
    num_frames, height, width = video_size(latents)
    output = pipeline(
        prompt_embeds=pad_text_embeds(prompt_embeds),
        negative_prompt_embeds=pad_text_embeds(negative_prompt_embeds),
        latents=latents,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        output_type=output_type,
        return_dict=False,
        **call_kwargs,
    )
    result: torch.Tensor = output[0]
    return result


def _smoke(model: str) -> int:
    """読み込みと 1 ステップ（CFG あり・潜在 `[1,16,3,16,16]` = S 192）+ decode が通ることを見る。

    埋め込みとノイズは固定 seed の乱数（umT5 は読まない）。数の照合はしない — 参照出力の作成は
    段 2 / 4 の台本。decode は上流の経路（逆正規化 → `vae.decode` → 後処理）をそのまま通す。
    """
    generator = torch.Generator().manual_seed(0)
    prompt = torch.randn(16, TEXT_DIM, generator=generator)
    negative = torch.randn(24, TEXT_DIM, generator=generator)
    latents = torch.randn(1, 16, 3, 16, 16, generator=generator)

    started = time.perf_counter()
    pipeline = load_pipeline(model)
    loaded = time.perf_counter()
    video = run(
        pipeline,
        prompt_embeds=prompt,
        negative_prompt_embeds=negative,
        latents=latents,
        num_inference_steps=1,
        output_type="pt",
    )
    finished = time.perf_counter()

    finite = bool(video.isfinite().all())
    print(f"load: {loaded - started:.1f} s")
    print(f"1 step (CFG) + vae decode: {finished - loaded:.1f} s  video {tuple(video.shape)}")
    print(f"finite: {finite}  range [{video.min().item():.4f}, {video.max().item():.4f}]")
    print(f"timesteps: {pipeline.scheduler.timesteps.tolist()}")
    return 0 if finite else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--smoke", action="store_true", help="読み込みと 1 ステップの疎通")
    args = parser.parse_args(argv)
    if not args.smoke:
        parser.error("今は --smoke だけ（参照出力の作成は段 2 / 4 の台本）")
    with torch.no_grad():
        return _smoke(args.model)


if __name__ == "__main__":
    sys.exit(main())
