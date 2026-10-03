"""Wan2.1 の DiT のホスト関数（TS — `packages/models/src/wan/dit-*.ts`）の照合用 fixture を書く。

TS が持つのは S 形グラフの外の 4 段（patchify / unpatchify / RoPE の表の並べ替え /
`timesteps_proj`）で、ここはその入力と**上流の値**を小さな格子で焼く（ADR 0118 決定 3・検収 段 2）。
追跡対象の fixture なので、大きさは格子 1 つ（潜在 `[16,21,6,10]` → F'·H'·W' = 21·3·5 = 315
トークン）と timestep 50 本に絞る。時間軸は 81 フレームの潜在の T' = 21 そのもの（ADR 0118 段 8 —
RoPE の t 軸の位置 0〜20 を全部、上流の表と突き合わせる）。

    uv run --group wan --inexact python -m wan.dit_host_fixture

出力（既定 `packages/models/tests/fixtures/wan-dit/`）:

    host.safetensors       patchify / unpatchify / RoPE の表 / timesteps_proj の入力と期待値
    rope_base.safetensors  資産 `rope_base` と同じ形式の素表（行を格子に要る 24 行へ切り詰めたもの）
    host.json              出所（repo / revision）・patch・格子・timestep の列

期待値の作り方（どれも上流の演算を呼ぶ — 式を写さない）:

- patchify: 上流の patch 埋め込みと同じ `Conv3d(k = s = patch)` に**1-hot の重み**を入れた畳み込み。
  出力 `[1,64,F',H',W']` の `flatten(2).transpose(1, 2)` が、入力の窓を conv の重みの平坦化順
  `(c, pt, ph, pw)` で並べたトークンそのもの（積は 1 か 0 で、和は 1 項だけが非ゼロ —
  丸めが入らない）。
- unpatchify: 上流の出口の逐語（`dit_patch.dit_unpatchify`）。
- RoPE の表: 上流 `model.rope` の出力（`dit_patch.dit_rope_tables`）。素表は
  `dit_patch.dit_rope_base_tables`。
- `timesteps_proj`: 上流 `condition_embedder.timesteps_proj`（`dit_patch.dit_timesteps_proj`）。
  timestep は参照の設定（50 ステップ・shift 3.0 — 決定 5）の UniPC の列に、端の 0 / 1 と、TS
  との差が最悪になる 745（全 1,001 通りの実測 — TS テストの doc）を足したもの。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

from _shared.paths import REPO_ROOT
from wan import dit_patch
from wan.export_dit import load_transformer
from wan.pipeline_ref import NUM_INFERENCE_STEPS
from wan.sources import DEFAULT_MODEL, SOURCES, local_snapshot

DEFAULT_OUT = REPO_ROOT / "packages" / "models" / "tests" / "fixtures" / "wan-dit"

#: fixture の潜在 `(F, H, W)`。格子 F'·H'·W' = 21·3·5 は 3 軸とも違う値（軸の取り違えが対合にならな
#: い）で、F' = 21 は 81 フレームの潜在の時間軸（ADR 0118 段 8）。
LATENT_SHAPE = (21, 6, 10)

#: 素表を切り詰める行数（格子の最大 F' = 21 を覆う。資産の 1,024 行のままだと fixture が 0.5 MiB
#: になる）。
ROPE_BASE_ROWS = 24

#: 参照の設定の列に足す timestep（端の 0 / 1 と、TS との差の実測最悪 745）。
EXTRA_TIMESTEPS = (0, 1, 745)

SEED = 20261002


def onehot_patchify(latents: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    """上流の patch 埋め込みと同じ `Conv3d(k = s = patch)` に 1-hot の重みを入れて窓を並べる。

    TS と Python の patchify の並びの独立オラクル（積は 1 か 0・和は 1 項だけが非ゼロ）。
    """
    channels = latents.shape[1]
    volume = patch_size[0] * patch_size[1] * patch_size[2]
    width = channels * volume
    conv = nn.Conv3d(channels, width, kernel_size=patch_size, stride=patch_size)
    with torch.no_grad():
        conv.weight.copy_(torch.eye(width).reshape(width, channels, *patch_size))
        conv.bias.zero_()
        return conv(latents).flatten(2).transpose(1, 2).contiguous()


def schedule_timesteps(model_name: str) -> list[int]:
    """参照の設定（50 ステップ・config の shift — 決定 5）の UniPC の timestep の列。"""
    from diffusers import UniPCMultistepScheduler

    scheduler = UniPCMultistepScheduler.from_pretrained(
        local_snapshot(model_name), subfolder="scheduler"
    )
    scheduler.set_timesteps(NUM_INFERENCE_STEPS)
    return [int(value) for value in scheduler.timesteps.tolist()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(SOURCES))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    model = load_transformer(args.model)
    patch_size = tuple(int(size) for size in model.config.patch_size)
    channels = int(model.config.in_channels)
    generator = torch.Generator().manual_seed(SEED)

    latents = torch.randn(1, channels, *LATENT_SHAPE, generator=generator)
    tokens = dit_patch.dit_patchify(latents, patch_size)
    oracle = onehot_patchify(latents, patch_size)
    if not torch.equal(tokens, oracle):
        raise AssertionError("dit_patchify が上流の patch 埋め込み（1-hot conv3d）の並びと違う")

    output_tokens = torch.randn(tokens.shape, generator=generator)
    unpatched = dit_patch.dit_unpatchify(output_tokens, LATENT_SHAPE, patch_size)

    with torch.no_grad():
        rope_cos, rope_sin = dit_patch.dit_rope_tables(model.rope, LATENT_SHAPE)
        base = dit_patch.dit_rope_base_tables(model.rope)
        timesteps = sorted(set(schedule_timesteps(args.model)) | set(EXTRA_TIMESTEPS), reverse=True)
        proj = dit_patch.dit_timesteps_proj(model, torch.tensor(timesteps, dtype=torch.int64))

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            # 潜在はホストの形（バッチ軸を持たない `[C,F,H,W]` — ADR 0118 決定 3）で置く。
            "patchify.latents": latents[0].contiguous(),
            "patchify.tokens": tokens.contiguous(),
            "unpatchify.tokens": output_tokens.contiguous(),
            "unpatchify.latents": unpatched[0].contiguous(),
            "rope.cos": rope_cos.contiguous(),
            "rope.sin": rope_sin.contiguous(),
            "timesteps.values": torch.tensor(timesteps, dtype=torch.int32),
            "timesteps.proj": proj.contiguous(),
        },
        str(args.out / "host.safetensors"),
    )
    save_file(
        {name: table[:ROPE_BASE_ROWS].contiguous() for name, table in base.items()},
        str(args.out / "rope_base.safetensors"),
    )
    source = SOURCES[args.model]
    meta = {
        "_doc": [
            "Wan2.1 の DiT のホスト関数（packages/models/src/wan/dit-*.ts）の"
            "突き合わせ用 fixture。",
            "生成: tools/export-recipes/wan/dit_host_fixture.py"
            "（期待値は上流の演算の出力 — 同ファイルの doc）。",
            "rope_base.safetensors は資産 rope_base と同じ形式で、"
            "行を格子に要る分へ切り詰めたもの。",
        ],
        "source": {"repo": source.repo, "revision": source.revision},
        "patch_size": list(patch_size),
        "latent_shape": [channels, *LATENT_SHAPE],
        "rope_base_rows": ROPE_BASE_ROWS,
        "timesteps": timesteps,
    }
    # 2 字下げ（`deno fmt` の JSON の形 — 追跡対象の fixture なので書き直しで差分を出さない）。
    (args.out / "host.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"fixture OK: {args.out}（timestep {len(timesteps)} 本・格子 {LATENT_SHAPE}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
