"""Write the fixture checking the Wan2.1 DiT host functions (TS `packages/models/src/wan/dit-*.ts`).

TS owns the 4 steps outside the S-shaped graph (patchify / unpatchify / the reordering of the RoPE
tables / `timesteps_proj`); this script bakes their inputs and **the upstream values** on a small
grid (ADR 0118 decision 3, acceptance step 2). The fixture is tracked, so it is kept to one grid
(latent `[16,21,6,10]` → F'·H'·W' = 21·3·5 = 315 tokens) and 50 timesteps. The time axis is the
T' = 21 of an 81-frame latent itself (ADR 0118 step 8 — every position 0 to 20 of the RoPE t axis
is checked against the upstream table).

    uv run --group wan --inexact python -m wan.dit_host_fixture

Output (default `packages/models/tests/fixtures/wan-dit/`):

    host.safetensors       inputs and expected values of patchify / unpatchify / the RoPE
                           tables / timesteps_proj
    rope_base.safetensors  the base table in the same form as the asset `rope_base` (rows
                           cut to the 24 the grid needs)
    host.json              provenance (repo / revision), patch, grid and the timestep list

How the expected values are made (each one calls the upstream op — no formula is copied):

- patchify: a convolution with the same `Conv3d(k = s = patch)` as the upstream patch embedding,
  loaded with **one-hot weights**. The `flatten(2).transpose(1, 2)` of the output `[1,64,F',H',W']`
  is exactly the tokens with the input window laid out in the flattening order of the conv weights
  `(c, pt, ph, pw)` (each product is 1 or 0 and only one term of each sum is non-zero — no
  rounding enters).
- unpatchify: the upstream exit, verbatim (`dit_patch.dit_unpatchify`).
- RoPE tables: the output of the upstream `model.rope` (`dit_patch.dit_rope_tables`). The base
  table is `dit_patch.dit_rope_base_tables`.
- `timesteps_proj`: the upstream `condition_embedder.timesteps_proj`
  (`dit_patch.dit_timesteps_proj`). The timesteps are the UniPC list of the reference settings
  (50 steps, shift 3.0 — decision 5) plus the ends 0 / 1 and 745, where the difference from TS is
  worst (measured over all 1,001 values — the doc of the TS test).
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
from wan.sources import DEFAULT_MODEL, SOURCES, WAN21_MODELS, local_snapshot

DEFAULT_OUT = REPO_ROOT / "packages" / "models" / "tests" / "fixtures" / "wan-dit"

#: The latent `(F, H, W)` of the fixture. The grid F'·H'·W' = 21·3·5 has a different value on each
#: of the 3 axes (a swapped axis cannot cancel out), and F' = 21 is the time axis of an 81-frame
#: latent (ADR 0118 step 8).
LATENT_SHAPE = (21, 6, 10)

#: The number of rows the base table is cut to (covers the grid's largest F' = 21; with the asset's
#: 1,024 rows the fixture would be 0.5 MiB).
ROPE_BASE_ROWS = 24

#: Timesteps added to the list of the reference settings (the ends 0 / 1 and 745, the measured worst
#: difference from TS).
EXTRA_TIMESTEPS = (0, 1, 745)

SEED = 20261002


def onehot_patchify(latents: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    """Lay out the windows with the same `Conv3d(k = s = patch)` as the upstream patch embedding,
    loaded with one-hot weights.

    An independent oracle for the patchify order of TS and Python (each product is 1 or 0, only one
    term of each sum is non-zero).
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
    """The UniPC timestep list of the reference settings (50 steps, the config's shift — decision
    5)."""
    from diffusers import UniPCMultistepScheduler

    scheduler = UniPCMultistepScheduler.from_pretrained(
        local_snapshot(model_name), subfolder="scheduler"
    )
    scheduler.set_timesteps(NUM_INFERENCE_STEPS)
    return [int(value) for value in scheduler.timesteps.tolist()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=WAN21_MODELS)
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
        raise AssertionError(
            "dit_patchify differs from the order of the upstream patch embedding (one-hot conv3d)"
        )

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
            # The latent is stored in the host's shape (`[C,F,H,W]` without the batch axis — ADR
            # 0118 decision 3).
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
            "Fixture for checking the Wan2.1 DiT host functions"
            " (packages/models/src/wan/dit-*.ts).",
            "Generated by tools/export-recipes/wan/dit_host_fixture.py"
            " (the expected values are outputs of the upstream ops — that file's doc).",
            "rope_base.safetensors has the same form as the asset rope_base,"
            " with the rows cut to what the grid needs.",
        ],
        "source": {"repo": source.repo, "revision": source.revision},
        "patch_size": list(patch_size),
        "latent_shape": [channels, *LATENT_SHAPE],
        "rope_base_rows": ROPE_BASE_ROWS,
        "timesteps": timesteps,
    }
    # Indent by 2 (the JSON shape of `deno fmt` — the fixture is tracked, so a rewrite must not
    # produce a diff).
    (args.out / "host.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"fixture OK: {args.out} ({len(timesteps)} timesteps, grid {LATENT_SHAPE})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
