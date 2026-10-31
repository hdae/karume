"""The stage-0 VAE reference of MOSS-Image: upstream's decode of fixed latents in f32, with the
tensors around the mid-block attention (ADR 0138 stage 0).

For each size (height × width in pixels: 256² and 512×768 — non-square so H and W cannot be
swapped unnoticed), seeded DiT-space latents `[1, 128, H/16, W/16]` go through the two steps the
pipeline runs before decoding (`moss.pipeline_ref.denormalize` — the BatchNorm running statistics —
and `unpatchify`, upstream's statements) and through `AutoencoderKLFlux2.decode` without tiling, as
`MOSSImagePipeline.__call__` decodes. The hooks (`moss.pipeline_ref.VaeRecorder`) keep the decode
input and the tensors before and after the mid attention — the boundaries of the three decoder
graphs of ADR 0138 decision 8 — and the decoder's output before the clamp.

Record (`vae-<H>x<W>.safetensors`): `latents`, `denormalized`, `vae.z` (the unpatchified decode
input — checked bit for bit against what the decode received), `vae.pre_attention`,
`vae.post_attention`, `vae.decoded` and `image` (`vae.decoded` clamped to [-1, 1], as upstream
clamps before post-processing).

    uv run --group moss --inexact python -m moss.vae_reference

Output: the records and `vae_reference.json` (time, peak RAM) in the stage-0 reference directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from moss import pipeline_ref, records, sources
from moss.records import bits_equal
from moss.reference import parse_size

#: `(height, width)` in pixels.
SIZES: tuple[tuple[int, int], ...] = ((256, 256), (512, 768))

#: The seed of the fixed latents.
SEED = 20261028

#: The single metadata key of the records.
METADATA_KEY = "karume.moss.vae_reference"


class VaeReferenceError(RuntimeError):
    """The hooks did not see the decode this script ran."""


def decode_case(vae: Any, height: int, width: int) -> dict[str, torch.Tensor]:
    """Decode the seeded latents of one size with the hooks; return the record's tensors."""
    latents = pipeline_ref.seeded_latents(SEED, height, width)
    denormalized = pipeline_ref.denormalize(vae, latents)
    z = pipeline_ref.unpatchify(denormalized)
    with pipeline_ref.VaeRecorder(vae) as recorder, torch.no_grad():
        decoded = vae.decode(z, return_dict=False)[0]
    seen = recorder.tensors
    if not bits_equal(seen["vae.z"], z) or not bits_equal(seen["vae.decoded"], decoded):
        raise VaeReferenceError(f"{height}x{width}: the hooks saw another decode")
    if decoded.shape != (1, 3, height, width):
        raise VaeReferenceError(f"{height}x{width}: decoded to {tuple(decoded.shape)}")
    return {"latents": latents, "denormalized": denormalized, **seen, "image": decoded.clamp(-1, 1)}


def generate(out: Path, sizes: Sequence[tuple[int, int]]) -> dict[str, Any]:
    """Decode every size, write one record per size; return the run summary."""
    clock = records.Clock()
    snapshot = sources.local_snapshot()
    vae = pipeline_ref.load_vae(snapshot)
    clock.lap("load vae (f32)")
    cases = []
    for height, width in sizes:
        tensors = decode_case(vae, height, width)
        decode_seconds = clock.lap(f"decode {height}x{width}")
        metadata = {
            "source": {
                "repo": sources.SOURCE.repo,
                "revision": sources.SOURCE.revision,
                "file": sources.VAE_FILE,
            },
            "size": [height, width],
            "seed": SEED,
            "noise": "torch.randn(generator=torch.Generator().manual_seed(seed), dtype=float32)",
            "batch_norm_eps": vae.config.batch_norm_eps,
            "tiling": False,
            "versions": records.versions(),
            "torch_threads": torch.get_num_threads(),
            "shapes": {name: list(t.shape) for name, t in tensors.items()},
        }
        path = out / f"vae-{height}x{width}.safetensors"
        records.write_record(path, tensors, metadata, key=METADATA_KEY)
        cases.append(
            {
                "size": [height, width],
                "path": str(path),
                "bytes": path.stat().st_size,
                "decode_seconds": decode_seconds,
                "shapes": metadata["shapes"],
            }
        )
    return {
        "cases": cases,
        "environment": records.environment(),
        "versions": records.versions(),
        "seconds": {**clock.phases, "total": clock.total()},
        "peak_rss_gib": round(records.peak_rss_gib(), 2),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--size", type=parse_size, action="append", help="<height>x<width>")
    parser.add_argument("--out", type=Path, default=None, help="default: the stage-0 reference dir")
    args = parser.parse_args(argv)
    out = args.out or records.default_reference_dir()
    summary = generate(out, args.size or SIZES)
    text = json.dumps(summary, indent=1)
    print(text)
    (out / "vae_reference.json").write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
