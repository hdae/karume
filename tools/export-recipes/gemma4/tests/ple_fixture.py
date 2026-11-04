"""Builds the synthetic packed PLE fixture (no values of a real model are copied).

    cd tools/export-recipes && uv run python -m gemma4.tests.ple_fixture

The expected values come from signed integers and Torch's two-step f32 multiplication. They are
not built with the same bit-unpacking formula as the packed reader; every signed value, the
per-layer scales and the row boundaries are checked independently.

MUST: the assets are baked with **the same functions as the recipe's production path**
({@link karume.ple.ple_assets} owns the index and how blocks are cut, and
{@link karume.pipeline.publish_model} places the container). A separate fixture-only writer would
make the reader-side tests check only "how the fixture was written", and they would stay green
while drifting from the shape of the containers actually distributed.

MUST: two runs are byte-identical (no generator tag and no timestamp are baked in — `_PROVENANCE`
is `license` only, for the same reason as `karume.goldens.GOLDEN_PROVENANCE`). The fixture is
tracked by git, so a form that produces a diff on every run would erase the meaning of "the asset
changed".
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

from _shared.paths import REPO_ROOT
from karume.container import Provenance
from karume.pipeline import export_module, publish_model
from karume.ple import ple_assets

#: Location of the fixture (the reader is `packages/models/tests/gemma4_ple_packed_test.ts`).
FIXTURE_ROOT = REPO_ROOT / "packages" / "models" / "tests" / "fixtures" / "gemma4-ple-packed"

#: Representative path of the container (the same spelling as `SERIES_MODEL_FILE` in
#: `helpers/ple-series.ts` on the TS side).
MODEL_FILE = "model.krm"

#: Graph name inside the container (= the weights key of the distribution manifest —
#: container-v1 §2.1).
GRAPH_NAME = "model"

#: Provenance. `writer` is **not written** — baking a generator tag would move the fixture's bytes
#: just by bumping the package version, breaking the premise that a regeneration reads as "the
#: asset changed" (the same treatment as goldens).
_PROVENANCE = Provenance(license="mit")

#: Synthetic dimensions (tokens / layers / dim). Small, yet made to cross **both row and block
#: boundaries**.
TOKENS, LAYERS, DIM = 9, 3, 32

#: Downstream factor applied to every layer (the index's `embedScale`).
EMBED_SCALE = 3.5

#: Limit for cutting the PLE blocks (**applied to the PLE assets only** — the block limit of the
#: container as a whole stays at its default; the index JSON is larger than this, so applying it to
#: the whole container would make `_plan_asset_parts` fail).
#:
#: A value is chosen **at which values and scales are cut at different points** (144 = 3 rows of
#: values in i4, 6 rows in i2, and 12 rows of scales = 1 block). In a fixture where both tables are
#: cut at the same boundaries, a reader error that looks up a different block per table (looking up
#: scales by the values' block number, and the like) would pass unnoticed.
BLOCK_BYTES = 144


class PleFixtureGraph(nn.Module):
    """Minimal graph that carries the assets.

    This fixture's gate reads only the PLE assets, but the container is **a model container with a
    graph** (`krm` kind = model), so it is baked in the real form with one weight. The values are
    deterministic (`arange`) — drawing random numbers would move the bytes between two runs.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.arange(dim, dtype=torch.float32) / dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


def _packed_values(bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Signed raw values `[tokens, layers, dim]` and packed `[tokens, layers, dim/factor]`."""
    factor = 8 // bits
    values = (
        torch.arange(TOKENS * LAYERS * DIM, dtype=torch.int32) * 7
        + torch.arange(TOKENS, dtype=torch.int32).repeat_interleave(LAYERS * DIM) * 3
    ).reshape(TOKENS, LAYERS, DIM) % (1 << bits) - (1 << (bits - 1))
    shifted = (values + (1 << (bits - 1))).reshape(TOKENS, LAYERS, DIM // factor, factor)
    packed = torch.zeros(TOKENS, LAYERS, DIM // factor, dtype=torch.uint8)
    for i in range(factor):
        packed |= (shifted[..., i] << (bits * i)).to(torch.uint8)
    return values, packed


def _reader(payload: bytes, stride: int) -> Callable[[int, int], bytes]:
    """Range read of the concatenated row-major payload (the input of {@link ple_assets}).

    MUST: return the same bytes for the same range (the writer reads twice: once to take the sha256
    and once to write).
    """

    def read(begin: int, end: int) -> bytes:
        if begin % stride or end % stride:
            raise AssertionError(
                f"PLE range [{begin}, {end}) is not a multiple of the {stride}-byte row"
            )
        return payload[begin:end]

    return read


def write_fixture(out: Path) -> None:
    """Write the I2 / I4 packed PLE as **assets of a container** with a CPU reference beside it."""
    for bits in (2, 4):
        storage = f"i{bits}"
        destination = out / storage
        destination.mkdir(parents=True, exist_ok=True)

        values, packed = _packed_values(bits)
        scale = (
            torch.arange(1, TOKENS + 1, dtype=torch.float32)[:, None]
            * torch.tensor([0.1, 1.5, 0.03125], dtype=torch.float32)[None, :]
        ).contiguous()
        values_payload = bytes(memoryview(packed.contiguous().numpy()).cast("B"))
        scales_payload = bytes(memoryview(scale.numpy()).cast("B"))

        graph, tensors = export_module(PleFixtureGraph(DIM), (torch.zeros(1, DIM),))
        publish_model(
            destination / MODEL_FILE,
            graph,
            tensors,
            provenance=_PROVENANCE,
            graph_name=GRAPH_NAME,
            assets=ple_assets(
                storage=storage,
                tokens=TOKENS,
                layers=LAYERS,
                dim=DIM,
                embed_scale=EMBED_SCALE,
                read_values=_reader(values_payload, LAYERS * DIM // (8 // bits)),
                read_scales=_reader(scales_payload, LAYERS * 4),
                block_bytes=BLOCK_BYTES,
            ),
        )

        ids = torch.tensor([8, 0, 4, 3, 4, 2, 6, 5, 7, 1, 0], dtype=torch.int32)
        expected = (values[ids].float() * scale[ids, :, None]) * EMBED_SCALE
        save_file(
            {"ids": ids, "expected": expected},
            str(destination / "oracle.safetensors"),
            metadata={
                "source": (
                    "synthetic signed integer values, "
                    f"torch {torch.__version__}, two f32 multiplies"
                )
            },
        )


def main() -> None:
    write_fixture(FIXTURE_ROOT)


if __name__ == "__main__":
    main()
