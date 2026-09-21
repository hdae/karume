"""packed PLE の合成 fixture を作る（実モデルの値は複製しない）。

    cd tools/export-recipes && uv run python -m gemma4.tests.ple_fixture

期待値は符号付き整数と Torch の二段 f32 乗算から作る。packed reader と同じ
ビット展開式で期待値を作らず、全符号値・層別 scale・行境界を独立に検査する。
"""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from _shared.paths import REPO_ROOT
from karume.emit import ContainerEntry, container_order, write_container
from karume.verify import assert_reader_layout


def write_fixture(out: Path) -> None:
    """I2 / I4 各3shardと非連続・重複idのCPU参照を書き出す。"""
    for bits in (2, 4):
        destination = out / f"i{bits}"
        destination.mkdir(parents=True, exist_ok=True)
        tokens, layers, dim = 9, 3, 32
        factor = 8 // bits
        values = (
            torch.arange(tokens * layers * dim, dtype=torch.int32) * 7
            + torch.arange(tokens, dtype=torch.int32).repeat_interleave(layers * dim) * 3
        ).reshape(tokens, layers, dim) % (1 << bits) - (1 << (bits - 1))
        scale = (
            torch.arange(1, tokens + 1, dtype=torch.float32)[:, None]
            * torch.tensor([0.1, 1.5, 0.03125], dtype=torch.float32)[None, :]
        ).contiguous()
        shifted = (values + (1 << (bits - 1))).reshape(tokens, layers, dim // factor, factor)
        packed = torch.zeros(tokens, layers, dim // factor, dtype=torch.uint8)
        for i in range(factor):
            packed |= (shifted[..., i] << (bits * i)).to(torch.uint8)
        common = {
            "schema": 2,
            "storage": f"i{bits}",
            "tokens": tokens,
            "layers": layers,
            "dim": dim,
            "embedScale": 3.5,
        }
        shards = []
        for number, start in enumerate(range(0, tokens, 3), 1):
            stop = start + 3
            name = f"ple-{number:05}-of-00003.safetensors"
            payload = {"values": packed[start:stop], "scales": scale[start:stop]}
            entries = container_order(
                [
                    ContainerEntry(
                        "values", f"I{bits}", (3, layers, dim), payload["values"].numel()
                    ),
                    ContainerEntry("scales", "F32", (3, layers), payload["scales"].numel() * 4),
                ]
            )
            metadata = {**common, "start": start, "stop": stop}
            write_container(
                destination / name,
                entries,
                {"karume_ple": json.dumps(metadata)},
                lambda e, payload=payload: [memoryview(payload[e.name].numpy()).cast("B")],
            )
            assert_reader_layout(destination / name)
            shards.append({"file": name, "start": start, "stop": stop})
        ids = torch.tensor([8, 0, 4, 3, 4, 2, 6, 5, 7, 1, 0], dtype=torch.int32)
        expected = (values[ids].float() * scale[ids, :, None]) * common["embedScale"]
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
        (destination / "ple.json").write_text(
            json.dumps({**common, "shards": shards}, indent=2) + "\n"
        )


def main() -> None:
    write_fixture(REPO_ROOT / "packages/models/tests/fixtures/gemma4-ple-packed")


if __name__ == "__main__":
    main()
