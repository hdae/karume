"""Writes `encoders.json` next to this file: seeded tensors stored by the exporter's own code.

Each case is one tensor in a source dtype (F32 / F16 / BF16) and the bytes the exporter stores for
it in a target codec: `int8-sym` through `fake_quant_int8` (the scale ledger) and
`_convert_for_storage` (the stored codes), `bf16` through `_convert_for_storage` (which refuses a
value that is not exactly a bf16 value), and `f32` as the tensor's own f32 bytes (the exporter
stores an f32 initializer without a conversion). The TypeScript encoders of `@karume/convert` are
tested against these bytes.

Re-run from the exporter's project (it has torch):

    cd tools/exporter && uv run python ../../packages/convert/tests/fixtures/generate_encoders.py

The output is deterministic (seeded generators, CPU).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from karume.emit import EmitError, _Conversion, _convert_for_storage
from karume.quantize import fake_quant_int8

OUT = Path(__file__).with_name("encoders.json")

TORCH_DTYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}


def raw(tensor: torch.Tensor) -> str:
    """The tensor's little-endian bytes in row-major order, as hex (bf16 through an int16 view)."""
    stored = tensor.detach().contiguous()
    if stored.dtype is torch.bfloat16:
        stored = stored.view(torch.int16)
    return stored.numpy().tobytes().hex()


def bits(pattern: int) -> float:
    """The f32 value of a 32-bit pattern."""
    return float(torch.tensor([pattern], dtype=torch.int32).view(torch.float32)[0])


def seeded(shape: list[int], seed: int, scale: float) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator) * scale


def int8_case(name: str, module: nn.Module, source: torch.Tensor, source_dtype: str) -> dict:
    """The exporter's int8 path: fake-quant on the module, then the stored codes and the scale."""
    with torch.no_grad():
        module.weight.copy_(source.to(torch.float32))
    report = fake_quant_int8(module)
    scale = report.scales["weight"]
    axis = 1 if isinstance(module, nn.ConvTranspose1d) else 0
    codes = _convert_for_storage(
        "weight", module.weight.detach(), _Conversion(dtype="i8", name=name, scale=scale)
    )
    return {
        "name": name,
        "codec": "int8-sym",
        "rowAxis": axis,
        "shape": list(source.shape),
        "sourceDtype": source_dtype,
        "source": raw(source),
        "data": raw(codes),
        "scale": raw(scale),
        "refusedElement": None,
    }


def first_refused(values: torch.Tensor) -> int:
    """The first element `_is_bf16_exact` fails on: a NaN, or a value with a non-zero lower half."""
    words = values.to(torch.float32).contiguous().view(torch.int32).reshape(-1).tolist()
    for index, word in enumerate(words):
        if (word & 0x7FFFFFFF) > 0x7F800000 or (word & 0xFFFF) != 0:
            return index
    raise AssertionError("no refused element")


def float_case(name: str, codec: str, source: torch.Tensor, source_dtype: str) -> dict:
    """A float target: every case has the keys of an int8 case (no scale; `data` is empty when the
    exporter refuses the tensor, and `refusedElement` names the first element it fails on)."""
    case = {
        "name": name,
        "codec": codec,
        "rowAxis": 0,
        "shape": list(source.shape),
        "sourceDtype": source_dtype,
        "source": raw(source),
        "scale": "",
    }
    widened = source.to(torch.float32)
    if codec == "f32":
        return {**case, "data": raw(widened), "refusedElement": None}
    try:
        stored = _convert_for_storage("weight", widened, _Conversion(dtype="bf16", name=name))
    except EmitError:
        return {**case, "data": "", "refusedElement": first_refused(widened)}
    return {**case, "data": raw(stored), "refusedElement": None}


def rows_source() -> torch.Tensor:
    """[8, 47] f32: seeded rows, an all-zero row, rounding ties, the smallest scale, ±0."""
    rows = seeded([8, 47], 1, 0.05)
    rows[3] = seeded([47], 2, 3.0)
    rows[4] = 0.0
    # amax 127 → scale exactly 1, so w / scale = k + 0.5 is a tie (half to even).
    rows[5] = torch.tensor([127.0] + [(k + 0.5) * (-1) ** k for k in range(46)])
    # amax 127·2^-126 → scale exactly 2^-126 (the smallest, not clamped), and ties again.
    tiny = 2.0**-126
    rows[6] = torch.tensor([127.0 * tiny] + [(k + 0.5) * (-1) ** k * tiny for k in range(46)])
    # amax below 127·2^-126 → the scale is clamped to 2^-126; every code is 0. ±0 kept apart.
    row7 = [0.0] * 47
    row7[0], row7[1], row7[2], row7[3] = 2.0**-140, -(2.0**-140), -0.0, 3.0 * 2.0**-133
    rows[7] = torch.tensor(row7)
    return rows


def main() -> None:
    cases = [
        int8_case("int8-rows", nn.Linear(47, 8, bias=False), rows_source(), "F32"),
        int8_case(
            "int8-axis1",
            nn.ConvTranspose1d(3, 5, 4, bias=False),
            seeded([3, 5, 4], 3, 0.2).to(torch.bfloat16),
            "BF16",
        ),
        int8_case(
            "int8-small",
            nn.Linear(5, 3, bias=False),
            seeded([3, 5], 4, 0.5).to(torch.float16),
            "F16",
        ),
        float_case(
            "bf16-from-f32",
            "bf16",
            torch.tensor(
                [
                    [0.0, -0.0, 2.0**-130],
                    [-(2.0**-133), float("inf"), float("-inf")],
                    [1.5, -3.25, bits(0x3E4C0000)],
                ]
            ),
            "F32",
        ),
        float_case("bf16-from-bf16", "bf16", seeded([4, 6], 5, 1.0).to(torch.bfloat16), "BF16"),
        float_case(
            "bf16-from-f16",
            "bf16",
            torch.tensor(
                [[2.0**-24, -(2.0**-20), 0.15625, -0.0], [96.0, -1.0, 2.0**-14, 3.0]],
                dtype=torch.float16,
            ),
            "F16",
        ),
        float_case(
            "bf16-inexact", "bf16", torch.tensor([[1.0, bits(0x3F800001)], [2.0, 3.0]]), "F32"
        ),
        float_case("bf16-nan", "bf16", torch.tensor([1.0, bits(0x7FC00000)]), "F32"),
        float_case("f32-from-bf16", "f32", seeded([5], 6, 1.0).to(torch.bfloat16), "BF16"),
        float_case(
            "f32-from-f16",
            "f32",
            torch.tensor([2.0**-24, -0.0, 65504.0, float("inf"), -1.5, 0.333], dtype=torch.float16),
            "F16",
        ),
        float_case("f32-from-f32", "f32", seeded([5], 7, 1.0), "F32"),
    ]
    OUT.write_text(
        dumps(
            {
                "generator": "packages/convert/tests/fixtures/generate_encoders.py",
                "torch": torch.__version__,
            },
            cases,
        )
    )


def dumps(head: dict, cases: list[dict]) -> str:
    """The document as the repository's formatter writes it (2-space indent, one field per line,
    a shape on one line), so a run leaves nothing for `vp check` to reformat."""
    lines = ["{", *(f"  {json.dumps(key)}: {json.dumps(value)}," for key, value in head.items())]
    lines.append('  "cases": [')
    for index, case in enumerate(cases):
        fields = [f"      {json.dumps(key)}: {json.dumps(value)}" for key, value in case.items()]
        lines += ["    {", ",\n".join(fields), "    }," if index < len(cases) - 1 else "    }"]
    lines += ["  ]", "}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
