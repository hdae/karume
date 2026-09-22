"""言語横断 fixture の合成モデルと生成器。

置き場は `packages/runtime/tests/fixtures/container/synthetic.krm`。

TS 側の読み手（`packages/runtime/src/format/container/*.ts`）が**この機以外で**書かれたコンテナを
開けることを示すための資産で、`packages/runtime/tests/container_fixture_test.ts` が
`openContainer` で開き、全 block の sha256 と descriptor の正準直列化を確かめる。

再生成は 1 コマンド:

```
KARUME_FIXTURE=write uv run pytest tests/test_container.py -k fixture
```

合成の中身は「読み手の分岐を 1 回ずつ踏む」ことだけを目的に選んである:

- const 領域（f32 と i32 の 2 本）と重み（f16 / int8-sym / int4-sym-g）
- 末尾の詰め物が要る payload（f16 で 42 バイト → 44 バイト）
- block 上限（この fixture では 256 B）を超えて piece 列になる重み
- `rowAxis: 1`（`conv_transpose1d` 形）の per-channel scale
- `shared` 宣言（束縛表の突合集合から外れる initializer）
- part 長（512 B）をまたぐ配置（part が 3 本に割れる）

MUST: バイト列は乱数ではなく seed から決まる（同じ入力から同じ fixture が出る）。
"""

from __future__ import annotations

import os
from pathlib import Path

from karume.container import Encoding, Provenance, write_model_container
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrShared, IrStorage, IrValue

#: リポジトリ直下（tools/exporter/tests → tools/exporter → tools → repo）。
REPO_ROOT = Path(__file__).resolve().parents[3]

#: 言語横断 fixture の置き場（TS 側テストが読む）。
FIXTURE_PATH = (
    REPO_ROOT / "packages" / "runtime" / "tests" / "fixtures" / "container" / "synthetic.krm"
)

#: 合成のグラフ名。
GRAPH_NAME = "synthetic"

#: fixture 用に下げた寸法（実物は 32 MiB / 256 MiB）。piece 分割と part またぎを数 KiB で踏む。
FIXTURE_BLOCK_BYTES = 256
FIXTURE_PART_BYTES = 512


def deterministic_bytes(length: int, seed: int) -> bytes:
    """seed から決まる疑似乱数バイト列（xorshift32 — TS 側テストの `bytesOf` と同じ流儀）。"""
    out = bytearray(length)
    state = seed & 0xFFFFFFFF or 1
    for index in range(length):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        state &= 0xFFFFFFFF
        out[index] = state & 0xFF
    return bytes(out)


def synthetic_graph() -> IrGraph:
    """IR v1 の合成グラフ（`ir_v2_document` が実体の鍵へ改名する前の形）。"""
    return IrGraph(
        symbols=["T"],
        inputs=[IrInput(name="x", dtype="f32", shape=["T", 8])],
        outputs=["h"],
        initializers={
            "p_enc_w": IrInitializer(
                tensor="enc.weight",
                storage=IrStorage(dtype="i4", scale="enc.weight_scale", group_size=32),
            ),
            "p_dec_w": IrInitializer(tensor="dec.weight", storage=IrStorage(dtype="f16")),
            "p_big_w": IrInitializer(
                tensor="big.weight", storage=IrStorage(dtype="i8", scale="big.weight_scale")
            ),
            "p_conv_w": IrInitializer(
                tensor="conv.weight", storage=IrStorage(dtype="i8", scale="conv.weight_scale")
            ),
            "const_a1b2c3d4e5f60718": IrInitializer(
                tensor="const.a1b2c3d4e5f60718", storage=IrStorage(dtype="f32")
            ),
            "const_b1b2c3d4e5f6071a": IrInitializer(
                tensor="const.b1b2c3d4e5f6071a", storage=IrStorage(dtype="i32")
            ),
            "p_lm_head": IrInitializer(
                shared=IrShared(tensor="lm_head.weight"), storage=IrStorage(dtype="f16")
            ),
        },
        values={
            "p_enc_w": IrValue(dtype="f32", shape=[8, 32]),
            "p_dec_w": IrValue(dtype="f32", shape=[3, 7]),
            "p_big_w": IrValue(dtype="f32", shape=[16, 32]),
            "p_conv_w": IrValue(dtype="f32", shape=[4, 3, 8]),
            "const_a1b2c3d4e5f60718": IrValue(dtype="f32", shape=[8]),
            "const_b1b2c3d4e5f6071a": IrValue(dtype="i32", shape=[4]),
            "p_lm_head": IrValue(dtype="f32", shape=[8, 8]),
            "h": IrValue(dtype="f32", shape=["T", 32]),
        },
        nodes=[IrNode(op="matmul", ins=["x", "p_enc_w"], outs=["h"], attrs={})],
    )


def synthetic_bindings() -> dict[str, Encoding]:
    """テンソルキー → 格納の指定（shared でない initializer ぜんぶを覆う）。"""
    return {
        "enc.weight": Encoding(
            "int4-sym-g", group_size=32, row_axis=0, scale_key="enc.weight_scale"
        ),
        "dec.weight": Encoding("f16"),
        "big.weight": Encoding("int8-sym", group_size=32, row_axis=0, scale_key="big.weight_scale"),
        # conv_transpose1d 形: 行の軸は 1（per-channel scale の長さは shape[1]）。
        "conv.weight": Encoding(
            "int8-sym", group_size=32, row_axis=1, scale_key="conv.weight_scale"
        ),
        "const.a1b2c3d4e5f60718": Encoding("f32"),
        "const.b1b2c3d4e5f6071a": Encoding("i32"),
    }


def synthetic_tensors() -> dict[str, bytes]:
    """テンソルキー → 生バイト（長さは宣言 shape と codec から決まる値ちょうど）。"""
    return {
        # i4: 8 x 32 = 256 要素 → 128 バイト・scale は [8, 1] の f32。
        "enc.weight": deterministic_bytes(128, 21),
        "enc.weight_scale": deterministic_bytes(8 * 4, 22),
        # f16: 3 x 7 = 21 要素 → 42 バイト（末尾の詰め物 2 バイトを書き手が焼く）。
        "dec.weight": deterministic_bytes(42, 23),
        # i8: 16 x 32 = 512 バイト → block 上限 256 B で 2 piece（8 行ずつ）。
        "big.weight": deterministic_bytes(512, 24),
        "big.weight_scale": deterministic_bytes(16 * 4, 25),
        # i8 rowAxis 1: 4 x 3 x 8 = 96 バイト・scale は [3, 1] の f32。
        "conv.weight": deterministic_bytes(96, 26),
        "conv.weight_scale": deterministic_bytes(3 * 4, 27),
        "const.a1b2c3d4e5f60718": deterministic_bytes(8 * 4, 31),
        "const.b1b2c3d4e5f6071a": deterministic_bytes(4 * 4, 32),
    }


def write_synthetic_container(path: Path, *, single: bool = True) -> list[Path]:
    """合成モデルを `path` へ書く（既定は単一形）。"""
    return write_model_container(
        path,
        synthetic_graph(),
        synthetic_tensors(),
        synthetic_bindings(),
        graph_name=GRAPH_NAME,
        provenance=Provenance(license="apache-2.0", writer="karume"),
        part_bytes=FIXTURE_PART_BYTES,
        block_bytes=FIXTURE_BLOCK_BYTES,
        single=single,
    )


def build_fixture_bytes(tmp_path: Path) -> bytes:
    """fixture のバイト列（`tmp_path` は書き出しの作業場）。"""
    written = write_synthetic_container(tmp_path / FIXTURE_PATH.name)
    return written[0].read_bytes()


def fixture_write_requested() -> bool:
    """`KARUME_FIXTURE=write` — 焼き直しを明示したときだけ resources を書き換える。"""
    return os.environ.get("KARUME_FIXTURE") == "write"
