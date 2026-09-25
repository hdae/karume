"""packed PLE の合成 fixture を作る（実モデルの値は複製しない）。

    cd tools/export-recipes && uv run python -m gemma4.tests.ple_fixture

期待値は符号付き整数と Torch の二段 f32 乗算から作る。packed reader と同じ
ビット展開式で期待値を作らず、全符号値・層別 scale・行境界を独立に検査する。

MUST: 資産は**recipe の本番経路と同じ関数**で焼く（{@link karume.ple.ple_assets} が索引と
block の切り方を持ち、{@link karume.pipeline.publish_model} が容器を据える）。fixture 専用の
書き出しを別に綴ると、読み手側のテストは「fixture の書き方」を検査するだけになり、実際に
配布される容器の形とずれたまま緑になる。

MUST: 2 度回してバイト同一であること（生成器タグも時刻も焼かない — `_PROVENANCE` は
`karume.goldens.GOLDEN_PROVENANCE` と同じ理由で `license` だけ）。git 追跡下の fixture なので、
回すたびに差分が出る形にすると「資産が変わった」の意味が消える。
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

#: fixture の置き場（読み手は `packages/models/tests/gemma4_ple_packed_test.ts`）。
FIXTURE_ROOT = REPO_ROOT / "packages" / "models" / "tests" / "fixtures" / "gemma4-ple-packed"

#: 容器の代表 path（TS 側 `helpers/ple-series.ts` の `SERIES_MODEL_FILE` と同じ綴り）。
MODEL_FILE = "model.krm"

#: 容器の中のグラフ名（= 配布 manifest の weights キー — container-v1 §2.1）。
GRAPH_NAME = "model"

#: 出所。`writer` は**書かない** — 生成器タグを焼くと、パッケージ版を上げただけで fixture の
#: バイト列が動き、「資産が変わった」と読める再生成の前提が壊れる（goldens と同じ扱い）。
_PROVENANCE = Provenance(license="mit")

#: 合成の寸法（token / 層 / 次元）。小さいまま**行境界と block 境界の両方**を踏ませる。
TOKENS, LAYERS, DIM = 9, 3, 32

#: 全層に掛かる後段の係数（索引 `embedScale`）。
EMBED_SCALE = 3.5

#: PLE の block を切る上限（**PLE の資産だけに掛ける** — 容器全体の block 上限は既定のまま。
#: 索引の JSON はここより大きいので、全体へ掛けると `_plan_asset_parts` が落ちる）。
#:
#: **values と scales で切り目がずれる**値を選ぶ（144 = values i4 で 3 行・i2 で 6 行・
#: scales で 12 行 = 1 block）。両表が同じ境界で切れる fixture では、表ごとに別の block を
#: 引く読み手の誤り（scales を values の block 番号で引く類）が素通りする。
BLOCK_BYTES = 144


class PleFixtureGraph(nn.Module):
    """資産を載せるための最小グラフ。

    この fixture の門が読むのは PLE の資産だけだが、容器は**グラフを持つ模型容器**
    （`krm` の kind = model）なので、重みが 1 本ある実際の形で焼く。値は決定的
    （`arange`）— 乱数を引くと 2 度回したときにバイトが動く。
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.arange(dim, dtype=torch.float32) / dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


def _packed_values(bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """符号付きの素の値 `[tokens, layers, dim]` と、詰めた `[tokens, layers, dim/factor]`。"""
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
    """連結した行優先 payload の区間読み（{@link ple_assets} の受け口）。

    MUST: 同じ区間からは同じバイト列を返す（書き手は sha256 を採るときと書くときの 2 度引く）。
    """

    def read(begin: int, end: int) -> bytes:
        if begin % stride or end % stride:
            raise AssertionError(f"PLE の区間 [{begin}, {end}) が 1 行 {stride} バイトの倍数でない")
        return payload[begin:end]

    return read


def write_fixture(out: Path) -> None:
    """I2 / I4 の packed PLE を**容器の資産**として書き、CPU 参照を隣に置く。"""
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
