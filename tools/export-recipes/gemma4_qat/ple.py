"""QAT PLE を固定 packed のまま**容器の資産**へ組む（ADR 0097 追記 4 / ADR 0109 決定 4）。"""

from __future__ import annotations

import math
from collections.abc import Buffer, Callable
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import torch
from safetensors.torch import save_file

from gemma4.export_product import (
    PLE_PROBE_FILE,
    PROBE_INPUTS_KEY,
    PROBE_TOKENS_KEY,
    ple_probe_tokens,
    spill_payload,
)
from karume.container import AssetInput
from karume.ple import ple_assets, ple_block_ranges

if TYPE_CHECKING:
    from transformers.integrations.gemma_quant import QuantizedEmbedding


class PleBuild(NamedTuple):
    """{@link build_ple} の戻り — 容器へ渡す資産と、参照の突合に要る事実。"""

    #: `publish_model(assets=…)` へそのまま渡す資産（索引 + `values` / `scales` の block 列）。
    assets: dict[str, AssetInput]
    #: 逆量子化ビット一致の参照に使う散点 token id。
    probe: tuple[int, ...]
    #: `values` の block 本数（`reference.json` の突合相手）。
    blocks: int
    #: token-major の payload を落とした作業席の一時ファイル（配布物ではない）。
    spills: tuple[Path, ...]

    def discard(self) -> None:
        """一時ファイルを消す（作業席ごと据わる前に呼ぶ MUST — 配布物に混ざらない）。"""
        for path in self.spills:
            path.unlink(missing_ok=True)


def _spill_reader(tensor: torch.Tensor, path: Path) -> Callable[[int, int], Buffer]:
    """行優先に連結済みの実体を作業席の一時ファイルへ落とし、その区間読みを返す。

    QAT の PLE は上流が既に token-major（`values[tokens, layers*dim//factor]` /
    `scales[tokens, layers]`）で持っているので、転置は要らず素のバイト列がそのまま payload。

    MUST: 実体を**持ち越さない**（製品系列の {@link gemma4.export_product._spill_tables} と
    同じ規律）— メモリ上のテンソルを掴む読み口にすると、packed の実体が `trace_qat` と
    書き出しの間ずっと常駐する。落としてしまえば呼び手は上流モジュールごと手放せる。
    """
    spill_payload(memoryview(tensor.numpy()).cast("B"), path)

    def read(begin: int, end: int) -> Buffer:
        with path.open("rb") as handle:
            handle.seek(begin)
            raw = handle.read(end - begin)
        if len(raw) != end - begin:
            raise AssertionError(f"{path}: PLE の区間 [{begin}, {end}) が途中で尽きた")
        return raw

    return read


def build_ple(module: QuantizedEmbedding, layers: int, dim: int, destination: Path) -> PleBuild:
    """固定表を容器の資産へ組み、散点参照（probe）を系列へ書く。

    `destination` は staging 側が作る席で、ここが書くのは**配布物でない** probe だけである
    （資産そのものは容器の中へ入るので、書くのは `publish_model` の仕事）。

    MUST: 上流の固定整数と scale を**再量子化しない** — 検査するのは形と値域だけで、
    バイト列は 1 ビットも作り替えずに資産の payload になる。
    """
    bits, tokens = module.num_bits, module.num_embeddings
    if bits not in (2, 4) or layers <= 0 or dim <= 0 or dim % 16:
        raise ValueError("PLE は I2/I4、正の layers、16の倍数の dim が必要")
    factor = 8 // bits
    values, scales = module.embedding_quantized.detach(), module.embedding_scale.detach()
    embed_scale = float(module.scalar_embed_scale)
    if not math.isfinite(embed_scale) or embed_scale <= 0:
        raise ValueError("PLE embedScale は正の有限数が必要")
    if values.device.type != "cpu" or values.dtype != torch.uint8 or not values.is_contiguous():
        raise ValueError("PLE packed values は CPU U8 の連続配置が必要")
    if values.shape != (tokens, layers * dim // factor):
        raise ValueError("PLE packed values の形が tokens/layers/dim と違う")
    if (
        scales.device.type != "cpu"
        or scales.dtype != torch.float32
        or scales.shape != (tokens, layers)
    ):
        raise ValueError("PLE scales は CPU F32 [tokens,layers] が必要")
    if not torch.isfinite(scales).all() or not (scales > 0).all():
        raise ValueError("PLE scales は正の有限数が必要")
    scales = scales.contiguous()
    storage = f"i{bits}"
    ranges = ple_block_ranges(storage=storage, tokens=tokens, layers=layers, dim=dim)
    probe = ple_probe_tokens(tokens, ranges["values"])
    with torch.inference_mode():
        expected = module(torch.tensor([list(probe)], dtype=torch.int64)).reshape(
            1, len(probe), layers, dim
        )
    save_file(
        {
            PROBE_TOKENS_KEY: torch.tensor(probe, dtype=torch.int32),
            PROBE_INPUTS_KEY: expected.contiguous(),
        },
        str(destination / PLE_PROBE_FILE),
    )
    spills = {key: destination / f".ple.{key}.spill" for key in ("values", "scales")}
    assets = ple_assets(
        storage=storage,
        tokens=tokens,
        layers=layers,
        dim=dim,
        embed_scale=embed_scale,
        read_values=_spill_reader(values, spills["values"]),
        read_scales=_spill_reader(scales, spills["scales"]),
    )
    return PleBuild(assets, probe, len(ranges["values"]), tuple(spills.values()))
