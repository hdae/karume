"""Wan2.1 の動画 VAE decoder の chunk グラフ 2 種を IR v2 へ書き出す台本（ADR 0118 決定 2 / 段 4）。

書き出すもの（系列 `outputs/series/wan2.1-t2v-1.3b-f16-dyn/` — 決定 7）:

- `vae_decoder_first/model.krm` — 最初の chunk 用。入力 = 潜在 `[16,1,t,t]` + cache 30 本・
  出力 = フレーム `[3,1,8t,8t]` + 更新後の cache 30 本（`time_conv` の 2 本を持たない）。
- `vae_decoder_next/model.krm` — それ以降用。入力 = 潜在 + cache 32 本・出力 = フレーム
  `[3,4,8t,8t]` + 更新後の cache 32 本。
- `vae_chunks.<case>.safetensors`（系列の根）— chunk 列の照合のフィクスチャ。固定 seed の乱数潜在
  （逆正規化済み `z·std + mean`）と、上流の非タイル `_decode` の chunk ループの**クランプ前**の
  出力（{@link wan.vae_patch.reference_decode_unclamped}）。GPU の chunk 列の照合（段 4 の検収）が
  読む。ケースは帯を決める `band`（9 chunk = 33 フレーム）と、受け入れを判定する別の潜在・別の
  chunk 境界の `accept`（5 chunk = 17 フレーム）の 2 本（ADR 0118 追記 2026-10-02 — 決定用と
  受入れ用を分ける）。

タイル辺 `t`（潜在）は引数（既定 32 = diffusers の `tile_sample_min` 256 px と同じ大きさ）。ホストは
タイル辺を literal で持たず、開いた資産の入力形から導く（決定 2）。

格納は **f16 席だけ**（決定 7）: 重みを f16 表現可能値へ丸めて（fake-quant — ADR 0006）から
export とフィクスチャの参照を採る。丸めより前に参照を採ると、照合の差が量子化誤差と実装誤差の
合成になって帯の意味が消える。

    uv run --group wan --inexact python -m wan.export_vae              # 2 グラフ + フィクスチャ
    uv run --group wan --inexact python -m wan.export_vae --verify     # eager 同値の実測（実重み）

MUST: diffusers / huggingface_hub は関数の中で import する（`wan` グループは既定の sync に
入らない — `tests/test_optional_group_imports.py`）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import save_file

from _shared.paths import SERIES_ROOT
from _shared.upstream import snapshot_license
from karume.artifacts import staged_publication
from karume.container import Provenance, container_parts
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
from karume.dist import NOTICE_FILENAME
from karume.emit import storage_breakdown
from karume.ir import IrGraph
from karume.pipeline import export_to_file
from karume.quantize import round_weights_to_f16
from wan import vae_patch
from wan.sources import DEFAULT_MODEL, SOURCES, local_snapshot

if TYPE_CHECKING:
    from diffusers import AutoencoderKLWan

#: 系列名（決定 7 — 接尾辞 `-dyn` は DiT の S 形に合わせた ADR 0077 の慣例。VAE の 2 グラフも
#: 同じ系列）。
SERIES_NAME = "wan2.1-t2v-1.3b-f16-dyn"

#: ターゲット名 = 部品名 = 容器のグラフ名（決定 7）。
TARGET_FIRST = "vae_decoder_first"
TARGET_NEXT = "vae_decoder_next"
TARGETS = (TARGET_FIRST, TARGET_NEXT)

MODEL_FILE = "model.krm"

#: 潜在タイルの既定の辺（決定 2 — diffusers の `tile_sample_min` 256 px ÷ 8）。
DEFAULT_TILE = 32

#: 空間の縮小率（vae の config の `scale_factor_spatial`）。
SPATIAL_SCALE = 8

#: next の chunk が出すフレーム数（時間 upsample 2 段 = 4 倍）。first は 1 枚。
NEXT_FRAMES = 4

#: chunk 列のフィクスチャのファイル名（系列の根に置く — `vae_chunks.<case>.safetensors`）。
FIXTURE_PREFIX = "vae_chunks."
FIXTURE_SUFFIX = ".safetensors"

#: グラフ入力の綴り（`WanVaeChunkDecoder.forward` の引数名がそのまま IR の入力名になる）。
LATENT_INPUT = "latent"

#: chunk グラフの IR に現れてよい op の集合（これ以外が出たら書き直しの漏れ — fail loudly）。
#: SiLU は `x · sigmoid(x)` に分解される・mid の attention は SDPA の保存（ADR 0023）。
EXPECTED_OPS = frozenset(
    {
        "add",
        "attention",
        "cat",
        "clamp_min",
        "conv2d",
        "conv3d",
        "div",
        "expand",
        "mul",
        "permute",
        "reshape",
        "sigmoid",
        "slice",
        "sqrt",
        "sum",
    }
)

#: 値の rank の上限（strided コピー族の上限 — 決定 2）。rank 5 は conv3d の重みだけに許す。
MAX_VALUE_RANK = 4


@dataclass(frozen=True)
class FixtureCase:
    """chunk 列のフィクスチャ 1 本（潜在の seed と chunk 数）。"""

    name: str
    seed: int
    chunks: int
    #: `band` = 帯を決めるケース・`accept` = 受け入れを判定するケース（追記 2026-10-02）。
    role: str


#: 帯の決定用と受入れ用で、潜在（seed）と chunk 境界（chunk 数）を両方変える。
FIXTURE_CASES = (
    FixtureCase("band", seed=20261002, chunks=9, role="band"),
    FixtureCase("accept", seed=20261003, chunks=5, role="accept"),
)


class ChunkGraphError(AssertionError):
    """export した chunk グラフが決定 2 の取り決め（rank・op・cache の入出力）から外れた。"""


def load_vae(model: str = DEFAULT_MODEL, *, round_f16: bool) -> AutoencoderKLWan:
    """pin した revision の VAE を CPU f32 で読む（`round_f16` なら f16 表現可能値へ丸める）。"""
    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(
        local_snapshot(model), subfolder="vae", torch_dtype=torch.float32
    ).eval()
    if round_f16:
        report = round_weights_to_f16(vae)
        print(f"[fake-quant] vae: f16 表現可能値へ丸めた — {report.describe()}", flush=True)
    return vae


def provenance(model: str = DEFAULT_MODEL) -> Provenance:
    """容器へ焼く出所（ライセンスは手元の snapshot の README・revision は pin した値）。

    snapshot のディレクトリ名は HF キャッシュの commit SHA そのもので、`local_snapshot` は pin した
    revision しか解決しない。食い違いは fail loudly（別の checkpoint の出所を名乗らない）。
    """
    snapshot = local_snapshot(model)
    revision = SOURCES[model].revision
    if snapshot.name != revision:
        raise ChunkGraphError(f"snapshot {snapshot} が pin した revision {revision} でない")
    return Provenance(
        license=snapshot_license(snapshot), notice=NOTICE_FILENAME, upstream_revision=revision
    )


def example_inputs(
    module: vae_patch.WanVaeChunkDecoder, tile: int
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """export の例示入力（潜在 1 フレーム + ゼロの cache — 値は形にしか効かない）。"""
    channels = int(module.post_quant_conv.in_channels)
    latent = torch.zeros(channels, 1, tile, tile)
    return latent, vae_patch.zero_caches(module.slots, tile, first=module.first)


def assert_chunk_graph(graph: IrGraph, module: vae_patch.WanVaeChunkDecoder, tile: int) -> None:
    """chunk グラフの IR が決定 2 の取り決めを満たすことを見る（外れたら fail loudly）。

    - op は {@link EXPECTED_OPS} の中だけ（pad は無い — 因果パディングは cache の cat で表す）。
    - 値は rank 4 以下。rank 5 は conv3d の重み（initializer・第 2 入力）だけ（rank 6 の reshape も
      この検査で落ちる）。
    - 入力は `latent` + cache（名前・形・順が表どおり）・出力はフレーム + 更新後の cache（同じ順）。
    - 出力の cache k は「cache 入力 k を時間の先頭に cat した値の slice」から作られている
      （順を取り違えたグラフは、形が揃う cache どうしで黙って入れ替わる — 形の検査では掴めない）。
    """
    ops = Counter(node.op for node in graph.nodes)
    unexpected = sorted(set(ops) - EXPECTED_OPS)
    if unexpected:
        raise ChunkGraphError(f"想定外の op {unexpected}（書き直しの漏れ）")

    producer = {out: node for node in graph.nodes for out in node.outs}
    conv3d_weights = {node.ins[1] for node in graph.nodes if node.op == "conv3d"}
    for name, value in graph.values.items():
        rank = len(value.shape)
        if rank <= MAX_VALUE_RANK:
            continue
        if name not in graph.initializers or name not in conv3d_weights or rank != 5:
            raise ChunkGraphError(f"rank {rank} の値 '{name}'（rank 5 は conv3d の重みだけ）")

    slots = module.cache_slots
    side = tile * SPATIAL_SCALE
    expected_inputs = [(LATENT_INPUT, [16, 1, tile, tile])]
    expected_inputs += [(slot.name, list(slot.shape(tile))) for slot in slots]
    actual_inputs = [(entry.name, [int(dim) for dim in entry.shape]) for entry in graph.inputs]
    if actual_inputs != expected_inputs:
        raise ChunkGraphError(f"入力 {actual_inputs} が表 {expected_inputs} と違う")

    if len(graph.outputs) != 1 + len(slots):
        raise ChunkGraphError(f"出力 {len(graph.outputs)} 本（期待 1 + {len(slots)}）")
    frames = 1 if module.first else NEXT_FRAMES
    frame_shape = [int(dim) for dim in graph.values[graph.outputs[0]].shape]
    if frame_shape != [3, frames, side, side]:
        raise ChunkGraphError(f"フレームの形 {frame_shape}（期待 [3, {frames}, {side}, {side}]）")
    for slot, output in zip(slots, graph.outputs[1:], strict=True):
        shape = [int(dim) for dim in graph.values[output].shape]
        if shape != list(slot.shape(tile)):
            raise ChunkGraphError(f"出力 '{output}' の形 {shape} が {slot.name} と違う")
        sliced = producer.get(output)
        joined = producer.get(sliced.ins[0]) if sliced is not None and sliced.ins else None
        if (
            sliced is None
            or sliced.op != "slice"
            or joined is None
            or joined.op != "cat"
            or joined.ins[0] != slot.name
        ):
            raise ChunkGraphError(
                f"出力 '{output}' が cat({slot.name}, …) の slice から作られていない"
                "（順の取り違え）"
            )


def emit_target(
    target: str, vae: AutoencoderKLWan, tile: int, out_root: Path, source: Provenance
) -> dict[str, Any]:
    """1 グラフを export し、取り決めの検査を通してから系列へ据える。要約を返す。"""
    if target not in TARGETS:
        raise ChunkGraphError(f"未知のターゲット '{target}'（既知: {', '.join(TARGETS)}）")
    started = time.perf_counter()
    module = vae_patch.WanVaeChunkDecoder(vae, first=target == TARGET_FIRST).eval()
    out_dir = out_root / target
    out_root.mkdir(parents=True, exist_ok=True)
    with staged_publication(out_dir) as staged, torch.no_grad():
        staged.mkdir()
        graph = export_to_file(
            module,
            example_inputs(module, tile),
            staged / MODEL_FILE,
            provenance=source,
            graph_name=target,
            weight_dtype="f16",
            preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
        )
        assert_chunk_graph(graph, module, tile)
    breakdown = storage_breakdown(graph)
    return {
        "target": target,
        "dir": str(out_dir),
        "tile": tile,
        "nodes": len(graph.nodes),
        "op_counts": dict(sorted(Counter(node.op for node in graph.nodes).items())),
        "inputs": [[entry.name, [int(dim) for dim in entry.shape]] for entry in graph.inputs],
        "frame_shape": [int(dim) for dim in graph.values[graph.outputs[0]].shape],
        "caches": len(graph.outputs) - 1,
        "initializers": len(graph.initializers),
        "compressed_tensors": breakdown.compressed_tensors,
        "compressed_bytes": breakdown.compressed_bytes,
        "plain_tensors": breakdown.plain_tensors,
        "plain_bytes": breakdown.plain_bytes,
        "model_bytes": sum(part.stat().st_size for part in container_parts(out_dir / MODEL_FILE)),
        "seconds": round(time.perf_counter() - started, 1),
    }


def fixture_latents(vae: AutoencoderKLWan, case: FixtureCase, tile: int) -> torch.Tensor:
    """固定 seed の乱数潜在を逆正規化した `[1, 16, F, t, t]`（上流 `WanPipeline` の decode 直前の
    値）。

    `WanPipeline` は `latents / (1/std) + mean` で逆正規化してから `vae.decode` へ渡す。グラフの
    入力はその後の値なので、フィクスチャも実運用と同じ値域で作る（帯の根拠を実運用の値域と
    対応させる — dacvae の golden と同じ判断）。
    """
    generator = torch.Generator().manual_seed(case.seed)
    channels = int(vae.config.z_dim)
    mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(1, channels, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, dtype=torch.float32).view(1, channels, 1, 1, 1)
    noise = torch.randn(1, channels, case.chunks, tile, tile, generator=generator)
    return noise * std + mean


def write_fixtures(vae: AutoencoderKLWan, tile: int, out_root: Path) -> list[dict[str, Any]]:
    """chunk 列のフィクスチャを系列の根へ書く（潜在 + 上流のクランプ前の出力）。"""
    out_root.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, Any]] = []
    for case in FIXTURE_CASES:
        started = time.perf_counter()
        latents = fixture_latents(vae, case, tile)
        with torch.no_grad():
            frames = vae_patch.reference_decode_unclamped(vae, latents)[0]
        path = out_root / f"{FIXTURE_PREFIX}{case.name}{FIXTURE_SUFFIX}"
        staging = path.with_name(path.name + ".staging")
        save_file(
            {"latents": latents[0].contiguous(), "frames": frames.contiguous()},
            str(staging),
            metadata={
                "seed": str(case.seed),
                "chunks": str(case.chunks),
                "tile": str(tile),
                "role": case.role,
                "weights": "f16-rounded",
                "reference": "diffusers AutoencoderKLWan._decode chunk loop before clamp (CPU f32)",
            },
        )
        staging.replace(path)
        written.append(
            {
                "case": case.name,
                "role": case.role,
                "latents": list(latents.shape[1:]),
                "frames": list(frames.shape),
                "abs_max": float(frames.abs().max()),
                "seconds": round(time.perf_counter() - started, 1),
                "path": str(path),
            }
        )
    return written


def verify(tile: int, chunks: int, seed: int) -> dict[str, Any]:
    """パッチの eager 同値を実重み（f32・丸め無し）で測る。

    - `clamp(上流のクランプ前の chunk ループ) == 上流 _decode`（参照の素性 — ビット一致）
    - cache の正規化だけを当てた形（上流のコードのまま）== 上流（ビット一致 — 決定 2 の導出の実測）
    - 書き直しの最終形（{@link wan.vae_patch.chunk_decode}）と上流の差（記録）
    """
    vae = load_vae(round_f16=False)
    case = FixtureCase("verify", seed=seed, chunks=chunks, role="verify")
    latents = fixture_latents(vae, case, tile)
    with torch.no_grad():
        reference = vae_patch.reference_decode_unclamped(vae, latents)
        decoded = vae._decode(latents, return_dict=False)[0]
        normalized = vae_patch.normalized_cache_decode(vae, latents, tile)
        first = vae_patch.WanVaeChunkDecoder(vae, first=True).eval()
        following = vae_patch.WanVaeChunkDecoder(vae, first=False).eval()
        final = vae_patch.chunk_decode(first, following, latents[0])
    difference = (final - reference[0]).abs()
    reference_max = float(reference.abs().max())
    per_frame = [float(difference[:, index].max()) for index in range(difference.shape[1])]
    return {
        "tile": tile,
        "chunks": chunks,
        "seed": seed,
        "frames": list(reference.shape[1:]),
        "reference_abs_max": reference_max,
        "clamped_reference_equals_decode": torch.equal(reference.clamp(-1.0, 1.0), decoded),
        "normalized_cache_bit_exact": torch.equal(normalized, reference),
        "final_bit_exact": torch.equal(final, reference[0]),
        "final_max_abs": float(difference.max()),
        "final_ratio": float(difference.max()) / reference_max,
        "final_max_abs_first_frame": per_frame[0],
        "final_max_abs_last_frame": per_frame[-1],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--tile", type=int, default=DEFAULT_TILE, help="潜在タイルの辺（既定 32）")
    parser.add_argument(
        "--out",
        type=Path,
        default=SERIES_ROOT / SERIES_NAME,
        help="系列の根（既定は決定 7 の系列）",
    )
    parser.add_argument(
        "--target", action="append", choices=TARGETS, default=None, help="書くグラフ（既定は両方）"
    )
    parser.add_argument(
        "--no-fixtures", action="store_true", help="chunk 列のフィクスチャを書かない"
    )
    parser.add_argument("--verify", action="store_true", help="eager 同値を測るだけ（書かない）")
    parser.add_argument("--chunks", type=int, default=9, help="--verify の chunk 数（既定 9）")
    parser.add_argument("--seed", type=int, default=0, help="--verify の潜在の seed")
    args = parser.parse_args(argv)
    if args.tile <= 0:
        parser.error(f"--tile は正の整数（{args.tile}）")

    if args.verify:
        print(json.dumps(verify(args.tile, args.chunks, args.seed), indent=1, ensure_ascii=False))
        return 0

    source = provenance()
    vae = load_vae(round_f16=True)
    summary: dict[str, Any] = {
        "series": str(args.out),
        "graphs": [
            emit_target(target, vae, args.tile, args.out, source)
            for target in (args.target or TARGETS)
        ],
    }
    if not args.no_fixtures:
        summary["fixtures"] = write_fixtures(vae, args.tile, args.out)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
