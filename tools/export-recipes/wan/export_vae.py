"""Wan の動画 VAE decoder の chunk グラフ 2 種を IR v2 へ書き出す台本（ADR 0118 / 0121 の段 4）。

モデル（`--model` — 既定 `t2v-1.3b`）ごとの系列・潜在タイル・フィクスチャのケースは
{@link VAE_SERIES} の表が一括で決める（渡し忘れても 2.1 の系列を 2.1 の値で書くだけで、別の
モデルの系列を黙って壊す経路は無い）:

| モデル | 系列（`outputs/series/` — 決定 7） | タイル t | 潜在 z | フレーム C |
|---|---|---|---|---|
| `t2v-1.3b`（Wan2.1） | `wan2.1-t2v-1.3b-f16-dyn`（ADR 0118） | 32 | 16 | 3（画素） |
| `ti2v-5b`（Wan2.2） | `wan2.2-ti2v-5b-f16-dyn`（ADR 0121） | 16 | 48 | 12（patchify 空間） |

書き出すもの:

- `vae_decoder_first/model.krm` — 最初の chunk 用。入力 = 潜在 `[z,1,t,t]` + cache 30 本・
  出力 = フレーム `[C,1,8t,8t]` + 更新後の cache 30 本（`time_conv` の 2 本を持たない）。
- `vae_decoder_next/model.krm` — それ以降用。入力 = 潜在 + cache 32 本・出力 = フレーム
  `[C,4,8t,8t]` + 更新後の cache 32 本。
- `vae_chunks.<case>.safetensors`（系列の根）— chunk 列の照合のフィクスチャ。固定 seed の乱数潜在
  （逆正規化済み `z·std + mean`）と、上流の非タイル `_decode` の chunk ループの**クランプ前**
  （2.2 は unpatchify の前でもある）の出力（{@link wan.vae_patch.reference_decode_unclamped}）。
  GPU の chunk 列の照合（段 4 の検収）が読む。ケースは帯を決める `band`（9 chunk = 33 フレーム）と、
  受け入れを判定する別の潜在・別の chunk 境界の `accept`（5 chunk = 17 フレーム）と `long`
  （21 chunk = 81 フレーム — ADR 0118 段 8・cache を 20 回持ち越す長さ）の 3 本（ADR 0118 追記
  2026-10-02 — 決定用と受入れ用を分ける）。seed はモデルごとに別。

タイル辺 `t`（潜在）は引数で上書きできる（既定は表 — 2.1 は diffusers の `tile_sample_min`
256 px ÷ 8、2.2 は 256 px ÷ 空間の圧縮 16）。ホストはタイル辺を literal で持たず、開いた資産の
入力形から導く（決定 2）。

格納は **f16 席だけ**（決定 7）: 重みを f16 表現可能値へ丸めて（fake-quant — ADR 0006）から
export とフィクスチャの参照を採る。丸めより前に参照を採ると、照合の差が量子化誤差と実装誤差の
合成になって帯の意味が消える。

    uv run --group wan --inexact python -m wan.export_vae              # 2 グラフ + フィクスチャ
    uv run --group wan --inexact python -m wan.export_vae --verify     # eager 同値の実測（実重み）
    uv run --group wan --inexact python -m wan.export_vae --model ti2v-5b [--verify]   # Wan2.2

MUST: diffusers / huggingface_hub は関数の中で import する（`wan` グループは既定の sync に
入らない — `tests/test_optional_group_imports.py`）。
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
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

#: chunk グラフの入出力の空間比（潜在 1 に対するフレームの辺 — 2.2 は patchify 空間の画素）。上流
#: config の `scale_factor_spatial` はこれ × `patch_size`（2.1 は 8 × 1・2.2 は 8 × 2 = 16 —
#: {@link assert_series_config} が見る・ADR 0121 決定 6）。
SPATIAL_SCALE = 8

#: next の chunk が出すフレーム数（時間 upsample 2 段 = 4 倍）。first は 1 枚。
NEXT_FRAMES = 4

#: chunk 列のフィクスチャのファイル名（系列の根に置く — `vae_chunks.<case>.safetensors`）。
FIXTURE_PREFIX = "vae_chunks."
FIXTURE_SUFFIX = ".safetensors"

#: フィクスチャのメタ `reference`（参照の素性）。patchify しない世代（2.1）と、chunk ループの
#: 出力が unpatchify の前の patchify 空間である世代（2.2）。
FIXTURE_REFERENCE = "diffusers AutoencoderKLWan._decode chunk loop before clamp (CPU f32)"
FIXTURE_REFERENCE_BEFORE_UNPATCHIFY = (
    "diffusers AutoencoderKLWan._decode chunk loop before unpatchify and clamp (CPU f32)"
)

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


#: 帯の決定用と受入れ用で、潜在（seed）と chunk 境界（chunk 数）を両方変える。`long` は 81 フレーム
#: （ADR 0118 段 8）の受入れで、帯は `band` のまま（帯を 21 chunk の結果から導き直さない）。
FIXTURE_CASES = (
    FixtureCase("band", seed=20261002, chunks=9, role="band"),
    FixtureCase("accept", seed=20261003, chunks=5, role="accept"),
    FixtureCase("long", seed=20261004, chunks=21, role="accept"),
)

#: Wan2.2 TI2V-5B の VAE の系列（ADR 0121 決定 7 — VAE だけの系列・DiT は持たない）。
TI2V_SERIES_NAME = "wan2.2-ti2v-5b-f16-dyn"

#: Wan2.2 の潜在タイルの辺（ADR 0121 決定 6 — 出力 256 px ÷ 空間の圧縮 16）。
TI2V_TILE = 16

#: Wan2.2 のケース（2.1 と同じ組み立て・seed だけ別 — 決定用 1 本 + 受入れ 2 本）。
TI2V_FIXTURE_CASES = (
    FixtureCase("band", seed=20261041, chunks=9, role="band"),
    FixtureCase("accept", seed=20261042, chunks=5, role="accept"),
    FixtureCase("long", seed=20261043, chunks=21, role="accept"),
)


@dataclass(frozen=True)
class VaeSeries:
    """モデル 1 つの VAE の系列（置き場・潜在タイル・patchify の倍率・フィクスチャのケース）。"""

    series: str
    tile: int
    #: 上流 config の `patch_size` の期待値（違えば fail loudly — {@link assert_series_config}）。
    patch_size: int | None
    cases: tuple[FixtureCase, ...]


#: モデル名（`wan.sources.SOURCES` のキー）→ 系列。
VAE_SERIES: dict[str, VaeSeries] = {
    "t2v-1.3b": VaeSeries(SERIES_NAME, DEFAULT_TILE, None, FIXTURE_CASES),
    "ti2v-5b": VaeSeries(TI2V_SERIES_NAME, TI2V_TILE, 2, TI2V_FIXTURE_CASES),
}


class ChunkGraphError(AssertionError):
    """export した chunk グラフが決定 2 の取り決め（rank・op・cache の入出力）から外れたか、書き手の
    前提（出所の revision・上流 config の `patch_size` と空間の圧縮）が系列と合わない。"""


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


def assert_series_config(vae: AutoencoderKLWan, series: VaeSeries) -> None:
    """上流 config が系列の前提（`patch_size` と空間の圧縮）と合うかを見る（外れたら fail loudly）。

    空間の圧縮 `scale_factor_spatial` は chunk グラフの入出力の空間比 {@link SPATIAL_SCALE} ×
    `patch_size` でなければならない — ホストの重なりの式（`vae_tiling.min_overlap_latent`）と
    フレームの辺 `8t` がこの比に依る。
    """
    config = vae.config
    if config.patch_size != series.patch_size:
        raise ChunkGraphError(
            f"上流の patch_size {config.patch_size} が系列 {series.series} の"
            f" {series.patch_size} でない"
        )
    expected = SPATIAL_SCALE * (series.patch_size or 1)
    if config.scale_factor_spatial != expected:
        raise ChunkGraphError(
            f"上流の scale_factor_spatial {config.scale_factor_spatial} が"
            f" {SPATIAL_SCALE} × patch_size = {expected} でない"
        )


def weight_summary(vae: AutoencoderKLWan) -> dict[str, Any]:
    """グラフが読む重み（decoder + post-quant）の要素数・最大絶対値・非有限の数（要約の記録用）。

    f16 の丸め（`round_weights_to_f16`）は有限値が非有限へ飽和すると fail loudly なので、丸めた後の
    非有限は 0 のはず — その実測をここで残す（ADR 0121 段 4 の検収）。
    """
    tensors = [*vae.decoder.parameters(), *vae.post_quant_conv.parameters()]
    with torch.no_grad():
        return {
            "elements": sum(tensor.numel() for tensor in tensors),
            "abs_max": max(float(tensor.abs().max()) for tensor in tensors),
            "nonfinite": sum(int((~torch.isfinite(tensor)).sum()) for tensor in tensors),
        }


def _peak_rss_gib() -> float:
    """このプロセスの RSS の最大（GiB — Linux の `ru_maxrss` は KiB）。"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)


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
      潜在とフレームのチャネル数はモジュール（post-quant の入力・`conv_out` の出力）から読む。
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
    latent_channels = int(module.post_quant_conv.in_channels)
    frame_channels = int(module.decoder.conv_out.out_channels)
    expected_inputs = [(LATENT_INPUT, [latent_channels, 1, tile, tile])]
    expected_inputs += [(slot.name, list(slot.shape(tile))) for slot in slots]
    actual_inputs = [(entry.name, [int(dim) for dim in entry.shape]) for entry in graph.inputs]
    if actual_inputs != expected_inputs:
        raise ChunkGraphError(f"入力 {actual_inputs} が表 {expected_inputs} と違う")

    if len(graph.outputs) != 1 + len(slots):
        raise ChunkGraphError(f"出力 {len(graph.outputs)} 本（期待 1 + {len(slots)}）")
    frames = 1 if module.first else NEXT_FRAMES
    frame_shape = [int(dim) for dim in graph.values[graph.outputs[0]].shape]
    if frame_shape != [frame_channels, frames, side, side]:
        raise ChunkGraphError(
            f"フレームの形 {frame_shape}（期待 [{frame_channels}, {frames}, {side}, {side}]）"
        )
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


@contextmanager
def _staged_set(finals: Sequence[Path]) -> Iterator[list[Path]]:
    """複数の final の作業席を `finals` の順で渡し、`with` を例外なく抜けたときにだけ全部を据える。

    1 つでも書き込み・検査で落ちれば全部の作業席が消え、どの final も 1 バイトも変わらない
    （{@link karume.artifacts.staged_publication} を席の数だけ重ねたもの）。据え替えは席ごとの
    rename なので、途中で落ちうる窓は最後の rename の列だけになる。
    """
    with ExitStack() as stack:
        yield [stack.enter_context(staged_publication(final)) for final in finals]


def _graph_summary(
    target: str, out_dir: Path, staged: Path, graph: IrGraph, tile: int, started: float
) -> dict[str, Any]:
    """1 グラフの要約（容器のバイト数は作業席の現物から数える — 据え替えは名前を変えるだけ）。

    `peak_rss_gib` はこのプロセスのここまでの RSS の最大（グラフの相の山。全体の山は要約の根の値）。
    """
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
        "cache_outputs": [
            [output, [int(dim) for dim in graph.values[output].shape]]
            for output in graph.outputs[1:]
        ],
        "initializers": len(graph.initializers),
        "compressed_tensors": breakdown.compressed_tensors,
        "compressed_bytes": breakdown.compressed_bytes,
        "plain_tensors": breakdown.plain_tensors,
        "plain_bytes": breakdown.plain_bytes,
        "model_bytes": sum(part.stat().st_size for part in container_parts(staged / MODEL_FILE)),
        "seconds": round(time.perf_counter() - started, 1),
        "peak_rss_gib": round(_peak_rss_gib(), 2),
    }


def fixture_path(out_root: Path, case: FixtureCase) -> Path:
    """chunk 列のフィクスチャの置き場（系列の根の `vae_chunks.<case>.safetensors`）。"""
    return out_root / f"{FIXTURE_PREFIX}{case.name}{FIXTURE_SUFFIX}"


def emit_targets(
    targets: Sequence[str],
    vae: AutoencoderKLWan,
    tile: int,
    out_root: Path,
    source: Provenance,
    *,
    cases: Sequence[FixtureCase],
    patch_size: int | None,
) -> dict[str, Any]:
    """グラフと chunk 列のフィクスチャ（`cases` — 空なら書かない）を一組で作業席へ書き、全部の
    検査を通してから据える。

    `patch_size` は系列の表の値（{@link VaeSeries}）で、フィクスチャのメタ `reference` の文言だけを
    決める（2.1 は unpatchify が無いので、文言は 2.1 の既存のフィクスチャと同じまま）。

    MUST: 一組で据える。グラフごと・ファイルごとに据えると、途中で落ちた実走が新旧の混ざった組
    （first だけ新しいタイル辺で、next と fixture は旧のまま）を系列に残す。タイル辺の違う混在は
    配布の門（`wan.distribution.assert_vae_chunk_pair`）と TS の `wanVaeChunkLayout` が拒むが、
    同じ形で中身の世代だけが違う混在はどちらにも見分けられない（出所は revision しか持たない —
    捕まえられるのは GPU の chunk 列の照合だけ）。
    """
    unknown = sorted(set(targets) - set(TARGETS))
    if unknown or len(set(targets)) != len(targets):
        raise ChunkGraphError(
            f"ターゲット {list(targets)} に未知か重複がある（既知: {', '.join(TARGETS)}）"
        )
    reference = FIXTURE_REFERENCE if patch_size is None else FIXTURE_REFERENCE_BEFORE_UNPATCHIFY
    out_root.mkdir(parents=True, exist_ok=True)
    finals = [out_root / target for target in targets]
    finals += [fixture_path(out_root, case) for case in cases]
    graphs: list[dict[str, Any]] = []
    written: list[dict[str, Any]] = []
    with _staged_set(finals) as seats:
        for target, staged in zip(targets, seats[: len(targets)], strict=True):
            started = time.perf_counter()
            module = vae_patch.WanVaeChunkDecoder(vae, first=target == TARGET_FIRST).eval()
            staged.mkdir()
            with torch.no_grad():
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
            graphs.append(_graph_summary(target, out_root / target, staged, graph, tile, started))
        for case, staged in zip(cases, seats[len(targets) :], strict=True):
            started = time.perf_counter()
            latents = fixture_latents(vae, case, tile)
            with torch.no_grad():
                frames = vae_patch.reference_decode_unclamped(vae, latents)[0]
            save_file(
                {"latents": latents[0].contiguous(), "frames": frames.contiguous()},
                str(staged),
                metadata={
                    "seed": str(case.seed),
                    "chunks": str(case.chunks),
                    "tile": str(tile),
                    "role": case.role,
                    "weights": "f16-rounded",
                    "reference": reference,
                },
            )
            written.append(
                {
                    "case": case.name,
                    "role": case.role,
                    "latents": list(latents.shape[1:]),
                    "frames": list(frames.shape),
                    "abs_max": float(frames.abs().max()),
                    "seconds": round(time.perf_counter() - started, 1),
                    "path": str(fixture_path(out_root, case)),
                }
            )
    summary: dict[str, Any] = {"series": str(out_root), "graphs": graphs}
    if cases:
        summary["fixtures"] = written
    return summary


def fixture_latents(vae: AutoencoderKLWan, case: FixtureCase, tile: int) -> torch.Tensor:
    """固定 seed の乱数潜在を逆正規化した `[1, z, F, t, t]`（上流 `WanPipeline` の decode 直前の
    値・z は上流 config の `z_dim` — 2.1 は 16・2.2 は 48）。

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


def verify(model: str, tile: int, chunks: int, seed: int) -> dict[str, Any]:
    """パッチの eager 同値を実重み（f32・丸め無し）で測る。

    - `clamp(unpatchify(上流のクランプ前の chunk ループ)) == 上流 _decode`（参照の素性 — ビット
      一致。unpatchify は patchify する世代〈2.2〉だけで、diffusers の関数そのものを当てる）
    - cache の正規化だけを当てた形（上流のコードのまま）== 上流（ビット一致 — 決定 2 の導出の実測）
    - 書き直しの最終形（{@link wan.vae_patch.chunk_decode}）と上流の差（記録）
    """
    from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

    series = VAE_SERIES[model]
    vae = load_vae(model, round_f16=False)
    assert_series_config(vae, series)
    case = FixtureCase("verify", seed=seed, chunks=chunks, role="verify")
    latents = fixture_latents(vae, case, tile)
    with torch.no_grad():
        reference = vae_patch.reference_decode_unclamped(vae, latents)
        decoded = vae._decode(latents, return_dict=False)[0]
        normalized = vae_patch.normalized_cache_decode(vae, latents, tile)
        first = vae_patch.WanVaeChunkDecoder(vae, first=True).eval()
        following = vae_patch.WanVaeChunkDecoder(vae, first=False).eval()
        final = vae_patch.chunk_decode(first, following, latents[0])
    restored = reference if series.patch_size is None else unpatchify(reference, series.patch_size)
    difference = (final - reference[0]).abs()
    reference_max = float(reference.abs().max())
    per_frame = [float(difference[:, index].max()) for index in range(difference.shape[1])]
    return {
        "model": model,
        "tile": tile,
        "chunks": chunks,
        "seed": seed,
        "frames": list(reference.shape[1:]),
        "reference_abs_max": reference_max,
        "clamped_reference_equals_decode": torch.equal(restored.clamp(-1.0, 1.0), decoded),
        "normalized_cache_bit_exact": torch.equal(normalized, reference),
        "final_bit_exact": torch.equal(final, reference[0]),
        "final_max_abs": float(difference.max()),
        "final_ratio": float(difference.max()) / reference_max,
        "final_max_abs_first_frame": per_frame[0],
        "final_max_abs_last_frame": per_frame[-1],
        "peak_rss_gib": round(_peak_rss_gib(), 2),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--model",
        choices=sorted(VAE_SERIES),
        default=DEFAULT_MODEL,
        help=f"モデル（系列・タイル・ケースは表から — 既定 {DEFAULT_MODEL}）",
    )
    parser.add_argument(
        "--tile", type=int, default=None, help="潜在タイルの辺（既定は表 — 2.1 は 32・2.2 は 16）"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="系列の根（既定はモデルの系列 — 表の値）"
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
    series = VAE_SERIES[args.model]
    tile = series.tile if args.tile is None else args.tile
    if tile <= 0:
        parser.error(f"--tile は正の整数（{tile}）")

    if args.verify:
        summary = verify(args.model, tile, args.chunks, args.seed)
        print(json.dumps(summary, indent=1, ensure_ascii=False))
        return 0

    source = provenance(args.model)
    vae = load_vae(args.model, round_f16=True)
    assert_series_config(vae, series)
    summary = emit_targets(
        args.target or TARGETS,
        vae,
        tile,
        args.out or SERIES_ROOT / series.series,
        source,
        cases=() if args.no_fixtures else series.cases,
        patch_size=series.patch_size,
    )
    summary["weights"] = weight_summary(vae)
    summary["peak_rss_gib"] = round(_peak_rss_gib(), 2)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
