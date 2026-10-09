"""Wan2.2 TI2V-5B の VAE encoder の chunk 0（I2V の条件の画像 1 枚）を IR v2 の 3 グラフへ書き出し、
GPU の門の golden を作る台本（ADR 0121 決定 11・段 9a）。

書き出すもの（系列 `outputs/series/wan2.2-ti2v-5b-f16-dyn/` — decoder の chunk グラフと同じ系列）:

- `vae_encoder_pre/model.krm` — 入力 `image` `[12, 1, 8h, 8w]`（patchify 後の画像）→
  `[640, 1, h, w]`（記号 h, w）。
- `vae_encoder_attn/model.krm` — 入力 `tokens` `[640, S]` → `[640, S]`（記号 S = h·w）。
- `vae_encoder_post/model.krm` — 入力 `hidden` `[640, 1, h, w]` → mu `[48, 1, h, w]`（記号 h, w）。
- `vae_encoder.<case>.safetensors`（系列の根）— GPU の門の golden（{@link GOLDEN_CASES}）。

容器は**グラフ 1 本ずつの 3 本**（部品 3 つ）。容器の形式は 1 容器に複数グラフを載せられるが、
manifest `karume/5` は「1 容器 1 グラフ・グラフ名 = 部品名 = weights のキー」で使わない
（ADR 0109 決定 2・追記 1）— 書き手（`karume.container` の `_graph_descriptor`）も 1 グラフしか
書かない。置き場の綴りは decoder（`vae_decoder_first/` / `vae_decoder_next/`）と同じ
「系列の根 / 部品名 / model.krm」。3 グラフの重みは互いに重ならない（重複は無い）。

格納は **f16 席だけ**（decoder と同じ — 決定 7）: 重みを f16 表現可能値へ丸めて（fake-quant —
ADR 0006）から 3 グラフと golden の参照を採る（丸めより前に参照を採ると、照合の差が量子化誤差と
実装誤差の合成になって帯の意味が消える）。

golden のケースはテスト画像 3 枚（`inputs/wan-i2v/<name>-832x480.png` — git 追跡外・sha256 を
{@link IMAGES} で固定）× 3 寸法（1280×704・704×1280〈横長の画像を明示した縦長にクロップ〉・
256×160〈縮小の経路 — 潜在 10×16〉）を fit = `crop`（公式の前処理）で、帯の決定用 2 枚
（boxing-cats・cat-dog-baking）と受入れ 1 枚（ferret）に割り振る（decoder の chunk の e2e と同じ
規律 — 受入れの結果を見て帯を変えない）。加えて fit = `stretch`（diffusers の前処理）の RGB8 だけを
1 寸法（1280×704）ぶん持つ（TS のホストテスト用・role `host`）。

    uv run --group wan --inexact python -m wan.export_vae_encoder             # 3 グラフ + golden
    uv run --group wan --inexact python -m wan.export_vae_encoder --verify    # eager 同値の実測

MUST: diffusers / Pillow は関数の中で import する（`wan` グループは既定の sync に入らない —
`tests/test_optional_group_imports.py`）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import save_file

from _shared.paths import INPUTS_ROOT, SERIES_ROOT
from karume.container import Provenance, container_parts
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
from karume.emit import storage_breakdown
from karume.ir import IrGraph
from karume.pipeline import export_to_file
from wan import export_vae, i2v_preprocess_ref, vae_encoder_patch
from wan.i2v_preprocess_ref import Fit
from wan.sources import SOURCES

if TYPE_CHECKING:
    from diffusers import AutoencoderKLWan

#: モデル（`wan.sources.SOURCES` のキー）と系列（decoder と同じ系列 — ADR 0121 決定 7）。
MODEL = "ti2v-5b"
SERIES_NAME = export_vae.TI2V_SERIES_NAME

#: ターゲット名 = 部品名 = 容器のグラフ名 = 部品ディレクトリ名。
TARGET_PRE = "vae_encoder_pre"
TARGET_ATTENTION = "vae_encoder_attn"
TARGET_POST = "vae_encoder_post"
TARGETS = (TARGET_PRE, TARGET_ATTENTION, TARGET_POST)

MODEL_FILE = "model.krm"

#: グラフ入力の綴り（各モジュールの forward の引数名がそのまま IR の入力名になる）。
INPUT_NAMES = {TARGET_PRE: "image", TARGET_ATTENTION: "tokens", TARGET_POST: "hidden"}

#: 記号の名前（pre / post は潜在の高さ・幅、attn は系列長 S = h·w）。
LATENT_SYMBOLS = ("h", "w")
SEQUENCE_SYMBOL = "S"

#: 記号の範囲（export の制約だけに効き、IR には載らない）。下限 2 は torch.export の 0 / 1 の
#: 特殊化を避ける線（DiT の `Dim("S")` と同じ）。上限は受理集合（潜在の辺 ≤ 80）より広く取る —
#: 受理寸法の門はパイプラインの側に置く（資産は解像度から独立 — ADR 0038 §4）。
LATENT_MIN = 2
LATENT_MAX = 160
SEQUENCE_MIN = 2
SEQUENCE_MAX = LATENT_MAX * LATENT_MAX

#: pre の入力の空間比（潜在 1 に対する patchify 後の画像の辺 — down 3 段で 1/8）。
PATCH_SCALE = 8

#: export の例示入力の潜在の寸法（1280×704 の横長 — h ≠ w にして記号の取り違えを形で掴む）。
EXAMPLE_LATENT = (44, 80)

#: グラフごとの IR に現れてよい op の集合（これ以外が出たら書き直しの漏れ — fail loudly）。
#: SiLU は `x · sigmoid(x)` に分解される・mid の attention は SDPA の保存（ADR 0023）・pad は
#: down の ZeroPad2d だけ・slice / cat はショートカットのチャネル対（と mu の切り出し・qkv の
#: 分割）。
EXPECTED_OPS: dict[str, frozenset[str]] = {
    TARGET_PRE: frozenset(
        {
            "add",
            "cat",
            "clamp_min",
            "conv2d",
            "conv3d",
            "div",
            "mul",
            "pad",
            "permute",
            "reshape",
            "sigmoid",
            "slice",
            "sqrt",
            "sum",
        }
    ),
    TARGET_ATTENTION: frozenset(
        {
            "add",
            "attention",
            "clamp_min",
            "div",
            "linear",
            "mul",
            "permute",
            "reshape",
            "slice",
            "sqrt",
            "sum",
        }
    ),
    TARGET_POST: frozenset(
        {
            "add",
            "clamp_min",
            "conv3d",
            "div",
            "mul",
            "reshape",
            "sigmoid",
            "slice",
            "sqrt",
            "sum",
        }
    ),
}

#: 値の rank の上限（strided コピー族の上限）。rank 5 は conv3d の重みだけに許す。
MAX_VALUE_RANK = 4


class EncoderGraphError(AssertionError):
    """export した encoder のグラフが取り決め（rank・op・記号・入出力の形）から外れた。"""


# ---- golden のケース ----------------------------------------------------------------


@dataclass(frozen=True)
class GoldenImage:
    """テスト画像 1 枚（`inputs/wan-i2v/<name>-832x480.png` — git 追跡外）と、その sha256。"""

    name: str
    sha256: str

    @property
    def path(self) -> Path:
        return IMAGE_DIR / f"{self.name}-{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}.png"


#: テスト画像の置き場と寸法（Wan2.1 の視認の先頭フレームの凍結コピー — 3 枚とも 832×480）。
IMAGE_DIR = INPUTS_ROOT / "wan-i2v"
IMAGE_SIZE = (832, 480)

#: テスト画像（sha256 は来歴の正本 — 食い違えば fail loudly）。
IMAGES = (
    GoldenImage("boxing-cats", "4689f815503859b4d08b464f21239b0ab5066b4da06e7130aa32a60ef1fe5888"),
    GoldenImage(
        "cat-dog-baking", "75d6d9c514190806baa4fc5a9d0877e04d4c50ba940057d737915bfb354f3c0e"
    ),
    GoldenImage("ferret", "a6b740a2ec936d2a102839441f9caabd787fc1d4e27126bbed4be429aef7bd54"),
)

#: 帯の決定用と受入れの割り振り（画像で分ける — 寸法ごとに決定用 2 本・受入れ 1 本）。
BAND_IMAGES = ("boxing-cats", "cat-dog-baking")
ACCEPT_IMAGES = ("ferret",)

#: encoder の寸法（幅, 高さ）: 受理集合の 2 寸法（704×1280 は横長の画像を明示した縦長）と、
#: 縮小の経路（832×480 → 277×160 → 256×160・潜在 10×16 — 受理の外だがグラフは寸法に依らない）。
ENCODER_SIZES = ((1280, 704), (704, 1280), (256, 160))

#: fit = `stretch` の RGB8 を作る寸法（TS のホストテスト用 — 1 寸法ぶん）。
STRETCH_SIZE = (1280, 704)


@dataclass(frozen=True)
class GoldenCase:
    """golden 1 本（画像・出力寸法・fit・役割）。"""

    image: str
    width: int
    height: int
    fit: Fit
    #: `band` = 帯の決定用・`accept` = 受入れ・`host` = 前処理だけ（GPU を回さない）。
    role: str

    @property
    def name(self) -> str:
        suffix = "" if self.fit == "crop" else f"-{self.fit}"
        return f"{self.image}-{self.width}x{self.height}{suffix}"

    @property
    def encodes(self) -> bool:
        """encoder の入力と mu を持つか（`stretch` は前処理の RGB8 だけ）。"""
        return self.fit == "crop"


def _golden_cases() -> tuple[GoldenCase, ...]:
    cases = [
        GoldenCase(image, width, height, "crop", "band" if image in BAND_IMAGES else "accept")
        for width, height in ENCODER_SIZES
        for image in (*BAND_IMAGES, *ACCEPT_IMAGES)
    ]
    cases += [GoldenCase(image.name, *STRETCH_SIZE, "stretch", "host") for image in IMAGES]
    return tuple(cases)


GOLDEN_CASES = _golden_cases()

#: golden の置き場（系列の根の `vae_encoder.<case>.safetensors`）。
GOLDEN_PREFIX = "vae_encoder."
GOLDEN_SUFFIX = ".safetensors"

#: golden のメタ `reference`（参照の素性）。
GOLDEN_REFERENCE = (
    "diffusers AutoencoderKLWan.encode (non-tiled) latent_dist.mode() (CPU f32), normalized as"
    " WanImageToVideoPipeline.prepare_latents"
)


def golden_path(out_root: Path, case: GoldenCase) -> Path:
    return out_root / f"{GOLDEN_PREFIX}{case.name}{GOLDEN_SUFFIX}"


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_image(image: GoldenImage) -> Any:
    """テスト画像を RGB で開く（sha256 と寸法が来歴と違えば fail loudly）。"""
    if not image.path.is_file():
        raise FileNotFoundError(
            f"テスト画像 {image.path} が無い（inputs/ は git 追跡外 — 凍結コピーを置く）"
        )
    digest = _sha256(image.path)
    if digest != image.sha256:
        raise EncoderGraphError(f"{image.path} の sha256 {digest} が来歴 {image.sha256} と違う")
    opened = i2v_preprocess_ref.load_rgb(image.path)
    if opened.size != IMAGE_SIZE:
        raise EncoderGraphError(f"{image.path} の寸法 {opened.size} が {IMAGE_SIZE} でない")
    return opened


def library_versions() -> dict[str, str]:
    """値を決めるライブラリの版（golden の来歴 — Pillow は LANCZOS の実装を決める）。"""
    return {name: version(name) for name in ("pillow", "torch", "torchvision", "diffusers")}


# ---- グラフ --------------------------------------------------------------------------


def _hidden_channels(vae: AutoencoderKLWan) -> int:
    """mid block のチャネル数（`base_dim × dim_mult[-1]` — 5B は 640）。"""
    return int(vae.config.base_dim) * int(vae.config.dim_mult[-1])


def example_inputs(target: str, vae: AutoencoderKLWan) -> tuple[torch.Tensor]:
    """export の例示入力（値は形にしか効かない）。"""
    height, width = EXAMPLE_LATENT
    hidden = _hidden_channels(vae)
    if target == TARGET_PRE:
        channels = int(vae.encoder.conv_in.in_channels)
        return (torch.zeros(channels, 1, PATCH_SCALE * height, PATCH_SCALE * width),)
    if target == TARGET_ATTENTION:
        return (torch.zeros(hidden, height * width),)
    if target == TARGET_POST:
        return (torch.zeros(hidden, 1, height, width),)
    raise EncoderGraphError(f"未知のターゲット {target}")


def dynamic_shapes(target: str) -> tuple[Any, ...]:
    """記号の宣言（pre は `8h` / `8w`・attn は `S`・post は `h` / `w`）。"""
    from torch.export import Dim

    if target == TARGET_ATTENTION:
        return ({1: Dim(SEQUENCE_SYMBOL, min=SEQUENCE_MIN, max=SEQUENCE_MAX)},)
    height = Dim(LATENT_SYMBOLS[0], min=LATENT_MIN, max=LATENT_MAX)
    width = Dim(LATENT_SYMBOLS[1], min=LATENT_MIN, max=LATENT_MAX)
    scale = PATCH_SCALE if target == TARGET_PRE else 1
    return ({2: scale * height, 3: scale * width},)


def symbol_names(target: str) -> tuple[str, ...]:
    return (SEQUENCE_SYMBOL,) if target == TARGET_ATTENTION else LATENT_SYMBOLS


def expected_io(target: str, vae: AutoencoderKLWan) -> tuple[list[Any], list[Any]]:
    """グラフの入力と出力の宣言 shape（記号は IR の綴り `8h` / `h` / `S`）。"""
    hidden = _hidden_channels(vae)
    latent = [hidden, 1, "h", "w"]
    if target == TARGET_PRE:
        channels = int(vae.encoder.conv_in.in_channels)
        return [channels, 1, f"{PATCH_SCALE}h", f"{PATCH_SCALE}w"], latent
    if target == TARGET_ATTENTION:
        return [hidden, "S"], [hidden, "S"]
    if target == TARGET_POST:
        return latent, [int(vae.config.z_dim), 1, "h", "w"]
    raise EncoderGraphError(f"未知のターゲット {target}")


def assert_encoder_graph(graph: IrGraph, target: str, vae: AutoencoderKLWan) -> None:
    """encoder のグラフの IR が取り決めを満たすことを見る（外れたら fail loudly）。

    - op は {@link EXPECTED_OPS} のそのグラフの集合の中だけ。
    - 値は rank 4 以下。rank 5 は conv3d の重み（initializer・第 2 入力）だけ。
    - 記号は宣言どおり（pre / post は `[h, w]`・attn は `[S]`）で、入力は 1 本（名前と形）・出力は
      1 本（形）。記号の積（`h·w`）は IR の次元式に載らない — 載ったら export が既に落ちている。
    """
    ops = Counter(node.op for node in graph.nodes)
    unexpected = sorted(set(ops) - EXPECTED_OPS[target])
    if unexpected:
        raise EncoderGraphError(f"{target}: 想定外の op {unexpected}（書き直しの漏れ）")

    conv3d_weights = {node.ins[1] for node in graph.nodes if node.op == "conv3d"}
    for name, value in graph.values.items():
        rank = len(value.shape)
        if rank <= MAX_VALUE_RANK:
            continue
        if name not in graph.initializers or name not in conv3d_weights or rank != 5:
            raise EncoderGraphError(
                f"{target}: rank {rank} の値 '{name}'（rank 5 は conv3d の重みだけ）"
            )

    expected_symbols = list(symbol_names(target))
    if list(graph.symbols) != expected_symbols:
        raise EncoderGraphError(f"{target}: 記号 {graph.symbols} が {expected_symbols} でない")
    expected_input, expected_output = expected_io(target, vae)
    actual_inputs = [(entry.name, list(entry.shape)) for entry in graph.inputs]
    if actual_inputs != [(INPUT_NAMES[target], expected_input)]:
        raise EncoderGraphError(
            f"{target}: 入力 {actual_inputs} が [({INPUT_NAMES[target]!r}, {expected_input})]"
            " でない"
        )
    if len(graph.outputs) != 1:
        raise EncoderGraphError(f"{target}: 出力 {len(graph.outputs)} 本（期待 1）")
    output_shape = list(graph.values[graph.outputs[0]].shape)
    if output_shape != expected_output:
        raise EncoderGraphError(f"{target}: 出力の形 {output_shape} が {expected_output} でない")


def graph_module(target: str, vae: AutoencoderKLWan) -> torch.nn.Module:
    """ターゲットのモジュール（eval — 重みは今の VAE の値の写し）。"""
    builders = {
        TARGET_PRE: vae_encoder_patch.VaeEncoderPre,
        TARGET_ATTENTION: vae_encoder_patch.VaeEncoderAttention,
        TARGET_POST: vae_encoder_patch.VaeEncoderPost,
    }
    if target not in builders:
        raise EncoderGraphError(f"未知のターゲット {target}（既知: {', '.join(TARGETS)}）")
    return builders[target](vae).eval()


def weight_summary(vae: AutoencoderKLWan) -> dict[str, Any]:
    """encoder + quant_conv の重みの要素数・最大絶対値・非有限の数（要約の記録用）。"""
    tensors = [*vae.encoder.parameters(), *vae.quant_conv.parameters()]
    with torch.no_grad():
        return {
            "elements": sum(tensor.numel() for tensor in tensors),
            "abs_max": max(float(tensor.abs().max()) for tensor in tensors),
            "nonfinite": sum(int((~torch.isfinite(tensor)).sum()) for tensor in tensors),
        }


def _peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)


def _graph_summary(
    target: str, out_dir: Path, staged: Path, graph: IrGraph, started: float
) -> dict[str, Any]:
    breakdown = storage_breakdown(graph)
    return {
        "target": target,
        "dir": str(out_dir),
        "symbols": list(graph.symbols),
        "inputs": [[entry.name, list(entry.shape)] for entry in graph.inputs],
        "output": list(graph.values[graph.outputs[0]].shape),
        "nodes": len(graph.nodes),
        "op_counts": dict(sorted(Counter(node.op for node in graph.nodes).items())),
        "initializers": len(graph.initializers),
        "compressed_tensors": breakdown.compressed_tensors,
        "compressed_bytes": breakdown.compressed_bytes,
        "plain_tensors": breakdown.plain_tensors,
        "plain_bytes": breakdown.plain_bytes,
        "model_bytes": sum(part.stat().st_size for part in container_parts(staged / MODEL_FILE)),
        "seconds": round(time.perf_counter() - started, 1),
        "peak_rss_gib": round(_peak_rss_gib(), 2),
    }


# ---- golden --------------------------------------------------------------------------


def golden_tensors(
    vae: AutoencoderKLWan,
    graphs: tuple[torch.nn.Module, ...] | None,
    case: GoldenCase,
    source: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """golden 1 本の中身と、書き直しの eager の差（`graphs` を渡したときだけ — 記録用）。

    テンソル（`stretch` は `source` と `rgb8` だけ）:

    - `source` `[480, 832, 3]` u8 — テスト画像の画素（来歴の sha256 の画像そのもの — TS のホスト
      テストが PNG を decode せずに前処理の鎖を回せる）。
    - `rgb8` `[H, W, 3]` u8 — fit の前処理の出力。
    - `encoder_input` `[12, 1, H/2, W/2]` f32 — [-1, 1] と patchify の後（pre のグラフの入力）。
    - `mu` `[48, 1, h, w]` f32 — 上流の非タイル encode の mu（post のグラフの出力の参照）。
    - `latent` `[48, 1, h, w]` f32 — mu の正規化 `(mu − mean)·f32(1/f32(std))`（DiT へ渡す条件）。
    """
    resized = i2v_preprocess_ref.preprocess(source, case.width, case.height, case.fit)
    tensors = {
        "source": i2v_preprocess_ref.rgb8(source),
        "rgb8": i2v_preprocess_ref.rgb8(resized),
    }
    differences: dict[str, float] = {}
    if not case.encodes:
        return tensors, differences
    sample = i2v_preprocess_ref.to_signed_unit(resized)
    image = i2v_preprocess_ref.encoder_input(sample, int(vae.config.patch_size))
    with torch.no_grad():
        mu = vae_encoder_patch.reference_mu(vae, sample)
        latent = vae_encoder_patch.normalize_condition(vae, mu)
    tensors |= {
        "encoder_input": image,
        "mu": mu[0].contiguous(),
        "latent": latent[0].contiguous(),
    }
    if graphs is not None:
        with torch.no_grad():
            eager = vae_encoder_patch.encode_chunk0(graphs, image)
        difference = float((eager - mu[0]).abs().max())
        differences = {
            "eager_max_abs": difference,
            "eager_ratio": difference / float(mu.abs().max()),
        }
    return tensors, differences


def golden_metadata(
    case: GoldenCase, image: GoldenImage, versions: dict[str, str]
) -> dict[str, str]:
    metadata = {
        "image": image.name,
        "image_sha256": image.sha256,
        "width": str(case.width),
        "height": str(case.height),
        "fit": case.fit,
        "role": case.role,
        "resample": "LANCZOS",
        "upstream_revision": SOURCES[MODEL].revision,
        **{f"version_{name}": value for name, value in versions.items()},
    }
    if case.encodes:
        metadata |= {"weights": "f16-rounded", "reference": GOLDEN_REFERENCE}
    return metadata


def emit_series(
    targets: Sequence[str],
    vae: AutoencoderKLWan,
    out_root: Path,
    source: Provenance,
    *,
    cases: Sequence[GoldenCase],
) -> dict[str, Any]:
    """グラフと golden（`cases` — 空なら書かない）を一組で作業席へ書き、全部の検査を通してから
    据える。

    MUST: 一組で据える（decoder の `emit_targets` と同じ理由 — 途中で落ちた実走が、別の丸めの世代の
    グラフと golden の混ざった組を系列に残すと、形が同じなので誰にも見分けられない）。
    """
    unknown = sorted(set(targets) - set(TARGETS))
    if unknown or len(set(targets)) != len(targets):
        raise EncoderGraphError(
            f"ターゲット {list(targets)} に未知か重複がある（既知: {', '.join(TARGETS)}）"
        )
    images = {image.name: image for image in IMAGES}
    sources = {name: load_image(images[name]) for name in sorted({case.image for case in cases})}
    versions = library_versions()
    out_root.mkdir(parents=True, exist_ok=True)
    finals = [out_root / target for target in targets]
    finals += [golden_path(out_root, case) for case in cases]
    summaries: list[dict[str, Any]] = []
    written: list[dict[str, Any]] = []
    with export_vae._staged_set(finals) as seats:
        for target, staged in zip(targets, seats[: len(targets)], strict=True):
            started = time.perf_counter()
            module = graph_module(target, vae)
            staged.mkdir()
            with torch.no_grad():
                graph = export_to_file(
                    module,
                    example_inputs(target, vae),
                    staged / MODEL_FILE,
                    provenance=source,
                    graph_name=target,
                    dynamic_shapes=dynamic_shapes(target),
                    symbol_names=symbol_names(target),
                    weight_dtype="f16",
                    preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
                )
            assert_encoder_graph(graph, target, vae)
            summaries.append(_graph_summary(target, out_root / target, staged, graph, started))
        graphs = vae_encoder_patch.encoder_graphs(vae) if cases else None
        for case, staged in zip(cases, seats[len(targets) :], strict=True):
            started = time.perf_counter()
            tensors, differences = golden_tensors(vae, graphs, case, sources[case.image])
            save_file(
                tensors,
                str(staged),
                metadata=golden_metadata(case, images[case.image], versions),
            )
            written.append(
                {
                    "case": case.name,
                    "role": case.role,
                    "fit": case.fit,
                    "shapes": {name: list(tensor.shape) for name, tensor in tensors.items()},
                    **differences,
                    "seconds": round(time.perf_counter() - started, 1),
                    "path": str(golden_path(out_root, case)),
                }
            )
    summary: dict[str, Any] = {"series": str(out_root), "graphs": summaries}
    if cases:
        summary["goldens"] = written
        summary["versions"] = versions
    return summary


# ---- eager 同値の実測 ----------------------------------------------------------------


def verify(cases: Sequence[GoldenCase]) -> dict[str, Any]:
    """書き直しの eager 同値を実重み（f32・丸め無し）で測る（テスト画像 × 寸法）。

    - 3 グラフを繋いだ mu と上流の非タイル encode の mu の差（最大絶対値・参照の最大絶対値に
      対する比）
    - 参照と書き直しの所要（CPU）
    """
    vae = export_vae.load_vae(MODEL, round_f16=False)
    export_vae.assert_series_config(vae, export_vae.VAE_SERIES[MODEL])
    graphs = vae_encoder_patch.encoder_graphs(vae)
    images = {image.name: image for image in IMAGES}
    rows: list[dict[str, Any]] = []
    for case in cases:
        if not case.encodes:
            continue
        resized = i2v_preprocess_ref.preprocess(
            load_image(images[case.image]), case.width, case.height, case.fit
        )
        sample = i2v_preprocess_ref.to_signed_unit(resized)
        image = i2v_preprocess_ref.encoder_input(sample, int(vae.config.patch_size))
        with torch.no_grad():
            started = time.perf_counter()
            reference = vae_encoder_patch.reference_mu(vae, sample)[0]
            middle = time.perf_counter()
            eager = vae_encoder_patch.encode_chunk0(graphs, image)
            finished = time.perf_counter()
        difference = float((eager - reference).abs().max())
        rows.append(
            {
                "case": case.name,
                "mu_shape": list(reference.shape),
                "reference_abs_max": float(reference.abs().max()),
                "bit_exact": torch.equal(eager, reference),
                "max_abs": difference,
                "ratio": difference / float(reference.abs().max()),
                "reference_seconds": round(middle - started, 1),
                "rewrite_seconds": round(finished - middle, 1),
            }
        )
    return {
        "threads": torch.get_num_threads(),
        "cases": rows,
        "worst_ratio": max(row["ratio"] for row in rows),
        "peak_rss_gib": round(_peak_rss_gib(), 2),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--out", type=Path, default=None, help=f"系列の根（既定 {SERIES_ROOT / SERIES_NAME}）"
    )
    parser.add_argument(
        "--target", action="append", choices=TARGETS, default=None, help="書くグラフ（既定は 3 本）"
    )
    parser.add_argument("--no-goldens", action="store_true", help="golden を書かない")
    parser.add_argument("--verify", action="store_true", help="eager 同値を測るだけ（書かない）")
    args = parser.parse_args(argv)

    if args.verify:
        print(json.dumps(verify(GOLDEN_CASES), indent=1, ensure_ascii=False))
        return 0

    source = export_vae.provenance(MODEL)
    vae = export_vae.load_vae(MODEL, round_f16=True)
    export_vae.assert_series_config(vae, export_vae.VAE_SERIES[MODEL])
    summary = emit_series(
        args.target or TARGETS,
        vae,
        args.out or SERIES_ROOT / SERIES_NAME,
        source,
        cases=() if args.no_goldens else GOLDEN_CASES,
    )
    summary["weights"] = weight_summary(vae)
    summary["peak_rss_gib"] = round(_peak_rss_gib(), 2)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
