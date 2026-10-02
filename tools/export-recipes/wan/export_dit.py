"""Wan2.1 の DiT を S 形の IR（`krm`）+ golden へ書き出す台本（ADR 0118 決定 3 / 8・段 2）。

起点は pin した revision の diffusers 版（`wan.sources.local_snapshot`）。グラフは
{@link wan.dit_patch.WanDitTokens}（入口は patchify の後・出口は unpatchify の前・次元はトークン長
`S` の 1 シンボル）。格納は f16 席（重みを f16 表現可能値へ丸めてから参照と golden を採り、
適格な重みスロットだけを f16 で格納する — ADR 0006 / 0018）。容器には重みに加えて**資産
`rope_base`**（ホストが RoPE の表を組むための軸別素表 — 役割 `rope-base`・ADR 0109 決定 4）が入る。

    uv run --group wan --inexact python -m wan.export_dit            # 製品のグラフ + golden
    uv run --group wan --inexact python -m wan.export_dit --layers   # 計測用（層別の出口）
    uv run --group wan --inexact python -m wan.export_dit --verify   # パッチ前後の eager 同値だけ

出力（既定 `outputs/series/wan2.1-t2v-1.3b-f16-dyn/transformer/`）:

    model-NNNNN-of-NNNNN.krm      重み・定数 + 2 文書の記述 + 資産 `rope_base`
    io.<case>.safetensors         グラフの入力（`input.*`）とパッチ後の torch CPU の出力
                                  （`output.0`）
    reference.<case>.safetensors  上流の素の diffusers（CPU f32・同じ f16 丸めの重み）の
                                  `latents` / `timestep`（i32）/ `output`（`[1,16,F,H,W]`）/
                                  `block.NN`（各ブロックの出力 `[1,S,1536]`）

`--layers` は層別の出口を足した計測用のグラフ（{@link wan.dit_patch.WanDitTokensLayers}）を
`…-f16-dyn-probe/transformer/` へ書く。golden は書かない（入力は製品の系列の `io.*`、期待値は同じ
`reference.*` の `block.NN` を使う — 入力と重みが同じなので）。

## ケース（決定 8 と追記 2026-10-02 の「帯の決定と受入れを分ける」）

| 役割 | 潜在 `[1,16,F,H,W]` | S | timestep | 有効長 | seed |
| --- | --- | --: | --: | --: | --: |
| `band`（帯の決定） | `[1,16,3,16,16]` | 192 | 999 | 24 | `SEED` + 0 |
| `band` | `[1,16,2,16,24]` | 192 | 750 | 50 | `SEED` + 3 |
| `band` | `[1,16,3,16,16]` | 192 | 500 | 12 | `SEED` + 4 |
| `band` | `[1,16,1,16,48]` | 192 | 250 | 80 | `SEED` + 5 |
| `band` | `[1,16,2,16,24]` | 192 | 600 | 30 | `SEED` + 8 |
| `band` | `[1,16,3,16,16]` | 192 | 113 | 60 | `SEED` + 6 |
| `accept`（受入れ） | `[1,16,2,16,24]` | 192 | 600 | 45 | 777001 |
| `accept` | `[1,16,4,8,16]` | 128 | 30 | 9 | 777002 |
| `accept` | `[1,16,2,24,16]` | 192 | 400 | 70 | 777003 |
| `growth`（S に対する伸び） | `[1,16,3,32,32]` | 768 | 999 | 24 | `SEED` + 2 |
| `growth` | `[1,16,3,32,32]` | 768 | 500 | 37 | `SEED` + 7 |

帯の指標は「最大絶対差 ÷ 参照の最大絶対値」（決定 8 の目安の形）で、帯 = `band` の 6 ケースの
最悪の比 × 5（TS 側 `e2e_wan_dit_test.ts` の `DIT_RATIO_BAND`）。`accept` の 3 ケースは `band` と
seed が違い、timestep も 1 本（600）を除いて違う（600 の 1 本は、旧帯 `atol` 3.2e-4 を超えた検証の
未見ケース — seed 777001・有効長 45 — をそのまま再現したもの）。

MUST: `accept` の結果を見て `band` のケースを足し引きしない（帯の決定と受入れの独立が崩れる）。
受入れが帯を外れたら、帯を広げずに原因を調べる。

決定用を 1 ケースにしないのは、誤差が入力で 1 桁近く動くため（実測: t = 999 の 1 ケースで決めた帯を
t = 500 の未見ケースが 1.6 倍超えた。中ほどの timestep は CPU の eager でも入力の 1e-5 級の揺れを
約 4 倍に増幅する — 移植の誤りではなく入力の感度）。決定用は timestep を参照の設定の列
（999 → 60）の全域に散らし、seed と格子も変える。最大絶対差でなく比を見るのは、12,288 要素の裾の
1 要素で決まる絶対値が参照の振れ幅ごと動くため。

格子は正方（F'·H'·W' = 3·8·8）と非正方（2·8·12 / 1·8·24 / 2·12·8 / 4·4·8）を混ぜる。h と w を
取り違えたホスト実装は正方の格子では値が一致してしまうので、非正方で初めて割れる。テキスト文脈は
固定 seed の乱数 `[有効長, 4096]` を上流と同じ形（後ろをゼロで 512 行まで埋める —
`wan.pipeline_ref.pad_text_embeds`）にしたもの。

MUST: 参照は**上流の素の forward**（`dit_patch.reference_dit_layers`）で採る。attn1 の processor の
差し替えは素の経路を変えない（`dit_patch` のモジュール docstring）ので、参照を採る順序の門は要らな
い。
MUST: f16 の丸めは参照より**前**（ADR 0006）。丸めは S 形のラッパに掛け、共有している上流の部品にも
同じ値が届く（patch 埋め込みは conv の重みの view — `dit_patch.patch_embedding_linear`）。RoPE の表
（`model.rope` のバッファ）は丸めない — 製品は f32 の素表を焼き、参照もその値で回す。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save, save_file
from torch import nn
from torch.export import Dim

from _shared.paths import SERIES_ROOT
from karume.artifacts import staged_publication
from karume.container import AssetInput, Provenance, container_parts
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION, normalize_boundary_tensor
from karume.dist import NOTICE_FILENAME
from karume.emit import storage_breakdown
from karume.pipeline import export_to_file
from karume.quantize import round_weights_to_f16
from wan import dit_patch
from wan.pipeline_ref import TEXT_DIM, assert_no_mps, pad_text_embeds
from wan.sources import DEFAULT_MODEL, SOURCES, local_snapshot

#: 系列（ADR 0118 決定 7 — 接尾辞 `-dyn` は ADR 0077 の慣例）と、計測用の層別出口の系列。
SERIES = SERIES_ROOT / "wan2.1-t2v-1.3b-f16-dyn"
PROBE_SERIES = SERIES_ROOT / "wan2.1-t2v-1.3b-f16-dyn-probe"

#: 部品名（= 容器のグラフ名 = 配布 manifest の weights のキー — container-v1 §2.1）。
TARGET = "transformer"
MODEL_FILE = "model.krm"

#: 資産の名前と役割（Anima の S 形 DiT と同じ綴り — ADR 0109 決定 4）。
ROPE_BASE_ASSET = "rope_base"
ROPE_BASE_ROLE = "rope-base"

IO_PREFIX = "io."
REFERENCE_PREFIX = "reference."
CASE_SUFFIX = ".safetensors"
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."

#: グラフ入力の名前（ラッパの forward の引数名がそのまま IR の入力名になる）。
INPUT_NAMES = ("tokens", "timesteps_proj", "encoder_hidden_states", "rope_cos", "rope_sin")

#: `Dim("S")` の上限 = 81 フレーム（潜在 `[16,21,60,104]` → 21·30·52）。グラフ側にトークン長の制約は
#: 無い（RoPE の表を入力へ出したので S 依存の焼き込みが無い）ので、ここは受理集合の最大を宣言する。
DIT_SYM_MAX = 32_760
#: 同下限（0 / 1 特殊化を避ける線 — Anima の `DIT_SYM_MIN` と同じ）。
DIT_SYM_MIN = 2

#: 決定用と伸びの記録のケースの乱数はここから派生させる（グローバル seed に依存しない — 再生成で
#: バイト一致させる）。受入れのケースは別系統の seed（777001〜）を直に持つ。
SEED = 20261002


@dataclass(frozen=True)
class CaseSpec:
    """golden 1 ケースの条件（潜在の形・timestep・テキストの有効長・乱数の seed）。"""

    role: str
    latent_shape: tuple[int, int, int]
    timestep: int
    text_length: int
    seed: int

    def name(self, patch_size: tuple[int, int, int]) -> str:
        frames, height, width = self.latent_shape
        tokens = (frames // patch_size[0]) * (height // patch_size[1]) * (width // patch_size[2])
        return f"{self.role}-s{tokens:05d}-t{self.timestep:04d}"


#: ケースの表（モジュール docstring の表と同じ）。先頭が export の例示入力（小さい方 —
#: `torch.export` はトレースで 1 回 forward を回す）。
#: MUST: `band`（帯の決定）と `accept`（受入れ）は固定する — `accept` の結果を見て `band` を
#: 変えない（モジュール docstring）。
CASES: tuple[CaseSpec, ...] = (
    CaseSpec("band", (3, 16, 16), 999, 24, SEED + 0),
    CaseSpec("band", (2, 16, 24), 750, 50, SEED + 3),
    CaseSpec("band", (3, 16, 16), 500, 12, SEED + 4),
    CaseSpec("band", (1, 16, 48), 250, 80, SEED + 5),
    CaseSpec("band", (2, 16, 24), 600, 30, SEED + 8),
    CaseSpec("band", (3, 16, 16), 113, 60, SEED + 6),
    CaseSpec("accept", (2, 16, 24), 600, 45, 777001),
    CaseSpec("accept", (4, 8, 16), 30, 9, 777002),
    CaseSpec("accept", (2, 24, 16), 400, 70, 777003),
    CaseSpec("growth", (3, 32, 32), 999, 24, SEED + 2),
    CaseSpec("growth", (3, 32, 32), 500, 37, SEED + 7),
)


@dataclass(frozen=True)
class Case:
    """組み上がった 1 ケース（グラフ入力・上流の参照・パッチ後の eager 出力）。"""

    name: str
    spec: CaseSpec
    inputs: tuple[torch.Tensor, ...]
    latents: torch.Tensor
    timestep: torch.Tensor
    reference: torch.Tensor
    reference_blocks: list[torch.Tensor]


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def load_transformer(model_name: str = DEFAULT_MODEL) -> nn.Module:
    """pin した revision の DiT を CPU f32 で読む（MPS の見える機では止める — 決定 3）。"""
    assert_no_mps()
    from diffusers import WanTransformer3DModel

    model = WanTransformer3DModel.from_pretrained(
        local_snapshot(model_name), subfolder="transformer", torch_dtype=torch.float32
    )
    return model.eval()


def round_to_f16(model: nn.Module, wrapper: nn.Module) -> str:
    """S 形のラッパ経由で重みを f16 表現可能値へ丸め、上流の全パラメータに届いたことを確かめる。

    ラッパが持つのは上流の部品への参照と、conv の重みの view の Linear だけ（`rope` は持たない）。
    だから丸めは上流のパラメータ全部に届き、`rope` のバッファ（RoPE の表）には届かない。
    MUST: 両方を確かめる — 片方でも外れると「参照だけ別の重み」か「参照の表だけ f16」になり、差に
    量子化誤差が混ざって帯の意味が消える。
    """
    rope_before = {name: buffer.clone() for name, buffer in model.rope.named_buffers()}
    report = round_weights_to_f16(wrapper)
    with torch.no_grad():
        missed = [
            name
            for name, parameter in model.named_parameters()
            if not torch.equal(parameter, parameter.to(torch.float16).to(torch.float32))
        ]
    if missed:
        raise AssertionError(f"f16 の丸めが上流のパラメータに届いていない: {missed[:5]}")
    moved = [
        name
        for name, buffer in model.rope.named_buffers()
        if not torch.equal(buffer, rope_before[name])
    ]
    if moved:
        raise AssertionError(f"RoPE の表（rope のバッファ）が丸められた: {moved}")
    return report.describe()


def build_case(model: nn.Module, spec: CaseSpec) -> Case:
    """1 ケースの入力を作り、上流の素の forward で参照（最終出力と各ブロックの出力）を採る。"""
    patch_size = tuple(int(size) for size in model.config.patch_size)
    generator = _generator(spec.seed)
    latents = torch.randn(1, model.config.in_channels, *spec.latent_shape, generator=generator)
    text = torch.randn(spec.text_length, TEXT_DIM, generator=generator)
    encoder_hidden_states = pad_text_embeds(text)
    # 上流のパイプラインは scheduler の int64 の timestep を `expand(batch)` して渡す。
    timestep = torch.tensor([spec.timestep], dtype=torch.int64)
    with torch.no_grad():
        reference, blocks = dit_patch.reference_dit_layers(
            model, latents, timestep, encoder_hidden_states
        )
        rope_cos, rope_sin = dit_patch.dit_rope_tables(model.rope, spec.latent_shape)
        inputs = (
            dit_patch.dit_patchify(latents, patch_size),
            dit_patch.dit_timesteps_proj(model, timestep),
            encoder_hidden_states,
            rope_cos,
            rope_sin,
        )
    return Case(
        name=spec.name(patch_size),
        spec=spec,
        inputs=inputs,
        latents=latents,
        timestep=timestep,
        reference=reference,
        reference_blocks=blocks,
    )


def dynamic_shapes() -> tuple[Any, ...]:
    """入力ごとの記号次元。RoPE の表も同じ `S` で宣言する（別シンボルにすると「表と本体の長さが
    ずれた」形が受理されて沈黙誤値になる — Anima と同じ）。"""
    tokens = Dim("S", min=DIT_SYM_MIN, max=DIT_SYM_MAX)
    return ({1: tokens}, None, None, {1: tokens}, {1: tokens})


def provenance(model_name: str) -> Provenance:
    """容器へ焼く出所（上流のライセンス識別子と pin した revision — `wan.sources` が正本）。"""
    source = SOURCES[model_name]
    return Provenance(
        license=source.license, notice=NOTICE_FILENAME, upstream_revision=source.revision
    )


def rope_base_asset(model: nn.Module) -> dict[str, AssetInput]:
    """資産 `rope_base`（軸別素表の safetensors のバイト列そのもの — Anima と同じ席）。"""
    payload = save(dit_patch.dit_rope_base_tables(model.rope))
    return {ROPE_BASE_ASSET: AssetInput(ROPE_BASE_ROLE, len(payload), payload)}


def eager_report(wrapper: dit_patch.WanDitTokens, model: nn.Module, case: Case) -> dict[str, Any]:
    """パッチ後の eager（ホストの unpatchify まで）と上流の素の出力の差（ケース 1 本）。

    `trunk` は patch 埋め込みを上流の conv3d の出力に差し替えた経路（RoPE の書き換えを含む本体だけの
    比較 — ビット一致が主張）、`full` は Linear 化した patch 埋め込みを含む製品の経路
    （差を記録する）。
    """
    patch_size = wrapper.patch_size
    with torch.no_grad():
        tokens_out = wrapper(*case.inputs)
        full = dit_patch.dit_unpatchify(tokens_out, case.spec.latent_shape, patch_size)
        hidden = model.patch_embedding(case.latents).flatten(2).transpose(1, 2)
        trunk = dit_patch.dit_unpatchify(
            wrapper.forward_hidden(hidden, *case.inputs[1:]), case.spec.latent_shape, patch_size
        )
        embedded = wrapper.patch_embedding(case.inputs[0])
    return {
        "case": case.name,
        "reference_max_abs": float(case.reference.abs().max()),
        "trunk_bit_exact": torch.equal(trunk, case.reference),
        "trunk_max_abs_diff": float((trunk - case.reference).abs().max()),
        "patch_embedding_max_abs_diff": float((embedded - hidden).abs().max()),
        "patch_embedding_max_abs": float(hidden.abs().max()),
        "full_bit_exact": torch.equal(full, case.reference),
        "full_max_abs_diff": float((full - case.reference).abs().max()),
    }


def _write_case_files(
    wrapper: nn.Module, graph_inputs: list[str], case: Case, out_dir: Path
) -> list[str]:
    """`io.<case>`（グラフ入力 + パッチ後の eager 出力）と `reference.<case>`（上流の値）を書く。"""
    if graph_inputs != list(INPUT_NAMES):
        raise AssertionError(f"グラフ入力名が宣言と不一致: {graph_inputs} vs {list(INPUT_NAMES)}")
    with torch.no_grad():
        output = wrapper(*case.inputs)
    io = {
        f"{INPUT_PREFIX}{key}": normalize_boundary_tensor(value, f"{case.name} の入力 '{key}'")
        for key, value in zip(INPUT_NAMES, case.inputs, strict=True)
    }
    io[f"{OUTPUT_PREFIX}0"] = normalize_boundary_tensor(
        output.detach().contiguous(), f"{case.name} の出力"
    )
    reference = {
        "latents": case.latents.contiguous(),
        "timestep": normalize_boundary_tensor(case.timestep, f"{case.name} の timestep"),
        "output": case.reference.contiguous(),
        **{
            f"block.{index:02d}": block.contiguous()
            for index, block in enumerate(case.reference_blocks)
        },
    }
    io_name = f"{IO_PREFIX}{case.name}{CASE_SUFFIX}"
    reference_name = f"{REFERENCE_PREFIX}{case.name}{CASE_SUFFIX}"
    save_file(io, str(out_dir / io_name))
    save_file(reference, str(out_dir / reference_name))
    return [io_name, reference_name]


def emit(args: argparse.Namespace) -> dict[str, Any]:
    """製品のグラフ（または `--layers` の計測用グラフ）を書き、要約を返す。"""
    started = time.perf_counter()
    model = load_transformer(args.model)
    wrapper_class = dit_patch.WanDitTokensLayers if args.layers else dit_patch.WanDitTokens
    wrapper = wrapper_class(model)
    rounded = round_to_f16(model, wrapper)
    print(f"[fake-quant] {TARGET}: f16 表現可能値へ丸めた — {rounded}", flush=True)
    # 計測用のグラフは golden を書かないので、例示入力（先頭のケース）だけを組む。
    cases = [build_case(model, spec) for spec in (CASES[:1] if args.layers else CASES)]
    eager = [eager_report(wrapper, model, case) for case in cases] if not args.layers else []
    for line in eager:
        print(f"[eager] {json.dumps(line, ensure_ascii=False)}", flush=True)

    out_dir = (PROBE_SERIES if args.layers else SERIES) / TARGET
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with staged_publication(out_dir) as staged:
        staged.mkdir()
        graph = export_to_file(
            wrapper,
            cases[0].inputs,
            staged / MODEL_FILE,
            provenance=provenance(args.model),
            graph_name=TARGET,
            assets=rope_base_asset(model),
            dynamic_shapes=dynamic_shapes(),
            symbol_names=("S",),
            weight_dtype="f16",
            preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
        )
        written: list[str] = []
        if not args.layers:
            declared = [entry.name for entry in graph.inputs]
            for case in cases:
                written += _write_case_files(wrapper, declared, case, staged)
    breakdown = storage_breakdown(graph)
    return {
        "target": TARGET,
        "dir": str(out_dir),
        "layers": args.layers,
        "nodes": len(graph.nodes),
        "outputs": len(graph.outputs),
        "initializers": len(graph.initializers),
        "compressed_tensors": breakdown.compressed_tensors,
        "compressed_bytes": breakdown.compressed_bytes,
        "plain_tensors": breakdown.plain_tensors,
        "plain_bytes": breakdown.plain_bytes,
        "model_bytes": sum(part.stat().st_size for part in container_parts(out_dir / MODEL_FILE)),
        "ops": sorted(graph.required_ops),
        "symbols": list(graph.symbols),
        "inputs": [[entry.name, list(entry.shape)] for entry in graph.inputs],
        "output_shapes": [list(graph.values[name].shape) for name in graph.outputs],
        "io": written,
        "eager": eager,
        "seconds": round(time.perf_counter() - started, 1),
    }


def verify(args: argparse.Namespace) -> list[dict[str, Any]]:
    """パッチ前後の eager 同値だけを実重みで実測する（容器は書かない）。"""
    model = load_transformer(args.model)
    wrapper = dit_patch.WanDitTokens(model)
    if not args.no_f16:
        round_to_f16(model, wrapper)
    return [eager_report(wrapper, model, build_case(model, spec)) for spec in CASES]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(SOURCES))
    parser.add_argument(
        "--layers",
        action="store_true",
        help="層別の出口を足した計測用のグラフを …-probe/ へ書く（golden は書かない）",
    )
    parser.add_argument(
        "--verify", action="store_true", help="パッチ前後の eager 同値だけを測る（容器は書かない）"
    )
    parser.add_argument(
        "--no-f16",
        action="store_true",
        help="--verify で f16 の丸めを掛けない（素の f32 の重みで測る）",
    )
    args = parser.parse_args(argv)
    if args.no_f16 and not args.verify:
        parser.error("--no-f16 は --verify とだけ使う（書き出す系列は f16 席だけ）")
    if args.verify and args.layers:
        parser.error("--verify と --layers は併用しない")
    if args.verify:
        print(json.dumps(verify(args), indent=1, ensure_ascii=False))
        return 0
    print(json.dumps(emit(args), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
