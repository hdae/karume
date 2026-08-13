"""実重み SigLIP2 の **vision tower** を IR v1 コンテナ + golden io へ書き出す台本。

`export_embeddinggemma.py` と同じ役割（実重み・単一ベクトル出力の数値一致）を、画像側で
受け持つ。生成物は `outputs/series/` 配下で、リポジトリ直下の `.gitignore` によりコミット
対象外（重み 350MB〜1.7GB）。

    uv run --group siglip2 python export_siglip2.py
    uv run --group siglip2 python export_siglip2.py --verify  # パッチ前 eager との同値検証

transformers は **5.14.1 でピン**する（`karume.patch_siglip2` がモデリングコードのクラス
属性を差し替えるため — pyproject.toml の `siglip2` グループ）。

## モデル軸

対象は `--model-dir` の 1 軸で、系列名はそのディレクトリ名から導く（`export_sbv2.py` /
`export_irodori.py` と同じ持ち方）— 出力先を固定にすると、別のモデルを書き出した瞬間に先の
系列が黙って上書きされる。

    uv run --group siglip2 python export_siglip2.py \
        --model-dir ../../inputs/siglip2/siglip2-so400m-patch14-384

規模の差（hidden / 層数 / head 数 / intermediate / patch / 解像度）は**全て config.json 由来**
で、台本はどこにも焼かない。合成画像の寸法も `config.image_size` / `config.patch_size` から
組む（{@link build_cases}）。

## 何をグラフに載せるか

`SiglipVisionModel` の **pooler_output 1 本だけ**（MAP head を通した `[B, hidden]`）。
`last_hidden_state` は出さない — 読み戻しが `[B, パッチ数, hidden]` と 2 桁大きく、埋め込み
用途では使われない（ADR 0026 の「出力を絞る」と同じ判断）。text tower は載せない。

batch は **静的 1**。画像側は解像度も系列長（= パッチ数）も固定で、動かす軸が無い。

MUST: SDPA は**保存しない**（既定の `PRESERVED_OP_PREFIXES` 11 本のまま）。attention は
エンコーダの層数 + MAP head の 1 本とも mul×2 + bmm + softmax + bmm へ分解される
（base は 12 + 1 / so400m は 27 + 1）。ADR 0023 の融合 attention はこの系列では opt-in しない。

MUST: モデルは `karume.patch_siglip2` の 2 段パッチを当ててから export する（当てないと
`aten.conv2d.padding` / `aten.select.int` / `aten.embedding` の i64 initializer が残る）。
golden の期待値は**差し替え後**のモジュールから採る — 差し替え前後の差はパッチ層の責務で、
`--verify` が 2 点評価で実測する（1 点目 = 形の畳み込みだけ・ビット同一が主張の中身、
2 点目 = MAP head まで・maxdiff を報告）。

MUST: `--verify` は emit しない（同一プロセスでは併用できない）。パッチはクラス属性の
プロセス全域差し替えなので、emit のために当てた後では「当てる前の参照」が採れない。

## 入力の約束（前処理はグラフに載せない）

グラフ入力は **正規化済みの** `pixel_values f32 [1, 3, 解像度, 解像度]`（base 224 /
so400m 384）。画像の decode はホストの責務、resize（`config.image_size` へ bicubic）/
rescale（1/255）/ normalize（mean = std = 0.5 → `[-1, 1]`）は karume 側に置く予定だが
**この段では作らない**。定数の正本は重みと同じ場所の `preprocessor_config.json`
（2 モデルとも mean = std = 0.5・resample 2 = bicubic で、違うのは寸法だけ）。

golden の入力も実画像ではなく**合成画像**（{@link build_cases}）— 前処理がまだ無い以上、
実画像を通すと「どの resize 実装で作ったか」が golden の暗黙の前提になる。

## 出力レイアウト

系列名は `--model-dir` のディレクトリ名（既定の `inputs/siglip2/siglip2-base-patch16-224/`
なら `siglip2-base-patch16-224`）:

    outputs/series/<系列名>/model.safetensors     重み・定数 + __metadata__
    outputs/series/<系列名>/io.<case>.safetensors 入力と torch CPU 期待出力

io のテンソルキー規約は tiny golden / DeBERTa / EmbeddingGemma と同じ
（`input.<グラフ入力名>` / `output.<位置>`）。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import nn

from karume import patch_siglip2
from karume.convert import normalize_boundary_tensor
from karume.ir import IrGraph
from karume.paths import INPUTS_ROOT, SERIES_ROOT
from karume.pipeline import export_to_file

#: 実重みの親（`hf download google/<名前> --local-dir inputs/siglip2/<名前>` の展開先）。
MODELS_ROOT = INPUTS_ROOT / "siglip2"

#: 既定のモデル（`--model-dir` 未指定のとき）。
DEFAULT_MODEL_DIR = MODELS_ROOT / "siglip2-base-patch16-224"


def default_out_dir(model_dir: Path) -> Path:
    """生成物の既定の置き場（`outputs/series/<実重みのディレクトリ名>/`）。

    モデル名を系列の綴りへ焼くのは `export_sbv2.py` の `default_out_root` と同じ理由 —
    固定の綴りを共有すると、別のモデルを書き出した瞬間に先の資産が黙って上書きされる。

    格納 dtype は f32 のみなので dtype 別の枝は無い（f16 / i8 は別系列で決める話）。
    """
    return SERIES_ROOT / model_dir.name


MODEL_FILE = "model.safetensors"
IO_PREFIX = "io."
IO_SUFFIX = ".safetensors"
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."

#: グラフ入力の名前（= `VisionPooler.forward` の引数名）。export 後に突合してずれていたら
#: 止める — 位置で渡す以上、黙ってずれると golden の入力だけが入れ替わる。
INPUT_NAME = "pixel_values"

#: 合成画像の乱数（`noise` ケース）。グローバル seed に依存しない。
SEED = 20260813

#: チャネルごとの明度オフセット。3 チャネルが同一の面だと、チャネルの取り違え（patch
#: 埋め込みの重みレイアウト誤り）が値に出ない。
CHANNEL_OFFSETS: tuple[float, ...] = (-0.2, 0.0, 0.2)

#: `ramp-dim` のコントラスト比（`ramp` と**構造は同じで強度だけ違う**近い対を作る）。
DIM_SCALE = 0.5

#: sanity で見る cosine の対（近い対, 遠い対）。近い対が遠い対を上回らなければ、
#: 「数値は動いているが画像埋め込みとして壊れている」— 生成時に fail loudly にする。
NEAR_PAIR = ("ramp", "ramp-dim")
FAR_PAIR = ("ramp", "checker")


class VisionPooler(nn.Module):
    """pooler_output（MAP head 経由の `[B, hidden]`）1 本だけを返す export 用ラッパ。"""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).pooler_output


def _stack_channels(plane: torch.Tensor, channels: int) -> torch.Tensor:
    """`[S, S]` の 1 面をチャネルごとにずらして `[1, C, S, S]` にする。"""
    offsets = torch.tensor(CHANNEL_OFFSETS[:channels]).reshape(channels, 1, 1)
    return (plane.unsqueeze(0) + offsets).clamp(-1.0, 1.0).unsqueeze(0)


def _ramp_plane(size: int) -> torch.Tensor:
    """左上から右下への対角ランプ（`[S, S]`・値域 `[-1, 1]`）。"""
    axis = torch.linspace(-1.0, 1.0, size)
    return (axis.reshape(1, size) + axis.reshape(size, 1)) / 2.0


def _checker_plane(size: int, cell: int) -> torch.Tensor:
    """パッチ格子に揃えた市松（`[S, S]`・値は ±1）。"""
    index = torch.arange(size) // cell
    parity = (index.reshape(1, size) + index.reshape(size, 1)) % 2
    return parity.to(torch.float32) * 2.0 - 1.0


def build_cases(config: Any) -> tuple[tuple[str, torch.Tensor], ...]:
    """golden 4 ケースの `(名前, pixel_values)`（**正規化済み**の合成画像）。

    実画像を使わないのはモジュール docstring の理由（前処理がまだ karume 側に無い）。
    構造だけを変えた対を持たせて、{@link _sanity} の cosine 順序が恒真にならないようにする:
    `ramp` と `ramp-dim` は同じ構造で強度だけ違い、`checker` は構造ごと違う。
    """
    size = int(config.image_size)
    channels = int(config.num_channels)
    ramp = _ramp_plane(size)
    generator = torch.Generator().manual_seed(SEED)
    noise = torch.rand((size, size), generator=generator) * 2.0 - 1.0
    return (
        ("ramp", _stack_channels(ramp, channels)),
        ("ramp-dim", _stack_channels(ramp * DIM_SCALE, channels)),
        ("checker", _stack_channels(_checker_plane(size, int(config.patch_size)), channels)),
        ("noise", _stack_channels(noise, channels)),
    )


def load_model(model_dir: Path) -> nn.Module:
    """vision tower の実重みを読み、差し替え版が前提にする config を検査する（パッチ前）。

    MUST: `attn_implementation="sdpa"` — エンコーダ 12 層の attention を SDPA として出させる
    （既定の分解表がそれを mul×2 + bmm + softmax + bmm に開く）。
    """
    from transformers import SiglipVisionModel

    model = SiglipVisionModel.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="sdpa"
    )
    model.eval()
    patch_siglip2.assert_supported(model.config)
    return model


def load_wrapper(model_dir: Path) -> VisionPooler:
    """パッチを 2 段とも当てた export 可能なラッパを返す。

    MUST: 差し替えは golden を採る**前**（`patch_siglip2` の docstring）— 後に当てると
    期待値だけが元の経路で計算され、グラフと食い違ったまま緑になる。
    """
    model = load_model(model_dir)
    patch_siglip2.apply_shape_patches(model.embeddings)
    patch_siglip2.apply_map_head_patch(model.head)
    return VisionPooler(model).eval()


def _write_io(
    wrapper: nn.Module,
    graph: IrGraph,
    cases: Sequence[tuple[str, torch.Tensor]],
    out_dir: Path,
) -> tuple[list[str], dict[str, torch.Tensor]]:
    """各ケースの入力と torch CPU 期待出力を `io.<case>.safetensors` へ書く。

    戻り値の 2 本目は sanity 記録用の期待出力（`[B, H]` の形のまま渡す）。
    """
    if len(graph.outputs) != 1:
        raise AssertionError(f"IR 出力が {len(graph.outputs)} 本（pooler_output は 1 本）")
    written: list[str] = []
    pooled: dict[str, torch.Tensor] = {}
    for name, pixel_values in cases:
        with torch.no_grad():
            output = wrapper(pixel_values)
        # MUST: io も IR の意味論 dtype の実表現へ落とす（ADR 0009 の境界正規化）。ランタイムが
        # 受け取る形と揃っていないと Deno 側 E2E が golden を読めない。
        tensors = {
            f"{INPUT_PREFIX}{INPUT_NAME}": normalize_boundary_tensor(
                pixel_values, f"{name} の入力 '{INPUT_NAME}'"
            ),
            f"{OUTPUT_PREFIX}0": normalize_boundary_tensor(
                output.detach().contiguous(), f"{name} の出力 0"
            ),
        }
        path = out_dir / f"{IO_PREFIX}{name}{IO_SUFFIX}"
        save_file(tensors, str(path))
        written.append(path.name)
        pooled[name] = output.detach()
    return written, pooled


def _sanity(pooled: dict[str, torch.Tensor]) -> dict[str, Any]:
    """構造の近い対の cosine が、構造ごと違う対を上回ることを見る。

    MUST: 恒真な sanity にしない。pooler_output は L2 正規化されていないのでノルム検査は
    使えず、「同一入力どうしの cosine = 1」も何も検証しない。**別々の入力**どうしの順序で
    見れば、埋め込みが 1 点へ潰れた場合（近い対も遠い対も cosine 1）も、パッチ埋め込みや
    MAP head が死んだ場合も落ちる。
    """
    vectors = {name: output.reshape(-1) for name, output in pooled.items()}

    def cosine(pair: tuple[str, str]) -> float:
        first, second = vectors[pair[0]], vectors[pair[1]]
        return float(torch.dot(first, second) / (first.norm() * second.norm()))

    near, far = cosine(NEAR_PAIR), cosine(FAR_PAIR)
    if near <= far:
        raise AssertionError(
            f"cosine の順序が構造と逆: {NEAR_PAIR}={near:.4f} <= {FAR_PAIR}={far:.4f}"
        )
    return {
        "l2_norms": {name: round(float(vector.norm()), 5) for name, vector in vectors.items()},
        "cosine": {
            f"{NEAR_PAIR[0]}×{NEAR_PAIR[1]}": round(near, 4),
            f"{FAR_PAIR[0]}×{FAR_PAIR[1]}": round(far, 4),
            "ramp×noise": round(cosine(("ramp", "noise")), 4),
        },
    }


def export_series(model_dir: Path, out_dir: Path) -> dict[str, Any]:
    """IR コンテナと golden io を書き、要約を返す。"""
    wrapper = load_wrapper(model_dir)
    cases = build_cases(wrapper.model.config)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, example = cases[0]
    # 動的軸は無い（解像度もパッチ数も固定 — モジュール docstring）。
    graph = export_to_file(wrapper, (example,), out_dir / MODEL_FILE, symbol_names=())
    declared = tuple(item.name for item in graph.inputs)
    if declared != (INPUT_NAME,):
        raise AssertionError(f"グラフ入力の並びが {declared} で、期待の {(INPUT_NAME,)} と違う")
    written, pooled = _write_io(wrapper, graph, cases, out_dir)
    return {
        "dir": str(out_dir),
        "nodes": len(graph.nodes),
        "outputs": len(graph.outputs),
        "initializers": len(graph.initializers),
        "model_bytes": (out_dir / MODEL_FILE).stat().st_size,
        "ops": sorted(graph.required_ops),
        "symbols": list(graph.symbols),
        "io": written,
        "input_shape": list(example.shape),
        "sanity": _sanity(pooled),
    }


def _diff_entry(
    stage: str,
    claim: str,
    got: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """1 段分の同値レポート（ケース名 → maxdiff と、全ケースのビット一致）。

    `bit_exact` は「差 0」より強い主張（`0.0 == -0.0` は差 0 だがビットは違う）で、形の
    畳み込みのように**ビット一致が主張の中身**である段で意味を持つ。
    """
    return {
        "stage": stage,
        "claim": claim,
        "maxdiff": {
            name: float((got[name] - expected[name]).abs().max()) for name in sorted(expected)
        },
        "bit_exact": all(torch.equal(got[name], expected[name]) for name in expected),
    }


def verify_patches(model_dir: Path) -> list[dict[str, Any]]:
    """パッチ前 eager との同値を **2 点**で実測する（`export_sbv2.py --verify` と同じ形）。

    1. 形の畳み込みだけ（conv padding + 位置埋め込み）→ **bit_exact が主張の中身**。演算列が
       1 対 1 で対応するので差は 0 でなければならず、外れたらここで落とす。
    2. MAP head の q/k/v 明示化まで → **maxdiff を報告**。`nn.MultiheadAttention` は q にだけ
       1/√d を掛けるのに対し SDPA は q と k に対称に割るのでビット一致にはならない
       （実測 7.75e-07〜2.38e-06 / ベクトルの L2 ノルムは 12.7〜13.1）。

    MUST: 順序は「**全ケースの参照値を確定** → 1 → 2」。パッチはクラス属性のプロセス全域
    差し替えなので、段ごとに参照を採り直すと 2 段目の参照がパッチ後の値になる（恒真化）。
    """
    if patch_siglip2.patches_applied():
        raise SystemExit(
            "パッチ適用済みのプロセスでは参照を採れない（同値検証が差 0 で恒真化する）"
        )
    model = load_model(model_dir)
    cases = build_cases(model.config)
    with torch.no_grad():
        reference = {
            name: model(pixel_values=pixel_values).pooler_output for name, pixel_values in cases
        }

    def pooled() -> dict[str, torch.Tensor]:
        with torch.no_grad():
            return {
                name: model(pixel_values=pixel_values).pooler_output for name, pixel_values in cases
            }

    patch_siglip2.apply_shape_patches(model.embeddings)
    folds = _diff_entry("shape-folds", "bit-exact", pooled(), reference)
    if not folds["bit_exact"]:
        raise AssertionError(
            f"形の畳み込みがビット同一でない: maxdiff={folds['maxdiff']} —"
            " conv padding / 位置埋め込みの書き換えが同値でない"
        )
    patch_siglip2.apply_map_head_patch(model.head)
    return [folds, _diff_entry("map-head", "max-abs-diff", pooled(), reference)]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="出力先（既定は outputs/series/<--model-dir のディレクトリ名>/）",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="パッチ前 eager との同値を 2 点で実測する（emit はしない — 同一プロセスでは"
        "クラス属性の差し替えが参照を汚染するので併用できない）",
    )
    args = parser.parse_args(argv)
    if args.verify:
        for entry in verify_patches(args.model_dir):
            print(
                f"{entry['stage']} ({entry['claim']}): bit_exact={entry['bit_exact']}"
                f" maxdiff={entry['maxdiff']}"
            )
        return
    out_dir = args.out if args.out is not None else default_out_dir(args.model_dir)
    summary = export_series(args.model_dir, out_dir)
    print(json.dumps({"model_dir": str(args.model_dir), **summary}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
