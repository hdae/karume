"""実重み DeBERTa-v2（SBV2 text front）を IR v1 コンテナ + golden io へ書き出す台本。

`karume.goldens` の tiny golden が「op 契約の被覆」を受け持つのに対し、こちらは
**実重み・実トークン列での数値一致**を受け持つ（M1-P2 波 5）。生成物は `outputs/series/`
配下で、リポジトリ直下の `.gitignore` によりコミット対象外（重み 1.3GB 級）。

    uv run --with 'transformers==5.14.1' python export_deberta.py
    uv run --with 'transformers==5.14.1' python export_deberta.py --layers 2
    uv run --with 'transformers==5.14.1' python export_deberta.py --dtype i8 --act-quant

transformers は **5.14.1 でピン**する（recon §6-5 — モデリングコードが変わるとグラフ形が
変わる）。pyproject.toml / uv.lock には入れず `--with` で一時的に足す。

出力レイアウト（Deno 側 `packages/runtime/tests/e2e_deberta_test.ts` が列挙する）:

    outputs/series/deberta/<variant>/model.safetensors     重み・定数 + __metadata__.karume_ir
    outputs/series/deberta/<variant>/io.<case>.safetensors 入力と torch CPU での期待出力

io のテンソルキー規約は tiny golden と同じ（`input.<グラフ入力名>` / `output.<位置>`）。
1 モデルに対して io が複数ある点だけが違う。

## 格納 dtype と w8a8 の鏡像（ADR 0019 / 0025）

`--dtype i8` は**別系列**（`outputs/series/deberta-i8/`）へ書く — f32 系列と同居させると既存 E2E の
網（f32 の tolerance）が黙って別の資産に掛かる。`--dtype f16` は**足さない**（SBV2 系列の
f16 化と一体で決める話 — タスク #30 の領分）。

`--act-quant` は w8a8（`SessionOptions.linearCompute: "i8a8"`）の torch 鏡像で、適格
`nn.Linear` の入力を per-token i8 へ落とした期待値を **`io-i8a8.<case>.safetensors`** という
別 prefix で追加で書く。**通常の golden io はフックなしで採る** MUST — フックを掛けたまま
`io.<case>` を書くと w8（f32 計算）E2E の期待値が活性量子化ごと汚染され、緑のまま検出力
だけが消える。prefix を分けるのは Deno 側の列挙（`io.` の startsWith）が鏡像を通常ケースと
取り違えないため。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import nn
from torch.export import Dim

from karume.act_quant import attach_act_quant, detach_act_quant
from karume.convert import normalize_boundary_tensor
from karume.ir import IrGraph
from karume.paths import SERIES_ROOT
from karume.pipeline import export_to_file
from karume.quantize import fake_quant_int8

#: SBV2 text front が使う BERT そのもの（recon §1）。
MODEL_ID = "ku-nlp/deberta-v2-large-japanese-char-wwm"

#: 対応する格納 dtype。**f16 は無い** — SBV2 系列と一体で決める（タスク #30）。
WEIGHT_DTYPES: tuple[str, ...] = ("f32", "i8")

#: 生成物の既定の置き場（格納 dtype 別の**系列**）。親は `SERIES_ROOT`（= outputs/series/）—
#: models/ は配布形だけの場所（karume.paths）。`.gitignore` の `outputs/` でコミット対象外。
#: MUST: dtype ごとに別ディレクトリ（ADR 0019）— 同居させると f32 系列の網が i8 資産に掛かる。
DEFAULT_OUT_ROOTS: Mapping[str, Path] = {
    "f32": SERIES_ROOT / "deberta",
    "i8": SERIES_ROOT / "deberta-i8",
}

MODEL_FILE = "model.safetensors"
IO_PREFIX = "io."
#: w8a8 鏡像 io の prefix。**`io.` で始まらない**こと MUST — Deno 側の通常ケース列挙は
#: `startsWith("io.")` なので、`io.` 始まりにすると鏡像が w8 の golden として拾われる。
ACT_IO_PREFIX = "io-i8a8."
IO_SUFFIX = ".safetensors"
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."

#: 記号次元 T の上限。config の `max_position_embeddings` と一致させる（Tmax 畳み込みの
#: 評価点はここから torch の range_constraints 経由で決まる — ADR 0010）。
SYM_MAX = 512

@dataclass(frozen=True)
class Variant:
    """1 つの出力ディレクトリの綴りと出力形。

    出力形を層数と一緒にここへ書くのは、**配布形（1 本出し）と検証用（全層出し）を
    取り違えると沈黙する**から — どちらも同じ shape のテンソルを出すので、間違えても
    ロードも実行も通ってしまう。
    """

    name: str
    #: SBV2 が使う 1 本だけを出力する（配布形）。False は全層の hidden_states を並べる。
    single_output: bool = False


#: 層数 → variant。2 層は開発イテレーション用、**22 層が SBV2 の配布形**、24 層は全層の
#: golden 検証用（層別の誤差の伸びを読む — ADR 0026 決定 2 の tolerance 導出）。
#:
#: 22 なのは SBV2 が使うのが `hidden_states[-3]`（= 先頭から 22 番目 = layer 21 の出力）だから
#: で、末尾 2 層は配布形で完全に死んでいる。切り詰めた 22 層モデルの最終出力が 24 層モデルの
#: `hidden_states[-3]` と**ビット一致する**ことは実測済み（f32 / i8 / i8+a8 の 3 構成）—
#: ADR 0044 / docs/research/2026-08-11-deberta-size-recon.md §4。
VARIANTS: dict[int, Variant] = {
    2: Variant("dev-2layer"),
    22: Variant("sbv2-22layer", single_output=True),
    24: Variant("full-24layer"),
}

#: golden の固定文（トークナイザは文字単位の BertJapaneseTokenizer）。
#: 短文 / 長め / 記号混じり の 3 本で T を散らす。
GOLDEN_SENTENCES: tuple[str, ...] = (
    "こんにちは、世界。",
    "音声合成のための特徴量を、この文から抽出します。",
    "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。",
)

#: padded ケースで足す [PAD] の本数。`attention_mask=0` を混ぜてマスク経路
#: （mul → cast → bitwise_not → masked_fill、および conv 経路の 0 埋め）を踏ませる。
PAD_COUNT = 5


class HiddenStatesWrapper(nn.Module):
    """`ModelOutput`（dict）ではなく hidden_states のタプルを返す export 用ラッパ。

    IR v1 のグラフ出力は位置で引く（`output.<i>`）ので、全層の hidden_states をそのまま
    出力に並べる。層ごとに突合できるため、**誤差が層数でどう伸びるか**が golden から
    直接読める（tolerance の根拠づけがこれで実測になる）。

    `single_output` を立てると**最終層の 1 本だけ**を出力する（配布形）。ランタイムは
    `graph.outputs` を全部 readback するので、25 本出しのままだと 1 本しか使わない SBV2 でも
    毎 run で 25 本ぶんの staging + mapAsync を払う（ADR 0026 の「読み戻し ~52MB が支配」）。
    層を切り詰めた variant では最終層 = SBV2 が使う層なので、絞っても情報は落ちない。
    """

    def __init__(self, model: nn.Module, *, single_output: bool = False) -> None:
        super().__init__()
        self.model = model
        self.single_output = single_output

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[Any, ...]:
        out = self.model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        if self.single_output:
            return (out.hidden_states[-1],)
        return out.hidden_states


def load_model(model_id: str, num_layers: int) -> nn.Module:
    """実重みを取得し、層数を切り詰めて eval にする。

    MUST: `attn_implementation="eager"` — SDPA 経路は torch.export で別のグラフ形になり、
    recon の実測（27 種の未対応 op 分類）と対応が取れなくなる。
    """
    from transformers import DebertaV2Model

    model = DebertaV2Model.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="eager"
    )
    if num_layers > model.config.num_hidden_layers:
        raise ValueError(
            f"--layers {num_layers} がモデルの層数 {model.config.num_hidden_layers} を超えている"
        )
    model.encoder.layer = model.encoder.layer[:num_layers]
    model.config.num_hidden_layers = num_layers
    model.eval()
    return model


def build_cases(tokenizer: Any) -> tuple[tuple[str, torch.Tensor, torch.Tensor], ...]:
    """golden 4 ケース（3 文 + padded）の `(名前, input_ids, attention_mask)`。"""
    cases: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for index, text in enumerate(GOLDEN_SENTENCES):
        ids = torch.tensor([tokenizer(text)["input_ids"]], dtype=torch.int64)
        cases.append((f"case{index}", ids, torch.ones_like(ids)))

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("トークナイザに pad_token_id が無い（padded ケースを作れない）")
    base_ids = cases[0][1]
    pad_ids = torch.full((1, PAD_COUNT), pad_id, dtype=torch.int64)
    cases.append(
        (
            "padded",
            torch.cat([base_ids, pad_ids], dim=1),
            torch.cat([torch.ones_like(base_ids), torch.zeros_like(pad_ids)], dim=1),
        )
    )
    return tuple(cases)


def _write_io(
    wrapper: nn.Module,
    graph: IrGraph,
    cases: Sequence[tuple[str, torch.Tensor, torch.Tensor]],
    out_dir: Path,
    *,
    prefix: str = IO_PREFIX,
) -> list[str]:
    """各ケースの入力と torch CPU 期待出力を `<prefix><case>.safetensors` へ書く。"""
    written: list[str] = []
    for name, ids, mask in cases:
        with torch.no_grad():
            outputs = wrapper(ids, mask)
        if len(outputs) != len(graph.outputs):
            raise AssertionError(
                f"{name}: torch 出力 {len(outputs)} 本 ≠ IR 出力 {len(graph.outputs)} 本"
            )
        # MUST: io も IR の意味論 dtype の実表現へ落とす（i64 → I32）。ランタイムが受け取る
        # 形と揃っていないと Deno 側 E2E が golden を読めない（ADR 0009 の境界正規化）。
        args = {"input_ids": ids, "attention_mask": mask}
        tensors = {
            f"{INPUT_PREFIX}{declared.name}": normalize_boundary_tensor(
                args[declared.name], f"{name} の入力 '{declared.name}'"
            )
            for declared in graph.inputs
        }
        for index, tensor in enumerate(outputs):
            tensors[f"{OUTPUT_PREFIX}{index}"] = normalize_boundary_tensor(
                tensor.detach().contiguous(), f"{name} の出力 {index}"
            )
        path = out_dir / f"{prefix}{name}{IO_SUFFIX}"
        save_file(tensors, str(path))
        written.append(path.name)
    return written


def _write_mirror_io(
    wrapper: nn.Module,
    graph: IrGraph,
    cases: Sequence[tuple[str, torch.Tensor, torch.Tensor]],
    out_dir: Path,
) -> tuple[list[str], int]:
    """活性 i8 のフックを掛けた期待値（w8a8 の鏡像）を `io-i8a8.<case>` へ書く。

    MUST: フックの区間はこの関数の中で閉じる（`finally` で必ず外す）— 掛けたまま通常の
    `io.<case>` を採ると w8 E2E の期待値が汚染される。呼び出し側は**通常の io を先に**
    書き終えてからここへ来る。

    戻り値の本数は Deno 側の診断キー検査（`linearI8a8Key` の dispatch 本数）の期待定数の
    根拠になる。0 本は fail loudly — 「w8a8 のつもりで w8 の数を採った」を沈黙させない。
    """
    handles, attached = attach_act_quant(wrapper)
    print(f"[act-quant] 適格 linear {attached} 本に per-token i8 を適用（w8a8 鏡像）")
    if attached == 0:
        raise SystemExit("--act-quant を指定したが適格 linear が 0 本（適格判定の破れ）")
    try:
        return _write_io(wrapper, graph, cases, out_dir, prefix=ACT_IO_PREFIX), attached
    finally:
        detach_act_quant(handles)


def _fake_quant(dtype: str, wrapper: nn.Module) -> Mapping[str, torch.Tensor]:
    """格納 dtype の表現可能値へ重みを丸め、i8 の per-channel scale 台帳を返す（ADR 0006）。

    MUST: **export する `nn.Module` そのもの**（= `HiddenStatesWrapper`）に当てる。scale 台帳の
    キーはここで見た FQN で、safetensors のテンソルキー（= `torch.export` が見る FQN）と同じで
    なければ emit 側の突合が空振りする（`quantize.py` の FQN 規律 — `id()` 突合は禁止）。

    MUST: 呼ぶのは **golden io の採取より前**（`quantize.py` の docstring）— 後に当てると
    期待値だけが元の重みで計算され、E2E の差に量子化誤差が混ざって tolerance の意味が消える。
    """
    if dtype == "f32":
        return {}
    report = fake_quant_int8(wrapper)
    print(f"[fake-quant] i8 per-channel へ丸めた — {report.describe()}", flush=True)
    return report.scales


def export_variant(
    model_id: str,
    num_layers: int,
    out_dir: Path,
    *,
    sym_max: int = SYM_MAX,
    dtype: str = "f32",
    act_quant: bool = False,
    single_output: bool = False,
) -> dict[str, Any]:
    """1 層数ぶんの IR コンテナと golden io を書き、要約を返す。"""
    from transformers import AutoTokenizer

    model = load_model(model_id, num_layers)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    cases = build_cases(tokenizer)
    wrapper = HiddenStatesWrapper(model, single_output=single_output)
    scales = _fake_quant(dtype, wrapper)

    # 例示入力は padded ケース（mask に 0 を含む実トークン列）。min=2 は 0/1 特殊化を避ける
    # ため、max は Tmax 畳み込みの評価点そのもの（ADR 0010 — 別ノブで二重管理しない）。
    _, example_ids, example_mask = cases[-1]
    seq = Dim("T", min=2, max=sym_max)
    graph = export_to_file(
        wrapper,
        (example_ids, example_mask),
        out_dir / MODEL_FILE,
        dynamic_shapes=({1: seq}, {1: seq}),
        weight_dtype=dtype,
        weight_scales=scales,
    )
    # MUST: 通常の golden io は**フックなし**で採る（`_write_mirror_io` の docstring）。
    written = _write_io(wrapper, graph, cases, out_dir)
    mirror: list[str] = []
    attached = 0
    if act_quant:
        mirror, attached = _write_mirror_io(wrapper, graph, cases, out_dir)

    model_bytes = (out_dir / MODEL_FILE).stat().st_size
    return {
        "layers": num_layers,
        "dir": str(out_dir),
        "dtype": dtype,
        "nodes": len(graph.nodes),
        "outputs": len(graph.outputs),
        "initializers": len(graph.initializers),
        "model_bytes": model_bytes,
        "ops": sorted(graph.required_ops),
        "symbols": list(graph.symbols),
        "io": written,
        "act_quant_io": mirror,
        "act_quant_linears": attached,
        "case_lengths": {name: int(ids.shape[1]) for name, ids, _ in cases},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="出力先（既定は --dtype ごとの系列 — outputs/series/deberta{,-i8}/）",
    )
    parser.add_argument(
        "--dtype",
        choices=WEIGHT_DTYPES,
        default="f32",
        help="重みの格納 dtype（i8 は fake-quant してから適格スロットだけ圧縮格納する — ADR 0019）",
    )
    parser.add_argument(
        "--act-quant",
        action="store_true",
        help="適格 linear の入力を per-token i8 へ fake-quant した鏡像 io を追加で書く"
        "（ランタイムの linearCompute:'i8a8' の鏡像 — 重み側は --dtype i8 と併用する）",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=sorted(VARIANTS),
        help=f"書き出す層数（既定 {sorted(VARIANTS)}）",
    )
    parser.add_argument("--sym-max", type=int, default=SYM_MAX)
    args = parser.parse_args()
    # MUST: 活性 i8 は **i8 常駐重みの linear** にしか効かない（ADR 0025 決定 1）。f32 資産に
    # 対する鏡像は「ランタイムでは絶対に再現されない期待値」なので機械的に拒否する。
    if args.act_quant and args.dtype != "i8":
        raise SystemExit(
            f"--act-quant は --dtype i8 と併用する（指定は {args.dtype}）— "
            "活性 i8 は i8 常駐重みの linear にしか効かない（ADR 0025）"
        )
    if args.out is None:
        args.out = DEFAULT_OUT_ROOTS[args.dtype]

    summaries = []
    for num_layers in args.layers:
        variant = VARIANTS.get(num_layers, Variant(f"{num_layers}layer"))
        summaries.append(
            export_variant(
                args.model,
                num_layers,
                args.out / variant.name,
                sym_max=args.sym_max,
                dtype=args.dtype,
                act_quant=args.act_quant,
                single_output=variant.single_output,
            )
        )
    print(json.dumps({"model": args.model, "variants": summaries}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
