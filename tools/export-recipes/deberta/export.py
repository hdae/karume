"""実重み DeBERTa-v2（SBV2 text front）を IR v1 コンテナ + golden io へ書き出す台本。

`karume.goldens` の tiny golden が「op 契約の被覆」を受け持つのに対し、こちらは
**実重み・実トークン列での数値一致**を受け持つ（M1-P2 波 5）。生成物は `outputs/series/`
配下で、リポジトリ直下の `.gitignore` によりコミット対象外（重み 1.3GB 級）。

    uv run --with 'transformers==5.14.1' python -m deberta.export
    uv run --with 'transformers==5.14.1' python -m deberta.export --layers 2
    uv run --with 'transformers==5.14.1' python -m deberta.export --dtype i8 --act-quant
    uv run --with 'transformers==5.14.1' python -m deberta.export --dtype i4 --layers 22

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

`--dtype i4` も同じ理由で**別系列**（`outputs/series/deberta-i4/`）で、中身は**混成**
（`nn.Linear` / `nn.Embedding` = i4 group32・残り = i8）。SBV2 配布形では `i8+bert4` quant の
`text_encoder` 席に入る（`sbv2/distribution.py`）。i4 の実行経路は linear / embedding の
重みスロット限定（ADR 0069 決定 5）なので、単一 dtype の i4 系列は原理的に作れない。

i4 系列の encoder linear は **GPTQ 校正付きで丸める**（既定 — perf-ledger Q-6 / `deberta.calib`）。
格納形は 1 バイトも変わらない（格子は RTN i4 g32 のまま）で、変わるのは丸め値と scale 台帳の
中身だけ。校正入力は {@link deberta.calib_texts.CALIB_TEXTS} の 48 文を **golden と同じ入力
構築経路**（{@link build_graph_inputs}）で通したもの。校正の失敗は fail loudly で、素の RTN へ
黙って落ちる分岐は持たない（「校正付きのつもりで校正なしを配った」は資産から読めない）。

`--act-quant` は w8a8（`SessionOptions.linearCompute: "a8"`）の torch 鏡像で、適格
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

from _shared.paths import SERIES_ROOT
from karume.act_quant import attach_act_quant, detach_act_quant
from karume.convert import normalize_boundary_tensor
from karume.ir import IrGraph
from karume.pipeline import export_to_file
from karume.quantize import (
    DEFAULT_GROUP_SIZE,
    channel_rows,
    fake_quant_int4,
    fake_quant_int8,
    iter_quant_targets,
)
from karume.shards import resolve_shards

from . import calib, patch
from .calib_texts import CALIB_TEXTS

#: SBV2 text front が使う BERT そのもの（recon §1）。
MODEL_ID = "ku-nlp/deberta-v2-large-japanese-char-wwm"

#: 対応する格納 dtype。**f16 は無い** — SBV2 系列と一体で決める（タスク #30）。
#: `i4` は**混成**の系列名で、実体は「{@link I4_MODULE_TYPES} の適格な重み = i4 group32・
#: それ以外（conv・group 長で割り切れない重み）= 従来どおり i8 per-channel」
#: （{@link BASE_WEIGHT_DTYPES} / {@link _fake_quant}）。i4 の実行経路が linear / embedding の
#: 重みスロット限定である以上（ADR 0069 決定 5）、系列としては混成にしかなり得ない。
WEIGHT_DTYPES: tuple[str, ...] = ("f32", "i8", "i4")

#: i4 group32 で丸める**モジュール型**（= i4 の実行経路を持つ op と対 — `karume.emit` の
#: `I4_WEIGHT_OPS`）。conv 系は展開経路が無いので入れない（入れると emit が fail loudly する）。
I4_MODULE_TYPES: tuple[type[nn.Module], ...] = (nn.Linear, nn.Embedding)

#: `nn.Embedding` の器に入っているが**表引きされない**重みの FQN。相対位置の埋め込み表は
#: `deberta.patch` が差し替えた forward で「表を切り出して query_proj / key_proj に通す」形で
#: 使われるので、グラフに `embedding` op が立たない — 重みスロットの消費がゼロなので圧縮格納の
#: 適格外で、i8 系列でも f32 のまま残っている。型だけで i4 に振ると emit が「適格でない」で
#: fail loudly するため、i4 の対象集合からは名前で外す（丸めは従来どおり i8 側が担う）。
#:
#: NOTE: 綴りが上流のモジュール名の変更で空振りしても沈黙はしない — 除外が消えた瞬間に emit の
#: 明示指定の門が「適格でない」でそのテンソル名を挙げて落ちる。
NON_LOOKUP_EMBEDDINGS: frozenset[str] = frozenset({"model.encoder.rel_embeddings"})

#: encoder stage の**外**に居る i4 適格の重み（= 語彙表 1 本）。校正付き丸めは stage 逐次の
#: 駆動なので stage の中しか丸められず、外の適格は素の RTN i4 が担う
#: （{@link _round_i4_calibrated}）。
#:
#: 名前で宣言するのは「黙った分類替え」を作らないため — 実測（config から組んだ全層 wrapper）で
#: stage 外の i4 適格はこの 1 本だけ（`embed_proj` は `embedding_size == hidden_size` で
#: `None`・`position_embeddings` / `token_type_embeddings` も config で `None`・`rel_embeddings` は
#: {@link NON_LOOKUP_EMBEDDINGS} で適格外）。上流の構成が変わって適格が増減したら、
#: {@link _round_i4_calibrated} の門がその FQN を挙げて落ちる。
NON_STAGE_I4_WEIGHTS: frozenset[str] = frozenset({"model.embeddings.word_embeddings"})

#: 系列名 → `export_to_file` へ渡す**既定**の格納 dtype。i4 系列だけ既定が i8 で、適格な
#: linear / embedding は 1 本単位の `weight_dtype_overrides` で i4 へ振る（gemma4 の混成
#: i8+i4 と同形）。
BASE_WEIGHT_DTYPES: Mapping[str, str] = {"f32": "f32", "i8": "i8", "i4": "i8"}

#: 生成物の既定の置き場（格納 dtype 別の**系列**）。親は `SERIES_ROOT`（= outputs/series/）—
#: models/ は配布形だけの場所（_shared.paths）。`.gitignore` の `outputs/` でコミット対象外。
#: MUST: dtype ごとに別ディレクトリ（ADR 0019）— 同居させると f32 系列の網が i8 資産に掛かる。
DEFAULT_OUT_ROOTS: Mapping[str, Path] = {
    "f32": SERIES_ROOT / "deberta",
    "i8": SERIES_ROOT / "deberta-i8",
    "i4": SERIES_ROOT / "deberta-i4",
}

#: 1 ケースぶんのグラフ入力（名前 → テンソル）。
InputArgs = Mapping[str, torch.Tensor]

#: グラフ入力の順序 = `HiddenStatesWrapper.forward` の引数順。torch.export はこの順で
#: `graph.inputs` を並べるので、export 後に突合して**ずれていたら止める**（位置で渡す以上、
#: 黙ってずれると golden の入力だけが入れ替わる）。
INPUT_ORDER: tuple[str, ...] = ("input_ids", "attention_mask", "c2p_pos", "p2c_pos")

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
#: ADR 0045 / docs/research/2026-08-11-deberta-size-recon.md §4。
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

    MUST: `DebertaV2Model.forward` は使わず embeddings + encoder を直接呼ぶ — 前者は
    `relative_pos` を encoder へ渡す口を持たず、相対位置の添字表を**外部供給にできない**
    （`deberta.patch` — 焼き込むと Tmax=512 で 2MiB の定数が 2 本入る）。z_steps 分岐は
    `self.z_steps = 0` のハードコードで恒久的に死んでいるので写さない。
    """

    def __init__(self, model: nn.Module, *, single_output: bool = False) -> None:
        super().__init__()
        self.model = model
        self.single_output = single_output

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        c2p_pos: torch.Tensor,
        p2c_pos: torch.Tensor,
    ) -> tuple[Any, ...]:
        model = self.model
        embedding_output = model.embeddings(
            input_ids=input_ids,
            token_type_ids=torch.zeros_like(input_ids),
            position_ids=None,
            mask=attention_mask,
            inputs_embeds=None,
        )
        hidden_states = model.encoder(
            embedding_output,
            attention_mask,
            output_hidden_states=True,
            relative_pos=(c2p_pos, p2c_pos),
            return_dict=True,
        ).hidden_states
        if self.single_output:
            return (hidden_states[-1],)
        return hidden_states


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


def encode_text(tokenizer: Any, text: str) -> tuple[torch.Tensor, torch.Tensor]:
    """1 文を `[1, T]` の `(input_ids, attention_mask)` へ落とす（mask は全 1）。"""
    ids = torch.tensor([tokenizer(text)["input_ids"]], dtype=torch.int64)
    return ids, torch.ones_like(ids)


def build_graph_inputs(model: nn.Module, ids: torch.Tensor, mask: torch.Tensor) -> InputArgs:
    """`(input_ids, attention_mask)` へ相対位置の添字表を足して 4 入力の組にする。

    相対位置の添字表はグラフ入力なのでケースごとに実長で作る（`deberta.patch` が正本）。
    バケット幅と最大位置は**モジュールから読む** — config の `max_relative_positions` は
    -1 のとき `max_position_embeddings` へフォールバックする規則を持つので、写すと二重管理になる。

    golden ケース（{@link build_cases}）と校正入力（{@link build_calib_args}）が**同じここを
    通る**こと MUST — 別の綴りで組むと、校正で見た活性と golden が流す活性が黙って別物になる。
    """
    attention = model.encoder.layer[0].attention.self
    c2p_pos, p2c_pos = patch.build_rel_pos_tables(
        int(ids.shape[1]),
        position_buckets=attention.position_buckets,
        max_position=attention.max_relative_positions,
    )
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "c2p_pos": c2p_pos,
        "p2c_pos": p2c_pos,
    }


def build_cases(tokenizer: Any, model: nn.Module) -> tuple[tuple[str, InputArgs], ...]:
    """golden 4 ケース（3 文 + padded）の `(名前, グラフ入力名 → テンソル)`。"""
    ids_cases: list[tuple[str, torch.Tensor, torch.Tensor]] = [
        (f"case{index}", *encode_text(tokenizer, text))
        for index, text in enumerate(GOLDEN_SENTENCES)
    ]

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("トークナイザに pad_token_id が無い（padded ケースを作れない）")
    base_ids = ids_cases[0][1]
    pad_ids = torch.full((1, PAD_COUNT), pad_id, dtype=torch.int64)
    ids_cases.append(
        (
            "padded",
            torch.cat([base_ids, pad_ids], dim=1),
            torch.cat([torch.ones_like(base_ids), torch.zeros_like(pad_ids)], dim=1),
        )
    )
    return tuple((name, build_graph_inputs(model, ids, mask)) for name, ids, mask in ids_cases)


def build_calib_args(
    tokenizer: Any, model: nn.Module, texts: Sequence[str]
) -> tuple[tuple[torch.Tensor, ...], ...]:
    """校正コーパスを wrapper の**位置引数**（{@link INPUT_ORDER} の並び）の列へ落とす。

    トークナイズも添字表も golden と同じ経路（{@link encode_text} / {@link build_graph_inputs}）
    を通す。位置引数にするのは、校正リグ（`deberta.calib`）がグラフ入力の**名前を知らない**で
    済ませるため（リグは wrapper を呼ぶだけ）。
    """
    if not texts:
        raise ValueError("校正コーパスが空（校正付き i4 は入力ゼロでは成立しない）")
    built: list[tuple[torch.Tensor, ...]] = []
    for text in texts:
        args = build_graph_inputs(model, *encode_text(tokenizer, text))
        built.append(tuple(args[key] for key in INPUT_ORDER))
    return tuple(built)


def _write_io(
    wrapper: nn.Module,
    graph: IrGraph,
    cases: Sequence[tuple[str, InputArgs]],
    out_dir: Path,
    *,
    prefix: str = IO_PREFIX,
) -> list[str]:
    """各ケースの入力と torch CPU 期待出力を `<prefix><case>.safetensors` へ書く。"""
    written: list[str] = []
    for name, args in cases:
        # 位置で渡すが並びは `graph.inputs` から引く（export 側で INPUT_ORDER と突合済み）。
        with torch.no_grad():
            outputs = wrapper(*(args[declared.name] for declared in graph.inputs))
        if len(outputs) != len(graph.outputs):
            raise AssertionError(
                f"{name}: torch 出力 {len(outputs)} 本 ≠ IR 出力 {len(graph.outputs)} 本"
            )
        # MUST: io も IR の意味論 dtype の実表現へ落とす（i64 → I32）。ランタイムが受け取る
        # 形と揃っていないと Deno 側 E2E が golden を読めない（ADR 0009 の境界正規化）。
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
    cases: Sequence[tuple[str, InputArgs]],
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


def _i4_module_names(wrapper: nn.Module) -> frozenset[str]:
    """i4 group32 で丸めるモジュールの FQN 集合（混成 i8+i4 の**排他割り**の唯一の源）。

    適格は 3 条件の積: ① {@link I4_MODULE_TYPES} であること（emit の i4 適格 = linear /
    embedding の重みスロット限定 — ADR 0069 決定 5。外れたテンソルへ i4 を明示指定すると emit が
    fail loudly する）② 量子化軸が group 長で割り切れること（i4 は端数 group を作らない MUST —
    同決定 2。外れた重みは**構成ごと落とすのではなく対象から外す**ので i8 側へ落ちる）
    ③ {@link NON_LOOKUP_EMBEDDINGS} でないこと（器は `nn.Embedding` でも表引きされない重み）。

    対象列挙を core（`iter_quant_targets`）に通すのは、丸めが見る集合とここが数える集合を
    1 本の実装のままにするため。i8 側の述語を「この集合に居ない」で書くのも同じ理由で、
    2 つの述語を別々の綴りから作ると、どちらにも入らない重みが**黙って f32 のまま残る**
    （二重丸め禁止の逆側の穴で、値は正しいままサイズだけが戻る）。
    """
    return frozenset(
        fqn.removesuffix(".weight")
        for fqn, weight, axis in iter_quant_targets(wrapper, I4_MODULE_TYPES)
        if channel_rows(weight, axis).shape[-1] % DEFAULT_GROUP_SIZE == 0
        and fqn.removesuffix(".weight") not in NON_LOOKUP_EMBEDDINGS
    )


def _round_i4_plain(
    wrapper: nn.Module, names: frozenset[str], label: str
) -> Mapping[str, torch.Tensor]:
    """名指しの集合を素の RTN i4 g32 で丸める（校正を通さない 1 段目の格子そのもの）。

    `label` は診断行の主語（助詞まで込み — 「〜を i4 group へ丸めた」に嵌まる形）。
    """
    report = fake_quant_int4(
        wrapper,
        include=lambda name: name in names,
        op_types=I4_MODULE_TYPES,
    )
    print(f"[fake-quant] {label} i4 group へ丸めた（RTN） — {report.describe()}", flush=True)
    return report.scales


def _round_i4_calibrated(
    wrapper: nn.Module,
    i4_names: frozenset[str],
    calib_args: Sequence[Sequence[torch.Tensor]],
) -> Mapping[str, torch.Tensor]:
    """i4 適格を「stage 外 = RTN」「encoder stage 内 = GPTQ」へ排他に割って丸める。

    順序 MUST（① → ② → ③）:

    1. **stage 外の適格（語彙表）を先に RTN i4 で丸める** — 配布実行時に encoder へ入るのは
       i4 の語彙表を引いた活性なので、校正入力を配布条件へ合わせる（語彙表の丸め誤差が
       伝播した状態で encoder の丸め先を選ぶ）。後に回すと「f32 の表を引いた活性」で校正した
       重みを、i4 の表と組んで配ることになる。
    2. 校正バッチの捕捉と stage 分解一致門（`deberta.calib.build_rig`）。
    3. stage 内の適格 linear を GPTQ × RTN 格子で丸める。

    MUST: stage 外の適格は {@link NON_STAGE_I4_WEIGHTS} と一致すること — 一致しないなら上流の
    構成が変わって適格が増減している。黙って RTN 側へ流す（= 校正が痩せる）のも、黙って i8 へ
    落とす（= サイズだけ戻る）のも数字から読めないので、その場で落とす。
    """
    stages = calib.encoder_stages(wrapper)
    stage_names = calib.stage_linear_names(stages) & i4_names
    plain_names = i4_names - stage_names
    if plain_names != NON_STAGE_I4_WEIGHTS:
        raise AssertionError(
            f"encoder stage の外の i4 適格が {sorted(plain_names)} で、宣言"
            f"（NON_STAGE_I4_WEIGHTS = {sorted(NON_STAGE_I4_WEIGHTS)}）と違う"
            " — 上流の構成が変わって i4 の割り方が動いている"
        )
    plain = _round_i4_plain(wrapper, plain_names, "stage 外の語彙表を")
    rig = calib.build_rig(wrapper, stages, calib_args)
    report, ledger = calib.calibrate_i4(
        rig, include=lambda local: f"{calib.STAGE_PREFIX}.{local}" in stage_names
    )
    calibrated = ledger.scales
    # MUST: 2 経路は互いに素（重なれば同じ重みを 2 度丸めたことになり、値だけが静かに狂う）。
    overlap = sorted(set(plain) & set(calibrated))
    if overlap:
        raise AssertionError(f"i4 の 2 経路が同じ重みを丸めている（二重丸め）: {overlap[:3]}")
    print(
        f"[fake-quant] encoder stage の linear を GPTQ 校正付きで丸めた — {report.describe()}"
        f" / 校正入力 {len(rig.batches)} 件・{rig.tokens:,} トークン",
        flush=True,
    )
    return {**plain, **calibrated}


def _fake_quant(
    dtype: str,
    wrapper: nn.Module,
    *,
    calib_args: Sequence[Sequence[torch.Tensor]] | None = None,
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, str]]:
    """格納 dtype の表現可能値へ重みを丸め、scale 台帳と 1 本単位の格納指定を返す（ADR 0006）。

    MUST: **export する `nn.Module` そのもの**（= `HiddenStatesWrapper`）に当てる。scale 台帳の
    キーはここで見た FQN で、safetensors のテンソルキー（= `torch.export` が見る FQN）と同じで
    なければ emit 側の突合が空振りする（`quantize.py` の FQN 規律 — `id()` 突合は禁止）。

    MUST: 呼ぶのは **golden io の採取より前**（`quantize.py` の docstring）— 後に当てると
    期待値だけが元の重みで計算され、E2E の差に量子化誤差が混ざって tolerance の意味が消える。
    校正付き丸め（GPTQ）も同じ理由でここに閉じる。

    i4 系列は混成（適格な linear / embedding = i4 group32・残り = i8 per-channel）で、2 つの
    述語は {@link _i4_module_names} から**排他に**割る（`quantize.py` の混成 MUST）。返す
    override は「i4 の scale 台帳のキー全部を i4 に振る」写像で、emit 側は明示指定を
    満たせなければ fail loudly する。

    `calib_args` は i4 のときだけ効く（`None` = 校正なしの素の RTN — テスト用の opt-out）。
    """
    if dtype == "f32":
        return {}, {}
    if dtype == "i8":
        report = fake_quant_int8(wrapper)
        print(f"[fake-quant] i8 per-channel へ丸めた — {report.describe()}", flush=True)
        return report.scales, {}
    i4_names = _i4_module_names(wrapper)
    int4_scales = (
        _round_i4_plain(wrapper, i4_names, "linear / embedding を")
        if calib_args is None
        else _round_i4_calibrated(wrapper, i4_names, calib_args)
    )
    # MUST: 丸めた集合 = 格納集合（override のキー）。ずれるのは「適格と数えたのに丸まって
    # いない」形で、i4 席に i8 の重みが混ざったまま緑になる。
    if set(int4_scales) != {f"{name}.weight" for name in i4_names}:
        raise AssertionError(
            f"i4 適格 {len(i4_names)} 本に対し丸めたのは {len(int4_scales)} 本"
            f"（過不足: {sorted(set(int4_scales) ^ {f'{name}.weight' for name in i4_names})[:3]}）"
        )
    int8 = fake_quant_int8(wrapper, include=lambda name: name not in i4_names)
    print(f"[fake-quant] 残りは i8 per-channel — {int8.describe()}", flush=True)
    return {**int8.scales, **int4_scales}, dict.fromkeys(int4_scales, "i4")


def export_variant(
    model_id: str,
    num_layers: int,
    out_dir: Path,
    *,
    sym_max: int = SYM_MAX,
    dtype: str = "f32",
    act_quant: bool = False,
    single_output: bool = False,
    calib_texts: Sequence[str] | None = CALIB_TEXTS,
) -> dict[str, Any]:
    """1 層数ぶんの IR コンテナと golden io を書き、要約を返す。

    `calib_texts` は **i4 系列だけ**が読む校正コーパス（既定 = 全 48 文）。`None` にすると
    校正なしの素の RTN i4 に戻る — テスト用の opt-out で、CLI からは届かない
    （`--dtype i4` は必ず校正付き）。
    """
    from transformers import AutoTokenizer

    # MUST: 差し替えは golden を採る前（`deberta.patch` の docstring）— 後に当てると期待値だけが
    # 元の経路（表を内部で作る形）で計算され、グラフと食い違ったまま緑になる。
    model = load_model(model_id, num_layers)
    patch.assert_supported(model.config)
    patch.apply_external_rel_pos_patch()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    cases = build_cases(tokenizer, model)
    wrapper = HiddenStatesWrapper(model, single_output=single_output)
    # MUST: 校正入力は**パッチ適用後の wrapper**へ流す（stage の実シグネチャがパッチで変わる）。
    calib_args = (
        build_calib_args(tokenizer, model, calib_texts)
        if dtype == "i4" and calib_texts is not None
        else None
    )
    scales, dtype_overrides = _fake_quant(dtype, wrapper, calib_args=calib_args)

    # 例示入力は padded ケース（mask に 0 を含む実トークン列）。min=2 は 0/1 特殊化を避ける
    # ため、max は Tmax 畳み込みの評価点そのもの（ADR 0010 — 別ノブで二重管理しない）。
    _, example_args = cases[-1]
    seq = Dim("T", min=2, max=sym_max)
    graph = export_to_file(
        wrapper,
        tuple(example_args[key] for key in INPUT_ORDER),
        out_dir / MODEL_FILE,
        # 添字表は `[T, T]` — 両軸が同じ記号（正方であることを export の段で縛る）。
        dynamic_shapes=({1: seq}, {1: seq}, {0: seq, 1: seq}, {0: seq, 1: seq}),
        weight_dtype=BASE_WEIGHT_DTYPES[dtype],
        weight_scales=scales,
        weight_dtype_overrides=dtype_overrides,
    )
    declared = tuple(item.name for item in graph.inputs)
    if declared != INPUT_ORDER:
        raise AssertionError(f"グラフ入力の並びが {declared} で、期待の {INPUT_ORDER} と違う")
    # MUST: 通常の golden io は**フックなし**で採る（`_write_mirror_io` の docstring）。
    written = _write_io(wrapper, graph, cases, out_dir)
    mirror: list[str] = []
    attached = 0
    if act_quant:
        mirror, attached = _write_mirror_io(wrapper, graph, cases, out_dir)

    model_bytes = sum(p.stat().st_size for p in resolve_shards(out_dir / MODEL_FILE))
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
        "calib_texts": len(calib_args) if calib_args is not None else 0,
        "case_lengths": {name: int(args["input_ids"].shape[1]) for name, args in cases},
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="出力先（既定は --dtype ごとの系列 — outputs/series/deberta{,-i8,-i4}/）",
    )
    parser.add_argument(
        "--dtype",
        choices=WEIGHT_DTYPES,
        default="f32",
        help="重みの格納 dtype（i8 は fake-quant してから適格スロットだけ圧縮格納する — ADR 0019。"
        "i4 は混成で、適格な linear / embedding が group32 の i4・残りは i8 — ADR 0069。"
        f"i4 の encoder linear は GPTQ 校正付き（校正コーパス {len(CALIB_TEXTS)} 文）で丸める）",
    )
    parser.add_argument(
        "--act-quant",
        action="store_true",
        help="適格 linear の入力を per-token i8 へ fake-quant した鏡像 io を追加で書く"
        "（ランタイムの linearCompute:'a8' の鏡像 — 重み側は --dtype i8 と併用する）",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=sorted(VARIANTS),
        help=f"書き出す層数（既定 {sorted(VARIANTS)}）",
    )
    parser.add_argument("--sym-max", type=int, default=SYM_MAX)
    args = parser.parse_args(argv)
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
