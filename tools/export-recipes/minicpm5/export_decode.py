"""実重み MiniCPM5-1B を **states 形の chunk グラフ**（IR v1 + golden）へ書き出す台本。

ADR [0066](../../../docs/decisions/0066-generation-context-state-slots.md)（GenerationContext と
名前付き state スロット）/
[0067](../../../docs/decisions/0067-autoregressive-attention-vocabulary.md) 決定 4〜5b
（state 参照つき attention と `state_append`）/
[0068](../../../docs/decisions/0068-decode-exit-multi-output.md) 決定 4（decode 出口）を
**実重みで検収する足場**。1-shot 形（{@link minicpm5.export}）の decode 版。モデル非依存の門
（greedy の採り方・余裕門・容量門・公開の入れ替え）の正本は {@link _shared.decode_series}、
1-shot 形のヘルパは {@link minicpm5.export} で、どちらも import して再利用する（同じ規律を
2 箇所に書かない）。

    uv run --with 'transformers==5.14.1' python -m minicpm5.export_decode

## 何をグラフに載せるか

`(input_ids[1,M], position_ids[1,M]) → (logits[1,M,V], token[1,M,1])` の **chunk ラッパ** 1 本。
`M` は物理 chunk 次元（ADR 0067 決定 4 — prefill は `chunkLength` / decode は 1）で、有効行は
先頭 `queryLength` 行の compact-prefix。`token` は `logits.argmax(-1, keepdim=True)`
（ADR 0068 決定 4 の decode 出口 — torch の i64 は境界正規化で i32 へ落ちる）。
**出力順は `[logits, token]` 固定**で、ランタイム側の slot 番号がこの順を読む。

## 位置は入力・RoPE は表引き（1-shot 形との唯一の構造差）

1-shot 形は `position_ids` を持たない — 位置は常に `0..T-1` なので、RoPE の cos / sin は
`arange` の定数畳み込みで Tmax 表 + `sym_prefix_slice` に落ちる（ADR 0010）。decode では
位置が `pastLength + row` で**実行時にしか決まらない**ので、この畳み込みは成立しない。

そこで `model.model.rotary_emb` を {@link RopeTable}（表引きモジュール）の instance へ
差し替える（上流のモデリングコードは 1 行も触らない）。表は**元の `rotary_emb` を
`arange(POS_MAX)[None]` で 1 回呼んで**得た cos / sin をそのまま持ち、forward は
`F.embedding(position_ids, table)` で引く。値同一は {@link swap_rope_table} が構築時に
3 種の position 列で `torch.equal` 突合する（引き方が合っていることは値でしか確かめられない）。

MUST: `rope.assert_rope_lifted` は**適用しない** — あれは `inv_freq` バッファを畳み込みの葉へ
降格する門で、本台本は rotary モジュールを丸ごと差し替えるため `inv_freq` は模型から消える。
代わりに {@link assert_ir_form_decode} が **`sin` 系 op の不在**を直接見る（`cos` は IR 語彙に
無いので、RoPE が残ったグラフは export 自体が落ちる — 到達しうる残骸は `sin` の側だけ）。

## mask は「trace を通すためだけ」に居る

`attention` の第 4 入力（加算 causal mask）は 1-shot 形と同じ {@link minicpm5.export
.additive_causal_mask} を使うが、states 形では causal が**述語計算**（`col ≤ pastLength + row`
— ADR 0067 決定 4）になるので mask tensor は要らない。手術（`karume.states.to_states_form`）が
mask 入力を落とし、Tmax² 定数と `sym_prefix_slice` を刈る。{@link assert_ir_form_decode} は
その残骸が 1 本も残っていないことを見る（残ると「誰も読まない 1MiB」が配布物に居座る）。

## 出力レイアウト

    outputs/series/minicpm5-1b-decode/model.safetensors         重み・定数 + karume_ir
    outputs/series/minicpm5-1b-decode/io.<case>.safetensors     無 pad 全長の入出力
    outputs/series/minicpm5-1b-decode/greedy.<case>.safetensors greedy 継続 K step の期待列

`io.*` は 1-shot 形と同じキー規約（`input.<グラフ入力名>` / `output.<位置>`）。
`greedy.*` は **decode 検収の正本**で、`prompt` i32 `[T]` / `expected` i32 `[K]` /
`margin` f32 `[K]` を持つ。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import nn
from torch.export import Dim
from torch.nn import functional

from _shared.decode_series import _write_greedy, assert_case_room, positions_for
from _shared.paths import SERIES_ROOT
from karume.artifacts import staged_publication
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION, normalize_boundary_tensor
from karume.ir import IrGraph
from karume.ops import ARGMAX_OP, ATTENTION_OP, STATE_APPEND_OP
from karume.pipeline import export_module, publish_model
from karume.shapes import declared_shape
from karume.shards import resolve_shards
from karume.states import StateAttentionSpec, StatesPlan, to_states_form
from minicpm5 import export as one_shot

#: 生成物の既定の置き場（1-shot 系列とは別ディレクトリ — グラフの形が違う別資産）。
DEFAULT_OUT_DIR = SERIES_ROOT / "minicpm5-1b-decode"

#: 公式重みの置き場（1-shot 形と同じ素材）。
DEFAULT_MODEL_DIR = one_shot.DEFAULT_MODEL_DIR

#: グラフ入力の名前（chunk ラッパの forward 引数名）。{@link assert_ir_form_decode} が
#: 「この 2 本だけ」を見るための綴り。
INPUT_IDS = one_shot.INPUT_IDS
POSITION_IDS = "position_ids"

#: 物理 chunk 次元の記号名（IR 上の名前は `symbol_names` が決める — Dim 名と同じ綴りにする）。
SEQ_SYMBOL = "M"

#: state スロットの容量記号（ADR 0066 決定 2 / 追記 7 — 束縛点は `createGenerationContext`）。
#: **値 shape には現れない states 専用記号**で、export 時に容量を焼かないための席。
CAPACITY_SYMBOL = "C"

#: RoPE 表の位置数（= 表引きできる絶対位置の上限）。`SYM_MAX` と同値だが**別の概念**なので
#: 別名で宣言する — SYM_MAX は 1 chunk の最大行数、こちらは prompt + 生成を通した絶対位置の
#: 上限で、decode では `pastLength + M` がこれを超えられない。片方だけ動かす日が来る。
ROPE_TABLE_POSITIONS = 512

#: greedy golden の継続 step 数（ADR 0068 決定 4 の出口を多 step で踏む）。
GREEDY_STEPS = 16

#: 各 step の top1 − top2 logit 差の下限。GPU 実行の偏差（実測 ~1e-4 オーダ）で greedy の列が
#: 割れないことの保証で、**恒真でない**（強く決まらない継続を持つケースはここで落ちる）。
#: 落ちたケースは K を下げずに {@link GREEDY_CASES} から外す — 余裕の無い列を golden に
#: すると「GPU 側が正しくても赤」の門になる。
#:
#: MUST: 検収門の前提（`e2e_minicpm5_greedy_test.ts` — `minMargin > 2 × atol`・
#: atol = `PREFILL_ATOL` 1e-3）**より上**に置く。下だと「台本は採るが門の前提で落ちる」
#: ケースが作れてしまい、門の「ここが落ちるのは台本と資産が食い違ったときだけ」が
#: 成立しない（gemma4 側と同じ規律 — 2026-08-19 レビュー G3-03）。
MARGIN_FLOOR = 1e-2

#: greedy golden を採るケース（{@link minicpm5.export.GOLDEN_CASES} の部分集合）。
#: io golden は 4 ケース全部で採る（こちらは 1 step ぶんなので余裕門が要らない）。
#:
#: **`capital-ja` は除外**（2026-08-18 実測 — K=16 の step 5 で margin 0.00773 < 1e-2）。
#: 4 ケースの余裕の実測最小値は capital-en 0.2111 (step 5) / capital-ja 0.00773 (step 5) /
#: context-en 0.03412 (step 3) / context-ja 0.02610 (step 15)。K は下げない — 短い列にしても
#: 「継続が強く決まらない」という性質そのものは消えず、他ケースの検出力だけが落ちる。
GREEDY_CASES: tuple[str, ...] = ("capital-en", "context-en", "context-ja")

#: 手術で死ぬべき残骸 op。`sym_prefix_slice` は mask の Tmax 畳み込み（ADR 0010）で、`sin` は
#: RoPE が畳み込みにも表引きにも落ちなかったときの生き残り。
#: NOTE: `cos` は IR 語彙に無い（ops.py の UNARY_OPS は `sin` だけ）ので、cos が残る形は
#: export が落ちて here まで来ない — 検査に載せるのは実際に到達しうる 2 本だけ。
RESIDUE_OPS = (one_shot.SYM_PREFIX_SLICE_OP, "sin")


def slot_name(layer: int, part: str) -> str:
    """層番号つきの state スロット名（`l0.k` / `l0.v` …）。

    綴りを 1 箇所に閉じる — 手術の指定（{@link states_plan}）と形検査
    （{@link assert_ir_form_decode}）が別々に組み立てると、綴りのズレが
    「スロットは在るが誰も読まない」形で通ってしまう。
    """
    return f"l{layer}.{part}"


class RopeTable(nn.Module):
    """`LlamaRotaryEmbedding` と同じ呼び出し規約を持つ**位置表の引き**モジュール。

    上流は `self.rotary_emb(hidden_states, position_ids=position_ids)` で呼び
    `(cos, sin)` の 2 タプル（各 `[B, seq, head_dim]`）を受ける（modeling_llama.py:408）。
    ここはその規約だけを満たし、`inv_freq` からの三角関数計算を**表の gather** に置き換える。

    MUST: 表は素の属性で持つ（バッファにしない）— `rope.lift_rope_buffers` が既存モデルで
    やっている降格と同じ形で、torch.export は lifted tensor constant として拾う。
    MUST: 戻り値に dtype キャストを挟まない。上流は `cos.to(dtype=x.dtype)` を最後に置くが、
    表は模型と同じ dtype で作る（{@link swap_rope_table} が突合する）ので恒等であり、
    恒等キャストを書くと export へ `_to_copy` が漏れうる。
    """

    def __init__(self, cos: torch.Tensor, sin: torch.Tensor) -> None:
        super().__init__()
        self.cos_table = cos
        self.sin_table = sin

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`position_ids[B,seq]` で表を引く（`x` は規約上の引数 — 値は読まない）。"""
        del x
        return (
            functional.embedding(position_ids, self.cos_table),
            functional.embedding(position_ids, self.sin_table),
        )


def build_rope_table(rotary: nn.Module, probe: torch.Tensor, positions: int) -> RopeTable:
    """元の rotary を `arange(positions)` で 1 回呼び、その cos / sin を表にした {@link RopeTable}。

    MUST: 表は**元実装の出力そのもの**で作る（`inv_freq` から作り直さない）— 作り直すと
    `attention_scaling` や rope_type 別の初期化を写し損ねる経路ができ、値同一の突合が
    「同じ式を 2 回書いて比べただけ」の恒真に近づく。
    """
    with torch.no_grad():
        cos, sin = rotary(probe, position_ids=torch.arange(positions).unsqueeze(0))
    return RopeTable(cos[0].detach().clone(), sin[0].detach().clone())


def assert_rope_table_matches(
    rotary: nn.Module, table: RopeTable, probe: torch.Tensor, positions: int
) -> None:
    """表引きが元実装と**完全一致**（タプル構成・shape・dtype・全要素）であることを見る。

    MUST: 恒真でない。表は `arange` 1 本から作るので、「添字 → 行」の引き方が正しいことは
    値の同一でしか確かめられない。踏む 3 種は ①連続（`0..4`）②オフセット付き（`3..9` —
    decode の `pastLength + row`）③非単調（並びに依らず**値で**引くことの確認）。
    """
    probes = (
        torch.arange(5).unsqueeze(0),
        torch.arange(3, 10).unsqueeze(0),
        torch.tensor([[positions - 1, 0, 17, 2]], dtype=torch.int64),
    )
    for position_ids in probes:
        with torch.no_grad():
            reference = rotary(probe, position_ids=position_ids)
            actual = table(probe, position_ids=position_ids)
        where = f"position_ids={position_ids.tolist()}"
        if len(actual) != len(reference):
            raise AssertionError(
                f"{where}: 戻りが {len(actual)} 本（元実装は {len(reference)} 本）"
            )
        for index, (got, want) in enumerate(zip(actual, reference, strict=True)):
            if got.dtype is not want.dtype or got.shape != want.shape:
                raise AssertionError(
                    f"{where}: 戻り {index} が {got.dtype} {tuple(got.shape)} —"
                    f" 元実装は {want.dtype} {tuple(want.shape)}"
                )
            if not torch.equal(got, want):
                raise AssertionError(
                    f"{where}: 戻り {index} の値が元実装と違う"
                    f"（最大差 {float((got - want).abs().max())}）— 表の引き方がずれている"
                )


def swap_rope_table(model: nn.Module, positions: int) -> RopeTable:
    """`model.model.rotary_emb` を表引きへ差し替える（値同一を確認してから差し替える）。

    差し替え**前**に突合するのは、失敗したときに模型が半端な状態で残らないようにするため。
    """
    rotary = model.model.rotary_emb
    probe = torch.zeros(1, 1, int(model.config.hidden_size), dtype=torch.float32)
    table = build_rope_table(rotary, probe, positions)
    assert_rope_table_matches(rotary, table, probe, positions)
    model.model.rotary_emb = table
    return table


class DecodeChunkWrapper(nn.Module):
    """`LlamaForCausalLM` を chunk 形（`(input_ids, position_ids) → (logits, token)`）に
    固定した export 用ラッパ。

    MUST: `attention_mask` は自前の 4D 加算マスク（1-shot 形と同じ理由 — 省略すると
    transformers が `is_causal` 任せの経路へ入り `gqa_sdpa_attention` の mask 門で落ちる）。
    trace を通すためだけの存在で、手術が落とす。
    MUST: `use_cache=False`。True だと `DynamicCache` が生えて KV を出力へ返す形になり、
    「state はグラフの通常 I/O にしない」（ADR 0066 決定 1）と正面から食い違う。
    MUST: 出力順は `[logits, token]`。ランタイム側は slot 番号でこの 2 本を読む。
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = one_shot.additive_causal_mask(input_ids.shape[1])
        logits = self.model(
            input_ids=input_ids,
            attention_mask=mask,
            position_ids=position_ids,
            use_cache=False,
        ).logits
        return logits, logits.argmax(-1, keepdim=True)


def load_wrapper(model_dir: Path, *, positions: int = ROPE_TABLE_POSITIONS) -> DecodeChunkWrapper:
    """実重みを f32 で読み、RoPE を表引きへ差し替えた export 可能な chunk ラッパを返す。"""
    from transformers import LlamaForCausalLM

    one_shot.register_attention()
    model = LlamaForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation=one_shot.ATTENTION_NAME
    )
    model.eval()
    swap_rope_table(model, positions)
    return DecodeChunkWrapper(model).eval()


def states_plan(
    graph: IrGraph, layers: int, *, capacity_symbol: str = CAPACITY_SYMBOL
) -> StatesPlan:
    """nodes 順の attention 1 本ずつへ層番号つき k / v スロットを割り当てた手術指定。

    MiniCPM5-1B は KV 共有層を持たない（全 24 層が自前の k / v を計算する）ので、スロットは
    層ごとに 2 本 = 48 本。window は指定しない（全 context の full 形 — sliding は Gemma 4 の話）。

    MUST: 本数が config の層数と食い違えば落とす。attention の取りこぼしは「手術されずに
    mask 形のまま残った層」で、その層だけが chunk 局所の causal になる（数値は動くが
    過去を見ない）沈黙誤値になる。
    """
    outputs = [node.outs[0] for node in graph.nodes if node.op == ATTENTION_OP]
    if len(outputs) != layers:
        raise AssertionError(f"attention が {len(outputs)} 本（{layers} 層と一致しない）")
    return StatesPlan(
        capacity_symbol=capacity_symbol,
        attentions=tuple(
            StateAttentionSpec(
                output=output,
                k_slot=slot_name(layer, "k"),
                v_slot=slot_name(layer, "v"),
            )
            for layer, output in enumerate(outputs)
        ),
    )


def assert_ir_form_decode(
    graph: IrGraph,
    config: Any,
    *,
    seq_symbol: str = SEQ_SYMBOL,
    capacity_symbol: str = CAPACITY_SYMBOL,
) -> dict[str, Any]:
    """states 形 decode グラフの形を検査する（**数値が合ったまま静かに壊れる**性質を全部見る）。

    golden の突合では捕まらない性質だけを並べる:

    - attention が 1 本でも従来形（mask 込み 4 本）で残ると、その層だけ過去を見ない
    - `window` attrs が付くと full 層が sliding として実行される（窓外を切り捨てる）
    - スロットの容量が記号でなく数値だと、context 生成時に容量を選べない（ADR 0066 決定 3）
    - `sym_prefix_slice` / `sin` が残ると、誰も読まない Tmax 定数や畳み残しが配布物に居座る
    - グラフ入力に mask が残ると「ホストが毎 chunk T² を作って渡す」別物になる

    MUST: `config.head_dim` を読む（`hidden_size // num_attention_heads` から導出しない —
    MiniCPM5-1B は 96 ≠ 128 で食い違う）。
    """
    heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    depth = int(config.head_dim)
    layers = int(config.num_hidden_layers)

    names = [spec.name for spec in graph.inputs]
    if names != [INPUT_IDS, POSITION_IDS]:
        raise AssertionError(
            f"グラフ入力が {names} — `{INPUT_IDS}` / `{POSITION_IDS}` の 2 本でない"
            "（mask が畳み込まれずに入力へ残っている可能性）"
        )
    if len(graph.outputs) != 2:
        raise AssertionError(
            f"IR 出力が {len(graph.outputs)} 本（decode 出口は logits / token の 2 本）"
        )
    producer = {out: node for node in graph.nodes for out in node.outs}
    token_source = producer.get(graph.outputs[1])
    if token_source is None or token_source.op != ARGMAX_OP:
        found = "ノード出力でない" if token_source is None else token_source.op
        raise AssertionError(
            f"出力 1 の供給元が {found} — `{ARGMAX_OP}` でない（ADR 0068 決定 4 の decode 出口）"
        )

    attentions = [node for node in graph.nodes if node.op == ATTENTION_OP]
    if len(attentions) != layers:
        raise AssertionError(f"attention が {len(attentions)} 本（{layers} 層と一致しない）")
    for layer, node in enumerate(attentions):
        where = f"attention[{layer}]"
        if len(node.ins) != 3:
            raise AssertionError(
                f"{where}: ins が {len(node.ins)} 本 — states 形は q / k / v の 3 本ちょうど"
                "（4 本なら mask 込みの従来形が残っている）"
            )
        expected_states = {"k": slot_name(layer, "k"), "v": slot_name(layer, "v")}
        if node.states != expected_states:
            raise AssertionError(f"{where}: states 欄が {node.states}（期待 {expected_states}）")
        if "window" in node.attrs:
            raise AssertionError(
                f"{where}: attrs に window={node.attrs['window']} — 全 context の full 形のはず"
            )
        query, key, value = (declared_shape(graph, name) for name in node.ins)
        if query[1] != heads or key[1] != kv_heads or value[1] != kv_heads:
            raise AssertionError(
                f"{where}: head 軸が {[query[1], key[1], value[1]]} —"
                f" 真の GQA 形 {[heads, kv_heads, kv_heads]} でない"
                "（k / v が H まで広がっていれば repeat_kv が実体化している）"
            )
        if query[3] != depth or key[3] != depth or value[3] != depth:
            raise AssertionError(
                f"{where}: D 軸が {[query[3], key[3], value[3]]} — config.head_dim {depth} と違う"
            )

    appends = [node for node in graph.nodes if node.op == STATE_APPEND_OP]
    written = sorted(node.states["slot"] for node in appends)
    expected_slots = sorted(
        slot_name(layer, part) for layer in range(layers) for part in ("k", "v")
    )
    if written != expected_slots:
        raise AssertionError(
            f"`{STATE_APPEND_OP}` の書き先が {len(written)} 本 — 全スロット 1 本ずつでない"
            f"（欠落 {sorted(set(expected_slots) - set(written))} /"
            f" 余剰 {sorted(set(written) - set(expected_slots))}）"
        )
    if sorted(graph.states) != expected_slots:
        raise AssertionError(
            f"states 宣言が {sorted(graph.states)} — 層ごとの k / v {len(expected_slots)} 本でない"
        )
    slot_shape = [1, kv_heads, capacity_symbol, depth]
    for name in expected_slots:
        slot = graph.states[name]
        if slot.dtype != "f32" or list(slot.shape) != slot_shape:
            raise AssertionError(
                f"states['{name}'] が {slot.dtype} {list(slot.shape)} —"
                f" f32 {slot_shape} でない（容量は記号のまま残す MUST）"
            )

    symbols = sorted(graph.symbols)
    if symbols != sorted({seq_symbol, capacity_symbol}):
        raise AssertionError(
            f"symbols が {symbols} — {sorted({seq_symbol, capacity_symbol})} でない"
        )
    residue = sorted(set(graph.required_ops) & set(RESIDUE_OPS))
    if residue:
        raise AssertionError(
            f"手術で死ぬはずの op が残っている: {residue}"
            "（mask の Tmax 畳み込み残骸 / 畳み残した RoPE）"
        )
    return {
        "attention_nodes": len(attentions),
        "state_append_nodes": len(appends),
        "slots": len(expected_slots),
        "slot_shape": slot_shape,
        "heads": [heads, kv_heads, kv_heads],
        "head_dim": depth,
    }


def _write_container(graph: IrGraph, tensors: Mapping[str, torch.Tensor], path: Path) -> IrGraph:
    """手術済みグラフを書いて検証する（公開の 3 段は `pipeline.publish_model` に預ける）。

    `export_to_file` は export → 書き出しが 1 本道で手術を挟む隙間が無いので、書き出し以降
    だけを core の入口から呼ぶ。**規則も原子性も再実装しない** — states 節・順序・shape の
    検査も、shard 分割（ADR 0070 決定 1）とその据え替え・後始末も core が持つ。

    刈り込みで死んだ initializer（mask の Tmax² 定数）は格納テンソルからも落とす —
    `write_model` は宣言と格納の**完全一致**を要求する。
    """
    declared = {init.tensor for init in graph.initializers.values()}
    stored = {name: tensor for name, tensor in tensors.items() if name in declared}
    return publish_model(path, graph, stored)


def _write_io(
    wrapper: nn.Module,
    graph: IrGraph,
    cases: Sequence[tuple[str, torch.Tensor]],
    out_dir: Path,
) -> list[str]:
    """各ケースの**無 pad 全長**の入出力を `io.<case>.safetensors` へ書く。

    NOTE: これは「同じ重み・同じ位置で torch が出す値」の**参照表**であって、ランタイムの
    実行台本ではない。decode 系列のグラフは chunk 形（`M = chunkLength`）で走るので、Deno 側は
    全長 T を chunk へ割って実行し、**有効行だけ**をここの `output.0` / `output.1` の対応行と
    突き合わせる。pad 行で 0 に固定されるのは **states 形 attention の出力だけ**（ADR 0066
    追記 8）で、後段の MLP / lm_head は pad 行にも意味のない値を書く — pad 行のグラフ出力は
    **読んではならない**（参照側に対応物も無い）。
    """
    written: list[str] = []
    for name, ids in cases:
        args = {INPUT_IDS: ids, POSITION_IDS: positions_for(ids)}
        with torch.no_grad():
            outputs = wrapper(args[INPUT_IDS], args[POSITION_IDS])
        tensors = {
            f"{one_shot.INPUT_PREFIX}{declared.name}": normalize_boundary_tensor(
                args[declared.name], f"{name} の入力 '{declared.name}'"
            )
            for declared in graph.inputs
        }
        for slot, output in enumerate(outputs):
            tensors[f"{one_shot.OUTPUT_PREFIX}{slot}"] = normalize_boundary_tensor(
                output.detach().contiguous(), f"{name} の出力 {slot}"
            )
        path = out_dir / f"{one_shot.IO_PREFIX}{name}{one_shot.IO_SUFFIX}"
        save_file(tensors, str(path))
        written.append(path.name)
    return written


def export_series(
    model_dir: Path,
    out_dir: Path,
    *,
    sym_max: int = one_shot.SYM_MAX,
    positions: int = ROPE_TABLE_POSITIONS,
    steps: int = GREEDY_STEPS,
) -> dict[str, Any]:
    """states 形 IR コンテナ・io golden・greedy golden を書き、要約を返す。

    MUST: 生成物は作業席へ書き、**全ての門**（形検査・margin 門・波 A 期待表との sanity）を
    通してから据える。門より前に final へ置くと、落ちた実走が「検収門を通れる資産」を残す
    （据え替えと後片付けの規律は core の原語 {@link karume.artifacts.staged_publication}）。
    """
    wrapper = load_wrapper(model_dir, positions=positions)
    cases = one_shot.build_cases(model_dir, sym_max)
    greedy_cases = tuple(case for case in cases if case[0] in GREEDY_CASES)
    # io は prompt を丸ごと 1 回引くだけ（継続分の位置は要らない）ので steps=0 で見る。
    assert_case_room(cases, 0, positions)
    assert_case_room(greedy_cases, steps, positions)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    with staged_publication(out_dir) as staging:
        # ディレクトリの席は書き手が作る（原語は席を作らない — path しか渡さない）。
        staging.mkdir()
        # 例示入力は最長ケース（記号次元の 0/1 特殊化から遠い）。min=2 は同じ理由、max は
        # mask の Tmax 畳み込みの評価点（手術で刈るので配布物には残らないが、trace は通る）。
        _, example_ids = max(cases, key=lambda case: case[1].shape[1])
        seq = Dim(SEQ_SYMBOL, min=2, max=sym_max)
        print("[export] torch.export → 変換", file=sys.stderr, flush=True)
        graph, tensors = export_module(
            wrapper,
            (example_ids, positions_for(example_ids)),
            dynamic_shapes=({1: seq}, {1: seq}),
            symbol_names=(SEQ_SYMBOL,),
            preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
        )
        config = wrapper.model.config
        print("[export] states 形へ手術 → 書き出し", file=sys.stderr, flush=True)
        surgical = to_states_form(graph, states_plan(graph, int(config.num_hidden_layers)))
        verified = _write_container(surgical, tensors, staging / one_shot.MODEL_FILE)
        form = assert_ir_form_decode(verified, config)

        print("[io] 全長 forward", file=sys.stderr, flush=True)
        io_written = _write_io(wrapper, verified, cases, staging)
        greedy_written, tokens, margins = _write_greedy(
            wrapper, greedy_cases, staging, steps=steps, floor=MARGIN_FLOOR
        )

        # 第 1 継続 token を波 A の期待表と突き合わせる（機構横断の突合 — 1-shot 形と decode 形の
        # 台本は別物なので、同じ重み・同じ prompt で 1 位が一致することが両者の交差検証になる）。
        # MUST: 公開より前に評価する（落ちたら作業席ごと消える — 混成資産を残さない）。
        tokenizer = one_shot.load_tokenizer(model_dir)
        first = {name: continuation[0] for name, continuation in tokens.items()}
        expected = {
            name: token
            for name, token in one_shot.expected_token_ids(tokenizer).items()
            if name in first
        }
        labels = {
            token: tokenizer.id_to_token(token)
            for token in set(first.values()) | set(expected.values())
        }
        sanity = one_shot._sanity(first, expected, labels)
    return {
        "dir": str(out_dir),
        "nodes": len(verified.nodes),
        "outputs": len(verified.outputs),
        "initializers": len(verified.initializers),
        "pruned_initializers": len(graph.initializers) - len(verified.initializers),
        "model_bytes": sum(p.stat().st_size for p in resolve_shards(out_dir / one_shot.MODEL_FILE)),
        "ops": sorted(verified.required_ops),
        "symbols": list(verified.symbols),
        "io": io_written,
        "greedy": greedy_written,
        "greedy_steps": steps,
        "case_lengths": {name: int(ids.shape[1]) for name, ids in cases},
        "form": form,
        "margin_min": {name: min(values) for name, values in sorted(margins.items())},
        "continuation": {
            name: tokenizer.decode(continuation) for name, continuation in sorted(tokens.items())
        },
        "sanity": sanity,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sym-max", type=int, default=one_shot.SYM_MAX)
    parser.add_argument("--positions", type=int, default=ROPE_TABLE_POSITIONS)
    parser.add_argument("--steps", type=int, default=GREEDY_STEPS)
    args = parser.parse_args(argv)
    summary = export_series(
        args.model_dir,
        args.out,
        sym_max=args.sym_max,
        positions=args.positions,
        steps=args.steps,
    )
    print(json.dumps({"model_dir": str(args.model_dir), **summary}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
