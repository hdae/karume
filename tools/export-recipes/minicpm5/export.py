"""実重み MiniCPM5-1B（causal LM）を **1-shot 形**で IR v1 コンテナ + golden io へ書き出す台本。

ADR [0067](../../../docs/decisions/0067-autoregressive-attention-vocabulary.md) 決定 1（GQA を
`attention` の整除 broadcast として受理）を**実モデルで検収する足場**。KV cache は載せない —
`input_ids[1,T]` を丸ごと食って `logits[1,T,vocab]` を返す prefill 相当の 1 本道で、
state スロット（ADR 0066）と decode 経路は後続の波が受け持つ。

    uv run --with 'transformers==5.14.1' python -m minicpm5.export

transformers は **5.14.1 でピン**する（`embeddinggemma/export.py` と同じ理由 — モデリング
コードが変わるとグラフ形が変わる）。pyproject.toml / uv.lock には入れず `--with` で一時的に
足す。

## 何をグラフに載せるか

`LlamaForCausalLM` の forward 丸ごと 1 本（24 層 + `lm_head`）。トークナイズと chat template の
適用はホスト側。出力は `[1, T, 130560]` の生 logits（softmax もサンプリングも載せない）。

MUST: config の **`head_dim` は独立フィールド**（128 — `hidden_size / num_attention_heads` =
1536 / 16 = 96 とは別物）。head_dim を hidden/heads から導出する形は書かない。参照する必要が
あるときは常に `config.head_dim` を読む。

## GQA を「真の形」で出す（決定 1 の検収点）

MiniCPM5-1B は 16:2 の GQA。transformers 5.14.1 の `sdpa_attention_forward` は
`use_gqa_in_sdpa`（integrations/sdpa_attention.py:38）が **`attention_mask is None` を条件に
持つ**ので、加算マスクを渡す本台本の形では必ず偽になり、`repeat_kv` が k / v を
`[1,2,T,128]` → `[1,16,T,128]` へ**実体化してから** SDPA を呼ぶ（同 99-100 行）。実測でも
素の `sdpa` は attention の ins が 3 本とも H=16 になり、GQA が消える。

そこで **transformers の公開拡張点**（`AttentionInterface.register` — 上流コードは 1 行も
差し替えない）へ {@link gqa_sdpa_attention} を登録し、`repeat_kv` を通さず
`enable_gqa=True` で SDPA を呼ぶ。IR の attention は
`q[1,16,T,128] / k[1,2,T,128] / v[1,2,T,128]` になる。

MUST: 出た IR の形は {@link assert_ir_form} が**必ず検査**する。`repeat_kv` 実体化形
（Hkv=16）は「数値は合っているが検収の意味が消えた」資産なので、黙って書かない。

## causal は加算 mask の定数畳み込みで表す

ランタイムの attention に causal 欄は無く、エクスポータは `is_causal=True` を拒否する
（aten_handlers.py の `_h_attention`）。本台本は **`is_causal` が SDPA へ届く経路を構造的に
持たない** — {@link gqa_sdpa_attention} は常に `is_causal=False` を渡し、マスクが `None` なら
（= 非因果に化ける形なら）fail loudly にする。

因果性は {@link additive_causal_mask} が作る加算 f32 `[1,1,T,T]`（0 / −inf）で表す。葉は
`arange` と `full` だけ・T はスライス長にしか現れないので、エクスポータの定数畳み込みが
**Tmax×Tmax の f32 initializer + `sym_prefix_slice`** に落とす（ADR 0010 — EmbeddingGemma の
帯マスクと同じ機構）。4D で渡すので transformers 側の `create_causal_mask` は
`_preprocess_mask_arguments`（masking_utils.py:818）でそのまま返し、`sdpa_mask` の
`is_causal` 省略経路には入らない。結果として **グラフ入力は `input_ids` 1 本だけ**になる。

## 出力レイアウト

    outputs/series/minicpm5-1b/model.safetensors     重み・定数 + __metadata__.karume_ir
    outputs/series/minicpm5-1b/io.<case>.safetensors 入力と torch CPU での期待出力

io のテンソルキー規約は tiny golden / DeBERTa / EmbeddingGemma と同じ
（`input.<グラフ入力名>` / `output.<位置>`）。logits は語彙 130560 なので 1 ケースあたり
`T × 522KB` — golden の T は 128 以下に抑える（{@link GOLDEN_CASES}）。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import nn
from torch.export import Dim

from _shared.paths import INPUTS_ROOT, SERIES_ROOT
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION, normalize_boundary_tensor
from karume.ir import IrGraph
from karume.pipeline import export_to_file
from karume.rope import assert_rope_lifted
from karume.shapes import declared_shape
from karume.shards import resolve_shards

#: 公式重みの置き場（`hf download openbmb/MiniCPM5-1B` の展開先）。
DEFAULT_MODEL_DIR = INPUTS_ROOT / "minicpm5" / "MiniCPM5-1B"

#: 生成物の既定の置き場。格納 dtype は f32 のみ（f16 / i8 / w4 は別系列で決める話）。
DEFAULT_OUT_DIR = SERIES_ROOT / "minicpm5-1b"

MODEL_FILE = "model.safetensors"
IO_PREFIX = "io."
IO_SUFFIX = ".safetensors"
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."

#: グラフ入力の名前（ラッパの forward 引数名）。{@link assert_ir_form} が「これ 1 本だけ」を
#: 見るための綴り — mask や position_ids が入力に増えたら畳み込みが効いていない。
INPUT_IDS = "input_ids"

#: 検査に使う IR op 名。
ATTENTION_OP = "attention"
SYM_PREFIX_SLICE_OP = "sym_prefix_slice"

#: 記号次元 T の上限。causal 定数が Tmax² で焼かれる（512² × 4B = 1MiB）ので、上げるなら
#: その代償を承知の上で上げる（ADR 0010 — 畳み込みの評価点そのもの）。モデルの
#: `max_position_embeddings` は 131072 だが、1-shot 検収に要る長さとは別の話。
SYM_MAX = 512

#: 加算マスクの遮断値。`normalize._additive_attn_mask` が bool マスクを落とす先と同じ値で、
#: 融合 attention の mask 契約（ADR 0023）そのまま。
NEG_INF = float("-inf")

#: `AttentionInterface` へ登録する名前。**`sdpa` / `flash_attention` / `flex_attention` の
#: いずれも部分文字列に含めない** — transformers の `get_correct_attn_implementation` が
#: 名前の部分一致でバックエンド固有の dispatch 検査へ分岐する（modeling_utils.py:2086-2091）。
ATTENTION_NAME = "karume_gqa"

#: 長め（T=87）の英語ケースの本文。末尾が「強く決まる継続」になる段落で、短文ケースと
#: 十分に離れた T で prefix スライスと RoPE 表が実長で効いていることを見る。
CONTEXT_EN = (
    "France is a country in Western Europe with a long written history. Its cities grew along "
    "the rivers that carried grain and stone, and the largest of them sits on the Seine, where "
    "an island in the middle of the river held the first settlement. That city has been the "
    "seat of government since the Middle Ages, and today it holds the parliament, the "
    "ministries, and the residence of the head of state. The capital of France is"
)

#: 中くらい（T=61）の日本語ケースの本文。CJK の byte-level BPE でも同じ形が通ることと、
#: 英語ケースと違う T を踏むためのもの。
CONTEXT_JA = (
    "日本は東アジアの島国で、四つの大きな島と多くの小さな島から成る。国会と中央官庁、"
    "そして皇居が置かれているのは関東平野の都市で、江戸と呼ばれていた時代から政治の"
    "中心だった。日本の首都は"
)

#: golden の固定文（`(ケース名, 本文)`）。トークナイズは `tokenizers` で公式
#: `tokenizer.json` を直接読む（`<s>` は post_processor が付ける — `tokenizer_config.json` の
#: `add_bos_token: false` は「テンプレート側で付ける」の意で、encode の結果は先頭が id 0）。
#: T は 6 / 12 / 61 / 87 に散らす（記号次元の 0/1 特殊化から遠く、上限 512 より十分内側）。
#: 4 ケース合計の logits は 86MB（1 ケース = T × 522KB）。
#:
#: capital-ja は**対構造**（「フランスの首都はパリ、」の前置き）で書く — 素の
#: 「日本の首都は」は greedy 1 位が読点 `、`（「〜は、東京」の頻出形）で期待が立たず、
#: 逆順の「日本の首都は東京、フランスの首都は」は 1 位が中文の `巴黎` になる（2026-08-17
#: 実測 — OpenBMB 系は中文優勢）。この語順だけが `東京` を 1 位で返す。
GOLDEN_CASES: tuple[tuple[str, str], ...] = (
    ("capital-en", "The capital of France is"),
    ("capital-ja", "フランスの首都はパリ、日本の首都は"),
    ("context-en", CONTEXT_EN),
    ("context-ja", CONTEXT_JA),
)

#: 各ケースの**最終位置の greedy トークン**に期待する継続（恒真でない sanity — ADR 0005 の
#: fail loudly）。いずれも「強く決まる継続」で、①重みの取り違え ②mask の向き（未来を見る）
#: ③RoPE 表の位置ずれ ④GQA の head 写像違い のどれかが起きれば最終位置の 1 位が変わる。
#: MUST: 期待継続は**単一トークン**でなければならない（{@link expected_token_ids} が検査）—
#: 複数トークンだと「1 位の一致」の意味が定まらない。
GREEDY_EXPECTATIONS: Mapping[str, str] = {
    "capital-en": " Paris",
    "capital-ja": "東京",
    "context-en": " Paris",
    "context-ja": "東京",
}


def additive_causal_mask(length: int) -> torch.Tensor:
    """`[1,1,length,length]` の加算 causal マスク（許可 0 / 遮断 −inf）を作る。

    MUST: 葉は `arange` と `full` だけで、`length` は **arange の長さ**にしか現れない形で
    書く（`convert.SYMBOL_EXTENT_ARGS` の extent 位置）。値の側に T が入ると Tmax 畳み込みが
    prefix と可換でなくなり、`_check_prefix_commutes` の 2 点評価で落ちる（ADR 0010 追記）。
    MUST: 比較は `cols <= rows`（`le.Tensor`）で書く — `rows >= cols`（`ge.Tensor`）は
    `convert.FOLDABLE_OPS` に無いので畳めず、IR 語彙に無い op として export ごと落ちる。

    NOTE: `length` は export 中は `torch.SymInt`（int として振る舞う）。
    """
    rows = torch.arange(length).unsqueeze(-1)
    cols = torch.arange(length).unsqueeze(0)
    keep = torch.full((), 0.0, dtype=torch.float32)
    drop = torch.full((), NEG_INF, dtype=torch.float32)
    return torch.where(cols <= rows, keep, drop).unsqueeze(0).unsqueeze(0)


def gqa_sdpa_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """`repeat_kv` を通さず `enable_gqa=True` で SDPA を呼ぶ attention 実装（GQA 形の保存）。

    transformers の `AttentionInterface` が要求する呼び出し規約そのまま
    （`(module, q, k, v, attention_mask, dropout=…, scaling=…, **kwargs)` →
    `(attn_output[B,M,H,D], attn_weights)`）。上流の `sdpa_attention_forward` から落とすのは
    ① `repeat_kv` の実体化 ② `is_causal` の自動判定 ③ NPU / paged 系の分岐 の 3 つだけ。

    MUST: `attention_mask is None` は fail loudly。上流はこの形を「`is_causal=True` に任せる」
    合図として使うが、ここで素通しすると**非因果**の attention が黙って出る（IR には causal の
    欄が無いので、後段のどの検査にも掛からない）。
    MUST: 数値の意味を変える kwargs（`is_causal` / `position_bias`）は受理しない — 無視すると
    eager 側と別の数値経路になったことが export でも golden でも見えない。
    """
    if attention_mask is None:
        raise ValueError(
            f"{type(module).__name__}: attn_mask 無しで attention が呼ばれた"
            "（因果性は加算 mask で表す — mask 無しは非因果に化ける）"
        )
    if float(dropout) != 0.0:
        raise ValueError(f"dropout={dropout} の attention は未対応（推論のみ）")
    unsupported = sorted(
        name for name in ("is_causal", "position_bias") if kwargs.get(name) is not None
    )
    if unsupported:
        raise ValueError(f"kwargs {unsupported} を伴う attention は未対応（数値経路が変わる）")
    output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        scale=scaling,
        is_causal=False,
        enable_gqa=True,
    )
    return output.transpose(1, 2).contiguous(), None


def register_attention() -> None:
    """{@link gqa_sdpa_attention} を transformers の attention registry へ登録する（冪等）。

    `AttentionInterface.register` はクラス属性の辞書を更新するので、同じ名前で何度呼んでも
    最後の 1 本が残る（上流の `sdpa` エントリには触らない）。
    """
    from transformers import AttentionInterface

    AttentionInterface.register(ATTENTION_NAME, gqa_sdpa_attention)


class CausalLmWrapper(nn.Module):
    """`LlamaForCausalLM` を 1-shot 形（`input_ids → logits`）に固定した export 用ラッパ。

    MUST: `attention_mask` は**必ず自前の 4D 加算マスク**を渡す。省略すると transformers が
    `is_causal` 任せの経路へ入り、{@link gqa_sdpa_attention} の mask 門で落ちる。
    MUST: `use_cache=False`。True だと `DynamicCache` が生えて KV を返す形になり、1-shot の
    契約（出力 1 本）から外れる。
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = additive_causal_mask(input_ids.shape[1])
        return self.model(input_ids=input_ids, attention_mask=mask, use_cache=False).logits


def load_wrapper(model_dir: Path) -> CausalLmWrapper:
    """実重みを f32 で読み、RoPE バッファを降格した export 可能なラッパを返す。"""
    from transformers import LlamaForCausalLM

    register_attention()
    model = LlamaForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation=ATTENTION_NAME
    )
    model.eval()
    # inv_freq がバッファのままだと定数畳み込みの葉にならず、sin / cos が IR に残る。
    assert_rope_lifted(model, "minicpm5")
    return CausalLmWrapper(model).eval()


def load_tokenizer(model_dir: Path) -> Any:
    """公式 `tokenizer.json` を `tokenizers` で直接読む（transformers の tokenizer 層は通さない）。

    戻りは `tokenizers.Tokenizer`。型を `Any` にしてあるのは遅延 import のため（モジュール
    先頭で import すると台本の import だけで tokenizers が要る）。
    """
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(model_dir / "tokenizer.json"))


def build_cases(model_dir: Path, sym_max: int) -> tuple[tuple[str, torch.Tensor], ...]:
    """golden 4 ケースの `(名前, input_ids)`。長さが記号次元の範囲外なら fail loudly。"""
    tokenizer = load_tokenizer(model_dir)
    cases: list[tuple[str, torch.Tensor]] = []
    for name, text in GOLDEN_CASES:
        ids = torch.tensor([tokenizer.encode(text).ids], dtype=torch.int64)
        length = int(ids.shape[1])
        if not 2 <= length <= sym_max:
            raise ValueError(f"{name}: T={length} が記号次元の範囲 [2, {sym_max}] の外")
        cases.append((name, ids))
    return tuple(cases)


def assert_ir_form(graph: IrGraph, config: Any, sym_max: int) -> dict[str, Any]:
    """IR が「真の GQA 形」かつ「causal が T 非依存の定数」であることを検査する。

    どちらも**数値は合ったまま静かに壊れる**種類の性質で、golden io の突合では捕まらない:

    - `repeat_kv` が実体化すると attention の Hkv が H に化ける（= 決定 1 の検収対象が消える）
    - mask がグラフ入力に残ると「ホストが毎回 T² を作って渡す」形になる（ADR 0010 の趣旨に反し、
      ランタイム側の資産としても別物）

    MUST: `config.head_dim` を読む（`hidden_size // num_attention_heads` から導出しない —
    MiniCPM5-1B は 96 ≠ 128 で食い違う）。
    """
    heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    depth = int(config.head_dim)
    layers = int(config.num_hidden_layers)

    names = [spec.name for spec in graph.inputs]
    if names != [INPUT_IDS]:
        raise AssertionError(
            f"グラフ入力が {names} — `{INPUT_IDS}` 1 本でない"
            "（mask が畳み込まれずに入力へ残っている可能性）"
        )
    if len(graph.outputs) != 1:
        raise AssertionError(f"IR 出力が {len(graph.outputs)} 本（1-shot の logits は 1 本）")

    producer = {out: node for node in graph.nodes for out in node.outs}
    attentions = [node for node in graph.nodes if node.op == ATTENTION_OP]
    if len(attentions) != layers:
        raise AssertionError(f"attention が {len(attentions)} 本（{layers} 層と一致しない）")

    mask_constants: set[str] = set()
    for index, node in enumerate(attentions):
        where = f"attention[{index}]"
        if len(node.ins) != 4:
            raise AssertionError(f"{where}: ins が {len(node.ins)} 本（q / k / v / mask の 4 本）")
        query, key, value, _ = (declared_shape(graph, name) for name in node.ins)
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
        source = producer.get(node.ins[3])
        if source is None or source.op != SYM_PREFIX_SLICE_OP:
            found = (
                "ノード出力でない（グラフ入力か initializer 直結）" if source is None else source.op
            )
            raise AssertionError(
                f"{where}: mask の供給元が {found} — Tmax 定数の {SYM_PREFIX_SLICE_OP} でない"
            )
        constant = source.ins[0]
        if constant not in graph.initializers:
            raise AssertionError(f"{where}: mask の元 '{constant}' が initializer でない")
        shape = declared_shape(graph, constant)
        if shape != [1, 1, sym_max, sym_max]:
            raise AssertionError(
                f"{where}: mask 定数の shape {shape} が Tmax 形 [1, 1, {sym_max}, {sym_max}] と違う"
            )
        mask_constants.add(constant)
    if len(mask_constants) != 1:
        raise AssertionError(
            f"mask 定数が {len(mask_constants)} 本（全層で 1 本を共有するはず）:"
            f" {sorted(mask_constants)}"
        )
    return {
        "attention_nodes": len(attentions),
        "heads": [heads, kv_heads, kv_heads],
        "head_dim": depth,
        "mask_constant": sorted(mask_constants)[0],
    }


def _write_io(
    wrapper: nn.Module,
    graph: IrGraph,
    cases: Sequence[tuple[str, torch.Tensor]],
    out_dir: Path,
) -> tuple[list[str], dict[str, torch.Tensor]]:
    """各ケースの入力と torch CPU 期待出力を `io.<case>.safetensors` へ書く。

    戻り値の 2 本目は sanity 用の logits（`[1, T, vocab]` のまま — 最終位置を引く形が要る）。
    """
    written: list[str] = []
    logits: dict[str, torch.Tensor] = {}
    for name, ids in cases:
        with torch.no_grad():
            output = wrapper(ids)
        args = {INPUT_IDS: ids}
        # MUST: io も IR の意味論 dtype の実表現へ落とす（i64 → i32）。ランタイムが受け取る
        # 形と揃っていないと Deno 側 E2E が golden を読めない（ADR 0009 の境界正規化）。
        tensors = {
            f"{INPUT_PREFIX}{declared.name}": normalize_boundary_tensor(
                args[declared.name], f"{name} の入力 '{declared.name}'"
            )
            for declared in graph.inputs
        }
        tensors[f"{OUTPUT_PREFIX}0"] = normalize_boundary_tensor(
            output.detach().contiguous(), f"{name} の出力 0"
        )
        path = out_dir / f"{IO_PREFIX}{name}{IO_SUFFIX}"
        save_file(tensors, str(path))
        written.append(path.name)
        logits[name] = output.detach()
    return written, logits


def greedy_tokens(logits: Mapping[str, torch.Tensor]) -> dict[str, int]:
    """各ケースの**最終位置**の 1 位トークン id。"""
    return {name: int(value[0, -1].argmax()) for name, value in logits.items()}


def expected_token_ids(tokenizer: Any) -> dict[str, int]:
    """{@link GREEDY_EXPECTATIONS} の期待継続 → トークン id（単一トークンでなければ落とす）。

    `add_special_tokens=False` で引く — 継続の話なので `<s>` は付かない。
    """
    expected: dict[str, int] = {}
    for name, continuation in GREEDY_EXPECTATIONS.items():
        ids = tokenizer.encode(continuation, add_special_tokens=False).ids
        if len(ids) != 1:
            raise AssertionError(
                f"{name}: 期待継続 {continuation!r} が {len(ids)} トークン {ids}"
                "（単一トークンでないと「1 位の一致」の意味が定まらない）"
            )
        expected[name] = int(ids[0])
    return expected


def _sanity(
    greedy: Mapping[str, int], expected: Mapping[str, int], labels: Mapping[int, str]
) -> dict[str, Any]:
    """最終位置の 1 位が期待継続と一致し、かつケース間で定数化していないことを見る。

    MUST: 不一致は落とす。ノルムや形だけの検査では「数値は動いているが言語モデルとして
    壊れている」（層の取り違え・mask の向き・RoPE のずれ）を検出できない。
    MUST: 全ケースの 1 位が同一なら落とす。期待の一致検査と重なるが、こちらは期待表を
    緩めても残る「定数出力」の検出線として独立に置く。
    """
    if set(greedy) != set(expected):
        raise AssertionError(
            f"ケース集合が食い違う: greedy={sorted(greedy)} 期待={sorted(expected)}"
        )

    def show(token: int) -> str:
        return f"{token}({labels.get(token, '?')})"

    wrong = {
        name: (show(token), show(expected[name]))
        for name, token in greedy.items()
        if token != expected[name]
    }
    if wrong:
        raise AssertionError(f"最終位置の 1 位が期待継続と違う（実測, 期待）: {wrong}")
    if len(set(greedy.values())) == 1:
        raise AssertionError(
            f"全ケースの最終位置の 1 位が同一 {show(next(iter(greedy.values())))} — 定数出力"
        )
    return {name: show(token) for name, token in sorted(greedy.items())}


def export_series(model_dir: Path, out_dir: Path, *, sym_max: int = SYM_MAX) -> dict[str, Any]:
    """IR コンテナと golden io を書き、要約を返す。"""
    wrapper = load_wrapper(model_dir)
    cases = build_cases(model_dir, sym_max)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 例示入力は最長ケース（T が上限に近いほど 0/1 特殊化から遠い）。min=2 は 0/1 特殊化を
    # 避けるため、max は Tmax 畳み込みの評価点そのもの（ADR 0010 — 別ノブで二重管理しない）。
    _, example_ids = max(cases, key=lambda case: case[1].shape[1])
    seq = Dim("T", min=2, max=sym_max)
    graph = export_to_file(
        wrapper,
        (example_ids,),
        out_dir / MODEL_FILE,
        dynamic_shapes=({1: seq},),
        preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    )
    form = assert_ir_form(graph, wrapper.model.config, sym_max)
    written, logits = _write_io(wrapper, graph, cases, out_dir)
    tokenizer = load_tokenizer(model_dir)
    greedy = greedy_tokens(logits)
    labels = {token: tokenizer.id_to_token(token) for token in set(greedy.values())}
    expected = expected_token_ids(tokenizer)
    labels.update({token: tokenizer.id_to_token(token) for token in set(expected.values())})
    return {
        "dir": str(out_dir),
        "nodes": len(graph.nodes),
        "outputs": len(graph.outputs),
        "initializers": len(graph.initializers),
        "model_bytes": sum(p.stat().st_size for p in resolve_shards(out_dir / MODEL_FILE)),
        "ops": sorted(graph.required_ops),
        "symbols": list(graph.symbols),
        "io": written,
        "case_lengths": {name: int(ids.shape[1]) for name, ids in cases},
        "form": form,
        "sanity": _sanity(greedy, expected, labels),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sym-max", type=int, default=SYM_MAX)
    args = parser.parse_args(argv)
    summary = export_series(args.model_dir, args.out, sym_max=args.sym_max)
    print(json.dumps({"model_dir": str(args.model_dir), **summary}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
