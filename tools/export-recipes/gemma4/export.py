"""実重み Gemma 4 E2B（text デコーダ）を **1-shot 形**で IR v1 コンテナ + golden io へ書き出す台本。

`minicpm5/export.py` の鏡像（`input_ids[1,T] → logits[1,T,262144]`・KV cache 無し）だが、
検収の主眼は 2 つ増えている:

- **層種別 2 本の帯マスク**（sliding 28 層 / full 7 層）が、どちらも T 非依存の Tmax 定数 +
  `sym_prefix_slice` に畳まれること（ADR 0010 — `embeddinggemma/export.py` の帯マスクと同じ機構を
  **因果**版で 2 種）。
- **混成量子化**（embedding 系 i8 × linear i4 — ADR 0069 決定 5 / `weight_dtype_overrides`）を
  実モデルで通すこと。E2B は語彙 262144 × PLE 表 8960 列という「embedding が重みの過半を占める」
  形なので、i4 一本でも i8 一本でも成り立たない。

    uv run --with 'transformers==5.14.1' python -m gemma4.export

transformers は **5.14.1 でピン**する（`minicpm5/export.py` と同じ理由 — モデリングコードが
変わるとグラフ形が変わる）。pyproject.toml / uv.lock には入れず `--with` で一時的に足す。

## 何をグラフに載せるか

`Gemma4ForCausalLM` の forward 丸ごと 1 本（35 層 + tied `lm_head` + `final_logit_softcapping`）。
公式チェックポイントは multimodal（`Gemma4ForConditionalGeneration`）なので、text 部のキー
`model.language_model.*` を `model.*` へ**付け替えて** text 専用のモデルへ読み込む
（{@link renamed_state}）。vision / audio 塔は読まない。トークナイズと chat template の適用は
ホスト側。

MUST: head_dim は**層種別で違う**（sliding = `config.head_dim` 256 / full =
`config.global_head_dim` 512）。`hidden_size / num_attention_heads` = 192 はどちらとも違うので、
導出形は書かない（{@link assert_ir_form} が層順で 1 本ずつ突合する）。

## GQA / MQA を「真の形」で出す

Gemma 4 E2B は 8:1 の MQA。迂回の理由と方法は `minicpm5/export.py` と同文 —
transformers の公開拡張点（`AttentionInterface.register`）へ {@link gqa_sdpa_attention} を
登録し、`repeat_kv` を通さず `enable_gqa=True` で SDPA を呼ぶ。IR の attention は
`q[1,8,T,D] / k[1,1,T,D] / v[1,1,T,D]` になる。

`Gemma4TextAttention` は attention 実装へ `sliding_window=` を渡してくるが、
{@link gqa_sdpa_attention} は**受理して無視する** — 窓の意味論は下の mask 辞書が正本で、
どの層にどちらの定数が届いているかは {@link assert_ir_form} が層順で検査する。scaling は
上流から 1.0 が来る（q に RMSNorm が掛かっている形なので）ので、そのまま SDPA へ渡す。

## 因果性と窓は加算 mask 2 本の定数畳み込みで表す

`Gemma4TextModel.forward` は `attention_mask` が **dict ならそのまま層へ配る**
（modeling_gemma4.py:1695）。そこで `{"full_attention": causal, "sliding_attention": 帯}` を
自前で作って渡す。どちらも葉は `arange` と `full` だけなので、エクスポータの定数畳み込みが
**Tmax×Tmax の f32 initializer + `sym_prefix_slice`** に落とす（ADR 0010）。結果として
**グラフ入力は `input_ids` 1 本だけ**になる。mask が `None` で届く経路は
{@link gqa_sdpa_attention} が fail loudly にする（非因果に化ける形を残さない）。

## PLE（Per-Layer Embeddings）を 35 分割で持つ

`embed_tokens_per_layer` の 1 枚表（f32 で 9.4GB）は**層別 35 本へ割って**持つ — 割り方も
行ブロック読みもビット一致検査も {@link gemma4.ple} が正本（台本 3 本が同じ 1 本を通す）。
台本側の契約はグラフの呼び方だけ: `stack` で組んだ `[1,T,35,256]` を `per_layer_inputs=` で
上流へ渡す（`input_ids` と `per_layer_inputs` の同時指定は上流が拒否するので、
`inputs_embeds` も自前で引く）。文脈射影と combine（`project_per_layer_inputs`）は
**上流実装をそのまま**通す。

## 混成量子化（この系列の核心）

丸めは**参照・golden の採取より前**（ADR 0006）。順序と対象は {@link quantize_wrapper}:

1. `fake_quant_int8` — embedding 系だけ（主 embedding + 分割 PLE 35 本）。tied な `lm_head` は
   ここで丸めた実体をそのまま使う。
2. `fake_quant_int4` — linear だけ（`lm_head` を除外）。

MUST: 2 つの対象は**排他**（tied `lm_head` を i4 にも通すと二重丸めになり、先に採った i8 の
scale 台帳が実値と食い違う）。格納は既定 `i8` + linear を 1 本ずつ `i4` へ明示指定する形にする
— 逆（既定 i4 + embedding を明示 i8）にすると、tied 実体に export が付ける FQN
（`model.lm_head.weight` / `model.model.embed_tokens.weight` のどちらか）を**書く前に**
知っている必要が出る。i4 側のキーは fake-quant の scale 台帳そのものなので、指定と丸めが
同じ 1 箇所から出る。

量子化しないもの（f32 のまま）: 全 norm weight・`layer_scalar`・RoPE 由来の定数・mask 定数。
いずれも重みスロットで消費されないので圧縮格納の適格集合の外（`emit.eligible_compressed_initializers`）。

## 出力レイアウト

    outputs/series/gemma4-e2b/model.safetensors     重み・定数 + __metadata__.karume_ir
    outputs/series/gemma4-e2b/io.<case>.safetensors 入力と torch CPU での期待出力

io のテンソルキー規約は tiny golden / DeBERTa / EmbeddingGemma / MiniCPM5 と同じ
（`input.<グラフ入力名>` / `output.<位置>`）。logits は語彙 262144 なので 1 ケースあたり
`T × 1.05MB`（`context-en` は T=598 で 627MB）— ディスクは潤沢なので全行を書く。

## BOS はこちらで付ける

`tokenizer.json` の post_processor は特殊トークンを**足さない**（`TemplateProcessing` の
`special_tokens` が空）。`tokenizer_config.json` にも `add_bos_token` は無く、`<bos>` を置くのは
`chat_template.jinja`（188 行目の `{{- bos_token -}}`）— つまり BOS はホストの仕事。
本台本は chat template を使わない素の継続を golden にするので、{@link build_cases} が
`<bos>` を 1 個だけ先頭に付ける。
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from torch.export import Dim

from _shared.paths import INPUTS_ROOT, SERIES_ROOT
from gemma4 import ple, rope
from karume.artifacts import staged_publication
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION, normalize_boundary_tensor
from karume.ir import IrGraph
from karume.pipeline import export_to_file
from karume.quantize import Int4Report, Int8Report, fake_quant_int4, fake_quant_int8
from karume.rope import assert_rope_lifted
from karume.shapes import declared_shape
from karume.shards import resolve_shards

#: 公式重みの置き場（`hf download google/gemma-4-E2B-it` の展開先）。
DEFAULT_MODEL_DIR = INPUTS_ROOT / "gemma4" / "gemma-4-E2B-it"

#: 生成物の既定の置き場。
DEFAULT_OUT_DIR = SERIES_ROOT / "gemma4-e2b"

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

#: 記号次元 T の上限。mask 定数が **層種別 2 本**それぞれ Tmax² で焼かれる
#: （768² × 4B × 2 = 4.5MiB）ので、上げるならその代償を承知の上で上げる。512 より上でないと
#: 帯マスクが causal と一致してしまい sliding 層が無検証になるため、`sliding_window`（512）
#: より確実に大きい点に置く。
SYM_MAX = 768

#: 加算マスクの遮断値（`normalize._additive_attn_mask` が bool マスクを落とす先と同じ値）。
NEG_INF = float("-inf")

#: `AttentionInterface` へ登録する名前。**`sdpa` / `flash_attention` / `flex_attention` の
#: いずれも部分文字列に含めない**（`minicpm5/export.py` と同じ理由 — 名前の部分一致で
#: バックエンド固有の dispatch 検査へ分岐する）。
ATTENTION_NAME = "karume_gqa"

#: `config.layer_types` に現れる層種別（= mask 辞書のキー = RoPE の宣言のキー）。綴りの正本は
#: {@link gemma4.rope}（配布 recipe も同じ語彙を読むので、torch を要さない側に置いてある）。
FULL_ATTENTION = rope.FULL_ATTENTION
SLIDING_ATTENTION = rope.SLIDING_ATTENTION

#: multimodal チェックポイントでの text 部のキー接頭辞と、text 専用モデルでの接頭辞。
CHECKPOINT_TEXT_PREFIX = "model.language_model."
MODEL_TEXT_PREFIX = "model."

#: PLE の 1 枚表（付け替え**前**のチェックポイントキー）。
PLE_CHECKPOINT_KEY = f"{CHECKPOINT_TEXT_PREFIX}embed_tokens_per_layer.weight"

#: 付け替え**後**の主 embedding / tied lm_head の state_dict キー。
EMBED_TOKENS_KEY = f"{MODEL_TEXT_PREFIX}embed_tokens.weight"
LM_HEAD_KEY = "lm_head.weight"

#: ラッパ基準のモジュール FQN（fake-quant の `include` と scale 台帳のキーが載る空間）。
PER_LAYER_PREFIX = "per_layer."
EMBED_TOKENS_MODULE = "model.model.embed_tokens"
LM_HEAD_MODULE = "model.lm_head"

#: 長め（T=598）の英語ケースの本文。**T > 512（`sliding_window`）** を踏むためのもので、
#: これが無いと帯マスクが causal と一致してしまい sliding 層の窓が一度も評価されない。
#:
#: 末尾は "… the city called"（継続 = `▁Paris`）。素の "The capital of France is" で終える形は
#: この長さの叙述文だと bf16 標準経路の実測でも 1 位が `▁the`（"the city of Paris" 系の継続）
#: になり、期待が立たない — 末尾 3 候補の実測（2026-08-19・margin 12.8 / **13.9** / 0.1）で
#: 最良のこの形に固定した。短文ケースの実績（`▁Paris` が 1 位）は長文へ外挿できない。
CONTEXT_EN = (
    "France is a country in Western Europe whose written history reaches back to the Roman "
    "province of Gaul. Its territory is shaped by four great river basins, and the towns that "
    "grew along those rivers became the markets, the bishoprics and eventually the "
    "administrative centres of the kingdom. The Loire carried stone and wine, the Rhone linked "
    "the Mediterranean ports to the interior, the Garonne drained the south west toward the "
    "Atlantic, and the Seine ran north west through a wide basin of wheat and pasture before "
    "reaching the sea at Le Havre. On an island in the middle of the Seine a Gaulish people "
    "called the Parisii kept a settlement that the Romans fortified and renamed Lutetia. "
    "The island was small enough to defend and the crossing was easy enough to tax, which is "
    "the combination that turns a river ford into a city. "
    "Over the following centuries the settlement spread from the island to both banks. The "
    "Capetian kings made it the seat of their household in the tenth century, and from then on "
    "the royal archives, the mint, the courts of justice and the university grew up around the "
    "same few streets. When the monarchy fell, the revolutionary assemblies met there; when the "
    "republic was rebuilt after each of its interruptions, the new institutions were installed "
    "in the same buildings. The result is a degree of centralisation that has no real parallel "
    "among comparable European states: the national library, the supreme administrative court, "
    "the central bank, the principal railway hub and the residence of the head of state are all "
    "within a few kilometres of one another. "
    "Administrative reform has repeatedly tried to loosen this concentration. Regional councils "
    "were given budgets and elected assemblies, several ministries moved departments to "
    "provincial cities, and the high speed rail network was built partly to make those cities "
    "reachable in an afternoon. The concentration nevertheless persists, because the courts, "
    "the ministries and the parliament did not move, and the private institutions that depend on "
    "them stayed where they were. A company that needs a permit, a broadcaster that needs an "
    "audience and a publisher that needs an editor all end up in the same place. "
    "The same pull is visible in the transport map. The motorways radiate outward like the "
    "spokes of a wheel and are numbered from the centre, the high speed lines all begin at one "
    "of six terminal stations, and a traveller going from one provincial city to another often "
    "finds that the quickest route runs back through the middle of the country and out again. "
    "Airline schedules follow the same shape, and so do the freight corridors that feed the "
    "ports of Le Havre and Marseille. Planners have described the pattern for a century as a "
    "wheel with one hub, and every attempt to add a second hub has so far produced a spoke. "
    "For a reader who has followed the geography, the political history and the administrative "
    "reforms above, the conclusion is not in doubt. The parliament meets there, the ministries "
    "are quartered there, the head of state lives there, and the Seine still runs past the island "
    "where the first settlement stood. The capital of France is the city called"
)

#: golden の固定文（`(ケース名, 本文)`）。T は `<bos>` 込みで 6 / 10 / 598 に散らす
#: （0/1 特殊化から遠く、`sliding_window` 512 の両側に跨り、上限 768 より内側）。
#: `capital-ja` は minicpm5 と同じ**対構造**（「フランスの首都はパリ、」の前置き）で、
#: 素の「日本の首都は」より継続が決まりやすい形にしてある。
GOLDEN_CASES: tuple[tuple[str, str], ...] = (
    ("capital-en", "The capital of France is"),
    ("capital-ja", "フランスの首都はパリ、日本の首都は"),
    ("context-en", CONTEXT_EN),
)

#: 各ケースの**最終位置の greedy トークン**に期待する継続（恒真でない sanity — ADR 0005 の
#: fail loudly）。いずれも「強く決まる継続」で、①重みの取り違え ②mask の向き（未来を見る）
#: ③RoPE の位置ずれ ④PLE の層割り付け違い のどれかが起きれば最終位置の 1 位が変わる。
#: MUST: 期待継続は**単一トークン**でなければならない（{@link expected_token_ids} が検査）。
GREEDY_EXPECTATIONS: Mapping[str, str] = {
    "capital-en": " Paris",
    "capital-ja": "東京",
    "context-en": " Paris",
}


def additive_causal_mask(length: int) -> torch.Tensor:
    """`[1,1,length,length]` の加算 causal マスク（許可 0 / 遮断 −inf）を作る。

    MUST: 葉は `arange` と `full` だけで、`length` は **arange の長さ**にしか現れない形で
    書く（`convert.SYMBOL_EXTENT_ARGS` の extent 位置）。値の側に T が入ると Tmax 畳み込みが
    prefix と可換でなくなる（ADR 0010 追記）。
    MUST: 比較は `cols <= rows`（`le.Tensor`）で書く — `rows >= cols`（`ge.Tensor`）は
    `convert.FOLDABLE_OPS` に無いので畳めず、IR 語彙に無い op として export ごと落ちる。

    NOTE: `length` は export 中は `torch.SymInt`（int として振る舞う）。
    """
    rows = torch.arange(length).unsqueeze(-1)
    cols = torch.arange(length).unsqueeze(0)
    keep = torch.full((), 0.0, dtype=torch.float32)
    drop = torch.full((), NEG_INF, dtype=torch.float32)
    return torch.where(cols <= rows, keep, drop).unsqueeze(0).unsqueeze(0)


def additive_sliding_mask(length: int, window: int) -> torch.Tensor:
    """`[1,1,length,length]` の加算 **帯** マスク（因果 かつ 距離 < window）を作る。

    上流の `sliding_window_mask_function`（modeling_gemma4.py:1915）と同値 —
    左窓は `0 <= dist < window`（self を含む）、右窓は因果なので無い。窓は 512 なので
    `T <= 512` では causal と一致する（golden に T > 512 のケースを 1 本置く理由）。

    MUST: {@link additive_causal_mask} と同じ制約（葉は arange / full だけ・`length` は
    extent 位置だけ）に加えて、距離判定は `sub` + `lt.Scalar` で書く
    （`convert.FOLDABLE_OPS` に載る形 — 載らない比較を使うと畳めずに export ごと落ちる）。
    """
    rows = torch.arange(length).unsqueeze(-1)
    cols = torch.arange(length).unsqueeze(0)
    keep = torch.full((), 0.0, dtype=torch.float32)
    drop = torch.full((), NEG_INF, dtype=torch.float32)
    within = (cols <= rows) & ((rows - cols) < window)
    return torch.where(within, keep, drop).unsqueeze(0).unsqueeze(0)


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
    """`repeat_kv` を通さず `enable_gqa=True` で SDPA を呼ぶ attention 実装（MQA 形の保存）。

    transformers の `AttentionInterface` が要求する呼び出し規約そのまま
    （`(module, q, k, v, attention_mask, dropout=…, scaling=…, **kwargs)` →
    `(attn_output[B,M,H,D], attn_weights)`）。上流の `sdpa_attention_forward` から落とすのは
    ① `repeat_kv` の実体化 ② `is_causal` の自動判定 ③ NPU / paged 系の分岐 の 3 つだけ。

    MUST: `attention_mask is None` は fail loudly。上流はこの形を「`is_causal=True` に任せる」
    合図として使うが、ここで素通しすると**非因果**の attention が黙って出る（IR には causal の
    欄が無いので、後段のどの検査にも掛からない）。
    MUST: 数値の意味を変える kwargs（`is_causal` / `position_bias`）は受理しない。
    NOTE: `sliding_window` は**受理して無視する** — 窓は呼び出し側が渡した mask 定数が正本で、
    層ごとの割り当ては {@link assert_ir_form} が層順に検査する。ここで窓を再解釈すると
    「mask 辞書と kwargs のどちらが効いているか」が 2 箇所に分かれる。
    """
    if attention_mask is None:
        raise ValueError(
            f"{type(module).__name__}: attn_mask 無しで attention が呼ばれた"
            "（因果性と窓は加算 mask で表す — mask 無しは非因果に化ける）"
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
    """{@link gqa_sdpa_attention} を transformers の attention registry へ登録する（冪等）。"""
    from transformers import AttentionInterface

    AttentionInterface.register(ATTENTION_NAME, gqa_sdpa_attention)


class Gemma4Wrapper(nn.Module):
    """`Gemma4ForCausalLM` を 1-shot 形（`input_ids → logits`）に固定した export 用ラッパ。

    MUST: `attention_mask` は**必ず層種別 2 本の辞書**を渡す。省略すると transformers 側で
    mask が組まれ（`create_causal_mask` / `create_sliding_window_causal_mask`）、
    `cache_position` 由来の値が畳み込みの葉に混ざる。
    MUST: `input_ids` ではなく `inputs_embeds` + `per_layer_inputs` で呼ぶ — 上流は
    `input_ids` と `per_layer_inputs` の同時指定を拒否する（modeling_gemma4.py:1675）。
    MUST: `use_cache=False`。True だと `DynamicCache` が生えて 1-shot の契約から外れる。
    """

    def __init__(self, model: nn.Module, tables: nn.ModuleList) -> None:
        super().__init__()
        self.model = model
        self.per_layer = tables
        self.per_layer_scale = ple.per_layer_scale(model.config)
        self.sliding_window = int(model.config.sliding_window)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        length = input_ids.shape[1]
        mask = {
            FULL_ATTENTION: additive_causal_mask(length),
            SLIDING_ATTENTION: additive_sliding_mask(length, self.sliding_window),
        }
        embeds = self.model.model.embed_tokens(input_ids)
        stacked = ple.per_layer_inputs(self.per_layer, input_ids, self.per_layer_scale)
        return self.model(
            inputs_embeds=embeds,
            per_layer_inputs=stacked,
            attention_mask=mask,
            use_cache=False,
        ).logits


def is_int8_module(name: str) -> bool:
    """i8 で丸める embedding 系モジュールか（主 embedding + 分割 PLE 35 本）。

    `lm_head` は `nn.Linear` なのでここには入れない — tied なので主 embedding を丸めれば
    実体が丸まる。{@link is_int4_module} と**排他**であることは tests が固定する。
    """
    return name == EMBED_TOKENS_MODULE or name.startswith(PER_LAYER_PREFIX)


def is_int4_module(name: str) -> bool:
    """i4 で丸める linear か（tied `lm_head` だけを外す）。

    `fake_quant_int4` は `nn.Linear` 以外を見ないので、ここで落とすのは `lm_head` だけで足りる
    （二重丸めの禁止 — `quantize.fake_quant_int4` の docstring）。
    """
    return name != LM_HEAD_MODULE


def load_text_config(model_dir: Path) -> Any:
    """multimodal config から text 部（`Gemma4TextConfig`）を取り出す。"""
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_dir).get_text_config()


def renamed_state(
    model: nn.Module, model_file: Path, probe: Sequence[int]
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """text 部のキーを付け替えた f32 の state_dict と、捨てたキーの一覧を返す。

    3 つの付け替え / 差し替えをする:

    - `model.language_model.*` → `model.*`（vision / audio 塔のキーは読まない）。
    - KV 共有層に残っている `k_proj` / `v_proj` / `k_norm` / `v_norm` の残骸を捨てる。
      捨てる対象は上流が持つ `_keys_to_ignore_on_load_unexpected`（`Gemma4TextModel.__init__`
      が層構成から組む）を**そのまま**使う — 共有の規則をここで書き直すと、上流が層の割り方を
      変えたときに 2 箇所が独立に動く。
    - PLE の 1 枚表は probe の {@link gemma4.ple.PLE_PROBE_ROWS} 行
      （{@link gemma4.ple.probe_rows} の散点）だけを検査席へ載せる（本体は
      {@link gemma4.ple.load_per_layer_tables} が持つ 35 分割）。

    tied な `lm_head.weight` は state_dict に**同じテンソルを 2 度**載せる — チェックポイントに
    実体が無く（`_tied_weights_keys`）、載せないと `load_state_dict` の missing に出る。
    """
    ignored = tuple(str(pattern) for pattern in model._keys_to_ignore_on_load_unexpected)
    state: dict[str, torch.Tensor] = {}
    dropped: list[str] = []
    with safe_open(str(model_file), framework="pt") as handle:
        for key in sorted(handle.keys()):
            if not key.startswith(CHECKPOINT_TEXT_PREFIX):
                continue
            name = MODEL_TEXT_PREFIX + key[len(CHECKPOINT_TEXT_PREFIX) :]
            if any(re.search(pattern, name) for pattern in ignored):
                dropped.append(name)
                continue
            if key == PLE_CHECKPOINT_KEY:
                sliced = handle.get_slice(key)
                rows = torch.cat([sliced[row : row + 1, :] for row in probe])
                state[name] = rows.to(torch.float32)
                continue
            state[name] = handle.get_tensor(key).to(torch.float32)
    if EMBED_TOKENS_KEY not in state:
        raise ValueError(f"チェックポイントに '{CHECKPOINT_TEXT_PREFIX}embed_tokens.weight' が無い")
    state[LM_HEAD_KEY] = state[EMBED_TOKENS_KEY]
    return state, dropped


def build_wrapper(model: nn.Module, tables: nn.ModuleList) -> Gemma4Wrapper:
    """検査席の PLE 表を落として export 用ラッパを組む。

    MUST: `embed_tokens_per_layer` を落とす — 分割 35 本が PLE の唯一の正本になる形にして
    おかないと、量子化の `include` 述語（{@link is_int8_module} / {@link is_int4_module}）が
    「どちらにも当たらない embedding」を残すことになり、対象の網羅性が言えなくなる。
    forward は元から参照していないので、落としてもグラフは変わらない。
    """
    del model.model.embed_tokens_per_layer
    return Gemma4Wrapper(model, tables).eval()


def load_model_and_tables(model_dir: Path) -> tuple[nn.Module, nn.ModuleList]:
    """実重みを f32 で読んだ text モデルと PLE の 35 分割を返す（検査席の表は載ったまま）。

    ラッパの組み立て（{@link build_wrapper}）と RoPE バッファの降格を**含まない**のは、
    decode 台本（{@link gemma4.export_decode}）が同じ素材を別のラッパ形・別の RoPE 形
    （ホスト供給の受け渡し口への差し替え）で使うから。素材の読み方と 3 つの等価検査（PLE 表の行数・
    KV 共有層の残骸落とし・35 分割のビット一致）は 1 箇所に閉じる。
    """
    from transformers import Gemma4ForCausalLM

    register_attention()
    config = load_text_config(model_dir)
    model_file = model_dir / MODEL_FILE
    rows, _ = ple.per_layer_table_shape(model_file, PLE_CHECKPOINT_KEY)
    if rows != int(config.vocab_size_per_layer_input):
        raise ValueError(
            f"PLE 表の行 {rows} が config の vocab_size_per_layer_input"
            f" {config.vocab_size_per_layer_input} と違う"
        )
    # 分割の等価検査に使う行はブロック境界・両端・中央の散点（{@link gemma4.ple.probe_rows}）。
    probe = ple.probe_rows(rows)
    config.vocab_size_per_layer_input = ple.PLE_PROBE_ROWS
    config._attn_implementation = ATTENTION_NAME

    model = Gemma4ForCausalLM(config).eval()
    state, dropped = renamed_state(model, model_file, probe)
    if not dropped:
        raise ValueError(
            "KV 共有層の k/v 残骸が 1 本も捨てられなかった"
            "（上流の _keys_to_ignore_on_load_unexpected が空 — 層構成の前提が変わった可能性）"
        )
    model.load_state_dict(state)
    del state
    tables = ple.load_per_layer_tables(
        model_file,
        PLE_CHECKPOINT_KEY,
        int(config.num_hidden_layers),
        int(config.hidden_size_per_layer_input),
    )
    ple.assert_per_layer_split(model, tables, probe)
    return model, tables


def load_wrapper(model_dir: Path) -> Gemma4Wrapper:
    """実重みを f32 で読み、RoPE バッファを降格した export 可能なラッパを返す。"""
    model, tables = load_model_and_tables(model_dir)
    # inv_freq がバッファのままだと定数畳み込みの葉にならず、sin / cos が IR に残る。
    assert_rope_lifted(model, "gemma4")
    return build_wrapper(model, tables)


def quantize_wrapper(wrapper: nn.Module) -> tuple[Int8Report, Int4Report, dict[str, torch.Tensor]]:
    """embedding 系を i8・linear を i4 で丸め、合流した scale 台帳を返す（ADR 0006 の順序 MUST）。

    台帳には tied 実体のキーを**両方の綴りで**載せる。torch.export は tied な
    `model.model.embed_tokens.weight` / `model.lm_head.weight` を 1 本の initializer に畳むが、
    どちらの FQN が残るかは export の内部順序で決まる（実測は `model.lm_head.weight`）。
    余分なキーは `emit` が引かないだけなので、両方載せておけばどちらでも通る。
    """
    int8 = fake_quant_int8(wrapper, include=is_int8_module)
    int4 = fake_quant_int4(wrapper, include=is_int4_module)
    scales: dict[str, torch.Tensor] = {**int8.scales, **int4.scales}
    scales[f"{LM_HEAD_MODULE}.weight"] = int8.scales[f"{EMBED_TOKENS_MODULE}.weight"]
    return int8, int4, scales


def load_tokenizer(model_dir: Path) -> Any:
    """公式 `tokenizer.json` を `tokenizers` で直接読む（transformers の層は通さない）。"""
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(model_dir / "tokenizer.json"))


def bos_token_id(model_dir: Path, tokenizer: Any) -> int:
    """`tokenizer_config.json` の `bos_token` を id へ引く（モジュール docstring の BOS 規約）。"""
    config = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    token = config["bos_token"]
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"tokenizer に bos_token {token!r} が無い")
    return int(token_id)


def assert_case_lengths(
    cases: Sequence[tuple[str, torch.Tensor]], sym_max: int, window: int
) -> None:
    """golden の T が記号次元に収まり、かつ**窓より長いケースが 1 本以上**あることを見る。

    MUST: `T <= window` しか無い golden は帯マスクが causal と一致するので、sliding 層の窓が
    一度も評価されない（数値は合うのに検収の意味が消える形 — この系列を作った理由の半分）。
    """
    for name, ids in cases:
        length = int(ids.shape[1])
        if not 2 <= length <= sym_max:
            raise ValueError(f"{name}: T={length} が記号次元の範囲 [2, {sym_max}] の外")
    longest = max((int(ids.shape[1]) for _, ids in cases), default=0)
    if longest <= window:
        raise ValueError(
            f"最長ケースの T={longest} が sliding_window {window} を超えない"
            "（帯マスクが causal と一致してしまい sliding 層が無検証になる）"
        )


def build_cases(model_dir: Path, sym_max: int, window: int) -> tuple[tuple[str, torch.Tensor], ...]:
    """golden ケースの `(名前, input_ids)`（`<bos>` はここで足す — モジュール docstring）。"""
    tokenizer = load_tokenizer(model_dir)
    bos = bos_token_id(model_dir, tokenizer)
    cases = tuple(
        (
            name,
            torch.tensor(
                [[bos, *tokenizer.encode(text, add_special_tokens=False).ids]], dtype=torch.int64
            ),
        )
        for name, text in GOLDEN_CASES
    )
    assert_case_lengths(cases, sym_max, window)
    return cases


def _attention_depth(config: Any, layer_type: str) -> int:
    """層種別ごとの head_dim（sliding = `head_dim` / full = `global_head_dim`）。"""
    if layer_type == SLIDING_ATTENTION:
        return int(config.head_dim)
    if layer_type == FULL_ATTENTION:
        return int(config.global_head_dim)
    raise AssertionError(f"未知の layer_type '{layer_type}'（mask 辞書のキーと対応しない）")


def _mask_constant(graph: IrGraph, node: Any, where: str, sym_max: int) -> str:
    """attention の mask 入力を辿って Tmax 定数の名前を返す（畳み込みが効いていなければ落とす）。"""
    producer = {out: item for item in graph.nodes for out in item.outs}
    source = producer.get(node.ins[3])
    if source is None or source.op != SYM_PREFIX_SLICE_OP:
        found = "ノード出力でない（グラフ入力か initializer 直結）" if source is None else source.op
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
    return constant


def assert_ir_form(
    graph: IrGraph, config: Any, sym_max: int, storage_expectation: Mapping[str, int]
) -> dict[str, Any]:
    """IR が「真の MQA 形」「層種別の mask が T 非依存の定数」「混成格納が効いた形」かを検査する。

    どれも**数値は合ったまま静かに壊れる**種類の性質で、golden io の突合では捕まらない:

    - `repeat_kv` が実体化すると attention の Hkv が H に化ける
    - mask がグラフ入力に残ると「ホストが毎回 T² を作って渡す」形になる（ADR 0010 の趣旨に反する）
    - sliding 層に causal 定数（またはその逆）が届いていても、T ≤ 512 の golden では同じ数が出る
    - 圧縮の適格判定を外した重みは**黙って f32 のまま**残る（`emit._plan_weight_dtype` の
      既定側は静かに落とす経路を持つ）ので、格納 dtype の本数を数えないと気づけない

    MUST: head_dim は層種別で引く（{@link _attention_depth}）— 片方の値で全層を見ると、
    もう片方の層が丸ごと無検査になる。
    """
    heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    layer_types = list(config.layer_types)
    layers = int(config.num_hidden_layers)
    if len(layer_types) != layers:
        raise AssertionError(f"config.layer_types が {len(layer_types)} 本（{layers} 層と違う）")

    names = [spec.name for spec in graph.inputs]
    if names != [INPUT_IDS]:
        raise AssertionError(
            f"グラフ入力が {names} — `{INPUT_IDS}` 1 本でない"
            "（mask が畳み込まれずに入力へ残っている可能性）"
        )
    if len(graph.outputs) != 1:
        raise AssertionError(f"IR 出力が {len(graph.outputs)} 本（1-shot の logits は 1 本）")

    attentions = [node for node in graph.nodes if node.op == ATTENTION_OP]
    if len(attentions) != layers:
        raise AssertionError(f"attention が {len(attentions)} 本（{layers} 層と一致しない）")

    constants: dict[str, set[str]] = {layer_type: set() for layer_type in layer_types}
    for index, node in enumerate(attentions):
        layer_type = layer_types[index]
        where = f"attention[{index}] ({layer_type})"
        if len(node.ins) != 4:
            raise AssertionError(f"{where}: ins が {len(node.ins)} 本（q / k / v / mask の 4 本）")
        query, key, value, _ = (declared_shape(graph, name) for name in node.ins)
        if query[1] != heads or key[1] != kv_heads or value[1] != kv_heads:
            raise AssertionError(
                f"{where}: head 軸が {[query[1], key[1], value[1]]} —"
                f" 真の GQA 形 {[heads, kv_heads, kv_heads]} でない"
                "（k / v が H まで広がっていれば repeat_kv が実体化している）"
            )
        depth = _attention_depth(config, layer_type)
        if query[3] != depth or key[3] != depth or value[3] != depth:
            raise AssertionError(
                f"{where}: D 軸が {[query[3], key[3], value[3]]} — この層種別の head_dim"
                f" {depth} と違う"
            )
        constants[layer_type].add(_mask_constant(graph, node, where, sym_max))

    for layer_type, found in constants.items():
        if len(found) != 1:
            raise AssertionError(
                f"{layer_type} の mask 定数が {len(found)} 本（層種別ごとに 1 本を共有するはず）:"
                f" {sorted(found)}"
            )
    shared = {name for found in constants.values() for name in found}
    if len(shared) != len(constants):
        raise AssertionError(
            f"層種別 {sorted(constants)} が mask 定数 {sorted(shared)} を共有している"
            "（causal と帯が同じ定数に畳まれている = 窓が効いていない）"
        )

    storage: dict[str, int] = {}
    for initializer in graph.initializers.values():
        dtype = initializer.storage.dtype
        storage[dtype] = storage.get(dtype, 0) + 1
    wrong = {
        dtype: (storage.get(dtype, 0), expected)
        for dtype, expected in storage_expectation.items()
        if storage.get(dtype, 0) != expected
    }
    if wrong:
        raise AssertionError(
            f"格納 dtype の本数が想定と違う（実測, 想定）: {wrong} / 全内訳 {storage}"
        )
    return {
        "attention_nodes": len(attentions),
        "heads": [heads, kv_heads, kv_heads],
        "head_dim": {layer_type: _attention_depth(config, layer_type) for layer_type in constants},
        "mask_constants": {layer_type: sorted(found)[0] for layer_type, found in constants.items()},
        "storage": dict(sorted(storage.items())),
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
    """{@link GREEDY_EXPECTATIONS} の期待継続 → トークン id（単一トークンでなければ落とす）。"""
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
    壊れている」（層の取り違え・mask の向き・RoPE のずれ・PLE の層違い）を検出できない。
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
    """IR コンテナと golden io を書き、要約を返す。

    MUST: 生成物は作業席へ書き、**全ての門**（形検査・sanity）を通してから据える。門より前に
    final へ置くと、落ちた実走が「検収門を通れる資産」を残す — io golden は同じ壊れたラッパから
    採るので互いに整合し、TS 側の突合は**緑になる**（「いつ公開してよいか」の綴りは
    {@link _shared.decode_series._publish}・据え替えと後片付けの規律は core の原語
    {@link karume.artifacts.staged_publication}）。
    """
    wrapper = load_wrapper(model_dir)
    # MUST: 丸めは参照・golden の採取より前（ADR 0006）— 後だと参照だけが元の重みで動く。
    int8, int4, scales = quantize_wrapper(wrapper)
    cases = build_cases(model_dir, sym_max, wrapper.sliding_window)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    # 例示入力は最長ケース（T が上限に近いほど 0/1 特殊化から遠い）。min=2 は 0/1 特殊化を
    # 避けるため、max は Tmax 畳み込みの評価点そのもの（ADR 0010 — 別ノブで二重管理しない）。
    _, example_ids = max(cases, key=lambda case: case[1].shape[1])
    seq = Dim("T", min=2, max=sym_max)
    with staged_publication(out_dir) as staged:
        # ディレクトリの席は書き手が作る（原語は席を作らない — path しか渡さない）。
        staged.mkdir()
        graph = export_to_file(
            wrapper,
            (example_ids,),
            staged / MODEL_FILE,
            dynamic_shapes=({1: seq},),
            preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
            weight_dtype="i8",
            weight_scales=scales,
            weight_dtype_overrides=dict.fromkeys(int4.scales, "i4"),
        )
        form = assert_ir_form(
            graph,
            wrapper.model.config,
            sym_max,
            {"i8": len(int8.scales), "i4": len(int4.scales)},
        )
        written, logits = _write_io(wrapper, graph, cases, staged)
        tokenizer = load_tokenizer(model_dir)
        greedy = greedy_tokens(logits)
        labels = {token: tokenizer.id_to_token(token) for token in set(greedy.values())}
        expected = expected_token_ids(tokenizer)
        labels.update({token: tokenizer.id_to_token(token) for token in set(expected.values())})
        # MUST: 公開より前に評価する（この系列で唯一の非恒真な検査 — 落ちたら席ごと消える）。
        sanity = _sanity(greedy, expected, labels)
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
        "quantized": {"i8": int8.describe(), "i4": int4.describe()},
        "form": form,
        "sanity": sanity,
    }


def series_parser(description: str, out_dir: Path) -> argparse.ArgumentParser:
    """この family の 3 台本に共通な CLI の骨組み（`--model-dir` / `--out` / `--sym-max`）。

    系列で違うのは既定の出力先と、chunk 系列だけが足す `--positions` / `--steps` だけ
    （{@link gemma4.export_decode.run_variant_cli}）。3 つの入口で綴りや既定が割れると、
    「台本ごとに違う名前の同じノブ」が生える。
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out", type=Path, default=out_dir)
    parser.add_argument("--sym-max", type=int, default=SYM_MAX)
    return parser


def run_series_cli(
    parser: argparse.ArgumentParser,
    run: Callable[..., dict[str, Any]],
    argv: Sequence[str] | None,
) -> None:
    """CLI を解いて `run(model_dir, out, **残りのノブ)` を呼び、要約 JSON を刷る。

    MUST: `--model-dir` / `--out` 以外は**そのまま名前付きで**渡す — argparse の dest と
    各 `export_series` のキーワード名が同じ綴りであることが条件で、系列ごとにノブの本数が
    違っても受け渡しを書き足さずに済む形。
    """
    args = parser.parse_args(argv)
    positional = ("model_dir", "out")
    options = {name: value for name, value in vars(args).items() if name not in positional}
    summary = run(args.model_dir, args.out, **options)
    print(json.dumps({"model_dir": str(args.model_dir), **summary}, indent=1, ensure_ascii=False))


def main(argv: Sequence[str] | None = None) -> None:
    run_series_cli(series_parser(__doc__.split("\n\n")[0], DEFAULT_OUT_DIR), export_series, argv)


if __name__ == "__main__":
    main()
