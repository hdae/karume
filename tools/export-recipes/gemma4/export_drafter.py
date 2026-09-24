"""Gemma 4 E2B の **MTP drafter**（HF `google/gemma-4-E2B-it-assistant`）を借り手グラフへ書き出す。

ADR [0096](../../../docs/decisions/0096-speculative-decoding.md) 段 2。

    uv run --with 'transformers==5.14.1' python -m gemma4.export_drafter

## 借り手グラフとは（3 つの宣言）

drafter は自前の KV を持たず、**target（製品グラフ）の l13〈sliding・window 512〉/ l14〈full〉の
k/v をそのまま読む**。出力ヘッドの手前で食う埋め込み表も target の主表を借りる。したがって
このグラフだけでは実行できず、実行時に貸し手 Session / context と束ねられる:

- `states` の 4 スロットは **external**（実体は貸し手の context）。`state_append` は 0 本
- attention は **readonly 形**（`ins` は q の 1 本・今 step の k/v も mask も持たない）
- 主表の initializer は **shared**（バイトを 1 つも書かず、貸し手のテンソルキーを名指す）

trace 上は KV を**通常の入力テンソル**として受け（torch.export は state を辿れない）、
`karume.states.to_external_states_form` がその 4 本の入力を落として readonly 形へ書き換える
（target 側の `to_states_form` と同じ規律 — 純関数・検査は verify）。

## 1 段の中身と k 段展開

    x       = cat(target_embed(token) * sqrt(1536), hidden)        [1,1,3072]
    state   = pre_projection(x)                                    [1,1,256]
    4 層     = q_proj → q_norm → RoPE（ホスト供給の 1 行）→ SDPA(q, K, V, scale=1.0, mask 0)
              → o_proj / 4 RMSNorm + 残差 + layer_scalar / MLP gelu_tanh
    state   = norm(state)
    hidden  = post_projection(state)                               [1,1,1536]
    token   = argmax(lm_head(state))                               [1,1,1]

これを **k = 3 段** Python ループで展開する（段間の唯一の状態は `(token, hidden)` — drafter に
KV は無い）。段ごとに 1 本ずつ、計 3 本の token をグラフ出力にする（IR の `cat` は f32 限定で
i32 を連結できない）。

MUST: `use_ordered_embeddings=False` で読む。配布 checkpoint の centroid 疎 softmax
（topk → 4096 候補 → scatter）は IR 語彙に無い（`topk` は `NON_EMITTABLE_OPS`）ので、
**全語彙の lm_head + argmax** に落とす。`masked_embedding` は `__init__` が config を見て作るので、
読み込んだ後に落とすのではなく**読む前に config を倒す**（`Gemma4AssistantForCausalLM.__init__`）。

## 量子化（席は i8 の 1 つだけ）

| 席 | 格納 | 理由 |
| --- | --- | --- |
| `nn.Linear` 全部（`drafter.lm_head` 込み） | i8 per-channel | 下の実測 — i4 は受理率を削る |
| `target_embed`（共有・バイト無し） | i8 per-channel | 貸し手の宣言と一致 MUST。丸めは golden 用 |
| norm / q_norm / layer_scalar | f32 | `fake_quant_int8` は `weight` を持つ 5 型しか見ない |

**drafter に i4 の席は無い**。2026-09-08 の実測（同一 target 軌跡・N = 200 cycle × 3 ケースの
逐次受理から E[a](k=3) を採る）で、drafter の linear（pre / post projection・q_proj・
o_proj・gate/up/down = 22 本・10.1M 要素）を i4 g32 に落とすと E[a] が丸め無し比で
**−33% / −21% / −15%**（ケース順）まで落ちた。同じ 22 本を i8 per-channel にすると **−2〜−3%**
に収まり、共有主表を i8 にする影響は **±0**。丸めの重い側は出力ヘッド（262,144×256 = 67.1M
要素）で、そちらは i8 のまま — linear 22 本を i4 g32（半バイト + group ごとの f32 scale）から
i8 per-channel（1 バイト + 行ごとの f32 scale）へ上げても配布形の増分は 4MB 弱で、受理率
1 〜 3 割の目減りと釣り合う量ではない（E[a] は投機 1 サイクルあたりの確定 token 数そのもの）。

`target_embed` は **target の product export と同じ丸め**（行ごと per-channel i8）を掛ける。
バイトは書かないので scale 台帳のキーは emit に引かれないが、golden はこの丸めた表で採る。
golden は丸めた後の `wrapper` をそのまま回す（{@link export_series} の順序）ので、配布形の
重みと golden の重みは同一実体 MUST。

## golden の呼び出し規約（ホストが 1 サイクルで渡す組）

貸し手の論理長を `P`（KV に入っている行数）、最後に確定した token を `b`（位置 `P`・**KV には
未投入**）とすると、drafter に渡すのは `token = b` / `hidden = h@(P−1)`（`b` を出した行の
最終 norm 後 hidden）/ RoPE の 1 行 = 位置 `P` / KV の列 `[P−min(P,W), P)`・`[0, P)`。
これは drafter 自身の段間再帰（`token` と「それを出した行の hidden」の組）と同じ組み方で、
段 3 の投機ループが 1 サイクル目に渡す組でもある。したがって `draft[t][j]`（`j` は 0 始まり）が
当てるのは `tokens[t + 1 + j]` — `d₁` は bonus token の**次**を当てる。

## 出力レイアウト

    outputs/series/gemma4-e2b-drafter/model.krm                重み + 2 文書の記述
    outputs/series/gemma4-e2b-drafter/drafter-golden.<case>.safetensors  期待 draft 列
    outputs/series/gemma4-e2b-drafter/reference.json           出所記録（target / drafter の指紋）
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from _shared.container_read import read_layouts
from _shared.decode_series import PROMPT_KEY
from _shared.paths import REPO_ROOT, SERIES_ROOT
from gemma4 import export as one_shot
from gemma4 import export_decode as decode
from gemma4 import export_product as product
from gemma4 import ple, provenance
from gemma4.distribution import GEMMA4_DRAFTER_ROLE
from karume.artifacts import staged_publication
from karume.container import container_parts
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
from karume.dist import ir_graph
from karume.emit import bakeable_initializers
from karume.ir import IrGraph, IrInitializer, IrShared, IrStorage
from karume.ops import ARGMAX_OP, ATTENTION_OP, EMBEDDING_OP, STATE_APPEND_OP, attention_readonly
from karume.pipeline import publish_model
from karume.quantize import fake_quant_int8
from karume.states import ExternalAttentionSpec, ExternalStatesPlan, to_external_states_form

#: 生成物の既定の置き場（製品系列とは別ディレクトリ — 単独では実行できない別資産）。
DEFAULT_OUT_DIR = SERIES_ROOT / "gemma4-e2b-drafter"

#: 貸し手（target）の製品系列。共有 initializer のテンソルキーと external スロットの実形は
#: **ここから読む**（写経すると、target を焼き直した日に借り手だけが古びる）。
DEFAULT_PRODUCT_DIR = product.DEFAULT_OUT_DIR

#: drafter のチェックポイント（HF ハブ — 手置き素材ではない）。
ASSISTANT_REPO = "google/gemma-4-E2B-it-assistant"

#: 1 forward に展開する draft 段数（配布形の drafter はこの 3 段で焼かれている。実行時の k は
#: `k <= 3` の範囲で選べ、それより多い段数が要るなら再 export — ADR 0096）。
DRAFT_STEPS = 3

# NOTE: 貸し手が読む KV の層（E2B は sliding 13 / full 14）は**定数で持たない** —
# `decode.kv_owner_layers(target_config)` が上流の層構成から導く（{@link lender_slots}）。

#: trace 用に置く K / V 入力の行数。手術で入力ごと落ちるので配布形には残らない — 小さければ何でも
#: よく、`1` だと softmax の縮約軸が退化しうるので余裕のある固定値にする。
TRACE_KV_ROWS = 8

#: グラフ入力の名前（**ラッパの forward 引数名そのもの** — torch.export がグラフ入力名に採る）。
#: 手術後に残るのは前半 6 本で、後半 4 本は落ちる。
DRAFTER_TOKEN = "token"
DRAFTER_HIDDEN = "hidden"
DRAFTER_KV_INPUTS: tuple[str, ...] = ("sliding_k", "sliding_v", "full_k", "full_v")

#: 手術後に残るグラフ入力の並び（ホストは名前で束ねる）。
DRAFTER_INPUTS: tuple[str, ...] = (DRAFTER_TOKEN, DRAFTER_HIDDEN, *decode.ROPE_INPUTS)

#: golden のファイル名とテンソルキー。
GOLDEN_PREFIX = "drafter-golden."
GOLDEN_SUFFIX = ".safetensors"
TOKENS_KEY = "tokens"
DRAFT_KEY = "draft"

#: golden の 1 ケースあたりの cycle 数（各 cycle が `(token, hidden)` 1 組 → k 本の draft）。
GOLDEN_CYCLES = 200

#: golden の prefill を割る行数。**karume の既定 chunkLength と同じ**にする（貸し手の実行が
#: 同じ割り方で積むので、積和の順序を揃えられる唯一のノブ）。綴りの正本は配布 recipe。
GOLDEN_PREFILL_CHUNK = 768

#: 短いケースの本文（E-4 の a と同じ会話 — 1 ターンの依頼文）。**窓 512 の内側**で終わる長さで、
#: `P < window` の側（列が窓より少ない）を踏む唯一のケース。
SHORT_PROMPT = (
    "Write a short story (about 300 words) about a lighthouse keeper who discovers a message "
    "in a bottle."
)

#: 長いケースの本文は**リポ内の git 追跡下の文書**から採る（`(ケース名, 相対 path, 文字数上限)`）。
#:
#: MUST: 素材をリポの中に置く — bench の作業席（`outputs/`）は git 追跡外なので、そこから読むと
#: 「同じ台本を回しても別の golden が出る」形になる。
#: MUST: 切るのは**段落境界**（上限の手前の最後の空行）。token 数で切ると文の途中で終わり、継続が
#: 同じ数語を繰り返す退化列になって golden が痩せる（実測: 途中切りの 2,560 token で
#: 「matmul, matmul, matmul」の 3 語ループ）。段落境界なら上限を跨がない限り長さも動かない。
CASE_DOCUMENTS: tuple[tuple[str, str, int], ...] = (
    ("readme-recipes", "tools/export-recipes/README.md", 12000),
    ("readme-exporter", "tools/exporter/README.md", 20000),
)

#: 長いケースの前置き（文書の**前**に置く指示）と、後ろに置く依頼文。
DOCUMENT_INSTRUCTION = "Read the following project document.\n\n"
DOCUMENT_QUESTION = (
    "\n\nBased on the document above, explain in your own words what this project is for and "
    "how its pieces fit together."
)


def truncate_paragraph(text: str, limit: int) -> str:
    """`limit` 文字を超えない範囲で、**最後の段落境界**（空行）まで切り詰める。"""
    if len(text) <= limit:
        return text.rstrip()
    cut = text.rfind("\n\n", 0, limit)
    if cut <= 0:
        raise AssertionError(f"上限 {limit} 文字の手前に段落境界が無い")
    return text[:cut].rstrip()


# ---- 素材 ------------------------------------------------------------------


def _open_mask(key: torch.Tensor) -> torch.Tensor:
    """`[1, 1, 1, N]` の全列許可（0）の加算 mask（N = K の列数）。

    trace 時は N が静的な整数なので**定数**として畳まれ、手術がノードの入力から外した時点で
    刈られる。golden を採るときは実長の K が来るので、同じ式がそのまま実長の mask を返す
    （`register_buffer` で固定長を持つと、実長で回した瞬間に軸が合わない）。
    """
    return torch.zeros(1, 1, 1, key.shape[2], dtype=key.dtype, device=key.device)


class DrafterWrapper(nn.Module):
    """`(token, hidden, RoPE 4 本, K/V 4 本) → (token_0 … token_{k-1})` の export 用ラッパ。

    MUST: forward の**引数名と並び**は {@link DRAFTER_INPUTS} + {@link DRAFTER_KV_INPUTS} と
    一致させる（torch.export が引数名をグラフ入力名に採るので、綴りがずれるとホストが
    束ねられない）。
    MUST: `hidden` は target の**最終 norm 後**の 1 行（製品グラフの出力 1）。lm_head を通す前の
    値であることが drafter 側の入力条件で、別の中間値でも shape は `[1,1536]` のまま合う。
    MUST: K / V は `[1, 1, T, D]`（貸し手スロットと同じ物理形）。trace 用の入力で、手術が
    external スロットへ置き換える。
    MUST: mask は**全列許可の 0 行**（q_len 1 の bidirectional は「全列を見る」— 列の絞りは
    window と論理長が持つ）。列数は K の軸 2 から採る（{@link _open_mask}）ので、golden を採る
    実長でもそのまま回る — 手術が落とす席なので、配布形には 1 バイトも残らない。
    MUST: `use_cache=False`・`position_ids=None`。位置は RoPE の受け渡し口（{@link
    decode.RopeInputs}）が持つので、上流が組む `arange` は誰にも読まれず export の到達解析で死ぬ。
    """

    def __init__(self, drafter: nn.Module, target_embed: nn.Module, steps: int) -> None:
        super().__init__()
        if steps < 1:
            raise ValueError(f"draft 段数 {steps} が 1 以上でない")
        self.drafter = drafter
        self.target_embed = target_embed
        self.steps = steps

    def forward(
        self,
        token: torch.Tensor,
        hidden: torch.Tensor,
        rope_sliding_attention_cos: torch.Tensor,
        rope_sliding_attention_sin: torch.Tensor,
        rope_full_attention_cos: torch.Tensor,
        rope_full_attention_sin: torch.Tensor,
        sliding_k: torch.Tensor,
        sliding_v: torch.Tensor,
        full_k: torch.Tensor,
        full_v: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        tables = decode.bound_rope(
            rope_sliding_attention_cos,
            rope_sliding_attention_sin,
            rope_full_attention_cos,
            rope_full_attention_sin,
        )
        shared = {
            one_shot.SLIDING_ATTENTION: (sliding_k, sliding_v),
            one_shot.FULL_ATTENTION: (full_k, full_v),
        }
        masks = {
            one_shot.SLIDING_ATTENTION: _open_mask(sliding_k),
            one_shot.FULL_ATTENTION: _open_mask(full_k),
        }
        state = hidden.unsqueeze(1)
        drafted: list[torch.Tensor] = []
        for _ in range(self.steps):
            embedded = self.target_embed(token)
            projected = self.drafter.pre_projection(torch.cat([embedded, state], dim=-1))
            with self.drafter.model.rotary_emb.bound(tables):
                inner = self.drafter.model(
                    inputs_embeds=projected,
                    attention_mask=masks,
                    position_ids=None,
                    use_cache=False,
                    shared_kv_states=shared,
                ).last_hidden_state
            state = self.drafter.post_projection(inner)
            token = self.drafter.lm_head(inner).argmax(-1, keepdim=True)
            drafted.append(token)
            token = token.reshape(1, 1)
        return tuple(drafted)


def load_target_embed(model_dir: Path, config: Any) -> nn.Module:
    """target の主埋め込み表を f32 で読み、上流と同じ**スケール付き**埋め込みとして返す。

    MUST: `Gemma4TextScaledWordEmbedding` そのものを使う（`nn.Embedding` + 自前のスカラ倍に
    置き換えない）— `embed_scale = sqrt(hidden_size)` の値も掛ける位置も上流のものであることが、
    貸し手が同じ表を同じ意味で使っていることの根拠になる。
    """
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextScaledWordEmbedding

    key = one_shot.PLE_CHECKPOINT_KEY.replace("embed_tokens_per_layer", "embed_tokens")
    with safe_open(str(model_dir / one_shot.CHECKPOINT_FILE), framework="pt") as handle:
        if key not in set(handle.keys()):
            raise AssertionError(f"target チェックポイントに '{key}' が無い: {model_dir}")
        weight = handle.get_tensor(key).to(torch.float32)
    vocab, hidden = int(config.vocab_size), int(config.hidden_size)
    if list(weight.shape) != [vocab, hidden]:
        raise AssertionError(f"主埋め込み表が {list(weight.shape)} — [{vocab}, {hidden}] でない")
    module = Gemma4TextScaledWordEmbedding(
        vocab, hidden, int(config.pad_token_id), embed_scale=hidden**0.5
    )
    module.weight.data = weight
    return module


def load_drafter(steps: int, model_dir: Path) -> DrafterWrapper:
    """drafter を f32 で読み、疎 softmax を外して RoPE を受け渡し口へ差し替えたラッパを返す。

    MUST: `use_ordered_embeddings` は**読み込む前に**倒す（`masked_embedding` は `__init__` が
    config を見て作るので、後から属性を消しても forward の分岐は疎 softmax 側に倒れたまま）。
    MUST: 内側の `model.embed_tokens` は落とす。tied なので実体は `lm_head` が保持し続け、
    落としておかないと同じ重みが 2 つの FQN から見えて i8 の丸めが二重に掛かる。
    """
    from transformers import AutoConfig, Gemma4AssistantForCausalLM

    one_shot.register_attention()
    config = AutoConfig.from_pretrained(ASSISTANT_REPO)
    config.use_ordered_embeddings = False
    config._attn_implementation = one_shot.ATTENTION_NAME
    text = config.get_text_config()
    text._attn_implementation = one_shot.ATTENTION_NAME
    drafter = Gemma4AssistantForCausalLM.from_pretrained(
        ASSISTANT_REPO, config=config, dtype=torch.float32
    ).eval()
    if drafter.masked_embedding is not None:
        raise AssertionError(
            "疎 softmax（masked_embedding）が残っている"
            " — use_ordered_embeddings を読み込み前に倒せていない"
        )
    del drafter.model.embed_tokens
    drafter.model.rotary_emb = decode.RopeInputs(decode.unique_layer_types(text))
    target_config = one_shot.load_text_config(model_dir)
    return DrafterWrapper(drafter, load_target_embed(model_dir, target_config), steps).eval()


#: 共有する主表のモジュール FQN（{@link DrafterWrapper} の組み立てが決める綴り）。`nn.Linear`
#: ではないので、i8 の席を構造で引く {@link int8_seats} が名前で足す唯一の 1 本。
TARGET_EMBED_MODULE = "target_embed"


def int8_seats(wrapper: nn.Module) -> set[str]:
    """i8 で丸まる**べき**重みの FQN — `nn.Linear` 全部 + 共有する主表（他は 1 本も丸めない）。

    {@link quantize_wrapper} が実際に丸めた台帳と突き合わせるための、**別の規則で引いた**
    期待集合（丸める側は `quantize.QUANT_CHANNEL_AXES` の 5 型で選ぶ）。2 つの集合が食い違う
    現場は 2 通りで、どちらも値にしか出ない:

    - 落としたはずの `drafter.model.embed_tokens` が生き返った — tied なので `lm_head` と
      同じ実体を指し、`fake_quant_int8` が 1 周で**同じ重みを 2 度**丸める（scale 台帳は
      後勝ちの 1 本しか残らないので、宣言した scale で戻らない重みが配布形に載る）
    - drafter の構造が変わって conv や新しい embedding が入った — 黙って i8 に落ちる
    """
    linears = {
        f"{name}.weight"
        for name, module in wrapper.named_modules()
        if isinstance(module, nn.Linear)
    }
    return linears | {f"{TARGET_EMBED_MODULE}.weight"}


def quantize_wrapper(wrapper: nn.Module) -> tuple[Any, dict[str, torch.Tensor]]:
    """重みスロットを持つモジュールを全部 i8 per-channel で丸め、scale 台帳を返す。

    MUST: 丸めは i8 の 1 周だけ（モジュール docstring の実測 — i4 は受理率を削る）。丸め方が
    1 つになっても「同じ実体を指す 2 つの FQN」は 1 周で 2 度丸まるので、実際に丸めた台帳を
    {@link int8_seats} と突き合わせる（`load_drafter` の `del` が落ちた日の検出線）。
    MUST: 共有する主表の丸めは target の {@link gemma4.export.quantize_wrapper} と 1 ビットも
    違わないこと（どちらも per-channel axis 0 の i8）— 貸し手の常駐重みをそのまま食える条件。
    """
    report = fake_quant_int8(wrapper)
    rounded, expected = set(report.scales), int8_seats(wrapper)
    if rounded != expected:
        raise AssertionError(
            f"i8 で丸めた重みが期待の席と違う（余剰 {sorted(rounded - expected)} /"
            f" 欠落 {sorted(expected - rounded)}）"
        )
    return report, dict(report.scales)


# ---- 貸し手（target 製品系列）から読む ---------------------------------------


@dataclass(frozen=True)
class LenderSlots:
    """貸し手の external スロットの実形と、共有 initializer の指し先。"""

    #: スロット名 → 宣言 shape（`[1, Hkv, capacity, D]`）。
    shapes: Mapping[str, list[Any]]
    #: 層種別 → (k スロット名, v スロット名)。
    slots: Mapping[str, tuple[str, str]]
    #: 共有する主表の**貸し手コンテナのテンソルキー**（IR v2 では initializer 名そのもの）。
    shared_tensor: str


def lender_container(product_dir: Path) -> Path:
    """製品コンテナの代表 path（part 列の解決は読み手が持つ）。"""
    return product_dir / one_shot.MODEL_FILE


def lender_graph(product_dir: Path) -> Mapping[str, Any]:
    """製品コンテナのグラフ記述（**IR v2 の文書**）を読む — 重みは 1 バイトも触らない。"""
    return ir_graph(lender_container(product_dir))


def lender_slots(
    graph: Mapping[str, Any], layouts: Mapping[str, str], config: Any, target_config: Any
) -> LenderSlots:
    """貸し手のスロット実形と共有テンソルキーを、製品コンテナの宣言から引く。

    MUST: 形も名前も**貸し手から読む**（drafter の config からは導かない）。借り手の宣言は
    「貸し手と一致していること」だけが正しさで、独立に組むと target を焼き直した日に借り手だけが
    古びる — 形が合っている限り沈黙する。

    MUST: 共有する主表は「embedding の重みスロットで消費される `[V, H]` の initializer」を
    **構造で**引く（キーの綴りを写経しない）。1 本に決まらなければ fail loudly。

    `layouts` はテンソルキー → 格納の layout（{@link _shared.container_read.read_layouts}）—
    IR v2 のグラフ記述は格納を持たない（正本は束縛表）ので、常駐形の突合はそちらから引く。
    """
    owners = decode.kv_owner_layers(target_config)
    states = graph.get("states") or {}
    initializers = graph.get("initializers") or {}
    values = graph.get("values") or {}
    slots: dict[str, tuple[str, str]] = {}
    shapes: dict[str, list[Any]] = {}
    for layer_type in decode.ROPE_LAYER_TYPES:
        if layer_type not in owners:
            raise AssertionError(f"貸し手に層種別 '{layer_type}' の KV 所有層が無い")
        owner = owners[layer_type]
        pair = (decode.slot_name(owner, "k"), decode.slot_name(owner, "v"))
        for name in pair:
            slot = states.get(name)
            if slot is None:
                raise AssertionError(
                    f"貸し手グラフに state スロット '{name}' が無い（宣言 {sorted(states)}）"
                )
            shapes[name] = list(slot["shape"])
        slots[layer_type] = pair

    vocab, hidden = int(target_config.vocab_size), int(target_config.hidden_size)
    weight_slot_names = {
        node["ins"][0] for node in graph["nodes"] if node["op"] == EMBEDDING_OP and node["ins"]
    }
    candidates = sorted(
        name
        for name in weight_slot_names
        if name in initializers and list(values[name]["shape"]) == [vocab, hidden]
    )
    if len(candidates) != 1:
        raise AssertionError(
            f"貸し手グラフの主埋め込み表（embedding の重みスロット・[{vocab}, {hidden}]）が"
            f" {len(candidates)} 本: {candidates}"
        )
    key = candidates[0]
    if initializers[key].get("shared") is True:
        raise AssertionError(f"貸し手の主表 '{key}' が実体を持たない（共有宣言）")
    layout = layouts.get(key)
    if layout != drafter_shared_storage(config).dtype:
        raise AssertionError(
            f"貸し手の主表の格納が '{layout}' —"
            f" 借り手の宣言 '{drafter_shared_storage(config).dtype}' と違う"
        )
    return LenderSlots(shapes=shapes, slots=slots, shared_tensor=key)


def drafter_shared_storage(_config: Any) -> IrStorage:
    """共有 initializer の格納宣言（貸し手と一致 MUST・scale は書かない）。"""
    return IrStorage(dtype="i8")


# ---- 手術 -------------------------------------------------------------------


def external_states_plan(
    graph: IrGraph,
    config: Any,
    lender: LenderSlots,
    *,
    capacity_symbol: str = decode.CAPACITY_SYMBOL,
) -> ExternalStatesPlan:
    """attention ノード（層順 = 出現順）を貸し手のスロットへ割り付ける手術指定。

    層種別ごとに読むスロットが決まる（sliding 層 3 本 → 貸し手の sliding 所有層 / full 層 1 本 →
    full 所有層）。k 段展開しているので、同じスロットへ `steps × 層数` 本の読者が付く。

    MUST: スロットの形は**貸し手の宣言から**引く（{@link lender_slots}）。窓は借り手 config の
    `sliding_window`（読む窓幅は借り手の意味論で、貸し手の物理容量とは別の数）。
    """
    layer_types = list(config.layer_types)
    nodes = [node for node in graph.nodes if node.op == ATTENTION_OP]
    if len(nodes) % len(layer_types):
        raise AssertionError(
            f"attention が {len(nodes)} 本 — 層数 {len(layer_types)} の整数倍でない（k 段展開）"
        )
    window = int(config.sliding_window)
    specs: list[ExternalAttentionSpec] = []
    for index, node in enumerate(nodes):
        layer_type = layer_types[index % len(layer_types)]
        k_slot, v_slot = lender.slots[layer_type]
        shape = lender.shapes[k_slot]
        sliding = layer_type == one_shot.SLIDING_ATTENTION
        k_input, v_input = (
            (DRAFTER_KV_INPUTS[0], DRAFTER_KV_INPUTS[1])
            if sliding
            else (DRAFTER_KV_INPUTS[2], DRAFTER_KV_INPUTS[3])
        )
        specs.append(
            ExternalAttentionSpec(
                output=node.outs[0],
                k_slot=k_slot,
                v_slot=v_slot,
                k_input=k_input,
                v_input=v_input,
                kv_heads=int(shape[1]),
                head_dim=int(shape[3]),
                capacity=shape[2],
                window=window if sliding else None,
            )
        )
    return ExternalStatesPlan(capacity_symbol=capacity_symbol, attentions=tuple(specs))


def share_target_embedding(
    graph: IrGraph, tensors: dict[str, torch.Tensor], lender: LenderSlots, config: Any
) -> tuple[IrGraph, dict[str, torch.Tensor], str]:
    """主表の initializer を **shared 宣言**へ置き換え、その実体を格納から外す。

    MUST: 対象は「embedding の重みスロットで消費される `[V, H]` の initializer」1 本ちょうど
    （借り手側でも構造で引く — FQN 由来のテンソルキーは torch.export の内部順序で決まる）。
    MUST: 実体を `tensors` から落とす。残すと `write_model` の宣言 / 格納の完全一致で
    「余剰」として落ちる（= 検出線はあるが、落とすのはこちらの仕事）。
    """
    vocab, hidden = int(config.vocab_size), int(config.hidden_size)
    consumed = {node.ins[0] for node in graph.nodes if node.op == EMBEDDING_OP and node.ins}
    candidates = sorted(
        name
        for name in consumed
        if name in graph.initializers and list(graph.values[name].shape) == [vocab, hidden]
    )
    if len(candidates) != 1:
        raise AssertionError(
            f"借り手グラフの主埋め込み表（[{vocab}, {hidden}]）が {len(candidates)} 本:"
            f" {candidates}"
        )
    name = candidates[0]
    key = graph.initializers[name].tensor
    initializers = dict(graph.initializers)
    initializers[name] = IrInitializer(
        shared=IrShared(tensor=lender.shared_tensor), storage=drafter_shared_storage(config)
    )
    remaining = {tensor: value for tensor, value in tensors.items() if tensor != key}
    if len(remaining) != len(tensors) - 1:
        raise AssertionError(f"主表の実体 '{key}' が格納テンソルに無い")
    shared_graph = IrGraph(
        symbols=list(graph.symbols),
        inputs=list(graph.inputs),
        outputs=list(graph.outputs),
        initializers=initializers,
        values=dict(graph.values),
        states=dict(graph.states),
        nodes=list(graph.nodes),
    )
    return shared_graph, remaining, name


# ---- 形検査 ------------------------------------------------------------------


def assert_ir_form_drafter(
    graph: IrGraph,
    config: Any,
    lender: LenderSlots,
    steps: int,
    *,
    capacity_symbol: str = decode.CAPACITY_SYMBOL,
) -> dict[str, Any]:
    """借り手グラフの形を検査する（**数値が合ったまま静かに壊れる**性質を全部見る）。

    - グラフ入力が `token` / `hidden` / RoPE 4 本ちょうど（K/V の placeholder が残っていれば、
      呼び手が渡す値をどのノードも読まない形になる）
    - 出力が k 本で、いずれも `argmax` 由来の i32 `[1,1,1]`（順序 = 段順 MUST）
    - state スロットが**全部 external**で、名前も形も貸し手の l13 / l14 と一致
    - `state_append` が **0 本**（借り手は貸し手の ring を 1 行も書かない）
    - attention が全部 readonly 形で、本数が `層数 × 段数`
    - initializer に共有宣言が 1 本（バイト無し）で、指し先が貸し手のテンソルキー
    - 記号は容量記号 1 本ちょうど（K/V を落としたので、束縛点は states shape だけ）
    """
    names = [spec.name for spec in graph.inputs]
    if names != list(DRAFTER_INPUTS):
        raise AssertionError(
            f"グラフ入力が {names} — {list(DRAFTER_INPUTS)} でない"
            "（K/V の placeholder が手術で落ちていない可能性）"
        )
    token_spec, hidden_spec = graph.inputs[0], graph.inputs[1]
    backbone = int(config.backbone_hidden_size)
    if token_spec.dtype != "i32" or list(token_spec.shape) != [1, 1]:
        raise AssertionError(
            f"'{DRAFTER_TOKEN}' が {token_spec.dtype} {list(token_spec.shape)} — i32 [1, 1] でない"
        )
    if hidden_spec.dtype != "f32" or list(hidden_spec.shape) != [1, backbone]:
        raise AssertionError(
            f"'{DRAFTER_HIDDEN}' が {hidden_spec.dtype} {list(hidden_spec.shape)} —"
            f" f32 [1, {backbone}] でない（target の最終 norm 後 hidden）"
        )
    text = config.get_text_config()
    specs = decode.rope_specs(text)
    for layer_type in decode.ROPE_LAYER_TYPES:
        name = decode.rope_input_name(layer_type, "cos")
        spec = next(item for item in graph.inputs if item.name == name)
        expected = [1, 1, specs[layer_type].head_dim]
        if spec.dtype != "f32" or list(spec.shape) != expected:
            raise AssertionError(
                f"グラフ入力 '{name}' が {spec.dtype} {list(spec.shape)} — f32 {expected} でない"
            )

    if len(graph.outputs) != steps:
        raise AssertionError(f"IR 出力が {len(graph.outputs)} 本（draft {steps} 段の 1 本ずつ）")
    producer = {out: node for node in graph.nodes for out in node.outs}
    for index, output in enumerate(graph.outputs):
        source = producer.get(output)
        if source is None or source.op != ARGMAX_OP:
            found = "ノード出力でない" if source is None else source.op
            raise AssertionError(f"出力 {index} の供給元が {found} — `{ARGMAX_OP}` でない")
        value = graph.values[output]
        if value.dtype != "i32" or list(value.shape) != [1, 1, 1]:
            raise AssertionError(
                f"出力 {index} が {value.dtype} {list(value.shape)} — i32 [1, 1, 1] でない"
            )

    expected_slots = {name: shape for name, shape in lender.shapes.items()}
    if sorted(graph.states) != sorted(expected_slots):
        raise AssertionError(
            f"states 宣言が {sorted(graph.states)} — 貸し手の {sorted(expected_slots)} でない"
        )
    for name, shape in sorted(expected_slots.items()):
        slot = graph.states[name]
        if not slot.external:
            raise AssertionError(f"states['{name}'] が external でない（借り手は実体を持たない）")
        if slot.dtype != "f32" or list(slot.shape) != shape:
            raise AssertionError(
                f"states['{name}'] が {slot.dtype} {list(slot.shape)} — 貸し手の f32 {shape} と違う"
            )
    appends = [node for node in graph.nodes if node.op == STATE_APPEND_OP]
    if appends:
        raise AssertionError(
            f"`{STATE_APPEND_OP}` が {len(appends)} 本残っている"
            "（借り手は貸し手の ring を 1 行も書かない）"
        )

    attentions = [node for node in graph.nodes if node.op == ATTENTION_OP]
    expected_attentions = int(text.num_hidden_layers) * steps
    if len(attentions) != expected_attentions:
        raise AssertionError(
            f"attention が {len(attentions)} 本 — 層数 {text.num_hidden_layers} × 段数 {steps} ="
            f" {expected_attentions} 本でない"
        )
    plain = [index for index, node in enumerate(attentions) if not attention_readonly(node.attrs)]
    if plain:
        raise AssertionError(
            f"readonly でない attention が {len(plain)} 本残っている（添字 {plain[:4]}）"
        )
    window = int(text.sliding_window)
    for index, node in enumerate(attentions):
        layer_type = list(text.layer_types)[index % int(text.num_hidden_layers)]
        expected_states = dict(zip(("k", "v"), lender.slots[layer_type], strict=True))
        if node.states != expected_states:
            raise AssertionError(
                f"attention[{index}] ({layer_type}): states 欄が {node.states}"
                f"（期待 {expected_states}）"
            )
        expected_window = window if layer_type == one_shot.SLIDING_ATTENTION else None
        if node.attrs.get("window") != expected_window:
            raise AssertionError(
                f"attention[{index}] ({layer_type}): attrs の window が"
                f" {node.attrs.get('window')}（期待 {expected_window}）"
            )

    shared = sorted(name for name, init in graph.initializers.items() if init.is_shared)
    if len(shared) != 1:
        raise AssertionError(f"共有 initializer が {len(shared)} 本: {shared}（1 本ちょうど）")
    declaration = graph.initializers[shared[0]]
    if declaration.shared.tensor != lender.shared_tensor:
        raise AssertionError(
            f"共有 initializer の指し先が '{declaration.shared.tensor}' —"
            f" 貸し手の '{lender.shared_tensor}' でない"
        )
    if declaration.storage.scale is not None:
        raise AssertionError("共有 initializer が scale を宣言している（貸し手の常駐重みが持つ）")

    if sorted(graph.symbols) != [capacity_symbol]:
        raise AssertionError(f"symbols が {sorted(graph.symbols)} — ['{capacity_symbol}'] でない")
    residue = sorted(set(graph.required_ops) & set(decode.RESIDUE_OPS))
    if residue:
        raise AssertionError(f"手術で死ぬはずの op が残っている: {residue}")

    storage: dict[str, int] = {}
    for initializer in graph.initializers.values():
        dtype = initializer.storage.dtype
        storage[dtype] = storage.get(dtype, 0) + 1
    return {
        "inputs": names,
        "outputs": len(graph.outputs),
        "attention_nodes": len(attentions),
        "external_slots": sorted(graph.states),
        "shared_initializer": shared[0],
        "shared_tensor": declaration.shared.tensor,
        "storage": dict(sorted(storage.items())),
    }


# ---- golden -----------------------------------------------------------------


def build_cases(model_dir: Path) -> tuple[tuple[str, torch.Tensor], ...]:
    """golden ケースの `(名前, input_ids)`（`<bos>` はここで足す — `gemma4.export` と同じ規約）。

    素材は短い依頼文 1 本 + **リポ内の git 追跡下の文書** 2 本（{@link CASE_DOCUMENTS}）。
    短いケースは窓 512 の内側で終わり、長い 2 本は窓を大きく跨ぐ — sliding スロットが
    `[P − W, P)` を巻きながら返す側と、巻かない側の両方を踏む。
    """
    tokenizer = one_shot.load_tokenizer(model_dir)
    bos = one_shot.bos_token_id(model_dir, tokenizer)

    def encode(text: str) -> torch.Tensor:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        return torch.tensor([[bos, *ids]], dtype=torch.int64)

    cases: list[tuple[str, torch.Tensor]] = [("short-en", encode(SHORT_PROMPT))]
    for name, relative, limit in CASE_DOCUMENTS:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise AssertionError(f"ケース '{name}' の素材 {path} が無い")
        document = truncate_paragraph(path.read_text(encoding="utf-8"), limit)
        cases.append((name, encode(DOCUMENT_INSTRUCTION + document + DOCUMENT_QUESTION)))
    return tuple(cases)


@dataclass
class _Lender:
    """golden を採るときの貸し手（実重みの target）— cache と共有 KV を持ち回る器。"""

    model: Any
    tables: Any
    scale: float
    specs: Mapping[str, Any]
    cache: Any
    shared: Any
    length: int = 0

    def step(self, ids: torch.Tensor) -> torch.Tensor:
        """`ids[1, L]` を cache へ流し、最終行の post-norm hidden `[1, H]` を返す。"""
        positions = torch.arange(self.length, self.length + int(ids.shape[1]))
        rope = decode.rope_args(self.specs, positions.unsqueeze(0))
        device = self.model.device
        embeds = self.model.model.embed_tokens(ids.to(device))
        stacked = ple.per_layer_inputs(self.tables, ids.to("cpu"), self.scale).to(device)
        with (
            torch.no_grad(),
            self.model.model.rotary_emb.bound(
                decode.bound_rope(*(table.to(device) for table in rope))
            ),
        ):
            out = self.model.model(
                inputs_embeds=embeds,
                per_layer_inputs=stacked,
                attention_mask=None,
                position_ids=positions.unsqueeze(0).to(device),
                past_key_values=self.cache,
                use_cache=True,
                shared_kv_states=self.shared,
                return_shared_kv_states=True,
            )
        self.length += int(ids.shape[1])
        return out.last_hidden_state[:, -1]

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model.lm_head(hidden)

    def kv(self, layer_type: str, columns: int) -> tuple[torch.Tensor, torch.Tensor]:
        """`shared_kv_states` から**末尾 `columns` 列ちょうど**を切り出す。

        MUST: 切り出しを台本が持つ（cache の保持本数に預けない）。HF の sliding 層は
        「直前 `window − 1` 本 + 今回ぶん」を返すので本数が step 形と chunk 形で違い、そのまま
        渡すと golden だけが別の列集合を見る。ランタイム側の契約は
        `[P − min(P, W), P)`（sliding）/ `[0, P)`（full）なので、そこへ揃える。
        """
        key, value = self.shared[layer_type]
        held = int(key.shape[2])
        if held < columns:
            raise AssertionError(
                f"共有 KV '{layer_type}' が {held} 列しか無い（{columns} 列が要る）"
            )
        return key[:, :, held - columns :], value[:, :, held - columns :]


def _new_lender(wrapper: Any, specs: Mapping[str, Any]) -> _Lender:
    """1 ケースぶんの貸し手（cache と共有 KV はケースごとに作り直す）。"""
    from transformers import DynamicCache

    return _Lender(
        model=wrapper.model,
        tables=wrapper.per_layer,
        scale=wrapper.per_layer_scale,
        specs=specs,
        cache=DynamicCache(config=wrapper.model.config),
        shared={},
    )


def draft_case(
    drafter: DrafterWrapper,
    wrapper: Any,
    specs: Mapping[str, Any],
    ids: torch.Tensor,
    cycles: int,
    *,
    device: str,
    chunk: int = GOLDEN_PREFILL_CHUNK,
) -> dict[str, torch.Tensor]:
    """1 ケースぶんの golden（`prompt` / `tokens` / `draft`）を採る。

    貸し手を prompt で prefill（`chunk` 行ずつ）してから、`cycles` 回の greedy 継続を回す。
    各 cycle の入り口は貸し手の論理長 `P`（= KV に入っている行数）で、drafter へ渡すのは

    - `token` = `b`（**最後に確定した token** — 位置 `P` に居るが KV には未投入）
    - `hidden` = `h@(P−1)`（`b` を出した行の最終 norm 後 hidden）
    - RoPE の 1 行 = 位置 **`P`**（`b` の位置）
    - 貸し手の l13 / l14 の列 `[P − min(P, W), P)` / `[0, P)`

    の 4 つ。drafter 自身の段間再帰（`token` と「それを出した行の hidden」の組を次段へ送る）と
    同じ組で、投機ループが 1 サイクル目に渡す組でもある。したがって `draft[t][j]`（0 始まりの
    `j`）が当てるのは位置 `P+1+j` の token = `tokens[t + 1 + j]` で、`d₁` は bonus token の
    次を当てる。

    MUST: `tokens` は `cycles + k` 本採る（最後の cycle の draft にも比較相手が要る）。

    MUST: 貸し手へ流す token は **drafter の draft ではなく greedy 継続**（teacher forcing）—
    受理・棄却のループを回すと golden がその実装に依存し、段 3 の投機ループを変えるたびに
    採り直しになる。
    """
    window = int(drafter.drafter.config.get_text_config().sliding_window)
    lender = _new_lender(wrapper, specs)
    for start in range(0, int(ids.shape[1]), chunk):
        hidden = lender.step(ids[:, start : start + chunk])
    tokens: list[int] = []
    drafts: list[list[int]] = []
    for cycle in range(cycles + drafter.steps):
        past = lender.length
        # 位置 P の frontier token（貸し手が h@(P−1) から選んだ 1 本 — まだ KV に無い）。
        nxt = int(lender.logits(hidden).argmax(-1))
        tokens.append(nxt)
        if cycle < cycles:
            sliding = lender.kv(one_shot.SLIDING_ATTENTION, min(past, window))
            full = lender.kv(one_shot.FULL_ATTENTION, past)
            rope = decode.rope_args(specs, torch.tensor([[past]]))
            with torch.no_grad():
                drafted = drafter(
                    torch.tensor([[nxt]], dtype=torch.int64, device=device),
                    hidden,
                    *(table.to(device) for table in rope),
                    *sliding,
                    *full,
                )
            drafts.append([int(value) for value in drafted])
        hidden = lender.step(torch.tensor([[nxt]], dtype=torch.int64))
    return {
        PROMPT_KEY: ids[0].to(torch.int32).contiguous(),
        TOKENS_KEY: torch.tensor(tokens, dtype=torch.int32).contiguous(),
        DRAFT_KEY: torch.tensor(drafts, dtype=torch.int32).contiguous(),
    }


def acceptance_stats(tensors: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """golden が示す受理の見込み（`per_step` = 位置別一致率・`tokens_per_cycle` = 逐次受理）。

    `draft[t][j]` が当てるのは `tokens[t + 1 + j]`（{@link draft_case}）。位置別一致率は段 `j`
    を独立に見た当たり率で、逐次受理は **先頭から一致が続く長さ `a`** から採る — 投機ループが
    1 サイクルで確定させるのは `1 + a` 本（棄却行にも bonus token が 1 本乗る）なので、
    `tokens_per_cycle` = `1 + mean(a)` がそのまま「1 サイクルあたりの確定 token 数」になる。
    """
    tokens, draft = tensors[TOKENS_KEY], tensors[DRAFT_KEY]
    cycles, steps = int(draft.shape[0]), int(draft.shape[1])
    if int(tokens.shape[0]) != cycles + steps:
        raise AssertionError(
            f"tokens が {int(tokens.shape[0])} 本 — cycles {cycles} + steps {steps} でない"
        )
    per_step = [0] * steps
    prefixes = 0
    for cycle in range(cycles):
        running = True
        for step in range(steps):
            hit = int(draft[cycle][step]) == int(tokens[cycle + 1 + step])
            per_step[step] += int(hit)
            running = running and hit
            prefixes += int(running)
    return {
        "per_step": [hits / cycles for hits in per_step],
        "tokens_per_cycle": 1 + prefixes / cycles,
    }


def write_goldens(
    drafter: DrafterWrapper,
    model_dir: Path,
    cases: Sequence[tuple[str, torch.Tensor]],
    out_dir: Path,
    cycles: int,
    device: str,
) -> dict[str, Any]:
    """全ケースの golden を採って書き、要約を返す（貸し手はここでだけ読む）。

    MUST: 貸し手は **製品系列と同じ丸め**（`export_product.load_wrapper` +
    `export.quantize_wrapper`）。別の丸めで採った期待列は「同じ向きに間違った 2 つ」を
    突き合わせる形になる。
    """
    print("[golden] target（製品系列と同じ丸め）を読む", file=sys.stderr, flush=True)
    wrapper = product.load_wrapper(model_dir)
    one_shot.quantize_wrapper(wrapper)
    # MUST: 貸し手だけ上流既定の SDPA へ戻す。`karume_gqa`
    # （{@link gemma4.export.gqa_sdpa_attention}）は「mask 無しは非因果に化ける」ので mask を
    # 必須にするが、KV cache を使う経路では上流が「causal は is_causal に任せる」判断で
    # mask を **None** にすることがある（`create_causal_mask`）。
    # ここで mask を自前で組むと、sliding 層の cache 切り詰め（`DynamicSlidingWindowLayer` は
    # 直前 `window − 1` 本しか残さない）に合わせた kv_offset の再実装まで抱えることになる。
    # 数値は同じ SDPA で、違いは repeat_kv を実体化するかどうかだけ（借り手側は export と同じ
    # `karume_gqa` のまま — golden の対象はそちら）。
    wrapper.model.config._attn_implementation = "sdpa"
    specs = decode.rope_specs(wrapper.model.config)
    # PLE 35 表（9.4GB）は CPU に残す MUST — GPU へ載せると 12GB 級のカードでは
    # 塔を外しても入らない。`wrapper.model` だけを移すので、兄弟属性の `per_layer` は動かない。
    wrapper.model.to(device)
    drafter.to(device)

    summary: dict[str, Any] = {}
    for name, ids in cases:
        print(f"[golden] {name}: prompt {int(ids.shape[1])} token", file=sys.stderr, flush=True)
        tensors = draft_case(drafter, wrapper, specs, ids, cycles, device=device)
        path = out_dir / f"{GOLDEN_PREFIX}{name}{GOLDEN_SUFFIX}"
        save_file(tensors, str(path))
        stats = acceptance_stats(tensors)
        summary[name] = {
            "file": path.name,
            "prompt": int(tensors[PROMPT_KEY].shape[0]),
            "cycles": int(tensors[DRAFT_KEY].shape[0]),
            "steps": int(tensors[DRAFT_KEY].shape[1]),
            **stats,
        }
        print(
            f"[golden] {name}: 位置別一致率 "
            + " / ".join(
                f"d{step + 1} {rate * 100:.1f}%" for step, rate in enumerate(stats["per_step"])
            )
            + f" / 逐次受理 {stats['tokens_per_cycle']:.3f} token/cycle",
            file=sys.stderr,
            flush=True,
        )
    drafter.to("cpu")
    return summary


# ---- 系列 --------------------------------------------------------------------


def drafter_reference(out_dir: Path, model_dir: Path, goldens: Mapping[str, Any]) -> dict[str, Any]:
    """出所記録（target と drafter の**両方**の指紋 + 採った golden の digest）。

    貸し手 / 借り手の 2 つのチェックポイントから出た資産なので、片方だけ差し替えた組み合わせが
    作れないよう両方を名指す（{@link gemma4.provenance} の 1 系列版の拡張）。
    """
    assistant = _assistant_snapshot_dir()
    return {
        "schema": provenance.SCHEMA,
        "series": out_dir.name,
        "checkpoint": provenance.checkpoint_fingerprint(model_dir),
        "drafter": {
            "repo": ASSISTANT_REPO,
            "dir": assistant.name,
            "files": {
                name: provenance.file_digest(assistant / name)
                for name in ("model.safetensors", "config.json")
            },
        },
        "goldens": dict(goldens),
    }


def _assistant_snapshot_dir() -> Path:
    """HF cache 上の drafter スナップショット（指紋を採る実体の在処）。"""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(ASSISTANT_REPO, allow_patterns=["*.json", "*.safetensors"]))


def export_series(
    model_dir: Path,
    out_dir: Path,
    *,
    steps: int = DRAFT_STEPS,
    product_dir: Path = DEFAULT_PRODUCT_DIR,
    cycles: int = GOLDEN_CYCLES,
    goldens: bool = True,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """借り手コンテナ + golden + 出所記録を書き、要約を返す。

    MUST: 生成物は作業席へ書き、**全ての門**（形検査・golden）を通してから据える
    （{@link karume.artifacts.staged_publication}）。
    MUST: 丸めは参照・golden の採取より前（ADR 0006）— 後だと golden だけが元の重みで動く。
    """
    print("[load] drafter（疎 softmax 無し）+ target の主表", file=sys.stderr, flush=True)
    wrapper = load_drafter(steps, model_dir)
    config = wrapper.drafter.config
    text = config.get_text_config()

    print("[quant] i8 per-channel（nn.Linear 全部 + 共有主表）", file=sys.stderr, flush=True)
    int8, scales = quantize_wrapper(wrapper)

    lender = lender_slots(
        lender_graph(product_dir),
        read_layouts(lender_container(product_dir)),
        config,
        one_shot.load_text_config(model_dir),
    )
    specs = decode.rope_specs(text)
    example = example_inputs(wrapper, specs)

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with staged_publication(out_dir) as staged:
        staged.mkdir()
        print("[export] torch.export → 変換", file=sys.stderr, flush=True)
        graph, tensors = export_module_with_attention(wrapper, example)
        print("[export] 手術 → 共有宣言 → 書き出し", file=sys.stderr, flush=True)
        surgical = to_external_states_form(graph, external_states_plan(graph, text, lender))
        shared_graph, remaining, shared_name = share_target_embedding(
            surgical, tensors, lender, one_shot.load_text_config(model_dir)
        )
        verified = publish_model(
            staged / one_shot.MODEL_FILE,
            shared_graph,
            {name: value for name, value in remaining.items() if name in _declared(shared_graph)},
            provenance=one_shot.PROVENANCE,
            # グラフ名は**部品名**（= karume.json の weights のキー）。ディレクトリ名から
            # 導かない — 系列名（`gemma4-e2b-product`）も作業席の名前も部品名とは一致しない
            # （container-v1 §2.1）。
            graph_name=GEMMA4_DRAFTER_ROLE,
            weight_dtype="i8",
            weight_scales=scales,
        )
        form = assert_ir_form_drafter(verified, config, lender, steps)

        golden_summary: dict[str, Any] = {}
        if goldens:
            cases = build_cases(model_dir)
            golden_summary = write_goldens(wrapper, model_dir, cases, staged, cycles, device)
        print("[provenance] チェックポイント指紋 → reference.json", file=sys.stderr, flush=True)
        provenance.write_record(staged, drafter_reference(out_dir, model_dir, golden_summary))

    return {
        "dir": str(out_dir),
        "steps": steps,
        "nodes": len(verified.nodes),
        "initializers": len(verified.initializers),
        "shared_initializer": shared_name,
        "model_bytes": sum(
            path.stat().st_size for path in container_parts(out_dir / one_shot.MODEL_FILE)
        ),
        "parts": [path.name for path in container_parts(out_dir / one_shot.MODEL_FILE)],
        "ops": sorted(verified.required_ops),
        "symbols": list(verified.symbols),
        "quantized": {"i8": int8.describe()},
        "form": form,
        "goldens": golden_summary,
    }


def _declared(graph: IrGraph) -> set[str]:
    """このコンテナに実体を書く initializer のテンソルキー（共有宣言を除く）。

    除外の規則そのものは core（{@link karume.emit.bakeable_initializers}）が持つ — ここは
    「名前 → テンソルキー」へ射影するだけで、規則を写さない。
    """
    return {graph.initializers[name].tensor for name in bakeable_initializers(graph)}


def export_module_with_attention(
    wrapper: nn.Module, example: tuple[torch.Tensor, ...]
) -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """SDPA を保存したまま export する（`attention` op が残る形 — ADR 0023 の opt-in）。

    記号は 1 本も宣言しない（K/V の placeholder は固定長で、容量記号は手術が足す）。
    """
    from karume.pipeline import export_module

    return export_module(
        wrapper,
        example,
        dynamic_shapes=None,
        symbol_names=(),
        preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    )


def example_inputs(wrapper: DrafterWrapper, specs: Mapping[str, Any]) -> tuple[torch.Tensor, ...]:
    """trace の例示入力（K / V は固定 {@link TRACE_KV_ROWS} 行の placeholder）。"""
    backbone = int(wrapper.drafter.config.backbone_hidden_size)
    generator = torch.Generator().manual_seed(20260908)
    rope = decode.rope_args(specs, torch.tensor([[TRACE_KV_ROWS - 1]]))
    sliding = specs[one_shot.SLIDING_ATTENTION].head_dim
    full = specs[one_shot.FULL_ATTENTION].head_dim
    return (
        torch.zeros(1, 1, dtype=torch.int64),
        torch.randn(1, backbone, generator=generator),
        *rope,
        torch.randn(1, 1, TRACE_KV_ROWS, sliding, generator=generator),
        torch.randn(1, 1, TRACE_KV_ROWS, sliding, generator=generator),
        torch.randn(1, 1, TRACE_KV_ROWS, full, generator=generator),
        torch.randn(1, 1, TRACE_KV_ROWS, full, generator=generator),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = one_shot.series_parser(__doc__.split("\n\n")[0], DEFAULT_OUT_DIR)
    parser.add_argument("--steps", type=int, default=DRAFT_STEPS)
    parser.add_argument("--product-dir", type=Path, default=DEFAULT_PRODUCT_DIR)
    parser.add_argument("--cycles", type=int, default=GOLDEN_CYCLES)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no-goldens", dest="goldens", action="store_false")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    options = {
        name: value
        for name, value in vars(args).items()
        if name not in ("model_dir", "out", "sym_max")
    }
    summary = export_series(args.model_dir, args.out, **options)
    print(json.dumps({"model_dir": str(args.model_dir), **summary}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
