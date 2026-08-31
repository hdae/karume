"""実重み Gemma 4 E2B を **states 形の chunk グラフ**（IR v1 + golden）へ書き出す台本。

ADR [0066](../../../docs/decisions/0066-generation-context-state-slots.md)（GenerationContext と
名前付き state スロット）/
[0067](../../../docs/decisions/0067-autoregressive-attention-vocabulary.md) 決定 4〜5b
（state 参照つき attention と `state_append`）/
[0068](../../../docs/decisions/0068-decode-exit-multi-output.md) 決定 4（decode 出口）/
[0069](../../../docs/decisions/0069-packed-w4-storage.md)（packed w4）を、**sliding + KV 共有 +
混成量子化**を同時に持つ実モデルで検収する足場。同じ枠の KV 共有なし版が
`minicpm5/export_decode.py`。モデル非依存の門（greedy の採り方・余裕門・容量門）の正本は
{@link _shared.decode_series}、1-shot 形のヘルパは {@link gemma4.export} で、どちらも import
して再利用する（同じ規律を 2 箇所に書かない）。

    uv run --with 'transformers==5.14.1' python -m gemma4.export_decode

## この台本は chunk 系列 2 本の中核でもある

states 形の chunk グラフは出口が 2 種類ある — logits opt-in 形（この台本の系列）と
token-only 既定出口（{@link gemma4.export_token} の系列 / ADR 0068 決定 4）。両者で違うのは
**出口の形と golden を採るかどうかだけ**で、素材の読み方・RoPE の表引き・KV 共有の手術・
混成量子化・門の順序は同一なので、経路は {@link ChunkVariant} 駆動の 1 本
（{@link load_wrapper} / {@link export_series} / {@link run_variant_cli}）に閉じる。
token-only 台本が持つのはラッパ 1 つと variant 記述と入口だけ。

MUST: 差分は variant の 4 欄に載る形だけにする。台本側へ経路を戻すと、片方にだけ門を足した
ときにもう片方が黙って弱くなる（`_shared.decode_series` の module docstring と同じ理由で、
実際に逐語コピーだった時期に margin 床の MUST が片方にしか無い状態が生まれた）。

## 何をグラフに載せるか

`(input_ids[1,M], position_ids[1,M]) → (logits[1,M,262144], token[1,M,1])` の **chunk ラッパ**
1 本。`M` は物理 chunk 次元（ADR 0067 決定 4 — prefill は `chunkLength` / decode は 1）で、
有効行は先頭 `queryLength` 行の compact-prefix。`token` は `logits.argmax(-1, keepdim=True)`
（ADR 0068 決定 4 の decode 出口）。**出力順は `[logits, token]` 固定**で、ランタイム側の
slot 番号がこの順を読む。

## 位置は入力・RoPE は表引き（1-shot 形との構造差 その 1）

1-shot 形は `position_ids` を持たない — 位置は常に `0..T-1` なので cos / sin は `arange` の
定数畳み込みで Tmax 表 + `sym_prefix_slice` に落ちる（ADR 0010）。decode では位置が
`pastLength + row` で**実行時にしか決まらない**ので、この畳み込みは成立しない。

そこで `model.model.rotary_emb` を {@link RopeTable}（表引きモジュール）へ差し替える
（上流のモデリングコードは 1 行も触らない）。Gemma 4 の rotary は Llama 系と違い
**`forward(x, position_ids, layer_type)`** で層種別ごとに呼ばれる（modeling_gemma4.py:1713）
ので、表も層種別ごとに 1 組ずつ持つ。表は**元の rotary を `arange(POS_MAX)[None]` で
層種別ごとに 1 回呼んで**得た cos / sin そのもので、値同一は {@link swap_rope_table} が
構築時に 3 種の position 列 × 全 layer_type で `torch.equal` 突合する。full 層の RoPE は
`proportional`（`partial_rotary_factor` 0.25）で零周波数が 192 本あるが、それは cos = 1 /
sin = 0 の列として表に自然に入るだけなので特別扱いは要らない。

MUST: `rope.assert_rope_lifted` は**適用しない** — あれは `inv_freq` バッファを畳み込みの葉へ
降格する門で、本台本は rotary モジュールを丸ごと差し替えるため `inv_freq` は模型から消える。
代わりに {@link assert_ir_form_decode} が **`sin` 系 op の不在**を直接見る（`cos` は IR 語彙に
無いので、RoPE が残ったグラフは export 自体が落ちる — 到達しうる残骸は `sin` の側だけ）。

## KV 共有層は「所有層のスロットを読む」（1-shot 形との構造差 その 2）

E2B は 35 層のうち後ろ 20 層（`num_kv_shared_layers`）が KV を共有する。上流は共有開始より
前の層のうち **その layer_type で最後の層**の k / v を `shared_kv_states[layer_type]` に置き、
共有層はそれをそのまま使う（modeling_gemma4.py:1287 / 1735）。よってスロットは
**所有層ぶんの 30 本**（層 0..14 × k/v）だけで、層 15..34 は sliding なら層 13 の・full なら
層 14 のスロットを読む（{@link kv_owner_layers}）。

手術（`karume.states.to_states_form`）は同じスロットへの複数登録を受理し、導出 shape /
`window` / **append の入力名**の完全一致を検査する（states.py の `_register`）。共有層の k / v が
所有層の**同じ値テンソル**であることは traced グラフが持っている事実なので、この検査が
「割り当てを間違えていない」ことの実証になる（間違えれば入力名が食い違って落ちる）。
`state_append` は所有層ぶんの 30 本で、それぞれ**そのスロットの最後の読者の直後**に置かれる
（順序と window 宣言の一致は `verify._assert_state_order` の担当 — ここでは写さない）。

## 窓幅 512 はそのまま宣言できる（意味論が厳密同値）

karume の sliding 述語は `in_window(col, limit) = col <= limit && (limit - col) < params.window`
（packages/runtime/src/kernels/state-attention.ts:234-235・`limit = pastLength + row`）。HF 側の
`sliding_window_mask_function` は causal と `kv_idx > q_idx - sliding_window` の AND
（masking_utils.py:98-99）で、どちらも「距離 `limit - col` が `0 <= dist < 512`」— **self を含む
幅 512 で厳密同値**。1-shot 形の帯 mask（`(cols <= rows) & ((rows - cols) < window)`）とも同じ
包含なので、`config.sliding_window` をそのまま `window` attrs へ宣言する。

容量は層種別で分ける（ADR 0066 追記 9）: full スロットは記号 `C`（context 生成時に選ぶ —
決定 3）・sliding スロットは `window` 実数。ring は window ちょうどで閉じるので、記号のままだと
`C - window` 行が常に死蔵になる（C=8192 で約 180MiB）。詳細は {@link states_plan}。

## mask は「trace を通すためだけ」に居る

`attention` の第 4 入力は 1-shot 形と同じ層種別 2 本の加算 mask 辞書
（{@link gemma4.export.additive_causal_mask} / `additive_sliding_mask`）だが、states 形では
causal も窓も**述語計算**になるので mask tensor は要らない。手術が全 35 本から mask 入力を
落とし、Tmax² 定数 2 本と `sym_prefix_slice` を刈る。{@link assert_ir_form_decode} はその残骸が
1 本も残っていないことを見る（残ると「誰も読まない 4.5MiB」が配布物に居座る）。

## 混成量子化は 1-shot と同じ（embedding i8 × linear i4）

丸めは**参照・golden の採取より前**（ADR 0006）で、対象の割り付けも格納の指定も
{@link gemma4.export.quantize_wrapper} をそのまま使う。{@link DecodeChunkWrapper} が 1-shot
ラッパの**派生**なのはこのため — モジュール FQN 空間が同一でないと、対象述語
（`is_int8_module` / `is_int4_module`）も scale 台帳のキーも流用できない。

## 出力レイアウト

    outputs/series/gemma4-e2b-decode/model.safetensors         重み・定数 + karume_ir
    outputs/series/gemma4-e2b-decode/io.<case>.safetensors     無 pad 全長の入出力
    outputs/series/gemma4-e2b-decode/greedy.<case>.safetensors greedy 継続 K step の期待列

`io.*` は 1-shot 形と同じキー規約（`input.<グラフ入力名>` / `output.<位置>`）。`greedy.*` は
**decode 検収の正本**で、`prompt` i32 `[T]` / `expected` i32 `[K]` / `margin` f32 `[K]` を持つ。
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import nn
from torch.export import Dim
from torch.nn import functional

from _shared.decode_series import _write_greedy, assert_case_room, positions_for
from _shared.paths import SERIES_ROOT
from gemma4 import export as one_shot
from gemma4 import ple, provenance
from karume.artifacts import staged_publication
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION, normalize_boundary_tensor
from karume.ir import IrGraph, IrNode
from karume.ops import ARGMAX_OP, ATTENTION_OP, LINEAR_OP, STATE_APPEND_OP
from karume.pipeline import export_module, publish_model
from karume.shapes import declared_shape
from karume.shards import resolve_shards
from karume.states import StateAttentionSpec, StatesPlan, to_states_form

#: 生成物の既定の置き場（1-shot 系列とは別ディレクトリ — グラフの形が違う別資産）。
#: 素材（`--model-dir` の既定）は 1-shot 形と同じで、綴りは共通の CLI 骨組みが持つ
#: （{@link gemma4.export.series_parser}）。
DEFAULT_OUT_DIR = SERIES_ROOT / "gemma4-e2b-decode"

#: グラフ入力の名前（chunk ラッパの forward 引数名）。{@link assert_ir_form_decode} が
#: 「この 2 本だけ」を見るための綴り。
INPUT_IDS = one_shot.INPUT_IDS
POSITION_IDS = "position_ids"

#: 物理 chunk 次元の記号名（IR 上の名前は `symbol_names` が決める — Dim 名と同じ綴りにする）。
SEQ_SYMBOL = "M"

#: 差し替えた RoPE 表のラッパ基準 FQN（{@link rope_table_keys} が initializer キーを組む起点）。
ROPE_TABLE_MODULE = "model.model.rotary_emb"

#: state スロットの容量記号（ADR 0066 決定 2 / 追記 7 — 束縛点は `createGenerationContext`）。
#: **値 shape には現れない states 専用記号**で、export 時に容量を焼かないための席。
CAPACITY_SYMBOL = "C"

#: RoPE 表の位置数（= 表引きできる絶対位置の上限）。`gemma4.export.SYM_MAX`（768）とは
#: **別の概念**で値も違う — あちらは 1 chunk の最大行数、こちらは prompt + 生成を通した絶対
#: 位置の上限（decode では `pastLength + M` がこれを超えられない）。長ケース T=598 に
#: {@link GREEDY_STEPS} を足しても余る点に置く。表は層種別 2 組（f32 で
#: 1024×(256+512)×2 = 6MiB）なので、上げるならその代償を承知の上で上げる。
ROPE_TABLE_POSITIONS = 1024

#: greedy golden の継続 step 数（ADR 0068 決定 4 の出口を多 step で踏む）。
GREEDY_STEPS = 16

#: 各 step の top1 − top2 logit 差の下限。GPU 実行の偏差で greedy の列が割れないことの保証で、
#: **恒真でない**（強く決まらない継続を持つケースはここで落ちる）。落ちたケースは K を下げずに
#: {@link GREEDY_CASES} から外す — 余裕の無い列を golden にすると「GPU 側が正しくても赤」の
#: 門になる。
#:
#: MUST: 検収門の前提（`e2e_gemma4_greedy_test.ts` — `minMargin > 2 × atol`・atol = 1e-2）
#: **より上**に置く。下だと「台本は採るが門の前提で落ちる」ケースが作れてしまい、門の
#: 「ここが落ちるのは台本と資産が食い違ったときだけ」が成立しない（Codex 波 H 指摘 H-04）。
MARGIN_FLOOR = 2.5e-2

#: greedy golden を採るケース（{@link gemma4.export.GOLDEN_CASES} の部分集合）。
#: io golden は 3 ケース全部で採る（こちらは 1 step ぶんなので余裕門が要らない）。
#:
#: NOTE: 現状は**全ケース採用の暫定**（minicpm5 は実測後の確定集合）。w4 込みの実走で
#: {@link assert_greedy_margins} が落としたケースをここから外して再走する — 門は全ケースを
#: 測り終えてから 1 度に掛かるので、1 回の実走で外す対象が揃う。
#: MUST: 外した結果 `GREEDY_EXPECTATIONS` の期待が 1 種類になる集合にはしない
#: （`_sanity` の定数出力検出線が恒真化する — tests が固定）。
GREEDY_CASES: tuple[str, ...] = ("capital-en", "capital-ja", "context-en")

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
    """`Gemma4TextRotaryEmbedding` と同じ呼び出し規約を持つ**位置表の引き**モジュール。

    上流は `self.rotary_emb(hidden_states, position_ids, layer_type)` を層種別ごとに呼び
    （modeling_gemma4.py:1713）、`(cos, sin)` の 2 タプル（各 `[B, seq, head_dim]`）を受ける。
    ここはその規約だけを満たし、`inv_freq` からの三角関数計算を**表の gather** に置き換える。
    層種別で `head_dim` が違う（sliding 256 / full 512）ので、表は種別ごとに別の組。

    MUST: 表は素の属性で持つ（バッファにしない）— `rope.lift_rope_buffers` が既存モデルで
    やっている降格と同じ形で、torch.export は lifted tensor constant として拾う。属性名を
    `<layer_type>_cos_table` にするのは上流の `<layer_type>_inv_freq` と同じ流儀
    （どの層種別の表かが initializer の FQN から読める）。
    MUST: 戻り値に dtype キャストを挟まない。上流は `cos.to(dtype=x.dtype)` を最後に置くが、
    表は模型と同じ dtype で作る（{@link swap_rope_table} が突合する）ので恒等であり、
    恒等キャストを書くと export へ `_to_copy` が漏れうる。
    """

    def __init__(self, tables: Mapping[str, tuple[torch.Tensor, torch.Tensor]]) -> None:
        super().__init__()
        self.layer_types = tuple(tables)
        for layer_type, (cos, sin) in tables.items():
            setattr(self, f"{layer_type}_cos_table", cos)
            setattr(self, f"{layer_type}_sin_table", sin)

    def tables_for(self, layer_type: str) -> tuple[torch.Tensor, torch.Tensor]:
        """層種別の `(cos 表, sin 表)`。

        MUST: 未知の層種別は fail loudly。既定の表へ落とすと「full 層が sliding の表を引く」
        （値は出るが位置の周波数が違う）形が黙って通る。
        """
        if layer_type not in self.layer_types:
            raise KeyError(
                f"layer_type '{layer_type}' の RoPE 表が無い（持っているのは {self.layer_types}）"
            )
        return getattr(self, f"{layer_type}_cos_table"), getattr(self, f"{layer_type}_sin_table")

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor, layer_type: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`position_ids[B,seq]` で層種別の表を引く（`x` は規約上の引数 — 値は読まない）。"""
        del x
        cos, sin = self.tables_for(layer_type)
        return (
            functional.embedding(position_ids, cos),
            functional.embedding(position_ids, sin),
        )


def unique_layer_types(config: Any) -> tuple[str, ...]:
    """`config.layer_types` の**出現順**の重複除去（表と検査の走査順を 1 箇所に閉じる）。"""
    return tuple(dict.fromkeys(config.layer_types))


def build_rope_table(
    rotary: nn.Module, probe: torch.Tensor, positions: int, layer_types: Sequence[str]
) -> RopeTable:
    """元の rotary を層種別ごとに `arange(positions)` で 1 回呼び、その cos / sin を表にする。

    MUST: 表は**元実装の出力そのもの**で作る（`inv_freq` から作り直さない）— 作り直すと
    `attention_scaling` や rope_type 別の初期化（full 層は `proportional`）を写し損ねる経路が
    でき、値同一の突合が「同じ式を 2 回書いて比べただけ」の恒真に近づく。
    """
    tables: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_type in layer_types:
        with torch.no_grad():
            cos, sin = rotary(
                probe, position_ids=torch.arange(positions).unsqueeze(0), layer_type=layer_type
            )
        tables[layer_type] = (cos[0].detach().clone(), sin[0].detach().clone())
    return RopeTable(tables)


def assert_rope_table_matches(
    rotary: nn.Module,
    table: RopeTable,
    probe: torch.Tensor,
    positions: int,
    layer_types: Sequence[str],
) -> None:
    """表引きが元実装と**完全一致**（タプル構成・shape・dtype・全要素）であることを見る。

    MUST: 恒真でない。表は `arange` 1 本から作るので、「添字 → 行」の引き方が正しいことは
    値の同一でしか確かめられない。踏む 3 種は ①連続（`0..4`）②オフセット付き（`3..9` —
    decode の `pastLength + row`）③非単調（並びに依らず**値で**引くことの確認）。
    MUST: 全 layer_type を踏む。片方だけ見ると、もう片方の表が丸ごと無検査になる
    （sliding と full は head_dim も rope_type も違う）。
    """
    probes = (
        torch.arange(5).unsqueeze(0),
        torch.arange(3, 10).unsqueeze(0),
        torch.tensor([[positions - 1, 0, 17, 2]], dtype=torch.int64),
    )
    for layer_type in layer_types:
        for position_ids in probes:
            with torch.no_grad():
                reference = rotary(probe, position_ids=position_ids, layer_type=layer_type)
                actual = table(probe, position_ids, layer_type)
            where = f"{layer_type} position_ids={position_ids.tolist()}"
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
    layer_types = unique_layer_types(model.config)
    probe = torch.zeros(1, 1, int(model.config.hidden_size), dtype=torch.float32)
    table = build_rope_table(rotary, probe, positions, layer_types)
    assert_rope_table_matches(rotary, table, probe, positions, layer_types)
    model.model.rotary_emb = table
    return table


class DecodeChunkWrapper(one_shot.Gemma4Wrapper):
    """1-shot ラッパ（{@link gemma4.export.Gemma4Wrapper}）の chunk 形
    （`(input_ids, position_ids) → (logits, token)`）版。

    MUST: **派生**で持つ（同じ構成を書き直さない）— 量子化の対象述語
    （`is_int8_module` / `is_int4_module`）も scale 台帳のキーもモジュール FQN 空間の上に
    載っているので、`model` / `per_layer` の綴りが 1-shot と同一であることが再利用の条件。
    MUST: `attention_mask` は 1-shot と同じ層種別 2 本の辞書（省略すると transformers 側で
    mask が組まれ、`cache_position` 由来の値が畳み込みの葉に混ざる）。trace を通すためだけの
    存在で、手術が落とす。
    MUST: `use_cache=False`。True だと `DynamicCache` が生えて KV を出力へ返す形になり、
    「state はグラフの通常 I/O にしない」（ADR 0066 決定 1）と正面から食い違う。
    MUST: 出力順は `[logits, token]`。ランタイム側は slot 番号でこの 2 本を読む。
    """

    def forward(  # type: ignore[override]
        self, input_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length = input_ids.shape[1]
        mask = {
            one_shot.FULL_ATTENTION: one_shot.additive_causal_mask(length),
            one_shot.SLIDING_ATTENTION: one_shot.additive_sliding_mask(length, self.sliding_window),
        }
        embeds = self.model.model.embed_tokens(input_ids)
        stacked = ple.per_layer_inputs(self.per_layer, input_ids, self.per_layer_scale)
        logits = self.model(
            inputs_embeds=embeds,
            per_layer_inputs=stacked,
            attention_mask=mask,
            position_ids=position_ids,
            use_cache=False,
        ).logits
        return logits, logits.argmax(-1, keepdim=True)


@dataclass(frozen=True)
class ChunkVariant:
    """chunk 系列 2 本（logits opt-in / token-only）の**差分だけ**を持つ記述子。

    載せるのは {@link export_series} が実際に分岐する 4 点 — 系列の置き場・組むラッパ・出口の
    形・自分の golden を採るか。手術も混成量子化も門の順序も両系列で同一なので、それ以外の席は
    作らない（席を作ると「片方だけ違う経路」が書けるようになる）。
    """

    #: 生成物の既定の置き場（系列ごとに別ディレクトリ — グラフの形が違う別資産）。
    out_dir: Path
    #: 組む chunk ラッパ（{@link DecodeChunkWrapper} の継承鎖のどれか）。
    wrapper: type[DecodeChunkWrapper]
    #: 出口が token-only（`last_row` 入力が増え、出力が token 1 本 — ADR 0068 決定 4）か。
    token_only: bool
    #: 自分の golden（io + greedy）を採るか。token-only 系列は logits opt-in 系列の greedy
    #: 記録を検収門がそのまま流用するので採らない（{@link gemma4.export_token} の docstring）
    #: — 出口の形とは独立の系列ごとの方針なので、`token_only` から導かない。
    goldens: bool


#: この台本の系列（logits opt-in 形 — 全 M 行の logits と token を出す）。
DECODE = ChunkVariant(
    out_dir=DEFAULT_OUT_DIR, wrapper=DecodeChunkWrapper, token_only=False, goldens=True
)


def load_wrapper(
    variant: ChunkVariant, model_dir: Path, *, positions: int = ROPE_TABLE_POSITIONS
) -> DecodeChunkWrapper:
    """実重みを f32 で読み、RoPE を表引きへ差し替えた export 可能な chunk ラッパを返す。

    variant で変わるのは最後に組むラッパ型だけ — 素材の読み方（3 つの等価検査を含む）と
    RoPE の差し替えは chunk 系列で同一。
    """
    model, tables = one_shot.load_model_and_tables(model_dir)
    # 検査席の PLE 表を落とす理由は {@link gemma4.export.build_wrapper} と同じ
    # （分割 35 本を PLE の唯一の正本にして、量子化の対象網羅を言える形にする）。
    del model.model.embed_tokens_per_layer
    swap_rope_table(model, positions)
    return variant.wrapper(model, tables).eval()


def rope_table_keys(wrapper: nn.Module) -> tuple[str, ...]:
    """RoPE 表の initializer テンソルキー（= ラッパ基準の FQN）。

    表引きにした以上、cos / sin 表は `embedding` の**重みスロット**で消費される — つまり
    圧縮格納の**適格集合に入る**（`emit.eligible_compressed_initializers`）。既定 i8 のまま
    書くと「重みスロット適格なのに per-channel scale が無い」で fail loudly になるので、
    `"f32"` の明示指定で圧縮既定から外す（`emit._plan_weight_dtype` の f32 明示 = 除外）。

    MUST: 除外であって「量子化し忘れ」ではない。位置表を丸めると RoPE の角度がずれ、
    長い位置ほど誤差が効く（重みの丸めと違って誤差が位置に沿って蓄積する）。1-shot 形では
    同じ表が `mul` の入力に畳まれていて適格外だったので、この席は decode 形にだけ要る。

    NOTE: 綴りは wrapper 内での rotary の FQN（{@link ROPE_TABLE_MODULE}）+ {@link RopeTable}
    の属性名。ズレたら `emit._plan_weight_dtype` の未知キー検査が落とす。
    """
    table = wrapper.model.model.rotary_emb
    return tuple(
        f"{ROPE_TABLE_MODULE}.{layer_type}_{part}_table"
        for layer_type in table.layer_types
        for part in ("cos", "sin")
    )


def first_shared_layer(config: Any) -> int:
    """KV 共有が始まる層番号（上流 `Gemma4TextAttention.__init__` と同じ式）。"""
    return int(config.num_hidden_layers) - int(getattr(config, "num_kv_shared_layers", 0))


def kv_owner_layers(config: Any) -> dict[str, int]:
    """layer_type → その型の KV を書き出す層（上流 `store_full_length_kv` と同じ規則）。

    上流は「共有開始より前の層のうち、その layer_type で**最後**の層」の k / v を
    `shared_kv_states[layer_type]` へ置き、共有層はそれを読む（modeling_gemma4.py:1287）。
    E2B は sliding → 層 13 / full → 層 14。

    MUST: 所有層の無い layer_type が共有側に居たら落とす（上流なら `shared_kv_states` の
    KeyError になる形で、層構成の前提そのものが崩れている）。
    """
    layer_types = list(config.layer_types)
    boundary = first_shared_layer(config)
    owners: dict[str, int] = {}
    for layer, layer_type in enumerate(layer_types[:boundary]):
        owners[layer_type] = layer
    orphans = sorted(set(layer_types[boundary:]) - set(owners))
    if orphans:
        raise AssertionError(
            f"共有層の layer_type {orphans} を書き出す層が共有開始 {boundary} より前に無い"
            "（層構成の前提が崩れている）"
        )
    return owners


def slot_layers(config: Any) -> tuple[int, ...]:
    """層 → その層が読む KV スロットの**所有層**番号。

    共有開始より前は自分自身、以後は layer_type ごとの所有層（{@link kv_owner_layers}）。
    """
    owners = kv_owner_layers(config)
    boundary = first_shared_layer(config)
    return tuple(
        layer if layer < boundary else owners[layer_type]
        for layer, layer_type in enumerate(config.layer_types)
    )


def attention_nodes(graph: IrGraph, config: Any) -> list[IrNode]:
    """nodes 順の attention を返す（**出現順 = 層順**という前提そのものを検査してから）。

    検査は q の D 軸と `config.layer_types` の突合。Gemma 4 は層種別で head_dim が違う
    （sliding 256 / full 512）ので、並びが層順でなければ 5 層ごとの 512 が別の位置に出る。

    MUST: 前提が崩れたまま spec を組むと、別層の KV を読むスロット割りが**形も型も合ったまま**
    通る（沈黙誤値）。本数の食い違いも同じ理由で落とす — 取りこぼした層は手術されずに
    mask 形のまま残り、その層だけが chunk 局所の causal になる。
    """
    layer_types = list(config.layer_types)
    nodes = [node for node in graph.nodes if node.op == ATTENTION_OP]
    if len(nodes) != len(layer_types):
        raise AssertionError(f"attention が {len(nodes)} 本（{len(layer_types)} 層と一致しない）")
    for layer, (node, layer_type) in enumerate(zip(nodes, layer_types, strict=True)):
        depth = one_shot._attention_depth(config, layer_type)
        found = declared_shape(graph, node.ins[0])[3]
        if found != depth:
            raise AssertionError(
                f"attention[{layer}] ({layer_type}): q の D 軸が {found} —"
                f" この層種別の head_dim {depth} と違う"
                "（nodes 順が層順でない可能性 — スロット割りの前提が崩れる）"
            )
    return nodes


def states_plan(
    graph: IrGraph, config: Any, *, capacity_symbol: str = CAPACITY_SYMBOL
) -> StatesPlan:
    """層種別と KV 共有を反映した手術指定（スロットは**所有層ぶんだけ**）。

    - 層 0..14（自前 KV）: 自分の k / v スロット。sliding は `window = sliding_window`・
      full は window 無し。
    - 層 15..34（共有読者）: 自分のスロットを作らず、layer_type の所有層のスロットを読む。
      sliding の読者は同じ `window` を宣言する（states.py の `_register` が完全一致を検査）。

    容量は層種別で分ける: **sliding スロットは `window` 実数**（ring は window ちょうどで
    閉じる — ADR 0067 決定 4 の「全読者が past を読み終えてから append」保証。記号のままだと
    容量 C ぶんの行を確保して window 超の行が死蔵になるだけで、読める行集合は変わらない）・
    **full スロットは記号のまま**（全 context を保持する側だけが「容量を実行時に選ぶ」自由 —
    ADR 0066 決定 3 — を必要とする）。
    """
    nodes = attention_nodes(graph, config)
    window = int(config.sliding_window)
    return StatesPlan(
        capacity_symbol=capacity_symbol,
        attentions=tuple(
            StateAttentionSpec(
                output=node.outs[0],
                k_slot=slot_name(owner, "k"),
                v_slot=slot_name(owner, "v"),
                window=window if layer_type == one_shot.SLIDING_ATTENTION else None,
                capacity=window if layer_type == one_shot.SLIDING_ATTENTION else None,
            )
            for node, layer_type, owner in zip(
                nodes, config.layer_types, slot_layers(config), strict=True
            )
        ),
    )


#: token-only 既定出口（ADR 0068 決定 4）の last_row 入力名。綴りは wrapper の forward 引数名
#: そのもの（torch.export がグラフ入力名に採る）— 正本はここで、`export_token` と
#: `assert_ir_form_decode(token_only=True)` の両方が参照する。
TOKEN_ONLY_LAST_ROW = "last_row"


def last_row_for(ids: torch.Tensor) -> torch.Tensor:
    """無 pad 全長 prompt の最終有効行の添字 `[T−1]`（token-only 出口の `last_row` 入力）。

    {@link positions_for} と対の「1 chunk で全 prompt を食う」形の値で、例示入力にも sanity の
    全長 forward にも同じものが要る（綴りが 2 箇所に割れると、行選択が最後の行を指さない形が
    片方だけで作れてしまう）。
    """
    return torch.tensor([int(ids.shape[1]) - 1], dtype=torch.int64)


def assert_layer_type_count(config: Any) -> None:
    """層種別の宣言本数を確かめる（形検査の**前口上** — 3 形すべてが最初に通る）。

    ここが崩れていると後段の `zip(..., strict=True)` が「本数が違う」という真因から遠い
    例外で落ちるので、入口・出口の検査より前に置く（{@link assert_ir_form_decode} /
    {@link gemma4.export_product.assert_ir_form_product} の並び）。
    """
    layers = int(config.num_hidden_layers)
    declared = len(list(config.layer_types))
    if declared != layers:
        raise AssertionError(f"config.layer_types が {declared} 本（{layers} 層と違う）")


def assert_single_row_lm_head(
    graph: IrGraph,
    producer: Mapping[str, IrNode],
    start: IrNode,
) -> None:
    """**1 行 lm_head の固定**（Codex 波 H 指摘 H-01）。

    行ごとの lm_head と行選択は可換なので「全行 lm_head → softcap → 行選択」でも token 列は
    一致する — ADR 0068 の実効（lm_head 1 行・`[M,V]` バッファ消滅）はこの構造検査でしか
    固定できない。`start`（token-only なら argmax ノード / 製品形なら logits 出力の生産者）
    から softcap 鎖（div/tanh/mul — いずれも `ins[0]` が本流）を遡って最初の linear が
    lm_head で、その入力が `[1,1,H]`（行 1 本）かつ祖先に `last_row` 入力を持つこと。
    """
    node = start
    for _ in range(8):
        source = producer.get(node.ins[0])
        if source is None:
            raise AssertionError(
                f"出力の祖先（'{node.ins[0]}'）が途切れた — lm_head（linear）に届かない"
            )
        node = source
        if node.op == LINEAR_OP:
            break
    else:
        raise AssertionError(f"出力の 8 段以内に lm_head（{LINEAR_OP}）が無い")
    row_shape = declared_shape(graph, node.ins[0])
    if list(row_shape[:2]) != [1, 1]:
        raise AssertionError(
            f"lm_head の入力が {row_shape} — [1,1,H]（選択済みの 1 行）でない"
            "（全行 lm_head へ退行している）"
        )
    input_names = {spec.name for spec in graph.inputs}
    frontier = [node.ins[0]]
    seen: set[str] = set()
    reachable: set[str] = set()
    while frontier:
        name = frontier.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in input_names:
            reachable.add(name)
            continue
        upstream = producer.get(name)
        if upstream is not None:
            frontier.extend(upstream.ins)
    if TOKEN_ONLY_LAST_ROW not in reachable:
        raise AssertionError(
            f"lm_head の入力が `{TOKEN_ONLY_LAST_ROW}` に依存しない"
            f"（到達した入力: {sorted(reachable)}）— 行選択が lm_head より前に居ない"
        )


def assert_ir_form_common(
    graph: IrGraph,
    config: Any,
    storage_expectation: Mapping[str, int],
    *,
    seq_symbol: str,
    capacity_symbol: str,
) -> dict[str, Any]:
    """入口・出口**以外**の全規律（3 形が共有する本体）。

    decode 系列は入口 / 出口が 3 形ある（logits opt-in / token-only / 製品）が、states の
    形・層種別との対応・スロット割り・記号・手術の残骸・格納内訳は**どの形でも同一**である。
    ここを 1 本にしておかないと、片方だけ検査が痩せていく（実際に製品形の追加時、同じ規律を
    独立に綴った結果 90 行が二重化していた）。

    golden の突合では捕まらない性質だけを並べる:

    - attention が 1 本でも従来形（mask 込み 4 本）で残ると、その層だけ過去を見ない
    - `window` の有無が層種別と食い違うと、full 層が窓外を捨てる / sliding 層が窓を無視する
    - スロットの割り当てを間違えると、共有層が別の層の KV を読む（形も型も合う）
    - full スロットの容量が記号でなく数値だと、context 生成時に容量を選べない（ADR 0066
      決定 3）。sliding スロットは逆に window 実数ちょうどでないと死蔵行が戻る
      （states_plan docstring — 層種別で理由が違うので検査も分けている）
    - `sym_prefix_slice` / `sin` が残ると、誰も読まない Tmax 定数や畳み残しが配布物に居座る
    - 圧縮の適格判定を外した重みは**黙って f32 のまま**残る（`emit._plan_weight_dtype` の
      既定側は静かに落とす経路を持つ）ので、格納 dtype の本数を数えないと気づけない

    MUST: head_dim は層種別で引く（`hidden_size / num_attention_heads` = 192 はどちらとも
    違う）。片方の値で全層を見ると、もう片方の層が丸ごと無検査になる。
    MUST: 呼び手は {@link assert_layer_type_count} を**先に**通していること（本数が合って
    いる前提で `zip(..., strict=True)` を張る）。
    """
    heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    layer_types = list(config.layer_types)
    layers = int(config.num_hidden_layers)
    owners = slot_layers(config)
    window = int(config.sliding_window)

    attentions = [node for node in graph.nodes if node.op == ATTENTION_OP]
    if len(attentions) != layers:
        raise AssertionError(f"attention が {len(attentions)} 本（{layers} 層と一致しない）")
    for layer, (node, layer_type, owner) in enumerate(
        zip(attentions, layer_types, owners, strict=True)
    ):
        where = f"attention[{layer}] ({layer_type})"
        if len(node.ins) != 3:
            raise AssertionError(
                f"{where}: ins が {len(node.ins)} 本 — states 形は q / k / v の 3 本ちょうど"
                "（4 本なら mask 込みの従来形が残っている）"
            )
        expected_states = {"k": slot_name(owner, "k"), "v": slot_name(owner, "v")}
        if node.states != expected_states:
            raise AssertionError(f"{where}: states 欄が {node.states}（期待 {expected_states}）")
        expected_window = window if layer_type == one_shot.SLIDING_ATTENTION else None
        if node.attrs.get("window") != expected_window:
            raise AssertionError(
                f"{where}: attrs の window が {node.attrs.get('window')}"
                f"（期待 {expected_window}）— 層種別と窓の対応が崩れている"
            )
        query, key, value = (declared_shape(graph, name) for name in node.ins)
        if query[1] != heads or key[1] != kv_heads or value[1] != kv_heads:
            raise AssertionError(
                f"{where}: head 軸が {[query[1], key[1], value[1]]} —"
                f" 真の GQA 形 {[heads, kv_heads, kv_heads]} でない"
                "（k / v が H まで広がっていれば repeat_kv が実体化している）"
            )
        depth = one_shot._attention_depth(config, layer_type)
        if query[3] != depth or key[3] != depth or value[3] != depth:
            raise AssertionError(
                f"{where}: D 軸が {[query[3], key[3], value[3]]} — この層種別の head_dim"
                f" {depth} と違う"
            )

    boundary = first_shared_layer(config)
    expected_slots = sorted(
        slot_name(layer, part) for layer in range(boundary) for part in ("k", "v")
    )
    appends = [node for node in graph.nodes if node.op == STATE_APPEND_OP]
    written = sorted(node.states["slot"] for node in appends)
    if written != expected_slots:
        raise AssertionError(
            f"`{STATE_APPEND_OP}` の書き先が {len(written)} 本 — 所有層のスロット"
            f" {len(expected_slots)} 本を 1 本ずつでない"
            f"（欠落 {sorted(set(expected_slots) - set(written))} /"
            f" 余剰 {sorted(set(written) - set(expected_slots))}）"
        )
    if sorted(graph.states) != expected_slots:
        raise AssertionError(
            f"states 宣言が {sorted(graph.states)} — 所有層 {boundary} 本の k / v"
            f" {len(expected_slots)} 本でない"
        )
    for layer in range(boundary):
        layer_type = layer_types[layer]
        depth = one_shot._attention_depth(config, layer_type)
        # 容量の検査は層種別で分ける MUST — 記号一色に緩めると「全部数値に焼かれた資産」
        # （容量を実行時に選べない — ADR 0066 決定 3）が素通りし、数値一色に緩めると full の
        # 容量自由が黙って消える。sliding は window 実数ちょうど（states_plan docstring）。
        sliding = layer_type == one_shot.SLIDING_ATTENTION
        slot_shape = [1, kv_heads, window if sliding else capacity_symbol, depth]
        for part in ("k", "v"):
            name = slot_name(layer, part)
            slot = graph.states[name]
            if slot.dtype != "f32" or list(slot.shape) != slot_shape:
                kind = (
                    "sliding は window 実数ちょうど"
                    if sliding
                    else "full の容量は記号のまま残す MUST"
                )
                raise AssertionError(
                    f"states['{name}'] が {slot.dtype} {list(slot.shape)} —"
                    f" f32 {slot_shape} でない（{kind}）"
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
        "state_append_nodes": len(appends),
        "slots": len(expected_slots),
        "kv_owners": kv_owner_layers(config),
        "heads": [heads, kv_heads, kv_heads],
        "head_dim": {
            layer_type: one_shot._attention_depth(config, layer_type)
            for layer_type in unique_layer_types(config)
        },
        "window": window,
        "storage": dict(sorted(storage.items())),
    }


def assert_ir_form_decode(
    graph: IrGraph,
    config: Any,
    storage_expectation: Mapping[str, int],
    *,
    seq_symbol: str = SEQ_SYMBOL,
    capacity_symbol: str = CAPACITY_SYMBOL,
    token_only: bool = False,
) -> dict[str, Any]:
    """states 形 decode グラフの形を検査する（**数値が合ったまま静かに壊れる**性質を全部見る）。

    `token_only` は ADR 0068 決定 4 の**既定出口形**（`export_token` — 入力に `last_row` が
    増え、出力が argmax token 1 本になる）。入口・出口以外の検査（states / 層種別 / 残骸 /
    格納内訳）は両形で同一なので、この 1 関数が両方を受ける（別関数に割ると片方だけ検査が
    痩せていく）。本体は {@link assert_ir_form_common} で、製品形
    （{@link gemma4.export_product.assert_ir_form_product}）とも共有する。

    ここが見るのは入口・出口だけ:

    - グラフ入力に mask が残ると「ホストが毎 chunk T² を作って渡す」別物になる
    - 出口が argmax でなければ decode 出口（ADR 0068 決定 4）ではない
    - `token_only` では行選択が lm_head より**前**に居ること（{@link assert_single_row_lm_head}）
    """
    assert_layer_type_count(config)

    names = [spec.name for spec in graph.inputs]
    expected_inputs = (
        [INPUT_IDS, POSITION_IDS, TOKEN_ONLY_LAST_ROW] if token_only else [INPUT_IDS, POSITION_IDS]
    )
    if names != expected_inputs:
        raise AssertionError(
            f"グラフ入力が {names} — {expected_inputs} でない"
            "（mask が畳み込まれずに入力へ残っている可能性）"
        )
    expected_outputs = 1 if token_only else 2
    if len(graph.outputs) != expected_outputs:
        kind = (
            "token-only 出口は token の 1 本"
            if token_only
            else "decode 出口は logits / token の 2 本"
        )
        raise AssertionError(f"IR 出力が {len(graph.outputs)} 本（{kind}）")
    producer = {out: node for node in graph.nodes for out in node.outs}
    token_source = producer.get(graph.outputs[-1])
    if token_source is None or token_source.op != ARGMAX_OP:
        found = "ノード出力でない" if token_source is None else token_source.op
        raise AssertionError(
            f"出力 {len(graph.outputs) - 1} の供給元が {found} — `{ARGMAX_OP}` でない"
            "（ADR 0068 決定 4 の decode 出口）"
        )
    if token_only:
        assert_single_row_lm_head(graph, producer, token_source)

    return assert_ir_form_common(
        graph,
        config,
        storage_expectation,
        seq_symbol=seq_symbol,
        capacity_symbol=capacity_symbol,
    )


def _write_container(
    graph: IrGraph,
    tensors: Mapping[str, torch.Tensor],
    path: Path,
    *,
    weight_dtype: str,
    weight_scales: Mapping[str, torch.Tensor],
    weight_dtype_overrides: Mapping[str, str],
) -> IrGraph:
    """手術済みグラフを書いて検証する（公開の 3 段は `pipeline.publish_model` に預ける）。

    `export_to_file` は export → 書き出しが 1 本道で手術を挟む隙間が無いので、書き出し以降
    だけを core の入口から呼ぶ。**規則も原子性も再実装しない** — states 節・順序・shape の
    検査も、shard 分割（ADR 0070 決定 1）とその据え替え・後始末も core が持つ。

    刈り込みで死んだ initializer（mask の Tmax² 定数）は格納テンソルからも落とす —
    `write_model` は宣言と格納の**完全一致**を要求する。scale 台帳（`weight_scales`）は
    刈られた重みの分が残っていてもよい（emit は計画した本数しか引かない）が、
    `weight_dtype_overrides` の側は**未知キーで fail loudly** になる — linear の重みが手術で
    消える形は起きてはならないので、その門はそのまま効かせる。
    """
    declared = {init.tensor for init in graph.initializers.values()}
    stored = {name: tensor for name, tensor in tensors.items() if name in declared}
    return publish_model(
        path,
        graph,
        stored,
        weight_dtype=weight_dtype,
        weight_scales=weight_scales,
        weight_dtype_overrides=weight_dtype_overrides,
    )


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


def _export_args(
    variant: ChunkVariant, ids: torch.Tensor, seq: Dim
) -> tuple[tuple[torch.Tensor, ...], tuple[Any, ...]]:
    """例示入力と `dynamic_shapes` の組（token-only は `last_row` が 1 本増える）。

    `last_row` は静的 `[1]`（動的次元なし）— 実行時スカラだが形は 1 要素で固定なので、
    記号次元を与えると M と紐づいた別物になる。
    """
    args: tuple[torch.Tensor, ...] = (ids, positions_for(ids))
    shapes: tuple[Any, ...] = ({1: seq}, {1: seq})
    if variant.token_only:
        return (*args, last_row_for(ids)), (*shapes, None)
    return args, shapes


def export_series(
    variant: ChunkVariant,
    model_dir: Path,
    out_dir: Path,
    *,
    sym_max: int = one_shot.SYM_MAX,
    positions: int = ROPE_TABLE_POSITIONS,
    steps: int = GREEDY_STEPS,
    reference: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """variant の states 形 IR コンテナ（golden を採る系列なら io / greedy も）を書き、要約を返す。

    MUST: 生成物は作業席へ書き、**全ての門**（形検査・margin 門・1-shot 期待表との sanity）を
    通してから据える。門より前に final へ置くと、落ちた実走が「検収門を通れる資産」を残す
    （据え替えと後片付けの規律は core の原語 {@link karume.artifacts.staged_publication}）。
    MUST: `steps` を読むのは golden を採る系列だけ（token-only 系列には greedy 記録が無い）。
    MUST: `reference` を読むのは golden を**採らない**系列だけ — 流用する greedy golden の
    置き場で、出所記録（{@link gemma4.provenance}）がその digest を容器と束ねる。
    """
    wrapper = load_wrapper(variant, model_dir, positions=positions)
    # MUST: 丸めは参照・golden の採取より前（ADR 0006）— 後だと参照だけが元の重みで動く。
    int8, int4, scales = one_shot.quantize_wrapper(wrapper)
    cases = one_shot.build_cases(model_dir, sym_max, wrapper.sliding_window)
    greedy_cases = tuple(case for case in cases if case[0] in GREEDY_CASES)
    # io / sanity は prompt を丸ごと 1 回引くだけ（継続分の位置は要らない）ので steps=0 で見る。
    assert_case_room(cases, 0, positions)
    if variant.goldens:
        assert_case_room(greedy_cases, steps, positions)
    # MUST: 流用する golden の検めは席へ入る**前**（席の外に掛かる前提 — 落ちるなら
    # 数十分の export を始める前に落とす）。読むだけで、参照系列には 1 バイトも書かない。
    reference_goldens = (
        {} if variant.goldens else provenance.assert_reference_goldens(reference, greedy_cases)
    )

    # 例示入力は最長ケース（記号次元の 0/1 特殊化から遠い）。min=2 は同じ理由、max は
    # mask の Tmax 畳み込みの評価点（手術で刈るので配布物には残らないが、trace は通る）。
    _, example_ids = max(cases, key=lambda case: case[1].shape[1])
    seq = Dim(SEQ_SYMBOL, min=2, max=sym_max)
    args, shapes = _export_args(variant, example_ids, seq)
    print("[export] torch.export → 変換", file=sys.stderr, flush=True)
    graph, tensors = export_module(
        wrapper,
        args,
        dynamic_shapes=shapes,
        symbol_names=(SEQ_SYMBOL,),
        preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    )
    config = wrapper.model.config
    print("[export] states 形へ手術 → 書き出し", file=sys.stderr, flush=True)
    surgical = to_states_form(graph, states_plan(graph, config))

    # 据える単位は**どちらの系列も「系列ディレクトリ丸ごと」**。golden を採る系列は
    # 新 model + 旧 greedy の混成を作れない形にするため、採らない系列は容器と出所記録
    # （{@link gemma4.provenance}）が食い違った組を作れない形にするため。
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with staged_publication(out_dir) as staged:
        # ディレクトリの席は書き手が作る（原語は席を作らない — path しか渡さない）。
        staged.mkdir()
        container = staged / one_shot.MODEL_FILE
        # 格納は既定 i8 + linear を 1 本ずつ i4（向きの根拠は 1-shot 台本の docstring —
        # tied 実体の FQN を書く前に知らずに済む側）。RoPE 表は既定 i8 の適格に入ってしまう
        # ので f32 を明示して外す（{@link rope_table_keys}）。
        verified = _write_container(
            surgical,
            tensors,
            container,
            weight_dtype="i8",
            weight_scales=scales,
            weight_dtype_overrides={
                **dict.fromkeys(int4.scales, "i4"),
                **dict.fromkeys(rope_table_keys(wrapper), "f32"),
            },
        )
        form = assert_ir_form_decode(
            verified,
            config,
            {"i8": len(int8.scales), "i4": len(int4.scales)},
            token_only=variant.token_only,
        )

        if variant.goldens:
            print("[io] 全長 forward", file=sys.stderr, flush=True)
            io_written = _write_io(wrapper, verified, cases, staged)
            greedy_written, tokens, margins = _write_greedy(
                wrapper, greedy_cases, staged, steps=steps, floor=MARGIN_FLOOR
            )
            first = {name: continuation[0] for name, continuation in tokens.items()}
        else:
            # golden を採らない系列は sanity のためだけに全長を 1 回引く（行選択は T−1）。
            print("[sanity] 全長 forward", file=sys.stderr, flush=True)
            first = {}
            for name, ids in cases:
                with torch.no_grad():
                    token = wrapper(ids, positions_for(ids), last_row_for(ids))
                first[name] = int(token[0, 0, 0])

        # 第 1 継続 token を 1-shot 台本の期待表と突き合わせる（機構横断の突合 — 1-shot 形と
        # chunk 形の台本は別物なので、同じ重み・同じ prompt で 1 位が一致することが両者の
        # 交差検証になる）。MUST: 公開より前に評価する（落ちたら作業席ごと消える）。
        tokenizer = one_shot.load_tokenizer(model_dir)
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

        if not variant.goldens:
            # 出所記録は容器と**同じ席**へ置く（据え替えが 1 回なので、新しい容器 + 古い
            # 記録という組が作れない）。名乗る系列名は据えた後の名前 — 席の名前ではない。
            print("[provenance] チェックポイント指紋 → reference.json", file=sys.stderr, flush=True)
            record = provenance.build_record(out_dir, model_dir, reference_goldens)
            provenance.write_record(staged, record)

    # 刈り込み本数と golden 3 種は golden を採る系列だけの欄（token-only の要約はコンテナ
    # 1 本ぶんなので増やさない）。
    pruned: dict[str, Any] = (
        {"pruned_initializers": len(graph.initializers) - len(verified.initializers)}
        if variant.goldens
        else {}
    )
    written: dict[str, Any] = (
        {"io": io_written, "greedy": greedy_written, "greedy_steps": steps}
        if variant.goldens
        else {}
    )
    measured: dict[str, Any] = (
        {
            "margin_min": {name: min(values) for name, values in sorted(margins.items())},
            "continuation": {
                name: tokenizer.decode(continuation)
                for name, continuation in sorted(tokens.items())
            },
        }
        if variant.goldens
        else {}
    )
    # 逆に出所記録は golden を採らない系列だけの欄（束ねた golden が要約からも見える）。
    bound: dict[str, Any] = {} if variant.goldens else {"reference": record["reference"]}
    return {
        "dir": str(out_dir),
        "nodes": len(verified.nodes),
        "outputs": len(verified.outputs),
        "initializers": len(verified.initializers),
        **pruned,
        "model_bytes": sum(p.stat().st_size for p in resolve_shards(out_dir / one_shot.MODEL_FILE)),
        "ops": sorted(verified.required_ops),
        "symbols": list(verified.symbols),
        **written,
        "case_lengths": {name: int(ids.shape[1]) for name, ids in cases},
        "quantized": {"i8": int8.describe(), "i4": int4.describe()},
        "form": form,
        **bound,
        **measured,
        "sanity": sanity,
    }


def run_variant_cli(variant: ChunkVariant, description: str, argv: Sequence[str] | None) -> None:
    """variant の CLI を組んで走らせる（chunk 系列 2 本の入口はこれ 1 本）。

    骨組みは 1-shot 台本と共有する（{@link gemma4.export.series_parser}）— chunk 系列が足すのは
    位置表の大きさ `--positions` と、golden を採る系列だけの `--steps`、採らない系列だけの
    `--reference`（流用する golden 系列の置き場）。
    """
    parser = one_shot.series_parser(description, variant.out_dir)
    parser.add_argument("--positions", type=int, default=ROPE_TABLE_POSITIONS)
    if variant.goldens:
        parser.add_argument("--steps", type=int, default=GREEDY_STEPS)
    else:
        parser.add_argument("--reference", type=Path, default=DEFAULT_OUT_DIR)
    one_shot.run_series_cli(parser, partial(export_series, variant), argv)


def main(argv: Sequence[str] | None = None) -> None:
    run_variant_cli(DECODE, __doc__.split("\n\n")[0], argv)


if __name__ == "__main__":
    main()
