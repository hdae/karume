"""実重み Gemma 4 E2B を **製品グラフ**（PLE 外出し + 最終行 logits 出口）へ書き出す台本。

ADR [0083](../../../docs/decisions/0083-generation-api-surface.md) 決定 6（出口は
`logits[1,1,V]`・sampling はホスト維持）と ADR
[0085](../../../docs/decisions/0085-ple-host-gather.md)（PLE をホスト gather へ外出し）を
**1 回の再 export に載せる**（案 α — ADR 0083 Consequences / backlog now の段 1b）。

    uv run --with 'transformers==5.14.1' python -m gemma4.export_product

## 既存 2 系列との差は入口 2 本・出口 1 本

chunk 系列の経路（素材の読み方・RoPE のホスト供給・KV 共有の手術・混成量子化・門の順序）は
{@link gemma4.export_decode} の中核をそのまま通す（import して使う — 同じ規律を 2 箇所に
書かない）。差分は 3 点だけ:

- 入力に **`per_layer_inputs[1,M,35,256]` f32** が増える。PLE lookup は `input_ids` **だけ**を
  引数に取る純粋な行 lookup なので、グラフから外してホストが供給する通常のグラフ入力に
  なる（ADR 0085 決定 6 — ランタイムの契約は 1 文字も変わらない）。容器からは i8 35 表
  2,240MiB + per-row scale 35MiB が消える。
- 入力に **`last_row[1]` i32** が増える（token-only 系列と同じ行選択の配線 —
  {@link gemma4.export_decode.TOKEN_ONLY_LAST_ROW}）。
- 出口は **`logits[1,1,V]`**（最終**行**のみ・argmax なし）。token-only 系列の
  `TokenOnlyChunkWrapper` から argmax を外した形そのもので、sampling / RNG はホストが持つ
  （ADR 0083 決定 6 の MUST）。prefill の読み戻しは `[1,M,V]` 形の 32MiB から 1MiB へ減る。

## PLE sidecar（token-major + vocab レンジ shard）

グラフから外した 35 表は **token-major**（`[token][layer][256]` i8 + `[token][layer]` の
per-row scale）へ再配置し、**vocab の範囲**で shard する（ADR 0085 決定 1 / 2）。1 token の
PLE が連続 1 読み（8,960B + 35 scale）になる形で、後から「キャッシュから行だけ読む」
（同 ADR の代替案 b）へ移るときに再 export も再アップロードも要らない。1 shard の大きさは
書き手の容量（{@link karume.shards.SHARD_DATA_CAPACITY}）をそのまま使う。

NOTE: sidecar は **IR コンテナではない**（付帯資産 — ADR 0038 §2 の extras と同じ位置づけで、
読み手は `parseSafetensors` の厳格リーダ）。したがって ADR 0081 の読み手契約 1（shard 0 =
グラフ shard・データ節空）は掛からず、掛かるのはバイト上限だけ。連番の綴りは
{@link karume.shards.shard_name} を共有する。

MUST: 再配置は **ビット同一**であること（{@link assert_ple_sidecar}）。i8 値と per-row scale の
対応を 1 層ずらしても形も型も dtype も合うので、`ple.py` の分割検査と同じ理由で
`torch.equal` の門が要る。

## golden はこの系列でも作らない（logits opt-in 系列との交差 parity が門）

期待列は `gemma4-e2b-decode/greedy.<case>.safetensors` を流用する（token-only 系列と同じ形 —
{@link gemma4.export_token} の docstring）。**ホスト側 PLE gather + `argmax(logits)`** で回した
列が既存 golden と厳密一致することが段 1b の合格線で（検収門は
`packages/models/tests/e2e_gemma4_product_test.ts`）、どの資産の組で見るべきかの束ねは
{@link gemma4.provenance} の出所記録が持つ。

PLE 逆量子化のビット一致は `ple.probe.safetensors`（散点 token の `per_layer_inputs` を
**35 表経路の torch が**計算したもの = グラフに残していたら embedding op が出していた値）を
TS 側の loader の出力と突き合わせて見る。

## 出力レイアウト

    outputs/series/gemma4-e2b-product/model.safetensors        重み・定数 + karume_ir
    outputs/series/gemma4-e2b-product/ple.json                 sidecar の索引（shard の token 範囲）
    outputs/series/gemma4-e2b-product/ple-NNNNN-of-NNNNN.safetensors  PLE sidecar
    outputs/series/gemma4-e2b-product/ple.probe.safetensors    逆量子化ビット一致の参照
    outputs/series/gemma4-e2b-product/reference.json           出所記録（指紋 + 流用 golden）
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.export import Dim
from torch.nn import functional

from _shared.decode_series import assert_case_room, positions_for
from _shared.paths import SERIES_ROOT
from gemma4 import export as one_shot
from gemma4 import export_decode as decode
from gemma4 import ple, provenance
from karume.artifacts import staged_publication
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
from karume.ir import IrGraph
from karume.ops import ARGMAX_OP, EMBEDDING_OP
from karume.pipeline import export_module
from karume.quantize import quantize_to_int8
from karume.shapes import declared_shape
from karume.shards import SHARD_DATA_CAPACITY, resolve_shards, shard_name
from karume.states import to_states_form

#: 生成物の既定の置き場（既存 2 系列とは別ディレクトリ — 入口も出口も違う別資産）。
DEFAULT_OUT_DIR = SERIES_ROOT / "gemma4-e2b-product"

#: 流用する greedy 期待列の置き場（正本は logits opt-in 系列 — token-only 系列と同じ参照先）。
REFERENCE_DIR = decode.DEFAULT_OUT_DIR

#: ホストが供給する PLE のグラフ入力名（ラッパの forward 引数名そのもの — torch.export が
#: グラフ入力名に採る）。綴りの正本はここで、{@link assert_ir_form_product} と TS 側の検収門が
#: 参照する。
PER_LAYER_INPUTS = "per_layer_inputs"

#: PLE sidecar の代表 path（実ファイルは常に連番 — {@link karume.shards.shard_name}）と、
#: 索引・逆量子化参照のファイル名。読み手は `packages/models/src/gemma/ple.ts`。
PLE_FILE = "ple.safetensors"
PLE_INDEX_FILE = "ple.json"
PLE_PROBE_FILE = "ple.probe.safetensors"

#: sidecar shard のテンソルキー（`values` = token-major i8 / `scales` = per-row f32）。
PLE_VALUES_KEY = "values"
PLE_SCALES_KEY = "scales"

#: `ple.probe.safetensors` のテンソルキー。
PROBE_TOKENS_KEY = "tokens"
PROBE_INPUTS_KEY = "per_layer_inputs"

#: 索引と shard メタデータの版（読み手が知らない版を黙って読まないための欄）。
PLE_SCHEMA = 1

#: shard の safetensors `__metadata__` に置く索引の写し（1 本だけで自己記述になる形）。
PLE_METADATA_KEY = "karume_ple"


class ProductChunkWrapper(decode.DecodeChunkWrapper):
    """`(input_ids, RoPE 4 本, per_layer_inputs, last_row) → logits[1,1,V]` の製品ラッパ。

    MUST: `DecodeChunkWrapper` の**派生**（モジュール FQN 空間の同一性 — 量子化の対象述語
    `is_int8_module` / `is_int4_module` と scale 台帳のキーの再利用条件。
    `export_decode.DecodeChunkWrapper` の docstring と同じ理由）。
    MUST: `self.per_layer`（PLE 35 表）は**量子化の対象としてだけ**持ち、forward からは引かない
    — PLE はホストが供給するグラフ入力になったので、export の前にモジュールごと落とす
    （{@link export_series}）。持ったまま export すると 9.4GB の未使用 state_dict を抱える。
    MUST: 行選択は `F.embedding`（token-only 形と同文 — 上流の `logits_to_keep` にテンソルを
    渡す形は `aten.index.Tensor` = IR 語彙外の advanced indexing に落ちる）。lm_head と
    softcap の 3 行は {@link gemma4.export_token.TokenOnlyChunkWrapper} の逐語同型で、
    **違いは argmax を置かないことだけ**（ADR 0083 決定 6 — sampling はホスト維持）。
    """

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        rope_sliding_attention_cos: torch.Tensor,
        rope_sliding_attention_sin: torch.Tensor,
        rope_full_attention_cos: torch.Tensor,
        rope_full_attention_sin: torch.Tensor,
        per_layer_inputs: torch.Tensor,
        last_row: torch.Tensor,
    ) -> torch.Tensor:
        length = input_ids.shape[1]
        mask = {
            one_shot.FULL_ATTENTION: one_shot.additive_causal_mask(length),
            one_shot.SLIDING_ATTENTION: one_shot.additive_sliding_mask(length, self.sliding_window),
        }
        embeds = self.model.model.embed_tokens(input_ids)
        tables = decode.bound_rope(
            rope_sliding_attention_cos,
            rope_sliding_attention_sin,
            rope_full_attention_cos,
            rope_full_attention_sin,
        )
        with self.model.model.rotary_emb.bound(tables):
            hidden = self.model.model(
                inputs_embeds=embeds,
                per_layer_inputs=per_layer_inputs,
                attention_mask=mask,
                position_ids=None,
                use_cache=False,
            ).last_hidden_state
        # 行選択のあと [1,1,H] へ上げてから lm_head へ通す（token-only 形と同文 — 1 行 lm_head の
        # 構造検査が「選択済みの 1 行」を見るのはこの形が前提）。
        rowed = functional.embedding(last_row, hidden[0]).unsqueeze(0)
        logits = self.model.lm_head(rowed)
        cap = float(self.model.config.final_logit_softcapping)
        return torch.tanh(logits / cap) * cap


#: {@link decode.load_wrapper} へ渡す variant。**読まれるのは `wrapper` 欄だけ**（素材の読み方と
#: RoPE の差し替えはラッパ型に依らない）。残り 3 欄は {@link decode.export_series} 用の分岐で、
#: 本台本はそちらを通らない — 製品形の差（入口 2 本増・出口 logits・sidecar）は
#: `ChunkVariant` の 4 欄に載らないので、系列の駆動はこのモジュールが持つ。
_LOAD_VARIANT = decode.ChunkVariant(
    out_dir=DEFAULT_OUT_DIR, wrapper=ProductChunkWrapper, token_only=False, goldens=False
)


def load_wrapper(model_dir: Path) -> ProductChunkWrapper:
    """実重みを f32 で読み、RoPE を受け渡し口へ差し替えた製品ラッパを返す。

    素材の読み方（3 つの等価検査を含む）と RoPE の差し替えは chunk 系列 3 本で同一なので、
    {@link decode.load_wrapper} をそのまま通す（同じ規律を 2 箇所に書かない）。
    """
    # variant の `wrapper` 欄が組む型そのものが返る（{@link _LOAD_VARIANT}）。
    return decode.load_wrapper(_LOAD_VARIANT, model_dir)


# ---- PLE sidecar -----------------------------------------------------------


def ple_token_bytes(layers: int, dim: int) -> int:
    """1 token ぶんの sidecar バイト数（i8 値 `layers × dim` + f32 scale `layers`）。

    E2B は 35 × 256 + 35 × 4 = **9,100 バイト/token**。token-major の狙いそのもので、
    1 token の PLE がこの長さの**連続 1 読み**になる（ADR 0085 決定 1）。
    """
    return layers * dim + layers * 4


def ple_table_rows(tables: Sequence[torch.nn.Module], vocab_size: int) -> int:
    """PLE 35 分割の行数（= sidecar の token 行数）。

    MUST: `config.vocab_size_per_layer_input` から取らない —
    {@link gemma4.export.load_model_and_tables} が**検査席の行数**（`ple.PLE_PROBE_ROWS` = 8）へ
    差し替えた後の値なので、そちらを読むと
    8 行の sidecar が形も型も合ったまま書かれる（実際に 1 度踏んだ）。行数の正本は
    「実際に読み込んだ分割表」だけ。

    MUST: 主 embedding の vocab 行数と一致することを見る（ADR 0085 決定 5 の**書き手側**の半分）
    — ホスト loader はこの 2 つの一致を前提に id 空間を突き合わせるので、食い違ったまま
    配ると読み手が拒否するか、拒否を緩めた瞬間に「別 token の有効な行」を引く。
    """
    rows = {int(table.weight.shape[0]) for table in tables}
    if len(rows) != 1:
        raise AssertionError(f"PLE 分割表の行数が揃っていない: {sorted(rows)}")
    found = rows.pop()
    if found != vocab_size:
        raise AssertionError(
            f"PLE 分割表の行数 {found} が主 embedding の vocab 行数 {vocab_size} と違う"
            "（ホスト gather の id 空間が主 embedding と別物になる — ADR 0085 決定 5）"
        )
    return found


def plan_ple_shards(
    tokens: int, token_bytes: int, limit: int = SHARD_DATA_CAPACITY
) -> tuple[tuple[int, int], ...]:
    """vocab を容量内の**最小本数**へ割り、行数を均した `[start, stop)` の列。

    方針は {@link karume.shards.pack_shards} と同型（最小本数 k を先に決めてから均す —
    端数 shard を作らない・ADR 0081）。単位が **1 token 固定長**なので貪欲の最小本数は
    `ceil(総量 / 容量)` に一致し、均しも行数の等分で済む（対の原子性も可変長も無い）。
    `limit` の既定は書き手の容量 {@link karume.shards.SHARD_DATA_CAPACITY}（sidecar は遅延
    ロードで触った shard だけをホストへ読むので、容量がそのまま 1 回の読みの上限になる）。

    MUST: 1 token が単独で容量を超える形は fail loudly（層数か層当たり次元が想定外に大きい
    — 分割の粒度をこれ以上細かくできないので、黙って容量を破るしかなくなる）。
    """
    if tokens < 1:
        raise ValueError(f"PLE sidecar の token 数 {tokens} が 1 以上でない")
    if token_bytes < 1:
        raise ValueError(f"PLE sidecar の 1 token {token_bytes} バイトが 1 以上でない")
    per_shard = limit // token_bytes
    if per_shard < 1:
        raise ValueError(f"1 token {token_bytes:,} バイトが shard 上限 {limit:,} を超える")
    count = -(-tokens // per_shard)
    ranges: list[tuple[int, int]] = []
    start = 0
    for opened in range(count):
        rows = -(-(tokens - start) // (count - opened))
        ranges.append((start, start + rows))
        start += rows
    return tuple(ranges)


def ple_probe_tokens(tokens: int, ranges: Sequence[tuple[int, int]]) -> tuple[int, ...]:
    """逆量子化ビット一致の参照に使う**散点** token id（shard 境界の両側 + 両端 + 中央）。

    連続 N 個だと 1 つの shard しか踏まず、別 shard だけの取り違え（範囲の off-by-one・
    scale の層ずれ）が門に映らない（{@link gemma4.ple.probe_rows} と同じ理由の shard 版）。
    """
    picked: list[int] = []
    for start, stop in ranges:
        picked.extend((start, start + 1, stop - 1))
    picked.extend((0, 1, tokens // 2, tokens - 2, tokens - 1))
    return tuple(sorted({token for token in picked if 0 <= token < tokens}))


def quantized_ple_tables(
    tables: Sequence[torch.nn.Module], scales: Mapping[str, torch.Tensor]
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """PLE 35 表を i8 値 `[V,D]` と per-row scale `[V]` へ落とす（f32 実体は手放す）。

    MUST: scale は fake-quant が使ったものを**そのまま**渡す
    （{@link karume.quantize.quantize_to_int8} の MUST — amax から引き直すと f32 の丸めで
    1ulp 動きうる）。
    MUST: 1 表ずつ f32 を手放す — 35 表の f32 は 9.4GB あり、i8 側の 2.35GB と同時に生かすと
    ピークが 2GB 級で増える（台本の RAM 予算は README の 24GB）。
    """
    values: list[torch.Tensor] = []
    row_scales: list[torch.Tensor] = []
    for index, table in enumerate(tables):
        key = f"{one_shot.PER_LAYER_PREFIX}{index}.weight"
        if key not in scales:
            raise AssertionError(f"PLE 表 {index} の scale 台帳キー '{key}' が無い")
        scale = scales[key]
        values.append(quantize_to_int8(table.weight.data, scale))
        row_scales.append(scale.reshape(-1).clone())
        # 実体を手放す（席は呼び手が `del wrapper.per_layer` で落とす）。
        table.weight.data = torch.empty(0)
    return values, row_scales


def _shard_metadata(index: Mapping[str, Any], start: int, stop: int) -> dict[str, str]:
    """shard 1 本の `__metadata__`（索引の写し + 自分の token 範囲）。

    索引（`ple.json`）と shard の自己申告が食い違う組み合わせを読み手が落とせる形にする —
    片方だけ作り直した資産は、範囲がずれたまま**形も dtype も合う**（= 別 token の有効な行を
    引く・ADR 0085 決定 5 と同じ沈黙誤値）。
    """
    record = {key: value for key, value in index.items() if key != "shards"}
    return {PLE_METADATA_KEY: json.dumps({**record, "start": start, "stop": stop})}


def write_ple_shards(
    values: Sequence[torch.Tensor],
    row_scales: Sequence[torch.Tensor],
    ranges: Sequence[tuple[int, int]],
    out_dir: Path,
    index: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """token-major の shard 列を書き、索引の `shards` 欄を返す。

    `values` / `row_scales` は**層ごと**（table-major）に持っているので、shard 1 本ぶんの
    行範囲を 35 本から `stack` して token-major へ転置する。転置がビット同一であることは
    {@link assert_ple_sidecar} が書いたバイト列を読み直して見る。
    """
    written: list[dict[str, Any]] = []
    total = len(ranges)
    for position, (start, stop) in enumerate(ranges, start=1):
        name = shard_name(PLE_FILE, position, total)
        block = torch.stack([table[start:stop] for table in values], dim=1).contiguous()
        scale = torch.stack([row[start:stop] for row in row_scales], dim=1).contiguous()
        save_file(
            {PLE_VALUES_KEY: block, PLE_SCALES_KEY: scale},
            str(out_dir / name),
            metadata=_shard_metadata(index, start, stop),
        )
        written.append({"file": name, "start": start, "stop": stop})
        print(
            f"[ple] {name} tokens [{start}, {stop}) / {(out_dir / name).stat().st_size:,} バイト",
            file=sys.stderr,
            flush=True,
        )
    return written


def ple_index(tokens: int, layers: int, dim: int, embed_scale: float) -> dict[str, Any]:
    """`ple.json` の骨（`shards` 欄は {@link write_ple_shards} が埋める）。"""
    return {
        "schema": PLE_SCHEMA,
        "tokens": tokens,
        "layers": layers,
        "dim": dim,
        "embedScale": embed_scale,
        "shards": [],
    }


def assert_ple_sidecar(
    out_dir: Path, index: Mapping[str, Any], probe: Sequence[int], reference: torch.Tensor
) -> None:
    """書いた sidecar から probe token を組み直し、35 表経路と**ビット一致**することを見る。

    参照側（`reference`）は {@link gemma4.ple.per_layer_inputs} が fake-quant 済みの 35 表から
    組んだ `[1,P,35,256]` そのもの — つまり **PLE をグラフに残していたら embedding op が
    出していた値**（i8 格納 + per-row scale の逆量子化は fake-quant の値を厳密に復元する
    — ADR 0019 の ±127 論証）。再配置側は**書いたバイト列を読み直し**、ホストと同じ順序
    （`f32(i8) * scale` → `* embed_scale`）で組む。

    MUST: `torch.equal`（ビット一致）で見る — scale の対応を 1 層ずらしても、shard の範囲を
    1 行ずらしても、形も型も dtype も合ったまま**別 token の有効な行**が出る。
    MUST: 読み直す（in-memory の配列を突き合わせない）— 転置は正しいのに書き出しの
    dtype / 形 / 順序が違う形を、この門が受け止める最後の位置。
    """
    layers = int(index["layers"])
    dim = int(index["dim"])
    embed_scale = float(index["embedScale"])
    expected_shape = (1, len(probe), layers, dim)
    if tuple(reference.shape) != expected_shape:
        raise AssertionError(f"参照 {tuple(reference.shape)} が {expected_shape} でない")

    rebuilt = torch.zeros(expected_shape, dtype=torch.float32)
    covered = 0
    for shard in index["shards"]:
        start, stop = int(shard["start"]), int(shard["stop"])
        path = out_dir / str(shard["file"])
        with safe_open(str(path), framework="pt") as handle:
            stored = sorted(handle.keys())
            if stored != sorted([PLE_VALUES_KEY, PLE_SCALES_KEY]):
                raise AssertionError(f"{path.name}: テンソルキーが {stored}")
            values = handle.get_slice(PLE_VALUES_KEY)
            scales = handle.get_slice(PLE_SCALES_KEY)
            rows = stop - start
            if values.get_shape() != [rows, layers, dim]:
                raise AssertionError(
                    f"{path.name}: '{PLE_VALUES_KEY}' が {values.get_shape()} —"
                    f" token-major [{rows}, {layers}, {dim}] でない"
                )
            if scales.get_shape() != [rows, layers]:
                raise AssertionError(
                    f"{path.name}: '{PLE_SCALES_KEY}' が {scales.get_shape()} —"
                    f" [{rows}, {layers}] でない"
                )
            for position, token in enumerate(probe):
                if not start <= token < stop:
                    continue
                row = token - start
                quantized = values[row : row + 1].to(torch.float32)
                scale = scales[row : row + 1].to(torch.float32).unsqueeze(-1)
                rebuilt[0, position] = (quantized * scale)[0] * embed_scale
                covered += 1
    if covered != len(probe):
        raise AssertionError(
            f"probe {len(probe)} 本のうち {covered} 本しか shard の範囲に載っていない"
            "（索引の [start, stop) が vocab を覆っていない）"
        )
    if not torch.equal(rebuilt, reference):
        worst = float((rebuilt - reference).abs().max())
        raise AssertionError(
            "PLE sidecar の再配置が 35 表経路とビット一致しない"
            f"（最大絶対差 {worst}）— i8 値と per-row scale の対応か token 範囲がずれている"
        )


# ---- 形検査 ----------------------------------------------------------------


def assert_ir_form_product(
    graph: IrGraph,
    config: Any,
    storage_expectation: Mapping[str, int],
    ple_rows: int,
    *,
    seq_symbol: str = decode.SEQ_SYMBOL,
    capacity_symbol: str = decode.CAPACITY_SYMBOL,
) -> dict[str, Any]:
    """製品グラフの形を検査する（**数値が合ったまま静かに壊れる**性質を全部見る）。

    states / 層種別 / 残骸 / 格納の各節は {@link decode.assert_ir_form_common} に預ける
    （3 形が共有する本体 — 段 1b では「入口 / 出口が排他だから」と独立に綴っていたが、
    共有部分だけを関数に括れば入口 / 出口の引数化は要らない）。ここが綴るのは製品形に固有の
    入口・出口だけである。

    製品形に固有の 3 本（PLE 外出しと logits 出口の実証）:

    - グラフ入力に **`per_layer_inputs` が居る**こと。居なければ PLE がグラフに残っている
      （常駐は 2,240MiB 戻り、ホスト gather の入力は誰にも読まれない）。
    - PLE 表の形（`[ple_rows, hidden_size_per_layer_input]`）を引く `embedding` が**1 本も
      無い**こと。入力が増えても表が残る形（両方通る配線）が書けてしまう。
      MUST: `ple_rows` は**引数で受ける** — `config.vocab_size_per_layer_input` は
      {@link gemma4.export.load_model_and_tables} が検査席の 8 行へ差し替えた後の値なので、
      config から引くとこの検査が実質空振りになる（実際に 1 度踏んだ）。
    - `embedding` ノードが**主 embedding + 行選択**の 2 本ちょうどであること。本数で見るのは、
      PLE の一部だけが残る形（35 本中 1 本の刈り漏れ）を数で捕まえるため。RoPE は
      ホスト供給の入力になったので、表を引く `embedding` はもう 1 本も居ない。
    - 出口が **argmax でない**こと。`argmax` が 1 本でも残っていれば sampling の余地が消える
      （ADR 0083 決定 6 — GPU 側は最終行 logits まで）。
    - 出力の宣言 shape が `[1, 1, vocab_size]`（最終**行**のみ）であること。全行 logits へ
      退行しても token 列は一致するので、構造検査でしか固定できない（ADR 0068 の実効）。
    """
    decode.assert_layer_type_count(config)
    layers = int(config.num_hidden_layers)
    ple_dim = int(config.hidden_size_per_layer_input)

    expected_inputs = [
        decode.INPUT_IDS,
        *decode.ROPE_INPUTS,
        PER_LAYER_INPUTS,
        decode.TOKEN_ONLY_LAST_ROW,
    ]
    names = [spec.name for spec in graph.inputs]
    if names != expected_inputs:
        raise AssertionError(
            f"グラフ入力が {names} — {expected_inputs} でない"
            "（PLE がグラフに残っている / mask や position_ids が畳み込まれずに入力へ残って"
            "いる可能性）"
        )
    decode.assert_rope_inputs(graph, config, seq_symbol=seq_symbol)
    per_layer_spec = next(spec for spec in graph.inputs if spec.name == PER_LAYER_INPUTS)
    expected_shape = [1, seq_symbol, layers, ple_dim]
    if per_layer_spec.dtype != "f32" or list(per_layer_spec.shape) != expected_shape:
        raise AssertionError(
            f"'{PER_LAYER_INPUTS}' が {per_layer_spec.dtype} {list(per_layer_spec.shape)} —"
            f" f32 {expected_shape} でない"
        )

    # PLE 表は **embedding の重みスロット**（`WEIGHT_SLOTS[embedding] = 0`）で消費される形
    # でしか居られないので、そこを名指しで見る。initializer の shape だけを見ると別スロットの
    # 同形テンソル（tiny 模型の `mlp.down_proj` 等）に当たる。
    embeddings = [node for node in graph.nodes if node.op == EMBEDDING_OP]
    residents = sorted(
        node.ins[0]
        for node in embeddings
        if list(declared_shape(graph, node.ins[0])) == [ple_rows, ple_dim]
    )
    if residents:
        raise AssertionError(
            f"PLE 表の形 [{ple_rows}, {ple_dim}] を引く `{EMBEDDING_OP}` が"
            f" {len(residents)} 本残っている: {residents[:4]}"
            "（ホスト gather へ外に出し切れていない）"
        )
    # 主 embedding（tied lm_head と同一実体）1 本 + 最終行の行選択 1 本。PLE が 1 本でも
    # 残ればここが増え、RoPE の表引きが戻ってもここが増える（どちらも退行の印）。
    expected_embeddings = 2
    if len(embeddings) != expected_embeddings:
        raise AssertionError(
            f"`{EMBEDDING_OP}` が {len(embeddings)} 本 — 主 embedding 1 + 行選択 1 の"
            f" {expected_embeddings} 本でない"
        )

    if len(graph.outputs) != 1:
        raise AssertionError(
            f"IR 出力が {len(graph.outputs)} 本（製品出口は logits の 1 本 — ADR 0083 決定 6）"
        )
    if ARGMAX_OP in graph.required_ops:
        raise AssertionError(
            f"`{ARGMAX_OP}` がグラフに残っている"
            "（製品出口は最終行 logits で、sampling / argmax はホスト側 — ADR 0083 決定 6）"
        )
    logits_shape = list(declared_shape(graph, graph.outputs[0]))
    expected_logits = [1, 1, int(config.vocab_size)]
    if logits_shape != expected_logits:
        raise AssertionError(
            f"出力 0 の宣言 shape が {logits_shape} — {expected_logits}（最終行のみ）でない"
        )

    # 1 行 lm_head の固定は token-only 形と**同じ構造検査**（decode 側の 1 本を通す）。
    producer = {out: node for node in graph.nodes for out in node.outs}
    logits_source = producer.get(graph.outputs[0])
    if logits_source is None:
        raise AssertionError(f"出力 0 ('{graph.outputs[0]}') がノード出力でない")
    decode.assert_single_row_lm_head(graph, producer, logits_source)

    form = decode.assert_ir_form_common(
        graph,
        config,
        storage_expectation,
        seq_symbol=seq_symbol,
        capacity_symbol=capacity_symbol,
    )
    return {**form, "embedding_nodes": len(embeddings), "logits": logits_shape}


# ---- 系列 ------------------------------------------------------------------


def export_series(
    model_dir: Path,
    out_dir: Path,
    *,
    sym_max: int = one_shot.SYM_MAX,
    reference: Path = REFERENCE_DIR,
) -> dict[str, Any]:
    """製品グラフのコンテナ + PLE sidecar + 出所記録を書き、要約を返す。

    MUST: 生成物は作業席へ書き、**全ての門**（sidecar のビット一致・形検査・1-shot 期待表との
    sanity）を通してから据える。門より前に final へ置くと、落ちた実走が「検収門を通れる資産」を
    残す（据え替えと後片付けの規律は core の原語 {@link karume.artifacts.staged_publication}）。
    MUST: 流用する golden の検めは席へ入る**前**（落ちるなら数十分の export を始める前に落とす）。
    MUST: PLE を落とす順序は「参照と各ケースの入力を 35 表経路で組む → i8 へ落とす →
    `per_layer` を落とす → export」。逆にすると参照側が i8 経路で作られ、ビット一致の門が
    「同じ向きに間違った 2 つ」を突き合わせる形になる。
    """
    wrapper = load_wrapper(model_dir)
    # MUST: 丸めは参照・golden の採取より前（ADR 0006）— 後だと参照だけが元の重みで動く。
    int8, int4, scales = one_shot.quantize_wrapper(wrapper)
    specs = decode.rope_specs(wrapper.model.config)
    cases = one_shot.build_cases(model_dir, sym_max, wrapper.sliding_window)
    greedy_cases = tuple(case for case in cases if case[0] in decode.GREEDY_CASES)
    assert_case_room(cases, 0, decode.max_position(wrapper.model.config))
    reference_goldens = provenance.assert_reference_goldens(reference, greedy_cases)

    config = wrapper.model.config
    layers = int(config.num_hidden_layers)
    dim = int(config.hidden_size_per_layer_input)
    # MUST: 行数の正本は読み込んだ分割表（config の欄は検査席の 8 行へ差し替え済み —
    # {@link ple_table_rows}）。
    tokens = ple_table_rows(wrapper.per_layer, int(config.vocab_size))
    embed_scale = float(wrapper.per_layer_scale)
    ranges = plan_ple_shards(tokens, ple_token_bytes(layers, dim))
    probe = ple_probe_tokens(tokens, ranges)

    # PLE をグラフから外す前に、①逆量子化ビット一致の参照 ②各ケースのグラフ入力 を
    # **35 表経路**（台本 3 本が通す {@link gemma4.ple.per_layer_inputs}）で 1 度だけ組む。
    print("[ple] 35 表経路で参照とケース入力を組む", file=sys.stderr, flush=True)
    with torch.no_grad():
        probe_reference = ple.per_layer_inputs(
            wrapper.per_layer, torch.tensor([list(probe)], dtype=torch.int64), embed_scale
        )
        case_inputs = {
            name: ple.per_layer_inputs(wrapper.per_layer, ids, embed_scale) for name, ids in cases
        }

    print("[ple] 35 表 → i8 値 + per-row scale", file=sys.stderr, flush=True)
    values, row_scales = quantized_ple_tables(wrapper.per_layer, int8.scales)
    # 実体は上で手放し済み。席そのものを落として、未使用の 35 表を export へ持ち込まない。
    del wrapper.per_layer

    example_name, example_ids = max(cases, key=lambda case: case[1].shape[1])
    seq = Dim(decode.SEQ_SYMBOL, min=2, max=sym_max)

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with staged_publication(out_dir) as staged:
        # ディレクトリの席は書き手が作る（原語は席を作らない — path しか渡さない）。
        staged.mkdir()
        index = ple_index(tokens, layers, dim, embed_scale)
        index["shards"] = write_ple_shards(values, row_scales, ranges, staged, index)
        assert_ple_sidecar(staged, index, probe, probe_reference)
        (staged / PLE_INDEX_FILE).write_text(
            json.dumps(index, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        save_file(
            {
                PROBE_TOKENS_KEY: torch.tensor(list(probe), dtype=torch.int32).contiguous(),
                PROBE_INPUTS_KEY: probe_reference.contiguous(),
            },
            str(staged / PLE_PROBE_FILE),
        )
        # sidecar は据えたので i8 実体を手放す（以降は模型ぶんの RAM だけで export へ入る）。
        values.clear()
        row_scales.clear()

        print("[export] torch.export → 変換", file=sys.stderr, flush=True)
        example_rope = decode.rope_args(specs, positions_for(example_ids))
        graph, tensors = export_module(
            wrapper,
            (
                example_ids,
                *example_rope,
                case_inputs[example_name],
                decode.last_row_for(example_ids),
            ),
            dynamic_shapes=(*({1: seq} for _ in range(2 + len(example_rope))), None),
            symbol_names=(decode.SEQ_SYMBOL,),
            preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
        )
        print("[export] states 形へ手術 → 書き出し", file=sys.stderr, flush=True)
        surgical = to_states_form(graph, decode.states_plan(graph, config))
        verified = decode._write_container(
            surgical,
            tensors,
            staged / one_shot.MODEL_FILE,
            weight_dtype="i8",
            weight_scales=scales,
            weight_dtype_overrides=dict.fromkeys(int4.scales, "i4"),
        )
        # i8 の initializer は **PLE 35 表を外したぶんだけ減る** — 残るのは主 embedding
        # （tied lm_head と同一実体）1 本。台帳の本数から引くので、外し漏れは本数で落ちる。
        form = assert_ir_form_product(
            verified,
            config,
            {"i8": len(int8.scales) - layers, "i4": len(int4.scales)},
            tokens,
        )

        print("[sanity] 全長 forward", file=sys.stderr, flush=True)
        first: dict[str, int] = {}
        for name, ids in cases:
            with torch.no_grad():
                logits = wrapper(
                    ids,
                    *decode.rope_args(specs, positions_for(ids)),
                    case_inputs[name],
                    decode.last_row_for(ids),
                )
            first[name] = int(logits[0, 0].argmax())

        # 第 1 継続 token を 1-shot 台本の期待表と突き合わせる（機構横断の突合 — 台本が別物
        # なので、同じ重み・同じ prompt で 1 位が一致することが交差検証になる）。
        # MUST: 公開より前に評価する（落ちたら作業席ごと消える）。
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

        # 出所記録は容器と**同じ席**へ置く（据え替えが 1 回なので、新しい容器 + 古い記録という
        # 組が作れない）。名乗る系列名は据えた後の名前 — 席の名前ではない。
        print("[provenance] チェックポイント指紋 → reference.json", file=sys.stderr, flush=True)
        record = provenance.build_record(out_dir, model_dir, reference_goldens)
        provenance.write_record(staged, record)

    ple_bytes = sum((out_dir / str(shard["file"])).stat().st_size for shard in index["shards"])
    return {
        "dir": str(out_dir),
        "nodes": len(verified.nodes),
        "outputs": len(verified.outputs),
        "initializers": len(verified.initializers),
        "model_bytes": sum(
            path.stat().st_size for path in resolve_shards(out_dir / one_shot.MODEL_FILE)
        ),
        "ple_bytes": ple_bytes,
        "ple_shards": index["shards"],
        "ple_probe_tokens": list(probe),
        "ops": sorted(verified.required_ops),
        "symbols": list(verified.symbols),
        "case_lengths": {name: int(ids.shape[1]) for name, ids in cases},
        "quantized": {"i8": int8.describe(), "i4": int4.describe()},
        "form": form,
        "reference": record["reference"],
        "sanity": sanity,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = one_shot.series_parser(__doc__.split("\n\n")[0], DEFAULT_OUT_DIR)
    parser.add_argument("--reference", type=Path, default=REFERENCE_DIR)
    one_shot.run_series_cli(parser, export_series, argv)


if __name__ == "__main__":
    main()
