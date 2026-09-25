"""実重み Gemma 4 E2B を **製品グラフ**（PLE 外出し + 選択行の logits / hidden）へ書き出す台本。

ADR [0083](../../../docs/decisions/0083-generation-api-surface.md) 決定 6（出口は選択行の
logits・sampling はホスト維持）と ADR
[0085](../../../docs/decisions/0085-ple-host-gather.md)（PLE をホスト gather へ外出し）を
**1 回の再 export に載せる**（案 α — ADR 0083 Consequences / backlog now の段 1b）。

    uv run --with 'transformers==5.14.1' python -m gemma4.export_product

## 既存 2 系列との差は入口 2 本・出口 2 本

chunk 系列の経路（素材の読み方・RoPE のホスト供給・KV 共有の手術・混成量子化・門の順序）は
{@link gemma4.export_decode} の中核をそのまま通す（import して使う — 同じ規律を 2 箇所に
書かない）。差分は 3 点だけ:

- 入力に **`per_layer_inputs[1,M,35,256]` f32** が増える。PLE lookup は `input_ids` **だけ**を
  引数に取る純粋な行 lookup なので、グラフから外してホストが供給する通常のグラフ入力に
  なる（ADR 0085 決定 6 — ランタイムの契約は 1 文字も変わらない）。容器からは i8 35 表
  2,240MiB + per-row scale 35MiB が消える。
- 入力に **`last_row[R]` i32** が増える（token-only 系列と同じ行選択の配線 —
  {@link gemma4.export_decode.TOKEN_ONLY_LAST_ROW}）。行数が記号 {@link ROW_SYMBOL} なのは
  投機デコードの verify run が 1 回で複数行を採点するため。**通常の prefill / decode は
  R = 1** で、その束縛では従来の 1 行出口と値も token 列もビット同一のまま。
- 出口は **`logits[1,R,V]`（出力 0）と最終 norm 後の hidden `[1,R,H]`（出力 1）**
  （選択**行**のみ・argmax なし）。token-only 系列の `TokenOnlyChunkWrapper` から argmax を
  外し、drafter が食う hidden を並べた形で、sampling / RNG はホストが持つ（ADR 0083 決定 6 の
  MUST）。prefill の読み戻しは `[1,M,V]` 形の 32MiB から 1MiB へ減る。
  MUST: 出力順は **logits → hidden** 固定（ランタイムはスロット番号で読む）。

## PLE（token-major + vocab レンジ block・容器の資産）

グラフから外した 35 表は **token-major**（`[token][layer][256]` i8 + `[token][layer]` の
per-row scale）へ再配置し、**vocab の範囲**で block に切る（ADR 0085 決定 1 / 2）。1 token の
PLE が連続 1 読み（8,960B + 35 scale）になる形で、後から「キャッシュから行だけ読む」
（同 ADR の代替案 b）へ移るときに再 export も再アップロードも要らない。

置き場は**モデル容器の資産**（`ple_index` / `ple.values.<k>` / `ple.scales.<k>` — ADR 0109
決定 4）。切り方も索引の綴りも core の {@link karume.ple.ple_assets} 1 本が持つ（移行 CLI と
同じ 1 本 — 2 経路で綴ると block の切り目が黙ってずれる）。

MUST: 再配置は **ビット同一**であること（{@link assert_ple_assets}）。i8 値と per-row scale の
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

    outputs/series/gemma4-e2b-product/model.krm                 重み・定数 + 2 文書 + PLE の資産
    outputs/series/gemma4-e2b-product/ple.probe.safetensors     逆量子化ビット一致の参照
    outputs/series/gemma4-e2b-product/reference.json            出所記録（指紋 + 流用 golden）
"""

from __future__ import annotations

import json
import sys
from collections.abc import Buffer, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch.export import Dim
from torch.nn import functional

from _shared.container_read import open_container, read_asset
from _shared.decode_series import assert_case_room, positions_for
from _shared.paths import SERIES_ROOT
from gemma4 import export as one_shot
from gemma4 import export_decode as decode
from gemma4 import ple, provenance
from gemma4.distribution import GEMMA4_ROLE
from karume.artifacts import staged_publication
from karume.container import BLOCK_MAX_BYTES, AssetInput, container_parts
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
from karume.ir import IrGraph
from karume.ops import ARGMAX_OP, EMBEDDING_OP
from karume.pipeline import export_module
from karume.ple import (
    PLE_INDEX_ASSET,
    PLE_PACK_FACTOR,
    ple_assets,
    ple_block_ranges,
    ple_row_bytes,
)
from karume.quantize import quantize_to_int8
from karume.shapes import declared_shape
from karume.states import to_states_form

#: 生成物の既定の置き場（既存 2 系列とは別ディレクトリ — 入口も出口も違う別資産）。
DEFAULT_OUT_DIR = SERIES_ROOT / "gemma4-e2b-product"

#: 流用する greedy 期待列の置き場（正本は logits opt-in 系列 — token-only 系列と同じ参照先）。
REFERENCE_DIR = decode.DEFAULT_OUT_DIR

#: ホストが供給する PLE のグラフ入力名（ラッパの forward 引数名そのもの — torch.export が
#: グラフ入力名に採る）。綴りの正本はここで、{@link assert_ir_form_product} と TS 側の検収門が
#: 参照する。
PER_LAYER_INPUTS = "per_layer_inputs"

#: 行選択の記号名（`last_row[R]` / 出口 2 本の行軸）。投機デコードの verify run が 1 回で
#: 採点する行数で、通常の prefill / decode は R = 1 に束縛する（従来と同じ 1 行出口）。
ROW_SYMBOL = "R"

#: 記号 R の trace 上限。verify が 1 回に採点する行数 = draft k 行 + 直前に確定した bonus 1 行で、
#: k の上限が sliding ring の余裕（{@link decode.SLIDING_SLACK_ROWS} — 棄却されうる行数の上限）
#: なので R ≤ 余裕 + 1。余裕と切り離して決めると「棄却行が live 窓を潰さない」設計が成立しない。
#:
#: NOTE: IR の `symbols` は名前の列だけで上限を持たない（`docs/ir-v2.md`）ので、この数は
#: **torch の guard を張る範囲**にしか効かない。ランタイム側の上限は context の `slidingSlack`
#: （deferred run の `queryLength ≤ slidingSlack + 1`）が持つ。
ROW_SYM_MAX = decode.SLIDING_SLACK_ROWS + 1

#: 例示入力の行数。**2 以上**でなければ torch.export が R を 1 に特殊化しうるし、記号として
#: 生きていることを IR で確かめられない。上限側（{@link ROW_SYM_MAX}）は Dim の宣言が持つ。
EXAMPLE_ROWS = 2

#: 逆量子化ビット一致の参照のファイル名（**配布物ではない** — 系列に残る検収用の golden）。
PLE_PROBE_FILE = "ple.probe.safetensors"

#: PLE の格納（`values` の詰め方 — `scales` は常に f32）。
PLE_STORAGE = "i8"

#: 行の種別（索引の欄名でもある — `karume.ple.PLE_ROLES` の綴り）。
PLE_VALUES_KEY = "values"
PLE_SCALES_KEY = "scales"

#: `ple.probe.safetensors` のテンソルキー。
PROBE_TOKENS_KEY = "tokens"
PROBE_INPUTS_KEY = "per_layer_inputs"


class ProductChunkWrapper(decode.DecodeChunkWrapper):
    """`(input_ids, RoPE 4 本, per_layer_inputs, last_row[R]) → (logits[1,R,V], hidden[1,R,H])`。

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
    MUST: 返す順は **(logits, hidden)** 固定 — ランタイムは出力スロット番号で読むので、
    入れ替えると形も dtype も合ったまま別のテンソルが sampling へ渡る。
    MUST: 第 2 出力は**最終 norm の後・lm_head の前**の行選択済み hidden そのもの
    （`rowed`）。lm_head を通す前の値であることが drafter 側の入力条件で、別の中間値を
    返しても shape は `[1,R,H]` のまま合う。
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        # 行選択のあと [1,R,H] へ上げてから lm_head へ通す（token-only 形と同文 — 行選択済み
        # lm_head の構造検査が「選択済みの行」を見るのはこの形が前提）。
        rowed = functional.embedding(last_row, hidden[0]).unsqueeze(0)
        logits = self.model.lm_head(rowed)
        cap = float(self.model.config.final_logit_softcapping)
        return torch.tanh(logits / cap) * cap, rowed


#: {@link decode.load_wrapper} へ渡す variant。**読まれるのは `wrapper` 欄だけ**（素材の読み方と
#: RoPE の差し替えはラッパ型に依らない）。残り 3 欄は {@link decode.export_series} 用の分岐で、
#: 本台本はそちらを通らない — 製品形の差（入口 2 本増・出口 logits・PLE 資産）は
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


# ---- PLE 資産 --------------------------------------------------------------


def ple_token_bytes(layers: int, dim: int) -> int:
    """1 token ぶんの PLE バイト数（i8 値 `layers × dim` + f32 scale `layers`）。

    E2B は 35 × 256 + 35 × 4 = **9,100 バイト/token**。token-major の狙いそのもので、
    1 token の PLE がこの長さの**連続 1 読み**になる（ADR 0085 決定 1）。
    """
    row_bytes = ple_row_bytes(PLE_STORAGE, layers, dim)
    return row_bytes[PLE_VALUES_KEY] + row_bytes[PLE_SCALES_KEY]


def ple_table_rows(tables: Sequence[torch.nn.Module], vocab_size: int) -> int:
    """PLE 35 分割の行数（= PLE 資産の token 行数）。

    MUST: `config.vocab_size_per_layer_input` から取らない —
    {@link gemma4.export.load_model_and_tables} が**検査席の行数**（`ple.PLE_PROBE_ROWS` = 8）へ
    差し替えた後の値なので、そちらを読むと
    8 行の PLE 資産が形も型も合ったまま書かれる（実際に 1 度踏んだ）。行数の正本は
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


def ple_probe_tokens(tokens: int, ranges: Sequence[tuple[int, int]]) -> tuple[int, ...]:
    """逆量子化ビット一致の参照に使う**散点** token id（block 境界の両側 + 両端 + 中央）。

    連続 N 個だと 1 つの block しか踏まず、別 block だけの取り違え（範囲の off-by-one・
    scale の層ずれ）が門に映らない（{@link gemma4.ple.probe_rows} と同じ理由の block 版）。
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


#: 一時ファイルへ落とすときの 1 回ぶんの行数の目安（バイト）。転置の中間が RAM に載る量なので、
#: 35 表ぶんの実体（E2B で約 2.35 GB）より 2 桁小さく取る。
_SPILL_CHUNK_BYTES = 64 * 1024 * 1024


def spill_payload(payload: Buffer, path: Path) -> None:
    """連結済みの payload を `path` へ 1 度だけ落とす（QAT 側 {@link gemma4_qat.ple} と共有）。

    MUST: 落とし先は**作業席の一時ファイル**で、呼び手は据え替えの前に消す（配布物ではない）。
    """
    with path.open("wb") as handle:
        handle.write(payload)


def _spill_tables(
    tables: Sequence[torch.Tensor], stride: int, rows: int, path: Path
) -> Callable[[int, int], bytes]:
    """層ごとの表を **token-major に連結した payload** として `path` へ 1 度だけ落とし、
    その区間読みを返す。

    `tables` は**層ごと**（table-major）に持っているので、行の塊ごとに 35 本から `stack` して
    token-major へ転置しながら書く（転置の中間は 1 塊ぶんだけ生きる）。

    MUST: 落とし先は**作業席の一時ファイル**で、呼び手は落とした直後に元の実体を手放す
    （`values.clear()` / `row_scales.clear()`）— 遅延読み口がメモリ上の 35 表を掴んだままだと、
    i8 の実体（E2B で約 2.35 GB）が torch.export と変換の間ずっと常駐する（台本の RAM 予算は
    README の 24GB）。据え替えの前に消すので配布物には出ない。

    MUST: 同じ区間からは**同じバイト列**が返る（書き手は sha256 を採るときと書くときの
    2 度引く）— 落とした後のファイルは誰も書き換えない。
    """
    per_chunk = max(1, _SPILL_CHUNK_BYTES // stride)
    with path.open("wb") as handle:
        for start in range(0, rows, per_chunk):
            stop = min(start + per_chunk, rows)
            chunk = torch.stack([table[start:stop] for table in tables], dim=1).contiguous()
            handle.write(memoryview(chunk.numpy()).cast("B"))
    written = path.stat().st_size
    if written != rows * stride:
        raise AssertionError(f"{path}: {written} バイト — {rows} 行 × {stride} バイトと違う")

    def read(begin: int, end: int) -> bytes:
        if begin % stride or end % stride:
            raise AssertionError(f"PLE の区間 [{begin}, {end}) が 1 行 {stride} バイトの倍数でない")
        with path.open("rb") as handle:
            handle.seek(begin)
            raw = handle.read(end - begin)
        if len(raw) != end - begin:
            raise AssertionError(f"{path}: PLE の区間 [{begin}, {end}) が途中で尽きた")
        return raw

    return read


@dataclass(frozen=True)
class PleContainerAssets:
    """{@link ple_container_assets} の戻り — 容器へ渡す資産と、消してよい一時ファイル。"""

    #: `publish_model(assets=…)` へそのまま渡す資産（索引 + `values` / `scales` の block 列）。
    assets: dict[str, AssetInput]
    #: token-major の payload を落とした作業席の一時ファイル（配布物ではない）。
    spills: tuple[Path, ...]

    def discard(self) -> None:
        """一時ファイルを消す（作業席ごと据わる前に呼ぶ MUST — 配布物に混ざらない）。"""
        for path in self.spills:
            path.unlink(missing_ok=True)


def ple_container_assets(
    values: Sequence[torch.Tensor],
    row_scales: Sequence[torch.Tensor],
    *,
    tokens: int,
    layers: int,
    dim: int,
    embed_scale: float,
    spill_dir: Path,
    block_bytes: int = BLOCK_MAX_BYTES,
) -> PleContainerAssets:
    """PLE を**モデル容器の資産**へ組む（索引 schema 3 + `values` / `scales` の block 列）。

    切り方も索引の綴りも core の {@link karume.ple.ple_assets} が持つ（移行 CLI と同じ 1 本）。
    ここが足すのは「層ごとの表 → token-major の連結 payload」の読み口だけである。

    MUST: 実体は `spill_dir`（作業席）の一時ファイルへ**1 度だけ**落とす（{@link _spill_tables}
    の MUST）— 呼び手はこの呼び出しの直後に `values` / `row_scales` を手放し、export 中に
    i8 の実体を持ち越さない。
    """
    row_bytes = ple_row_bytes(PLE_STORAGE, layers, dim)
    spills = {key: spill_dir / f".ple.{key}.spill" for key in (PLE_VALUES_KEY, PLE_SCALES_KEY)}
    readers = {
        PLE_VALUES_KEY: _spill_tables(
            values, row_bytes[PLE_VALUES_KEY], tokens, spills[PLE_VALUES_KEY]
        ),
        PLE_SCALES_KEY: _spill_tables(
            row_scales, row_bytes[PLE_SCALES_KEY], tokens, spills[PLE_SCALES_KEY]
        ),
    }
    return PleContainerAssets(
        ple_assets(
            storage=PLE_STORAGE,
            tokens=tokens,
            layers=layers,
            dim=dim,
            embed_scale=embed_scale,
            read_values=readers[PLE_VALUES_KEY],
            read_scales=readers[PLE_SCALES_KEY],
            block_bytes=block_bytes,
        ),
        tuple(spills.values()),
    )


def _locate_block(blocks: Sequence[Mapping[str, Any]], token: int) -> Mapping[str, Any] | None:
    """token を含む block（索引は昇順の隙間なし分割なので線形走査で足りる — probe は散点）。"""
    for block in blocks:
        if int(block["start"]) <= token < int(block["stop"]):
            return block
    return None


def _unpack_ple_values(packed: torch.Tensor, storage: str) -> torch.Tensor:
    """`values` の生バイト（u8）を量子化値 q（i8）へ展開する — 末尾軸が `× 詰め数` に伸びる。

    定義は container-v1 §6.3 の codec 台帳（TS の読み手の `int8-sym` / `int4-sym-g` /
    `int2-off`）と同じ: i8 は `u = q`、i4 は `u = q + 8`（バイト内の**下位** nibble が先）、
    i2 は `u = q + 2`（**下位 2bit から**順に）。上流 `QuantizedEmbedding` の展開とも同じ
    定義で、門はこの 1 本だけを持つ（詰め順を取り違えると参照側と「同じ向きに間違った 2 つ」を
    突き合わせることになるので、参照側は上流モジュールの出力のまま保つ）。
    """
    if storage == "i8":
        return packed.view(torch.int8)
    bits = 8 // PLE_PACK_FACTOR[storage]
    mask = (1 << bits) - 1
    offset = 1 << (bits - 1)
    fields = [((packed >> shift) & mask).to(torch.int8) - offset for shift in range(0, 8, bits)]
    return torch.stack(fields, dim=-1).flatten(-2)


def assert_ple_assets(
    container: Path, index: Mapping[str, Any], probe: Sequence[int], reference: torch.Tensor
) -> None:
    """据えた容器の資産から probe token を組み直し、35 表経路と**ビット一致**することを見る。

    参照側（`reference`）は {@link gemma4.ple.per_layer_inputs} が fake-quant 済みの 35 表から
    組んだ `[1,P,35,256]` そのもの — つまり **PLE をグラフに残していたら embedding op が
    出していた値**（i8 格納 + per-row scale の逆量子化は fake-quant の値を厳密に復元する
    — ADR 0019 の ±127 論証）。再配置側は**書いたバイト列を読み直し**、ホストと同じ順序
    （`f32(q) * scale` → `* embed_scale`）で組む。

    格納（索引の `storage`）は i8 / i4 / i2 のどれでも同じ門を通る — QAT の固定 packed の PLE
    （ADR 0097 追記 4）の参照は上流 `QuantizedEmbedding` の出力で、packed の値は
    {@link _unpack_ple_values} で展開する。行のバイト数は索引の `rowBytes` を正本に読み、
    格納と寸法から導く値（`karume.ple.ple_row_bytes`）と食い違えば落とす。

    MUST: `torch.equal`（ビット一致）で見る — scale の対応を 1 層ずらしても、block の範囲を
    1 行ずらしても、形も型も dtype も合ったまま**別 token の有効な行**が出る。
    MUST: 読み直す（in-memory の配列を突き合わせない）— 転置は正しいのに書き出しの
    dtype / 形 / 順序が違う形を、この門が受け止める最後の位置。
    MUST: 読んだ block は **probe 行だけ取り出して捨てる**（block 丸ごとを器に残さない）—
    probe は全 block の両端を踏むので、残すと PLE 全量（E2B で約 2.35 GB）がこの門の実行中に
    もう一度 RAM に載る。容器も 1 度だけ開く（block ごとに `verify_container` を回さない）。
    MUST: 同じ block を**2 度読まない** — probe は 1 block の両端と中を踏むので、行ごとに
    読み直すと同じ数十 MB の資産を最大 3 度復号する。要る行を先に集めてから block 単位で
    1 度だけ読む（常駐は行の器だけで変わらない）。
    """
    layers = int(index["layers"])
    dim = int(index["dim"])
    embed_scale = float(index["embedScale"])
    storage = str(index["storage"])
    row_bytes = {key: int(index[key]["rowBytes"]) for key in (PLE_VALUES_KEY, PLE_SCALES_KEY)}
    derived = ple_row_bytes(storage, layers, dim)
    if row_bytes != derived:
        raise AssertionError(
            f"索引の rowBytes {row_bytes} が格納 '{storage}'・layers {layers}・dim {dim}"
            f" から導く {derived} と違う"
        )
    expected_shape = (1, len(probe), layers, dim)
    if tuple(reference.shape) != expected_shape:
        raise AssertionError(f"参照 {tuple(reference.shape)} が {expected_shape} でない")

    rebuilt = torch.zeros(expected_shape, dtype=torch.float32)
    opened = open_container(container)

    def rows_of(
        wanted: Sequence[tuple[Mapping[str, Any], int]], stride: int
    ) -> dict[tuple[str, int], torch.Tensor]:
        """要る `(block, token)` を **block 1 本につき 1 度の読み**で集める（残りは捨てる）。

        行は u8 の生バイト（`stride` バイト）で返す — 解釈（展開・f32 への読み替え）は呼び手。
        """
        by_asset: dict[str, tuple[Mapping[str, Any], set[int]]] = {}
        for block, token in wanted:
            by_asset.setdefault(str(block["asset"]), (block, set()))[1].add(token)
        picked: dict[tuple[str, int], torch.Tensor] = {}
        for name, (block, tokens) in by_asset.items():
            raw = bytearray(read_asset(opened, name))
            start = int(block["start"])
            rows = int(block["stop"]) - start
            expected = rows * stride
            if len(raw) != expected:
                raise AssertionError(
                    f"資産 '{name}' が {len(raw)} バイト — 索引の範囲"
                    f" [{block['start']}, {block['stop']}) から組んだ期待は {expected}"
                )
            view = torch.frombuffer(raw, dtype=torch.uint8).reshape(rows, stride)
            for token in sorted(tokens):
                picked[(name, token)] = view[token - start].clone()
            # 次の block を読む前に block 丸ごとの器を手放す（`view` は `raw` を共有する）。
            del view, raw
        return picked

    located: list[tuple[int, int, Mapping[str, Any], Mapping[str, Any]]] = []
    for position, token in enumerate(probe):
        blocks = {
            key: _locate_block(index[key]["blocks"], token)
            for key in (PLE_VALUES_KEY, PLE_SCALES_KEY)
        }
        values_block = blocks[PLE_VALUES_KEY]
        scales_block = blocks[PLE_SCALES_KEY]
        if values_block is None or scales_block is None:
            continue
        located.append((position, token, values_block, scales_block))
    if len(located) != len(probe):
        raise AssertionError(
            f"probe {len(probe)} 本のうち {len(located)} 本しか block の範囲に載っていない"
            "（索引の [start, stop) が vocab を覆っていない）"
        )

    values = rows_of(
        [(block, token) for _p, token, block, _s in located], row_bytes[PLE_VALUES_KEY]
    )
    scales = rows_of(
        [(block, token) for _p, token, _v, block in located], row_bytes[PLE_SCALES_KEY]
    )
    for position, token, values_block, scales_block in located:
        packed = values[(str(values_block["asset"]), token)]
        quantized = _unpack_ple_values(packed, storage).reshape(layers, dim).to(torch.float32)
        scale = scales[(str(scales_block["asset"]), token)].view(torch.float32).reshape(layers, 1)
        rebuilt[0, position] = quantized * scale * embed_scale
    if not torch.equal(rebuilt, reference):
        worst = float((rebuilt - reference).abs().max())
        raise AssertionError(
            "PLE の再配置が 35 表経路とビット一致しない"
            f"（最大絶対差 {worst}）— 値と per-row scale の対応か token 範囲がずれている"
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
    row_symbol: str = ROW_SYMBOL,
) -> dict[str, Any]:
    """製品グラフの形を検査する（**数値が合ったまま静かに壊れる**性質を全部見る）。

    states / 層種別 / 残骸 / 格納の各節は {@link decode.assert_ir_form_common} に預ける
    （3 形が共有する本体 — 段 1b では「入口 / 出口が排他だから」と独立に綴っていたが、
    共有部分だけを関数に括れば入口 / 出口の引数化は要らない）。ここが綴るのは製品形に固有の
    入口・出口だけである。

    製品形に固有の門（PLE 外出しと、選択行の logits / hidden 出口の実証）:

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
      （ADR 0083 決定 6 — GPU 側は選択行の logits まで）。
    - 出力が **2 本で、順序は logits → hidden** であること。ランタイムはスロット番号で読むので、
      入れ替わっても両方 `[1, R, *]` の f32 で軸は合う。順序を固定するのは幅の突合と、
      **出力 0 の祖先が lm_head に届く**という構造検査の 2 本（幅が偶然一致しても後者が残る）。
    - 出力の宣言 shape が `[1, R, vocab_size]` / `[1, R, hidden_size]`（選択**行**のみ）である
      こと。全行 logits へ退行しても token 列は一致するので、構造検査でしか固定できない
      （ADR 0068 の実効）。
    - `last_row` の宣言 shape が `[R]` であること。R を束縛できる入力はこれ 1 本なので、
      静的 `[1]` へ退行すると「記号が居るのに束縛点が無い」形が書ける。
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
    last_row_spec = next(spec for spec in graph.inputs if spec.name == decode.TOKEN_ONLY_LAST_ROW)
    if list(last_row_spec.shape) != [row_symbol]:
        raise AssertionError(
            f"'{decode.TOKEN_ONLY_LAST_ROW}' の宣言 shape が {list(last_row_spec.shape)} —"
            f" [{row_symbol}] でない（記号 {row_symbol} の唯一の束縛点）"
        )
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

    if len(graph.outputs) != 2:
        raise AssertionError(
            f"IR 出力が {len(graph.outputs)} 本"
            "（製品出口は logits + hidden の 2 本 — ADR 0083 決定 6 / 投機 verify の足場）"
        )
    if ARGMAX_OP in graph.required_ops:
        raise AssertionError(
            f"`{ARGMAX_OP}` がグラフに残っている"
            "（製品出口は選択行 logits で、sampling / argmax はホスト側 — ADR 0083 決定 6）"
        )
    logits_shape = list(declared_shape(graph, graph.outputs[0]))
    hidden_shape = list(declared_shape(graph, graph.outputs[1]))
    expected_logits = [1, row_symbol, int(config.vocab_size)]
    expected_hidden = [1, row_symbol, int(config.hidden_size)]
    # MUST: 順序まで見る（幅が違うだけで両方 f32 [1,R,*] なので、入れ替えは形では落ちない）。
    for slot, (found, expected, what) in enumerate(
        ((logits_shape, expected_logits, "logits"), (hidden_shape, expected_hidden, "hidden"))
    ):
        if found != expected:
            raise AssertionError(
                f"出力 {slot} の宣言 shape が {found} — {expected}（選択行の {what}）でない"
            )

    # 行選択済み lm_head の固定は token-only 形と**同じ構造検査**（decode 側の 1 本を通す）。
    producer = {out: node for node in graph.nodes for out in node.outs}
    logits_source = producer.get(graph.outputs[0])
    if logits_source is None:
        raise AssertionError(f"出力 0 ('{graph.outputs[0]}') がノード出力でない")
    decode.assert_row_selected_lm_head(graph, producer, logits_source, rows=row_symbol)

    form = decode.assert_ir_form_common(
        graph,
        config,
        storage_expectation,
        seq_symbol=seq_symbol,
        capacity_symbol=capacity_symbol,
        extra_symbols=(row_symbol,),
    )
    return {
        **form,
        "embedding_nodes": len(embeddings),
        "logits": logits_shape,
        "hidden": hidden_shape,
    }


# ---- 系列 ------------------------------------------------------------------


def export_series(
    model_dir: Path,
    out_dir: Path,
    *,
    sym_max: int = one_shot.SYM_MAX,
    reference: Path = REFERENCE_DIR,
) -> dict[str, Any]:
    """製品グラフのコンテナ（PLE を資産として同梱）+ 出所記録を書き、要約を返す。

    MUST: 生成物は作業席へ書き、**全ての門**（PLE のビット一致・形検査・1-shot 期待表との
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
    # block の切り目は寸法だけで決まる（実体は要らない）ので、i8 へ落とす前に probe を選べる。
    ranges = ple_block_ranges(storage=PLE_STORAGE, tokens=tokens, layers=layers, dim=dim)
    probe = ple_probe_tokens(tokens, ranges[PLE_VALUES_KEY])

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
    # min=1 は MUST — 通常の prefill / decode は R = 1 で走るので、そこを特殊化されると
    # 従来経路が動かない。torch.export は size 1 を含む Dim を記号のまま保つ（2.13 で実測）。
    rows = Dim(ROW_SYMBOL, min=1, max=ROW_SYM_MAX)

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with staged_publication(out_dir) as staged:
        # ディレクトリの席は書き手が作る（原語は席を作らない — path しか渡さない）。
        staged.mkdir()
        ple_assets_built = ple_container_assets(
            values,
            row_scales,
            tokens=tokens,
            layers=layers,
            dim=dim,
            embed_scale=embed_scale,
            spill_dir=staged,
        )
        assets = ple_assets_built.assets
        # MUST: i8 実体は**ここで**手放す（以降は模型ぶんの RAM だけで export へ入る）— 資産の
        # 読み口は作業席の一時ファイルを指しているので、35 表を生かしておく理由がもう無い。
        values.clear()
        row_scales.clear()
        save_file(
            {
                PROBE_TOKENS_KEY: torch.tensor(list(probe), dtype=torch.int32).contiguous(),
                PROBE_INPUTS_KEY: probe_reference.contiguous(),
            },
            str(staged / PLE_PROBE_FILE),
        )

        print("[export] torch.export → 変換", file=sys.stderr, flush=True)
        try:
            example_rope = decode.rope_args(specs, positions_for(example_ids))
            graph, tensors = export_module(
                wrapper,
                (
                    example_ids,
                    *example_rope,
                    case_inputs[example_name],
                    decode.last_rows_for(example_ids, EXAMPLE_ROWS),
                ),
                dynamic_shapes=(*({1: seq} for _ in range(2 + len(example_rope))), {0: rows}),
                # 割り当ては user 入力 placeholder の出現順
                # （`karume.convert._assign_input_symbols`）なので、`input_ids[1,M]` →
                # `last_row[R]` の順で並べる。
                symbol_names=(decode.SEQ_SYMBOL, ROW_SYMBOL),
                preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
            )
            print("[export] states 形へ手術 → 書き出し", file=sys.stderr, flush=True)
            surgical = to_states_form(graph, decode.states_plan(graph, config))
            verified = decode._write_container(
                surgical,
                tensors,
                staged / one_shot.MODEL_FILE,
                # グラフ名は**部品名**（= karume.json の weights のキー）。ディレクトリ名から
                # 導かない — 系列名（`gemma4-e2b-product`）も作業席の名前も部品名とは一致しない
                # （container-v1 §2.1）。
                graph_name=GEMMA4_ROLE,
                weight_dtype="i8",
                weight_scales=scales,
                weight_dtype_overrides=dict.fromkeys(int4.scales, "i4"),
                assets=assets,
            )
        finally:
            # MUST: 一時ファイルは据え替えの前に消す（作業席ごと据わるので、残すと配布物に混ざる）。
            ple_assets_built.discard()
        index = json.loads(bytes(assets[PLE_INDEX_ASSET].payload))
        print("[ple] 据えた容器の資産から probe を組み直す", file=sys.stderr, flush=True)
        assert_ple_assets(staged / one_shot.MODEL_FILE, index, probe, probe_reference)
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
                # 例示入力と同じ R 行で引き、**最終行**の 1 位を見る（R > 1 の行選択そのものを
                # sanity でも 1 度踏む — 期待表と突き合わせるのは prompt 最終行の継続だけ）。
                logits, _hidden = wrapper(
                    ids,
                    *decode.rope_args(specs, positions_for(ids)),
                    case_inputs[name],
                    decode.last_rows_for(ids, EXAMPLE_ROWS),
                )
            first[name] = int(logits[0, -1].argmax())

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

    return {
        "dir": str(out_dir),
        "nodes": len(verified.nodes),
        "outputs": len(verified.outputs),
        "initializers": len(verified.initializers),
        "model_bytes": sum(
            path.stat().st_size for path in container_parts(out_dir / one_shot.MODEL_FILE)
        ),
        # PLE は容器の中の資産なので、バイト数は索引が名乗る行数 × 1 行から出す
        # （容器の part はグラフの重みと同居するので、ファイルサイズからは切り出せない）。
        "ple_bytes": tokens * ple_token_bytes(layers, dim),
        "ple_blocks": {key: len(index[key]["blocks"]) for key in (PLE_VALUES_KEY, PLE_SCALES_KEY)},
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
