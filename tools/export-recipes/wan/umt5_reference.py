"""Wan2.1 の umT5（i8 系列）の層逐次の CPU 参照と、移植の門の golden
（ADR 0119 決定 8 ①・段 10c）。

CLI は `wan.umt5_export reference`（容器は書かない — 既に書かれた i8 系列の容器から
重みを読む）。

## 参照の形（決定 8 ① — 同じ i8 の fake-quant 重みで回した CPU の層逐次の参照）

移植の門の参照（`output.f64` / `output.f32`）は i8 の重みで、品質の記録の基準（受入れだけ —
`output.unquantized.*`）は量子化しない重み（{@link CheckpointWeights}）で採る。計算はどちらも
同じ書き下しで、重みの読み口だけが違う。

- **重み**: i8 系列の容器から 1 本ずつ読む（{@link ContainerWeights}）。i8 の重みは
  packed × scale を f32 で掛けた値で、export の fake-quant（`umt5_export.quantize_rows`
  の `rounded`）と同じ値 — GPU が回すのと同じ重み。F32 の表（相対位置の表・RMSNorm）は
  そのまま。f64 の参照はこの f32 の値を f64 へ広げる（f64 で丸め直すと scale と丸めの値が
  f32 の参照と食い違う — DiT の `export_dit.float64_references` と同じ規則）。
- **計算**: 上流の forward を**書き下す**（{@link encode_layerwise}）: 語彙埋め込み →
  層ごとに〈RMSNorm → 自己 attention（相対位置のバイアス・スケール無し・softmax）→
  残差 → RMSNorm → ゲート付き GELU（tanh 近似 — 10c の準備の裁定 (a)）の FFN → 残差〉
  → 最後の RMSNorm。dropout とマスクは持たない（有効長だけ — 決定 4）。
- **f64 と f32 は同じ書き下しで dtype だけが違う**。上流は RMSNorm の分散と softmax を
  f32 に落とす（`.to(torch.float32)` / `.float()`）ので、上流のモデルを `.double()` しても
  f64 の参照にならない（ADR 0119 追記「10c の準備」）。書き下しはその寄せを持たない。
  f32 では上流の寄せが恒等なので、上流の eager（活性を tanh に差し替え・fake-quant 済み）と
  ビット一致する（pytest が小模型で縛る）。
- **層逐次**: 全ケースの活性 `[1, L, 4096]` を持ったまま層を 1 枚ずつ進める。重みは
  1 層ぶん（f32 約 0.77 GB + f64 約 1.55 GB）しか同時に持たない — fake-quant 後の f32 で
  全体を 1 回 forward すると匿名メモリが約 21 GiB になる（ADR 0119 追記「10b の結果」）。
  語彙埋め込みは使う行の入った block だけを容器から読む。
- 層の中の演算は上流と同じ形（`[1, L, d]` のまま `functional.linear`・ケースごと）で
  回す。ケースを行方向に連結すると GEMM の分割が変わり、f32 の参照が上流の eager と
  ビットで割れうる。

## golden（`reference.<case>.safetensors` — 1 ケース 1 ファイル）

- `input_ids`（I32 `[L]`）: 上流の経路（`prompt_clean` → transformers 5.14.1 — 10a の
  `umt5_tokenizer.build_case`）の id 列
- `relative_position_buckets`（I32 `[L, L]`）: 上流の式
  （`umt5_patch.relative_position_buckets`）
- `output.f64`（F32 `[1, L, 4096]`）: f64 の参照を f32 へ丸めた値（TS の safetensors は
  F64 を読まない — 丸めの影響はメタの `ratios`）
- `output.f32`（F32 `[1, L, 4096]`）: f32 の参照（正規化比 r の分母）
- `output.unquantized.f64` / `output.unquantized.f32`（F32 `[1, L, 4096]`・**受入れだけ**）: 量子化
  しない重み（pin した checkpoint の F32 をそのまま — {@link CheckpointWeights}）で回した同じ
  書き下し（f64 は格納で f32 へ丸める）。品質の記録の基準（ADR 0119 追記「10b の結果」の
  「f32 参照を基準にする」— i8 の丸めと bf16 の影響を同じ基準で分けて測る）で、門には使わない

メタはキー 1 つ（{@link METADATA_KEY}）に JSON（キー整列・区切りの空白なし —
safetensors はメタを HashMap で書くので、キーが複数だと同じ入力でバイトが割れる —
`wan.text_embeds` と同じ理由）。中身はプロンプトの原文と前処理後・L・役割・出所・容器の
part 0 の sha256（容器を書き直したら golden も書き直す — TS が突き合わせる）・重みと活性の
規則・分母と丸めの影響（{@link reference_ratios}）・受入れは量子化なしの参照の同じ比。

TS の e2e（`packages/models/tests/e2e_wan_umt5_test.ts`）は、golden の原文を TS の
トークナイザに通した id 列と TS のバケット表が golden の入力とビット一致することを
見てから（TS と Python の両経路）、GPU の出力を正規化比
r = (GPU の f64 に対する比) ÷ (CPU f32 の f64 に対する比) で判定する。

## ケース（決定 8 ① — 帯の決定と受入れを分ける）

- **決定用（`band`）6 本**: 10a のパリティ fixture
  （`packages/models/tests/fixtures/wan-text/parity.json`）の受理した乱択（`random-*` —
  受理集合の文字の乱択・3〜42 トークン）から seed {@link BAND_SEED} で選ぶ。
  - 単体 3 本: 長さの帯 {@link BAND_SINGLE_LENGTHS}（短・中・長）ごとに 1 本。
  - 合成 3 本: 乱択に 512 付近が無いので、単体に選ばなかった乱択を同じ seed で並べ替え、
    空白で連ねて長さの帯 {@link BAND_COMPOSITE_LENGTHS}（max_distance 128 を越える・
    中ほど・上限 512 の手前）に入るまで先頭から足す（帯を越える 1 本は飛ばす）。連ねた
    文字列も上流の経路で id 列を採り直す。
- **受入れ（`accept`）4 本**: 固定 4 プロンプト（`wan.prompts.FIXED_PROMPTS` —
  長さ 28 / 118 / 50 / 126）。bf16 の事前計算資産があるのはこの 4 本だけ（品質の記録の
  3 点目 — 決定 8 ②）。

MUST: 受入れの結果を見て決定用のケースも選び方も変えない（帯の決定と受入れの独立 —
ADR 0118 決定 8）。

MUST: transformers / diffusers は関数の中で import する（`wan` グループは既定の sync に
入らない — `tests/test_optional_group_imports.py`）。
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import torch
from safetensors.torch import save_file
from torch.nn import functional

from karume.container import codec_entry, container_parts
from karume.convert import normalize_boundary_tensor
from karume.verify import InitializerSupply, verify_container
from wan import umt5_patch
from wan.umt5_tokenizer import FIXED_PROMPT_LENGTHS, FIXTURE_DIR, MAX_LENGTH, PARITY_FIXTURE

#: golden のファイル名（`reference.<case>.safetensors` — DiT の golden と同じ綴り）。
REFERENCE_PREFIX = "reference."
CASE_SUFFIX = ".safetensors"

#: golden のキー（入力はグラフ入力と同じ名前 — TS 側 `e2e_wan_umt5_test.ts` と同じ綴り）。
INPUT_IDS_KEY, BUCKETS_KEY = umt5_patch.INPUT_NAMES
OUTPUT_F64_KEY = "output.f64"
OUTPUT_F32_KEY = "output.f32"
#: 量子化なしの参照（受入れだけ — 品質の記録の基準）。
OUTPUT_UNQUANTIZED_F64_KEY = "output.unquantized.f64"
OUTPUT_UNQUANTIZED_F32_KEY = "output.unquantized.f32"

#: メタのキー（1 つだけ — モジュール doc）と形式の版。
METADATA_KEY = "karume.wan.umt5_reference"
#: /2 = 受入れに量子化なしの参照を足した形（TS は受入れでこのキーを必須にする）。
REFERENCE_FORMAT = "karume-wan-umt5-reference/2"

#: 容器のテンソルキー（容器の initializer 名 = ラッパ `Umt5EncoderTokens` のテンソルキー）。
EMBED_KEY = "encoder.embed_tokens.weight"
FINAL_NORM_KEY = "encoder.final_layer_norm.weight"

#: 量子化の codec（i8 per-channel・行ごとの scale — 決定 5）。
I8_CODEC = "int8-sym"

#: 決定用のケースを選ぶ seed（`random.Random`）。
BAND_SEED = 20261003

#: 決定用の単体の長さの帯（両端を含む）。パリティ fixture の受理した乱択は 3〜42 トークン。
BAND_SINGLE_LENGTHS: tuple[tuple[int, int], ...] = ((2, 8), (16, 24), (34, MAX_LENGTH))

#: 決定用の合成の長さの帯（両端を含む）: max_distance 128（バケットの対数の区間の上限）を
#: 越える・中ほど・上限 512 の手前。
BAND_COMPOSITE_LENGTHS: tuple[tuple[int, int], ...] = ((160, 200), (300, 360), (480, MAX_LENGTH))

#: パリティ fixture の乱択ケースの id の接頭辞。
RANDOM_PREFIX = "random-"

Role = Literal["band", "accept"]


class Umt5ReferenceError(RuntimeError):
    """容器・ケース・参照が想定（決定 5 / 8）から外れた。"""


# ---------------------------------------------------------------------------
# 重み（容器から 1 本ずつ）
# ---------------------------------------------------------------------------


class WeightSource(Protocol):
    """参照の重みの読み口（値は f32 — i8 は packed × scale）。"""

    def tensor(self, key: str) -> torch.Tensor: ...

    def rows(self, key: str, indices: torch.Tensor) -> torch.Tensor: ...


def _f32_of(raw: bytes) -> torch.Tensor:
    # bytearray で写す — 読み取り専用のバッファに frombuffer を張ると torch が警告を出す。
    return torch.frombuffer(bytearray(raw), dtype=torch.float32)


class ContainerWeights:
    """i8 系列の容器（`krm` の part 列）から重みを 1 本ずつ読む。

    block は取るたびに sha256 を宣言と突き合わせる（`ReadContainer.block`）。i8 は
    packed × 行ごとの scale を f32 で掛けた値を返す — export の fake-quant と同じ値
    （ビット一致は pytest が小模型の容器で縛る）。MUST: codec は `f32` と `int8-sym`
    （軸 0・行ごとに scale 1 つ）だけを受ける — ほかは参照が黙って別の重みになる。
    """

    def __init__(self, path: Path) -> None:
        parts = container_parts(path)
        verified = verify_container(list(parts))
        if len(verified.graphs) != 1:
            raise Umt5ReferenceError(f"{path}: グラフが {sorted(verified.graphs)}（1 本のはず）")
        (graph,) = verified.graphs
        self.graph = graph
        self.parts = len(parts)
        with parts[0].open("rb") as stream:
            #: part 0（ヘッダ + 2 文書 — 全 block と part の sha256 を宣言する）の sha256。
            #: 容器の中身を推移的に名指すので、golden と容器の取り違えを TS がこれで拾う。
            self.part0_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        self._read = verified.read
        self._values = verified.read.graph.graphs[graph]["values"]
        self._supplies = verified.graphs[graph].supplies

    def _supply(self, key: str) -> InitializerSupply:
        supply = self._supplies.get(key)
        if supply is None:
            raise Umt5ReferenceError(f"容器に '{key}' の実体が無い")
        return supply

    def _shape(self, key: str) -> list[int]:
        return [int(size) for size in self._values[key]["shape"]]

    def _payload(self, supply: InitializerSupply, index: int) -> bytes:
        block = supply.blocks[index]
        return self._read.block(block.id)[: block.payload_bytes]

    def _scale(self, key: str, supply: InitializerSupply, rows: int, cols: int) -> torch.Tensor:
        """行ごとの scale `[rows, 1]`（f32）。codec の宣言が行ごとの i8 であることも見る。"""
        encoding = supply.encoding
        if encoding.codec != I8_CODEC or encoding.row_axis != 0 or encoding.group_size != cols:
            raise Umt5ReferenceError(
                f"'{key}': codec {encoding.codec}・rowAxis {encoding.row_axis}・groupSize "
                f"{encoding.group_size}（{I8_CODEC}・0・{cols} だけを読む）"
            )
        if supply.scale is None:
            raise Umt5ReferenceError(f"'{key}': i8 なのに scale が無い")
        scale = _f32_of(self._read.block(supply.scale.id)[: supply.scale.payload_bytes])
        if scale.numel() != rows:
            raise Umt5ReferenceError(f"'{key}': scale {scale.numel()} 個が行数 {rows} と違う")
        return scale.reshape(rows, 1)

    def tensor(self, key: str) -> torch.Tensor:
        """1 本を丸ごと f32 で（i8 は packed × scale）。"""
        supply = self._supply(key)
        shape = self._shape(key)
        raw = b"".join(self._payload(supply, index) for index in range(len(supply.blocks)))
        layout = codec_entry(supply.encoding.codec).layout
        if layout == "f32":
            return _f32_of(raw).reshape(shape)
        if layout != "i8" or len(shape) != 2:
            raise Umt5ReferenceError(f"'{key}': 格納 {layout}・形 {shape} は読まない")
        rows, cols = shape
        packed = torch.frombuffer(bytearray(raw), dtype=torch.int8).reshape(rows, cols)
        return packed.to(torch.float32) * self._scale(key, supply, rows, cols)

    def rows(self, key: str, indices: torch.Tensor) -> torch.Tensor:
        """i8 の表の行 `indices`（昇順・重複なし）だけを f32 で（語彙埋め込み — 要る block
        だけを読む）。"""
        supply = self._supply(key)
        rows, cols = self._shape(key)
        wanted = indices.tolist()
        if wanted != sorted(set(wanted)) or not wanted or wanted[0] < 0 or wanted[-1] >= rows:
            raise Umt5ReferenceError(f"'{key}': 行の添字は 0〜{rows - 1} の昇順・重複なしで渡す")
        if codec_entry(supply.encoding.codec).layout != "i8":
            raise Umt5ReferenceError(f"'{key}': 行の読み出しは i8 の表だけ")
        scale = self._scale(key, supply, rows, cols)
        gathered = torch.empty((len(wanted), cols), dtype=torch.float32)
        covered = torch.zeros(len(wanted), dtype=torch.bool)
        for index, block in enumerate(supply.blocks):
            begin, end = block.rows
            # (gathered の中の位置, block の中の行) — block の並びに依らず置く。
            inside = [(at, row - begin) for at, row in enumerate(wanted) if begin <= row < end]
            if not inside:
                continue
            positions = torch.tensor([at for at, _ in inside])
            local = torch.tensor([row for _, row in inside])
            packed = torch.frombuffer(bytearray(self._payload(supply, index)), dtype=torch.int8)
            gathered[positions] = packed.reshape(end - begin, cols)[local].to(torch.float32)
            covered[positions] = True
        if not bool(covered.all()):
            raise Umt5ReferenceError(f"'{key}': block の行範囲が要る行を覆わない")
        return gathered * scale[indices]


class CheckpointReader(Protocol):
    """上流の checkpoint の読み口（`umt5_export.Checkpoint` — F32 だけを返す）。"""

    def read(self, key: str) -> torch.Tensor: ...

    def read_rows(self, key: str, start: int, stop: int) -> torch.Tensor: ...


class CheckpointWeights:
    """量子化しない重み（pin した checkpoint の F32 をそのまま）を 1 本ずつ読む（品質の記録の
    基準）。

    {@link ContainerWeights} と同じ読み口で、書き下しは同じものを使う。`keys` は容器のテンソル
    キー → checkpoint のキー（tied な語彙埋め込みは checkpoint に片方の名前でしか無い —
    `umt5_export.checkpoint_weights` が組む）。checkpoint の safetensors は mmap で開くので、
    全体を f32 で持たない（層逐次 — 決定 8 ①）。MUST: 対応の無いキーは fail loudly（黙って
    容器の値に落とさない）。
    """

    def __init__(self, checkpoint: CheckpointReader, keys: Mapping[str, str]) -> None:
        self._checkpoint = checkpoint
        self._keys = dict(keys)

    def _key(self, key: str) -> str:
        mapped = self._keys.get(key)
        if mapped is None:
            raise Umt5ReferenceError(f"'{key}' の checkpoint のキーが無い")
        return mapped

    def tensor(self, key: str) -> torch.Tensor:
        return self._checkpoint.read(self._key(key))

    def rows(self, key: str, indices: torch.Tensor) -> torch.Tensor:
        """行 `indices`（昇順・重複なし）だけ — 連続する行の区間ごとに読む。"""
        wanted = indices.tolist()
        if wanted != sorted(set(wanted)) or not wanted or wanted[0] < 0:
            raise Umt5ReferenceError(f"'{key}': 行の添字は 0 以上の昇順・重複なしで渡す")
        mapped = self._key(key)
        runs: list[tuple[int, int]] = []
        for row in wanted:
            if runs and runs[-1][1] == row:
                runs[-1] = (runs[-1][0], row + 1)
            else:
                runs.append((row, row + 1))
        chunks = [self._checkpoint.read_rows(mapped, start, stop) for start, stop in runs]
        gathered = torch.cat(chunks)
        if gathered.shape[0] != len(wanted):
            raise Umt5ReferenceError(f"'{key}': 行 {wanted[-1]} が表の外")
        return gathered


# ---------------------------------------------------------------------------
# forward の書き下し（f64 / f32 共通）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncoderShape:
    """書き下しに要る構成（上流の `UMT5Config` から）。"""

    layers: int
    heads: int
    head_dim: int
    eps: float

    @staticmethod
    def of(config: Any) -> EncoderShape:
        umt5_patch.check_supported(config)
        return EncoderShape(
            layers=int(config.num_layers),
            heads=int(config.num_heads),
            head_dim=int(config.d_kv),
            eps=float(config.layer_norm_epsilon),
        )


def layer_keys(index: int) -> dict[str, str]:
    """層 `index` の重みのテンソルキー（書き下しの中の名前 → 容器の名前）。"""
    prefix = f"encoder.block.{index}.layer"
    attention = f"{prefix}.0.SelfAttention"
    dense = f"{prefix}.1.DenseReluDense"
    return {
        "attention_norm": f"{prefix}.0.layer_norm.weight",
        "q": f"{attention}.q.weight",
        "k": f"{attention}.k.weight",
        "v": f"{attention}.v.weight",
        "o": f"{attention}.o.weight",
        "relative_bias": f"{attention}.{umt5_patch.RELATIVE_BIAS_ATTRIBUTE}.weight",
        "ffn_norm": f"{prefix}.1.layer_norm.weight",
        "wi_0": f"{dense}.wi_0.weight",
        "wi_1": f"{dense}.wi_1.weight",
        "wo": f"{dense}.wo.weight",
    }


def rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """`UMT5LayerNorm` の書き下し（上流の分散の f32 への寄せは持たない — f32 では恒等）。"""
    variance = hidden.pow(2).mean(-1, keepdim=True)
    return weight * (hidden * torch.rsqrt(variance + eps))


def self_attention(
    hidden: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    buckets: torch.Tensor,
    shape: EncoderShape,
) -> torch.Tensor:
    """`UMT5LayerSelfAttention` の書き下し（スケール無し・バイアスは表の gather・softmax は
    活性の dtype のまま）。"""
    normed = rms_norm(hidden, weights["attention_norm"], shape.eps)
    batch, length = normed.shape[:2]

    def heads_of(name: str) -> torch.Tensor:
        projected = functional.linear(normed, weights[name])
        return projected.view(batch, -1, shape.heads, shape.head_dim).transpose(1, 2)

    query, key, value = heads_of("q"), heads_of("k"), heads_of("v")
    scores = torch.matmul(query, key.transpose(3, 2))
    bias = functional.embedding(buckets, weights["relative_bias"]).permute([2, 0, 1]).unsqueeze(0)
    probabilities = functional.softmax(scores + bias, dim=-1)
    output = torch.matmul(probabilities, value).transpose(1, 2).contiguous()
    return hidden + functional.linear(output.view(batch, length, -1), weights["o"])


def feed_forward(
    hidden: torch.Tensor, weights: Mapping[str, torch.Tensor], shape: EncoderShape
) -> torch.Tensor:
    """`UMT5LayerFF`（ゲート付き・活性は tanh 近似の GELU — 決定の形）の書き下し。"""
    normed = rms_norm(hidden, weights["ffn_norm"], shape.eps)
    gate = functional.gelu(functional.linear(normed, weights["wi_0"]), approximate="tanh")
    gated = gate * functional.linear(normed, weights["wi_1"])
    return hidden + functional.linear(gated, weights["wo"])


def encode_layerwise(
    source: WeightSource,
    shape: EncoderShape,
    inputs: Sequence[tuple[torch.Tensor, torch.Tensor]],
    dtypes: Sequence[torch.dtype],
    *,
    on_layer: Callable[[int, float], None] | None = None,
) -> dict[torch.dtype, list[torch.Tensor]]:
    """全ケースの出力 `[1, L, d]` を dtype ごとに層逐次で作る（重みは 1 層ずつ読んで捨てる）。

    `inputs` はケースごとの token id `[1, L]` とバケット表 `[L, L]`（どちらも int64）。
    """
    for ids, buckets in inputs:
        length = ids.shape[1]
        if ids.shape[0] != 1 or tuple(buckets.shape) != (length, length):
            raise Umt5ReferenceError(f"入力の形 {tuple(ids.shape)} / {tuple(buckets.shape)}")
    with torch.no_grad():
        vocabulary = torch.unique(torch.cat([ids.flatten() for ids, _ in inputs]))
        table = source.rows(EMBED_KEY, vocabulary)
        # 引くのは dtype へ広げた後 — f64 の経路に f32 の演算を 1 つも置かない（pytest の監視）。
        hidden = {
            dtype: [table.to(dtype)[torch.searchsorted(vocabulary, ids)] for ids, _ in inputs]
            for dtype in dtypes
        }
        for index in range(shape.layers):
            started = time.perf_counter()
            loaded = {name: source.tensor(key) for name, key in layer_keys(index).items()}
            for dtype in dtypes:
                weights = {name: value.to(dtype) for name, value in loaded.items()}
                hidden[dtype] = [
                    feed_forward(self_attention(state, weights, buckets, shape), weights, shape)
                    for state, (_, buckets) in zip(hidden[dtype], inputs, strict=True)
                ]
                del weights
            del loaded
            if on_layer is not None:
                on_layer(index, time.perf_counter() - started)
        final = source.tensor(FINAL_NORM_KEY)
        return {
            dtype: [rms_norm(state, final.to(dtype), shape.eps) for state in states]
            for dtype, states in hidden.items()
        }


# ---------------------------------------------------------------------------
# ケース
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceCase:
    """golden 1 本のケース（id 列は上流の経路で採ったもの）。"""

    name: str
    role: Role
    #: 前処理の前の原文（TS は golden のこの値をトークナイザに通す）。
    prompt: str
    #: 上流の `prompt_clean` の出力。
    cleaned: str
    ids: tuple[int, ...]
    #: 出所（パリティ fixture のケース id・固定プロンプトの名前）。
    source: Mapping[str, Any] = field(default_factory=dict)


#: 原文 → 上流の経路の決着（受理なら `cleaned` と `ids`・拒否なら `reject` —
#: 10a の `umt5_tokenizer.build_case` の形）。
Encode = Callable[[str], Mapping[str, Any]]


def upstream_encoder(model: str) -> Encode:
    """10a の fixture と同じ経路で原文を id 列にする口（上流の `prompt_clean` と鏡像の一致 →
    transformers 5.14.1 と `tokenizer.json` の 2 経路の一致）。"""
    from wan import prompt_clean as pc
    from wan import umt5_tokenizer as ut

    truth = ut.load_truth(ut.tokenizer_snapshot(model))
    compiled = pc.CompiledTables(pc.build_tables())
    clean_truth = pc.upstream_prompt_clean()
    return lambda text: ut.build_case(truth, compiled, clean_truth, "reference", text, "参照")


def _accepted(outcome: Mapping[str, Any], where: str) -> tuple[str, tuple[int, ...]]:
    if "ids" not in outcome:
        raise Umt5ReferenceError(f"{where}: 上流の経路が拒んだ（{outcome.get('reject')}）")
    return str(outcome["cleaned"]), tuple(int(token) for token in outcome["ids"])


def _join(members: Sequence[Mapping[str, Any]]) -> str:
    return " ".join(str(member["text"]) for member in members)


def band_cases(
    parity_cases: Sequence[Mapping[str, Any]], encode: Encode, seed: int = BAND_SEED
) -> list[ReferenceCase]:
    """決定用 6 本（選び方はモジュール doc）。単体の id 列がパリティ fixture の値と一致する
    ことも見る（fixture と上流の経路の取り違えを拾う）。"""
    pool = [case for case in parity_cases if case["id"].startswith(RANDOM_PREFIX) and "ids" in case]
    rng = random.Random(seed)
    singles: list[Mapping[str, Any]] = []
    for low, high in BAND_SINGLE_LENGTHS:
        candidates = [
            case for case in pool if low <= len(case["ids"]) <= high and case not in singles
        ]
        if not candidates:
            raise Umt5ReferenceError(f"長さ {low}〜{high} の受理した乱択が無い")
        singles.append(rng.choice(candidates))
    rest = [case for case in pool if case not in singles]
    rng.shuffle(rest)
    feed: Iterator[Mapping[str, Any]] = iter(rest)
    composites: list[tuple[list[Mapping[str, Any]], Mapping[str, Any]]] = []
    for low, high in BAND_COMPOSITE_LENGTHS:
        members: list[Mapping[str, Any]] = []
        outcome: Mapping[str, Any] = {"ids": []}
        while len(outcome["ids"]) < low:
            candidate = next(feed, None)
            if candidate is None:
                raise Umt5ReferenceError(f"乱択が尽きて長さ {low}〜{high} に届かない")
            trial = encode(_join([*members, candidate]))
            if "ids" in trial and len(trial["ids"]) <= high:
                members.append(candidate)
                outcome = trial
        composites.append((members, outcome))
    cases: list[ReferenceCase] = []
    for single in singles:
        cleaned, ids = _accepted(encode(single["text"]), single["id"])
        if list(ids) != list(single["ids"]):
            raise Umt5ReferenceError(f"{single['id']}: 上流の id 列がパリティ fixture と違う")
        cases.append(_band_case(single["text"], cleaned, ids, [single["id"]], seed))
    for members, outcome in composites:
        cleaned, ids = _accepted(outcome, "合成")
        cases.append(_band_case(_join(members), cleaned, ids, [m["id"] for m in members], seed))
    names = [case.name for case in cases]
    if len(set(names)) != len(names):
        raise Umt5ReferenceError(f"決定用の名前（長さ）が重なった: {names}")
    return cases


def _band_case(
    prompt: str, cleaned: str, ids: tuple[int, ...], members: list[str], seed: int
) -> ReferenceCase:
    return ReferenceCase(
        name=f"band-l{len(ids):04d}",
        role="band",
        prompt=prompt,
        cleaned=cleaned,
        ids=ids,
        source={"parityFixture": PARITY_FIXTURE, "parityCases": members, "seed": seed},
    )


def accept_cases(encode: Encode) -> list[ReferenceCase]:
    """受入れ 4 本（固定プロンプト — 長さが 10a の値と一致することも見る）。"""
    from wan.prompts import FIXED_PROMPTS

    cases = []
    for prompt in FIXED_PROMPTS:
        cleaned, ids = _accepted(encode(prompt.text), prompt.name)
        if len(ids) != FIXED_PROMPT_LENGTHS[prompt.name]:
            raise Umt5ReferenceError(f"{prompt.name}: 長さ {len(ids)} が 10a の値と違う")
        cases.append(
            ReferenceCase(
                name=f"accept-{prompt.name}",
                role="accept",
                prompt=prompt.text,
                cleaned=cleaned,
                ids=ids,
                source={"fixedPrompt": prompt.name},
            )
        )
    return cases


def reference_cases(
    encode: Encode, parity: Path = FIXTURE_DIR / PARITY_FIXTURE
) -> list[ReferenceCase]:
    """決定用 6 本 + 受入れ 4 本。"""
    fixture = json.loads(parity.read_text(encoding="utf-8"))
    return [*band_cases(fixture["cases"], encode), *accept_cases(encode)]


# ---------------------------------------------------------------------------
# golden
# ---------------------------------------------------------------------------


def reference_ratios(f64: torch.Tensor, f32: torch.Tensor) -> dict[str, float]:
    """正規化の分母と、f64 を f32 へ丸めて格納した影響（比はどれも
    `umt5_patch.max_ratio` = 最大絶対差 ÷ 参照の最大絶対値 — TS の `ratioOf` と同じ形）。

    - `f32VsF64`: CPU f32 の参照の f64 の参照に対する比（真の分母）
    - `f32VsStoredF64`: 同じ比を格納した値（f32 へ丸めた f64）で採ったもの（TS が割る分母）
    - `storedF64Rounding`: 格納の丸めそのものの比（≤ 2⁻²⁴ ≈ 6e-8）
    """
    stored = f64.to(torch.float32)
    return {
        "f32VsF64": umt5_patch.max_ratio(f32, f64),
        "f32VsStoredF64": umt5_patch.max_ratio(f32, stored),
        "storedF64Rounding": umt5_patch.max_ratio(stored, f64),
    }


def row_quality(actual: torch.Tensor, base: torch.Tensor) -> dict[str, float]:
    """品質の記録の 3 指標（決定 8 ② — 行ごとのコサイン類似度の最小・相対フロベニウス・
    最大絶対差）。"""
    left = actual.double().reshape(-1, actual.shape[-1])
    right = base.double().reshape(-1, base.shape[-1])
    cosine = functional.cosine_similarity(left, right, dim=-1)
    return {
        "minRowCosine": float(cosine.min()),
        "relativeFrobenius": float((left - right).norm() / right.norm()),
        "maxAbs": float((left - right).abs().max()),
    }


def _check_unquantized(case: ReferenceCase, unquantized: object | None) -> None:
    """量子化なしの参照は受入れにだけ在る（決定用に混ぜない・受入れで欠かさない）。"""
    if (case.role == "accept") != (unquantized is not None):
        raise Umt5ReferenceError(
            f"{case.name}: 量子化なしの参照は受入れだけが持つ（役割 {case.role}）"
        )


def golden_tensors(
    case: ReferenceCase,
    buckets: torch.Tensor,
    f64: torch.Tensor,
    f32: torch.Tensor,
    unquantized: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """1 ケースの golden のテンソル（境界の i64 → i32 は `normalize_boundary_tensor` で）。

    `unquantized` は量子化なしの参照の (f64, f32)（受入れだけ — {@link _check_unquantized}）。
    """
    _check_unquantized(case, unquantized)
    ids = torch.tensor(case.ids, dtype=torch.long)
    tensors = {
        INPUT_IDS_KEY: normalize_boundary_tensor(ids, f"{case.name} の {INPUT_IDS_KEY}"),
        BUCKETS_KEY: normalize_boundary_tensor(buckets, f"{case.name} の {BUCKETS_KEY}"),
        OUTPUT_F64_KEY: f64.to(torch.float32).contiguous(),
        OUTPUT_F32_KEY: f32.to(torch.float32).contiguous(),
    }
    if unquantized is not None:
        high, low = unquantized
        tensors[OUTPUT_UNQUANTIZED_F64_KEY] = high.to(torch.float32).contiguous()
        tensors[OUTPUT_UNQUANTIZED_F32_KEY] = low.to(torch.float32).contiguous()
    return tensors


def golden_metadata(
    case: ReferenceCase,
    weights: ContainerWeights,
    ratios: Mapping[str, float],
    versions: Mapping[str, str],
    unquantized_ratios: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    _check_unquantized(case, unquantized_ratios)
    unquantized = (
        {}
        if unquantized_ratios is None
        else {
            "unquantized": {
                "weights": (
                    "pin した checkpoint の F32 をそのまま（量子化しない）・"
                    "f64 はその値を広げたもの"
                ),
                "output": "f64 は f32 へ丸めて格納・品質の記録の基準（門ではない）",
                "ratios": dict(unquantized_ratios),
            }
        }
    )
    return {
        "format": REFERENCE_FORMAT,
        "case": case.name,
        "role": case.role,
        "prompt": case.prompt,
        "cleaned": case.cleaned,
        "length": len(case.ids),
        "source": dict(case.source),
        "container": {"part0Sha256": weights.part0_sha256, "parts": weights.parts},
        "weights": (
            "i8 は packed × 行ごとの scale を f32 で掛けた値（fake-quant と同じ）・"
            "F32 はそのまま・f64 はその f32 の値を広げたもの"
        ),
        "activation": "ゲート付き GELU の tanh 近似（nn.GELU(approximate='tanh')）",
        "outputF64": "f64 の参照を f32 へ丸めて格納（TS の safetensors は F64 を読まない）",
        "ratios": dict(ratios),
        "versions": dict(versions),
        **unquantized,
    }


def write_golden(
    path: Path, tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, Any]
) -> int:
    payload = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    save_file(dict(tensors), str(path), metadata={METADATA_KEY: payload})
    return path.stat().st_size


def _versions() -> dict[str, str]:
    import transformers

    return {"torch": torch.__version__, "transformers": transformers.__version__}


#: 段を測る口（`umt5_export.MemoryMonitor.stage`）。
Stage = Callable[[str], AbstractContextManager[Any]]


def write_references(
    *,
    container: Path,
    out_dir: Path,
    config: Any,
    cases: Sequence[ReferenceCase],
    stage: Stage,
    unquantized: WeightSource,
    embeddings: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """層逐次の f64 / f32 の参照を採り、golden を `out_dir` に書く（容器は書かない）。

    i8 の参照（容器の重み）は全ケースで、量子化なしの参照（`unquantized` — {@link
    CheckpointWeights}）は受入れだけで採る。i8 の回は全ケースを 1 回の層逐次で回す（受入れを
    足す前と同じ呼び方 — 既存の値をバイトで変えない）。

    `embeddings` は bf16 の事前計算資産（固定プロンプトの名前 → `[L, 4096]`）。渡すと
    受入れのケースの要約に量子化なしの f32 の参照との差を足す（記録だけ — 品質の記録の正本は
    TS の e2e）。
    """
    shape = EncoderShape.of(config)
    attention = umt5_patch.bucket_attention(config)
    weights = ContainerWeights(container)
    inputs = [
        (
            torch.tensor([case.ids], dtype=torch.long),
            umt5_patch.relative_position_buckets(len(case.ids), attention),
        )
        for case in cases
    ]
    accepted = [index for index, case in enumerate(cases) if case.role == "accept"]
    layer_seconds: dict[str, list[float]] = {"i8": [], "unquantized": []}

    def on_layer(label: str) -> Callable[[int, float], None]:
        def report(index: int, seconds: float) -> None:
            layer_seconds[label].append(round(seconds, 1))
            print(f"[layer] {label} {index + 1}/{shape.layers} {seconds:.1f} s", flush=True)

        return report

    dtypes = (torch.float64, torch.float32)
    with stage("reference"):
        outputs = encode_layerwise(weights, shape, inputs, dtypes, on_layer=on_layer("i8"))
    with stage("reference-unquantized"):
        plain = encode_layerwise(
            unquantized,
            shape,
            [inputs[index] for index in accepted],
            dtypes,
            on_layer=on_layer("unquantized"),
        )
    unquantized_of = {
        index: (plain[torch.float64][at], plain[torch.float32][at])
        for at, index in enumerate(accepted)
    }
    versions = _versions()
    rows: list[dict[str, Any]] = []
    with stage("write"):
        for index, case in enumerate(cases):
            f64 = outputs[torch.float64][index]
            f32 = outputs[torch.float32][index]
            ratios = reference_ratios(f64, f32)
            pair = unquantized_of.get(index)
            plain_ratios = None if pair is None else reference_ratios(*pair)
            path = out_dir / f"{REFERENCE_PREFIX}{case.name}{CASE_SUFFIX}"
            size = write_golden(
                path,
                golden_tensors(case, inputs[index][1], f64, f32, pair),
                golden_metadata(case, weights, ratios, versions, plain_ratios),
            )
            row: dict[str, Any] = {
                "case": case.name,
                "role": case.role,
                "length": len(case.ids),
                "bytes": size,
                **ratios,
            }
            if pair is not None:
                row["unquantizedRatios"] = plain_ratios
                # i8 の丸めの影響（CPU — GPU の誤差を含まない）。
                row["i8F32VsUnquantizedF32"] = row_quality(f32[0], pair[1][0])
            fixed = case.source.get("fixedPrompt")
            if embeddings is not None and fixed is not None:
                if pair is None:
                    raise Umt5ReferenceError(f"{case.name}: 固定プロンプトなのに受入れでない")
                # bf16 の影響（資産の経路）と、i8 の f32 の参照との差（10c の記録の続き）。
                row["bf16AssetVsUnquantizedF32"] = row_quality(embeddings[fixed], pair[1][0])
                row["bf16AssetVsF32"] = row_quality(embeddings[fixed], f32[0])
            rows.append(row)
            print(f"[golden] {json.dumps(row, ensure_ascii=False)}", flush=True)
    return {
        "container": {"part0Sha256": weights.part0_sha256, "parts": weights.parts},
        "cases": rows,
        "layerSeconds": layer_seconds,
        "goldenBytes": sum(row["bytes"] for row in rows),
    }
