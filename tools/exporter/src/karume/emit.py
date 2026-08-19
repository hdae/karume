"""配布形（safetensors 1 ファイル + `__metadata__` へのグラフ JSON 埋め込み）の書き出し。

格納の分岐は **f16（ADR 0018）と i8（ADR 0019）と i4（ADR 0069）**。分岐が増えても「格納形式
= ランタイムが受理する形式」の対応が 1 箇所で決まるよう、書き出し経路はこの関数だけにする
（ここが枝分かれすると export は緑のまま実行だけ落ちる）。

## 適格判定（ADR 0018 / 0019 — f16 と i8 で**同じ規則**・i4 だけ狭い）

`weight_dtype="f16"` / `"i8"` / `"i4"` で書くとき、圧縮のまま格納するのは**適格な
initializer だけ**:

1. その initializer の消費が `ops.WEIGHT_SLOTS` の重みスロット**だけ**である
   （ランタイム側 `packages/runtime/src/runtime/plan.ts` の
   `eligibleCompressedInitializers` と同じ規則。
   bias も norm 系 weight も混在消費も消費ゼロも適格外 — ADR 0006）。
   **i4 はさらに狭く `linear` の重みスロット限定**（ADR 0069 決定 5 — 実行経路が linear で
   始まる。一般適格でも linear 以外の重み〈embedding / conv 系〉は f32 のまま残す — i8 の
   適格外と同じ受け皿で、ランタイム側の eligible ∩ linearOnly と対）。
2. **逆変換がビット一致**する（= fake-quant 済み）。f16 は `f32 → f16 → f32` の往復、
   i8 は `q8.to(f32) · scale`、i4 は `dequant(unpack(pack(q4)))`（`scale` は fake-quant が
   使った値を**そのまま**）。

2 を条件に入れるのは、丸めていない値を圧縮格納すると「golden は元値・実行は丸め値」に
なって量子化誤差と実装誤差が混ざるから。適格なのに 2 を満たさない initializer は
**fail loudly**（丸めの掛け忘れ・掛ける順序の誤り・畳み込み定数が重みスロットへ流れた、の
いずれかで、どれも黙って通すと E2E の tolerance が意味を失う）。i4 の門だけ**格納した
バイトから**辿るのは、pack / unpack を取り違えても形も型も合う沈黙誤値になるため
（ADR 0069 決定 4 ③）。

## companion scale（ADR 0019 / 0069）

scale は safetensors の**素のテンソル**として同じファイルに入り、IR 側は `storage.scale` で
そのキーを明示宣言する。キーは `_scale_key`（`karume.scale.<重みキー>`）で機械的に作り、
**実テンソルとの衝突**を書き出し前に検査する（衝突すると「別の重みを scale として読む」
形になり、ロードは通って値だけが壊れる）。形は格納で 2 通り — i8 は per-channel の keepdim
broadcast 形、i4 は**同 rank・最終次元だけ group 数**の group 形で、後者は `storage.group_size`
も宣言する。scale は fake-quant が使った値を `quantize.fake_quant_int8` /
`quantize.fake_quant_int4` から受け取ってそのまま書く — ここで amax から引き直すと f32 の
丸めで 1ulp 動きうるので、golden を採ったときの重みとの対応が壊れる。

## safetensors の並び順（ADR 0063 — docs/limitations.md）

Karume のリーダはデータ節を「隙間なく・型ごとの整列単位に整列して」覆うことを要求する。
要素数が奇数の F16（バイト長 ≡ 2 mod 4）の**直後**に F32 / I32 を置くと絶対 offset が
4 の倍数から外れてロードできない。並べ替えはエクスポータの責務なので、書き出し順を
`_write_order` が明示的に決める（`safetensors.torch.save_file` は自前の順序で書くため
使わない — 順序を外部ライブラリの実装詳細に預けない）。並びは**整列単位の降順** —
F32 / I32 / **I4** が 4 バイト整列群（I4 の節は必ず 8 の倍数バイトなので、後続の整列を崩さない
— ADR 0069 追記 2）、次に F16、**I8 は要素サイズ 1 で整列制約が無いぶん任意長を作れる**ので
末尾。書いた直後の `verify.assert_reader_layout` がリーダ規則を写して検査する。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

import torch

from karume.ir import IR_METADATA_KEY, MIN_GROUP_SIZE, IrGraph, IrInitializer, IrStorage
from karume.ops import LINEAR_OP, WEIGHT_CHANNEL_AXES, WEIGHT_SLOTS
from karume.quantize import (
    QuantizeError,
    dequantize_int4,
    group_size_of,
    quantize_to_int4,
    quantize_to_int8,
)

#: 書き出せる格納 dtype（重みスロット向け）。bf16 は実行経路が無い（ADR 0006）。
WEIGHT_DTYPES = ("f32", "f16", "i8", "i4")

#: torch dtype → safetensors dtype 名 / 1 要素の **bit** 数。ここに無い dtype は fail loudly。
#: bit 単位で持つのは packed 4bit（1 バイトに 2 要素 — ADR 0069 決定 2）が要素バイト数で
#: 表せないため。
_SAFETENSORS_DTYPE: Mapping[torch.dtype, tuple[str, int]] = {
    torch.float32: ("F32", 32),
    torch.float16: ("F16", 16),
    torch.int32: ("I32", 32),
    torch.int8: ("I8", 8),
}

#: 圧縮格納 dtype → safetensors dtype 名 / 1 要素の bit 数。ヘッダを**変換前**に決めるための
#: 対応（`_stored_dtype_of`）。torch dtype を経由しないのは i4 のため — packed の器は uint8
#: だが、宣言は `I4` + **論理 numel** で、器の dtype からは引けない（ADR 0069 決定 2）。
_STORAGE_ENCODING: Mapping[str, tuple[str, int]] = {
    "f16": ("F16", 16),
    "i8": ("I8", 8),
    "i4": ("I4", 4),
}

#: 書き出し順の第 1 キー（safetensors dtype → 群）。**整列単位の降順** — 4 バイト整列を必要と
#: する F32 / I32 / I4 を先に置き、次に F16（奇数要素はさらに後ろ）、末尾へ任意長を作れる I8 を
#: 寄せる（上の「並び順」節）。F32 → I32 の順と群内の名前昇順は `safetensors.torch.save_file`
#: が f32 のみのファイルに対して出す並びと一致するので、f16 / i8 / i4 を含まない資産のバイト列
#: はこの writer に切り替えても変わらない。
_DTYPE_GROUP = {"F32": 0, "I32": 1, "I4": 2, "F16": 3, "I8": 4}

#: packed nibble の offset（格納値は `u = q + 8`・値域 [1,15] で 0 は未使用 = 15 準位）。
#: 非対称化する日にはこの定数が「`storage.zero_point` 省略時の既定 = 8」へ読み替わるだけで、
#: pack 形式・pack 順・値域は動かない（ADR 0069 決定 3 の予約 2）。
INT4_OFFSET = 8

#: safetensors のヘッダ長はこの倍数へパディングする（データ節先頭を 4 バイト境界へ載せる
#: ための整列。`save_file` と同じ規約で、余りは空白で埋める）。
_HEADER_ALIGN = 8


class EmitError(ValueError):
    """宣言（グラフ JSON）と格納テンソルの集合が食い違っている / 格納の前提が崩れている。"""


@dataclass(frozen=True)
class StorageBreakdown:
    """格納 dtype ごとの本数とバイト数（ADR 0006 の常設診断のエクスポータ側）。"""

    #: 圧縮のまま GPU 常駐する（= 適格な）initializer。
    compressed_tensors: int
    compressed_bytes: int
    #: f32 のまま格納する initializer（bias / norm 系 weight / 定数 / 適格外の重み）。
    plain_tensors: int
    plain_bytes: int
    #: 量子化格納（i8 / i4）の companion scale（重み本体とは別テンソル）。ランタイムの
    #: `residentCompressedBytes` も scale を足すので、診断の意味を両側で揃える。
    scale_bytes: int = 0

    def describe(self) -> str:
        scale = f", scale {self.scale_bytes:,} B" if self.scale_bytes else ""
        return (
            f"適格 {self.compressed_tensors} 本 / {self.compressed_bytes:,} B{scale}, "
            f"適格外 {self.plain_tensors} 本 / {self.plain_bytes:,} B"
        )


def eligible_compressed_initializers(graph: IrGraph) -> set[str]:
    """圧縮格納のまま GPU 常駐**できる** initializer
    （`packages/runtime/src/runtime/plan.ts` の鏡像）。

    適格 = 「その initializer の消費が融合 5 op の重みスロット（WEIGHT_SLOTS）だけ」。
    MUST: 消費が 1 つでも重みスロット以外にあれば適格外（そのカーネルは f32 として読むので、
    圧縮のまま上げるとビット列の読み替えになる）。消費ゼロも適格外（実行に使われないバイトを
    「GPU 常駐圧縮」と数えると診断が実態からずれる）。
    MUST: `graph.outputs` に載った initializer も適格外（ランタイムの readback は semantic f32
    の 4 バイト / 要素を仮定して重みバッファから写すので、圧縮のまま常駐させると validation で
    落ちるか、極小サイズではビット列の読み替えが黙って返る）。
    """
    initializers = set(graph.initializers)
    eligible: set[str] = set()
    disqualified: set[str] = set(graph.outputs)
    for node in graph.nodes:
        weight_slot = WEIGHT_SLOTS.get(node.op)
        for slot, name in enumerate(node.ins):
            if name not in initializers:
                continue
            if slot == weight_slot:
                eligible.add(name)
            else:
                disqualified.add(name)
    return eligible - disqualified


def weight_channel_axes(graph: IrGraph) -> dict[str, int]:
    """重みスロットで消費される initializer → per-channel 軸
    （`packages/runtime/src/runtime/plan.ts` の鏡像）。

    軸は**消費側の op** から引く（重みの shape だけでは linear `[out,in]` と
    conv_transpose1d `[Cin,Cout,K]` を区別できない）。同じ initializer を軸の違う op が
    消費している場合は 1 つに決まらないので fail loudly。
    """
    axes: dict[str, int] = {}
    for node in graph.nodes:
        slot = WEIGHT_SLOTS.get(node.op)
        if slot is None or slot >= len(node.ins):
            continue
        name = node.ins[slot]
        if name not in graph.initializers:
            continue
        axis = WEIGHT_CHANNEL_AXES[node.op]
        if axes.setdefault(name, axis) != axis:
            raise EmitError(
                f"initializer '{name}': 消費 op ごとに per-channel 軸が違う"
                f"（{axes[name]} と {axis}）— 1 本の scale では表せない"
            )
    return axes


def linear_weight_initializers(graph: IrGraph) -> set[str]:
    """重みスロットでの消費が `linear` **だけ**の initializer（i4 の適格集合 — ADR 0069 決定 5）。

    i4 の実行経路は linear の重みスロット限定で始まるので、他の重みスロット（conv 系 /
    embedding）でも消費される initializer は i4 で格納できない。`weight_channel_axes` と
    同じく**消費側の op** から引く（重みの shape だけでは区別できない）。
    """
    linear: set[str] = set()
    other: set[str] = set()
    for node in graph.nodes:
        slot = WEIGHT_SLOTS.get(node.op)
        if slot is None or slot >= len(node.ins):
            continue
        name = node.ins[slot]
        if name not in graph.initializers:
            continue
        (linear if node.op == LINEAR_OP else other).add(name)
    return linear - other


def _scale_key(tensor_key: str) -> str:
    """companion scale の safetensors キー（重みキーから機械的に作る）。"""
    return f"karume.scale.{tensor_key}"


def pack_int4(quantized: torch.Tensor) -> torch.Tensor:
    """`q ∈ [−7,7]`（int8 の器・論理形）を packed 4bit（uint8 の 1 次元）へ詰める。

    MUST: **平坦添字で連続する 2 要素が 1 バイト**に入り、要素 `2i` が**下位** nibble・
    `2i+1` が**上位** nibble（ADR 0069 決定 4 — `unpack4xU8` 2 発の展開順と対）。格納値は
    offset 8 の unsigned nibble `u = q + 8`。llama.cpp Q4_0 の split-half 順と取り違えても
    形も型も合う**沈黙誤値**にしかならないので、順序の正本はここで、`tests/test_emit.py` が
    バイト値で固定する。

    MUST: 要素数が奇数の形は fail loudly（末尾要素が半バイトだけ突き出し、テンソル長が宣言
    から一意に決まらない）— i4 の量子化軸は `group_size`（2 冪かつ 16 以上）で割り切れるので
    到達しないが、ここは pack 順の正本なので前提を黙って仮定しない。
    """
    flat = quantized.reshape(-1)
    if flat.numel() % 2:
        raise EmitError(
            f"packed 4bit の要素数 {flat.numel()} が奇数（1 バイト 2 要素で詰められない）"
        )
    nibbles = (flat + INT4_OFFSET).to(torch.uint8)
    return nibbles[0::2] | (nibbles[1::2] << 4)


def unpack_int4(packed: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """packed 4bit を `q ∈ [−7,7]`（int8 の器・`shape` の論理形）へ戻す（`pack_int4` の逆）。

    ランタイム側の展開式（WGSL / CPU 展開）と同じ対応: 平坦添字 `i` の要素は
    `packed[i / 2]` の `i % 2 == 0 ? 下位 : 上位` nibble から `u − 8` で戻る。
    """
    low = (packed & 0x0F).to(torch.int16) - INT4_OFFSET
    high = (packed >> 4).to(torch.int16) - INT4_OFFSET
    return torch.stack((low, high), dim=-1).reshape(tuple(shape)).to(torch.int8)


def _is_f16_exact(tensor: torch.Tensor) -> bool:
    """f32 → f16 → f32 の往復がビット一致するか（= fake-quant 済みか）。"""
    return bool(torch.equal(tensor, tensor.to(torch.float16).to(torch.float32)))


def _unrounded(name: str, tensor_key: str, detail: str) -> EmitError:
    return EmitError(
        f"initializer '{name}' ({tensor_key}) は重みスロット適格なのに {detail} —"
        " fake-quant が未適用か、参照採取より後に掛かっている"
        "（ADR 0006: 丸めは参照・golden の採取より前 MUST）"
    )


@dataclass(frozen=True)
class _Conversion:
    """格納の直前に 1 本ずつ掛ける圧縮変換（計画段では**掛けない**）。

    `name` は診断メッセージ用の IR 値名、`scale` は i8 のときの per-channel scale
    （fake-quant が使った値そのまま — ADR 0019）。
    """

    dtype: str
    name: str
    scale: torch.Tensor | None = None


@dataclass(frozen=True)
class _StoragePlan:
    """圧縮格納の計画。重み本体には 1 バイトも触らない。"""

    #: IR 値名 → 新しい initializer 宣言（全バイトを書き終えてから commit する）。
    declarations: dict[str, IrInitializer]
    #: 追加で格納する companion scale（safetensors キー → テンソル）。scale は小さいので
    #: 計画段で実体化してよい（重み本体と違って集合で持ってもピークに出ない）。
    scales: dict[str, torch.Tensor]
    #: safetensors キー → 書き出し直前に掛ける変換。
    conversions: dict[str, _Conversion]


#: 変換なし（素のテンソルをそのまま書く）— `_write_order` / `_save_ordered` の既定。
_NO_CONVERSIONS: Mapping[str, _Conversion] = MappingProxyType({})


def _plan_i8(
    graph: IrGraph,
    reserved: Set[str],
    name: str,
    tensor: torch.Tensor,
    scales: Mapping[str, torch.Tensor],
    axes: Mapping[str, int],
) -> tuple[str, torch.Tensor, IrInitializer]:
    """i8 格納の計画（scale のキーとテンソル・新しい宣言）を返す — 重みには触らない。

    `reserved` は「既に埋まっている格納テンソルキー」（入力ぶん + 先行して計画した scale）。
    量子化そのものと逆変換のビット一致検査は、実データを読む重い側なので
    `_convert_for_storage`（書き出し直前）へ回す。
    """
    initializer = graph.initializers[name]
    key = initializer.tensor
    scale = scales.get(key)
    if scale is None:
        # 適格なのに scale が無い = fake-quant が届いていない重み（畳み込み定数が重み
        # スロットへ流れた / 対象モジュール型の取りこぼし）。f32 のまま落とすと VRAM 削減が
        # 黙って減るだけなので、i8 指定では受理しない。
        raise _unrounded(name, key, "per-channel scale が無い")
    if scale.dtype is not torch.float32:
        # companion scale は F32 固定（ADR 0019 / docs/ir-v1.md）。writer は F16 もそのまま
        # 直列化でき、逆変換の等値検査も「同じ f16 scale で fake-quant 済み」なら通ってしまう
        # ので、計画段で落とす（診断が「書いた後の verify」から前倒しになる）。
        raise EmitError(
            f"initializer '{name}' ({key}): scale の dtype が {scale.dtype} — F32 のみ受理する"
        )
    axis = axes.get(name)
    expected = [dim if index == axis else 1 for index, dim in enumerate(tensor.shape)]
    if list(scale.shape) != expected:
        raise EmitError(
            f"initializer '{name}' ({key}): scale {list(scale.shape)} が重み"
            f" {list(tensor.shape)} の軸 {axis} の keepdim 形 {expected} でない"
        )
    scale_key = _scale_key(key)
    if scale_key in reserved:
        raise EmitError(
            f"initializer '{name}': scale テンソルのキー '{scale_key}' が実テンソルと衝突する"
        )
    return (
        scale_key,
        scale.detach().contiguous(),
        IrInitializer(tensor=key, storage=IrStorage(dtype="i8", scale=scale_key)),
    )


def _plan_i4(
    graph: IrGraph,
    reserved: Set[str],
    name: str,
    tensor: torch.Tensor,
    scales: Mapping[str, torch.Tensor],
    linear_weights: Set[str],
) -> tuple[str, torch.Tensor, IrInitializer]:
    """i4 格納の計画（scale のキーとテンソル・新しい宣言）を返す — 重みには触らない。

    `_plan_i8` と同じ流儀（キー生成・衝突検査・scale の dtype 検査）で、違うのは 3 点:
    適格が **linear の重みスロット限定**（ADR 0069 決定 5）・scale が **group 形**・宣言が
    `storage.group_size` を持つこと。group 長は**渡された scale の形から引く**
    （`quantize.group_size_of` — 別引数で受けると fake-quant が使った group と別の値を宣言
    できてしまい、形も型も合う沈黙誤値になる）。

    量子化・pack と逆変換のビット一致検査は、実データを読む重い側なので
    `_convert_for_storage`（書き出し直前）へ回す。
    """
    initializer = graph.initializers[name]
    key = initializer.tensor
    if name not in linear_weights:
        raise EmitError(
            f"initializer '{name}' ({key}): 格納 i4 は linear の重みスロットだけ"
            "（ADR 0069 決定 5 — 実行経路が linear 限定で始まる）"
        )
    scale = scales.get(key)
    if scale is None:
        # 適格なのに scale が無い = fake-quant が届いていない重み（`_plan_i8` と同じ理由）。
        raise _unrounded(name, key, "group scale が無い")
    if scale.dtype is not torch.float32:
        raise EmitError(
            f"initializer '{name}' ({key}): scale の dtype が {scale.dtype} — F32 のみ受理する"
        )
    try:
        group_size = group_size_of(tensor, scale)
    except QuantizeError as cause:
        raise EmitError(f"initializer '{name}' ({key}): {cause}") from cause
    # 受理集合（2 冪かつ 16 以上）は宣言層と同じ規則を**書く前に**張る（`_plan_i8` が scale の
    # keepdim 形を verify と二段で見るのと同じ流儀）。書いた後の門は `verify.parse_ir_graph`。
    if group_size & (group_size - 1) or group_size < MIN_GROUP_SIZE:
        raise EmitError(
            f"initializer '{name}' ({key}): scale から引いた group_size {group_size} が"
            f" 2 冪かつ {MIN_GROUP_SIZE} 以上でない（ADR 0069 決定 2）"
        )
    scale_key = _scale_key(key)
    if scale_key in reserved:
        raise EmitError(
            f"initializer '{name}': scale テンソルのキー '{scale_key}' が実テンソルと衝突する"
        )
    return (
        scale_key,
        scale.detach().contiguous(),
        IrInitializer(
            tensor=key,
            storage=IrStorage(dtype="i4", scale=scale_key, group_size=group_size),
        ),
    )


def _plan_weight_dtype(
    graph: IrGraph,
    tensors: Mapping[str, torch.Tensor],
    weight_dtype: str,
    weight_scales: Mapping[str, torch.Tensor],
    weight_dtype_overrides: Mapping[str, str],
) -> _StoragePlan:
    """適格な重みの圧縮格納を**計画だけ**する（宣言も実体もまだ書き換えない）。

    MUST: この段で圧縮テンソルを作らない。全件を先に変換すると、圧縮側の集合が呼び出し側の
    f32 集合と**同時に**生きてピーク RAM が両者の和になる（Irodori 規模で f32 3.44GB に
    f16 1.72GB / i8 0.87GB が重なる）。ここで済ませるのは実データを読まない検査だけ
    （格納 dtype の妥当性・適格判定・scale の有無と形〈i8 = keepdim / i4 = group〉・
    scale キーの衝突・「適格 0 本」）で、実データを読む検査は `_convert_for_storage` が
    書き出し直前に 1 本ずつ受け持つ。

    `weight_dtype_overrides` は**テンソルキー（FQN）→ 格納 dtype** の明示指定（混成格納 —
    LLM の「embedding は i8・linear は i4」が初出）。既定 `weight_dtype` が適格フィルタで
    **静かに f32 へ残す**のに対し、明示指定は**満たせなければ fail loudly**（未知キー・
    適格外・非 f32 実体・i4 × 非 linear のどれも通さない）— 呼び出し側の意図が 1 本単位で
    書かれている以上、黙って別の格納にする余地は無い。`"f32"` の明示は「圧縮既定からの
    除外」として使える。
    """
    if weight_dtype not in WEIGHT_DTYPES:
        raise EmitError(
            f"格納 dtype '{weight_dtype}' は書き出せない（{' / '.join(WEIGHT_DTYPES)}）"
        )
    for key, dtype in weight_dtype_overrides.items():
        if dtype not in WEIGHT_DTYPES:
            raise EmitError(
                f"weight_dtype_overrides['{key}'] の格納 dtype '{dtype}' は書き出せない"
                f"（{' / '.join(WEIGHT_DTYPES)}）"
            )
    plan = _StoragePlan(declarations={}, scales={}, conversions={})
    if weight_dtype == "f32" and not weight_dtype_overrides:
        return plan
    eligible = eligible_compressed_initializers(graph)
    initializer_by_key = {graph.initializers[name].tensor: name for name in graph.initializers}
    unknown = sorted(set(weight_dtype_overrides) - set(initializer_by_key))
    if unknown:
        raise EmitError(
            f"weight_dtype_overrides のキー {unknown} がどの initializer のテンソルでもない"
            "（FQN の綴りを確認する — 黙って無視すると指定した圧縮が静かに消える）"
        )
    for key in sorted(weight_dtype_overrides):
        if weight_dtype_overrides[key] == "f32":
            continue
        name = initializer_by_key[key]
        if name not in eligible:
            raise EmitError(
                f"initializer '{name}' ({key}): 明示指定 '{weight_dtype_overrides[key]}' だが"
                "適格でない（重みスロット以外の消費がある — 混ざった消費は ADR 0018/0019 の"
                "適格集合の外）"
            )
    requested = {weight_dtype, *weight_dtype_overrides.values()}
    axes = weight_channel_axes(graph) if "i8" in requested else {}
    linear_weights = linear_weight_initializers(graph) if "i4" in requested else set()
    reserved = set(tensors)
    for name in sorted(eligible):
        key = graph.initializers[name].tensor
        dtype = weight_dtype_overrides.get(key, weight_dtype)
        if dtype == "f32":
            continue
        tensor = tensors[key]
        if tensor.dtype is not torch.float32:
            # i32 格納（記号依存定数 — ADR 0010）が重みスロットに来ることはないが、来たなら
            # 意味論ごと違う話なので黙って圧縮格納にしない。明示指定なら fail loudly。
            if key in weight_dtype_overrides:
                raise EmitError(
                    f"initializer '{name}' ({key}): 明示指定 '{dtype}' だが実体が"
                    f" {tensor.dtype}（圧縮格納は f32 実体のみ）"
                )
            continue
        # i4 の適格は **linear の重みスロット限定**（ADR 0069 決定 5）— 既定 i4 では linear
        # 以外（embedding / conv 系）を f32 のまま静かに残す。i8 の適格外が f32 で残るのと
        # 同じ設計で、ランタイム側（executor.ts の eligible ∩ linearOnly）と対。ここで
        # 除外しないと、linear + embedding を持つ普通の LLM が「embedding を i4 にできない」
        # で export 不能になる（Codex 波 F 指摘 I4-ELIG-01）。明示指定の i4 × 非 linear は
        # `_plan_i4` 自身の門が fail loudly で受ける。
        if dtype == "i4" and key not in weight_dtype_overrides and name not in linear_weights:
            continue
        if dtype == "f16":
            declaration = IrInitializer(tensor=key, storage=IrStorage(dtype="f16"))
            conversion = _Conversion(dtype="f16", name=name)
        else:
            if dtype == "i8":
                scale_key, scale, declaration = _plan_i8(
                    graph, reserved, name, tensor, weight_scales, axes
                )
            else:
                scale_key, scale, declaration = _plan_i4(
                    graph, reserved, name, tensor, weight_scales, linear_weights
                )
            plan.scales[scale_key] = scale
            reserved.add(scale_key)
            conversion = _Conversion(dtype=dtype, name=name, scale=scale)
        plan.conversions[key] = conversion
        plan.declarations[name] = declaration
    committed = {**graph.initializers, **plan.declarations}
    if weight_dtype != "f32" and not any(
        init.storage.dtype == weight_dtype for init in committed.values()
    ):
        # ADR 0006 の「圧縮指定なのに適格 0MB を沈黙させない」をエクスポータ側でも張る。
        # 明示指定側は 1 本単位で計画済みか fail loudly 済みなので、ここで見るのは既定だけ。
        raise EmitError(
            f"格納 {weight_dtype} を指定したが適格な重みスロットが 1 本も無い"
            f"（融合 op の重みを持たないグラフに {weight_dtype} を指定していないか確認する）"
        )
    return plan


def _convert_for_storage(key: str, tensor: torch.Tensor, conversion: _Conversion) -> torch.Tensor:
    """圧縮格納への変換と、実データを読む適格性検査（重い側 — 1 本ぶんだけ生かす）。

    ここで落ちるとデータ節を書きかけたファイルが残る（`write_model` の docstring）。
    """
    if conversion.dtype == "f16":
        if not _is_f16_exact(tensor):
            raise _unrounded(conversion.name, key, "f16 で表現できない値を含む")
        return tensor.to(torch.float16)
    if conversion.dtype == "i8":
        quantized = quantize_to_int8(tensor, conversion.scale)
        # MUST: scale は fake-quant が使った値**そのまま**で逆変換する（再計算禁止 — ADR 0019）。
        # ここが一致しない = 格納値と golden を採ったときの重みが別物、という一点に集約される。
        if not torch.equal(quantized.to(torch.float32) * conversion.scale, tensor):
            raise _unrounded(
                conversion.name, key, "i8 × per-channel scale で逆変換してもビット一致しない"
            )
        return quantized
    packed = pack_int4(quantize_to_int4(tensor, conversion.scale))
    # MUST: 逆変換は**これから書くバイトから**辿る（ADR 0069 決定 4 ③）。量子化の戻り値と
    # 突き合わせると pack / unpack の取り違え（沈黙誤値 — 形も型も合う）がこの門を素通りする。
    restored = dequantize_int4(unpack_int4(packed, tensor.shape), conversion.scale)
    if not torch.equal(restored, tensor):
        raise _unrounded(conversion.name, key, "i4 × group scale で逆変換してもビット一致しない")
    return packed


#: 格納 dtype → 1 要素の **bit** 数（IR の宣言だけからバイト数を導出するための表）。
#:
#: 語彙は verify の `STORAGE_DTYPES` に揃える（`bf16` は宣言だけ受理する格納 dtype で、
#: 実行可否は `verify.assert_runtime_support` が単独で持つ）— ここは**バイト数を数えるだけ**の
#: 層なので、読めるグラフを数えられない状態を作らない。bit 単位なのは packed 4bit
#: （1 バイトに 2 要素 — ADR 0069 決定 2）が要素バイト数で表せないため。
_STORAGE_BITS = {"f32": 32, "f16": 16, "bf16": 16, "i8": 8, "i4": 4, "i32": 32}
#: 圧縮に数えない格納 dtype（= 素の 4 バイト表現）。圧縮側を列挙すると格納 dtype を足すたびに
#: 2 箇所直す形になるので、**否定形**で書く。
_PLAIN_STORAGE_DTYPES = ("f32", "i32")
#: companion scale の要素バイト数（F32 固定 — ADR 0019）。
_SCALE_BYTES = 4


def storage_breakdown(graph: IrGraph) -> StorageBreakdown:
    """格納宣言から内訳を数える（実テンソルを持ち回らず、毎回グラフから導出する）。

    initializer の shape は記号を含まない静的形であることが IR v1 の規則（verify.py）なので、
    宣言 shape × 格納 dtype の bit 幅で実バイト数がそのまま出る。scale も同じく宣言から
    導出する（実テンソルを見ない）— i8 は「チャネル軸長 × 4B」、i4 は「group 数 =
    要素数 / group_size × 4B」。
    """
    compressed_tensors = compressed_bytes = plain_tensors = plain_bytes = scale_bytes = 0
    axes = weight_channel_axes(graph)
    for name, initializer in graph.initializers.items():
        shape = [int(dim) for dim in graph.values[name].shape]
        count = 1
        for dim in shape:
            count *= dim
        dtype = initializer.storage.dtype
        nbytes = count * _STORAGE_BITS[dtype] // 8
        if dtype not in _PLAIN_STORAGE_DTYPES:
            compressed_tensors += 1
            compressed_bytes += nbytes
            if dtype == "i8":
                scale_bytes += shape[axes[name]] * _SCALE_BYTES
            elif dtype == "i4":
                # 整除は読めるグラフなら保証済み（verify の `_check_group_quantized_shape`）。
                # 存在は型の上でだけ optional なので、既定で埋めず言い直す — 1 で埋めると
                # 「scale が重みと同じ本数ある」という別のモデルの数字が黙って出る。
                group_size = initializer.storage.group_size
                if group_size is None:
                    raise EmitError(f"initializer '{name}': 格納 i4 なのに group_size が無い")
                scale_bytes += count // group_size * _SCALE_BYTES
        else:
            plain_tensors += 1
            plain_bytes += nbytes
    return StorageBreakdown(
        compressed_tensors=compressed_tensors,
        compressed_bytes=compressed_bytes,
        plain_tensors=plain_tensors,
        plain_bytes=plain_bytes,
        scale_bytes=scale_bytes,
    )


def _write_order(
    tensors: Mapping[str, torch.Tensor],
    conversions: Mapping[str, _Conversion] = _NO_CONVERSIONS,
) -> list[str]:
    """書き出し順（データ節に並ぶ順）。

    第 1 キーは dtype 群（F32 → I32 → **I4** → F16 → **I8**）、F16 のうち**要素数が奇数のもの**
    （バイト長 ≡ 2 mod 4）はさらに後ろへ寄せる。第 2 キーは名前昇順。

    これで「4 バイト整列を要求するテンソルの前に、4 の倍数でないバイト長のテンソルが来る」
    ことが構造的に起こらない — 奇数 F16 より前は全て 4 の倍数長なので累積 offset は 4 の
    倍数を保ち、奇数 F16 どうしは 2 バイト整列だけを要求するので偶数 offset で足りる。
    I4 も**先頭 4 バイト整列**を要求する（要素整列の概念が無く、展開カーネルが `array<u32>` で
    束縛する — ADR 0069 決定 2）ぶん F32 / I32 と同じ群に置く。バイト長は必ず 8 の倍数
    （量子化軸が 2 冪 ≥ 16 の group_size で割り切れる ⇒ 要素数は 16 の倍数）なので、群内の
    どこに来ても後続の整列を崩さない。I8 は要素サイズ 1 で整列制約が無いかわりに**任意の
    バイト長**を作るので、群の末尾に置く（F16 より前に来ると 2 バイト整列すら壊す）。

    `conversions` を渡すと、そのテンソルは**変換後**の dtype で並べる（変換はまだ掛けない
    — 順序は変換前の要素数と計画だけで決まる）。
    """

    def key(name: str) -> tuple[int, int, str]:
        tensor = tensors[name]
        dtype_name, bits = _stored_dtype_of(name, tensor, conversions.get(name))
        # 「後ろへ寄せる」の対象は F16 だけ（I8 は既に最後の群で、群内の順序は整列に
        # 影響しない — 名前昇順のまま安定させる）。
        odd = 1 if dtype_name == "F16" and (tensor.numel() * bits // 8) % 4 != 0 else 0
        return (_DTYPE_GROUP[dtype_name], odd, name)

    return sorted(tensors, key=key)


def _dtype_of(tensor: torch.Tensor, name: str) -> tuple[str, int]:
    entry = _SAFETENSORS_DTYPE.get(tensor.dtype)
    if entry is None:
        raise EmitError(
            f"テンソル '{name}': dtype {tensor.dtype} は配布形に書けない"
            f"（{' / '.join(label for label, _ in _SAFETENSORS_DTYPE.values())} のみ）"
        )
    return entry


def _stored_dtype_of(
    name: str, tensor: torch.Tensor, conversion: _Conversion | None
) -> tuple[str, int]:
    """**格納後**の safetensors dtype と 1 要素の bit 数。

    圧縮変換は**論理**要素数も shape も変えないので、ヘッダは変換前のテンソルと計画だけで
    決まる（= データ節を流しながら書ける）。i4 で変わるのはバイト長だけで、それも
    `論理 numel × bits / 8` として同じ式から出る（ADR 0069 決定 2 — shape は論理形のまま）。
    """
    if conversion is not None:
        return _STORAGE_ENCODING[conversion.dtype]
    return _dtype_of(tensor, name)


def _payload_bytes(name: str, dtype_name: str, count: int, bits: int) -> int:
    """論理要素数と bit 幅からデータ節のバイト長を出す。

    MUST: bit 総量が byte 境界に乗らない形（I4 の要素数が奇数）は fail loudly — 末尾要素が
    半バイトだけ突き出し、テンソルの長さが宣言から一意に決まらない
    （リーダ側 `verify.assert_reader_layout` と同じ規則を書き出し側でも張る）。
    """
    total = count * bits
    if total % 8:
        raise EmitError(
            f"テンソル '{name}': {dtype_name}（1 要素 {bits}bit）の要素数 {count} が"
            " 奇数で byte 境界に乗らない"
        )
    return total // 8


def _save_ordered(
    path: Path,
    tensors: Mapping[str, torch.Tensor],
    order: Sequence[str],
    metadata: Mapping[str, str],
    conversions: Mapping[str, _Conversion] = _NO_CONVERSIONS,
) -> None:
    """safetensors を**指定順で**書く（`save_file` は自前の順序で書くので使わない）。

    レイアウトは仕様どおり `[u64 LE ヘッダ長][ヘッダ JSON][データ節]`。ヘッダ JSON は
    `__metadata__` を先頭に、テンソルはデータ節に並ぶ順で載せる（HF のリーダはヘッダの
    宣言順にオフセットの連続性を見る）。

    `conversions` に載ったテンソルは**書く直前に 1 本ずつ**圧縮格納へ変換し、書いたら
    即座に手放す。MUST: 変換済みを次の 1 本へ持ち越さない（同時に生きる圧縮テンソルを
    1 本に保つのがこの writer の存在理由）。変換に伴う検査が落ちるとデータ節を書きかけた
    ファイルが残る — 配布物の原子性は `pipeline.export_to_file` の一時ファイル層が持つ。
    """
    header: dict[str, object] = {"__metadata__": dict(metadata)}
    offset = 0
    for name in order:
        tensor = tensors[name]
        dtype_name, bits = _stored_dtype_of(name, tensor, conversions.get(name))
        nbytes = _payload_bytes(name, dtype_name, tensor.numel(), bits)
        header[name] = {
            "dtype": dtype_name,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = -len(blob) % _HEADER_ALIGN
    blob += b" " * padding
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(len(blob).to_bytes(8, "little"))
        handle.write(blob)
        for name in order:
            tensor = tensors[name]
            conversion = conversions.get(name)
            if conversion is not None:
                tensor = _convert_for_storage(name, tensor, conversion)
            # memoryview 経由で書く（`tobytes()` は 1 本ぶんの複製を作る — DiT の重みは
            # 1 テンソルで数百 MB あり、ピーク RAM をそのぶん押し上げる）。
            handle.write(memoryview(tensor.numpy()).cast("B"))
            del tensor


def write_model(
    path: str | Path,
    graph: IrGraph,
    tensors: dict[str, torch.Tensor],
    *,
    weight_dtype: str = "f32",
    weight_scales: Mapping[str, torch.Tensor] | None = None,
    weight_dtype_overrides: Mapping[str, str] | None = None,
) -> Path:
    """グラフと格納テンソルを 1 ファイルに書き、書いたパスを返す。

    `weight_dtype` が `"f16"` / `"i8"` / `"i4"` のとき、**適格な重みスロットだけ**が圧縮格納に
    なる（宣言と実体が 1 経路で決まる — 別々に決めると「宣言 f16 / 実体 f32」の沈黙誤読が
    作れる）。i4 の適格は linear の重みスロット限定（ADR 0069 決定 5）。

    `weight_dtype_overrides`（テンソルキー → 格納 dtype）は 1 本単位の明示指定で、既定
    `weight_dtype` に**優先**する（混成格納 — 意味と fail loudly の線引きは
    `_plan_weight_dtype` の docstring）。scale が要る dtype を混ぜるときは `weight_scales` に
    i8 / i4 の台帳を**合流して**渡す（キー空間が FQN で重ならないので 1 つの Mapping で足りる）。

    MUST: 「計画（実データを読まない検査）→ データ節を 1 本ずつ変換しながら流す →
    全バイトを書き終えてから宣言を commit」の 3 段で進める。

    1. 圧縮テンソルを先にまとめて作ると、呼び出し側が持つ f32 集合と同時に生きてピーク
       RAM が両者の和になる（Irodori 規模で f32 3.44GB + f16 1.72GB / i8 0.87GB）。
       ヘッダは「変換前の shape・要素数 + 計画の格納 dtype」だけで決まるので、データ節を
       流しながらでも宣言は動かない。
    2. MUST: commit は `dataclasses.replace` の**ビューの中だけ**に閉じ、渡された `graph` は
       1 バイトも変えない。呼び出し側の宣言を書き換えると、同じ graph を使い回す経路
       （別 dtype で書き直す / 内訳を数える）が前回の圧縮宣言を引きずる — f32 の計画は
       空プランで宣言を**復元しない**ので、f16 → f32 の順に書くと「宣言 f16 / 実体 f32」の
       壊れたコンテナがそのまま出る。書いた後の宣言が要る呼び手は `verify_model` の戻り
       （= ファイルから読み直したグラフ）を使う。

    NOTE: 書き出し中の検査（f16 の往復・i8 / i4 の逆変換）が落ちると書きかけのファイルが残る
    — 配布物の原子性は `pipeline.export_to_file` の一時ファイル層が持つ。

    `weight_scales` は `quantize.fake_quant_int8` / `quantize.fake_quant_int4` が返した
    **FQN → scale** の台帳（`"i8"` / `"i4"` のときだけ要る）。キーは safetensors のテンソル
    キーと同じ空間で、`id(tensor)` 突合はしない（ADR 0006）。i4 の `storage.group_size` は
    この scale の形から引く（`quantize.group_size_of`）。
    """
    declared = {init.tensor for init in graph.initializers.values()}
    stored = set(tensors)
    if declared != stored:
        raise EmitError(
            "宣言テンソルと格納テンソルが一致しない: "
            f"欠落 {sorted(declared - stored)} / 余剰 {sorted(stored - declared)}"
        )
    out = Path(path)
    contiguous = {key: value.detach().contiguous() for key, value in tensors.items()}
    plan = _plan_weight_dtype(
        graph, contiguous, weight_dtype, weight_scales or {}, weight_dtype_overrides or {}
    )
    source = {**contiguous, **plan.scales}
    committed = replace(graph, initializers={**graph.initializers, **plan.declarations})
    _save_ordered(
        out,
        source,
        _write_order(source, plan.conversions),
        {IR_METADATA_KEY: committed.to_json()},
        plan.conversions,
    )
    return out
