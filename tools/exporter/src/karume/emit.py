"""配布形の**格納変換**と、容器へ渡す材料（生バイト + 束縛表）の組み立て。

器そのもの（`krm` の物理形式・descriptor・block / part の配置）は `karume.container` が持ち、
公開の 3 段（書く → 読み直して検証 → 据え替え）は `karume.publish` が持つ。ここが決めるのは
**どの initializer をどの格納で焼くか**だけで、それが {@link stored_model} の 1 本道に閉じる
（ここが枝分かれすると export は緑のまま実行だけ落ちる）。

格納の分岐は **f16（ADR 0018）と i8（ADR 0019）と i4（ADR 0069）**。

## 適格判定（ADR 0018 / 0019 — f16 と i8 で**同じ規則**・i4 だけ狭い）

`weight_dtype="f16"` / `"i8"` / `"i4"` で書くとき、圧縮のまま格納するのは**適格な
initializer だけ**:

1. その initializer の消費が `ops.WEIGHT_SLOTS` の重みスロット**だけ**である
   （ランタイム側 `packages/runtime/src/runtime/plan.ts` の
   `eligibleCompressedInitializers` と同じ規則。
   bias も norm 系 weight も混在消費も消費ゼロも適格外 — ADR 0006）。
   **i4 はさらに狭く {@link I4_WEIGHT_OPS}（`linear` / `embedding` / `conv1d`）の重みスロット
   限定**（ADR 0069 決定 5 と embedding / conv1d 追補 — i4 の展開経路を持つ op はこの 3 つ
   だけで、conv1d はさらに `groups == 1` と格納行長の整除が要る）。一般適格でも conv2d /
   conv_transpose1d の重みは f32 のまま残す — i8 の適格外と同じ受け皿で、ランタイム側の
   eligible ∩ i4Eligible と対。
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

scale は**独立したテンソルキー**として同じ容器に入り、IR 側は `storage.scale` で
そのキーを明示宣言する。キーは `_scale_key`（`karume.scale.<重みキー>`）で機械的に作り、
**実テンソルとの衝突**を書き出し前に検査する（衝突すると「別の重みを scale として読む」
形になり、ロードは通って値だけが壊れる）。形は格納で 2 通り — i8 は per-channel の keepdim
broadcast 形、i4 は **rank2（行数 = 重みの先頭次元・最終次元 = group 数）**の group 形で、
後者は `storage.group_size` も宣言する（rank2 の重みでは「同 rank・最終次元だけ group 数」と
同値で、conv1d `[Cout,Cin,K]` は `[Cout,(Cin·K)/g]` — `quantize.storage_rows`）。scale は
fake-quant が使った値を `quantize.fake_quant_int8` / `quantize.fake_quant_int4` から
受け取ってそのまま書く — ここで amax から引き直すと f32 の
丸めで 1ulp 動きうるので、golden を採ったときの重みとの対応が壊れる。

## 容器へ渡す 3 点（{@link StoredModel}）

配置（block の切り方・part の詰め方・並び順）は `karume.container` の責務なので、ここが渡すのは
「格納宣言を commit したグラフ」「テンソルキー → **格納後の生バイト**」「テンソルキー →
{@link karume.container.Encoding}」の 3 点だけである。

- 生バイトの口は**遅延**（{@link _StoredTensors}）— 引かれた 1 本だけを変換して返す。全件を
  先に変換すると、呼び出し側が持つ f32 集合と同時に生きてピーク RAM が両者の和になる
  （Irodori 規模で f32 3.44GB に f16 1.72GB / i8 0.87GB が重なる）。書き手は実体を**2 度引く**
  （sha256 を採る走査と書き出しの走査 — container-v1 §4.1）ので、同じキーからは**毎回同じ
  バイト列**が返る MUST。変換は決定的なのでこれは成立する。
- 束縛表は {@link karume.container.container_bindings} の 1 本から出す（移行 CLI と**同じ関数**・
  torch を要らない側に在る）。2 経路で導くと「宣言 i4 / 実体 i8」のような、形も型も合う
  沈黙誤値が作れる。
"""

from __future__ import annotations

from collections.abc import Buffer, Iterator, Mapping, Sequence, Set
from dataclasses import dataclass, replace
from typing import Literal

import torch

from karume.container import (
    Encoding,
    bakeable_initializers,
    container_bindings,
    weight_channel_axes,
)
from karume.ir import MIN_GROUP_SIZE, IrGraph, IrInitializer, IrNode, IrStorage
from karume.ops import (
    CONV1D_OP,
    EMBEDDING_OP,
    LINEAR_OP,
    WEIGHT_SLOTS,
    conv1d_attrs,
)
from karume.quantize import (
    DEFAULT_GROUP_SIZE,
    QuantizeError,
    dequantize_int4,
    group_size_of,
    quantize_to_int4,
    quantize_to_int8,
)

#: 書き出せる格納 dtype（重みスロット向け）。bf16 は実行経路が無い（ADR 0006）。
WEIGHT_DTYPES = ("f32", "f16", "i8", "i4")

#: i4 の**実行経路を持つ** op（= i4 適格な重みスロットの消費先 — ADR 0069 決定 5 の linear と
#: その embedding / conv1d 追補）。ランタイム側 `packages/runtime/src/runtime/plan.ts` の
#: `I4_WEIGHT_OPS` の鏡像で、両者が割れると「export は緑・ロードで CPU 展開に落ちる（VRAM 削減が
#: 黙って消える）」か、その逆に「宣言できない格納を要求して export だけが落ちる」形になる。
#:
#: MUST: conv2d と conv_transpose1d を入れない。conv1d に付いた i4 は igemm 経路
#: （`groups == 1` の変種）だけが展開でき、direct カーネル（`groups > 1`）と conv2d /
#: conv_transpose1d の生成入力へ i4 を渡すと不成立 WGSL になる
#: （`packages/runtime/src/kernels/weight-storage.ts`）。`groups == 1` の絞り込みは
#: {@link i4_eligible_initializers} が node の attrs から見る。
#: MUST: conv_transpose1d は**重み `[Cin,Cout,K]` の行軸が先頭でない**ぶん、pack のバイト列が
#: 行の並びと食い違う（`quantize.storage_rows`）— 展開経路を足すだけでは適格にできない
#: （2026-08-20 ユーザー裁定: permuted pack は買わない）。
I4_WEIGHT_OPS: frozenset[str] = frozenset({LINEAR_OP, EMBEDDING_OP, CONV1D_OP})

#: packed nibble の offset（格納値は `u = q + 8`・値域 [1,15] で 0 は未使用 = 15 準位）。
#: 非対称化する日にはこの定数が「`storage.zero_point` 省略時の既定 = 8」へ読み替わるだけで、
#: pack 形式・pack 順・値域は動かない（ADR 0069 決定 3 の予約 2）。
INT4_OFFSET = 8


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
    MUST: 共有 initializer（ADR 0096 段 2 — バイトを持たない宣言）は適格集合に載せない。
    このコンテナに書くバイトが 1 つも無いので「圧縮格納する / しない」の判断対象ではなく、
    載せると `_plan_weight_dtype` が存在しないテンソルキーを引きにいく。
    """
    initializers = bakeable_initializers(graph)
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


def _has_i4_kernel(node: IrNode) -> bool:
    """このノードが重みスロットの i4 を**展開できる**か（{@link I4_WEIGHT_OPS} の絞り込み）。

    conv1d の i4 は igemm 変種（`groups == 1`）だけが展開でき、direct カーネル（`groups > 1`）は
    i4 非対応のまま（ランタイム側 `plan.ts` の `i4Executable` の鏡像）。

    MUST: `groups` は**既定値で補完しない**（ADR 0012）— 契約表の `ops.conv1d_attrs` を通して
    欠落ごと落とす。ここで黙って 1 を仮定すると、depthwise の重みが i4 で常駐して direct
    カーネルが packed バイトを f32 として読む（例外は出ない）。
    """
    if node.op not in I4_WEIGHT_OPS:
        return False
    if node.op != CONV1D_OP:
        return True
    return conv1d_attrs(node.attrs, f"graph.nodes ({node.op})")[3] == 1


def _storage_row_length(graph: IrGraph, name: str) -> int | None:
    """initializer の**格納行長**（`quantize.storage_rows` の行長 = 先頭次元を除く積）。

    group scale の形が rank に依らずこの行長で決まる（ADR 0069 決定 3）ので、i4 の適格判定も
    ここを除数に見る。rank 1 以下・記号次元を含む形・宣言の無い名前は None（格納 i4 の形を
    そもそも持てない）。
    """
    value = graph.values.get(name)
    if value is None or len(value.shape) < 2:
        return None
    length = 1
    for dim in value.shape[1:]:
        if not isinstance(dim, int):
            return None
        length *= dim
    return length


def _scale_group_size(
    graph: IrGraph, scales: Mapping[str, torch.Tensor], name: str, row_length: int
) -> int | None:
    """その重み自身の group scale が宣言している group 長（引けなければ None）。

    源は `quantize.group_size_of` と同じ「**渡された scale の形**だけ」（`[行, group 数]` →
    行長 / group 数）。適格判定と格納宣言が別の源を持つと、fake-quant が使った g と判定に
    使った g が独立に動ける — 食い違っても形も型も合うので沈黙する。

    台帳に無い / group 形でない scale は None を返して呼び手の既定へ委ねる（「適格なのに
    scale が無い」は `_plan_i4` が fail loudly で受けるので、ここで適格から外さない）。
    """
    scale = scales.get(graph.initializers[name].tensor)
    if scale is None or scale.dim() != 2:
        return None
    groups = int(scale.shape[1])
    if groups < 1 or row_length % groups:
        return None
    return row_length // groups


def i4_eligible_initializers(
    graph: IrGraph,
    group_size: int = DEFAULT_GROUP_SIZE,
    scales: Mapping[str, torch.Tensor] | None = None,
) -> set[str]:
    """重みスロットでの消費が {@link I4_WEIGHT_OPS} **だけ**で、格納行長が group 長で
    割り切れる initializer（i4 の適格集合 — ADR 0069 決定 5 とその追補）。

    i4 の展開経路（`unpack4xU8` + group scale）を持つカーネルは linear / embedding / conv1d
    （`groups == 1`）だけなので、それ以外の重みスロット（conv2d / conv_transpose1d /
    groups > 1 の conv1d）でも消費される initializer は i4 で格納できない — そのカーネルは
    packed バイトを f32 として読む（例外は出ない）。`weight_channel_axes` と同じく**消費側の
    op** から引く（重みの shape だけでは区別できない）。

    行長の門（{@link _storage_row_length} % group 長）が要るのは、端数 group を作らない
    MUST（ADR 0069 決定 2）で丸められない重みを**適格から外して静かに f32 に残す**ため。
    ここで拾ってしまうと「fake-quant が届いていない重み」として export 全体が落ちる
    （linear + 割り切れない conv1d を持つ普通のグラフが i4 で書けなくなる — I4-ELIG-01 と
    同型）。

    MUST: 除数は **1 本ずつその重み自身の group scale から引く**（`scales` の台帳 →
    {@link _scale_group_size}）。除数を出荷の既定 group 長（`quantize.DEFAULT_GROUP_SIZE`）に
    固定すると、g16 で丸めた行長 48 の重みが `48 % 32 != 0` で適格から外れ、値は i4 グリッド
    へ丸められたまま **f32 で格納**される — 品質劣化だけ乗ってサイズは 1 バイトも縮まず、
    例外も診断も出ない（ADR 0006 が名指しした「圧縮指定なのに実質 f32」の沈黙）。
    `group_size` は scale を引けない名前だけに掛かる後詰めの既定。
    """
    bakeable = bakeable_initializers(graph)
    executable: set[str] = set()
    other: set[str] = set()
    for node in graph.nodes:
        slot = WEIGHT_SLOTS.get(node.op)
        if slot is None or slot >= len(node.ins):
            continue
        name = node.ins[slot]
        if name not in bakeable:
            continue
        (executable if _has_i4_kernel(node) else other).add(name)
    ledger = scales or {}
    eligible: set[str] = set()
    for name in executable - other:
        row_length = _storage_row_length(graph, name)
        if row_length is None:
            continue
        divisor = _scale_group_size(graph, ledger, name, row_length) or group_size
        if row_length % divisor == 0:
            eligible.add(name)
    return eligible


def _scale_key(tensor_key: str) -> str:
    """companion scale のテンソルキー（重みキーから機械的に作る）。"""
    return f"karume.scale.{tensor_key}"


def pack_int2(quantized: torch.Tensor) -> torch.Tensor:
    """固定整数 [-2,1] を下位2bitから詰める。量子化やscaleの変更は行わない。

    i2 のバイト順の正本（ADR 0097）: 平坦添字で連続する 4 要素が 1 バイトに入り、要素 `4i` が
    最下位 2bit、格納値は offset 2 の `u = q + 2`。書き手（gemma4_qat recipe）は上流の packed
    バイトをそのまま渡すので本体はこの関数を呼ばない。それでも順序を取り違えると形も型も合う
    沈黙誤値にしかならないので、順序はここに置き、`tests/test_i2_storage.py` がバイト値で固定する。
    """
    if quantized.dtype != torch.int8:
        raise EmitError("INT2 の整数列は int8 の器が必要")
    flat = quantized.reshape(-1)
    if flat.numel() % 4:
        raise EmitError("INT2 の要素数は4の倍数が必要")
    if bool(((flat < -2) | (flat > 1)).any()):
        raise EmitError("INT2 の整数は [-2,1] が必要")
    u = (flat + 2).to(torch.uint8)
    return u[0::4] | (u[1::4] << 2) | (u[2::4] << 4) | (u[3::4] << 6)


def unpack_int2(packed: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """packed INT2 を固定整数へ戻す。論理形と全バイト数の一致を要求する。"""
    count = 1
    for dim in shape:
        if type(dim) is not int or dim < 0:
            raise EmitError("INT2 の shape は非負整数が必要")
        count *= dim
    if packed.dtype != torch.uint8 or packed.numel() * 4 != count:
        raise EmitError("INT2 のバイト列と論理 shape が一致しない")
    flat = packed.reshape(-1)
    lanes = torch.stack(tuple((flat >> shift) & 3 for shift in (0, 2, 4, 6)), dim=-1)
    return (lanes.to(torch.int8) - 2).reshape(tuple(shape))


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
class FixedQuantizedWeight:
    """固定整数の packed 実体と scale。論理形は対応する f32/meta Tensor が持つ。"""

    dtype: Literal["i2", "i4", "i8"]
    packed: torch.Tensor
    scale: torch.Tensor


@dataclass(frozen=True)
class _Conversion:
    """格納の直前に 1 本ずつ掛ける圧縮変換（計画段では**掛けない**）。

    `name` は診断メッセージ用の IR 値名、`scale` は i8 のときの per-channel scale
    （fake-quant が使った値そのまま — ADR 0019）。
    """

    dtype: str
    name: str
    scale: torch.Tensor | None = None
    fixed_payload: torch.Tensor | None = None


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
        # companion scale は F32 固定（ADR 0019 / docs/ir-v2.md）。writer は F16 もそのまま
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
    i4_eligible: Set[str],
) -> tuple[str, torch.Tensor, IrInitializer]:
    """i4 格納の計画（scale のキーとテンソル・新しい宣言）を返す — 重みには触らない。

    `_plan_i8` と同じ流儀（キー生成・衝突検査・scale の dtype 検査）で、違うのは 3 点:
    適格が **{@link I4_WEIGHT_OPS} の重みスロット限定**（ADR 0069 決定 5）・scale が
    **group 形**・宣言が `storage.group_size` を持つこと。group 長は**渡された scale の形から引く**
    （`quantize.group_size_of` — 別引数で受けると fake-quant が使った group と別の値を宣言
    できてしまい、形も型も合う沈黙誤値になる）。

    量子化・pack と逆変換のビット一致検査は、実データを読む重い側なので
    `_convert_for_storage`（書き出し直前）へ回す。
    """
    initializer = graph.initializers[name]
    key = initializer.tensor
    if name not in i4_eligible:
        raise EmitError(
            f"initializer '{name}' ({key}): 格納 i4 は"
            f" {' / '.join(sorted(I4_WEIGHT_OPS))} の重みスロットだけ"
            "（ADR 0069 決定 5 とその追補 — conv1d は groups == 1 と格納行長が"
            " group 長で割り切れることも要る）"
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
    # keepdim 形を verify と二段で見るのと同じ流儀）。書いた後の門は
    # `verify.ir_graph_from_container`（据えた容器の束縛表から IR を起こし直す側）。
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


def _plan_fixed_weights(
    graph: IrGraph,
    tensors: Mapping[str, torch.Tensor],
    fixed_weights: Mapping[str, FixedQuantizedWeight],
) -> _StoragePlan:
    """固定 bytes を唯一の値として扱い、再量子化・f32 実体化をしない（ADR 0097）。"""
    bakeable = bakeable_initializers(graph)
    names = {graph.initializers[name].tensor: name for name in bakeable}
    if len(names) != len(bakeable):
        raise EmitError("固定格納の initializer 名とテンソルキーが 1:1 でない")
    unknown = sorted(set(fixed_weights) - set(names))
    if unknown:
        raise EmitError(f"fixed_weights の未知キー: {unknown}")
    eligible = eligible_compressed_initializers(graph)
    plan = _StoragePlan(declarations={}, scales={}, conversions={})
    reserved = set(tensors)
    for key, fixed in fixed_weights.items():
        name = names[key]
        tensor = tensors[key]
        shape = list(tensor.shape)
        if (
            not tensor.is_meta
            or tensor.dtype != torch.float32
            or len(shape) != 2
            or any(dim <= 0 for dim in shape)
            or graph.values[name].dtype != "f32"
            or graph.values[name].shape != shape
            or graph.initializers[name].storage.dtype != "f32"
        ):
            raise EmitError(f"fixed_weights['{key}']: 宣言と同形の正の rank2 f32/meta が必要")
        consumers = [node for node in graph.nodes if name in node.ins]
        if name not in eligible or any(
            node.op not in (LINEAR_OP, EMBEDDING_OP) for node in consumers
        ):
            raise EmitError(f"fixed_weights['{key}']: linear/embedding の重み専用")
        if fixed.dtype not in ("i2", "i4", "i8"):
            raise EmitError(f"fixed_weights['{key}']: dtype は i2/i4/i8 が必要")
        rows, width = shape
        factor = {"i2": 4, "i4": 2, "i8": 1}[fixed.dtype]
        packed_dtype = torch.int8 if fixed.dtype == "i8" else torch.uint8
        packed = fixed.packed
        if (
            packed.device.type != "cpu"
            or packed.dtype != packed_dtype
            or not packed.is_contiguous()
            or width % factor
            or list(packed.shape) != [rows, width // factor]
        ):
            raise EmitError(f"fixed_weights['{key}']: packed の CPU dtype・形・連続配置が違う")
        scale = fixed.scale
        if (
            scale.device.type != "cpu"
            or scale.dtype != torch.float32
            or scale.ndim != 2
            or scale.shape[0] != rows
            or scale.shape[1] <= 0
        ):
            raise EmitError(f"fixed_weights['{key}']: scale は CPU F32 [rows,groups] が必要")
        group_size = None
        if fixed.dtype == "i4":
            groups = scale.shape[1]
            group_size = width // groups
            if width % groups or group_size < 16 or group_size & (group_size - 1):
                raise EmitError(f"fixed_weights['{key}']: I4 group は行を割り切る2冪・16以上が必要")
        elif list(scale.shape) != [rows, 1] or (fixed.dtype == "i2" and width % 16):
            raise EmitError(f"fixed_weights['{key}']: 行 scale [rows,1] と I2 の16要素整列が必要")
        scale_key = _scale_key(key)
        if scale_key in reserved:
            raise EmitError(f"fixed_weights['{key}']: scale キーが衝突する: {scale_key}")
        scale = scale.detach().contiguous()
        reserved.add(scale_key)
        plan.scales[scale_key] = scale
        plan.declarations[name] = IrInitializer(
            tensor=key, storage=IrStorage(dtype=fixed.dtype, scale=scale_key, group_size=group_size)
        )
        plan.conversions[key] = _Conversion(
            dtype=fixed.dtype, name=name, scale=scale, fixed_payload=packed.detach()
        )
    return plan


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
    （名前 ↔ テンソルキーの 1:1・格納 dtype の妥当性・適格判定・scale の有無と形
    〈i8 = keepdim / i4 = group〉・scale キーの衝突・「適格 0 本」）で、実データを読む検査は
    `_convert_for_storage` が書き出し直前に 1 本ずつ受け持つ。

    `weight_dtype_overrides` は**テンソルキー（FQN）→ 格納 dtype** の明示指定（混成格納 —
    LLM の「embedding は i8・linear は i4」が初出）。既定 `weight_dtype` が適格フィルタで
    **静かに f32 へ残す**のに対し、明示指定は**満たせなければ fail loudly**（未知キー・
    適格外・非 f32 実体・i4 × i4 適格外のどれも通さない）— 呼び出し側の意図が 1 本単位で
    書かれている以上、黙って別の格納にする余地は無い。`"f32"` の明示は「圧縮既定からの
    除外」として使える。
    """
    # MUST: 走査は**実体を書く initializer だけ**（共有宣言は貸し手のバイトを借りるので、
    # 格納 dtype の計画対象ではない — ADR 0096 段 2）。
    bakeable = bakeable_initializers(graph)
    initializer_by_key = {graph.initializers[name].tensor: name for name in bakeable}
    if len(initializer_by_key) != len(bakeable):
        # MUST: initializer 名 ↔ テンソルキーは 1:1。潰れると適格検査は後勝ちで残った 1 名しか
        # 見ないのに、計画ループは適格な**全ての**名前を回す — 実体は key 単位で packed に
        # 変換され、適格外だった名前の宣言は f32 のまま残るので、形も型も合ったまま値だけが
        # 壊れる。現行の convert 経路（torch.export の placeholder → FQN は単射・`_emit_const`
        # は digest 一意）では到達しないが、その 1:1 は上流の実装挙動 1 点に乗っているだけで
        # どこにも書かれていないので、ここを唯一の門にする。
        names_by_key: dict[str, list[str]] = {}
        for name in sorted(bakeable):
            names_by_key.setdefault(graph.initializers[name].tensor, []).append(name)
        collided = {key: names for key, names in sorted(names_by_key.items()) if len(names) > 1}
        raise EmitError(
            f"initializer 名とテンソルキーが 1:1 でない: {collided}"
            "（宣言は名前単位・実体はキー単位なので、同じキーを指す名前の一部だけが圧縮格納に"
            "なると宣言と実体がずれる）"
        )
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
    # 適格判定の除数は出荷の実 g（= `weight_scales` の形）から引く — `_plan_i4` が
    # `group_size_of` で宣言する値と同じ源にしないと、g16 資産の一部が黙って f32 で残る。
    i4_eligible = (
        i4_eligible_initializers(graph, scales=weight_scales) if "i4" in requested else set()
    )
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
        # i4 の適格は **{@link I4_WEIGHT_OPS} の重みスロット限定**（ADR 0069 決定 5）— 既定 i4
        # では適格外（conv2d / conv_transpose1d / groups > 1 の conv1d / 格納行長が group 長で
        # 割り切れない重み）を f32 のまま静かに残す。i8 の適格外が f32 で残るのと同じ設計で、
        # ランタイム側（executor.ts の eligible ∩ i4Eligible）と対。ここで除外しないと、
        # conv を混ぜたグラフが「conv の重みを i4 にできない」で export 不能になる
        # （Codex 波 F 指摘 I4-ELIG-01）。明示指定の i4 × 適格外は `_plan_i4` 自身の門が
        # fail loudly で受ける。
        if dtype == "i4" and key not in weight_dtype_overrides and name not in i4_eligible:
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
                    graph, reserved, name, tensor, weight_scales, i4_eligible
                )
            plan.scales[scale_key] = scale
            reserved.add(scale_key)
            conversion = _Conversion(dtype=dtype, name=name, scale=scale)
        plan.conversions[key] = conversion
        plan.declarations[name] = declaration
    if weight_dtype != "f32" and not plan.declarations:
        # ADR 0006 の「圧縮指定なのに適格 0MB を沈黙させない」をエクスポータ側でも張る。
        # 見るのは「**圧縮格納が 1 本も計画されなかったか**」— 既定 dtype と同じ格納 dtype が
        # 0 本でも、明示指定が適格な全件を別 dtype で覆っていれば意図は果たされている
        # （混成格納を 1 本単位で書き切った形をここで落とさない）。
        explicit = sorted({dtype for dtype in weight_dtype_overrides.values() if dtype != "f32"})
        # 原因は 1 つに断定しない（i4 は「展開経路を持つ消費先が無い」と「格納行長が group 長で
        # 割り切れない」の両方が同じ枝に落ちる）。
        if weight_dtype == "i4":
            cause = (
                "i4 の展開経路（linear / embedding / conv1d の groups==1）を持つ消費先が無いか、"
                "格納行長が group 長で割り切れない可能性がある"
            )
        else:
            cause = (
                f"融合 op の重みスロットを持たないグラフに {weight_dtype} を"
                "指定していないか確認する"
            )
        raise EmitError(
            f"圧縮格納が 1 本も計画されなかった（既定 {weight_dtype} / 明示指定"
            f" {explicit or 'なし'}）— {cause}"
        )
    return plan


def _convert_for_storage(key: str, tensor: torch.Tensor, conversion: _Conversion) -> torch.Tensor:
    """圧縮格納への変換と、実データを読む適格性検査（重い側 — 1 本ぶんだけ生かす）。

    ここで落ちても最終名には 1 バイトも残らない — 書き出しは一時 path（`.partial`）へ行い、
    据え替えは検証を通った回だけ（`karume.publish` のモジュール doc の 3 段）。
    """
    if conversion.fixed_payload is not None:
        return conversion.fixed_payload
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
_STORAGE_BITS = {"f32": 32, "f16": 16, "bf16": 16, "i8": 8, "i4": 4, "i2": 2, "i32": 32}
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
            elif dtype == "i2":
                scale_bytes += shape[0] * _SCALE_BYTES
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


# ---------------------------------------------------------------------------
# 容器へ渡す材料
# ---------------------------------------------------------------------------


class _StoredTensors(Mapping[str, Buffer]):
    """テンソルキー → **格納後の生バイト**の遅延写像（引かれた 1 本だけを変換する）。

    MUST: 同じキーからは**毎回同じバイト列**を返す（書き手は sha256 の走査と書き出しの走査で
    2 度引く — container-v1 §4.1）。変換は決定的なので成立するが、ここに乱択やキャッシュ無効化
    を入れた瞬間に「宣言した sha256 と実バイトが違う容器」が出る。

    MUST: 変換済みを持ち越さない（同時に生きる圧縮テンソルは 1 本 — 書き手側の
    `container._Payloads` が 1 本だけ掴む）。
    """

    def __init__(
        self,
        tensors: Mapping[str, torch.Tensor],
        conversions: Mapping[str, _Conversion],
    ) -> None:
        self._tensors = tensors
        self._conversions = conversions

    def __getitem__(self, key: str) -> Buffer:
        tensor = self._tensors[key]
        conversion = self._conversions.get(key)
        if conversion is not None:
            tensor = _convert_for_storage(key, tensor, conversion)
        elif tensor.is_meta:
            raise EmitError(f"テンソル '{key}': meta のまま格納しようとしている（実体が無い）")
        # memoryview 経由で渡す（`tobytes()` は 1 本ぶんの複製を作る — DiT の重みは
        # 1 テンソルで数百 MB あり、ピーク RAM をそのぶん押し上げる）。
        return memoryview(tensor.contiguous().numpy()).cast("B")

    def __iter__(self) -> Iterator[str]:
        return iter(self._tensors)

    def __len__(self) -> int:
        return len(self._tensors)


@dataclass(frozen=True)
class StoredModel:
    """容器へ渡す 3 点（モジュール doc の「容器へ渡す 3 点」）。"""

    #: 格納宣言を commit したグラフ（渡された `graph` は 1 バイトも変えない）。
    graph: IrGraph
    #: テンソルキー → 格納後の生バイト（遅延・1 本ずつ）。
    tensors: Mapping[str, Buffer]
    #: テンソルキー → 容器の格納宣言。
    bindings: Mapping[str, Encoding]


def stored_model(
    graph: IrGraph,
    tensors: Mapping[str, torch.Tensor],
    *,
    weight_dtype: str = "f32",
    weight_scales: Mapping[str, torch.Tensor] | None = None,
    weight_dtype_overrides: Mapping[str, str] | None = None,
    fixed_weights: Mapping[str, FixedQuantizedWeight] | None = None,
) -> StoredModel:
    """格納変換を計画し、容器へ渡す {@link StoredModel} を組む（1 バイトも書かない）。

    `weight_dtype` が `"f16"` / `"i8"` / `"i4"` のとき、**適格な重みスロットだけ**が圧縮格納に
    なる（宣言と実体が 1 経路で決まる — 別々に決めると「宣言 f16 / 実体 f32」の沈黙誤読が
    作れる）。i4 の適格は {@link I4_WEIGHT_OPS}（linear / embedding / conv1d）の重みスロット
    限定で、conv1d はさらに `groups == 1` と格納行長の整除が要る
    （{@link i4_eligible_initializers} — ADR 0069 決定 5 とその追補）。

    `fixed_weights` は固定 packed 値を再量子化せず保存する入口（ADR 0097）。対応する `tensors`
    は論理形だけの f32/meta とし、meta のキー集合と完全一致させる。linear / embedding の rank2
    重みに限り、I2 / I4 / I8 の混成を保持する。実 f32 値との二重指定と、自動量子化の
    dtype / scale / override 指定との混在は拒否する。

    `weight_dtype_overrides`（テンソルキー → 格納 dtype）は 1 本単位の明示指定で、既定
    `weight_dtype` に**優先**する（混成格納 — 意味と fail loudly の線引きは
    `_plan_weight_dtype` の docstring）。scale が要る dtype を混ぜるときは `weight_scales` に
    i8 / i4 の台帳を**合流して**渡す（キー空間が FQN で重ならないので 1 つの Mapping で足りる）。

    MUST: 計画段で圧縮テンソルを作らない（実データを読む検査は {@link _StoredTensors} が
    引かれたときに 1 本ずつ受け持つ）。全件を先に変換すると、呼び出し側が持つ f32 集合と同時に
    生きてピーク RAM が両者の和になる。

    MUST: commit は `dataclasses.replace` の**ビューの中だけ**に閉じ、渡された `graph` は
    1 バイトも変えない。呼び出し側の宣言を書き換えると、同じ graph を使い回す経路
    （別 dtype で書き直す / 内訳を数える）が前回の圧縮宣言を引きずる。
    """
    # 共有 initializer（ADR 0096 段 2）はこの容器に実体を持たないので、宣言 / 格納の
    # 完全一致からは外す（宣言の storage はそのまま焼く — 計画も変換も掛からない）。
    declared = {graph.initializers[name].tensor for name in bakeable_initializers(graph)}
    stored = set(tensors)
    if declared != stored:
        raise EmitError(
            "宣言テンソルと格納テンソルが一致しない: "
            f"欠落 {sorted(declared - stored)} / 余剰 {sorted(stored - declared)}"
        )
    fixed = {} if fixed_weights is None else fixed_weights
    meta_keys = {key for key, value in tensors.items() if value.is_meta}
    if meta_keys != set(fixed):
        raise EmitError("f32/meta の論理重みと fixed_weights のキー集合が一致しない")
    if fixed and (weight_dtype != "f32" or weight_scales or weight_dtype_overrides):
        raise EmitError("fixed_weights と自動量子化指定は同時に使えない")
    contiguous = {key: value.detach().contiguous() for key, value in tensors.items()}
    plan = (
        _plan_fixed_weights(graph, contiguous, fixed)
        if fixed
        else _plan_weight_dtype(
            graph, contiguous, weight_dtype, weight_scales or {}, weight_dtype_overrides or {}
        )
    )
    committed = replace(graph, initializers={**graph.initializers, **plan.declarations})
    source = {**contiguous, **plan.scales}
    return StoredModel(
        graph=committed,
        tensors=_StoredTensors(source, plan.conversions),
        bindings=container_bindings(committed),
    )
