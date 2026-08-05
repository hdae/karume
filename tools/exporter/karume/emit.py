"""配布形（safetensors 1 ファイル + `__metadata__` へのグラフ JSON 埋め込み）の書き出し。

格納の分岐は **f16（ADR 0018）と i8（ADR 0019）**。分岐が増えても「格納形式 = ランタイムが
受理する形式」の対応が 1 箇所で決まるよう、書き出し経路はこの関数だけにする（ここが枝分かれ
すると export は緑のまま実行だけ落ちる）。

## 適格判定（ADR 0018 / 0019 — f16 と i8 で**同じ規則**）

`weight_dtype="f16"` / `"i8"` で書くとき、圧縮のまま格納するのは**適格な initializer だけ**:

1. その initializer の消費が `ops.WEIGHT_SLOTS` の重みスロット**だけ**である
   （ランタイム側 `packages/runtime/src/runtime/plan.ts` の
   `eligibleCompressedInitializers` と同じ規則。
   bias も norm 系 weight も混在消費も消費ゼロも適格外 — ADR 0006）。
2. **逆変換がビット一致**する（= fake-quant 済み）。f16 は `f32 → f16 → f32` の往復、
   i8 は `q8.to(f32) · scale`（`scale` は fake-quant が使った値を**そのまま**）。

2 を条件に入れるのは、丸めていない値を圧縮格納すると「golden は元値・実行は丸め値」に
なって量子化誤差と実装誤差が混ざるから。適格なのに 2 を満たさない initializer は
**fail loudly**（丸めの掛け忘れ・掛ける順序の誤り・畳み込み定数が重みスロットへ流れた、の
いずれかで、どれも黙って通すと E2E の tolerance が意味を失う）。

## i8 の companion scale（ADR 0019）

per-channel scale は safetensors の**素のテンソル**として同じファイルに入り、IR 側は
`storage.scale` でそのキーを明示宣言する。キーは `_scale_key`（`karume.scale.<重みキー>`）で
機械的に作り、**実テンソルとの衝突**を書き出し前に検査する（衝突すると「別の重みを scale
として読む」形になり、ロードは通って値だけが壊れる）。scale は fake-quant が使った値を
`quantize.fake_quant_int8` から受け取ってそのまま書く — ここで amax から引き直すと f32 の
丸めで 1ulp 動きうるので、golden を採ったときの重みとの対応が壊れる。

## safetensors の並び順（docs/limitations.md）

Karume のリーダはデータ節を「隙間なく・要素サイズに整列して」覆うことを要求する。
要素数が奇数の F16（バイト長 ≡ 2 mod 4）の**直後**に F32 / I32 を置くと絶対 offset が
4 の倍数から外れてロードできない。並べ替えはエクスポータの責務なので、書き出し順を
`_write_order` が明示的に決める（`safetensors.torch.save_file` は自前の順序で書くため
使わない — 順序を外部ライブラリの実装詳細に預けない）。**I8 は要素サイズ 1 で整列制約が
無いぶん任意長を作れる**ので、既存の F16 規則の**後ろ = 末尾**に置く。書いた直後の
`verify.assert_reader_layout` がリーダ規則を写して検査する。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from karume.ir import IR_METADATA_KEY, IrGraph, IrInitializer, IrStorage
from karume.ops import WEIGHT_CHANNEL_AXES, WEIGHT_SLOTS
from karume.quantize import quantize_to_int8

#: 書き出せる格納 dtype（重みスロット向け）。bf16 は実行経路が無い（ADR 0006）。
WEIGHT_DTYPES = ("f32", "f16", "i8")

#: torch dtype → safetensors dtype 名 / 要素バイト数。ここに無い dtype は fail loudly。
_SAFETENSORS_DTYPE: Mapping[torch.dtype, tuple[str, int]] = {
    torch.float32: ("F32", 4),
    torch.float16: ("F16", 2),
    torch.int32: ("I32", 4),
    torch.int8: ("I8", 1),
}

#: 書き出し順の第 1 キー（safetensors dtype → 群）。**4 バイト整列を必要とする群を先に**
#: 置き、奇数要素の F16、さらに後ろへ任意長を作れる I8 を寄せる（上の「並び順」節）。
#: F32 → I32 の順と群内の名前昇順は `safetensors.torch.save_file` が f32 のみのファイルに
#: 対して出す並びと一致するので、f16 / i8 を含まない資産のバイト列はこの writer に
#: 切り替えても変わらない。
_DTYPE_GROUP = {"F32": 0, "I32": 1, "F16": 2, "I8": 3}

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
    #: i8 の companion scale（重み本体とは別テンソル）。ランタイムの
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
    """
    initializers = set(graph.initializers)
    eligible: set[str] = set()
    disqualified: set[str] = set()
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


def _scale_key(tensor_key: str) -> str:
    """companion scale の safetensors キー（重みキーから機械的に作る）。"""
    return f"karume.scale.{tensor_key}"


def _is_f16_exact(tensor: torch.Tensor) -> bool:
    """f32 → f16 → f32 の往復がビット一致するか（= fake-quant 済みか）。"""
    return bool(torch.equal(tensor, tensor.to(torch.float16).to(torch.float32)))


def _unrounded(name: str, tensor_key: str, detail: str) -> EmitError:
    return EmitError(
        f"initializer '{name}' ({tensor_key}) は重みスロット適格なのに {detail} —"
        " fake-quant が未適用か、参照採取より後に掛かっている"
        "（ADR 0006: 丸めは参照・golden の採取より前 MUST）"
    )


def _store_f16(
    graph: IrGraph, out: dict[str, torch.Tensor], name: str, tensor: torch.Tensor
) -> None:
    initializer = graph.initializers[name]
    if not _is_f16_exact(tensor):
        raise _unrounded(name, initializer.tensor, "f16 で表現できない値を含む")
    out[initializer.tensor] = tensor.to(torch.float16)
    graph.initializers[name] = IrInitializer(
        tensor=initializer.tensor, storage=IrStorage(dtype="f16")
    )


def _store_i8(
    graph: IrGraph,
    out: dict[str, torch.Tensor],
    name: str,
    tensor: torch.Tensor,
    scales: Mapping[str, torch.Tensor],
    axes: Mapping[str, int],
) -> None:
    initializer = graph.initializers[name]
    key = initializer.tensor
    scale = scales.get(key)
    if scale is None:
        # 適格なのに scale が無い = fake-quant が届いていない重み（畳み込み定数が重み
        # スロットへ流れた / 対象モジュール型の取りこぼし）。f32 のまま落とすと VRAM 削減が
        # 黙って減るだけなので、i8 指定では受理しない。
        raise _unrounded(name, key, "per-channel scale が無い")
    axis = axes.get(name)
    expected = [dim if index == axis else 1 for index, dim in enumerate(tensor.shape)]
    if list(scale.shape) != expected:
        raise EmitError(
            f"initializer '{name}' ({key}): scale {list(scale.shape)} が重み"
            f" {list(tensor.shape)} の軸 {axis} の keepdim 形 {expected} でない"
        )
    quantized = quantize_to_int8(tensor, scale)
    # MUST: scale は fake-quant が使った値**そのまま**で逆変換する（再計算禁止 — ADR 0019）。
    # ここが一致しない = 格納値と golden を採ったときの重みが別物、という一点に集約される。
    if not torch.equal(quantized.to(torch.float32) * scale, tensor):
        raise _unrounded(name, key, "i8 × per-channel scale で逆変換してもビット一致しない")
    scale_key = _scale_key(key)
    if scale_key in out:
        raise EmitError(
            f"initializer '{name}': scale テンソルのキー '{scale_key}' が実テンソルと衝突する"
        )
    out[key] = quantized
    out[scale_key] = scale.detach().contiguous()
    graph.initializers[name] = IrInitializer(
        tensor=key, storage=IrStorage(dtype="i8", scale=scale_key)
    )


def _apply_weight_dtype(
    graph: IrGraph,
    tensors: dict[str, torch.Tensor],
    weight_dtype: str,
    weight_scales: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """適格な重みを圧縮格納へ落とし、graph の宣言をそれに合わせる（宣言と実体は 1 経路）。"""
    if weight_dtype not in WEIGHT_DTYPES:
        raise EmitError(
            f"格納 dtype '{weight_dtype}' は書き出せない（{' / '.join(WEIGHT_DTYPES)}）"
        )
    if weight_dtype == "f32":
        return tensors
    eligible = eligible_compressed_initializers(graph)
    axes = weight_channel_axes(graph) if weight_dtype == "i8" else {}
    out = dict(tensors)
    for name in sorted(eligible):
        tensor = tensors[graph.initializers[name].tensor]
        if tensor.dtype is not torch.float32:
            # i32 格納（記号依存定数 — ADR 0010）が重みスロットに来ることはないが、来たなら
            # 意味論ごと違う話なので黙って圧縮格納にしない。
            continue
        if weight_dtype == "f16":
            _store_f16(graph, out, name, tensor)
        else:
            _store_i8(graph, out, name, tensor, weight_scales, axes)
    if not any(init.storage.dtype == weight_dtype for init in graph.initializers.values()):
        # ADR 0006 の「圧縮指定なのに適格 0MB を沈黙させない」をエクスポータ側でも張る。
        raise EmitError(
            f"格納 {weight_dtype} を指定したが適格な重みスロットが 1 本も無い"
            f"（融合 op の重みを持たないグラフに {weight_dtype} を指定していないか確認する）"
        )
    return out


#: 格納 dtype → 要素バイト数（IR の宣言だけからバイト数を導出するための表）。
_STORAGE_BYTES = {"f32": 4, "f16": 2, "i8": 1, "i32": 4}
#: companion scale の要素バイト数（F32 固定 — ADR 0019）。
_SCALE_BYTES = 4


def storage_breakdown(graph: IrGraph) -> StorageBreakdown:
    """格納宣言から内訳を数える（実テンソルを持ち回らず、毎回グラフから導出する）。

    initializer の shape は記号を含まない静的形であることが IR v1 の規則（verify.py）なので、
    宣言 shape × 格納 dtype の要素サイズで実バイト数がそのまま出る。i8 の scale も同じく
    「重みの宣言 shape のチャネル軸長 × 4B」で導出する（実テンソルを見ない）。
    """
    compressed_tensors = compressed_bytes = plain_tensors = plain_bytes = scale_bytes = 0
    axes = weight_channel_axes(graph)
    for name, initializer in graph.initializers.items():
        shape = [int(dim) for dim in graph.values[name].shape]
        count = 1
        for dim in shape:
            count *= dim
        dtype = initializer.storage.dtype
        nbytes = count * _STORAGE_BYTES[dtype]
        if dtype in ("f16", "i8"):
            compressed_tensors += 1
            compressed_bytes += nbytes
            if dtype == "i8":
                scale_bytes += shape[axes[name]] * _SCALE_BYTES
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


def _write_order(tensors: Mapping[str, torch.Tensor]) -> list[str]:
    """書き出し順（データ節に並ぶ順）。

    第 1 キーは dtype 群（F32 → I32 → F16 → **I8**）、F16 のうち**要素数が奇数のもの**
    （バイト長 ≡ 2 mod 4）はさらに後ろへ寄せる。第 2 キーは名前昇順。

    これで「4 バイト整列を要求するテンソルの前に、4 の倍数でないバイト長のテンソルが来る」
    ことが構造的に起こらない — 奇数 F16 より前は全て 4 の倍数長なので累積 offset は 4 の
    倍数を保ち、奇数 F16 どうしは 2 バイト整列だけを要求するので偶数 offset で足りる。
    I8 は要素サイズ 1 で整列制約が無いかわりに**任意のバイト長**を作るので、群の末尾に置く
    （F16 より前に来ると 2 バイト整列すら壊す）。
    """

    def key(name: str) -> tuple[int, int, str]:
        tensor = tensors[name]
        dtype_name, item_bytes = _dtype_of(tensor, name)
        # 「後ろへ寄せる」の対象は F16 だけ（I8 は既に最後の群で、群内の順序は整列に
        # 影響しない — 名前昇順のまま安定させる）。
        odd = 1 if dtype_name == "F16" and (tensor.numel() * item_bytes) % 4 != 0 else 0
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


def _save_ordered(
    path: Path,
    tensors: Mapping[str, torch.Tensor],
    order: Sequence[str],
    metadata: Mapping[str, str],
) -> None:
    """safetensors を**指定順で**書く（`save_file` は自前の順序で書くので使わない）。

    レイアウトは仕様どおり `[u64 LE ヘッダ長][ヘッダ JSON][データ節]`。ヘッダ JSON は
    `__metadata__` を先頭に、テンソルはデータ節に並ぶ順で載せる（HF のリーダはヘッダの
    宣言順にオフセットの連続性を見る）。
    """
    header: dict[str, object] = {"__metadata__": dict(metadata)}
    offset = 0
    for name in order:
        tensor = tensors[name]
        dtype_name, item_bytes = _dtype_of(tensor, name)
        nbytes = tensor.numel() * item_bytes
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
            array = tensors[name].numpy()
            # memoryview 経由で書く（`tobytes()` は 1 本ぶんの複製を作る — DiT の重みは
            # 1 テンソルで数百 MB あり、ピーク RAM をそのぶん押し上げる）。
            handle.write(memoryview(array).cast("B"))


def write_model(
    path: str | Path,
    graph: IrGraph,
    tensors: dict[str, torch.Tensor],
    *,
    weight_dtype: str = "f32",
    weight_scales: Mapping[str, torch.Tensor] | None = None,
) -> Path:
    """グラフと格納テンソルを 1 ファイルに書き、書いたパスを返す。

    `weight_dtype` が `"f16"` / `"i8"` のとき、**適格な重みスロットだけ**を圧縮格納にして
    `graph` の格納宣言を書き換える（宣言と実体が 1 経路で決まる — 別々に決めると
    「宣言 f16 / 実体 f32」の沈黙誤読が作れる）。

    `weight_scales` は `quantize.fake_quant_int8` が返した **FQN → scale** の台帳
    （`"i8"` のときだけ要る）。キーは safetensors のテンソルキーと同じ空間で、
    `id(tensor)` 突合はしない（ADR 0006）。
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
    contiguous = _apply_weight_dtype(graph, contiguous, weight_dtype, weight_scales or {})
    _save_ordered(out, contiguous, _write_order(contiguous), {IR_METADATA_KEY: graph.to_json()})
    return out
