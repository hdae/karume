"""Wan の DiT の層逐次の CPU 参照（f64 / f32 — ADR 0121 決定 7・段 0）と、1.3B の既存 golden との
ビット一致の突き合わせ。

    uv run --group wan --inexact python -m wan.dit_reference compare --dtype i8 --max-tokens 192
    uv run --group wan --inexact python -m wan.dit_reference compare --dtype f16
    uv run --group wan --inexact python -m wan.dit_reference compare --dtype i8 --source module

## なぜ層逐次か

Wan2.1 の f64 の参照は f32 のモデルを読んで `model.double()` で全体を f64 にしていた（1.3B で約
10.4 GB）。5B では f64 の全量が 37.25 GiB になり、31 GiB 機に載らない（ADR 0121 容量の表）。そこで
DiT を **1 ブロックずつ** f64 / f32 にして回す（{@link LayerwiseDit}）。同時に持つのは、ブロックの外
（patch 埋め込み・条件の埋め込み・出力の射影 — 5B の f64 で 0.67 GiB）と 1 ブロック（同 1.22 GiB）
だけ。上流のブロックの列のループ変数は前のブロックを握ったまま次を求めるので、前のブロックの重みを
手放してから次を読む（{@link LayerwiseDit._opener}）。

## 書き下さずに上流の forward を回す

上流 `WanTransformer3DModel` を重みを持たない meta で組み、`blocks` を「回すたびに 1 ブロックずつ
重みを読んで実体化する列」（{@link _LazyBlocks}）に差し替える。上流の forward（RoPE の表・patch
埋め込み・条件の埋め込み・ブロックの列・出力の norm / 射影・unpatchify）は**上流のコードがそのまま
走る** — 書き写さないので、写し間違いを持ち込まない。RoPE の表はバッファだけの部品なので、上流の
初期化を CPU でもう 1 度回して作る（重みではない — `model.rope` の値は config だけから決まる）。

Wan2.2 の TI2V の分岐（トークンごとの timestep `[1,S]`）も上流の同じ forward が持ち、f32 はその
まま回る。f64 は回らない: f64 の監視が許す f64 でない値は batch 1 の時刻の sinusoid の形だけ
（`dit_patch._timestep_sinusoid_shapes`）で、`[1,S]` の timestep はその外の形を作る。f64 の対応は
ADR 0121 段 1（I2V 対応のパッチと参照ラッパ）で行い、それまでは入口で止める
（{@link LayerwiseDit.forward}）。

## 重みの読み口（{@link WeightSource} — 値はどれも f32）

- {@link ContainerDitWeights}: 系列の容器（`krm`）から 1 本ずつ。i8 は packed × 行ごとの scale を
  **f32 で**掛けた値（export の fake-quant と同じ値 — 逆量子化を f64 で行うと、既存の golden〈f32 の
  fake-quant 値を f64 へ広げた値〉とビットで割れる — ADR 0121 決定 7）。f16 は f16 → f32。
- {@link ModuleWeights}: 読み込み済み・丸め済みの上流モデル（`wan.export_dit` の 1.3B — f32 全量が
  載る機）。
- {@link CheckpointDitWeights}: 上流の checkpoint（F32）から 1 本ずつ読み、系列の丸め（f16 / i8 の
  fake-quant）を 1 本ずつ掛ける（容器がまだ無い 5B の下見 — `wan.dit_probe`）。丸めは 1 本の中で
  閉じる（f16 は要素ごと・i8 は行ごとの scale）ので、モデル全体に掛けた値とビット一致する。

f64 の参照は、どの読み口でも f32 の値を f64 へ広げる（`.to(float64)` は厳密）。

## f32 の参照と重みのアドレス（実測 2026-10-04 — torch 2.13.0+cpu・MKL 2024.2・AVX2）

この機の torch CPU の f32 の GEMM（MKL）は、**重みの先頭アドレスの 64 バイト境界からのずれ**で
最終ビットが変わる。値・形・stride が同じ重みでも、ずれだけを変えると 1.3B の S = 128 で出力が
最大 4.6e-6 動いた（`compare` の i8 系列 `accept-s00128-t0030`）。f64 の参照は同じケースでビット
一致した（f64 の重みはどちらの経路も新しい確保 — 64 バイト境界）。

上流の f32 の eager（`from_pretrained` で読んだモデル）の重みは、checkpoint の safetensors を
mmap した番地にある。ずれは `(8 + ヘッダ長 + data_offsets[0]) mod 64` で決まる（1.3B の 825 本
全部で実測と一致）。そこで f32 の参照は、`alignment`（キー → ずれ —
{@link upstream_alignment}）を渡されたとき、重みをそのずれの番地へ置いてから回す
（{@link _placed}）。これで容器から読んだ重みでも「上流の f32 の eager」とビット一致する
（ADR 0121 決定 7 の「層逐次の f32 が上流の f32 eager とビット一致」）。
NOTE: これは MKL の実装の振る舞いで、仕様の保証ではない（MKL の CNR〈`MKL_CBWR`〉を有効に
すればずれに依らなくなるが、既存の f32 の golden とビットで割れる）。

## f64 の経路で f64 以外の浮動小数を作らない

f64 の forward は `dit_patch` の f64 の参照と同じ規則で縛る: 上流の `.float()` の寄せを f64 では
素通しにし、f64 でない浮動小数の値が作られたら止める（例外は時刻の sinusoid の形だけ —
`dit_patch.float64_forward`）。ただし重みの読み込み（f32 での逆量子化を含む）は f64 の経路の外で、
その間だけ監視を止める（{@link _PausableWatch}）。

## 1.3B の既存 golden との突き合わせ（`compare` — 段 0 の緑の条件）

`wan.export_dit` のケースの表（{@link wan.export_dit.series_cases}）を層逐次で採り直す。
**別の置き場**（既定 {@link DEFAULT_OUT}）へ `reference.<case>.safetensors` を書いてから、
系列の既存の `reference.<case>` とテンソルごとにビットで比べる（ファイルの sha256 の一致も
記録する）。既存の golden・容器には書かない（読むだけ）。済んだケースは結果の行
（`results.jsonl`）を見て飛ばすので、途中で止めても同じ `--out` で続きから回る（{@link _done} —
行は実行の形〈全量 / f64 だけ〉と比較の版〈{@link COMPARE_VERSION}〉を持ち、合わない行は済みに
数えない）。終わりに要約を `summary.json` に書く（合否は `differ` が空か）。行と要約は f32 の
ビット一致が依る実行環境（torch のスレッド数・CPU の命令セット・MKL / OpenMP の環境変数 —
{@link numeric_environment}）も持つ。

MUST: diffusers は関数の中で import する（`wan` グループは既定の sync に入らない —
`tests/test_optional_group_imports.py`）。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import struct
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, Protocol

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from _shared.paths import BENCH_ROOT
from karume.container import codec_entry
from karume.quantize import QUANT_MODULE_TYPES, channel_scale, iter_quant_targets, quantize_to_int8
from wan import dit_patch
from wan.pipeline_ref import assert_no_mps
from wan.umt5_export import Checkpoint
from wan.umt5_reference import ContainerWeights

#: 突き合わせの既定の出力先（`outputs/bench/<配布名>/<日付>_<目的>/` — `docs/assets-layout.md`。
#: 既存の系列・golden とは別 — 走行中のジョブが系列を読む）。日付は起動した日なので、日をまたいで
#: 続きから回すときは前の置き場を `--out` で渡す。
DEFAULT_OUT = BENCH_ROOT / "karume-wan2.1" / f"{date.today().isoformat()}_dit-layerwise"

#: 突き合わせの結果の行（1 ケース 1 行の JSON — 再開の目印）と、終わりに書く要約。
RESULTS_FILE = "results.jsonl"
SUMMARY_FILE = "summary.json"

#: 突き合わせの版。比べる中身（キー・ケースの入力・書く golden の組み方）か行の欄を変えたら上げる —
#: 版の違う行は再開で「済み」に数えない（{@link _done}）。
COMPARE_VERSION = 2

#: 実行の形（全量 = f32 の output・block と f64 / f64 だけ = `--f64-only`）。
CompareMode = Literal["full", "f64-only"]

#: 実行の形 → 「済み」に数える行の形（全量の行は f64 だけの実行も覆う — 逆は覆わない）。
_COVERING_MODES: dict[str, frozenset[str]] = {
    "full": frozenset({"full"}),
    "f64-only": frozenset({"full", "f64-only"}),
}

#: 記録する環境変数の接頭辞（MKL の CNR〈`MKL_CBWR`〉・スレッド数・OpenMP の割り当て）。
_NUMERIC_ENV_PREFIXES = ("MKL_", "OMP_", "KMP_")

#: 容器が Linear 化した形（`[Cout, C·pt·ph·pw]`）で持つ、上流では Conv3d の重み。
PATCH_EMBEDDING_WEIGHT = "patch_embedding.weight"

#: diffusers の部品の checkpoint のファイル名（分割形の索引・単一形）。
DIFFUSERS_INDEX = "diffusion_pytorch_model.safetensors.index.json"
DIFFUSERS_SINGLE = "diffusion_pytorch_model.safetensors"

#: 上流の部品のディレクトリ（snapshot の下）。
TRANSFORMER_SUBFOLDER = "transformer"

#: 重みの丸め（系列の格納 dtype — `none` は量子化しない上流の F32 のまま）。
Rounding = Literal["f16", "i8", "none"]


class DitReferenceError(RuntimeError):
    """重み・形・dtype が層逐次の参照の前提から外れた。"""


# ---------------------------------------------------------------------------
# 重みの読み口
# ---------------------------------------------------------------------------


class WeightSource(Protocol):
    """上流の state dict のキー → f32 の値（patch 埋め込みは Linear 化した形でもよい）。"""

    def tensor(self, key: str) -> torch.Tensor: ...


class ModuleWeights:
    """読み込み済み・丸め済みの上流モデルの重み（参照を持つだけ — 複製しない）。"""

    def __init__(self, model: nn.Module) -> None:
        self._state = model.state_dict()

    def tensor(self, key: str) -> torch.Tensor:
        value = self._state.get(key)
        if value is None:
            raise DitReferenceError(f"モデルに '{key}' が無い")
        return value.detach()


class ContainerDitWeights(ContainerWeights):
    """系列の容器（`krm`）から重みを 1 本ずつ f32 で読む（f32 / f16 / 行ごとの i8）。

    f32 と i8 は `umt5_reference.ContainerWeights` の読み方そのもの（i8 は packed × scale を f32
    で — fake-quant と同じ値）。f16 は f16 → f32（厳密）。block は取るたびに sha256 を宣言と
    突き合わせる。
    MUST: ほかの格納は fail loudly（参照が黙って別の重みになる）。
    """

    def tensor(self, key: str) -> torch.Tensor:
        supply = self._supply(key)
        if codec_entry(supply.encoding.codec).layout != "f16":
            return super().tensor(key)
        raw = b"".join(self._payload(supply, index) for index in range(len(supply.blocks)))
        half = torch.frombuffer(bytearray(raw), dtype=torch.float16)
        return half.reshape(self._shape(key)).to(torch.float32)


class CheckpointDitWeights:
    """上流の checkpoint（F32）から 1 本ずつ読み、系列の丸めを 1 本ずつ掛ける。

    - `f16`: 全ての重みを f16 の表現可能値へ（`karume.quantize.round_weights_to_f16` と同じ要素
      ごとの丸め — 有限値が非有限へ飽和したら fail loudly）
    - `i8`: `quant_keys`（linear と patch 埋め込みの重み — {@link quant_keys}）だけを per-channel
      symmetric の RTN へ（`fake_quant_int8` と同じ `quantize(w, s)·s` を、出力チャネルを行にした
      2 次元で）。残り（bias・norm・`scale_shift_table`）は上流の f32 のまま
    - `none`: 上流の F32 のまま（量子化なしの参照）
    """

    def __init__(
        self, directory: Path, rounding: Rounding, quant_keys: frozenset[str] = frozenset()
    ) -> None:
        if rounding == "i8" and not quant_keys:
            raise DitReferenceError("i8 の丸めに量子化の対象（quant_keys）が渡されていない")
        self._checkpoint = Checkpoint(directory, index=DIFFUSERS_INDEX, single=DIFFUSERS_SINGLE)
        self._rounding = rounding
        self._quant_keys = quant_keys

    def tensor(self, key: str) -> torch.Tensor:
        value = self._checkpoint.read(key)
        if self._rounding == "f16":
            rounded = value.to(torch.float16).to(torch.float32)
            if bool((torch.isfinite(value) & ~torch.isfinite(rounded)).any()):
                raise DitReferenceError(f"'{key}': f16 への丸めで有限値が非有限へ飽和した")
            return rounded
        if self._rounding == "i8" and key in self._quant_keys:
            rows = value.reshape(value.shape[0], -1)
            scale = channel_scale(rows, 0)
            return (quantize_to_int8(rows, scale).to(torch.float32) * scale).reshape(value.shape)
        return value


def quant_keys(config: Mapping[str, Any]) -> frozenset[str]:
    """i8 系列の量子化の対象（S 形のラッパの linear の重み — export と同じ選び方 — の FQN）。

    ラッパの FQN は上流の state dict のキーと同じ空間（ラッパは上流の部品への参照を持つ）。
    patch 埋め込みはラッパでは Linear・上流では Conv3d で、キーは同じ `patch_embedding.weight`。
    """
    wrapper = dit_patch.WanDitTokens(meta_model(config))
    return frozenset(fqn for fqn, _, _ in iter_quant_targets(wrapper, op_types=QUANT_MODULE_TYPES))


def load_config(directory: Path) -> dict[str, Any]:
    """上流の DiT の config（`transformer/` のディレクトリ — 重みは読まない）。"""
    from diffusers import WanTransformer3DModel

    return dict(WanTransformer3DModel.load_config(directory))


#: 重みのアドレスのずれを測る境界（バイト — モジュール doc「f32 の参照と重みのアドレス」）。
ALIGNMENT_BYTES = 64


def upstream_alignment(directory: Path) -> dict[str, int]:
    """上流の checkpoint を `from_pretrained` で読んだときの、各重みの先頭アドレスの
    {@link ALIGNMENT_BYTES} 境界からのずれ（キー → バイト）。

    safetensors は先頭 8 バイトの長さ + JSON のヘッダの後ろにデータを置き、mmap の先頭はページ境界
    なので、ずれは `(8 + ヘッダ長 + data_offsets[0]) mod 64`。読むのはヘッダだけ。
    """
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise DitReferenceError(f"{directory} に safetensors が無い")
    alignment: dict[str, int] = {}
    for file in files:
        with file.open("rb") as stream:
            (length,) = struct.unpack("<Q", stream.read(8))
            header = json.loads(stream.read(length))
        header.pop("__metadata__", None)
        for key, entry in header.items():
            alignment[key] = (8 + length + int(entry["data_offsets"][0])) % ALIGNMENT_BYTES
    return alignment


def _placed(value: torch.Tensor, offset: int) -> torch.Tensor:
    """`value` の写しを、先頭アドレスが {@link ALIGNMENT_BYTES} 境界から `offset` バイトずれた番地に
    置いて返す（連続・同じ形 — 値は変えない）。"""
    size = value.element_size()
    if offset % size or not 0 <= offset < ALIGNMENT_BYTES:
        raise DitReferenceError(f"ずれ {offset} バイトに {value.dtype} を置けない")
    buffer = torch.empty(value.numel() + ALIGNMENT_BYTES // size, dtype=value.dtype)
    shift = ((offset - buffer.data_ptr() % ALIGNMENT_BYTES) % ALIGNMENT_BYTES) // size
    placed = buffer[shift : shift + value.numel()].view(value.shape)
    placed.copy_(value)
    if placed.data_ptr() % ALIGNMENT_BYTES != offset:
        raise DitReferenceError(f"ずれ {offset} バイトの番地に置けなかった")
    return placed


# ---------------------------------------------------------------------------
# 層逐次の forward
# ---------------------------------------------------------------------------


def meta_model(config: Mapping[str, Any]) -> nn.Module:
    """config だけから組んだ上流の `WanTransformer3DModel`（重みもバッファも meta・eval）。"""
    from diffusers import WanTransformer3DModel

    with torch.device("meta"):
        return WanTransformer3DModel.from_config(dict(config)).eval()


class _PausableWatch(dit_patch._NarrowFloatWatch):
    """f64 の経路の監視（`dit_patch` と同じ）に、重みの読み込みの間だけ止める口を足したもの。"""

    def __init__(self, allowed: frozenset[tuple[int, ...]]) -> None:
        super().__init__(allowed)
        self._paused = False

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if self._paused:
            return func(*args, **(kwargs or {}))
        return super().__torch_dispatch__(func, types, args, kwargs)

    @contextmanager
    def paused(self) -> Iterator[None]:
        self._paused = True
        try:
            yield
        finally:
            self._paused = False


class _LazyBlocks(nn.Module):
    """上流の `for block in self.blocks:` が回すたびに、1 ブロックずつ実体化して渡す列。

    子モジュールを登録しない（`parameters()` が meta の雛形を拾わない — 外側の dtype の検査が
    ブロックの外だけを見る）。実体化は {@link LayerwiseDit} が `opener` で渡す。
    """

    def __init__(self, count: int) -> None:
        super().__init__()
        self.count = count
        self.opener: Callable[[int], nn.Module] | None = None

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[nn.Module]:
        opener = self.opener
        if opener is None:
            raise DitReferenceError("層逐次の列が重みの読み口なしで回された")
        for index in range(self.count):
            yield opener(index)


@dataclass(frozen=True)
class DitReference:
    """層逐次の forward 1 回の結果（出力 `[1,C,F,H,W]` は回した dtype のまま）。"""

    output: torch.Tensor
    #: 各ブロックの出力 `[1,S,dim]`（`collect_blocks` のときだけ — 回した dtype のまま）。
    blocks: list[torch.Tensor]
    seconds: float


class LayerwiseDit:
    """上流 `WanTransformer3DModel` を 1 ブロックずつ重みを読んで回す CPU 参照（モジュール doc）。

    `alignment`（キー → ずれ — {@link upstream_alignment}）を渡すと、f32 の forward は重みを上流の
    `from_pretrained` と同じずれの番地へ置いてから回す（モジュール doc「f32 の参照と重みの
    アドレス」）。
    渡さないと読み口の値をそのまま使う（`ModuleWeights` は上流のモデルの番地そのもの）。
    """

    def __init__(
        self, config: Mapping[str, Any], *, alignment: Mapping[str, int] | None = None
    ) -> None:
        assert_no_mps()
        self._alignment = None if alignment is None else dict(alignment)
        from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed

        template = meta_model(config)
        self.config = template.config
        self._blocks = list(template.blocks)
        template.blocks = _LazyBlocks(len(self._blocks))
        # RoPE の表は config だけから決まるバッファ — 上流の初期化を CPU で回して作る（MPS の
        # 無い機で float64 から落とした f32 の表 — `from_pretrained` の `model.rope` と同じ値）。
        template.rope = WanRotaryPosEmbed(
            int(self.config.attention_head_dim),
            tuple(self.config.patch_size),
            int(self.config.rope_max_seq_len),
        )
        self._outer = template
        self.outer_shapes = {name: list(p.shape) for name, p in template.named_parameters()}
        self.block_shapes = {name: list(p.shape) for name, p in self._blocks[0].named_parameters()}
        stray = [name for name, _ in self._blocks[0].named_buffers()]
        stray += [name for name, _ in template.named_buffers() if not name.startswith("rope.")]
        if stray:
            raise DitReferenceError(f"重みでないバッファ {stray} がある（層逐次は読まない）")

    @property
    def patch_size(self) -> tuple[int, int, int]:
        patch_t, patch_h, patch_w = (int(size) for size in self.config.patch_size)
        return patch_t, patch_h, patch_w

    @property
    def layers(self) -> int:
        return len(self._blocks)

    def weight_keys(self) -> list[str]:
        """読む重みのキー全部（上流の state dict の綴り — ブロックの外 + 全ブロック）。"""
        inner = [
            f"blocks.{index}.{name}" for index in range(self.layers) for name in self.block_shapes
        ]
        return [*self.outer_shapes, *inner]

    def forward(
        self,
        source: WeightSource,
        dtype: torch.dtype,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        collect_blocks: bool = False,
        on_block: Callable[[int, float], None] | None = None,
    ) -> DitReference:
        """上流の素の forward を `dtype`（f32 / f64）で回す（入力の f32 の値を `dtype` へ広げる）。

        f32 は `dit_patch.reference_dit`、f64 は `dit_patch.reference_dit_f64` と同じ呼び方
        （attention は CPU の flash 経路に固定）で、違いは重みを 1 ブロックずつ読むことだけ。
        """
        if dtype not in (torch.float32, torch.float64):
            raise DitReferenceError(f"参照の dtype は f32 / f64 だけ（{dtype}）")
        if dtype == torch.float64 and timestep.ndim != 1:
            raise DitReferenceError(
                f"f64 の層逐次の参照はトークンごとの timestep（形 {list(timestep.shape)}）にまだ"
                " 対応していない — ADR 0121 段 1（I2V 対応のパッチと参照ラッパ）で対応する"
                "（今は 1 次元の timestep だけ — モジュール doc）"
            )
        started = time.perf_counter()
        collected: list[torch.Tensor] = []
        with torch.no_grad():
            model = copy.deepcopy(self._outer)
            state = {
                key: self._read(source, key, shape, dtype)
                for key, shape in self.outer_shapes.items()
            }
            model.load_state_dict(state, strict=True, assign=True)
            del state
            model.rope = model.rope.to(dtype)
            narrow = sorted(
                {str(tensor.dtype) for tensor in (*model.parameters(), *model.buffers())}
                - {str(dtype)}
            )
            if narrow:
                raise DitReferenceError(f"ブロックの外に {narrow} の重み / バッファが残っている")
            sink = collected if collect_blocks else None
            if dtype == torch.float32:
                model.blocks.opener = self._opener(source, dtype, sink, on_block, None)
                output = dit_patch.reference_dit(model, latents, timestep, encoder_hidden_states)
            else:
                watch = _PausableWatch(
                    dit_patch._timestep_sinusoid_shapes(int(self.config.freq_dim))
                )
                model.blocks.opener = self._opener(source, dtype, sink, on_block, watch)
                with dit_patch._float_keeps_float64(), watch:
                    output = dit_patch.reference_dit(
                        model, latents.double(), timestep, encoder_hidden_states.double()
                    )
                if watch.found:
                    raise DitReferenceError(
                        f"f64 の参照の中で f64 でない値が作られた: {watch.found[:5]}"
                    )
        if collect_blocks and len(collected) != self.layers:
            raise DitReferenceError(f"ブロックの出力を {len(collected)} 本拾った（{self.layers}）")
        return DitReference(
            output=output,
            blocks=collected if collect_blocks else [],
            seconds=time.perf_counter() - started,
        )

    def _opener(
        self,
        source: WeightSource,
        dtype: torch.dtype,
        sink: list[torch.Tensor] | None,
        on_block: Callable[[int, float], None] | None,
        watch: _PausableWatch | None,
    ) -> Callable[[int], nn.Module]:
        """ブロック `index` を実体化する関数（重みの読み込みは f64 の監視の外）。

        ブロックの出力は `sink` があるときだけ写す（S = 32,760 で 30 本持つと f32 で約 6 GB）。
        `on_block` には読み込みを含むブロック 1 枚の所要を渡す（進捗の 1 行用）。

        MUST: 前のブロックの重みを手放してから次を読む。上流の `for block in self.blocks` の
        ループ変数は前のブロックを握ったまま次を求めるので、手放さないと 2 ブロックが同時に生きる
        （5B の f64 で 1.22 → 2.44 GiB）。手放すのは前のブロックの forward が済んだ後（次を
        求められた時点）なので、値は変わらない。
        """
        opened: list[nn.Module] = []

        def open_block(index: int) -> nn.Module:
            started = time.perf_counter()
            with watch.paused() if watch is not None else nullcontext():
                while opened:
                    opened.pop().to_empty(device="meta")
                block = copy.deepcopy(self._blocks[index])
                state = {
                    name: self._read(source, f"blocks.{index}.{name}", shape, dtype)
                    for name, shape in self.block_shapes.items()
                }
                block.load_state_dict(state, strict=True, assign=True)
                del state
            wrong = sorted({str(p.dtype) for p in block.parameters()} - {str(dtype)})
            if wrong:
                raise DitReferenceError(f"ブロック {index} に {wrong} の重みが残っている")

            def after(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
                if sink is not None:
                    sink.append(output.detach().clone())
                if on_block is not None:
                    on_block(index, time.perf_counter() - started)

            block.register_forward_hook(after)
            opened.append(block)
            return block

        return open_block

    def _read(
        self, source: WeightSource, key: str, shape: list[int], dtype: torch.dtype
    ) -> torch.Tensor:
        """読み口の値を上流の形・`dtype` で（読み口は f32 だけを受ける・patch 埋め込みだけは
        Linear 化の形を戻す）。f32 で `alignment` があれば上流のずれの番地へ置く。"""
        value = source.tensor(key)
        if value.dtype != torch.float32:
            raise DitReferenceError(f"'{key}' が {value.dtype}（読み口は f32 を返す）")
        if list(value.shape) != shape:
            if key != PATCH_EMBEDDING_WEIGHT or value.numel() != math.prod(shape):
                raise DitReferenceError(f"'{key}': 形 {list(value.shape)} が上流の {shape} と違う")
            value = value.reshape(shape)
        if dtype != torch.float32:
            return value.to(dtype)
        if self._alignment is None:
            return value
        offset = self._alignment.get(key)
        if offset is None:
            raise DitReferenceError(f"'{key}' の上流のずれが無い（alignment の表の外）")
        return _placed(value, offset)


# ---------------------------------------------------------------------------
# 1.3B の既存 golden との突き合わせ
# ---------------------------------------------------------------------------


def _tensor_bits(tensor: torch.Tensor) -> torch.Tensor:
    """ビット列としての比較用（-0.0 と 0.0・NaN の扱いを数値の等号に任せない）。"""
    return tensor.detach().contiguous().flatten().view(torch.uint8)


def compare_tensors(
    actual: Mapping[str, torch.Tensor], expected_path: Path, keys: Sequence[str] | None = None
) -> dict[str, Any]:
    """書いた golden（`actual`）と既存の `reference.<case>` をテンソルごとにビットで比べる。

    `keys` を渡すとそのキーだけを比べる（`--f64-only`）。既存に無い・形 / dtype が違う・ビットが
    違うキーを名指す。
    """
    compared = sorted(actual) if keys is None else sorted(keys)
    mismatched: dict[str, str] = {}
    with safe_open(str(expected_path), framework="pt") as handle:
        present = set(handle.keys())
        wanted = set(compared) if keys is not None else set(compared) | present
        for key in sorted(wanted):
            if key not in actual:
                mismatched[key] = "書いた側に無い"
                continue
            if key not in present:
                mismatched[key] = "既存に無い"
                continue
            mine, theirs = actual[key], handle.get_tensor(key)
            if mine.dtype != theirs.dtype or list(mine.shape) != list(theirs.shape):
                mismatched[key] = (
                    f"{mine.dtype} {list(mine.shape)} / 既存 {theirs.dtype} {list(theirs.shape)}"
                )
            elif not torch.equal(_tensor_bits(mine), _tensor_bits(theirs)):
                difference = (mine.double() - theirs.double()).abs().max()
                mismatched[key] = f"ビットが違う（最大絶対差 {float(difference):.3e}）"
    return {"keys": sorted(wanted), "mismatched": mismatched}


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def numeric_environment() -> dict[str, Any]:
    """f32 のビット一致が依る実行環境（モジュール doc「f32 の参照と重みのアドレス」— 仕様の保証で
    なく MKL の振る舞いなので、結果の行に残して後から読めるようにする）。"""
    return {
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "env": {
            name: value
            for name, value in sorted(os.environ.items())
            if name.startswith(_NUMERIC_ENV_PREFIXES)
        },
    }


def _done(results: Path, mode: CompareMode) -> dict[str, dict[str, Any]]:
    """結果の行（再開の目印）— この実行で「済み」に数えるケース名 → 行。

    数えるのは版が {@link COMPARE_VERSION} と同じで、形がこの実行を覆う行（{@link _COVERING_MODES}）
    だけ。書きかけで落ちた末尾の行（改行で終わらない — 行は 1 回の write で改行まで書く）は捨てて
    ファイルからも切り落とし、そのケースは回し直す。末尾でない行が読めないのは書きかけではないので
    fail loudly。
    """
    if not results.is_file():
        return {}
    lines = results.read_text(encoding="utf-8").split("\n")
    partial = lines.pop()
    if partial:
        print(
            f"[compare] {results}: 末尾の書きかけの行（{len(partial)} 文字）を捨てる"
            " — そのケースは回し直す",
            flush=True,
        )
        results.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    done: dict[str, dict[str, Any]] = {}
    stale = 0
    for number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise DitReferenceError(f"{results}:{number} の行が読めない（末尾ではない）") from error
        if (
            row.get("compare_version") == COMPARE_VERSION
            and row.get("mode") in (_COVERING_MODES[mode])
        ):
            done[row["case"]] = row
        else:
            stale += 1
    if stale:
        print(
            f"[compare] {results}: 比較の版（{COMPARE_VERSION}）か実行の形（{mode}）が合わない行"
            f" {stale} 本は済みに数えない",
            flush=True,
        )
    return done


def _append_row(results: Path, row: Mapping[str, Any]) -> None:
    """結果の行を 1 回の write で改行まで書いて、ディスクへ落とす（落ちても半端な行は末尾だけ）。"""
    with results.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def compare_series(
    dtype: str,
    *,
    source_kind: Literal["container", "module"] = "container",
    names: Sequence[str] = (),
    max_tokens: int | None = None,
    f64_only: bool = False,
    out: Path = DEFAULT_OUT,
) -> dict[str, Any]:
    """1.3B の系列（`dtype` = f16 / i8）の既存 golden を層逐次で採り直してビットで比べる。

    書くのは `out/<系列名>-<source_kind>/` の下だけ（既存の系列は読むだけ）。ケースは
    `wan.export_dit.series_cases(dtype)` の順で、`names` / `max_tokens` で絞れる。
    """
    from wan import export_dit
    from wan.sources import DEFAULT_MODEL, local_snapshot

    existing = export_dit.series_dir(dtype) / export_dit.TARGET
    target = out / f"{existing.parent.name}-{source_kind}"
    target.mkdir(parents=True, exist_ok=True)
    results = target / RESULTS_FILE
    mode: CompareMode = "f64-only" if f64_only else "full"
    done = _done(results, mode)
    environment = numeric_environment()
    upstream = local_snapshot(DEFAULT_MODEL) / TRANSFORMER_SUBFOLDER
    # 容器の重みは上流の mmap の番地に無い — f32 の参照を上流の eager と同じずれで回す（モジュール
    # doc）。module の読み口は上流のモデルの番地そのものなので置き直さない。
    alignment = upstream_alignment(upstream) if source_kind == "container" else None
    writer = LayerwiseDit(load_config(upstream), alignment=alignment)
    known = {spec.name(writer.patch_size) for spec in export_dit.series_cases(dtype)}
    if set(names) - known:
        raise DitReferenceError(f"知らないケース名がある: {sorted(set(names) - known)}")
    specs = [
        spec
        for spec in export_dit.series_cases(dtype)
        if (not names or spec.name(writer.patch_size) in names)
        and (max_tokens is None or _tokens(spec, writer.patch_size) <= max_tokens)
        # `--f64-only` は f64 の参照を持つケースだけ（f16 系列の小さい S は f32 の参照だけ）。
        and (not f64_only or export_dit.wants_float64(dtype, spec))
    ]
    source = _source(source_kind, dtype, existing)
    finished = sum(spec.name(writer.patch_size) in done for spec in specs)
    print(
        f"[compare] {existing} の {len(specs)} ケース（済み {finished}）"
        f" → {target}（読み口 {source_kind}・形 {mode}・環境 {json.dumps(environment)}）",
        flush=True,
    )
    for position, spec in enumerate(specs, start=1):
        name = spec.name(writer.patch_size)
        if name in done:
            print(f"[{position}/{len(specs)}] {name}: 済み（{done[name]['status']}）", flush=True)
            continue
        row = {
            **_compare_case(writer, source, dtype, spec, existing, target, f64_only=f64_only),
            "mode": mode,
            "compare_version": COMPARE_VERSION,
            "environment": environment,
        }
        _append_row(results, row)
        done[name] = row
        print(f"[{position}/{len(specs)}] {json.dumps(row, ensure_ascii=False)}", flush=True)
    chosen = [done[spec.name(writer.patch_size)] for spec in specs]
    environments = []
    for row in chosen:
        if row["environment"] not in environments:
            environments.append(row["environment"])
    summary = {
        "series": str(existing),
        "out": str(target),
        "mode": mode,
        "compare_version": COMPARE_VERSION,
        "cases": len(chosen),
        "equal": sum(row["status"] == "equal" for row in chosen),
        "differ": [row["case"] for row in chosen if row["status"] != "equal"],
        # 行を採った環境（再開をまたぐと複数になりうる — f32 のビットは環境に依る）。
        "environments": environments,
    }
    (target / SUMMARY_FILE).write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return summary


#: ブロックごとの進捗の行を出す S の下限（実寸のケース — 1 forward が分単位）。
PROGRESS_TOKENS = 4096


def _block_progress(name: str, label: str, layers: int) -> Callable[[int, float], None]:
    def report(index: int, seconds: float) -> None:
        print(f"[block] {name} {label} {index + 1}/{layers} {seconds:.1f} s", flush=True)

    return report


def _tokens(spec: Any, patch_size: tuple[int, int, int]) -> int:
    frames, height, width = spec.latent_shape
    return (frames // patch_size[0]) * (height // patch_size[1]) * (width // patch_size[2])


def _source(kind: str, dtype: str, existing: Path) -> WeightSource:
    """突き合わせの読み口（容器 = 5B と同じ経路・module = `export_dit` の 1.3B の経路）。"""
    from wan import export_dit

    if kind == "container":
        return ContainerDitWeights(existing / export_dit.MODEL_FILE)
    model = export_dit.load_transformer(export_dit.DEFAULT_MODEL)
    export_dit.fake_quant(dtype, model, dit_patch.WanDitTokens(model))
    return ModuleWeights(model)


def _compare_case(
    writer: LayerwiseDit,
    source: WeightSource,
    dtype: str,
    spec: Any,
    existing: Path,
    target: Path,
    *,
    f64_only: bool,
) -> dict[str, Any]:
    """1 ケースを採り直して書き、既存の golden と比べた結果の行。"""
    from wan import export_dit

    name = spec.name(writer.patch_size)
    latents, timestep, encoder_hidden_states = export_dit.case_inputs(writer, spec)
    long_case = _tokens(spec, writer.patch_size) >= PROGRESS_TOKENS
    seconds: dict[str, float] = {}
    reference32 = None
    if not f64_only:
        reference32 = writer.forward(
            source,
            torch.float32,
            latents,
            timestep,
            encoder_hidden_states,
            collect_blocks=spec.blocks,
            on_block=_block_progress(name, "f32", writer.layers) if long_case else None,
        )
        seconds["f32"] = round(reference32.seconds, 1)
    reference64 = None
    if export_dit.wants_float64(dtype, spec):
        reference64 = writer.forward(
            source,
            torch.float64,
            latents,
            timestep,
            encoder_hidden_states,
            on_block=_block_progress(name, "f64", writer.layers) if long_case else None,
        )
        seconds["f64"] = round(reference64.seconds, 1)
    if reference32 is None and reference64 is None:
        raise DitReferenceError(f"{name}: --f64-only なのに f64 の参照を持たないケース")
    tensors = export_dit.reference_tensors(
        name,
        latents=latents,
        timestep=timestep,
        output=None if reference32 is None else reference32.output,
        blocks=[] if reference32 is None else reference32.blocks,
        output_f64=None if reference64 is None else reference64.output,
    )
    filename = f"{export_dit.REFERENCE_PREFIX}{name}{export_dit.CASE_SUFFIX}"
    save_file(tensors, str(target / filename))
    keys = [export_dit.REFERENCE_F64_KEY] if f64_only else None
    comparison = compare_tensors(tensors, existing / filename, keys)
    return {
        "case": name,
        "status": "differ" if comparison["mismatched"] else "equal",
        "compared": len(comparison["keys"]),
        "mismatched": comparison["mismatched"],
        "file_sha256_equal": file_sha256(target / filename) == file_sha256(existing / filename),
        "seconds": seconds,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("command", choices=("compare",))
    parser.add_argument("--dtype", required=True, choices=("f16", "i8"), help="1.3B の系列")
    parser.add_argument(
        "--source",
        default="container",
        choices=("container", "module"),
        help="重みの読み口（container = 系列の容器・module = f32 のモデル + fake-quant）",
    )
    parser.add_argument("--case", action="append", default=[], help="ケース名で絞る（複数可）")
    parser.add_argument("--max-tokens", type=int, help="S がこれ以下のケースだけ")
    parser.add_argument(
        "--f64-only", action="store_true", help="f64 の参照（output.f64）だけを採って比べる"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    summary = compare_series(
        args.dtype,
        source_kind=args.source,
        names=args.case,
        max_tokens=args.max_tokens,
        f64_only=args.f64_only,
        out=args.out,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0 if not summary["differ"] else 1


if __name__ == "__main__":
    sys.exit(main())
