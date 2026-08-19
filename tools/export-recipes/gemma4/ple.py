"""Gemma 4 の PLE（Per-Layer Embeddings）— 1 枚表を層別 35 本へ割って持つ側の正本。

`embed_tokens_per_layer` は `[262144, 35×256]` の 1 枚表で、f32 実体は 9.4GB — **WebGPU の
1 バッファ上限を単独で超える**（i8 格納でも 2.19GiB）。そこで層別 35 本の `nn.Embedding`
（各 `[262144, 256]`・列スライス）へ割り、`stack` で `[1,T,35,256]` へ組み直して
`per_layer_inputs=` として上流へ渡す。

台本 3 本（1-shot の {@link gemma4.export}・chunk 系列の {@link gemma4.export_decode} /
{@link gemma4.export_token}）は**どれもこの 1 本を通す** — 分割の規則が 2 箇所に生えると、
片方だけ直したときの食い違いが golden の突合まで誰にも見えない。

## 行ブロック読み（{@link load_per_layer_tables}）

MUST: 1 枚表を f32 で丸ごと持たない。f32 の `[262144, 8960]` は 9.4GB あり、分割の複製と
同時に生きるとピーク RAM が倍になる（このモデル全体の f32 が 18.5GB なので、それだけで
載らなくなる）。行ブロック（{@link PLE_ROW_BLOCK}）で読み、そのつど 35 本へ配る。

## 35 本への配分（{@link per_layer_inputs}）

上流 `get_per_layer_inputs` は `[B,T,L*D]` の 1 枚 lookup を `reshape` するので、層 `i` は列
`[i*D, (i+1)*D)` を読む。分割表はその列スライスなので、`stack(dim=2)` が同じ並びになる。
スケール（{@link per_layer_scale} = `hidden_size_per_layer_input ** 0.5` = 16.0）は 2 冪なので
f32 の乗算が厳密 — 下のビット一致検査が成立する条件でもある。

## torch.equal のビット一致検査（{@link assert_per_layer_split}）

MUST: 分割が上流 `get_per_layer_inputs` と**ビット一致**することを export 前に 1 回検査する。
列の割り付けを間違えても形も型も dtype も合うので、`allclose` にすると「近いが別の表」を
通してしまい、やはり golden の突合まで誰にも見えない。検査の参照側は「チェックポイントの
数行だけを載せたモデル本体の `embed_tokens_per_layer`」で、9.4GB の f32 表をメモリに二重に
持たないための形（{@link PLE_PROBE_ROWS} 行・行の選び方は {@link probe_rows} の散点）。

NOTE: チェックポイントのキーの綴り（`model.language_model.` 接頭辞）はここが持たない —
キーの付け替えは台本側（{@link gemma4.export.renamed_state}）の知識なので、1 枚表のキーは
引数で受ける。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import nn

#: PLE 分割の等価検査に使う行数（行の選び方は {@link probe_rows} — ブロック境界・両端・
#: 中央の散点）。モデル本体の `embed_tokens_per_layer` にはこの行数だけを確保し
#: （`config.vocab_size_per_layer_input` を差し替える — この欄は上流でも
#: `Gemma4TextModel.__init__` と `resize_token_embeddings` からしか読まれない）、
#: 262144 行の f32 表がメモリに二重に載る形を作らない。分割 35 本は
#: {@link load_per_layer_tables} が safetensors から直接組む。
PLE_PROBE_ROWS = 8

#: 表を 1 枚読みせずに 35 本へ配るときの行ブロック（f32 で 8192×8960×4B = 293MB）。
PLE_ROW_BLOCK = 8192


def per_layer_scale(config: Any) -> float:
    """PLE lookup に掛かる embed_scale（上流 `Gemma4TextScaledWordEmbedding` と同値）。

    `hidden_size_per_layer_input` は 256 なので値は 16.0 — 2 冪なので f32 の乗算が厳密で、
    {@link assert_per_layer_split} のビット一致が成立する条件でもある。
    """
    return float(config.hidden_size_per_layer_input) ** 0.5


def per_layer_inputs(
    tables: Sequence[nn.Module], input_ids: torch.Tensor, scale: float
) -> torch.Tensor:
    """層別 35 本の lookup を `[B,T,L,D]` へ組む（上流 `get_per_layer_inputs` と同値）。

    上流は `[B,T,L*D]` の 1 枚 lookup を `reshape` するので、層 `i` は列
    `[i*D, (i+1)*D)` を読む。分割表はその列スライスなので、`stack(dim=2)` が同じ並びになる。
    """
    return torch.stack([table(input_ids) * scale for table in tables], dim=2)


def per_layer_table_shape(model_file: Path, key: str) -> tuple[int, int]:
    """PLE 1 枚表の `[行, 列]`（実体は読まない）。"""
    with safe_open(str(model_file), framework="pt") as handle:
        rows, packed = handle.get_slice(key).get_shape()
    return int(rows), int(packed)


def load_per_layer_tables(model_file: Path, key: str, layers: int, dim: int) -> nn.ModuleList:
    """PLE 1 枚表を層別 35 本の `nn.Embedding` へ割って読む。

    MUST: 1 枚表を f32 で丸ごと持たない（理由はモジュール docstring）。行ブロック
    （{@link PLE_ROW_BLOCK}）で読み、そのつど 35 本へ配る。
    """
    rows, packed = per_layer_table_shape(model_file, key)
    if packed != layers * dim:
        raise ValueError(
            f"PLE 表の列 {packed} が 層数 {layers} × 層当たり {dim} = {layers * dim} と違う"
        )
    tables = [torch.empty(rows, dim, dtype=torch.float32) for _ in range(layers)]
    with safe_open(str(model_file), framework="pt") as handle:
        packed_table = handle.get_slice(key)
        for start in range(0, rows, PLE_ROW_BLOCK):
            stop = min(start + PLE_ROW_BLOCK, rows)
            block = packed_table[start:stop, :].to(torch.float32)
            for index, table in enumerate(tables):
                table[start:stop] = block[:, index * dim : (index + 1) * dim]
    return nn.ModuleList([nn.Embedding.from_pretrained(table, freeze=True) for table in tables])


def probe_rows(rows: int) -> tuple[int, ...]:
    """PLE 分割の等価検査に使う {@link PLE_PROBE_ROWS} 本の**散点**行。

    連続 8 行だと 1 つの行ブロック（{@link PLE_ROW_BLOCK}）しか踏まず、別ブロックだけの
    取り違え・破損が門に映らない（Codex 波 H 指摘 H-05）。先頭・最初のブロック境界の両側・
    中央対・末尾ブロック先頭・末尾対を採り、足りない分は先頭から順に埋めて**常にちょうど**
    {@link PLE_PROBE_ROWS} 本にする（tiny 表でもブロック境界が無いだけで同数 — 検査席の
    行数 = 本数が config に焼かれるため可変にしない）。
    """
    if rows < PLE_PROBE_ROWS:
        raise ValueError(f"PLE 表の行数 {rows} が probe の {PLE_PROBE_ROWS} 本に足りない")
    candidates = [
        0,
        PLE_ROW_BLOCK - 1,
        PLE_ROW_BLOCK,
        rows // 2,
        rows // 2 + 1,
        rows - PLE_ROW_BLOCK,
        rows - 2,
        rows - 1,
    ]
    picked: list[int] = []
    for row in candidates:
        if 0 <= row < rows and row not in picked:
            picked.append(row)
    filler = 1
    while len(picked) < PLE_PROBE_ROWS:
        if filler not in picked:
            picked.append(filler)
        filler += 1
    return tuple(picked[:PLE_PROBE_ROWS])


def assert_per_layer_split(model: nn.Module, tables: nn.ModuleList, probe: Sequence[int]) -> None:
    """35 分割が上流 `get_per_layer_inputs` と**ビット一致**することを検査する。

    参照側はモデル本体の `embed_tokens_per_layer`（チェックポイントの probe 行 —
    {@link probe_rows} の散点 — だけをその並びで載せた席）で、上流のスケール
    （`hidden_size_per_layer_input ** 0.5`）と reshape を通った出力そのもの。分割側は
    **元の行番号**で 35 本から引いて {@link per_layer_inputs} で組み直したもの。

    MUST: `torch.equal`（ビット一致）で見る — 列の割り付けを間違えても形も型も dtype も
    合うので、`allclose` にすると「近いが別の表」を通す。
    """
    local = torch.arange(len(probe), dtype=torch.int64).unsqueeze(0)
    original = torch.tensor([list(probe)], dtype=torch.int64)
    with torch.no_grad():
        reference = model.model.get_per_layer_inputs(local, None)
        rebuilt = per_layer_inputs(tables, original, per_layer_scale(model.config))
    if tuple(reference.shape) != tuple(rebuilt.shape):
        raise AssertionError(
            f"PLE 分割の形 {tuple(rebuilt.shape)} が上流 {tuple(reference.shape)} と違う"
        )
    if not torch.equal(reference, rebuilt):
        worst = float((reference - rebuilt).abs().max())
        raise AssertionError(
            f"PLE の {len(tables)} 分割が上流 get_per_layer_inputs とビット一致しない"
            f"（最大絶対差 {worst}）"
            " — 列の割り付けかスケールが違う"
        )
