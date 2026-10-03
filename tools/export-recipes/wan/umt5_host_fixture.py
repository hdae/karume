"""Wan2.1 の umT5 の相対位置のバケット表について、TS の生成器
（`packages/models/src/wan/umt5/relative-position.ts`）と突き合わせる fixture を書く
（ADR 0119 決定 3・段 10c）。

    uv run --group wan --inexact python -m wan.umt5_host_fixture

出力（既定 `packages/models/tests/fixtures/wan-umt5/`）:

    relative-position.safetensors  `bucket_by_distance [1023]`（距離 = キーの位置 − クエリの位置
                                   = −511〜511 のバケット）と代表の有効長の表
                                   `table.lNNN [L, L]`（I32）
    relative-position.json         出所・バケットの構成・有効長の範囲・グラフ入力名・代表の有効長・
                                   全有効長の digest

期待値は上流の式（`UMT5Attention._relative_position_bucket` を {@link wan.umt5_patch} 経由で
呼んだもの — 式を写さない）。構成（バケット数・max_distance）は pin した revision の
`text_encoder/config.json` から読む（config だけ — 重みは読まない。取得は
`uv run --group wan --inexact python -m wan.text_embeds --fetch`）。

digest は有効長 2〜512 の全部の表を縛る: 表ごとに i32 リトルエンディアン・行優先のバイト列の
SHA-256（32 B）を取り、有効長の昇順に連ねたバイト列の SHA-256（16 進）。表を全部連ねると約 179 MB に
なり、TS の WebCrypto は 1 回で渡したバイト列しかハッシュできないので、表ごとに畳む。

代表の有効長（{@link REPRESENTATIVE_LENGTHS}）は表を丸ごと持つ（digest が割れたときに要素で
名指しするため）。バケットの境界（距離 8・12・16・23・32・46・64・91 — 調査 §4.3）を跨ぐ長さと、
固定 4 プロンプトの長さの一部を選ぶ。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import nn

from _shared.paths import REPO_ROOT
from karume.convert import normalize_boundary_tensor
from wan import umt5_patch
from wan.sources import DEFAULT_MODEL, SOURCES, text_snapshot
from wan.umt5_tokenizer import MAX_LENGTH, MIN_TOKENS

DEFAULT_OUT = REPO_ROOT / "packages" / "models" / "tests" / "fixtures" / "wan-umt5"
TENSORS_FILE = "relative-position.safetensors"
META_FILE = "relative-position.json"

#: 表を丸ごと持つ有効長: 下限 2・距離 7 まで（全部が線形の区間）の 8・初めて対数の区間に入る 9
#: （距離 8）・境界 16 を跨ぐ 17・固定プロンプトの 28 / 50 / 126・上限のバケット 15 に初めて届く 92
#: （距離 91）。
REPRESENTATIVE_LENGTHS: tuple[int, ...] = (2, 8, 9, 17, 28, 50, 92, 126)

#: 距離ごとのバケットのテンソル名と、その先頭の距離（= −(上限 − 1)）。
DISTANCE_KEY = "bucket_by_distance"
DISTANCE_OFFSET = -(MAX_LENGTH - 1)


def table_key(length: int) -> str:
    """代表の表のテンソル名（`table.l028` — 桁を揃えて並びを長さ順にする）。"""
    return f"table.l{length:03d}"


def table_bytes(table: torch.Tensor) -> bytes:
    """表の i32 リトルエンディアン・行優先のバイト列（TS の `Int32Array` の中身と同じ並び）。"""
    values = normalize_boundary_tensor(table, "バケット表").contiguous()
    return values.numpy().astype("<i4", copy=False).tobytes()


def tables_digest(attention: nn.Module) -> str:
    """有効長 2〜512 の全表の digest（モジュールの docstring の形）。"""
    chained = hashlib.sha256()
    for length in range(MIN_TOKENS, MAX_LENGTH + 1):
        table = umt5_patch.relative_position_buckets(length, attention)
        chained.update(hashlib.sha256(table_bytes(table)).digest())
    return chained.hexdigest()


def bucket_by_distance(attention: nn.Module) -> torch.Tensor:
    """距離 −511〜511 のバケット（上流の式に距離の列をそのまま渡したもの）。"""
    distances = torch.arange(DISTANCE_OFFSET, MAX_LENGTH, dtype=torch.long)
    return attention._relative_position_bucket(distances)


def check_consistent(attention: nn.Module, by_distance: torch.Tensor) -> None:
    """上限の表が距離の列の Toeplitz 展開（`table[i][j] = by_distance[j − i]`）であること。

    MUST: 書く前に見る — 距離の列と表が別の向きで焼かれると、TS 側は片方とだけ一致して
    もう片方で割れ、どちらが正かを fixture から決められなくなる。
    """
    table = umt5_patch.relative_position_buckets(MAX_LENGTH, attention)
    positions = torch.arange(MAX_LENGTH)
    expanded = by_distance[positions[None, :] - positions[:, None] - DISTANCE_OFFSET]
    if not torch.equal(table, expanded):
        raise AssertionError("上限の表が距離の列の Toeplitz 展開と違う（向きか添字の取り違え）")


def build_fixture(
    config: Any, source: Mapping[str, str]
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """fixture の中身（テンソルとメタ）を作る（ファイルは書かない — pytest が鮮度を見る口）。"""
    import transformers

    attention = umt5_patch.bucket_attention(config)
    by_distance = bucket_by_distance(attention)
    check_consistent(attention, by_distance)
    tensors = {DISTANCE_KEY: normalize_boundary_tensor(by_distance, DISTANCE_KEY)}
    for length in REPRESENTATIVE_LENGTHS:
        table = umt5_patch.relative_position_buckets(length, attention)
        tensors[table_key(length)] = normalize_boundary_tensor(table, table_key(length))
    meta = {
        "_doc": [
            "Wan2.1 の umT5 の相対位置のバケット表"
            "（packages/models/src/wan/umt5/relative-position.ts）の突き合わせ用 fixture。",
            "生成: tools/export-recipes/wan/umt5_host_fixture.py（期待値は上流の"
            " UMT5Attention._relative_position_bucket の出力 — 同ファイルの doc）。",
            "表の向きは上流の compute_bias と同じ table[i][j] = bucket(j − i)"
            "（キーの位置 − クエリの位置）。",
            "digest は有効長ごとの表（i32 LE・行優先）の SHA-256 を昇順に連ねたものの SHA-256。",
        ],
        "source": {
            **source,
            "config": "text_encoder/config.json",
            "transformers": transformers.__version__,
        },
        "num_buckets": int(config.relative_attention_num_buckets),
        "max_distance": int(config.relative_attention_max_distance),
        "bidirectional": not config.is_decoder,
        "min_length": MIN_TOKENS,
        "max_length": MAX_LENGTH,
        "input_names": list(umt5_patch.INPUT_NAMES),
        "distance_offset": DISTANCE_OFFSET,
        "representative_lengths": list(REPRESENTATIVE_LENGTHS),
        "digest": tables_digest(attention),
    }
    return tensors, meta


def pinned_config(model: str = DEFAULT_MODEL) -> Any:
    """pin した revision の `text_encoder/config.json`（config だけを読む）。"""
    from transformers import UMT5Config

    return UMT5Config.from_pretrained(text_snapshot(model) / "text_encoder")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(SOURCES))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    source = SOURCES[args.model]
    tensors, meta = build_fixture(
        pinned_config(args.model), {"repo": source.repo, "revision": source.revision}
    )
    args.out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out / TENSORS_FILE))
    # 2 字下げ（`deno fmt` の JSON の形 — 追跡対象の fixture なので書き直しで差分を出さない）。
    (args.out / META_FILE).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    representatives = len(REPRESENTATIVE_LENGTHS)
    print(f"fixture OK: {args.out}（digest {meta['digest'][:16]}…・代表 {representatives} 本）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
