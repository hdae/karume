"""**非正方**の rope 表フィクスチャ（#23 — 任意解像度・非正方形の配線）。

S 形の DiT（`--dit-graph dyn` — ADR 0034）はホストが rope の cos / sin 表を組む。ホストが
持つのは**軸別の素表**（系列ディレクトリの `rope_base.safetensors`）だけで、`[1,1,S,128]` の
表は「素表を `[t,h,w,t,h,w]` に並べる」だけで作る。正方の解像度ではこの並べ方の
**h ↔ w 取り違えが原理的に検出できない** — Anima の `rope_scale` は h と w が同値
（`[1.0, 4.0, 4.0]`）なので `cos_h` と `cos_w` がバイト単位で一致し、H'=W' なら表そのものが
同じ値になる（ADR 0034 の検出限界 1）。

**非正方では位置の取り違えが割れる**（H'≠W' だと `[h の位置 i]` と `[w の位置 i]` が別の行を
指す）。そこで上流の `model.rope` が出す表を 4 幾何ぶん焼き、TS 側（`ropeTables`）の再構成と
**Uint32 完全一致**で突き合わせる。焼くのは `anima.patch.dit_rope_tables` の出力そのもの
（式は写さない — 素表と同じ規律）。

出力（既定 `<repo>/outputs/series/anima-rope-nonsquare/`）:

    rope.safetensors   幾何ごとの `cos_<WxH>` / `sin_<WxH>`（各 `[1,1,S,head_dim]`）
    rope.json          幾何の索引（latent 寸法・トークン格子・S・素表の行数）

    uv run python -m anima.rope_tables

**重みは要らない**（`CosmosRotaryPosEmbed` はパラメータもバッファも持たない純計算）ので、
モデルは `meta` デバイス上に config から組む — 7.3GiB のロードを 1 バイトも行わない。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from _shared.paths import SERIES_ROOT

from .patch import dit_rope_tables
from .resolution import format_resolution

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"
DEFAULT_OUT = SERIES_ROOT / "anima-rope-nonsquare"
SPATIAL_COMPRESSION = 8

#: 焼く幾何（ピクセル・`(幅, 高さ)`）。**16:9 と 3:4 の縦横 4 パターン**（#23 のユーザー指示）。
#:
#: MUST: 縦横の対を必ず両方入れる。片方だけだと「h と w を入れ替えた実装」がもう一方の幾何の
#: 表と一致してしまい、対応表のキーが違うだけで数は合う — 対で焼いて初めて取り違えが割れる。
GEOMETRIES: tuple[tuple[int, int], ...] = (
    (1344, 768),
    (768, 1344),
    (1152, 896),
    (896, 1152),
)


def load_rope_model(repo: str) -> torch.nn.Module:
    """rope だけが要るので **meta デバイス**に config から組む（重みは読まない）。

    `CosmosRotaryPosEmbed` はパラメータもバッファも持たず、`forward` は入力テンソルの
    デバイス上に表を作る（実測: 上流 `transformer_cosmos.py`）。したがって本体が meta でも、
    CPU のプローブを渡せば表は CPU の実数で出る。
    """
    from diffusers import CosmosTransformer3DModel

    config = CosmosTransformer3DModel.load_config(repo, subfolder="transformer")
    with torch.device("meta"):
        return CosmosTransformer3DModel.from_config(config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if any(width == height for width, height in GEOMETRIES):
        raise SystemExit("GEOMETRIES に正方が混ざっている（正方では h↔w 取り違えを検出できない）")

    model = load_rope_model(args.repo)
    patch = tuple(int(size) for size in model.rope.patch_size)
    tensors: dict[str, torch.Tensor] = {}
    entries: list[dict[str, Any]] = []
    for width, height in GEOMETRIES:
        label = format_resolution(width, height)
        latent_height = height // SPATIAL_COMPRESSION
        latent_width = width // SPATIAL_COMPRESSION
        cos, sin = dit_rope_tables(model, latent_height, latent_width)
        tensors[f"cos_{label}"] = cos.contiguous()
        tensors[f"sin_{label}"] = sin.contiguous()
        entries.append(
            {
                "resolution": label,
                "width": width,
                "height": height,
                "latent_height": latent_height,
                "latent_width": latent_width,
                "token_rows": latent_height // patch[1],
                "token_cols": latent_width // patch[2],
                "tokens": int(cos.shape[2]),
                "head_dim": int(cos.shape[3]),
            }
        )
        print(
            f"[rope] {label}: latent {latent_height}×{latent_width}"
            f" / S={cos.shape[2]} / head_dim={cos.shape[3]}",
            flush=True,
        )

    meta: dict[str, Any] = {
        "repo": args.repo,
        "spatial_compression": SPATIAL_COMPRESSION,
        "patch_size": list(patch),
        # 素表の行数（= 上流 `seq = arange(max(max_size))` の長さ）。ホスト側の天井の正本。
        "rope_base_rows": max(int(size) for size in model.rope.max_size),
        "geometries": entries,
        "tensors": {name: list(value.shape) for name, value in tensors.items()},
    }

    from safetensors.torch import save_file

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out / "rope.safetensors"))
    (args.out / "rope.json").write_text(
        json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"fixture OK: {len(tensors)} tensors → {args.out}", flush=True)


if __name__ == "__main__":
    main()
