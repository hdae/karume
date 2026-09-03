"""TS 側の RoPE parity フィクスチャの**生成**と、上流実装の呼び出し口。

    cd tools/export-recipes && uv run --with 'transformers==5.14.1' \
        python -m gemma4.tests.rope_fixture

書き出す先は `packages/models/tests/fixtures/gemma4-rope-upstream.json`。中身は上流
`Gemma4TextRotaryEmbedding` の**実出力**（f32 のバイト列を base64）で、TS 側の実装が同じ
位置・同じ宣言でその値へ近づくことを見る門の相手になる。

MUST: 値は上流モジュールから採る（式を書き直さない）— こちら側の式で作ると「同じ式を 2 回
書いて比べただけ」の恒真になり、TS 実装との突合が上流との突合でなくなる。
MUST: E2B の宣言値（theta / head_dim / partial_rotary_factor）は**上流チェックポイントの
`config.json` と同じ数**を持つ（{@link E2B_TEXT_CONFIG_FIELDS}）。ここが実物とずれると、
フィクスチャは自己整合したまま配布形と無関係な表になる。門は `test_rope.py` の
`test_the_declaration_matches_the_upstream_checkpoint`（{@link E2B_CONFIG_PATH} が無い環境は
明示 SKIP）。
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from _shared.paths import INPUTS_ROOT, REPO_ROOT
from gemma4 import rope

#: TS 側の fixture の置き場（読み手は `packages/models/tests/` の parity 門）。
FIXTURE_PATH = (
    REPO_ROOT / "packages" / "models" / "tests" / "fixtures" / "gemma4-rope-upstream.json"
)

#: 上流チェックポイントの config（{@link E2B_TEXT_CONFIG_FIELDS} の出所）。綴りの正本は
#: `_shared.paths.INPUTS_ROOT` 側（`export.py` の `DEFAULT_MODEL_DIR` と同じ置き場）。
E2B_CONFIG_PATH = INPUTS_ROOT / "gemma4" / "gemma-4-E2B-it" / "config.json"

#: 突合に使う位置（両端・境界・累乗の前後・上限）。**連続でない点を混ぜる**のは、位置に
#: 比例して開く誤差の当て方が「先頭だけ合っている」形にならないようにするため。
FIXTURE_POSITIONS: tuple[int, ...] = (
    0,
    1,
    2,
    3,
    17,
    511,
    512,
    1023,
    1024,
    4095,
    16383,
    65535,
    131071,
)

#: E2B の text config のうち RoPE の式に効く欄（`inputs/gemma4/gemma-4-E2B-it/config.json` の
#: `text_config` と同じ数）。模型は組まない — rotary モジュール 1 つを作るためだけの config。
E2B_TEXT_CONFIG_FIELDS: Mapping[str, Any] = {
    "hidden_size": 1536,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "head_dim": 256,
    "global_head_dim": 512,
    "num_hidden_layers": 5,
    "layer_types": [
        rope.SLIDING_ATTENTION,
        rope.SLIDING_ATTENTION,
        rope.SLIDING_ATTENTION,
        rope.SLIDING_ATTENTION,
        rope.FULL_ATTENTION,
    ],
    "sliding_window": 512,
    "max_position_embeddings": 131072,
    "rope_parameters": {
        rope.SLIDING_ATTENTION: {"rope_type": "default", "rope_theta": 10000.0},
        rope.FULL_ATTENTION: {
            "rope_type": "proportional",
            "rope_theta": 1000000.0,
            "partial_rotary_factor": 0.25,
        },
    },
}


def e2b_text_config() -> Any:
    """E2B と同じ RoPE 宣言を持つ `Gemma4TextConfig`（重みは読まない）。"""
    import transformers

    return transformers.Gemma4TextConfig(**dict(E2B_TEXT_CONFIG_FIELDS))


def upstream_tables(
    config: Any, positions: Sequence[int]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """上流 `Gemma4TextRotaryEmbedding` の実出力（層種別 → `(cos, sin)` の f32 配列）。"""
    import torch
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    rotary = Gemma4TextRotaryEmbedding(config)
    probe = torch.zeros(1, 1, int(config.hidden_size), dtype=torch.float32)
    position_ids = torch.tensor([list(positions)], dtype=torch.int64)
    tables: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for layer_type in dict.fromkeys(config.layer_types):
        with torch.no_grad():
            cos, sin = rotary(probe, position_ids=position_ids, layer_type=layer_type)
        tables[layer_type] = (cos[0].numpy().copy(), sin[0].numpy().copy())
    return tables


def _encode(rows: np.ndarray) -> str:
    """f32 の行列 → base64（リトルエンディアンの生バイト列 — TS 側は `DataView` で読む）。"""
    return base64.b64encode(np.ascontiguousarray(rows, dtype="<f4").tobytes()).decode("ascii")


def build_fixture(positions: Sequence[int] = FIXTURE_POSITIONS) -> dict[str, Any]:
    """フィクスチャの中身（宣言 + 上流の実出力）。"""
    config = e2b_text_config()
    specs = rope.rope_specs(config)
    tables = upstream_tables(config, positions)
    return {
        "positions": list(positions),
        "spec": {layer_type: specs[layer_type].declaration() for layer_type in sorted(specs)},
        "tables": {
            layer_type: {"cos": _encode(cos), "sin": _encode(sin)}
            for layer_type, (cos, sin) in sorted(tables.items())
        },
    }


def fixture_json(positions: Sequence[int] = FIXTURE_POSITIONS) -> str:
    """書き出す本文。**インデント 2**（TS 側の置き場なので `deno fmt` の整形に合わせる）。"""
    return json.dumps(build_fixture(positions), indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(fixture_json(), encoding="utf-8")
    print(f"[rope] {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
