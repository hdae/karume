"""**Generates** the RoPE parity fixture of the TS side, and calls the upstream implementation.

    cd tools/export-recipes && uv run --with 'transformers==5.14.1' \
        python -m gemma4.tests.rope_fixture

The output is `packages/models/tests/fixtures/gemma4-rope-upstream.json`. It holds the **actual
output** of upstream `Gemma4TextRotaryEmbedding` (the f32 bytes in base64), the counterpart of the
gate that checks the TS implementation approaches those values at the same positions with the same
declaration.

MUST: the values are taken from the upstream module (the formula is not rewritten) — building them
with this side's formula would be a tautology of "the same formula written twice and compared",
and comparing with the TS implementation would no longer be a comparison with upstream.
MUST: the E2B declaration values (theta / head_dim / partial_rotary_factor) hold **the same numbers
as `config.json` of the upstream checkpoint** ({@link E2B_TEXT_CONFIG_FIELDS}). If they drift from
the real one, the fixture stays self-consistent but becomes a table unrelated to the distribution.
The gate is `test_the_declaration_matches_the_upstream_checkpoint` in `test_rope.py` (an explicit
SKIP in an environment without {@link E2B_CONFIG_PATH}).
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from _shared.paths import INPUTS_ROOT, REPO_ROOT
from gemma4 import rope

#: Location of the TS side's fixture (the reader is the parity gate in `packages/models/tests/`).
FIXTURE_PATH = (
    REPO_ROOT / "packages" / "models" / "tests" / "fixtures" / "gemma4-rope-upstream.json"
)

#: Config of the upstream checkpoint (the source of {@link E2B_TEXT_CONFIG_FIELDS}). The source of
#: truth for the spelling is `_shared.paths.INPUTS_ROOT` (the same location as `DEFAULT_MODEL_DIR`
#: of `export.py`).
E2B_CONFIG_PATH = INPUTS_ROOT / "gemma4" / "gemma-4-E2B-it" / "config.json"

#: Positions used for the comparison (both ends, boundaries, around powers of two, the limit).
#: **Non-consecutive points are mixed in** so that the error that opens in proportion to the
#: position is not probed in a form where "only the start matches".
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

#: The fields of E2B's text config that affect the RoPE formula (the same numbers as `text_config`
#: of `inputs/gemma4/gemma-4-E2B-it/config.json`). No model is built — the config exists only to
#: create one rotary module.
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
    """A `Gemma4TextConfig` with the same RoPE declaration as E2B (no weights are read)."""
    import transformers

    return transformers.Gemma4TextConfig(**dict(E2B_TEXT_CONFIG_FIELDS))


def upstream_tables(
    config: Any, positions: Sequence[int]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The actual output of upstream `Gemma4TextRotaryEmbedding` (layer type → f32 arrays
    `(cos, sin)`)."""
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
    """f32 matrix → base64 (raw little-endian bytes — the TS side reads them with `DataView`)."""
    return base64.b64encode(np.ascontiguousarray(rows, dtype="<f4").tobytes()).decode("ascii")


def build_fixture(positions: Sequence[int] = FIXTURE_POSITIONS) -> dict[str, Any]:
    """Contents of the fixture (declaration + the upstream's actual output)."""
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
    """The text to write. **Indent 2** (it lives on the TS side, so it follows `deno fmt`'s
    formatting)."""
    return json.dumps(build_fixture(positions), indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(fixture_json(), encoding="utf-8")
    print(f"[rope] {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
