"""QAT の取得元と系列名。dist から torch を import しないための小さな正本。"""

from collections.abc import Mapping

CHECKPOINTS: Mapping[str, str] = {
    "e2b": "gemma-4-E2B-it-qat-mobile-transformers",
    "e4b": "gemma-4-E4B-it-qat-mobile-transformers",
}
MAX_CHUNK_LENGTH = 128
MAX_SELECTED_ROWS = 9


def checkpoint_name(model: str) -> str:
    if model not in CHECKPOINTS:
        raise ValueError(f"未対応 QAT model: {model}")
    return CHECKPOINTS[model]


def series_name(model: str) -> str:
    checkpoint_name(model)
    return f"gemma4-qat-{model}-product"
