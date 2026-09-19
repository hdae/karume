"""QAT の取得元と系列名。dist から torch を import しないための小さな正本。"""

from collections.abc import Mapping

CHECKPOINTS: Mapping[str, str] = {
    "e2b": "gemma-4-E2B-it-qat-mobile-transformers",
    "e4b": "gemma-4-E4B-it-qat-mobile-transformers",
}

#: モデル → PLE の格納ビット数（一次情報は上流 `QuantizedEmbedding.num_bits`）。上流検査
#: （`checkpoint.py`）と配布の突合（`distribution.py`）は同じ期待値を別々に綴っていたので、
#: Python 側の綴りをここ 1 箇所にする（TS 側は `packages/models/src/gemma/qat.ts` が持つ —
#: 言語境界は越えられないので到達点は「各言語 1 箇所」）。
PLE_BITS: Mapping[str, int] = {"e2b": 4, "e4b": 2}

#: 記号 `M`（prefill 1 回の行数）の trace 時の上限。`trace.py` の `Dim` と、配布形が宣言する
#: `pipelineConfig.maxChunkLength` の出どころ。
MAX_CHUNK_LENGTH = 768
MAX_SELECTED_ROWS = 9

#: 既定の会話容量（`capacity`）と prefill 1 回の行数（`chunkLength`）。**通常 Gemma と同じ値**
#: （2026-09-19 裁定・ADR 0097 追記 7）— 初期値 128 / 32 は trace 上限の定数を会話容量に
#: 使い回したもので、モデルの位置上限（公式 config は 131,072）ではなく変換側の都合だった。
#: どちらも実行時ノブなので、利用者は `--capacity` / `--chunk-length` で上書きできる。
DEFAULT_CAPACITY = 4096
DEFAULT_CHUNK_LENGTH = 768

#: `reference.json` の schema 版。schema 2 で「常に True の bytes 一致フラグ」を落とし、
#: 照合した固定重みの本数・格納内訳・PLE shard 本数を配布側が現物と突き合わせる形にした。
REFERENCE_SCHEMA = 2


def checkpoint_name(model: str) -> str:
    if model not in CHECKPOINTS:
        raise ValueError(f"未対応 QAT model: {model}")
    return CHECKPOINTS[model]


def series_name(model: str) -> str:
    checkpoint_name(model)
    return f"gemma4-qat-{model}-product"
