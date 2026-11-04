"""QAT の取得元と系列名。dist から torch を import しないための小さな正本。"""

from collections.abc import Mapping
from dataclasses import dataclass

from gemma4.sources import REGULAR_TOKENIZER_PINS, HubPin, ModelPins

CHECKPOINTS: Mapping[str, str] = {
    "e2b": "gemma-4-E2B-it-qat-mobile-transformers",
    "e4b": "gemma-4-E4B-it-qat-mobile-transformers",
}


@dataclass(frozen=True)
class QatSource:
    """The pinned upstream commits one model was exported from: the mobile QAT repository for the
    weights and their configuration, the regular `-it` repository for the tokenizer."""

    #: The weights pin (`model.safetensors`, `config.json` — `generation_config.json` is read from
    #: the same snapshot) and the tokenizer pin (`tokenizer.json`). Each names the SHA-256 of the
    #: files `reference.json` fingerprints from that source (`gemma4.provenance.FILES_BY_ROLE`):
    #: the LFS oid the HF API reports for `model.safetensors` and `tokenizer.json`, and the digest
    #: of `config.json` as served raw at that commit. The export and the distribution check the
    #: series' record against them, so the card's "at commit" lines name the bytes actually
    #: converted.
    pins: ModelPins
    #: `license:` of the upstream `README.md` front matter at both pinned commits (one value: the
    #: card attributes both sources under one licence).
    license: str


#: Model → the pinned upstream sources.
#:
#: - weights: the revisions the local inputs were fetched at on 2026-09-11
#:   (`inputs/gemma4-qat/<checkpoint>/.cache/huggingface/download/*.metadata`), still `main` of
#:   both repositories on 2026-11-01 (HF API; last modified 2026-07-20; not gated).
#: - tokenizer: the regular `-it` repository at Google's glitch-token fixes of 2026-11-03
#:   (`gemma4.sources.REGULAR_TOKENIZER_PINS` — the q4_0 QAT line's pins, one spelling). The
#:   mobile repositories still ship the `tokenizer.json` from before those fixes (SHA-256
#:   `cc8d3a0c…`, the same in both), and the maintainer approved shipping the fixed tokenizer
#:   (2026-11-04).
#:
#: The licence review is `THIRD_PARTY_NOTICES.md`; moving a pin needs that review again.
SOURCES: Mapping[str, QatSource] = {
    "e2b": QatSource(
        pins=ModelPins(
            weights=HubPin(
                repo=f"google/{CHECKPOINTS['e2b']}",
                revision="dd693ff40353f057ca5f07e945ad867f4afbf2ec",
                sha256={
                    "model.safetensors": (
                        "efab429012b97ab986c4d4838a46ff3ad95d618b42ce514771ca40fadc76a9a4"
                    ),
                    "config.json": (
                        "cf6d7dc22738b5e6beb364bac833d78b869f5a6ffd57dfc96c6be3f2abc80424"
                    ),
                },
            ),
            tokenizer=REGULAR_TOKENIZER_PINS["e2b"],
        ),
        license="apache-2.0",
    ),
    "e4b": QatSource(
        pins=ModelPins(
            weights=HubPin(
                repo=f"google/{CHECKPOINTS['e4b']}",
                revision="8e3ac78f1a4a97c19f5f194ab362a2206f71d961",
                sha256={
                    "model.safetensors": (
                        "391946da2e0bec22288e9fe50a4d31d2401c9570ba3baaf0ba43d644dadeb1d4"
                    ),
                    "config.json": (
                        "c4847fd621d80f160945b62f4a2d9413fc82b81f17955b49ae32ea9feb00c5c6"
                    ),
                },
            ),
            tokenizer=REGULAR_TOKENIZER_PINS["e4b"],
        ),
        license="apache-2.0",
    ),
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
#: 照合した固定重みの本数・格納内訳・PLE の本数を配布側が現物と突き合わせる形にした。
#:
#: schema 3 は**PLE の数え方の繰り上げ**: sidecar の shard 本数（`pleShards`）から容器の資産の
#: block 本数（`pleBlocks`）へ変わった。block の切り方は旧 shard 境界を無視するので 2 つは
#: 別の数で、schema を据え置くと旧世代の記録が版の門を素通りして「欄が無い」でだけ落ちる
#: （何が食い違っているか綴られない）。移行は記録も schema 3 へ追随させる
#: （`migrate_series.rewrite_qat_reference`）。
REFERENCE_SCHEMA = 3


def checkpoint_name(model: str) -> str:
    if model not in CHECKPOINTS:
        raise ValueError(f"未対応 QAT model: {model}")
    return CHECKPOINTS[model]


def checkpoint_repo(model: str) -> str:
    """The upstream HF repository ID of `model` (`google/<checkpoint>`)."""
    return f"google/{checkpoint_name(model)}"


def series_name(model: str) -> str:
    checkpoint_name(model)
    return f"gemma4-qat-{model}-product"
