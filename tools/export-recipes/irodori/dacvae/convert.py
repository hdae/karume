"""DACVAE の PyTorch チェックポイント（`weights.pth`）を safetensors へ **1:1 変換**する台本。

    uv run python -m irodori.dacvae.convert

`inputs/<family>/<name>/` に手で置いた実重み（生成物ではない — docs/assets-layout.md）を、
IR export の入力素材として扱える形に**中身を変えずに**詰め替えるだけの一回性ユーティリティ。
既定の入出力は `inputs/irodori/dacvae-32dim/weights.pth` →
`inputs/irodori/dacvae-32dim/weights.safetensors`（+ `metadata.json`）。

## 1:1 の意味（この台本がしないこと）

**鍵名不変・dtype 不変・値バイト不変**。具体的に、以下は**一切しない**:

- `weight_norm` の焼き込み（このチェックポイントは旧式の `weight_g` / `weight_v` ペアで
  持っている）。`w = g · v/‖v‖` の合成は**モデル形に依存する export 段の責務**で、ここで
  焼くと「入力素材」が既に 1 つの解釈に固まってしまう。
- 鍵名の正規化・接頭辞の付け替え・Snake の `.alpha` などの畳み込み。
- dtype の変換（f16 / i8 格納は配布形の話 — ADR 0018 / 0019 で `karume.emit` が扱う）。

## metadata の扱い

`.pth` の外側は `{state_dict, metadata}` のラッパ dict で、`metadata` はテンソルを含まない
純粋な構成値（`kwargs`）。safetensors の `__metadata__` は**文字列しか持てない**ので、
構造つきの正本は同ディレクトリの `metadata.json` に書き、`__metadata__` には出所
（元ファイル名と sha256）と同じ内容のコンパクト JSON を 1 本の文字列として載せる
（safetensors が単体で持ち出されても素性が追える）。両者は**この 1 パスで同じ源から**
書かれ、片方だけ更新される経路は無い。

## 書き出し経路

`karume.emit` の writer をそのまま借りる。Karume のリーダはデータ節を「隙間なく・要素サイズに
整列して」覆うことを要求し（docs/limitations.md）、その並び順の実装を 2 本に増やさないため
（`emit.write_model` は IR グラフを `__metadata__` に埋める**配布形**専用なので、その下層の
`_write_order` / `_save_ordered` だけを使う。`tests/test_emit.py` も同じ層を直接叩いている）。
書いた直後に `verify.assert_reader_layout` でリーダ規則を写した検査を通す。
なお本チェックポイントは全 F32 なので整列制約は自明に満たされるが、検査は無条件で通す。

## 自己検証

この台本は資産依存の一回性ユーティリティなので pytest の門は持たない。かわりに変換の直後に
**全テンソルを safetensors 側から読み直してバイト一致**を確認し（`safetensors.safe_open` —
書いた実装とは別実装のリーダで読む）、一致件数と出力の sha256 を要約に出す。ここが落ちたら
出力は信用しない。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from safetensors import safe_open

from _shared.paths import INPUTS_ROOT
from karume.emit import _save_ordered, _write_order
from karume.verify import assert_reader_layout

#: 既定の入力（手置きの実重み — `inputs/<family>/<name>/`）。
DEFAULT_CKPT = INPUTS_ROOT / "irodori" / "dacvae-32dim" / "weights.pth"

#: `.pth` の外側ラッパに期待する鍵。増減はモデル配布側の変更なので fail loudly。
WRAPPER_KEYS = ("state_dict", "metadata")

#: `__metadata__` に載せる鍵（safetensors の仕様上、値は文字列だけ）。
SOURCE_FILE_KEY = "source_file"
SOURCE_SHA256_KEY = "source_sha256"
SOURCE_METADATA_KEY = "source_metadata"

_SHA_CHUNK = 1 << 22


class ConvertError(ValueError):
    """チェックポイントの形が前提と違う / 変換結果が元とバイト一致しない。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_SHA_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], object]:
    """`weights_only=True` で読み、外側の形と state_dict の中身を検査して返す。"""
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict):
        raise ConvertError(f"{path}: 最上位が dict でない（{type(obj).__name__}）")
    if set(obj) != set(WRAPPER_KEYS):
        raise ConvertError(
            f"{path}: 外側ラッパの鍵が期待と違う（期待 {list(WRAPPER_KEYS)} / 実際 {list(obj)}）"
        )
    state_dict = obj["state_dict"]
    if not isinstance(state_dict, dict):
        raise ConvertError(f"{path}: state_dict が dict でない（{type(state_dict).__name__}）")
    non_tensor = [key for key, value in state_dict.items() if not torch.is_tensor(value)]
    if non_tensor:
        raise ConvertError(f"{path}: state_dict にテンソルでない値がある: {sorted(non_tensor)}")
    return state_dict, obj["metadata"]


def _metadata_json(metadata: object) -> str:
    """非テンソル metadata を JSON 文字列にする（往復で同値にならなければ fail loudly）。

    `default=` の逃げは置かない — 文字列化で落ちる型が混ざったら、それは「JSON に入れられる
    構成値」という前提が崩れているということで、黙って `repr` を書くと素性を偽ることになる。
    """
    try:
        text = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    except TypeError as cause:
        raise ConvertError(f"metadata を JSON にできない: {cause}") from cause
    if json.loads(text) != metadata:
        raise ConvertError("metadata の JSON 往復が同値にならない（tuple / 非文字列鍵の混入）")
    return text


def _assert_byte_identical(path: Path, tensors: Mapping[str, torch.Tensor]) -> int:
    """書いた safetensors を**別実装のリーダ**で読み直し、全テンソルのバイト一致を見る。

    dtype・shape だけでなく生バイト列で突き合わせる（NaN のビット列や -0.0 まで含めて
    「値が変わっていない」を主張するため）。一致した本数を返す。
    """
    matched = 0
    with safe_open(str(path), framework="pt") as handle:
        keys = set(handle.keys())
        expected = set(tensors)
        if keys != expected:
            raise ConvertError(
                f"{path}: 鍵集合が一致しない（欠落 {sorted(expected - keys)} / "
                f"余剰 {sorted(keys - expected)}）"
            )
        for name in sorted(expected):
            source = tensors[name]
            restored = handle.get_tensor(name)
            if restored.dtype != source.dtype:
                raise ConvertError(
                    f"テンソル '{name}': dtype 不一致（元 {source.dtype} / 読み直し "
                    f"{restored.dtype}）"
                )
            if tuple(restored.shape) != tuple(source.shape):
                raise ConvertError(
                    f"テンソル '{name}': shape 不一致（元 {tuple(source.shape)} / 読み直し "
                    f"{tuple(restored.shape)}）"
                )
            if restored.numpy().tobytes() != source.numpy().tobytes():
                raise ConvertError(f"テンソル '{name}': バイト列が一致しない")
            matched += 1
    return matched


def convert(ckpt: Path, out: Path | None = None) -> dict[str, object]:
    """`ckpt` を safetensors へ 1:1 変換し、要約を返す（`metadata.json` も同ディレクトリへ）。"""
    if not ckpt.is_file():
        raise ConvertError(f"チェックポイントが見つからない: {ckpt}")
    target = out if out is not None else ckpt.with_suffix(".safetensors")
    metadata_path = target.with_name("metadata.json")

    source_sha256 = _sha256(ckpt)
    state_dict, metadata = _load_checkpoint(ckpt)
    metadata_text = _metadata_json(metadata)

    # detach は保険（weights_only=True の読み込みは requires_grad を持たない）。contiguous は
    # writer が numpy 経由で生バイトを書く前提。どちらも値は変えない。
    tensors = {key: value.detach().contiguous() for key, value in state_dict.items()}
    _save_ordered(
        target,
        tensors,
        _write_order(tensors),
        {
            SOURCE_FILE_KEY: ckpt.name,
            SOURCE_SHA256_KEY: source_sha256,
            SOURCE_METADATA_KEY: metadata_text,
        },
    )
    assert_reader_layout(target)
    matched = _assert_byte_identical(target, tensors)

    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "source": str(ckpt),
        "source_sha256": source_sha256,
        "out": str(target),
        "out_sha256": _sha256(target),
        "out_bytes": target.stat().st_size,
        "metadata_json": str(metadata_path),
        "tensors": len(tensors),
        "byte_identical": matched,
        "parameters": sum(tensor.numel() for tensor in tensors.values()),
        "dtypes": sorted({str(tensor.dtype) for tensor in tensors.values()}),
        "prefixes": {
            prefix: sum(1 for key in tensors if key.split(".")[0] == prefix)
            for prefix in sorted({key.split(".")[0] for key in tensors})
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="出力 safetensors（既定は --ckpt の拡張子違い）。metadata.json は同ディレクトリ。",
    )
    args = parser.parse_args(argv)
    print(json.dumps(convert(args.ckpt, args.out), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
