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
（元ファイル名・sha256・構成値）を**キー 1 本**に畳んだ正準 JSON を載せる
（safetensors が単体で持ち出されても素性が追える）。両者は**この 1 パスで同じ源から**
書かれ、片方だけ更新される経路は無い。

MUST: `__metadata__` の鍵は 1 つだけ（{@link SOURCE_KEY}）。`safetensors` は `__metadata__` を
Rust 側の `HashMap` で持ち、**並びを保存も整列もしない** — 実測で、同じ入力から 3 通りの
並びが出た（鍵 3 本の頃）。鍵が 2 つ以上あるとヘッダのバイト列が実行ごとに動き、下の
「同じ入力からは同じバイト列」が成立しない（データ節は同一なのにファイルの sha256 だけが
動くので、「資産が変わった」と読める差分になる）。

## 書き出し経路

`safetensors.torch.save_file` で素直に書く。これは配布形ではなく**手置き資産の隣に置く入力**
（`inputs/irodori/dacvae-32dim/weights.safetensors`）で、配布形の器（`krm`）とは別の話である。
Karume のリーダはデータ節を「隙間なく・要素サイズに整列して」覆うことを要求するので
（docs/limitations.md）、書いた直後に `verify.assert_reader_layout` で同じ規則を通す。
なお本チェックポイントは全 F32 なので整列制約は自明に満たされるが、検査は無条件で通す。
テンソルの並びはキーの昇順に固定し、`__metadata__` は鍵 1 本・値もキー昇順の正準 JSON に
する（同じ入力からは**ファイル全体が**同じバイト列 — 再生成で差分が出ない）。門は
`dacvae/tests/test_convert.py::TestConvert::test_two_runs_write_the_same_bytes`。

書き先は**作業席**で、門を全部通ってから正規 path へ据える（ADR 0052 — 他の書き手と同じ
規律）。出力先が手置き資産と同居するディレクトリなので、据え替えは
{@link karume.artifacts.staged_publication} の**ファイル単位**を 2 回（`weights.safetensors` /
`metadata.json`）— ディレクトリ単位で据えると隣の `weights.pth` まで据え替え対象になる。

## 自己検証

実重みの変換は手動（一回性ユーティリティ）だが、資産に依存しない部分は
`dacvae/tests/test_convert.py` が縛る。実走側では変換の直後に**全テンソルを safetensors 側から
読み直してバイト一致**を確認し（`safetensors.safe_open` — 書いた実装とは別実装のリーダで
読む）、一致件数と出力の sha256 を要約に出す。ここが落ちたら作業席ごと捨てられ、正規 path は
1 バイトも動かない。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from _shared.paths import INPUTS_ROOT
from karume.artifacts import staged_publication
from karume.verify import assert_reader_layout

#: 既定の入力（手置きの実重み — `inputs/<family>/<name>/`）。
DEFAULT_CKPT = INPUTS_ROOT / "irodori" / "dacvae-32dim" / "weights.pth"

#: `.pth` の外側ラッパに期待する鍵。増減はモデル配布側の変更なので fail loudly。
WRAPPER_KEYS = ("state_dict", "metadata")

#: `__metadata__` の**唯一の**鍵（値は 1 本の正準 JSON 文字列 — モジュール doc の MUST）。
SOURCE_KEY = "karume_source"

#: その正準 JSON の中の鍵（safetensors の仕様上、`__metadata__` の値は文字列だけなので
#: 構造はここで持つ）。
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


def _source_document(ckpt: Path, source_sha256: str, metadata: object) -> str:
    """`__metadata__` の唯一の値（出所 3 欄をキー昇順の正準 JSON へ畳んだもの）。

    畳む前に {@link _metadata_json} の往復の門を通し、**その往復を生き延びた値**をそのまま
    埋める（門を通した値と載せる値が別物にならない）。`sort_keys` は決定性の片側で、もう
    片側（`__metadata__` の鍵を 1 本に保つ）はモジュール doc の MUST。
    """
    document = {
        SOURCE_FILE_KEY: ckpt.name,
        SOURCE_SHA256_KEY: source_sha256,
        SOURCE_METADATA_KEY: json.loads(_metadata_json(metadata)),
    }
    return json.dumps(document, ensure_ascii=False, sort_keys=True)


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
    source_text = _source_document(ckpt, source_sha256, metadata)

    # detach は保険（weights_only=True の読み込みは requires_grad を持たない）。contiguous は
    # writer が numpy 経由で生バイトを書く前提。どちらも値は変えない。
    tensors = {key: value.detach().contiguous() for key, value in state_dict.items()}
    # MUST: 門を通ってから据える（ADR 0052）。出力先は**手置き資産と同居する**
    # `inputs/<family>/<name>/` なので、ディレクトリ単位ではなく**ファイル単位**で 2 回据える
    # （ディレクトリごと据え替えると隣の `weights.pth` まで据え替え対象になる）。
    with (
        staged_publication(target) as staged_weights,
        staged_publication(metadata_path) as staged_metadata,
    ):
        save_file(
            {key: tensors[key] for key in sorted(tensors)},
            str(staged_weights),
            # MUST: 鍵は 1 本（モジュール doc）— `__metadata__` の並びは保存されない。
            metadata={SOURCE_KEY: source_text},
        )
        assert_reader_layout(staged_weights)
        matched = _assert_byte_identical(staged_weights, tensors)
        staged_metadata.write_text(
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
