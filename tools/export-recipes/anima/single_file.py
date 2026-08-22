"""civitai の単一ファイル checkpoint を diffusers レイアウトへ組み直す（バリアントの入口）。

    uv run --group anima python -m anima.single_file \
        --checkpoint ../../inputs/anima/waiANIMA_v10Base10.safetensors \
        --out ../../outputs/anima-diffusers/anima-wai

`anima.export` の `--repo` は `from_pretrained(repo, subfolder=...)` にそのまま渡る文字列なので、
受けられるのは **diffusers レイアウトのリポ / ディレクトリ**だけ（単一ファイルを読む経路は
無い）。ここが橋渡しを引き受ける。

出来上がるディレクトリは「transformer と text_conditioner だけが差し替わった base のコピー」:

- **checkpoint が持つのは DiT（567 テンソル）と llm_adapter（118 テンソル）の 2 つだけ**で、
  text_encoder / VAE / tokenizer は入っていない（2026-08-22 実測）。したがってそれらは base から
  そのまま引く（**symlink** — 数 GB を種類ぶん複製しない）。
- 鍵名は ComfyUI 系の綴り（`net.` か `model.diffusion_model.` 前置）なので、diffusers の
  Cosmos 用変換表を通す。実測では base の transformer の鍵を**過不足なく**満たし、shape の
  食い違いも 0 だった。

MUST: 変換後の鍵集合を base の現物と突き合わせてから書く。表が上流で動いたときに黙って
「一部だけ移った重み」を書き出すと、export も配布も通って**絵だけが静かに壊れる**。
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping
from pathlib import Path

import torch
from diffusers.loaders.single_file_utils import (
    convert_cosmos_transformer_checkpoint_to_diffusers,
)
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

#: 上流の diffusers リポ（`anima/export.py` の `DEFAULT_REPO` と同じもの）。
BASE_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"

#: 変換表が期待する前置。civitai 側は `net.` か `model.diffusion_model.` のどちらかで包む。
CANONICAL_PREFIX = "net."
KNOWN_PREFIXES = ("model.diffusion_model.", "net.")

#: checkpoint 側で text_conditioner に当たる部分木。
CONDITIONER_PREFIX = "llm_adapter."

#: base から symlink で引く部分（checkpoint が持たないもの）。
SHARED_ENTRIES = ("text_encoder", "vae", "tokenizer", "t5_tokenizer", "scheduler")

#: 差し替える 2 つと、その重みファイル名（diffusers の既定）。
WEIGHT_FILENAME = "diffusion_pytorch_model.safetensors"

#: 変換の出所を残す記録（人が辿るためのもの — 配布の帰属は `anima/card.py` が持つ）。
SOURCE_PROVENANCE_FILE = "source_provenance.json"


def _strip_prefix(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """既知の前置を剥がして、変換表が期待する `net.` へ揃える。"""
    for prefix in KNOWN_PREFIXES:
        if all(name.startswith(prefix) for name in state):
            return {
                f"{CANONICAL_PREFIX}{name.removeprefix(prefix)}": value
                for name, value in state.items()
            }
    heads = sorted({name.split(".")[0] for name in state})
    raise SystemExit(f"知らない鍵の前置: {heads}（想定: {', '.join(KNOWN_PREFIXES)}）")


def _assert_covers(converted: Mapping[str, torch.Tensor], reference: Path, label: str) -> None:
    """変換後が base の現物を過不足なく満たすことを確かめる（鍵集合と shape）。"""
    expected = load_file(reference)
    missing = sorted(set(expected) - set(converted))
    extra = sorted(set(converted) - set(expected))
    if missing or extra:
        raise SystemExit(
            f"{label}: 変換後の鍵が base と一致しない — 足りない {len(missing)} 件"
            f"{missing[:3]} / 余る {len(extra)} 件{extra[:3]}"
        )
    mismatched = [
        name for name in expected if tuple(expected[name].shape) != tuple(converted[name].shape)
    ]
    if mismatched:
        raise SystemExit(f"{label}: shape が base と違う {len(mismatched)} 件{mismatched[:3]}")


def _link(target: Path, link: Path) -> None:
    """既存を消してから貼り直す（冪等 — 再実行で古い向き先が残らない）。

    実ディレクトリが居座っている場合も消す — symlink のつもりで作った場所に実体が入るのは、
    誰かが中身を写した後（`cp -r` は symlink を辿る）に起こる。`unlink` は
    `IsADirectoryError` で落ちるので、種類で分ける。
    """
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.is_dir():
        shutil.rmtree(link)
    link.symlink_to(target)


def convert(checkpoint: Path, out: Path, base_repo: str = BASE_REPO) -> None:
    base = Path(snapshot_download(base_repo))
    stripped = _strip_prefix(load_file(checkpoint))
    # MUST: text_conditioner は**変換表を通さない**、かつ**表を呼ぶ前に取り出す**。
    # ①Cosmos の表は llm_adapter の中まで diffusers 名（`transformer_blocks.0.attn1.*`）へ
    # 書き換えるが、base の text_conditioner は元の綴り（`blocks.0.cross_attn.*`）のままで、
    # 読み手（`AnimaTextConditioner`）もそちらを期待する。checkpoint 側の生の鍵は前置を
    # 剥がすだけで base と 1:1 に対応する（2026-08-22 実測）。②表は入力 dict を pop で
    # 消費するので、呼んだ後に元の鍵を引こうとすると空になる（同日に踏んだ）。
    conditioner_prefix = f"{CANONICAL_PREFIX}{CONDITIONER_PREFIX}"
    conditioner = {
        name.removeprefix(conditioner_prefix): value
        for name, value in stripped.items()
        if name.startswith(conditioner_prefix)
    }
    converted = convert_cosmos_transformer_checkpoint_to_diffusers(stripped)
    transformer = {
        name: value for name, value in converted.items() if not name.startswith(CONDITIONER_PREFIX)
    }
    _assert_covers(transformer, base / "transformer" / WEIGHT_FILENAME, "transformer")
    _assert_covers(conditioner, base / "text_conditioner" / WEIGHT_FILENAME, "text_conditioner")

    out.mkdir(parents=True, exist_ok=True)
    for name, state in (("transformer", transformer), ("text_conditioner", conditioner)):
        target = out / name
        target.mkdir(parents=True, exist_ok=True)
        # config は checkpoint に入っていない（アーキテクチャは base と同一）ので base から写す。
        (target / "config.json").write_bytes((base / name / "config.json").read_bytes())
        save_file(state, target / WEIGHT_FILENAME)
        print(f"[single-file] {name}: {len(state)} テンソル", flush=True)

    for entry in SHARED_ENTRIES:
        _link(base / entry, out / entry)
    print(f"[single-file] base から symlink: {', '.join(SHARED_ENTRIES)}", flush=True)

    (out / SOURCE_PROVENANCE_FILE).write_text(
        json.dumps(
            {"file": checkpoint.name, "base_repo": base_repo},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[single-file] {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="civitai の単一ファイル")
    parser.add_argument("--out", type=Path, required=True, help="組み直す先のディレクトリ")
    parser.add_argument("--base-repo", default=BASE_REPO, help="不足部分を引く diffusers リポ")
    args = parser.parse_args()
    convert(args.checkpoint, args.out, args.base_repo)


if __name__ == "__main__":
    main()
