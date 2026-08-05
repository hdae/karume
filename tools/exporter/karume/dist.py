"""配布ディレクトリの組み立て — 系列ディレクトリ群 → HF へそのまま上げられる 1 リポ形。

仕様の正本は ADR 0038（`docs/decisions/0038-manifest-v1.md`）。ここが作るのは §2 の規約名で
並んだファイル群と、それを宣言する `karume.json`（§1〜§3）。

MUST: **manifest は手書きせず資産から導出する**（ADR 0038 Context）。`size` / `sha256` は
組み立て後の実ファイルから streaming で採る — 数 GB を丸読みしないことと、「表と現物が
食い違う」失敗様式を構造的に起こさないことの両方がここに掛かっている。

MUST: 系列に散らばる `io.*.safetensors`（E2E の入出力フィクスチャ）は**配布に含めない**。
出力へ入るのは {@link OUTPUT_PATHS} に載ったファイルだけで、表に無いものは黙って混ざらない。

MUST: `rope_base.safetensors` は f16 / i8 の 2 系列に同名で並ぶ。両者のバイト同一を
sha256 で確かめてから 1 本化する — 食い違ったまま片方を選ぶと、選ばれなかった系列の preset が
「別の幾何の rope 表で走る」形になり、ロードも実行も通って絵だけが静かに壊れる。

同一ファイルシステム上の組み立てなので配置は `os.link`（ハードリンク）を優先し、リンクを
張れない場合（別 FS・権限）だけ copy へ落ちる。数 GB × 2 系列を複製しないため。

    uv run python -m karume.dist
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: リポジトリの `models/`（karume/dist.py → karume → tools/exporter → tools → repo）。
#: P4 の CLI 化ではパスを明示で受け取るため、この既定はリポ内実行の利便のためだけにある。
_REPO_MODELS = Path(__file__).resolve().parents[3] / "models"

#: manifest のファイル名（ADR 0038 §1 — リポジトリ直下の固定名）。
MANIFEST_FILENAME = "karume.json"

#: sha256 の読み出し単位。数 GB を丸読みしないための唯一の要件で、値自体は素の I/O 単位。
_CHUNK_BYTES = 1 << 20

#: 出力の相対 path（ADR 0038 §2 の規約名）— **配置表と manifest が共有する 1 箇所**。
#: 役割名でだけ引くので、綴りが 2 箇所で独立に動くことは起きない。
OUTPUT_PATHS: Mapping[str, str] = {
    "text_encoder": "text_encoder/model.safetensors",
    "text_conditioner": "text_conditioner/model.safetensors",
    "transformer_f16": "transformer/model.f16.safetensors",
    "transformer_i8": "transformer/model.i8.safetensors",
    "rope_base": "transformer/rope_base.safetensors",
    "vae_decoder": "vae_decoder/model.safetensors",
    "tokenizer": "tokenizer/qwen2-tokenizer.json",
    "tokenizer_2": "tokenizer_2/t5-tokenizer.json",
}

#: preset 表。ADR 0038 Examples の Anima 表が正本（`session` の語彙は manifest 所有 — §3）。
ANIMA_PRESETS: Mapping[str, Any] = {
    "f16": {"weights": {"transformer": "f16"}, "session": {}},
    "i8": {"weights": {"transformer": "i8"}, "session": {}},
    "w8a8": {"weights": {"transformer": "i8"}, "session": {"linearCompute": "i8a8"}},
    "w8a8-a8": {
        "weights": {"transformer": "i8"},
        "session": {"linearCompute": "i8a8", "attentionCompute": "i8a8"},
    },
    "w8a8-s16": {
        "weights": {"transformer": "i8"},
        "session": {
            "linearCompute": "i8a8",
            "attentionCompute": "i8a8",
            "attentionScoreStorage": "f16",
        },
    },
    "f16-c16": {
        "weights": {"transformer": "f16"},
        "session": {"linearCompute": "f16", "attentionCompute": "f16"},
        "gpuFeatures": {"shaderF16": True},
    },
}

ANIMA_DEFAULT_PRESET = "w8a8-s16"

#: パイプライン所有の設定（hub は素通し — ADR 0038 §1）。値は移行元の実装定数と一致する:
#: `shift` / `numTrainTimesteps` は sampler の `ANIMA_SHIFT` / `ANIMA_NUM_TRAIN_TIMESTEPS`
#: （エクスポータ側の `SHIFT` / `NUM_TRAIN_TIMESTEPS` = scheduler_config.json と同値）、
#: `steps` / `guidanceScale` は turbo 既定（8 step / cfg 1 — ADR 0038 Examples が正。品質目視
#: ゲート・最終ベンチ・PNG 参照 sha の採取は全て 8 step で行われており、配布既定はそれに揃える。
#: 移行元 CLI の 10 は検証履歴を持たない値）。`negativePrompt` は既定ネガティブプロンプト。
#: `resolution` だけは移行元 CLI の既定（512）を採らない — あちらの 512 は「静的資産の最小」
#: であって推奨値ではなく、配布形は S 形 1 本（ADR 0038 §4）で解像度に依存しない。配布の
#: 推奨既定は ADR 0038 Examples のとおり 1024²。
ANIMA_PIPELINE_CONFIG: Mapping[str, Any] = {
    "scheduler": {"shift": 3, "numTrainTimesteps": 1000},
    "defaults": {
        "steps": 8,
        "guidanceScale": 1,
        "resolution": {"width": 1024, "height": 1024},
        "negativePrompt": "low quality, worst quality, blurry, bad anatomy, jpeg artifacts",
    },
}


class DistError(ValueError):
    """組み立ての前提が破れた（資産の欠落・rope 素表の不一致・manifest と現物の食い違い）。"""


#: 各役割の safetensors ヘッダに**要求する格納 dtype**（存在検査）。実測の事故が根拠:
#: f16 系列のつもりで `--dtype` を付け忘れた素の F32 資産は、組み立て・ロード・実行の全てを
#: 通って**PNG の参照一致まで露見しなかった**。格納形は series ディレクトリ名でなくヘッダが正。
#: f16 系列は fake-quant 対象だけが F16 になる（norm/bias 等は F32 のまま）ので「F16 を含む」
#: を要求する。rope_base（F32 のみ）と tokenizer（JSON）はここに載せない。
STORAGE_REQUIREMENTS: Mapping[str, str] = {
    "text_encoder": "F16",
    "text_conditioner": "F16",
    "transformer_f16": "F16",
    "transformer_i8": "I8",
    "vae_decoder": "F16",
}


def storage_dtypes(path: Path) -> set[str]:
    """safetensors ヘッダのテンソル dtype 集合（ヘッダだけ読む — 数 GB を舐めない）。"""
    size = path.stat().st_size
    with path.open("rb") as stream:
        header_len = int.from_bytes(stream.read(8), "little")
        # 宣言長はファイル実長で拘束する（不正な 8 バイトをそのまま read すると巨大確保になる）。
        if header_len <= 0 or header_len > size - 8:
            raise DistError(
                f"{path}: safetensors ヘッダが読めない（ヘッダ長 {header_len} がファイル長"
                f" {size} と矛盾）"
            )
        try:
            header = json.loads(stream.read(header_len))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DistError(f"{path}: safetensors ヘッダが読めない") from error
    return {spec["dtype"] for name, spec in header.items() if name != "__metadata__"}


def assert_storage(role: str, path: Path) -> None:
    """役割が要求する格納 dtype がヘッダに存在することを検査する（無関係な役割は素通し）。"""
    required = STORAGE_REQUIREMENTS.get(role)
    if required is None:
        return
    if not path.is_file():
        raise DistError(f"組み立ての入力が無い: {path}")
    found = storage_dtypes(path)
    if required not in found:
        raise DistError(
            f"{role}: {path} の格納 dtype に {required} が無い（実際: {sorted(found)}）。"
            "系列を焼いたときの --dtype を確認する（f16 系列は --dtype f16 の fake-quant が必要）"
        )


@dataclass(frozen=True)
class AnimaSources:
    """組み立ての入力となる系列ディレクトリ群。

    テキスト経路と VAE は DiT の格納 dtype に依らないので f16 系列 1 本を共有する
    （ADR 0019）。transformer だけが f16 / i8 の 2 系列に分かれる。
    """

    transformer_f16: Path
    transformer_i8: Path
    base: Path
    tokenizers: Path


def anima_sources(models_dir: Path) -> AnimaSources:
    """リポ内の既定配置（`models/` 直下の系列名）。"""
    return AnimaSources(
        transformer_f16=models_dir / "anima-turbo-f16-dyn",
        transformer_i8=models_dir / "anima-turbo-i8-dyn",
        base=models_dir / "anima-f16",
        tokenizers=models_dir / "anima-demo" / "text",
    )


def sha256_file(path: Path) -> str:
    """ファイルの sha256（小文字 hex 64 桁）を streaming で採る。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def place_file(source: Path, dest: Path) -> str:
    """`source` を `dest` へ置く。ハードリンク優先・不可なら copy。返り値は採れた手段。

    既存の `dest` は先に外す（再組み立てで前回のリンクが残っていると `os.link` が落ちる）。
    """
    if not source.is_file():
        raise DistError(f"組み立ての入力が無い: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)
    try:
        os.link(source, dest)
    except OSError:
        shutil.copyfile(source, dest)
        return "copy"
    return "link"


def file_ref(out_dir: Path, rel_path: str) -> dict[str, Any]:
    """ADR 0038 §2 の 3 点セット `{path, size, sha256}` を実ファイルから導出する。"""
    path = out_dir / rel_path
    return {"path": rel_path, "size": path.stat().st_size, "sha256": sha256_file(path)}


def shared_rope_base(sources: AnimaSources) -> Path:
    """f16 / i8 系列の rope 素表がバイト同一であることを確かめ、1 本化する元を返す。"""
    candidates = [
        series / "transformer" / "rope_base.safetensors"
        for series in (sources.transformer_f16, sources.transformer_i8)
    ]
    for path in candidates:
        if not path.is_file():
            raise DistError(f"組み立ての入力が無い: {path}")
    digests = {path: sha256_file(path) for path in candidates}
    if len(set(digests.values())) != 1:
        listing = "\n".join(f"  {digest}  {path}" for path, digest in digests.items())
        raise DistError(
            "rope_base.safetensors が系列間でバイト同一でない — 1 本化できない。"
            f"どちらが正かはここでは決められないので組み立てを止める:\n{listing}"
        )
    return candidates[0]


def anima_placements(sources: AnimaSources) -> dict[str, Path]:
    """役割名 → 出所のファイル。出力の path は {@link OUTPUT_PATHS} が持つ。

    この表に無いものは出力へ入らない（`io.*.safetensors` を落とす仕掛けはこれで足りる）。
    """
    return {
        "text_encoder": sources.base / "text_encoder" / "model.safetensors",
        "text_conditioner": sources.base / "text_conditioner" / "model.safetensors",
        "transformer_f16": sources.transformer_f16 / "transformer" / "model.safetensors",
        "transformer_i8": sources.transformer_i8 / "transformer" / "model.safetensors",
        "rope_base": shared_rope_base(sources),
        "vae_decoder": sources.base / "vae_decoder" / "model.safetensors",
        "tokenizer": sources.tokenizers / "qwen2-tokenizer.json",
        "tokenizer_2": sources.tokenizers / "t5-tokenizer.json",
    }


def generator_tag() -> str:
    """`generator` に焼く値（ADR 0038 §1 — 障害報告の照合用・実行意味論なし）。"""
    return f"karume/{importlib.metadata.version('karume')}"


def anima_manifest(out_dir: Path) -> dict[str, Any]:
    """組み立て済みディレクトリから Anima の `karume.json` を導出する。"""
    refs = {role: file_ref(out_dir, path) for role, path in OUTPUT_PATHS.items()}
    rope_base = refs["rope_base"]
    return {
        "format": "karume/1",
        "generator": generator_tag(),
        "pipeline": "anima/1",
        "components": {
            "text_encoder": {"file": refs["text_encoder"]},
            "text_conditioner": {"file": refs["text_conditioner"]},
            "transformer": {
                "variants": {
                    "f16": {
                        "file": refs["transformer_f16"],
                        "extras": {"rope_base": rope_base},
                    },
                    "i8": {
                        "file": refs["transformer_i8"],
                        "extras": {"rope_base": rope_base},
                    },
                }
            },
            "vae_decoder": {"file": refs["vae_decoder"]},
            "tokenizer": {"file": refs["tokenizer"]},
            "tokenizer_2": {"file": refs["tokenizer_2"]},
        },
        "presets": dict(ANIMA_PRESETS),
        "defaultPreset": ANIMA_DEFAULT_PRESET,
        "pipelineConfig": dict(ANIMA_PIPELINE_CONFIG),
    }


def assemble_anima(sources: AnimaSources, out_dir: Path) -> dict[str, Any]:
    """系列群を配布形へ組み立て、`karume.json` を書いて manifest を返す。"""
    placements = anima_placements(sources)
    # MUST: 検査は配置の**前**に全役割ぶん済ませる（rope 不一致と同じ規律 — 落ちるなら
    # 途中の配布形を 1 ファイルも残さない）。
    for role, source in placements.items():
        assert_storage(role, source)
    out_dir.mkdir(parents=True, exist_ok=True)
    for role, source in placements.items():
        place_file(source, out_dir / OUTPUT_PATHS[role])
    manifest = anima_manifest(out_dir)
    (out_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def _referenced_sizes(manifest: Mapping[str, Any]) -> dict[str, int]:
    """manifest が参照する全ファイルの `{path: size}`（同一 path の重複は 1 つに畳む）。"""
    sizes: dict[str, int] = {}
    for component in manifest["components"].values():
        entries = component["variants"].values() if "variants" in component else (component,)
        for entry in entries:
            for ref in (entry["file"], *entry.get("extras", {}).values()):
                sizes[ref["path"]] = ref["size"]
    return sizes


def verify_dist(out_dir: Path) -> dict[str, int]:
    """`karume.json` と現物を突き合わせる（実在・size 一致・宣言外ファイルの不在）。

    sha256 は組み立て時に実ファイルから採っているので採り直さない（数 GB の再ハッシュは
    ここでは新しい事実を生まない）。見るのは「表が現物を覆っているか」だけ。
    """
    manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    declared = _referenced_sizes(manifest)
    for rel_path, size in sorted(declared.items()):
        path = out_dir / rel_path
        if not path.is_file():
            raise DistError(f"manifest が参照するファイルが無い: {rel_path}")
        actual = path.stat().st_size
        if actual != size:
            raise DistError(f"{rel_path}: size が manifest と違う（宣言 {size} / 現物 {actual}）")
    present = {
        str(path.relative_to(out_dir))
        for path in out_dir.rglob("*")
        if path.is_file() and path.name != MANIFEST_FILENAME
    }
    extra = sorted(present - set(declared))
    if extra:
        raise DistError(f"manifest が宣言していないファイルが混ざっている: {', '.join(extra)}")
    return declared


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="配布ディレクトリ（HF アップ可能形）の組み立て")
    parser.add_argument(
        "--models",
        type=Path,
        default=_REPO_MODELS,
        help="系列ディレクトリ群の親（既定: リポの models/）",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="出力先（既定: <--models>/anima-turbo）"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out_dir = args.out if args.out is not None else args.models / "anima-turbo"
    manifest = assemble_anima(anima_sources(args.models), out_dir)
    verified = verify_dist(out_dir)
    for rel_path, size in sorted(verified.items()):
        print(f"{size:>12}  {rel_path}")
    print(f"{(out_dir / MANIFEST_FILENAME).stat().st_size:>12}  {MANIFEST_FILENAME}")
    print(f"[dist] {out_dir} — {manifest['generator']} / preset {manifest['defaultPreset']}")


if __name__ == "__main__":
    main()
