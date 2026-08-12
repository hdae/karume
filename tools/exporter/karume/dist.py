"""配布ディレクトリの組み立て — 系列ディレクトリ群 → HF へそのまま上げられる 1 リポ形。

仕様の正本は ADR 0041（`docs/decisions/0041-manifest-v2.md`）。ここが作るのは §2 の形で
並んだファイル群と、それを宣言する `karume.json`（`karume/2`）、そして manifest から機械導出
したモデルカード `README.md`（ADR 0037 §3 の「そのまま HF リポとして上げられる形」）。

**リポ内レイアウトは一律「モデル別サブツリー + `shared/` + 直下 `karume.json` / `README.md`」**
（ADR 0041 §9）。単一モデルのリポも同じ規則で `<モデル名>/…` の下へ入る — 1 モデルだけ平置き
という例外を作ると、2 個目のモデルを足した瞬間に既存 path が全部動く。

**pipeline 別のディスパッチ**（`--pipeline anima|sbv2|irodori`）と**モデル別の軸**（`--model`）。
共有するのは「置く / sha256 を採る / 共有ファイルを畳む / 宣言と現物を突き合わせる」層だけで、
どのファイルをどの名前で並べ何を宣言するかは pipeline ごとの表が持つ（{@link PIPELINES}）。

MUST: **manifest は手書きせず資産から導出する**（ADR 0038 Context を v2 でも維持）。`size` /
`sha256` は組み立て後の実ファイルから streaming で採る — 数 GB を丸読みしないことと、「表と
現物が食い違う」失敗様式を構造的に起こさないことの両方がここに掛かっている。

MUST: 系列に散らばる `io.*.safetensors`（E2E の入出力フィクスチャ）は**配布に含めない**。
出力へ入るのは pipeline ごとの出力 path 表に載ったファイルだけで、表に無いものは黙って
混ざらない。

MUST: 検査（格納 dtype / rope 素表のバイト同一 / スタイル表・話者表の行数）は**配置の前**に
**全モデルぶん**済ませる — 落ちるなら途中の配布形を 1 ファイルも残さない。組み立ては
「計画（{@link ModelPlan} を組む = 検査と読み取りの全部）→ 実体化（{@link assemble_family}）」の
2 段で、前段は 1 バイトも書かない。

配置は常に**独立したコピー**（ハードリンク禁止 — 2026-08-09 裁定・ADR 0041 追記）。系列の
書き手は既存ファイルを truncate で上書きするため、リンク共有した配布形は系列の再 export で
黙って中身が変わり、manifest の sha256 と現物が食い違う。配布形は系列から独立した
自己完結スナップショットとして吐き出す。

    uv run karume dist                                    # = uv run python -m karume.dist
    uv run karume dist --pipeline sbv2 --card-profile fn
    uv run karume dist --pipeline irodori                 # → models/karume-irodori-v4-small/
    uv run karume dist --pipeline sbv2 --card-profile jvnv \\
        --model F1 --model F2 --out ../../models/karume-sbv2-jvnv

`--card-profile` はモデルカードの**帰属**（出所・ライセンス・引用）の選択で、sbv2 では必須
（{@link resolve_card_renderer}）。組み立てる資産には掛からない — 同じ重みでも、どのファミリー
として配るかで帰属が変わる。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save

from karume.ir import IR_METADATA_KEY
from karume.modelcard import (
    HF_OWNER,
    SBV2_CARD_PROFILES,
    render_irodori_model_card,
    render_model_card,
    render_sbv2_model_card,
)
from karume.paths import DIST_ROOT, INPUTS_ROOT, OUTPUTS_ROOT, SERIES_ROOT

# ---- ① 共有部: 置き場の綴り・配置・ハッシュ・共有の畳み込み・宣言と現物の突合 -------

#: manifest のファイル名（ADR 0041 §1 — リポジトリ直下の固定名）。
MANIFEST_FILENAME = "karume.json"

#: manifest の形式識別子（ADR 0041 §1 — hub は v2 だけを読む）。
MANIFEST_FORMAT = "karume/2"

#: モデルカードのファイル名（ADR 0037 §3 — HF が frontmatter を読む固定名）。
MODEL_CARD_FILENAME = "README.md"

#: 複数モデルが**同じ相対 path・同じ sha256** で持つファイルを 1 回だけ置く席（ADR 0041 §5）。
#: 専用の間接参照は持たず、各モデルの manifest エントリが同じ path を書くだけで共有になる。
SHARED_DIRNAME = "shared"

#: 配布形の**メタファイル**（配布形そのものの説明であって、manifest が宣言する資産ではない）。
#: 宣言外ファイル検査はこの 2 つだけを例外にする — 例外を名前でなく相対 path で持つのは、
#: 下位ディレクトリに紛れ込んだ同名ファイルまで見逃さないため。**在ることは要求しない**
#: （{@link verify_dist} はモデルカードを書く**前**に走るので、無いまま通る必要がある）。
META_PATHS = frozenset({MANIFEST_FILENAME, MODEL_CARD_FILENAME})

#: 規模上限（ADR 0041 §7）。hub が同じ値で弾くので、**焼く側で先に落とす**
#: （配布してから利用者の手元で初めて分かる形にしない）。
MAX_MODELS = 32
MAX_WEIGHTS = 32
MAX_ASSETS = 32
MAX_QUANTS = 32
MAX_PIPELINE_CONFIG_BYTES = 256 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024

#: モデル名の許可文字。モデル名は manifest のキーであると同時に**リポ内のディレクトリ名**
#: なので、hub の path セグメント検査（ADR 0041 §6）と同じ集合に縛る。
MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]*$")

#: sha256 の読み出し単位。数 GB を丸読みしないための唯一の要件で、値自体は素の I/O 単位。
_CHUNK_BYTES = 1 << 20


class DistError(ValueError):
    """組み立ての前提が破れた（資産の欠落・rope 素表の不一致・manifest と現物の食い違い）。"""


def safetensors_header(path: Path) -> Mapping[str, Any]:
    """safetensors のヘッダ JSON だけを読む（数 GB のペイロードを舐めない）。"""
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
    if not isinstance(header, dict):
        raise DistError(f"{path}: safetensors ヘッダが最上位オブジェクトでない")
    return header


def storage_dtypes(path: Path) -> set[str]:
    """safetensors ヘッダのテンソル dtype 集合（ヘッダだけ読む — 数 GB を舐めない）。"""
    header = safetensors_header(path)
    return {spec["dtype"] for name, spec in header.items() if name != "__metadata__"}


def assert_storage(role: str, path: Path, requirements: Mapping[str, str]) -> None:
    """役割が要求する格納 dtype がヘッダに存在することを検査する（無関係な役割は素通し）。

    要求表を引数で受けるのは、役割名が pipeline 間で衝突するため（Anima の `text_encoder` は
    F16 を要求し、SBV2 の `text_encoder` は I8 を要求する）。1 つの表に混ぜると、どちらかの
    要求が黙って他方に掛かる。
    """
    required = requirements.get(role)
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


def sha256_file(path: Path) -> str:
    """ファイルの sha256（小文字 hex 64 桁）を streaming で採る。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def place_file(source: Path, dest: Path) -> None:
    """`source` を `dest` へ**独立したコピー**として置く。

    MUST: ハードリンクは使わない — 系列の書き手（`emit` の `open("wb")`）は既存ファイルを
    truncate で上書きするため、リンクで置いた配布形は系列の再 export で黙って書き換わり、
    manifest の sha256 と現物が食い違う（verify_dist は sha256 を採り直さないので沈黙する）。

    既存の `dest` は先に外す — リンク方式だった頃の配布形の上へ再組み立てするとき、unlink が
    先にリンクを切る（外さず開くと系列側の実ファイルを書き換える）。
    """
    if not source.is_file():
        raise DistError(f"組み立ての入力が無い: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)
    shutil.copyfile(source, dest)


@dataclass(frozen=True)
class Artifact:
    """配布形へ入る 1 ファイル。`rel_path` は**モデルサブツリー内**の相対 path。

    出所は 2 通りだけ: 系列からの**配置**（`source`）と、組み立てが作る**生成物**（`payload`
    — `.npy` / ckpt から移した表など）。どちらか一方だけを持つ。生成物をバイト列で持つのは、
    「置く前に中身が確定している」ことを共有判定（{@link assemble_family}）と同じ規律で
    扱えるようにするため。
    """

    rel_path: str
    source: Path | None = None
    payload: bytes | None = None

    def __post_init__(self) -> None:
        if (self.source is None) == (self.payload is None):
            raise DistError(f"{self.rel_path}: Artifact は source / payload のどちらか一方を持つ")


def materialize(artifact: Artifact, dest: Path) -> None:
    """`Artifact` の実体を `dest` に作る（配置 or 書き出し）。

    MUST: 生成物も既存 `dest` を先に外してから書く — リンク方式だった頃の配布形が残っている
    場合、外さず開いて書いた瞬間に系列側の実ファイルを書き換えることになる。
    """
    if artifact.source is not None:
        place_file(artifact.source, dest)
        return
    assert artifact.payload is not None  # __post_init__ の不変条件
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)
    dest.write_bytes(artifact.payload)


@dataclass(frozen=True)
class WeightFiles:
    """weights の 1 dtype ぶん（ADR 0041 §3 の `{file, extras?}`）。中身は**役割名**。

    実 path は {@link Artifact} 側が持つ — 共有の畳み込みで path が `shared/…` へ動くので、
    宣言側が path を直に握っていると 2 箇所が独立に動く。
    """

    file: str
    extras: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelPlan:
    """1 モデルぶんの組み立て計画 — **検査と読み取りを全部済ませた**、まだ 1 バイトも書いて
    いない状態。ファミリー組み立てはこれを N 本受け取ってから初めて配置に入る。
    """

    name: str
    #: パイプライン契約（`anima/1` — ADR 0041 §2 でモデル単位になった）。
    pipeline: str
    #: 役割名 → 配布形へ入る 1 ファイル。
    artifacts: Mapping[str, Artifact]
    #: weights 名 → dtype ラベル → ファイル群（役割名で指す）。
    weights: Mapping[str, Mapping[str, WeightFiles]]
    #: assets 名 → 役割名（quant 選択に依存しない無条件ファイル）。
    assets: Mapping[str, str]
    quants: Mapping[str, Any]
    default_quant: str
    pipeline_config: Mapping[str, Any]


def generator_tag() -> str:
    """`generator` に焼く値（ADR 0041 §2 — 障害報告の照合用・実行意味論なし）。"""
    return f"karume/{importlib.metadata.version('karume')}"


def assert_model_name(name: str) -> str:
    """モデル名を検査して返す（manifest のキー兼リポ内ディレクトリ名）。"""
    if not MODEL_NAME_RE.match(name):
        raise DistError(
            f"モデル名 '{name}' は許可文字 [A-Za-z0-9._-]（先頭のドット不可）に一致しない"
            " — モデル名はリポ内のディレクトリ名でもあるので hub の path 検査と同じ集合に縛る"
        )
    if name == SHARED_DIRNAME:
        raise DistError(f"モデル名に '{SHARED_DIRNAME}' は使えない（共有ファイルの席と衝突する）")
    return name


def complete_quant_weights(
    weights: Mapping[str, Mapping[str, WeightFiles]],
    quants: Mapping[str, Any],
) -> dict[str, Any]:
    """quant の `weights` 写像を**完全写像**へ埋める（hub の受理要件 — ADR 0041 §3）。

    dtype ラベルが 1 つしかない weights は quant で選びようがないので表に書かせず、ここが
    機械的に埋める（書かせると「選択肢の無い席」が quant の数だけ複製され、dtype を増やした
    ときに一斉更新が要る = 導出可能な状態の二重化）。**2 つ以上あるものは表が名指ししていな
    ければ落とす** — 既定を勝手に選ぶと、黙って別の格納形が配られる。
    """
    completed: dict[str, Any] = {}
    for quant_name, quant in quants.items():
        declared = quant["weights"]
        for name in declared:
            if name not in weights:
                raise DistError(f"quant '{quant_name}': 未知の weights '{name}'")
            if declared[name] not in weights[name]:
                raise DistError(
                    f"quant '{quant_name}': weights '{name}' に dtype '{declared[name]}' が無い"
                    f"（利用可能: {sorted(weights[name])}）"
                )
        # MUST: 並びは **weights の宣言順**（表が書いた順ではない）— 埋めた席だけが末尾に
        # 溜まると、manifest の weights 節と quant 節で同じ役割が別の順に並ぶ。
        mapping: dict[str, str] = {}
        for name, labels in weights.items():
            if name in declared:
                mapping[name] = declared[name]
                continue
            if len(labels) != 1:
                raise DistError(
                    f"quant '{quant_name}': weights '{name}' の dtype が {sorted(labels)} の"
                    "複数あるのに指定が無い — quant 表が名指しする"
                )
            mapping[name] = next(iter(labels))
        completed[quant_name] = {**quant, "weights": mapping}
    return completed


def file_ref(out_dir: Path, rel_path: str, sha256: str) -> dict[str, Any]:
    """ADR 0041 §2 の 3 点セット `{path, size, sha256}`（size は置いた現物から採る）。"""
    return {
        "path": rel_path,
        "size": (out_dir / rel_path).stat().st_size,
        "sha256": sha256,
    }


def _prune_empty_dirs(out_dir: Path, start: Path) -> None:
    """`start` から `out_dir` まで、空になったディレクトリだけを畳む（共有の畳み込みの後始末）。"""
    current = start
    while current != out_dir and current.is_dir() and not any(current.iterdir()):
        current.rmdir()
        current = current.parent


def _fold_shared(
    out_dir: Path,
    plans: Sequence[ModelPlan],
    placed: dict[tuple[str, str], str],
    digests: dict[str, str],
) -> None:
    """同じ相対 path・同じ sha256 のファイルを 2 モデル以上が持つとき `shared/` へ 1 回だけ置く。

    MUST: 一致の条件は **モデルサブツリー内の相対 path と sha256 の両方**（ADR 0041 §5 の
    「path の一致で共有」を、中身が同じであることまで確かめてから成立させる）。中身違いを
    同じ席へ寄せると、どちらのモデルかで別の重みが黙って読まれる。
    """
    groups: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for plan in plans:
        for role, artifact in plan.artifacts.items():
            key = (artifact.rel_path, digests[placed[(plan.name, role)]])
            groups.setdefault(key, []).append((plan.name, role))
    for (rel_path, digest), members in groups.items():
        if len({model for model, _ in members}) < 2:
            continue
        target = f"{SHARED_DIRNAME}/{rel_path}"
        (out_dir / target).parent.mkdir(parents=True, exist_ok=True)
        (out_dir / target).unlink(missing_ok=True)
        moved = False
        for model, role in members:
            source_rel = placed[(model, role)]
            if source_rel == target:
                continue
            source = out_dir / source_rel
            if moved:
                source.unlink(missing_ok=True)
            else:
                os.replace(source, out_dir / target)
                moved = True
            digests.pop(source_rel, None)
            _prune_empty_dirs(out_dir, source.parent)
            placed[(model, role)] = target
        digests[target] = digest


def _model_entry(plan: ModelPlan, refs: Mapping[str, dict[str, Any]]) -> dict:
    """1 モデルぶんの manifest エントリ（ADR 0041 §2）。`refs` は役割名 → 3 点セット。"""
    weights: dict[str, Any] = {}
    for name, labels in plan.weights.items():
        entry: dict[str, Any] = {}
        for label, files in labels.items():
            extras = {extra: refs[role] for extra, role in files.extras.items()}
            entry[label] = {"file": refs[files.file], **({"extras": extras} if extras else {})}
        weights[name] = entry
    return {
        "pipeline": plan.pipeline,
        "weights": weights,
        "assets": {name: refs[role] for name, role in plan.assets.items()},
        "quants": dict(plan.quants),
        "defaultQuant": plan.default_quant,
        "pipelineConfig": dict(plan.pipeline_config),
    }


def assert_manifest_limits(manifest: Mapping[str, Any]) -> None:
    """規模上限（ADR 0041 §7）を焼く側で先に落とす。"""
    models = manifest["models"]
    if len(models) > MAX_MODELS:
        raise DistError(f"models が {len(models)} 件で上限 {MAX_MODELS} を超えた")
    for name, model in models.items():
        for key, limit in (
            ("weights", MAX_WEIGHTS),
            ("assets", MAX_ASSETS),
            ("quants", MAX_QUANTS),
        ):
            if len(model[key]) > limit:
                raise DistError(f"{name}.{key} が {len(model[key])} 件で上限 {limit} を超えた")
        config_bytes = len(json.dumps(model["pipelineConfig"], ensure_ascii=False).encode("utf-8"))
        if config_bytes > MAX_PIPELINE_CONFIG_BYTES:
            raise DistError(
                f"{name}.pipelineConfig が {config_bytes} バイトで上限"
                f" {MAX_PIPELINE_CONFIG_BYTES} を超えた"
            )
    total = len(manifest_text(manifest).encode("utf-8"))
    if total > MAX_MANIFEST_BYTES:
        raise DistError(f"manifest が {total} バイトで上限 {MAX_MANIFEST_BYTES} を超えた")


def manifest_text(manifest: Mapping[str, Any]) -> str:
    """`karume.json` に書く綴り（末尾改行つき — 上限検査もこの綴りで測る）。"""
    return json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"


def assemble_family(
    plans: Sequence[ModelPlan], out_dir: Path, default_model: str
) -> dict[str, Any]:
    """計画済みのモデル群を 1 リポへ組み立て、`karume.json` を書いて manifest を返す。

    ① 各モデルのサブツリーへ置く → ② 共有ファイルを `shared/` へ畳む → ③ 現物から manifest。
    検査は計画（`plans` を組む段）で済んでいるので、ここから先は落ちない前提で書ける。
    """
    if not plans:
        raise DistError("組み立てるモデルが 1 つも無い")
    names = [plan.name for plan in plans]
    if len(set(names)) != len(names):
        raise DistError(f"モデル名が重複している: {names}")
    if default_model not in names:
        raise DistError(f"既定モデル '{default_model}' が組み立て対象 {names} に無い")

    out_dir.mkdir(parents=True, exist_ok=True)
    placed: dict[tuple[str, str], str] = {}
    digests: dict[str, str] = {}
    for plan in plans:
        for role, artifact in plan.artifacts.items():
            rel_path = f"{plan.name}/{artifact.rel_path}"
            materialize(artifact, out_dir / rel_path)
            placed[(plan.name, role)] = rel_path
            # sha256 は**置いた現物**から採る（表と現物が食い違う失敗様式を構造的に消す）。
            digests[rel_path] = sha256_file(out_dir / rel_path)
    _fold_shared(out_dir, plans, placed, digests)

    models: dict[str, Any] = {}
    for plan in plans:
        refs = {}
        for role in plan.artifacts:
            rel_path = placed[(plan.name, role)]
            refs[role] = file_ref(out_dir, rel_path, digests[rel_path])
        models[plan.name] = _model_entry(plan, refs)
    manifest = {
        "format": MANIFEST_FORMAT,
        "generator": generator_tag(),
        "defaultModel": default_model,
        "models": models,
    }
    assert_manifest_limits(manifest)
    (out_dir / MANIFEST_FILENAME).write_text(manifest_text(manifest), encoding="utf-8")
    return manifest


# ---- ② 宣言と現物の突合 ------------------------------------------------------


def _declared_refs(manifest: Mapping[str, Any]) -> Iterator[tuple[str, Mapping[str, Any]]]:
    """manifest が参照する全ファイルを `(場所, 3 点セット)` で流す（重複はそのまま流す）。"""
    for model_name, model in manifest["models"].items():
        for name, labels in model["weights"].items():
            for label, entry in labels.items():
                yield f"models.{model_name}.weights.{name}.{label}", entry["file"]
                for extra, ref in entry.get("extras", {}).items():
                    yield f"models.{model_name}.weights.{name}.{label}.extras.{extra}", ref
        for name, ref in model["assets"].items():
            yield f"models.{model_name}.assets.{name}", ref


def _assert_manifest_shape(manifest: Mapping[str, Any]) -> None:
    """`karume/2` の構造整合（hub のパーサが受理する形かを焼いた側でも見る）。

    ここが見るのは**この配布形が自分で閉じているか**だけ — `defaultModel` / `defaultQuant` の
    指し先、quant の weights 完全写像、そしてレイアウト（ADR 0041 §9）。hub の全検査を写経
    しても正本が 2 つになるだけなので、写すのは「組み立てが壊れたら真っ先に破れる」規則に絞る。
    """
    if manifest.get("format") != MANIFEST_FORMAT:
        raise DistError(f"format が '{MANIFEST_FORMAT}' でない: {manifest.get('format')!r}")
    if not manifest.get("generator"):
        raise DistError("generator が無い / 空")
    models = manifest.get("models")
    if not isinstance(models, dict) or not models:
        raise DistError("models が非空のオブジェクトでない")
    if manifest.get("defaultModel") not in models:
        raise DistError(
            f"defaultModel '{manifest.get('defaultModel')}' が models {sorted(models)} に無い"
        )
    for model_name, model in models.items():
        weights = model["weights"]
        quants = model["quants"]
        if model["defaultQuant"] not in quants:
            raise DistError(
                f"{model_name}.defaultQuant '{model['defaultQuant']}' が"
                f" quants {sorted(quants)} に無い"
            )
        for quant_name, quant in quants.items():
            where = f"{model_name}.quants.{quant_name}"
            if set(quant["weights"]) != set(weights):
                raise DistError(
                    f"{where}.weights が weights の完全写像でない"
                    f"（宣言 {sorted(quant['weights'])} / weights {sorted(weights)}）"
                )
            for name, label in quant["weights"].items():
                if label not in weights[name]:
                    raise DistError(
                        f"{where}.weights: '{name}' に dtype '{label}' が無い"
                        f"（利用可能: {sorted(weights[name])}）"
                    )
    allowed_roots = set(models) | {SHARED_DIRNAME}
    for where, ref in _declared_refs(manifest):
        root = ref["path"].split("/")[0]
        if root not in allowed_roots:
            raise DistError(
                f"{where}: path '{ref['path']}' の先頭が {sorted(allowed_roots)} のどれでもない"
                " — レイアウトはモデル別サブツリー + shared/（ADR 0041 §9）"
            )


def _declared_sizes(manifest: Mapping[str, Any]) -> dict[str, int]:
    """manifest が参照する全ファイルの `{path: size}`。重複 path は 3 点セットの一致を要求する。

    同一 path の重複参照はモデル間の共有そのもの（ADR 0041 §5）なので合法だが、`{size, sha256}`
    が食い違えば取得層のキャッシュが振動する — hub と同じ規則をここでも落とす。
    """
    sizes: dict[str, int] = {}
    seen: dict[str, Mapping[str, Any]] = {}
    for where, ref in _declared_refs(manifest):
        previous = seen.get(ref["path"])
        if previous is not None and (
            previous["size"] != ref["size"] or previous["sha256"] != ref["sha256"]
        ):
            raise DistError(
                f"{where}: 重複 path '{ref['path']}' の size / sha256 が食い違う"
                f"（{previous['size']} / {previous['sha256']} と {ref['size']} / {ref['sha256']}）"
            )
        seen[ref["path"]] = ref
        sizes[ref["path"]] = ref["size"]
    return sizes


def verify_dist(out_dir: Path) -> dict[str, int]:
    """`karume.json` と現物を突き合わせる（構造整合・実在・size 一致・宣言外ファイルの不在）。

    sha256 は組み立て時に実ファイルから採っているので採り直さない（数 GB の再ハッシュは
    ここでは新しい事実を生まない）。見るのは「表が自分で閉じていて、現物を覆っているか」だけ。

    宣言外ファイルの例外は {@link META_PATHS} の 2 つ（`karume.json` と `README.md`）だけ —
    どちらも配布形そのものの説明で、manifest が宣言する資産ではない。それ以外は従来どおり
    fail loudly（前回の組み立ての残骸や `io.*` の混入を後段へ見せない）。
    """
    manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    _assert_manifest_shape(manifest)
    declared = _declared_sizes(manifest)
    for rel_path, size in sorted(declared.items()):
        path = out_dir / rel_path
        if not path.is_file():
            raise DistError(f"manifest が参照するファイルが無い: {rel_path}")
        actual = path.stat().st_size
        if actual != size:
            raise DistError(f"{rel_path}: size が manifest と違う（宣言 {size} / 現物 {actual}）")
    present = {
        relative
        for path in out_dir.rglob("*")
        if path.is_file() and (relative := str(path.relative_to(out_dir))) not in META_PATHS
    }
    extra = sorted(present - set(declared))
    if extra:
        raise DistError(f"manifest が宣言していないファイルが混ざっている: {', '.join(extra)}")
    return declared


# ---- ③ Anima（text-to-image）-------------------------------------------------

#: 既定のモデル名（= 単一モデルなら既定のリポ名でもある）。turbo LoRA を焼き込んだ配布物で
#: あることを名前に出す。
ANIMA_MODEL_NAME = "anima-turbo"

#: パイプライン契約（ADR 0041 §2 — モデル単位）。
ANIMA_PIPELINE = "anima/1"

#: モデル名に依らない系列（base の text 経路 / VAE と、tokenizer を書く台本の出力）。
ANIMA_BASE_SERIES = "anima-f16"
ANIMA_TOKENIZER_SERIES = "anima-demo"

#: 出力の相対 path（**モデルサブツリー内**）— 配置表と manifest が共有する 1 箇所。
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

#: quant 表（v1 の `presets` — ADR 0041 §3 で改名）。`session` の語彙は manifest 所有。
#: **dtype ラベルが 1 つしかない weights は書かない** — {@link complete_quant_weights} が
#: 完全写像へ埋める（写せば済む席を quant の数だけ複製しない）。
ANIMA_QUANTS: Mapping[str, Any] = {
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

ANIMA_DEFAULT_QUANT = "w8a8-s16"

#: パイプライン所有の設定（hub は素通し — ADR 0041 §2）。値は移行元の実装定数と一致する:
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

#: weights の宣言（dtype ラベル → 役割名）。ラベルは**格納 dtype 語彙**で、
#: {@link STORAGE_REQUIREMENTS} が要求する格納形と 1:1（ADR 0041 §3）。
ANIMA_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    "text_encoder": {"f16": WeightFiles("text_encoder")},
    "text_conditioner": {"f16": WeightFiles("text_conditioner")},
    "transformer": {
        "f16": WeightFiles("transformer_f16", {"rope_base": "rope_base"}),
        "i8": WeightFiles("transformer_i8", {"rope_base": "rope_base"}),
    },
    "vae_decoder": {"f16": WeightFiles("vae_decoder")},
}

#: assets の宣言（quant 選択に依存しない無条件ファイル — ADR 0041 §3）。
ANIMA_ASSETS: Mapping[str, str] = {"tokenizer": "tokenizer", "tokenizer_2": "tokenizer_2"}


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


def anima_sources(series_dir: Path, model: str = ANIMA_MODEL_NAME) -> AnimaSources:
    """系列の親ディレクトリ（`outputs/series/`）から Anima の 4 系列を引く。

    モデル名は transformer の 2 系列にだけ掛かる — base（text 経路 / VAE）と tokenizer は
    素のアーキテクチャ側の出力で、turbo かどうかに依らない。
    """
    return AnimaSources(
        transformer_f16=series_dir / f"{model}-f16-dyn",
        transformer_i8=series_dir / f"{model}-i8-dyn",
        base=series_dir / ANIMA_BASE_SERIES,
        tokenizers=series_dir / ANIMA_TOKENIZER_SERIES / "text",
    )


def shared_rope_base(sources: AnimaSources) -> Path:
    """f16 / i8 系列の rope 素表がバイト同一であることを確かめ、1 本化する元を返す。

    MUST: `rope_base.safetensors` は f16 / i8 の 2 系列に同名で並ぶ。両者のバイト同一を
    sha256 で確かめてから 1 本化する — 食い違ったまま片方を選ぶと、選ばれなかった系列の
    quant が「別の幾何の rope 表で走る」形になり、ロードも実行も通って絵だけが静かに壊れる。
    """
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


def anima_plan(sources: AnimaSources, model: str = ANIMA_MODEL_NAME) -> ModelPlan:
    """Anima 1 モデルぶんの計画を組む（検査はここで全部済ませる — 1 バイトも書かない）。"""
    assert_model_name(model)
    placements = anima_placements(sources)
    for role, source in placements.items():
        assert_storage(role, source, STORAGE_REQUIREMENTS)
    return ModelPlan(
        name=model,
        pipeline=ANIMA_PIPELINE,
        artifacts={
            role: Artifact(OUTPUT_PATHS[role], source=source) for role, source in placements.items()
        },
        weights=ANIMA_WEIGHTS,
        assets=ANIMA_ASSETS,
        quants=complete_quant_weights(ANIMA_WEIGHTS, ANIMA_QUANTS),
        default_quant=ANIMA_DEFAULT_QUANT,
        pipeline_config=ANIMA_PIPELINE_CONFIG,
    )


# ---- ④ SBV2（text-to-speech）------------------------------------------------
#
# 配布するのは**実行に要る 3 グラフ + ホスト資産 4 本**だけ（ADR 0038 §2 の SBV2 例は
# 5 グラフを並べるが、あれは形の例示）。`front` = enc_p + dp + sdp、`voice` = flow + dec の
# 融合なので、`dp` / `flow` / `dec` は golden 検証専用の単体グラフで配布形には入らない。
# `text_encoder`（DeBERTa）が i8 単体なのは、`export_deberta.py` が f16 を持たず、i8 は
# ADR 0026 が聴感ゲート込みで受理済みだから（f32 の 1.32GB は配布に非現実的）。
#
# ホスト資産のうち `style_vectors` / `speaker_embeddings` は**表を配って実行時に行を引く**形。
# `front` / `voice` のグラフ入力 `style_vec[1,256]` / `g[1,512,1]` はこの 2 表から作られ、
# 名前 → 行の対応は `pipelineConfig` の `styles` / `speakers` が持つ（3 つで 1 組）。

#: 既定のモデル名 — 系列の綴り（`sbv2-FN4{,-f16,-i8}`）と実重みの置き場を束ねる 1 語。
#: `export_sbv2.default_out_root` が `--model-dir` のディレクトリ名から系列名を作るので、
#: 読み手のこちらも同じ 1 語から組む。
SBV2_DEFAULT_MODEL = "FN4"

#: 系列名とリポ名の接頭辞（`sbv2-<モデル名>`）。
SBV2_SERIES_PREFIX = "sbv2"

#: パイプライン契約（ADR 0041 §2 — モデル単位）。
SBV2_PIPELINE = "sbv2/1"

#: DeBERTa の系列とその variant ディレクトリ（`export_deberta.py` の綴り）。モデル名に依らない
#: （ファミリー組み立てでは全モデルが同じ text_encoder を指し、`shared/` へ 1 回だけ入る）。
#: 22 層 variant なのは末尾 2 層が SBV2 の経路で死んでいるから（{@link SBV2_TEXT_ENCODER_LAYERS}）。
SBV2_TEXT_ENCODER_SERIES = "deberta-i8"
SBV2_TEXT_ENCODER_VARIANT = "sbv2-22layer"

#: 配布 text_encoder に残っている encoder 層の数。SBV2 が使う `hidden_states[-3]`
#: （`[0]` = embedding 出力・`[i+1]` = layer i の出力なので先頭から 22 番目 = layer 21 の出力）を
#: **グラフの最終出力**にするための層数で、参照実装の添字をこの 1 つの数へ解いてある。
SBV2_TEXT_ENCODER_LAYERS = 22

#: 配布 text_encoder のグラフ出力の本数。SBV2 が読むのは 1 本だけで、ランタイムは
#: `graph.outputs` を**全部** readback するため、全層出しのまま配ると毎 run で使わない
#: 22 本ぶんの staging + mapAsync を払う（ADR 0045 波 2 の実測 — T=512 で −10.6%）。
SBV2_TEXT_ENCODER_OUTPUTS = 1

#: 配布 text_encoder のグラフ入力の並び（`export_deberta.INPUT_ORDER` と同じ）。相対位置の
#: 添字表 2 本が**入力に居ること**が波 3 の成果そのもので、焼き込みへ戻ると 2MiB の死荷重が
#: 復活する（値は正しいままなので E2E では捕まらない）。
SBV2_TEXT_ENCODER_INPUTS: tuple[str, ...] = (
    "input_ids",
    "attention_mask",
    "c2p_pos",
    "p2c_pos",
)

#: initializer 名から encoder の層番号を拾う（`p_model_encoder_layer_<i>_...` — torch.export が
#: FQN を正規化した綴り）。層数の門はこれで数える。
SBV2_LAYER_PATTERN = re.compile(r"layer[._](\d+)[._]")

#: `sbv2_demo.py assets` が書くホスト資産の置き場と綴り。系列（IR + io）ではないので
#: `outputs/series/` の下ではない。
SBV2_DEMO_DIRNAME = "sbv2-demo"
SBV2_SYMBOLS_FILE = "symbols.json"
SBV2_TOKENIZER_FILE = "deberta-tokenizer.json"

#: 実重みと config の置き場（`export_sbv2.DEFAULT_MODEL_DIR` と同じ場所）。
SBV2_CONFIG_FILE = "config.json"
SBV2_STYLE_FILE = "style_vectors.npy"

#: 話者埋め込みの出所（ckpt のテンソルキー）。`front` / `voice` はどちらも `g[1,512,1]` を
#: グラフ入力に取るので、**この表が無いと配布形だけではグラフを実行できない**。
SBV2_SPEAKER_TENSOR = "emb_g.weight"

#: 配布する表のテンソルキー（`.npy` / ckpt を 1 テンソルの safetensors へ移すときの唯一のキー）。
SBV2_STYLE_KEY = "style_vectors"
SBV2_SPEAKER_KEY = "speaker_embeddings"

#: 出力の相対 path（**モデルサブツリー内**）— 配置表・変換先・manifest が共有する 1 箇所。
#: `style_vectors` / `speaker_embeddings` だけは配置ではなく変換の出力なので
#: {@link sbv2_placements} には現れない。
SBV2_OUTPUT_PATHS: Mapping[str, str] = {
    "text_encoder": "text_encoder/model.i8.safetensors",
    "front_f16": "front/model.f16.safetensors",
    "front_i8": "front/model.i8.safetensors",
    "voice_f16": "voice/model.f16.safetensors",
    "voice_i8": "voice/model.i8.safetensors",
    "tokenizer": "tokenizer/deberta-tokenizer.json",
    "symbols": "text/symbols.json",
    "style_vectors": "styles/style_vectors.safetensors",
    "speaker_embeddings": "speakers/speaker_embeddings.safetensors",
}

#: 格納 dtype の要求（Anima の {@link STORAGE_REQUIREMENTS} と同じ根拠 — 素の F32 資産が
#: 組み立て・ロード・実行を全て通って参照一致の門まで沈黙した実測事故）。`text_encoder` は
#: i8 系列 1 本だけを配るので I8 を要求する。tokenizer / symbols（JSON）と style_vectors
#: （こちらが書く F32）はここに載せない。
SBV2_STORAGE_REQUIREMENTS: Mapping[str, str] = {
    "text_encoder": "I8",
    "front_f16": "F16",
    "front_i8": "I8",
    "voice_f16": "F16",
    "voice_i8": "I8",
}

#: weights の宣言（dtype ラベル → 役割名）。`text_encoder` は i8 単体でも**dtype キー必須**の
#: 統一形で書く（ADR 0041 §3 — v1 の `{file}` / `{variants}` の 2 形は消えた）。
SBV2_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    "text_encoder": {"i8": WeightFiles("text_encoder")},
    "front": {"f16": WeightFiles("front_f16"), "i8": WeightFiles("front_i8")},
    "voice": {"f16": WeightFiles("voice_f16"), "i8": WeightFiles("voice_i8")},
}

#: assets の宣言（quant 選択に依存しない無条件ファイル）。
SBV2_ASSETS: Mapping[str, str] = {
    "tokenizer": "tokenizer",
    "symbols": "symbols",
    "style_vectors": "style_vectors",
    "speaker_embeddings": "speaker_embeddings",
}

#: quant 表。dtype が 1 つしかない `text_encoder` は書かない（{@link complete_quant_weights}
#: が完全写像へ埋める — hub は写像の完全性を実行時にも検査する）。
SBV2_QUANTS: Mapping[str, Any] = {
    "f16": {"weights": {"front": "f16", "voice": "f16"}, "session": {}},
    "w8": {"weights": {"front": "i8", "voice": "i8"}, "session": {}},
    "w8a8": {
        "weights": {"front": "i8", "voice": "i8"},
        "session": {"linearCompute": "i8a8"},
    },
}

SBV2_DEFAULT_QUANT = "w8"

#: `pipelineConfig.defaults` に載る実行時ノブ（`style_bert_vits2.constants` 由来）。綴りは
#: `symbols.json` の `defaults` と共有する — 同じ源から引いた同じ値が配布形の 2 つの資産に
#: 並ぶので、食い違いは組み立てで落とす（{@link sbv2_knob_defaults}）。
SBV2_KNOB_KEYS: tuple[str, ...] = (
    "style",
    "styleWeight",
    "sdpRatio",
    "noiseScale",
    "noiseScaleW",
    "lengthScale",
)


def sbv2_series_name(model: str) -> str:
    """系列名の幹（`outputs/series/<この名前>-{f16,i8}/`）。

    綴りは `export_sbv2.default_out_root` と同一 — 書き手と読み手が同じ 1 語から組む。
    """
    return f"{SBV2_SERIES_PREFIX}-{model}"


def sbv2_repo_name(model: str) -> str:
    """単一モデルの配布リポ名（`<DIST_ROOT>/<この名前>/` が 1 つの HF リポになる）。

    `karume-` を前置する（リポ名裁定 2026-08-09 — HF org を作らない代わりに配布リポは
    `karume-` prefix で名前空間を切る）。系列名（{@link sbv2_series_name}）には掛からない。
    """
    return f"karume-{SBV2_SERIES_PREFIX}-{model}"


@dataclass(frozen=True)
class Sbv2Sources:
    """組み立ての入力。系列 2 本のほかに、系列でない 2 つの置き場を跨ぐ。

    `demo` は `sbv2_demo.py assets` が書くホスト資産（`outputs/` 直下 — 系列ではない）、
    `model` は ckpt と `config.json` / `style_vectors.npy`（`inputs/` — 生成物ではない）。
    どちらも `--series` の下に無いので、系列の親から機械的に導けるのは前 3 つだけ。
    """

    series_f16: Path
    series_i8: Path
    text_encoder: Path
    demo: Path
    model: Path


def sbv2_sources(series_dir: Path, model: str = SBV2_DEFAULT_MODEL) -> Sbv2Sources:
    """系列の親ディレクトリ（`outputs/series/`）と `karume.paths` の綴りから入力を引く。"""
    return Sbv2Sources(
        series_f16=series_dir / f"{sbv2_series_name(model)}-f16",
        series_i8=series_dir / f"{sbv2_series_name(model)}-i8",
        text_encoder=series_dir / SBV2_TEXT_ENCODER_SERIES / SBV2_TEXT_ENCODER_VARIANT,
        demo=OUTPUTS_ROOT / SBV2_DEMO_DIRNAME,
        model=INPUTS_ROOT / SBV2_SERIES_PREFIX / model,
    )


def sbv2_placements(sources: Sbv2Sources) -> dict[str, Path]:
    """役割名 → 出所のファイル。出力の path は {@link SBV2_OUTPUT_PATHS} が持つ。

    この表に無いものは出力へ入らない（`io.*.safetensors` を落とす仕掛けはこれで足りる）。
    `style_vectors`（`.npy` → safetensors）と `speaker_embeddings`（ckpt の 1 テンソル →
    safetensors）は配置ではなく**変換**なのでここには現れない。
    """
    return {
        "text_encoder": sources.text_encoder / "model.safetensors",
        "front_f16": sources.series_f16 / "front" / "model.safetensors",
        "front_i8": sources.series_i8 / "front" / "model.safetensors",
        "voice_f16": sources.series_f16 / "voice" / "model.safetensors",
        "voice_i8": sources.series_i8 / "voice" / "model.safetensors",
        "tokenizer": sources.demo / SBV2_TOKENIZER_FILE,
        "symbols": sources.demo / SBV2_SYMBOLS_FILE,
    }


def sbv2_knob_defaults(symbols_path: Path) -> dict[str, Any]:
    """実行時ノブの既定を `style_bert_vits2.constants` から引く（定数を写経しない）。

    `sbv2_demo.jp_extra_rules` が `symbols.json` の `defaults` へ焼くのと**同じ源・同じ値**。
    配布形には両方が並ぶので、食い違いをここで落とす — TS 側は `symbols.json` からノブを
    読み（`parseJpExtraRules`）、hub 利用側は `karume.json` の `pipelineConfig.defaults` を
    読むため、ずれると「どちらの既定で鳴ったのか」が沈黙で分かれる。

    NOTE: `style_bert_vits2` は optional な `sbv2` dependency-group なので import は関数内。
    Anima の組み立てと `karume.dist` の import 自体はこの依存に触れない。
    """
    from style_bert_vits2.constants import (
        DEFAULT_LENGTH,
        DEFAULT_NOISE,
        DEFAULT_NOISEW,
        DEFAULT_SDP_RATIO,
        DEFAULT_STYLE,
        DEFAULT_STYLE_WEIGHT,
    )

    knobs: dict[str, Any] = {
        "style": DEFAULT_STYLE,
        "styleWeight": DEFAULT_STYLE_WEIGHT,
        "sdpRatio": DEFAULT_SDP_RATIO,
        "noiseScale": DEFAULT_NOISE,
        "noiseScaleW": DEFAULT_NOISEW,
        "lengthScale": DEFAULT_LENGTH,
    }
    if not symbols_path.is_file():
        raise DistError(f"組み立ての入力が無い: {symbols_path}")
    shipped = json.loads(symbols_path.read_text(encoding="utf-8")).get("defaults")
    if not isinstance(shipped, dict):
        raise DistError(f"{symbols_path}: 'defaults' 節が無い（実行時ノブの写しの正本）")
    disagreed = [
        f"{key}: constants={knobs[key]!r} / symbols.json={shipped.get(key)!r}"
        for key in SBV2_KNOB_KEYS
        if shipped.get(key) != knobs[key]
    ]
    if disagreed:
        raise DistError(
            f"{symbols_path} の defaults が style_bert_vits2 の定数と食い違う"
            f"（{', '.join(disagreed)}）— 資産を焼いたときと今の package が別版。"
            "`sbv2_demo.py assets` を採り直す"
        )
    return knobs


def assert_bert_hidden(text_encoder: Path, symbols_path: Path) -> None:
    """配布 text_encoder が「SBV2 が使う層の出力を 1 本だけ出す」形であることを検査する。

    正しい組み合わせは **22 層 × 出力 1 本 × `bertHiddenFromEnd` 1** の 1 通りしかないが、
    層数と出力形は `export_deberta.py` の variant が、取り出し位置は `sbv2_demo.py` の定数が
    持つ**別々の台本**なので、片方だけ動いた配布形が普通に組み上がってしまう。

    MUST: ずれても shape は合ったままロードも実行も通り、**別の層の BERT 特徴で音が出る**
    だけで沈黙する（スタイル表・話者表の行数門と同じ機序）。3 つを別々に見るのは、それぞれが
    別の取り違えを捕まえるため — 層数は「どの層の出力か」、出力本数は「検証用の全層出し資産が
    混ざっていないか」、位置は「symbols.json だけ古いか」。
    """
    header = safetensors_header(text_encoder)
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict) or IR_METADATA_KEY not in metadata:
        raise DistError(f"{text_encoder}: IR メタデータ（{IR_METADATA_KEY}）が無い")
    try:
        graph = json.loads(metadata[IR_METADATA_KEY])
    except json.JSONDecodeError as error:
        raise DistError(f"{text_encoder}: IR メタデータが JSON として読めない") from error
    if not isinstance(graph, dict):
        raise DistError(f"{text_encoder}: IR メタデータが最上位オブジェクトでない")

    initializers = graph.get("initializers")
    if not isinstance(initializers, dict) or not initializers:
        raise DistError(f"{text_encoder}: IR メタデータに非空の initializers が無い")
    layers = {match.group(1) for name in initializers if (match := SBV2_LAYER_PATTERN.search(name))}
    if len(layers) != SBV2_TEXT_ENCODER_LAYERS:
        raise DistError(
            f"{text_encoder} の encoder は {len(layers)} 層で、期待の"
            f" {SBV2_TEXT_ENCODER_LAYERS} 層でない — SBV2 が使う hidden_states[-3] を最終出力に"
            "するには 22 層で切り詰めた variant が要る（export_deberta.py の VARIANTS）"
        )

    outputs = graph.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != SBV2_TEXT_ENCODER_OUTPUTS:
        count = len(outputs) if isinstance(outputs, list) else outputs
        raise DistError(
            f"{text_encoder} のグラフ出力が {count} 本で、配布形が要求する"
            f" {SBV2_TEXT_ENCODER_OUTPUTS} 本でない — 全層出し（検証用）の資産が混ざっている"
        )

    inputs = graph.get("inputs")
    names = (
        tuple(item.get("name") for item in inputs if isinstance(item, dict))
        if isinstance(inputs, list)
        else ()
    )
    if names != SBV2_TEXT_ENCODER_INPUTS:
        raise DistError(
            f"{text_encoder} のグラフ入力が {list(names)} で、期待の"
            f" {list(SBV2_TEXT_ENCODER_INPUTS)} と違う — 相対位置の添字表が入力から外れると"
            "Tmax ぶんの定数（2MiB）が焼き戻る（値は正しいままなので E2E では捕まらない）"
        )

    shipped = json.loads(symbols_path.read_text(encoding="utf-8"))
    from_end = shipped.get("bertHiddenFromEnd") if isinstance(shipped, dict) else None
    if from_end != SBV2_TEXT_ENCODER_OUTPUTS:
        raise DistError(
            f"{symbols_path} の bertHiddenFromEnd={from_end!r} が、出力 1 本のグラフで唯一"
            f" 意味を持つ値 {SBV2_TEXT_ENCODER_OUTPUTS} でない — `sbv2_demo.py assets` を採り直す"
        )


def sbv2_config(model_dir: Path) -> Mapping[str, Any]:
    """`config.json` を読む（styles / speakers / 表の行数と列数の正本）。"""
    path = model_dir / SBV2_CONFIG_FILE
    if not path.is_file():
        raise DistError(f"組み立ての入力が無い: {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DistError(f"{path}: JSON として読めない") from error
    if not isinstance(config, dict):
        raise DistError(f"{path}: 最上位がオブジェクトでない")
    return config


def _sbv2_section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """`config.json` の 1 節（`data` / `model`）を検査して読む。"""
    section = config.get(name)
    if not isinstance(section, dict):
        raise DistError(f"config.json に '{name}' 節が無い（実際: {section!r}）")
    return section


def _sbv2_id_map(data: Mapping[str, Any], key: str) -> dict[str, int]:
    """`config.json` の `data.<key>`（名前 → ID の非空マップ）を検査して読む。"""
    table = data.get(key)
    if not isinstance(table, dict) or not table:
        raise DistError(f"config.json の data.{key} が非空のマップでない（実際: {table!r}）")
    for name, value in table.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise DistError(f"config.json の data.{key}[{name!r}] が整数でない（{value!r}）")
    return dict(table)


def _sbv2_int(section: Mapping[str, Any], key: str, where: str) -> int:
    """`config.json` の整数フィールド（表の行数 / 列数の宣言）を検査して読む。"""
    value = section.get(key)
    # bool は int の派生。`"n_speakers": true` を 1 として通すと表の行数門が緩む。
    if not isinstance(value, int) or isinstance(value, bool):
        raise DistError(f"config.json の {where}.{key} が整数でない（{value!r}）")
    return value


def sbv2_pipeline_config(config: Mapping[str, Any], knobs: Mapping[str, Any]) -> dict[str, Any]:
    """`pipelineConfig` を config.json と実行時ノブから組む（表を焼き込まない）。

    MUST: `styles` / `speakers` はハードコードしない。ckpt が変われば名前も並びも変わり
    （この FN4 は Neutral / high / low / NSFW の 4 つ、別の ckpt は Neutral / Angry / … の
    7 つ）、写した表を配ると **shape は合ったまま別のスタイルの声が出る**。`defaults` の
    数値も同じ理由で `style_bert_vits2` から引いた値（{@link sbv2_knob_defaults}）を受ける。

    `defaults.speaker` は `spk2id` の先頭キー（`sbv2_demo.resolve_style_and_speaker` と同式）。
    `speakers` の名前 → 行の解決先は配布形の `speaker_embeddings`
    （{@link sbv2_speaker_embeddings}）、`styles` の解決先は `style_vectors`。
    """
    data = _sbv2_section(config, "data")
    styles = _sbv2_id_map(data, "style2id")
    speakers = _sbv2_id_map(data, "spk2id")
    missing = [key for key in SBV2_KNOB_KEYS if key not in knobs]
    if missing:
        raise DistError(f"実行時ノブの既定が足りない: {missing}")
    if knobs["style"] not in styles:
        raise DistError(
            f"既定スタイル {knobs['style']!r} が config の style2id {sorted(styles)} に無い"
            " — 存在しないスタイル名を既定に据えた配布形は起動時にしか落ちない"
        )
    return {
        "styles": styles,
        "speakers": speakers,
        "defaults": {
            "speaker": next(iter(speakers)),
            **{key: knobs[key] for key in SBV2_KNOB_KEYS},
        },
    }


def sbv2_style_vectors(model_dir: Path, config: Mapping[str, Any]) -> np.ndarray:
    """`style_vectors.npy` を検査して f32 の `[スタイル数, 256]` として読む。

    MUST: 行数が `data.num_styles` と `len(data.style2id)` の**両方**に一致すること。
    スタイルの ID は行番号そのものなので、行と名前がずれてもロードも実行も通り、
    **別のスタイルの声が出る**だけで沈黙する（表の行数を合わせる以外に検出手段がない）。
    """
    data = _sbv2_section(config, "data")
    path = model_dir / SBV2_STYLE_FILE
    if not path.is_file():
        raise DistError(f"組み立ての入力が無い: {path}")
    table = np.load(path)
    if table.ndim != 2:
        raise DistError(f"{path}: 形 {table.shape} が [スタイル数, 256] でない")
    styles = _sbv2_id_map(data, "style2id")
    num_styles = _sbv2_int(data, "num_styles", "data")
    if table.shape[0] != num_styles or table.shape[0] != len(styles):
        raise DistError(
            f"{path}: 行数 {table.shape[0]} が config の num_styles {num_styles} /"
            f" style2id {len(styles)} 件と一致しない — スタイル ID は行番号なので、"
            "ずれたまま配ると別のスタイルの声が出る"
        )
    return np.ascontiguousarray(table, dtype=np.float32)


def sbv2_ckpt(model_dir: Path) -> Path:
    """実重みの ckpt。`*.safetensors` の一意存在を要求する（`export_sbv2.load_net_g` と同じ）。

    複数あると「どれから話者埋め込みを引いたか」が黙って変わる。
    """
    ckpts = sorted(model_dir.glob("*.safetensors"))
    if len(ckpts) != 1:
        raise DistError(
            f"{model_dir} の ckpt が一意でない（{len(ckpts)} 件: {[p.name for p in ckpts]}）"
        )
    return ckpts[0]


def sbv2_speaker_embeddings(model_dir: Path, config: Mapping[str, Any]) -> np.ndarray:
    """ckpt の `emb_g.weight` を f32 の `[話者数, gin_channels]` として引く。

    `front` / `voice` はどちらも話者埋め込み `g` を**グラフ入力**に取るので、この表が無いと
    配布形だけではグラフを実行できない（デモ経路は `assets.safetensors` に焼いていた）。
    `style_vectors` と同じく「表を配って実行時に行を引く」形にする。

    MUST: 行数が `data.n_speakers` と `len(data.spk2id)` の**両方**に一致すること —
    話者 ID は行番号そのものなので、ずれてもロードも実行も通り、**別の話者の声が出る**
    だけで沈黙する（スタイル表の行数門と同じ機序）。列数は `model.gin_channels` に一致
    すること（こちらは config から導出できる値なので shape ごと縛れる）。

    MUST: ckpt は 251MB 級。`safe_open` の遅延読みで**このテンソル 1 本だけ**を引く
    （`load_file` は全量を numpy へ展開する）。
    """
    data = _sbv2_section(config, "data")
    model = _sbv2_section(config, "model")
    speakers = _sbv2_id_map(data, "spk2id")
    num_speakers = _sbv2_int(data, "n_speakers", "data")
    gin_channels = _sbv2_int(model, "gin_channels", "model")
    ckpt = sbv2_ckpt(model_dir)
    with safe_open(str(ckpt), framework="np") as handle:
        # `keys()` はヘッダのテンソル名一覧（dict ではない）。`get_tensor` はそのテンソルの
        # バイト範囲だけを読むので、251MB の ckpt から 2KB を引くのにファイル全量は載らない。
        available = handle.keys()
        if SBV2_SPEAKER_TENSOR not in available:
            raise DistError(
                f"{ckpt} に {SBV2_SPEAKER_TENSOR} が無い — 話者埋め込みの出所が変わった"
            )
        table = handle.get_tensor(SBV2_SPEAKER_TENSOR)
    if table.ndim != 2:
        raise DistError(f"{ckpt}: {SBV2_SPEAKER_TENSOR} の形 {table.shape} が 2 次元でない")
    if table.shape[0] != num_speakers or table.shape[0] != len(speakers):
        raise DistError(
            f"{ckpt}: {SBV2_SPEAKER_TENSOR} の行数 {table.shape[0]} が config の"
            f" n_speakers {num_speakers} / spk2id {len(speakers)} 件と一致しない —"
            "話者 ID は行番号なので、ずれたまま配ると別の話者の声が出る"
        )
    if table.shape[1] != gin_channels:
        raise DistError(
            f"{ckpt}: {SBV2_SPEAKER_TENSOR} の列数 {table.shape[1]} が config の"
            f" gin_channels {gin_channels} と一致しない — グラフ入力 g の幅と食い違う"
        )
    return np.ascontiguousarray(table, dtype=np.float32)


def _table_payload(key: str, table: np.ndarray) -> bytes:
    """1 テンソルだけの safetensors のバイト列（`.npy` / ckpt から移した表の配布形）。"""
    return save({key: table})


def sbv2_plan(
    sources: Sbv2Sources, knobs: Mapping[str, Any], model: str = SBV2_DEFAULT_MODEL
) -> ModelPlan:
    """SBV2 1 モデルぶんの計画を組む（検査と表の読み取りをここで全部済ませる）。

    ノブの既定を引数で受けるのは、値の**出所**（`style_bert_vits2` の定数 — optional な
    dependency-group）と配布形の**組み立て**を分けるため。出所の解決は
    {@link sbv2_knob_defaults} が持つ。
    """
    assert_model_name(model)
    placements = sbv2_placements(sources)
    config = sbv2_config(sources.model)
    pipeline_config = sbv2_pipeline_config(config, knobs)
    style_vectors = sbv2_style_vectors(sources.model, config)
    speaker_embeddings = sbv2_speaker_embeddings(sources.model, config)
    for role, source in placements.items():
        assert_storage(role, source, SBV2_STORAGE_REQUIREMENTS)
    assert_bert_hidden(placements["text_encoder"], placements["symbols"])
    artifacts = {
        role: Artifact(SBV2_OUTPUT_PATHS[role], source=source)
        for role, source in placements.items()
    }
    artifacts["style_vectors"] = Artifact(
        SBV2_OUTPUT_PATHS["style_vectors"],
        payload=_table_payload(SBV2_STYLE_KEY, style_vectors),
    )
    artifacts["speaker_embeddings"] = Artifact(
        SBV2_OUTPUT_PATHS["speaker_embeddings"],
        payload=_table_payload(SBV2_SPEAKER_KEY, speaker_embeddings),
    )
    return ModelPlan(
        name=model,
        pipeline=SBV2_PIPELINE,
        artifacts=artifacts,
        weights=SBV2_WEIGHTS,
        assets=SBV2_ASSETS,
        quants=complete_quant_weights(SBV2_WEIGHTS, SBV2_QUANTS),
        default_quant=SBV2_DEFAULT_QUANT,
        pipeline_config=pipeline_config,
    )


# ---- ⑤ Irodori（text-to-speech）----------------------------------------------
#
# 配布するのは**実行に要る 8 グラフ + tokenizer 資産 1 本**だけ。8 のうち 2 本は波形 ↔ latent の
# コーデック（DACVAE — 上流では別リポ・別重みだが、テキストから音声まで 1 リポで完走させるため
# ここへ同梱する）。実行形ノブは持たない — 格納形は f32 の 1 本きりで、quant は `f32` の
# 1 席だけが並ぶ（`session` は空 = 低精度ノブなし）。
#
# `pipelineConfig` は 2 系統に割れる: **モデル固有の数**（条件 state の宣言長・話者行数・
# latent 幅・t_embed 幅）はチェックポイントの config から導出し、**実行時ノブ**（step 数・
# CFG の強さと区間・秒数の clamp）は上流 `SamplingRequest` の既定を定数として持つ。前者は
# 焼き込むと重みを差し替えたときにホストだけが古い数を持って沈黙誤値になるので、必ず導出する
# （TS 側の正本 `packages/models/src/irodori/config.ts` のモジュール doc と同じ理由）。

#: 既定のモデル名 — 系列（`outputs/series/irodori-<この名前>/`）と実重みの置き場
#: （`inputs/irodori/<この名前>/`）を束ねる 1 語。`irodori_tokenizer.default_out_dir` が
#: `--model-dir` のディレクトリ名から系列名を作るので、読み手のこちらも同じ 1 語から組む。
IRODORI_DEFAULT_MODEL = "v4-small"

#: 系列名とリポ名の接頭辞（`irodori-<モデル名>`）。
IRODORI_SERIES_PREFIX = "irodori"

#: パイプライン契約（ADR 0041 §2 — モデル単位）。TS 側の受理集合は
#: `IRODORI_PIPELINE_NAME` / `IRODORI_PIPELINE_MAJOR`。
IRODORI_PIPELINE = "irodori/1"

#: チェックポイント（`inputs/irodori/<モデル名>/`）のファイル名と、`__metadata__` が持つ
#: config の綴り。どちらも `export_irodori.py`（`MODEL_FILE` / `MODEL_CONFIG_META_KEY`）と同じ
#: — 重みが実際に構成されたときの形の正本はチェックポイントの中にある（HF から引き直さない）。
IRODORI_CKPT_FILE = "model.safetensors"
IRODORI_CONFIG_META_KEY = "config_json"

#: 役割名 → 系列のターゲットディレクトリ名（`export_irodori.TARGETS` の綴り）。役割名は
#: manifest の weights / assets キーでもあるので、ハイフン綴りの系列名とはここで縁を切る。
IRODORI_SERIES_DIRS: Mapping[str, str] = {
    "backbone": "backbone",
    "text_proj": "text-proj",
    "caption_proj": "caption-proj",
    "speaker": "speaker",
    "duration": "duration",
    "dit": "dit",
}

#: コーデック（DACVAE）の 1 語 — **別リポ・別重み**なので系列も入力素材も専用の名前を持つ
#: （`outputs/series/<この名前>/` に export 済みグラフ・`inputs/irodori/<この名前>/` に
#: `convert_dacvae.py` が書いた `metadata.json`）。Irodori のモデル名（`v4-small`）とは独立に
#: 動くので、`--model` の軸には乗せない。
IRODORI_CODEC_NAME = "dacvae-32dim"

#: 役割名 → コーデック系列のターゲットディレクトリ名（`export_dacvae.TARGETS` の綴り）。
IRODORI_CODEC_DIRS: Mapping[str, str] = {
    "codec_decoder": "decoder",
    "codec_encoder": "encoder",
}

#: グラフを持つ役割の全体（Irodori 本体 6 + コーデック 2）。
IRODORI_GRAPH_ROLES: tuple[str, ...] = (*IRODORI_SERIES_DIRS, *IRODORI_CODEC_DIRS)

#: `convert_dacvae.py` が書く構成ファイルと、そこから読む 2 つのキー。**sampleRate /
#: hopLength を直書きしない**ための出どころ（`export_dacvae.hop_length` と同じ式
#: — `hop_length = prod(encoder_rates)`）。
IRODORI_CODEC_METADATA_FILE = "metadata.json"
IRODORI_CODEC_SAMPLE_RATE_KEY = "sample_rate"
IRODORI_CODEC_RATES_KEY = "encoder_rates"

#: tokenizer 資産の出所（`irodori_tokenizer.py` が系列の下へ書く 4 ファイルのうち、配布へ入る
#: のは資産本体だけ — golden / nfkc 表は検証用で実行に要らない）。
IRODORI_TOKENIZER_DIR = "tokenizer"
IRODORI_TOKENIZER_FILE = "tokenizer.json"

#: 出力の相対 path（**モデルサブツリー内**）— 配置表と manifest が共有する 1 箇所。
IRODORI_OUTPUT_PATHS: Mapping[str, str] = {
    **{role: f"{role}/model.safetensors" for role in IRODORI_GRAPH_ROLES},
    "tokenizer": f"{IRODORI_TOKENIZER_DIR}/{IRODORI_TOKENIZER_FILE}",
}

#: 格納 dtype の要求（Anima / SBV2 と同じ根拠 — 素の F32 資産が組み立て・ロード・実行を全て
#: 通って参照一致の門まで沈黙した実測事故）。Irodori は f32 系列 1 本だけを配るので、8 グラフ
#: 全てに F32 を要求する（tokenizer は JSON なので載せない）。
IRODORI_STORAGE_REQUIREMENTS: Mapping[str, str] = dict.fromkeys(IRODORI_GRAPH_ROLES, "F32")

#: weights の宣言（dtype ラベル → 役割名）。1 dtype しか無い席も**dtype キー必須**の統一形
#: （ADR 0041 §3）。
IRODORI_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    role: {"f32": WeightFiles(role)} for role in IRODORI_GRAPH_ROLES
}

#: assets の宣言（quant 選択に依存しない無条件ファイル）。
IRODORI_ASSETS: Mapping[str, str] = {"tokenizer": "tokenizer"}

#: quant 表。dtype が 1 つしかない weights は書かない（{@link complete_quant_weights} が完全
#: 写像へ埋める）。実行形ノブがまだ 1 つも無い（f32 の素の経路だけ）ので `session` は空で、
#: 席は 1 つきり — 低精度の軸が生えたらここに並ぶ。
IRODORI_QUANTS: Mapping[str, Any] = {"f32": {"weights": {}, "session": {}}}

IRODORI_DEFAULT_QUANT = "f32"

#: DACVAE のフレームレート（Hz）— 48kHz / hop 1920 = 25。`export_irodori.CODEC_FRAME_RATE` と
#: 同値で、コーデックが別リポ・別重みなのでチェックポイントの config には入っていない。
#: `sampleRate` / `hopLength` はコーデックの `metadata.json` から導出し、3 者の整合
#: （`sampleRate == frameRate × hopLength`）は {@link irodori_pipeline_config} が見る。
IRODORI_FRAME_RATE = 25

#: codec decoder のタイル分割で採用区間の両側へ足す latent フレーム数。**受容野由来のモデル
#: 定数**（片側 13,793 サンプル = 7.19 フレーム → 8 フレーム = 15,360 サンプルで覆う）で、
#: `metadata.json` には入っていないのでここが唯一の出どころ。実測の根拠は decoder 主経路に
#: 因果層が無く全層が対称 pad か厳密 `L·stride` の convT である（= 平行移動同変）こと。
IRODORI_CODEC_HALO_FRAMES = 8

#: 発話長 clamp の秒数（上流 `SamplingRequest.min_seconds` / `max_seconds` の既定）。
#: `max_seconds` は **`dit` の記号次元 S の上限を決めた値でもある**
#: （`export_irodori.DIT_MAX_SECONDS` が同じ 30.0 を「実装側の正本は SamplingRequest の既定」
#: として持つ）。1 つの定数から両方を組むのは、配布形の clamp と焼かれたグラフの上限が
#: 独立に動くと「S は通るのに RoPE 表が足りない」形で実行時にしか出ないため。
IRODORI_MIN_SECONDS = 0.5
IRODORI_MAX_SECONDS = 30.0

#: 実行時ノブの既定（上流 `SamplingRequest` の同名フィールドと `rf.sample_euler_rf_cfg` の
#: `init_scale`）。**モデル固有ではない**のでチェックポイントの config には無く、ここが唯一の
#: 出どころになる（Anima の {@link ANIMA_PIPELINE_CONFIG} と同じ性格）。
#:
#: MUST: `speakerUncondMode` / `cfgGuidanceMode` は分岐用ではなく**宣言**（ADR 0047 決定 1）。
#: TS 側はこの 2 値以外を parse 時に拒否するので、別のモードで焼いた配布形は読まれる前に落ちる。
#: full-loop golden（`irodori_pipeline.py` の `meta.json`）も同じ値で焼かれており、golden 再生成
#: と dist 再生成がずれたら E2E 門（`e2e_irodori_latent_test.ts`）が実効値 drift として落とす。
IRODORI_SAMPLING_DEFAULTS: Mapping[str, Any] = {
    "steps": 40,
    "initScale": 0.999,
    "cfgMinT": 0.5,
    "cfgMaxT": 1.0,
    "cfgScales": {"text": 3.0, "speaker": 5.0, "caption": 3.0},
    "minSeconds": IRODORI_MIN_SECONDS,
    "maxSeconds": IRODORI_MAX_SECONDS,
    "speakerUncondMode": "mask",
    "cfgGuidanceMode": "independent",
}

#: 各グラフに要求する `(入力名の並び, 出力の本数)`。**入力の並びは実行時に位置で読まれる形の
#: 正本**で、出力本数は「検証用の別資産が混ざっていないか」を見る席（SBV2 の
#: {@link assert_bert_hidden} と同じ機序 — `caption_proj` が 1 出力の資産に差し替わると
#: `caption_vec` を第 1 出力から採る別のベクトルで duration が回り、shape は合ったまま沈黙する）。
IRODORI_GRAPH_SHAPES: Mapping[str, tuple[tuple[str, ...], int]] = {
    "backbone": (("input_ids",), 1),
    "text_proj": (("hidden",), 1),
    "caption_proj": (("hidden",), 2),
    "speaker": (("latent",), 1),
    "duration": (
        ("text_state", "speaker_vec", "has_speaker", "caption_vec", "has_caption"),
        1,
    ),
    "dit": (("x_t", "t_embed", "mask", "text_state", "speaker_state", "caption_state"), 1),
    # コーデック 2 本は位置表もマスクも持たない純畳み込み網（入力 1 本 / 出力 1 本）。
    "codec_decoder": (("latent",), 1),
    "codec_encoder": (("wav",), 1),
}

#: `(役割, グラフ入力, 軸, pipelineConfig の欄)` — グラフの**静的**次元と宣言の突合表。
#: TS 側 `IrodoriPipeline.fromAssets` の `assertStaticDim` と同じ組み合わせを**焼く側でも**
#: 見る（配ってから利用者の手元で初めて落ちる形にしない）。
IRODORI_STATIC_DIMS: tuple[tuple[str, str, int, str], ...] = (
    ("dit", "x_t", 2, "latentDim"),
    ("dit", "t_embed", 1, "timestepEmbedDim"),
    ("dit", "text_state", 1, "maxTextLen"),
    ("dit", "text_state", 2, "textDim"),
    ("dit", "speaker_state", 1, "speakerRows"),
    ("dit", "speaker_state", 2, "speakerDim"),
    ("dit", "caption_state", 1, "maxCaptionLen"),
    ("dit", "caption_state", 2, "captionDim"),
    ("duration", "text_state", 2, "textDim"),
    ("duration", "speaker_vec", 1, "speakerDim"),
    ("duration", "caption_vec", 1, "captionDim"),
    # コーデックは**別リポ・別重み**なので、latent の幅と 1 フレームのサンプル数が Irodori 側の
    # 宣言と噛み合っている保証がここにしか無い（別次元の DACVAE を混ぜると shape は合ったまま
    # 別の声になる / 波形長だけが静かにずれる）。
    ("codec_decoder", "latent", 2, "latentDim"),
    ("codec_encoder", "wav", 2, "hopLength"),
)


def irodori_series_name(model: str) -> str:
    """系列名（`outputs/series/<この名前>/`）— `irodori_tokenizer.default_out_dir` と同じ綴り。"""
    return f"{IRODORI_SERIES_PREFIX}-{model}"


def irodori_repo_name(model: str) -> str:
    """単一モデルの配布リポ名（`karume-` prefix はリポ名裁定 2026-08-09）。"""
    return f"karume-{IRODORI_SERIES_PREFIX}-{model}"


@dataclass(frozen=True)
class IrodoriSources:
    """組み立ての入力。系列とチェックポイントの置き場（`inputs/` — 生成物ではない）が 2 組。

    チェックポイント側が要るのは `pipelineConfig` の数を**焼き込まずに導出**するため
    （`__metadata__` / `metadata.json` だけを読むので 2.9GB のペイロードは舐めない）。
    コーデックが別の 2 本を持つのは**別リポ・別重み**だから — Irodori のモデル名を動かしても
    コーデックは動かない。
    """

    series: Path
    model: Path
    codec_series: Path
    codec_model: Path


def irodori_sources(series_dir: Path, model: str = IRODORI_DEFAULT_MODEL) -> IrodoriSources:
    """系列の親ディレクトリ（`outputs/series/`）と `karume.paths` の綴りから入力を引く。"""
    return IrodoriSources(
        series=series_dir / irodori_series_name(model),
        model=INPUTS_ROOT / IRODORI_SERIES_PREFIX / model,
        codec_series=series_dir / IRODORI_CODEC_NAME,
        codec_model=INPUTS_ROOT / IRODORI_SERIES_PREFIX / IRODORI_CODEC_NAME,
    )


def irodori_placements(sources: IrodoriSources) -> dict[str, Path]:
    """役割名 → 出所のファイル。出力の path は {@link IRODORI_OUTPUT_PATHS} が持つ。

    この表に無いものは出力へ入らない（`io.*.safetensors` と tokenizer の golden 3 本は
    これで落ちる）。
    """
    return {
        **{
            role: sources.series / directory / "model.safetensors"
            for role, directory in IRODORI_SERIES_DIRS.items()
        },
        **{
            role: sources.codec_series / directory / "model.safetensors"
            for role, directory in IRODORI_CODEC_DIRS.items()
        },
        "tokenizer": sources.series / IRODORI_TOKENIZER_DIR / IRODORI_TOKENIZER_FILE,
    }


def irodori_model_config(model_dir: Path) -> Mapping[str, Any]:
    """チェックポイントの `__metadata__` から `config_json` を読む（ヘッダだけ読む）。

    `export_irodori.read_configs` と同じ出どころ・同じ理由（HF から config を引き直さない）。
    こちらが torch を経由しないのは、組み立てが要るのが JSON 1 本だけだから。
    """
    path = model_dir / IRODORI_CKPT_FILE
    if not path.is_file():
        raise DistError(f"組み立ての入力が無い: {path}")
    metadata = safetensors_header(path).get("__metadata__")
    if not isinstance(metadata, dict) or IRODORI_CONFIG_META_KEY not in metadata:
        raise DistError(f"{path} の __metadata__ に '{IRODORI_CONFIG_META_KEY}' が無い")
    try:
        config = json.loads(metadata[IRODORI_CONFIG_META_KEY])
    except json.JSONDecodeError as error:
        raise DistError(f"{path}: {IRODORI_CONFIG_META_KEY} が JSON として読めない") from error
    if not isinstance(config, dict):
        raise DistError(f"{path}: {IRODORI_CONFIG_META_KEY} が最上位オブジェクトでない")
    return config


def _irodori_int(config: Mapping[str, Any], key: str) -> int:
    """チェックポイント config の整数フィールド（宣言長 / 幅）を検査して読む。"""
    value = config.get(key)
    # bool は int の派生。`"latent_dim": true` を 1 として通すと幅の突合が緩む。
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise DistError(f"{IRODORI_CONFIG_META_KEY} の {key} が正の整数でない（{value!r}）")
    return value


def _irodori_float(config: Mapping[str, Any], key: str) -> float:
    """チェックポイント config の実数フィールド（秒数）を検査して読む。"""
    value = config.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
        raise DistError(f"{IRODORI_CONFIG_META_KEY} の {key} が正の数でない（{value!r}）")
    return float(value)


class IrodoriCodecNumbers(NamedTuple):
    """コーデックの `metadata.json` から導く 2 つ（秒 ↔ サンプル ↔ フレームの換算）。"""

    sample_rate: int
    hop_length: int


def irodori_codec_numbers(codec_model_dir: Path) -> IrodoriCodecNumbers:
    """`convert_dacvae.py` が書いた `metadata.json` から `sampleRate` / `hopLength` を導く。

    MUST: 写経しない — 別次元・別 hop の DACVAE へ差し替えたときに、ホストだけが古い換算を
    持ったまま「それらしい長さの音声」を出す（例外は出ない）。`hop_length` は
    `export_dacvae.hop_length` と同じ式（`prod(encoder_rates)` — `DACVAE.__init__` の綴り）。
    """
    path = codec_model_dir / IRODORI_CODEC_METADATA_FILE
    if not path.is_file():
        raise DistError(f"組み立ての入力が無い: {path}（`uv run python convert_dacvae.py` で作る）")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DistError(f"{path} が JSON として読めない") from error
    kwargs = raw.get("kwargs") if isinstance(raw, dict) else None
    if not isinstance(kwargs, dict):
        raise DistError(f"{path} に 'kwargs' が無い")
    sample_rate = kwargs.get(IRODORI_CODEC_SAMPLE_RATE_KEY)
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise DistError(f"{path} の {IRODORI_CODEC_SAMPLE_RATE_KEY} が正の整数でない")
    rates = kwargs.get(IRODORI_CODEC_RATES_KEY)
    if not isinstance(rates, list) or not rates:
        raise DistError(f"{path} の {IRODORI_CODEC_RATES_KEY} が非空のリストでない")
    hop_length = 1
    for rate in rates:
        if not isinstance(rate, int) or isinstance(rate, bool) or rate <= 0:
            raise DistError(f"{path} の {IRODORI_CODEC_RATES_KEY} に正の整数でない要素がある")
        hop_length *= rate
    return IrodoriCodecNumbers(sample_rate=sample_rate, hop_length=hop_length)


def irodori_pipeline_config(
    config: Mapping[str, Any], codec: IrodoriCodecNumbers
) -> dict[str, Any]:
    """`pipelineConfig`（TS 側スキーマの 23 欄）をチェックポイント config から組む。

    MUST: モデル固有の数を写経しない — 重みを差し替えたときにホストだけが古い数を持つと、
    右 pad も行数計算もそのまま通って（shape は合う）**別の位置の条件を読んだ**結果が沈黙で出る。
    欄名と値域の正本は `packages/models/src/irodori/config.ts`。

    `latentDim` が 2 つの役割（`dit` の `x_t` の幅 = `latent_dim × latent_patch_size` と、
    参照 latent の 1 フレームの幅 = `latent_dim`）を兼ねているので、両者が一致する
    `latent_patch_size == 1` でなければ組めない — 兼ねられなくなったら TS 側の欄を割る話に
    なるので、黙って片方を選ばずここで落とす。
    """
    latent_patch = _irodori_int(config, "latent_patch_size")
    if latent_patch != 1:
        raise DistError(
            f"latent_patch_size が {latent_patch} — pipelineConfig の latentDim は"
            " `dit` の x_t 幅と参照 latent の 1 フレーム幅を兼ねており、1 でなければ"
            "両者が一致しない（TS 側の欄を割る変更が要る）"
        )
    # 秒 → フレーム（`frameRate`）と 秒 → サンプル → フレーム（コーデック由来の 2 値）が
    # 独立に動く形を作らない。TS 側も parse 時に同じ式を見るが、**配ってから落ちる**のを
    # 避けるためにここでも見る（別 hop のコーデックを混ぜた瞬間に気づける席）。
    if codec.sample_rate != IRODORI_FRAME_RATE * codec.hop_length:
        raise DistError(
            f"コーデックの sample_rate {codec.sample_rate} が frameRate {IRODORI_FRAME_RATE}"
            f" × hop_length {codec.hop_length} と違う"
        )
    frames = int(_irodori_float(config, "ref_max_seconds") * IRODORI_FRAME_RATE)
    speaker_patch = _irodori_int(config, "speaker_patch_size")
    return {
        "maxTextLen": _irodori_int(config, "max_text_len"),
        "maxCaptionLen": _irodori_int(config, "max_caption_len"),
        # 参照 latent の patch 後の上限（`export_irodori.speaker_sym_max`）+ 平均トークン 1 本。
        "speakerRows": frames // speaker_patch + 1,
        # 生成できる latent の上限（`export_irodori.dit_sym_max` と同じ式）。
        "ditSymMax": int(IRODORI_MAX_SECONDS * IRODORI_FRAME_RATE) // latent_patch,
        "frameRate": IRODORI_FRAME_RATE,
        "sampleRate": codec.sample_rate,
        "hopLength": codec.hop_length,
        "codecHaloFrames": IRODORI_CODEC_HALO_FRAMES,
        "latentDim": _irodori_int(config, "latent_dim"),
        "speakerPatchSize": speaker_patch,
        "speakerDim": _irodori_int(config, "speaker_dim"),
        "textDim": _irodori_int(config, "text_dim"),
        "captionDim": _irodori_int(config, "caption_dim"),
        "timestepEmbedDim": _irodori_int(config, "timestep_embed_dim"),
        **IRODORI_SAMPLING_DEFAULTS,
    }


def ir_graph(path: Path) -> Mapping[str, Any]:
    """コンテナの `__metadata__` から IR グラフの JSON を読む（ヘッダだけ読む）。"""
    metadata = safetensors_header(path).get("__metadata__")
    if not isinstance(metadata, dict) or IR_METADATA_KEY not in metadata:
        raise DistError(f"{path}: IR メタデータ（{IR_METADATA_KEY}）が無い")
    try:
        graph = json.loads(metadata[IR_METADATA_KEY])
    except json.JSONDecodeError as error:
        raise DistError(f"{path}: IR メタデータが JSON として読めない") from error
    if not isinstance(graph, dict):
        raise DistError(f"{path}: IR メタデータが最上位オブジェクトでない")
    return graph


def _graph_inputs(graph: Mapping[str, Any], path: Path) -> dict[str, list[Any]]:
    """グラフ入力の `{名前: 形}`（並びの検査は呼び出し側 — ここは引くための表）。"""
    inputs = graph.get("inputs")
    if not isinstance(inputs, list):
        raise DistError(f"{path}: IR メタデータに inputs が無い")
    return {
        item["name"]: item["shape"]
        for item in inputs
        if isinstance(item, dict) and isinstance(item.get("shape"), list)
    }


def assert_irodori_graphs(
    placements: Mapping[str, Path], pipeline_config: Mapping[str, Any]
) -> None:
    """8 グラフが**読み出せて**、入力の並び・出力本数・静的次元が宣言どおりであることを見る。

    MUST: ずれても shape は合ったままロードも実行も通る組み合わせがある（`caption_proj` の
    出力本数・条件 state の宣言長・`speaker` の patch 幅）ので、配布形を並べる前にここで落とす。
    SBV2 の {@link assert_bert_hidden} と同じ規律 — 別々の台本（`export_irodori.py` の
    ターゲットと、この manifest）が持つ数を突き合わせる席がここしか無い。
    """
    graphs = {role: ir_graph(placements[role]) for role in IRODORI_GRAPH_ROLES}
    for role, (expected_inputs, expected_outputs) in IRODORI_GRAPH_SHAPES.items():
        path = placements[role]
        graph = graphs[role]
        names = tuple(_graph_inputs(graph, path))
        if names != expected_inputs:
            raise DistError(
                f"{path} のグラフ入力が {list(names)} で、期待の {list(expected_inputs)} と違う"
                " — 実行側は名前で束ねるので、1 つでも綴りが変われば束ねられない"
                "（並びまで見るのは export 側の宣言順が動いていないことの証跡）"
            )
        outputs = graph.get("outputs")
        count = len(outputs) if isinstance(outputs, list) else outputs
        if count != expected_outputs:
            raise DistError(
                f"{path} のグラフ出力が {count} 本で、配布形が要求する {expected_outputs} 本で"
                "ない — 別のターゲットの資産が混ざっている"
            )
    for role, name, axis, field_name in IRODORI_STATIC_DIMS:
        declared = _graph_inputs(graphs[role], placements[role])[name][axis]
        expected = pipeline_config[field_name]
        if declared != expected:
            raise DistError(
                f"{placements[role]} の入力 '{name}' の軸 {axis} が {declared!r}、"
                f"pipelineConfig の {field_name} は {expected} — チェックポイントの config と"
                "焼かれたグラフが別の版"
            )
    # 参照 latent は patch してから `speaker` へ渡す（ADR 0047 決定 4）ので、入力幅は 2 欄の積。
    patched = pipeline_config["latentDim"] * pipeline_config["speakerPatchSize"]
    width = _graph_inputs(graphs["speaker"], placements["speaker"])["latent"][2]
    if width != patched:
        raise DistError(
            f"{placements['speaker']} の入力 'latent' の軸 2 が {width!r}、pipelineConfig の"
            f" latentDim × speakerPatchSize は {patched}"
        )
    # `dit` の `mask` は「latent S + 条件 3 区間」の長さで宣言される（ADR 0046 の派生次元）。
    # 区間の合計がずれると、マスクの区間割りだけが黙って別の位置を指す。
    symbols = graphs["dit"].get("symbols")
    if not isinstance(symbols, list) or len(symbols) != 1:
        raise DistError(f"{placements['dit']}: 記号次元が 1 本でない（{symbols!r}）")
    total = sum(
        pipeline_config[field_name] for field_name in ("maxTextLen", "speakerRows", "maxCaptionLen")
    )
    declared_mask = _graph_inputs(graphs["dit"], placements["dit"])["mask"][3]
    if declared_mask != f"{symbols[0]}+{total}":
        raise DistError(
            f"{placements['dit']} の入力 'mask' の軸 3 が {declared_mask!r}、pipelineConfig の"
            f" 条件 3 区間の合計は {total}（期待 '{symbols[0]}+{total}'）"
        )


def irodori_plan(sources: IrodoriSources, model: str = IRODORI_DEFAULT_MODEL) -> ModelPlan:
    """Irodori 1 モデルぶんの計画を組む（検査と config の読み取りをここで全部済ませる）。"""
    assert_model_name(model)
    placements = irodori_placements(sources)
    pipeline_config = irodori_pipeline_config(
        irodori_model_config(sources.model), irodori_codec_numbers(sources.codec_model)
    )
    for role, source in placements.items():
        assert_storage(role, source, IRODORI_STORAGE_REQUIREMENTS)
    assert_irodori_graphs(placements, pipeline_config)
    return ModelPlan(
        name=model,
        pipeline=IRODORI_PIPELINE,
        artifacts={
            role: Artifact(IRODORI_OUTPUT_PATHS[role], source=source)
            for role, source in placements.items()
        },
        weights=IRODORI_WEIGHTS,
        assets=IRODORI_ASSETS,
        quants=complete_quant_weights(IRODORI_WEIGHTS, IRODORI_QUANTS),
        default_quant=IRODORI_DEFAULT_QUANT,
        pipeline_config=pipeline_config,
    )


# ---- ⑥ pipeline 別ディスパッチと CLI -----------------------------------------


#: モデルカードの描き手（manifest とリポ ID から本文 1 枚）。
CardRenderer = Callable[[Mapping[str, Any], str], str]


@dataclass(frozen=True)
class Pipeline:
    """pipeline ごとに違うものだけを持つディスパッチ表の 1 行。

    共有部（配置・共有の畳み込み・sha256・宣言と現物の突合）は上の汎用関数が持つので、
    ここに並ぶのは「既定のモデル名 / 単一モデルの既定リポ名 / 系列とモデル名から計画を組む
    手順 / モデルカードの描き手」の 4 つだけ。
    """

    default_model: str
    #: 単一モデルのときの既定の出力ディレクトリ名（複数モデルのリポ名は導出できない）。
    repo_name: Callable[[str], str]
    plan: Callable[[Path, str], ModelPlan]
    #: モデルカードの**帰属プロファイル**（名前 → 描き手）。テンプレートは pipeline 固有でも、
    #: 帰属（出所・ライセンス・引用）は声のファミリーごとに別の事実になるため 1 対 1 ではない。
    card_profiles: Mapping[str, CardRenderer]


def resolve_card_renderer(pipeline: Pipeline, profile: str | None) -> CardRenderer:
    """`--card-profile` を描き手へ解決する。

    MUST: 選択肢が 2 つ以上あるとき、省略は**選択肢を並べて落とす** — 既定を黙って選ぶと、
    新しいファミリーのリポへ前のファミリーの帰属がそのまま描かれる。表も使い方も正しいまま
    帰属だけが誤るので、配ってからでないと誰も気づけない。1 つしかない pipeline は選びようが
    ないので省略で通す（2 つ目が生えた瞬間に、この規則が自動で明示を要求しはじめる）。
    """
    choices = ", ".join(sorted(pipeline.card_profiles))
    if profile is None:
        if len(pipeline.card_profiles) != 1:
            raise DistError(
                f"--card-profile が要る（選択肢: {choices}）— モデルカードの帰属は"
                "ファミリーごとに違う事実なので、既定を黙って選ばない"
            )
        return next(iter(pipeline.card_profiles.values()))
    renderer = pipeline.card_profiles.get(profile)
    if renderer is None:
        raise DistError(f"帰属プロファイル '{profile}' は無い（選択肢: {choices}）")
    return renderer


def anima_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """`--series` の親から Anima 1 モデルの計画を組む（CLI のディスパッチ先）。"""
    return anima_plan(anima_sources(series_dir, model), model)


def sbv2_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """`--series` の親から SBV2 1 モデルの計画を組む（CLI のディスパッチ先）。"""
    sources = sbv2_sources(series_dir, model)
    return sbv2_plan(sources, sbv2_knob_defaults(sources.demo / SBV2_SYMBOLS_FILE), model)


def irodori_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """`--series` の親から Irodori 1 モデルの計画を組む（CLI のディスパッチ先）。"""
    return irodori_plan(irodori_sources(series_dir, model), model)


PIPELINES: Mapping[str, Pipeline] = {
    "anima": Pipeline(
        default_model=ANIMA_MODEL_NAME,
        # `karume-` prefix はリポ名裁定（2026-08-09）— HF org の代わりの名前空間。
        repo_name=lambda model: f"karume-{model}",
        plan=anima_dist_plan,
        # 帰属は 1 通りだけ（LoRA を焼いた base 1 本）— 選択肢が無いので省略で通る。
        card_profiles={"anima": render_model_card},
    ),
    "sbv2": Pipeline(
        default_model=SBV2_DEFAULT_MODEL,
        repo_name=sbv2_repo_name,
        plan=sbv2_dist_plan,
        card_profiles={
            name: partial(render_sbv2_model_card, profile=profile)
            for name, profile in SBV2_CARD_PROFILES.items()
        },
    ),
    "irodori": Pipeline(
        default_model=IRODORI_DEFAULT_MODEL,
        repo_name=irodori_repo_name,
        plan=irodori_dist_plan,
        # 帰属は 1 通りだけ（上流 1 リポの重みを格納形へ落とし直したもの）— 選択肢が無いので
        # 省略で通る。2 つ目のファミリーが生えた瞬間に明示が要求されはじめる。
        card_profiles={"irodori": render_irodori_model_card},
    ),
}

DEFAULT_PIPELINE = "anima"


def default_out_dir(pipeline: Pipeline, models: Sequence[str]) -> Path:
    """`--out` 省略時の出力先。

    複数モデルのリポ名は**導出できない**（`karume-sbv2-jvnv` のようなファミリー名は命名の
    決定であって、モデル名の並びからは決まらない）ので、明示を求めて落とす。
    """
    if len(models) != 1:
        raise DistError(
            f"モデルを {len(models)} 個組む場合はリポ名を導出できない — --out で出力先を指定する"
        )
    return DIST_ROOT / pipeline.repo_name(models[0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="配布ディレクトリ（HF アップ可能形）の組み立て")
    parser.add_argument(
        "--pipeline",
        choices=sorted(PIPELINES),
        default=DEFAULT_PIPELINE,
        help=f"組み立てるパイプライン（既定: {DEFAULT_PIPELINE}）",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        metavar="NAME",
        help="組み立てるモデル名（繰り返すと 1 リポへまとめて組む = ファミリー組み立て。"
        "最初の 1 つが defaultModel。既定: "
        + " / ".join(f"{name}={spec.default_model}" for name, spec in PIPELINES.items())
        + "）",
    )
    parser.add_argument(
        "--card-profile",
        dest="card_profile",
        metavar="NAME",
        default=None,
        help="モデルカードの帰属プロファイル（出所・ライセンス・引用）。選択肢が 2 つ以上ある"
        " pipeline では必須: "
        + " / ".join(
            f"{name}={'|'.join(sorted(spec.card_profiles))}" for name, spec in PIPELINES.items()
        ),
    )
    parser.add_argument(
        "--series",
        type=Path,
        default=SERIES_ROOT,
        help="系列ディレクトリ群の親（既定: リポの outputs/series/）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="出力先（既定は単一モデルのときだけ models/<リポ名>/ — 1 ディレクトリ = 1 HF リポ。"
        "複数モデルを組むときは必須）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    pipeline = PIPELINES[args.pipeline]
    models = args.models if args.models else [pipeline.default_model]
    out_dir = args.out if args.out is not None else default_out_dir(pipeline, models)
    # 帰属プロファイルは**組み立ての前**に解決する — 誤った / 足りない指定で数 GB を並べてから
    # 最後の 1 枚で落ちる形にしない。
    render_card = resolve_card_renderer(pipeline, args.card_profile)
    # MUST: 全モデルの計画（= 検査と読み取り）を配置の**前**に済ませる — 2 モデル目で落ちる
    # 形でも、1 モデル目だけ入った配布形を後段に見せない。
    plans = [pipeline.plan(args.series, model) for model in models]
    manifest = assemble_family(plans, out_dir, models[0])
    verified = verify_dist(out_dir)
    # モデルカードは**検証を通った manifest** から描く（表と現物が食い違ったまま説明だけ
    # 生えることがない順序）。リポ ID は組み立て先のディレクトリ名から導く — manifest は
    # 自分の在り処を知らず、ファミリーリポの名前は pipeline の定数にもできない。
    card = render_card(manifest, f"{HF_OWNER}/{out_dir.name}")
    (out_dir / MODEL_CARD_FILENAME).write_text(card, encoding="utf-8")
    for rel_path, size in sorted(verified.items()):
        print(f"{size:>12}  {rel_path}")
    for rel_path in sorted(META_PATHS):
        meta = out_dir / rel_path
        if meta.is_file():
            print(f"{meta.stat().st_size:>12}  {rel_path}")
    listing = ", ".join(
        f"{name}({model['defaultQuant']})" for name, model in manifest["models"].items()
    )
    print(
        f"[dist] {out_dir} — {manifest['generator']} /"
        f" models {listing} / default {manifest['defaultModel']}"
    )


if __name__ == "__main__":
    main()
