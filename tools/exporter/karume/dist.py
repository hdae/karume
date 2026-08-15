"""配布ディレクトリの組み立て — 系列ディレクトリ群 → HF へそのまま上げられる 1 リポ形。

仕様の正本は ADR 0041（`docs/decisions/0041-manifest-v2.md`）。ここが作るのは §2 の形で
並んだファイル群と、それを宣言する `karume.json`（`karume/2`）、そして manifest から機械導出
したモデルカード `README.md`（ADR 0037 §3 の「そのまま HF リポとして上げられる形」）。

**リポ内レイアウトは一律「モデル別サブツリー + `shared/` + 直下 `karume.json` / `README.md`」**
（ADR 0041 §9）。単一モデルのリポも同じ規則で `<モデル名>/…` の下へ入る — 1 モデルだけ平置き
という例外を作ると、2 個目のモデルを足した瞬間に既存 path が全部動く。

**pipeline 別のディスパッチ**（`--pipeline`）と**モデル別の軸**（`--model`）。
共有するのは「共有席を決める / 置く / sha256 を採る / 宣言と現物を突き合わせる」層だけで、
どのファイルをどの名前で並べ何を宣言するかは pipeline ごとの表が持つ（{@link Pipeline}）。

MUST: **受理集合は呼び出し側が渡す**（{@link main} の `pipelines`）— core が持つ
{@link PIPELINES} は「core wheel だけで組める pipeline」であって全量ではない。リポ専用の
recipe（`tools/export-recipes/<family>/`）は wheel の外にあるので、リポの dist ドライバ
（`tools/export-recipes/dist.py`）が core の表へ合成して渡す。既定 pipeline を core が
持たないのも同じ理由 — 表が呼び出し側で変わる以上、既定を焼くと「渡していない pipeline を
既定で組む」形が生まれる（ADR 0065 決定 2）。

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

    uv run karume dist --pipeline siglip2                 # = uv run python -m karume.dist
    uv run karume dist --pipeline siglip2 --model so400m  # → models/karume-siglip2-so400m/
    uv run karume dist --pipeline birefnet --model lucida  # → models/karume-lucida/
    uv run karume dist --pipeline depth-anything          # → models/karume-depth-anything-v2-small/

`--card-profile` はモデルカードの**帰属**（出所・ライセンス・引用）の選択で、選択肢が 2 つ
以上ある pipeline では必須（{@link resolve_card_renderer}）。組み立てる資産には掛からない —
同じ重みでも、どのファミリーとして配るかで帰属が変わる。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from safetensors.numpy import save

from karume.ir import IR_METADATA_KEY
from karume.modelcard import (
    BIREFNET_UPSTREAM,
    DEPTH_ANYTHING_UPSTREAM,
    HF_OWNER,
    SIGLIP2_UPSTREAM,
    render_birefnet_model_card,
    render_depth_anything_model_card,
    render_siglip2_model_card,
    render_vowel_detector_model_card,
)
from karume.paths import DIST_ROOT, INPUTS_ROOT, SERIES_ROOT

# ---- ① 共有部: 置き場の綴り・共有席の決定・配置・ハッシュ・宣言と現物の突合 -------

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

#: 組み立て中のツリー（staging）と、差し替え直前まで残す旧ツリーの置き場 — どちらも**出力先と
#: 同じ親**に `<出力先の名前><接尾辞>` で作る。同じ親なのは rename が同一ファイルシステム内で
#: しか原子的でないため（`/tmp` などへ逃がすと swap が跨デバイスコピーへ落ち、原子性ごと消える）。
STAGING_SUFFIX = ".staging"
SUPERSEDED_SUFFIX = ".old"

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


def assert_storage_absent(role: str, path: Path, forbidden: Mapping[str, tuple[str, ...]]) -> None:
    """役割が**持ってはならない**格納 dtype がヘッダに無いことを検査する（無関係な役割は素通し）。

    {@link assert_storage} の存在検査だけでは **f32 席に圧縮系列の資産を挿し込む取り違えが
    素通りする** — 圧縮系列のコンテナは適格外の重み（bias / norm / グラフ定数・i8 なら
    per-channel scale も）を F32 で持つので、「F32 を含む」は f16 / i8 資産でも真になる。
    片方向の存在検査を両側から挟んで初めて「系列 × 格納 dtype」が集合として一意に決まる
    （ADR 0027 / 0029 の検出限界 — **系列 root の取り違えは数値網では原理的に検出できない**
    ので、ここが唯一の検出器）。

    MUST: 禁止は**役割ごとに集合**で持つ（1 つだけだと圧縮系列が 2 本以上あるときに、名指し
    しなかったほうが黙って素通りする）。逆向き（圧縮席に f32 資産）は {@link assert_storage}
    が要求 dtype の不在で落とすので、禁止表は素の席にだけ要る。
    """
    banned = forbidden.get(role)
    if not banned:
        return
    if not path.is_file():
        raise DistError(f"組み立ての入力が無い: {path}")
    found = storage_dtypes(path)
    intruders = [dtype for dtype in banned if dtype in found]
    if intruders:
        raise DistError(
            f"{role}: {path} の格納 dtype に {' / '.join(intruders)} がある"
            f"（実際: {sorted(found)}）。"
            f"素の f32 系列を指すべき席に圧縮系列の資産が混ざっている"
            "（系列 root の取り違え — 数値の門では検出できないのでここで落とす）"
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


def table_payload(key: str, table: np.ndarray) -> bytes:
    """1 テンソルだけの safetensors のバイト列（`.npy` / ckpt から移した表の配布形）。"""
    return save({key: table})


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


def graph_inputs(graph: Mapping[str, Any], path: Path) -> dict[str, list[Any]]:
    """グラフ入力の `{名前: 形}`（並びの検査は呼び出し側 — ここは引くための表）。"""
    inputs = graph.get("inputs")
    if not isinstance(inputs, list):
        raise DistError(f"{path}: IR メタデータに inputs が無い")
    return {
        item["name"]: item["shape"]
        for item in inputs
        if isinstance(item, dict) and isinstance(item.get("shape"), list)
    }


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


class _SharedSeat(NamedTuple):
    """`shared/<rel_path>` へ **1 回だけ**置く席 — 代表の実体・出所の sha256・そこを指す座席群。"""

    rel_path: str
    artifact: Artifact
    digest: str
    #: この席を指す `(モデル名, 役割名)`。各モデルの manifest エントリが同じ path を書く。
    members: tuple[tuple[str, str], ...]


def _source_digest(artifact: Artifact, memo: dict[Path, str]) -> str:
    """`Artifact` の**出所**の sha256（同じ実ファイルは 1 回しか読まない）。

    共有判定を**置く前**に済ませるための唯一の読み取りなので、memo のキーは resolve 済みの
    実 path — 同じファイルを別綴りで指す複数モデルを 1 回に畳む。生成物（`payload`）は既に
    メモリ上に中身があるので I/O は要らない。
    """
    if artifact.source is None:
        assert artifact.payload is not None  # __post_init__ の不変条件
        return hashlib.sha256(artifact.payload).hexdigest()
    resolved = artifact.source.resolve()
    digest = memo.get(resolved)
    if digest is None:
        digest = memo[resolved] = sha256_file(resolved)
    return digest


def _plan_shared(plans: Sequence[ModelPlan]) -> list[_SharedSeat]:
    """同じ相対 path・同じ sha256 のファイルを 2 モデル以上が持つ席を**配置の前**に決める。

    MUST: 一致の条件は **モデルサブツリー内の相対 path と sha256 の両方**（ADR 0041 §5 の
    「path の一致で共有」を、中身が同じであることまで確かめてから成立させる）。中身違いを
    同じ席へ寄せると、どちらのモデルかで別の重みが黙って読まれる。

    MUST: **1 つの相対 path が持てる席は 1 つだけ**（`shared/<相対 path>` は path だけで決まる）
    — 同じ path に畳める組が 2 つ以上あるときは、どの組も畳まない（各モデルのサブツリーに
    独立コピーのまま残す）。畳むと後の組が先の組の実体を上書きし、先の組のモデルが**別の中身**
    を指す manifest で配られる（{@link verify_dist} は sha256 を採り直さないので沈黙する）。

    MUST: 判定は**席ごと**（`(モデル名, 役割名)`）で、相対 path ごとではない — 1 つの
    相対 path に「畳める組」と「単独の中身違い」が同居するとき、畳むのは組だけで単独は
    自分のサブツリーに残る。

    出所を hash するのは **2 モデル以上が使う相対 path** の席だけ — 1 モデルしか使わない
    ファイルは畳みようがなく、sha256 は置いた現物から採るので、ここで読む理由が無い。
    """
    users: dict[str, set[str]] = {}
    for plan in plans:
        for artifact in plan.artifacts.values():
            users.setdefault(artifact.rel_path, set()).add(plan.name)
    memo: dict[Path, str] = {}
    groups: dict[tuple[str, str], list[tuple[str, str]]] = {}
    samples: dict[tuple[str, str], Artifact] = {}
    for plan in plans:
        for role, artifact in plan.artifacts.items():
            if len(users[artifact.rel_path]) < 2:
                continue
            key = (artifact.rel_path, _source_digest(artifact, memo))
            groups.setdefault(key, []).append((plan.name, role))
            samples.setdefault(key, artifact)
    foldable = [key for key, members in groups.items() if len({model for model, _ in members}) >= 2]
    seats = Counter(rel_path for rel_path, _ in foldable)
    shared: list[_SharedSeat] = []
    for key in foldable:
        rel_path, digest = key
        if seats[rel_path] > 1:
            continue
        shared.append(_SharedSeat(rel_path, samples[key], digest, tuple(groups[key])))
    return shared


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


def _assert_model_scale(
    name: str,
    *,
    weights: int,
    assets: int,
    quants: int,
    pipeline_config: Mapping[str, Any],
) -> None:
    """1 モデルぶんの規模上限（ADR 0041 §7）。件数だけを受けるのは、同じ規則を計画
    （{@link ModelPlan}）と manifest の両方から掛けるため — 上限の綴りは 1 箇所に置く。
    """
    for key, count, limit in (
        ("weights", weights, MAX_WEIGHTS),
        ("assets", assets, MAX_ASSETS),
        ("quants", quants, MAX_QUANTS),
    ):
        if count > limit:
            raise DistError(f"{name}.{key} が {count} 件で上限 {limit} を超えた")
    config_bytes = len(json.dumps(dict(pipeline_config), ensure_ascii=False).encode("utf-8"))
    if config_bytes > MAX_PIPELINE_CONFIG_BYTES:
        raise DistError(
            f"{name}.pipelineConfig が {config_bytes} バイトで上限"
            f" {MAX_PIPELINE_CONFIG_BYTES} を超えた"
        )


def assert_plan_limits(plans: Sequence[ModelPlan]) -> None:
    """計画だけから決まる規模上限（ADR 0041 §7）を**配置の前**に落とす。

    `MAX_MANIFEST_BYTES` はここに無い — manifest 全体の綴りは置いた現物の
    `{path, size, sha256}` が決まるまで測れないので、{@link assert_manifest_limits} に残る。
    """
    if len(plans) > MAX_MODELS:
        raise DistError(f"models が {len(plans)} 件で上限 {MAX_MODELS} を超えた")
    for plan in plans:
        _assert_model_scale(
            plan.name,
            weights=len(plan.weights),
            assets=len(plan.assets),
            quants=len(plan.quants),
            pipeline_config=plan.pipeline_config,
        )


def assert_plan_sources(plans: Sequence[ModelPlan]) -> None:
    """計画が指す入力ファイルが全部実在することを**配置の前**に落とす。

    格納 dtype の要求表（{@link assert_storage}）は要求の無い役割を素通しするので、tokenizer の
    ような「dtype を要求しないファイル」の欠落だけが計画段をすり抜けて配置の**途中**で出ていた。
    出所が生成物（`payload`）の {@link Artifact} は書く中身を既に持っているので対象外。
    """
    for plan in plans:
        for role, artifact in plan.artifacts.items():
            if artifact.source is not None and not artifact.source.is_file():
                raise DistError(f"{plan.name}.{role}: 組み立ての入力が無い: {artifact.source}")


def assert_manifest_limits(manifest: Mapping[str, Any]) -> None:
    """規模上限（ADR 0041 §7）を焼く側で先に落とす。"""
    models = manifest["models"]
    if len(models) > MAX_MODELS:
        raise DistError(f"models が {len(models)} 件で上限 {MAX_MODELS} を超えた")
    for name, model in models.items():
        _assert_model_scale(
            name,
            weights=len(model["weights"]),
            assets=len(model["assets"]),
            quants=len(model["quants"]),
            pipeline_config=model["pipelineConfig"],
        )
    total = len(manifest_text(manifest).encode("utf-8"))
    if total > MAX_MANIFEST_BYTES:
        raise DistError(f"manifest が {total} バイトで上限 {MAX_MANIFEST_BYTES} を超えた")


def manifest_text(manifest: Mapping[str, Any]) -> str:
    """`karume.json` に書く綴り（末尾改行つき — 上限検査もこの綴りで測る）。"""
    return json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"


def _discard_tree(path: Path) -> None:
    """`path` を（在れば）丸ごと消す — staging / 退避先の後始末。

    中断が残した作業ディレクトリを踏み直すと、plan に無いファイルが新しいツリーへ黙って混ざる
    （{@link verify_dist} が宣言外ファイルとして落とすが、数 GB を並べ切った後になる）。
    """
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _materialize_family(
    plans: Sequence[ModelPlan], out_dir: Path, default_model: str
) -> dict[str, Any]:
    """検査済みの計画群を `out_dir` へ並べ、`karume.json` を書いて manifest を返す。

    ① 共有席を決める（{@link _plan_shared} — 出所の sha256 だけを見る）→ ② 共有席は
    `shared/` へ 1 回だけ・残りは各モデルのサブツリーへ置く → ③ 現物から manifest。

    MUST: 共有の判定は**置く前**に済ませる — 全モデルへ複製してから畳み直す形は、共有 1 本
    （サイズ S・M モデル）につき「複製 M 回 + hash M 回 + 移動 1 回」で ~3MS の論理 I/O を
    払う。決めてから置けば S の複製も hash も 1 回で済む（配布の大半は共有資産）。

    書き先は呼び手（{@link assemble_family}）が用意する staging で、**空から作る**前提
    （配布先を直接更新しないので、途中で落ちても捨てるだけで済む）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    placed: dict[tuple[str, str], str] = {}
    digests: dict[str, str] = {}
    folded: dict[tuple[str, str], str] = {}
    for seat in _plan_shared(plans):
        target = f"{SHARED_DIRNAME}/{seat.rel_path}"
        materialize(seat.artifact, out_dir / target)
        # sha256 は**置いた現物**から採る（表と現物が食い違う失敗様式を構造的に消す）。
        digests[target] = sha256_file(out_dir / target)
        # MUST: 共有席だけは出所の sha256 とも突き合わせる — 1 本しか置かないので、コピーが
        # 壊れれば**全モデルが揃って壊れた実体**を指す。宣言と現物は一致したままなので
        # {@link verify_dist} は沈黙する（採り直さない）。
        if digests[target] != seat.digest:
            raise DistError(
                f"{target}: 置いた現物の sha256 が出所と食い違う"
                f"（出所 {seat.digest} / 現物 {digests[target]}）"
            )
        for member in seat.members:
            folded[member] = target
    for plan in plans:
        for role, artifact in plan.artifacts.items():
            target = folded.get((plan.name, role))
            if target is not None:
                placed[(plan.name, role)] = target
                continue
            rel_path = f"{plan.name}/{artifact.rel_path}"
            materialize(artifact, out_dir / rel_path)
            placed[(plan.name, role)] = rel_path
            digests[rel_path] = sha256_file(out_dir / rel_path)

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


def assemble_family(
    plans: Sequence[ModelPlan],
    out_dir: Path,
    default_model: str,
    render_card: Callable[[Mapping[str, Any]], str] | None = None,
) -> dict[str, Any]:
    """計画済みのモデル群を 1 リポへ組み立て、`out_dir` をその形へ**丸ごと差し替える**。

    staging（`<出力先の名前>.staging`・同じ親）へ全部作り（共有席の決定 → 配置 →
    `karume.json` → {@link verify_dist} → `README.md`）、通ってから rename で据える。既存の
    `out_dir` は最後の rename まで 1 バイトも触らない — 途中で落ちれば staging だけが消えて、
    配布形は前回のまま残る（in-place で更新していた頃の「旧 manifest + 新旧混在ツリー」という
    失敗様式が構造的に無くなる）。前回の中断が残した staging は黙って捨てて作り直し、退避先は
    出力先が在れば捨てる・無ければ（= 前回が rename 2 回の間で落ちた形）出力先へ戻す。

    MUST: 差し替えは**丸ごと**で、`out_dir` の元の中身は 1 つも引き継がない — `A` + `B` で
    組んだ出力へ `A` だけを組み直すと `B` は消える（全部コピーし終えてから宣言外ファイルとして
    見つかるのではなく、そもそも残らない）。`README.md` も staging の中で描くので、据わった
    配布形はカードまで揃っている（`render_card` は**検証を通った** manifest を受け取る）。

    MUST: `plans` 全体に掛かる検査は**最初の 1 バイトを書く前**に全部済ませる（staging すら
    作らない）— 途中で落ちると数 GB を並べ直すことになる。
    """
    if not plans:
        raise DistError("組み立てるモデルが 1 つも無い")
    names = [plan.name for plan in plans]
    if len(set(names)) != len(names):
        raise DistError(f"モデル名が重複している: {names}")
    if default_model not in names:
        raise DistError(f"既定モデル '{default_model}' が組み立て対象 {names} に無い")
    assert_plan_limits(plans)
    assert_plan_sources(plans)

    staging = out_dir.with_name(out_dir.name + STAGING_SUFFIX)
    superseded = out_dir.with_name(out_dir.name + SUPERSEDED_SUFFIX)
    _discard_tree(staging)
    # 退避先だけが在って出力先が無いのは、前回が rename 2 回の**間**で落ちた形 — 退避先が
    # last-known-good の配布形そのものなので、捨てると ADR 0052 Decision 2 の「既存配布形は
    # 不変」が次の起動で破れる。戻してから作り直す（戻せば退避先は消えるので下は no-op・
    # 出力先が在るときの退避先は従来どおりただの残骸）。
    if superseded.is_dir() and not out_dir.exists():
        os.replace(superseded, out_dir)
    _discard_tree(superseded)
    try:
        manifest = _materialize_family(plans, staging, default_model)
        verify_dist(staging)
        if render_card is not None:
            (staging / MODEL_CARD_FILENAME).write_text(render_card(manifest), encoding="utf-8")
    except BaseException:
        # 中断（Ctrl-C）も含めて staging は残さない — 既存の配布形は触っていないので不変。
        _discard_tree(staging)
        raise
    # 据えるのは rename 2 回。非空ディレクトリの上へは rename できないので既存を先に退避する。
    if out_dir.exists():
        os.replace(out_dir, superseded)
    try:
        os.replace(staging, out_dir)
    except OSError as error:
        # 2 回目が落ちると、唯一の正常な配布形が退避先にしか無い状態のまま止まる — 戻し、
        # 据わらなかった staging は捨てる（ADR 0052 Decision 2 の「失敗は staging だけを消し、
        # 既存配布形は不変」— 他の失敗経路と同じ後片付けへ揃える）。
        if superseded.exists():
            os.replace(superseded, out_dir)
        _discard_tree(staging)
        raise DistError(f"{out_dir} への据え替え（rename）に失敗した") from error
    _discard_tree(superseded)
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


# ---- ⑥ SigLIP2（image-feature-extraction）------------------------------------
#
# 配布するのは **vision tower の 1 グラフだけ**（text tower は載っていない — `export_siglip2.py`
# の docstring）。実行に要る資産もそれ 1 本で、tokenizer も表も無い（`assets` は空）。格納
# dtype は f32 の 1 系列だけなので quant 席も 1 つ。
#
# **1 リポ 1 モデル**（base と so400m は解像度も hidden も違う別物で、同居させると利用者が
# 何も指定しなかったときに引く既定が寸法ごと変わる）。ファミリー組み立ての機構は共有部が
# 持ったままだが、既定の出力先は `karume-siglip2-<モデル名>` の単一モデル形。
#
# `pipelineConfig` の数は**2 つの独立した出どころ**から来る: 前処理の定数は上流の
# `preprocessor_config.json`、`hiddenDim` は焼かれたグラフの出力宣言。どちらも写経しない
# （TS 側の正本 `packages/models/src/siglip2/config.ts` のモジュール doc と同じ理由）。前処理の
# 寸法と焼かれた解像度は別々に決まるので、噛み合っていることは {@link assert_siglip2_graph} が
# 実測する — ずれた組み合わせは「resize 先だけが違う」形で、値が静かに崩れる。

#: パイプライン契約（ADR 0041 §2 — モデル単位）。TS 側の受理集合は
#: `SIGLIP2_PIPELINE_NAME` / `SIGLIP2_PIPELINE_MAJOR`。
SIGLIP2_PIPELINE = "siglip2/1"

#: 既定のモデル名（= 既定のリポ名 `karume-siglip2-base` の末尾）。綴りの受理集合は帰属表
#: （`karume.modelcard.SIGLIP2_UPSTREAM`）が持つ。
SIGLIP2_DEFAULT_MODEL = "base"

#: 系列名とリポ名の接頭辞（`karume-siglip2-<モデル名>`）。
SIGLIP2_PREFIX = "siglip2"

#: 実重みと `preprocessor_config.json` の親（`hf download google/<名前> --local-dir
#: inputs/siglip2/<名前>` の展開先 — `export_siglip2.MODELS_ROOT` と同じ場所）。
SIGLIP2_INPUTS_DIRNAME = "siglip2"

#: 唯一の役割名（manifest の weights キー・TS 側 `pipeline.ts` の `VISION`）。
SIGLIP2_ROLE = "vision"

#: グラフ入力の名前（`export_siglip2.INPUT_NAME`）。
SIGLIP2_INPUT = "pixel_values"

#: 入力のチャネル数（`patch_siglip2.assert_supported` が config 側でも要求する 3）。
SIGLIP2_CHANNELS = 3

#: 前処理定数の出どころ（重みと同じディレクトリ）。
SIGLIP2_PREPROCESSOR_FILE = "preprocessor_config.json"

#: TS 側が実装している唯一の補間と、それに対応する `resample` の値（**PIL の定数で BILINEAR**
#: — BICUBIC は 3。クラス属性の既定が BICUBIC なので読み違えやすいが、チェックポイントの
#: config が上書きしている）。`siglip2_preprocess.py` の `EXPECTED_RESAMPLE` と同じ実測対象。
SIGLIP2_INTERPOLATION = "bilinear"
SIGLIP2_RESAMPLE = 2

#: TS 側 `normalizeToNchw` が持つ除数 255 の逆数。**pipelineConfig には載せない**（実行時に
#: 選べない数を宣言だけ持たせると正本が 2 つになる — `config.ts` の NOTE）ので、違う値の
#: チェックポイントはここで落とす。
SIGLIP2_RESCALE_FACTOR = 1.0 / 255.0

#: 出力の相対 path（**モデルサブツリー内**）— 配置表と manifest が共有する 1 箇所。格納 dtype を
#: ファイル名に出すのは Irodori と同じ形（系列が 2 本並んでも取り違えようがない綴り）。
SIGLIP2_OUTPUT_PATHS: Mapping[str, str] = {SIGLIP2_ROLE: f"{SIGLIP2_ROLE}/model.f32.safetensors"}

#: 格納 dtype の要求（Anima / SBV2 / Irodori と同じ根拠 — 素の資産が組み立て・ロード・実行を
#: 全て通って参照一致の門まで沈黙した実測事故）。
#:
#: NOTE: 禁止表（{@link assert_storage_absent}）は持たない — 圧縮系列が 1 本も無いので、
#: 「F32 を含む」で系列 × 格納 dtype が一意に決まる。f16 / i8 の席を足すときは**同時に**禁止表も
#: 足す（圧縮系列も適格外の重みを F32 で持つので、存在検査だけでは f32 席へ混入する）。
SIGLIP2_STORAGE_REQUIREMENTS: Mapping[str, str] = {SIGLIP2_ROLE: "F32"}

#: weights の宣言（dtype ラベル → 役割名）。dtype が 1 つしかないので quant 表は空でよい
#: （{@link complete_quant_weights} が完全写像へ埋める）。
SIGLIP2_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    SIGLIP2_ROLE: {"f32": WeightFiles(SIGLIP2_ROLE)}
}

#: assets の宣言。**空**（tokenizer も表も要らない — 実行に要るのはグラフ 1 本だけ）。
SIGLIP2_ASSETS: Mapping[str, str] = {}

SIGLIP2_QUANTS: Mapping[str, Any] = {"f32": {"weights": {}, "session": {}}}
SIGLIP2_DEFAULT_QUANT = "f32"


def siglip2_checkpoint(model: str) -> str:
    """モデル名 → 上流チェックポイントのディレクトリ名（= 系列名 = 上流の HF リポ名）。

    `export_siglip2.py` は `--model-dir` のディレクトリ名をそのまま系列名にし、そのディレクトリ
    名は `hf download <リポ>` の展開先（= リポ名の末尾）なので、綴りの事実は「このモデルが
    どの上流リポか」1 つしかない。帰属表（`karume.modelcard.SIGLIP2_UPSTREAM`）から導いて、
    ここに 2 つ目の表を持たない — 2 表になると、片方だけ動いたときに「別のモデルの重みを
    別のモデルとして帰属する」形が黙って作れる。
    """
    repo = SIGLIP2_UPSTREAM.get(model)
    if repo is None:
        raise DistError(
            f"SigLIP2 のモデル '{model}' は知らない（既知: {' / '.join(sorted(SIGLIP2_UPSTREAM))}）"
        )
    return repo.split("/", 1)[1]


def siglip2_repo_name(model: str) -> str:
    """単一モデルの配布リポ名（`karume-` prefix はリポ名裁定 2026-08-09）。"""
    return f"karume-{SIGLIP2_PREFIX}-{model}"


@dataclass(frozen=True)
class Siglip2Sources:
    """組み立ての入力。系列（グラフ 1 本）と実重みの置き場（`inputs/` — 生成物ではない）。

    後者が要るのは `pipelineConfig` の前処理定数を**焼き込まずに導出**するため
    （読むのは `preprocessor_config.json` 1 本だけなので、1.5GB の重みには触らない）。
    """

    series: Path
    model: Path


def siglip2_sources(series_dir: Path, model: str = SIGLIP2_DEFAULT_MODEL) -> Siglip2Sources:
    """系列の親ディレクトリ（`outputs/series/`）と `karume.paths` の綴りから入力を引く。"""
    checkpoint = siglip2_checkpoint(model)
    return Siglip2Sources(
        series=series_dir / checkpoint,
        model=INPUTS_ROOT / SIGLIP2_INPUTS_DIRNAME / checkpoint,
    )


def siglip2_placements(sources: Siglip2Sources) -> dict[str, Path]:
    """役割名 → 出所のファイル。出力の path は {@link SIGLIP2_OUTPUT_PATHS} が持つ。

    この表に無いものは出力へ入らない（系列に並ぶ `io.*.safetensors` はこれで落ちる）。
    """
    return {SIGLIP2_ROLE: sources.series / "model.safetensors"}


def siglip2_preprocessor(model_dir: Path) -> Mapping[str, Any]:
    """上流の `preprocessor_config.json` を読む（前処理定数の唯一の出どころ）。"""
    path = model_dir / SIGLIP2_PREPROCESSOR_FILE
    if not path.is_file():
        raise DistError(f"組み立ての入力が無い: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DistError(f"{path} が JSON として読めない") from error
    if not isinstance(raw, dict):
        raise DistError(f"{path}: 最上位オブジェクトでない")
    return raw


def _preprocessor_int(raw: Mapping[str, Any], key: str, where: str) -> int:
    """前処理 config の整数フィールド（寸法）を検査して読む。

    HF の `preprocessor_config.json` を読む節（SigLIP2 / Depth Anything V2）で共有する —
    値域は TS 側の各 `config.ts` と同じで、独立に動く写しを増やすと片方だけ緩む。
    """
    value = raw.get(key)
    # bool は int の派生。`"height": true` を 1 として通すと寸法の突合が緩む。
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise DistError(f"{where} の {key} が正の整数でない（{value!r}）")
    return value


def _preprocessor_channels(
    raw: Mapping[str, Any], key: str, where: str, *, channels: int, positive: bool
) -> list[float]:
    """`image_mean` / `image_std` を検査して読む（TS 側 `config.ts` と同じ値域）。

    `std` に 0 を通さないのは、0 除算が例外を出さずに `±Infinity` の `pixel_values` を作るため。
    チャネル数は**呼び出し側の節が持つ定数**を受ける（読み手だけを共有し、各節の宣言は
    それぞれの席に残す）。
    """
    value = raw.get(key)
    if not isinstance(value, list) or len(value) != channels:
        raise DistError(f"{where} の {key} が長さ {channels} の配列でない（{value!r}）")
    channels_out: list[float] = []
    for entry in value:
        if not isinstance(entry, int | float) or isinstance(entry, bool):
            raise DistError(f"{where} の {key} に数でない要素がある（{value!r}）")
        if positive and entry <= 0:
            raise DistError(f"{where} の {key} に正でない要素がある（{value!r}）")
        channels_out.append(float(entry))
    return channels_out


def siglip2_pipeline_config(
    preprocessor: Mapping[str, Any], hidden_dim: int, where: str
) -> dict[str, Any]:
    """`pipelineConfig`（TS 側スキーマの 6 欄）を前処理 config と焼かれたグラフから組む。

    MUST: 対応外の前処理は**受理しない**（黙って近似しない）。TS 側が持つのは
    「antialias 付き bilinear → `/255` → `(x − mean) / std`」の 1 本きりなので、
    `resample` / `rescale_factor` / `do_*` が外れたチェックポイントは、値が最大 47/255 ずれた
    まま（bicubic の実測）ロードも実行も通る。ここが唯一の検出器になる。
    """
    size = preprocessor.get("size")
    if not isinstance(size, dict):
        raise DistError(f"{where} に size が無い / オブジェクトでない（{size!r}）")
    resample = preprocessor.get("resample")
    if resample != SIGLIP2_RESAMPLE:
        raise DistError(
            f"{where} の resample が {resample!r}（期待 {SIGLIP2_RESAMPLE} = PIL の BILINEAR）"
            f" — TS 側の前処理は {SIGLIP2_INTERPOLATION} の 1 本しか持たない"
        )
    rescale = preprocessor.get("rescale_factor")
    if rescale != SIGLIP2_RESCALE_FACTOR:
        raise DistError(
            f"{where} の rescale_factor が {rescale!r}（期待 {SIGLIP2_RESCALE_FACTOR}）"
            " — TS 側は 8bit 画素を 255 で割る形で閉じている"
        )
    for flag in ("do_resize", "do_rescale", "do_normalize"):
        if preprocessor.get(flag) is not True:
            raise DistError(f"{where} の {flag} が真でない — 前処理の 3 段が揃っていない")
    for flag in ("do_center_crop", "do_pad"):
        if preprocessor.get(flag):
            raise DistError(f"{where} の {flag} が真 — TS 側は crop も pad も持たない")
    return {
        "imageWidth": _preprocessor_int(size, "width", f"{where} の size"),
        "imageHeight": _preprocessor_int(size, "height", f"{where} の size"),
        "imageMean": _preprocessor_channels(
            preprocessor, "image_mean", where, channels=SIGLIP2_CHANNELS, positive=False
        ),
        "imageStd": _preprocessor_channels(
            preprocessor, "image_std", where, channels=SIGLIP2_CHANNELS, positive=True
        ),
        "hiddenDim": hidden_dim,
        "interpolation": SIGLIP2_INTERPOLATION,
    }


def siglip2_hidden_dim(graph: Mapping[str, Any], path: Path) -> int:
    """焼かれたグラフの出力宣言から `pooler_output` の幅を読む（写経しない）。

    出力が 1 本であることまで見るのは、`last_hidden_state` 込みの別の export や golden 用の
    多出力版が混ざると、**幅だけが静かに別の意味の数**になるため。
    """
    outputs = graph.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise DistError(
            f"{path}: グラフ出力が {outputs!r} — vision tower が出すのは pooler_output 1 本だけ"
        )
    values = graph.get("values")
    value = values.get(outputs[0]) if isinstance(values, dict) else None
    shape = value.get("shape") if isinstance(value, dict) else None
    if not isinstance(shape, list) or len(shape) != 2 or shape[0] != 1:
        raise DistError(f"{path}: グラフ出力 '{outputs[0]}' の形が [1, hidden] でない（{shape!r}）")
    hidden = shape[1]
    if not isinstance(hidden, int) or isinstance(hidden, bool) or hidden <= 0:
        raise DistError(f"{path}: グラフ出力の hidden が正の整数でない（{hidden!r}）")
    return hidden


def assert_siglip2_graph(
    graph: Mapping[str, Any], path: Path, pipeline_config: Mapping[str, Any]
) -> None:
    """グラフの入力が**前処理の宣言と噛み合う**ことを、配置の前に実測する。

    MUST: 落とさない。前処理の寸法（`preprocessor_config.json`）と焼かれた解像度（重みの
    `config.json`）は別々に決まるので、base の前処理 config と so400m のグラフを組み合わせても
    ここまでは何も落ちない — ずれたまま配ると、利用者の手元で Session の shape 検査が
    「どちらの数が正しいのか」を伝えないまま落ちる。
    """
    inputs = graph_inputs(graph, path)
    if tuple(inputs) != (SIGLIP2_INPUT,):
        raise DistError(
            f"{path} のグラフ入力が {list(inputs)} で、期待の {[SIGLIP2_INPUT]} と違う"
            " — 実行側は名前で束ねるので、綴りが変われば束ねられない"
        )
    symbols = graph.get("symbols")
    if not isinstance(symbols, list) or symbols:
        raise DistError(
            f"{path}: 記号次元 {symbols!r} がある — 解像度もパッチ数も固定で、動かす軸は無い"
        )
    expected = [
        1,
        SIGLIP2_CHANNELS,
        pipeline_config["imageHeight"],
        pipeline_config["imageWidth"],
    ]
    if inputs[SIGLIP2_INPUT] != expected:
        raise DistError(
            f"{path} の入力 '{SIGLIP2_INPUT}' が {inputs[SIGLIP2_INPUT]!r}、"
            f"{SIGLIP2_PREPROCESSOR_FILE} の size から組んだ期待は {expected}"
            " — 前処理の寸法と焼かれた解像度が別の版"
        )


def siglip2_plan(sources: Siglip2Sources, model: str = SIGLIP2_DEFAULT_MODEL) -> ModelPlan:
    """SigLIP2 1 モデルぶんの計画を組む（検査と config の読み取りをここで全部済ませる）。"""
    assert_model_name(model)
    placements = siglip2_placements(sources)
    for role, source in placements.items():
        assert_storage(role, source, SIGLIP2_STORAGE_REQUIREMENTS)
    container = placements[SIGLIP2_ROLE]
    graph = ir_graph(container)
    pipeline_config = siglip2_pipeline_config(
        siglip2_preprocessor(sources.model),
        siglip2_hidden_dim(graph, container),
        str(sources.model / SIGLIP2_PREPROCESSOR_FILE),
    )
    assert_siglip2_graph(graph, container, pipeline_config)
    return ModelPlan(
        name=model,
        pipeline=SIGLIP2_PIPELINE,
        artifacts={
            role: Artifact(SIGLIP2_OUTPUT_PATHS[role], source=source)
            for role, source in placements.items()
        },
        weights=SIGLIP2_WEIGHTS,
        assets=SIGLIP2_ASSETS,
        quants=complete_quant_weights(SIGLIP2_WEIGHTS, SIGLIP2_QUANTS),
        default_quant=SIGLIP2_DEFAULT_QUANT,
        pipeline_config=pipeline_config,
    )


# ---- ⑦ BiRefNet 系（image-segmentation）--------------------------------------
#
# 配布するのは **マット 1 グラフだけ**（`export_birefnet.py` は最終段の logit 1 本しか出さない）。
# 実行に要る資産もそれ 1 本で、tokenizer も表も無い（`assets` は空）。格納 dtype は f32 の
# 1 系列だけなので quant 席も 1 つ — ここまでは SigLIP2 と同じ形。
#
# **1 リポ 1 モデル**（ユーザー裁定 — BiRefNet_HR と Lucida は構造が同一で重みだけが違う
# fine-tune 同士なので、まとめると「何も指定しなかったときにどちらの学習が動くか」が
# 既定に隠れる）。リポ名は導出せず {@link BIREFNET_REPO_NAMES} が持つ — `karume-lucida` は
# 「BiRefNet 系の 1 つ」ではなく上流が名前で売っているモデルで、綴りは命名の決定であって
# モデル名から決まらない（SBV2 のファミリー名と同じ性質）。
#
# `pipelineConfig` の数の出どころは **2 つとも独立**:
#
# - resize 先（`imageWidth` / `imageHeight`）は**焼かれたグラフの入力宣言**から導く（写経しない）。
# - 正規化定数は上流に機械可読な出どころが**無い** — SigLIP2 の `preprocessor_config.json` に
#   当たるものが BiRefNet 系には無く、事実は同梱 `handler.py` の `ImagePreprocessor`（と
#   モデルカードの利用例）の中にしか書かれていない。したがってここが宣言として持ち
#   （{@link BIREFNET_IMAGE_MEAN}）、台本側の写し（`export_birefnet.IMAGENET_MEAN`）との一致は
#   pytest が毎回突き合わせる（2 表が独立に動く形にはしない）。

#: パイプライン契約（ADR 0041 §2 — モデル単位）。TS 側の受理集合は
#: `BIREFNET_PIPELINE_NAME` / `BIREFNET_PIPELINE_MAJOR`。
BIREFNET_PIPELINE = "birefnet/1"

#: 既定のモデル名。綴りの受理集合は帰属表（`karume.modelcard.BIREFNET_UPSTREAM`）が持つ。
BIREFNET_DEFAULT_MODEL = "hr"

#: 実重みの親（`hf download <リポ> --local-dir inputs/birefnet/<名前>` の展開先 —
#: `export_birefnet.MODELS_ROOT` と同じ場所）。系列名はこのディレクトリ名から決まる。
BIREFNET_INPUTS_DIRNAME = "birefnet"

#: 単一モデルの配布リポ名（`karume-` prefix はリポ名裁定 2026-08-09）。**導出しない** —
#: 節の冒頭に書いたとおり、綴りは命名の決定。
BIREFNET_REPO_NAMES: Mapping[str, str] = {
    "hr": "karume-birefnet-hr",
    "lucida": "karume-lucida",
}

#: 唯一の役割名（manifest の weights キー・TS 側 `pipeline.ts` の `MATTE`）。
BIREFNET_ROLE = "matte"

#: グラフ入力の名前（`export_birefnet.INPUT_NAME`）。
BIREFNET_INPUT = "pixel_values"

#: 入力のチャネル数（RGB）と、出力（マット）のチャネル数。
BIREFNET_CHANNELS = 3
BIREFNET_MATTE_CHANNELS = 1

#: TS 側が実装している唯一の補間。上流（`handler.py` / モデルカードの利用例）はどちらも
#: `torchvision.transforms.Resize((S, S))` を既定の補間で通す = bilinear。
BIREFNET_INTERPOLATION = "bilinear"

#: 前処理の正規化定数（節の冒頭のとおり、上流に機械可読な出どころが無いのでここが持つ）。
#: 正本は同梱 `handler.py` の `ImagePreprocessor` = ImageNet 統計。
BIREFNET_IMAGE_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
BIREFNET_IMAGE_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

#: 配る解像度。系列は解像度ごとに別（`export_birefnet.py` の「解像度軸」）なので、系列名を
#: 引くのにこの数が要る。**2048²（本家 handler の General-HR）は配らない** — conv2d の
#: dispatch 上限と中間 3.22GB で実行段が未実測（docs/limitations.md）。焼かれたグラフが別の
#: 解像度なら {@link birefnet_pipeline_config} が落とす。
BIREFNET_RESOLUTION = 1024

#: 出力の相対 path（**モデルサブツリー内**）— 配置表と manifest が共有する 1 箇所。
BIREFNET_OUTPUT_PATHS: Mapping[str, str] = {BIREFNET_ROLE: f"{BIREFNET_ROLE}/model.f32.safetensors"}

#: 格納 dtype の要求（Anima / SBV2 / Irodori / SigLIP2 と同じ根拠 — 素の資産が組み立て・
#: ロード・実行を全て通って参照一致の門まで沈黙した実測事故）。
#:
#: NOTE: 禁止表（{@link assert_storage_absent}）は持たない — 圧縮系列が 1 本も無いので、
#: 「F32 を含む」で系列 × 格納 dtype が一意に決まる。f16 / i8 の席を足すときは**同時に**禁止表も
#: 足す（圧縮系列も適格外の重みを F32 で持つので、存在検査だけでは f32 席へ混入する）。
BIREFNET_STORAGE_REQUIREMENTS: Mapping[str, str] = {BIREFNET_ROLE: "F32"}

#: weights の宣言（dtype ラベル → 役割名）。dtype が 1 つしかないので quant 表は空でよい。
BIREFNET_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    BIREFNET_ROLE: {"f32": WeightFiles(BIREFNET_ROLE)}
}

#: assets の宣言。**空**（実行に要るのはグラフ 1 本だけ）。
BIREFNET_ASSETS: Mapping[str, str] = {}

BIREFNET_QUANTS: Mapping[str, Any] = {"f32": {"weights": {}, "session": {}}}
BIREFNET_DEFAULT_QUANT = "f32"


def birefnet_checkpoint(model: str) -> str:
    """モデル名 → 上流チェックポイントのディレクトリ名（= 上流の HF リポ名の末尾）。

    `export_birefnet.py` は `--model-dir` のディレクトリ名を系列名にし、そのディレクトリ名は
    `hf download <リポ>` の展開先なので、綴りの事実は「このモデルがどの上流リポか」1 つしか
    ない。帰属表（`karume.modelcard.BIREFNET_UPSTREAM`）から導いて、ここに 2 つ目の表を
    持たない（SigLIP2 の {@link siglip2_checkpoint} と同じ規律）。
    """
    repo = BIREFNET_UPSTREAM.get(model)
    if repo is None:
        raise DistError(
            f"BiRefNet 系のモデル '{model}' は知らない"
            f"（既知: {' / '.join(sorted(BIREFNET_UPSTREAM))}）"
        )
    return repo.split("/", 1)[1]


def birefnet_series_name(model: str) -> str:
    """モデル名 → 系列ディレクトリ名（`export_birefnet.default_out_dir` と同じ式）。"""
    return f"{birefnet_checkpoint(model).lower().replace('_', '-')}-{BIREFNET_RESOLUTION}"


def birefnet_repo_name(model: str) -> str:
    """単一モデルの配布リポ名（{@link BIREFNET_REPO_NAMES}）。"""
    name = BIREFNET_REPO_NAMES.get(model)
    if name is None:
        raise DistError(
            f"BiRefNet 系のモデル '{model}' のリポ名が無い"
            f"（既知: {' / '.join(sorted(BIREFNET_REPO_NAMES))}）"
        )
    return name


@dataclass(frozen=True)
class BirefnetSources:
    """組み立ての入力。系列（グラフ 1 本）だけ。

    SigLIP2 と違って実重みの置き場を持たないのは、前処理定数の出どころになる機械可読な
    ファイルが上流に無いから（節の冒頭）。**どのモデルの重みかを言えるのは系列 path だけ**
    なので、系列名の導出（{@link birefnet_series_name}）が帰属の唯一の紐づけになる。
    """

    series: Path


def birefnet_sources(series_dir: Path, model: str = BIREFNET_DEFAULT_MODEL) -> BirefnetSources:
    """系列の親ディレクトリ（`outputs/series/`）とモデル名から入力を引く。"""
    return BirefnetSources(series=series_dir / birefnet_series_name(model))


def birefnet_placements(sources: BirefnetSources) -> dict[str, Path]:
    """役割名 → 出所のファイル。出力の path は {@link BIREFNET_OUTPUT_PATHS} が持つ。

    この表に無いものは出力へ入らない（系列に並ぶ `io.*.safetensors` はこれで落ちる）。
    """
    return {BIREFNET_ROLE: sources.series / "model.safetensors"}


def birefnet_pipeline_config(graph: Mapping[str, Any], path: Path) -> dict[str, Any]:
    """`pipelineConfig`（TS 側スキーマの 5 欄）を焼かれたグラフと正規化定数から組む。

    resize 先はグラフの入力宣言そのもの — 前処理が別の寸法へ伸ばすと、値が静かに崩れたまま
    shape だけ合う。入力の名前・階数・batch・チャネル数もここで見る（後段の
    {@link assert_birefnet_graph} が出力側を見る）。
    """
    inputs = graph_inputs(graph, path)
    if tuple(inputs) != (BIREFNET_INPUT,):
        raise DistError(
            f"{path} のグラフ入力が {list(inputs)} で、期待の {[BIREFNET_INPUT]} と違う"
            " — 実行側は名前で束ねるので、綴りが変われば束ねられない"
        )
    shape = inputs[BIREFNET_INPUT]
    if len(shape) != 4 or shape[0] != 1 or shape[1] != BIREFNET_CHANNELS:
        raise DistError(
            f"{path} の入力 '{BIREFNET_INPUT}' が {shape!r}"
            f" — 期待は [1, {BIREFNET_CHANNELS}, H, W]（batch もチャネル数も静的）"
        )
    height, width = shape[2], shape[3]
    for axis, value in (("H", height), ("W", width)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise DistError(
                f"{path} の入力 '{BIREFNET_INPUT}' の {axis} が正の整数でない（{value!r}）"
            )
    if (height, width) != (BIREFNET_RESOLUTION, BIREFNET_RESOLUTION):
        raise DistError(
            f"{path} は {height}×{width} で焼かれている — 配るのは"
            f" {BIREFNET_RESOLUTION}² だけ（それ以外は実行段が未実測 — docs/limitations.md）"
        )
    return {
        "imageWidth": width,
        "imageHeight": height,
        "imageMean": list(BIREFNET_IMAGE_MEAN),
        "imageStd": list(BIREFNET_IMAGE_STD),
        "interpolation": BIREFNET_INTERPOLATION,
    }


def assert_birefnet_graph(
    graph: Mapping[str, Any], path: Path, pipeline_config: Mapping[str, Any]
) -> None:
    """出力が**マット 1 本**で、入力と同じ寸法であることを配置の前に実測する。

    MUST: 落とさない。`export_birefnet.py` のラッパは学習モードだと multi-scale supervision の
    中間予測まで返す形なので、そちら向けに焼かれたグラフは**位置で引く後段が別の値を α として
    読む**（要素数だけ見る実装なら shape も通る）。記号次元が無いことも同じ席で見る — 解像度も
    窓マスクも定数として焼かれており、動かせる軸は 1 本も無い。
    """
    symbols = graph.get("symbols")
    if not isinstance(symbols, list) or symbols:
        raise DistError(
            f"{path}: 記号次元 {symbols!r} がある — 解像度も窓マスクも定数として焼かれている"
        )
    outputs = graph.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise DistError(f"{path}: グラフ出力が {outputs!r} — 最終段のマット 1 本だけが要る")
    values = graph.get("values")
    value = values.get(outputs[0]) if isinstance(values, dict) else None
    shape = value.get("shape") if isinstance(value, dict) else None
    expected = [
        1,
        BIREFNET_MATTE_CHANNELS,
        pipeline_config["imageHeight"],
        pipeline_config["imageWidth"],
    ]
    if shape != expected:
        raise DistError(
            f"{path}: グラフ出力 '{outputs[0]}' の形が {shape!r}、期待は {expected}"
            " — マットは入力と同じ寸法の 1 チャネル"
        )


def birefnet_plan(sources: BirefnetSources, model: str = BIREFNET_DEFAULT_MODEL) -> ModelPlan:
    """BiRefNet 系 1 モデルぶんの計画を組む（検査と config の読み取りをここで全部済ませる）。"""
    assert_model_name(model)
    # 帰属を名乗れないモデル名は系列を読む前に落とす（リポ名の表と帰属表は独立に欠けうる）。
    birefnet_checkpoint(model)
    birefnet_repo_name(model)
    placements = birefnet_placements(sources)
    for role, source in placements.items():
        assert_storage(role, source, BIREFNET_STORAGE_REQUIREMENTS)
    container = placements[BIREFNET_ROLE]
    graph = ir_graph(container)
    pipeline_config = birefnet_pipeline_config(graph, container)
    assert_birefnet_graph(graph, container, pipeline_config)
    return ModelPlan(
        name=model,
        pipeline=BIREFNET_PIPELINE,
        artifacts={
            role: Artifact(BIREFNET_OUTPUT_PATHS[role], source=source)
            for role, source in placements.items()
        },
        weights=BIREFNET_WEIGHTS,
        assets=BIREFNET_ASSETS,
        quants=complete_quant_weights(BIREFNET_WEIGHTS, BIREFNET_QUANTS),
        default_quant=BIREFNET_DEFAULT_QUANT,
        pipeline_config=pipeline_config,
    )


# ---- ⑧ 母音検出（音声 → リップシンク用の母音系列）-----------------------------
#
# 配布するのは **CRNN グラフ 1 本**と、特徴抽出が要る **mel 基底 1 本**（`assets` 席）。
# 格納 dtype は f32 の 1 系列だけなので quant 席も 1 つ。
#
# ## 長さバケットは無くなった（ADR 0056 / 0057）
#
# かつては `aten.gru.input` が時間方向へ完全展開されて T を動的軸にできず、長さバケット 4 本
# （250 / 500 / 1000 / 2000 フレーム）を weights の役割軸に並べて右ゼロ pad で丸めていた。
# `gru_scan` への差し替え（ADR 0056）と派生次元からのシンボル束縛（ADR 0057）で、**任意長が
# 1 本のグラフで通る**ようになったので、役割は `crnn` 1 つだけになった。
#
# 副作用として **pad 由来の数値差が消える**: 逆方向 GRU が pad 側から状態を持ち帰るせいで
# バケット経路の `.lab` は「末尾 40ms の pau」「20ms の境界ずれ」「発話中間の pau」の 3 型で
# 実長経路と割れていた（実音声 4 本 × pad 10 段の実測）。今はパイプラインが実長のまま回すので、
# 実重み E2E が固定している `.lab` と**同じもの**が配布形からも出る。
#
# ## `pipelineConfig` の出どころ
#
# 特徴の契約（`sampleRate` / `featureDim` / `classes`）は上流 `feature_config.json` の逐語。
# `maxFrames` は**焼いたグラフの記号次元の上限**で、IR は値域を持たない（`docs/ir-v1.md` の
# `symbols` は名前だけ）ので配布形にしか無い数になる — SBV2 の `maxTokens` / `maxFrames` と
# 同じ持ち方で、台本の定数との一致は `tests/test_dist.py` が突き合わせる。
# 噛み合っていることは {@link assert_vowel_detector_graph} が実測する。

#: パイプライン契約（ADR 0041 §2 — モデル単位）。TS 側の受理集合は
#: `VOWEL_DETECTOR_PIPELINE_NAME` / `VOWEL_DETECTOR_PIPELINE_MAJOR`。
VOWEL_DETECTOR_PIPELINE = "vowel-detector/1"

#: 既定のモデル名 = **チェックポイントの世代**（`export_vowel_detector.default_out_dir` が系列名へ
#: 焼く綴りと同じ導出 — `crnn_epoch3.pt` → `crnn-epoch3`）。学習し直した重みは別のモデル名で
#: 並ぶ（manifest のキーが世代の正本）。
VOWEL_DETECTOR_DEFAULT_MODEL = "crnn-epoch3"

#: 系列名の接頭辞（`vowel-detector-<モデル名>` — 台本の `MODELS_ROOT.name` と同じ 1 語）。
VOWEL_DETECTOR_PREFIX = "vowel-detector"

#: 上流素材の置き場（`inputs/vowel-detector/` — `export_vowel_detector.MODELS_ROOT` と同じ）。
VOWEL_DETECTOR_INPUTS_DIRNAME = "vowel-detector"

#: グラフの役割名（配布形に CRNN は 1 本きり）。配置表・出力 path・weights 宣言が共有する 1 語。
VOWEL_DETECTOR_GRAPH_ROLE = "crnn"

#: `pipelineConfig` に載る**運用上限**（10ms フレーム数 = 10 分）。焼いたグラフの記号次元
#: `T`（20ms 格子）の上限 `export_vowel_detector.SYM_MAX` を入力側の単位へ直したもので、
#: 一致は `tests/test_dist.py` が突き合わせる。
#:
#: MUST: 台本の値と一致させる。IR は記号の値域を持たない（名前だけ）ので、宣言より長い入力を
#: 止められるのは**この数を読むパイプラインだけ**。ずれると超過は配布形の門ではなく利用者の
#: 手元の確保失敗として出る。
VOWEL_DETECTOR_MAX_FRAMES = 60_000

#: グラフ入力の名前（`export_vowel_detector.INPUT_NAME`）と、出力の時間軸の刻み（conv の
#: stride 2 — 入力 2 フレームで出力 1 フレーム）。
VOWEL_DETECTOR_INPUT = "features"
VOWEL_DETECTOR_TIME_STRIDE = 2

#: 時間軸の記号名（`export_vowel_detector` が `convert(symbol_names=("T",))` で焼く綴り）。
#: 入力は `2T`（10ms 格子）・出力は `T`（20ms 格子）で現れる。
VOWEL_DETECTOR_SYMBOL = "T"

#: 特徴の契約の出どころ（上流 `assets/feature_config.json` を `inputs/vowel-detector/` へ手置き）。
#: mel 基底もこの中にあり、配布形へは 1 テンソルの safetensors として移す。
VOWEL_DETECTOR_FEATURE_CONFIG_FILE = "feature_config.json"

#: 配布する mel 基底のテンソルキー（TS 側 `pipeline.ts` の `MEL_BASIS` と対）。
VOWEL_DETECTOR_MEL_BASIS_KEY = "mel_basis"

#: DSP 補助特徴の本数（有声性 / log エネルギー / 零交差率）。`feature_dim = n_mels + 3` の
#: 内部整合を見るためだけに持つ（値の正本は上流の特徴抽出）。
VOWEL_DETECTOR_DSP_DIM = 3

#: 出力クラスの本数（`feature_config.json` の `classes` — 並びは配らない側の関心事で、
#: **並びが id** であることの検査は TS 側 `config.ts` が持つ）。
VOWEL_DETECTOR_CLASS_COUNT = 8


#: 出力の相対 path（**モデルサブツリー内**）— 配置表と manifest が共有する 1 箇所。格納 dtype を
#: ファイル名に出すのは他ファミリと同じ形。
VOWEL_DETECTOR_OUTPUT_PATHS: Mapping[str, str] = {
    VOWEL_DETECTOR_GRAPH_ROLE: "model.f32.safetensors",
    VOWEL_DETECTOR_MEL_BASIS_KEY: f"features/{VOWEL_DETECTOR_MEL_BASIS_KEY}.safetensors",
}

#: 格納 dtype の要求（他ファミリと同じ根拠 — 素の資産が組み立て・ロード・実行を全て通って
#: 参照一致の門まで沈黙した実測事故）。mel 基底はこちらが書く F32 なので載せない。
#:
#: NOTE: 禁止表（{@link assert_storage_absent}）は持たない — 圧縮系列が 1 本も無いので、
#: 「F32 を含む」で系列 × 格納 dtype が一意に決まる。f16 / i8 の席を足すときは**同時に**禁止表も
#: 足す（圧縮系列も適格外の重みを F32 で持つので、存在検査だけでは f32 席へ混入する）。
VOWEL_DETECTOR_STORAGE_REQUIREMENTS: Mapping[str, str] = {VOWEL_DETECTOR_GRAPH_ROLE: "F32"}

#: weights の宣言（dtype ラベル → 役割名）。グラフは 1 本で dtype も 1 つしかないので quant 表は
#: 空でよい（{@link complete_quant_weights} が完全写像へ埋める）。
VOWEL_DETECTOR_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    VOWEL_DETECTOR_GRAPH_ROLE: {"f32": WeightFiles(VOWEL_DETECTOR_GRAPH_ROLE)}
}

#: assets の宣言（quant 選択に依存しない無条件ファイル — 特徴抽出の mel 基底 1 本）。
VOWEL_DETECTOR_ASSETS: Mapping[str, str] = {
    VOWEL_DETECTOR_MEL_BASIS_KEY: VOWEL_DETECTOR_MEL_BASIS_KEY
}

VOWEL_DETECTOR_QUANTS: Mapping[str, Any] = {"f32": {"weights": {}, "session": {}}}
VOWEL_DETECTOR_DEFAULT_QUANT = "f32"


def vowel_detector_series_name(model: str) -> str:
    """モデル名 → 系列ディレクトリ名（`export_vowel_detector.default_out_dir` と同じ式）。"""
    return f"{VOWEL_DETECTOR_PREFIX}-{model}"


def vowel_detector_repo_name(model: str) -> str:
    """配布リポ名。**モデル名を含めない** — チェックポイントの世代（`crnn-epoch3`）はリポの
    名前ではなく manifest のキーが綴る事実で、世代が上がるたびにリポが増える形にしない
    （上流の配布も `vowel-detector` 1 リポ）。
    """
    return f"karume-{VOWEL_DETECTOR_PREFIX}"


@dataclass(frozen=True)
class VowelDetectorSources:
    """組み立ての入力。CRNN 1 本の系列と、上流素材の置き場（`inputs/` — 生成物ではない）。

    後者が要るのは特徴の契約と mel 基底を**焼き込まずに導出**するため（読むのは
    `feature_config.json` 1 本だけ）。
    """

    #: 系列ディレクトリ群の親（`outputs/series/`）— 系列名はここから組む。
    series_dir: Path
    #: 上流素材（`feature_config.json` を置いたディレクトリ）。
    model: Path
    #: 系列を引くモデル名（世代）。
    model_name: str

    @property
    def series(self) -> Path:
        return self.series_dir / vowel_detector_series_name(self.model_name)


def vowel_detector_sources(
    series_dir: Path, model: str = VOWEL_DETECTOR_DEFAULT_MODEL
) -> VowelDetectorSources:
    """系列の親ディレクトリ（`outputs/series/`）と `karume.paths` の綴りから入力を引く。"""
    return VowelDetectorSources(
        series_dir=series_dir,
        model=INPUTS_ROOT / VOWEL_DETECTOR_INPUTS_DIRNAME,
        model_name=model,
    )


def vowel_detector_placements(sources: VowelDetectorSources) -> dict[str, Path]:
    """役割名 → 出所のファイル。出力の path は {@link VOWEL_DETECTOR_OUTPUT_PATHS} が持つ。

    この表に無いものは出力へ入らない（系列に並ぶ `io.*.safetensors` はこれで落ちる）。
    mel 基底は配置ではなく変換の出力なので、ここには現れない（SBV2 の表 2 本と同じ形）。
    """
    return {VOWEL_DETECTOR_GRAPH_ROLE: sources.series / "model.safetensors"}


def vowel_detector_feature_config(model_dir: Path) -> Mapping[str, Any]:
    """上流の `feature_config.json` を読む（特徴の契約と mel 基底の唯一の出どころ）。"""
    path = model_dir / VOWEL_DETECTOR_FEATURE_CONFIG_FILE
    if not path.is_file():
        raise DistError(
            f"組み立ての入力が無い: {path}"
            "（上流 vowel-detector の assets/feature_config.json をここへ置く）"
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DistError(f"{path} が JSON として読めない") from error
    if not isinstance(raw, dict):
        raise DistError(f"{path}: 最上位オブジェクトでない")
    return raw


def _vowel_detector_int(raw: Mapping[str, Any], key: str, where: str) -> int:
    """特徴 config の正整数フィールドを検査して読む。"""
    value = raw.get(key)
    # bool は int の派生。`"n_mels": true` を 1 として通すと寸法の突合が緩む。
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise DistError(f"{where} の {key} が正の整数でない（{value!r}）")
    return value


def vowel_detector_classes(raw: Mapping[str, Any], where: str) -> list[str]:
    """クラス語彙を読む（**並びのまま**配る — 並びがそのままクラス id）。

    NOTE: 語彙そのもの（`a`/`i`/…/`cons` の綴りと並び）はここが持たない。上流 config が唯一の
    出どころで、それを写した 2 つ目の表をここに置くと、片方だけ動いたときに「宣言は通るのに
    ラベルが置換されている」形が黙って作れる。**受理集合の検査はロード側**
    （`packages/models/src/vowel-detector/config.ts`）が持ち、ここは構造だけを見る。
    """
    value = raw.get("classes")
    if not isinstance(value, list) or len(value) != VOWEL_DETECTOR_CLASS_COUNT:
        raise DistError(
            f"{where} の classes が長さ {VOWEL_DETECTOR_CLASS_COUNT} の配列でない（{value!r}）"
        )
    for entry in value:
        if not isinstance(entry, str) or not entry:
            raise DistError(f"{where} の classes に空でない文字列でない要素がある（{value!r}）")
    if len(set(value)) != len(value):
        raise DistError(f"{where} の classes に重複がある（{value!r}）")
    return list(value)


def vowel_detector_mel_basis(raw: Mapping[str, Any], where: str) -> np.ndarray:
    """mel 基底 `[n_mels, n_fft // 2 + 1]` を f32 の行列として読む。

    MUST: 形は `n_mels` と `n_fft` の**両方**から組んだ期待と突き合わせる — 基底がずれても
    特徴は「それらしい別の値」になるだけで、shape も値域も合ったまま最後まで通る。

    MUST: 全行に正の要素があることまで見る。空の三角窓（帯域外へ落ちた mel チャネル）は
    その列を常に `log(1e-5)` に張り付かせるが、80 本の中の 1 本なので目視でも数値でも
    「静かにおかしい」形にしかならない。
    """
    n_mels = _vowel_detector_int(raw, "n_mels", where)
    n_fft = _vowel_detector_int(raw, "n_fft", where)
    expected = (n_mels, n_fft // 2 + 1)
    value = raw.get(VOWEL_DETECTOR_MEL_BASIS_KEY)
    if not isinstance(value, list):
        raise DistError(f"{where} の {VOWEL_DETECTOR_MEL_BASIS_KEY} が配列でない（{type(value)}）")
    try:
        basis = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise DistError(
            f"{where} の {VOWEL_DETECTOR_MEL_BASIS_KEY} が f32 行列にならない"
        ) from error
    if basis.shape != expected:
        raise DistError(
            f"{where} の {VOWEL_DETECTOR_MEL_BASIS_KEY} の形が {basis.shape}、"
            f"n_mels / n_fft から組んだ期待は {expected}"
        )
    if not np.isfinite(basis).all():
        raise DistError(f"{where} の {VOWEL_DETECTOR_MEL_BASIS_KEY} に有限でない要素がある")
    empty = [int(row) for row in np.flatnonzero(basis.max(axis=1) <= 0)]
    if empty:
        raise DistError(
            f"{where} の {VOWEL_DETECTOR_MEL_BASIS_KEY} に空の mel チャネルがある（行 {empty}）"
        )
    return basis


def vowel_detector_pipeline_config(feature_config: Mapping[str, Any], where: str) -> dict[str, Any]:
    """`pipelineConfig`（TS 側スキーマの 4 欄）を上流 config と台本の宣言から組む。

    `maxFrames` だけが**焼いたグラフ側の数**（{@link VOWEL_DETECTOR_MAX_FRAMES}）で、
    残り 3 欄は上流 config の逐語。
    """
    sample_rate = _vowel_detector_int(feature_config, "sample_rate", where)
    feature_dim = _vowel_detector_int(feature_config, "feature_dim", where)
    n_mels = _vowel_detector_int(feature_config, "n_mels", where)
    if feature_dim != n_mels + VOWEL_DETECTOR_DSP_DIM:
        raise DistError(
            f"{where} の feature_dim {feature_dim} が n_mels {n_mels} +"
            f" DSP {VOWEL_DETECTOR_DSP_DIM} と違う — 特徴の内訳が上流と食い違っている"
        )
    return {
        "sampleRate": sample_rate,
        "featureDim": feature_dim,
        "classes": vowel_detector_classes(feature_config, where),
        "maxFrames": VOWEL_DETECTOR_MAX_FRAMES,
    }


def assert_vowel_detector_graph(
    placements: Mapping[str, Path], pipeline_config: Mapping[str, Any]
) -> None:
    """CRNN グラフの入出力が `pipelineConfig` と噛み合うことを、配置の前に実測する。

    MUST: 落とさない。特徴次元とクラス数は上流 config 由来、形はグラフ由来で**別々に決まる**
    ので、別の特徴で学習された派生の重みと今の config を組み合わせても、ここまでは何も落ちない。

    MUST: 時間軸が**記号**（入力 `2T` / 出力 `T`）であることまで見る。長さを固定して焼いた
    古い形のグラフは名前も階数も同じで、載せてしまうと「その 1 長でしか動かない配布形」が
    manifest としては成立する（利用者の手元では、その長さ以外の全ての音声で落ちる）。
    """
    path = placements[VOWEL_DETECTOR_GRAPH_ROLE]
    graph = ir_graph(path)
    inputs = graph_inputs(graph, path)
    if tuple(inputs) != (VOWEL_DETECTOR_INPUT,):
        raise DistError(
            f"{path} のグラフ入力が {list(inputs)} で、期待の {[VOWEL_DETECTOR_INPUT]} と違う"
            " — 実行側は名前で束ねるので、綴りが変われば束ねられない"
        )
    if graph.get("symbols") != [VOWEL_DETECTOR_SYMBOL]:
        raise DistError(
            f"{path}: 記号次元が {graph.get('symbols')!r} — 期待は"
            f" [{VOWEL_DETECTOR_SYMBOL!r}]（時間軸 1 本だけが記号）"
        )
    feature_dim = pipeline_config["featureDim"]
    stride_dim = f"{VOWEL_DETECTOR_TIME_STRIDE}{VOWEL_DETECTOR_SYMBOL}"
    expected_input = [1, stride_dim, feature_dim]
    if inputs[VOWEL_DETECTOR_INPUT] != expected_input:
        raise DistError(
            f"{path} の入力 '{VOWEL_DETECTOR_INPUT}' が"
            f" {inputs[VOWEL_DETECTOR_INPUT]!r}、期待は {expected_input!r}"
            " — 10ms 格子の長さは記号 T の 2 倍（batch は静的 1）"
        )
    outputs = graph.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise DistError(f"{path}: グラフ出力が {outputs!r} — ロジット 1 本だけが要る")
    values = graph.get("values")
    value = values.get(outputs[0]) if isinstance(values, dict) else None
    out_shape = value.get("shape") if isinstance(value, dict) else None
    expected_output = [1, VOWEL_DETECTOR_SYMBOL, len(pipeline_config["classes"])]
    if out_shape != expected_output:
        raise DistError(
            f"{path}: グラフ出力 '{outputs[0]}' の形が {out_shape!r}、期待は {expected_output!r}"
            " — 出力は 20ms 格子（入力 2 フレームで 1 本）× クラス数"
        )


def vowel_detector_plan(
    sources: VowelDetectorSources, model: str = VOWEL_DETECTOR_DEFAULT_MODEL
) -> ModelPlan:
    """母音検出 1 モデルぶんの計画を組む（検査と config の読み取りをここで全部済ませる）。"""
    assert_model_name(model)
    placements = vowel_detector_placements(sources)
    for role, source in placements.items():
        assert_storage(role, source, VOWEL_DETECTOR_STORAGE_REQUIREMENTS)
    feature_config = vowel_detector_feature_config(sources.model)
    where = str(sources.model / VOWEL_DETECTOR_FEATURE_CONFIG_FILE)
    pipeline_config = vowel_detector_pipeline_config(feature_config, where)
    assert_vowel_detector_graph(placements, pipeline_config)
    artifacts = {
        role: Artifact(VOWEL_DETECTOR_OUTPUT_PATHS[role], source=source)
        for role, source in placements.items()
    }
    artifacts[VOWEL_DETECTOR_MEL_BASIS_KEY] = Artifact(
        VOWEL_DETECTOR_OUTPUT_PATHS[VOWEL_DETECTOR_MEL_BASIS_KEY],
        payload=table_payload(
            VOWEL_DETECTOR_MEL_BASIS_KEY, vowel_detector_mel_basis(feature_config, where)
        ),
    )
    return ModelPlan(
        name=model,
        pipeline=VOWEL_DETECTOR_PIPELINE,
        artifacts=artifacts,
        weights=VOWEL_DETECTOR_WEIGHTS,
        assets=VOWEL_DETECTOR_ASSETS,
        quants=complete_quant_weights(VOWEL_DETECTOR_WEIGHTS, VOWEL_DETECTOR_QUANTS),
        default_quant=VOWEL_DETECTOR_DEFAULT_QUANT,
        pipeline_config=pipeline_config,
    )


# ---- ⑨ Depth Anything V2（depth-estimation）----------------------------------
#
# 配布するのは **深度マップ 1 グラフだけ**（`export_depth_anything.py` は `predicted_depth`
# 1 本しか出さない）。実行に要る資産もそれ 1 本で、tokenizer も表も無い（`assets` は空）。
# 格納 dtype は f32 の 1 系列だけなので quant 席も 1 つ — ここまでは SigLIP2 / BiRefNet と同じ形。
#
# **1 リポ 1 モデル = 1 サイズ**（`karume-depth-anything-v2-<サイズ>`）。サイズは重みも
# 埋め込み幅も違う別物で、同居させると利用者が何も指定しなかったときに引く既定が丸ごと変わる。
#
# MUST: 配れるのは**帰属表（`karume.modelcard.DEPTH_ANYTHING_UPSTREAM`）に載っているサイズだけ**。
# 上流で Apache-2.0 なのは Small のみで、Base / Large は CC BY-NC 4.0（同表の MUST に実地確認の
# 日付つき）。台本（`export_depth_anything.py`）は `--model-dir` でどのサイズも焼けるので、
# 「焼けたものは配れる」と読める形にしない — この表が唯一の門。
#
# `pipelineConfig` の数は **2 つの独立した出どころ**から来る: 前処理の定数（寸法・統計・
# フィルタ）は上流の `preprocessor_config.json`、噛み合っているかは**焼かれたグラフの入出力
# 宣言**。どちらも写経しない（TS 側の正本 `packages/models/src/depth-anything/config.ts` の
# モジュール doc と同じ理由）。ずれた組み合わせは「resize 先だけが違う」形で値が静かに崩れる
# ので、{@link assert_depth_anything_graph} が配置の前に実測する。

#: パイプライン契約（ADR 0041 §2 — モデル単位）。TS 側の受理集合は
#: `DEPTH_ANYTHING_PIPELINE_NAME` / `DEPTH_ANYTHING_PIPELINE_MAJOR`。
DEPTH_ANYTHING_PIPELINE = "depth-anything/1"

#: 既定のモデル名（= 既定のリポ名 `karume-depth-anything-v2-small` の末尾）。綴りの受理集合は
#: 帰属表（`karume.modelcard.DEPTH_ANYTHING_UPSTREAM`）が持つ。
DEPTH_ANYTHING_DEFAULT_MODEL = "small"

#: リポ名の接頭辞（`karume-depth-anything-v2-<サイズ>`）。系列名は別（上流リポ名の小文字化 —
#: {@link depth_anything_series_name}）。V2 まで綴りへ入れるのは、
#: V3 が別アーキ（多視点 + カメラ姿勢の DualDPT）で単一画像 depth の後継ではないため —
#: 世代を落とすと「新しい方が良い」と読める並びになる。
DEPTH_ANYTHING_PREFIX = "depth-anything-v2"

#: 実重みと `preprocessor_config.json` の親（`hf download depth-anything/<名前> --local-dir
#: inputs/depth-anything/<名前>` の展開先 — `export_depth_anything.MODELS_ROOT` と同じ場所）。
DEPTH_ANYTHING_INPUTS_DIRNAME = "depth-anything"

#: 唯一の役割名（manifest の weights キー・TS 側 `pipeline.ts` の `DEPTH`）。
DEPTH_ANYTHING_ROLE = "depth"

#: グラフ入力の名前（`export_depth_anything.INPUT_NAME`）。
DEPTH_ANYTHING_INPUT = "pixel_values"

#: 入力のチャネル数（RGB）。出力は**チャネル軸を持たない** `[1, H, W]` の地図。
DEPTH_ANYTHING_CHANNELS = 3

#: 前処理定数の出どころ（重みと同じディレクトリ）。
DEPTH_ANYTHING_PREPROCESSOR_FILE = "preprocessor_config.json"

#: TS 側が実装している補間と、それに対応する `resample` の値（**PIL の定数で BICUBIC**）。
#: SigLIP2 / BiRefNet の bilinear（2）と**違う**ので、TS 側 `config.ts` の受理集合も 1 値だけ
#: bicubic に絞ってある。`export_depth_anything.check_processor` が emit のたびに現物から実測する
#: のと同じ対象を、こちらは配布形の宣言として書く。
DEPTH_ANYTHING_INTERPOLATION = "bicubic"
DEPTH_ANYTHING_RESAMPLE = 3

#: TS 側 `normalizeToNchw` が持つ除数 255 の逆数。**pipelineConfig には載せない**（実行時に
#: 選べない数を宣言だけ持たせると正本が 2 つになる — `config.ts` の NOTE）ので、違う値の
#: チェックポイントはここで落とす。
DEPTH_ANYTHING_RESCALE_FACTOR = 1.0 / 255.0

#: 出力の相対 path（**モデルサブツリー内**）— 配置表と manifest が共有する 1 箇所。
DEPTH_ANYTHING_OUTPUT_PATHS: Mapping[str, str] = {
    DEPTH_ANYTHING_ROLE: f"{DEPTH_ANYTHING_ROLE}/model.f32.safetensors"
}

#: 格納 dtype の要求（Anima / SBV2 / Irodori / SigLIP2 / BiRefNet と同じ根拠 — 素の資産が
#: 組み立て・ロード・実行を全て通って参照一致の門まで沈黙した実測事故）。
#:
#: NOTE: 禁止表（{@link assert_storage_absent}）は持たない — 圧縮系列が 1 本も無いので、
#: 「F32 を含む」で系列 × 格納 dtype が一意に決まる。f16 / i8 の席を足すときは**同時に**禁止表も
#: 足す（圧縮系列も適格外の重みを F32 で持つので、存在検査だけでは f32 席へ混入する）。
DEPTH_ANYTHING_STORAGE_REQUIREMENTS: Mapping[str, str] = {DEPTH_ANYTHING_ROLE: "F32"}

#: weights の宣言（dtype ラベル → 役割名）。dtype が 1 つしかないので quant 表は空でよい
#: （{@link complete_quant_weights} が完全写像へ埋める）。
DEPTH_ANYTHING_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    DEPTH_ANYTHING_ROLE: {"f32": WeightFiles(DEPTH_ANYTHING_ROLE)}
}

#: assets の宣言。**空**（実行に要るのはグラフ 1 本だけ）。
DEPTH_ANYTHING_ASSETS: Mapping[str, str] = {}

DEPTH_ANYTHING_QUANTS: Mapping[str, Any] = {"f32": {"weights": {}, "session": {}}}
DEPTH_ANYTHING_DEFAULT_QUANT = "f32"


def depth_anything_checkpoint(model: str) -> str:
    """モデル名 → 上流チェックポイントのディレクトリ名（= 上流の HF リポ名の末尾）。

    `export_depth_anything.py` は `--model-dir` のディレクトリ名を小文字化して系列名にし、
    そのディレクトリ名は `hf download <リポ>` の展開先なので、綴りの事実は「このモデルが
    どの上流リポか」1 つしかない。帰属表（`karume.modelcard.DEPTH_ANYTHING_UPSTREAM`）から
    導いて、ここに 2 つ目の表を持たない（SigLIP2 の {@link siglip2_checkpoint} と同じ規律）。

    MUST: 表に無いサイズはここで落ちる — それが「Apache-2.0 のものしか配らない」門の実体
    （節の冒頭の MUST）。
    """
    repo = DEPTH_ANYTHING_UPSTREAM.get(model)
    if repo is None:
        raise DistError(
            f"Depth Anything V2 のモデル '{model}' は知らない"
            f"（既知: {' / '.join(sorted(DEPTH_ANYTHING_UPSTREAM))}）"
            " — 上流で Apache-2.0 のサイズだけが帰属表に載っている"
        )
    return repo.split("/", 1)[1]


def depth_anything_series_name(model: str) -> str:
    """モデル名 → 系列ディレクトリ名（`export_depth_anything.default_out_dir` と同じ式）。"""
    return depth_anything_checkpoint(model).lower()


def depth_anything_repo_name(model: str) -> str:
    """単一モデルの配布リポ名（`karume-` prefix はリポ名裁定 2026-08-09）。"""
    return f"karume-{DEPTH_ANYTHING_PREFIX}-{model}"


@dataclass(frozen=True)
class DepthAnythingSources:
    """組み立ての入力。系列（グラフ 1 本）と実重みの置き場（`inputs/` — 生成物ではない）。

    後者が要るのは `pipelineConfig` の前処理定数を**焼き込まずに導出**するため
    （読むのは `preprocessor_config.json` 1 本だけなので、99MB の重みには触らない）。
    """

    series: Path
    model: Path


def depth_anything_sources(
    series_dir: Path, model: str = DEPTH_ANYTHING_DEFAULT_MODEL
) -> DepthAnythingSources:
    """系列の親ディレクトリ（`outputs/series/`）と `karume.paths` の綴りから入力を引く。"""
    checkpoint = depth_anything_checkpoint(model)
    return DepthAnythingSources(
        series=series_dir / depth_anything_series_name(model),
        model=INPUTS_ROOT / DEPTH_ANYTHING_INPUTS_DIRNAME / checkpoint,
    )


def depth_anything_placements(sources: DepthAnythingSources) -> dict[str, Path]:
    """役割名 → 出所のファイル。出力の path は {@link DEPTH_ANYTHING_OUTPUT_PATHS} が持つ。

    この表に無いものは出力へ入らない（系列に並ぶ `io.*.safetensors` はこれで落ちる）。
    """
    return {DEPTH_ANYTHING_ROLE: sources.series / "model.safetensors"}


def depth_anything_preprocessor(model_dir: Path) -> Mapping[str, Any]:
    """上流の `preprocessor_config.json` を読む（前処理定数の唯一の出どころ）。"""
    path = model_dir / DEPTH_ANYTHING_PREPROCESSOR_FILE
    if not path.is_file():
        raise DistError(f"組み立ての入力が無い: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DistError(f"{path} が JSON として読めない") from error
    if not isinstance(raw, dict):
        raise DistError(f"{path}: 最上位オブジェクトでない")
    return raw


def depth_anything_pipeline_config(preprocessor: Mapping[str, Any], where: str) -> dict[str, Any]:
    """`pipelineConfig`（TS 側スキーマの 5 欄）を上流の前処理 config から組む。

    MUST: 対応外の前処理は**受理しない**（黙って近似しない）。TS 側が持つのは
    「antialias 付き bicubic → `/255` → `(x − mean) / std`」の 1 本きりなので、
    `resample` / `rescale_factor` / `do_*` が外れたチェックポイントは、値がずれたまま
    ロードも実行も通る。ここが唯一の検出器になる。

    NOTE: `keep_aspect_ratio` / `ensure_multiple_of` は**読むが宣言へ写さない**。焼かれた
    グラフが正方 1 点でしか受け取らないので（DINOv2 の位置埋め込みがパッチ格子に紐づいて
    いる）、アスペクト比を保つ経路には行き先が無く、宣言だけ置くと「保てるのに保っていない」
    と読める欄になる。事実はモデルカードの散文が持つ（`_depth_anything_shape`）。
    """
    size = preprocessor.get("size")
    if not isinstance(size, dict):
        raise DistError(f"{where} に size が無い / オブジェクトでない（{size!r}）")
    resample = preprocessor.get("resample")
    if resample != DEPTH_ANYTHING_RESAMPLE:
        raise DistError(
            f"{where} の resample が {resample!r}"
            f"（期待 {DEPTH_ANYTHING_RESAMPLE} = PIL の BICUBIC）"
            f" — TS 側の前処理は {DEPTH_ANYTHING_INTERPOLATION} の 1 本しか持たない"
        )
    rescale = preprocessor.get("rescale_factor")
    if rescale != DEPTH_ANYTHING_RESCALE_FACTOR:
        raise DistError(
            f"{where} の rescale_factor が {rescale!r}（期待 {DEPTH_ANYTHING_RESCALE_FACTOR}）"
            " — TS 側は 8bit 画素を 255 で割る形で閉じている"
        )
    for flag in ("do_resize", "do_rescale", "do_normalize"):
        if preprocessor.get(flag) is not True:
            raise DistError(f"{where} の {flag} が真でない — 前処理の 3 段が揃っていない")
    for flag in ("do_center_crop", "do_pad"):
        if preprocessor.get(flag):
            raise DistError(f"{where} の {flag} が真 — TS 側は crop も pad も持たない")
    return {
        "imageWidth": _preprocessor_int(size, "width", f"{where} の size"),
        "imageHeight": _preprocessor_int(size, "height", f"{where} の size"),
        "imageMean": _preprocessor_channels(
            preprocessor, "image_mean", where, channels=DEPTH_ANYTHING_CHANNELS, positive=False
        ),
        "imageStd": _preprocessor_channels(
            preprocessor, "image_std", where, channels=DEPTH_ANYTHING_CHANNELS, positive=True
        ),
        "interpolation": DEPTH_ANYTHING_INTERPOLATION,
    }


def assert_depth_anything_graph(
    graph: Mapping[str, Any], path: Path, pipeline_config: Mapping[str, Any]
) -> None:
    """グラフの入出力が**前処理の宣言と噛み合う**ことを、配置の前に実測する。

    MUST: 落とさない。前処理の寸法（`preprocessor_config.json`）と焼かれた解像度（重みの
    `config.json` の `image_size`）は別々に決まるので、別サイズの組み合わせでもここまでは
    何も落ちない — ずれたまま配ると、利用者の手元で Session の shape 検査が「どちらの数が
    正しいのか」を伝えないまま落ちる。

    MUST: 出力の**階数**まで見る。深度地図は `[1, H, W]`（チャネル軸を持たない）で、
    `[1, 1, H, W]` と要素数では区別できない — 位置で引く後段は要素数しか見ないので、
    チャネル軸つきで焼かれたグラフは黙って通ってしまう。
    """
    inputs = graph_inputs(graph, path)
    if tuple(inputs) != (DEPTH_ANYTHING_INPUT,):
        raise DistError(
            f"{path} のグラフ入力が {list(inputs)} で、期待の {[DEPTH_ANYTHING_INPUT]} と違う"
            " — 実行側は名前で束ねるので、綴りが変われば束ねられない"
        )
    symbols = graph.get("symbols")
    if not isinstance(symbols, list) or symbols:
        raise DistError(
            f"{path}: 記号次元 {symbols!r} がある — 解像度もパッチ数も固定で、動かす軸は無い"
        )
    height, width = pipeline_config["imageHeight"], pipeline_config["imageWidth"]
    expected_input = [1, DEPTH_ANYTHING_CHANNELS, height, width]
    if inputs[DEPTH_ANYTHING_INPUT] != expected_input:
        raise DistError(
            f"{path} の入力 '{DEPTH_ANYTHING_INPUT}' が {inputs[DEPTH_ANYTHING_INPUT]!r}、"
            f"{DEPTH_ANYTHING_PREPROCESSOR_FILE} の size から組んだ期待は {expected_input}"
            " — 前処理の寸法と焼かれた解像度が別の版"
        )
    outputs = graph.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise DistError(f"{path}: グラフ出力が {outputs!r} — 深度マップ 1 本だけが要る")
    values = graph.get("values")
    value = values.get(outputs[0]) if isinstance(values, dict) else None
    shape = value.get("shape") if isinstance(value, dict) else None
    expected_output = [1, height, width]
    if shape != expected_output:
        raise DistError(
            f"{path}: グラフ出力 '{outputs[0]}' の形が {shape!r}、期待は {expected_output}"
            " — 深度マップは入力と同じ寸法・チャネル軸なし"
        )


def depth_anything_plan(
    sources: DepthAnythingSources, model: str = DEPTH_ANYTHING_DEFAULT_MODEL
) -> ModelPlan:
    """Depth Anything V2 の 1 モデルぶんの計画を組む（検査と読み取りをここで全部済ませる）。"""
    assert_model_name(model)
    # 帰属を名乗れないサイズは系列を読む前に落とす（節の冒頭の MUST）。
    depth_anything_checkpoint(model)
    placements = depth_anything_placements(sources)
    for role, source in placements.items():
        assert_storage(role, source, DEPTH_ANYTHING_STORAGE_REQUIREMENTS)
    container = placements[DEPTH_ANYTHING_ROLE]
    graph = ir_graph(container)
    pipeline_config = depth_anything_pipeline_config(
        depth_anything_preprocessor(sources.model),
        str(sources.model / DEPTH_ANYTHING_PREPROCESSOR_FILE),
    )
    assert_depth_anything_graph(graph, container, pipeline_config)
    return ModelPlan(
        name=model,
        pipeline=DEPTH_ANYTHING_PIPELINE,
        artifacts={
            role: Artifact(DEPTH_ANYTHING_OUTPUT_PATHS[role], source=source)
            for role, source in placements.items()
        },
        weights=DEPTH_ANYTHING_WEIGHTS,
        assets=DEPTH_ANYTHING_ASSETS,
        quants=complete_quant_weights(DEPTH_ANYTHING_WEIGHTS, DEPTH_ANYTHING_QUANTS),
        default_quant=DEPTH_ANYTHING_DEFAULT_QUANT,
        pipeline_config=pipeline_config,
    )


# ---- ⑩ pipeline 別ディスパッチと CLI -----------------------------------------


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


def siglip2_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """`--series` の親から SigLIP2 1 モデルの計画を組む（CLI のディスパッチ先）。"""
    return siglip2_plan(siglip2_sources(series_dir, model), model)


def birefnet_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """`--series` の親から BiRefNet 系 1 モデルの計画を組む（CLI のディスパッチ先）。"""
    return birefnet_plan(birefnet_sources(series_dir, model), model)


def vowel_detector_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """`--series` の親から母音検出 1 モデルの計画を組む（CLI のディスパッチ先）。"""
    return vowel_detector_plan(vowel_detector_sources(series_dir, model), model)


def depth_anything_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """`--series` の親から Depth Anything V2 の 1 モデルの計画を組む（CLI のディスパッチ先）。"""
    return depth_anything_plan(depth_anything_sources(series_dir, model), model)


#: core wheel だけで組める pipeline（全量ではない — モジュール doc の MUST）。
PIPELINES: Mapping[str, Pipeline] = {
    "siglip2": Pipeline(
        default_model=SIGLIP2_DEFAULT_MODEL,
        repo_name=siglip2_repo_name,
        plan=siglip2_dist_plan,
        # 帰属は**モデル名から一意に決まる**（`SIGLIP2_UPSTREAM` — base / so400m は別リポの
        # 重み）ので、選ばせる軸にしない。プロファイルを分けると「so400m を base の帰属で
        # 配る」取り違えを操作者が起こせるようになる。
        card_profiles={"siglip2": render_siglip2_model_card},
    ),
    "birefnet": Pipeline(
        default_model=BIREFNET_DEFAULT_MODEL,
        repo_name=birefnet_repo_name,
        plan=birefnet_dist_plan,
        # SigLIP2 と同じ理由で選ばせる軸にしない — 帰属（上流リポ・ライセンス・学習データ）は
        # モデル名から一意に決まる。プロファイルを分けると「Lucida を BiRefNet_HR の帰属で
        # 配る」取り違えを操作者が起こせるようになる。
        card_profiles={"birefnet": render_birefnet_model_card},
    ),
    "depth-anything": Pipeline(
        default_model=DEPTH_ANYTHING_DEFAULT_MODEL,
        repo_name=depth_anything_repo_name,
        plan=depth_anything_dist_plan,
        # SigLIP2 / BiRefNet と同じ理由で選ばせる軸にしない — 帰属（上流リポ・ライセンス）は
        # モデル名から一意に決まる。プロファイルを分けると「Base を Small の帰属で配る」
        # 取り違えを操作者が起こせるようになる（しかも Base は CC BY-NC 4.0）。
        card_profiles={"depth-anything": render_depth_anything_model_card},
    ),
    "vowel-detector": Pipeline(
        default_model=VOWEL_DETECTOR_DEFAULT_MODEL,
        repo_name=vowel_detector_repo_name,
        plan=vowel_detector_dist_plan,
        # 帰属は 1 通りだけ（上流 1 リポ・1 ライセンス — 学習素材の帰属も重みに紐づいた
        # 1 組）。選択肢が無いので省略で通る。
        card_profiles={"vowel-detector": render_vowel_detector_model_card},
    ),
}


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


def build_parser(
    pipelines: Mapping[str, Pipeline] = PIPELINES, default_pipeline: str | None = None
) -> argparse.ArgumentParser:
    """`--pipeline` の受理集合を**引数で**受ける（ドライバが recipe を足した表を渡す）。

    `default_pipeline` を渡さない呼び出し（= core 単体）は `--pipeline` を必須にする
    （{@link main} が選択肢を並べて落とす）。
    """
    parser = argparse.ArgumentParser(description="配布ディレクトリ（HF アップ可能形）の組み立て")
    default_note = "必須" if default_pipeline is None else f"既定: {default_pipeline}"
    parser.add_argument(
        "--pipeline",
        choices=sorted(pipelines),
        default=default_pipeline,
        help=f"組み立てるパイプライン（{default_note}）",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        metavar="NAME",
        help="組み立てるモデル名（繰り返すと 1 リポへまとめて組む = ファミリー組み立て。"
        "最初の 1 つが defaultModel。既定: "
        + " / ".join(f"{name}={spec.default_model}" for name, spec in pipelines.items())
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
            f"{name}={'|'.join(sorted(spec.card_profiles))}" for name, spec in pipelines.items()
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


def main(
    argv: Sequence[str] | None = None,
    *,
    pipelines: Mapping[str, Pipeline] = PIPELINES,
    default_pipeline: str | None = None,
) -> None:
    args = build_parser(pipelines, default_pipeline).parse_args(argv)
    if args.pipeline is None:
        # MUST: 既定を core に焼かない（モジュール doc の MUST）— 表が呼び出し側で変わる。
        raise DistError(
            f"--pipeline が要る（選択肢: {', '.join(sorted(pipelines))}）— "
            "リポ専用 recipe まで含めた表は tools/export-recipes/dist.py が持つ"
        )
    pipeline = pipelines[args.pipeline]
    models = args.models if args.models else [pipeline.default_model]
    out_dir = args.out if args.out is not None else default_out_dir(pipeline, models)
    # 帰属プロファイルは**組み立ての前**に解決する — 誤った / 足りない指定で数 GB を並べてから
    # 最後の 1 枚で落ちる形にしない。
    render_card = resolve_card_renderer(pipeline, args.card_profile)
    # MUST: 全モデルの計画（= 検査と読み取り）を配置の**前**に済ませる — 2 モデル目で落ちる
    # 形でも、1 モデル目だけ入った配布形を後段に見せない。
    plans = [pipeline.plan(args.series, model) for model in models]
    # モデルカードは組み立ての中（staging）で、**検証を通った manifest** から描く（表と現物が
    # 食い違ったまま説明だけ生えることがない順序・カードごと 1 回で据わる）。リポ ID は組み立て
    # 先のディレクトリ名から導く — manifest は自分の在り処を知らず、ファミリーリポの名前は
    # pipeline の定数にもできない。
    manifest = assemble_family(
        plans,
        out_dir,
        models[0],
        render_card=partial(render_card, repo=f"{HF_OWNER}/{out_dir.name}"),
    )
    verified = verify_dist(out_dir)
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
