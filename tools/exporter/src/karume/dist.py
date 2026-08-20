"""配布ディレクトリの組み立て — 系列ディレクトリ群 → HF へそのまま上げられる 1 リポ形。

仕様の正本は ADR 0041（`docs/decisions/0041-manifest-v2.md`）。ここが作るのは §2 の形で
並んだファイル群と、それを宣言する `karume.json`（`karume/3`）、そして manifest から機械導出
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

MUST: **置き場の既定も呼び出し側が渡す**（{@link main} の `default_out_dir` / `default_series`）
— リポの `models/` や `outputs/series/` は repo topology であって汎用 exporter の知識ではない
（ADR 0065 Consequences — 綴りは `tools/export-recipes/_shared/paths.py` が 1 箇所で持つ）。
渡されなければ core 単体でも `--out` / `--series` を必須にして落とす（黙って cwd 相対の
どこかへ組まない）。

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

受理集合が空の core 単体では 1 つも組めないので、使い方はリポの dist ドライバ側で綴る
（`tools/export-recipes/dist.py` のモジュール doc）:

    uv run python dist.py --pipeline siglip2               # export-recipes ルートで
    uv run python dist.py --pipeline siglip2 --model so400m
    uv run python dist.py --pipeline birefnet --model lucida

`--card-profile` はモデルカードの**帰属**（出所・ライセンス・引用）の選択で、選択肢が 2 つ
以上ある pipeline では必須（{@link resolve_card_renderer}）。組み立てる資産には掛からない —
同じ重みでも、どのファミリーとして配るかで帰属が変わる。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
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

from karume.artifacts import ArtifactSwapError, staged_publication
from karume.ir import IR_METADATA_KEY
from karume.modelcard import HF_OWNER

# ---- ① 共有部: 置き場の綴り・共有席の決定・配置・ハッシュ・宣言と現物の突合 -------

#: manifest のファイル名（ADR 0041 §1 — リポジトリ直下の固定名）。
MANIFEST_FILENAME = "karume.json"

#: manifest の形式識別子（ADR 0041 §1 — hub は 1 形しか読まない）。`karume/3` は weights の
#: dtype エントリを **shard 列**（`{shards, extras?}`）にした形で、単一ファイルの資産は
#: 1 要素の列として宣言される（ADR 0070 決定 1 の shard 欄の確定）。
MANIFEST_FORMAT = "karume/3"

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
#: 1 dtype エントリが並べられる shard 数（`karume/3`）。exporter は分割規則を持たない
#: （常に 1 要素を書く）が、検査は hub と同じ値で弾く — {@link verify_dist} は手元のどの
#: 配布形にも掛けられる門なので、上限を知らない検査になっていると v3 の受理集合が 2 つに割れる。
MAX_SHARDS = 1024
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
    """
    if not source.is_file():
        raise DistError(f"組み立ての入力が無い: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
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
    """`Artifact` の実体を `dest` に作る（配置 or 書き出し）。"""
    if artifact.source is not None:
        place_file(artifact.source, dest)
        return
    assert artifact.payload is not None  # __post_init__ の不変条件
    dest.parent.mkdir(parents=True, exist_ok=True)
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


def preprocessor_int(raw: Mapping[str, Any], key: str, where: str) -> int:
    """前処理 config の整数フィールド（寸法）を検査して読む。

    HF の `preprocessor_config.json` を読む recipe（SigLIP2 / Depth Anything V2）で共有する —
    値域は TS 側の各 `config.ts` と同じで、独立に動く写しを増やすと片方だけ緩む。モデル固有の
    知識を 1 つも持たない読み手なので core 側に置く（境界跨ぎに private 名を使わない）。
    """
    value = raw.get(key)
    # bool は int の派生。`"height": true` を 1 として通すと寸法の突合が緩む。
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise DistError(f"{where} の {key} が正の整数でない（{value!r}）")
    return value


def preprocessor_channels(
    raw: Mapping[str, Any], key: str, where: str, *, channels: int, positive: bool
) -> list[float]:
    """`image_mean` / `image_std` を検査して読む（TS 側 `config.ts` と同じ値域）。

    `std` に 0 を通さないのは、0 除算が例外を出さずに `±Infinity` の `pixel_values` を作るため。
    チャネル数は**呼び出し側の recipe が持つ定数**を受ける（読み手だけを共有し、各 recipe の
    宣言はそれぞれの席に残す）。
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


@dataclass(frozen=True)
class WeightFiles:
    """weights の 1 dtype ぶん（`karume/3` の `{shards, extras?}`）。中身は**役割名**。

    実 path は {@link Artifact} 側が持つ — 共有の畳み込みで path が `shared/…` へ動くので、
    宣言側が path を直に握っていると 2 箇所が独立に動く。

    `file` が単数なのは、**exporter に shard 分割規則を持たせない**ため（ADR 0070 決定 1 —
    分割の実需は LLM 配布で、そこまでは席だけ）。現行資産のコンテナは `karume_ir` を持つ
    1 本なので、そのまま「先頭 = グラフ shard」の 1 要素 shard 列として宣言される。
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


def _artifact_size(artifact: Artifact) -> int:
    """`Artifact` の実体のバイト数（**中身は読まない** — 出所は stat 1 回だけ）。

    共有判定の前置フィルタ（{@link _plan_shared}）のためだけの寸法。sha256 と違って
    「違えば中身も必ず違う」方向にしか使えないので、一致しても畳む根拠にはならない。
    """
    if artifact.source is None:
        assert artifact.payload is not None  # __post_init__ の不変条件
        return len(artifact.payload)
    return artifact.source.stat().st_size


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

    出所を hash するのは **2 モデル以上が使う (相対 path, サイズ)** の席だけ — 相対 path が
    1 モデルにしか使われていなければ畳みようがなく、サイズが違えば hash も必ず違うので、
    どちらも読む理由が無い（sha256 は置いた現物から採る）。サイズは畳む根拠ではなく
    **落とす根拠**にしか使わない: 同じサイズの席は従来どおり hash で弁別されるので、畳む席の
    集合は前置の有無で完全に同じ（同じ相対 path に「畳める組」と「サイズ違いの単独」が
    同居しても、落ちるのは単独の側だけ）。
    """
    sizes: dict[tuple[str, str], int] = {}
    users: dict[tuple[str, int], set[str]] = {}
    for plan in plans:
        for role, artifact in plan.artifacts.items():
            size = sizes[(plan.name, role)] = _artifact_size(artifact)
            users.setdefault((artifact.rel_path, size), set()).add(plan.name)
    memo: dict[Path, str] = {}
    groups: dict[tuple[str, str], list[tuple[str, str]]] = {}
    samples: dict[tuple[str, str], Artifact] = {}
    for plan in plans:
        for role, artifact in plan.artifacts.items():
            if len(users[(artifact.rel_path, sizes[(plan.name, role)])]) < 2:
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
            # `shards` は順序付きの列で、**先頭がグラフ shard**（`karume_ir` を持つコンテナ）。
            # 分割規則を持たない現状は常に 1 要素 = そのコンテナそのもの（{@link WeightFiles}）。
            entry[label] = {
                "shards": [refs[files.file]],
                **({"extras": extras} if extras else {}),
            }
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

    公開の規律（staging へ作る → 通ってから据える → 失敗は staging だけを消す → 前回の中断が
    残した席の始末）は core の原語に預ける（{@link karume.artifacts.staged_publication}）—
    ここが持つのは「席の中で何を作り、どの門を通すか」だけ（共有席の決定 → 配置 →
    `karume.json` → {@link verify_dist} → `README.md`）。既存の `out_dir` は最後の rename まで
    1 バイトも触らないので、途中で落ちれば配布形は前回のまま残る（in-place で更新していた頃の
    「旧 manifest + 新旧混在ツリー」という失敗様式が構造的に無くなる）。

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

    try:
        with staged_publication(out_dir) as staging:
            manifest = _materialize_family(plans, staging, default_model)
            verify_dist(staging)
            if render_card is not None:
                (staging / MODEL_CARD_FILENAME).write_text(render_card(manifest), encoding="utf-8")
    except ArtifactSwapError as error:
        # 据え替えの失敗を組み立ての失敗と取り違えない（原因の I/O 故障は連鎖に残る）。
        raise DistError(str(error)) from error
    return manifest


# ---- ② 宣言と現物の突合 ------------------------------------------------------


def _declared_refs(manifest: Mapping[str, Any]) -> Iterator[tuple[str, Mapping[str, Any]]]:
    """manifest が参照する全ファイルを `(場所, 3 点セット)` で流す（重複はそのまま流す）。"""
    for model_name, model in manifest["models"].items():
        for name, labels in model["weights"].items():
            for label, entry in labels.items():
                for index, ref in enumerate(entry["shards"]):
                    yield f"models.{model_name}.weights.{name}.{label}.shards[{index}]", ref
                for extra, ref in entry.get("extras", {}).items():
                    yield f"models.{model_name}.weights.{name}.{label}.extras.{extra}", ref
        for name, ref in model["assets"].items():
            yield f"models.{model_name}.assets.{name}", ref


def _assert_manifest_shape(manifest: Mapping[str, Any]) -> None:
    """`karume/3` の構造整合（hub のパーサが受理する形かを焼いた側でも見る）。

    ここが見るのは**この配布形が自分で閉じているか**だけ — `defaultModel` / `defaultQuant` の
    指し先、quant の weights 完全写像、weights の shard 列、そしてレイアウト（ADR 0041 §9）。
    hub の全検査を写経しても正本が 2 つになるだけなので、写すのは「組み立てが壊れたら真っ先に
    破れる」規則に絞る。
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
        for name, labels in weights.items():
            for label, entry in labels.items():
                # MUST: shard 列は**非空**（先頭がグラフ shard = `karume_ir` を持つコンテナ）。
                # v2 の `{file}` を持ったままの manifest もここで落ちる — 形式識別子だけ書き換え
                # て中身が旧形の配布形は、hub が読めないのに焼く側では通ってしまう。
                shards = entry.get("shards")
                if not isinstance(shards, list) or not 1 <= len(shards) <= MAX_SHARDS:
                    # 実物が列なら件数だけを言う（上限超えの列をそのまま綴ると診断が数 MB になる）。
                    found = f"{len(shards)} 要素" if isinstance(shards, list) else repr(shards)
                    raise DistError(
                        f"{model_name}.weights.{name}.{label}.shards が"
                        f" 1〜{MAX_SHARDS} 要素の配列でない（実際: {found}）"
                    )
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


# ---- ③ pipeline 別ディスパッチと CLI -----------------------------------------


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


#: core wheel だけで組める pipeline（全量ではない — モジュール doc の MUST）。
#:
#: NOTE: **空**。family 別 recipe は 1 つ残らず wheel の外（`tools/export-recipes/<family>/`）
#: へ出たので（ADR 0065 段 3+4 完了）、受理集合の正本はリポの dist ドライバ
#: （`tools/export-recipes/dist.py`）の辞書だけになった。この席を残すのは、core が
#: 「表を受け取る側」であって「表を持たない側」ではないことを型で示すため — 空 mapping の
#: まま {@link main} を呼べば `--pipeline` が何を渡しても落ちる（core 単体では 1 つも組めない、
#: が正しい振る舞い）。
PIPELINES: Mapping[str, Pipeline] = {}


#: `--out` 省略時の出力先を作る hook（{@link main} が受ける）。リポの `models/<リポ名>/` を
#: 綴れるのは repo topology を知っている呼び出し側だけなので、core は形だけ持つ。
DefaultOutDir = Callable[[Pipeline, Sequence[str]], Path]


def build_parser(
    pipelines: Mapping[str, Pipeline] = PIPELINES,
    default_pipeline: str | None = None,
    *,
    default_series: Path | None = None,
    has_default_out_dir: bool = False,
) -> argparse.ArgumentParser:
    """`--pipeline` の受理集合を**引数で**受ける（ドライバが recipe を足した表を渡す）。

    `default_pipeline` を渡さない呼び出し（= core 単体）は `--pipeline` を必須にする
    （{@link main} が選択肢を並べて落とす）。`--series` / `--out` の既定も同じ扱いで、
    置き場を渡されていなければ help でそう名乗り、{@link main} が落とす。
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
        default=default_series,
        help="系列ディレクトリ群の親（"
        + (f"既定: {default_series}" if default_series is not None else "必須")
        + "）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="出力先 = 1 ディレクトリ 1 HF リポ（"
        + (
            "省略時は呼び出し側の既定。複数モデルを組むときは必須"
            if has_default_out_dir
            else "必須"
        )
        + "）",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    pipelines: Mapping[str, Pipeline] = PIPELINES,
    default_pipeline: str | None = None,
    default_out_dir: DefaultOutDir | None = None,
    default_series: Path | None = None,
) -> None:
    args = build_parser(
        pipelines,
        default_pipeline,
        default_series=default_series,
        has_default_out_dir=default_out_dir is not None,
    ).parse_args(argv)
    if args.pipeline is None:
        # MUST: 既定を core に焼かない（モジュール doc の MUST）— 表が呼び出し側で変わる。
        raise DistError(
            f"--pipeline が要る（選択肢: {', '.join(sorted(pipelines))}）— "
            "リポ専用 recipe まで含めた表は tools/export-recipes/dist.py が持つ"
        )
    if args.series is None:
        # MUST: repo topology を core に焼かない（モジュール doc の MUST）。
        raise DistError("--series が要る（系列ディレクトリ群の親）— 呼び出し側が既定を渡していない")
    pipeline = pipelines[args.pipeline]
    models = args.models if args.models else [pipeline.default_model]
    if args.out is not None:
        out_dir = args.out
    elif default_out_dir is not None:
        out_dir = default_out_dir(pipeline, models)
    else:
        raise DistError("--out が要る（配布形の出力先）— 呼び出し側が既定を渡していない")
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
