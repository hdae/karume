"""配布ディレクトリの組み立て — 系列ディレクトリ群 → HF へそのまま上げられる 1 リポ形。

仕様の正本は ADR 0041（`docs/decisions/0041-manifest-v2.md`）。ここが作るのは §2 の形で
並んだファイル群と、それを宣言する `karume.json`（`karume/5`）、そして manifest から機械導出
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

MUST: 検査（weights コンテナの全検証 / 格納 / rope 素表のバイト同一 / スタイル表・話者表の
行数）は**配置の前**に**全モデルぶん**済ませる — 落ちるなら途中の配布形を 1 ファイルも
残さない。組み立ては「計画（{@link ModelPlan} を組む = 検査と読み取りの全部）→ 実体化
（{@link assemble_family}）」の 2 段で、前段は 1 バイトも書かない。

MUST: **入力コンテナは「過去に検証済み」と信頼しない** — weights が指すコンポーネントは
{@link assert_weight_components_verified} が `karume.verify.verify_container` で丸ごと見る
（読むのは 2 文書だけ）。組み立てが自前で読むのは格納と IR の一部（{@link ir_graph}）でしか
ないので、ここを省くと descriptor / 束縛表の壊れた系列が family 固有の門をすり抜けて配布形に
据わり、利用者の `createSession` で初めて落ちる。

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

`--repo` はカードの Usage 例に綴る HF リポ ID の明示の上書き口で、既定は pipeline の宣言
（{@link resolve_repo}）。`--out` のディレクトリ名からは導かない。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import shutil
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from safetensors.numpy import save

from karume.artifacts import ArtifactSwapError, staged_publication

# 上限（part 件数・part 長の天井・descriptor 長）は import で引く — 書く側と宣言側
# （manifest 検査）が同じ綴りを見るため。{@link verify_dist} は手元のどの配布形にも掛け
# られる門なので、上限を知らない検査になっていると受理集合が 2 つに割れる。
from karume.container import (
    HEADER_BYTES,
    MAX_DESCRIPTOR_BYTES,
    MAX_PARTS,
    PART_MAX_BYTES,
    ContainerFormatError,
    DocumentRef,
    codec_entry,
    container_parts,
    numbered_name,
    read_descriptor_refs,
)
from karume.limits import LimitsError, max_state_slot_bytes, required_limits
from karume.modelcard import HF_OWNER
from karume.verify import ContainerError, VerifiedContainer, assert_ir_accepted, verify_container

# ---- ① 共有部: 置き場の綴り・共有席の決定・配置・ハッシュ・宣言と現物の突合 -------

#: manifest のファイル名（ADR 0041 §1 — リポジトリ直下の固定名）。
MANIFEST_FILENAME = "karume.json"

#: manifest の形式識別子（ADR 0109 決定 1 — hub は 1 形しか読まない）。`karume/5` は weights の
#: dtype エントリが**コンテナの入口**（`{container: {descriptor, parts}}`）になった形で、
#: `karume/4` の shard 列（`{shards, extras?}`）も `extras` の席も退役した（extras の実物 1 種
#: `rope_base` はコンテナの資産へ移った — ADR 0109 決定 4）。
#:
#: MUST: 旧 major は読まない（両読みは実装しない — ADR 0109 決定 1）。黙って読める形にせず
#: major で断絶を宣言する。
MANIFEST_FORMAT = "karume/5"

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

#: 配布リポ直下へ置ける**法的テキスト**の席（{@link Pipeline.root_files}）。上流の重み
#: ライセンスが再配布の条件として要求するファイル — ライセンス文そのもののコピーと、帰属 /
#: 改変を告げる Notice の 2 つだけ。名前を集合で縛るのは、ここが「任意ファイルを直下へ
#: 持ち込む口」ではないことを**検査で**示すため（型では法的テキストかどうかを言えない）。
#: {@link META_PATHS} と同じく**在ることは要求しない**（要求する pipeline だけが渡す）。

#: 改変告知の置き場（配布リポ直下）。容器の `provenance.notice` が指すのはこの **path 断片**で、
#: 本文は容器に載らない（container-v1 §2.3）。
NOTICE_FILENAME = "NOTICE.md"

#: 上流ライセンス本文の置き場（同上）。
LICENSE_FILENAME = "LICENSE.md"

LEGAL_PATHS = frozenset({LICENSE_FILENAME, NOTICE_FILENAME})

#: 規模上限（ADR 0041 §7）。hub が同じ値で弾くので、**焼く側で先に落とす**
#: （配布してから利用者の手元で初めて分かる形にしない）。
MAX_MODELS = 32
MAX_WEIGHTS = 32
MAX_ASSETS = 32
MAX_QUANTS = 32
MAX_PIPELINE_CONFIG_BYTES = 256 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024

#: weights の dtype ラベルの受理語彙 = runtime の**格納 dtype 語彙**。
#:
#: NOTE: 正本は codec 台帳の展開経路の語彙（TS `packages/runtime/src/format/container/codecs.ts`
#: の `CodecLayout` 型・Python {@link karume.container.CodecEntry} の `layout`）だが、TS 側の layout
#: の値は export された `CODEC_LEDGER` の各エントリが実行時の値として持つだけで、layout 語彙の
#: 一覧定数も、op 契約表のような機械突合の正本ファイルも無い — ここは**人手の写し**。ただし
#: 同じパッケージ内に IR 検証側の写し（{@link karume.verify.STORAGE_DTYPES}）があるので、
#: そちらを包含することをテストで突き合わせる（`tests/test_dist.py` の
#: `TestDtypeLabelVocabulary`）— 片方だけが古びる失敗様式は、その 1 本で機械化してある。
#:
#: 縛るのは、ラベルが「格納 dtype 語彙」であるという主張をカードが本文で述べるため
#: （{@link karume.modelcard.quants}）。格納の正本は descriptor の `encoding.codec` で、
#: ラベルは選択・表示の語彙（ADR 0071 決定 3）— その線引きをラベル側の受理集合で固定する。
STORAGE_DTYPE_LABELS = frozenset({"f32", "f16", "bf16", "i8", "i4", "i2", "i32"})

#: quant の表示欄（ADR 0075 決定 1）の文字数上限。`label` は選択肢に出す短い表示名、
#: `description` は 1 行の説明で、どちらも optional。hub は同じ値で境界検査するので
#: （manifest は外部入力 — ADR 0075 決定 2）、他の上限と同じく**焼く側で先に落とす**。
MAX_QUANT_LABEL_CHARS = 64
MAX_QUANT_DESCRIPTION_CHARS = 200

#: 表示欄の席と上限（{@link assert_quant_presentation} が引く 1 箇所）。
QUANT_PRESENTATION_LIMITS: Mapping[str, int] = {
    "label": MAX_QUANT_LABEL_CHARS,
    "description": MAX_QUANT_DESCRIPTION_CHARS,
}

#: 越境コンポーネント参照（ADR 0038 §7 の optional `repo` / `revision` 席）の revision の綴り
#: — HF の**完全な commit sha**（40 桁 hex 小文字）。ブランチ名やタグは指し先が動くので
#: pin にならない（{@link ExternalComponents}）。
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")

#: 同・参照先リポの綴り（`<owner>/<name>` — hub の path 検査と同じ文字集合）。
REPO_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")

#: 越境参照のファイル参照が持つ欄（3 点セット + 出所 2 席）。
EXTERNAL_REF_KEYS = frozenset({"repo", "revision", "path", "size", "sha256"})

#: モデル名の許可文字。モデル名は manifest のキーであると同時に**リポ内のディレクトリ名**
#: なので、hub の path セグメント検査（ADR 0041 §6）と同じ集合に縛る。
MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]*$")

#: モデルサブツリー内の相対 path の 1 セグメント（hub の `SEGMENT_RE` の鏡像 — ADR 0041 §6）。
PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")

#: sha256 の読み出し単位。数 GB を丸読みしないための唯一の要件で、値自体は素の I/O 単位。
_CHUNK_BYTES = 1 << 20


class DistError(ValueError):
    """組み立ての前提が破れた（資産の欠落・rope 素表の不一致・manifest と現物の食い違い）。"""


def component_parts(path: Path) -> tuple[Path, ...]:
    """コンポーネントの代表 path → 実在する part 列（連番が無ければ代表 path 自身の 1 要素）。

    連番規約（container-v1 §8）の失敗を組み立ての語彙へ翻訳するだけの薄い層。組み立て側の
    入口はここ 1 箇所で、格納の門も IR の読みも計画の展開も同じ列を見る。
    """
    try:
        return container_parts(path)
    except ContainerFormatError as cause:
        raise DistError(str(cause)) from cause


def assert_component_present(path: Path) -> None:
    """コンポーネントの現物（part 列）が在ることを落とす。

    綴りを 1 箇所に持つのは、不在の診断が「代表 path」で出続けるようにするため — 分割の
    有無で診断のファイル名が変わると、系列を焼き直す側が何を探せばよいか分からなくなる。
    """
    if not all(part.is_file() for part in component_parts(path)):
        raise DistError(f"組み立ての入力が無い: {path}")


def _container_of(path: Path) -> VerifiedContainer:
    """コンポーネントを開いて合流まで通す（{@link karume.verify.verify_container} の薄い翻訳）。

    NOTE: path 単位の覚書は**置かない** — 系列ディレクトリは据え替えで書き換わる可変な場所
    （モジュール doc）なので、プロセス寿命の写しは「同じ path の別の現物」を黙って返す。
    同じ需要計算の中で 2 度開いていた経路は、開いた結果を引き回す形で畳んである
    （{@link component_demand_bytes}）。
    """
    assert_component_present(path)
    try:
        return verify_container(component_parts(path))
    except (ContainerError, ContainerFormatError) as cause:
        raise DistError(f"{path}: コンテナとして読めない: {cause}") from cause


def storage_dtypes(path: Path) -> set[str]:
    """コンポーネントが実際に使っている**格納の語彙**（codec の展開経路 — 束縛表の和）。

    語彙は codec 台帳の `layout`（`f32` / `f16` / `bf16` / `i32` / `i8` / `i4` / `i2`）で、
    safetensors 方言の dtype 名（`F16` / `I4`）は退役した。和で見るのは、混成の資産は 1 本の
    容器に複数の codec が同居するため（`i4` 系列の scale は f32・重みは i4）。

    MUST: 正本は**束縛表**（descriptor の `encoding.codec`）であって dtype ラベルではない
    （ADR 0071 決定 3 の「ラベルは格納を主張しない」— ラベルは選択と表示の語彙）。
    """
    verified = _container_of(path)
    found: set[str] = set()
    for bound in verified.graphs.values():
        for supply in bound.supplies.values():
            entry = codec_entry(supply.encoding.codec)
            found.add(entry.layout)
            if supply.scale is not None:
                # companion scale は f32 固定（container-v1 §6.1）— 混成の資産が「f32 を含む」
                # ことの根拠はここなので、束縛表から明示的に足す。
                found.add("f32")
    return found


def assert_storage(role: str, path: Path, requirements: Mapping[str, str]) -> None:
    """役割が要求する格納が束縛表に存在することを検査する（無関係な役割は素通し）。

    要求表を引数で受けるのは、役割名が pipeline 間で衝突するため（Anima の `text_encoder` は
    f16 を要求し、SBV2 の `text_encoder` は i8 を要求する）。1 つの表に混ぜると、どちらかの
    要求が黙って他方に掛かる。
    """
    required = requirements.get(role)
    if required is None:
        return
    found = storage_dtypes(path)
    if required not in found:
        raise DistError(
            f"{role}: {path} の格納に {required} が無い（実際: {sorted(found)}）。"
            "系列を焼いたときの --dtype を確認する（f16 系列は --dtype f16 の fake-quant が必要）"
        )


def assert_storage_absent(role: str, path: Path, forbidden: Mapping[str, tuple[str, ...]]) -> None:
    """役割が**持ってはならない**格納が束縛表に無いことを検査する（無関係な役割は素通し）。

    {@link assert_storage} の存在検査だけでは **f32 席に圧縮系列の資産を挿し込む取り違えが
    素通りする** — 圧縮系列の容器は適格外の重み（bias / norm / グラフ定数・量子化の scale も）
    を f32 で持つので、「f32 を含む」は f16 / i8 資産でも真になる。片方向の存在検査を両側から
    挟んで初めて「系列 × 格納」が集合として一意に決まる（ADR 0027 / 0029 の検出限界 —
    **系列 root の取り違えは数値網では原理的に検出できない**ので、ここが唯一の検出器）。

    MUST: 禁止は**役割ごとに集合**で持つ（1 つだけだと圧縮系列が 2 本以上あるときに、名指し
    しなかったほうが黙って素通りする）。逆向き（圧縮席に f32 資産）は {@link assert_storage}
    が要求の不在で落とすので、禁止表は素の席にだけ要る。
    """
    banned = forbidden.get(role)
    if not banned:
        return
    found = storage_dtypes(path)
    intruders = [dtype for dtype in banned if dtype in found]
    if intruders:
        raise DistError(
            f"{role}: {path} の格納に {' / '.join(intruders)} がある"
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
    扱えるようにするため。生成物を据えられるのは `assets` の席だけで、weights が
    指す役割は配置に限る（{@link ModelPlan.__post_init__}）。
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


def ir_graph(path: Path, verified: VerifiedContainer | None = None) -> Mapping[str, Any]:
    """コンテナのグラフ記述から **IR v2 のグラフ JSON** を読む（part 0 だけ読む）。

    `karume/5` の容器は 1 コンテナ 1 グラフ（ADR 0109 決定 2）なので、`graphs` の唯一の値を
    返す。IR v1 との差は 4 点（docs/ir-v2.md）で、呼び手に効くのは **initializer 名が
    テンソルキーへ改名されている**ことと `storage` が束縛表へ出ていることの 2 つ。

    `verified` を渡すとその容器をそのまま使う（開き直さない）— 同じ需要計算の中で合流結果と
    グラフ文書の両方を使う呼び手（{@link component_demand_bytes}）が 2 度開かないための席。
    """
    graphs = (verified if verified is not None else _container_of(path)).read.graph.graphs
    if len(graphs) != 1:
        raise DistError(
            f"{path}: グラフが {len(graphs)} 本ある（`karume/5` は 1 コンテナ 1 グラフ）"
        )
    return next(iter(graphs.values()))


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

    有限性を別に見るのは、この値が `pipelineConfig` 経由で `karume.json` に載るため。上流の
    config を読むのは Python の `json.loads` で、これは `NaN` / `Infinity` を**既定で受理する**
    — 素通しすると標準 JSON に無いリテラルが manifest に焼かれ、ブラウザの `JSON.parse` が
    落ちる（{@link manifest_text} と対）。`NaN` は `<= 0` が偽なので正数検査もすり抜ける。
    """
    value = raw.get(key)
    if not isinstance(value, list) or len(value) != channels:
        raise DistError(f"{where} の {key} が長さ {channels} の配列でない（{value!r}）")
    channels_out: list[float] = []
    for entry in value:
        if not isinstance(entry, int | float) or isinstance(entry, bool):
            raise DistError(f"{where} の {key} に数でない要素がある（{value!r}）")
        if not math.isfinite(entry):
            raise DistError(f"{where} の {key} に有限でない要素がある（{value!r}）")
        if positive and entry <= 0:
            raise DistError(f"{where} の {key} に正でない要素がある（{value!r}）")
        channels_out.append(float(entry))
    return channels_out


@dataclass(frozen=True)
class WeightFiles:
    """weights の 1 dtype ぶん（`karume/5` の `{container}`）。中身は**役割名**。

    実 path は {@link Artifact} 側が持つ — 共有の畳み込みで path が `shared/…` へ動くので、
    宣言側が path を直に握っていると 2 箇所が独立に動く。

    席が 1 つだけなのは、コンテナが**部品の役割 1 つ**に対応するため（ADR 0109 決定 2）。
    `karume/4` の `extras` は退役し、実物 1 種（`rope_base`）はコンテナの資産へ移った
    （ADR 0109 決定 4）。ここが指すのはコンテナの**代表 1 役**で、実際に何本の part として
    宣言されるかは {@link expand_weight_parts} が組み立て時に現物から解決する。
    """

    file: str


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

    def __post_init__(self) -> None:
        """weights の席が指す役割は**系列からの配置**（`source`）だけ MUST。

        生成物（`payload`）はコンテナではないので、weights に据わると
        {@link expand_weight_parts} の part 解決も {@link bake_required_limits} の需要も
        {@link assert_weight_components_verified} の全検証も掛からないまま
        `parts: [1 要素]` として宣言される（表・sidecar の役割名を weights へ渡す綴り誤りが
        全門緑で通り、利用者の `createSession` で初めて落ちる）。計画の受け口で落とす。
        """
        for labels in self.weights.values():
            for files in labels.values():
                # 役割そのものの欠落は展開が役割名を綴って落とす（{@link expand_weight_parts}）。
                artifact = self.artifacts.get(files.file)
                if artifact is not None and artifact.source is None:
                    raise DistError(
                        f"{self.name}.weights: 役割 '{files.file}' が生成物（payload）を指す"
                        f"（{artifact.rel_path}）— weights は系列からの配置（source）だけ"
                    )


class PartitionedPlan(NamedTuple):
    """{@link expand_weight_parts} の出力 — part 展開済みの計画と、その振り分け表。"""

    plan: ModelPlan
    #: weights の代表役割名 → **順序付き**の part 役割名（part 0 から。単一形なら 1 要素）。
    parts: Mapping[str, tuple[str, ...]]


#: 展開で作る part 役割名の綴り（`<代表役割>#<1 始まりの番号>`）。役割名は manifest に出ない
#: 内部キーなので、`#` は「recipe が書いた役割名」と衝突しないための区切りでしかない。
PART_ROLE_SEPARATOR = "#"


def expand_weight_parts(plan: ModelPlan) -> PartitionedPlan:
    """weights の役割が**分割形のコンテナ**を指しているとき、part ごとの役割へ展開する。

    MUST: part の本数は**現物から**解決する（{@link component_parts}）— 何本に割れるかは
    書いたバイト数で決まるので、pipeline の表にも recipe の定数にも書けない。表に書かせると
    「再 export で本数が変わったのに宣言は前回のまま」という、形も型も合う沈黙誤宣言が作れる。

    単一形（連番でない現物）は 1 要素の列として素通しする — `karume/5` の `container.parts` は
    2 要素以上 MUST（ADR 0109 決定 3）なので、{@link _assert_manifest_shape} が名指しで落とす。
    展開が起きた役割は代表役割を**artifacts から外し**、part 1 本ごとに `Artifact` を作る
    （相対 path は代表 path へ同じ連番規則を掛けたもの — 系列側のファイル名と配布形の
    ファイル名が同じ 1 本の綴りから出る）。
    """
    roles = sorted({files.file for labels in plan.weights.values() for files in labels.values()})
    # weights 以外の席（assets）から**同じ役割**を指している綴りは、展開で役割名が消えると
    # 宣言の側だけが行き場を失う。数は数本なので、集めて名指しで落とす。
    elsewhere = set(plan.assets.values())
    artifacts = dict(plan.artifacts)
    parts: dict[str, tuple[str, ...]] = {}
    for role in roles:
        artifact = plan.artifacts.get(role)
        if artifact is None:
            raise DistError(
                f"{plan.name}: weights が役割 '{role}' を指しているが artifacts に無い"
                f"（役割: {sorted(plan.artifacts)}）"
            )
        assert artifact.source is not None  # ModelPlan.__post_init__ の不変条件
        sources = component_parts(artifact.source)
        if len(sources) == 1:
            parts[role] = (role,)
            continue
        if role in elsewhere:
            raise DistError(
                f"{plan.name}: 分割形のコンテナの役割 '{role}' を assets も指している"
                "（あちらの席は 1 ファイル参照なので、複数 part を宣言できない）"
            )
        del artifacts[role]
        members: list[str] = []
        total = len(sources)
        for index, source in enumerate(sources, start=1):
            member = f"{role}{PART_ROLE_SEPARATOR}{index}"
            if member in artifacts:
                raise DistError(
                    f"{plan.name}: part 役割名 '{member}' が既存の役割と衝突する"
                    f"（'{PART_ROLE_SEPARATOR}' を含む役割名は使えない）"
                )
            artifacts[member] = Artifact(
                numbered_name(artifact.rel_path, index, total), source=source
            )
            members.append(member)
        parts[role] = tuple(members)
    return PartitionedPlan(replace(plan, artifacts=artifacts), parts)


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


def assert_rel_path(rel_path: str, where: str) -> str:
    """`Artifact.rel_path` がモデルサブツリーの中に収まることを検査して返す。

    hub の `assertPath`（`packages/hub/src/manifest.ts`）と同じ**許可リスト**で見る。禁止列挙
    （`..` を弾く）にしないのは hub と同じ理由 — 取得層はセグメントを percent-encode しても
    ドットを透過するので、列挙の抜けがそのまま traversal になる。

    組み立て側の実害は取得層より一段重い: 配置は `out_dir / f"{モデル名}/{rel_path}"` へ
    書くので、`A/../../victim` のような形は **staging の外の既存ファイルを truncate で潰す**。
    しかも manifest 側の root 検査（{@link _assert_manifest_shape}）は先頭セグメントしか見ず、
    宣言外ファイル検査（{@link verify_dist}）は staging の中しか見ないので、
    **書いた後の門は 1 つも鳴らない**。
    """
    for segment in rel_path.split("/"):
        if not segment:
            raise DistError(
                f"{where}: rel_path '{rel_path}' に空セグメントがある"
                "（先頭 / 末尾 / 連続スラッシュ）"
            )
        if segment.startswith("."):
            raise DistError(
                f"{where}: rel_path '{rel_path}' に先頭ドットのセグメント '{segment}' がある"
            )
        if not PATH_SEGMENT_RE.match(segment):
            raise DistError(
                f"{where}: rel_path '{rel_path}' のセグメント '{segment}' が"
                "許可文字 [A-Za-z0-9._-] に一致しない"
            )
    return rel_path


def assert_plan_paths(plans: Sequence[ModelPlan]) -> None:
    """モデル名と `rel_path` の収まりを**配置の前**に落とす。

    名前と path の検査だけが計画段の外に居たので、リポ内のプログラマ誤り（recipe の定数）が
    そのまま staging 外への書き込みになりえた（{@link assert_rel_path}）。

    MUST: **1 つの相対 path が持てる席はモデル内でも 1 つだけ** — モデル**間**には
    {@link _plan_shared} が同じ MUST を張っているのに、モデル**内**だけが素通しだった。
    2 役が同じ path を主張すると後の役が先の役の実体を上書きし、manifest は両役を**後の
    digest** で宣言する。現物と表は一致するので {@link verify_dist} は沈黙し、先の役の
    セッションが別のグラフを読む配布形がそのまま出荷される。
    """
    for plan in plans:
        assert_model_name(plan.name)
        seats: dict[str, str] = {}
        for role, artifact in plan.artifacts.items():
            assert_rel_path(artifact.rel_path, f"{plan.name}.{role}")
            claimed = seats.setdefault(artifact.rel_path, role)
            if claimed != role:
                raise DistError(
                    f"{plan.name}: 相対 path '{artifact.rel_path}' を役割 '{claimed}' と"
                    f" '{role}' が両方主張している（1 つの相対 path が持てる席は 1 つだけ —"
                    "後勝ちで上書きされ、manifest は両役を後の digest で宣言する）"
                )


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


#: quant の optional 欄の綴り（ADR 0038 §7 — hub の `parseRequiredLimits` が読む）。
REQUIRED_LIMITS_KEY = "requiredLimits"


def component_demand_bytes(
    source: Path, pipeline_config: Mapping[str, Any], memo: dict[Path, int]
) -> int:
    """コンポーネント 1 つの**常駐 1 バッファの最大バイト数**（重みテンソル / state スロット）。

    読むのは **part 0 の 2 文書だけ**（重みの payload は 1 バイトも舐めない）。`memo` は代表
    path 単位の覚え書き — 複数の quant が同じコンポーネントを選ぶ席（共有 text_encoder の
    ような形）で、同じ descriptor を読み直さないため。

    MUST: 需要は**合流した供給計画**（`bind_graphs`）から出す。piece 列に割れた重みは GPU 側で
    親 1 本のバッファに戻る（container-v1 §5 の規則④）ので、piece ごとの最大を採ると
    `requiredLimits` が過小に焼かれ、「宣言は満たすのに `createSession` で落ちる」という最も
    損な形になる。companion scale は別バッファなので別に数える。const 領域（part 1）の
    initializer も常駐するので同じ集合に入る。
    """
    remembered = memo.get(source)
    if remembered is not None:
        return remembered
    verified = _container_of(source)
    demand = 0
    for bound in verified.graphs.values():
        for supply in bound.supplies.values():
            demand = max(demand, sum(block.payload_bytes for block in supply.blocks))
            if supply.scale is not None:
                demand = max(demand, supply.scale.payload_bytes)
    try:
        demand = max(
            demand,
            max_state_slot_bytes(ir_graph(source, verified), pipeline_config, str(source)),
        )
    except LimitsError as cause:
        raise DistError(str(cause)) from cause
    memo[source] = demand
    return demand


def bake_required_limits(plan: ModelPlan) -> ModelPlan:
    """quant ごとの `requiredLimits`（ADR 0038 §7）を**現物から**導いて焼いた計画を返す。

    需要は quant が選ぶコンポーネントの ① 最大テンソルの格納 payload と ② 最大 state スロット
    （容量を `pipelineConfig` で数値化した寸法）の最大で、保証既定を超える席だけが欄になる
    （既定以内なら欄そのものが無い — 規則と理由の正本は {@link karume.limits}）。

    MUST: **計画側が同じ欄を書いていたら fail loudly** — 導出できる値を recipe の quant 表にも
    持たせると、格納形や容量を変えた回に「表だけが古い下限を名乗る」形が作れる（宣言と現物が
    独立に動く二重管理）。読み手は宣言を信じて取得の前に拒否するので、古い下限は「宣言は
    満たすのに動かない」配布物になる。
    """
    memo: dict[Path, int] = {}
    quants: dict[str, Any] = {}
    for quant_name, quant in plan.quants.items():
        where = f"{plan.name}.quants.{quant_name}"
        if REQUIRED_LIMITS_KEY in quant:
            raise DistError(
                f"{where}.{REQUIRED_LIMITS_KEY} を計画が書いている — この欄は組み立てが配布に"
                "入る現物（テンソル寸法と state スロット）から導く席で、表に書く席ではない"
            )
        demand = 0
        for name, label in quant["weights"].items():
            labels = plan.weights.get(name)
            files = labels.get(label) if labels is not None else None
            if files is None:
                raise DistError(f"{where}.weights: '{name}' に dtype '{label}' が無い")
            artifact = plan.artifacts.get(files.file)
            if artifact is None:
                raise DistError(f"{where}.weights: 役割 '{files.file}' が artifacts に無い")
            # assets は見ない — 表・tokenizer・sidecar が GPU の常駐バッファになるかは
            # family 側の事情で、コンテナの宣言からは決まらない。
            assert artifact.source is not None  # ModelPlan.__post_init__ の不変条件
            demand = max(
                demand, component_demand_bytes(artifact.source, plan.pipeline_config, memo)
            )
        limits = required_limits(demand)
        quants[quant_name] = {**quant, **({REQUIRED_LIMITS_KEY: limits} if limits else {})}
    return replace(plan, quants=quants)


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


def container_asset_bytes(source: Path) -> int:
    """容器が宣言する資産の**論理長の和**（ADR 0109 決定 4 — PLE / `rope_base` が入った席）。

    読むのは descriptor だけ（payload は 1 バイトも読まない）。モデルカードの「host 読みの
    資産」の勘定がここを引く — 資産は part の中に入ったので、manifest の `assets`（= 独立した
    ファイルの席）を数えるだけでは、容器へ移った表が内訳から黙って消える。
    """
    model = _container_of(source).read.model
    return 0 if model is None else sum(record.length for record in model.assets.values())


def _model_entry(
    plan: ModelPlan,
    refs: Mapping[str, dict[str, Any]],
    parts: Mapping[str, tuple[str, ...]],
    descriptors: Mapping[str, tuple[DocumentRef, DocumentRef]],
    host_assets: dict[str, int],
) -> dict:
    """1 モデルぶんの manifest エントリ（ADR 0109 決定 3）。`refs` は役割名 → 3 点セット。

    `parts` は {@link expand_weight_parts} が現物から解決した振り分け表（代表役割 → 順序付き
    part 役割）、`descriptors` は**置いた現物**の part 0 から採った 2 文書の期待値。

    `host_assets`（part 0 の配布相対 path → {@link container_asset_bytes}）は**書き足し先**で、
    manifest には載らない（モデルカードの内訳注記だけが読む派生値 — 宣言は容器が持つ）。

    MUST: 資産の勘定は **quant が選ぶ席だけ**を開く。どの quant も選ばない dtype 席まで開くと、
    入力コンテナの門（{@link assert_weight_components_verified}）が唯一の検出器である穴に
    2 つ目の検出点ができ、あちらの故障注入（門を外すと壊れた容器が据わる）が恒真化する。
    """
    selected = {
        (name, label)
        for quant in plan.quants.values()
        for name, label in quant.get("weights", {}).items()
    }
    weights: dict[str, Any] = {}
    for name, labels in plan.weights.items():
        entry: dict[str, Any] = {}
        for label, files in labels.items():
            graph_ref, model_ref = descriptors[files.file]
            # `parts` は順序付きの列で、**先頭が part 0**（ヘッダ + 2 文書のファイル）。
            # 並びは添字順 MUST — 添字が part の id なので、列を触った瞬間に識別子が壊れる
            # （container-v1 §8 / ADR 0109 決定 3）。
            members = parts[files.file]
            placed = [refs[role] for role in members]
            entry[label] = {
                "container": {
                    "descriptor": {
                        "graph": graph_ref.to_document(),
                        "model": model_ref.to_document(),
                    },
                    "parts": placed,
                }
            }
            if (name, label) in selected:
                # 資産の宣言は**出所の容器**から採る（置いた現物と同じバイト列であることは
                # {@link verify_dist} と越境の突合が別に見る）。
                head = plan.artifacts[members[0]].source
                assert head is not None  # ModelPlan.__post_init__ の不変条件
                host_assets[placed[0]["path"]] = container_asset_bytes(head)
        weights[name] = entry
    return {
        "pipeline": plan.pipeline,
        "weights": weights,
        "assets": {name: refs[role] for name, role in plan.assets.items()},
        "quants": dict(plan.quants),
        "defaultQuant": plan.default_quant,
        "pipelineConfig": dict(plan.pipeline_config),
    }


def assert_quant_presentation(where: str, quant: Mapping[str, Any]) -> None:
    """quant の表示欄（ADR 0075 決定 1 の `label` / `description`）の形と上限を落とす。

    どちらも optional（設定の無い席は呼び手が id をそのまま出す）だが、**書いたなら 1 行の
    非空文字列で上限以内**であることを要求する。hub は同じ境界検査を持つので、緩いまま配ると
    「焼けたのに読めない manifest」になる。改行を弾くのは `description` が 1 行の説明だから
    （選択 UI の 1 行にも、モデルカードの表の 1 セルにも改行は入れられない）。
    """
    for key, limit in QUANT_PRESENTATION_LIMITS.items():
        value = quant.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise DistError(f"{where}.{key} が非空の文字列でない（{value!r}）")
        if "\n" in value:
            raise DistError(f"{where}.{key} が 1 行でない（改行を含む）: {value!r}")
        if len(value) > limit:
            raise DistError(f"{where}.{key} が {len(value)} 字で上限 {limit} を超えた")


def _assert_model_scale(
    name: str,
    *,
    weights: int,
    assets: int,
    quants: Mapping[str, Any],
    pipeline_config: Mapping[str, Any],
) -> None:
    """1 モデルぶんの規模上限（ADR 0041 §7）と quant の表示欄（ADR 0075 決定 1）。

    件数以外に `quants` の中身まで受けるのは表示欄の上限を見るため。同じ規則を計画
    （{@link ModelPlan}）と manifest の両方から掛けるので、上限の綴りは 1 箇所に置く。
    """
    for key, count, limit in (
        ("weights", weights, MAX_WEIGHTS),
        ("assets", assets, MAX_ASSETS),
        ("quants", len(quants), MAX_QUANTS),
    ):
        if count > limit:
            raise DistError(f"{name}.{key} が {count} 件で上限 {limit} を超えた")
    for quant_name, quant in quants.items():
        assert_quant_presentation(f"{name}.quants.{quant_name}", quant)
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
            quants=plan.quants,
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


def weight_components(partitioned: Sequence[PartitionedPlan]) -> list[tuple[tuple[Path, ...], str]]:
    """weights が指す**ローカル**コンテナの part 列（添字順）と、その席の weights のキーを集める。

    共有コンポーネント（複数モデルが同じ系列を指す席）は (part 列, キー) で dedupe する
    — 同じバイト列を同じ名前で 2 度検証しても結論は変わらない。**キーまで含めて**畳むのは、
    1 本のコンテナが 2 つの違うキーの席に据わっている形が {@link
    assert_weight_components_verified} のグラフ名の門に掛かる欠陥そのものだから（1 本の
    コンテナは 1 本のグラフしか持たないので、両方は成立しえない）。生成物（`payload`）の
    役割は weights に現れない（{@link ModelPlan.__post_init__} が入口で落とす）ので、ここは
    全役割を集める。

    NOTE: 見るのは weights だけ。`assets`（tokenizer・スタイル表などの table safetensors）は
    コンテナではないので、コンテナの門を掛ける先ではない。
    """
    components: list[tuple[tuple[Path, ...], str]] = []
    seen: set[tuple[tuple[Path, ...], str]] = set()
    for item in partitioned:
        seats = sorted(
            {
                (files.file, component)
                for component, labels in item.plan.weights.items()
                for files in labels.values()
            }
        )
        for role, component in seats:
            sources: list[Path] = []
            for member in item.parts[role]:
                source = item.plan.artifacts[member].source
                assert source is not None  # ModelPlan.__post_init__ の不変条件
                sources.append(source)
            entry = (tuple(sources), component)
            if entry in seen:
                continue
            seen.add(entry)
            components.append(entry)
    return components


def assert_weight_components_verified(partitioned: Sequence[PartitionedPlan]) -> None:
    """weights の全コンテナを**宣言で決まる全規則**で検証する（配置の前）。

    MUST: 組み立ては入力コンテナを「過去に検証済み」と信頼しない。系列ディレクトリは
    据え替えで書き換わる可変な場所（モジュール doc）なので、**古いエクスポータで焼いた系列を
    `--series` で指す**運用事故が現実にありうる。組み立てが自前で見るのは格納と IR の一部
    （{@link ir_graph}）なので、descriptor / 束縛表の壊れは family 固有の門をすり抜けて配布形に
    据わり、利用者の `createSession` で初めて落ちる。

    掛かるのは読み手の構造検査（2 文書・block 目次・codec 台帳）と合流（`bind_graphs`）と、
    **IR の受理規則**（op 語彙 / ランタイム支援 / op 契約 —
    {@link karume.verify.assert_ir_accepted}）である。どれも**宣言だけ**を見るので、数 GB の
    payload は 1 バイトも読まない（container-v1 §7 のハッシュ 3 分離）。block の sha256 は
    書き手側の据え替え前検証（`karume.publish`）が既に通している。

    MUST: IR の受理規則をここで掛ける — 構造検査だけだと「語彙外の op / ランタイム未対応の
    attrs / 契約違反の shape」を宣言した容器が配布形に据わり、利用者の `createSession` で
    初めて落ちる（`karume.verify` のモジュール doc が掲げる目的の、組み立て側の半分）。

    MUST: **容器のグラフ名 == その席の weights のキー**もここで見る。ランタイムは
    `prepareContainer(opened, <weights キー>)` で名前で引く（container-v1 §2.1）ので、綴りが
    割れた容器は manifest ごと据わり、利用者の `createSession` で初めて「コンテナにグラフが
    無い」になる。書き手側の綴りの門（各 recipe の定数と AST の突合）はソースしか見ないので、
    **現物と宣言を突き合わせる唯一の門**がここである。
    """
    for parts, component in weight_components(partitioned):
        try:
            verified = verify_container(parts)
            assert_ir_accepted(verified.read)
        except (ContainerError, ContainerFormatError) as cause:
            raise DistError(
                f"{parts[0]}: 組み立ての入力がコンテナの規則を満たさない: {cause}"
            ) from cause
        graphs = set(verified.read.graph.graphs)
        if graphs != {component}:
            raise DistError(
                f"{parts[0]}: 容器のグラフ名 {sorted(graphs)} が weights のキー"
                f" '{component}' と違う — ランタイムはこのキーでグラフを引く"
                "（container-v1 §2.1）ので、このまま配ると createSession で落ちる"
            )


def assert_root_files(root_files: Mapping[str, str]) -> None:
    """配布リポ直下へ書く名前が法的テキストの席（{@link LEGAL_PATHS}）に収まることを落とす。

    MUST: 未知の名前は fail loudly — 直下は manifest が宣言しない唯一の場所なので、ここを
    素通しにすると「宣言外ファイルの不在」（{@link verify_dist}）の網が名前 1 つぶんずつ
    緩む。席の意味（上流ライセンスが要求する再配布条件のファイル）は型では言えないので、
    受理集合を検査で綴る。
    """
    unknown = sorted(set(root_files) - LEGAL_PATHS)
    if unknown:
        raise DistError(
            f"配布リポ直下へ書けるのは {sorted(LEGAL_PATHS)} だけ（法的テキスト専用の席）: "
            + ", ".join(unknown)
        )


#: 長さ 0 の実体（const が空の part 1）が名乗る sha256。実体は 1 通りしかないので値も 1 つに
#: 決まる（ADR 0109 決定 3 — hub の `EMPTY_SHA256` と同値）。
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def assert_container_limits(where: str, container: Mapping[str, Any]) -> None:
    """コンテナの入口 1 件が `karume/5` の上限と規則に収まることを落とす（ADR 0109 決定 6）。

    MUST: 上限の綴りは**コンテナ側の定数**（`karume.container`）から引く — hub 側
    （`packages/hub/src/manifest.ts`）も同じ値を持つので、写しを 3 つ目に増やさない。突合は
    `tests/test_dist.py` の定数突合テストが機械で見る。

    見るのは**宣言だけで閉じるもの**（件数・天井・`size: 0` の規則・part 0 の長さ・越境の
    一様性・descriptor 長）。descriptor と parts の整合・block 目次・codec 台帳の突合は
    `verify_container` が持つので、ここには置かない（検査点を 2 つにしない）。

    MUST: 欄の欠落も `DistError` で落とす（素の添字で `KeyError` / `TypeError` を出さない）—
    {@link verify_dist} は**外から来た `karume.json`**（手元のどの配布形でも）を受ける門なので、
    組み立ての語彙から外れた例外が出ると「壊れた manifest」と「門の不具合」が区別できない。
    """
    documents: list[int] = []
    descriptor = _dist_object(container.get("descriptor"), f"{where}.descriptor")
    for key in ("graph", "model"):
        document = _dist_object(descriptor.get(key), f"{where}.descriptor.{key}")
        length = document.get("length")
        if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
            raise DistError(f"{where}.descriptor.{key}.length が正整数でない（{length!r}）")
        if length > MAX_DESCRIPTOR_BYTES:
            raise DistError(
                f"{where}.descriptor.{key}.length {length} が上限 {MAX_DESCRIPTOR_BYTES} を超えた"
            )
        documents.append(length)
    parts = container.get("parts")
    if not isinstance(parts, list) or len(parts) < 2:
        found = f"{len(parts)} 件" if isinstance(parts, list) else repr(parts)
        raise DistError(
            f"{where}.parts が 2 件以上の配列でない（実際: {found}）"
            " — part 0〈descriptor〉+ part 1〈const〉が要る（container-v1 §8）"
        )
    if len(parts) > MAX_PARTS:
        raise DistError(f"{where}.parts が {len(parts)} 件で上限 {MAX_PARTS} を超えた")
    refs = [_dist_part_ref(part, f"{where}.parts[{index}]") for index, part in enumerate(parts)]
    part0 = HEADER_BYTES + documents[0] + documents[1]
    if refs[0]["size"] != part0:
        raise DistError(
            f"{where}.parts[0]: part 0 の size {refs[0]['size']} が"
            f" ヘッダ {HEADER_BYTES} + 2 文書 {documents[0]} + {documents[1]} = {part0} と違う"
        )
    head = (refs[0].get("repo"), refs[0].get("revision"))
    for index, ref in enumerate(refs):
        if ref["size"] > PART_MAX_BYTES:
            raise DistError(
                f"{where}.parts[{index}]: part '{ref['path']}' が {ref['size']} バイトで"
                f" 上限 {PART_MAX_BYTES}（part 長の天井 — container-v1 §10）を超えた"
            )
        # 長さ 0 は「const が空の part 1」の形だけ。part 0 はここへ来るまでに
        # 「ヘッダ + 2 文書」ちょうどであることが済んでいる（2 文書は正整数なので 0 に
        # ならない）ので、残る規則は空列の sha256 だけ。
        if ref["size"] == 0 and ref["sha256"] != EMPTY_SHA256:
            raise DistError(
                f"{where}.parts[{index}]: size 0 の sha256 が空列の値 {EMPTY_SHA256} でない"
            )
        # 越境参照は**容器単位**（ADR 0109 決定 3）— 片方だけ・混在は、残りの part を
        # セッションの repo へ取りに行く形になり、そこには無いので取得の途中で初めて落ちる。
        if (ref.get("repo"), ref.get("revision")) != head:
            raise DistError(
                f"{where}.parts[{index}]: 越境参照が容器の中で混在している"
                f"（part 0 は {head[0] or '自リポ'} / この part は {ref.get('repo') or '自リポ'}）"
            )


def _dist_object(value: Any, where: str) -> Mapping[str, Any]:
    """manifest の 1 節をオブジェクトとして受ける（欄の欠落を組み立ての語彙へ翻訳する）。"""
    if not isinstance(value, dict):
        raise DistError(f"{where} がオブジェクトでない（実際: {value!r}）")
    return value


def _dist_part_ref(value: Any, where: str) -> Mapping[str, Any]:
    """part 1 件の 3 点セット（`path` / `size` / `sha256`）が揃っていることを落とす。"""
    ref = _dist_object(value, where)
    if not isinstance(ref.get("path"), str) or not ref["path"]:
        raise DistError(f"{where}.path が非空の文字列でない（実際: {ref.get('path')!r}）")
    size = ref.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise DistError(f"{where}.size が非負整数でない（実際: {size!r}）")
    if not isinstance(ref.get("sha256"), str) or not ref["sha256"]:
        raise DistError(f"{where}.sha256 が非空の文字列でない（実際: {ref.get('sha256')!r}）")
    return ref


def assert_manifest_limits(manifest: Mapping[str, Any]) -> None:
    """規模上限（ADR 0041 §7 + ADR 0109 決定 6）を焼く側で先に落とす。"""
    models = manifest["models"]
    if len(models) > MAX_MODELS:
        raise DistError(f"models が {len(models)} 件で上限 {MAX_MODELS} を超えた")
    for name, model in models.items():
        _assert_model_scale(
            name,
            weights=len(model["weights"]),
            assets=len(model["assets"]),
            quants=model["quants"],
            pipeline_config=model["pipelineConfig"],
        )
        for component, labels in model["weights"].items():
            for label, entry in labels.items():
                assert_container_limits(
                    f"{name}.weights.{component}.{label}.container", entry["container"]
                )
    total = len(manifest_text(manifest).encode("utf-8"))
    if total > MAX_MANIFEST_BYTES:
        raise DistError(f"manifest が {total} バイトで上限 {MAX_MANIFEST_BYTES} を超えた")


def manifest_text(manifest: Mapping[str, Any]) -> str:
    """`karume.json` に書く綴り（末尾改行つき — 上限検査もこの綴りで測る）。

    allow_nan=False は必須（`karume.ir.IrGraph.to_json` と同文）— NaN / Infinity は JSON の
    標準リテラルに無く、ブラウザの `JSON.parse`（hub の読み口）が落ちる。Python の
    `json.dumps` は既定でこれらを綴り、`json.loads` は既定で読み返すので、{@link verify_dist}
    まで含めて焼く側の門は 1 つも鳴らない — 受理集合をランタイム側に揃えるため、書き出しの
    時点で失敗させる。
    """
    return json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _descriptor_refs(part0: Path) -> tuple[DocumentRef, DocumentRef]:
    """part 0 のファイル → 2 文書の期待値（読み取りの失敗を組み立ての語彙へ翻訳する）。"""
    try:
        return read_descriptor_refs(part0)
    except ContainerFormatError as cause:
        raise DistError(f"{part0}: 2 文書の期待値を採れない: {cause}") from cause


def _materialize_family(
    partitioned: Sequence[PartitionedPlan],
    out_dir: Path,
    default_model: str,
    external: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, int]]:
    """検査済みの計画群を `out_dir` へ並べ、`karume.json` を書いて manifest を返す。

    戻りの 2 本目はモデルカード用の派生値（part 0 の配布相対 path →
    {@link container_asset_bytes}）— manifest には載らない（正本は容器の資産宣言）。

    ① 共有席を決める（{@link _plan_shared} — 出所の sha256 だけを見る）→ ② 共有席は
    `shared/` へ 1 回だけ・残りは各モデルのサブツリーへ置く → ③ 現物から manifest。

    `external`（役割名 → 越境参照）に載った役割は**自リポへ置かない** — 参照にする意味が
    「同じバイト列を 2 つのリポへ上げない」ことなので、置いた上で参照を書くと利得がゼロになる。

    MUST: 共有の判定は**置く前**に済ませる — 全モデルへ複製してから畳み直す形は、共有 1 本
    （サイズ S・M モデル）につき「複製 M 回 + hash M 回 + 移動 1 回」で ~3MS の論理 I/O を
    払う。決めてから置けば S の複製も hash も 1 回で済む（配布の大半は共有資産）。

    書き先は呼び手（{@link assemble_family}）が用意する staging で、**空から作る**前提
    （配布先を直接更新しないので、途中で落ちても捨てるだけで済む）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plans = [item.plan for item in partitioned]
    placed: dict[tuple[str, str], str] = {}
    digests: dict[str, str] = {}
    folded: dict[tuple[str, str], str] = {}
    # MUST: 共有席の判定は**越境参照の役割を外した集合**で行う — 自リポへ置かない役割が
    # `shared/` の席を取ると、どのモデルも宣言していない現物が 1 本残る（複数モデルが同じ
    # バイト列を借りる形＝ADR 0087 の extra リポでちょうどこれが起きる）。下の配置ループが
    # `external` を飛ばすのと同じ集合を見る。
    shareable = [
        replace(
            plan,
            artifacts={role: a for role, a in plan.artifacts.items() if role not in external},
        )
        for plan in plans
    ]
    for seat in _plan_shared(shareable):
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
            if role in external:
                continue
            target = folded.get((plan.name, role))
            if target is not None:
                placed[(plan.name, role)] = target
                continue
            rel_path = f"{plan.name}/{artifact.rel_path}"
            materialize(artifact, out_dir / rel_path)
            placed[(plan.name, role)] = rel_path
            digests[rel_path] = sha256_file(out_dir / rel_path)

    models: dict[str, Any] = {}
    host_assets: dict[str, int] = {}
    for item in partitioned:
        plan = item.plan
        refs = {}
        for role in plan.artifacts:
            reference = external.get(role)
            if reference is not None:
                refs[role] = dict(reference)
                continue
            rel_path = placed[(plan.name, role)]
            refs[role] = file_ref(out_dir, rel_path, digests[rel_path])
        # descriptor の期待値も**置いた現物**から採る（表と現物が食い違う失敗様式を構造的に
        # 消す — モジュール doc の MUST）。越境参照の席は自リポに置かないので、代わりに
        # 手元の出所から採る — {@link external_refs} が「参照先は自分で組むはずだったバイト列と
        # 同一」まで確かめてあるので、同じ事実になる。
        descriptors: dict[str, tuple[DocumentRef, DocumentRef]] = {}
        for role, members in item.parts.items():
            head = members[0]
            if head in external:
                source = plan.artifacts[head].source
                assert source is not None  # ModelPlan.__post_init__ の不変条件
                descriptors[role] = _descriptor_refs(source)
            else:
                descriptors[role] = _descriptor_refs(out_dir / placed[(plan.name, head)])
        models[plan.name] = _model_entry(plan, refs, item.parts, descriptors, host_assets)
    manifest = {
        "format": MANIFEST_FORMAT,
        "generator": generator_tag(),
        "defaultModel": default_model,
        "models": models,
    }
    assert_manifest_limits(manifest)
    (out_dir / MANIFEST_FILENAME).write_text(manifest_text(manifest), encoding="utf-8")
    return manifest, host_assets


def assemble_family(
    plans: Sequence[ModelPlan],
    out_dir: Path,
    default_model: str,
    render_card: Callable[..., str] | None = None,
    root_files: Mapping[str, str] | None = None,
    external: ExternalComponents | None = None,
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

    `root_files`（ファイル名 → テキスト）は**リポ直下**へ UTF-8 で書かれる法的テキスト
    （{@link LEGAL_PATHS}）— 上流の重みライセンスが再配布の条件として要求するもので、manifest が
    宣言する資産ではない（どのモデルにも属さず、配布リポそのものに掛かる）。`karume.json` /
    `README.md` と同じく {@link verify_dist} の宣言外ファイル検査の例外側に居る。

    `external`（{@link ExternalComponents}）を渡すと、指名された役割だけが**別リポの pin 済み
    revision への参照**として宣言され、自リポには置かれない（ADR 0038 §7）。渡さない組み立ては
    完全に自己完結で、1 バイトも変わらない。

    MUST: `plans` 全体に掛かる検査は**最初の 1 バイトを書く前**に全部済ませる（staging すら
    作らない）— 途中で落ちると数 GB を並べ直すことになる。越境参照の解決（参照元 manifest の
    読み・現物の hash）も同じ理由でここに置く。
    """
    if not plans:
        raise DistError("組み立てるモデルが 1 つも無い")
    # MUST: `requiredLimits` は組み立てが**現物から**導いて焼く（{@link bake_required_limits}）
    # — 宣言の意味は「このバイト列を常駐させるのに要る device limit」なので、配布に入る現物
    # 以外に正本は無い。展開より前に置くのは、導出の入口が**代表 path**（`component_parts` /
    # `ir_graph` が受ける口）だから — 展開後の part 役割からは代表 path を引き直せない。
    plans = [bake_required_limits(plan) for plan in plans]
    # MUST: 展開は**全ての門より前**（1 バイトも書く前）— 展開後の役割が持つ相対 path も
    # 出所も、以降の検査（path の収まり・入力の実在・共有の畳み込み）に掛かる必要がある。
    partitioned = [expand_weight_parts(plan) for plan in plans]
    plans = [item.plan for item in partitioned]
    names = [plan.name for plan in plans]
    if len(set(names)) != len(names):
        raise DistError(f"モデル名が重複している: {names}")
    if default_model not in names:
        raise DistError(f"既定モデル '{default_model}' が組み立て対象 {names} に無い")
    assert_plan_paths(plans)
    assert_plan_limits(plans)
    assert_plan_sources(plans)
    assert_root_files(root_files or {})
    # 入力コンテナの全検証は**実在検査の後・1 バイトも書く前**（ヘッダしか読まないので、
    # ここに置いても数 GB の再読みにはならない）。
    assert_weight_components_verified(partitioned)
    # 越境参照は**複数モデルへ掛けてよい**。MUST: 指定役割の現物が全モデルで参照先とバイト
    # 同一（plan ごとの {@link external_refs} の突合 + 全 plan の参照一致）でなければ落とす —
    # 「同じ役割名が別バイトを指す」形が曖昧さの実体なので、モデル数で代理せずそれを直接
    # 検査する（食い違えば役割名と両モデル名を綴って fail loudly）。
    #
    # 分割形のコンテナも越境参照にできる — `container.parts` は**要素ごとに**従来の FileRef
    # 検査を通る配列なので（ADR 0038 §7 / ADR 0109 決定 3）、part 1 本を参照 1 つで指せる。
    # 展開（{@link expand_weight_parts}）が代表役割を part 役割へ割った後にここへ来るので、
    # {@link external_refs} は part ごとに参照先の現物を引き当てる。
    references: dict[str, dict[str, Any]] = {}
    if external is not None:
        resolved = [(item.plan.name, external_refs(external, item)) for item in partitioned]
        first_model, references = resolved[0]
        for model_name, refs in resolved[1:]:
            differing = sorted(
                role
                for role in set(references) | set(refs)
                if references.get(role) != refs.get(role)
            )
            if differing:
                raise DistError(
                    f"越境参照の指し先がモデル間で食い違う: {differing}"
                    f"（'{first_model}' と '{model_name}'）"
                    " — 同じ役割名が別バイトを指す形は 1 つの参照では宣言できない"
                )

    try:
        with staged_publication(out_dir) as staging:
            manifest, host_assets = _materialize_family(
                partitioned, staging, default_model, references
            )
            # 法的テキストは検証の**前**に置く — 例外側に居ることを組み立てのたびに
            # {@link verify_dist} で通しておかないと、例外が外れた回に据わってから気づく。
            for name, text in (root_files or {}).items():
                (staging / name).write_text(text, encoding="utf-8")
            verify_dist(staging)
            if render_card is not None:
                card = render_card(manifest, host_assets=host_assets)
                (staging / MODEL_CARD_FILENAME).write_text(card, encoding="utf-8")
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
                where = f"models.{model_name}.weights.{name}.{label}.container.parts"
                for index, ref in enumerate(entry["container"]["parts"]):
                    yield f"{where}[{index}]", ref
        for name, ref in model["assets"].items():
            yield f"models.{model_name}.assets.{name}", ref


def is_external_ref(ref: Mapping[str, Any]) -> bool:
    """別リポを指すファイル参照か（ADR 0038 §7 の `repo` / `revision` 席を持つ形）。

    自リポの 3 点セットと**同じ場所に並ぶ別の形**なので、判別子は欄の有無で持つ
    （{@link ExternalComponents}）。手元の現物と突き合わせる層は全部この判別で外す。
    """
    return "repo" in ref


def _assert_manifest_shape(manifest: Mapping[str, Any]) -> None:
    """`karume/5` の構造整合（hub のパーサが受理する形かを焼いた側でも見る）。

    ここが見るのは**この配布形が自分で閉じているか**だけ — `defaultModel` / `defaultQuant` の
    指し先、quant の weights 完全写像、weights のコンテナ入口、そしてレイアウト（ADR 0041 §9）。
    hub の全検査を写経しても正本が 2 つになるだけなので、写すのは「組み立てが壊れたら真っ先に
    破れる」規則に絞る（上限と part の規則は {@link assert_container_limits}）。
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
                # MUST: dtype ラベルは格納 dtype 語彙（{@link STORAGE_DTYPE_LABELS}）の内側。
                if label not in STORAGE_DTYPE_LABELS:
                    raise DistError(
                        f"{model_name}.weights.{name} の dtype ラベル '{label}' が格納 dtype"
                        f" 語彙 {sorted(STORAGE_DTYPE_LABELS)} に無い"
                    )
                # MUST: dtype エントリはコンテナの入口 1 つだけ（ADR 0109 決定 3）。`karume/4`
                # の `{shards, extras?}` を持ったままの manifest もここで落ちる — 形式識別子
                # だけ書き換えて中身が旧形の配布形は、hub が読めないのに焼く側では通ってしまう。
                if set(entry) != {"container"}:
                    raise DistError(
                        f"{model_name}.weights.{name}.{label} の欄が ['container'] でない"
                        f"（実際: {sorted(entry)}）"
                    )
                container = entry["container"]
                if not isinstance(container, dict) or set(container) != {"descriptor", "parts"}:
                    found = sorted(container) if isinstance(container, dict) else repr(container)
                    raise DistError(
                        f"{model_name}.weights.{name}.{label}.container の欄が"
                        f" ['descriptor', 'parts'] でない（実際: {found}）"
                    )
                descriptor = container["descriptor"]
                if not isinstance(descriptor, dict) or set(descriptor) != {"graph", "model"}:
                    found = sorted(descriptor) if isinstance(descriptor, dict) else repr(descriptor)
                    raise DistError(
                        f"{model_name}.weights.{name}.{label}.container.descriptor の欄が"
                        f" ['graph', 'model'] でない（実際: {found}）"
                    )
                assert_container_limits(f"{model_name}.weights.{name}.{label}.container", container)
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
        if is_external_ref(ref):
            # 越境参照の path は**参照先リポのレイアウト**なので、こちらの root 集合には
            # 収まらない（形だけを見る）。
            assert_external_ref(where, ref)
            continue
        root = ref["path"].split("/")[0]
        if root not in allowed_roots:
            raise DistError(
                f"{where}: path '{ref['path']}' の先頭が {sorted(allowed_roots)} のどれでもない"
                " — レイアウトはモデル別サブツリー + shared/（ADR 0041 §9）"
            )


def _declared_sizes(manifest: Mapping[str, Any]) -> dict[str, int]:
    """manifest が参照する**自リポの**全ファイルの `{path: size}`。重複 path は 3 点セットの
    一致を要求する。

    同一 path の重複参照はモデル間の共有そのもの（ADR 0041 §5）なので合法だが、`{size, sha256}`
    が食い違えば取得層のキャッシュが振動する — hub と同じ規則をここでも落とす。

    越境参照（{@link is_external_ref}）は**外す** — 手元に無いのが正しいファイルなので、
    実在検査にも宣言外ファイル検査にも掛けない（path 空間も参照先リポのもので、こちらの
    path とは別の名前空間）。
    """
    sizes: dict[str, int] = {}
    seen: dict[str, Mapping[str, Any]] = {}
    for where, ref in _declared_refs(manifest):
        if is_external_ref(ref):
            continue
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

    宣言外ファイルの例外は {@link META_PATHS} の 2 つ（`karume.json` と `README.md`）と
    {@link LEGAL_PATHS} の 2 つ（`LICENSE.md` / `NOTICE.md`）だけ — 前者は配布形そのものの説明、
    後者は上流ライセンスが再配布の条件として要求する法的テキストで、どちらも manifest が
    宣言する資産ではない。それ以外は従来どおり fail loudly（前回の組み立ての残骸や `io.*` の
    混入を後段へ見せない）。例外を名前でなく相対 path で持つのは共通で、下位ディレクトリに
    紛れ込んだ同名ファイルは従来どおり落ちる。

    越境参照（{@link ExternalComponents}）のファイルは**このリポに無いのが正しい**ので
    実在検査の対象にならない（{@link _declared_sizes} が外す）。形の検査だけは
    {@link _assert_manifest_shape} が {@link assert_external_ref} で掛ける。
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
    exempt = META_PATHS | LEGAL_PATHS
    present = {
        relative
        for path in out_dir.rglob("*")
        if path.is_file() and (relative := str(path.relative_to(out_dir))) not in exempt
    }
    extra = sorted(present - set(declared))
    if extra:
        raise DistError(f"manifest が宣言していないファイルが混ざっている: {', '.join(extra)}")
    return declared


# ---- ②' 越境コンポーネント参照（別リポの pin 済み revision を指す席） ----------


@dataclass(frozen=True)
class ExternalComponents:
    """指定した役割を、自リポへ格納する代わりに**別リポの pin 済み revision への参照**として
    宣言する opt-in（ADR 0038 §7 の optional `repo` / `revision` 席）。

    用途は「同じバイト列を 2 つのリポへ二重に上げない」こと 1 つだけ — 例えば turbo リポの
    text encoder は素版リポのものとバイト単位で同一なので、参照にすれば数 GB の再アップロード
    が要らなくなる。**指定が無ければ配布形は完全に自己完結**（既定の組み立ては 1 バイトも
    変わらない）。

    MUST: `revision` は **40 桁の commit sha**（{@link REVISION_RE}）。ブランチ名やタグを許すと
    指し先が後から動き、こちらが宣言した `size` / `sha256` と現物が食い違う — 参照の側は
    「不変のバイト列を指す」ことが成立条件そのものなので、pin でない綴りは受理しない。

    MUST: `size` / `sha256` は**ローカルの参照元 dist の実ファイル**から採る
    （{@link external_refs}）— 参照元の `karume.json` の値を写すと、表と現物が食い違う失敗様式が
    そのまま越境で復活する（manifest を導出物にした意味が消える）。

    `dist` / `model` は参照元の配布形（既に組み上がっているもの）とその中のモデル名。参照先の
    path はリポごとのレイアウト（モデル別サブツリー / `shared/`）で決まるので、こちら側の
    相対 path からは導けない — 参照元の `karume.json` が宣言している path を引く。
    """

    #: 参照先の HF リポジトリ（`<owner>/<name>`）。
    repo: str
    #: 参照先の commit sha（40 桁 hex）。
    revision: str
    #: ローカルに組み上がっている参照元の配布形（`size` / `sha256` の出所）。
    dist: Path
    #: 参照元 dist の中のモデル名（同じ相対 path をどのモデルの席から採るか）。
    model: str
    #: 参照へ差し替える役割名。分割形のコンテナを指す役割は、part 1 本ごとの参照
    #: （manifest の `container.parts` 配列の各要素）へ展開される（{@link external_refs}）。
    roles: tuple[str, ...]

    def __post_init__(self) -> None:
        if not REPO_RE.match(self.repo):
            raise DistError(f"越境参照の repo '{self.repo}' が '<owner>/<name>' の形でない")
        if not REVISION_RE.match(self.revision):
            raise DistError(
                f"越境参照の revision '{self.revision}' が 40 桁の commit sha でない"
                " — ブランチ名やタグは pin にならない（指し先が動くと宣言と現物が食い違う）"
            )
        assert_model_name(self.model)
        if not self.roles:
            raise DistError("越境参照する役割が 1 つも無い")


def assert_external_ref(where: str, ref: Mapping[str, Any]) -> None:
    """越境参照 1 つの形（欄の集合・repo の綴り・40 桁 hex の revision・path の収まり）。"""
    if set(ref) != set(EXTERNAL_REF_KEYS):
        raise DistError(
            f"{where}: 越境参照の欄が {sorted(EXTERNAL_REF_KEYS)} でない（実際: {sorted(ref)}）"
        )
    if not REPO_RE.match(str(ref["repo"])):
        raise DistError(f"{where}: repo '{ref['repo']}' が '<owner>/<name>' の形でない")
    if not REVISION_RE.match(str(ref["revision"])):
        raise DistError(
            f"{where}: revision '{ref['revision']}' が 40 桁の commit sha でない"
            " — 参照は不変のバイト列を指すのが成立条件"
        )
    assert_rel_path(str(ref["path"]), where)


def _external_ref(
    components: ExternalComponents,
    declared: set[str],
    role: str,
    artifact: Artifact,
    memo: dict[Path, str],
    *,
    part_seat: bool,
) -> dict[str, Any]:
    """越境ファイル参照 1 つ（{@link external_refs} の 1 要素）。

    `part_seat` は「この席が part 列を書けるか」— weights の dtype エントリだけが真で、
    assets は偽（1 ファイル参照しか書けない席）。偽の席で参照先が分割形だったら
    fail loudly: 黙って part 0 だけを指すと、残りのバイト列がどこからも取れない配布形が
    出来上がる。
    """
    seats = [
        f"{components.model}/{artifact.rel_path}",
        f"{SHARED_DIRNAME}/{artifact.rel_path}",
    ]
    found = [seat for seat in seats if seat in declared]
    if not found:
        if not part_seat and any(
            len(component_parts(components.dist / seat)) > 1 for seat in seats
        ):
            raise DistError(
                f"越境参照は分割形のコンテナに掛けられない: ['{role}']"
                "（1 役が複数 part へ割れているので 1 つの参照では指せない）"
                " — assets の席は 1 ファイル参照しか書けない"
            )
        raise DistError(
            f"役割 '{role}' のファイルが参照元 {components.dist} に無い"
            f"（{MANIFEST_FILENAME} は {seats} のどちらも宣言していない）"
            " — 参照先に無いものは参照できない"
        )
    source = components.dist / found[0]
    if not source.is_file():
        raise DistError(f"参照元が宣言するファイルの現物が無い: {source}")
    digest = sha256_file(source)
    local = _source_digest(artifact, memo)
    if digest != local:
        raise DistError(
            f"役割 '{role}' の参照先が自分で組むバイト列と違う: {source} は {digest}、"
            f"{artifact.rel_path} の出所は {local}"
            " — 中身の違う参照は「別のモデルの重み」を自分のものとして配る形になる"
        )
    return {
        "repo": components.repo,
        "revision": components.revision,
        "path": found[0],
        "size": source.stat().st_size,
        "sha256": digest,
    }


def external_refs(
    components: ExternalComponents, partitioned: PartitionedPlan
) -> dict[str, dict[str, Any]]:
    """役割名 → 越境ファイル参照 `{repo, revision, path, size, sha256}`。

    参照先の path は**参照元 dist の `karume.json` が宣言している path**から引く（モデル別
    サブツリーか `shared/` かは向こうの組み立てが決めた事実で、こちらからは導けない）。
    宣言に無い役割は fail loudly — 参照先に無いものは参照できない。

    **分割形のコンテナは part 役割ごとに 1 つの参照**を返す（manifest の `container.parts`
    配列の各要素が参照になる — ADR 0038 §7 / ADR 0109 決定 3）。`repo` / `revision` は全要素
    同一 MUST（越境は容器単位）で、`path` / `size` / `sha256` は part ごとに別。並びは
    {@link expand_weight_parts} が**現物から**解決した添字順そのままなので、先頭 = part 0 の
    規約は参照でも変わらない。part のファイル名は連番と総数を綴りに持つ（`-NNNNN-of-NNNNN`）
    ので、参照先の part 数がこちらと違えば「参照元に無い」で必ず落ちる（本数の食い違いは
    黙って解決しない）。

    MUST: 宣言と現物の突合を越境でも切らさない。`size` / `sha256` はローカルの実ファイルから
    採り、さらに**自分で組むはずだったバイト列と一致すること**まで確かめる — 一致しない参照は
    「別のモデルの重みを自分のものとして配る」形になり、shape も manifest も正しいまま沈黙する。
    分割形の役割はこの突合を**part 列の全要素**へ掛ける。

    NOTE: 参照元の参照（多段）は辿らない。{@link _declared_sizes} が越境参照を外すので、
    参照元がさらに別リポを指している席はここで「宣言に無い」として落ちる。
    """
    manifest_path = components.dist / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise DistError(f"越境参照の参照元に {MANIFEST_FILENAME} が無い: {manifest_path}")
    declared = set(_declared_sizes(json.loads(manifest_path.read_text(encoding="utf-8"))))
    plan = partitioned.plan
    memo: dict[Path, str] = {}
    refs: dict[str, dict[str, Any]] = {}
    for role in components.roles:
        # weights の役割は振り分け表に載っている（分割形なら part 役割へ割れていて、代表役割は
        # artifacts から消えている）。載っていない役割は assets の席なので代表役割のまま
        # 1 ファイル参照を引く。
        for member in partitioned.parts.get(role, (role,)):
            artifact = plan.artifacts.get(member)
            if artifact is None:
                raise DistError(
                    f"越境参照が知らない役割を指している: '{role}'"
                    f"（このモデルの役割: {sorted(plan.artifacts)}）"
                )
            refs[member] = _external_ref(
                components,
                declared,
                member,
                artifact,
                memo,
                part_seat=role in partitioned.parts,
            )
    return refs


def resolve_external_components(
    *,
    repo: str | None,
    revision: str | None,
    dist: Path | None,
    model: str | None,
    roles: Sequence[str] | None,
) -> ExternalComponents | None:
    """CLI の 5 指定を {@link ExternalComponents} へ解決する（1 つも無ければ `None`）。

    MUST: **全部揃うか 1 つも無いか**の 2 通りだけ。部分指定を黙って無視すると、参照するつもり
    の組み立てが自己完結の配布形として静かに出来上がる（数 GB を上げ直してから気づく）。
    """
    given = {
        "--ref-repo": repo,
        "--ref-revision": revision,
        "--ref-dist": dist,
        "--ref-model": model,
        "--ref-role": roles,
    }
    if all(value is None for value in given.values()):
        return None
    missing = sorted(name for name, value in given.items() if value is None)
    if missing:
        raise DistError(
            f"越境参照は 5 つの指定が揃って初めて成立する（足りない: {', '.join(missing)}）"
        )
    assert repo is not None and revision is not None  # missing が空 = 全部揃っている
    assert dist is not None and model is not None and roles is not None
    return ExternalComponents(
        repo=repo, revision=revision, dist=dist, model=model, roles=tuple(roles)
    )


# ---- ③ pipeline 別ディスパッチと CLI -----------------------------------------


#: モデルカードの描き手（manifest とリポ ID と「host 読みの資産」の勘定から本文 1 枚）。
#:
#: `host_assets`（part 0 の配布相対 path → 容器が宣言する資産の論理長の和）は manifest から
#: 導けない**容器の事実**なので、組み立てが引いて渡す（ADR 0109 決定 4 で PLE / `rope_base` が
#: 容器の中へ入り、独立したファイルの席が消えた）。
CardRenderer = Callable[[Mapping[str, Any], str, Mapping[str, int]], str]


@dataclass(frozen=True)
class Pipeline:
    """pipeline ごとに違うものだけを持つディスパッチ表の 1 行。

    共有部（配置・共有の畳み込み・sha256・宣言と現物の突合）は上の汎用関数が持つので、
    ここに並ぶのは「既定のモデル名 / モデルが属する配布リポ名 / 系列とモデル名から計画を組む
    手順 / モデルカードの描き手」の 4 つだけ。
    """

    default_model: str
    #: モデル名 → そのモデルを収める配布リポ名（`HF_OWNER` 抜き）。カードの Usage 例に綴る
    #: repo の正本（{@link resolve_repo}）で、呼び出し側の既定の出力先もここから綴る。
    #: 組むモデル全部が同じ名前を返すときだけ 1 リポの名前として成立する。
    repo_name: Callable[[str], str]
    plan: Callable[[Path, str], ModelPlan]
    #: モデルカードの**帰属プロファイル**（名前 → 描き手）。テンプレートは pipeline 固有でも、
    #: 帰属（出所・ライセンス・引用）は声のファミリーごとに別の事実になるため 1 対 1 ではない。
    card_profiles: Mapping[str, CardRenderer]
    #: 配布リポ**直下**へ入れる法的テキスト（ファイル名 → 中身）。受理する名前は
    #: {@link LEGAL_PATHS} だけ。上流の重みライセンスが再配布の条件としてライセンス文の
    #: コピーや Attribution Notice を要求する pipeline だけが渡す（既定は空 = 何も置かない）。
    root_files: Mapping[str, str] = field(default_factory=dict)


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


def resolve_repo(pipeline: Pipeline, models: Sequence[str], repo: str | None) -> str:
    """モデルカードの Usage 例に綴る HF リポ ID（`<owner>/<name>`）を決める。

    正本は pipeline の宣言（{@link Pipeline.repo_name}）で、`--repo` はそれを明示で上書きする口。

    MUST: 出力先のディレクトリ名からは導かない — 置き場はリポの事実ではないので、別名の
    ディレクトリ（ステージング）で焼いた配布形のカードが、実在しないリポを指したまま公開される。
    MUST: 組むモデルの宣言が 1 つに揃わないとき（モデル名をリポ名に含む pipeline で複数モデルを
    1 リポへ束ねるとき）は推定せず `--repo` を求めて落とす — 束ねたリポの名前は命名の決定で
    あって、モデル名の並びからは決まらない。
    """
    if repo is not None:
        if not REPO_RE.match(repo):
            raise DistError(f"--repo '{repo}' が '<owner>/<name>' の形でない")
        return repo
    names = sorted({pipeline.repo_name(model) for model in models})
    if len(names) != 1:
        raise DistError(
            f"モデル {' / '.join(models)} が宣言するリポ名が 1 つに揃わない"
            f"（{' / '.join(names)}）— 1 リポへ束ねる名前は --repo OWNER/NAME で明示する"
        )
    return f"{HF_OWNER}/{names[0]}"


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
    parser.add_argument(
        "--repo",
        metavar="OWNER/NAME",
        default=None,
        help="モデルカードの Usage 例に綴る HF リポ ID（既定: pipeline が宣言するリポ名。"
        "組むモデルの宣言が 1 つに揃わないときは必須）。--out のディレクトリ名とは独立で、"
        "別名のディレクトリへ焼いてもカードはこの名前を指す。上書きするときは --out も明示する",
    )
    # 越境参照（ADR 0038 §7）— **5 つ揃って初めて成立する opt-in**（1 つも無ければ従来どおり
    # 完全自己完結の配布形になる。部分指定は {@link resolve_external_components} が落とす）。
    parser.add_argument(
        "--ref-repo",
        dest="ref_repo",
        metavar="OWNER/NAME",
        default=None,
        help="越境参照する HF リポジトリ（指名した役割を自リポへ格納せずここへ向ける）",
    )
    parser.add_argument(
        "--ref-revision",
        dest="ref_revision",
        metavar="SHA",
        default=None,
        help="同・参照先の commit sha（40 桁 hex — ブランチ名やタグは pin にならないので不可）",
    )
    parser.add_argument(
        "--ref-dist",
        dest="ref_dist",
        type=Path,
        default=None,
        help="同・ローカルに組み上がっている参照元の配布形（size / sha256 の出所）",
    )
    parser.add_argument(
        "--ref-model",
        dest="ref_model",
        metavar="NAME",
        default=None,
        help="同・参照元 dist の中のモデル名",
    )
    parser.add_argument(
        "--ref-role",
        action="append",
        dest="ref_roles",
        metavar="ROLE",
        default=None,
        help="同・参照へ差し替える役割名（繰り返し可）",
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
    elif args.repo is not None:
        # 既定の出力先は pipeline の宣言から綴られるので、名前だけ上書きすると「あるリポの名前の
        # ディレクトリに、別のリポを指すカード」が組み上がる。
        raise DistError("--repo で上書きするときは --out も明示する（既定の出力先は宣言の名前）")
    elif default_out_dir is not None:
        out_dir = default_out_dir(pipeline, models)
    else:
        raise DistError("--out が要る（配布形の出力先）— 呼び出し側が既定を渡していない")
    # 帰属プロファイルとカードのリポ ID は**組み立ての前**に解決する — 誤った / 足りない指定で
    # 数 GB を並べてから最後の 1 枚で落ちる形にしない。越境参照の指定も同じ理由でここで形だけ
    # 確かめる。
    render_card = resolve_card_renderer(pipeline, args.card_profile)
    repo = resolve_repo(pipeline, models, args.repo)
    external = resolve_external_components(
        repo=args.ref_repo,
        revision=args.ref_revision,
        dist=args.ref_dist,
        model=args.ref_model,
        roles=args.ref_roles,
    )
    # MUST: 全モデルの計画（= 検査と読み取り）を配置の**前**に済ませる — 2 モデル目で落ちる
    # 形でも、1 モデル目だけ入った配布形を後段に見せない。
    plans = [pipeline.plan(args.series, model) for model in models]
    # モデルカードは組み立ての中（staging）で、**検証を通った manifest** から描く（表と現物が
    # 食い違ったまま説明だけ生えることがない順序・カードごと 1 回で据わる）。
    manifest = assemble_family(
        plans,
        out_dir,
        models[0],
        render_card=partial(render_card, repo=repo),
        root_files=pipeline.root_files,
        external=external,
    )
    verified = verify_dist(out_dir)
    for rel_path, size in sorted(verified.items()):
        print(f"{size:>12}  {rel_path}")
    for rel_path in sorted(META_PATHS | LEGAL_PATHS):
        meta = out_dir / rel_path
        if meta.is_file():
            print(f"{meta.stat().st_size:>12}  {rel_path}")
    # 越境参照は手元に現物が無いので `verify_dist` の一覧には出ない — 何を他リポへ預けたかが
    # 組み立ての出力から読めるように、指し先ごと 1 行で並べる（同じ席を複数の dtype 席が
    # 指すので path で一意化する）。
    borrowed = {
        (ref["repo"], ref["revision"], ref["path"]): ref["size"]
        for _, ref in _declared_refs(manifest)
        if is_external_ref(ref)
    }
    for ref_repo, revision, rel_path in sorted(borrowed):
        size = borrowed[(ref_repo, revision, rel_path)]
        print(f"{size:>12}  {ref_repo}@{revision[:12]} {rel_path}")
    listing = ", ".join(
        f"{name}({model['defaultQuant']})" for name, model in manifest["models"].items()
    )
    print(
        f"[dist] {out_dir} — {manifest['generator']} /"
        f" models {listing} / default {manifest['defaultModel']}"
    )


if __name__ == "__main__":
    main()
