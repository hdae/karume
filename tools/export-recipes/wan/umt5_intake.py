"""第三者の umT5-XXL 互換 encoder（HF のリポの safetensors 1 本）を取り込む（ADR 0122 決定 5 / 6 —
段 b。ADR 0088 の Civitai 取り込みの umT5 版）。

    uv run --group wan --inexact python -m wan.umt5_intake \\
        --repo <owner/name> --revision <40 桁の commit> --file <名前>.safetensors --name <名前>

責務は**取得・照合・記録まで**の 1 段。容器への変換は `wan.umt5_export`（`--intake`）、手元の
実験用ミラーの組み立ては `dist.py --pipeline umt5 --intake`（`wan.umt5_distribution`）が持つ。

1. HF API の値（pin した revision の LFS の sha256・`cardData` のライセンスとベース）を読む。
2. `config.json` を置く。上流が持っていれば構成の欄（{@link CONFIG_FIELDS} — グラフに効く欄と
   相対位置のバケットの構成）を本家の pin の config と突き合わせ、違えば fail loudly（TS が
   バケットの構成を固定しているため）。持たなければ本家の pin の config を写し、「構成は推定
   （本家の写し）」と記録する（その場合の照合は空 — `relative_attention_max_distance` は重みの
   形から導けない）。置いたバイト列の sha256 を記録し、{@link load_intake} が読むたびに照合する
   （取り込みの後の書き換えを、本家と照合していない構成として拒む）。
3. 指定の 1 本を pin した revision で取り、sha256 を API の値と突き合わせる（`.part` → 一致で
   rename）。受けるのは safetensors だけ（pickle の `.bin` を開くのは本家の pin の行だけ —
   決定 2）。取ったファイルは**名前も中身もそのまま** `inputs/umt5/<名前>/` に置く（F32 の写しは
   作らない —
   sha256 とファイル名が上流と一致したまま残り、後から API の値と突き合わせ直せる）。取り込み先
   （`--out` の下の `<名前>`）が `models/` の下なら、API を引く前に落とす（実 path どうし —
   {@link assert_outside_distribution_root}）。
4. ヘッダを読み、dtype（全テンソルが同じ BF16 か F32 — FP8・混在は拒む）・キー集合（Wan の
   `text_encoder` と同じ形 — tied な別名の対は 1 本と数え、両方を持つなら決定 2 の規則）・形を
   見る（`wan.umt5_export.inspect_intake_checkpoint`）。取り込み先に置いてよいファイル（{@link
   assert_only_intake_files} の許可の一覧）の外のものがあれば、記録を書く前に落とす。
5. 出所の記録 {@link INTAKE_FILE}（機械専有 — 人は追記しない）を書く。

**ライセンスの判定はしない**（ADR 0088 決定 5）。宣言されたライセンスを写すだけで、宣言が無いか、
再配布の条件を識別しない値（`unknown`・空・`license_name` の無い `other` など —
{@link UNIDENTIFIED_LICENSES}）なら未宣言の印 {@link UNDECLARED_LICENSE} を記録する。
未宣言の重みから容器や実験用ミラーを作るのは、書き手と組み立ての門で明示の引数
{@link ALLOW_UNDECLARED_LICENSE_FLAG} を付けたときだけ（決定 6）。
取り込みを行う人・エージェントが確認した内容は同じディレクトリの `license-review.md` に残す
（このコマンドはそちらに触らない — 機械の記録に人の追記を混ぜると、取り直しのたびに片方が消える）。

MUST: module 直下で torch / transformers / huggingface_hub を import しない — 配布 recipe
（`wan.umt5_distribution` — `import dist` は torch を読まない）がここから記録の読み口を引く。
重い依存は使う関数の中で import する（`tests/test_optional_group_imports.py`）。

MUST: 第三者のリポ名・commit を追跡されるファイルに書かない（ADR 0122 制約）— 固有名は実行時の
引数と、git 追跡外の `inputs/umt5/<名前>/intake.json` にだけ現れる。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from _shared.paths import DIST_ROOT, INPUTS_ROOT
from karume.dist import REPO_RE, DistError, assert_model_name
from wan.sources import UMT5_CONFIG, UMT5_SOURCES

#: 取り込み先の親（`inputs/umt5/<名前>/` — docs/assets-layout.md の入力素材の席）。
INTAKE_ROOT = INPUTS_ROOT / "umt5"

#: 機械が専有する出所の記録（人の確認記録は `license-review.md` へ）。
INTAKE_FILE = "intake.json"

#: 記録の形式の綴り（形を変えたら版を上げる — 読み口は知らない版を fail loudly で拒む）。
#: `/2` で `config.sha256` を足した。古い版の記録は移行せずに拒む（未公開の道具 — 取り込みを同じ
#: 引数でやり直せば今の版で書き直る）。
INTAKE_FORMAT = "karume-umt5-intake/2"

#: 取り込みを行う人・エージェントの確認の記録（このコマンドは書かない — モジュール doc）。
LICENSE_REVIEW_FILE = "license-review.md"

#: ライセンス未宣言の印（SPDX の `NOASSERTION` — ADR 0122 決定 6・未解決「未宣言の印の綴り」の
#: 決定）。記録の `license`・容器の `provenance.license`・実験用ミラーのカードの frontmatter で
#: 同じ綴りを使う。公開前の門（`tools/release/container_license.ts`）も同じ綴りで拒む。
UNDECLARED_LICENSE = "NOASSERTION"

#: 再配布の条件を識別しない値（{@link license_unidentified} — 前後の空白と大文字小文字を無視して
#: 比べる）。HF の `license: unknown` は公式の選択肢だが条件を何も名乗らないので、宣言が無いのと
#: 同じ扱いにする。公開前の門（`tools/release/container_license.ts` の `UNIDENTIFIED_LICENSES`）も
#: 同じ集合で拒む。
#:
#: NOTE: `other` は入れない — 既存の配布形（anima）の容器が `other` を名乗り、条件を NOTICE に書く
#: 形で公開している。取り込みでは `license_name` の無い `other` を {@link _card_license} が未宣言の
#: 印にし、記録の素の `other` は {@link load_intake} が拒む（どちらもこの集合とは別の検査）。
UNIDENTIFIED_LICENSES = frozenset({"", "noassertion", "unknown", "none"})

#: HF の `license: other`（識別子は `license_name` が持つ — 素の `other` は条件を名指さない）。
_OTHER_LICENSE = "other"

#: 未宣言の印を焼いた容器と実験用ミラーを作ることの明示（書き手と dist ドライバが同じ綴りで
#: 受ける）。
ALLOW_UNDECLARED_LICENSE_FLAG = "--allow-undeclared-license"

#: 受ける元の dtype（safetensors のヘッダの綴り）。BF16 は書き手が読みの時点で F32 へ広げる
#: （無損失）。FP8 などは情報を失った派生なので受けない。
INTAKE_DTYPES: tuple[str, ...] = ("BF16", "F32")

#: 構成の照合の欄（グラフに効く欄と相対位置のバケットの構成）。バケットの構成はグラフ記述に
#: 入らない（TS が `numBuckets` 32・`maxDistance` 128 を固定する）ので、違えば容器がバイト同一の
#: まま出力だけが変わりうる — 取り込みで本家の config と突き合わせる。
CONFIG_FIELDS: tuple[str, ...] = (
    "d_model",
    "d_kv",
    "d_ff",
    "num_layers",
    "num_heads",
    "vocab_size",
    "feed_forward_proj",
    "dense_act_fn",
    "is_gated_act",
    "layer_norm_epsilon",
    "relative_attention_num_buckets",
    "relative_attention_max_distance",
)

#: config の出所（`upstream` = 上流が持つ config を本家と照合して置いた / `base-copy` = 上流が
#: 持たないので本家の pin の config を写した — 構成は推定）。
CONFIG_SOURCES: tuple[str, ...] = ("upstream", "base-copy")

#: 構成の照合と写しの基準にする本家の行（`wan.sources.UMT5_SOURCES` のキー — umT5 の配布形の
#: モデル名 `wan.umt5_distribution.UMT5_DEFAULT_MODEL` と同じ綴り。あちらがこのモジュールを
#: import するので、こちらからは import できない — 一致はテストが見る）。
BASE_UPSTREAM = "xxl"

#: HF の commit SHA（40 桁の小文字 16 進）。ブランチ名やタグは pin にならない。
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")

#: 読み書きの刻み（11 GB 級の safetensors を写しながら sha256 を取る）。
_CHUNK_BYTES = 8 << 20

#: 記録の欄（多くも少なくもない — 知らない欄は fail loudly）。
_RECORD_KEYS = frozenset(
    {
        "format",
        "name",
        "repo",
        "revision",
        "file",
        "dtype",
        "license",
        "base_model",
        "config",
        "fetched_at",
    }
)


class Umt5IntakeError(RuntimeError):
    """取り込みの入力・上流の値・取り込んだ記録が想定（ADR 0122 決定 5）から外れた。"""


# ---------------------------------------------------------------------------
# 出所の記録
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntakeFile:
    """取り込んだ safetensors 1 本（上流のファイル名・バイト数・LFS の sha256）。"""

    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ConfigOrigin:
    """`config.json` の出所（{@link CONFIG_SOURCES}）と、照合した / 写した本家の pin、置いた
    バイト列の sha256（{@link load_intake} が読むたびに照合する）。"""

    source: str
    base_repo: str
    base_revision: str
    sha256: str


@dataclass(frozen=True)
class Umt5Intake:
    """`intake.json` の中身（{@link load_intake} が検査して組む）。

    `directory` は記録を読んだ置き場（記録には書かない — 取り込んだファイルと config の在処）。
    """

    directory: Path
    name: str
    repo: str
    revision: str
    file: IntakeFile
    #: 元の dtype（{@link INTAKE_DTYPES}）— 書き手はこの dtype だけを受ける。
    dtype: str
    #: 宣言されたライセンスの識別子、宣言が無ければ {@link UNDECLARED_LICENSE}。
    license: str
    #: 宣言されたベース（宣言が無ければ `None`）。
    base_model: tuple[str, ...] | None
    config: ConfigOrigin
    fetched_at: str

    @property
    def undeclared(self) -> bool:
        """上流がライセンスを宣言していない（未宣言の印を記録した）か。

        {@link load_intake} は識別しない値を印の綴りだけに絞るが、ここも同じ述語で判定する
        （記録を経ずに組んだ値でも、識別しない値を宣言済みとして扱わない）。"""
        return license_unidentified(self.license)

    @property
    def config_assumed(self) -> bool:
        """構成が推定（上流が config を持たず、本家の pin の config を写した）か。"""
        return self.config.source == "base-copy"

    @property
    def checkpoint(self) -> Path:
        return self.directory / self.file.name

    def to_document(self) -> dict[str, Any]:
        return {
            "format": INTAKE_FORMAT,
            "name": self.name,
            "repo": self.repo,
            "revision": self.revision,
            "file": {"name": self.file.name, "size": self.file.size, "sha256": self.file.sha256},
            "dtype": self.dtype,
            "license": self.license,
            "base_model": None if self.base_model is None else list(self.base_model),
            "config": {
                "source": self.config.source,
                "base": {"repo": self.config.base_repo, "revision": self.config.base_revision},
                "sha256": self.config.sha256,
            },
            "fetched_at": self.fetched_at,
        }


def license_unidentified(value: str) -> bool:
    """`value` が再配布の条件を識別しない値（{@link UNIDENTIFIED_LICENSES}）か。"""
    return value.strip().lower() in UNIDENTIFIED_LICENSES


def assert_outside_distribution_root(path: Path, *, root: Path) -> None:
    """`path` が `models/`（`root`）の下なら落とす（取り込み由来のファイル — ADR 0122 決定 5）。

    取り込み（`--out`）と実験用ミラーの組み立て（`dist.py --intake` の `--out`）が共有する 1 本。

    MUST: 実 path（symlink と `..` を解いた path）どうしで比べる — 綴りで比べると、`models/` を
    指す symlink や `..` を含む相対の path が素通りする。`models/` は公開の台本が
    `models/<repo>` をそのまま上げる席で、公開の経路を持たない取り込み由来のファイル（生の第三者の
    safetensors も、そこから書いた容器も）が居る理由が無い（ライセンスの宣言の有無によらない）。
    """
    resolved = path.resolve()
    real_root = root.resolve()
    if resolved.is_relative_to(real_root):
        raise Umt5IntakeError(
            f"{path}（実 path {resolved}）が {real_root} の下 — 取り込み由来のファイルは"
            " models/ に置かない"
        )


def assert_only_intake_files(directory: Path, file: str) -> None:
    """取り込み先 `directory` の直下に、置いてよいファイル（記録が名指すファイル `file`・
    `config.json`・{@link INTAKE_FILE}・{@link LICENSE_REVIEW_FILE}）の外のものが無いことを見る
    （サブディレクトリも外のものとして数える）。

    MUST: 在れば fail loudly — 読み口は記録のファイルだけを開くが、記録の外の重みが同じ席に
    居ると、どれが照合したものかを人も道具も取り違える（索引があれば `weight_map` の先を読む
    読み口に渡った時点で、照合していないファイルが黙って読まれる）。
    MUST: 許可の一覧で閉じる（名前どおりに比べる — 大文字小文字も区別する）— 重みの拡張子の
    一覧で開くと、大文字の拡張子・一覧に無い形式（`.pkl`・`.npz` など）・書きかけの `.part` が
    素通りする。
    MUST: symlink は許可の名前でも拒む — `is_file()` は symlink を辿るので、許可の名前の symlink で
    記録の外の重みを同じ席に置ける。
    """
    allowed = {file, UMT5_CONFIG, INTAKE_FILE, LICENSE_REVIEW_FILE}
    foreign = sorted(
        path.name + ("/" if path.is_dir() else "")
        for path in directory.iterdir()
        if path.name not in allowed or path.is_symlink() or not path.is_file()
    )
    if foreign:
        raise Umt5IntakeError(
            f"{directory} に記録の外のファイルがある: {foreign} — 置いてよいのは"
            f" {sorted(allowed)} だけ（記録の外の重みのファイルを取り違えないよう、種類によらず"
            "取り込み先には置かない）"
        )


def _assert_config_pinned(directory: Path, sha256: str, where: str) -> None:
    """取り込み先の `config.json` が記録の sha256（取り込みの時点に置いたバイト列）と同じか。

    MUST: 違えば・無ければ fail loudly — 書き手と参照はグラフとバケットの構成をこのファイルから
    読み直すので、取り込みの後に書き換えると（例 `relative_attention_max_distance`）本家との照合を
    経ていない構成で容器と golden ができる。
    """
    path = directory / UMT5_CONFIG
    if not path.is_file():
        raise Umt5IntakeError(f"{where}: {path} が無い — 取り込みをやり直す")
    actual = _sha256(path)
    if actual != sha256:
        raise Umt5IntakeError(
            f"{where}: {path} の sha256 {actual} が記録の {sha256} と違う — 取り込みの後に"
            " config を書き換えた（構成は本家と照合した取り込みの時点の値だけを受ける。直すなら"
            "取り込みをやり直す）"
        )


def assert_intake_name(name: str) -> str:
    """取り込みの名前（系列名 `umt5-xxl-<名前>-i8-dyn` と実験用ミラーのモデル名になる）を検査する。

    配布のモデル名の受理集合（ADR 0077）に縛り、本家の行の名前（`wan.sources.UMT5_SOURCES` の
    キー）とは重ねない — 重ねると配布の門が本家の行として扱い、出所の照合先が入れ替わる。
    """
    try:
        assert_model_name(name)
    except DistError as cause:
        raise Umt5IntakeError(str(cause)) from cause
    if name in UMT5_SOURCES:
        raise Umt5IntakeError(
            f"名前 '{name}' は本家の行（wan.sources.UMT5_SOURCES）と同じ — 取り込みには別の"
            "名前を付ける"
        )
    return name


def _assert_file_name(name: str) -> str:
    """取り込むファイル名（リポ直下の safetensors 1 本 — 区切り・`..`・pickle を受けない）。"""
    if not name or name in {".", ".."} or "/" in name or "\\" in name or ":" in name:
        raise Umt5IntakeError(f"ファイル名 {name!r} がリポ直下の basename でない")
    if not name.endswith(".safetensors"):
        raise Umt5IntakeError(
            f"ファイル {name!r} は safetensors でない — 取り込みは safetensors だけを受ける"
            "（pickle の .bin を開くのは本家の pin の行だけ — ADR 0122 決定 2）"
        )
    return name


def _require(mapping: Mapping[str, Any], key: str, kind: type, where: str) -> Any:
    value = mapping.get(key)
    if not isinstance(value, kind) or (isinstance(value, bool) and kind is not bool):
        raise Umt5IntakeError(f"{where}: '{key}' が {kind.__name__} でない（{value!r}）")
    return value


def load_intake(directory: Path) -> Umt5Intake:
    """取り込み済みのディレクトリから記録を読み、形を検査して返す。

    MUST: 知らない形式・欄の過不足・綴りの外れは fail loudly — 書き手は記録の `dtype` だけを受け、
    容器と配布の門は記録の `license` / `revision` を出所の正本にするので、壊れた記録を黙って
    読むと出所を偽る容器ができる。
    """
    path = directory / INTAKE_FILE
    if not path.is_file():
        raise Umt5IntakeError(
            f"{path} が無い — 先に `python -m wan.umt5_intake` で取り込む（記録の無い重みは"
            "読まない）"
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    where = str(path)
    if not isinstance(document, dict):
        raise Umt5IntakeError(f"{where}: JSON の object でない")
    if document.get("format") != INTAKE_FORMAT:
        raise Umt5IntakeError(
            f"{where}: format {document.get('format')!r} が {INTAKE_FORMAT} でない（古い版の記録は"
            " 移行しない — `python -m wan.umt5_intake` を同じ引数でやり直す）"
        )
    if set(document) != _RECORD_KEYS:
        raise Umt5IntakeError(
            f"{where}: 欄が違う（不足 {sorted(_RECORD_KEYS - set(document))} /"
            f" 余剰 {sorted(set(document) - _RECORD_KEYS)}）"
        )
    name = assert_intake_name(_require(document, "name", str, where))
    repo = _require(document, "repo", str, where)
    if REPO_RE.match(repo) is None:
        raise Umt5IntakeError(f"{where}: repo {repo!r} が '<owner>/<name>' の形でない")
    revision = _require(document, "revision", str, where)
    if _COMMIT_SHA.fullmatch(revision) is None:
        raise Umt5IntakeError(f"{where}: revision {revision!r} が 40 桁の commit SHA でない")
    file = _require(document, "file", dict, where)
    if set(file) != {"name", "size", "sha256"}:
        raise Umt5IntakeError(f"{where}: file の欄が name / size / sha256 でない（{sorted(file)}）")
    sha256 = _require(file, "sha256", str, f"{where} file")
    if _SHA256.fullmatch(sha256) is None:
        raise Umt5IntakeError(f"{where}: file.sha256 {sha256!r} が小文字 16 進 64 桁でない")
    size = _require(file, "size", int, f"{where} file")
    if size <= 0:
        raise Umt5IntakeError(f"{where}: file.size {size} が正でない")
    dtype = _require(document, "dtype", str, where)
    if dtype not in INTAKE_DTYPES:
        raise Umt5IntakeError(f"{where}: dtype {dtype!r} は {list(INTAKE_DTYPES)} の外")
    license_id = _require(document, "license", str, where)
    if not license_id.strip():
        raise Umt5IntakeError(f"{where}: license が空（未宣言なら {UNDECLARED_LICENSE}）")
    if license_unidentified(license_id) and license_id != UNDECLARED_LICENSE:
        # 取り込みは識別しない値を印の綴りへ寄せて書く。それ以外の綴りは手で書き換えた記録で、
        # 容器へそのまま焼くと公開前の門が綴りの違いで読み落としうる（ここで 1 通りに絞る）。
        raise Umt5IntakeError(
            f"{where}: license {license_id!r} は再配布の条件を識別しない値 — 未宣言の印は"
            f" {UNDECLARED_LICENSE} の綴りだけを受ける"
        )
    if license_id.strip().lower() == _OTHER_LICENSE:
        raise Umt5IntakeError(
            f"{where}: license {license_id!r} は再配布の条件を識別しない値 — 取り込みは"
            f" license_name の識別子を記録する（無ければ {UNDECLARED_LICENSE}）"
        )
    base_model = document["base_model"]
    if base_model is not None and (
        not isinstance(base_model, list)
        or not base_model
        or not all(isinstance(item, str) and item for item in base_model)
    ):
        raise Umt5IntakeError(f"{where}: base_model が null か空でない文字列の並びでない")
    config = _require(document, "config", dict, where)
    base = _require(config, "base", dict, f"{where} config")
    source = _require(config, "source", str, f"{where} config")
    if (
        set(config) != {"source", "base", "sha256"}
        or set(base) != {"repo", "revision"}
        or source not in CONFIG_SOURCES
    ):
        raise Umt5IntakeError(
            f"{where}: config が {{source: {list(CONFIG_SOURCES)},"
            " base: {repo, revision}, sha256} でない"
        )
    config_sha256 = _require(config, "sha256", str, f"{where} config")
    if _SHA256.fullmatch(config_sha256) is None:
        raise Umt5IntakeError(
            f"{where}: config.sha256 {config_sha256!r} が小文字 16 進 64 桁でない"
        )
    _assert_config_pinned(directory, config_sha256, where)
    return Umt5Intake(
        directory=directory,
        name=name,
        repo=repo,
        revision=revision,
        file=IntakeFile(
            name=_assert_file_name(_require(file, "name", str, where)), size=size, sha256=sha256
        ),
        dtype=dtype,
        license=license_id,
        base_model=None if base_model is None else tuple(base_model),
        config=ConfigOrigin(
            source=source,
            base_repo=_require(base, "repo", str, f"{where} config.base"),
            base_revision=_require(base, "revision", str, f"{where} config.base"),
            sha256=config_sha256,
        ),
        fetched_at=_require(document, "fetched_at", str, where),
    )


def assert_license_intent(intake: Umt5Intake, *, allowed: bool) -> None:
    """未宣言の印の取り込みなら、明示（{@link ALLOW_UNDECLARED_LICENSE_FLAG}）を要求する。

    MUST: 明示が無ければ 1 バイトも書く前に落とす（ADR 0122 決定 6）。判定するのは「上流が
    ライセンスを宣言していない」という機械で決まる事実だけで、許諾の解釈はしない。
    """
    if intake.undeclared and not allowed:
        raise Umt5IntakeError(
            f"取り込み '{intake.name}' の上流はライセンスを宣言していない（{UNDECLARED_LICENSE}）—"
            f" 手元の実験用に限って作るなら {ALLOW_UNDECLARED_LICENSE_FLAG} を明示する"
            "（ADR 0122 決定 6）"
        )


# ---------------------------------------------------------------------------
# HF API（差し替えの席 — テストは模擬の応答を渡す）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HubFile:
    """revision の中のファイル 1 本（バイト数と LFS の sha256 — LFS でなければ `None`）。"""

    size: int
    sha256: str | None


@dataclass(frozen=True)
class HubRevision:
    """取り込みが使う HF API の値（`model_info(revision=…, files_metadata=True)` から）。"""

    #: API が返した commit（指定の revision と同じであることを見る）。
    sha: str
    files: Mapping[str, HubFile]
    #: `cardData` の `license`（`other` なら `license_name`）。宣言が無い・再配布の条件を識別しない
    #: 値（{@link license_unidentified}）なら `None`。
    license: str | None
    #: `cardData` の `base_model`（宣言が無ければ `None`）。
    base_model: tuple[str, ...] | None


class Hub(Protocol):
    def revision(self, repo: str, revision: str) -> HubRevision: ...

    def download(self, repo: str, revision: str, filename: str) -> Path: ...


def _card_license(license_id: Any, license_name: Any, repo: str) -> str | None:
    """`cardData` の `license` / `license_name` → 識別子。

    宣言が無い・再配布の条件を識別しない値（`unknown`・空・`license_name` の無い `other` など —
    {@link license_unidentified}）は `None`（記録は未宣言の印になる）。文字列でない値は形の壊れで、
    fail loudly。
    """
    if license_id is None:
        return None
    if not isinstance(license_id, str):
        raise Umt5IntakeError(f"{repo}: cardData の license {license_id!r} が文字列でない")
    if license_id.strip().lower() != _OTHER_LICENSE:
        return None if license_unidentified(license_id) else license_id
    if license_name is None:
        return None
    if not isinstance(license_name, str):
        raise Umt5IntakeError(f"{repo}: cardData の license_name {license_name!r} が文字列でない")
    return None if license_unidentified(license_name) else license_name


def _card_base_model(value: Any, repo: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    values = [value] if isinstance(value, str) else value
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(v, str) and v for v in values)
    ):
        raise Umt5IntakeError(
            f"{repo}: cardData の base_model {value!r} が文字列か文字列の並びでない"
        )
    return tuple(values)


class HfHub:
    """huggingface_hub で読む既定の口（取得は HF の既定キャッシュを通す — 取得済みなら
    落とさない）。"""

    def revision(self, repo: str, revision: str) -> HubRevision:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        card = info.card_data
        files = {
            sibling.rfilename: HubFile(
                size=int(sibling.size or 0),
                sha256=None if sibling.lfs is None else sibling.lfs.sha256,
            )
            for sibling in info.siblings or ()
        }
        return HubRevision(
            sha=str(info.sha),
            files=files,
            license=_card_license(
                getattr(card, "license", None), getattr(card, "license_name", None), repo
            ),
            base_model=_card_base_model(getattr(card, "base_model", None), repo),
        )

    def download(self, repo: str, revision: str, filename: str) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo, filename, revision=revision))


# ---------------------------------------------------------------------------
# 取り込み
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _place_checkpoint(source: Path, dest: Path, expected: HubFile) -> None:
    """取ったファイルを `dest` へ写して据える（既に同じ sha256 で在れば何もしない — 再実行安全）。

    書くのは `<dest>.part` で、sha256 とバイト数が API の値と合ってから rename する — 途中で
    切れたファイルが完成品の名前で残ると、次の実行がそれを「取り込み済み」と読む。
    """
    assert expected.sha256 is not None
    if dest.exists():
        actual = _sha256(dest)
        if actual != expected.sha256:
            raise Umt5IntakeError(
                f"同名で別物が置かれている: {dest} — API {expected.sha256} / 実測 {actual}"
                "（上流が差し替えたか手置きの別ファイル。確かめてから消す）"
            )
        return
    part = dest.with_name(f"{dest.name}.part")
    digest = hashlib.sha256()
    written = 0
    with source.open("rb") as reader, part.open("wb") as writer:
        while chunk := reader.read(_CHUNK_BYTES):
            writer.write(chunk)
            digest.update(chunk)
            written += len(chunk)
    actual = digest.hexdigest()
    if actual != expected.sha256 or written != expected.size:
        raise Umt5IntakeError(
            f"取ったファイルが API の値と違う: sha256 API {expected.sha256} / 実測 {actual}・"
            f"バイト数 API {expected.size} / 実測 {written}（{part} は残す — 確かめてから消す）"
        )
    part.rename(dest)


def _place_config(
    hub: Hub, info: HubRevision, repo: str, revision: str, dest: Path
) -> ConfigOrigin:
    """`config.json` を置き、その出所を返す（モジュール doc の手順 2）。

    記録する sha256 は照合した（写した）バイト列そのものから取る — 置いた後に読み直して取ると、
    その間の書き換えを記録が追認する。
    """
    base = UMT5_SOURCES[BASE_UPSTREAM].source
    base_bytes = hub.download(base.repo, base.revision, UMT5_CONFIG).read_bytes()
    if UMT5_CONFIG not in info.files:
        source, placed = "base-copy", base_bytes
    else:
        source, placed = "upstream", hub.download(repo, revision, UMT5_CONFIG).read_bytes()
        upstream = json.loads(placed.decode("utf-8"))
        reference = json.loads(base_bytes.decode("utf-8"))
        differing = {
            key: (upstream.get(key), reference.get(key))
            for key in CONFIG_FIELDS
            if upstream.get(key) != reference.get(key)
        }
        if differing:
            raise Umt5IntakeError(
                f"{repo}@{revision} の {UMT5_CONFIG} が本家 {base.repo}@{base.revision} と構成の"
                f"欄で違う（上流, 本家）: {differing} — グラフかバケットの構成が違う umT5 は互換の"
                " encoder でない"
            )
    (dest / UMT5_CONFIG).write_bytes(placed)
    return ConfigOrigin(
        source=source,
        base_repo=base.repo,
        base_revision=base.revision,
        sha256=hashlib.sha256(placed).hexdigest(),
    )


def intake(
    repo: str,
    revision: str,
    file: str,
    name: str,
    *,
    hub: Hub,
    out_root: Path = INTAKE_ROOT,
) -> Umt5Intake:
    """上流の safetensors 1 本を `<out_root>/<name>/` へ取り込み、記録を書いて返す。"""
    if REPO_RE.match(repo) is None:
        raise Umt5IntakeError(f"--repo {repo!r} が '<owner>/<name>' の形でない")
    if _COMMIT_SHA.fullmatch(revision) is None:
        raise Umt5IntakeError(
            f"--revision {revision!r} が 40 桁の commit SHA でない（ブランチ名やタグは pin に"
            "ならない）"
        )
    _assert_file_name(file)
    assert_intake_name(name)
    destination = out_root / name
    # API を引く前・1 バイトも置く前に落とす。
    assert_outside_distribution_root(destination, root=DIST_ROOT)
    info = hub.revision(repo, revision)
    if info.sha != revision:
        raise Umt5IntakeError(f"API の commit {info.sha} が指定の revision {revision} と違う")
    entry = info.files.get(file)
    if entry is None:
        raise Umt5IntakeError(
            f"{repo}@{revision} に {file!r} が無い（在るもの: {sorted(info.files)}）"
        )
    if entry.sha256 is None:
        raise Umt5IntakeError(
            f"{repo}@{revision} の {file!r} が LFS でない — sha256 を API から取れない"
        )

    record_path = destination / INTAKE_FILE
    if record_path.is_file():
        previous = json.loads(record_path.read_text(encoding="utf-8"))
        if (
            previous.get("repo"),
            previous.get("revision"),
            previous.get("file", {}).get("name"),
        ) != (
            repo,
            revision,
            file,
        ):
            raise Umt5IntakeError(
                f"{destination} は別の取り込み"
                f"（{previous.get('repo')}@{previous.get('revision')}）の"
                " 置き場 — 別の名前を付けるか、確かめてから消す"
            )
    destination.mkdir(parents=True, exist_ok=True)
    config = _place_config(hub, info, repo, revision, destination)
    _place_checkpoint(hub.download(repo, revision, file), destination / file, entry)

    from wan.umt5_export import inspect_intake_checkpoint

    dtype = inspect_intake_checkpoint(destination, file, entry.sha256)
    record = Umt5Intake(
        directory=destination,
        name=name,
        repo=repo,
        revision=revision,
        file=IntakeFile(name=file, size=entry.size, sha256=entry.sha256),
        dtype=dtype,
        license=(
            UNDECLARED_LICENSE
            if info.license is None or license_unidentified(info.license)
            else info.license
        ),
        base_model=info.base_model,
        config=config,
        fetched_at=datetime.now(UTC).isoformat(),
    )
    record_path.write_text(
        json.dumps(record.to_document(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # 書いた記録は読み口の検査を通ること（書き手と読み手で形を割らない）。
    return load_intake(destination)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--repo", required=True, help="上流の HF リポ（<owner>/<name>）")
    parser.add_argument("--revision", required=True, help="pin する commit SHA（40 桁）")
    parser.add_argument("--file", required=True, help="取り込む safetensors（リポ直下の 1 本）")
    parser.add_argument(
        "--name",
        required=True,
        help="取り込みの名前（inputs/umt5/<名前>/・系列 umt5-xxl-<名前>-i8-dyn）",
    )
    parser.add_argument(
        "--out", type=Path, default=INTAKE_ROOT, help="取り込み先の親（models/ の下は受けない）"
    )
    args = parser.parse_args(argv)
    record = intake(args.repo, args.revision, args.file, args.name, hub=HfHub(), out_root=args.out)
    print(json.dumps(record.to_document(), ensure_ascii=False, indent=1))
    print(f"[umt5-intake] {record.directory / INTAKE_FILE}", flush=True)
    if record.undeclared:
        print(
            f"[umt5-intake] 上流はライセンスを宣言していない（{UNDECLARED_LICENSE}）— 書き手と"
            f"組み立ては {ALLOW_UNDECLARED_LICENSE_FLAG} を明示したときだけ手元の実験用に作れる。"
            "確認した内容は"
            f" {record.directory / LICENSE_REVIEW_FILE} に残す",
            flush=True,
        )
    print(
        "[umt5-intake] 次: uv run --group wan --inexact python -m wan.umt5_export write"
        f" --intake {record.directory}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
