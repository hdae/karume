"""civitai の checkpoint を AIR / URL で取り込む（重み + 機械 provenance を `inputs/` へ）。

    uv run python -m anima.civitai --air "urn:air:anima:checkpoint:civitai:2544636@2983680"
    uv run python -m anima.civitai --url \
        "https://civitai.com/models/2544636/wai-anima?modelVersionId=2983680"
    uv run python -m anima.civitai --url "https://civitai.com/models/2544636"  # 版一覧だけ出す

責務は**取得・検証・記録まで**の 1 段。diffusers レイアウトへの組み直しは
{@link anima.single_file}、配布カードの帰属は {@link anima.card} が持つ。

**ライセンスの判定はしない** — API の許諾 4 欄と説明本文（HTML）を `civitai.json` へ写すだけで、
可否は取り込む人が原文を読んで決める（2026-09-01 裁定 — 確認フラグは設けない）。人が確認した
記録は同じディレクトリの `license-review.md` に置く運用で、このコマンドはそちらに一切触らない
（機械が書く `civitai.json` に人の追記を混ぜると、再取得のたびに片方が消える）。

MUST: torch / diffusers / huggingface_hub を import しない（`anima.single_file` の NOTE と同じ
理由 — 依存グループ非同期の環境でも動く必要がある）。ここは重み 1 本を落として記録するだけで、
標準ライブラリで足りる。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from _shared.paths import INPUTS_ROOT

#: メタ取得の口（無認証で読める）。
API_ROOT = "https://civitai.com/api/v1"

#: DL トークンの置き場（https://civitai.com/user/account の API Keys で発行する）。
TOKEN_ENV = "CIVITAI_API_TOKEN"

#: 取り込み先の親（`inputs/anima/civitai-<versionId>/` — docs/assets-layout.md）。
DEFAULT_OUT = INPUTS_ROOT / "anima"

#: 機械が専有する取り込み記録。人の確認記録はここではなく `license-review.md` へ。
PROVENANCE_FILE = "civitai.json"

MIB = 1024 * 1024

#: 読み書きの刻みと、進捗を出す間隔（4GB 級の checkpoint を無言で待たせない）。
CHUNK_BYTES = 8 * MIB
PROGRESS_BYTES = 256 * MIB

#: AIR（civitai の資産 URN）の綴り。`ecosystem` / `type` は**検証しない**（記録するだけ —
#: 上流が語彙を増やしたときに、こちらの門が理由で取り込めなくなる形にしない）。
AIR_PATTERN = re.compile(
    r"urn:air:(?P<ecosystem>[^:]+):(?P<type>[^:]+):(?P<source>[^:@]+):(?P<model>\d+)"
    r"(?:@(?P<version>\d+))?"
)

#: モデルページの path（`/models/<id>` + 任意のスラグ）。版は query の `modelVersionId`。
MODEL_PATH_PATTERN = re.compile(r"/models/(?P<model>\d+)(?:/[^/]*)?/?")

#: 配布名の受理集合（ADR 0077 — `karume.dist` の `MODEL_NAME_RE` の**逐語の写し**）。
#:
#: MUST: 写しにするのは上の MUST（torch を import しない）から — `karume.dist` を import すると
#: `karume/__init__.py` 経由で `karume.convert` が torch を引く（2026-09-05 実測）ので、正本を
#: 直接呼べない。正本が規則を足したらここも同じ日に直す。
MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]*$")

#: 配布名に使えない予約名（`karume.dist` の `SHARED_DIRNAME` — 共有ファイルの席と衝突する）。
RESERVED_MODEL_NAME = "shared"

#: 導出名から落とすファミリ名（`WAI-ANIMA` → `wai` — 前置の `anima-` と二重にしない）。
FAMILY_TOKEN = "anima"


@dataclass(frozen=True)
class Air:
    """AIR の中身（`urn:air:{ecosystem}:{type}:civitai:{model_id}@{version_id}`）。"""

    ecosystem: str
    type: str
    model_id: int
    version_id: int

    @property
    def urn(self) -> str:
        """自前で組み立て直した綴り（API が返す値との突合に使う）。"""
        return f"urn:air:{self.ecosystem}:{self.type}:civitai:{self.model_id}@{self.version_id}"


@dataclass(frozen=True)
class Target:
    """モデルページが指すもの。版は URL では省ける（省いた場合は版一覧の案内へ回る）。"""

    model_id: int
    version_id: int | None


def parse_air(text: str) -> Air:
    """AIR を読む。civitai 以外の source と、版を持たない AIR は受けない。"""
    match = AIR_PATTERN.fullmatch(text.strip())
    if match is None:
        raise SystemExit(
            f"AIR として読めない: {text}"
            "（想定: urn:air:<ecosystem>:<type>:civitai:<modelId>@<versionId>）"
        )
    source = match["source"]
    if source != "civitai":
        raise SystemExit(f"civitai 以外の source は扱わない: {source}（{text}）")
    if match["version"] is None:
        # 版が無いと「最新」を勝手に選ぶことになるが、上流は順序の付かない系統を同時に配る
        # （ADR 0077）。どれを取り込むかは人が決める。
        raise SystemExit(f"版まで指定する: {text}@<versionId>")
    return Air(
        ecosystem=match["ecosystem"],
        type=match["type"],
        model_id=int(match["model"]),
        version_id=int(match["version"]),
    )


def parse_url(text: str) -> Target:
    """モデルページの URL を読む（`?modelVersionId=` があれば版まで、無ければモデルだけ）。"""
    parts = urlsplit(text.strip())
    if parts.netloc.removeprefix("www.") != "civitai.com":
        raise SystemExit(f"civitai.com の URL ではない: {text}")
    match = MODEL_PATH_PATTERN.fullmatch(parts.path)
    if match is None:
        raise SystemExit(f"モデルページの URL として読めない: {text}（想定: /models/<modelId>）")
    values = parse_qs(parts.query).get("modelVersionId", [])
    if not values:
        return Target(model_id=int(match["model"]), version_id=None)
    if not values[0].isdigit():
        raise SystemExit(f"modelVersionId が数値でない: {values[0]}（{text}）")
    return Target(model_id=int(match["model"]), version_id=int(values[0]))


def _slug(text: str) -> str:
    """受理集合の外を `-` へ潰して均す（連続と両端の区切りは残さない）。"""
    slug = re.sub(r"[^a-z0-9._-]+", "-", text.lower())
    return re.sub(r"-+", "-", slug).strip("-._")


def _assert_name(name: str) -> str:
    """配布名として通る綴りかを最後に確かめる（ADR 0077 の受理集合）。"""
    if MODEL_NAME_PATTERN.match(name) is None:
        raise SystemExit(
            f"配布名に使えない文字がある: {name}（受理集合: A-Za-z0-9._- ・先頭のドット不可）"
        )
    if name == RESERVED_MODEL_NAME:
        raise SystemExit(
            f"配布名に '{RESERVED_MODEL_NAME}' は使えない（共有ファイルの席と衝突する）"
        )
    return name


def derive_model_name(model_name: str, version_name: str) -> str:
    """上流の名乗りから配布名を導く（`WAI-ANIMA` + `v1.0(base 1.0)` → `anima-wai-v1.0`）。

    機械正規化は綴りを受理集合へ落とすだけで、版の名乗りを別書式へ揃えることはしない
    （ADR 0088 — 日付名乗りと semver 風が混在するが、こちらで揃えると出所ページと突き
    合わせられなくなる。逐語の名乗りは `civitai.json` が保持する）。落とすのは**丸括弧の
    補足**（`(base 1.0)` は上流ページ上の但し書きで、版の識別には要らない）と、モデル名側の
    ファミリ名だけ。
    """
    version = _slug(re.sub(r"\([^)]*\)", "", version_name))
    tokens = [token for token in _slug(model_name).split("-") if token and token != FAMILY_TOKEN]
    model = "-".join(tokens)
    if not model or not version:
        raise SystemExit(f"名前を導けない: モデル名 {model_name!r} / 版名 {version_name!r}")
    return _assert_name(f"{FAMILY_TOKEN}-{model}-{version}")


def _masked(url: str) -> str:
    """表示用（トークンは残さない — 端末とログに撒かない）。"""
    return re.sub(r"([?&]token=)[^&]*", r"\1***", url)


#: 送信する User-Agent。urllib 既定の `Python-urllib/…` は civitai 側が名指しで 403 に
#: する（2026-09-01 実測 — 同じ URL が curl と本 UA では 200）。名乗りは出所が辿れる形に。
USER_AGENT = "karume-export-recipes/anima.civitai"


def _open(url: str) -> Any:
    """GET を張る。失敗は status と行き先を添えて止める（黙って空を返さない）。"""
    request = urllib.request.Request(url, headers={"Accept": "*/*", "User-Agent": USER_AGENT})
    try:
        return urllib.request.urlopen(request)
    except urllib.error.HTTPError as error:
        raise SystemExit(f"HTTP {error.code}: {_masked(url)}") from error
    except urllib.error.URLError as error:
        raise SystemExit(f"到達できない: {_masked(url)}（{error.reason}）") from error


def _get_json(url: str) -> dict[str, Any]:
    with _open(url) as response:
        payload = response.read()
    return json.loads(payload)


def fetch_version(version_id: int) -> dict[str, Any]:
    """版のメタ（air / files / hashes / usageControl / 版の説明）。"""
    return _get_json(f"{API_ROOT}/model-versions/{version_id}")


def fetch_model(model_id: int) -> dict[str, Any]:
    """モデルのメタ（名前 / type / 許諾 4 欄 / 説明 / 版の一覧）。"""
    return _get_json(f"{API_ROOT}/models/{model_id}")


def select_file(files: list[dict[str, Any]]) -> dict[str, Any]:
    """本体 1 本を選ぶ。同梱の Text Encoder / VAE は取らない（base 側を共有するため）。"""
    primary = [entry for entry in files if entry.get("primary")]
    if len(primary) == 1:
        return primary[0]
    if not primary:
        models = [entry for entry in files if entry.get("type") == "Model"]
        if len(models) == 1:
            return models[0]
    listing = [(entry.get("name"), entry.get("type")) for entry in files]
    raise SystemExit(f"本体のファイルを特定できない: {listing}")


def _require_token() -> str:
    token = os.environ.get(TOKEN_ENV, "").strip()
    if not token:
        raise SystemExit(
            f"DL には civitai の API トークンが要る — env {TOKEN_ENV} に設定する"
            "（https://civitai.com/user/account の API Keys）"
        )
    return token


def _with_token(url: str, token: str) -> str:
    """トークンは**クエリで**渡す。

    MUST: `Authorization` ヘッダにしない — DL の口は S3 の署名 URL へ 307 で飛び、リダイレクト
    先へヘッダが持ち越されると署名と衝突して 400 で落ちうる。クエリなら civitai 側で解決される。
    """
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}token={token}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path, expected_sha256: str) -> None:
    """本体を取って据える。既に同じ sha で在れば何もしない（再実行安全）。

    書くのは `<dest>.part` で、**sha が API の値と合ってから** rename する — 途中で切れた
    ファイルが完成品の名前で残ると、次の実行がそれを「取得済み」と読んでしまう。
    """
    expected = expected_sha256.casefold()
    if dest.exists():
        actual = _sha256(dest)
        if actual == expected:
            print(f"[civitai] 取得済み（sha256 一致）: {dest}", flush=True)
            return
        raise SystemExit(
            f"同名で別物が置かれている: {dest} — API {expected} / 実測 {actual}"
            "（上流が差し替えたか手置きの別ファイル。確かめてから消す）"
        )

    token = _require_token()
    part = dest.with_name(f"{dest.name}.part")
    source = _with_token(url, token)
    print(f"[civitai] GET {_masked(source)}", flush=True)
    digest = hashlib.sha256()
    written = 0
    reported = 0
    with _open(source) as response, part.open("wb") as handle:
        while chunk := response.read(CHUNK_BYTES):
            handle.write(chunk)
            digest.update(chunk)
            written += len(chunk)
            if written - reported >= PROGRESS_BYTES:
                reported = written
                print(f"[civitai] {dest.name}: {written // MIB} MiB", flush=True)

    actual = digest.hexdigest()
    if actual != expected:
        raise SystemExit(
            f"sha256 が API の値と違う: API {expected} / 実測 {actual}"
            f"（{part} は残す — 中身を確かめてから消す）"
        )
    part.rename(dest)
    print(f"[civitai] {dest}（{written // MIB} MiB・sha256 一致）", flush=True)


def _resolve_air(version: dict[str, Any], requested: Air | None) -> str | None:
    """API が発行した AIR を採る（自前組み立てと食い違えば、その旨を出してから API 側）。"""
    served = version.get("air")
    if served is None:
        return requested.urn if requested is not None else None
    if requested is not None and served != requested.urn:
        print(f"[civitai] 警告: AIR が指定と違う — 指定 {requested.urn} / API {served}", flush=True)
    return str(served)


def build_record(
    model: dict[str, Any],
    version: dict[str, Any],
    file: dict[str, Any],
    derived_name: str,
    air: str | None,
) -> dict[str, Any]:
    """`civitai.json` の中身（許諾は**判定せず**そのまま写す）。"""
    return {
        "air": air,
        "model_id": model["id"],
        "version_id": version["id"],
        "model_name": model["name"],
        "version_name": version["name"],
        "derived_name": derived_name,
        "base_model": version.get("baseModel"),
        "file": {
            "name": file["name"],
            "size_kb": file.get("sizeKB"),
            "sha256": file["hashes"]["SHA256"].casefold(),
            "download_url": file["downloadUrl"],
        },
        "permissions": {
            "allowNoCredit": model.get("allowNoCredit"),
            "allowCommercialUse": model.get("allowCommercialUse"),
            "allowDerivatives": model.get("allowDerivatives"),
            "allowDifferentLicense": model.get("allowDifferentLicense"),
            "usageControl": version.get("usageControl"),
        },
        "descriptions": {
            "model": model.get("description"),
            "version": version.get("description"),
        },
        "fetched_at": datetime.now(UTC).isoformat(),
    }


def print_versions(model_id: int) -> None:
    """版を選び直すための一覧（版未指定の URL で来たときの案内モード — DL はしない）。"""
    model = fetch_model(model_id)
    print(f"[civitai] {model['name']}（{model.get('type')}）の版:", flush=True)
    for version in model.get("modelVersions", []):
        files = version.get("files", [])
        primary = next((entry.get("name") for entry in files if entry.get("primary")), "-")
        print(f"  {version['id']}  {version['name']}  {primary}", flush=True)
    print(
        f"[civitai] 版を選んで指定し直す: --url '…/models/{model_id}?modelVersionId=<id>'"
        " か --air 'urn:air:…@<id>'",
        flush=True,
    )


def fetch_checkpoint(
    model_id: int,
    version_id: int,
    out: Path = DEFAULT_OUT,
    name: str | None = None,
    requested_air: Air | None = None,
) -> Path:
    """版 1 つを `<out>/civitai-<versionId>/` へ取り込み、置いた本体の path を返す。"""
    version = fetch_version(version_id)
    served_model_id = int(version["modelId"])
    if served_model_id != model_id:
        # URL の `/models/<id>` は版と食い違いうる（別モデルの版 id を貼った URL）。版が持つ
        # `modelId` が正 — 許諾欄を取りに行く先を間違えない。
        print(
            f"[civitai] 警告: modelId が指定と違う — 指定 {model_id} / API {served_model_id}",
            flush=True,
        )
    model = fetch_model(served_model_id)

    derived_name = _assert_name(name) if name else derive_model_name(model["name"], version["name"])
    file = select_file(version.get("files", []))
    others = [entry for entry in version.get("files", []) if entry is not file]
    if others:
        listing = ", ".join(f"{entry.get('name')}（{entry.get('type')}）" for entry in others)
        print(f"[civitai] 取得しない同梱ファイル: {listing}", flush=True)
    hashes = file.get("hashes", {})
    if "SHA256" not in hashes:
        raise SystemExit(f"API が sha256 を持たない: {file.get('name')}（突合できない）")

    destination = out / f"civitai-{version['id']}"
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / file["name"]
    download(file["downloadUrl"], checkpoint, hashes["SHA256"])

    record = build_record(model, version, file, derived_name, _resolve_air(version, requested_air))
    (destination / PROVENANCE_FILE).write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[civitai] {destination / PROVENANCE_FILE}", flush=True)
    print(
        "[civitai] 次: uv run --group anima python -m anima.single_file "
        f"--checkpoint {checkpoint} --out ../../outputs/misc/anima-diffusers/{derived_name}",
        flush=True,
    )
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--air", help="urn:air:<ecosystem>:<type>:civitai:<modelId>@<versionId>")
    source.add_argument("--url", help="モデルページの URL（版は ?modelVersionId=<id>）")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="取り込み先の親")
    parser.add_argument("--name", help="配布名の上書き（既定は上流の名乗りから導く）")
    args = parser.parse_args()

    if args.air:
        air = parse_air(args.air)
        fetch_checkpoint(air.model_id, air.version_id, args.out, args.name, requested_air=air)
        return

    target = parse_url(args.url)
    if target.version_id is None:
        print_versions(target.model_id)
        return
    fetch_checkpoint(target.model_id, target.version_id, args.out, args.name)


if __name__ == "__main__":
    main()
