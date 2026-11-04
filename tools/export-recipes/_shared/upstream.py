"""上流 checkpoint の出所を**手元の実物から**読む（容器の `provenance` へ焼く値と、その突合）。

容器の `provenance`（container-v1 §2.3）は、単一の `krm` を 1 ファイルとして持ち出しても出所が
辿れるための席である。ここへ定数を焼くと、`--model-dir` が別の checkpoint を指しても容器は同じ
出所を名乗る（Base の重みが Small の `apache-2.0` を名乗る形 — 2026-09-24 全域レビュー E-RC6-1）。
そこで値は `--model-dir` の実物から導く:

- **revision** は `hf download <repo> --local-dir <dir>` が残す記録
  `<dir>/.cache/huggingface/download/<file>.metadata` の 1 行目（commit SHA — 2 行目は etag・
  3 行目は取得時刻）。
- **ライセンス識別子**は同じ取得で落ちた `README.md` の front matter の `license:`
  （`other` のときは `license_name:` — HF で独自ライセンスを名乗る綴り）。

MUST: 記録が無い・読めないときは **fail loudly**（{@link UpstreamProvenanceError}）。
「分からないので省く」「既定値を名乗る」はどちらも容器の出所を偽る形になる。

組み立ての側は、焼かれた出所を帰属表の値と突き合わせる（{@link assert_upstream_provenance}）—
配布の門を「モデル名の表」ではなく「容器の中身の出所」で閉じるための 1 本。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from _shared.container_read import read_provenance
from karume.container import Provenance
from karume.dist import DistError
from karume.modelcard import HF_OWNER, CardMetadata, Upstream

#: `hf download --local-dir` が取得の記録を置く場所（`<model_dir>` からの相対）。
SNAPSHOT_RECORD_DIR = Path(".cache") / "huggingface" / "download"

#: 既定で revision を読むファイル（HF の checkpoint にはほぼ必ずある 1 本）。
SNAPSHOT_RECORD_FILE = "config.json"

#: HF の commit SHA（40 桁の小文字 16 進）。
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")

#: front matter の最上位キー 1 行（`license: apache-2.0` / `license: "mit"`）。
_FRONT_MATTER_KEY = re.compile(r"^(?P<key>[A-Za-z_][A-Za-z0-9_]*):\s*(?P<value>.*?)\s*$")


class UpstreamProvenanceError(ValueError):
    """容器へ焼く出所（revision / ライセンス識別子）が決まらない。

    リポの流儀は `Error` サブクラス（`DistError` / `ContainerReadError`）。素の `ValueError`
    だと、呼び手の `except` がこの**出所の門**と「引数が変」一般を区別できない。
    """


def hub_cache() -> Path:
    """The Hugging Face hub cache, as huggingface_hub itself resolves it
    (`huggingface_hub.constants.HF_HUB_CACHE` — read once, when huggingface_hub is imported).

    MUST: imported here, not at module scope — the default sync of this workspace does not
    install huggingface_hub (it comes with transformers, which the gemma4 recipes run with, and
    with the wan / moss / sbv2 groups), and every recipe imports this module.
    """
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def hub_snapshot_dir(repo: str, revision: str) -> Path:
    """The local hub cache's snapshot of `repo` at commit `revision`.

    A path only (nothing is read or fetched): the readers of the files fail loudly when the
    snapshot is not there. MUST: `revision` is a commit SHA — a branch or a tag is not a pin, and
    the cache names a snapshot directory by the commit it resolved.
    """
    if _COMMIT_SHA.fullmatch(revision) is None:
        raise UpstreamProvenanceError(
            f"{repo}: revision {revision!r} is not a commit SHA (40 lower-case hex digits)"
        )
    return hub_cache() / f"models--{repo.replace('/', '--')}" / "snapshots" / revision


def snapshot_revision(model_dir: Path, file: str = SNAPSHOT_RECORD_FILE) -> str:
    """`hf download --local-dir` の記録から、手元の checkpoint の commit SHA を読む。"""
    record = model_dir / SNAPSHOT_RECORD_DIR / f"{file}.metadata"
    if not record.is_file():
        raise UpstreamProvenanceError(
            f"{record} が無い — `hf download <repo> --local-dir {model_dir}` で取得した"
            " checkpoint でないと上流の revision を名乗れない"
        )
    lines = record.read_text(encoding="utf-8").splitlines()
    revision = lines[0].strip() if lines else ""
    if _COMMIT_SHA.fullmatch(revision) is None:
        raise UpstreamProvenanceError(
            f"{record} の 1 行目 {revision!r} が commit SHA（40 桁の 16 進）でない"
        )
    return revision


def _front_matter(readme: Path) -> dict[str, str]:
    """`README.md` の front matter の**最上位のスカラー**だけを拾う（入れ子と列は読まない）。"""
    lines = readme.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        raise UpstreamProvenanceError(f"{readme} が front matter（'---' の行）で始まらない")
    fields: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return fields
        matched = _FRONT_MATTER_KEY.match(line)
        if matched is not None and matched["value"]:
            fields[matched["key"]] = matched["value"].strip("\"'")
    raise UpstreamProvenanceError(
        f"{readme} の front matter が閉じていない（2 本目の '---' が無い）"
    )


def snapshot_license(model_dir: Path) -> str:
    """手元の checkpoint の `README.md` の front matter からライセンス識別子を読む。

    `license: other` は「識別子の無い独自ライセンス」を示す HF の綴りで、識別子そのものは
    `license_name:` にある（container-v1 §2.3 の `license` は識別子の席 — `other` では持ち出した
    容器からライセンスが特定できない）。
    """
    readme = model_dir / "README.md"
    if not readme.is_file():
        raise UpstreamProvenanceError(f"{readme} が無い — 上流のライセンス識別子を読めない")
    fields = _front_matter(readme)
    license_id = fields.get("license")
    if license_id is None:
        raise UpstreamProvenanceError(f"{readme} の front matter に 'license' が無い")
    return _license_identifier(license_id, fields.get("license_name"), str(readme))


def _license_identifier(license_id: str, license_name: str | None, where: str) -> str:
    """HF の `license` / `license_name` の組 → 容器へ焼くライセンス識別子。"""
    if license_id != "other":
        return license_id
    if license_name is None:
        raise UpstreamProvenanceError(
            f"{where} は 'license: other' で 'license_name' が無い — 識別子が決まらない"
        )
    return license_name


def card_license_identifier(metadata: CardMetadata, where: str) -> str:
    """カードの frontmatter（上流で実地確認した値）から、容器へ焼くライセンス識別子を引く。

    手元に上流の README が無い family（HF キャッシュから直接読む anima・Booth 由来の sbv2 FN）
    の席。`other` を焼かないのは {@link snapshot_license} と同じ理由 — `other` は「識別子の無い
    独自ライセンス」を示す HF の綴りで、持ち出した容器からライセンスが特定できない
    （2026-09-24 全域レビュー W-RC3-6）。
    """
    return _license_identifier(metadata.license, metadata.license_name, where)


def snapshot_provenance(model_dir: Path, *, notice: str | None) -> Provenance:
    """`--model-dir` の実物から、容器へ焼く出所を組む（ライセンス識別子 + revision）。"""
    return Provenance(
        license=snapshot_license(model_dir),
        notice=notice,
        upstream_revision=snapshot_revision(model_dir),
    )


def assert_upstream_provenance(container: Path, *, license: str, revision: str | None) -> None:
    """配布候補の容器が名乗る出所を、帰属表の値（と取得した revision）へ突き合わせる。

    `revision` は期待する上流 revision — 手元の checkpoint の取得記録から導いた値（siglip2・
    depth_anything）か、帰属表に固定した値（wan・birefnet）。`None` なら「revision を名乗って
    いること」までを見る。

    MUST: 容器の出所で門を閉じる。モデル名の表だけで閉じると、別の checkpoint を焼いた容器を
    系列 path へ置いたときに格納・入出力形・前処理の突合がすべて通り、別ライセンスの重みが
    帰属表のライセンスを名乗る配布形になる（2026-09-24 全域レビュー E-RC6-1）。
    """
    provenance = read_provenance(container)
    if provenance.license != license:
        raise DistError(
            f"{container} の provenance.license が {provenance.license!r} で、帰属表の"
            f" {license!r} と違う — 帰属表に無い checkpoint を焼いた容器は配らない"
        )
    if provenance.upstream_revision is None:
        raise DistError(
            f"{container} が provenance.upstreamRevision を持たない — 上流の revision を"
            "名乗れない容器は配らない（台本で焼き直す）"
        )
    if revision is not None and provenance.upstream_revision != revision:
        raise DistError(
            f"{container} の provenance.upstreamRevision が {provenance.upstream_revision}、"
            f"期待する revision は {revision} — 別の revision から焼いた容器"
        )


#: HF account (the owner part of a `base_model` repository ID) → who the cards name as its
#: publisher in the house non-endorsement sentence ({@link karume.modelcard.introduction}). Spelled
#: as this repository already names them where it does — in a licence's notice, a copyright line
#: or a card's attribution section; an account with no such name is spelled as the account itself.
#:
#: MUST: an account missing here fails the card ({@link hf_upstreams}) — adding an upstream means
#: stating who published it, never a placeholder.
UPSTREAM_AUTHORS: Mapping[str, str] = {
    # anima — the licensor in the CircleStone licence's attribution notice.
    "circlestone-labs": "CircleStone Labs LLC",
    # birefnet / lucida — the copyright lines of the bundled MIT licence.
    "ZhengPeng7": "ZhengPeng",
    "egeorcun": "egeorcun",
    "depth-anything": "depth-anything",
    # gemma4 / gemma4_qat / siglip2 / umt5 — Gemma's licensor and the publisher of the others.
    "google": "Google",
    # irodori — the copyright lines of the bundled MIT licence.
    "Aratako": "Aratako",
    "sbintuitions": "SB Intuitions",
    # moss — the card's attribution section and the Qwen licence's copyright line.
    "OpenMOSS-Team": "OpenMOSS",
    "Qwen": "Alibaba Cloud",
    "black-forest-labs": "Black Forest Labs",
    # sbv2 — the voices' and the text encoder's publishers.
    "litagin": "litagin",
    "ku-nlp": "ku-nlp",
    "rufflet17": "rufflet17",
    # wan — the card's attribution section ("released by Wan-AI").
    "Wan-AI": "Wan-AI",
}


def hf_upstreams(repos: Sequence[str]) -> tuple[Upstream, ...]:
    """The house sentence's upstreams for HF repositories (a card's `base_model`, typically): each
    repository ID with its account's publisher from {@link UPSTREAM_AUTHORS}.

    MUST: a repository of our own account (`karume.modelcard.HF_OWNER`) is refused — a conversion
    of our own release is not unofficial, and the sentence would disclaim ourselves.
    """
    upstreams: list[Upstream] = []
    for repo in repos:
        owner, _, name = repo.partition("/")
        if not owner or not name:
            raise ValueError(f"{repo!r} is not an HF repository ID (<owner>/<name>)")
        if owner == HF_OWNER:
            raise ValueError(f"{repo} is our own release — it is not an unofficial conversion")
        author = UPSTREAM_AUTHORS.get(owner)
        if author is None:
            raise ValueError(
                f"{repo}: the publisher of HF account '{owner}' is not in UPSTREAM_AUTHORS"
                " — a card that cannot say whose work it converts is not drawn"
            )
        upstreams.append(Upstream(repo, author))
    return tuple(upstreams)
