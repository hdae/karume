"""系列出力（`outputs/series/`）の旧 shard 列を容器（`krm`）へ移す — recipe 側のドライバ。

    uv run python -m migrate_series                          # 参照されている系列を全部
    uv run python -m migrate_series --only deberta --dry-run
    uv run python -m migrate_series --series-dir /path/to/series

core の `karume migrate` に系列モードを**足さない**のは、「どのディレクトリがどの部品名か」を
知っているのが family の `distribution.py` だけだから（ADR 0065 — core は家族を知らない）。
系列のレイアウトは family ごとに違い、しかも**ディレクトリ名は部品名ではない**:

- 系列直下に 1 部品（siglip2 / birefnet / depth-anything / vowel-detector / gemma4 /
  gemma4-qat / minicpm5 / embeddinggemma）— ディレクトリ名は系列名。
- サブディレクトリに部品が並ぶ（anima / sbv2 / irodori / dacvae / deberta）— 綴りが一致
  しないものがある（`caption-proj` → `caption_proj`・`decoder` → `codec_decoder`・deberta の
  `full-24layer` → SBV2 が消費する `text_encoder` 席）。

**旧入力は読むだけ**で、`krm` は同じディレクトリに並べる（旧 shard と sidecar は消さない —
退避はディレクトリ単位で別に行う）。触らないものは golden（`io.*` / `io-i8a8.*` / `greedy.*` /
`drafter-golden.*` / `case.*` / `t-embed.*` / `pipeline.*` / `trim.*`）・`ple.probe.safetensors`・
`*.json` / `*.txt` / `*.md`・資産ディレクトリ（`text/` / `tokenizer/` / `pipeline/` /
`host/`）である。

**例外が 1 つある**: QAT 系列の `reference.json` は書き換える（{@link rewrite_qat_reference}）—
記録が名乗る PLE の本数は旧 sidecar の shard 本数で、畳んだ後はどこにも存在しない数になる。
移行は記録が指す実体を作り直しているので、記録も追随させる。

sidecar（重みではないが同じ容器で配るバイト列）は 2 種だけ実在する:

- `rope_base.safetensors`（anima の `-dyn` 系列）→ 資産 `rope_base`
  （`karume.migrate.EXTRA_ASSETS` の写し先）。
- `ple.json` + `ple-NNNNN-of-NNNNN.safetensors`（gemma4 / gemma4-qat の製品系列）→ 索引
  `ple_index` + `ple.values.<k>` / `ple.scales.<k>`（`karume.migrate.ple_sidecar_assets` —
  リポ丸ごとモードと同じ 1 本）。

MUST: 1 部品でも落ちたらそこで止まる（残りを移して最後にまとめない）。落ちた系列は**部分的に
据わったまま残る**ので、手で消してから回し直す（消すのは移行 CLI の仕事ではない — 移行は
「出力先は空から作る」が成立条件で、次の実行は `MigrateError: 出力先に前回の成果物が残っている`
で止まる）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from _shared.paths import INPUTS_ROOT, SERIES_ROOT
from anima import export as anima_export
from anima.distribution import ANIMA_WEIGHTS, ROPE_BASE_ASSET
from birefnet import export as birefnet_export
from birefnet.distribution import BIREFNET_ROLE
from deberta import export as deberta_export
from depth_anything import export as depth_anything_export
from depth_anything.distribution import DEPTH_ANYTHING_ROLE
from embeddinggemma import export as embeddinggemma_export
from gemma4 import export as gemma4_export
from gemma4.distribution import GEMMA4_DRAFTER_ROLE, GEMMA4_DRAFTER_SUFFIX, GEMMA4_ROLE
from gemma4_qat import export as gemma4_qat_export
from gemma4_qat.config import REFERENCE_SCHEMA
from gemma4_qat.distribution import REFERENCE_BLOCKS_FIELD, REFERENCE_SHARDS_FIELD
from irodori import export as irodori_export
from irodori.dacvae import export as dacvae_export
from irodori.distribution import IRODORI_CODEC_ROLES, IRODORI_SERIES_ROLES
from karume.container import BLOCK_MAX_BYTES, AssetInput, Provenance, sequence_siblings
from karume.migrate import (
    EXTRA_ASSETS,
    MigrateError,
    migrate_component,
    ple_sidecar_assets,
    read_ple_index,
)
from karume.ple import PLE_INDEX_ASSET
from minicpm5 import export as minicpm5_export
from sbv2 import export as sbv2_export
from sbv2.distribution import SBV2_FRONT_ROLE, SBV2_SERIES_PREFIX, SBV2_VOICE_ROLE
from siglip2 import export as siglip2_export
from siglip2.distribution import SIGLIP2_ROLE
from vowel_detector import export as vowel_detector_export
from vowel_detector.distribution import VOWEL_DETECTOR_GRAPH_ROLE

#: 旧 shard 列の代表名（どの family も `model` — 書き手の `MODEL_FILE` の stem）。
SHARD_STEM = "model"

#: 旧 shard の拡張子と、据える容器の拡張子。
SHARD_SUFFIX = ".safetensors"
MODEL_SUFFIX = ".krm"

#: 容器の資産へ畳む sidecar のファイル名（core の写し先と 1:1）。
ROPE_BASE_FILE = f"{ROPE_BASE_ASSET}{SHARD_SUFFIX}"
PLE_INDEX_FILE = "ple.json"

#: 系列のどのディレクトリにも同居してよい `.safetensors` の接頭辞（golden と検収用の参照）。
#: **表に無い `.safetensors` は fail loudly** — 新しい sidecar が生えた日に「移したつもりで
#: 資産が 1 つ落ちた容器」を黙って作らないための門（移行は 1 度きりなので、落ちたことに
#: 気づく機会が後から来ない）。
#:
#: 後ろの 4 つは**部品ディレクトリの外**に居る golden で、読み手は全て TS 側のテスト:
#: `case.*`（irodori の pipeline / dacvae の host 参照）・`t-embed.*`
#: （`irodori.pipeline_ref.T_EMBED_FILE`）・`pipeline.*`（anima / irodori の段ごとの参照）・
#: `trim.*`（`irodori.dacvae.host.TRIM_FILE`）。
GOLDEN_PREFIXES: tuple[str, ...] = (
    "io.",
    "io-i8a8.",
    "greedy.",
    "drafter-golden.",
    "ple.probe",
    "case.",
    "t-embed.",
    "pipeline.",
    "trim.",
)

#: QAT family の名前（{@link FAMILIES} の要素と {@link rewrite_qat_reference} の引き当てで
#: 綴りを割らないための 1 本）。
QAT_FAMILY = "gemma4-qat"

#: QAT 系列が持つ検収記録（移行が schema を上げて書き換える唯一のファイル）。
QAT_REFERENCE_FILE = "reference.json"

#: 旧 sidecar 世代の記録が名乗る PLE の本数 → 畳んだ資産の block 本数。綴りの正本は**読み手**
#: （`gemma4_qat.distribution` の突合）— 写しを持つと、片方だけ動いた日に「書き換えたのに
#: 配布側が見ていない欄」ができる。
QAT_SHARDS_FIELD = REFERENCE_SHARDS_FIELD
QAT_BLOCKS_FIELD = REFERENCE_BLOCKS_FIELD

#: 変換しない系列の接尾（実験の記録と棄却の記録）。**読み手がいない**ので移さない — 配布にも
#: TS 側の検収にも入らず、`tools/llm-baseline/data.py` の profile も 2026-09-23 に外した
#: （llm-speed は配布形専用になった）。実測そのものは `docs/research/` に文章として残る。
UNREFERENCED_SUFFIXES: tuple[str, ...] = ("-probe", "-rejected")


class SeriesMigrationError(ValueError):
    """系列の移行の前提が破れた（分類できない系列・表に無い部品・畳み先の無い sidecar）。

    リポの流儀は専用の `Error` サブクラス（`DistError` / `Sbv2FamilyError`）— 素の
    `ValueError` だと、呼び手の `except` がこの門と「引数が変」一般を区別できない。
    """


@dataclass(frozen=True)
class Component:
    """移す部品 1 つ（旧 shard 列 1 本 + 同じ容器へ畳む sidecar）。"""

    #: 旧 shard 列が並ぶディレクトリ（`krm` もここへ据わる）。
    directory: Path
    #: 容器のグラフ名 = 配布形の部品名 = `karume.json` の weights のキー（container-v1 §12）。
    graph_name: str

    @property
    def representative(self) -> Path:
        """旧 shard 列の代表 path（`karume.legacy.resolve_shards` が連番へ解決する）。"""
        return self.directory / f"{SHARD_STEM}{SHARD_SUFFIX}"

    @property
    def container(self) -> Path:
        """据わる `krm` の代表 path（現物は連番の part 列）。"""
        return self.directory / f"{SHARD_STEM}{MODEL_SUFFIX}"


@dataclass(frozen=True)
class Family:
    """1 family の系列レイアウト（系列名の判定・部品名・出所）。

    `roles` が空なら**系列直下に 1 部品**で、部品名は {@link flat_role} が系列名から決める。
    空でなければ**サブディレクトリ名 → 部品名**の表で、表に無いディレクトリは fail loudly。
    """

    name: str
    #: 系列名 → この family か（綴りの正本は各 family の distribution / export の定数）。
    matches: Callable[[str], bool]
    #: 系列名 → 出所（容器へ焼く `provenance`。recipe の export が渡す値と同じ）。
    provenance: Callable[[str], Provenance]
    #: サブディレクトリ名 → 部品名。
    roles: Mapping[str, str] = field(default_factory=dict)
    #: 系列直下に 1 部品を置く family の「系列名 → 部品名」。
    flat_role: Callable[[str], str] | None = None


def sbv2_model(series: str) -> str:
    """sbv2 の系列名 → 話者名（`sbv2-F1-i8` → `F1`）。

    出所（ライセンスと上流の版）は声のファミリーごとに違い、引けるのは話者名からだけなので
    ここで剥がす（`sbv2.export.default_out_root` の逆向き）。
    """
    stem = series.removeprefix(f"{SBV2_SERIES_PREFIX}-")
    for dtype in ("f16", "i8", "i4"):
        stem = stem.removesuffix(f"-{dtype}")
    return stem


#: deberta の variant ディレクトリ → 部品名（3 つともは SBV2 の `text_encoder` 席を焼く）。
DEBERTA_ROLES: Mapping[str, str] = {
    variant.name: deberta_export.GRAPH_NAME for variant in deberta_export.VARIANTS.values()
}

#: sbv2 のターゲットディレクトリ → 部品名。配布に載る 2 席は `distribution.py` の定数で、
#: golden 専用の 3 本（`dp` / `flow` / `dec`）は部品名を持たないのでターゲット名そのもの。
SBV2_ROLES: Mapping[str, str] = {
    sbv2_export.TARGET_FRONT: SBV2_FRONT_ROLE,
    sbv2_export.TARGET_VOICE: SBV2_VOICE_ROLE,
    sbv2_export.TARGET_DP: sbv2_export.TARGET_DP,
    sbv2_export.TARGET_FLOW: sbv2_export.TARGET_FLOW,
    sbv2_export.TARGET_DEC: sbv2_export.TARGET_DEC,
}

#: family の表。判定の順序が意味を持つのは gemma4-qat / gemma4 だけ（前者が後者の接頭辞を
#: 共有する）なので、**qat を先に置く**。
FAMILIES: tuple[Family, ...] = (
    Family(
        "anima",
        # `anima-f16`（共有の text 経路 + VAE）・`<モデル>-f16`（自前の text_conditioner）・
        # `<モデル>-<格納>-dyn`（transformer）。tokenizer / pipeline 系列は旧 shard 列を持たない。
        lambda name: name.startswith("anima-"),
        lambda _: anima_export.PROVENANCE,
        roles={role: role for role in ANIMA_WEIGHTS},
    ),
    Family(
        "birefnet",
        # 系列名は `birefnet_series_name(checkpoint, model)` = 上流リポ名の小文字 + 解像度。
        lambda name: name.startswith(("birefnet-", "lucida-")),
        lambda _: birefnet_export.PROVENANCE,
        flat_role=lambda _: BIREFNET_ROLE,
    ),
    Family(
        "depth-anything",
        lambda name: name.startswith("depth-anything-"),
        lambda _: depth_anything_export.PROVENANCE,
        flat_role=lambda _: DEPTH_ANYTHING_ROLE,
    ),
    Family(
        "siglip2",
        lambda name: name.startswith("siglip2-"),
        lambda _: siglip2_export.PROVENANCE,
        flat_role=lambda _: SIGLIP2_ROLE,
    ),
    Family(
        "vowel-detector",
        lambda name: name.startswith("vowel-detector-"),
        lambda _: vowel_detector_export.PROVENANCE,
        flat_role=lambda _: VOWEL_DETECTOR_GRAPH_ROLE,
    ),
    Family(
        "deberta",
        # `deberta{,-i8,-i4}`（`deberta.export.DEFAULT_OUT_ROOTS` の綴り）。
        lambda name: name == "deberta" or name.startswith("deberta-"),
        lambda _: deberta_export.PROVENANCE,
        roles=DEBERTA_ROLES,
    ),
    Family(
        "sbv2",
        lambda name: name.startswith(f"{SBV2_SERIES_PREFIX}-"),
        lambda name: sbv2_export.sbv2_provenance(
            INPUTS_ROOT / SBV2_SERIES_PREFIX / sbv2_model(name)
        ),
        roles=SBV2_ROLES,
    ),
    Family(
        "irodori",
        lambda name: name.startswith("irodori-"),
        lambda _: irodori_export.PROVENANCE,
        roles=IRODORI_SERIES_ROLES,
    ),
    Family(
        "dacvae",
        # コーデックは別リポ・別重み（`IRODORI_CODEC_NAME` + dtype 接尾）。
        lambda name: name.startswith("dacvae-"),
        lambda _: dacvae_export.PROVENANCE,
        roles=IRODORI_CODEC_ROLES,
    ),
    Family(
        QAT_FAMILY,
        lambda name: name.startswith(f"{QAT_FAMILY}-"),
        lambda _: gemma4_qat_export.PROVENANCE,
        flat_role=lambda _: GEMMA4_ROLE,
    ),
    Family(
        "gemma4",
        lambda name: name.startswith("gemma4-"),
        lambda _: gemma4_export.PROVENANCE,
        # MTP drafter だけが別の席（系列接尾 = `gemma4.export_drafter` の既定出力先）。
        flat_role=lambda name: (
            GEMMA4_DRAFTER_ROLE if name.endswith(f"-{GEMMA4_DRAFTER_SUFFIX}") else GEMMA4_ROLE
        ),
    ),
    Family(
        "minicpm5",
        lambda name: name.startswith("minicpm5-"),
        lambda _: minicpm5_export.PROVENANCE,
        flat_role=lambda _: minicpm5_export.GRAPH_NAME,
    ),
    Family(
        "embeddinggemma",
        lambda name: name.startswith("embeddinggemma-"),
        lambda _: embeddinggemma_export.PROVENANCE,
        flat_role=lambda _: embeddinggemma_export.GRAPH_NAME,
    ),
)


@dataclass(frozen=True)
class SeriesPlan:
    """1 系列ぶんの移行計画（この段は 1 バイトも書かない）。"""

    name: str
    family: str
    provenance: Provenance
    components: tuple[Component, ...]


def classify(name: str) -> Family:
    """系列名 → family（表に無ければ fail loudly — 部品名が決まらない）。"""
    for family in FAMILIES:
        if family.matches(name):
            return family
    raise SeriesMigrationError(
        f"系列 '{name}' がどの family の綴りにも当たらない"
        f"（既知: {', '.join(entry.name for entry in FAMILIES)}）— 部品名が決まらない"
    )


def is_unreferenced(name: str) -> bool:
    """変換しない系列か（{@link UNREFERENCED_SUFFIXES}）。"""
    return name.endswith(UNREFERENCED_SUFFIXES)


def shard_paths(directory: Path) -> tuple[Path, ...]:
    """そのディレクトリに在る旧 shard の現物（無ければ空）。"""
    return sequence_siblings(directory / f"{SHARD_STEM}{SHARD_SUFFIX}")


def component_directories(series: Path) -> list[Path]:
    """旧 shard 列を持つディレクトリ（系列直下とその 1 段下）。"""
    candidates = [series, *sorted(entry for entry in series.iterdir() if entry.is_dir())]
    return [directory for directory in candidates if shard_paths(directory)]


def series_directories(series: Path) -> list[Path]:
    """系列の**全ディレクトリ**（系列直下から深さ無制限）。

    取りこぼしの門（{@link _assert_nothing_unaccounted}）が見る範囲。部品ディレクトリだけを
    見ると、sidecar 専用のサブディレクトリ（`<系列>/pipeline/` / `<系列>/host/`）と、旧 shard
    列を持たない系列まるごと（`anima-pipeline*`）が門の外に落ちる。
    """
    return [series, *sorted(entry for entry in series.rglob("*") if entry.is_dir())]


def plan_series(series: Path) -> SeriesPlan:
    """1 系列の移行計画を組む（部品名の割り当てと sidecar の取りこぼしをここで全部落とす）。"""
    name = series.name
    family = classify(name)
    components = [
        Component(directory, _graph_name(family, name, directory, series))
        for directory in component_directories(series)
    ]
    # MUST: 門は**系列の全ディレクトリ**に掛ける（部品が 0 本の系列でも通す）— 畳み先の無い
    # `.safetensors` を置き去りにしたことに気づく機会は、移行が 1 度きりなので後から来ない。
    for directory in series_directories(series):
        _assert_nothing_unaccounted(directory, series)
    return SeriesPlan(name, family.name, family.provenance(name), tuple(components))


def _graph_name(family: Family, name: str, directory: Path, series: Path) -> str:
    """部品ディレクトリ → 部品名（= 容器のグラフ名）。"""
    if directory == series:
        if family.flat_role is None:
            raise SeriesMigrationError(
                f"{name}: 系列直下に旧 shard 列が在るが、family '{family.name}' は"
                "サブディレクトリに部品を置く（部品名が決まらない）"
            )
        return family.flat_role(name)
    role = family.roles.get(directory.name)
    if role is None:
        raise SeriesMigrationError(
            f"{name}: 部品ディレクトリ '{directory.name}' が family '{family.name}' の表に無い"
            f"（既知: {', '.join(sorted(family.roles))}）"
        )
    return role


def _assert_nothing_unaccounted(directory: Path, series: Path) -> None:
    """そのディレクトリの `.safetensors` が「旧 shard / golden / 畳む sidecar」で尽きること。

    MUST: 知らない `.safetensors` で止まる — 移行は 1 度きりなので、新種の sidecar を黙って
    置き去りにすると「資産が 1 つ足りない容器」が配布形まで通る（実行時に初めて分かる）。
    """
    accounted = {path.name for path in shard_paths(directory)}
    accounted.add(ROPE_BASE_FILE)
    accounted.update(ple_shard_names(directory))
    where = directory.relative_to(series.parent)
    for path in sorted(directory.glob(f"*{SHARD_SUFFIX}")):
        if path.name in accounted or path.name.startswith(GOLDEN_PREFIXES):
            continue
        raise SeriesMigrationError(
            f"{where}: '{path.name}' の畳み先が無い"
            f"（旧 shard でも golden〈{' / '.join(GOLDEN_PREFIXES)}〉でも sidecar でもない）"
        )


def ple_shard_names(directory: Path) -> tuple[str, ...]:
    """PLE sidecar の shard 名を**索引の宣言から**引く（索引が無ければ空）。

    MUST: 勘定は宣言から引く（現物の glob ではない）— 畳むのは {@link ple_sidecar_assets} が
    索引から引く列なので、glob で勘定すると「索引に載っていない現物」（版を変えた再生成の
    残骸など）が**畳まれないのに門も通る**。索引と現物の食い違いは畳む側が落とす。
    """
    index_path = directory / PLE_INDEX_FILE
    if not index_path.is_file():
        return ()
    index = read_ple_index(index_path, str(index_path))
    return tuple(str(entry["file"]) for entry in index["shards"])


def component_assets(
    component: Component, *, block_bytes: int = BLOCK_MAX_BYTES
) -> dict[str, AssetInput]:
    """部品に同居する sidecar を容器の資産へ（実在は PLE と `rope_base` の 2 種）。

    並びがそのまま物理配置の順になる（PLE は token 順 — container-v1 §4.2 の走査型取得元）
    ので、区間読みを要する PLE を先に置く。

    `block_bytes` は資産の block 分割の刻みで、**容器側の刻みと同じ値** MUST（呼び手は
    {@link migrate_series} の 1 本から両方へ回す）— 別々に決まる形だと、寸法を差し込んだ
    実行で資産だけが既定の刻みのまま据わる。
    """
    assets: dict[str, AssetInput] = {}
    index = component.directory / PLE_INDEX_FILE
    if index.is_file():
        assets.update(ple_sidecar_assets(index, block_bytes=block_bytes))
    rope_base = component.directory / ROPE_BASE_FILE
    if rope_base.is_file():
        payload = rope_base.read_bytes()
        assets[ROPE_BASE_ASSET] = AssetInput(
            EXTRA_ASSETS[ROPE_BASE_ASSET][1], len(payload), payload
        )
    return assets


def migrate_series(plan: SeriesPlan, *, block_bytes: int = BLOCK_MAX_BYTES) -> None:
    """計画どおり 1 部品ずつ移す（自己検査は `migrate_component` のもの）。"""
    for component in plan.components:
        assets = component_assets(component, block_bytes=block_bytes)
        result = migrate_component(
            component.representative,
            component.directory,
            provenance=plan.provenance,
            graph_name=component.graph_name,
            assets=assets,
            _block_bytes=block_bytes,
        )
        print(
            f"done {component.directory} graph={component.graph_name}"
            f" parts={len(result.parts)} initializers={result.initializers}"
            f" payloads={result.payloads} assets={result.assets}"
            f" bytes={total_bytes(result.parts)}",
            flush=True,
        )
        if plan.family == QAT_FAMILY:
            rewrite_qat_reference(component.directory, assets)


def rewrite_qat_reference(directory: Path, assets: Mapping[str, AssetInput]) -> None:
    """QAT 系列の `reference.json` を schema 3（`pleBlocks`）へ書き換える。

    「触らないもの」の**唯一の例外**。旧記録が名乗る `pleShards` は sidecar の shard 本数で、
    畳んだ後はどこにも存在しない数になる（block の切り方は shard 境界を無視する）。移行は
    記録が指す実体を作り直しているので、記録も追随させる MUST — 追随させないと、配布側の突合
    （`gemma4_qat.distribution.qat_plan`）が「欄が無い」で落ちるだけの系列が残る。

    他の欄（`fixedWeights` / `storageCounts` / `checkpoint` …）は**据え置き**: 移行は値を
    1 ビットも変えないので、現物と突き合わせる数はそのまま正しい。
    """
    path = directory / QAT_REFERENCE_FILE
    if not path.is_file():
        raise SeriesMigrationError(f"{directory}: QAT 系列に {QAT_REFERENCE_FILE} が無い")
    record = json.loads(path.read_text(encoding="utf-8"))
    if QAT_SHARDS_FIELD not in record or QAT_BLOCKS_FIELD in record:
        raise SeriesMigrationError(
            f"{path}: 畳む前の記録ではない（`{QAT_SHARDS_FIELD}` を持ち"
            f" `{QAT_BLOCKS_FIELD}` を持たない形だけを上げる）— schema"
            f" {record.get('schema')} / 欄 {sorted(record)}"
        )
    if PLE_INDEX_ASSET not in assets:
        raise SeriesMigrationError(
            f"{directory}: PLE を畳んでいないのに {QAT_REFERENCE_FILE} が在る"
        )
    index = json.loads(bytes(assets[PLE_INDEX_ASSET].payload))
    blocks = len(index["values"]["blocks"])
    # 欄の並びは据え置く（`pleShards` の席に `pleBlocks` が入る）— 再 export が書く記録と
    # 同じ並びになるので、2 つの記録を逐語で突き合わせられる。
    rewritten = {
        (QAT_BLOCKS_FIELD if key == QAT_SHARDS_FIELD else key): (
            blocks if key == QAT_SHARDS_FIELD else value
        )
        for key, value in record.items()
    }
    rewritten["schema"] = REFERENCE_SCHEMA
    path.write_text(json.dumps(rewritten, indent=2) + "\n", encoding="utf-8")
    print(f"rewrote {path} schema={REFERENCE_SCHEMA} {QAT_BLOCKS_FIELD}={blocks}", flush=True)


def total_bytes(paths: Sequence[Path]) -> int:
    return sum(path.stat().st_size for path in paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="系列出力の旧 shard 列を krm へ移す（旧入力は読むだけ・krm は隣に並べる）"
    )
    parser.add_argument("--series-dir", type=Path, default=SERIES_ROOT, help="系列の親ディレクトリ")
    parser.add_argument(
        "--only", action="append", default=[], metavar="SERIES", help="この系列だけを移す（複数可）"
    )
    parser.add_argument("--dry-run", action="store_true", help="計画だけ出して 1 バイトも書かない")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """参照されている系列を全部移す（落ちたところで止まる）。"""
    args = build_parser().parse_args(argv)
    root: Path = args.series_dir
    if not root.is_dir():
        raise SeriesMigrationError(f"系列の親ディレクトリが無い: {root}")
    names = sorted(entry.name for entry in root.iterdir() if entry.is_dir())
    wanted = set(args.only)
    if wanted - set(names):
        raise SeriesMigrationError(
            f"--only が指す系列が {root} に無い: {', '.join(sorted(wanted - set(names)))}"
        )
    plans: list[SeriesPlan] = []
    for name in names:
        if wanted and name not in wanted:
            continue
        if is_unreferenced(name):
            print(f"skip {name}（実験 / 棄却の記録）")
            continue
        plan = plan_series(root / name)
        if not plan.components:
            print(f"skip {name}（旧 shard 列が無い）")
            continue
        plans.append(plan)
    for plan in plans:
        print(f"plan {plan.name} family={plan.family} provenance={plan.provenance.to_document()}")
        for component in plan.components:
            source = shard_paths(component.directory)
            print(
                f"  {component.directory.relative_to(root)} graph={component.graph_name}"
                f" shards={len(source)} bytes={total_bytes(source)}"
            )
    if args.dry_run:
        return
    for plan in plans:
        migrate_series(plan)


if __name__ == "__main__":
    try:
        main()
    except (SeriesMigrationError, MigrateError) as error:
        print(f"migrate_series: {error}", file=sys.stderr)
        raise SystemExit(1) from error
