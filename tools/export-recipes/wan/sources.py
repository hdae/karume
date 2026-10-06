"""Wan の上流取得元（HF リポと revision の pin）と部品の取得（ADR 0118 決定 5 / 7・0121 決定 1）。

`SOURCES` が上流取得元の表の正本。値の revision は HF API で解決した commit SHA（40 桁）で、
ADR 0092 決定 3（配布リポの revision を焼く流儀）を上流側にも当てたもの。recipe の台本は
ここを通して上流を読む — 台本ごとに `from_pretrained(repo)` を綴ると、既定ブランチの先頭が
動いた日に参照と export が黙って別の checkpoint を読む。

取得の要否（{@link COMPONENTS}）と期待パラメータ数（{@link EXPECTED_PARAMETERS}）はモデル名を
キーにした表（ADR 0121 決定 1 — 世代で部品の扱いが違う）。どのモデルも取得するのは DiT
（`transformer`）・VAE（`vae`）・scheduler の config と、ライセンスの front matter を持つ
`README.md`・部品の索引 `model_index.json` だけ。

- Wan2.1（`t2v-1.3b`）: umT5（`text_encoder`・f32 で約 22.7 GB）とトークナイザは DiT / VAE の取得
  では取らない — 段 6 のテキスト埋め込みの別プロセス（`wan.text_embeds`）が {@link text_snapshot}
  で取る（開発機のホスト RAM 31 GiB に umT5 と DiT を同居させない — ADR 0118 決定 4）。
- Wan2.2（`ti2v-5b`）: `text_encoder` と `tokenizer` はどの経路でも取らない（ADR 0121 決定 9 —
  umT5 は `karume-umt5-xxl` を越境参照し、トークナイザ資産は Wan2.1 の系列から写す）。

umT5 の容器の上流は Wan ではなく本家 `google/umt5-xxl`（{@link UMT5_SOURCES} — ADR 0122 決定 1）。
本家は pickle の `.bin` 分割形だけを持つので、取得口（{@link umt5_snapshot}）は索引から encoder に
要る shard を導き、表の shard（sha256 の pin つき）と突き合わせてからその分だけを取る（決定 2）。
Wan の snapshot（{@link text_snapshot}）はトークナイザ資産・check-mask・事前計算・golden の id 列の
ために残る（決定 3）。

    uv run --group wan python -m wan.sources --fetch     # 取得（HF の既定キャッシュへ）
    uv run --group wan python -m wan.sources             # 取得済みの検査（パラメータ数）
    uv run --group wan python -m wan.sources --model ti2v-5b --fetch   # TI2V-5B（約 22.8 GB）
    uv run --group wan python -m wan.sources --umt5 xxl --fetch        # 本家 umT5（約 29.8 GB）

MUST: `huggingface_hub` は関数の中で import する（`wan` グループは既定の sync に入らない —
`tests/test_optional_group_imports.py`）。
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UpstreamSource:
    """上流の HF リポ 1 本（repo + pin した revision + ライセンス識別子）。"""

    repo: str
    #: HF の commit SHA（40 桁の小文字 16 進）。ブランチ名やタグは置かない。
    revision: str
    #: 上流 `README.md` の front matter の `license:`（2026-10-02 に HF API で確認）。
    license: str


@dataclass(frozen=True)
class Component:
    """上流リポの部品 1 つ（`subfolder`）と、それを取得するか。"""

    subfolder: str
    fetch: bool
    #: 取得の要否の理由（台本と報告の読み手向け）。
    why: str
    #: DiT / VAE の取得（`fetch=False`）から外したうえで、テキスト埋め込みの別プロセスが
    #: {@link text_snapshot} で取る部品か（Wan2.1 の umT5 とトークナイザ）。
    text: bool = False


#: 上流取得元の表。キーは配布形のモデル名（`models/karume-wan2.1/` のモデル `t2v-1.3b`
#: — ADR 0118 決定 7・`models/karume-wan2.2/` のモデル `ti2v-5b` — ADR 0121 決定 1 / 10）。
SOURCES: dict[str, UpstreamSource] = {
    "t2v-1.3b": UpstreamSource(
        repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        revision="0fad780a534b6463e45facd96134c9f345acfa5b",
        license="apache-2.0",
    ),
    "ti2v-5b": UpstreamSource(
        repo="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        revision="b8fff7315c768468a5333511427288870b2e9635",
        license="apache-2.0",
    ),
}

#: 既定のモデル（Wan2.1 の台本の既定 — {@link WAN21_MODELS}）。
DEFAULT_MODEL = "t2v-1.3b"

#: Wan2.1 の台本（系列の置き場・ケースの表・テキスト段が Wan2.1 に固定）が `--model` で受ける
#: モデル。`sorted(SOURCES)` を選択肢にすると、`ti2v-5b` を渡したときに Wan2.1 の系列の置き場へ
#: 5B を書く口が開く（ADR 0121 の段 0 で表に `ti2v-5b` を足したときに閉じた）。
WAN21_MODELS: tuple[str, ...] = ("t2v-1.3b",)

#: Wan2.2 の配布（`karume-wan2.2`）が配るモデル — 配布 recipe の計画の門
#: （`wan.distribution.WAN22`）とカードの帰属の門（`wan.card.WAN22_CARD`）が共有する 1 つの集合。
#: {@link WAN21_MODELS} と同じく、表（{@link SOURCES}）に有ることでは通さない（表は両方の世代の
#: 行を持つ）。
WAN22_MODELS: tuple[str, ...] = ("ti2v-5b",)

#: Wan2.2 の配布が持つテキスト資産 2 本（埋め込み・トークナイザ）の出所のモデル = Wan2.1 の
#: checkpoint（ADR 0121 決定 9 — 同じ umT5 と同じトークナイザなので、2.1 の系列のファイルそのもの
#: を配る）。配布 recipe の出所の門（`wan.distribution.WAN22` の `text_model`）とカードの記述が
#: 共有する。
WAN22_TEXT_MODEL = "t2v-1.3b"

#: 上流リポの部品の列挙（モデル名 → 部品）。`fetch=False` の部品は DiT / VAE の取得では落とさない。
COMPONENTS: dict[str, tuple[Component, ...]] = {
    "t2v-1.3b": (
        Component(
            "transformer", fetch=True, why="DiT（段 2 / 3）— fp32 safetensors 2 分割・約 5.7 GB"
        ),
        Component("vae", fetch=True, why="動画 VAE（段 4 / 5）— fp32 約 0.5 GB"),
        Component("scheduler", fetch=True, why="UniPC の config（shift 3.0 — 決定 5）"),
        Component(
            "text_encoder",
            fetch=False,
            text=True,
            why="umT5-XXL f32 約 22.7 GB — 段 6 の別プロセスが取る",
        ),
        Component(
            "tokenizer", fetch=False, text=True, why="umT5 のトークナイザ — 段 6 の別プロセスが取る"
        ),
    ),
    "ti2v-5b": (
        Component(
            "transformer",
            fetch=True,
            why="DiT — fp32 safetensors 5 分割・約 19.99 GB（ADR 0121 決定 1）",
        ),
        Component(
            "vae", fetch=True, why="Wan2.2-VAE — fp32 約 2.82 GB（encoder を含む・ADR 0121 決定 1）"
        ),
        Component(
            "scheduler",
            fetch=True,
            why="UniPC の config（flow_shift は golden のメタに書く — ADR 0121 決定 1 / 8）",
        ),
        Component(
            "text_encoder",
            fetch=False,
            why="umT5 は karume-umt5-xxl を越境参照する — 取らない（ADR 0121 決定 9）",
        ),
        Component(
            "tokenizer",
            fetch=False,
            why="トークナイザ資産は Wan2.1 の系列から写す — 取らない（ADR 0121 決定 9）",
        ),
    ),
}

#: 部品の外で取得するリポ直下のファイル（ライセンスの front matter と部品の索引）。
ROOT_FILES: tuple[str, ...] = ("README.md", "model_index.json")


@dataclass(frozen=True)
class Umt5Source:
    """umT5 の上流 1 本（pin した HF リポと、encoder に要る pickle の shard の sha256 —
    ADR 0122 決定 1 / 2）。encoder はリポ直下にある（subfolder を持たない）。"""

    source: UpstreamSource
    #: encoder に要る shard（ファイル名 → HF API の LFS の sha256・64 桁の小文字 16 進）。
    #: 取得口と読み口は、索引から導いた shard の集合がこの表と同じことを見る（上流の索引が
    #: 動いた・表が古い、を黙って通さない）。読み口は unpickle の前に各 shard の sha256 を照合する
    #: — pickle は読むと任意コードが走りうるので、1 枚目の防御は内容の pin（2 枚目が
    #: `weights_only=True`）。
    shards: Mapping[str, str]


#: umT5 の上流の表（キーは umT5 の配布形のモデル名 — `wan.umt5_distribution.UMT5_DEFAULT_MODEL`）。
#: 本家の main の commit（tags なし）・ライセンスは API の `cardData` と README の front matter・
#: shard の sha256 は `/api/models/google/umt5-xxl/revision/<commit>?blobs=true` の LFS の値
#: （2026-10-04 に確認 — research `2026-10-04-umt5-upstream-provenance` §1）。
UMT5_SOURCES: dict[str, Umt5Source] = {
    "xxl": Umt5Source(
        source=UpstreamSource(
            repo="google/umt5-xxl",
            revision="66cb9e7e85526fe440a945569e42c72fb6cbc0ad",
            license="apache-2.0",
        ),
        shards={
            "pytorch_model-00001-of-00006.bin": (
                "382094214dfe74d782769f61ad95cfe32fdd297ae51f16f9208afa180b355e61"
            ),
            "pytorch_model-00002-of-00006.bin": (
                "b49efce006c907ea93eb38658577d5c8d4e85b4bab398a0c6ba25141927529d4"
            ),
            "pytorch_model-00003-of-00006.bin": (
                "da3d39fffe6464247531c20696715860ccaabdaaad3d5a2979dc2e2fbb7789fc"
            ),
        },
    ),
}

#: 本家の pickle 分割形の索引と config（transformers の `save_pretrained` の綴り）。
UMT5_BIN_INDEX = "pytorch_model.bin.index.json"
UMT5_CONFIG = "config.json"

#: encoder が使う重みのキーの接頭辞（語彙埋め込み `shared.weight` と `encoder.*` — decoder と
#: `lm_head` は取らない）。
UMT5_ENCODER_PREFIXES: tuple[str, ...] = ("shared.", "encoder.")

#: 取得済みの checkpoint が持つべきパラメータ数（モデル名 → 部品名 → safetensors のヘッダの
#: 要素数の和）。VAE は encoder を含む。
#: - `t2v-1.3b`: 調査（2026-10-02）の値で、pin した revision の実物でも一致を確かめた。
#: - `ti2v-5b`: ADR 0121 決定 1 の値（pin した revision の safetensors のヘッダから集計 — vae は
#:   decoder + `post_quant_conv` 555,051,580・encoder + `quant_conv` 149,637,088）。
EXPECTED_PARAMETERS: dict[str, dict[str, int]] = {
    "t2v-1.3b": {
        "transformer": 1_418_996_800,
        "vae": 126_892_531,
    },
    "ti2v-5b": {
        "transformer": 4_999_787_712,
        "vae": 704_688_668,
    },
}

#: safetensors のヘッダの dtype → 要素あたりバイト数（ヘッダの整合検査用）。
_DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I8": 1, "U8": 1}


class WanSourceError(RuntimeError):
    """上流の checkpoint が pin した形で手元に無い・ヘッダが読めない。"""


def _components(model: str) -> tuple[Component, ...]:
    """モデルの部品の表（表に無いモデルは fail loudly — 黙って別のモデルの表を使わない）。"""
    parts = COMPONENTS.get(model)
    if parts is None:
        raise WanSourceError(f"モデル {model!r} の部品の表が無い（既知: {sorted(COMPONENTS)}）")
    return parts


def text_components(model: str = DEFAULT_MODEL) -> tuple[str, ...]:
    """テキスト埋め込みの別プロセス（`wan.text_embeds`）だけが取る部品（無いモデルは空）。"""
    return tuple(part.subfolder for part in _components(model) if part.text)


def allow_patterns(model: str = DEFAULT_MODEL) -> list[str]:
    """`snapshot_download` へ渡す取得対象（取得する部品の配下 + リポ直下の 2 本）。"""
    return [*ROOT_FILES, *(f"{part.subfolder}/*" for part in _components(model) if part.fetch)]


def fetch(model: str = DEFAULT_MODEL) -> Path:
    """pin した revision の取得対象を HF の既定キャッシュへ落とし、snapshot の置き場を返す。"""
    from huggingface_hub import snapshot_download

    source = SOURCES[model]
    return Path(
        snapshot_download(
            source.repo, revision=source.revision, allow_patterns=allow_patterns(model)
        )
    )


def local_snapshot(model: str = DEFAULT_MODEL) -> Path:
    """取得済みの snapshot のディレクトリ（ネットワークに出ない）。

    無ければ fail loudly — 台本が黙って取得を始めると、5.7 GB の取得が参照作成の途中に紛れる。
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    source = SOURCES[model]
    try:
        snapshot = Path(
            snapshot_download(
                source.repo,
                revision=source.revision,
                allow_patterns=allow_patterns(model),
                local_files_only=True,
            )
        )
    except LocalEntryNotFoundError as error:
        raise WanSourceError(
            f"{source.repo}@{source.revision} が HF キャッシュに無い — 先に"
            f" `uv run --group wan python -m wan.sources --model {model} --fetch` で取得する"
        ) from error
    missing = [part.subfolder for part in _components(model) if part.fetch]
    missing = [name for name in missing if not (snapshot / name).is_dir()]
    if missing:
        raise WanSourceError(f"{snapshot} に部品 {missing} が無い — `--fetch` で取り直す")
    return snapshot


def text_snapshot(model: str = DEFAULT_MODEL, *, fetch: bool = False) -> Path:
    """pin した revision の umT5 とトークナイザ（{@link text_components}）の snapshot。

    `fetch=False` ならネットワークに出ず、無ければ fail loudly（{@link local_snapshot} と同じ理由 —
    約 23 GB の取得が埋め込みの生成に紛れない）。DiT / VAE の取得対象とは別の呼び口にして、
    参照パイプラインの取得に umT5 が混ざらない形を保つ。
    MUST: テキスト段の部品を上流から取らないモデル（`ti2v-5b` — ADR 0121 決定 9）は fail loudly。
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    source = SOURCES[model]
    names = text_components(model)
    if not names:
        raise WanSourceError(
            f"モデル {model!r} はテキスト段の部品（umT5・トークナイザ）を上流から取らない"
            "（ADR 0121 決定 9 — umT5 は越境参照・トークナイザ資産は Wan2.1 の系列から写す）"
        )
    patterns = [f"{name}/*" for name in names]
    try:
        snapshot = Path(
            snapshot_download(
                source.repo,
                revision=source.revision,
                allow_patterns=patterns,
                local_files_only=not fetch,
            )
        )
    except LocalEntryNotFoundError as error:
        raise WanSourceError(
            f"{source.repo}@{source.revision} の {list(names)} が HF キャッシュに無い —"
            " 先に `uv run --group wan --inexact python -m wan.text_embeds --fetch` で取得する"
        ) from error
    missing = [name for name in names if not (snapshot / name).is_dir()]
    if missing:
        raise WanSourceError(f"{snapshot} に部品 {missing} が無い — `--fetch` で取り直す")
    return snapshot


def umt5_encoder_shards(weight_map: Mapping[str, str]) -> frozenset[str]:
    """索引の `weight_map`（キー → shard のファイル名）から、encoder の重み
    （{@link UMT5_ENCODER_PREFIXES}）が載る shard を導く。"""
    return frozenset(
        shard for key, shard in weight_map.items() if key.startswith(UMT5_ENCODER_PREFIXES)
    )


def pinned_umt5_shards(index: Path, pinned: Mapping[str, str]) -> dict[str, str]:
    """索引から導いた shard の集合が表（`pinned`）と同じことを見て、索引の `weight_map` のうち
    表の shard に載るキーを返す（キー → shard のファイル名）。

    MUST: 集合が違えば fail loudly — 上流の索引が動いたか表が古い。どちらでも、表の sha256 が
    覆わない shard を読むか、要る shard を取りこぼす。
    """
    weight_map: dict[str, str] = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    derived = umt5_encoder_shards(weight_map)
    if derived != frozenset(pinned):
        raise WanSourceError(
            f"{index}: 索引から導いた encoder の shard {sorted(derived)} が表の shard"
            f" {sorted(pinned)} と違う — 上流の索引が動いたか表（wan.sources.UMT5_SOURCES）が古い"
        )
    return {key: shard for key, shard in weight_map.items() if shard in pinned}


def umt5_snapshot(model: str, *, fetch: bool = False) -> Path:
    """umT5 の上流の pin（{@link UMT5_SOURCES}）の snapshot（config・索引・encoder の shard）。

    先に config と索引だけを取り、索引から導いた shard（{@link pinned_umt5_shards} — 表と
    違えば落ちる）を取る。decoder だけの shard は取らない（本家は 6 本・約 51.9 GB のうち encoder に
    要るのは約 29.8 GB）。`fetch=False` ならネットワークに出ず、無ければ fail loudly
    （{@link local_snapshot} と同じ理由）。shard の sha256 の照合は読み口
    （`wan.umt5_export.Checkpoint`）が unpickle の前に掛ける — 取得の記録ではなく読む実物で閉じる。
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    row = UMT5_SOURCES.get(model)
    if row is None:
        raise WanSourceError(
            f"umT5 のモデル {model!r} は上流の表に無い（既知: {sorted(UMT5_SOURCES)}）"
        )
    source = row.source

    def download(patterns: list[str]) -> Path:
        try:
            return Path(
                snapshot_download(
                    source.repo,
                    revision=source.revision,
                    allow_patterns=patterns,
                    local_files_only=not fetch,
                )
            )
        except LocalEntryNotFoundError as error:
            raise WanSourceError(
                f"{source.repo}@{source.revision} の {patterns} が HF キャッシュに無い — 先に"
                f" `uv run --group wan python -m wan.sources --umt5 {model} --fetch` で取得する"
            ) from error

    snapshot = download([UMT5_CONFIG, UMT5_BIN_INDEX])
    index = snapshot / UMT5_BIN_INDEX
    if not index.is_file() or not (snapshot / UMT5_CONFIG).is_file():
        raise WanSourceError(f"{snapshot} に {UMT5_CONFIG} / {UMT5_BIN_INDEX} が無い — `--fetch`")
    shards = sorted(set(pinned_umt5_shards(index, row.shards).values()))
    snapshot = download(shards)
    missing = [name for name in shards if not (snapshot / name).is_file()]
    if missing:
        raise WanSourceError(f"{snapshot} に shard {missing} が無い — `--fetch` で取り直す")
    return snapshot


def read_safetensors_header(path: Path) -> dict[str, dict[str, object]]:
    """safetensors のヘッダ（先頭 8 バイトの長さ + JSON）だけを読む（本体は読まない）。"""
    with path.open("rb") as stream:
        (length,) = struct.unpack("<Q", stream.read(8))
        header = json.loads(stream.read(length))
    header.pop("__metadata__", None)
    return header


def count_parameters(component_dir: Path) -> int:
    """部品のディレクトリにある safetensors の全テンソルの要素数の和。

    ヘッダの `data_offsets` の幅と `dtype × shape` が食い違う行は fail loudly（ヘッダの読み違い
    で数が合ってしまう形を潰す）。
    """
    files = sorted(component_dir.glob("*.safetensors"))
    if not files:
        raise WanSourceError(f"{component_dir} に safetensors が無い")
    total = 0
    for file in files:
        for name, entry in read_safetensors_header(file).items():
            shape = entry["shape"]
            dtype = entry["dtype"]
            offsets = entry["data_offsets"]
            assert isinstance(shape, list) and isinstance(dtype, str) and isinstance(offsets, list)
            count = 1
            for size in shape:
                count *= int(size)
            if count * _DTYPE_BYTES[dtype] != int(offsets[1]) - int(offsets[0]):
                raise WanSourceError(f"{file} の {name}: dtype {dtype} × {shape} と幅が合わない")
            total += count
    return total


def check_snapshot(snapshot: Path, model: str = DEFAULT_MODEL) -> dict[str, int]:
    """取得済みの部品のパラメータ数を数え、{@link EXPECTED_PARAMETERS} のモデルの行と
    突き合わせる。"""
    expected_counts = EXPECTED_PARAMETERS.get(model)
    if expected_counts is None:
        raise WanSourceError(
            f"モデル {model!r} の期待パラメータ数が無い（既知: {sorted(EXPECTED_PARAMETERS)}）"
        )
    counts = {name: count_parameters(snapshot / name) for name in expected_counts}
    wrong = {
        name: (counts[name], expected)
        for name, expected in expected_counts.items()
        if counts[name] != expected
    }
    if wrong:
        raise WanSourceError(f"{model}: パラメータ数が調査の値と違う（実物, 期待）: {wrong}")
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(SOURCES))
    parser.add_argument("--fetch", action="store_true", help="pin した revision で取得する")
    parser.add_argument(
        "--umt5",
        choices=sorted(UMT5_SOURCES),
        help="Wan ではなく umT5 の上流（本家の encoder の shard）を取得 / 検査する",
    )
    args = parser.parse_args(argv)

    if args.umt5 is not None:
        snapshot = umt5_snapshot(args.umt5, fetch=args.fetch)
        print(f"snapshot: {snapshot}")
        for name in sorted(UMT5_SOURCES[args.umt5].shards):
            print(f"{name}: {(snapshot / name).stat().st_size:,} bytes")
        return 0
    snapshot = fetch(args.model) if args.fetch else local_snapshot(args.model)
    counts = check_snapshot(snapshot, args.model)
    print(f"snapshot: {snapshot}")
    for name, count in counts.items():
        print(f"{name}: {count:,} parameters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
