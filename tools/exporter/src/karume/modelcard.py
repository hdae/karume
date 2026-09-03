"""配布形モデルカード（`README.md`）の**汎用描画部品** — manifest から機械導出する純関数。

HF は `README.md` の YAML frontmatter をモデルの機械可読メタデータとして読む（ADR 0037 §3 の
「そのまま HF リポとしてアップロードできる形」の最後の 1 枚）。ライセンスと由来は manifest に
書かない決定なので（ADR 0038 §6）、その責務はカード側が持つ。

ここにあるのは「frontmatter を組む / モデル一覧を並べる / quant 表を並べる / 節を組み立てる」
層だけで、**pipeline 別のテンプレートは 1 つも持たない** — 何を説明し
何を使い方に綴るかは family 固有の事実なので、`tools/export-recipes/<family>/card.py` が持つ
（ADR 0065 決定 1・2）。`karume.dist` と同じ層分け。

NOTE: 描画部品（{@link frontmatter} / {@link models} / {@link quants} /
{@link model_sections} / {@link render} / {@link require_pipeline} / {@link default_model} /
{@link knob} / {@link from_pretrained}）は **recipe 向けの公開面**（ADR 0065 段 6）。wheel の
外にある `card.py` が名指しで呼ぶ以上、private 名のままにしておくと「private を跨いで呼ぶ」
形が既定になる。`_` 始まりで残しているのはこのモジュールの中だけで閉じる部品
（{@link _base_model} など）。

**帰属はテンプレートと別の軸**（{@link sbv2.card.Sbv2CardProfile}）。同じテンプレートでも、
どのファミリーの重みを配るかで出所・ライセンス・引用が丸ごと変わる pipeline があるので、
そこだけをプロファイルに分けて**呼び出し側に明示させる**（選ばせる規則は
{@link karume.dist.resolve_card_renderer}）。

manifest（現行は `karume/4` — ADR 0041 で複数モデル化し、ADR 0070 決定 1 で weights の
dtype エントリが shard 列になり、ADR 0075 決定 1 で quant へ表示欄が付いた形）は
**1 リポに複数モデル**を持てるので、カードも
「リポ全体の説明 → モデル一覧 → 使い方 → モデルごとの節」の形にする。モデルごとの節が
`## Model: <name>` で、その中に quant 表・（SBV2 は）スタイル表と話者表が並ぶ。
単一モデルのリポでも同じ形で描く（配布形のレイアウトが一様なのと同じ理由 — 2 個目が増えた
瞬間に構成が変わるカードは、読み手の目印も壊す）。

shard 上限を 256 MiB へ下げてファイル本数が 3〜4 倍になったので、**shard 1 本 1 行のファイル表は
廃止した**（2026-09-03 裁定 — 5 モデルのリポでカードの半分が表になり、読み手が知りたい
「このプリセットで何 GiB 落ちるか」がその中に埋もれた）。per-file の `size` / `sha256` の正本は
`karume.json` で、カードは quant ごとの合計（{@link quants} の Download 欄）だけを持つ。

MUST: **数値・ダウンロード量・quant 表・dtype ラベル・スタイル表・話者表は 1 つ残らず manifest
から導出する**。手書きのサイズや dtype 名は資産と独立に動けてしまい、「表と現物が食い違う」
失敗様式が manifest を導出物にした意味ごと消える（ADR 0038 Context）。テンプレート側が持って
よい定数は manifest に**存在しない事実**（base model・ライセンス・焼き込んだ LoRA の出所）
だけで、この MUST は recipe の `card.py` にもそのまま掛かる。リポ ID も manifest には無いので、
組み立て先から引いた値を引数で受ける。

MUST: 描画は決定的 — 同一 manifest なら**バイト単位で同一**。時刻・環境・集合の反復順に
依存するものを混ぜない（差分が出れば「資産が変わった」と読めることが再組み立ての前提）。
JSON 由来の dict は挿入順が保たれるので、`models` / `quants` / `styles` / `speakers` は
manifest に並んだ順のまま出す（並べ替えを挟むと「manifest の並び」という事実がカードから消える）。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: モデルごとの節を描く関数（モデルエントリ → 行の並び）。
ModelSection = Callable[[Mapping[str, Any]], list[str]]

#: HF の `library_name`（全 pipeline 共通 — 読み手はこのリポジトリの runtime 1 つだけ）。
LIBRARY_NAME = "karume"

#: 配布先の HF アカウント。リポ名は組み立て先のディレクトリ名から決まる（`karume.dist` が
#: `<HF_OWNER>/<ディレクトリ名>` を渡す）ので、ここが持つのは所有者だけ。
HF_OWNER = "hdae"

#: 注記に出す commit sha の桁数（完全な値は karume.json が持つ — 注記は指し先の目安）。
_SHA_DIGITS = 16

_UNITS = (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10))


@dataclass(frozen=True)
class CardMetadata:
    """frontmatter に載る **manifest に無い事実**（ADR 0038 §6 で manifest から外した領分）。

    実地確認: https://huggingface.co/docs/hub/model-cards 。`base_model_relation` の語彙は
    adapter / merge / quantized / finetune の 4 値で、f16 / i8 の配布形は「base の重みを別の
    格納形へ落とし直したもの」なので `quantized` を採る。merge は **Hub 上の base_model を
    2 つ以上並べる**形に紐づいた値で、Hub にない出所（Anima の LoRA）を正しく表現できない。
    **格納形を変えない配布形は 4 値のどれでもない**ので席ごと空ける（同じ実地確認: 未指定なら
    Hub が推論する）— 4 値の中から一番近いものを当てると、カードが事実でない主張を持つ。

    `base_model` が並びなのは、**この配布形が再配布している上流を全部並べる**ため（重みの出所と
    再配布する text encoder は別リポ）。並べても関係は `quantized` のまま — 重みを融合した
    わけではなく、それぞれを格納形へ落とし直して 1 リポに同居させているだけ。
    """

    pipeline_tag: str
    base_model: tuple[str, ...]
    license: str
    tags: tuple[str, ...]
    #: 4 値のどれかが**事実として当たるとき**だけ書く（当たらなければ Hub の推論に任せる）。
    base_model_relation: str | None = None
    #: `license: other` のときだけ HF が読む 2 席。SPDX 識別子を当てられた配布形は**持たない**
    #: （空欄で並べると「名前の無い独自ライセンス」に読める）。
    license_name: str | None = None
    license_link: str | None = None


def _download_size(size: int) -> str:
    """ダウンロード量を有効 3 桁で綴る（`3.24 GiB` / `248 MiB` — 単位を跨いでも精度が揃う）。

    生バイトは併記しない。ここで読み手が決めたいのは「この回線とこのディスクで現実的か」の
    1 点で、per-file の正確な値は `karume.json` が持つ（その所在は表の注記が指す）。
    """
    for unit, scale in _UNITS:
        if size >= scale:
            value = size / scale
            return f"{value:.{max(0, 3 - len(str(int(value))))}f} {unit}"
    return f"{size:,} B"


def _is_shared(ref: Mapping[str, Any]) -> bool:
    """2 本目のモデルでは落とし直さない参照か（同リポの `shared/` か、別リポからの越境）。"""
    return ref["path"].startswith("shared/") or "repo" in ref


def _collect(refs: dict[str, Mapping[str, Any]], weight_files: Mapping[str, Any]) -> None:
    """1 つの dtype エントリのファイル（shard 列 + 付帯）を **path で一意化**して足す。

    一意化は 2 つのコンポーネントが同じファイルを指す形（1 本化済みの rope_base と同型）で
    バイトを二重に数えないため。
    """
    for ref in weight_files["shards"]:
        refs.setdefault(ref["path"], ref)
    for ref in weight_files.get("extras", {}).values():
        refs.setdefault(ref["path"], ref)


def _quant_refs(model: Mapping[str, Any], quant: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """その quant を選んだ読み手が実際に落とすファイル参照。

    重みは quant が選んだ dtype のファイルだけ。assets は quant 選択に依存しない無条件
    ファイル（ADR 0041 §3）なので全部入る。
    """
    refs: dict[str, Mapping[str, Any]] = {}
    for name, label in quant["weights"].items():
        _collect(refs, model["weights"][name][label])
    for ref in model["assets"].values():
        refs.setdefault(ref["path"], ref)
    return list(refs.values())


def _model_refs(model: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """モデルが載せる全ファイル参照（dtype を跨いだ和 + assets）。

    注記は quant 表 1 つに 1 度出るので、条件も**モデル単位**で見る（選んだ quant 次第で
    注記が現れたり消えたりすると、同じモデルの説明が席ごとに変わる）。
    """
    refs: dict[str, Mapping[str, Any]] = {}
    for entry in model["weights"].values():
        for weight_files in entry.values():
            _collect(refs, weight_files)
    for ref in model["assets"].values():
        refs.setdefault(ref["path"], ref)
    return list(refs.values())


def _borrowed(refs: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    """越境参照の指し先（`(リポ, revision)` の重複なし・現れた順 — ADR 0038 §7）。"""
    return list(dict.fromkeys((ref["repo"], ref["revision"]) for ref in refs if "repo" in ref))


def require_pipeline(manifest: Mapping[str, Any], supported: str) -> None:
    """全モデルが本文の前提にしている pipeline 契約であることを確かめる（違えば描かない）。

    テンプレートは pipeline 固有なので、別契約の manifest を食わせると「表は合っているのに
    説明だけ別のモデルの話」というカードが黙って出る。ファミリーリポは別アーキを混ぜられる
    （ADR 0041 §2）ので、1 つでも違えば描けない。
    """
    for name, model in manifest["models"].items():
        if model["pipeline"] != supported:
            raise ValueError(
                f"モデルカードの本文は {supported} 固有 — "
                f"モデル '{name}' の pipeline '{model['pipeline']}' のカードは書けない"
            )


def _base_model(base_models: Sequence[str]) -> list[str]:
    """`base_model`（1 本なら scalar・複数なら YAML の並び — HF はどちらも読む）。"""
    if len(base_models) == 1:
        return [f"base_model: {base_models[0]}"]
    return ["base_model:", *(f"  - {model}" for model in base_models)]


def frontmatter(metadata: CardMetadata) -> list[str]:
    return [
        "---",
        f"library_name: {LIBRARY_NAME}",
        f"pipeline_tag: {metadata.pipeline_tag}",
        *_base_model(metadata.base_model),
        *(
            []
            if metadata.base_model_relation is None
            else [f"base_model_relation: {metadata.base_model_relation}"]
        ),
        f"license: {metadata.license}",
        *([f"license_name: {metadata.license_name}"] if metadata.license_name is not None else []),
        *([f"license_link: {metadata.license_link}"] if metadata.license_link is not None else []),
        "tags:",
        *(f"  - {tag}" for tag in metadata.tags),
        "---",
    ]


def models(manifest: Mapping[str, Any]) -> list[str]:
    """リポが載せているモデルの一覧（v2 で初めて機械可読になった軸 — ADR 0041 §2）。"""
    default = manifest["defaultModel"]
    lines = [
        "## Models",
        "",
        "| Model | Pipeline | Quants | Default quant |",
        "| ----- | -------- | ------ | ------------- |",
    ]
    for name, model in manifest["models"].items():
        mark = " (default)" if name == default else ""
        quants = " / ".join(f"`{quant}`" for quant in model["quants"])
        lines.append(
            f"| `{name}`{mark} | `{model['pipeline']}` | {quants} | `{model['defaultQuant']}` |"
        )
    lines += [
        "",
        f"`model` selects one of these; omitted, it is `{default}`."
        " `quant` defaults to that model's own default quant.",
    ]
    return lines


def _session(quant: Mapping[str, Any]) -> str:
    session = quant["session"]
    features = quant.get("gpuFeatures", {})
    parts = [f"`{key}` = `{value}`" for key, value in session.items()]
    parts += [f"requires `{key}`" for key, value in features.items() if value]
    return " / ".join(parts) if parts else "—"


def _presentation(quant: Mapping[str, Any]) -> str:
    """quant の表示欄（ADR 0075 決定 1 の `label` / `description`）を 1 セルに綴る。

    どちらも optional なので 4 通りある。書いていない席は id をそのまま読ませる（`—`）—
    ここで id を再掲すると、表の 1 列目と同じ文字列が 2 度並ぶだけになる。
    """
    label = quant.get("label")
    description = quant.get("description")
    if label and description:
        return f"**{label}** — {description}"
    if label:
        return f"**{label}**"
    return description or "—"


def _download(model: Mapping[str, Any], quant: Mapping[str, Any]) -> str:
    """quant 表の Download 欄 — この席を選んだときに落ちる合計と、合計と GPU 常駐量の差。

    注記は 2 つ。`shared` は「2 本目のモデルではこの分を落とさない」量（{@link _is_shared}）で、
    assets は**ホスト側で読むだけで GPU に載らない**分（gemma4 の PLE sidecar のように
    Download の大半が assets という配布形がある）。後者は無視できない比率のときだけ添える —
    どのカードにも書くと、数 MiB の tokenizer に読み手の目を向けさせるだけになる。
    """
    refs = _quant_refs(model, quant)
    asset_paths = {ref["path"] for ref in model["assets"].values()}
    total = sum(ref["size"] for ref in refs)
    shared = sum(ref["size"] for ref in refs if _is_shared(ref))
    assets = sum(ref["size"] for ref in refs if ref["path"] in asset_paths)
    notes = []
    if shared:
        notes.append(f"{_download_size(shared)} shared")
    if assets and assets * 100 >= total:
        notes.append(f"{_download_size(assets)} of assets, read on the host")
    return _download_size(total) + (f" ({'; '.join(notes)})" if notes else "")


def quants(
    model: Mapping[str, Any], *, abbreviations: Mapping[str, str] | None = None
) -> list[str]:
    """quant 表（ADR 0074 の席名 + ADR 0075 の表示欄 + 席ごとのダウンロード量）。

    `abbreviations` は席名の部品上書きトークン（`i8+bert4` の `bert`）→ その weights 名。
    **略称は recipe が定めるので、対応をカードに必ず出す**（ADR 0074 決定 4）— 表の 1 列目と
    Weights 列の綴りが繋がるのはこの 1 行だけで、無いと `bert4` がどの部品の話か読めない。
    略称を使っていない family は渡さない（トークンが weights 名そのものなら対応表は要らない）。

    表の後ろの注記は、廃止したファイル表から**掛かるときだけ**引き継いだもの（sha256 の正本の
    所在・dtype 語彙・`i4` 方言・`shared/`・越境参照）。掛からない配布形に載せると事実でない
    主張になるので、条件は manifest から見る。
    """
    default = model["defaultQuant"]
    lines = [
        "### Quants",
        "",
        "| Quant | What it is | Download | Weights | Compute |",
        "| ----- | ---------- | -------- | ---- | ---- |",
    ]
    for name, quant in model["quants"].items():
        weights = " / ".join(
            f"`{weight}` = `{label}`" for weight, label in quant["weights"].items()
        )
        mark = " (default)" if name == default else ""
        lines.append(
            f"| `{name}`{mark} | {_presentation(quant)} | {_download(model, quant)} |"
            f" {weights} | {_session(quant)} |"
        )
    lines += [
        "",
        f"If no quant is given, it runs as `{default}` (this model's recommended default).",
    ]
    if abbreviations:
        lines.append(
            "In a quant name, "
            + " / ".join(
                f"`{token}` is the `{name}` component" for token, name in abbreviations.items()
            )
            + "."
        )
    refs = _model_refs(model)
    lines += [
        "Per-file `size` and `sha256` live in `karume.json` — verify against that at the fetch"
        " layer.",
        "Dtype labels use the runtime's **storage dtype vocabulary** (`f16` / `i8` / `i4`), not"
        " the `fp16` spelling common elsewhere in the ecosystem.",
    ]
    # `I4` は safetensors の方言（ADR 0069・docs/limitations.md「格納 dtype `I4` は
    # safetensors の方言」）— 公式パーサで開けない事実は、その配布形を選んだ読み手にだけ
    # 関わるので i4 の席があるときだけ綴る（全カードに載せると事実でない主張になる）。
    if any("i4" in entry for entry in model["weights"].values()):
        lines.append(
            "A component stored as `i4` uses a packed int4 dtype (`I4`) that is **not part of the"
            " official safetensors specification** — the official `safetensors` library rejects a"
            " file that contains it (checked with 0.8.0). Karume's runtime and exporter read it;"
            " files without `i4` stay fully compatible."
        )
    if any(ref["path"].startswith("shared/") for ref in refs):
        lines.append(
            "A path under `shared/` is one this model shares byte for byte with another model in"
            " this repository (it is fetched and cached once)."
        )
    for repo, revision in _borrowed(refs):
        lines.append(
            f"Some components are fetched from [`{repo}`](https://huggingface.co/{repo}) at commit"
            f" `{revision[:_SHA_DIGITS]}…` — those bytes are identical to this model's own, so they"
            " are not stored here a second time."
        )
    return lines


#: `fromPretrained` の source 側に添える **revision を pin する手順**（全 family 共通の 1 組）。
#:
#: 既定は `main` 追従で、そちらの方が「とりあえず動く」ので既定のまま書く。ただし配布リポは
#: 上げ直す（ファイル名の変更・manifest の format 繰り上げ）ので、pin していない読み手は
#: ある日突然壊れる — 既定の行の**すぐ隣**にコメントアウトで置き、外すだけで pin できる形に
#: する（サンプルを 2 本に割ると、どちらが推奨か読めなくなる）。
REVISION_PIN_COMMENT: tuple[str, ...] = (
    "  // Pin a commit for reproducible builds — without it you track `main`, and a future",
    "  // repo update (renamed files, new manifest format) may break your app.",
    '  // Copy the full hash from this repo\'s "Files and versions" tab:',
    '  // revision: "<full commit sha>",',
)


def from_pretrained(
    pipeline_class: str,
    repo: str,
    options: Sequence[str],
    *,
    disposable: str = "using",
) -> list[str]:
    """使い方スニペットの `fromPretrained` 呼び出し 1 つ（**object ref 形** + pin の席）。

    source を文字列 1 本ではなく `{ repo, revision? }` で綴るのは、revision を書く席がここ
    以外に無いため（{@link REVISION_PIN_COMMENT}）。`options` は第 2 引数のオブジェクトへ
    そのまま入る行の並び（family 固有のノブ — 2 スペース字下げ済みで渡す）。

    `disposable` は宣言の綴り（`using` / `await using`）— pipeline が同期 / 非同期どちらの
    dispose を持つかは family 側の事実なので、ここでは選ばない。
    """
    return [
        f"{disposable} pipeline = await {pipeline_class}.fromPretrained({{",
        f'  repo: "{repo}",',
        *REVISION_PIN_COMMENT,
        "}, {",
        *options,
        "});",
    ]


def default_model(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    """既定モデルのエントリ（リポ全体を 1 つ紹介するときの代表 — 使い方の既定でもある）。"""
    return manifest["models"][manifest["defaultModel"]]


def knob(value: Any) -> str:
    """`defaults` の 1 値を綴る（文字列だけコード体 — 数値と名前を見分けられるように）。"""
    return f"`{value}`" if isinstance(value, str) else f"{value}"


def render(sections: Sequence[Sequence[str]]) -> str:
    """節の並びを 1 本の本文にする（末尾改行つき）。"""
    return "\n".join(line for section in sections for line in section) + "\n"


def model_sections(
    manifest: Mapping[str, Any],
    per_model: Sequence[ModelSection],
) -> list[Sequence[str]]:
    """モデルごとの節（`## Model: <name>`）を manifest の並びのまま組む。

    `per_model` は「モデルエントリを受けて節（行の並び）を返す」描き手の列 — pipeline 固有の
    節（SBV2 のスタイル表など）をここへ差し込むための唯一の軸。
    """
    sections: list[Sequence[str]] = []
    for name, model in manifest["models"].items():
        sections.append([""])
        sections.append([f"## Model: {name}", ""])
        for index, render_section in enumerate(per_model):
            if index:
                sections.append([""])
            sections.append(render_section(model))
    return sections
