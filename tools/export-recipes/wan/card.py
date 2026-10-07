"""Wan の配布形のモデルカード（`README.md`）— manifest から機械導出する純関数（Wan2.1 / Wan2.2）。

汎用の描画部品（frontmatter・モデル一覧・quant 表・節の組み立て）は `karume.modelcard` が持つ。
ここが持つのは **Wan の世代ごとの事実**だけ: 帰属（出所・pin した revision・ライセンス）と、この
pipeline のカードに何を書くか（第 1 段の固定プロンプト・受理する入力・数値の門・実行資源の実測）。

世代で違う事実は世代の表 {@link WanCard}（{@link WAN21_CARD} / {@link WAN22_CARD}）に集め、カードの
骨組み（節の並び・帰属の行・Usage・プロンプト・受理する入力の枠・既定値・宣言 limit）は 1 本の
{@link render_wan_model_card} が描く。世代の表が持つのは、値として引数化できる事実（pipeline 契約・
モデルの集合・受理集合・公開 class 名）と、散文が丸ごと違う節（概要・改変の要約・検証の範囲・実行
資源の実測）の描き手。

MUST: **数値・ダウンロード量・quant 表・dtype ラベル・既定値は 1 つ残らず manifest から導出する**
（`karume.modelcard` の同 MUST がそのまま掛かる — text_encoder の取得量・越境参照の先・宣言された
device limit も manifest から引く）。ここが持ってよい定数は manifest に**存在しない事実**だけ —
上流の取得元と pin（`wan.sources.SOURCES` が正本）・固定プロンプトの表（`wan.prompts.FIXED_PROMPTS`
が正本 — 組み立ての門 `wan.distribution.assert_text_embeds` が資産のメタとの一致を見るので、ここに
描く本文と配る資産は食い違わない）・TS 側の受理集合（`packages/models/src/wan/descriptor.ts` の
`WAN21_GENERATION` の `acceptedSizes` / `minFrames` / `maxFrames`・テキストの経路の選択
`textEncoder` — ADR 0119 追記 B・プロンプトの受理規則 — ADR 0119 決定 1 / 2 / 4 と追記 10a。
Wan2.2 は `WAN22_TI2V_GENERATION`）・実行資源の実測（`_wan21_resources` / `_wan22_resources` —
ADR 0089 決定 3）。

MUST: torch を import しない（`import dist` が torch を読まない —
`tests/test_dist_driver.py` の `TestImportingTheDriver`）。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

from karume.modelcard import (
    CardMetadata,
    default_model,
    from_pretrained,
    frontmatter,
    knob,
    model_sections,
    models,
    quants,
    render,
    require_pipeline,
)
from wan.prompts import FIXED_PROMPTS
from wan.sources import SOURCES, UMT5_SOURCES, WAN21_MODELS, WAN22_MODELS, WAN22_TEXT_MODEL
from wan.umt5_distribution import UMT5_DEFAULT_MODEL, UMT5_ROLE

#: このテンプレートが説明できるパイプライン契約（ADR 0041 §2 — モデル単位）。
WAN_SUPPORTED_PIPELINE = "wan/1"

#: HF の pipeline tag（上流と同じ）。
WAN_PIPELINE_TAG = "text-to-video"

#: 上流の技術レポート。
WAN_PAPER = "arxiv.org/abs/2503.20314"

#: 原文の在処（配布リポ直下の `LICENSE.md` と同じテキスト — Apache 2.0 §4(a)）。
WAN_LICENSE_TEXT_LINK = "https://www.apache.org/licenses/LICENSE-2.0"

#: TS 側が受理する寸法とフレーム数（`packages/models/src/wan/descriptor.ts` の受理集合の写し —
#: manifest に無い事実）。
#: MUST: TS 側と同じ値（`packages/models/tests/fixtures/wan-card-limits.json` を挟んで両側の
#: テストが突き合わせる — 片側だけ変えると赤）。
WAN_ACCEPTED_SIZES: tuple[tuple[int, int], ...] = ((832, 480), (480, 832))
WAN_FRAMES = (5, 81)

#: TS 側の生成の既定（`WAN21_GENERATION.defaults` の写し — Usage のコメントが描く）。
WAN_DEFAULT_SIZE = (832, 480)
WAN_DEFAULT_FRAMES = 33

#: テキストの経路を選ぶ構築時のオプション（TS 側 `WanPipeline.fromPretrained` の第 2 引数 —
#: ADR 0119 追記 B。manifest に無い事実）。既定は GPU 経路（決定 7）。
WAN_TEXT_ENCODER_OPTION = "textEncoder"
WAN_TEXT_ENCODER_PATHS = ("gpu", "precomputed")

#: GPU 経路が受けるプロンプトの token 数（末尾の `</s>` を含む — ADR 0119 決定 4。TS 側の
#: 受理集合で、上限はトークナイザ資産の `maxLength`）。
WAN_PROMPT_TOKENS = (2, 512)

#: umT5 の部品名（manifest の weights のキー — `wan.distribution.WAN_TEXT_ENCODER_ROLE` と
#: 同じ正本）。
WAN_TEXT_ENCODER_COMPONENT = UMT5_ROLE

#: text encoder の上流（本家の encoder — ADR 0122 決定 1 / 3。この checkpoint の `text_encoder` とは
#: f32 でビット一致するが、出所は本家を名乗る — umT5 の配布形のカードと食い違わせない）。
WAN_TEXT_ENCODER_UPSTREAM = UMT5_SOURCES[UMT5_DEFAULT_MODEL].source.repo

#: 通しの実行資源（``_wan_resources`` の段ごとの表）を実測した quant 席。MUST: カードは数を
#: **この席の数として**名乗る — 席に無い配布形では描かない（実測していない席の数は名乗らない —
#: BiRefNet のカードと同じ）。
WAN_RESOURCE_QUANT = "f16"

#: 席ごとの transformer の実測（ADR 0120 追記「段 3 / 4 / 5 の結果」〈DiT 単体・通常モード・
#: B570〉と「段 6 の素材」〈実用席の 50 ステップ・33 フレームの通し・precomputed の経路〉・
#: 「50 ステップの実用席の sha 行 2 本」〈実用席の 81 フレームの通し: 壁 3,703 s = DiT 段 3,383 s
#: + VAE 段 319 s〉）。
#: 行は `(フレーム数, 1 forward, DiT 単体の確保, 50 ステップの通し)`。`f16` の行は比べる基準で、
#: 通しの欄は上の段ごとの表と同じ通し（ADR 0118 段 6 / 8）。参照席 `f16+dit8` の 1 forward は
#: r 門の計測（S = 14,040）で `f16` と同じ時間 — 確保と 81 フレームは計測していない。
#: MUST: 計測していない欄は「not measured」/「not run」と書き、推し量った数で埋めない。
#: 表に無い席は「未計測」と名乗る（席の並びは manifest のまま）。
WAN_QUANT_TRANSFORMER: Mapping[str, tuple[tuple[int, str, str, str], ...]] = {
    "f16": (
        (33, "17.1 s", "5.15 GiB", "~30 minutes"),
        (81, "68.5 s", "6.17 GiB", "~2 hours"),
    ),
    "f16+dit8": (
        (33, "same as `f16`", "not measured", "not run"),
        (81, "not measured", "not measured", "not run"),
    ),
    "f16+dit8-a8-attn8-s16": (
        (33, "8.5 s", "4.02 GiB", "952 s (~16 minutes)"),
        (81, "34.0 s", "5.46 GiB", "3,703 s (~62 minutes)"),
    ),
}

#: 実用席の step 1 の潜在の相対 RMS 誤差（参照席 `f16+dit8` に対して — ADR 0120 段 4 の自機
#: A/B 門の実測）。`(席, 参照席, 33 フレーム, 81 フレーム)`。カードが併記する視認の結果は ADR 0120
#: 裁定 2026-10-04 の 1（段 6 の 12 対 — `f16` と実用席・50 ステップ・33 フレーム）。
WAN_PRACTICAL_QUANT_ERROR = ("f16+dit8-a8-attn8-s16", "f16+dit8", "0.107", "0.210")

#: Wan2.2 TI2V 5B のカードが説明できるパイプライン契約（TS 側 `packages/models/src/wan/config.ts` の
#: `WAN_TI2V_PIPELINE_NAME` / `WAN_TI2V_PIPELINE_MAJOR`）。
WAN22_SUPPORTED_PIPELINE = "wan-ti2v/1"

#: TS 側が受理する寸法とフレーム数（`packages/models/src/wan/descriptor.ts` の
#: `WAN22_TI2V_GENERATION` の写し — ADR 0121 追記「受理寸法を公式の 2 寸法へ」）。
#: MUST: TS 側と同じ値（`packages/models/tests/fixtures/wan-ti2v-card-limits.json` を挟んで両側の
#: テストが突き合わせる — 片側だけ変えると赤）。
WAN22_ACCEPTED_SIZES: tuple[tuple[int, int], ...] = ((1280, 704), (704, 1280))
WAN22_FRAMES = (5, 121)

#: TS 側の生成の既定（`WAN22_TI2V_GENERATION.defaults` の写し — ADR 0121 追記「受理寸法を公式の
#: 2 寸法へ」で「仮置き — 視認で確定する」）。NOTE: fixture `wan-ti2v-card-limits.json` はまだ既定を
#: 持たないので、TS 側との突き合わせは無い — 既定を替えるときは descriptor.ts とここを両方直す。
WAN22_DEFAULT_SIZE = (1280, 704)
WAN22_DEFAULT_FRAMES = 33

#: Wan2.2 の席ごとの実測。行は `(フレーム数, 1 forward, DiT の段の VRAM の山, DiT の段, VAE の段,
#: 通し)`。50 ステップの通しは全て 1280×704・shift 5・guidance 5・seed 42（時間は DiT の段 /
#: VAE の段 / 壁）。33 / 49 フレームの行は B570・Deno 2.9.6・2026-10-05・precomputed の経路
#: （VRAM の山は fdinfo の DiT の段）。121 フレームの行だけは GPU も測り方も違う（下の項）。
#:
#: - 33 フレームの行は 2 席とも**製品の class**（`WanTi2vPipeline`）の opt-in の通し（ADR 0121 追記
#:   「段 6 の結果」の「50 ステップの通し」— 参照席 2,142.9 / 439.2 / 2,582.5 s・7.859 GiB、実用席
#:   959.9 / 438.5 / 1,398.8 s・7.843 GiB）。
#: - 49 フレームの行は、製品の class ではなく同じ製品部品を呼ぶ**生成スクリプト**の通し（追記
#:   「受理寸法を公式の 2 寸法へ」の試走の表）。同じ要求の実用席 33 フレームでは、生成スクリプトの
#:   DiT の段の山（7.72 GiB）が製品の class（7.843 GiB）より 0.12 GiB 小さかった（出力はバイト
#:   同一）ので、カードはその向きを注に書く。
#: - 121 フレームの行（実用席だけ）は ADR 0121 追記「RTX 3080 Ti のレーンとフル verify」の
#:   開発機の通し: RTX 3080 Ti（12,288 MiB）・Deno 2.9.6・2026-10-06・**GPU の umT5 の経路**・
#:   cat-dog-baking・生成スクリプト（`outputs/diag/wan22-121.ts` — 記述子の上限だけを 121 へ広げた
#:   家族で内部 API を呼ぶ・配布形を読む）。壁 3,787.9 s = umT5 9.1 / DiT 3,068.2 / VAE 709.8 s。
#:   **VRAM の山は nvidia-smi の GPU 全体**（DiT の段 11,361 MiB = 11.09 GiB — fdinfo ではない）で、
#:   時間は熱制限込み（93 ℃・throttle `sw_thermal`）なので B570 の行と比べられない — カードの注が
#:   その行の GPU・runtime・測り方・熱制限を名乗る。
#:   記録 `outputs/misc/wan22-121-2026-10-06/record.json`。
#:   参照席の 121 フレームの 50 ステップは回していないので行を持たない（推し量って埋めない）。
#: - 1 forward は追記「段 2 の結果」（2026-10-04）の所要（DiT 単体・通常モード・batch 1・
#:   S = 7,920 = 1280×704×33）。49 フレーム（S = 11,440）と 121 フレーム（S = 27,280）の
#:   1 forward は計測していない。
#:
#: MUST: 計測していない欄は「not measured」と書き、推し量った数で埋めない。表に無い席は「未計測」と
#: 名乗る（席の並びは manifest のまま — 2.1 の {@link WAN_QUANT_TRANSFORMER} と同じ扱い）。
WAN22_RESOURCES: Mapping[str, tuple[tuple[int, str, str, str, str, str], ...]] = {
    "f16+dit8": (
        (33, "21.8 s", "7.86 GiB", "2,143 s", "439 s", "43 min 3 s"),
        (49, "not measured", "8.45 GiB", "3,360 s", "647 s", "66 min 53 s"),
    ),
    "f16+dit8-a8-attn8-s16": (
        (33, "9.9 s", "7.84 GiB", "960 s", "439 s", "23 min 19 s"),
        (49, "not measured", "8.49 GiB", "1,504 s", "655 s", "36 min 5 s"),
        (121, "not measured", "11.09 GiB", "3,068 s", "710 s", "63 min 8 s"),
    ),
}

#: Wan2.2 の Usage が呼ぶ公開 class（`@karume/models`）。資源の注も名指しする（33 フレームの通しを
#: 回した class）。
WAN22_PIPELINE_CLASS = "WanTi2vPipeline"

#: Wan2.2 の実用席と参照席（実用席の品質の節が名指しする 2 席 — 段 7 の自機 A/B 門は未計測）。
WAN22_PRACTICAL_QUANT = ("f16+dit8-a8-attn8-s16", "f16+dit8")


@dataclass(frozen=True)
class WanCard:
    """カードの世代の表 — 世代ごとに違う事実（値）と、散文が丸ごと違う節の描き手。

    骨組み（節の並び・帰属の行・Usage・プロンプト・受理する入力の枠・既定値・宣言 limit）は
    {@link render_wan_model_card} が 1 本で描く。
    """

    #: 世代の名前（帰属の行・門の文言 — `Wan2.1` / `Wan2.2`）。
    generation: str
    #: カードの見出しに出す checkpoint の名前。
    title: str
    #: このカードが説明できるパイプライン契約（ADR 0041 §2）。
    supported_pipeline: str
    #: この世代の配布が配るモデル（帰属の門 — 取得元の表の全モデルは通さない）。
    models: tuple[str, ...]
    #: Usage が呼ぶ公開 class（`@karume/models`）。
    pipeline_class: str
    #: 上流の技術レポート（上流の README が引くもの — 確かめられないなら None で行を省く）。
    paper: str | None
    #: TS 側の受理集合の写し（寸法 `(width, height)` の並び・フレーム数の `(下限, 上限)`）。
    accepted_sizes: tuple[tuple[int, int], ...]
    frames: tuple[int, int]
    #: TS 側の生成の既定（寸法 `(width, height)` とフレーム数 — `descriptor.ts` の `defaults` の
    #: 写し）。Usage のコメントが描く。受理集合の並びの先頭からは導かない（既定は視認で替わりうる —
    #: Wan2.2 は ADR 0121 追記「受理寸法を公式の 2 寸法へ」の「仮置き」）。
    default_size: tuple[int, int]
    default_frames: int
    #: 受理するフレーム数の行に続けて書く「GPU で確かめた範囲」の行（先頭の要素は範囲と同じ行に
    #: 続く）。
    frames_checked: tuple[str, ...]
    #: 「Determinism and verification」節のうち、ビット同一の項の後に続く項の行。
    verification: tuple[str, ...]
    #: 「What is this」節（manifest から — text_encoder の取得元を描く）。
    overview: Callable[[Mapping[str, Any]], list[str]]
    #: 帰属節のうち、モデルごとの出所の行の後に続く項（作者・改変の要約・text encoder・資産）。
    attribution_notes: Callable[[Mapping[str, Any]], list[str]]
    #: 「Resources」節（実測の表と注 — ADR 0089 決定 3）。
    resources: Callable[[Mapping[str, Any]], list[str]]


def _upstream(name: str, card: WanCard) -> Any:
    """モデル名 → 上流の取得元（その世代のモデルでなければ描かない — このカードはその世代の配布の
    事実だけを書く。取得元の表は別の世代の行も持つので、表に有ることでは通さない）。"""
    if name not in card.models:
        raise ValueError(
            f"モデル '{name}' は {card.generation} のモデルでない"
            f"（既知: {list(card.models)}）— {card.generation} のカードに別の世代の出所を書かない"
        )
    return SOURCES[name]


def _wan_metadata(manifest: Mapping[str, Any], card: WanCard) -> CardMetadata:
    """frontmatter を manifest に並んだモデルから組む（`base_model` は manifest のモデルの上流 —
    HF の「派生元のモデル」の意味。Wan2.2 のテキスト資産 2 本の出所〈Wan2.1 の checkpoint〉は
    派生元のモデルではないので載せず、帰属の節が書く）。"""
    licenses = {_upstream(name, card).license for name in manifest["models"]}
    if len(licenses) != 1:
        raise ValueError(
            f"モデルごとにライセンスが割れている（{sorted(licenses)}）— 1 値で書けない"
        )
    return CardMetadata(
        pipeline_tag=WAN_PIPELINE_TAG,
        base_model=tuple(_upstream(name, card).repo for name in manifest["models"]),
        # f32 の上流を f16 / i8 へ落とし直した配布形（`CardMetadata` の doc — f16 / i8 の配布形は
        # `quantized`）。
        base_model_relation="quantized",
        license=next(iter(licenses)),
        tags=(WAN_PIPELINE_TAG, "video-generation", "wan", "webgpu"),
    )


def _text_encoder(manifest: Mapping[str, Any]) -> tuple[int, tuple[str, str] | None]:
    """既定モデルの既定 quant が取る text_encoder の容器の `(バイト数, 越境参照の先)`。

    越境参照の先は `(repo, revision)`（容器単位で全 part が同じ — ADR 0109 決定 3）で、自リポに
    置いた配布形なら `None`。カードは「どこから取るか」を manifest のこの事実から描く。
    """
    model = default_model(manifest)
    weights = model["weights"].get(WAN_TEXT_ENCODER_COMPONENT)
    if weights is None:
        raise ValueError(
            f"manifest の weights に '{WAN_TEXT_ENCODER_COMPONENT}' が無い — テキストの GPU 経路を"
            " 描けない"
        )
    label = model["quants"][model["defaultQuant"]]["weights"][WAN_TEXT_ENCODER_COMPONENT]
    parts = weights[label]["container"]["parts"]
    borrowed = {(ref["repo"], ref["revision"]) for ref in parts if "repo" in ref}
    if len(borrowed) > 1:
        raise ValueError(f"text_encoder の越境参照の先が 1 つでない: {sorted(borrowed)}")
    return sum(ref["size"] for ref in parts), next(iter(borrowed), None)


def _gib(size: int) -> str:
    """バイト数を GiB の 3 桁で綴る（このカードの資源の数と同じ書式）。"""
    return f"{size / (1 << 30):.2f} GiB"


def _encoder_whereabouts(manifest: Mapping[str, Any]) -> str:
    """text_encoder の置き場の句（自リポか、越境参照の先のリポか — manifest から）。"""
    _, borrowed = _text_encoder(manifest)
    return (
        "stored in this repository"
        if borrowed is None
        else f"referenced from [`{borrowed[0]}`](https://huggingface.co/{borrowed[0]})"
    )


def _reader_lines(manifest: Mapping[str, Any], pipeline: str) -> list[str]:
    """概要の末尾 2 項（読み手の契約と、変換に使った exporter・manifest の形式）。"""
    return [
        "- Not readable by diffusers (it's a different container with an embedded graph); the"
        f" reader is a pipeline that implements `{pipeline}`.",
        f"- Exporter used for the conversion: `{manifest['generator']}`. The distribution manifest"
        f" is `karume.json` (`{manifest['format']}`).",
    ]


def _wan21_overview(manifest: Mapping[str, Any]) -> list[str]:
    where = _encoder_whereabouts(manifest)
    return [
        "## What is this",
        "",
        "A **text-to-video** distribution of Wan2.1 T2V 1.3B, converted into the WebGPU",
        "inference runtime **Karume**'s container format (a `.krm` part sequence whose first part",
        "carries the graph and model descriptors).",
        "",
        f"- Four graphs: `{WAN_TEXT_ENCODER_COMPONENT}` (the umT5-XXL text encoder in int8,"
        f" {where}),",
        "  `transformer` (the diffusion transformer, on patchified latent tokens) and",
        "  `vae_decoder_first` / `vae_decoder_next` (the video VAE decoder, one latent frame per",
        "  call, with the causal cache passed in and out).",
        "- The rest runs on the host in TypeScript: the prompt cleaning and the tokenizer (or,",
        f'  with `{WAN_TEXT_ENCODER_OPTION}: "{WAN_TEXT_ENCODER_PATHS[1]}"`, the lookup of the',
        "  precomputed embeddings), the relative-position buckets, the patchify and RoPE tables,",
        "  classifier-free guidance as two batch-1 passes, the flow-matching UniPC scheduler, and",
        "  the tiled VAE decode. The output is `[3, frames, height, width]` float32 in `[-1, 1]`.",
        "- Verified end to end in Deno (Intel Arc B570, Deno 2.9.6): with the text encoder on the",
        "  GPU in 2-step runs (a fixed prompt and a free prompt, 832 × 480, 33 frames, each pinned",
        "  by the SHA-256 of its frames), and with the precomputed embeddings in full 50-step",
        "  runs. A 50-step run through the text encoder has not been done yet.",
        "- In a browser, Chrome on an NVIDIA GeForce RTX 5070 Ti finished one 50-step run (the",
        "  `f16` quant with the precomputed embeddings, 832 × 480, 81 frames) in 46.1 minutes.",
        "  The text encoder on the GPU and the int8 quants have not been run in a browser yet.",
        *_reader_lines(manifest, WAN_SUPPORTED_PIPELINE),
    ]


def _wan22_overview(manifest: Mapping[str, Any]) -> list[str]:
    """Wan2.2 の概要。検証の範囲は ADR 0121 追記「段 6 の結果」（sha 行 6 本は全て参照席・
    17 フレーム — 1280×704 と 704×1280 は precomputed、GPU のテキスト経路は 1280×704 だけ・
    50 ステップの製品の class の通しは 2 席・1280×704・33 フレーム）。49 / 121 フレームの生成
    スクリプトの通しは受理する入力の節（{@link _WAN22_FRAMES_CHECKED}）が書く。ブラウザは
    RTX 5070 Ti の Chrome で実用席・GPU の umT5・1280×704×121 の通し 1 本だけ（ADR 0121 追記
    「RTX 5070 Ti の Chrome」— 壁 2,417.5 s。gpu-lab が記述子の上限を広げて内部 API で呼んだ）。"""
    where = _encoder_whereabouts(manifest)
    return [
        "## What is this",
        "",
        "A **text-to-video** distribution of Wan2.2 TI2V 5B, converted into the WebGPU",
        "inference runtime **Karume**'s container format (a `.krm` part sequence whose first part",
        "carries the graph and model descriptors).",
        "",
        f"- Four graphs: `{WAN_TEXT_ENCODER_COMPONENT}` (the umT5-XXL text encoder in int8,"
        f" {where}),",
        "  `transformer` (the diffusion transformer in int8, on patchified latent tokens) and",
        "  `vae_decoder_first` / `vae_decoder_next` (the Wan2.2 video VAE decoder, one latent",
        "  frame per call, with the causal cache passed in and out).",
        "- Text to video only. The transformer graph already takes the image-conditioning inputs",
        "  (a second timestep and a condition mask), but image-to-video is not available yet.",
        "- The rest runs on the host in TypeScript: the prompt cleaning and the tokenizer (or,",
        f'  with `{WAN_TEXT_ENCODER_OPTION}: "{WAN_TEXT_ENCODER_PATHS[1]}"`, the lookup of the',
        "  precomputed embeddings), the relative-position buckets, the patchify and RoPE tables,",
        "  classifier-free guidance as two batch-1 passes, the flow-matching UniPC scheduler, the",
        "  tiled VAE decode and the unpatchify of its output. The output is",
        "  `[3, frames, height, width]` float32 in `[-1, 1]`, at 24 fps.",
        "- Verified end to end in Deno (Intel Arc B570, Deno 2.9.6) with the `f16+dit8` quant in",
        "  2-step runs of 17 frames — at 1280 × 704 and 704 × 1280 with the precomputed",
        "  embeddings, and at 1280 × 704 with the text encoder on the GPU — each pinned by the",
        "  SHA-256 of its frames; and in full 50-step runs of both quants at 1280 × 704 with 33",
        "  frames (see Resources).",
        "- In a browser, Chrome on an NVIDIA GeForce RTX 5070 Ti finished one 50-step run (the",
        "  `f16+dit8-a8-attn8-s16` quant with the text encoder on the GPU, 1280 × 704, 121 frames)",
        "  in 40.3 minutes. Shorter clips, the `f16+dit8` quant and the precomputed embeddings",
        "  have not been run in a browser yet.",
        *_reader_lines(manifest, WAN22_SUPPORTED_PIPELINE),
    ]


def _wan_base_weights(manifest: Mapping[str, Any], card: WanCard) -> list[str]:
    """帰属節。上流の revision は pin した 40 桁を全部出す（容器の provenance と同じ値）。"""
    lines = [
        "## Base weights and attribution",
        "",
        "Converted into the container format — the original checkpoint is not distributed here.",
        "",
    ]
    for name in manifest["models"]:
        source = _upstream(name, card)
        lines.append(
            f"- **`{name}`**: [{source.repo}](https://huggingface.co/{source.repo}) at commit"
            f" `{source.revision}`, licensed **{source.license}** (as of retrieval;"
            f" [full text]({WAN_LICENSE_TEXT_LINK}) — a verbatim copy is in `LICENSE.md`)."
        )
    if card.paper is not None:
        lines.append(
            f"- **Authors**: {card.generation} is released by Wan-AI"
            f" ([technical report](https://{card.paper}))."
        )
    return lines + card.attribution_notes(manifest)


def _wan21_attribution_notes(manifest: Mapping[str, Any]) -> list[str]:
    """Wan2.1 の改変の要約・text encoder の置き場・事前計算の埋め込み。"""
    lines = [
        "- **Changes made here** (listed in full in `NOTICE.md`, per Apache 2.0 §4(b)):",
        "  conversion into the Karume container format; every parameter of the float16",
        "  transformer and the VAE rounded to the nearest float16 value (weight matrices and",
        "  kernels stored as float16, computation in float32); a second, int8 copy of the",
        "  transformer whose linear weights are quantized per output channel (biases and",
        "  normalization weights kept in float32 at the source values);",
        "  the transformer graph re-expressed on patchified tokens, with the RoPE tables built on",
        "  the host from per-axis base tables and applied in a pair-swap form;",
        "  the VAE decoder re-expressed as two one-frame graphs with an explicit causal cache,",
        "  always decoded in overlapping tiles; the tokenizer converted into one table together",
        "  with lookup tables for the upstream prompt cleaning. No retraining and no fine-tuning.",
    ]
    _, borrowed = _text_encoder(manifest)
    if borrowed is None:
        lines.append(
            f"- **The text encoder** (the encoder of `{WAN_TEXT_ENCODER_UPSTREAM}`, bit-identical"
            " in float32 to this checkpoint's `text_encoder`, in int8) is stored in this"
            " repository."
        )
    else:
        repo, revision = borrowed
        lines += [
            "- **The text encoder is not stored here.** `karume.json` references the encoder of",
            f"  `{WAN_TEXT_ENCODER_UPSTREAM}` (bit-identical in float32 to this checkpoint's",
            "  `text_encoder`), converted to int8, at commit"
            f" `{revision[:16]}…` of [`{repo}`](https://huggingface.co/{repo})",
            "  (with the size and the SHA-256 of every part); that repository's `NOTICE.md` lists",
            "  the changes made to it.",
        ]
    lines += [
        "- **Precomputed embeddings**: the repository also ships the encoder's outputs for a",
        "  fixed set of prompts (see below), computed with the upstream encoder in bfloat16 and",
        "  stored as float32, for use without the text encoder.",
    ]
    return lines


def _wan22_attribution_notes(manifest: Mapping[str, Any]) -> list[str]:
    """Wan2.2 の改変の要約・text encoder の置き場・Wan2.1 の checkpoint から作った資産 2 本。

    umT5 は本家の encoder と書き、この checkpoint の bf16 の `text_encoder` はその丸めと書く
    （ADR 0122 決定 8）。資産 2 本の出所は Wan2.1 の pin（ADR 0121 決定 9 — 組み立ての門
    `wan.distribution.assert_text_embeds` / `assert_umt5_tokenizer` がその pin で突き合わせる）。
    トークナイザのファイルが 2 つの checkpoint で同じことは、pin した 2 つの revision の
    `tokenizer/` の 4 ファイルの sha256 の一致で確かめた（2026-10-06）。
    """
    text = SOURCES[WAN22_TEXT_MODEL]
    lines = [
        "- **Changes made here** (listed in full in `NOTICE.md`, per Apache 2.0 §4(b)):",
        "  conversion into the Karume container format; the transformer shipped only in int8,",
        "  its linear weights quantized per output channel (biases and normalization weights kept",
        "  in float32 at the source values); every parameter of the VAE decoder rounded to the",
        "  nearest float16 value (weight matrices and kernels stored as float16, computation in",
        "  float32); the transformer graph re-expressed on patchified tokens, with the RoPE tables",
        "  built on the host from per-axis base tables and applied in a pair-swap form, and with",
        "  the image-conditioning inputs that text-to-video leaves inactive; the VAE decoder",
        "  re-expressed as two one-frame graphs with an explicit causal cache that return the",
        "  patchified output for the host to unpatchify, always decoded in overlapping tiles; the",
        "  tokenizer converted into one table together with lookup tables for the upstream prompt",
        "  cleaning. No retraining and no fine-tuning.",
    ]
    _, borrowed = _text_encoder(manifest)
    if borrowed is None:
        lines.append(
            f"- **The text encoder** (the encoder of `{WAN_TEXT_ENCODER_UPSTREAM}`, which this"
            " checkpoint's `text_encoder` holds rounded to bfloat16, in int8) is stored in this"
            " repository."
        )
    else:
        repo, revision = borrowed
        lines += [
            "- **The text encoder is not stored here.** `karume.json` references the encoder of",
            f"  `{WAN_TEXT_ENCODER_UPSTREAM}` (this checkpoint's `text_encoder` holds the same",
            "  encoder rounded to bfloat16), converted to int8, at commit"
            f" `{revision[:16]}…` of [`{repo}`](https://huggingface.co/{repo})",
            "  (with the size and the SHA-256 of every part); that repository's `NOTICE.md` lists",
            "  the changes made to it.",
        ]
    lines += [
        "- **Precomputed embeddings and tokenizer**: the repository also ships the encoder's",
        "  outputs for a fixed set of prompts (see below), computed in bfloat16 with the umT5-XXL",
        f"  encoder of [{text.repo}](https://huggingface.co/{text.repo}) at commit"
        f" `{text.revision}`",
        "  and stored as float32, for use without the text encoder, and the tokenizer table",
        "  converted from the same commit (its tokenizer files are the same as this checkpoint's).",
    ]
    return lines


def _wan_usage(manifest: Mapping[str, Any], repo: str, card: WanCard) -> list[str]:
    """Usage 例: 動く最小形 + 普通に使いそうな optional はコメントアウトで併記する。

    NOTE: `fromAssets` は案内しない（Depth Anything のカードと同じ裁定 — HF から使う読者の入口は
    `fromPretrained`）。
    """
    model_name = manifest["defaultModel"]
    model = default_model(manifest)
    quant = model["defaultQuant"]
    model_names = " / ".join(manifest["models"])
    quant_names = " / ".join(model["quants"])
    width, height = card.default_size
    alternatives = " or ".join(
        f"{other_width} × {other_height}"
        for other_width, other_height in card.accepted_sizes
        if (other_width, other_height) != card.default_size
    )
    return [
        "## Usage",
        "",
        "```ts",
        f'import {{ encodePng, wanFrameToRgba, {card.pipeline_class} }} from "jsr:@karume/models";',
        "",
        *from_pretrained(
            card.pipeline_class,
            repo,
            [
                f'  // model: "{model_name}", // default — available: {model_names}',
                f'  // quant: "{quant}", // default — available: {quant_names}',
                f'  // {WAN_TEXT_ENCODER_OPTION}: "{WAN_TEXT_ENCODER_PATHS[0]}", // default —'
                f' "{WAN_TEXT_ENCODER_PATHS[1]}" skips the text encoder (see Prompts below)',
            ],
            disposable="await using",
        ),
        "",
        "const video = await pipeline.generate({",
        '  prompt: "A cat walks on the grass, realistic style.",',
        "  seed: 42,",
        '  // negativePrompt: "low quality, blurry", // default: the official negative prompt',
        f"  // frames: {card.default_frames}, // 4n+1, {card.frames[0]} to {card.frames[1]}",
        f"  // width: {width}, height: {height}, // or {alternatives}",
        "});",
        "",
        "for (let frame = 0; frame < video.frames; frame += 1) {",
        "  const rgba = wanFrameToRgba(video, frame);",
        "  const png = await encodePng(rgba, video.width, video.height);",
        '  await Deno.writeFile(`frame-${String(frame).padStart(2, "0")}.png`, png);',
        "}",
        "```",
        "",
        "`generate()` opens one stage at a time — the text encoder session is disposed before the",
        "transformer session is opened, and the transformer session before the VAE sessions — so",
        "no two stages are resident together. Concurrent calls are queued. Weights are fetched",
        "once and cached (verified against `karume.json`'s `size` / `sha256`). GPU memory and time",
        "per clip are listed under Resources below.",
    ]


def _wan_prompts() -> list[str]:
    """プロンプトの受理集合（GPU 経路の規則と、precomputed の経路の固定プロンプトの表 — 原文は
    逐語で出す）。"""
    gpu, precomputed = WAN_TEXT_ENCODER_PATHS
    low, high = WAN_PROMPT_TOKENS
    lines = [
        "## Prompts",
        "",
        f'With the text encoder (`{WAN_TEXT_ENCODER_OPTION}: "{gpu}"`, the default), `generate()`',
        "accepts free text. The prompt goes through a port of the upstream prompt cleaning and",
        "the umT5 tokenizer on the host, and these are rejected with `ModelInputError` before",
        "anything runs on the GPU, where the upstream pipeline would silently truncate them,",
        "repair them or map them to an unknown token instead:",
        "",
        f"- fewer than {low} or more than {high} tokens (the end token included) — an empty prompt,"
        " or",
        "  one too long;",
        "- characters outside the tokenizer's vocabulary, and special tokens such as `</s>`",
        "  written in the text;",
        "- what the cleaning would treat as an HTML character reference (write `R & D`, not",
        "  `R&D`) or as mojibake;",
        "- C1 control characters and code points unassigned in Unicode 16.0.0.",
        "",
        "When `negativePrompt` is omitted, the official negative prompt (the `negative` row",
        "below) is used, and it goes through the text encoder too.",
        "",
        f'With `{WAN_TEXT_ENCODER_OPTION}: "{precomputed}"`, the text encoder is not fetched: the'
        " repository's",
        "umT5-XXL outputs of the prompts below are used, and `generate()` accepts exactly these",
        "strings (the original text or its normalized form — `pipeline.prompts` lists both). Any",
        "other string is rejected with `ModelInputError`. The texts are shown here for reading;",
        "pass the strings from `pipeline.prompts`, since the match is exact (`ferret`, for one,",
        "starts and ends with a line break).",
        "",
        "| Name | Role | Source |",
        "| ---- | ---- | ------ |",
    ]
    lines += [
        f"| `{prompt.name}` | {prompt.role} | [source]({prompt.url}) |" for prompt in FIXED_PROMPTS
    ]
    for prompt in FIXED_PROMPTS:
        lines += ["", f"`{prompt.name}`:", "", "```text", prompt.text.strip("\n"), "```"]
    return lines


#: Wan2.1 の受理するフレーム数のうち GPU で通しを確かめた範囲（ADR 0118 段 6 / 8・ADR 0119
#: 段 10c）。
_WAN21_FRAMES_CHECKED = (
    "Only 832 × 480 has been checked end to end on",
    "  the GPU: 33 and 81 frames with the precomputed embeddings, and 33 frames with the",
    "  text encoder.",
)

#: Wan2.1 の上流との照合（`f16` 席の 2 ステップの通しと、実寸の transformer 1 forward の f64 参照
#: — ADR 0118）とタイル decode の注。
_WAN21_VERIFICATION = (
    "- **Against the upstream reference** (the `f16` quant, with the precomputed",
    "  embeddings): a 2-step run with injected noise is compared with diffusers on CPU in",
    "  float32 (the same f16-rounded weights, the same tiled decode), and one transformer",
    "  forward at each full size (33 and 81 frames) against a float64 reference. Differences",
    "  stay within tolerances measured on separate decision cases; the remaining gap is",
    "  float32 rounding in the GPU matrix products, not a porting difference.",
    "- **Tiled decode**: the VAE always decodes in overlapping tiles, so the frames differ",
    "  slightly from the upstream untiled decode.",
)

#: Wan2.2 の受理するフレーム数のうち GPU で通しを確かめた範囲（ADR 0121 追記「段 6 の結果」の
#: 2 ステップ・17 フレームの sha 行と 50 ステップの製品の class の通し〈33 フレーム〉、追記「受理
#: 寸法を公式の 2 寸法へ」の生成スクリプトの 50 ステップの試走〈49 フレーム — 製品の class では
#: 回していない。生成スクリプト ≡ 製品を示したのは実用席の 33 フレームだけ〉）、追記「RTX 3080 Ti の
#: レーンとフル verify」の開発機の 121 フレーム〈実用席・生成スクリプト・DiT の段の山 11,361 MiB /
#: 12,288 MiB — 余裕 927 MiB〉と追記「RTX 5070 Ti の Chrome」の 121 フレーム〈gpu-lab〉。
#: 3080 Ti の山は nvidia-smi の GPU 全体の値で、0.9 GiB は物理容量との差 — Deno で先に効くのは
#: 総確保の天井（ドライバの予算の 97%・時点ごとに動く）なので、実際の余裕はもっと薄いことがある
#: （docs/limitations.md の Wan2.2 の節 — 推測）。天井の数は時点で動くのでカードには書かない。
#: B570（Deno の総確保の天井 9,600 MiB）で確かめた長さは 57 フレームまで（追記「受理寸法を
#: 公式の 2 寸法へ」の試走の表 — 2 席とも 49・実用席だけ 57、どちらも生成スクリプト）。
#: それより長いクリップは ADR 0121 追記「段 10 — 121 フレームの受理」の決着どおり**非対応**と
#: 言い切る。非対応の理由は B570 で回していないことで、入るかどうかの見込みは推測: Context
#: 「容量と時間の見積り」の表が見積ったのは 2 点だけで、81 フレーム（S = 18,480）は実用席が
#: 入らず、121 は 2 席とも入らない（121 は DiT の段の診断値〈重み + backing〉も 3080 Ti で
#: 10.64 GiB と天井を超える）。61〜77 と 85〜117 は見積りも無いので「分からない」と書く（入らない
#: 見込みとは書かない）。NOTE: カードには 2.1 の既定の 81 を写さない門
#: （`test_it_carries_none_of_the_wan21_sizes_or_frame_counts`）があるので、81 は「77 フレームの次の
#: 長さ」と書く（4n+1 で 77 の次が 81 — 見積りが求めた境界ではない）。受理集合は機ごとではないので
#: admission では拒まれず、入らなければ実行の途中で落ちる（OOM の errorScope か、天井の付近では
#: device lost）と書く。資源の表の B570 の行は 49 で止まる（57 は表に無い）ことも書く。
#: 704×1280 は 2 ステップ × 17 フレームの sha 行だけ（50 ステップの通しは全て 1280×704）。
_WAN22_FRAMES_CHECKED = (
    "Checked end to end on the GPU: 17 frames at",
    "  both sizes in 2-step runs, and 33 frames at 1280 × 704 in 50-step runs. 49 frames at",
    "  1280 × 704 ran 50 steps through the same pipeline stages, driven by a development script",
    "  rather than the pipeline class.",
    "  121 frames at 1280 × 704 (the `f16+dit8-a8-attn8-s16` quant) ran 50 steps the same way on",
    "  an NVIDIA GeForce RTX 3080 Ti (12 GiB) in Deno, and in Chrome on an NVIDIA GeForce",
    "  RTX 5070 Ti. 704 × 1280 has been run only in the 2-step runs at 17 frames.",
    "  On the RTX 3080 Ti, the memory in use on the whole GPU (read with nvidia-smi) peaked at",
    "  11.09 GiB during the transformer stage, about 0.9 GiB below the card's 12 GiB; by the",
    "  runtime's own count, that stage allocated 10.64 GiB of weights and buffers. The headroom",
    "  Deno actually has can be thinner, since Deno stops allocating at a ceiling below the",
    "  card's size that varies over time, and on a 12 GiB GPU shared with other programs",
    "  121 frames may not fit (an estimate — not run).",
    "  On a GPU with about 10 GB, such as the Intel Arc B570 (where Deno can allocate about",
    "  9.4 GiB in total), clips up to 57 frames have been checked: the longest 50-step runs",
    "  there were 49 frames with both quants and 57 frames with the `f16+dit8-a8-attn8-s16`",
    "  quant, driven by the development script (its rows under Resources stop at 49 frames).",
    "  Longer clips are not supported there, because none of them has been run on the B570. A",
    "  memory estimate made before the runs covers only two of those lengths and puts both",
    "  beyond that ceiling: the next length after 77 frames with the `f16+dit8-a8-attn8-s16`",
    "  quant, and 121 frames with both quants (the 10.64 GiB the transformer stage allocated at",
    "  121 frames on the RTX 3080 Ti is above it too). The other lengths were not estimated, so",
    "  whether they fit is not known. The accepted set does not depend on the GPU, so such a",
    "  request is still accepted, and if it does not fit it fails during the run — with an",
    "  out-of-memory error (`GpuOutOfMemoryError`), or a lost device near the limit — rather",
    "  than with `ModelInputError`, and the time spent on the stages before it is lost.",
)

#: Wan2.2 の上流との照合（参照席の 2 ステップの通し — ADR 0121 追記「段 6 の結果」・実寸の
#: transformer 1 forward の f64 参照 — 追記「段 2 の結果」の r 門〈S = 8,190 / 7,920〉）とタイル
#: decode の注（patchify 空間でブレンドしてから unpatchify — 決定 6）。実用席は自機 A/B 門
#: （段 7）が未計測なので、数値では比べていないと書き、既定に採った根拠（視認 12 本）だけを書く。
_WAN22_VERIFICATION = (
    "- **Against the upstream reference** (the `f16+dit8` quant, with the precomputed",
    "  embeddings): a 2-step run at 1280 × 704 with 17 frames and injected noise is compared",
    "  with diffusers on CPU in float32 (the same int8 transformer weights, the same",
    "  f16-rounded VAE weights and tiled decode), and single transformer forwards of up to",
    "  8,190 tokens (1280 × 704 with 33 frames among them) against a float64 reference.",
    "  Differences stay within tolerances measured on separate decision cases. The",
    "  `f16+dit8-a8-attn8-s16` quant has not been compared numerically yet; it became the default",
    "  after a visual check of 12 clips (3 prompts × seeds 42–45, 1280 × 704 with 33 frames, 50",
    "  steps) on an NVIDIA GeForce RTX 3080 Ti in Deno on 2026-10-06.",
    "- **Tiled decode**: the VAE always decodes in overlapping tiles, blended in the patchified",
    "  space before the unpatchify, so the frames differ slightly from the upstream untiled",
    "  decode.",
)


def _wan_inputs(card: WanCard) -> list[str]:
    """受理する入力と、数値の門（ビット同一・参照照合）の説明。"""
    sizes = " or ".join(f"{width} × {height}" for width, height in card.accepted_sizes)
    low, high = card.frames
    first, *rest = card.frames_checked
    return [
        "## Accepted inputs",
        "",
        "- **prompt** / **negativePrompt**: see Prompts above.",
        f"- **size**: {sizes}.",
        f"- **frames**: 4n+1 from {low} to {high}. {first}",
        *rest,
        "- **steps** ≥ 1, **guidance** ≥ 1 (1 turns classifier-free guidance off and the negative",
        "  prompt is then rejected), **shift** > 0.",
        "- **seed** (the host noise generator — not torch's `randn`) or the initial noise as",
        "  `latents`.",
        "",
        "Anything outside these sets throws `ModelInputError` before any weight reaches the GPU.",
        "",
        "## Determinism and verification",
        "",
        "- **Bit-identical runs**: the same request (prompt, seed or latents, knobs) produces the",
        "  same bytes on the same GPU and driver. Release verification pins the SHA-256 of the",
        "  8-bit frames per test environment and fails on any change — the check is never relaxed",
        "  to a tolerance.",
        *card.verification,
    ]


def _wan21_resources(manifest: Mapping[str, Any]) -> list[str]:
    """実行資源の目安（ADR 0089 決定 3 — 中間テンソルの確保と束縛の大きさは manifest の
    `requiredLimits` が数えない、**manifest に存在しない事実**。BiRefNet のカードと同じ扱い）。

    数は B570・Deno の 50 ステップの通しの実測 — 33 フレームは 2026-10-02（ADR 0118 段 6 の検収）、
    81 フレームは 2026-10-03（段 8）: VRAM は fdinfo の `drm-total-vram0` の段ごとの山・時間は
    transformer 1 パス（batch 1）と VAE のタイル decode 全体。
    1 パス × 100 + VAE は通しと合う（33 フレーム: 16.8 × 100 + 129 ≈ 1,809 s / 1,811 s・
    81 フレーム: 68.6 × 100 + 315 ≈ 7,175 s / 7,173 s）。中間テンソルの大きさは形からの計算
    （FFN の `[S, 8960]` f32 は S = 14,040 で 503,193,600 B・S = 32,760 で 1,174,118,400 B。
    self-attention のスコアの行ブロック 1 枚は 81 フレームで 2,146,435,200 B —
    B570 の束縛上限 2,147,483,644 B の内）。

    text 段（GPU 経路）の数は ADR 0119 追記「段 10c の GPU の門と段 10d-4 の結果」の実測
    （2 ステップ・33 フレームの通し 1 回と、umT5 単体の 1 forward）。席ごとの transformer の数は
    `WAN_QUANT_TRANSFORMER` の出所のとおり。

    MUST: 実測していない条件の数は載せない — 受理集合の別の寸法・フレーム数・別の GPU の数を推し
    量って書かない。ブラウザは RTX 5070 Ti の Chrome で `f16` 席・資産の経路の 81 フレームの
    通しだけを確かめた（ADR 0118 追記「段 9 の結果」）ので、それ以外は未確認と書く。
    """
    for name, model in manifest["models"].items():
        if WAN_RESOURCE_QUANT not in model["quants"]:
            raise ValueError(
                f"実行資源を実測した quant '{WAN_RESOURCE_QUANT}' がモデル '{name}' の席に無い"
                f"（席: {sorted(model['quants'])}）— 実測していない席の数は名乗らない"
            )
    # 席の並びは manifest のまま。表に無い席は数を推し量らず、未計測と名乗る。
    seats = dict.fromkeys(
        quant for model in manifest["models"].values() for quant in model["quants"]
    )
    measured = [quant for quant in seats if quant in WAN_QUANT_TRANSFORMER]
    unmeasured = [f"`{quant}`" for quant in seats if quant not in WAN_QUANT_TRANSFORMER]
    practical, reference, error_33, error_81 = WAN_PRACTICAL_QUANT_ERROR
    return [
        "## Resources",
        "",
        "Measured on an Intel Arc B570 in Deno 2.9.6, with the"
        f" `{WAN_RESOURCE_QUANT}` quant at 832 × 480,",
        "50 steps and guidance 5 (classifier-free guidance runs two transformer passes per step,",
        "so 100 passes); 33 frames on 2026-10-02 and 81 frames on 2026-10-03:",
        "",
        "| Frames | Transformer peak | VAE peak | Pass   | VAE decode | Clip        |",
        "| ------ | ---------------- | -------- | ------ | ---------- | ----------- |",
        "| 33     | 6.19 GiB         | 3.32 GiB | 16.8 s | 129 s      | ~30 minutes |",
        "| 81     | 7.31 GiB         | 3.78 GiB | 68.6 s | 315 s      | ~2 hours    |",
        "",
        "Pass is one transformer pass (batch 1); VAE decode is the tiled decode of the whole clip.",
        "These runs used the precomputed embeddings.",
        "",
        f'With the text encoder (`{WAN_TEXT_ENCODER_OPTION}: "{WAN_TEXT_ENCODER_PATHS[0]}"`),'
        " measured on 2026-10-03 in a 2-step run at",
        "832 × 480 and 33 frames: the text encoder stage (the umT5-XXL encoder run for the",
        "positive and the negative prompt, including building its session) took 10.4 s and",
        "peaked at 6.30 GiB; after it was disposed, 0.08 GiB more than before the stage remained",
        "allocated. In the same run the transformer stage peaked at 5.94 GiB and the VAE stage at",
        "3.32 GiB. One forward of the text encoder alone takes 0.14 s (8 tokens) to 1.73 s",
        "(488 tokens). A 50-step run through the text encoder has not been done yet.",
        "",
        "Per quant, at 832 × 480 (the 50-step clips with the precomputed embeddings and",
        "guidance 5):",
        "",
        "| Quant | Frames | Pass | Transformer allocation | 50-step clip |",
        "| ----- | ------ | ---- | ---------------------- | ------------ |",
        *(
            f"| `{quant}` | {frames} | {forward} | {allocation} | {clip} |"
            for quant in measured
            for frames, forward, allocation, clip in WAN_QUANT_TRANSFORMER[quant]
        ),
        "",
        "Pass and allocation here are of the transformer run alone (one pass, batch 1); the",
        "allocation is the runtime's own count of what it allocated, not the driver's total.",
        "",
        *(
            [f"The other quants ({' / '.join(unmeasured)}) have not been measured yet.", ""]
            if unmeasured
            else []
        ),
        *(
            [
                f"- **Quality of `{practical}`**: after the first step its latent differs from",
                f"  `{reference}`'s (the same int8 weights, computed in float32) by a relative",
                f"  RMS error of {error_33} at 33 frames and {error_81} at 81 frames, mostly",
                "  from the int8 attention. Side by side with `f16` on twelve 50-step clips at",
                "  33 frames (seeds 42 to 45 with the three fixed prompts), no clear degradation",
                "  was seen.",
            ]
            if practical in seats and reference in seats
            else []
        ),
        *_download_and_memory(manifest),
        "- **Storage buffer size**: at these sizes some of the transformer's intermediate tensors",
        "  are larger than WebGPU's default `maxStorageBufferBindingSize` (128 MiB) — the",
        "  feed-forward activation `[S, 8960]` in float32 alone is about 480 MiB at 33 frames",
        "  (S = 14,040) and about 1.09 GiB at 81 frames (S = 32,760), and one row block of the",
        "  self-attention scores at 81 frames is 2.00 GiB, just under the 2 GiB binding limit the",
        "  adapter reports.",
        "  The runtime requests the adapter's own limits, which Deno grants on the B570; in a",
        "  browser the environment has to grant the adapter's limits as well. Chrome on the",
        "  RTX 5070 Ti granted them (2 GiB buffers) and ran the 81-frame clip above; other",
        "  browsers and GPUs have not been checked.",
        *_undeclared(manifest),
    ]


def _wan22_resources(manifest: Mapping[str, Any]) -> list[str]:
    """Wan2.2 の実行資源の目安（ADR 0089 決定 3 — 2.1 の {@link _wan21_resources} と同じ扱い）。

    数は {@link WAN22_RESOURCES} の出所のとおり。VAE の段の山は受理するフレーム数の通しの値だけを
    書く: 33 フレームは製品の class の通しの 4.360 / 4.361 GiB（参照席 / 実用席 — 追記「段 6 の
    結果」・記録 `outputs/bench/karume-wan2.2/2026-10-05_stage6-gpu/g1-full.log`）、49 フレームは
    生成スクリプトの通しの 4.513 / 4.512 GiB（記録
    `outputs/verify/deno-intel-graphics-bmg-g21/2026-10-05_wan22-visual/` の `vramPeaksGiB`）。
    試走の表の注の範囲 4.36〜4.52 GiB は受理の外の 57 フレームの試走を含むので使わない。

    中間テンソルの大きさは形からの計算（FFN の `[S, 14336]` f32 は S = 7,920 で 454,164,480 B・
    S = 11,440 で 656,015,360 B・S = 27,280 で 1,564,344,320 B — `ffn_dim` は pin した
    `transformer/config.json`）。121 フレームの束縛の最大 約 1.96 GiB（self-attention のスコアの
    行ブロック）は ADR 0121 追記（2026-10-07）「段 10 — 121 フレームの受理」の値。

    121 フレームの行と、テキストエンコーダの段の 9.1 s は RTX 3080 Ti の開発機の通し（{@link
    WAN22_RESOURCES} の出所）。ブラウザの数は RTX 5070 Ti の Chrome の同じ要求の壁 2,417.5 s と、
    adapter / device の `maxStorageBufferBindingSize` 2,147,483,644 B（ADR 0121 追記「RTX 5070 Ti の
    Chrome」）— どちらも別の GPU・runtime の値と名乗る。

    MUST: 実測していない条件の数は載せない（2.1 と同じ規律）。測っていない席・フレーム数・GPU・
    ブラウザの組は、まだ計測していないと書く。実用席の品質（段 7 の自機 A/B 門の相対 RMS 誤差）も
    未計測なので、数を書かない。
    """
    seats = dict.fromkeys(
        quant for model in manifest["models"].values() for quant in model["quants"]
    )
    measured = [quant for quant in seats if quant in WAN22_RESOURCES]
    if not measured:
        raise ValueError(
            f"実行資源を実測した quant（{sorted(WAN22_RESOURCES)}）が manifest の席"
            f"（{list(seats)}）に 1 つも無い — 実測していない席の数は名乗らない"
        )
    unmeasured = [f"`{quant}`" for quant in seats if quant not in WAN22_RESOURCES]
    practical, reference = WAN22_PRACTICAL_QUANT
    return [
        "## Resources",
        "",
        "The 33- and 49-frame rows were measured on an Intel Arc B570 in Deno 2.9.6 on 2026-10-05,",
        "at 1280 × 704 with 50 steps, shift 5 and guidance 5 (classifier-free guidance runs two",
        "transformer passes per step, so 100 passes); the 121-frame row at the same size, steps,",
        "shift and guidance on an NVIDIA GeForce RTX 3080 Ti (12 GiB) in Deno 2.9.6 on 2026-10-06:",
        "",
        "| Quant | Frames | Pass | Transformer peak | Transformer stage | VAE decode | Clip |",
        "| ----- | ------ | ---- | ---------------- | ----------------- | ---------- | ---- |",
        *(
            f"| `{quant}` | {frames} | {forward} | {peak} | {stage} | {decode} | {clip} |"
            for quant in measured
            for frames, forward, peak, stage, decode, clip in WAN22_RESOURCES[quant]
        ),
        "",
        "Pass is one transformer pass (batch 1, 7,920 tokens) timed on its own on 2026-10-04.",
        "For the B570 rows, Transformer peak is the total allocation during the transformer stage",
        "(the driver's fdinfo). The stage and the tiled decode of the whole clip are timed within",
        "the 50-step run. The 33-frame runs went",
        f"through `{WAN22_PIPELINE_CLASS}`; the 49-frame runs drove the same pipeline stages",
        "from a development script instead, whose transformer peak read 0.12 GiB lower than the",
        "class's in the one run measured both ways (33 frames). The VAE stage peaked at 4.36 GiB",
        "at 33 frames and 4.51 GiB at 49 frames with either quant.",
        "",
        "The 121-frame row comes from a different GPU and a different reading, so it is not",
        "comparable with the B570 rows: it was driven by the same development script with another",
        "prompt (`cat-dog-baking`, where the B570 rows used `boxing-cats`) and the text encoder on",
        "the GPU (that stage took 9.1 s and is included in Clip); its transformer peak is the",
        "memory in use on the whole GPU as read by nvidia-smi, not the driver's fdinfo; and its",
        "times include thermal throttling (the GPU reached 93 °C). The 121-frame clip has not",
        "been run for 50 steps with the `f16+dit8` quant.",
        "",
        *(
            [f"The other quants ({' / '.join(unmeasured)}) have not been measured yet.", ""]
            if unmeasured
            else []
        ),
        f'The text encoder stage (`{WAN_TEXT_ENCODER_OPTION}: "{WAN_TEXT_ENCODER_PATHS[0]}"`) has'
        " been measured with this",
        "distribution only in the 121-frame run above. In a browser, Chrome on an NVIDIA GeForce",
        "RTX 5070 Ti finished the same 121-frame request in 2,417.5 s (40.3 minutes) — a different",
        "GPU and runtime from the rows above.",
        "",
        *(
            [
                f"- **Quality of `{practical}`**: not measured yet (no relative error against",
                f"  `{reference}`); it is the default after a visual check of 12 clips on an",
                "  RTX 3080 Ti (see Verification).",
            ]
            if practical in seats and reference in seats
            else []
        ),
        *_download_and_memory(
            manifest,
            peaks="the B570 rows' peaks are of the total allocation (the driver's fdinfo), and"
            " the 121-frame row's is the memory in use on the whole GPU (nvidia-smi)",
        ),
        "- **Storage buffer size**: at these sizes some of the transformer's intermediate tensors",
        "  are larger than WebGPU's default `maxStorageBufferBindingSize` (128 MiB) — the",
        "  feed-forward activation `[S, 14336]` in float32 alone is about 433 MiB at 33 frames",
        "  (S = 7,920), about 626 MiB at 49 frames (S = 11,440) and about 1.46 GiB at 121 frames",
        "  (S = 27,280); the largest binding at 121 frames is about 1.96 GiB (a block of rows of",
        "  the self-attention scores). The runtime requests the adapter's own limits, which Deno",
        "  grants on the B570 and the RTX 3080 Ti; in a browser the environment has to grant the",
        "  adapter's limits as well. Chrome on the RTX 5070 Ti granted them (bindings of up to",
        "  2,147,483,644 bytes) and ran the 121-frame clip above; other browsers and GPUs have not",
        "  been checked.",
        *_undeclared(manifest),
    ]


def _download_and_memory(
    manifest: Mapping[str, Any],
    peaks: str = "the peaks are of the total allocation (the driver's fdinfo)",
) -> list[str]:
    """資源の注のうち、text_encoder の取得量（manifest から）と VRAM の山の読み方の 2 項。

    `peaks` は山の測り方の文。2.1 は既定のまま（全て fdinfo）。2.2 は表に nvidia-smi の行
    （121 フレーム）が混ざるので、行ごとの測り方を渡す。
    """
    encoder_bytes, borrowed = _text_encoder(manifest)
    source = "" if borrowed is None else f" from `{borrowed[0]}`"
    return [
        f"- **Download**: the quant table's Download column includes the text encoder"
        f" ({_gib(encoder_bytes)}{source});",
        f'  with `{WAN_TEXT_ENCODER_OPTION}: "{WAN_TEXT_ENCODER_PATHS[1]}"` it is not fetched.',
        f"- **GPU memory**: {peaks}. The",
        "  stages are never resident together, so a clip's peak is the largest stage peak.",
    ]


def _undeclared(manifest: Mapping[str, Any]) -> list[str]:
    """資源の注の末尾（manifest が宣言しない数であることと、宣言された limit）。"""
    return [
        "- `karume.json` does not declare these figures: its declared limits cover the resident",
        "  weights and state, not the intermediate tensors a run allocates.",
        *_declared_limits(manifest),
    ]


def _declared_limits(manifest: Mapping[str, Any]) -> list[str]:
    """manifest が quant ごとに宣言した device limit（`requiredLimits` — 無ければ行も無い）。

    Wan の宣言は text_encoder の i8 の語彙埋め込み（1 バッファ）で決まる — どの席も同じ
    text_encoder を選ぶ（weights は完全写像）ので、全席が同じ宣言を持つ。席ごとに違う manifest は
    この 1 行では描けないので落とす（推し量って 1 つに畳まない）。
    """
    declared = [quant.get("requiredLimits") for quant in default_model(manifest)["quants"].values()]
    if not any(declared):
        return []
    if any(limits != declared[0] for limits in declared):
        raise ValueError(
            f"席ごとに requiredLimits が違う（{declared}）— 全席に同じ宣言が掛かる形しか描かない"
        )
    spelled = " and ".join(f"`{key}` ≥ {value:,} bytes" for key, value in declared[0].items())
    return [
        f"- **Declared limits**: every quant declares {spelled} in `karume.json` — the largest",
        "  resident weight is the text encoder's int8 vocabulary embedding, held as one buffer.",
        "  The declaration belongs to the quant, so it applies whether or not the text encoder is",
        "  used.",
    ]


def _wan_defaults(model: Mapping[str, Any]) -> list[str]:
    """`pipelineConfig` が持つ生成の既定（manifest から）。"""
    config = model["pipelineConfig"]
    defaults = config["defaults"]
    return [
        "### Defaults",
        "",
        "Any of these knobs not passed to `generate()` is filled in from the manifest:",
        "",
        f"- **steps**: {knob(defaults['steps'])}",
        f"- **guidance**: {knob(defaults['guidance'])}",
        f"- **shift** (flow-matching shift): {knob(config['scheduler']['shift'])}",
    ]


#: Wan2.1 T2V 1.3B（`karume-wan2.1`）のカードの世代の表。
WAN21_CARD = WanCard(
    generation="Wan2.1",
    title="Wan2.1 T2V 1.3B",
    supported_pipeline=WAN_SUPPORTED_PIPELINE,
    models=WAN21_MODELS,
    pipeline_class="WanPipeline",
    paper=WAN_PAPER,
    accepted_sizes=WAN_ACCEPTED_SIZES,
    frames=WAN_FRAMES,
    default_size=WAN_DEFAULT_SIZE,
    default_frames=WAN_DEFAULT_FRAMES,
    frames_checked=_WAN21_FRAMES_CHECKED,
    verification=_WAN21_VERIFICATION,
    overview=_wan21_overview,
    attribution_notes=_wan21_attribution_notes,
    resources=_wan21_resources,
)

#: Wan2.2 TI2V 5B（`karume-wan2.2`）のカードの世代の表。技術レポートは pin した revision の上流
#: `README.md` が引くもの（Wan2.1 と同じ arXiv 2503.20314 — 2026-10-06 に HF キャッシュの pin の
#: README で確かめた）。
WAN22_CARD = WanCard(
    generation="Wan2.2",
    title="Wan2.2 TI2V 5B",
    supported_pipeline=WAN22_SUPPORTED_PIPELINE,
    models=WAN22_MODELS,
    pipeline_class=WAN22_PIPELINE_CLASS,
    paper=WAN_PAPER,
    accepted_sizes=WAN22_ACCEPTED_SIZES,
    frames=WAN22_FRAMES,
    default_size=WAN22_DEFAULT_SIZE,
    default_frames=WAN22_DEFAULT_FRAMES,
    frames_checked=_WAN22_FRAMES_CHECKED,
    verification=_WAN22_VERIFICATION,
    overview=_wan22_overview,
    attribution_notes=_wan22_attribution_notes,
    resources=_wan22_resources,
)


def render_wan_model_card(
    manifest: Mapping[str, Any],
    repo: str,
    abbreviations: Mapping[str, str],
    host_assets: Mapping[str, int] = {},
    card: WanCard = WAN21_CARD,
) -> str:
    """Wan の配布形の `README.md` 本文を組み立てる（純関数・末尾改行つき）。

    `abbreviations` は席名の部品上書きトークンの対応表（正本は `wan.distribution` の
    `WAN_QUANT_ABBREVIATIONS` — ADR 0074 決定 4）。manifest に無い事実なので、定数として写さず
    引数で受ける（anima のカードと同じ形）。`card` は世代の表（配布 recipe の Pipeline が
    世代ごとに束ねて渡す — 既定は Wan2.1）。
    """
    require_pipeline(manifest, card.supported_pipeline)
    return render(
        (
            frontmatter(_wan_metadata(manifest, card)),
            ["", f"# {card.title} — Karume", ""],
            card.overview(manifest),
            [""],
            _wan_base_weights(manifest, card),
            [""],
            models(manifest),
            [""],
            _wan_usage(manifest, repo, card),
            [""],
            _wan_prompts(),
            [""],
            _wan_inputs(card),
            [""],
            card.resources(manifest),
            *model_sections(
                manifest,
                (
                    partial(quants, abbreviations=abbreviations, host_assets=host_assets),
                    _wan_defaults,
                ),
            ),
        )
    )
