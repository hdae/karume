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

#: TS 側の生成の既定（`WAN22_TI2V_GENERATION.defaults` の写し — 利用者の裁定 2026-10-09・
#: ADR 0121 追記「段 7 の結果」）。NOTE: fixture `wan-ti2v-card-limits.json` はまだ既定を
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

#: Wan2.2 の実用席の step 1 の潜在の相対 RMS 誤差（参照席 `f16+dit8` に対して — ADR 0121 段 7 の
#: 自機 A/B 門の実測・RTX 3080 Ti・2026-10-09）。`(席, 参照席, 33 フレーム, 121 フレーム)`。
#: カードが併記する視認の結果は同じ追記の利用者の判定（参照席と実用席の 12 対・50 ステップ・
#: 1280×704・33 フレーム — 劣化なし）。
WAN22_PRACTICAL_QUANT_ERROR = ("f16+dit8-a8-attn8-s16", "f16+dit8", "0.044", "0.049")
#: I2V の実用席の自機 A/B（ADR 0121 段 9d・`e2e_wan_ti2v_ab_test.ts` の i2v-33f —
#: 1280×704・33 フレーム・条件画像 1 枚・fit crop・opt-in）: step 1 の潜在の relRMS。帯は × 2 の
#: 切り上げ（9.3e-2）。
WAN22_PRACTICAL_QUANT_I2V_ERROR_33 = "0.046"


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
    #: 写し）。Usage のコメントが描く。受理集合の並びの先頭からは導かない（既定は受理集合と
    #: 別の裁定 — Wan2.2 は ADR 0121 追記「段 7 の結果」）。
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
    #: 以下は I2V を受ける世代（Wan2.2）だけが持つ行。既定は空 — Wan2.1 のカードはバイト不変
    #: （2.1 の `WanPipeline` は `image` を `ModelInputError` で拒む — ADR 0121 段 9b）。
    #: frontmatter の tags のうち pipeline tag の後に足すもの（pipeline tag は HF では 1 つなので
    #: text-to-video のまま — I2V は tag で名乗る）。
    extra_tags: tuple[str, ...] = ()
    #: Usage 節の末尾に足す行（I2V の例と画像の decode の注）。
    usage_extra: tuple[str, ...] = ()
    #: 受理する入力の節で seed の項の後に足す項（`image` / `fit`）。
    inputs_extra: tuple[str, ...] = ()


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
        tags=(WAN_PIPELINE_TAG, *card.extra_tags, "video-generation", "wan", "webgpu"),
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
    「RTX 5070 Ti の Chrome」— 壁 2,417.5 s。gpu-lab が記述子の上限を広げて内部 API で呼んだ）。

    I2V（ADR 0121 決定 11・追記「段 9b の結果」）: 条件づけは diffusers の `expand_timesteps` の形
    （`packages/models/src/wan/dit-loop.ts` の `withWanFirstFrameCondition` / `wanConditionMask` —
    条件マスクは先頭の潜在フレームのトークンが真・条件側の時刻は t = 0）。検証は RTX 3080 Ti・
    precomputed の経路・2 ステップ × 9 フレームと opt-in の 50 ステップ × 33 フレームの sha 行だけ
    （詳細は {@link _WAN22_VERIFICATION}）。ブラウザは RTX 5070 Ti の Chrome（gpu-lab・
    2026-10-10）で実用席・GPU の umT5・条件画像 896×1152 → 704×1280（crop）・20 ステップ・
    shift 7 の 33 / 121 フレームが完走した組だけ（壁 390.0 / 1,346.5 s — 一次資料は git 追跡外の
    `outputs/bench-browser/wan-browser-2026-10-10T14-11-07.191Z.json`）。参照ケースとは照合しない
    （gpu-lab の仕様）ので、品質は主張しない。"""
    where = _encoder_whereabouts(manifest)
    return [
        "## What is this",
        "",
        "A **text-to-video** and **image-to-video** distribution of Wan2.2 TI2V 5B, converted into",
        "the WebGPU inference runtime **Karume**'s container format (a `.krm` part sequence whose",
        "first part carries the graph and model descriptors).",
        "",
        f"- Seven graphs: `{WAN_TEXT_ENCODER_COMPONENT}` (the umT5-XXL text encoder in int8,"
        f" {where}),",
        "  `transformer` (the diffusion transformer in int8, on patchified latent tokens),",
        "  `vae_decoder_first` / `vae_decoder_next` (the Wan2.2 video VAE decoder, one latent",
        "  frame per call, with the causal cache passed in and out) and `vae_encoder_pre` /",
        "  `vae_encoder_attn` / `vae_encoder_post` (the Wan2.2 video VAE encoder for one",
        "  conditioning image, split around its middle attention).",
        "- Text to video by default; image to video when `generate()` is given an `image` (the",
        "  first frame — see Usage and Accepted inputs). The image is fitted to the output size,",
        "  encoded once by the VAE encoder, and at every step the first latent frame of the",
        "  transformer's input is replaced with it (the diffusers form: the tokens of that frame",
        "  are marked by the condition mask and take timestep 0). The first output frame is",
        "  therefore the fitted image after a round trip through the VAE.",
        "- The rest runs on the host in TypeScript: the prompt cleaning and the tokenizer (or,",
        f'  with `{WAN_TEXT_ENCODER_OPTION}: "{WAN_TEXT_ENCODER_PATHS[1]}"`, the lookup of the',
        "  precomputed embeddings), the relative-position buckets, the patchify and RoPE tables,",
        "  classifier-free guidance as two batch-1 passes, the flow-matching UniPC scheduler, the",
        "  tiled VAE decode and the unpatchify of its output, and for image to video the resize,",
        "  normalization and patchify of the image and the replacement of the first latent frame.",
        "  The output is `[3, frames, height, width]` float32 in `[-1, 1]`, at 24 fps.",
        "- Text to video was verified end to end in Deno (Intel Arc B570, Deno 2.9.6) with the",
        "  `f16+dit8` quant in 2-step runs of 17 frames — at 1280 × 704 and 704 × 1280 with the",
        "  precomputed embeddings, and at 1280 × 704 with the text encoder on the GPU — each",
        "  pinned by the SHA-256 of its frames; and in full 50-step runs of both quants at",
        "  1280 × 704 with 33 frames (see Resources).",
        "- Image to video was verified end to end in Deno on an NVIDIA GeForce RTX 3080 Ti with",
        "  the precomputed embeddings, in 2-step runs of 9 frames and (opt-in) 50-step runs of 33",
        "  frames pinned by the SHA-256 of their frames (see Determinism and verification).",
        "- In a browser, Chrome on an NVIDIA GeForce RTX 5070 Ti finished one 50-step run (the",
        "  `f16+dit8-a8-attn8-s16` quant with the text encoder on the GPU, 1280 × 704, 121 frames)",
        "  in 40.3 minutes. Shorter text-to-video clips, the `f16+dit8` quant and the precomputed",
        "  embeddings have not been run in a browser yet. Image to video finished there with the",
        "  same quant and the text encoder on the GPU, from an 896 × 1152 image cropped to",
        "  704 × 1280, in 20-step runs with shift 7: 390.0 s for 33 frames and 1,346.5 s for 121",
        "  frames. These browser runs were not compared with a reference.",
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
        "  the image-conditioning inputs (a condition mask and a second timestep — in image to",
        "  video the tokens of the first latent frame are masked in and take timestep 0, in text",
        "  to video the mask is all false); the VAE decoder re-expressed as two one-frame graphs",
        "  with an explicit causal cache that return the patchified output for the host to",
        "  unpatchify, always decoded in overlapping tiles; the part of the VAE encoder that",
        "  encodes one conditioning image re-expressed as three graphs split around its middle",
        "  attention, with its parameters rounded to float16 like the decoder's; the tokenizer",
        "  converted into one table together with lookup tables for the upstream prompt cleaning.",
        "  No retraining and no fine-tuning.",
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
        *card.usage_extra,
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
#: ここは全て T2V の事実（I2V の確かめた範囲は {@link _WAN22_I2V_INPUTS} と {@link
#: _WAN22_VERIFICATION} が書く）。
_WAN22_FRAMES_CHECKED = (
    "Text to video has been checked end to end",
    "  on the GPU: 17 frames at both sizes in 2-step runs, and 33 frames at 1280 × 704 in",
    "  50-step runs. 49 frames at",
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
#:
#: I2V の項（追記「段 9a の結果」「段 9b の結果」と、テストの定数 — encoder の帯
#: `WAN_I2V_ENCODER_BAND` 7.8e-5〈`packages/models/tests/helpers/wan-i2v-image.ts` — 決定用の最悪
#: 1.550e-5 × 5・参照の最大絶対値 2.89〜4.04・故障注入 5 件は帯の 1.5 万倍以上〉、通しの帯
#: `LATENT_RATIO_BANDS` 2.9e-3 / 3.5e-3・`FINAL_RATIO_BAND` 3.9e-3・`FRAME_RATIO_BAND` 8.9e-3
#: 〈`e2e_wan_ti2v_i2v_pipeline_test.ts` — 決定用 2 本の最悪 × 5・受入れ ferret は帯の内〉、故障注入
#: 5 件は latents.1 で帯の 138〜358 倍、T2V の latents.1 の帯 9.5e-4
#: 〈`e2e_wan_ti2v_pipeline_test.ts`〉）。
#: 帯が T2V より広い理由は追記「段 9b の結果」の切り分け（cat-dog-baking の条件の潜在の maxAbs
#: 1.26e-5 を DiT が条件のトークンで約 250 倍に増幅・参照の条件へ差し替えると T2V の桁に戻る）。
#: T2V の帯との比は latents.1 だけを書く（latents.0 は T2V の帯 3.6e-5 と桁が違い「約 4 倍」と
#: 一般化できない・フレームの帯は I2V の方が狭い）。sha 行は全て RTX 3080 Ti・precomputed の経路
#: （既定のレーン = 参照席と実用席の boxing-cats・opt-in = cat-dog-baking / ferret / 704×1280）。
#: 50 ステップの I2V（1280×704・33 フレーム）は opt-in の sha 行を参照席と実用席の 2 本持つ
#: （実用席は参照席と 1 bit 以上違う床 — `136cee60` / `28ab44fe`・`KARUME_WAN_TI2V_I2V_FULL=1`）。
#: 実用席の品質の数（段 7 の A/B と視認）は T2V で測ったもの。I2V の席の比べは段 9d で済
#: （A/B 0.046・視認 12 対に破綻なし — 2026-10-10）で、同じ文の中に I2V の分も書く。カードは上流に
#: 対する I2V の品質は主張しない。
#: NOTE: 元画像の寸法（832×480）は書かない — 2.1 の寸法を写さない門
#: （`test_it_carries_none_of_the_wan21_sizes_or_frame_counts`）に掛かる。
_WAN22_VERIFICATION = (
    "- **Text to video against the upstream reference** (the `f16+dit8` quant, with the",
    "  precomputed embeddings): a 2-step run at 1280 × 704 with 17 frames and injected noise is",
    "  compared with diffusers on CPU in float32 (the same int8 transformer weights, the same",
    "  f16-rounded VAE weights and tiled decode), and single transformer forwards of up to",
    "  8,190 tokens (1280 × 704 with 33 frames among them) against a float64 reference.",
    "  Differences stay within tolerances measured on separate decision cases. The",
    "  `f16+dit8-a8-attn8-s16` quant is compared with `f16+dit8` on the same GPU: the first",
    "  step's latent at 1280 × 704 with 33 and 121 frames stays within twice the measured",
    "  relative error, differs from `f16+dit8`, and repeats bit for bit. It became the default",
    "  after a visual check of 12 clips (3 prompts × seeds 42–45, 1280 × 704 with 33 frames, 50",
    "  steps) on an NVIDIA GeForce RTX 3080 Ti in Deno on 2026-10-06, and the same 12 clips",
    "  side by side with `f16+dit8` showed no clear degradation on 2026-10-09. On image to",
    "  video the same first-step check holds at 1280 × 704 with 33 frames (one image, `crop`,",
    f"  an opt-in case: relative RMS error {WAN22_PRACTICAL_QUANT_I2V_ERROR_33}, within twice the",
    "  measured error), and 12 clips (3 images × seeds 42–45, 1280 × 704 with 33 frames, 50",
    "  steps) side by side with `f16+dit8` showed no clear degradation on 2026-10-10 either.",
    "- **Image to video against the upstream reference** (the `f16+dit8` quant, with the",
    "  precomputed embeddings, on an NVIDIA GeForce RTX 3080 Ti): the resized image and the",
    "  encoder input match Pillow and the upstream preprocessing bit for bit. The encoded image",
    "  is compared with the upstream untiled VAE encode (diffusers on CPU in float32) for three",
    "  images, each at the two accepted sizes and at a small 256 × 160 size outside them: the",
    "  largest absolute difference of the normalized latent stays within 7.8e-5, where the",
    "  latent's values reach about 3 to 4. A 2-step run at 1280 × 704",
    "  with 9 frames, injected noise and the same three images is compared with diffusers'",
    "  `WanImageToVideoPipeline` on CPU in float32 (the same int8 transformer weights, the same",
    "  f16-rounded VAE weights and tiled decode). The largest difference, relative to the",
    "  reference's largest value, stays within 2.9e-3 for the latent after the first step,",
    "  3.5e-3 after the second, 3.9e-3 for the latent passed to the VAE and 8.9e-3 for the",
    "  frames, and the encoded image inside the run stays within the encoder's 7.8e-5. Each",
    "  tolerance is five times the worst of two deciding images, and the third image is",
    "  checked against it.",
    "- **Why the image-to-video latent tolerances are wider** than the text-to-video ones",
    "  (3.5e-3 against 9.5e-4 after the second step): the difference gathers in the tokens of",
    "  the conditioning frame. On the image that sets these tolerances, the encoder's output",
    "  differs from the reference by 1.26e-5 at most (well inside its own tolerance), and the",
    "  transformer amplifies that about 250 times on those tokens. With the reference's encoded",
    "  image put in its place, the difference falls back to the text-to-video size, so it does",
    "  not come from the transformer's own precision.",
    "- **Injected faults**: five deliberate faults in the conditioning (an all-false condition",
    "  mask, the mask shifted by one token, the two timestep inputs swapped, the first frame",
    "  not replaced, the encoded image not normalized) put the latent after the second step 138",
    "  to 358 times its tolerance away, and each also fails a direct check of the input it",
    "  breaks. Five faults in the encoder land 15,000 times its tolerance away or more.",
    "- **Pinned image-to-video runs**: the same request with the same `image` and `fit`",
    "  repeats bit for bit, and the SHA-256 of the frames is pinned for one image at",
    "  1280 × 704 with 9 frames (2 steps) with both quants — the `f16+dit8-a8-attn8-s16` result",
    "  is also required to differ from the `f16+dit8` one, so the quant is known to be in use —",
    "  and, in an opt-in set, for the other two images and for 704 × 1280 with the `f16+dit8`",
    "  quant. In the same opt-in set, 50-step image-to-video runs at 1280 × 704 with 33 frames",
    "  are pinned with both quants on the NVIDIA GeForce RTX 3080 Ti, the",
    "  `f16+dit8-a8-attn8-s16` result again required to differ from the `f16+dit8` one.",
    "- **Tiled decode**: the VAE always decodes in overlapping tiles, blended in the patchified",
    "  space before the unpatchify, so the frames differ slightly from the upstream untiled",
    "  decode.",
)

#: Wan2.2 の受理する入力のうち I2V の項（`image` / `fit` —
#: `packages/models/src/wan/ti2v-pipeline.ts` の `WanTi2vGenerateRequest`・門 `plan.ts` の
#: `planWanRequest`・前処理 `i2v-preprocess.ts`）。寸法の選択は受理集合から公式の比較式
#: `max(r / rc, rc / r)`・同点は横長（`selectWanI2vSize`）— 受理集合が 2 寸法なら「幅 ≥ 高さ →
#: 1280×704」と同じ。明示した欄は受理集合に合う寸法でなければ `ModelInputError`。公式は受理集合の
#: 外の寸法（16:9 で 1248×704）も選ぶ。LANCZOS は Pillow 12.3.0 の逐語の移植
#: （`packages/models/src/image/lanczos.ts` — fixture と掃引で uint8 が全一致）。`fit` の綴り違いは
#: 素の `Error`・`image` 無しの `fit` は `ModelInputError`（ADR 0121 追記「段 9b の結果」）。
#: 画素はそのまま使う（色空間の変換をしない）。
_WAN22_I2V_INPUTS = (
    "- **image**: the first frame for image to video, as decoded RGB8 pixels",
    "  `{ data, width, height }` (`data` a `Uint8Array` with 3 bytes per pixel, row by row; any",
    "  size), used as given (no color-space conversion). Without `width` / `height`, the",
    "  accepted size closest to the image's aspect ratio is used — 1280 × 704 when the image is",
    "  at least as wide as it is tall, 704 × 1280 otherwise. A given `width` and/or `height` must",
    "  match one accepted size, and that size is used. The official Wan2.2 code also picks",
    "  sizes outside this set (1248 × 704 for a 16:9 image), so the same image can lose more to",
    "  the crop here.",
    '- **fit** (only with `image`): `"crop"` (the default, as in the official Wan2.2 code)',
    "  scales the image to cover the output size, keeping its aspect ratio, and cuts out the",
    '  center; `"stretch"` (as in diffusers\' `WanImageToVideoPipeline`) resizes the width and',
    "  the height separately. Both resize with a port of Pillow's LANCZOS filter that gives the",
    "  same bytes as Pillow 12.3.0 for the same RGB8 input. `fit` without `image` throws",
    "  `ModelInputError`; a value other than these two throws a plain `Error` (the one",
    "  exception to the sentence below).",
)

#: Wan2.2 の Usage に足す I2V の例（画像の decode は呼び手 — `generate()` は RGB8 を受ける。Deno の
#: example は fast-png / jpeg-js〈`formatAsRGBA: false` で RGB を直接受ける —
#: `examples/wan/decode-image.ts`〉、ブラウザは createImageBitmap と canvas〈RGBA を出す〉）。
#: 復号の段の参照との差（EXIF の向き・アルファ・JPEG の復号器の差）は `examples/wan/README.md` と
#: `tools/gpu-lab/README.md` と同じ事実を写す。Deno の I2V は precomputed の経路でだけ回した（GPU の
#: umT5 との組は e2e でも 50 ステップの sha 行でも未実行 — ADR 0121 追記「段 9b の結果」・
#: `e2e_wan_ti2v_i2v_pipeline_test.ts`）。GPU の umT5 との組は RTX 5070 Ti の Chrome（gpu-lab・
#: 2026-10-10）でだけ完走した（{@link _wan22_overview} の docstring）。段の順は
#: `pipeline.ts` の「（encoder →）text → DiT →
#: VAE」（text の段は `"gpu"` の経路だけ）。
_WAN22_I2V_USAGE = (
    "",
    "For image to video, pass the first frame as `image`. Decoding the image file is up to the",
    "caller: the Deno example in the Karume repository (`examples/wan`) uses fast-png and",
    "jpeg-js, and in a browser `createImageBitmap` and a canvas work. `generate()` takes 8-bit",
    "RGB, so convert what the decoder gives: a canvas gives RGBA, and so does jpeg-js by",
    "default (`formatAsRGBA: false` gives RGB, as the example uses) — drop the alpha channel.",
    "",
    "The bit-for-bit match with Pillow (Accepted inputs) starts from the RGB8 pixels, so the",
    "decoding decides whether a file gives the reference's input. The official Wan2.2 code",
    "reads the file with Pillow, drops the alpha channel without compositing it over a",
    "background, and does not apply the EXIF orientation. `createImageBitmap` applies the",
    "orientation by default, and a canvas can lose precision in partly transparent pixels (it",
    "stores premultiplied alpha). For PNG, fast-png gives the same pixels as Pillow; JPEG",
    "decoders differ slightly, so jpeg-js and a browser can give pixels a little different from",
    "Pillow's for the same JPEG file.",
    "",
    "```ts",
    "// `rgb`: the decoded image, 3 bytes (R, G, B) per pixel, row by row.",
    "const clip = await pipeline.generate({",
    '  prompt: "A cat walks on the grass, realistic style.",',
    "  image: { data: rgb, width: imageWidth, height: imageHeight },",
    "  seed: 42,",
    '  // fit: "crop", // default — "stretch" resizes without keeping the aspect ratio',
    "  // width / height: omitted — the accepted size closest to the image's aspect ratio",
    "});",
    "```",
    "",
    "With an `image`, the VAE encoder stage runs first and is disposed before the next stage",
    "(the text encoder, or the transformer with the precomputed embeddings) is opened. In Deno,",
    "image to video has so far been run only with the precomputed embeddings; with the text",
    "encoder on the GPU (the default in this example) it has been run only in Chrome (see What",
    "is this).",
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
        *card.inputs_extra,
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
    ブラウザの組は、まだ計測していないと書く。実用席の品質は段 7 の自機 A/B 門の相対 RMS 誤差
    （{@link WAN22_PRACTICAL_QUANT_ERROR}）と視認の判定を書き、どちらも T2V で測ったと範囲を名乗る
    （I2V の席の比べは段 9d — I2V の品質は主張しない）。

    I2V の資源は表に入れない（表は全て T2V）。書くのは測った範囲だけ: encoder 単独のプロセスの
    VRAM の山 1,421 MiB（ADR 0121 追記「段 9a の結果」— nvidia-smi・RTX 3080 Ti）と、50 ステップの
    I2V の壁（1280×704・33 フレーム・precomputed — 参照席 1,395.3 s・実用席 723.5 s・encoder の段は
    0.5 / 0.8 s。`outputs/verify/deno-nvidia-geforce-rtx-3080-ti/` の
    `2026-10-09_wan-ti2v-i2v-pipeline-full/results.json` の壁。ADR の 23 分 20 秒 / 12 分 8 秒は
    同じ走行の `elapsedMs`）。その走行は
    VRAM を読んでいない（n/a）ので、I2V の 1 本の山は未計測と書く。
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
    practical, reference, error_33, error_121 = WAN22_PRACTICAL_QUANT_ERROR
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
        "All of the above is text to video. Image to video adds the VAE encoder stage, which runs",
        "once before the others and is released before the next stage opens. Run on its own on",
        "the RTX 3080 Ti in Deno 2.9.6, the encoder kept the whole process at a peak of 1,421 MiB",
        "(nvidia-smi) while encoding images at 1280 × 704, 704 × 1280 and 256 × 160.",
        *(
            [
                "On the same GPU on 2026-10-09, 50-step image-to-video runs at 1280 × 704 with 33",
                "frames and the precomputed embeddings took 1,395.3 s with"
                f" `{reference}` and 723.5 s",
                f"with `{practical}` in all, the encoder stage under 1 s of it; the GPU memory of",
                "those runs was not read, so an image-to-video clip's peak has not been measured.",
            ]
            if practical in seats and reference in seats
            else [
                "Image-to-video clips have not been measured for this table.",
            ]
        ),
        "",
        *(
            [
                f"- **Quality of `{practical}`**: after the first step its latent differs from",
                f"  `{reference}`'s (the same int8 weights, computed in float32) by a relative",
                f"  RMS error of {error_33} at 33 frames and {error_121} at 121 frames. Side",
                f"  by side with `{reference}` on twelve 50-step clips at 1280 × 704 with 33",
                "  frames (seeds 42 to 45 with the three fixed prompts), no clear degradation",
                "  was seen. Both were measured with text to video; on image to video the",
                f"  first-step error is {WAN22_PRACTICAL_QUANT_I2V_ERROR_33} at 33 frames (one",
                "  image, `crop`) and twelve 50-step clips (3 images × seeds 42–45) side by side",
                "  showed no clear degradation either.",
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
    extra_tags=("image-to-video",),
    usage_extra=_WAN22_I2V_USAGE,
    inputs_extra=_WAN22_I2V_INPUTS,
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
