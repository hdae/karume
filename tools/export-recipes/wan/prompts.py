"""Wan2.1 の固定プロンプト（第 1 段の受理集合 — ADR 0118 決定 4・利用者裁定 2026-10-02）。

テキスト埋め込み資産を書く台本（`wan.text_embeds`）と、配布形のモデルカード（`wan.card`）・
組み立ての門（`wan.distribution`）が同じ 1 表を引く。torch も diffusers も import しない —
`import dist` は torch を読まない（`tests/test_dist_driver.py` の `TestImportingTheDriver`）ので、
カードと門が引く表は torch 依存の台本の外に置く。

文面の出所（公式 README / Diffusers ドキュメントの例文）と動きの量の選び方は
`wan.text_embeds` のモジュール doc。
"""

from __future__ import annotations

from dataclasses import dataclass

#: 役割の語彙。
POSITIVE = "positive"
NEGATIVE = "negative"

#: 出所の commit（URL の版を固定するため — ブランチ名の URL は先頭が動くと別の文面を指しうる）。
#: 公式リポは 2026-10-02 時点の main の先頭（2026-03-05 の commit）、Diffusers は v0.39.0 の
#: タグが指す commit。
_WAN_COMMIT = "9737cba9c1c3c4d04b33fcad41c111989865d315"
_DIFFUSERS_COMMIT = "a3608b512ed7248499a44c61d954965ed9bdae4d"


@dataclass(frozen=True)
class FixedPrompt:
    """固定プロンプト 1 本（資産のテンソル名・役割・原文・出所）。"""

    name: str
    role: str
    #: 出所の原文（改行や全角の約物を含めて逐語 — 正規化は生成時に上流の関数で掛ける）。
    text: str
    #: 出所の URL（commit を固定した版）。
    url: str
    #: URL の中の位置（読み手が原文を探せる粒度）。
    locator: str


FIXED_PROMPTS: tuple[FixedPrompt, ...] = (
    FixedPrompt(
        name="boxing-cats",
        role=POSITIVE,
        text=(
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a"
            " spotlighted stage."
        ),
        url=f"https://github.com/Wan-Video/Wan2.1/blob/{_WAN_COMMIT}/README.md",
        locator=(
            "README の t2v-1.3B の実行例の --prompt"
            "（generate.py の EXAMPLE_PROMPT['t2v-1.3B'] と同文）"
        ),
    ),
    FixedPrompt(
        name="ferret",
        role=POSITIVE,
        text=(
            "\nThe camera rushes from far to near in a low-angle shot,\n"
            "revealing a white ferret on a log. It plays, leaps into the water, and emerges, as the"
            " camera zooms in\n"
            "for a close-up. Water splashes berry bushes nearby, while moss, snow, and leaves"
            " blanket the ground.\n"
            "Birch trees and a light blue sky frame the scene, with ferns in the foreground. Side"
            " lighting casts dynamic\n"
            "shadows and warm highlights. Medium composition, front view, low angle, with depth of"
            " field.\n"
        ),
        url=(
            f"https://github.com/huggingface/diffusers/blob/{_DIFFUSERS_COMMIT}"
            "/docs/source/en/api/pipelines/wan.md"
        ),
        locator='Text-to-Video の例（T2V memory / T2V inference speed のタブ）の prompt = """…"""',
    ),
    FixedPrompt(
        name="cat-dog-baking",
        role=POSITIVE,
        text=(
            "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring"
            " flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy,"
            " with sunlight streaming through the window."
        ),
        url=(
            f"https://github.com/huggingface/diffusers/blob/{_DIFFUSERS_COMMIT}"
            "/src/diffusers/pipelines/wan/pipeline_wan.py"
        ),
        locator=(
            "EXAMPLE_DOC_STRING の prompt（WanPipeline.__call__ の例 — ドキュメントの API"
            " リファレンスに出る）"
        ),
    ),
    FixedPrompt(
        name="negative",
        role=NEGATIVE,
        text=(
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
            "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
            "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
            "三条腿，背景人很多，倒着走"
        ),
        url=f"https://github.com/Wan-Video/Wan2.1/blob/{_WAN_COMMIT}/wan/configs/shared_config.py",
        locator="wan_shared_cfg.sample_neg_prompt",
    ),
)
