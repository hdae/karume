"""The fixed prompts of Wan2.1 (the accepted set of step 1 — ADR 0118 decision 4, maintainer's
ruling 2026-10-02).

The script that writes the text-embeddings asset (`wan.text_embeds`), the model card of the
distribution (`wan.card`) and the assembly check (`wan.distribution`) read the same single table.
It imports neither torch nor diffusers — `import dist` does not load torch
(`TestImportingTheDriver` in `tests/test_dist_driver.py`), so the table that the card and the check
read lives outside the torch-dependent script.

Where the texts come from (example prompts of the official README / the Diffusers documentation) and
how the amount of motion was chosen: the module doc of `wan.text_embeds`.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The role vocabulary.
POSITIVE = "positive"
NEGATIVE = "negative"

#: The source commits (to pin the version in the URLs — a URL with a branch name can point at a
#: different text once the branch head moves). The official repository: the head of main as of
#: 2026-10-02 (a commit of 2026-03-05); Diffusers: the commit the v0.39.0 tag points at.
_WAN_COMMIT = "9737cba9c1c3c4d04b33fcad41c111989865d315"
_DIFFUSERS_COMMIT = "a3608b512ed7248499a44c61d954965ed9bdae4d"


@dataclass(frozen=True)
class FixedPrompt:
    """One fixed prompt (tensor name in the asset, role, original text, source)."""

    name: str
    role: str
    #: The original text of the source, verbatim (including line breaks and full-width punctuation
    #: — normalization is applied by the upstream function at generation time).
    text: str
    #: The source URL (the version with a pinned commit).
    url: str
    #: The position within the URL (fine enough for a reader to find the original text).
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
            "The --prompt of the t2v-1.3B run example in the README"
            " (the same text as EXAMPLE_PROMPT['t2v-1.3B'] in generate.py)"
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
        locator=(
            'The prompt = """…""" of the Text-to-Video example'
            " (the T2V memory / T2V inference speed tabs)"
        ),
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
            "The prompt of EXAMPLE_DOC_STRING (the WanPipeline.__call__ example — shown in the"
            " API reference of the documentation)"
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
