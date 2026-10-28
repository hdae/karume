"""The fixed prompts of MOSS-Image (the inputs the precomputed text embeddings cover — ADR NNNN
decision 12).

The embedding script (`moss.text_embeds`) and, later, the card and the assembly checks read this one
table. Neither torch nor diffusers is imported here, so the readers that must not load torch can
import it.

The texts are quoted verbatim (line breaks and full-width punctuation included) from the HF model
card's quick-start example and from the two "recommended example prompts" of the upstream user
guide; the negative is the upstream default, the empty string (HF README, "Sampling settings").

NOTE: the second user-guide example is the model card's example with typographic quotes (“ ”)
around the Chinese note instead of ASCII quotes — both are kept, as quoted. The user guide pairs
its first example with the negative prompt "worst quality, bad anatomy"; only the upstream default
(empty) negative is embedded.

NOTE: this file holds the Chinese text of the prompts; the language gate allows it by exact path
(`tools/lang/language-allow-list.ts`).
"""

from __future__ import annotations

from dataclasses import dataclass

# `moss.sources` imports the standard library only at the top level, so this stays torch-free.
from moss.sources import SOURCE, USER_GUIDE

#: The role vocabulary.
POSITIVE = "positive"
NEGATIVE = "negative"

#: The commits the source URLs pin (a branch URL can point at another text once the branch moves):
#: the checkpoint's revision and the user guide's commit, both from `moss.sources` (one pin each).
_HF_README_URL = f"https://huggingface.co/{SOURCE.repo}/blob/{SOURCE.revision}/README.md"


@dataclass(frozen=True)
class FixedPrompt:
    """One fixed prompt (the asset's tensor name, its role, the text and its source)."""

    name: str
    role: str
    #: The source text verbatim (no normalisation — upstream applies none either).
    text: str
    #: The source URL (pinned to a commit).
    url: str
    #: Where in the source the text is (fine enough for a reader to find it).
    locator: str


FIXED_PROMPTS: tuple[FixedPrompt, ...] = (
    FixedPrompt(
        name="fridge-note",
        role=POSITIVE,
        text=(
            "A wide view of a refrigerator beside a kitchen counter with a fruit bowl. A small"
            " sticky note on the refrigerator door is handwritten in Chinese:\n"
            '"记得买牛奶和鸡蛋！\n'
            "牛奶两盒，鸡蛋一打。\n"
            '顺便带一袋面包。"'
        ),
        url=_HF_README_URL,
        locator="Quick start: prompt = '''…''' (the triple-quoted string passed to pipe)",
    ),
    FixedPrompt(
        name="guide-whiteboard",
        role=POSITIVE,
        text=(
            '一个穿着"MOSI"标志的T恤的中国美女正拿着黑色的马克笔面向镜头微笑。'
            '她身后的玻璃板上手写体写着 "一、MOSS-Image的技术路线：'
            "5B 的单流 DiT 模型，基于 Muon 训练。二、MOSS-Image的模型特色：1、复杂文字渲染。"
            "支持中英渲染、自动布局； 2、使用 Layout 进行精准图像控制，支持文字编辑和物体增减等"
            '自由编辑。三、MOSS-Image的后续发展：探索 R2I 参考生图、助力生成式AI发展。"'
        ),
        url=USER_GUIDE.url,
        locator="Prompt Guidelines: the first recommended example prompt (line 38)",
    ),
    FixedPrompt(
        name="guide-fridge-note",
        role=POSITIVE,
        text=(
            "A wide view of a refrigerator beside a kitchen counter with a fruit bowl. A small"
            " sticky note on the refrigerator door is handwritten in Chinese:\n"
            "“记得买牛奶和鸡蛋！\n"
            "牛奶两盒，鸡蛋一打。\n"
            "顺便带一袋面包。”"
        ),
        url=USER_GUIDE.url,
        locator="Prompt Guidelines: the second recommended example prompt (lines 46-49)",
    ),
    FixedPrompt(
        name="negative",
        role=NEGATIVE,
        text="",
        url=_HF_README_URL,
        locator="Sampling settings: Negative prompt = Empty (pipeline default negative_prompt='')",
    ),
)


class FixedPromptError(KeyError):
    """No fixed prompt has that name."""


def prompt_by_name(name: str) -> FixedPrompt:
    """The fixed prompt called `name` (fails loudly when there is none)."""
    for prompt in FIXED_PROMPTS:
        if prompt.name == name:
            return prompt
    raise FixedPromptError(f"no fixed prompt {name!r} ({[p.name for p in FIXED_PROMPTS]})")
