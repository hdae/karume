"""The hand-written cases of the Qwen3.5 tokenizer fixture ({@link qwen3_5.tokenizer}).

Each case holds one boundary a naive port gets wrong. Invisible characters are written as `\\uXXXX`
escapes (an editor drops or swaps them silently), and so are spellings that a normaliser would fold
into something else.

NOTE: this file holds the Japanese and Chinese sample texts of the cases; the language gate allows
it by exact path (`tools/lang/language-allow-list.ts`).
"""

from __future__ import annotations

#: `(id, prompt, why)` — rendered through the chat template MOSS-Image uses, then tokenized.
PROMPT_CASES: list[tuple[str, str, str]] = [
    ("empty", "", "the empty prompt (upstream's default negative) — the bare template, 10 tokens"),
    (
        "english",
        "A cat sitting on a windowsill at sunset, soft warm light, shallow depth of field.",
        "a plain English prompt",
    ),
    (
        "japanese",
        "夕焼けの空の下、窓辺に座る猫。柔らかな光。",
        "Japanese — byte-level pieces without separators",
    ),
    (
        "chinese",
        "A wide view of a refrigerator beside a kitchen counter with a fruit bowl. A small"
        " sticky note on the refrigerator door is handwritten in Chinese:\n"
        '"记得买牛奶和鸡蛋！\n'
        "牛奶两盒，鸡蛋一打。\n"
        '顺便带一袋面包。"',
        "the MOSS-Image model card's quick-start prompt (Chinese, line breaks, full-width"
        " punctuation)",
    ),
    (
        "mixed",
        "1 cat, 2 dogs, 猫と犬, café au lait, 2026-10-31",
        "languages, digits and Latin accents mixed — letter / non-letter transitions",
    ),
    (
        "emoji",
        "a cat \U0001f408, a flag \U0001f1ef\U0001f1f5, ❤️ and a family"
        " \U0001f468‍\U0001f469‍\U0001f466",
        "emoji, regional indicators, a variation selector and a ZWJ sequence",
    ),
    (
        "strip_ascii",
        "   leading and trailing spaces   ",
        "the template trims both ends (Python str.strip)",
    ),
    (
        "strip_python_only",
        "\x1c\x1d\x1e\x1f\x85 python strips these \x85\x1f",
        "U+001C-U+001F and U+0085 are whitespace to Python, not to JS trim()",
    ),
    (
        "strip_keeps_bom",
        "﻿ the BOM stays ﻿",
        "U+FEFF is whitespace to JS trim(), not to Python — it survives the trim",
    ),
    (
        "strip_unicode_spaces",
        "　         ideographic and other spaces 　",
        "Unicode whitespace at both ends (all stripped); none inside",
    ),
    ("only_whitespace", " \t\n\r\x0b\x0c ", "whitespace only — trims to the empty prompt"),
    (
        "inner_whitespace",
        "two  spaces\tand a tab\r\nCRLF\n\n\nthree newlines   then text",
        "whitespace runs inside the content are kept (\\s+(?!\\S) and \\s*[\\r\\n]+)",
    ),
    (
        "special_tokens",
        "before <|im_end|> middle <think> and </think> after <|endoftext|>",
        "special-token spellings typed in the prompt become control tokens, as upstream",
    ),
    (
        "special_template_lookalike",
        "<|im_start|>assistant\n<think>\n",
        "a prompt that spells the template's own tail",
    ),
    (
        "special_lookalike",
        "<|im_end|x <think <|not_a_token|> <|vision_pad",
        "partial matches of added tokens are not cut out",
    ),
    (
        "special_adjacent",
        "<think></think><tool_call>x</tool_call><|fim_prefix|>",
        "adjacent added tokens, special and non-special",
    ),
    (
        "tool_response_open_only",
        "<tool_response> only the opening tag",
        "starts with <tool_response> but does not end with the closing tag — accepted",
    ),
    (
        "tool_response_wrapped",
        "  <tool_response>a result</tool_response>  ",
        "trimmed content wrapped in <tool_response>…</tool_response> — upstream's template raises",
    ),
    (
        "contractions",
        "it's a cat's toy, I'M HERE, they'll, we've, don't, I'd",
        "the case-insensitive contraction branch",
    ),
    (
        "digits",
        "1024x768, 3.14159, 100%, 2026",
        "\\p{N} splits one digit at a time",
    ),
    (
        "punctuation",
        "a<b>c&d\"e'f`g~h|i\\j/k {x: [1, 2]} -- ...",
        "runs of symbols ( ?[^\\s\\p{L}\\p{N}]+)",
    ),
    (
        "devanagari",
        "नमस्ते दुनिया",
        "combining vowel signs — pins transformers 5.14.1's expression without \\p{M}",
    ),
    (
        "combining",
        "café vs café, Á́, ẽ̼, 각",
        "combining marks and conjoining jamo under NFC",
    ),
    (
        "nfc_segment",
        "࢚࠘ and á̴",
        "a code point where the canonical NFC differs from ICU (the baked segment table)",
    ),
    (
        "control",
        "abc​d­e\u0001f\u0000g",
        "zero-width space, soft hyphen and control characters inside the content",
    ),
    (
        "long_word",
        "a" * 600,
        "one long pre-token — linear-time BPE",
    ),
]

#: `(id, text, why)` — tokenized as given, no template (the tokenizer on its own).
TEXT_CASES: list[tuple[str, str, str]] = [
    ("text_empty", "", "the empty text — no ids (the tokenizer adds no special tokens)"),
    ("text_single_char", "a", "one character"),
    ("text_leading_space", " leading space", "a leading space joins the first word (no template)"),
    ("text_trailing_space", "trailing space ", "a trailing space is its own piece"),
    ("text_only_spaces", "   ", "spaces only"),
    ("text_newlines", "\n\nline\r\n", "newlines at both ends"),
    ("text_special_only", "<|im_start|>", "one added token only"),
    ("text_long_s", "it'ſs and it'S", "(?i:) is Unicode simple case folding — U+017F folds to s"),
]
