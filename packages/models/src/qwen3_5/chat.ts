/**
 * The Qwen3.5 chat prompt MOSS-Image's text encoder reads — **one family-specific pure function**,
 * no template evaluator (the precedent is gemma4's chat format, ADR 0084 decision 5).
 *
 * Upstream (`pipeline_moss_image.py`, `encode_prompt`) renders
 * `apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
 * add_generation_prompt=True, enable_thinking=True)` with the shipped `chat_template.jinja`
 * (byte-identical in `OpenMOSS-Team/MOSS-Image-SFT@f972bd4b…/tokenizer/` and `Qwen/Qwen3.5-9B`). For
 * that one input shape the template reduces to a fixed prefix, the trimmed content and a fixed
 * suffix — no system prompt, no tools, the thinking branch open:
 *
 *     <|im_start|>user\n{content|trim}<|im_end|>\n<|im_start|>assistant\n<think>\n
 *
 * `trim` is Jinja's, i.e. Python's `str.strip()` — not JS's `trim()` (`src/text/python-strip.ts`).
 * Special-token spellings typed in the content are left as they are; the tokenizer then cuts them
 * out as control tokens, as upstream does (`./tokenizer.ts`).
 *
 * MUST: the one input the template refuses fails here too. A content that, after trimming, starts
 * with `<tool_response>` and ends with `</tool_response>` is taken by the template as a tool result,
 * leaves no user query, and upstream raises `No user query found in messages.` — rendering it
 * anyway would produce a prompt upstream never encodes.
 *
 * MUST: every module is side-effect free.
 */

import { ModelInputError } from "../errors.ts";
import { pythonStrip } from "../text/python-strip.ts";

/** The spellings of the template (`<|im_start|>` / `<|im_end|>` are added tokens 248045 / 248046). */
const PREFIX = "<|im_start|>user\n";
const SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n";
const TOOL_RESPONSE_OPEN = "<tool_response>";
const TOOL_RESPONSE_CLOSE = "</tool_response>";

/**
 * Renders one prompt the way MOSS-Image's `encode_prompt` does (module doc): the Qwen3.5 chat
 * template for a single user message, generation prompt with thinking enabled. The empty prompt
 * (upstream's default negative) renders the bare template, 10 tokens.
 */
export const renderQwen3_5ChatPrompt = (prompt: string, label: string = "prompt"): string => {
  if (typeof prompt !== "string") {
    throw new ModelInputError(`${label} is not a string`);
  }
  const content = pythonStrip(prompt);
  if (content.startsWith(TOOL_RESPONSE_OPEN) && content.endsWith(TOOL_RESPONSE_CLOSE)) {
    throw new ModelInputError(
      `${label}: wrapped in ${TOOL_RESPONSE_OPEN}…${TOOL_RESPONSE_CLOSE} — the chat template reads` +
        " it as a tool result, not a user query, and upstream refuses it. Remove the wrapping",
    );
  }
  return `${PREFIX}${content}${SUFFIX}`;
};
