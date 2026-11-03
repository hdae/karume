/**
 * The chat dialects of the Gemma 4 family — which upstream `chat_template.jinja` a distribution's
 * model follows (declared by `pipelineConfig.chatDialect`; absent = `"gemma4"`).
 *
 * - `"gemma4"` — the template of gemma-4-E2B-it / gemma-4-E4B-it.
 * - `"gemma4-unified"` — the template of gemma-4-12B-it (`gemma4_unified`). For plain conversation
 *   it differs in one place: with `add_generation_prompt` and thinking off, the generation prompt
 *   `<|turn>model\n` is followed by an empty thought channel `<|channel>thought\n<channel|>`.
 *
 * The rendering lives in `./chat.ts`; this module holds only the closed set, so that the config
 * parser (`../config.ts`), the tokenizer (`./tokenizer.ts`) and the renderer share one list without
 * importing each other.
 *
 * MUST: zero module side effects (the list is a frozen literal, nothing runs at import).
 */

/** One chat dialect (the closed set {@link CHAT_DIALECTS}). */
export type ChatDialect = "gemma4" | "gemma4-unified";

/** The dialect a distribution means when it does not declare one (E2B / E4B). */
export const DEFAULT_CHAT_DIALECT: ChatDialect = "gemma4";

/**
 * The accepted dialects. MUST: a closed set — an unknown spelling fails loudly instead of rendering
 * with the default template (a model prompted in another model's dialect still generates text, so
 * nothing else would notice).
 */
export const CHAT_DIALECTS: readonly ChatDialect[] = Object.freeze(["gemma4", "gemma4-unified"]);

/**
 * Type guard for {@link CHAT_DIALECTS}. Both entries of a dialect check with it and fail loudly on
 * a miss: `parsePipelineConfig` (`../config.ts`, the distribution's declaration) and the
 * `Tokenizer` constructor (`./tokenizer.ts`, a value that reached it without the parser).
 */
export const isChatDialect = (value: unknown): value is ChatDialect =>
  CHAT_DIALECTS.some((dialect) => dialect === value);
