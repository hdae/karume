/**
 * Wan's text stage — per path ({@link WanTextEncoderRoute}): the admission parts, the umT5 graph
 * contract, asset loading, the entry prompt check, and building the DiT's context input (`"gpu"`
 * creates and closes the umT5 Session in {@link encodeWanPrompts}; `"precomputed"` pads the asset's
 * rows in {@link precomputedContexts}).
 *
 * The stage order and Session lifetimes (text → DiT → VAE, each created and closed one stage at a
 * time) are defined by the doc at the top of `./pipeline.ts`; this module holds only the text stage's
 * parts. Shared by the Wan2.1 / 2.2 classes (umT5 is the same as 2.1 — ADR 0121 decisions 9 and 10).
 * `owner` is the message prefix (`"WanPipeline"` for Wan2.1, `"WanTi2vPipeline"` for Wan2.2).
 *
 * NOTE: the public types (`WanPipelineOptions` / `WanGenerateEvent` / `WanRunComponent` / `WanAssets`)
 * are taken from `./pipeline.ts` with `import type` (type-only references are erased, so there is no
 * import cycle).
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state —
 * CLAUDE.md).
 */

import type { GpuContext, SessionDiagnostics } from "@karume/runtime";
import type { Manifest, ModelEntry } from "@karume/hub";

import { ModelInputError } from "../errors.ts";
import { settleAbort } from "../concurrency/abort.ts";
import { disposeSteps } from "../session/dispose-steps.ts";
import { readAssetBuffer, readAssetJson } from "../hub/asset-readers.ts";
import type { ComponentOpener, GraphOwner, ModelComponent } from "../hub/components.ts";
import { umt5GraphContract } from "../umt5/graph-contract.ts";
import { buildUmt5RelativePositionBuckets, UMT5_MAX_LENGTH } from "../umt5/relative-position.ts";
import { padUmt5Context, umt5SessionInputs } from "../umt5/session-io.ts";
import type {
  WanAssets,
  WanGenerateEvent,
  WanPipelineOptions,
  WanRunComponent,
} from "./pipeline.ts";
import type { PromptGate, WanGenerationPlan } from "./plan.ts";
import { DIT_CONTEXT, type DitContract, type WanContexts } from "./dit-loop.ts";
import { firstNonFinite } from "./graph-io.ts";
import {
  findWanTextEmbedding,
  padWanTextEmbedding,
  parseWanTextEmbeds,
  type WanTextEmbedding,
  type WanTextEmbeds,
} from "./text-embeds.ts";
import { parseWanTokenizerAsset, WanPromptEncoder } from "./text/tokenizer.ts";

/**
 * The umT5 component (ADR 0119 step 10d design addendum D — in the manifest a cross-repository
 * reference to umT5's distribution repository). Fetched only on the `"gpu"` path.
 */
export const TEXT_ENCODER = "text_encoder";

/** The text encoder path ({@link WanPipelineOptions.textEncoder}). */
export type WanTextEncoderRoute = NonNullable<WanPipelineOptions["textEncoder"]>;

/**
 * The default path (ADR 0119 decision 7 — no automatic switching, so the path never changes silently
 * for the same string).
 */
const DEFAULT_TEXT_ENCODER: WanTextEncoderRoute = "gpu";

/**
 * Reads the spelling of the path in the construction options (omitted means the default `"gpu"`). An
 * unknown spelling is a plain `Error` (the same handling as a misspelled model / quant name — ADR 0107
 * decision 3). MUST: call before fetching — the components to fetch depend on the path.
 */
export const textEncoderRouteOf = (
  options: Pick<WanPipelineOptions, "textEncoder">,
  owner: string,
): WanTextEncoderRoute => {
  const route: unknown = options.textEncoder ?? DEFAULT_TEXT_ENCODER;
  if (route !== "gpu" && route !== "precomputed") {
    throw new Error(`${owner}: textEncoder '${String(route)}' is neither 'gpu' nor 'precomputed'`);
  }
  return route;
};

/**
 * On the `"gpu"` path, checks whether the chosen model declares the umT5 component (before fetching
 * and before opening containers).
 *
 * WHY: without the declaration the failure would come with the generic message of the fetch / full
 * surface (that the container of component 'text_encoder' is missing), which tells neither that the
 * default is `"gpu"` nor that a distribution without umT5 can be read with `"precomputed"`. An
 * unknown model is not checked here (admission fails with the list of available ones).
 */
export const assertTextEncoderDeclared = (
  manifest: Manifest,
  options: Pick<WanPipelineOptions, "model">,
  route: WanTextEncoderRoute,
  owner: string,
): void => {
  if (route !== "gpu") return;
  const modelName = options.model ?? manifest.defaultModel;
  if (!Object.hasOwn(manifest.models, modelName)) return;
  if (!Object.hasOwn(manifest.models[modelName].weights, TEXT_ENCODER)) {
    throw new Error(
      `${owner}: the weights of model '${modelName}' have no umT5 component '${TEXT_ENCODER}'` +
        ' (the default textEncoder "gpu" takes it. To run a distribution without umT5 on precomputed embeddings, use ' +
        'textEncoder: "precomputed")',
    );
  }
};

/**
 * The key of the text embedding asset (the per-model `assets` of the step 7 manifest — decision 4).
 * Read on both paths — the accepted set of `"precomputed"`, and on `"gpu"` the source of
 * `WanPipeline.prompts` (the list of examples).
 */
const TEXT_EMBEDS = "text_embeds";

/**
 * The key of umT5's tokenizer asset (the manifest's per-model `assets` — ADR 0119 step 10d design
 * addendum C, format `karume-wan-umt5-tokenizer/1`). Read only on the `"gpu"` path.
 */
const UMT5_TOKENIZER = "umt5_tokenizer";

/**
 * The default when negative is omitted (the original text of official Wan2.1's
 * `wan_shared_cfg.sample_neg_prompt` — verbatim, including the full-width commas; preprocessing folds
 * them to `,`). The `"gpu"` path encodes this string on the GPU (ADR 0119 step 10d design addendum —
 * the asset's row is not used: decision 7's "positive and negative are both made on the GPU").
 *
 * MUST: bit-identical to the original text of the text embedding asset's `negative` row (it points to
 * the same string as the `"precomputed"` default). `wan_pipeline_test.ts` checks it against both the
 * recipe's fixed prompt (`tools/export-recipes/wan/prompts.py` — `fixed-negative` of fixture
 * `wan-text/parity.json`) and the asset's row.
 *
 * NOTE: `export` is for that test (not on the `mod.ts` / subpath surface — ADR 0008).
 */
export const WAN_DEFAULT_NEGATIVE_PROMPT: string =
  "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，" +
  "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，" +
  "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，" +
  "三条腿，背景人很多，倒着走";

/** The umT5 graph contract confirmed at construction (the `"gpu"` path). */
type Umt5Contract = {
  /** The name of the graph output (`[1, L, width]` — width is the width of the DiT context). */
  readonly output: string;
};

/**
 * Matches the umT5 graph declaration against the inputs the host builds, the output it reads and the
 * storage it accepts (the check shared with the standalone umT5 — `umt5GraphContract` in
 * `src/umt5/graph-contract.ts`) and against the DiT's context input (ADR 0119 decisions 3, 4 and 5).
 *
 * MUST: call in the family admission (before the weight parts are fetched). The host pads the output
 * into the DiT as `[1, L, contextWidth]` ({@link padUmt5Context}), so a differing declaration would
 * fail only in the Session's shape check after the umT5 weights (about 5.3 GiB in i8) are uploaded, or
 * in the padding check.
 *
 * - The shared check: two inputs (`input_ids` `[1, L]`, `relative_position_buckets` `[L, L]`, i32,
 *   the same symbol L), one f32 output `[1, L, W]`, storage limited to i8 per-channel weights and f32
 *   tables. Here W is passed as the width of the DiT's `encoder_hidden_states` (`dit.contextWidth`).
 * - Wan's part: the upper bound of the valid length ({@link UMT5_MAX_LENGTH}) fits in the row count of
 *   the DiT context.
 *
 * NOTE: `export` is for the family admission and for tests that pin the checks without a GPU (not on
 * the `mod.ts` / subpath surface — ADR 0008).
 */
export const umt5Contract = (
  textEncoder: GraphOwner,
  dit: Pick<DitContract, "contextRows" | "contextWidth">,
  owner: string,
): Umt5Contract => {
  const { output } = umt5GraphContract(textEncoder, owner, dit.contextWidth);
  if (UMT5_MAX_LENGTH > dit.contextRows) {
    throw new Error(
      `${owner}: umT5's valid-length upper limit ${UMT5_MAX_LENGTH} exceeds the DiT context row count ${dit.contextRows}`,
    );
  }
  return { output };
};

/** The text-stage material settled by the family admission (`"gpu"` carries the umT5 contract). */
export type WanTextAdmission =
  | { readonly route: "precomputed" }
  | { readonly route: "gpu"; readonly contract: Umt5Contract };

/**
 * The `"gpu"` path reads the tokenizer asset — without a declaration, fail before fetching umT5 (about
 * 5.3 GiB) (called in the family admission; the contents are checked by {@link loadWanTextStage} once
 * the asset arrives).
 */
export const assertTokenizerDeclared = (
  entry: ModelEntry,
  route: WanTextEncoderRoute,
  owner: string,
): void => {
  if (route === "gpu" && !Object.hasOwn(entry.assets, UMT5_TOKENIZER)) {
    throw new Error(
      `${owner}: the manifest assets have no umT5 tokenizer asset '${UMT5_TOKENIZER}'` +
        ' (the textEncoder: "gpu" path reads it. To run on precomputed embeddings only, use textEncoder: "precomputed")',
    );
  }
};

/**
 * The path's admission (called in the family admission — before the weight parts are fetched).
 * `"gpu"` matches the umT5 graph declaration against the DiT's context input ({@link umt5Contract}).
 */
export const admitWanText = (
  route: WanTextEncoderRoute,
  open: ComponentOpener,
  dit: DitContract,
  owner: string,
): WanTextAdmission =>
  route === "gpu" ? { route, contract: umt5Contract(open(TEXT_ENCODER), dit, owner) } : { route };

/** Checks that the embedding asset's width and valid lengths fit the DiT context `[1, rows, width]`. */
const assertEmbedsFitContext = (dit: DitContract, embeds: WanTextEmbeds, owner: string): void => {
  if (dit.contextWidth !== embeds.width) {
    throw new Error(
      `${owner}: '${DIT_CONTEXT}' width ${dit.contextWidth} differs from the embedding asset width ${embeds.width}`,
    );
  }
  const longest = Math.max(...embeds.entries.map((entry) => entry.tokens));
  if (longest > dit.contextRows) {
    throw new Error(
      `${owner}: embedding valid length ${longest} exceeds the context row count ${dit.contextRows}`,
    );
  }
};

/** The text stage's material (per path). */
export type WanTextStage =
  /** Looks up the asset's embeddings (`textEmbeds` — no Session is created). */
  | { readonly kind: "precomputed" }
  | {
      readonly kind: "gpu";
      /** The prompt layer (preprocessing → tokenizer — the entry check). */
      readonly encoder: WanPromptEncoder;
      /** The umT5 component (a Session is created and closed per generate — decision 11). */
      readonly component: ModelComponent;
      readonly contract: Umt5Contract;
    };

/**
 * Reads and builds the text stage's assets (called at construction — after the assets arrive, before
 * the GPU is acquired). Parses the embedding asset and matches it against the DiT's context input; on
 * the `"gpu"` path, parses the tokenizer asset and looks up the umT5 component. The step-boundary
 * checks before and after (`settleAbort`) are placed by the caller.
 */
export const loadWanTextStage = async (
  textEncoder: WanTextAdmission,
  assets: WanAssets["assets"],
  open: ComponentOpener,
  dit: DitContract,
  signal: AbortSignal | undefined,
  owner: string,
): Promise<{ readonly textEmbeds: WanTextEmbeds; readonly text: WanTextStage }> => {
  const textEmbeds = parseWanTextEmbeds(
    readAssetBuffer(owner, "weights / assets", assets, TEXT_EMBEDS),
  );
  assertEmbedsFitContext(dit, textEmbeds, owner);
  let text: WanTextStage = { kind: "precomputed" };
  if (textEncoder.route === "gpu") {
    await settleAbort(signal);
    const encoder = new WanPromptEncoder(
      parseWanTokenizerAsset(
        readAssetJson(owner, "weights / assets", assets, UMT5_TOKENIZER),
        UMT5_TOKENIZER,
      ),
    );
    // The tokenizer's upper bound is the same value as the upper bound of the graph's symbolic
    // dimension and of the bucket-table generator (decision 4). If they differ, a prompt between the
    // bounds passes the tokenizer and then fails in bucket-table generation (the larger side), or a
    // length upstream accepts is silently rejected (the smaller side).
    if (encoder.maxLength !== UMT5_MAX_LENGTH) {
      throw new Error(
        `${owner}: the tokenizer asset's maxLength ${encoder.maxLength} differs from umT5's valid-length upper limit ` +
          `${UMT5_MAX_LENGTH}`,
      );
    }
    // The default negative MUST pass this asset's check — if it does not, the asset is inconsistent,
    // and failing at the entry of a generation that omitted negativePrompt as an input error
    // (`ModelInputError`) would blame the wrong party (it is not the caller's input).
    try {
      encoder.encode(WAN_DEFAULT_NEGATIVE_PROMPT, "default negative");
    } catch (cause) {
      throw new Error(
        `${owner}: the default negative (the official sample_neg_prompt) does not pass the tokenizer asset's check`,
        { cause },
      );
    }
    text = {
      kind: "gpu",
      encoder,
      component: open(TEXT_ENCODER),
      contract: textEncoder.contract,
    };
  }
  return { textEmbeds, text };
};

/**
 * The prompt check of the `"precomputed"` path — looks the prompt up in the text embedding asset's
 * set (outside the set is a `ModelInputError`).
 */
export const precomputedPromptGate = (
  embeds: WanTextEmbeds,
  owner: string,
): PromptGate<WanTextEmbedding> => ({
  resolve: (text, what) => {
    const entry = findWanTextEmbedding(embeds, text);
    if (entry === undefined) {
      throw new ModelInputError(
        `${what} is not in the text embedding asset's set (the precomputed path [textEncoder: "precomputed"] accepts ` +
          `only these ${embeds.entries.length} entries: ${embeds.entries
            .map((entry) => entry.name)
            .join(
              " / ",
            )} — match the original or the normalized text exactly. ${owner}.prompts lists them)`,
      );
    }
    return entry;
  },
  defaultNegative: () => {
    const defaults = embeds.entries.filter((entry) => entry.role === "negative");
    if (defaults.length !== 1) {
      throw new ModelInputError(
        `negativePrompt was omitted, but the asset has ${defaults.length} negative rows (only a single one becomes the default)`,
      );
    }
    return defaults[0];
  },
});

/**
 * The prompt check of the `"gpu"` path — turns the prompt into an id sequence through umT5's prompt
 * layer (the `prompt_clean` mirror → tokenizer — {@link WanPromptEncoder.encode}). When
 * `negativePrompt` is omitted, {@link WAN_DEFAULT_NEGATIVE_PROMPT} is encoded through the same check.
 */
export const gpuPromptGate = (encoder: WanPromptEncoder): PromptGate<Int32Array<ArrayBuffer>> => ({
  resolve: (text, what) => encoder.encode(text, what),
  defaultNegative: () => encoder.encode(WAN_DEFAULT_NEGATIVE_PROMPT, "negativePrompt (default)"),
});

/**
 * The context input of the `"precomputed"` path (the asset's rows padded with zeros up to the DiT
 * context's row count — no Session).
 */
export const precomputedContexts = (
  plan: WanGenerationPlan<WanTextEmbedding>,
  dit: DitContract,
): WanContexts => ({
  positive: padWanTextEmbedding(plan.positive, dit.contextRows, dit.contextWidth),
  negative:
    plan.negative === undefined
      ? undefined
      : padWanTextEmbedding(plan.negative, dit.contextRows, dit.contextWidth),
});

/**
 * The built material the text stage reads (this stage's part of the pipeline's internal state — taken
 * structurally).
 */
type WanTextEncodeState = {
  readonly gpu: GpuContext;
  readonly dit: DitContract;
  readonly onRunDiagnostics?: (component: WanRunComponent, diagnostics: SessionDiagnostics) => void;
};

/**
 * The text stage (the `"gpu"` path — creates the umT5 Session, runs positive → negative once each and
 * closes it; `end` comes after the close). Returns the output `[1, L, width]` padded with zeros up to
 * the DiT context's row count ({@link padUmt5Context}).
 *
 * MUST: close before the DiT stage is created (ADR 0119 decision 11 — umT5 i8 at 5.30 GiB and the DiT
 * stage cannot coexist under the B570's ceiling). A failure to close does not overwrite the main
 * failure (a run failure, a cancellation, the non-finite check) (the same form as the DiT stage). The
 * Session's execution options are `{}` (the quant's `session` is taken only by the DiT Session — the
 * execution of i8 weights is decided by the storage; ADR 0119 decision 5).
 *
 * MUST: check the output's finiteness (O(L·width)). Passing a non-finite context to the DiT would fail
 * at the step-1 latent check with a message pointing at the DiT (blaming the wrong stage for the true
 * cause).
 */
export const encodeWanPrompts = async (
  state: WanTextEncodeState,
  text: Extract<WanTextStage, { readonly kind: "gpu" }>,
  plan: WanGenerationPlan<Int32Array<ArrayBuffer>>,
  emit: (event: WanGenerateEvent) => Promise<void>,
  signal: AbortSignal | undefined,
  owner: string,
): Promise<WanContexts> => {
  const { dit } = state;
  const observe = state.onRunDiagnostics;
  const prompts: readonly { readonly ids: Int32Array<ArrayBuffer>; readonly label: string }[] = [
    { ids: plan.positive, label: "prompt" },
    ...(plan.negative === undefined ? [] : [{ ids: plan.negative, label: "negativePrompt" }]),
  ];

  await emit({ kind: "stage", component: "text_encoder", at: "start" });
  const session = await text.component.createSession(state.gpu, {});
  const contexts: Float32Array<ArrayBuffer>[] = [];
  let failure: { readonly error: unknown } | undefined;
  try {
    for (const { ids, label } of prompts) {
      // Before each run (one run is indivisible — cancellation takes effect before the next run).
      await settleAbort(signal);
      const outputs = await session.run(
        umt5SessionInputs(ids, buildUmt5RelativePositionBuckets(ids.length)),
      );
      observe?.("text_encoder", session.diagnostics());
      if (!Object.hasOwn(outputs, text.contract.output)) {
        throw new Error(`${owner}: umT5 output '${text.contract.output}' is missing`);
      }
      const context = padUmt5Context(
        outputs[text.contract.output],
        ids.length,
        dit.contextRows,
        dit.contextWidth,
      );
      const broken = firstNonFinite(context);
      if (broken !== -1) {
        throw new Error(
          `${owner}: umT5 output (${label}, ${ids.length} tokens) row ` +
            `${Math.floor(broken / dit.contextWidth)}, column ${broken % dit.contextWidth} is non-finite` +
            ` (${context[broken]}) — not passed to the DiT stage`,
        );
      }
      contexts.push(context);
    }
  } catch (error) {
    failure = { error };
    throw error;
  } finally {
    await disposeSteps([
      () => {
        if (failure !== undefined) throw failure.error;
      },
      () => session.dispose(),
    ]);
  }
  await emit({ kind: "stage", component: "text_encoder", at: "end" });
  return { positive: contexts[0], negative: contexts.at(1) };
};
