/**
 * `Encoder` — the one surface for text → the final hidden state of the umT5-XXL encoder
 * (`[L, 4096]`).
 *
 * Pipeline (every stage in Karume, no torch):
 *
 * 1. Host: turns the string into an id sequence `[L]` with the distribution's tokenizer asset
 *    (`tokenizer` — format `karume-umt5-tokenizer/1`) (`Umt5Tokenizer` in `./tokenizer.ts` — no
 *    preprocessing, no Unicode normalization, trailing `</s>`, `ModelInputError` outside the accepted
 *    set)
 * 2. Host: builds the relative-position bucket table `[L, L]`, which depends only on the valid length
 *    L (`./relative-position.ts`)
 * 3. Runs `text_encoder` once → returns the f32 `[1, L, 4096]` as `[L, 4096]` (row-major)
 *
 * No padding and no mask (the graph runs on the symbolic dimension of the valid length L — ADR 0119
 * decision 4). Padding into a fixed-length context, as Wan does, is the caller's job.
 *
 * ## MUST: one Session is opened at construction and kept until `dispose`
 *
 * The weights are about 5.3 GiB in i8, and an "open and fold every time" form would **re-upload all
 * weights for every encoding**. Producing many embeddings is the normal use, so residency is the
 * default form, as in siglip2 / depth-anything. When it alternates with another huge graph, as in
 * Wan, Wan's text stage (`src/wan/text-stage.ts`) opens and folds per stage — this class is not the
 * surface for that use.
 *
 * MUST: it still goes on the serialization chain — concurrent `encode` calls would share one Session
 * between two `run`s, and `dispose` would destroy the GPU without waiting for in-flight work to
 * complete (a flush-before-destroy violation).
 *
 * ## Cancellation (`signal`)
 *
 * Construction uses {@link FromPretrainedOptions.signal} (the fetch layer — the same place
 * as siglip2 / depth-anything); encoding uses {@link EncodeOptions.signal} (checked after waiting
 * for the turn on the serialization chain and before the run, after yielding to the event loop once —
 * a single run is indivisible). Cancellation throws `signal.reason` **as is** (not wrapped — the
 * consumer can identify it with `error === controller.signal.reason`).
 *
 * ## Distribution (`karume-umt5-xxl` — pipeline `umt5-encoder/1`)
 *
 * The first argument of {@link Encoder.fromPretrained} is required; the source this package
 * version verified is `"umt5-xxl"` in {@link SOURCES} (`./config.ts`). A local distribution is
 * passed with `denoDirectory` of `@karume/hub/deno`.
 *
 * ## MUST: numerical correctness is not guaranteed here
 *
 * Correctness is guaranteed by the real-GPU e2e (`packages/e2e/tests/e2e_wan_umt5_test.ts` — the
 * same golden and the same band as Wan's umT5 port check in that file), the tokenizer parity
 * (`umt5_tokenizer_test.ts`) and the byte-identical bucket tables
 * (`umt5_relative_position_test.ts`).
 */

import {
  acquireGpu,
  type GpuContext,
  type Session,
  type SessionDiagnostics,
  type SessionOptions,
} from "@karume/runtime";
import {
  type DistributionSource,
  type GpuFeaturesSpec,
  type HubRepoRef,
  loadManifest,
  type Manifest,
  type ModelEntry,
  type Quant,
  resolveSelection,
} from "@karume/hub";

import { settleAbort } from "../concurrency/abort.ts";
import { createOperationChain } from "../concurrency/serial.ts";
import { readAssetBuffer, readAssetJson } from "../hub/asset-readers.ts";
import {
  assetComponentOpener,
  type ComponentOpener,
  loadContainerComponents,
} from "../hub/components.ts";
import {
  type FromPretrainedComponentOptions,
  type FromPretrainedHubOptions,
  hubLoadOptions,
} from "../hub/load-options.ts";
import { toManifestSource } from "../hub/repo-ref.ts";
import { disposeSteps } from "../session/dispose-steps.ts";
import {
  assertGpuFeaturesGranted,
  assertRequiredLimitsBeforeDownload,
  assertRequiredLimitsSatisfied,
  sessionGpuFeatures,
  toAcquireGpuOptions,
} from "../session/gpu-features.ts";
import { type FamilySessionPolicy, resolveSessionOptions } from "../session/options.ts";
import { assertUmt5PipelineConfig, UMT5_PIPELINE_MAJOR, UMT5_PIPELINE_NAME } from "./config.ts";
import { type Umt5GraphContract, umt5GraphContract } from "./graph-contract.ts";
import { buildUmt5RelativePositionBuckets, UMT5_MAX_LENGTH } from "./relative-position.ts";
import { umt5HiddenStates, umt5SessionInputs } from "./session-io.ts";
import { parseUmt5TokenizerAsset, Umt5Tokenizer } from "./tokenizer.ts";

/** The message prefix (the class name). */
const OWNER = "umt5.Encoder";

/**
 * The component name in the manifest's weights table (also the role name the replacement option
 * accepts — the conventional name of ADR 0041 §3).
 */
const TEXT_ENCODER = "text_encoder";

/** This family's weights components. */
const COMPONENT_KEYS = [TEXT_ENCODER] as const;

/**
 * The key of the tokenizer asset (the manifest's per-model `assets` — `UMT5_TOKENIZER_ROLE` in the
 * recipe `wan/umt5_distribution.py`, format `karume-umt5-tokenizer/1`).
 */
const TOKENIZER = "tokenizer";

/** Construction options (shared by {@link Encoder.fromAssets} / {@link Encoder.fromPretrained}). */
export type EncoderOptions = {
  /**
   * Shares an existing GPU. **The side that passes it owns it**, so {@link Encoder.dispose} does
   * not destroy it. When omitted, the encoder calls `acquireGpu` internally and destroys it in
   * `dispose()`.
   */
  readonly gpu?: GpuContext;
  /** The model (a key of the manifest's models). `defaultModel` when omitted. */
  readonly model?: string;
  /** The execution configuration (a key of that model's quants). `defaultQuant` when omitted. */
  readonly quant?: string;
  /**
   * An observation hook that receives the diagnostics of every run (one encoding = one
   * `text_encoder` run). When per-op GPU time (`lastRunTiming`) is needed, pass
   * `acquireGpu({ gpuTiming: true })` as `gpu` (ADR 0021 — no timing by default).
   *
   * NOTE: no component name is passed — there is only one graph, so it would always be the same
   * value (as in siglip2).
   *
   * Exceptions from the callback are not swallowed (fail loudly — the encoding fails with it).
   */
  readonly onRunDiagnostics?: (diagnostics: SessionDiagnostics) => void;
};

/**
 * Fetch-layer options used only by {@link Encoder.fromPretrained} (passed through to the hub).
 * The component replacement option (`components`) has the single role `text_encoder`.
 *
 * NOTE: the description of the fetch-layer knobs, including that `headers` / `fetch` / `caches` /
 * `onRetry` are **for HTTP sources only**, lives in one place: {@link FromPretrainedHubOptions}.
 */
export type FromPretrainedOptions = EncoderOptions &
  FromPretrainedHubOptions &
  FromPretrainedComponentOptions & {
    /**
     * Cancels fetching (the manifest, about 5.3 GiB of weights, the tokenizer asset). Not carried
     * over past construction.
     */
    readonly signal?: AbortSignal;
  };

/**
 * The input for building directly from an already-fetched manifest + assets (a shape that accepts
 * the hub's `fetchAssets` result as is). `assets` holds the component `text_encoder` (the key
 * `text_encoder` of a single-form `krm`, or the part list `text_encoder[0]` / `text_encoder[1]` / …)
 * and the tokenizer asset `tokenizer` of the manifest's `assets` (JSON bytes).
 */
export type Assets = {
  readonly manifest: Manifest;
  readonly assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>;
};

/** Options for one call of {@link Encoder.encode}. */
export type EncodeOptions = {
  /**
   * Cancels the encoding. Checked after waiting for the turn on the serialization chain and before the
   * run, after yielding to the event loop once (a single run is indivisible). On cancellation it
   * throws `signal.reason` **as is** (not wrapped). The same encoder can run the next `encode` after a
   * cancellation.
   */
  readonly signal?: AbortSignal;
};

/** The result of one encoding. */
export type Encoding = {
  /** The tokenizer's id sequence (i32, length `length`, ending in `</s>` = 1). */
  readonly inputIds: Int32Array<ArrayBuffer>;
  /** The encoder's final hidden state `[length, hiddenSize]` (f32, row-major — one row per token). */
  readonly hiddenStates: Float32Array<ArrayBuffer>;
  /** The token count L (including `</s>` — 2 to 512). */
  readonly length: number;
  /** The width of one row (4,096 for umT5-XXL — read from the graph declaration). */
  readonly hiddenSize: number;
};

/**
 * The execution knobs the umT5 encoder accepts (the manifest's quant declaration). It accepts no key:
 * the distribution has a single preset, i8 weights run in f32, and the storage decides the form of
 * execution (ADR 0119 decision 5 — the storage check in `./graph-contract.ts`). If a declaration
 * appears, verify it against reference values before accepting it — never pass it to the runtime
 * silently.
 *
 * NOTE: the `export` is for the tests that cover the acceptance tables of every family and pin the
 * equivalence of the distribution mirror (not exposed on `mod.ts` / the subpath surface — ADR 0008).
 */
export const UMT5_ENCODER_SESSION_POLICY: FamilySessionPolicy = {
  linearCompute: false,
  convCompute: false,
  attentionCompute: false,
  attentionScoreStorage: false,
  linearGemvReduce: false,
  stateAttentionReduce: false,
  fuseRmsNormAdd: false,
  fuseLinearStaticQuantize: false,
  fuseGeluMulStaticQuantize: false,
  packedStaticQuantize: false,
  integerGemv: false,
};

/** The material the family admission ({@link admitUmt5}) settles. */
type Umt5Admission = {
  readonly quantName: string;
  readonly quant: Quant;
  /** The effective settings after passing the quant declaration through the acceptance table ({@link UMT5_ENCODER_SESSION_POLICY}). */
  readonly sessionOptions: SessionOptions;
  /**
   * The quant declaration plus the features the effective settings need (both the request and the
   * check read this).
   */
  readonly gpuFeatures: GpuFeaturesSpec | undefined;
  readonly contract: Umt5GraphContract;
};

/** The internal state of {@link Encoder} (not on the public surface). */
type Umt5State = {
  readonly gpu: GpuContext;
  readonly ownsGpu: boolean;
  readonly tokenizer: Umt5Tokenizer;
  readonly contract: Umt5GraphContract;
  /** The single Session opened at construction and kept until `dispose` (the MUST in the module doc). */
  readonly session: Session;
  readonly onRunDiagnostics?: (diagnostics: SessionDiagnostics) => void;
};

/**
 * The component supplier of the whole-bytes surface (`fromAssets`) (the receiving implementation is
 * shared by every family — {@link assetComponentOpener}).
 *
 * MUST: do not copy with `slice` — the weights are about 5.3 GiB, and a copy doubles the host RAM peak
 * ({@link readAssetBuffer}).
 */
const assetOpener = (assets: Assets["assets"]): Promise<ComponentOpener> =>
  assetComponentOpener(
    OWNER,
    assets,
    (key) => readAssetBuffer(OWNER, "weights / assets", assets, key),
    COMPONENT_KEYS,
  );

/**
 * Checks whether this manifest and this graph can run as the umT5 encoder (the
 * {@link FamilyAdmission} hook of `hub/components.ts` — on the fetching surface it is called
 * **before a single byte of the weight parts is fetched**).
 *
 * MUST: gather the family checks in this one function (pipeline name / major, `pipelineConfig`,
 * quant, the tokenizer asset declaration, the quant's `session`, the shared GPU's capabilities, the
 * graph declaration × the host contract). Spread over later stages, on the fetching surface they
 * would fail only **after** downloading 5.3 GiB of weights (ADR 0070 decision 5).
 * MUST: manifest contract violations and the match against the graph fail **before acquiring the
 * GPU** (the same order as the other families).
 */
const admitUmt5 = (
  manifest: Manifest,
  open: ComponentOpener,
  options: EncoderOptions,
): Umt5Admission => {
  const modelName = options.model ?? manifest.defaultModel;
  if (!Object.hasOwn(manifest.models, modelName)) {
    throw new Error(
      `${OWNER}: model '${modelName}' is not in the manifest` +
        ` (available: ${manifest.available.models.join(" / ")})`,
    );
  }
  const entry: ModelEntry = manifest.models[modelName];
  const { name, major } = entry.pipeline;
  if (name !== UMT5_PIPELINE_NAME) {
    throw new Error(
      `${OWNER}: the manifest's pipeline is '${name}/${major}'` +
        ` ('${UMT5_PIPELINE_NAME}/${UMT5_PIPELINE_MAJOR}' is required)`,
    );
  }
  if (major !== UMT5_PIPELINE_MAJOR) {
    // The only check that stops the silent degradation of "old implementation × new repository"
    // (ADR 0038 §6).
    throw new Error(
      `${OWNER}: pipeline '${name}/${major}' has an unsupported major` +
        ` (this implementation reads ${UMT5_PIPELINE_NAME}/${UMT5_PIPELINE_MAJOR})`,
    );
  }
  assertUmt5PipelineConfig(entry.pipelineConfig);

  const quantName = options.quant ?? entry.defaultQuant;
  if (!Object.hasOwn(entry.quants, quantName)) {
    throw new Error(
      `${OWNER}: quant '${quantName}' is not in the manifest` +
        ` (available: ${entry.available.quants.join(" / ")})`,
    );
  }
  const quant = entry.quants[quantName];
  // The tokenizer asset's contents are checked once it arrives ({@link buildUmt5}), but a missing
  // declaration fails before the weights (about 5.3 GiB) are fetched.
  if (!Object.hasOwn(entry.assets, TOKENIZER)) {
    throw new Error(
      `${OWNER}: the manifest assets of model '${modelName}' have no tokenizer asset '${TOKENIZER}'` +
        " (the standalone encoder encodes with the distribution's tokenizer. A distribution without a tokenizer [such as a local" +
        " experimental mirror made by intake] can be used only as Wan's component replacement [text_encoder in components])",
    );
  }
  // An unsupported declaration fails before the components are opened (= before the weight parts are
  // fetched) (the one implementation shared by every family).
  const sessionOptions = resolveSessionOptions(
    UMT5_ENCODER_SESSION_POLICY,
    quant.session,
    {},
    `${OWNER}: quant '${quantName}'`,
  );
  const gpuFeatures = sessionGpuFeatures(quant.gpuFeatures, sessionOptions);

  const contract = umt5GraphContract(open(TEXT_ENCODER), OWNER);

  // MUST: a shared GPU lacking a capability (feature / device limit) fails here — the only family
  // check that can tell before the weights are downloaded (the mapping between request and check is
  // the one in `session/gpu-features.ts`, and the later check calls the same function).
  if (options.gpu !== undefined) {
    assertGpuFeaturesGranted(gpuFeatures, options.gpu, `${OWNER}: quant '${quantName}'`);
    assertRequiredLimitsSatisfied(
      quant.requiredLimits,
      options.gpu.limits,
      `${OWNER}: quant '${quantName}'`,
    );
  }
  return { quantName, quant, sessionOptions, gpuFeatures, contract };
};

/**
 * Builds the run state from the admitted material + assets (parse the tokenizer asset → acquire the
 * GPU → build the Session).
 *
 * MUST: parse and match the assets **before acquiring the GPU** (so a different exception in an
 * environment without a GPU does not hide the real cause, a broken asset).
 * MUST: the only checks allowed after acquiring the GPU are GPU capabilities (features — ADR 0028 /
 * device limits — ADR 0089 decision 5).
 */
const buildUmt5 = async (
  admitted: Umt5Admission,
  assets: Assets["assets"],
  open: ComponentOpener,
  options: EncoderOptions,
): Promise<Umt5State> => {
  const { quantName, quant, sessionOptions, gpuFeatures, contract } = admitted;
  const tokenizer = new Umt5Tokenizer(
    parseUmt5TokenizerAsset(readAssetJson(OWNER, "weights / assets", assets, TOKENIZER), TOKENIZER),
  );
  // The tokenizer's limit equals the upper limit of the graph's symbolic dimension and of the bucket
  // table generator (ADR 0119 decision 4). Otherwise an input between the limits passes the tokenizer
  // and then fails in bucket table generation (when larger), or a length upstream accepts is rejected
  // silently (when smaller).
  if (tokenizer.maxLength !== UMT5_MAX_LENGTH) {
    throw new Error(
      `${OWNER}: the tokenizer asset's maxLength ${tokenizer.maxLength} differs from umT5's valid-length upper limit ` +
        `${UMT5_MAX_LENGTH}`,
    );
  }
  // The supplier is **the same one** admission looked at (`open` only looks up the opened component).
  const textEncoder = open(TEXT_ENCODER);
  // MUST: declared features can be requested only at device creation (ADR 0028). {@link admitUmt5}
  // checks a shared GPU with the same function, but for a device acquired here this is the only check.
  const gpu = options.gpu ?? (await acquireGpu(toAcquireGpuOptions(gpuFeatures)));
  const ownsGpu = options.gpu === undefined;
  try {
    assertGpuFeaturesGranted(gpuFeatures, gpu, `${OWNER}: quant '${quantName}'`);
    // The limits of a device acquired here are also checked here — `fromAssets` has no place for a
    // check before fetching (`fromPretrained` checks the adapter values). {@link admitUmt5} checks a
    // shared GPU.
    if (ownsGpu) {
      assertRequiredLimitsSatisfied(
        quant.requiredLimits,
        gpu.limits,
        `${OWNER}: quant '${quantName}'`,
      );
    }
    return {
      gpu,
      ownsGpu,
      tokenizer,
      contract,
      session: await textEncoder.createSession(gpu, sessionOptions),
      ...(options.onRunDiagnostics === undefined
        ? {}
        : { onRunDiagnostics: options.onRunDiagnostics }),
    };
  } catch (error) {
    // A GPU acquired internally can be released by no one else if construction fails, so return it
    // here.
    if (ownsGpu) gpu.destroy();
    throw error;
  }
};

/** Runs one id sequence (bucket table → one `text_encoder` run → output check). */
const encodeIds = async (
  state: Umt5State,
  inputIds: Int32Array<ArrayBuffer>,
  signal: AbortSignal | undefined,
): Promise<Encoding> => {
  // A cancellation that arrived while waiting for the turn on the serialization chain also takes
  // effect here (a run is indivisible — cancellation works only before the run).
  await settleAbort(signal);
  const { contract, session } = state;
  const outputs = await session.run(
    umt5SessionInputs(inputIds, buildUmt5RelativePositionBuckets(inputIds.length)),
  );
  state.onRunDiagnostics?.(session.diagnostics());
  if (!Object.hasOwn(outputs, contract.output)) {
    throw new Error(`${OWNER}: graph output '${contract.output}' is missing from the run results`);
  }
  const hiddenStates = umt5HiddenStates(
    outputs[contract.output],
    inputIds.length,
    contract.hiddenSize,
  );
  // MUST: never return non-finite output (O(L·hiddenSize)). If it failed only after reaching another
  // model downstream, the message would not point at the stage of the real cause (the same check as
  // Wan's text stage — `encodeWanPrompts` in `src/wan/text-stage.ts`).
  const broken = hiddenStates.findIndex((value) => !Number.isFinite(value));
  if (broken !== -1) {
    throw new Error(
      `${OWNER}: umT5 output (${inputIds.length} tokens) row ` +
        `${Math.floor(broken / contract.hiddenSize)}, column ${broken % contract.hiddenSize} is non-finite` +
        ` (${hiddenStates[broken]})`,
    );
  }
  return { inputIds, hiddenStates, length: inputIds.length, hiddenSize: contract.hiddenSize };
};

/**
 * umT5-XXL text → the encoder's final hidden state (one graph, resident Session).
 *
 * Construction has only two entries: {@link Encoder.fromPretrained} (fetch from the
 * distribution) and {@link Encoder.fromAssets} (already-fetched bytes) — the constructor is
 * private so that no half-built state can bypass the manifest check and the asset matching
 * (ADR 0008).
 */
export class Encoder {
  readonly #state: Umt5State;
  /** The serialization chain of encode and dispose (the public API enforces "one at a time" — module doc). */
  readonly #chain = createOperationChain();
  /**
   * The single dispose. **Not undefined means "disposed"** (a separate boolean would be derived state
   * updated independently).
   */
  #disposal: Promise<void> | undefined;

  private constructor(state: Umt5State) {
    this.#state = state;
  }

  /**
   * Fetches from the distribution and builds (`loadManifest` → `resolveSelection` → fetch **only the
   * components' descriptors (part 0)** and run admission → warm up the weight parts → `fetchAssets`
   * of the tokenizer asset → build). Blocks are read in part order at the moment the Session is built
   * (ADR 0109 — `src/hub/components.ts`). To replace a component with the same role from another
   * repository, use {@link FromPretrainedComponentOptions.components} (the single role
   * `text_encoder` — the tokenizer is not replaced).
   *
   * **`ref` is required** (there is no default source — the MUST in `src/hub/repo-ref.ts`). The
   * source this package version verified is `"umt5-xxl"` in {@link SOURCES} (`./config.ts`) —
   * to pin reproducibility yourself, the caller gives `{ repo, revision }` explicitly. A string `ref`
   * is read as `{ repo }` (= following `main`). A local distribution (`models/karume-umt5-xxl`) is
   * passed as a **source handle** (`localDirectory` / `denoDirectory` of `@karume/hub/deno`) — it
   * bypasses the check of the HF `owner/name` spelling, the network and CacheStorage (the HTTP-only
   * knobs of {@link FromPretrainedOptions} have no effect).
   */
  static async fromPretrained(
    ref: string | HubRepoRef | DistributionSource,
    options: FromPretrainedOptions = {},
  ): Promise<Encoder> {
    const source = toManifestSource(
      ref,
      `${OWNER}.fromPretrained`,
      'SOURCES["umt5-xxl"] (@karume/models/umt5)',
    );
    const hubOptions = hubLoadOptions(options);
    const loaded = await loadManifest(source, hubOptions);
    const choice = {
      ...(options.model === undefined ? {} : { model: options.model }),
      ...(options.quant === undefined ? {} : { quant: options.quant }),
    };
    const selection = resolveSelection(loaded.manifest, choice);
    const buildOptions: EncoderOptions = {
      ...(options.gpu === undefined ? {} : { gpu: options.gpu }),
      ...choice,
      ...(options.onRunDiagnostics === undefined
        ? {}
        : { onRunDiagnostics: options.onRunDiagnostics }),
    };
    // The family checks run in the admission hook (before the weight parts are fetched —
    // `hub/components.ts`).
    const { admitted, assets, open } = await loadContainerComponents(
      `${OWNER}.fromPretrained`,
      loaded,
      selection,
      COMPONENT_KEYS,
      async (open) => {
        const admitted = admitUmt5(loaded.manifest, open, buildOptions);
        // The `requiredLimits` the distribution declares are checked here **before the weight parts
        // are fetched** (ADR 0089 decision 5 — the limits of a shared GPU, or the measured adapter
        // values on the path that acquires its own).
        await assertRequiredLimitsBeforeDownload(
          admitted.quant.requiredLimits,
          buildOptions.gpu,
          `${OWNER}: quant '${admitted.quantName}'`,
        );
        return admitted;
      },
      {
        ...hubOptions,
        ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
        ...(options.components === undefined ? {} : { components: options.components }),
      },
    );
    return new Encoder(await buildUmt5(admitted, assets, open, buildOptions));
  }

  /**
   * Builds from an already-fetched manifest + assets ({@link Assets}). Opens the
   * containers, goes through the same family admission and assembly as the fetching surface, and
   * returns with **one `text_encoder` Session opened** (the MUST in the module doc).
   */
  static async fromAssets(input: Assets, options: EncoderOptions = {}): Promise<Encoder> {
    const open = await assetOpener(input.assets);
    const admitted = admitUmt5(input.manifest, open, options);
    return new Encoder(await buildUmt5(admitted, input.assets, open, options));
  }

  /**
   * Runs only the tokenizer (no GPU — callable after dispose). Returns the same id sequence as
   * {@link encode}'s `inputIds`. Outside the accepted set it is a `ModelInputError` (the rejection
   * rules are in the module doc of `src/umt5/tokenizer.ts` — out-of-vocabulary characters, special
   * tokens in the text, `▁` right after whitespace, whitespace other than U+0020 and trailing
   * whitespace, more than 512 tokens, the empty string).
   */
  tokenize(text: string): Int32Array<ArrayBuffer> {
    return this.#state.tokenizer.tokenize(text);
  }

  /**
   * Encodes one string (tokenizer → bucket table → one `text_encoder` run). Returns the id sequence and
   * the final hidden state `[length, hiddenSize]` ({@link Encoding}). An input outside the
   * accepted set is a `ModelInputError` before the GPU is touched (the same rules as
   * {@link tokenize}).
   *
   * Concurrent calls **wait and run in order** (one Session is never shared by two runs).
   */
  async encode(text: string, options: EncodeOptions = {}): Promise<Encoding> {
    // The disposed check happens at call time (checking inside the chain would also fail encodings
    // accepted before dispose).
    if (this.#disposal !== undefined) {
      throw new Error(`${OWNER}: cannot encode after dispose`);
    }
    // Input-error rejections return without waiting for the turn (the tokenizer is pure and
    // stateless).
    const inputIds = this.#state.tokenizer.tokenize(text);
    return await this.#chain(() => encodeIds(this.#state, inputIds, options.signal));
  }

  /**
   * Releases. Always folds the Session and destroys **only a GPU acquired internally** (a GpuContext
   * passed in `options.gpu` belongs to the caller and is not touched).
   *
   * MUST: wait for an in-flight encoding to complete before destroying (flush-before-destroy) —
   * putting the destruction on the chain too fixes the order of waiting and destroying in one place.
   * Later calls return the same completion.
   *
   * MUST: `gpu.destroy()` always runs even if `Session.dispose()` fails ({@link disposeSteps}).
   */
  dispose(): Promise<void> {
    this.#disposal ??= this.#chain(() =>
      disposeSteps([
        () => this.#state.session.dispose(),
        () => {
          if (this.#state.ownsGpu) this.#state.gpu.destroy();
        },
      ]),
    );
    return this.#disposal;
  }

  /** `await using` support (Explicit Resource Management) — an alias of {@link dispose}. */
  [Symbol.asyncDispose](): Promise<void> {
    return this.dispose();
  }
}
