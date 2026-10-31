/**
 * `Qwen3_5TextEncoder` — token ids → the final hidden state of the Qwen3.5 text tower
 * (`[T, 4096]` for Qwen3.5-9B, after the final norm; ADR 0137).
 *
 * This is what MOSS-Image's `encode_prompt` reads (`last_hidden_state` of the text tower), but the
 * module holds nothing of MOSS: it takes an id sequence (the tokenizer of `./tokenizer.ts` over the
 * chat prompt of `./chat.ts` makes one) and returns its rows. Any Qwen3.5 text tower exported by the
 * recipe `tools/export-recipes/qwen3_5` with the same graph contract runs here.
 *
 * Per encoding (one graph run):
 *
 * 1. Host: positions 0 … T−1 and the RoPE rows (`./rope.ts`) next to the ids (`./session-io.ts`)
 * 2. Runs `text_encoder` once → returns the f32 `[1, T, W]` as `[T, W]` (row-major)
 *
 * No padding and no mask input (the graph runs on the valid length T and builds its causal mask from
 * `positions`).
 *
 * ## Lifetime: one Session from {@link Qwen3_5TextEncoder.open} to `dispose`
 *
 * The i8 weights are about 7.4 GiB, so the Session is built once and kept for a batch of encodings
 * (re-uploading per encoding would cost the whole upload each time). A pipeline whose other stages
 * cannot sit beside it (MOSS-Image's DiT on a 12 GB GPU) opens it per generation, encodes the
 * positive and negative prompts, and disposes it before its next stage — Wan's umT5 text stage is
 * the precedent (ADR 0119 decision 11, `src/wan/text-stage.ts`).
 *
 * The GPU belongs to the caller: `dispose` folds the Session and never destroys the device.
 *
 * MUST: encodings go on one serialization chain — two concurrent `encode` calls would share one
 * Session between two runs, and `dispose` would fold the Session without waiting for an in-flight
 * run (a flush-before-destroy violation).
 *
 * ## Cancellation (`signal`)
 *
 * {@link Qwen3_5EncodeOptions.signal} is checked after waiting for the turn on the chain and before
 * the run (after yielding to the event loop once — a single run is indivisible). Cancellation throws
 * `signal.reason` as is.
 *
 * ## MUST: numerical correctness is not guaranteed here
 *
 * It is guaranteed by the real-GPU e2e (`packages/e2e/tests/e2e_qwen3_5_encoder_test.ts` — the GPU
 * rows against Karume's CPU reference execution of the same container within a measured band, plus
 * sha256 rows per environment) and by the bit identity of the host rows with the recipe's
 * (`qwen3_5_session_io_test.ts`).
 *
 * NOTE: internal only (no public subpath yet — the module doc of `./tokenizer.ts`).
 *
 * MUST: modules are side-effect free (CLAUDE.md).
 */

import type { GpuContext, Session, SessionDiagnostics, SessionOptions } from "@karume/runtime";
import type { SessionSpec } from "@karume/hub";

import { settleAbort } from "../concurrency/abort.ts";
import { createOperationChain } from "../concurrency/serial.ts";
import type { GraphOwner } from "../hub/components.ts";
import { type FamilySessionPolicy, resolveSessionOptions } from "../session/options.ts";
import { type Qwen3_5GraphContract, qwen3_5GraphContract } from "./graph-contract.ts";
import { qwen3_5HiddenStates, qwen3_5SessionInputs } from "./session-io.ts";

/** The message prefix (the class name). */
const OWNER = "Qwen3_5TextEncoder";

/**
 * The execution knobs the text encoder accepts from a distribution's `session` declaration: none.
 * The verified form is the i8 storage run in f32 (ADR 0137 decision 5 — the storage check in
 * `./graph-contract.ts`); a knob is accepted only after it is verified against the CPU reference the
 * same way, never passed to the runtime silently.
 *
 * NOTE: the `export` is for the host test that pins "no key accepted" (not on any public surface).
 */
export const QWEN3_5_TEXT_ENCODER_SESSION_POLICY: FamilySessionPolicy = {
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
};

/** The part of a Session the encoder uses (narrow, so a test can stand in for the GPU). */
export type Qwen3_5Session = Pick<Session, "run" | "diagnostics" | "dispose">;

/**
 * What the encoder needs of a component: the graph declaration and the Session entry. A
 * `ModelComponent` a pipeline's admission opened (`src/hub/components.ts`) is one as it is;
 * `prepareContainer`'s `PreparedModel` is one through `createContainerSession`.
 */
export type Qwen3_5TextComponent = GraphOwner & {
  readonly createSession: (gpu: GpuContext, options?: SessionOptions) => Promise<Qwen3_5Session>;
};

/** Options of {@link Qwen3_5TextEncoder.open}. */
export type Qwen3_5TextEncoderOptions = {
  /**
   * The distribution's `session` declaration for this component (default `{}`). It goes through
   * {@link QWEN3_5_TEXT_ENCODER_SESSION_POLICY}, which accepts no key — a declaration fails loudly
   * here instead of reaching the runtime unverified.
   */
  readonly session?: SessionSpec;
  /**
   * An observation hook that receives the diagnostics of every run (one encoding = one run).
   * Exceptions from the callback are not swallowed (the encoding fails with it).
   */
  readonly onRunDiagnostics?: (diagnostics: SessionDiagnostics) => void;
};

/** Options for one call of {@link Qwen3_5TextEncoder.encode}. */
export type Qwen3_5EncodeOptions = {
  /**
   * Cancels the encoding (checked after the turn on the chain, before the run). Throws
   * `signal.reason` as is; the encoder can run the next `encode` afterwards.
   */
  readonly signal?: AbortSignal;
};

/**
 * Qwen3.5 token ids → the text tower's final hidden state (one graph, one Session per instance).
 *
 * The constructor is private: {@link Qwen3_5TextEncoder.open} checks the graph contract before the
 * weights are uploaded, so no instance exists over a graph the host cannot feed.
 */
export class Qwen3_5TextEncoder {
  readonly #session: Qwen3_5Session;
  readonly #contract: Qwen3_5GraphContract;
  readonly #onRunDiagnostics: ((diagnostics: SessionDiagnostics) => void) | undefined;
  /** The serialization chain of encode and dispose (module doc). */
  readonly #chain = createOperationChain();
  /** The single dispose. **Not undefined means "disposed"** (no separate flag to keep in step). */
  #disposal: Promise<void> | undefined;

  private constructor(
    session: Qwen3_5Session,
    contract: Qwen3_5GraphContract,
    onRunDiagnostics: ((diagnostics: SessionDiagnostics) => void) | undefined,
  ) {
    this.#session = session;
    this.#contract = contract;
    this.#onRunDiagnostics = onRunDiagnostics;
  }

  /**
   * Checks the graph contract and the `session` declaration, then builds the Session on `gpu`
   * (uploads the weights — the caller keeps owning `gpu`).
   *
   * MUST: the contract and the declaration are checked before `createSession`, so a container the
   * host cannot feed fails before about 7.4 GiB is uploaded.
   */
  static async open(
    component: Qwen3_5TextComponent,
    gpu: GpuContext,
    options: Qwen3_5TextEncoderOptions = {},
  ): Promise<Qwen3_5TextEncoder> {
    const contract = qwen3_5GraphContract(component, OWNER);
    const sessionOptions = resolveSessionOptions(
      QWEN3_5_TEXT_ENCODER_SESSION_POLICY,
      options.session ?? {},
      {},
      OWNER,
    );
    const session = await component.createSession(gpu, sessionOptions);
    return new Qwen3_5TextEncoder(session, contract, options.onRunDiagnostics);
  }

  /** The width of one output row (4,096 for Qwen3.5-9B — read from the graph declaration). */
  get hiddenSize(): number {
    return this.#contract.hiddenSize;
  }

  /**
   * Encodes one id sequence (length 1 … 4096, every id inside the embedding table) and returns its
   * rows `[ids.length, hiddenSize]` (f32, row-major — one row per token). The ids are checked before
   * waiting for the turn; they are borrowed until the run settles (not copied).
   *
   * Concurrent calls **wait and run in order** (one Session is never shared by two runs).
   */
  async encode(
    ids: Int32Array<ArrayBuffer>,
    options: Qwen3_5EncodeOptions = {},
  ): Promise<Float32Array<ArrayBuffer>> {
    // Checked at call time (checking inside the chain would also fail encodings accepted before
    // dispose).
    if (this.#disposal !== undefined) {
      throw new Error(`${OWNER}: cannot encode after dispose`);
    }
    const { output, hiddenSize, vocabularySize } = this.#contract;
    const inputs = qwen3_5SessionInputs(ids, vocabularySize);
    return await this.#chain(async () => {
      await settleAbort(options.signal);
      const outputs = await this.#session.run(inputs);
      this.#onRunDiagnostics?.(this.#session.diagnostics());
      if (!Object.hasOwn(outputs, output)) {
        throw new Error(`${OWNER}: graph output '${output}' is missing from the run results`);
      }
      const rows = qwen3_5HiddenStates(outputs[output], ids.length, hiddenSize);
      // MUST: never return non-finite rows (O(T·hiddenSize)). Failing only in the next model would
      // point at the wrong stage.
      const broken = rows.findIndex((value) => !Number.isFinite(value));
      if (broken !== -1) {
        throw new Error(
          `${OWNER}: output (${ids.length} tokens) row ${Math.floor(broken / hiddenSize)}, column ` +
            `${broken % hiddenSize} is non-finite (${rows[broken]})`,
        );
      }
      return rows;
    });
  }

  /**
   * Folds the Session (never the caller's GPU). Waits for an in-flight encoding first
   * (flush-before-destroy — the fold is queued on the same chain). Later calls return the same
   * completion.
   */
  dispose(): Promise<void> {
    this.#disposal ??= this.#chain(() => this.#session.dispose());
    return this.#disposal;
  }

  /** `await using` support (Explicit Resource Management) — an alias of {@link dispose}. */
  [Symbol.asyncDispose](): Promise<void> {
    return this.dispose();
  }
}
