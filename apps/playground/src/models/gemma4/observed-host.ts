/**
 * A `ChatSessionHost` that passes everything through to the pipeline and observes the
 * sequences the session allocates: the prompt each turn sends (its token count, and how many
 * positions the carried KV already held), and how many positions the held sequence occupies between
 * turns.
 *
 * The session keeps its sequence private; the host type is the documented seam ("the session uses
 * nothing inside the public surface"), so wrapping it is how a consumer reads these figures without
 * re-deriving the session's rendering rules. The wrapper has the shape of the pipeline's own
 * hand-out wrapper: `used` stays a getter (it is a derived value), everything else delegates.
 */

import type { ChatSessionHost, GenerationSequence } from "@karume/models/gemma";

/** What one `generate` call of the session sends. */
export type GenerateObservation = {
  /** The prompt tokens the session passed (the delta when the KV is carried, else the whole history). */
  readonly promptTokens: number;
  /** Positions the sequence already held (0 = a fresh sequence: the whole history is prefilled). */
  readonly usedBefore: number;
  readonly capacity: number;
};

export type ObservedHost = {
  /** Pass this to `new ChatSession(...)` in place of the pipeline. */
  readonly host: ChatSessionHost;
  /**
   * Positions the session's sequence occupies now; `undefined` while it holds none (before the first
   * turn, and after a turn that did not close with EOS — its next send re-renders the history).
   */
  readonly kvUsed: () => number | undefined;
};

export const observeHost = (
  inner: ChatSessionHost,
  onGenerate: (observation: GenerateObservation) => void,
): ObservedHost => {
  let held: GenerationSequence | undefined;
  const host: ChatSessionHost = {
    get tokenizer() {
      return inner.tokenizer;
    },
    get program() {
      return inner.program;
    },
    get defaultSampler() {
      return inner.defaultSampler;
    },
    // MUST: forwarded, or the session behind this wrapper refuses image parts for a pipeline that
    // can take them.
    get vision() {
      return inner.vision;
    },
    embedImage: (image) => {
      if (inner.embedImage === undefined) throw new Error("This host has no image embedder");
      return inner.embedImage(image);
    },
    sequence: async (options) => {
      const sequence = await inner.sequence(options);
      const observed: GenerationSequence = {
        capacity: sequence.capacity,
        get used(): number {
          return sequence.used;
        },
        generate: (request) => {
          onGenerate({
            promptTokens: request.prompt.length,
            usedBefore: sequence.used,
            capacity: sequence.capacity,
          });
          return sequence.generate(request);
        },
        fork: () => sequence.fork(),
        dispose: async () => {
          // Cleared before awaiting: from here on the session holds no sequence.
          if (held === observed) held = undefined;
          await sequence.dispose();
        },
      };
      held = observed;
      return observed;
    },
  };
  return { host, kvUsed: () => held?.used };
};
