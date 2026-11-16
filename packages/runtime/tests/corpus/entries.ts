/**
 * The regression corpus's entries: the graph of a real model, cut to a few layers, run on generated
 * weights by a driver that knows only the declaration.
 *
 * An entry's fixture is a graph container (`<name>.krg` — the declaration and its const region)
 * and the storage table (`<name>.storage.json` — the codec the distribution stores each model-side
 * weight in, which a `krg` does not carry). `author.ts` writes both from a local distribution
 * mirror; no test reads the mirror.
 *
 * SPEC: docs/spec/verification.md#an-entry
 */

import type { CorpusPlan } from "./driver.ts";

export type CorpusEntry = {
  readonly name: string;
  /** The graph's name inside the graph container. */
  readonly graph: string;
  /** Seed of the generated weights and inputs. */
  readonly seed: number;
  readonly plan: CorpusPlan;
  /** A quantized weight whose element 0 moves one level, to show the comparisons can fail. */
  readonly perturbed: string;
  /** Where `author.ts` cuts the entry from (relative to the repository root; read-only). */
  readonly source: {
    readonly mirror: string;
    readonly graph: string;
    readonly layers: readonly number[];
  };
};

/** Where an entry's fixture files live. */
export const fixtureUrls = (
  entry: Pick<CorpusEntry, "name">,
): { readonly graph: URL; readonly storage: URL } => ({
  graph: new URL(`../fixtures/corpus/${entry.name}.krg`, import.meta.url),
  storage: new URL(`../fixtures/corpus/${entry.name}.storage.json`, import.meta.url),
});

export const CORPUS_ENTRIES: readonly CorpusEntry[] = [
  {
    // Gemma 4 QAT E2B, one layer of each of its four kinds: 13 (sliding attention with its own
    // key/value states), 14 (full attention with its own states), 15 (sliding, reading layer 13's
    // states) and 34 (full, reading layer 14's states, and the exit). Weights int4-sym-g, group 32.
    name: "gemma4-qat-e2b-cut4",
    graph: "model",
    seed: 1,
    plan: {
      rowSymbol: "M",
      bindings: { R: 1 },
      capacitySymbol: "C",
      runs: [
        // At 8 rows a `reorder` prefill takes the parallel GEMV; at 64 both levels take the batched
        // GEMV. Each prefill is followed by four single-row decode steps.
        { name: "prefill8", steps: [8, 1, 1, 1, 1], capacity: 16 },
        { name: "prefill64", steps: [64, 1, 1, 1, 1], capacity: 80 },
      ],
    },
    perturbed: "model.model.layers.13.mlp.down_proj.weight",
    source: {
      mirror: "models/karume-gemma4-qat/e2b/model/model.i4.krm",
      graph: "model",
      layers: [13, 14, 15, 34],
    },
  },
];
