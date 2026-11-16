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

import type { SessionOptions } from "../../src/runtime/executor.ts";
import type { CorpusPlan } from "./driver.ts";

/**
 * One Session setting an entry is checked under.
 *
 * `kind` is what the setting promises against another plan of the same graph, stated by the entry
 * (never derived from the options): `"level"` — a level alone, no named approximation — promises
 * the bits of its unfused plan, so the fusion identity is checked under it; `"approximation"` names
 * an approximation and promises nothing against another plan.
 *
 * SPEC: docs/spec/numerics.md#fusions
 */
export type CorpusSetting = {
  /** The setting's name in row ids and messages (unique within the entry). */
  readonly name: string;
  readonly kind: "level" | "approximation";
  readonly options: SessionOptions;
};

/** The two levels alone. Every entry has `exact`: the perturbed weights and the distances use it. */
const LEVEL_SETTINGS: readonly CorpusSetting[] = [
  { name: "exact", kind: "level", options: { numerics: { level: "exact" } } },
  { name: "reorder", kind: "level", options: { numerics: { level: "reorder" } } },
];

export type CorpusEntry = {
  readonly name: string;
  /** The graph's name inside the graph container. */
  readonly graph: string;
  /** Seed of the generated weights and inputs. */
  readonly seed: number;
  readonly plan: CorpusPlan;
  /** The Session settings the entry is checked under. */
  readonly settings: readonly CorpusSetting[];
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
    settings: LEVEL_SETTINGS,
    perturbed: "model.model.layers.13.mlp.down_proj.weight",
    source: {
      mirror: "models/karume-gemma4-qat/e2b/model/model.i4.krm",
      graph: "model",
      layers: [13, 14, 15, 34],
    },
  },
  {
    // Gemma 4 QAT E2B, mobile line: mixed codecs per layer (int4-sym-g attention and own-KV MLPs at
    // groups 512 / 2048 / 4096, int2-off MLPs in the KV-shared layers and the tied table, int8-sym
    // per-layer-input linears) and a static_quantize before and after every linear. Its presets are
    // `exact`, `reorder` and `reorder` with packed activations; the integer GEMV is the Session
    // option on top of those.
    name: "gemma4-qat-mobile-e2b-cut3",
    graph: "model",
    seed: 1,
    plan: {
      rowSymbol: "M",
      bindings: { R: 1 },
      capacitySymbol: "C",
      runs: [
        // At 8 rows the parallel, packed and integer GEMVs take the prefill (their range is
        // M = 1..8); at 64 every setting takes the batched GEMV. Four single-row decode steps follow.
        { name: "prefill8", steps: [8, 1, 1, 1, 1], capacity: 16 },
        { name: "prefill64", steps: [64, 1, 1, 1, 1], capacity: 80 },
      ],
    },
    settings: [
      ...LEVEL_SETTINGS,
      {
        name: "packed",
        kind: "approximation",
        options: { numerics: { level: "reorder" }, packedStaticQuantize: true },
      },
      {
        name: "integer",
        kind: "approximation",
        options: { numerics: { level: "reorder" }, packedStaticQuantize: true, integerGemv: true },
      },
    ],
    // The tied table (int2-off): its logits reach the outputs through no static_quantize. A level of
    // a layer weight moves its linear's output by less than the quantization step that follows, and
    // the outputs read one row per step, so in this graph it reaches them only by chance (layer 13's
    // down_proj reaches them in prefill8 and not in prefill64).
    perturbed: "model.lm_head.weight",
    source: {
      mirror: "models/karume-gemma4-qat-mobile/e2b/model/model.i4.krm",
      graph: "model",
      layers: [13, 14, 34],
    },
  },
];
