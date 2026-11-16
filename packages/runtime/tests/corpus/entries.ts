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

/** Every entry has `exact`: the perturbed weights and the distances use it. */
const EXACT: CorpusSetting = {
  name: "exact",
  kind: "level",
  options: { numerics: { level: "exact" } },
};

/** The two levels alone. */
const LEVEL_SETTINGS: readonly CorpusSetting[] = [
  EXACT,
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
    /** The module path segment the layers are numbered by (`<segment>.<n>.` in initializer names). */
    readonly segment: string;
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
      segment: "layers",
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
      segment: "layers",
      layers: [13, 14, 34],
    },
  },
  {
    // Irodori v4-small, `dit`: no state slots, int8-sym linear weights, attention built from bmm and
    // safe_softmax. Blocks 0 and 11 of 12: in block 0 one of the two ropes is not fused (it
    // dispatches a neg the other blocks do not), 11 computes the output, and block 0 hands its
    // residual to block 11. Its presets are `i8` (exact) and `i8-a8`; `reorder` dispatches the
    // same kernels as `exact` here (no strategy fits its linears, and it has no attention op).
    name: "irodori-v4-small-dit-cut2",
    graph: "dit",
    seed: 1,
    plan: {
      rowSymbol: "S",
      bindings: {},
      runs: [
        // One run of two independent steps (the graph has no state slots). 48 rows: the GEMVs of
        // the linears (M ≤ 64) and the bmm without v4 (48 + 1519 keys is not a multiple of 4).
        // 513 rows: the 128×128 GEMM of exact, the 128×64 i8a8 tile of a8 (M > 512) and the bmm
        // with v4.
        { name: "s48-s513", steps: [48, 513] },
      ],
    },
    settings: [
      EXACT,
      {
        name: "a8",
        kind: "approximation",
        options: { numerics: { level: "exact" }, linearCompute: "a8" },
      },
    ],
    // The output projection: element 0 reaches every row of output column 0.
    perturbed: "out_proj.weight",
    source: {
      mirror: "models/karume-irodori-v4-small/v4-small/dit/model.i8.krm",
      graph: "dit",
      segment: "blocks",
      layers: [0, 11],
    },
  },
  {
    // Wan2.2 TI2V-5B, `transformer`: no state slots, int8-sym linear weights, self- and
    // cross-attention through the `attention` op. Blocks 28 and 29 of 30 (29 computes the output).
    // Its presets add int8 attention with the scores stored in f16 or not stored at all (the
    // online kernel); `reorder` dispatches the same kernels as `exact` here.
    name: "wan2.2-ti2v-5b-transformer-cut2",
    graph: "transformer",
    seed: 1,
    plan: {
      rowSymbol: "S",
      bindings: {},
      runs: [
        // One run of two independent steps (the graph has no state slots). 16 rows: the GEMVs of
        // the linears and the attention kernels with v4 (S a multiple of 4, so the scores are
        // stored in f16). 513 rows: the 128×128 GEMM of exact, the 128×64 i8a8 tile (M > 512),
        // attention without v4 (scores in f32) and the online kernel with padded keys.
        // One run, not two: on a device whose workgroup storage cannot hold the online kernel, the
        // online setting falls back to stored f32 scores, which is what the f16 setting runs at
        // 513 rows too — the two settings then differ only at 16 rows, and "no two settings
        // dispatch the same kernels" is judged over a run.
        { name: "s16-s513", steps: [16, 513] },
      ],
    },
    settings: [
      EXACT,
      {
        name: "a8-attn8-s16",
        kind: "approximation",
        options: {
          numerics: { level: "exact" },
          linearCompute: "a8",
          attentionCompute: "a8",
          attentionScoreStorage: "f16",
        },
      },
      {
        name: "a8-attn8-online",
        kind: "approximation",
        options: {
          numerics: { level: "exact" },
          linearCompute: "a8",
          attentionCompute: "a8",
          attentionScoreStorage: "none",
        },
      },
    ],
    // The output projection: element 0 reaches every row of output column 0.
    perturbed: "proj_out.weight",
    source: {
      mirror: "models/karume-wan2.2/ti2v-5b/transformer/model.i8.krm",
      graph: "transformer",
      segment: "blocks",
      layers: [28, 29],
    },
  },
];
