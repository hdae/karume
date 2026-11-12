/**
 * The attention bench's shapes (`runAttentionRouteBench` of `@karume/runtime/tune`) and the shape
 * set the profile derivation requires of attention-bench records (`ATTENTION_BENCH_SHAPES` — ADR
 * 0117 addendum 2026-11-12): pure data, read by the bench, the derivation, the kernel fingerprint and
 * matching (`attentionBenchShapeSetId` of fingerprint.ts).
 *
 * MUST: every module is side-effect free (functions and frozen-by-convention constants only).
 */

/**
 * One attention shape (batch 1 — `heads` is the folded `B·H` axis). `queries` is the node's full
 * query count M (it decides the production split); `rowsPerDispatch` overrides the rows one measured
 * dispatch covers (default: the first row block production would run for the route).
 */
export type AttentionBenchShape = {
  /** A name for the record (free text). */
  readonly id: string;
  readonly heads: number;
  readonly queries: number;
  readonly keys: number;
  readonly depth: number;
  /** The op's half scale (`√scale_factor`, applied to both q and k — ADR 0023 decision 3). */
  readonly scale: number;
  readonly rowsPerDispatch?: number;
};

/** The half scale the exporter writes for SDPA's default scale `1/√D` (`f32(D^-1/4)`). */
const halfScale = (depth: number): number => Math.fround(Math.sqrt(1 / Math.sqrt(depth)));

const WAN22_HEADS = 24;
const WAN22_DEPTH = 128;
/** Latent tokens of Wan2.2 TI2V-5B at 1280×704: 44 × 80 / (2 × 2) patches per latent frame. */
const WAN22_TOKENS_PER_LATENT_FRAME = 880;

/**
 * Wan2.2 TI2V-5B's attention shapes (24 heads, D = 128): self-attention at 33 frames (9 latent
 * frames, N = 7,920) and 121 frames (31 latent frames, N = 27,280), and cross-attention over the 512
 * text tokens at 121 frames.
 */
export const WAN22_ATTENTION_SHAPES: readonly AttentionBenchShape[] = [
  {
    id: "wan22-self-33f",
    heads: WAN22_HEADS,
    queries: 9 * WAN22_TOKENS_PER_LATENT_FRAME,
    keys: 9 * WAN22_TOKENS_PER_LATENT_FRAME,
    depth: WAN22_DEPTH,
    scale: halfScale(WAN22_DEPTH),
  },
  {
    id: "wan22-self-121f",
    heads: WAN22_HEADS,
    queries: 31 * WAN22_TOKENS_PER_LATENT_FRAME,
    keys: 31 * WAN22_TOKENS_PER_LATENT_FRAME,
    depth: WAN22_DEPTH,
    scale: halfScale(WAN22_DEPTH),
  },
  {
    id: "wan22-cross-121f",
    heads: WAN22_HEADS,
    queries: 31 * WAN22_TOKENS_PER_LATENT_FRAME,
    keys: 512,
    depth: WAN22_DEPTH,
    scale: halfScale(WAN22_DEPTH),
  },
];

/**
 * The bench's default shapes and the set an attention-bench record given to the profile derivation
 * must cover (`deriveGeometryProfile`'s `attentionBench` — ADR 0117 addendum 2026-11-12): every shape
 * a field the derivation decides from these records applies to on a default preset.
 * MUST: a change here changes `attentionBenchShapeSetId`, so a saved table whose `attentionOnline`
 * was decided on another set mismatches (`geometryProfileMismatch`) and records of another set are
 * refused.
 */
export const ATTENTION_BENCH_SHAPES: readonly AttentionBenchShape[] = [...WAN22_ATTENTION_SHAPES];
