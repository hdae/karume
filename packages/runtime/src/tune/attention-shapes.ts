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

const MOSS_IMAGE_HEADS = 24;
const MOSS_IMAGE_DEPTH = 128;
/** Image tokens of a 1024² MOSS-Image request: 64 × 64 latent cells of 16 px. */
const MOSS_IMAGE_TOKENS_1024 = 64 * 64;
/** The text tokens of the fridge-note prompt (the request of the family's 1024² sha rows). */
const MOSS_IMAGE_FRIDGE_NOTE_TOKENS = 63;

/**
 * MOSS-Image's attention at the default preset's 1024² request (`f32+dit8-a8-attn8-online`, the
 * online route): the joint self-attention of `dit` over S = 4,096 image tokens + 63 text tokens =
 * 4,159 (N % 4 = 3 — the padded form where the profile chooses none), 24 heads of D = 128 (the
 * `dit` graph's 42 attention nodes are all `[1, 24, S, 128]` with the half scale of D = 128 — the
 * distribution's graph declaration; `assertMossImageSize` / `mossImageTokens` of
 * packages/models/src/moss give the image tokens, the session A/B of e2e_moss_session_ab_test.ts the
 * text tokens). `dit_skip22`'s negative branch (S = 4,106 there) is the same kind of shape and is
 * not measured separately.
 */
export const MOSS_IMAGE_ATTENTION_SHAPE: AttentionBenchShape = {
  id: "moss-image-1024",
  heads: MOSS_IMAGE_HEADS,
  queries: MOSS_IMAGE_TOKENS_1024 + MOSS_IMAGE_FRIDGE_NOTE_TOKENS,
  keys: MOSS_IMAGE_TOKENS_1024 + MOSS_IMAGE_FRIDGE_NOTE_TOKENS,
  depth: MOSS_IMAGE_DEPTH,
  scale: halfScale(MOSS_IMAGE_DEPTH),
};

/**
 * The bench's default shapes and the set an attention-bench record given to the profile derivation
 * must cover (`deriveGeometryProfile`'s `attentionBench` — ADR 0117 addendum 2026-11-12): Wan2.2's
 * three and MOSS-Image's one, the online-route shapes of the default presets the derivation's field
 * applies to.
 * MUST: a change here changes `attentionBenchShapeSetId`, so a saved table whose `attentionOnline`
 * was decided on another set mismatches (`geometryProfileMismatch`) and records of another set are
 * refused.
 */
export const ATTENTION_BENCH_SHAPES: readonly AttentionBenchShape[] = [
  ...WAN22_ATTENTION_SHAPES,
  MOSS_IMAGE_ATTENTION_SHAPE,
];
