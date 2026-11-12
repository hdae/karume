/**
 * The **shape table** measured by the tile-geometry sweep (perf-ledger K-70) (pure data).
 *
 * The source is anima's op census (`scenarios[0].weights[]` of
 * `outputs/bench/karume-anima/2026-09-04_op-census/summary.json` — 1024px generation, patch tokens
 * S = 4096, quant `f16+dit8-a8-attn8-s16`). The values are **copied** from the census, and the
 * census is not read at run time (the tool does not depend on the census being present or on its
 * format — the shapes do not move unless anima's IR changes).
 *
 * `censusCount` is the census's `count` (the number of nodes in one graph = one forward of each
 * component). Rows with M = 1024 (512²) are not in the census — they replace the M of the 1024px
 * row (S = 4096) with the 512px S = 1024, and carry the same count because the node count does not
 * depend on the resolution. linear's M = 16 / 32 / 128 / 256 and bmm's M = 16 / 32 / 128 / 256 are
 * likewise derived by replacing the M of a census row (M values absent from the census): controls
 * that measure the upper end of each bucket of the geometry profile's `gemmRows`
 * ({@link PROFILE_GEMM_ROWS_BOUNDS}) on the three paths linear / matmul / bmm (ADR 0116 decisions 1
 * / 4).
 *
 * rank-2 matmul is not in the census (no family's census, anima included, has a row for it). The
 * matmul cases are **counterparts** of one linear case per bucket (same M / N / K, B side `[K,N]`);
 * `censusCount: 0` and `mirrorOf` (the case id of the linear case it mirrors) mark that they do not
 * come from the census.
 *
 * The conv3d w8a8 cases have another source: the VAE decoder chunk graphs of Wan2.2 TI2V-5B at
 * 1280×704 (the containers `vae_decoder_first` / `vae_decoder_next` at tile 16 and
 * `vae_decoder_first_32` / `vae_decoder_next_32` at tile 32 — ADR 0134), read from the graphs'
 * static shapes. They too are copied and not read at run time ({@link CONV3D_I8A8_CASES}).
 *
 * The i8a8 linear has two more sources besides anima's DiT (ADR 0115 addendum decision 14): the
 * Wan2.2 TI2V-5B DiT at 33 frames ({@link WAN22_DIT_LINEAR}) and a guard of the Irodori v4-small
 * DiT at small M with an edge case at the upper end of the bucket ≤ 256
 * ({@link IRODORI_I8A8_LINEAR_CASES}) — the shapes the families running the i8a8 linear by default
 * reach, at the row counts the field's buckets ({@link I8A8_LINEAR_ROWS_BOUNDS}) split.
 *
 * The evaluation sets `k85` and `k88` (perf-ledger K-85 / K-88) have tables of their own
 * ({@link K85_CASES}, {@link K88_CASES}), outside {@link SWEEP_CASES}: the case-set version and the
 * kernel fingerprints of the table-deriving sets do not see them ({@link sweepCasesFor}).
 */

import type { Conv3dDims } from "../kernels/conv3d.ts";
import { type CandidateSet, DEFAULT_CANDIDATE_SET } from "./geometries.ts";

/**
 * The op families of the sweep (the vocabulary of the CLI's `--op` and the page's checkboxes — both
 * removed in 0.15).
 */
export type SweepOp =
  | "linear"
  | "matmul"
  | "bmm"
  | "i8a8-linear"
  | "attention"
  | "i8a8-attention"
  | "conv2d"
  | "i8a8-conv3d";

export const SWEEP_OPS: readonly SweepOp[] = [
  "linear",
  "matmul",
  "bmm",
  "i8a8-linear",
  "attention",
  "i8a8-attention",
  "conv2d",
  "i8a8-conv3d",
];

/**
 * linear (`x[M,K] · Wᵀ[K,N] + b[N]`). `linear` is f16-stored weights × f32 compute (the same `wf16`
 * as the DiT and the text components of quant `f16`); `i8a8-linear` is the integer dot product of
 * i8 weights (per-channel scale) × per-token i8 activations.
 */
export type LinearCase = {
  readonly id: string;
  readonly op: "linear" | "i8a8-linear";
  readonly m: number;
  readonly n: number;
  readonly k: number;
  /**
   * The weight storage of a `linear` case (absent = `f16`, the storage of the census). The
   * `k88` cases carry the storage of the model they are copied from; `i4` needs
   * `groupSize`. An `i8a8-linear` case is always i8 and has no such field.
   */
  readonly weight?: CaseWeight;
  /** The i4 group length (with `weight: "i4"` only). */
  readonly groupSize?: number;
  readonly censusCount: number;
  /** What `censusCount` counts (the `k88` cases — {@link K88_CENSUS}). */
  readonly censusGroup?: string;
  /** Which census row (the component and its input shape in the census). */
  readonly source: string;
};

/**
 * The weight storages of the f32-compute cases: `f16` is the census's, the others those of the
 * models the `k88` cases are copied from (i4 at a group length — {@link LinearCase.groupSize}).
 */
export type CaseWeight = "f32" | "f16" | "i8" | "i4";

/**
 * matmul (`a[M,K] · b[K,N]`, f32 × f32). The same GEMM skeleton as linear; the only differences are
 * the fill that reads the B side as a dense `[K,N]` and the absence of bias. The census has no row
 * for it, so counterparts of the linear cases are measured.
 */
export type MatmulCase = {
  readonly id: string;
  readonly op: "matmul";
  readonly m: number;
  readonly n: number;
  readonly k: number;
  /** The census has no matmul row, so always 0. */
  readonly censusCount: 0;
  /** The case id of the mirrored linear case (same M / N / K). */
  readonly mirrorOf: string;
  readonly source: string;
};

/**
 * bmm (`a[B,M,K] · b[B,K,N]`, f32 × f32). The batch is the dispatch's z axis, and the tile-geometry
 * bucket is the M of one matrix (buildBmm in src/runtime/recipe-builders/linear.ts).
 */
export type BmmCase = {
  readonly id: string;
  readonly op: "bmm";
  readonly batch: number;
  readonly m: number;
  readonly n: number;
  readonly k: number;
  readonly censusCount: number;
  readonly source: string;
};

/**
 * Fused attention's ①QK / ③PV (B·H = 16, D = 128). `attention` is f32 compute with S stored in f32;
 * `i8a8-attention` is the integer dot product with S stored in f16 (s16 — same as quant
 * `…-attn8-s16`). ③PV takes as input the S and row statistics from one run of ①QK → ②row
 * statistics of the same shape with the default geometry.
 */
export type AttentionCase = {
  readonly id: string;
  readonly op: "attention" | "i8a8-attention";
  readonly stage: "qk" | "pv";
  /** B·H (the count folded into the batch axis). */
  readonly batchHeads: number;
  /** Query length M. */
  readonly m: number;
  /** Key length N. */
  readonly n: number;
  /** Head dimension D. */
  readonly d: number;
  /** The `scale` of attrs (half scale — applied to both q and k). */
  readonly scale: number;
  readonly score: "f32" | "f16";
  readonly censusCount: number;
  readonly source: string;
};

/**
 * conv2d's implicit GEMM (groups = 1, 3×3, stride 1, padding 1, dilation 1, f16-stored weights and
 * B = 1 unless the case says otherwise). `M = Cout`, `N = H·W`, `K = Cin·9`; the batch is the
 * dispatch's z axis.
 */
export type Conv2dCase = {
  readonly id: string;
  readonly op: "conv2d";
  /** The batch (absent = 1 — the `k88` Wan2.2 VAE cases fold the chunk's frames into it). */
  readonly batch?: number;
  readonly channelsIn: number;
  readonly channelsOut: number;
  readonly height: number;
  readonly width: number;
  /** The weight storage (absent = `f16`, the census's). */
  readonly weight?: CaseWeight;
  readonly censusCount: number;
  /** What `censusCount` counts (the `k88` cases — {@link K88_CENSUS}). */
  readonly censusGroup?: string;
  readonly source: string;
};

/**
 * conv3d's w8a8 implicit GEMM (the ② dispatch of the `convCompute: "a8"` path — ADR 0133): i8
 * tap-major weights × per-pixel i8 activations, `M = Cout`, `N = Tout·Hout·Wout`, reduced over Cin
 * per tap. `dims` are the kernel's dims as the recipe builder passes them (B = 1).
 */
export type Conv3dI8a8Case = {
  readonly id: string;
  readonly op: "i8a8-conv3d";
  readonly dims: Conv3dDims;
  /** The node count of the shape in the chunk graph named by `source`. */
  readonly censusCount: number;
  readonly source: string;
};

export type SweepCase =
  | LinearCase
  | MatmulCase
  | BmmCase
  | AttentionCase
  | Conv2dCase
  | Conv3dI8a8Case;

/**
 * The bucket boundaries of the geometry profile's `gemmRows` (the `maxRows` column, ADR 0116
 * decision 1). The only constant by which the generator cuts buckets; the entries other than the
 * last MUST equal the set of M (≤ 512) of the linear / matmul / bmm sweep cases (tune_cases_test
 * enforces it — so that no bucket is created whose upper end is not measured). It MUST be a
 * refinement that contains the default table's boundaries 64 / 512 (ADR 0116 decision 3 — so that
 * the geometry of the sweep's default row does not split within a bucket).
 */
export const PROFILE_GEMM_ROWS_BOUNDS: readonly number[] = [
  16,
  32,
  64,
  128,
  256,
  512,
  Number.POSITIVE_INFINITY,
];

/**
 * The bucket boundaries of the geometry profile's `i8a8.linear` (the `maxRows` column — ADR 0115
 * addendum decision 14), the `i8a8-linear` family's bounds (they enter that family's case-set
 * version and no other). The generator writes one rule per bucket for every table. The entries
 * other than the last MUST each equal the M of an `i8a8-linear` sweep case (tune_cases_test
 * enforces it — a bucket's upper end is measured, ADR 0116).
 */
export const I8A8_LINEAR_ROWS_BOUNDS: readonly number[] = [256, 512, Number.POSITIVE_INFINITY];

/** The census's attention `attrs.scale` (common to the DiT's self / cross — `128^-0.25`). */
const DIT_ATTENTION_SCALE = 0.2973017692565918;

/**
 * The DiT's three linear shapes (census: transformer's `[1,4096,K] × [N,K]`, count 168 / 28 / 28).
 */
const DIT_LINEAR: readonly {
  readonly n: number;
  readonly k: number;
  readonly count: number;
}[] = [
  { n: 2048, k: 2048, count: 168 },
  { n: 8192, k: 2048, count: 28 },
  { n: 2048, k: 8192, count: 28 },
];

/** The DiT's M (512² = 1024, 1024² = 4096). */
const DIT_ROWS = [1024, 4096] as const;

const linearId = (op: LinearCase["op"], m: number, n: number, k: number): string =>
  `${op}-m${m}-n${n}-k${k}`;

const ditLinear = (op: LinearCase["op"]): LinearCase[] =>
  DIT_ROWS.flatMap((m) =>
    DIT_LINEAR.map(({ n, k, count }) => ({
      id: linearId(op, m, n, k),
      op,
      m,
      n,
      k,
      censusCount: count,
      source: `transformer [1,${m},${k}] × [${n},${k}]${
        m === 4096 ? "" : " (census M 4096 replaced with 1024)"
      }`,
    })),
  );

/** The source of the Wan2.2 DiT's i8a8 linear cases (the forward the census counts are taken from). */
const WAN22_DIT = "Wan2.2 TI2V-5B transformer (33 frames at 1280×704, S = 7,920)";

/**
 * The Wan2.2 TI2V-5B DiT's linears (the `transformer` graph of the distribution,
 * `models/karume-wan2.2/ti2v-5b/transformer`, its `linear` nodes by input and weight shape) at
 * `S = 7,920` (33 frames at 1280×704), all i8 weights. Copied, not read at run time.
 *
 * `count` is the node count **in one Wan2.2 33f DiT forward**: self-attention q / k / v / o and
 * cross-attention q / o 180 (30 blocks × 6), the FFN up and down 30 each, the cross-attention k / v
 * at the 512 text tokens 60 plus the text embedder's second layer 1, and the time projection at
 * M = 1 2 (one per timestep input). Those are 303 of the graph's 310 linears; the 7 left out are the
 * patch embedding, the output projection and the text embedder's first layer (1 node each) and the
 * timestep embedder's two layers at M = 1 (2 each).
 */
const WAN22_DIT_LINEAR: readonly {
  readonly m: number;
  readonly n: number;
  readonly k: number;
  readonly count: number;
  readonly what: string;
}[] = [
  { m: 7920, n: 3072, k: 3072, count: 180, what: "self q / k / v / o and cross q / o" },
  { m: 7920, n: 14336, k: 3072, count: 30, what: "FFN up" },
  { m: 7920, n: 3072, k: 14336, count: 30, what: "FFN down" },
  { m: 512, n: 3072, k: 3072, count: 61, what: "cross k / v and the text embedder's second layer" },
  { m: 1, n: 18432, k: 3072, count: 2, what: "time projection" },
];

/** The Wan2.2 DiT's i8a8 linear cases ({@link WAN22_DIT_LINEAR}, in its order). */
const WAN22_I8A8_LINEAR_CASES: readonly LinearCase[] = WAN22_DIT_LINEAR.map(
  ({ m, n, k, count, what }): LinearCase => ({
    id: linearId("i8a8-linear", m, n, k),
    op: "i8a8-linear",
    m,
    n,
    k,
    censusCount: count,
    source: `${WAN22_DIT} [1,${m},${k}] × [${n},${k}] (${what})`,
  }),
);

/**
 * The Irodori v4-small DiT's linear `[1,185,1280] × [1280,1280]` (the `dit` graph, 60 nodes per DiT
 * step at M = S = 185 — the M of the research's regression measurement,
 * docs/research/2026-10-25-speed-recon-3080ti.md §5): a guard for the families whose a8 linears run
 * at small M (Irodori's default preset `i8-a8` reads the same `i8a8.linear` field), and the same
 * shape at M = 256, the upper end of the bucket ≤ 256 ({@link I8A8_LINEAR_ROWS_BOUNDS}). Count 0:
 * neither is in the Wan2.2 forward the other counts weigh.
 */
const IRODORI_I8A8_LINEAR_CASES: readonly LinearCase[] = [185, 256].map((m): LinearCase => ({
  id: linearId("i8a8-linear", m, 1280, 1280),
  op: "i8a8-linear",
  m,
  n: 1280,
  k: 1280,
  censusCount: 0,
  source: `Irodori v4-small dit [1,${m},1280] × [1280,1280] (${
    m === 185 ? "60 nodes per DiT step" : "the guard's M 185 replaced with 256"
  }; not in the Wan2.2 forward)`,
}));

const LINEAR_CASES: readonly LinearCase[] = [
  ...ditLinear("linear"),
  {
    // Medium-M control (census: the k / v projection of transformer's cross-attention
    // `[1,512,1024] × [2048,1024]`)
    id: linearId("linear", 512, 2048, 1024),
    op: "linear",
    m: 512,
    n: 2048,
    k: 1024,
    censusCount: 56,
    source: "transformer [1,512,1024] × [2048,1024]",
  },
  ...[128, 256].map((m): LinearCase => ({
    // Controls for the upper ends of buckets 65–128 / 129–256 (the M of the M 512 row above
    // replaced)
    id: linearId("linear", m, 2048, 1024),
    op: "linear",
    m,
    n: 2048,
    k: 1024,
    censusCount: 56,
    source: `transformer [1,${m},1024] × [2048,1024] (census M 512 replaced with ${m})`,
  })),
  {
    // Small-M control (census: text_encoder's `[1,64,1024] × [3072,1024]`, stored in f16)
    id: linearId("linear", 64, 3072, 1024),
    op: "linear",
    m: 64,
    n: 3072,
    k: 1024,
    censusCount: 56,
    source: "text_encoder [1,64,1024] × [3072,1024]",
  },
  ...[16, 32].map((m): LinearCase => ({
    // Controls for the upper ends of buckets ≤ 16 / 17–32 (the M of the M 64 row above replaced)
    id: linearId("linear", m, 3072, 1024),
    op: "linear",
    m,
    n: 3072,
    k: 1024,
    censusCount: 56,
    source: `text_encoder [1,${m},1024] × [3072,1024] (census M 64 replaced with ${m})`,
  })),
  ...ditLinear("i8a8-linear"),
  ...WAN22_I8A8_LINEAR_CASES,
  ...IRODORI_I8A8_LINEAR_CASES,
];

/**
 * Seven matmul counterparts (one linear case from each of the seven buckets of the geometry
 * profile's `gemmRows` — same M / N / K). The census has no matmul row, but since the table
 * (gemmRows) applies to matmul too, these are observations that keep a geometry slow on that
 * skeleton from being adopted. The four with M 16 / 32 / 128 / 256 mirror linear cases that are
 * themselves derived by replacing the census M (ADR 0116 decision 4).
 */
const MATMUL_CASES: readonly MatmulCase[] = [
  { m: 64, n: 3072, k: 1024 },
  { m: 512, n: 2048, k: 1024 },
  { m: 4096, n: 2048, k: 2048 },
  { m: 16, n: 3072, k: 1024 },
  { m: 32, n: 3072, k: 1024 },
  { m: 128, n: 2048, k: 1024 },
  { m: 256, n: 2048, k: 1024 },
].map(({ m, n, k }) => {
  const mirrorOf = linearId("linear", m, n, k);
  return {
    id: `matmul-m${m}-n${n}-k${k}`,
    op: "matmul" as const,
    m,
    n,
    k,
    censusCount: 0 as const,
    mirrorOf,
    source: `mirror of ${mirrorOf} [${m},${k}] × [${k},${n}] (the census has no matmul row)`,
  };
});

/**
 * Five bmm cases (census: two shapes of text_encoder, three of text_conditioner. The shapes are
 * `[B,M,K] × [B,K,N]`, the census in_shapes as is).
 */
const BMM_CASES: readonly BmmCase[] = [
  { component: "text_encoder", batch: 16, m: 64, k: 64, n: 128, count: 28 },
  { component: "text_encoder", batch: 16, m: 64, k: 128, n: 64, count: 28 },
  { component: "text_conditioner", batch: 16, m: 512, k: 64, n: 64, count: 12 },
  { component: "text_conditioner", batch: 16, m: 512, k: 64, n: 512, count: 6 },
  { component: "text_conditioner", batch: 16, m: 512, k: 512, n: 64, count: 6 },
].map(({ component, batch, m, k, n, count }) => ({
  id: `bmm-b${batch}-m${m}-n${n}-k${k}`,
  op: "bmm" as const,
  batch,
  m,
  n,
  k,
  censusCount: count,
  source: `${component} bmm [${batch},${m},${k}] × [${batch},${k},${n}]`,
}));

/**
 * Eight derived bmm cases (ADR 0116 decision 4): two N 64 shapes for each of the buckets M 16 / 32
 * / 128 / 256, with the M of a census row replaced. `K 64` replaces the M of text_conditioner
 * `[16,512,64] × [16,64,64]`, `K 128` that of text_encoder `[16,64,128] × [16,128,64]` (M values
 * absent from the census). The N 64 shapes were chosen because, at M 64 / 512 where both linear and
 * bmm were measured, it was the N 64 shapes that stopped the linear-only winners. count carries the
 * count of the replaced row (handled like the linear derivations).
 */
const BMM_DERIVED_CASES: readonly BmmCase[] = [16, 32, 128, 256].flatMap((m) =>
  [
    { component: "text_conditioner", origin: 512, k: 64, count: 12 },
    { component: "text_encoder", origin: 64, k: 128, count: 28 },
  ].map(({ component, origin, k, count }) => ({
    id: `bmm-b16-m${m}-n64-k${k}`,
    op: "bmm" as const,
    batch: 16,
    m,
    n: 64,
    k,
    censusCount: count,
    source: `${component} bmm [16,${m},${k}] × [16,${k},64] (census M ${origin} replaced with ${m})`,
  })),
);

/**
 * The DiT's four attention shapes (census: transformer's self `[1,16,4096,128]`², cross N = 512,
 * count 28 each).
 */
const DIT_ATTENTION: readonly { readonly kind: string; readonly m: number; readonly n: number }[] =
  [
    { kind: "self", m: 1024, n: 1024 },
    { kind: "self", m: 4096, n: 4096 },
    { kind: "cross", m: 1024, n: 512 },
    { kind: "cross", m: 4096, n: 512 },
  ];

const attentionCases = (op: AttentionCase["op"]): AttentionCase[] =>
  DIT_ATTENTION.flatMap(({ kind, m, n }) =>
    (["qk", "pv"] as const).map((stage) => ({
      id: `${op}-${stage}-${kind}-m${m}-n${n}`,
      op,
      stage,
      batchHeads: 16,
      m,
      n,
      d: 128,
      scale: DIT_ATTENTION_SCALE,
      score: op === "attention" ? ("f32" as const) : ("f16" as const),
      censusCount: 28,
      source: `transformer attention [1,16,${m},128] × [1,16,${n},128]${
        m === 4096 ? "" : " (census M 4096 replaced with 1024)"
      }`,
    })),
  );

const ATTENTION_CASES: readonly AttentionCase[] = [
  ...attentionCases("attention"),
  ...attentionCases("i8a8-attention"),
];

/**
 * The VAE's top three conv2d rows (census `count × out_elements` descending — 150,994,944 /
 * 75,497,472 / 31,457,280). Cout = 96 uses the 32-row m tile (`igemm32x128`), 192 / 384 the 64-row
 * one (`igemm64x128`).
 */
const CONV2D_CASES: readonly Conv2dCase[] = [
  { channels: 96, size: 512, count: 6 },
  { channels: 192, size: 256, count: 6 },
  { channels: 384, size: 128, count: 5 },
].map(({ channels, size, count }) => ({
  id: `conv2d-c${channels}-${size}x${size}`,
  op: "conv2d" as const,
  channelsIn: channels,
  channelsOut: channels,
  height: size,
  width: size,
  censusCount: count,
  source: `vae_decoder conv2d [1,${channels},${size},${size}] × [${channels},${channels},3,3]`,
}));

/**
 * One conv3d w8a8 shape of the Wan2.2 VAE decoder: a `k`×`k`×`k` kernel (3×3×3 with spatial
 * padding 1 and no time padding — the causal cache frames are part of the input — or the 1×1×1
 * shortcut), stride 1, dilation 1, groups 1.
 */
const conv3dCase = (shape: {
  readonly channelsIn: number;
  readonly channelsOut: number;
  readonly kernel: 1 | 3;
  readonly input: readonly [number, number, number];
  readonly count: number;
  readonly source: string;
}): Conv3dI8a8Case => {
  const { channelsIn, channelsOut, kernel, input, count, source } = shape;
  const [timeIn, heightIn, widthIn] = input;
  const padding = (kernel - 1) / 2;
  const dims: Conv3dDims = {
    channelsIn,
    channelsOut,
    timeIn,
    heightIn,
    widthIn,
    timeOut: timeIn - kernel + 1,
    heightOut: heightIn,
    widthOut: widthIn,
    kernelT: kernel,
    kernelH: kernel,
    kernelW: kernel,
    strideT: 1,
    strideH: 1,
    strideW: 1,
    paddingT: 0,
    paddingH: padding,
    paddingW: padding,
    dilationT: 1,
    dilationH: 1,
    dilationW: 1,
    groups: 1,
  };
  const k = `${kernel}${kernel}${kernel}`;
  return {
    id: `i8a8-conv3d-cin${channelsIn}-cout${channelsOut}-k${k}-in${timeIn}x${heightIn}x${widthIn}`,
    op: "i8a8-conv3d",
    dims,
    censusCount: count,
    source: `${source} conv3d [${channelsIn},${timeIn},${heightIn},${widthIn}] × [${channelsOut},${channelsIn},${kernel},${kernel},${kernel}] (${count} nodes)`,
  };
};

/** The six heavy 3×3×3 shapes of one decoder tile edge (16 or 32 latent pixels). */
const heavyConv3dShapes = (tile: 16 | 32) => {
  const next = tile === 16 ? "vae_decoder_next" : "vae_decoder_next_32";
  const both =
    tile === 16
      ? "vae_decoder_first / vae_decoder_next"
      : "vae_decoder_first_32 / vae_decoder_next_32";
  return [
    { channelsIn: 256, channelsOut: 256, input: [6, 8 * tile, 8 * tile], count: 5, source: next },
    { channelsIn: 512, channelsOut: 512, input: [6, 4 * tile, 4 * tile], count: 5, source: next },
    { channelsIn: 1024, channelsOut: 1024, input: [4, 2 * tile, 2 * tile], count: 6, source: next },
    { channelsIn: 1024, channelsOut: 1024, input: [3, tile, tile], count: 10, source: both },
    { channelsIn: 1024, channelsOut: 512, input: [6, 4 * tile, 4 * tile], count: 1, source: next },
    { channelsIn: 512, channelsOut: 256, input: [6, 8 * tile, 8 * tile], count: 1, source: next },
  ] as const;
};

/**
 * The conv3d w8a8 cases (ADR 0115 addendum 2026-10-26). One geometry serves every conv3d shape, so
 * the cases are the shapes that decide its time plus the edges that could hide a regression:
 *
 * - The six heavy 3×3×3 shapes of the default tile-16 decoder — about 93% of its conv3d time per
 *   tile (a recon tally at the default geometry, one first chunk and eight next chunks).
 * - Two edge guards of the tile-16 decoder: the 12-channel `conv_out` (the smallest Cout) and one
 *   1×1×1 shortcut (the 512×1024 one at 4×64², where the recon's best geometry had its smallest
 *   gain).
 * - The six heavy 3×3×3 shapes of the tile-32 pair (ADR 0134 — 4× the pixels per tile).
 *
 * Where the first and next chunks differ, the shape is the next chunk's (it runs for every chunk
 * after the first). `censusCount` is the node count in that graph.
 */
const CONV3D_I8A8_CASES: readonly Conv3dI8a8Case[] = [
  ...heavyConv3dShapes(16).map((shape) => conv3dCase({ ...shape, kernel: 3 })),
  conv3dCase({
    channelsIn: 256,
    channelsOut: 12,
    kernel: 3,
    input: [6, 128, 128],
    count: 1,
    source: "vae_decoder_next",
  }),
  conv3dCase({
    channelsIn: 1024,
    channelsOut: 512,
    kernel: 1,
    input: [4, 64, 64],
    count: 1,
    source: "vae_decoder_next",
  }),
  ...heavyConv3dShapes(32).map((shape) => conv3dCase({ ...shape, kernel: 3 })),
];

/** Every case (in op-family order = {@link SWEEP_OPS}). */
export const SWEEP_CASES: readonly SweepCase[] = [
  ...LINEAR_CASES.filter((c) => c.op === "linear"),
  ...MATMUL_CASES,
  ...BMM_CASES,
  ...BMM_DERIVED_CASES,
  ...LINEAR_CASES.filter((c) => c.op === "i8a8-linear"),
  ...ATTENTION_CASES.filter((c) => c.op === "attention"),
  ...ATTENTION_CASES.filter((c) => c.op === "i8a8-attention"),
  ...CONV2D_CASES,
  ...CONV3D_I8A8_CASES,
];

/**
 * The id of a K-85 case (`k85-` before the linear spelling, so that it never meets an id of
 * {@link SWEEP_CASES} — the measured-candidate fingerprint looks rows up in both tables).
 */
const k85Id = (m: number, n: number, k: number): string =>
  `k85-${linearId("i8a8-linear", m, n, k)}`;

/**
 * The cases of the evaluation set `k85` (perf-ledger K-85 — the vec4 skeleton of the i8a8 linear),
 * all `i8a8-linear` with i8 weights: the Wan2.2 DiT's linears ({@link WAN22_DIT_LINEAR}, the
 * weight of the set's forward sum) under the set's own ids, then the Irodori guard.
 */
export const K85_CASES: readonly LinearCase[] = WAN22_DIT_LINEAR.map(
  ({ m, n, k, count, what }): LinearCase => ({
    id: k85Id(m, n, k),
    op: "i8a8-linear",
    m,
    n,
    k,
    censusCount: count,
    source: `${WAN22_DIT} [1,${m},${k}] × [${n},${k}] (${what})`,
  }),
);

/** The small-M guard of {@link K85_CASES} (Irodori v4-small, not in the Wan2.2 forward). */
const K85_GUARD_CASES: readonly LinearCase[] = [
  {
    id: k85Id(185, 1280, 1280),
    op: "i8a8-linear",
    m: 185,
    n: 1280,
    k: 1280,
    censusCount: 0,
    source:
      "Irodori v4-small dit [1,185,1280] × [1280,1280] (60 nodes per DiT step; not in the Wan2.2 forward)",
  },
];

/** Every case of the evaluation set `k85` (the Wan2.2 DiT forward, then the guard). */
const K85_TABLE: readonly SweepCase[] = [...K85_CASES, ...K85_GUARD_CASES];

/**
 * What the `censusCount` of a `k88` case counts (its `censusGroup` — the unit of the set's
 * sums). Each count is the node count of the shape in that unit's graph.
 */
export const K88_CENSUS = {
  gemma4Prefill: "one gemma4 E2B prefill chunk (M = 768)",
  umt5: "one umT5-XXL encoder forward (512 tokens)",
  siglip2: "one siglip2 so400m vision forward (729 tokens)",
  wanNext: "one Wan2.2 TI2V-5B VAE decoder next chunk (tile 16)",
  wanFirst: "one Wan2.2 TI2V-5B VAE decoder first chunk (tile 16)",
  anima: "one anima VAE decode (the graph's 64×64 latent)",
} as const;

/** A `k88` linear (f32 compute, the model's weight storage). */
const k88Linear = (shape: {
  readonly m: number;
  readonly n: number;
  readonly k: number;
  readonly weight: CaseWeight;
  readonly groupSize?: number;
  readonly count: number;
  readonly group?: string;
  readonly source: string;
}): LinearCase => {
  const { m, n, k, weight, groupSize, count, group, source } = shape;
  return {
    id: `k88-${linearId("linear", m, n, k)}-w${weight}${groupSize === undefined ? "" : `g${groupSize}`}`,
    op: "linear",
    m,
    n,
    k,
    weight,
    ...(groupSize === undefined ? {} : { groupSize }),
    censusCount: count,
    ...(group === undefined ? {} : { censusGroup: group }),
    source,
  };
};

/** A `k88` conv2d (3×3, stride 1, padding 1, Cin = Cout — the shapes of the two VAE decoders). */
const k88Conv2d = (shape: {
  readonly batch: number;
  readonly channels: number;
  readonly size: number;
  readonly weight: CaseWeight;
  readonly count: number;
  readonly group: string;
  readonly source: string;
}): Conv2dCase => {
  const { batch, channels, size, weight, count, group, source } = shape;
  return {
    id: `k88-conv2d-b${batch}-c${channels}-${size}x${size}-w${weight}`,
    op: "conv2d",
    batch,
    channelsIn: channels,
    channelsOut: channels,
    height: size,
    width: size,
    weight,
    censusCount: count,
    censusGroup: group,
    source: `${source} conv2d [${batch},${channels},${size},${size}] × [${channels},${channels},3,3]`,
  };
};

/** The spike's square f32 linear (4,096³, the "big-f32" of its shape list; no model). */
const K88_SQUARE_LINEAR = k88Linear({
  m: 4096,
  n: 4096,
  k: 4096,
  weight: "f32",
  count: 0,
  source: "square linear [4096,4096] × [4096,4096] f32 (the spike's shape list; no model)",
});

/**
 * The cases of the evaluation set `k88` (perf-ledger K-88 — the f32 skeleton forms and the conv2d
 * counter decode), copied from the distributions' graphs (`models/`, the nodes by input and weight
 * shape and the weight's storage) and not read at run time:
 *
 * - gemma4 E2B (`karume-gemma4-qat` `e2b`, int4 at group 32) at the 768-row prefill chunk: the MLP
 *   gate / up of the 6,144-wide and the 12,288-wide layers (30 / 40 nodes), their down (15 / 20)
 *   and the q projection of the sliding and the full layers (28 / 7) — 140 of the chunk's 277
 *   linears.
 * - umT5-XXL (`karume-umt5-xxl`, int8) at 512 tokens: the feed-forward wi (48 nodes).
 * - siglip2 so400m (`karume-siglip2`, f32) at 729 tokens: the MLP fc1 (27 nodes).
 * - From the spike's shape list (no model — count 0): a small-M linear (M = 64, 4,096², f32), the
 *   square 4,096³ f32 linear and its matmul counterpart.
 * - The Wan2.2 TI2V-5B VAE decoder (`vae_decoder_next` / `vae_decoder_first` at tile 16, int8 as the
 *   default preset stores it): the three 3×3 conv2d of a next chunk (the chunk's frames in the
 *   batch: 2 at 32², 4 at 64², 4 at 128²) and of a first chunk (1 frame each) — the three heaviest
 *   of each chunk's five (the other two are 1×1 at 16²).
 * - anima's VAE decoder (`karume-anima`, f16): c192 at 256² (6 nodes) and c384 at 128² (5).
 *
 * `censusCount` counts the nodes in the unit its `censusGroup` names ({@link K88_CENSUS}).
 */
export const K88_CASES: readonly SweepCase[] = [
  ...[
    { n: 6144, k: 1536, count: 30, what: "MLP gate / up of the 6,144-wide layers" },
    { n: 12288, k: 1536, count: 40, what: "MLP gate / up of the 12,288-wide layers" },
    { n: 1536, k: 6144, count: 15, what: "MLP down of the 6,144-wide layers" },
    { n: 1536, k: 12288, count: 20, what: "MLP down of the 12,288-wide layers" },
    { n: 2048, k: 1536, count: 28, what: "q of the sliding layers" },
    { n: 4096, k: 1536, count: 7, what: "q of the full layers" },
  ].map(({ n, k, count, what }) =>
    k88Linear({
      m: 768,
      n,
      k,
      weight: "i4",
      groupSize: 32,
      count,
      group: K88_CENSUS.gemma4Prefill,
      source: `gemma4 E2B model [1,768,${k}] × [${n},${k}] int4 g32 (${what})`,
    }),
  ),
  k88Linear({
    m: 512,
    n: 10240,
    k: 4096,
    weight: "i8",
    count: 48,
    group: K88_CENSUS.umt5,
    source: "umT5-XXL text_encoder [1,512,4096] × [10240,4096] int8 (feed-forward wi)",
  }),
  k88Linear({
    m: 729,
    n: 4304,
    k: 1152,
    weight: "f32",
    count: 27,
    group: K88_CENSUS.siglip2,
    source: "siglip2 so400m vision [1,729,1152] × [4304,1152] f32 (MLP fc1)",
  }),
  k88Linear({
    m: 64,
    n: 4096,
    k: 4096,
    weight: "f32",
    count: 0,
    source: "small-M guard [64,4096] × [4096,4096] f32 (the spike's shape list; no model)",
  }),
  K88_SQUARE_LINEAR,
  {
    id: "k88-matmul-m4096-n4096-k4096",
    op: "matmul",
    m: 4096,
    n: 4096,
    k: 4096,
    censusCount: 0,
    mirrorOf: K88_SQUARE_LINEAR.id,
    source: `square matmul [4096,4096] × [4096,4096], the B-side counterpart of ${K88_SQUARE_LINEAR.id} (no model)`,
  },
  ...[
    { batch: 2, channels: 1024, size: 32 },
    { batch: 4, channels: 1024, size: 64 },
    { batch: 4, channels: 512, size: 128 },
  ].map(({ batch, channels, size }) =>
    k88Conv2d({
      batch,
      channels,
      size,
      weight: "i8",
      count: 1,
      group: K88_CENSUS.wanNext,
      source: "Wan2.2 TI2V-5B vae_decoder_next",
    }),
  ),
  ...[
    { channels: 1024, size: 32 },
    { channels: 1024, size: 64 },
    { channels: 512, size: 128 },
  ].map(({ channels, size }) =>
    k88Conv2d({
      batch: 1,
      channels,
      size,
      weight: "i8",
      count: 1,
      group: K88_CENSUS.wanFirst,
      source: "Wan2.2 TI2V-5B vae_decoder_first",
    }),
  ),
  ...[
    { channels: 192, size: 256, count: 6 },
    { channels: 384, size: 128, count: 5 },
  ].map(({ channels, size, count }) =>
    k88Conv2d({
      batch: 1,
      channels,
      size,
      weight: "f16",
      count,
      group: K88_CENSUS.anima,
      source: "anima vae_decoder",
    }),
  ),
];

/**
 * The case table a candidate set sweeps: the evaluation sets `k85` and `k88` their own tables,
 * every other set {@link SWEEP_CASES}. The sweep, its record's case-set version and default
 * fingerprint, and the page's case list take it from here.
 */
export const sweepCasesFor = (set: CandidateSet): readonly SweepCase[] =>
  set === "k85" ? K85_TABLE : set === "k88" ? K88_CASES : SWEEP_CASES;

/**
 * Every case any set sweeps (the lookup of the measured-candidate fingerprint — a row names its case
 * by id; the ids of the three tables never meet).
 */
export const ALL_SWEEP_CASES: readonly SweepCase[] = [...SWEEP_CASES, ...K85_TABLE, ...K88_CASES];

/**
 * A sweep case as an application sees it: the id `GeometrySweepOptions.cases` takes, and its op.
 * `censusCount` is set on the cases of the evaluation sets only: for `k85` the node count in one
 * Wan2.2 TI2V-5B 33f DiT forward (0 for a guard shape of another model) — the weight of that set's
 * forward sum; for `k88` the node count in the unit `censusGroup` names (0 and no group for a
 * shape of no model) — the weight of that unit's sum.
 */
export type SweepCaseEntry = {
  readonly id: string;
  readonly op: SweepOp;
  readonly censusCount?: number;
  readonly censusGroup?: string;
};

/**
 * Every case of the case table of `set` ({@link sweepCasesFor} — default: {@link SWEEP_CASES}) as
 * its id and op, in the table's order (the `k85` cases with their `censusCount`, the `k88` cases
 * with their `censusCount` and `censusGroup`). A new array of new objects on every call: the table
 * itself stays internal (its other descriptive fields are no public surface), and nothing a caller
 * does to the result reaches the case set or the fingerprints.
 */
export const sweepCaseEntries = (set: CandidateSet = DEFAULT_CANDIDATE_SET): SweepCaseEntry[] =>
  set === "k85"
    ? K85_TABLE.map(({ id, op, censusCount }) => ({ id, op, censusCount }))
    : set === "k88"
      ? K88_CASES.map((sweepCase) => ({
          id: sweepCase.id,
          op: sweepCase.op,
          censusCount: sweepCase.censusCount,
          ...("censusGroup" in sweepCase && sweepCase.censusGroup !== undefined
            ? { censusGroup: sweepCase.censusGroup }
            : {}),
        }))
      : SWEEP_CASES.map(({ id, op }) => ({ id, op }));

/**
 * The operation count of one dispatch (multiply-add = 2). For i8a8, the number of integer
 * operations (the TOPS equivalent).
 */
export const caseFlops = (sweepCase: SweepCase): number => {
  switch (sweepCase.op) {
    case "linear":
    case "matmul":
    case "i8a8-linear":
      return 2 * sweepCase.m * sweepCase.n * sweepCase.k;
    case "bmm":
      return 2 * sweepCase.batch * sweepCase.m * sweepCase.n * sweepCase.k;
    case "attention":
    case "i8a8-attention":
      return 2 * sweepCase.batchHeads * sweepCase.m * sweepCase.n * sweepCase.d;
    case "conv2d":
      // As implicit GEMM (M = Cout, N = H·W, K = Cin·9, per batch — skipped reads of the padding
      // are not counted)
      return (
        2 *
        (sweepCase.batch ?? 1) *
        sweepCase.channelsOut *
        sweepCase.height *
        sweepCase.width *
        sweepCase.channelsIn *
        9
      );
    case "i8a8-conv3d": {
      // As implicit GEMM (M = Cout, N = Tout·Hout·Wout, K = Cin·Kt·Kh·Kw — out-of-range taps are
      // counted too)
      const { dims } = sweepCase;
      const pixelsOut = dims.timeOut * dims.heightOut * dims.widthOut;
      const taps = dims.kernelT * dims.kernelH * dims.kernelW;
      return 2 * dims.channelsOut * pixelsOut * dims.channelsIn * taps;
    }
  }
};

/**
 * One line of shape for display. The generator (slotOf in derive.ts) reads the M of linear / matmul
 * / bmm with a regular expression, so when changing the spelling of M (`M{m}`), change slotOf to
 * match.
 */
export const caseShape = (sweepCase: SweepCase): string => {
  switch (sweepCase.op) {
    case "linear":
    case "matmul":
    case "i8a8-linear":
      return `M${sweepCase.m} N${sweepCase.n} K${sweepCase.k}`;
    case "bmm":
      return `B${sweepCase.batch} M${sweepCase.m} N${sweepCase.n} K${sweepCase.k}`;
    case "attention":
    case "i8a8-attention":
      return `${sweepCase.stage} BH${sweepCase.batchHeads} M${sweepCase.m} N${sweepCase.n} D${sweepCase.d}`;
    case "conv2d":
      return `${sweepCase.batch === undefined ? "" : `B${sweepCase.batch} `}Cin${sweepCase.channelsIn} Cout${sweepCase.channelsOut} ${sweepCase.height}x${sweepCase.width} 3x3`;
    case "i8a8-conv3d": {
      const { dims } = sweepCase;
      const kernel = `${dims.kernelT}x${dims.kernelH}x${dims.kernelW}`;
      const input = `in${dims.timeIn}x${dims.heightIn}x${dims.widthIn}`;
      const output = `out${dims.timeOut}x${dims.heightOut}x${dims.widthOut}`;
      return `Cin${dims.channelsIn} Cout${dims.channelsOut} ${kernel} ${input} ${output}`;
    }
  }
};
