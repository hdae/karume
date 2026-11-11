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
 * The evaluation set `k85` (perf-ledger K-85) has a table of its own ({@link K85_CASES}), outside
 * {@link SWEEP_CASES}: the case-set version and the kernel fingerprints of the table-deriving sets
 * do not see it ({@link sweepCasesFor}).
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
  readonly censusCount: number;
  /** Which census row (the component and its input shape in the census). */
  readonly source: string;
};

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
 * conv2d's implicit GEMM (groups = 1, 3×3, stride 1, padding 1, dilation 1, B = 1, f16-stored
 * weights). `M = Cout`, `N = H·W`, `K = Cin·9`.
 */
export type Conv2dCase = {
  readonly id: string;
  readonly op: "conv2d";
  readonly channelsIn: number;
  readonly channelsOut: number;
  readonly height: number;
  readonly width: number;
  readonly censusCount: number;
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

/** The source of the K-85 cases of the Wan2.2 DiT (the forward the census counts are taken from). */
const WAN22_DIT = "Wan2.2 TI2V-5B transformer (33 frames at 1280×704, S = 7,920)";

/**
 * The cases of the evaluation set `k85` (perf-ledger K-85 — the vec4 skeleton of the i8a8 linear),
 * all `i8a8-linear` with i8 weights. The source is the `transformer` graph of the Wan2.2 TI2V-5B
 * distribution (`models/karume-wan2.2/ti2v-5b/transformer`, its `linear` nodes by input and weight
 * shape) with `S = 7,920` (33 frames at 1280×704), and the `dit` graph of Irodori v4-small for the
 * small-M guard. Copied, not read at run time.
 *
 * `censusCount` is the node count **in one Wan2.2 33f DiT forward** (the weight of the set's forward
 * sum): self-attention q / k / v / o and cross-attention q / o 180 (30 blocks × 6), the FFN up and
 * down 30 each, the cross-attention k / v at the 512 text tokens 60 plus the text embedder's second
 * layer 1, and the time projection at M = 1 2 (one per timestep input). Those are 303 of the
 * graph's 310 linears; the 7 left out are the patch embedding, the output projection and the text
 * embedder's first layer (1 node each) and the timestep embedder's two layers at M = 1 (2 each). The
 * Irodori case is a guard for the families whose a8 linears run at small M (its default preset
 * `i8-a8` reads the same `i8a8.linear` field): count 0, it is not in the Wan2.2 forward (60 nodes
 * per Irodori DiT step at M = S = 185, the M of the research's regression measurement —
 * docs/research/2026-10-25-speed-recon-3080ti.md §5).
 */
export const K85_CASES: readonly LinearCase[] = [
  { m: 7920, n: 3072, k: 3072, count: 180, what: "self q / k / v / o and cross q / o" },
  { m: 7920, n: 14336, k: 3072, count: 30, what: "FFN up" },
  { m: 7920, n: 3072, k: 14336, count: 30, what: "FFN down" },
  { m: 512, n: 3072, k: 3072, count: 61, what: "cross k / v and the text embedder's second layer" },
  { m: 1, n: 18432, k: 3072, count: 2, what: "time projection" },
].map(({ m, n, k, count, what }): LinearCase => ({
  id: k85Id(m, n, k),
  op: "i8a8-linear",
  m,
  n,
  k,
  censusCount: count,
  source: `${WAN22_DIT} [1,${m},${k}] × [${n},${k}] (${what})`,
}));

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
 * The case table a candidate set sweeps: the evaluation set `k85` its own table, every other set
 * {@link SWEEP_CASES}. The sweep, its record's case-set version and default fingerprint, and the
 * page's case list take it from here.
 */
export const sweepCasesFor = (set: CandidateSet): readonly SweepCase[] =>
  set === "k85" ? K85_TABLE : SWEEP_CASES;

/**
 * Every case any set sweeps (the lookup of the measured-candidate fingerprint — a row names its case
 * by id; the ids of the two tables never meet).
 */
export const ALL_SWEEP_CASES: readonly SweepCase[] = [...SWEEP_CASES, ...K85_TABLE];

/**
 * A sweep case as an application sees it: the id `GeometrySweepOptions.cases` takes, and its op.
 * `censusCount` is set on the cases of the evaluation set `k85` only: the node count in one Wan2.2
 * TI2V-5B 33f DiT forward (0 for a guard shape of another model) — the weight of that set's forward
 * sum.
 */
export type SweepCaseEntry = {
  readonly id: string;
  readonly op: SweepOp;
  readonly censusCount?: number;
};

/**
 * Every case of the case table of `set` ({@link sweepCasesFor} — default: {@link SWEEP_CASES}) as
 * its id and op, in the table's order (the `k85` cases with their `censusCount`). A new array of new
 * objects on every call: the table itself stays internal (its other descriptive fields are no public
 * surface), and nothing a caller does to the result reaches the case set or the fingerprints.
 */
export const sweepCaseEntries = (set: CandidateSet = DEFAULT_CANDIDATE_SET): SweepCaseEntry[] =>
  set === "k85"
    ? K85_TABLE.map(({ id, op, censusCount }) => ({ id, op, censusCount }))
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
      // As implicit GEMM (M = Cout, N = H·W, K = Cin·9 — skipped reads of the padding are not
      // counted)
      return (
        2 * sweepCase.channelsOut * sweepCase.height * sweepCase.width * sweepCase.channelsIn * 9
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
      return `Cin${sweepCase.channelsIn} Cout${sweepCase.channelsOut} ${sweepCase.height}x${sweepCase.width} 3x3`;
    case "i8a8-conv3d": {
      const { dims } = sweepCase;
      const kernel = `${dims.kernelT}x${dims.kernelH}x${dims.kernelW}`;
      const input = `in${dims.timeIn}x${dims.heightIn}x${dims.widthIn}`;
      const output = `out${dims.timeOut}x${dims.heightOut}x${dims.widthOut}`;
      return `Cin${dims.channelsIn} Cout${dims.channelsOut} ${kernel} ${input} ${output}`;
    }
  }
};
