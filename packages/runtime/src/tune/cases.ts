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
 */

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
  | "conv2d";

export const SWEEP_OPS: readonly SweepOp[] = [
  "linear",
  "matmul",
  "bmm",
  "i8a8-linear",
  "attention",
  "i8a8-attention",
  "conv2d",
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

export type SweepCase = LinearCase | MatmulCase | BmmCase | AttentionCase | Conv2dCase;

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
];

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
  }
};
