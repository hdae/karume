/**
 * タイル幾何の掃引（perf-ledger K-70）で測る**形状表**（純データ）。
 *
 * 出典は anima の op census（`outputs/bench/karume-anima/2026-09-04_op-census/summary.json` の
 * `scenarios[0].weights[]` — 1024px 生成・パッチトークン S = 4096・quant `f16+dit8-a8-attn8-s16`）。
 * 値は census から**書き写した**もので、実行時に census を読まない（census の在否と形式に道具を
 * 依存させない — 形状は anima の IR が変わらない限り動かない）。
 *
 * `censusCount` は census の `count`（そのグラフ 1 本 = 各コンポーネント 1 回の forward の中の
 * ノード本数）。M = 1024（512²）の行は census に無い — 1024px の行の M（S = 4096）を 512px の
 * S = 1024 に置き換えたもので、ノード本数は解像度に依らないので同じ count を載せる。
 */

/** 掃引の op 族（CLI の `--op` とページのチェックボックスの語彙）。 */
export type SweepOp = "linear" | "i8a8-linear" | "attention" | "i8a8-attention" | "conv2d";

export const SWEEP_OPS: readonly SweepOp[] = [
  "linear",
  "i8a8-linear",
  "attention",
  "i8a8-attention",
  "conv2d",
];

/**
 * linear（`x[M,K] · Wᵀ[K,N] + b[N]`）。`linear` は重み f16 格納 × f32 計算（quant `f16` の DiT と
 * text 系と同じ `wf16`）、`i8a8-linear` は重み i8（per-channel scale）× 活性 per-token i8 の整数内積。
 */
export type LinearCase = {
  readonly id: string;
  readonly op: "linear" | "i8a8-linear";
  readonly m: number;
  readonly n: number;
  readonly k: number;
  readonly censusCount: number;
  /** census のどの行か（コンポーネントと census 上の入力形）。 */
  readonly source: string;
};

/**
 * 融合 attention の ①QK / ③PV（B·H = 16・D = 128）。`attention` は f32 計算・S は f32 格納、
 * `i8a8-attention` は整数内積・S は f16 格納（s16 — quant `…-attn8-s16` と同じ）。
 * ③PV は同じ形の ①QK → ②行統計 を既定幾何で 1 度回した S と行統計を入力にする。
 */
export type AttentionCase = {
  readonly id: string;
  readonly op: "attention" | "i8a8-attention";
  readonly stage: "qk" | "pv";
  /** B·H（バッチ軸に畳んだ本数）。 */
  readonly batchHeads: number;
  /** クエリ長 M。 */
  readonly m: number;
  /** キー長 N。 */
  readonly n: number;
  /** head の次元 D。 */
  readonly d: number;
  /** attrs の `scale`（半スケール — q / k の両方に掛かる）。 */
  readonly scale: number;
  readonly score: "f32" | "f16";
  readonly censusCount: number;
  readonly source: string;
};

/**
 * conv2d の implicit GEMM（groups = 1・3×3・stride 1・padding 1・dilation 1・B = 1・重み f16 格納）。
 * `M = Cout`・`N = H·W`・`K = Cin·9`。
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

export type SweepCase = LinearCase | AttentionCase | Conv2dCase;

/** census の attention の `attrs.scale`（DiT の self / cross 共通 — `128^-0.25`）。 */
const DIT_ATTENTION_SCALE = 0.2973017692565918;

/** DiT の linear 3 形（census: transformer の `[1,4096,K] × [N,K]`・count 168 / 28 / 28）。 */
const DIT_LINEAR: readonly {
  readonly n: number;
  readonly k: number;
  readonly count: number;
}[] = [
  { n: 2048, k: 2048, count: 168 },
  { n: 8192, k: 2048, count: 28 },
  { n: 2048, k: 8192, count: 28 },
];

/** DiT の M（512² = 1024・1024² = 4096）。 */
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
        m === 4096 ? "" : "（census の M 4096 を 1024 に置換）"
      }`,
    }))
  );

const LINEAR_CASES: readonly LinearCase[] = [
  ...ditLinear("linear"),
  {
    // 中 M の対照（census: transformer の cross-attention の k / v 射影 `[1,512,1024] × [2048,1024]`）
    id: linearId("linear", 512, 2048, 1024),
    op: "linear",
    m: 512,
    n: 2048,
    k: 1024,
    censusCount: 56,
    source: "transformer [1,512,1024] × [2048,1024]",
  },
  {
    // 小 M の対照（census: text_encoder の `[1,64,1024] × [3072,1024]`・格納 f16）
    id: linearId("linear", 64, 3072, 1024),
    op: "linear",
    m: 64,
    n: 3072,
    k: 1024,
    censusCount: 56,
    source: "text_encoder [1,64,1024] × [3072,1024]",
  },
  ...ditLinear("i8a8-linear"),
];

/** DiT の attention 4 形（census: transformer の self `[1,16,4096,128]`²・cross N = 512・各 count 28）。 */
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
      score: op === "attention" ? "f32" as const : "f16" as const,
      censusCount: 28,
      source: `transformer attention [1,16,${m},128] × [1,16,${n},128]${
        m === 4096 ? "" : "（census の M 4096 を 1024 に置換）"
      }`,
    }))
  );

const ATTENTION_CASES: readonly AttentionCase[] = [
  ...attentionCases("attention"),
  ...attentionCases("i8a8-attention"),
];

/**
 * VAE の conv2d 上位 3 行（census の `count × out_elements` 降順 — 150,994,944 / 75,497,472 /
 * 31,457,280）。Cout = 96 は m タイル 32 行（`igemm32x128`）、192 / 384 は 64 行（`igemm64x128`）。
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

/** 全ケース（op 族の順 = {@link SWEEP_OPS}）。 */
export const SWEEP_CASES: readonly SweepCase[] = [
  ...LINEAR_CASES.filter((c) => c.op === "linear"),
  ...LINEAR_CASES.filter((c) => c.op === "i8a8-linear"),
  ...ATTENTION_CASES.filter((c) => c.op === "attention"),
  ...ATTENTION_CASES.filter((c) => c.op === "i8a8-attention"),
  ...CONV2D_CASES,
];

/** 1 dispatch の演算数（積和 = 2）。i8a8 は整数演算の数（TOPS 相当）。 */
export const caseFlops = (sweepCase: SweepCase): number => {
  switch (sweepCase.op) {
    case "linear":
    case "i8a8-linear":
      return 2 * sweepCase.m * sweepCase.n * sweepCase.k;
    case "attention":
    case "i8a8-attention":
      return 2 * sweepCase.batchHeads * sweepCase.m * sweepCase.n * sweepCase.d;
    case "conv2d":
      // implicit GEMM 換算（M = Cout・N = H·W・K = Cin·9 — padding 域の読み飛ばしは数えない）
      return 2 * sweepCase.channelsOut * sweepCase.height * sweepCase.width *
        sweepCase.channelsIn * 9;
  }
};

/** 表示用の形状 1 行。 */
export const caseShape = (sweepCase: SweepCase): string => {
  switch (sweepCase.op) {
    case "linear":
    case "i8a8-linear":
      return `M${sweepCase.m} N${sweepCase.n} K${sweepCase.k}`;
    case "attention":
    case "i8a8-attention":
      return `${sweepCase.stage} BH${sweepCase.batchHeads} M${sweepCase.m} N${sweepCase.n} D${sweepCase.d}`;
    case "conv2d":
      return `Cin${sweepCase.channelsIn} Cout${sweepCase.channelsOut} ${sweepCase.height}x${sweepCase.width} 3x3`;
  }
};
