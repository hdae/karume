/**
 * **束の census 表**（ADR 0110 決定 5 ① / ADR 0058 決定 4 ③）— 系列 × モデル × 実効 SessionOptions ×
 * 部品 × 相 → 席ごとの**期待本数（ちょうど）**。
 *
 * 席の判別述語は runtime の `tests/helpers/pipeline-census.ts`（`SEAT_SIGNATURES`）が持ち、ここは
 * 「その束のその相で、変種キーが何本・参照変種のキーが何本走るか」だけを持つ。e2e はこの表に
 * 行がある束を {@link phaseExpectations} → `assertPipelineCensus` で**ちょうど**の本数で見る
 * （行の無い束は従来どおり `assertSeatsApplied` の 1 本以上 / 0 本）。
 *
 * ## 鍵
 *
 * 束の鍵は**実効** SessionOptions（manifest の quant 席の `session` と明示指定を家族の合成
 * `resolveSessionOptions` に通した結果 — ADR 0111）から {@link bundleKey} で作る。manifest の
 * `session` の字面で引かないのは、合成の外で値が足されると（stateAttentionReduce 昇格前の gemma の
 * models 既定がそうだった）表の鍵と実際に走る束がずれるため。参照値の欄は鍵から落とす
 * （`linearGemvReduce: "sequential"` の明示は未指定と同じ束）。
 *
 * ## 値の出どころ（2026-09-26 時点・全て GPU 不要で導出）
 *
 * - **既存テストに散っていた値**: irodori DiT の i8a8 linear 317（`e2e_irodori_w8a8_test.ts` —
 *   ADR 0114 でグラフを割った後は `dit` 245 + `dit_context` 72）・
 *   gemma4 E2B decode の GEMV 277（`e2e_gemma4_greedy_test.ts`）・RMS→add 融合 106 / linear→SRQ
 *   融合 275 / packed SRQ 70（`runtime/tests/assets_fusion_counts_test.ts`）。
 * - **配布ミラーの計画から機械的に導いた値**: 容器の part 0（グラフ宣言だけ）を読み、
 *   `gpu_linear_gemv_parallel_census_test.ts` の GPU 不要 describe と同じ手筋 —
 *   linear の格納（`planWeightResidency`）と形を recipe-builder と同じ述語
 *   （`linearGemvParallelEligible` / i8a8 の「i8・i4 × k % 4 == 0」/ f16 計算の「i8・i4 以外」）に
 *   通して本数を出す。融合で決まる本数（RMS→add・linear→SRQ・packed）は `planFusions` の
 *   融合ステップが計画時に持つ dispatch キーを `SEAT_SIGNATURES` で数えた値（WebGPU core 既定の
 *   limits — `assets_fusion_counts_test.ts` と同じ）。states 形 attention は
 *   `recipe-builders/attention.ts` の選択（M ≤ 8 で ①' / M < 16 で ③'・readonly は常に ①'③'）と
 *   M ≤ 8 の行ブロック 1 枚から、attention ノード 1 本あたり ①③ 各 1 本。
 *   gemma4-qat E4B の行はこの手筋で導いた（同じ計算が E2B の既知値 275 / 2・106・275・70 + 70・
 *   35 層を再現することを確かめてから、M 1 / 4 / 8〈prefill の RMS→add は M 32 / 768〉で読んだ値）。
 *
 * ## 書かないもの（未導出 — 行 / 相 / 席を置かない）
 *
 * - prefill の並列 GEMV・states 形 attention: 物理 chunk 行数（バケット）に依って本数が変わる。
 *   prefill に書くのは M に依らない融合（RMS→add）だけ。
 * - `stateAttentionReduce: "parallel-fused"`: 融合 kernel の採否が full 層の列上限（= 容量）で決まる。
 * - `attentionCompute` / `attentionScoreStorage`: 非 states 形 attention の行ブロック枚数が device の
 *   `maxStorageBufferBindingSize` で決まる（anima DiT 1024px の S は 1 枚に収まらない）。
 *
 * NOTE: 相 `verify` の値は投機の検証行数 M = k + 1 ≤ 8 の前提（並列 GEMV の実測表と ①' の適用
 * 範囲が M ≤ 8）。k ≥ 8 の投機でこの表を引くと並列 GEMV が 0 本になり赤になる。
 * NOTE: census は**計画上の**本数（`SessionDiagnostics.lastRunPipelines`）で、states 形の仕事量ゼロの
 * dispatch も 1 本と数える。ここに書く states 形の本数は M ≤ 8 の decode / verify / draft だけで、
 * 行ブロックが 1 枚なので計画と発行の差は出ない。
 */

import type { SessionSpec } from "@karume/hub";
import type { SessionDiagnostics, SessionOptions } from "@karume/runtime";
import { linearKey } from "../../../runtime/src/kernels/linear.ts";
import { linearGemvKey, linearGemvParallelKey } from "../../../runtime/src/kernels/linear-gemv.ts";
import { statePvKey, stateQkKey } from "../../../runtime/src/kernels/state-attention.ts";
import {
  assertPipelineCensus,
  type CensusExpectation,
  type KeyPredicate,
  type NumericSeat,
  SEAT_REFERENCE,
  SEAT_SIGNATURES,
} from "../../../runtime/tests/helpers/pipeline-census.ts";
import { ANIMA_SESSION_POLICY } from "../../src/anima/pipeline.ts";
import { BIREFNET_SESSION_POLICY } from "../../src/birefnet/pipeline.ts";
import { DEPTH_ANYTHING_SESSION_POLICY } from "../../src/depth-anything/pipeline.ts";
import { GEMMA_SESSION_POLICY } from "../../src/gemma/session-options.ts";
import { IRODORI_SESSION_POLICY } from "../../src/irodori/admission.ts";
import { SBV2_SESSION_POLICY } from "../../src/sbv2/pipeline.ts";
import {
  type FamilySessionPolicy,
  resolveSessionOptions,
  type SessionOverrides,
} from "../../src/session/options.ts";
import { SIGLIP2_SESSION_POLICY } from "../../src/siglip2/pipeline.ts";
import { VOWEL_DETECTOR_SESSION_POLICY } from "../../src/vowel-detector/pipeline.ts";
import { WAN_SESSION_POLICY } from "../../src/wan/family.ts";

// ---------------------------------------------------------------------------
// 鍵
// ---------------------------------------------------------------------------

/**
 * umT5 の encoder（配布形 `karume-umt5-xxl` — pipeline `umt5-encoder`）の受理表。単体の公開クラスは無く、
 * Wan の text 段が Session を `{}` で張る（`src/wan` の text 段 — quant 席の `session` を読まない）。だから
 * 受けるキーは 1 つも無い: 配布形が `session` に何かを宣言したら、実行時に黙って無視される宣言として
 * ここで落とす。
 */
export const UMT5_ENCODER_SESSION_POLICY: FamilySessionPolicy = {
  linearCompute: false,
  attentionCompute: false,
  attentionScoreStorage: false,
  linearGemvReduce: false,
  stateAttentionReduce: false,
  fuseRmsNormAdd: false,
  fuseLinearStaticQuantize: false,
  packedStaticQuantize: false,
};

/**
 * manifest の pipeline 名 → 家族の受理表（実効 SessionOptions の合成に使う）。
 *
 * MUST: 未知の pipeline 名は {@link effectiveSessionOptions} が落とす — 新しい家族のミラーを
 * 足した日に、census の門が黙ってその家族を外さないように。
 */
export const SESSION_POLICIES: Readonly<Record<string, FamilySessionPolicy>> = {
  anima: ANIMA_SESSION_POLICY,
  birefnet: BIREFNET_SESSION_POLICY,
  "depth-anything": DEPTH_ANYTHING_SESSION_POLICY,
  gemma4: GEMMA_SESSION_POLICY,
  "gemma4-qat": GEMMA_SESSION_POLICY,
  irodori: IRODORI_SESSION_POLICY,
  sbv2: SBV2_SESSION_POLICY,
  siglip2: SIGLIP2_SESSION_POLICY,
  "umt5-encoder": UMT5_ENCODER_SESSION_POLICY,
  "vowel-detector": VOWEL_DETECTOR_SESSION_POLICY,
  wan: WAN_SESSION_POLICY,
};

/**
 * quant 席の宣言と明示指定を**家族の合成**（`resolveSessionOptions` — 入口が実際に呼ぶ 1 本）に
 * 通した実効 SessionOptions。表の鍵はこれから作る。
 */
export const effectiveSessionOptions = (
  family: string,
  declared: SessionSpec,
  overrides: SessionOverrides = {},
  where = `census: ${family}`,
): SessionOptions => {
  if (!Object.hasOwn(SESSION_POLICIES, family)) {
    throw new Error(
      `${where}: pipeline '${family}' の受理表が census 表に無い（SESSION_POLICIES へ足す）`,
    );
  }
  return resolveSessionOptions(SESSION_POLICIES[family], declared, overrides, where);
};

/** 網羅表の席名（`Object.keys` は string[] を返すので、網羅表の型から取り直す）。 */
const NUMERIC_SEATS = (Object.keys(SEAT_REFERENCE) as NumericSeat[]).toSorted();

/**
 * 実効 SessionOptions → 束の鍵（数値を変える席のうち**非参照値**の欄だけを、席名の辞書順に
 * `席=値` で `,` 連結）。全席が参照値（= 参照の束）なら空文字列。
 *
 * NOTE: 数値を変えない欄（`submitPolicy` など — pipeline-census.ts の `NonNumericField`）は
 * census に効かないので鍵に入れない。
 */
export const bundleKey = (options: SessionOptions): string =>
  NUMERIC_SEATS.flatMap((seat) => {
    const value = options[seat];
    return value === undefined || value === SEAT_REFERENCE[seat] ? [] : [`${seat}=${value}`];
  }).join(",");

// ---------------------------------------------------------------------------
// 参照変種の述語
// ---------------------------------------------------------------------------

/** キーの族（先頭の `:` より前）。 */
const familyOf = (key: string): string => {
  const colon = key.indexOf(":");
  return colon < 0 ? key : key.slice(0, colon);
};

/** linear op の dispatch が載る族（既定 GEMM 骨格・逐次 GEMV〈M=1 / 行ブロック〉・並列 GEMV）。 */
const LINEAR_OP_FAMILIES: ReadonlySet<string> = new Set(
  [linearKey("f32", true), linearGemvKey("i8"), linearGemvParallelKey("i8", undefined, 4)].map(
    familyOf,
  ),
);

/** 席の非参照値のどれかの述語に当たるか。 */
const anyVariant = (predicates: Readonly<Record<string, KeyPredicate>>): KeyPredicate => (key) =>
  Object.values(predicates).some((predicate) => predicate(key));

/** 逐次の ①QK / ③PV（states 形の参照経路）の全キー（`(sliding, gqa)` の 4 通り）。 */
const STATE_SEQUENTIAL_KEYS: ReadonlySet<string> = new Set(
  [stateQkKey, statePvKey].flatMap((key) =>
    [false, true].flatMap((sliding) => [false, true].map((gqa) => key(sliding, gqa)))
  ),
);

/**
 * 参照変種（席が非参照値なのに、その席の参照経路に残った dispatch）の述語を持つ席。
 *
 * 完全適格なら 0 本（ADR 0110 決定 5 ①）。0 でない束は、対象外の形が逐次へ落ちる本数を
 * そのまま書く（適格表から形が外れると増える側に動く — Fable レビュー I8-02）。
 * 真偽の融合席は「融合しなかったノード」が構造上いつも残るので持たない（変種の本数だけで
 * 外れは見える）。
 */
type ReferenceSeat = "linearCompute" | "linearGemvReduce" | "stateAttentionReduce";

const REFERENCE_SIGNATURES: { readonly [Seat in ReferenceSeat]: KeyPredicate } = {
  // linear op の dispatch のうち、linearCompute のどの変種（i8a8 / c16）にも当たらないもの。
  linearCompute: (key) =>
    LINEAR_OP_FAMILIES.has(familyOf(key)) && !anyVariant(SEAT_SIGNATURES.linearCompute)(key),
  // linear op の dispatch のうち、並列 GEMV 族に落ちなかったもの（逐次 GEMV・既定 GEMM）。
  linearGemvReduce: (key) =>
    LINEAR_OP_FAMILIES.has(familyOf(key)) && !anyVariant(SEAT_SIGNATURES.linearGemvReduce)(key),
  // 逐次の ①QK / ③PV（行タイル変種 ①ₜ③ₜ は席に依らない既定なので含めない）。
  stateAttentionReduce: (key) => STATE_SEQUENTIAL_KEYS.has(key),
};

// ---------------------------------------------------------------------------
// 表の型
// ---------------------------------------------------------------------------

/** 席 1 つぶんの期待本数（run 1 回あたり・ちょうど）。 */
type SeatCount<Seat extends NumericSeat> = Seat extends ReferenceSeat ? {
    /** 束が指定した非参照値の変種キーの本数。 */
    readonly variant: number;
    /** その席の参照経路に残ったキーの本数（完全適格なら 0）。 */
    readonly reference: number;
  }
  : { readonly variant: number };

/**
 * 相 1 つぶんの期待（束の非参照値の席だけを書く — 値は束の `session` が決める）。
 *
 * NOTE: 書いていない席（未導出）はこの相では本数を見ない。e2e はその席を `assertSeatsApplied`
 * の 1 本以上 / 0 本で見続ける。
 */
export type PhaseCensus = { readonly [Seat in NumericSeat]?: SeatCount<Seat> };

/** 系列ごとの部品 → 相（`onRunDiagnostics` が名乗る部品名・相の綴り）。 */
type FamilyAxes = {
  readonly gemma4: GemmaAxes;
  readonly "gemma4-qat": GemmaAxes;
  readonly irodori: { readonly dit: "step"; readonly "dit-context": "run" };
  readonly anima: { readonly transformer: "step" };
  readonly sbv2: { readonly front: "run"; readonly voice: "run" };
  /** 相 `pass` は DiT の 1 forward（CFG の 1 ステップは条件つき / 条件なしの 2 パス）。 */
  readonly wan: { readonly transformer: "pass" };
};

/** gemma の部品（製品グラフ = target・投機の drafter）と相（`GenerationRunPhase["kind"]`）。 */
type GemmaAxes = {
  readonly target: "prefill" | "decode" | "verify";
  readonly drafter: "draft";
};

/** 表に行を持つ系列（manifest の pipeline 名）。 */
export type CensusFamily = keyof FamilyAxes;

type ComponentCensus<Family extends CensusFamily> = {
  readonly [Component in keyof FamilyAxes[Family]]?: {
    readonly [Phase in FamilyAxes[Family][Component] & string]?: PhaseCensus;
  };
};

/** 表の 1 行（系列 × モデル群 × 束）。 */
type FamilyBundleRow<Family extends CensusFamily> = {
  readonly family: Family;
  /** 同じグラフ（同じ本数）を持つモデル（manifest の models のキー）。 */
  readonly models: readonly string[];
  /** 束（実効 SessionOptions — 鍵は {@link bundleKey}）。 */
  readonly session: SessionOverrides;
  readonly census: ComponentCensus<Family>;
};

export type BundleCensusRow = { readonly [Family in CensusFamily]: FamilyBundleRow<Family> }[
  CensusFamily
];

// ---------------------------------------------------------------------------
// 表
// ---------------------------------------------------------------------------

/** gemma4 E2B（製品グラフ）: linear 277 本（i4 276 + lm_head i8 1）が M ≤ 8 で全て並列 GEMV の行に載る。 */
const GEMMA4_E2B_GEMV = { variant: 277, reference: 0 } as const;
/**
 * gemma4 E2B の drafter: linear 68 本（全て i8）のうち並列 GEMV の実測表に載るのは 1536×256 の
 * 2 本だけ — 残り 66 本（256×{1024,2048,3072}・{1024,2048}×256・lm_head 262144×256）は逐次 GEMV へ
 * 落ちる（実測表が drafter の形を持たない。census は現状の本数をそのまま固定する）。
 */
const GEMMA4_E2B_DRAFTER_GEMV = { variant: 2, reference: 66 } as const;
/**
 * gemma4-qat E2B（製品グラフ）: linear 277 本のうち並列 GEMV は 275 本（i2 60 + i4 145 + i8 70）。
 * 残り 2 本は逐次へ落ちる — i2 の lm_head 262144×1536（実測表に無い形）と f32 格納の
 * per_layer_model_projection 8960×1536。
 */
const GEMMA4_QAT_E2B_GEMV = { variant: 275, reference: 2 } as const;
/** RMS→add 融合（M に依らない — `assets_fusion_counts_test.ts` が M 1〜64 で固定）。 */
const GEMMA4_E2B_RMS_ADD = { variant: 106 } as const;
const GEMMA4_E2B_DRAFTER_RMS_ADD = { variant: 24 } as const;
/** linear→SRQ 融合（並列 GEMV に落ちる 275 本の全て — M ≤ 8）。 */
const GEMMA4_QAT_E2B_LINEAR_SRQ = { variant: 275 } as const;
/**
 * packed int8 活性（ADR 0105）: 生産側の単体 SRQ（packed 変種）70 本 + それを読む並列 GEMV の
 * packed 変種 70 本（linear→SRQ 融合の有無で消費側の族は変わるが、どちらも packed 断片を持つ）。
 */
const GEMMA4_QAT_E2B_PACKED = { variant: 140 } as const;
/** states 形 attention: 製品グラフの 35 層 × ①' / ③' 各 1 本（M ≤ 8・行ブロック 1 枚）。 */
const GEMMA4_E2B_STATE_PARALLEL = { variant: 70, reference: 0 } as const;
/** drafter の readonly attention 12 本 × ①' / ③' 各 1 本（readonly は席に依らず常に並列形）。 */
const GEMMA4_E2B_DRAFTER_STATE_PARALLEL = { variant: 24, reference: 0 } as const;

/**
 * gemma4-qat E4B（製品グラフ・42 層）: linear 344 本のうち並列 GEMV は 342 本 — 形ごとの内訳は
 * runtime `gpu_linear_gemv_parallel_census_test.ts` の QAT E4B の describe が固定している。残り 2 本は
 * E2B と同じ機序（i2 の lm_head と f32 格納の per_layer_model_projection）で逐次へ落ちる。
 */
const GEMMA4_QAT_E4B_GEMV = { variant: 342, reference: 2 } as const;
/** RMS→add 融合（M 1〜8 の decode 計画と M 32 / 768 の prefill 計画で同じ本数）。 */
const GEMMA4_QAT_E4B_RMS_ADD = { variant: 127 } as const;
/** linear→SRQ 融合（並列 GEMV に落ちる 342 本の全て — M ≤ 8）。 */
const GEMMA4_QAT_E4B_LINEAR_SRQ = { variant: 342 } as const;
/**
 * packed int8 活性: 生産側の単体 SRQ（packed 変種）84 本 + それを読む並列 GEMV の packed 変種 84 本
 * （o 42 + down 42 — g2048 / g4096 の l32。E2B の 70 + 70 と同じ内訳の取り方）。
 */
const GEMMA4_QAT_E4B_PACKED = { variant: 168 } as const;
/** states 形 attention: 製品グラフの 42 層 × ①' / ③' 各 1 本（M ≤ 8・行ブロック 1 枚）。 */
const GEMMA4_QAT_E4B_STATE_PARALLEL = { variant: 84, reference: 0 } as const;

const E2B = ["e2b"] as const;
const E4B = ["e4b"] as const;

/**
 * irodori v4 系の DiT: linear 245 本（分割前の 317 本の k ∈ {32, 192, 512, 768, 1280, 3680} — 全て
 * i8 × k % 4 == 0）。
 *
 * 期待値の変更（317 → 245 / `dit_context` 72）はグラフが変わったため（ADR 0114 — 条件側 K/V
 * 射影 72 本〈12 ブロック × text / speaker / caption × K / V〉を `dit_context` へ割り出した）。
 * 本数を緩めたのではない。合計 317 の検査は置かない（部品ごとにちょうどで縛れば足り、合計は
 * 片方の取り違えを打ち消しうる）。
 */
const IRODORI_DIT_A8 = { variant: 245, reference: 0 } as const;
/** irodori v4 系の `dit_context`: 条件側 K/V 射影の linear 72 本（k ∈ {512, 768} — 生成 1 回に run 1 回）。 */
const IRODORI_DIT_CONTEXT_A8 = { variant: 72, reference: 0 } as const;
/** anima DiT: linear 454 本（公式 5 変種・追加 2 変種とも同じグラフ）。 */
const ANIMA_DIT_LINEARS = { variant: 454, reference: 0 } as const;
const ANIMA_MODELS = [
  "anima-turbo-v1.1",
  "anima-v1.0",
  "anima-aesthetic-v1.1",
  "anima-turbo-v1.0",
  "anima-aesthetic-v1.0",
  "anima-wai-v1.0",
  "anima-copycat-20260610",
] as const;
const SBV2_MODELS = ["F1", "F2", "M1", "M2"] as const;
/**
 * Wan2.1 T2V 1.3B の DiT: linear 307 本（k ∈ {64, 256, 1536, 4096, 8960} — 全て i8 × k % 4 == 0。
 * M = 1 の時刻 MLP と `time_proj` も含む — ADR 0120 決定 2・調査 §2.2）。
 */
const WAN_DIT_LINEARS = { variant: 307, reference: 0 } as const;

/**
 * 束の census 表。
 *
 * MUST: 数値を変える非参照値を 1 つでも持つ配布ミラーの quant 席は、ここに行が要る
 * （`census_table_test.ts` が GPU 無しで落とす — census 無しで束が増えるのを防ぐ）。
 * 行の中身（相・席）は導出できた分だけ（未導出の相は書かない — ファイル冒頭）。
 */
export const CENSUS_TABLE: readonly BundleCensusRow[] = [
  // --- gemma4 E2B -----------------------------------------------------------
  {
    // e2e_gemma4_quant_test の `i4-gemvpar` 差し替え席 / 明示 sequential で ③ を戻した `i4-gemvpar`。
    family: "gemma4",
    models: E2B,
    session: { linearGemvReduce: "parallel" },
    census: {
      target: {
        decode: { linearGemvReduce: GEMMA4_E2B_GEMV },
        verify: { linearGemvReduce: GEMMA4_E2B_GEMV },
      },
      drafter: { draft: { linearGemvReduce: GEMMA4_E2B_DRAFTER_GEMV } },
    },
  },
  {
    // e2e_gemma4_quant_test の `i4-fast` 差し替え席（投機の検査は ③ を明示 sequential へ倒す）。
    family: "gemma4",
    models: E2B,
    session: { linearGemvReduce: "parallel", fuseRmsNormAdd: true },
    census: {
      target: {
        prefill: { fuseRmsNormAdd: GEMMA4_E2B_RMS_ADD },
        decode: { linearGemvReduce: GEMMA4_E2B_GEMV, fuseRmsNormAdd: GEMMA4_E2B_RMS_ADD },
        verify: { linearGemvReduce: GEMMA4_E2B_GEMV, fuseRmsNormAdd: GEMMA4_E2B_RMS_ADD },
      },
      drafter: {
        draft: {
          linearGemvReduce: GEMMA4_E2B_DRAFTER_GEMV,
          fuseRmsNormAdd: GEMMA4_E2B_DRAFTER_RMS_ADD,
        },
      },
    },
  },
  {
    // 配布ミラーの `i4-gemvpar`。
    family: "gemma4",
    models: E2B,
    session: { linearGemvReduce: "parallel", stateAttentionReduce: "parallel" },
    census: {
      target: {
        decode: {
          linearGemvReduce: GEMMA4_E2B_GEMV,
          stateAttentionReduce: GEMMA4_E2B_STATE_PARALLEL,
        },
        verify: {
          linearGemvReduce: GEMMA4_E2B_GEMV,
          stateAttentionReduce: GEMMA4_E2B_STATE_PARALLEL,
        },
      },
      drafter: {
        draft: {
          linearGemvReduce: GEMMA4_E2B_DRAFTER_GEMV,
          stateAttentionReduce: GEMMA4_E2B_DRAFTER_STATE_PARALLEL,
        },
      },
    },
  },
  {
    // 配布ミラーの `i4-fast`（③ の parallel-fused は未導出 — ファイル冒頭）。
    family: "gemma4",
    models: E2B,
    session: {
      linearGemvReduce: "parallel",
      fuseRmsNormAdd: true,
      stateAttentionReduce: "parallel-fused",
    },
    census: {
      target: {
        prefill: { fuseRmsNormAdd: GEMMA4_E2B_RMS_ADD },
        decode: { linearGemvReduce: GEMMA4_E2B_GEMV, fuseRmsNormAdd: GEMMA4_E2B_RMS_ADD },
        verify: { linearGemvReduce: GEMMA4_E2B_GEMV, fuseRmsNormAdd: GEMMA4_E2B_RMS_ADD },
      },
      drafter: {
        draft: {
          linearGemvReduce: GEMMA4_E2B_DRAFTER_GEMV,
          fuseRmsNormAdd: GEMMA4_E2B_DRAFTER_RMS_ADD,
        },
      },
    },
  },
  // --- gemma4-qat E2B（drafter を持たない）----------------------------------
  {
    // e2e_gemma4_quant_test の `i4-gemvpar` 差し替え席。
    family: "gemma4-qat",
    models: E2B,
    session: { linearGemvReduce: "parallel" },
    census: {
      target: {
        decode: { linearGemvReduce: GEMMA4_QAT_E2B_GEMV },
        verify: { linearGemvReduce: GEMMA4_QAT_E2B_GEMV },
      },
    },
  },
  {
    // 配布ミラーの `i4-gemvpar`。
    family: "gemma4-qat",
    models: E2B,
    session: { linearGemvReduce: "parallel", stateAttentionReduce: "parallel" },
    census: {
      target: {
        decode: {
          linearGemvReduce: GEMMA4_QAT_E2B_GEMV,
          stateAttentionReduce: GEMMA4_E2B_STATE_PARALLEL,
        },
        verify: {
          linearGemvReduce: GEMMA4_QAT_E2B_GEMV,
          stateAttentionReduce: GEMMA4_E2B_STATE_PARALLEL,
        },
      },
    },
  },
  {
    // 配布ミラーの `i4-fast`（③ の parallel-fused は未導出 — ファイル冒頭）。
    family: "gemma4-qat",
    models: E2B,
    session: {
      linearGemvReduce: "parallel",
      fuseRmsNormAdd: true,
      fuseLinearStaticQuantize: true,
      packedStaticQuantize: true,
      stateAttentionReduce: "parallel-fused",
    },
    census: {
      target: {
        prefill: { fuseRmsNormAdd: GEMMA4_E2B_RMS_ADD },
        decode: {
          linearGemvReduce: GEMMA4_QAT_E2B_GEMV,
          fuseRmsNormAdd: GEMMA4_E2B_RMS_ADD,
          fuseLinearStaticQuantize: GEMMA4_QAT_E2B_LINEAR_SRQ,
          packedStaticQuantize: GEMMA4_QAT_E2B_PACKED,
        },
        verify: {
          linearGemvReduce: GEMMA4_QAT_E2B_GEMV,
          fuseRmsNormAdd: GEMMA4_E2B_RMS_ADD,
          fuseLinearStaticQuantize: GEMMA4_QAT_E2B_LINEAR_SRQ,
          packedStaticQuantize: GEMMA4_QAT_E2B_PACKED,
        },
      },
    },
  },
  {
    // e2e_gemma4_quant_test の `i4-fast` 差し替え席（既定 / explicit-fast）。
    family: "gemma4-qat",
    models: E2B,
    session: {
      linearGemvReduce: "parallel",
      fuseRmsNormAdd: true,
      fuseLinearStaticQuantize: true,
      packedStaticQuantize: true,
    },
    census: {
      target: {
        prefill: { fuseRmsNormAdd: GEMMA4_E2B_RMS_ADD },
        decode: {
          linearGemvReduce: GEMMA4_QAT_E2B_GEMV,
          fuseRmsNormAdd: GEMMA4_E2B_RMS_ADD,
          fuseLinearStaticQuantize: GEMMA4_QAT_E2B_LINEAR_SRQ,
          packedStaticQuantize: GEMMA4_QAT_E2B_PACKED,
        },
      },
    },
  },
  {
    // e2e_gemma4_quant_test の disable-rms。
    family: "gemma4-qat",
    models: E2B,
    session: {
      linearGemvReduce: "parallel",
      fuseLinearStaticQuantize: true,
      packedStaticQuantize: true,
    },
    census: {
      target: {
        decode: {
          linearGemvReduce: GEMMA4_QAT_E2B_GEMV,
          fuseLinearStaticQuantize: GEMMA4_QAT_E2B_LINEAR_SRQ,
          packedStaticQuantize: GEMMA4_QAT_E2B_PACKED,
        },
      },
    },
  },
  {
    // e2e_gemma4_quant_test の disable-srq（packed の消費側は素の並列 GEMV の packed 変種）。
    family: "gemma4-qat",
    models: E2B,
    session: { linearGemvReduce: "parallel", fuseRmsNormAdd: true, packedStaticQuantize: true },
    census: {
      target: {
        prefill: { fuseRmsNormAdd: GEMMA4_E2B_RMS_ADD },
        decode: {
          linearGemvReduce: GEMMA4_QAT_E2B_GEMV,
          fuseRmsNormAdd: GEMMA4_E2B_RMS_ADD,
          packedStaticQuantize: GEMMA4_QAT_E2B_PACKED,
        },
      },
    },
  },
  {
    // e2e_gemma4_quant_test の disable-both。
    family: "gemma4-qat",
    models: E2B,
    session: { linearGemvReduce: "parallel", packedStaticQuantize: true },
    census: {
      target: {
        decode: {
          linearGemvReduce: GEMMA4_QAT_E2B_GEMV,
          packedStaticQuantize: GEMMA4_QAT_E2B_PACKED,
        },
      },
    },
  },
  // --- gemma4-qat E4B（drafter を持たない）----------------------------------
  {
    // 配布ミラーの `i4-gemvpar`。
    family: "gemma4-qat",
    models: E4B,
    session: { linearGemvReduce: "parallel", stateAttentionReduce: "parallel" },
    census: {
      target: {
        decode: {
          linearGemvReduce: GEMMA4_QAT_E4B_GEMV,
          stateAttentionReduce: GEMMA4_QAT_E4B_STATE_PARALLEL,
        },
        verify: {
          linearGemvReduce: GEMMA4_QAT_E4B_GEMV,
          stateAttentionReduce: GEMMA4_QAT_E4B_STATE_PARALLEL,
        },
      },
    },
  },
  {
    // 配布ミラーの `i4-fast`（③ の parallel-fused は未導出 — ファイル冒頭）。
    family: "gemma4-qat",
    models: E4B,
    session: {
      linearGemvReduce: "parallel",
      fuseRmsNormAdd: true,
      fuseLinearStaticQuantize: true,
      packedStaticQuantize: true,
      stateAttentionReduce: "parallel-fused",
    },
    census: {
      target: {
        prefill: { fuseRmsNormAdd: GEMMA4_QAT_E4B_RMS_ADD },
        decode: {
          linearGemvReduce: GEMMA4_QAT_E4B_GEMV,
          fuseRmsNormAdd: GEMMA4_QAT_E4B_RMS_ADD,
          fuseLinearStaticQuantize: GEMMA4_QAT_E4B_LINEAR_SRQ,
          packedStaticQuantize: GEMMA4_QAT_E4B_PACKED,
        },
        verify: {
          linearGemvReduce: GEMMA4_QAT_E4B_GEMV,
          fuseRmsNormAdd: GEMMA4_QAT_E4B_RMS_ADD,
          fuseLinearStaticQuantize: GEMMA4_QAT_E4B_LINEAR_SRQ,
          packedStaticQuantize: GEMMA4_QAT_E4B_PACKED,
        },
      },
    },
  },
  // --- irodori（quant の session は dit と dit_context にだけ渡る）-----------
  {
    family: "irodori",
    models: ["v4-small", "v4.1-small"],
    session: { linearCompute: "a8" },
    census: {
      dit: { step: { linearCompute: IRODORI_DIT_A8 } },
      "dit-context": { run: { linearCompute: IRODORI_DIT_CONTEXT_A8 } },
    },
  },
  // --- anima（quant の session は DiT = transformer にだけ渡る）---------------
  {
    family: "anima",
    models: ANIMA_MODELS,
    session: { linearCompute: "a8" },
    census: { transformer: { step: { linearCompute: ANIMA_DIT_LINEARS } } },
  },
  {
    // attentionCompute は未導出（ファイル冒頭）。
    family: "anima",
    models: ANIMA_MODELS,
    session: { linearCompute: "a8", attentionCompute: "a8" },
    census: { transformer: { step: { linearCompute: ANIMA_DIT_LINEARS } } },
  },
  {
    // attentionCompute / attentionScoreStorage は未導出（ファイル冒頭）。
    family: "anima",
    models: ANIMA_MODELS,
    session: { linearCompute: "a8", attentionCompute: "a8", attentionScoreStorage: "f16" },
    census: { transformer: { step: { linearCompute: ANIMA_DIT_LINEARS } } },
  },
  {
    // f16 格納の linear は全て c16 の GEMM 骨格へ（M=1 の GEMV は f32 計算だけ）。
    // attentionCompute は未導出（ファイル冒頭）。
    family: "anima",
    models: ANIMA_MODELS,
    session: { linearCompute: "f16", attentionCompute: "f16" },
    census: { transformer: { step: { linearCompute: ANIMA_DIT_LINEARS } } },
  },
  // --- sbv2（quant の session は front / voice にだけ渡る）---------------------
  {
    // front は linear 2 本・voice は 4 本（残りは conv1d で linearCompute の対象外）。
    family: "sbv2",
    models: SBV2_MODELS,
    session: { linearCompute: "a8" },
    census: {
      front: { run: { linearCompute: { variant: 2, reference: 0 } } },
      voice: { run: { linearCompute: { variant: 4, reference: 0 } } },
    },
  },
  // --- wan（quant の session は DiT = transformer にだけ渡る）-----------------
  {
    // 配布ミラーの実用席 `f16+dit8-a8-attn8-s16`（ADR 0120 決定 4）。
    // attentionCompute / attentionScoreStorage は未導出（ファイル冒頭）。
    family: "wan",
    models: ["t2v-1.3b"],
    session: { linearCompute: "a8", attentionCompute: "a8", attentionScoreStorage: "f16" },
    census: { transformer: { pass: { linearCompute: WAN_DIT_LINEARS } } },
  },
];

// ---------------------------------------------------------------------------
// 引く・検査する
// ---------------------------------------------------------------------------

/** (系列, モデル, 実効 SessionOptions) の行。無ければ `undefined`（= census は 1 本以上 / 0 本）。 */
export const censusRowOf = (
  family: string,
  model: string,
  effective: SessionOptions,
): BundleCensusRow | undefined => {
  const key = bundleKey(effective);
  return CENSUS_TABLE.find((row) =>
    row.family === family && row.models.includes(model) && bundleKey(row.session) === key
  );
};

/** 行の相の表（部品・相を文字列で引くため — 型は行の系列で決まる）。 */
const phaseOf = (
  row: BundleCensusRow,
  component: string,
  phase: string,
): PhaseCensus | undefined => {
  const census: Readonly<Record<string, Readonly<Record<string, PhaseCensus>> | undefined>> =
    row.census;
  const phases = Object.hasOwn(census, component) ? census[component] : undefined;
  return phases !== undefined && Object.hasOwn(phases, phase) ? phases[phase] : undefined;
};

/**
 * 行の (部品, 相) の期待列。相が未記入なら `undefined`。
 *
 * 席ごとに「非参照値の変種キーが `variant` 本ちょうど」と、参照変種の述語を持つ席では
 * 「その席の参照経路に残ったキーが `reference` 本ちょうど」の 2 件を出す。
 */
export const phaseExpectations = (
  row: BundleCensusRow,
  component: string,
  phase: string,
): readonly CensusExpectation[] | undefined => {
  const census = phaseOf(row, component, phase);
  if (census === undefined) return undefined;
  return NUMERIC_SEATS.flatMap((seat): CensusExpectation[] => {
    const count: { readonly variant: number; readonly reference?: number } | undefined =
      census[seat];
    if (count === undefined) return [];
    const value = row.session[seat as keyof SessionOverrides];
    const signatures: Readonly<Record<string, KeyPredicate>> = SEAT_SIGNATURES[seat];
    const predicate = value === undefined ? undefined : signatures[String(value)];
    if (predicate === undefined) {
      throw new Error(
        `census 表: ${row.family} [${bundleKey(row.session)}] の ${component}/${phase} が束に` +
          `無い（または参照値の）席 ${seat} の本数を持っている`,
      );
    }
    const variant: CensusExpectation = {
      label: `${seat}=${String(value)} の変種`,
      match: predicate,
      count: count.variant,
    };
    if (count.reference === undefined) return [variant];
    const reference = (REFERENCE_SIGNATURES as Readonly<Record<string, KeyPredicate>>)[seat];
    return [variant, {
      label: `${seat} の参照経路に残った dispatch`,
      match: reference,
      count: count.reference,
    }];
  });
};

/**
 * 相ごとの run の census を、行の期待と**run 1 回ずつ**突き合わせる（本数は run 1 回あたり）。
 *
 * 返り値は検査した run の数。MUST: 呼び手は 0 を通さない — 行はあるのに期待を持つ相が 1 度も
 * 走っていなければ、ちょうどの検査は 1 つも行われていない。
 *
 * @param runs 相 → その相の run の `lastRunPipelines` 列。
 * @param componentOf 相 → 部品（gemma は draft だけが drafter）。
 */
export const assertRowCensus = (
  row: BundleCensusRow,
  runs: ReadonlyMap<string, readonly SessionDiagnostics["lastRunPipelines"][]>,
  componentOf: (phase: string) => string,
  where: string,
): number => {
  let checked = 0;
  for (const [phase, phaseRuns] of runs) {
    const component = componentOf(phase);
    const expected = phaseExpectations(row, component, phase);
    if (expected === undefined) continue;
    phaseRuns.forEach((census, index) => {
      assertPipelineCensus(census, expected, `${where} ${component}/${phase} #${index}`);
      checked++;
    });
  }
  return checked;
};

/** gemma の相 → 部品（投機の draft だけが drafter の Session・他は製品グラフ）。 */
export const gemmaComponentOf = (phase: string): string => phase === "draft" ? "drafter" : "target";
