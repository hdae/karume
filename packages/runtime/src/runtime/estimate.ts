/**
 * メモリ必要量 estimator（ADR 0070 決定 5）— GPU に一切触れない純関数で「必要側」のバイト数を
 * カテゴリ別に出す。
 *
 * MUST: **空き側との比較はしない**。WebGPU は総 / 空き VRAM を露出しないので、比較の形にした
 * 瞬間に「予算が取れない環境での当て推量」になる（ADR 0070 決定 5 — llama.cpp の 0/0 デバイス
 * 除外と同型の前例）。ここが返すのは数字だけで、可否の判定は一切しない。
 * MUST: 見積りは**絶対保証ではない**。勘定に入っていないものは
 * {@link AdmissionReport.unaccounted} が形式として認め、実際の**最終門は out-of-memory
 * errorScope**（src/gpu/device.ts の 2 本組）のまま — estimator は Session 構築 /
 * `createGenerationContext` の事前診断であって、超えていても実行は止めない。
 *
 * 報告の形は**常駐 + シナリオ**の 2 段（ADR 0070 決定 5 の再具体化）。Session が寿命を通じて
 * 抱えるもの（重み・state スロット）は {@link AdmissionReport.resident} の 1 組で、run の形
 * （prefill / decode）ごとに変わるものは {@link AdmissionReport.scenarios} に**別々の計算**
 * として並ぶ。同じ数字を 1 本に潰すと、prefill 形でしか出ない中間の山を decode しか回さない
 * 呼び手に押しつけるか、その逆に decode の数字で prefill を通すかのどちらかになる。
 * 2 形のどちらでもない形（投機デコードの verify run — ADR 0096）は、呼び手が
 * {@link AdmissionScenarioSpec} で**同じ計算経路に**足す（式を呼び手側で組み直させない）。
 *
 * 実測（{@link SessionDiagnostics}）との対応:
 * - {@link AdmissionReport.resident}`.weights.compressedBytes` ↔ `storage.residentCompressedBytes`
 * - {@link AdmissionReport.resident}`.weights.expandedBytes` ↔ `storage.hostExpandedBytes`
 * - {@link AdmissionReport.resident}`.weights.uncompressedBytes` — 対応する診断欄は無い
 *   （ADR 0006 の `storage` 診断は低精度格納の実績だけを数える。実測の総量は
 *   `weights.allocatedBytes` — params 込み — でしか観測できない）
 * - {@link AdmissionReport.resident}`.stateBytes` ↔ `stateBacking.residentBytes`（context 1 本のとき）
 * - {@link AdmissionScenario.workspaceBytes} ↔ `planBacking.residentBytes` /
 *   `lastRun.peakTransientBytes`（そのシナリオの形で回した run のもの — 診断の
 *   `residentBytes` は**保持中の backing 全ての総和**なので、対応するのは 1 形だけを回した
 *   Session か `retainedCount === 1` のとき。予算つきの保持は ADR 0095 決定 1）
 *
 * MUST: 分類も算式も**実装と同じ導出元**から引く（重みの席と宣言由来バイト数は
 * weight-residency.ts の純関数プランナ、整列とサイズクラスは `toSizeClass`、state の 1 要素
 * バイト数と論理長 uniform は generation-context.ts の定数、物理 chunk 行 `M` の載る軸は
 * ops/shapes.ts の op 契約）。式をここで書き直すと、片方だけ直された実装に対して estimator が
 * 例外も警告も無く別の数を主張し続ける。
 */

import { parseDim, solveDim } from "../format/dims.ts";
import type { IrDim, IrGraph } from "../format/ir.ts";
import { toSizeClass } from "../gpu/arena.ts";
import { parseStorageRoles } from "../gpu/pipeline-cache.ts";
import { numel, RUNTIME_SUPPORT, stateWindow, topkK } from "../ops.ts";
import { assertRuntimeSupport } from "../ops/support.ts";
import { type FusedStep, planAliases, planFusions, rowBlockLimit } from "./fusion.ts";
import {
  allowedChunkRows,
  assertChunkBuckets,
  assertChunkLength,
  assertSlotWithinLimits,
  LENGTHS_BYTES,
  resolveBindings,
  resolveSlotShape,
  type SlotLimits,
  STATE_ELEMENT_BYTES,
} from "./generation-context.ts";
import {
  assertGenerationBindings,
  countUses,
  describeInputValue,
  ExecutionError,
  type NodePlan,
  planGraph,
  statesOnlySymbols,
  type SymbolBindings,
  validateGraphContracts,
} from "./plan.ts";
import {
  CORE_TRANSIENT_LIMITS,
  planTransients,
  type TransientDispatchSpec,
  type TransientOutputSpec,
  type TransientRef,
  type TransientStepSpec,
  type TransientTempSpec,
} from "./transient-plan.ts";
import {
  DEFAULT_PLAN_BACKING_BUDGET_BYTES,
  type GenerationContextSpec,
  STATE_ATTENTION_REDUCES,
  type StateAttentionReduce,
} from "./session-types.ts";
import {
  argmaxSplitGroups,
  argmaxSplitPartialBytes,
  topkOneSplitGroups,
} from "../kernels/argmax.ts";
import { assertChunkRowsAllowed } from "./recipe.ts";
import { planStateAttention, type StateAttentionBlock } from "./state-attention-plan.ts";
import { planWeightBuffers, type WeightResidency } from "./weight-residency.ts";

/**
 * **予約名**のシナリオ。`generation` を渡さない見積りは `"run"` の 1 本、渡した見積りは
 * ADR 0066 決定 4 の実行 2 形（`"prefill"` / `"decode"`）で、この 3 つは呼び手が
 * {@link AdmissionScenarioSpec.name} に使えない。
 */
export type AdmissionScenarioName = "run" | "prefill" | "decode";

/**
 * 呼び手が足す追加シナリオの仕様（{@link EstimateOptions}`.generation.scenarios`）。
 *
 * 既存 2 形（prefill / decode）と**同じ計算経路**を、物理 chunk 行だけ差し替えて 1 本増やす口。
 * 投機デコードの verify run（`queryLength = k+1` 行を 1 回で流す形 — ADR 0096）のように、
 * ADR 0066 決定 4 の 2 形のどちらでもない形の必要量を、呼び手が式を組み直さずに引くためにある。
 */
export type AdmissionScenarioSpec = {
  /**
   * 報告に載る名前。予約名（{@link AdmissionScenarioName} の 3 つ）と、同じ列の他の追加名との
   * 重複は fail loudly（名前で引く呼び手が黙って先頭だけを見る形を作らない）。
   */
  readonly name: string;
  /**
   * この形の物理 chunk 行 `M`。`1..generation.chunkLength` の整数（外は fail loudly —
   * `queryLength ≤ chunkLength` の契約の外にある行数を名乗らせない）で、さらに context の許す
   * 行数（1・`generation.chunkBuckets`・`generation.chunkLength`）のどれか（外は run が拒否する
   * 形なので、同じく fail loudly）。
   */
  readonly chunkLength: number;
  /**
   * グラフ入力側の記号次元の**上書き**（`options.bindings` の写しに重ねる）。省略すると
   * `options.bindings` そのまま。値域と束縛点の門は既存 2 形と同じ（states 専用記号と物理
   * chunk 行の記号は受けない）。
   */
  readonly bindings?: SymbolBindings;
};

/**
 * run の形 1 つぶんの必要バイト数（{@link AdmissionReport.scenarios} の 1 要素）。
 *
 * ここに入るのは**run の形で変わるものだけ** — 重みと state スロットは形に依らないので
 * {@link AdmissionReport.resident} の側にある。
 */
export type AdmissionScenario = {
  /** 予約名（{@link AdmissionScenarioName}）か、呼び手が足した {@link AdmissionScenarioSpec.name}。 */
  readonly name: string;
  /** グラフ入力バッファ + 出力 readback staging（厳密）。 */
  readonly ioBytes: number;
  /**
   * 中間（transient）slot 表の必要バイト = prepared backing の必要側と同義（**近似**）。
   *
   * 融合前のノード列（行ブロック attention は融合後の形）を宣言順に歩き、実行相と同じ確保規則
   * （消費回数は `countUses`・グラフ出力は pinned で解放しない・別名化できる出力は確保せず
   * 入力の実体を別名で使う）で slot の生存区間を作り、実行相と同じ配置関数（生存区間と offset
   * による配置 — ADR 0093）が割り当てた領域の総和。**states 形 attention の
   * ノード内一時（スコア S と行統計）はここに入る** — 融合の成立に依存せず必ず出て、capacity
   * 律速のグラフでは中間の主役になるため（算式は {@link stateAttentionTemps}）。
   * **行ブロック attention（ADR 0060）の窓は融合後の形で入る** — 実行相と同じ融合計画が掴んだ
   * 窓は、スコア・mask 加算後・softmax 後の全幅ではなく、行ブロック 1 枚ぶんの一時と出力で数える
   * （ADR 0093 追記 2026-10-10）。それ以外のノード内一時と、他の融合が畳んで消す中間は勘定に
   * 入らない（{@link AdmissionReport.unaccounted}）。
   */
  readonly workspaceBytes: number;
};

/** 必要バイト数の報告（{@link estimateGraphMemory} / `PreparedModel.estimate` の戻り）。 */
export type AdmissionReport = {
  /** run の形に依らず Session の寿命ぶん抱えるもの。 */
  readonly resident: {
    readonly weights: {
      /**
       * 圧縮のまま GPU 常駐する重み + companion scale のバイト数（整列詰め物込み — 厳密）。
       * 診断の `storage.residentCompressedBytes` と厳密に対応する（0 要素テンソルの 4 バイト床
       * だけが唯一の差 — 診断側は床なしの実バイトを数える）。
       */
      readonly compressedBytes: number;
      /**
       * 圧縮しない格納（f32 / i32）のまま常駐する重みのバイト数（厳密）。
       *
       * MUST: 圧縮 / 展開と別の欄で持つ — 診断の `storage` は低精度格納の実績だけを数える
       * （ADR 0006）ので、ここを `compressedBytes` に混ぜると「診断との厳密一致」が f32 重みを
       * 持つモデルで沈黙して崩れる。
       */
      readonly uncompressedBytes: number;
      /** 適格外でロード時に f32 展開して常駐する重みの展開後バイト数（厳密）。 */
      readonly expandedBytes: number;
      /** 上記 3 欄の和。 */
      readonly totalBytes: number;
    };
    /**
     * `GenerationContext` 1 本ぶんの state スロット + 論理長 uniform（厳密）。
     *
     * NOTE: `options.generation` を省略すると 0。`graph.states` が非空なグラフでは
     * {@link AdmissionScenario.workspaceBytes} の計画が state スロットの解決済み shape を
     * 要求するので、そもそも generation 無しでは見積れず fail loudly になる（黙って 0 を返す
     * 窓は無い）。
     */
    readonly stateBytes: number;
  };
  /**
   * run の形ごとの必要バイト数（1 要素以上 — 名前の決まり方は {@link AdmissionScenarioName}）。
   * 呼び手が {@link EstimateOptions}`.generation.scenarios` を渡すと、既存 2 形の**後ろ**に
   * 宣言順で並ぶ。
   */
  readonly scenarios: readonly AdmissionScenario[];
  /**
   * この見積りが使った slot backing の保持予算（{@link EstimateOptions.planBackingBudgetBytes} —
   * 省略時は `DEFAULT_PLAN_BACKING_BUDGET_BYTES`）。{@link AdmissionReport.peakAccountedBytes} の
   * 片側の項がそのまま読めるように報告する（同じ数を呼び手が組み直さずに済む）。
   */
  readonly planBackingBudgetBytes: number;
  /**
   * 補助Sessionや常駐入出力の追加勘定。resident / scenarios の内訳とは重複させない。
   * 省略は0。runtimeの単体Session見積りは省略し、modelsが補助資源を持つ場合に加える。
   * DECIDED: docs/decisions/0070-shard-loading-admission.md#補助資源の見積り内訳2026-09-12
   */
  readonly auxiliaryBytes?: number;
  /**
   * 常駐の総和 + **保持集合の上限** + 補助資源（= `resident.weights.totalBytes + resident.stateBytes +
   * max(planBackingBudgetBytes, max(ioBytes + workspaceBytes)) + (auxiliaryBytes ?? 0)` — 内側の
   * max はシナリオとバケット形の全体）。
   *
   * **上限保証ではなく「勘定に入れた分のピーク」**を名乗る欄（名前の由来）。シナリオ側を和では
   * なく `max(予算, 最大シナリオ)` で足すのは、Session が抱える slot backing の保持集合が
   * **その量を超えないから**（ADR 0095 決定 1 / 4）: 新しい signature のヒット run は
   * 「保持分 + 新規 ≤ 予算」になるまで古い順に退役させてから確保し、新規 1 本だけで予算を
   * 超える形は保持中を全て退役させてその 1 本だけを持つ（executor の
   * `Session.#evictBackingsFor`）。計画（`PreparedPlan`）が `PREPARED_PLAN_CAPACITY` 本まで
   * LRU で残るのとは別の勘定である。
   *
   * 「最大シナリオ」は**追加シナリオも含めた** {@link AdmissionReport.scenarios} 全体の最大
   * （追加分だけ勘定から漏れると、verify 形が最大のときにピークが過小になる）に、
   * `generation.chunkBuckets` の各物理行数で同じ計算をした形（バケット形）を加えた最大。
   * バケット形は {@link AdmissionReport.scenarios} に列挙しない（公開の列は prefill / decode と
   * 追加シナリオのまま）。最大 M の prefill 形で代表させないのは、行ブロックの等分が枚数の
   * 境界で「行数の少ない形のほうが一時が大きい」を作り、中間が M に単調でないため。
   * バケット形が予算の内側で他の形と同時に保持されるぶんは、予算の項が上から押さえる（過大側へ
   * 倒す — 「勘定に入れた分のピーク」の意味論はそのまま）。予算は各 backing の領域 + 所有する
   * 入力バッファを勘定する（診断の `residentBytes + inputBytes`）。
   */
  readonly peakAccountedBytes: number;
  /** 勘定に入っていないもの（見積りが絶対保証でないことを形式が認める欄 — ADR 0070 決定 5）。 */
  readonly unaccounted: readonly string[];
};

export type EstimateOptions = {
  /** Session と同じ states attention 設定。省略時は参照経路の一時を見積る。 */
  readonly stateAttentionReduce?: StateAttentionReduce;
  /**
   * 記号次元の値（グラフ入力側 — run に渡すのと同じ束縛）。states 専用記号（容量）と
   * 物理 chunk 行 `M` の記号は、どちらも {@link EstimateOptions.generation} の側から決まるので
   * ここでは受けない（前者は `generation.bindings`・後者は `generation.chunkLength`）。
   */
  readonly bindings?: SymbolBindings;
  /**
   * 見積る `GenerationContext` の仕様（省略すると state を数えない）。
   *
   * `scenarios` は ADR 0066 決定 4 の 2 形に**加えて**数える形の列（{@link AdmissionScenarioSpec}）—
   * `GenerationContext` の形には 1 バイトも効かない見積り専用の欄で、常駐側（state スロット）は
   * どの形でも同じなので動かない。
   */
  readonly generation?: GenerationContextSpec & {
    readonly scenarios?: readonly AdmissionScenarioSpec[];
  };
  /**
   * `maxStorageBufferBindingSize` の granted 値（`GPUDevice.limits` / `readAdapterLimits`）。
   *
   * 使うのは 2 箇所。① states 形 attention のノード内一時の行ブロック枚数 — 一時は**行ブロック
   * 1 枚ぶん**しか同時生存せず、その枚数はこの上限だけで決まる（{@link planStateAttention} —
   * ADR 0022 の実行時オートチューン禁止により device の granted limit と束縛だけの純関数）。
   * ② 全グラフの中間配置の束縛上限 — 行ブロック attention の枚数と、slot 実寸の上限検査に使う
   * （実行相と同じ granted 値で詰める — ADR 0093 決定 4 / 5）。② は無指定なら WebGPU core 既定で詰める。
   *
   * MUST: states 形 attention を持つグラフでは**必須**（{@link stateAttentionTemps} が
   * fail loudly）。既定値で埋めると、どの device でも実際には出ない枚数を estimator だけが
   * 主張する。**値域（正の安全整数）はグラフの形に依らず入口で見る**（① を読まない形でだけ
   * 不正値が黙って通ると、値域門の位置が呼び手から見えない）。
   */
  readonly maxStorageBufferBindingSize?: number;
  /**
   * `maxBufferSize` の granted 値（`GPUDevice.limits` / `readAdapterLimits`）。
   *
   * 使うのは state スロット単体の上限検査（`maxStorageBufferBindingSize` と**対**— 実構築は
   * 2 本とも見る: ADR 0066 追記 5）と、中間パッキングの領域の上限（実行相と同じ granted 値で
   * 詰める — ADR 0093 決定 4）。後者は `workspaceBytes` を動かしうる（first-fit の領域の割れ方が
   * 上限で変わる）。offset 整列は device 値を受ける欄が無いので WebGPU core 既定（256）で詰める。
   *
   * 任意欄にしてあるのは、device を持たない呼び手（グラフだけを握った見積り）を壊さないため。
   * 渡した欄だけが検査に効き、未指定の側は既定値で埋めない（どの device でも実際には出ない
   * 境界を estimator だけが主張しないため）。中間パッキングだけは無指定でも領域の上限が要るので、
   * 束縛上限と WebGPU core 既定の大きいほうで詰める。値域（正の安全整数）はグラフの形に依らず
   * 入口で見る。
   */
  readonly maxBufferSize?: number;
  /**
   * slot backing の保持予算（`SessionOptions.planBackingBudgetBytes` と同じ値・既定
   * `DEFAULT_PLAN_BACKING_BUDGET_BYTES` = 256 MiB）。
   *
   * MUST: 検査は `createContainerSession` と同じ（非負の安全な整数以外は fail loudly）。見積りだけが
   * Session の作れない予算を受けると、「見積れたのに構築が落ちる」形になる。
   * 効くのは {@link AdmissionReport.peakAccountedBytes} だけで、シナリオごとの数字は動かない
   * （予算は保持**本数**の側の量で、1 本の形の必要量ではない）。予算 0 = 常に 1 本なので、
   * ピークは従来どおりシナリオ側の最大になる。
   */
  readonly planBackingBudgetBytes?: number;
  /**
   * 行ブロックの一時 1 本ごとの予算（`SessionOptions.attentionRowBlockMaxBytes` と同じ値・省略 = 予算なし
   * — ADR 0124）。
   *
   * 行ブロックの枚数は実行と同じ導出（`rowBlockLimit` — 束縛上限とこの値の小さいほう）で決める。
   * 効くのは行ブロックで割る勘定（states 形 attention のスコア S と行統計・行ブロック attention の窓の
   * 一時）で、中間の配置の上限検査は束縛上限のまま（実行と同じ）。
   * MUST: 検査は `createContainerSession` と同じ（正の安全な整数以外は fail loudly）。
   * MUST: 1 行ぶんの一時がこの値を超える形は、実行の計画導出と同じ文言で fail loudly — 見積りが
   * 予算違反を事前に知る手段（実行では重みを上げた後の run で落ちる）。
   */
  readonly attentionRowBlockMaxBytes?: number;
};

/**
 * 勘定に入っていないもの（定数）。
 *
 * MUST: 「入っていない」と分かっている項目を全部書く。空欄にすると、見積りが上限であるかの
 * ように読まれる（ADR 0070 決定 5 が unaccounted 欄を要求した理由そのもの）。
 */
const UNACCOUNTED: readonly string[] = Object.freeze([
  "行ブロック attention 以外の融合が畳んで消す中間と、その融合ルールが宣言するノード内一時（どちらも device limit と融合の成立に依存する。別名化した値と、行ブロック attention の一時は勘定に入っている）",
  "states 形でない attention のノード内一時（スコアの行ブロックと i8a8 の量子化中間）と、linear i8a8 の量子化中間 — どれも数値変種と device limit に依存する。states 形 attention のスコア S と、stateAttentionReduce に応じて必要な行統計は勘定に入っている",
  "params バッファ（カーネル定数 — Session 常駐・内容アドレスキャッシュ）",
  "queue.writeBuffer の実装 staging（submit の完了まで解放されない）",
  "退役の窓（予算超過 / 計画の LRU 追い出しで退役した slot backing は、次の計画の確保より前に destroy されず flush 後の後始末まで生きるので、その run では退役分 + 新規が同時に載る）",
]);

/** 解決済み shape を引く（planGraph が全値を載せているので、欠けは簿記の破れ）。 */
const shapeOf = (
  shapes: ReadonlyMap<string, readonly number[]>,
  name: string,
): readonly number[] => {
  const shape = shapes.get(name);
  if (shape === undefined) throw new ExecutionError(`値 '${name}' の shape が未解決`);
  return shape;
};

/**
 * 物理 chunk 行 `M` が載る軸。
 *
 * MUST: 位置の正本は **op 契約**（ops/shapes.ts）。states 形の attention は
 * `q[B,H,M,D]` / `k`,`v``[B,Hkv,M,D]`（M は q / k / v で一致することも契約が見ている）、
 * `state_append` は `x[B,Hkv,M,D]` で、どちらも軸 2。ここで別の軸を書くと、estimator だけが
 * 実行と違う次元を chunk 行と見なして prefill / decode を取り違える。
 */
const CHUNK_ROW_AXIS = 2;

/**
 * 物理 chunk 行 `M` を決めている**次元宣言**（記号を含むものだけ・重複排除済み）。
 *
 * 判別は「states 欄が非空なノードの入力」= states 形の判別そのもの（ADR 0067 決定 4 —
 * recipe-builders/attention.ts の `buildStateAttention` / `buildStateAppend` が `states.chunkRows` へ
 * 積むのと同じノード集合）。数値次元は記号で動かない形なので拾わない — その形の decode step は
 * 物理 chunk 行を prefill と同じまま `queryLength = 1` で回るので、2 シナリオが同じ数字になるのが
 * 正しい。ただし数値の `M` が context の許す集合に入っているかは別の門で、シナリオごとに
 * {@link stateChunkRows} を照合する（`chunkLength` と一致するとは限らない）。
 */
const chunkRowDims = (graph: IrGraph): readonly string[] => {
  const declared = new Map<string, readonly IrDim[]>();
  for (const spec of graph.inputs) declared.set(spec.name, spec.shape);
  for (const [name, value] of Object.entries(graph.values)) declared.set(name, value.shape);
  const dims = new Set<string>();
  for (const node of graph.nodes) {
    if (Object.keys(node.states).length === 0) continue;
    // 宣言が引けない名前（initializer だけの値など）は planGraph が落とす — ここでは飛ばす。
    for (const name of node.ins) {
      const dim = declared.get(name)?.[CHUNK_ROW_AXIS];
      if (typeof dim === "string") dims.add(dim);
    }
  }
  return [...dims];
};

/**
 * 解決済み計画の state ノードが使う物理 chunk 行数（同値は畳む）。
 *
 * MUST: 拾うノードと軸は実行の集積（recipe-builders/attention.ts の `buildStateAttention` が
 * `q[2]`・`buildStateAppend` が `x[2]` を `states.chunkRows` へ積む）と同じ — どちらも第 1 入力の
 * {@link CHUNK_ROW_AXIS}。readonly states 形 attention の `q` は M = 1 を shape 層が保証するので、
 * 拾っても許可集合（常に 1 を含む）の判定は動かない。
 */
const stateChunkRows = (nodes: readonly NodePlan[]): ReadonlySet<number> => {
  const rows = new Set<number>();
  for (const step of nodes) {
    if (Object.keys(step.node.states).length === 0) continue;
    rows.add(step.inputShapes[0][CHUNK_ROW_AXIS]);
  }
  return rows;
};

/**
 * 値 shape の解決に使う束縛を検査して写す。
 *
 * MUST: states 専用記号をここで受けない（`bindSymbols` と同じ分担 — 容量の正本は
 * `createGenerationContext` の側で、ここで受けると「state 容量を渡したのに 0 と出る」
 * 沈黙誤答になる）。未束縛は fail loudly（黙って 0 で埋めない）。
 * MUST: chunk 行の記号もここで受けない（束縛点は `options.generation.chunkLength` と decode 形の
 * 1 で、シナリオごとに値が変わる第 3 の束縛点）。受けて上書きすると「prefill と決め打った値を
 * 渡したのに decode の数字が返る」沈黙上書きになり、受けて優先すると 2 本のシナリオが同じ形に
 * 潰れる。
 */
const planBindings = (
  graph: IrGraph,
  bindings: SymbolBindings | undefined,
  chunkSymbols: ReadonlySet<string>,
): SymbolBindings => {
  // 記号の実在と非負整数の検査、および null プロトタイプの器は resolveBindings が持つ。
  const resolved = resolveBindings(graph, bindings);
  const statesOnly = statesOnlySymbols(graph);
  for (const sym of Object.keys(resolved)) {
    if (statesOnly.has(sym)) {
      throw new ExecutionError(
        `束縛 '${sym}' は states 専用記号 — options.generation.bindings で与えること` +
          "（ADR 0066 追記 7 の束縛点 2）",
      );
    }
    if (chunkSymbols.has(sym)) {
      throw new ExecutionError(
        `束縛 '${sym}' は物理 chunk 行 M の記号 — 値は options.generation.chunkLength` +
          "（prefill 形）と 1（decode 形）から導く（ADR 0066 決定 4 の実行 2 形）",
      );
    }
  }
  for (const sym of graph.symbols) {
    if (statesOnly.has(sym) || chunkSymbols.has(sym) || Object.hasOwn(resolved, sym)) continue;
    throw new ExecutionError(
      `シンボル '${sym}' が束縛されていない（options.bindings で与えること）`,
    );
  }
  return resolved;
};

/**
 * chunk 行を `rows` 行にしたシナリオの束縛（`base` の写しに chunk 記号を足す）。
 *
 * MUST: 逆解きは実行と同じ 1 本（{@link solveDim} — `bindSymbols` が実入力の実寸から束縛を
 * 解くのに使うのと同じ関数）。派生形（`2M`）を丸めて受けると、宣言と 1 行ずれた chunk を
 * 見積った数字が「その形で走る」顔をして返る。
 */
const bindChunkRows = (
  base: SymbolBindings,
  dims: readonly string[],
  rows: number,
  scenario: string,
): SymbolBindings => {
  const bindings: Record<string, number> = Object.assign(Object.create(null), base);
  for (const dim of dims) {
    const expr = parseDim(dim);
    const solved = solveDim(expr, rows);
    if (solved === undefined) {
      throw new ExecutionError(
        `シナリオ '${scenario}': 物理 chunk 行 ${rows} が宣言 '${dim}' の形をしていない` +
          `（${expr.coeff} で割り切れる ${expr.offset} 以上の行数が要る — ADR 0066 決定 4）`,
      );
    }
    if (Object.hasOwn(bindings, expr.sym) && bindings[expr.sym] !== solved) {
      throw new ExecutionError(
        `シナリオ '${scenario}': 記号 '${expr.sym}' の束縛が衝突（${bindings[expr.sym]} と ` +
          `${solved}） — chunk 行の宣言が 1 グラフ内で割れている`,
      );
    }
    bindings[expr.sym] = solved;
  }
  return bindings;
};

/**
 * device 上限欄の値域検査（省略可・与えるなら正の安全整数）。
 *
 * MUST: 読み手（`stateAttentionTemps` / `assertSlotWithinLimits`）の内側でなく入口に置く —
 * 読まない形のグラフでだけ不正値が黙って通ると、値域門の位置が呼び手から見えない。
 */
const assertLimitOption = (name: string, limit: number | undefined): void => {
  if (limit !== undefined && (!Number.isSafeInteger(limit) || limit < 1)) {
    throw new ExecutionError(`options.${name} ${limit} は正の安全整数でなければならない`);
  }
};

/** state スロットの見積り（{@link GenerationContext} の確保と同じ式 — 値の複製はしない）。 */
const stateEstimate = (
  graph: IrGraph,
  spec: GenerationContextSpec | undefined,
  limits: SlotLimits,
): {
  readonly bytes: number;
  readonly shapes: ReadonlyMap<string, readonly number[]> | undefined;
  readonly bindings: SymbolBindings | undefined;
} => {
  if (spec === undefined) return { bytes: 0, shapes: undefined, bindings: undefined };
  const names = Object.keys(graph.states);
  // MUST: states 宣言の無いグラフで generation を受けない（`GenerationContext.create` が
  // 拒否する形。通すと「作れない context のバイト数」を数えた見積りが返る）。
  if (names.length === 0) {
    throw new ExecutionError(
      "このグラフは states 宣言を持たない（GenerationContext は作れないので options.generation を渡さないこと）",
    );
  }
  // MUST: 実構築（GenerationContext.create）が拒否する spec に見積りを返さない — 値域は
  // 同じ門（assertChunkLength / assertChunkBuckets）を、スロット単体の寸法は同じ
  // `assertSlotWithinLimits` を通す（下の走査内）。
  assertChunkLength(spec.chunkLength);
  assertChunkBuckets(spec.chunkBuckets, spec.chunkLength);
  const bindings = resolveBindings(graph, spec.bindings);
  const shapes = new Map<string, readonly number[]>();
  let bytes = 0;
  for (const name of names) {
    const shape = resolveSlotShape(name, graph.states[name].shape, bindings);
    shapes.set(name, shape);
    // 借り物スロット（external — ADR 0096 段 2 §1.1）は借り手 context が確保しない。形は
    // 一時（S / 行統計）の勘定に要るので解決するが、バイト数は 1 つも数えない
    // （`GenerationContext.create` の確保と同じ分岐 = 見積りと実構築が同じ数を主張する）。
    if (graph.states[name].external) continue;
    const byteLength = numel(shape) * STATE_ELEMENT_BYTES;
    // MUST: 実構築（`GenerationContext.create`）と同じ位置・同じ門でスロット単体の上限を見る
    // （ADR 0066 追記 5）。ここを抜くと、CLI が `--capacity` の値検査を見積りに兼ねさせている
    // 経路（examples/shared/gemma-main.ts）で「起動時には数字が出て、最初のターンで落ちる」。
    assertSlotWithinLimits(name, shape, byteLength, limits);
    bytes += byteLength;
  }
  // 論理長 uniform は context 1 本につき 1 枚（スロット数に依らない）。
  return { bytes: bytes + LENGTHS_BYTES, shapes, bindings };
};

/**
 * 重みの見積り。分類も宣言由来のバイト数も **Session 構築と同じプランナ**
 * （{@link planWeightResidency}）の結果を受け取り、席から実バッファへの展開も構築 / 上限検査と
 * 同じ 1 本（{@link planWeightBuffers}）を通す — ここで分類や展開を書き直すと、片方だけ直された
 * 実装に対して estimator が例外も警告も無く別の数を主張し続ける（モジュール doc の MUST）。
 *
 * ここが足すのは席ごとのカテゴリ分けだけ。整列詰め物とバッファ床（{@link toSizeClass}）は
 * `WeightBuffer.byteLength` 側が持つ — `alignF16Payload` / `alignI8Payload` の 4 バイト切り上げと
 * `allocHostWritten` の `Math.max(4, …)` を合わせた値と同じ。
 */
const weightEstimate = (
  residency: ReadonlyMap<string, WeightResidency>,
): { readonly compressed: number; readonly uncompressed: number; readonly expanded: number } => {
  let compressed = 0;
  let uncompressed = 0;
  let expanded = 0;
  for (const buffer of planWeightBuffers(residency)) {
    switch (buffer.seat) {
      case "raw":
        // 圧縮しない格納は生バイトがそのまま GPU 表現（executor の分岐 3 本目）。
        uncompressed += buffer.byteLength;
        break;
      case "expanded":
        // 適格外はロード時に CPU で f32 展開する（VRAM 削減はゼロ）。
        // MUST: この欄だけ整列前のバイト数で数える — 診断 `storage.hostExpandedBytes` は
        // writeBuffer した実バイトを積むので 4 バイト床を含まず、ここを確保寸法に替えると
        // 0 要素テンソルを持つモデルで「厳密対応」が沈黙して崩れる。
        expanded += buffer.declaredBytes;
        break;
      // 圧縮常駐は payload と companion scale の**両方**を数える（executor の
      // residentCompressedBytes と同じ規律 — 実際に GPU が抱えるバイト数を表す欄なので、
      // scale を除くと実績と食い違う）。どちらも `WeightBuffer` として並んでいる。
      case "f16":
      case "i8":
      case "i4":
      case "i2":
        compressed += buffer.byteLength;
        break;
      default: {
        // MUST: 未処理の席をここで止める。switch が素通りすると、そのバッファのバイト数が
        // 3 欄のどれにも入らないまま `resident.weights.totalBytes` が実ロードより小さい数を
        // 主張し続ける（見積りは実行を止めない面なので、ずれても赤くならない）。型としては
        // 到達不能で、席を足したときの結線漏れをコンパイル時に赤くするのがこの分岐の役目。
        const unhandled: never = buffer.seat;
        throw new ExecutionError(`未処理の常駐席: ${JSON.stringify(unhandled)}`);
      }
    }
  }
  return { compressed, uncompressed, expanded };
};

/**
 * states 形 attention か（ADR 0067 決定 4 — **欄の有無が形を判別する**）。
 *
 * MUST: 判別は recipe-builder の `#buildStep` と同じ 1 条件（op が `attention` で `states` 欄が
 * 非空）。別条件を書くと、融合 attention とは 1 バイトも共有しない別族カーネルの一時を
 * estimator だけが取り違える。
 */
const isStateAttention = (node: NodePlan): boolean =>
  node.contract.kind === "attention" && Object.keys(node.node.states).length > 0;

/**
 * 2 相形の argmax / topk(k=1)（長い行 — src/kernels/argmax.ts の「2 相分割」）が持つ部分結果の一時。
 * 経路の選択は実行相と同じ純関数 `argmaxSplitGroups` / `topkOneSplitGroups` を通す
 * — ここで閾値を書き直すと、片方だけ直された実装に対して estimator が別の数を主張し続ける。
 * 1 dispatch 形（短い行）は 0 を返す。
 */
const selectionSplitTempBytes = (node: NodePlan): number => {
  if (
    node.node.op !== "argmax" &&
    !(node.node.op === "topk" && topkK(node.node.attrs, "topk estimate") === 1)
  ) return 0;
  const input = node.inputShapes[0];
  const dim = input[input.length - 1];
  const groups = node.node.op === "topk" ? topkOneSplitGroups(dim) : argmaxSplitGroups(dim);
  return groups === 0 ? 0 : argmaxSplitPartialBytes(numel(input.slice(0, -1)), groups);
};

/**
 * states 形 attention 1 ノードが出すノード内一時（スコア S と行統計）を行ブロック順に並べる。
 *
 * 融合の成立に依存せず必ず出て、大きさは列容量（full = スロット容量 `C` / sliding = 窓の
 * resident 幅 `W−1+M`）に比例するため、可変 capacity のグラフでは中間の主役になる。
 *
 * MUST: 算式は {@link planStateAttention}（実行計画 `buildStateAttention` と共有する 1 本）
 * だけから引く。ここで式を書き直すと、片方だけ直された実装に対して estimator が例外も警告も
 * 無く別の数を主張し続ける（モジュール doc の MUST）。ここが持つのは、見積り固有の材料
 * （記号解決済みのスロット shape）を幾何へ翻訳する部分だけ。
 * MUST: `maxStorageBufferBindingSize` が無ければ fail loudly。枚数は device の granted limit
 * （と利用者の予算）だけで決まるので、既定値で埋めると「どの device でも出ない一時の大きさ」を
 * 名乗ることになる。
 * MUST: 予算（`budget` — `EstimateOptions.attentionRowBlockMaxBytes`）は実行と同じ `rowBlockLimit` で
 * 束縛上限と合わせる（ADR 0124 決定 7）。
 * NOTE: 値域（正の安全整数）は {@link estimateGraphMemory} の入口が見る — states 形 attention を
 * 持たないグラフはここを通らないので、内側に置くと呼び手から見えない位置の門になる。
 */
const stateAttentionTemps = (
  node: NodePlan,
  stateShapes: ReadonlyMap<string, readonly number[]> | undefined,
  limit: number | undefined,
  budget: number | undefined,
  stateAttentionReduce: StateAttentionReduce,
): readonly StateAttentionBlock[] => {
  const q = node.inputShapes[0];
  const where = `attention (states) [${q.join(",")}]`;
  if (limit === undefined) {
    throw new ExecutionError(
      `${where}: states 形 attention の一時は options.maxStorageBufferBindingSize が要る` +
        "（行ブロックの枚数は device の granted limit だけで決まる — ADR 0060 / ADR 0022）",
    );
  }
  const [batch, heads, chunkRows] = q;
  // k / v スロットは同形（shape 層が済ませている）ので容量は片方から引く。
  const slotName = node.node.states["k"];
  const slotShape = slotName === undefined ? undefined : stateShapes?.get(slotName);
  if (slotShape === undefined) {
    throw new ExecutionError(
      `${where}: state スロット 'k' の解決済み容量が引けない` +
        "（GenerationContext を伴わない見積りでは states 形 attention を数えられない）",
    );
  }
  // 行ブロックの枚数を明示する `forced`（テスト専用 `ROW_BLOCK_SPLIT`）は渡さない —
  // `EstimateOptions` に受け口が無く、実運用の呼び手も使わないため（見積りは既定の等分だけを
  // 名乗る）。
  return planStateAttention({
    batchHeads: batch * heads,
    chunkRows,
    capacity: slotShape[2],
    window: stateWindow(node.node.attrs, where) ?? 0,
    fuseStatsPv: node.node.ins.length === 3 && stateAttentionReduce === "parallel-fused",
  }, rowBlockLimit(limit, budget)).blocks;
};

/**
 * 融合計画（実行相と同じ {@link planFusions}）のうち、見積りが**融合後の形**で数えるステップを
 * 「窓の先頭ノードの添字 → ステップ」で引く（ADR 0093 追記 2026-10-10）。
 *
 * 対象は行ブロック attention だけ。融合前の列で数えると、融合で実体化されないスコア・mask 加算後・
 * softmax 後の全幅に束縛上限が当たり、実行できる形の見積りだけが例外になるため（ADR 0060 の需要形）。
 * 他のルールは従来どおり融合前の列で数える（`unaccounted` の側）。
 *
 * MUST: 束縛上限は確保プログラムの上限検査と同じ値を渡す（行ブロックの枚数はこの値と予算だけで
 * 決まる — 別の値で割ると、見積りだけが実行と違う枚数の一時を数える）。予算（`budget`）は実行と同じく
 * 融合の文脈の `attentionRowBlockMaxBytes` として渡す（ADR 0124 決定 7）。行ブロック枚数の強制
 * （`rowBlockSplit` — テスト専用）は渡さない（{@link stateAttentionTemps} と同じ理由）。
 * NOTE: `maxComputeWorkgroupsPerDimension` は dispatch 数の上限検査にしか効かず、バイト数を動かさない。
 * 見積りは素のノードでも dispatch 数を検査しない（受ける欄も無い）ので、ここも検査しない値を渡す。
 */
const rowBlockAttentionSteps = (
  nodes: readonly NodePlan[],
  uses: ReadonlyMap<string, number>,
  outputNames: ReadonlySet<string>,
  bindingLimit: number,
  budget: number | undefined,
): ReadonlyMap<number, FusedStep> => {
  const { steps } = planFusions(nodes, {
    useCounts: uses,
    outputNames,
    limits: {
      maxStorageBufferBindingSize: bindingLimit,
      maxComputeWorkgroupsPerDimension: Number.MAX_SAFE_INTEGER,
    },
    ...(budget === undefined ? {} : { attentionRowBlockMaxBytes: budget }),
  });
  const found = new Map<number, FusedStep>();
  // 走査は窓ごとに「passthrough のノード → 融合ステップ」を並べて窓幅ぶん進むので、ステップ列の
  // ノード数の累計は常に次の窓の先頭を指す。行ブロック attention の窓は passthrough を持たない
  // （窓 = 鎖）ので、この累計がそのまま融合前の列での先頭になる。
  let index = 0;
  for (const step of steps) {
    if (step.kind === "node") {
      index += 1;
      continue;
    }
    if (step.rule === "rowBlockAttention") {
      // MUST: 鎖の末尾が融合ステップの出力と一致しなければ簿記の破れ（黙って別のノードを
      // 置き換えると、見積りが実行と違う列を数える）。
      if (nodes[index + step.nodeCount - 1]?.outputs[0]?.name !== step.outputName) {
        throw new ExecutionError(
          `見積り: 行ブロック attention '${step.outputName}' の窓が融合前の列 ${index} に一致しない`,
        );
      }
      found.set(index, step);
    }
    index += step.nodeCount;
  }
  return found;
};

/**
 * 行ブロック attention の融合ステップ 1 つの確保プログラム（recipe-builder の `#buildStep` /
 * `#buildFused` の簿記面の写し）。
 *
 * 出力は単一出力の実体（`numel × 4`）、一時はルールが宣言した寿命のまま、解放は外部入力の延べ列
 * （{@link FusedStep.ins}）。dispatch の読み書きは実行相と同じく**WGSL の storage 宣言**から採る
 * （{@link parseStorageRoles} — 手書きの表を持たない）。
 *
 * MUST: 行ブロック attention 専用。写しているのはこのルールが使う形（全 dispatch が operands を
 * 宣言する・常駐 scale を束ねない）だけで、他のルールの形（operands 省略時の既定の並び・
 * `weightScale`）は写していない。他のルールを渡すと、見積りが実行と違う確保プログラムを
 * 黙って数えるので、ルール名と形の両方で fail loudly にする。他のルールへ広げるときは、
 * 写す形をここに足してから呼び手の対象を広げる。
 */
const fusedTransientStep = (
  step: FusedStep,
  uses: ReadonlyMap<string, number>,
  outputNames: ReadonlySet<string>,
): TransientStepSpec => {
  const where = `見積り: 融合ルール '${step.rule}'（出力 '${step.outputName}'）`;
  if (step.rule !== "rowBlockAttention") {
    throw new ExecutionError(`${where} は融合後の形で数えられない（行ブロック attention 専用）`);
  }
  const output: TransientRef = { kind: "value", name: step.outputName };
  const bindRef = (index: number): TransientRef => {
    const name = step.binds[index];
    // MUST: 範囲外の添字は fail loudly（実行相 recipe-builder の同じ門と揃える — 黙って名前の
    // 無い参照を作ると、見積りだけが実行相の落ちる形を数として返す）。
    if (name === undefined) {
      throw new ExecutionError(`${where}: bind 添字 ${index} が宣言 ${step.binds.length} 本の外`);
    }
    return { kind: "value", name };
  };
  const dispatches = step.dispatches.map((dispatch): TransientDispatchSpec => {
    if (dispatch.operands === undefined) {
      throw new ExecutionError(
        `${where} の dispatch '${dispatch.key}' が operands を宣言していない`,
      );
    }
    const roles = parseStorageRoles(dispatch.wgsl());
    const reads: TransientRef[] = [];
    const writes: TransientRef[] = [];
    dispatch.operands.forEach((operand, slot) => {
      if (operand.kind === "weightScale") {
        throw new ExecutionError(
          `${where} の dispatch '${dispatch.key}' が常駐 scale を束ねている`,
        );
      }
      const ref: TransientRef = operand.kind === "temp"
        ? { kind: "temp", id: operand.id }
        : operand.kind === "output"
        ? output
        : bindRef(operand.index);
      // binding 0 は params なので、オペランドは binding 1 から並ぶ。
      const binding = slot + 1;
      if (roles.writes.has(binding)) writes.push(ref);
      else if (roles.reads.has(binding)) reads.push(ref);
      else {
        throw new ExecutionError(
          `${where} の dispatch '${dispatch.key}' の束縛 ${binding} に WGSL の` +
            " storage 宣言が無い（読み / 読み書きの役割が採れない）",
        );
      }
    });
    return { reads, writes };
  });
  return {
    outputs: [{
      name: step.outputName,
      kind: "alloc",
      byteLength: numel(step.outputShape) * 4,
      uses: uses.get(step.outputName) ?? 0,
      pinned: outputNames.has(step.outputName),
    }],
    temps: step.temps,
    dispatches,
    releases: step.ins,
  };
};

/** 中間の勘定が読む実行ノブ（Session と同じ値 — {@link estimateGraphMemory} の入口で値域を見た後のもの）。 */
type TransientKnobs = {
  readonly stateAttentionReduce: StateAttentionReduce;
  /** 行ブロックの一時 1 本ごとの予算（`undefined` = 予算なし — ADR 0124）。 */
  readonly attentionRowBlockMaxBytes: number | undefined;
};

/**
 * 中間バッファ（ノード出力と融合ルールの一時）の GPU バイト数を、実行と**同じ計画関数**で数える
 * （ADR 0093 決定 4 — {@link planTransients} の共有。ADR 0070 決定 5「実行と同じ規則」を関数共有で
 * 満たす）。返るのは領域の総和（`totalBytes`）。
 *
 * MUST: 確保プログラムは融合前ノード列から組み、行ブロック attention の窓だけを実行相と同じ
 * 融合ステップに置き換える（ADR 0093 追記 2026-10-10 — {@link rowBlockAttentionSteps}）。他の融合が
 * 畳む中間と融合ルールの一時は `unaccounted` の側。素のノードの dispatch は 1 ノード 1 本の近似
 * （読み = `ins`・書き = `outs`）で、usage scope の別領域規則が効く相手を実行相と同じ向きで与える。
 * states 形 attention だけは実行相の 3 dispatch（①qk が scores を書く → ②stats が scores を読み
 * 行統計を書く → ③pv が両方を読み出力を書く）を行ブロックごとに写す — 一時同士の重なりと領域の
 * 分かれ方が総和に効くため。
 * MUST: 上限超過（slot 実寸 > 束縛上限）は計画関数が**全件列挙して**落とす（ADR 0093 決定 5）。
 * 見積りの入口でも同じ検査が走るので、「この device では中間が束ねられない」形は数を返さず
 * fail loudly になる。検査が当たるのは融合後に実際に束縛するバッファ（行ブロック attention では
 * 1 枚ぶんの一時）で、融合で実体化されない中間には当てない。上限は `limit`
 * （`EstimateOptions.maxStorageBufferBindingSize` — 無指定なら WebGPU core 既定）で、領域の上限は
 * `maxBufferSize`（`EstimateOptions.maxBufferSize`）。後者が
 * 無指定のときだけ束縛上限と core 既定の大きいほうで埋める（束縛上限 ≤ `maxBufferSize` は仕様の
 * 不変条件）。渡された granted 値を使うのは、first-fit の領域の割れ方が上限で変わり、実行相
 * （granted 値で計画する）と総和が食い違うため（ADR 0093 決定 4）。
 * MUST: 簿記の単位は**値名ではなく slot**（別名は根の生存を延ばす）。
 */
const transientSlotBytes = (
  graph: IrGraph,
  nodes: readonly NodePlan[],
  stateShapes: ReadonlyMap<string, readonly number[]> | undefined,
  limit: number | undefined,
  maxBufferSize: number | undefined,
  knobs: TransientKnobs,
): number => {
  const { stateAttentionReduce, attentionRowBlockMaxBytes } = knobs;
  const uses = countUses(graph);
  const outputNames = new Set(graph.outputs);
  const aliases = planAliases(nodes);
  const bindingLimit = limit ?? CORE_TRANSIENT_LIMITS.maxStorageBufferBindingSize;
  const nodeStep = (node: NodePlan): TransientStepSpec => {
    const isAlias = aliases.has(node);
    const outputs: TransientOutputSpec[] = node.outputs.map((out) => ({
      name: out.name,
      kind: isAlias ? "alias" : "alloc",
      byteLength: isAlias ? 0 : numel(out.shape) * 4,
      source: isAlias ? node.node.ins[0] : undefined,
      uses: uses.get(out.name) ?? 0,
      pinned: outputNames.has(out.name),
    }));
    const reads: TransientRef[] = node.node.ins.map((name) => ({ kind: "value", name }));
    const outputRefs: TransientRef[] = node.outputs.map((out) => ({
      kind: "value",
      name: out.name,
    }));
    const temps: TransientTempSpec[] = [];
    const dispatches: TransientDispatchSpec[] = [];
    if (isStateAttention(node)) {
      // 行ブロックごとに実行相の 2 / 3 dispatch と一時の寿命を写す。
      // scores は QK の直前、非融合時の stats は行統計の直前に確保し、PV の直後に解放する。
      // readonly 変種（借り手の cross-attention — ADR 0096・ins は q だけ）は K/V を external
      // スロットから読み、今 step の k / v 入力を持たない。読みの参照は**在る入力だけ**にする
      // （無い名前を参照に載せると transient 計画が未定義の参照で落ちる）。
      const [q, k, v] = reads;
      const blocks = stateAttentionTemps(
        node,
        stateShapes,
        limit,
        attentionRowBlockMaxBytes,
        stateAttentionReduce,
      );
      for (const temp of blocks) {
        const base = dispatches.length;
        const fused = temp.statsBytes === 0;
        const last = base + (fused ? 1 : 2);
        const scores: TransientRef = { kind: "temp", id: temps.length };
        temps.push({ byteLength: temp.scoreBytes, allocBefore: base, releaseAfter: last });
        dispatches.push({ reads: k === undefined ? [q] : [q, k], writes: [scores] });
        if (fused) {
          dispatches.push({ reads: v === undefined ? [scores] : [scores, v], writes: outputRefs });
        } else {
          const stats: TransientRef = { kind: "temp", id: temps.length };
          temps.push({ byteLength: temp.statsBytes, allocBefore: base + 1, releaseAfter: last });
          dispatches.push({ reads: [scores], writes: [stats] });
          dispatches.push({
            reads: v === undefined ? [scores, stats] : [scores, stats, v],
            writes: outputRefs,
          });
        }
      }
    } else if (!isAlias && selectionSplitTempBytes(node) > 0) {
      // 部分最大から最小添字へ縮約する。topkはmergeで元入力の値ビットも読む。
      const partial: TransientRef = { kind: "temp", id: temps.length };
      temps.push({
        byteLength: selectionSplitTempBytes(node),
        allocBefore: 0,
        releaseAfter: 1,
      });
      dispatches.push({ reads, writes: [partial] });
      dispatches.push({
        reads: node.node.op === "topk" ? [partial, ...reads] : [partial],
        writes: outputRefs,
      });
    } else if (!isAlias) {
      dispatches.push({ reads, writes: outputRefs });
    }
    // 別名も入力を消費する（`uses` に数えられている）ので、解放は全ノードで入力の延べ列。
    return { outputs, temps, dispatches, releases: node.node.ins };
  };
  const fused = rowBlockAttentionSteps(
    nodes,
    uses,
    outputNames,
    bindingLimit,
    attentionRowBlockMaxBytes,
  );
  const program: TransientStepSpec[] = [];
  for (let index = 0; index < nodes.length;) {
    const step = fused.get(index);
    if (step === undefined) {
      program.push(nodeStep(nodes[index]));
      index += 1;
      continue;
    }
    program.push(fusedTransientStep(step, uses, outputNames));
    index += step.nodeCount;
  }
  return planTransients(program, {
    maxBufferSize: maxBufferSize ?? Math.max(CORE_TRANSIENT_LIMITS.maxBufferSize, bindingLimit),
    maxStorageBufferBindingSize: bindingLimit,
    offsetAlignment: CORE_TRANSIENT_LIMITS.offsetAlignment,
  }).totalBytes;
};

/**
 * Session 1 本ぶんの必要メモリを見積る（ADR 0070 決定 5 — GPU も device も要らない純関数）。
 * `PreparedModel.estimate` が呼ぶのもこの 1 本である。
 *
 * 判定はしない: 返るのはカテゴリ別のバイト数と、勘定に入っていないものの列だけ。可否の
 * 最終門は out-of-memory errorScope のままで、この見積りを超えていても実行は止まらない。
 *
 * **グラフ単位の見積り口**でもある: ロードを終えて初期化子のバイト列を手放した呼び手
 * （models のパイプラインは `PreparedModel` を捨てて `IrGraph` だけ残す）が、握っている
 * グラフと `planWeightResidency(graph)` の計画、そして `{capacity, chunkLength}` を
 * `options.generation` に載せて同じ {@link AdmissionReport} を得る。
 *
 * MUST: 常駐計画を**持っている呼び手はそれを渡す**（`PreparedModel` は prepare 時に 1 回だけ
 * 計算して構築とも共有する）。構築に使ったのと別の計画を組み直して渡すと、「見積りに使った席」
 * と「実際に上げた席」が別の計算結果になりうる形が復活する。
 * 持っていない呼び手（`PreparedModel` を捨てて `IrGraph` だけ残す models のパイプライン）は
 * `planWeightResidency(graph)` で引いてよい — グラフが同じなら計画も同じ純関数なので、席の
 * 食い違いは生じない（`mod.ts` がこの 2 本を公開している理由そのもの）。
 *
 * MUST: 渡す `graph` は `PreparedModel.graph`（構築と同じ門 — capability 検査と契約検査 — を
 * 通った検証済みのグラフ）から取る。下の門はこの関数の中で走るが、`planWeightResidency(graph)`
 * は引数の評価で**先に**走るので、未検証のグラフでは常駐計画の内側（conv1d の attrs や payload
 * 長の検査）が先に落ち、失敗の文言が `nodes[i] (op)` を名乗らない（fail loudly 自体は保たれる）。
 *
 * @param options.bindings グラフ入力側の記号次元（run に渡すのと同じ束縛）。未束縛の記号が
 *   要る形は fail loudly。物理 chunk 行 `M` の記号だけは**ここでは受けない**（束縛点は
 *   `options.generation.chunkLength` と decode 形の 1 — {@link planBindings}）。
 * @param options.generation 見積る `GenerationContext` の仕様。省略すると
 *   `resident.stateBytes` は 0・シナリオは `"run"` の 1 本（states 形グラフでは省略できない —
 *   中間ピークの計画がスロットの解決済み shape を要求する）。渡すとシナリオは
 *   `"prefill"` / `"decode"` の 2 本になり、`generation.scenarios` を足すとその後ろに宣言順で
 *   並ぶ（{@link AdmissionScenarioSpec}）。`generation.chunkBuckets` の各形はシナリオに並べず、
 *   {@link AdmissionReport.peakAccountedBytes} の最大にだけ入る。
 * @param options.maxStorageBufferBindingSize device の granted 上限。states 形 attention を
 *   持つグラフでは必須（ノード内一時の行ブロック枚数がこれだけで決まる）。全グラフの中間配置の
 *   束縛上限にも使う（無指定なら WebGPU core 既定）。
 * @param options.planBackingBudgetBytes slot backing の保持予算（`SessionOptions` と同じ値）。
 *   {@link AdmissionReport.peakAccountedBytes} の片側の項になる。
 * @param options.attentionRowBlockMaxBytes 行ブロックの一時 1 本ごとの予算（`SessionOptions` と同じ値 —
 *   ADR 0124）。行ブロックの枚数を実行と同じ導出で決め、1 行ぶんにも満たない予算は実行と同じ文言で
 *   fail loudly（予算違反を重みの読み込みの前に知る手段）。
 */
export const estimateGraphMemory = (
  graph: IrGraph,
  residency: ReadonlyMap<string, WeightResidency>,
  options: EstimateOptions,
): AdmissionReport => {
  // MUST: 実構築（`createSessionFromContainer`）と**同じ門**を先に通す（見積り本体より先 —
  // 常駐計画 `residency` は呼び手が先に計算済みなので、門の後に計画する順序を保証するのは
  // `PreparedModel` 経由、または検証済みの `PreparedModel.graph` を渡す呼び手だけ）。
  // 作れない構成へ見積りを返すと、格納 dtype や op が非対応のモデルに対して estimator だけが
  // もっともらしい総量を主張する（例: 格納 `bf16` は IR の語彙にはあるが RUNTIME_SUPPORT に
  // 無く、Session 構築は必ず落ちる）。門を共有すれば語彙が増えたときの抜けも同時に塞がる。
  // NOTE: これは capability 検査であって空き側との比較ではないので、ADR 0070 決定 5 の
  // 「比較をしない」規律には抵触しない（GPU にも触らず純関数のまま）。
  assertRuntimeSupport(graph, RUNTIME_SUPPORT);
  // グラフ全体を見ないと決まらない契約（state_append の本数と終端・Tmax 形・slice / flip の
  // 静的軸・cat の連結軸・入力の意味論 dtype）も同じ位置で通す。`PreparedModel.estimate` は
  // 構築時と二重に通ることになるが、どちらも純関数・冪等でグラフ 1 走査ぶんの費用しかない。
  validateGraphContracts(graph);
  const stateAttentionReduce = options.stateAttentionReduce ?? "sequential";
  if (typeof stateAttentionReduce !== "string") {
    throw new ExecutionError(
      `options.stateAttentionReduce は文字列でなければならない（${typeof stateAttentionReduce}）`,
    );
  }
  if (!Object.hasOwn(STATE_ATTENTION_REDUCES, stateAttentionReduce)) {
    throw new ExecutionError(
      "options.stateAttentionReduce '" + stateAttentionReduce + "' は未対応",
    );
  }
  // MUST: 値域はグラフの形に依らずここで見る。stateAttentionTemps の内側だけで検査すると、
  // states 形 attention を持たないグラフで -1 / 1.5 / NaN が黙って受理される。
  assertLimitOption("maxStorageBufferBindingSize", options.maxStorageBufferBindingSize);
  assertLimitOption("maxBufferSize", options.maxBufferSize);
  // MUST: 予算の値域も**ここで**見る（`createContainerSession` と同じ検査 — ADR 0095 決定 1）。読む位置
  // （下のピーク）だけに置くと、Session が作れない予算で見積りだけが数を返す。
  const budgetBytes = options.planBackingBudgetBytes ?? DEFAULT_PLAN_BACKING_BUDGET_BYTES;
  if (!Number.isSafeInteger(budgetBytes) || budgetBytes < 0) {
    throw new ExecutionError(
      `options.planBackingBudgetBytes ${
        describeInputValue(budgetBytes)
      } は非負の安全な整数でなければならない`,
    );
  }
  // 行ブロックの一時 1 本ごとの予算も同じ位置・同じ文言（`sessionOptionsViolation` の門 — ADR 0124
  // 決定 5）。行ブロックで割る形を持たないグラフでも見る（値域門の位置をグラフの形に依らせない）。
  const attentionRowBlockMaxBytes = options.attentionRowBlockMaxBytes;
  if (
    attentionRowBlockMaxBytes !== undefined &&
    (!Number.isSafeInteger(attentionRowBlockMaxBytes) || attentionRowBlockMaxBytes < 1)
  ) {
    throw new ExecutionError(
      `options.attentionRowBlockMaxBytes ${
        describeInputValue(attentionRowBlockMaxBytes)
      } は正の安全な整数でなければならない`,
    );
  }
  const chunkDims = chunkRowDims(graph);
  const chunkSymbols = new Set(chunkDims.map((dim) => parseDim(dim).sym));
  const bindings = planBindings(graph, options.bindings, chunkSymbols);
  const state = stateEstimate(graph, options.generation, options);
  const weights = weightEstimate(residency);

  const chunkLength = options.generation?.chunkLength;
  if (chunkLength === undefined && chunkDims.length > 0) {
    // states 形ノードを持つグラフは GenerationContext 無しでは走らない（ADR 0066 決定 1）。
    // ここで落とさないと、chunk 記号が未束縛のまま planGraph の次元評価で落ちて、束縛点を
    // 名乗らない DimError になる。
    throw new ExecutionError(
      `物理 chunk 行 M の記号 [${chunkDims.join(", ")}] は options.generation.chunkLength で` +
        "しか束縛できない（states 形ノードを持つグラフは GenerationContext が要る）",
    );
  }
  // ADR 0066 決定 4 の実行 2 形をそれぞれ独立に計算する（chunk 行だけが違う同じグラフ）。
  const plans: [name: string, bindings: SymbolBindings][] = chunkLength === undefined
    ? [["run", bindings]]
    : [
      ["prefill", bindChunkRows(bindings, chunkDims, chunkLength, "prefill")],
      ["decode", bindChunkRows(bindings, chunkDims, 1, "decode")],
    ];
  // 呼び手が足す形（ADR 0096 の verify run など）は既存 2 形の後ろに宣言順で並ぶ。計算経路は
  // 同じ 1 本で、変わるのは物理 chunk 行と（渡されたなら）グラフ入力側の束縛だけ。
  const generation = options.generation;
  if (generation?.scenarios !== undefined) {
    // MUST: 名前の重複は fail loudly（予約名 3 つ + 追加名どうし）。同名が 2 本並ぶと、
    // 名前で引く呼び手が黙って先頭だけを見る。
    const taken = new Set<string>(["run", "prefill", "decode"] satisfies AdmissionScenarioName[]);
    for (const spec of generation.scenarios) {
      if (taken.has(spec.name)) {
        throw new ExecutionError(
          `シナリオ名 '${spec.name}' が既にある（予約名は run / prefill / decode）`,
        );
      }
      taken.add(spec.name);
      // MUST: 値域は `1..chunkLength`（`queryLength ≤ chunkLength` の契約と同じ上限 —
      // 宣言 shape に載らない行数の見積りを返さない）。
      if (
        !Number.isSafeInteger(spec.chunkLength) || spec.chunkLength < 1 ||
        spec.chunkLength > generation.chunkLength
      ) {
        throw new ExecutionError(
          `シナリオ '${spec.name}': 物理 chunk 行 ${spec.chunkLength} が ` +
            `1..${generation.chunkLength}（generation.chunkLength）の整数でない`,
        );
      }
      // 束縛は既定の写しに上書きしてから、既存 2 形と同じ門（`planBindings`）を通す。
      const base = planBindings(
        graph,
        { ...options.bindings, ...spec.bindings },
        chunkSymbols,
      );
      plans.push([spec.name, bindChunkRows(base, chunkDims, spec.chunkLength, spec.name)]);
    }
  }

  // `chunkBuckets`（ADR 0066 追記〈バケット〉）の各形はピークにだけ入れる内部の計画で、公開の
  // シナリオには並べない（名前で引く呼び手の列を動かさない — ADR 0095 追記 2026-10-10）。
  // MUST: ピークを最大 M の prefill 形だけで代表させない。行ブロックの等分（planRowBlocks）は
  // 枚数が 1 → 2 に変わる境界で行数の少ない形のほうが一時を多く使う（例: 1 枚 32 行の
  // 128 MiB に対し 33 行は 17 + 16 行の 2 枚で約 68 MiB）ので、中間は M に単調でない。許される
  // 物理行数を全て計算して最大を取れば、分割規則が変わっても仮定が崩れない。
  const buckets: [name: string, bindings: SymbolBindings][] = (generation?.chunkBuckets ?? []).map(
    (rows) => {
      const name = `bucket ${rows}`;
      return [name, bindChunkRows(bindings, chunkDims, rows, name)];
    },
  );

  const allowedRows = generation === undefined
    ? undefined
    : allowedChunkRows(generation.chunkLength, generation.chunkBuckets ?? []);
  const estimateScenario = (
    [name, scenarioBindings]: [name: string, bindings: SymbolBindings],
  ): AdmissionScenario => {
    // MUST: states と入力の両方に現れる記号は 2 つの束縛点で同じ値（run が拒否する分裂 —
    // ADR 0066 追記 7 — に見積りだけが正常値を返さない）。シナリオごとに解決値が変わるので
    // 照合もシナリオごと。
    if (state.bindings !== undefined) assertGenerationBindings(state.bindings, scenarioBindings);
    const plan = planGraph(graph, scenarioBindings, state.shapes);
    // MUST: 解決した物理 chunk 行は run 前検査と同じ許可集合で照合する（同じ 1 本 —
    // `assertChunkRowsAllowed`）。数値で固定した `M` や追加シナリオの行数が集合に無い形は、
    // run が最初の 1 回で必ず拒否するので、見積りだけが正常値を返さない。
    if (allowedRows !== undefined) assertChunkRowsAllowed(stateChunkRows(plan.nodes), allowedRows);
    // 入力バッファと出力 readback staging は同じ床つきの式（executor の `#bindInput` /
    // `#activateBacking` / `#stageOutputs` — `Math.max(4, numel×4)` は toSizeClass と同値）。
    let ioBytes = 0;
    for (const spec of graph.inputs) {
      ioBytes += toSizeClass(numel(shapeOf(plan.shapes, spec.name)) * 4);
    }
    for (const name of graph.outputs) ioBytes += toSizeClass(numel(shapeOf(plan.shapes, name)) * 4);
    return {
      name,
      ioBytes,
      workspaceBytes: transientSlotBytes(
        graph,
        plan.nodes,
        state.shapes,
        options.maxStorageBufferBindingSize,
        options.maxBufferSize,
        { stateAttentionReduce, attentionRowBlockMaxBytes },
      ),
    };
  };
  const scenarios = plans.map(estimateScenario);

  const weightBytes = weights.compressed + weights.uncompressed + weights.expanded;
  const peakScenarioBytes = Math.max(
    ...[...scenarios, ...buckets.map(estimateScenario)].map((scenario) =>
      scenario.ioBytes + scenario.workspaceBytes
    ),
  );
  return {
    resident: {
      weights: {
        compressedBytes: weights.compressed,
        uncompressedBytes: weights.uncompressed,
        expandedBytes: weights.expanded,
        totalBytes: weightBytes,
      },
      stateBytes: state.bytes,
    },
    scenarios,
    planBackingBudgetBytes: budgetBytes,
    // 保持集合は `max(予算, 最大 1 本)` を超えない（executor の `#evictBackingsFor`）。
    peakAccountedBytes: weightBytes + state.bytes + Math.max(budgetBytes, peakScenarioBytes),
    unaccounted: UNACCOUNTED,
  };
};
