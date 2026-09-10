/**
 * ターン 1 本の run を**名前付きバケット**へ畳む（**純関数だけ** — GPU も I/O も時計も触らない）。
 *
 * 測るのは「自己採算ゲート付き（`auto`）のターンの時間が、どの局面に落ちているか」である。
 * `summary.ts` の `runs[kind]` は run の**形**（prefill / decode / draft / verify）で分けるので、
 * ゲートが落とした plain step と `plain` モードの decode が同じ欄に混ざり、探索バーストの verify も
 * 定常の投機と区別できない。局面を分けるのに要る情報は観測席の 1 通に全部載っている
 * （`GenerationRunPhase` の `wallMs` / `delivered` / `gate`）ので、ここはその読み替えだけを持つ。
 *
 * 分割は 2 系統ある — **局面別**（`speculate` … `unmeasured`・run 1 本はちょうど 1 つに入る）と
 * **位置別**（`cold` / `rest`・ターン先頭の立ち上がりを分ける）で、同じ run が両方に入る。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 */

import type { GenerationGateTrace, GenerationRunPhase } from "../../packages/models/gemma.ts";

/**
 * ターン先頭の何 run を `cold` に入れるか。
 *
 * 先頭の数 run は PLE shard の cold miss と PreparedPlan / bind group の初出を含む（ゲートが
 * 最初の cycle を観測に混ぜないのも同じ理由 — `speculation-gate.ts`）。8 本なのはターン先頭の
 * 立ち上がりを見る粒度で、既定の探索間隔（`exploreBase` = 16）の半分にあたる。定常の壁はその後ろの
 * `rest` で読む。
 */
const COLD_RUNS = 8;

/** 1 バケットぶんの集計（本数・壁の合計・確定した token 数の合計）。 */
export type TraceBucket = {
  readonly runs: number;
  /** その run の壁の合計（ms — `GenerationRunPhase.wallMs` の和）。 */
  readonly ms: number;
  /** その run が確定させた token 数の合計（verify は cycle の確定数・decode は 1）。 */
  readonly delivered: number;
};

/**
 * 1 ターンぶんの内訳（JSON の `turns[].trace` にこのまま載る）。
 *
 * 局面別の 6 つは互いに排他で、合計が「生成相の run 全部」になる（prefill と draft は入らない —
 * どちらも壁を名乗らない）。`cold` / `rest` はその全部を別の軸で 2 つに割ったものである。
 */
export type TurnTrace = {
  /** verify・ゲート無し or（定常 speculate かつこのターンで 1 度も切り替わっていない）。 */
  readonly speculate: TraceBucket;
  /** verify・定常 speculate かつこのターンで切り替わっている（plain から戻った後の投機）。 */
  readonly speculateAfterReturn: TraceBucket;
  /** verify・定常 plain（plain 側の探索バースト）。 */
  readonly burst: TraceBucket;
  /** decode・定常 speculate（speculate 中の `W1` プローブ）。 */
  readonly w1Probe: TraceBucket;
  /** decode・ゲート無し or 定常 plain（plain の定常）。 */
  readonly plain: TraceBucket;
  /** `gate.measured === false` の run（壁は載っているが判定の材料ではない）。 */
  readonly unmeasured: TraceBucket;
  /**
   * 最初に `gate.switches` が 1 になった run の、そのターン内での通し番号（1 始まり）。
   *
   * 数えるのは `cold` / `rest` と同じ列（prefill と draft を除く decode / verify）である。
   * 切替が 1 度も起きなければ**欄ごと無い** — 0 は「1 本目で切り替わった」と読めてしまう。
   */
  readonly firstExitRun?: number;
  /** ターン先頭 {@link COLD_RUNS} 本（全モード — cold の効きを見る）。 */
  readonly cold: TraceBucket;
  /** {@link COLD_RUNS} 本目より後ろの残り（全モード）。 */
  readonly rest: TraceBucket;
};

/** 集計中のバケット（{@link traceOf} の中だけで可変）。 */
type Tally = { runs: number; ms: number; delivered: number };

const emptyTally = (): Tally => ({ runs: 0, ms: 0, delivered: 0 });

const add = (tally: Tally, wallMs: number, delivered: number): void => {
  tally.runs += 1;
  tally.ms += wallMs;
  tally.delivered += delivered;
};

/**
 * 観測席の 1 ターンぶんの列 → 内訳。
 *
 * MUST: `wallMs` の無い run は落とす（黙って 0 にすると、その run の時間が内訳から消えたまま
 * 「host 時間」にも現れない = 表が時間を失う）。
 */
export const traceOf = (phases: readonly GenerationRunPhase[]): TurnTrace => {
  const speculate = emptyTally();
  const speculateAfterReturn = emptyTally();
  const burst = emptyTally();
  const w1Probe = emptyTally();
  const plain = emptyTally();
  const unmeasured = emptyTally();
  const cold = emptyTally();
  const rest = emptyTally();
  /** その run の局面（局面別の 6 つは互いに排他 — 表の合計が run の総数になる条件）。 */
  const placeOf = (kind: "decode" | "verify", gate: GenerationGateTrace | undefined): Tally => {
    // ゲートの居ないターン（`plain` / `always`）は形がそのまま局面である。
    if (gate === undefined) return kind === "verify" ? speculate : plain;
    if (!gate.measured) return unmeasured;
    if (kind === "verify") {
      // 定常 plain の verify は探索バースト。定常 speculate なら、このターンで 1 度も切り替わって
      // いなければ定常の投機・切り替わっていれば plain から戻った後の投機である。
      return gate.mode === "plain" ? burst : gate.switches === 0 ? speculate : speculateAfterReturn;
    }
    return gate.mode === "speculate" ? w1Probe : plain;
  };

  /** 生成相の run の通し番号（1 始まり）。 */
  let index = 0;
  let firstExitRun: number | undefined;
  for (const phase of phases) {
    // 数えるのは生成相の 2 種別だけ。prefill はターンの立ち上げ（`firstTokenMs` が持つ）で、
    // draft は借り手の run（貸し手の cycle の壁の**内側**にある）なので、どちらも壁を名乗らない。
    if (phase.kind !== "decode" && phase.kind !== "verify") continue;
    index += 1;
    const { wallMs, gate } = phase;
    if (wallMs === undefined) {
      throw new Error(
        `[mtp-bench] ${phase.kind} run ${index}: 観測に wallMs が無い（生成面が壁を載せていない）`,
      );
    }
    // decode は 1 token・verify はその cycle の確定数。`delivered` が無い観測では `accepted + 1`
    // で補う（停止 token で列挙を打ち切った cycle だけ 1 だけ過大になる近似である）。
    const delivered = phase.kind === "verify" ? phase.delivered ?? phase.accepted + 1 : 1;
    // `switches` は単調に増えるので、最初に 1 以上を名乗った run が「1 になった run」である。
    if (firstExitRun === undefined && gate !== undefined && gate.switches >= 1) {
      firstExitRun = index;
    }
    add(index <= COLD_RUNS ? cold : rest, wallMs, delivered);
    add(placeOf(phase.kind, gate), wallMs, delivered);
  }
  return {
    speculate,
    speculateAfterReturn,
    burst,
    w1Probe,
    plain,
    unmeasured,
    ...(firstExitRun === undefined ? {} : { firstExitRun }),
    cold,
    rest,
  };
};
