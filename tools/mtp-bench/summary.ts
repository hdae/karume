/**
 * MTP 計測のターン記録と要約の計算（**純関数だけ** — GPU も I/O も時計も触らない）。
 *
 * 測るのは「投機を張ると 1 token が何 ms 速くなるか」で、割り算の分母を間違えると答えが変わる
 * ので、式はこのモジュール 1 本に集める（`main.ts` は記録を積むだけ）:
 *
 * - **`tokens − 1` で割る** — 最初の token は prefill 直後の同じ run で出るので、decode / cycle の
 *   仕事は `tokens − 1` 個ぶんしか無い。`tokens` で割ると両モードとも一律に速く見え、比（取り分）も
 *   歪む。
 * - **中央値を採る**（平均ではない）— 1 ターンの間に PLE shard の読み直しやドライバの clock 変化が
 *   入ると壁が跳ねる。ABBA で交互に回すのは順序効果を打ち消すためで、跳ねの側は中央値で落とす。
 * - **暖機は要約から除く**（記録には残す）— 初回ターンはパイプラインの立ち上げ（WGSL の解析・
 *   params の生成）を含み、定常の 1 token とは別物である。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 */

import type { GenerationSpeculation } from "../../packages/models/gemma.ts";

/** 測る 2 つの構成（`plain` = 非投機 = 1 token 1 run / `speculative` = 投機 = 1 cycle 複数 token）。 */
export type BenchMode = "plain" | "speculative";

/** run の種別（`Gemma4RunPhase.kind` そのもの — 生成面が名乗る 4 つ）。 */
export type RunKind = "prefill" | "decode" | "draft" | "verify";

/** GPU 内訳の欄の並び（`main.ts` が `--gpu-timing` の表をこの順に読む — 唯一の読み手）。 */
export const RUN_KINDS: readonly RunKind[] = ["prefill", "decode", "draft", "verify"];

/** 1 ターンぶんの run 壁の集計（1 種別 = 本数 + 壁の合計）。 */
export type RunWall = {
  readonly count: number;
  /** その種別の run 壁の合計（ms — 発行から戻りまで。GPU 時間ではない）。 */
  readonly wallMs: number;
};

/** これから回す 1 ターン（{@link turnPlan} が組む台本）。 */
export type TurnPlan = {
  readonly mode: BenchMode;
  /** ABBA の反復番号（1 始まり・暖機は 0）。 */
  readonly round: number;
  readonly warmup: boolean;
};

/** 1 ターンの記録（要約の材料 — JSON の `turns[]` にもこのまま載る）。 */
export type TurnRecord = TurnPlan & {
  /** 生成した token 数（`GenerationStop.tokens` — 停止 token も 1 個として数える）。 */
  readonly tokens: number;
  /** 停止理由（`GenerationStop.reason`）。 */
  readonly stopReason: string;
  /** `token` イベントの id 列（停止 token は含まれない — ビット同一性の突合はこの列で見る）。 */
  readonly ids: readonly number[];
  /** `ids` の復号（読んで意味のある文が出ているかを人が見るため）。 */
  readonly text: string;
  /** `generate` の発行から `done` まで（ms）。 */
  readonly turnMs: number;
  /** 発行から最初の `token` イベントまで（ms — prefill 相そのもの）。 */
  readonly firstTokenMs: number;
  /** `turnMs − firstTokenMs`（ms — 生成相だけの壁）。 */
  readonly generationMs: number;
  readonly runs: { readonly [K in RunKind]: RunWall };
  /** 投機の勘定（`GenerationStop.speculation` の写し — `speculative` のターンにだけ載る）。 */
  readonly speculation?: GenerationSpeculation;
};

/** 1 種別ぶんの要約（暖機を除くターンの中央値）。 */
export type RunSummary = {
  /** 1 ターンあたりの本数（中央値）。 */
  readonly countPerTurn: number;
  /**
   * run 1 本あたりの壁（ms — ターンごとの `wallMs / count` の中央値）。
   *
   * その種別の run が 1 本も無いターンだけの場合は**欄ごと無い**（`plain` の draft / verify）。
   * 0 を書くと「測ったら 0 ms だった」と読めてしまう。
   */
  readonly msPerRun?: number;
};

/** 1 モードぶんの要約（暖機を除く・全て中央値）。 */
export type ModeSummary = {
  /** 要約に入ったターン数（暖機を除く）。 */
  readonly turns: number;
  /** 1 token あたりの生成相の壁（ms — `generationMs / (tokens − 1)`）。この tool の主指標。 */
  readonly msPerToken: number;
  readonly generationMs: number;
  readonly turnMs: number;
  readonly firstTokenMs: number;
  readonly runs: { readonly [K in RunKind]: RunSummary };
  /**
   * GPU の外で使った時間（ms/token — `plain` だけ）。
   *
   * `(generationMs − decode の run 壁) / (tokens − 1)`。decode 相の壁のうち run の外側
   * （抽選・復号・イベント配送・PLE の読み）に落ちているぶんで、投機で削れない下限を表す。
   */
  readonly hostMsPerToken?: number;
  /** 1 cycle で確定した token 数（{@link tokensPerCycle} の中央値 — `speculative` だけ）。 */
  readonly tokensPerCycle?: number;
  /**
   * 実効 k（1 cycle で引く draft の本数・`speculative` だけ）。
   *
   * 受理数ヒストグラムの長さ − 1 から出す（`createSpeculationTally` が長さ `k+1` で作る —
   * `packages/models/src/generation/speculation.ts`）。公開面に k の定数は無いので、走行が
   * 実際に何段で回ったのかを名乗れる口はここだけである（`--k` の省略時は配布形の段数で回る）。
   */
  readonly k?: number;
  /** 1 cycle の壁（ms — `generationMs / cycles`・`speculative` だけ）。 */
  readonly cycleMs?: number;
  /** run の外側に落ちた時間（ms/cycle — `(generationMs − draft − verify の run 壁) / cycles`）。 */
  readonly hostMsPerCycle?: number;
  /** 受理数ヒストグラムの**合計**（添字 = その cycle の受理数 `a`・`speculative` だけ）。 */
  readonly acceptedHistogram?: readonly number[];
};

/** token 列の一致（投機は「速度だけのノブ」— 列が動いていたら測る意味が無い）。 */
export type IdentitySummary = {
  /** 暖機を除く `plain` 同士の `ids` が全て一致するか。 */
  readonly plainConsistent: boolean;
  readonly speculativeConsistent: boolean;
  /** `plain` の最初の非暖機ターンと `speculative` の最初の非暖機ターンの `ids` が一致するか。 */
  readonly identical: boolean;
  /** 一致しないなら最初に食い違った添字（片方が他方の接頭辞なら短い側の長さ）。 */
  readonly firstDivergence?: number;
};

export type BenchSummary = {
  readonly plain: ModeSummary;
  readonly speculative: ModeSummary;
  /** 取り分（`plain.msPerToken / speculative.msPerToken` — 1 より大きいほど投機が速い）。 */
  readonly speedup: number;
  readonly identity: IdentitySummary;
};

/** ABBA の 1 反復（順序効果を打ち消す最小単位）。 */
const ABBA: readonly BenchMode[] = ["plain", "speculative", "speculative", "plain"];

/**
 * 回すターンの台本 — 暖機 2 本（各モード 1 本）の後に `rounds` 回の ABBA。
 *
 * 暖機を**記録する**のは、立ち上げの費用がどれだけあったかを後から読めるようにするためで、
 * 要約には入らない（{@link summarizeTurns} が `warmup` で落とす）。
 */
export const turnPlan = (rounds: number): readonly TurnPlan[] => {
  if (!Number.isInteger(rounds) || rounds < 1) {
    throw new Error(`rounds ${rounds} が 1 以上の整数でない`);
  }
  const plans: TurnPlan[] = [
    { mode: "plain", round: 0, warmup: true },
    { mode: "speculative", round: 0, warmup: true },
  ];
  for (let round = 1; round <= rounds; round += 1) {
    for (const mode of ABBA) plans.push({ mode, round, warmup: false });
  }
  return plans;
};

/**
 * 中央値（昇順に並べて中央・偶数なら**上側** `[n >> 1]`）。
 *
 * 補間しないのは先例（`outputs/bench/.../turn-wall.ts`）と同じ取り方に揃えるためで、
 * 実測値そのものが返る = 出た数字がどのターンのものか辿れる。
 */
export const median = (values: readonly number[]): number => {
  if (values.length === 0) throw new Error("中央値: 空の列");
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[sorted.length >> 1];
};

/**
 * 生成相の仕事量にあたる token 数（最初の 1 個は prefill run が出すので引く）。
 *
 * 進捗行（`main.ts`）も要約もこの 1 本を呼ぶ — 同じ分母を 2 実装すると、片方だけ直した日に
 * 画面の数字と JSON の数字が食い違う。
 */
export const tokensAfterFirst = (turn: TurnRecord): number => {
  if (turn.tokens < 2) {
    throw new Error(
      `${turn.mode} round ${turn.round}: token ${turn.tokens} 個では ms/token を出せない` +
        "（最初の token は prefill run が出すので 2 個以上要る）",
    );
  }
  return turn.tokens - 1;
};

/**
 * 1 cycle で確定した token 数（`(accepted + cycles) / cycles`）。
 *
 * 棄却された cycle も frontier を 1 個は進めるので、1 cycle は `1 + a` token を確定させる。
 * `cycles === 0` は割れないので落とす — 0 を返すと「1 cycle で 0 token 進んだ」と読める。
 */
export const tokensPerCycle = (speculation: GenerationSpeculation): number => {
  if (speculation.cycles === 0) {
    throw new Error("tok/cycle: cycles 0 では割れない（投機の cycle が 1 本も回っていない）");
  }
  return (speculation.accepted + speculation.cycles) / speculation.cycles;
};

/** 投機のターンには必ず勘定が載る（無ければ簿記の破れ — 黙って 0 として数えない）。 */
const speculationOf = (turn: TurnRecord): GenerationSpeculation => {
  if (turn.speculation === undefined) {
    throw new Error(`speculative round ${turn.round}: GenerationStop.speculation が無い`);
  }
  return turn.speculation;
};

const summarizeRun = (turns: readonly TurnRecord[], kind: RunKind): RunSummary => {
  const countPerTurn = median(turns.map((turn) => turn.runs[kind].count));
  const ran = turns.filter((turn) => turn.runs[kind].count > 0);
  return {
    countPerTurn,
    ...(ran.length === 0
      ? {}
      : { msPerRun: median(ran.map((turn) => turn.runs[kind].wallMs / turn.runs[kind].count)) }),
  };
};

/** 受理数ヒストグラムの要素ごとの合計（長さは最長のものに合わせる）。 */
const sumHistograms = (histograms: readonly (readonly number[])[]): readonly number[] => {
  const length = histograms.reduce((max, one) => Math.max(max, one.length), 0);
  const total = new Array<number>(length).fill(0);
  for (const one of histograms) {
    one.forEach((count, at) => {
      total[at] += count;
    });
  }
  return total;
};

/**
 * 実効 k（ヒストグラムの長さ − 1）。走行の途中で段数が変わることは無いので、食い違えば落とす。
 *
 * 段数が変わる走行が仮にあれば、`acceptedHistogram` の要素ごとの合計（{@link sumHistograms}）は
 * 添字の意味が混ざった表になる — その表から倍率を読む前に落ちるほうが安い。
 */
const effectiveK = (tallies: readonly GenerationSpeculation[]): number => {
  const k = tallies[0].acceptedHistogram.length - 1;
  for (const tally of tallies) {
    const other = tally.acceptedHistogram.length - 1;
    if (other !== k) {
      throw new Error(`投機ターンの間で実効 k が食い違う（${k} と ${other}）`);
    }
  }
  return k;
};

const summarizeMode = (measured: readonly TurnRecord[], mode: BenchMode): ModeSummary => {
  const turns = measured.filter((turn) => turn.mode === mode);
  if (turns.length === 0) throw new Error(`mode ${mode} の（暖機を除く）ターンが 1 本も無い`);
  const runs = {
    prefill: summarizeRun(turns, "prefill"),
    decode: summarizeRun(turns, "decode"),
    draft: summarizeRun(turns, "draft"),
    verify: summarizeRun(turns, "verify"),
  };
  const base = {
    turns: turns.length,
    msPerToken: median(turns.map((turn) => turn.generationMs / tokensAfterFirst(turn))),
    generationMs: median(turns.map((turn) => turn.generationMs)),
    turnMs: median(turns.map((turn) => turn.turnMs)),
    firstTokenMs: median(turns.map((turn) => turn.firstTokenMs)),
    runs,
  };
  if (mode === "plain") {
    return {
      ...base,
      hostMsPerToken: median(
        turns.map((turn) => (turn.generationMs - turn.runs.decode.wallMs) / tokensAfterFirst(turn)),
      ),
    };
  }
  const tallies = turns.map((turn) => speculationOf(turn));
  return {
    ...base,
    tokensPerCycle: median(tallies.map((tally) => tokensPerCycle(tally))),
    k: effectiveK(tallies),
    cycleMs: median(turns.map((turn, at) => turn.generationMs / tallies[at].cycles)),
    hostMsPerCycle: median(
      turns.map((turn, at) =>
        (turn.generationMs - turn.runs.draft.wallMs - turn.runs.verify.wallMs) / tallies[at].cycles
      ),
    ),
    acceptedHistogram: sumHistograms(tallies.map((tally) => tally.acceptedHistogram)),
  };
};

const sameIds = (left: readonly number[], right: readonly number[]): boolean =>
  left.length === right.length && left.every((id, at) => id === right[at]);

/** 最初に食い違った添字（片方が接頭辞なら短い側の長さ）。一致していれば `undefined`。 */
const firstDivergenceOf = (
  left: readonly number[],
  right: readonly number[],
): number | undefined => {
  const shared = Math.min(left.length, right.length);
  for (let at = 0; at < shared; at += 1) {
    if (left[at] !== right[at]) return at;
  }
  return left.length === right.length ? undefined : shared;
};

const summarizeIdentity = (measured: readonly TurnRecord[]): IdentitySummary => {
  const plain = measured.filter((turn) => turn.mode === "plain");
  const speculative = measured.filter((turn) => turn.mode === "speculative");
  if (plain.length === 0 || speculative.length === 0) {
    throw new Error("token 列の突合: 暖機を除くターンが片方のモードに無い");
  }
  const divergence = firstDivergenceOf(plain[0].ids, speculative[0].ids);
  return {
    plainConsistent: plain.every((turn) => sameIds(turn.ids, plain[0].ids)),
    speculativeConsistent: speculative.every((turn) => sameIds(turn.ids, speculative[0].ids)),
    identical: divergence === undefined,
    ...(divergence === undefined ? {} : { firstDivergence: divergence }),
  };
};

/** ターン記録（暖機込み）→ 要約。暖機はここで落ちる。 */
export const summarizeTurns = (turns: readonly TurnRecord[]): BenchSummary => {
  const measured = turns.filter((turn) => !turn.warmup);
  const plain = summarizeMode(measured, "plain");
  const speculative = summarizeMode(measured, "speculative");
  return {
    plain,
    speculative,
    speedup: plain.msPerToken / speculative.msPerToken,
    identity: summarizeIdentity(measured),
  };
};
