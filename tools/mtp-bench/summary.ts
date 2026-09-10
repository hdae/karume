/**
 * MTP 計測のターン記録と要約の計算（**純関数だけ** — GPU も I/O も時計も触らない）。
 *
 * 測るのは「投機を張ると 1 token が何 ms 速くなるか」で、割り算の分母を間違えると答えが変わる
 * ので、式はこのモジュール 1 本に集める（`main.ts` は記録を積むだけ）:
 *
 * - **`tokens − 1` で割る** — 最初の token は prefill 直後の同じ run で出るので、decode / cycle の
 *   仕事は `tokens − 1` 個ぶんしか無い。`tokens` で割ると全モード一律に速く見え、比（取り分）も
 *   歪む。
 * - **中央値を採る**（平均ではない）— 1 ターンの間に PLE shard の読み直しやドライバの clock 変化が
 *   入ると壁が跳ねる。交互に回すのは順序効果を打ち消すためで、跳ねの側は中央値で落とす。
 * - **暖機は要約から除く**（記録には残す）— 初回ターンはパイプラインの立ち上げ（WGSL の解析・
 *   params の生成）を含み、定常の 1 token とは別物である。
 * - **warm（`--warm`）では自ターン番号を揃える** — sequence を使い回す走行ではターンごとに
 *   context が伸びるので、1 round に 4 本回る plain と 2 本の投機モードを全部混ぜると
 *   「長い会話の plain」と「短い会話の always」の比になる（{@link summarizeTurns}）。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 */

import type { GenerationSpeculation } from "../../packages/models/gemma.ts";
import type { TraceBucket, TurnTrace } from "./trace.ts";

/**
 * 測る 3 つの構成（`Gemma4SequenceOptions.speculative` の 3 値そのもの）。
 *
 * - `plain` … `speculative: false`（1 token = 1 run）
 * - `always` … `speculative: "always"`（ゲートを作らず常に投機）
 * - `auto` … `speculative: true`（自己採算ゲート付き — 負けている間は decode 形へ落ちる）
 *
 * `always` は「投機が最大でどれだけ効くか」、`auto` は「既定のまま付けて損しないか」を測る席で、
 * 2 つは別の問いである（同じ走行の中で両方を回すのはそのため）。
 */
export type BenchMode = "plain" | "always" | "auto";

/** モードの並び（`--gpu-timing` の表がこの順に mode の欄を書く — `timing.ts`）。 */
export const BENCH_MODES: readonly BenchMode[] = ["plain", "always", "auto"];

/** run の種別（`Gemma4RunPhase.kind` そのもの — 生成面が名乗る 4 つ）。 */
export type RunKind = "prefill" | "decode" | "draft" | "verify";

/** GPU 内訳の欄の並び（`--gpu-timing` の表がこの順に kind の欄を書く — `timing.ts`）。 */
export const RUN_KINDS: readonly RunKind[] = ["prefill", "decode", "draft", "verify"];

/**
 * 生成相に出る run の種別（`prefill` は最初の token より前に済むので入らない）。
 *
 * `hostMsPerToken` の分子は `generationMs`（= `turnMs − firstTokenMs`）から run 壁を引いた残りで、
 * その `generationMs` に prefill run の壁は**そもそも入っていない**。引くと二重に引くことになり、
 * 長文脈のワークロードでは負の「host 時間」が出る。
 */
const GENERATION_RUN_KINDS: readonly RunKind[] = ["decode", "draft", "verify"];

/** 1 ターンぶんの run 壁の集計（1 種別 = 本数 + 壁の合計）。 */
export type RunWall = {
  readonly count: number;
  /** その種別の run 壁の合計（ms — 発行から戻りまで。GPU 時間ではない）。 */
  readonly wallMs: number;
};

/** これから回す 1 ターン（{@link turnPlan} が組む台本）。 */
export type TurnPlan = {
  readonly mode: BenchMode;
  /** ローテーションの反復番号（1 始まり・暖機は 0）。 */
  readonly round: number;
  readonly warmup: boolean;
};

/** 1 ターンの記録（要約の材料 — JSON の `turns[]` にもこのまま載る）。 */
export type TurnRecord = TurnPlan & {
  /**
   * そのモードの何本目のターンか（1 始まり・**暖機を 1 本目として数える**）。
   *
   * `--warm`（モードごとに sequence を使い回す多ターン）の要約が揃える軸である。1 round に
   * plain は 4 本・投機の 2 モードは 2 本ずつ回るので、warm では同じ round の中でも plain の
   * 後ろのターンほど context が長い — 番号を揃えないと「長い会話の plain」と「短い会話の
   * always」を比べることになる。cold（1 ターン = 1 sequence）では要約は読まない。
   */
  readonly ownIndex: number;
  /**
   * 生成を発行する**前**に会話が占めていた論理位置の数（`GenerationSequence.used`）。
   *
   * cold では常に 0（毎ターン新しい sequence）。warm では前ターンまでの prompt と生成が
   * 積み上がった値で、閉じ札の frontier（未 commit の `pendingToken`）も 1 として入る。
   */
  readonly contextTokens: number;
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
  /**
   * 局面別の内訳（`trace.ts` — **全モードのターン**に載る）。
   *
   * `runs` が run の**形**で分ける表なのに対し、こちらは「投機の定常 / 探索バースト / `W1`
   * プローブ / plain」で分ける表である（`auto` のターンでは形が同じ run が別の局面にいる）。
   */
  readonly trace: TurnTrace;
  /** 投機の勘定（`GenerationStop.speculation` の写し — `always` / `auto` のターンにだけ載る）。 */
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

/**
 * 局面別内訳の要約（バケットごとに `runs` / `ms` / `delivered` の**それぞれ**を中央値にしたもの）。
 *
 * 3 つを別々に中央値へ落とすので、同じバケットの `ms / runs` は「どれか 1 ターンの 1 本あたりの
 * 壁」ではない（run 別の要約 {@link RunSummary} が `msPerRun` を per-turn の比の中央値で採るのと
 * 違う取り方である）。読むのは「ターン 1 本ぶんの内訳がどのくらいの規模か」で、1 本あたりの壁が
 * 要るなら `turns[].trace` から採る。
 */
export type TraceSummary =
  & {
    readonly [B in Exclude<keyof TurnTrace, "firstExitRun">]: TraceBucket;
  }
  & {
    /** {@link TurnTrace.firstExitRun} の中央値（切替が起きたターンだけの中央値・無ければ欄ごと無い）。 */
    readonly firstExitRun?: number;
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
  /** 局面別の内訳（**全モード** — {@link TraceSummary}）。 */
  readonly trace: TraceSummary;
  /**
   * GPU の外で使った時間（ms/token — **全モード**）。
   *
   * `(generationMs − 生成相の run 壁の和) / (tokens − 1)`（{@link GENERATION_RUN_KINDS}）。
   * 生成相の壁のうち run の外側（抽選・復号・イベント配送・PLE の読み）に落ちているぶんで、
   * 投機で削れない下限を表す。token を分母に採るのは 3 モードで比べられる唯一の分母だから
   * である — `auto` は 1 ターンの中で cycle と plain step が混ざるので cycle では割れない。
   */
  readonly hostMsPerToken: number;
  /**
   * 1 cycle で確定した token 数（{@link tokensPerCycle} の中央値 — `always` / `auto`）。
   *
   * `auto` では**投機 cycle だけ**の勘定である（plain step は `cycles` に入らない）。
   */
  readonly tokensPerCycle?: number;
  /**
   * 実効 k（1 cycle で引く draft の本数・`always` / `auto`）。
   *
   * 受理数ヒストグラムの長さ − 1 から出す（`createSpeculationTally` が長さ `k+1` で作る —
   * `packages/models/src/generation/speculation.ts`）。公開面に k の定数は無いので、走行が
   * 実際に何段で回ったのかを名乗れる口はここだけである（`--k` の省略時は配布形の段数で回る）。
   */
  readonly k?: number;
  /**
   * 1 cycle の壁（ms — `generationMs / cycles`・**`always` だけ**）。
   *
   * `auto` に出さないのは分母が定まらないため。生成相の壁には plain step のぶんも入っているのに
   * `cycles` は投機 cycle しか数えないので、割った値はどちらの 1 本の壁でもない。
   */
  readonly cycleMs?: number;
  /**
   * run の外側に落ちた時間（ms/cycle — `(generationMs − draft − verify の run 壁) / cycles`・
   * **`always` だけ**）。`auto` に出さない理由は {@link cycleMs} と同じである。
   */
  readonly hostMsPerCycle?: number;
  /** 受理数ヒストグラムの**合計**（添字 = その cycle の受理数 `a`・`always` / `auto`）。 */
  readonly acceptedHistogram?: readonly number[];
  /**
   * 自己採算ゲートが decode 形（M=1）で回した step 数の中央値（**`auto` だけ**）。
   *
   * ゲートが何度落ちたかそのもの。0 なら「1 度も落ちなかった」= そのワークロードでは投機が
   * 終始勝っていた、と読む。
   */
  readonly plainSteps?: number;
  /** ゲートが speculate ↔ plain を切り替えた回数の中央値（**`auto` だけ**）。 */
  readonly switches?: number;
};

/** token 列の一致（投機は「速度だけのノブ」— 列が動いていたら測る意味が無い）。 */
export type IdentitySummary = {
  /**
   * 暖機を除く `plain` 同士の `ids` が全て一致するか。
   *
   * warm では**欄ごと無い** — モードの中の各ターンは会話の続きなので、同じ prompt を 2 度流した
   * ときの再現性という問いがそもそも立たない（`false` を書くと破れの報せに読める）。
   */
  readonly plainConsistent?: boolean;
  readonly alwaysConsistent?: boolean;
  readonly autoConsistent?: boolean;
  /**
   * `plain` と `always` の `ids` が一致するか。
   *
   * cold は最初の非暖機ターン同士 1 対。warm は**自ターン番号ごと**（{@link TurnRecord.ownIndex}）
   * に対を作り、全ての番号で一致したときだけ立つ。
   */
  readonly identical: boolean;
  /** 一致しないなら最初に食い違った添字（片方が他方の接頭辞なら短い側の長さ）。 */
  readonly firstDivergence?: number;
  /**
   * warm で最初に食い違った**自ターン番号**（cold では欄ごと無い — 突合は 1 対だけなので
   * 番号を名乗る意味が無い）。
   */
  readonly firstDivergenceTurn?: number;
  /** 同じ突合を `plain` と `auto` で見たもの。 */
  readonly identicalAuto: boolean;
  readonly firstDivergenceAuto?: number;
  readonly firstDivergenceAutoTurn?: number;
  /**
   * warm で、同じ自ターン番号の 3 モードの context 長（{@link TurnRecord.contextTokens}）が
   * 全て揃っていたか（cold では**欄ごと無い** — 毎ターン新しい sequence なので常に 0 である）。
   *
   * 番号を揃えても、context が揃っていなければ「同じ位置から始まったターン」ではない。ずれるのは
   * 片側の列が割れた後（生成した token 数が違えば以後の context も違う）で、それ以降のターンは
   * 別の長さの会話の比較になる。落とさずに記録するのは、列の破れ自体が既知の限界
   * （`docs/limitations.md`）で、読めるようにするのがこの欄の役目だからである。
   */
  readonly contextAligned?: boolean;
  /** 揃わなかった最初の自ターン番号（揃っていれば・cold では欄ごと無い）。 */
  readonly firstContextMismatchTurn?: number;
};

export type BenchSummary = {
  readonly plain: ModeSummary;
  readonly always: ModeSummary;
  readonly auto: ModeSummary;
  /** 取り分（`plain.msPerToken / always.msPerToken` — 1 より大きいほど投機が速い）。 */
  readonly speedup: number;
  /**
   * 既定のまま付けたときの取り分（`plain.msPerToken / auto.msPerToken`）。
   *
   * 「付けたままで損しないか」の検収値である。`speedup` が投機の伸びしろを測るのに対し、
   * こちらは**ゲートが負けを止められているか**を測る — 1 を割ったらゲートの失敗である。
   */
  readonly speedupAuto: number;
  readonly identity: IdentitySummary;
  /**
   * warm の要約に入れた自ターン番号の上限（cold では欄ごと無い）。
   *
   * 3 モードに揃う番号だけを採る（= 各モードの本数の最小 — {@link summarizeTurns}）。倍率も
   * 列一致もこの範囲の中で出た値である。
   */
  readonly ownTurnLimit?: number;
};

/**
 * 1 反復のローテーション（順序効果を打ち消す最小単位）。
 *
 * `plain` を 2 モードそれぞれの前後に置く（奇数 round は P S S P P A A P・偶数 round は
 * P A A P P S S P）。`always` と `auto` を隣り合わせにしないのは、2 つの投機モードの差が
 * 「後のほうが速い / 遅い」に乗るのを避けるためで、両者が挟む `plain` は同じ本数（各 2 本）で
 * ある。round ごとに 2 モードの席を入れ替えるのは、1 プロセスの中で GPU が温まって後半ほど遅く
 * なる単調な漂流（RTX の実測で 1 プロセス ≈ 3 分の間に plain が +4%）が、固定順だと常に後席の
 * モードに乗るため — 交互にすれば中央値の比からその偏りが消える。
 */
const ROTATION_ODD: readonly BenchMode[] = [
  "plain",
  "always",
  "always",
  "plain",
  "plain",
  "auto",
  "auto",
  "plain",
];
const ROTATION_EVEN: readonly BenchMode[] = [
  "plain",
  "auto",
  "auto",
  "plain",
  "plain",
  "always",
  "always",
  "plain",
];

/**
 * 回すターンの台本 — 暖機 3 本（各モード 1 本）の後に `rounds` 回のローテーション（奇数 round は
 * {@link ROTATION_ODD}・偶数 round は {@link ROTATION_EVEN}）。
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
    { mode: "always", round: 0, warmup: true },
    { mode: "auto", round: 0, warmup: true },
  ];
  for (let round = 1; round <= rounds; round += 1) {
    const rotation = round % 2 === 1 ? ROTATION_ODD : ROTATION_EVEN;
    for (const mode of rotation) plans.push({ mode, round, warmup: false });
  }
  return plans;
};

/**
 * warm で `turns` 本回したときに要る論理位置の最大（= **最後のターンのピーク**）。
 *
 * 1 本目は `promptTokens`（会話全体の描画）から始まり、2 本目以降は `turnTokens`（`gemma4ChatTurn`
 * の追記ぶん）だけを積む。1 ターンが会話に足す位置は「追記 + 生成」で、生成は最大
 * `newTokens`（停止 token を 1 個として数えた上限）である。よって最後のターンの発行時点の
 * 占有は `promptTokens + (turns − 1) × (newTokens + turnTokens)` で、そのターンが最大まで
 * 生成すると `+ newTokens − 1` 位置目まで届く（最初の token は prompt の最後の行が出すので
 * 1 引く — runtime 側の門 `pastLength + promptLength + maxNewTokens − 1 ≤ capacity` と同じ式）。
 */
export const warmPeakPositions = (input: {
  readonly turns: number;
  readonly promptTokens: number;
  readonly turnTokens: number;
  readonly newTokens: number;
}): number => {
  if (!Number.isInteger(input.turns) || input.turns < 1) {
    throw new Error(`warm の自ターン数 ${input.turns} が 1 以上の整数でない`);
  }
  return input.promptTokens + input.turns * input.newTokens +
    (input.turns - 1) * input.turnTokens - 1;
};

/**
 * warm の走行が容量に入ることを**測る前に**確かめる（入らないなら fail loudly）。
 *
 * 走ってから溢れると、そのモードの途中までのターンだけが記録に残る（要約は 3 モードに揃う番号
 * しか採らないので、数字は出るのに片側だけ短い走行になる）。長文脈のワークロード（prompt ≈4.8K）は
 * 既定の `--rounds` では入らない — warm は**負ける課題**（自由文・対話）のための口なので、
 * それでよい。
 */
export const assertWarmCapacity = (request: {
  readonly plans: readonly TurnPlan[];
  readonly capacity: number;
  readonly promptTokens: number;
  readonly turnTokens: number;
  readonly newTokens: number;
}): void => {
  const over: string[] = [];
  for (const mode of BENCH_MODES) {
    const turns = request.plans.filter((plan) => plan.mode === mode).length;
    if (turns === 0) throw new Error(`--warm: mode ${mode} のターンが台本に 1 本も無い`);
    const needed = warmPeakPositions({ ...request, turns });
    if (needed > request.capacity) over.push(`${mode} は自ターン ${turns} 本で ${needed} 位置`);
  }
  if (over.length === 0) return;
  throw new Error(
    `--warm: 会話が capacity ${request.capacity} に入らない（${over.join(" / ")}` +
      ` — prompt ${request.promptTokens} + 追記 ${request.turnTokens}/ターン +` +
      ` 生成 ${request.newTokens}/ターン）。--rounds か --new-tokens を下げるか、` +
      `--capacity を上げること（長文脈のワークロードは warm の対象外）`,
  );
};

/**
 * 前ターンの停止のうち、次のターンの前置を決めるのに要る枝だけ（`GenerationStop` がそのまま入る）。
 *
 * 停止 token で閉じた 2 枝（配布形が宣言した `eos` と、要求が足した `stop-token`）だけが token を
 * 運ぶ — 残りの 3 枝は「model turn を閉じていない」側である。
 */
export type WarmPriorStop =
  | { readonly reason: "eos" | "stop-token"; readonly token: number }
  | { readonly reason: "max-tokens" | "aborted" | "closed" };

/**
 * warm の次のターンの id 列に前置するもの（`[]` = 前置しない）。
 *
 * `gemma4ChatTurn` の差分は「前の model turn を閉じる閉じ札は sequence の frontier が前置する」
 * 前提で描かれる。前ターンの停止で 3 つに分かれる:
 *
 * - **閉じ札で閉じた** … frontier がその id なので前置しない。
 * - **打ち切った**（`--new-tokens` の `max-tokens`・消費側が閉じた `closed` / `aborted` も同じ）…
 *   frontier は本文の token なので、閉じ札を 1 個前置して model turn を閉じる。生成が閉じ札まで
 *   出したときと同じ id 列になる。
 * - **閉じ札以外の停止 token で閉じた**（配布形の `<eos>`・要求が足した `<|tool_response>` など）…
 *   前置すると `本文 <eos> <turn|> 差分` という、会話を全部描き直しても出ない id 列を KV に積む
 *   ことになり、前置しなければ `gemma4ChatTurn` の前提が破れる。どちらも黙って測ってはいけない
 *   ので**落とす**（warm はこの形を扱わない）。
 */
export const warmTurnPrefix = (input: {
  readonly prior: WarmPriorStop;
  readonly mode: BenchMode;
  readonly endOfTurnId: number;
}): readonly number[] => {
  const { prior, mode, endOfTurnId } = input;
  if (prior.reason !== "eos" && prior.reason !== "stop-token") return [endOfTurnId];
  if (prior.token === endOfTurnId) return [];
  throw new Error(
    `--warm: mode ${mode} の前のターンが閉じ札（id ${endOfTurnId}）ではない停止 ` +
      `token id ${prior.token}（${prior.reason}）で閉じた — warm はこの形を扱わない` +
      `（閉じ札を前置すると、会話を描き直しても出ない id 列を KV に積むことになる）。` +
      `--warm を外して cold で測ること`,
  );
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
 * 1 cycle で確定した token 数（**`delivered / cycles`**）。
 *
 * 正本が `delivered`（cycle が確定させた token の総数）なのは、受理した draft が停止 token だった
 * cycle はそこで列挙を打ち切るためである — 旧来の `(accepted + cycles) / cycles` は「棄却でも
 * frontier は 1 個進む」を全 cycle に当てはめるので、その cycle 1 本につき分子が 1 だけ過大に
 * なる（`GenerationSpeculation` の doc）。`cycles === 0` は割れないので落とす — 0 を返すと
 * 「1 cycle で 0 token 進んだ」と読める。
 */
export const tokensPerCycle = (speculation: GenerationSpeculation): number => {
  if (speculation.cycles === 0) {
    throw new Error("tok/cycle: cycles 0 では割れない（投機の cycle が 1 本も回っていない）");
  }
  return speculation.delivered / speculation.cycles;
};

/** 投機のターンには必ず勘定が載る（無ければ簿記の破れ — 黙って 0 として数えない）。 */
const speculationOf = (turn: TurnRecord): GenerationSpeculation => {
  if (turn.speculation === undefined) {
    throw new Error(`${turn.mode} round ${turn.round}: GenerationStop.speculation が無い`);
  }
  return turn.speculation;
};

/**
 * ゲート付き（`auto`）のターンにだけ載る欄。
 *
 * MUST: 欄の不在を 0 に丸めない — `0` は「ゲートが 1 度も落ちなかった」という観測で、欄が無いのは
 * 観測そのものが無い（= `"always"` の勘定が `auto` の席に紛れている）ことである。
 */
const gatedCountOf = (turn: TurnRecord, field: "plainSteps" | "switches"): number => {
  const value = speculationOf(turn)[field];
  if (value === undefined) {
    throw new Error(
      `auto round ${turn.round}: GenerationSpeculation.${field} が無い（ゲート付きの勘定でない）`,
    );
  }
  return value;
};

/** 生成相に出た run 壁の合計（ms — prefill は入らない）。 */
const generationRunWallMs = (turn: TurnRecord): number =>
  GENERATION_RUN_KINDS.reduce((sum, kind) => sum + turn.runs[kind].wallMs, 0);

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

/**
 * 局面別内訳の中央値（{@link TraceSummary}）。
 *
 * `firstExitRun` は**起きたターンだけ**の中央値である（起きなかったターンを 0 で埋めると
 * 「1 本目より前に切り替わった」という有り得ない値が中央へ寄る）。
 */
const summarizeTrace = (traces: readonly TurnTrace[]): TraceSummary => {
  const bucket = (of: (trace: TurnTrace) => TraceBucket): TraceBucket => ({
    runs: median(traces.map((trace) => of(trace).runs)),
    ms: median(traces.map((trace) => of(trace).ms)),
    delivered: median(traces.map((trace) => of(trace).delivered)),
  });
  const exits = traces.flatMap((trace) =>
    trace.firstExitRun === undefined ? [] : [trace.firstExitRun]
  );
  return {
    speculate: bucket((trace) => trace.speculate),
    speculateAfterReturn: bucket((trace) => trace.speculateAfterReturn),
    burst: bucket((trace) => trace.burst),
    w1Probe: bucket((trace) => trace.w1Probe),
    plain: bucket((trace) => trace.plain),
    unmeasured: bucket((trace) => trace.unmeasured),
    cold: bucket((trace) => trace.cold),
    rest: bucket((trace) => trace.rest),
    ...(exits.length === 0 ? {} : { firstExitRun: median(exits) }),
  };
};

const summarizeMode = (measured: readonly TurnRecord[], mode: BenchMode): ModeSummary => {
  const turns = measured.filter((turn) => turn.mode === mode);
  if (turns.length === 0) throw new Error(`mode ${mode} の（暖機を除く）ターンが 1 本も無い`);
  const base = {
    turns: turns.length,
    msPerToken: median(turns.map((turn) => turn.generationMs / tokensAfterFirst(turn))),
    generationMs: median(turns.map((turn) => turn.generationMs)),
    turnMs: median(turns.map((turn) => turn.turnMs)),
    firstTokenMs: median(turns.map((turn) => turn.firstTokenMs)),
    runs: {
      prefill: summarizeRun(turns, "prefill"),
      decode: summarizeRun(turns, "decode"),
      draft: summarizeRun(turns, "draft"),
      verify: summarizeRun(turns, "verify"),
    },
    trace: summarizeTrace(turns.map((turn) => turn.trace)),
    hostMsPerToken: median(
      turns.map((turn) => (turn.generationMs - generationRunWallMs(turn)) / tokensAfterFirst(turn)),
    ),
  };
  if (mode === "plain") return base;
  const tallies = turns.map((turn) => speculationOf(turn));
  const speculative = {
    ...base,
    tokensPerCycle: median(tallies.map((tally) => tokensPerCycle(tally))),
    k: effectiveK(tallies),
    acceptedHistogram: sumHistograms(tallies.map((tally) => tally.acceptedHistogram)),
  };
  if (mode === "always") {
    return {
      ...speculative,
      cycleMs: median(turns.map((turn, at) => turn.generationMs / tallies[at].cycles)),
      hostMsPerCycle: median(
        turns.map((turn, at) =>
          (turn.generationMs - turn.runs.draft.wallMs - turn.runs.verify.wallMs) /
          tallies[at].cycles
        ),
      ),
    };
  }
  return {
    ...speculative,
    plainSteps: median(turns.map((turn) => gatedCountOf(turn, "plainSteps"))),
    switches: median(turns.map((turn) => gatedCountOf(turn, "switches"))),
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

/**
 * warm の突合 — **自ターン番号ごと**に `plain` と比べ、最初に割れた (番号, 位置) を返す。
 *
 * warm ではターンごとに会話が伸びるので、番号を揃えない突合は「別の会話の列」を比べてしまう
 * （必ず割れる）。番号が片側にしか無いターンは飛ばす（要約に入る範囲は揃っているので、実際に
 * 飛ぶのは要約の外を渡されたときだけ）。
 */
const firstDivergenceByOwnTurn = (
  plain: readonly TurnRecord[],
  other: readonly TurnRecord[],
): { readonly turn: number; readonly at: number } | undefined => {
  const mates = new Map(other.map((turn) => [turn.ownIndex, turn]));
  const ordered = [...plain].sort((left, right) => left.ownIndex - right.ownIndex);
  let compared = 0;
  for (const turn of ordered) {
    const mate = mates.get(turn.ownIndex);
    if (mate === undefined) continue;
    compared += 1;
    const at = firstDivergenceOf(turn.ids, mate.ids);
    if (at !== undefined) return { turn: turn.ownIndex, at };
  }
  if (compared === 0) {
    throw new Error("token 列の突合: plain と揃う自ターン番号が 1 つも無い");
  }
  return undefined;
};

/**
 * warm の context 長の検査 — 自ターン番号ごとに 3 モードの {@link TurnRecord.contextTokens} を
 * 比べ、最初に揃わなかった番号を返す（全て揃っていれば `undefined`）。
 *
 * 見るのは要約に入るターンだけ（呼び手が絞った集合）で、片側にしか無い番号は 1 本だけの群に
 * なるので自動的に揃う扱いになる。
 */
const firstContextMismatchOf = (turns: readonly TurnRecord[]): number | undefined => {
  const byOwnTurn = new Map<number, number[]>();
  for (const turn of turns) {
    const seen = byOwnTurn.get(turn.ownIndex);
    if (seen === undefined) byOwnTurn.set(turn.ownIndex, [turn.contextTokens]);
    else seen.push(turn.contextTokens);
  }
  const ordered = [...byOwnTurn.entries()].sort(([left], [right]) => left - right);
  for (const [ownIndex, contexts] of ordered) {
    if (contexts.some((one) => one !== contexts[0])) return ownIndex;
  }
  return undefined;
};

const summarizeIdentity = (measured: readonly TurnRecord[], warm: boolean): IdentitySummary => {
  const ofMode = (mode: BenchMode): readonly TurnRecord[] =>
    measured.filter((turn) => turn.mode === mode);
  const plain = ofMode("plain");
  const always = ofMode("always");
  const auto = ofMode("auto");
  if (plain.length === 0 || always.length === 0 || auto.length === 0) {
    throw new Error("token 列の突合: 暖機を除くターンが無いモードがある");
  }
  if (warm) {
    const divergence = firstDivergenceByOwnTurn(plain, always);
    const divergenceAuto = firstDivergenceByOwnTurn(plain, auto);
    const contextMismatch = firstContextMismatchOf(measured);
    return {
      identical: divergence === undefined,
      ...(divergence === undefined
        ? {}
        : { firstDivergence: divergence.at, firstDivergenceTurn: divergence.turn }),
      identicalAuto: divergenceAuto === undefined,
      ...(divergenceAuto === undefined
        ? {}
        : { firstDivergenceAuto: divergenceAuto.at, firstDivergenceAutoTurn: divergenceAuto.turn }),
      contextAligned: contextMismatch === undefined,
      ...(contextMismatch === undefined ? {} : { firstContextMismatchTurn: contextMismatch }),
    };
  }
  const consistent = (turns: readonly TurnRecord[]): boolean =>
    turns.every((turn) => sameIds(turn.ids, turns[0].ids));
  const divergence = firstDivergenceOf(plain[0].ids, always[0].ids);
  const divergenceAuto = firstDivergenceOf(plain[0].ids, auto[0].ids);
  return {
    plainConsistent: consistent(plain),
    alwaysConsistent: consistent(always),
    autoConsistent: consistent(auto),
    identical: divergence === undefined,
    ...(divergence === undefined ? {} : { firstDivergence: divergence }),
    identicalAuto: divergenceAuto === undefined,
    ...(divergenceAuto === undefined ? {} : { firstDivergenceAuto: divergenceAuto }),
  };
};

/**
 * 3 モードに揃う自ターン番号の上限（= 各モードの**本数**の最小・暖機込みで数える）。
 *
 * 暖機は各モードちょうど 1 本目なので、本数の最小で切るのと「各モードの測定ターンを頭から
 * 同じ本数だけ採る」のは同じ集合になる（plain 13 本 / 投機 7 本 → 番号 7 まで = 測定 6 本ずつ）。
 */
const sharedOwnTurnLimit = (turns: readonly TurnRecord[]): number => {
  const counts = BENCH_MODES.map((mode) => ({
    mode,
    count: turns.filter((turn) => turn.mode === mode).length,
  }));
  const missing = counts.filter((one) => one.count === 0);
  if (missing.length > 0) {
    throw new Error(
      `warm の要約: mode ${missing.map((one) => one.mode).join(" / ")} のターンが 1 本も無い`,
    );
  }
  return Math.min(...counts.map((one) => one.count));
};

/** {@link summarizeTurns} のノブ。 */
export type SummarizeOptions = {
  /**
   * モードごとに sequence を使い回した走行か（`--warm`）。
   *
   * 立つと要約の対象が「3 モードに揃う自ターン番号」に絞られ、列一致は番号ごとの突合になる。
   */
  readonly warm?: boolean;
};

/** ターン記録（暖機込み）→ 要約。暖機はここで落ちる。 */
export const summarizeTurns = (
  turns: readonly TurnRecord[],
  options: SummarizeOptions = {},
): BenchSummary => {
  const warm = options.warm ?? false;
  const ownTurnLimit = warm ? sharedOwnTurnLimit(turns) : undefined;
  const measured = turns.filter((turn) => !turn.warmup);
  // warm では context 長がターンごとに伸びるので、モード間で番号が揃うところまでしか採らない。
  const used = ownTurnLimit === undefined
    ? measured
    : measured.filter((turn) => turn.ownIndex <= ownTurnLimit);
  const plain = summarizeMode(used, "plain");
  const always = summarizeMode(used, "always");
  const auto = summarizeMode(used, "auto");
  return {
    plain,
    always,
    auto,
    speedup: plain.msPerToken / always.msPerToken,
    speedupAuto: plain.msPerToken / auto.msPerToken,
    identity: summarizeIdentity(used, warm),
    ...(ownTurnLimit === undefined ? {} : { ownTurnLimit }),
  };
};
