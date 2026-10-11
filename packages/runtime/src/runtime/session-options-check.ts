/**
 * {@link SessionOptions} の GPU の能力に依らない受理条件（型・綴り・組合せ・値域）と、省略された実行形ノブの
 * runtime 既定 — Session 構築（session-build.ts）と見積り（estimate.ts）が同じ 1 本で読む純関数。
 *
 * 構築相のモジュールから分けてあるのは、見積り（GPU に触らない純関数の面）が構築相の値 import を
 * 持たないため（依存は session-build.ts / estimate.ts → ここの一方向）。
 * MUST: GPU・device・バッファに触れる import を持ち込まない（見積りの依存がそれを引き込む）。
 */

import type { ScoreStorage } from "../kernels/score-storage.ts";
import { describeInputValue, ExecutionError } from "./plan.ts";
import {
  type ComputePrecision,
  DEFAULT_PLAN_BACKING_BUDGET_BYTES,
  type LinearGemvReduce,
  type SessionOptions,
  STATE_ATTENTION_REDUCES,
  type StateAttentionReduce,
} from "./session-types.ts";

/**
 * {@link SessionOptions} の実行形ノブの受理集合。
 *
 * MUST: 器は `Record<union, true>` — union に値を足してここを直し忘れると、キーの欠落が
 * **型検査で**赤くなる（値の配列で持つと、足した値が黙って受理集合から落ちる）。
 */
const LINEAR_COMPUTES: Readonly<Record<NonNullable<SessionOptions["linearCompute"]>, true>> = {
  f32: true,
  a8: true,
  f16: true,
};
const ATTENTION_COMPUTES: Readonly<Record<ComputePrecision, true>> = {
  f32: true,
  f16: true,
  a8: true,
};
const SCORE_STORAGES: Readonly<Record<ScoreStorage, true>> = { f32: true, f16: true };
const LINEAR_GEMV_REDUCES: Readonly<Record<LinearGemvReduce, true>> = {
  sequential: true,
  parallel: true,
  "parallel-subgroup32": true,
};

/**
 * 実行形ノブの綴りを {@link "./executor.ts"} の `Session.build` の入口で検査する。
 *
 * MUST: union 外の綴りは fail loudly。下流の消費は全て `=== "a8"` / `=== "f16"` /
 * `=== "parallel"` の等値比較なので、1 文字でも違えば**既定（f32 / sequential）で黙って走る** —
 * opt-in が適用されないまま「a8 を測った」と読める形になる。TS の型で守られているのは TS の
 * 呼び手だけで、JS の呼び手と、改名前の綴りを残したコード（`linearCompute` は 0.5.0 で
 * `"i8a8"` → `"a8"`・ADR 0074 決定 3・互換シムは置かない）は網の外にある。
 * MUST: 違反は**全件列挙して 1 回で落とす**（`assertWeightsWithinLimits` と同じ流儀 — 1 本ずつ
 * 落とすと、直すたびに次の 1 本が現れて何本直せば通るのかが最後まで分からない）。
 * MUST: 受理集合を引く**前に** `typeof` で型（文字列）を見る。`Object.hasOwn` は値をプロパティ
 * キーへ変換するので、この門が無いと `["a8"]` が `'a8'` として受理され、下流の厳密比較では
 * 外れて既定へ黙って縮退する。
 * MUST: 診断でも利用者の変換（`toString` / `Symbol.toPrimitive` / `toJSON`）を呼ばない。非文字列は
 * `typeof` の型名だけを出す — 入力境界の診断で利用者のコードを走らせると、`ExecutionError` の
 * 代わりに利用者側の例外が `createContainerSession` から抜ける。
 *
 * パッケージ内向けに export しているのはテスト用（GPU に触れない純関数なので、アダプタ無し
 * 環境でも回帰を撃てる）。`mod.ts` の公開面には出さない（ADR 0008）。
 */
export const assertExecutionKnobs = (
  linearCompute: NonNullable<SessionOptions["linearCompute"]>,
  attentionCompute: ComputePrecision,
  attentionScoreStorage: ScoreStorage,
  stateAttentionReduce: StateAttentionReduce,
  linearGemvReduce: LinearGemvReduce,
): void => {
  const violation = executionKnobsViolation(
    linearCompute,
    attentionCompute,
    attentionScoreStorage,
    stateAttentionReduce,
    linearGemvReduce,
  );
  if (violation !== undefined) throw new ExecutionError(violation);
};

/** {@link assertExecutionKnobs} の判定本体（違反の文言を返し、送出は呼び手が決める）。 */
const executionKnobsViolation = (
  linearCompute: NonNullable<SessionOptions["linearCompute"]>,
  attentionCompute: ComputePrecision,
  attentionScoreStorage: ScoreStorage,
  stateAttentionReduce: StateAttentionReduce,
  linearGemvReduce: LinearGemvReduce,
): string | undefined => {
  const knobs: readonly (readonly [string, string, Readonly<Record<string, true>>])[] = [
    ["linearCompute", linearCompute, LINEAR_COMPUTES],
    ["attentionCompute", attentionCompute, ATTENTION_COMPUTES],
    ["attentionScoreStorage", attentionScoreStorage, SCORE_STORAGES],
    ["stateAttentionReduce", stateAttentionReduce, STATE_ATTENTION_REDUCES],
    ["linearGemvReduce", linearGemvReduce, LINEAR_GEMV_REDUCES],
  ];
  const violations = knobs
    .filter(([, value, accepted]) => typeof value !== "string" || !Object.hasOwn(accepted, value))
    .map(([name, value, accepted]) =>
      `  - ${name}: ${
        typeof value === "string" ? JSON.stringify(value) : typeof value
      }（受理するのは ${Object.keys(accepted).map((accept) => `'${accept}'`).join(" / ")}）`
    );
  if (violations.length === 0) return undefined;
  return `SessionOptions の実行形ノブ ${violations.length} 本が受理集合の外（既定へ黙って縮退させない）:\n` +
    `${violations.join("\n")}\n` +
    "綴りを確認すること（linearCompute の 'i8a8' は 0.5.0 で 'a8' へ改名した — ADR 0074 決定 3）";
};

/**
 * 省略された実行形ノブへ runtime 既定を入れた値（{@link sessionOptionsViolation} と
 * session-build.ts の `buildSessionState` が同じ 1 本で読む — 既定の綴りを 2 か所に書くと、判定した値と
 * 構築に使う値が割れうる）。
 */
export const executionKnobDefaults = (options: SessionOptions) => ({
  rmsNormReduce: options.rmsNormReduce === undefined ? "workgroup" : options.rmsNormReduce,
  linearCompute: options.linearCompute ?? "f32",
  attentionCompute: options.attentionCompute ?? "f32",
  attentionScoreStorage: options.attentionScoreStorage ?? "f32",
  stateAttentionReduce: options.stateAttentionReduce ?? "sequential",
  linearGemvReduce: options.linearGemvReduce ?? "sequential",
  planBackingBudgetBytes: options.planBackingBudgetBytes ?? DEFAULT_PLAN_BACKING_BUDGET_BYTES,
});

/**
 * {@link SessionOptions} のうち **GPU の能力に依らない**受理条件（型・綴り・組合せ・値域）を
 * 1 本で判定し、最初の違反の文言を返す（違反が無ければ `undefined`）。
 *
 * Session 構築（`createSessionFromContainer` / `PreparedModel.createContainerSession`）は
 * 入口でこの関数を呼び、違反を `ExecutionError` として送出する。構築まで待たずに同じ判定を
 * 借りたい呼び手 — 配布形の宣言と利用者の明示指定を合成して、**重みを 1 バイトも取る前**に
 * 不受理を落としたい上位層 — のために公開している。送出型を返さず文言だけを返すのは、同じ
 * 違反でも打つ手が出所（配布形の宣言か、呼び手の指定か）で違い、型を選べるのは出所を知って
 * いる呼び手だけだから。
 *
 * MUST: GPU の能力（`shader-f16` / subgroups）を要る条件はここに含めない — それは device を
 * 見ないと決まらないので、構築側に残る。ここが `undefined` を返しても構築が落ちることはある。
 * MUST: 診断で利用者の変換（`toString` / `Symbol.toPrimitive`）を呼ばない（{@link assertExecutionKnobs}
 * と同じ規律）。
 * NOTE: 省略した欄は runtime 既定（`f32` / `sequential` 等）として判定する。`null` のような
 * 欄の値も `??` で既定に読まれる（構築と同じ読み方）ので、「欄はあるが値が不正」を
 * 既定へ戻さずに拒否したい呼び手は、自分の入口で値域を先に見ること。
 */
export const sessionOptionsViolation = (options: SessionOptions): string | undefined => {
  if (
    options.fuseLinearStaticQuantize !== undefined &&
    typeof options.fuseLinearStaticQuantize !== "boolean"
  ) {
    return "options.fuseLinearStaticQuantize はbooleanでなければならない";
  }
  if (options.fuseRmsNormAdd !== undefined && typeof options.fuseRmsNormAdd !== "boolean") {
    return "options.fuseRmsNormAdd はbooleanでなければならない";
  }
  if (
    options.packedStaticQuantize !== undefined &&
    typeof options.packedStaticQuantize !== "boolean"
  ) {
    return "options.packedStaticQuantize はbooleanでなければならない";
  }
  const {
    rmsNormReduce,
    linearCompute,
    attentionCompute,
    attentionScoreStorage,
    stateAttentionReduce,
    linearGemvReduce,
    planBackingBudgetBytes,
  } = executionKnobDefaults(options);
  if (rmsNormReduce !== "workgroup" && rmsNormReduce !== "subgroup32") {
    // 診断で利用者の変換を呼ばない（`String(x)` は `Symbol.toPrimitive` / `toString` を走らせ、
    // 例外を投げるオブジェクトでは `ExecutionError` の代わりにそれが抜ける）。
    return `options.rmsNormReduce: 未対応の値 ${
      typeof rmsNormReduce === "string" ? JSON.stringify(rmsNormReduce) : typeof rmsNormReduce
    }`;
  }
  // MUST: 綴りの検査は既定代入の直後・以降の全ゲートより前。ここを通った後は s16×c16 ゲートも
  // f16 feature ゲートも union 内の値だけを見ればよい。
  const knobs = executionKnobsViolation(
    linearCompute,
    attentionCompute,
    attentionScoreStorage,
    stateAttentionReduce,
    linearGemvReduce,
  );
  if (knobs !== undefined) return knobs;
  if (
    options.fuseLinearStaticQuantize === true &&
    (linearGemvReduce !== "parallel" || linearCompute !== "f32")
  ) {
    return "fuseLinearStaticQuantize は linearGemvReduce: parallel / linearCompute: f32 のみ対応";
  }
  // packed 活性の変種を持つのは並列 GEMV 族だけ（ADR 0105）。黙って f32 経路へ落とすと
  // 「指定したのに効かない」席になるので、融合と同じ流儀で拒否する。
  if (
    options.packedStaticQuantize === true &&
    (linearGemvReduce !== "parallel" || linearCompute !== "f32")
  ) {
    return "packedStaticQuantize は linearGemvReduce: parallel / linearCompute: f32 のみ対応";
  }
  if (linearGemvReduce !== "sequential" && linearCompute !== "f32") {
    return `linearGemvReduce: ${linearGemvReduce} は linearCompute: f32 のみ対応`;
  }
  // 値域の検査（union を読まない）は綴りの門の後 — 文言は estimate.ts の同じ門と揃える。
  if (!Number.isSafeInteger(planBackingBudgetBytes) || planBackingBudgetBytes < 0) {
    return `options.planBackingBudgetBytes ${
      describeInputValue(planBackingBudgetBytes)
    } は非負の安全な整数で` +
      "なければならない";
  }
  // 行ブロック gemv の並列度目標も同じ形で検査する（0 / 負 / 非整数は「スレッド数の目標」として
  // 意味を持たず、黙って受けると rows が最大のまま選ばれた走行になる）。
  const linearGemvRowsThreadTarget = options.linearGemvRowsThreadTarget;
  if (
    linearGemvRowsThreadTarget !== undefined &&
    (!Number.isSafeInteger(linearGemvRowsThreadTarget) || linearGemvRowsThreadTarget < 1)
  ) {
    return `options.linearGemvRowsThreadTarget ${
      describeInputValue(linearGemvRowsThreadTarget)
    } は 1 以上の` +
      "安全な整数でなければならない";
  }
  // 行ブロックの一時 1 本ごとの予算（ADR 0124 決定 5）。ここで落とすのは構築時に分かる不正値だけ —
  // 「1 行ぶんにも満たない予算」は形が決まる run の計画導出で予算を名指して落ちる（構築時には
  // 形が記号のまま）。文言は estimate.ts の同じ門と揃える。
  const attentionRowBlockMaxBytes = options.attentionRowBlockMaxBytes;
  if (
    attentionRowBlockMaxBytes !== undefined &&
    (!Number.isSafeInteger(attentionRowBlockMaxBytes) || attentionRowBlockMaxBytes < 1)
  ) {
    return `options.attentionRowBlockMaxBytes ${
      describeInputValue(attentionRowBlockMaxBytes)
    } は正の安全な整数でなければならない`;
  }
  // MUST: S の格納形は 1 つに決まらなければならない。`:c16` は S を array<f16> で持つ
  // **別の形**（ADR 0028）なので、s16 と併記されたら黙ってどちらかに解釈せず落とす
  // （どちらの丸め列で走ったのかが診断からも数値からも見えなくなる）。
  if (attentionScoreStorage === "f16" && attentionCompute === "f16") {
    return "attentionScoreStorage 'f16' と attentionCompute 'f16' は同時に指定できない" +
      "（attentionCompute 'f16' は S を array<f16> で持つ別の格納形 — " +
      "shader-f16 無しで S を半分にするなら attentionCompute を 'f32' か 'a8' にすること）";
  }
  return undefined;
};
