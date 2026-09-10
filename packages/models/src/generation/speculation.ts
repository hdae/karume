/**
 * 投機的デコードの**純粋部分**（draft の受理・停止 token での打ち切り・verify の行組み立て）。
 * `sequence.ts` の投機経路が呼ぶ。**パイプライン非依存・GPU 非依存**なので `sampler.ts` と同じ
 * `src/generation/` に置く（DECIDED: ADR 0096 決定 3 / 5 / 7・追記〈段 3〉）。
 *
 * ## 1 cycle の形（ADR 0096 決定 5〜7）
 *
 * 会話の論理長を `P`（target が消費した行数）、frontier を `b`（位置 `P` の確定済み token — まだ KV に
 * 無い）とする。drafter は `(b, b を出した行の hidden, 位置 P)` から `d₁..d_k` を出し、verify は
 * `[b, d₁..d_k']` の `k'+1` 行を 1 run で流す。行 `j` は token `[b, d₁..][j]` を位置 `P+j` で処理する
 * ので、行 `j` の logits は**位置 `P+j+1` の予測**である。
 *
 * 受理は先頭一致: 行 `j` の抽選結果 `t_j` が `d_{j+1}` と一致する間だけ進み、止まった `j` が受理数
 * `a`、その `t_a` が新しい frontier `b'`。確定するのは `[d₁..d_a, b']` の `a+1` 個で、`a = 0` でも
 * `b'` は必ず確定する（= 非投機の decode 1 step に退化する — 投機は遅くなるだけで結果は変えない）。
 *
 * ただし**確定した token が停止 token ならその場で列挙を止める**。非投機はそこで生成を終えるので、
 * その先の行の logits を見ることも `sampler.next` を呼ぶこともない — 止めずに進むと、非投機なら
 * 触れない logits（範囲外 gather の NaN 汚染など）で落ちる cycle が出るし、RNG の消費数も余分に
 * 増える。受理した draft が停止 token だった cycle は確定列が `[d₁..d_a]` の `a` 個になる。
 *
 * 抽選は**温度に依らず**非投機と同じ列を出す。row ごとの `sampler.next` は、非投機の decode が同じ
 * 位置で行う抽選と同じ logits・同じ `history`（その行までの確定列）・同じ順で呼ばれ、確定 token
 * 1 個につき 1 回だけ RNG を消費する（棄却された行の先は抽選しない）ので、RNG の消費列まで一致する。
 * 温度 > 0 では「draft と同じ token を引いたら受理」が one-hot draft の speculative sampling そのもの
 * （受理確率 = target 分布での draft の確率・引いた token がそのまま新しい frontier）。
 */

import type { Sampler } from "./sampler.ts";

/** drafter への 1 cycle ぶんの入力（`sequence.ts` が組む — 値の意味は上のモジュール doc）。 */
export type DraftCycle = {
  /** frontier `b`（位置 `P` の確定済み token — KV 未投入）。 */
  readonly token: number;
  /** `b` を出した行（位置 `P−1`）の最終 norm 後 hidden（sequence が写した実体 — 呼び手が保持してよい）。 */
  readonly hidden: Float32Array<ArrayBuffer>;
  /** `b` の論理位置 `P`（= drafter の query の位置）。 */
  readonly position: number;
};

/**
 * 生成ループが drafter に要求する面（narrow interface — DI で fake を差せる）。
 *
 * 実体は gemma4 の借り手 context（`src/gemma/speculative.ts`）だが、この面は Session も
 * GenerationContext も知らない — 受理 0 を強制する fake・停止 token を draft の途中に置く fake が
 * GPU 無しで書ける（テスト専用の注入口を公開面に出さないため）。
 */
export type DraftFace = {
  /** 1 run で出す draft の本数（配布形に焼かれた段数 — gemma4 は 3）。 */
  readonly steps: number;
  /** `steps` 本の draft token（先頭から `d₁, d₂, …`）。呼び手は先頭 `k'` 本だけを使う。 */
  draft(cycle: DraftCycle): Promise<Int32Array<ArrayBuffer>>;
  /** 借り手を畳む（貸し手 context より**先**に呼ばれる — 順序は sequence が持つ）。 */
  dispose(): Promise<void>;
};

/**
 * この cycle で使う draft の本数 `k' = min(k, remaining − 1)`。
 *
 * 確定するのは最大 `k'+1` 個なので、残り予算 `remaining`（≥ 1）を超えない。`remaining = 1` なら
 * `k' = 0` = draft を採らず verify 1 行（= decode）で最後の 1 個を確定する。容量の項は要らない
 * （`assertBudget` が `pastLength + prompt + maxNewTokens − 1 ≤ capacity` を run の前に保証済み）。
 */
export const planDraftLength = (k: number, remaining: number): number => Math.min(k, remaining - 1);

/**
 * verify run の `last_row`（読む行の添字列 — 要素数がその run の R を束縛する）。
 *
 * `k' = 0` は `[0]`（decode 形そのもの — PreparedPlan の鍵まで同一）。`k' ≥ 1` は `[0..k']` を
 * **末尾添字で `k+1` 本に pad** する — R は PreparedPlan の鍵に入るので、予算末尾で `k'` が縮む
 * cycle ごとに別形（M=4・R=3 / R=2）を作らないため。pad 行の logits は読まない。
 */
export const verifyRowIndices = (drafted: number, k: number): number[] => {
  if (drafted === 0) return [0];
  const rows = Array.from({ length: drafted + 1 }, (_unused, index) => index);
  while (rows.length < k + 1) rows.push(drafted);
  return rows;
};

/**
 * drafter の返り値から先頭 `drafted` 本を取り、語彙の値域を見る。
 *
 * draft は verify の `input_ids` と `history` に入る（prompt と同じ経路）ので、prompt と同じ門を
 * 通す — 語彙外は embedding の範囲外 gather = 行ごと NaN 汚染になり、落ちる位置が真因から遠い。
 */
export const takeDrafts = (
  raw: Int32Array<ArrayBuffer>,
  drafted: number,
  vocabSize: number,
): number[] => {
  if (raw.length < drafted) {
    throw new Error(`draft が ${raw.length} 本しか無い（${drafted} 本要る）`);
  }
  const drafts: number[] = [];
  for (let index = 0; index < drafted; index += 1) {
    const id = raw[index];
    if (!Number.isSafeInteger(id) || id < 0 || id >= vocabSize) {
      throw new Error(`draft[${index}] ${id} が語彙 0..${vocabSize - 1} の外`);
    }
    drafts.push(id);
  }
  return drafts;
};

/**
 * 受理の結果（`confirmed` = この cycle が配送する確定列そのもの）。
 *
 * 長さは `accepted + 1`（`[d₁..d_a, b']`）。ただし**最後の受理 draft が停止 token**なら列挙はそこで
 * 止まるので `b'` が無く、長さは `accepted`（`[d₁..d_a]`）になる。停止 token 自体は列に残る
 * （非投機の停止規則と同じ — `pendingToken` として会話に残り、次ターンの prefill 先頭へ連結される）。
 */
export type DraftAcceptance = {
  readonly accepted: number;
  readonly confirmed: readonly number[];
};

/**
 * 先頭一致で受理する（**早期打ち切りの列挙**）。
 *
 * `row(j)` は verify 出力の行 `j` の logits（`j = 0..drafts.length` の `drafts.length + 1` 行が
 * 要る — 全受理のとき行 `drafts.length` が `b'` を出す）。`history` はこのターンの確定列（frontier
 * `b` を含む）で、行 `j` の抽選には `history + [d₁..d_j]` を使う。棄却された行の draft は
 * `history` に積まない（呼び手の `history` は触らない — 確定列は呼び手が配送しながら伸ばす）。
 *
 * MUST: 確定した token が `isStop` なら**そこで止める**（モジュール doc の停止の節）。受理した
 * draft が停止 token だったときは次の行の `sampler.next` を呼ばない = 非投機が止まる位置より先の
 * logits に触れない。棄却時の `b'` が停止 token なら列はそのまま（もともとそこで返る）。
 */
export const acceptDrafts = (
  row: (index: number) => Float32Array<ArrayBuffer>,
  drafts: readonly number[],
  sampler: Sampler,
  history: readonly number[],
  isStop: (token: number) => boolean,
): DraftAcceptance => {
  // plain step は履歴を伸ばさない。通常 decode と同じ履歴をそのまま抽選へ渡す。
  if (drafts.length === 0) return { accepted: 0, confirmed: [sampler.next(row(0), history)] };
  const extended = [...history];
  let accepted = 0;
  for (;;) {
    const token = sampler.next(row(accepted), extended);
    if (accepted < drafts.length && token === drafts[accepted]) {
      extended.push(token);
      accepted += 1;
      if (isStop(token)) return { accepted, confirmed: drafts.slice(0, accepted) };
      continue;
    }
    return { accepted, confirmed: [...drafts.slice(0, accepted), token] };
  }
};

/** 投機の勘定（`GenerationStop.speculation` の `used: true` 側）— cycle ごとに `record` で積む。 */
export type SpeculationTally = {
  cycles: number;
  draftRuns: number;
  drafted: number;
  accepted: number;
  /** cycle が確定させた token の総数（Σ `confirmed.length`）。 */
  delivered: number;
  /** 添字 = その cycle の受理数 `a`（長さ `k+1`）。 */
  readonly acceptedHistogram: number[];
};

export const createSpeculationTally = (k: number): SpeculationTally => ({
  cycles: 0,
  draftRuns: 0,
  drafted: 0,
  accepted: 0,
  delivered: 0,
  acceptedHistogram: Array.from({ length: k + 1 }, () => 0),
});
