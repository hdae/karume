/** Gemmaのquant実行設定を利用者の明示指定で上書きする（ADR 0104）。 */
import type { SessionSpec } from "@karume/hub";
import type { SessionOptions } from "@karume/runtime";

import {
  assertSessionOverrides,
  type FamilySessionPolicy,
  resolveSessionOptions,
} from "../session/options.ts";

type GemmaSessionOptions = Pick<
  SessionOptions,
  | "linearGemvReduce"
  | "stateAttentionReduce"
  | "fuseRmsNormAdd"
  | "fuseLinearStaticQuantize"
  | "packedStaticQuantize"
>;

/**
 * Gemma が受けるキー（manifest の宣言と明示指定の両方）。優先順位・値域・組合せ・送出型の
 * 分類は全家族共通の {@link resolveSessionOptions} が持ち、ここは受理集合だけを決める。
 *
 * linear / attention の実行形（`linearCompute` / `attentionCompute` / `attentionScoreStorage`）を
 * 受けないのは、Gemma の配布形が宣言するのは並列 GEMV・states 形 attention の縮約形・融合の
 * 5 欄だけで、それ以外の宣言は検証されていない組合せだから（ADR 0104）。
 *
 * `stateAttentionReduce` も他の欄と同じ規則に乗る（ADR 0104 追記 2026-09-26）: 宣言が無い quant
 * 席（参照の `i4`）と manifest を持たない `fromAssets` は runtime の参照経路 `"sequential"` で
 * 走る。家族のコード既定は持たない — 持つと `session` が空の席まで参照経路でなくなる。
 *
 * NOTE: `export` は同値テストがミラーの全 quant を同じ表で通すため（`mod.ts` / サブパス面には
 * 出さない — ADR 0008）。
 */
export const GEMMA_SESSION_POLICY: FamilySessionPolicy = {
  linearCompute: false,
  attentionCompute: false,
  attentionScoreStorage: false,
  linearGemvReduce: true,
  stateAttentionReduce: true,
  fuseRmsNormAdd: true,
  fuseLinearStaticQuantize: true,
  packedStaticQuantize: true,
};

/**
 * 呼び手の**明示指定だけ**を相手にする門（入力起因 = `ModelInputError`）。
 *
 * MUST: `fromAssets` はquant宣言を持たない面なので、この門を**資産を1バイトも開く前**に通す
 * （通さないと同じ誤指定がSession構築まで降り、runtime の `ExecutionError` に化けて
 * `fromPretrained` と分類が割れる）。quant宣言と突き合わせる組合せは見ないので、既定 `{}` を
 * 相手にしたときと同じ判定になる。
 *
 * NOTE: `export` は2つの入口が同じ門を通すため（`mod.ts` / サブパス面には出さない — ADR 0008）。
 */
export const assertGemmaSessionOverrides = (
  overrides: GemmaSessionOptions,
  where: string,
): void => assertSessionOverrides(GEMMA_SESSION_POLICY, overrides, where);

// DECIDED: docs/decisions/0104-gemma-fast-quant.md
export const resolveGemmaSessionOptions = (
  quant: SessionSpec,
  overrides: GemmaSessionOptions,
  where: string,
): GemmaSessionOptions => resolveSessionOptions(GEMMA_SESSION_POLICY, quant, overrides, where);
