/** Gemmaのquant実行設定を利用者の明示指定で上書きする（ADR 0104）。 */
import type { SessionSpec } from "@karume/hub";
import type { SessionOptions } from "@karume/runtime";

import { ModelInputError } from "../errors.ts";

type GemmaSessionOptions = Pick<
  SessionOptions,
  "linearGemvReduce" | "fuseRmsNormAdd" | "fuseLinearStaticQuantize" | "packedStaticQuantize"
>;

/**
 * 4欄の値域・型・組合せを**1本**で見る（受理条件の正本 — 条件は出所で分けない）。
 *
 * 送出型は決めず、違反の文言だけを返す。同じ条件でも打つ手が出所で違うためで、manifest宣言
 * だけで成立する違反は資産の齟齬（呼び手が入力を直しても直らない）、呼び手の明示指定が関与
 * する違反は入力起因である（ADR 0107 決定2）。型を選ぶのは出所を知っている呼び出し側。
 */
const sessionOptionViolation = (
  options: GemmaSessionOptions,
  where: string,
): string | undefined => {
  const { linearGemvReduce, fuseRmsNormAdd, fuseLinearStaticQuantize, packedStaticQuantize } =
    options;
  if (
    linearGemvReduce !== undefined && linearGemvReduce !== "sequential" &&
    linearGemvReduce !== "parallel" && linearGemvReduce !== "parallel-subgroup32"
  ) return `${where}: linearGemvReduceが不正`;
  if (fuseRmsNormAdd !== undefined && typeof fuseRmsNormAdd !== "boolean") {
    return `${where}: fuseRmsNormAddはbooleanでなければならない`;
  }
  if (fuseLinearStaticQuantize !== undefined && typeof fuseLinearStaticQuantize !== "boolean") {
    return `${where}: fuseLinearStaticQuantizeはbooleanでなければならない`;
  }
  if (packedStaticQuantize !== undefined && typeof packedStaticQuantize !== "boolean") {
    return `${where}: packedStaticQuantizeはbooleanでなければならない`;
  }
  if (fuseLinearStaticQuantize === true && linearGemvReduce !== "parallel") {
    return `${where}: fuseLinearStaticQuantizeはlinearGemvReduce: parallelが必要`;
  }
  if (packedStaticQuantize === true && linearGemvReduce !== "parallel") {
    return `${where}: packedStaticQuantizeはlinearGemvReduce: parallelが必要`;
  }
  return undefined;
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
): void => {
  const violation = sessionOptionViolation(overrides, where);
  if (violation !== undefined) throw new ModelInputError(violation);
};

// DECIDED: docs/decisions/0104-gemma-fast-quant.md
export const resolveGemmaSessionOptions = (
  quant: SessionSpec,
  overrides: GemmaSessionOptions,
  where: string,
): GemmaSessionOptions => {
  for (const key of Object.keys(quant)) {
    if (
      key !== "linearGemvReduce" && key !== "fuseRmsNormAdd" &&
      key !== "fuseLinearStaticQuantize" && key !== "packedStaticQuantize"
    ) throw new Error(`${where}: session.${key}は未対応`);
  }
  // 判定した値とマージに使う値がgetterですり替わらないよう、明示指定の欄は1度だけ読む。
  const explicit: GemmaSessionOptions = {
    linearGemvReduce: overrides.linearGemvReduce,
    fuseRmsNormAdd: overrides.fuseRmsNormAdd,
    fuseLinearStaticQuantize: overrides.fuseLinearStaticQuantize,
    packedStaticQuantize: overrides.packedStaticQuantize,
  };
  // ??はnullをquant定義へ戻すので使わない。不正な明示値は重み取得前に拒否する。
  const linearGemvReduce = explicit.linearGemvReduce === undefined
    ? quant.linearGemvReduce
    : explicit.linearGemvReduce;
  const fuseRmsNormAdd = explicit.fuseRmsNormAdd === undefined
    ? quant.fuseRmsNormAdd
    : explicit.fuseRmsNormAdd;
  const fuseLinearStaticQuantize = explicit.fuseLinearStaticQuantize === undefined
    ? quant.fuseLinearStaticQuantize
    : explicit.fuseLinearStaticQuantize;
  // packedStaticQuantize（ADR 0105 追記 2〈語彙への昇格〉）は他の3欄と同じ順序で入る —
  // manifest 所有の語彙（hub の SessionSpec）に席があり、QAT の i4-fast が宣言する。
  const packedStaticQuantize = explicit.packedStaticQuantize === undefined
    ? quant.packedStaticQuantize
    : explicit.packedStaticQuantize;
  // 受理集合は実効設定 1 本で決める（上書きが manifest の不備を埋める形も通す — 従来どおり）。
  const violation = sessionOptionViolation(
    { linearGemvReduce, fuseRmsNormAdd, fuseLinearStaticQuantize, packedStaticQuantize },
    where,
  );
  if (violation !== undefined) {
    // 送出型だけを出所で分ける（ADR 0107 決定2）。同じ違反が manifest 宣言**だけ**でも成立する
    // なら打つ手は配布の修正 = 資産の齟齬（500相当）で、上書きが関与して初めて成立するなら
    // 打つ手は指定の修正 = 入力起因（400相当）。判定は違反の同一性で見る。
    const declared = sessionOptionViolation(quant, where);
    throw violation === declared ? new Error(violation) : new ModelInputError(violation);
  }
  return {
    ...(linearGemvReduce === undefined ? {} : { linearGemvReduce }),
    ...(fuseRmsNormAdd === undefined ? {} : { fuseRmsNormAdd }),
    ...(fuseLinearStaticQuantize === undefined ? {} : { fuseLinearStaticQuantize }),
    ...(packedStaticQuantize === undefined ? {} : { packedStaticQuantize }),
  };
};
