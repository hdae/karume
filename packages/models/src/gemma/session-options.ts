/** Gemmaのquant実行設定を利用者の明示指定で上書きする（ADR 0104）。 */
import type { SessionSpec } from "@karume/hub";
import type { SessionOptions } from "@karume/runtime";

type GemmaSessionOptions = Pick<
  SessionOptions,
  "linearGemvReduce" | "fuseRmsNormAdd" | "fuseLinearStaticQuantize"
>;

// DECIDED: docs/decisions/0104-gemma-fast-quant.md
export const resolveGemmaSessionOptions = (
  quant: SessionSpec,
  overrides: GemmaSessionOptions,
  where: string,
): GemmaSessionOptions => {
  for (const key of Object.keys(quant)) {
    if (
      key !== "linearGemvReduce" && key !== "fuseRmsNormAdd" &&
      key !== "fuseLinearStaticQuantize"
    ) throw new Error(`${where}: session.${key}は未対応`);
  }
  // ??はnullをquant定義へ戻すので使わない。不正な明示値は重み取得前に拒否する。
  const linearGemvReduce = overrides.linearGemvReduce === undefined
    ? quant.linearGemvReduce
    : overrides.linearGemvReduce;
  const fuseRmsNormAdd = overrides.fuseRmsNormAdd === undefined
    ? quant.fuseRmsNormAdd
    : overrides.fuseRmsNormAdd;
  const fuseLinearStaticQuantize = overrides.fuseLinearStaticQuantize === undefined
    ? quant.fuseLinearStaticQuantize
    : overrides.fuseLinearStaticQuantize;
  if (
    linearGemvReduce !== undefined && linearGemvReduce !== "sequential" &&
    linearGemvReduce !== "parallel" && linearGemvReduce !== "parallel-subgroup32"
  ) throw new Error(`${where}: linearGemvReduceが不正`);
  if (fuseRmsNormAdd !== undefined && typeof fuseRmsNormAdd !== "boolean") {
    throw new Error(`${where}: fuseRmsNormAddはbooleanでなければならない`);
  }
  if (fuseLinearStaticQuantize !== undefined && typeof fuseLinearStaticQuantize !== "boolean") {
    throw new Error(`${where}: fuseLinearStaticQuantizeはbooleanでなければならない`);
  }
  if (fuseLinearStaticQuantize === true && linearGemvReduce !== "parallel") {
    throw new Error(
      `${where}: fuseLinearStaticQuantizeはlinearGemvReduce: parallelが必要`,
    );
  }
  return {
    ...(linearGemvReduce === undefined ? {} : { linearGemvReduce }),
    ...(fuseRmsNormAdd === undefined ? {} : { fuseRmsNormAdd }),
    ...(fuseLinearStaticQuantize === undefined ? {} : { fuseLinearStaticQuantize }),
  };
};
