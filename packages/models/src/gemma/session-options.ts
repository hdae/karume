/** Gemmaのquant実行設定を利用者の明示指定で上書きする（ADR 0104）。 */
import type { SessionSpec } from "@karume/hub";
import type { SessionOptions } from "@karume/runtime";

import { ModelInputError } from "../errors.ts";

type GemmaSessionOptions = Pick<
  SessionOptions,
  "linearGemvReduce" | "fuseRmsNormAdd" | "fuseLinearStaticQuantize" | "packedStaticQuantize"
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
      key !== "fuseLinearStaticQuantize" && key !== "packedStaticQuantize"
    ) throw new ModelInputError(`${where}: session.${key}は未対応`);
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
  // packedStaticQuantize（ADR 0105 追記 2〈語彙への昇格〉）は他の3欄と同じ順序で入る —
  // manifest 所有の語彙（hub の SessionSpec）に席があり、QAT の i4-fast が宣言する。
  const packedStaticQuantize = overrides.packedStaticQuantize === undefined
    ? quant.packedStaticQuantize
    : overrides.packedStaticQuantize;
  if (
    linearGemvReduce !== undefined && linearGemvReduce !== "sequential" &&
    linearGemvReduce !== "parallel" && linearGemvReduce !== "parallel-subgroup32"
  ) throw new ModelInputError(`${where}: linearGemvReduceが不正`);
  if (fuseRmsNormAdd !== undefined && typeof fuseRmsNormAdd !== "boolean") {
    throw new ModelInputError(`${where}: fuseRmsNormAddはbooleanでなければならない`);
  }
  if (fuseLinearStaticQuantize !== undefined && typeof fuseLinearStaticQuantize !== "boolean") {
    throw new ModelInputError(`${where}: fuseLinearStaticQuantizeはbooleanでなければならない`);
  }
  if (packedStaticQuantize !== undefined && typeof packedStaticQuantize !== "boolean") {
    throw new ModelInputError(`${where}: packedStaticQuantizeはbooleanでなければならない`);
  }
  if (fuseLinearStaticQuantize === true && linearGemvReduce !== "parallel") {
    throw new ModelInputError(
      `${where}: fuseLinearStaticQuantizeはlinearGemvReduce: parallelが必要`,
    );
  }
  if (packedStaticQuantize === true && linearGemvReduce !== "parallel") {
    throw new ModelInputError(
      `${where}: packedStaticQuantizeはlinearGemvReduce: parallelが必要`,
    );
  }
  return {
    ...(linearGemvReduce === undefined ? {} : { linearGemvReduce }),
    ...(fuseRmsNormAdd === undefined ? {} : { fuseRmsNormAdd }),
    ...(fuseLinearStaticQuantize === undefined ? {} : { fuseLinearStaticQuantize }),
    ...(packedStaticQuantize === undefined ? {} : { packedStaticQuantize }),
  };
};
