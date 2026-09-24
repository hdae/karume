/**
 * 焼かれたグラフの**宣言**と `pipelineConfig` の宣言を突き合わせる門（family 横断で 1 本に
 * 寄せたもの）。
 *
 * 突合は逐次面でも全量面でも同じ形で要るので、受けるのは {@link GraphOwner}（グラフ宣言だけ）
 * に限る — Session も GPU も触らない。family ごとに違うのは**文言のラベル**（family 名と、
 * どの宣言を見ていたか）だけなので、それを引数で受ける。
 */

import type { GraphOwner } from "./components.ts";

/**
 * グラフ入力の 1 軸ぶんの**静的**次元が `pipelineConfig` の宣言と一致することを見る。
 *
 * MUST: 落とさない。ホスト側の前処理は宣言の寸法に合わせて作るので、グラフが別の寸法で焼かれて
 * いても**ホスト側は最後まで通る**（落ちるのは Session の shape 検査で、そのときには「どちらの
 * 数が正しいのか」が読み手に伝わらない）。資産の取り違えはここが唯一の検出器になる。
 * MUST: 形全体ではなく**要る軸だけ**を見る。データ依存の軸（記号次元・系列長）は静的でない
 * まま正当なので、全軸を静的と要求すると正しい配布形が通らない。
 */
export const assertGraphInputDim = (
  family: string,
  model: GraphOwner,
  inputName: string,
  axis: number,
  expected: number,
  where: string,
): void => {
  const spec = model.graph.inputs.find((input) => input.name === inputName);
  if (spec === undefined) {
    throw new Error(`${family}: グラフ入力 '${inputName}' が無い（${where}）`);
  }
  const dim = spec.shape[axis];
  if (dim !== expected) {
    throw new Error(
      `${family}: ${where} — グラフ入力 '${inputName}' の軸 ${axis} が ${String(dim)}、` +
        `pipelineConfig は ${expected}`,
    );
  }
};
