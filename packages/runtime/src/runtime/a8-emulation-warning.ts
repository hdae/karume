/**
 * int8 活性（`linearCompute` / `attentionCompute: "a8"`）を dp4a の無い device で走らせるときの
 * 警告（DECIDED: ADR 0128 決定 5 — dp4a 非対応はフォールバックでも自動の兄弟プリセットでもなく、警告）。
 *
 * i8a8 の整数内積は dp4a（WGSL 言語機能 `packed_4x8_integer_dot_product`）があればその組み込みで、
 * 無ければエミュレーションで走る。**両変種の数値は同一**（src/kernels/linear-i8a8.ts）なので
 * 正しさの問題ではなく、速度だけの問題 — エミュレーションは f32 の経路より速くならず、Apple の
 * Metal では遅くなる実測がある（ADR 0115 / ADR 0128 Context）。a8 を既定に持つ配布形
 * （anima / irodori / wan）は dp4a のある GPU で測って選ばれたもので、dp4a の無い device の利用者は
 * 速度の利得が無いまま活性量子化の品質差だけを払う。黙って通すとその事実が利用者から見えない
 * （診断のパイプラインキー `:dp4aEmu` には出るが、既定で使う人は診断を読まない）。
 *
 * 判定は純関数にして GPU 無しで検証できる形に置く。device ごとに 1 回に絞るのは呼び手
 * （session-build.ts）が {@link GpuContext} の `warnOnce` で担う — a8 の Session を部品ごとに
 * 並べる家族（anima の text encoder / DiT）で同じ 1 事実が Session の本数だけ流れるのを避ける。
 */

import { DP4A_WGSL_FEATURE, dp4aAvailable } from "../kernels/linear-i8a8.ts";
import type { ComputePrecision } from "./session-types.ts";

/** `GpuContext.warnOnce` のキー（device ごとに 1 回の単位 = 「この device に dp4a が無い」という事実）。 */
export const A8_EMULATION_WARNING_KEY = "a8-without-dp4a";

/**
 * a8 を要求されたが device に dp4a が無いときの警告文。それ以外（a8 を使わない・dp4a がある）は undefined。
 *
 * 文言は attention のカナリアの警告（session-build.ts）と同じ流儀 — 何が起きたか・何をすれば消えるか。
 * 逃げ先の「同じ重みで活性量子化を持たないプリセット」が必ずあることは配布形の機械検査
 * （`distribution_gate_test.ts` — ADR 0128 決定 2）が保証する。
 */
export const a8EmulationWarning = (
  linearCompute: ComputePrecision,
  attentionCompute: ComputePrecision,
  wgslLanguageFeatures: ReadonlySet<string>,
): string | undefined => {
  const requested = [
    ...(linearCompute === "a8" ? ["linearCompute"] : []),
    ...(attentionCompute === "a8" ? ["attentionCompute"] : []),
  ];
  if (requested.length === 0 || dp4aAvailable(wgslLanguageFeatures)) return undefined;
  return (
    `@karume/runtime: この device は dp4a（WGSL 言語機能 ${DP4A_WGSL_FEATURE}）を持たないため、\n` +
    `  int8 活性（${requested.join(" / ")}: "a8"）の整数内積はエミュレーションで走る。\n` +
    `数値は dp4a 版と同一だが、f32 の経路より速くならない（Apple の Metal では遅くなる実測がある）。\n` +
    `速度のために選んだのなら、同じ重みで活性量子化を持たないプリセット（モデルカードの Quants 表で\n` +
    `Compute 列が空のもの）を quant で明示するか、linearCompute / attentionCompute を 'f32' にすること。`
  );
};
