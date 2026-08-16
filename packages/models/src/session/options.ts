/**
 * manifest の `session`（3 キー固定の manifest 所有語彙）→ runtime `SessionOptions` の写像
 * （**パイプライン非依存の共通処理** — 7 家族の `fromAssets` が同じ形で使う）。
 *
 * MUST: barrel には出さない。これは配布形の宣言を runtime のノブへ翻訳する内部機構で、
 * 利用者が触る面ではない（`export` はパッケージ内テストが写像そのものを叩くため — 写像の
 * 抜けは GPU を回さないと露見しない位置にある）。
 *
 * NOTE: 元は 7 家族の `pipeline.ts` へバイト単位で複製されていた。複製は「綴りの改名」は
 * 型検査で捕まえられる一方、「`SessionSpec` へのキー追加」は**どの家族も型検査を通る**
 * （写像は書いていないキーを黙って落とすだけ）ため、追随を忘れた家族がそのまま沈黙劣化した。
 * 1 本化と下の網羅表で、その余地を構造的に消している。
 */

import type { SessionSpec } from "@karume/hub";
import type { SessionOptions } from "@karume/runtime";

/** 写像 1 キーぶん — `spec` の欄が埋まっているときだけ `SessionOptions` の欄を作る。 */
type SpecWriter = (spec: SessionSpec) => SessionOptions;

/**
 * manifest 所有の各キーを `SessionOptions` の欄へ写す表。
 *
 * MUST: キー集合は `Required<SessionSpec>` の**網羅** — `SessionSpec` にノブが増えたら
 * この宣言が型検査で落ちるので、写像の追随漏れ（= 配布形が宣言したノブが runtime へ届かない
 * 沈黙劣化）が起きない。
 *
 * MUST: 個々の writer はスプレッドで丸投げしない（ADR 0038 §3 — 素通しにすると綴りが変わった
 * 瞬間に runtime が未知キーを黙って無視する。写像を明示すると綴りが割れた時点で型検査が
 * 落ちる）。`SessionOptions.submitPolicy`（TDR 予算 = ホスト政策）のように manifest 側に
 * 席の無いノブが混ざらないのも、この明示写像の効果。
 */
const WRITERS: { readonly [K in keyof Required<SessionSpec>]: SpecWriter } = {
  linearCompute: (spec) =>
    spec.linearCompute === undefined ? {} : { linearCompute: spec.linearCompute },
  attentionCompute: (spec) =>
    spec.attentionCompute === undefined ? {} : { attentionCompute: spec.attentionCompute },
  attentionScoreStorage: (spec) =>
    spec.attentionScoreStorage === undefined
      ? {}
      : { attentionScoreStorage: spec.attentionScoreStorage },
};

/** 宣言された欄だけを持つ `SessionOptions` を組む（未指定のキーは欄ごと作らない）。 */
export const toSessionOptions = (spec: SessionSpec): SessionOptions => {
  let options: SessionOptions = {};
  for (const write of Object.values(WRITERS)) options = { ...options, ...write(spec) };
  return options;
};
