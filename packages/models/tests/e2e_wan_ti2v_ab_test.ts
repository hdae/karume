/**
 * Wan2.2 TI2V-5B の**自機 A/B 門**（実 GPU・ADR 0121 段 7 — 規則は ADR 0110 決定 4〜7 と ADR 0120 決定 4 の形のまま）。実用席
 * `f16+dit8-a8-attn8-s16`（既定席 — `session` に linear a8 / attention a8 / S の f16 格納を束ねた席）を、**同じ機・同じ i8 重み**
 * で `session` が空の参照席（manifest から導く — `referencePartnerOf`。今の配布形では `f16+dit8`）と比べる。比較相手は同機の
 * 参照層で、torch golden は使わない（ADR 0110 決定 5 ②）。
 *
 * 手順・判定・故障注入は 2.1（`e2e_wan_ab_test.ts`）と共有の 1 本（`helpers/wan-ab-gate.ts` の {@link runWanAbCase}）:
 * 参照席で step 1 の潜在（2 ステップの通しの 1 step 目）を採って打ち切る → 実用席で同じことを独立に 2 回 → 決定性（2 回が
 * ビット一致）・census（実用席の DiT の各パスで linear 310 本ちょうどが i8a8・参照経路の linear 0 本 — 束の census 表の
 * `wan-ti2v` の行。attention の 2 ノブは 1 本以上）・帯（`judgeAb`）。
 *
 * 入力は `boxing-cats`・seed 42・2 ステップ（step 1 = CFG の 2 パス）・1280×704・guidance 5・shift 5・事前計算の埋め込み
 * （`"precomputed"`）— 通しの e2e の seed 経路の sha 行（`e2e_wan_ti2v_pipeline_test.ts` の `TWO_STEP`）と同じ要求。寸法と
 * guidance / shift は明示する（世代の既定に黙って乗せない — manifest の既定が動いても帯の前提〈step 1 の σ と CFG のパス数〉が
 * 動かないように）。
 *
 * ## ケース（利用者の裁定 2026-10-09）
 *
 * ADR 0120 決定 4 の形（既定の長さと最長の 2 点で持つ — attention a8 の誤差は行の長さで増えるので、短い方の帯を長い方へ
 * 外挿しない）で、2.2 の 2 点は:
 *
 * - **33 フレーム**（既定のフレーム数・S = 7,920 — 既定のレーン）。故障注入 5 件もこのケースで回す（門の空振りを示すのに長さは
 *   要らない）。
 * - **121 フレーム**（受理の上限・S = 27,280 — env の opt-in `KARUME_WAN_TI2V_121F=1` のときだけ。通しの e2e の 121 フレームの
 *   ケースと同じ変数）。既定のレーンに入れないのはフル verify の所要を延ばさないため（対策は別に考える）。
 *
 * 121 フレームを回すコマンド（`--filter` で**単独のプロセス**にする — 下の「device の使い方」）:
 *
 * ```
 * KARUME_WAN_TI2V_121F=1 deno test -A --v8-flags=--expose-gc packages/models/tests/e2e_wan_ti2v_ab_test.ts --filter "121f"
 * ```
 *
 * ノブ単位の記録（2.1 の `knobRungs`）は持たない — 段 7 の検収条件に無く、既定のレーンの所要を増やさない。最終出力（フレーム）は
 * ここでは作らない — sha 行（`e2e_wan_ti2v_pipeline_test.ts`）が持ち、帯にしない（ADR 0110 決定 5-4）。結果の席は既定のレーンの
 * 33 フレームが `wan-ti2v-ab`・opt-in の 121 フレームが `wan-ti2v-ab-121f`（`results.json` は走行ごとに丸ごと書き直されるので、
 * 別のプロセスで回す 121 フレームが同じ日の 33 フレームの記録を消さないように分ける — 通しの e2e の 121 フレームと同じ扱い）。
 *
 * ## device の使い方
 *
 * device は Deno.test 1 本で 1 つ取り、観測の間に `settleReleases` で解放を待つ（2.1 と同じ）。opt-in の env を付けて `--filter`
 * なしで回すと 1 プロセスで device を 2 つ順に取る（121 → 33）。`destroy()` が VRAM をすぐ返さない機（B570 — ADR 0120 リスク 6）
 * では 121 フレームの確保の残りが 33 フレームの確保に重なりうるので、121 フレームは上のコマンドで単独に回す（通しの e2e の
 * opt-in と同じ規律）。並びを 121 フレームが先にしているのは、`--filter` なしで回したときでも大きい確保を device の残りを
 * 背負わずに張るため。
 *
 * MUST: 資産は `models/karume-wan2.2/`（untracked・実 GPU 機のローカル資産）。無い環境と GPU 無し環境は理由（組み立ての
 * コマンド）を出して**明示 SKIP** する（ADR 0005）。故障注入の席は同じ配布形を symlink で借りた一時 manifest で張り、元の
 * 配布形は 1 バイトも書き換えない。
 */

import { WanTi2vPipeline } from "../wan.ts";
import { assertAdapterMatchesEnvironment } from "../../runtime/tests/helpers/environment.ts";
import { openResults, type Results } from "../../runtime/tests/helpers/results.ts";
import type { AbBand } from "./helpers/ab-gate.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "./helpers/gpu.ts";
import { settleReleases } from "./helpers/settle-releases.ts";
import { runWanAbCase, type WanAbSubject } from "./helpers/wan-ab-gate.ts";
import {
  readWanTi2vDistributionManifestText,
  WAN_TI2V_ASSEMBLE_COMMAND,
  WAN_TI2V_DIST_MODEL,
  WAN_TI2V_DIST_ROOT,
  WAN_TI2V_PRACTICAL_QUANT,
} from "./helpers/wan-ti2v-pipeline.ts";

/** 実測と決着の置き場（`outputs/verify/<環境キー>/<日付>_wan-ti2v-ab/` — 消して安全）。 */
const results = openResults("wan-ti2v-ab");
/** opt-in の 121 フレームの置き場（`<日付>_wan-ti2v-ab-121f/` — 既定のレーンと分ける理由はモジュール doc）。 */
const longClipResults = openResults("wan-ti2v-ab-121f");

/**
 * 要求の固定値（受理集合の横長・guidance / shift は通しの e2e の `TWO_STEP` と同じ値 — 今の manifest の既定とも同じ値を明示する）。
 */
const REQUEST = { width: 1280, height: 704, guidance: 5, shift: 5 } as const;

/** 121 フレームのケースの opt-in（`e2e_wan_ti2v_pipeline_test.ts` の 121 フレームのケースと同じ変数）。 */
const LONG_CLIP = Deno.env.get("KARUME_WAN_TI2V_121F") === "1";

/**
 * ケース（フレーム数）と帯。並びは確保の大きい 121 フレームが先（モジュール doc）。
 *
 * 帯は**宣言**で、環境キー別の行にしない・`KARUME_REFERENCE` で書き換えない（ADR 0110 決定 4）。導出規則（決定 5）:
 * 上限 = 観測点の実測 relRMS × 2 程度（有効数字 2 桁へ切り上げ）をケースごとに独立に導く（33 フレームの値を 121 フレームへ
 * 外挿しない — attention a8 の誤差は行の長さ S で増える・ADR 0120 決定 4）。床 = 1 bit 以上違う（MUST — `judgeAb` が帯と無関係に
 * 常に見る）で、`floor` を 0 より上に置くのはノブ 1 つを外した実測が全部入りより明確に小さいと示せたときだけ。2.1 の帯
 * （33 フレーム 2.2e-1 / 81 フレーム 4.2e-1）は 2.2 の根拠にならない（モデル・行の長さ・潜在の形が違う）。
 *
 * MUST: 導出前は `undefined` にする — `judgeAb` が実測を出して赤で止まる（ADR 0120 決定 4）。通る値を仮置きして検出力の無い門を
 * 増やさない（`ab-gate.ts` の MUST）。帯が `undefined` の間は故障注入 ⑤（別 seed の潜在が上限で赤）も「帯が未導出」で赤になる
 * （帯の宣言の後に有効 — 2.1 の導出前と同じ振る舞い）。
 *
 * | ケース | S      | 実測 relRMS | 実測 maxAbs | floor | ceiling            | 採った日・機                    |
 * | ------ | ------ | ----------- | ----------- | ----- | ------------------ | ------------------------------- |
 * | 121f   | 27,280 | 4.9151e-2   | 2.1226e-1   | 0     | 9.9e-2（× 2 切上） | 2026-10-09・RTX 3080 Ti（Deno） |
 * | 33f    | 7,920  | 4.3881e-2   | 1.9291e-1   | 0     | 8.8e-2（× 2 切上） | 2026-10-09・RTX 3080 Ti（Deno） |
 *
 * 33 フレームの故障注入（記録）: ③（attention の a8 を外した席）は relRMS 4.0983e-2 で全部入りの 9 割強 — 2.2 の 33 フレームでは
 * attention の a8 の寄与は小さい（残りは linear の a8 と S の f16 格納 — 内訳は測っていない。2.1 は逆に attention の a8 が大半 —
 * 33 フレームで linear の a8 だけ 2.9991e-2 → 全部入り 1.0676e-1）。
 * ③ は帯の内に留まるので、census が要ることの実証になる。⑤（別 seed）は 1.3782e+0（崩壊の代理）。
 */
const CASES: readonly {
  readonly frames: number;
  readonly band: AbBand | undefined;
  /** env の opt-in（`KARUME_WAN_TI2V_121F=1`）のときだけ回すケースか。 */
  readonly optIn: boolean;
}[] = [
  { frames: 121, band: { floor: 0, ceiling: 9.9e-2 }, optIn: true },
  { frames: 33, band: { floor: 0, ceiling: 8.8e-2 }, optIn: false },
];

/** 故障注入を回すフレーム数（既定のレーンの 33 — 門の空振りを示すのに長さは要らない）。 */
const FAULT_FRAMES = 33;

const manifestText = await readWanTi2vDistributionManifestText();
if (manifestText === undefined) {
  console.warn(
    `[karume] ${WAN_TI2V_DIST_ROOT.pathname} に karume.json が無いため Wan2.2 の自機 A/B 門を SKIP する` +
      `（組み立て: ${WAN_TI2V_ASSEMBLE_COMMAND}）`,
  );
}
const RUNNABLE = GPU_AVAILABLE && manifestText !== undefined;

/**
 * 2.2 の門の対象（共有の段 {@link runWanAbCase} へ渡す）。`caseResults` はケースの結果の席（opt-in の 121 フレームは
 * 既定のレーンと分ける — モジュール doc）。
 *
 * MUST: 呼ぶのは {@link RUNNABLE} のテストの中だけ（配布形が無いのにここへ来たら ignore の判定の破れなので落とす）。
 */
const subjectOf = (caseResults: Results): WanAbSubject => {
  if (manifestText === undefined) {
    throw new Error(
      `${WAN_TI2V_DIST_ROOT.pathname} に karume.json が無いのに A/B 門が走った（ignore の判定の破れ）`,
    );
  }
  return {
    label: "wan-ti2v",
    family: "wan-ti2v",
    model: WAN_TI2V_DIST_MODEL,
    practical: WAN_TI2V_PRACTICAL_QUANT,
    assetsRoot: WAN_TI2V_DIST_ROOT,
    manifestText,
    results: caseResults,
    fromPretrained: (source, options) => WanTi2vPipeline.fromPretrained(source, options),
    request: REQUEST,
  };
};

for (const { frames, band, optIn } of CASES) {
  const id = `${WAN_TI2V_PRACTICAL_QUANT}-${frames}f`;
  const faults = frames === FAULT_FRAMES;
  Deno.test({
    name:
      `e2e(実GPU): Wan2.2 ${id} の step 1 の潜在が参照席と決定性・census・帯で釣り合う（自機 A/B）` +
      (faults ? "・故障注入 5 件は赤" : "") +
      (optIn ? "（opt-in KARUME_WAN_TI2V_121F=1）" : ""),
    ignore: !RUNNABLE || (optIn && !LONG_CLIP),
    fn: async (t) => {
      const gpu = await acquireTestGpu();
      try {
        assertAdapterMatchesEnvironment(gpu);
        await runWanAbCase(t, gpu, subjectOf(optIn ? longClipResults : results), {
          id,
          frames,
          band,
          faults,
        });
      } finally {
        await settleReleases(gpu);
        gpu.destroy();
      }
    },
  });
}
