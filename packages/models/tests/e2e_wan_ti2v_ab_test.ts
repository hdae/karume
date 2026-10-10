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
 * ## I2V のケース（ADR 0121 段 9d・利用者の裁定 2026-10-09〜10）
 *
 * 条件画像からの生成（I2V）の実用席の数値の門。要求は 33 フレームのケースに条件画像と fit を足したもの（1280×704・33 フレーム・
 * guidance 5・shift 5・seed 42・`boxing-cats`・事前計算の埋め込み）。条件画像は I2V の通しの e2e（`e2e_wan_ti2v_i2v_pipeline_test.ts`）
 * の既定のケースと同じ `boxing-cats` の元画像で、CPU の参照（`pipeline_steps_i2v.band-boxing-cats.safetensors`）に埋め込まれた
 * RGB8（`source` 832×480 — PNG の decode にも別の資産にも頼らない）を同じ参照から読む。fit は明示の `"crop"`（製品の既定に
 * 黙って乗せない — 既定が動いても帯の前提が動かないように）。寸法は明示するので画像の縦横比での選択に預けない。参照が無い
 * 環境は理由と生成のコマンドを出して**明示 SKIP** する。
 *
 * env の opt-in `KARUME_WAN_TI2V_I2V_FULL=1`（I2V の通しの e2e の opt-in と同じ変数）のときだけ回す（既定のレーンに入れない —
 * 利用者の裁定）。故障注入は回さない（門の空振りは T2V の 33 フレームが示す — 手順は同じ 1 本）。census と決定性は
 * {@link runWanAbCase} のまま。加えて {@link runWanAbCase} は条件画像を渡したケースの各観測で encoder の 3 グラフの run が
 * 届いたことを見る（画像が要求から落ちて両席とも T2V で回ると、relRMS は T2V の値で帯の内に収まり census も DiT しか
 * 見ないので、帯と census では掴めない）。回すコマンド:
 *
 * ```
 * KARUME_WAN_TI2V_I2V_FULL=1 deno test -A --v8-flags=--expose-gc packages/models/tests/e2e_wan_ti2v_ab_test.ts --filter "i2v"
 * ```
 *
 * ノブ単位の記録（2.1 の `knobRungs`）は持たない — 段 7 の検収条件に無く、既定のレーンの所要を増やさない。最終出力（フレーム）は
 * ここでは作らない — sha 行（`e2e_wan_ti2v_pipeline_test.ts` / `e2e_wan_ti2v_i2v_pipeline_test.ts`）が持ち、帯にしない（ADR 0110
 * 決定 5-4）。結果の席は既定のレーンの 33 フレームが `wan-ti2v-ab`・opt-in の 121 フレームが `wan-ti2v-ab-121f`・opt-in の I2V が
 * `wan-ti2v-ab-i2v`（`results.json` は走行ごとに丸ごと書き直されるので、別のプロセスで回す opt-in のケースが同じ日の
 * 33 フレームの記録を消さないように分ける — 通しの e2e の 121 フレームと同じ扱い）。
 *
 * ## device の使い方
 *
 * device は Deno.test 1 本で 1 つ取り、観測の間に `settleReleases` で解放を待つ（2.1 と同じ）。opt-in の env を付けて `--filter`
 * なしで回すと 1 プロセスで device を順に取る（121 → 33 → I2V）。`destroy()` が VRAM をすぐ返さない機（B570 — ADR 0120 リスク 6）
 * では前のケースの確保の残りが次のケースの確保に重なりうるので、opt-in のケースは上のコマンドで単独に回す（通しの e2e の
 * opt-in と同じ規律）。並びを 121 フレームが先にしているのは、`--filter` なしで回したときでも大きい確保を device の残りを
 * 背負わずに張るため。
 *
 * MUST: 資産は `models/karume-wan2.2/`（untracked・実 GPU 機のローカル資産）。無い環境と GPU 無し環境は理由（組み立ての
 * コマンド）を出して**明示 SKIP** する（ADR 0005）。故障注入の席は同じ配布形を symlink で借りた一時 manifest で張り、元の
 * 配布形は 1 バイトも書き換えない。
 */

import { assertEquals } from "@std/assert";
import { parseSafetensors } from "@karume/runtime";
import { type Rgb8Image, WanTi2vPipeline } from "../wan.ts";
import { assertAdapterMatchesEnvironment } from "../../runtime/tests/helpers/environment.ts";
import { sha256Hex } from "../../runtime/tests/helpers/reference.ts";
import { openResults, type Results } from "../../runtime/tests/helpers/results.ts";
import type { AbBand } from "./helpers/ab-gate.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "./helpers/gpu.ts";
import { settleReleases } from "./helpers/settle-releases.ts";
import { runWanAbCase, type WanAbSubject } from "./helpers/wan-ab-gate.ts";
import { WAN_I2V_IMAGE_SHA256 } from "./helpers/wan-i2v-image.ts";
import {
  WAN_I2V_SOURCE_SHAPE,
  WAN_I2V_STEPS_GENERATE,
  wanI2vStepsFixtureUrl,
} from "./helpers/wan-i2v-fixture.ts";
import { filePresent } from "./helpers/wan-ti2v-dit.ts";
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
/** opt-in の I2V の置き場（`<日付>_wan-ti2v-ab-i2v/` — 既定のレーンと分ける理由はモジュール doc）。 */
const i2vResults = openResults("wan-ti2v-ab-i2v");

/**
 * 要求の固定値（受理集合の横長・guidance / shift は通しの e2e の `TWO_STEP` と同じ値 — 今の manifest の既定とも同じ値を明示する）。
 */
const REQUEST = { width: 1280, height: 704, guidance: 5, shift: 5 } as const;

/** 121 フレームのケースの opt-in（`e2e_wan_ti2v_pipeline_test.ts` の 121 フレームのケースと同じ変数）。 */
const LONG_CLIP = Deno.env.get("KARUME_WAN_TI2V_121F") === "1";

/** I2V のケースの opt-in（`e2e_wan_ti2v_i2v_pipeline_test.ts` の opt-in と同じ変数）。 */
const I2V_FULL = Deno.env.get("KARUME_WAN_TI2V_I2V_FULL") === "1";

/** I2V のケースの条件画像の名前（プロンプトと同じ名前 — I2V の通しの e2e の既定のケースの元画像）。 */
const I2V_IMAGE = "boxing-cats";

/**
 * I2V の条件画像を埋め込んだ CPU の参照（I2V の通しの e2e の既定のケース `band-boxing-cats` — 置き場・生成のコマンド・
 * 元画像の形はその e2e と共有の `helpers/wan-i2v-fixture.ts`）。
 */
const I2V_FIXTURE_URL = wanI2vStepsFixtureUrl(`band-${I2V_IMAGE}`);

/**
 * I2V のケースの帯（{@link CASES} と同じ規則で独立に導く — 条件のトークンが加わるので 33 フレームの T2V の帯を流用しない。
 * 実測と導出は {@link CASES} の表の `i2v-33f` の行）。
 *
 * MUST: 導出前は `undefined`（{@link CASES} の MUST と同じ）。
 */
const I2V_BAND: AbBand | undefined = { floor: 0, ceiling: 9.3e-2 };

/** I2V のケースのフレーム数（33 フレームのケースと同じ要求に条件画像と fit を足す）。 */
const I2V_FRAMES = 33;

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
 * | ケース  | S      | 実測 relRMS | 実測 maxAbs | floor | ceiling            | 採った日・機                    |
 * | ------- | ------ | ----------- | ----------- | ----- | ------------------ | ------------------------------- |
 * | 121f    | 27,280 | 4.9151e-2   | 2.1226e-1   | 0     | 9.9e-2（× 2 切上） | 2026-10-09・RTX 3080 Ti（Deno） |
 * | 33f     | 7,920  | 4.3881e-2   | 1.9291e-1   | 0     | 8.8e-2（× 2 切上） | 2026-10-09・RTX 3080 Ti（Deno） |
 * | i2v-33f | 7,920  | 4.6146e-2   | 4.9261e-1   | 0     | 9.3e-2（× 2 切上） | 2026-10-10・RTX 3080 Ti（Deno） |
 *
 * `i2v-33f` は I2V のケース（{@link I2V_BAND} — 33 フレームに条件画像 `boxing-cats` と fit `"crop"` を足した要求・opt-in）。
 * T2V の 33 フレームに比べ relRMS は 1.05 倍でほぼ同じ、maxAbs は 2.6 倍。maxAbs は潜在フレーム 0（観測点は条件で置き換える
 * 前の状態なので、出力に届かないフレーム）にある: フレーム 0 は relRMS 6.6217e-2・maxAbs 4.9261e-1、フレーム 1〜8 をまとめると
 * relRMS 4.1754e-2・フレームごとの maxAbs は 2.7906e-1 以下（2026-10-10・同じ機・一度だけの診断の計測 — テストには残していない）。
 * 帯は relRMS の宣言なので T2V と同じ桁の上限になる。
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
 * 2.2 の門の対象（共有の段 {@link runWanAbCase} へ渡す）。`caseResults` はケースの結果の席（opt-in のケースは
 * 既定のレーンと分ける — モジュール doc）、`request` は要求の固定値（I2V のケースは条件画像と fit を足す）。
 *
 * MUST: 呼ぶのは {@link RUNNABLE} のテストの中だけ（配布形が無いのにここへ来たら ignore の判定の破れなので落とす）。
 */
const subjectOf = (
  caseResults: Results,
  request: NonNullable<WanAbSubject["request"]> = REQUEST,
): WanAbSubject => {
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
    request,
  };
};

const I2V_FIXTURE_PRESENT = filePresent(I2V_FIXTURE_URL);
if (RUNNABLE && I2V_FULL && !I2V_FIXTURE_PRESENT) {
  console.warn(
    `[karume] ${I2V_FIXTURE_URL.pathname} が無いため Wan2.2 の I2V の自機 A/B 門を SKIP する` +
      `（条件画像をこの参照から読む）。生成（CPU で約 42 分・RAM の山 約 24 GiB）: ${WAN_I2V_STEPS_GENERATE}`,
  );
}

/**
 * I2V の条件画像（参照に埋め込まれた元画像の RGB8・HWC — I2V の通しの e2e の `sourceImageOf` と同じ値）。来歴（画像の名前・
 * PNG の sha256・RGB8 の sha256）を参照のメタと突き合わせてから渡す（別の画像で帯を測らないように）。
 */
const readI2vImage = async (): Promise<Rgb8Image> => {
  const bytes = await Deno.readFile(I2V_FIXTURE_URL);
  const file = parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  const where = I2V_FIXTURE_URL.pathname;
  const meta = (key: string): string | undefined => file.metadata.get(key);
  assertEquals(meta("image"), I2V_IMAGE, `${where}: 画像の名前`);
  assertEquals(meta("image_sha256"), WAN_I2V_IMAGE_SHA256[I2V_IMAGE], `${where}: 画像の sha256`);
  const source = file.tensors.get("source");
  if (source === undefined || source.dtype !== "U8") {
    throw new Error(`${where}: 'source' が U8 で無い`);
  }
  assertEquals(source.shape, WAN_I2V_SOURCE_SHAPE, `${where}: 元画像の形`);
  const data = new Uint8Array(file.buffer, source.byteOffset, source.byteLength);
  assertEquals(
    await sha256Hex(data),
    meta("source_rgb8_sha256"),
    `${where}: 元画像の RGB8 の sha256`,
  );
  return { width: WAN_I2V_SOURCE_SHAPE[1], height: WAN_I2V_SOURCE_SHAPE[0], data };
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

{
  const id = `${WAN_TI2V_PRACTICAL_QUANT}-i2v-crop-${I2V_FRAMES}f`;
  Deno.test({
    name: `e2e(実GPU): Wan2.2 ${id} の step 1 の潜在が参照席と決定性・census・帯で釣り合う` +
      "（自機 A/B・条件画像 boxing-cats・opt-in KARUME_WAN_TI2V_I2V_FULL=1）",
    ignore: !RUNNABLE || !I2V_FULL || !I2V_FIXTURE_PRESENT,
    fn: async (t) => {
      const image = await readI2vImage();
      const gpu = await acquireTestGpu();
      try {
        assertAdapterMatchesEnvironment(gpu);
        await runWanAbCase(
          t,
          gpu,
          subjectOf(i2vResults, { ...REQUEST, image, fit: "crop" }),
          { id, frames: I2V_FRAMES, band: I2V_BAND, faults: false },
        );
      } finally {
        await settleReleases(gpu);
        gpu.destroy();
      }
    },
  });
}
