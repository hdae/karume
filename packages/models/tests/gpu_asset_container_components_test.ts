/**
 * 全量面（`from*Assets`）の受け口（`src/hub/components.ts` の `assetComponentOpener`）の門の
 * うち、実 GPU を取る 2 点（受理側の門は asset_container_components_test.ts）:
 *
 * ① **分割形の Record（`<役割>[i]` のキー列）から Session が張れる**（X2-101 —
 *    「`fromPretrained` で読める配布形は `fromAssets` でも読める」）。バイト列は連結せず part 列の
 *    まま開くので、重みの block を持つ part が GPU まで届いたことを run の出力で見る。
 * ② **同じ供給口から Session を 2 本続けて張れる**（block は呼ぶたびに読み直す — 1 度読んだら
 *    終わりの列にしていないか）。
 *
 * 重みの part が欠けた列を受け口で落とす門は GPU を取らないので asset_container_components_test.ts
 * に置く。
 */

import { assertEquals } from "@std/assert";
import { acquireGpu } from "@karume/runtime";
import { openerOf, parts, single } from "./helpers/asset-container-fixture.ts";
import { LINEAR_PROBE_Y, runLinearProbe } from "./helpers/container-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

Deno.test({
  name: "assetComponentOpener: 分割形の Record から Session が張れる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const open = await openerOf(parts("dit"));
    const component = open("dit");
    // グラフ宣言は part 0（descriptor）から読めている。
    assertEquals(component.graph.outputs, ["y"]);

    const gpu = await acquireGpu();
    try {
      // 重みの block は末尾の part にしか無いので、既知の出力が出ること自体が「part 列が
      // 読まれて GPU まで届いた」証拠。2 本続けて張るのは②の門 — 2 本目が空の block 列を
      // 受けて 0 埋めで張れた形もここで出力が崩れる。
      for (let index = 0; index < 2; index += 1) {
        // MUST: `open` を呼び直さない — 呼び直すと供給口ごと作り直されるので、狙っている
        // 「1 つの部品の `createSession` が毎回 block を読み直す」を 1 度も踏まない。
        const session = await component.createSession(gpu);
        assertEquals(await runLinearProbe(session), LINEAR_PROBE_Y, `Session ${index + 1} 本目`);
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "assetComponentOpener: 単一形の 1 本からも Session が張れる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const open = await openerOf({ dit: single() });
    assertEquals(open("dit").graph.outputs, ["y"]);

    const gpu = await acquireGpu();
    try {
      const session = await open("dit").createSession(gpu);
      assertEquals(await runLinearProbe(session), LINEAR_PROBE_Y);
      await session.dispose();
    } finally {
      gpu.destroy();
    }
  },
});
