/**
 * 全量面（`from*Assets`）の受け口（`src/hub/components.ts` の `assetComponentOpener`）の門の
 * うち、実 GPU を取る 3 点（受理側の門は asset_container_components_test.ts）:
 *
 * ① **分割形の Record（`<役割>[i]` のキー列）から Session が張れる**（X2-101 —
 *    「`fromPretrained` で読める配布形は `fromAssets` でも読める」）。バイト列は連結せず part 列の
 *    まま開くので、重みの block を持つ part が GPU まで届くこと自体が経路の証拠になる。
 * ② **同じ供給口から Session を 2 本続けて張れる**（block は呼ぶたびに読み直す — 1 度読んだら
 *    終わりの列にしていないか）。
 * ③ **重みの part が欠けた列は受け口で落ちる**（黙って部分 Session を返さない）。
 */

import { assertEquals, assertRejects } from "@std/assert";
import { acquireGpu } from "@karume/runtime";
import { openerOf, part, parts, single } from "./helpers/asset-container-fixture.ts";
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
      // 重みの block は末尾の part にしか無いので、Session が張れること自体が「part 列が
      // 読まれた」証拠。2 本続けて張るのは②の門。
      for (let index = 0; index < 2; index += 1) {
        // MUST: `open` を呼び直さない — 呼び直すと供給口ごと作り直されるので、狙っている
        // 「1 つの部品の `createSession` が毎回 block を読み直す」を 1 度も踏まない。
        const session = await component.createSession(gpu);
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
      await session.dispose();
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "assetComponentOpener: 重みの part が欠けた列は受け口で落ちる（実 GPU 不要の対偶つき）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // 末尾の part（重みの block を持つ）を落とした列 — 宣言された part 数と合わないので、
    // 落ちるのは容器の読み手（黙って部分 Session を返さない）。
    await assertRejects(() => openerOf({ "dit[0]": part(0), "dit[1]": part(1) }), Error);
  },
});
