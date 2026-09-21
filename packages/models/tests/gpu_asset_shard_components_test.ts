/**
 * 全量面（`from*Assets`）の受け口（`src/hub/components.ts` の `assetComponentOpener`）の門の
 * うち、実 GPU を取る 2 点（受理側の門は asset_shard_components_test.ts）:
 *
 * ① **shard 分割形の Record（`<役割>[i]` のキー列）から Session が張れる**（X2-101 —
 *    「`fromPretrained` で読める配布形は `fromAssets` でも読める」）。バイト列は連結せず
 *    shard 逐次面へ流すので、`[1]` にしか無い重みが GPU まで届くこと自体が経路の証拠になる。
 * ② **同じ供給口から Session を 2 本続けて張れる**（列を呼ぶたびに作り直しているか — 使い切った
 *    列を使い回すと 2 本目が空の列を受ける）。
 *
 * NOTE: 素の 1 本の面（従来の全量面）は 7 家族の `fromAssets` テストが既に縛っているので、
 * ここでは「分割形と同居しても従来どおり」の 1 点だけ確認する。
 */

import { assertEquals, assertRejects } from "@std/assert";
import { acquireGpu } from "@karume/runtime";
import { graphShard, openerOf, weightShard, wholeShard } from "./helpers/asset-shard-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

Deno.test({
  name: "assetComponentOpener: shard 分割形の Record から Session が張れる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const open = openerOf({ "dit[0]": graphShard(), "dit[1]": weightShard() });
    const component = open("dit");
    // グラフ宣言はグラフ shard（`[0]`）から読めている。
    assertEquals(component.graph.outputs, ["y"]);

    const gpu = await acquireGpu();
    try {
      // `m.w` は `[1]` にしか無いので、Session が張れること自体が「shard 列が流れた」証拠。
      // 2 本続けて張るのは、列を呼ぶたびに作り直しているかの門（②）。
      for (let index = 0; index < 2; index += 1) {
        // MUST: `open` を呼び直さない — 呼び直すと供給口ごと作り直されるので、狙っている
        // 「1 つのコンポーネントの `createSession` が毎回新しい列を流す」を 1 度も踏まない。
        const session = await component.createSession(gpu);
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "assetComponentOpener: 素の 1 本は従来どおり全量面で組む（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const open = openerOf({ dit: wholeShard(), "vae[0]": wholeShard() });
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
  name: "assetComponentOpener: 分割形でも重みが足りなければ Session 構築で落ちる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // `[1]`（`m.w` を持つ shard）を落とした列 — 連続はしているので受け口は通り、
    // 落ちるのは runtime の重み検査（黙って部分 Session を返さない）。
    const open = openerOf({ "dit[0]": graphShard() });
    const gpu = await acquireGpu();
    try {
      await assertRejects(() => open("dit").createSession(gpu), Error);
    } finally {
      gpu.destroy();
    }
  },
});
