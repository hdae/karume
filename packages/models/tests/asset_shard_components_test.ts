/**
 * 全量面（`from*Assets`）の受け口（`src/hub/components.ts` の `assetComponentOpener`）の門 —
 * 資産非依存で 4 点を固定する:
 *
 * ① **shard 分割形の Record（`<役割>[i]` のキー列）から Session が張れる**（X2-101 —
 *    「`fromPretrained` で読める配布形は `fromAssets` でも読める」）。バイト列は連結せず
 *    shard 逐次面へ流すので、`[1]` にしか無い重みが GPU まで届くこと自体が経路の証拠になる。
 * ② **同じ供給口から Session を 2 本続けて張れる**（列を呼ぶたびに作り直しているか — 使い切った
 *    列を使い回すと 2 本目が空の列を受ける）。
 * ③ **添字の欠番は fail loudly**（`[0]` と `[2]` だけの Record を黙って 1 本で読まない）。
 *    `[0]` が**無い**形（`dit[1]` だけ / 素キー + `dit[1]`）も同じ門で落ちる — 添字つきキーが
 *    1 本でもあるのに `[0]` が無い並びは、以前は混在検査も欠番検査も飛ばして素の 1 本になった。
 * ④ **素キーと `[i]` の混在は fail loudly**（どちらを正とするかは決められない）。
 *
 * ③④ の診断は `shard` の語と**揃っているキーの列挙**を含むこと（既存の資産診断の流儀）まで
 * 見る — 落ちること自体は取得キーの作り方が壊れている印で、読み手が現物を突き合わせられる
 * 形でないと意味が無い。
 *
 * NOTE: 素の 1 本の面（従来の全量面）は 7 家族の `fromAssets` テストが既に縛っているので、
 * ここでは「分割形と同居しても従来どおり」の 1 点だけ確認する。
 */

import { assertEquals, assertRejects, assertStringIncludes, assertThrows } from "@std/assert";
import { acquireGpu } from "@karume/runtime";
import { assetComponentOpener } from "../src/hub/components.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { type DumpTensor, writeSafetensors } from "./helpers/safetensors-write.ts";

const f32Tensor = (shape: readonly number[], value: number): DumpTensor => ({
  dtype: "F32",
  shape: [...shape],
  data: new Float32Array(shape.reduce((product, dim) => product * dim, 1)).fill(value),
});

/** 最小の IR グラフ 1 本（`linear` 1 段 — `shard_loading_test.ts` と同型）。 */
const miniGraph = (): unknown => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["linear"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [2, 2] }],
  outputs: ["y"],
  initializers: {
    w: { tensor: "m.w", storage: { dtype: "f32" } },
    b: { tensor: "m.b", storage: { dtype: "f32" } },
  },
  values: {
    w: { dtype: "f32", shape: [2, 2] },
    b: { dtype: "f32", shape: [2] },
    y: { dtype: "f32", shape: [2, 2] },
  },
  states: {},
  nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
});

/** グラフ shard（`karume_ir` + 同居テンソル）。重み `m.w` は**入れない**（②の証拠）。 */
const graphShard = (): Uint8Array<ArrayBuffer> =>
  writeSafetensors(new Map([["m.b", f32Tensor([2], 0.25)]]), {
    karume_ir: JSON.stringify(miniGraph()),
  });

/** 重み shard（`karume_ir` を持たない）。 */
const weightShard = (): Uint8Array<ArrayBuffer> =>
  writeSafetensors(new Map([["m.w", f32Tensor([2, 2], 0.5)]]), {});

/** 全テンソル同居の 1 本（素の配布形）。 */
const wholeShard = (): Uint8Array<ArrayBuffer> =>
  writeSafetensors(
    new Map([["m.b", f32Tensor([2], 0.25)], ["m.w", f32Tensor([2, 2], 0.5)]]),
    { karume_ir: JSON.stringify(miniGraph()) },
  );

/** 家族側の資産アクセサ（`assetBuffer`）と同じ姿の最小実装。 */
const bufferOf =
  (assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>) => (key: string): ArrayBuffer => {
    if (!Object.hasOwn(assets, key)) {
      throw new Error(
        `test: 資産 '${key}' が無い（揃っているキー: ${Object.keys(assets).join(" / ")}）`,
      );
    }
    return assets[key].buffer;
  };

const openerOf = (assets: Record<string, Uint8Array<ArrayBuffer>>) =>
  assetComponentOpener("test", assets, bufferOf(assets));

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

Deno.test("assetComponentOpener: 添字の欠番は shard を名乗って落ちる", () => {
  const open = openerOf({ "dit[0]": graphShard(), "dit[2]": weightShard() });
  const error = assertThrows(() => open("dit"), Error);
  assertStringIncludes(error.message, "shard 添字が [0] から連続していない");
  // 揃っているキーを列挙する（読み手が現物と突き合わせられる形 — 既存の資産診断の流儀）。
  assertStringIncludes(error.message, "dit[0] / dit[2]");
});

Deno.test("assetComponentOpener: 素キーと分割キーの混在は shard を名乗って落ちる", () => {
  const open = openerOf({ dit: wholeShard(), "dit[0]": graphShard(), "dit[1]": weightShard() });
  const error = assertThrows(() => open("dit"), Error);
  assertStringIncludes(error.message, "素のキーと shard 分割キー");
  assertStringIncludes(error.message, "dit / dit[0] / dit[1]");
});

Deno.test("assetComponentOpener: 素キーと [0] 以外の分割キーの混在も落ちる（[0] 欠落で門を素通りしない）", () => {
  // `assetShardKeys` は `[0]` から連続する範囲だけを拾うので、`[0]` が無いと混在検査を飛ばして
  // 「素の 1 本」として組んでしまう形があった（`dit[1]` を黙って無視する = 分割配布のつもりで
  // 組んだ Record が遠くの層から「重みが足りない」で落ちる）。
  const open = openerOf({ dit: wholeShard(), "dit[1]": weightShard() });
  const error = assertThrows(() => open("dit"), Error);
  assertStringIncludes(error.message, "素のキーと shard 分割キー");
  assertStringIncludes(error.message, "dit / dit[1]");
});

Deno.test("assetComponentOpener: 添字が [0] から始まらない列は始点を名指しして落ちる", () => {
  const open = openerOf({ "dit[1]": graphShard(), "dit[2]": weightShard() });
  const error = assertThrows(() => open("dit"), Error);
  assertStringIncludes(error.message, "shard 添字が [0] から始まっていない");
  assertStringIncludes(error.message, "dit[1] / dit[2]");
});

Deno.test("assetComponentOpener: [0] から連続する列は従来どおり通る（上 2 件が恒真でないこと）", () => {
  const open = openerOf({ "dit[0]": graphShard(), "dit[1]": weightShard() });
  assertEquals(open("dit").graph.outputs, ["y"]);
});

Deno.test("assetComponentOpener: キーがどちらの形でも無ければ家族の診断のまま落ちる", () => {
  const open = openerOf({ "dit[0]": graphShard() });
  const error = assertThrows(() => open("vae"), Error);
  assertStringIncludes(error.message, "資産 'vae' が無い");
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
