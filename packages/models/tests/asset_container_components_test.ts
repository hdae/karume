/**
 * 全量面（`from*Assets`）の受け口（`src/hub/components.ts` の `assetComponentOpener`）の門。
 * ①② open a Session on a real GPU (each case is explicitly skipped via `ignore` when no adapter
 * is present); ③〜⑥ are the acceptance-side gates and need no GPU.
 *
 * ① **分割形の Record（`<役割>[i]` のキー列）から Session が張れる**（X2-101 —
 *    「`fromPretrained` で読める配布形は `fromAssets` でも読める」）。バイト列は連結せず part 列の
 *    まま開くので、重みの block を持つ part が GPU まで届いたことを run の出力で見る。
 * ② **同じ供給口から Session を 2 本続けて張れる**（block は呼ぶたびに読み直す — 1 度読んだら
 *    終わりの列にしていないか）。
 * ③ **添字の欠番は fail loudly**（`[0]` と `[2]` だけの Record を黙って 1 本で読まない）。
 *    `[0]` が**無い**形（`dit[1]` だけ / 単一形キー + `dit[1]`）も同じ門で落ちる — 添字つきキーが
 *    1 本でもあるのに `[0]` が無い並びは、混在検査も欠番検査も飛ばして単一形の 1 本になりうる。
 * ④ **単一形キーと `[i]` の混在は fail loudly**（どちらを正とするかは決められない）。
 * ⑤ **部品の容器がまったく無い形は、2 形の綴りを添えて落ちる**（綴り間違いが「資産が無い」の
 *    遠い診断に化けない）。
 * ⑥ **重みの part が欠けた列は受け口で落ちる**（黙って部分 Session を返さない）。落とすのは容器の
 *    読み手（宣言された part 数との突き合わせ）で、GPU は要らない。
 *
 * ③④⑤ の診断は `part` の語と**揃っているキーの列挙**を含むこと（既存の資産診断の流儀）まで
 * 見る — 落ちること自体はキーの作り方が壊れている印で、読み手が現物を突き合わせられる形で
 * ないと意味が無い。
 *
 * NOTE: `assetComponentOpener` は**開く前に全部品を開く**（同期の供給口を返すには開いてから
 * 配るしかない）ので、これらは `open(key)` ではなく**供給口を作る時点**で落ちる。
 */

import { test } from "@karume/test-support/test";
import { assertEquals, assertRejects, assertStringIncludes } from "@std/assert";
import { acquireGpu, ContainerFormatError } from "@karume/runtime";
import { openerOf, part, parts, single } from "./helpers/asset-container-fixture.ts";
import { LINEAR_PROBE_Y, runLinearProbe } from "./helpers/container-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

test({
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

test({
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

test("assetComponentOpener: 添字の欠番は part を名乗って落ちる", async () => {
  const error = await assertRejects(
    () => openerOf({ "dit[0]": part(0), "dit[2]": part(2) }),
    Error,
  );
  assertStringIncludes(error.message, "part 添字が [0] から連続していない");
  // 揃っているキーを列挙する（読み手が現物と突き合わせられる形 — 既存の資産診断の流儀）。
  assertStringIncludes(error.message, "dit[0] / dit[2]");
});

test("assetComponentOpener: 単一形キーと分割形キーの混在は落ちる", async () => {
  const error = await assertRejects(() => openerOf({ dit: single(), ...parts("dit") }), Error);
  assertStringIncludes(error.message, "単一形のキーと分割形のキー");
  assertStringIncludes(error.message, "dit / dit[0]");
});

test("assetComponentOpener: 単一形キーと [0] 以外の分割形キーの混在も落ちる（[0] 欠落で門を素通りしない）", async () => {
  // `[0]` から連続する範囲だけを拾う実装だと、`[0]` が無いときに混在検査を飛ばして「単一形の
  // 1 本」として組んでしまう（`dit[1]` を黙って無視する = 分割配布のつもりで組んだ Record が
  // 遠くの層から「part が足りない」で落ちる）。
  const error = await assertRejects(() => openerOf({ dit: single(), "dit[1]": part(1) }), Error);
  assertStringIncludes(error.message, "単一形のキーと分割形のキー");
  assertStringIncludes(error.message, "dit / dit[1]");
});

test("assetComponentOpener: 添字が [0] から始まらない列は始点を名指しして落ちる", async () => {
  const error = await assertRejects(
    () => openerOf({ "dit[1]": part(1), "dit[2]": part(2) }),
    Error,
  );
  assertStringIncludes(error.message, "part 添字が [0] から始まっていない");
  assertStringIncludes(error.message, "dit[1] / dit[2]");
});

test("assetComponentOpener: [0] から連続する列は part 列のまま開ける", async () => {
  const open = await openerOf(parts("dit"));
  assertEquals(open("dit").graph.outputs, ["y"]);
});

test("assetComponentOpener: 単一形の 1 本も同じ供給口から開ける", async () => {
  const open = await openerOf({ dit: single() });
  assertEquals(open("dit").graph.outputs, ["y"]);
});

test("assetComponentOpener: キーがどちらの形でも無ければ 2 形の綴りを添えて落ちる", async () => {
  const error = await assertRejects(() => openerOf(parts("dit"), ["dit", "vae"]), Error);
  assertStringIncludes(error.message, "部品 'vae' の容器が無い");
  assertStringIncludes(error.message, "分割形なら 'vae[0]' から添字順");
});

test("assetComponentOpener: 重みの part が欠けた列は part 数と宣言の食い違いを名乗って落ちる", async () => {
  // 末尾の part（重みの block を持つ）を落とした列 — descriptor が宣言する part 数と合わない。
  const error = await assertRejects(
    () => openerOf({ "dit[0]": part(0), "dit[1]": part(1) }),
    ContainerFormatError,
  );
  assertStringIncludes(error.message, "part が 2 本だが宣言は 3 本");
});
