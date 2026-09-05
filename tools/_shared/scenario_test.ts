// シナリオ束縛の門のうち、**修飾なしキー**（`SYM`）の誤綴りを見る 1 本。
//
// 修飾キー（`<component>.SYM`）側は resolveComponentBindings が落とし、census_test.ts が
// 押さえている。ここが見るのはその非対称の残り — どのコンポーネントも宣言しない綴りは、
// 値が効かないだけでなく `unused_bindings` にも出ないので、誤綴りの記録がどこにも残らない。

import { assertThrows } from "@std/assert";
import { parseIrGraph } from "../../packages/runtime/src/format/ir.ts";
import type { IrGraph } from "../../packages/runtime/src/format/ir.ts";
import { assertPlainBindingKeys, parseScenario } from "./scenario.ts";

/**
 * 記号を名乗る最小グラフ（ノードは要らない — 見るのは `symbols` の和集合）。
 * 記号は入力 shape の次元位置に置く（IR は束縛の取れない記号宣言を受け付けない）。
 */
const graphWith = (symbols: readonly string[]): IrGraph =>
  parseIrGraph(JSON.stringify({
    format: "karume-ir",
    version: 1,
    requires: { ops: [] },
    symbols,
    inputs: symbols.map((symbol, at) => ({ name: `x${at}`, dtype: "f32", shape: [symbol] })),
    outputs: [],
    initializers: {},
    values: {},
    nodes: [],
  }));

Deno.test("assertPlainBindingKeys: どのコンポーネントも宣言しない記号は既知一覧つきで落ちる", () => {
  assertThrows(
    () =>
      assertPlainBindingKeys(parseScenario("decode=M:1,CC:4096"), [
        graphWith(["M", "C"]),
        graphWith(["M"]),
      ]),
    Error,
    "CC",
  );
  assertThrows(
    () => assertPlainBindingKeys(parseScenario("decode=M:1,CC:4096"), [graphWith(["M", "C"])]),
    Error,
    "既知: C, M",
  );
});

Deno.test("assertPlainBindingKeys: 1 つのコンポーネントだけが宣言する記号は和集合で通る", () => {
  // 家族共通の束縛を全コンポーネントへ配る使い方（irodori の T など）を壊さない。
  assertPlainBindingKeys(parseScenario("rep=T:256,S:750"), [
    graphWith(["T"]),
    graphWith(["S"]),
  ]);
});

Deno.test("assertPlainBindingKeys: 修飾キーはここでは見ない（component 側の門が持つ）", () => {
  assertPlainBindingKeys(parseScenario("rep=codec_encoder.T:750"), [graphWith(["S"])]);
});
