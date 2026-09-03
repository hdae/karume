// 静的配線（`src/generation/program.ts`）の setup 検証テスト。GPU も実資産も要らない。
//
// ここで縛るのは ADR 0083 決定 1 の「program は setup 時に**全結線**を検証する不変オブジェクト」
// という契約そのもの。名前の取り違えも形の食い違いも、実行時には**例外を出さない**か真因から
// 遠い場所で出る（形の合う別の出力を掴めば、もっともらしい token 列が黙って返る）ので、
// 検証が抜けた欄はそのまま沈黙劣化の入口になる。

import { assertEquals, assertThrows } from "@std/assert";
import type { PreparedModel } from "@karume/runtime";
import {
  createGenerationProgram,
  type GenerationGraph,
  type GenerationProgramSpec,
} from "../src/generation/program.ts";

const VOCAB = 64;
const IDS = "input_ids";
const LAST_ROW = "last_row";
const LOGITS = "logits";
const DERIVED = "per_layer_inputs";

type GraphInput = GenerationGraph["inputs"][number];

/** 製品グラフ（gemma4 の実形を縮めたもの）— 派生入力の有無だけ選べる。 */
const graphOf = (options: { readonly derived?: boolean } = {}): GenerationGraph => ({
  symbols: ["C", "M"],
  inputs: [
    { name: IDS, dtype: "i32", shape: [1, "M"] },
    ...(options.derived === false
      ? []
      : [{ name: DERIVED, dtype: "f32", shape: [1, "M", 2, 3] } satisfies GraphInput]),
    { name: LAST_ROW, dtype: "i32", shape: [1] },
  ],
  outputs: [LOGITS],
  values: { [LOGITS]: { dtype: "f32", shape: [1, 1, VOCAB] } },
});

const specOf = (
  override: Partial<GenerationProgramSpec> = {},
): GenerationProgramSpec => ({
  graph: graphOf(),
  inputIds: IDS,
  lastRow: LAST_ROW,
  logits: LOGITS,
  chunkLength: 4,
  maxPosition: 128,
  capacity: 64,
  vocabSize: VOCAB,
  stopTokens: [7],
  capacitySymbol: "C",
  derivedInputs: { names: [DERIVED], derive: () => Promise.resolve({}) },
  ...override,
});

Deno.test("createGenerationProgram: 製品形の配線をそのまま通し、graph を持ち越さない", () => {
  const program = createGenerationProgram(specOf());
  assertEquals(program.inputIds, IDS);
  assertEquals(program.logits, LOGITS);
  assertEquals(program.chunkLength, 4);
  assertEquals(program.capacity, 64);
  assertEquals(program.stopTokens, [7]);
  // 検証に使ったグラフは program に残さない（不変の配線だけを持つ — ADR 0083 決定 1）。
  assertEquals(Object.hasOwn(program, "graph"), false);
});

Deno.test("createGenerationProgram: stopTokens は複製する（呼び手の配列と縁を切る）", () => {
  const stopTokens = [1, 7];
  const program = createGenerationProgram(specOf({ stopTokens }));
  stopTokens.push(9);
  assertEquals(program.stopTokens, [1, 7]);
});

Deno.test("createGenerationProgram: 派生入力の無いグラフは derivedInputs 省略で通る", () => {
  const program = createGenerationProgram(
    specOf({ graph: graphOf({ derived: false }), derivedInputs: undefined }),
  );
  assertEquals(program.derivedInputs, undefined);
});

Deno.test("createGenerationProgram: 数値の受理集合", () => {
  const cases: readonly (readonly [Partial<GenerationProgramSpec>, string])[] = [
    [{ chunkLength: 0 }, "chunkLength 0"],
    [{ chunkLength: 2.5 }, "chunkLength 2.5"],
    [{ maxPosition: 0 }, "maxPosition 0"],
    [{ capacity: -1 }, "capacity -1"],
    [{ vocabSize: 0 }, "vocabSize 0"],
    // 停止 token が語彙の外だと「絶対に成立しない停止条件」が黙って積まれる。
    [{ stopTokens: [VOCAB] }, `stopTokens[0] ${VOCAB} が語彙 0..${VOCAB - 1} の外`],
    [{ stopTokens: [-1] }, "stopTokens[0] -1"],
    [{ stopTokens: [0, 1.5] }, "stopTokens[1] 1.5"],
  ];
  for (const [override, message] of cases) {
    assertThrows(() => createGenerationProgram(specOf(override)), Error, message);
  }
});

Deno.test("createGenerationProgram: 入力名 / dtype / 形が違えば fail loudly", () => {
  const cases: readonly (readonly [string, Partial<GenerationProgramSpec>, string])[] = [
    ["token id 入力が無い", { inputIds: "tokens" }, "token id 入力 'tokens' がグラフ入力に無い"],
    ["last_row が無い", { lastRow: "row" }, "last_row 入力 'row' がグラフ入力に無い"],
    [
      "token id 入力が f32",
      {
        graph: {
          ...graphOf(),
          inputs: graphOf().inputs.map((input) =>
            input.name === IDS ? { ...input, dtype: "f32" } : input
          ),
        },
      },
      "token id 入力 'input_ids' の dtype が f32",
    ],
    [
      // 固定数の M は prefill 形（M=chunkLength）と decode 形（M=1）を同じグラフで回せない。
      "M が固定数",
      {
        graph: {
          ...graphOf(),
          inputs: graphOf().inputs.map((input) =>
            input.name === IDS ? { ...input, shape: [1, 4] } : input
          ),
        },
      },
      "の shape [1,4] が [1,<記号>] でない",
    ],
    [
      "last_row の形が [1] でない",
      {
        graph: {
          ...graphOf(),
          inputs: graphOf().inputs.map((input) =>
            input.name === LAST_ROW ? { ...input, shape: [1, 1] } : input
          ),
        },
      },
      "last_row 入力 'last_row' の shape [1,1] が [1] でない",
    ],
  ];
  for (const [name, override, message] of cases) {
    assertThrows(() => createGenerationProgram(specOf(override)), Error, message, name);
  }
});

Deno.test("createGenerationProgram: logits 出口の実在 / 形 / 語彙数を見る", () => {
  assertThrows(
    () => createGenerationProgram(specOf({ logits: "scores" })),
    Error,
    "logits 出口 'scores' がグラフ出力に無い",
  );
  // ノード出力として存在するだけの名前は run から返ってこない（グラフ出力に載っているかを見る）。
  assertThrows(
    () =>
      createGenerationProgram(
        specOf({
          graph: {
            ...graphOf(),
            outputs: ["hidden"],
            values: { hidden: { dtype: "f32", shape: [1, 1, VOCAB] } },
          },
          logits: LOGITS,
        }),
      ),
    Error,
    "logits 出口 'logits' がグラフ出力に無い",
  );
  // 全行 logits（`[1,M,V]`）への退行 = 最終行出口でない。
  assertThrows(
    () =>
      createGenerationProgram(
        specOf({
          graph: { ...graphOf(), values: { [LOGITS]: { dtype: "f32", shape: [1, "M", VOCAB] } } },
        }),
      ),
    Error,
    "が [1,1,64] でない",
  );
  // 語彙数の食い違い（別世代の資産と program の組み合わせ）。
  assertThrows(
    () => createGenerationProgram(specOf({ vocabSize: 32, stopTokens: [7] })),
    Error,
    "が [1,1,32] でない",
  );
  assertThrows(
    () =>
      createGenerationProgram(
        specOf({
          graph: { ...graphOf(), values: { [LOGITS]: { dtype: "i32", shape: [1, 1, VOCAB] } } },
        }),
      ),
    Error,
    "logits 出口 'logits' の dtype が i32",
  );
});

Deno.test("createGenerationProgram: グラフ入力の被覆を両方向で見る", () => {
  // 欠け: per_layer_inputs をホスト側で供給しないまま program を組む。
  assertThrows(
    () => createGenerationProgram(specOf({ derivedInputs: undefined })),
    Error,
    `グラフ入力 ${DERIVED} が結線されていない`,
  );
  // 余り: 宣言した名前がグラフに無い（毎 run 無視される入力になる）。
  assertThrows(
    () =>
      createGenerationProgram(
        specOf({
          graph: graphOf({ derived: false }),
          derivedInputs: { names: ["ple"], derive: () => Promise.resolve({}) },
        }),
      ),
    Error,
    "結線した ple がグラフ入力に無い",
  );
  // 重複: 同じ名前を 2 度結線した形（片方が黙って上書きされる）。
  assertThrows(
    () =>
      createGenerationProgram(
        specOf({
          graph: graphOf({ derived: false }),
          derivedInputs: { names: [IDS], derive: () => Promise.resolve({}) },
        }),
      ),
    Error,
    "結線した入力名に重複がある",
  );
});

Deno.test("createGenerationProgram: 記号は入力 shape か容量記号のどちらかで決まること", () => {
  // C（state スロットの容量記号）は入力 shape に現れない = 容量記号が唯一の源。
  assertThrows(
    () => createGenerationProgram(specOf({ capacitySymbol: "K" })),
    Error,
    "容量記号 K がグラフの symbols [C, M] に無い",
  );
  // 入力 shape から決まる記号を容量記号に選ぶと、run の束縛と context の束縛が分裂する。
  assertThrows(
    () => createGenerationProgram(specOf({ capacitySymbol: "M" })),
    Error,
    "容量記号 M は入力 shape から決まる記号である",
  );
  // 容量記号が 1 本足りない形（states の記号が 2 本ある資産）。
  assertThrows(
    () =>
      createGenerationProgram(
        specOf({ graph: { ...graphOf(), symbols: ["C", "D", "M"] } }),
      ),
    Error,
    "記号 D が入力 shape からも容量記号からも決まらない",
  );
  // M は入力 shape から決まるので容量記号に要らない（C だけで通る）。
  assertEquals(createGenerationProgram(specOf()).capacitySymbol, "C");
});

Deno.test("GenerationGraph: 実 IrGraph がこの面を満たす（綴りのドリフト検出）", () => {
  // 型検査だけの門（実行時は何もしない）。`GenerationGraph` は `PreparedModel["graph"]` を
  // **写して**いるので、IR の inputs / outputs / values / symbols の綴りが runtime 側で
  // 変わるとこの 1 行がコンパイルエラーになる。
  const asGenerationGraph = (graph: PreparedModel["graph"]): GenerationGraph => graph;
  assertEquals(typeof asGenerationGraph, "function");
});
