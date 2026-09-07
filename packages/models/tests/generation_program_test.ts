// 静的配線（`src/generation/program.ts`）の setup 検証テスト。GPU も実資産も要らない。
//
// ここで縛るのは ADR 0083 決定 1 の「program は setup 時に**全結線**を検証する不変オブジェクト」
// という契約そのもの。名前の取り違えも形の食い違いも、実行時には**例外を出さない**か真因から
// 遠い場所で出る（形の合う別の出力を掴めば、もっともらしい token 列が黙って返る）ので、
// 検証が抜けた欄はそのまま沈黙劣化の入口になる。

import { assert, assertEquals, assertThrows } from "@std/assert";
import type { PreparedModel } from "@karume/runtime";
import {
  createGenerationProgram,
  type GenerationGraph,
  generationProgramFace,
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

Deno.test("createGenerationProgram: chunkBuckets は複製し、省略は空配列へ畳む", () => {
  const chunkBuckets = [2, 3];
  const program = createGenerationProgram(specOf({ chunkBuckets }));
  assertEquals(program.chunkBuckets, [2, 3]);
  // 呼び手の配列と縁を切る（後から足された段は「context が許さない物理行数」になる）。
  chunkBuckets.push(3);
  assertEquals(program.chunkBuckets, [2, 3]);
  // 省略 = 追加なし。`undefined` のまま持ち回すと、選ぶ側が `?? []` を書き忘れても赤くならない。
  assertEquals(createGenerationProgram(specOf()).chunkBuckets, []);
});

Deno.test("createGenerationProgram: chunkBuckets の受理集合は runtime の門をそのまま通す", () => {
  // 規則（2 以上 chunkLength 未満・狭義昇順）は models 側に写していないので、ここで縛るのは
  // 「context を作る前に落ちる」こと自体である（写した規則が育つと context と食い違う）。
  const cases: readonly (readonly [readonly number[], string])[] = [
    // 1 は decode 形そのもの / chunkLength 以上は queryLength ≤ chunkLength 契約の外。
    [[1], "chunkBuckets[0] 1 が 2..3 の整数でない"],
    [[4], "chunkBuckets[0] 4 が 2..3 の整数でない"],
    [[2.5], "chunkBuckets[0] 2.5 が 2..3 の整数でない"],
    // 重複と降順は「queryLength 以上の最小バケット」を線形走査で選べなくする。
    [[2, 2], "chunkBuckets[1] 2 が直前の 2 以下"],
    [[3, 2], "chunkBuckets[1] 2 が直前の 3 以下"],
  ];
  for (const [chunkBuckets, message] of cases) {
    assertThrows(() => createGenerationProgram(specOf({ chunkBuckets })), Error, message);
  }
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

// ---- 公開の読み口（`generationProgramFace`）----------------------------------
//
// `GenerationProgram` は `Gemma4Pipeline.program` としてパッケージの公開面に出る値なので、
// 守る先は消費者側である。これを叩く門は実資産 + 実 GPU の e2e 1 箇所にしか無く、
// `Object.freeze` を外す / `[...wiring.stopTokens]` を `wiring.stopTokens` に戻す変更が、
// 資産の無い環境（CI・多くの開発機）では全部緑のまま通っていた。

Deno.test("generationProgramFace: 出る欄は数 6 つだけ（内部配線を出さない）", () => {
  const wiring = createGenerationProgram(specOf({ stopTokens: [1, 7], chunkBuckets: [2] }));
  const face = generationProgramFace(wiring);
  assertEquals(Object.keys(face).sort(), [
    "capacity",
    "chunkBuckets",
    "chunkLength",
    "maxPosition",
    "stopTokens",
    "vocabSize",
  ]);
  assertEquals(face, {
    chunkLength: 4,
    chunkBuckets: [2],
    maxPosition: 128,
    capacity: 64,
    vocabSize: VOCAB,
    stopTokens: [1, 7],
  });
});

Deno.test("generationProgramFace: 凍結コピーを返す（消費者の書き換えが停止集合へ届かない）", () => {
  const wiring = createGenerationProgram(specOf({ stopTokens: [1, 7], chunkBuckets: [2, 3] }));
  const face = generationProgramFace(wiring);
  assertEquals(Object.isFrozen(face), true, "face が凍結されていない");
  assertEquals(Object.isFrozen(face.stopTokens), true, "stopTokens が凍結されていない");
  // バケットも同じ扱い（並べ替えられると物理行数の選び方そのものが変わる）。
  assertEquals(Object.isFrozen(face.chunkBuckets), true, "chunkBuckets が凍結されていない");
  assertThrows(() => {
    (face.chunkBuckets as number[]).reverse();
  }, TypeError);
  assert(face.chunkBuckets !== wiring.chunkBuckets, "配線の配列をそのまま出している");
  // ESM は常に strict mode なので、凍結配列への書き込みは黙って捨てられず TypeError になる。
  assertThrows(() => {
    (face.stopTokens as number[]).length = 0;
  }, TypeError);
  assertThrows(() => {
    (face.stopTokens as number[]).sort();
  }, TypeError);
  // 別実体であること（凍結だけ残してコピーを落とすと、wiring 側の配列まで凍る副作用が出る）。
  assert(face.stopTokens !== wiring.stopTokens, "配線の配列をそのまま出している");
  assertEquals(Object.isFrozen(wiring.stopTokens), false, "配線側まで凍らせている");
  assertEquals([...wiring.stopTokens], [1, 7], "配線側の停止集合は無傷");
});
