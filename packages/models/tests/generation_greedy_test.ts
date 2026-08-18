// 固定長 greedy 生成ループ（`src/generation/greedy.ts`）の挙動テスト。GPU も実資産も要らない。
//
// ここで縛るのは「ホストが組み立てる 1 step ぶんの形」— 固定長 chunk の割り方・pad 行の 0・
// 絶対位置の進み方・前 step の返り token の feedback・context の寿命。どれも実 GPU で回すと
// 「値が少しおかしい」という形でしか出ず（ADR 0066 決定 4 の pad no-op 契約は、間違った位置や
// 残骸を食っても例外を出さない）、実装の誤りが検収 e2e の 1 個の不一致に潰れてしまう位置に
// ある。Session は narrow interface（`GreedySession`）で受けるので、fake は素の object 1 個。

import { assertEquals, assertRejects, assertThrows } from "@std/assert";
import type {
  GenerationContextSpec,
  RunInputs,
  RunOutputs,
  Session,
  SymbolBindings,
} from "@karume/runtime";
import { generateGreedy, type GreedySession, planPrefillChunks } from "../src/generation/greedy.ts";

type FakeContext = { dispose(): Promise<void> };

/** run 1 回ぶんの記録（呼び出し列だけで step の形が全部読める粒度にしてある）。 */
type RunCall = {
  readonly ids: readonly number[];
  readonly idsShape: readonly number[];
  readonly positions: readonly number[];
  readonly positionsShape: readonly number[];
  readonly bindings: SymbolBindings | undefined;
  readonly queryLength: number;
  readonly sameContext: boolean;
};

const IDS = "input_ids";
const POSITIONS = "position_ids";
const TOKEN = "next_token";

const readRow = (
  inputs: RunInputs,
  name: string,
): { readonly shape: readonly number[]; readonly values: readonly number[] } => {
  if (!Object.hasOwn(inputs, name)) throw new Error(`fake: 入力 '${name}' が渡っていない`);
  const tensor = inputs[name];
  if (!("data" in tensor)) throw new Error(`fake: 入力 '${name}' がホストテンソルでない`);
  if (tensor.dtype !== "i32") throw new Error(`fake: 入力 '${name}' が i32 でない`);
  return { shape: tensor.shape, values: [...tensor.data] };
};

/**
 * 既定の出力: `next_token[0, row, 0] = call * 1000 + row`。
 *
 * 呼び出し回数と行番号の両方を値へ畳んであるので、「どの run の何行目を読んだか」が返り値
 * 1 個で判別できる（行を取り違える実装 — 例えば常に 0 行目 — が prefill で必ず落ちる）。
 * 形の合う decoy（`logits`）を並べてあるのは、名前ではなく形で引く実装を落とすため。
 */
const defaultOutputs = (call: number, rows: number): RunOutputs => ({
  logits: { dtype: "f32", shape: [1, rows, 1], data: new Float32Array(rows) },
  [TOKEN]: {
    dtype: "i32",
    shape: [1, rows, 1],
    data: Int32Array.from({ length: rows }, (_, row) => call * 1000 + row),
  },
});

type FakeOptions = {
  /** run の返す出力（既定は {@link defaultOutputs}）。 */
  readonly outputs?: (call: number, rows: number) => RunOutputs;
  /** この回数目（0 始まり）の run を失敗させる。 */
  readonly failAt?: number;
};

const fakeSession = (options: FakeOptions = {}) => {
  const calls: RunCall[] = [];
  const specs: GenerationContextSpec[] = [];
  let disposals = 0;
  const context: FakeContext = {
    dispose: () => {
      disposals += 1;
      return Promise.resolve();
    },
  };
  const session: GreedySession<FakeContext> = {
    createGenerationContext: (spec) => {
      specs.push(spec);
      return Promise.resolve(context);
    },
    run: (inputs, bindings, generation) => {
      const ids = readRow(inputs, IDS);
      const positions = readRow(inputs, POSITIONS);
      const call = calls.length;
      calls.push({
        ids: ids.values,
        idsShape: ids.shape,
        positions: positions.values,
        positionsShape: positions.shape,
        bindings,
        queryLength: generation.queryLength,
        sameContext: generation.context === context,
      });
      if (options.failAt === call) return Promise.reject(new Error("run が落ちた"));
      return Promise.resolve((options.outputs ?? defaultOutputs)(call, ids.values.length));
    },
  };
  return { session, calls, specs, disposals: () => disposals };
};

/** 検証用のプロンプト（値が位置と 1 対 1 なので、ずれがそのまま値に出る）。 */
const promptOf = (length: number, base = 100): number[] =>
  Array.from({ length }, (_, index) => base + index);

Deno.test("planPrefillChunks: T=87 / L=32 を 32 + 32 + 23 の 3 chunk に割る", () => {
  assertEquals(planPrefillChunks(87, 32), [
    { position: 0, queryLength: 32 },
    { position: 32, queryLength: 32 },
    { position: 64, queryLength: 23 },
  ]);
});

Deno.test("planPrefillChunks: T が L の倍数なら pad 行の出る chunk が 1 本も無い", () => {
  assertEquals(planPrefillChunks(32, 32), [{ position: 0, queryLength: 32 }]);
  assertEquals(planPrefillChunks(64, 32), [
    { position: 0, queryLength: 32 },
    { position: 32, queryLength: 32 },
  ]);
});

Deno.test("planPrefillChunks: T < L は 1 chunk（残りが pad 行になる）", () => {
  assertEquals(planPrefillChunks(1, 32), [{ position: 0, queryLength: 1 }]);
  assertEquals(planPrefillChunks(31, 32), [{ position: 0, queryLength: 31 }]);
});

Deno.test("planPrefillChunks: 位置は飛ばず重ならず、有効行の総和が T に一致する", () => {
  // 割り方の不変条件そのもの（境界を 1 つ間違えると、どれかが必ず破れる）。
  for (const [length, chunkLength] of [[87, 32], [1, 1], [5, 7], [64, 8], [9, 4]]) {
    const chunks = planPrefillChunks(length, chunkLength);
    let expected = 0;
    for (const chunk of chunks) {
      assertEquals(chunk.position, expected);
      assertEquals(chunk.queryLength >= 1 && chunk.queryLength <= chunkLength, true);
      expected += chunk.queryLength;
    }
    assertEquals(expected, length);
  }
});

Deno.test("planPrefillChunks: 長さ / chunkLength が 1 以上の整数でなければ落ちる", () => {
  assertThrows(() => planPrefillChunks(0, 32), Error, "prompt の長さ 0");
  assertThrows(() => planPrefillChunks(-1, 32), Error, "prompt の長さ -1");
  assertThrows(() => planPrefillChunks(1.5, 32), Error, "prompt の長さ 1.5");
  assertThrows(() => planPrefillChunks(8, 0), Error, "chunkLength 0");
  assertThrows(() => planPrefillChunks(8, 2.5), Error, "chunkLength 2.5");
});

Deno.test("generateGreedy: prefill は固定長 chunk で流し、pad 行は 0・位置は絶対値", async () => {
  const fake = fakeSession();
  const prompt = promptOf(87);
  const generated = await generateGreedy({
    session: fake.session,
    inputIds: IDS,
    positionIds: POSITIONS,
    token: TOKEN,
    chunkLength: 32,
    prompt,
    maxNewTokens: 1,
  });

  assertEquals(fake.calls.length, 3);
  assertEquals(fake.calls.map((call) => call.queryLength), [32, 32, 23]);
  // 物理 chunk 行数は 3 本とも chunkLength 固定（ADR 0066 決定 4）。
  assertEquals(fake.calls.map((call) => call.idsShape), [[1, 32], [1, 32], [1, 32]]);
  assertEquals(fake.calls.map((call) => call.positionsShape), [[1, 32], [1, 32], [1, 32]]);

  assertEquals(fake.calls[0].ids, prompt.slice(0, 32));
  assertEquals(fake.calls[1].ids, prompt.slice(32, 64));
  // 末尾 chunk は 23 行が有効で、残り 9 行は pad = 0（追記 6 の値契約）。
  assertEquals(fake.calls[2].ids, [...prompt.slice(64, 87), ...new Array(9).fill(0)]);

  assertEquals(fake.calls[0].positions, Array.from({ length: 32 }, (_, i) => i));
  assertEquals(fake.calls[1].positions, Array.from({ length: 32 }, (_, i) => 32 + i));
  assertEquals(fake.calls[2].positions, [
    ...Array.from({ length: 23 }, (_, i) => 64 + i),
    ...new Array(9).fill(0),
  ]);

  // 最初の生成 token は**最終 chunk の最終有効行**（call=2 / row=22）から読む。
  assertEquals(generated, [2022]);
});

Deno.test("generateGreedy: T = chunkLength ちょうどは 1 chunk で pad 行が出ない", async () => {
  const fake = fakeSession();
  const prompt = promptOf(4);
  const generated = await generateGreedy({
    session: fake.session,
    inputIds: IDS,
    positionIds: POSITIONS,
    token: TOKEN,
    chunkLength: 4,
    prompt,
    maxNewTokens: 1,
  });

  assertEquals(fake.calls.length, 1);
  assertEquals(fake.calls[0].queryLength, 4);
  assertEquals(fake.calls[0].ids, prompt);
  assertEquals(fake.calls[0].positions, [0, 1, 2, 3]);
  assertEquals(generated, [3]);
});

Deno.test("generateGreedy: T < chunkLength は 1 chunk・末尾が pad 0 で埋まる", async () => {
  const fake = fakeSession();
  const generated = await generateGreedy({
    session: fake.session,
    inputIds: IDS,
    positionIds: POSITIONS,
    token: TOKEN,
    chunkLength: 4,
    prompt: [7, 8],
    maxNewTokens: 1,
  });

  assertEquals(fake.calls.length, 1);
  assertEquals(fake.calls[0].queryLength, 2);
  assertEquals(fake.calls[0].ids, [7, 8, 0, 0]);
  assertEquals(fake.calls[0].positions, [0, 1, 0, 0]);
  // 読むのは最終有効行（row=1）で、pad 行（row=2,3）ではない。
  assertEquals(generated, [1]);
});

Deno.test("generateGreedy: decode は queryLength=1 固定で、位置が T, T+1, … と進む", async () => {
  const fake = fakeSession();
  const generated = await generateGreedy({
    session: fake.session,
    inputIds: IDS,
    positionIds: POSITIONS,
    token: TOKEN,
    chunkLength: 4,
    prompt: [10, 11, 12],
    maxNewTokens: 3,
  });

  assertEquals(fake.calls.length, 3);
  assertEquals(fake.calls.map((call) => call.queryLength), [4 - 1, 1, 1]);
  // decode の物理 chunk 行数は 1（prefill 形とは別の PreparedPlan — ADR 0066 決定 4）。
  assertEquals(fake.calls[1].idsShape, [1, 1]);
  assertEquals(fake.calls[2].idsShape, [1, 1]);
  assertEquals(fake.calls[1].positions, [3]);
  assertEquals(fake.calls[2].positions, [4]);

  // feedback: 前 step の返り token がそのまま次の入力になる。
  assertEquals(generated, [2, 1000, 2000]);
  assertEquals(fake.calls[1].ids, [generated[0]]);
  assertEquals(fake.calls[2].ids, [generated[1]]);
});

Deno.test("generateGreedy: maxNewTokens=1 は decode を 1 回も回さない", async () => {
  const fake = fakeSession();
  const generated = await generateGreedy({
    session: fake.session,
    inputIds: IDS,
    positionIds: POSITIONS,
    token: TOKEN,
    chunkLength: 4,
    prompt: [10, 11, 12],
    maxNewTokens: 1,
  });

  assertEquals(generated.length, 1);
  assertEquals(fake.calls.length, 1);
});

Deno.test("generateGreedy: 容量記号は createGenerationContext だけへ渡す（run には渡さない）", async () => {
  // ADR 0066 追記 7 — states 専用記号の束縛点は context 生成だけ。run の bindings へ回すと
  // 「states は束縛源にならない」検査に引っかかるか、計画鍵に余分な記号が混ざる。
  const fake = fakeSession();
  await generateGreedy({
    session: fake.session,
    inputIds: IDS,
    positionIds: POSITIONS,
    token: TOKEN,
    chunkLength: 4,
    bindings: { C: 8 },
    prompt: [10, 11],
    maxNewTokens: 2,
  });

  assertEquals(fake.specs.length, 1);
  assertEquals(fake.specs[0].chunkLength, 4);
  assertEquals(fake.specs[0].bindings, { C: 8 });
  assertEquals(fake.calls.map((call) => call.bindings), [undefined, undefined]);
  // 全 step が同じ context を使う（1 生成 = 1 context）。
  assertEquals(fake.calls.every((call) => call.sameContext), true);
});

Deno.test("generateGreedy: 正常終了で context を 1 度だけ dispose する", async () => {
  const fake = fakeSession();
  await generateGreedy({
    session: fake.session,
    inputIds: IDS,
    positionIds: POSITIONS,
    token: TOKEN,
    chunkLength: 4,
    prompt: [10, 11],
    maxNewTokens: 3,
  });
  assertEquals(fake.disposals(), 1);
});

Deno.test("generateGreedy: run が落ちても context を dispose し、例外はそのまま伝播する", async () => {
  // 失敗した context は poison 化して再利用できない（ADR 0066 追記 3）ので、抱えたままにすると
  // KV 容量ぶんの VRAM が生成 1 本ごとに積み上がる。prefill 側と decode 側の両方を見る。
  for (const failAt of [0, 1]) {
    const fake = fakeSession({ failAt });
    await assertRejects(
      () =>
        generateGreedy({
          session: fake.session,
          inputIds: IDS,
          positionIds: POSITIONS,
          token: TOKEN,
          chunkLength: 4,
          prompt: [10, 11],
          maxNewTokens: 3,
        }),
      Error,
      "run が落ちた",
    );
    assertEquals(fake.calls.length, failAt + 1);
    assertEquals(fake.disposals(), 1);
  }
});

Deno.test("generateGreedy: token 出力の名前 / dtype / 形が違えば fail loudly", async () => {
  const cases: readonly (readonly [string, (call: number, rows: number) => RunOutputs, string])[] =
    [
      [
        "名前が無い",
        (_call, rows) => ({
          logits: { dtype: "f32", shape: [1, rows, 1], data: new Float32Array(rows) },
        }),
        "グラフ出力 'next_token' が無い",
      ],
      [
        "dtype が f32",
        (_call, rows) => ({
          [TOKEN]: { dtype: "f32", shape: [1, rows, 1], data: new Float32Array(rows) },
        }),
        "が i32 でない",
      ],
      [
        "rank が 2",
        (_call, rows) => ({
          [TOKEN]: { dtype: "i32", shape: [1, rows], data: new Int32Array(rows) },
        }),
        "が [1,4,1] でない",
      ],
      [
        "行数が物理 chunk 行数と違う",
        (_call, rows) => ({
          [TOKEN]: { dtype: "i32", shape: [1, rows + 1, 1], data: new Int32Array(rows + 1) },
        }),
        "が [1,4,1] でない",
      ],
      [
        "最終軸が 1 でない",
        (_call, rows) => ({
          [TOKEN]: { dtype: "i32", shape: [1, rows, 2], data: new Int32Array(rows * 2) },
        }),
        "が [1,4,1] でない",
      ],
    ];

  for (const [name, outputs, message] of cases) {
    const fake = fakeSession({ outputs });
    await assertRejects(
      () =>
        generateGreedy({
          session: fake.session,
          inputIds: IDS,
          positionIds: POSITIONS,
          token: TOKEN,
          chunkLength: 4,
          prompt: [10, 11],
          maxNewTokens: 2,
        }),
      Error,
      message,
      name,
    );
    // 検査で落ちても context は返る。
    assertEquals(fake.disposals(), 1, name);
  }
});

Deno.test("generateGreedy: 入口検査は context を作る前に落ちる", async () => {
  const base = {
    inputIds: IDS,
    positionIds: POSITIONS,
    token: TOKEN,
    chunkLength: 4,
    prompt: [10, 11],
    maxNewTokens: 2,
  };
  const cases: readonly (readonly [Partial<typeof base>, string])[] = [
    [{ prompt: [] }, "prompt の長さ 0"],
    [{ chunkLength: 0 }, "chunkLength 0"],
    [{ chunkLength: 2.5 }, "chunkLength 2.5"],
    [{ maxNewTokens: 0 }, "maxNewTokens 0"],
    [{ maxNewTokens: -1 }, "maxNewTokens -1"],
    [{ maxNewTokens: 1.5 }, "maxNewTokens 1.5"],
    [{ prompt: [10, 1.5] }, "prompt[1] 1.5 が整数でない"],
  ];

  for (const [override, message] of cases) {
    const fake = fakeSession();
    await assertRejects(
      () => generateGreedy({ ...base, ...override, session: fake.session }),
      Error,
      message,
    );
    assertEquals(fake.specs.length, 0, message);
    assertEquals(fake.calls.length, 0, message);
  }
});

Deno.test("GreedySession: 実 Session の面を型で満たす（綴りのドリフト検出）", () => {
  // 型検査だけの門（実行時は何もしない）。`GreedySession` は fake を差せるように実 Session の
  // 面を**写して**いるので、`run` / `createGenerationContext` の綴りや引数が runtime 側で
  // 変わると、この 1 行がコンパイルエラーになる（写しがずれたまま単体テストだけ緑、という
  // 沈黙劣化の唯一の検出点）。
  const asGreedySession = (session: Session): GreedySession => session;
  assertEquals(typeof asGreedySession, "function");
});
