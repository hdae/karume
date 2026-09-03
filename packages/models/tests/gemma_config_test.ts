// gemma4 の `pipelineConfig` スキーマ（`src/gemma/config.ts`）の門 —— GPU も資産も要らない。
//
// hub は `pipelineConfig` を素通しするので、形の正本は models 側のこのパーサ 1 本しかない。
// ここで押さえるのは 3 つ:
//
//  ① **欄の受理集合**（未知キー・欠落・型・値域）— 綴り違いが黙って既定へ縮退すると、配布者の
//     宣言と実行が食い違ったまま気づけない
//  ② **3 つの数の関係**（`chunkLength ≤ capacity ≤ maxPosition`）— 容量いっぱいの会話は位置
//     `capacity - 1` まで進むので、RoPE 表の行数を超える宣言は**長い会話でだけ**落ちる
//  ③ **焼く側との一致** — 手元に配布形ミラーがあれば、その `karume.json` の宣言がこのパーサを
//     素通りし、推奨サンプラが上流 `generation_config.json` の値そのものであること
//     （ADR 0083 決定 7 の「既定値は配布形が宣言する」の実測）
//
// ③ の資産（`models/karume-gemma4-e2b/`）はリポジトリ管理外なので、無い環境では**その 1 本だけ**
// を明示 SKIP する（①② は常に走る）。

import { assertEquals, assertRejects, assertThrows } from "@std/assert";
import { parseManifest } from "@karume/hub";
import { type Gemma4PipelineConfig, parseGemma4PipelineConfig } from "../src/gemma/config.ts";
import { Gemma4Pipeline } from "../src/gemma/pipeline.ts";

/**
 * 受理される最小形（3 つの数 + `rope` — `sampler` だけが optional）。
 *
 * `rope` は Gemma 4 E2B の実値（sliding は default rope・full は partial rotary 0.25 の
 * proportional）。表は配布形に無く、cos / sin はこの宣言からホストが作る。
 */
const ROPE = {
  sliding_attention: { theta: 10000, headDim: 256, rotaryDim: 256 },
  full_attention: { theta: 1000000, headDim: 512, rotaryDim: 128 },
} as const;
const MINIMAL = { chunkLength: 32, maxPosition: 1024, capacity: 640, rope: ROPE } as const;
/** 配布形ミラーが宣言する既定（capacity / chunkLength はどちらも実行時ノブの**既定**）。 */
const SHIPPED_CHUNK_LENGTH = 768;
const SHIPPED_CAPACITY = 4096;
/** 上流 `max_position_embeddings`（= モデルが宣言する位置上限）。 */
const SHIPPED_MAX_POSITION = 131072;

/** 上流 `gemma-4-E2B-it` の `generation_config.json` の推奨（ADR 0083 決定 7）。 */
const RECOMMENDED = { temperature: 1, topK: 64, topP: 0.95 } as const;

const MIRROR = new URL("../../../models/karume-gemma4-e2b/karume.json", import.meta.url);

const readMirror = (): string | undefined => {
  try {
    return Deno.readTextFileSync(MIRROR);
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return undefined;
    throw cause;
  }
};

Deno.test("gemma4 pipelineConfig: 3 つの数 + rope と optional な sampler を読む", () => {
  assertEquals(parseGemma4PipelineConfig(MINIMAL), MINIMAL);
  assertEquals(
    parseGemma4PipelineConfig({ ...MINIMAL, sampler: { ...RECOMMENDED } }),
    { ...MINIMAL, sampler: RECOMMENDED },
  );
});

Deno.test("gemma4 pipelineConfig: 未知キーと欠落は fail loudly", async (t) => {
  await t.step("オブジェクトでない宣言（`unknown` の境界パーサ）", () => {
    for (const raw of [undefined, null, 42, "chunkLength=32", [MINIMAL]]) {
      assertThrows(() => parseGemma4PipelineConfig(raw), Error, "オブジェクトでない");
    }
  });
  await t.step("根の未知キー（綴り違いが既定へ縮退しない）", () => {
    assertThrows(
      () => parseGemma4PipelineConfig({ ...MINIMAL, capasity: 640 }),
      Error,
      "未知キー 'capasity'",
    );
  });
  await t.step("sampler の未知キー（宣言できる欄は 3 つだけ）", () => {
    assertThrows(
      () => parseGemma4PipelineConfig({ ...MINIMAL, sampler: { ...RECOMMENDED, seed: 7 } }),
      Error,
      "未知キー 'seed'",
    );
  });
  await t.step("欄の欠落", () => {
    const { capacity: _dropped, ...rest } = MINIMAL;
    assertThrows(() => parseGemma4PipelineConfig(rest), Error, "pipelineConfig.capacity: 無い");
  });
  await t.step("rope は必須（欠けたまま動くと回転しない attention が黙って走る）", () => {
    const { rope: _dropped, ...rest } = MINIMAL;
    assertThrows(() => parseGemma4PipelineConfig(rest), Error, "pipelineConfig.rope: 無い");
  });
  await t.step("sampler は欄ごと無ければ通る（= 低層の既定 = greedy）", () => {
    assertEquals(parseGemma4PipelineConfig(MINIMAL).sampler, undefined);
  });
});

Deno.test("gemma4 pipelineConfig: rope の受理集合", async (t) => {
  const withRope = (rope: unknown): unknown => ({ ...MINIMAL, rope });

  await t.step("層種別は 2 本ちょうど（未知の層種別・欠落）", () => {
    assertThrows(
      () => parseGemma4PipelineConfig(withRope({ ...ROPE, local_attention: ROPE.full_attention })),
      Error,
      "未知キー 'local_attention'",
    );
    const { full_attention: _dropped, ...sliding } = ROPE;
    assertThrows(
      () => parseGemma4PipelineConfig(withRope(sliding)),
      Error,
      "pipelineConfig.rope.full_attention: 無い",
    );
  });

  await t.step("層種別の欄は 3 つちょうど（綴り違いが既定へ縮退しない）", () => {
    assertThrows(
      () =>
        parseGemma4PipelineConfig(
          withRope({ ...ROPE, full_attention: { ...ROPE.full_attention, ropeType: "linear" } }),
        ),
      Error,
      "未知キー 'ropeType'",
    );
    assertThrows(
      () =>
        parseGemma4PipelineConfig(
          withRope({ ...ROPE, sliding_attention: { theta: 10000, headDim: 256 } }),
        ),
      Error,
      "pipelineConfig.rope.sliding_attention.rotaryDim: 無い",
    );
    assertThrows(
      () =>
        parseGemma4PipelineConfig(
          withRope({ ...ROPE, sliding_attention: { ...ROPE.sliding_attention, theta: "10000" } }),
        ),
      Error,
      "pipelineConfig.rope.sliding_attention.theta: 数でない",
    );
  });

  await t.step("値域の門は rope.ts が持つ（同じ式を 2 実装しない）", () => {
    // 奇数の rotaryDim は「前半 = 後半」の並びが崩れる = 上流と別の表を黙って作る。
    assertThrows(
      () =>
        parseGemma4PipelineConfig(
          withRope({ ...ROPE, full_attention: { ...ROPE.full_attention, rotaryDim: 129 } }),
        ),
      Error,
      "rotaryDim は 2 以上 headDim 以下の偶数",
    );
    assertThrows(
      () =>
        parseGemma4PipelineConfig(
          withRope({ ...ROPE, full_attention: { ...ROPE.full_attention, rotaryDim: 1024 } }),
        ),
      Error,
      "rotaryDim は 2 以上 headDim 以下の偶数",
    );
    assertThrows(
      () =>
        parseGemma4PipelineConfig(
          withRope({ ...ROPE, sliding_attention: { ...ROPE.sliding_attention, theta: 0 } }),
        ),
      Error,
      "theta は正の有限値",
    );
    assertThrows(
      () =>
        parseGemma4PipelineConfig(
          withRope({ ...ROPE, sliding_attention: { ...ROPE.sliding_attention, headDim: 255 } }),
        ),
      Error,
      "headDim は 2 以上の偶数",
    );
  });
});

Deno.test("gemma4 pipelineConfig: 値域", async (t) => {
  await t.step("数は 1 以上の整数", () => {
    assertThrows(() => parseGemma4PipelineConfig({ ...MINIMAL, chunkLength: 0 }), Error);
    assertThrows(() => parseGemma4PipelineConfig({ ...MINIMAL, maxPosition: 1.5 }), Error);
  });
  await t.step("sampler の値域は generation/sampler.ts の assertSpec と同じ", () => {
    const sampler = { ...RECOMMENDED };
    assertThrows(
      () => parseGemma4PipelineConfig({ ...MINIMAL, sampler: { ...sampler, temperature: -1 } }),
      Error,
      "0 以上の有限数でない",
    );
    assertThrows(
      () => parseGemma4PipelineConfig({ ...MINIMAL, sampler: { ...sampler, topK: 0 } }),
      Error,
      "1 以上の整数でない",
    );
    assertThrows(
      () => parseGemma4PipelineConfig({ ...MINIMAL, sampler: { ...sampler, topP: 0 } }),
      Error,
      "0 < topP ≤ 1 の範囲にない",
    );
  });
});

Deno.test("gemma4 pipelineConfig: chunkLength ≤ capacity ≤ maxPosition", async (t) => {
  await t.step("1 chunk すら入らない容量", () => {
    assertThrows(
      () => parseGemma4PipelineConfig({ ...MINIMAL, chunkLength: 64, capacity: 32 }),
      Error,
      "capacity 32 を超えた",
    );
  });
  await t.step("容量いっぱいの会話がモデルの位置上限の外を引く", () => {
    assertThrows(
      () => parseGemma4PipelineConfig({ ...MINIMAL, capacity: 2048, maxPosition: 1024 }),
      Error,
      "maxPosition 1024 を超えた",
    );
  });
});

// ---- 入口との結線 ----------------------------------------------------------
//
// `fromPretrained` は manifest を読む段でこのパーサを通す（`gemma4ManifestConfig`）。`fromAssets`
// は TS の型で受けるが、型は未知キーも値域も見ない — 門が無いと `temperature: -1` の宣言が
// **3.7GiB を読み切った後の初 `chat`** で初めて落ちる。ここで見るのは「同じパーサを、しかも
// バイト列を 1 本も開く前に通す」ことなので、`model` にはゴミを渡す（開かれていれば
// コンテナ側の文言で落ちる = この門が抜けている）。

Deno.test("gemma4 fromAssets: config は fromPretrained と同じ門を、バイト列を開く前に通る", async (t) => {
  const junk = new Uint8Array<ArrayBuffer>(new ArrayBuffer(8));
  const fromAssets = (config: unknown): Promise<Gemma4Pipeline> =>
    Gemma4Pipeline.fromAssets({
      // 型を持たない JS からの呼び出しの形（TS では未知キーを型検査が先に落とす）。
      config: config as Gemma4PipelineConfig,
      model: [junk],
      tokenizer: junk,
      pleIndex: junk,
      readPleShard: () => Promise.reject(new Error("gemma_config_test: PLE を読みに行った")),
    });

  await t.step("未知キー", async () => {
    await assertRejects(
      () => fromAssets({ ...MINIMAL, capasity: 640 }),
      Error,
      "未知キー 'capasity'",
    );
  });
  await t.step("3 つの数の関係", async () => {
    await assertRejects(
      () => fromAssets({ ...MINIMAL, capacity: 2048 }),
      Error,
      "maxPosition 1024 を超えた",
    );
  });
  await t.step("sampler の値域（V2-b が名指しした非対称そのもの）", async () => {
    await assertRejects(
      () => fromAssets({ ...MINIMAL, sampler: { ...RECOMMENDED, temperature: -1 } }),
      Error,
      "0 以上の有限数でない",
    );
  });
  await t.step("宣言が正しければ門を抜け、コンテナ側の文言で落ちる（陰性対照）", async () => {
    // 「config の門で全部落ちているから緑」ではないことを見る対。
    const error = await assertRejects(() => fromAssets({ ...MINIMAL }), Error);
    assertEquals(error.message.includes("pipelineConfig"), false, error.message);
  });
});

Deno.test("gemma4 pipelineConfig: 配布形ミラーの宣言がこのパーサを素通りする", () => {
  const text = readMirror();
  if (text === undefined) {
    console.warn(
      "[karume] models/karume-gemma4-e2b/ が無いため配布形の宣言の門を SKIP する。" +
        "生成: cd tools/export-recipes && uv run python dist.py --pipeline gemma4",
    );
    return;
  }
  const manifest = parseManifest(text);
  const entry = manifest.models[manifest.defaultModel];
  const config = parseGemma4PipelineConfig(entry.pipelineConfig);
  // 推奨サンプラは**上流の宣言そのもの**（写経していれば値が動く）。
  assertEquals(config.sampler, RECOMMENDED, "配布形が宣言する sampler の既定");
  assertEquals(config.chunkLength, SHIPPED_CHUNK_LENGTH);
  // capacity / chunkLength は実行時ノブで、配布形が宣言するのはその**既定**である。
  assertEquals(config.capacity, SHIPPED_CAPACITY);
  // 位置上限は上流 `max_position_embeddings`（表の行数ではない — 表は配布形から外れた）。
  assertEquals(config.maxPosition, SHIPPED_MAX_POSITION);
  // RoPE のパラメータは上流 config の実値そのもの（写し損ねると回転が別物になる）。
  assertEquals(config.rope, ROPE, "配布形が宣言する rope");
});
