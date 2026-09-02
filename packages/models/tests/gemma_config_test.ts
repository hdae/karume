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

/** 受理される最小形（3 つの数だけ — `sampler` は optional）。 */
const MINIMAL = { chunkLength: 32, maxPosition: 1024, capacity: 640 } as const;
/** 配布形ミラーが宣言する capacity（= 焼き込んだ RoPE 表の行数 `maxPosition`）。 */
const SHIPPED_CAPACITY = 1024;

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

Deno.test("gemma4 pipelineConfig: 3 つの数と optional な sampler を読む", () => {
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
  await t.step("sampler は欄ごと無ければ通る（= 低層の既定 = greedy）", () => {
    assertEquals(parseGemma4PipelineConfig(MINIMAL).sampler, undefined);
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
      () => parseGemma4PipelineConfig({ chunkLength: 64, capacity: 32, maxPosition: 1024 }),
      Error,
      "capacity 32 を超えた",
    );
  });
  await t.step("容量いっぱいの会話が位置表の外を引く", () => {
    assertThrows(
      () => parseGemma4PipelineConfig({ chunkLength: 32, capacity: 2048, maxPosition: 1024 }),
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
  assertEquals(config.chunkLength, MINIMAL.chunkLength);
  // 配布形の capacity は RoPE 表の上限まで引き上げ済み（2026-09-02・recipes gemma4/distribution.py の
  // GEMMA4_CAPACITY）。合成の MINIMAL（640）は「表の内側の任意値」で、配布形の宣言とは別物。
  assertEquals(config.capacity, SHIPPED_CAPACITY);
  assertEquals(config.maxPosition, MINIMAL.maxPosition);
});
