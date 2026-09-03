// gemma4 の `pipelineConfig` スキーマ（`src/gemma/config.ts`）の門 —— GPU も資産も要らない。
//
// hub は `pipelineConfig` を素通しするので、形の正本は models 側のこのパーサ 1 本しかない。
// ここで押さえるのは 4 つ:
//
//  ① **欄の受理集合**（未知キー・欠落・型・値域）— 綴り違いが黙って既定へ縮退すると、配布者の
//     宣言と実行が食い違ったまま気づけない
//  ② **数の関係**（`chunkLength ≤ maxChunkLength` と `chunkLength ≤ capacity ≤ maxPosition`）—
//     容量いっぱいの会話は位置 `capacity - 1` まで進むので、位置上限を超える宣言は**長い会話で
//     だけ**落ちる。`maxChunkLength` は記号 `M` の trace 範囲の上端で、資産からは読めない
//     （IR の `symbols` は名前の列だけ）ので宣言が唯一の出どころ
//  ③ **焼く側との一致** — 手元に配布形ミラーがあれば、その `karume.json` の宣言がこのパーサを
//     素通りし、推奨サンプラが上流 `generation_config.json` の値そのものであること
//     （ADR 0083 決定 7 の「既定値は配布形が宣言する」の実測）
//  ④ **グラフ宣言との突合**（`rope.<層種>.headDim` = `rope_*` 入力の最終次元）— 宣言だけが
//     正しくてもグラフと食い違えば表の幅が違う。落ちる位置を初 `run` から admission へ引き戻す
//
// ③ の資産（`models/karume-gemma4-e2b/`）はリポジトリ管理外なので、無い環境では**その 1 本だけ**
// を明示 SKIP する（①② は常に走る）。

import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { parseManifest } from "@karume/hub";
import { type Gemma4PipelineConfig, parseGemma4PipelineConfig } from "../src/gemma/config.ts";
import { assertRopeInputShapes, Gemma4Pipeline } from "../src/gemma/pipeline.ts";
import type { GenerationGraph } from "../src/generation/program.ts";
import { stubModel } from "./helpers/stub-model.ts";

/**
 * 受理される最小形（4 つの数 + `rope` — `sampler` だけが optional）。
 *
 * `rope` は Gemma 4 E2B の実値（sliding は default rope・full は partial rotary 0.25 の
 * proportional）。表は配布形に無く、cos / sin はこの宣言からホストが作る。
 */
const ROPE = {
  sliding_attention: { theta: 10000, headDim: 256, rotaryDim: 256 },
  full_attention: { theta: 1000000, headDim: 512, rotaryDim: 128 },
} as const;
const MINIMAL = {
  chunkLength: 32,
  maxChunkLength: 128,
  maxPosition: 1024,
  capacity: 640,
  rope: ROPE,
} as const;
/** 配布形ミラーが宣言する既定（capacity / chunkLength はどちらも実行時ノブの**既定**）。 */
const SHIPPED_CHUNK_LENGTH = 768;
/** 記号 `M` の trace 上限（焼く側の `gemma4.export.SYM_MAX` — 既定と同値だが別の事実）。 */
const SHIPPED_MAX_CHUNK_LENGTH = 768;
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
  await t.step("maxChunkLength は必須（無いと trace 範囲の外を誰も落とせない）", () => {
    // 欠落を「上限なし」として通すと、上限 768 の資産に chunkLength 1024 を渡す形が例外なしで
    // 走る（門を入れる前の実測 — 2026-09-03）。宣言が唯一の出どころなので欠落は fail loudly。
    const { maxChunkLength: _dropped, ...rest } = MINIMAL;
    assertThrows(
      () => parseGemma4PipelineConfig(rest),
      Error,
      "pipelineConfig.maxChunkLength: 無い",
    );
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

Deno.test("gemma4 pipelineConfig: chunkLength ≤ maxChunkLength / capacity ≤ maxPosition", async (t) => {
  await t.step("既定の chunkLength が記号 M の trace 範囲の外", () => {
    assertThrows(
      () => parseGemma4PipelineConfig({ ...MINIMAL, chunkLength: 129 }),
      Error,
      "maxChunkLength 128 を超えた",
    );
  });
  await t.step("上限ちょうどは通る（門が広すぎないことの対）", () => {
    const config = parseGemma4PipelineConfig({ ...MINIMAL, chunkLength: 128 });
    assertEquals(config.chunkLength, 128);
  });
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

// ---- グラフ宣言との突合（RoPE の幅）----------------------------------------
//
// `pipelineConfig.rope.<層種>.headDim` は**ホストが作る表の幅そのもの**なので、グラフ入力の
// 宣言（`[1, M, headDim]`）と食い違うとホストは最後まで通り、落ちるのは最初の `run` になる
// （3.7GiB のロードの**後**・文言は「要素数が shape と合わない」= どちらの宣言が誤りか読めない）。
// 突合は家族 admission（`admitGemma4` = 重み shard を 1 バイトも取る前）が通す。焼く側の鏡像は
// `tools/export-recipes/gemma4/export_decode.py` の `assert_rope_inputs`。

/** 実配布形と同じ宣言（sliding 256 / full 512）。`patch` で 1 本だけ壊す。 */
const ropeGraph = (
  patch: Readonly<Record<string, readonly (number | string)[]>> = {},
): GenerationGraph =>
  stubModel({
    symbols: ["C", "M"],
    inputs: [
      { name: "input_ids", shape: [1, "M"] },
      { name: "last_row", shape: [1] },
      { name: "per_layer_inputs", shape: [1, "M", 35, 256] },
      { name: "rope_sliding_attention_cos", shape: patch.slidingCos ?? [1, "M", 256] },
      { name: "rope_sliding_attention_sin", shape: patch.slidingSin ?? [1, "M", 256] },
      { name: "rope_full_attention_cos", shape: patch.fullCos ?? [1, "M", 512] },
      { name: "rope_full_attention_sin", shape: patch.fullSin ?? [1, "M", 512] },
    ].filter((input) => input.shape.length > 0),
    outputs: ["logits"],
    values: { logits: [1, 1, 262144] },
  }).graph;

Deno.test("gemma4 rope 突合: 宣言どおりのグラフは通り、幅の食い違いは名指しで落ちる", async (t) => {
  const config = parseGemma4PipelineConfig(MINIMAL);

  await t.step("宣言どおり（陰性対照 — 門が全部落として緑になっていないこと）", () => {
    assertRopeInputShapes(ropeGraph(), config);
  });

  await t.step("層種別の取り違え（full の席に sliding の幅）", () => {
    // exporter 側 `rope.py` が `head_dim` / `global_head_dim` の分岐で自認している間違い方。
    const error = assertThrows(
      () => assertRopeInputShapes(ropeGraph({ fullCos: [1, "M", 256] }), config),
      Error,
      "rope_full_attention_cos",
    );
    assert(error.message.includes("headDim 512"), error.message);
  });

  await t.step("配布形が headDim を偽った宣言（同じ食い違いの逆側）", () => {
    const lying = parseGemma4PipelineConfig({
      ...MINIMAL,
      rope: { ...ROPE, sliding_attention: { theta: 10000, headDim: 128, rotaryDim: 128 } },
    });
    assertThrows(
      () => assertRopeInputShapes(ropeGraph(), lying),
      Error,
      "rope_sliding_attention_cos",
    );
  });

  await t.step("次元の数が違う宣言（`[1, M, headDim]` でない）", () => {
    assertThrows(
      () => assertRopeInputShapes(ropeGraph({ fullSin: [1, 512] }), config),
      Error,
      "rope_full_attention_sin",
    );
  });

  await t.step("RoPE がホスト供給でない資産（入力そのものが無い）", () => {
    assertThrows(
      () => assertRopeInputShapes(ropeGraph({ slidingSin: [] }), config),
      Error,
      "グラフ入力 'rope_sliding_attention_sin' が無い",
    );
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
  // 記号 `M` の trace 上限は資産に残らない（IR の symbols は名前だけ）ので、配布形の宣言が
  // 唯一の出どころ。焼く側（`gemma4.export.SYM_MAX`）との同値は recipe 側の pytest が見る。
  assertEquals(config.maxChunkLength, SHIPPED_MAX_CHUNK_LENGTH);
  // capacity / chunkLength は実行時ノブで、配布形が宣言するのはその**既定**である。
  assertEquals(config.capacity, SHIPPED_CAPACITY);
  // 位置上限は上流 `max_position_embeddings`（表の行数ではない — 表は配布形から外れた）。
  assertEquals(config.maxPosition, SHIPPED_MAX_POSITION);
  // RoPE のパラメータは上流 config の実値そのもの（写し損ねると回転が別物になる）。
  assertEquals(config.rope, ROPE, "配布形が宣言する rope");
});
