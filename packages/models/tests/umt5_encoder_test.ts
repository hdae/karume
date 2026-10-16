/**
 * `Umt5Encoder`（`@karume/models/umt5`）の**構築の門**と umT5 のグラフ宣言の取り決め（`umt5GraphContract`）。
 * GPU も実資産も要らない範囲だけを見る（実 GPU の符号化は `e2e_umt5_encoder_test.ts`）。
 *
 * 押さえるのは 3 点:
 *
 * ① **家族 admission**（pipeline 名 / major・`pipelineConfig`・quant・トークナイザ資産の宣言・quant の
 *    `session`・グラフ宣言）は **GPU に触る前**に落ちる。全量面（`fromAssets`）では容器を揃えたまま、その違反の
 *    文言で落ちることを見る（容器が無い形は受け口の診断で落ちる — 対照）。
 * ② 取得面（`fromPretrained` — 疑似 HF）では、admission の違反は **重みの part を 1 本も取らずに**落ちる
 *    （取ったのは manifest と part 0〈descriptor〉だけ）。正の経路は「トークナイザ資産の解析」で止まる — admission と
 *    重みの part の取得・資産の取得を通り終えた（`wan_components_test.ts` と同じ「次の段の文言で通過を見る」形）。
 *    GPU は共有の模擬 GPU を渡し、アダプタにも触らない。
 * ③ 取得元に既定は無い（`UMT5_SOURCES` は呼び手が渡すオプトインの対応表で、既定ではない — ADR 0073。ref 欠落時の
 *    案内の文言は `repo_ref_test.ts`）。
 */

import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { type FileRef, parseManifest } from "@karume/hub";
import { ModelInputError } from "../src/errors.ts";
import type { GraphOwner } from "../src/hub/components.ts";
import { Umt5Encoder, type Umt5EncoderFromPretrainedOptions } from "../src/umt5/encoder.ts";
import { umt5GraphContract } from "../src/umt5/graph-contract.ts";
import { UMT5_TOKENIZER_FORMAT } from "../src/umt5/tokenizer.ts";
import type { ModelInput, TensorInput } from "../../runtime/tests/helpers/container-write.ts";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";
import {
  parseIrDeclarationValue,
  partAssets,
  writeContainer,
} from "./helpers/container-fixture.ts";
import {
  fileRef,
  HUB_URL,
  MANIFEST_PATH,
  REPO,
  serveContainer,
  serveRepos,
  SHA,
} from "./helpers/container-loading-fixture.ts";
import { MemoryCacheStorage } from "./helpers/memory-cache.ts";

/** umT5 の語彙埋め込みの形（語彙 2 行 × 幅 32 — 幅は宣言から読まれる）。 */
const EMBED_ROWS = 2;
const HIDDEN = 32;

const f32Bytes = (values: readonly number[]): Uint8Array<ArrayBuffer> =>
  new Uint8Array(Float32Array.from(values).buffer);

/** 語彙埋め込みの格納（i8 = per-channel・i4 = 32 要素ごとの group — 格納の門の対照）。 */
const embedWeight = (storage: "i8" | "i4"): TensorInput => {
  const elements = EMBED_ROWS * HIDDEN;
  if (storage === "i8") {
    return {
      graph: "text_encoder",
      initializer: "embed",
      bytes: new Uint8Array(elements).fill(1),
      encoding: {
        codec: "int8-sym",
        groupSize: HIDDEN,
        scale: { bytes: f32Bytes(Array(EMBED_ROWS).fill(0.5)), dtype: "f32" },
      },
    };
  }
  const groupSize = 32;
  return {
    graph: "text_encoder",
    initializer: "embed",
    bytes: new Uint8Array(elements / 2).fill(1),
    encoding: {
      codec: "int4-sym-g",
      groupSize,
      scale: {
        bytes: f32Bytes(Array(EMBED_ROWS * (HIDDEN / groupSize)).fill(0.5)),
        dtype: "f32",
      },
    },
  };
};

/**
 * umT5 の宣言（取り決めの正常形 — 入力 2 本の i32・記号 L・出力 `[1, L, 32]`・i8 の重み）。実行はしない。
 */
const umt5Input = (storage: "i8" | "i4" = "i8"): ModelInput => ({
  graphs: {
    text_encoder: parseIrDeclarationValue({
      format: "karume-ir",
      version: 2,
      requires: { ops: ["embedding"] },
      symbols: ["L"],
      inputs: [
        { name: "input_ids", dtype: "i32", shape: [1, "L"] },
        { name: "relative_position_buckets", dtype: "i32", shape: ["L", "L"] },
      ],
      outputs: ["last_hidden_state"],
      initializers: { embed: {} },
      values: {
        embed: { dtype: "f32", shape: [EMBED_ROWS, HIDDEN] },
        last_hidden_state: { dtype: "f32", shape: [1, "L", HIDDEN] },
      },
      states: {},
      nodes: [{
        op: "embedding",
        ins: ["embed", "input_ids"],
        outs: ["last_hidden_state"],
        attrs: { padding_idx: -1 },
      }],
    }),
  },
  consts: [],
  weights: [embedWeight(storage)],
  assets: [],
  provenance: { license: "test" },
});

/** 最小のトークナイザ資産（形式 `karume-umt5-tokenizer/1` — 語彙 6 行・上限 512）。 */
const tokenizerJson = (patch: Record<string, unknown> = {}): Uint8Array<ArrayBuffer> =>
  new TextEncoder().encode(JSON.stringify({
    format: UMT5_TOKENIZER_FORMAT,
    maxLength: 512,
    unkId: 2,
    eosId: 1,
    addedTokens: [["<pad>", 0], ["</s>", 1], ["<s>", 2], ["<unk>", 3]],
    space: [[0x09, 0x0d], [0x20, 0x20]],
    vocabText: "<pad>\n</s>\n<s>\n<unk>\n▁cat\n▁",
    scores: [0, 0, 0, 0, -1, -2],
    ...patch,
  }));

/** トークナイザ資産の宣言（`fromAssets` の門は中身を読まないので 3 点セットは綴りだけ合わせる）。 */
const TOKENIZER_REF = { path: "xxl/tokenizer/tokenizer.json", size: 8, sha256: "f".repeat(64) };

/**
 * 配布形 `karume-umt5-xxl` の骨格（`models/karume-umt5-xxl/karume.json` と同じ欄 — 検査に要る分だけ）。`patch` は
 * `models["xxl"]` の中身を上書きする。
 */
const manifestOf = (
  textEncoder: Record<string, unknown>,
  patch: Record<string, unknown> = {},
) =>
  parseManifest(JSON.stringify({
    format: "karume/5",
    generator: "karume/0.1.0",
    defaultModel: "xxl",
    models: {
      xxl: {
        pipeline: "umt5-encoder/1",
        weights: { text_encoder: { i8: textEncoder } },
        assets: { tokenizer: TOKENIZER_REF },
        quants: { i8: { weights: { text_encoder: "i8" }, session: {} } },
        defaultQuant: "i8",
        pipelineConfig: {},
        ...patch,
      },
    },
  }));

const written = await writeContainer(umt5Input());
const writtenI4 = await writeContainer(umt5Input("i4"));
/** manifest の宣言（全量面の門は part の実体を読まないので、descriptor と part の 3 点セットだけ）。 */
const DECLARED = {
  container: {
    descriptor: written.descriptor,
    parts: await Promise.all(
      written.parts.map((bytes, index) => fileRef(`xxl/text_encoder/model.i8-${index}.krm`, bytes)),
    ),
  },
};
const COMPONENTS = partAssets("text_encoder", written);

/** 共有の模擬 GPU（アダプタの limits を読みに行かない・Session は張れない — GPU に触ったら別の文言で落ちる）。 */
const fakeGpu = () => fakeGpuContext(fakeDevice());

describe("fromAssets: 家族 admission は GPU に触る前に落ちる（容器は揃えたまま）", () => {
  const rejected: readonly [string, Record<string, unknown>, string][] = [
    ["pipeline の名前違い", { pipeline: "wan/1" }, "manifest の pipeline が 'wan/1'"],
    ["未知の major", { pipeline: "umt5-encoder/2" }, "major に未対応"],
    ["pipelineConfig の欄", { pipelineConfig: { maxLength: 512 } }, "未知キー 'maxLength'"],
    [
      "トークナイザ資産の宣言が無い（取り込み由来のミラーの形）",
      { assets: {} },
      "assets にトークナイザ資産 'tokenizer' が無い",
    ],
    [
      "quant の session に実行ノブ",
      { quants: { i8: { weights: { text_encoder: "i8" }, session: { linearCompute: "a8" } } } },
      "session.linearComputeは未対応",
    ],
  ];
  for (const [label, patch, message] of rejected) {
    it(label, async () => {
      const error = await assertRejects(
        () =>
          Umt5Encoder.fromAssets(
            {
              manifest: manifestOf(DECLARED, patch),
              assets: { ...COMPONENTS, tokenizer: tokenizerJson() },
            },
            { gpu: fakeGpu() },
          ),
        Error,
        message,
      );
      assert(!(error instanceof ModelInputError), `資産の齟齬を入力起因にしない: ${label}`);
    });
  }

  it("model / quant の綴り違いは利用可能な一覧つきで落ちる", async () => {
    const input = { manifest: manifestOf(DECLARED), assets: COMPONENTS };
    await assertRejects(
      () => Umt5Encoder.fromAssets(input, { model: "base", gpu: fakeGpu() }),
      Error,
      "model 'base' は manifest に無い（利用可能: xxl）",
    );
    await assertRejects(
      () => Umt5Encoder.fromAssets(input, { quant: "f16", gpu: fakeGpu() }),
      Error,
      "quant 'f16' は manifest に無い（利用可能: i8）",
    );
  });

  it("グラフ宣言の格納が i8 / f32 でない容器（i4）は落ちる", async () => {
    await assertRejects(
      () =>
        Umt5Encoder.fromAssets(
          {
            manifest: manifestOf(DECLARED),
            assets: { ...partAssets("text_encoder", writtenI4), tokenizer: tokenizerJson() },
          },
          { gpu: fakeGpu() },
        ),
      Error,
      "text_encoder の initializer 'embed' の格納 int4-sym-g は受けない",
    );
  });

  it("対照: 容器が無ければ manifest の門ではなく受け口の診断で落ちる", async () => {
    await assertRejects(
      () => Umt5Encoder.fromAssets({ manifest: manifestOf(DECLARED), assets: {} }),
      Error,
      "部品 'text_encoder' の容器が無い",
    );
  });
});

describe("fromAssets: トークナイザ資産の中身は GPU に触る前に見る", () => {
  const build = (tokenizer: Uint8Array<ArrayBuffer>) =>
    Umt5Encoder.fromAssets(
      { manifest: manifestOf(DECLARED), assets: { ...COMPONENTS, tokenizer } },
      { gpu: fakeGpu() },
    );

  it("宣言はあるが資産のバイト列が無ければ、揃っているキーを添えて落ちる", async () => {
    await assertRejects(
      () =>
        Umt5Encoder.fromAssets(
          { manifest: manifestOf(DECLARED), assets: COMPONENTS },
          { gpu: fakeGpu() },
        ),
      Error,
      "資産 'tokenizer' が無い",
    );
  });

  it("Wan のトークナイザ資産（前処理の表つきの別形式）は読まない", async () => {
    await assertRejects(
      () => build(tokenizerJson({ format: "karume-wan-umt5-tokenizer/1" })),
      Error,
      "形式 karume-wan-umt5-tokenizer/1 を読めない",
    );
  });

  it("最大長がグラフの有効長の上限 512 と違う資産は落ちる", async () => {
    await assertRejects(
      () => build(tokenizerJson({ maxLength: 256 })),
      Error,
      "maxLength 256 が umT5 の有効長の上限 512 と違う",
    );
  });
});

// ---- 取得面（疑似 HF）---------------------------------------------------------

/** 実体を持つ part（長さ 0 の part は取得の対象外 — hub も取らない）。 */
const nonEmpty = (parts: readonly FileRef[]): readonly FileRef[] =>
  parts.filter((part) => part.size > 0);

/** 疑似 HF に載せた umT5 の配布形（`patch` は `models["test"]` の欄を差し替える）。 */
const serveUmt5 = async (patch: Record<string, unknown>, tokenizer: Uint8Array<ArrayBuffer>) => {
  const served = await serveContainer("xxl/text_encoder/model.i8", umt5Input(), {
    partBytes: 1024,
    blockBytes: 512,
  });
  const tokenizerRef = await fileRef("xxl/tokenizer/tokenizer.json", tokenizer);
  const mock = serveRepos([{
    repo: REPO,
    models: {
      test: {
        pipeline: "umt5-encoder/1",
        weights: { text_encoder: { i8: served.entry } },
        assets: { tokenizer: tokenizerRef },
        quants: { i8: { weights: { text_encoder: "i8" }, session: {} } },
        defaultQuant: "i8",
        pipelineConfig: {},
        ...patch,
      },
    },
    files: [...served.files, [tokenizerRef.path, tokenizer]],
  }]);
  const load = (options: Umt5EncoderFromPretrainedOptions = {}) =>
    Umt5Encoder.fromPretrained({ repo: REPO, revision: SHA, hubUrl: HUB_URL }, {
      gpu: fakeGpu(),
      fetch: mock.fetch,
      caches: new MemoryCacheStorage(),
      ...options,
    });
  return { load, paths: mock.paths, parts: served.parts, tokenizerRef };
};

describe("fromPretrained: admission の違反は重みの part を 1 本も取らずに落ちる", () => {
  const rejected: readonly [string, Record<string, unknown>, string][] = [
    [
      "トークナイザ資産の宣言が無い",
      { assets: {} },
      "assets にトークナイザ資産 'tokenizer' が無い",
    ],
    ["pipeline の名前違い", { pipeline: "siglip2/1" }, "manifest の pipeline が 'siglip2/1'"],
    ["未知の major", { pipeline: "umt5-encoder/2" }, "major に未対応"],
  ];
  for (const [label, patch, message] of rejected) {
    it(label, async () => {
      const rig = await serveUmt5(patch, tokenizerJson());
      await assertRejects(() => rig.load(), Error, message);
      // 取ったのは manifest と part 0（descriptor）だけ — 重みの part とトークナイザ資産は取っていない。
      assertEquals(rig.paths, [MANIFEST_PATH, rig.parts[0].path]);
    });
  }

  it("正の経路: admission を通り、重みの part とトークナイザ資産を取り終えてから資産の解析で止まる", async () => {
    // 観測の仕掛け: トークナイザ資産を形式欄の無い JSON にする。構築は admission → 重みの part の取得 → 資産の
    // 取得 → 資産の解析の順なので、この文言で落ちれば手前の段は全部通っている（GPU には触らない）。
    const rig = await serveUmt5({}, new TextEncoder().encode("{}"));
    await assertRejects(() => rig.load(), Error, "tokenizer.format: 文字列でない");
    for (const part of nonEmpty(rig.parts)) {
      assert(rig.paths.includes(part.path), `重みの part ${part.path} を取っていない`);
    }
    assert(rig.paths.includes(rig.tokenizerRef.path), "トークナイザ資産を取っていない");
  });
});

describe("fromPretrained: 取得元に既定は無い", () => {
  it("ref を省くと（型検査の外の JS の呼び出し）、repo が必須であることを言って落ちる", async () => {
    await assertRejects(
      () => Reflect.apply(Umt5Encoder.fromPretrained, Umt5Encoder, []),
      Error,
      "Umt5Encoder.fromPretrained: repo が必須（取得元に既定は無い）",
    );
  });
});

// ---- グラフ宣言の取り決め（門を直接叩く）-------------------------------------

/** 宣言だけの `GraphOwner`（出力の形だけを差し替える — 重みは i8 の 1 本）。 */
const ownerOf = (outputShape: readonly (number | string)[]): GraphOwner => ({
  graph: {
    format: "karume-ir",
    version: 2,
    requires: { ops: [] },
    symbols: ["L", "W"],
    inputs: [
      { name: "input_ids", dtype: "i32", shape: [1, "L"] },
      { name: "relative_position_buckets", dtype: "i32", shape: ["L", "L"] },
    ],
    outputs: ["out"],
    initializers: { w: { storage: { codec: "int8-sym" } } },
    values: { out: { dtype: "f32", shape: [...outputShape] } },
    states: {},
    nodes: [],
  },
});

describe("umt5GraphContract: 出力の幅", () => {
  it("幅を渡さなければ宣言の静的な幅を読む（単体の encoder）", () => {
    assertEquals(umt5GraphContract(ownerOf([1, "L", 4096]), "Umt5Encoder"), {
      output: "out",
      hiddenSize: 4096,
    });
  });

  it("宣言の幅が記号次元なら読めないと言って落ちる（ホストは幅を宣言から読む）", () => {
    assertThrows(
      () => umt5GraphContract(ownerOf([1, "L", "W"]), "Umt5Encoder"),
      Error,
      "軸 2 が静的な正の整数でない",
    );
  });

  it("幅を渡せば宣言の幅をその値と照合する（Wan は DiT の文脈の幅を渡す）", () => {
    assertEquals(umt5GraphContract(ownerOf([1, "L", 4096]), "WanPipeline", 4096).hiddenSize, 4096);
    assertThrows(
      () => umt5GraphContract(ownerOf([1, "L", 2048]), "WanPipeline", 4096),
      Error,
      "出力 'out' の形 [1, L, 2048] がホストの組む [1, L, 4096] と違う",
    );
  });
});
