// 実重み Gemma 4 E2B の**製品グラフ**（PLE 外出し + 最終行 logits 出口）の検収門 — 段 1b。
//
// 検収するのは ADR [0083](../../../docs/decisions/0083-generation-api-surface.md) 決定 6 と ADR
// [0085](../../../docs/decisions/0085-ple-host-gather.md) を**同じ再 export に載せた**製品形
// （案 α）で、既存 2 系列（logits opt-in / token-only）は検収 fixture として残っている。
//
// 門は 4 本:
//
// ① 形の前提（入力 7 本 = input_ids / **per_layer_inputs** / **rope 4 本** / last_row・出力 1 本 =
//    最終行 logits `[1,1,V]`・**argmax はグラフに無い**・PLE 表を引く embedding が無い・
//    states 30 スロットは既存 2 系列と同一）
// ② **PLE 逆量子化のビット一致**（ADR 0085 決定 4）: ホスト loader の gather が、台本が torch の
//    35 表経路で採った `ple.probe.safetensors`（= PLE をグラフに残していたら `embedding` +
//    直後の `mul` が出していた値そのもの）と**厳密一致**する。GPU を要さない。
//    併せて「触った shard だけ遅延ロード + LRU」（決定 3）を実測で見る。
// ③ **交差 parity**: ホスト PLE gather + ホスト `argmax(logits)` の greedy ループが、logits
//    opt-in 系列の `greedy.<case>.safetensors`（torch full re-forward の期待列）と 3 ケース ×
//    K=16 で厳密一致する。PLE を外に出しても・出口から argmax を外しても**機能が不変**である
//    ことの証明で、割れたら PLE gather / 逆量子化 / 行選択のどれかの誤り。
// ④ **出所（provenance）の束縛**: 製品系列が書く `reference.json` が、元チェックポイントの
//    指紋と③で流用する golden 1 本ずつの digest を持っていること（token-only 系列と同文 —
//    片方だけ古い組み合わせでも③は緑になれてしまう）。
//
// ホスト側の greedy ループは**この門が自前で組む**。`src/generation/greedy.ts` はグラフ側 argmax
// 出力を読む形（token-only 出口）で、ホスト sampling を持たない MUST があるため通せない
// （製品面の `GenerationSequence` は段 2〜3 の担当 — ADR 0083 決定 1〜8）。chunk の割り方だけは
// `planPrefillChunks` を共有する（割り方の誤りを期待値側へ写さないため、割り方**以外**は書き下す）。
//
// ## 資産
//
// `outputs/series/gemma4-e2b-product/`（コンテナ 1.51GiB + PLE sidecar 2.22GiB — リポジトリ
// 管理外）と `outputs/series/gemma4-e2b-decode/`（期待列の正本）。製品系列が無い環境では
// **明示 SKIP** し、自系列があるのに正本が欠けている形は SKIP でなく **FAIL** にする。

import { assert, assertEquals } from "@std/assert";
import {
  acquireGpu,
  parseSafetensors,
  type PreparedModel,
  prepareModel,
  type SafetensorsFile,
  type Tensor,
} from "@karume/runtime";
import {
  createGemma4Ple,
  defaultGemma4PleResidentBytes,
  type Gemma4Ple,
  type Gemma4PleIndex,
  gemma4PleShardBytes,
  parseGemma4PleIndex,
} from "../src/gemma/ple.ts";
import { gemma4RopeInputNames, gemma4RopeInputs, type Gemma4RopeSpec } from "../src/gemma/rope.ts";
import { planPrefillChunks } from "../src/generation/greedy.ts";
import {
  modelPresent,
  readShard,
  resolveShards,
  streamShards,
} from "../../runtime/tests/helpers/shard-files.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const PRODUCT_ROOT = new URL("../../../outputs/series/gemma4-e2b-product/", import.meta.url);
const GOLDEN_ROOT = new URL("../../../outputs/series/gemma4-e2b-decode/", import.meta.url);
const MODEL_FILE = "model.safetensors";
const GREEDY_PREFIX = "greedy.";
const SUFFIX = ".safetensors";

/** PLE sidecar の索引・逆量子化参照（綴りの正本は `gemma4/export_product.py`）。 */
const PLE_INDEX_FILE = "ple.json";
const PLE_PROBE_FILE = "ple.probe.safetensors";
const PROBE_TOKENS_KEY = "tokens";
const PROBE_INPUTS_KEY = "per_layer_inputs";

/** 出所記録のファイル名と版（綴りの正本は `gemma4/provenance.py`）。 */
const REFERENCE_FILE = "reference.json";
const REFERENCE_SCHEMA = 1;

/** SKIP 時にそのまま貼れる生成コマンド。 */
const GENERATE_COMMAND =
  "cd tools/export-recipes && uv run --with 'transformers==5.14.1' python -m gemma4.export_product" +
  "（期待列の正本は … python -m gemma4.export_decode）";

/** 期待列の正本 = logits opt-in 系列の greedy golden（正典は export_decode.GREEDY_CASES）。 */
const EXPECTED_CASES = ["capital-en", "capital-ja", "context-en"] as const;
const GREEDY_STEPS = 16;

/** 実行条件は既存 2 系列の門と同値（同じ資産世代の裁定をそのまま使う）。 */
const CHUNK_LENGTH = 768;
const CAPACITY_SYMBOL = "C";
const CAPACITY = 4096;
const MAX_POSITION = 131072;

/** RoPE のパラメータ（配布形の宣言と同じ値 — 表は資産に無く cos / sin はホストが作る）。 */
const ROPE: Gemma4RopeSpec = {
  sliding_attention: { theta: 10000, headDim: 256, rotaryDim: 256 },
  full_attention: { theta: 1000000, headDim: 512, rotaryDim: 128 },
};

/** グラフ入力の名前（正本は `export_decode` / `export_product` の定数）。 */
const INPUT_IDS = "input_ids";
const PER_LAYER_INPUTS = "per_layer_inputs";
const LAST_ROW = "last_row";

/** 実資産の形（config の `num_hidden_layers` / `hidden_size_per_layer_input` / `vocab_size`）。 */
const LAYERS = 35;
const PLE_DIM = 256;
const VOCAB = 262144;
/** states スロット本数（所有層 15 × k/v — 既存 2 系列と同一）。 */
const SLOTS = 30;

/**
 * 常駐した圧縮重みの上限（ADR 0085 の主張そのもの — PLE を外して 3.70 → 1.51 GiB）。
 * decode / token-only 系列は同じ観測で 3.0e9 を**超える**ので、この線は両者を分ける。
 */
const RESIDENT_CEILING = 2_000_000_000;

const exists = (url: URL): boolean => {
  try {
    return Deno.statSync(url).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

const MODEL_PRESENT = modelPresent(new URL(MODEL_FILE, PRODUCT_ROOT));
const SIDECAR_PRESENT = exists(new URL(PLE_INDEX_FILE, PRODUCT_ROOT));
const GOLDENS_PRESENT = EXPECTED_CASES.every((name) =>
  exists(new URL(`${GREEDY_PREFIX}${name}${SUFFIX}`, GOLDEN_ROOT))
);
const AVAILABLE = MODEL_PRESENT && SIDECAR_PRESENT && GOLDENS_PRESENT;

if (!MODEL_PRESENT) {
  console.warn(
    `[karume] 製品系列（${PRODUCT_ROOT.pathname}）が無いため Gemma 4 E2B 製品グラフ検収を ` +
      `SKIP する。生成: ${GENERATE_COMMAND}`,
  );
}

/**
 * 依存資産の完全性（欠落を SKIP に畳まない）。この門は 2 系列 + sidecar に依存する:
 * 製品系列の model（自系列 — 無ければ「未生成」で SKIP が正しい）と、同じ系列の PLE sidecar、
 * そして logits opt-in 系列の greedy golden（期待列の正本）。**自系列があるのに片割れが
 * 欠けている**のは未生成でなく欠損なので、SKIP でなく FAIL にする。
 */
Deno.test({
  name: "Gemma 4 E2B 製品資産: PLE sidecar と期待列の正本が揃っている",
  ignore: !MODEL_PRESENT,
  fn: () => {
    assert(
      SIDECAR_PRESENT,
      `${PLE_INDEX_FILE} が ${PRODUCT_ROOT.pathname} に無い（コンテナはあるのに PLE sidecar が` +
        `欠けている — この資産では per_layer_inputs を作れない）`,
    );
    assert(
      exists(new URL(PLE_PROBE_FILE, PRODUCT_ROOT)),
      `${PLE_PROBE_FILE} が ${PRODUCT_ROOT.pathname} に無い（逆量子化ビット一致の参照が欠けている）`,
    );
    for (const name of EXPECTED_CASES) {
      assert(
        exists(new URL(`${GREEDY_PREFIX}${name}${SUFFIX}`, GOLDEN_ROOT)),
        `${GREEDY_PREFIX}${name}${SUFFIX} が ${GOLDEN_ROOT.pathname} に無い` +
          `（製品系列はあるのに期待列の正本が欠けている）`,
      );
    }
  },
});

/** 系列ディレクトリの名前（記録が名乗る `series` と突き合わせる側）。 */
const seriesName = (root: URL): string =>
  root.pathname.split("/").filter((part) => part !== "").at(-1) ?? "";

/** JSON の 1 段を「キー → 未検査の値」へ落とす（未知の形は明確な文言で落とす）。 */
const objectAt = (value: unknown, where: string): Map<string, unknown> => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${REFERENCE_FILE}: ${where} がオブジェクトでない`);
  }
  return new Map(Object.entries(value));
};

const stringAt = (entries: Map<string, unknown>, key: string, where: string): string => {
  const value = entries.get(key);
  if (typeof value !== "string" || value === "") {
    throw new Error(`${REFERENCE_FILE}: ${where}.${key} が非空の文字列でない`);
  }
  return value;
};

/** `{bytes, sha256}` の 1 件（sha256 は 64 桁の小文字 hex であることまで見る）。 */
const digestAt = (
  entries: Map<string, unknown>,
  key: string,
  where: string,
): { bytes: number; sha256: string } => {
  const digest = objectAt(entries.get(key), `${where}.${key}`);
  const bytes = digest.get("bytes");
  if (typeof bytes !== "number" || !Number.isSafeInteger(bytes) || bytes <= 0) {
    throw new Error(`${REFERENCE_FILE}: ${where}.${key}.bytes が正の整数でない`);
  }
  const sha256 = stringAt(digest, "sha256", `${where}.${key}`);
  if (!/^[0-9a-f]{64}$/.test(sha256)) {
    throw new Error(`${REFERENCE_FILE}: ${where}.${key}.sha256 が 64 桁の hex でない`);
  }
  return { bytes, sha256 };
};

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> => {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return [...digest].map((byte) => byte.toString(16).padStart(2, "0")).join("");
};

/**
 * ④ 出所の束縛。③が流用する golden は「この容器を作った実走がその場で digest を採ったもの」で
 * なければならない — logits opt-in 系列を採り直せば digest が動き、この門が落ちて製品系列の
 * 再 export を強制する（token-only 系列の同名門と同文）。
 */
Deno.test({
  name: "Gemma 4 E2B 製品資産: 出所記録が参照 golden を束ねている",
  ignore: !MODEL_PRESENT,
  fn: async () => {
    const path = new URL(REFERENCE_FILE, PRODUCT_ROOT);
    assert(
      exists(path),
      `${REFERENCE_FILE} が ${PRODUCT_ROOT.pathname} に無い` +
        `（出所記録の無い世代の資産 — 再生成: ${GENERATE_COMMAND}）`,
    );
    const record = objectAt(JSON.parse(await Deno.readTextFile(path)), "根");

    assertEquals(record.get("schema"), REFERENCE_SCHEMA, `${REFERENCE_FILE} の schema`);
    assertEquals(stringAt(record, "series", "根"), seriesName(PRODUCT_ROOT), "記録が名乗る系列");
    const checkpoint = objectAt(record.get("checkpoint"), "checkpoint");
    stringAt(checkpoint, "dir", "checkpoint");
    const fingerprint = objectAt(checkpoint.get("files"), "checkpoint.files");
    assert(fingerprint.size > 0, `${REFERENCE_FILE}: checkpoint.files が空`);
    for (const file of fingerprint.keys()) digestAt(fingerprint, file, "checkpoint.files");

    const reference = objectAt(record.get("reference"), "reference");
    assertEquals(
      stringAt(reference, "series", "reference"),
      seriesName(GOLDEN_ROOT),
      "束ねられた golden 系列（③が読む系列と同じであること）",
    );
    const goldens = objectAt(reference.get("goldens"), "reference.goldens");
    assertEquals(
      [...goldens.keys()].sort(),
      EXPECTED_CASES.map((name) => `${GREEDY_PREFIX}${name}${SUFFIX}`).sort(),
      "束ねられた golden の集合（③が読む 3 本と過不足なく一致）",
    );
    for (const name of EXPECTED_CASES) {
      const file = `${GREEDY_PREFIX}${name}${SUFFIX}`;
      const digest = digestAt(goldens, file, "reference.goldens");
      const bytes = await Deno.readFile(new URL(file, GOLDEN_ROOT));
      assertEquals(bytes.byteLength, digest.bytes, `${file}: byte 数`);
      assertEquals(
        await sha256Hex(bytes),
        digest.sha256,
        `${file}: sha256（製品系列が束ねた golden と別物 — ` +
          `どちらかの系列だけを作り直した組み合わせ。再生成: ${GENERATE_COMMAND}）`,
      );
    }
  },
});

/**
 * ファイル 1 本を `ArrayBuffer` として読む。
 * MUST: view が buffer 全体を覆っているなら slice しない — PLE sidecar は 1 本 758MB 級で、
 * 無条件の `slice` はピークを倍増させる。
 */
const readBuffer = async (root: URL, file: string): Promise<ArrayBuffer> => {
  const bytes = await Deno.readFile(new URL(file, root));
  if (bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength) {
    return bytes.buffer;
  }
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};

const goldenI32 = (file: SafetensorsFile, name: string): Int32Array<ArrayBuffer> => {
  const view = file.tensors.get(name);
  assert(view !== undefined, `golden に '${name}' が無い`);
  assertEquals(view.dtype, "I32", `golden '${name}' の格納 dtype`);
  return new Int32Array(file.buffer, view.byteOffset, view.byteLength / 4);
};

const goldenF32 = (file: SafetensorsFile, name: string): Float32Array<ArrayBuffer> => {
  const view = file.tensors.get(name);
  assert(view !== undefined, `golden に '${name}' が無い`);
  assertEquals(view.dtype, "F32", `golden '${name}' の格納 dtype`);
  return new Float32Array(file.buffer, view.byteOffset, view.byteLength / 4);
};

/** sidecar 全量を常駐させる予算（= 読み直しゼロ）。 */
const allResidentBytes = (index: Gemma4PleIndex): number =>
  index.shards.reduce((sum, shard) => sum + gemma4PleShardBytes(index, shard), 0);

/** 索引を読んで loader を組む（shard の読みは実ファイル — hub は通さない）。 */
const openPle = async (maxResidentBytes: number): Promise<Gemma4Ple> => {
  const index = parseGemma4PleIndex(
    JSON.parse(await Deno.readTextFile(new URL(PLE_INDEX_FILE, PRODUCT_ROOT))),
  );
  return createGemma4Ple({
    index,
    readShard: (file) => readBuffer(PRODUCT_ROOT, file),
    vocabSize: VOCAB,
    maxResidentBytes,
  });
};

const readPleIndex = async () =>
  parseGemma4PleIndex(JSON.parse(await Deno.readTextFile(new URL(PLE_INDEX_FILE, PRODUCT_ROOT))));

/**
 * ② PLE 逆量子化のビット一致（ADR 0085 決定 4）+ 遅延ロード / LRU（決定 3）。
 *
 * 参照は台本が **35 表経路の torch** で採った `ple.probe.safetensors` — PLE をグラフに残して
 * いれば `embedding` + 直後の `mul` がその値を出していた。ホスト側は `f32(i8) × per-row scale`
 * → `× embedScale` を同じ順序で組むので、**1 bit も違ってはならない**。ここが割れると③の
 * token 列 parity も割れる（が、そのときどこが原因かは③からは読めない — 切り分けの線）。
 *
 * GPU を要さない（loader は Web 標準 API だけで閉じている）。
 */
Deno.test({
  name: "Gemma 4 E2B 製品検収: PLE 逆量子化がグラフ内 embedding とビット一致（GPU 不要）",
  ignore: !MODEL_PRESENT || !SIDECAR_PRESENT,
  fn: async () => {
    const index = await readPleIndex();
    assertEquals(index.tokens, VOCAB, "sidecar の token 行数（= 主 embedding の vocab）");
    assertEquals(index.layers, LAYERS, "sidecar の層数");
    assertEquals(index.dim, PLE_DIM, "sidecar の層当たり次元");
    assertEquals(index.embedScale, Math.sqrt(PLE_DIM), "embed scale（hidden_per_layer ** 0.5）");
    assert(
      index.shards.length > 1,
      `sidecar が ${index.shards.length} 本（vocab レンジ分割が無い）`,
    );

    const probeFile = parseSafetensors(await readBuffer(PRODUCT_ROOT, PLE_PROBE_FILE));
    const tokens = [...goldenI32(probeFile, PROBE_TOKENS_KEY)];
    const expected = goldenF32(probeFile, PROBE_INPUTS_KEY);
    assertEquals(
      expected.length,
      tokens.length * LAYERS * PLE_DIM,
      "参照 per_layer_inputs の要素数",
    );
    // probe は shard 境界の両側を踏むので、全 shard を触る（触らないと LRU の観測が空振りする）。
    const touched = new Set(
      tokens.map((token) => index.shards.findIndex((shard) => token < shard.stop)),
    );
    assertEquals(touched.size, index.shards.length, "probe が踏む shard 数（全 shard を踏むこと）");

    // 既定の予算（= 最大 shard 2 本ぶん）で引く。本数ではなくバイトで頭打ちになることを見る。
    const budget = defaultGemma4PleResidentBytes(index);
    const ple = await openPle(budget);
    const gathered = await ple.gather(tokens);
    assertEquals(gathered.dtype, "f32", "gather の dtype");
    assertEquals(gathered.shape, [1, tokens.length, LAYERS, PLE_DIM], "gather の shape");

    // 厳密一致（tolerance を持たない）。1 要素でも違えば最初の位置を名指しで落とす。
    let mismatches = 0;
    let first = "";
    for (let element = 0; element < expected.length; element += 1) {
      if (gathered.data[element] === expected[element]) continue;
      mismatches += 1;
      if (first !== "") continue;
      const token = tokens[Math.floor(element / (LAYERS * PLE_DIM))];
      const layer = Math.floor(element / PLE_DIM) % LAYERS;
      first = `token ${token} / 層 ${layer} / 列 ${element % PLE_DIM}: ` +
        `${gathered.data[element]} ≠ ${expected[element]}`;
    }
    assertEquals(
      mismatches,
      0,
      `PLE 逆量子化が torch の 35 表経路と違う（${mismatches} 要素 / 最初の食い違い ${first}）`,
    );

    // 決定 3 の実測: 触った shard だけ読み、LRU がバイト予算で落とす。
    const stats = ple.stats();
    assertEquals(stats.loads, index.shards.length, "取りに行った shard 数（触ったぶんだけ）");
    assertEquals(stats.resident, 2, "常駐 shard 数（既定 = 最大 shard 2 本ぶんで頭打ち）");
    assert(
      stats.residentBytes <= budget,
      `常駐 ${stats.residentBytes} バイトが予算 ${budget} を超えている`,
    );
    console.log(
      `[e2e] gemma4 product PLE: probe ${tokens.length} token × ${LAYERS} 層 × ${PLE_DIM} 次元が` +
        `ビット一致 / shard ${stats.loads} 本ロード・常駐 ${stats.resident} 本` +
        `（${(stats.residentBytes / 1024 / 1024).toFixed(0)} / 予算 ${
          (budget / 1024 / 1024).toFixed(0)
        } MiB）`,
    );
  },
});

/**
 * ① 形の前提。既存 2 系列との差分（入力 +2 本・出力は最終行 logits・argmax 不在・PLE 表の不在）を
 * ここで見る — states / 層種別 / 格納の本体検査は書き手側（`export_product.assert_ir_form_product`）
 * が持つ。ここが緩むと**別のグラフを検収してしまう**。
 */
const assertProductForm = (parsed: PreparedModel): void => {
  const graph = parsed.graph;
  assertEquals(
    // 順序は書き手の宣言順に依らせない（名前で見る — 増えたのは rope 4 本で、`position_ids` は
    // 消えた: 位置は「表を引く添字」ではなく「表そのもの」としてホストから渡る）。
    [...graph.inputs.map((spec) => spec.name)].sort(),
    [INPUT_IDS, PER_LAYER_INPUTS, LAST_ROW, ...gemma4RopeInputNames()].sort(),
    "グラフ入力（製品形は per_layer_inputs / rope 4 本 / last_row）",
  );
  const perLayer = graph.inputs.find((spec) => spec.name === PER_LAYER_INPUTS);
  assert(perLayer !== undefined, `'${PER_LAYER_INPUTS}' が無い`);
  assertEquals(perLayer.dtype, "f32", `'${PER_LAYER_INPUTS}' の dtype`);
  assertEquals(
    perLayer.shape,
    [1, "M", LAYERS, PLE_DIM],
    `'${PER_LAYER_INPUTS}' の shape（ホストが供給する [1,M,35,256]）`,
  );

  assertEquals(graph.outputs.length, 1, "graph.outputs の本数（最終行 logits の 1 本）");
  assertEquals(
    graph.values[graph.outputs[0]].shape,
    [1, 1, VOCAB],
    "出力 0 の shape（最終**行**のみ — 全行 logits への退行検出）",
  );
  const producer = new Map<string, (typeof graph.nodes)[number]>();
  for (const node of graph.nodes) {
    for (const out of node.outs) producer.set(out, node);
  }
  const exit = producer.get(graph.outputs[0]);
  assert(exit !== undefined, "出力 0 がノード出力でない");
  assert(
    exit.op !== "argmax",
    "出力 0 の供給元が argmax（製品出口は logits で、sampling はホスト — ADR 0083 決定 6）",
  );
  assertEquals(
    graph.nodes.filter((node) => node.op === "argmax").length,
    0,
    "argmax ノードの本数（グラフに 1 本も無いこと）",
  );

  // PLE 表がグラフに残っていないこと（常駐 2,240MiB が戻る形）。embedding の重みスロットは
  // ins[0]（`WEIGHT_SLOTS[embedding] = 0`）。
  const embeddings = graph.nodes.filter((node) => node.op === "embedding");
  assertEquals(
    embeddings.filter((node) => {
      const shape = graph.values[node.ins[0]].shape;
      return shape.length === 2 && shape[0] === VOCAB && shape[1] === PLE_DIM;
    }).length,
    0,
    `PLE 表 [${VOCAB},${PLE_DIM}] を引く embedding が残っている（ホスト gather へ出し切れていない）`,
  );
  // 主 embedding 1 + 最終行の行選択 1（RoPE の cos / sin は表引きでなくホスト供給の派生入力 —
  // ADR 0091 — なので gather は無い）。
  assertEquals(
    embeddings.length,
    2,
    "embedding ノードの本数（PLE 35 本と RoPE 表 4 本が消えた形）",
  );

  assertEquals(
    Object.keys(graph.states).length,
    SLOTS,
    "states スロットの本数（既存 2 系列と同一）",
  );
  assertEquals([...graph.symbols].sort(), ["C", "M"], "graph.symbols");
};

/** i32 の入力テンソル 1 本（token id 列も絶対位置列も `[1, rows]`）。 */
const i32Row = (rows: number, data: Int32Array<ArrayBuffer>): Tensor => ({
  dtype: "i32",
  shape: [1, rows],
  data,
});

/** token-only 形と同じ行選択入力（`[1]` の i32）。 */
const lastRowInput = (row: number): Tensor => ({
  dtype: "i32",
  shape: [1],
  data: Int32Array.of(row),
});

/**
 * 最終行 logits `[1,1,V]` からホスト側 argmax で次 token を読む。
 *
 * MUST: 最大値は**先に出た方**を採る（`>` で更新）— torch の `argmax` と同じ tie-break で、
 * `>=` にすると同値の並びで列が割れる。
 * MUST: dtype と shape を検査する（出力名の取り違えは例外も警告も出ないまま別の列を返す）。
 */
const argmaxToken = (tensor: Tensor, where: string): number => {
  assertEquals(tensor.dtype, "f32", `${where}: logits の dtype`);
  assertEquals(tensor.shape, [1, 1, VOCAB], `${where}: logits の shape`);
  const data = tensor.data;
  let best = 0;
  let bestValue = data[0];
  for (let index = 1; index < VOCAB; index += 1) {
    if (data[index] > bestValue) {
      bestValue = data[index];
      best = index;
    }
  }
  // 非有限（位置表の外の gather / 壊れた PLE）は「もっともらしい token id」に畳まれるので、
  // 畳まれる**前**にここで落とす。
  assert(Number.isFinite(bestValue), `${where}: logits の最大値が非有限（${bestValue}）`);
  return best;
};

/** 生成 1 本ぶんの実測（遅延ロードが効いていることを恒真でなく見るための欄）。 */
type GenerationRecord = { readonly tokens: number[]; readonly gathers: number };

/**
 * ホスト PLE gather + ホスト argmax の greedy ループ（製品面の縮退形）。
 *
 * MUST: pad 行にも `input_ids` と同じ 0 の PLE を渡す — グラフ内で引いていたときは pad 行が
 * token 0 の行を引いていたので、0 埋めにすると pad 行の値が別物になる（ADR 0066 追記 6 の
 * 値契約は「ホストが 0 で埋める」であって「PLE を 0 にする」ではない）。
 */
const generate = async (
  session: Awaited<ReturnType<PreparedModel["createSession"]>>,
  logitsName: string,
  ple: Gemma4Ple,
  prompt: readonly number[],
  maxNewTokens: number,
): Promise<GenerationRecord> => {
  const chunks = planPrefillChunks(prompt.length, CHUNK_LENGTH);
  const lastPosition = prompt.length + maxNewTokens - 2;
  assert(lastPosition < MAX_POSITION, `最終位置 ${lastPosition} がモデルの位置上限の外`);
  assert(prompt.length + maxNewTokens <= CAPACITY, `T + K が容量 ${CAPACITY} を超える`);

  const context = await session.createGenerationContext({
    bindings: { [CAPACITY_SYMBOL]: CAPACITY },
    chunkLength: CHUNK_LENGTH,
  });
  let gathers = 0;
  try {
    let token = 0;
    for (const chunk of chunks) {
      const ids = new Int32Array(CHUNK_LENGTH);
      const positions = new Int32Array(CHUNK_LENGTH);
      for (let row = 0; row < chunk.queryLength; row += 1) {
        ids[row] = prompt[chunk.position + row];
        positions[row] = chunk.position + row;
      }
      const perLayer = await ple.gather([...ids]);
      gathers += 1;
      const outputs = await session.run(
        {
          [INPUT_IDS]: i32Row(CHUNK_LENGTH, ids),
          [PER_LAYER_INPUTS]: perLayer,
          ...gemma4RopeInputs(ROPE, positions),
          [LAST_ROW]: lastRowInput(chunk.queryLength - 1),
        },
        undefined,
        { context, queryLength: chunk.queryLength },
      );
      token = argmaxToken(outputs[logitsName], `prefill@${chunk.position}`);
    }

    const tokens = [token];
    for (let index = 0; index + 1 < maxNewTokens; index += 1) {
      const current = tokens[index];
      const perLayer = await ple.gather([current]);
      gathers += 1;
      const outputs = await session.run(
        {
          [INPUT_IDS]: i32Row(1, Int32Array.of(current)),
          [PER_LAYER_INPUTS]: perLayer,
          ...gemma4RopeInputs(ROPE, [prompt.length + index]),
          [LAST_ROW]: lastRowInput(0),
        },
        undefined,
        { context, queryLength: 1 },
      );
      tokens.push(argmaxToken(outputs[logitsName], `decode@${index}`));
    }
    return { tokens, gathers };
  } finally {
    // MUST: 途中で落ちても返す（KV 容量ぶんの VRAM を抱えたままにしない）。
    await context.dispose();
  }
};

Deno.test({
  name: "Gemma 4 E2B 製品検収: 交差 parity（実 GPU / ホスト PLE + ホスト argmax）",
  ignore: !AVAILABLE || !GPU_AVAILABLE,
  fn: async (t) => {
    const shards = resolveShards(new URL(MODEL_FILE, PRODUCT_ROOT));
    const parsed = prepareModel(await readShard(shards[0]));
    const logitsName = parsed.graph.outputs[0];

    await t.step("① 形の前提: PLE 外出し + 最終行 logits 出口である", () => {
      assertProductForm(parsed);
    });

    const index = await readPleIndex();
    // 全 shard 常駐（LRU の観測は GPU 不要の②が持つ — こちらは生成の往復で shard を
    // 読み直さないことだけを見る）。
    const ple = await openPle(allResidentBytes(index));
    const gpu = await acquireGpu();
    const session = await parsed.createSession(gpu, streamShards(shards.slice(1)));
    try {
      const storage = session.diagnostics().storage;
      assert(storage !== undefined, "diagnostics.storage が無い");
      assertEquals(storage.hostExpandedBytes, 0, "hostExpandedBytes（適格落ちの CPU 展開）");
      // ADR 0085 の主張そのもの: PLE を外した容器は常駐が半分以下になる。
      assert(
        storage.residentCompressedBytes < RESIDENT_CEILING,
        `residentCompressedBytes ${storage.residentCompressedBytes} が ${RESIDENT_CEILING} 以上` +
          `（PLE がグラフに残った容器を読んでいる — ADR 0085 の常駐削減が効いていない）`,
      );
      console.log(
        `[e2e] gemma4 product 常駐: 圧縮 ${storage.residentCompressedBytes} バイト` +
          `（PLE 外出し前の decode 系列は 3.0e9 超）`,
      );

      await t.step("③ 3 ケース × 16 step が opt-in 系列の期待列と厳密一致", async () => {
        let totalGathers = 0;
        for (const caseName of EXPECTED_CASES) {
          const golden = parseSafetensors(
            await readBuffer(GOLDEN_ROOT, `${GREEDY_PREFIX}${caseName}${SUFFIX}`),
          );
          const prompt = [...goldenI32(golden, "prompt")];
          const expected = [...goldenI32(golden, "expected")];
          assertEquals(expected.length, GREEDY_STEPS, `${caseName}: golden の step 数`);

          const started = performance.now();
          const generated = await generate(session, logitsName, ple, prompt, GREEDY_STEPS);
          totalGathers += generated.gathers;
          assertEquals(
            generated.tokens,
            expected,
            `${caseName}: 生成 token 列（対 logits opt-in 系列 golden）`,
          );
          console.log(
            `[e2e] gemma4 product ${caseName}: T=${prompt.length} + K=${GREEDY_STEPS} step / ` +
              `PLE gather ${generated.gathers} 回 / ${(performance.now() - started).toFixed(0)}ms`,
          );
        }
        // 遅延ロードが実効（恒真でない）: gather は数十回走るのに、shard の取得は最大でも
        // 本数ぶん。毎回読み直す実装ならここが gather 回数まで伸びる。
        const stats = ple.stats();
        assert(
          totalGathers > stats.loads,
          `PLE gather ${totalGathers} 回に対し shard 取得 ${stats.loads} 回` +
            `（キャッシュが効いていない）`,
        );
        assert(
          stats.loads <= index.shards.length,
          `shard 取得 ${stats.loads} 回が本数 ${index.shards.length} を超える（読み直しが起きている）`,
        );
        console.log(
          `[e2e] gemma4 product PLE 遅延ロード: gather ${totalGathers} 回 / shard 取得 ` +
            `${stats.loads} 回（全 ${index.shards.length} 本中）`,
        );
      });
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});
