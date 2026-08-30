// 実重み Gemma 4 E2B の **token-only 既定出口**（ADR 0068 決定 4）の実 GPU 検収門。
//
// `e2e_gemma4_greedy_test.ts`（logits opt-in 形 — `[logits, token]` の 2 出力）が生成ループと
// KV 機構の本体を受け持つのに対し、こちらは**出口の差し替えだけ**を検収する:
//
// ① 形の前提（入力 3 本 = input_ids / position_ids / **last_row**・出力 1 本 = argmax token・
//    states 30 スロットは opt-in 形と同一）
// ② **系列間交差 parity**: token-only 系列の greedy が logits opt-in 系列の
//    `greedy.<case>.safetensors`（torch full re-forward の期待列）と 3 ケース × K=16 で
//    厳密一致する。同じ重み・同じ丸め・同じ手術で出口だけが違うので、列が割れたら
//    行選択（last_row 配線）か 1 行 lm_head の側の誤り。
// ③ **出所（provenance）の束縛**: token-only 系列が書く `reference.json`（書き手側の正本は
//    `tools/export-recipes/gemma4/provenance.py`）が、元チェックポイントの指紋と、②で流用する
//    golden 1 本ずつの digest を持っていること。②の「同じ重み」という前提は資産の存在確認
//    だけでは守れない（片方だけ古い組み合わせでも門は緑になる — 全体レビュー CX-2.3）ので、
//    記録が無い / digest が合わない組み合わせは SKIP でなく **FAIL** にする。
//
// 期待列の正本を**再計算しない**のがこの門の設計 — logits opt-in 系列の golden をそのまま
// 流用することで、「両系列が同じ列を吐く」という交差検証そのものが門になる（torch 参照の
// 再実走〈数十分〉を払わず、しかも独立性は落ちない — 期待値の出所は full re-forward のまま）。
//
// 資産 2 系列を両方要求する: `outputs/series/gemma4-e2b-decode-token/`（model 本体 +
// `reference.json` — `tools/export-recipes/gemma4/export_token.py`）と
// `outputs/series/gemma4-e2b-decode/`（greedy golden — `export_decode.py`）。
// token-only 系列が無い環境では**明示 SKIP** する。

import { assert, assertEquals } from "@std/assert";
import {
  acquireGpu,
  parseSafetensors,
  type PreparedModel,
  prepareModel,
  type SafetensorsFile,
} from "@karume/runtime";
import { generateGreedy } from "../src/generation/greedy.ts";
import {
  modelPresent,
  readShard,
  resolveShards,
  streamShards,
} from "../../runtime/tests/helpers/shard-files.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const TOKEN_ROOT = new URL("../../../outputs/series/gemma4-e2b-decode-token/", import.meta.url);
const GOLDEN_ROOT = new URL("../../../outputs/series/gemma4-e2b-decode/", import.meta.url);
const MODEL_FILE = "model.safetensors";
const GREEDY_PREFIX = "greedy.";
const SUFFIX = ".safetensors";
/** 出所記録のファイル名と版（綴りの正本は `gemma4/provenance.py`）。 */
const REFERENCE_FILE = "reference.json";
const REFERENCE_SCHEMA = 1;

/** SKIP 時にそのまま貼れる生成コマンド。 */
const GENERATE_COMMAND =
  "cd tools/export-recipes && uv run --with 'transformers==5.14.1' python -m gemma4.export_token" +
  "（golden 側は … python -m gemma4.export_decode）";

/** 期待列の正本 = logits opt-in 系列の greedy golden（正典は export_decode.GREEDY_CASES）。 */
const EXPECTED_CASES = ["capital-en", "capital-ja", "context-en"] as const;
const GREEDY_STEPS = 16;

/** 実行条件は e2e_gemma4_greedy_test.ts と同値（同じ資産世代の裁定をそのまま使う）。 */
const CHUNK_LENGTH = 32;
const CAPACITY_SYMBOL = "C";
const CAPACITY = 640;
const MAX_POSITION = 1024;

const INPUT_IDS = "input_ids";
const POSITION_IDS = "position_ids";
/** token-only 形の行選択入力（正本は export_decode.TOKEN_ONLY_LAST_ROW）。 */
const LAST_ROW = "last_row";

const exists = (url: URL): boolean => {
  try {
    return Deno.statSync(url).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

const MODEL_PRESENT = modelPresent(new URL(MODEL_FILE, TOKEN_ROOT));
const GOLDENS_PRESENT = EXPECTED_CASES.every((name) =>
  exists(new URL(`${GREEDY_PREFIX}${name}${SUFFIX}`, GOLDEN_ROOT))
);
const AVAILABLE = MODEL_PRESENT && GOLDENS_PRESENT;

if (!MODEL_PRESENT) {
  console.warn(
    `[karume] token-only 系列（${TOKEN_ROOT.pathname}）が無いため Gemma 4 E2B token-only 検収を ` +
      `SKIP する。生成: ${GENERATE_COMMAND}`,
  );
}

/**
 * 依存資産の完全性（Codex 波 H 指摘 H-02 — 欠落を SKIP に畳まない）。この門は 2 系列に
 * 依存する: token-only 系列の model（自系列 — 無ければ「未生成」で SKIP が正しい）と、
 * logits opt-in 系列の greedy golden（期待列の正本）。**自系列があるのに正本が欠けている**のは
 * 未生成でなく欠損なので、SKIP でなく FAIL にする（opt-in 系列内部の欠けは
 * `e2e_gemma4_greedy_test.ts` の完全性テストが受け持つ — ここは系列間の依存だけを見る）。
 */
Deno.test({
  name: "Gemma 4 E2B token-only 資産: 期待列の正本（opt-in 系列 golden）が揃っている",
  ignore: !MODEL_PRESENT,
  fn: () => {
    for (const name of EXPECTED_CASES) {
      assert(
        exists(new URL(`${GREEDY_PREFIX}${name}${SUFFIX}`, GOLDEN_ROOT)),
        `${GREEDY_PREFIX}${name}${SUFFIX} が ${GOLDEN_ROOT.pathname} に無い` +
          `（token-only 系列はあるのに期待列の正本が欠けている）`,
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
 * ③ 出所の束縛（全体レビュー CX-2.3）。②が流用する golden は「この容器を作った実走が
 * その場で digest を採ったもの」でなければならない — logits opt-in 系列を採り直せば digest が
 * 動き、この門が落ちて token-only 系列の再 export を強制する。
 *
 * NOTE: `checkpoint` の指紋は**照合できない**（元チェックポイントはリポジトリの資産ではない）
 * ので、ここで見るのは記録として立っていることまで。両系列が同じ重みを踏んでいることの
 * 数値上の実証は②が受け持つ — この門はその②を「どの資産の組で見るべきか」に縛る側。
 */
Deno.test({
  name: "Gemma 4 E2B token-only 資産: 出所記録が参照 golden を束ねている",
  ignore: !MODEL_PRESENT,
  fn: async () => {
    const path = new URL(REFERENCE_FILE, TOKEN_ROOT);
    assert(
      exists(path),
      `${REFERENCE_FILE} が ${TOKEN_ROOT.pathname} に無い` +
        `（出所記録の無い世代の資産 — 再生成: ${GENERATE_COMMAND}）`,
    );
    const record = objectAt(JSON.parse(await Deno.readTextFile(path)), "根");

    assertEquals(record.get("schema"), REFERENCE_SCHEMA, `${REFERENCE_FILE} の schema`);
    assertEquals(stringAt(record, "series", "根"), seriesName(TOKEN_ROOT), "記録が名乗る系列");
    const checkpoint = objectAt(record.get("checkpoint"), "checkpoint");
    stringAt(checkpoint, "dir", "checkpoint");
    const fingerprint = objectAt(checkpoint.get("files"), "checkpoint.files");
    assert(fingerprint.size > 0, `${REFERENCE_FILE}: checkpoint.files が空`);
    for (const file of fingerprint.keys()) digestAt(fingerprint, file, "checkpoint.files");

    const reference = objectAt(record.get("reference"), "reference");
    assertEquals(
      stringAt(reference, "series", "reference"),
      seriesName(GOLDEN_ROOT),
      "束ねられた golden 系列（②が読む系列と同じであること）",
    );
    const goldens = objectAt(reference.get("goldens"), "reference.goldens");
    assertEquals(
      [...goldens.keys()].sort(),
      EXPECTED_CASES.map((name) => `${GREEDY_PREFIX}${name}${SUFFIX}`).sort(),
      "束ねられた golden の集合（②が読む 3 本と過不足なく一致）",
    );
    for (const name of EXPECTED_CASES) {
      const file = `${GREEDY_PREFIX}${name}${SUFFIX}`;
      const digest = digestAt(goldens, file, "reference.goldens");
      const bytes = await Deno.readFile(new URL(file, GOLDEN_ROOT));
      assertEquals(bytes.byteLength, digest.bytes, `${file}: byte 数`);
      assertEquals(
        await sha256Hex(bytes),
        digest.sha256,
        `${file}: sha256（token-only 系列が束ねた golden と別物 — ` +
          `どちらかの系列だけを作り直した組み合わせ。再生成: ${GENERATE_COMMAND}）`,
      );
    }
  },
});

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

/**
 * ① 形の前提。opt-in 形との差分（入力 +1 本・出力 −1 本・argmax 直結）だけをここで見る —
 * states / 層種別 / 格納の本体検査は書き手側（`export_decode.assert_ir_form_decode` の
 * `token_only=True`）と opt-in 門が持つ。
 */
const assertTokenOnlyForm = (parsed: PreparedModel): void => {
  const graph = parsed.graph;
  assertEquals(
    graph.inputs.map((spec) => spec.name),
    [INPUT_IDS, POSITION_IDS, LAST_ROW],
    "グラフ入力（token-only は last_row が増える）",
  );
  assertEquals(graph.outputs.length, 1, "graph.outputs の本数（token 1 本 — logits は出さない）");
  const producer = new Map<string, (typeof graph.nodes)[number]>();
  for (const node of graph.nodes) {
    for (const out of node.outs) producer.set(out, node);
  }
  const argmax = producer.get(graph.outputs[0]);
  assert(argmax !== undefined, "出力 0 がノード出力でない");
  assertEquals(argmax.op, "argmax", "出力 0 の供給元（argmax 直結）");
  assertEquals(
    Object.keys(graph.states).length,
    30,
    "states スロットの本数（opt-in 形と同一の 30 本）",
  );

  // **1 行 lm_head の固定**（Codex 波 H 指摘 H-01）。行ごとの lm_head と行選択は可換なので
  // 「全行 lm_head → 行選択 → argmax」でも token 列は一致する — ADR 0068 の実効（lm_head
  // 1 行・[M,V] バッファ消滅）はこの構造検査でしか固定できない。argmax から softcap 鎖
  // （ins[0] が本流）を遡った最初の linear が lm_head で、その入力は [1,1,H]（選択済みの
  // 1 行）・祖先に last_row 入力を持つ。
  let lmHead = argmax;
  let found = false;
  for (let step = 0; step < 8; step += 1) {
    const source = producer.get(lmHead.ins[0]);
    assert(source !== undefined, `token 出力の祖先（'${lmHead.ins[0]}'）が途切れた`);
    lmHead = source;
    if (lmHead.op === "linear") {
      found = true;
      break;
    }
  }
  assert(found, "token 出力の 8 段以内に lm_head（linear）が無い");
  const rowShape = graph.values[lmHead.ins[0]].shape;
  assertEquals(rowShape.slice(0, 2), [1, 1], "lm_head 入力の行数（全行 lm_head への退行検出）");
  const inputNames = new Set(graph.inputs.map((spec) => spec.name));
  const frontier = [lmHead.ins[0]];
  const seen = new Set<string>();
  const reachable = new Set<string>();
  while (frontier.length > 0) {
    const name = frontier.pop()!;
    if (seen.has(name)) continue;
    seen.add(name);
    if (inputNames.has(name)) {
      reachable.add(name);
      continue;
    }
    const upstream = producer.get(name);
    if (upstream !== undefined) frontier.push(...upstream.ins);
  }
  assert(
    reachable.has(LAST_ROW),
    `lm_head の入力が '${LAST_ROW}' に依存しない（到達した入力: ${[...reachable].sort()}）`,
  );
};

Deno.test({
  name: "Gemma 4 E2B token-only 検収: 系列間交差 parity（実 GPU / opt-in 系列の期待列）",
  ignore: !AVAILABLE || !GPU_AVAILABLE,
  fn: async (t) => {
    const shards = resolveShards(new URL(MODEL_FILE, TOKEN_ROOT));
    const parsed = prepareModel(await readShard(shards[0]));
    const tokenName = parsed.graph.outputs[0];

    await t.step("① 形の前提: token-only 出口である", () => {
      assertTokenOnlyForm(parsed);
    });

    const gpu = await acquireGpu();
    const session = await parsed.createSession(gpu, streamShards(shards.slice(1)));
    try {
      // 混成格納の常駐（ADR 0069 の検収条件 — 適格落ちは例外を出さず CPU 展開されるだけ
      // なので、hostExpandedBytes が唯一の直接観測）。
      const storage = session.diagnostics().storage;
      assert(storage !== undefined, "diagnostics.storage が無い");
      assertEquals(storage.hostExpandedBytes, 0, "hostExpandedBytes（適格落ちの CPU 展開）");

      await t.step("② 3 ケース × 16 step が opt-in 系列の期待列と厳密一致", async () => {
        for (const caseName of EXPECTED_CASES) {
          const golden = parseSafetensors(
            await readBuffer(GOLDEN_ROOT, `${GREEDY_PREFIX}${caseName}${SUFFIX}`),
          );
          const prompt = [...goldenI32(golden, "prompt")];
          const expected = [...goldenI32(golden, "expected")];
          assertEquals(expected.length, GREEDY_STEPS, `${caseName}: golden の step 数`);

          const started = performance.now();
          const generated = await generateGreedy({
            session,
            inputIds: INPUT_IDS,
            positionIds: POSITION_IDS,
            token: tokenName,
            lastRow: LAST_ROW,
            chunkLength: CHUNK_LENGTH,
            maxPosition: MAX_POSITION,
            bindings: { [CAPACITY_SYMBOL]: CAPACITY },
            prompt,
            maxNewTokens: GREEDY_STEPS,
          });
          assertEquals(generated, expected, `${caseName}: 生成 token 列（対 opt-in 系列 golden）`);
          console.log(
            `[e2e] gemma4 token-only ${caseName}: T=${prompt.length} + K=${GREEDY_STEPS} step / ` +
              `${(performance.now() - started).toFixed(0)}ms`,
          );
        }
      });
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});
