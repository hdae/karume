/**
 * PLE の GPU 常駐席（ADR 0085 追記〈GPU 常駐席〉）のうち、**GPU を要さない部分**の門。
 *
 * 見るのは 4 つ — ①席の指定の受理と拒否 ②gather IR の形（ノード列・shape・initializer の宣言）と
 * グラフ単体の規則の門（`parseIrDeclarationValue`）③容器の資産 → メモリ内容器の突合（格納
 * dtype → codec、piece の並びと part 割り、繋ぎ直したバイト列が原本と一致すること）④block が
 * 1 本の索引の供給形。実 GPU での値の一致は `e2e_gemma4_ple_gpu_test.ts` が持つ。
 */
import { assert, assertEquals, assertThrows } from "@std/assert";
import {
  type IrDeclaration,
  IrError,
  parseIrDeclarationValue,
  prepareContainer,
} from "@karume/runtime";
import {
  buildGemma4PleGatherContainer,
  gemma4PleGatherGraph,
  gemma4PleGpuBytes,
} from "../src/gemma/ple-gpu.ts";
import { ModelInputError } from "../src/errors.ts";
import { resolveGemma4PleResidency } from "../src/gemma/pipeline.ts";
import { pleFixture, type PleFixtureSpec, scaleOf } from "./helpers/ple-fixture.ts";

/** 索引 1 本ぶんの作り物（実資産と同じ綴り・同じ関係で、桁だけ小さい）。 */
const SPEC: PleFixtureSpec = {
  storage: "i4",
  tokens: 6,
  layers: 2,
  dim: 16,
  embedScale: 16,
  valueRows: 4,
};

/** 1 層ぶんのバイト数（i4 なので dim / 2）。 */
const ROW_BYTES = SPEC.dim / 2;
/** 重みの行数（token × 層）。 */
const WEIGHT_ROWS = SPEC.tokens * SPEC.layers;

const rig = pleFixture(SPEC);
const index = rig.index;

Deno.test("pleResidency: 受理と、併用できないノブの拒否", () => {
  assertEquals(resolveGemma4PleResidency("T", {}), "host");
  assertEquals(resolveGemma4PleResidency("T", { pleResidency: "host" }), "host");
  assertEquals(resolveGemma4PleResidency("T", { pleResidency: "gpu" }), "gpu");
  // 綴り違いを既定へ倒さない（型の外から来る指定）。
  assertThrows(
    () => resolveGemma4PleResidency("T", { pleResidency: "GPU" as "gpu" }),
    Error,
    "pleResidency",
  );
  // 併用できないノブの組合せは呼び手のオプションだけで決まるので入力起因（ADR 0107 決定 2）。
  assertThrows(
    () => resolveGemma4PleResidency("T", { pleResidency: "gpu", maxResidentPleBytes: 0 }),
    ModelInputError,
    "maxResidentPleBytes",
  );
  assertThrows(
    () => resolveGemma4PleResidency("T", { pleResidency: "gpu", speculative: {} }),
    ModelInputError,
    "speculative",
  );
  // 逆側: 綴り違いは「受理集合を引き直す」側なので入力起因にしない（決定 3 の除外）。
  const misspelled = assertThrows(() =>
    resolveGemma4PleResidency("T", { pleResidency: "GPU" as "gpu" })
  );
  assert(!(misspelled instanceof ModelInputError));
});

Deno.test("gather IR: embedding → mul の 2 ノードで、添字の形がそのまま per-layer の形になる", () => {
  const graph = gemma4PleGatherGraph(index);
  assertEquals(graph.nodes.map((node) => node.op), ["embedding", "mul"]);
  assertEquals(graph.nodes[0].ins, ["ple", "ple_index"]);
  assertEquals(graph.nodes[1].ins, ["ple_raw", "embed_scale"]);
  assertEquals(graph.inputs, [{ name: "ple_index", dtype: "i32", shape: [1, "M", 2] }]);
  // 重みは (token, layer) を行に畳んだ形 — 1 行 = 1 層ぶんなので scale の並びが索引と揃う。
  assertEquals(graph.values.ple.shape, [WEIGHT_ROWS, SPEC.dim]);
  assertEquals(graph.values[graph.outputs[0]].shape, [1, "M", 2, SPEC.dim]);
  // IR v2 の宣言が持つのは名前だけ（格納は束縛表 = メモリ内容器の encoding 側）。
  assertEquals(Object.keys(graph.initializers), ["ple", "embed_scale"]);
  assertEquals(graph.initializers.ple, { shared: false });
});

Deno.test("メモリ内容器: 格納 dtype が codec 台帳の登録名と group 長に写る", async () => {
  const mapping = [["i8", "int8-sym"], ["i4", "int4-sym-g"], ["i2", "int2-off"]] as const;
  for (const [storage, codec] of mapping) {
    const fixture = pleFixture({ ...SPEC, storage });
    const bound = await buildGemma4PleGatherContainer(fixture.index, fixture.openBlock, "T");
    const encoding = bound.graphs.ple.supplies.get("ple")?.encoding;
    assertEquals(encoding?.codec, codec);
    // i4 の group = 行長・i8 / i2 の per-channel = 行長。どちらも group 数は 1 なので、
    // companion scale は `[tokens × layers, 1]` の f32 列そのものになる。
    assertEquals(encoding?.groupSize, SPEC.dim);
    assertEquals(encoding?.rowAxis, 0);
    // 合流後の格納まで 1 本で見る（3 種とも capability 門を通ること — GPU は触らない）。
    assertEquals(prepareContainer(bound, "ple").graph.initializers.ple.storage, {
      codec,
      groupSize: SPEC.dim,
      rowAxis: 0,
    });
  }
});

Deno.test("メモリ内容器: block 1 本の索引は piece に割らず丸ごと 1 本で供給される", async () => {
  // `valueRows` を省くと block は 1 本（`pieces` は 2 本以上 MUST なので `bytes` の分岐）。
  // 旧実装はここを「PLE の行数が 2 行未満」で落としていたので、通ることを縛る。
  const fixture = pleFixture({ ...SPEC, storage: "i8", valueRows: undefined });
  assertEquals(fixture.index.values.blocks.length, 1);
  const bound = await buildGemma4PleGatherContainer(fixture.index, fixture.openBlock, "T");
  const supply = bound.graphs.ple.supplies.get("ple");
  assert(supply !== undefined, "initializer 'ple' の供給が無い");
  assertEquals(supply.blocks.length, 1);
  assertEquals(supply.blocks[0].rows, [0, WEIGHT_ROWS]);
  assertEquals(
    await bound.readBlock(supply.blocks[0].id),
    fixture.blocks.get(fixture.index.values.blocks[0].asset),
  );
  // companion scale は丸ごと 1 本の供給とも同じ part に置かれる（規則③）。
  assertEquals(supply.scale?.part, supply.blocks[0].part);
  assertEquals(prepareContainer(bound, "ple").graph.outputs, ["per_layer"]);
});

/**
 * 解析済みの宣言を JSON 文書へ戻す（`shared: false` は綴れないので欄ごと落とす）— 仕込みを
 * 載せる土台。
 */
const documentOf = (declaration: IrDeclaration): Record<string, unknown> => ({
  ...declaration,
  initializers: Object.fromEntries(
    Object.keys(declaration.initializers).map((name) => [name, {}]),
  ),
});

Deno.test("gather IR: グラフ単体の規則を通る（嘘の requires.ops は宣言の段で落ちる）", () => {
  const document = documentOf(gemma4PleGatherGraph(index));
  // 陰性対照: 仕込みの無い文書はそのまま通り、同じ宣言に戻る。
  assertEquals(parseIrDeclarationValue(document), gemma4PleGatherGraph(index));
  // 使っていない op を宣言しても、合流（宣言 × 供給）も `prepareContainer`（capability +
  // op 契約）も見ない — 落とせるのはグラフ単体の規則を見るこの門だけ。
  assertThrows(
    () => parseIrDeclarationValue({ ...document, requires: { ops: ["embedding", "mul", "topk"] } }),
    IrError,
    "requires.ops",
  );
});

Deno.test("gather IR: ランタイムの op 契約と格納を通る", async () => {
  const bound = await buildGemma4PleGatherContainer(index, rig.openBlock, "T");
  // prepareContainer は非対応 op / 格納 codec と IR 契約を全件見る（GPU は触らない）。
  const prepared = prepareContainer(bound, "ple");
  assertEquals(prepared.graph.outputs, ["per_layer"]);
  assertEquals(prepared.graph.symbols, ["M"]);
});

Deno.test("常駐バイト数: 索引だけで決まる（E2B QAT の実値）", () => {
  const e2b = pleFixture({ storage: "i4", tokens: 4096, layers: 35, dim: 256, embedScale: 16 })
    .index;
  assertEquals(gemma4PleGpuBytes(e2b), {
    values: 4096 * 35 * 256 / 2,
    scales: 4096 * 35 * 4,
    total: 4096 * 35 * 256 / 2 + 4096 * 35 * 4,
  });
  // i8（通常 Gemma 4）は同じ寸法で倍になる。
  const i8 = pleFixture({ storage: "i8", tokens: 4096, layers: 35, dim: 256, embedScale: 16 })
    .index;
  assertEquals(gemma4PleGpuBytes(i8).values, 4096 * 35 * 256);
  const i2 = pleFixture({ storage: "i2", tokens: 4096, layers: 35, dim: 256, embedScale: 16 })
    .index;
  assertEquals(gemma4PleGpuBytes(i2).values, 4096 * 35 * 256 / 4);
});

Deno.test("メモリ内容器: piece が順に並び、繋ぎ直すと原本の values と一致する", async () => {
  const bound = await buildGemma4PleGatherContainer(index, rig.openBlock, "T");
  const supply = bound.graphs.ple.supplies.get("ple");
  assert(supply !== undefined, "initializer 'ple' の供給が無い");
  // piece は `values` の block と 1 対 1（行範囲は token 区間 × 層数 — 1 行 = 1 層ぶん）。
  assertEquals(supply.blocks.map((block) => block.rows), [[0, 8], [8, 12]]);
  // part 割り: piece k はそれぞれ別 part（= 構築時のフェンス 1 本ぶん）で、companion scale は
  // piece 1 と同じ part（container-v1 §13.3 の規則③）。
  const parts = supply.blocks.map((block) => block.part);
  assertEquals(new Set(parts).size, parts.length, "piece が同じ part に同居している");
  assertEquals(supply.scale?.part, parts[0]);

  // block 順に読み直した piece を繋ぐと、原本の values の連結そのものになる。
  const collected: Uint8Array[] = [];
  for (const block of supply.blocks) collected.push(await bound.readBlock(block.id));
  assertEquals(collected.map((piece) => piece.byteLength / ROW_BYTES), [8, 4]);
  const joined = new Uint8Array(WEIGHT_ROWS * ROW_BYTES);
  let cursor = 0;
  for (const piece of collected) {
    joined.set(piece, cursor);
    cursor += piece.byteLength;
  }
  const expected = new Uint8Array(WEIGHT_ROWS * ROW_BYTES);
  let at = 0;
  for (const block of index.values.blocks) {
    const bytes = rig.blocks.get(block.asset)!;
    expected.set(bytes, at);
    at += bytes.byteLength;
  }
  assertEquals(joined, expected);

  // scale の block は、平坦添字 `id * layers + layer` に並べ直した scale 表そのもの。
  // fixture の scale は token にも依存する（`scaleOf`）ので、この主張は token 方向のずれ
  // （行の巡回・block の取り違え）でも落ちる。
  const scaleBytes = await bound.readBlock(supply.scale!.id);
  assertEquals(scaleBytes.byteLength, WEIGHT_ROWS * 4);
  const scales = new Float32Array(scaleBytes.buffer, scaleBytes.byteOffset, WEIGHT_ROWS);
  assertEquals(
    [...scales],
    Array.from(
      { length: WEIGHT_ROWS },
      (_row, row) => scaleOf(Math.floor(row / SPEC.layers), row % SPEC.layers),
    ),
  );

  // embed_scale は丸ごと 1 本（f32・詰め物なし）。
  const embedScale = bound.graphs.ple.supplies.get("embed_scale");
  assertEquals(embedScale?.encoding.codec, "f32");
  const embedBytes = await bound.readBlock(embedScale!.blocks[0].id);
  assertEquals(
    new Float32Array(embedBytes.buffer, embedBytes.byteOffset, 1)[0],
    SPEC.embedScale,
  );
});
