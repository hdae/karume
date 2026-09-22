/**
 * 言語横断 fixture — **Python の書き手**（`tools/exporter/src/karume/container.py`）が書いた `krm` を
 * TS の読み手で開く。docs/container-v1.md §7 / §9、docs/ir-v2.md「正準直列化」。
 *
 * ここが見るのは「TS の書き手と TS の読み手が閉じている」ことではなく、**別実装が書いたバイト列を
 * 開けること**である（`container_format_test.ts` は前者で、往復だけでは 2 実装の食い違いを
 * 検出できない）。
 *
 * 資産の再生成は exporter 側の 1 コマンド:
 * `cd tools/exporter && KARUME_FIXTURE=write uv run pytest tests/test_container.py -k fixture`
 */

import { assert, assertEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import {
  parseGraphDescriptor,
  parseModelDescriptor,
  serializeGraphDescriptor,
  serializeModelDescriptor,
  sha256Hex,
} from "../src/format/container/descriptor.ts";
import { readHeader } from "../src/format/container/header.ts";
import { HEADER_BYTES } from "../src/format/container/limits.ts";
import { openContainer } from "../src/format/container/open.ts";

const FIXTURE = new URL("./fixtures/container/synthetic.krm", import.meta.url);

/** 分割形 fixture の part 列（part 0 から — const は長さ 0 のファイルとして並ぶ）。 */
const SPLIT_FIXTURE_PARTS = 5;
const splitFixture = (index: number): URL =>
  new URL(
    `./fixtures/container/synthetic-split-${String(index + 1).padStart(5, "0")}-of-${
      String(SPLIT_FIXTURE_PARTS).padStart(5, "0")
    }.krm`,
    import.meta.url,
  );

const fixtureBytes = async (): Promise<Uint8Array<ArrayBuffer>> =>
  await Deno.readFile(FIXTURE) as Uint8Array<ArrayBuffer>;

const splitParts = async (): Promise<Uint8Array<ArrayBuffer>[]> =>
  await Promise.all(
    Array.from(
      { length: SPLIT_FIXTURE_PARTS },
      async (_, i) => await Deno.readFile(splitFixture(i)) as Uint8Array<ArrayBuffer>,
    ),
  );

/**
 * fixture の payload（Python 側 `container_fixture.pattern_bytes` と同じ**単純な**パターン —
 * `seed` から 1 ずつ上がるバイト列）。生成器の写しではなく 1 行で書ける形なので、fixture を
 * 焼き直したときに「ミラーがずれた」で赤くなる余地が無い。
 */
const patternBytes = (length: number, seed: number): Uint8Array<ArrayBuffer> =>
  Uint8Array.from({ length }, (_, i) => (seed + i) & 0xff);

describe("container fixture (Python writer)", () => {
  it("単一形を開き、全 block が sha256 検証つきで戻る", async () => {
    const bytes = await fixtureBytes();
    const container = await openContainer({ kind: "bytes", bytes });

    assertEquals(container.header.kind, "model");
    assertEquals(container.header.version, 1);
    assertEquals(Object.keys(container.graphs), ["synthetic"]);
    assertEquals(container.model?.provenance, { license: "apache-2.0", writer: "karume" });

    const ids = [
      ...container.graph.const.blocks.map((block) => block.id),
      ...(container.model?.blocks ?? []).map((block) => block.id),
    ];
    assert(ids.length > 0, "block が 1 本も無い");
    for (const id of ids) {
      // readBlock は宣言の sha256 と突き合わせるので、返ってきた時点で検証済み。
      const block = await container.readBlock(id);
      assertEquals(await sha256Hex(block), container.locate(id).record.sha256, id);
    }
  });

  it("合流層が Python の束縛表から供給計画を出す（piece 列 / rowAxis 1 / const / shared）", async () => {
    const bytes = await fixtureBytes();
    const container = await openContainer({ kind: "bytes", bytes });
    const bound = container.graphs["synthetic"];

    // shared 宣言は束縛表の突合集合の外（供給計画を持たない）。
    assertEquals(bound.declaration.initializers["lm_head.weight"], { shared: true });
    assertEquals(bound.supplies.has("lm_head.weight"), false);

    const enc = bound.supplies.get("enc.weight");
    assertEquals(enc?.origin, "model");
    assertEquals(enc?.encoding.codec, "int4-sym-g");
    assertEquals(enc?.encoding.groupSize, 32);
    assertEquals(enc?.blocks.length, 1);
    assertEquals(enc?.scale?.payloadBytes, 8 * 4);

    // block 上限（fixture は 256 B）を超えた重みは行境界で 2 本に割れている。
    const big = bound.supplies.get("big.weight");
    assertEquals(big?.blocks.map((block) => block.rows), [[0, 8], [8, 16]]);
    // 規則③: companion scale は piece 1 と同じ part。
    assertEquals(big?.scale?.part, big?.blocks[0].part);

    // conv_transpose1d 形（行の軸が 1）— scale は shape[1] 本。
    const conv = bound.supplies.get("conv.weight");
    assertEquals(conv?.encoding.rowAxis, 1);
    assertEquals(conv?.scale?.payloadBytes, 3 * 4);

    // 末尾の詰め物は書き手が焼く（payload 42 バイト → block 44 バイト）。
    const dec = bound.supplies.get("dec.weight");
    assertEquals(dec?.blocks[0].payloadBytes, 42);
    assertEquals(dec?.blocks[0].length, 44);

    // const 領域（part 1）から供給される initializer。
    assertEquals(bound.supplies.get("const.a1b2c3d4e5f60718")?.origin, "const");
    assertEquals(bound.supplies.get("const.b1b2c3d4e5f6071a")?.encoding.codec, "i32");
  });

  it("2 文書は正準直列化と一致する（parse → serialize でバイト同一）", async () => {
    const bytes = await fixtureBytes();
    const header = readHeader(bytes);
    const graphBytes = bytes.subarray(HEADER_BYTES, HEADER_BYTES + header.graphDescriptorLength)
      .slice();
    const modelBytes = bytes.subarray(
      HEADER_BYTES + header.graphDescriptorLength,
      HEADER_BYTES + header.graphDescriptorLength + header.modelDescriptorLength,
    ).slice();

    assertEquals(serializeGraphDescriptor(parseGraphDescriptor(graphBytes)), graphBytes);
    assertEquals(serializeModelDescriptor(parseModelDescriptor(modelBytes)), modelBytes);
  });

  it("krg 抽出は krm のバイトコピーで成り立ち、グラフ記述はそのまま載る", async () => {
    const bytes = await fixtureBytes();
    const container = await openContainer({ kind: "bytes", bytes });
    const extracted = await container.extractGraph();

    const header = readHeader(extracted);
    assertEquals(header.kind, "graph");
    assertEquals(header.modelDescriptorLength, 0);
    assertEquals(
      extracted.subarray(HEADER_BYTES, HEADER_BYTES + header.graphDescriptorLength),
      bytes.subarray(HEADER_BYTES, HEADER_BYTES + header.graphDescriptorLength),
    );
    const reopened = await openContainer({ kind: "bytes", bytes: extracted });
    assertEquals(reopened.model, undefined);
    assertEquals(Object.keys(reopened.graphs), ["synthetic"]);
  });
});

describe("split container fixture (Python writer)", () => {
  it("part 列を開き、全 block が sha256 検証つきで戻る", async () => {
    const parts = await splitParts();
    const container = await openContainer({ kind: "parts", parts });

    assertEquals(container.header.kind, "model");
    assertEquals(Object.keys(container.graphs), ["synthetic-split"]);
    // const が空でも part 1 は 0 バイトのファイルとして並ぶ（§8）。
    assertEquals(container.graph.const.blocks, []);
    assertEquals(container.model?.parts[0].length, 0);
    assertEquals(parts[1].byteLength, 0);
    // 重みの part が 2 本 + 資産の part が 1 本（part 0 を除いて 4 本 — 内訳は下のケース）。
    assertEquals(container.model?.parts.length, SPLIT_FIXTURE_PARTS - 1);

    for (const block of container.model?.blocks ?? []) {
      const bytes = await container.readBlock(block.id);
      assertEquals(await sha256Hex(bytes), block.sha256, block.id);
    }
  });

  it("assets は名前 / 役割 / 論理長を宣言し、block は payload + 0x00 の詰め物になる", async () => {
    const parts = await splitParts();
    const container = await openContainer({ kind: "parts", parts });
    const assets = container.model?.assets ?? {};

    assertEquals(
      Object.fromEntries(Object.entries(assets).map(([name, a]) => [name, [a.role, a.length]])),
      { ple_index: ["ple-index", 64], rope_base: ["rope-base", 37] },
    );

    // 37 バイトの payload は 40 バイトの block になり、末尾 3 バイトは 0x00（§4.1）。
    // 論理長が宣言に在るので、消費側は末尾の 0x00 を推測で剥がない（ADR 0109 決定 4）。
    const rope = await container.readBlock(assets["rope_base"].block);
    assertEquals(rope.byteLength, 40);
    assertEquals(rope.subarray(0, assets["rope_base"].length), patternBytes(37, 52));
    assertEquals(rope.subarray(assets["rope_base"].length), new Uint8Array([0, 0, 0]));

    // 詰め物の要らない資産は payload ちょうど。
    const index = await container.readBlock(assets["ple_index"].block);
    assertEquals(index, patternBytes(64, 51));

    // 資産の block は重み block と part を共有しない（§4.2 — 区間読みの資産は専用 part）。
    const assetBlocks = new Set(Object.values(assets).map((a) => a.block));
    const blocks = container.model?.blocks ?? [];
    const assetParts = new Set(blocks.filter((b) => assetBlocks.has(b.id)).map((b) => b.part));
    const weightParts = new Set(blocks.filter((b) => !assetBlocks.has(b.id)).map((b) => b.part));
    // 空振りしない形で固定する: 資産の part が 1 本・重みの part が 2 本（合わせて part 3 本）。
    assertEquals(assetParts.size, 1);
    assertEquals(weightParts.size, 2);
    assertEquals(assetParts.intersection(weightParts).size, 0);
  });

  it("2 文書は正準直列化と一致する（parse → serialize でバイト同一）", async () => {
    const parts = await splitParts();
    const header = readHeader(parts[0]);
    const graphBytes = parts[0].subarray(HEADER_BYTES, HEADER_BYTES + header.graphDescriptorLength)
      .slice();
    const modelBytes = parts[0].subarray(
      HEADER_BYTES + header.graphDescriptorLength,
      HEADER_BYTES + header.graphDescriptorLength + header.modelDescriptorLength,
    ).slice();

    assertEquals(serializeGraphDescriptor(parseGraphDescriptor(graphBytes)), graphBytes);
    assertEquals(serializeModelDescriptor(parseModelDescriptor(modelBytes)), modelBytes);
  });
});
