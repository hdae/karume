// `helpers/container-files.ts`（系列出力 / ミラーの `krm` を part 列として開く面）の門。
// GPU も実資産も要らない — 合成の容器（`helpers/container-write.ts`）を一時ディレクトリへ
// 分割形で書き、そのファイル列から開き直す。
//
// このヘルパを使う e2e は**正常な実資産しか渡さない**ので、fail-loudly の分岐（単一形と連番の
// 同居 / `of` の食い違い / 欠番 / はみ出し / 別 dtype の混入）も、区間読みと sha256 検証も、
// 実資産のある機でしか踏まれない。ここはそれを CPU だけで踏み切る席である。
//
// 鏡像の Python 側は `tools/exporter/tests/test_container.py` / `test_shards.py`。

import {
  assert,
  assertEquals,
  assertRejects,
  assertStringIncludes,
  assertThrows,
} from "@std/assert";
import {
  fileBlockSource,
  modelPresent,
  openSeriesContainer,
  resolveParts,
} from "./helpers/container-files.ts";
import {
  type ModelInput,
  writeModelContainer,
  type WrittenContainer,
} from "./helpers/container-write.ts";
import { type IrDeclaration, parseIrDeclaration } from "../src/format/ir.ts";

// ---------------------------------------------------------------------------
// 合成の容器
// ---------------------------------------------------------------------------

/** 決定的な疑似乱数バイト列（seed から xorshift32）。 */
const bytesOf = (length: number, seed: number): Uint8Array<ArrayBuffer> => {
  const out = new Uint8Array(new ArrayBuffer(length));
  let state = seed >>> 0 || 1;
  for (let i = 0; i < length; i += 1) {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    out[i] = state & 0xff;
  }
  return out;
};

type InitializerSpec = { readonly shape: readonly number[]; readonly shared?: true };

/** initializer ごとに 1 ノードで消費する最小の IR v2 グラフ。 */
const declaration = (initializers: Readonly<Record<string, InitializerSpec>>): IrDeclaration => {
  const names = Object.keys(initializers);
  return parseIrDeclaration(JSON.stringify({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["matmul"] },
    symbols: ["T"],
    inputs: [{ name: "x", dtype: "f32", shape: ["T", 4] }],
    outputs: names.map((_, i) => `y${i}`),
    initializers: Object.fromEntries(
      names.map((name) => [name, initializers[name].shared ? { shared: true } : {}]),
    ),
    values: {
      ...Object.fromEntries(
        names.map((name) => [name, { dtype: "f32", shape: initializers[name].shape }]),
      ),
      ...Object.fromEntries(names.map((_, i) => [`y${i}`, { dtype: "f32", shape: ["T", 4] }])),
    },
    nodes: names.map((name, i) => ({ op: "matmul", ins: ["x", name], outs: [`y${i}`], attrs: {} })),
  }));
};

const F32 = 4;

/** グラフ名は系列の部品名ではなく **family の weights キー**（`model`）を綴る。 */
const GRAPH = "model";

const syntheticModel = (): ModelInput => ({
  graphs: {
    [GRAPH]: declaration({
      "enc.weight": { shape: [8, 16] },
      "proj.weight": { shape: [6, 16] },
      "const.rope": { shape: [16] },
      "head.weight": { shape: [4, 4], shared: true },
    }),
  },
  consts: [{
    graph: GRAPH,
    initializer: "const.rope",
    bytes: bytesOf(16 * F32, 11),
    encoding: { codec: "f32" },
  }],
  weights: [
    {
      graph: GRAPH,
      initializer: "enc.weight",
      bytes: bytesOf(8 * 16 * 2, 21),
      encoding: { codec: "f16" },
    },
    {
      // i8 + companion scale（行長 16 の per-channel）。
      graph: GRAPH,
      initializer: "proj.weight",
      bytes: bytesOf(6 * 16, 22),
      encoding: {
        codec: "int8-sym",
        groupSize: 16,
        scale: { bytes: bytesOf(6 * F32, 23), dtype: "f32" },
      },
    },
  ],
  assets: [
    { name: "ple_index", role: "ple-index", bytes: bytesOf(200, 31) },
    { name: "style", role: "style-vectors", bytes: bytesOf(120, 32), dedicatedPart: true },
  ],
  provenance: { license: "apache-2.0", writer: "test" },
});

/** part 長 / block 上限を絞って、部品 1 本でも part が複数本になるようにする。 */
const OPTIONS = { partBytes: 320, blockBytes: 256 } as const;

const REPRESENTATIVE = "model.krm";

/** `BlockSource` の part 添字（0 始まり）→ 分割形のファイル名（連番は 1 始まり）。 */
const partName = (index: number, total: number): string =>
  `model-${String(index + 1).padStart(5, "0")}-of-${String(total).padStart(5, "0")}.krm`;

/** 0 バイトのファイルを置いた一時ディレクトリ（末尾 `/` 付きの URL を渡す）。 */
const withFiles = async (names: readonly string[], body: (dir: URL) => void): Promise<void> => {
  const path = await Deno.makeTempDir({ prefix: "karume-container-files-" });
  try {
    const dir = new URL(`file://${path}/`);
    for (const name of names) Deno.writeFileSync(new URL(name, dir), new Uint8Array(0));
    body(dir);
  } finally {
    await Deno.remove(path, { recursive: true });
  }
};

/** 合成の容器を分割形で置いた一時ディレクトリ。 */
const withContainer = async (
  body: (dir: URL, written: WrittenContainer) => Promise<void>,
): Promise<void> => {
  const written = await writeModelContainer(syntheticModel(), OPTIONS);
  const path = await Deno.makeTempDir({ prefix: "karume-container-files-" });
  try {
    const dir = new URL(`file://${path}/`);
    for (const [index, bytes] of written.parts.entries()) {
      Deno.writeFileSync(new URL(partName(index, written.parts.length), dir), bytes);
    }
    await body(dir, written);
  } finally {
    await Deno.remove(path, { recursive: true });
  }
};

// ---------------------------------------------------------------------------
// resolveParts / modelPresent（ファイルの見つけ方）
// ---------------------------------------------------------------------------

Deno.test("resolveParts は分割されていない容器を代表 path 1 本として返す", async () => {
  await withFiles([REPRESENTATIVE], (dir) => {
    const representative = new URL(REPRESENTATIVE, dir);
    assertEquals(resolveParts(representative).map((url) => url.href), [representative.href]);
  });
});

Deno.test("resolveParts は連番の part を番号昇順で返す（part 添字の保証）", async () => {
  // 置く順は番号順ではない（`Deno.readDirSync` の列挙順に依らないことの検出器）。
  const names = [
    "model-00003-of-00003.krm",
    "model-00001-of-00003.krm",
    "model-00002-of-00003.krm",
  ];
  await withFiles(names, (dir) => {
    const files = resolveParts(new URL(REPRESENTATIVE, dir)).map((url) => url.href);
    assertEquals(files, [...names].sort().map((name) => new URL(name, dir).href));
  });
});

Deno.test("resolveParts は単一形と連番の同居を fail loudly にする", async () => {
  await withFiles(
    [REPRESENTATIVE, "model-00001-of-00002.krm", "model-00002-of-00002.krm"],
    (dir) => {
      const error = assertThrows(() => resolveParts(new URL(REPRESENTATIVE, dir)), Error);
      assertStringIncludes(error.message, "同居している");
    },
  );
});

Deno.test("resolveParts は of の総数が食い違う連番を fail loudly にする", async () => {
  await withFiles(["model-00001-of-00003.krm", "model-00002-of-00004.krm"], (dir) => {
    const error = assertThrows(() => resolveParts(new URL(REPRESENTATIVE, dir)), Error);
    assertStringIncludes(error.message, "食い違っている");
  });
});

Deno.test("resolveParts は連番の欠番を何本目かを名指して fail loudly にする", async () => {
  await withFiles(["model-00001-of-00003.krm", "model-00003-of-00003.krm"], (dir) => {
    const error = assertThrows(() => resolveParts(new URL(REPRESENTATIVE, dir)), Error);
    assertStringIncludes(error.message, "2 本目が無い");
  });
});

Deno.test("resolveParts は総数からはみ出した番号を fail loudly にする", async () => {
  await withFiles(
    [
      "model-00001-of-00003.krm",
      "model-00002-of-00003.krm",
      "model-00003-of-00003.krm",
      "model-00004-of-00003.krm",
    ],
    (dir) => {
      const error = assertThrows(() => resolveParts(new URL(REPRESENTATIVE, dir)), Error);
      assertStringIncludes(error.message, "はみ出した番号 4");
    },
  );
});

// 0 以下の番号（1 始まりの連番の下側へのはみ出し）。上側だけを数えると「はみ出した番号 []」と
// 空リストで落ち、落ちてはいるのに何が余計なのかが読めない診断になる。
Deno.test("resolveParts は 0 番の part も余計な番号として名指す", async () => {
  await withFiles(
    [
      "model-00000-of-00002.krm",
      "model-00001-of-00002.krm",
      "model-00002-of-00002.krm",
    ],
    (dir) => {
      const error = assertThrows(() => resolveParts(new URL(REPRESENTATIVE, dir)), Error);
      assertStringIncludes(error.message, "はみ出した番号 0");
    },
  );
});

// stem は正規表現へ埋め込まれるので、escape が外れると `.` が任意 1 文字になり別 dtype の
// 連番を拾う（拾えば総数 {1,2} の食い違いで落ちるので、この門は「1 本だけ返る」ことで escape と
// stem 一致の両方を縛る）。
Deno.test("resolveParts は別 dtype の容器を拾わない（stem の厳密一致）", async () => {
  await withFiles(
    ["model.i8-00001-of-00001.krm", "modelXi8-00002-of-00002.krm", "other-00001-of-00002.krm"],
    (dir) => {
      const files = resolveParts(new URL("model.i8.krm", dir));
      assertEquals(files.map((url) => url.href), [
        new URL("model.i8-00001-of-00001.krm", dir).href,
      ]);
    },
  );
});

Deno.test("modelPresent は連番だけが置かれた容器を在りと見る", async () => {
  await withFiles(["model-00001-of-00002.krm", "model-00002-of-00002.krm"], (dir) => {
    assert(modelPresent(new URL(REPRESENTATIVE, dir)));
  });
});

Deno.test("modelPresent は容器が 1 本も無いディレクトリを無しと見る", async () => {
  await withFiles(["io.ramp.safetensors"], (dir) => {
    assertEquals(modelPresent(new URL(REPRESENTATIVE, dir)), false);
  });
});

// ---------------------------------------------------------------------------
// fileBlockSource（区間の読み方）
// ---------------------------------------------------------------------------

Deno.test("fileBlockSource は part の本数と長さをファイルから配る", async () => {
  await withContainer((dir, written) => {
    const source = fileBlockSource(resolveParts(new URL(REPRESENTATIVE, dir)));
    assertEquals(source.partCount, written.parts.length);
    assert(source.partCount >= 3, `part が ${source.partCount} 本では分割形の門にならない`);
    assertEquals(
      written.parts.map((_, index) => source.partLength(index)),
      written.parts.map((part) => part.byteLength),
    );
    return Promise.resolve();
  });
});

// MUST: 誰も検証していない取得元が `verified: true` を名乗ると、`openContainer` は block ごとの
// sha256 を掛けなくなる（container-v1 §7）= 改ざんが素通りする。
Deno.test("fileBlockSource は検証済みを名乗らない", async () => {
  await withContainer((dir) => {
    assertEquals(fileBlockSource(resolveParts(new URL(REPRESENTATIVE, dir))).verified, false);
    return Promise.resolve();
  });
});

Deno.test("fileBlockSource は part の途中から要求した区間だけを返す", async () => {
  await withContainer(async (dir, written) => {
    const source = fileBlockSource(resolveParts(new URL(REPRESENTATIVE, dir)));
    // 最後の part（資産の専用 part）の中ほどを取る — seek が効いていなければ先頭が返る。
    const part = written.parts.length - 1;
    const bytes = written.parts[part];
    const offset = Math.floor(bytes.byteLength / 3);
    const length = Math.min(17, bytes.byteLength - offset);
    assertEquals(
      await source.read(part, offset, length),
      bytes.subarray(offset, offset + length),
    );
  });
});

Deno.test("fileBlockSource は part 長をはみ出す区間を fail loudly にする", async () => {
  await withContainer(async (dir, written) => {
    const source = fileBlockSource(resolveParts(new URL(REPRESENTATIVE, dir)));
    const error = await assertRejects(
      () => source.read(0, written.parts[0].byteLength - 1, 2),
      Error,
    );
    assertStringIncludes(error.message, "part 長");
  });
});

Deno.test("fileBlockSource は存在しない part 添字を fail loudly にする", async () => {
  await withContainer(async (dir, written) => {
    const source = fileBlockSource(resolveParts(new URL(REPRESENTATIVE, dir)));
    const error = await assertRejects(() => source.read(written.parts.length, 0, 1), Error);
    assertStringIncludes(error.message, "は無い");
  });
});

// ---------------------------------------------------------------------------
// openSeriesContainer（part 列 → 開いた容器）
// ---------------------------------------------------------------------------

Deno.test("openSeriesContainer は分割形の part 列から容器を開く", async () => {
  await withContainer(async (dir) => {
    const opened = await openSeriesContainer(new URL(REPRESENTATIVE, dir));
    assertEquals(opened.header.kind, "model");
    assertEquals(Object.keys(opened.graphs), [GRAPH]);
    assertEquals(opened.source.verified, false);
  });
});

Deno.test("openSeriesContainer で開いた容器は block を宣言どおりのバイト列で配る", async () => {
  await withContainer(async (dir, written) => {
    const opened = await openSeriesContainer(new URL(REPRESENTATIVE, dir));
    for (const block of written.model.blocks) {
      assertEquals(
        await opened.readBlock(block.id),
        written.parts[block.part].subarray(block.offset, block.offset + block.length),
        `block '${block.id}'`,
      );
    }
  });
});

// 誤りの注入 — ファイル 1 バイトを潰すと block の sha256 が食い違う。ここが緑のままなら
// `verified: false` の経路が効いていない（= 区間読みの取得元で検証が抜けている）。
Deno.test("openSeriesContainer は改ざんされた part を block の sha256 で落とす", async () => {
  await withContainer(async (dir, written) => {
    const block = written.model.blocks[0];
    const file = new URL(partName(block.part, written.parts.length), dir);
    const bytes = Deno.readFileSync(file);
    bytes[block.offset] ^= 0xff;
    Deno.writeFileSync(file, bytes);
    const opened = await openSeriesContainer(new URL(REPRESENTATIVE, dir));
    const error = await assertRejects(() => opened.readBlock(block.id), Error);
    assertStringIncludes(error.message, "sha256");
  });
});

Deno.test("openSeriesContainer で開いた容器は資産を区間で読める", async () => {
  await withContainer(async (dir) => {
    const opened = await openSeriesContainer(new URL(REPRESENTATIVE, dir));
    const reader = opened.asset("ple_index");
    assertEquals(reader.role, "ple-index");
    assertEquals(reader.length, 200);
    const whole = syntheticModel().assets[0].bytes;
    assertEquals(await reader.read(0, reader.length), whole);
    assertEquals(await reader.read(64, 8), whole.subarray(64, 72));
  });
});

Deno.test("openSeriesContainer は part が 1 本欠けた系列を fail loudly にする", async () => {
  await withContainer(async (dir, written) => {
    await Deno.remove(new URL(partName(1, written.parts.length), dir));
    const error = await assertRejects(
      () => openSeriesContainer(new URL(REPRESENTATIVE, dir)),
      Error,
    );
    assertStringIncludes(error.message, "2 本目が無い");
  });
});
