// `helpers/shard-files.ts` の shard 解決（ADR 0081）の門。GPU も実資産も要らない純ロジック。
//
// このヘルパを使う 20 本超のテストは**正常な実資産しか渡さない**ので、fail-loudly の 4 分岐
// （単一と連番の同居 / `of` の食い違い / 欠番 / はみ出し）は 1 度も踏まれていなかった。門が
// 壊れると「前回の書き出しの残骸を今回の期待値で読む」事故が無音で戻る。
//
// 鏡像の Python 側（`tools/exporter/tests/test_shards.py`）は同居 / 欠番 / 総数食い違い /
// 別コンポーネントの shard を拾わない、の 4 形を持つ。TS 側だけ無検査だったので同じ形を置く
// （「はみ出し」は Python 側に無く、こちらが先行する）。

import { assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import { resolveShards } from "./helpers/shard-files.ts";

/** 0 バイトのファイルを置いた一時ディレクトリ（末尾 `/` 付きの URL を返す）。 */
const withFiles = async (
  names: readonly string[],
  body: (dir: URL) => void,
): Promise<void> => {
  const path = await Deno.makeTempDir({ prefix: "karume-shard-files-" });
  try {
    const dir = new URL(`file://${path}/`);
    for (const name of names) Deno.writeFileSync(new URL(name, dir), new Uint8Array(0));
    body(dir);
  } finally {
    await Deno.remove(path, { recursive: true });
  }
};

const REPRESENTATIVE = "model.safetensors";

Deno.test("resolveShards は分割されていない資産を代表 path 1 本として返す", async () => {
  await withFiles([REPRESENTATIVE], (dir) => {
    const representative = new URL(REPRESENTATIVE, dir);
    assertEquals(resolveShards(representative).map((url) => url.href), [representative.href]);
  });
});

Deno.test("resolveShards は連番の shard を番号昇順で返す（読む順の保証）", async () => {
  // 置く順は番号順ではない（`Deno.readDirSync` の列挙順に依らないことの検出器）。
  const names = [
    "model-00003-of-00003.safetensors",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
  ];
  await withFiles(names, (dir) => {
    const files = resolveShards(new URL(REPRESENTATIVE, dir)).map((url) => url.href);
    assertEquals(files, [...names].sort().map((name) => new URL(name, dir).href));
  });
});

Deno.test("resolveShards は単一ファイルと連番の同居を fail loudly にする", async () => {
  await withFiles(
    [REPRESENTATIVE, "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
    (dir) => {
      const error = assertThrows(() => resolveShards(new URL(REPRESENTATIVE, dir)), Error);
      assertStringIncludes(error.message, "同居している");
    },
  );
});

Deno.test("resolveShards は of の総数が食い違う連番を fail loudly にする", async () => {
  await withFiles(
    ["model-00001-of-00003.safetensors", "model-00002-of-00004.safetensors"],
    (dir) => {
      const error = assertThrows(() => resolveShards(new URL(REPRESENTATIVE, dir)), Error);
      assertStringIncludes(error.message, "食い違っている");
    },
  );
});

Deno.test("resolveShards は連番の欠番を何本目かを名指して fail loudly にする", async () => {
  await withFiles(
    ["model-00001-of-00003.safetensors", "model-00003-of-00003.safetensors"],
    (dir) => {
      const error = assertThrows(() => resolveShards(new URL(REPRESENTATIVE, dir)), Error);
      assertStringIncludes(error.message, "2 本目が無い");
    },
  );
});

Deno.test("resolveShards は総数からはみ出した番号を fail loudly にする", async () => {
  await withFiles(
    [
      "model-00001-of-00003.safetensors",
      "model-00002-of-00003.safetensors",
      "model-00003-of-00003.safetensors",
      "model-00004-of-00003.safetensors",
    ],
    (dir) => {
      const error = assertThrows(() => resolveShards(new URL(REPRESENTATIVE, dir)), Error);
      assertStringIncludes(error.message, "はみ出した番号 4");
    },
  );
});

// stem は正規表現へ埋め込まれるので、escape が外れると `.` が任意 1 文字になり別
// コンポーネントの連番を拾う（拾えば総数 {1,2} の食い違いで落ちるので、この門は
// 「1 本だけ返る」ことで escape と stem 一致の両方を縛る）。
Deno.test("resolveShards は別コンポーネントの shard を拾わない（stem の厳密一致）", async () => {
  await withFiles(
    [
      "model.i8-00001-of-00001.safetensors",
      "modelXi8-00002-of-00002.safetensors",
      "other-00001-of-00002.safetensors",
    ],
    (dir) => {
      const files = resolveShards(new URL("model.i8.safetensors", dir));
      assertEquals(files.map((url) => url.href), [
        new URL("model.i8-00001-of-00001.safetensors", dir).href,
      ]);
    },
  );
});
