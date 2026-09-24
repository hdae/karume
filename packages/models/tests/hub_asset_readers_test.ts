// 取得済み資産バイト列の読み口（`src/hub/asset-readers.ts`）。7 family が同じ 1 本を通るので、
// ここが受理集合を広げると全 family の資産門が一斉に緩む。
//
// 見るのは 5 点（最後の 1 点は容器の内側の資産 — `readWholeAsset`）:
//
// ① **写さない**こと。返るのは view の `buffer` そのもの（GB 級の重みで RAM ピークが倍に
//    ならない条件）で、同一性で縛れる。
// ② buffer 全体を占めない view は落とす。通すと `openContainer` は隣のバイト列をヘッダとして読む。
// ③ 文言のラベル（family 名と manifest の表）は呼び手が渡したものが逐語で出る — 各 family の
//    テストが文言を逐語 assert しているので、ここが落とすと 5 本が一斉に赤になる。
// ④ decode 段（不正 UTF-8）と parse 段（JSON 構文違反）を別の文言で落とす。

import { assertEquals, assertStrictEquals, assertThrows } from "@std/assert";
import { openContainer } from "@karume/runtime";
import { readAssetBuffer, readAssetJson, readWholeAsset } from "../src/hub/asset-readers.ts";
import { tensorlessContainer } from "./helpers/container-fixture.ts";

/** buffer 全体を占める view（hub が返す契約どおりの形）。 */
const wholeView = (bytes: readonly number[]): Uint8Array<ArrayBuffer> => Uint8Array.from(bytes);

Deno.test("readAssetBuffer: 返るのは view の buffer そのもの（1 バイトも写さない）", () => {
  const bytes = wholeView([1, 2, 3, 4]);
  assertStrictEquals(
    readAssetBuffer("sbv2", "weights / assets", { model: bytes }, "model"),
    bytes.buffer,
  );
});

Deno.test("readAssetBuffer: 資産が無ければ落とし、文言に family・表・揃っているキーが入る", () => {
  assertThrows(
    () => readAssetBuffer("sbv2", "weights / assets", { front: wholeView([0]) }, "voice"),
    Error,
    "sbv2: 資産 'voice' が無い（manifest の weights / assets に voice が要る）" +
      "（揃っているキー: front）",
  );
});

Deno.test("readAssetBuffer: manifest の表は呼び手のラベルがそのまま出る（family ごとに違う）", () => {
  assertThrows(
    () => readAssetBuffer("birefnet", "weights", {}, "matte"),
    Error,
    "birefnet: 資産 'matte' が無い（manifest の weights に matte が要る）（揃っているキー: ）",
  );
});

Deno.test("readAssetBuffer: 資産の有無は Object.hasOwn で見る（prototype の値を拾わない）", () => {
  // `assets["toString"]` は Object.prototype 由来の関数を返す。`in` で見ていると、そこから
  // `byteOffset` を読もうとして「資産が無い」とは別の場所で落ちる。
  assertThrows(
    () => readAssetBuffer("anima", "weights / assets", {}, "toString"),
    Error,
    "anima: 資産 'toString' が無い",
  );
});

Deno.test("readAssetBuffer: buffer 全体を占めない view は取得層の不変条件破れとして落ちる", () => {
  // 先頭を 1 バイト飛ばした view。返してしまうと `openContainer` は隣のバイト列をヘッダに読む。
  const backing = wholeView([1, 2, 3, 4]);
  const shifted = new Uint8Array(backing.buffer, 1, 2);
  assertThrows(
    () => readAssetBuffer("irodori", "weights / assets", { dit: shifted }, "dit"),
    Error,
    "irodori: 資産 'dit' の bytes が buffer 全体を占めていない（byteOffset 1 / byteLength 2 /" +
      " buffer 4）",
  );
});

/** JSON としては閉じているが、文字列値に不正 UTF-8（0xff）を 1 バイト混ぜた資産。 */
const brokenUtf8Asset = (): Uint8Array<ArrayBuffer> =>
  Uint8Array.from([
    ...new TextEncoder().encode('{"unkId":'),
    0x22,
    0xff,
    0x22,
    ...new TextEncoder().encode("}"),
  ]);

Deno.test("readAssetJson: 不正 UTF-8 の資産は decode 段の文言で落ちる（JSON 段まで進まない）", () => {
  const broken = brokenUtf8Asset();
  // 前提の固定: 既定の TextDecoder は 0xff を U+FFFD へ置換するので、置換して読むと
  // **内容の違う valid JSON** が黙って通ってしまう。塞いだのはこの経路。
  assertEquals(
    JSON.parse(new TextDecoder().decode(broken)),
    { unkId: "�" },
    "置換 decode が valid JSON にならない前提が崩れた（このテストの主題が消える）",
  );
  assertThrows(
    () => readAssetJson("sbv2", "weights / assets", { tokenizer: broken }, "tokenizer"),
    Error,
    "sbv2: 資産 'tokenizer' が UTF-8 として読めない",
  );
});

Deno.test("readAssetJson: JSON 構文違反は decode とは別の文言で落ちる（正常域は不変）", () => {
  const truncated = new TextEncoder().encode('{"unkId":');
  assertThrows(
    () => readAssetJson("irodori", "weights / assets", { tokenizer: truncated }, "tokenizer"),
    Error,
    "irodori: 資産 'tokenizer' が JSON として読めない",
  );
  // 正しい UTF-8 は多バイト文字を含んでもそのまま通る。
  const valid = new TextEncoder().encode('{"unkId":0,"space":"あ"}');
  assertEquals(
    readAssetJson("irodori", "weights / assets", { tokenizer: valid }, "tokenizer"),
    { unkId: 0, space: "あ" },
  );
});

Deno.test("readAssetJson: 資産そのものが無い場合は buffer 段の文言で落ちる", () => {
  assertThrows(
    () => readAssetJson("sbv2", "weights / assets", {}, "symbols"),
    Error,
    "sbv2: 資産 'symbols' が無い（manifest の weights / assets に symbols が要る）",
  );
});

// ---- 容器の資産（`readWholeAsset`）------------------------------------------
//
// 容器の内側の資産（`rope_base` / PLE の索引）は manifest の表に載らないので、読み口は
// `AssetReader` 1 本。見るのは 2 点: **論理長ぶんだけ**返ること（block の詰め物を渡さない）と、
// 返る ArrayBuffer が payload だけを占めること（`parseSafetensors` がそのまま読める形）。

Deno.test("readWholeAsset: 論理長ぶんだけ読み、詰め物を含まない buffer を返す", async () => {
  // 長さ 6 の資産（block 長は 4 の倍数へ切り上がるので 8 バイト = 詰め物 2 バイト）。
  const payload = Uint8Array.from([1, 2, 3, 4, 5, 6]) as Uint8Array<ArrayBuffer>;
  const written = await tensorlessContainer(
    "unit",
    { inputs: [{ name: "x", shape: [1, 2] }], output: { name: "y", shape: [1, 2] } },
    [{ name: "probe", role: "test-probe", bytes: payload }],
  );
  const opened = await openContainer({ kind: "bytes", bytes: written.single });
  const reader = opened.asset("probe");
  assertEquals(reader.role, "test-probe");
  assertEquals(reader.length, payload.byteLength);

  const buffer = await readWholeAsset(reader);
  assertEquals(buffer.byteLength, payload.byteLength, "詰め物まで返している");
  assertEquals([...new Uint8Array(buffer)], [...payload]);
});
