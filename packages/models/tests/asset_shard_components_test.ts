/**
 * 全量面（`from*Assets`）の受け口（`src/hub/components.ts` の `assetComponentOpener`）の門の
 * うち、GPU を取らない受理側の 2 点（Session を張る 2 点は
 * gpu_asset_shard_components_test.ts）:
 *
 * ③ **添字の欠番は fail loudly**（`[0]` と `[2]` だけの Record を黙って 1 本で読まない）。
 *    `[0]` が**無い**形（`dit[1]` だけ / 素キー + `dit[1]`）も同じ門で落ちる — 添字つきキーが
 *    1 本でもあるのに `[0]` が無い並びは、以前は混在検査も欠番検査も飛ばして素の 1 本になった。
 * ④ **素キーと `[i]` の混在は fail loudly**（どちらを正とするかは決められない）。
 *
 * ③④ の診断は `shard` の語と**揃っているキーの列挙**を含むこと（既存の資産診断の流儀）まで
 * 見る — 落ちること自体は取得キーの作り方が壊れている印で、読み手が現物を突き合わせられる
 * 形でないと意味が無い。
 */

import { assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import { graphShard, openerOf, weightShard, wholeShard } from "./helpers/asset-shard-fixture.ts";

Deno.test("assetComponentOpener: 添字の欠番は shard を名乗って落ちる", () => {
  const open = openerOf({ "dit[0]": graphShard(), "dit[2]": weightShard() });
  const error = assertThrows(() => open("dit"), Error);
  assertStringIncludes(error.message, "shard 添字が [0] から連続していない");
  // 揃っているキーを列挙する（読み手が現物と突き合わせられる形 — 既存の資産診断の流儀）。
  assertStringIncludes(error.message, "dit[0] / dit[2]");
});

Deno.test("assetComponentOpener: 素キーと分割キーの混在は shard を名乗って落ちる", () => {
  const open = openerOf({ dit: wholeShard(), "dit[0]": graphShard(), "dit[1]": weightShard() });
  const error = assertThrows(() => open("dit"), Error);
  assertStringIncludes(error.message, "素のキーと shard 分割キー");
  assertStringIncludes(error.message, "dit / dit[0] / dit[1]");
});

Deno.test("assetComponentOpener: 素キーと [0] 以外の分割キーの混在も落ちる（[0] 欠落で門を素通りしない）", () => {
  // `assetShardKeys` は `[0]` から連続する範囲だけを拾うので、`[0]` が無いと混在検査を飛ばして
  // 「素の 1 本」として組んでしまう形があった（`dit[1]` を黙って無視する = 分割配布のつもりで
  // 組んだ Record が遠くの層から「重みが足りない」で落ちる）。
  const open = openerOf({ dit: wholeShard(), "dit[1]": weightShard() });
  const error = assertThrows(() => open("dit"), Error);
  assertStringIncludes(error.message, "素のキーと shard 分割キー");
  assertStringIncludes(error.message, "dit / dit[1]");
});

Deno.test("assetComponentOpener: 添字が [0] から始まらない列は始点を名指しして落ちる", () => {
  const open = openerOf({ "dit[1]": graphShard(), "dit[2]": weightShard() });
  const error = assertThrows(() => open("dit"), Error);
  assertStringIncludes(error.message, "shard 添字が [0] から始まっていない");
  assertStringIncludes(error.message, "dit[1] / dit[2]");
});

Deno.test("assetComponentOpener: [0] から連続する列は従来どおり通る（上 2 件が恒真でないこと）", () => {
  const open = openerOf({ "dit[0]": graphShard(), "dit[1]": weightShard() });
  assertEquals(open("dit").graph.outputs, ["y"]);
});

Deno.test("assetComponentOpener: キーがどちらの形でも無ければ家族の診断のまま落ちる", () => {
  const open = openerOf({ "dit[0]": graphShard() });
  const error = assertThrows(() => open("vae"), Error);
  assertStringIncludes(error.message, "資産 'vae' が無い");
});
