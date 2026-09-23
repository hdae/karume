// 系列 → グラフ名の表（helpers/series-graphs.ts）の引き口の門。GPU も実資産も要らない。
//
// 表を 1 箇所に集めた目的は「門番と e2e が別々の表を持たない」ことだが、引き口が黙って
// `undefined` を返すなら、集めた意味が半分消える —— 呼び手は `prepareContainer` へ
// `undefined` を渡すか、既定値で**別のグラフ**を開く。どちらも綴り違いを数値で拾えない形
// （`prepareContainer` は名前が違えば落ちるが、既定値で他のグラフが実在すればそのまま通る）。
// ここは引き口が綴り違いをその場で落とすことを固定する。
//
// 表の中身が実物と合っているかは `assets_gate_test.ts` が実資産で見る（この門の射程外）。

import { assertEquals, assertThrows } from "@std/assert";
import { GRAPH_NAME_PATTERN } from "../src/format/container/limits.ts";
import { SERIES_GRAPHS, seriesComponents, seriesGraph } from "./helpers/series-graphs.ts";

Deno.test("seriesGraph は表どおりのグラフ名を返す（根直下と部品ディレクトリの両方）", () => {
  assertEquals(seriesGraph("gemma4-e2b-product"), "model");
  assertEquals(seriesGraph("depth-anything-v2-small-hf"), "depth");
  // ディレクトリ名と綴りが割れる代表 2 つ（この割れが表を要求している理由そのもの）。
  assertEquals(seriesGraph("dacvae-32dim", "decoder"), "codec_decoder");
  assertEquals(seriesGraph("irodori-v4-small", "caption-proj"), "caption_proj");
});

Deno.test("seriesGraph は表に無い系列を名指しで落とす", () => {
  const error = assertThrows(
    () => seriesGraph("gemma4-e2b-produkt"),
    Error,
    "gemma4-e2b-produkt",
  );
  // 在る綴りを一緒に出す（打ち間違いは一覧が無いと直せない）。
  assertEquals(error.message.includes("gemma4-e2b-product"), true, error.message);
});

Deno.test("seriesGraph は表に在る系列でも、表に無い部品を落とす", () => {
  const error = assertThrows(
    () => seriesGraph("dacvae-32dim", "decoders"),
    Error,
    "decoders",
  );
  assertEquals(error.message.includes("decoder"), true, error.message);
});

Deno.test("seriesGraph は部品の有無を取り違えた引き方を落とす", () => {
  // 部品ディレクトリを持つ系列を根直下として引く（`""` は表に無い）。
  assertThrows(() => seriesGraph("dacvae-32dim"), Error, "<根直下>");
  // 根直下の系列に部品名を渡す（在る部品として `<根直下>` を挙げる）。
  assertThrows(() => seriesGraph("gemma4-e2b-product", "model"), Error, "<根直下>");
});

Deno.test("seriesComponents は表に無い系列を落とし、在る系列では部品の表をそのまま配る", () => {
  assertThrows(() => seriesComponents("sbv2-F9"), Error, "sbv2-F9");
  assertEquals(Object.keys(seriesComponents("deberta-i8")).sort(), [
    "full-24layer",
    "sbv2-22layer",
  ]);
});

Deno.test("表のグラフ名はすべて容器のグラフ名語彙に収まる", () => {
  // `codec decoder` のような空白混じりの綴りは `prepareContainer` まで行かないと落ちない
  // （そこまで行くには実資産が要る）。表を足した時点で落ちる形にしておく。
  const outside = Object.entries(SERIES_GRAPHS).flatMap(([series, components]) =>
    Object.entries(components)
      .filter(([, graph]) => !GRAPH_NAME_PATTERN.test(graph))
      .map(([component, graph]) => `${series}/${component}: ${graph}`)
  );
  assertEquals(outside, [], `グラフ名の語彙（${GRAPH_NAME_PATTERN}）から外れた行`);
});
