// スタイル / 話者の**表引き**（`src/sbv2/style.ts`）のふるまい。合成の小さな表だけで書き、
// 実資産にも GPU にも依存しない。
//
// ここが沈黙誤値の巣であることが設計の前提（ADR 0039 決定 4）: スタイル ID も話者 ID も
// **表の物理行そのもの**なので、1 行ずれても shape は合ったままロードも実行も通り、別の
// スタイル・別の話者の声が出るだけで例外が出ない。固定するふるまいは 3 つ:
//
//  ① 行数・列数は**現物の表**が決める（実装に焼き込まない）
//  ② 混合式 `mean + (picked − mean)·weight` の `mean` は**行 0**（列ごとの平均ではない）
//  ③ 表と行番号・weight の食い違いは fail loudly
//
// NOTE: 「pipelineConfig の `styles` 件数と表の行数が一致すること」の突合だけはここに書けない
// — 検査（`pipeline.ts` の `assertTableFits`）がグラフ入力の静的次元を要求するため、実資産と
// GPU が要る。そちらは `e2e_sbv2_wav_test.ts` が持つ。

import { assert, assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import {
  parseSbv2Table,
  SPEAKER_TENSOR,
  speakerEmbedding,
  STYLE_TENSOR,
  styleVector,
} from "../src/sbv2/style.ts";
import { buildSafetensors, f32Bytes } from "./helpers/safetensors.ts";

const COLS = 3;
/**
 * 4 スタイル × 3 列の合成表。**行 0 が平均スタイル**（SBV2 の規約）で、列ごとの平均
 * （1.5 / 7 / 14.5）とは**わざと**違えてある — 「平均を列から取り直す」実装を混合式の
 * 故障注入で弁別するため。
 */
const STYLE_ROWS: readonly (readonly number[])[] = [
  [0, 10, 20],
  [2, 14, 26],
  [-4, 6, 12],
  [8, -2, 0],
];
const STYLE_TABLE = Float32Array.from(STYLE_ROWS.flat());

/** 2 話者 × 4 列（スタイル表と列数を違えて、取り違えを検出できる形にする）。 */
const SPEAKER_ROWS: readonly (readonly number[])[] = [
  [1, 2, 3, 4],
  [-1, -2, -3, -4],
];
const SPEAKER_TABLE = Float32Array.from(SPEAKER_ROWS.flat());

const styleTableAsset = (
  rows: readonly (readonly number[])[],
  name: string = STYLE_TENSOR,
): ArrayBuffer =>
  buildSafetensors([{
    name,
    dtype: "F32",
    shape: [rows.length, rows[0].length],
    data: f32Bytes(rows.flat()),
  }]);

Deno.test("スタイル表の解釈: 行数・列数は現物の safetensors が決める", async (t) => {
  await t.step("shape をそのまま [行, 列] として読み、値は行優先で並ぶ", () => {
    const table = parseSbv2Table(styleTableAsset(STYLE_ROWS), STYLE_TENSOR);
    assertEquals(table.rows, 4);
    assertEquals(table.cols, COLS);
    assertEquals([...table.data], STYLE_ROWS.flat());
  });

  await t.step("行数の違う表を渡せば読み取り結果も追随する（表を焼き込んでいない）", () => {
    // ckpt が変わればスタイルの数も並びも変わる（ADR 0039 決定 3）。件数を実装側に持つと
    // 「shape は合ったまま別のスタイル」になるので、現物だけが正であることを固定する。
    const table = parseSbv2Table(styleTableAsset(STYLE_ROWS.slice(0, 2)), STYLE_TENSOR);
    assertEquals(table.rows, 2);
    assertEquals(table.cols, COLS);
    assertEquals([...table.data], STYLE_ROWS.slice(0, 2).flat());
  });

  await t.step("話者表も同じ読み手で引ける（テンソル名だけが違う）", () => {
    const table = parseSbv2Table(styleTableAsset(SPEAKER_ROWS, SPEAKER_TENSOR), SPEAKER_TENSOR);
    assertEquals([table.rows, table.cols], [2, 4]);
    assertEquals([...table.data], SPEAKER_ROWS.flat());
  });

  await t.step("要求したテンソルが無ければ、入っているものを添えて落ちる", () => {
    // 資産の取り違え（style の資産を speaker として読む等）は名前でしか検出できない。
    assertThrows(
      () => parseSbv2Table(styleTableAsset(STYLE_ROWS), SPEAKER_TENSOR),
      Error,
      `資産にテンソル '${SPEAKER_TENSOR}' が無い`,
    );
  });

  await t.step("F32 でない表は落ちる（バイト解釈がずれれば全ての行が別物）", () => {
    const buffer = buildSafetensors([{
      name: STYLE_TENSOR,
      dtype: "F16",
      shape: [4, COLS],
      data: new Uint8Array(4 * COLS * 2),
    }]);
    assertThrows(() => parseSbv2Table(buffer, STYLE_TENSOR), Error, "F32 でない");
  });

  await t.step("2 次元でない表は落ちる（行の切り出し幅が決まらない）", () => {
    const buffer = buildSafetensors([{
      name: STYLE_TENSOR,
      dtype: "F32",
      shape: [4 * COLS],
      data: f32Bytes(STYLE_TABLE),
    }]);
    assertThrows(() => parseSbv2Table(buffer, STYLE_TENSOR), Error, "2 次元でない");
  });
});

Deno.test("スタイル混合 mean + (picked − mean)·weight: mean は行 0", async (t) => {
  const mix = (
    index: number,
    weight: number,
  ): number[] => [...styleVector(STYLE_TABLE, STYLE_ROWS.length, COLS, index, weight).data];

  await t.step("weight 1 は選んだ行そのもの（[1, cols] の f32 テンソル）", () => {
    const tensor = styleVector(STYLE_TABLE, STYLE_ROWS.length, COLS, 2, 1);
    assertEquals(tensor.dtype, "f32");
    assertEquals(tensor.shape, [1, COLS]);
    assertEquals([...tensor.data], STYLE_ROWS[2]);
  });

  await t.step("weight 0 はどの行を選んでも平均行（行 0）", () => {
    for (let index = 0; index < STYLE_ROWS.length; index += 1) {
      assertEquals(mix(index, 0), STYLE_ROWS[0], `行 ${index}`);
    }
  });

  await t.step("中間の weight は行 0 と選んだ行の線形補間", () => {
    // 行 0 [0,10,20] と行 1 [2,14,26] の中点。
    assertEquals(mix(1, 0.5), [1, 12, 23]);
    // 負の行（行 0 から見て逆側）も同じ式で出る。
    assertEquals(mix(2, 0.5), [-2, 8, 16]);
  });

  await t.step("weight > 1 は外挿する（強さの上限を実装側で握らない）", () => {
    assertEquals(mix(1, 2), [4, 18, 32]);
  });

  await t.step("故障注入: 平均を列ごとに取り直す実装とは別の値になる", () => {
    // 列平均は [1.5, 7, 14.5]。行 0 を平均と読む実装との差はここでしか出ない
    // （weight 1 と weight 0 の両端では両実装が一致してしまう）。
    const columnMeans = Array.from(
      { length: COLS },
      (_, col) => STYLE_ROWS.reduce((sum, row) => sum + row[col], 0) / STYLE_ROWS.length,
    );
    const byColumnMean = columnMeans.map((mean, col) => mean + (STYLE_ROWS[1][col] - mean) * 0.5);
    assertEquals(mix(1, 0.5), [1, 12, 23]);
    assertNotEquals(mix(1, 0.5), byColumnMean);
  });

  await t.step("行 0 自身を選ぶと weight に関わらず行 0（picked = mean）", () => {
    for (const weight of [0, 0.3, 1, 2]) {
      assertEquals(mix(0, weight), STYLE_ROWS[0], `weight ${weight}`);
    }
  });
});

Deno.test("話者埋め込み: 表から行を引いて [1, cols, 1] にする", async (t) => {
  const pick = (index: number) => speakerEmbedding(SPEAKER_TABLE, SPEAKER_ROWS.length, 4, index);

  await t.step("選んだ行をそのまま返す（front / voice が受ける g の形）", () => {
    const tensor = pick(1);
    assertEquals(tensor.dtype, "f32");
    assertEquals(tensor.shape, [1, 4, 1]);
    assertEquals([...tensor.data], SPEAKER_ROWS[1]);
  });

  await t.step("行が 1 つずれれば別の声になる（shape は合ったまま）", () => {
    const [first, second] = [pick(0), pick(1)];
    assertEquals(first.shape, second.shape, "形では検出できない");
    assert(
      [...first.data].some((value, at) => value !== second.data[at]),
      "行を変えても同じベクトルが出ている",
    );
  });

  await t.step("表の写しを返す（返り値を書き換えても表は不変）", () => {
    // 表はパイプラインの寿命ぶん使い回されるので、view を返すと 1 回の生成が以降の生成を汚す。
    const tensor = pick(0);
    tensor.data[0] = 999;
    assertEquals([...SPEAKER_TABLE], SPEAKER_ROWS.flat());
  });
});

Deno.test("行番号・weight・表の食い違いは fail loudly", async (t) => {
  const rows = STYLE_ROWS.length;

  await t.step("範囲外の行番号は落ちる（スタイル・話者とも）", () => {
    assertThrows(
      () => styleVector(STYLE_TABLE, rows, COLS, rows, 1),
      Error,
      `行番号 ${rows} が表の範囲外（0..${rows - 1}）`,
    );
    assertThrows(() => styleVector(STYLE_TABLE, rows, COLS, -1, 1), Error, "表の範囲外");
    assertThrows(
      () => speakerEmbedding(SPEAKER_TABLE, SPEAKER_ROWS.length, 4, 2),
      Error,
      "表の範囲外",
    );
  });

  await t.step("非整数の行番号は落ちる（切り出し位置が行の途中になる）", () => {
    assertThrows(() => styleVector(STYLE_TABLE, rows, COLS, 1.5, 1), Error, "表の範囲外");
  });

  await t.step("要素数が 行×列 と違う表は落ちる", () => {
    // 行数だけを取り違えると、範囲内の行番号でも表の外を読む。
    assertThrows(
      () => styleVector(STYLE_TABLE, rows + 1, COLS, 4, 1),
      Error,
      `表の要素数 ${STYLE_TABLE.length} が ${rows + 1}×${COLS} と違う`,
    );
  });

  await t.step("非有限の weight は落ちる（NaN が波形まで伝播する）", () => {
    for (const weight of [Number.NaN, Number.POSITIVE_INFINITY]) {
      assertThrows(
        () => styleVector(STYLE_TABLE, rows, COLS, 1, weight),
        Error,
        "有限の数でない",
      );
    }
  });
});
