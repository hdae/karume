// linear の GEMV 族・**行ブロック変種の選択**（ADR 0082 追記 5 / perf-ledger K-21）の純関数門。
// GPU を要らない（生成物とキーだけを見る）。
//
// 行ブロックの高さ `rows` は **(格納, m, n, 並列度の目標) の純関数** `linearGemvRowsForShape` が
// 決める（目標は Session 生成時に固定される静的なノブで、省略時は参照 device の飽和点 16384）。
// この純関数性が「同一キー → バイト同一 WGSL」と実行時オートチューン禁止（ADR 0022）の
// 両方を担保しているので、見るのは 3 つ:
//
// 1. **選択表そのもの** — 掃引（RTX 3080 Ti・docs/research/2026-09-07-gemv-rows-k21.md）で採った (格納, m, n) → rows の
//    対応を固定する。MUST: 表は**実装の写し**であって独立な正本ではない — 掃引をやり直して
//    実装を変えるときは、この表も実測と一緒に差し替える（ここが赤いだけでは実装の欠陥の証明に
//    ならない）。逆に、実装を触っていないのにここが赤くなれば選択が状態を持った証拠になる。
// 2. **fail loudly の端** — 門の外（m=0 / m > 上限 / n=0 / 非整数）は黙って丸めずに落ちる。
//    丸めると「門が閉じているはずの形が別の rows で走る」= 沈黙誤値の入口になる。
// 3. **キーに `rows` が載る** — 載っていないと rows 4 の資産が rows 16 の dispatch に配られ、
//    アキュムレータ本数と y タイルの高さが食い違って即座に誤値になる。
//
// 数値のビット同一（既定 GEMM 経路との u32 完全一致）は実 GPU 側 = tests/gpu_linear_gemv_test.ts。

import { assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import { CodegenError } from "../src/codegen/errors.ts";
import {
  defaultLinearGemvRowsVariant,
  defaultLinearGemvVariant,
  LINEAR_GEMV_MAX_ROWS,
  linearGemvKey,
  linearGemvParams,
  linearGemvRowsForShape,
  linearGemvRowsKey,
  linearGemvRowsWgsl,
} from "../src/kernels/linear-gemv.ts";
import type { WeightStorage } from "../src/kernels/weight-storage.ts";

/** 選択表 1 行（1 つの (格納, n) に対する M → rows の対応）。 */
type RowsTable = {
  readonly storage: WeightStorage;
  readonly n: number;
  /** 何の形か（census のどの層に当たるか）。 */
  readonly note: string;
  /** `[m, rows]` の対。M は 1 と上限 64 を必ず含める。 */
  readonly picks: readonly (readonly [m: number, rows: number])[];
};

/**
 * (格納, m, n) → rows の表。
 *
 * MUST: **n の桁を跨いで並べる** — 選択は「スレッド数 = n × ceil(m / rows) が目標 16384 に
 * 届くまで rows を半分にする」形なので、n が小さい層（常に rows=1）と大きい層（天井まで伸びる）を
 * 両方置かないと、目標値の側と天井の側のどちらが効いているのか区別できない。
 * MUST: 格納 2 種を置く — 天井（1 スレッドが 1 語あたり抱える要素数 512）が rows へ効く形は
 * 格納で違う（i4 は 1 語 32 要素 → 8 行まで・i8 は 16 要素 → 16 行まで）。
 */
const ROWS_TABLES: readonly RowsTable[] = [
  {
    storage: "i4",
    n: 256,
    note: "小 n（gemma4 E2B の最小層）— スレッド数が目標に届かず全 M で rows=1",
    picks: [[1, 1], [2, 1], [4, 1], [7, 1], [8, 1], [16, 1], [32, 1], [64, 1]],
  },
  {
    storage: "i4",
    n: 2048,
    note: "中 n — M が増えて初めて目標に届き、rows が段階的に立つ",
    picks: [[1, 1], [2, 1], [4, 1], [7, 1], [8, 1], [16, 2], [32, 4], [64, 8]],
  },
  {
    storage: "i4",
    n: 4096,
    note: "中 n（2048 の倍）— 同じ M で rows がちょうど 1 段深くなり、M=64 で i4 の天井 8 に当たる",
    picks: [[1, 1], [2, 1], [4, 1], [7, 2], [8, 2], [16, 4], [32, 8], [64, 8]],
  },
  {
    storage: "i4",
    n: 12288,
    note: "大 n — M=16 で i4 の天井 8 行に当たり、M=32 / 64 でも 8 のまま伸びない",
    picks: [[1, 1], [2, 1], [4, 2], [7, 4], [8, 4], [16, 8], [32, 8], [64, 8]],
  },
  {
    storage: "i8",
    n: 262144,
    note: "lm_head（i8）— どの M でも目標に届くので rows は M そのもの、天井 16 で頭打ち",
    picks: [[1, 1], [2, 2], [4, 4], [7, 4], [8, 8], [16, 16], [32, 16], [64, 16]],
  },
  {
    storage: "i4",
    n: 1,
    note: "退化（列 1 本）— 目標に届きようがないので rows=1",
    picks: [[1, 1], [2, 1], [64, 1]],
  },
  {
    storage: "i8",
    n: 1,
    note: "退化（列 1 本・i8）",
    picks: [[1, 1], [2, 1], [64, 1]],
  },
];

Deno.test("行ブロックの rows は (格納, m, n) の純関数で、掃引で採った表どおりに決まる", () => {
  for (const { storage, n, note, picks } of ROWS_TABLES) {
    for (const [m, rows] of picks) {
      const actual = linearGemvRowsForShape(storage, m, n);
      assertEquals(actual, rows, `${storage} n=${n} m=${m}（${note}）`);
      // 同じ入力を 2 度引いて同じ値（選択が状態を持たないこと = キーと WGSL の対応の前提）
      assertEquals(linearGemvRowsForShape(storage, m, n), actual, `${storage} n=${n} m=${m}: 再現`);
      // 既定変種は cols / unroll を M=1 変種から引き継ぎ、rows だけがこの純関数から来る
      assertEquals(
        defaultLinearGemvRowsVariant(storage, m, n),
        { ...defaultLinearGemvVariant(), rows },
        `${storage} n=${n} m=${m}: 既定変種`,
      );
    }
  }
});

Deno.test("並列度の目標を下げると rows が増える（他 device 用の静的ノブ）", () => {
  // 目標は `SessionOptions.linearGemvRowsThreadTarget` が Session 生成時に固定する限界値で、
  // 飽和点が小さい GPU では下げて `rows` を立てる（重みの読み直しを減らす）。i4・m=4・n=2048 は
  // スレッド数 = 2048 × ceil(4 / rows) なので、目標 16384 では rows=1 まで落ち、目標を半分に
  // するたびに 1 段ずつ止まる位置が上がる。
  assertEquals(linearGemvRowsForShape("i4", 4, 2048, 16384), 1, "既定の目標");
  assertEquals(linearGemvRowsForShape("i4", 4, 2048, 4096), 2, "目標 1/4");
  assertEquals(linearGemvRowsForShape("i4", 4, 2048, 2048), 4, "目標 1/8");
  // 目標を下げても天井（1 語あたり 256 要素 = i4 8 行）は超えない。
  assertEquals(linearGemvRowsForShape("i4", 64, 2048, 1), 8, "目標 1 でも i4 の天井 8");
  // 第 4 引数の省略 = 既定の目標（16384）を渡すのと同じ（既存の呼び出しは無変更で同じ結果）。
  for (const { storage, n, picks } of ROWS_TABLES) {
    for (const [m] of picks) {
      assertEquals(
        linearGemvRowsForShape(storage, m, n),
        linearGemvRowsForShape(storage, m, n, 16384),
        `${storage} n=${n} m=${m}: 省略 = 既定 16384`,
      );
      assertEquals(
        defaultLinearGemvRowsVariant(storage, m, n),
        defaultLinearGemvRowsVariant(storage, m, n, 16384),
        `${storage} n=${n} m=${m}: 変種も同じ`,
      );
    }
  }
  // 変種の側にも目標が通る（キーの `r<rows>` が動く = 生成物が別テキストになる）。
  assertEquals(defaultLinearGemvRowsVariant("i4", 4, 2048, 2048).rows, 4);
});

Deno.test("M=1 は n によらず rows=1（decode の生成物を行ブロック化しない）", () => {
  for (const n of [1, 256, 2048, 16384, 262144]) {
    assertEquals(linearGemvRowsForShape("i4", 1, n), 1, `i4 n=${n}`);
    assertEquals(linearGemvRowsForShape("i8", 1, n), 1, `i8 n=${n}`);
  }
});

Deno.test("門の外の (m, n) は黙って丸めず落ちる", () => {
  // m の域: 1..LINEAR_GEMV_MAX_ROWS。上限の外は既定の GEMM 骨格が受け持つ形なので、
  // ここが黙って上限へ丸めると「門が閉じた形が行ブロックで走る」沈黙誤値になる。
  assertThrows(() => linearGemvRowsForShape("i4", 0, 2048), CodegenError, "行数 0");
  assertThrows(
    () => linearGemvRowsForShape("i4", LINEAR_GEMV_MAX_ROWS + 1, 2048),
    CodegenError,
    `行数 ${LINEAR_GEMV_MAX_ROWS + 1}`,
  );
  assertThrows(() => linearGemvRowsForShape("i8", -1, 2048), CodegenError, "行数 -1");
  assertThrows(() => linearGemvRowsForShape("i4", 1.5, 2048), CodegenError, "行数 1.5");
  // n の域: 正整数。0 / 非整数は「スレッド数」の計算がそもそも意味を持たない。
  assertThrows(() => linearGemvRowsForShape("i4", 8, 0), CodegenError, "列数 0");
  assertThrows(() => linearGemvRowsForShape("i8", 8, -256), CodegenError, "列数 -256");
  assertThrows(() => linearGemvRowsForShape("i4", 8, 2048.5), CodegenError, "列数 2048.5");
  // 格納は圧縮 2 種だけ（実測した範囲に留める — ADR 0082 決定 4）
  assertThrows(() => linearGemvRowsForShape("f32", 8, 2048), CodegenError, "f32 格納");
});

Deno.test("キーに rows が載り、rows=1 の行ブロックは M=1 変種と別キーになる", () => {
  assertEquals(
    linearGemvRowsKey("i4", 32, defaultLinearGemvRowsVariant("i4", 32, 2048)),
    "linear_gemv:v1:f32:c32u4r4:wi4g32",
  );
  assertEquals(
    linearGemvRowsKey("i4", 64, defaultLinearGemvRowsVariant("i4", 32, 12288)),
    "linear_gemv:v1:f32:c32u4r8:wi4g64",
  );
  assertEquals(
    linearGemvRowsKey("i8", undefined, defaultLinearGemvRowsVariant("i8", 32, 262144)),
    "linear_gemv:v1:f32:c32u4r16:wi8",
  );
  // MUST: `r1` は M=1 変種のキーと別物（テキストも別 — y タイル化された行ブロック）。
  // 同じキーに割り当たると、M=1 の資産が M ≥ 2 の dispatch に配られて 1 行しか書かれない。
  assertNotEquals(
    linearGemvRowsKey("i4", 32, defaultLinearGemvRowsVariant("i4", 2, 64)),
    linearGemvKey("i4", 32),
  );
  assertEquals(
    linearGemvRowsKey("i4", 32, defaultLinearGemvRowsVariant("i4", 2, 64)),
    "linear_gemv:v1:f32:c32u4r1:wi4g32",
  );
  // rows も cols / unroll と同じく域を持つ（キーの生成が域外を通すと WGSL 側で初めて落ちる）
  assertThrows(
    () => linearGemvRowsKey("i4", 32, { cols: 32, unroll: 4, rows: 0 }),
    CodegenError,
    "rows は",
  );
  assertThrows(
    () => linearGemvRowsKey("i4", 32, { cols: 32, unroll: 4, rows: LINEAR_GEMV_MAX_ROWS + 1 }),
    CodegenError,
    "rows は",
  );
});

Deno.test("uniform の m は 1..上限（行ブロックの y タイルが受ける範囲）", () => {
  // 上限ちょうどは通る（dims.m は行ブロックの書き戻しガードが読む値そのもの）
  assertEquals(
    [...linearGemvParams("i4", LINEAR_GEMV_MAX_ROWS, 2048, 128, 32)],
    [LINEAR_GEMV_MAX_ROWS, 2048, 128, 0],
  );
  assertEquals([...linearGemvParams("i8", 2, 64, 64)], [2, 64, 64, 0]);
  // 域の外（m=0 / 上限 + 1）は落とす。通すと dispatch の y 枚数と dims.m が食い違い、
  // 書き戻しガードが黙って全行を捨てる / 範囲外の行を書く形になる。
  assertThrows(() => linearGemvParams("i4", 0, 2048, 128, 32), CodegenError, "m は 1..");
  assertThrows(
    () => linearGemvParams("i4", LINEAR_GEMV_MAX_ROWS + 1, 2048, 128, 32),
    CodegenError,
    "m は 1..",
  );
  assertThrows(
    () => linearGemvParams("i8", LINEAR_GEMV_MAX_ROWS + 1, 64, 64),
    CodegenError,
    "m は 1..",
  );
});

Deno.test("f16 GEMV は M=1 と 8 要素整列のみを受け、行ブロックの生成を拒否する", () => {
  assertEquals([...linearGemvParams("f16", 1, 36, 40)], [1, 36, 40, 0]);
  assertEquals([...linearGemvParams("f16", 1, 4, 0)], [1, 4, 0, 0]);
  assertThrows(() => linearGemvParams("f16", 2, 36, 40), CodegenError, "m=1 のみ");
  assertThrows(() => linearGemvParams("f16", 1, 36, 36), CodegenError, "k");
  assertThrows(() => linearGemvParams("f16", 1, 36, 40, 32), CodegenError);
  assertThrows(() => linearGemvRowsForShape("f16", 1, 36), CodegenError, "行ブロックは未対応");
  const variant = { cols: 32, unroll: 4, rows: 1 };
  assertThrows(
    () => linearGemvRowsKey("f16", undefined, variant),
    CodegenError,
    "行ブロックは未対応",
  );
  assertThrows(
    () => linearGemvRowsWgsl("f16", undefined, variant),
    CodegenError,
    "行ブロックは未対応",
  );
});

Deno.test("f32 GEMV は M=1 と 4 要素整列のみを受け、行ブロックの生成を拒否する", () => {
  assertEquals([...linearGemvParams("f32", 1, 36, 40)], [1, 36, 40, 0]);
  assertEquals([...linearGemvParams("f32", 1, 4, 0)], [1, 4, 0, 0]);
  assertThrows(() => linearGemvParams("f32", 2, 36, 40), CodegenError, "m=1 のみ");
  assertThrows(() => linearGemvParams("f32", 1, 36, 38), CodegenError, "k");
  assertThrows(() => linearGemvParams("f32", 1, 36, 40, 32), CodegenError);
  assertThrows(() => linearGemvRowsForShape("f32", 1, 36), CodegenError, "行ブロックは未対応");
  const variant = { cols: 32, unroll: 4, rows: 1 };
  assertThrows(
    () => linearGemvRowsKey("f32", undefined, variant),
    CodegenError,
    "行ブロックは未対応",
  );
  assertThrows(
    () => linearGemvRowsWgsl("f32", undefined, variant),
    CodegenError,
    "行ブロックは未対応",
  );
});

Deno.test("INT2 行ブロックは既定の全高さでシェーダーを10 KB未満に保つ", () => {
  // 初回解析費の回帰を検出する。従来の行別展開は r4 で76 KBだった（ADR 0082 追記8）。
  for (const rows of [1, 2, 4]) {
    const variant = { cols: 32, unroll: 4, rows };
    const source = linearGemvRowsWgsl("i2", undefined, variant);
    assertEquals(source.length < 10_000, true, `${rows} 行: ${source.length} bytes`);
    assertEquals(
      linearGemvRowsKey("i2", undefined, variant),
      `linear_gemv:v2:f32:c32u4r${rows}:wi2`,
    );
  }
});
