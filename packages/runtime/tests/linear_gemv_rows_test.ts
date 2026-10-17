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

import { test } from "@karume/test-support/test";
import { assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import { CodegenError } from "../src/codegen/errors.ts";
import {
  defaultLinearGemvRowsVariant,
  defaultLinearGemvVariant,
  LINEAR_GEMV_MAX_ROWS,
  linearGemvKey,
  linearGemvParallelKey,
  linearGemvParallelLanes,
  linearGemvParallelWgsl,
  linearGemvParams,
  linearGemvRowsForShape,
  linearGemvRowsKey,
  linearGemvRowsWgsl,
  linearGemvSubgroupKey,
  linearGemvSubgroupWgsl,
  linearGemvWgsl,
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
    picks: [
      [1, 1],
      [2, 1],
      [4, 1],
      [7, 1],
      [8, 1],
      [16, 1],
      [32, 1],
      [64, 1],
    ],
  },
  {
    storage: "i4",
    n: 2048,
    note: "中 n — M が増えて初めて目標に届き、rows が段階的に立つ",
    picks: [
      [1, 1],
      [2, 1],
      [4, 1],
      [7, 1],
      [8, 1],
      [16, 2],
      [32, 4],
      [64, 8],
    ],
  },
  {
    storage: "i4",
    n: 4096,
    note: "中 n（2048 の倍）— 同じ M で rows がちょうど 1 段深くなり、M=64 で i4 の天井 8 に当たる",
    picks: [
      [1, 1],
      [2, 1],
      [4, 1],
      [7, 2],
      [8, 2],
      [16, 4],
      [32, 8],
      [64, 8],
    ],
  },
  {
    storage: "i4",
    n: 12288,
    note: "大 n — M=16 で i4 の天井 8 行に当たり、M=32 / 64 でも 8 のまま伸びない",
    picks: [
      [1, 1],
      [2, 1],
      [4, 2],
      [7, 4],
      [8, 4],
      [16, 8],
      [32, 8],
      [64, 8],
    ],
  },
  {
    storage: "i8",
    n: 262144,
    note: "lm_head（i8）— どの M でも目標に届くので rows は M そのもの、天井 16 で頭打ち",
    picks: [
      [1, 1],
      [2, 2],
      [4, 4],
      [7, 4],
      [8, 8],
      [16, 16],
      [32, 16],
      [64, 16],
    ],
  },
  {
    storage: "i4",
    n: 1,
    note: "退化（列 1 本）— 目標に届きようがないので rows=1",
    picks: [
      [1, 1],
      [2, 1],
      [64, 1],
    ],
  },
  {
    storage: "i8",
    n: 1,
    note: "退化（列 1 本・i8）",
    picks: [
      [1, 1],
      [2, 1],
      [64, 1],
    ],
  },
];

test("行ブロックの rows は (格納, m, n) の純関数で、掃引で採った表どおりに決まる", () => {
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

test("並列度の目標を下げると rows が増える（他 device 用の静的ノブ）", () => {
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

test("M=1 は n によらず rows=1（decode の生成物を行ブロック化しない）", () => {
  for (const n of [1, 256, 2048, 16384, 262144]) {
    assertEquals(linearGemvRowsForShape("i4", 1, n), 1, `i4 n=${n}`);
    assertEquals(linearGemvRowsForShape("i8", 1, n), 1, `i8 n=${n}`);
  }
});

test("門の外の (m, n) は黙って丸めず落ちる", () => {
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

test("キーに rows が載り、rows=1 の行ブロックは M=1 変種と別キーになる", () => {
  assertEquals(
    linearGemvRowsKey("i4", 32, defaultLinearGemvRowsVariant("i4", 32, 2048)),
    "linear_gemv:v2:f32:c32u4r4:wi4g32",
  );
  assertEquals(
    linearGemvRowsKey("i4", 64, defaultLinearGemvRowsVariant("i4", 32, 12288)),
    "linear_gemv:v2:f32:c32u4r8:wi4g64",
  );
  assertEquals(
    linearGemvRowsKey("i8", undefined, defaultLinearGemvRowsVariant("i8", 32, 262144)),
    "linear_gemv:v2:f32:c32u4r16:wi8",
  );
  // MUST: `r1` は M=1 変種のキーと別物（テキストも別 — y タイル化された行ブロック）。
  // 同じキーに割り当たると、M=1 の資産が M ≥ 2 の dispatch に配られて 1 行しか書かれない。
  assertNotEquals(
    linearGemvRowsKey("i4", 32, defaultLinearGemvRowsVariant("i4", 2, 64)),
    linearGemvKey("i4", 32),
  );
  assertEquals(
    linearGemvRowsKey("i4", 32, defaultLinearGemvRowsVariant("i4", 2, 64)),
    "linear_gemv:v2:f32:c32u4r1:wi4g32",
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

test("uniform の m は 1..上限（行ブロックの y タイルが受ける範囲）", () => {
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

test("f16 GEMV は M=1 と 8 要素整列のみを受け、行ブロックの生成を拒否する", () => {
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

test("f32 GEMV は M=1 と 4 要素整列のみを受け、行ブロックの生成を拒否する", () => {
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

test("INT2 行ブロックは既定の全高さでシェーダーを10 KB未満に保つ", () => {
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

test("INT4・INT8 行ブロックは既定の全高さでシェーダーを12 KB未満に保つ", () => {
  // 従来の最大高さは約70 KB。行展開へ戻って初回コンパイルが増える回帰を検出する。
  for (const storage of ["i4", "i8"] as const) {
    for (const rows of storage === "i4" ? [1, 2, 4, 8] : [1, 2, 4, 8, 16]) {
      const variant = { cols: 32, unroll: 4, rows };
      const source = linearGemvRowsWgsl(storage, storage === "i4" ? 32 : undefined, variant);
      assertEquals(source.length < 12_000, true, `${storage} ${rows} 行: ${source.length} 文字`);
    }
  }
});

// The M=1 variant choice and the parallel-add (lanes) selection share the selection-function
// subject of the row-block gates above: pure functions of the shape, keys that tell every
// generated text apart, and fail-loudly rejection before codegen.

test("M=1 GEMVの形状選択", async (t) => {
  await t.step("実測した大語彙INT8をc16へ選び、別キーでWGSLを識別する", () => {
    const selected = defaultLinearGemvVariant({ storage: "i8", n: 262144, k: 1536 });
    assertEquals(selected, { cols: 16, unroll: 4 });
    assertEquals(linearGemvKey("i8", undefined, selected), "linear_gemv:v1:f32:c16u4:wi8");
    assertNotEquals(linearGemvKey("i8", undefined, selected), linearGemvKey("i8"));
    assertEquals(
      linearGemvWgsl("i8", undefined, selected),
      linearGemvWgsl("i8")
        .replace("32 列 / wg", "16 列 / wg")
        .replace("@workgroup_size(32)", "@workgroup_size(16)"),
    );
  });

  await t.step("格納型や行列形が異なる場合と形状未指定の生成は従来の変種を保つ", () => {
    assertEquals(defaultLinearGemvVariant(), { cols: 32, unroll: 4 });
    const storageTypes: readonly WeightStorage[] = ["i2", "i4", "f16", "f32"];
    for (const storage of storageTypes) {
      assertEquals(defaultLinearGemvVariant({ storage, n: 262144, k: 1536 }), {
        cols: 32,
        unroll: 4,
      });
    }
    for (const [n, k] of [
      [262140, 1536],
      [262148, 1536],
      [262144, 1520],
      [262144, 1552],
      [6144, 1536],
    ]) {
      assertEquals(defaultLinearGemvVariant({ storage: "i8", n, k }), { cols: 32, unroll: 4 });
    }
  });
});

test("GEMV並列加算の選択とコード生成", async (t) => {
  await t.step("decodeと少数行の検証で同じ幾何を選び、prefillと未計測形へ拡張しない", () => {
    for (const m of [1, 4, 8]) {
      assertEquals(linearGemvParallelLanes("i4", m, 256, 1536, 32), 32);
      assertEquals(linearGemvParallelLanes("i4", m, 6144, 1536, 512), 4);
      assertEquals(linearGemvParallelLanes("i4", m, 1536, 4096, 4096), 32);
      assertEquals(linearGemvParallelLanes("i2", m, 12288, 1536), 2);
      assertEquals(linearGemvParallelLanes("i8", m, 262144, 1536), 16);
    }
    for (const m of [0, 9, 64, 65]) {
      assertEquals(linearGemvParallelLanes("i4", m, 256, 1536, 32), undefined);
    }
    assertEquals(linearGemvParallelLanes("i2", 1, 262144, 1536), undefined);
    assertEquals(linearGemvParallelLanes("i4", 1, 256, 1536, 64), undefined);
    assertEquals(linearGemvParallelLanes("i4", 1, 260, 1536, 32), undefined);
    assertEquals(linearGemvParallelLanes("f16", 1, 256, 1536), undefined);
  });
  await t.step(
    "通常 E4B（i4 g32 + i8 lm_head）の形を E2B の同種行の lanes で選ぶ（未実測の写し）",
    () => {
      // 形は上流 config（hidden 2560・intermediate 10240・kv 2・層 42・PLE 256）から導いた実形。
      // QAT E4B は配布ミラーの記述子を根にする census テスト側が見る。
      const rows: readonly [number, number, 4 | 16 | 32][] = [
        [10752, 2560, 4],
        [2048, 2560, 4],
        [512, 2560, 32],
        [256, 2560, 32],
        [2560, 2048, 32],
        [10240, 2560, 4],
        [2560, 10240, 32],
        [2560, 256, 4],
        [4096, 2560, 4],
        [1024, 2560, 32],
        [2560, 4096, 32],
      ];
      for (const m of [1, 4, 8]) {
        for (const [n, k, lanes] of rows) {
          assertEquals(linearGemvParallelLanes("i4", m, n, k, 32), lanes, `i4 ${n}×${k} M${m}`);
        }
        assertEquals(linearGemvParallelLanes("i8", m, 262144, 2560), 16);
      }
      assertEquals(linearGemvParallelLanes("i4", 9, 10240, 2560, 32), undefined);
      // i2 の lm_head（QAT E4B）は E2B にも行が無いので足していない。
      assertEquals(linearGemvParallelLanes("i2", 1, 262144, 2560), undefined);
    },
  );
  await t.step("キーが加算形・格納・group・laneを区別する", () => {
    const keys = new Map<string, string>();
    for (const storage of ["i2", "i4", "i8"] as const) {
      for (const group of storage === "i4" ? [32, 512, 2048, 4096] : [undefined]) {
        for (const lanes of [2, 4, 8, 16, 32] as const) {
          const key = linearGemvParallelKey(storage, group, lanes);
          const shader = linearGemvParallelWgsl(storage, group, lanes);
          assertNotEquals(key, linearGemvKey(storage, group));
          assertEquals(keys.has(key), false);
          keys.set(key, shader);
          assertEquals(linearGemvParallelWgsl(storage, group, lanes), shader);
          const subgroupKey = linearGemvSubgroupKey(storage, group, lanes);
          const subgroupShader = linearGemvSubgroupWgsl(storage, group, lanes);
          assertNotEquals(subgroupKey, key);
          assertEquals(keys.has(subgroupKey), false);
          keys.set(subgroupKey, subgroupShader);
          assertEquals(linearGemvSubgroupWgsl(storage, group, lanes), subgroupShader);
        }
      }
    }
    assertEquals(keys.size, 60);
  });
  await t.step("未対応格納と不正なgroupをコード生成前に拒否する", () => {
    assertThrows(() => linearGemvParallelWgsl("f16", undefined, 4));
    assertThrows(() => linearGemvParallelWgsl("f32", undefined, 4));
    assertThrows(() => linearGemvParallelWgsl("i4", 16, 4));
    assertThrows(() => linearGemvParallelWgsl("i8", 32, 4));
    assertThrows(() => linearGemvSubgroupWgsl("f16", undefined, 4));
    assertThrows(() => linearGemvSubgroupWgsl("f32", undefined, 4));
    assertThrows(() => linearGemvSubgroupWgsl("i4", 16, 4));
    assertThrows(() => linearGemvSubgroupWgsl("i8", 32, 4));
  });
});

test("subgroup32変種の積和はparallelと同じ綴り", async (t) => {
  // ADR 0101 決定 3（積和は parallel と同一）を綴りで縛る。parallel 族だけ fma 化して subgroup が
  // `acc + x * d` のまま残ると、Metal では縮約の入れ方が式形で変わり u32 一致が崩れる（ADR 0105 追記 4）。
  // GPU 実走の u32 一致門（helpers/gemv-subgroup-check.ts）は Deno で SKIP になるので、この門が常設側。
  const macLines = (wgsl: string): string[] =>
    wgsl
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.startsWith("acc = "));
  await t.step("全60形でfma行の列が一致し、積和にacc + x * dが残らない", () => {
    let compared = 0;
    for (const storage of ["i2", "i4", "i8"] as const) {
      for (const group of storage === "i4" ? [32, 512, 2048, 4096] : [undefined]) {
        for (const lanes of [2, 4, 8, 16, 32] as const) {
          const parallel = macLines(linearGemvParallelWgsl(storage, group, lanes));
          const subgroup = macLines(linearGemvSubgroupWgsl(storage, group, lanes));
          const fma = (lines: string[]): string[] =>
            lines.filter((l) => l.startsWith("acc = fma("));
          assertEquals(fma(subgroup), fma(parallel), `${storage} g${group} l${lanes}`);
          assertEquals(fma(subgroup).length > 0, true);
          // subgroup の値交換（`if (lane < width) { acc = acc + other; }`）は if の行なので、
          // `acc = ` で始まる行に `acc + x * d` の綴りが 1 本も無いことが積和の門になる。
          assertEquals(
            subgroup.filter((l) => l.startsWith("acc = acc +")),
            [],
            `${storage} g${group} l${lanes}: 積和に acc + x * d が残っている`,
          );
          compared++;
        }
      }
    }
    assertEquals(compared, 30);
  });
});
