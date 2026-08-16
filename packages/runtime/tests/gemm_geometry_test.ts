// skinny-M 向けタイル幾何の選択（src/kernels/gemm-geometry.ts の `gemmGeometryForRows`）と、
// それが matmul / bmm / linear のキー・生成物へどう流れるか（GPU 不要）。
//
// 検証の眼目は 3 点:
//
// 1. **バケット境界**が表のとおりで、M ≥ 128 は既定幾何のまま（Anima / SBV2 の既存キーと
//    既存性能を動かさない境界）。
// 2. **キーと生成物が同じ幾何**から出る。片方だけ行数を渡し忘れると、キャッシュに載った別幾何の
//    WGSL が executor の dispatch 数と噛み合わず、出力タイルが例外なしに欠ける。
// 3. **幾何を変えても数値契約は動かない**（K タイル 16・外側 t 昇順・内側 kk 昇順・積和の字面）。
//    ここは字面の固定で、値のビット一致そのものは tests/gpu_gemm_skinny_test.ts が実 GPU で見る。

import { assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import { CodegenError } from "../src/codegen/errors.ts";
import {
  assertGemmGeometry,
  defaultGemmGeometry,
  GEMM_TILE_K,
  type GemmGeometry,
  gemmGeometryForRows,
  gemmQuadsPerThread,
  gemmTileM,
  gemmTileN,
} from "../src/kernels/gemm-geometry.ts";
import { bmmKey, bmmWgsl } from "../src/kernels/bmm.ts";
import { linearKey, linearWgsl } from "../src/kernels/linear.ts";
import { matmulKey, matmulWgsl } from "../src/kernels/matmul.ts";

/** 表の代表点（バケットごとに「境界の内側」と「境界を 1 越えた側」を持つ）。 */
const BUCKET_ROWS = [1, 4, 16, 17, 64, 65, 128, 4096] as const;

const geometryOf = (rows: number): GemmGeometry => gemmGeometryForRows(rows);

Deno.test("gemmGeometryForRows のバケット境界は 64 / 512 で、M >= 513 は既定幾何へ落ちる", () => {
  // 小 M バケット（2026-08-11 掃引: M=1/4/64 とも M16N16 が対既定 ×3.0〜3.2 の最良。
  // M=64 は行タイル 4 枚 = 重みを 4 回読み直してもなお workgroup 数の勝ち）
  for (const rows of [0, 1, 2, 4, 16, 17, 32, 63, 64]) {
    assertEquals(geometryOf(rows), { regM: 1, regN: 4, wgX: 4, wgY: 16 }, `M=${rows}`);
  }
  // 中 M バケット（掃引 M=318 ×1.67 / M=512 ×1.28。波①で Anima/SBV2 の E2E A/B とセット採用）
  for (const rows of [65, 127, 128, 129, 318, 512]) {
    assertEquals(geometryOf(rows), { regM: 4, regN: 4, wgX: 8, wgY: 16 }, `M=${rows}`);
  }
  // MUST: ここから上は既定のまま（掃引の実測点の外 — DiT の 2026-08-10 実測選定を動かさない）
  for (const rows of [513, 1024, 4096, 65536]) {
    assertEquals(geometryOf(rows), defaultGemmGeometry(), `M=${rows}`);
  }
});

Deno.test("バケットの幾何は全て充填の整除条件を満たす（穴の空くタイル形を表に置けない）", () => {
  for (const rows of BUCKET_ROWS) {
    const geometry = geometryOf(rows);
    // 落ちる組み合わせは「充填が届かない語が 0 のまま内積へ入る」= 例外の出ない誤値
    assertGemmGeometry(geometry, `M=${rows}`);
    // 共有メモリは WebGPU 既定上限 16,384 B（sa = tileM·16 f32 + sb = 16·(tileN/4) vec4）
    const shared = gemmTileM(geometry) * GEMM_TILE_K * 4 +
      GEMM_TILE_K * (gemmTileN(geometry) / 4) * 16;
    assertEquals(shared <= 16384, true, `M=${rows}: 共有 ${shared} B`);
    // 小 M バケットは行タイル 4 枚（= 重みの読み直し 4 回）以内。掃引の結論は「読み直しを
    // 増やしてでも workgroup を増やす方が勝つ」だが、際限なく増えると転送律速へ戻る
    if (rows <= 64) {
      assertEquals(
        Math.ceil(Math.max(rows, 1) / gemmTileM(geometry)) <= 4,
        true,
        `M=${rows}: 行タイルが 5 枚以上になる`,
      );
    }
  }
});

Deno.test("gemmGeometryForRows は非負整数以外を fail loudly にする", () => {
  for (const rows of [-1, 1.5, Number.NaN, Number.POSITIVE_INFINITY]) {
    assertThrows(() => gemmGeometryForRows(rows), CodegenError, "行数 M は u32 の非負整数");
  }
});

Deno.test("行数はキーに載り、バケットが違えば別パイプラインになる", () => {
  // 幾何判別子 `r{regM}x{regN}w{wgX}` はタイル辺と併せて幾何を一意に決める（wgY は tileM/regM）
  assertEquals(linearKey("f32", true, "f32", 4), "linear:v2:f32:reg16x16r1x4w4v4");
  assertEquals(linearKey("f32", true, "f32", 64), "linear:v2:f32:reg16x16r1x4w4v4");
  assertEquals(linearKey("f32", true, "f32", 318), "linear:v2:f32:reg64x32r4x4w8v4");
  assertEquals(matmulKey(false, 4), "matmul:v2:f32:reg16x16r1x4w4");
  assertEquals(bmmKey(true, 17), "bmm:v2:f32:reg16x16r1x4w4v4");
  // 3 バケットのキーは互いに衝突しない
  const keys = [4, 318, 4096].map((rows) => linearKey("f32", true, "f32", rows));
  assertEquals(new Set(keys).size, keys.length, keys.join(", "));
});

Deno.test("行数を渡さない呼び出しと M >= 513 は既存のキー・生成物をバイト単位で保つ", () => {
  // 融合 attention とスナップショットが通る経路（行数を持たない）
  assertEquals(linearKey("f32", true), "linear:v2:f32:reg128x128r8x8w16v4");
  assertEquals(matmulKey(true), "matmul:v2:f32:reg128x128r8x8w16v4");
  assertEquals(bmmKey(false), "bmm:v2:f32:reg128x128r8x8w16");
  for (const rows of [513, 1024, 4096]) {
    assertEquals(linearKey("i8", false, "f32", rows), linearKey("i8", false));
    assertEquals(linearWgsl("f16", true, "f16", rows), linearWgsl("f16", true, "f16"));
    assertEquals(matmulWgsl(true, rows), matmulWgsl(true));
    assertEquals(bmmWgsl(false, rows), bmmWgsl(false));
  }
});

Deno.test("生成物の workgroup 形・共有タイル・出力タイル原点はキーの幾何と一致する", () => {
  for (const rows of BUCKET_ROWS) {
    const geometry = geometryOf(rows);
    const tileM = gemmTileM(geometry);
    const tileN = gemmTileN(geometry);
    const quads = gemmQuadsPerThread(geometry);
    const variants: readonly (readonly [string, string])[] = [
      [`linear M=${rows}`, linearWgsl("f32", true, "f32", rows)],
      [`matmul M=${rows}`, matmulWgsl(true, rows)],
      [`bmm M=${rows}`, bmmWgsl(true, rows)],
    ];
    for (const [where, wgsl] of variants) {
      assertEquals(
        wgsl.includes(`@compute @workgroup_size(${geometry.wgX}, ${geometry.wgY})`),
        true,
        `${where}: workgroup 形`,
      );
      assertEquals(
        wgsl.includes(`var<workgroup> sa: array<f32, ${tileM * GEMM_TILE_K}>;`),
        true,
        `${where}: 共有 A タイル`,
      );
      assertEquals(
        wgsl.includes(`var<workgroup> sb: array<vec4<f32>, ${GEMM_TILE_K * (tileN / 4)}>;`),
        true,
        `${where}: 共有 B タイル`,
      );
      assertEquals(
        wgsl.includes(`let orow0 = wid.y * ${tileM}u + lid.y * ${geometry.regM}u;`),
        true,
        `${where}: 出力タイルの行原点`,
      );
      // acc は regM × (regN/4) 本ちょうど（1 本多い / 少ないは静的展開の取り違え）
      assertEquals(
        wgsl.includes(`var acc${geometry.regM - 1}_${quads - 1} = vec4<f32>(`),
        true,
        `${where}: acc の右下`,
      );
      assertEquals(wgsl.includes(`var acc${geometry.regM}_0 =`), false, `${where}: acc の行超過`);
      assertEquals(wgsl.includes(`var acc0_${quads} =`), false, `${where}: acc の列超過`);
    }
  }
});

Deno.test("幾何を変えても K 縮約の刻みと積和の字面は動かない（ADR 0022 の数値契約）", () => {
  for (const rows of BUCKET_ROWS) {
    const wgsl = linearWgsl("f32", true, "f32", rows);
    const where = `linear M=${rows}`;
    // 外側 t 昇順 / 内側 kk 昇順 / K タイル 16 — 1 出力要素あたりの加算順序そのもの
    assertEquals(wgsl.includes(`let tiles = (dims.k + 15u) / ${GEMM_TILE_K}u;`), true, where);
    assertEquals(wgsl.includes("for (var t = 0u; t < tiles; t = t + 1u) {"), true, where);
    assertEquals(
      wgsl.includes(`for (var kk = 0u; kk < ${GEMM_TILE_K}u; kk = kk + 1u) {`),
      true,
      where,
    );
    // MUST: 積和は `acc = acc + a * b` の字面のまま（fma を明示しない）
    assertEquals(wgsl.includes("acc0_0 = acc0_0 + "), true, where);
    assertEquals(wgsl.includes("fma("), false, `${where}: fma の明示`);
  }
});

Deno.test("同じ行数からは常にバイト単位で同じ WGSL が出る（決定性）", () => {
  for (const rows of BUCKET_ROWS) {
    assertEquals(linearWgsl("f32", true, "f32", rows), linearWgsl("f32", true, "f32", rows));
    assertEquals(matmulWgsl(false, rows), matmulWgsl(false, rows));
  }
  // バケットが違えば生成物も違い（キーだけ分かれて中身が同じ = 幾何が流れていない）、
  // 同じバケットなら行数が違っても生成物は同じ（キーの基数が M の値ぶん爆発しない）
  assertNotEquals(matmulWgsl(true, 64), matmulWgsl(true, 65));
  assertNotEquals(matmulWgsl(true, 512), matmulWgsl(true, 513));
  assertEquals(matmulWgsl(true, 4), matmulWgsl(true, 32));
  assertEquals(matmulWgsl(true, 65), matmulWgsl(true, 512));
});
