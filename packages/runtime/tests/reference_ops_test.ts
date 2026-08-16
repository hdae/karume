import { assertAlmostEquals, assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import {
  allclose,
  AllcloseError,
  compareTensors,
  DEFAULT_TOLERANCE,
  EXACT_TOLERANCE,
  formatAllclose,
} from "../src/reference/allclose.ts";
import {
  applyReferenceOp,
  referenceAttention,
  referenceBinary,
  referenceBmm,
  referenceCast,
  referenceCat,
  referenceConv1d,
  referenceConv2d,
  referenceConvTranspose1d,
  referenceCumsum,
  referenceDeformConv2d,
  referenceEmbedding,
  referenceExpand,
  referenceFlip,
  referenceGather,
  referenceGruScan,
  referenceLayerNorm,
  referenceLinear,
  referenceMaskedFill,
  referenceMatmul,
  ReferenceOpError,
  referencePad,
  referencePermute,
  referenceReshape,
  referenceRmsNorm,
  referenceRowReduce,
  referenceSafeSoftmax,
  referenceSlice,
  referenceSoftmax,
  referenceSymPrefixSlice,
  referenceUnary,
  referenceUpsampleBilinear2d,
  referenceWhere,
  type RefTensor,
  refTensor,
} from "../src/reference/ops.ts";
import { OpContractError, type UnaryOpName } from "../src/ops.ts";

const t = (shape: readonly number[], values: readonly number[]): RefTensor =>
  refTensor(shape, Float32Array.from(values));

const i32 = (shape: readonly number[], values: readonly number[]): RefTensor =>
  refTensor(shape, Int32Array.from(values));

const bools = (shape: readonly number[], values: readonly number[]): RefTensor =>
  refTensor(shape, Uint32Array.from(values));

Deno.test("unary の既知値（torch 既定の定義に一致する）", () => {
  const at = (op: UnaryOpName, x: number): number => referenceUnary(op, t([1], [x])).data[0];
  assertEquals(at("neg", -2), 2);
  assertEquals(at("abs", -2.5), 2.5);
  assertAlmostEquals(at("exp", 1), Math.E, 1e-6);
  assertAlmostEquals(at("log", Math.E), 1, 1e-6);
  assertEquals(at("sqrt", 9), 3);
  // sin は奇関数で π の整数倍が零点（DACVAE の Snake 活性が踏む定義域 — ADR 0043 第 1 層）。
  assertEquals(at("sin", 0), 0);
  assertAlmostEquals(at("sin", 1), 0.8414709848078965, 1e-6);
  assertAlmostEquals(at("sin", -1), -0.8414709848078965, 1e-6);
  assertAlmostEquals(at("sin", Math.PI / 2), 1, 1e-6);
  assertAlmostEquals(at("sin", Math.PI), 0, 1e-6);
  assertAlmostEquals(at("tanh", 1), 0.7615941559557649, 1e-6);
  assertEquals(at("sigmoid", 0), 0.5);
  assertAlmostEquals(at("sigmoid", 2), 0.8807970779778823, 1e-6);
  assertEquals(at("relu", -1), 0);
  assertEquals(at("relu", 3), 3);
  // gelu(x) = x·Φ(x)（approximate="none"）— Φ(1)=0.8413447460685429
  assertEquals(at("gelu", 0), 0);
  assertAlmostEquals(at("gelu", 1), 0.8413447460685429, 1e-6);
  assertAlmostEquals(at("gelu", -1), -0.15865525393145707, 1e-6);
  assertAlmostEquals(at("gelu", 3), 2.9959503059051097, 1e-6);
  // |x| > 4 で erf を ±1 に丸めても f32 では区別できない
  assertAlmostEquals(at("gelu", 8), 8, 1e-6);
  // gelu_tanh(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715x³)))（approximate="tanh"）。
  // erf 形とは x = ±1 で 3e-4 級ずれる — 同じ op に畳めないことがこの差で見える。
  assertEquals(at("gelu_tanh", 0), 0);
  assertAlmostEquals(at("gelu_tanh", 1), 0.8411919906082768, 1e-6);
  assertAlmostEquals(at("gelu_tanh", -1), -0.15880800939172324, 1e-6);
  assertAlmostEquals(at("gelu_tanh", 3), 2.996362607918227, 1e-6);
  assertAlmostEquals(at("gelu_tanh", 8), 8, 1e-6);
});

Deno.test("binary は右詰め broadcast で評価する", () => {
  const sum = referenceBinary("add", t([2, 3], [1, 2, 3, 4, 5, 6]), t([3], [10, 20, 30]));
  assertEquals(sum.shape, [2, 3]);
  assertEquals([...sum.data], [11, 22, 33, 14, 25, 36]);

  const product = referenceBinary("mul", t([2, 1], [1, 2]), t([1, 3], [10, 20, 30]));
  assertEquals(product.shape, [2, 3]);
  assertEquals([...product.data], [10, 20, 30, 20, 40, 60]);

  const diff = referenceBinary("sub", t([2], [5, 7]), t([2], [1, 2]));
  assertEquals([...diff.data], [4, 5]);
  const quotient = referenceBinary("div", t([2], [6, 9]), t([2], [2, 3]));
  assertEquals([...quotient.data], [3, 3]);
});

// 波3 の数理 op。既知値は torch の定義（clamp / leaky_relu / log1p / 比較）そのもの。
Deno.test("attrs 付き unary は torch と同じ定義で、スカラは f32 に丸めて使う", () => {
  const clamp = referenceUnary("clamp", t([5], [-3, -1, 0, 1, 3]), { min: -1, max: 1 });
  assertEquals([...clamp.data], [-1, -1, 0, 1, 1]);
  // 境界は両端とも「そのまま」（< / > の向きを取り違えると等値点で値が変わる）
  assertEquals([...referenceUnary("clamp", t([2], [-1, 1]), { min: -1, max: 1 }).data], [-1, 1]);

  // clamp_min は下限だけ（上限は素通し）。境界はそのまま返す。
  const clampMin = referenceUnary("clamp_min", t([5], [-3, -1, 0, 1, 3]), { min: -1 });
  assertEquals([...clampMin.data], [-1, -1, 0, 1, 3]);
  // MUST: NaN は伝播する（torch の clamp(NaN, min=m) = NaN）。`x >= min ? x : min` の
  // 向きで書くと NaN が min に化けるので、この 1 本が向きの検出器になる。
  assertEquals(
    Number.isNaN(referenceUnary("clamp_min", t([1], [NaN]), { min: 0 }).data[0]),
    true,
  );
  // 実測のチャネル L2 正規化（clamp(min=eps)）— 下限が極小でも 0 が持ち上がる
  assertEquals(
    [...referenceUnary("clamp_min", t([2], [0, 5]), { min: 1e-12 }).data],
    [Math.fround(1e-12), 5],
  );
  assertThrows(() => referenceUnary("clamp_min", t([1], [0]), {}), OpContractError);

  // slope 2 種（ADR 0015 — 実測は 0.1 と 0.01 が混在する）
  const steep = referenceUnary("leaky_relu", t([4], [-2, -1, 0, 3]), { negative_slope: 0.1 });
  assertEquals([...steep.data], [-0.2, -0.1, 0, 3].map((v) => Math.fround(v)));
  const shallow = referenceUnary("leaky_relu", t([2], [-2, 2]), { negative_slope: 0.01 });
  assertEquals([...shallow.data], [Math.fround(-0.02), 2]);
  // MUST: NaN は伝播する（torch の leaky_relu(NaN) = NaN）
  assertEquals(
    Number.isNaN(referenceUnary("leaky_relu", t([1], [NaN]), { negative_slope: 0.1 }).data[0]),
    true,
  );

  // log1p は Math.log1p 準拠 — 素朴な log(1+x) が 0 に潰れる領域で値を持つ
  assertEquals(referenceUnary("log1p", t([1], [0]), {}).data[0], 0);
  assertAlmostEquals(referenceUnary("log1p", t([1], [Math.E - 1]), {}).data[0], 1, 1e-6);
  assertEquals(referenceUnary("log1p", t([1], [1e-8]), {}).data[0], Math.fround(1e-8));

  // 比較は bool（u32 の 0/1）を返す。等値点で ge と gt が割れる。
  const ge = referenceUnary("ge_scalar", t([3], [-1, 0, 1]), { value: 0 });
  assertEquals(ge.dtype, "bool");
  assertEquals([...ge.data], [0, 1, 1]);
  assertEquals([...referenceUnary("gt_scalar", t([3], [-1, 0, 1]), { value: 0 }).data], [0, 0, 1]);
  assertEquals([...referenceUnary("le_scalar", t([3], [-1, 0, 1]), { value: 0 }).data], [1, 1, 0]);
  // 必須 attr の欠落・値域外は契約が拒否する
  assertThrows(() => referenceUnary("clamp", t([1], [0]), { min: 0 }), OpContractError);
  assertThrows(() => referenceUnary("clamp", t([1], [0]), { min: 1, max: 0 }), OpContractError);
  assertThrows(() => referenceUnary("leaky_relu", t([1], [0]), {}), OpContractError);
});

Deno.test("ge / bitwise_and は broadcast したうえで bool を返す", () => {
  // searchsorted の形（inputs[…,None] >= bl）
  const ge = referenceBinary("ge", t([3, 1], [-1, 0.5, 2]), t([3], [0, 1, 2]));
  assertEquals(ge.dtype, "bool");
  assertEquals(ge.shape, [3, 3]);
  assertEquals([...ge.data], [0, 0, 0, 1, 0, 0, 1, 1, 1]);

  const and = referenceBinary(
    "bitwise_and",
    bools([2, 2], [1, 1, 0, 0]),
    bools([2], [1, 0]),
  );
  assertEquals(and.dtype, "bool");
  assertEquals([...and.data], [1, 0, 0, 0]);
  // dtype の混在と契約外 dtype は拒否
  assertThrows(() => referenceBinary("bitwise_and", t([1], [1]), t([1], [1])), OpContractError);
  assertThrows(() => referenceBinary("ge", bools([1], [1]), bools([1], [1])), OpContractError);
});

// MUST: 分岐の向き（真なら第 2 引数）は値でしか検出できない — a / b を別の値域で埋める。
Deno.test("where は条件の真で a、偽で b を取り、三者を右詰め broadcast する", () => {
  const out = referenceWhere(
    bools([2, 3], [1, 0, 1, 0, 1, 0]),
    t([2, 3], [1, 2, 3, 4, 5, 6]),
    t([3], [-10, -20, -30]),
  );
  assertEquals(out.dtype, "f32");
  assertEquals(out.shape, [2, 3]);
  assertEquals([...out.data], [1, -20, 3, -10, 5, -30]);
  // 条件が値より低い rank（spline の inside 判定の形）
  const wide = referenceWhere(bools([2], [1, 0]), t([1], [7]), t([2, 2], [1, 2, 3, 4]));
  assertEquals(wide.shape, [2, 2]);
  assertEquals([...wide.data], [7, 2, 7, 4]);
  assertThrows(
    () => referenceWhere(bools([1], [1]), bools([1], [1]), t([1], [0])),
    OpContractError,
  );
});

// MUST: 累積方向は非対称な列でしか検出できない（[1,1,1] は逆向きでも同じ）。
Deno.test("cumsum は最終次元の前縁和で、行ごとに独立している", () => {
  const out = referenceCumsum(t([2, 4], [1, 2, 3, 4, 10, 20, 30, 40]), { dim: 1 });
  assertEquals(out.shape, [2, 4]);
  assertEquals([...out.data], [1, 3, 6, 10, 10, 30, 60, 100]);
  // 逆向きなら [10,9,7,4] になる形（前縁と後縁が区別できる列）
  assertEquals([...referenceCumsum(t([1, 4], [1, 2, 3, 4]), { dim: 1 }).data], [1, 3, 6, 10]);
  // rank1（行 1 本）と最終次元以外の拒否
  assertEquals([...referenceCumsum(t([3], [2, -1, 5]), { dim: 0 }).data], [2, 1, 6]);
  assertThrows(() => referenceCumsum(t([2, 2], [1, 2, 3, 4]), { dim: 0 }), OpContractError);
});

Deno.test("matmul は row-major の既知値を返す", () => {
  const out = referenceMatmul(t([2, 2], [1, 2, 3, 4]), t([2, 2], [5, 6, 7, 8]));
  assertEquals(out.shape, [2, 2]);
  assertEquals([...out.data], [19, 22, 43, 50]);

  const rect = referenceMatmul(t([2, 3], [1, 2, 3, 4, 5, 6]), t([3, 1], [1, 1, 1]));
  assertEquals(rect.shape, [2, 1]);
  assertEquals([...rect.data], [6, 15]);
});

// MUST: B / M / K / N を全て違う長さで確かめる（ACTIVE_DESIGN の Pitfalls — 軸の取り違えは
// 正方形や対称な形では数値に出ない）。ここは手計算の既知値で固定する。
Deno.test("bmm はバッチごとに独立した rank-3 の行列積を返す", () => {
  // B=2, M=3, K=2, N=1（全て違う長さ）。バッチ 1 は全要素を 10 倍にしてある。
  const a = t([2, 3, 2], [1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 60]);
  const b = t([2, 2, 1], [1, 10, 100, 1000]);
  const out = referenceBmm(a, b);
  assertEquals(out.shape, [2, 3, 1]);
  // バッチ 0: [1·1+2·10, 3·1+4·10, 5·1+6·10] / バッチ 1: [10·100+20·1000, ...]
  assertEquals([...out.data], [21, 43, 65, 21000, 43000, 65000]);

  // バッチが 1 枚でも rank-3（matmul へは縮退しない）
  const single = referenceBmm(t([1, 2, 2], [1, 2, 3, 4]), t([1, 2, 2], [5, 6, 7, 8]));
  assertEquals(single.shape, [1, 2, 2]);
  assertEquals([...single.data], [19, 22, 43, 50]);

  // rank-2 / バッチ不一致 / 縮約不一致は契約が落とす
  assertThrows(
    () => referenceBmm(t([2, 2], [1, 2, 3, 4]), t([2, 2], [1, 2, 3, 4])),
    OpContractError,
  );
  assertThrows(
    () => referenceBmm(t([2, 1, 2], [1, 2, 3, 4]), t([1, 2, 1], [1, 2])),
    OpContractError,
  );
  assertThrows(() => referenceBmm(i32([1, 1, 1], [1]), t([1, 1, 1], [1])), OpContractError);
});

// 契約: out[..., j] = src[..., index[..., j]]（最終次元固定）。範囲外添字は**必ず throw**
// （GPU 側は NaN 汚染 — src/kernels/gather.ts の裁定。オラクル側は緩めない）。
Deno.test("gather は行ごとに最終次元を引き直し、範囲外添字を拒否する", () => {
  const src = t([2, 4], [1, 2, 3, 4, 10, 20, 30, 40]);
  const index = i32([2, 3], [3, 0, 2, 1, 1, 0]);
  const out = referenceGather(src, index);
  assertEquals(out.dtype, "f32");
  assertEquals(out.shape, [2, 3]);
  // 行 0 は自分の行からだけ引く（行を跨いだら 10/20 が混ざる）
  assertEquals([...out.data], [4, 1, 3, 20, 20, 10]);

  // 実測形（src f32[16,T,512] / index i32[16,T,T]）と同型の rank-3
  const cube = referenceGather(
    t([2, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    i32([2, 2, 2], [2, 0, 1, 1, 0, 2, 2, 2]),
  );
  assertEquals(cube.shape, [2, 2, 2]);
  assertEquals([...cube.data], [3, 1, 5, 5, 7, 9, 12, 12]);

  // 範囲外（上側 / 負）は fail loudly
  assertThrows(() => referenceGather(src, i32([2, 3], [4, 0, 0, 0, 0, 0])), ReferenceOpError);
  assertThrows(() => referenceGather(src, i32([2, 3], [0, 0, 0, 0, -1, 0])), ReferenceOpError);
  // スロットの取り違え（src と index を逆に渡す）は契約が落とす
  assertThrows(() => referenceGather(index, src as never), OpContractError);
  assertThrows(() => referenceGather(src, t([2, 3], [0, 1, 2, 0, 1, 2])), OpContractError);
  // 先行次元の不一致
  assertThrows(() => referenceGather(src, i32([3, 3], new Array(9).fill(0))), OpContractError);
});

Deno.test("行 reduce は最終次元を keepdim 無しで畳む", () => {
  const x = t([2, 3], [1, 2, 3, 4, 5, 6]);
  assertEquals([...referenceRowReduce("sum", x, 1).data], [6, 15]);
  assertEquals([...referenceRowReduce("amax", x, 1).data], [3, 6]);
  assertEquals([...referenceRowReduce("amin", x, 1).data], [1, 4]);
  assertEquals(referenceRowReduce("sum", x, 1).shape, [2]);

  // rank 1 → rank 0（スカラ）
  const scalar = referenceRowReduce("sum", t([3], [1, 2, 3]), 0);
  assertEquals(scalar.shape, []);
  assertEquals([...scalar.data], [6]);

  const cube = referenceRowReduce("amax", t([2, 2, 2], [1, 8, 3, 2, 9, 4, 5, 6]), 2);
  assertEquals(cube.shape, [2, 2]);
  assertEquals([...cube.data], [8, 3, 9, 6]);
});

// 軸 reduce（最終次元以外）。**恒真化しない形**を選ぶ: 全ての軸で値が違い、軸を取り違えたら
// 必ず別の数になる非対称なデータを使う（rank-3 の 3 軸それぞれで答えが割れる）。
Deno.test("reduce は attrs.dim の軸だけを畳む（軸ごとに答えが割れる）", () => {
  // x[i,j,k] = 100i + 10j + k（どの軸を畳んでも別の値になる）
  const shape = [2, 3, 4];
  const data = new Array<number>(24);
  for (let i = 0; i < 2; i += 1) {
    for (let j = 0; j < 3; j += 1) {
      for (let k = 0; k < 4; k += 1) data[(i * 3 + j) * 4 + k] = 100 * i + 10 * j + k;
    }
  }
  const x = t(shape, data);

  const axis0 = referenceRowReduce("sum", x, 0);
  assertEquals(axis0.shape, [3, 4]);
  // Σ_i (100i + 10j + k) = 100 + 2·(10j + k)
  assertEquals([...axis0.data], [100, 102, 104, 106, 120, 122, 124, 126, 140, 142, 144, 146]);

  const axis1 = referenceRowReduce("sum", x, 1);
  assertEquals(axis1.shape, [2, 4]);
  // Σ_j (100i + 10j + k) = 300i + 30 + 3k
  assertEquals([...axis1.data], [30, 33, 36, 39, 330, 333, 336, 339]);

  const axis2 = referenceRowReduce("sum", x, 2);
  assertEquals(axis2.shape, [2, 3]);
  // Σ_k (100i + 10j + k) = 400i + 40j + 6
  assertEquals([...axis2.data], [6, 46, 86, 406, 446, 486]);

  // amax / amin も軸で割れる（sum の添字式だけを直しても通らない形）
  assertEquals([...referenceRowReduce("amax", x, 1).data], [20, 21, 22, 23, 120, 121, 122, 123]);
  assertEquals([...referenceRowReduce("amin", x, 0).data], [
    0,
    1,
    2,
    3,
    10,
    11,
    12,
    13,
    20,
    21,
    22,
    23,
  ]);

  // 軸は attrs 経由でも同じ（applyReferenceOp の結線）
  assertEquals([...applyReferenceOp("sum", [x], { dim: 1 }).data], [...axis1.data]);
  // 宣言必須（既定値補完をしない）
  assertThrows(() => applyReferenceOp("sum", [x]), OpContractError);
  assertThrows(() => applyReferenceOp("sum", [x], { dim: 3 }), OpContractError);
});

// 契約（src/ops.ts）: f32 → i32 は torch 準拠の truncate（0 方向切り捨て）。
// MUST: round / floor と区別できる値を使う — -2.7 は trunc -2 / floor -3 / round -3。
Deno.test("cast は f32 → i32 を 0 方向へ切り捨てる（round でも floor でもない）", () => {
  const out = referenceCast(t([6], [2.7, -2.7, 0.5, -0.5, 1.5, -1.5]), "i32");
  assertEquals(out.dtype, "i32");
  assertEquals([...out.data], [2, -2, 0, 0, 1, -1]);
});

Deno.test("cast は x != 0 で真偽化し、bool は u32 の 0/1 として読む", () => {
  const fromFloat = referenceCast(t([4], [0, -0.25, 3, -7]), "bool");
  assertEquals(fromFloat.dtype, "bool");
  assertEquals([...fromFloat.data], [0, 1, 1, 1]);
  assertEquals([...referenceCast(i32([3], [0, -1, 5]), "bool").data], [0, 1, 1]);
  // bool → 数値は 0/1 がそのまま出る
  assertEquals([...referenceCast(bools([2], [0, 1]), "f32").data], [0, 1]);
  assertEquals([...referenceCast(bools([2], [0, 1]), "i32").data], [0, 1]);
  // 同型 cast は恒等コピー
  assertEquals([...referenceCast(i32([2], [7, -7]), "i32").data], [7, -7]);
});

Deno.test("bitwise_not は bool の否定で、bool 以外は契約が拒否する", () => {
  const out = referenceUnary("bitwise_not", bools([4], [0, 1, 1, 0]));
  assertEquals(out.dtype, "bool");
  assertEquals([...out.data], [1, 0, 0, 1]);
  assertThrows(() => referenceUnary("bitwise_not", t([2], [0, 1])), OpContractError);
  assertThrows(() => referenceUnary("relu", bools([2], [0, 1])), OpContractError);
});

Deno.test("i32 の binary は解禁した op だけを 32bit 演算で評価する", () => {
  // mask 外積（実測グラフの形）: [3,1] × [1,3] の broadcast
  const outer = referenceBinary("mul", i32([3, 1], [1, 0, 1]), i32([1, 3], [1, 1, 0]));
  assertEquals(outer.dtype, "i32");
  assertEquals([...outer.data], [1, 1, 0, 0, 0, 0, 1, 1, 0]);
  assertEquals([...referenceBinary("sub", i32([2], [1, 1]), i32([2], [0, 1])).data], [1, 0]);
  // i32 は 2 の補数で折り返す（GPU の i32 演算と同じ）
  assertEquals([...referenceBinary("mul", i32([1], [65536]), i32([1], [65536])).data], [0]);
  // 契約表が解禁していない組み合わせ
  assertThrows(() => referenceBinary("add", i32([1], [1]), i32([1], [1])), OpContractError);
  assertThrows(() => referenceBinary("div", i32([1], [4]), i32([1], [2])), OpContractError);
  // dtype 混在（両方とも mul の契約 dtype ではあるが、混合型の elementwise は語彙に無い）
  assertThrows(() => referenceBinary("mul", i32([1], [1]), t([1], [1])), ReferenceOpError);
});

Deno.test("reshape は要素順を変えずに形だけ付け替える", () => {
  const out = referenceReshape(t([2, 3], [1, 2, 3, 4, 5, 6]), [3, 2]);
  assertEquals(out.shape, [3, 2]);
  assertEquals([...out.data], [1, 2, 3, 4, 5, 6]);
  // dtype も素通し（要素に触れない op）
  assertEquals(referenceReshape(bools([4], [0, 1, 1, 0]), [2, 2]).dtype, "bool");
  assertThrows(() => referenceReshape(t([2, 3], [1, 2, 3, 4, 5, 6]), [4, 2]), ReferenceOpError);
});

// 期待値は手計算（codegen の stride 表と独立 — 参照実装は scatter 形で導く）。
Deno.test("permute は dims の並べ替えで要素を配り直す", () => {
  // [2,3] の転置: row-major [1,2,3 / 4,5,6] → [1,4 / 2,5 / 3,6]
  const flat = referencePermute(t([2, 3], [1, 2, 3, 4, 5, 6]), [1, 0]);
  assertEquals(flat.shape, [3, 2]);
  assertEquals([...flat.data], [1, 4, 2, 5, 3, 6]);
  // 実測形 [0,2,1,3]（attention の head 整形）を rank4 で踏む
  const src = Array.from({ length: 24 }, (_, i) => i);
  const heads = referencePermute(t([1, 2, 3, 4], src), [0, 2, 1, 3]);
  assertEquals(heads.shape, [1, 3, 2, 4]);
  assertEquals(
    [...heads.data].join(","),
    "0,1,2,3,12,13,14,15,4,5,6,7,16,17,18,19,8,9,10,11,20,21,22,23",
  );
  // MUST: 巡回長 3 の並べ替えを 1 本持つ（実測の形は全て対合で、逆置換との取り違えを
  // 検出できない）。[2,3] の [1,0] とは違い [1,2,0] の逆置換は [2,0,1] で結果が変わる。
  const cycled = referencePermute(t([1, 2, 3], [1, 2, 3, 4, 5, 6]), [1, 2, 0]);
  assertEquals(cycled.shape, [2, 3, 1]);
  assertEquals([...cycled.data], [1, 2, 3, 4, 5, 6]);
  const rotated = referencePermute(t([2, 1, 3], [1, 2, 3, 4, 5, 6]), [1, 2, 0]);
  assertEquals(rotated.shape, [1, 3, 2]);
  assertEquals([...rotated.data], [1, 4, 2, 5, 3, 6]);
  // 恒等な並べ替えは要素順を変えない
  assertEquals([...referencePermute(t([2, 2], [1, 2, 3, 4]), [0, 1]).data], [1, 2, 3, 4]);
  assertThrows(() => referencePermute(t([2, 3], [1, 2, 3, 4, 5, 6]), [0, 0]), ReferenceOpError);
  assertThrows(() => referencePermute(t([2, 3], [1, 2, 3, 4, 5, 6]), [0]), ReferenceOpError);
});

Deno.test("expand は長さ 1 の次元だけを複製する", () => {
  // gather 添字の実測形 [1,T,T] → [16,T,T] を縮めた [1,2,2] → [3,2,2]
  const spread = referenceExpand(i32([1, 2, 2], [1, 2, 3, 4]), [3, 2, 2]);
  assertEquals(spread.shape, [3, 2, 2]);
  assertEquals([...spread.data], [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]);
  // bool マスクの実測形 [1,T,1] → [1,T,C]
  const mask = referenceExpand(bools([1, 2, 1], [1, 0]), [1, 2, 3]);
  assertEquals([...mask.data], [1, 1, 1, 0, 0, 0]);
  assertEquals(mask.dtype, "bool");
  // rank が増える形（右詰め）
  assertEquals([...referenceExpand(i32([2], [7, 8]), [2, 2]).data], [7, 8, 7, 8]);
  assertThrows(() => referenceExpand(i32([2, 2], [1, 2, 3, 4]), [2, 4]), ReferenceOpError);
  assertThrows(() => referenceExpand(i32([2, 2], [1, 2, 3, 4]), [4]), ReferenceOpError);
});

// ---- レイアウト第 2 群（ADR 0014）------------------------------------------

Deno.test("slice は静的軸の [start, end) だけを取り出す", () => {
  // enc_p の stats[:, :c] / stats[:, c:] を縮めた形（チャネル軸 4 → 前半 2 / 後半 2）
  const stats = t([1, 4, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
  const head = referenceSlice(stats, { dim: 1, start: 0, end: 2 });
  assertEquals(head.shape, [1, 2, 2]);
  assertEquals([...head.data], [1, 2, 3, 4]);
  const tail = referenceSlice(stats, { dim: 1, start: 2, end: 4 });
  assertEquals([...tail.data], [5, 6, 7, 8]);
  // 最終次元の切り出し（spline の bin_locations[..., :-1] と同型）— 行の送り幅は入力側のまま
  const bins = t([2, 3], [1, 2, 3, 4, 5, 6]);
  assertEquals([...referenceSlice(bins, { dim: 1, start: 0, end: 2 }).data], [1, 2, 4, 5]);
  assertEquals([...referenceSlice(bins, { dim: 1, start: 1, end: 3 }).data], [2, 3, 5, 6]);
  // 先頭軸の切り出しと、全長（恒等コピー）
  assertEquals([...referenceSlice(bins, { dim: 0, start: 1, end: 2 }).data], [4, 5, 6]);
  assertEquals([...referenceSlice(bins, { dim: 0, start: 0, end: 2 }).data], [...bins.data]);
  // 契約違反（範囲外・逆転・軸の外）は shape 層が落とす
  assertThrows(() => referenceSlice(bins, { dim: 1, start: 1, end: 4 }), OpContractError);
  assertThrows(() => referenceSlice(bins, { dim: 1, start: 2, end: 1 }), OpContractError);
  assertThrows(() => referenceSlice(bins, { dim: 2, start: 0, end: 1 }), OpContractError);
});

Deno.test("cat は連結軸に入力を順に並べ、出力を全て埋める", () => {
  // coupling reverse の cat([x0,x1], 1) を縮めた形（軸 1 で 2 + 1）
  const a = t([2, 2], [1, 2, 3, 4]);
  const b = t([2, 1], [5, 6]);
  const joined = referenceCat([a, b], { dim: 1 });
  assertEquals(joined.shape, [2, 3]);
  // MUST: 行ごとに交互に並ぶ（offset を「入力の要素数ぶん」で組む誤りだと [1,2,3,4,5,6] になる）
  assertEquals([...joined.data], [1, 2, 5, 3, 4, 6]);
  // 先頭軸の連結は素直な連結（行の並びがそのまま）
  assertEquals([...referenceCat([a, t([1, 2], [9, 8])], { dim: 0 }).data], [1, 2, 3, 4, 9, 8]);
  // 3 入力（可変アリティ）— 順序が値に出る形
  const three = referenceCat([t([1, 1], [1]), t([1, 2], [2, 3]), t([1, 1], [4])], { dim: 1 });
  assertEquals([...three.data], [1, 2, 3, 4]);
  // 契約違反は shape 層が落とす（連結軸以外の不一致・入力 1 本）
  assertThrows(() => referenceCat([a, t([3, 1], [1, 2, 3])], { dim: 1 }), OpContractError);
  assertThrows(() => referenceCat([a], { dim: 1 }), OpContractError);
});

Deno.test("pad は最終次元を 0 で埋め、範囲外が厳密に 0 になる", () => {
  // 相対位置 value 側の F.pad(p_attn, [w, w]) を縮めた形（w = 1）
  const attn = t([2, 2], [1, 2, 3, 4]);
  const padded = referencePad(attn, { left: 1, right: 1 });
  assertEquals(padded.shape, [2, 4]);
  assertEquals([...padded.data], [0, 1, 2, 0, 0, 3, 4, 0]);
  // 非対称・片側 0（境界の off-by-one が値に出る形）
  assertEquals([...referencePad(attn, { left: 2, right: 0 }).data], [0, 0, 1, 2, 0, 0, 3, 4]);
  assertEquals([...referencePad(attn, { left: 0, right: 1 }).data], [1, 2, 0, 3, 4, 0]);
  // 幅 0 は恒等コピー
  assertEquals([...referencePad(attn, { left: 0, right: 0 }).data], [...attn.data]);
  assertThrows(() => referencePad(attn, { left: -1, right: 0 }), OpContractError);
});

Deno.test("flip は静的軸の添字を反転する", () => {
  // flow の Flip を縮めた形（軸 1 の長さ 3 — 長さ 2 だと off-by-one が対称で消える）
  const x = t([2, 3, 2], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  const flipped = referenceFlip(x, { dim: 1 });
  assertEquals(flipped.shape, [2, 3, 2]);
  assertEquals([...flipped.data], [5, 6, 3, 4, 1, 2, 11, 12, 9, 10, 7, 8]);
  // 軸を取り違えると別の値になる（先頭軸・最終軸それぞれ）
  assertEquals([...referenceFlip(x, { dim: 0 }).data], [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]);
  assertEquals([...referenceFlip(x, { dim: 2 }).data], [2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]);
  // 反転は対合（2 回で元に戻る）
  assertEquals([...referenceFlip(referenceFlip(x, { dim: 1 }), { dim: 1 }).data], [...x.data]);
  assertThrows(() => referenceFlip(x, { dim: 3 }), OpContractError);
});

Deno.test("sym_prefix_slice は Tmax 定数の先頭だけを切り出す", () => {
  // 相対位置バケット表の実測形 — Tmax 4×4 の定数から T=2 の先頭 2×2 を取り出す
  const table = i32([4, 4], [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
  ]);
  const sliced = referenceSymPrefixSlice(table, [2, 2]);
  assertEquals(sliced.shape, [2, 2]);
  // MUST: 行の送り幅は **Tmax 形（4）** — 出力 shape（2）から組むと [0,1,2,3] になる
  assertEquals([...sliced.data], [0, 1, 4, 5]);
  assertEquals(sliced.dtype, "i32");
  // 1 軸だけ縮める形（位置テーブル）と、縮めない形（恒等）
  assertEquals([...referenceSymPrefixSlice(table, [2, 4]).data], [0, 1, 2, 3, 4, 5, 6, 7]);
  assertEquals([...referenceSymPrefixSlice(table, [4, 4]).data], [...table.data]);
  // f32 も同じ経路（位置スケール表）
  const scale = t([3, 2], [0, 1, 2, 3, 4, 5]);
  assertEquals([...referenceSymPrefixSlice(scale, [2, 1]).data], [0, 2]);
  // 定数に収まらない prefix と rank 違いは fail loudly
  assertThrows(() => referenceSymPrefixSlice(table, [5, 4]), ReferenceOpError);
  assertThrows(() => referenceSymPrefixSlice(table, [4]), ReferenceOpError);
});

// ---- 融合 op（ADR 0012）の既知値 ------------------------------------------

Deno.test("linear は転置レイアウトの重みで既知値を返す", () => {
  // x[2,3] × W[2,3]（W は [out,in]）+ b[2]
  const out = referenceLinear(
    t([2, 3], [1, 2, 3, 4, 5, 6]),
    t([2, 3], [1, 0, -1, 2, 2, 2]),
    t([2], [10, -1]),
  );
  assertEquals(out.shape, [2, 2]);
  // 行 0: [1·1+2·0+3·(−1)+10, 1·2+2·2+3·2+(−1)] = [8, 11]
  // 行 1: [4·1+5·0+6·(−1)+10, 4·2+5·2+6·2+(−1)] = [8, 29]
  assertEquals([...out.data], [8, 11, 8, 29]);
  // 先行次元は平坦化される（rank-3 でも重みは同じ）
  assertEquals(
    referenceLinear(
      t([2, 1, 3], [1, 2, 3, 4, 5, 6]),
      t([2, 3], [1, 0, -1, 2, 2, 2]),
      t([2], [
        10,
        -1,
      ]),
    ).shape,
    [2, 1, 2],
  );
  // 特徴数・bias 長の不一致は契約が落とす
  assertThrows(
    () => referenceLinear(t([2, 3], [1, 2, 3, 4, 5, 6]), t([2, 2], [1, 0, 1, 0]), t([2], [0, 0])),
    OpContractError,
  );
});

// MUST: 分散は母分散（N 割り）。torch.var の既定（N−1）で組むと dim が小さいほど乖離する。
Deno.test("layer_norm は母分散（correction=0）で正規化する", () => {
  const attrs = { normalized_shape: [4], eps: 1e-12 };
  const out = referenceLayerNorm(
    t([1, 4], [1, 2, 3, 4]),
    t([4], [1, 1, 1, 1]),
    t([4], [0, 0, 0, 0]),
    attrs,
  );
  // mean = 2.5 / 母分散 = 1.25（N−1 なら 1.6667 で、下の期待値と 15% 違う）
  const inv = 1 / Math.sqrt(1.25);
  assertAlmostEquals(out.data[0], -1.5 * inv, 1e-5);
  assertAlmostEquals(out.data[3], 1.5 * inv, 1e-5);
  // affine が効く（weight で伸ばし bias で平行移動）
  const affine = referenceLayerNorm(
    t([1, 4], [1, 2, 3, 4]),
    t([4], [2, 2, 2, 2]),
    t([4], [5, 5, 5, 5]),
    attrs,
  );
  assertAlmostEquals(affine.data[0], -1.5 * inv * 2 + 5, 1e-5);
  // 分散 0 の行は eps だけが残る（出力は bias そのもの）
  const constant = referenceLayerNorm(
    t([1, 4], [7, 7, 7, 7]),
    t([4], [3, 3, 3, 3]),
    t([4], [1, 2, 3, 4]),
    { normalized_shape: [4], eps: 1e-7 },
  );
  assertEquals([...constant.data], [1, 2, 3, 4]);
});

// MUST: rms_norm は**平均を引かない**（layer_norm の写し間違いを手計算で押さえる）。
Deno.test("rms_norm は二乗平均で正規化し、平均を引かない", () => {
  const attrs = { eps: 1e-12 };
  // mean(x²) = (1+4+9+16)/4 = 7.5 → rms = √7.5。layer_norm なら mean 2.5 を引くので
  // 先頭が負になる（ここは全て正のまま = 平均を引いていない証拠）。
  const out = referenceRmsNorm(t([1, 4], [1, 2, 3, 4]), t([4], [1, 1, 1, 1]), attrs);
  const inv = 1 / Math.sqrt(7.5);
  assertAlmostEquals(out.data[0], 1 * inv, 1e-6);
  assertAlmostEquals(out.data[3], 4 * inv, 1e-6);
  // weight は要素ごとに掛かる（bias は無い — アリティ 2）
  const scaled = referenceRmsNorm(t([1, 4], [1, 2, 3, 4]), t([4], [2, 0, -1, 0.5]), attrs);
  assertAlmostEquals(scaled.data[0], 1 * inv * 2, 1e-6);
  assertEquals(scaled.data[1], 0);
  assertAlmostEquals(scaled.data[2], 3 * inv * -1, 1e-6);
  // 行ごとに独立（先行次元の平坦化が正しいこと）
  const rows = referenceRmsNorm(t([2, 2], [3, 4, 6, 8]), t([2], [1, 1]), attrs);
  assertAlmostEquals(rows.data[0], 3 / Math.sqrt(12.5), 1e-6);
  assertAlmostEquals(rows.data[2], 6 / Math.sqrt(50), 1e-6);
  // MUST: eps は**平方根の中**（外に足すと全要素 0 の行で 0 ではなく eps 倍の値が出る）。
  // 全要素 0 の行は eps が無ければ 0/0 = NaN。ここが 0 でなければ eps の位置が誤っている。
  const zeros = referenceRmsNorm(t([1, 3], [0, 0, 0]), t([3], [5, 5, 5]), { eps: 1e-6 });
  assertEquals([...zeros.data], [0, 0, 0]);
  // eps の効き方を値で押さえる（x=1 の 1 要素行: 1/√(1+eps)）
  const epsy = referenceRmsNorm(t([1, 1], [1]), t([1], [1]), { eps: 0.5 });
  assertAlmostEquals(epsy.data[0], 1 / Math.sqrt(1.5), 1e-6);
  // weight 長・rank の不一致は契約が拒否する
  assertThrows(
    () => referenceRmsNorm(t([1, 4], [1, 2, 3, 4]), t([3], [1, 1, 1]), attrs),
    OpContractError,
  );
  assertThrows(
    () => referenceRmsNorm(t([1, 4], [1, 2, 3, 4]), t([1, 4], [1, 1, 1, 1]), attrs),
    OpContractError,
  );
});

// MUST: safe-softmax（amax 減算）。素朴形は台帳の反例集どおり大入力で NaN になる。
Deno.test("softmax は amax を引いて大入力でも壊れない", () => {
  const attrs = { dim: 1 };
  const out = referenceSoftmax(t([1, 3], [1, 2, 3]), attrs);
  const total = Math.exp(-2) + Math.exp(-1) + 1;
  assertAlmostEquals(out.data[0], Math.exp(-2) / total, 1e-6);
  assertAlmostEquals(out.data[0] + out.data[1] + out.data[2], 1, 1e-6);
  // 素朴形なら exp(-200) が f32 で 0 に潰れて 0/0 = NaN になる領域
  const huge = referenceSoftmax(t([1, 2], [-200, -199]), attrs);
  assertEquals(huge.data.every((v) => Number.isFinite(v)), true);
  assertAlmostEquals(huge.data[0] + huge.data[1], 1, 1e-6);
  assertAlmostEquals(huge.data[1] / huge.data[0], Math.E, 1e-4);
  // 全要素が同じ行（masked_fill で全マスクされた行）は一様分布
  const masked = referenceSoftmax(t([1, 4], Array(4).fill(-3.4028234663852886e+38)), attrs);
  assertEquals([...masked.data], [0.25, 0.25, 0.25, 0.25]);
  // 最終次元以外は語彙に無い
  assertThrows(() => referenceSoftmax(t([2, 3], [1, 2, 3, 4, 5, 6]), { dim: 0 }), OpContractError);
});

// safe_softmax（ADR 0044）— softmax の全機能 + 「行 max が −inf の行は全 0」。torch の
// SDPA ガード（`where(¬any(¬eq(src,−inf)), 0, softmax(src))`）と同じ値になることを固定する。
Deno.test("safe_softmax は全 −inf の行に 0 を書き、それ以外は softmax と同値", () => {
  const attrs = { dim: 1 };
  const neg = Number.NEGATIVE_INFINITY;
  // 行 0 = 通常 / 行 1 = 全 −inf（torch のガードが発火する行）/ 行 2 = 一部 −inf
  const x = t([3, 3], [1, 2, 3, neg, neg, neg, neg, 0, 1]);
  const out = referenceSafeSoftmax(x, attrs);
  assertEquals([...out.data.slice(3, 6)], [0, 0, 0]);
  // 素の softmax は同じ行で 0/0 = NaN（この 1 行が「別 op である理由」そのもの）
  assertEquals(referenceSoftmax(x, attrs).data.slice(3, 6).every(Number.isNaN), true);
  // −inf を含まない行 / 一部だけ −inf の行は素の softmax と**ビット単位で**一致する
  const plain = referenceSoftmax(x, attrs);
  assertEquals([...out.data.slice(0, 3)], [...plain.data.slice(0, 3)]);
  assertEquals([...out.data.slice(6, 9)], [...plain.data.slice(6, 9)]);
  assertAlmostEquals(out.data[6], 0, 1e-12);
  assertAlmostEquals(out.data[7] + out.data[8], 1, 1e-6);
  // masked_fill の埋め値（−F32_MAX）は **有限**なので空行ではない（一様分布のまま）
  const filled = referenceSafeSoftmax(t([1, 4], Array(4).fill(-3.4028234663852886e+38)), attrs);
  assertEquals([...filled.data], [0.25, 0.25, 0.25, 0.25]);
  // 契約は softmax と同一（最終次元のみ）
  assertThrows(
    () => referenceSafeSoftmax(t([2, 3], [1, 2, 3, 4, 5, 6]), { dim: 0 }),
    OpContractError,
  );
});

// 融合 attention（ADR 0023）。オラクルの規律は「素直な 3 段」で、GPU の融合形（P 非実体化）を
// 写さないこと。**半スケール**（scale が q と k の両方に掛かる）と safe 化をここで固定する。
Deno.test("attention は scale を q と k の両方へ掛け、safe-softmax で大入力を壊さない", () => {
  // B=1 / H=1 / M=1 / N=2 / D=2 の手計算ケース。scale=2 なら内積に 4 が掛かる。
  const q = t([1, 1, 1, 2], [1, 0]);
  const key = t([1, 1, 2, 2], [1, 0, 0, 1]);
  const value = t([1, 1, 2, 2], [10, 20, 30, 40]);
  const out = referenceAttention(q, key, value, { scale: 2 });
  assertEquals(out.shape, [1, 1, 1, 2]);
  // S = [(1·2)·(1·2), (1·2)·(0·2)] = [4, 0] — **片側だけに掛けると [2, 0]** になる
  const weight = 1 / (1 + Math.exp(-4));
  assertAlmostEquals(out.data[0], weight * 10 + (1 - weight) * 30, 1e-5);
  assertAlmostEquals(out.data[1], weight * 20 + (1 - weight) * 40, 1e-5);
  // 片側スケール（誤実装）の値と一致しないことを明示的に固定する
  const halfWeight = 1 / (1 + Math.exp(-2));
  assertEquals(Math.abs(out.data[0] - (halfWeight * 10 + (1 - halfWeight) * 30)) > 1e-3, true);

  // 全 logit が −200 級の行（素朴 softmax なら exp が f32 で 0 に潰れて 0/0 = NaN）
  const huge = referenceAttention(
    t([1, 1, 1, 2], [-10, -10]),
    t([1, 1, 2, 2], [10, 10, 10, 11]),
    value,
    { scale: 1 },
  );
  assertEquals([...huge.data].every((v) => Number.isFinite(v)), true);

  // B / H を別々の軸として扱う（積が同じ形で取り違えが起きない）
  const batched = referenceAttention(
    t([2, 1, 1, 1], [1, 2]),
    t([2, 1, 1, 1], [1, 1]),
    t([2, 1, 1, 1], [5, 7]),
    { scale: 1 },
  );
  // N=1 なので softmax は恒等（確率 1）— 出力は v そのもの
  assertEquals([...batched.data], [5, 7]);

  // 契約違反（rank-3 / D 不一致 / scale 欠落）はオラクル側でも落ちる
  assertThrows(() => referenceAttention(t([1, 1, 1], [1]), key, value, { scale: 1 }), Error);
  assertThrows(
    () => referenceAttention(q, key, t([1, 1, 2, 3], [1, 2, 3, 4, 5, 6]), { scale: 1 }),
    OpContractError,
  );
  assertThrows(() => referenceAttention(q, key, value, {}), OpContractError);
  // 統一入口からも同じ結果が出る（kind 分岐の結線）
  assertEquals(
    [...applyReferenceOp("attention", [q, key, value], { scale: 2 }).data],
    [...out.data],
  );

  // 加算 mask（省略可能な第 4 入力 — ADR 0023 改訂）。S を丸めた**後**に足す。
  // 上と同じ手計算ケースで列 1 を落とすと、出力は v の 0 行目そのものになる。
  const masked = referenceAttention(q, key, value, { scale: 2 }, t([1, 1, 1, 2], [0, -1e30]));
  assertAlmostEquals(masked.data[0], 10, 1e-5);
  assertAlmostEquals(masked.data[1], 20, 1e-5);
  // MUST: mask は B·H へ broadcast する（[1,1,M,N] の 1 枚を全 head が読む）。
  // 2 head の入力に列 0 落としの mask を掛けると、両 head とも v の 1 行目になる。
  const twoHeads = referenceAttention(
    t([1, 2, 1, 2], [1, 0, 1, 0]),
    t([1, 2, 2, 2], [1, 0, 0, 1, 1, 0, 0, 1]),
    t([1, 2, 2, 2], [10, 20, 30, 40, 50, 60, 70, 80]),
    { scale: 2 },
    t([1, 1, 1, 2], [-1e30, 0]),
  );
  assertEquals([...twoHeads.data].map((v) => Math.round(v)), [30, 40, 70, 80]);
  // 統一入口も 4 本目を落とさず渡す（渡し忘れると mask 無しの値が黙って返る）
  assertEquals(
    [
      ...applyReferenceOp("attention", [q, key, value, t([1, 1, 1, 2], [0, -1e30])], { scale: 2 })
        .data,
    ],
    [...masked.data],
  );
  // 契約違反の mask（[1,1,M,N] から外れる形）はオラクル側でも落ちる
  assertThrows(
    () => referenceAttention(q, key, value, { scale: 2 }, t([1, 1, 2, 2], [0, 0, 0, 0])),
    OpContractError,
  );
});

Deno.test("embedding は行を引き直し、範囲外添字を拒否する", () => {
  const weight = t([3, 2], [1, 2, 3, 4, 5, 6]);
  const out = referenceEmbedding(weight, i32([2, 2], [2, 0, 1, 2]));
  assertEquals(out.shape, [2, 2, 2]);
  assertEquals([...out.data], [5, 6, 1, 2, 3, 4, 5, 6]);
  // padding_idx は forward に効かない（受け取らないので影響のしようが無い）
  assertEquals(
    [...applyReferenceOp("embedding", [weight, i32([1], [0])], { padding_idx: 0 }).data],
    [1, 2],
  );
  // MUST: 範囲外は throw（GPU 側の NaN 汚染に合わせて緩めない）
  assertThrows(() => referenceEmbedding(weight, i32([1], [3])), ReferenceOpError);
  assertThrows(() => referenceEmbedding(weight, i32([1], [-1])), ReferenceOpError);
});

Deno.test("masked_fill は mask の真の位置だけを埋め、mask を右詰め broadcast する", () => {
  const out = referenceMaskedFill(
    t([2, 3], [1, 2, 3, 4, 5, 6]),
    bools([3], [1, 0, 1]),
    { value: -7.5 },
  );
  assertEquals(out.shape, [2, 3]);
  assertEquals([...out.data], [-7.5, 2, -7.5, -7.5, 5, -7.5]);
  // 実測の埋め値（f32 の最小有限値）が丸めで動かない
  const extreme = referenceMaskedFill(t([2], [1, 2]), bools([2], [1, 0]), {
    value: -3.4028234663852886e+38,
  });
  assertEquals([...extreme.data], [-3.4028234663852886e+38, 2]);
  // mask が x を広げる形は契約が落とす（埋め値が本来無い要素へ漏れない）
  assertThrows(
    () => referenceMaskedFill(t([1], [1]), bools([3], [1, 0, 1]), { value: 0 }),
    OpContractError,
  );
});

const CONV = { stride: 1, padding: 0, dilation: 1, groups: 1 } as const;

Deno.test("conv1d は padding を 0 詰めで扱い、stride で出力長が決まる", () => {
  // x[1,1,4] = [1,2,3,4] / W[1,1,3] = [1,1,1] / bias = 0、stride 1 / padding 1
  const same = referenceConv1d(
    t([1, 1, 4], [1, 2, 3, 4]),
    t([1, 1, 3], [1, 1, 1]),
    t([1], [0]),
    { ...CONV, padding: 1 },
  );
  assertEquals(same.shape, [1, 1, 4]);
  // 両端は padding 域を読み飛ばす（[0+1+2, 1+2+3, 2+3+4, 3+4+0]）
  assertEquals([...same.data], [3, 6, 9, 7]);
  // padding を 0 扱いにすると出力長が 2 に縮む（別の形）
  const valid = referenceConv1d(
    t([1, 1, 4], [1, 2, 3, 4]),
    t([1, 1, 3], [1, 1, 1]),
    t([1], [0]),
    CONV,
  );
  assertEquals(valid.shape, [1, 1, 2]);
  assertEquals([...valid.data], [6, 9]);
  // stride 2（出力長 floor((4+0−3)/2)+1 = 1）と bias
  const strided = referenceConv1d(
    t([1, 1, 4], [1, 2, 3, 4]),
    t([1, 1, 3], [1, 1, 1]),
    t([1], [100]),
    { ...CONV, stride: 2 },
  );
  assertEquals(strided.shape, [1, 1, 1]);
  assertEquals([...strided.data], [106]);
  // 入力チャネルごとに縮約する（Cin=2 / Cout=2）
  const multi = referenceConv1d(
    t([1, 2, 3], [1, 2, 3, 10, 20, 30]),
    t([2, 2, 1], [1, 0, 0, 1]),
    t([2], [0, 0]),
    CONV,
  );
  assertEquals(multi.shape, [1, 2, 3]);
  assertEquals([...multi.data], [1, 2, 3, 10, 20, 30]);
});

Deno.test("conv1d の dilation はカーネル位置を飛ばし、groups は入力チャネル帯を絞る", () => {
  // dilation 2 / k 3 は [x0, x2, x4] を見る（張りは 5 — 出力長 L − 4）
  const dilated = referenceConv1d(
    t([1, 1, 7], [1, 2, 3, 4, 5, 6, 7]),
    t([1, 1, 3], [1, 1, 1]),
    t([1], [0]),
    { ...CONV, dilation: 2 },
  );
  assertEquals(dilated.shape, [1, 1, 3]);
  assertEquals([...dilated.data], [1 + 3 + 5, 2 + 4 + 6, 3 + 5 + 7]);
  // MUST: dilation は k に掛かる（ox に掛ける誤りは stride と同じ形になり、この
  // 「入力長は据え置きで出力長だけ縮む」ケースでしか区別できない）
  const strided = referenceConv1d(
    t([1, 1, 7], [1, 2, 3, 4, 5, 6, 7]),
    t([1, 1, 3], [1, 1, 1]),
    t([1], [0]),
    { ...CONV, stride: 2 },
  );
  assertEquals([...strided.data], [1 + 2 + 3, 3 + 4 + 5, 5 + 6 + 7]);

  // depthwise（groups = Cin = Cout = 2）— 各出力チャネルは同番のチャネルだけを見る。
  // グループ跨ぎで読む誤りは「別チャネルの値が混ざる」形で必ず値に出る。
  const depthwise = referenceConv1d(
    t([1, 2, 3], [1, 2, 3, 10, 20, 30]),
    t([2, 1, 1], [1, 100]),
    t([2], [0, 0]),
    { ...CONV, groups: 2 },
  );
  assertEquals(depthwise.shape, [1, 2, 3]);
  assertEquals([...depthwise.data], [1, 2, 3, 1000, 2000, 3000]);

  // 中間の groups（Cin 4 / Cout 2 / g 2）— 重みの第 2 軸は Cin/groups = 2
  const grouped = referenceConv1d(
    t([1, 4, 2], [1, 2, 3, 4, 10, 20, 30, 40]),
    t([2, 2, 1], [1, 1, 1, 1]),
    t([2], [0, 0]),
    { ...CONV, groups: 2 },
  );
  assertEquals(grouped.shape, [1, 2, 2]);
  assertEquals([...grouped.data], [1 + 3, 2 + 4, 10 + 30, 20 + 40]);
});

/** conv2d の既定 attrs（各ケースは必要な軸だけ差し替える）。 */
const CONV2D = {
  stride: [1, 1],
  padding: [0, 0],
  dilation: [1, 1],
  groups: 1,
} as const;

// MUST: 重みは [Cout, Cin/groups, Kh, Kw]。Kh と Kw を入れ替えて読む実装は**正方カーネルでは
// 値まで一致する**ので、手計算ケースは必ず Kh ≠ Kw かつ重みが非対称なものにする。
Deno.test("conv2d は [Cout,Cin/g,Kh,Kw] の重みで畳み込む（手計算・Kh≠Kw）", () => {
  // x [[1,2,3],[4,5,6]] × w [[1,2,3],[4,5,6]]（Kh=2 / Kw=3）→ 出力 1 点
  // Σ = 1·1 + 2·2 + 3·3 + 4·4 + 5·5 + 6·6 = 91。
  // Kh/Kw を転置して読むと 1·1 + 2·3 + 3·5 + 4·2 + 5·4 + 6·6 = 86 になり必ず食い違う。
  const exact = referenceConv2d(
    t([1, 1, 2, 3], [1, 2, 3, 4, 5, 6]),
    t([1, 1, 2, 3], [1, 2, 3, 4, 5, 6]),
    t([1], [0]),
    CONV2D,
  );
  assertEquals(exact.shape, [1, 1, 1, 1]);
  assertEquals([...exact.data], [91]);
  // bias は出力チャネルごとに 1 度だけ足す
  const biased = referenceConv2d(
    t([1, 1, 2, 3], [1, 2, 3, 4, 5, 6]),
    t([1, 1, 2, 3], [1, 2, 3, 4, 5, 6]),
    t([1], [9]),
    CONV2D,
  );
  assertEquals([...biased.data], [100]);
});

// MUST: 空間 2 軸は独立に効く。H と W で同じ値を使うケースだけだと、軸を取り違えた
// stride / padding / dilation が数値でも赤くならない。
Deno.test("conv2d の stride / padding / dilation は H と W で独立に効く", () => {
  // 3×4 入力・2×2 の識別可能な重み（1 / 10 / 100 / 1000）で、どのタップが効いたか値から読める
  const x = t([1, 1, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  const taps = t([1, 1, 2, 2], [1, 10, 100, 1000]);
  // stride [2,1] / padding [1,0]: H は (3+2−1−1)/2+1 = 2、W は (4−1−1)/1+1 = 3
  const asym = referenceConv2d(x, taps, t([1], [0]), {
    ...CONV2D,
    stride: [2, 1],
    padding: [1, 0],
  });
  assertEquals(asym.shape, [1, 1, 2, 3]);
  assertEquals([...asym.data], [2100, 3200, 4300, 10965, 12076, 13187]);

  // dilation [2,1]: H だけカーネル張りが 3 に伸びる（H は 1・W は 4）
  const dilated = referenceConv2d(
    t([1, 1, 3, 5], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
    taps,
    t([1], [0]),
    { ...CONV2D, dilation: [2, 1] },
  );
  assertEquals(dilated.shape, [1, 1, 1, 4]);
  assertEquals([...dilated.data], [13121, 14232, 15343, 16454]);

  // padding 域は 0 詰めで**読み飛ばす**（両端が padding だけを見る出力を持つ形）
  const padded = referenceConv2d(
    t([1, 1, 1, 1], [7]),
    t([1, 1, 1, 1], [3]),
    t([1], [0]),
    { ...CONV2D, padding: [1, 0] },
  );
  assertEquals(padded.shape, [1, 1, 3, 1]);
  assertEquals([...padded.data], [0, 21, 0]);
});

// MUST: Cin / Cout はどちらも 2 以上で互いに違う値を持つケースを用意する（片方が 1 だと
// グループの帯オフセットを落とす誤りが偶然一致する — conv_transpose1d の教訓）。
Deno.test("conv2d の groups は入力チャネル帯を絞る（depthwise と中間 groups）", () => {
  // Cin 4 / Cout 6 / groups 2 → in_per_group 2 / out_per_group 3。
  // oc 0..2 は ic 0,1 を、oc 3..5 は ic 2,3 だけを見る。
  const grouped = referenceConv2d(
    t([1, 4, 1, 2], [1, 2, 3, 4, 5, 6, 7, 8]),
    t([6, 2, 1, 1], [1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1]),
    t([6], [0, 0, 0, 0, 0, 0]),
    { ...CONV2D, groups: 2 },
  );
  assertEquals(grouped.shape, [1, 6, 1, 2]);
  // 帯オフセットを落とすと oc 3..5 が [1,2] / [3,4] / [4,6] に化ける
  assertEquals([...grouped.data], [1, 2, 3, 4, 4, 6, 5, 6, 7, 8, 12, 14]);

  // depthwise（groups = Cin = Cout = 3 → 重みの第 2 軸は 1）
  const depthwise = referenceConv2d(
    t([1, 3, 1, 1], [1, 1, 1]),
    t([3, 1, 1, 1], [2, 3, 4]),
    t([3], [0, 0, 0]),
    { ...CONV2D, groups: 3 },
  );
  assertEquals([...depthwise.data], [2, 3, 4]);
});

Deno.test("conv_transpose1d は [Cin,Cout,K] の重みで入力を stride 倍に伸ばす", () => {
  // x[1,1,3] = [1,2,3] / W[1,1,2] = [1,10] / stride 2 / padding 0 → 出力長 6
  // out[2i] = x[i]·w[0] / out[2i+1] = x[i]·w[1]
  const up = referenceConvTranspose1d(
    t([1, 1, 3], [1, 2, 3]),
    t([1, 1, 2], [1, 10]),
    t([1], [0]),
    { stride: 2, padding: 0 },
  );
  assertEquals(up.shape, [1, 1, 6]);
  assertEquals([...up.data], [1, 10, 2, 20, 3, 30]);

  // MUST: 重みは [Cin, Cout, K]。非対称チャネル（Cin 2 / Cout 1）で転置の取り違えを固定する。
  // 転置して読むと w[1] を w[0] の位置で拾い、値が [1+20, …] ではなくなる。
  const asym = referenceConvTranspose1d(
    t([1, 2, 2], [1, 2, 10, 20]),
    t([2, 1, 2], [1, 0, 0, 1]),
    t([1], [0]),
    { stride: 2, padding: 0 },
  );
  assertEquals(asym.shape, [1, 1, 4]);
  // ic=0 は w=[1,0]（偶数位置へ）/ ic=1 は w=[0,1]（奇数位置へ）
  assertEquals([...asym.data], [1, 10, 2, 20]);

  // padding は出力の両端を削る（2P == K − S: K 4 / S 2 / P 1 で出力長 L·S）
  const padded = referenceConvTranspose1d(
    t([1, 1, 3], [1, 2, 3]),
    t([1, 1, 4], [1, 2, 3, 4]),
    t([1], [0]),
    { stride: 2, padding: 1 },
  );
  assertEquals(padded.shape, [1, 1, 6]);
  // 手計算（out[ox] = Σ_k x[(ox+1−k)/2]·w[k]。(ox+1−k) が非負の偶数で商 < 3 のときのみ）:
  //   ox0: x0·w1                       = 2
  //   ox1: x1·w0 + x0·w2               = 2 + 3   = 5
  //   ox2: x1·w1 + x0·w3               = 4 + 4   = 8
  //   ox3: x2·w0 + x1·w2               = 3 + 6   = 9
  //   ox4: x2·w1 + x1·w3               = 6 + 8   = 14
  //   ox5: x2·w2                       = 9
  // padding を 0 扱いにすると全体が 1 つ左へずれる（両端の非対称性が検出器）。
  assertEquals([...padded.data], [2, 5, 8, 9, 14, 9]);

  // MUST: Cin / Cout が**どちらも 2 以上で互いに違う**ケースを 1 本持つ。転置して読む誤りは
  // Cin == 1 でも Cout == 1 でも添字が一致してしまい、上の 2 ケースだけでは緑のまま通る
  // （故障注入で実証済み — ADR 0015 の「非対称チャネル数で固定する」はここまで含む）。
  const wide = referenceConvTranspose1d(
    t([1, 2, 2], [1, 2, 10, 20]),
    // W[ic][oc][k] = 1..12（ic 主・oc 次・k 最内）
    t([2, 3, 2], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    t([3], [0, 0, 0]),
    { stride: 2, padding: 0 },
  );
  assertEquals(wide.shape, [1, 3, 4]);
  // 手計算: out[oc][2i+j] = x0[i]·W[0][oc][j] + x1[i]·W[1][oc][j]
  assertEquals([...wide.data], [
    71,
    82,
    142,
    164, // oc0
    93,
    104,
    186,
    208, // oc1
    115,
    126,
    230,
    252, // oc2
  ]);

  // bias は出力チャネルごとに 1 度だけ足される（寄与本数に比例しない）
  const biased = referenceConvTranspose1d(
    t([1, 1, 2], [1, 1]),
    t([1, 2, 2], [1, 1, 1, 1]),
    t([2], [5, 7]),
    { stride: 2, padding: 0 },
  );
  assertEquals([...biased.data], [6, 6, 6, 6, 8, 8, 8, 8]);
});

// deform_conv2d（ADR 0055）。1×1 カーネル・単一チャネルに絞ると出力が
// 「mask · bilinear(x, 基準 + offset)」そのものになるので、offset の 3 分岐（内側 / 隅だけ
// 範囲外 / 中心が範囲外）を手計算で固定できる。
Deno.test("deform_conv2d の 1×1 分岐は offset の 3 分岐を手計算どおりに踏む", () => {
  const x = t([1, 1, 2, 2], [1, 2, 3, 4]);
  const weight = t([1, 1, 1, 1], [1]);
  const bias = t([1], [0]);
  const ones = t([1, 1, 2, 2], [1, 1, 1, 1]);
  /** offset は `[y 平面 4 要素, x 平面 4 要素]`（偶数チャネル = y / 奇数 = x）。 */
  const offset = (shiftY: number, shiftX: number): RefTensor =>
    t([1, 2, 2, 2], [
      shiftY,
      shiftY,
      shiftY,
      shiftY,
      shiftX,
      shiftX,
      shiftX,
      shiftX,
    ]);
  const run = (off: RefTensor, mask: RefTensor = ones): readonly number[] => [
    ...applyReferenceOp("deform_conv2d", [x, weight, off, mask, bias], { padding: [0, 0] }).data,
  ];
  // offset 0 は素の 1×1 conv = 恒等（mask 1・bias 0）
  assertEquals(run(offset(0, 0)), [1, 2, 3, 4]);
  // y に +0.5: oy=0 は行 0/1 の中点、oy=1 は y=1.5 で**下側の隅だけ範囲外**（0 埋め）
  assertEquals(run(offset(0.5, 0)), [2, 3, 1.5, 2]);
  // y に −1.5: oy=0 は中心が −1.5 ≤ −1 で**タップ全体 0**、oy=1 は y=−0.5 で上側の隅だけ 0
  assertEquals(run(offset(-1.5, 0)), [0, 0, 0.5, 1]);
  // MUST: 偶数 = y / 奇数 = x。x に +1 は列方向のずらしで、y に +1 とは別の値になる
  assertEquals(run(offset(0, 1)), [2, 0, 4, 0]);
  assertEquals(run(offset(1, 0)), [3, 4, 0, 0]);
  // modulator は補間の**後**に掛かる（BiRefNet の値域 [0,2] の上端）
  assertEquals(run(offset(0.5, 0), t([1, 1, 2, 2], [2, 2, 2, 2])), [4, 6, 3, 4]);
});

// GRU 隠れ側スキャン（ADR 0056）。`W_hh = 0` に落とすと隠れ側の縮約が消え、残るのは
// **ゲートの並び（r / z / n）・reset ゲートの掛かる先・時間の進み方**だけになるので、
// そこを手計算で固定できる（縮約そのものは gpu_ops_test の CPU 参照突合が見る）。
//
// MUST: `b_hh` を非ゼロにする。reset ゲートは **b_hh 込みの隠れ側積**に掛かるので
// （`n = tanh(i_n + (Σ W·h + b_n)·r)`）、bias を reset の外へ出す誤り形はここでだけ赤くなる。
Deno.test("gru_scan は W_hh = 0 のときゲート並びと時間の進みを手計算どおりに踏む", () => {
  const hidden = 1;
  // gi[T=2, N=1, 3H] = [i_r, i_z, i_n] × 2 ステップ（ゲートごとに違う値 = 並びの取り違えが出る）
  const gi = t([2, 1, 3], [0.5, -0.25, 0.75, -1.5, 0.125, -0.5]);
  const h0 = t([1, 1], [0.4]);
  const weight = t([3, 1], [0, 0, 0]);
  const bias = t([3], [0.1, -0.2, 0.3]);
  /** 契約の式をそのまま JS の f64 で解いた期待値（f32 の丸め列は参照側の仕事）。 */
  const advance = (state: number, step: number): number => {
    const gate = (index: number): number => gi.data[step * 3 + index];
    const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));
    const reset = sigmoid(bias.data[0] + gate(0));
    const update = sigmoid(bias.data[1] + gate(1));
    const candidate = Math.tanh(gate(2) + bias.data[2] * reset);
    return (state - candidate) * update + candidate;
  };
  const first = advance(0.4, 0);
  const second = advance(first, 1);

  const forward = applyReferenceOp("gru_scan", [gi, h0, weight, bias]);
  assertEquals(forward.shape, [2, 1, hidden]);
  assertAlmostEquals(forward.data[0], first, 1e-6);
  assertAlmostEquals(forward.data[1], second, 1e-6);

  // 逆方向は末尾から走査するので、初期状態 h0 を受けるのは **t = 1**。書き出しは順方向の
  // 時間順のまま（`flip` を挟まない = op が走査方向を畳んでいる形の直接の門）。
  const reverseFirst = advance(0.4, 1);
  const reverseSecond = advance(reverseFirst, 0);
  const backward = applyReferenceOp("gru_scan_reverse", [gi, h0, weight, bias]);
  assertAlmostEquals(backward.data[1], reverseFirst, 1e-6);
  assertAlmostEquals(backward.data[0], reverseSecond, 1e-6);
});

// 状態が**バッチ間で混ざらない**ことと、`W_hh` が本当に隠れ側だけに掛かることの門。
// バッチ 0 の h0 だけを非ゼロにすると、W_hh の寄与はそのバッチにしか現れない。
Deno.test("gru_scan は W_hh をバッチごとの状態にだけ掛ける", () => {
  const gi = t([1, 2, 6], [0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3]);
  const weight = t([6, 2], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
  const bias = t([6], [0, 0, 0, 0, 0, 0]);
  // バッチ 0 の状態だけ非ゼロ（gi はバッチ間で同一なので、差は W_hh·h からしか出ない）
  const mixed = applyReferenceOp("gru_scan", [gi, t([2, 2], [0.7, -0.3, 0, 0]), weight, bias]);
  const zeroed = applyReferenceOp("gru_scan", [gi, t([2, 2], [0, 0, 0, 0]), weight, bias]);
  assertEquals(mixed.shape, [1, 2, 2]);
  // バッチ 1（後半 2 要素）は h0 = 0 なので両者で一致し、バッチ 0 は一致しない
  assertEquals([...mixed.data.slice(2)], [...zeroed.data.slice(2)]);
  assertNotEquals([...mixed.data.slice(0, 2)], [...zeroed.data.slice(0, 2)]);
});

// 退化ケース（offset 全 0・mask 全 1）が素の conv2d と**厳密に一致**する。新規原子の唯一の
// A/B オラクルで、実 GPU 側（gpu_deform_conv2d_test.ts）はこれをビット一致で見る。
// MUST: 非対称形（Cin ≠ Cout / Kh ≠ Kw / padding の H ≠ W）で固定する — 対称形では
// 重みレイアウトと offset のチャネル順の取り違えが値に出ない。
Deno.test("deform_conv2d は offset 0・mask 1 で conv2d と厳密に一致する", () => {
  const series = (count: number, step: number, base: number): readonly number[] =>
    Array.from({ length: count }, (_unused, index) => base + step * ((index % 13) - 6));
  const x = t([1, 2, 3, 4], series(24, 0.25, 0.5));
  const weight = t([3, 2, 2, 3], series(36, 0.125, -0.25));
  const bias = t([3], [0.5, -0.25, 0.75]);
  // padding [1,0] → Hout = 3 + 2 − 1 = 4 / Wout = 4 + 0 − 2 = 2
  const offset = t([1, 12, 4, 2], new Array(96).fill(0));
  const mask = t([1, 6, 4, 2], new Array(48).fill(1));
  const deform = applyReferenceOp("deform_conv2d", [x, weight, offset, mask, bias], {
    padding: [1, 0],
  });
  const plain = applyReferenceOp("conv2d", [x, weight, bias], {
    stride: [1, 1],
    padding: [1, 0],
    dilation: [1, 1],
    groups: 1,
  });
  assertEquals(deform.shape, [1, 3, 4, 2]);
  assertEquals(deform.shape, plain.shape);
  assertEquals([...deform.data], [...plain.data]);
});

// offset の NaN は「範囲外」ではない。正の形の範囲判定（`> −1 && < in`）だけだと NaN が
// false 側 = 0 寄与に落ちて沈黙誤値になるので、NaN は出力へ伝播させる（ADR 0055 決定 5）。
Deno.test("deform_conv2d は NaN の offset を 0 に落とさず伝播させる", () => {
  const x = t([1, 1, 2, 2], [1, 2, 3, 4]);
  const weight = t([1, 1, 1, 1], [1]);
  const bias = t([1], [0]);
  const mask = t([1, 1, 2, 2], [1, 1, 1, 1]);
  // y 平面の先頭 1 要素だけ NaN（残りは 0）→ 出力の先頭要素だけ NaN
  const offset = t([1, 2, 2, 2], [Number.NaN, 0, 0, 0, 0, 0, 0, 0]);
  const out = applyReferenceOp("deform_conv2d", [x, weight, offset, mask, bias], {
    padding: [0, 0],
  });
  assertEquals(Number.isNaN(out.data[0]), true, "NaN の offset は出力へ伝播する");
  assertEquals([...out.data.slice(1)], [2, 3, 4], "他の要素は巻き添えにならない");
  // ±Inf は torch と同じく「範囲外 = 0」（NaN とは別扱い）
  const infinite = t([1, 2, 2, 2], [
    Number.POSITIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
    0,
    0,
    0,
    0,
    0,
    0,
  ]);
  const clipped = applyReferenceOp("deform_conv2d", [x, weight, infinite, mask, bias], {
    padding: [0, 0],
  });
  assertEquals([...clipped.data], [0, 0, 3, 4]);
});

// MUST: 4 近傍と重みが手計算で追える形だけで固める（新規原子は A/B オラクルが無いので、
// 参照実装そのものが「読んで正しさが分かる」ことが門になる）。値は全て f32 で厳密。
Deno.test("upsample_bilinear2d は align_corners の 4 近傍を手計算どおりに混ぜる", () => {
  // 2×2 → 3×3。scale = (2−1)/(3−1) = 0.5 なので、出力の偶数位置は入力そのまま・
  // 奇数位置は隣り合う 2 点の中点になる。
  const up = applyReferenceOp("upsample_bilinear2d", [t([1, 1, 2, 2], [1, 2, 3, 4])], {
    output_size: [3, 3],
  });
  assertEquals(up.shape, [1, 1, 3, 3]);
  assertEquals([...up.data], [1, 1.5, 2, 2, 2.5, 3, 3, 3.5, 4]);

  // 3×3 → 2×2（縮小）。scale = 2 なので λ は 0 で、**両端の 2 点しか読まない**
  // （antialias / area とは違う torch の仕様どおりの情報落ち）。
  const down = applyReferenceOp(
    "upsample_bilinear2d",
    [t([1, 1, 3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9])],
    { output_size: [2, 2] },
  );
  assertEquals([...down.data], [1, 3, 7, 9]);

  // 入力の高さ 1（ASPP の GAP 枝と同型）。H の scale は 0 で全行が同じ行を読み、
  // W だけが本当に補間される。
  const broadcast = applyReferenceOp("upsample_bilinear2d", [t([1, 1, 1, 2], [10, 20])], {
    output_size: [2, 3],
  });
  assertEquals([...broadcast.data], [10, 15, 20, 10, 15, 20]);
});

// align_corners = True の定義そのもの: 出力の 4 隅は入力の 4 隅と**厳密に一致**する
// （λ が 0 か 1 に潰れるので丸めが入らない）。H と W を別長にして軸の取り違えも同時に見る。
Deno.test("upsample_bilinear2d は出力の 4 隅を入力の 4 隅へ厳密に一致させる", () => {
  const x = t([1, 1, 2, 3], [1, 2, 3, 4, 5, 6]);
  const out = applyReferenceOp("upsample_bilinear2d", [x], { output_size: [5, 7] });
  assertEquals(out.shape, [1, 1, 5, 7]);
  const at = (row: number, column: number): number => out.data[row * 7 + column];
  assertEquals(at(0, 0), 1, "左上");
  assertEquals(at(0, 6), 3, "右上");
  assertEquals(at(4, 0), 4, "左下");
  assertEquals(at(4, 6), 6, "右下");
  // 端の**行と列**も入力の端の 1 次元補間になる（4 隅だけでは軸の取り違えが残る）
  assertEquals([at(0, 0), at(0, 3), at(0, 6)], [1, 2, 3], "上端の行");
  assertEquals([at(0, 0), at(2, 0), at(4, 0)], [1, 2.5, 4], "左端の列");
});

// ここから 3 本（deform_conv2d / upsample_bilinear2d / gru_scan）は **applyReferenceOp を
// 経由しない直接呼びの門**。上の入口経由の門と違い、export された関数そのものを呼ぶ経路を
// 固定する（結線を差し替えても関数側の契約が独立に赤くなる）。

// MUST: オラクルは referenceConv2d（独立の実装）に取る。offset を一様な整数にすると
// 源座標は `oy − ph + kh + dy` = 「padding を `ph − dy` に付け替えた conv2d」になるので、
// deform 側の双線形タップを再計算せずに ① タップの整数位置 ② offset のチャネル並び
// （偶数 = y / 奇数 = x）③ 範囲外のゼロ埋め を conv2d の値で縛れる。
// MUST: dy ≠ dx にする。同じずらし量だと y / x 平面を取り違えた読みが値でも一致する。
Deno.test("deform_conv2d は一様な整数 offset を conv2d の窓ずらしとして踏む（直接呼び）", () => {
  // 1/8 刻みの決め打ち列（f32 で厳密・0 も対称性も持たない）
  const series = (count: number, seed: number): readonly number[] =>
    Array.from({ length: count }, (_unused, index) => (((index * 7 + seed) % 17) - 8) / 8);
  // Cin 2 ≠ Cout 3 / Kh 2 ≠ Kw 3（重みレイアウトの取り違えが値に出る形）
  const x = t([1, 2, 4, 5], series(40, 0));
  const weight = t([3, 2, 2, 3], series(36, 5));
  const bias = t([3], [0.5, -0.25, 0.75]);
  // padding [2,2] → Hout = 4 + 4 − 1 = 7 / Wout = 5 + 4 − 2 = 7
  const attrs = { padding: [2, 2] } as const;
  const mask = t([1, 6, 7, 7], new Array(6 * 49).fill(1));
  /** 全タップ・全画素で同じずらし量を持つ offset（偶数チャネル = y / 奇数 = x）。 */
  const uniformOffset = (shiftY: number, shiftX: number): RefTensor =>
    t(
      [1, 12, 7, 7],
      Array.from(
        { length: 12 * 49 },
        (_unused, index) => Math.floor(index / 49) % 2 === 0 ? shiftY : shiftX,
      ),
    );

  // ① 退化ケース（offset 0・mask 1）は素の conv2d と厳密一致
  const degenerate = referenceDeformConv2d(x, weight, uniformOffset(0, 0), mask, bias, attrs);
  const plain = referenceConv2d(x, weight, bias, { ...CONV2D, padding: [2, 2] });
  assertEquals(degenerate.shape, [1, 3, 7, 7]);
  assertEquals(degenerate.shape, plain.shape);
  assertEquals([...degenerate.data], [...plain.data]);

  // ② dy = 2 / dx = 1 → 源座標は `oy + kh` と `ox − 1 + kw` = padding [0,1] の conv2d。
  // conv2d 側の出力（3×5）は deform 側（7×7）の左上の窓にそのまま埋まっているはず。
  const shifted = referenceDeformConv2d(x, weight, uniformOffset(2, 1), mask, bias, attrs);
  const window = referenceConv2d(x, weight, bias, { ...CONV2D, padding: [0, 1] });
  assertEquals(window.shape, [1, 3, 3, 5]);
  const cropped: number[] = [];
  for (let channel = 0; channel < 3; channel += 1) {
    for (let row = 0; row < 3; row += 1) {
      for (let column = 0; column < 5; column += 1) {
        cropped.push(shifted.data[(channel * 7 + row) * 7 + column]);
      }
    }
  }
  // y / x を取り違えると padding [1,0] の conv2d になり、この窓は一致しない
  assertEquals(cropped, [...window.data]);
});

// MUST: λ は 1/4 刻みにする。既存の 1/2 ずらしのケースは `(1 − λ)` と `λ` が同値なので、
// 4 隅の重みを左右／上下で取り違えた実装が緑のまま通る。
// MUST: 入力は平面（座標に線形）にしない。平面だと双線形の交差項が消えて、重みの配り方が
// 間違っていても値が一致する（ここは 4 点を 1,2,3,7 = 非平面に取る）。
Deno.test("deform_conv2d の 4 隅の重みは 1/4 刻みの λ で手計算どおり（直接呼び）", () => {
  // x[0,0] = [[1,2],[3,7]] / 1×1 カーネル w = 2 / mask = 3 / bias = 0.5・padding 0
  const x = t([1, 1, 2, 2], [1, 2, 3, 7]);
  const weight = t([1, 1, 1, 1], [2]);
  const bias = t([1], [0.5]);
  const mask = t([1, 1, 2, 2], [3, 3, 3, 3]);
  const attrs = { padding: [0, 0] } as const;
  /** 出力 (0,0) だけをずらす offset（`[y 平面 4 要素, x 平面 4 要素]`）。 */
  const corner = (shiftY: number, shiftX: number): RefTensor =>
    t([1, 2, 2, 2], [shiftY, 0, 0, 0, shiftX, 0, 0, 0]);

  // 源 (0.25, 0.5): 4 隅の重みは (1−λy)(1−λx), (1−λy)λx, λy(1−λx), λyλx = 0.375, 0.375,
  //   0.125, 0.125 → 補間値 = 0.375·1 + 0.375·2 + 0.125·3 + 0.125·7 = 2.375
  //   → 0.5 + (3 · 2.375) · 2 = 14.75（mask は補間の後・weight はさらに後・bias は 1 度だけ）
  // ずらさない 3 画素は素のサンプル: 0.5 + 3·x·2 = 12.5 / 18.5 / 42.5
  const tilted = referenceDeformConv2d(x, weight, corner(0.25, 0.5), mask, bias, attrs);
  assertEquals(tilted.shape, [1, 1, 2, 2]);
  assertEquals([...tilted.data], [14.75, 12.5, 18.5, 42.5]);

  // 源 (0.5, 0.25) は上と λ が入れ替わるだけだが、非平面な入力では別の値になる:
  //   0.375·1 + 0.125·2 + 0.375·3 + 0.125·7 = 2.625 → 0.5 + (3 · 2.625) · 2 = 16.25
  const mirrored = referenceDeformConv2d(x, weight, corner(0.5, 0.25), mask, bias, attrs);
  assertEquals([...mirrored.data], [16.25, 12.5, 18.5, 42.5]);
});

// align_corners = True の定義（源座標 `s = i·(In − 1)/(Out − 1)`）を、**座標に線形な入力**で
// 縛る。双線形補間は線形関数を厳密に再現する（1 軸で `(1 − λ)·f(k) + λ·f(k+1) = f(k + λ)`、
// 2 軸目も同じ）ので、期待値は「源座標を線形式へ代入した値」— 実装の畳み方とは独立に出る。
// align_corners = False（`s = (i + 0.5)·In/Out − 0.5`）なら源座標がずれて必ず赤くなる。
// MUST: N と C を**どちらも 2** にして平面ごとに違う切片を持たせる（平面添字 `n·C + c` の
// 取り違えは片方が 1 だと現れない）。H と W で傾きを変えて軸の取り違えも同時に見る。
Deno.test("upsample_bilinear2d は align_corners の源座標で線形入力を再現する（直接呼び）", () => {
  const heightIn = 3;
  const widthIn = 4;
  /** f(n,c,y,x) = (0.5·n + 0.25·c) + 0.75·y − 0.5·x（座標に線形 = 双線形の不動点）。 */
  const value = (item: number, channel: number, y: number, x: number): number =>
    0.5 * item + 0.25 * channel + 0.75 * y - 0.5 * x;
  const input: number[] = [];
  for (let item = 0; item < 2; item += 1) {
    for (let channel = 0; channel < 2; channel += 1) {
      for (let row = 0; row < heightIn; row += 1) {
        for (let column = 0; column < widthIn; column += 1) {
          input.push(value(item, channel, row, column));
        }
      }
    }
  }
  const x = t([2, 2, heightIn, widthIn], input);
  /** align_corners = True の源座標（Out = 1 は 0 固定）。 */
  const source = (index: number, lengthIn: number, lengthOut: number): number =>
    lengthOut > 1 ? index * ((lengthIn - 1) / (lengthOut - 1)) : 0;
  const check = (heightOut: number, widthOut: number): void => {
    const up = referenceUpsampleBilinear2d(x, { output_size: [heightOut, widthOut] });
    assertEquals(up.shape, [2, 2, heightOut, widthOut]);
    for (let item = 0; item < 2; item += 1) {
      for (let channel = 0; channel < 2; channel += 1) {
        for (let row = 0; row < heightOut; row += 1) {
          for (let column = 0; column < widthOut; column += 1) {
            const expected = value(
              item,
              channel,
              source(row, heightIn, heightOut),
              source(column, widthIn, widthOut),
            );
            const flat = ((item * 2 + channel) * heightOut + row) * widthOut + column;
            assertAlmostEquals(
              up.data[flat],
              expected,
              1e-6,
              `[${item},${channel},${row},${column}] @ ${heightOut}×${widthOut}`,
            );
          }
        }
      }
    }
  };
  // 整数倍（scale = 0.5 ちょうど）と非整数倍（scale = 2/3 と 0.6）の両方
  check(5, 7);
  check(4, 6);
});

// W_hh の縮約まで含めた直接門。gi を「隠れ側の値をちょうど打ち消す」ように置くと、
// r = σ(0) = 0.5 / z = σ(0) = 0.5 / n = tanh(0) = 0 になり、更新式は `h' = h·0.5` へ潰れる。
// 打ち消しの成立自体が ① 行の並び（gate 主・`(g·H + j)` 行）② 列 `k` の並び ③ b_hh が
// reset の**内側**に入ること ④ 状態がステップ間で運ばれること を全部要求するので、どれか 1 つ
// でも崩れると σ / tanh の引数が 0 から外れて値が一致しなくなる（全て f32 で厳密な値）。
//
// 手計算（h = [2,1]・W と b は下のリテラル）:
//   gh_r = [1·2 − 2·1 + 0.5, 0.5·2 + 1·1 − 1] = [0.5, 1]
//   gh_z = [−1·2 + 3·1 + 0.25, 2·2 − 0.5·1 + 0] = [1.25, 3.5]
//   gh_n = [3·2 + 1·1 − 1, −2·2 + 2·1 + 0.5]   = [6, −1.5]
//   → i_r = −gh_r / i_z = −gh_z / i_n = −gh_n·0.5 と置けば h' = [2,1]·0.5 = [1, 0.5]
//   次のステップは h = [1, 0.5] に対して同じ置き方: gh_r = [0.5, 0] / gh_z = [0.75, 1.75] /
//   gh_n = [2.5, −0.5] → h'' = [0.5, 0.25]
Deno.test("gru_scan は W_hh の縮約と状態の引き回しを手計算どおりに踏む（直接呼び）", () => {
  const weight = t([6, 2], [1, -2, 0.5, 1, -1, 3, 2, -0.5, 3, 1, -2, 2]);
  const bias = t([6], [0.5, -1, 0.25, 0, -1, 0.5]);
  const h0 = t([1, 2], [2, 1]);
  // 各ステップの gi は [i_r(2), i_z(2), i_n(2)]
  const firstStep = [-0.5, -1, -1.25, -3.5, -3, 0.75] as const;
  const secondStep = [-0.5, 0, -0.75, -1.75, -1.25, 0.25] as const;

  const forward = referenceGruScan(
    "gru_scan",
    t([2, 1, 6], [...firstStep, ...secondStep]),
    h0,
    weight,
    bias,
  );
  assertEquals(forward.shape, [2, 1, 2]);
  assertEquals([...forward.data], [1, 0.5, 0.5, 0.25]);

  // 逆方向は**走査順だけ**が反転する。時間を入れ替えて同じ列を与えれば、h0 を受けるのは
  // t = 1 側で、書き出しは順方向の時間添字のまま（t=1 に第 1 ステップ・t=0 に第 2 ステップ）。
  const backward = referenceGruScan(
    "gru_scan_reverse",
    t([2, 1, 6], [...secondStep, ...firstStep]),
    h0,
    weight,
    bias,
  );
  assertEquals([...backward.data], [0.5, 0.25, 1, 0.5]);
});

Deno.test("applyReferenceOp は契約表の kind で分岐し、アリティ違反を拒否する", () => {
  assertEquals([...applyReferenceOp("relu", [t([2], [-1, 2])]).data], [0, 2]);
  assertEquals([...applyReferenceOp("sum", [t([1, 2], [3, 4])], { dim: 1 }).data], [7]);
  assertEquals([...applyReferenceOp("cast", [t([2], [1.9, 0])], { to: "i32" }).data], [1, 0]);
  assertThrows(() => applyReferenceOp("add", [t([2], [1, 2])]), ReferenceOpError);
  // スロット別 dtype 契約の op も同じ入口を通る（テストがグラフをそのまま辿れる）
  assertEquals(
    [...applyReferenceOp("bmm", [t([1, 1, 2], [1, 2]), t([1, 2, 1], [3, 4])]).data],
    [11],
  );
  assertEquals(
    [...applyReferenceOp("gather", [t([1, 3], [7, 8, 9]), i32([1, 2], [2, 0])]).data],
    [9, 7],
  );
  // cast の attrs は契約検査を通っていなくても参照実装側で落ちる
  assertThrows(() => applyReferenceOp("cast", [t([2], [1, 2])]), OpContractError);
  assertThrows(() => refTensor([2, 2], Float32Array.from([1, 2])), ReferenceOpError);

  // レイアウト op（ADR 0011）— reshape / expand は目標形が要る
  assertEquals(applyReferenceOp("reshape", [t([4], [1, 2, 3, 4])], {}, [2, 2]).shape, [2, 2]);
  const transposed = applyReferenceOp("permute", [t([2, 2], [1, 2, 3, 4])], { dims: [1, 0] });
  assertEquals([...transposed.data], [1, 3, 2, 4]);
  assertEquals(applyReferenceOp("expand", [i32([1, 2], [5, 6])], {}, [2, 2]).shape, [2, 2]);
  // MUST: 目標形を渡さずに黙って推測しない
  assertThrows(() => applyReferenceOp("reshape", [t([4], [1, 2, 3, 4])]), ReferenceOpError);
  assertThrows(() => applyReferenceOp("expand", [i32([1, 2], [5, 6])]), ReferenceOpError);
  assertThrows(() => applyReferenceOp("permute", [t([2, 2], [1, 2, 3, 4])]), OpContractError);

  // レイアウト第 2 群（ADR 0014）— 出力 shape は attrs から決まるので目標形は要らない
  assertEquals(
    [...applyReferenceOp("slice", [t([2, 2], [1, 2, 3, 4])], { dim: 1, start: 1, end: 2 }).data],
    [2, 4],
  );
  assertEquals(
    [...applyReferenceOp("cat", [t([1], [1]), t([2], [2, 3])], { dim: 0 }).data],
    [1, 2, 3],
  );
  assertEquals(
    [...applyReferenceOp("pad", [t([1, 2], [1, 2])], { left: 1, right: 0 }).data],
    [0, 1, 2],
  );
  assertEquals([...applyReferenceOp("flip", [t([3], [1, 2, 3])], { dim: 0 }).data], [3, 2, 1]);
  // 可変アリティは**下限**（1 本の cat は受理しない）
  assertThrows(() => applyReferenceOp("cat", [t([2], [1, 2])], { dim: 0 }), ReferenceOpError);

  // ADR 0017 の 3 本も同じ入口を通る（結線漏れは kind の網羅で型が止めるが、実行経路は別）
  assertEquals(
    [...applyReferenceOp("clamp_min", [t([2], [-1, 2])], { min: 0 }).data],
    [0, 2],
  );
  // mean(x²) = 9 → rms 3 → 3/3·2 = 2（eps は 1e-12 で無視できる大きさ）
  assertAlmostEquals(
    applyReferenceOp("rms_norm", [t([1, 1], [3]), t([1], [2])], { eps: 1e-12 }).data[0],
    2,
    1e-6,
  );
  assertEquals(
    [
      ...applyReferenceOp(
        "conv2d",
        [t([1, 1, 1, 1], [2]), t([1, 1, 1, 1], [3]), t([1], [1])],
        { stride: [1, 1], padding: [0, 0], dilation: [1, 1], groups: 1 },
      ).data,
    ],
    [7],
  );
  assertThrows(
    () => applyReferenceOp("rms_norm", [t([1, 1], [3])], { eps: 1e-6 }),
    ReferenceOpError,
  );
});

Deno.test("compareTensors は dtype で許容誤差を選び、dtype 混在を拒否する", () => {
  assertEquals(EXACT_TOLERANCE, { atol: 0, rtol: 0 });
  // f32 は既定の許容誤差で通る差
  assertEquals(compareTensors(t([1], [1.000001]), t([1], [1])).pass, true);
  // 整数は 1 でもずれたら不合格
  assertEquals(compareTensors(i32([2], [1, 2]), i32([2], [1, 2])).pass, true);
  assertEquals(compareTensors(i32([2], [1, 3]), i32([2], [1, 2])).pass, false);
  assertEquals(compareTensors(bools([2], [0, 1]), bools([2], [0, 0])).pass, false);
  assertThrows(() => compareTensors(i32([1], [1]), t([1], [1])), AllcloseError);
});

Deno.test("allclose は atol + rtol·|ref| で判定する", () => {
  const expected = Float32Array.from([0, 1, 100]);
  assertEquals(allclose(expected, expected).pass, true);
  // atol=1e-5 の範囲内 / rtol=1e-3 が効く範囲内
  assertEquals(allclose(Float32Array.from([5e-6, 1.0005, 100.05]), expected).pass, true);
  const report = allclose(Float32Array.from([0, 1, 101]), expected);
  assertEquals(report.pass, false);
  assertEquals(report.failCount, 1);
  assertEquals(report.worstIndex, 2);
  assertEquals(formatAllclose(report).startsWith("fail=1"), true);
});

Deno.test("allclose は NaN / Inf をどちらの側でも不合格にする", () => {
  const expected = Float32Array.from([1, 2]);
  const withNan = allclose(Float32Array.from([Number.NaN, 2]), expected);
  assertEquals(withNan.pass, false);
  assertEquals(withNan.nonFiniteCount, 1);
  assertEquals(withNan.maxAbsError, Number.POSITIVE_INFINITY);

  const referenceInf = allclose(expected, Float32Array.from([Number.POSITIVE_INFINITY, 2]));
  assertEquals(referenceInf.pass, false);
  assertEquals(referenceInf.nonFiniteCount, 1);

  assertEquals(DEFAULT_TOLERANCE.atol, 1e-5);
  assertEquals(DEFAULT_TOLERANCE.rtol, 1e-3);
  assertThrows(() => allclose(expected, Float32Array.from([1])), AllcloseError);
});
