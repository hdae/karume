// w8a8 参照オラクル（src/reference/i8a8.ts）の **CPU 単独**の門。
//
// 参照は「GPU と atol=0 で一致する」ことを主張する数値契約の正本で、その主張は
// tests/gpu_i8a8_test.ts などの実 GPU 突合が担う。ところが突合は**参照と GPU が同じ向きに
// 壊れたら通る**（恒真化）ため、参照そのものが仕様（設計 doc §4.2 / 各 docstring の MUST）を
// 満たしていることは、GPU に依らない既知値の門で別に固定しなければならない。
//
// MUST: このファイルは `ignore: !GPU_AVAILABLE` を持たない。アダプタ無し環境（CI）でも
// 参照側の退行を検出できることがこのファイルの存在理由そのもの。
// MUST: 期待値は参照の実装式ではなく**仕様から独立に**書き下した既知値で持つ（同じ式を
// 写すと故障注入で落ちない恒真な門になる）。tests/reference_ops_test.ts と同じ規律。
//
// 検出できる変異（設計時に確認した故障注入）:
// - `Math.max(..., F32_TINY)` の床を外す → 全ゼロ行の scale が 0 になり ①が落ちる
// - `sawNan` 分岐を落として素の `Math.max` にする → NaN が飲まれて ②が落ちる
// - 量子化格子を ±128 にする → 絶対値最大が 128（= i8 では −128）へ乗り ③が落ちる
// - 乗算順序を逐次形へ崩す → ⑤が落ちる（畳み形と値が割れるデータを選んである）
// - group を跨いで i32 のまま足し込む（丸めが 1 回に減る）→ ⑤c が落ちる（w4a8 の畳み順）
// - `roundTiesToEven` を `Math.round` にする → ⑦が落ちる（同点が +∞ 方向へ倒れる）

import { assert, assertEquals, assertNotEquals } from "@std/assert";
import {
  quantizeRowsReference,
  referenceAttentionPvI8a8Core,
  referenceAttentionPvQuant,
  referenceAttentionQkI8a8,
  referenceLinearI8a8,
  referenceLinearW4a8,
} from "../src/reference/i8a8.ts";

/**
 * f32 の最小 normal（`torch.finfo(float32).tiny`）。
 *
 * MUST: 参照の定数を輸入しない（参照が床を落としたときに期待値も一緒に動くと門が消える）。
 */
const F32_TINY = 1.1754943508222875e-38;

// ---------------------------------------------------------------------------
// ① quantizeRowsReference — 全ゼロ行の scale 床
// ---------------------------------------------------------------------------

Deno.test("i8a8 参照: 全ゼロ行の scale は f32 tiny（床が無いと 0/0 = NaN になる）", () => {
  const { q, scale } = quantizeRowsReference([0, 0, 0, 0], 1, 4);
  assertEquals(scale[0], F32_TINY, "全ゼロ行の scale");
  // 床があるので x/s は 0/tiny = 0 で厳密に復元される（床が無いと NaN → i8 化で 0 に化け、
  // q だけを見ても退行が見えない — scale の側が検出器）
  assertEquals([...q], [0, 0, 0, 0], "全ゼロ行の q");
});

// ---------------------------------------------------------------------------
// ② quantizeRowsReference — 非有限の伝播（行の独立性込み）
// ---------------------------------------------------------------------------

Deno.test("i8a8 参照: NaN / ±Inf は行 scale へ伝播し、隣の行を汚さない", () => {
  // 行 0 = NaN 混じり / 行 1 = +Inf / 行 2 = −Inf / 行 3 = 通常
  const x = [
    1,
    Number.NaN,
    -3,
    2,
    Number.POSITIVE_INFINITY,
    1,
    2,
    Number.NEGATIVE_INFINITY,
    1,
    1,
    2,
    3,
  ];
  const { q, scale } = quantizeRowsReference(x, 4, 3);
  // MUST: 素の Math.max は NaN を飲む（`NaN > amax` が偽）ので amax = 3 の有限 scale になる
  assert(Number.isNaN(scale[0]), `NaN 行の scale が ${scale[0]}`);
  assertEquals(scale[1], Number.POSITIVE_INFINITY, "+Inf 行の scale");
  // |−Inf| = +Inf なので絶対値最大は +Inf（scale に符号は残らない）
  assertEquals(scale[2], Number.POSITIVE_INFINITY, "−Inf 行の scale");
  // 通常行は無傷 — s = f32(3 · (1/127))、q = round(x/s)
  assertEquals(scale[3], 0.023622047156095505, "通常行の scale");
  assertEquals([...q.slice(9, 12)], [42, 85, 127], "通常行の q");
});

// ---------------------------------------------------------------------------
// ③ quantizeRowsReference — 格子の端（±127）
// ---------------------------------------------------------------------------

Deno.test("i8a8 参照: 絶対値最大の要素は ±127 へちょうど乗る（±128 の格子では割れる）", () => {
  // amax = 127 にすると s = f32(127 · (1/127)) = 1.0 ちょうどになり、x/s が入力そのもの
  // （除数 1.0 の除算は厳密）なので格子の端だけを切り出せる。
  const { q, scale } = quantizeRowsReference([127, -127, 3, 0], 1, 4);
  assertEquals(scale[0], 1, "s は 1.0 ちょうど");
  // 格子を ±128 にすると s = f32(127/128) < 1 になり、127/s = 128 → i8 では −128 に化ける
  assertEquals([...q], [127, -127, 3, 0], "±127 の端点");
});

// ---------------------------------------------------------------------------
// ④⑤ referenceLinearI8a8 — 既知値と乗算順序
// ---------------------------------------------------------------------------

Deno.test("i8a8 参照: linear の既知値（整数内積 → f32 dequant → bias）", () => {
  // x = [8, 3] → amax = 8・s = f32(8/127) = 0.06299212574958801・q = [127, 48]
  const x = [8, 3];
  const quantized = quantizeRowsReference(x, 1, 2);
  assertEquals([...quantized.q], [127, 48], "活性の量子化値");
  assertEquals(quantized.scale[0], 0.06299212574958801, "活性の行 scale");
  // W[2,2] の量子化値（行 = 出力チャネル）と per-channel scale
  //   acc[0] = 127·1 + 48·2   =  223 → 223·(s·0.5)  + 1  ≈  7.0236218 + 1
  //   acc[1] = 127·(−3) + 48·4 = −189 → −189·(s·0.25) − 1 ≈ −2.9763780 − 1
  const out = referenceLinearI8a8({
    x,
    weight: [1, 2, -3, 4],
    weightScale: [0.5, 0.25],
    bias: [1, -1],
    m: 1,
    n: 2,
    k: 2,
  });
  assertEquals([...out], [8.023621559143066, -3.9763779640197754], "w8a8 linear の出力");
});

Deno.test("i8a8 参照: linear は xs·wscale を先に畳む（逐次形と値が割れるデータで固定）", () => {
  // acc = 127・s = f32(0.001/127)・wscale = f32(0.001) の組は、
  //   畳み形 f32(127 · f32(s·w))   = 9.999999974752427e-7
  //   逐次形 f32(f32(127·s) · w)   = 1.0000001111620804e-6
  // と丸めの位置が違う（GPU 側は fma の第 2 引数を 1 つの f32 に畳む — 設計 §4.2）。
  const out = referenceLinearI8a8({
    x: [0.001, 0],
    weight: [1, 0],
    weightScale: [0.001],
    bias: [0],
    m: 1,
    n: 1,
    k: 2,
  });
  assertEquals(out[0], 9.999999974752427e-7, "畳み形の値");
  assertNotEquals(out[0], 1.0000001111620804e-6, "逐次形へ崩れている");
});

// ---------------------------------------------------------------------------
// ⑤b⑤c referenceLinearW4a8 — 既知値と group ごとの畳み順
// ---------------------------------------------------------------------------

Deno.test("w4a8 参照: linear の既知値（group ごとに i32 → group 境界で f32 flush → xs）", () => {
  // x = [8, 3, −2, 5] → amax = 8・s = f32(8 · f32(1/127)) = 0.06299212574958801
  //   q = [127, 48, −32, 79]（半整数から十分離れているので偶数丸めの分岐は踏まない）
  // group_size 2 なので行内は 2 group。scale は 2 冪だけを使い、flush までの算術を
  // f32 で厳密にしてある（最後の xs の乗算だけが丸める形 = 検算が手でできる）。
  //   行 0: g0 acc = 127·1 + 48·2   =  223 → 223·0.5   = 111.5
  //         g1 acc = −32·(−3) + 79·4 =  412 → 412·0.25  = 103   → accf = 214.5
  //         out = f32(214.5 · s + 1)
  //   行 1: g0 acc = 127·7 + 48·(−7) =  553 → 553·0.125 =  69.125
  //         g1 acc = −32·5 + 79·(−6) = −634 → −634·2    = −1268 → accf = −1198.875
  //         out = f32(−1198.875 · s − 1)
  const out = referenceLinearW4a8({
    x: [8, 3, -2, 5],
    weight: [1, 2, -3, 4, 7, -7, 5, -6],
    weightScale: [0.5, 0.25, 0.125, 2],
    bias: [1, -1],
    m: 1,
    n: 2,
    k: 4,
    groupSize: 2,
  });
  assertEquals([...out], [14.511811256408691, -76.51968383789062], "w4a8 linear の出力");
});

Deno.test("w4a8 参照: group ごとに f32 へ畳む（全 group を 1 本の和にした形と値が割れる）", () => {
  // amax = 127 の行は s = f32(127 · f32(1/127)) = 1.0 ちょうど（除数 1.0 の除算は厳密）
  // なので q = x そのもの・最後の `accf · xs` も恒等になり、**畳み順だけ**が出力に残る。
  //   g0 acc = 127·(−7) + 37·(−3) = −1000 / g1 acc = 121·(−7) + 20·(−6) = −967
  //   group ごと  : f32(f32(−967 · s) + f32(−1000 · s)) = −33.45867156982422
  //   1 本の和    : f32(−1967 · s)                      = −33.45866775512695
  // MUST: group を跨いで i32 のまま足し込む実装（丸めが 1 回に減る）はここで落ちる。
  const scale = 0.017009999603033066;
  const out = referenceLinearW4a8({
    x: [127, 37, 121, 20],
    weight: [-7, -3, -7, -6],
    weightScale: [scale, scale],
    bias: [0],
    m: 1,
    n: 1,
    k: 4,
    groupSize: 2,
  });
  assertEquals(out[0], -33.45867156982422, "group ごとの畳み");
  assertNotEquals(out[0], -33.45866775512695, "全 group を 1 本の和にまとめている");
});

// ---------------------------------------------------------------------------
// ⑥ referenceAttentionQkI8a8 — 半スケールが q / k の両側へ掛かる
// ---------------------------------------------------------------------------

Deno.test("i8a8 参照: QK の半スケールは q と k の双方へ掛かる（scale 2 倍 → 出力 4 倍）", () => {
  // q = k = [8, 3] → 両側とも q̂ = [127, 48]・s = f32(8/127)
  //   acc = 127² + 48² = 18433 / S = f32(acc · f32((s·scale)·(s·scale)))
  const input = { q: [8, 3], k: [8, 3], batch: 1, m: 1, n: 1, d: 2 };
  const half = referenceAttentionQkI8a8({ ...input, scale: 0.5 });
  const full = referenceAttentionQkI8a8({ ...input, scale: 1 });
  assertEquals(half[0], 18.285572052001953, "scale = 0.5 の S");
  // 2 倍は f32 で厳密なので、片側にしか掛かっていなければ 2 倍にしかならない
  assertEquals(full[0], 73.14228820800781, "scale = 1.0 の S");
  assertEquals(full[0], half[0] * 4, "scale が二乗で効いていない");
});

// ---------------------------------------------------------------------------
// ⑦ referenceAttentionPvQuant — 127·exp(0) と偶数丸め
// ---------------------------------------------------------------------------

Deno.test("i8a8 参照: P̃ の量子化は round(127·exp(S−m)) で、同点は偶数側へ倒れる", () => {
  // 行の最大 m を引いた位置は exp(0) = 1 → 127 ちょうど（格子の上端に張り付く）
  const qp = referenceAttentionPvQuant([5, 4], [5], 1, 2);
  assertEquals([...qp], [127, 47], "exp(0) → 127 / exp(−1)·127 ≈ 46.7 → 47");
  // 同点の実例: S−m = f32(−0.195) では f32(f32(exp(x))·127) がちょうど 104.5 に乗る。
  // 偶数丸めなら 104、Math.round（half-up）なら 105 になるので両者が割れる。
  const tie = referenceAttentionPvQuant([Math.fround(-0.195)], [0], 1, 1);
  assertEquals(tie[0], 104, "同点は偶数側（Math.round なら 105）");
});

// ---------------------------------------------------------------------------
// ⑧ referenceAttentionPvI8a8Core — 既知値（整数を受け取ってからの純関数）
// ---------------------------------------------------------------------------

Deno.test("i8a8 参照: ③PV は f32(Σ qP·vq) · ((inv·(1/127))·vs) の既知値と一致する", () => {
  //   acc  = 127·100 + 64·(−50) = 9500
  //   prow = f32(0.5 · f32(1/127)) = 0.003937007859349251
  //   O    = f32(9500 · f32(prow · 2)) ≈ 74.803149
  const out = referenceAttentionPvI8a8Core({
    qp: [127, 64],
    vq: [100, -50],
    rowInv: [0.5],
    vs: [2],
    batch: 1,
    m: 1,
    n: 2,
    d: 1,
  });
  assertEquals([...out], [74.80314636230469], "③PV の出力");
});
