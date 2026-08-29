// 融合 attention の i8a8 内積変種を**実走で決める**カナリア（src/gpu/attention-dp4a-canary.ts）。
//
// 背景 = docs/known-issues.md「Metal（Apple GPU）で attention i8a8 と conv2d の 2 経路一致が
// 崩れる」。「dot4I8Packed 版とエミュ版は同じ整数を返す」という主張（src/kernels/
// linear-i8a8.ts）は linear では実機で保たれているのに **融合 attention だけ** Apple M2 で
// 崩れる。アダプタ能力から数値を変える経路を自動選択しない（ADR 0058 決定 2）を保つには、
// 列挙ではなく**既知解との突合**で変種を決めるほかない。
//
// ## このファイルが固定する契約（判定則 v2）
//
// 1. 健全な device では **dp4a / emu の両腕とも既知解と atol=0 で一致**し、カナリアは dp4a を
//    選ぶ（= 従来と 1 ビットも変わらない挙動）。
// 2. dp4a だけが既知解を外す device では attention だけ emu へ落ちる。
// 3. **どちらの腕も厳密一致しない**ときは sanity 帯で裁く — 片腕だけ帯内ならその腕、両腕とも
//    帯内で腕同士がビット同一なら dp4a（Apple M2 の正常経路。共有 f32 エピローグの丸めが
//    既知解と数 ULP ずれるだけで、変種選択は数値に無関係）。判定は「厳密一致ではなかった」
//    事実を戻り値に載せる。
// 4. **両腕とも帯を外したら fail loudly**（変種の選び方の問題ではないので縮退先が無い）。
// 5. 判定は **device 単位に 1 度**（Promise メモ化）。a8 でない Session は 1 dispatch も払わない。
// 6. 判定結果は診断（パイプラインキーの `:dp4aEmu`）に現れる（ADR 0058 決定 3）。
//
// ## 故障注入について
//
// 不一致経路・帯内経路・両不一致経路は健全な実機では作れない（src/gpu/device.ts の shader-f16
// カナリアと同じ事情）。そこで**生成 WGSL を意図的に壊す差し替え口**（`CanaryWgslPatch`）を
// 通して、判定の分岐そのものを実 GPU 上で踏む。MUST: 注入が効いていること自体を先に門にする
// （置換が空振りすると「壊したのに一致した」= 恒真テストになる）。

import { assert, assertEquals, assertRejects } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { f16BitsToF32, roundToF16 } from "../src/format/f16.ts";
import {
  acquireGpu,
  type GpuContext,
  GpuFeatureError,
  RUNTIME_INTERNAL,
} from "../src/gpu/device.ts";
import {
  type AttentionI8a8Decision,
  buildPvCase,
  buildQkCase,
  type CanaryWgslPatch,
  decideAttentionI8a8Dot,
  formatAttentionI8a8Decision,
  packScoresF16,
  probeAttentionI8a8Dot,
} from "../src/gpu/attention-dp4a-canary.ts";
import {
  attentionPvI8a8Key,
  attentionPvI8a8UsesVec4,
  attentionPvI8a8Wgsl,
  attentionQkI8a8Key,
  attentionQkI8a8UsesVec4,
  attentionQkI8a8Wgsl,
} from "../src/kernels/attention-i8a8.ts";
import {
  defaultI8a8Geometry,
  i8a8KPacks,
  i8a8TileM,
  i8a8TileN,
} from "../src/kernels/i8a8-geometry.ts";
import { createSession, type SessionOptions } from "../src/runtime/executor.ts";
import type { I8a8Dot } from "../src/runtime/session-types.ts";
import { fill, graphModelBuffer, singleOpGraph } from "./helpers/graph.ts";
import { GPU_AVAILABLE, TIMING_ACQUIRE_OPTIONS } from "./helpers/gpu.ts";

/** `127·exp(S−m)` が半整数から離れているべき最小の余裕（WGSL の `exp` 誤差 ~1e-5 の桁上）。 */
const QUANT_MARGIN = 0.3;

/** dp4a 変種**だけ**を壊す（エミュ側の生成物にこの綴りは現れない）。 */
const BREAK_DP4A: CanaryWgslPatch = (wgsl) =>
  wgsl.replace("return dot4I8Packed(a, b);", "return dot4I8Packed(a, b) + 1;");

/** 両変種を壊す（`idot` の本体はどちらも `  return dot` で始まる）。 */
const BREAK_BOTH: CanaryWgslPatch = (wgsl) => wgsl.replace("  return dot", "  return 1 + dot");

/**
 * **③PV のエピローグだけ**に相対 `factor − 1` の摂動を入れる（両腕に等しく効く — この綴りは
 * dp4a / emu のどちらの生成物にも同じ形で現れる）。
 *
 * なぜ ③PV だけか: ①QK の s16 変種は出力を f16 に丸めて格納するので、微小摂動が丸め境界を
 * 跨ぐと**差が f16 の 1 ULP（相対 ~1e-3）に化けて帯外へ飛ぶ**。固定入力の ①QK 既知解には
 * 境界まで相対 1.8e-7 しかない要素が実在する（実測）ので、微小摂動の実験にならない。③PV は
 * 3 変種とも O を f32 で書くため、入れた摂動がそのままの倍率で観測できる。
 */
const nudgePv = (factor: string): CanaryWgslPatch => (wgsl) =>
  wgsl.replaceAll("let prow = stats[", `let prow = ${factor} * stats[`);

/**
 * 帯**内**の摂動（両腕・同一）。`1.0000001` は f32 で 1 + 1 ULP に丸まるので、掛けると出力は
 * 必ず 1〜2 ULP 動く（= 厳密一致は必ず外れる）が、相対 ~1.2e-7 は帯（1e-5）の 2 桁下。
 */
const NUDGE_BOTH = nudgePv("1.0000001");

/** 帯**外**の摂動（両腕・同一）。相対 1e-4 は帯の 1 桁上 — M2 の故障注入実測 9.5e-5 と同水準。 */
const SKEW_BOTH = nudgePv("1.0001");

/** 片腕（emu）だけが帯内に残る形 = M2 で dp4a が本当に壊れている device の姿。 */
const NUDGE_BOTH_BREAK_DP4A: CanaryWgslPatch = (wgsl) => BREAK_DP4A(NUDGE_BOTH(wgsl));

// ---------------------------------------------------------------------------
// (0) 固定入力の性質（既知解が atol=0 で立つ前提 — GPU 不要）
// ---------------------------------------------------------------------------

Deno.test("カナリアの形は production 幾何の 1 タイル全域 × K タイル 2 枚以上を埋める", () => {
  const qkGeometry = defaultI8a8Geometry("attention_qk");
  const qk = buildQkCase();
  assertEquals(qk.rows, i8a8TileM(qkGeometry), "①QK の M がタイル辺と違う");
  assertEquals(qk.cols, i8a8TileN(qkGeometry), "①QK の N がタイル辺と違う");
  assert(
    qk.depth / 4 / i8a8KPacks(qkGeometry) >= 2,
    `①QK の K タイルが ${qk.depth / 4 / i8a8KPacks(qkGeometry)} 枚しかない`,
  );
  const pvGeometry = defaultI8a8Geometry("attention_pv");
  const pv = buildPvCase();
  assertEquals(pv.rows, i8a8TileM(pvGeometry), "③PV の M がタイル辺と違う");
  assertEquals(pv.depth, i8a8TileN(pvGeometry), "③PV の D がタイル辺と違う");
  assert(
    pv.cols / 4 / i8a8KPacks(pvGeometry) >= 2,
    `③PV の K タイルが ${pv.cols / 4 / i8a8KPacks(pvGeometry)} 枚しかない`,
  );
});

Deno.test("③PV の固定 S は f16 ちょうどで表せ、往復で 1 ビットも動かない（s16 変種の前提）", () => {
  const pv = buildPvCase();
  const words = packScoresF16(pv.scores);
  assertEquals(words.length, pv.scores.length / 2, "詰め直しの語数");
  for (let i = 0; i < pv.scores.length; i += 1) {
    const value = pv.scores[i];
    // ① Math.f16round（丸めの正本）から見て f16 ちょうど
    assertEquals(roundToF16(value), value, `S[${i}] が f16 ちょうどでない`);
    // ② カナリアの詰め直し器（exactF16Bits）と展開の正本が往復する
    const word = words[i >> 1];
    const half = (i & 1) === 0 ? word & 0xffff : word >>> 16;
    assertEquals(f16BitsToF32(half), value, `S[${i}] の f16 往復`);
  }
});

Deno.test("③PV の qP は丸め境界から十分離れている（GPU の exp 誤差では段が動かない）", () => {
  const pv = buildPvCase();
  let worst = Number.POSITIVE_INFINITY;
  const seen = new Set<number>();
  for (let row = 0; row < pv.rows; row += 1) {
    const max = pv.stats[row * 2];
    for (let i = 0; i < pv.cols; i += 1) {
      const shifted = Math.fround(pv.scores[row * pv.cols + i] - max);
      const p = Math.fround(Math.fround(Math.exp(shifted)) * 127);
      seen.add(Math.round(p));
      worst = Math.min(worst, Math.abs((p - Math.floor(p)) - 0.5));
    }
  }
  assert(worst > QUANT_MARGIN, `qP の最悪余裕が ${worst} しかない（固定 S を選び直すこと）`);
  // 段が 1 つに潰れていない（潰れると添字の取り違えが値に出ない = 突合が恒真化する）
  assert(seen.size >= 8, `qP の段が ${seen.size} 種類しかない`);
});

Deno.test("①QK の既知解は f16 格納の可視域に収まり、定数に潰れていない", () => {
  const qk = buildQkCase();
  let distinct = 0;
  const values = new Set<number>();
  for (const value of qk.expected) {
    assert(Number.isFinite(value), "既知解 S に非有限値がある");
    assert(Math.abs(value) < 65504, `既知解 S=${value} が f16 の可視域を超える`);
    if (!values.has(value)) {
      values.add(value);
      distinct += 1;
    }
  }
  assert(distinct > 100, `既知解 S が ${distinct} 種類しかない（比較が恒真になりうる）`);
});

Deno.test("故障注入の置換は狙った変種にだけ効く（注入の空振りを先に塞ぐ）", () => {
  const geometry = defaultI8a8Geometry("attention_qk");
  const dp4aWgsl = attentionQkI8a8Wgsl(true, true, "f32", geometry);
  const emuWgsl = attentionQkI8a8Wgsl(true, false, "f32", geometry);
  assert(BREAK_DP4A(dp4aWgsl) !== dp4aWgsl, "dp4a 生成物に注入が効いていない");
  assertEquals(BREAK_DP4A(emuWgsl), emuWgsl, "dp4a 専用の注入がエミュ生成物にも効いている");
  assert(BREAK_BOTH(dp4aWgsl) !== dp4aWgsl, "両変種注入が dp4a に効いていない");
  assert(BREAK_BOTH(emuWgsl) !== emuWgsl, "両変種注入がエミュに効いていない");
});

Deno.test("エピローグ摂動は ③PV の 3 変種すべてに、両腕へ等しく効く（①QK は素通し）", () => {
  const pvGeometry = defaultI8a8Geometry("attention_pv");
  for (const patch of [NUDGE_BOTH, SKEW_BOTH]) {
    for (const score of ["f32", "f16"] as const) {
      for (const v4 of score === "f16" ? [true] : [true, false]) {
        for (const dp4a of [true, false]) {
          const wgsl = attentionPvI8a8Wgsl(v4, dp4a, score, pvGeometry);
          assert(patch(wgsl) !== wgsl, `③PV v4=${v4} score=${score} dp4a=${dp4a} に効いていない`);
        }
      }
    }
    // ①QK は f16 格納が丸め境界を跨ぐ危険があるので、摂動の対象外であることを門にする
    const qkWgsl = attentionQkI8a8Wgsl(true, true, "f16", defaultI8a8Geometry("attention_qk"));
    assertEquals(patch(qkWgsl), qkWgsl, "エピローグ摂動が ①QK にも効いている");
  }
});

Deno.test("判定の根拠 1 行は分岐と最大誤差を持つ（警告文言がそのまま使う本文）", () => {
  const decision: AttentionI8a8Decision = {
    dot: "dp4a",
    exact: false,
    branch: "band-both-identical",
    maxAbsError: 1.9073486328125e-6,
    maxBandRatio: 0.0067,
  };
  const line = formatAttentionI8a8Decision(decision);
  assert(line.includes("ビット同一"), line);
  assert(line.includes("'dp4a'"), line);
  assert(line.includes(String(decision.maxAbsError)), line);
  assert(line.includes(String(decision.maxBandRatio)), line);
});

// ---------------------------------------------------------------------------
// (1) 健全な device（この Linux / NVIDIA）
// ---------------------------------------------------------------------------

Deno.test({
  name: "素の判定は dp4a を選び、分岐と厳密一致フラグが整合する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      const decision = await decideAttentionI8a8Dot(gpu);
      // 期待はプラットフォームで分かれる: 参照機（Linux / NVIDIA）は dp4a-exact、Apple M2 は
      // 共有エピローグの 1 ULP 差により band-both-identical（known-issues の Metal 節）。
      // どちらでも**選ばれる腕は dp4a** — それがこのテストの主題。参照機のビット厳密性
      // そのものは attention parity 門（atol=0）が別途固定している。
      assertEquals(decision.dot, "dp4a", "素の判定で dp4a が選ばれない");
      assert(
        decision.branch === "dp4a-exact" || decision.branch === "band-both-identical",
        `素の判定が想定外の分岐: ${decision.branch}`,
      );
      assertEquals(
        decision.exact,
        decision.branch === "dp4a-exact",
        "厳密一致フラグと分岐が食い違っている",
      );
      if (decision.exact) {
        assertEquals(decision.maxAbsError, 0, "厳密一致なのに誤差が乗っている");
        // 参照機ではエミュ側も既知解に乗る（= カナリアの沈黙が「両方同じだけずれている」形で
        // はない）。band 機では非対象 — 両腕の同一性は band-both-identical 分岐自体が検証済み。
        // MUST: 直接呼ぶときも device 単位の errorScope 区間ロックの中で（probe の doc）。
        const emu = await gpu[RUNTIME_INTERNAL].withScopeLock(() =>
          probeAttentionI8a8Dot(gpu, "emu")
        );
        assertEquals(emu.mismatch, undefined, "エミュ変種が既知解を外した");
        assert(emu.exact, "エミュ変種が atol=0 で一致していない");
        assertEquals(emu.variants.length, 6, "撃った変種が 6 本でない");
      }
    } finally {
      gpu.destroy();
    }
  },
});

// ---------------------------------------------------------------------------
// (2) 故障注入（不一致経路の分岐）
// ---------------------------------------------------------------------------

Deno.test({
  name: "dp4a 変種だけが既知解を外すと attention は emu へ落ちる（実 GPU・故障注入）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      const decision = await decideAttentionI8a8Dot(gpu, BREAK_DP4A);
      assertEquals(decision.dot, "emu");
      // 参照機は emu-exact、Apple M2 はエミュ側も 1 ULP 差なので band-single-arm 経由で emu
      // （どちらも「壊れた dp4a を捨てて emu を採る」という主題は同じ）。
      assert(
        decision.branch === "emu-exact" || decision.branch === "band-single-arm",
        `想定外の分岐: ${decision.branch}`,
      );
      assertEquals(
        decision.exact,
        decision.branch === "emu-exact",
        "厳密一致フラグと分岐が食い違っている",
      );
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "両変種が既知解を外すと GpuFeatureError で落ちる（縮退先が無い・実 GPU・故障注入）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      const error = await assertRejects(
        () => decideAttentionI8a8Dot(gpu, BREAK_BOTH),
        GpuFeatureError,
      );
      // 診断は「どちらの腕が、どの生成物の、どの要素で外したか」まで持つ
      assert(error.message.includes("attention_qk:v3:i8a8:"), error.message);
      assert(error.message.includes("dp4a:"), error.message);
      assert(error.message.includes("emu:"), error.message);
      // 「帯は既に許容した上で外している」ことが読める（v2 の裁定順が診断に出る）
      assert(error.message.includes("sanity 帯"), error.message);
    } finally {
      gpu.destroy();
    }
  },
});

// ---------------------------------------------------------------------------
// (2') 故障注入（判定則 v2 の帯の分岐 — Apple M2 の実測が要求した経路）
// ---------------------------------------------------------------------------

Deno.test({
  name: "両腕が同一の微小エピローグ摂動で帯内に留まると dp4a のまま通る（実 GPU・故障注入）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      const decision = await decideAttentionI8a8Dot(gpu, NUDGE_BOTH);
      // 腕同士がビット同一 = 変種選択は数値に無関係（ADR 0058 決定 2）→ 既定の dp4a を採る
      assertEquals(decision.branch, "band-both-identical");
      assertEquals(decision.dot, "dp4a");
      // MUST: 厳密一致ではなかった事実が戻り値に残る（呼び手が警告を出せる形）
      assert(!decision.exact, "帯内で通したのに厳密一致フラグが立っている");
      assert(decision.maxAbsError > 0, "摂動が空振りしている（誤差 0）");
      assert(decision.maxBandRatio <= 1, `帯余裕 ${decision.maxBandRatio} が帯を超えている`);
      // 摂動は相対 ~1.2e-7 = 帯の 2 桁下（帯幅を緩めなくても収まることの実測）
      assert(decision.maxBandRatio < 0.1, `帯余裕 ${decision.maxBandRatio} が想定より大きい`);
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "両腕が同一でも帯外の摂動なら GpuFeatureError（ビット同一は免罪符でない・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      const error = await assertRejects(
        () => decideAttentionI8a8Dot(gpu, SKEW_BOTH),
        GpuFeatureError,
      );
      assert(error.message.includes("attention_pv:v3:i8a8:"), error.message);
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "片腕だけが帯内ならその腕を採る（M2 + dp4a 故障の姿・実 GPU・故障注入）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      const decision = await decideAttentionI8a8Dot(gpu, NUDGE_BOTH_BREAK_DP4A);
      assertEquals(decision.branch, "band-single-arm");
      assertEquals(decision.dot, "emu");
      assert(!decision.exact, "帯内で通したのに厳密一致フラグが立っている");
      assert(decision.maxBandRatio <= 1, `選んだ腕の帯余裕 ${decision.maxBandRatio} が帯外`);
    } finally {
      gpu.destroy();
    }
  },
});

// ---------------------------------------------------------------------------
// (3) Session からの利用（メモ化 1 回 / 席の反映 / a8 でない Session は払わない）
// ---------------------------------------------------------------------------

/** メモの席へ直に置く判定（カナリアを走らせずに席そのものの挙動を見るための実体）。 */
const seatDecision = (dot: I8a8Dot): AttentionI8a8Decision => ({
  dot,
  exact: true,
  branch: dot === "dp4a" ? "dp4a-exact" : "emu-exact",
  maxAbsError: 0,
  maxBandRatio: 0,
});

const QUERY = (i: number): number => (((i * 3) % 29) - 14) * 0.3717 + 0.0419;
const KEY = (i: number): number => (((i * 3) % 41) - 20) * 0.2917 - 0.0173;
const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.3119;

/** 融合 attention 1 ノードの Session を 1 回走らせ、走ったパイプラインキーを返す。 */
const runAttention = async (
  gpu: GpuContext,
  options: SessionOptions,
): Promise<readonly string[]> => {
  const [b, h, m, n, d] = [1, 2, 20, 16, 8];
  const q = fill([b, h, m, d], QUERY);
  const k = fill([b, h, n, d], KEY);
  const v = fill([b, h, n, d], VALUE);
  const graph = singleOpGraph("attention", [q.shape, k.shape, v.shape], [[b, h, m, d]], {
    attrs: { scale: Math.fround(Math.sqrt(1 / Math.sqrt(d))) },
  });
  const session = await createSession(gpu, openModel(graphModelBuffer(graph)), options);
  try {
    await session.run({ x0: q, x1: k, x2: v });
    return (session.diagnostics().lastRunTiming?.entries ?? []).map((entry) => entry.key);
  } finally {
    await session.dispose();
  }
};

Deno.test({
  name: "カナリアの判定は device 単位に 1 度で、a8 Session はその席を読む（実 GPU・故障注入）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      // 実機は健全なので、注入した判定をメモへ**先に**焼く（焼かなければ dp4a になる）。
      // Session がこの席を読んでいることは、キーが `:dp4aEmu` になることでしか観測できない。
      const seeded = await gpu[RUNTIME_INTERNAL].attentionI8a8Dot(() =>
        decideAttentionI8a8Dot(gpu, BREAK_DP4A)
      );
      assertEquals(seeded.dot, "emu");

      for (const pass of [1, 2]) {
        const keys = new Set(await runAttention(gpu, { attentionCompute: "a8" }));
        if (keys.size === 0) continue; // 計測が無い環境ではキー検査ごと空振りする
        const qkV4 = attentionQkI8a8UsesVec4(16);
        const pvV4 = attentionPvI8a8UsesVec4(8);
        assert(keys.has(attentionQkI8a8Key(qkV4, false)), `${pass} 本目: ①QK が emu 変種でない`);
        assert(!keys.has(attentionQkI8a8Key(qkV4, true)), `${pass} 本目: dp4a 変種が残っている`);
        assert(keys.has(attentionPvI8a8Key(pvV4, false)), `${pass} 本目: ③PV が emu 変種でない`);
        assert(!keys.has(attentionPvI8a8Key(pvV4, true)), `${pass} 本目: dp4a 変種が残っている`);
      }

      // 2 本目まで走らせてもカナリアは 1 度きり（メモが Promise で効いている）
      let extraRuns = 0;
      const memoized = await gpu[RUNTIME_INTERNAL].attentionI8a8Dot(() => {
        extraRuns += 1;
        return Promise.resolve(seatDecision("dp4a"));
      });
      assertEquals(extraRuns, 0, "Session ごとにカナリアが走り直している");
      assertEquals(memoized.dot, "emu", "メモが最初の判定を配っていない");
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "attentionCompute が 'a8' でない Session はカナリアを 1 dispatch も払わない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      await runAttention(gpu, {});
      await runAttention(gpu, { attentionCompute: "f16" }).catch(() => undefined);
      // 席が空のままなら、ここで渡した実体が初めて走る（= 上の 2 本は払っていない）
      let ran = 0;
      const verdict = await gpu[RUNTIME_INTERNAL].attentionI8a8Dot(() => {
        ran += 1;
        return Promise.resolve(seatDecision("emu"));
      });
      assertEquals(ran, 1, "a8 でない Session がカナリアを走らせている");
      assertEquals(verdict.dot, "emu");
    } finally {
      gpu.destroy();
    }
  },
});
