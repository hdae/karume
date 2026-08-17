// GQA（整除 broadcast — ADR 0067 決定 1〜3）の**ビット同一の門**。
//
// 同じ乱数入力に対し、
//
//   ① `k` / `v` を `[B,Hkv,N,D]` のまま渡す GQA 形の `attention`
//   ② ホスト側で kv-head を H 本へ**実体化**（repeat_kv）した従来形の `attention`
//
// を**同じ実 GPU**で流し、出力を **f32 のビット列**で突き合わせる。GQA 変種の差分は K / V の
// バッチ base 1 行と uniform 1 語だけ（ADR 0067 決定 2）なので、共有タイル充填も K 縮約順も
// 動かない = ビット単位で一致するのが**期待値そのもの**。allclose では head 対応の取り違えが
// tolerance に隠れる。
//
// MUST: 恒真化しないこと。①② は**別々のグラフ**（入力 shape が違う）を別々の Session で
// 走らせる。②の実体化はホスト側の素朴なコピーで、カーネルの写像式（`wid.z / r`）とは
// 独立に書く。
// MUST: 一致だけでなく**故障注入で不一致が出ること**を同じテストで見る（head 対応を取り違えた
// 実体化と一致してしまうなら、この門は何も見ていない）。
// MUST: B / H / Hkv / M / N / D は複数形状（端数・v4・行タイル 2 枚跨ぎ）を回す。カーネルは
// B·H を 1 本のバッチ軸に畳むので、B=1 だけでは `b·Hkv + h/r` の恒等式が検証されない。

import { assert, assertEquals, assertRejects } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { attentionPvKey, attentionQkKey } from "../src/kernels/attention.ts";
import { gemmUsesVec4 } from "../src/kernels/gemm.ts";
import { attentionScoreUsesF16 } from "../src/kernels/score-storage.ts";
import { createSession, type SessionOptions, type Tensor } from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { graphModelBuffer, singleOpGraph } from "./helpers/graph.ts";
import { GPU_AVAILABLE, TIMESTAMP_QUERY_AVAILABLE, TIMING_ACQUIRE_OPTIONS } from "./helpers/gpu.ts";

/** 半スケール（torch math decomp の `√scale_factor`）。D から導く契約どおりの値。 */
const halfScale = (depth: number): number => Math.fround(Math.sqrt(1 / Math.sqrt(depth)));

/** 決定的なデータ列（乱数は使わない — 失敗が再現しないため）。 */
const QUERY = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
const KEY = (i: number): number => (((i * 11) % 19) - 9) * 0.23;
const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.31;

type F32Tensor = {
  readonly dtype: "f32";
  readonly shape: readonly number[];
  readonly data: Float32Array<ArrayBuffer>;
};

const f32 = (shape: readonly number[], generator: (index: number) => number): F32Tensor => {
  const count = shape.reduce((total, dim) => total * dim, 1);
  const data = new Float32Array(count);
  for (let index = 0; index < count; index += 1) data[index] = generator(index);
  return { dtype: "f32", shape, data };
};

/** f32 のビット列（末尾 1 ulp の差も `-0.0` も見える形）。 */
const bits = (tensor: Tensor): Uint32Array =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length);

type GqaShape = {
  readonly name: string;
  readonly b: number;
  /** q の head 数 H。 */
  readonly h: number;
  /** k / v の head 数 Hkv（`h % kv == 0` — r = h / kv）。 */
  readonly kv: number;
  readonly m: number;
  readonly n: number;
  readonly d: number;
};

const SHAPES: readonly GqaShape[] = [
  // r=1（対照 — GQA ビットが立たない形。キーも生成物も従来のまま）
  { name: "r1 B2 H4 M5 N7 D4", b: 2, h: 4, kv: 4, m: 5, n: 7, d: 4 },
  // r=2 かつ全軸タイル端数（D%4 ≠ 0 でスカラ変種へ落ちる）
  { name: "r2 B2 H4 M5 N7 D6", b: 2, h: 4, kv: 2, m: 5, n: 7, d: 6 },
  // r=8 の MQA（Gemma 4 E2B の 8:1 — Hkv=1 も同じ整除式）。①③ とも v4 経路
  { name: "r8 B2 H8 M3 N8 D8", b: 2, h: 8, kv: 1, m: 3, n: 8, d: 8 },
  // r=4 × B=3 で行タイル 2 枚を跨ぐ（`b·Hkv + h/r` の b 項が効く形）
  { name: "r4 B3 H8 M68 N20 D12", b: 3, h: 8, kv: 2, m: 68, n: 20, d: 12 },
  // 素の GQA 16:2（MiniCPM5-1B）を縮めた形。M / N とも 64 の倍数 = 実モデルの経路
  { name: "r8 B1 H16 M64 N64 D64", b: 1, h: 16, kv: 2, m: 64, n: 64, d: 64 },
];

/** 故障注入で使う「r=4 を跨ぐ形」（head ごとにデータが違うので写像の誤りが値に出る）。 */
const INJECT_SHAPE: GqaShape = SHAPES[3];

/**
 * i8a8 の**段別適格判定**（①QK は `D % 4`・③PV は `N % 4` — src/runtime/recipe-builder.ts）を
 * 跨いだ GQA 形。GQA × i8a8 の拒否は**要求されたモード**で決まる（ADR 0067 決定 3）ので、
 * 段が片方だけ非適格でも両方非適格でも**同じ 1 本のエラー**で落ちる。
 *
 * MUST: 適格形（SHAPES[2] は D=8 / N=8 で ①③ とも適格）だけを被験体にしない — 拒否条件を段別
 * 適格判定へ誤用する退行（`gqa && qkI8a8 && pvI8a8` 等）は、**非適格形だけが黙って f32 / 混成へ
 * 縮退する**形で出るので、適格形しか踏まない門は緑のまま通す。
 */
const I8A8_INELIGIBLE: readonly GqaShape[] = [
  { name: "r2 D%4≠0（①QK 非適格・③PV 適格）", b: 2, h: 4, kv: 2, m: 5, n: 8, d: 6 },
  { name: "r2 N%4≠0（①QK 適格・③PV 非適格）", b: 2, h: 4, kv: 2, m: 5, n: 7, d: 4 },
  { name: "r2 D%4≠0 かつ N%4≠0（①③ とも非適格）", b: 2, h: 4, kv: 2, m: 5, n: 7, d: 6 },
];

/** 拒否理由の**不変部分**（形ごとに変わるのは `${where}` と `H / Hkv / r` の表示だけ）。 */
const rejectionReason = (message: string, where: string): string => {
  const at = message.indexOf("× i8a8");
  assert(at >= 0, `${where}: GQA × i8a8 の拒否の文言が変わっている: ${message}`);
  return message.slice(at);
};

/**
 * kv-head を H 本へ**実体化**する（repeat_kv）。
 *
 * `pick` が head → kv-head の写像で、既定（{@link BLOCK_REPEAT}）は `⌊h/r⌋` の block repeat。
 * 故障注入はここに別の写像を差す（カーネルの写像式とは独立にホスト側で組むのが要点）。
 */
const materializeKv = (
  source: F32Tensor,
  heads: number,
  pick: (head: number, kvHeads: number, repeat: number) => number,
): F32Tensor => {
  const [batches, kvHeads, cols, depth] = source.shape;
  const repeat = heads / kvHeads;
  const block = cols * depth;
  const data = new Float32Array(batches * heads * block);
  for (let batch = 0; batch < batches; batch += 1) {
    for (let head = 0; head < heads; head += 1) {
      const kvHead = pick(head, kvHeads, repeat);
      const from = (batch * kvHeads + kvHead) * block;
      data.set(source.data.subarray(from, from + block), (batch * heads + head) * block);
    }
  }
  return { dtype: "f32", shape: [batches, heads, cols, depth], data };
};

/** 契約どおりの写像（`h/r` の整数除算 — ADR 0067 決定 2 の恒等式のホスト側の写し）。 */
const BLOCK_REPEAT = (head: number, _kvHeads: number, repeat: number): number =>
  Math.floor(head / repeat);

/**
 * 実測形の band mask（`[1,1,M,N]` の加算項 — GQA とは独立の軸）。
 *
 * MUST: **全ての行に非マスク列が 1 本以上**あること（行が丸ごとマスクだと amax が −inf で
 * `exp(−inf − (−inf))` が NaN になり、両側が同じ NaN を出してビット比較が何も見なくなる）。
 */
const bandMask = (m: number, n: number, width: number, blocked: number): F32Tensor => {
  const data = new Float32Array(m * n);
  for (let row = 0; row < m; row += 1) {
    const center = Math.floor((row * n) / m);
    for (let col = 0; col < n; col += 1) {
      data[row * n + col] = Math.abs(col - center) <= width ? 0 : blocked;
    }
  }
  return { dtype: "f32", shape: [1, 1, m, n], data };
};

type RunResult = {
  readonly output: Tensor;
  /** 直近 run のパイプラインキー（timestamp-query が無い機では空）。 */
  readonly keys: readonly string[];
};

/**
 * 1 ノードの `attention` を走らせる。`kv` の shape が `q` と違えば GQA 形、同じなら従来形
 * （**同じ 1 本の経路**を通す — 形だけが違う）。
 */
const runAttention = async (
  gpu: GpuContext,
  q: F32Tensor,
  k: F32Tensor,
  v: F32Tensor,
  mask?: F32Tensor,
  options: SessionOptions = {},
): Promise<RunResult> => {
  const out = [q.shape[0], q.shape[1], q.shape[2], q.shape[3]];
  const shapes = mask === undefined
    ? [q.shape, k.shape, v.shape]
    : [q.shape, k.shape, v.shape, mask.shape];
  const graph = singleOpGraph("attention", shapes, out, {
    attrs: { scale: halfScale(q.shape[3]) },
  });
  const inputs: Record<string, F32Tensor> = { x0: q, x1: k, x2: v };
  if (mask !== undefined) inputs["x3"] = mask;
  const session = await createSession(gpu, openModel(graphModelBuffer(graph)), options);
  try {
    const output = (await session.run(inputs))["y"];
    const entries = session.diagnostics().lastRunTiming?.entries ?? [];
    return { output, keys: entries.map((entry) => entry.key) };
  } finally {
    await session.dispose();
  }
};

/** GQA 形と実体化形の入力 3 本（`pick` を差し替えると故障注入になる）。 */
const inputsFor = (
  shape: GqaShape,
  pick: (head: number, kvHeads: number, repeat: number) => number = BLOCK_REPEAT,
): {
  readonly q: F32Tensor;
  readonly k: F32Tensor;
  readonly v: F32Tensor;
  readonly kFull: F32Tensor;
  readonly vFull: F32Tensor;
} => {
  const { b, h, kv, m, n, d } = shape;
  const q = f32([b, h, m, d], QUERY);
  const k = f32([b, kv, n, d], KEY);
  const v = f32([b, kv, n, d], VALUE);
  return {
    q,
    k,
    v,
    kFull: materializeKv(k, h, pick),
    vFull: materializeKv(v, h, pick),
  };
};

/** ビット列の食い違い（最初の 4 件だけ拾う）。 */
const mismatches = (left: Tensor, right: Tensor): readonly number[] => {
  const a = bits(left);
  const c = bits(right);
  const found: number[] = [];
  for (let i = 0; i < a.length && found.length < 4; i += 1) {
    if (a[i] !== c[i]) found.push(i);
  }
  return found;
};

const assertBitEqual = (fused: Tensor, split: Tensor, where: string): void => {
  assertEquals(fused.shape, split.shape, where);
  assertEquals(
    mismatches(fused, split),
    [],
    `${where}: 実体化版とビット列が違う（最初の食い違い: ${
      mismatches(fused, split).map((i) => `[${i}] ${fused.data[i]} vs ${split.data[i]}`).join(" / ")
    }）`,
  );
  // 恒真化の門: 出力が定数なら「一致」は何も検証していない
  assert(new Set(bits(fused)).size > 1, `${where}: 出力が定数（ビット一致が恒真になっている）`);
};

Deno.test({
  name:
    "GQA 形 attention の出力が repeat_kv 実体化版と**ビット単位で一致**する（r ∈ {1,2,4,8}・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const shape of SHAPES) {
        const { q, k, v, kFull, vFull } = inputsFor(shape);
        const gqa = await runAttention(gpu, q, k, v);
        const split = await runAttention(gpu, q, kFull, vFull);
        assertBitEqual(gqa.output, split.output, shape.name);
        // 出力は q 側の H（実体化版と同形になっていることを shape の側でも固定する）
        assertEquals(gqa.output.shape, [shape.b, shape.h, shape.m, shape.d], shape.name);
      }
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * mask / s16 との**直交**（GQA は K / V の base だけの軸）。mask は `[1,1,M,N]` のまま Hkv に
 * 依らず、S の f16 格納も書き手と読み手が同時に切り替わるだけで写像には触れない。
 */
Deno.test({
  name: "GQA は加算 mask と S の f16 格納（s16）と直交してビット同一を保つ（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      // mask 付き（r=4・行タイル 2 枚跨ぎ）
      const masked = SHAPES[3];
      {
        const { q, k, v, kFull, vFull } = inputsFor(masked);
        const mask = bandMask(masked.m, masked.n, 2, -3.4028234663852886e38);
        const gqa = await runAttention(gpu, q, k, v, mask);
        const split = await runAttention(gpu, q, kFull, vFull, mask);
        assertBitEqual(gqa.output, split.output, `${masked.name} + mask`);
        // 恒真化の門: mask が結果に効いている（mask 無しと違う値になる）
        const plain = await runAttention(gpu, q, k, v);
        assert(
          bits(gqa.output).some((word, i) => word !== bits(plain.output)[i]),
          `${masked.name}: mask 付きと mask 無しの出力が同一（mask が効いていない）`,
        );
        assert(
          gqa.output.data.every((value) => Number.isFinite(value)),
          `${masked.name}: 出力に非有限値（行が丸ごとマスクされている）`,
        );
      }
      // s16（S の f16 格納 — 適格形であることを判定関数で確かめてから回す）
      const scored = SHAPES[2];
      {
        assertEquals(
          attentionScoreUsesF16(scored.d, scored.n),
          true,
          `${scored.name}: s16 の適格形のはず`,
        );
        const { q, k, v, kFull, vFull } = inputsFor(scored);
        const options: SessionOptions = { attentionScoreStorage: "f16" };
        const gqa = await runAttention(gpu, q, k, v, undefined, options);
        const split = await runAttention(gpu, q, kFull, vFull, undefined, options);
        assertBitEqual(gqa.output, split.output, `${scored.name} + s16`);
        // 恒真化の門: s16 が効いている（f32 格納と値が動く = 丸めが 1 段増えている）
        const f32Score = await runAttention(gpu, q, k, v);
        assert(
          bits(gqa.output).some((word, i) => word !== bits(f32Score.output)[i]),
          `${scored.name}: s16 と f32 格納の出力が同一（opt-in が効いていない）`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * **故障注入**（ADR 0067 決定 2 の受入条件）。head 対応を取り違えた実体化と突き合わせて
 * **不一致が出ること**を確かめる — 出なければ上のビット同一の門は何も見ていない。
 *
 * ① `r` 誤り: `⌊h/2⌋ mod Hkv`（r'=2 の写像）で実体化する。真の r=4 とは head 0..7 の
 *    対応が 0,0,1,1,0,0,1,1 と 0,0,0,0,1,1,1,1 で違う。
 * ② ③PV 側の写像漏れ: K だけ正しく実体化し、V は **kv-head 0 に張り付いた**形にする
 *    （①QK を写して ③PV を写し忘れたカーネルが出す値と同型 — S は正しいのに V だけ別 head）。
 */
Deno.test({
  name: "GQA のビット同一の門は head 写像の誤り（r 誤り・③PV の写像漏れ）を検出する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const shape = INJECT_SHAPE;
    const gpu = await acquireGpu();
    try {
      const { q, k, v, kFull, vFull } = inputsFor(shape);
      const truth = await runAttention(gpu, q, k, v);
      // 正しい実体化とは一致する（不一致側の対照 — 環境ではなく写像を見ていることの証拠）
      assertBitEqual(truth.output, (await runAttention(gpu, q, kFull, vFull)).output, shape.name);

      // ① r 誤り（r'=2 の写像で K / V とも実体化）
      const wrongR = inputsFor(shape, (head, kvHeads) => Math.floor(head / 2) % kvHeads);
      const injectedR = await runAttention(gpu, q, wrongR.kFull, wrongR.vFull);
      assert(
        mismatches(truth.output, injectedR.output).length > 0,
        `${shape.name}: r 誤り（r'=2）の実体化と一致してしまう（門が写像を見ていない）`,
      );

      // ② ③PV 側の写像漏れ（K は正しく・V は kv-head 0 に張り付き）
      const stuckV = materializeKv(v, shape.h, () => 0);
      const injectedPv = await runAttention(gpu, q, kFull, stuckV);
      assert(
        mismatches(truth.output, injectedPv.output).length > 0,
        `${shape.name}: V の写像漏れと一致してしまう（③PV 側を見ていない）`,
      );
      // 逆側（K だけ張り付き）も検出する — ①QK / ③PV のどちら片方の漏れも見える形
      const stuckK = materializeKv(k, shape.h, () => 0);
      const injectedQk = await runAttention(gpu, q, stuckK, vFull);
      assert(
        mismatches(truth.output, injectedQk.output).length > 0,
        `${shape.name}: K の写像漏れと一致してしまう（①QK 側を見ていない）`,
      );
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * **census**（ADR 0058 決定 4）。r > 1 では GQA 変種のキーが**実際に走った**こと、r = 1 では
 * `:gqa` が 1 本も出ないこと（決定 2 の「r=1 はバイト同一」の実行側の裏）。
 *
 * MUST: `TIMING_ACQUIRE_OPTIONS` を渡す（素の `acquireGpu()` では `lastRunTiming` が
 * undefined になり、キー検査が黙って空振りする）。
 * MUST: 計測を要求しない device（`TIMESTAMP_QUERY_AVAILABLE` が偽）では**明示 SKIP** し、走る
 * ときは空の内訳を無条件に FAIL にする（`TIMING_ACQUIRE_OPTIONS` は feature 不在で
 * `gpuTiming: false` に落ちるので、「entries が空なら次の形へ」で守ると全ケースが無検査のまま
 * 緑になる — e2e_minicpm5_test.ts の census と同じ形）。
 */
Deno.test({
  name: "GQA のキーは r > 1 でだけ立つ（r=1 は従来キーのまま・実 GPU / timestamp-query）",
  ignore: !GPU_AVAILABLE || !TIMESTAMP_QUERY_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      for (const shape of SHAPES) {
        const { q, k, v } = inputsFor(shape);
        const { keys } = await runAttention(gpu, q, k, v);
        assert(keys.length > 0, `${shape.name}: 内訳が空（キー検査が空振りしている）`);
        const gqa = shape.h !== shape.kv;
        const attention = keys.filter((key) =>
          key.startsWith("attention_qk") || key.startsWith("attention_pv")
        );
        assertEquals(attention.length, 2, `${shape.name}: ①③ が 1 本ずつ走る`);
        assertEquals(
          keys.includes(attentionQkKey(gemmUsesVec4(shape.d, shape.n), "f32", "f32", false, gqa)),
          true,
          `${shape.name}: ①QK の期待キーが無い（走ったキー: ${attention.join(" / ")}）`,
        );
        assertEquals(
          keys.includes(attentionPvKey(gemmUsesVec4(shape.n, shape.d), "f32", "f32", gqa)),
          true,
          `${shape.name}: ③PV の期待キーが無い（走ったキー: ${attention.join(" / ")}）`,
        );
        // r=1 は `:gqa` を 1 本も持たない / r>1 は ①③ とも持つ
        assertEquals(
          attention.filter((key) => key.endsWith(":gqa")).length,
          gqa ? 2 : 0,
          `${shape.name}: :gqa の本数`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * **GQA × i8a8 は fail loudly**（ADR 0067 決定 3 — 黙って f32 経路へ落とすと性能が静かに変わる。
 * i8a8 は head 基底が 5 本で K / V の量子化・確保も `B·H` 前提なので、GQA 形は取り違えになる）。
 *
 * 恒真化の門: **同じ形の非 GQA（H = Hkv）は i8a8 で走る**こと（拒否の原因が GQA であって
 * 「その形が i8a8 で走らない」ではないことを固定する）。
 *
 * MUST: 段別適格判定を跨いだ非適格形（{@link I8A8_INELIGIBLE}）も**同じ**エラーで落ちること
 * まで見る — 拒否が要求モードではなく段の適格性で決まる実装では、そこだけが沈黙で縮退する。
 */
Deno.test({
  name:
    "GQA × attentionCompute 'i8a8' は fail loudly（段別適格性に依らず・同形の非 GQA は走る・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const shape = SHAPES[2];
    const gpu = await acquireGpu();
    try {
      const { q, k, v, kFull, vFull } = inputsFor(shape);
      const options: SessionOptions = { attentionCompute: "i8a8" };
      const error = await assertRejects(
        () => runAttention(gpu, q, k, v, undefined, options),
        ExecutionError,
        "GQA",
      );
      assert(error.message.includes("ADR 0067"), `案内が足りない: ${error.message}`);
      // ①③ とも i8a8 適格な形（D=8 / N=8）での拒否理由。以下の非適格形はこれと**同一**の理由で
      // 落ちる（形ごとに変わるのは H / Hkv / r の表示だけ）。
      const reason = rejectionReason(error.message, shape.name);
      for (const ineligible of I8A8_INELIGIBLE) {
        // 被験体が本当に非適格であることを先に固定する（適格形が紛れ込むと被覆が消える）
        assert(
          ineligible.d % 4 !== 0 || ineligible.n % 4 !== 0,
          `${ineligible.name}: ①③ とも i8a8 適格 = 非適格形の被覆になっていない`,
        );
        const inputs = inputsFor(ineligible);
        const rejected = await assertRejects(
          () => runAttention(gpu, inputs.q, inputs.k, inputs.v, undefined, options),
          ExecutionError,
          "GQA",
        );
        assertEquals(
          rejectionReason(rejected.message, ineligible.name),
          reason,
          `${ineligible.name}: 適格形と拒否理由が違う（段別適格判定が拒否条件へ漏れている）`,
        );
      }
      // 非 GQA（実体化して H = Hkv にした同じ値）は i8a8 で走る
      const plain = await runAttention(gpu, q, kFull, vFull, undefined, options);
      assertEquals(plain.output.shape, [shape.b, shape.h, shape.m, shape.d], shape.name);
      assert(
        plain.output.data.every((value) => Number.isFinite(value)),
        `${shape.name}: 非 GQA の i8a8 出力に非有限値`,
      );
    } finally {
      gpu.destroy();
    }
  },
});
