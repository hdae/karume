/**
 * 融合 attention（ADR 0023）の**クエリ行ブロック実行**の実 GPU 門。分解経路の行ブロック
 * （融合ルール `rowBlockAttention` — tests/gpu_row_block_attention_test.ts）と同じ 2 本
 * （①強制分割 parity / ②ポータビリティ）に、融合経路だけが持つ 2 本
 * （ヒット run の slot 導出・行窓キーの census）を足した 4 本。
 *
 * ① **強制分割 parity**: 同じ入力・同じ Session 設定で
 *    - 既定（= 上限に余裕がある機では 1 枚 = 分割前と完全に同一の経路）
 *    - `ROW_BLOCK_SPLIT` で 2 / 3 / 5 枚を強制した形
 *    を回して **f32 のビット列**で突き合わせる。allclose ではなく `Uint32Array` の完全一致で
 *    見るのが要点 — 行ブロック化は「どの workgroup がどの行を担当するか」しか変えないので、
 *    1 ulp でも動いたら K 縮約順か丸め列（あるいは行 / 統計の添字）が動いている。
 *    変種（s16 格納 / i8a8 / 加算 mask / GQA）を横断で回すのは、行窓が**段ごとに別の側**
 *    （①QK は A 側 = q、③PV は C 側 = O）へ入るため、片側だけ写し忘れた実装が
 *    「値は合っているのに片方の変種だけ壊れる」形で残るから。
 *
 * ② **ポータビリティ門**: `maxStorageBufferBindingSize` を WebGPU core 既定（128MiB）へ絞った
 *    device で、S が 128MiB を超える形が通ること。**同じ device・同じ形で `ROW_BLOCK_SPLIT: 1`
 *    を強制すると fail loudly になる**ことも対で見る（落ちない形を緑にしても、行ブロックが
 *    効いている証拠にならない）。
 *
 * MUST: 恒真化しないこと。①は中間ピーク（`peakTransientBytes`）が枚数とともに**単調に縮む**
 * ことを同時に見る — 見なければ「分割が発火しないまま 2 回同じ経路を撃った」形が常に緑になる。
 * 出力が定数でないことも毎回確かめる。
 */

import { assert, assertEquals, assertRejects } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { acquireGpu, type GpuContext, LIMIT_CAPS } from "../src/gpu/device.ts";
import { planRowBlocks } from "../src/runtime/fusion.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import {
  createSession,
  ROW_BLOCK_SPLIT,
  type SessionOptions,
  type Tensor,
} from "../src/runtime/executor.ts";
import { fill, type FilledTensor, graphModelBuffer, singleOpGraph } from "./helpers/graph.ts";
import { GPU_AVAILABLE, TIMESTAMP_QUERY_AVAILABLE, TIMING_ACQUIRE_OPTIONS } from "./helpers/gpu.ts";

/** WebGPU core 既定のストレージ束縛上限。ポータビリティ門はここを再現する。 */
const CORE_STORAGE_BINDING_LIMIT = 128 * 1024 * 1024;

/** 半スケール（torch math decomp の `√scale_factor`）。D から導く契約どおりの値。 */
const halfScale = (depth: number): number => Math.fround(Math.sqrt(1 / Math.sqrt(depth)));

/** 決定的なデータ列（乱数は使わない — 失敗が再現しないため）。 */
const QUERY = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
const KEY = (i: number): number => (((i * 11) % 19) - 9) * 0.23;
const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.31;

type Shape = {
  readonly name: string;
  readonly b: number;
  readonly h: number;
  /** K / V の head 数（`h` と違えば GQA — ADR 0067）。 */
  readonly kv: number;
  readonly m: number;
  readonly n: number;
  readonly d: number;
};

type Variant = {
  readonly name: string;
  readonly options: SessionOptions;
  /** 加算 mask `[1,1,M,N]` を付けるか（**行ごとに違う** mask = 行オフセットの検出器）。 */
  readonly mask?: boolean;
  /** GQA 形（`h != kv`）で回せるか（i8a8 は GQA と組めない — ADR 0067 決定 3）。 */
  readonly gqa?: boolean;
};

/**
 * 変種の横断。行窓は ①QK（A 側 = q / mask の行）と ③PV（C 側 = O）で**別の場所**に入り、
 * i8a8 では更に別カーネル（q の量子化結果を全 M ストライドで読む）になるので、
 * 経路ごとに 1 本ずつ通す。
 */
const VARIANTS: readonly Variant[] = [
  { name: "f32", options: {}, gqa: true },
  { name: "f32 + mask", options: {}, mask: true, gqa: true },
  { name: "s16", options: { attentionScoreStorage: "f16" }, gqa: true },
  { name: "a8", options: { attentionCompute: "a8" } },
  { name: "a8 + s16", options: { attentionCompute: "a8", attentionScoreStorage: "f16" } },
];

const SHAPES: readonly Shape[] = [
  // 全軸が端数（D % 4 ≠ 0 / N % 4 ≠ 0 = ①③ ともスカラ変種・s16 も i8a8 も縮退する経路）。
  // M = 17 は 2 / 3 / 5 枚のどれでも端数割になる。
  { name: "端数 B2 H3 M17 N19 D13", b: 2, h: 3, kv: 3, m: 17, n: 19, d: 13 },
  // v4 経路（D % 4 == 0 && N % 4 == 0）で M が行タイル 2 枚を跨ぐ。
  { name: "v4 B1 H2 M68 N20 D12", b: 1, h: 2, kv: 2, m: 68, n: 20, d: 12 },
  // DiT の self-attention を縮めた形（D = 128・M / N とも 64 の倍数 = 実モデルの経路）。
  { name: "DiT 形 B1 H4 M64 N64 D128", b: 1, h: 4, kv: 4, m: 64, n: 64, d: 128 },
  // GQA（r = 4）。①QK と ③PV で kv-head への写像が入ったまま行窓が乗る形。
  { name: "GQA B1 H8 Hkv2 M40 N24 D16", b: 1, h: 8, kv: 2, m: 40, n: 24, d: 16 },
];

/** 強制する枚数（1 = 既定と同じ形・2/3 は等分・5 は端数割）。 */
const SPLITS: readonly number[] = [2, 3, 5];

/**
 * 実測形の band mask（行ごとに中心が動く = **行オフセットを取り違えたら必ず値が変わる**）。
 *
 * MUST: **全ての行に非マスク列が 1 本以上**あること（行が丸ごと −inf だと amax が −inf に
 * なり `exp(−inf −(−inf))` が NaN — 分割側も非分割側も同じ NaN を出すのでビット比較は
 * 通ってしまい、検証としては何も見ていない状態になる）。対称バンドは中心が必ず非マスク。
 */
const bandMask = (m: number, n: number): FilledTensor => {
  const width = Math.max(1, Math.floor(n / 4));
  return fill([1, 1, m, n], (index) => {
    const row = Math.floor(index / n);
    const col = index % n;
    const center = Math.floor((row * n) / m);
    return Math.abs(col - center) <= width ? 0 : Number.NEGATIVE_INFINITY;
  });
};

/** 出力と、その run が実際に確保した中間バッファのピーク（行ブロックの効きの観測点）。 */
type RunResult = {
  /** run ごとの出力（2 本目以降は導出済み計画のヒット run — 別の束縛経路を通る）。 */
  readonly outputs: readonly Tensor[];
  readonly output: Tensor;
  /** **1 run 目**（アリーナ経路）の中間ピーク。 */
  readonly peakTransientBytes: number;
  /** パイプラインキー別の dispatch 回数（timestamp-query がある機だけ埋まる）。 */
  readonly dispatchCounts: ReadonlyMap<string, number>;
};

const runAttention = async (
  gpu: GpuContext,
  shape: Shape,
  variant: Variant,
  split?: number,
  runs = 1,
): Promise<RunResult> => {
  const { b, h, kv, m, n, d } = shape;
  const q = fill([b, h, m, d], QUERY);
  const k = fill([b, kv, n, d], KEY);
  const v = fill([b, kv, n, d], VALUE);
  const mask = variant.mask === true ? bandMask(m, n) : undefined;
  const graph = singleOpGraph(
    "attention",
    [q.shape, k.shape, v.shape, ...(mask === undefined ? [] : [mask.shape])],
    [[b, h, m, d]],
    { attrs: { scale: halfScale(d) } },
  );
  const options: SessionOptions = {
    ...variant.options,
    ...(split === undefined ? {} : { [ROW_BLOCK_SPLIT]: split }),
  };
  const session = await createSession(gpu, openModel(graphModelBuffer(graph)), options);
  try {
    const inputs: Record<string, FilledTensor> = { x0: q, x1: k, x2: v };
    if (mask !== undefined) inputs["x3"] = mask;
    const outputs: Tensor[] = [];
    let peakTransientBytes = 0;
    let dispatchCounts = new Map<string, number>();
    for (let run = 0; run < runs; run += 1) {
      outputs.push((await session.run(inputs))["y"]);
      const diagnostics = session.diagnostics();
      // MUST: ピークと内訳は **1 run 目**（アリーナ経路）のもの。ヒット run は常駐 backing を
      // 使い回すので一時の確保が出ず、上書きすると観測点が空振りする。
      if (run > 0) continue;
      peakTransientBytes = diagnostics.lastRun?.peakTransientBytes ?? 0;
      dispatchCounts = new Map(
        (diagnostics.lastRunTiming?.entries ?? []).map((entry) => [entry.key, entry.dispatchCount]),
      );
    }
    return { outputs, output: outputs[0], peakTransientBytes, dispatchCounts };
  } finally {
    await session.dispose();
  }
};

/** f32 のビット列（`0.0` と `-0.0` の差も、末尾 1 ulp の差も見える形）。 */
const bits = (tensor: Tensor): Uint32Array =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length);

const assertSameBits = (actual: Tensor, expected: Tensor, label: string): void => {
  assertEquals(actual.shape, expected.shape, label);
  const a = bits(actual);
  const e = bits(expected);
  const mismatches: number[] = [];
  for (let i = 0; i < a.length && mismatches.length < 4; i += 1) {
    if (a[i] !== e[i]) mismatches.push(i);
  }
  assertEquals(
    mismatches,
    [],
    `${label}: ビット列が違う（最初の食い違い: ${
      mismatches.map((i) => `[${i}] ${actual.data[i]} vs ${expected.data[i]}`).join(" / ")
    }）`,
  );
  // 恒真化の門: 出力が定数なら「一致」は何も検証していない。
  assert(new Set(a).size > 1, `${label}: 出力が定数（ビット一致が恒真になっている）`);
};

Deno.test({
  name:
    "融合 attention の行ブロック実行は枚数を変えても 1 枚実行と**ビット単位で一致**する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const shape of SHAPES) {
        const gqaShape = shape.h !== shape.kv;
        for (const variant of VARIANTS) {
          if (gqaShape && variant.gqa !== true) continue;
          const label = `${shape.name} / ${variant.name}`;
          // 正本 = 分割前（上限に余裕がある実機では 1 枚 = 行窓を立てない既存の経路そのもの）。
          const reference = await runAttention(gpu, shape, variant);
          let previousPeak = reference.peakTransientBytes;
          assert(previousPeak > 0, `${label}: 中間ピークが 0（観測点が空振りしている）`);
          for (const split of SPLITS) {
            const blocked = await runAttention(gpu, shape, variant, split);
            assertSameBits(blocked.output, reference.output, `${label} / ${split} 枚`);
            // 枚数を増やすほど S の実体化幅は縮む（値が同じでも**実体化幅が縮んでいる**ことの
            // 観測点 — ここが単調でなければ一時の寿命宣言かプール再利用が壊れている）。
            assert(
              blocked.peakTransientBytes < previousPeak,
              `${label} / ${split} 枚: 中間ピーク ${blocked.peakTransientBytes}B が ${previousPeak}B から縮んでいない`,
            );
            previousPeak = blocked.peakTransientBytes;
          }
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * **ヒット run**（導出済み計画 = 常駐 backing + 焼き込み bind group — src/runtime/executor.ts の
 * `#activateBacking`）でも、行ブロックの一時がミス run と同じ slot 割りになること。
 *
 * 行ブロック化はステップ内一時の**本数と寿命**を変えるので、slot 導出（`derivePlanSlots`）が
 * アリーナ経路と食い違うと「2 run 目だけ別のバッファへ書く」形の沈黙誤値になる — 例外は
 * 出ないので、ビット比較でしか見えない。
 */
Deno.test({
  name: "行ブロック実行はヒット run（導出済み計画）でもミス run とビット単位で一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      const shape = SHAPES[2];
      for (const variant of [VARIANTS[0], VARIANTS[4]]) {
        const reference = await runAttention(gpu, shape, variant);
        // 端数割（M=64 → 22/21/21）で回す — 等分だとブロック間で slot が使い回されるので、
        // サイズが違う 2 種類の slot が並ぶ形の方が導出の食い違いが出やすい。
        const blocked = await runAttention(gpu, shape, variant, 3, 2);
        assertEquals(blocked.outputs.length, 2);
        for (const [index, output] of blocked.outputs.entries()) {
          assertSameBits(
            output,
            reference.output,
            `${shape.name} / ${variant.name} / run ${index}`,
          );
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * **census**（ADR 0058 決定 4）。行窓変種のキー（`:rwa` / `:rwc`）が n ≥ 2 で**実際に走り**、
 * その dispatch 回数がちょうど枚数になること・1 枚では 1 本も立たないこと。
 *
 * MUST: `TIMING_ACQUIRE_OPTIONS` を渡す（素の `acquireGpu()` では `lastRunTiming` が
 * undefined になり、キー検査が黙って空振りする）。計測を要求できない機では**明示 SKIP**。
 */
Deno.test({
  name: "行窓のキーは n ≥ 2 でだけ立ち、dispatch 回数が枚数と一致する（実 GPU / timestamp-query）",
  ignore: !GPU_AVAILABLE || !TIMESTAMP_QUERY_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      const shape = SHAPES[2];
      for (const variant of VARIANTS) {
        const label = `${shape.name} / ${variant.name}`;
        const single = await runAttention(gpu, shape, variant);
        assert(single.dispatchCounts.size > 0, `${label}: 内訳が空（キー検査が空振りしている）`);
        assertEquals(
          [...single.dispatchCounts.keys()].filter((key) => key.includes(":rw")),
          [],
          `${label}: 1 枚なのに行窓のキーが立っている`,
        );
        for (const split of SPLITS) {
          const blocked = await runAttention(gpu, shape, variant, split);
          const windowed = [...blocked.dispatchCounts].filter(([key]) => key.includes(":rw"));
          assertEquals(
            windowed.length,
            2,
            `${label} / ${split} 枚: 行窓のキーは ①QK と ③PV の 2 本（実際は ${
              windowed.map(([key]) => key).join(" / ")
            }）`,
          );
          for (const [key, count] of windowed) {
            assertEquals(count, split, `${label} / ${split} 枚: ${key} の dispatch 回数`);
          }
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * ポータビリティ門の形。**実資産を要らない**形で `S = B·H·M·N·4` を 128MiB 超へ持ち上げる
 * （4 × 2048 × 4200 × 4 = 137,625,600B）。D を 8 に絞ってあるので GEMM の仕事量自体は小さく、
 * 見ているのは「上限を超える中間を実体化せずに済むか」の 1 点だけ。
 */
const OVERSIZE: Shape = {
  name: "上限超 B1 H4 M2048 N4200 D8",
  b: 1,
  h: 4,
  kv: 4,
  m: 2048,
  n: 4200,
  d: 8,
};

Deno.test({
  name:
    "maxStorageBufferBindingSize=128MiB へ絞った device で、S が上限を超える融合 attention が行ブロックで通る（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const { b, h, m, n } = OVERSIZE;
    const scoreBytes = b * h * m * n * 4;
    assert(
      scoreBytes > CORE_STORAGE_BINDING_LIMIT,
      `合成グラフの S ${scoreBytes}B が core 既定の上限を超えていない（門が空振りする）`,
    );
    // 実行が読むのと**同じ純関数**で、この形が 2 枚に割れて 1 枚が上限の内側へ収まることを
    // 先に固定する（GPU 側が緑になった理由を「たまたま」にしない）。
    const blocks = planRowBlocks(m, b * h * n * 4, CORE_STORAGE_BINDING_LIMIT);
    assertEquals(blocks.length, 2, "上限に収まる最小枚数");
    const gpu = await acquireGpu({
      [LIMIT_CAPS]: { maxStorageBufferBindingSize: CORE_STORAGE_BINDING_LIMIT },
    });
    try {
      assertEquals(
        gpu.limits.maxStorageBufferBindingSize,
        CORE_STORAGE_BINDING_LIMIT,
        "requiredLimits が絞られていない（絞れていなければ門は何も見ていない）",
      );
      const blocked = await runAttention(gpu, OVERSIZE, VARIANTS[0]);
      assertEquals(blocked.output.shape, [b, h, m, OVERSIZE.d]);
      assert(new Set(bits(blocked.output)).size > 1, "出力が定数（門が空振りしている）");

      // 門の効力証明: **同じ device・同じ形**でも 1 枚を強制すると、S を丸ごと実体化する形に
      // なるので fail loudly になる。ここが緑のままなら上の緑は行ブロックのおかげではない。
      await assertRejects(
        () => runAttention(gpu, OVERSIZE, VARIANTS[0], 1),
        ExecutionError,
        "上限に収まらない",
      );
    } finally {
      gpu.destroy();
    }
  },
});
