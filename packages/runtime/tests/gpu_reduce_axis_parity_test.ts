// 軸 reduce 変種の**ビット同一の門**。本波で最重要のテスト。
//
// 同じ入力バッファに対し、
//
//   ① 現行の表現（`permute` で縮約軸を最終次元へ回す → 行 reduce `reduce:v2:…:wg256`）
//   ② 軸 reduce 変種（`reduce:v2:…:axis:wg256` — 1 スレッド 1 出力・permute 無し）
//
// を**同じ実 GPU**で流し、出力を **f32 のビット列**で突き合わせる。allclose ではなく
// `Uint32Array` の完全一致で見るのが要点で、tolerance に隠れる**縮約順序の変化**は
// ここでしか検出できない（数 ulp なので tolerance は素通りする）。
//
// 成立の根拠（設計 recon §5.2）: 行 reduce の縮約順序は「256 レーンの grid-stride 走査 +
// 256 幅のビット反転二分木」で、これを 1 スレッドの **bitrev carry-stack** で厳密再現できる
// ことを記号式で確認済み。`+0.0`（identity）の畳み込みも省略していない。
//
// MUST: 恒真化しないこと。①② は**別のパイプライン・別の WGSL**で、生成関数も別。
// MUST: **巡回長 3 以上の軸ケース**を必ず 1 本持つ（ACTIVE_DESIGN の落とし穴「実測グラフの
// permute は全て対合」対策 — 軸並べ替えの取り違えが対合ケースだけでは空振りする）。
// MUST: rows（= 出力要素数）が 65,535 の**両側**を踏む。行 reduce 側は行数がそのまま
// workgroup 数になるので、上限超えは縮退経路（本番経路）になる。
// MUST: `amax` / `amin` の **NaN 入りケース**を持つ（縮約長が 256 の両側で 1 本ずつ）。
// ADR 0020 決定 1 の「実 GPU で実走して確かめる」水準は、軸変種についてはここにしか無い
// （tests/gpu_ops_test.ts の NaN ケースは rank 2 の最終次元 = 行変種しか踏まない）。

import { assert, assertEquals } from "@std/assert";
import { gridStrideWorkgroups } from "../src/codegen/dispatch.ts";
import {
  AXIS_REDUCE_WORKGROUP_SIZE,
  axisReduceKey,
  axisReduceParams,
  axisReduceWgsl,
  reduceKey,
  reduceParams,
  reduceWgsl,
} from "../src/codegen/reduce.ts";
import {
  permuteSrcStrides,
  STRIDED_WORKGROUP_SIZE,
  stridedKey,
  stridedParams,
  stridedWgsl,
} from "../src/codegen/strided.ts";
import { openModel } from "../src/format/container.ts";
import { RunArena } from "../src/gpu/arena.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { PipelineCache } from "../src/gpu/pipeline-cache.ts";
import { SubmitScheduler } from "../src/gpu/submit.ts";
import type { IrDtype } from "../src/format/ir.ts";
import { numel, type ReduceOpName } from "../src/ops.ts";
import { allclose } from "../src/reference/allclose.ts";
import { referenceRowReduce, refTensor } from "../src/reference/ops.ts";
import { createSession } from "../src/runtime/executor.ts";
import { fill, graphModelBuffer, singleOpGraph } from "./helpers/graph.ts";
import { GPU_AVAILABLE, TIMESTAMP_QUERY_AVAILABLE, TIMING_ACQUIRE_OPTIONS } from "./helpers/gpu.ts";

const STORAGE_IN = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const UNIFORM_IN = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

/**
 * CPU 参照との突合の許容誤差。参照は JS の f64 で縮約するので f32 の 2 段縮約とは丸めが違い、
 * ビット一致は原理的に成立しない（`amax` / `amin` / bool の `sum` は丸めが無いので実測 0）。
 *
 * 実測（2026-08-04・`atol=rtol=0` の素の突合、下の {@link CASES} の NaN 無し 9 本）:
 * `[300,7] axis=0` maxAbs **7.15e-7** / maxRel **2.37e-6**（最悪）、
 * `[1,96,255,257]` 5.96e-7 / 1.67e-6、`[1,384,128,128]` 5.36e-7 / 7.01e-7。
 * atol 5e-6 は最悪の約 7.0 倍・rtol 2e-5 は約 8.4 倍。**実装バグの誤差は O(1)** なので
 * 桁が大きく離れている。
 */
const CPU_REFERENCE_TOLERANCE = { atol: 5e-6, rtol: 2e-5 } as const;

/** 決定的なデータ列（乱数は使わない — 失敗が再現しないため）。 */
const SIGNED = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
/** bool 入力（真の個数を数える形 — 0/1 が偏らないように 3 要素周期にしない）。 */
const BOOLEAN = (i: number): number => ((i * 5) % 7 < 3 ? 1 : 0);

/**
 * 縮約軸に沿って NaN を散らす入力生成器（`amax` / `amin` の NaN 伝播 — ADR 0020 決定 1）。
 *
 * 出力 1 要素ぶんの縮約列ごとに NaN の位置を **先頭 / 中間 / 末尾 / 無し**の 4 通りへ
 * 振り分ける。MUST: **NaN を含まない対照列**を必ず混ぜる — 位置の一致だけを見ると
 * 「常に NaN を返す」実装が落ちない（tests/gpu_ops_test.ts の `NAN_REDUCE_ROWS` と同じ理由）。
 */
/**
 * 縮約列を**丸ごと ±Infinity** にする入力生成器（`amax` / `amin` の identity の門 — E-R2-2）。
 *
 * 出力 1 要素ぶんの縮約列を、偶数番目は全要素 `sign`·Infinity、奇数番目は有限の対照列にする。
 * MUST: 全 −Inf の列の `amax` は GPU・CPU 参照とも **−F32_MAX**（identity 始まりで畳む —
 * `reference/ops.ts` の MUST）。参照が先頭要素始まりだった頃は −Infinity を返し、GPU と
 * 例外なしに割れていた。
 */
const infAlongAxis = (
  shape: readonly number[],
  axis: number,
  sign: -1 | 1,
): (i: number) => number => {
  const axisLen = shape[axis];
  const inner = numel(shape.slice(axis + 1));
  return (i: number): number => {
    const out = Math.floor(i / (axisLen * inner)) * inner + (i % inner);
    return out % 2 === 0 ? sign * Number.POSITIVE_INFINITY : SIGNED(i);
  };
};

const nanAlongAxis = (
  shape: readonly number[],
  axis: number,
): (i: number) => number => {
  const axisLen = shape[axis];
  const inner = numel(shape.slice(axis + 1));
  /** 縮約列ごとの NaN の位置（-1 = NaN 無しの対照）。 */
  const positions = [0, axisLen >> 1, axisLen - 1, -1];
  return (i: number): number => {
    // 入力添字 i = (outer·axis_len + c)·inner + j を分解する
    const c = Math.floor(i / inner) % axisLen;
    const out = Math.floor(i / (axisLen * inner)) * inner + (i % inner);
    return c === positions[out % positions.length] ? Number.NaN : SIGNED(i);
  };
};

type ParityCase = {
  readonly name: string;
  readonly shape: readonly number[];
  /** 縮約軸（最終次元は行 reduce の担当なので、ここでは常に最終次元より前）。 */
  readonly axis: number;
  readonly op: ReduceOpName;
  readonly dtype: IrDtype;
  /** NaN 入り入力（{@link nanAlongAxis}）を使うか。既定は NaN を含まない決定的な列。 */
  readonly nan?: true;
  /** 縮約列を丸ごと ±Infinity にする（{@link infAlongAxis} — identity の門）。 */
  readonly inf?: -1 | 1;
};

const readbackBits = async (
  device: GPUDevice,
  buffer: GPUBuffer,
  count: number,
): Promise<Uint32Array<ArrayBuffer>> => {
  const size = Math.max(4, count * 4);
  const staging = device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const copy = staging.getMappedRange().slice(0);
  staging.unmap();
  staging.destroy();
  return new Uint32Array(copy, 0, count);
};

/**
 * 「permute + 行 reduce」と「軸 reduce」を**同じ入力バッファ**で走らせ、出力のビット列を
 * 組で返す。ホスト側で 2 度書くと「書き込みの差」が「カーネルの差」に化けるので、入力は
 * 1 本だけ確保して両方に束縛する。
 */
const runBoth = async (
  gpu: GpuContext,
  testCase: ParityCase,
): Promise<
  {
    readonly viaPermute: Uint32Array;
    readonly viaAxis: Uint32Array;
    readonly permuteDims: readonly number[];
    readonly reference: Float32Array | Int32Array;
  }
> => {
  const { shape, axis, op, dtype } = testCase;
  const scheduler = new SubmitScheduler(gpu);
  const cache = new PipelineCache(gpu.device);
  const arena = new RunArena(gpu.device, () => scheduler.flush());
  try {
    const generator = testCase.nan === true
      ? nanAlongAxis(shape, axis)
      : testCase.inf !== undefined
      ? infAlongAxis(shape, axis, testCase.inf)
      : dtype === "bool"
      ? BOOLEAN
      : SIGNED;
    const x = fill(shape, generator, dtype === "bool" ? "bool" : "f32");
    const upload = (data: ArrayBufferView<ArrayBuffer>): GPUBuffer => {
      const buffer = arena.allocHostWritten(Math.max(4, data.byteLength), STORAGE_IN);
      gpu.device.queue.writeBuffer(buffer, 0, data);
      return buffer;
    };
    const xBuffer = upload(x.data);

    const axisLen = shape[axis];
    const outShape = [...shape.slice(0, axis), ...shape.slice(axis + 1)];
    const outCount = numel(outShape);
    const inner = numel(shape.slice(axis + 1));
    const limit = gpu.limits.maxComputeWorkgroupsPerDimension;

    const allocOut = (): GPUBuffer => {
      const buffer = arena.allocStorage(Math.max(4, outCount * 4));
      arena.retain(buffer, 0, { pinned: true });
      return buffer;
    };
    const bind = (
      layout: GPUBindGroupLayout,
      params: Uint32Array<ArrayBuffer>,
      usage: number,
      input: GPUBuffer,
      out: GPUBuffer,
    ): GPUBindGroup => {
      const paramsBuffer = arena.allocHostWritten(params.byteLength, usage);
      gpu.device.queue.writeBuffer(paramsBuffer, 0, params);
      return gpu.device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: input } },
          { binding: 2, resource: { buffer: out } },
        ],
      });
    };

    // ① 縮約軸を最終次元へ回す permute（現行のエクスポータが出していた形）→ 行 reduce
    const permuteDims = [...shape.keys()].filter((d) => d !== axis).concat(axis);
    const permutedShape = permuteDims.map((d) => shape[d]);
    const permuted = arena.allocStorage(Math.max(4, numel(shape) * 4));
    arena.retain(permuted, 0, { pinned: true });
    const { pipeline: stridedPipeline, layout: stridedLayout } = await cache.get(
      stridedKey({ dtype }),
      stridedWgsl({ dtype }),
    );
    const stridedGroup = gpu.device.createBindGroup({
      layout: stridedLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: (() => {
              const params = stridedParams(
                permutedShape,
                permuteSrcStrides(shape, permuteDims),
                0,
              );
              const buffer = arena.allocHostWritten(params.byteLength, STORAGE_IN);
              gpu.device.queue.writeBuffer(buffer, 0, params);
              return buffer;
            })(),
          },
        },
        { binding: 1, resource: { buffer: xBuffer } },
        { binding: 2, resource: { buffer: permuted } },
      ],
    });
    scheduler.dispatch(stridedPipeline, stridedGroup, [
      gridStrideWorkgroups(numel(shape), STRIDED_WORKGROUP_SIZE, limit),
      1,
      1,
    ], stridedKey({ dtype }));

    const rowKey = reduceKey({ op, dtype });
    const { pipeline: rowPipeline, layout: rowLayout } = await cache.get(
      rowKey,
      reduceWgsl({ op, dtype }),
    );
    const rowOut = allocOut();
    scheduler.dispatch(
      rowPipeline,
      bind(rowLayout, reduceParams(outCount, axisLen), UNIFORM_IN, permuted, rowOut),
      [gridStrideWorkgroups(outCount, 1, limit), 1, 1],
      rowKey,
    );

    // ② 軸 reduce 変種（permute 無し・1 スレッド 1 出力）
    const axisKey = axisReduceKey({ op, dtype });
    const { pipeline: axisPipeline, layout: axisLayout } = await cache.get(
      axisKey,
      axisReduceWgsl({ op, dtype }),
    );
    const axisOut = allocOut();
    scheduler.dispatch(
      axisPipeline,
      bind(
        axisLayout,
        axisReduceParams(outCount, axisLen, inner),
        UNIFORM_IN,
        xBuffer,
        axisOut,
      ),
      [gridStrideWorkgroups(outCount, AXIS_REDUCE_WORKGROUP_SIZE, limit), 1, 1],
      axisKey,
    );

    await scheduler.flush();
    const reference = referenceRowReduce(op, refTensor(shape, x.data), axis);
    return {
      viaPermute: await readbackBits(gpu.device, rowOut, outCount),
      viaAxis: await readbackBits(gpu.device, axisOut, outCount),
      permuteDims,
      reference: reference.data as Float32Array | Int32Array,
    };
  } finally {
    await arena.destroy();
  }
};

const CASES: readonly ParityCase[] = [
  {
    // census（VAE decoder 1024px）の実形状: rows 16,384 / dim 384（11 ノード）。
    // rows < 65,535 なので行 reduce 側は縮退しない。
    name: "実形状 [1,384,128,128] axis=1（rows 16,384 / dim 384）",
    shape: [1, 384, 128, 128],
    axis: 1,
    op: "sum",
    dtype: "f32",
  },
  {
    // census 実形状: rows 65,536 / dim 192（6 ノード側の縮小）。**rows > 65,535** なので
    // 行 reduce 側は grid-stride の縮退経路（本番経路）に入る。
    name: "実形状 [1,192,256,256] axis=1（rows 65,536 > 65,535 / dim 192）",
    shape: [1, 192, 256, 256],
    axis: 1,
    op: "sum",
    dtype: "f32",
  },
  {
    // rows = 65,535 ちょうど（クランプの境界そのもの）。dim 96 は census で要素の 58% を
    // 占める形で、workgroup 幅 256 に対し 160 レーンが identity のまま木を回る。
    name: "実形状 [1,96,255,257] axis=1（rows 65,535 = 境界 / dim 96）",
    shape: [1, 96, 255, 257],
    axis: 1,
    op: "sum",
    dtype: "f32",
  },
  {
    // rows が上限の 2 倍超（縮退経路を確実に 2 周させる）。dim 96。
    name: "実形状 [1,96,512,257] axis=1（rows 131,584 / dim 96）",
    shape: [1, 96, 512, 257],
    axis: 1,
    op: "sum",
    dtype: "f32",
  },
  {
    // **巡回長 3 以上の軸ケース（MUST）**: rank-4 の中間軸を畳むと permute は [0,2,3,1] で
    // 逆置換は [0,3,1,2]（= 自分自身ではない）。しかも 4 次元が全て違うので、outer / inner /
    // axis_len のどれを取り違えても値が割れる。
    name: "巡回長 3 [2,3,4,5] axis=1（permute [0,2,3,1] は対合でない）",
    shape: [2, 3, 4, 5],
    axis: 1,
    op: "sum",
    dtype: "f32",
  },
  {
    // 縮約長が 256 を跨ぐ（slot ごとの走査が 2 周する = carry-stack の葉が「1 要素」でない）。
    // 先頭軸の縮約（outer = 1）も同時に踏む。
    name: "縮約長 300 > 256 [300,7] axis=0（slot 走査が 2 周）",
    shape: [300, 7],
    axis: 0,
    op: "sum",
    dtype: "f32",
  },
  {
    // amax（NaN 伝播の畳み込み関数と ±F32_MAX の identity）。軸は rank-3 の中間。
    name: "amax [5,9,11] axis=1",
    shape: [5, 9, 11],
    axis: 1,
    op: "amax",
    dtype: "f32",
  },
  {
    // amin も同じ骨格（identity の符号だけが違う — 取り違えると全要素が +F32_MAX になる）
    name: "amin [5,9,11] axis=1",
    shape: [5, 9, 11],
    axis: 1,
    op: "amin",
    dtype: "f32",
  },
  {
    // **NaN 伝播の実 GPU の門（ADR 0020 決定 1・4）**。生成 WGSL の文字列の門は軸変種にも
    // 掛かっているが、ADR 0020 が「文字列では足りない」と名指しした畳み込みの故障
    // （バックエンドが `nan_max` を素の `max` へ畳む）は実走でしか見えない。縮約長 9 < 256 =
    // slot ごとの走査は 1 周（葉が 1 要素）。
    name: "amax NaN [5,9,11] axis=1（縮約長 9 < 256 / 走査 1 周）",
    shape: [5, 9, 11],
    axis: 1,
    op: "amax",
    dtype: "f32",
    nan: true,
  },
  {
    // amin も同じ骨格（NaN を飲むと identity の +F32_MAX に化ける）
    name: "amin NaN [5,9,11] axis=1（縮約長 9 < 256 / 走査 1 周）",
    shape: [5, 9, 11],
    axis: 1,
    op: "amin",
    dtype: "f32",
    nan: true,
  },
  {
    // 縮約長 300 > 256 = slot ごとの走査が 2 周し、carry-stack が実際に段を積む形。
    // 走査ループと畳み込みループの**両方**を NaN が通ることをここで見る。
    name: "amax NaN [300,7] axis=0（縮約長 300 > 256 / 走査 2 周）",
    shape: [300, 7],
    axis: 0,
    op: "amax",
    dtype: "f32",
    nan: true,
  },
  {
    name: "amin NaN [300,7] axis=0（縮約長 300 > 256 / 走査 2 周）",
    shape: [300, 7],
    axis: 0,
    op: "amin",
    dtype: "f32",
    nan: true,
  },
  {
    // **identity の門（E-R2-2）**: 縮約列が丸ごと −Inf の `amax` は −F32_MAX を返す（GPU は
    // identity 始まりで畳む。CPU 参照も同じ始点でなければ −Infinity を返して例外なしに割れる —
    // 有限の対照列を混ぜて `SIGNED` 側の値も同時に見る）。
    name: "amax 全 −Inf 列 [4,9,6] axis=1（identity −F32_MAX の門）",
    shape: [4, 9, 6],
    axis: 1,
    op: "amax",
    dtype: "f32",
    inf: -1,
  },
  {
    name: "amin 全 +Inf 列 [4,9,6] axis=1（identity +F32_MAX の門）",
    shape: [4, 9, 6],
    axis: 1,
    op: "amin",
    dtype: "f32",
    inf: 1,
  },
  {
    // bool → i32（真の個数）。累算器の型と `load` の真偽化が軸変種にも入っていること。
    name: "bool sum [3,13,5] axis=1（真の個数 → i32）",
    shape: [3, 13, 5],
    axis: 1,
    op: "sum",
    dtype: "bool",
  },
];

/** ビット列の食い違い（先頭 4 件）を人が読める形にする。 */
const firstMismatches = (a: Uint32Array, b: Uint32Array): readonly string[] => {
  const found: string[] = [];
  for (let i = 0; i < a.length && found.length < 4; i += 1) {
    if (a[i] !== b[i]) found.push(`[${i}] 0x${a[i].toString(16)} vs 0x${b[i].toString(16)}`);
  }
  return found;
};

Deno.test({
  name: "軸 reduce は「permute + 行 reduce」と**ビット単位で一致**する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    let sawNonInvolution = false;
    try {
      for (const testCase of CASES) {
        const result = await runBoth(gpu, testCase);
        const { viaPermute, viaAxis, permuteDims, reference } = result;
        assertEquals(
          firstMismatches(viaPermute, viaAxis),
          [],
          `${testCase.name}: permute + 行 reduce とビット列が違う`,
        );
        // 恒真化の門（出力が定数なら一致は何も検証していない）
        assert(
          new Set(viaAxis).size > 1,
          `${testCase.name}: 出力が定数（ビット一致が恒真になっている）`,
        );
        // 両カーネルが**揃って**誤る形の検出器（ビット比較だけでは気づけない）
        const values = testCase.dtype === "bool"
          ? new Int32Array(viaAxis.buffer, viaAxis.byteOffset, viaAxis.length)
          : new Float32Array(viaAxis.buffer, viaAxis.byteOffset, viaAxis.length);
        if (testCase.nan === true) {
          // MUST: allclose は NaN をどちらの側でも不合格にするので NaN ケースには使えない —
          // **isNaN パターンの一致 + 非 NaN 要素の厳密一致**で突き合わせる
          // （tests/gpu_ops_test.ts の行変種の NaN 判定と同じ形）。
          const actualValues = [...values];
          const expectedValues = [...reference];
          const actualNan = actualValues.map((value) => Number.isNaN(value));
          const expectedNan = expectedValues.map((value) => Number.isNaN(value));
          assert(expectedNan.includes(true), `${testCase.name}: 参照に NaN が 1 つも無い`);
          // MUST: 対照要素を持つこと（「常に NaN を返す」実装は位置の一致では落ちない）
          assert(expectedNan.includes(false), `${testCase.name}: 対照要素が無い`);
          assertEquals(actualNan, expectedNan, `${testCase.name}: NaN の位置`);
          // 非 NaN 側は `amax` / `amin` なので丸め差が入らない（NaN 同士は比較が成立しない
          // ので該当要素だけ 0 に潰してから見る）
          const finiteOnly = (
            source: readonly number[],
            isNan: readonly boolean[],
          ): readonly number[] => source.map((value, index) => (isNan[index] ? 0 : value));
          assertEquals(
            finiteOnly(actualValues, actualNan),
            finiteOnly(expectedValues, expectedNan),
            `${testCase.name}: 非 NaN の値`,
          );
        } else {
          const report = allclose(values, reference, CPU_REFERENCE_TOLERANCE);
          assert(
            report.pass,
            `${testCase.name}: CPU 参照と食い違う（maxAbs ${report.maxAbsError} @${report.worstIndex} / 破り ${report.failCount}）`,
          );
        }
        // 巡回長 3 以上（逆置換が自分自身でない）を 1 本以上踏んでいること
        const inverse = permuteDims.map((_, d) => permuteDims.indexOf(d));
        if (inverse.some((value, index) => value !== permuteDims[index])) sawNonInvolution = true;
      }
      assert(sawNonInvolution, "permute が全て対合のケースしか無い（軸取り違えが空振りする）");
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * executor の踏み分け。最終次元 = 行 reduce（既存キー）、それ以外 = 軸変種。
 *
 * MUST: 値ではなく**実際に走ったパイプラインキー**で見る。どちらのカーネルも正しいので、
 * 出力の一致は踏み分けの証拠にならない（ADR 0024 / 0030 と同型の規律）。
 */
const reduceKeysUsed = async (
  shape: readonly number[],
  axis: number,
): Promise<ReadonlySet<string>> => {
  const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
  const outShape = [...shape.slice(0, axis), ...shape.slice(axis + 1)];
  const graph = singleOpGraph("sum", [shape], [outShape], { attrs: { dim: axis } });
  const session = await createSession(gpu, openModel(graphModelBuffer(graph)));
  try {
    await session.run({ x0: fill(shape, SIGNED) });
    const timing = session.diagnostics().lastRunTiming;
    assert(timing !== undefined, "timestamp-query が無効（キー別内訳が取れない）");
    return new Set(timing.entries.map((entry) => entry.key));
  } finally {
    await session.dispose();
    gpu.destroy();
  }
};

Deno.test({
  name: "executor は縮約軸で 2 カーネルを踏み分ける（最終次元 = 行 / それ以外 = 軸・実 GPU）",
  ignore: !GPU_AVAILABLE || !TIMESTAMP_QUERY_AVAILABLE,
  fn: async () => {
    assertEquals(
      await reduceKeysUsed([4, 6, 8], 2),
      new Set([reduceKey({ op: "sum", dtype: "f32" })]),
    );
    assertEquals(
      await reduceKeysUsed([4, 6, 8], 1),
      new Set([axisReduceKey({ op: "sum", dtype: "f32" })]),
    );
    assertEquals(
      await reduceKeysUsed([4, 6, 8], 0),
      new Set([axisReduceKey({ op: "sum", dtype: "f32" })]),
    );
  },
});
