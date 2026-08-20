// i4（K 方向 group symmetric packed 4bit）格納の実行経路（ADR 0069）— 適格判定
// （linear / embedding 限定）・カーネル変種・group scale 添字・診断の通し検証。
//
// f16 / i8 と同じ 2 経路を踏む:
//   適格（消費が **linear / embedding の重みスロットだけ** — i8 より狭い）→ packed のまま
//     GPU 常駐し、dequant はカーネル内（`unpack4xU8` + マスク/シフト + group scale）
//   適格外（展開経路の無い重みスロット〈conv 系〉と共有・グラフ出力 等）→ ロード時に CPU で
//     f32 展開
//
// MUST: 重みは **group ごとに大きさを変える**（scale が全 group で同じだと、scale 添字の
// 取り違え〈行/group の入れ替え・shift 誤り〉が一切値に出ない — i8 の行別 scale と同型の罠）。
// MUST: 値は**隣接要素が全て異なる非対称パターン**を含める（pack の上下 nibble を取り違えても
// 対称パターンでは値が合う — ADR 0069 決定 4 ①）。
// MUST: 期待値は**丸め後の重み**（fake-quant — ADR 0006）で作る。

import { assert, assertEquals, assertRejects } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { decodeI4 } from "../src/format/i4.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { compareTensors, formatAllclose } from "../src/reference/allclose.ts";
import { applyReferenceOp, type RefTensor, refTensor } from "../src/reference/ops.ts";
import { createSession, type Tensor } from "../src/runtime/executor.ts";
import { i4EligibleInitializers } from "../src/runtime/plan.ts";
import { buildSafetensors, f32Bytes, type GraphJson } from "./helpers/format.ts";
import { quantizeI4 } from "./helpers/i4.ts";
import { fill, type FilledTensor } from "./helpers/graph.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const SIGNED = (i: number): number => ((i % 13) - 6) * 0.75;

/**
 * group ごとに振幅の違う重み（group scale の添字が値に出る形 — 恒真化の防波堤）。
 * さらに隣接要素の符号を交互にして、pack の上下 nibble の取り違えも値に出す。
 */
const groupVarying = (cols: number, groupSize: number) => (i: number): number => {
  const group = Math.floor((i % cols) / groupSize);
  const row = Math.floor(i / cols);
  const base = (0.125 + (i % 11) * 0.5) * (i % 2 === 0 ? 1 : -1);
  return base * (1 + group * 0.75 + (row % 5) * 0.25);
};

type I4Case = {
  readonly name: string;
  readonly x: FilledTensor;
  readonly weight: FilledTensor;
  readonly bias: FilledTensor;
  readonly groupSize: number;
  readonly outShape: readonly number[];
};

/** `linear(x, w, b)` 1 本のグラフ（w は i4 + group scale）。`wAsOutput` で w を適格外にできる。 */
const i4LinearModel = (
  testCase: I4Case,
  quantized: ReturnType<typeof quantizeI4>,
  { wAsOutput = false }: { wAsOutput?: boolean } = {},
): ArrayBuffer => {
  const graph: GraphJson = {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [...testCase.x.shape] }],
    outputs: wAsOutput ? ["y", "w"] : ["y"],
    initializers: {
      w: {
        tensor: "m.w",
        storage: { dtype: "i4", scale: "m.s", group_size: testCase.groupSize },
      },
      b: { tensor: "m.b", storage: { dtype: "f32" } },
    },
    values: {
      w: { dtype: "f32", shape: [...testCase.weight.shape] },
      b: { dtype: "f32", shape: [...testCase.bias.shape] },
      y: { dtype: "f32", shape: [...testCase.outShape] },
    },
    nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
  };
  return buildSafetensors(
    [
      { name: "m.w", dtype: "I4", shape: [...testCase.weight.shape], data: quantized.bytes },
      {
        name: "m.s",
        dtype: "F32",
        shape: [...quantized.scaleShape],
        data: f32Bytes([...quantized.scale]),
      },
      {
        name: "m.b",
        dtype: "F32",
        shape: [...testCase.bias.shape],
        data: f32Bytes([...testCase.bias.data]),
      },
    ],
    { karume_ir: JSON.stringify(graph) },
  );
};

const expectedLinear = (testCase: I4Case, quantized: ReturnType<typeof quantizeI4>): RefTensor =>
  applyReferenceOp(
    "linear",
    [
      testCase.x as RefTensor,
      refTensor(testCase.weight.shape, quantized.values),
      testCase.bias as RefTensor,
    ],
    {},
    testCase.outShape,
  );

/** GEMM タイル境界・group 境界・v4 / スカラの両経路を踏む形状群。 */
const CASES: readonly I4Case[] = [
  {
    // v4 経路（k, n とも 4 の倍数）。k = 48 は group 16 が 3 つ = 行内で scale が 2 回変わり、
    // K タイル（16 要素）も跨ぐ。n = 68 は最終 N タイルが端数、m = 65 は行タイル 2 枚。
    name: "linear v4 [65,48] × W[68,48] g16",
    x: fill([65, 48], SIGNED),
    weight: fill([68, 48], groupVarying(48, 16)),
    bias: fill([68], SIGNED),
    groupSize: 16,
    outShape: [65, 68],
  },
  {
    // スカラ経路（n = 19 が 4 の倍数でない）。k = 32・group 16 = 行内 2 group。
    name: "linear スカラ [17,32] × W[19,32] g16",
    x: fill([17, 32], SIGNED),
    weight: fill([19, 32], groupVarying(32, 16)),
    bias: fill([19], SIGNED),
    groupSize: 16,
    outShape: [17, 19],
  },
  {
    // group 32（既定 — ADR 0069 追記 1）。k = 96 で 3 group / 行。
    name: "linear v4 [9,96] × W[20,96] g32",
    x: fill([9, 96], SIGNED),
    weight: fill([20, 96], groupVarying(96, 32)),
    bias: fill([20], SIGNED),
    groupSize: 32,
    outShape: [9, 20],
  },
];

Deno.test({
  name: "w=i4 の linear が CPU 参照（丸め後の重み）と一致する（v4 / スカラ / group 2 種・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of CASES) {
        const quantized = quantizeI4(
          testCase.weight.data,
          testCase.weight.shape,
          testCase.groupSize,
        );
        const session = await createSession(gpu, openModel(i4LinearModel(testCase, quantized)));
        let output: Tensor;
        let residentBytes: number;
        try {
          output = (await session.run({ x: testCase.x }))["y"];
          residentBytes = session.diagnostics().storage.residentCompressedBytes;
        } finally {
          await session.dispose();
        }
        // packed のまま常駐している（= CPU 展開に落ちていない）。i4 は詰め物が原理的に不要
        // （バイト長 = numel / 2 は必ず 8 の倍数）なので、payload + scale がちょうど載る。
        assertEquals(
          residentBytes,
          quantized.bytes.byteLength + quantized.scale.byteLength,
          `${testCase.name}: GPU 常駐圧縮バイト数（scale 込み）`,
        );
        const expected = expectedLinear(testCase, quantized);
        assertEquals(output.shape, expected.shape, testCase.name);
        const report = compareTensors(output, expected);
        assertEquals(report.pass, true, `${testCase.name}: ${formatAllclose(report)}`);
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "同一グラフ内の group 16 / 32 が別パイプラインで正しく走る（キーの g 部の分離・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // 同じ m / n / k / v4 の linear 2 本 — キーが g 部でしか違わない形。キーに group が
    // 乗っていないと片方の WGSL（焼き込んだ shift）が他方の資産で走り、scale 添字が
    // ずれた沈黙誤値になる。
    const x = fill([9, 32], SIGNED);
    const w16 = fill([20, 32], groupVarying(32, 16));
    const w32 = fill([20, 32], groupVarying(32, 32));
    const bias = fill([20], SIGNED);
    const q16 = quantizeI4(w16.data, w16.shape, 16);
    const q32 = quantizeI4(w32.data, w32.shape, 32);
    const graph: GraphJson = {
      format: "karume-ir",
      version: 1,
      requires: { ops: ["linear"] },
      symbols: [],
      inputs: [{ name: "x", dtype: "f32", shape: [9, 32] }],
      outputs: ["y16", "y32"],
      initializers: {
        w16: { tensor: "m.w16", storage: { dtype: "i4", scale: "m.s16", group_size: 16 } },
        w32: { tensor: "m.w32", storage: { dtype: "i4", scale: "m.s32", group_size: 32 } },
        b: { tensor: "m.b", storage: { dtype: "f32" } },
      },
      values: {
        w16: { dtype: "f32", shape: [20, 32] },
        w32: { dtype: "f32", shape: [20, 32] },
        b: { dtype: "f32", shape: [20] },
        y16: { dtype: "f32", shape: [9, 20] },
        y32: { dtype: "f32", shape: [9, 20] },
      },
      nodes: [
        { op: "linear", ins: ["x", "w16", "b"], outs: ["y16"], attrs: {} },
        { op: "linear", ins: ["x", "w32", "b"], outs: ["y32"], attrs: {} },
      ],
    };
    const buffer = buildSafetensors(
      [
        { name: "m.w16", dtype: "I4", shape: [20, 32], data: q16.bytes },
        { name: "m.s16", dtype: "F32", shape: [...q16.scaleShape], data: f32Bytes([...q16.scale]) },
        { name: "m.w32", dtype: "I4", shape: [20, 32], data: q32.bytes },
        { name: "m.s32", dtype: "F32", shape: [...q32.scaleShape], data: f32Bytes([...q32.scale]) },
        { name: "m.b", dtype: "F32", shape: [20], data: f32Bytes([...bias.data]) },
      ],
      { karume_ir: JSON.stringify(graph) },
    );
    const gpu = await acquireGpu();
    try {
      const session = await createSession(gpu, openModel(buffer));
      try {
        const outputs = await session.run({ x });
        for (
          const [name, weight, quantized] of [
            ["y16", w16, q16] as const,
            ["y32", w32, q32] as const,
          ]
        ) {
          const expected = applyReferenceOp(
            "linear",
            [x as RefTensor, refTensor(weight.shape, quantized.values), bias as RefTensor],
            {},
            [9, 20],
          );
          const report = compareTensors(outputs[name], expected);
          assertEquals(report.pass, true, `${name}: ${formatAllclose(report)}`);
        }
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "適格（GPU dequant）と適格外（CPU 展開）の出力がビット単位で一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // w をグラフ出力に足すと適格外（消費が重みスロット以外にもある扱い）になり、ロード時に
    // decodeI4 で f32 展開される。dequant がビット一致なら、同じ GEMM カーネルが同じ f32 の
    // MAC 列を同じ順で畳むので、y は**ビット単位で**一致する（ADR 0069 受入 ④ の e2e 形）。
    const testCase = CASES[1];
    const quantized = quantizeI4(testCase.weight.data, testCase.weight.shape, testCase.groupSize);
    const gpu = await acquireGpu();
    try {
      const run = async (wAsOutput: boolean) => {
        const session = await createSession(
          gpu,
          openModel(i4LinearModel(testCase, quantized, { wAsOutput })),
        );
        try {
          const outputs = await session.run({ x: testCase.x });
          return {
            y: Float32Array.from(outputs["y"].data as Float32Array),
            diagnostics: session.diagnostics().storage,
          };
        } finally {
          await session.dispose();
        }
      };
      const resident = await run(false);
      const expanded = await run(true);
      assert(resident.diagnostics.residentCompressedBytes > 0, "適格側が常駐していない");
      assertEquals(expanded.diagnostics.residentCompressedBytes, 0, "適格外側が常駐している");
      assert(expanded.diagnostics.hostExpandedBytes > 0, "適格外側が CPU 展開されていない");
      assertEquals(resident.y, expanded.y, "GPU dequant と CPU 展開の出力がビット一致しない");
    } finally {
      gpu.destroy();
    }
  },
});

/** embedding の語彙表 `[V,D]` を i4 で持つグラフ（`wAsOutput` で適格外にできる）。 */
const i4EmbeddingModel = (
  weight: FilledTensor,
  index: FilledTensor,
  quantized: ReturnType<typeof quantizeI4>,
  groupSize: number,
  { wAsOutput = false }: { wAsOutput?: boolean } = {},
): ArrayBuffer => {
  const [vocab, hidden] = weight.shape;
  const graph: GraphJson = {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["embedding"] },
    symbols: [],
    inputs: [{ name: "index", dtype: "i32", shape: [...index.shape] }],
    outputs: wAsOutput ? ["y", "w"] : ["y"],
    initializers: {
      w: { tensor: "m.w", storage: { dtype: "i4", scale: "m.s", group_size: groupSize } },
    },
    values: {
      w: { dtype: "f32", shape: [vocab, hidden] },
      y: { dtype: "f32", shape: [index.shape[0], hidden] },
    },
    nodes: [{ op: "embedding", ins: ["w", "index"], outs: ["y"], attrs: { padding_idx: -1 } }],
  };
  return buildSafetensors(
    [
      { name: "m.w", dtype: "I4", shape: [vocab, hidden], data: quantized.bytes },
      {
        name: "m.s",
        dtype: "F32",
        shape: [...quantized.scaleShape],
        data: f32Bytes([...quantized.scale]),
      },
    ],
    { karume_ir: JSON.stringify(graph) },
  );
};

Deno.test({
  name: "w=i4 の embedding が CPU 参照（丸め後の表）と一致する（group 2 種・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // 語彙表は `[V,D]`（`channel_rows` が恒等）で、group は D 軸に沿って切られる。1 スレッドが
    // 1 出力要素なので scale は要素ごとに引かれる — 行 / group の取り違えは
    // {@link groupVarying} の振幅差でそのまま値に出る。D は group で割り切れる形だけ
    // （i4 は端数 group を作らない MUST — ADR 0069 決定 2）で、どちらも行内に group が複数入る。
    const cases = [
      { name: "embedding [7,48] g16", vocab: 7, hidden: 48, groupSize: 16, picks: 11 },
      { name: "embedding [5,96] g32", vocab: 5, hidden: 96, groupSize: 32, picks: 9 },
    ] as const;
    const gpu = await acquireGpu();
    try {
      for (const testCase of cases) {
        const weight = fill(
          [testCase.vocab, testCase.hidden],
          groupVarying(testCase.hidden, testCase.groupSize),
        );
        const quantized = quantizeI4(weight.data, weight.shape, testCase.groupSize);
        // 添字は語彙を一巡しない並び（行の取り違えが出る形・同じ行を 2 度引く形も含む）
        const index = fill([testCase.picks], (i) => (i * 3 + 1) % testCase.vocab, "i32");
        const session = await createSession(
          gpu,
          openModel(i4EmbeddingModel(weight, index, quantized, testCase.groupSize)),
        );
        let output: Tensor;
        let residentBytes: number;
        try {
          output = (await session.run({ index }))["y"];
          residentBytes = session.diagnostics().storage.residentCompressedBytes;
        } finally {
          await session.dispose();
        }
        assertEquals(
          residentBytes,
          quantized.bytes.byteLength + quantized.scale.byteLength,
          `${testCase.name}: GPU 常駐圧縮バイト数（scale 込み）`,
        );
        const expected = applyReferenceOp(
          "embedding",
          [refTensor(weight.shape, quantized.values), index as RefTensor],
          { padding_idx: -1 },
          [testCase.picks, testCase.hidden],
        );
        assertEquals(output.shape, expected.shape, testCase.name);
        const report = compareTensors(output, expected);
        assertEquals(report.pass, true, `${testCase.name}: ${formatAllclose(report)}`);
      }
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * GPU の `unpack4xU8` + group scale と CPU の {@link decodeI4} が**全 15 nibble 値 × 語内 8 位置で
 * ビット一致**する（i8 側 `gpu_i8_weights_test.ts` の「全 256 値 × 4 レーン」の i4 鏡像）。
 *
 * 運び方: embedding の weight を `[16, 32]`（g16 = 行あたり 2 group）とし、行 r・列 c の量子化値を
 * `((r + c) % 15) − 7` にする。平坦添字は `r·32 + c` なので、**同じ nibble 値が語内 8 位置すべてに
 * 現れる**（32 は 8 の倍数なので行頭は必ず語境界 — これは i4 の格納制約〈量子化軸が 2 冪 ≥ 16 の
 * group で割り切れる〉から来る不変で、i8 / f16 の「行長が語の倍数でない罠」は i4 には存在しない）。
 * 振幅は行 × group で変えるので、scale 添字の誤りも同時に検出できる。
 */
Deno.test({
  name: "GPU の unpack4xU8 と CPU 展開が全 15 nibble 値 × 語内 8 位置でビット一致（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const vocab = 16;
    const hidden = 32;
    const groupSize = 16;
    const weight = fill([vocab, hidden], (i) => {
      const row = Math.floor(i / hidden);
      const col = i % hidden;
      const q = ((row + col) % 15) - 7;
      // scale が行 × group で違う値になる振幅（同じだと添字の取り違えが値に出ない）
      return q * (0.1 + row * 0.017 + Math.floor(col / groupSize) * 0.31);
    });
    const quantized = quantizeI4(weight.data, weight.shape, groupSize);
    // 恒真化の門: 語内 8 位置それぞれで nibble 値が 15 種すべて現れている
    for (let position = 0; position < 8; position += 1) {
      const seen = new Set<number>();
      for (let i = position; i < vocab * hidden; i += 8) {
        const byte = quantized.bytes[i >> 1];
        seen.add((i & 1) === 1 ? byte >> 4 : byte & 0x0f);
      }
      assertEquals(seen.size, 15, `語内位置 ${position}: nibble 値が 15 種そろっていない`);
    }
    const index = fill([vocab], (i) => i, "i32");
    const gpu = await acquireGpu();
    let actual: Float32Array<ArrayBuffer>;
    try {
      const session = await createSession(
        gpu,
        openModel(i4EmbeddingModel(weight, index, quantized, groupSize)),
      );
      try {
        actual = (await session.run({ index }))["y"].data as Float32Array<ArrayBuffer>;
        assertEquals(
          session.diagnostics().storage.residentCompressedBytes,
          quantized.bytes.byteLength + quantized.scale.byteLength,
        );
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
    const expected = decodeI4(
      quantized.bytes,
      [vocab, hidden],
      quantized.scale,
      quantized.scaleShape,
      groupSize,
    );
    const actualBits = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
    const expectedBits = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
    assertEquals(actualBits, expectedBits, "GPU 展開と CPU 展開がビット一致しない");
  },
});

Deno.test({
  name: "embedding も適格（GPU dequant）と適格外（CPU 展開）でビット一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // 適格外の受け皿は `decodeI4` による `[V,D]` の f32 展開。展開がビット一致なら、同じ
    // gather カーネルが同じバイトを写すので出力は**ビット単位で**一致する。
    const groupSize = 32;
    const weight = fill([9, 64], groupVarying(64, groupSize));
    const quantized = quantizeI4(weight.data, weight.shape, groupSize);
    const index = fill([13], (i) => (i * 5 + 2) % 9, "i32");
    const gpu = await acquireGpu();
    try {
      const run = async (wAsOutput: boolean) => {
        const session = await createSession(
          gpu,
          openModel(i4EmbeddingModel(weight, index, quantized, groupSize, { wAsOutput })),
        );
        try {
          const outputs = await session.run({ index });
          return {
            y: Float32Array.from(outputs["y"].data as Float32Array),
            diagnostics: session.diagnostics().storage,
          };
        } finally {
          await session.dispose();
        }
      };
      const resident = await run(false);
      const expanded = await run(true);
      assert(resident.diagnostics.residentCompressedBytes > 0, "適格側が常駐していない");
      assertEquals(expanded.diagnostics.residentCompressedBytes, 0, "適格外側が常駐している");
      assert(expanded.diagnostics.hostExpandedBytes > 0, "適格外側が CPU 展開されていない");
      assertEquals(resident.y, expanded.y, "GPU dequant と CPU 展開の出力がビット一致しない");
    } finally {
      gpu.destroy();
    }
  },
});

// ---------------------------------------------------------------------------
// 適格判定（i4 の狭め — GPU 非依存）
// ---------------------------------------------------------------------------

Deno.test("i4 の適格は linear / embedding の重みスロット限定（conv 系は落ちる）", () => {
  // 展開経路（`unpack4xU8` + group scale）を持つカーネルは linear と embedding だけ。conv 系にも
  // 食われる重みを常駐させると、そちらのカーネルが packed バイトを f32 として読む（例外は
  // 出ない沈黙誤値）。ここは**適格判定だけ**を見るので、shape 契約の通らない組でも成立する
  // 形（同じ実体を rank 2 と rank 3 の両方の重みスロットに置く形）を直接与える。
  const eligible = (nodes: readonly unknown[]): readonly string[] =>
    [
      ...i4EligibleInitializers(
        { initializers: { w: { tensor: "m.w" } }, nodes } as never,
      ),
    ].sort();
  assertEquals(eligible([{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }]), ["w"]);
  assertEquals(eligible([{ op: "embedding", ins: ["w", "x"], outs: ["y"], attrs: {} }]), ["w"]);
  // linear と embedding の両方で食われる形（tied lm_head）はどちらも展開経路を持つので適格
  assertEquals(
    eligible([
      { op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} },
      { op: "embedding", ins: ["w", "x"], outs: ["z"], attrs: {} },
    ]),
    ["w"],
  );
  for (const op of ["conv1d", "conv2d", "conv_transpose1d"]) {
    assertEquals(eligible([{ op, ins: ["x", "w", "b"], outs: ["y"], attrs: {} }]), [], op);
    assertEquals(
      eligible([
        { op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} },
        { op, ins: ["x", "w", "b"], outs: ["z"], attrs: {} },
      ]),
      [],
      `linear と ${op} の共有`,
    );
  }
  // 重みスロット**以外**の消費はここでは見ない（`eligibleCompressedInitializers` との積で使う）
  assertEquals(eligible([{ op: "add", ins: ["w", "w"], outs: ["y"], attrs: {} }]), []);
});

Deno.test({
  name: "i4 常駐の重み × linearCompute 'f16'（w4a16）は fail loudly（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const testCase = CASES[1];
    const quantized = quantizeI4(testCase.weight.data, testCase.weight.shape, testCase.groupSize);
    // f16 計算変種は device 作成時の 'shader-f16' が前提（無いと別の門で落ちて w4a16 の
    // 門に届かない）。
    const gpu = await acquireGpu({ shaderF16: true });
    try {
      const session = await createSession(
        gpu,
        openModel(i4LinearModel(testCase, quantized)),
        { linearCompute: "f16" },
      );
      try {
        await assertRejects(
          () => session.run({ x: testCase.x }),
          Error,
          "w4a16",
        );
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "linearCompute 'i8a8' でも i4 常駐は通常の f32 計算経路で正しく走る（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // i8a8 は「i8 常駐」を条件に含む opt-in（ADR 0025 系）— i4 常駐は f32 / f16 常駐と同じく
    // 通常経路で走る（縮退ではなく i4 の実装済み経路そのもの — 挙動をここで固定する）。
    const testCase = CASES[1];
    const quantized = quantizeI4(testCase.weight.data, testCase.weight.shape, testCase.groupSize);
    const gpu = await acquireGpu();
    try {
      const session = await createSession(
        gpu,
        openModel(i4LinearModel(testCase, quantized)),
        { linearCompute: "i8a8" },
      );
      try {
        const output = (await session.run({ x: testCase.x }))["y"];
        const report = compareTensors(output, expectedLinear(testCase, quantized));
        assertEquals(report.pass, true, formatAllclose(report));
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
});
