/**
 * コンテナ経路（`openContainer` → `createSessionFromContainer`）と旧 safetensors 経路（`openModel` →
 * `createSession`）が**同じ重みバイト列**から**同じ出力バイト列**を出すこと（ADR 0108 段 1 検収③の
 * 縮図）。片方だけが CPU 展開 / GPU 常駐へ倒れると、診断の常駐バイト数で差が出る。
 *
 * 合成モデル: linear（i4 + group scale）→ add（f16 重み）→ linear（i8 + per-channel scale・bias f32）。
 * 3 codec の供給計画・scale の正規化（rank 2）・piece 分割（block 上限を下げて i8 の重みを割る）を
 * 1 本で通す。
 */

import { assert, assertEquals, assertRejects } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { openModel } from "../src/format/container.ts";
import { ContainerFormatError } from "../src/format/container/header.ts";
import { openContainer } from "../src/format/container/open.ts";
import { type IrDeclaration, parseIrDeclaration } from "../src/format/ir.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import {
  createSession,
  createSessionFromContainer,
  prepareContainer,
} from "../src/runtime/executor.ts";
import { estimateSessionMemory } from "../src/runtime/estimate.ts";
import { buildSafetensors, f32Bytes, type TensorSpec } from "./helpers/format.ts";
import { fill } from "./helpers/graph.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { quantizeI4 } from "./helpers/i4.ts";
import { quantizeI8 } from "./helpers/i8.ts";
import { writeGraphContainer, writeModelContainer } from "./helpers/container-write.ts";

const M = 4;
const K = 32;
const H = 24;
const N = 8;

const f16Bytes = (values: ArrayLike<number>): Uint8Array<ArrayBuffer> => {
  const out = new Uint8Array(new ArrayBuffer(values.length * 2));
  const view = new DataView(out.buffer);
  for (let i = 0; i < values.length; i += 1) view.setFloat16(i * 2, values[i], true);
  return out;
};

/** 決定的な重み（乱数は使わない）。 */
const weights = () => ({
  w1: fill([H, K], (i) => Math.sin(i * 0.37) * 0.5),
  b1: fill([H], (i) => (i % 3) * 0.05),
  h: fill([H], (i) => (i % 5) * 0.125 - 0.25),
  w2: fill([N, H], (i) => Math.cos(i * 0.91) * 0.75),
  b2: fill([N], (i) => i * 0.01),
});

const graphJson = (version: 1 | 2) => ({
  format: "karume-ir",
  version,
  requires: { ops: ["add", "linear"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [M, K] }],
  outputs: ["y"],
  values: {
    "enc.w1": { dtype: "f32", shape: [H, K] },
    "enc.b1": { dtype: "f32", shape: [H] },
    "enc.h": { dtype: "f32", shape: [H] },
    "enc.w2": { dtype: "f32", shape: [N, H] },
    "enc.b2": { dtype: "f32", shape: [N] },
    t: { dtype: "f32", shape: [M, H] },
    u: { dtype: "f32", shape: [M, H] },
    y: { dtype: "f32", shape: [M, N] },
  },
  nodes: [
    { op: "linear", ins: ["x", "enc.w1", "enc.b1"], outs: ["t"], attrs: {} },
    { op: "add", ins: ["t", "enc.h"], outs: ["u"], attrs: {} },
    { op: "linear", ins: ["u", "enc.w2", "enc.b2"], outs: ["y"], attrs: {} },
  ],
});

const declaration = (): IrDeclaration =>
  parseIrDeclaration(JSON.stringify({
    ...graphJson(2),
    initializers: { "enc.w1": {}, "enc.b1": {}, "enc.h": {}, "enc.w2": {}, "enc.b2": {} },
  }));

type Quantized = {
  readonly w1: ReturnType<typeof quantizeI4>;
  readonly b1: Uint8Array<ArrayBuffer>;
  readonly h16: Uint8Array<ArrayBuffer>;
  readonly w2: ReturnType<typeof quantizeI8>;
  readonly b2: Uint8Array<ArrayBuffer>;
};

const quantize = (): Quantized => {
  const w = weights();
  return {
    w1: quantizeI4(w.w1.data, [H, K], 16),
    b1: f32Bytes(w.b1.data),
    h16: f16Bytes(w.h.data),
    w2: quantizeI8(w.w2.data, [N, H], 0),
    b2: f32Bytes(w.b2.data),
  };
};

/** 旧配布形（v1 JSON + safetensors 1 本）。 */
const legacyModel = (q: Quantized): ArrayBuffer => {
  const graph = {
    ...graphJson(1),
    initializers: {
      w1: { tensor: "enc.w1", storage: { dtype: "i4", scale: "enc.w1_scale", group_size: 16 } },
      b1: { tensor: "enc.b1", storage: { dtype: "f32" } },
      h: { tensor: "enc.h", storage: { dtype: "f16" } },
      w2: { tensor: "enc.w2", storage: { dtype: "i8", scale: "enc.w2_scale" } },
      b2: { tensor: "enc.b2", storage: { dtype: "f32" } },
    },
    // v1 では initializer 名が placeholder 名で、実体キーが tensor 欄 — 読み手が改名する。
    values: {
      w1: { dtype: "f32", shape: [H, K] },
      b1: { dtype: "f32", shape: [H] },
      h: { dtype: "f32", shape: [H] },
      w2: { dtype: "f32", shape: [N, H] },
      b2: { dtype: "f32", shape: [N] },
      t: { dtype: "f32", shape: [M, H] },
      u: { dtype: "f32", shape: [M, H] },
      y: { dtype: "f32", shape: [M, N] },
    },
    nodes: [
      { op: "linear", ins: ["x", "w1", "b1"], outs: ["t"], attrs: {} },
      { op: "add", ins: ["t", "h"], outs: ["u"], attrs: {} },
      { op: "linear", ins: ["u", "w2", "b2"], outs: ["y"], attrs: {} },
    ],
  };
  const tensors: TensorSpec[] = [
    { name: "enc.w1", dtype: "I4", shape: [H, K], data: q.w1.bytes },
    { name: "enc.w1_scale", dtype: "F32", shape: [...q.w1.scaleShape], data: f32Bytes(q.w1.scale) },
    { name: "enc.b1", dtype: "F32", shape: [H], data: q.b1 },
    { name: "enc.h", dtype: "F16", shape: [H], data: q.h16 },
    { name: "enc.w2", dtype: "I8", shape: [N, H], data: q.w2.bytes },
    { name: "enc.w2_scale", dtype: "F32", shape: [...q.w2.scaleShape], data: f32Bytes(q.w2.scale) },
    { name: "enc.b2", dtype: "F32", shape: [N], data: q.b2 },
  ];
  return buildSafetensors(tensors, { karume_ir: JSON.stringify(graph) });
};

/** 新配布形（krm）。block 上限を 128 B に下げて i8 の重み（192 B）を piece 分割させる。 */
const containerInput = (q: Quantized) => ({
  graphs: { main: declaration() },
  consts: [],
  weights: [
    {
      graph: "main",
      initializer: "enc.w1",
      bytes: q.w1.bytes,
      encoding: {
        codec: "int4-sym-g" as const,
        groupSize: 16,
        scale: { bytes: f32Bytes(q.w1.scale), dtype: "f32" as const },
      },
    },
    { graph: "main", initializer: "enc.b1", bytes: q.b1, encoding: { codec: "f32" as const } },
    { graph: "main", initializer: "enc.h", bytes: q.h16, encoding: { codec: "f16" as const } },
    {
      graph: "main",
      initializer: "enc.w2",
      bytes: q.w2.bytes,
      encoding: {
        codec: "int8-sym" as const,
        groupSize: H,
        scale: { bytes: f32Bytes(q.w2.scale), dtype: "f32" as const },
      },
    },
    { graph: "main", initializer: "enc.b2", bytes: q.b2, encoding: { codec: "f32" as const } },
  ],
  assets: [],
  provenance: { license: "test" },
});

const OPTIONS = { partBytes: 1024, blockBytes: 128 } as const;

describe("container session", { ignore: !GPU_AVAILABLE }, () => {
  it("krm から作った Session は旧 safetensors 経路と出力バイト列・常駐バイト数が一致する", async () => {
    const q = quantize();
    const written = await writeModelContainer(containerInput(q), OPTIONS);
    const gpu = await acquireGpu();
    try {
      const x = fill([M, K], (i) => ((i * 7) % 11) / 11 - 0.5);
      const legacy = await createSession(gpu, openModel(legacyModel(q)));
      let expected: Float32Array<ArrayBuffer>;
      let expectedResident: number;
      try {
        expected = (await legacy.run({ x }))["y"].data as Float32Array<ArrayBuffer>;
        expectedResident = legacy.diagnostics().storage.residentCompressedBytes;
      } finally {
        await legacy.dispose();
      }
      const opened = await openContainer({ kind: "parts", parts: written.parts });
      // i8 の重み（192 B）は block 上限 128 B で 2 piece に割れている。
      assertEquals(opened.graphs.main.supplies.get("enc.w2")?.blocks.length, 2);
      const session = await createSessionFromContainer(gpu, opened, "main");
      try {
        const actual = (await session.run({ x }))["y"].data as Float32Array<ArrayBuffer>;
        assertEquals(new Uint8Array(actual.buffer), new Uint8Array(expected.buffer));
        assertEquals(session.diagnostics().storage.residentCompressedBytes, expectedResident);
        assert(
          expectedResident > 0,
          "圧縮のまま常駐した重みが無い（i4 / f16 / i8 の席が効いていない）",
        );
      } finally {
        await session.dispose();
      }
      // 見積りも同じ（席の分類が経路で割れていない）。
      const prepared = prepareContainer(opened, "main");
      assertEquals(prepared.estimate(), estimateSessionMemory(openModel(legacyModel(q))));
    } finally {
      gpu.destroy();
    }
  });

  it("krg（重みの供給が無い）からは Session を組めない", async () => {
    const q = quantize();
    const graphOnly = await writeGraphContainer(containerInput(q), OPTIONS);
    const opened = await openContainer({ kind: "bytes", bytes: graphOnly.bytes });
    const gpu = await acquireGpu();
    try {
      await assertRejects(
        () => createSessionFromContainer(gpu, opened, "main"),
        ContainerFormatError,
        "重みの供給が無い",
      );
    } finally {
      gpu.destroy();
    }
  });
});
