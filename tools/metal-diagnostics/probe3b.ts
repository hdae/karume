// probe3b — karume runtime・実バイト梯子。
//
// 「最初に NaN を作る op」を実バイト × 実形状で挟み撃ちにする。gemma4 製品グラフ
// （models/karume-gemma4-e2b）から**実際の重みバイトを切り出して**最小グラフを組み、
// 段ごとに実 GPU 実行 → TS 参照と突合する。段が上がるほど製品に近づく。
//
//   ① embedding : 実 model.lm_head.weight（i8 [262144,1536] 384MiB + per-row scale）
//                 × ids [2,105,2364,107] → ホストの f32(i8)·scale とビット一致を見る
//   ② linear i4 : 実 layers.0.mlp.{gate,up}_proj（i4 g32 [6144,1536]）× 合成入力
//                 → decodeI4 + CPU 参照 linear
//   ③ rms_norm  : 実 layers.0.input_layernorm.weight（f32 [1536]）× 合成入力 → CPU 参照
//   ④ states 形 attention: **製品の実形状**（D=256 W=512 C=512 / D=512 full C=640・
//                 GQA r=8・prefill M=32 と decode M=1）を合成データで踏む
//                 — 合成テストの上限は D=32 / C=64 / W=8 なので、この帯は未踏
//   ⑤ linear i8 GEMV: 実 lm_head を **製品どおり N=262144 · K=1536 · M=1** で撃つ
//                 （製品グラフで logits を実際に書く最後の linear）→ 抽出行の CPU 参照
//   ⑥ 混在消費 : 同一の i8 lm_head を embedding と linear が**同時に**消費する形
//                 （製品グラフと同じ — 適格判定と常駐が 1 本を共有する）
//
// MUST: 各段の判定は「非有限が 0 本」を第一門にする（この probe が探しているのは NaN の
// 発生点であって tolerance ではない）。数値一致は第二門。
//
// 使い方: リポルートで `deno run -A <このファイル> [--only 1,2,5]`

import { decodeI4 } from "../../packages/runtime/src/format/i4.ts";
import { decodeI8 } from "../../packages/runtime/src/format/i8.ts";
import { openModel } from "../../packages/runtime/src/format/container.ts";
import { acquireGpu } from "../../packages/runtime/src/gpu/device.ts";
import { applyReferenceOp, refTensor } from "../../packages/runtime/src/reference/ops.ts";
import { referenceStateAttention } from "../../packages/runtime/src/reference/state-attention.ts";
import { createSession } from "../../packages/runtime/src/runtime/executor.ts";
import type { GraphJson, TensorSpec } from "../../packages/runtime/tests/helpers/format.ts";
import { buildSafetensors } from "../../packages/runtime/tests/helpers/format.ts";
import {
  caseColCap,
  halfScale,
  runStateAttention,
  type StateCase,
} from "../../packages/runtime/tests/helpers/state-dispatch.ts";

// ---------------------------------------------------------------------------
// 実資産の切り出し
// ---------------------------------------------------------------------------

const MIRROR = "models/karume-gemma4-e2b/e2b/model";
const SHARDS = [
  `${MIRROR}/model.i4-00001-of-00003.safetensors`,
  `${MIRROR}/model.i4-00002-of-00003.safetensors`,
  `${MIRROR}/model.i4-00003-of-00003.safetensors`,
];

type Located = {
  readonly path: string;
  readonly dtype: string;
  readonly shape: number[];
  readonly start: number;
  readonly end: number;
};

const index = new Map<string, Located>();

const buildIndex = async (): Promise<void> => {
  for (const path of SHARDS) {
    const file = await Deno.open(path);
    let headerLength: number;
    try {
      const head = new Uint8Array(8);
      await file.read(head);
      headerLength = Number(new DataView(head.buffer).getBigUint64(0, true));
      const body = new Uint8Array(headerLength);
      let filled = 0;
      while (filled < headerLength) {
        const read = await file.read(body.subarray(filled));
        if (read === null) throw new Error("ヘッダが尽きた");
        filled += read;
      }
      const header = JSON.parse(new TextDecoder().decode(body)) as Record<string, {
        dtype: string;
        shape: number[];
        data_offsets: [number, number];
      }>;
      for (const [name, value] of Object.entries(header)) {
        if (name === "__metadata__") continue;
        index.set(name, {
          path,
          dtype: value.dtype,
          shape: value.shape,
          start: 8 + headerLength + value.data_offsets[0],
          end: 8 + headerLength + value.data_offsets[1],
        });
      }
    } finally {
      file.close();
    }
  }
};

/** 実バイトを 1 本だけ読む（1.5GiB のミラー全体はメモリに載せない）。 */
const readTensor = async (
  name: string,
): Promise<{ located: Located; bytes: Uint8Array<ArrayBuffer> }> => {
  const located = index.get(name);
  if (located === undefined) throw new Error(`テンソル '${name}' がミラーに無い`);
  const file = await Deno.open(located.path);
  try {
    await file.seek(located.start, Deno.SeekMode.Start);
    const bytes = new Uint8Array(located.end - located.start);
    let filled = 0;
    while (filled < bytes.length) {
      const read = await file.read(bytes.subarray(filled));
      if (read === null) throw new Error(`'${name}' が途中で尽きた`);
      filled += read;
    }
    return { located, bytes };
  } finally {
    file.close();
  }
};

const readF32 = async (
  name: string,
): Promise<{ located: Located; data: Float32Array<ArrayBuffer> }> => {
  const { located, bytes } = await readTensor(name);
  if (located.dtype !== "F32") throw new Error(`'${name}' は F32 ではない（${located.dtype}）`);
  return { located, data: new Float32Array(bytes.buffer, 0, bytes.byteLength / 4) };
};

// ---------------------------------------------------------------------------
// 突合（第一門 = 非有限・第二門 = 数値）
// ---------------------------------------------------------------------------

type Verdict = {
  readonly stage: string;
  readonly detail: string;
  readonly nan: number;
  readonly inf: number;
  readonly maxAbs: number;
  readonly maxRel: number;
  readonly exact: number;
  readonly total: number;
  readonly pass: boolean;
};

const verdicts: Verdict[] = [];

const judge = (
  stage: string,
  detail: string,
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  atol: number,
  rtol: number,
): void => {
  let nan = 0;
  let inf = 0;
  let maxAbs = 0;
  let maxRel = 0;
  let exact = 0;
  let fail = 0;
  for (let i = 0; i < expected.length; i += 1) {
    const a = actual[i];
    const e = expected[i];
    if (Number.isNaN(a)) {
      nan += 1;
      continue;
    }
    if (!Number.isFinite(a)) {
      inf += 1;
      continue;
    }
    if (a === e) exact += 1;
    const abs = Math.abs(a - e);
    const rel = Math.abs(e) > 0 ? abs / Math.abs(e) : abs;
    if (abs > maxAbs) maxAbs = abs;
    if (rel > maxRel) maxRel = rel;
    if (abs > atol + rtol * Math.abs(e)) fail += 1;
  }
  const pass = nan === 0 && inf === 0 && fail === 0;
  verdicts.push({
    stage,
    detail: `${detail} fail=${fail}`,
    nan,
    inf,
    maxAbs,
    maxRel,
    exact,
    total: expected.length,
    pass,
  });
};

/** 参照を持たない大きい出力（④ の一部など）の非有限だけを見る門。 */
const judgeFiniteOnly = (stage: string, detail: string, actual: ArrayLike<number>): void => {
  let nan = 0;
  let inf = 0;
  for (let i = 0; i < actual.length; i += 1) {
    if (Number.isNaN(actual[i])) nan += 1;
    else if (!Number.isFinite(actual[i])) inf += 1;
  }
  verdicts.push({
    stage,
    detail,
    nan,
    inf,
    maxAbs: Number.NaN,
    maxRel: Number.NaN,
    exact: 0,
    total: actual.length,
    pass: nan === 0 && inf === 0,
  });
};

const model = (graph: GraphJson, tensors: readonly TensorSpec[]): ArrayBuffer =>
  buildSafetensors(tensors, { karume_ir: JSON.stringify(graph) });

// 決定的な合成入力（乱数は使わない — 失敗が再現しないため）
const SIGNED = (i: number): number => (((i * 7) % 23) - 11) * 0.11;
const seededF32 = (n: number, f: (i: number) => number): Float32Array<ArrayBuffer> => {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 1) out[i] = f(i);
  return out;
};

// ---------------------------------------------------------------------------

const only = (() => {
  const at = Deno.args.indexOf("--only");
  if (at < 0) return undefined;
  return new Set(Deno.args[at + 1].split(",").map((s) => s.trim()));
})();
const wants = (stage: string): boolean => only === undefined || only.has(stage);

await buildIndex();
console.log(`[probe3b] ミラーのテンソル索引 ${index.size} 本`);

const gpu = await acquireGpu();
console.log(
  `[probe3b] device limits: maxBufferSize=${gpu.limits.maxBufferSize} ` +
    `maxStorageBufferBindingSize=${gpu.limits.maxStorageBufferBindingSize} ` +
    `maxStorageBuffersPerShaderStage=${gpu.limits.maxStorageBuffersPerShaderStage} ` +
    `maxComputeWorkgroupStorageSize=${gpu.limits.maxComputeWorkgroupStorageSize} ` +
    `maxComputeWorkgroupsPerDimension=${gpu.limits.maxComputeWorkgroupsPerDimension}`,
);

const IDS = Int32Array.from([2, 105, 2364, 107]);
const LM_HEAD = "model.lm_head.weight";
const LM_SCALE = "karume.scale.model.lm_head.weight";

// ── ① embedding（実 i8 lm_head 384MiB） ──────────────────────────────────────
if (wants("1") || wants("5") || wants("6")) {
  const w = await readTensor(LM_HEAD);
  const s = await readF32(LM_SCALE);
  const [vocab, dim] = w.located.shape;
  console.log(
    `[probe3b] lm_head ${w.located.dtype} [${vocab},${dim}] ${w.bytes.byteLength} bytes 読了`,
  );

  // 期待値は 4 行だけ手で組む（262144×1536 の f32 展開は 1.6GiB になるので全展開しない）。
  // decodeI8 の仕様（q·scale を f32 へ 1 度だけ丸める）をこちらで書き下す。
  const expected = new Float32Array(IDS.length * dim);
  const q = new Int8Array(w.bytes.buffer, w.bytes.byteOffset, w.bytes.byteLength);
  for (const [row, id] of IDS.entries()) {
    const scale = s.data[id];
    for (let d = 0; d < dim; d += 1) {
      expected[row * dim + d] = Math.fround(q[id * dim + d] * scale);
    }
  }
  // decodeI8 の鏡像であることを 1 行で確かめる（書き下しが本番実装と割れていないこと）
  const sample = decodeI8(
    w.bytes.slice(IDS[0] * dim, (IDS[0] + 1) * dim),
    [1, dim],
    Float32Array.from([s.data[IDS[0]]]),
    [1, 1],
  );
  for (let d = 0; d < dim; d += 1) {
    if (sample[d] !== expected[d]) throw new Error(`① 期待値の書き下しが decodeI8 と割れた d=${d}`);
  }

  if (wants("1")) {
    const graph: GraphJson = {
      format: "karume-ir",
      version: 1,
      requires: { ops: ["embedding"] },
      symbols: [],
      inputs: [{ name: "ids", dtype: "i32", shape: [IDS.length] }],
      outputs: ["y"],
      initializers: { w: { tensor: "m.w", storage: { dtype: "i8", scale: "m.s" } } },
      values: {
        w: { dtype: "f32", shape: [vocab, dim] },
        y: { dtype: "f32", shape: [IDS.length, dim] },
      },
      nodes: [{ op: "embedding", ins: ["w", "ids"], outs: ["y"], attrs: { padding_idx: -1 } }],
    };
    const buffer = model(graph, [
      { name: "m.w", dtype: "I8", shape: [vocab, dim], data: w.bytes },
      {
        name: "m.s",
        dtype: "F32",
        shape: [vocab, 1],
        data: new Uint8Array(s.data.buffer.slice(0)),
      },
    ]);
    const session = await createSession(gpu, openModel(buffer));
    try {
      const out =
        (await session.run({ ids: { dtype: "i32", shape: [IDS.length], data: IDS } }))["y"];
      judge(
        "①embedding-i8-real",
        `[${IDS.length},${dim}] ids=${[...IDS]}`,
        out.data,
        expected,
        0,
        0,
      );
    } finally {
      await session.dispose();
    }
  }

  // ── ⑥ 混在消費（embedding と linear が同一の i8 テーブルを共有 — 製品と同形） ─────
  if (wants("6")) {
    const x = seededF32(dim, SIGNED);
    const bias = new Float32Array(vocab);
    const graph: GraphJson = {
      format: "karume-ir",
      version: 1,
      requires: { ops: ["embedding", "linear"] },
      symbols: [],
      inputs: [
        { name: "ids", dtype: "i32", shape: [IDS.length] },
        { name: "x", dtype: "f32", shape: [1, dim] },
      ],
      outputs: ["y", "z"],
      initializers: {
        w: { tensor: "m.w", storage: { dtype: "i8", scale: "m.s" } },
        b: { tensor: "m.b", storage: { dtype: "f32" } },
      },
      values: {
        w: { dtype: "f32", shape: [vocab, dim] },
        b: { dtype: "f32", shape: [vocab] },
        y: { dtype: "f32", shape: [IDS.length, dim] },
        z: { dtype: "f32", shape: [1, vocab] },
      },
      nodes: [
        { op: "embedding", ins: ["w", "ids"], outs: ["y"], attrs: { padding_idx: -1 } },
        { op: "linear", ins: ["x", "w", "b"], outs: ["z"], attrs: {} },
      ],
    };
    const buffer = model(graph, [
      { name: "m.w", dtype: "I8", shape: [vocab, dim], data: w.bytes },
      {
        name: "m.s",
        dtype: "F32",
        shape: [vocab, 1],
        data: new Uint8Array(s.data.buffer.slice(0)),
      },
      { name: "m.b", dtype: "F32", shape: [vocab], data: new Uint8Array(bias.buffer) },
    ]);
    const session = await createSession(gpu, openModel(buffer));
    try {
      const outputs = await session.run({
        ids: { dtype: "i32", shape: [IDS.length], data: IDS },
        x: { dtype: "f32", shape: [1, dim], data: x },
      });
      judge("⑥mixed-embedding", `[${IDS.length},${dim}]`, outputs["y"].data, expected, 0, 0);
      judgeFiniteOnly("⑥mixed-linear", `[1,${vocab}] 非有限のみ`, outputs["z"].data);
    } finally {
      await session.dispose();
    }
  }

  // ── ⑤ linear i8 GEMV（製品の logits を実際に書く最後の linear） ────────────────
  if (wants("5")) {
    const x = seededF32(dim, SIGNED);
    const bias = new Float32Array(vocab);
    const graph: GraphJson = {
      format: "karume-ir",
      version: 1,
      requires: { ops: ["linear"] },
      symbols: [],
      inputs: [{ name: "x", dtype: "f32", shape: [1, 1, dim] }],
      outputs: ["y"],
      initializers: {
        w: { tensor: "m.w", storage: { dtype: "i8", scale: "m.s" } },
        b: { tensor: "m.b", storage: { dtype: "f32" } },
      },
      values: {
        w: { dtype: "f32", shape: [vocab, dim] },
        b: { dtype: "f32", shape: [vocab] },
        y: { dtype: "f32", shape: [1, 1, vocab] },
      },
      nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
    };
    const buffer = model(graph, [
      { name: "m.w", dtype: "I8", shape: [vocab, dim], data: w.bytes },
      {
        name: "m.s",
        dtype: "F32",
        shape: [vocab, 1],
        data: new Uint8Array(s.data.buffer.slice(0)),
      },
      { name: "m.b", dtype: "F32", shape: [vocab], data: new Uint8Array(bias.buffer) },
    ]);
    const session = await createSession(gpu, openModel(buffer));
    try {
      const out = (await session.run({ x: { dtype: "f32", shape: [1, 1, dim], data: x } }))["y"];
      const actual = out.data as Float32Array;
      judgeFiniteOnly("⑤linear-i8-gemv(全域)", `[1,1,${vocab}] 非有限のみ`, actual);
      // 抽出行だけ CPU 参照（全域の f32 展開は 1.6GiB になるため）
      const picks: number[] = [];
      for (let n = 0; n < vocab; n += Math.floor(vocab / 512)) picks.push(n);
      const sampledActual = new Float32Array(picks.length);
      const sampledExpected = new Float32Array(picks.length);
      for (const [j, n] of picks.entries()) {
        const scale = s.data[n];
        let acc = 0;
        for (let k = 0; k < dim; k += 1) acc += Math.fround(q[n * dim + k] * scale) * x[k];
        sampledExpected[j] = acc;
        sampledActual[j] = actual[n];
      }
      judge(
        "⑤linear-i8-gemv(抽出)",
        `${picks.length} 行`,
        sampledActual,
        sampledExpected,
        1e-3,
        1e-4,
      );
    } finally {
      await session.dispose();
    }
  }
}

// ── ② linear i4（実 layer0 gate/up） ─────────────────────────────────────────
if (wants("2")) {
  for (const key of ["gate_proj", "up_proj"] as const) {
    const name = `model.model.layers.0.mlp.${key}.weight`;
    const w = await readTensor(name);
    const s = await readF32(`karume.scale.${name}`);
    const [outDim, inDim] = w.located.shape;
    const groupSize = inDim / s.located.shape[1];
    const weights = decodeI4(w.bytes, [outDim, inDim], s.data, s.located.shape, groupSize);
    const rows = 4;
    const x = seededF32(rows * inDim, SIGNED);
    const bias = new Float32Array(outDim);
    const graph: GraphJson = {
      format: "karume-ir",
      version: 1,
      requires: { ops: ["linear"] },
      symbols: [],
      inputs: [{ name: "x", dtype: "f32", shape: [rows, inDim] }],
      outputs: ["y"],
      initializers: {
        w: { tensor: "m.w", storage: { dtype: "i4", scale: "m.s", group_size: groupSize } },
        b: { tensor: "m.b", storage: { dtype: "f32" } },
      },
      values: {
        w: { dtype: "f32", shape: [outDim, inDim] },
        b: { dtype: "f32", shape: [outDim] },
        y: { dtype: "f32", shape: [rows, outDim] },
      },
      nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
    };
    const buffer = model(graph, [
      { name: "m.w", dtype: "I4", shape: [outDim, inDim], data: w.bytes },
      {
        name: "m.s",
        dtype: "F32",
        shape: [...s.located.shape],
        data: new Uint8Array(s.data.buffer.slice(0)),
      },
      { name: "m.b", dtype: "F32", shape: [outDim], data: new Uint8Array(bias.buffer) },
    ]);
    const expected = applyReferenceOp(
      "linear",
      [
        refTensor([rows, inDim], x),
        refTensor([outDim, inDim], weights),
        refTensor([outDim], bias),
      ],
      {},
      [rows, outDim],
    );
    const session = await createSession(gpu, openModel(buffer));
    try {
      const out = (await session.run({ x: { dtype: "f32", shape: [rows, inDim], data: x } }))["y"];
      judge(
        `②linear-i4-${key}`,
        `[${rows},${inDim}]×[${outDim},${inDim}] g${groupSize}`,
        out.data,
        expected.data,
        2e-4,
        1e-5,
      );
    } finally {
      await session.dispose();
    }
  }

  // decode 形（M=1 = linear_gemv 族が走る形）も踏む
  const name = "model.model.layers.0.mlp.gate_proj.weight";
  const w = await readTensor(name);
  const s = await readF32(`karume.scale.${name}`);
  const [outDim, inDim] = w.located.shape;
  const groupSize = inDim / s.located.shape[1];
  const weights = decodeI4(w.bytes, [outDim, inDim], s.data, s.located.shape, groupSize);
  const x = seededF32(inDim, SIGNED);
  const bias = new Float32Array(outDim);
  const graph: GraphJson = {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [1, 1, inDim] }],
    outputs: ["y"],
    initializers: {
      w: { tensor: "m.w", storage: { dtype: "i4", scale: "m.s", group_size: groupSize } },
      b: { tensor: "m.b", storage: { dtype: "f32" } },
    },
    values: {
      w: { dtype: "f32", shape: [outDim, inDim] },
      b: { dtype: "f32", shape: [outDim] },
      y: { dtype: "f32", shape: [1, 1, outDim] },
    },
    nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
  };
  const buffer = model(graph, [
    { name: "m.w", dtype: "I4", shape: [outDim, inDim], data: w.bytes },
    {
      name: "m.s",
      dtype: "F32",
      shape: [...s.located.shape],
      data: new Uint8Array(s.data.buffer.slice(0)),
    },
    { name: "m.b", dtype: "F32", shape: [outDim], data: new Uint8Array(bias.buffer) },
  ]);
  const expected = applyReferenceOp(
    "linear",
    [refTensor([1, inDim], x), refTensor([outDim, inDim], weights), refTensor([outDim], bias)],
    {},
    [1, outDim],
  );
  const session = await createSession(gpu, openModel(buffer));
  try {
    const out = (await session.run({ x: { dtype: "f32", shape: [1, 1, inDim], data: x } }))["y"];
    judge(
      "②linear-i4-gemv(M=1)",
      `[1,${inDim}]×[${outDim},${inDim}] g${groupSize}`,
      out.data,
      expected.data,
      2e-4,
      1e-5,
    );
  } finally {
    await session.dispose();
  }
}

// ── ③ rms_norm（実 gamma） ───────────────────────────────────────────────────
if (wants("3")) {
  for (
    const name of [
      "model.model.layers.0.input_layernorm.weight",
      "model.model.layers.0.self_attn.q_norm.weight",
      "model.model.norm.weight",
    ]
  ) {
    const g = await readF32(name);
    const dim = g.located.shape[0];
    const rows = 32;
    const x = seededF32(rows * dim, SIGNED);
    const eps = 1e-6;
    const graph: GraphJson = {
      format: "karume-ir",
      version: 1,
      requires: { ops: ["rms_norm"] },
      symbols: [],
      inputs: [{ name: "x", dtype: "f32", shape: [rows, dim] }],
      outputs: ["y"],
      initializers: { g: { tensor: "m.g", storage: { dtype: "f32" } } },
      values: {
        g: { dtype: "f32", shape: [dim] },
        y: { dtype: "f32", shape: [rows, dim] },
      },
      nodes: [{ op: "rms_norm", ins: ["x", "g"], outs: ["y"], attrs: { eps } }],
    };
    const buffer = model(graph, [
      { name: "m.g", dtype: "F32", shape: [dim], data: new Uint8Array(g.data.buffer.slice(0)) },
    ]);
    const expected = applyReferenceOp(
      "rms_norm",
      [refTensor([rows, dim], x), refTensor([dim], g.data)],
      { eps },
      [rows, dim],
    );
    const session = await createSession(gpu, openModel(buffer));
    try {
      const out = (await session.run({ x: { dtype: "f32", shape: [rows, dim], data: x } }))["y"];
      judge(
        `③rms_norm ${name.split(".").slice(-2).join(".")}`,
        `[${rows},${dim}]`,
        out.data,
        expected.data,
        1e-5,
        1e-5,
      );
    } finally {
      await session.dispose();
    }
  }
}

// ── ④ states 形 attention を製品の実形状で踏む ───────────────────────────────
if (wants("4")) {
  // 製品の実形状（IR 実測）: 全 35 層が states 形・B=1 H=8 Hkv=1（GQA r=8）。
  // sliding 層 = D 256 / W 512 / C 512、full 層 = D 512 / W 0 / C 640（pipelineConfig.capacity）。
  // prefill は chunkLength 32、decode は M=1。
  const PRODUCT_CASES: readonly StateCase[] = [
    {
      name: "sliding D256 W512 C512 prefill",
      batch: 1,
      heads: 8,
      kvHeads: 1,
      chunkRows: 32,
      depth: 256,
      capacity: 512,
      window: 512,
      past: 600,
      query: 32,
    },
    {
      name: "sliding D256 W512 C512 decode",
      batch: 1,
      heads: 8,
      kvHeads: 1,
      chunkRows: 1,
      depth: 256,
      capacity: 512,
      window: 512,
      past: 613,
      query: 1,
    },
    {
      name: "sliding D256 W512 C512 P0",
      batch: 1,
      heads: 8,
      kvHeads: 1,
      chunkRows: 32,
      depth: 256,
      capacity: 512,
      window: 512,
      past: 0,
      query: 32,
    },
    {
      name: "full D512 C640 prefill",
      batch: 1,
      heads: 8,
      kvHeads: 1,
      chunkRows: 32,
      depth: 512,
      capacity: 640,
      window: 0,
      past: 576,
      query: 32,
    },
    {
      name: "full D512 C640 decode",
      batch: 1,
      heads: 8,
      kvHeads: 1,
      chunkRows: 1,
      depth: 512,
      capacity: 640,
      window: 0,
      past: 617,
      query: 1,
    },
    {
      name: "full D512 C640 P0",
      batch: 1,
      heads: 8,
      kvHeads: 1,
      chunkRows: 32,
      depth: 512,
      capacity: 640,
      window: 0,
      past: 0,
      query: 32,
    },
  ];
  const QUERY = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
  const KEY = (i: number): number => (((i * 11) % 19) - 9) * 0.23;
  const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.31;
  const cache = new Map<string, GPUComputePipeline>();
  for (const spec of PRODUCT_CASES) {
    // `runStateAttention` は scale を halfScale(D) に固定するので、製品の attrs scale=1 を
    // 出すために q / k を 1/halfScale 倍して渡す（積 q·halfScale × k·halfScale = q·k）。
    // 参照側も同じ入力・同じ halfScale を見るので突合は自己整合のまま、S の大きさだけが
    // 製品と同じ帯（scale=1）になる。
    for (const emulateScaleOne of [false, true]) {
      const boost = emulateScaleOne ? 1 / halfScale(spec.depth) : 1;
      const kvPlanes = spec.batch * spec.kvHeads;
      const inputs = {
        q: seededF32(
          spec.batch * spec.heads * spec.chunkRows * spec.depth,
          (i) => QUERY(i) * boost,
        ),
        insK: seededF32(kvPlanes * spec.chunkRows * spec.depth, (i) => KEY(i) * boost),
        insV: seededF32(kvPlanes * spec.chunkRows * spec.depth, VALUE),
        slotK: seededF32(kvPlanes * spec.capacity * spec.depth, (i) => KEY(i + 3) * boost),
        slotV: seededF32(kvPlanes * spec.capacity * spec.depth, (i) => VALUE(i + 5)),
      };
      const result = await runStateAttention(gpu.device, spec, inputs, { cache });
      const expected = referenceStateAttention({
        batch: spec.batch,
        heads: spec.heads,
        kvHeads: spec.kvHeads,
        chunkRows: spec.chunkRows,
        depth: spec.depth,
        capacity: spec.capacity,
        window: spec.window,
        past: spec.past,
        query: spec.query,
        scale: halfScale(spec.depth),
        ...inputs,
      });
      judge(
        `④state-attn ${spec.name}${emulateScaleOne ? " scale=1" : ""}`,
        `colCap=${caseColCap(spec)} live=${spec.past}+${spec.query}`,
        result.out,
        expected.data,
        5e-5,
        0,
      );
    }
  }
}

// ---------------------------------------------------------------------------

console.log("");
console.log("段 | 判定 | NaN | Inf | maxAbs | maxRel | exact/total | 詳細");
for (const v of verdicts) {
  console.log(
    `${v.stage} | ${v.pass ? "OK" : "NG"} | ${v.nan} | ${v.inf} | ` +
      `${Number.isNaN(v.maxAbs) ? "-" : v.maxAbs.toExponential(2)} | ` +
      `${Number.isNaN(v.maxRel) ? "-" : v.maxRel.toExponential(2)} | ` +
      `${v.exact}/${v.total} | ${v.detail}`,
  );
}
const failed = verdicts.filter((v) => !v.pass);
console.log(`[probe3b] 総括: ${verdicts.length} 段中 ${failed.length} 段 NG`);
gpu.device.destroy();
Deno.exit(failed.length === 0 ? 0 : 1);
