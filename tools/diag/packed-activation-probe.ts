// packed int8 活性（ADR 0105）の並列 GEMV が f32 経路と u32 で割れる機（Metal / M2）で、
// **どの変換で割れるか**を切り分ける診断スクリプト。製品コードは触らず、生成した WGSL の字面を
// 変種ごとに書き換えて同じ入力で回し、f32 経路の出力と要素ごとに突き合わせる。
//
// 変種（全て同じ活性・重み・bias・scale）:
//   f32          既定経路（参照）
//   f32/again    同じ pipeline をもう 1 度（自己一致 = 非決定性の有無）
//   f32+xbar     f32 の活性ロード値を実行時 0 との XOR 障壁に通す（数値は恒等のはず）
//   f32+fma      積和を明示 fma() にする
//   packed       現行の packed 変種（quad 障壁つき — ADR 0105 追記 3）
//   packed-nobar 追記 3 の障壁を外した packed（追記 2 までの形）
//   packed+pbar  積ごとに XOR 障壁（丸めを演算ごとに固定）
//   packed+fma   積和を明示 fma() にする
//
// 読み方: f32+xbar が f32 と割れれば、コンパイラは同じ数式でも式形で結果を変える（fast-math の
// 再結合）— 式の書き換えでは 2 つのカーネルを一致させられない。f32+fma と packed+fma が互いに
// 一致し f32 と割れれば、差の出所は fma 縮約の有無で、明示 fma を両経路に置けば揃う。
//
// 使い方:
//   deno run -A tools/diag/packed-activation-probe.ts
//   出力（表とヘッダ）をそのまま貼る。

import { acquireGpu, type GpuContext } from "../../packages/runtime/src/gpu/device.ts";
import { BUFFER_USAGE as U, MAP_MODE } from "../../packages/runtime/src/gpu/webgpu-constants.ts";
import {
  linearGemvParallelPackedParams,
  linearGemvParallelPackedWgsl,
  linearGemvParallelWgsl,
  linearGemvParams,
  linearGemvUnit,
} from "../../packages/runtime/src/kernels/linear-gemv.ts";
import {
  STATIC_QUANTIZE_PACKED_WGSL,
  STATIC_QUANTIZE_WGSL,
  staticQuantizePackedParams,
  staticQuantizeParams,
} from "../../packages/runtime/src/kernels/static-quantize.ts";

type Storage = "i2" | "i4" | "i8";
type Lanes = 2 | 4 | 8 | 16 | 32;
type Case = { storage: Storage; lanes: Lanes; group?: number };

/** 落ちた組（i2 / lanes 2）を先頭に、格納と lane を変えた 3 組を足す。k は unit × 5・m = 1・n = 36。 */
const CASES: readonly Case[] = [
  { storage: "i2", lanes: 2 },
  { storage: "i2", lanes: 32 },
  { storage: "i4", lanes: 4, group: 32 },
  { storage: "i8", lanes: 16 },
];
const X_SCALE = Math.fround(0.005826006643474102);
const N = 36;

/** 置換の件数を必ず出す（0 件の書き換えは「変種になっていない」ので fail loudly）。 */
const rewrite = (code: string, pattern: RegExp, replacement: string, what: string): string => {
  let count = 0;
  const out = code.replace(pattern, (...args) => {
    count += 1;
    return replacement.replace(/\$(\d)/g, (_, index) => String(args[Number(index)]));
  });
  if (count === 0) throw new Error(`${what}: 置換対象が 0 件（式形が想定と違う）`);
  return out;
};
// 積和 1 行: `acc = acc + <活性>.c * <重み項>;`（重み項は i2 の `dt.c`、i4 / i8 の `(f32(…) * wst)`）。
const PRODUCT = /acc = acc \+ (\w+\.[xyzw]) \* (.+);$/gm;
const F32_LOAD = /let (\w+) = x\[(xq\w+ \+ \d+u)\];/g;
const PACKED_QUAD =
  /bitcast<vec4<f32>>\(bitcast<vec4<u32>>\(vec4<f32>\(unpack4xI8\(([^)]+)\)\) \* dims\.x_scale\) \^ vec4<u32>\(dims\.rounding_mask\)\)/g;

const variants = (c: Case): readonly [name: string, packed: boolean, code: string][] => {
  const f32 = linearGemvParallelWgsl(c.storage, c.group, c.lanes);
  const packed = linearGemvParallelPackedWgsl(c.storage, c.group, c.lanes);
  // f32 の Dims には実行時 0 の語が無いので `dims.m - 1u`（m = 1）を使う。
  return [
    ["f32", false, f32],
    ["f32/again", false, f32],
    [
      "f32+xbar",
      false,
      rewrite(
        f32,
        F32_LOAD,
        "let $1 = bitcast<vec4<f32>>(bitcast<vec4<u32>>(x[$2]) ^ vec4<u32>(dims.m - 1u));",
        "f32+xbar",
      ),
    ],
    ["f32+fma", false, rewrite(f32, PRODUCT, "acc = fma($1, $2, acc);", "f32+fma")],
    ["packed", true, packed],
    [
      "packed-nobar",
      true,
      rewrite(packed, PACKED_QUAD, "vec4<f32>(unpack4xI8($1)) * dims.x_scale", "packed-nobar"),
    ],
    [
      "packed+pbar",
      true,
      rewrite(
        packed,
        PRODUCT,
        "acc = acc + bitcast<f32>(bitcast<u32>($1 * $2) ^ dims.rounding_mask);",
        "packed+pbar",
      ),
    ],
    ["packed+fma", true, rewrite(packed, PRODUCT, "acc = fma($1, $2, acc);", "packed+fma")],
  ];
};

const hex = (v: number): string => "0x" + (v >>> 0).toString(16).padStart(8, "0");
/** 2 つの f32 ビット列の差を ULP（符号つき整数の差の絶対値）で。 */
const ulp = (a: number, b: number): number => {
  const sa = a >>> 31 ? -(a & 0x7fffffff) : a, sb = b >>> 31 ? -(b & 0x7fffffff) : b;
  return Math.abs(sa - sb);
};

const runCase = async (gpu: GpuContext, c: Case): Promise<void> => {
  const d = gpu.device;
  const k = linearGemvUnit(c.storage) * 5, m = 1;
  const bits = c.storage === "i2" ? 2 : c.storage === "i4" ? 4 : 8;
  const owned: GPUBuffer[] = [];
  const make = (
    v: number | ArrayBufferView<ArrayBuffer>,
    usage = U.STORAGE | U.COPY_DST | U.COPY_SRC,
  ): GPUBuffer => {
    const b = d.createBuffer({ size: typeof v === "number" ? v : v.byteLength, usage });
    owned.push(b);
    if (typeof v !== "number") d.queue.writeBuffer(b, 0, v);
    return b;
  };
  try {
    // gpu_packed_static_quantize_test.ts の「normal」シナリオと同じ入力。
    const x = Float32Array.from({ length: m * k }, (_, i) => Math.sin(i * 0.037) * 0.75);
    const w = Uint32Array.from(
      { length: N * k * bits / 32 },
      (_, i) => Math.imul(i + 1, 0x9e3779b9) >>> 0,
    );
    const bias = Float32Array.from({ length: N }, (_, i) => (i % 7 - 3) * 0.11);
    const ws = Float32Array.from(
      { length: N * (c.group === undefined ? 1 : k / c.group) },
      (_, i) => (i % 17 + 1) * 0.00017,
    );
    const input = make(x), weight = make(w), biasBuf = make(bias), scaleBuf = make(ws);
    const xf32 = make(m * k * 4), xPacked = make(m * k);
    const sp = make(staticQuantizeParams(m * k, X_SCALE), U.UNIFORM | U.COPY_DST);
    const pp = make(staticQuantizePackedParams(m * k, X_SCALE), U.UNIFORM | U.COPY_DST);
    const lp = make(linearGemvParams(c.storage, m, N, k, c.group), U.UNIFORM | U.COPY_DST);
    const kp = make(
      linearGemvParallelPackedParams(c.storage, m, N, k, X_SCALE, c.group),
      U.UNIFORM | U.COPY_DST,
    );
    const list = variants(c);
    const outputs = list.map(() => make(m * N * 4));
    const stage = make(m * N * 4 * list.length, U.MAP_READ | U.COPY_DST);
    const encoder = d.createCommandEncoder(), pass = encoder.beginComputePass();
    const dispatch = (code: string, buffers: GPUBuffer[], gx: number): void => {
      const pipeline = d.createComputePipeline({
        layout: "auto",
        compute: { module: d.createShaderModule({ code }), entryPoint: "main" },
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(
        0,
        d.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
        }),
      );
      pass.dispatchWorkgroups(gx, m);
    };
    dispatch(STATIC_QUANTIZE_WGSL, [sp, input, xf32], Math.ceil(m * k / 128));
    dispatch(STATIC_QUANTIZE_PACKED_WGSL, [pp, input, xPacked], Math.ceil(m * k / 128));
    const tiles = Math.ceil(N / (128 / c.lanes));
    list.forEach(([, packed, code], index) => {
      dispatch(
        code,
        [packed ? kp : lp, packed ? xPacked : xf32, weight, biasBuf, outputs[index], scaleBuf],
        tiles,
      );
    });
    pass.end();
    outputs.forEach((out, index) => {
      encoder.copyBufferToBuffer(out, 0, stage, index * m * N * 4, m * N * 4);
    });
    d.queue.submit([encoder.finish()]);
    await stage.mapAsync(MAP_MODE.READ);
    const words = new Uint32Array(stage.getMappedRange().slice(0));
    stage.unmap();
    const reference = words.subarray(0, m * N);
    const label = `${c.storage}${
      c.group === undefined ? "" : `g${c.group}`
    } k=${k} lanes=${c.lanes}`;
    console.log(`\n== ${label} (n=${N}, m=${m}) — f32 との差 ==`);
    console.log("variant        mismatch  maxULP  first mismatches (index: f32 / variant)");
    list.forEach(([name], index) => {
      const got = words.subarray(index * m * N, (index + 1) * m * N);
      const bad: string[] = [];
      let maxUlp = 0;
      for (let i = 0; i < m * N; i += 1) {
        if (got[i] === reference[i]) continue;
        maxUlp = Math.max(maxUlp, ulp(reference[i], got[i]));
        if (bad.length < 3) bad.push(`${i}: ${hex(reference[i])} / ${hex(got[i])}`);
      }
      const mismatch = countMismatch(reference, got);
      console.log(
        `${name.padEnd(14)} ${String(mismatch).padStart(8)} ${String(maxUlp).padStart(7)}  ${
          bad.join("  ")
        }`,
      );
    });
    // 変種同士の一致も出す（fma 同士・障壁同士）。
    const same = (a: number, b: number): boolean => {
      const l = words.subarray(a * m * N, (a + 1) * m * N),
        r = words.subarray(b * m * N, (b + 1) * m * N);
      for (let i = 0; i < m * N; i += 1) if (l[i] !== r[i]) return false;
      return true;
    };
    const at = (name: string): number => list.findIndex(([n]) => n === name);
    console.log(
      `pairs: f32+fma==packed+fma ${same(at("f32+fma"), at("packed+fma"))} / ` +
        `f32+xbar==f32 ${same(at("f32+xbar"), at("f32"))} / ` +
        `packed==packed-nobar ${same(at("packed"), at("packed-nobar"))} / ` +
        `packed+pbar==f32 ${same(at("packed+pbar"), at("f32"))}`,
    );
  } finally {
    d.queue.submit([]);
    await d.queue.onSubmittedWorkDone();
    for (const b of owned) b.destroy();
  }
};

const countMismatch = (a: Uint32Array, b: Uint32Array): number => {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) if (a[i] !== b[i]) n += 1;
  return n;
};

const main = async (): Promise<void> => {
  const gpu = await acquireGpu();
  try {
    const info = gpu.adapterInfo;
    console.log(
      `adapter: ${info.vendor} / ${info.architecture} / ${info.description} — ${Deno.build.os} ${Deno.build.arch} Deno ${Deno.version.deno}`,
    );
    for (const c of CASES) await runCase(gpu, c);
  } finally {
    gpu.destroy();
  }
};

if (import.meta.main) await main();
