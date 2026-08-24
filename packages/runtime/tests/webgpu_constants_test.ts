// 自前定義した WebGPU ビットフラグ（src/gpu/webgpu-constants.ts）の**仕様転記の門**。
//
// 自前定義の動機は import 安全性（WebGPU 非対応環境で import しただけで落ちないこと）なので、
// 値そのものは仕様からの手写しになる。転記ミスは沈黙誤動作に化ける（usage 不足は
// createBuffer の validation エラー、余剰は静かに通る）ため、名前空間を持つ環境では実物と
// 全ビットを突合する。期待値は**プラットフォームの名前空間から取る** — 自前定義と同じ数字を
// もう一度書くと恒真になり、転記ミスが 1 件も落ちない。
//
// 名前空間は WebGPU 非対応環境に存在しないので、判定は `Object.hasOwn(globalThis, ...)`
// （裸の識別子参照はこのファイルの内側でも評価時に落ちる）。無い環境は明示 SKIP する。

import { assertEquals } from "@std/assert";
import { BUFFER_USAGE, MAP_MODE } from "../src/gpu/webgpu-constants.ts";

const BUFFER_USAGE_NAMESPACE = Object.hasOwn(globalThis, "GPUBufferUsage");
const MAP_MODE_NAMESPACE = Object.hasOwn(globalThis, "GPUMapMode");

if (!BUFFER_USAGE_NAMESPACE || !MAP_MODE_NAMESPACE) {
  console.warn(
    "[karume] WebGPU のフラグ名前空間が無いため仕様転記の突合を SKIP する" +
      "（自前定数の値が仕様と一致するかは検証されない）",
  );
}

Deno.test({
  name: "BUFFER_USAGE は GPUBufferUsage の全ビットを仕様どおり写している",
  ignore: !BUFFER_USAGE_NAMESPACE,
  fn: () => {
    assertEquals({ ...BUFFER_USAGE }, {
      MAP_READ: GPUBufferUsage.MAP_READ,
      MAP_WRITE: GPUBufferUsage.MAP_WRITE,
      COPY_SRC: GPUBufferUsage.COPY_SRC,
      COPY_DST: GPUBufferUsage.COPY_DST,
      INDEX: GPUBufferUsage.INDEX,
      VERTEX: GPUBufferUsage.VERTEX,
      UNIFORM: GPUBufferUsage.UNIFORM,
      STORAGE: GPUBufferUsage.STORAGE,
      INDIRECT: GPUBufferUsage.INDIRECT,
      QUERY_RESOLVE: GPUBufferUsage.QUERY_RESOLVE,
    });
  },
});

Deno.test({
  name: "MAP_MODE は GPUMapMode の全ビットを仕様どおり写している",
  ignore: !MAP_MODE_NAMESPACE,
  fn: () => {
    assertEquals({ ...MAP_MODE }, {
      READ: GPUMapMode.READ,
      WRITE: GPUMapMode.WRITE,
    });
  },
});
