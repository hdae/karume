import { assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import { GpuValidationError } from "../src/gpu/device.ts";
import { PipelineCache, PipelineKeyConflictError } from "../src/gpu/pipeline-cache.ts";

/** PipelineCache が触る面だけを持つフェイク。DOM 型全体は再現しないため cast で渡す。 */
type FakeGpu = {
  readonly device: GPUDevice;
  /** 呼び出し順の記録（errorScope の張り方を観測する）。 */
  readonly calls: string[];
  readonly modules: string[];
};

const createFakeGpu = (scopeError: GPUError | null = null): FakeGpu => {
  const calls: string[] = [];
  const modules: string[] = [];
  const device = {
    pushErrorScope: (filter: string): void => {
      calls.push(`push:${filter}`);
    },
    popErrorScope: (): Promise<GPUError | null> => {
      calls.push("pop");
      return Promise.resolve(scopeError);
    },
    createShaderModule: (descriptor: { readonly code: string }) => {
      calls.push("createShaderModule");
      modules.push(descriptor.code);
      return {};
    },
    createComputePipeline: () => {
      calls.push("createComputePipeline");
      return { id: modules.length };
    },
  };
  return { device: device as unknown as GPUDevice, calls, modules };
};

const WGSL_A = "@compute @workgroup_size(1) fn main() {}";
const WGSL_B = "@compute @workgroup_size(2) fn main() {}";

Deno.test("PipelineCache は同一キー・同一 WGSL でパイプラインを再生成しない", async () => {
  const gpu = createFakeGpu();
  const cache = new PipelineCache(gpu.device);

  const first = await cache.get("elementwise:add:f32", WGSL_A);
  const second = await cache.get("elementwise:add:f32", WGSL_A);

  assertStrictEquals(first, second);
  assertEquals(cache.size, 1);
  assertEquals(gpu.modules, [WGSL_A], "2 回目はシェーダモジュールを作らない");
});

Deno.test("PipelineCache は同一キーに異なる WGSL が来たら即例外にする（決定性の破れ）", async () => {
  const gpu = createFakeGpu();
  const cache = new PipelineCache(gpu.device);

  await cache.get("elementwise:add:f32", WGSL_A);
  await assertRejects(
    () => cache.get("elementwise:add:f32", WGSL_B),
    PipelineKeyConflictError,
    "elementwise:add:f32",
  );
  assertEquals(gpu.modules, [WGSL_A], "衝突時は device に触らない");
});

Deno.test("PipelineCache はパイプライン生成を validation errorScope で囲む", async () => {
  const gpu = createFakeGpu();
  const cache = new PipelineCache(gpu.device);

  await cache.get("k", WGSL_A);

  assertEquals(gpu.calls, [
    "push:validation",
    "createShaderModule",
    "createComputePipeline",
    "pop",
  ]);
});

Deno.test("PipelineCache は errorScope が捕捉した検証エラーを例外に変換し、キャッシュしない", async () => {
  const error = { message: "workgroup size exceeds limit" } as unknown as GPUError;
  const gpu = createFakeGpu(error);
  const cache = new PipelineCache(gpu.device);

  await assertRejects(
    () => cache.get("too-large", WGSL_A),
    GpuValidationError,
    "workgroup size exceeds limit",
  );
  assertEquals(cache.size, 0);
});
