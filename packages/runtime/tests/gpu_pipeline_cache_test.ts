import { assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import { GpuValidationError } from "../src/gpu/device.ts";
import { PipelineCache, PipelineKeyConflictError } from "../src/gpu/pipeline-cache.ts";

/** PipelineCache が触る面だけを持つフェイク。DOM 型全体は再現しないため cast で渡す。 */
type FakeGpu = {
  readonly device: GPUDevice;
  /** 呼び出し順の記録（errorScope の張り方を観測する）。 */
  readonly calls: string[];
  readonly modules: string[];
  /** 次の popErrorScope が返す検証エラー。差し替えて「失敗 → 再試行で成功」を作る。 */
  scopeError: GPUError | null;
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
      return Promise.resolve(fake.scopeError);
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
  const fake: FakeGpu = { device: device as unknown as GPUDevice, calls, modules, scopeError };
  return fake;
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

Deno.test("PipelineCache は未決着の同一キー要求でも生成を 1 本だけ共有する", async () => {
  const gpu = createFakeGpu();
  const cache = new PipelineCache(gpu.device);

  // await を挟まずに 2 本走らせる = 1 本目がまだ解決していない状態で 2 本目が来る
  const first = cache.get("elementwise:add:f32", WGSL_A);
  const second = cache.get("elementwise:add:f32", WGSL_A);
  const [firstPipeline, secondPipeline] = await Promise.all([first, second]);

  assertStrictEquals(firstPipeline, secondPipeline);
  assertEquals(cache.size, 1);
  assertEquals(gpu.modules, [WGSL_A], "同時要求でもシェーダモジュールは 1 回だけ作る");
  assertEquals(
    gpu.calls.filter((call) => call === "createComputePipeline").length,
    1,
    "同時要求でもパイプライン生成は 1 回だけ",
  );
  assertEquals(
    gpu.calls.filter((call) => call === "push:validation").length,
    1,
    "errorScope も重ねない",
  );
});

Deno.test("PipelineCache は失敗した生成を持ち越さず、再試行で作り直す", async () => {
  const error = { message: "shader compile failed" } as unknown as GPUError;
  const gpu = createFakeGpu(error);
  const cache = new PipelineCache(gpu.device);

  await assertRejects(
    () => cache.get("retry", WGSL_A),
    GpuValidationError,
    "shader compile failed",
  );
  assertEquals(cache.size, 0, "失敗したエントリは残さない");

  // 失敗が拒否済み Promise としてキャッシュに残っていると、以後の get() は device に
  // 触れないまま同じ拒否を返し続ける（= 一時的な失敗から復帰できなくなる）
  gpu.scopeError = null;
  const pipeline = await cache.get("retry", WGSL_A);

  assertEquals(cache.size, 1);
  assertEquals(gpu.modules, [WGSL_A, WGSL_A], "再試行では作り直す");
  assertStrictEquals(await cache.get("retry", WGSL_A), pipeline, "成功後は共有に戻る");
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
