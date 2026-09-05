import { assert, assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { GpuInternalError, GpuValidationError } from "../src/gpu/error-scope.ts";
import {
  PipelineCache,
  PipelineKeyConflictError,
  SessionPipelines,
} from "../src/gpu/pipeline-cache.ts";
import { createSession } from "../src/runtime/executor.ts";
import { fill, graphModelBuffer, singleOpGraph } from "./helpers/graph.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/** PipelineCache が触る面だけを持つフェイク。DOM 型全体は再現しないため cast で渡す。 */
type FakeGpu = {
  readonly device: GPUDevice;
  /** 呼び出し順の記録（errorScope の張り方を観測する）。 */
  readonly calls: string[];
  readonly modules: string[];
  /** 次の popErrorScope が返す検証エラー。差し替えて「失敗 → 再試行で成功」を作る。 */
  scopeError: GPUError | null;
  /**
   * 次の popErrorScope が **internal スコープ**で返すエラー。実 GPU では作れない失敗
   * （シェーダが複雑すぎてコンパイルできない等）を踏むための面。
   */
  internalError: GPUError | null;
};

const createFakeGpu = (
  scopeError: GPUError | null = null,
  internalError: GPUError | null = null,
): FakeGpu => {
  const calls: string[] = [];
  const modules: string[] = [];
  // 実 device はフィルタに一致したスコープのエラーしか返さないため、フェイクも push された
  // フィルタを LIFO で追跡し、フィルタ別に返し分ける（両方に同じエラーを返すと、
  // withPipelineScope の internal 優先規則が誤発火する）。
  const filters: string[] = [];
  const device = {
    pushErrorScope: (filter: string): void => {
      calls.push(`push:${filter}`);
      filters.push(filter);
    },
    popErrorScope: (): Promise<GPUError | null> => {
      calls.push("pop");
      const filter = filters.pop();
      if (filter === "validation") return Promise.resolve(fake.scopeError);
      return Promise.resolve(filter === "internal" ? fake.internalError : null);
    },
    createShaderModule: (descriptor: { readonly code: string }) => {
      calls.push("createShaderModule");
      modules.push(descriptor.code);
      return {};
    },
    createComputePipeline: () => {
      calls.push("createComputePipeline");
      return { id: modules.length, getBindGroupLayout: () => ({ id: modules.length }) };
    },
  };
  const fake: FakeGpu = {
    device: device as unknown as GPUDevice,
    calls,
    modules,
    scopeError,
    internalError,
  };
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

Deno.test("PipelineCache はパイプライン生成を internal + validation の 2 スコープで囲む", async () => {
  const gpu = createFakeGpu();
  const cache = new PipelineCache(gpu.device);

  await cache.get("k", WGSL_A);

  assertEquals(gpu.calls, [
    "push:internal",
    "push:validation",
    "createShaderModule",
    "createComputePipeline",
    "pop",
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

// GpuInternalError は実 GPU では作れない（internal エラーは実装都合の失敗）ため、この型を
// 生成する経路はフェイクでしか踏めない。環境の限界（internal）と WGSL の記述不正（validation）
// の分岐先が違うことが公開面の意味なので、型のマッピングをここで固定する。
Deno.test("PipelineCache は internal スコープの捕捉を GpuInternalError に変換する", async () => {
  const internal = { message: "shader too complex" } as unknown as GPUError;
  const gpu = createFakeGpu(null, internal);
  const cache = new PipelineCache(gpu.device);

  const error = await assertRejects(() => cache.get("k", WGSL_A), GpuInternalError);
  assert(error.message.includes("k"), error.message);
  assert(error.message.includes("shader too complex"), error.message);
  assertEquals(cache.size, 0, "失敗したエントリは残さない");

  // 積み残しが無いことの裏（2 本とも pop されていれば直後の正常な生成が通る）。
  gpu.internalError = null;
  assert(await cache.get("k", WGSL_A) !== undefined);
});

// 優先順位の門: internal で無効化されたパイプラインを触る後続の操作は**派生の** validation を
// 立てるので、validation を先に返すと根因の internal が捨てられる。この 2 本同時の形だけが
// 「internal 優先」を落とす（internal 単独のケースは順序を戻しても緑のまま通る）。
Deno.test("PipelineCache は internal と validation を同時に捕捉したら internal を返す", async () => {
  const gpu = createFakeGpu(
    { message: "derived: pipeline is invalid" } as unknown as GPUError,
    { message: "shader too complex" } as unknown as GPUError,
  );
  const cache = new PipelineCache(gpu.device);

  const error = await assertRejects(() => cache.get("both", WGSL_A), GpuInternalError);
  assert(error.message.includes("shader too complex"), error.message);
  assert(
    !error.message.includes("derived"),
    `派生の validation が根因を上書きしている: ${error.message}`,
  );
  assertEquals(cache.size, 0);
});

Deno.test("SessionPipelines は device 寿命の 1 本を共有し、2 本目の Session で再コンパイルしない", async () => {
  const gpu = createFakeGpu();
  const cache = new PipelineCache(gpu.device);
  const first = new SessionPipelines(cache);
  const second = new SessionPipelines(cache);

  await first.get("elementwise:add:f32", WGSL_A);
  await second.get("elementwise:add:f32", WGSL_A);

  assertEquals(gpu.modules, [WGSL_A], "2 本目の Session はシェーダモジュールを作らない");
  assertEquals(first.usedCount, 1);
  assertEquals(second.usedCount, 1, "共有していても『自分が使った本数』は自分ぶんだけ");
  assertEquals(second.deviceCount, 1, "device 合計は和集合");
});

Deno.test("SessionPipelines の使用本数はキー集合の大きさ（引いた回数ではない）", async () => {
  const gpu = createFakeGpu();
  const use = new SessionPipelines(new PipelineCache(gpu.device));

  await use.get("k1", WGSL_A);
  await use.get("k1", WGSL_A);
  await use.get("k2", WGSL_B);

  assertEquals(use.usedCount, 2);
});

Deno.test("SessionPipelines は失敗したキーを使用集合に数えない", async () => {
  const error = { message: "shader compile failed" } as unknown as GPUError;
  const gpu = createFakeGpu(error);
  const use = new SessionPipelines(new PipelineCache(gpu.device));

  await assertRejects(() => use.get("broken", WGSL_A), GpuValidationError);

  assertEquals(use.usedCount, 0, "device に残らないキーを Session だけが使ったことにしない");
  assertEquals(use.deviceCount, 0);
});

Deno.test("PipelineKeyConflictError は Session を跨いで検出される（同一 device・同一キー）", async () => {
  const gpu = createFakeGpu();
  // 決定性の破れは「同じキーに別の WGSL」で起きる。キャッシュが Session ごとだった頃は
  // 別 Session 同士の衝突がそもそも突き合わされず、後発の Session が黙って別カーネルを
  // 走らせられた（ブラウザの暗黙キャッシュは WGSL 文字列がキーなので警告も出ない）。
  const cache = new PipelineCache(gpu.device);
  await new SessionPipelines(cache).get("matmul:v4", WGSL_A);

  await assertRejects(
    () => new SessionPipelines(cache).get("matmul:v4", WGSL_B),
    PipelineKeyConflictError,
    "matmul:v4",
  );
  assertEquals(gpu.modules, [WGSL_A], "衝突時は device に触らない");
});

Deno.test({
  name: "パイプラインキャッシュは device 寿命で、dispose 済み Session のキーも残る（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const model = () => openModel(graphModelBuffer(singleOpGraph("relu", [[8, 8]], [[8, 8]])));
    const x = fill([8, 8], (i) => (i % 13) - 6);

    try {
      const first = await createSession(gpu, model());
      assertEquals(
        first.diagnostics().pipelineCount,
        0,
        "構築相はパイプラインを 1 本も引かない（生成は初回 run のミス経路）",
      );
      await first.run({ x0: x });
      const used = first.diagnostics().pipelineCount;
      assert(used > 0, "run でパイプラインが立つ");
      assertEquals(
        first.diagnostics().devicePipelineCount,
        used,
        "この device には 1 本目ぶんだけ",
      );
      await first.dispose();

      const second = await createSession(gpu, model());
      assertEquals(
        second.diagnostics().devicePipelineCount,
        used,
        "dispose 済み Session のキーは device 合計に残る（寿命は GpuContext と一致）",
      );
      assertEquals(second.diagnostics().pipelineCount, 0, "使用本数は Session ごとに 0 から");

      await second.run({ x0: x });

      assertEquals(second.diagnostics().pipelineCount, used, "使った本数は 1 本目と同じ");
      assertEquals(
        second.diagnostics().devicePipelineCount,
        used,
        "2 本目は device 合計を 1 本も増やさない = 同じ WGSL を再コンパイルしていない",
      );
      await second.dispose();
    } finally {
      gpu.destroy();
    }
  },
});
