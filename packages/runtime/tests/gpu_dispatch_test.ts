import { assertEquals } from "@std/assert";
import { RunArena } from "../src/gpu/arena.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { PipelineCache } from "../src/gpu/pipeline-cache.ts";
import { SubmitScheduler } from "../src/gpu/submit.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const WORKGROUP_SIZE = 64;

/** 自明な elementwise（y = 2x + 1）。grid-stride で 1D 上限を跨いでも成立する形。 */
const DOUBLE_PLUS_ONE_WGSL = `
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
  let stride = groups.x * ${WORKGROUP_SIZE}u;
  let n = arrayLength(&src);
  var i = gid.x;
  while (i < n) {
    dst[i] = src[i] * 2.0 + 1.0;
    i = i + stride;
  }
}
`;

Deno.test({
  name: "device / cache / scheduler / arena を通した compute dispatch が正しい値を返す（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const scheduler = new SubmitScheduler(gpu);
    const cache = new PipelineCache(gpu.device);
    const arena = new RunArena(gpu.device, () => scheduler.flush());
    try {
      const count = 4096;
      const input = Float32Array.from({ length: count }, (_, i) => i);
      const bytes = input.byteLength;

      const src = arena.allocHostWritten(
        bytes,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      );
      gpu.device.queue.writeBuffer(src, 0, input);

      const dst = arena.allocStorage(bytes);
      arena.retain(dst, 0, { pinned: true });
      assertEquals(arena.isReadable(dst), true);

      const pipeline = await cache.get("test:double_plus_one:f32", DOUBLE_PLUS_ONE_WGSL);
      const bindGroup = gpu.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: src } },
          { binding: 1, resource: { buffer: dst } },
        ],
      });
      const groups = Math.ceil(count / WORKGROUP_SIZE);
      scheduler.dispatch(pipeline, bindGroup, [groups, 1, 1], "test:double_plus_one:f32");
      await scheduler.flush();

      const staging = gpu.device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const encoder = gpu.device.createCommandEncoder();
      encoder.copyBufferToBuffer(dst, 0, staging, 0, bytes);
      gpu.device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const actual = new Float32Array<ArrayBuffer>(staging.getMappedRange().slice(0));
      staging.unmap();
      staging.destroy();

      assertEquals(actual.length, count);
      for (let i = 0; i < count; i += 1) {
        // 2x+1 は整数で、f32 で厳密表現できる範囲に収めてある
        assertEquals(actual[i], i * 2 + 1, `index ${i}`);
      }
      assertEquals(scheduler.stats.dispatchCount, 1);
      assertEquals(scheduler.stats.submitCount, 1);
    } finally {
      await arena.destroy();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "アリーナのプール再利用が同じバッファを配り直しても結果が壊れない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const scheduler = new SubmitScheduler(gpu);
    const cache = new PipelineCache(gpu.device);
    const arena = new RunArena(gpu.device, () => scheduler.flush());
    try {
      const count = 256;
      const bytes = count * 4;
      const source = arena.allocHostWritten(
        bytes,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      );
      gpu.device.queue.writeBuffer(source, 0, Float32Array.from({ length: count }, () => 1));

      const pipeline = await cache.get("test:double_plus_one:f32", DOUBLE_PLUS_ONE_WGSL);
      const dispatchInto = (src: GPUBuffer, dst: GPUBuffer): void => {
        const bindGroup = gpu.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: src } },
            { binding: 1, resource: { buffer: dst } },
          ],
        });
        scheduler.dispatch(
          pipeline,
          bindGroup,
          [Math.ceil(count / WORKGROUP_SIZE), 1, 1],
          "test:double_plus_one:f32",
        );
      };

      // 中間値 → 最終出力。executor と同じ規律で、各ノードの境界で「入力の消費ぶん」と
      // 「自ノード出力の定義ぶん」を解放する。中間値は次の確保で再利用させる。
      const intermediate = arena.allocStorage(bytes);
      arena.retain(intermediate, 1);
      dispatchInto(source, intermediate);
      arena.release(intermediate); // ノード 1 の境界: 定義ぶん

      const output = arena.allocStorage(bytes);
      arena.retain(output, 0, { pinned: true });
      dispatchInto(intermediate, output);
      arena.release(intermediate); // ノード 2 の境界: 消費 1 回ぶん
      arena.release(output); // ノード 2 の境界: 定義ぶん
      arena.assertDrained();

      const recycled = arena.allocStorage(bytes);
      assertEquals(recycled === intermediate, true, "解放済み中間値が再利用される");
      assertEquals(arena.stats.reuseCount, 1);

      await scheduler.flush();

      const staging = gpu.device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const encoder = gpu.device.createCommandEncoder();
      encoder.copyBufferToBuffer(output, 0, staging, 0, bytes);
      gpu.device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const actual = new Float32Array<ArrayBuffer>(staging.getMappedRange().slice(0));
      staging.unmap();
      staging.destroy();

      // 1 → 3 → 7
      for (let i = 0; i < count; i += 1) {
        assertEquals(actual[i], 7, `index ${i}`);
      }
    } finally {
      await arena.destroy();
      gpu.destroy();
    }
  },
});
