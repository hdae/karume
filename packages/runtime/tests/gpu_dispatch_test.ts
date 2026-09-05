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

      const dst = arena.allocRegion(bytes);

      const { pipeline, layout } = await cache.get(
        "test:double_plus_one:f32",
        DOUBLE_PLUS_ONE_WGSL,
      );
      const bindGroup = gpu.device.createBindGroup({
        layout,
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
  name:
    "領域の offset 束縛で中間と出力を 1 本のバッファに同居させ、配り直しても結果が壊れない（実 GPU）",
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

      const { pipeline, layout, roles } = await cache.get(
        "test:double_plus_one:f32",
        DOUBLE_PLUS_ONE_WGSL,
      );
      assertEquals(roles, { reads: new Set([0]), writes: new Set([1]) }, "WGSL 宣言から採った役割");
      const dispatchInto = (src: GPUBufferBinding, dst: GPUBufferBinding): void => {
        const bindGroup = gpu.device.createBindGroup({
          layout,
          entries: [
            { binding: 0, resource: src },
            { binding: 1, resource: dst },
          ],
        });
        scheduler.dispatch(
          pipeline,
          bindGroup,
          [Math.ceil(count / WORKGROUP_SIZE), 1, 1],
          "test:double_plus_one:f32",
        );
      };

      // 計画（ADR 0093）と同じ形: 1 本の領域に中間（offset 0）と出力（256 整列の次の区間）を
      // 同居させる。同じ dispatch で読む中間と書く出力が同じバッファに載るのは usage scope の
      // 制約に触れるので、そこだけは別バッファ（source）から読む形で組む。
      const align = gpu.device.limits.minStorageBufferOffsetAlignment;
      const outputOffset = Math.ceil(bytes / align) * align;
      const region = arena.allocRegion(outputOffset + bytes);
      const intermediate: GPUBufferBinding = { buffer: region, offset: 0, size: bytes };
      const output: GPUBufferBinding = { buffer: region, offset: outputOffset, size: bytes };
      const second = arena.allocRegion(bytes);
      const relay: GPUBufferBinding = { buffer: second, offset: 0, size: bytes };

      dispatchInto({ buffer: source }, intermediate); // 1 → 3（領域の先頭区間へ）
      dispatchInto(intermediate, relay); // 3 → 7（別バッファへ）
      // 中間の生存が終わった区間へ別の値を配り直す（同じ offset 0 を出力側として再利用）。
      dispatchInto(relay, intermediate); // 7 → 15
      dispatchInto({ buffer: source }, output); // 1 → 3（同じ領域の別区間 — 配り直しに巻き込まれない）

      await scheduler.flush();

      const staging = gpu.device.createBuffer({
        size: bytes * 2,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const encoder = gpu.device.createCommandEncoder();
      encoder.copyBufferToBuffer(region, 0, staging, 0, bytes);
      encoder.copyBufferToBuffer(region, outputOffset, staging, bytes, bytes);
      gpu.device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const actual = new Float32Array<ArrayBuffer>(staging.getMappedRange().slice(0));
      staging.unmap();
      staging.destroy();

      for (let i = 0; i < count; i += 1) {
        assertEquals(actual[i], 15, `配り直した区間 index ${i}`);
        assertEquals(actual[count + i], 3, `同居する出力区間 index ${i}`);
      }
      assertEquals(arena.stats.allocCount, 3);
    } finally {
      await arena.destroy();
      gpu.destroy();
    }
  },
});
