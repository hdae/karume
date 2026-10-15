// timestamp-query の 1 tick の実測（環境キーごとの TIMESTAMP_UNIT_NS の行を足す前に回す）。
// 使い方: deno run --unstable-webgpu -A tools/diag/timestamp-period-probe.ts
// 読み方: raw（tick）が「壁時計 − submit の固定費（空の submit の壁時計・RTX 3080 Ti で約 11 ms）」と一致すれば 1 tick = 1 ns。
// 2026-10-06 RTX 3080 Ti: 133〜153 ms の pass で raw ≈ 壁 − 11 ms（1% 以内）→ 行は 1。
// period の実測（重い pass で壁時計と比べる）+ query set の使い回しで 0 になるかの切り分け。
const adapter = (await navigator.gpu.requestAdapter({ powerPreference: "high-performance" }))!;
const device = await adapter.requestDevice({ requiredFeatures: ["timestamp-query"] });
const N = 1 << 22;
const buf = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE });
const wgsl = `@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) { var x = f32(gid.x) * 1e-3; for (var i = 0u; i < 40000u; i++) { x = sin(x) * 1.0001 + 0.1; } data[gid.x] = x; }`;
const pipeline = device.createComputePipeline({
  layout: "auto",
  compute: { module: device.createShaderModule({ code: wgsl }), entryPoint: "main" },
});
const bg = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [{ binding: 0, resource: { buffer: buf } }],
});
const run = async (qs: GPUQuerySet, label: string) => {
  const qbuf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const rbuf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  device.pushErrorScope("validation");
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass({
    timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(N / 256);
  pass.end();
  enc.resolveQuerySet(qs, 0, 2, qbuf, 0);
  enc.copyBufferToBuffer(qbuf, 0, rbuf, 0, 16);
  const t0 = performance.now();
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  const wall = performance.now() - t0;
  const verr = await device.popErrorScope();
  if (verr) console.log("VALIDATION:", verr.message);
  await rbuf.mapAsync(GPUMapMode.READ);
  const ts = new BigUint64Array(rbuf.getMappedRange().slice(0));
  rbuf.unmap();
  const raw = Number(ts[1] - ts[0]);
  console.log(
    `${label}: raw=${raw} (${(raw / 1e6).toFixed(2)} ms if 1 tick = 1 ns)  wall=${
      wall.toFixed(2)
    } ms  raw_ns/wall=${(raw / (wall * 1e6)).toFixed(4)}`,
  );
  qbuf.destroy();
  rbuf.destroy();
};
// 使い捨て（runtime の #encodeTimedChunk と同じ: submit ごとに新しい query set）
for (let i = 0; i < 3; i++) {
  await run(device.createQuerySet({ type: "timestamp", count: 2 }), `fresh#${i}`);
}
// 使い回し（同じ query set を submit をまたいで使う）
const shared = device.createQuerySet({ type: "timestamp", count: 2 });
for (let i = 0; i < 3; i++) await run(shared, `reused#${i}`);
device.destroy();
