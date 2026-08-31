// probe3a — 素 WebGPU・累積確保検証。
//
// gemma4 製品資産（models/karume-gemma4-e2b）の safetensors ヘッダから **835 本のバッファ
// サイズ列を実読み**し、そのとおりに createBuffer + writeBuffer してから、**全 835 本**を
// storage 束縛で読み直して 1 語ずつ突合する。probe2（単一バッファ 64〜640MiB）の多本化で、
// 狙いは「累積確保（実プロファイル 835 本・計 1.5GiB）で後発の確保が先発のバッファを壊す」
// 仮説 (ii) の直撃。
//
// MUST: 検証は **835 本を全て確保し終えてから** 行う（書いた直後に読むと後発確保による
// 破れが構造的に見えない）。
// MUST: 期待値はホストと WGSL で**同じ式を別々に書く**（片方から引くと恒真化する）。
// 出力は「不一致のあったバッファだけ列挙 + 総括 1 行」。
//
// 使い方: deno run -A probe3a.ts [--limit N] [--scale F]
//   --limit N : 先頭 N 本だけ（切り分け用。既定 = 全部）
//   --scale F : 各バッファのサイズを F 倍（VRAM の小さい機械での縮小再現。既定 1）

const MIRROR = "models/karume-gemma4-e2b/e2b/model";
const SHARDS = [
  `${MIRROR}/model.i4-00001-of-00003.safetensors`,
  `${MIRROR}/model.i4-00002-of-00003.safetensors`,
  `${MIRROR}/model.i4-00003-of-00003.safetensors`,
];

type Entry = { readonly name: string; readonly bytes: number };

/** safetensors ヘッダだけを読む（本体 1.5GiB は触らない）。 */
const readHeader = async (path: string): Promise<Record<string, unknown>> => {
  const file = await Deno.open(path);
  try {
    const head = new Uint8Array(8);
    await file.read(head);
    const length = Number(new DataView(head.buffer).getBigUint64(0, true));
    const body = new Uint8Array(length);
    let filled = 0;
    while (filled < length) {
      const read = await file.read(body.subarray(filled));
      if (read === null) throw new Error(`${path}: ヘッダが途中で尽きた`);
      filled += read;
    }
    return JSON.parse(new TextDecoder().decode(body));
  } finally {
    file.close();
  }
};

const collectEntries = async (): Promise<Entry[]> => {
  const entries: Entry[] = [];
  for (const path of SHARDS) {
    const header = await readHeader(path);
    for (const [name, value] of Object.entries(header)) {
      if (name === "__metadata__") continue;
      const offsets = (value as { data_offsets: [number, number] }).data_offsets;
      entries.push({ name, bytes: offsets[1] - offsets[0] });
    }
  }
  return entries;
};

/**
 * 期待値（ホスト側）。WGSL 側（下の `EXPECT_WGSL`）と**同じ式を別に書く** — 引き写しではなく
 * 手で二度書くのが恒真化の防波堤。u32 の乗算は Math.imul で 32bit に閉じる。
 */
const expectedWord = (buffer: number, index: number): number => {
  let v = (Math.imul(buffer + 1, 2654435761) + Math.imul(index + 1, 2246822519)) >>> 0;
  v = (v ^ (v >>> 15)) >>> 0;
  v = Math.imul(v, 2654435761) >>> 0;
  return (v ^ (v >>> 13)) >>> 0;
};

const VERIFY_WGSL = `
struct Params { buffer_index: u32, word_count: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> first_index: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: Params;

fn expect(buffer: u32, index: u32) -> u32 {
  var v: u32 = (buffer + 1u) * 2654435761u + (index + 1u) * 2246822519u;
  v = v ^ (v >> 15u);
  v = v * 2654435761u;
  return v ^ (v >> 13u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * 256u;
  var i = gid.x;
  loop {
    if (i >= params.word_count) { break; }
    if (data[i] != expect(params.buffer_index, i)) {
      atomicAdd(&counts[params.buffer_index], 1u);
      atomicMin(&first_index[params.buffer_index], i);
    }
    i = i + stride;
  }
}
`;

const args = Deno.args;
const limitArg = args.indexOf("--limit");
const scaleArg = args.indexOf("--scale");
const limit = limitArg >= 0 ? Number(args[limitArg + 1]) : Number.POSITIVE_INFINITY;
const scale = scaleArg >= 0 ? Number(args[scaleArg + 1]) : 1;

const all = await collectEntries();
const entries = all.slice(0, Math.min(all.length, limit)).map((e) => ({
  name: e.name,
  bytes: Math.max(4, Math.floor((e.bytes * scale) / 4) * 4),
}));
const totalBytes = entries.reduce((a, e) => a + e.bytes, 0);
console.log(
  `[probe3a] 実プロファイル ${all.length} 本中 ${entries.length} 本を確保（scale=${scale}）: ` +
    `計 ${(totalBytes / 2 ** 20).toFixed(1)} MiB / 最大 ${
      (Math.max(...entries.map((e) => e.bytes)) / 2 ** 20).toFixed(1)
    } MiB`,
);

const adapter = await navigator.gpu.requestAdapter();
if (adapter === null) throw new Error("GPUAdapter が無い");
const L = adapter.limits;
console.log(
  `[probe3a] adapter: maxBufferSize=${L.maxBufferSize} maxStorageBufferBindingSize=${L.maxStorageBufferBindingSize} ` +
    `maxStorageBuffersPerShaderStage=${L.maxStorageBuffersPerShaderStage} ` +
    `maxComputeWorkgroupStorageSize=${L.maxComputeWorkgroupStorageSize} ` +
    `maxComputeWorkgroupsPerDimension=${L.maxComputeWorkgroupsPerDimension}`,
);
const device = await adapter.requestDevice({
  requiredLimits: {
    maxBufferSize: L.maxBufferSize,
    maxStorageBufferBindingSize: Math.min(L.maxStorageBufferBindingSize, L.maxBufferSize),
    maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage,
    maxComputeInvocationsPerWorkgroup: L.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupsPerDimension: L.maxComputeWorkgroupsPerDimension,
  },
});
const errors: string[] = [];
device.addEventListener("uncapturederror", (event) => {
  errors.push(String((event as GPUUncapturedErrorEvent).error.message));
});
device.lost.then((info) => errors.push(`device lost: ${info.reason} ${info.message}`));

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
const CHUNK_WORDS = 1 << 20; // 4 MiB の staging（ホスト側の常駐を抑える）
const staging = new Uint32Array(CHUNK_WORDS);

// ── 段 1: 835 本を確保 + 書込 ───────────────────────────────────────────────
device.pushErrorScope("out-of-memory");
device.pushErrorScope("validation");
const t0 = performance.now();
const buffers: GPUBuffer[] = [];
for (const [index, entry] of entries.entries()) {
  const buffer = device.createBuffer({ size: entry.bytes, usage: STORAGE });
  buffers.push(buffer);
  const words = entry.bytes / 4;
  for (let base = 0; base < words; base += CHUNK_WORDS) {
    const n = Math.min(CHUNK_WORDS, words - base);
    for (let i = 0; i < n; i += 1) staging[i] = expectedWord(index, base + i);
    device.queue.writeBuffer(buffer, base * 4, staging, 0, n);
  }
}
// 故障注入（門が空振りしないことの証明 — 検出しない門は何も見ていない）。
const poisonArg = args.indexOf("--poison");
if (poisonArg >= 0) {
  const target = Number(args[poisonArg + 1]);
  device.queue.writeBuffer(buffers[target], 0, new Uint32Array([0xdeadbeef]));
  console.log(`[probe3a] 故障注入: #${target} の word 0 を破壊した`);
}
await device.queue.onSubmittedWorkDone();
const writeError = await device.popErrorScope();
const oomError = await device.popErrorScope();
console.log(
  `[probe3a] 確保 + 書込 完了 ${((performance.now() - t0) / 1000).toFixed(1)}s ` +
    `validation=${writeError === null ? "clean" : writeError.message} ` +
    `oom=${oomError === null ? "clean" : oomError.message}`,
);

// ── 段 2: 全本を storage 束縛で読み直して突合 ──────────────────────────────
const counts = device.createBuffer({
  size: entries.length * 4,
  usage: STORAGE | GPUBufferUsage.COPY_SRC,
});
const firstIndex = device.createBuffer({
  size: entries.length * 4,
  usage: STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});
device.queue.writeBuffer(firstIndex, 0, new Uint32Array(entries.length).fill(0xffffffff));

const module = device.createShaderModule({ code: VERIFY_WGSL });
const pipeline = device.createComputePipeline({
  layout: "auto",
  compute: { module, entryPoint: "main" },
});

device.pushErrorScope("validation");
const encoder = device.createCommandEncoder();
const pass = encoder.beginComputePass();
pass.setPipeline(pipeline);
const paramBuffers: GPUBuffer[] = [];
const maxGroups = device.limits.maxComputeWorkgroupsPerDimension;
for (const [index, entry] of entries.entries()) {
  const words = entry.bytes / 4;
  const params = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  paramBuffers.push(params);
  device.queue.writeBuffer(params, 0, new Uint32Array([index, words, 0, 0]));
  const group = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers[index] } },
      { binding: 1, resource: { buffer: counts } },
      { binding: 2, resource: { buffer: firstIndex } },
      { binding: 3, resource: { buffer: params } },
    ],
  });
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(Math.min(maxGroups, Math.max(1, Math.ceil(words / 256))));
}
pass.end();
const readCounts = device.createBuffer({
  size: entries.length * 4,
  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
});
const readFirst = device.createBuffer({
  size: entries.length * 4,
  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
});
encoder.copyBufferToBuffer(counts, 0, readCounts, 0, entries.length * 4);
encoder.copyBufferToBuffer(firstIndex, 0, readFirst, 0, entries.length * 4);
device.queue.submit([encoder.finish()]);
await device.queue.onSubmittedWorkDone();
const verifyError = await device.popErrorScope();

await readCounts.mapAsync(GPUMapMode.READ);
await readFirst.mapAsync(GPUMapMode.READ);
const countValues = new Uint32Array(readCounts.getMappedRange().slice(0));
const firstValues = new Uint32Array(readFirst.getMappedRange().slice(0));
readCounts.unmap();
readFirst.unmap();

let bad = 0;
for (const [index, entry] of entries.entries()) {
  if (countValues[index] === 0) continue;
  bad += 1;
  console.log(
    `[probe3a] MISMATCH #${index} ${entry.name} bytes=${entry.bytes} ` +
      `mismatchWords=${countValues[index]}/${entry.bytes / 4} firstWord=${firstValues[index]}`,
  );
}
console.log(
  `[probe3a] 総括: 検証 ${entries.length} 本 / 不一致 ${bad} 本 / ` +
    `verify-validation=${verifyError === null ? "clean" : verifyError.message} / ` +
    `uncaptured=${errors.length === 0 ? "none" : errors.join(" | ")}`,
);

for (const b of buffers) b.destroy();
for (const b of paramBuffers) b.destroy();
device.destroy();
Deno.exit(bad === 0 && verifyError === null && errors.length === 0 ? 0 : 1);
