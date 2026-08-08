// VRAM を占有し続けるだけの診断スクリプト（別プロセスから圧を掛けるための道具）。
//
// 用途:
// - **OOM 余裕の実測** — 「保持量を上げていって初めて落ちる線」が、その機械での実効の余裕。
//   確保天井は heapBudget 比例の動的な値なので、机上ではなく毎回この形で測る
//   （docs/research/2026-08-03-wgpu-memory-ceiling.md）。
// - **f16-1024 の OOM 誤報告の再現** — 本体を回している最中にこれを走らせると、確保失敗が
//   派生 validation（`Buffer with '' label is invalid`）として現れる経路をなぞれる
//   （機序と弁別表は docs/research/2026-08-08-vram-oom-misreport.md）。
// - **重み staging 二重計上の修正（F2）の効果測定** — 「初回ピーク + 保持量 > 天井」の線が
//   どこへ動いたかで、ピーク低減を間接に測れる。
//
// 使い方:
//   deno run -A tools/diag/hold-vram.ts [MiB]   # 既定 4608MiB
//   別端末で本体を回し、Ctrl-C で解放する。
//   例: deno run -A tools/diag/hold-vram.ts 4608
//       deno test -A packages/models/tests/e2e_anima_test.ts --filter f16

const CHUNK_MIB = 256;

const parseMib = (arg: string | undefined): number => {
  if (arg === undefined) return 4608;
  const mib = Number(arg);
  // 診断の道具でも「黙って既定に落ちる」は作らない（測った条件が記録と食い違う）。
  if (!Number.isInteger(mib) || mib <= 0) {
    throw new Error(`保持量は正の整数 MiB で指定する（受け取った値: ${arg}）`);
  }
  return mib;
};

const totalMib = parseMib(Deno.args[0]);
const adapter = await navigator.gpu.requestAdapter();
if (adapter === null) throw new Error("WebGPU アダプタが無い");
const device = await adapter.requestDevice();

// 256MiB は WebGPU 既定の maxBufferSize と同値なので、limits を要求せずに刻める。
const chunks: GPUBuffer[] = [];
let heldMib = 0;
while (heldMib < totalMib) {
  const mib = Math.min(CHUNK_MIB, totalMib - heldMib);
  device.pushErrorScope("out-of-memory");
  const buffer = device.createBuffer({
    label: `hold-${heldMib}MiB`,
    size: mib * 1024 * 1024,
    usage: GPUBufferUsage.STORAGE,
  });
  const failure = await device.popErrorScope();
  if (failure !== null) {
    throw new Error(`${heldMib}MiB まで確保して力尽きた: ${failure.message}`);
  }
  chunks.push(buffer);
  heldMib += mib;
}

console.log(`${heldMib}MiB を ${chunks.length} 本の GPUBuffer で保持中。Ctrl-C で解放する。`);
// 保持し続けるだけ（プロセスが生きている限り VRAM は返らない）。決して解決しない promise を
// await すると Deno がイベントループの枯渇を検出して即エラー終了するため、タイマで生かす。
setInterval(() => {}, 1 << 30);
