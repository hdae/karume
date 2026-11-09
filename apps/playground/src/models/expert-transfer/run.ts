/**
 * The expert transfer bench: what an exact expert cache pays per miss round on this browser — the
 * host round trip of a fence and of a small readback, and the upload of a batch of experts by a
 * staging ring (`MAP_WRITE | COPY_SRC` slots, mapped ahead, then `copyBufferToBuffer`) or by
 * `queue.writeBuffer`. Every sample is wall-clock time from the first host call to the resolved
 * promise (`onSubmittedWorkDone` or `mapAsync`), so it is what a decode loop would wait.
 */

import { EXPERT_BYTES, type Summary, summarize, type TransferCase } from "./stats";

export type TransferCaseResult = {
  readonly id: string;
  readonly kind: TransferCase["kind"];
  readonly method?: "ring" | "writeBuffer";
  readonly experts?: number;
  readonly bytes: number;
  /** Host wall ms per sample, in run order (warm-up excluded). */
  readonly samples: readonly number[];
  readonly ms: Summary;
  /** Ring only: ms to map every slot again after the copy (off the critical path in a decode loop). */
  readonly remapSamples?: readonly number[];
  readonly remapMs?: Summary;
};

export type TransferBenchOptions = {
  readonly cases: readonly TransferCase[];
  readonly samples: number;
  readonly warmup: number;
  readonly signal: AbortSignal;
  readonly onProgress: (progress: {
    readonly caseId: string;
    readonly done: number;
    readonly total: number;
  }) => void;
};

/** One expert's worth of bytes per slot, filled once with xorshift32 (the contents do not matter). */
export const makeSources = (count: number): Uint8Array<ArrayBuffer>[] => {
  let state = 0x9e3779b9;
  return Array.from({ length: count }, () => {
    const words = new Uint32Array(EXPERT_BYTES / 4);
    for (let index = 0; index < words.length; index += 1) {
      state ^= state << 13;
      state ^= state >>> 17;
      state ^= state << 5;
      words[index] = state >>> 0;
    }
    return new Uint8Array(words.buffer);
  });
};

/**
 * Runs `body` under validation and out-of-memory scopes. A GPU error the scopes caught is thrown in
 * preference to `body`'s own rejection (that is usually its consequence); the scopes are popped
 * either way.
 */
export const scoped = async <T>(
  device: GPUDevice,
  where: string,
  body: () => Promise<T>,
): Promise<T> => {
  device.pushErrorScope("validation");
  device.pushErrorScope("out-of-memory");
  const outcome = await body().then(
    (value) => ({ ok: true as const, value }),
    (error: unknown) => ({ ok: false as const, error }),
  );
  const outOfMemory = await device.popErrorScope();
  const validation = await device.popErrorScope();
  const gpuError = outOfMemory ?? validation;
  if (gpuError !== null) throw new Error(`${where}: ${gpuError.message}`);
  if (!outcome.ok) throw outcome.error;
  return outcome.value;
};

/**
 * One sample function per case: returns the critical-path ms (and, for the ring, the remap ms).
 * The buffers live for the whole case and are destroyed by the returned `dispose`.
 */
type Sampler = {
  readonly sample: () => Promise<{ ms: number; remapMs?: number }>;
  readonly dispose: () => void;
};

const fenceSampler = (device: GPUDevice, copy: boolean): Sampler => {
  const source = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_SRC });
  const target = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST });
  return {
    sample: async () => {
      const started = performance.now();
      const encoder = device.createCommandEncoder();
      if (copy) encoder.copyBufferToBuffer(source, 0, target, 0, 4);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      return { ms: performance.now() - started };
    },
    dispose: () => {
      source.destroy();
      target.destroy();
    },
  };
};

const readbackSampler = (device: GPUDevice, bytes: number): Sampler => {
  const source = device.createBuffer({
    size: bytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const target = device.createBuffer({
    size: bytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  return {
    sample: async () => {
      const started = performance.now();
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(source, 0, target, 0, bytes);
      device.queue.submit([encoder.finish()]);
      await target.mapAsync(GPUMapMode.READ);
      // Read what a decode loop would read (the ids), then hand the buffer back.
      new Uint32Array(target.getMappedRange()).slice();
      target.unmap();
      return { ms: performance.now() - started };
    },
    dispose: () => {
      source.destroy();
      target.destroy();
    },
  };
};

const uploadSampler = (
  device: GPUDevice,
  method: "ring" | "writeBuffer",
  sources: readonly Uint8Array<ArrayBuffer>[],
): Sampler => {
  const lanes = sources.map((source) => ({
    source,
    target: device.createBuffer({
      size: EXPERT_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
    slot:
      method === "ring"
        ? device.createBuffer({
            size: EXPERT_BYTES,
            usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true,
          })
        : undefined,
  }));
  const dispose = () => {
    for (const { target, slot } of lanes) {
      target.destroy();
      slot?.destroy();
    }
  };
  if (method === "writeBuffer") {
    return {
      sample: async () => {
        const started = performance.now();
        for (const { source, target } of lanes) device.queue.writeBuffer(target, 0, source);
        await device.queue.onSubmittedWorkDone();
        return { ms: performance.now() - started };
      },
      dispose,
    };
  }
  const slots = lanes.flatMap(({ slot }) => (slot === undefined ? [] : [slot]));
  return {
    sample: async () => {
      const started = performance.now();
      const encoder = device.createCommandEncoder();
      for (const { source, target, slot } of lanes) {
        if (slot === undefined) continue;
        new Uint8Array(slot.getMappedRange()).set(source);
        slot.unmap();
        encoder.copyBufferToBuffer(slot, 0, target, 0, EXPERT_BYTES);
      }
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      const copied = performance.now();
      // The slots are mapped again for the next sample; a decode loop would do this ahead of need.
      await Promise.all(slots.map((slot) => slot.mapAsync(GPUMapMode.WRITE)));
      return { ms: copied - started, remapMs: performance.now() - copied };
    },
    dispose,
  };
};

const samplerFor = (
  device: GPUDevice,
  testCase: TransferCase,
  sources: readonly Uint8Array<ArrayBuffer>[],
): Sampler => {
  switch (testCase.kind) {
    case "fence-empty":
      return fenceSampler(device, false);
    case "fence-copy":
      return fenceSampler(device, true);
    case "readback":
      return readbackSampler(device, testCase.bytes);
    case "upload":
      return uploadSampler(device, testCase.method, sources.slice(0, testCase.experts));
  }
};

/**
 * Runs the cases one after another, `warmup` + `samples` samples each, back to back. An abort ends
 * the run before the next sample with the signal's reason (the results so far are not returned).
 */
export const runTransferBench = async (
  device: GPUDevice,
  options: TransferBenchOptions,
): Promise<TransferCaseResult[]> => {
  const { cases, samples, warmup, signal, onProgress } = options;
  const experts = Math.max(0, ...cases.map((item) => (item.kind === "upload" ? item.experts : 0)));
  const sources = makeSources(experts);
  const results: TransferCaseResult[] = [];
  for (const testCase of cases) {
    const result = await scoped(device, testCase.id, async () => {
      const sampler = samplerFor(device, testCase, sources);
      try {
        const times: number[] = [];
        const remaps: number[] = [];
        for (let index = 0; index < warmup + samples; index += 1) {
          signal.throwIfAborted();
          const { ms, remapMs } = await sampler.sample();
          if (index >= warmup) {
            times.push(ms);
            if (remapMs !== undefined) remaps.push(remapMs);
          }
          onProgress({ caseId: testCase.id, done: index + 1, total: warmup + samples });
        }
        return {
          id: testCase.id,
          kind: testCase.kind,
          ...(testCase.kind === "upload"
            ? { method: testCase.method, experts: testCase.experts }
            : {}),
          bytes: testCase.bytes,
          samples: times,
          ms: summarize(times),
          ...(remaps.length === 0 ? {} : { remapSamples: remaps, remapMs: summarize(remaps) }),
        };
      } finally {
        sampler.dispose();
      }
    });
    results.push(result);
  }
  return results;
};
