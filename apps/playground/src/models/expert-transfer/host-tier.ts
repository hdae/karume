/**
 * The host tier measurement (R0 of the expert cache's RAM tier, ADR 0146): for each form of the tier
 * — plain `ArrayBuffer`s (A, `"array"`) or `MAP_WRITE | COPY_SRC` GPUBuffers filled at creation and
 * never mapped again (B, `"upload-buffer"`) — how many bytes this tab can hold, and what one hit
 * costs to bring experts from the tier into a VRAM pool. A form allocates its chunks up to the
 * target (stopping at the first failure, keeping what it holds), times the empty fence and the hits
 * with the tier held, then releases the tier before the next form: B by `destroy()`, A by dropping
 * its references (the garbage collector frees it later).
 *
 * Every hit sample is wall-clock time from the first host call to `onSubmittedWorkDone`, as in the
 * transfer bench. A device lost during the run ends the GPU steps and is reported in the result;
 * A's allocation still runs (it does not need the device).
 */

import {
  allocationPlan,
  drawDistinct,
  type ExpertPlace,
  fillByRepeating,
  HIT_BATCHES,
  HIT_METHODS,
  type HitMethod,
  HOST_TIER_CHUNK_BYTES,
  HOST_TIER_FORMS,
  type HostTierForm,
  leastSquaresSlope,
  perExpertMs,
  placeExperts,
  POOL_EXPERTS,
} from "./host-tier-plan";
import { makeSources, runTransferBench, scoped } from "./run";
import { EXPERT_BYTES, gigabytesPerSecond, type Summary, summarize } from "./stats";

export type AllocatedChunk = {
  readonly bytes: number;
  /** ms of the allocation: `new Uint8Array` (A), `createBuffer` + `getMappedRange` (B). */
  readonly allocateMs: number;
  /** ms of writing every byte (one expert repeated with `set`). */
  readonly fillMs: number;
  /** B only: ms of the `unmap()` call. */
  readonly unmapMs?: number;
  /** B only: ms from the `unmap()` until the error scopes popped (a round trip to the GPU process). */
  readonly scopeMs?: number;
};

export type HostTierHit = {
  readonly id: string;
  readonly method: HitMethod;
  readonly experts: number;
  readonly bytes: number;
  /** Host wall ms per sample, in run order (warm-up excluded). */
  readonly samples: readonly number[];
  readonly ms: Summary;
  /** (median − the form's empty-fence median) / experts. */
  readonly perExpertMs: number;
  /** Ring only: ms to map the used slots again after the copy (off the critical path). */
  readonly remapSamples?: readonly number[];
  readonly remapMs?: Summary;
};

export type HostTierFormResult = {
  readonly form: HostTierForm;
  readonly targetBytes: number;
  readonly chunkBytes: number;
  /** The chunks held, in allocation order. */
  readonly chunks: readonly AllocatedChunk[];
  readonly heldBytes: number;
  readonly reached: boolean;
  /** Why the allocation stopped short of the target (the exception's or GPU error's message). */
  readonly failure?: string;
  /** Bytes held / the sum of the fill ms, GB/s. */
  readonly fillGBps?: number;
  /** Whole experts the held chunks place (none straddles two chunks). */
  readonly experts: number;
  /** The empty submit + `onSubmittedWorkDone`, timed with the tier held. */
  readonly fence?: { readonly samples: readonly number[]; readonly ms: Summary };
  readonly hits: readonly HostTierHit[];
  /** Least-squares slope of the batch medians over the experts per batch, ms per expert. */
  readonly slopes: readonly { readonly method: HitMethod; readonly msPerExpert: number }[];
  /** Why some or all hit cases did not run. */
  readonly hitsSkipped?: string;
};

export type HostTierResult = {
  readonly settings: {
    readonly forms: readonly HostTierForm[];
    readonly targetBytes: number;
    readonly batches: readonly number[];
    readonly poolExperts: number;
  };
  readonly forms: readonly HostTierFormResult[];
  /** The device was lost during the run: where it was noticed and what the device said. */
  readonly deviceLost?: {
    readonly during: string;
    readonly reason: string;
    readonly message: string;
  };
};

export type HostTierOptions = {
  readonly forms: readonly HostTierForm[];
  readonly targetBytes: number;
  readonly samples: number;
  readonly warmup: number;
  readonly signal: AbortSignal;
  readonly onProgress: (progress: {
    readonly stage: string;
    readonly done: number;
    readonly total: number;
    readonly unit: "chunks" | "samples";
    readonly note?: string;
  }) => void;
};

const FORM_LABEL: Readonly<Record<HostTierForm, string>> = {
  "upload-buffer": "B upload-buffer",
  array: "A array",
};

const messageOf = (error: unknown): string =>
  error instanceof Error ? `${error.name}: ${error.message}` : String(error);

const gigabytes = (bytes: number): string => `${(bytes / 1e9).toFixed(2)} GB held`;

/** The tier's chunks, owned by the form that runs: it releases them however the form ends. */
type Tier = { readonly arrays: Uint8Array<ArrayBuffer>[]; readonly buffers: GPUBuffer[] };

type Allocation = { readonly chunks: AllocatedChunk[]; readonly failure?: string };

/** Watches the device for a loss; `check` records the first step that saw it. */
const lossWatch = (device: GPUDevice) => {
  let info: GPUDeviceLostInfo | undefined;
  let seen: HostTierResult["deviceLost"];
  void device.lost.then((lost) => {
    info = lost;
  });
  return {
    check: (during: string): boolean => {
      if (info === undefined) return false;
      seen ??= { during, reason: info.reason, message: info.message };
      return true;
    },
    seen: () => seen,
  };
};

type LossWatch = ReturnType<typeof lossWatch>;

const allocateArrays = (
  sizes: readonly number[],
  expert: Uint8Array<ArrayBuffer>,
  tier: Tier,
  options: HostTierOptions,
): Allocation => {
  const chunks: AllocatedChunk[] = [];
  let held = 0;
  for (const [index, size] of sizes.entries()) {
    options.signal.throwIfAborted();
    options.onProgress({
      stage: `${FORM_LABEL.array}: allocating chunk ${index + 1} / ${sizes.length}`,
      done: index,
      total: sizes.length,
      unit: "chunks",
      note: gigabytes(held),
    });
    try {
      const started = performance.now();
      const chunk = new Uint8Array(size);
      const allocated = performance.now();
      fillByRepeating(chunk, expert);
      const filled = performance.now();
      tier.arrays.push(chunk);
      chunks.push({ bytes: size, allocateMs: allocated - started, fillMs: filled - allocated });
      held += size;
    } catch (error) {
      return { chunks, failure: `chunk ${index + 1}: ${messageOf(error)}` };
    }
  }
  return { chunks };
};

const allocateUploadBuffers = async (
  device: GPUDevice,
  sizes: readonly number[],
  expert: Uint8Array<ArrayBuffer>,
  tier: Tier,
  loss: LossWatch,
  options: HostTierOptions,
): Promise<Allocation> => {
  const label = FORM_LABEL["upload-buffer"];
  const chunks: AllocatedChunk[] = [];
  let held = 0;
  for (const [index, size] of sizes.entries()) {
    options.signal.throwIfAborted();
    const where = `${label}: chunk ${index + 1}`;
    if (loss.check(`${where}, before the allocation`)) {
      return { chunks, failure: `${where}: the device was lost` };
    }
    options.onProgress({
      stage: `${label}: allocating chunk ${index + 1} / ${sizes.length}`,
      done: index,
      total: sizes.length,
      unit: "chunks",
      note: gigabytes(held),
    });
    // The buffer is kept outside the scoped body so that a failed chunk is destroyed too.
    const attempt: { buffer?: GPUBuffer } = {};
    try {
      const timing = await scoped(device, `chunk ${index + 1}`, async () => {
        const started = performance.now();
        const buffer = device.createBuffer({
          size,
          usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
          mappedAtCreation: true,
        });
        attempt.buffer = buffer;
        const range = new Uint8Array(buffer.getMappedRange());
        const allocated = performance.now();
        fillByRepeating(range, expert);
        const filled = performance.now();
        buffer.unmap();
        const unmapped = performance.now();
        return {
          buffer,
          allocateMs: allocated - started,
          fillMs: filled - allocated,
          unmapMs: unmapped - filled,
          unmapped,
        };
      });
      const scopeMs = performance.now() - timing.unmapped;
      if (loss.check(`${where}, after the allocation`)) {
        timing.buffer.destroy();
        return { chunks, failure: `${where}: the device was lost` };
      }
      tier.buffers.push(timing.buffer);
      chunks.push({
        bytes: size,
        allocateMs: timing.allocateMs,
        fillMs: timing.fillMs,
        unmapMs: timing.unmapMs,
        scopeMs,
      });
      held += size;
    } catch (error) {
      attempt.buffer?.destroy();
      loss.check(`${where}, failed`);
      return { chunks, failure: `chunk ${index + 1}: ${messageOf(error)}` };
    }
  }
  return { chunks };
};

/** One hit sample over the experts at `places` (pool slot = index in `places`). */
type HitSampler = {
  readonly sample: (places: readonly ExpertPlace[]) => Promise<{ ms: number; remapMs?: number }>;
  readonly dispose: () => void;
};

const at = <T>(items: readonly T[], index: number, what: string): T => {
  const item = items[index];
  if (item === undefined) throw new Error(`${what} ${index} does not exist`);
  return item;
};

const hitSampler = (device: GPUDevice, method: HitMethod, tier: Tier): HitSampler => {
  const pool = device.createBuffer({
    size: POOL_EXPERTS * EXPERT_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const views = (places: readonly ExpertPlace[]) =>
    places.map(({ chunk, offset }) =>
      at(tier.arrays, chunk, "array chunk").subarray(offset, offset + EXPERT_BYTES),
    );
  if (method === "copy") {
    return {
      sample: async (places) => {
        const sources = places.map(({ chunk, offset }) => ({
          buffer: at(tier.buffers, chunk, "upload-buffer chunk"),
          offset,
        }));
        const started = performance.now();
        const encoder = device.createCommandEncoder();
        sources.forEach(({ buffer, offset }, slot) =>
          encoder.copyBufferToBuffer(buffer, offset, pool, slot * EXPERT_BYTES, EXPERT_BYTES),
        );
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        return { ms: performance.now() - started };
      },
      dispose: () => pool.destroy(),
    };
  }
  if (method === "writeBuffer") {
    return {
      sample: async (places) => {
        const sources = views(places);
        const started = performance.now();
        sources.forEach((source, slot) =>
          device.queue.writeBuffer(pool, slot * EXPERT_BYTES, source),
        );
        await device.queue.onSubmittedWorkDone();
        return { ms: performance.now() - started };
      },
      dispose: () => pool.destroy(),
    };
  }
  const slots = Array.from({ length: POOL_EXPERTS }, () =>
    device.createBuffer({
      size: EXPERT_BYTES,
      usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    }),
  );
  return {
    sample: async (places) => {
      const sources = views(places);
      const used = slots.slice(0, sources.length);
      const started = performance.now();
      const encoder = device.createCommandEncoder();
      sources.forEach((source, index) => {
        const slot = at(used, index, "ring slot");
        new Uint8Array(slot.getMappedRange()).set(source);
        slot.unmap();
        encoder.copyBufferToBuffer(slot, 0, pool, index * EXPERT_BYTES, EXPERT_BYTES);
      });
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      const copied = performance.now();
      // Only the used slots were unmapped; a decode loop would map them again ahead of need.
      await Promise.all(used.map((slot) => slot.mapAsync(GPUMapMode.WRITE)));
      return { ms: copied - started, remapMs: performance.now() - copied };
    },
    dispose: () => {
      pool.destroy();
      for (const slot of slots) slot.destroy();
    },
  };
};

/** Thrown inside a hit case when the device is lost, to end the GPU steps without failing the run. */
class DeviceLostDuringRun extends Error {}

type Hits = Pick<HostTierFormResult, "fence" | "hits" | "slopes" | "hitsSkipped">;

const runHits = async (
  device: GPUDevice,
  form: HostTierForm,
  tier: Tier,
  places: readonly ExpertPlace[],
  loss: LossWatch,
  options: HostTierOptions,
): Promise<Hits> => {
  const { samples, warmup, signal, onProgress } = options;
  const label = FORM_LABEL[form];
  const hits: HostTierHit[] = [];
  const finish = (fence: Hits["fence"], hitsSkipped?: string): Hits => ({
    ...(fence === undefined ? {} : { fence }),
    hits,
    slopes: HIT_METHODS[form].flatMap((method) => {
      const msPerExpert = leastSquaresSlope(
        hits
          .filter((hit) => hit.method === method)
          .map((hit) => ({ x: hit.experts, y: hit.ms.median })),
      );
      return msPerExpert === undefined ? [] : [{ method, msPerExpert }];
    }),
    ...(hitsSkipped === undefined ? {} : { hitsSkipped }),
  });
  if (places.length === 0) return finish(undefined, "the tier holds no whole expert");
  if (loss.check(`${label}: before the empty fence`))
    return finish(undefined, "the device was lost");

  const [fenceCase] = await runTransferBench(device, {
    cases: [{ id: `${form}-fence-empty`, kind: "fence-empty", bytes: 0 }],
    samples,
    warmup,
    signal,
    onProgress: ({ done, total }) =>
      onProgress({ stage: `${label}: empty fence`, done, total, unit: "samples" }),
  });
  if (fenceCase === undefined) throw new Error("the empty fence returned no result");
  const fence = { samples: fenceCase.samples, ms: fenceCase.ms };
  if (loss.check(`${label}: the empty fence`)) return finish(fence, "the device was lost");

  const runnable = HIT_BATCHES.filter((experts) => experts <= places.length);
  const tooLarge =
    runnable.length === HIT_BATCHES.length
      ? undefined
      : `batches above ${places.length} experts skipped: the tier holds only ${places.length} experts`;
  for (const experts of runnable) {
    for (const method of HIT_METHODS[form]) {
      const id = `${form}-${method}-${experts}`;
      const stage = `${label}: ${method}, ${experts} expert${experts === 1 ? "" : "s"}`;
      try {
        const hit = await scoped(device, id, async () => {
          const sampler = hitSampler(device, method, tier);
          try {
            const times: number[] = [];
            const remaps: number[] = [];
            for (let index = 0; index < warmup + samples; index += 1) {
              signal.throwIfAborted();
              if (loss.check(stage)) throw new DeviceLostDuringRun(stage);
              // The same seed for every method and form: the same expert indices per sample.
              const picks = drawDistinct(experts, places.length, experts * 0x1_0000 + index + 1);
              const { ms, remapMs } = await sampler.sample(
                picks.map((pick) => at(places, pick, "expert")),
              );
              if (index >= warmup) {
                times.push(ms);
                if (remapMs !== undefined) remaps.push(remapMs);
              }
              onProgress({ stage, done: index + 1, total: warmup + samples, unit: "samples" });
            }
            if (loss.check(stage)) throw new DeviceLostDuringRun(stage);
            const summary = summarize(times);
            return {
              id,
              method,
              experts,
              bytes: experts * EXPERT_BYTES,
              samples: times,
              ms: summary,
              perExpertMs: perExpertMs(summary.median, fence.ms.median, experts),
              ...(remaps.length === 0 ? {} : { remapSamples: remaps, remapMs: summarize(remaps) }),
            };
          } finally {
            sampler.dispose();
          }
        });
        hits.push(hit);
      } catch (error) {
        // A rejection on a lost device (a remap, say) is the loss, not a failure of its own.
        if (error instanceof DeviceLostDuringRun || loss.check(stage)) {
          return finish(fence, `the device was lost during ${id} (that case is dropped)`);
        }
        throw error;
      }
    }
  }
  return finish(fence, tooLarge);
};

const runForm = async (
  device: GPUDevice,
  form: HostTierForm,
  expert: Uint8Array<ArrayBuffer>,
  loss: LossWatch,
  options: HostTierOptions,
): Promise<HostTierFormResult> => {
  const { targetBytes } = options;
  const chunkBytes =
    form === "upload-buffer"
      ? Math.min(HOST_TIER_CHUNK_BYTES[form], Math.floor(device.limits.maxBufferSize / 4) * 4)
      : HOST_TIER_CHUNK_BYTES[form];
  const sizes = allocationPlan(targetBytes, chunkBytes);
  const tier: Tier = { arrays: [], buffers: [] };
  try {
    const { chunks, failure } =
      form === "array"
        ? allocateArrays(sizes, expert, tier, options)
        : await allocateUploadBuffers(device, sizes, expert, tier, loss, options);
    const heldBytes = chunks.reduce((sum, { bytes }) => sum + bytes, 0);
    const fillGBps = gigabytesPerSecond(
      heldBytes,
      chunks.reduce((sum, { fillMs }) => sum + fillMs, 0),
    );
    const places = placeExperts(chunks.map(({ bytes }) => bytes));
    const hits = await runHits(device, form, tier, places, loss, options);
    return {
      form,
      targetBytes,
      chunkBytes,
      chunks,
      heldBytes,
      reached: heldBytes === targetBytes,
      ...(failure === undefined ? {} : { failure }),
      ...(fillGBps === undefined ? {} : { fillGBps }),
      experts: places.length,
      ...hits,
    };
  } finally {
    // B is freed now; A only loses its last references here (the GC reclaims it when it runs).
    for (const buffer of tier.buffers) buffer.destroy();
    tier.buffers.length = 0;
    tier.arrays.length = 0;
  }
};

/**
 * Runs the selected forms in {@link HOST_TIER_FORMS} order, one after another, each released before
 * the next. An abort ends the run between chunks or samples with the signal's reason (the tier is
 * released; the results so far are not returned).
 */
export const runHostTier = async (
  device: GPUDevice,
  options: HostTierOptions,
): Promise<HostTierResult> => {
  const [expert] = makeSources(1);
  if (expert === undefined) throw new Error("no expert pattern to fill the tier with");
  const loss = lossWatch(device);
  const selected = HOST_TIER_FORMS.filter((form) => options.forms.includes(form));
  const forms: HostTierFormResult[] = [];
  for (const form of selected) forms.push(await runForm(device, form, expert, loss, options));
  const deviceLost = loss.seen();
  return {
    settings: {
      forms: selected,
      targetBytes: options.targetBytes,
      batches: HIT_BATCHES,
      poolExperts: POOL_EXPERTS,
    },
    forms,
    ...(deviceLost === undefined ? {} : { deviceLost }),
  };
};
