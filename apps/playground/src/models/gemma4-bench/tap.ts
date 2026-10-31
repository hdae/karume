/**
 * A GPU tap for the decode bench: a `GPU` to pass to `acquireGpu({ gpu })` whose devices count the
 * compute passes, dispatches and submits the product encodes, and time every compute pass with a
 * pair of pass-boundary timestamps — in the product's own pass structure.
 *
 * Why not karume's own timing mode: a device acquired with `gpuTiming: true` encodes one dispatch
 * per pass (ADR 0021), which adds about 4 µs per dispatch and changes what is measured — QAT runs
 * about 210 more dispatches per token than regular Gemma 4, so the mode would bias the very A/B
 * this page exists for. The tap requests `timestamp-query` itself (when the adapter lists it) and
 * hides it from `device.features`, so the runtime sees a device without timing
 * (`gpuTimingEnabled` false) and keeps its plain 1-chunk-1-pass encode; the tap then adds
 * `timestampWrites` to each compute pass and resolves the pair right after the pass.
 *
 * Each device gets its own probe ({@link GpuTap.probe}, keyed by the device): the page's GPU can be
 * acquired again (after a device loss) while an older acquisition is still settling.
 *
 * A disabled probe passes everything straight through (one flag check per command encoder and per
 * submit), so a tap-off arm measures the wall clock of the same device without the tap's per-pass
 * and per-dispatch work.
 *
 * Plain TypeScript over the WebGPU API (no runtime import), so `vp test` drives it with a fake GPU.
 */

/** WebGPU usage flags (literal values: WebGPU globals are not read at module scope). */
const BUFFER_MAP_READ = 0x0001;
const BUFFER_COPY_SRC = 0x0004;
const BUFFER_COPY_DST = 0x0008;
const BUFFER_QUERY_RESOLVE = 0x0200;
const MAP_MODE_READ = 0x0001;

const TIMESTAMP_QUERY = "timestamp-query";
/** Queries in the set (two per pass; reused round-robin — each pair is resolved right after its pass). */
const QUERY_COUNT = 4096;
/** Bytes per resolved pair (`resolveQuerySet`'s destination offset must be a multiple of 256). */
const SLOT_BYTES = 256;
/** Timed passes between two resets; later passes are counted but untimed. */
export const TIMED_PASS_SLOTS = 16384;

/** One compute pass the product encoded while the probe was enabled. */
export type TapPass = {
  /** Its resolve slot; -1 = untimed (no timestamp-query, or past {@link TIMED_PASS_SLOTS}). */
  readonly slot: number;
  dispatches: number;
  /** Pass-boundary timestamps (ns), filled by {@link DeviceProbe.readTimes}. */
  beginNs?: bigint;
  endNs?: bigint;
};

/** The tap's view of one device. */
export type DeviceProbe = {
  /** Whether this device times passes (the adapter listed timestamp-query). */
  readonly timestamps: boolean;
  /** On = record and time; off = pass through. Applies to command encoders created from now on. */
  enabled: boolean;
  /** The passes recorded since the last {@link DeviceProbe.reset}, in encode order. */
  readonly passes: () => readonly TapPass[];
  /** Submits recorded since the last reset. */
  readonly submits: () => number;
  /**
   * Copies every pending timestamp pair back and fills the passes' `beginNs` / `endNs`. Call it after
   * the measured work (it submits a copy of its own, which is not recorded).
   */
  readonly readTimes: () => Promise<void>;
  /** Drops the records (the query set and the resolve buffer stay). */
  readonly reset: () => void;
};

export type GpuTap = {
  /** Pass this to `acquireGpu({ gpu })`. */
  readonly gpu: GPU;
  /** The probe of a device acquired through {@link GpuTap.gpu}; `undefined` for any other device. */
  readonly probe: (device: GPUDevice) => DeviceProbe | undefined;
};

/** Hides timestamp-query from the device's feature list (the runtime then encodes without timing). */
const hideTimestampFeature = (device: GPUDevice): void => {
  const visible = new Set([...device.features].filter((feature) => feature !== TIMESTAMP_QUERY));
  Object.defineProperty(device, "features", { value: visible, configurable: true });
};

const instrument = (device: GPUDevice, timestamps: boolean): DeviceProbe => {
  const querySet = timestamps
    ? device.createQuerySet({ type: "timestamp", count: QUERY_COUNT })
    : undefined;
  const resolveBuffer = timestamps
    ? device.createBuffer({
        size: TIMED_PASS_SLOTS * SLOT_BYTES,
        usage: BUFFER_QUERY_RESOLVE | BUFFER_COPY_SRC,
      })
    : undefined;
  let passes: TapPass[] = [];
  let submits = 0;
  let nextSlot = 0;

  const createEncoder = device.createCommandEncoder.bind(device);
  const queue = device.queue;
  const submit = queue.submit.bind(queue);

  const probe: DeviceProbe = {
    timestamps,
    enabled: false,
    passes: () => passes,
    submits: () => submits,
    readTimes: async () => {
      if (resolveBuffer === undefined) return;
      const pending = passes.filter((pass) => pass.slot >= 0 && pass.beginNs === undefined);
      if (pending.length === 0) return;
      const bytes = (Math.max(...pending.map(({ slot }) => slot)) + 1) * SLOT_BYTES;
      const staging = device.createBuffer({
        size: bytes,
        usage: BUFFER_MAP_READ | BUFFER_COPY_DST,
      });
      try {
        const encoder = createEncoder();
        encoder.copyBufferToBuffer(resolveBuffer, 0, staging, 0, bytes);
        submit([encoder.finish()]);
        await staging.mapAsync(MAP_MODE_READ);
        const view = new BigUint64Array(staging.getMappedRange().slice(0));
        staging.unmap();
        for (const pass of pending) {
          pass.beginNs = view[(pass.slot * SLOT_BYTES) / 8];
          pass.endNs = view[(pass.slot * SLOT_BYTES) / 8 + 1];
        }
      } finally {
        staging.destroy();
      }
    },
    reset: () => {
      passes = [];
      submits = 0;
      nextSlot = 0;
    },
  };

  queue.submit = (buffers) => {
    if (probe.enabled) submits += 1;
    return submit(buffers);
  };

  device.createCommandEncoder = (descriptor) => {
    const encoder = createEncoder(descriptor);
    if (!probe.enabled) return encoder;
    const beginComputePass = encoder.beginComputePass.bind(encoder);
    encoder.beginComputePass = (passDescriptor) => {
      // A pass that writes its own timestamps keeps them (the product's timing mode is off, so
      // none does today).
      const timed =
        querySet !== undefined &&
        passDescriptor?.timestampWrites === undefined &&
        nextSlot < TIMED_PASS_SLOTS;
      const slot = timed ? nextSlot++ : -1;
      const record: TapPass = { slot, dispatches: 0 };
      passes.push(record);
      const first = (slot * 2) % QUERY_COUNT;
      const pass = beginComputePass(
        timed
          ? {
              ...passDescriptor,
              timestampWrites: {
                querySet,
                beginningOfPassWriteIndex: first,
                endOfPassWriteIndex: first + 1,
              },
            }
          : passDescriptor,
      );
      const dispatchWorkgroups = pass.dispatchWorkgroups.bind(pass);
      pass.dispatchWorkgroups = (x, y, z) => {
        record.dispatches += 1;
        return dispatchWorkgroups(x, y, z);
      };
      const end = pass.end.bind(pass);
      pass.end = () => {
        const ended = end();
        if (timed && resolveBuffer !== undefined) {
          encoder.resolveQuerySet(querySet, first, 2, resolveBuffer, slot * SLOT_BYTES);
        }
        return ended;
      };
      return pass;
    };
    return encoder;
  };

  return probe;
};

/**
 * Wraps `inner` (the browser's `navigator.gpu`). Nothing is touched until `acquireGpu` requests an
 * adapter and a device through the wrapper; `inner` itself is never modified.
 */
export const createGpuTap = (inner: GPU): GpuTap => {
  const probes = new WeakMap<GPUDevice, DeviceProbe>();

  const wrapAdapter = (adapter: GPUAdapter): GPUAdapter => {
    const requestDevice = adapter.requestDevice.bind(adapter);
    const timestamps = adapter.features.has(TIMESTAMP_QUERY);
    adapter.requestDevice = async (descriptor) => {
      const features = [...(descriptor?.requiredFeatures ?? [])];
      if (timestamps && !features.includes(TIMESTAMP_QUERY)) features.push(TIMESTAMP_QUERY);
      const device = await requestDevice({ ...descriptor, requiredFeatures: features });
      probes.set(device, instrument(device, timestamps));
      hideTimestampFeature(device);
      return device;
    };
    return adapter;
  };

  const gpu: GPU = {
    __brand: "GPU",
    requestAdapter: async (options) => {
      const adapter = await inner.requestAdapter(options);
      return adapter === null ? null : wrapAdapter(adapter);
    },
    getPreferredCanvasFormat: () => inner.getPreferredCanvasFormat(),
    get wgslLanguageFeatures() {
      return inner.wgslLanguageFeatures;
    },
  };
  return { gpu, probe: (device) => probes.get(device) };
};
