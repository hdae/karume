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
 * `timestampWrites` to each compute pass and resolves the pairs before the command encoder finishes.
 *
 * The split mode ({@link TapMode} `"dispatches"`) is the attribution view: the tap ends the real
 * pass before every dispatch after the first and begins a new timed one, replaying the pipeline and
 * the bind groups, so every dispatch gets its own GPU time and the pipeline label (the runtime's
 * cache key) it ran with. The extra pass boundaries cost GPU time of their own, so absolute times
 * are inflated; the shares per kernel are what it is for.
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
/** Queries in the set (WebGPU's maximum): two per timed pass, reused round-robin. */
const QUERY_COUNT = 4096;
/** Timed passes whose pairs fit in the query set at once (slot `s` writes pair `s mod QUERY_PAIRS`). */
const QUERY_PAIRS = QUERY_COUNT / 2;
/** Bytes per resolved pair (two u64 timestamps), packed back to back in the results buffer. */
const PAIR_BYTES = 16;
/**
 * Timed passes between two resets; later passes are counted but untimed. Sized for the split mode:
 * QAT E2B runs about 1,036 dispatches per decode token, so 128 new tokens plus a prefill fit
 * (262,144 × 16 B = 4 MiB of results, plus at most as much again for the read-back copy).
 */
export const TIMED_PASS_SLOTS = 262_144;

/**
 * What the probe records: `"passes"` = the product's own compute passes (the A/B view);
 * `"dispatches"` = one timed pass per dispatch, labelled with its pipeline (the attribution view).
 */
export type TapMode = "passes" | "dispatches";

/** One compute pass the product encoded (split: one dispatch) while the probe was enabled. */
export type TapPass = {
  /** Its resolve slot; -1 = untimed (no timestamp-query, or past {@link TIMED_PASS_SLOTS}). */
  readonly slot: number;
  dispatches: number;
  /** The pipeline label (the runtime's cache key) of the pass's first dispatch; recorded split only. */
  label?: string;
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
  /** `"passes"` (the default) or `"dispatches"`. Applies to command encoders created from now on. */
  mode: TapMode;
  /** The passes recorded since the last {@link DeviceProbe.reset}, in encode order. */
  readonly passes: () => readonly TapPass[];
  /** Submits recorded since the last reset. */
  readonly submits: () => number;
  /**
   * Copies every pending timestamp pair back and fills the passes' `beginNs` / `endNs`. Call it after
   * the measured work (it submits a copy of its own, which is not recorded).
   */
  readonly readTimes: () => Promise<void>;
  /** Drops the records (the query set and the buffers stay). */
  readonly reset: () => void;
};

export type GpuTap = {
  /** Pass this to `acquireGpu({ gpu })`. */
  readonly gpu: GPU;
  /** The probe of a device acquired through {@link GpuTap.gpu}; `undefined` for any other device. */
  readonly probe: (device: GPUDevice) => DeviceProbe | undefined;
};

/**
 * The slots one command encoder timed and has not resolved yet: `[start, end)`, consecutive and
 * within one round of the query set, so one `resolveQuerySet` covers them. `generation` ties the run
 * to the records it belongs to (a reset starts a new generation, and a stale run is dropped).
 */
type PendingRun = { start: number; end: number; generation: number };

/** A bind group as the product set it, with its dynamic offsets copied (replayed after a split). */
type BoundGroup = {
  readonly group: GPUBindGroup | null | undefined;
  readonly offsets: readonly number[];
};

type BeginPass = (descriptor?: GPUComputePassDescriptor) => GPUComputePassEncoder;

/** Hides timestamp-query from the device's feature list (the runtime then encodes without timing). */
const hideTimestampFeature = (device: GPUDevice): void => {
  const visible = new Set([...device.features].filter((feature) => feature !== TIMESTAMP_QUERY));
  Object.defineProperty(device, "features", { value: visible, configurable: true });
};

/**
 * Resolving: a timed pass with slot `s` writes pair `s mod QUERY_PAIRS` of the query set. Its
 * encoder collects consecutive slots into a {@link PendingRun}, resolves the run into a scratch
 * buffer (`resolveQuerySet` needs a 256-byte-aligned destination) and copies it to `16 × start` of
 * the results buffer, so the results stay packed (16 B per pass) whatever the run's start. A run is
 * resolved when its encoder finishes, and earlier when the encoder's next slot would not extend it
 * or would start a new round of the query set — so a pair is resolved, by the encoder that wrote
 * it, before a later slot of that encoder overwrites it. A slot whose pair another unfinished
 * encoder still holds unresolved is left untimed. This assumes command buffers are submitted in the
 * order their encoders finish (the runtime encodes and submits one at a time).
 */
const instrument = (device: GPUDevice, timestamps: boolean): DeviceProbe => {
  const querySet = timestamps
    ? device.createQuerySet({ type: "timestamp", count: QUERY_COUNT })
    : undefined;
  const scratch = timestamps
    ? device.createBuffer({
        size: QUERY_COUNT * 8,
        usage: BUFFER_QUERY_RESOLVE | BUFFER_COPY_SRC,
      })
    : undefined;
  const results = timestamps
    ? device.createBuffer({
        size: TIMED_PASS_SLOTS * PAIR_BYTES,
        usage: BUFFER_COPY_DST | BUFFER_COPY_SRC,
      })
    : undefined;
  let passes: TapPass[] = [];
  let submits = 0;
  let nextSlot = 0;
  let generation = 0;
  /** The non-empty runs of encoders that have not finished. */
  const pendingRuns = new Set<PendingRun>();

  const createEncoder = device.createCommandEncoder.bind(device);
  const queue = device.queue;
  const submit = queue.submit.bind(queue);

  /** Encodes the run's resolve (outside a pass) and empties the run. */
  const flush = (run: PendingRun, encoder: GPUCommandEncoder): void => {
    pendingRuns.delete(run);
    const count = run.end - run.start;
    if (
      count > 0 &&
      run.generation === generation &&
      querySet !== undefined &&
      scratch !== undefined &&
      results !== undefined
    ) {
      encoder.resolveQuerySet(querySet, (run.start % QUERY_PAIRS) * 2, count * 2, scratch, 0);
      encoder.copyBufferToBuffer(scratch, 0, results, run.start * PAIR_BYTES, count * PAIR_BYTES);
    }
    run.start = run.end;
  };

  /** The slot of a pass `encoder` is about to begin (called outside a pass), or -1 = untimed. */
  const takeSlot = (run: PendingRun, encoder: GPUCommandEncoder): number => {
    if (querySet === undefined || nextSlot >= TIMED_PASS_SLOTS) return -1;
    const slot = nextSlot;
    if (run.generation !== generation) {
      run.start = slot;
      run.end = slot;
      run.generation = generation;
    }
    if (run.end > run.start && (run.end !== slot || slot % QUERY_PAIRS === 0)) flush(run, encoder);
    for (const other of pendingRuns) {
      if (other !== run && other.start <= slot - QUERY_PAIRS) return -1;
    }
    nextSlot += 1;
    if (run.end === run.start) {
      run.start = slot;
      pendingRuns.add(run);
    }
    run.end = slot + 1;
    return slot;
  };

  /** One real, recorded pass on `encoder`: timed when `timed` and a slot is left. */
  const beginRecorded = (
    encoder: GPUCommandEncoder,
    begin: BeginPass,
    run: PendingRun,
    descriptor: GPUComputePassDescriptor | undefined,
    timed: boolean,
  ): { readonly pass: GPUComputePassEncoder; readonly record: TapPass } => {
    const slot = timed ? takeSlot(run, encoder) : -1;
    const record: TapPass = { slot, dispatches: 0 };
    passes.push(record);
    if (slot < 0 || querySet === undefined) return { pass: begin(descriptor), record };
    const first = (slot % QUERY_PAIRS) * 2;
    const pass = begin({
      ...descriptor,
      timestampWrites: {
        querySet,
        beginningOfPassWriteIndex: first,
        endOfPassWriteIndex: first + 1,
      },
    });
    return { pass, record };
  };

  /** The product's pass as it encoded it (the A/B view): counted, and timed as one pass. */
  const productPass = (
    encoder: GPUCommandEncoder,
    begin: BeginPass,
    run: PendingRun,
    descriptor: GPUComputePassDescriptor | undefined,
    timed: boolean,
  ): GPUComputePassEncoder => {
    const { pass, record } = beginRecorded(encoder, begin, run, descriptor, timed);
    const dispatchWorkgroups = pass.dispatchWorkgroups.bind(pass);
    pass.dispatchWorkgroups = (x, y, z) => {
      record.dispatches += 1;
      return dispatchWorkgroups(x, y, z);
    };
    return pass;
  };

  /**
   * The product's pass split into one real pass per dispatch (the attribution view). The returned
   * encoder keeps the pass state the product set — pipeline, bind groups, open debug groups, label —
   * and replays it on every new real pass.
   */
  const splitPass = (
    encoder: GPUCommandEncoder,
    begin: BeginPass,
    run: PendingRun,
    descriptor: GPUComputePassDescriptor | undefined,
  ): GPUComputePassEncoder => {
    let current = beginRecorded(encoder, begin, run, descriptor, true);
    /** A label the product set on the pass after beginning it (replayed); `undefined` = none. */
    let label: string | undefined;
    let pipeline: GPUComputePipeline | undefined;
    const bindGroups = new Map<number, BoundGroup>();
    const debugGroups: string[] = [];

    /** Ends the real pass and begins the next with the state replayed (between two dispatches). */
    const split = (): void => {
      for (let index = 0; index < debugGroups.length; index += 1) current.pass.popDebugGroup();
      current.pass.end();
      current = beginRecorded(encoder, begin, run, descriptor, true);
      if (label !== undefined) current.pass.label = label;
      if (pipeline !== undefined) current.pass.setPipeline(pipeline);
      for (const [index, bound] of [...bindGroups].toSorted(([left], [right]) => left - right)) {
        current.pass.setBindGroup(index, bound.group, bound.offsets);
      }
      for (const group of debugGroups) current.pass.pushDebugGroup(group);
    };

    /** The real pass a dispatch goes into: a fresh one unless the current has none yet. */
    const passForDispatch = (): GPUComputePassEncoder => {
      if (current.record.dispatches > 0) split();
      current.record.dispatches += 1;
      if (pipeline !== undefined) current.record.label = pipeline.label;
      return current.pass;
    };

    const setBindGroup = (
      index: GPUIndex32,
      group: GPUBindGroup | null | undefined,
      offsets?: Iterable<GPUBufferDynamicOffset> | Uint32Array,
      start?: GPUSize64,
      length?: GPUSize32,
    ): undefined => {
      if (start !== undefined && offsets instanceof Uint32Array) {
        const count = length ?? 0;
        bindGroups.set(index, { group, offsets: [...offsets.subarray(start, start + count)] });
        return current.pass.setBindGroup(index, group, offsets, start, count);
      }
      bindGroups.set(index, { group, offsets: [...(offsets ?? [])] });
      return current.pass.setBindGroup(index, group, offsets);
    };

    return {
      __brand: "GPUComputePassEncoder",
      get label() {
        return label ?? current.pass.label;
      },
      set label(value: string) {
        label = value;
        current.pass.label = value;
      },
      setPipeline: (value) => {
        pipeline = value;
        return current.pass.setPipeline(value);
      },
      setBindGroup,
      setImmediates: () => {
        // Immediate data would have to be replayed on every new pass; nothing on the decode path
        // sets it, so the tap fails loudly instead of timing a pass that lost it.
        throw new Error("The GPU tap's split mode does not support setImmediates");
      },
      dispatchWorkgroups: (x, y, z) => passForDispatch().dispatchWorkgroups(x, y, z),
      dispatchWorkgroupsIndirect: (buffer, offset) =>
        passForDispatch().dispatchWorkgroupsIndirect(buffer, offset),
      pushDebugGroup: (group) => {
        debugGroups.push(group);
        return current.pass.pushDebugGroup(group);
      },
      popDebugGroup: () => {
        debugGroups.pop();
        return current.pass.popDebugGroup();
      },
      insertDebugMarker: (marker) => current.pass.insertDebugMarker(marker),
      end: () => current.pass.end(),
    };
  };

  const probe: DeviceProbe = {
    timestamps,
    enabled: false,
    mode: "passes",
    passes: () => passes,
    submits: () => submits,
    readTimes: async () => {
      if (results === undefined) return;
      const pending = passes.filter((pass) => pass.slot >= 0 && pass.beginNs === undefined);
      if (pending.length === 0) return;
      let first = pending[0].slot;
      let last = first;
      for (const { slot } of pending) {
        first = Math.min(first, slot);
        last = Math.max(last, slot);
      }
      const bytes = (last - first + 1) * PAIR_BYTES;
      const staging = device.createBuffer({
        size: bytes,
        usage: BUFFER_MAP_READ | BUFFER_COPY_DST,
      });
      try {
        const encoder = createEncoder();
        encoder.copyBufferToBuffer(results, first * PAIR_BYTES, staging, 0, bytes);
        submit([encoder.finish()]);
        await staging.mapAsync(MAP_MODE_READ);
        const view = new BigUint64Array(staging.getMappedRange().slice(0));
        staging.unmap();
        for (const pass of pending) {
          const at = ((pass.slot - first) * PAIR_BYTES) / 8;
          pass.beginNs = view[at];
          pass.endNs = view[at + 1];
        }
      } finally {
        staging.destroy();
      }
    },
    reset: () => {
      passes = [];
      submits = 0;
      nextSlot = 0;
      generation += 1;
      pendingRuns.clear();
    },
  };

  queue.submit = (buffers) => {
    if (probe.enabled) submits += 1;
    return submit(buffers);
  };

  device.createCommandEncoder = (descriptor) => {
    const encoder = createEncoder(descriptor);
    if (!probe.enabled) return encoder;
    const split = probe.mode === "dispatches";
    const run: PendingRun = { start: 0, end: 0, generation };
    const begin: BeginPass = encoder.beginComputePass.bind(encoder);
    encoder.beginComputePass = (passDescriptor) => {
      // A pass that writes its own timestamps keeps them and is neither timed nor split (the
      // product's timing mode is off, so none does today).
      const own = passDescriptor?.timestampWrites !== undefined;
      return split && !own
        ? splitPass(encoder, begin, run, passDescriptor)
        : productPass(encoder, begin, run, passDescriptor, !own);
    };
    const finish = encoder.finish.bind(encoder);
    encoder.finish = (finishDescriptor) => {
      flush(run, encoder);
      return finish(finishDescriptor);
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
