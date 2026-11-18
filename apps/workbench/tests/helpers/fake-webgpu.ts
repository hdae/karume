/** What a real (fake) compute pass received, in order. */
export type PassCall = readonly unknown[];

/** One compute pass the fake began: its descriptor and every call it received. */
export type RealPass = {
  readonly descriptor?: GPUComputePassDescriptor;
  readonly calls: PassCall[];
};

/** A device the fake handed out: the features it was asked for and the submits and writes it took. */
export type FakeDevice = {
  readonly requested: readonly string[];
  /** The command buffer count of each submit, in order. */
  readonly submitted: number[];
  /** The buffer offset of each `queue.writeBuffer`, in order. */
  readonly written: number[];
};

export type FakeWebGpuOptions = {
  readonly adapterFeatures?: readonly string[];
  /** The adapter's `limits`; a real GPUSupportedLimits keeps its values on the prototype. */
  readonly adapterLimits?: object;
  readonly adapterInfo?: Partial<GPUAdapterInfo>;
  readonly wgslLanguageFeatures?: readonly string[];
};

/**
 * A fake WebGPU that runs every command at encode time: a timestamped pass writes `begin` / `end`
 * (1,000 ns apart per pass, 400 ns long) into its query set when it ends, `resolveQuerySet` and
 * `copyBufferToBuffer` copy bytes, and a mapped buffer shows its bytes. Every compute pass logs
 * what it received, and one that ends with a debug group open throws (WebGPU's validation). Each
 * `requestDevice` creates a device whose features are the ones requested.
 */
export const fakeWebGpu = ({
  adapterFeatures = ["timestamp-query"],
  adapterLimits = {},
  adapterInfo = {},
  wgslLanguageFeatures = ["packed_4x8_integer_dot_product"],
}: FakeWebGpuOptions = {}) => {
  const realPasses: RealPass[] = [];
  const devices: FakeDevice[] = [];
  let clockNs = 0n;
  const bufferBytes = new WeakMap<object, Uint8Array>();
  const queryValues = new WeakMap<object, BigUint64Array>();
  const createBuffer = (descriptor: GPUBufferDescriptor) => {
    const bytes = new Uint8Array(descriptor.size);
    const buffer = {
      size: descriptor.size,
      usage: descriptor.usage,
      mapAsync: () => Promise.resolve(),
      getMappedRange: () => bytes.buffer,
      unmap: () => {},
      destroy: () => {},
    };
    bufferBytes.set(buffer, bytes);
    return buffer;
  };
  const createQuerySet = (query: GPUQuerySetDescriptor) => {
    const querySet = {};
    queryValues.set(querySet, new BigUint64Array(query.count));
    return querySet;
  };
  const beginComputePass = (descriptor?: GPUComputePassDescriptor) => {
    const calls: PassCall[] = [];
    realPasses.push({ descriptor, calls });
    let depth = 0;
    return {
      label: descriptor?.label ?? "",
      setPipeline: (pipeline: GPUComputePipeline) => calls.push(["setPipeline", pipeline.label]),
      setBindGroup: (
        index: number,
        group: unknown,
        offsets?: Iterable<number> | Uint32Array,
        start?: number,
        length?: number,
      ) =>
        calls.push([
          "setBindGroup",
          index,
          group,
          start === undefined || !(offsets instanceof Uint32Array)
            ? [...(offsets ?? [])]
            : [...offsets.subarray(start, start + (length ?? 0))],
        ]),
      dispatchWorkgroups: (x: number, y?: number, z?: number) =>
        calls.push(["dispatch", x, y ?? 1, z ?? 1]),
      dispatchWorkgroupsIndirect: (_buffer: unknown, offset: number) =>
        calls.push(["dispatchIndirect", offset]),
      pushDebugGroup: (group: string) => {
        depth += 1;
        calls.push(["push", group]);
      },
      popDebugGroup: () => {
        depth -= 1;
        calls.push(["pop"]);
      },
      insertDebugMarker: (marker: string) => calls.push(["marker", marker]),
      end: () => {
        if (depth !== 0) throw new Error("a debug group is still open at the end of the pass");
        calls.push(["end"]);
        const writes = descriptor?.timestampWrites;
        if (writes === undefined) return;
        const values = queryValues.get(writes.querySet);
        if (values === undefined) throw new Error("the pass writes into a foreign query set");
        clockNs += 1000n;
        values[writes.beginningOfPassWriteIndex ?? 0] = clockNs;
        values[writes.endOfPassWriteIndex ?? 0] = clockNs + 400n;
      },
    };
  };
  const createCommandEncoder = () => ({
    beginComputePass,
    resolveQuerySet: (
      querySet: object,
      first: number,
      count: number,
      destination: object,
      offset: number,
    ) => {
      const source = queryValues.get(querySet)?.subarray(first, first + count);
      if (source === undefined) return;
      bufferBytes
        .get(destination)
        ?.set(new Uint8Array(source.buffer, source.byteOffset, source.byteLength), offset);
    },
    copyBufferToBuffer: (
      source: object,
      sourceOffset: number,
      destination: object,
      destinationOffset: number,
      size: number,
    ) => {
      const from = bufferBytes.get(source);
      bufferBytes
        .get(destination)
        ?.set(from?.subarray(sourceOffset, sourceOffset + size) ?? [], destinationOffset);
    },
    finish: () => ({}),
  });
  const requestDevice = (descriptor?: GPUDeviceDescriptor) => {
    const requested = [...(descriptor?.requiredFeatures ?? [])];
    const submitted: number[] = [];
    const written: number[] = [];
    const device = {
      features: new Set<string>(requested),
      queue: {
        submit: (buffers: Iterable<unknown>) => submitted.push([...buffers].length),
        writeBuffer: (_buffer: unknown, offset: number) => {
          written.push(offset);
        },
      },
      createBuffer,
      createQuerySet,
      createCommandEncoder,
    };
    devices.push({ requested, submitted, written });
    return Promise.resolve(device);
  };
  const gpu = {
    requestAdapter: () =>
      Promise.resolve({
        features: new Set(adapterFeatures),
        limits: adapterLimits,
        info: adapterInfo,
        requestDevice,
      }),
    getPreferredCanvasFormat: () => "bgra8unorm",
    wgslLanguageFeatures: new Set(wgslLanguageFeatures),
  };
  // A fake of the members the tap and the tests use, not a whole GPU.
  return { gpu: gpu as unknown as GPU, realPasses, devices };
};
