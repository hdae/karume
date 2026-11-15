/**
 * Records which WGSL a test file runs on the GPU.
 *
 * A sha256 row says "this output had these bytes". When the bytes move, the first question is whether
 * the kernels moved: a different WGSL text explains a different output, the same set of texts does not.
 * The verification cache asks the same thing before a run — did the kernels this file compiles change?
 * Both need the set of WGSL texts a test actually dispatched, and only running the test tells: every
 * real-weight test imports every kernel module, so the import graph answers "all of them".
 *
 * The record is taken from outside the runtime, at the three WebGPU calls every dispatch goes through
 * (`createShaderModule`, `createComputePipeline[Async]`, `setPipeline`), so package sources carry no
 * hook and behave the same with or without it. A kernel counts as used when a compute pass selects its
 * pipeline — not when it is compiled — so a pipeline reused from an earlier test of the same file is
 * still seen by the test that reuses it.
 *
 * Uses are kept in two buckets: the test that is running, and everything outside a test (`beforeAll`
 * and other hooks, where many files do their shared work). A test's kernels are its own bucket plus the
 * shared one. Tests of one file run one at a time; with concurrent tests the per-test bucket would mix
 * them, and {@link kernelsInUse} would name a superset.
 */

import { createHash } from "node:crypto";

/** One WGSL text a test dispatched: the pipeline's label (the pipeline key) and the text's sha256. */
export type KernelUse = {
  readonly label: string;
  readonly wgslSha256: string;
};

/** The WebGPU classes whose prototypes carry the three calls (Dawn's `globals`, or fakes in a test). */
export type RecordedClasses = {
  readonly GPUDevice: { readonly prototype: object };
  readonly GPUComputePassEncoder: { readonly prototype: object };
};

type RecorderState = {
  /** Classes already patched (a setup file is evaluated once per test file, the patch is per worker). */
  readonly patched: WeakSet<object>;
  readonly wgslOfModule: WeakMap<object, string>;
  readonly kernelOfPipeline: WeakMap<object, KernelUse>;
  /** Uses outside a test since the file began. */
  shared: Map<string, KernelUse>;
  /** Uses of the running test (`undefined` between tests). */
  test: Map<string, KernelUse> | undefined;
  /** Every use since the file began. */
  file: Map<string, KernelUse>;
};

type RecorderHost = typeof globalThis & { __karumeKernelRecord?: RecorderState };

const stateOf = (): RecorderState => {
  const host = globalThis as RecorderHost;
  host.__karumeKernelRecord ??= {
    patched: new WeakSet(),
    wgslOfModule: new WeakMap(),
    kernelOfPipeline: new WeakMap(),
    shared: new Map(),
    test: undefined,
    file: new Map(),
  };
  return host.__karumeKernelRecord;
};

const identityOf = (kernel: KernelUse): string => `${kernel.label}\n${kernel.wgslSha256}`;

const sorted = (kernels: Iterable<KernelUse>): KernelUse[] =>
  [...kernels].sort((a, b) => {
    const left = identityOf(a);
    const right = identityOf(b);
    return left < right ? -1 : left > right ? 1 : 0;
  });

type Method = (this: unknown, ...args: unknown[]) => unknown;

/** Replaces `prototype[name]` with a function that calls the original and then `after`. */
const wrapMethod = (
  prototype: object,
  name: string,
  after: (args: readonly unknown[], result: unknown) => void,
): void => {
  const original: unknown = Reflect.get(prototype, name);
  if (typeof original !== "function") {
    throw new Error(`the kernel record cannot wrap '${name}': it is not a method of the prototype`);
  }
  const wrapped: Method = function (this: unknown, ...args: unknown[]): unknown {
    const result: unknown = Reflect.apply(original, this, args);
    after(args, result);
    return result;
  };
  Reflect.set(prototype, name, wrapped);
};

const isObject = (value: unknown): value is object => typeof value === "object" && value !== null;

const textField = (value: unknown, name: string): string | undefined => {
  if (!isObject(value)) return undefined;
  const field: unknown = Reflect.get(value, name);
  return typeof field === "string" ? field : undefined;
};

/**
 * Starts recording on the given classes. Calling it again for the same classes does nothing, so a
 * setup file may call it for every test file of a worker.
 */
export const installKernelRecord = (classes: RecordedClasses): void => {
  const state = stateOf();
  const device = classes.GPUDevice.prototype;
  const pass = classes.GPUComputePassEncoder.prototype;
  if (state.patched.has(device) && state.patched.has(pass)) return;

  wrapMethod(device, "createShaderModule", ([descriptor], module) => {
    const code = textField(descriptor, "code");
    if (code !== undefined && isObject(module)) state.wgslOfModule.set(module, code);
  });

  const rememberPipeline = (descriptor: unknown, pipeline: unknown): void => {
    if (!isObject(descriptor) || !isObject(pipeline)) return;
    const compute: unknown = Reflect.get(descriptor, "compute");
    const module: unknown = isObject(compute) ? Reflect.get(compute, "module") : undefined;
    const wgsl = isObject(module) ? state.wgslOfModule.get(module) : undefined;
    if (wgsl === undefined) return;
    state.kernelOfPipeline.set(pipeline, {
      label: textField(descriptor, "label") ?? textField(module, "label") ?? "",
      wgslSha256: createHash("sha256").update(wgsl).digest("hex"),
    });
  };
  wrapMethod(device, "createComputePipeline", ([descriptor], pipeline) => {
    rememberPipeline(descriptor, pipeline);
  });
  wrapMethod(device, "createComputePipelineAsync", ([descriptor], pending) => {
    if (!(pending instanceof Promise)) return;
    // A rejected creation has no pipeline to remember; the caller owns the rejection.
    pending.then(
      (pipeline: unknown) => rememberPipeline(descriptor, pipeline),
      () => undefined,
    );
  });

  wrapMethod(pass, "setPipeline", ([pipeline]) => {
    const kernel = isObject(pipeline) ? state.kernelOfPipeline.get(pipeline) : undefined;
    if (kernel === undefined) return;
    const identity = identityOf(kernel);
    (state.test ?? state.shared).set(identity, kernel);
    state.file.set(identity, kernel);
  });

  state.patched.add(device);
  state.patched.add(pass);
};

/** Forgets the uses of the previous test file (called when a test file starts). */
export const beginKernelRecordFile = (): void => {
  const state = stateOf();
  state.shared = new Map();
  state.test = undefined;
  state.file = new Map();
};

/** Opens the bucket of one test (called before each test). */
export const beginKernelRecordTest = (): void => {
  stateOf().test = new Map();
};

/** Closes the bucket of the running test (called after each test). */
export const endKernelRecordTest = (): void => {
  stateOf().test = undefined;
};

/**
 * The kernels behind what the running test computed: the ones dispatched outside any test since the
 * file began, plus the ones dispatched inside this test. Sorted by label, then by sha256.
 */
export const kernelsInUse = (): KernelUse[] => {
  const state = stateOf();
  return sorted(new Map([...state.shared, ...(state.test ?? [])]).values());
};

/** Every kernel dispatched since the test file began. Sorted by label, then by sha256. */
export const kernelsOfFile = (): KernelUse[] => sorted(stateOf().file.values());

/** One value for a set of kernels: the sha256 of its sorted `label` / `sha256` lines. */
export const kernelSetFingerprint = (kernels: readonly KernelUse[]): string => {
  const hash = createHash("sha256");
  for (const kernel of sorted(kernels)) hash.update(`${identityOf(kernel)}\n`);
  return hash.digest("hex");
};
