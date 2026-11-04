// Injects a "non-fatal OOM" into errorScope pops (wraps a real GPU device).
//
// Non-fatal OOM = `createBuffer` returns an invalid buffer without throwing, and the failure only
// shows up at the later `popErrorScope` (the out-of-memory scope). A real GPU cannot be made to
// produce this on purpose, so only the pop result of the out-of-memory scope is replaced (the
// alternative to opening an injection surface in production code — the same shape as
// `injectStorageFault` in `gpu_plan_backing_test.ts`).
//
// MUST: the real pop is always issued **synchronously**. Swallowing it would leave the pushed scope
// on the device's LIFO, and later failures would be caught by the wrong scope. `popFailureScopes`
// has the convention of issuing its two pops in the same synchronous section, so the wrapper also
// issues the real pop before its first await.
// MUST: push is wrapped to keep a stack of filters. pop does not take a filter, so which pop is the
// out-of-memory scope is known only from what push recorded (so that the validation / internal
// scopes of pipeline creation are not hit).

/** The control surface of {@link injectOutOfMemory}. */
export type OutOfMemoryInjection = {
  /** Puts an OOM on the next out-of-memory scope to be popped. */
  readonly arm: () => void;
  /**
   * How many times the injection actually took effect (used to confirm the check is not a no-op).
   */
  readonly fired: () => number;
  readonly restore: () => void;
};

export const injectOutOfMemory = (device: GPUDevice): OutOfMemoryInjection => {
  // oxlint-disable-next-line typescript/unbound-method -- kept only to reassign in restore(); called via bind below
  const originalPush = device.pushErrorScope;
  // oxlint-disable-next-line typescript/unbound-method -- kept only to reassign in restore(); called via bind below
  const originalPop = device.popErrorScope;
  const push = originalPush.bind(device);
  const pop = originalPop.bind(device);
  // Scopes pushed before the injection was installed are not recorded (they pop as `undefined`).
  const filters: GPUErrorFilter[] = [];
  let armed = false;
  let fired = 0;
  device.pushErrorScope = (filter: GPUErrorFilter): undefined => {
    filters.push(filter);
    return push(filter);
  };
  device.popErrorScope = (): Promise<GPUError | null> => {
    const filter = filters.pop();
    const real = pop();
    if (!armed || filter !== "out-of-memory") return real;
    armed = false;
    fired += 1;
    return real.then((error) => error ?? new GPUOutOfMemoryError("注入: 非致命 OOM"));
  };
  return {
    arm: (): void => {
      armed = true;
    },
    fired: (): number => fired,
    restore: (): void => {
      device.pushErrorScope = originalPush;
      device.popErrorScope = originalPop;
    },
  };
};
