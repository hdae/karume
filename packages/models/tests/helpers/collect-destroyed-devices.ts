/**
 * Has V8 collect destroyed devices before a new device is acquired (called by the tests' GPU
 * acquisition entry point).
 *
 * NOTE: this is a **symptomatic workaround** (maintainer's ruling 2026-10-05 — the root-cause fix
 * is deferred). Deno 2.9.6's `GPUDevice.destroy()` only marks the device invalid and does not
 * return its VRAM; the device and its resources are released only when V8's GC collects the
 * device's wrapper. The tests create and drop one device per test, so destroyed devices that
 * survive until the GC eat the process's VRAM and make a later test's large allocation OOM
 * (docs/known-issues.md "Deno: `GPUDevice.destroy()` does not return VRAM…").
 *
 * NOTE: what this does not guarantee:
 * - When collection happens is V8 / cppgc implementation behaviour, not a spec guarantee. That
 *   two `gc()` calls release it was only observed on a B570 (see {@link GC_SETTLE_MS} below);
 *   another version may need a different count or wait.
 * - Whether collection happened is not checked (a check on the DRM client count is deferred). It
 *   proceeds silently even when nothing was released.
 * - `gc` exists only in runs with `--expose-gc`. The Vitest config passes that flag to every
 *   worker (`execArgv` in `vite.config.ts`, checked by
 *   `packages/runtime/tests/verify_lanes_test.ts`), but a run without the flag does nothing here
 *   (only the mitigation is missing; the tests mean the same).
 * - A GpuContext that is still reachable (held by a variable or a closure) is not collected.
 */

/**
 * The wait after each `gc()` (ms).
 *
 * Measured with the probe (step d of `outputs/diag/dead-device-probe.ts` — `gc()` → 100 ms →
 * observe; `outputs/bench/2026-10-05_dead-device/p1-settle-gc.log`): nothing was released in the
 * 100 ms after the first call, and in the 100 ms after the second call the destroyed devices' DRM
 * clients and all of vram0 / gtt were released (the clients went back to the single fd of the
 * physical device). The wait gives cppgc's finalizers a chance to run across macrotasks; values
 * shorter than 100 ms were not measured, so do not go below the measured value.
 */
export const GC_SETTLE_MS = 100;

const wait = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Calls `host.gc` twice and waits {@link GC_SETTLE_MS} after each call. Does nothing without `gc`.
 *
 * `host` is the object that owns `gc` (default `globalThis`). It takes the owner rather than the
 * function itself because a default parameter reads `undefined` as "use the default" — so that "no
 * gc" can be injected, the absent case is passed as `{}`.
 */
export const collectDestroyedDevices = async (host: object = globalThis): Promise<void> => {
  const gc = "gc" in host ? host.gc : undefined;
  if (typeof gc !== "function") return;
  // Twice: in the probe's measurement the first call releases nothing. The chain that explains it:
  // the first call finalizes the GPUDevice, which drops its strong reference to the GPUQueue, and
  // the GPUQueue (which closes the fd) is collected by the next GC (inferred from the upstream
  // source — only the count is measured).
  gc();
  await wait(GC_SETTLE_MS);
  gc();
  await wait(GC_SETTLE_MS);
};
