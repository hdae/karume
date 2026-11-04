/**
 * Identifies the execution environment (the key that separates per-device reference values and
 * results).
 *
 * All the sha256 reference check can claim is "on **the same GPU and the same runtime** not a
 * single bit moves"; bit identity across devices is not guaranteed by the spec (the mechanism is in
 * the "the sha256 reference check is specific to the reference environment" section of
 * docs/limitations.md). With the reference values as constants in the test source, every check
 * would go red the moment the machine changes, and the check's very purpose — "catch a regression
 * on this machine" — would be gone. So the reference values are kept as **one row per
 * environment**, and this file builds the key that looks the row up.
 *
 * MUST: the key has to be "the same every time on the same machine". Of the identifiers WebGPU
 * exposes, the only one that does not shift with a reboot or a driver update is the name in
 * `GPUAdapterInfo` (the driver version is not available from the API). When a driver update moves
 * the numbers, **the rows of the same key disagree** = it is noticed as red (the version is not
 * mixed into the key to divert silently to another row).
 *
 * Implementation differences: Deno puts the GPU name in `description` and leaves `architecture`
 * empty; Chrome puts names in `vendor` / `architecture` and leaves `description` empty. One of the
 * two is always filled, so the filled one is the base. Node (Dawn) puts the driver version in
 * `description`, so `device` (the GPU name) is the base there (keeping the MUST of no version in
 * the key).
 */

import { createRequire } from "node:module";
import { acquireGpu, readAdapterInfo } from "../../src/gpu/device.ts";

/** Version of the `typescript` package the workspace type-checks with (informational, like Deno's). */
const typescriptVersion = (): string => {
  const manifest: unknown = createRequire(import.meta.url)("typescript/package.json");
  return typeof manifest === "object" &&
    manifest !== null &&
    "version" in manifest &&
    typeof manifest.version === "string"
    ? manifest.version
    : "";
};

/**
 * The runtime that separates reference values (on the same GPU, a different runtime can move the
 * numbers).
 */
export type EnvironmentRuntime = "deno" | "chrome" | "node";

/** The identity of this execution environment (written into the result JSON as is). */
export type Environment = {
  /**
   * The key that looks up reference values and results. An environment without a GPU has **none**
   * (no row can be made for it).
   */
  readonly key?: string;
  readonly runtime: {
    readonly name: EnvironmentRuntime;
    readonly version: string;
    readonly v8: string;
    readonly typescript: string;
  };
  readonly adapter: {
    readonly vendor: string;
    readonly architecture: string;
    readonly device: string;
    readonly description: string;
  };
  readonly os: {
    readonly platform: string;
    readonly arch: string;
  };
};

/** Trademark marks dropped from the base (`Intel(R) Graphics` → `intel graphics`). */
const TRADEMARKS = ["(r)", "(tm)", "(c)"] as const;

/**
 * The environment key (`<runtime>-<adapter name>`).
 *
 * Examples: Deno / `Intel(R) Graphics (BMG G21)` → `deno-intel-graphics-bmg-g21`;
 * Chrome / vendor `apple` + architecture `metal-3` → `chrome-apple-metal-3`.
 *
 * MUST: throw in an environment where no name at all can be taken (mixing silently into a row with
 * an empty key would compare against another machine's reference values as "this machine's
 * reference values").
 */
export const environmentKey = (runtime: EnvironmentRuntime, info: GPUAdapterInfo): string => {
  const base =
    runtime === "node"
      ? info.device
      : info.description !== ""
        ? info.description
        : `${info.vendor}-${info.architecture}`;
  let slug = base.toLowerCase();
  for (const mark of TRADEMARKS) slug = slug.replaceAll(mark, "");
  slug = slug.replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  if (slug === "") {
    throw new Error(
      "GPUAdapterInfo から環境キーを作れない（description / vendor / architecture が全て空）。" +
        "参照値をこの環境の行として持てないので、アダプタ情報が読めない理由を先に確かめること",
    );
  }
  return `${runtime}-${slug}`;
};

/**
 * Settles this execution environment once (the adapter info is needed when the tests are
 * registered, so it is carried around synchronously).
 *
 * An environment without a GPU has no {@link Environment.key} — neither the reference row nor the
 * place for the results can be decided, and filling it with an empty string or the like would mix
 * it with another environment's rows.
 */
export const describeEnvironment = async (): Promise<Environment> => {
  const gpu: GPU | undefined = navigator.gpu;
  const adapter = gpu === undefined ? null : await gpu.requestAdapter();
  // `readAdapterInfo` is the one reader that normalizes implementations without `info` to empty
  // values. Without an adapter the same empty values go through (`{}` has the same shape as a
  // missing `info`).
  const info = readAdapterInfo(adapter ?? {});
  return {
    ...(adapter === null ? {} : { key: environmentKey("node", info) }),
    runtime: {
      name: "node",
      version: process.versions.node,
      v8: process.versions.v8,
      typescript: typescriptVersion(),
    },
    adapter: {
      vendor: info.vendor,
      architecture: info.architecture,
      device: info.device,
      description: info.description,
    },
    os: { platform: process.platform, arch: process.arch },
  };
};

/** This execution environment (settled once, at module evaluation). */
export const ENVIRONMENT: Environment = await describeEnvironment();

/**
 * Checks that the adapter the run actually uses is the adapter the environment key was taken from.
 *
 * The key of {@link ENVIRONMENT} comes from a separate `requestAdapter()` at module evaluation, and
 * `acquireGpu` on the execution side acquires an adapter again. **The WebGPU spec does not
 * guarantee that two requests return the same physical adapter** (the same premise as the note on
 * `readAdapterLimits`). A mix-up on a multi-GPU machine would make a `KARUME_REFERENCE` write
 * **silently put another adapter's numbers** into a tracked fixture — it does not go red, and only
 * a person deleting the rows can undo it, an irreversible kind of breakage — so this throws before
 * going on.
 */
export const assertAdapterMatchesEnvironment = (
  context: { readonly adapterInfo: GPUAdapterInfo },
  environment: Environment = ENVIRONMENT,
): void => {
  const running = environmentKey(environment.runtime.name, context.adapterInfo);
  if (running === environment.key) return;
  throw new Error(
    `参照値を索くキー（${environment.key ?? "GPU なし"}）と、` +
      `この走行が実際に取ったアダプタのキー（${running}）が違う。` +
      "同じ機の複数の GPU から requestAdapter が別のアダプタを選んでいる。" +
      "数値をどちらの行として扱うか決められないのでここで止める（参照値の書き込みは戻せない）",
  );
};

/**
 * Checks the GPU used for the real run with {@link assertAdapterMatchesEnvironment} and returns it
 * as is (passed right after the acquisition, in the form
 * `const gpu = verifyRunningGpu(await acquireTestGpu(…))`).
 *
 * The check runs on **the very GPU that produces the numbers**. Checking a GPU acquired separately
 * for the check leaves the real one unchecked when the real run's acquisition picks another
 * adapter (two acquisitions are not guaranteed to return the same physical adapter — the note on
 * {@link assertAdapterMatchesEnvironment}).
 *
 * MUST: a GPU that fails the check is destroyed before the throw — so that this can be used right
 * after the acquisition (outside the caller's try/finally); if it is not returned here, nobody can
 * destroy it.
 */
export const verifyRunningGpu = <
  Gpu extends { readonly adapterInfo: GPUAdapterInfo; destroy(): void },
>(
  gpu: Gpu,
  environment: Environment = ENVIRONMENT,
): Gpu => {
  try {
    assertAdapterMatchesEnvironment(gpu, environment);
  } catch (error) {
    gpu.destroy();
    throw error;
  }
  return gpu;
};

/**
 * Runs {@link assertAdapterMatchesEnvironment} once per run (called at the entry of the paths that
 * write reference values).
 *
 * It acquires a `GpuContext` for the check once, discards it, and caches the result in the
 * process. It runs only once so that the device is not recreated while a pipeline holds assets.
 *
 * NOTE: what this checks is the acquisition made for the check, not the real run's acquisition
 * (it lets a real run that picks another adapter through). A path that holds the real run's GPU
 * uses {@link verifyRunningGpu}. This one is only for paths where the pipeline acquires the GPU
 * internally.
 */
let runningAdapterCheck: Promise<void> | undefined;
export const assertRunningAdapter = (): Promise<void> => {
  runningAdapterCheck ??= (async (): Promise<void> => {
    const gpu = await acquireGpu();
    try {
      assertAdapterMatchesEnvironment(gpu);
    } finally {
      gpu.destroy();
    }
  })();
  return runningAdapterCheck;
};
