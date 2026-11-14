/**
 * Decides whether the real-GPU tests can run.
 *
 * Without an adapter the tests are **explicitly SKIPped** with `test.skipIf` (the tests
 * are not removed to go green silently). The decision is needed when the tests are registered, so
 * it runs once, at module evaluation. An environment where `requestAdapter()` throws is "a broken
 * environment", so the exception propagates instead of being swallowed.
 */
const detectAdapter = async (): Promise<boolean> => {
  const gpu: GPU | undefined = navigator.gpu;
  if (gpu === undefined) {
    return false;
  }
  return (await gpu.requestAdapter()) !== null;
};

export const GPU_AVAILABLE: boolean = await detectAdapter();

/**
 * Whether the adapter can use the f16 **compute** variants (ADR 0028).
 *
 * MUST: the feature list only says "a candidate that may work" — the check that it actually works
 * is the executing canary of `acquireGpu`, and when that fails the tests go red even with the
 * feature listed (denoland/deno#23125). f16 weight **storage** (ADR 0018) does not need this
 * feature, so the storage tests run unconditionally.
 */
const detectShaderF16 = async (): Promise<boolean> => {
  const gpu: GPU | undefined = navigator.gpu;
  if (gpu === undefined) return false;
  const adapter = await gpu.requestAdapter();
  return adapter !== null && adapter.features.has("shader-f16");
};

export const SHADER_F16_AVAILABLE: boolean = await detectShaderF16();

/**
 * Whether the adapter can use the GPU timing diagnostics (ADR 0021).
 *
 * MUST: a test that looks at the GPU time itself (`lastRunTiming`) passes
 * `acquireGpu({gpuTiming: true})` explicitly, and where the feature is not listed this flag SKIPs
 * the case (`true` fails loudly when the feature is absent).
 * MUST NOT: request timing to see which keys ran how many times (the census). The census comes from
 * `lastRunPipelines` (tests/helpers/pipeline-census.ts), which does not depend on timing;
 * requesting timing would SKIP on machines without it and go red on machines where timing takes
 * the device down (Metal's query set limit — docs/known-issues.md).
 */
const detectTimestampQuery = async (): Promise<boolean> => {
  const gpu: GPU | undefined = navigator.gpu;
  if (gpu === undefined) return false;
  const adapter = await gpu.requestAdapter();
  return adapter !== null && adapter.features.has("timestamp-query");
};

export const TIMESTAMP_QUERY_AVAILABLE: boolean = await detectTimestampQuery();

/**
 * Opt-out that explicitly allows "every case SKIPped without an adapter".
 *
 * MUST: the default is fail loudly (ADR 0005 "all cases SKIPped is an explicit FAIL"). A state
 * that stays green when every GPU test disappears is exactly the silent apparent success that
 * makes unverified work read as verified. The guard lives in `tests/gpu_gate_test.ts`, and getting
 * past it requires this environment variable as a statement of intent.
 */
export const ALLOW_NO_GPU: boolean = process.env.KARUME_ALLOW_NO_GPU === "1";

/**
 * Opt-out that explicitly allows "SKIP without `shader-f16`" (same shape as
 * {@link ALLOW_NO_GPU}).
 *
 * MUST: the default is fail loudly. On a machine without this feature the cases of the f16 compute
 * variants (ADR 0028) disappear entirely, and verify goes green without ever checking the write-out
 * guard. This check stops the verified scope from shrinking silently the moment the machine
 * changes (the guard is `tests/gpu_gate_test.ts`).
 */
export const ALLOW_NO_SHADER_F16: boolean = process.env.KARUME_ALLOW_NO_SHADER_F16 === "1";

/**
 * Opt-out that explicitly allows "SKIP without `timestamp-query`" (same as above).
 *
 * On a machine without this feature the cases of the GPU timing diagnostics (ADR 0021) disappear
 * (the census does not depend on timing, so it stays).
 */
export const ALLOW_NO_TIMESTAMP_QUERY: boolean =
  process.env.KARUME_ALLOW_NO_TIMESTAMP_QUERY === "1";

if (!GPU_AVAILABLE) {
  console.warn(
    "[karume] No GPUAdapter, so the real-GPU tests are SKIPped (a release requires them green on a real GPU). " +
      "Set KARUME_ALLOW_NO_GPU=1 to pass on purpose in an environment without a GPU",
  );
} else {
  if (!SHADER_F16_AVAILABLE) {
    console.warn(
      "[karume] The adapter does not list 'shader-f16', so the real-GPU tests of the f16 compute variants (ADR 0028) " +
        "are SKIPped. The checks of the default f32 path are unaffected. " +
        "Set KARUME_ALLOW_NO_SHADER_F16=1 to pass on purpose on this machine",
    );
  }
  if (!TIMESTAMP_QUERY_AVAILABLE) {
    console.warn(
      "[karume] The adapter does not list 'timestamp-query', so the real-GPU tests of the GPU timing diagnostics (ADR 0021) " +
        "are SKIPped. The numeric checks and the census (runs per key) are unaffected. " +
        "Set KARUME_ALLOW_NO_TIMESTAMP_QUERY=1 to pass on purpose on this machine",
    );
  }
}
