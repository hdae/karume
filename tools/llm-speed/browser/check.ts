/// <reference lib="dom" />
/**
 * Chrome で並列 GEMV subgroup32 の u32 一致門（tests/helpers/gemv-subgroup-check.ts）を走らせる。
 *
 * Deno は subgroups を提供しないので `gpu_linear_gemv_subgroup_test.ts` の実走は明示 SKIP になる
 * （ADR 0101「Chrome で同じ検査を実行する」）。このページがその Chrome 側の入口。速度計測ではない。
 */
import { acquireGpu } from "../../../packages/runtime/src/gpu/device.ts";
import { checkGemvSubgroup } from "../../../packages/runtime/tests/helpers/gemv-subgroup-check.ts";

const output = document.getElementById("output") as HTMLPreElement;
const log = (line: string): void => {
  output.textContent += `${line}\n`;
};

export const runCheck = async (): Promise<void> => {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw Error("WebGPU adapter unavailable");
  const info = adapter.info;
  log(`adapter: ${info.vendor} / ${info.architecture} / ${info.device} / ${info.description}`);
  log(`userAgent: ${navigator.userAgent}`);
  const config = await (await fetch("/config.json")).json();
  log(
    `checkout: ${config.revision}${config.dirty ? " (dirty)" : ""} bundle: ${config.bundleSha256}`,
  );
  const features = new Set<string>(adapter.features);
  const host: GPU & { wgslLanguageFeatures?: Iterable<string> } = navigator.gpu;
  for (const feature of ["subgroups", "subgroup-size-control", "timestamp-query"]) {
    log(`feature ${feature}: ${features.has(feature)}`);
  }
  log(`wgsl subgroup_id: ${new Set(host.wgslLanguageFeatures).has("subgroup_id")}`);
  const gpu = await acquireGpu({ subgroups: true, gpuTiming: true });
  try {
    const started = performance.now();
    const result = await checkGemvSubgroup(gpu);
    log(
      `PASS: parallel と subgroup32 の u32 一致 — cases ${result.cases} / elements ${result.elements} / ${
        Math.round(performance.now() - started)
      } ms`,
    );
  } finally {
    gpu.destroy();
  }
};

runCheck().catch((error: unknown) => {
  log(`FAIL: ${error instanceof Error ? error.message : String(error)}`);
  if (error instanceof Error && error.stack) log(error.stack);
});
