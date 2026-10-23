import type { SessionDiagnostics } from "@karume/runtime";

/**
 * conv3d dispatch counts of a Session's last run, split by the w8a8 key and the f32 key, and the
 * `quantize_rows` preludes the w8a8 kernels bring (ADR 0133 decision 8 — the census of the decoder on
 * int8 activations). Shared by the decoder's A/B (`e2e_wan_ti2v_vae_a8_test.ts`) and its 32-latent
 * pair (`e2e_wan_ti2v_vae_tile32_test.ts`).
 *
 * MUST: a run without a pipeline census is not counted as 0 dispatches (the census would pass
 * vacuously) — it throws.
 */
export const conv3dCensus = (
  graph: string,
  diagnostics: SessionDiagnostics,
): { readonly w8a8: number; readonly f32: number; readonly preludes: number } => {
  const pipelines = diagnostics.lastRunPipelines;
  if (pipelines === undefined) throw new Error(`${graph}: 直近 run の pipeline の census が無い`);
  const sum = (predicate: (key: string) => boolean): number =>
    pipelines
      .filter(({ key }) => predicate(key))
      .reduce((total, { dispatchCount }) => total + dispatchCount, 0);
  return {
    w8a8: sum((key) => key.startsWith("conv3d:") && key.split(":")[2] === "i8a8"),
    f32: sum((key) => key.startsWith("conv3d:") && key.split(":")[2] === "f32"),
    preludes: sum((key) => key.startsWith("quantize_rows:")),
  };
};
