import type { SessionDiagnostics } from "@karume/runtime";
import { CONV3D_I8A8_QUANTIZE_GEOMETRY } from "../../../runtime/src/kernels/conv3d-i8a8.ts";
import { quantizeRowsChannelMajorKey } from "../../../runtime/src/kernels/quantize-rows.ts";

/**
 * conv3d dispatch counts of a Session's last run, split by the w8a8 key and the f32 key, and the
 * `quantize_rows` preludes the w8a8 kernels bring (ADR 0133 decision 8 — the census of the decoder on
 * int8 activations). Shared by the decoder's A/B (`e2e_wan_ti2v_vae_a8_test.ts`) and its 32-latent
 * pair (`e2e_wan_ti2v_vae_tile32_test.ts`).
 *
 * `channelMajorPreludes` counts the production key of the channel-major quantiser (ADR 0133 addendum
 * 2026-10-26) and `strided` every strided dispatch: the cat nodes of the graph run strided keys too,
 * so a re-introduced transposing prelude shows as a strided count above the control's, not as a
 * non-zero one.
 *
 * MUST: a run without a pipeline census is not counted as 0 dispatches (the census would pass
 * vacuously) — it throws.
 */
export const conv3dCensus = (
  graph: string,
  diagnostics: SessionDiagnostics,
): {
  readonly w8a8: number;
  readonly f32: number;
  readonly preludes: number;
  readonly channelMajorPreludes: number;
  readonly strided: number;
} => {
  const pipelines = diagnostics.lastRunPipelines;
  if (pipelines === undefined) throw new Error(`${graph}: the last run has no pipeline census`);
  const sum = (predicate: (key: string) => boolean): number =>
    pipelines
      .filter(({ key }) => predicate(key))
      .reduce((total, { dispatchCount }) => total + dispatchCount, 0);
  const channelMajorKey = quantizeRowsChannelMajorKey(CONV3D_I8A8_QUANTIZE_GEOMETRY);
  return {
    w8a8: sum((key) => key.startsWith("conv3d:") && key.split(":")[2] === "i8a8"),
    f32: sum((key) => key.startsWith("conv3d:") && key.split(":")[2] === "f32"),
    preludes: sum((key) => key.startsWith("quantize_rows:")),
    channelMajorPreludes: sum((key) => key === channelMajorKey),
    strided: sum((key) => key.startsWith("strided:")),
  };
};
