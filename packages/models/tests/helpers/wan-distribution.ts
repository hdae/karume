/**
 * Wan2.1 の配布形ミラー（`models/karume-wan2.1/`）を組むコマンド。配布形を読む e2e とホストのテストの SKIP 文言が同じ
 * コマンドを案内する（2.2 の {@link WAN_TI2V_ASSEMBLE_COMMAND} と同じ形 — `wan-ti2v-pipeline.ts`）。
 *
 * The mirror is local-only (not published) and is still the distribution form the tests read (ADR 0139 decision 1 —
 * an unpublished family reads its local mirror built by `dist.py`).
 *
 * 副作用ゼロ: 定数と、呼ばれたときだけ読む関数だけを持つ。
 */

import type { OpenedContainer } from "@karume/runtime";
import { mirrorRoot, openSelectedContainer, selectMirrorComponent } from "./mirror.ts";

/**
 * 配布形を組むコマンド。umT5 の越境参照は仮の SHA（`--allow-placeholder-ref`）で焼く — ローカルのミラーを根にする e2e は
 * 取得元の `crossRepo` の mapping で `models/karume-umt5-xxl/` を渡すので、revision は読まない。
 */
export const WAN_ASSEMBLE_COMMAND =
  "cd tools/export-recipes && uv run python dist.py --pipeline wan --allow-placeholder-ref";

/** The Wan2.1 mirror (the default output of `dist.py --pipeline wan`). */
export const WAN_DIST_ROOT: URL = mirrorRoot("karume-wan2.1");
/** The mirror's model name (the key of the manifest's `models`). */
export const WAN_DIST_MODEL = "t2v-1.3b";

/**
 * Opens one component of the mirror in the given storage, through the manifest (ADR 0139 decision 1): f16 through the
 * `f16` preset, i8 through the `f16+dit8` preset (every preset that stores a component in a dtype names the same
 * container). Range reads; the Session options stay the caller's — the preset only selects the container.
 *
 * MUST: the preset's declared storage is checked — a mirror whose preset stores the component differently would
 * otherwise run the check against the other container without saying so.
 */
export const openWanComponent = async (
  component: string,
  dtype: "f16" | "i8",
): Promise<OpenedContainer> => {
  const quant = dtype === "f16" ? "f16" : "f16+dit8";
  const chosen = await selectMirrorComponent(WAN_DIST_ROOT, component, {
    model: WAN_DIST_MODEL,
    quant,
  });
  const declared = chosen.loaded.manifest.models[WAN_DIST_MODEL].quants[quant].weights[component];
  if (declared !== dtype) {
    throw new Error(
      `test: preset '${quant}' of ${WAN_DIST_ROOT.pathname} stores '${component}' as ${declared}, not ${dtype}`,
    );
  }
  return await openSelectedContainer(chosen);
};
