// `assetComponentOpener` の門が使う最小の資産フィクスチャ
// （asset_shard_components_test.ts と gpu_asset_shard_components_test.ts の共有）。

import { assetComponentOpener } from "../../src/hub/components.ts";
import { type DumpTensor, writeSafetensors } from "./safetensors-write.ts";

const f32Tensor = (shape: readonly number[], value: number): DumpTensor => ({
  dtype: "F32",
  shape: [...shape],
  data: new Float32Array(shape.reduce((product, dim) => product * dim, 1)).fill(value),
});

/** 最小の IR グラフ 1 本（`linear` 1 段 — `shard_loading_test.ts` と同型）。 */
const miniGraph = (): unknown => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["linear"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [2, 2] }],
  outputs: ["y"],
  initializers: {
    w: { tensor: "m.w", storage: { dtype: "f32" } },
    b: { tensor: "m.b", storage: { dtype: "f32" } },
  },
  values: {
    w: { dtype: "f32", shape: [2, 2] },
    b: { dtype: "f32", shape: [2] },
    y: { dtype: "f32", shape: [2, 2] },
  },
  states: {},
  nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
});

/** グラフ shard（`karume_ir` + 同居テンソル）。重み `m.w` は**入れない**（②の証拠）。 */
export const graphShard = (): Uint8Array<ArrayBuffer> =>
  writeSafetensors(new Map([["m.b", f32Tensor([2], 0.25)]]), {
    karume_ir: JSON.stringify(miniGraph()),
  });

/** 重み shard（`karume_ir` を持たない）。 */
export const weightShard = (): Uint8Array<ArrayBuffer> =>
  writeSafetensors(new Map([["m.w", f32Tensor([2, 2], 0.5)]]), {});

/** 全テンソル同居の 1 本（素の配布形）。 */
export const wholeShard = (): Uint8Array<ArrayBuffer> =>
  writeSafetensors(
    new Map([["m.b", f32Tensor([2], 0.25)], ["m.w", f32Tensor([2, 2], 0.5)]]),
    { karume_ir: JSON.stringify(miniGraph()) },
  );

/** 家族側の資産アクセサ（`assetBuffer`）と同じ姿の最小実装。 */
const bufferOf =
  (assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>) => (key: string): ArrayBuffer => {
    if (!Object.hasOwn(assets, key)) {
      throw new Error(
        `test: 資産 '${key}' が無い（揃っているキー: ${Object.keys(assets).join(" / ")}）`,
      );
    }
    return assets[key].buffer;
  };

export const openerOf = (
  assets: Record<string, Uint8Array<ArrayBuffer>>,
): ReturnType<typeof assetComponentOpener> =>
  assetComponentOpener("test", assets, bufferOf(assets));
