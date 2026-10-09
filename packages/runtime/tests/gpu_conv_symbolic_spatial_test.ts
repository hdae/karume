// 空間の長さを記号にした conv3d → 非対称 pad（permute + 最終軸の pad）→ stride 2 の conv2d の実 GPU の門
// （ADR 0121 段 9a — Wan2.2 の VAE encoder の chunk 0 の down の並びを縮めたもの）。
//
// encoder の pre / post のグラフは空間の長さを記号 h, w のまま持つ（資産を解像度から独立させる — ADR 0038 §4）。
// runtime は束縛してから具体の形で計画するので conv の形の規則は記号を受けるが、記号の空間で conv を GPU に回す
// のは encoder が初めて。ここでは**同じ Session** に束縛を 3 回（寸法 A → 寸法 B → 寸法 A）渡し、
//
// 1. 各回の出力（conv3d の出口と、pad → conv2d の後の出口）が CPU 参照と帯の中（conv の parity テストと同じ帯）
// 2. 束縛で conv3d の踏み分けが変わる（出力幅 2w が 4 の倍数なら v4・でなければスカラ）
// 3. 戻りの回が 1 回目と Uint32 で一致する（計画のキャッシュが前の束縛の形・変種を持ち越さない — 両方の出口で）
//
// を見る。encoder の実寸の幅（潜在 80 / 44 / 16 の 8・4・2 倍）はどれも 4 の倍数なので、e2e は v4 しか踏まない —
// 束縛による変種の切り替えはここでしか見えない。
//
// NOTE: pre / post のほかのノード（記号の割り算を含む reshape の space-to-depth・channel 軸の slice / cat・RMS norm の
// sum の縮約・clamp_min / sqrt / div）と f16 の重みの格納を記号の束縛を変えて回すのは、実重みの e2e
// （`packages/models/tests/e2e_wan_ti2v_vae_encoder_test.ts`）に任せる — 既定のレーンで同じ Session を 3 寸法
// （1280×704 → 704×1280 → 256×160）に束縛し直し、束縛ごとに golden と突き合わせる。

import { assert, assertEquals } from "@std/assert";
import { acquireGpu } from "../src/gpu/device.ts";
import { conv2dIgemmMTile } from "../src/kernels/conv2d.ts";
import { conv3dIgemmKey } from "../src/kernels/conv3d.ts";
import { allclose } from "../src/reference/allclose.ts";
import {
  referenceConv2d,
  referenceConv3d,
  referencePad,
  referencePermute,
  type RefTensor,
  refTensor,
} from "../src/reference/ops.ts";
import { createSessionFromContainer } from "../src/runtime/executor.ts";
import type { Tensor } from "../src/runtime/session-types.ts";
import { type DeclarationJson, fill, GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";

/** CPU 参照との突合の帯（conv2d / conv3d の parity テストと同じ — 参照は f64 の縮約）。 */
const CPU_REFERENCE_TOLERANCE = { atol: 1e-5, rtol: 1e-5 } as const;

/** チャネル数（conv3d の kFlat = 4·1·3·3 = 36 は 4 の倍数 — v4 に乗るかは出力幅だけで決まる）。 */
const CHANNELS_IN = 4;
const CHANNELS_MID = 6;
const CHANNELS_OUT = 5;

const CONV3D_ATTRS = { stride: [1, 1, 1], padding: [0, 1, 1], dilation: [1, 1, 1], groups: 1 };
const CONV2D_ATTRS = { stride: [2, 2], padding: [0, 0], dilation: [1, 1], groups: 1 };
const PAD_RIGHT = { left: 0, right: 1 };
const TO_BATCH = [1, 0, 2, 3];
const SWAP_LAST = [0, 1, 3, 2];

/**
 * encoder の down の並び（recipe `wan/vae_encoder_patch.py` の `LastSliceConv3d` → `downsample`）:
 * `x [Cin,1,2h,2w]` → conv3d（時間 1 枚・空間 pad 1）→ フレームを batch へ → W 側 pad → H 側を permute で最終軸へ
 * 回して pad → 戻す → conv2d stride 2 → `[1,Cout,h,w]`。
 */
const downGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["conv2d", "conv3d", "pad", "permute"] },
  symbols: ["h", "w"],
  inputs: [
    { name: "x", dtype: "f32", shape: [CHANNELS_IN, 1, "2h", "2w"] },
    { name: "w3", dtype: "f32", shape: [CHANNELS_MID, CHANNELS_IN, 1, 3, 3] },
    { name: "b3", dtype: "f32", shape: [CHANNELS_MID] },
    { name: "w2", dtype: "f32", shape: [CHANNELS_OUT, CHANNELS_MID, 3, 3] },
    { name: "b2", dtype: "f32", shape: [CHANNELS_OUT] },
  ],
  outputs: ["y3", "y"],
  initializers: {},
  values: {
    y3: { dtype: "f32", shape: [CHANNELS_MID, 1, "2h", "2w"] },
    p0: { dtype: "f32", shape: [1, CHANNELS_MID, "2h", "2w"] },
    q0: { dtype: "f32", shape: [1, CHANNELS_MID, "2h", "2w+1"] },
    p1: { dtype: "f32", shape: [1, CHANNELS_MID, "2w+1", "2h"] },
    q1: { dtype: "f32", shape: [1, CHANNELS_MID, "2w+1", "2h+1"] },
    p2: { dtype: "f32", shape: [1, CHANNELS_MID, "2h+1", "2w+1"] },
    y: { dtype: "f32", shape: [1, CHANNELS_OUT, "h", "w"] },
  },
  nodes: [
    { op: "conv3d", ins: ["x", "w3", "b3"], outs: ["y3"], attrs: CONV3D_ATTRS },
    { op: "permute", ins: ["y3"], outs: ["p0"], attrs: { dims: TO_BATCH } },
    { op: "pad", ins: ["p0"], outs: ["q0"], attrs: PAD_RIGHT },
    { op: "permute", ins: ["q0"], outs: ["p1"], attrs: { dims: SWAP_LAST } },
    { op: "pad", ins: ["p1"], outs: ["q1"], attrs: PAD_RIGHT },
    { op: "permute", ins: ["q1"], outs: ["p2"], attrs: { dims: SWAP_LAST } },
    { op: "conv2d", ins: ["p2", "w2", "b2"], outs: ["y"], attrs: CONV2D_ATTRS },
  ],
});

/** 決定的なデータ列（乱数は使わない — 失敗が再現しないため）。bias は非ゼロ（符号付きゼロの領域を避ける）。 */
const SIGNED = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
const WEIGHT = (i: number): number => (((i * 11) % 19) - 9) * 0.031;
const BIAS = (i: number): number => 0.125 + i * 0.25;

const WEIGHTS = {
  w3: fill([CHANNELS_MID, CHANNELS_IN, 1, 3, 3], WEIGHT),
  b3: fill([CHANNELS_MID], BIAS),
  w2: fill([CHANNELS_OUT, CHANNELS_MID, 3, 3], (i) => WEIGHT(i + 5)),
  b2: fill([CHANNELS_OUT], (i) => BIAS(i) - 0.5),
} as const;

const refOf = (tensor: Tensor): RefTensor => {
  if (tensor.dtype !== "f32") throw new Error(`f32 だけを受ける（${tensor.dtype}）`);
  return refTensor(tensor.shape, tensor.data);
};

/** CPU 参照（同じ並びを reference op で組む）。 */
const referenceDown = (x: Tensor): { readonly y3: RefTensor; readonly y: RefTensor } => {
  const y3 = referenceConv3d(refOf(x), refOf(WEIGHTS.w3), refOf(WEIGHTS.b3), CONV3D_ATTRS);
  const widened = referencePad(referencePermute(y3, TO_BATCH), PAD_RIGHT);
  const heightened = referencePad(referencePermute(widened, SWAP_LAST), PAD_RIGHT);
  const y = referenceConv2d(
    referencePermute(heightened, SWAP_LAST),
    refOf(WEIGHTS.w2),
    refOf(WEIGHTS.b2),
    CONV2D_ATTRS,
  );
  return { y3, y };
};

/**
 * 束縛の列: 出力幅 2w = 10（4 の倍数でない — conv3d はスカラ）→ 2w = 12（v4）→ 1 回目へ戻る。高さと幅を
 * 別の値にして記号の取り違えを形で掴む。
 */
const BINDINGS: readonly { readonly h: number; readonly w: number; readonly v4: boolean }[] = [
  { h: 3, w: 5, v4: false },
  { h: 4, w: 6, v4: true },
  { h: 3, w: 5, v4: false },
];

Deno.test({
  name:
    "記号の空間の conv3d → 非対称 pad → stride 2 の conv2d は、同じ Session の束縛 3 回で CPU 参照と一致し、" +
    "束縛で conv3d の変種を踏み分け、戻りの回は 1 回目とビット一致（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async (t) => {
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(
      gpu,
      await openModelBytes(downGraph(), []),
      GRAPH_NAME,
    );
    try {
      const mTile = conv2dIgemmMTile(CHANNELS_MID);
      const outputsByRun: { readonly y3: Float32Array; readonly y: Float32Array }[] = [];
      for (const [index, { h, w, v4 }] of BINDINGS.entries()) {
        await t.step(
          `run ${index + 1}: h = ${h}・w = ${w}（conv3d は ${v4 ? "v4" : "スカラ"}）`,
          async () => {
            const x = fill([CHANNELS_IN, 1, 2 * h, 2 * w], (i) => SIGNED(i + h * 31 + w));
            const outputs = await session.run({ x, ...WEIGHTS });
            const reference = referenceDown(x);
            assertEquals(outputs["y3"].shape, [CHANNELS_MID, 1, 2 * h, 2 * w], "conv3d の出口の形");
            assertEquals(outputs["y"].shape, [1, CHANNELS_OUT, h, w], "conv2d の出口の形");
            for (const [name, expected] of [["y3", reference.y3], ["y", reference.y]] as const) {
              const got = outputs[name].data;
              if (!(got instanceof Float32Array) || !(expected.data instanceof Float32Array)) {
                throw new Error(`${name}: f32 でない`);
              }
              const report = allclose(got, expected.data, CPU_REFERENCE_TOLERANCE);
              assert(
                report.pass,
                `${name}: CPU 参照と帯の外（maxAbs ${report.maxAbsError}・maxRel ${report.maxRelError}）`,
              );
              // 出力が定数でない（恒真化していない）こと。
              assert(new Set(got).size > 1, `${name}: 出力が定数`);
            }
            const conv3dKeys = requireCensus(
              session.diagnostics().lastRunPipelines,
              `run ${index + 1}`,
            )
              .map((row) => row.key)
              .filter((key) => key.startsWith("conv3d:"));
            assertEquals(conv3dKeys, [conv3dIgemmKey("f32", v4, mTile)], "踏んだ conv3d の変種");
            const [y3, y] = [outputs["y3"].data, outputs["y"].data];
            if (!(y3 instanceof Float32Array) || !(y instanceof Float32Array)) {
              throw new Error("y3 / y が f32 でない");
            }
            outputsByRun.push({ y3, y });
          },
        );
      }
      await t.step(
        "戻りの回（run 3）は 1 回目と Uint32 で一致する（conv3d の出口 y3 と最終の出口 y）",
        () => {
          const [first, , back] = outputsByRun;
          assert(first !== undefined && back !== undefined, "run 1 / 3 の出力が無い");
          const bits = (values: Float32Array): number[] => [
            ...new Uint32Array(values.buffer, values.byteOffset, values.length),
          ];
          assertEquals(
            bits(back.y3),
            bits(first.y3),
            "y3（conv3d の変種がスカラ → v4 → スカラと切り替わる）",
          );
          assertEquals(bits(back.y), bits(first.y), "y");
        },
      );
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});
