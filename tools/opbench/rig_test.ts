// summary.json の rig 欄（GPU の名乗り）が空にならないことの門（GPU 不要）。
//
// `GPUAdapterInfo` の 4 欄は prototype 上の getter なので、素のオブジェクトを fixture にすると
// spread でも欄が写ってしまい「spread へ戻した」退行を検出できない。ここでは実物と同じ
// 「prototype に getter・インスタンスは空」の形を `Object.create` で作って固定する。

import { assertEquals } from "@std/assert";
import type { CensusSummary } from "./census.ts";
import { type AdapterInfoFields, buildSingleSummary, rigOf } from "./single.ts";

/** 実物と同じ「4 欄は prototype 上の getter」な adapterInfo（インスタンスは自前の欄を持たない）。 */
const adapterInfoWithGetters = (): AdapterInfoFields => {
  const prototype: AdapterInfoFields = {
    get vendor(): string {
      return "intel";
    },
    get architecture(): string {
      return "xe-2lpg";
    },
    get device(): string {
      return "0x7d55";
    },
    get description(): string {
      return "Intel(R) Arc(TM) Graphics";
    },
  };
  return Object.create(prototype);
};

const CENSUS: CensusSummary = {
  generated_at: "2026-09-22T00:00:00.000Z",
  source: "synthetic",
  family: "unit",
  model: "unit",
  quant: "i4",
  session: null,
  scenarios: [],
};

Deno.test("rigOf: prototype 上の getter しか持たない adapterInfo でも 4 欄が残る", () => {
  const adapterInfo = adapterInfoWithGetters();

  assertEquals(rigOf({ adapterInfo }), {
    vendor: "intel",
    architecture: "xe-2lpg",
    device: "0x7d55",
    description: "Intel(R) Arc(TM) Graphics",
    deno: Deno.version.deno,
  });
});

Deno.test("rigOf: fixture 自体は spread で空になる（この門が検出したい退行の形）", () => {
  // 逆側の固定 — 素のオブジェクトを fixture にすると spread でも通ってしまい、門が無意味になる。
  assertEquals({ ...adapterInfoWithGetters() }, {});
});

Deno.test("buildSingleSummary: rig に GPU の名乗りと計測規約の 2 欄が載る", () => {
  const summary = buildSingleSummary(
    "outputs/bench/unit/census",
    CENSUS,
    { adapterInfo: adapterInfoWithGetters() },
    { session: {}, mode: "timing", rounds: 3 },
    [],
    {},
    [],
  );

  assertEquals(summary.rig.vendor, "intel");
  assertEquals(summary.rig.architecture, "xe-2lpg");
  assertEquals(summary.rig.device, "0x7d55");
  assertEquals(summary.rig.description, "Intel(R) Arc(TM) Graphics");
  assertEquals(summary.rig.rounds, 3);
});
