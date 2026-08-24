/**
 * `requiredLimits` の **limit 名の語彙**が hub と runtime で一致していることの門。
 *
 * hub の `RequiredLimitName`（manifest 所有の allowlist — ADR 0038 §3 / §7）と runtime の
 * `REQUIRED_LIMIT_KEYS`（`acquireGpu` が `requestDevice` へ渡す requiredLimits のキー —
 * src/gpu/device.ts）は**同じ語彙の写し**だが、hub は runtime に依存しないので互いを参照
 * できない。片方だけ名前が増減しても双方の型検査は緑のままで、ずれは「manifest が宣言した
 * limit が黙って要求されない」= **静かな頭打ち**としてしか現れない。`@karume/models` は両方へ
 * 依存できる唯一の位置なので、検出をここに置く。
 *
 * MUST: 検出は 2 段で、どちらが動いても**もう片方への追随を要求する**形にする。
 * ① {@link HUB_LIMIT_NAMES} は `Record<RequiredLimitName, true>` として型検査されるので、
 *    hub 側で名前が増減するとこのファイルが**型検査**で落ちる。
 * ② その表のキー集合を runtime の `REQUIRED_LIMIT_KEYS` と実行時に突き合わせるので、
 *    runtime 側で増減すると**このテスト**が落ちる。
 *
 * NOTE: 表を手で綴るのは「union は実行時に列挙できない」ため（派生状態の二重持ちではない —
 * 綴りは①が型で釘付けしており、黙って割れる余地が無い）。runtime 側の語彙は mod.ts の
 * 公開面に無い（薄い面 — ADR 0008）ので src から相対 import する。公開面を広げずに検査する
 * ためで、テストからのパッケージ跨ぎ相対 import は既存の先例（`../../runtime/tests/helpers/`）
 * と同じ流儀。
 */

import { assertEquals } from "@std/assert";
import type { RequiredLimitName } from "@karume/hub";
import { REQUIRED_LIMIT_KEYS } from "../../runtime/src/gpu/device.ts";

/** hub の語彙を値として立てた表（型が `RequiredLimitName` の網羅を強制する）。 */
const HUB_LIMIT_NAMES = {
  maxBufferSize: true,
  maxStorageBufferBindingSize: true,
  maxUniformBufferBindingSize: true,
  maxStorageBuffersPerShaderStage: true,
  maxUniformBuffersPerShaderStage: true,
  maxComputeWorkgroupStorageSize: true,
  maxComputeInvocationsPerWorkgroup: true,
  maxComputeWorkgroupSizeX: true,
  maxComputeWorkgroupSizeY: true,
  maxComputeWorkgroupSizeZ: true,
  maxComputeWorkgroupsPerDimension: true,
} as const satisfies Record<RequiredLimitName, true>;

Deno.test("requiredLimits の limit 名は hub の allowlist と runtime の要求キーで完全一致する", () => {
  assertEquals(
    Object.keys(HUB_LIMIT_NAMES).toSorted(),
    [...REQUIRED_LIMIT_KEYS].toSorted(),
    "hub の RequiredLimitName と runtime の REQUIRED_LIMIT_KEYS がずれている — " +
      "増えた側の名前をもう片方へ写すこと（hub 側だけに足すと manifest が受理した要求が " +
      "device へ届かず、runtime 側だけに足すと配布形からその limit を宣言できない）",
  );
});
