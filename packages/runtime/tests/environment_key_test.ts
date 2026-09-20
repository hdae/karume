// 環境キー（デバイス別の参照値を索く ID）の綴り方を固定する。
//
// キーが実装差（Deno は description / Chrome は vendor + architecture）を吸収できていないと、
// 参照値が「別の機の行」へ黙って書かれるか、機を替えるたびに新しい行が増える。どちらも赤に
// ならないまま門の意味が消えるので、綴りの規則そのものをここで縛る。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { describeEnvironment, ENVIRONMENT, environmentKey } from "./helpers/environment.ts";

/** `GPUAdapterInfo` の実体（読むのは 4 欄だけだが、型は全欄を要求する）。 */
const adapterInfo = (fields: Partial<GPUAdapterInfo>): GPUAdapterInfo => ({
  vendor: "",
  architecture: "",
  device: "",
  description: "",
  subgroupMinSize: 0,
  subgroupMaxSize: 0,
  isFallbackAdapter: false,
  ...fields,
});

Deno.test("環境キー: Deno は description から作る（商標の飾りは落ちる）", () => {
  // 実測値（Intel Arc B570 / Deno 2.9 — architecture は空で device は数値 ID）。
  assertEquals(
    environmentKey(
      "deno",
      adapterInfo({ vendor: "32902", device: "57868", description: "Intel(R) Graphics (BMG G21)" }),
    ),
    "deno-intel-graphics-bmg-g21",
  );
});

Deno.test("環境キー: 空白と大文字は潰れる（RTX の記録上の綴り）", () => {
  assertEquals(
    environmentKey(
      "deno",
      adapterInfo({ vendor: "4318", description: "NVIDIA GeForce RTX 3080 Ti" }),
    ),
    "deno-nvidia-geforce-rtx-3080-ti",
  );
});

Deno.test("環境キー: description が空なら vendor + architecture から作る（Chrome 形）", () => {
  assertEquals(
    environmentKey("chrome", adapterInfo({ vendor: "apple", architecture: "metal-3" })),
    "chrome-apple-metal-3",
  );
});

Deno.test("環境キー: 名前が 1 つも採れなければ throw する（空キーの行を作らない）", () => {
  assertThrows(
    () => environmentKey("deno", adapterInfo({ device: "57868" })),
    Error,
    "環境キーを作れない",
  );
});

Deno.test("環境キー: この実行環境の素性は runtime / os を必ず名乗る", async () => {
  const environment = await describeEnvironment();
  assertEquals(environment.runtime.name, "deno");
  assertEquals(environment.runtime.version, Deno.version.deno);
  assertEquals(environment.os, { platform: Deno.build.os, arch: Deno.build.arch });
  // GPU が無い環境では key を持たない（持つ場合は必ずランタイム名で始まる）。
  assert(
    ENVIRONMENT.key === undefined || ENVIRONMENT.key.startsWith("deno-"),
    `環境キーの綴りが想定外: ${String(ENVIRONMENT.key)}`,
  );
});
