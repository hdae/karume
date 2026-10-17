// 環境キー（デバイス別の参照値を索く ID）の綴り方を固定する。
//
// キーが実装差（Deno は description / Chrome は vendor + architecture）を吸収できていないと、
// 参照値が「別の機の行」へ黙って書かれるか、機を替えるたびに新しい行が増える。どちらも赤に
// ならないまま門の意味が消えるので、綴りの規則そのものをここで縛る。

import { platformObject } from "@karume/test-support/webgpu";
import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import {
  assertAdapterMatchesEnvironment,
  describeEnvironment,
  ENVIRONMENT,
  type Environment,
  environmentKey,
  verifyRunningGpu,
} from "./helpers/environment.ts";

/** `GPUAdapterInfo` の実体（読むのは 4 欄だけだが、型は全欄を要求する）。 */
const adapterInfo = (fields: Partial<Omit<GPUAdapterInfo, "__brand">>): GPUAdapterInfo =>
  platformObject<GPUAdapterInfo>({
    vendor: "",
    architecture: "",
    device: "",
    description: "",
    subgroupMinSize: 0,
    subgroupMaxSize: 0,
    isFallbackAdapter: false,
    ...fields,
  });

test("環境キー: Node（Dawn）は device から作る（description はドライバ版なので使わない）", () => {
  assertEquals(
    environmentKey(
      "node",
      adapterInfo({
        vendor: "nvidia",
        architecture: "ampere",
        device: "nvidia-geforce-rtx-3080-ti",
        description: "NVIDIA: 615.71.09 615.71.9.0",
      }),
    ),
    "node-nvidia-geforce-rtx-3080-ti",
  );
});

test("環境キー: Deno は description から作る（商標の飾りは落ちる）", () => {
  // 実測値（Intel Arc B570 / Deno 2.9 — architecture は空で device は数値 ID）。
  assertEquals(
    environmentKey(
      "deno",
      adapterInfo({ vendor: "32902", device: "57868", description: "Intel(R) Graphics (BMG G21)" }),
    ),
    "deno-intel-graphics-bmg-g21",
  );
});

test("環境キー: 空白と大文字は潰れる（RTX の記録上の綴り）", () => {
  assertEquals(
    environmentKey(
      "deno",
      adapterInfo({ vendor: "4318", description: "NVIDIA GeForce RTX 3080 Ti" }),
    ),
    "deno-nvidia-geforce-rtx-3080-ti",
  );
});

test("環境キー: description が空なら vendor + architecture から作る（Chrome 形）", () => {
  assertEquals(
    environmentKey("chrome", adapterInfo({ vendor: "apple", architecture: "metal-3" })),
    "chrome-apple-metal-3",
  );
});

test("環境キー: 名前が 1 つも採れなければ throw する（空キーの行を作らない）", () => {
  assertThrows(
    () => environmentKey("deno", adapterInfo({ device: "57868" })),
    Error,
    "環境キーを作れない",
  );
});

test("環境キー: 走行が取ったアダプタが環境キーと違えば throw する", () => {
  const environment: Environment = {
    key: "deno-intel-graphics-bmg-g21",
    runtime: { name: "deno", version: "2.9.6", v8: "15.0.245.2", typescript: "6.0.3" },
    adapter: {
      vendor: "32902",
      architecture: "",
      device: "57868",
      description: "Intel(R) Graphics (BMG G21)",
    },
    os: { platform: "linux", arch: "x86_64" },
  };
  const context = (description: string): { readonly adapterInfo: GPUAdapterInfo } => ({
    adapterInfo: adapterInfo({ description }),
  });
  // 対照: 同じアダプタなら何も起きない。
  assertAdapterMatchesEnvironment(context("Intel(R) Graphics (BMG G21)"), environment);
  assertThrows(
    () => assertAdapterMatchesEnvironment(context("NVIDIA GeForce RTX 3080 Ti"), environment),
    Error,
    "この走行が実際に取ったアダプタのキー",
  );
});

test("環境キー: この実行環境の素性は runtime / os を必ず名乗る", async () => {
  const environment = await describeEnvironment();
  assertEquals(environment.runtime.name, "node");
  assertEquals(environment.runtime.version, process.versions.node);
  assertEquals(environment.os, { platform: process.platform, arch: process.arch });
  // GPU が無い環境では key を持たない（持つ場合は必ずランタイム名で始まる）。
  assert(
    ENVIRONMENT.key === undefined || ENVIRONMENT.key.startsWith("node-"),
    `環境キーの綴りが想定外: ${String(ENVIRONMENT.key)}`,
  );
});

test("環境キー: 実走の GPU は取得のたびに照合し、外れた取得は破棄してから throw する", () => {
  // 複数 GPU の機で取得のたびに別のアダプタが選ばれうる（2 回の requestAdapter が同じ物理アダプタを
  // 返す保証は無い）。検査用に 1 度だけ取って結果をキャッシュする形だと、A→A→B の 3 度目の B が
  // 照合されないまま A の行と突き合わされる（書き込み運用なら A の行に B の出力が入る）。
  const environment: Environment = {
    key: "deno-intel-graphics-bmg-g21",
    runtime: { name: "deno", version: "2.9.6", v8: "15.0.245.2", typescript: "6.0.3" },
    adapter: {
      vendor: "32902",
      architecture: "",
      device: "57868",
      description: "Intel(R) Graphics (BMG G21)",
    },
    os: { platform: "linux", arch: "x86_64" },
  };
  const destroyed: string[] = [];
  const acquire = (description: string) => ({
    adapterInfo: adapterInfo({ description }),
    destroy: (): void => {
      destroyed.push(description);
    },
  });
  const sequence = [
    "Intel(R) Graphics (BMG G21)",
    "Intel(R) Graphics (BMG G21)",
    "NVIDIA GeForce RTX 3080 Ti",
  ];
  const [first, second, third] = sequence.map(acquire);
  // 照合を通った取得はそのまま返る（呼び手が使って、呼び手が破棄する）。
  assert(verifyRunningGpu(first, environment) === first);
  assert(verifyRunningGpu(second, environment) === second);
  assertEquals(destroyed, []);
  assertThrows(
    () => verifyRunningGpu(third, environment),
    Error,
    "この走行が実際に取ったアダプタのキー",
  );
  // 呼び手の try より前で使えるように、外れた取得は自分で返してから落ちる。
  assertEquals(destroyed, ["NVIDIA GeForce RTX 3080 Ti"]);
});
