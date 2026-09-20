/**
 * 実行環境の識別（デバイス別の参照値と結果を分けるキー）。
 *
 * sha256 の参照門が主張できるのは「**同じ GPU・同じランタイム**なら 1 ビットも動かない」こと
 * だけで、クロスデバイスのビット同一は仕様として保証しない（機序は docs/limitations.md の
 * 「sha256 参照門は参照環境専用」節）。参照値をテストのソースに定数で持つと、機を替えた瞬間に
 * 全門が赤になり「この機での退行を掴む」という門の役目そのものが消える。そこで参照値は
 * **環境ごとの行**で持ち、その行を索くキーをここが作る。
 *
 * MUST: キーは「同じ機なら毎回同じ」でなければならない。WebGPU から採れる識別子のうち、
 * 再起動やドライバ更新で揺れない面は `GPUAdapterInfo` の名前だけである（ドライバ版は API から
 * 採れない）。ドライバ更新で数値が動いた場合は**同じキーの行が割れる** = 赤で気づく形になる
 * （キーに版を混ぜて黙って別の行へ逃がさない）。
 *
 * 実装差: Deno は `description` に GPU 名を入れ `architecture` が空、Chrome は
 * `vendor` / `architecture` に名前を入れ `description` が空。どちらかは必ず埋まるので、
 * 埋まっている側を基底にする。
 */

import { readAdapterInfo } from "../../src/gpu/device.ts";

/** 参照値を分ける実行系（同じ GPU でもランタイムが違えば数値は動きうる）。 */
export type EnvironmentRuntime = "deno" | "chrome";

/** この実行環境の素性（結果 JSON にそのまま載る）。 */
export type Environment = {
  /** 参照値・結果を索くキー。GPU が無い環境では**持たない**（行を作れないため）。 */
  readonly key?: string;
  readonly runtime: {
    readonly name: EnvironmentRuntime;
    readonly version: string;
    readonly v8: string;
    readonly typescript: string;
  };
  readonly adapter: {
    readonly vendor: string;
    readonly architecture: string;
    readonly device: string;
    readonly description: string;
  };
  readonly os: {
    readonly platform: string;
    readonly arch: string;
  };
};

/** 基底から落とす商標の飾り（`Intel(R) Graphics` → `intel graphics`）。 */
const TRADEMARKS = ["(r)", "(tm)", "(c)"] as const;

/**
 * 環境キー（`<ランタイム>-<アダプタ名>`）。
 *
 * 例: Deno / `Intel(R) Graphics (BMG G21)` → `deno-intel-graphics-bmg-g21`、
 * Chrome / vendor `apple` + architecture `metal-3` → `chrome-apple-metal-3`。
 *
 * MUST: 名前が 1 つも採れない環境では throw する（空キーの行へ黙って混ぜると、別の機の
 * 参照値を「この機の参照値」として突き合わせることになる）。
 */
export const environmentKey = (runtime: EnvironmentRuntime, info: GPUAdapterInfo): string => {
  const base = info.description !== "" ? info.description : `${info.vendor}-${info.architecture}`;
  let slug = base.toLowerCase();
  for (const mark of TRADEMARKS) slug = slug.replaceAll(mark, "");
  slug = slug.replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  if (slug === "") {
    throw new Error(
      "GPUAdapterInfo から環境キーを作れない（description / vendor / architecture が全て空）。" +
        "参照値をこの環境の行として持てないので、アダプタ情報が読めない理由を先に確かめること",
    );
  }
  return `${runtime}-${slug}`;
};

/**
 * この実行環境を 1 回だけ確定する（アダプタ情報はテスト登録時点で要るので同期的に持ち回る）。
 *
 * GPU が無い環境では {@link Environment.key} を持たない — 参照値の行も結果の置き場も決め
 * られないためで、そこを空文字などで埋めると別環境の行と混ざる。
 */
export const describeEnvironment = async (): Promise<Environment> => {
  const gpu: GPU | undefined = navigator.gpu;
  const adapter = gpu === undefined ? null : await gpu.requestAdapter();
  // `readAdapterInfo` は `info` を持たない実装差を空値へ正規化する唯一の読み口。アダプタが
  // 無い場合も同じ空値を通す（`{}` は `info` 欠落と同じ形）。
  const info = readAdapterInfo(adapter ?? {});
  return {
    ...(adapter === null ? {} : { key: environmentKey("deno", info) }),
    runtime: {
      name: "deno",
      version: Deno.version.deno,
      v8: Deno.version.v8,
      typescript: Deno.version.typescript,
    },
    adapter: {
      vendor: info.vendor,
      architecture: info.architecture,
      device: info.device,
      description: info.description,
    },
    os: { platform: Deno.build.os, arch: Deno.build.arch },
  };
};

/** この実行環境（モジュール評価時に 1 回だけ確定）。 */
export const ENVIRONMENT: Environment = await describeEnvironment();
