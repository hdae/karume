/**
 * 容器のバイト列の常駐量の実測（`container_loading_test.ts` が別プロセスで起動する台本）。
 *
 * MUST: 別プロセスなのは `--v8-flags=--expose-gc` が要るため — `deno test` 側にその旗を
 * 立てる口が無く、旗が無いと「到達不能になったバイト列がまだ回収されていないだけ」と
 * 「握られたまま」を区別できない（測る前に必ず gc を 1 回踏む）。
 *
 * 測るのは `Deno.memoryUsage().external`（ArrayBuffer の実体の合計）。**キャッシュの中身を
 * 明示的に捨ててから**測ることで、残る外部メモリ = models 側が握っているバイト列だけになる。
 * `open(key)` は保持したまま測る（部品の供給口が生きている限り常駐する、という元の姿を
 * 再現するため）。コンテナ経路では供給口が握るのは**区間読みの口**だけで、重みの block は
 * Session を組むその瞬間にしか読まれない — だから重み 1 本ぶんが丸ごと残ったら回帰である。
 *
 * 失敗時は理由を stderr へ出して非ゼロ終了する（呼び手はその出力をそのまま見せる）。
 */

import { loadContainerComponents } from "../../src/hub/components.ts";
import { parseIrDeclarationValue } from "./container-fixture.ts";
import { type RepoSpec, serveContainer, serveRepos } from "./container-loading-fixture.ts";
import { MemoryCacheStorage } from "./memory-cache.ts";
import { loadManifest, resolveSelection } from "@karume/hub";
import type { ModelInput } from "../../../runtime/tests/helpers/container-write.ts";

const HUB_URL = "https://hub.test";
const REPO = "karume-test/retention";
const SHA = "0123456789abcdef0123456789abcdef01234567";

/** 重み 1 本の大きさ（2048² f32 = 16MiB）。 */
const WIDTH = 2048;
const WEIGHT_BYTES = WIDTH * WIDTH * 4;
/** 合格線: gc 後に残ってよい外部メモリ。握っていれば 16MiB が丸ごと残る。 */
const ALLOWED_RESIDENT_BYTES = 4 * 1024 * 1024;

/** `linear` 1 段。`w`（16MiB）が重みの block に載る。 */
const component = (): ModelInput => ({
  graphs: {
    dit: parseIrDeclarationValue({
      format: "karume-ir",
      version: 2,
      requires: { ops: ["linear"] },
      symbols: [],
      inputs: [{ name: "x", dtype: "f32", shape: [2, WIDTH] }],
      outputs: ["y"],
      initializers: { w: {}, b: {} },
      values: {
        w: { dtype: "f32", shape: [WIDTH, WIDTH] },
        b: { dtype: "f32", shape: [WIDTH] },
        y: { dtype: "f32", shape: [2, WIDTH] },
      },
      states: {},
      nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
    }),
  },
  consts: [],
  weights: [
    {
      graph: "dit",
      initializer: "w",
      bytes: new Uint8Array(new ArrayBuffer(WEIGHT_BYTES)),
      encoding: { codec: "f32" },
    },
    {
      graph: "dit",
      initializer: "b",
      bytes: new Uint8Array(new ArrayBuffer(WIDTH * 4)),
      encoding: { codec: "f32" },
    },
  ],
  assets: [],
  provenance: { license: "test" },
});

const mib = (bytes: number): string => `${(bytes / 1024 / 1024).toFixed(1)}MiB`;

/** 到達不能になった ArrayBuffer を確実に回収させてから外部メモリを読む。 */
const externalAfterGc = (): number => {
  const collect = (globalThis as { gc?: () => void }).gc;
  if (collect === undefined) {
    console.error("descriptor-retention: --v8-flags=--expose-gc 付きで起動していない");
    Deno.exit(2);
  }
  collect();
  collect();
  return Deno.memoryUsage().external;
};

/**
 * 配る現物の表を作る。**フィクスチャのバイト列を `main` の枠に置かない**ための独立関数 —
 * 実行中の関数のローカルは（内側から参照されていなくても）その枠が生きている限り回収されず、
 * 16MiB のフィクスチャがそのまま測定値へ乗ってしまう。
 */
const buildServed = async (): Promise<RepoSpec> => {
  const container = await serveContainer("dit/model.f32", component(), {
    partBytes: 32 * 1024 * 1024,
    blockBytes: 32 * 1024 * 1024,
  });
  return {
    repo: REPO,
    models: {
      test: {
        pipeline: "test/1",
        weights: { dit: { f32: container.entry } },
        assets: {},
        quants: { f32: { weights: { dit: "f32" }, session: {} } },
        defaultQuant: "f32",
        pipelineConfig: {},
      },
    },
    files: container.files,
  };
};

const main = async (): Promise<void> => {
  // MUST: `RepoSpec` を `main` の枠へ束縛しない（フィクスチャのバイト列が測定値へ乗る）。
  const mock = serveRepos([await buildServed()]);
  const { fetch } = mock;

  const caches = new MemoryCacheStorage();
  const hubOptions = { fetch, caches };
  const loaded = await loadManifest({ repo: REPO, revision: SHA, hubUrl: HUB_URL }, hubOptions);
  const { open } = await loadContainerComponents(
    "test.fromPretrained",
    loaded,
    resolveSelection(loaded.manifest),
    ["dit"],
    // 家族 admission の席（この台本が測るのは常駐量なので、門は 1 つも置かない）。
    () => undefined,
    hubOptions,
  );

  // 供給口は保持したまま、models の外にあるバイト列（フィクスチャ・キャッシュ）を捨てる —
  // 残る外部メモリ = models 側が握っているぶん、という等式を成立させるため（疑似 HF の表は
  // `fetch` の閉包が握るので、取得面が生きている限り勝手には消えない）。
  mock.clear();
  for (const cache of caches.namespaces.values()) cache.entries.clear();

  const resident = externalAfterGc();
  // `open` を gc の後まで生かす（ここで初めて到達可能性が切れる）。
  if (open("dit").graph.outputs[0] !== "y") {
    console.error("descriptor-retention: グラフ宣言が読めない（フィクスチャの誤り）");
    Deno.exit(2);
  }

  if (resident > ALLOWED_RESIDENT_BYTES) {
    console.error(
      `descriptor-retention: 容器のバイト列が常駐している` +
        `（external=${mib(resident)} > 許容 ${mib(ALLOWED_RESIDENT_BYTES)}` +
        ` / 重み 1 本 ${mib(WEIGHT_BYTES)}）`,
    );
    Deno.exit(1);
  }
  console.log(`descriptor-retention: external=${mib(resident)}`);
};

await main();
