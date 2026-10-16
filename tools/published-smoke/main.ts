// 公開済み JSR パッケージからの疎通（docs/release-runbook.md §5）。
//
// Node（主）と Deno（従 — Deno から公開物が import できることの確認）の両方で同じファイルを走らせる。
// 公開物は bare 指定子（`@karume/hub` / `@karume/models`）で引き、その解決先を registry の公開物に
// 固定するのはこのディレクトリ自前の設定:
//   - Node: package.json の `jsr:^<版>` 依存（pnpm が npm.jsr.io から入れる — workspace:* ではない）
//   - Deno: deno.json の import map（`jsr:@karume/*@<版>`）。ワークスペース配下で `jsr:@karume/*` を
//     引くと Deno は**ローカルの workspace member に解決する**ので、自前の deno.json（workspace 非
//     member）で起動する（起動時に「parent workspace を無視する」warning が 1 行出るのは想定どおり）。
//     公開直後は依存の最小経過時間チェックに掛かるので、同 deno.json が `jsr:@karume/*` だけを除外している。
// 解決された版は期待の版（packages/models/package.json）と突き合わせ、食い違えば落とす。
//
// 検査 = models barrel の取得元対応表（`KARUME_SOURCES` — 全家族の和集合。ADR 0092）の全
// エントリで manifest を解決し、1 家族（sbv2）は `fromPretrained` まで通す（GPU が要る —
// `--manifests-only` で省く）。
// 起動: node tools/published-smoke/main.ts [--manifests-only]
//       deno run -A main.ts [--manifests-only]（tools/published-smoke で）

import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { isPinRef, parseSmokeArgs, type PinRef } from "./cli.ts";

type HubModule = {
  loadManifest: (ref: PinRef) => Promise<{ manifest: { format?: unknown } }>;
};
type Sbv2Module = {
  Sbv2Pipeline: {
    fromPretrained: (ref: PinRef) => Promise<{ dispose: () => Promise<void> }>;
  };
  SBV2_SOURCES: Record<string, unknown>;
};
type DawnModule = { create: (flags: string[]) => GPU };

// 動的 import の指定子は `string` 型の変数で渡す（型は付かない — 使う面だけ構造で宣言して受ける）。
// 型付けの対象はワークスペースの版ではなく公開物で、リテラルにすると Deno の型検査はワークスペース
// member へ、tsc は未導入の node_modules へ解決しに行く。
const HUB: string = "@karume/hub";
const MODELS: string = "@karume/models";
const MODELS_SBV2: string = "@karume/models/sbv2";
const DAWN: string = "webgpu";

/**
 * 動的 import で受けた値を**キー付きの表として引ける形**へ絞る。`typeof === "object"` だけだと
 * 型は `object` にしかならず、`Object.entries` の戻りが `any` に落ちて以降の検査が型に効かない。
 */
const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null;

const readVersion = async (path: string): Promise<string> => {
  const parsed: unknown = JSON.parse(await readFile(path, "utf8"));
  const version = isRecord(parsed) ? parsed.version : undefined;
  if (typeof version !== "string") {
    throw new Error(`version が読めない: ${path}`);
  }
  return version;
};

/**
 * bare 指定子が実際に解決される公開版。
 *
 * - Deno は import map の jsr 指定子を版の解決前の綴りのまま返す（`jsr:@karume/hub@0.14.0`）—
 *   deno.json の imports は完全一致の版で書く（`^` を付けると期待の版と一致せず落ちる）。
 * - Node は導入済みパッケージ内の file URL を返す — 最寄りの package.json が版を持つ。
 */
const resolvedVersion = async (specifier: string): Promise<string> => {
  const resolved = import.meta.resolve(specifier);
  const jsr = /^jsr:\/?@karume\/[^@/]+@([^/]+)/.exec(resolved)?.[1];
  if (jsr !== undefined) return jsr;
  if (resolved.startsWith("file:")) {
    let dir = dirname(fileURLToPath(resolved));
    while (!existsSync(join(dir, "package.json"))) {
      const parent = dirname(dir);
      if (parent === dir) break;
      dir = parent;
    }
    return await readVersion(join(dir, "package.json"));
  }
  throw new Error(`${specifier} の解決先から版が読めない: ${resolved}`);
};

/**
 * Node には `navigator.gpu` が無く、公開版の `fromPretrained` は `navigator.gpu` からアダプタを取る。
 * Node でだけ Dawn（npm `webgpu`）を `navigator.gpu` として置く（ブラウザと Deno が持つ形 —
 * test-support の setup と同じ置き方）。
 */
const ensureNavigatorGpu = async (): Promise<void> => {
  const current: GPU | undefined = navigator.gpu;
  if (current !== undefined) return;
  const { create } = (await import(DAWN)) as DawnModule;
  Object.defineProperty(navigator, "gpu", {
    value: create([]),
    configurable: true,
    enumerable: true,
  });
};

const { manifestsOnly } = parseSmokeArgs(process.argv.slice(2));
const version = await readVersion(
  fileURLToPath(new URL("../../packages/models/package.json", import.meta.url)),
);
console.log(`published smoke: @karume/* @ ${version}`);

for (const specifier of [HUB, MODELS]) {
  const resolved = await resolvedVersion(specifier);
  if (resolved !== version) {
    throw new Error(
      `${specifier} が ${resolved} に解決された（期待 ${version} — packages/models/package.json）。` +
        "Node は tools/published-smoke/package.json の依存を上げて pnpm install、" +
        "Deno は tools/published-smoke/deno.json の imports を合わせる",
    );
  }
}

const models = (await import(MODELS)) as Record<string, unknown>;
const hub = (await import(HUB)) as HubModule;

const sources: unknown = models.KARUME_SOURCES;
if (!isRecord(sources)) {
  throw new Error("models barrel に KARUME_SOURCES が見つからない");
}
// filter は tuple の第 2 要素を絞れないので flatMap で型ごと絞る（as を使わない）。
const pins = Object.entries(sources).flatMap(([name, value]) =>
  isPinRef(value) ? [[name, value] as const] : [],
);
if (pins.length === 0) {
  throw new Error("KARUME_SOURCES にエントリが 1 本も無い");
}

for (const [name, ref] of pins) {
  const started = performance.now();
  const loaded = await hub.loadManifest(ref);
  const format = loaded.manifest.format;
  if (typeof format !== "string") {
    throw new Error(`${name}: manifest.format が文字列でない`);
  }
  console.log(
    `  manifest OK ${name} ${ref.repo}@${ref.revision.slice(0, 8)} format=${format} (${(
      performance.now() - started
    ).toFixed(0)}ms)`,
  );
}

if (manifestsOnly) {
  console.log(`published smoke OK: ${pins.length} manifests（fromPretrained は省略）`);
} else {
  await ensureNavigatorGpu();
  const { Sbv2Pipeline, SBV2_SOURCES } = (await import(MODELS_SBV2)) as Sbv2Module;
  const pin: unknown = SBV2_SOURCES["sbv2-jvnv"];
  if (!isPinRef(pin)) throw new Error('SBV2_SOURCES["sbv2-jvnv"] が pin エントリでない');
  const started = performance.now();
  const pipeline = await Sbv2Pipeline.fromPretrained(pin);
  console.log(
    `  fromPretrained OK sbv2-jvnv (${((performance.now() - started) / 1000).toFixed(1)}s)`,
  );
  await pipeline.dispose();
  console.log(`published smoke OK: ${pins.length} manifests + sbv2 fromPretrained`);
}
