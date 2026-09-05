// 公開済み JSR パッケージからの疎通（docs/release-runbook.md §5）。
//
// ワークスペース配下で `jsr:@karume/*` を import すると Deno は**ローカルの workspace member に
// 解決する**ため、リポの deno.json の下で走らせても公開物の疎通にならない。このツールは自前の
// deno.json（workspace 非 member）で起動して registry の公開物を取りに行く（起動時に Deno が
// 「parent workspace を無視する」warning を 1 行出すのは想定どおり）。公開直後は依存の最小経過
// 時間チェックに掛かるので、同 deno.json が `jsr:@karume/*` だけを除外している。
//
// 検査 = models barrel の取得元対応表（`KARUME_SOURCES` — 全家族の和集合。ADR 0092）の全
// エントリで manifest を解決し、1 家族（sbv2）は `fromPretrained` まで通す（GPU が要る —
// `--manifests-only` で省く）。
// 起動: deno task smoke:published [--manifests-only]

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

/**
 * 動的 import で受けた値を**キー付きの表として引ける形**へ絞る。`typeof === "object"` だけだと
 * 型は `object` にしかならず、`Object.entries` の戻りが `any` に落ちて以降の検査が型に効かない。
 */
const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null;

const readPublishedVersion = async (): Promise<string> => {
  const url = new URL("../../packages/models/deno.json", import.meta.url);
  const parsed: unknown = JSON.parse(await Deno.readTextFile(url));
  const version = (parsed as { version?: unknown }).version;
  if (typeof version !== "string") {
    throw new Error(`version が読めない: ${url}`);
  }
  return version;
};

const { manifestsOnly } = parseSmokeArgs(Deno.args);
const version = await readPublishedVersion();
console.log(`published smoke: @karume/* @ ${version}`);

// 動的 import の指定子は実行時の文字列なので型は付かない — 使う面だけ構造で宣言して受ける。
const models = (await import(`jsr:@karume/models@${version}`)) as Record<
  string,
  unknown
>;
const hub = (await import(`jsr:@karume/hub@${version}`)) as HubModule;

const sources: unknown = models.KARUME_SOURCES;
if (!isRecord(sources)) {
  throw new Error("models barrel に KARUME_SOURCES が見つからない");
}
// filter は tuple の第 2 要素を絞れないので flatMap で型ごと絞る（as を使わない）。
const pins = Object.entries(sources).flatMap(([name, value]) =>
  isPinRef(value) ? [[name, value] as const] : []
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
    `  manifest OK ${name} ${ref.repo}@${ref.revision.slice(0, 8)} format=${format} (${
      (performance.now() - started).toFixed(0)
    }ms)`,
  );
}

if (manifestsOnly) {
  console.log(
    `published smoke OK: ${pins.length} manifests（fromPretrained は省略）`,
  );
} else {
  const { Sbv2Pipeline, SBV2_SOURCES } =
    (await import(`jsr:@karume/models@${version}/sbv2`)) as Sbv2Module;
  const pin: unknown = SBV2_SOURCES["sbv2-jvnv"];
  if (!isPinRef(pin)) throw new Error('SBV2_SOURCES["sbv2-jvnv"] が pin エントリでない');
  const started = performance.now();
  const pipeline = await Sbv2Pipeline.fromPretrained(pin);
  console.log(
    `  fromPretrained OK sbv2-jvnv (${((performance.now() - started) / 1000).toFixed(1)}s)`,
  );
  await pipeline.dispose();
  console.log(
    `published smoke OK: ${pins.length} manifests + sbv2 fromPretrained`,
  );
}
