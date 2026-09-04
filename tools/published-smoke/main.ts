// 公開済み JSR パッケージからの疎通（docs/release-runbook.md §5）。
//
// ワークスペース配下で `jsr:@karume/*` を import すると Deno は**ローカルの workspace member に
// 解決する**ため、リポの deno.json の下で走らせても公開物の疎通にならない。このツールは自前の
// deno.json（workspace 非 member）で起動して registry の公開物を取りに行く（起動時に Deno が
// 「parent workspace を無視する」warning を 1 行出すのは想定どおり）。公開直後は依存の最小経過
// 時間チェックに掛かるので、同 deno.json が `jsr:@karume/*` だけを除外している。
//
// 検査 = models barrel が export する全 pin 定数（`*_CURRENT`）で manifest を解決し、1 家族
// （sbv2）は `fromPretrained` まで通す（GPU が要る — `--manifests-only` で省く）。
// 起動: deno task smoke:published [--manifests-only]

type PinRef = { readonly repo: string; readonly revision: string };
type HubModule = {
  loadManifest: (ref: PinRef) => Promise<{ manifest: { format?: unknown } }>;
};
type Sbv2Module = {
  Sbv2Pipeline: {
    fromPretrained: (ref: PinRef) => Promise<{ dispose: () => Promise<void> }>;
  };
};

const isPinRef = (value: unknown): value is PinRef =>
  typeof value === "object" && value !== null &&
  typeof (value as { repo?: unknown }).repo === "string" &&
  typeof (value as { revision?: unknown }).revision === "string";

const readPublishedVersion = async (): Promise<string> => {
  const url = new URL("../../packages/models/deno.json", import.meta.url);
  const parsed: unknown = JSON.parse(await Deno.readTextFile(url));
  const version = (parsed as { version?: unknown }).version;
  if (typeof version !== "string") {
    throw new Error(`version が読めない: ${url}`);
  }
  return version;
};

const manifestsOnly = Deno.args.includes("--manifests-only");
const version = await readPublishedVersion();
console.log(`published smoke: @karume/* @ ${version}`);

// 動的 import の指定子は実行時の文字列なので型は付かない — 使う面だけ構造で宣言して受ける。
const models = (await import(`jsr:@karume/models@${version}`)) as Record<
  string,
  unknown
>;
const hub = (await import(`jsr:@karume/hub@${version}`)) as HubModule;

const pins = Object.entries(models).filter(([name, value]) =>
  name.endsWith("_CURRENT") && isPinRef(value)
) as [string, PinRef][];
if (pins.length === 0) {
  throw new Error("models barrel に *_CURRENT の pin 定数が見つからない");
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
  const pin = models.SBV2_JVNV_CURRENT;
  if (!isPinRef(pin)) throw new Error("SBV2_JVNV_CURRENT が pin 定数でない");
  const { Sbv2Pipeline } = (await import(`jsr:@karume/models@${version}/sbv2`)) as Sbv2Module;
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
