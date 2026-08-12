/**
 * Irodori の実 GPU E2E 門が共有する**読み口**（配布形の資産と full-loop golden）。
 *
 * 門そのものは席ごとに別ファイルへ分かれる（格納 dtype の席は数値パリティ網 =
 * `e2e_irodori_latent_test.ts`・活性量子化の `w8a8` 席は判別帯 + キー census =
 * `e2e_irodori_w8a8_test.ts` — 網の性格が違うので同じ表には乗らない）。共有するのは
 * 「どこから何を読むか」だけで、**何を要求するかは各門が持つ**。
 *
 * NOTE: ローカル読みに `Deno.readFile` を使うのはテストだけ。パッケージ本体は Web 標準 API
 * のみ（横断不変条件）で、`fromAssets` はバイト列を受け取るだけの面。
 */

import { parseManifest, resolveFiles } from "@karume/hub";
import type { Manifest, ModelEntry } from "@karume/hub";
import { parseSafetensors } from "@karume/runtime";

/** 配布形の置き場（`karume dist --pipeline irodori` の既定の出力先）。 */
export const ASSETS_DIR = new URL("../../../../models/karume-irodori-v4-small/", import.meta.url);

/** SKIP 時にそのまま貼れる生成コマンド。 */
export const DIST_COMMAND = "cd tools/exporter && uv run karume dist --pipeline irodori";
export const GOLDEN_COMMAND =
  "cd tools/exporter && uv run --with 'transformers==5.14.1' python irodori_pipeline.py";

export const MODEL = "v4-small";

/** 系列 root（格納 dtype の名前）から full-loop golden の置き場を組む。 */
export const goldenDir = (seriesRoot: string): URL =>
  new URL(`../../../../outputs/series/${seriesRoot}/pipeline/`, import.meta.url);

/** 配布形の manifest（無い環境では undefined = 焼いていない）。 */
export const manifestText: string | undefined = await Deno.readTextFile(
  new URL("karume.json", ASSETS_DIR),
).catch(() => undefined);

/** golden `meta.json` のうち、門が読む欄だけ（形は exporter が持つ — ここは読み口）。 */
export type GoldenCase = {
  readonly text: string;
  readonly caption: string;
  readonly S: number;
  readonly forwards: number;
  readonly zAbsMax: number;
  readonly reference: { readonly frames: number } | null;
};

export type GoldenMeta = {
  readonly steps: number;
  readonly initScale: number;
  readonly cfgRange: readonly [number, number];
  readonly cfgScales: { readonly text: number; readonly speaker: number; readonly caption: number };
  readonly frameRate: number;
  readonly tEmbedDim: number;
  readonly caps: { readonly text: number; readonly speaker: number; readonly caption: number };
  readonly cases: Readonly<Record<string, GoldenCase>>;
};

export const readManifest = (): Manifest => parseManifest(manifestText as string);

export const modelEntry = (manifest: Manifest): ModelEntry => {
  if (!Object.hasOwn(manifest.models, MODEL)) {
    throw new Error(
      `配布形に model '${MODEL}' が無い（あるもの: ${manifest.available.models.join(" / ")}）`,
    );
  }
  return manifest.models[MODEL];
};

/** 配布形にこの quant 席があるか（席が無い環境は「組み直していない」= SKIP 側）。 */
export const hasQuantSeat = (quant: string): boolean =>
  manifestText !== undefined && Object.hasOwn(modelEntry(readManifest()).quants, quant);

/** 配布形が要求する資産をローカルから読む（`fetchAssets` のローカル版 — 取得層を通さない）。 */
export const loadLocalAssets = async (
  manifest: Manifest,
  quant: string,
): Promise<Record<string, Uint8Array<ArrayBuffer>>> => {
  const files = resolveFiles(manifest, { model: MODEL, quant });
  const byPath = new Map<string, Uint8Array<ArrayBuffer>>();
  let assets: Record<string, Uint8Array<ArrayBuffer>> = {};
  for (const key of Object.keys(files)) {
    const { path } = files[key];
    const cached = byPath.get(path);
    const bytes = cached ?? await Deno.readFile(new URL(path, ASSETS_DIR));
    if (cached === undefined) byPath.set(path, bytes);
    assets = { ...assets, [key]: bytes };
  }
  return assets;
};

/** golden の 1 ケース（`case.<name>.safetensors`）から f32 テンソルを引く。 */
export const readCase = async (
  directory: URL,
  name: string,
): Promise<(tensor: string) => Float32Array<ArrayBuffer>> => {
  const bytes = await Deno.readFile(new URL(`case.${name}.safetensors`, directory));
  const file = parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  return (tensor: string): Float32Array<ArrayBuffer> => {
    const spec = file.tensors.get(tensor);
    if (spec === undefined) throw new Error(`golden case.${name} に '${tensor}' が無い`);
    if (spec.dtype !== "F32") throw new Error(`golden '${tensor}' の dtype が ${spec.dtype}`);
    // MUST: 写して返す（golden の buffer は 1 本を全テンソルで共有しており、`initialNoise` は
    // パイプラインが**そのまま書き換える**配列になる — 借りたままだと 2 回目の突合が壊れる）。
    return new Float32Array(
      file.buffer.slice(spec.byteOffset, spec.byteOffset + spec.byteLength),
    ) as Float32Array<ArrayBuffer>;
  };
};

/** 全要素の最大絶対差と、その位置（差分の報告用）。 */
export const worstDifference = (
  actual: Float32Array,
  expected: Float32Array,
): { readonly maxAbs: number; readonly at: number } => {
  if (actual.length !== expected.length) {
    throw new Error(`要素数が違う（実測 ${actual.length} / golden ${expected.length}）`);
  }
  let maxAbs = 0;
  let at = 0;
  for (let index = 0; index < actual.length; index += 1) {
    const difference = Math.abs(actual[index] - expected[index]);
    if (difference > maxAbs) {
      maxAbs = difference;
      at = index;
    }
  }
  return { maxAbs, at };
};
