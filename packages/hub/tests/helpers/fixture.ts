/**
 * `tests/fixtures/manifest-fetch.json`（`karume/5`）を読む側の共有ヘルパ。
 *
 * fixture は**コンテナ配布形**（ADR 0109）なので、1 つの部品が part 列を持つ。テストはその
 * path を名前で引く（`TEXT_ENCODER_PARTS` 等）。選択の表と配信表を組む道具は
 * `helpers/selection.ts`（manifest を選ばない汎用）。
 *
 * MUST: path を手で写さない — fixture を焼き直したら constants も一緒に動く形にしておく
 * （写すと「fixture を直したのにテストが古い path を配信し続ける」が黙って起きる）。
 */

import { type FileRef, type Manifest, parseManifest } from "../../mod.ts";

export const FETCH_MANIFEST_TEXT = await Deno.readTextFile(
  new URL("../fixtures/manifest-fetch.json", import.meta.url),
);

export const FETCH_MANIFEST_BYTES: Uint8Array<ArrayBuffer> = new TextEncoder().encode(
  FETCH_MANIFEST_TEXT,
);

export const fetchManifest: Manifest = parseManifest(FETCH_MANIFEST_TEXT);

const partsOf = (model: string, weight: string, dtype: string): readonly FileRef[] =>
  fetchManifest.models[model].weights[weight][dtype].container.parts;

const pathsOf = (model: string, weight: string, dtype: string): readonly string[] =>
  partsOf(model, weight, dtype).map((ref) => ref.path);

/** part 0 + part 1 + part 2（fixture の容器はどれも 3 本）。 */
export const TEXT_ENCODER_PARTS: readonly string[] = pathsOf("anima-turbo", "text_encoder", "f16");
export const TEXT_CONDITIONER_PARTS: readonly string[] = pathsOf(
  "anima-turbo",
  "text_conditioner",
  "f16",
);
export const TRANSFORMER_F16_PARTS: readonly string[] = pathsOf(
  "anima-turbo",
  "transformer",
  "f16",
);
/** i8 の容器だけが**長さ 0 の part 1**を持つ（ADR 0109 決定 3 の形）。 */
export const TRANSFORMER_I8_PARTS: readonly string[] = pathsOf("anima-turbo", "transformer", "i8");
export const VAE_DECODER_PARTS: readonly string[] = pathsOf("anima-turbo", "vae_decoder", "f16");

/** 長さ 0 の part を除いた i8 の取得対象（選択の列に載るのはこの 2 本）。 */
export const TRANSFORMER_I8_FETCHED: readonly string[] = partsOf(
  "anima-turbo",
  "transformer",
  "i8",
).filter((ref) => ref.size > 0).map((ref) => ref.path);

/** 長さ 0 の part の path（取得も読みも起きない 1 本）。 */
export const TRANSFORMER_I8_EMPTY: string = partsOf("anima-turbo", "transformer", "i8")
  .filter((ref) => ref.size === 0).map((ref) => ref.path)[0];

export const TOKENIZER = "tokenizer/qwen2-tokenizer.json";
export const STYLE_VECTORS = "style/style_vectors.safetensors";
