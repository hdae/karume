/**
 * Shared helpers for reading `tests/fixtures/manifest-fetch.json` (`karume/5`).
 *
 * The fixture is in the **container distribution form** (ADR 0109), so one component has a list of
 * parts. Tests look up those paths by name (`TEXT_ENCODER_PARTS` and so on). The tools that build
 * selection tables and serving tables are in `helpers/selection.ts` (generic over any manifest).
 *
 * MUST: do not copy paths by hand — keep the constants in a form that moves along when the fixture
 * is regenerated (a copy silently leads to "the fixture was fixed but the tests keep serving the
 * old
 * path").
 */

import { readTextFile } from "@karume/test-support/fs";
import { type FileRef, type Manifest, parseManifest } from "../../mod.ts";

export const FETCH_MANIFEST_TEXT = await readTextFile(
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

/** part 0 + part 1 + part 2 (every container in the fixture has 3). */
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
/** Only the i8 container has a **zero-length part 1** (the shape of ADR 0109 decision 3). */
export const TRANSFORMER_I8_PARTS: readonly string[] = pathsOf("anima-turbo", "transformer", "i8");
export const VAE_DECODER_PARTS: readonly string[] = pathsOf("anima-turbo", "vae_decoder", "f16");

/** The i8 fetch targets without the zero-length part (these 2 are what the selection lists). */
export const TRANSFORMER_I8_FETCHED: readonly string[] = partsOf("anima-turbo", "transformer", "i8")
  .filter((ref) => ref.size > 0)
  .map((ref) => ref.path);

/** The path of the zero-length part (one that is neither fetched nor read). */
export const TRANSFORMER_I8_EMPTY: string = partsOf("anima-turbo", "transformer", "i8")
  .filter((ref) => ref.size === 0)
  .map((ref) => ref.path)[0];

export const TOKENIZER = "tokenizer/qwen2-tokenizer.json";
export const STYLE_VECTORS = "style/style_vectors.safetensors";
