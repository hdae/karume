/**
 * SBV2's **style check** (real GPU): the `style` / `styleWeight` knobs of `generate` reach the
 * audio.
 *
 * The style travels a path no other check fails on when it dies: the host builds the style vector
 * from the distribution's table (`mean + (picked − mean) · weight`, with the mean style at row 0 —
 * `src/sbv2/style.ts`), and front takes it as the graph input `style_vec` (reshape → the
 * `style_proj` linear → added into the text encoder). The sha256 rows of `e2e_sbv2_wav_test.ts`
 * are all taken with the mean style at weight 1, so a style vector that stopped carrying the picked
 * row (or a graph that stopped reading it) would keep every one of them green — the audio would
 * simply sound the same for every style.
 *
 * ## How it compares
 *
 * Inside one run, against itself — no reference rows are read or written. Every noise source is
 * off (`noiseScale` 0, `noiseScaleW` 0) and the duration comes from the deterministic predictor
 * alone (`sdpRatio` 0), with the seed fixed, so the only input that differs between two
 * generations is the style vector. The three behaviours:
 *
 * 1. Another style at weight 1 gives audio different from the mean style's (the style reaches the
 *    audio at all).
 * 2. The mean style at two weights gives bit-identical audio — the mean row minus itself is zero,
 *    so the weight cancels. This also pins that the generation is deterministic, which keeps case
 *    1 from passing on run-to-run noise.
 * 3. Another style at weight 0 gives bit-identical audio to the mean style (weight 0 is the mean
 *    style — the weight is applied around row 0, not around zero or a per-column mean).
 *
 * The styles are taken from the model's own `pipelineConfig.styles` (the mean style = the name at
 * row 0, the other style = the name at the next row present), not from hard-coded row numbers.
 *
 * MUST: the assets are `models/karume-sbv2-jvnv/` (untracked, local assets of a real-GPU machine).
 * Environments without them and without a GPU **SKIP explicitly** with a reason (ADR 0005).
 * Analyzing the utterance is the caller's responsibility, so this test fetches the Japanese
 * dictionary with `@hdae/yomi` (network on the first run only — the same as `e2e_sbv2_wav_test.ts`).
 */

import { readFile } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { type FileRef, type Manifest, parseManifest, resolveSelection } from "@karume/hub";
import { analyzeWithWords } from "@hdae/yomi";
import { getDictionary } from "@hdae/yomi/loader";
import {
  type GeneratedAudio,
  type GenerateOptions,
  Pipeline,
  type Utterance,
  toUtterance,
} from "../../models/sbv2.ts";
import { parseSbv2PipelineConfig } from "../../models/src/sbv2/config.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { readTextIfPresent } from "../../models/tests/helpers/read-if-present.ts";

/** Where the assets are (`models/karume-sbv2-jvnv/` at the repository root). */
const ASSETS_DIR = new URL("../../../models/karume-sbv2-jvnv/", import.meta.url);

const MODEL = "F1";
/** The smallest distribution of the model (all three components mixed i4) — the check is cheap. */
const QUANT = "i4";
/** The same sentence as the sha check of `e2e_sbv2_wav_test.ts`. */
const TEXT = "こんにちは、これはテストです。";
const SEED = 0;
/**
 * The knobs that make the generation deterministic: no noise on z_p, no noise in the sdp reverse,
 * and the duration from the deterministic predictor only.
 */
const NOISE_OFF = {
  noiseScale: 0,
  noiseScaleW: 0,
  sdpRatio: 0,
  seed: SEED,
} as const satisfies GenerateOptions;

const manifestText = await readTextIfPresent(new URL("karume.json", ASSETS_DIR));
if (manifestText === undefined) {
  console.warn(
    `[karume] ${ASSETS_DIR.pathname} has no karume.json, so the SBV2 style check is SKIPped` +
      " (bake it with the exporter's dist.py)",
  );
}
const RUNNABLE = GPU_AVAILABLE && manifestText !== undefined;

/**
 * Reads the assets a quant requires from local files (a local version of `fetchAssets` — the same
 * form as the reader of `e2e_sbv2_wav_test.ts`). MUST: zero-length parts are listed too (the index
 * is the id inside the container).
 */
const loadLocalAssets = async (
  manifest: Manifest,
): Promise<Record<string, Uint8Array<ArrayBuffer>>> => {
  const selection = resolveSelection(manifest, { model: MODEL, quant: QUANT });
  const byPath = new Map<string, Uint8Array<ArrayBuffer>>();
  let assets: Record<string, Uint8Array<ArrayBuffer>> = {};
  const read = async (key: string, ref: FileRef): Promise<void> => {
    const cached = byPath.get(ref.path);
    const bytes = cached ?? (await readFile(new URL(ref.path, ASSETS_DIR)));
    if (cached === undefined) byPath.set(ref.path, bytes);
    assets = { ...assets, [key]: bytes };
  };
  for (const name of Object.keys(selection.containers)) {
    const { parts } = selection.containers[name];
    for (const [index, ref] of parts.entries()) await read(`${name}[${index}]`, ref);
  }
  for (const name of Object.keys(selection.assets)) await read(name, selection.assets[name]);
  return assets;
};

/**
 * The mean style (the name at row 0 — SBV2's convention) and one other style (the name at the
 * lowest other row), from the model's own style list.
 */
const pickStyles = (manifest: Manifest): { readonly mean: string; readonly other: string } => {
  if (!Object.hasOwn(manifest.models, MODEL)) {
    throw new Error(
      `the distribution has no model '${MODEL}' (present: ${manifest.available.models.join(" / ")})`,
    );
  }
  const { styles } = parseSbv2PipelineConfig(manifest.models[MODEL].pipelineConfig);
  const byRow = [...styles].sort(([, a], [, b]) => a - b);
  const [first, second] = byRow;
  if (first === undefined || first[1] !== 0 || second === undefined) {
    throw new Error(
      `${MODEL}'s styles need the mean style at row 0 and at least one other style ` +
        `(have: ${byRow.map(([name, row]) => `${name}=${row}`).join(" / ")})`,
    );
  }
  return { mean: first[0], other: second[0] };
};

/** Where two waveforms first differ, or `undefined` when they are bit-identical. */
const firstDifference = (a: GeneratedAudio, b: GeneratedAudio): string | undefined => {
  if (a.sampleRate !== b.sampleRate) return `sample rate ${a.sampleRate} vs ${b.sampleRate}`;
  if (a.data.length !== b.data.length) return `length ${a.data.length} vs ${b.data.length} samples`;
  const at = a.data.findIndex((sample, index) => !Object.is(sample, b.data[index]));
  return at < 0 ? undefined : `sample ${at}: ${a.data[at]} vs ${b.data[at]}`;
};

describe.skipIf(!RUNNABLE)(
  `e2e (real GPU): with every noise source off, the style setting of ${MODEL} / ${QUANT} reaches the audio`,
  () => {
    // Assigned in beforeAll; the its below run sequentially against this one pipeline.
    let pipeline: Pipeline;
    let utterance: Utterance;
    let styles: { readonly mean: string; readonly other: string };
    let speaker: string;
    /** The mean style at weight 1 (the baseline every case compares against). */
    let baseline: GeneratedAudio;

    const generate = (style: string, styleWeight: number): Promise<GeneratedAudio> =>
      pipeline.generate(utterance, { ...NOISE_OFF, speaker, style, styleWeight });

    beforeAll(async () => {
      const manifest = parseManifest(manifestText as string);
      styles = pickStyles(manifest);
      speaker = parseSbv2PipelineConfig(manifest.models[MODEL].pipelineConfig).defaults.speaker;
      const assets = await loadLocalAssets(manifest);
      pipeline = await Pipeline.fromAssets({ manifest, assets }, { model: MODEL, quant: QUANT });
      const dictionary = await getDictionary();
      utterance = toUtterance(analyzeWithWords(dictionary, TEXT));
      baseline = await generate(styles.mean, 1);
      console.log(
        `[e2e] sbv2 style ${MODEL}/${QUANT}: mean style '${styles.mean}', other style ` +
          `'${styles.other}', speaker '${speaker}', baseline ${baseline.data.length} samples`,
      );
    });
    afterAll(async () => {
      if (pipeline !== undefined) await pipeline.dispose();
    });

    it("another style at weight 1 gives audio different from the mean style", async () => {
      const styled = await generate(styles.other, 1);
      const difference = firstDifference(baseline, styled);
      console.log(
        `[e2e] sbv2 style '${styles.other}' x1 vs '${styles.mean}' x1: ${difference ?? "identical"}`,
      );
      assert(
        difference !== undefined,
        `'${styles.other}' at weight 1 gave bit-identical audio to '${styles.mean}' ` +
          "(the style vector does not reach the audio)",
      );
    });

    it("the mean style gives bit-identical audio at weights 1 and 5 (the weight cancels)", async () => {
      const weighted = await generate(styles.mean, 5);
      assertEquals(
        firstDifference(baseline, weighted),
        undefined,
        `'${styles.mean}' at weight 5 differs from weight 1`,
      );
    });

    it("another style at weight 0 gives bit-identical audio to the mean style (weight 0 is the mean style)", async () => {
      const muted = await generate(styles.other, 0);
      assertEquals(
        firstDifference(baseline, muted),
        undefined,
        `'${styles.other}' at weight 0 differs from '${styles.mean}'`,
      );
    });
  },
);
