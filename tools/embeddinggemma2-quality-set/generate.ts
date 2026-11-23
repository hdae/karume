/**
 * The generator of the quality set of EmbeddingGemma 2: images with captions and speech with
 * transcripts, made by Karume's own pipelines on Node + Dawn from the local mirrors under
 * `models/`, for checking quantized image and audio towers against their reference preset. What the
 * set holds is `set.ts`; this file runs the three pipelines and writes the files.
 *
 *     flock /tmp/karume-gpu.lock node tools/embeddinggemma2-quality-set/generate.ts [<dir>] [--limit <n>]
 *
 * The default directory is `inputs/embeddinggemma2/quality-set/` (outside git). It writes
 * `images/<generator>-<subject>.png` (8-bit RGB), `audio/<speaker>-<nn>.wav` (16-bit PCM mono at
 * SBV2's own rate — nothing is resampled) and `manifest.json` last; an earlier `manifest.json` is
 * removed first, so a run that fails leaves none. Every mirror the run needs is opened before the
 * GPU is touched, so a missing one fails at once. The reading dictionary of SBV2's text analysis is
 * fetched by `@hdae/yomi` at the revision its version pins (Node has no Cache Storage, so every run
 * fetches it).
 */

import { mkdir, rm, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { analyzeWithWords } from "@hdae/yomi";
import { getDictionary } from "@hdae/yomi/loader";
import { parseManifest } from "@karume/hub";
import { type OpenedMountDirectory, openMountDirectory } from "@karume/local-distributions/node";
import { encodePng, encodeWav } from "@karume/models";
import * as anima from "@karume/models/anima";
import * as moss from "@karume/models/moss";
import * as sbv2 from "@karume/models/sbv2";
import { installDawn } from "@karume/test-support/dawn";

import {
  type AudioItem,
  IMAGE_GENERATORS,
  type ImageGeneratorName,
  type ImageItem,
  MANIFEST_FORMAT,
  type PlannedClip,
  type PlannedImage,
  parseCommandLine,
  planSet,
  type QualitySetManifest,
  SPEAKERS,
  SPEECH_SEED,
  UsageError,
} from "./set.ts";

const REPOSITORY_ROOT = fileURLToPath(new URL("../../", import.meta.url));
const DEFAULT_OUTPUT = join(REPOSITORY_ROOT, "inputs/embeddinggemma2/quality-set");
const SBV2_MOUNT = "sbv2";

const log = (text: string): void => {
  process.stdout.write(`${text}\n`);
};

const seconds = (since: number): string => `${((performance.now() - since) / 1000).toFixed(1)} s`;

const writeOutput = async (root: string, file: string, bytes: Uint8Array): Promise<void> => {
  const path = join(root, file);
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, bytes);
};

/** The preset a pipeline built without `quant` takes: the model's `defaultQuant`. */
const defaultQuantOf = (opened: OpenedMountDirectory, model: string): string => {
  const manifest = parseManifest(new TextDecoder().decode(opened.manifestBytes));
  if (!Object.hasOwn(manifest.models, model)) {
    throw new Error(
      `${opened.directory} has no model ${model} (it has ${Object.keys(manifest.models).join(", ")})`,
    );
  }
  return manifest.models[model].defaultQuant;
};

/** One image generator's planned images: build, generate each, dispose. */
const generateImages = async (
  root: string,
  name: ImageGeneratorName,
  opened: OpenedMountDirectory,
  planned: readonly PlannedImage[],
): Promise<ImageItem[]> => {
  if (planned.length === 0) return [];
  const { model } = IMAGE_GENERATORS[name];
  log(`${opened.directory} / ${model} / ${defaultQuantOf(opened, model)}: building`);
  const built = performance.now();
  const written: ImageItem[] = [];
  if (name === "anima") {
    await using pipeline = await anima.Pipeline.fromPretrained(opened.source, { model });
    log(`  built in ${seconds(built)}`);
    for (const { subject, item } of planned) {
      const started = performance.now();
      const image = await pipeline.generate({
        prompt: item.prompt,
        resolution: subject.shape,
        seed: item.seed,
      });
      await writeOutput(root, item.file, await encodePng(image.data, image.width, image.height));
      log(`  ${item.file} ${image.width}×${image.height}: ${seconds(started)}`);
      written.push(item);
    }
  } else {
    await using pipeline = await moss.Pipeline.fromPretrained(opened.source, { model });
    log(`  built in ${seconds(built)}`);
    for (const { subject, item } of planned) {
      const started = performance.now();
      const image = await pipeline.generate({
        prompt: item.prompt,
        width: subject.shape.width,
        height: subject.shape.height,
        seed: item.seed,
      });
      const rgba = moss.imageToRgba(image);
      await writeOutput(root, item.file, await encodePng(rgba, image.width, image.height));
      log(`  ${item.file} ${image.width}×${image.height}: ${seconds(started)}`);
      written.push(item);
    }
  }
  return written;
};

/** Every planned clip, one SBV2 pipeline per speaker (each speaker is a model of the mirror). */
const generateAudio = async (
  root: string,
  opened: OpenedMountDirectory,
  planned: readonly PlannedClip[],
): Promise<AudioItem[]> => {
  if (planned.length === 0) return [];
  const dictionaryLoaded = performance.now();
  const dictionary = await getDictionary();
  log(`reading dictionary of @hdae/yomi: ${seconds(dictionaryLoaded)}`);
  const written: AudioItem[] = [];
  for (const speaker of SPEAKERS) {
    const clips = planned.filter((clip) => clip.speaker === speaker);
    if (clips.length === 0) continue;
    log(`${opened.directory} / ${speaker} / ${defaultQuantOf(opened, speaker)}: building`);
    const built = performance.now();
    await using pipeline = await sbv2.Pipeline.fromPretrained(opened.source, { model: speaker });
    log(`  built in ${seconds(built)}`);
    for (const { item } of clips) {
      const started = performance.now();
      const utterance = sbv2.toUtterance(analyzeWithWords(dictionary, item.transcript));
      const audio = await pipeline.generate(utterance, { seed: SPEECH_SEED });
      await writeOutput(root, item.file, encodeWav(audio.data, audio.sampleRate));
      const spoken = (audio.data.length / audio.sampleRate).toFixed(2);
      log(`  ${item.file} ${spoken} s at ${audio.sampleRate} Hz: ${seconds(started)}`);
      written.push({ ...item, sampleRate: audio.sampleRate });
    }
  }
  return written;
};

const run = async (args: readonly string[]): Promise<void> => {
  const line = parseCommandLine(args);
  const root = line.output === undefined ? DEFAULT_OUTPUT : resolve(line.output);
  const plan = planSet(line.limit);
  // Every mirror first: a missing one fails before any GPU work.
  const mirrors = {
    anima: await openMountDirectory(REPOSITORY_ROOT, IMAGE_GENERATORS.anima.mount),
    moss: await openMountDirectory(REPOSITORY_ROOT, IMAGE_GENERATORS.moss.mount),
    sbv2: await openMountDirectory(REPOSITORY_ROOT, SBV2_MOUNT),
  } as const;
  log(
    `${plan.images.length} images and ${plan.audio.length} clips into ${root}` +
      (line.limit === undefined ? "" : ` (--limit ${line.limit} per generator)`),
  );
  const manifestPath = join(root, "manifest.json");
  await rm(manifestPath, { force: true });
  installDawn();
  const started = performance.now();
  const images = [
    ...(await generateImages(
      root,
      "anima",
      mirrors.anima,
      plan.images.filter((image) => image.generator === "anima"),
    )),
    ...(await generateImages(
      root,
      "moss",
      mirrors.moss,
      plan.images.filter((image) => image.generator === "moss"),
    )),
  ];
  const audio = await generateAudio(root, mirrors.sbv2, plan.audio);
  const manifest: QualitySetManifest = { format: MANIFEST_FORMAT, images, audio };
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  log(
    `wrote ${manifestPath} (${images.length} images, ${audio.length} clips) in ${seconds(started)}`,
  );
};

/** Runs the command and returns its exit code (2 for a command line it cannot act on). */
export const main = async (args: readonly string[]): Promise<number> => {
  try {
    await run(args);
    return 0;
  } catch (error) {
    if (error instanceof UsageError) {
      process.stderr.write(`${error.message}\n`);
      return 2;
    }
    process.stderr.write(
      `${error instanceof Error ? (error.stack ?? error.message) : String(error)}\n`,
    );
    return 1;
  }
};

if (import.meta.main) {
  process.exitCode = await main(process.argv.slice(2));
}
