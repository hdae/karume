/**
 * The quality set of EmbeddingGemma 2 through the real pipeline, in the presets `bf16` and `i8`:
 * how far the `i8` preset (the text tower and both towers as per-row int8) is from `bf16` on the
 * GPU, item by item, and what each preset takes per item.
 *
 *     flock /tmp/karume-gpu.lock node tools/embeddinggemma2-quality-set/compare-presets.ts [<dir>]
 *
 * It opens the local mirror (`models/karume-embeddinggemma-2`) and builds `embeddinggemma.Pipeline`
 * twice with `vision: true` and `audio: true`, once per preset, both resident. It embeds every
 * image of `inputs/embeddinggemma2/quality-set/` and every clip of
 * `inputs/embeddinggemma2/quality-set-16k/` (the 16 kHz, 16-bit WAVs the tower screening writes —
 * `measure_quant_towers.py --write-clips` — so the pipeline reads the screening's own samples),
 * each alone and without a prompt, and the distinct captions and transcripts with the `query`
 * prompt (the screening's convention: the query prefix on the text side, none on images and audio).
 *
 * Every input runs twice per preset. The first pass is the warm-up and gives the vectors; the
 * second is timed and must give the same bits. In the timed pass the two presets alternate which
 * runs first, input by input, so neither always meets a GPU the other has just warmed. A time is
 * the whole `embed` call's wall time (the pipeline does not report its tower runs apart).
 *
 * Printed per modality: the cosine of each item between the presets (min / mean / worst item) at
 * the widths 768 and 128, the same for its texts, the cross-modal top-1 of each preset and how many
 * choices the presets share, then the mean and median time per item. The record goes to
 * `<dir>/record.json`, by default `outputs/bench/embeddinggemma-2/<date>_tower-i8-gpu/` (outside
 * git).
 */

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { openMountDirectory } from "@karume/local-distributions/node";
import { decodeWav } from "@karume/models";
import * as embeddinggemma from "@karume/models/embeddinggemma";
import { installDawn } from "@karume/test-support/dawn";
import { decodePng } from "@karume/test-support/png-decode";

import {
  agreement,
  CLIP_SAMPLE_RATE,
  CompareInputError,
  type CosineSummary,
  cosine,
  distinctTexts,
  parseCompareCommandLine,
  parseQualitySetManifest,
  retrieval,
  summarizeCosines,
  summarizeTimes,
  type TimeSummary,
  WIDTHS,
} from "./compare.ts";

const REPOSITORY_ROOT = fileURLToPath(new URL("../../", import.meta.url));
const IMAGE_SET = join(REPOSITORY_ROOT, "inputs/embeddinggemma2/quality-set");
const CLIP_SET = join(REPOSITORY_ROOT, "inputs/embeddinggemma2/quality-set-16k");
const MOUNT = "embeddinggemma-2";
const MODEL = "text";
const PRESETS = ["bf16", "i8"] as const;
type Preset = (typeof PRESETS)[number];
const TEXT_PROMPT = "query";
const RECORD_FORMAT = "karume-embeddinggemma2-tower-i8-gpu/1";

const log = (text: string): void => {
  process.stdout.write(`${text}\n`);
};

/** One input of a run: what it is, and the call that embeds it. */
type Input = {
  readonly name: string;
  readonly kind: "image" | "audio" | "caption" | "transcript";
  readonly embed: (pipeline: embeddinggemma.Pipeline) => Promise<Float32Array<ArrayBuffer>>;
  readonly tokens: (pipeline: embeddinggemma.Pipeline) => number;
};

const imageInput = (name: string, image: embeddinggemma.Rgb8Image): Input => {
  const parts = [{ type: "image", image }] as const;
  return {
    name,
    kind: "image",
    embed: (pipeline) => pipeline.embed(parts),
    tokens: (pipeline) => pipeline.tokenize(parts).length,
  };
};

const audioInput = (name: string, audio: embeddinggemma.MonoAudio): Input => {
  const parts = [{ type: "audio", audio }] as const;
  return {
    name,
    kind: "audio",
    embed: (pipeline) => pipeline.embed(parts),
    tokens: (pipeline) => pipeline.tokenize(parts).length,
  };
};

const textInput = (text: string, kind: "caption" | "transcript"): Input => ({
  name: text,
  kind,
  embed: (pipeline) => pipeline.embed(text, { prompt: TEXT_PROMPT }),
  tokens: (pipeline) => pipeline.tokenize(text, { prompt: TEXT_PROMPT }).length,
});

/** Reads the set's manifest from `directory` (fails loudly on a missing or malformed one). */
const readManifest = async (directory: string) => {
  const path = join(directory, "manifest.json");
  return parseQualitySetManifest(await readFile(path, "utf8"), path);
};

/** The vectors and the per-pass times of one input in one preset. */
type Measured = { readonly vector: Float32Array<ArrayBuffer>; readonly ms: number[] };

const elapsed = async <T>(run: () => Promise<T>): Promise<{ value: T; ms: number }> => {
  const start = performance.now();
  const value = await run();
  return { value, ms: performance.now() - start };
};

const sameBits = (left: Float32Array<ArrayBuffer>, right: Float32Array<ArrayBuffer>): boolean =>
  left.length === right.length && left.every((value, index) => Object.is(value, right[index]));

const fixed = (value: number, digits: number): string => value.toFixed(digits);

const cosineCells = (summary: CosineSummary): string =>
  `${fixed(summary.min, 6)} | ${fixed(summary.mean, 6)} | ${summary.worst}`;

const timeCells = (summary: TimeSummary): string =>
  `${fixed(summary.mean, 1)} | ${fixed(summary.median, 1)} | ${fixed(summary.min, 1)} | ${fixed(summary.max, 1)}`;

const run = async (args: readonly string[]): Promise<void> => {
  const date = new Date().toLocaleDateString("sv-SE");
  const requested = parseCompareCommandLine(args);
  const out =
    requested === undefined
      ? join(REPOSITORY_ROOT, `outputs/bench/embeddinggemma-2/${date}_tower-i8-gpu`)
      : resolve(requested);
  // Every input first: a missing file or a clip at another rate fails before any GPU work.
  const imageSet = await readManifest(IMAGE_SET);
  const clipSet = await readManifest(CLIP_SET);
  const opened = await openMountDirectory(REPOSITORY_ROOT, MOUNT);
  const images = await Promise.all(
    imageSet.images.map(async (item) => {
      const path = join(IMAGE_SET, item.file);
      return imageInput(item.file, await decodePng(new Uint8Array(await readFile(path)), path));
    }),
  );
  const clips = await Promise.all(
    clipSet.audio.map(async (item) => {
      const audio = decodeWav(new Uint8Array(await readFile(join(CLIP_SET, item.file))));
      if (item.sampleRate !== CLIP_SAMPLE_RATE || audio.sampleRate !== CLIP_SAMPLE_RATE) {
        throw new CompareInputError(
          `${join(CLIP_SET, item.file)}: ${audio.sampleRate} Hz (manifest ${item.sampleRate} Hz), not` +
            ` ${CLIP_SAMPLE_RATE} — write the clips with measure_quant_towers.py --write-clips`,
        );
      }
      return audioInput(item.file, audio);
    }),
  );
  const captions = distinctTexts(imageSet.images.map(({ caption }) => caption));
  const transcripts = distinctTexts(clipSet.audio.map(({ transcript }) => transcript));
  const inputs: readonly Input[] = [
    ...images,
    ...clips,
    ...captions.texts.map((text) => textInput(text, "caption")),
    ...transcripts.texts.map((text) => textInput(text, "transcript")),
  ];
  log(
    `${images.length} images, ${clips.length} clips, ${captions.texts.length} captions and` +
      ` ${transcripts.texts.length} transcripts from ${opened.directory}, presets ${PRESETS.join(" / ")}`,
  );

  installDawn();
  const adapter = await navigator.gpu.requestAdapter();
  if (adapter === null) throw new Error("no WebGPU adapter");
  const { vendor, architecture, device, description } = adapter.info;
  log(`adapter: ${vendor} / ${architecture} / ${device} / ${description}`);

  const pipelines = new Map<Preset, embeddinggemma.Pipeline>();
  const builds: Record<string, number> = {};
  try {
    for (const quant of PRESETS) {
      const { value, ms } = await elapsed(() =>
        embeddinggemma.Pipeline.fromPretrained(opened.source, {
          model: MODEL,
          quant,
          vision: true,
          audio: true,
        }),
      );
      pipelines.set(quant, value);
      builds[quant] = ms;
      log(`${quant}: built in ${fixed(ms / 1000, 2)} s`);
    }
    const pipelineOf = (quant: Preset): embeddinggemma.Pipeline => {
      const pipeline = pipelines.get(quant);
      if (pipeline === undefined) throw new Error(`no pipeline of ${quant}`);
      return pipeline;
    };

    const measured = new Map<Preset, Measured[]>(PRESETS.map((quant) => [quant, []]));
    const warmStart = performance.now();
    for (const input of inputs) {
      for (const quant of PRESETS) {
        const { value, ms } = await elapsed(() => input.embed(pipelineOf(quant)));
        measured.get(quant)?.push({ vector: value, ms: [ms] });
      }
    }
    log(`warm-up pass: ${fixed((performance.now() - warmStart) / 1000, 1)} s`);
    const timedStart = performance.now();
    for (const [index, input] of inputs.entries()) {
      const order = index % 2 === 0 ? PRESETS : [...PRESETS].reverse();
      for (const quant of order) {
        const entry = measured.get(quant)?.[index];
        if (entry === undefined) throw new Error(`no warm-up of ${input.name} in ${quant}`);
        const { value, ms } = await elapsed(() => input.embed(pipelineOf(quant)));
        if (!sameBits(value, entry.vector)) {
          throw new Error(`${quant} ${input.name}: the timed run differs from the warm-up's bits`);
        }
        entry.ms.push(ms);
      }
    }
    log(`timed pass: ${fixed((performance.now() - timedStart) / 1000, 1)} s`);

    const at = (quant: Preset, index: number): Measured => {
      const entry = measured.get(quant)?.[index];
      if (entry === undefined) throw new Error(`no measurement ${index} of ${quant}`);
      return entry;
    };
    const indicesOf = (kind: Input["kind"]): number[] =>
      inputs.flatMap((input, index) => (input.kind === kind ? [index] : []));
    const vectorsOf = (quant: Preset, indices: readonly number[]) =>
      indices.map((index) => at(quant, index).vector);

    const modalities = [
      { name: "image", items: indicesOf("image"), texts: indicesOf("caption"), map: captions },
      {
        name: "audio",
        items: indicesOf("audio"),
        texts: indicesOf("transcript"),
        map: transcripts,
      },
    ] as const;
    const summary: Record<string, unknown> = {};
    for (const modality of modalities) {
      const widths: Record<string, unknown> = {};
      log("");
      log(
        `${modality.name}: ${modality.items.length} items, ${modality.texts.length} distinct texts` +
          ` (prompt ${TEXT_PROMPT}) — i8 against bf16`,
      );
      log(
        "| width | item cosine min | mean | worst item | text cosine min | mean | worst text |" +
          " item→text top-1 bf16 / i8 / shared | text→item top-1 bf16 / i8 / shared |",
      );
      log("| --- | --- | --- | --- | --- | --- | --- | --- | --- |");
      for (const width of WIDTHS) {
        const cosinesOf = (indices: readonly number[]) =>
          indices.map((index) => ({
            name: inputs[index].name,
            value: cosine(at("i8", index).vector, at("bf16", index).vector, width),
          }));
        const itemCosines = cosinesOf(modality.items);
        const textCosines = cosinesOf(modality.texts);
        const [reference, quantized] = PRESETS.map((quant) =>
          retrieval(
            vectorsOf(quant, modality.items),
            vectorsOf(quant, modality.texts),
            modality.map.itemText,
            width,
          ),
        );
        const forwardShared = agreement(reference.itemToText.choices, quantized.itemToText.choices);
        const backwardShared = agreement(
          reference.textToItem.choices,
          quantized.textToItem.choices,
        );
        const items = summarizeCosines(itemCosines);
        const texts = summarizeCosines(textCosines);
        log(
          `| ${width} | ${cosineCells(items)} | ${cosineCells(texts)} |` +
            ` ${reference.itemToText.hits} / ${quantized.itemToText.hits} / ${forwardShared} of ${modality.items.length} |` +
            ` ${reference.textToItem.hits} / ${quantized.textToItem.hits} / ${backwardShared} of ${modality.texts.length} |`,
        );
        widths[String(width)] = {
          items,
          texts,
          itemCosines: Object.fromEntries(itemCosines.map(({ name, value }) => [name, value])),
          textCosines: Object.fromEntries(textCosines.map(({ name, value }) => [name, value])),
          itemToText: {
            bf16: reference.itemToText,
            i8: quantized.itemToText,
            shared: forwardShared,
          },
          textToItem: {
            bf16: reference.textToItem,
            i8: quantized.textToItem,
            shared: backwardShared,
          },
        };
      }
      summary[modality.name] = { widths };
    }

    log("");
    log("time per item, the timed pass (ms, the whole embed call)");
    log("| inputs | preset | count | mean | median | min | max |");
    log("| --- | --- | --- | --- | --- | --- | --- |");
    const times: Record<string, Record<string, TimeSummary>> = {};
    for (const kind of ["image", "audio", "caption", "transcript"] as const) {
      const indices = indicesOf(kind);
      times[kind] = {};
      for (const quant of PRESETS) {
        const timed = summarizeTimes(indices.map((index) => at(quant, index).ms[1]));
        times[kind][quant] = timed;
        log(`| ${kind} | ${quant} | ${indices.length} | ${timeCells(timed)} |`);
      }
    }

    const record = {
      format: RECORD_FORMAT,
      date,
      node: process.version,
      adapter: { vendor, architecture, device, description },
      mirror: opened.directory,
      model: MODEL,
      presets: PRESETS,
      buildMs: builds,
      imageSet: IMAGE_SET,
      clipSet: CLIP_SET,
      textPrompt: TEXT_PROMPT,
      widths: WIDTHS,
      summary,
      times,
      inputs: inputs.map((input, index) => ({
        name: input.name,
        kind: input.kind,
        tokens: input.tokens(pipelineOf("bf16")),
        ms: Object.fromEntries(PRESETS.map((quant) => [quant, at(quant, index).ms])),
      })),
    };
    await mkdir(out, { recursive: true });
    const path = join(out, "record.json");
    await writeFile(path, `${JSON.stringify(record, null, 2)}\n`);
    log("");
    log(`wrote ${path}`);
  } finally {
    for (const pipeline of pipelines.values()) await pipeline.dispose();
  }
};

/** Runs the command and returns its exit code (2 for a command line or input it cannot act on). */
export const main = async (args: readonly string[]): Promise<number> => {
  try {
    await run(args);
    return 0;
  } catch (error) {
    if (error instanceof CompareInputError) {
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
