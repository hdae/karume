/**
 * A single Anima weights file run through `anima.Pipeline.fromSingleFile` on a GPU: the file is a
 * path under `inputs/` (read from disk under Node, by ranged requests to the dev server in a
 * browser), the distribution the `anima` mount. The pipeline is built and makes two images of one
 * small request (the second shows what a generation costs once the first has run: in the default
 * per-stage residency every generate builds its Sessions again, so the file is converted again,
 * unless `retainConverted` keeps the converted weights in memory);
 * optionally `fromPretrained` of the same model and quant does the same, and the record says
 * whether the images are equal. It only measures and compares: no verdict.
 *
 * Which components came from the file is read from the file's placement by the distribution's name
 * map: the transformer always (`fromSingleFile` refuses a file without one), the text conditioner
 * when the file holds one and the run takes it from the file.
 *
 * SPEC: docs/spec/workbench.md#the-entries
 */

import { parseNameMap, placeTensors } from "@karume/convert";
import { fetchAssets, type LoadedManifest, loadManifest, resolveSelection } from "@karume/hub";
import { Pipeline } from "@karume/models/anima";
import { acquireGpu, type GpuContext } from "@karume/runtime";
import { openSafetensors } from "@karume/safetensors";

import {
  animaSingleFileData,
  animaSingleFileSummary,
  type AnimaSingleFileRequest,
  type ArmRecord,
  TEXT_CONDITIONER_SOURCES,
} from "../../anima-single-file/record.ts";
import {
  booleanValue,
  choiceValue,
  type Entry,
  integerValue,
  type RangedFile,
  textValue,
} from "../../entry.ts";
import { SOURCE_KINDS, sourceParameter } from "../../local-parameters.ts";
import { sha256Hex } from "../../sha256.ts";

/** The request of the existing Anima e2e for the base model, at fewer steps. */
const PROMPT =
  "1girl, solo, long hair, blue eyes, school uniform, cherry blossoms, outdoors, " +
  "smile, upper body, masterpiece, best quality";
const RESOLUTION = { width: 512, height: 512 } as const;

/** The file's placement by the distribution's name map: its layout and the keys per component. */
const placementOf = async (
  loaded: LoadedManifest,
  file: RangedFile,
  signal: AbortSignal,
): Promise<{ readonly layout: string; readonly placed: Readonly<Record<string, number>> }> => {
  const ref = loaded.manifest.nameMap;
  if (ref === undefined) throw new Error("the anima mount's manifest has no name map");
  const { nameMap } = await fetchAssets(loaded, { nameMap: ref }, { signal });
  const map = parseNameMap(JSON.parse(new TextDecoder().decode(nameMap)));
  const placement = placeTensors(map, (await openSafetensors(file)).tensors.keys());
  return {
    layout: placement.layout,
    placed: Object.fromEntries([...placement.components].map(([role, keys]) => [role, keys.size])),
  };
};

/** The RGB8 pixels the Anima e2e hashes (`data` is RGBA with alpha always 255). */
const rgbPixels = (data: Uint8Array<ArrayBuffer>): Uint8Array<ArrayBuffer> => {
  const pixels = new Uint8Array(new ArrayBuffer((data.length / 4) * 3));
  for (let source = 0, target = 0; source < data.length; source += 4, target += 3) {
    pixels[target] = data[source];
    pixels[target + 1] = data[source + 1];
    pixels[target + 2] = data[source + 2];
  }
  return pixels;
};

/** Builds one arm's pipeline, makes two images and disposes it. */
const measureArm = async (
  name: ArmRecord["name"],
  build: () => Promise<Pipeline>,
  request: AnimaSingleFileRequest,
  signal: AbortSignal,
  progress: (text: string) => void,
): Promise<ArmRecord> => {
  progress(`${name}: building the pipeline`);
  const started = performance.now();
  const pipeline = await build();
  const buildSeconds = (performance.now() - started) / 1000;
  try {
    const generations = [];
    for (const index of [1, 2]) {
      signal.throwIfAborted();
      progress(`${name}: generation ${index} of 2`);
      const begun = performance.now();
      const image = await pipeline.generate({
        prompt: request.prompt,
        resolution: request.resolution,
        steps: request.steps,
        seed: request.seed,
      });
      generations.push({
        seconds: (performance.now() - begun) / 1000,
        pixelSha256: await sha256Hex(rgbPixels(image.data)),
      });
    }
    return { name, buildSeconds, generations };
  } finally {
    await pipeline.dispose();
  }
};

const entry: Entry = {
  id: "anima-single-file",
  title: "Anima from a single file",
  description:
    "An Anima single file (a path under inputs/) through fromSingleFile on the anima mount: the build time and two small generations, optionally against fromPretrained of the same model and quant, with the pixel sha256 of every image and whether they are equal.",
  parameters: [
    {
      kind: "text",
      name: "file",
      label: "The file, relative to inputs/",
      default:
        "anima/upstream-single-file/split_files/diffusion_models/anima-base-v1.0.safetensors",
    },
    { kind: "text", name: "model", label: "Model of the anima mount", default: "anima-v1.0" },
    {
      kind: "text",
      name: "quant",
      label: "Quant (empty: the model's default)",
      default: "",
    },
    {
      kind: "choice",
      name: "textConditioner",
      label: "Text conditioner: from the file (when it holds one) or the distribution's",
      options: TEXT_CONDITIONER_SOURCES,
      default: "file",
    },
    {
      kind: "boolean",
      name: "retainConverted",
      label: "fromSingleFile keeps the converted weights in memory (retainConverted)",
      default: false,
    },
    {
      kind: "boolean",
      name: "compare",
      label: "Also run fromPretrained of the same model and quant",
      default: true,
    },
    { kind: "integer", name: "steps", label: "Denoise steps (512²)", min: 2, max: 50, default: 4 },
    { kind: "integer", name: "seed", label: "Seed", min: 0, max: 2 ** 31 - 1, default: 42 },
    sourceParameter("How a browser reads the distribution"),
  ],
  run: async ({ parameters, gpuOptions, signal, progress, openDistribution, openInput }) => {
    const textConditioner = choiceValue(parameters, "textConditioner", TEXT_CONDITIONER_SOURCES);
    const retainConverted = booleanValue(parameters, "retainConverted");
    const compare = booleanValue(parameters, "compare");
    const opened = await openDistribution("anima", choiceValue(parameters, "source", SOURCE_KINDS));
    const input = await openInput(textValue(parameters, "file"));
    const loaded = await loadManifest(opened.ref, { signal });
    const model = textValue(parameters, "model");
    const quant = textValue(parameters, "quant");
    const selection = resolveSelection(loaded.manifest, {
      model,
      ...(quant === "" ? {} : { quant }),
    });
    progress("reading the file's header and placing its tensors");
    const placement = await placementOf(loaded, input.file, signal);
    const fromFile = [
      "transformer",
      ...(textConditioner === "file" && (placement.placed["text_conditioner"] ?? 0) > 0
        ? ["text_conditioner"]
        : []),
    ];
    const request: AnimaSingleFileRequest = {
      model,
      quant: selection.quant,
      textConditioner,
      retainConverted,
      compare,
      prompt: PROMPT,
      resolution: RESOLUTION,
      steps: integerValue(parameters, "steps"),
      seed: integerValue(parameters, "seed"),
    };
    const gpu: GpuContext = await acquireGpu({ ...gpuOptions });
    try {
      const common = { gpu, model, quant: selection.quant, signal };
      const arms = [
        await measureArm(
          "fromSingleFile",
          () =>
            Pipeline.fromSingleFile(input.file, opened.ref, {
              ...common,
              retainConverted,
              ...(textConditioner === "distribution"
                ? { components: { text_conditioner: { source: opened.ref } } }
                : {}),
            }),
          request,
          signal,
          progress,
        ),
      ];
      if (compare) {
        arms.push(
          await measureArm(
            "fromPretrained",
            () => Pipeline.fromPretrained(opened.ref, common),
            request,
            signal,
            progress,
          ),
        );
      }
      const data = animaSingleFileData({
        request,
        file: { ...input.facts, ...placement },
        distribution: opened.facts,
        fromFile,
        arms,
      });
      return { summary: animaSingleFileSummary(data), data };
    } finally {
      gpu.destroy();
    }
  },
};

export default entry;
