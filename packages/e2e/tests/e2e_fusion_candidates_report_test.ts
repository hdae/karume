/**
 * **The opt-in fusion candidate report**: with `KARUME_FUSION_CANDIDATES=<dir>`, the fusion candidate
 * tables (`./helpers/fusion-candidates.ts`) of every graph of every local distribution mirror under
 * `models/` are written to `<dir>` — one Markdown table and one JSONL file per graph, and an index.
 * Without the variable the case SKIPs with the reason. No GPU; only each container's part 0 is read.
 *
 *     KARUME_FUSION_CANDIDATES=outputs/diag/<date>_fusion-candidates vp test packages/e2e/tests/e2e_fusion_candidates_report_test.ts
 *
 * Every table comes from the stable search (`stableFusionCandidates`): the window limit starts at the
 * default (12) and doubles until the rows stop changing, up to the ceiling. The index shows the
 * limit each table stopped at and whether its search stabilised; a table that did not says so at
 * its head, and its rows keep their limit marks.
 *
 * MUST: it writes only under `<dir>` (a directory under `outputs/`, outside git —
 * docs/assets-layout.md). It asserts nothing about the models: the hit-count and candidate checks live in the
 * family tests (`e2e_<family>_fusion_counts_test.ts`). It fails loudly on a graph whose symbols the
 * binding table below does not bind (or binds names the graph does not have), so a new mirror or a
 * renamed symbol is noticed instead of searched at made-up shapes.
 *
 * Each graph is planned with the Session defaults (`{}` — no preset knobs) for the default model
 * and the first storage its manifest lists. A component held as a cross-repository reference is
 * listed in the index and searched under the mirror that holds it.
 */

import { mkdir, readDirSync, readTextFile, writeTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { evalDim, parseDim } from "../../runtime/src/format/dims.ts";
import type { IrDim, IrGraph } from "../../runtime/src/format/ir.ts";
import {
  type ContainerManifest,
  containerPart0,
} from "../../runtime/tests/helpers/container-graph.ts";
import {
  type CandidateScenario,
  candidatesJsonl,
  candidatesMarkdown,
  DEFAULT_MAX_WINDOW,
  MAX_WINDOW_CEILING,
  stableFusionCandidates,
} from "./helpers/fusion-candidates.ts";
import { exists, planDistributionFusions, readDistributionGraph } from "./helpers/fusion-counts.ts";

const OUTPUT = process.env.KARUME_FUSION_CANDIDATES;
if (OUTPUT === undefined) {
  console.warn(
    "[karume] the fusion candidate report SKIPs: set KARUME_FUSION_CANDIDATES=<dir> (a directory" +
      " under outputs/, e.g. outputs/diag/<date>_fusion-candidates) to write the tables of every local mirror",
  );
}

const MODELS_ROOT = new URL("../../../models/", import.meta.url);

/** Self-contained reads only: a cross-repository reference is listed, not followed. */
const NO_MIRRORS: ReadonlyMap<string, URL> = new Map();

type Binding = Readonly<Record<string, number>>;

/** The decode binding of the Gemma 4 graphs: one row (M = 1), a 640-slot cache (C). */
const GEMMA_DECODE: Readonly<Record<string, Binding>> = {
  model: { C: 640, M: 1, R: 1 },
  drafter: { C: 640 },
};

/** The anima DiT at 1024² (S = 64 × 64 patch tokens), 64 prompt tokens, the 512 target tokens. */
const ANIMA: Readonly<Record<string, Binding>> = {
  text_encoder: { T: 64 },
  text_conditioner: { Tsrc: 64, Ttgt: 512 },
  transformer: { S: 4096 },
};

/**
 * The symbol binding of each graph searched: mirror directory → component → symbol → value (the
 * scenario printed at the head of each table). A graph without symbols needs no entry.
 */
const BINDINGS: Readonly<Record<string, Readonly<Record<string, Binding>>>> = {
  "karume-anima": ANIMA,
  "karume-anima-extra": ANIMA,
  "karume-gemma4": GEMMA_DECODE,
  "karume-gemma4-qat": GEMMA_DECODE,
  "karume-gemma4-qat-mobile": GEMMA_DECODE,
  "karume-irodori-v4-small": {
    backbone: { T: 256 },
    text_proj: { T: 256 },
    caption_proj: { T: 512 },
    speaker: { S: 750 },
    duration: { T: 256 },
    dit: { S: 750 },
    codec_decoder: { S: 750 },
    codec_encoder: { T: 750 },
  },
  "karume-irodori-v4.1-small": { duration: { T: 256 } },
  // 1024²: 4,096 image tokens + 128 text tokens for the DiT pair; the VAE latent 128 × 128.
  "karume-moss-image": {
    text_encoder: { T: 128 },
    dit_text: { T: 128 },
    dit: { S: 4096 + 128 },
    dit_skip22: { S: 4096 + 128 },
    vae_decoder_pre: { h: 128, w: 128 },
    vae_decoder_attn: { S: 128 * 128 },
    vae_decoder_post: { h: 128, w: 128 },
  },
  "karume-sbv2-jvnv": { text_encoder: { T: 64 }, front: { P: 64 }, voice: { T: 256 } },
  "karume-umt5-xxl": { text_encoder: { L: 512 } },
  "karume-wan2.1": { transformer: { S: 14040 } },
  // ti2v at 1280 × 704: the latent grid 80 × 44.
  "karume-wan2.2": {
    transformer: { S: 7920 },
    vae_encoder_pre: { h: 44, w: 80 },
    vae_encoder_attn: { S: 44 * 80 },
    vae_encoder_post: { h: 44, w: 80 },
  },
};

/** A shape with its symbols bound (an unbound symbol fails loudly — `evalDim`). */
const boundShape = (shape: readonly IrDim[], binding: Binding, where: string): number[] =>
  shape.map((dim) => {
    if (typeof dim === "number") return dim;
    try {
      return evalDim(parseDim(dim), binding);
    } catch (cause) {
      throw new Error(`${where}: dimension '${dim}' (${String(cause)})`, { cause });
    }
  });

/** The input and state shapes of `graph` under `binding`, refusing names the graph does not have. */
const scenarioShapes = (graph: IrGraph, binding: Binding, where: string) => {
  const stray = Object.keys(binding).filter((name) => !graph.symbols.includes(name));
  if (stray.length > 0) {
    throw new Error(
      `${where}: the binding names ${stray.join(", ")}, not symbols of the graph [${graph.symbols.join(", ")}]`,
    );
  }
  return {
    inputs: Object.fromEntries(
      graph.inputs.map((spec) => [
        spec.name,
        boundShape(spec.shape, binding, `${where} ${spec.name}`),
      ]),
    ),
    states: new Map(
      Object.entries(graph.states).map(([name, slot]) => [
        name,
        boundShape(slot.shape, binding, `${where} state ${name}`),
      ]),
    ),
  };
};

/** The local mirrors (directories under `models/` with a manifest), sorted. */
const localMirrors = async (): Promise<readonly string[]> => {
  const found: string[] = [];
  for (const entry of readDirSync(MODELS_ROOT)) {
    if (entry.isDirectory && (await exists(new URL(`${entry.name}/karume.json`, MODELS_ROOT)))) {
      found.push(entry.name);
    }
  }
  return found.sort();
};

test.skipIf(OUTPUT === undefined)(
  "writes the fusion candidate tables of every graph of every local mirror",
  async () => {
    const out = OUTPUT;
    if (out === undefined)
      throw new Error("unreachable: the case SKIPs without KARUME_FUSION_CANDIDATES");
    const index = [
      "# Fusion candidates — index",
      "",
      `The window limit of each table is where its search stopped (from ${DEFAULT_MAX_WINDOW}, doubled` +
        ` until the rows stop changing, ceiling ${MAX_WINDOW_CEILING}); Session defaults; the default` +
        " model and the first storage of each manifest.",
      "",
      "| Mirror | Model | Component | Storage | Binding | Window limit | Stabilised | Rows | Rows at the limit | Table |",
      "| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |",
    ];
    for (const mirror of await localMirrors()) {
      const root = new URL(`${mirror}/`, MODELS_ROOT);
      const manifest: ContainerManifest = JSON.parse(
        await readTextFile(new URL("karume.json", root)),
      );
      const model = manifest.defaultModel;
      const components = Object.entries(manifest.models[model].weights).sort(([a], [b]) =>
        a < b ? -1 : a > b ? 1 : 0,
      );
      for (const [component, storages] of components) {
        const storage = Object.keys(storages)[0];
        const head = containerPart0(manifest, model, component, storage);
        if (head.repo !== undefined) {
          index.push(
            `| ${mirror} | ${model} | ${component} | ${storage} | — | — | — | — | — | cross-repository reference to ${head.repo} |`,
          );
          continue;
        }
        const where = `${mirror}/${model}/${component}`;
        const graph = await readDistributionGraph(
          root,
          manifest,
          model,
          component,
          storage,
          NO_MIRRORS,
        );
        const binding = BINDINGS[mirror]?.[component] ?? {};
        const shapes = scenarioShapes(graph, binding, where);
        const { plan, context } = planDistributionFusions(
          graph,
          shapes.inputs,
          {},
          undefined,
          shapes.states,
        );
        const { rows, maxWindow, stable } = stableFusionCandidates(plan.steps, context);
        const scenario: CandidateScenario = {
          graph: `${where} (${storage})`,
          shapes: shapes.inputs,
          options: {},
          maxWindow,
          stable,
        };
        await mkdir(`${out}/${mirror}`, { recursive: true });
        await writeTextFile(`${out}/${mirror}/${component}.md`, candidatesMarkdown(rows, scenario));
        await writeTextFile(`${out}/${mirror}/${component}.jsonl`, candidatesJsonl(rows, scenario));
        const bound = Object.entries(binding)
          .map(([name, value]) => `${name}=${value}`)
          .join(" ");
        index.push(
          `| ${mirror} | ${model} | ${component} | ${storage} | ${bound === "" ? "static" : bound}` +
            ` | ${maxWindow} | ${stable ? "yes" : "**no**"} | ${rows.length}` +
            ` | ${rows.filter((row) => row.hitLimit).length} | [${component}.md](${mirror}/${component}.md) |`,
        );
      }
    }
    await mkdir(out, { recursive: true });
    await writeTextFile(`${out}/index.md`, `${index.join("\n")}\n`);
  },
);
