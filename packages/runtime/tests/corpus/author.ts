/**
 * The authoring tool of the regression corpus: writes one entry's fixture from a local
 * distribution mirror. A maintainer runs it when an entry is added (it needs the mirror and a GPU);
 * no test runs it, and the mirror is only read.
 *
 *     flock /tmp/karume-gpu.lock node packages/runtime/tests/corpus/author.ts <entry name>
 *
 * It cuts the mirror's declaration to the entry's layers, keeps the const blocks the cut still
 * reads, and takes the stored encoding of every model-side weight left. Before writing, it runs the
 * full graph and the cut graph on generated weights, under every Session setting of the entry, on
 * every run of the entry's plan, and refuses to write unless every step dispatched the same set of
 * kernel keys in both (the cut must exercise what the whole model does). The same mirror and entry
 * give the same bytes.
 */

import { mkdir, writeFile, writeTextFile } from "@karume/test-support/fs";
import { installDawn } from "@karume/test-support/dawn";
import type { BoundContainer, BoundGraph } from "../../src/format/container/bind.ts";
import { payloadBytes } from "../../src/format/container/codecs.ts";
import { openContainer, type OpenedContainer } from "../../src/format/container/open.ts";
import type { IrDeclaration } from "../../src/format/ir.ts";
import { acquireGpu, type GpuContext } from "../../src/gpu/device.ts";
import { createSessionFromContainer, type SessionOptions } from "../../src/runtime/executor.ts";
import { openSeriesContainer } from "../helpers/container-files.ts";
import { type TensorInput, writeGraphContainer } from "../helpers/container-write.ts";
import { cutLayers } from "./cut.ts";
import { driveRun, type StepResult } from "./driver.ts";
import { CORPUS_ENTRIES, type CorpusEntry, fixtureUrls } from "./entries.ts";
import { serializeStorageTable } from "./fixture.ts";
import { constSideOf, corpusSupply } from "./supply.ts";
import { generateWeights, type StoredEncoding } from "./weights.ts";

/** The largest fixture the repository takes (a `krg`, declaration and const region). */
const FIXTURE_LIMIT_BYTES = 2 * 1024 * 1024;

const REPOSITORY = new URL("../../../../", import.meta.url);

/** The stored encoding of every model-side weight of a distribution's graph. */
const storageOf = (bound: BoundGraph): Map<string, StoredEncoding> => {
  const table = new Map<string, StoredEncoding>();
  for (const [name, supply] of bound.supplies) {
    if (supply.origin !== "model") continue;
    const { codec, groupSize, rowAxis } = supply.encoding;
    table.set(name, {
      codec,
      ...(groupSize === undefined ? {} : { groupSize }),
      ...(rowAxis === undefined ? {} : { rowAxis }),
    });
  }
  return table;
};

/** The const tensors the cut still reads, in initializer order (code points). */
const keptConsts = async (
  mirror: OpenedContainer,
  sourceGraph: string,
  cut: IrDeclaration,
  graph: string,
): Promise<TensorInput[]> => {
  const consts: TensorInput[] = [];
  const bindings = mirror.graph.const.constants
    .filter(
      (entry) => entry.graph === sourceGraph && Object.hasOwn(cut.initializers, entry.initializer),
    )
    .sort((a, b) => (a.initializer < b.initializer ? -1 : a.initializer > b.initializer ? 1 : 0));
  for (const binding of bindings) {
    if (binding.encoding.scale !== undefined || binding.encoding.zeroPoint !== undefined) {
      throw new Error(
        `const '${binding.initializer}' is quantized: the authoring tool copies unquantized consts only`,
      );
    }
    const shape = cut.values[binding.initializer].shape.map(Number);
    const length = payloadBytes(
      binding.encoding.codec,
      shape.reduce((count, dim) => count * dim, 1),
      `const '${binding.initializer}'`,
    );
    const block = await mirror.readBlock(binding.block);
    consts.push({
      graph,
      initializer: binding.initializer,
      bytes: block.slice(0, length),
      encoding: { codec: binding.encoding.codec },
    });
  }
  return consts;
};

/** Runs every run of the plan at one level and returns each run's steps. */
const driveAll = async (
  gpu: GpuContext,
  supply: BoundContainer,
  graph: string,
  declaration: IrDeclaration,
  entry: CorpusEntry,
  options: SessionOptions,
): Promise<Map<string, StepResult[]>> => {
  const session = await createSessionFromContainer(gpu, supply, graph, options);
  try {
    const runs = new Map<string, StepResult[]>();
    for (const run of entry.plan.runs) {
      runs.set(run.name, await driveRun(session, declaration, entry.plan, run, entry.seed));
    }
    return runs;
  } finally {
    await session.dispose();
  }
};

const keySet = (step: StepResult): Set<string> => new Set(step.pipelines.map((row) => row.key));

const main = async (): Promise<void> => {
  const name = process.argv[2];
  const entry = CORPUS_ENTRIES.find((candidate) => candidate.name === name);
  if (entry === undefined) {
    throw new Error(
      `usage: node packages/runtime/tests/corpus/author.ts <entry> (entries: ${CORPUS_ENTRIES.map((candidate) => candidate.name).join(", ")})`,
    );
  }
  const mirror = await openSeriesContainer(new URL(entry.source.mirror, REPOSITORY));
  const full = Object.hasOwn(mirror.graphs, entry.source.graph)
    ? mirror.graphs[entry.source.graph]
    : undefined;
  if (full === undefined)
    throw new Error(`${entry.source.mirror} has no graph '${entry.source.graph}'`);
  const { declaration: cut, report } = cutLayers(full.declaration, entry.source.layers);
  console.log(
    `cut to layers [${report.kept.join(", ")}]: ${report.nodes} nodes, ${report.initializers} initializers, ` +
      `${report.states.length} state slots; rewired ${report.rewired.map(([from, to]) => `${from} → ${to}`).join(", ")}`,
  );
  const fullStorage = storageOf(full);
  const cutStorage = new Map(
    [...fullStorage].filter(([initializer]) => Object.hasOwn(cut.initializers, initializer)),
  );
  const consts = await keptConsts(mirror, entry.source.graph, cut, entry.graph);
  const { bytes, descriptorBytes } = await writeGraphContainer({
    graphs: { [entry.graph]: cut },
    consts,
    weights: [],
    assets: [],
    // A graph container carries no provenance (only a krm's model descriptor does).
    provenance: { license: "none" },
  });
  const constBytes = consts.reduce((total, tensor) => total + tensor.bytes.byteLength, 0);
  console.log(
    `fixture: ${bytes.byteLength} bytes (descriptor ${descriptorBytes.byteLength}, ${consts.length} consts of ${constBytes} bytes)`,
  );
  if (bytes.byteLength > FIXTURE_LIMIT_BYTES) {
    const largest = [...consts]
      .sort((a, b) => b.bytes.byteLength - a.bytes.byteLength)
      .slice(0, 5)
      .map((tensor) => `${tensor.initializer} ${tensor.bytes.byteLength}`);
    throw new Error(
      `the fixture is ${bytes.byteLength} bytes, above the ${FIXTURE_LIMIT_BYTES} the repository takes ` +
        `(descriptor ${descriptorBytes.byteLength}, consts ${constBytes}; largest consts: ${largest.join(", ")}): choose a smaller cut`,
    );
  }

  installDawn();
  const gpu = await acquireGpu();
  try {
    let start = performance.now();
    const fullWeights = generateWeights(full, fullStorage, entry.seed);
    console.log(
      `full graph: generated ${fullWeights.size} weights in ${(performance.now() - start).toFixed(0)} ms`,
    );
    const fullSupply = corpusSupply({
      graphName: entry.source.graph,
      declaration: full.declaration,
      consts: constSideOf(mirror),
      weights: fullWeights,
    });
    // The cut goes through the fixture's own bytes, the way the corpus test reads it.
    const krg = await openContainer({ kind: "bytes", bytes });
    const cutBound = krg.graphs[entry.graph];
    start = performance.now();
    const cutWeights = generateWeights(cutBound, cutStorage, entry.seed);
    console.log(
      `cut graph: generated ${cutWeights.size} weights in ${(performance.now() - start).toFixed(0)} ms`,
    );
    const cutSupply = corpusSupply({
      graphName: entry.graph,
      declaration: cutBound.declaration,
      consts: constSideOf(krg),
      weights: cutWeights,
    });
    const differences: string[] = [];
    for (const { name: setting, options } of entry.settings) {
      const fullRuns = await driveAll(
        gpu,
        fullSupply,
        entry.source.graph,
        full.declaration,
        entry,
        options,
      );
      const cutRuns = await driveAll(
        gpu,
        cutSupply,
        entry.graph,
        cutBound.declaration,
        entry,
        options,
      );
      for (const run of entry.plan.runs) {
        const fullSteps = fullRuns.get(run.name) ?? [];
        const cutSteps = cutRuns.get(run.name) ?? [];
        cutSteps.forEach((step, index) => {
          const cutKeys = keySet(step);
          const fullKeys = keySet(fullSteps[index]);
          const onlyCut = [...cutKeys].filter((key) => !fullKeys.has(key));
          const onlyFull = [...fullKeys].filter((key) => !cutKeys.has(key));
          console.log(
            `${setting} ${run.name} step ${index}: ${cutKeys.size} kernel keys in the cut, ${fullKeys.size} in the full graph`,
          );
          if (onlyCut.length > 0 || onlyFull.length > 0) {
            differences.push(
              `${setting} ${run.name} step ${index}: only in the cut [${onlyCut.join(", ")}]; only in the full graph [${onlyFull.join(", ")}]`,
            );
          }
        });
      }
    }
    if (differences.length > 0) {
      throw new Error(
        `the cut does not dispatch the full graph's kernel keys, so nothing was written:\n  ${differences.join("\n  ")}`,
      );
    }
  } finally {
    gpu.destroy();
  }
  const urls = fixtureUrls(entry);
  await mkdir(new URL(".", urls.graph), { recursive: true });
  await writeFile(urls.graph, bytes);
  await writeTextFile(urls.storage, serializeStorageTable(entry.graph, cutStorage));
  console.log(`wrote ${urls.graph.pathname} and ${urls.storage.pathname}`);
};

if (import.meta.main) await main();
