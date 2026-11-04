/**
 * Reads one graph's IR from part 0 of a distribution mirror (`krm`) (up to merging the graph
 * descriptor with the binding table).
 *
 * The real-asset checks (fusion hit counts and the like) look only at the graph declaration, so
 * this never reads into the weight blocks — only the header (24 B) and the two documents. The
 * graph name = the weights key of the distribution manifest (container-v1 §2.1).
 *
 * MUST: when the named graph is missing, fail and list the graphs that are there (silently taking
 * the first one would quietly produce "a table that counted a different component" for a
 * container whose component and graph names have drifted apart).
 */

import { readRange } from "@karume/test-support/fs";
import { bindGraphs, mergedGraph } from "../../src/format/container/bind.ts";
import {
  parseGraphDescriptor,
  parseModelDescriptor,
} from "../../src/format/container/descriptor.ts";
import { readHeader } from "../../src/format/container/header.ts";
import { HEADER_BYTES } from "../../src/format/container/limits.ts";
import type { IrGraph } from "../../src/format/ir.ts";

/**
 * Only the fields of a `karume/5` manifest that this file reads (the distribution form carries
 * the spellings).
 */
export type ContainerManifest = {
  readonly defaultModel: string;
  readonly models: Readonly<
    Record<
      string,
      {
        readonly weights: Readonly<
          Record<
            string,
            Readonly<Record<string, { readonly container: { readonly parts: readonly PartRef[] } }>>
          >
        >;
      }
    >
  >;
};
export type PartRef = { readonly path: string; readonly repo?: string };

const readPrefix = (source: URL, length: number): Promise<Uint8Array<ArrayBuffer>> =>
  readRange(source, 0, length);

/**
 * Reads the merged IR of `graphName` from a container's part 0 (`…-00001-of-NNNNN.krm` or the
 * single form).
 */
export const readContainerGraph = async (part0: URL, graphName: string): Promise<IrGraph> => {
  const header = readHeader(await readPrefix(part0, HEADER_BYTES));
  const documents = await readPrefix(
    part0,
    HEADER_BYTES + header.graphDescriptorLength + header.modelDescriptorLength,
  );
  const graph = parseGraphDescriptor(
    documents.slice(HEADER_BYTES, HEADER_BYTES + header.graphDescriptorLength),
  );
  const model =
    header.kind === "model"
      ? parseModelDescriptor(documents.slice(HEADER_BYTES + header.graphDescriptorLength))
      : undefined;
  const bound = bindGraphs(graph, model)[graphName];
  if (bound === undefined) {
    throw new Error(
      `${part0.pathname}: the container has no graph '${graphName}' (it has ${Object.keys(
        graph.graphs,
      ).join(" / ")})`,
    );
  }
  return mergedGraph(bound, graphName);
};

/**
 * Looks up part 0 of the (model, component, dtype) container in the manifest. The caller routes a
 * cross-repository reference (one with `repo`) to another mirror, so the path is returned as is.
 */
export const containerPart0 = (
  manifest: ContainerManifest,
  model: string,
  component: string,
  dtype: string,
): PartRef => {
  const entry = manifest.models[model]?.weights[component]?.[dtype];
  if (entry === undefined) {
    throw new Error(`the manifest has no container for ${model} / ${component} / ${dtype}`);
  }
  return entry.container.parts[0];
};
