/**
 * An entry's fixture on disk: the graph container (`krg`) and the storage table beside it.
 *
 * The storage table is the one thing a `krg` cannot carry: how the distribution stores each
 * model-side weight (codec, group size, row axis). The generator writes its bytes in that encoding,
 * so a corpus Session selects the kernels the real weights select.
 */

import { readFile, readTextFile } from "@karume/test-support/fs";
import type { BoundGraph } from "../../src/format/container/bind.ts";
import { isCodecName } from "../../src/format/container/codecs.ts";
import { sortedByCodePoints } from "../../src/format/container/json.ts";
import { openContainer, type OpenedContainer } from "../../src/format/container/open.ts";
import { type CorpusEntry, fixtureUrls } from "./entries.ts";
import type { StoredEncoding } from "./weights.ts";

const STORAGE_SCHEMA = 1;

/** The storage table's text: initializers in code point order, one encoding each. */
export const serializeStorageTable = (
  graph: string,
  encodings: ReadonlyMap<string, StoredEncoding>,
): string => {
  const initializers: Record<string, StoredEncoding> = {};
  for (const name of sortedByCodePoints(encodings.keys())) {
    const encoding = encodings.get(name);
    if (encoding === undefined) continue;
    initializers[name] = {
      codec: encoding.codec,
      ...(encoding.groupSize === undefined ? {} : { groupSize: encoding.groupSize }),
      ...(encoding.rowAxis === undefined ? {} : { rowAxis: encoding.rowAxis }),
    };
  }
  return `${JSON.stringify({ schema: STORAGE_SCHEMA, graph, initializers }, undefined, 2)}\n`;
};

const isRecord = (value: unknown): value is Readonly<Record<string, unknown>> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/** Reads a storage table; anything outside its shape throws with the place. */
export const parseStorageTable = (
  text: string,
  graph: string,
  where: string,
): ReadonlyMap<string, StoredEncoding> => {
  const document: unknown = JSON.parse(text);
  if (!isRecord(document) || document["schema"] !== STORAGE_SCHEMA) {
    throw new Error(`${where}: not a storage table of schema ${STORAGE_SCHEMA}`);
  }
  if (document["graph"] !== graph) {
    throw new Error(`${where}: written for graph '${String(document["graph"])}', not '${graph}'`);
  }
  const initializers = document["initializers"];
  if (!isRecord(initializers)) throw new Error(`${where}: 'initializers' is not an object`);
  const table = new Map<string, StoredEncoding>();
  for (const [name, raw] of Object.entries(initializers)) {
    const at = `${where} initializer '${name}'`;
    if (!isRecord(raw)) throw new Error(`${at}: not an object`);
    const unknown = Object.keys(raw).filter(
      (key) => key !== "codec" && key !== "groupSize" && key !== "rowAxis",
    );
    if (unknown.length > 0) throw new Error(`${at}: unknown fields [${unknown.join(", ")}]`);
    const { codec, groupSize, rowAxis } = raw;
    if (!isCodecName(codec))
      throw new Error(`${at}: ${JSON.stringify(codec)} is not a registered codec`);
    if (
      groupSize !== undefined &&
      (typeof groupSize !== "number" || !Number.isSafeInteger(groupSize) || groupSize < 1)
    ) {
      throw new Error(`${at}: groupSize ${JSON.stringify(groupSize)} is not a positive integer`);
    }
    if (rowAxis !== undefined && rowAxis !== 0 && rowAxis !== 1) {
      throw new Error(`${at}: rowAxis ${JSON.stringify(rowAxis)} is neither 0 nor 1`);
    }
    table.set(name, {
      codec,
      ...(groupSize === undefined ? {} : { groupSize }),
      ...(rowAxis === undefined ? {} : { rowAxis }),
    });
  }
  return table;
};

/** An opened fixture. */
export type CorpusFixture = {
  readonly container: OpenedContainer;
  readonly bound: BoundGraph;
  readonly storage: ReadonlyMap<string, StoredEncoding>;
  /** Size of the `krg` file. */
  readonly graphBytes: number;
};

/** Opens an entry's fixture (the `krg` through the container reader, the storage table beside it). */
export const openCorpusFixture = async (entry: CorpusEntry): Promise<CorpusFixture> => {
  const urls = fixtureUrls(entry);
  const bytes = await readFile(urls.graph);
  const container = await openContainer({ kind: "bytes", bytes });
  if (container.model !== undefined) {
    throw new Error(
      `${urls.graph.pathname}: a corpus fixture is a graph container (krg), not a krm`,
    );
  }
  const bound = Object.hasOwn(container.graphs, entry.graph)
    ? container.graphs[entry.graph]
    : undefined;
  if (bound === undefined) {
    throw new Error(`${urls.graph.pathname}: no graph '${entry.graph}'`);
  }
  const storage = parseStorageTable(
    await readTextFile(urls.storage),
    entry.graph,
    urls.storage.pathname,
  );
  return { container, bound, storage, graphBytes: bytes.byteLength };
};
