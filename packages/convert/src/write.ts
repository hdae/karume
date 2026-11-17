/**
 * The optional writer: a container source (the converting source, typically) written out part by
 * part with its real hashes.
 *
 * Each part from 1 on is read whole, its sha256 and the sha256 of every block in it are computed,
 * and its bytes go to the sink; part 0 goes last, its model descriptor serialized again with those
 * hashes (the same length — a sha256 is always 64 characters — so the header does not change).
 * NOTE: the sha256 of a part is one Web Crypto digest over the whole part (`sha256Hex` of the
 * runtime): the repository's incremental sha256 is a dependency of the hub, which this package does
 * not use, so a whole part is held while it is hashed and written.
 *
 * SPEC: docs/spec/convert.md#the-writer
 */

import { type BlockSource, openContainer } from "@karume/runtime";
import {
  type DataBlockRecord,
  type ModelDescriptor,
  type Sha256Hex,
  serializeModelDescriptor,
  sha256Hex,
} from "@karume/runtime/container";
import { ConvertError } from "./errors.ts";

/** Where the parts go: `write` is called once per part (part 0 last), at offset 0 with the whole part. */
export type ContainerSink = {
  write(part: number, offset: number, bytes: Uint8Array<ArrayBuffer>): Promise<void>;
};

/** A length and the sha256 of those bytes. */
export type Digest = { readonly length: number; readonly sha256: string };

/** What a manifest entry needs of the written container (`container` of manifest `karume/5`). */
export type WrittenContainer = {
  /** Every part, part 0 included, in index order. */
  readonly parts: readonly Digest[];
  /** The two documents of part 0 — the expectation `openContainer` takes. */
  readonly descriptor: { readonly graph: Digest; readonly model: Digest };
};

/**
 * Writes the container `source` serves to `sink`, with the real sha256 of every block and part.
 * For the converting source of a distributed preset fed with that preset's own weights, the parts
 * are byte-identical to the distributed files.
 */
export const writeContainer = async (
  source: BlockSource,
  sink: ContainerSink,
): Promise<WrittenContainer> => {
  const opened = await openContainer({ kind: "source", source });
  const model = opened.model;
  if (model === undefined) {
    throw new ConvertError("the source is a graph container (krg); only a krm is written");
  }
  const blocksOf = new Map<number, DataBlockRecord[]>();
  for (const block of model.blocks) {
    const list = blocksOf.get(block.part) ?? [];
    list.push(block);
    blocksOf.set(block.part, list);
  }
  const blockHashes = new Map<string, Sha256Hex>();
  const partHashes = new Map<number, Sha256Hex>();
  const parts: Digest[] = [];
  for (let index = 1; index < source.partCount; index += 1) {
    const length = source.partLength(index);
    const bytes =
      length === 0 ? new Uint8Array(new ArrayBuffer(0)) : await source.read(index, 0, length);
    if (bytes.byteLength !== length) {
      throw new ConvertError(`part ${index}: read ${bytes.byteLength} bytes of its ${length}`);
    }
    for (const block of blocksOf.get(index) ?? []) {
      blockHashes.set(
        block.id,
        await sha256Hex(bytes.subarray(block.offset, block.offset + block.length)),
      );
    }
    const sha256 = await sha256Hex(bytes);
    partHashes.set(index, sha256);
    parts[index] = { length, sha256 };
    await sink.write(index, 0, bytes);
  }

  const written: ModelDescriptor = {
    ...model,
    parts: model.parts.map((part) => ({
      ...part,
      sha256: partHashes.get(part.index) ?? part.sha256,
    })),
    blocks: model.blocks.map((block) => ({
      ...block,
      sha256: blockHashes.get(block.id) ?? block.sha256,
    })),
  };
  const modelBytes = serializeModelDescriptor(written);
  const { graphDescriptorLength, modelDescriptorLength } = opened.header;
  if (modelBytes.byteLength !== modelDescriptorLength) {
    throw new Error(
      `the model descriptor with its real hashes is ${modelBytes.byteLength} bytes, the served one ${modelDescriptorLength}`,
    );
  }
  const part0 = (await source.read(0, 0, source.partLength(0))).slice();
  // Part 0 is exactly the header and the two documents, the model descriptor last (container-v1 §1).
  const modelStart = part0.byteLength - modelDescriptorLength;
  part0.set(modelBytes, modelStart);
  const graphBytes = part0.subarray(modelStart - graphDescriptorLength, modelStart);
  parts[0] = { length: part0.byteLength, sha256: await sha256Hex(part0) };
  const descriptor = {
    graph: { length: graphBytes.byteLength, sha256: await sha256Hex(graphBytes) },
    model: { length: modelBytes.byteLength, sha256: await sha256Hex(modelBytes) },
  };
  await sink.write(0, 0, part0);
  return { parts, descriptor };
};
