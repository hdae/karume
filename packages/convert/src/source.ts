/**
 * The converting block source: the target preset's container as a `BlockSource` whose weight
 * blocks are encoded from a tensor list while they are read.
 *
 * - Part 0 is the target's, with the sha256 of every converted block and of every part that holds
 *   one replaced by {@link INVALID_SHA256} (the same length, so no other byte moves): the document
 *   never states a hash the served bytes do not have. The writer (`write.ts`) puts the real ones in.
 * - The const part and the asset parts are the target's bytes. When the target is unverified, each
 *   block a read touches is read whole and checked against its declared sha256 first (the check
 *   `openContainer` would have made), because this source declares itself verified.
 * - A weight part is built from the encoded payloads: each block's payload at its offset, its tail
 *   padding and the gaps between blocks 0x00. No weight part of the target is ever read.
 *
 * SPEC: docs/spec/convert.md#what-is-served
 */

import type { BlockSource, CodecName } from "@karume/runtime";
import {
  asSha256Hex,
  type ModelDescriptor,
  type Sha256Hex,
  serializeModelDescriptor,
  sha256Hex,
  type SupplyBlock,
} from "@karume/runtime/container";
import { type Encoded, ENCODED_CODECS, encodeValues } from "./encode.ts";
import { ConvertError } from "./errors.ts";
import { type PlannedInitializer, readPlan } from "./plan.ts";
import { tensorByteLength, type TensorList, tensorValues } from "./tensors.ts";

/** The sha256 a served part 0 states for what is converted here (not the digest of any bytes). */
export const INVALID_SHA256 = "0".repeat(64);

export type ConvertingSourceInput = {
  /**
   * Reads the target preset's container. Only part 0, the const part and the asset parts are read
   * from it.
   */
  readonly target: BlockSource;
  readonly tensors: TensorList;
  /** Initializer name → tensor name (the identity when omitted). */
  readonly tensorName?: (initializer: string) => string;
  /**
   * Keep the encoded payloads of every initializer once they are encoded, for as long as the source
   * is reachable: a block of an initializer encoded before is served from memory, its tensor not
   * read or encoded again. The memory held grows to the converted size of what was read (every
   * payload, without the tail padding and gaps). Default `false`: the payloads of the last tensor
   * asked for only.
   *
   * SPEC: docs/spec/convert.md#what-is-served
   */
  readonly retainEncoded?: boolean;
};

export type ConvertingSource = {
  /** The target's container with its weights encoded from the tensor list (`verified: true`). */
  readonly source: BlockSource;
  /** The tensors of the list no initializer of the plan uses, in the list's order. */
  readonly unusedTensors: readonly string[];
  /**
   * The parts this source builds from the tensor list (every part that holds a converted block), in
   * index order. The target is never read for them, so a caller that fetches the target's parts
   * ahead of the reads leaves these out; every other part (part 0, the const part, the asset parts)
   * is read from the target.
   */
  readonly convertedParts: readonly number[];
};

/** One initializer this source encodes. */
type Conversion = {
  readonly planned: PlannedInitializer;
  readonly tensor: string;
  readonly where: string;
};

/** A converted block: where it is, and which bytes of its initializer's payload it holds. */
type ConvertedBlock = {
  readonly offset: number;
  readonly length: number;
  readonly conversion: Conversion;
  readonly of: "data" | "scale";
  /** The block's first byte in the payload of its data or its scale. */
  readonly payloadOffset: number;
  readonly payloadBytes: number;
};

/** A block passed through from the target, with the sha256 it is checked against. */
type PassedBlock = { readonly offset: number; readonly length: number; readonly sha256: string };

const numelOf = (shape: readonly number[]): number => shape.reduce((count, dim) => count * dim, 1);

const sameShape = (left: readonly number[], right: readonly number[]): boolean =>
  left.length === right.length && left.every((dim, i) => dim === right[i]);

const isEncoded = (codec: CodecName): boolean => ENCODED_CODECS.includes(codec);

/**
 * Builds the converting source. Every check that needs no tensor byte runs here and is reported
 * together, before any tensor is read: an initializer of the plan with no tensor, a tensor whose
 * shape is not the declared one, a codec this package does not encode, and a part that holds both
 * converted blocks and asset blocks (the exporter never writes one).
 *
 * SPEC: docs/spec/convert.md#the-checks-when-it-is-built
 */
export const openConvertingSource = async (
  input: ConvertingSourceInput,
): Promise<ConvertingSource> => {
  const { target, tensors } = input;
  const plan = await readPlan(target);
  const problems: string[] = [];
  const used = new Set<string>();
  const conversions: Conversion[] = [];
  for (const planned of plan.initializers) {
    const { graph, initializer, shape, supply } = planned;
    const tensor = input.tensorName === undefined ? initializer : input.tensorName(initializer);
    const where = `graph '${graph}' initializer '${initializer}'`;
    const codec = supply.encoding.codec;
    if (!isEncoded(codec)) {
      problems.push(
        `${where}: codec '${codec}' is not one this package encodes (${ENCODED_CODECS.join(" / ")})`,
      );
    }
    const source = tensors.get(tensor);
    if (source === undefined) {
      problems.push(`${where}: the tensor list has no tensor '${tensor}'`);
      continue;
    }
    used.add(tensor);
    if (!sameShape(source.shape, shape)) {
      problems.push(
        `${where}: tensor '${tensor}' has shape [${source.shape.join(", ")}] (${numelOf(
          source.shape,
        )} elements), the plan declares [${shape.join(", ")}] (${numelOf(shape)} elements)`,
      );
      continue;
    }
    if (isEncoded(codec))
      conversions.push({ planned, tensor, where: `${where} (tensor '${tensor}')` });
  }

  const converted = new Map<number, ConvertedBlock[]>();
  const convertedIds = new Set<string>();
  const place = (block: SupplyBlock, entry: Omit<ConvertedBlock, "offset" | "length">): void => {
    convertedIds.add(block.id);
    const list = converted.get(block.part) ?? [];
    list.push({ offset: block.offset, length: block.length, ...entry });
    converted.set(block.part, list);
  };
  for (const conversion of conversions) {
    const { supply } = conversion.planned;
    let payloadOffset = 0;
    for (const block of supply.blocks) {
      place(block, { conversion, of: "data", payloadOffset, payloadBytes: block.payloadBytes });
      payloadOffset += block.payloadBytes;
    }
    if (supply.scale === undefined) continue;
    payloadOffset = 0;
    for (const block of supply.scale.pieces ?? [supply.scale]) {
      place(block, { conversion, of: "scale", payloadOffset, payloadBytes: block.payloadBytes });
      payloadOffset += block.payloadBytes;
    }
  }
  for (const list of converted.values()) list.sort((a, b) => a.offset - b.offset);

  const passed = new Map<number, PassedBlock[]>([[1, [...plan.opened.graph.const.blocks]]]);
  for (const block of plan.model.blocks) {
    if (block.role !== "asset") continue;
    if (converted.has(block.part)) {
      problems.push(
        `part ${block.part} holds asset block '${block.id}' beside converted blocks (the exporter writes assets in parts of their own)`,
      );
    }
    const list = passed.get(block.part) ?? [];
    list.push(block);
    passed.set(block.part, list);
  }
  if (problems.length > 0) {
    throw new ConvertError(
      `the tensor list cannot be converted to the target's plan:\n- ${problems.join("\n- ")}`,
    );
  }

  const invalid: Sha256Hex = asSha256Hex(INVALID_SHA256, "the invalid sha256");
  const served: ModelDescriptor = {
    ...plan.model,
    parts: plan.model.parts.map((part) =>
      converted.has(part.index) ? { ...part, sha256: invalid } : part,
    ),
    blocks: plan.model.blocks.map((block) =>
      convertedIds.has(block.id) ? { ...block, sha256: invalid } : block,
    ),
  };
  const part0 = plan.part0.slice();
  // Part 0 is exactly the header and the two documents, the model descriptor last (container-v1 §1).
  part0.set(
    serializeModelDescriptor(served),
    part0.byteLength - plan.opened.header.modelDescriptorLength,
  );

  // The encoded payloads kept. By default those of the last tensor asked for (its data and its scale
  // are separate blocks, read one after the other), nothing older; with `retainEncoded`, every
  // initializer's once encoded. Nothing outside this closure holds them, so they go when the source
  // does.
  const retain = input.retainEncoded === true;
  const kept = new Map<Conversion, Promise<Encoded>>();
  const encode = async ({ planned, tensor, where }: Conversion): Promise<Encoded> => {
    const source = tensors.get(tensor);
    if (source === undefined) throw new ConvertError(`${where}: the tensor list has no tensor`);
    const bytes = await source.read();
    const expected = tensorByteLength(source.dtype, numelOf(planned.shape));
    if (bytes.byteLength !== expected) {
      throw new ConvertError(
        `${where}: reading the tensor gave ${bytes.byteLength} bytes, ${source.dtype} [${source.shape.join(
          ", ",
        )}] is ${expected}`,
      );
    }
    const { supply } = planned;
    const encoded = encodeValues(
      supply.encoding.codec,
      tensorValues(source.dtype, bytes),
      planned.shape,
      supply.encoding.rowAxis ?? 0,
      where,
    );
    const dataBytes = supply.blocks.reduce((total, block) => total + block.payloadBytes, 0);
    if (
      encoded.data.byteLength !== dataBytes ||
      encoded.scale?.byteLength !== supply.scale?.payloadBytes
    ) {
      throw new Error(
        `${where}: the encoded payload (${encoded.data.byteLength} / scale ${String(
          encoded.scale?.byteLength,
        )} bytes) differs from the plan's (${dataBytes} / scale ${String(supply.scale?.payloadBytes)})`,
      );
    }
    return encoded;
  };
  const encodedOf = (conversion: Conversion): Promise<Encoded> => {
    const found = kept.get(conversion);
    if (found !== undefined) return found;
    const encoded = encode(conversion);
    if (!retain) kept.clear();
    kept.set(conversion, encoded);
    // A failure is not kept: a later read tries again (and fails again when the input is at fault).
    encoded.catch(() => {
      if (kept.get(conversion) === encoded) kept.delete(conversion);
    });
    return encoded;
  };

  const buildWeightRange = async (
    blocks: readonly ConvertedBlock[],
    offset: number,
    length: number,
  ): Promise<Uint8Array<ArrayBuffer>> => {
    const out = new Uint8Array(new ArrayBuffer(length));
    const end = offset + length;
    for (const block of blocks) {
      if (block.offset >= end) break;
      const from = Math.max(offset, block.offset);
      const to = Math.min(end, block.offset + block.payloadBytes);
      if (from >= to) continue;
      const encoded = await encodedOf(block.conversion);
      const payload = block.of === "data" ? encoded.data : encoded.scale;
      if (payload === undefined) throw new Error(`${block.conversion.where}: no scale was encoded`);
      const start = block.payloadOffset + (from - block.offset);
      out.set(payload.subarray(start, start + (to - from)), from - offset);
    }
    return out;
  };

  const passThrough = async (
    part: number,
    offset: number,
    length: number,
  ): Promise<Uint8Array<ArrayBuffer>> => {
    if (target.verified) return target.read(part, offset, length);
    const end = offset + length;
    const touched = (passed.get(part) ?? []).filter(
      (block) => block.offset < end && block.offset + block.length > offset,
    );
    if (touched.length === 0) return target.read(part, offset, length);
    const start = Math.min(offset, ...touched.map((block) => block.offset));
    const stop = Math.max(end, ...touched.map((block) => block.offset + block.length));
    const bytes = await target.read(part, start, stop - start);
    if (bytes.byteLength !== stop - start) {
      throw new ConvertError(
        `part ${part}: the target read gave ${bytes.byteLength} bytes, not ${stop - start}`,
      );
    }
    for (const block of touched) {
      const actual = await sha256Hex(
        bytes.subarray(block.offset - start, block.offset - start + block.length),
      );
      if (actual !== block.sha256) {
        throw new ConvertError(
          `part ${part}: the target's block at ${block.offset} has sha256 ${actual}, its descriptor declares ${block.sha256}`,
        );
      }
    }
    return bytes.subarray(offset - start, end - start);
  };

  const partLength = (index: number): number => target.partLength(index);
  const source: BlockSource = {
    partCount: target.partCount,
    verified: true,
    partLength,
    read: async (part, offset, length) => {
      if (!Number.isSafeInteger(part) || part < 0 || part >= target.partCount) {
        throw new ConvertError(`part ${part} does not exist (${target.partCount} parts)`);
      }
      if (
        !Number.isSafeInteger(offset) ||
        !Number.isSafeInteger(length) ||
        offset < 0 ||
        length < 0 ||
        offset + length > partLength(part)
      ) {
        throw new ConvertError(
          `part ${part}: range (offset ${offset}, length ${length}) is not inside its ${partLength(part)} bytes`,
        );
      }
      if (length === 0) return new Uint8Array(new ArrayBuffer(0));
      if (part === 0) return part0.subarray(offset, offset + length);
      const blocks = converted.get(part);
      if (blocks !== undefined) return buildWeightRange(blocks, offset, length);
      return passThrough(part, offset, length);
    },
  };
  return {
    source,
    unusedTensors: [...tensors.keys()].filter((name) => !used.has(name)),
    convertedParts: [...converted.keys()].sort((a, b) => a - b),
  };
};
