/**
 * The plan reader: the target preset's part 0 says, per initializer, how it is stored (codec, row
 * axis, group size), in which blocks (the data's pieces and their rows, the scale) and where (part,
 * offset, length). The exporter's rules are not re-derived here — the distributed preset is the
 * plan, read with the runtime's own reader. The one change a caller may ask for is the 16-bit float
 * codec ({@link PlanOptions.float16Codec}): f16 and bf16 store an element in the same two bytes, so
 * naming one where the plan names the other moves no block.
 */

import {
  type BlockSource,
  type CodecName,
  type OpenedContainer,
  openContainer,
} from "@karume/runtime";
import {
  type InitializerSupply,
  type ModelDescriptor,
  serializeModelDescriptor,
} from "@karume/runtime/container";
import { ConvertError } from "./errors.ts";

/**
 * The 16-bit float codecs: an element in 2 bytes, no scale, the same packing in the codec ledger —
 * so one can stand where a plan has the other with every block, part, offset and length unchanged.
 */
export type Float16Codec = "f16" | "bf16";

const FLOAT16_CODECS: readonly CodecName[] = ["f16", "bf16"];

/** One supply of the model descriptor's binding table. */
type WeightSupply = ModelDescriptor["binding"][string][string];

/** How the plan is read. */
export type PlanOptions = {
  /**
   * The codec every initializer the target stores as a 16-bit float (`f16` or `bf16`) is stored
   * as: the plan is the target's with this codec in their place, everything else unchanged. Its
   * part 0 is one of its own — the model descriptor names the codec, so its length, and part 0's,
   * differ from the target's by the length of the names. Omitted: the target's plan as it is.
   *
   * SPEC: docs/spec/convert.md#a-plan-with-another-16-bit-float-codec
   */
  readonly float16Codec?: Float16Codec;
};

/** One initializer the plan stores in the weight parts (not a const and not a shared one). */
export type PlannedInitializer = {
  readonly graph: string;
  readonly initializer: string;
  readonly shape: readonly number[];
  readonly supply: InitializerSupply;
};

/** The target's part 0, read once (with another 16-bit float codec, the part 0 made from it). */
export type Plan = {
  /**
   * The bytes of part 0: the target's, or with another 16-bit float codec the target's header and
   * graph descriptor with the relabelled model descriptor (its length in the header).
   */
  readonly part0: Uint8Array<ArrayBuffer>;
  /** The parsed and merged part 0 (`header`, `graph`, `graphs`); its source reads only part 0. */
  readonly opened: OpenedContainer;
  readonly model: ModelDescriptor;
  /** Every model-bound initializer, in graph and declaration order. */
  readonly initializers: readonly PlannedInitializer[];
};

const sameBytes = (left: Uint8Array<ArrayBuffer>, right: Uint8Array<ArrayBuffer>): boolean =>
  left.byteLength === right.byteLength && left.every((byte, i) => byte === right[i]);

/**
 * Opens `part0` as the part 0 of a container with the target's other parts: `openContainer` parses
 * and merges exactly these bytes, and checks the target's part count and lengths against the
 * declarations too. No other part is read.
 */
const openPartZero = async (
  target: BlockSource,
  part0: Uint8Array<ArrayBuffer>,
): Promise<OpenedContainer> => {
  const partZero: BlockSource = {
    partCount: target.partCount,
    verified: target.verified,
    partLength: (index) => (index === 0 ? part0.byteLength : target.partLength(index)),
    read: (part, offset, size) =>
      part === 0
        ? Promise.resolve(part0.subarray(offset, offset + size))
        : Promise.reject(new ConvertError(`the plan reads part 0 only, not part ${part}`)),
  };
  return await openContainer({ kind: "source", source: partZero });
};

const initializersOf = (opened: OpenedContainer): PlannedInitializer[] => {
  const initializers: PlannedInitializer[] = [];
  for (const [graph, bound] of Object.entries(opened.graphs)) {
    for (const [initializer, supply] of bound.supplies) {
      if (supply.origin !== "model") continue;
      initializers.push({
        graph,
        initializer,
        shape: bound.declaration.values[initializer].shape.map(Number),
        supply,
      });
    }
  }
  return initializers;
};

/**
 * The model descriptor with `codec` in place of every 16-bit float codec of its binding table and
 * of its codec set, or `undefined` when no initializer changes. Every other field is the target's:
 * the two codecs have the same packing, so the declared lengths still hold.
 */
const withFloat16Codec = (
  model: ModelDescriptor,
  codec: Float16Codec,
): ModelDescriptor | undefined => {
  let changed = false;
  const relabel = (supply: WeightSupply): WeightSupply => {
    const from = supply.encoding.codec;
    if (from === codec || !FLOAT16_CODECS.includes(from)) return supply;
    changed = true;
    return { ...supply, encoding: { ...supply.encoding, codec } };
  };
  const binding = Object.fromEntries(
    Object.entries(model.binding).map(([graph, supplies]) => [
      graph,
      Object.fromEntries(
        Object.entries(supplies).map(([initializer, supply]) => [initializer, relabel(supply)]),
      ),
    ]),
  );
  if (!changed) return undefined;
  const codecs = [
    ...new Set(model.codecs.map((name) => (FLOAT16_CODECS.includes(name) ? codec : name))),
  ];
  return { ...model, codecs, binding };
};

/**
 * Part 0 of the plan with `model` as its model descriptor: the target's header with the new model
 * descriptor length, the target's graph descriptor, then `model` serialized. The header's last 8
 * bytes are the model descriptor's length (u64, little endian — docs/container-v1.md §1); the
 * runtime's reader parses the result again, so a header this writes wrongly is refused there.
 */
const part0With = (
  part0: Uint8Array<ArrayBuffer>,
  opened: OpenedContainer,
  model: ModelDescriptor,
): Uint8Array<ArrayBuffer> => {
  const modelBytes = serializeModelDescriptor(model);
  const head = part0.byteLength - opened.header.modelDescriptorLength;
  const out = new Uint8Array(new ArrayBuffer(head + modelBytes.byteLength));
  out.set(part0.subarray(0, head), 0);
  out.set(modelBytes, head);
  const headerBytes = head - opened.header.graphDescriptorLength;
  new DataView(out.buffer).setBigUint64(headerBytes - 8, BigInt(modelBytes.byteLength), true);
  return out;
};

/**
 * Reads the plan from the target's part 0 (the only part it reads), with the 16-bit float codec of
 * `options` in place when one is given ({@link PlanOptions.float16Codec}).
 *
 * MUST: the target's model descriptor is in canonical form (`serializeModelDescriptor` gives its
 * bytes back), so that its hashes can be replaced in place without moving another byte. A plan with
 * another 16-bit float codec is serialized here, so it is canonical by construction, and it is
 * opened by the runtime's reader like the target's.
 */
export const readPlan = async (target: BlockSource, options: PlanOptions = {}): Promise<Plan> => {
  const length = target.partLength(0);
  const part0 = await target.read(0, 0, length);
  if (part0.byteLength !== length) {
    throw new ConvertError(
      `the target's part 0 read gave ${part0.byteLength} bytes, not its length ${length}`,
    );
  }
  const opened = await openPartZero(target, part0);
  const model = opened.model;
  if (model === undefined) {
    throw new ConvertError("the target is a graph container (krg); a plan needs a krm");
  }
  const modelBytes = part0.subarray(length - opened.header.modelDescriptorLength);
  if (!sameBytes(serializeModelDescriptor(model), modelBytes)) {
    throw new ConvertError(
      "the target's model descriptor is not in canonical form, so its hashes cannot be replaced in place",
    );
  }
  const relabelled =
    options.float16Codec === undefined ? undefined : withFloat16Codec(model, options.float16Codec);
  if (relabelled === undefined) {
    return { part0, opened, model, initializers: initializersOf(opened) };
  }
  const derived = part0With(part0, opened, relabelled);
  const reopened = await openPartZero(target, derived);
  return {
    part0: derived,
    opened: reopened,
    model: relabelled,
    initializers: initializersOf(reopened),
  };
};

/** How the plan stores one initializer: what a caller decides on before it converts. */
export type PlannedCodec = {
  readonly graph: string;
  readonly initializer: string;
  readonly codec: CodecName;
};

/**
 * The codec of every initializer the target's plan stores in its weight parts, in the order of its
 * graphs and of their binding tables — what the converting source would encode, read from part 0 alone (the same
 * reading and the same checks of part 0). A caller decides with it before it gives a tensor list:
 * whether every codec is one this package encodes (`ENCODED_CODECS`), whether a tensor's dtype can
 * be stored as its initializer's codec. With `options.float16Codec`, the codecs of the plan the
 * converting source given the same option encodes against.
 *
 * SPEC: docs/spec/convert.md#the-plan-before-a-conversion
 */
export const readPlanCodecs = async (
  target: BlockSource,
  options: PlanOptions = {},
): Promise<readonly PlannedCodec[]> =>
  (await readPlan(target, options)).initializers.map(({ graph, initializer, supply }) => ({
    graph,
    initializer,
    codec: supply.encoding.codec,
  }));
