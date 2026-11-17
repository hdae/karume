/**
 * The plan reader: the target preset's part 0 says, per initializer, how it is stored (codec, row
 * axis, group size), in which blocks (the data's pieces and their rows, the scale) and where (part,
 * offset, length). The exporter's rules are not re-derived here — the distributed preset is the
 * plan, read with the runtime's own reader.
 */

import { type BlockSource, type OpenedContainer, openContainer } from "@karume/runtime";
import {
  type InitializerSupply,
  type ModelDescriptor,
  serializeModelDescriptor,
} from "@karume/runtime/container";
import { ConvertError } from "./errors.ts";

/** One initializer the plan stores in the weight parts (not a const and not a shared one). */
export type PlannedInitializer = {
  readonly graph: string;
  readonly initializer: string;
  readonly shape: readonly number[];
  readonly supply: InitializerSupply;
};

/** The target's part 0, read once. */
export type Plan = {
  /** The bytes of part 0 as the target holds them. */
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
 * Reads the plan from the target's part 0 (the only part it reads).
 *
 * `openContainer` parses and merges exactly the bytes read here: it is given a source that serves
 * them as part 0, with the target's part count and lengths, so it checks those against the
 * declarations too. MUST: the model descriptor is in canonical form (`serializeModelDescriptor`
 * gives its bytes back), so that its hashes can be replaced in place without moving another byte.
 */
export const readPlan = async (target: BlockSource): Promise<Plan> => {
  const length = target.partLength(0);
  const part0 = await target.read(0, 0, length);
  if (part0.byteLength !== length) {
    throw new ConvertError(
      `the target's part 0 read gave ${part0.byteLength} bytes, not its length ${length}`,
    );
  }
  const partZero: BlockSource = {
    partCount: target.partCount,
    verified: target.verified,
    partLength: (index) => target.partLength(index),
    read: (part, offset, size) =>
      part === 0
        ? Promise.resolve(part0.subarray(offset, offset + size))
        : Promise.reject(new ConvertError(`the plan reads part 0 only, not part ${part}`)),
  };
  const opened = await openContainer({ kind: "source", source: partZero });
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
  return { part0, opened, model, initializers };
};
