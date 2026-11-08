// Small helpers that build the joined `IrGraph` from a declaration (IR v2).
//
// An IR v2 declaration has no storage, so even tests that only look at **the shape of a graph
// with storage** (plan / fusion / contract checks) need one supply per initializer to get an
// `IrGraph`. What this builds is "a supply whose contents nobody reads" — zero-filled bytes of the
// length the declared shape and codec determine — and the join goes through one in-memory
// container (`openMemoryContainer` → `bindDeclarations`).
//
// MUST: the length formulas come from the codec table (`codecs.ts`). Recounting them in the tests
// would let the length the join layer expects and the supplied length **share the same mistake**,
// making the comparison tautological.

import { mergedGraph } from "../../src/format/container/bind.ts";
import {
  codecEntry,
  type CodecName,
  groupScaleShape,
  payloadBytes,
  perChannelGroupSize,
  quantizedRowLength,
  scaleBytes,
} from "../../src/format/container/codecs.ts";
import type { IrGraph } from "../../src/format/ir.ts";
import type { EncodingInput, TensorInput } from "./container-write.ts";
import { type DeclarationJson, GRAPH_NAME, memoryModel } from "./model-fixture.ts";

/** Storage spec (among the quantized codecs only the group codecs need `groupSize`). */
export type StorageSpec =
  | CodecName
  | { readonly codec: CodecName; readonly groupSize?: number; readonly rowAxis?: 0 | 1 };

/** Initializer name → storage. Names left out get `f32` / `i32` from the value's semantic dtype. */
export type StorageMap = Readonly<Record<string, StorageSpec>>;

const zeros = (length: number): Uint8Array<ArrayBuffer> => new Uint8Array(new ArrayBuffer(length));

const encodingOf = (
  spec: StorageSpec | undefined,
  dtype: string,
  shape: readonly number[],
  name: string,
): EncodingInput => {
  const codec: CodecName =
    spec === undefined
      ? dtype === "i32"
        ? "i32"
        : "f32"
      : typeof spec === "string"
        ? spec
        : spec.codec;
  const entry = codecEntry(codec);
  if (entry.scale === "forbidden" || entry.scaleDtype === undefined) return { codec };
  const rowAxis = typeof spec === "object" ? (spec.rowAxis ?? 0) : 0;
  const rowLength = quantizedRowLength(shape, rowAxis);
  const groupSize =
    entry.grouping === "channel"
      ? perChannelGroupSize(rowLength)
      : typeof spec === "object" && spec.groupSize !== undefined
        ? spec.groupSize
        : rowLength;
  return {
    codec,
    ...(rowAxis === 0 ? {} : { rowAxis }),
    groupSize,
    scale: {
      bytes: zeros(
        scaleBytes(
          codec,
          groupScaleShape(shape, rowAxis, groupSize).reduce((count, dim) => count * dim, 1),
          `scale of ${name}`,
        ),
      ),
      dtype: entry.scaleDtype,
    },
  };
};

/** Builds one contents-unread supply for every initializer of the declaration (except `shared`). */
export const autoTensors = (
  declaration: DeclarationJson,
  storage: StorageMap = {},
): readonly TensorInput[] =>
  Object.entries(declaration.initializers)
    .filter(([, init]) => init.shared !== true)
    .map(([name]) => {
      const value = declaration.values[name];
      const shape = value.shape.map(Number);
      const numel = shape.reduce((count, dim) => count * dim, 1);
      const encoding = encodingOf(storage[name], value.dtype, shape, name);
      return {
        graph: GRAPH_NAME,
        initializer: name,
        bytes: zeros(payloadBytes(encoding.codec, numel, name)),
        encoding,
      };
    });

/**
 * Declaration + **explicit supply** → the joined `IrGraph`. For tests that pass bytes with real
 * contents (actually quantized weights and the like). Tests that do not read the contents use
 * {@link mergeGraph}.
 */
export const mergeTensors = (
  declaration: DeclarationJson,
  tensors: readonly TensorInput[],
): IrGraph => mergedGraph(memoryModel(declaration, tensors).graphs[GRAPH_NAME], GRAPH_NAME);

/**
 * Declaration → the joined `IrGraph`. {@link autoTensors} builds the supply from the declaration
 * (zero-filled). Tests that vary the storage pass a codec per initializer name in `storage`.
 */
export const mergeGraph = (declaration: DeclarationJson, storage: StorageMap = {}): IrGraph =>
  mergeTensors(declaration, autoTensors(declaration, storage));
