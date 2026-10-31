/**
 * The Qwen3.5 text tower's graph-declaration contract — a check, run before a single weight byte is
 * uploaded, that the graph declaration matches the inputs the host builds (`./session-io.ts`), the
 * output it reads and the storage this module was verified with (ADR 0137 decisions 2 and 5).
 *
 * `owner` is the message prefix (`"Qwen3_5TextEncoder"`, or a pipeline's name when it checks the
 * component during its own admission).
 *
 * NOTE: internal only (no public subpath yet — the module doc of `./tokenizer.ts`).
 *
 * MUST: modules are side-effect free (CLAUDE.md).
 */

import { codecLayout } from "@karume/runtime";

import type { GraphOwner } from "../hub/components.ts";
import { QWEN3_5_ROTARY_DIM } from "./rope.ts";
import {
  QWEN3_5_INPUT_IDS,
  QWEN3_5_POSITIONS,
  QWEN3_5_ROPE_COS,
  QWEN3_5_ROPE_SIN,
} from "./session-io.ts";

/** The Qwen3.5 graph contract verified before the Session is built. */
export type Qwen3_5GraphContract = {
  /** The name of the graph output (`[1, T, hiddenSize]`). */
  readonly output: string;
  /** The output width (4,096 for Qwen3.5-9B — read from the static dimension of the declaration). */
  readonly hiddenSize: number;
  /** The row count of the embedding table `input_ids` indexes (248,320 for Qwen3.5-9B). */
  readonly vocabularySize: number;
};

type GraphInput = GraphOwner["graph"]["inputs"][number];

/** The declaration of one graph input (fails loudly when absent). */
const inputOf = (owner: string, textEncoder: GraphOwner, name: string): GraphInput => {
  const spec = textEncoder.graph.inputs.find((input) => input.name === name);
  if (spec === undefined) {
    const declared = textEncoder.graph.inputs.map((input) => input.name).join(" / ");
    throw new Error(
      `${owner}: text_encoder graph input '${name}' is missing (declared: ${declared})`,
    );
  }
  return spec;
};

/**
 * Matches a declared shape on rank and every axis (a number is a static dimension, a string the name
 * of a symbolic dimension) against the shape exactly as the host builds it.
 */
const assertDims = (
  owner: string,
  dims: readonly (number | string)[],
  expected: readonly (number | string)[],
  where: string,
): void => {
  if (dims.length !== expected.length || dims.some((dim, axis) => dim !== expected[axis])) {
    throw new Error(
      `${owner}: ${where} shape [${dims.join(", ")}] differs from the host's [${expected.join(", ")}]`,
    );
  }
};

/** A static positive integer dimension (the host reads widths from the declaration). */
const staticDim = (dim: number | string | undefined): number | undefined =>
  typeof dim === "number" && Number.isInteger(dim) && dim > 0 ? dim : undefined;

/**
 * Matches the Qwen3.5 graph declaration against the inputs the host builds, the output it reads and
 * the storage it accepts.
 *
 * MUST: call it before the Session is built (and, on a fetching surface, before the weight parts are
 * fetched). The host builds the inputs from its own constants, so a declaration that differed would
 * fail only after uploading about 7.4 GiB of i8 weights, in the Session's shape check — or not at
 * all (a different rotary width in the rows would rotate other dimensions, silently).
 *
 * - Exactly 4 inputs: `input_ids` i32 `[1, T]`, `positions` f32 `[T]`, `rope_cos` / `rope_sin` f32
 *   `[1, T, 64]`, where T is **one and the same symbolic dimension** and 64 is the pinned rotary
 *   width ({@link QWEN3_5_ROTARY_DIM} — the only RoPE constant the container can be checked
 *   against; theta cannot, `./rope.ts`).
 * - One f32 output `[1, T, W]` with T the same symbol and W a static positive integer.
 * - Exactly one `embedding` node reads `input_ids`, and its table is `[V, W]` with V static (the
 *   host's id range check — `qwen3_5SessionInputs`).
 * - Storage is only i8 per-channel weights and f32 tensors, with at least one i8 (ADR 0137 decision
 *   5 — the combination the GPU parity check verified; the Session takes no execution knobs, so the
 *   storage decides the form of execution). BF16 storage and an all-f32 container are rejected until
 *   they are verified the same way.
 */
export const qwen3_5GraphContract = (
  textEncoder: GraphOwner,
  owner: string,
): Qwen3_5GraphContract => {
  const { graph } = textEncoder;
  const ids = inputOf(owner, textEncoder, QWEN3_5_INPUT_IDS);
  const positions = inputOf(owner, textEncoder, QWEN3_5_POSITIONS);
  const cos = inputOf(owner, textEncoder, QWEN3_5_ROPE_COS);
  const sin = inputOf(owner, textEncoder, QWEN3_5_ROPE_SIN);
  if (graph.inputs.length !== 4) {
    throw new Error(
      `${owner}: text_encoder has ${graph.inputs.length} graph inputs (the host builds only the 4 inputs ` +
        `'${QWEN3_5_INPUT_IDS}', '${QWEN3_5_POSITIONS}', '${QWEN3_5_ROPE_COS}' and '${QWEN3_5_ROPE_SIN}')`,
    );
  }
  const length = ids.shape.at(1);
  if (typeof length !== "string") {
    throw new Error(
      `${owner}: text_encoder's '${QWEN3_5_INPUT_IDS}' axis 1 is not a symbolic dimension` +
        ` (${String(length)}) — the host passes a different shape for each valid length T`,
    );
  }
  for (const [spec, dtype, expected] of [
    [ids, "i32", [1, length]],
    [positions, "f32", [length]],
    [cos, "f32", [1, length, QWEN3_5_ROTARY_DIM]],
    [sin, "f32", [1, length, QWEN3_5_ROTARY_DIM]],
  ] as const) {
    if (spec.dtype !== dtype) {
      throw new Error(
        `${owner}: text_encoder's '${spec.name}' has dtype ${spec.dtype}, not ${dtype}`,
      );
    }
    assertDims(owner, spec.shape, expected, `text_encoder's '${spec.name}'`);
  }

  if (graph.outputs.length !== 1) {
    throw new Error(
      `${owner}: text_encoder has ${graph.outputs.length} outputs (expected one [1, T, W])`,
    );
  }
  const [output] = graph.outputs;
  if (!Object.hasOwn(graph.values, output)) {
    throw new Error(`${owner}: graph value '${output}' is not declared`);
  }
  const value = graph.values[output];
  if (value.dtype !== "f32") {
    throw new Error(`${owner}: text_encoder output '${output}' has dtype ${value.dtype}, not f32`);
  }
  const hiddenSize = staticDim(value.shape.at(2));
  if (hiddenSize === undefined) {
    throw new Error(
      `${owner}: text_encoder output '${output}' has shape [${value.shape.join(", ")}], whose axis 2` +
        " is not a static positive integer (the host reads W of [1, T, W] from the declaration)",
    );
  }
  assertDims(owner, value.shape, [1, length, hiddenSize], `text_encoder output '${output}'`);

  const lookups = graph.nodes.filter(
    (node) => node.op === "embedding" && node.ins[1] === QWEN3_5_INPUT_IDS,
  );
  if (lookups.length !== 1) {
    throw new Error(
      `${owner}: text_encoder has ${lookups.length} embedding nodes that read '${QWEN3_5_INPUT_IDS}' (expected 1)`,
    );
  }
  const table = lookups[0].ins[0];
  const tableShape = Object.hasOwn(graph.values, table) ? graph.values[table].shape : undefined;
  const vocabularySize = staticDim(tableShape?.[0]);
  if (tableShape?.length !== 2 || vocabularySize === undefined) {
    throw new Error(
      `${owner}: text_encoder's embedding table '${table}' has shape [${tableShape?.join(", ") ?? "undeclared"}]` +
        " (expected a static [V, W])",
    );
  }

  let quantized = 0;
  for (const [name, initializer] of Object.entries(graph.initializers)) {
    if (initializer.storage === undefined) {
      throw new Error(
        `${owner}: text_encoder initializer '${name}' is a shared declaration (the text tower borrows no weights)`,
      );
    }
    const layout = codecLayout(initializer.storage.codec);
    if (layout === "i8") quantized += 1;
    else if (layout !== "f32") {
      throw new Error(
        `${owner}: text_encoder initializer '${name}' has storage ${initializer.storage.codec}, which is not accepted` +
          " (only i8 per-channel weights and f32 tensors are verified — ADR 0137 decision 5)",
      );
    }
  }
  if (quantized === 0) {
    throw new Error(
      `${owner}: text_encoder has no i8 weights at all (only the i8 series is verified — ADR 0137 decision 5)`,
    );
  }
  return { output, hiddenSize, vocabularySize };
};
