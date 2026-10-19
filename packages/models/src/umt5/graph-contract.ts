/**
 * The umT5-XXL graph-declaration contract — a check, run before a single weight byte is fetched,
 * that the graph declaration matches the inputs the host builds (i32 id sequence `[1, L]` and bucket
 * table `[L, L]` — `./session-io.ts`), the output it reads (f32 `[1, L, W]`) and the storage (ADR 0119
 * decisions 3, 4 and 5).
 *
 * Shared by the standalone `Umt5Encoder` (`./encoder.ts`) and Wan's text stage (`umt5Contract` in
 * `src/wan/text-stage.ts` — which adds the match against the DiT's context input). `owner` is the
 * message prefix (`"Umt5Encoder"` / `"WanPipeline"` etc.).
 *
 * NOTE: internal only (not exposed on `mod.ts` / the subpath surface — ADR 0008).
 *
 * MUST: modules are side-effect free (CLAUDE.md).
 */

import { codecLayout } from "@karume/runtime";

import type { GraphOwner } from "../hub/components.ts";
import { UMT5_INPUT_IDS, UMT5_RELATIVE_POSITION_BUCKETS } from "./session-io.ts";

/** The umT5 graph contract verified at construction. */
export type Umt5GraphContract = {
  /** The name of the graph output (`[1, L, hiddenSize]`). */
  readonly output: string;
  /** The output width (4,096 for umT5-XXL — read from the static dimension of the graph declaration). */
  readonly hiddenSize: number;
};

/** The declaration of a umT5 graph input (fails loudly when absent). */
const umt5Input = (
  owner: string,
  textEncoder: GraphOwner,
  name: string,
): GraphOwner["graph"]["inputs"][number] => {
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
 * of a symbolic dimension). `expected` is the shape exactly as the host builds it (the message has
 * the same form as `assertDims` in `src/wan/graph-io.ts`).
 */
const assertDims = (
  owner: string,
  dims: readonly (number | string)[],
  expected: readonly (number | string)[],
  where: string,
): void => {
  if (dims.length !== expected.length || dims.some((dim, axis) => dim !== expected[axis])) {
    const declared = dims.join(", ");
    const host = expected.join(", ");
    throw new Error(`${owner}: ${where} shape [${declared}] differs from the host's [${host}]`);
  }
};

/**
 * Matches the umT5 graph declaration against the inputs the host builds, the output it reads and
 * the storage it accepts (ADR 0119 decisions 3, 4 and 5).
 *
 * MUST: call it at family admission (before fetching the weight parts). The host builds the inputs
 * from its own constants and reads the output as `[1, L, hiddenSize]`, so if the declaration differed
 * the failure would come only after uploading umT5's weights (about 5.3 GiB in i8), in the Session's
 * shape check or in the output check.
 *
 * - Exactly 2 inputs (`input_ids` `[1, L]`, `relative_position_buckets` `[L, L]`, both i32), where L
 *   is **one and the same symbolic dimension** (the shape changes per valid length — decision 4).
 * - One f32 output `[1, L, W]`, with L the same symbol as the inputs. W is `hiddenSize` when given
 *   (Wan passes the width of the DiT's `encoder_hidden_states`); when omitted, the declaration's
 *   static positive integer is taken as is (the standalone encoder).
 * - Storage is only i8 (`int8-sym` = per-channel) weights and f32 tables (decision 5 — the
 *   combination verified by the GPU port check. umT5's Session takes no quant execution knobs, so
 *   the storage decides the form of execution). A container with no i8 at all (all f32, etc.) is not
 *   accepted either.
 */
export const umt5GraphContract = (
  textEncoder: GraphOwner,
  owner: string,
  hiddenSize?: number,
): Umt5GraphContract => {
  const { graph } = textEncoder;
  const ids = umt5Input(owner, textEncoder, UMT5_INPUT_IDS);
  const buckets = umt5Input(owner, textEncoder, UMT5_RELATIVE_POSITION_BUCKETS);
  if (graph.inputs.length !== 2) {
    throw new Error(
      `${owner}: text_encoder has ${graph.inputs.length} graph inputs (` +
        `the host builds only the 2 inputs '${UMT5_INPUT_IDS}' and '${UMT5_RELATIVE_POSITION_BUCKETS}')`,
    );
  }
  const length = ids.shape.at(1);
  if (typeof length !== "string") {
    throw new Error(
      `${owner}: text_encoder's '${UMT5_INPUT_IDS}' axis 1 is not a symbolic dimension` +
        ` (${String(length)}) — the host passes a different shape for each valid length L`,
    );
  }
  for (const [spec, expected] of [
    [ids, [1, length]],
    [buckets, [length, length]],
  ] as const) {
    if (spec.dtype !== "i32") {
      throw new Error(`${owner}: text_encoder's '${spec.name}' has dtype ${spec.dtype}, not i32`);
    }
    assertDims(owner, spec.shape, expected, `text_encoder's '${spec.name}'`);
  }
  if (graph.outputs.length !== 1) {
    throw new Error(
      `${owner}: text_encoder has ${graph.outputs.length} outputs (expected one [1, L, W])`,
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
  const declaredWidth = value.shape.at(2);
  const width = hiddenSize ?? declaredWidth;
  if (typeof width !== "number" || !Number.isInteger(width) || width <= 0) {
    throw new Error(
      `${owner}: text_encoder output '${output}' has shape [${value.shape.join(", ")}], whose axis 2` +
        " is not a static positive integer (the host reads W of [1, L, W] from the declaration)",
    );
  }
  assertDims(owner, value.shape, [1, length, width], `text_encoder output '${output}'`);
  let quantized = 0;
  for (const [name, initializer] of Object.entries(graph.initializers)) {
    if (initializer.storage === undefined) {
      // A shared declaration (borrows the weights of a lender Session — no storage of its own). umT5
      // has no lender.
      throw new Error(
        `${owner}: text_encoder initializer '${name}' is a shared declaration (umT5 borrows no weights)`,
      );
    }
    const layout = codecLayout(initializer.storage.codec);
    if (layout === "i8") quantized += 1;
    else if (layout !== "f32") {
      throw new Error(
        `${owner}: text_encoder initializer '${name}' has storage ${initializer.storage.codec}, which is not accepted` +
          " (only i8 per-channel weights and f32 tables are accepted — ADR 0119 decision 5)",
      );
    }
  }
  if (quantized === 0) {
    throw new Error(
      `${owner}: text_encoder has no i8 weights at all (only i8 per-channel weights and f32 tables are accepted — ` +
        "ADR 0119 decision 5)",
    );
  }
  return { output, hiddenSize: width };
};
