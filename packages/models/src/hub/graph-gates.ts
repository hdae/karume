/**
 * Checks that match the **declaration** of an exported graph against the declaration of
 * `pipelineConfig` (consolidated into one across families).
 *
 * The check is needed in the same form on the sequential surface and on the whole-bytes surface,
 * so it accepts only {@link GraphOwner} (the graph declaration alone) — it touches neither a
 * Session nor the GPU. What differs per family is only **the label in the message** (the family
 * name and which declaration was being checked), so that is taken as an argument.
 */

import type { GraphOwner } from "./components.ts";

/**
 * Checks that the **static** dimension of one axis of a graph input matches the declaration in
 * `pipelineConfig`.
 *
 * MUST: do not drop this check. The host-side preprocessing is built to the declared size, so even
 * when the graph was exported with a different size **the host side runs to the end** (what fails is
 * the Session's shape check, and by then the reader cannot tell "which number is right"). This is
 * the only detector of mixed-up assets.
 * MUST: check **only the axes that matter**, not the whole shape. Data-dependent axes (symbolic
 * dimensions, sequence length) are legitimately non-static, so requiring every axis to be static
 * would reject a correct distribution.
 */
export const assertGraphInputDim = (
  family: string,
  model: GraphOwner,
  inputName: string,
  axis: number,
  expected: number,
  where: string,
): void => {
  const spec = model.graph.inputs.find((input) => input.name === inputName);
  if (spec === undefined) {
    throw new Error(`${family}: graph input '${inputName}' is missing (${where})`);
  }
  const dim = spec.shape[axis];
  if (dim !== expected) {
    throw new Error(
      `${family}: ${where} — graph input '${inputName}' axis ${axis} is ${String(dim)}, ` +
        `pipelineConfig has ${expected}`,
    );
  }
};

/**
 * Checks that the dtype of a graph input matches the dtype the family's preprocessing writes.
 *
 * MUST: do not drop this check. {@link assertGraphInputDim} looks only at axis sizes, so a graph that
 * takes the same shape with a different dtype (e.g. an i32 `pixel_values`) passes the
 * pre-construction checks and fails in the runtime input check — after the GPU and the Session are
 * created and the weights are read — with a message that does not point at the mixed-up asset. As
 * with axes, the family names **only the inputs that matter** (not every input uniformly).
 */
export const assertGraphInputDtype = (
  family: string,
  model: GraphOwner,
  inputName: string,
  expected: GraphOwner["graph"]["inputs"][number]["dtype"],
  where: string,
): void => {
  const spec = model.graph.inputs.find((input) => input.name === inputName);
  if (spec === undefined) {
    throw new Error(`${family}: graph input '${inputName}' is missing (${where})`);
  }
  if (spec.dtype !== expected) {
    throw new Error(
      `${family}: ${where} — graph input '${inputName}' dtype is ${spec.dtype}, ` +
        `the preprocessing writes ${expected}`,
    );
  }
};
