/**
 * The rounding barrier of a fused elementwise kernel, as one generated entry point: a grid-stride
 * loop that keeps every value the unfused sequence wrote to a buffer at its own rounding point.
 *
 * Folding `mul → add` into one expression `r + y * g` lets the shader compiler contract it into an
 * fma, which rounds once where the two plain nodes round twice — the fused kernel then differs from
 * the unfused sequence in the last bit of some finite outputs. The barrier is the form the built-in
 * fusions use (src/kernels/gated-residual.ts, silu.ts, rope.ts, adaln-norm.ts): the bit pattern of the
 * product is written to a workgroup `u32`, and read back after a `workgroupBarrier()`, so the
 * product exists as a rounded f32 before the add sees it.
 *
 * **The WGSL specification does not guarantee this.** It allows floating-point contraction and does
 * not require a workgroup memory round trip to be respected as an optimisation barrier. That the
 * staged form writes the unfused sequence's words is a measured fact per backend, and a rule built
 * on it MUST carry its own fused-vs-unfused u32 check on each backend it ships on
 * (docs/kernel-rules.md); the evidence for this form is the built-in fusions' twin checks, and
 * docs/known-issues.md lists the backends where a staged kernel still differs. The
 * XOR-with-a-runtime-zero form is not offered: it was measured not to act as a barrier on Metal
 * (docs/known-issues.md).
 *
 * How to use it: one staged value per value the unfused sequence materialised and a later
 * arithmetic node consumed — at least every product that feeds an add or a subtract. Values that
 * depend on staged values go in a later round; the last node's value is written by `finish`.
 *
 * MUST: `count` is uniform across the workgroup (a field of a uniform). The barriers sit in the
 * loop `while (base < count)`, and WGSL requires `workgroupBarrier()` in uniform control flow.
 * MUST: `index`, the staged values and `finish` use only `i` (the element index) and `lid` (the
 * local invocation index) of the names this entry point binds; `wid`, `nwg`, `stride` and `base`
 * belong to the loop. They must not write a `var<workgroup>` of their own or call a barrier.
 * MUST: dispatch with the grid-stride workgroup count for `count` elements at this workgroup size
 * (`gridStrideWorkgroups` in src/codegen/dispatch.ts): every element is written however few
 * workgroups run, and the loop arithmetic stays inside u32.
 */

import { CodegenError } from "../codegen/errors.ts";
import { workgroupVariableBytes } from "./workgroup-storage.ts";

/** The WebGPU core defaults a kernel may rely on without requesting a limit. */
const CORE_WORKGROUP_STORAGE_BYTES = 16384;
const CORE_WORKGROUP_INVOCATIONS = 256;

const STAGE_NAME = /^[a-z][a-z0-9_]*$/;

/** One value kept at its rounding point: a name and the f32 WGSL expression of element `i`. */
export type StagedValue = {
  /** `[a-z][a-z0-9_]*`, unique in the kernel; the workgroup variable is `staged_<name>`. */
  readonly name: string;
  /** An f32 expression. It reads an earlier round's value through {@link stagedF32}. */
  readonly value: string;
};

export type StagedLoop = {
  /** Invocations per workgroup, 1 to 256 (the core limit). */
  readonly workgroupSize: number;
  /** A u32 expression of the element count, uniform across the workgroup (e.g. `params.n`). */
  readonly count: string;
  /**
   * Statements deriving indices from `i` (e.g. the row and the offset in the row). Repeated at the
   * top of every block that runs for element `i`, so they must be free of side effects.
   */
  readonly index?: readonly string[];
  /** The rounds in dependency order; a barrier follows each. */
  readonly rounds: readonly (readonly StagedValue[])[];
  /** The statements writing the outputs of element `i`, after the last round. */
  readonly finish: readonly string[];
};

/** The f32 a staged value holds, for a later round's value or for `finish`. */
export const stagedF32 = (name: string): string => `bitcast<f32>(staged_${name}[lid])`;

const assertStagedLoop = (loop: StagedLoop): readonly StagedValue[] => {
  const { workgroupSize } = loop;
  if (
    !Number.isSafeInteger(workgroupSize) ||
    workgroupSize < 1 ||
    workgroupSize > CORE_WORKGROUP_INVOCATIONS
  ) {
    throw new CodegenError(
      `staged loop: the workgroup size must be an integer from 1 to ${CORE_WORKGROUP_INVOCATIONS} (${workgroupSize})`,
    );
  }
  if (loop.rounds.length === 0 || loop.rounds.some((round) => round.length === 0)) {
    throw new CodegenError("staged loop: there must be at least one round, and no round is empty");
  }
  if (loop.finish.length === 0) {
    throw new CodegenError("staged loop: finish must write the outputs");
  }
  const values = loop.rounds.flat();
  const names = new Set<string>();
  for (const { name } of values) {
    if (!STAGE_NAME.test(name)) {
      throw new CodegenError(`staged loop: '${name}' is not a staged value name ([a-z][a-z0-9_]*)`);
    }
    if (names.has(name)) throw new CodegenError(`staged loop: the name '${name}' is used twice`);
    names.add(name);
  }
  // Each `var<workgroup>` counts separately, rounded up to 16 bytes (./workgroup-storage.ts)
  const bytes = values.length * workgroupVariableBytes(4 * workgroupSize);
  if (bytes > CORE_WORKGROUP_STORAGE_BYTES) {
    throw new CodegenError(
      `staged loop: ${values.length} staged values of ${workgroupSize} words take ${bytes} bytes of` +
        ` workgroup storage, above the core limit of ${CORE_WORKGROUP_STORAGE_BYTES}`,
    );
  }
  return values;
};

const block = (count: string, statements: readonly string[]): string =>
  [`    if (i < ${count}) {`, ...statements.map((line) => `      ${line}`), "    }"].join("\n");

/**
 * The workgroup variables and the entry point `main` of a staged grid-stride loop. The caller
 * writes what precedes it: the `Params` struct, the bindings and any helper functions.
 *
 * MUST: the same `loop` gives byte-identical WGSL (the pipeline cache's determinism contract); a
 * kernel built from it bumps its key when its `loop` changes.
 */
export const stagedLoopWgsl = (loop: StagedLoop): string => {
  const values = assertStagedLoop(loop);
  const { workgroupSize, count } = loop;
  const index = loop.index ?? [];
  const rounds = loop.rounds.map((round) =>
    [
      block(count, [
        ...index,
        ...round.map(({ name, value }) => `staged_${name}[lid] = bitcast<u32>(${value});`),
      ]),
      "    workgroupBarrier();",
    ].join("\n"),
  );
  return `${values.map(({ name }) => `var<workgroup> staged_${name}: array<u32, ${workgroupSize}>;`).join("\n")}

@compute @workgroup_size(${workgroupSize})
fn main(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let stride = nwg.x * ${workgroupSize}u;
  var base = wid.x * ${workgroupSize}u;
  while (base < ${count}) {
    let i = base + lid;
${rounds.join("\n")}
${block(count, [...index, ...loop.finish])}
    workgroupBarrier();
    base = base + stride;
  }
}
`;
};
