/**
 * A fusion rule written against the kernel entry (`../../kernel.ts`, the module behind
 * `@karume/runtime/kernel`) inside the runtime's own test tree: `gelu(gate)` then the `mul` that
 * reads it, adjacent, in one dispatch (ADR 0148). It is the runtime's GPU twin of an external rule
 * (tests/gpu_external_fusion_rule_test.ts); the models package's `gelu_tanh` rule is checked the
 * same way on the GPU by the e2e package, so the runtime's tests import nothing from the models
 * package.
 *
 * Bit identity with the unfused plan (class E): the gelu value and the product are the plain
 * elementwise kernels' own text (`prologueUnaryWgsl`, `fusedBinaryWgsl`, in the mul's operand
 * order), and the gelu value — the one value the unfused pair stores before the mul reads it — is
 * staged by the rounding barrier (`stagedLoopWgsl`).
 */

import {
  defineRule,
  type FusionMatch,
  fusedBinaryWgsl,
  type NodePlan,
  prologueUnaryWgsl,
  sameShape,
  stagedF32,
  stagedLoopWgsl,
} from "../../kernel.ts";

/** The rule's name, and its counter in `lastRunFusions`. */
export const EXTERNAL_GELU_MUL_RULE_NAME = "test/gelu_mul";

const WORKGROUP_SIZE = 128;

/** The mul's slot of the gelu value: `gelu-other` = `mul(gelu(gate), other)`. */
type Order = "gelu-other" | "other-gelu";

type GeluMulMatch = FusionMatch & {
  readonly gate: string;
  readonly other: string;
  readonly order: Order;
  readonly count: number;
};

const wgsl = (order: Order): string => {
  const gelu = prologueUnaryWgsl("gelu", "gate[i]");
  const act = stagedF32("act");
  const product =
    order === "gelu-other"
      ? fusedBinaryWgsl("mul", act, "other[i]")
      : fusedBinaryWgsl("mul", "other[i]", act);
  return `// karume runtime test rule test/gelu_mul: gelu -> mul (${order}, f32)
struct Params { n: u32, reserved0: u32, reserved1: u32, reserved2: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read> other: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

${gelu.helpers.join("\n\n")}

${stagedLoopWgsl({
  workgroupSize: WORKGROUP_SIZE,
  count: "params.n",
  rounds: [[{ name: "act", value: gelu.value }]],
  finish: [`out[i] = ${product};`],
})}`;
};

/**
 * `gelu(gate)` at `index` and the `mul` right after it that reads it, every value of one shape (no
 * broadcast) and at least one element, the mul's other operand neither the gate nor the gelu value.
 * The runtime checks the
 * dtypes (all f32) and the gelu value's privacy (`internals: "private"`).
 */
const match = (nodes: readonly NodePlan[], index: number): GeluMulMatch | undefined => {
  const [gelu, mul] = [nodes[index], nodes[index + 1]];
  if (gelu?.node.op !== "gelu" || mul?.node.op !== "mul") return undefined;
  const value = gelu.outputs[0].name;
  const order: Order | undefined =
    mul.node.ins[0] === value ? "gelu-other" : mul.node.ins[1] === value ? "other-gelu" : undefined;
  if (order === undefined || gelu.node.ins.length !== 1 || mul.node.ins.length !== 2) {
    return undefined;
  }
  const gate = gelu.node.ins[0];
  const other = mul.node.ins[order === "gelu-other" ? 1 : 0];
  const shape = gelu.inputShapes[0];
  const shapes = [gelu.outputs[0].shape, ...mul.inputShapes, mul.outputs[0].shape];
  if (other === gate || other === value || !shapes.every((each) => sameShape(each, shape))) {
    return undefined;
  }
  const count = shape.reduce((product, dim) => product * dim, 1);
  if (count < 1) return undefined;
  return { window: [gelu, mul], chain: [gelu, mul], gate, other, order, count };
};

export const EXTERNAL_GELU_MUL_RULE = defineRule<GeluMulMatch, typeof EXTERNAL_GELU_MUL_RULE_NAME>({
  name: EXTERNAL_GELU_MUL_RULE_NAME,
  heads: ["gelu"],
  contract: "E",
  internals: "private",
  match,
  build: (matched) => {
    const product = matched.chain[1].outputs[0];
    return {
      binds: [matched.gate, matched.other],
      outputName: product.name,
      outputShape: product.shape,
      temps: [],
      dispatches: [
        {
          key: `ext:test:gelu_mul:v1:${matched.order}:f32:wg${WORKGROUP_SIZE}`,
          wgsl: () => wgsl(matched.order),
          params: new Uint32Array([matched.count, 0, 0, 0]),
          workgroups: { kind: "gridStride", items: matched.count, size: WORKGROUP_SIZE },
        },
      ],
    };
  },
});
