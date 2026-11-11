/**
 * A candidate fusion rule written the way a package outside the runtime writes one, against
 * `@karume/runtime/kernel` alone (packages/models/tests/external_fusion_rule_test.ts pins that
 * import list): the chain `gelu_tanh(gate) → mul(·, other)` in one dispatch. A candidate, not an
 * adopted rule: no pipeline passes it by default; the session A/B (the opt-in e2e test
 * e2e_gemma4_session_ab_test.ts and the playground's `/session-ab` page) passes it through the gemma
 * pipeline's experimental `fusionRules` to measure it.
 *
 * It is the rewrite of the 2026-11-10 investigation prototype (perf-ledger K-105, research
 * 2026-11-10 §8.1), which reached the rule list through a module load hook and runtime internals.
 * The accepted windows are the prototype's, minus empty shapes (a shape with no element is not taken
 * — the prototype had no such clause): the two nodes adjacent (Gemma 4 26B's routed experts), or one
 * independent `linear` between them (26B's dense MLP: gate linear → gelu_tanh → up linear → mul),
 * which the scan places before the fused step as an in-window passthrough.
 *
 * Bit identity with the unfused plan (class E): the gelu value and the product use the plain
 * elementwise kernels' own text (`prologueUnaryWgsl`, `fusedBinaryWgsl`, in the mul's operand
 * order), and the gelu value — the one value the unfused chain stores before the mul reads it — is
 * staged through workgroup memory by the rounding barrier (`stagedLoopWgsl`), so the compiler
 * cannot fold it into the product. That the barrier holds is measured per backend, not guaranteed
 * by WGSL: the fused-against-unfused check is packages/e2e/tests/e2e_external_fusion_rule_test.ts.
 */

import {
  allF32,
  defineRule,
  type FusionMatch,
  fusedBinaryWgsl,
  type NodePlan,
  prologueUnaryWgsl,
  sameShape,
  stagedF32,
  stagedLoopWgsl,
} from "@karume/runtime/kernel";

/** The rule's name, and its counter in `lastRunFusions`. */
export const GELU_MUL_RULE_NAME = "candidate/gelu_mul";

const WORKGROUP_SIZE = 128;

/** The mul's slot of the gelu value: `gelu-other` = `mul(gelu(gate), other)`. */
type Order = "gelu-other" | "other-gelu";

type GeluMulMatch = FusionMatch & {
  readonly gate: string;
  readonly other: string;
  readonly order: Order;
  readonly count: number;
};

/** MUST: the order stays in the key (it changes the WGSL); the key is under `ext:candidate:`. */
const key = (order: Order): string => `ext:candidate:gelu_mul:v1:${order}:f32:wg${WORKGROUP_SIZE}`;

const wgsl = (order: Order): string => {
  const gelu = prologueUnaryWgsl("gelu_tanh", "gate[i]");
  const act = stagedF32("act");
  const product =
    order === "gelu-other"
      ? fusedBinaryWgsl("mul", act, "other[i]")
      : fusedBinaryWgsl("mul", "other[i]", act);
  return `// karume candidate rule candidate/gelu_mul: gelu_tanh -> mul (${order}, f32)
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

/** The element count of a shape. */
const elements = (shape: readonly number[]): number => shape.reduce((count, dim) => count * dim, 1);

/**
 * `gelu_tanh(gate)` at `index`, then the `mul` that reads it — next, or after one `linear`.
 *
 * Accepted only on an exact match: every slot f32, the gelu value read once and only by the mul
 * (the declared `internals: "private"`, which `defineRule` applies), the mul's other operand
 * neither the gate nor the gelu value, and one shape for the gate, the gelu value, both mul
 * operands and the product (no broadcast), with at least one element.
 */
const match = (nodes: readonly NodePlan[], index: number): GeluMulMatch | undefined => {
  const gelu = nodes[index];
  if (gelu?.node.op !== "gelu_tanh") return undefined;
  const second = nodes[index + 1];
  const between = second?.node.op === "linear" ? second : undefined;
  const mul = between === undefined ? second : nodes[index + 2];
  if (mul?.node.op !== "mul") return undefined;
  const chain = [gelu, mul];
  if (!allF32(chain) || gelu.node.ins.length !== 1 || mul.node.ins.length !== 2) return undefined;
  const value = gelu.outputs[0].name;
  const order: Order | undefined =
    mul.node.ins[0] === value ? "gelu-other" : mul.node.ins[1] === value ? "other-gelu" : undefined;
  if (order === undefined) return undefined;
  const gate = gelu.node.ins[0];
  const other = mul.node.ins[order === "gelu-other" ? 1 : 0];
  if (other === gate || other === value) return undefined;
  const shape = gelu.inputShapes[0];
  if (
    !sameShape(gelu.outputs[0].shape, shape) ||
    mul.inputShapes.some((inputShape) => !sameShape(inputShape, shape)) ||
    !sameShape(mul.outputs[0].shape, shape) ||
    elements(shape) < 1
  ) {
    return undefined;
  }
  return {
    window: between === undefined ? chain : [gelu, between, mul],
    chain,
    gate,
    other,
    order,
    count: elements(shape),
  };
};

export const GELU_MUL_RULE = defineRule<GeluMulMatch, typeof GELU_MUL_RULE_NAME>({
  name: GELU_MUL_RULE_NAME,
  heads: ["gelu_tanh"],
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
          key: key(matched.order),
          wgsl: () => wgsl(matched.order),
          params: new Uint32Array([matched.count, 0, 0, 0]),
          workgroups: { kind: "gridStride", items: matched.count, size: WORKGROUP_SIZE },
        },
      ],
    };
  },
});
