/**
 * A fusion rule for the video / image VAE's channel RMS norm, with the SiLU that follows it, in one
 * dispatch (`models/vae_rms_silu`, perf-ledger V-10). Written the way a package outside the runtime
 * writes a rule, against `@karume/runtime/kernel` alone
 * (packages/models/tests/external_fusion_rule_test.ts pins that import list). Adopted on
 * 2026-11-12: the Wan (2.1 and 2.2) and anima pipelines pass it by default to the Sessions of their
 * VAE decoder graphs (`./fusion-rules.ts`); no other Session gets it by default.
 *
 * The chain it takes is what the exporter lowers the norm to, node for node:
 *
 *     sq = mul(x, x) → s = sum(sq, dim) → r = sqrt(s) → n = clamp_min(r, floor) → reshape (keepdim)
 *       → d = div(x, n) → m1 = mul(d, scale) → m2 = mul(m1, gamma) [→ g = sigmoid(m2) → mul(m2, g)]
 *
 * with `x` laid out as `[1…, C, P…]` (every axis before the channel axis has length 1), `scale` one
 * element and `gamma` one element per channel. The unfused plan runs it as eight dispatches that
 * each pass over the whole activation; the fused kernel reads `x` twice and writes the output once.
 *
 * Bit identity with the unfused plan (class E), and what it rests on:
 * - The channel sum is the axis reduce's tree: 256 leaves per pixel (leaf `slot` folds the squares
 *   of channels `slot, slot + 256, …` from 0.0 in ascending order), then the balanced pairwise tree
 *   over the leaves by halving strides 128, 64, … 1. One workgroup of 256 invocations takes
 *   `256 / 8` = 32 pixels; each invocation builds `32 / 8` = 4 of a pixel's 32 stride-32 nodes
 *   (levels 128, 64, 32), and one invocation per pixel finishes levels 16 … 1.
 *   MUST: this mirrors packages/runtime/src/codegen/reduce.ts. The entry exports no text for a
 *   reduction, so the tree here is a copy; the rule's own u32 check is its only evidence.
 * - Every other value is the plain elementwise kernel's own text (`elementwiseValueWgsl`), in the
 *   node's operand order.
 * - Each value the unfused plan stores before a later arithmetic node reads it passes the rounding
 *   barrier `bitcast(bitcast(v) ^ params.mask)` with a mask that is 0 at run time, so the compiler
 *   cannot contract across it. The staged barrier of the entry (`stagedLoopWgsl`) is an elementwise
 *   loop and does not fit a reduction leaf. That the barrier holds is measured behaviour, not a WGSL
 *   guarantee: the XOR form is measured to hold on Vulkan and D3D12 on NVIDIA (this rule's u32
 *   check, and its D3D12 runs in perf-ledger V-10). On Metal the same form was measured not to act
 *   as a barrier in other kernels (docs/known-issues.md), and this rule has not been run there. It
 *   runs on Metal all the same (a backend difference is recorded, not gated — ADR 0148 addendum of
 *   2026-11-12); where the barrier does not hold, its output can differ from the unfused plan by the
 *   rounding of a contracted multiply-add.
 *
 * The geometry is fixed: 8 channel groups of a workgroup, 32 pixels per workgroup (`g8` in the key).
 * The 4-group form (64 pixels) measured within the noise floor of it and went away with the adoption.
 */

import {
  allF32,
  defineRule,
  elementwiseValueWgsl,
  type FusionMatch,
  type FusionRule,
  type NodePlan,
  sameShape,
} from "@karume/runtime/kernel";

/** The rule's name, and its counter in `lastRunFusions`. */
export const VAE_RMS_SILU_RULE_NAME = "models/vae_rms_silu";

const WORKGROUP_SIZE = 256;
/** The channel groups of one workgroup: `WORKGROUP_SIZE / GROUPS` = 32 pixels per workgroup. */
const GROUPS = 8;
const PIXELS_PER_WORKGROUP = WORKGROUP_SIZE / GROUPS;
/** The leaves of the axis reduce's tree, and the nodes left after its levels 128, 64 and 32. */
const LEAVES = 256;
const STRIDE_32_NODES = 32;

const U32_MAX = 0xffff_ffff;

type Tail = "silu" | "none";

type VaeRmsSiluMatch = FusionMatch & {
  readonly x: string;
  readonly scale: string;
  readonly gamma: string;
  readonly channels: number;
  readonly pixels: number;
  /** The clamp's floor, as the node's attribute. */
  readonly floor: number;
  readonly tail: Tail;
};

/** MUST: the geometry and the tail stay in the key (both change the WGSL); under `ext:models:`. */
const key = (tail: Tail): string =>
  `ext:models:vae_rms_silu:v1:g${GROUPS}:${tail}:f32:wg${WORKGROUP_SIZE}:xor`;

/** The rounding barrier: `value` as a rounded f32 the compiler cannot fold into its reader. */
const rounded = (value: string): string => `bitcast<f32>(bitcast<u32>(${value}) ^ m)`;

const wgsl = (tail: Tail): string => {
  const nodesPerGroup = STRIDE_32_NODES / GROUPS;
  const square = elementwiseValueWgsl("mul", ["xv", "xv"]);
  const root = elementwiseValueWgsl("sqrt", ["s[0]"]);
  const clamped = elementwiseValueWgsl("clamp_min", ["r"], ["floor"]);
  const divided = elementwiseValueWgsl("div", ["xv", "n"]);
  const scaled = elementwiseValueWgsl("mul", ["dv", "sc"]);
  const weighted = elementwiseValueWgsl("mul", ["m1", "gamma[c]"]);
  const gate = elementwiseValueWgsl("sigmoid", ["m2"]);
  const gated = elementwiseValueWgsl("mul", ["m2", "sg"]);
  const used =
    tail === "silu"
      ? [square, root, clamped, divided, scaled, weighted, gate, gated]
      : [square, root, clamped, divided, scaled, weighted];
  // Module-scope declarations may come in any order; sorting the distinct bodies fixes it.
  const helpers = [...new Set(used.flatMap((value) => value.helpers))].sort();
  const finish =
    tail === "silu"
      ? [
          `        let m2 = ${rounded(weighted.value)};`,
          `        let sg = ${rounded(gate.value)};`,
          `        out[c * P + t] = ${gated.value};`,
        ]
      : [`        out[c * P + t] = ${weighted.value};`];
  return `// karume rule models/vae_rms_silu: channel RMS norm${tail === "silu" ? " + SiLU" : ""}, ${GROUPS} channel groups x ${PIXELS_PER_WORKGROUP} pixels per workgroup
struct Params { pixels: u32, channels: u32, floor_bits: u32, mask: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read> gamma: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

${helpers.join("\n\n")}

const PIX: u32 = ${PIXELS_PER_WORKGROUP}u;
// One row of ${STRIDE_32_NODES} stride-32 nodes per pixel, padded to ${STRIDE_32_NODES + 1} words.
var<workgroup> nodes: array<f32, ${PIXELS_PER_WORKGROUP * (STRIDE_32_NODES + 1)}>;
var<workgroup> norms: array<f32, ${PIXELS_PER_WORKGROUP}>;

// Leaf \`slot\` of pixel \`t\`: the squares of channels slot, slot + ${LEAVES}, … folded from 0.0.
fn leaf(slot: u32, t: u32, valid: bool, C: u32, P: u32, m: u32) -> f32 {
  var v = 0.0;
  var c = slot;
  while (c < C) {
    if (valid) {
      let xv = x[c * P + t];
      v = v + ${rounded(square.value)};
    }
    c = c + ${LEAVES}u;
  }
  return v;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let P = params.pixels;
  let C = params.channels;
  let m = params.mask;
  let floor = bitcast<f32>(params.floor_bits);
  let lane = lid % PIX;
  let g = lid / PIX;
  let tiles = P / PIX + select(0u, 1u, P % PIX != 0u);
  var tile = wid.x;
  while (tile < tiles) {
    let t = tile * PIX + lane;
    let valid = t < P;
    for (var q = 0u; q < ${nodesPerGroup}u; q = q + 1u) {
      let l = g * ${nodesPerGroup}u + q;
      // The stride-32 node l of the ${LEAVES}-leaf tree: levels 128, 64, 32
      let a = leaf(l, t, valid, C, P, m) + leaf(l + 128u, t, valid, C, P, m);
      let b = leaf(l + 64u, t, valid, C, P, m) + leaf(l + 192u, t, valid, C, P, m);
      let c2 = leaf(l + 32u, t, valid, C, P, m) + leaf(l + 160u, t, valid, C, P, m);
      let d = leaf(l + 96u, t, valid, C, P, m) + leaf(l + 224u, t, valid, C, P, m);
      nodes[lane * ${STRIDE_32_NODES + 1}u + l] = (a + b) + (c2 + d);
    }
    workgroupBarrier();
    if (g == 0u) {
      var s: array<f32, ${STRIDE_32_NODES}>;
      for (var l = 0u; l < ${STRIDE_32_NODES}u; l = l + 1u) { s[l] = nodes[lane * ${STRIDE_32_NODES + 1}u + l]; }
      for (var st = ${STRIDE_32_NODES / 2}u; st > 0u; st = st / 2u) {
        for (var l = 0u; l < st; l = l + 1u) { s[l] = s[l] + s[l + st]; }
      }
      let r = ${root.value};
      norms[lane] = ${clamped.value};
    }
    workgroupBarrier();
    let n = norms[lane];
    let sc = scale[0u];
    if (valid) {
      for (var c = g; c < C; c = c + ${GROUPS}u) {
        let xv = x[c * P + t];
        let dv = ${rounded(divided.value)};
        let m1 = ${rounded(scaled.value)};
${finish.join("\n")}
      }
    }
    workgroupBarrier();
    tile = tile + nwg.x;
  }
}
`;
};

const elements = (shape: readonly number[]): number => shape.reduce((count, dim) => count * dim, 1);

const single = (step: NodePlan | undefined, op: string, arity: number): step is NodePlan =>
  step !== undefined &&
  step.node.op === op &&
  step.node.ins.length === arity &&
  step.outputs.length === 1;

/**
 * The norm chain that starts at `nodes[index]`, or `undefined`.
 *
 * Every slot of every internal value is pinned (the condition of `internals: "stayInChain"`): the
 * square reads `x` twice, each of sum / sqrt / clamp_min / reshape reads the node before it, the
 * div reads `x` then the reshaped norm, the two muls read the value before them in slot 0, and —
 * with the SiLU — the sigmoid reads `m2` and the last mul reads `m2` then the sigmoid. `x`, `scale`
 * and `gamma` are three distinct values, none produced inside the chain.
 */
const match = (nodes: readonly NodePlan[], index: number): VaeRmsSiluMatch | undefined => {
  const [square, sum, root, clamp, reshape, div, scaled, weighted, gate, gated] = nodes.slice(
    index,
    index + 10,
  );
  if (
    !single(square, "mul", 2) ||
    !single(sum, "sum", 1) ||
    !single(root, "sqrt", 1) ||
    !single(clamp, "clamp_min", 1) ||
    !single(reshape, "reshape", 1) ||
    !single(div, "div", 2) ||
    !single(scaled, "mul", 2) ||
    !single(weighted, "mul", 2)
  ) {
    return undefined;
  }
  const out = (step: NodePlan): string => step.outputs[0].name;
  const x = square.node.ins[0];
  const [scaledIn, scale] = scaled.node.ins;
  const [weightedIn, gamma] = weighted.node.ins;
  if (
    square.node.ins[1] !== x ||
    sum.node.ins[0] !== out(square) ||
    root.node.ins[0] !== out(sum) ||
    clamp.node.ins[0] !== out(root) ||
    reshape.node.ins[0] !== out(clamp) ||
    div.node.ins[0] !== x ||
    div.node.ins[1] !== out(reshape) ||
    scaledIn !== out(div) ||
    weightedIn !== out(scaled)
  ) {
    return undefined;
  }
  const norm = [square, sum, root, clamp, reshape, div, scaled, weighted];
  const internal = new Set(norm.map(out));
  if (
    new Set([x, scale, gamma]).size !== 3 ||
    [x, scale, gamma].some((name) => internal.has(name))
  ) {
    return undefined;
  }

  // The layout: x = [1…, C, P…] reduced over the channel axis; the norm is x without that axis.
  const shape = square.inputShapes[0];
  const axis: unknown = sum.node.attrs?.dim;
  if (typeof axis !== "number" || !Number.isInteger(axis) || axis < 0 || axis >= shape.length) {
    return undefined;
  }
  const channels = shape[axis];
  const pixels = elements(shape.slice(axis + 1));
  const count = elements(shape);
  if (elements(shape.slice(0, axis)) !== 1 || count < 1 || count > U32_MAX) return undefined;
  const reduced = shape.filter((_, at) => at !== axis);
  const kept = shape.map((dim, at) => (at === axis ? 1 : dim));
  const gammaShape = weighted.inputShapes[1];
  const gammaPadded = [
    ...Array<number>(Math.max(0, shape.length - gammaShape.length)).fill(1),
    ...gammaShape,
  ];
  const floor: unknown = clamp.node.attrs?.min;
  if (
    typeof floor !== "number" ||
    !Number.isFinite(Math.fround(floor)) ||
    !sameShape(square.outputs[0].shape, shape) ||
    !sameShape(sum.outputs[0].shape, reduced) ||
    !sameShape(root.outputs[0].shape, reduced) ||
    !sameShape(clamp.outputs[0].shape, reduced) ||
    !sameShape(reshape.outputs[0].shape, kept) ||
    !sameShape(div.outputs[0].shape, shape) ||
    !sameShape(scaled.outputs[0].shape, shape) ||
    !sameShape(weighted.outputs[0].shape, shape) ||
    elements(scaled.inputShapes[1]) !== 1 ||
    // gamma: one element per channel, on the channel axis
    !sameShape(
      gammaPadded,
      shape.map((dim, at) => (at === axis ? dim : 1)),
    )
  ) {
    return undefined;
  }

  const base = { x, scale, gamma, channels, pixels, floor };
  const withTail =
    single(gate, "sigmoid", 1) &&
    single(gated, "mul", 2) &&
    gate.node.ins[0] === out(weighted) &&
    gated.node.ins[0] === out(weighted) &&
    gated.node.ins[1] === out(gate) &&
    sameShape(gate.outputs[0].shape, shape) &&
    sameShape(gated.outputs[0].shape, shape);
  const chain = withTail ? [...norm, gate, gated] : norm;
  if (!allF32(chain)) return undefined;
  return { window: chain, chain, ...base, tail: withTail ? "silu" : "none" };
};

/**
 * The rule (class E). The module-scope call is marked pure for the barrel's tree-shaking: a call a
 * bundler cannot prove pure would keep the module's import of `@karume/runtime/kernel` in every
 * bundle of `@karume/models`.
 */
export const VAE_RMS_SILU_RULE: FusionRule<typeof VAE_RMS_SILU_RULE_NAME> =
  /* @__PURE__ */ defineRule<VaeRmsSiluMatch, typeof VAE_RMS_SILU_RULE_NAME>({
    name: VAE_RMS_SILU_RULE_NAME,
    heads: ["mul"],
    contract: "E",
    internals: "stayInChain",
    match,
    build: (matched) => {
      const product = matched.chain[matched.chain.length - 1].outputs[0];
      const floorBits = new Uint32Array(new Float32Array([matched.floor]).buffer)[0];
      return {
        binds: [matched.x, matched.scale, matched.gamma],
        outputName: product.name,
        outputShape: product.shape,
        temps: [],
        dispatches: [
          {
            key: key(matched.tail),
            wgsl: () => wgsl(matched.tail),
            // The mask is 0: the XOR changes no bit, it only stands between two operations.
            params: new Uint32Array([matched.pixels, matched.channels, floorBits, 0]),
            workgroups: {
              kind: "gridStride",
              // One workgroup per tile of pixels; the tile loop covers what the device limit cuts.
              items: Math.ceil(matched.pixels / PIXELS_PER_WORKGROUP) * WORKGROUP_SIZE,
              size: WORKGROUP_SIZE,
            },
          },
        ],
      };
    },
  });
