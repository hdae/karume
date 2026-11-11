/**
 * A fused attention on an a8 seat that absorbs the [0, 2, 1, 3] permutes of its q / k / v
 * (perf-ledger F-3 A4 / F-8 — `attentionPermutesAt` of the runtime's src/runtime/fusion.ts,
 * honoured by `buildAttention` on every route an a8 seat can take) and, on the online route only,
 * the permute of O after it (the kernel stores O token-major), against the same graph with a
 * zero-dispatch reshape between each input permute and the attention and between the attention and
 * the output permute: the values and the physical kernels stay the graph's, only the exact match
 * stops matching. u32 identity with NaN by payload, attentionPermute at the route's count and 0.
 *
 * Two parts: the routes any device takes (the three-dispatch s16 route, an online seat whose shape
 * falls back, the two a8 shapes whose stage falls back to f32, and the f32 seat, which absorbs
 * nothing), and the online routes in their sg32 form, which need a device whose online kernel takes
 * that form (on another device the part is not run). The online routes in the wg form need a device
 * without subgroups, so they run in the runtime's node test only.
 */

import type { GpuContext } from "../../../runtime/src/gpu/device.ts";
import {
  attentionDeviceFacts,
  attentionRoute,
} from "../../../runtime/src/runtime/attention-route.ts";
import type { SessionOptions } from "../../../runtime/src/runtime/executor.ts";
import {
  type DeclarationJson,
  fill,
  type FilledTensor,
  openGraphModel,
} from "../../../runtime/tests/helpers/model-fixture.ts";
import { checkFusionTwin } from "../fusion-twin.ts";
import { type CheckOptions, type CheckPart, type ExpectedCases, notRunPart } from "../result.ts";
import { twinCases, withTwinSessions } from "./session.ts";

export type AttentionPermuteShape = {
  readonly b: number;
  readonly h: number;
  readonly s: number;
  readonly n: number;
  readonly d: number;
};

/**
 * Token-major q / k / v → permute to head-major → attention → permute O back (the Wan DiT form).
 * `interpose` puts a zero-dispatch reshape after each input permute and after the attention.
 */
export const permutedAttentionGraph = (
  shape: AttentionPermuteShape,
  interpose: boolean,
): DeclarationJson => {
  const { b, h, s, n, d } = shape;
  const dtype = "f32";
  const token = { q: [b, s, h, d], k: [b, n, h, d], v: [b, n, h, d] } as const;
  const head = { q: [b, h, s, d], k: [b, h, n, d], v: [b, h, n, d] } as const;
  const values: DeclarationJson["values"] = {
    o: { dtype, shape: [b, h, s, d] },
    ot: { dtype, shape: [b, s, h, d] },
  };
  const nodes: DeclarationJson["nodes"] = [];
  for (const operand of ["q", "k", "v"] as const) {
    values[operand] = { dtype, shape: [...head[operand]] };
    nodes.push({
      op: "permute",
      ins: [`${operand}t`],
      outs: [operand],
      attrs: { dims: [0, 2, 1, 3] },
    });
  }
  const read = (name: string, shape: readonly number[]): string => {
    if (!interpose) return name;
    values[`${name}_r`] = { dtype, shape: [...shape] };
    nodes.push({ op: "reshape", ins: [name], outs: [`${name}_r`], attrs: {} });
    return `${name}_r`;
  };
  const ins = (["q", "k", "v"] as const).map((operand) => read(operand, head[operand]));
  const scale = Math.fround(1 / Math.sqrt(d));
  const attended = interpose ? "o_a" : "o";
  if (interpose) values[attended] = { dtype, shape: [b, h, s, d] };
  nodes.push({ op: "attention", ins, outs: [attended], attrs: { scale } });
  if (interpose) nodes.push({ op: "reshape", ins: [attended], outs: ["o"], attrs: {} });
  nodes.push({ op: "permute", ins: ["o"], outs: ["ot"], attrs: { dims: [0, 2, 1, 3] } });
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))].sort() },
    symbols: [],
    inputs: (["q", "k", "v"] as const).map((operand) => ({
      name: `${operand}t`,
      dtype,
      shape: [...token[operand]],
    })),
    outputs: ["ot"],
    initializers: {},
    values,
    nodes,
  };
};

export const attentionPermuteFeeds = (
  shape: AttentionPermuteShape,
): Readonly<Record<string, FilledTensor>> => {
  const { b, h, s, n, d } = shape;
  return {
    qt: fill([b, s, h, d], (i) => Math.sin(i * 0.37) * 1.7 + ((i % 13) - 6) / 9),
    kt: fill([b, n, h, d], (i) => Math.cos(i * 0.23) * 1.3 + ((i % 7) - 3) / 7),
    vt: fill([b, n, h, d], (i) => Math.sin(i * 0.11 + 1) * 2.1 + ((i % 5) - 2) / 3),
  };
};

/** The dispatches of the kernels the absorption moves, and the total (the node test's census). */
export type AttentionPermuteDispatches = {
  readonly strided: number;
  readonly transpose: number;
  readonly quantizeBshd: number;
  readonly quantizeRows: number;
  readonly total: number;
};

export type AttentionPermuteRoute = {
  readonly name: string;
  readonly options: SessionOptions;
  readonly shape: AttentionPermuteShape;
  /**
   * The subgroup request of the device (default "available"); `false` leaves the online kernel only
   * its wg form.
   */
  readonly subgroups?: false;
  /** The online kernel's reduction form the route must take (online routes only). */
  readonly reduce?: "sg32" | "wg";
  /**
   * An online seat whose shape the online kernel does not admit: the route must fall back to the
   * three a8 dispatches with S in f32, and the run must be the a8 × "f32" Session's.
   */
  readonly fallsBack?: true;
  /** The attentionPermute count of the graph without the reshapes. */
  readonly absorbedPermutes: number;
  /** The census of the absorbed graph. */
  readonly absorbed: AttentionPermuteDispatches;
  /** The census of the reference (the permutes as nodes). */
  readonly reference: AttentionPermuteDispatches;
};

const WAN_LIKE: AttentionPermuteShape = { b: 2, h: 3, s: 150, n: 200, d: 128 };
/** N % 4 = 3: the online kernel reads a padded Vᵀ (`…:kpad`), the quantisation is `quantize_rows:…:pad`. */
const PADDED_KEYS: AttentionPermuteShape = { ...WAN_LIKE, n: 203 };

export const ATTENTION_PERMUTE_ROUTES: readonly AttentionPermuteRoute[] = [
  {
    // a8 + "none": q / k read token-major by the quantisation, Vᵀ by the tiled transpose, O stored
    // token-major by the online kernel; no strided copy is left. The reference: the three input
    // permutes, the strided Vᵀ, the output permute.
    name: "online (sg32, D 128)",
    options: { attentionCompute: "a8", attentionScoreStorage: "none" },
    shape: WAN_LIKE,
    reduce: "sg32",
    absorbedPermutes: 4,
    absorbed: { strided: 0, transpose: 1, quantizeBshd: 2, quantizeRows: 1, total: 5 },
    reference: { strided: 5, transpose: 0, quantizeBshd: 0, quantizeRows: 3, total: 9 },
  },
  {
    // The same at D 64.
    name: "online (sg32, D 64)",
    options: { attentionCompute: "a8", attentionScoreStorage: "none" },
    shape: { b: 2, h: 3, s: 150, n: 200, d: 64 },
    reduce: "sg32",
    absorbedPermutes: 4,
    absorbed: { strided: 0, transpose: 1, quantizeBshd: 2, quantizeRows: 1, total: 5 },
    reference: { strided: 5, transpose: 0, quantizeBshd: 0, quantizeRows: 3, total: 9 },
  },
  {
    // A device without subgroups: the online kernel's wg form stores O token-major the same way.
    name: "online (wg, D 128)",
    options: { attentionCompute: "a8", attentionScoreStorage: "none" },
    shape: WAN_LIKE,
    subgroups: false,
    reduce: "wg",
    absorbedPermutes: 4,
    absorbed: { strided: 0, transpose: 1, quantizeBshd: 2, quantizeRows: 1, total: 5 },
    reference: { strided: 5, transpose: 0, quantizeBshd: 0, quantizeRows: 3, total: 9 },
  },
  {
    // The padded key axis (N % 4 != 0) on the same route: the same dispatches, the padded
    // Vᵀ quantisation counted among quantizeRows, O stored token-major by the `…:obshd:kpad` kernel.
    name: "online (sg32, D 128, N 203)",
    options: { attentionCompute: "a8", attentionScoreStorage: "none" },
    shape: PADDED_KEYS,
    reduce: "sg32",
    absorbedPermutes: 4,
    absorbed: { strided: 0, transpose: 1, quantizeBshd: 2, quantizeRows: 1, total: 5 },
    reference: { strided: 5, transpose: 0, quantizeBshd: 0, quantizeRows: 3, total: 9 },
  },
  {
    // The same at D 64.
    name: "online (sg32, D 64, N 203)",
    options: { attentionCompute: "a8", attentionScoreStorage: "none" },
    shape: { ...PADDED_KEYS, d: 64 },
    reduce: "sg32",
    absorbedPermutes: 4,
    absorbed: { strided: 0, transpose: 1, quantizeBshd: 2, quantizeRows: 1, total: 5 },
    reference: { strided: 5, transpose: 0, quantizeBshd: 0, quantizeRows: 3, total: 9 },
  },
  {
    // The padded key axis on the wg form.
    name: "online (wg, D 128, N 203)",
    options: { attentionCompute: "a8", attentionScoreStorage: "none" },
    shape: PADDED_KEYS,
    subgroups: false,
    reduce: "wg",
    absorbedPermutes: 4,
    absorbed: { strided: 0, transpose: 1, quantizeBshd: 2, quantizeRows: 1, total: 5 },
    reference: { strided: 5, transpose: 0, quantizeBshd: 0, quantizeRows: 3, total: 9 },
  },
  {
    // a8 + S in f16 (the three dispatches): the same quantisation, ③PV writes O head-major for the
    // output permute node.
    name: "three-dispatch s16",
    options: { attentionCompute: "a8", attentionScoreStorage: "f16" },
    shape: WAN_LIKE,
    absorbedPermutes: 3,
    absorbed: { strided: 1, transpose: 1, quantizeBshd: 2, quantizeRows: 1, total: 8 },
    reference: { strided: 5, transpose: 0, quantizeBshd: 0, quantizeRows: 3, total: 11 },
  },
  {
    // a8 + "none" at D = 96 (not 64 / 128): the online kernel does not admit the shape, so the route
    // is the three a8 dispatches with S in f32 — the same dispatches as s16, O stored head-major and
    // its permute a node.
    name: "online seat falling back (D 96)",
    options: { attentionCompute: "a8", attentionScoreStorage: "none" },
    shape: { b: 2, h: 3, s: 150, n: 200, d: 96 },
    fallsBack: true,
    absorbedPermutes: 3,
    absorbed: { strided: 1, transpose: 1, quantizeBshd: 2, quantizeRows: 1, total: 8 },
    reference: { strided: 5, transpose: 0, quantizeBshd: 0, quantizeRows: 3, total: 11 },
  },
  {
    // a8 with D = 14 (D % 4 != 0): ①QK falls back to f32 and reads head-major copies of q / k,
    // ③PV stays i8a8 and transposes the token-major V.
    name: "a8 with ①QK on f32 (D = 14)",
    options: { attentionCompute: "a8" },
    shape: { b: 2, h: 3, s: 37, n: 44, d: 14 },
    absorbedPermutes: 3,
    absorbed: { strided: 3, transpose: 1, quantizeBshd: 0, quantizeRows: 1, total: 8 },
    reference: { strided: 5, transpose: 0, quantizeBshd: 0, quantizeRows: 1, total: 9 },
  },
  {
    // a8 with N = 43 (N % 4 != 0): ③PV falls back to f32 and reads a head-major copy of V, ①QK
    // stays i8a8 and quantises the token-major q / k.
    name: "a8 with ③PV on f32 (N = 43)",
    options: { attentionCompute: "a8" },
    shape: { b: 2, h: 3, s: 37, n: 43, d: 16 },
    absorbedPermutes: 3,
    absorbed: { strided: 2, transpose: 0, quantizeBshd: 2, quantizeRows: 0, total: 7 },
    reference: { strided: 4, transpose: 0, quantizeBshd: 0, quantizeRows: 2, total: 9 },
  },
  {
    // The f32 seat absorbs nothing: both graphs run the four permutes as nodes.
    name: "f32 seat",
    options: {},
    shape: WAN_LIKE,
    absorbedPermutes: 0,
    absorbed: { strided: 4, transpose: 0, quantizeBshd: 0, quantizeRows: 0, total: 7 },
    reference: { strided: 4, transpose: 0, quantizeBshd: 0, quantizeRows: 0, total: 7 },
  },
];

/** The online reduction form the device's attention takes at a route's shape (undefined: none). */
export const attentionPermuteOnlineForm = (
  gpu: GpuContext,
  shape: AttentionPermuteShape,
): "sg32" | "wg" | undefined =>
  attentionRoute(
    { attentionCompute: "a8", attentionScoreStorage: "none" },
    shape.d,
    shape.n,
    attentionDeviceFacts(gpu),
  ).online;

const BASE_PART = "fusion-attention-permute";
const SG32_PART = "fusion-attention-permute-sg32";
const SG32_TITLE =
  "attentionPermute: the online attention (sg32) absorbing q / k / v and O vs permutes as nodes";

/** The routes any device takes, and the online routes in the sg32 form. */
const BASE_ROUTES = ATTENTION_PERMUTE_ROUTES.filter(({ reduce }) => reduce === undefined);
const SG32_ROUTES = ATTENTION_PERMUTE_ROUTES.filter(({ reduce }) => reduce === "sg32");

/**
 * The cases of {@link checkAttentionPermuteTwins} (the s16 route, the online seat falling back, the
 * two a8 shapes with a stage on f32, the f32 seat) and of {@link checkAttentionPermuteSg32Twins}
 * (the online sg32 form at D 128 / 64 × N 200 / 203).
 */
export const ATTENTION_PERMUTE_TWIN_CASES = {
  [BASE_PART]: 5,
  [SG32_PART]: 4,
} as const satisfies ExpectedCases;

/**
 * What the device lacks for the sg32 part (its online attention takes another form at a route's
 * shape — no subgroups, a lane mapping that is not linear, or too little workgroup storage), or
 * undefined.
 */
export const attentionPermuteSg32Shortfall = (gpu: GpuContext): string | undefined => {
  const other = SG32_ROUTES.map(
    (route) => [route, attentionPermuteOnlineForm(gpu, route.shape)] as const,
  )
    .filter(([, form]) => form !== "sg32")
    .map(([route, form]) => `${route.name}: ${form ?? "no online form"}`);
  return other.length === 0
    ? undefined
    : `the online attention takes another form than sg32 on this device (${other.join("; ")}); sg32 needs subgroups, subgroup-size-control, WGSL subgroup_id and a linear lane mapping`;
};

const runRoutes = async (
  gpu: GpuContext,
  part: string,
  routes: readonly AttentionPermuteRoute[],
  options: CheckOptions,
): Promise<CheckPart["cases"]> => {
  const sink = twinCases(part, routes.length, options);
  for (const route of routes) {
    sink.next();
    await withTwinSessions(
      gpu,
      route.name,
      {
        model: await openGraphModel(permutedAttentionGraph(route.shape, false)),
        options: route.options,
      },
      {
        model: await openGraphModel(permutedAttentionGraph(route.shape, true)),
        options: route.options,
      },
      async ({ fused, plain }) => {
        sink.add(
          await checkFusionTwin(gpu, {
            label: `${route.name}: absorbed vs reference`,
            fused,
            plain,
            feeds: attentionPermuteFeeds(route.shape),
            counter: "attentionPermute",
            hits: route.absorbedPermutes,
            nan: "payload",
          }),
        );
      },
    );
  }
  return sink.cases;
};

/**
 * The routes any device takes: the absorbed graph against the interposed one (checkFusionTwin, NaN
 * by payload), attentionPermute at the route's count and 0.
 */
export const checkAttentionPermuteTwins = async (
  gpu: GpuContext,
  options: CheckOptions = {},
): Promise<CheckPart> => ({
  id: BASE_PART,
  title: "attentionPermute: permutes absorbed by the a8 attention vs permutes as nodes",
  cases: await runRoutes(gpu, BASE_PART, BASE_ROUTES, options),
  requirements: [],
});

/**
 * The online routes in the sg32 form, as {@link checkAttentionPermuteTwins}. On a device whose
 * online attention takes another form the part is not run ({@link attentionPermuteSg32Shortfall} in
 * `notRun`).
 */
export const checkAttentionPermuteSg32Twins = async (
  gpu: GpuContext,
  options: CheckOptions = {},
): Promise<CheckPart> => {
  const shortfall = attentionPermuteSg32Shortfall(gpu);
  if (shortfall !== undefined) return notRunPart(SG32_PART, SG32_TITLE, shortfall);
  return {
    id: SG32_PART,
    title: SG32_TITLE,
    cases: await runRoutes(gpu, SG32_PART, SG32_ROUTES, options),
    requirements: [],
  };
};
