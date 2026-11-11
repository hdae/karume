/**
 * The `geluMulStaticQuantize` fusion (perf-ledger K-48 step 1b): a Session with
 * `fuseGeluMulStaticQuantize` on against the same graph without it on the same backend, on values
 * landing on SRQ boundaries after `gelu_tanh(gate) · ple`, ±0, ±inf, NaN payloads, subnormals and
 * several scales including 0 (the identity SRQ). The SRQ words and the gate's later consumer are
 * u32-identical with NaN by payload, geluMulStaticQuantize at 1 and 0, the fused side in 3 dispatches
 * with the QAT slice operand and 2 with a plain input operand.
 *
 * Each chain opens a third Session, the probe — the unfused chain with the gelu value and the
 * product as graph outputs — whose gelu values tune the operand onto the SRQ boundaries.
 */

import type { GpuContext } from "../../../runtime/src/gpu/device.ts";
import type { GeluMulOrder } from "../../../runtime/src/kernels/gelu-mul-static-quantize.ts";
import { staticQuantizeParams } from "../../../runtime/src/kernels/static-quantize.ts";
import type { Session } from "../../../runtime/src/runtime/executor.ts";
import {
  geluMulStaticQuantizeGraph,
  type GeluMulStaticQuantizeGraphOptions,
  PLE_LAYERS,
} from "../../../runtime/tests/helpers/gelu-mul-static-quantize-graph.ts";
import { openGraphModel } from "../../../runtime/tests/helpers/model-fixture.ts";
import { checkFusionTwin } from "../fusion-twin.ts";
import {
  type CheckOptions,
  type CheckPart,
  type ExpectedCases,
  f32Word,
  scopedGpuWork,
} from "../result.ts";
import { openTwinSession, twinCases } from "./session.ts";

export const GELU_MUL_DIM = 256;

/** The f32 value of a u32 word. */
const valueOf = (bits: number): number => new Float32Array(Uint32Array.of(bits).buffer)[0];

/** A seeded xorshift32 — the data is fixed by the seed (a failure reproduces). */
const generator = (seed: number) => {
  let state = seed >>> 0 || 1;
  const bits = (): number => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state;
  };
  const uniform = (): number => (bits() + 0.5) / 2 ** 32;
  const gauss = (): number =>
    Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform());
  return { bits, uniform, gauss };
};

/** ±0, ±inf, quiet and signalling NaN payloads of both signs, subnormals, the f32 extremes, 1. */
const SPECIALS = [
  0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00000, 0x7f800001, 0xffc00001, 0x7fa5a5a5, 1,
  0x80000001, 0x007fffff, 0x807fffff, 0x7f7fffff, 0xff7fffff, 0x00800000, 0x80800000, 0x3f800000,
];

/** The scales: QAT-sized, small, tiny, large, the smallest normal, and 0 (identity). */
const SCALES = [0.0371, 0.001, 1e-30, 3.3, 0.5, 123.25, 2 ** -126, 0].map(Math.fround);

/** The SRQ boundary words of `scale` ±2 ulp, both signs (finite only). Empty for the identity. */
const boundaryTargets = (scale: number): readonly number[] => {
  const table = staticQuantizeParams(GELU_MUL_DIM, scale);
  if (table[258] !== 0) return [];
  const targets: number[] = [];
  for (let level = 1; level <= 128; level += 1) {
    for (let ulp = -2; ulp <= 2; ulp += 1) {
      for (const sign of [0, 0x80000000]) {
        const bits = ((table[level] + ulp) | sign) >>> 0;
        if ((bits & 0x7fffffff) < 0x7f800000) targets.push(bits);
      }
    }
  }
  return targets;
};

/**
 * The f32 `w` with `fround(act · w)` = the target bits, stepping `w` by ulps from `target / act`
 * (an f32 multiply is correctly rounded), or `undefined`.
 */
const tune = (act: number, targetBits: number): number | undefined => {
  const target = valueOf(targetBits);
  if (!Number.isFinite(act) || act === 0) return undefined;
  let w = Math.fround(target / act);
  for (let step = 0; step < 16 && Number.isFinite(w) && w !== 0; step += 1) {
    const product = Math.fround(act * w);
    if (f32Word(product) === targetBits) return w;
    const bits = f32Word(w);
    w = valueOf((Math.abs(product) < Math.abs(target) ? bits + 1 : bits - 1) >>> 0);
  }
  return undefined;
};

/** The gate: normal, small, large (gelu saturation) and raw-bit values; specials at the end. */
export const geluMulGateWords = (n: number, seed: number): Uint32Array<ArrayBuffer> => {
  const random = generator(seed);
  const gate = new Uint32Array(n);
  for (let i = 0; i < n; i += 1) {
    const kind = i % 6;
    gate[i] = kind === 4 ? random.bits() : f32Word(random.gauss() * [1, 0.01, 5, 12, 0, 3][kind]);
  }
  SPECIALS.forEach((bits, k) => (gate[n - SPECIALS.length + k] = bits));
  return gate;
};

export type GeluMulChain = {
  readonly options: GeluMulStaticQuantizeGraphOptions;
  /** The unfused chain with the gelu value and the product as graph outputs. */
  readonly probe: Session;
  readonly reference: Session;
  readonly fused: Session;
};

/** Opens the probe, the reference (the option off) and the fused Session, under the error scopes. */
export const openGeluMulChain = async (
  gpu: GpuContext,
  options: GeluMulStaticQuantizeGraphOptions,
): Promise<GeluMulChain> => {
  const where = `gelu→mul→SRQ rows=${options.rows ?? 4} scale=${options.scale}`;
  const chain = await openGraphModel(geluMulStaticQuantizeGraph(options));
  const opened: Session[] = [];
  try {
    const probe = await openTwinSession(
      gpu,
      `${where} probe`,
      await openGraphModel(
        geluMulStaticQuantizeGraph({ ...options, geluOutput: true, mulOutput: true }),
      ),
    );
    opened.push(probe);
    const reference = await openTwinSession(gpu, `${where} reference`, chain);
    opened.push(reference);
    const fused = await openTwinSession(gpu, `${where} fused`, chain, {
      fuseGeluMulStaticQuantize: true,
    });
    return { options, probe, reference, fused };
  } catch (error) {
    for (const session of opened) await session.dispose();
    throw error;
  }
};

export const disposeGeluMulChain = async (chain: GeluMulChain): Promise<void> => {
  await chain.probe.dispose();
  await chain.reference.dispose();
  await chain.fused.dispose();
};

export const geluMulF32Tensor = (shape: readonly number[], data: Uint32Array<ArrayBuffer>) => ({
  dtype: "f32" as const,
  shape,
  data: new Float32Array(data.buffer, data.byteOffset, data.length),
});

/** The graph inputs carrying `ple` in the chain's operand form (the slice form: layer 1 of `pli`). */
export const geluMulInputs = (
  options: GeluMulStaticQuantizeGraphOptions,
  gate: Uint32Array<ArrayBuffer>,
  ple: Uint32Array<ArrayBuffer>,
  seed: number,
): Readonly<Record<string, ReturnType<typeof geluMulF32Tensor>>> => {
  const rows = options.rows ?? 4;
  const shape = [1, rows, GELU_MUL_DIM];
  if ((options.other ?? "slice") === "input") {
    return { gate: geluMulF32Tensor(shape, gate), ple: geluMulF32Tensor(shape, ple) };
  }
  const random = generator(seed);
  const pli = new Uint32Array(rows * PLE_LAYERS * GELU_MUL_DIM);
  for (let i = 0; i < pli.length; i += 1) pli[i] = f32Word(random.gauss());
  for (let row = 0; row < rows; row += 1) {
    pli.set(
      ple.subarray(row * GELU_MUL_DIM, (row + 1) * GELU_MUL_DIM),
      (row * PLE_LAYERS + 1) * GELU_MUL_DIM,
    );
  }
  return {
    gate: geluMulF32Tensor(shape, gate),
    pli: geluMulF32Tensor([1, rows, PLE_LAYERS, GELU_MUL_DIM], pli),
  };
};

/**
 * The operand values for `gate`: every other element tuned so the product lands on an SRQ boundary
 * ±2 ulp (from the probe's gelu value), the rest random bits or normal values, specials at the end.
 */
export const geluMulPleWords = async (
  chain: GeluMulChain,
  gate: Uint32Array<ArrayBuffer>,
  scale: number,
  seed: number,
): Promise<{ readonly ple: Uint32Array<ArrayBuffer>; readonly targeted: Map<number, number> }> => {
  const n = gate.length;
  const random = generator(seed ^ 0x9e3779b9);
  const ones = new Uint32Array(n).fill(f32Word(1));
  const act = new Float32Array(
    (await chain.probe.run(geluMulInputs(chain.options, gate, ones, seed))).g.data,
  );
  const targets = boundaryTargets(scale);
  const ple = new Uint32Array(n);
  const targeted = new Map<number, number>();
  let next = 0;
  for (let i = 0; i < n - 2 * SPECIALS.length; i += 1) {
    let tuned: number | undefined;
    for (
      let attempt = 0;
      i % 2 === 0 && targets.length > 0 && attempt < 8 && tuned === undefined;
      attempt += 1
    ) {
      const target = targets[next % targets.length];
      next += 7;
      tuned = tune(act[i], target);
      if (tuned !== undefined) targeted.set(i, target);
    }
    ple[i] =
      tuned !== undefined
        ? f32Word(tuned)
        : random.uniform() < 0.3
          ? random.bits()
          : f32Word(random.gauss() * 2);
  }
  for (let i = n - 2 * SPECIALS.length; i < n; i += 1) ple[i] = SPECIALS[i % SPECIALS.length];
  return { ple, targeted };
};

/** One case's inputs (the gate, the tuned operand) and the indices tuned onto a boundary. */
export const geluMulCase = async (
  chain: GeluMulChain,
  scale: number,
  seed: number,
): Promise<{
  readonly inputs: ReturnType<typeof geluMulInputs>;
  readonly targeted: Map<number, number>;
}> => {
  const gate = geluMulGateWords((chain.options.rows ?? 4) * GELU_MUL_DIM, seed);
  const { ple, targeted } = await geluMulPleWords(chain, gate, scale, seed);
  return { inputs: geluMulInputs(chain.options, gate, ple, seed), targeted };
};

/** The cases of one chain: its options, its scale and each case's seed and label. */
export type GeluMulGroup = {
  readonly options: GeluMulStaticQuantizeGraphOptions;
  readonly scale: number;
  readonly cases: readonly { readonly seed: number; readonly label: string }[];
};

const FORMS: readonly (readonly [GeluMulOrder, "slice" | "input"])[] = [
  ["other-gelu", "slice"],
  ["other-gelu", "input"],
  ["gelu-other", "input"],
];

/**
 * Rows 1 / 4 / 8 at every scale with two seeds each (the QAT operand form), then the node's operand
 * order and a plain input operand at rows 8 with a QAT scale and the identity.
 */
export const GELU_MUL_GROUPS: readonly GeluMulGroup[] = [
  ...[1, 4, 8].flatMap((rows) =>
    SCALES.map((scale, index) => ({
      options: { rows, scale },
      scale,
      cases: [11, 29].map((seed) => ({
        seed: seed + index + rows * 101,
        label: `rows ${rows} / scale ${scale} / seed ${seed}`,
      })),
    })),
  ),
  ...FORMS.flatMap(([order, other]) =>
    [Math.fround(0.0371), 0].map((scale) => ({
      options: { rows: 8, order, other, scale },
      scale,
      cases: [{ seed: 7, label: `${order} / ${other} / scale ${scale}` }],
    })),
  ),
];

const PART = "fusion-gelu-mul-static-quantize";

/** The cases of {@link checkGeluMulStaticQuantizeTwins}: 3 rows × 8 scales × 2 seeds; 3 forms × 2 scales. */
export const GELU_MUL_STATIC_QUANTIZE_TWIN_CASES = { [PART]: 54 } as const satisfies ExpectedCases;

/**
 * Every chain and seed: the Session with `fuseGeluMulStaticQuantize` against the one without it
 * (checkFusionTwin, NaN by payload) on the SRQ words (y) and the gate after the fold (copy),
 * geluMulStaticQuantize at 1 and 0, the fused side's dispatches.
 */
export const checkGeluMulStaticQuantizeTwins = async (
  gpu: GpuContext,
  options: CheckOptions = {},
): Promise<CheckPart> => {
  const sink = twinCases(PART, GELU_MUL_STATIC_QUANTIZE_TWIN_CASES[PART], options);
  for (const group of GELU_MUL_GROUPS) {
    sink.next();
    const chain = await openGeluMulChain(gpu, group.options);
    try {
      for (const { seed, label } of group.cases) {
        sink.next();
        const { inputs } = await scopedGpuWork(gpu, `${label}: inputs`, () =>
          geluMulCase(chain, group.scale, seed),
        );
        // gelu_tanh + mul + static_quantize + neg (+ slice) → fused + neg (+ slice).
        const fusedDispatches = (group.options.other ?? "slice") === "slice" ? 3 : 2;
        sink.add(
          await checkFusionTwin(gpu, {
            label,
            fused: chain.fused,
            plain: chain.reference,
            feeds: inputs,
            counter: "geluMulStaticQuantize",
            hits: 1,
            nan: "payload",
            dispatches: { fused: fusedDispatches },
          }),
        );
      }
    } finally {
      await disposeGeluMulChain(chain);
    }
  }
  return {
    id: PART,
    title: "geluMulStaticQuantize: fused vs gelu_tanh → mul → static_quantize (SRQ boundaries)",
    cases: sink.cases,
    requirements: [],
  };
};
