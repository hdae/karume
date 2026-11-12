/**
 * The geometry sweep page's settings: the form (op families, cases, candidate set, rounds, idle
 * seconds between cases), its plan (the case list `runGeometrySweep` is called with, one case per
 * call) and the presets. Pure, so the case selection is tested without a GPU.
 *
 * The case list depends on the candidate set: the evaluation sets `k85` and `k88` sweep tables of
 * their own (`geometrySweepCases("k85")`, `geometrySweepCases("k88")`). Every function that takes
 * `cases` takes the list of the form's set (of the preset's set for {@link presetForm}).
 */

import type {
  GeometrySweepCandidateSet,
  GeometrySweepCase,
  GeometrySweepOp,
} from "@karume/runtime/tune";

/**
 * The candidate sets in growing size (the runtime's default is `quick+`), then the evaluation sets
 * `k85` and `k88`.
 */
export const CANDIDATE_SETS = [
  "quick",
  "quick+",
  "full",
  "k85",
  "k88",
] as const satisfies readonly GeometrySweepCandidateSet[];

export type SweepForm = {
  /** The op families shown and swept. */
  readonly ops: ReadonlySet<GeometrySweepOp>;
  /** The ticked cases; a ticked case of an op that is not chosen is not swept. */
  readonly cases: ReadonlySet<string>;
  readonly candidateSet: GeometrySweepCandidateSet;
  /** Text of the rounds input. */
  readonly rounds: string;
  /** Text of the idle-seconds input. */
  readonly idleSeconds: string;
};

/** A checked form: what one sweep runs. */
export type SweepPlan = {
  /** The cases in the table's order — one `runGeometrySweep` call each. */
  readonly caseIds: readonly string[];
  /** The op families of those cases, in the table's order (`ops` of every call). */
  readonly ops: readonly GeometrySweepOp[];
  readonly candidateSet: GeometrySweepCandidateSet;
  readonly rounds: number;
  /** Idle time before every case but the first (cooling). */
  readonly idleSeconds: number;
};

export type SweepPreset = {
  readonly id: string;
  readonly label: string;
  readonly hint: string;
  readonly ops: readonly GeometrySweepOp[];
  readonly candidateSet: GeometrySweepCandidateSet;
  readonly rounds: number;
  readonly idleSeconds: number;
};

export const SWEEP_PRESETS: readonly SweepPreset[] = [
  {
    id: "v15-conv3d",
    label: "V-15: conv3d, full, 5 rounds, 90 s idle",
    hint: "Every conv3d w8a8 case with every i8a8 geometry of the full set, cooling 90 s between cases: the record from which the derivation chooses i8a8.conv3d.",
    ops: ["i8a8-conv3d"],
    candidateSet: "full",
    rounds: 5,
    idleSeconds: 90,
  },
  {
    id: "k85-linear",
    label: "K-85 (i8a8 linear skeleton)",
    hint: "The k85 set — this adapter's table geometry, the default and 204 grid geometries with and without the vec4 skeleton — on the Wan2.2 TI2V-5B DiT linears at 33 frames and a small-M guard, 5 rounds, 90 s idle: per case the best skeleton against the table geometry, and the sum over one Wan2.2 33f DiT forward. No table is derived from it.",
    ops: ["i8a8-linear"],
    candidateSet: "k85",
    rounds: 5,
    idleSeconds: 90,
  },
  {
    id: "k88-forms",
    label: "K-88 (GEMM forms / conv2d decode)",
    hint: "The k88 set — per case the geometry the default and the registered tables give it, without the fields, with each skeleton form and, for conv2d, with the counter decode alone and with each form — on the gemma4 E2B prefill, umT5, siglip2 and guard linears, the square matmul and the Wan2.2 / anima VAE conv2d, 5 rounds, 90 s idle: per case the best form, the decode and both against the table geometry, and the sums over one gemma4 prefill chunk, one Wan2.2 VAE next chunk and the other units. No table is derived from it.",
    ops: ["linear", "matmul", "conv2d"],
    candidateSet: "k88",
    rounds: 5,
    idleSeconds: 90,
  },
];

/** The op families of the table, in its order (each once). */
export const sweepOps = (cases: readonly GeometrySweepCase[]): GeometrySweepOp[] => [
  ...new Set(cases.map(({ op }) => op)),
];

/** Every case of the table and every op ticked; the runtime's candidate set; 5 rounds; 90 s idle. */
export const initialForm = (cases: readonly GeometrySweepCase[]): SweepForm => ({
  ops: new Set(sweepOps(cases)),
  cases: new Set(cases.map(({ id }) => id)),
  candidateSet: "quick+",
  rounds: "5",
  idleSeconds: "90",
});

/** The form a preset fills: its ops with every case of them ticked. */
export const presetForm = (
  preset: SweepPreset,
  cases: readonly GeometrySweepCase[],
): SweepForm => ({
  ops: new Set(preset.ops),
  cases: new Set(cases.filter(({ op }) => preset.ops.includes(op)).map(({ id }) => id)),
  candidateSet: preset.candidateSet,
  rounds: String(preset.rounds),
  idleSeconds: String(preset.idleSeconds),
});

/**
 * The form after the candidate set changes to `set`. When the new set sweeps another case table
 * (`casesOf` — `geometrySweepCases`), every case and op of that table is ticked (the old ticks name
 * cases the new table does not have); otherwise only the set changes.
 */
export const changeCandidateSet = (
  form: SweepForm,
  set: GeometrySweepCandidateSet,
  casesOf: (set: GeometrySweepCandidateSet) => readonly GeometrySweepCase[],
): SweepForm => {
  const ids = (cases: readonly GeometrySweepCase[]): string => cases.map(({ id }) => id).join("\n");
  const cases = casesOf(set);
  if (ids(casesOf(form.candidateSet)) === ids(cases)) return { ...form, candidateSet: set };
  return {
    ...form,
    candidateSet: set,
    ops: new Set(sweepOps(cases)),
    cases: new Set(cases.map(({ id }) => id)),
  };
};

/** The text as an integer ≥ `min`, else undefined. */
const integerAtLeast = (text: string, min: number): number | undefined => {
  if (text.trim() === "") return undefined;
  const value = Number(text);
  return Number.isSafeInteger(value) && value >= min ? value : undefined;
};

/** The plan of a form, or why it cannot run. */
export const planSweep = (
  form: SweepForm,
  cases: readonly GeometrySweepCase[],
): { readonly plan: SweepPlan } | { readonly error: string } => {
  const rounds = integerAtLeast(form.rounds, 1);
  if (rounds === undefined) return { error: "Rounds must be an integer of at least 1" };
  const idleSeconds = integerAtLeast(form.idleSeconds, 0);
  if (idleSeconds === undefined) return { error: "Idle seconds must be an integer of at least 0" };
  const chosen = cases.filter(({ id, op }) => form.ops.has(op) && form.cases.has(id));
  if (chosen.length === 0) return { error: "Choose at least one case" };
  return {
    plan: {
      caseIds: chosen.map(({ id }) => id),
      ops: sweepOps(chosen),
      candidateSet: form.candidateSet,
      rounds,
      idleSeconds,
    },
  };
};

/** The preset whose plan the form's plan equals, if any (worked out on every render). */
export const matchingPreset = (
  form: SweepForm,
  cases: readonly GeometrySweepCase[],
  presets: readonly SweepPreset[] = SWEEP_PRESETS,
): SweepPreset | undefined => {
  const planned = planSweep(form, cases);
  if (!("plan" in planned)) return undefined;
  const { plan } = planned;
  return presets.find((preset) => {
    const presetPlan = planSweep(presetForm(preset, cases), cases);
    return (
      "plan" in presetPlan &&
      presetPlan.plan.caseIds.join("\n") === plan.caseIds.join("\n") &&
      presetPlan.plan.candidateSet === plan.candidateSet &&
      presetPlan.plan.rounds === plan.rounds &&
      presetPlan.plan.idleSeconds === plan.idleSeconds
    );
  });
};
