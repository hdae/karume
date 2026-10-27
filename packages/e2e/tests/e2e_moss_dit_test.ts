/**
 * The MOSS-Image DiT stage on a real GPU — the port check of the reference path (i8 weights × f32
 * compute, no session knobs) against the layer-by-layer CPU f64 reference, the Session lifetimes
 * and the plan backing (ADR NNNN decisions 3–5 and 11, plan step 2).
 *
 * The goldens are written by `tools/export-recipes/moss/dit_reference.py` from the same series
 * containers (the dequantised i8 weights the GPU reads — ADR 0121 decision 7), one golden per case:
 * the CPU f32 output (the denominator) and the CPU f64 output (the reference) of one graph on the
 * inputs the host functions build. The f64 output is stored as two f32 parts whose double sum is
 * the f64 value (`output.f64` + `output.f64.lo` — the f32 rounding alone would be as large as the
 * f32 run's own error where that error is 1–2 ULP). The tests refuse goldens of other containers
 * (the part-0 sha256 recorded in each golden) and of another case than the table names.
 *
 * ## Metric and bands (ADR 0118 decision 8, ADR 0120 decision 3)
 *
 *   r = (max|GPU − f64| ÷ max|f64|) ÷ (max|CPU f32 − f64| ÷ max|f64|)
 *
 * — how many times the CPU f32 run's own error the GPU's error is, on the same inputs. Per group
 * (`dit_text`; `dit` / `dit_skip22` at 256²; at 1024²) the six band cases set the band as their
 * worst r × 5, rounded up to two significant digits, and the four acceptance cases are judged
 * against it; the groups' bands are derived independently (the small band is not carried to full
 * size). `undefined` = not derived: every case reports its r and fails, and the group's last step
 * prints the candidate. MUST: the bands, the band cases and the metric are not changed after seeing
 * the acceptance results; an acceptance case outside the band is investigated, not absorbed.
 *
 * ### A second metric for `dit_text`
 *
 * `dit_text`'s output has one outlier channel (max |value| about 1,800 against a median channel
 * maximum of about 14). Both maxima of r then sit in that channel, and the band (45 × the f32
 * error there) allows an error of about 4e-4 of a typical channel's scale anywhere else, so a small
 * defect outside the outlier would not show. The per-channel r divides each channel's error by
 * that channel's scale before taking the maxima:
 *
 *   r_ch = max_{t,c} (|GPU − f64| ÷ s_c) ÷ max_{t,c} (|CPU f32 − f64| ÷ s_c),
 *   s_c = max(max_t |f64_{t,c}|, the median over channels of max_t |f64_{t,c}|)
 *
 * — the floor keeps a channel whose values are all near zero (T = 1 has one value per channel)
 * from turning rounding noise into the maximum. Its band is derived the same way from the same six
 * cases, and both bands must hold.
 *
 * ## Fault injections (on the acceptance cases — each must leave the band by a margin of 2)
 *
 * The margin is ADR 0118's (addendum 2026-10-02: a fault is found outside the band by at least 2×).
 * A RoPE fault must move positions **non-uniformly**: attention scores under RoPE depend on
 * relative positions only, so shifting every token by the same amount changes nothing.
 *
 * - DiT pair: the time row of the next training-grid index (index + 1 — the neighbouring timestep;
 *   both branches) and the image tokens one position further from the text (row T of the `f` table
 *   reads row T + 1, text rows unchanged — the bug the T-dependence of the image positions invites;
 *   the positive branch, run alone).
 * - `dit_text`: text token i at position 2i, the last text token one position further (row T − 1
 *   reads row T — the smallest position fault) and the last embedding row replaced by the one
 *   before it (a layout off by one), on the acceptance cases with T ≥ 2 — a single token has no
 *   relative position and no second row, so the T = 1 case is checked against the bands only.
 *
 * ## The product path
 *
 * On the acceptance cases the DiT pair also runs on the text context `dit_text` produced on the GPU
 * from the stage-0 embeddings (the stage order of a generation), and must stay inside the band.
 *
 * ## Default run and opt-in upstream parity (maintainer ruling 2026-10-17 — ADR 0129 addendum (2))
 *
 * The r gates and their fault injections compare with a torch reference, so they run only with
 * `KARUME_UPSTREAM_PARITY=1` (parity is proven when the family is implemented). The default run
 * keeps the checks that read no reference: `dit_text` at T = 1 (the graph was traced with T ≥ 2,
 * and nothing but the GPU runs the IR at T = 1 — plan step 2 MUST), finite and pinned by a sha256
 * row (ADR 0106 — `fixtures/references/moss.json`; the r gate is opt-in, so the row is the default
 * lane's only check of its values — plan step 4; MUST: with `KARUME_REFERENCE` the row is written
 * only after the r gate of the text cases, T = 1 among them, passed earlier in the same run — that
 * gate is declared before the default run (Vitest runs a file's tests in declaration order), and
 * otherwise the case fails before writing, naming the gate that did not pass,
 * `assertMossRowWritable`; compare mode reads no gate); one
 * plan backing per Session
 * over several predictions, both for a backing inside the plan backing budget (256²) and for one
 * over it (768² — the runtime keeps such a backing alone); the borrower holding no weight of its
 * own; finite outputs; and the lifetimes — `dit_text` closed before the pair opens, everything
 * closed when the body or the borrower's opening throws, nothing opened when the headroom check
 * refuses.
 *
 * Without a GPU adapter, the series, the stage-0 embeddings or the goldens, the groups skip
 * explicitly with the command that writes what they lack (ADR 0005).
 */

import { afterAll, beforeAll, describe, expect, it } from "vite-plus/test";
import { assert, assertEquals, assertRejects } from "@std/assert";
import {
  DEFAULT_PLAN_BACKING_BUDGET_BYTES,
  type GpuContext,
  parseSafetensors,
  type Session,
  type SessionDiagnostics,
} from "@karume/runtime";
import type { ModelComponent } from "../../models/src/hub/components.ts";
import {
  mossDitContract,
  type MossDitContract,
  mossDitGraphContract,
  readMossDitAssets,
} from "../../models/src/moss/dit-contract.ts";
import type { MossRopeBase } from "../../models/src/moss/dit-rope.ts";
import {
  type MossBranches,
  type MossDitComponents,
  type MossDitPlan,
  type MossDitStageState,
  type MossRefinedText,
  planMossDit,
  runMossTextStage,
  withMossDitPair,
} from "../../models/src/moss/dit-stage.ts";
import type { MossTextEmbeddings } from "../../models/src/moss/request.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { verifyRunningGpu } from "../../runtime/tests/helpers/environment.ts";
import {
  f32ArtifactBytes,
  openReferences,
  type ReferenceSettlement,
  referenceMismatchMessage,
  registerReferenceGate,
  settleOrObserve,
} from "../../runtime/tests/helpers/reference.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";
import {
  containerDigests,
  filePresent,
  floatsOf,
  goldenUrl,
  loadCaseGolden,
  MOSS_DIT_CASES,
  MOSS_GENERATE_EMBEDS,
  MOSS_GENERATE_GOLDENS,
  MOSS_GENERATE_SERIES,
  MOSS_TEXT_CASES,
  assertMossRowWritable,
  MOSS_TEXT_EMBEDS,
  withResidentLog,
  type MossDitCase,
  type MossGolden,
  mossSeriesPresent,
  type MossTextCase,
  openMossSeries,
  readBuffer,
} from "../../models/tests/helpers/moss-dit.ts";
import { settleReleases } from "../../models/tests/helpers/settle-releases.ts";
import { UPSTREAM_PARITY, UPSTREAM_PARITY_OPT_IN } from "./helpers/upstream-parity.ts";

/**
 * The `dit_text` band of r (six band cases). `undefined` = not derived (module doc).
 *
 * Derivation (2026-10-27, RTX 3080 Ti, Node + Dawn, goldens of writer version 2 — the f64 output
 * in two parts): band-case r 0.748 (T = 2), 6.59 (T = 10), 2.80 (T = 31), 11.1 (T = 62), 7.42
 * (T = 63), 5.75 (T = 134); worst 11.1 × 5 = 55.5 → 56. The denominator is still the CPU f32
 * run's 1–2 ULP at the outlier channel (4e-8 – 3e-7 relative), hence the spread; the per-channel
 * band below is the steadier of the two. Acceptance r on the same run: 1.15 (T = 1), 9.75 (T = 7),
 * 3.64 (T = 45), 20.4 (T = 100).
 *
 * MUST: not changed after seeing the acceptance results.
 */
const TEXT_BAND: number | undefined = 56;

/**
 * The `dit_text` band of the per-channel r (module doc "A second metric for `dit_text`"; the same
 * six band cases). `undefined` = not derived.
 *
 * Derivation (same run): band-case r 2.35 (T = 2), 3.58 (T = 10), 5.19 (T = 31), 6.79 (T = 62),
 * 6.49 (T = 63), 7.23 (T = 134); worst 7.23 × 5 = 36.2 → 37. Acceptance r: 1.93 (T = 1), 3.23
 * (T = 7), 4.52 (T = 45), 5.06 (T = 100). The smallest fault, the last text token one position
 * further, lands at 184–2,500 × this band (58–590 × the band of r).
 *
 * MUST: not changed after seeing the acceptance results.
 */
const TEXT_CHANNEL_BAND: number | undefined = 37;

/**
 * The 256² band of `dit` / `dit_skip22` (six band cases). `undefined` = not derived.
 *
 * Derivation (2026-10-27, RTX 3080 Ti, Node + Dawn, goldens of writer version 2): band-case r —
 * positive 2.76 (index 0), 1.95 (400), 4.26 (850); negative 3.22, 1.51, 2.19; worst 4.26 × 5 =
 * 21.3 → 22. Acceptance r on the same run: positive 0.847 (index 200), 3.00 (950); negative 1.98,
 * 1.56.
 *
 * MUST: not changed after seeing the acceptance results.
 */
const SMALL_BAND: number | undefined = 22;

/**
 * The 1024² band of `dit` / `dit_skip22` (six band cases). `undefined` = not derived. Derived
 * independently of the 256² band (ADR 0118 decision 8 — no carrying from small S to full size).
 *
 * Derivation (2026-10-27, RTX 3080 Ti, Node + Dawn, goldens of writer version 2): band-case r —
 * positive 2.21 (index 0), 2.57 (400), 3.01 (850); negative 4.86, 1.78, 3.09; worst 4.86 × 5 =
 * 24.3 → 25. Acceptance r on the same run: positive 5.09 (index 200), 5.31 (950); negative 3.22,
 * 2.38.
 *
 * MUST: not changed after seeing the acceptance results.
 */
const FULL_BAND: number | undefined = 25;

/** How far outside its band a fault must land (module doc "Fault injections"). */
const FAULT_MARGIN = 2;

/** The sha256 rows (environment rows — ADR 0106; shared with `e2e_moss_generate_test.ts`). */
const references = openReferences(
  new URL("../../models/tests/fixtures/references/moss.json", import.meta.url),
);
/** Artifacts and verdicts (`outputs/verify/<environment key>/<date>_moss-dit/` — safe to delete). */
const results = openResults("moss-dit");

/** The row of `dit_text` at T = 1 (the reference path: i8 weights × f32 compute; fridge-note's first token). */
const TEXT_T1_CASE = "i8-dit_text-fridge-note-t1";

const SERIES_PRESENT = mossSeriesPresent();
const EMBEDS_PRESENT = filePresent(MOSS_TEXT_EMBEDS);
const present = (cases: readonly { readonly name: string }[]): boolean =>
  cases.every(({ name }) => filePresent(goldenUrl(name)));
const SMALL = MOSS_DIT_CASES.filter(({ group }) => group === "small");
/** The id of the text cases' r gate (T = 1 among them) — the T = 1 row's write needs it passed. */
const TEXT_PARITY_ID = "dit_text";
const FULL = MOSS_DIT_CASES.filter(({ group }) => group === "full");

const skipped = (ok: boolean, what: string, command: string): string =>
  ok ? "" : ` (SKIP: no ${what} — ${command})`;

/** The pixels per side of a DiT group. */
const SIDE = { small: 256, full: 1024 } as const;

/** The width of `dit_text`'s output rows (the DiT's `dim`). */
const TEXT_OUTPUT_WIDTH = 3072;

/** Max |a − b| (in f64) and the count of non-finite values of `a`. */
const difference = (
  actual: Float32Array,
  expected: Float64Array,
): { readonly maxAbs: number; readonly nonFinite: number } => {
  assertEquals(actual.length, expected.length, "element count");
  let maxAbs = 0;
  let nonFinite = 0;
  for (let index = 0; index < expected.length; index += 1) {
    if (!Number.isFinite(actual[index])) nonFinite += 1;
    maxAbs = Math.max(maxAbs, Math.abs(actual[index] - expected[index]));
  }
  return { maxAbs, nonFinite };
};

const maxAbsOf = (values: Float64Array): number =>
  values.reduce((max, value) => Math.max(max, Math.abs(value)), 0);

/** The golden's f64 output in double: its f32 rounding + the f32 rounding of the rest. */
const referenceOf = (golden: MossGolden, where: string): Float64Array<ArrayBuffer> => {
  const high = floatsOf(golden.file, "output.f64", where);
  const low = floatsOf(golden.file, "output.f64.lo", where);
  assertEquals(low.length, high.length, `${where}: output.f64.lo length`);
  return Float64Array.from(high, (value, index) => value + low[index]);
};

/** One output against its golden: r and its two ratios. */
type Ratio = {
  readonly r: number;
  readonly gpu: number;
  readonly reference: number;
  readonly nonFinite: number;
};

/**
 * r of `actual` against the golden's f64 output, normalised by its CPU f32 output (module doc). The
 * denominator is checked against the one the writer computed from the f64 tensor itself (relative
 * 1e-6 — the two parts hold the f64 value to 2⁻⁴⁸, so a larger gap means a broken golden).
 */
const ratioOf = (actual: Float32Array, golden: MossGolden, where: string): Ratio => {
  const f64 = referenceOf(golden, where);
  const f32 = floatsOf(golden.file, "output", where);
  const scale = maxAbsOf(f64);
  const reference = difference(f32, f64).maxAbs / scale;
  if (!(reference > 0)) throw new Error(`${where}: the CPU f32 output equals f64 (r undefined)`);
  const written = golden.metadata["f32_ratio"];
  if (typeof written !== "number" || !(Math.abs(reference / written - 1) <= 1e-6)) {
    throw new Error(
      `${where}: the CPU f32 ratio from the stored parts (${reference}) is not the writer's` +
        ` (${String(written)})`,
    );
  }
  const gpu = difference(actual, f64);
  return {
    r: gpu.maxAbs / scale / reference,
    gpu: gpu.maxAbs / scale,
    reference,
    nonFinite: gpu.nonFinite,
  };
};

/** The per-channel r of a `[T, width]` output (module doc "A second metric for `dit_text`"). */
const channelRatioOf = (
  actual: Float32Array,
  golden: MossGolden,
  where: string,
  width: number,
): Ratio => {
  const f64 = referenceOf(golden, where);
  const f32 = floatsOf(golden.file, "output", where);
  assertEquals(actual.length, f64.length, `${where}: element count`);
  const rows = f64.length / width;
  const channelMax = new Float64Array(width);
  for (let index = 0; index < f64.length; index += 1) {
    const channel = index % width;
    channelMax[channel] = Math.max(channelMax[channel], Math.abs(f64[index]));
  }
  const floor = Float64Array.from(channelMax).sort()[width >> 1];
  if (!(floor > 0)) throw new Error(`${where}: the median channel scale is ${floor}`);
  let gpu = 0;
  let reference = 0;
  let nonFinite = 0;
  for (let row = 0; row < rows; row += 1) {
    for (let channel = 0; channel < width; channel += 1) {
      const index = row * width + channel;
      const scale = Math.max(channelMax[channel], floor);
      if (!Number.isFinite(actual[index])) nonFinite += 1;
      gpu = Math.max(gpu, Math.abs(actual[index] - f64[index]) / scale);
      reference = Math.max(reference, Math.abs(f32[index] - f64[index]) / scale);
    }
  }
  if (!(reference > 0)) throw new Error(`${where}: the CPU f32 output equals f64 (r undefined)`);
  return { r: gpu / reference, gpu, reference, nonFinite };
};

const formatRatio = (label: string, ratio: Ratio): string =>
  `${label}: r ${ratio.r.toPrecision(3)} (GPU ${ratio.gpu.toExponential(2)} / CPU f32 ${ratio.reference.toExponential(2)})`;

/** Worst band r × 5, rounded up to two significant digits (the Wan rule). */
const bandCandidate = (worst: number): number => {
  const unit = 10 ** (Math.floor(Math.log10(worst * 5)) - 1);
  return Number((Math.ceil((worst * 5) / unit) * unit).toPrecision(2));
};

type Row = { readonly name: string; readonly role: string; readonly ratio: Ratio };
type Fault = { readonly name: string; readonly ratio: Ratio };

/** The faults that do not land outside the band by {@link FAULT_MARGIN} (module doc). */
const faultsInside = (faults: readonly Fault[], band: number): string[] =>
  faults
    .filter(({ ratio }) => !(ratio.r > FAULT_MARGIN * band))
    .map(
      ({ name, ratio }) =>
        `${name}: the fault's r ${ratio.r.toPrecision(3)} is not outside ${FAULT_MARGIN} × the band ${band}`,
    );

/**
 * Logs the cases' r values and returns what fails (empty = passed): band and acceptance cases
 * against the band, faults outside it by the margin. Returning instead of throwing lets every
 * metric of a group log its candidate in one run.
 */
const judge = (
  label: string,
  band: number | undefined,
  rows: readonly Row[],
  faults: readonly Fault[],
): string[] => {
  for (const { name, ratio } of rows)
    console.log(`[moss-dit] ${label} ${formatRatio(name, ratio)}`);
  for (const { name, ratio } of faults) {
    console.log(`[moss-dit] ${label} fault ${formatRatio(name, ratio)}`);
  }
  const failures = rows
    .filter(({ ratio }) => ratio.nonFinite !== 0)
    .map(({ name, ratio }) => `${name}: ${ratio.nonFinite} non-finite values`);
  const bandRows = rows.filter(({ role }) => role === "band");
  const worst = Math.max(...bandRows.map(({ ratio }) => ratio.r));
  console.log(
    `[moss-dit] ${label}: worst band r ${worst.toPrecision(3)} over ${bandRows.length} cases` +
      ` → candidate ${bandCandidate(worst)}` +
      (band === undefined ? "" : ` (band ${band})`),
  );
  if (band === undefined) {
    return [
      ...failures,
      `${label}: the band is not derived — worst band r ${worst.toPrecision(3)} × 5 →` +
        ` ${bandCandidate(worst)}; acceptance r: ${rows
          .filter(({ role }) => role === "accept")
          .map(({ name, ratio }) => `${name} ${ratio.r.toPrecision(3)}`)
          .join(" / ")}`,
    ];
  }
  return [
    ...failures,
    ...faultsInside(faults, band),
    ...rows
      .filter(({ ratio }) => !(ratio.r <= band))
      .map(({ name, ratio }) => `${formatRatio(name, ratio)} is outside the band ${band}`),
  ];
};

const assertJudged = (failures: readonly string[]): void => {
  assert(failures.length === 0, failures.join("\n"));
};

/** The base tables with the `f` rows remapped (`row(i)` = the source row of row i). */
const remappedF = (base: MossRopeBase, row: (index: number) => number): MossRopeBase => {
  const remap = (table: Float32Array<ArrayBuffer>): Float32Array<ArrayBuffer> => {
    const out = new Float32Array(table.length);
    for (let index = 0; index <= base.maxText; index += 1) {
      const from = row(index) * base.bands;
      out.set(table.subarray(from, from + base.bands), index * base.bands);
    }
    return out;
  };
  return {
    ...base,
    sin: { ...base.sin, f: remap(base.sin.f) },
    cos: { ...base.cos, f: remap(base.cos.f) },
  };
};

/** Text token i at position 2i (module doc "Fault injections"). */
const doubledPositions = (base: MossRopeBase): MossRopeBase =>
  remappedF(base, (index) => (2 * index <= base.maxText ? 2 * index : index));

/** The last of `tokens` text tokens one position further (row T − 1 reads row T). */
const lastTokenOneFurther = (base: MossRopeBase, tokens: number): MossRopeBase =>
  remappedF(base, (index) => (index === tokens - 1 ? tokens : index));

/** The image tokens of a text of `tokens` at f = T + 1, the text rows unchanged. */
const imageOneFurther = (base: MossRopeBase, tokens: number): MossRopeBase =>
  remappedF(base, (index) => (index === tokens ? tokens + 1 : index));

/** The stage-0 embeddings by prompt name (`moss.text_embeds`). */
const readEmbeddings = async (): Promise<ReadonlyMap<string, MossTextEmbeddings>> => {
  const file = parseSafetensors(await readBuffer(MOSS_TEXT_EMBEDS));
  return new Map(
    [...file.tensors.values()].map((view) => [
      view.name,
      { data: floatsOf(file, view.name, "text embeds").slice(), shape: view.shape },
    ]),
  );
};

const embeddingsOf = (
  all: ReadonlyMap<string, MossTextEmbeddings>,
  prompt: string,
  tokens?: number,
) => {
  const found = all.get(prompt);
  if (found === undefined) throw new Error(`no stage-0 embedding '${prompt}'`);
  const length = tokens ?? found.shape[0];
  return { data: found.data.slice(0, length * 4096), shape: [length, 4096] };
};

/** The latents of a DiT golden as the host's `[C, H, W]`. */
const latentsOf = (golden: MossGolden, where: string): Float32Array<ArrayBuffer> =>
  floatsOf(golden.file, "latents", where).slice();

/** The refined text of a branch, identical in every golden of that branch (checked). */
const refinedOf = (goldens: readonly MossGolden[], where: string): MossRefinedText => {
  const [first, ...rest] = goldens.map((golden) => floatsOf(golden.file, "text_context", where));
  for (const other of rest) {
    assertEquals(
      other.length,
      first.length,
      `${where}: text contexts of one branch differ in length`,
    );
    assert(
      other.every((value, index) => Object.is(value, first[index])),
      `${where}: text contexts differ`,
    );
  }
  return { data: first.slice(), tokens: first.length / TEXT_OUTPUT_WIDTH };
};

/** Synthetic latents `[128, side/16, side/16]` (the default run reads no golden). */
const syntheticLatents = (side: number): Float32Array<ArrayBuffer> =>
  Float32Array.from({ length: 128 * (side / 16) ** 2 }, (_, index) => Math.sin(index * 0.37));

/**
 * A component whose Sessions log `open <name>` / `dispose <name>` into `events`, in order
 * (`refuse` makes the opening throw before anything is created).
 */
const observed = (
  component: ModelComponent,
  name: string,
  events: string[],
  refuse?: Error,
): { readonly component: ModelComponent; readonly sessions: Session[] } => {
  const sessions: Session[] = [];
  return {
    sessions,
    component: {
      ...component,
      createSession: async (gpu, options) => {
        if (refuse !== undefined) throw refuse;
        const session = await component.createSession(gpu, options);
        events.push(`open ${name}`);
        sessions.push(session);
        const dispose = session.dispose.bind(session);
        Object.defineProperty(session, "dispose", {
          configurable: true,
          value: async () => {
            events.push(`dispose ${name}`);
            await dispose();
          },
        });
        return session;
      },
    },
  };
};

/** The backing bytes the plan backing budget counts (`PlanBackingStats`). */
const budgetedBytes = (diagnostics: SessionDiagnostics): number =>
  diagnostics.planBacking.residentBytes + diagnostics.planBacking.inputBytes;

describe.skipIf(!GPU_AVAILABLE || !SERIES_PRESENT || !EMBEDS_PRESENT)(
  `MOSS-Image DiT stage on the GPU${skipped(GPU_AVAILABLE, "GPU adapter", "a WebGPU device")}${skipped(SERIES_PRESENT, "series", MOSS_GENERATE_SERIES)}${skipped(EMBEDS_PRESENT, "stage-0 embeddings", MOSS_GENERATE_EMBEDS)}`,
  () => {
    let gpu: GpuContext | undefined;
    let base: Omit<MossDitStageState, "gpu"> | undefined;
    let embeds: ReadonlyMap<string, MossTextEmbeddings> | undefined;
    let digests: Readonly<Record<string, string>> | undefined;
    beforeAll(async () => {
      const series = await openMossSeries();
      const assets = await readMossDitAssets(series.components.dit);
      const contract = mossDitContract(
        mossDitGraphContract(series.components, "MOSS-Image"),
        assets.ropeBase,
        assets.trainingGrid,
        "MOSS-Image",
      );
      base = { components: series.components, contract };
      embeds = await readEmbeddings();
      digests = await containerDigests();
      // The sha rows may be written: the adapter that runs must be the one the row key names.
      gpu = verifyRunningGpu(await acquireTestGpu());
    });
    afterAll(async () => {
      if (gpu === undefined) return;
      await settleReleases(gpu);
      gpu.destroy();
    });
    const stateOf = (
      contract?: MossDitContract,
      components?: MossDitComponents,
    ): MossDitStageState => {
      assert(gpu !== undefined && base !== undefined, "beforeAll did not finish");
      return {
        gpu,
        contract: contract ?? base.contract,
        components: components ?? base.components,
      };
    };
    const baseState = () => {
      assert(base !== undefined, "beforeAll did not finish");
      return base;
    };
    const allEmbeddings = () => {
      assert(embeds !== undefined, "beforeAll did not finish");
      return embeds;
    };
    const golden = async (testCase: MossDitCase | MossTextCase): Promise<MossGolden> => {
      assert(digests !== undefined, "beforeAll did not finish");
      return loadCaseGolden(testCase, digests);
    };
    /** Both branches (fridge-note T = 63, the empty negative T = 10) at a square size. */
    const planAt = (side: number): MossDitPlan =>
      planMossDit({
        width: side,
        height: side,
        positive: embeddingsOf(allEmbeddings(), "fridge-note"),
        negative: embeddingsOf(allEmbeddings(), "negative"),
      });
    /** The r gates that passed in this run (the text gate adds {@link TEXT_PARITY_ID} last). */
    const parityPassed = new Set<string>();

    describe.skipIf(!UPSTREAM_PARITY || !present(MOSS_TEXT_CASES))(
      `dit_text against the f64 reference${UPSTREAM_PARITY_OPT_IN}${skipped(present(MOSS_TEXT_CASES), "text goldens", MOSS_GENERATE_GOLDENS)}`,
      () => {
        const runText = async (
          testCase: MossTextCase,
          contract?: MossDitContract,
          edit: (data: Float32Array<ArrayBuffer>) => void = () => {},
        ) => {
          const loaded = await golden(testCase);
          const data = floatsOf(loaded.file, "text_embeds", testCase.name).slice();
          edit(data);
          const plan = planMossDit({
            width: 256,
            height: 256,
            positive: { data, shape: [testCase.tokens, 4096] },
            negative: undefined,
          });
          const refined = await runMossTextStage(stateOf(contract), plan);
          return { loaded, output: refined.positive.data };
        };

        it("each case within both bands of the six band cases (T = 1 included); the three faults leave them", async () => {
          const rows: Row[] = [];
          const channelRows: Row[] = [];
          const faults: Fault[] = [];
          const channelFaults: Fault[] = [];
          const contract = stateOf().contract;
          const doubled = { ...contract, ropeBase: doubledPositions(contract.ropeBase) };
          const measure = (output: Float32Array, loaded: MossGolden, name: string) => ({
            ratio: ratioOf(output, loaded, name),
            channel: channelRatioOf(output, loaded, name, TEXT_OUTPUT_WIDTH),
          });
          const fault = (name: string, measured: ReturnType<typeof measure>) => {
            faults.push({ name, ratio: measured.ratio });
            channelFaults.push({ name, ratio: measured.channel });
          };
          for (const testCase of MOSS_TEXT_CASES) {
            const { name, role, tokens } = testCase;
            const { loaded, output } = await runText(testCase);
            const measured = measure(output, loaded, name);
            rows.push({ name, role, ratio: measured.ratio });
            channelRows.push({ name, role, ratio: measured.channel });
            if (role === "accept" && tokens >= 2) {
              const positions = await runText(testCase, doubled);
              fault(`${name} positions×2`, measure(positions.output, loaded, name));
              const last = {
                ...contract,
                ropeBase: lastTokenOneFurther(contract.ropeBase, tokens),
              };
              const moved = await runText(testCase, last);
              fault(`${name} last-token+1`, measure(moved.output, loaded, name));
              const width = 4096;
              const layout = await runText(testCase, undefined, (data) =>
                data.copyWithin(data.length - width, data.length - 2 * width, data.length - width),
              );
              fault(`${name} last-row`, measure(layout.output, loaded, name));
            }
          }
          assertJudged([
            ...judge("dit_text", TEXT_BAND, rows, faults),
            ...judge("dit_text per channel", TEXT_CHANNEL_BAND, channelRows, channelFaults),
          ]);
          parityPassed.add(TEXT_PARITY_ID);
        });
      },
    );

    describe("default run (no reference read)", () => {
      it("runs dit_text at T = 1 (the graph was traced with T ≥ 2) to a finite [1, 3072], pinned by its sha256 row", async () => {
        const plan = planMossDit({
          width: 256,
          height: 256,
          positive: embeddingsOf(allEmbeddings(), "fridge-note", 1),
          negative: undefined,
        });
        let settlement: ReferenceSettlement | undefined;

        await runRecordedCase(results, { id: TEXT_T1_CASE }, async () => {
          const refined = await runMossTextStage(stateOf(), plan);

          expect(refined.positive.tokens).toBe(1);
          expect(refined.positive.data.length).toBe(TEXT_OUTPUT_WIDTH);
          assert(refined.positive.data.every(Number.isFinite));
          // The row is settled only after the checks above held (no row from a broken output),
          // and written only after the text cases' r gate passed earlier in this run (module doc).
          assertMossRowWritable(
            references.mode,
            { expected: [TEXT_PARITY_ID], passed: parityPassed },
            TEXT_T1_CASE,
          );
          const outcome = await settleOrObserve(references, results, {
            id: TEXT_T1_CASE,
            artifact: `${TEXT_T1_CASE}.safetensors`,
            bytes: f32ArtifactBytes("dit_text", [1, 1, TEXT_OUTPUT_WIDTH], refined.positive.data),
          });
          settlement = outcome.settlement;
          return outcome.fields;
        });

        if (settlement?.check.status === "fail") {
          throw new Error(referenceMismatchMessage(TEXT_T1_CASE, settlement, references));
        }
      });

      it("keeps one plan backing per Session over three steps (inside the budget), the borrower holding no weight", async () => {
        const state = stateOf();
        const plan = planAt(256);
        const refined = await runMossTextStage(state, plan);
        const latents = syntheticLatents(256);

        const diagnostics = await withMossDitPair(state, plan, refined, async (pair) => {
          for (const index of [0, 500, 976]) {
            const outputs = await pair.predict(latents, index);
            assert(outputs.negative !== undefined);
            assertEquals(outputs.positive.length, 128 * 256);
            assert(
              outputs.positive.every(Number.isFinite) && outputs.negative.every(Number.isFinite),
            );
          }
          return pair.diagnostics();
        });

        assert(diagnostics.negative !== undefined);
        expect(diagnostics.positive.planBacking.buildCount).toBe(1);
        expect(diagnostics.negative.planBacking.buildCount).toBe(1);
        // The path this size takes: each backing fits the plan backing budget.
        expect(budgetedBytes(diagnostics.positive)).toBeLessThanOrEqual(
          DEFAULT_PLAN_BACKING_BUDGET_BYTES,
        );
        expect(budgetedBytes(diagnostics.negative)).toBeLessThanOrEqual(
          DEFAULT_PLAN_BACKING_BUDGET_BYTES,
        );
        // The headroom check's estimates count what the Sessions hold: the lender every i8 weight,
        // the borrower none (its compressed weights are the lender's).
        const limits = {
          maxStorageBufferBindingSize: state.gpu.limits.maxStorageBufferBindingSize,
          maxBufferSize: state.gpu.limits.maxBufferSize,
        };
        const lender = state.components.dit.estimate({ ...limits, bindings: { S: 256 + 63 } });
        const borrower = state.components.ditSkip22.estimate({
          ...limits,
          bindings: { S: 256 + 10 },
        });
        expect(lender.resident.weights.compressedBytes).toBeGreaterThan(4 * 2 ** 30);
        expect(diagnostics.positive.storage.residentCompressedBytes).toBe(
          lender.resident.weights.compressedBytes,
        );
        expect(borrower.resident.weights.totalBytes).toBe(0);
        expect(diagnostics.negative.storage.residentCompressedBytes).toBe(0);
      });

      it("keeps one plan backing per Session over two steps when the backing is over the budget (768²)", async () => {
        const state = stateOf();
        const plan = planAt(768);
        const refined = await runMossTextStage(state, plan);
        const latents = syntheticLatents(768);

        const diagnostics = await withMossDitPair(state, plan, refined, async (pair) => {
          for (const index of [0, 500]) {
            const outputs = await pair.predict(latents, index);
            assert(outputs.negative !== undefined);
            assert(
              outputs.positive.every(Number.isFinite) && outputs.negative.every(Number.isFinite),
            );
          }
          return pair.diagnostics();
        });

        assert(diagnostics.negative !== undefined);
        // The path this size takes: each backing alone exceeds the budget (kept alone). 512² does
        // not (236,530,524 B for dit, measured 2026-10-27).
        expect(budgetedBytes(diagnostics.positive)).toBeGreaterThan(
          DEFAULT_PLAN_BACKING_BUDGET_BYTES,
        );
        expect(budgetedBytes(diagnostics.negative)).toBeGreaterThan(
          DEFAULT_PLAN_BACKING_BUDGET_BYTES,
        );
        expect(diagnostics.positive.planBacking.buildCount).toBe(1);
        expect(diagnostics.negative.planBacking.buildCount).toBe(1);
      });

      describe("lifetimes", () => {
        /** The three components, each observed into `events` (`refuse` = the borrower's). */
        const observedComponents = (events: string[], refuse?: Error) => {
          const { components } = baseState();
          const dit = observed(components.dit, "dit", events);
          const ditSkip22 = observed(components.ditSkip22, "dit_skip22", events, refuse);
          const ditText = observed(components.ditText, "dit_text", events);
          return {
            components: {
              dit: dit.component,
              ditSkip22: ditSkip22.component,
              ditText: ditText.component,
            },
            lender: dit.sessions,
            text: ditText.sessions,
          };
        };

        it("closes dit_text before the pair opens, and the whole pair and its contexts when the body throws", async () => {
          const events: string[] = [];
          const { components, lender } = observedComponents(events);
          const state = stateOf(undefined, components);
          const plan = planAt(256);
          const thrown = new Error("the body failed");

          const residents = await withResidentLog(state.gpu, async (residents) => {
            const refined = await runMossTextStage(state, plan);
            const error = await assertRejects(() =>
              withMossDitPair(state, plan, refined, async (pair) => {
                await pair.predict(syntheticLatents(256), 0);
                throw thrown;
              }),
            );
            expect(error).toBe(thrown);
            return residents;
          });

          expect(events).toEqual([
            "open dit_text",
            "dispose dit_text",
            "open dit",
            "open dit_skip22",
            "dispose dit_skip22",
            "dispose dit",
          ]);
          expect(residents.length).toBe(2);
          assert(residents.every((resident) => resident.disposed));
          const [borrowed] = baseState().contract.borrowed;
          expect(() => lender[0].exportWeight(borrowed)).toThrow(/disposed/);
        });

        it("closes the lender and its context when the borrower fails to open", async () => {
          const events: string[] = [];
          const refused = new Error("the borrower failed to open");
          const { components } = observedComponents(events, refused);
          const state = stateOf(undefined, components);
          const plan = planAt(256);
          const refined = await runMossTextStage(state, plan);
          events.length = 0;

          const residents = await withResidentLog(state.gpu, async (residents) => {
            const error = await assertRejects(() =>
              withMossDitPair(state, plan, refined, async () => undefined),
            );
            expect(error).toBe(refused);
            return residents;
          });

          expect(events).toEqual(["open dit", "dispose dit"]);
          expect(residents.length).toBe(1);
          assert(residents[0].disposed);
        });

        it("closes dit_text when its run fails (an output name the graph does not have)", async () => {
          const events: string[] = [];
          const { components, text } = observedComponents(events);
          const { contract } = baseState();
          const misnamed = {
            ...contract,
            outputs: { ...contract.outputs, ditText: "no_such_output" },
          };

          await assertRejects(
            () => runMossTextStage(stateOf(misnamed, components), planAt(256)),
            Error,
            "dit_text returned no 'no_such_output'",
          );

          expect(events).toEqual(["open dit_text", "dispose dit_text"]);
          await assertRejects(() => text[0].run({}), Error, "disposed");
        });

        it("opens nothing when the headroom check refuses a stage, naming the stage", async () => {
          const events: string[] = [];
          const { components } = observedComponents(events);
          const huge = (component: ModelComponent): ModelComponent => ({
            ...component,
            estimate: (options) => ({
              ...component.estimate(options),
              peakAccountedBytes: 2 ** 50,
            }),
          });
          const plan = planAt(256);

          await assertRejects(
            () =>
              runMossTextStage(
                stateOf(undefined, { ...components, ditText: huge(components.ditText) }),
                plan,
              ),
            Error,
            "the dit_text stage needs",
          );
          expect(events).toEqual([]);

          const state = stateOf(undefined, components);
          const refined = await runMossTextStage(state, plan);
          events.length = 0;
          await assertRejects(
            () =>
              withMossDitPair(
                { ...state, components: { ...components, dit: huge(components.dit) } },
                plan,
                refined,
                async () => undefined,
              ),
            Error,
            "the dit + dit_skip22 stage needs",
          );
          expect(events).toEqual([]);
        });
      });
    });

    const ditGate = (
      group: "small" | "full",
      cases: readonly MossDitCase[],
      band: number | undefined,
    ) => {
      const side = SIDE[group];
      const indices = [...new Set(cases.map(({ gridIndex }) => gridIndex))];
      const accepted = [
        ...new Set(cases.filter(({ role }) => role === "accept").map(({ gridIndex }) => gridIndex)),
      ];
      const caseAt = (branch: "positive" | "negative", index: number): MossDitCase => {
        const found = cases.find((c) => c.branch === branch && c.gridIndex === index);
        if (found === undefined) throw new Error(`${group}: no ${branch} case at index ${index}`);
        return found;
      };
      const load = async () => {
        const goldens = new Map<string, MossGolden>();
        for (const testCase of cases) goldens.set(testCase.name, await golden(testCase));
        const of = (testCase: MossDitCase): MossGolden => {
          const found = goldens.get(testCase.name);
          assert(found !== undefined);
          return found;
        };
        const branch = (name: "positive" | "negative") =>
          cases.filter((c) => c.branch === name).map(of);
        const refined: MossBranches<MossRefinedText> = {
          positive: refinedOf(branch("positive"), `${group} positive`),
          negative: refinedOf(branch("negative"), `${group} negative`),
        };
        const plan = planAt(side);
        assertEquals(plan.texts.positive.tokens, refined.positive.tokens, "the positive T");
        return { of, refined, plan };
      };
      /** Both branches' r at a grid index (`timeRow` other than `index` = the time fault). */
      const predictAt = async (
        pair: Parameters<Parameters<typeof withMossDitPair>[3]>[0],
        of: (c: MossDitCase) => MossGolden,
        index: number,
        timeRow = index,
      ) => {
        const latents = latentsOf(of(caseAt("positive", index)), group);
        const outputs = await pair.predict(latents, timeRow);
        assert(outputs.negative !== undefined);
        return (["positive", "negative"] as const).map((branch) => {
          const testCase = caseAt(branch, index);
          const output = branch === "positive" ? outputs.positive : outputs.negative;
          assert(output !== undefined);
          return { testCase, ratio: ratioOf(output, of(testCase), testCase.name) };
        });
      };

      describe.skipIf(!UPSTREAM_PARITY || !present(cases))(
        `${side}² dit / dit_skip22 against the f64 reference${UPSTREAM_PARITY_OPT_IN}${skipped(present(cases), `${group} goldens`, MOSS_GENERATE_GOLDENS)}`,
        () => {
          it("each case within the band of the six band cases; the time row of index + 1 leaves it; one backing per Session", async () => {
            const { of, refined, plan } = await load();
            const rows: Row[] = [];
            const faults: Fault[] = [];
            const diagnostics = await withMossDitPair(stateOf(), plan, refined, async (pair) => {
              for (const index of indices) {
                for (const { testCase, ratio } of await predictAt(pair, of, index)) {
                  rows.push({ name: testCase.name, role: testCase.role, ratio });
                }
              }
              for (const index of accepted) {
                for (const { testCase, ratio } of await predictAt(pair, of, index, index + 1)) {
                  faults.push({ name: `${testCase.name} time+1`, ratio });
                }
              }
              return pair.diagnostics();
            });
            assert(diagnostics.negative !== undefined);
            expect(diagnostics.positive.planBacking.buildCount).toBe(1);
            expect(diagnostics.negative.planBacking.buildCount).toBe(1);
            assertJudged(judge(`${side}²`, band, rows, faults));
          });

          it("image tokens one position further from the text leave the band (positive acceptance cases)", async () => {
            const { of, refined, plan } = await load();
            const contract = stateOf().contract;
            const tokens = refined.positive.tokens;
            const moved = { ...contract, ropeBase: imageOneFurther(contract.ropeBase, tokens) };
            // The positive branch alone (guidance 1): its text rows 0 … T − 1 keep their positions.
            const alone: MossDitPlan = { ...plan, texts: { ...plan.texts, negative: undefined } };
            const faults: Fault[] = [];
            await withMossDitPair(
              stateOf(moved),
              alone,
              { ...refined, negative: undefined },
              async (pair) => {
                for (const index of accepted) {
                  const testCase = caseAt("positive", index);
                  const outputs = await pair.predict(latentsOf(of(testCase), group), index);
                  faults.push({
                    name: `${testCase.name} image-f+1`,
                    ratio: ratioOf(outputs.positive, of(testCase), testCase.name),
                  });
                }
              },
            );
            for (const { name, ratio } of faults)
              console.log(`[moss-dit] fault ${formatRatio(name, ratio)}`);
            assert(band !== undefined, `${side}²: the band is not derived`);
            assertJudged(faultsInside(faults, band));
          });

          it("the product path (dit_text on the GPU from the stage-0 embeddings) stays inside the band", async () => {
            const { of, plan } = await load();
            const state = stateOf();
            const refined = await runMossTextStage(state, plan);
            const rows: Row[] = [];
            await withMossDitPair(state, plan, refined, async (pair) => {
              for (const index of accepted) {
                for (const { testCase, ratio } of await predictAt(pair, of, index)) {
                  rows.push({ name: `${testCase.name} product`, role: "accept", ratio });
                }
              }
            });
            for (const { name, ratio } of rows)
              console.log(`[moss-dit] ${formatRatio(name, ratio)}`);
            assert(band !== undefined, `${side}²: the band is not derived`);
            for (const { name, ratio } of rows) {
              assert(ratio.r <= band, `${formatRatio(name, ratio)} is outside the band ${band}`);
            }
          });
        },
      );
    };

    ditGate("small", SMALL, SMALL_BAND);
    ditGate("full", FULL, FULL_BAND);
  },
);

/** The sha rows this file settles (the default run's `dit_text` at T = 1). */
const CASE_IDS: readonly string[] = [TEXT_T1_CASE];
const RUNNABLE = GPU_AVAILABLE && SERIES_PRESENT && EMBEDS_PRESENT;
if (RUNNABLE) references.warnMissing(CASE_IDS);
// A missing row of this environment is not a silent pass (ADR 0005's gatekeeper form).
registerReferenceGate(references, { runnable: RUNNABLE, caseIds: CASE_IDS });
