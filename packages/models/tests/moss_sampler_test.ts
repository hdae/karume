/**
 * The MOSS-Image sampler on the CPU (ADR NNNN decisions 9 and 11, stage 4):
 * the σ schedule against upstream's own `set_timesteps`, upstream's CFG line, the Euler update and
 * the seeded noise.
 *
 * Source of the expected values: the stage-0 host record (`host.safetensors` of
 * `moss.host_fixture`, taken at two torch thread counts) — for every recorded step count (1–200,
 * 500, 1,000) upstream's requested grid before snapping (`steps.<n>.requested`, read from the
 * `torch.searchsorted` argument inside `set_timesteps`), the training-grid index each requested
 * point snapped to (`steps.<n>.nearest`), the de-duplicated indices (`steps.<n>.grid`) and the σ /
 * timestep columns the scheduler kept, at the pinned config's shift (`scheduler_config.shift`); the
 * same five for eleven other shifts × eight step counts, two of them above the 1,000 training points
 * (`shift.<i>.steps.<n>.*`, the metadata's `shift_cases`); and upstream's CFG combination at nine
 * guidance scales, most not representable in f32 (`cfg.*`, one step of upstream's `__call__` on a
 * tiny DiT — the metadata's `cfg`). The training grid itself comes from the series' asset
 * `training_grid` (the product's source — `moss_dit_host_test.ts` checks it against the same
 * record), and each timestep is looked up by grid index as the DiT stage does (`mossTimestep`).
 *
 * The CFG + Euler trajectory against upstream's recorded DiT outputs is in
 * `moss_generate_host_test.ts` (through the generate path's own loop).
 *
 * Each group skips explicitly, naming the command that writes what it lacks (ADR 0005).
 */

import { describe, expect, it } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { parseSafetensors, type SafetensorsFile } from "@karume/runtime";
import { ModelInputError } from "../src/errors.ts";
import { readMossDitAssets } from "../src/moss/dit-contract.ts";
import { mossClassifierFreeGuidance } from "../src/moss/dit-stage.ts";
import { type MossTrainingGrid, mossTimestep } from "../src/moss/dit-timestep.ts";
import { MossRandn } from "../src/moss/random.ts";
import {
  mossEulerStep,
  mossNearestGridIndex,
  mossRequestedSigmas,
  mossSchedule,
} from "../src/moss/sampler.ts";
import { WanRandn } from "../src/wan/random.ts";
import {
  filePresent,
  firstBitMismatch,
  floatsOf,
  MOSS_GENERATE_SERIES,
  MOSS_GENERATE_STAGE0,
  mossSeriesPresent,
  openMossSeries,
  readBuffer,
  stage0Dir,
  syntheticTrainingGrid,
  viewOf,
} from "./helpers/moss-dit.ts";

const STAGE0 = stage0Dir();
const HOST_RECORD = STAGE0 === undefined ? undefined : new URL("host.safetensors", STAGE0);
const HOST_PRESENT = HOST_RECORD !== undefined && filePresent(HOST_RECORD);
const SERIES_PRESENT = mossSeriesPresent();

const skipped = (present: boolean, what: string, command: string): string =>
  present ? "" : ` (SKIP: no ${what} — ${command})`;

/** The recorded step counts (`moss.host_fixture.STEP_COUNTS`). */
const STEP_COUNTS = [...Array.from({ length: 200 }, (_, index) => index + 1), 500, 1000];

const int32Of = (file: SafetensorsFile, key: string): Int32Array<ArrayBuffer> => {
  const view = viewOf(file, key, "host record");
  if (view.dtype !== "I32") throw new Error(`host record: '${key}' is ${view.dtype}`);
  return new Int32Array(file.buffer, view.byteOffset, view.byteLength / 4);
};

/** One object field of the host record's metadata (failing loudly when the record predates it). */
const metadataField = (file: SafetensorsFile, key: string): Readonly<Record<string, unknown>> => {
  const raw = file.metadata.get("karume.moss.host_fixture");
  assert(raw !== undefined, "the host record has no metadata");
  const metadata: unknown = JSON.parse(raw);
  assert(typeof metadata === "object" && metadata !== null);
  const field: unknown = Object.hasOwn(metadata, key)
    ? Object.fromEntries(Object.entries(metadata))[key]
    : undefined;
  assert(
    typeof field === "object" && field !== null,
    `the host record has no '${key}' — regenerate it: ${MOSS_GENERATE_STAGE0}`,
  );
  return Object.fromEntries(Object.entries(field));
};

/** A number list of a metadata field. */
const numbersOf = (value: unknown, what: string): readonly number[] => {
  assert(
    Array.isArray(value) && value.every((item) => typeof item === "number"),
    `${what} is not a list of numbers`,
  );
  return value;
};

/**
 * The failures of one recorded sequence (`<where>.requested` / `nearest` / `grid` / `sigmas` /
 * `timesteps`) against the TS schedule of `steps` at `shift` (bit for bit).
 */
const sequenceFailures = (
  host: SafetensorsFile,
  grid: MossTrainingGrid,
  where: string,
  steps: number,
  shift: number,
): string[] => {
  const failures: string[] = [];
  const requested = mossRequestedSigmas(steps, shift, grid.points);
  if (firstBitMismatch(requested, floatsOf(host, `${where}.requested`, "host")) !== -1) {
    failures.push(`${where}: requested grid`);
  }
  const nearest = Int32Array.from(requested, (sigma) => mossNearestGridIndex(grid, sigma));
  if (
    firstBitMismatch(
      Uint32Array.from(nearest),
      Uint32Array.from(int32Of(host, `${where}.nearest`)),
    ) !== -1
  ) {
    failures.push(`${where}: snapped indices`);
  }
  const schedule = mossSchedule(grid, steps, shift);
  if (
    firstBitMismatch(
      Uint32Array.from(schedule.gridIndices),
      Uint32Array.from(int32Of(host, `${where}.grid`)),
    ) !== -1
  ) {
    failures.push(`${where}: de-duplicated indices`);
  }
  if (firstBitMismatch(schedule.sigmas, floatsOf(host, `${where}.sigmas`, "host")) !== -1) {
    failures.push(`${where}: σ column`);
  }
  const timesteps = Float32Array.from(schedule.gridIndices, (index) => mossTimestep(grid, index));
  if (firstBitMismatch(timesteps, floatsOf(host, `${where}.timesteps`, "host")) !== -1) {
    failures.push(`${where}: timestep column`);
  }
  return failures;
};

describe.skipIf(!HOST_PRESENT || !SERIES_PRESENT)(
  `MOSS-Image σ schedule against upstream's set_timesteps${skipped(HOST_PRESENT, "stage-0 host record", MOSS_GENERATE_STAGE0)}${skipped(SERIES_PRESENT, "series", MOSS_GENERATE_SERIES)}`,
  () => {
    const load = async () => {
      assert(HOST_RECORD !== undefined);
      const host = parseSafetensors(await readBuffer(HOST_RECORD));
      const { trainingGrid } = await readMossDitAssets((await openMossSeries()).components.dit);
      const config = metadataField(host, "scheduler_config");
      assert(typeof config["shift"] === "number");
      assertEquals(config["num_train_timesteps"], trainingGrid.points);
      return { host, grid: trainingGrid, shift: config["shift"] };
    };

    it("reproduces the requested grid, the snapping, the de-duplication and the σ / timestep columns of all 202 recorded counts bit for bit", async () => {
      const { host, grid, shift } = await load();

      const failures = STEP_COUNTS.flatMap((steps) =>
        sequenceFailures(host, grid, `steps.${steps}`, steps, shift),
      );

      assertEquals(failures, []);
    });

    it("reproduces the same five columns at eleven other shifts × eight step counts (two above the 1,000 training points) bit for bit", async () => {
      const { host, grid, shift } = await load();
      const cases = metadataField(host, "shift_cases");
      const shifts = numbersOf(cases["shifts"], "shift_cases.shifts");
      const counts = numbersOf(cases["step_counts"], "shift_cases.step_counts");
      assert(shifts.length === 11 && !shifts.includes(shift), `shifts ${shifts.join(", ")}`);
      assert(
        counts.some((steps) => steps > grid.points),
        `counts ${counts.join(", ")}`,
      );

      const failures = shifts.flatMap((other, index) =>
        counts.flatMap((steps) =>
          sequenceFailures(host, grid, `shift.${index}.steps.${steps}`, steps, other),
        ),
      );

      assertEquals(failures, []);
    });

    it("reproduces upstream's CFG combination at nine guidance scales (most not representable in f32) bit for bit", async () => {
      const { host } = await load();
      const scales = numbersOf(metadataField(host, "cfg")["scales"], "cfg.scales");
      assert(
        scales.some((scale) => Math.fround(scale) !== scale),
        "no scale needs the f32 rounding",
      );
      const cond = floatsOf(host, "cfg.cond", "host");
      const uncond = floatsOf(host, "cfg.uncond", "host");

      const failures = scales.flatMap((scale, index) =>
        firstBitMismatch(
          mossClassifierFreeGuidance(cond, uncond, scale),
          floatsOf(host, `cfg.${index}.combined`, "host"),
        ) === -1
          ? []
          : [`guidance ${scale}`],
      );

      assertEquals(failures, []);
    });

    it("runs fewer steps than requested where the snapping merges points (500 → 415, 1,000 → 615)", async () => {
      const { grid, shift } = await load();

      expect(mossSchedule(grid, 500, shift).gridIndices.length).toBe(415);
      expect(mossSchedule(grid, 1000, shift).gridIndices.length).toBe(615);
      expect(mossSchedule(grid, 50, shift).gridIndices.length).toBe(50);
    });
  },
);

describe("MOSS-Image σ schedule structure", () => {
  it("runs one step at σ = 1 for a single requested step, then the terminal 0", () => {
    const grid = syntheticTrainingGrid(1000);

    const schedule = mossSchedule(grid, 1, 5);

    expect(schedule.gridIndices).toEqual([0]);
    expect([...schedule.sigmas]).toEqual([1, 0]);
    expect(mossTimestep(grid, schedule.gridIndices[0])).toBe(1000);
  });

  it("takes a step count above the grid's points (5,000): strictly increasing indices, at most the grid's points", () => {
    // Not a bound for every count: past about 2·10⁶ steps the indices stop increasing, and they
    // outnumber the points only at about 10⁷ (1,099; 989 at 5·10⁶), as upstream's (sampler.ts
    // "Very large step counts").
    const grid = syntheticTrainingGrid(1000);

    const schedule = mossSchedule(grid, 5000, 5);

    assert(schedule.gridIndices.length <= 1000);
    assert(
      schedule.gridIndices.every((index, at) => at === 0 || index > schedule.gridIndices[at - 1]),
    );
    expect(schedule.sigmas.length).toBe(schedule.gridIndices.length + 1);
  });

  it("snaps a tie to the smaller σ (upstream's `<=` picks the lower neighbour)", () => {
    const grid: MossTrainingGrid = {
      points: 3,
      width: 1,
      sigmas: Float32Array.of(1, 0.5, 0.25),
      timesteps: Float32Array.of(3, 1.5, 0.75),
      sinusoid: new Float32Array(3),
    };

    expect(mossNearestGridIndex(grid, 0.75)).toBe(1);
    expect(mossNearestGridIndex(grid, 0.375)).toBe(2);
    expect(mossNearestGridIndex(grid, 2)).toBe(0);
    expect(mossNearestGridIndex(grid, 0)).toBe(2);
  });

  it("refuses a requested σ that is not finite (a shift so small that the grid divides by zero)", () => {
    assertThrows(
      () => mossSchedule(syntheticTrainingGrid(1000), 2, 1e-45),
      RangeError,
      "requested σ 0 of 2",
    );
    assertThrows(
      () => mossNearestGridIndex(syntheticTrainingGrid(10), Number.NaN),
      RangeError,
      "not finite",
    );
  });

  it("refuses a step count or a shift outside its range (the acceptance check's job — a caller's bug here)", () => {
    for (const steps of [0, 2.5, -1]) {
      assertThrows(
        () => mossSchedule(syntheticTrainingGrid(10), steps, 5),
        RangeError,
        "not a positive integer",
      );
    }
    for (const shift of [0, -1, Number.POSITIVE_INFINITY, 1e39]) {
      assertThrows(
        () => mossSchedule(syntheticTrainingGrid(10), 2, shift),
        RangeError,
        "not a positive finite",
      );
    }
  });
});

describe("MOSS-Image Euler step (f32 — FlowMatchEulerDiscreteScheduler.step)", () => {
  it("is latents + f32(σ_next − σ) · prediction, each operation rounded to f32", () => {
    const latents = Float32Array.of(1, 2, 0.1);
    const prediction = Float32Array.of(0.5, -1, 0.3);
    const sigma = Math.fround(0.7);
    const next = Math.fround(0.1);
    const delta = Math.fround(next - sigma);

    const out = mossEulerStep(latents, prediction, sigma, next);

    expect([...out]).toEqual([
      Math.fround(1 + Math.fround(delta * 0.5)),
      Math.fround(2 + Math.fround(delta * -1)),
      Math.fround(latents[2] + Math.fround(delta * prediction[2])),
    ]);
    // One rounding fewer (the step in double) moves the last value by an ULP.
    expect(out[2]).not.toBe(Math.fround(latents[2] + (next - sigma) * prediction[2]));
  });

  it("refuses latents and a prediction of different lengths", () => {
    assertThrows(
      () => mossEulerStep(new Float32Array(2), new Float32Array(3), 1, 0),
      Error,
      "Euler step of 2 latents and 3 predictions",
    );
  });
});

describe("MOSS-Image seeded noise (splitmix64 + Box–Muller — not torch's randn)", () => {
  it("pins the sequence by value: the first eight normals of seed 42 (the sha rows' seed)", () => {
    // "Same seed, same sequence" holds for any generator (a cos / sin swap, another draw order),
    // and a machine without sha rows would see no change at all, so the values themselves are fixed.
    // Baked with V8 (ECMAScript leaves Math.log / cos / sin to the implementation's approximation).
    assertEquals(
      [...new MossRandn(42).normals(8)],
      [
        0.41471976041793823, 0.6526812314987183, -0.8918862342834473, 1.3268336057662964,
        1.72959303855896, -1.883416771888733, 0.5456204414367676, -1.6568357944488525,
      ],
    );
  });

  it("is the same sequence as WanRandn (a deliberate copy of its core) for every seed tried", () => {
    for (const seed of [0, 1, 42, 20261027, Number.MAX_SAFE_INTEGER]) {
      assertEquals(
        firstBitMismatch(new MossRandn(seed).normals(1001), new WanRandn(seed).normals(1001)),
        -1,
        `seed ${seed}`,
      );
    }
  });

  it("gives the same sequence for the same seed and another one for another seed", () => {
    const first = new MossRandn(42).normals(1001);
    const again = new MossRandn(42).normals(1001);
    const other = new MossRandn(43).normals(1001);

    assertEquals(firstBitMismatch(first, again), -1);
    assert(firstBitMismatch(first, other) !== -1);
  });

  it("consumes whole Box–Muller pairs, also for an odd count", () => {
    const random = new MossRandn(7);
    random.normals(3);
    const after = random.normals(2);

    assertEquals(firstBitMismatch(after, new MossRandn(7).normals(6).subarray(4)), -1);
  });

  it("draws standard normal values (mean ≈ 0, variance ≈ 1 over 200,000 draws)", () => {
    const values = new MossRandn(20261027).normals(200_000);
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;

    assert(Math.abs(mean) < 0.01, `mean ${mean}`);
    assert(Math.abs(variance - 1) < 0.01, `variance ${variance}`);
    assert(values.every(Number.isFinite));
  });

  it("refuses a seed that is not a non-negative safe integer", () => {
    for (const seed of [-1, 1.5, 2 ** 53]) {
      assertThrows(
        () => new MossRandn(seed),
        ModelInputError,
        "is not a non-negative safe integer",
      );
    }
  });
});
