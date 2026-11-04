// Compares the two procedures of BiRefNet's α-matte postprocessing (`src/birefnet/matte.ts`) with
// goldens made by each series' upstream function (ADR 0125 — no GPU, no assets).
//
// The golden is the git-tracked `fixtures/birefnet-matte/matte.{json,safetensors}` (its writer is
// the recipe's `tools/export-recipes/birefnet/matte_fixture.py` — it calls the upstream functions
// as they are, with only the quantisation replaced by round-half-up). If it is missing the result
// is red, not SKIP (a state where nobody checks upstream parity must not be green).
//
// - `u8-pil-bicubic` (HR's upstream intent): **zero mismatches** against the golden's `<case>.hr`
//   (Pillow's 8-bit BICUBIC final α).
// - `float-bilinear` (Lucida's upstream intent = the very path of the pipeline before the procedure
//   was chosen by declaration): matches the final α obtained by rounding the golden's
//   `<case>.lucida` (torch's `255α` before rounding) half up, or may be off by one step only when
//   the golden's `255α` lies within ε (the fixture's `lucida.epsilon`) of `k + 0.5` (rounding of
//   torch's f32 coordinates and weights — ADR 0125).
//
// It also pins directly the spec that does not depend on how the fixture is made (quantisation
// rounds half up, there is no default procedure, an unknown procedure is rejected, a non-finite
// logit fails with pixel coordinates).

import { readTextFile } from "@karume/test-support/fs";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { parseSafetensors, type SafetensorsFile } from "@karume/runtime";
import { MATTE_PROCEDURES, matteFromLogits, type MatteProcedure } from "../src/birefnet/matte.ts";
import { floatsOf, readBuffer, viewOf } from "./helpers/safetensors.ts";

const FIXTURE = new URL("fixtures/birefnet-matte/matte.json", import.meta.url);

/** One row of the fixture's input table (the logits are `<name>.logits` in the safetensors). */
type MatteInputRow = {
  readonly name: string;
  readonly width: number;
  readonly height: number;
};

/**
 * One row of the fixture's case table (the expected values are `<name>.hr` / `<name>.lucida` in the
 * safetensors).
 */
type MatteCaseRow = {
  readonly name: string;
  readonly input: string;
  readonly outWidth: number;
  readonly outHeight: number;
};

type MatteFixture = {
  readonly versions: Readonly<Record<string, string>>;
  readonly epsilon: number;
  readonly inputs: ReadonlyMap<string, MatteInputRow>;
  readonly cases: readonly MatteCaseRow[];
  readonly tensors: SafetensorsFile;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null;

const isPositiveInteger = (value: unknown): value is number =>
  typeof value === "number" && Number.isInteger(value) && value > 0;

const isInputRow = (value: unknown): value is MatteInputRow =>
  isRecord(value) &&
  typeof value.name === "string" &&
  isPositiveInteger(value.width) &&
  isPositiveInteger(value.height);

const isCaseRow = (value: unknown): value is MatteCaseRow =>
  isRecord(value) &&
  typeof value.name === "string" &&
  typeof value.input === "string" &&
  isPositiveInteger(value.outWidth) &&
  isPositiveInteger(value.outHeight);

/**
 * Reads the fixture (the JSON of the input and case tables + the safetensors of the tensors). Fails
 * loudly if the shape departs from the writer's doc. MUST: throw when it is missing (it is a
 * git-tracked fixture, so a missing one is a mix-up or a botched write).
 */
const readFixture = async (): Promise<MatteFixture> => {
  const raw: unknown = JSON.parse(await readTextFile(FIXTURE));
  if (
    !isRecord(raw) ||
    !isRecord(raw.source) ||
    !isRecord(raw.source.versions) ||
    !isRecord(raw.lucida) ||
    typeof raw.lucida.epsilon !== "number" ||
    typeof raw.tensors !== "string" ||
    !Array.isArray(raw.inputs) ||
    !raw.inputs.every(isInputRow) ||
    !Array.isArray(raw.cases) ||
    !raw.cases.every(isCaseRow)
  ) {
    throw new Error(
      `${FIXTURE.pathname}: the shape is not {source.versions, lucida.epsilon, tensors, inputs, cases}`,
    );
  }
  const versions: Record<string, string> = {};
  for (const [name, version] of Object.entries(raw.source.versions)) {
    if (typeof version !== "string") {
      throw new Error(`${FIXTURE.pathname}: version ${name} is not a string`);
    }
    versions[name] = version;
  }
  return {
    versions,
    epsilon: raw.lucida.epsilon,
    inputs: new Map(raw.inputs.map((row) => [row.name, row])),
    cases: raw.cases,
    tensors: parseSafetensors(await readBuffer(new URL(raw.tensors, FIXTURE))),
  };
};

const fixture = await readFixture();

/** The input logits (F32 `[height, width]` — a shape mismatch fails loudly). */
const logitsOf = (row: MatteInputRow): Float32Array<ArrayBuffer> => {
  const key = `${row.name}.logits`;
  assertEquals([...viewOf(fixture.tensors, key, "matte fixture").shape], [row.height, row.width]);
  return floatsOf(fixture.tensors, key, "matte fixture");
};

/** The case's HR final α (U8 `[outHeight, outWidth]`). */
const hrOf = (row: MatteCaseRow): Uint8Array<ArrayBuffer> => {
  const key = `${row.name}.hr`;
  const view = viewOf(fixture.tensors, key, "matte fixture");
  assertEquals(view.dtype, "U8", `dtype of ${key}`);
  assertEquals([...view.shape], [row.outHeight, row.outWidth], `shape of ${key}`);
  return new Uint8Array(fixture.tensors.buffer, view.byteOffset, view.byteLength);
};

/** The case's Lucida `255α` before rounding (F32 `[outHeight, outWidth]`). */
const lucidaOf = (row: MatteCaseRow): Float32Array<ArrayBuffer> => {
  const key = `${row.name}.lucida`;
  assertEquals(
    [...viewOf(fixture.tensors, key, "matte fixture").shape],
    [row.outHeight, row.outWidth],
    `shape of ${key}`,
  );
  return floatsOf(fixture.tensors, key, "matte fixture");
};

const inputOf = (row: MatteCaseRow): MatteInputRow => {
  const input = fixture.inputs.get(row.input);
  if (input === undefined)
    throw new Error(`case ${row.name}: input ${row.input} is not in the input table`);
  return input;
};

/** Runs a case through a procedure. */
const run = (row: MatteCaseRow, procedure: MatteProcedure): Uint8Array<ArrayBuffer> => {
  const input = inputOf(row);
  const matte = matteFromLogits(
    logitsOf(input),
    input.width,
    input.height,
    row.outWidth,
    row.outHeight,
    procedure,
  );
  assertEquals(
    [matte.width, matte.height],
    [row.outWidth, row.outHeight],
    `dimensions of ${row.name}`,
  );
  return matte.data;
};

describe("u8-pil-bicubic (HR's upstream intent — Pillow's 8-bit BICUBIC)", () => {
  it("every fixture case has zero mismatches against the upstream function's final α", () => {
    const failures: string[] = [];
    for (const row of fixture.cases) {
      const expected = hrOf(row);
      const actual = run(row, "u8-pil-bicubic");
      let mismatches = 0;
      let first = -1;
      for (let index = 0; index < expected.length; index += 1) {
        if (actual[index] === expected[index]) continue;
        mismatches += 1;
        if (first < 0) first = index;
      }
      if (mismatches > 0) {
        failures.push(
          `${row.name}: ${mismatches} / ${expected.length} samples differ` +
            ` (the first is ${first}: ${actual[first]} vs ${expected[first]})`,
        );
      }
    }
    assertEquals(
      failures,
      [],
      `Pillow ${fixture.versions.pillow} golden: of ${fixture.cases.length} cases, ` +
        `${failures.length} differ`,
    );
  });

  it("quantisation rounds 0.5 up (not the upstream truncation)", () => {
    // At identity dimensions resize is a copy and only the quantisation remains. logit 0 → 255σ =
    // 127.5 → 128 (truncation would give 127), logit 7 → 254.77 → 255 (truncation would give 254).
    const matte = matteFromLogits(Float32Array.of(0, 7, -40, 40), 2, 2, 2, 2, "u8-pil-bicubic");
    assertEquals([...matte.data], [128, 255, 0, 255]);
  });
});

describe("float-bilinear (Lucida's upstream intent)", () => {
  it("every fixture case matches the upstream function's final α, off by one step only within ε", () => {
    const { epsilon } = fixture;
    const failures: string[] = [];
    for (const row of fixture.cases) {
      const scaled = lucidaOf(row);
      const actual = run(row, "float-bilinear");
      for (let index = 0; index < scaled.length; index += 1) {
        const expected = Math.floor(scaled[index] + 0.5);
        if (actual[index] === expected) continue;
        const boundary = Math.min(actual[index], expected) + 0.5;
        if (
          Math.abs(actual[index] - expected) === 1 &&
          Math.abs(scaled[index] - boundary) <= epsilon
        ) {
          continue;
        }
        failures.push(
          `${row.name}[${index}]: ${actual[index]} vs ${expected} (the golden's 255α = ${
            scaled[index]
          })`,
        );
      }
    }
    assertEquals(
      failures.slice(0, 8),
      [],
      `outside ε ${epsilon}: ${failures.length} mismatching samples`,
    );
  });
});

describe("the shared contract of the procedures", () => {
  it("the fixture includes upscaling, downscaling, opposite directions per axis, one axis only and identity, and avoids the graph's 1024²", () => {
    const shapes = fixture.cases.map((row) => ({ ...row, input: inputOf(row) }));
    const has = (predicate: (shape: (typeof shapes)[number]) => boolean, what: string): void =>
      assert(shapes.some(predicate), `no ${what} case`);
    has(
      ({ input, outWidth, outHeight }) => outWidth > input.width && outHeight > input.height,
      "upscaling",
    );
    has(
      ({ input, outWidth, outHeight }) => outWidth < input.width && outHeight < input.height,
      "downscaling",
    );
    has(
      ({ input, outWidth, outHeight }) => outWidth < input.width && outHeight > input.height,
      "horizontal downscaling with vertical upscaling",
    );
    has(
      ({ input, outWidth, outHeight }) => outWidth !== input.width && outHeight === input.height,
      "horizontal-only",
    );
    has(
      ({ input, outWidth, outHeight }) => outWidth === input.width && outHeight !== input.height,
      "vertical-only",
    );
    has(
      ({ input, outWidth, outHeight }) =>
        outWidth === input.width && outHeight === input.height && input.width > 1,
      "identity",
    );
    assert(
      shapes.every(({ outWidth, outHeight }) => outWidth !== 1024 || outHeight !== 1024),
      "there is a case at the graph's dimensions 1024² (resize becomes the identity and does not exercise interpolation)",
    );
  });

  it("the two procedures give different values when upscaling a step (catches a mix-up of procedures)", () => {
    // Upscaling a step makes bicubic's negative lobe differ from bilinear (one measured example,
    // R6B-001's 4 → 8).
    const step = Float32Array.of(-12, -12, 12, 12);
    const bilinear = matteFromLogits(step, 4, 1, 8, 1, "float-bilinear").data;
    const bicubic = matteFromLogits(step, 4, 1, 8, 1, "u8-pil-bicubic").data;
    assert(
      [...bilinear].some((value, index) => value !== bicubic[index]),
      "the two procedures give the same values",
    );
  });

  it("rejects an unknown procedure name (Object.prototype members are not accepted as procedures either)", () => {
    for (const name of ["bilinear", "toString", "__proto__"]) {
      assertThrows(
        // A cast to mimic a spelling outside the type (a JS caller).
        () => matteFromLogits(new Float32Array(4), 2, 2, 2, 2, name as MatteProcedure),
        Error,
        "postprocessing procedure",
      );
    }
  });

  it("matteFromLogits has no default procedure (a caller that passes no declaration fails)", () => {
    // The procedure argument is where the distribution's declaration (`pipelineConfig.postprocess`)
    // is received. With a default, a caller that forgot to pass the declaration would silently run
    // one series' procedure. It mimics a call outside the type (a JS caller) that omits the
    // argument.
    assertThrows(
      () => Reflect.apply(matteFromLogits, undefined, [new Float32Array(4), 2, 2, 2, 2]),
      Error,
      "postprocessing procedure undefined is not supported",
    );
  });

  it("a non-finite logit fails with pixel coordinates in every procedure (NaN does not turn into a transparent pixel)", () => {
    // Assigning into a `Uint8Array` silently rounds NaN to 0 = a fully transparent pixel, so
    // without the check "a matte that looks normal" would come back. ±Inf would fall to 0/1 through
    // σ, but reaching that point means the graph is broken, so it is rejected too (the same wording
    // as the quantisation exits of the other three families). Both procedures go through the same
    // check point.
    for (const procedure of MATTE_PROCEDURES) {
      assertThrows(
        () => matteFromLogits(Float32Array.of(0, 0, NaN, 0), 2, 2, 3, 3, procedure),
        Error,
        "the matte logit (x=0, y=1) is non-finite",
      );
      assertThrows(
        () => matteFromLogits(Float32Array.of(0, NaN, 0, 0), 2, 2, 2, 2, procedure),
        Error,
        "the matte logit (x=1, y=0) is non-finite",
      );
      assertThrows(
        () => matteFromLogits(Float32Array.of(0, 0, 0, Infinity), 2, 2, 2, 2, procedure),
        Error,
        "the matte logit (x=1, y=1) is non-finite",
      );
      assertThrows(
        () => matteFromLogits(Float32Array.of(-Infinity, 0, 0, 0), 2, 2, 2, 2, procedure),
        Error,
        "the matte logit (x=0, y=0) is non-finite",
      );
    }
  });
});
