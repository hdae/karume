/**
 * The host side of the MOSS-Image DiT stage on the CPU (ADR 0138 decisions 2, 9 and 11, stage 2):
 * the acceptance check, the host functions against upstream's recorded values (Uint32-identical —
 * ADR 0138 decision 11), the graph contract of the distribution's DiT containers with its fault
 * injections, and the headroom arithmetic with the borrower's estimate.
 *
 * Sources of the expected values:
 *
 * - the stage-0 records (`goldens/moss-image-sft-stage0-reference/` of the test assets): `host.safetensors`
 *   (`moss.host_fixture` — upstream's RoPE tables, training grid, schedules and time sinusoid) and
 *   `dit-256x256.safetensors` / `dit-256x384.safetensors` (`moss.reference` — one upstream
 *   `pipe(...)` per size, 2 steps: the patchified tokens, the sinusoid and timestep of each step, the
 *   RoPE tables, both branches' samples and the prediction upstream passed to `scheduler.step`; the
 *   non-square size tells H from W);
 * - the stage-2 goldens (`moss.dit_reference`, read from the test-asset root
 *   `goldenDir("moss-image-sft-i8-dyn-reference")` — ADR 0139, fetched with
 *   `node tools/test-assets/fetch.ts`): the six graph inputs exactly as the recipe's
 *   `dit_patch.dit_inputs` built them (256² cases), the graph output before and after the recipe's
 *   `head_rows`, and `dit_text`'s RoPE inputs;
 * - the DiT containers of the distribution mirror `models/karume-moss-image/` (read through its
 *   manifest — byte copies of the series `moss-image-sft-i8-dyn` the goldens were taken from): the
 *   graph declarations and the assets `rope_base` / `training_grid`.
 *
 * Each group skips explicitly, naming the command that gets what it lacks (ADR 0005).
 */

import { describe, expect, it } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { parseSafetensors, type SafetensorsFile } from "@karume/runtime";
import { ModelInputError } from "../src/errors.ts";
import { HEADROOM_MARGIN_BYTES } from "../src/session/headroom-margin.ts";
import {
  MOSS_DIT_INPUTS,
  mossDitContract,
  mossDitGraphContract,
  type MossDitGraphs,
  readMossDitAssets,
} from "../src/moss/dit-contract.ts";
import {
  mossRopeTables,
  mossRopeWidth,
  mossTextRopeTables,
  parseMossRopeBase,
} from "../src/moss/dit-rope.ts";
import {
  mossTimestep,
  mossTimestepsProj,
  parseMossTrainingGrid,
} from "../src/moss/dit-timestep.ts";
import {
  mossHeadRows,
  mossIsImage,
  mossLatentTokens,
  mossTextContext,
} from "../src/moss/dit-tokens.ts";
import {
  mossClassifierFreeGuidance,
  mossDitStageNeeds,
  mossTextStageInputs,
  planMossDit,
} from "../src/moss/dit-stage.ts";
import { MOSS_PRACTICAL_PRESET, MOSS_REFERENCE_PRESET, mossPreset } from "../src/moss/preset.ts";
import {
  assertMossImageSize,
  assertMossTextEmbeddings,
  MOSS_MAX_SIDE_PX,
  MOSS_MAX_TEXT_TOKENS,
} from "../src/moss/request.ts";
import {
  assertTestAssetsPresent,
  boolsOf,
  containerDigests,
  firstBitMismatch,
  floatsOf,
  goldenUrl,
  intOf,
  loadCaseGolden,
  MOSS_DIT_CASES,
  MOSS_FETCH_GOLDENS,
  MOSS_GET_MIRROR,
  MOSS_REFERENCE_DIR,
  MOSS_TEXT_CASES,
  type MossDitCase,
  mossMirrorPresent,
  type MossTextCase,
  openMossDit,
  readBuffer,
  readRecordTensors,
  stage0Record,
  testAssetsRunnable,
} from "./helpers/moss-dit.ts";
import { buildSafetensors, f32Bytes } from "./helpers/safetensors.ts";

const HOST_RECORD = stage0Record("host.safetensors");
/** Present, or absent from a complete fetch of the pin (then the group FAILs naming it). */
const HOST_PRESENT = testAssetsRunnable([HOST_RECORD]);
const MIRROR_PRESENT = mossMirrorPresent();

/** The stage-0 DiT records (`moss.reference.SIZES`): latent cells and both branches' T. */
const DIT_RECORDS = [
  { file: "dit-256x256.safetensors", cells: { rows: 16, cols: 16 } },
  { file: "dit-256x384.safetensors", cells: { rows: 16, cols: 24 } },
] as const;
const BRANCH_TOKENS = [
  ["positive", 63],
  ["negative", 10],
] as const;

const SMALL_CASES = MOSS_DIT_CASES.filter((testCase) => testCase.group === "small");
const FULL_CASES = MOSS_DIT_CASES.filter((testCase) => testCase.group === "full");
/** Present, or absent from a complete fetch of the pin (then `caseGoldens` FAILs naming it). */
const goldensRunnable = (cases: readonly { readonly name: string }[]): boolean =>
  testAssetsRunnable(cases.map(({ name }) => goldenUrl(name)));
const SMALL_GOLDENS = goldensRunnable(SMALL_CASES);
const FULL_GOLDENS = goldensRunnable(FULL_CASES);
const TEXT_GOLDENS = goldensRunnable(MOSS_TEXT_CASES);

const skipped = (present: boolean, what: string, command: string): string =>
  present ? "" : ` (SKIP: no ${what} — ${command})`;

/** The RoPE cases of the host record (`moss.host_fixture.ROPE_CASES` — `(H, W, T)` in cells). */
const ROPE_CASES = [
  [16, 16, 10],
  [16, 24, 11],
  [24, 16, 10],
  [15, 17, 12],
  [17, 14, 13],
  [1, 1, 1],
  [64, 64, 10],
] as const;

const recordFloats = (file: SafetensorsFile, key: string): Float32Array<ArrayBuffer> =>
  floatsOf(file, key, "host record");

const embeddings = (tokens: number, fill = 0.5) => ({
  data: new Float32Array(tokens * 4096).fill(fill),
  shape: [tokens, 4096],
});

/**
 * Upstream's `[S, 128]` table `[sin 64 | cos 64]` against the repeated halves of the graph inputs:
 * row s of `rope_sin` is `[sin, sin]`, of `rope_cos` `[cos, cos]`.
 */
const assertRopeMatchesUpstream = (
  tables: { readonly cos: Float32Array; readonly sin: Float32Array },
  upstream: Float32Array,
  rows: number,
  where: string,
): void => {
  assertEquals(tables.sin.length, rows * 128, `${where}: rope_sin length`);
  const half = 64;
  const expectedSin = new Float32Array(rows * 128);
  const expectedCos = new Float32Array(rows * 128);
  for (let row = 0; row < rows; row += 1) {
    const sin = upstream.subarray(row * 128, row * 128 + half);
    const cos = upstream.subarray(row * 128 + half, (row + 1) * 128);
    expectedSin.set(sin, row * 128);
    expectedSin.set(sin, row * 128 + half);
    expectedCos.set(cos, row * 128);
    expectedCos.set(cos, row * 128 + half);
  }
  assertEquals(firstBitMismatch(tables.sin, expectedSin), -1, `${where}: rope_sin bits`);
  assertEquals(firstBitMismatch(tables.cos, expectedCos), -1, `${where}: rope_cos bits`);
};

describe("MOSS-Image acceptance check (stage 2, ADR 0138 decision 9)", () => {
  describe("the image size", () => {
    it("accepts positive multiples of 16 and returns the latent grid (H/16 × W/16)", () => {
      expect(assertMossImageSize(1024, 768)).toEqual({ rows: 48, cols: 64 });
      expect(assertMossImageSize(16, 16)).toEqual({ rows: 1, cols: 1 });
    });

    it("accepts the longest side at exactly 65,536 px", () => {
      expect(assertMossImageSize(MOSS_MAX_SIDE_PX, 16)).toEqual({ rows: 1, cols: 4096 });
    });

    it("rejects a side that is not a positive multiple of 16, naming the side", () => {
      for (const [width, height, message] of [
        [1000, 1024, "width 1000 px is not a positive multiple of 16"],
        [1024, 0, "height 0 px is not a positive multiple of 16"],
        [-16, 1024, "width -16 px is not a positive multiple of 16"],
        [1024, 16.5, "height 16.5 px is not a positive multiple of 16"],
      ] as const) {
        assertThrows(() => assertMossImageSize(width, height), ModelInputError, message);
      }
    });

    it("rejects a longest side over 65,536 px with an error that names the limit", () => {
      for (const [width, height] of [
        [MOSS_MAX_SIDE_PX + 16, 16],
        [16, MOSS_MAX_SIDE_PX + 16],
      ]) {
        assertThrows(
          () => assertMossImageSize(width, height),
          ModelInputError,
          `the longest side ${MOSS_MAX_SIDE_PX + 16} px exceeds the accepted 65536 px (4096 latent cells`,
        );
      }
    });
  });

  describe("the text embeddings", () => {
    it("accepts f32 [T, 4096] with 1 ≤ T ≤ 4,096 and returns T", () => {
      expect(assertMossTextEmbeddings(embeddings(1), "positive")).toBe(1);
      expect(assertMossTextEmbeddings(embeddings(MOSS_MAX_TEXT_TOKENS), "positive")).toBe(4096);
    });

    it("rejects each malformed embedding with its own message", () => {
      // A caller outside the types (JavaScript, a cast) can pass another array — the run-time check.
      const f64 = new Float64Array(4096) as unknown as Float32Array<ArrayBuffer>;
      const cases: readonly [Parameters<typeof assertMossTextEmbeddings>[0], string][] = [
        [{ data: f64, shape: [1, 4096] }, "are not a Float32Array (f32 values are required)"],
        [{ data: new Float32Array(0), shape: [0, 4096] }, "have 0 tokens (at least 1 is required)"],
        [
          { data: new Float32Array(4096), shape: [2.5, 4096] },
          "have a token count 2.5 that is not an integer",
        ],
        [embeddings(4097), "have 4097 tokens, over the accepted 4096"],
        [
          { data: new Float32Array(10 * 4095), shape: [10, 4095] },
          "have shape [10, 4095] (expected [T, 4096])",
        ],
        [{ data: new Float32Array(4096), shape: [1, 1, 4096] }, "have shape [1, 1, 4096]"],
        [
          { data: new Float32Array(4096), shape: [2, 4096] },
          "hold 4096 values for shape [2, 4096]",
        ],
      ];
      for (const [input, message] of cases) {
        assertThrows(() => assertMossTextEmbeddings(input, "negative"), ModelInputError, message);
      }
    });

    it("rejects a non-finite value, naming its row and column", () => {
      const input = embeddings(3);
      input.data[2 * 4096 + 17] = Number.NaN;

      assertThrows(
        () => assertMossTextEmbeddings(input, "positive"),
        ModelInputError,
        "the positive text embeddings: row 2, column 17 is non-finite (NaN)",
      );
    });
  });

  it("plans one branch when there is no negative text (guidance 1)", () => {
    const plan = planMossDit({
      width: 256,
      height: 512,
      positive: embeddings(5),
      negative: undefined,
    });

    expect(plan.cells).toEqual({ rows: 32, cols: 16 });
    expect(plan.texts.positive.tokens).toBe(5);
    expect(plan.texts.negative).toBeUndefined();
  });
});

describe("MOSS-Image host CFG (f32 — upstream's formula)", () => {
  it("is element-wise uncond + scale · (cond − uncond) in f32", () => {
    const cond = Float32Array.of(1, 2.5, -3);
    const uncond = Float32Array.of(0.5, 2.5, 1);

    expect([...mossClassifierFreeGuidance(cond, uncond, 3)]).toEqual([2, 2.5, -11]);
  });

  it("refuses samples of different lengths", () => {
    assertThrows(
      () => mossClassifierFreeGuidance(new Float32Array(2), new Float32Array(3), 3),
      Error,
      "CFG of 2 and 3 values",
    );
  });

  for (const { file } of DIT_RECORDS) {
    const url = stage0Record(file);
    const present = testAssetsRunnable([url]);
    describe.skipIf(!present)(
      `against the stage-0 record ${file}${skipped(present, "stage-0 DiT record", MOSS_FETCH_GOLDENS)}`,
      () => {
        // `stepN.prediction` is the model_output upstream passed to scheduler.step (recorded by
        // wrapping it; `moss.reference`'s gate checks it against its CFG line).
        it("reproduces the prediction upstream passed to scheduler.step at both steps bit for bit (guidance 3)", async () => {
          assertTestAssetsPresent([url]);
          for (const step of [0, 1]) {
            const record = await readRecordTensors(url, [
              `step${step}.positive.sample`,
              `step${step}.negative.sample`,
              `step${step}.prediction`,
            ]);
            const prediction = mossClassifierFreeGuidance(
              record[`step${step}.positive.sample`],
              record[`step${step}.negative.sample`],
              3,
            );

            assertEquals(
              firstBitMismatch(prediction, record[`step${step}.prediction`]),
              -1,
              `step ${step}`,
            );
          }
        });
      },
    );
  }
});

describe.skipIf(!MIRROR_PRESENT || !HOST_PRESENT)(
  `MOSS-Image host tables from the mirror's assets${skipped(MIRROR_PRESENT, "mirror models/karume-moss-image/", MOSS_GET_MIRROR)}${skipped(HOST_PRESENT, "stage-0 host record", MOSS_FETCH_GOLDENS)}`,
  () => {
    const load = async () => {
      assertTestAssetsPresent([HOST_RECORD]);
      const series = await openMossDit();
      const assets = await readMossDitAssets(series.components.dit);
      const host = parseSafetensors(await readBuffer(HOST_RECORD));
      return { series, assets, host };
    };

    it("reads rope_base as 16 bands covering 4,096 cells per side and 4,096 text tokens", async () => {
      const { assets } = await load();

      expect(assets.ropeBase.bands).toBe(16);
      expect(assets.ropeBase.maxSide).toBe(4096);
      expect(assets.ropeBase.maxText).toBe(4096);
      expect(mossRopeWidth(assets.ropeBase)).toBe(128);
    });

    it("assembles upstream's RoPE table bit for bit for every recorded (H, W, T)", async () => {
      const { assets, host } = await load();
      for (const [height, width, tokens] of ROPE_CASES) {
        const where = `rope.${height}x${width}.t${tokens}`;
        const tables = mossRopeTables(assets.ropeBase, { rows: height, cols: width }, tokens);

        assertRopeMatchesUpstream(
          tables,
          recordFloats(host, where),
          height * width + tokens,
          where,
        );
      }
    });

    it("builds dit_text's tables as the text rows of the joint table", async () => {
      const { assets } = await load();
      const joint = mossRopeTables(assets.ropeBase, { rows: 3, cols: 5 }, 9);
      const text = mossTextRopeTables(assets.ropeBase, 9);

      assertEquals(firstBitMismatch(text.cos, joint.cos.subarray(15 * 128)), -1);
      assertEquals(firstBitMismatch(text.sin, joint.sin.subarray(15 * 128)), -1);
    });

    it("refuses a side or a text the base tables do not cover", async () => {
      const { assets } = await load();

      assertThrows(
        () => mossRopeTables(assets.ropeBase, { rows: 4097, cols: 1 }, 1),
        Error,
        "outside 1 … 4096",
      );
      assertThrows(
        () => mossRopeTables(assets.ropeBase, { rows: 1, cols: 1 }, 4097),
        Error,
        "text length 4097",
      );
      assertThrows(() => mossTextRopeTables(assets.ropeBase, 0), Error, "text length 0");
    });

    it("holds upstream's training grid and time sinusoid bit for bit", async () => {
      const { assets, host } = await load();
      const grid = assets.trainingGrid;

      expect(grid.points).toBe(1000);
      expect(grid.width).toBe(256);
      assertEquals(firstBitMismatch(grid.sigmas, recordFloats(host, "training.sigmas")), -1);
      assertEquals(firstBitMismatch(grid.timesteps, recordFloats(host, "training.timesteps")), -1);
      assertEquals(firstBitMismatch(grid.sinusoid, recordFloats(host, "time_sinusoid")), -1);
    });

    it("looks up every scheduled timestep of the recorded schedules by its grid index", async () => {
      const { assets, host } = await load();
      for (const steps of [1, 2, 50, 200, 1000]) {
        const indices = host.tensors.get(`steps.${steps}.grid`);
        assert(indices !== undefined && indices.dtype === "I32");
        const grid = new Int32Array(host.buffer, indices.byteOffset, indices.byteLength / 4);
        const timesteps = Float32Array.from(grid, (index) =>
          mossTimestep(assets.trainingGrid, index),
        );

        assertEquals(
          firstBitMismatch(timesteps, recordFloats(host, `steps.${steps}.timesteps`)),
          -1,
          `${steps} steps`,
        );
      }
    });

    it("refuses a grid index outside 0 … 999", async () => {
      const { assets } = await load();
      for (const index of [-1, 1000, 1.5]) {
        assertThrows(() => mossTimestepsProj(assets.trainingGrid, index), Error, "outside 0 … 999");
      }
    });

    for (const { file, cells } of DIT_RECORDS) {
      const url = stage0Record(file);
      const present = testAssetsRunnable([url]);
      const image = cells.rows * cells.cols;
      describe.skipIf(!present)(
        `against upstream's own forward (${file})${skipped(present, "stage-0 DiT record", MOSS_FETCH_GOLDENS)}`,
        () => {
          it("feeds the sinusoid and the timestep upstream fed at each step of the 2-step schedule", async () => {
            assertTestAssetsPresent([url]);
            const { assets, host } = await load();
            const indices = host.tensors.get("steps.2.grid");
            assert(indices !== undefined);
            const grid = new Int32Array(host.buffer, indices.byteOffset, 2);
            for (const step of [0, 1]) {
              const record = await readRecordTensors(url, [
                `step${step}.time_sinusoid`,
                `step${step}.timestep`,
              ]);

              assertEquals(
                firstBitMismatch(
                  mossTimestepsProj(assets.trainingGrid, grid[step]),
                  record[`step${step}.time_sinusoid`],
                ),
                -1,
              );
              assertEquals(
                firstBitMismatch(
                  Float32Array.of(mossTimestep(assets.trainingGrid, grid[step])),
                  record[`step${step}.timestep`],
                ),
                -1,
              );
            }
          });

          it("assembles the RoPE tables upstream's forward used for both branches (T = 63 and 10)", async () => {
            assertTestAssetsPresent([url]);
            const { assets } = await load();
            for (const [branch, tokens] of BRANCH_TOKENS) {
              const record = await readRecordTensors(url, [`${branch}.rope`]);
              const tables = mossRopeTables(assets.ropeBase, cells, tokens);

              assertRopeMatchesUpstream(
                tables,
                record[`${branch}.rope`],
                image + tokens,
                `${branch}.rope`,
              );
            }
          });

          it("patchifies the latents into upstream's video tokens (h-major) with zero text rows", async () => {
            assertTestAssetsPresent([url]);
            for (const step of [0, 1]) {
              const record = await readRecordTensors(url, [
                `step${step}.latents_in`,
                `step${step}.video_tokens`,
              ]);
              const tokens = mossLatentTokens(record[`step${step}.latents_in`], 128, cells, 10);

              assertEquals(
                firstBitMismatch(
                  tokens.subarray(0, image * 128),
                  record[`step${step}.video_tokens`],
                ),
                -1,
              );
              assert(
                tokens.subarray(image * 128).every((value) => Object.is(value, 0)),
                "text rows are +0",
              );
            }
          });
        },
      );
    }
  },
);

/**
 * A loader of case goldens that refuses absent ones (`assertTestAssetsPresent`), stale ones and ones
 * of another case (`loadCaseGolden`); the container digests are read once per group.
 */
const caseGoldens = () => {
  let digests: Promise<Readonly<Record<string, string>>> | undefined;
  return async (testCase: MossDitCase | MossTextCase) => {
    assertTestAssetsPresent([goldenUrl(testCase.name)]);
    digests ??= containerDigests();
    return loadCaseGolden(testCase, await digests);
  };
};

/** The latent side of a DiT group. */
const GROUP_SIDE = { small: 16, full: 64 } as const;

const headRowsCases = (cases: readonly MossDitCase[]) => {
  const golden = caseGoldens();
  it.each(cases.map((testCase) => [testCase.name, testCase] as const))(
    "%s: the head rows are dit_patch.head_rows bit for bit",
    async (name, testCase) => {
      const { file, metadata } = await golden(testCase);
      const side = GROUP_SIDE[testCase.group];
      const tokens = Number(metadata["T"]);

      const sample = mossHeadRows(
        floatsOf(file, "output.rows", name),
        128,
        { rows: side, cols: side },
        tokens,
      );

      assertEquals(firstBitMismatch(sample, floatsOf(file, "output", name)), -1);
    },
  );
};

describe.skipIf(!SMALL_GOLDENS || !MIRROR_PRESENT)(
  `MOSS-Image host functions against the recipe's dit_patch at 256²${skipped(SMALL_GOLDENS, `256² goldens in ${MOSS_REFERENCE_DIR.pathname}`, MOSS_FETCH_GOLDENS)}${skipped(MIRROR_PRESENT, "mirror models/karume-moss-image/", MOSS_GET_MIRROR)}`,
  () => {
    const golden = caseGoldens();
    it.each(SMALL_CASES.map((testCase) => [testCase.name, testCase] as const))(
      "%s: the six graph inputs are dit_patch.dit_inputs bit for bit",
      async (name, testCase) => {
        const series = await openMossDit();
        const assets = await readMossDitAssets(series.components.dit);
        const { file } = await golden(testCase);
        const latents = floatsOf(file, "latents", name);
        const context = floatsOf(file, "text_context", name);
        const tokens = context.length / 3072;
        const cells = { rows: 16, cols: 16 };
        const rope = mossRopeTables(assets.ropeBase, cells, tokens);
        const built: Readonly<Record<string, Float32Array | Uint32Array>> = {
          [MOSS_DIT_INPUTS.latentTokens]: mossLatentTokens(latents, 128, cells, tokens),
          [MOSS_DIT_INPUTS.textContext]: mossTextContext(context, 3072, cells, tokens),
          [MOSS_DIT_INPUTS.isImage]: mossIsImage(cells, tokens),
          [MOSS_DIT_INPUTS.ropeCos]: rope.cos,
          [MOSS_DIT_INPUTS.ropeSin]: rope.sin,
          [MOSS_DIT_INPUTS.timestepsProj]: mossTimestepsProj(
            assets.trainingGrid,
            intOf(file, "grid_index", name),
          ),
        };
        for (const [input, value] of Object.entries(built)) {
          const expected =
            input === MOSS_DIT_INPUTS.isImage
              ? boolsOf(file, `input.${input}`, name)
              : floatsOf(file, `input.${input}`, name);
          assertEquals(firstBitMismatch(value, expected), -1, `${name}: ${input}`);
        }
      },
    );

    headRowsCases(SMALL_CASES);
  },
);

describe.skipIf(!FULL_GOLDENS || !MIRROR_PRESENT)(
  `MOSS-Image head rows against the recipe's dit_patch at 1024²${skipped(FULL_GOLDENS, `1024² goldens in ${MOSS_REFERENCE_DIR.pathname}`, MOSS_FETCH_GOLDENS)}${skipped(MIRROR_PRESENT, "mirror models/karume-moss-image/", MOSS_GET_MIRROR)}`,
  () => {
    headRowsCases(FULL_CASES);
  },
);

describe.skipIf(!TEXT_GOLDENS || !MIRROR_PRESENT)(
  `MOSS-Image dit_text inputs against the recipe's${skipped(TEXT_GOLDENS, `text goldens in ${MOSS_REFERENCE_DIR.pathname}`, MOSS_FETCH_GOLDENS)}${skipped(MIRROR_PRESENT, "mirror models/karume-moss-image/", MOSS_GET_MIRROR)}`,
  () => {
    const golden = caseGoldens();
    it.each(MOSS_TEXT_CASES.map((testCase) => [testCase.name, testCase] as const))(
      "%s: dit_text's inputs are the recipe's",
      async (name, testCase) => {
        const { tokens } = testCase;
        const series = await openMossDit();
        const assets = await readMossDitAssets(series.components.dit);
        const contract = mossDitContract(
          mossDitGraphContract(series.components, "test"),
          assets.ropeBase,
          assets.trainingGrid,
          "test",
        );
        const { file } = await golden(testCase);
        const embeds = floatsOf(file, "text_embeds", name);

        const inputs = mossTextStageInputs(contract, {
          embeddings: { data: embeds, shape: [tokens, 4096] },
          tokens,
        });

        for (const [input, key] of [
          ["rope_cos", "rope_cos"],
          ["rope_sin", "rope_sin"],
          ["text_embeds", "text_embeds"],
        ] as const) {
          const tensor = inputs[input];
          assert("dtype" in tensor && tensor.dtype === "f32");
          assertEquals(
            firstBitMismatch(tensor.data, floatsOf(file, key, name)),
            -1,
            `${name}: ${input}`,
          );
        }
      },
    );
  },
);

describe("MOSS-Image asset parsers refuse malformed tables", () => {
  const ropeSpecs = (overrides: Partial<Record<string, number>> = {}) =>
    ["r", "f", "h", "w"].flatMap((axis) =>
      ["sin", "cos"].map((kind) => {
        const rows = overrides[axis] ?? { r: 1, f: 5, h: 7, w: 7 }[axis] ?? 1;
        return {
          name: `${kind}_${axis}`,
          dtype: "F32",
          shape: [rows, 2],
          data: f32Bytes(new Float32Array(rows * 2)),
        };
      }),
    );

  it("reads a well-formed rope base", () => {
    const base = parseMossRopeBase(buildSafetensors(ropeSpecs()));

    expect([base.bands, base.maxText, base.maxSide]).toEqual([2, 4, 4]);
  });

  it("refuses an r table of more than one row and h / w tables of different or even counts", () => {
    assertThrows(
      () => parseMossRopeBase(buildSafetensors(ropeSpecs({ r: 2 }))),
      Error,
      "r: 2 rows",
    );
    assertThrows(
      () => parseMossRopeBase(buildSafetensors(ropeSpecs({ w: 5 }))),
      Error,
      "h / w: 7 / 5 rows",
    );
    assertThrows(
      () => parseMossRopeBase(buildSafetensors(ropeSpecs({ h: 8, w: 8 }))),
      Error,
      "one odd count",
    );
  });

  it("refuses an unexpected rope table", () => {
    const specs = [
      ...ropeSpecs(),
      { name: "sin_t", dtype: "F32", shape: [1, 2], data: f32Bytes([0, 0]) },
    ];

    assertThrows(() => parseMossRopeBase(buildSafetensors(specs)), Error, "unexpected 'sin_t'");
  });

  const gridSpecs = (sigmas: readonly number[]) => [
    { name: "sigmas", dtype: "F32", shape: [sigmas.length], data: f32Bytes(sigmas) },
    {
      name: "timesteps",
      dtype: "F32",
      shape: [sigmas.length],
      data: f32Bytes(sigmas.map((s) => s * 1000)),
    },
    {
      name: "time_sinusoid",
      dtype: "F32",
      shape: [sigmas.length, 2],
      data: f32Bytes(new Float32Array(sigmas.length * 2)),
    },
  ];

  it("reads a training grid descending from 1", () => {
    const grid = parseMossTrainingGrid(buildSafetensors(gridSpecs([1, 0.5, 0.25])));

    expect([grid.points, grid.width]).toEqual([3, 2]);
  });

  it("refuses a training grid that is not strictly descending from 1", () => {
    for (const sigmas of [
      [0.5, 1, 0.25],
      [1, 0.5, 0.5],
      [0.9, 0.5, 0.25],
    ]) {
      assertThrows(
        () => parseMossTrainingGrid(buildSafetensors(gridSpecs(sigmas))),
        Error,
        "strictly descending",
      );
    }
  });

  it("refuses a training grid without its sinusoid", () => {
    const specs = gridSpecs([1, 0.5]).filter(({ name }) => name !== "time_sinusoid");

    assertThrows(
      () => parseMossTrainingGrid(buildSafetensors(specs)),
      Error,
      "holds [sigmas, timesteps]",
    );
  });
});

describe.skipIf(!MIRROR_PRESENT)(
  `MOSS-Image graph contract of the mirror's DiT containers${skipped(MIRROR_PRESENT, "mirror models/karume-moss-image/", MOSS_GET_MIRROR)}`,
  () => {
    it("accepts the three graphs and settles C, D, R, P and the borrowed weights", async () => {
      const series = await openMossDit();
      const assets = await readMossDitAssets(series.components.dit);

      const contract = mossDitContract(
        mossDitGraphContract(series.components, "test"),
        assets.ropeBase,
        assets.trainingGrid,
        "test",
      );

      expect([
        contract.latentChannels,
        contract.width,
        contract.ropeWidth,
        contract.projWidth,
      ]).toEqual([128, 3072, 128, 256]);
      // Every initializer of dit but block 22's (the hoisted constants included).
      const lent = Object.keys(series.components.dit.graph.initializers);
      const block22 = lent.filter((name) => name.startsWith("layers.22."));
      expect(block22.length).toBe(10);
      expect(new Set(contract.borrowed)).toEqual(
        new Set(lent.filter((name) => !name.startsWith("layers.22."))),
      );
    });

    /** The real graphs with one field of one graph replaced (the fault injections below). */
    const withGraph = async (
      which: keyof MossDitGraphs,
      edit: (graph: MossDitGraphs["dit"]["graph"]) => MossDitGraphs["dit"]["graph"],
    ): Promise<MossDitGraphs> => {
      const { components } = await openMossDit();
      const graphs = {
        dit: components.dit,
        ditSkip22: components.ditSkip22,
        ditText: components.ditText,
      };
      return { ...graphs, [which]: { ...graphs[which], graph: edit(graphs[which].graph) } };
    };

    it("names a missing input", async () => {
      const graphs = await withGraph("dit", (graph) => ({
        ...graph,
        inputs: graph.inputs.filter((input) => input.name !== "is_image"),
      }));

      assertThrows(
        () => mossDitGraphContract(graphs, "test"),
        Error,
        "test: dit inputs are [latent_tokens, text_context, rope_cos, rope_sin, timesteps_proj]",
      );
    });

    it("names an input of another shape", async () => {
      const graphs = await withGraph("ditText", (graph) => ({
        ...graph,
        inputs: graph.inputs.map((input) =>
          input.name === "text_embeds" ? { ...input, shape: [1, "T", 3584] } : input,
        ),
      }));

      assertThrows(
        () => mossDitGraphContract(graphs, "test"),
        Error,
        "dit_text input 'text_embeds' has shape [1, T, 3584] (the host builds [1, T, 4096])",
      );
    });

    it("names a skip graph that owns a weight", async () => {
      const graphs = await withGraph("ditSkip22", (graph) => {
        const [first] = Object.keys(graph.initializers);
        return {
          ...graph,
          initializers: { ...graph.initializers, [first]: { storage: { codec: "f32" } } },
        };
      });

      assertThrows(
        () => mossDitGraphContract(graphs, "test"),
        Error,
        "dit_skip22 owns 1 initializers",
      );
    });

    it("names a skip graph that borrows a weight dit does not lend", async () => {
      const graphs = await withGraph("ditSkip22", (graph) => ({
        ...graph,
        initializers: { ...graph.initializers, "layers.99.proj.q.weight": { shared: true } },
      }));

      assertThrows(
        () => mossDitGraphContract(graphs, "test"),
        Error,
        "have no weight of the same name in dit",
      );
    });

    it("names a missing or mis-roled asset", async () => {
      const { components } = await openMossDit();
      const without = {
        ...components,
        dit: { ...components.dit, assets: { rope_base: "rope-base" } },
      };
      const misroled = {
        ...components,
        dit: { ...components.dit, assets: { ...components.dit.assets, rope_base: "rope_base" } },
      };

      assertThrows(
        () => mossDitGraphContract(without, "test"),
        Error,
        "has no asset 'training_grid'",
      );
      assertThrows(
        () => mossDitGraphContract(misroled, "test"),
        Error,
        "has role 'rope_base' (expected 'rope-base')",
      );
    });

    it("names base tables that cover other limits than the accepted ones", async () => {
      const series = await openMossDit();
      const assets = await readMossDitAssets(series.components.dit);
      const graph = mossDitGraphContract(series.components, "test");

      assertThrows(
        () =>
          mossDitContract(
            graph,
            { ...assets.ropeBase, maxSide: 2048 },
            assets.trainingGrid,
            "test",
          ),
        Error,
        "covers 2048 cells per side and 4096 text tokens (the accepted 4096 and 4096)",
      );
    });
  },
);

describe.skipIf(!MIRROR_PRESENT)(
  `MOSS-Image headroom of the DiT stages${skipped(MIRROR_PRESENT, "mirror models/karume-moss-image/", MOSS_GET_MIRROR)}`,
  () => {
    /** The 3080 Ti's limits (any device's would do — the estimate is a pure function of them). */
    const limits = { maxStorageBufferBindingSize: 2_147_483_644, maxBufferSize: 2_147_483_648 };
    /** A device with the online kernel's subgroup form (the 3080 Ti's facts). */
    const device = {
      limits,
      attention: { subgroups: true, maxComputeWorkgroupStorageSize: 49_152 },
    };
    const reference = mossPreset(MOSS_REFERENCE_PRESET);

    it("counts no weight in the borrower's estimate and every weight in the lender's", async () => {
      const { components } = await openMossDit();

      const lender = components.dit.estimate({ ...limits, bindings: { S: 4159 } });
      const borrower = components.ditSkip22.estimate({ ...limits, bindings: { S: 4106 } });
      console.log(
        `[moss-dit] 1024² estimates: dit weights ${lender.resident.weights.totalBytes} B` +
          ` (compressed ${lender.resident.weights.compressedBytes} B), peak ${lender.peakAccountedBytes} B;` +
          ` dit_skip22 weights ${borrower.resident.weights.totalBytes} B, peak ${borrower.peakAccountedBytes} B`,
      );

      expect(borrower.resident.weights.totalBytes).toBe(0);
      // 4.73 GiB of i8 linears minus dit_text's 0.22 GiB, plus the scales and the f32 norms.
      expect(lender.resident.weights.totalBytes).toBeGreaterThan(4 * 2 ** 30);
      expect(borrower.peakAccountedBytes).toBeLessThan(lender.resident.weights.totalBytes);
    });

    it("sums both Sessions of the pair, counting the weights once, plus the two text contexts", async () => {
      const series = await openMossDit();
      const assets = await readMossDitAssets(series.components.dit);
      const contract = mossDitContract(
        mossDitGraphContract(series.components, "test"),
        assets.ropeBase,
        assets.trainingGrid,
        "test",
      );
      const state = { device, components: series.components, contract, preset: reference };
      const plan = planMossDit({
        width: 1024,
        height: 1024,
        positive: embeddings(63),
        negative: embeddings(10),
      });
      const single = planMossDit({
        width: 1024,
        height: 1024,
        positive: embeddings(63),
        negative: undefined,
      });
      const borrower = series.components.ditSkip22.estimate({ ...limits, bindings: { S: 4106 } });

      const pair = mossDitStageNeeds(state, plan).pair;
      const alone = mossDitStageNeeds(state, single).pair;
      console.log(
        `[moss-dit] 1024² pair need ${pair} B (positive alone ${alone} B; borrower peak` +
          ` ${borrower.peakAccountedBytes} B, weights ${borrower.resident.weights.totalBytes} B)`,
      );

      expect(pair - alone).toBe(borrower.peakAccountedBytes + 4106 * 3072 * 4);
      // The largest part is the lender's: the borrower has no weight part (all its weights are
      // borrowed), so `Math.max` over the two parts is not exercised by this series.
      const lender = series.components.dit.estimate({ ...limits, bindings: { S: 4159 } });
      expect(alone).toBe(
        lender.peakAccountedBytes +
          4159 * 3072 * 4 +
          series.components.dit.maxPartBytes +
          HEADROOM_MARGIN_BYTES,
      );
    });

    it("estimates the pair with the practical preset's knobs and the device's attention facts, and dit_text with none", async () => {
      const series = await openMossDit();
      const assets = await readMossDitAssets(series.components.dit);
      const contract = mossDitContract(
        mossDitGraphContract(series.components, "test"),
        assets.ropeBase,
        assets.trainingGrid,
        "test",
      );
      const practical = mossPreset(MOSS_PRACTICAL_PRESET);
      const plan = planMossDit({
        width: 1024,
        height: 1024,
        positive: embeddings(63),
        negative: embeddings(10),
      });
      const knobs = { ...limits, ...practical.dit, attentionDevice: device.attention };
      const lender = series.components.dit.estimate({ ...knobs, bindings: { S: 4159 } });
      const borrower = series.components.ditSkip22.estimate({ ...knobs, bindings: { S: 4106 } });
      const components = series.components;

      const needs = mossDitStageNeeds({ device, components, contract, preset: practical }, plan);
      const referenceNeeds = mossDitStageNeeds(
        { device, components, contract, preset: reference },
        plan,
      );
      // A device with neither form of the online kernel (no subgroups, 32 KiB of workgroup
      // storage): its route is the three dispatches, which store S.
      const portable = {
        limits,
        attention: { subgroups: false, maxComputeWorkgroupStorageSize: 32_768 },
      };
      const fallback = mossDitStageNeeds(
        { device: portable, components, contract, preset: practical },
        plan,
      );
      console.log(
        `[moss-dit] 1024² pair need: ${practical.name} ${needs.pair} B, ${reference.name}` +
          ` ${referenceNeeds.pair} B, ${practical.name} without an online form ${fallback.pair} B`,
      );

      expect(needs.pair).toBe(
        lender.peakAccountedBytes +
          borrower.peakAccountedBytes +
          (4159 + 4106) * 3072 * 4 +
          components.dit.maxPartBytes +
          HEADROOM_MARGIN_BYTES,
      );
      // The knobs and the device facts reach the estimates (each changes the need).
      expect(needs.pair).toBeLessThan(referenceNeeds.pair);
      expect(needs.pair).toBeLessThan(fallback.pair);
      // dit_text runs f32 in every preset.
      expect(needs.text).toBe(referenceNeeds.text);
    });

    it("sizes the text stage at the longer branch, whichever it is", async () => {
      const series = await openMossDit();
      const assets = await readMossDitAssets(series.components.dit);
      const contract = mossDitContract(
        mossDitGraphContract(series.components, "test"),
        assets.ropeBase,
        assets.trainingGrid,
        "test",
      );
      const state = { device, components: series.components, contract, preset: reference };
      const estimate = (tokens: number) =>
        series.components.ditText.estimate({ ...limits, bindings: { T: tokens } });
      // Precondition: the two lengths have different estimates (below some T the plan backing
      // budget is the whole scenario term, so short texts all estimate the same).
      expect(estimate(7).peakAccountedBytes).toBeLessThan(estimate(4096).peakAccountedBytes);
      const expected =
        estimate(4096).peakAccountedBytes +
        series.components.ditText.maxPartBytes +
        HEADROOM_MARGIN_BYTES;

      for (const [positive, negative] of [
        [7, 4096],
        [4096, 7],
      ]) {
        const plan = planMossDit({
          width: 256,
          height: 256,
          positive: embeddings(positive),
          negative: embeddings(negative),
        });

        expect(mossDitStageNeeds(state, plan).text).toBe(expected);
      }
    });
  },
);
