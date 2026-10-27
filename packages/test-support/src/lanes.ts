/**
 * Test lanes — the split of the full suite into execution units (what `deno task test:core` and
 * `deno task test:models:<family>` used to define). Vitest runs each lane as a project:
 *
 *     vp test --project core                 # runtime / hub / models tests without local model assets
 *     vp test --project models:wan-ti2v      # one model family (real weights, long GPU runs)
 *     vp test                                # every project = the full suite
 *
 * `core` is everything the family lanes do not claim. A family lane claims every test whose file
 * name carries the family's token, in `packages/models/tests` (host-side tests of the family) and in
 * `packages/e2e/tests` (real-weight end-to-end tests). Lanes may overlap with each other (the QAT
 * lane shares one file with gemma4) but never with `core`; `verify_lanes_test.ts` checks the
 * coverage, the disjointness, that no lane is empty, and that no family row is dropped (core holds
 * only the GPU-free end-to-end gates; a family lane without end-to-end tests is named there).
 */

export type Lane = {
  readonly name: string;
  readonly include: readonly string[];
  readonly exclude: readonly string[];
};

const MODELS = "packages/models/tests";
const E2E = "packages/e2e/tests";

/** `*<token>*_test.ts` in both test directories. */
const both = (token: string): string[] => [
  `${MODELS}/*${token}*_test.ts`,
  `${E2E}/*${token}*_test.ts`,
];

const family = (
  name: string,
  include: readonly string[],
  exclude: readonly string[] = [],
): Lane => ({
  name: `models:${name}`,
  include,
  exclude,
});

export const FAMILY_LANES: readonly Lane[] = [
  family("anima", both("anima")),
  family("sbv2", both("sbv2")),
  family("irodori", both("irodori")),
  family("gemma4", both("gemma4"), both("gemma4_qat")),
  family("gemma4-qat", [...both("gemma4_qat"), `${E2E}/e2e_gemma4_ple_gpu_test.ts`]),
  family("birefnet", both("birefnet")),
  family("depth-anything", both("depth_anything")),
  family("siglip2", both("siglip2")),
  family("vowel-detector", both("vowel_detector")),
  family("minicpm5", both("minicpm5")),
  family("embeddinggemma", [`${E2E}/e2e_embeddinggemma*_test.ts`]),
  family("deberta", [`${E2E}/e2e_deberta*_test.ts`]),
  family("wan", both("wan"), both("wan_ti2v")),
  family("wan-ti2v", both("wan_ti2v")),
  family("umt5", both("umt5"), both("wan")),
  family("qwen3_5", both("qwen3_5")),
  family("moss", both("moss")),
];

/** Every test file of the workspace. */
export const ALL_TESTS = "{packages,tools}/*/tests/**/*_test.ts";

export const CORE_LANE: Lane = {
  name: "core",
  include: [ALL_TESTS],
  // Everything a family lane claims (the lanes' own excludes do not matter here: a file excluded
  // from one lane because another lane owns it is still not core).
  exclude: [...new Set(FAMILY_LANES.flatMap((lane) => lane.include))],
};

export const LANES: readonly Lane[] = [CORE_LANE, ...FAMILY_LANES];

/**
 * Families that are maintained but not run by default (maintainer ruling 2026-10-17): their
 * recipes, pipelines and tests stay, and `core` still excludes their files, but `vp test` runs
 * their lanes only when `KARUME_ALL_FAMILIES=1` is set (`KARUME_ALL_FAMILIES=1 vp test --project
 * models:siglip2`). The release run uses the default set.
 */
export const LOW_PRIORITY_FAMILIES: readonly string[] = ["minicpm5", "siglip2"];

/** The lanes a run without `KARUME_ALL_FAMILIES=1` executes. */
export const defaultLanes = (
  env: Readonly<Record<string, string | undefined>> = process.env,
): readonly Lane[] =>
  env.KARUME_ALL_FAMILIES === "1"
    ? LANES
    : LANES.filter((lane) => !LOW_PRIORITY_FAMILIES.includes(lane.name.replace(/^models:/, "")));

/** Vitest `test.projects` entries for the default lanes; every project inherits the root config. */
export const laneProjects = (
  env: Readonly<Record<string, string | undefined>> = process.env,
): {
  readonly extends: true;
  readonly test: { readonly name: string; readonly include: string[]; readonly exclude: string[] };
}[] =>
  defaultLanes(env).map((lane) => ({
    extends: true,
    test: { name: lane.name, include: [...lane.include], exclude: [...lane.exclude] },
  }));
