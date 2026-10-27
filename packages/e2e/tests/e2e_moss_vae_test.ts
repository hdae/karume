/**
 * The MOSS-Image VAE stage on the GPU (ADR NNNN decisions 5 and 8, plan step 3): the untiled FLUX.2
 * decoder as three Sessions in one batch against upstream's f32 decode, the census, the lifetimes,
 * the headroom check, the per-device area limit and (opt-in) the area-cap decode with its VRAM.
 *
 * The goldens are written by `tools/export-recipes/moss/export_vae.py` with the series (one staged
 * set): per golden the DiT-space `latents`, the decode input `z`, upstream's untiled f32 decode
 * `decoded` (CPU) and the clamped `image`. The stage starts from `latents` (the host's BatchNorm
 * restoration and unpatchify are in the run; `moss_vae_host_test.ts` checks them Uint32-identical).
 *
 * ## The band (ADR 0118 decision 8's upstream check of a VAE — opt-in `KARUME_UPSTREAM_PARITY=1`)
 *
 * This is a port check of the reference path against upstream (ADR 0118 decision 8: an f32
 * reference, the ratio m, about 5 × the measured worst, band cases separate from the acceptance
 * cases), not an E2E A/B check of a practical preset (ADR 0110 decision 4's MUST NOT on "5–10 ×
 * the measured" is about those) and not ADR 0120 decision 3's r check (whose six decision cases
 * belong to the DiT's f64 reference — ADR NNNN decision 11's VAE tier).
 *
 * The metric m = max |GPU − upstream| / max |upstream| over the clamped image: the stage's output
 * against the goldens' `image` (upstream's f32 decode clamped as upstream clamps; no golden pixel
 * saturates — max |decoded| is 1.042 and the share of clamped pixels is 0.0000 — so the clamp
 * hides nothing of the difference). relRMS is reported beside it. Per size the band is the worst m
 * of the three band cases × 5, rounded up to two significant digits; the two acceptance cases (one
 * is the stage-0 record's latents) must stay inside it. The 256² band is not carried to 512×768
 * (ADR 0118 decision 8). A fault (the BatchNorm's `mean` and `std` swapped, the latents of
 * another seed, the unpatchify's 2×2 patch transposed) must land outside the band by a margin of 2
 * (ADR 0118 addendum). The band does not see a one-pass GroupNorm variance (random latents keep
 * the decoder's GroupNorm inputs out of the cancellation range — measured 3.0e-6 … 9.2e-6 with the
 * fault in the kernel, in the step-3 review); that property is held by `gpu_group_norm_test.ts` ① and ④ alone.
 *
 * Derivation (2026-10-27, RTX 3080 Ti, Node + Dawn — see {@link BANDS}).
 *
 * ## The area cap (opt-in `KARUME_UPSTREAM_PARITY=1` + `KARUME_MOSS_VAE_CAP=1`)
 *
 * 1696×1648 (2,795,008 px — near the largest area of this device's 2 GiB binding): the stage's
 * headroom need against the device, the decode against upstream's CPU decode (`write --cap`), and
 * the VRAM the three Sessions hold (their plan backings with their input buffers and weights, the
 * residents) — the plan's estimate was 5.37 GB (§1.5); then the largest accepted area of this
 * device decodes finite through the product's own headroom check. The process-level peak is
 * sampled outside the test (nvidia-smi; the raw samples and the test's output are kept in
 * `outputs/bench/karume-moss-image/2026-10-27_vae-cap-vram/`). The stage's peak on the device is
 * max(the trial allocation of the headroom need, what the stage holds) — the trial allocation
 * comes first and is the larger one at the cap.
 *
 * The default run (no opt-in): one 256² decode (finite, the census — the three ×2 upsamples run as
 * `upsample2x`, every GroupNorm as group_norm's five dispatches — and what the stage holds within
 * its estimates), the lifetimes (every Session and resident closed, the Sessions first, after a
 * run, after a failure inside the batch and after a failure while opening; nothing opened when the
 * headroom check refuses) and the area limit's refusal before any GPU work. Its values are not
 * compared with upstream (the parity check is opt-in); the sha256 rows of plan step 4 pin them.
 */

import { afterAll, beforeAll, describe, expect, it } from "vite-plus/test";
import { assert, assertEquals, assertRejects } from "@std/assert";
import {
  fitsHeadroom,
  type GpuContext,
  type ResidentTensor,
  type Session,
  type SessionDiagnostics,
} from "@karume/runtime";
import { HEADROOM_MARGIN_BYTES } from "../../models/src/anima/residency.ts";
import type { ModelComponent } from "../../models/src/hub/components.ts";
import { ModelInputError } from "../../models/src/errors.ts";
import { mossVaeAreaLimit } from "../../models/src/moss/vae-contract.ts";
import {
  decodeMossVae,
  type MossVaeComponent,
  type MossVaeComponents,
  mossVaeStageNeeds,
  type MossVaeStageState,
  planMossVae,
} from "../../models/src/moss/vae-stage.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import {
  loadVaeGolden,
  MOSS_GENERATE_VAE_CAP,
  MOSS_GENERATE_VAE_SERIES,
  MOSS_VAE_CAP_CASE,
  MOSS_VAE_CASES,
  MOSS_VAE_STAGE0_SEED,
  type MossVaeCase,
  type MossVaeSeries,
  mossVaeSeriesPresent,
  openMossVaeSeries,
  vaeGoldenPresent,
} from "../../models/tests/helpers/moss-vae.ts";
import { settleReleases } from "../../models/tests/helpers/settle-releases.ts";
import { UPSTREAM_PARITY, UPSTREAM_PARITY_OPT_IN } from "./helpers/upstream-parity.ts";

const SERIES = mossVaeSeriesPresent() && MOSS_VAE_CASES.every(vaeGoldenPresent);
const CAP = process.env.KARUME_MOSS_VAE_CAP === "1";

/** The suffix that names what a skipped group lacks and the command that writes it. */
const skipped = (ok: boolean, what: string, command: string): string =>
  ok ? "" : ` (SKIP: no ${what} — ${command})`;

/**
 * The band of m per size (module doc "The band"). Derived 2026-10-27 on the RTX 3080 Ti (Node +
 * Dawn) from the three band cases of each size (seeds 20261101 / 02 / 03 — ADR 0118 decision 8
 * fixes no count; the Wan VAE checks use one decision case for the tiles and two per size for the
 * encoder); the acceptance cases
 * (the stage-0 seed 20261028 and 20261104) were measured in the same run and the bands were not
 * changed after seeing them:
 *
 * | Size    | band-case m                        | worst × 5  | band   | acceptance m         |
 * | ------- | ---------------------------------- | ---------- | ------ | -------------------- |
 * | 256²    | 3.878e-6, 3.054e-6, 3.018e-6       | 1.939e-5   | 2.0e-5 | 3.567e-6, 4.489e-6   |
 * | 512×768 | 6.385e-6, 6.586e-6, 5.722e-6       | 3.293e-5   | 3.3e-5 | 7.978e-6, 5.005e-6   |
 *
 * relRMS was 9.5e-7 … 1.0e-6 on every case (reported, not banded). m grows with the size (the
 * maximum over more pixels), so the 256² band is not carried to 512×768.
 */
const BANDS: Readonly<Record<string, number>> = {
  "256x256": 2.0e-5,
  "512x768": 3.3e-5,
};

/** The fault margin (a fault must land outside 2 × the band). */
const FAULT_MARGIN = 2;

const sizeKey = (testCase: Pick<MossVaeCase, "height" | "width">): string =>
  `${testCase.height}x${testCase.width}`;

/** m and relRMS of `actual` against `expected` (module doc). */
const metric = (actual: Float32Array<ArrayBuffer>, expected: Float32Array<ArrayBuffer>) => {
  let worst = 0;
  let scale = 0;
  let squares = 0;
  let reference = 0;
  for (let at = 0; at < expected.length; at += 1) {
    const difference = Math.abs(actual[at] - expected[at]);
    if (!(difference <= worst)) worst = difference;
    scale = Math.max(scale, Math.abs(expected[at]));
    squares += difference * difference;
    reference += expected[at] * expected[at];
  }
  return { m: worst / scale, relRms: Math.sqrt(squares / reference) };
};

/** A component whose Sessions log `open <name>` / `dispose <name>` into `events`. */
const observed = (
  component: ModelComponent,
  name: string,
  events: string[],
  sessions: Session[],
): ModelComponent => ({
  ...component,
  createSession: async (gpu, options) => {
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
});

/**
 * Runs `body` with every resident `gpu.createResident` makes logging `dispose resident <label>`
 * into `events` and collected into the list `body` gets (an instance property shadowing the
 * class's method, removed afterwards — the DiT e2e's form).
 */
const withResidentLog = async <T>(
  gpu: GpuContext,
  events: string[],
  body: (residents: readonly ResidentTensor[]) => Promise<T>,
): Promise<T> => {
  assert(!Object.hasOwn(gpu, "createResident"), "createResident is already shadowed");
  const residents: ResidentTensor[] = [];
  const create = gpu.createResident.bind(gpu);
  Object.defineProperty(gpu, "createResident", {
    configurable: true,
    value: async (byteLength: number, label?: string) => {
      const resident = await create(byteLength, label);
      residents.push(resident);
      const dispose = resident.dispose.bind(resident);
      Object.defineProperty(resident, "dispose", {
        configurable: true,
        value: () => {
          events.push(`dispose resident ${resident.label}`);
          dispose();
        },
      });
      return resident;
    },
  });
  try {
    return await body(residents);
  } finally {
    Reflect.deleteProperty(gpu, "createResident");
  }
};

/** What one Session holds: its plan backing with the input buffers it owns, and its weights. */
const heldBytes = (diagnostics: SessionDiagnostics): number =>
  diagnostics.planBacking.residentBytes +
  diagnostics.planBacking.inputBytes +
  diagnostics.weights.allocatedBytes;

/** Decodes `testCase`'s latents without the clamp's information loss check (the image). */
const decodeCase = async (
  state: MossVaeStageState,
  testCase: MossVaeCase,
  latents: Float32Array<ArrayBuffer>,
  observe?: (component: MossVaeComponent, diagnostics: SessionDiagnostics) => void,
): Promise<Float32Array<ArrayBuffer>> =>
  decodeMossVae(
    state,
    planMossVae(
      { width: testCase.width, height: testCase.height },
      state.contract,
      state.gpu.limits,
    ),
    latents,
    observe,
  );

describe.skipIf(!GPU_AVAILABLE || !SERIES)(
  `MOSS-Image VAE stage on the GPU${skipped(GPU_AVAILABLE, "GPU adapter", "a WebGPU device")}${skipped(SERIES, "VAE series or goldens", MOSS_GENERATE_VAE_SERIES)}`,
  () => {
    let gpu: GpuContext;
    let series: MossVaeSeries;
    let state: MossVaeStageState;

    beforeAll(async () => {
      gpu = await acquireTestGpu();
      series = await openMossVaeSeries();
      state = { gpu, components: series.components, contract: series.contract };
    });
    afterAll(() => {
      gpu?.destroy();
    });

    const stage0Case = MOSS_VAE_CASES.find(
      (testCase) => testCase.seed === MOSS_VAE_STAGE0_SEED && testCase.height === 256,
    );

    it("decodes 256² finite, with the upsample fusion and group_norm's dispatches in the census", async () => {
      assert(stage0Case !== undefined);
      const golden = await loadVaeGolden(stage0Case);
      const census = new Map<MossVaeComponent, SessionDiagnostics>();
      const image = await decodeCase(state, stage0Case, golden.latents, (component, diagnostics) =>
        census.set(component, diagnostics),
      );
      assertEquals(image.length, 3 * 256 * 256);
      assert(image.every((value) => value >= -1 && value <= 1));
      const count = (component: MossVaeComponent, prefix: string): number =>
        (census.get(component)?.lastRunPipelines ?? [])
          .filter((row) => row.key.startsWith(prefix))
          .reduce((total, row) => total + row.dispatchCount, 0);
      // The three ×2 upsamples run as the fusion rule's kernel (the recipe's reshape / expand form).
      expect(count("vae_decoder_post", "upsample2x:")).toBe(3);
      expect(census.get("vae_decoder_post")?.lastRunFusions).toMatchObject({ upsample2x: 3 });
      // Every GroupNorm is one group_norm node of five dispatches: 2 + 1 + 27 of the 30.
      for (const [component, norms] of [
        ["vae_decoder_pre", 2],
        ["vae_decoder_attn", 1],
        ["vae_decoder_post", 27],
      ] as const) {
        expect(count(component, "group_norm_")).toBe(5 * norms);
      }
      // The mid attention runs as one fused attention node (single head, D = 384).
      expect(count("vae_decoder_attn", "attention_")).toBeGreaterThan(0);
      // The headroom need rests on the estimates covering what the stage holds: the three
      // Sessions' backings and weights plus the residents stay within the need without its part
      // and margin.
      const plan = planMossVae({ width: 256, height: 256 }, state.contract, gpu.limits);
      const covered =
        mossVaeStageNeeds({ ...state, limits: gpu.limits }, plan) -
        HEADROOM_MARGIN_BYTES -
        Math.max(...Object.values(series.components).map((component) => component.maxPartBytes));
      let held = 2 * 384 * 32 * 32 * 4 + 3 * 256 * 256 * 4;
      for (const diagnostics of census.values()) held += heldBytes(diagnostics);
      console.info(`[moss vae] 256²: held ${held} B within the estimates + residents ${covered} B`);
      expect(held).toBeLessThanOrEqual(covered);
    });

    it("closes every Session and resident, the Sessions first, after a run and after a failure", async () => {
      assert(stage0Case !== undefined);
      const golden = await loadVaeGolden(stage0Case);
      const events: string[] = [];
      const sessions: Session[] = [];
      const components: MossVaeComponents = {
        pre: observed(series.components.pre, "pre", events, sessions),
        attn: observed(series.components.attn, "attn", events, sessions),
        post: observed(series.components.post, "post", events, sessions),
      };
      const closed = [
        "dispose pre",
        "dispose attn",
        "dispose post",
        "dispose resident moss vae_decoder_pre",
        "dispose resident moss vae_decoder_attn",
        "dispose resident moss vae_decoder_post",
      ];
      const residents = await withResidentLog(gpu, events, async (residents) => {
        await decodeCase({ ...state, components }, stage0Case, golden.latents);
        expect(events).toEqual(["open pre", "open attn", "open post", ...closed]);
        events.length = 0;
        // A failure after the batch span (the observer runs after `finishAndRead`) closes the same
        // six. A failure inside the span is the next test's.
        const failure = new Error("the observer failed");
        await assertRejects(
          () =>
            decodeCase({ ...state, components }, stage0Case, golden.latents, () => {
              throw failure;
            }),
          Error,
          "the observer failed",
        );
        expect(events).toEqual(["open pre", "open attn", "open post", ...closed]);
        return residents;
      });
      expect(residents).toHaveLength(6);
      assert(residents.every((resident) => resident.disposed));
      for (const session of sessions) await assertRejects(() => session.run({}), Error, "disposed");
    });

    it("returns the batch span and closes everything when an enqueue fails inside it", async () => {
      assert(stage0Case !== undefined);
      const golden = await loadVaeGolden(stage0Case);
      const events: string[] = [];
      const sessions: Session[] = [];
      const attn = observed(series.components.attn, "attn", events, sessions);
      const failing: ModelComponent = {
        ...attn,
        createSession: async (gpu, options) => {
          const session = await attn.createSession(gpu, options);
          Object.defineProperty(session, "enqueue", {
            configurable: true,
            value: () => Promise.reject(new Error("attn cannot enqueue")),
          });
          return session;
        },
      };
      const components: MossVaeComponents = {
        pre: observed(series.components.pre, "pre", events, sessions),
        attn: failing,
        post: observed(series.components.post, "post", events, sessions),
      };
      const residents = await withResidentLog(gpu, events, async (residents) => {
        await assertRejects(
          () => decodeCase({ ...state, components }, stage0Case, golden.latents),
          Error,
          "attn cannot enqueue",
        );
        return residents;
      });
      expect(events).toEqual([
        "open pre",
        "open attn",
        "open post",
        "dispose pre",
        "dispose attn",
        "dispose post",
        "dispose resident moss vae_decoder_pre",
        "dispose resident moss vae_decoder_attn",
        "dispose resident moss vae_decoder_post",
      ]);
      expect(residents).toHaveLength(3);
      assert(residents.every((resident) => resident.disposed));
      // The span's lock came back: a later batch on the same context runs to the end. A span left
      // open would make it wait on the lock forever, so the wait is bounded and names the cause.
      let timer: ReturnType<typeof setTimeout> | undefined;
      const stuck = new Promise<never>((_, reject) => {
        timer = setTimeout(
          () => reject(new Error("the failed batch did not return the span's lock")),
          60_000,
        );
      });
      try {
        const image = await Promise.race([decodeCase(state, stage0Case, golden.latents), stuck]);
        expect(image.length).toBe(3 * stage0Case.width * stage0Case.height);
      } finally {
        clearTimeout(timer);
      }
    });

    it("closes what it opened when a later Session fails to open", async () => {
      assert(stage0Case !== undefined);
      const golden = await loadVaeGolden(stage0Case);
      const events: string[] = [];
      const sessions: Session[] = [];
      const failing: ModelComponent = {
        ...series.components.post,
        createSession: () => Promise.reject(new Error("post cannot open")),
      };
      const components: MossVaeComponents = {
        pre: observed(series.components.pre, "pre", events, sessions),
        attn: observed(series.components.attn, "attn", events, sessions),
        post: failing,
      };
      const residents = await withResidentLog(gpu, events, async (residents) => {
        await assertRejects(
          () => decodeCase({ ...state, components }, stage0Case, golden.latents),
          Error,
          "post cannot open",
        );
        return residents;
      });
      expect(events).toEqual(["open pre", "open attn", "dispose pre", "dispose attn"]);
      expect(residents).toHaveLength(0);
      for (const session of sessions) await assertRejects(() => session.run({}), Error, "disposed");
    });

    it("opens nothing when the headroom check refuses, naming the stage", async () => {
      assert(stage0Case !== undefined);
      const golden = await loadVaeGolden(stage0Case);
      const events: string[] = [];
      const sessions: Session[] = [];
      const huge: ModelComponent = {
        ...observed(series.components.post, "post", events, sessions),
        estimate: (options) => ({
          ...series.components.post.estimate(options),
          peakAccountedBytes: 2 ** 50,
        }),
      };
      await assertRejects(
        () =>
          decodeCase(
            { ...state, components: { ...series.components, post: huge } },
            stage0Case,
            golden.latents,
          ),
        Error,
        "the VAE stage needs",
      );
      expect(events).toEqual([]);
    });

    it("refuses an area over this device's limit before any GPU work, naming the limit", () => {
      const limit = mossVaeAreaLimit(state.contract, gpu.limits);
      console.info(
        `[moss vae] this device: binding ${limit.bindingBytes} B → area limit ${limit.maxPixels} px (${limit.bytesPerPixel} B/px)`,
      );
      // The smallest square of 16-px sides above the limit.
      const side = 16 * (Math.floor(Math.sqrt(limit.maxPixels) / 16) + 1);
      expect(() => planMossVae({ width: side, height: side }, state.contract, gpu.limits)).toThrow(
        ModelInputError,
      );
      expect(() => planMossVae({ width: side, height: side }, state.contract, gpu.limits)).toThrow(
        `area limit of ${limit.maxPixels} px`,
      );
    });

    describe.skipIf(!UPSTREAM_PARITY)(
      `against upstream's f32 decode ${UPSTREAM_PARITY_OPT_IN}`,
      () => {
        const results = new Map<string, { m: number; relRms: number }>();

        it("derives each size's band from its band cases and holds it on the acceptance cases", async () => {
          for (const testCase of MOSS_VAE_CASES) {
            const golden = await loadVaeGolden(testCase);
            const image = await decodeCase(state, testCase, golden.latents);
            // The image is clamped; upstream's `decoded` is not — compare the clamped pair.
            const result = metric(image, golden.image);
            results.set(testCase.name, result);
            console.info(
              `[moss vae band] ${testCase.name}: m = ${result.m.toExponential(3)}, relRMS = ${result.relRms.toExponential(3)}`,
            );
          }
          for (const testCase of MOSS_VAE_CASES) {
            const band = BANDS[sizeKey(testCase)];
            const { m } = results.get(testCase.name) ?? { m: Number.NaN };
            assert(m <= band, `${testCase.name}: m = ${m} outside the band ${band}`);
          }
        });

        it("puts injected faults outside the band", async () => {
          for (const size of ["256x256", "512x768"]) {
            const accept = MOSS_VAE_CASES.filter(
              (testCase) => sizeKey(testCase) === size && testCase.role === "accept",
            );
            const first = await loadVaeGolden(accept[0]);
            const second = await loadVaeGolden(accept[1]);
            const swapped: MossVaeStageState = {
              ...state,
              contract: {
                ...state.contract,
                batchNorm: {
                  mean: state.contract.batchNorm.std,
                  std: state.contract.batchNorm.mean,
                },
              },
            };
            // The unpatchify's patch transposed (row a / column b swapped): the latents' channels
            // 4c + 2a + b and 4c + 2b + a exchanged, with the BatchNorm vectors exchanged alike.
            const swap = (index: number): number =>
              index - (index % 4) + 2 * (index % 2) + Math.floor((index % 4) / 2);
            const planes = first.latents.length / 128;
            const transposed = new Float32Array(first.latents.length);
            for (let c = 0; c < 128; c += 1) {
              transposed.set(
                first.latents.subarray(swap(c) * planes, (swap(c) + 1) * planes),
                c * planes,
              );
            }
            const permute = (values: Float32Array<ArrayBuffer>) =>
              Float32Array.from(values, (_, c) => values[swap(c)]);
            const patchTransposed: MossVaeStageState = {
              ...state,
              contract: {
                ...state.contract,
                batchNorm: {
                  mean: permute(state.contract.batchNorm.mean),
                  std: permute(state.contract.batchNorm.std),
                },
              },
            };
            // One stage at a time (each opens the three Sessions).
            const faults: readonly (readonly [string, () => Promise<Float32Array<ArrayBuffer>>])[] =
              [
                [
                  "BatchNorm mean / std swapped",
                  () => decodeCase(swapped, accept[0], first.latents),
                ],
                ["the latents of another seed", () => decodeCase(state, accept[0], second.latents)],
                [
                  "the unpatchify's 2×2 patch transposed",
                  () => decodeCase(patchTransposed, accept[0], transposed),
                ],
              ];
            for (const [name, run] of faults) {
              const { m } = metric(await run(), first.image);
              console.info(`[moss vae fault] ${size} ${name}: m = ${m.toExponential(3)}`);
              assert(
                m > FAULT_MARGIN * BANDS[size],
                `${size} ${name}: m = ${m} inside 2 × the band`,
              );
            }
          }
        });
      },
    );

    describe.skipIf(!UPSTREAM_PARITY || !CAP || !vaeGoldenPresent(MOSS_VAE_CAP_CASE))(
      `at the area cap 1696×1648 (${MOSS_GENERATE_VAE_CAP}; KARUME_MOSS_VAE_CAP=1) ${UPSTREAM_PARITY_OPT_IN}`,
      () => {
        it("records the stage's headroom need and whether this device admits it now", async () => {
          const plan = planMossVae({ width: 1648, height: 1696 }, state.contract, gpu.limits);
          const needs = mossVaeStageNeeds({ ...state, limits: gpu.limits }, plan);
          await settleReleases(gpu);
          const admitted = await fitsHeadroom(gpu, needs);
          console.info(
            `[moss vae cap] headroom need (estimates + residents + part + margin): ${needs} B — ` +
              `this device ${admitted ? "admits" : "refuses"} it now`,
          );
          // The outcome is a fact of the moment (other VRAM users); what holds either way: a
          // refusal of the probe is the product's refusal, loudly and before any Session opens.
          if (!admitted) {
            const golden = await loadVaeGolden(MOSS_VAE_CAP_CASE);
            await assertRejects(
              () => decodeMossVae(state, plan, golden.latents),
              Error,
              "the VAE stage needs",
            );
          }
          for (const [name, component, bindings] of [
            ["pre", state.components.pre, { h: 212, w: 206 }],
            ["attn", state.components.attn, { S: 212 * 206 }],
            ["post", state.components.post, { h: 212, w: 206 }],
          ] as const) {
            const report = component.estimate({
              maxStorageBufferBindingSize: gpu.limits.maxStorageBufferBindingSize,
              maxBufferSize: gpu.limits.maxBufferSize,
              bindings,
            });
            console.info(
              `[moss vae cap] estimate ${name}: peak ${report.peakAccountedBytes} B,` +
                ` workspace ${report.scenarios.map((scenario) => scenario.workspaceBytes).join(" / ")} B`,
            );
          }
        });

        it("decodes against upstream and reports the VRAM held (the probe bypassed for the measurement)", async () => {
          // MUST: a measurement harness only — the estimates are replaced by 0 so the stage's own
          // probe checks the residents, the part and the margin alone, and the Sessions that run
          // are the product's (the previous case records what the product's probe decides).
          const bypass = (component: ModelComponent): ModelComponent => ({
            ...component,
            estimate: (options) => ({ ...component.estimate(options), peakAccountedBytes: 0 }),
          });
          const measured: MossVaeStageState = {
            ...state,
            components: {
              pre: bypass(state.components.pre),
              attn: bypass(state.components.attn),
              post: bypass(state.components.post),
            },
          };
          const plan = planMossVae({ width: 1648, height: 1696 }, state.contract, gpu.limits);
          const golden = await loadVaeGolden(MOSS_VAE_CAP_CASE);
          const held = new Map<MossVaeComponent, SessionDiagnostics>();
          const started = performance.now();
          console.info(`[moss vae cap] decode start ${new Date().toISOString()}`);
          const image = await decodeMossVae(
            measured,
            plan,
            golden.latents,
            (component, diagnostics) => held.set(component, diagnostics),
          );
          console.info(
            `[moss vae cap] decode end ${new Date().toISOString()} (${((performance.now() - started) / 1000).toFixed(1)} s)`,
          );
          const result = metric(image, golden.image);
          console.info(
            `[moss vae cap] m = ${result.m.toExponential(3)}, relRMS = ${result.relRms.toExponential(3)}`,
          );
          let total = 2 * 384 * 212 * 206 * 4 + 3 * 1696 * 1648 * 4;
          for (const [component, diagnostics] of held) {
            const backing = diagnostics.planBacking.residentBytes;
            const inputs = diagnostics.planBacking.inputBytes;
            const weights = diagnostics.weights.allocatedBytes;
            total += heldBytes(diagnostics);
            console.info(
              `[moss vae cap] ${component}: plan backing ${backing} B, input buffers ${inputs} B,` +
                ` weights ${weights} B`,
            );
          }
          console.info(
            `[moss vae cap] held by the stage (backings + inputs + weights + residents): ${total} B`,
          );
          // The product's headroom need (the estimates of the real components) covers what the
          // stage held: the estimates + residents alone, without the part and the margin.
          const covered =
            mossVaeStageNeeds({ ...state, limits: gpu.limits }, plan) -
            HEADROOM_MARGIN_BYTES -
            Math.max(...Object.values(state.components).map((component) => component.maxPartBytes));
          expect(total).toBeLessThanOrEqual(covered);
          // Measured 2026-10-27 (RTX 3080 Ti 12 GiB, otherwise idle): m = 1.272e-5, relRMS 9.773e-7,
          // decode 2.6 s; held 9,096,149,120 B (plan backings pre 201,306,368 / attn 2,175,651,696 /
          // post 6,439,698,688 — 2,304 B per pixel — weights 111,791,888, residents 167,700,480)
          // plus pre's input buffer 5,590,016 = 9,101,739,136 B; process peak while held 8,709 MiB
          // by nvidia-smi at 50 ms. The headroom need of the previous case was 12,252,394,824 B,
          // admitted on the idle device; its trial allocation alone occupied 11,695 MiB for a
          // moment, so the stage's peak on the device is the trial allocation's, not the 8.5 GiB
          // held. Raw samples: module doc.
          // The 512×768 band is not re-derived for one case (ADR 0118 decision 8 — the cap
          // compares against the largest derived size).
          assert(result.m <= BANDS["512x768"], `cap: m = ${result.m} outside the 512×768 band`);
        });

        it("decodes the largest accepted area of this device finite, through the product's headroom check", async () => {
          // The largest area of 16-px sides (each ≤ 65,536) within this device's area limit.
          const limit = mossVaeAreaLimit(state.contract, gpu.limits);
          let best = { width: 0, height: 0 };
          for (let width = 16; width <= 65_536; width += 16) {
            const height = Math.min(65_536, 16 * Math.floor(limit.maxPixels / width / 16));
            if (width * height > best.width * best.height) best = { width, height };
          }
          const plan = planMossVae(best, state.contract, gpu.limits);
          // Deterministic DiT-space latents in [-2, 2) (a hash — no Math.random; no upstream golden
          // at this size, so the check is the finite, bounded image).
          const latents = new Float32Array(128 * plan.cells.rows * plan.cells.cols);
          for (let at = 0; at < latents.length; at += 1) {
            latents[at] = ((Math.imul(at ^ 0x5eed, 0x9e3779b1) >>> 0) / 2 ** 31 - 1) * 2;
          }
          const started = performance.now();
          const image = await decodeMossVae(state, plan, latents);
          console.info(
            `[moss vae cap] largest accepted area ${best.width}×${best.height} (${best.width * best.height} px of ${limit.maxPixels}):` +
              ` decode ${((performance.now() - started) / 1000).toFixed(1)} s`,
          );
          expect(image.length).toBe(3 * best.width * best.height);
          // The stage fails loudly on a non-finite value before the clamp; the clamp bounds the rest.
          assert(image.every((value) => value >= -1 && value <= 1));
        });
      },
    );
  },
);
