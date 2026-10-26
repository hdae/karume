/**
 * The Wan2.2 TI2V-5B VAE decoder on **int8 activations** (`convCompute: "a8"` — ADR 0133; real GPU): the
 * E2E A/B of ADR 0110 §5 at the decoder's own observation point.
 *
 * Both chunk graphs of the int8 series (`outputs/series/wan2.2-ti2v-5b-i8-dyn/vae_decoder_{first,next}` —
 * written by `wan.export_vae --model ti2v-5b --dtype i8`, the same weights the `i8` and the
 * `i8-a8-attn8-online-conv8` presets ship) run the chunk sequence of the f16 series' fixtures twice on one
 * device: the **control** (`{}` — int8 weights dequantised in the f32 implicit GEMM, the `i8` preset's
 * decoder) and the **subject** (`{ convCompute: "a8" }` — the conv8 preset's decoder). The latents come from
 * the f16 series' chunk fixtures (`vae_chunks.*` — the same inputs as the chunk-sequence parity), and their
 * f16-weight reference frames are recorded against both runs (not a gate: the int8 weights alone move them).
 *
 * ## Gates (ADR 0110 §5, ADR 0133 decision 8)
 *
 * 1. **Census**: in the subject every conv3d dispatch of both graphs runs the w8a8 key (33 nodes in first,
 *    35 in next — the counts of the chunk graphs), and no f32 conv3d key remains; the control runs no w8a8
 *    key. Each w8a8 conv3d brings its one prelude (the channel-major `quantize_rows`, which reads the conv
 *    input in place).
 * 2. **Difference floor**: the subject's frames differ from the control's by at least one bit (`accept`).
 * 3. **Collapse ceiling**: the relative RMS of the subject against the control at the observation point —
 *    **the first chunk's output frame** (the first value the knob changes) — is under {@link A8_CEILING};
 *    the whole-sequence relative RMS is recorded (`band` = 9 chunks and `accept` = 5 chunks).
 * 4. **No non-finite values** in either run (the frames are decoded from NaN-poisoned residents).
 * 5. The ceiling's docstring records the fault injection that turned it red (see {@link A8_CEILING}).
 *
 * `undefined` ceiling = not derived: the subject's ratios are recorded and the step fails with the
 * candidate (measured × 2, the convention of ADR 0110 §5-3), like the bands of the sibling tests.
 *
 * Without the int8 decoder containers or a GPU the suite SKIPs with the generation command.
 */

import { afterAll, beforeAll, describe, it } from "vite-plus/test";
import { assert, assertEquals, assertNotEquals } from "@std/assert";
import {
  type GpuContext,
  prepareContainer,
  type Session,
  type SessionOptions,
} from "@karume/runtime";
import {
  decodeWanVaeTile,
  WanVaeChunkCaches,
  wanVaeChunkCount,
  wanVaeChunkLayout,
  type WanVaeChunkLayout,
} from "../../models/src/wan/vae-chunks.ts";
import { disposeSteps } from "../../models/src/session/dispose-steps.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { settleReleases } from "../../models/tests/helpers/settle-releases.ts";
import {
  decodeWithFault,
  type Fixture,
  readFixture,
} from "../../models/tests/helpers/wan-vae-chunk-loop.ts";
import {
  ti2vVaeAssets,
  ti2vVaeFixtureUrl,
  ti2vVaeI8Assets,
  ti2vVaeI8ModelUrl,
  WAN_TI2V_VAE_FIRST as FIRST,
  WAN_TI2V_VAE_GENERATE as GENERATE_CHUNKS,
  WAN_TI2V_VAE_I8_GENERATE as GENERATE_I8,
  WAN_TI2V_VAE_I8_ROOT as I8_ROOT,
  WAN_TI2V_VAE_I8_SERIES as I8_SERIES,
  WAN_TI2V_VAE_NEXT as NEXT,
  WAN_TI2V_VAE_SAMPLE_CHANNELS as SAMPLE_CHANNELS,
  WAN_TI2V_VAE_SAMPLE_TILE as SAMPLE_TILE,
} from "../../models/tests/helpers/wan-ti2v-vae.ts";
import { openSeriesContainer } from "../../runtime/tests/helpers/container-files.ts";
import { seriesGraph } from "../../models/tests/helpers/series-graphs.ts";
import { verifyRunningGpu } from "../../runtime/tests/helpers/environment.ts";
import { conv3dCensus } from "./helpers/conv3d-census.ts";

/**
 * The collapse ceiling of the subject against the control at the first chunk's output frame (relative
 * RMS). `undefined` = not derived (module doc).
 *
 * Derived 2026-10-23 (RTX 3080 Ti, Node + Dawn): band 1.284e-2 / accept 1.230e-2 at the first chunk's
 * frame (whole sequence 1.176e-2 / 1.164e-2) → measured max × 2 = 2.57e-2, rounded up to 2.6e-2.
 * Recorded against the f16-weight reference frames: the control (int8 weights, f32 compute) 2.99e-2 /
 * 2.93e-2, the subject 3.21e-2 / 3.15e-2 — the int8 weights alone move the decode more than the int8
 * activations add. Fault injection that turns the ceiling red: dropping one cache from the subject's
 * chunk loop (`drop-cache cache_31`, the step below).
 *
 * MUST: do not widen it to pass a later run — investigate the cause instead.
 */
const A8_CEILING: number | undefined = 2.6e-2;

/** The conv3d node counts of the two chunk graphs (every one runs the w8a8 key in the subject). */
const CONV3D_NODES = { first: 33, next: 35 } as const;

const I8_ASSETS = ti2vVaeI8Assets();
const I8_PRESENT = I8_ASSETS.every(({ present }) => present);
const FIXTURES_PRESENT = ti2vVaeAssets().every(({ present }) => present);
if (!I8_PRESENT) {
  console.warn(
    `[karume] ${I8_ROOT.pathname} に Wan2.2 TI2V の int8 の VAE decoder の容器が無いため、` +
      `convCompute 'a8' の A/B を SKIP する。生成: ${GENERATE_I8}`,
  );
} else if (!FIXTURES_PRESENT) {
  console.warn(
    `[karume] f16 系列の chunk フィクスチャが無いため、convCompute 'a8' の A/B を SKIP する。生成: ${GENERATE_CHUNKS}`,
  );
}

type Run = {
  readonly gpu: GpuContext;
  readonly layout: WanVaeChunkLayout;
  readonly control: { readonly first: Session; readonly next: Session };
  readonly subject: { readonly first: Session; readonly next: Session };
  readonly caches: WanVaeChunkCaches;
  readonly fixtures: { readonly band: Fixture; readonly accept: Fixture };
};

const countNonFinite = (values: Float32Array): number =>
  values.reduce((count, value) => count + (Number.isFinite(value) ? 0 : 1), 0);

/** Relative RMS of `got` against `reference` (RMS of the difference over the RMS of the reference). */
const relativeRms = (got: Float32Array, reference: Float32Array): number => {
  assertEquals(got.length, reference.length, "relative RMS: 要素数");
  let diff = 0;
  let base = 0;
  for (let i = 0; i < got.length; i += 1) {
    diff += (got[i] - reference[i]) ** 2;
    base += reference[i] ** 2;
  }
  return Math.sqrt(diff / base);
};

/** The first chunk's output frame (`[12, 1, 128, 128]` — the first frame of the sequence). */
const firstChunkFrame = (frames: Float32Array<ArrayBuffer>): Float32Array<ArrayBuffer> =>
  frames.slice(0, SAMPLE_CHANNELS * SAMPLE_TILE * SAMPLE_TILE);

const poisonFrames = async (caches: WanVaeChunkCaches, chunks: number): Promise<void> => {
  for (const resident of await caches.frames(chunks)) {
    resident.write(new Float32Array(resident.byteLength / 4).fill(Number.NaN));
  }
};

describe.skipIf(!I8_PRESENT || !FIXTURES_PRESENT || !GPU_AVAILABLE)(
  "Wan2.2 TI2V VAE decoder on int8 activations (convCompute 'a8' — ADR 0133): census, difference floor," +
    " collapse ceiling at the first chunk's frame, no non-finite values (real GPU)",
  () => {
    let gpu: GpuContext | undefined;
    const sessions: Session[] = [];
    let caches: WanVaeChunkCaches | undefined;
    let run: Run | undefined;
    const current = (): Run => {
      if (run === undefined) throw new Error("setup (beforeAll) did not complete");
      return run;
    };
    /** Relative RMS at the observation point per case (the ceiling candidate's material). */
    const observed = new Map<"band" | "accept", number>();

    beforeAll(async () => {
      const [firstOpened, nextOpened] = await Promise.all([
        openSeriesContainer(ti2vVaeI8ModelUrl(FIRST)),
        openSeriesContainer(ti2vVaeI8ModelUrl(NEXT)),
      ]);
      const models = {
        first: prepareContainer(firstOpened, seriesGraph(I8_SERIES, FIRST)),
        next: prepareContainer(nextOpened, seriesGraph(I8_SERIES, NEXT)),
      };
      const layout = wanVaeChunkLayout(models.first, models.next);
      const fixtures = {
        band: await readFixture(ti2vVaeFixtureUrl("band")),
        accept: await readFixture(ti2vVaeFixtureUrl("accept")),
      };
      const device = verifyRunningGpu(await acquireTestGpu());
      gpu = device;
      const open = async (options: SessionOptions) => {
        const first = await models.first.createContainerSession(device, options);
        sessions.push(first);
        const next = await models.next.createContainerSession(device, options);
        sessions.push(next);
        return { first, next };
      };
      const control = await open({});
      const subject = await open({ convCompute: "a8" });
      caches = await WanVaeChunkCaches.create(device, layout);
      run = { gpu: device, layout, control, subject, caches, fixtures };
    });
    afterAll(async () => {
      const device = gpu;
      await disposeSteps([
        ...sessions.map((session) => () => session.dispose()),
        () => caches?.dispose(),
        () => (device === undefined ? undefined : settleReleases(device)),
        () => device?.destroy(),
      ]);
    });

    const decode = async (
      which: "control" | "subject",
      name: "band" | "accept",
    ): Promise<Float32Array<ArrayBuffer>> => {
      const { gpu: device, layout, caches: liveCaches, fixtures } = current();
      const fixture = fixtures[name];
      const chunks = wanVaeChunkCount(layout, fixture.latents);
      await poisonFrames(liveCaches, chunks);
      return await decodeWanVaeTile(device, current()[which], liveCaches, fixture.latents);
    };

    for (const name of ["band", "accept"] as const) {
      it(`${name}: both runs decode with 0 non-finite values; the subject differs from the control (floor) and its relative RMS at the first chunk's frame is recorded`, async () => {
        const control = await decode("control", name);
        const subject = await decode("subject", name);
        assertEquals(countNonFinite(control), 0, `${name}: control の非有限`);
        assertEquals(countNonFinite(subject), 0, `${name}: subject の非有限`);
        // Difference floor — a silent fallback of the knob would make the two runs bit-identical.
        assertNotEquals([...subject], [...control], `${name}: subject が control と同じ値（床）`);
        const atFirstChunk = relativeRms(firstChunkFrame(subject), firstChunkFrame(control));
        const whole = relativeRms(subject, control);
        const { fixtures } = current();
        const controlVsF16 = relativeRms(control, fixtures[name].frames);
        const subjectVsF16 = relativeRms(subject, fixtures[name].frames);
        observed.set(name, atFirstChunk);
        console.log(
          `[wan-ti2v-vae-a8] ${name}: relRMS(subject/control) first-chunk=${atFirstChunk.toExponential(3)} ` +
            `whole=${whole.toExponential(3)} / vs f16 reference: control=${controlVsF16.toExponential(3)} ` +
            `subject=${subjectVsF16.toExponential(3)}`,
        );
      });
    }

    it("census: every conv3d of both graphs runs the w8a8 key in the subject, none in the control", () => {
      const { control, subject } = current();
      for (const [graph, expected] of [
        ["first", CONV3D_NODES.first],
        ["next", CONV3D_NODES.next],
      ] as const) {
        const subjectCensus = conv3dCensus(graph, subject[graph].diagnostics());
        assertEquals(subjectCensus.w8a8, expected, `${graph}: subject の w8a8 conv3d の本数`);
        assertEquals(subjectCensus.f32, 0, `${graph}: subject に f32 の conv3d が残った`);
        assertEquals(subjectCensus.preludes, expected, `${graph}: subject の quantize_rows の本数`);
        const controlCensus = conv3dCensus(graph, control[graph].diagnostics());
        assertEquals(controlCensus.w8a8, 0, `${graph}: control に w8a8 の conv3d が走った`);
        assertEquals(controlCensus.f32, expected, `${graph}: control の f32 conv3d の本数`);
      }
    });

    it("collapse ceiling: the relative RMS at the first chunk's frame is under the ceiling (both cases)", () => {
      const band = observed.get("band");
      const accept = observed.get("accept");
      assert(band !== undefined && accept !== undefined, "the cases did not record their ratios");
      if (A8_CEILING === undefined) {
        throw new Error(
          `the ceiling is not derived — candidate: measured max ${Math.max(band, accept).toExponential(3)} × 2 = ` +
            `${(Math.max(band, accept) * 2).toExponential(2)} (band ${band.toExponential(3)} / accept ${accept.toExponential(3)})`,
        );
      }
      assert(band < A8_CEILING, `band: ${band} is not under the ceiling ${A8_CEILING}`);
      assert(accept < A8_CEILING, `accept: ${accept} is not under the ceiling ${A8_CEILING}`);
    });

    it("fault injection: dropping one cache from the subject's chunk loop puts the whole-sequence relative RMS over the ceiling (the ceiling is live)", async () => {
      const { gpu: device, layout, subject, caches: liveCaches, fixtures } = current();
      const control = await decode("control", "accept");
      const chunks = wanVaeChunkCount(layout, fixtures.accept.latents);
      await poisonFrames(liveCaches, chunks);
      const faulted = await decodeWithFault(device, subject, liveCaches, fixtures.accept.latents, {
        kind: "drop-cache",
        cache: "cache_31",
      });
      const ratio = relativeRms(faulted, control);
      console.log(
        `[wan-ti2v-vae-a8] fault drop-cache cache_31: relRMS(subject/control) whole=${ratio.toExponential(3)}`,
      );
      assert(A8_CEILING !== undefined, "the ceiling is not derived");
      assert(ratio > A8_CEILING, `the fault stayed under the ceiling (${ratio} ≤ ${A8_CEILING})`);
    });
  },
);
