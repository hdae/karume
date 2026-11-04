/**
 * SBV2's **self A/B check** (real GPU, ADR 0110 decision 5). It compares the practical preset
 * `i8-a8` (`linearCompute: "a8"`, which runs the linears of `front` / `voice` as integer dot
 * products down to the activations — not the default preset), **on the same machine with the same
 * i8 weights**, against the reference preset whose `session` is empty (derived from the manifest —
 * {@link referencePartnerOf}; `i8` in the current distribution). This preset had no check at all
 * until now.
 *
 * ## Observation points = the 4 outputs of front (right after the preset first takes effect —
 * decision 5-4)
 *
 * SBV2 has no DiT step. A quant's `session` reaches only the `front` and `voice` Sessions
 * (`text_encoder` gets `{}` — the module doc of `src/sbv2/pipeline.ts`), and `front` (enc_p + dp +
 * sdp reverse) runs first, so its **4 graph outputs** (logw_sdp / logw_dp / m_p / logs_p — looked
 * up by position) are the observation points.
 *
 * - Why the final output (the waveform) is not an observation point: front's logw goes through the
 *   duration `ceil` and decides the frame count, so if the a8 difference splits the length of even
 *   one phoneme, the waveform's length changes and an element-wise comparison no longer holds
 *   (compare before the discretization).
 * - Why all 4 outputs are held to a band: front has only 2 eligible linears
 *   (docs/research/2026-08-03-dp4a-w8a8-design.md), and how much each affects which output is not
 *   known until measured. Narrowing to one would leave the floor permanently red if it picked an
 *   unaffected output.
 *
 * The public surface (`Pipeline.generate`) does not return front's outputs, so this check runs
 * **the same synthesis body** as `Pipeline` (`openSbv2State` → `synthesizeSbv2` — the same entry
 * point as the dump path `examples/sbv2/dump.ts`) and swaps only the `front` component for a
 * wrapper that "copies the Session's outputs out" ({@link capturingOutputs} — it does not touch the
 * values). The wiring quant → SessionOptions → the front / voice Sessions is outside the wrapper
 * (admission and `synthesizeSbv2`), so the check covers that wiring as well.
 *
 * The mapping of checks to fault injections is the same as the module doc of
 * `e2e_anima_ab_test.ts`. The census looks at the front and voice Sessions **separately** (to catch
 * wiring where the preset reaches only one of the Sessions). The bundle has only one knob,
 * `linearCompute`, so there are two fault injections: ① (a manifest with an empty `session` → the
 * floor) and ② (another seed for the second run only → determinism).
 *
 * MUST: the assets are `models/karume-sbv2-jvnv/` (untracked, local assets of a real-GPU machine).
 * Environments without them and without a GPU **SKIP explicitly** with a reason (ADR 0005).
 * Analyzing the utterance is the caller's responsibility, so this test fetches the Japanese
 * dictionary with `@hdae/yomi` (network on the first run only — the same as `e2e_sbv2_wav_test.ts`).
 */

import { readFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assertThrows } from "@std/assert";
import { type FileRef, type Manifest, parseManifest, resolveSelection } from "@karume/hub";
import type { Session, SessionDiagnostics } from "@karume/runtime";
import { analyzeWithWords } from "@hdae/yomi";
import { getDictionary } from "@hdae/yomi/loader";
import { type Utterance, toUtterance } from "../../models/sbv2.ts";
import {
  assetOpener,
  closeSbv2State,
  openSbv2State,
  type RunComponent,
  synthesizeSbv2,
} from "../../models/src/sbv2/pipeline.ts";
import type { ComponentOpener, ModelComponent } from "../../models/src/hub/components.ts";
import {
  assertSeatsApplied,
  type CensusRow,
  mergeCensus,
} from "../../runtime/tests/helpers/pipeline-census.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";
import {
  type AbBand,
  type AbMeasurement,
  assertBitIdentical,
  comparisonOf,
  describeAb,
  describeBundleCensus,
  judgeAb,
  measureAb,
  overrideQuantSession,
  quantOf,
  referencePartnerOf,
} from "./helpers/ab-gate.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { readTextIfPresent } from "../../models/tests/helpers/read-if-present.ts";

/** Where the assets are (`models/karume-sbv2-jvnv/` at the repository root). */
const ASSETS_DIR = new URL("../../../models/karume-sbv2-jvnv/", import.meta.url);
/**
 * Where the measurements and verdicts go (`outputs/verify/<environment key>/<date>_sbv2-ab/` —
 * safe to delete).
 */
const results = openResults("sbv2-ab");

const MODEL = "F1";
const PRACTICAL = "i8-a8";
/**
 * The input (the same sentence and seed as the sha check of `e2e_sbv2_wav_test.ts` — the knobs
 * are the distribution's defaults).
 */
const TEXT = "こんにちは、これはテストです。";
const SEED = 0;

/**
 * front's outputs (in graph-output **position** order — the IR's output names are torch node names
 * and carry no meaning) and their bands. `band: undefined` = not derived yet — the check prints the
 * measurement and stops red.
 *
 * The band derivation rule is the same as the doc of `CASES` in `e2e_anima_ab_test.ts` (ceiling =
 * about the measurement × 2; floor = the MUST of at least one bit different + a factor only when one
 * can be shown). Derivation table (2026-09-26, Intel Arc B570 / Deno, reference preset `i8`,
 * ceiling = the measurement × 2 to two significant digits):
 *
 * | Output          | relRMS      | maxAbs      | floor | ceiling | Date, machine   |
 * | --------------- | ----------- | ----------- | ----- | ------- | --------------- |
 * | front.logw_sdp  | 5.1713e-3   | 2.3147e-2   | 0     | 1.0e-2  | 2026-09-26 B570 |
 * | front.logw_dp   | 1.7008e-3   | 7.4958e-3   | 0     | 3.4e-3  | 2026-09-26 B570 |
 * | front.m_p       | 3.9264e-3   | 2.5642e-2   | 0     | 7.9e-3  | 2026-09-26 B570 |
 * | front.logs_p    | 1.4743e-4   | 3.3269e-3   | 0     | 3.0e-4  | 2026-09-26 B570 |
 */
const FRONT_OUTPUTS: readonly { readonly name: string; readonly band: AbBand | undefined }[] = [
  { name: "front.logw_sdp", band: { floor: 0, ceiling: 1.0e-2 } },
  { name: "front.logw_dp", band: { floor: 0, ceiling: 3.4e-3 } },
  { name: "front.m_p", band: { floor: 0, ceiling: 7.9e-3 } },
  { name: "front.logs_p", band: { floor: 0, ceiling: 3.0e-4 } },
];

const manifestText = await readTextIfPresent(new URL("karume.json", ASSETS_DIR));
if (manifestText === undefined) {
  console.warn(
    `[karume] ${ASSETS_DIR.pathname} has no karume.json, so the SBV2 self A/B check is SKIPped` +
      " (bake it with the exporter's dist.py)",
  );
}
const RUNNABLE = GPU_AVAILABLE && manifestText !== undefined;

const readManifest = (): Manifest => parseManifest(manifestText as string);

/** The utterance (analyzed only once — the dictionary is not fetched again per check). */
const utteranceOnce = (() => {
  let cached: Promise<Utterance> | undefined;
  return (): Promise<Utterance> => {
    // TypeScript 7 does not narrow `cached` after `??=`; keep the single-evaluation semantics explicitly.
    const utterance =
      cached ??
      getDictionary().then((dictionary) => toUtterance(analyzeWithWords(dictionary, TEXT)));
    cached = utterance;
    return utterance;
  };
})();

/**
 * Reads the assets a quant requires from local files (a local version of `fetchAssets` — the same
 * form as the reader of `e2e_sbv2_wav_test.ts`). MUST: zero-length parts are listed too (the index
 * is the id inside the container).
 */
const loadLocalAssets = async (
  manifest: Manifest,
  quant: string,
): Promise<Record<string, Uint8Array<ArrayBuffer>>> => {
  const selection = resolveSelection(manifest, { model: MODEL, quant });
  const byPath = new Map<string, Uint8Array<ArrayBuffer>>();
  let assets: Record<string, Uint8Array<ArrayBuffer>> = {};
  const read = async (key: string, ref: FileRef): Promise<void> => {
    const cached = byPath.get(ref.path);
    const bytes = cached ?? (await readFile(new URL(ref.path, ASSETS_DIR)));
    if (cached === undefined) byPath.set(ref.path, bytes);
    assets = { ...assets, [key]: bytes };
  };
  for (const name of Object.keys(selection.containers)) {
    const { parts } = selection.containers[name];
    for (const [index, ref] of parts.entries()) await read(`${name}[${index}]`, ref);
  }
  for (const name of Object.keys(selection.assets)) await read(name, selection.assets[name]);
  return assets;
};

/**
 * A component that copies the outputs of the Session's `run` (in graph-output position order) and
 * hands them to `sink`. It does not touch the values — the caller gets the original outputs as
 * they are, and `sink` gets the copies.
 *
 * NOTE: `Session` has private fields, so members other than `run` are returned bound to the
 * original Session (calling them with the Proxy as the receiver would fail to read the private
 * fields).
 */
const capturingOutputs = (
  component: ModelComponent,
  sink: (outputs: readonly Float32Array<ArrayBuffer>[]) => void,
): ModelComponent => ({
  ...component,
  createSession: async (gpu, options) => {
    const session = await component.createSession(gpu, options);
    const run = async (
      ...args: Parameters<Session["run"]>
    ): Promise<Awaited<ReturnType<Session["run"]>>> => {
      const outputs = await session.run(...args);
      sink(
        component.graph.outputs.map((name) => {
          const tensor = outputs[name];
          if (tensor === undefined || tensor.dtype !== "f32") {
            throw new Error(`front's output '${name}' is not f32 / is missing`);
          }
          return new Float32Array(tensor.data);
        }),
      );
      return outputs;
    };
    return new Proxy(session, {
      get: (target, property) => {
        if (property === "run") return run;
        const value: unknown = Reflect.get(target, property, target);
        return typeof value === "function" ? value.bind(target) : value;
      },
    });
  },
});

/** The observation of one synthesis (front's 4 outputs, the waveform, the front / voice census). */
type Observation = {
  readonly front: readonly Float32Array<ArrayBuffer>[];
  /**
   * The waveform (not held to a band — used only for the determinism check; determinism including
   * the voice Session).
   */
  readonly audio: Float32Array<ArrayBuffer>;
  readonly census: Readonly<Record<"front" | "voice", readonly CensusRow[]>>;
};

/**
 * Synthesizes once with `quant` and takes front's outputs and the census (the state acquires and
 * releases the GPU itself).
 */
const observeFront = async (
  manifest: Manifest,
  quant: string,
  seed: number,
): Promise<Observation> => {
  const where = `${quant} / seed ${seed}`;
  const assets = await loadLocalAssets(manifest, quant);
  const base = await assetOpener(assets);
  const captured: (readonly Float32Array<ArrayBuffer>[])[] = [];
  const open: ComponentOpener = (key) =>
    key === "front" ? capturingOutputs(base(key), (outputs) => captured.push(outputs)) : base(key);
  const runs: Record<RunComponent, SessionDiagnostics["lastRunPipelines"][]> = {
    text_encoder: [],
    front: [],
    voice: [],
  };
  const state = await openSbv2State({ manifest, assets }, open, {
    model: MODEL,
    quant,
    onRunDiagnostics: (component, diagnostics) => {
      runs[component].push(diagnostics.lastRunPipelines);
    },
  });
  let audio: Float32Array<ArrayBuffer>;
  try {
    ({ audio } = await synthesizeSbv2(state, await utteranceOnce(), { seed }));
  } finally {
    closeSbv2State(state);
  }
  if (captured.length !== 1) {
    throw new Error(
      `${where}: front ran ${captured.length} times (one synthesis should be one run)`,
    );
  }
  return {
    front: captured[0],
    audio,
    census: {
      front: mergeCensus(runs.front, `${where} front`),
      voice: mergeCensus(runs.voice, `${where} voice`),
    },
  };
};

/** The measurement per front output (paired with {@link FRONT_OUTPUTS} by position). */
const measureFront = (practical: Observation, reference: Observation): readonly AbMeasurement[] => {
  if (practical.front.length !== FRONT_OUTPUTS.length) {
    throw new Error(
      `front has ${practical.front.length} outputs (expected ${FRONT_OUTPUTS.length} — the positional mapping broke)`,
    );
  }
  return FRONT_OUTPUTS.map((_, index) => measureAb(practical.front[index], reference.front[index]));
};

const errorText = (cause: unknown): string =>
  cause instanceof Error ? cause.message : String(cause);

const ID = `${MODEL}-${PRACTICAL}`;

test.skipIf(!RUNNABLE)(
  `e2e (real GPU): ${ID}'s front outputs balance against the reference preset in determinism, census and band (self A/B)`,
  async () => {
    await runRecordedCase(results, { id: ID, failureNote: errorText }, async (recorded) => {
      const manifest = readManifest();
      const reference = referencePartnerOf(manifest, MODEL, PRACTICAL);
      const declared = quantOf(manifest, MODEL, PRACTICAL).session;
      const referenceRun = await observeFront(manifest, reference, SEED);
      const first = await observeFront(manifest, PRACTICAL, SEED);
      const second = await observeFront(manifest, PRACTICAL, SEED);
      const measured = measureFront(first, referenceRun);
      FRONT_OUTPUTS.forEach(({ name, band }, index) => {
        recorded.comparisons.push(
          comparisonOf({ output: name, reference, practical: PRACTICAL }, measured[index], band),
        );
        console.log(`[e2e] sbv2 A/B ${ID} ${name}: ${describeAb(measured[index])}`);
      });
      const census = `front ${describeBundleCensus(first.census.front, declared)} / voice ${describeBundleCensus(
        first.census.voice,
        declared,
      )}`;
      console.log(`[e2e] sbv2 A/B ${ID}: ${reference} → ${PRACTICAL} / census ${census}`);
      FRONT_OUTPUTS.forEach(({ name }, index) => {
        assertBitIdentical(
          first.front[index],
          second.front[index],
          `${ID}: two runs of the practical preset ${name}`,
        );
      });
      assertBitIdentical(
        first.audio,
        second.audio,
        `${ID}: waveforms of two runs of the practical preset`,
      );
      assertSeatsApplied(first.census.front, declared, `${ID}: practical preset front`);
      assertSeatsApplied(first.census.voice, declared, `${ID}: practical preset voice`);
      FRONT_OUTPUTS.forEach(({ name, band }, index) => {
        judgeAb(measured[index], band, `${ID} ${name}`);
      });
      return { status: "pass", note: `census ${census}` };
    });
  },
);

// --- Fault injection (ADR 0110 decision 5-5 — asserts that the check turns red) ---------

test.skipIf(!RUNNABLE)(
  "e2e (real GPU): fault injection ① a manifest with the practical preset's session emptied is red at the floor",
  async () => {
    const id = `fault1-empty-session-${MODEL}`;
    await runRecordedCase(results, { id, failureNote: errorText }, async (recorded) => {
      const manifest = readManifest();
      const reference = referencePartnerOf(manifest, MODEL, PRACTICAL);
      const injected = parseManifest(
        JSON.stringify(
          overrideQuantSession(JSON.parse(manifestText as string), MODEL, PRACTICAL, {}),
        ),
      );
      const referenceRun = await observeFront(manifest, reference, SEED);
      const faulty = await observeFront(injected, PRACTICAL, SEED);
      const measured = measureFront(faulty, referenceRun);
      FRONT_OUTPUTS.forEach(({ name, band }, index) => {
        recorded.comparisons.push(
          comparisonOf(
            { output: name, reference, practical: `${PRACTICAL} (session empty)` },
            measured[index],
            band,
          ),
        );
        console.log(`[e2e] sbv2 fault injection ① ${name}: ${describeAb(measured[index])}`);
        assertThrows(() => judgeAb(measured[index], band, `${id} ${name}`), Error, "floor failure");
      });
    });
  },
);

test.skipIf(!RUNNABLE)(
  "e2e (real GPU): fault injection ② another seed for the practical preset's second run only turns the determinism check red",
  async () => {
    const manifest = readManifest();
    const first = await observeFront(manifest, PRACTICAL, SEED);
    const second = await observeFront(manifest, PRACTICAL, SEED + 1);
    // The seed affects the sdp reverse noise (z_noise), so the output that surely diverges is
    // logw_sdp (position 0).
    assertThrows(
      () => assertBitIdentical(first.front[0], second.front[0], "fault injection ②"),
      Error,
      "not bit-identical",
    );
  },
);
