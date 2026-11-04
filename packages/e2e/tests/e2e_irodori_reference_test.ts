// The check of **the whole chain** from reference audio to the speaker latent (real GPU).
//
// Where `irodori_reference_test.ts` compares only the host preprocessing with the golden, this
// one connects **from the raw waveform to the latent** in one go and compares with the golden's
// `latent` — normalisation → reflect pad → framing into `[1,T,1920]` → `codec_encoder` (real
// GPU).
//
// Framing is only a reinterpretation of the tensor, but **a wrong width still fits the shape**
// (dividing by another number instead of 1920 only changes T and raises no exception). This is the
// only place that catches that mix-up.
//
// The golden's `latent` is upstream `codec.encode_waveform`'s output itself
// (`tools/export-recipes/irodori/dacvae/host.py` measures bit identity between "host chain →
// graph" and upstream before writing it), so this check closing means
// "the TS reference-audio path produces the same speaker condition as upstream".
//
// An environment without the assets or without a GPU SKIPs explicitly, naming what is missing
// (ADR 0005); a host golden absent from a complete fetch of the pinned test assets FAILs (ADR 0139).

import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { acquireGpu, parseSafetensors } from "@karume/runtime";
import { goldenDir, testAssetsFetched } from "@karume/test-support/test-assets";
import { normalizeReference, reflectPadToHop } from "../../models/src/irodori/host/reference.ts";
import {
  mirrorPresent,
  mirrorRoot,
  openMirrorComponent,
} from "../../models/tests/helpers/mirror.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { readFileIfPresent } from "../../models/tests/helpers/read-if-present.ts";
import { UPSTREAM_PARITY, UPSTREAM_PARITY_OPT_IN } from "./helpers/upstream-parity.ts";

/** The operating values of real-weight v4-small (the same numbers `pipelineConfig` carries). */
const SAMPLE_RATE = 48000;
const HOP_LENGTH = 1920;
const LATENT_DIM = 32;

/**
 * The (absolute) tolerance for the whole path from the raw waveform to the latent.
 *
 * Measured (a plain comparison with `atol = rtol = 0`; 2026-08-12, real GPU, the golden's 3 LUFS
 * normalisation cases):
 *
 * | Case        | T   | Chain maxAbs  | Encoder-only maxAbs | Preprocessing only | \|latent\| max  |
 * | ----------- | --- | ------------- | ------------------- | ------------------ | --------------- |
 * | ref-default | 190 | 2.1979e-5     | 2.5570e-5           | 1.1742e-5        | 5.31            |
 * | ref-short   | 10  | 8.7023e-6     | 8.1062e-6           | 2.2054e-6        | 3.44            |
 * | ref-odd     | 76  | **2.8685e-5** | 2.7344e-5           | 1.2457e-5        | 3.93            |
 *
 * "Encoder-only" is the difference when the golden's `padded` goes into the graph as is;
 * "preprocessing only" is the difference between the results of feeding the TS preprocessing's
 * output and the golden's `padded`. **The preprocessing's contribution (1.25e-5) is smaller than
 * the encoder's own difference (2.73e-5)**, and the chain stays about the same size as the encoder
 * alone — the reading of these 3 rows is that LUFS's f64/f32 difference (a relative 2.5e-6 in the
 * gain) is buried in the conv's reduction-order difference.
 *
 * atol 2e-4 is **7.0 times** the measured worst 2.8685e-5 (ref-odd). Landing on the same value as
 * the runtime's encoder-only check (`ENCODER_TOLERANCE` of `e2e_dacvae_test.ts` (removed in the
 * 0.15 triage)) is not chance but a consequence of "the preprocessing's contribution is buried in
 * the encoder's difference" above (it is still **derived independently** — if the preprocessing
 * side worsens, this measurement table moves first).
 *
 * **A wrong formula** (the target dB, the pad direction, the frame width) shows at O(0.1–1), the
 * same as the latent's range (±5.3).
 */
const LATENT_ATOL = 2e-4;

/**
 * The golden cases that go through LUFS normalisation (the 3 with `normalizeDb: -16` — every branch
 * this path has).
 */
const CASES: readonly string[] = ["ref-default", "ref-short", "ref-odd"];

/** The host goldens (`outputs/test-assets/goldens/dacvae-32dim/host/` — ADR 0139 decision 2). */
const GOLDEN_DIR = new URL("host/", goldenDir("dacvae-32dim"));
const MIRROR = mirrorRoot("karume-irodori-v4-small");
/** The encoder graph (the manifest's weights key) at the f32 storage the golden latent came from. */
const ENCODER_GRAPH = "codec_encoder";
const ENCODER_CHOICE = { model: "v4-small", quant: "f32" } as const;

const FETCH_HINT =
  "fetch: node tools/test-assets/fetch.ts — the assets gate accepts only a complete fetch; " +
  "--include 'goldens/dacvae-32dim/host/**' with KARUME_ALLOW_NO_ASSETS=1 runs this file alone";

/**
 * Whether `outputs/test-assets/` holds a complete fetch of the pin (ADR 0139). Then an absent host
 * golden means the dataset at the pin lacks it, so the cases FAIL instead of SKIPping.
 */
const FETCHED = testAssetsFetched();

const readBytes = async (url: URL): Promise<ArrayBuffer | undefined> => {
  const bytes = await readFileIfPresent(url);
  return bytes === undefined
    ? undefined
    : (bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
};

const ENCODER_PRESENT = mirrorPresent(MIRROR);
const caseBytes = new Map<string, ArrayBuffer>();
const absentCases: string[] = [];
for (const name of CASES) {
  const url = new URL(`case.${name}.safetensors`, GOLDEN_DIR);
  const bytes = await readBytes(url);
  if (bytes === undefined) absentCases.push(url.pathname);
  else caseBytes.set(name, bytes);
}
// An absent mirror SKIPs. An absent host golden SKIPs unless the test assets are a complete fetch
// of the pin, where `runChainOverCases` FAILs.
const ASSETS_AVAILABLE = ENCODER_PRESENT && (absentCases.length === 0 || FETCHED);
if (!ASSETS_AVAILABLE) {
  console.warn(
    "[karume] The reference audio → latent e2e SKIPs: missing " +
      [
        ...(ENCODER_PRESENT ? [] : [`the mirror ${MIRROR.pathname} (its codec_encoder)`]),
        ...(absentCases.length === 0 ? [] : [`${absentCases.join(" / ")} (${FETCH_HINT})`]),
      ].join(" / "),
  );
}

const RUNNABLE = GPU_AVAILABLE && ASSETS_AVAILABLE;

const tensorOf = (buffer: ArrayBuffer, key: string, where: string): Float32Array<ArrayBuffer> => {
  const file = parseSafetensors(buffer);
  const spec = file.tensors.get(key);
  if (spec === undefined) throw new Error(`${where}: '${key}' is missing`);
  return new Float32Array(
    file.buffer.slice(spec.byteOffset, spec.byteOffset + spec.byteLength),
  ) as Float32Array<ArrayBuffer>;
};

/** One reference case pushed through the pipeline's chain, with its golden latent beside it. */
type ChainResult = {
  readonly rawLength: number;
  readonly frames: number;
  readonly actual: Float32Array;
  readonly expected: Float32Array<ArrayBuffer>;
};

/**
 * Runs every golden case through normalize -> reflect pad -> framing -> codec encoder on one
 * session and hands each result to `check`. Both cases below share this one path so the shape
 * gate and the parity band always look at the same chain.
 */
const runChainOverCases = async (
  check: (name: string, result: ChainResult) => void,
): Promise<void> => {
  if (absentCases.length > 0) {
    throw new Error(
      `${absentCases.join(" / ")} absent, but outputs/test-assets/ holds a complete fetch of the ` +
        "pinned revision: the test-asset dataset at the pin lacks it",
    );
  }
  const encoder = await openMirrorComponent(MIRROR, ENCODER_GRAPH, ENCODER_CHOICE);
  const gpu = await acquireGpu();
  try {
    const session = await encoder.createSession(gpu, {});
    try {
      for (const name of CASES) {
        const buffer = caseBytes.get(name) as ArrayBuffer;
        const raw = tensorOf(buffer, "raw", `golden case.${name}`);
        const expected = tensorOf(buffer, "latent", `golden case.${name}`);

        // This is the same chain as the pipeline's (`encodeReferenceAudio` in `pipeline.ts`).
        const padded = reflectPadToHop(normalizeReference(raw, SAMPLE_RATE).data, HOP_LENGTH);
        const frames = padded.length / HOP_LENGTH;
        const outputs = await session.run({
          wav: { dtype: "f32", shape: [1, frames, HOP_LENGTH], data: padded },
        });
        const tensor = outputs[encoder.graph.outputs[0]];
        if (tensor.dtype !== "f32") throw new Error(`the encoder output is ${tensor.dtype}`);
        check(name, { rawLength: raw.length, frames, actual: tensor.data, expected });
      }
    } finally {
      await session.dispose();
    }
  } finally {
    gpu.destroy();
  }
};

const assertLatentShape = (name: string, { frames, actual, expected }: ChainResult): void => {
  assertEquals(
    actual.length,
    expected.length,
    `${name}: latent element count (a mix-up of the frame width ${HOP_LENGTH} shows here)`,
  );
  assertEquals(actual.length / LATENT_DIM, frames, `${name}: latent frame count`);
};

// The shape gate stays in the default run: a wrong frame width still yields a valid tensor, and
// only the element count against the golden latent catches it. The value comparison (atol) is
// upstream parity and is opt-in.
test.skipIf(!RUNNABLE)(
  "e2e (real GPU): the latent shape of reference audio → normalisation → pad → codec encoder matches the golden",
  () => runChainOverCases(assertLatentShape),
);

test.skipIf(!RUNNABLE || !UPSTREAM_PARITY)(
  "e2e (real GPU): reference audio → normalisation → pad → codec encoder matches the golden latent" +
    UPSTREAM_PARITY_OPT_IN,
  async () => {
    let worst = 0;
    await runChainOverCases((name, result) => {
      assertLatentShape(name, result);
      const { rawLength, frames, actual, expected } = result;
      let maxAbs = 0;
      let absMax = 0;
      for (let i = 0; i < actual.length; i += 1) {
        maxAbs = Math.max(maxAbs, Math.abs(actual[i] - expected[i]));
        absMax = Math.max(absMax, Math.abs(expected[i]));
      }
      worst = Math.max(worst, maxAbs);
      console.log(
        `[e2e] irodori reference audio ${name}: T ${frames} / ${rawLength} samples → ` +
          `latent ${actual.length} elements / maxAbs ${maxAbs.toExponential(4)}` +
          ` (|latent| max ${absMax.toFixed(2)})`,
      );
      assertEquals(
        maxAbs < LATENT_ATOL,
        true,
        `${name}: the latent's maximum absolute difference ${maxAbs} exceeds atol ${LATENT_ATOL}` +
          " (do not loosen it; measure separately whether the preprocessing or the encoder moved)",
      );
    });
    console.log(`[e2e] irodori reference audio: worst maxAbs ${worst.toExponential(4)}`);
  },
);
