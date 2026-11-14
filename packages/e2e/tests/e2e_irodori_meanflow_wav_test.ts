/**
 * The **WAV sha256 check** of the MeanFlow distribution `models/karume-irodori-v4.1-small-mf/`
 * (real GPU, ADR 0150) — the MeanFlow counterpart of `e2e_irodori_wav_test.ts`, with the same two
 * cases and the same reference audio: the whole chain from text to waveform (the internal `Randn`,
 * the v4.1 duration predictor's choice of S, the 4-step MeanFlow loop with the interval embedding,
 * the tail trim, codec tile decode, the cut to seconds, `encodeWav`) pinned **to one bit, as a
 * regression detector in this execution environment**. It does not look at upstream (the latent
 * parity of `e2e_irodori_meanflow_test.ts` does).
 *
 * The reference digests are rows per environment key in `fixtures/references/irodori.json` under
 * their own case IDs (`meanflow-voice-clone` / `meanflow-no-ref` — ADR 0106). Switching to a
 * tolerance is forbidden; a machine without a row SKIPs explicitly and the reference gate at the end
 * is red. Rows are made with `KARUME_REFERENCE=write` (existing rows are never touched).
 *
 * MUST: the assets are the mirror (with the v4-small / v4.1-small mirrors it borrows from) and the
 * reference audio `inputs/irodori/v4-small/samples/clone_ref1.wav`; an environment without them or
 * without a GPU SKIPs explicitly (ADR 0005).
 */

import { writeFile } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { parseManifest } from "@karume/hub";
import { type GenerateRequest, Pipeline } from "../../models/irodori.ts";
import { type DecodedWav, decodeWav, encodeWav } from "../../models/mod.ts";
import { parseIrodoriPipelineConfig } from "../../models/src/irodori/config.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { mirrorDirectory, mirrorHasModel, mirrorRoot } from "../../models/tests/helpers/mirror.ts";
import {
  readFileIfPresent,
  readTextIfPresent,
} from "../../models/tests/helpers/read-if-present.ts";
import { assertRunningAdapter } from "../../runtime/tests/helpers/environment.ts";
import {
  announceCheck,
  expectedOf,
  openReferences,
  registerReferenceGate,
} from "../../runtime/tests/helpers/reference.ts";
import { openResults } from "../../runtime/tests/helpers/results.ts";

const MIRROR = mirrorRoot("karume-irodori-v4.1-small-mf");
const MODEL = "v4.1-small-mf";
const QUANT = "f32";
/** The reference audio of the RF check (48 kHz mono PCM16 — a hand-placed real asset). */
const REFERENCE_AUDIO = new URL(
  "../../../inputs/irodori/v4-small/samples/clone_ref1.wav",
  import.meta.url,
);
const references = openReferences(
  new URL("../../models/tests/fixtures/references/irodori.json", import.meta.url),
);
/** `outputs/verify/<environment key>/<date>_irodori-meanflow/` (safe to delete). */
const results = openResults("irodori-meanflow");

type WavCase = {
  readonly id: string;
  readonly why: string;
  readonly text: string;
  readonly caption?: string;
  readonly withReference: boolean;
  readonly seed: number;
};

/** The two cases of the RF check (the same sentences, seeds and reference audio). */
const CASES: readonly WavCase[] = [
  {
    id: "meanflow-voice-clone",
    why: "reference audio + caption (wav path = LUFS normalization + codec_encoder, 4 MeanFlow steps)",
    text: "今日は近くの店まで歩いて行きました。とても良い天気でしたね。",
    caption:
      "若く元気な女性の声。カフェの店員のように、明るくハキハキとした少し高めのトーンで話している。",
    withReference: true,
    seed: 1234,
  },
  {
    id: "meanflow-no-ref",
    why: "no reference / no caption (the speaker's zero short-circuit, 4 MeanFlow steps)",
    text: "本日はお越しいただき、誠にありがとうございます。",
    withReference: false,
    seed: 1235,
  },
];
const CASE_IDS: readonly string[] = CASES.map((item) => item.id);

const MIRROR_READY = mirrorHasModel(MIRROR, MODEL);
const manifestText = MIRROR_READY
  ? await readTextIfPresent(new URL("karume.json", MIRROR))
  : undefined;
const referenceBytes = await readFileIfPresent(REFERENCE_AUDIO);
const ASSETS_AVAILABLE = manifestText !== undefined && referenceBytes !== undefined;
if (!ASSETS_AVAILABLE) {
  console.warn(
    `[karume] skipping the MeanFlow WAV sha256 check (it needs the mirror ${MIRROR.pathname} with` +
      ` its referenced mirrors, and the reference audio ${REFERENCE_AUDIO.pathname})`,
  );
}
const RUNNABLE = GPU_AVAILABLE && ASSETS_AVAILABLE;
if (RUNNABLE) references.warnMissing(CASE_IDS);

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

describe.skipIf(!RUNNABLE)(
  `e2e (real GPU): the WAV of distribution ${MODEL} / quant ${QUANT} matches the reference sha256`,
  () => {
    let pipeline: Pipeline | undefined;
    let reference: DecodedWav;

    beforeAll(async () => {
      // This run may write reference rows, so the adapter the key was taken from must be running.
      await assertRunningAdapter();
      const config = parseIrodoriPipelineConfig(
        parseManifest(manifestText as string).models[MODEL].pipelineConfig,
      );
      // The rows were baked with the MeanFlow sampler at 4 steps — a moved declaration says so
      // before the digest does.
      assertEquals(
        [config.flowParameterization, config.steps],
        ["meanflow", 4],
        "the distribution's sampler differs from the one the reference digests were baked with",
      );
      reference = decodeWav(referenceBytes as Uint8Array<ArrayBuffer>);
      pipeline = await Pipeline.fromPretrained(mirrorDirectory(MIRROR), {
        model: MODEL,
        quant: QUANT,
      });
    });
    afterAll(() => pipeline?.[Symbol.asyncDispose]());

    for (const item of CASES) {
      it.skipIf(references.lacksReference(item.id))(`${item.id}: ${item.why}`, async () => {
        assert(pipeline !== undefined, "the pipeline is not open (beforeAll failed)");
        const request: GenerateRequest = {
          text: item.text,
          ...(item.caption === undefined ? {} : { caption: item.caption }),
          ...(item.withReference ? { speaker: { audio: reference } } : {}),
          seed: item.seed,
        };
        const started = performance.now();
        const audio = await pipeline.generate(request);
        const wav = encodeWav(audio.data, audio.sampleRate);
        const actual = await sha256Hex(wav);
        const elapsedMs = performance.now() - started;
        const dumped = results.artifact(`${item.id}.wav`);
        await writeFile(dumped, wav);
        console.log(
          `[e2e] irodori ${MODEL}/${QUANT} ${item.id}: ${(elapsedMs / 1000).toFixed(1)}s / S ${audio.frames} / ` +
            `forwards ${audio.forwards} / ${(audio.data.length / audio.sampleRate).toFixed(2)}s / sha256 ${actual}`,
        );
        const check = references.check(item.id, actual);
        announceCheck(item.id, check, actual);
        const expected = expectedOf(check);
        await results.record({
          id: item.id,
          status: check.status,
          ...(expected === undefined ? {} : { expected }),
          actual,
          artifact: `${item.id}.wav`,
          elapsedMs,
        });
        assertEquals(audio.forwards, 4, "one conditional forward per MeanFlow step");
        if (check.status === "fail") {
          throw new Error(
            `${item.id}: the output WAV's sha256 does not match the reference (expected ${check.expected}` +
              ` / actual ${actual} / S ${audio.frames} / artifact ${dumped.pathname}) — find the cause` +
              " before re-baking the row",
          );
        }
      });
    }
  },
);

registerReferenceGate(references, { runnable: RUNNABLE, caseIds: CASE_IDS });
