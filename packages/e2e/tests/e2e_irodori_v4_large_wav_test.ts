/**
 * The **WAV sha256 check** of the v4-Large distribution `models/karume-irodori-v4-large/` (real GPU,
 * ADR 0151) — the v4-Large counterpart of `e2e_irodori_wav_test.ts`, with the same two cases and
 * the same reference audio, plus a third case whose caption the backbone's sliding window cuts
 * (the full-loop golden case `caption-window`): the whole chain from text to waveform (the internal
 * `Randn`, the duration predictor's choice of S, the 40-step Euler loop with CFG, the T5Gemma 2
 * backbone and the Gemma tokenizer, the tail trim, the borrowed v4-small codec's tile decode, the
 * cut to seconds, `encodeWav`) pinned **to one bit, as a regression detector in this execution
 * environment**. It does not look at upstream (the latent parity of `e2e_irodori_v4_large_test.ts`
 * does).
 *
 * The preset is `f16`, the most faithful one this model ships (there is no `f32` preset — ADR 0151
 * decision 3), as the RF and MeanFlow checks pin their most faithful preset (`f32`).
 *
 * The reference digests are rows per environment key in `fixtures/references/irodori.json` under
 * their own case IDs (`v4-large-voice-clone` / `v4-large-no-ref` / `v4-large-caption-window` —
 * ADR 0106). Switching to a tolerance is forbidden; a machine without a row SKIPs explicitly and the
 * reference gate at the end is red. Rows are made with `KARUME_REFERENCE=write` (existing rows are
 * never touched).
 *
 * MUST: the assets are the mirror (with the v4-small mirror it borrows the codec from) and the
 * reference audio `inputs/irodori/v4-small/samples/clone_ref1.wav`; an environment without them or
 * without a GPU SKIPs explicitly (ADR 0005).
 */

import { writeFile } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { parseManifest } from "@karume/hub";
import { type GenerateRequest, Pipeline } from "../../models/irodori.ts";
import { type DecodedWav, decodeWav, encodeWav } from "../../models/src/audio/wav.ts";
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

const MIRROR = mirrorRoot("karume-irodori-v4-large");
const MODEL = "v4-large";
const QUANT = "f16";
/** The reference audio of the RF check (48 kHz mono PCM16 — a hand-placed real asset). */
const REFERENCE_AUDIO = new URL(
  "../../../inputs/irodori/v4-small/samples/clone_ref1.wav",
  import.meta.url,
);
const references = openReferences(
  new URL("../../models/tests/fixtures/references/irodori.json", import.meta.url),
);
/** `outputs/verify/<environment key>/<date>_irodori-v4-large/` (safe to delete). */
const results = openResults("irodori-v4-large");

/**
 * The sampler knobs the rows are baked with (the values of the distribution's `pipelineConfig`
 * and of the full-loop golden's `meta.json`) — a moved declaration says so before the digest does.
 */
const REFERENCE_KNOBS = {
  steps: 40,
  initScale: 0.999,
  cfgMinT: 0.5,
  cfgMaxT: 1,
  cfgScales: { text: 3, speaker: 5, caption: 3 },
} as const;

type WavCase = {
  readonly id: string;
  readonly why: string;
  readonly text: string;
  readonly caption?: string;
  readonly withReference: boolean;
  readonly seed: number;
  /** The forward count of the 40 RF steps under the knobs above (the golden cases' counts). */
  readonly forwards: number;
};

/**
 * The voice description of the tokenizer goldens (`LONG_CAPTION` of
 * `tools/export-recipes/irodori/tokenizer_ref.py`, about 230 Gemma tokens). The full-loop golden
 * case `caption-window` passes it twice (459 caption tokens), beyond the backbone's 256-token
 * sliding window.
 */
const LONG_CAPTION =
  "落ち着いた低めの声の男性が、ゆっくりとした速さで話している。" +
  "呼びかけには「はい！」と短く答える。" +
  "声には少しかすれた響きがあり、文の終わりを丁寧に下げて話す癖がある。" +
  "ニュース番組のナレーターのように発音ははっきりしているが、硬すぎることはなく、" +
  "聞き手にそっと語りかけるような柔らかさも感じられる。" +
  "録音は静かな部屋で行われたようで、背景の雑音はほとんど無く、マイクとの距離は近い。" +
  "息継ぎの音がわずかに聞こえ、文と文の間には短い間が置かれている。" +
  "感情の起伏は小さく、全体として穏やかで安定した印象を与える。" +
  "語尾が伸びることは少なく、一つ一つの言葉を確かめるように区切って発音している。" +
  "笑いを含んだ声になる場面はほとんど無いが、冷たい印象ではなく、" +
  "相手の話を最後まで聞いてから答える落ち着きが声全体から伝わってくる。" +
  "年齢は四十代から五十代くらいに聞こえ、長年人前で話してきた人のような余裕がある。";

/**
 * The two cases of the RF check (the same sentences, seeds and reference audio), and the
 * full-loop golden case `caption-window` (its text, caption and seed) — the only case of the
 * default run that passes a caption beyond the backbone's window through the pipeline.
 */
const CASES: readonly WavCase[] = [
  {
    id: "v4-large-voice-clone",
    why: "reference audio + caption (wav path = LUFS normalization + codec_encoder, all 3 CFG branches)",
    text: "今日は近くの店まで歩いて行きました。とても良い天気でしたね。",
    caption:
      "若く元気な女性の声。カフェの店員のように、明るくハキハキとした少し高めのトーンで話している。",
    withReference: true,
    seed: 1234,
    forwards: 100,
  },
  {
    id: "v4-large-no-ref",
    why: "no reference / no caption (the speaker's zero short-circuit, the caption's CFG off)",
    text: "本日はお越しいただき、誠にありがとうございます。",
    withReference: false,
    seed: 1235,
    forwards: 60,
  },
  {
    id: "v4-large-caption-window",
    why: "no reference + a 459-token caption the backbone's sliding window cuts (text + caption CFG)",
    text: "今日は近くの店まで歩いて行きました。とても良い天気でしたね。",
    caption: LONG_CAPTION.repeat(2),
    withReference: false,
    seed: 1236,
    forwards: 80,
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
    `[karume] skipping the v4-Large WAV sha256 check (it needs the mirror ${MIRROR.pathname} with` +
      ` model '${MODEL}' and the v4-small mirror it borrows the codec from, and the reference audio` +
      ` ${REFERENCE_AUDIO.pathname})`,
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
      if (config.flowParameterization !== "rf_velocity") {
        throw new Error(
          "the distribution does not declare the RF sampler the digests were baked with",
        );
      }
      assertEquals(
        {
          steps: config.steps,
          initScale: config.initScale,
          cfgMinT: config.cfgMinT,
          cfgMaxT: config.cfgMaxT,
          cfgScales: { ...config.cfgScales },
        },
        REFERENCE_KNOBS,
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
        const evidence = references.observe(item.id, actual, check, dumped);
        const expected = expectedOf(check);
        await results.record({
          id: item.id,
          status: check.status,
          ...(expected === undefined ? {} : { expected }),
          actual,
          artifact: `${item.id}.wav`,
          elapsedMs,
        });
        assertEquals(audio.forwards, item.forwards, "the forward count of the RF loop with CFG");
        if (check.status === "fail") {
          throw new Error(
            `${item.id}: the output WAV's sha256 does not match the reference (expected ${check.expected}` +
              ` / actual ${actual} / S ${audio.frames} / artifact ${dumped.pathname}) — find the cause` +
              " before re-baking the row" +
              evidence,
          );
        }
      });
    }
  },
);

registerReferenceGate(references, { runnable: RUNNABLE, caseIds: CASE_IDS });
