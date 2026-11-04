// The **build guards** of `Pipeline`. Only the range that needs neither a GPU nor real assets is
// checked here (the real-GPU comparisons belong to the `e2e_irodori_*_test.ts` files).
//
// Two points are pinned here:
//  ① `fromAssets` rejects **a violation of the manifest contract before going to acquire the GPU**
//     (the MUST that `admitIrodori` of `src/irodori/admission.ts` states). If the order drifts, in
//     an environment without a GPU it turns into another exception and the reader cannot tell
//     "what was wrong".
//  ② The build's `signal` works **both at the entry and after execution has started** (if the
//     assembly after the download cannot be aborted, there is a window where the UI's cancel button
//     does nothing). The latter is checked with empty assets up to "the first stage boundary" —
//     the boundaries after that need real assets and a GPU, so they cannot be checked here.
//
// The observation setup: **every case has the 9 containers present** (declarations only). So
//  - a manifest-contract violation case fails with "that violation's message" = even with the
//    assets present, the manifest check comes first (the containers are opened before admission —
//    `assetComponentOpener` returns a synchronous supply entry, so it opens every component first)
//  - an empty Record fails with `component 'backbone' has no container` (the receiving side's
//    diagnostic)
// and the two together pin the order of the checks itself. The 12-point check against the graph
// declarations is exercised by `irodori_admission_test.ts` with the same declarations as the real
// thing, so it is not covered here. The decode check of asset JSON (`assetJson`) cannot be reached
// from `fromAssets` for the same reason, so the 2 tests near the end **drive the check directly**.
// `copyLatents` of `denoise-step` (`latentSnapshot`) is also a pure function that can be pinned
// without a GPU, so the last test drives it directly as well — if it breaks, **another step's
// latent silently** reaches the subscriber, and the real-GPU WAV check stays green because it does
// not go through the observation hook.

import { test } from "vite-plus/test";
import {
  assertEquals,
  assertNotStrictEquals,
  assertRejects,
  assertStrictEquals,
  assertThrows,
} from "@std/assert";
import { parseManifest } from "@karume/hub";
import { parseIrodoriPipelineConfig } from "../src/irodori/config.ts";
import { assetJson } from "../src/irodori/admission.ts";
import { latentSnapshot } from "../src/irodori/dit-loop.ts";
import { assertIrodoriRequest, Pipeline } from "../src/irodori/pipeline.ts";
import { ModelInputError } from "../src/errors.ts";
import { declaredContainer, partAssets, tensorlessContainer } from "./helpers/container-fixture.ts";

const FILE = {
  path: "tokenizer.json",
  size: 16,
  sha256: "a".repeat(64),
};

/** The graph assets' names (in the order `openIrodoriState` looks them up with `assetBuffer`). */
const WEIGHT_NAMES = [
  "backbone",
  "text_proj",
  "caption_proj",
  "speaker",
  "duration",
  "dit",
  "dit_context",
  "codec_decoder",
  "codec_encoder",
] as const;

/** The real `pipelineConfig` of `models/karume-irodori-v4-small/karume.json` (23 fields). */
const PIPELINE_CONFIG: Record<string, unknown> = {
  maxTextLen: 256,
  maxCaptionLen: 512,
  speakerRows: 751,
  ditSymMax: 750,
  frameRate: 25,
  sampleRate: 48000,
  hopLength: 1920,
  codecHaloFrames: 8,
  latentDim: 32,
  speakerPatchSize: 4,
  speakerDim: 768,
  textDim: 512,
  captionDim: 512,
  timestepEmbedDim: 512,
  steps: 40,
  initScale: 0.999,
  cfgMinT: 0.5,
  cfgMaxT: 1,
  cfgScales: { text: 3, speaker: 5, caption: 3 },
  minSeconds: 0.5,
  maxSeconds: 30,
  speakerUncondMode: "mask",
  cfgGuidanceMode: "independent",
};

/**
 * The distribution's skeleton (only the fields the checks need). `patch` overwrites the contents of
 * `models["v4-small"]`.
 */
const manifestText = (patch: Record<string, unknown> = {}): string => {
  let weights: Record<string, unknown> = {};
  let mapping: Record<string, string> = {};
  for (const name of WEIGHT_NAMES) {
    weights = { ...weights, [name]: { f32: declaredContainer(`${name}/model.f32`) } };
    mapping = { ...mapping, [name]: "f32" };
  }
  return JSON.stringify({
    format: "karume/5",
    generator: "karume/0.1.0",
    defaultModel: "v4-small",
    models: {
      "v4-small": {
        pipeline: "irodori/1",
        weights,
        assets: { tokenizer: { ...FILE, path: "tokenizer.json" } },
        quants: { f32: { weights: mapping, session: {} } },
        defaultQuant: "f32",
        pipelineConfig: PIPELINE_CONFIG,
        ...patch,
      },
    },
  });
};

const emptyAssets = {} as Record<string, Uint8Array<ArrayBuffer>>;

/**
 * The 9 containers that can be opened (minimal declarations — the graph check itself is exercised
 * by `irodori_admission_test.ts` with the same declarations as the real thing). `tokenizer` is not
 * included: it is left as the one that fails after every check has passed.
 */
const COMPONENTS: Record<string, Uint8Array<ArrayBuffer>> = {};
for (const name of WEIGHT_NAMES) {
  Object.assign(
    COMPONENTS,
    partAssets(
      name,
      await tensorlessContainer(name, {
        inputs: [{ name: "x", shape: [1, 4] }],
        output: { name: "y", shape: [1, 4] },
      }),
    ),
  );
}

/** A manifest with only `pipelineConfig` replaced (the rest stays the skeleton). */
const withConfig = (config: Record<string, unknown>): string =>
  manifestText({ pipelineConfig: config });

test("fromAssets: a missing model fails with the list of available ones", async () => {
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENTS }, { model: "nope" }),
    Error,
    "model 'nope' is not in the manifest",
  );
});

test("fromAssets: rejects a manifest whose pipeline contract name is not irodori", async () => {
  const manifest = parseManifest(manifestText({ pipeline: "sbv2/1" }));
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENTS }),
    Error,
    "the manifest pipeline is 'sbv2/1'",
  );
});

test("fromAssets: an unknown major fails loudly (the check is the models side's job — ADR 0038 §1)", async () => {
  // The only check that stops the silent degradation of "an old implementation × a new repo". The
  // hub does not check the major.
  const manifest = parseManifest(manifestText({ pipeline: "irodori/2" }));
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENTS }),
    Error,
    "has an unsupported major",
  );
});

test("fromAssets: a missing quant fails with the list of available ones", async () => {
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENTS }, { quant: "nope" }),
    Error,
    "quant 'nope' is not in the manifest",
  );
});

test("fromAssets: an unknown pipelineConfig key fails at build time", async () => {
  // No path where a misspelling (`step` for `steps`) silently degrades to a default (the MUST of
  // config.ts).
  const manifest = parseManifest(withConfig({ ...PIPELINE_CONFIG, step: 40 }));
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENTS }),
    Error,
    "pipelineConfig: unknown key 'step'",
  );
});

test("fromAssets: a missing pipelineConfig field fails at build time", async () => {
  // If a missing field were filled with a default, only the host would hold another number while
  // **the shapes still match**, a silent wrong value (the MUST at the top of config.ts — the
  // manifest is the source of truth for model-specific numbers).
  const { hopLength: _dropped, ...missing } = PIPELINE_CONFIG;
  const manifest = parseManifest(withConfig(missing));
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENTS }),
    Error,
    "pipelineConfig.hopLength: missing",
  );
});

test("fromAssets: a missing component container fails with the spellings of the 2 forms (the receiving side's diagnostic)", async () => {
  // The flip side of the 6 cases above. A Record without the containers fails on **the missing
  // component** (not with a manifest message) = the contrapositive showing that the cases above are
  // not dragged in by missing assets.
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "component 'backbone' has no container",
  );
});

// ---- Aborting the build (`options.signal`) ----------------------------------------
//
// The UI's cancel button must also work during the assembly after the download (asset parsing →
// acquireGpu). Checked with the same manifest + empty assets as the checks above: what fails with
// "asset 'backbone' is missing" without a signal fails with `signal.reason` when already
// aborted / aborted while running.

test("fromAssets: an already aborted signal rejects with the reason itself before touching the assets", async () => {
  const manifest = parseManifest(manifestText());
  const controller = new AbortController();
  const reason = new Error("cancel button");
  controller.abort(reason);
  const error = await assertRejects(() =>
    Pipeline.fromAssets({ manifest, assets: COMPONENTS }, { signal: controller.signal }),
  );
  // Not wrapped (the consumer can identify its own abort by `error === controller.signal.reason`).
  assertStrictEquals(error, reason);
});

test("fromAssets: an abort that arrives after execution has started also works at the first stage boundary", async () => {
  // The check above sees only "aborted before the call". The cancel button is pressed **while the
  // assembly is running**, so a stage-boundary check is a dead letter unless it yields to the event
  // loop first (abort() is delivered as a task — it is never observed during synchronous parsing).
  // Set up before the call, it fails with the reason instead of the asset error ("asset
  // 'backbone' is missing").
  const manifest = parseManifest(manifestText());
  const controller = new AbortController();
  const reason = new Error("cancel button (while running)");
  setTimeout(() => controller.abort(reason), 0);
  const error = await assertRejects(() =>
    Pipeline.fromAssets({ manifest, assets: COMPONENTS }, { signal: controller.signal }),
  );
  assertStrictEquals(error, reason);
});

// ---- Decoding asset JSON (invalid UTF-8 is not silently replaced) ------------------

/**
 * An asset that is closed as JSON but has one invalid UTF-8 byte (0xff) mixed into a string value.
 */
const brokenUtf8Asset = (): Uint8Array<ArrayBuffer> =>
  Uint8Array.from([
    ...new TextEncoder().encode('{"unkId":'),
    0x22,
    0xff,
    0x22,
    ...new TextEncoder().encode("}"),
  ]);

test("assetJson: an invalid UTF-8 asset fails with the decode-stage message (it does not reach the JSON stage)", () => {
  const broken = brokenUtf8Asset();
  // Pinning the premise: the default TextDecoder replaces 0xff with U+FFFD, so reading with
  // replacement silently lets **valid JSON with different contents** through. That path is closed.
  assertEquals(
    JSON.parse(new TextDecoder().decode(broken)),
    { unkId: "�" },
    "the premise that the replacing decode does not give valid JSON broke (this test's subject is gone)",
  );
  assertThrows(
    () => assetJson({ tokenizer: broken }, "tokenizer"),
    Error,
    "irodori: asset 'tokenizer' cannot be read as UTF-8",
  );
});

test("assetJson: a JSON syntax error fails with a message different from decode (the valid domain is unchanged)", () => {
  const truncated = new TextEncoder().encode('{"unkId":');
  assertThrows(
    () => assetJson({ tokenizer: truncated }, "tokenizer"),
    Error,
    "irodori: asset 'tokenizer' cannot be read as JSON",
  );
  // Correct UTF-8 passes as is, even with multi-byte characters.
  const valid = new TextEncoder().encode('{"unkId":0,"space":"あ"}');
  assertEquals(assetJson({ tokenizer: valid }, "tokenizer"), { unkId: 0, space: "あ" });
});

test("assertIrodoriRequest: a misspelled request knob fails before the heavy computation", () => {
  // These checks originally existed only **after** latent generation (`planCodecTiles` right before
  // decode, `Randn` at stage ⑦), so they failed after consuming the 1.26GB weight load and every
  // DiT step. Borrowing the same accepted set at the entry keeps a misspelling from costing compute
  // time.
  const config = parseIrodoriPipelineConfig(PIPELINE_CONFIG);
  const text = "テスト";

  // codecTileFrames is "an integer larger than 2 halos". Against the default halo 8, 10 leaves no
  // taken range (`10 <= 16`), and passed through it would run up to right before decode and fail.
  for (const codecTileFrames of [10, 0, 1.5]) {
    assertThrows(
      () => assertIrodoriRequest({ text, codecTileFrames }, config),
      ModelInputError,
      "is not an integer greater than two halos",
    );
  }
  for (const seed of [-1, 1.5]) {
    assertThrows(
      () => assertIrodoriRequest({ text, seed }, config),
      ModelInputError,
      "non-negative safe integer",
    );
  }
  assertThrows(
    () => assertIrodoriRequest({ text, durationSeconds: NaN }, config),
    ModelInputError,
    "durationSeconds",
  );

  // Valid values pass (the evidence that the check does not always fail is on the side of the 6
  // cases above).
  assertIrodoriRequest({ text, codecTileFrames: 182, seed: 0, durationSeconds: 3 }, config);
  assertIrodoriRequest({ text }, config);
});

test("assertIrodoriRequest: a speaker reference shorter than 2 tokens (the export's Dim(min=2)) fails before the GPU runs", () => {
  // The reference is patched at stage ④ (hop 1920 samples = 1 codec frame, patch width 4 frames =
  // 1 token). Without the check at the entry, after running the text / caption stages and the codec
  // encoder on the GPU, 0 tokens give a plain Error and 1 token sends S = 1, below the lower bound,
  // to the speaker graph.
  const config = parseIrodoriPipelineConfig(PIPELINE_CONFIG);
  const text = "テスト";
  const audio = (samples: number) => ({
    audio: { data: new Float32Array(samples), sampleRate: 48000 },
  });
  // 0 samples, 80ms (2 frames = 0 tokens), exactly 7 frames (1 token).
  for (const samples of [0, 3840, 7 * 1920]) {
    assertThrows(
      () => assertIrodoriRequest({ text, speaker: audio(samples) }, config),
      ModelInputError,
      "tokens",
    );
  }
  // The tail is rounded up to a multiple of hop by reflect pad, so 7 frames + 1 sample is 8 frames
  // = 2 tokens.
  assertIrodoriRequest({ text, speaker: audio(7 * 1920 + 1) }, config);
  assertThrows(
    () =>
      assertIrodoriRequest(
        {
          text,
          speaker: { audio: { data: new Float32Array(8 * 1920), sampleRate: 24000 } },
        },
        config,
      ),
    ModelInputError,
    "resampler",
  );

  const latent = (frames: number) => ({ latent: new Float32Array(frames * 32) });
  for (const frames of [0, 4, 7]) {
    assertThrows(
      () => assertIrodoriRequest({ text, speaker: latent(frames) }, config),
      ModelInputError,
      "is less than",
    );
  }
  assertIrodoriRequest({ text, speaker: latent(8) }, config);
});

test("assertIrodoriRequest: checks the default tile even for a request that does not pass codecTileFrames", () => {
  // Callers are not obliged to pass `codecTileFrames`, so a form where "not passing it passes"
  // would make a distribution with a broken relation between the default tile and
  // `codecHaloFrames` **always** fail after the DiT loop.
  const config = parseIrodoriPipelineConfig({ ...PIPELINE_CONFIG, codecHaloFrames: 91 });
  assertThrows(
    () => assertIrodoriRequest({ text: "テスト" }, config),
    ModelInputError,
    "tileFrames 182 is not an integer greater than two halos (182)",
  );
});

test("latentSnapshot: copies the latent as of binding (advancing the step does not change the copy)", () => {
  const shape = [2, 2];
  // Reproducing the DiT loop: `eulerStep` is a pure function, so `x` is replaced by **a new array**
  // every step. The event entry is made and passed per step.
  const step1 = Float32Array.from([1, 2, 3, 4]);
  const first = latentSnapshot(step1, shape);
  const step2 = Float32Array.from([5, 6, 7, 8]);
  const second = latentSnapshot(step2, shape);

  // Even called after all generation is done, it returns the value "as of creation". An
  // implementation that captures the loop variable would give [5, 6, 7, 8] for both here (a mix-up
  // the subscriber cannot detect).
  assertEquals(Array.from(first().data), [1, 2, 3, 4]);
  assertEquals(Array.from(second().data), [5, 6, 7, 8]);
  assertEquals(first().shape, shape);

  // What it returns is **a separate copy** every time. Even if the subscriber rewrites the array it
  // received, neither the next copy nor the pipeline's array is affected (structurally ruling out
  // the accident of a held reference).
  const copy = first();
  assertNotStrictEquals(copy.data, step1);
  copy.data[0] = 99;
  assertEquals(Array.from(first().data), [1, 2, 3, 4]);
  assertEquals(Array.from(step1), [1, 2, 3, 4]);
  assertNotStrictEquals(copy.data, first().data);

  // `shape` gets the same treatment (the same MUST as anima's `latentSnapshot`). Returned by
  // reference, two copies from calling `copyLatents()` twice in the same step would share one
  // array, and if the subscriber rewrote the first one's `shape`, the second copy would silently
  // claim another shape.
  assertNotStrictEquals(copy.shape, shape);
  // The two copies do not share `shape` (the W-Q6-2 breakage shows up here — by reference,
  // rewriting the first copy would spread to the second copy).
  assertNotStrictEquals(copy.shape, first().shape);
  shape[0] = 99;
  assertEquals(copy.shape, [2, 2]);
});
