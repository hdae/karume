/**
 * 母音検出の**配布形の門**（実 GPU）— 配布形 → `VowelDetectorPipeline.fromAssets` →
 * `detect` の `.lab` が、同じ音声を**実長でグラフへ直接通した** `.lab` とバイト一致するか
 * だけを見る。
 *
 * ## なぜこの門が要るのか（バケット時代の負債の検出器）
 *
 * かつて配布形は長さバケット 4 本を持ち、`detect` は入力を右ゼロ pad して丸めていた。逆方向
 * GRU が pad から状態を持ち帰るので、配布形の `.lab` は実長経路の `.lab` と ①20ms の境界
 * ずれ ②末尾に 40ms の `pau` が増減 ③発話中間に 40ms の `pau` が入る、の 3 型で割れており、
 * **その差を測る門がどこにも無かった**（実重み E2E は実長経路だけを固定していた）。
 * 記号長 1 グラフ（ADR 0056 / 0057）で pad が消えた今、両者は一致するはずで、ここがその
 * 一致を毎回踏む。
 *
 * MUST: 一致しなくなったら**配布形の側を疑う**（tolerance 化はしない — `.lab` は離散の
 * 成果物そのもので、丸める余地が無い）。パイプラインが pad や丸めを再導入したか、
 * `pipelineConfig` の宣言がグラフとずれたか、のどちらか。
 *
 * ## 絶対値の正本はここではない
 *
 * `.lab` の**中身**（どの音素がどこに出るか）を固定しているのは
 * `packages/runtime/tests/e2e_vowel_detector_test.ts`（実音声 4 本の全文 + WAV の sha256）。
 * ここは「2 経路が同じものを出す」だけを主張する — 期待 `.lab` の表を 2 か所に持つと、
 * 片方だけ採り直したときに黙ってずれる。`vowels` ケースだけは意味の側からも見る
 * （5 母音が順に出る = 両経路が揃って壊れている形の検出）。
 *
 * MUST: 資産は `models/karume-vowel-detector/`（配布形・untracked）と
 * `outputs/series/vowel-detector-crnn-epoch3/`（系列）と `outputs/demo/vowel-*.wav`。
 * 無い環境と GPU 無し環境は理由を出して**明示 SKIP** する（ADR 0005）。
 */

import { assert, assertEquals } from "@std/assert";
import { parseManifest, resolveFiles } from "@karume/hub";
import type { Manifest } from "@karume/hub";
import { acquireGpu, createSession, type KarumeModel, openModel } from "@karume/runtime";
import { decodeWav, VowelDetectorPipeline } from "../mod.ts";
import { parseMelBasis } from "../src/vowel-detector/pipeline.ts";
import { extractFeatures, FEATURE_DIM } from "../src/vowel-detector/features.ts";
import { logitsToSegments, toLab } from "../src/vowel-detector/postprocess.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/** 配布形（`karume dist --pipeline vowel-detector` の出力）。 */
const DIST_DIR = new URL("../../../models/karume-vowel-detector/", import.meta.url);
/** 系列（`export_vowel_detector.py` の出力 — 直接経路の相手）。 */
const SERIES_DIR = new URL(
  "../../../outputs/series/vowel-detector-crnn-epoch3/",
  import.meta.url,
);
const DEMO_DIR = new URL("../../../outputs/demo/", import.meta.url);

/** 実音声（`packages/runtime/tests/e2e_vowel_detector_test.ts` の `CASES` と同じ 4 本）。 */
const CASES = ["short", "vowels", "mid", "long"] as const;

/** `vowels` の意味の門（実重み E2E の `VOWEL_SEQUENCE` と同じ主張）。 */
const VOWEL_SEQUENCE = ["a", "i", "u", "e", "o"];

const INPUT_NAME = "features";
const TIME_STRIDE = 2;

const manifestText = await Deno.readTextFile(new URL("karume.json", DIST_DIR)).catch(
  () => undefined,
);
const seriesBytes = await Deno.readFile(new URL("model.safetensors", SERIES_DIR)).catch(
  () => undefined,
);
const ASSETS_AVAILABLE = manifestText !== undefined && seriesBytes !== undefined;
if (!ASSETS_AVAILABLE) {
  console.warn(
    `[karume] ${DIST_DIR.pathname} / ${SERIES_DIR.pathname} が揃っていないため母音検出の` +
      "配布形 E2E を SKIP する（生成: cd tools/exporter && uv run --frozen python " +
      "export_vowel_detector.py && uv run --frozen karume dist --pipeline vowel-detector）",
  );
}
const RUNNABLE = GPU_AVAILABLE && ASSETS_AVAILABLE;

/** 配布形の資産をローカルから読む（`fetchAssets` のローカル版 — 取得層を通さない）。 */
const loadLocalAssets = async (
  manifest: Manifest,
): Promise<Record<string, Uint8Array<ArrayBuffer>>> => {
  const files = resolveFiles(manifest, {});
  let assets: Record<string, Uint8Array<ArrayBuffer>> = {};
  for (const key of Object.keys(files)) {
    assets = { ...assets, [key]: await Deno.readFile(new URL(files[key].path, DIST_DIR)) };
  }
  return assets;
};

const openSeries = (): KarumeModel => {
  const bytes = seriesBytes as Uint8Array<ArrayBuffer>;
  return openModel(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
};

/**
 * 系列のグラフを実長で 1 回回して `.lab` を作る（配布形を通さない直接経路）。
 *
 * 特徴抽出と後処理はパイプラインと同じホスト実装を呼ぶ — ここで別実装を書くと、
 * 比べているものが「配布形経路 vs 直接経路」ではなく「実装 A vs 実装 B」になる。
 */
const directLab = async (
  model: KarumeModel,
  audio: Float32Array,
  melBasis: Float32Array,
): Promise<string> => {
  const features = extractFeatures(audio, melBasis);
  const usable = features.frames - (features.frames % TIME_STRIDE);
  const gpu = await acquireGpu();
  const session = await createSession(gpu, model);
  try {
    const outputs = await session.run({
      [INPUT_NAME]: {
        dtype: "f32",
        shape: [1, usable, FEATURE_DIM],
        data: features.data.slice(0, usable * FEATURE_DIM),
      },
    });
    const tensor = outputs[model.graph.outputs[0]];
    assert(tensor.dtype === "f32", `ロジットの dtype が ${tensor.dtype}`);
    return toLab(logitsToSegments(tensor.data, usable / TIME_STRIDE));
  } finally {
    await session.dispose();
    gpu.destroy();
  }
};

Deno.test({
  name: "母音検出 配布形の .lab: 実音声 4 本が実長の直接経路と完全一致（pad もバケットも無い）",
  ignore: !RUNNABLE,
  fn: async () => {
    const manifest = parseManifest(manifestText as string);
    const assets = await loadLocalAssets(manifest);
    const series = openSeries();
    // mel 基底は配布形の資産そのものを使う（直接経路も同じ行列で特徴を採る）。
    const melBytes = assets["mel_basis"];
    const melBasis = parseMelBasis(
      melBytes.buffer.slice(melBytes.byteOffset, melBytes.byteOffset + melBytes.byteLength),
    );

    await using pipeline = await VowelDetectorPipeline.fromAssets({ manifest, assets });
    for (const name of CASES) {
      const wav = decodeWav(await Deno.readFile(new URL(`vowel-${name}.wav`, DEMO_DIR)));
      const result = await pipeline.detect(wav.data);
      assertEquals(
        result.lab,
        await directLab(series, wav.data, melBasis),
        `${name}: 配布形経由の .lab が実長の直接経路と違う`,
      );
      assert(result.segments.length > 1, `${name}: 区間が 1 本しか出ていない`);
      if (name === "vowels") {
        // 両経路が揃って壊れている形の検出（一致だけでは意味を主張できない）。
        assertEquals(
          result.segments.map((segment) => segment.label).filter((label) => label !== "pau"),
          VOWEL_SEQUENCE,
          "「あ、い、う、え、お。」から 5 母音が順に出ていない",
        );
      }
    }
  },
});
