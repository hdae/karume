// `VowelDetectorPipeline` の**構築ガード**と、長さバケット / mel 基底資産の結線。GPU も実資産も
// 要らない範囲だけを見る（実 GPU の突合は `packages/runtime/tests/e2e_vowel_detector_test.ts` が
// 持つ — 重複させない。ホスト層の Python 正本とのパリティは
// `packages/models/tests/vowel_detector_host_test.ts`）。
//
// 押さえるのは 4 点:
//
// ① `fromAssets` は **manifest の契約違反を、資産を開く前・GPU を取りに行く前**に落とす
//    （`src/vowel-detector/pipeline.ts` の `openVowelDetectorState` が掲げる MUST）。観測の
//    仕掛けは SigLIP2 / BiRefNet と同じ — **全ケースで `assets` は空**にしておき、
//     - 契約違反ケースが「その違反の文言」で落ちる = 資産解析より前に落ちている
//     - 正しい manifest + 空 assets が `資産 'crnn_t250' が無い` で落ちる = 契約検査が全部
//       済んだ後に初めて資産へ触る（上の対偶）
//    の 2 つで門の順序そのものを縛る。
//
// ② `pipelineConfig` の**宣言 3 欄**（sampleRate / featureDim / classes）は受理集合が 1 値
//    きりで、外れた配布形はパース時に落ちる。とくに `classes` は**並びが id** なので、
//    置換された宣言が通ると `.lab` は成立したままラベルだけが入れ替わる。
//
// ③ **バケット選択**（`pickFrameLength`）— 入力長以上の最小を選び、上限超過は fail loudly。
//    ここは「黙って切り詰めない」という配布形の約束そのもので、境界（ちょうどの長さ・
//    1 フレーム超過）を名指しで踏む。
//
// ④ **mel 基底資産**（`parseMelBasis`）— テンソル名・dtype・形の 3 つを見る。基底がずれても
//    特徴は「それらしい別の値」になるだけで、shape も値域も合ったまま最後まで通る。

import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { parseManifest } from "@karume/hub";
import {
  parseVowelDetectorPipelineConfig,
  type VowelDetectorPipelineConfig,
} from "../src/vowel-detector/config.ts";
import {
  parseMelBasis,
  pickFrameLength,
  VowelDetectorPipeline,
} from "../src/vowel-detector/pipeline.ts";
import { FEATURE_DIM, MEL_BINS, N_MELS, SAMPLE_RATE } from "../src/vowel-detector/features.ts";
import { LIPSYNC_CLASSES } from "../src/vowel-detector/postprocess.ts";
import { writeSafetensors } from "../../../examples/sbv2/host/safetensors-write.ts";

/** 配布形が持つ長さバケット（`karume/dist.py` の `VOWEL_DETECTOR_FRAME_LENGTHS` と同じ並び）。 */
const FRAME_LENGTHS = [250, 500, 1000, 2000];

const fileRef = (path: string) => ({ path, size: 16, sha256: "a".repeat(64) });

/** `models/karume-vowel-detector/karume.json` の `pipelineConfig` 実物（4 欄）。 */
const PIPELINE_CONFIG: Record<string, unknown> = {
  sampleRate: SAMPLE_RATE,
  featureDim: FEATURE_DIM,
  classes: [...LIPSYNC_CLASSES],
  frameLengths: FRAME_LENGTHS,
};

/** 配布形の骨格（検査に要る欄だけ）。`patch` は `models["crnn-epoch3"]` の中身を上書きする。 */
const manifestText = (patch: Record<string, unknown> = {}): string =>
  JSON.stringify({
    format: "karume/2",
    generator: "karume/0.2.2",
    defaultModel: "crnn-epoch3",
    models: {
      "crnn-epoch3": {
        pipeline: "vowel-detector/1",
        weights: Object.fromEntries(
          FRAME_LENGTHS.map((length) => [
            `crnn_t${length}`,
            { f32: { file: fileRef(`crnn-epoch3/t${length}/model.f32.safetensors`) } },
          ]),
        ),
        assets: { mel_basis: fileRef("crnn-epoch3/features/mel-basis.safetensors") },
        quants: {
          f32: {
            weights: Object.fromEntries(
              FRAME_LENGTHS.map((length) => [`crnn_t${length}`, "f32"]),
            ),
            session: {},
          },
        },
        defaultQuant: "f32",
        pipelineConfig: PIPELINE_CONFIG,
        ...patch,
      },
    },
  });

const emptyAssets = {} as Record<string, Uint8Array<ArrayBuffer>>;

Deno.test("fromAssets: 存在しない model は利用可能な一覧を添えて落とす", async () => {
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => VowelDetectorPipeline.fromAssets({ manifest, assets: emptyAssets }, { model: "nope" }),
    Error,
    "model 'nope' は manifest に無い",
  );
});

Deno.test("fromAssets: pipeline の契約名が vowel-detector でない manifest を落とす", async () => {
  const manifest = parseManifest(manifestText({ pipeline: "sbv2/1" }));
  await assertRejects(
    () => VowelDetectorPipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "manifest の pipeline が 'sbv2/1'",
  );
});

Deno.test("fromAssets: 未知 major は fail loudly（検査責務は models 側 — ADR 0038 §1）", async () => {
  // 「古い実装 × 新しいリポ」の沈黙劣化を止める唯一の門。hub は major を検査しない。
  const manifest = parseManifest(manifestText({ pipeline: "vowel-detector/2" }));
  await assertRejects(
    () => VowelDetectorPipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "major に未対応",
  );
});

Deno.test("fromAssets: 存在しない quant は利用可能な一覧を添えて落とす", async () => {
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => VowelDetectorPipeline.fromAssets({ manifest, assets: emptyAssets }, { quant: "nope" }),
    Error,
    "quant 'nope' は manifest に無い",
  );
});

Deno.test("fromAssets: pipelineConfig の未知キーは構築時に落ちる", async () => {
  // 綴り違い（`frameLengths` に対する `frame_lengths`）が黙って既定へ縮退する経路を作らない。
  const manifest = parseManifest(
    manifestText({ pipelineConfig: { ...PIPELINE_CONFIG, frame_lengths: [200] } }),
  );
  await assertRejects(
    () => VowelDetectorPipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "pipelineConfig: 未知キー 'frame_lengths'",
  );
});

Deno.test("fromAssets: manifest 契約を全て満たして初めて資産へ触る（門の順序の対偶）", async () => {
  // 上の 5 ケースが「資産が空でも manifest の文言で落ちる」ことの裏返し。正しい manifest なら
  // 検査は資産まで進み、**最小のバケット**の不在で落ちる（= 契約検査は全て資産より前）。
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => VowelDetectorPipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    `資産 'crnn_t${FRAME_LENGTHS[0]}' が無い`,
  );
});

// ---- pipelineConfig のスキーマ（宣言 3 欄 + バケット）-------------------------

const config = (patch: Record<string, unknown> = {}): VowelDetectorPipelineConfig =>
  parseVowelDetectorPipelineConfig({ ...PIPELINE_CONFIG, ...patch });

Deno.test("pipelineConfig: 実物の 4 欄をそのまま読める", () => {
  const parsed = config();
  assertEquals(parsed.sampleRate, SAMPLE_RATE);
  assertEquals(parsed.featureDim, FEATURE_DIM);
  assertEquals([...parsed.classes], [...LIPSYNC_CLASSES]);
  assertEquals([...parsed.frameLengths], FRAME_LENGTHS);
});

Deno.test("pipelineConfig: 16kHz 以外の配布形は受理しない（リサンプラを持たない）", () => {
  assertThrows(
    () => config({ sampleRate: 22050 }),
    Error,
    "pipelineConfig.sampleRate: この実装が対応するのは 16000 だけ",
  );
});

Deno.test("pipelineConfig: 特徴次元が 83 でない配布形は受理しない", () => {
  assertThrows(
    () => config({ featureDim: 80 }),
    Error,
    "pipelineConfig.featureDim: この実装が対応するのは 83 だけ",
  );
});

Deno.test("pipelineConfig: クラスの**並び**が違う配布形は受理しない（並びが id）", () => {
  // 集合としては同じで並びだけが違う = ラベルが置換されるだけで .lab は完全に成立する形。
  const swapped = [...LIPSYNC_CLASSES];
  [swapped[0], swapped[1]] = [swapped[1], swapped[0]];
  assertThrows(
    () => config({ classes: swapped }),
    Error,
    "pipelineConfig.classes: この実装が対応するのは",
  );
});

Deno.test("pipelineConfig: 昇順でないバケット宣言は落とす（過大なグラフを黙って選ぶ）", () => {
  assertThrows(
    () => config({ frameLengths: [500, 250] }),
    Error,
    "pipelineConfig.frameLengths: 昇順でない",
  );
});

Deno.test("pipelineConfig: 奇数長・空のバケット宣言は落とす", () => {
  assertThrows(
    () => config({ frameLengths: [251] }),
    Error,
    "pipelineConfig.frameLengths: 正の 2 の倍数でない",
  );
  assertThrows(
    () => config({ frameLengths: [] }),
    Error,
    "pipelineConfig.frameLengths: 空でない配列でない",
  );
});

// ---- バケット選択（入力長 → 回すグラフ）--------------------------------------

Deno.test("バケット選択: 入力長以上の**最小**を選ぶ", () => {
  const parsed = config();
  assertEquals(pickFrameLength(parsed, 2), 250);
  assertEquals(pickFrameLength(parsed, 284), 500);
  assertEquals(pickFrameLength(parsed, 800), 1000);
  assertEquals(pickFrameLength(parsed, 1236), 2000);
});

Deno.test("バケット選択: ちょうどの長さは pad 無しでそのバケット（境界は下側に閉じる）", () => {
  const parsed = config();
  for (const length of FRAME_LENGTHS) {
    assertEquals(pickFrameLength(parsed, length), length, `t${length} ちょうど`);
    // 1 フレーム足りない入力も同じバケット（= 境界が「以上」で閉じている側の主張。
    // `>` で書くと、ちょうどの長さが 1 つ上のバケットへ黙って上がる）。
    assertEquals(pickFrameLength(parsed, length - 1), length, `t${length} の 1 つ手前`);
  }
});

Deno.test("バケット選択: 上限超過は fail loudly（黙って切り詰めない）", () => {
  const parsed = config();
  const error = assertThrows(
    () => pickFrameLength(parsed, 2001),
    Error,
    "音声が長すぎる",
  );
  // 何秒までなら通るのかが文言に出ていること（切り詰めの代わりに呼び出し側が区切るため）。
  assert(error.message.includes("2000 フレーム"), error.message);
  assert(error.message.includes("20.00 秒"), error.message);
});

// ---- mel 基底の資産 ----------------------------------------------------------

const melBasisBytes = (
  name = "mel_basis",
  shape: readonly number[] = [N_MELS, MEL_BINS],
): Uint8Array<ArrayBuffer> =>
  writeSafetensors(
    new Map([[name, {
      dtype: "F32" as const,
      shape,
      data: new Float32Array(shape.reduce((product, dim) => product * dim, 1)),
    }]]),
    {},
  );

Deno.test("mel 基底: [80, 257] の f32 を 1 テンソルだけ読む", () => {
  const basis = parseMelBasis(melBasisBytes().buffer);
  assertEquals(basis.length, N_MELS * MEL_BINS);
});

Deno.test("mel 基底: テンソル名が違えば入っているものを添えて落とす", () => {
  assertThrows(
    () => parseMelBasis(melBasisBytes("mel").buffer),
    Error,
    "テンソル 'mel_basis' が無い",
  );
});

Deno.test("mel 基底: 転置は要素数が同じでも落とす（形を 2 軸とも見る）", () => {
  assertThrows(
    () => parseMelBasis(melBasisBytes("mel_basis", [MEL_BINS, N_MELS]).buffer),
    Error,
    `期待は [${N_MELS}, ${MEL_BINS}]`,
  );
});

Deno.test("mel 基底: f32 でない格納形は落とす", () => {
  const bytes = writeSafetensors(
    new Map([["mel_basis", {
      dtype: "I32" as const,
      shape: [N_MELS, MEL_BINS],
      data: new Int32Array(N_MELS * MEL_BINS),
    }]]),
    {},
  );
  assertThrows(() => parseMelBasis(bytes.buffer), Error, "mel 基底が F32 でない");
});
