// `VowelDetectorPipeline` の**構築ガード**と、運用上限 / mel 基底資産の結線。GPU も実資産も
// 要らない範囲だけを見る（実 GPU の突合は `packages/runtime/tests/e2e_vowel_detector_test.ts`
//〈合成 golden のロジット〉と `e2e_vowel_detector_chain_test.ts`〈実音声の全鎖 → `.lab`〉が
// 持つ — 重複させない。ホスト層の Python 正本とのパリティは
// `packages/models/tests/vowel_detector_host_test.ts`）。
//
// 押さえるのは 5 点:
//
// ① `fromAssets` は **manifest の契約違反を、資産を開く前・GPU を取りに行く前**に落とす
//    （`src/vowel-detector/pipeline.ts` の `openVowelDetectorState` が掲げる MUST）。観測の
//    仕掛けは SigLIP2 / BiRefNet と同じ — **全ケースで `assets` は空**にしておき、
//     - 契約違反ケースが「その違反の文言」で落ちる = 資産解析より前に落ちている
//     - 正しい manifest + 空 assets が `資産 'crnn' が無い` で落ちる = 契約検査が全部
//       済んだ後に初めて資産へ触る（上の対偶）
//    の 2 つで門の順序そのものを縛る。
//
// ② グラフ宣言との突合（`assertGraph`）の**拒否経路**。`fromAssets` の中では実資産が
//    揃わないと踏めないので、門を直接叩く（`tests/helpers/stub-model.ts` が宣言だけの
//    `KarumeModel` を組む）。長さを固定して焼いた古い形は**入出力の名前も階数も同じ**なので、
//    門の綴りが `format/dims.ts` の正準表記からずれても正常系だけなら緑のまま通る。
//
// ③ `pipelineConfig` の**宣言 3 欄**（sampleRate / featureDim / classes）は受理集合が 1 値
//    きりで、外れた配布形はパース時に落ちる。とくに `classes` は**並びが id** なので、
//    置換された宣言が通ると `.lab` は成立したままラベルだけが入れ替わる。
//
// ④ **運用上限**（`assertFrameLimit`）— 上限超過は fail loudly。ここは「黙って切り詰め
//    ない」という配布形の約束そのもので、境界（ちょうどの長さ・1 フレーム超過）を名指しで
//    踏む。上限は配布形の宣言なので、TS 側に定数を持たない（`config.ts` の MUST）。
//
// ⑤ **mel 基底資産**（`parseMelBasis`）— テンソル名・dtype・形の 3 つを見る。基底がずれても
//    特徴は「それらしい別の値」になるだけで、shape も値域も合ったまま最後まで通る。

import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { parseManifest } from "@karume/hub";
import {
  parseVowelDetectorPipelineConfig,
  type VowelDetectorPipelineConfig,
} from "../src/vowel-detector/config.ts";
import {
  assertFrameLimit,
  assertGraph,
  parseMelBasis,
  VowelDetectorPipeline,
} from "../src/vowel-detector/pipeline.ts";
import { FEATURE_DIM, MEL_BINS, N_MELS, SAMPLE_RATE } from "../src/vowel-detector/features.ts";
import { LIPSYNC_CLASSES } from "../src/vowel-detector/postprocess.ts";
import { writeSafetensors } from "./helpers/safetensors-write.ts";
import { type StubDim, stubModel } from "./helpers/stub-model.ts";

/** 配布形が宣言する運用上限（`karume/dist.py` の `VOWEL_DETECTOR_MAX_FRAMES` と同じ数）。 */
const MAX_FRAMES = 60_000;

const fileRef = (path: string) => ({ path, size: 16, sha256: "a".repeat(64) });

/** `models/karume-vowel-detector/karume.json` の `pipelineConfig` 実物（4 欄）。 */
const PIPELINE_CONFIG: Record<string, unknown> = {
  sampleRate: SAMPLE_RATE,
  featureDim: FEATURE_DIM,
  classes: [...LIPSYNC_CLASSES],
  maxFrames: MAX_FRAMES,
};

/** 配布形の骨格（検査に要る欄だけ）。`patch` は `models["crnn-epoch3"]` の中身を上書きする。 */
const manifestText = (patch: Record<string, unknown> = {}): string =>
  JSON.stringify({
    format: "karume/3",
    generator: "karume/0.2.2",
    defaultModel: "crnn-epoch3",
    models: {
      "crnn-epoch3": {
        pipeline: "vowel-detector/1",
        weights: {
          crnn: { f32: { shards: [fileRef("crnn-epoch3/model.f32.safetensors")] } },
        },
        assets: { mel_basis: fileRef("crnn-epoch3/features/mel-basis.safetensors") },
        quants: { f32: { weights: { crnn: "f32" }, session: {} } },
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
  // 綴り違い（`maxFrames` に対する `max_frames`）が黙って既定へ縮退する経路を作らない。
  const manifest = parseManifest(
    manifestText({ pipelineConfig: { ...PIPELINE_CONFIG, max_frames: 200 } }),
  );
  await assertRejects(
    () => VowelDetectorPipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "pipelineConfig: 未知キー 'max_frames'",
  );
});

Deno.test("fromAssets: manifest 契約を全て満たして初めて資産へ触る（門の順序の対偶）", async () => {
  // 上の 5 ケースが「資産が空でも manifest の文言で落ちる」ことの裏返し。正しい manifest なら
  // 検査は資産まで進み、**CRNN グラフ**の不在で落ちる（= 契約検査は全て資産より前）。
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => VowelDetectorPipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "資産 'crnn' が無い",
  );
});

// ---- グラフ宣言との突合（拒否経路）------------------------------------------

/** 実配布形と同じ pipelineConfig（この節は宣言の突合だけを見る）。 */
const graphConfig = parseVowelDetectorPipelineConfig(PIPELINE_CONFIG);

/** 記号長 1 グラフ（ADR 0056 / 0057）の宣言。`patch` で 1 点だけ壊す。 */
const crnnGraph = (
  patch: {
    readonly symbols?: readonly string[];
    readonly inputName?: string;
    readonly inputShape?: readonly StubDim[];
    readonly outputShape?: readonly StubDim[];
  } = {},
) =>
  stubModel({
    symbols: patch.symbols ?? ["T"],
    inputs: [{
      name: patch.inputName ?? "features",
      shape: patch.inputShape ?? [1, "2T", FEATURE_DIM],
    }],
    outputs: ["logits"],
    values: { logits: patch.outputShape ?? [1, "T", LIPSYNC_CLASSES.length] },
  });

Deno.test("assertGraph: 記号長で焼かれたグラフは時間軸の記号名を返す", () => {
  assertEquals(assertGraph(crnnGraph(), graphConfig), "T");
});

Deno.test("assertGraph: 長さを固定して焼いた古い形は記号次元で落ちる", () => {
  // 入出力の名前も階数も同じなので、同じ席に置かれても構築は通ってしまう形。
  assertThrows(
    () =>
      assertGraph(
        crnnGraph({
          symbols: [],
          inputShape: [1, 2000, FEATURE_DIM],
          outputShape: [1, 1000, LIPSYNC_CLASSES.length],
        }),
        graphConfig,
      ),
    Error,
    "VowelDetectorPipeline: グラフの記号次元が []",
  );
});

Deno.test("assertGraph: 入力の時間軸から 2 倍の係数が抜けたら落とす", () => {
  // 係数が抜けた配布形は `.lab` の時間が 2 倍に伸びるだけで、形は成立する。
  assertThrows(
    () => assertGraph(crnnGraph({ inputShape: [1, "T", FEATURE_DIM] }), graphConfig),
    Error,
    "VowelDetectorPipeline: グラフ入力の形が [1, T, 83]、期待は [1, 2T, 83]",
  );
});

Deno.test("assertGraph: 入力名が features でなければ落とす", () => {
  assertThrows(
    () => assertGraph(crnnGraph({ inputName: "input" }), graphConfig),
    Error,
    "VowelDetectorPipeline: グラフ入力が 'input'",
  );
});

Deno.test("assertGraph: 出力のクラス数が宣言と違えば落とす", () => {
  // クラス数がずれると `.lab` は成立したままラベルの割り当てだけが崩れる。
  assertThrows(
    () => assertGraph(crnnGraph({ outputShape: [1, "T", 5] }), graphConfig),
    Error,
    "VowelDetectorPipeline: グラフ出力の形が [1, T, 5]",
  );
});

// ---- pipelineConfig のスキーマ（宣言 3 欄 + 運用上限）------------------------

const config = (patch: Record<string, unknown> = {}): VowelDetectorPipelineConfig =>
  parseVowelDetectorPipelineConfig({ ...PIPELINE_CONFIG, ...patch });

Deno.test("pipelineConfig: 実物の 4 欄をそのまま読める", () => {
  const parsed = config();
  assertEquals(parsed.sampleRate, SAMPLE_RATE);
  assertEquals(parsed.featureDim, FEATURE_DIM);
  assertEquals([...parsed.classes], [...LIPSYNC_CLASSES]);
  assertEquals(parsed.maxFrames, MAX_FRAMES);
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

Deno.test("pipelineConfig: 奇数・非整数・非正の上限宣言は落とす", () => {
  // グラフ入力は `2T` なので、奇数の上限は「その 1 本だけ通らない上限」= 意味が壊れている。
  for (const value of [251, 0, -2, 2.5, "2000"]) {
    assertThrows(
      () => config({ maxFrames: value }),
      Error,
      "pipelineConfig.maxFrames: 正の 2 の倍数でない",
      `maxFrames=${JSON.stringify(value)}`,
    );
  }
});

Deno.test("pipelineConfig: 上限の欠落は落とす（既定へ縮退しない）", () => {
  const { maxFrames: _dropped, ...rest } = PIPELINE_CONFIG;
  assertThrows(
    () => parseVowelDetectorPipelineConfig(rest),
    Error,
    "pipelineConfig.maxFrames: 無い",
  );
});

// ---- 運用上限（入力長の門）---------------------------------------------------

Deno.test("運用上限: ちょうどの長さは通り、1 フレーム超過で落ちる（境界は上側に閉じる）", () => {
  const parsed = config();
  assertFrameLimit(parsed, 2);
  assertFrameLimit(parsed, MAX_FRAMES);
  const error = assertThrows(
    () => assertFrameLimit(parsed, MAX_FRAMES + 2),
    Error,
    "音声が長すぎる",
  );
  // 何秒までなら通るのかが文言に出ていること（切り詰めの代わりに呼び出し側が区切るため）。
  assert(error.message.includes(`${MAX_FRAMES} フレーム`), error.message);
  assert(error.message.includes("600.00 秒"), error.message);
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
