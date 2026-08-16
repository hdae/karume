// 実重み母音認識 CRNN の**実音声の全鎖**（WAV → 特徴抽出 → 実 GPU → `.lab`）の門
// （ADR 0005 の段 3）。
//
// 鎖の両端（`src/audio/wav.ts` / `src/vowel-detector/features.ts` /
// `src/vowel-detector/postprocess.ts`）は models の実装なので、この門は models 側にある。
// もともとは `packages/runtime/tests/e2e_vowel_detector_test.ts` に同居していたが、runtime の
// テストが models の実装を相対 import する**逆向きの依存**になっていたため、鎖ごとこちらへ
// 移した（SigLIP2 で先に踏んだ分担 — `image_preprocess_real_test.ts` 冒頭）。
//
// ## runtime 側との分担
//
// - **ロジットの数値突合**（合成 golden 4 ケース・torch CPU 期待値）は runtime 側が持つ
//   （`e2e_vowel_detector_test.ts`）。`.lab` だけだと「ロジットが動いてもラベルが変わらな
//   かった」変更が素通りするので、あちらが 1e-5 級の tolerance で押さえる。
// - **`.lab` の完全一致**（実音声・全鎖）は**ここ**。離散のラベル列なので tolerance を持たない
//   （PNG / WAV の sha256 門と同じ哲学 — 出口の成果物そのものを固定する）。
//
// 鎖の途中（ホスト側の特徴抽出と後処理）は `vowel_detector_host_test.ts` が Python 正本との
// パリティで押さえており、ここはその 2 つを実 GPU 推論で繋いだときに**意味のある `.lab` が
// 出る**ことを見る。
//
// ## 系列は 1 本（グラフが 1 本・長さは記号）
//
// 時間軸は記号 `T`（20ms 格子）で、グラフ入力は `[1, 2T, 83]`（ADR 0056 / 0057）。したがって
// **4 本の音声はすべて同じグラフ**を、それぞれの実長で回る。束縛は明示 seed を渡さず
// **入力 shape から解かせる**（`2T` の派生次元からの束縛 = ADR 0057 の経路を、この門が実測で
// 踏む唯一の席。配布形パイプラインの側は逆に明示 seed を渡す）。
//
// **pad は 1 要素も無い**（奇数フレームの端数 1 本だけを切り捨てる）。以前は配布形だけが
// 長さバケット + 右ゼロ pad を採っていて、pad が末尾のロジットを O(1) 動かすせいで `.lab` が
// この門と割れていた（実測: pad 2 フレーム = 40ms でも max abs diff 2.2〜3.2）。バケットが
// 消えた今は**配布形の `.lab` もここと完全一致**する（`e2e_vowel_detector_lab_test.ts` が
// 同じ 4 本で実測する）。
//
// ## 資産
//
// - グラフ: `outputs/series/vowel-detector-crnn-epoch3/`（`.gitignore` の `outputs/`）。
//   生成コマンドは {@link GENERATE_COMMAND} がそのまま正本。
// - 実音声: `outputs/demo/vowel-<ケース>.wav`（16kHz mono）。生成台本は
//   `examples/irodori/eval-audio.ts`（テキスト / seed / リサンプルの正本はあちら）で、日本語
//   TTS（Irodori）が焼いた 48kHz を 1/3 に間引いたもの。**素材の同一性は sha256 で固定する** —
//   焼き直したら期待 `.lab` も採り直しになる。
// - mel 基底: `tests/fixtures/vowel-detector/parity.json`（git 追跡・上流
//   `assets/feature_config.json` そのまま）。配布形は同じ行列を `assets` 席の safetensors で
//   配る（`dist.py --pipeline vowel-detector`）が、この門は配布形を経由しない。
//
// 資産が無い環境では**明示 SKIP** する（系列ごと・音声ごとに独立）。ADR 0005 の「全 SKIP は
// 明示 FAIL」門番は *GPU アダプタの有無* だけを見ており、この SKIP とは独立。golden ケースの
// 列挙と完全性（中途半端な資産は FAIL）は runtime 側の「資産の完全性」テストが持つ — ここが
// 要るのはグラフ本体と WAV だけなので、系列の有無も `model.safetensors` の有無で見る。

import { assert, assertEquals } from "@std/assert";
import { acquireGpu, createSession, type KarumeModel, openModel } from "@karume/runtime";
import { decodeWav } from "../src/audio/wav.ts";
import { extractFeatures, FEATURE_DIM, SAMPLE_RATE } from "../src/vowel-detector/features.ts";
import { logitsToSegments, toLab } from "../src/vowel-detector/postprocess.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/**
 * 実音声 1 本の宣言（グラフは 4 本で共有 — 違うのは長さだけ）。
 *
 * `frames` / `length` / `sha256` / `lab` は**全て実測値**で、音声を焼き直したら 4 つとも
 * 採り直しになる（{@link AUDIO_COMMAND} → {@link GENERATE_COMMAND} の順）。
 */
type Case = {
  /** 音声ケース名（`examples/irodori/eval-audio.ts` の `CASES` と 1:1）。 */
  readonly name: string;
  /** 何を測る素材か（人が読むための欄 — 台本側の `why` の写し）。 */
  readonly why: string;
  /** `extractFeatures` が返す 10ms フレーム数（波形長から決まる）。 */
  readonly frames: number;
  /** グラフへ渡す入力長 T10（= `frames` を偶数へ落としたもの）。 */
  readonly length: number;
  /** 16kHz WAV の sha256（golden を採った素材と同一であることを実行前に見る）。 */
  readonly sha256: string;
  /** 期待する `.lab` 全文（実 GPU 出力・torch CPU 出力の両方と一致することを確認済み）。 */
  readonly lab: string;
};

/**
 * 実走するケース。**列挙ではなくここで固定する** — 生成済みのものを拾う形にすると、
 * 一部だけ生成し忘れた環境で「緑だが未検証」になる。
 *
 * 長さは 2.88s / 3.48s / 8.04s / 12.40s に散らしてある（**同じ 18 ノードのグラフ**を 4 通りの
 * 束縛で回す — 展開列だった頃はここが 10,812 / 13,092 / 30,420 / 46,988 ノードの別グラフ
 * 4 本だった）。`vowels` だけは内容も検査に使う（{@link VOWEL_SEQUENCE}）。
 */
const CASES: readonly Case[] = [
  {
    name: "short",
    why: "2 秒級・1 文（「ありがとうございます。」）",
    frames: 285,
    length: 284,
    sha256: "0b48165fd11bf34c1f2b7904e4d53b5efa3f52c9baefc6a54566f850a9a30dd8",
    lab: `0.0000000 0.2000000 a
0.2000000 0.5000000 o
0.5000000 0.6400000 a
0.6400000 0.7000000 i
0.7000000 1.7400000 a
1.7400000 2.3800000 u
2.3800000 2.8000000 i
2.8000000 2.8400000 pau
`,
  },
  {
    name: "vowels",
    why: "3 秒級・5 母音を区切って読む（「あ、い、う、え、お。」）",
    frames: 345,
    length: 344,
    sha256: "ae3712e01da6906a214b46c52490d106e1dfa598528e611e8f0f85ce6207a829",
    lab: `0.0000000 0.0400000 pau
0.0400000 0.4000000 a
0.4000000 0.9600000 i
0.9600000 1.6000000 u
1.6000000 2.3000000 e
2.3000000 3.4400000 o
`,
  },
  {
    name: "mid",
    why: "8 秒級・3 文",
    frames: 801,
    length: 800,
    sha256: "b63d35c12d9d60af44754db209b907f03e3fd5f46f25e6bb0a7ccb5195df734a",
    lab: `0.0000000 0.0400000 pau
0.0400000 0.1600000 o
0.1600000 0.2200000 N
0.2200000 0.5400000 i
0.5400000 0.8800000 a
0.8800000 0.9200000 pau
0.9200000 1.8000000 o
1.8000000 2.0200000 a
2.0200000 2.1800000 o
2.1800000 2.3200000 e
2.3200000 2.5200000 o
2.5200000 2.7200000 i
2.7200000 2.9000000 e
2.9000000 2.9600000 N
2.9600000 3.0800000 i
3.0800000 3.2000000 e
3.2000000 3.3400000 u
3.3400000 3.7400000 e
3.7400000 4.6000000 a
4.6000000 4.6600000 N
4.6600000 4.8200000 o
4.8200000 4.9600000 i
4.9600000 5.0800000 e
5.0800000 5.2000000 a
5.2000000 5.3400000 e
5.3400000 5.4400000 u
5.4400000 5.5600000 i
5.5600000 5.8600000 a
5.8600000 5.9400000 pau
5.9400000 6.5600000 o
6.5600000 6.9000000 i
6.9000000 7.0200000 e
7.0200000 7.1400000 u
7.1400000 7.2600000 a
7.2600000 7.6000000 o
7.6000000 7.6600000 i
7.6600000 7.9000000 a
7.9000000 8.0000000 pau
`,
  },
  {
    name: "long",
    why: "12 秒級・2 文（この門で最長の束縛 T=618）",
    frames: 1237,
    length: 1236,
    sha256: "28013dab4551c353d3f95e51733e69d71de43a719bf3c4f120eaa1082013f59b",
    lab: `0.0000000 0.0400000 pau
0.0400000 0.2400000 o
0.2400000 0.5600000 e
0.5600000 0.9400000 a
0.9400000 1.0000000 pau
1.0000000 1.3800000 o
1.3800000 1.5000000 i
1.5000000 1.6000000 N
1.6000000 1.7400000 o
1.7400000 2.0200000 a
2.0200000 2.1600000 i
2.1600000 2.3800000 o
2.3800000 2.4800000 i
2.4800000 2.6200000 a
2.6200000 2.7400000 i
2.7400000 3.0200000 e
3.0200000 3.0800000 pau
3.0800000 3.4000000 u
3.4000000 3.5400000 i
3.5400000 3.6600000 o
3.6600000 3.9200000 a
3.9200000 4.0400000 i
4.0400000 4.3400000 a
4.3400000 4.4000000 i
4.4000000 4.5200000 u
4.5200000 4.6600000 a
4.6600000 4.7600000 i
4.7600000 4.9000000 o
4.9000000 5.1200000 u
5.1200000 5.2200000 i
5.2200000 5.3800000 a
5.3800000 7.1800000 i
7.1800000 7.3400000 a
7.3400000 7.4400000 i
7.4400000 7.6000000 u
7.6000000 7.6600000 N
7.6600000 7.8000000 e
7.8000000 7.9800000 o
7.9800000 8.3600000 a
8.3600000 8.4600000 i
8.4600000 8.6000000 u
8.6000000 8.6800000 N
8.6800000 8.8200000 e
8.8200000 9.1000000 o
9.1000000 9.1600000 pau
9.1600000 9.5400000 o
9.5400000 9.6800000 a
9.6800000 9.8200000 i
9.8200000 9.9600000 e
9.9600000 10.1400000 u
10.1400000 10.1800000 N
10.1800000 10.3400000 e
10.3400000 10.5000000 o
10.5000000 10.6000000 i
10.6000000 10.7400000 e
10.7400000 10.8400000 i
10.8400000 10.9400000 u
10.9400000 11.3400000 o
11.3400000 11.4800000 a
11.4800000 11.5800000 i
11.5800000 11.7000000 a
11.7000000 11.9400000 e
11.9400000 12.0000000 i
12.0000000 12.2600000 a
12.2600000 12.3600000 pau
`,
  },
];

/**
 * `vowels` ケースの `.lab` から無音を除いたラベル列。
 *
 * 期待 `.lab` の完全一致とは**主張が違う** — あちらは「前回と同じ値が出る」で、こちらは
 * 「5 母音を順に読ませたら 5 母音が順に出る」= 鎖が意味を持つことの主張。音声を焼き直して
 * 期待 `.lab` を採り直すとき、壊れた出力をそのまま固定してしまう事故はこちらが止める。
 */
const VOWEL_SEQUENCE: readonly string[] = ["a", "i", "u", "e", "o"];
const VOWEL_CASE = "vowels";

/** 8 クラス（`LIPSYNC_CLASSES` の本数 — 出力の最終軸）。 */
const CLASS_COUNT = 8;

const SERIES_NAME = "vowel-detector-crnn-epoch3";
const SERIES_ROOT = new URL(`../../../outputs/series/${SERIES_NAME}/`, import.meta.url);
const DEMO_DIR = new URL("../../../outputs/demo/", import.meta.url);
const FIXTURE_PATH = new URL("./fixtures/vowel-detector/parity.json", import.meta.url);
const MODEL_FILE = "model.safetensors";
const INPUT_NAME = "features";

const audioFile = (entry: Case): string => `vowel-${entry.name}.wav`;

/** SKIP 時にそのまま貼れる生成コマンド（グラフは 1 本 — 長さの指定は要らない）。 */
const GENERATE_COMMAND = "cd tools/export-recipes && uv run python -m vowel_detector.export";

/** 実音声そのものを焼き直すコマンド（テキスト / seed の正本は台本側）。 */
const AUDIO_COMMAND = "deno task demo:eval-audio --source <Irodori 配布形のパス>";

/** mel 基底のフィクスチャ（`melBasis` 以外の欄はこの門では使わない）。 */
type Fixture = { readonly melBasis: number[][] };

const melBasis: Float32Array = Float32Array.from(
  (JSON.parse(await Deno.readTextFile(FIXTURE_PATH)) as Fixture).melBasis.flat(),
);

/**
 * ファイルの有無。
 * MUST: NotFound 以外は伝播させる — 権限エラー等を「資産が無い」と読み替えると、
 * 実行されていない検証が SKIP として静かに緑になる。
 */
const exists = (url: URL): boolean => {
  try {
    return Deno.statSync(url).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

const readBuffer = async (url: URL): Promise<ArrayBuffer> => {
  const bytes = await Deno.readFile(url);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

/**
 * グラフの時間軸が**記号**（入力 `2T` / 出力 `T`）で焼かれていることを見る。
 *
 * MUST: 落とさない。長さを固定して焼いた古い形は入出力の名前も階数も同じなので、置き換わって
 * いても「その 1 長のケースだけ」は緑のまま通る（残り 3 本が別の理由で落ちたように見える）。
 *
 * NOTE: runtime 側 e2e にも同じ検査がある（あちらは合成 golden の側で踏む）。**共有しない**
 * のは、片方が消えたときにもう片方が黙って通る形にしないため。
 */
const assertSymbolicTimeAxis = (parsed: KarumeModel): void => {
  assertEquals(parsed.graph.symbols, ["T"], `${SERIES_NAME} の記号次元`);
  assertEquals(
    parsed.graph.inputs[0].shape,
    [1, "2T", FEATURE_DIM],
    `${SERIES_NAME} の入力 '${INPUT_NAME}' の宣言`,
  );
  assertEquals(
    parsed.graph.values[parsed.graph.outputs[0]].shape,
    [1, "T", CLASS_COUNT],
    `${SERIES_NAME} の出力の宣言`,
  );
};

/** 資産の有無（登録時点で必要なので同期で見る）。 */
const available = exists(new URL(MODEL_FILE, SERIES_ROOT));

if (!available) {
  console.warn(
    `[karume] ${SERIES_ROOT.pathname} に export 済み資産が無いため実音声の全鎖 E2E を ` +
      `SKIP する（重みがリポジトリ管理外）。生成: ${GENERATE_COMMAND}`,
  );
}

for (const entry of CASES) {
  /** 実音声の門。系列と WAV の**両方**が揃ってはじめて実走する。 */
  const audioAvailable = available && exists(new URL(audioFile(entry), DEMO_DIR));

  if (available && !audioAvailable) {
    console.warn(
      `[karume] 実音声ケース (${entry.name}) を SKIP する（${audioFile(entry)} が無い）。` +
        `生成: ${AUDIO_COMMAND}`,
    );
  }

  Deno.test({
    name: `母音検出 実音声の全鎖: ${entry.name} — WAV → 特徴 → 実 GPU → .lab（${entry.why}）`,
    ignore: !audioAvailable || !GPU_AVAILABLE,
    fn: async () => {
      const wavBytes = await Deno.readFile(new URL(audioFile(entry), DEMO_DIR));
      // ① 期待 `.lab` を採った音声と、いま読んでいる音声が同一であること。**tolerance では
      // 吸収されない差**（台本を回し直して期待値を採り直していない）を、実行の前に名指しで落とす。
      assertEquals(
        await sha256Hex(wavBytes),
        entry.sha256,
        `${audioFile(entry)} が期待 .lab を採った音声と違う（採り直す: ${AUDIO_COMMAND}）`,
      );

      const wav = decodeWav(wavBytes);
      assertEquals(wav.sampleRate, SAMPLE_RATE, `${audioFile(entry)} の周波数`);
      const features = extractFeatures(wav.data, melBasis);
      assertEquals(features.frames, entry.frames, `${entry.name} の 10ms フレーム数`);
      // 出力は 20ms 格子なので、奇数フレームの端数 1 本は落として偶数長で回す（切り捨てで
      // あって pad ではない — 冒頭の「pad は 1 要素も無い」）。
      const usable = features.frames - (features.frames % 2);
      assertEquals(usable, entry.length, `${entry.name} の偶数化フレーム数`);

      const parsed = openModel(await readBuffer(new URL(MODEL_FILE, SERIES_ROOT)));
      assertSymbolicTimeAxis(parsed);
      const [outputName] = parsed.graph.outputs;

      const gpu = await acquireGpu();
      const session = await createSession(gpu, parsed);
      let logits: Float32Array;
      try {
        const output = (await session.run({
          [INPUT_NAME]: {
            dtype: "f32",
            shape: [1, usable, FEATURE_DIM],
            data: features.data.slice(0, usable * FEATURE_DIM),
          },
        }))[outputName];
        // 判別子で絞る（Float32Array へのキャストは dtype がずれたときに黙って通る）。
        assert(output.dtype === "f32", `${entry.name}: ロジットの dtype が ${output.dtype}`);
        assertEquals(output.shape, [1, usable / 2, CLASS_COUNT], `${entry.name}: 出力の形`);
        logits = output.data;
      } finally {
        await session.dispose();
        gpu.destroy();
      }

      // ② `.lab` の完全一致（tolerance を持たない離散の門）。
      const segments = logitsToSegments(logits, usable / 2);
      assertEquals(toLab(segments), entry.lab, `${entry.name} の .lab`);

      if (entry.name === VOWEL_CASE) {
        // 期待 `.lab` の一致とは別の主張（{@link VOWEL_SEQUENCE}）。
        assertEquals(
          segments.map((segment) => segment.label).filter((label) => label !== "pau"),
          [...VOWEL_SEQUENCE],
          "「あ、い、う、え、お。」から 5 母音が順に出ていない",
        );
      }
    },
  });
}
