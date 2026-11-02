// 参照音声 → 話者 latent の**鎖ぜんぶ**の門（実 GPU）。
//
// `irodori_reference_test.ts` がホスト前処理だけを golden と突き合わせるのに対し、こちらは
// **生波形から latent まで**を 1 本に繋いで golden の `latent` と突き合わせる —
// 正規化 → reflect pad → `[1,T,1920]` へのフレーム化 → `codec_encoder`（実 GPU）。
//
// フレーム化はテンソルの読み替えでしかないが、**幅を取り違えても shape は合う**（1920 の
// 代わりに別の数で割ると T が変わるだけで例外は出ない）。ここが唯一その取り違えを掴む席。
//
// golden 側の `latent` は上流 `codec.encode_waveform` の出力そのもの
// （`tools/export-recipes/irodori/dacvae/host.py` が「ホスト鎖 → グラフ」と上流とのビット一致を
// 実測してから書く）なので、この門が閉じることは
// 「TS の参照音声経路が上流と同じ話者条件を作る」を意味する。
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

/** 実重み v4-small の運用値（`pipelineConfig` が運ぶ数と同じ）。 */
const SAMPLE_RATE = 48000;
const HOP_LENGTH = 1920;
const LATENT_DIM = 32;

/**
 * 生波形 → latent の全経路に使う許容誤差（絶対）。
 *
 * 実測（`atol = rtol = 0` の素の突合。2026-08-12 / 実 GPU・golden の LUFS 正規化 3 ケース）:
 *
 * | ケース      | T   | 複合 maxAbs   | encoder 単体 maxAbs | 前処理だけの寄与 | \|latent\| 上端 |
 * | ----------- | --- | ------------- | ------------------- | ---------------- | --------------- |
 * | ref-default | 190 | 2.1979e-5     | 2.5570e-5           | 1.1742e-5        | 5.31            |
 * | ref-short   | 10  | 8.7023e-6     | 8.1062e-6           | 2.2054e-6        | 3.44            |
 * | ref-odd     | 76  | **2.8685e-5** | 2.7344e-5           | 1.2457e-5        | 3.93            |
 *
 * 「encoder 単体」は golden の `padded` をそのままグラフへ流したときの差、「前処理だけの
 * 寄与」は TS 前処理の出力と golden の `padded` をそれぞれ流した結果どうしの差。
 * **前処理の寄与（1.25e-5）は encoder 自身の差（2.73e-5）より小さく**、複合が encoder 単体と
 * ほぼ同じ大きさに留まる — LUFS の f64/f32 差（利得の相対 2.5e-6）が conv の縮約順序差に
 * 埋もれている、というのがこの 3 行の読み。
 *
 * atol 2e-4 は実測最悪 2.8685e-5（ref-odd）の **7.0 倍**。runtime の encoder 単体門
 * （`e2e_dacvae_test.ts`〈0.15 の選別で削除〉の `ENCODER_TOLERANCE`）と同じ値になったのは偶然ではなく、上の
 * 「前処理の寄与が encoder の差に埋もれる」の帰結（それでも**独立に導出**する — 前処理側が
 * 悪化したときに、こちらの実測表が先に動く）。
 *
 * **式の取り違え**（目標 dB・pad の向き・フレーム幅）は latent の値域（±5.3）と同じ
 * O(0.1〜1) で出る。
 */
const LATENT_ATOL = 2e-4;

/** LUFS 正規化を通る golden ケース（`normalizeDb: -16` の 3 本 — この経路が持つ枝の全部）。 */
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
  if (spec === undefined) throw new Error(`${where}: '${key}' が無い`);
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

        // ここがパイプラインと同じ鎖（`pipeline.ts` の `encodeReferenceAudio`）。
        const padded = reflectPadToHop(normalizeReference(raw, SAMPLE_RATE).data, HOP_LENGTH);
        const frames = padded.length / HOP_LENGTH;
        const outputs = await session.run({
          wav: { dtype: "f32", shape: [1, frames, HOP_LENGTH], data: padded },
        });
        const tensor = outputs[encoder.graph.outputs[0]];
        if (tensor.dtype !== "f32") throw new Error(`encoder の出力が ${tensor.dtype}`);
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
    `${name}: latent の要素数（フレーム幅 ${HOP_LENGTH} の取り違えはここに出る）`,
  );
  assertEquals(actual.length / LATENT_DIM, frames, `${name}: latent のフレーム数`);
};

// The shape gate stays in the default run: a wrong frame width still yields a valid tensor, and
// only the element count against the golden latent catches it. The value comparison (atol) is
// upstream parity and is opt-in.
test.skipIf(!RUNNABLE)(
  "e2e(実GPU): 参照音声 → 正規化 → pad → codec encoder の latent の形が golden と一致する",
  () => runChainOverCases(assertLatentShape),
);

test.skipIf(!RUNNABLE || !UPSTREAM_PARITY)(
  "e2e(実GPU): 参照音声 → 正規化 → pad → codec encoder が golden の latent と一致する" +
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
        `[e2e] irodori 参照音声 ${name}: T ${frames} / ${rawLength} サンプル → ` +
          `latent ${actual.length} 要素 / maxAbs ${maxAbs.toExponential(4)}` +
          `（|latent| 上端 ${absMax.toFixed(2)}）`,
      );
      assertEquals(
        maxAbs < LATENT_ATOL,
        true,
        `${name}: latent の最大絶対差 ${maxAbs} が atol ${LATENT_ATOL} を超えた` +
          "（緩めず、前処理と encoder のどちらが動いたかを分けて実測する）",
      );
    });
    console.log(`[e2e] irodori 参照音声: 最悪 maxAbs ${worst.toExponential(4)}`);
  },
);
