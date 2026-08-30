// 参照音声 → 話者 latent の**鎖ぜんぶ**の門（実 GPU）。
//
// `irodori_reference_test.ts` がホスト前処理だけを golden と突き合わせるのに対し、こちらは
// **生波形から latent まで**を 1 本に繋いで golden の `latent` と突き合わせる —
// 正規化 → reflect pad → `[1,T,1920]` へのフレーム化 → `codec_encoder`（実 GPU）。
//
// フレーム化はテンソルの読み替えでしかないが、**幅を取り違えても shape は合う**（1920 の
// 代わりに別の数で割ると T が変わるだけで例外は出ない）。ここが唯一その取り違えを掴む席。
//
// golden 側の `latent` は上流 `codec.encode_waveform` の出力そのもの（`dacvae_host.py` が
// 「ホスト鎖 → グラフ」と上流とのビット一致を実測してから書く）なので、この門が閉じることは
// 「TS の参照音声経路が上流と同じ話者条件を作る」を意味する。
//
// 資産が欠けた環境と GPU 無し環境は生成コマンド付きで**明示 SKIP**する（ADR 0005）。

import { assertEquals } from "@std/assert";
import { acquireGpu, parseSafetensors, prepareModel } from "@karume/runtime";
import { normalizeReference, reflectPadToHop } from "../src/irodori/host/reference.ts";
import {
  modelPresent,
  readShard,
  resolveShards,
  streamShards,
} from "../../runtime/tests/helpers/shard-files.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

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
 * （`e2e_dacvae_test.ts` の `ENCODER_TOLERANCE`）と同じ値になったのは偶然ではなく、上の
 * 「前処理の寄与が encoder の差に埋もれる」の帰結（それでも**独立に導出**する — 前処理側が
 * 悪化したときに、こちらの実測表が先に動く）。
 *
 * **式の取り違え**（目標 dB・pad の向き・フレーム幅）は latent の値域（±5.3）と同じ
 * O(0.1〜1) で出る。
 */
const LATENT_ATOL = 2e-4;

/** LUFS 正規化を通る golden ケース（`normalizeDb: -16` の 3 本 — この経路が持つ枝の全部）。 */
const CASES: readonly string[] = ["ref-default", "ref-short", "ref-odd"];

const GOLDEN_DIR = new URL("../../../outputs/series/dacvae-32dim/host/", import.meta.url);
const ENCODER_URL = new URL(
  "../../../outputs/series/dacvae-32dim/encoder/model.safetensors",
  import.meta.url,
);

const HOST_COMMAND =
  "cd tools/exporter && uv run --with descript-audiotools --with einops --with 'transformers==5.14.1' python dacvae_host.py";
const ENCODER_COMMAND =
  "cd tools/exporter && uv run --with descript-audiotools --with einops python export_dacvae.py";

const readBytes = async (url: URL): Promise<ArrayBuffer | undefined> => {
  const bytes = await Deno.readFile(url).catch(() => undefined);
  return bytes === undefined
    ? undefined
    : (bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
};

/** encoder の配布形（先頭がグラフ shard — ADR 0081。分割されていなければ 1 本）。 */
const ENCODER_SHARDS = resolveShards(ENCODER_URL);
const ENCODER_PRESENT = modelPresent(ENCODER_URL);
const caseBytes = new Map<string, ArrayBuffer>();
for (const name of CASES) {
  const bytes = await readBytes(new URL(`case.${name}.safetensors`, GOLDEN_DIR));
  if (bytes !== undefined) caseBytes.set(name, bytes);
}
const ASSETS_AVAILABLE = ENCODER_PRESENT && caseBytes.size === CASES.length;
if (!ASSETS_AVAILABLE) {
  console.warn(
    "[karume] 参照音声 → latent の e2e を SKIP する（encoder 資産とホスト golden の両方が要る）。" +
      `生成: ${ENCODER_PRESENT ? HOST_COMMAND : ENCODER_COMMAND}`,
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

Deno.test({
  name: "e2e(実GPU): 参照音声 → 正規化 → pad → codec encoder が golden の latent と一致する",
  ignore: !RUNNABLE,
  fn: async () => {
    const prepared = prepareModel(await readShard(ENCODER_SHARDS[0]));
    const gpu = await acquireGpu();
    try {
      const session = await prepared.createSession(gpu, streamShards(ENCODER_SHARDS.slice(1)), {});
      try {
        let worst = 0;
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
          const tensor = outputs[prepared.graph.outputs[0]];
          if (tensor.dtype !== "f32") throw new Error(`encoder の出力が ${tensor.dtype}`);
          const actual = tensor.data;

          assertEquals(
            actual.length,
            expected.length,
            `${name}: latent の要素数（フレーム幅 ${HOP_LENGTH} の取り違えはここに出る）`,
          );
          assertEquals(actual.length / LATENT_DIM, frames, `${name}: latent のフレーム数`);
          let maxAbs = 0;
          let absMax = 0;
          for (let i = 0; i < actual.length; i += 1) {
            maxAbs = Math.max(maxAbs, Math.abs(actual[i] - expected[i]));
            absMax = Math.max(absMax, Math.abs(expected[i]));
          }
          worst = Math.max(worst, maxAbs);
          console.log(
            `[e2e] irodori 参照音声 ${name}: T ${frames} / ${raw.length} サンプル → ` +
              `latent ${actual.length} 要素 / maxAbs ${maxAbs.toExponential(4)}` +
              `（|latent| 上端 ${absMax.toFixed(2)}）`,
          );
          assertEquals(
            maxAbs < LATENT_ATOL,
            true,
            `${name}: latent の最大絶対差 ${maxAbs} が atol ${LATENT_ATOL} を超えた` +
              "（緩めず、前処理と encoder のどちらが動いたかを分けて実測する）",
          );
        }
        console.log(`[e2e] irodori 参照音声: 最悪 maxAbs ${worst.toExponential(4)}`);
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
});
