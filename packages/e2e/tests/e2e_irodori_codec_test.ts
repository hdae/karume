// Irodori codec decoder の**タイル同値門**（実 GPU）。
//
// タイル分割は「halo を捨てた採用区間は全長 decode と**ビット一致**する」という命題の上に
// 立っている（`src/irodori/codec.ts` のモジュール doc — 因果層ゼロ・全層が対称 pad か厳密
// `L·stride` の convT・karume の conv は出力要素ごとに固定順で縮約する gather 形）。命題が
// 成り立たなければタイル長という**性能ノブが出力を変える**ので、そこは tolerance で緩めず
// Uint32 の完全一致で縛る。
//
// The input is a real z (`z` of the golden
// `outputs/test-assets/goldens/irodori-v4-small/pipeline/case.full.safetensors`, S = 161); the
// decoder is `codec_decoder` at f32 of the mirror `models/karume-irodori-v4-small/`.
// 合成乱数ではなく実運用の値域を使うのは、Snake の `sin(αx)` が入力の値域に強く効くため
// （`e2e_dacvae_test.ts`〈0.15 の選別で削除〉 の tolerance 導出と同じ理由）。
//
// 資産が欠けた環境と GPU 無し環境は生成コマンド付きで**明示 SKIP**する（ADR 0005）。

import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { acquireGpu, parseSafetensors, type Tensor } from "@karume/runtime";
import { goldenDir, testAssetsFetched } from "@karume/test-support/test-assets";
import { decodeTiles, planCodecTiles } from "../../models/src/irodori/codec.ts";
import {
  mirrorPresent,
  mirrorRoot,
  openMirrorComponent,
} from "../../models/tests/helpers/mirror.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { readFileIfPresent } from "../../models/tests/helpers/read-if-present.ts";

/** 実重み v4-small の運用値（`pipelineConfig` が運ぶ数と同じ）。 */
const LATENT_DIM = 32;
const HOP_LENGTH = 1920;
const HALO_FRAMES = 8;

/**
 * 門で使う小さいタイル長。既定（182）より**十分小さく**取るのは、S = 161 で複数枚に割れる
 * 必要があるから（既定のままだと 1 枚に縮退して命題を一度も試さない）。
 */
const GATE_TILE_FRAMES = 64;

const MIRROR = mirrorRoot("karume-irodori-v4-small");
/** The decoder graph (the manifest's weights key) at the storage the golden z was decoded with. */
const DECODER_GRAPH = "codec_decoder";
const DECODER_CHOICE = { model: "v4-small", quant: "f32" } as const;
const LATENT_URL = new URL("pipeline/case.full.safetensors", goldenDir("irodori-v4-small"));

const FETCH_HINT =
  "fetch: node tools/test-assets/fetch.ts — the assets gate accepts only a complete fetch; " +
  "--include 'goldens/irodori-v4-small/pipeline/**' with KARUME_ALLOW_NO_ASSETS=1 runs this file alone";

/**
 * Whether `outputs/test-assets/` holds a complete fetch of the pin (ADR 0139). Then an absent z
 * golden means the dataset at the pin lacks it, so the case FAILs instead of SKIPping.
 */
const FETCHED = testAssetsFetched();

const readFile = async (url: URL): Promise<ArrayBuffer | undefined> => {
  const bytes = await readFileIfPresent(url);
  return bytes === undefined
    ? undefined
    : bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};

const DECODER_PRESENT = mirrorPresent(MIRROR);
const latentBytes = await readFile(LATENT_URL);
// An absent mirror SKIPs. An absent z golden SKIPs unless the test assets are a complete fetch of
// the pin, where `readLatent` FAILs.
const ASSETS_AVAILABLE = DECODER_PRESENT && (latentBytes !== undefined || FETCHED);
if (!ASSETS_AVAILABLE) {
  const missing = [
    ...(DECODER_PRESENT ? [] : [`the mirror ${MIRROR.pathname} (its codec_decoder)`]),
    ...(latentBytes === undefined ? [`${LATENT_URL.pathname} (${FETCH_HINT})`] : []),
  ];
  console.warn(`[karume] The codec tile-equivalence check SKIPs: missing ${missing.join(" / ")}`);
}

const RUNNABLE = GPU_AVAILABLE && ASSETS_AVAILABLE;

/** golden の `z`（`[1,S,32]`）を読む。 */
const readLatent = (): Float32Array<ArrayBuffer> => {
  if (latentBytes === undefined) {
    throw new Error(
      `${LATENT_URL.pathname} absent, but outputs/test-assets/ holds a complete fetch of the ` +
        "pinned revision: the test-asset dataset at the pin lacks it",
    );
  }
  const file = parseSafetensors(latentBytes);
  const spec = file.tensors.get("z");
  if (spec === undefined) throw new Error("full-loop golden に 'z' が無い");
  return new Float32Array(
    file.buffer.slice(spec.byteOffset, spec.byteOffset + spec.byteLength),
  ) as Float32Array<ArrayBuffer>;
};

test.skipIf(!RUNNABLE)(
  "e2e(実GPU): codec のタイル decode は単発 decode と全サンプルでビット一致する",
  async () => {
    const latent = readLatent();
    const frames = latent.length / LATENT_DIM;
    const decoder = await openMirrorComponent(MIRROR, DECODER_GRAPH, DECODER_CHOICE);
    const gpu = await acquireGpu();
    try {
      const session = await decoder.createSession(gpu, {});
      try {
        const run = async (
          slice: Float32Array<ArrayBuffer>,
          tileFrames: number,
        ): Promise<Float32Array<ArrayBuffer>> => {
          const input: Tensor = {
            dtype: "f32",
            shape: [1, tileFrames, LATENT_DIM],
            data: slice,
          };
          const outputs = await session.run({ latent: input });
          const tensor = outputs[decoder.graph.outputs[0]];
          if (tensor.dtype !== "f32") throw new Error(`decoder の出力が ${tensor.dtype}`);
          return tensor.data;
        };
        const geometry = { latentDim: LATENT_DIM, hopLength: HOP_LENGTH };

        const single = await decodeTiles(
          latent,
          { ...geometry, tiles: planCodecTiles(frames, { tileFrames: frames, haloFrames: 0 }) },
          run,
        );
        const tiles = planCodecTiles(frames, {
          tileFrames: GATE_TILE_FRAMES,
          haloFrames: HALO_FRAMES,
        });
        const tiled = await decodeTiles(latent, { ...geometry, tiles }, run);

        console.log(
          `[e2e] irodori codec: S ${frames} / 単発 1 枚 vs タイル ${tiles.length} 枚` +
            `（decode 長 ${GATE_TILE_FRAMES} / halo ${HALO_FRAMES}）/ ${single.length} サンプル`,
        );
        assertEquals(tiles.length > 1, true, "タイルが 1 枚に縮退している（門が何も試していない）");
        assertEquals(single.length, frames * HOP_LENGTH, "単発 decode の出力長");
        // Uint32 で比べる — f32 の等値比較だと −0 と +0 の差を見逃す。
        assertEquals(
          new Uint32Array(tiled.buffer),
          new Uint32Array(single.buffer),
          "タイル decode が単発 decode と 1 ビットでも違う（halo が足りないか、貼り付けの幾何が" +
            "ずれている — tolerance で緩めず幾何を疑う）",
        );
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);
