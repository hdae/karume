// Irodori codec decoder の**タイル同値門**（実 GPU）。
//
// タイル分割は「halo を捨てた採用区間は全長 decode と**ビット一致**する」という命題の上に
// 立っている（`src/irodori/codec.ts` のモジュール doc — 因果層ゼロ・全層が対称 pad か厳密
// `L·stride` の convT・karume の conv は出力要素ごとに固定順で縮約する gather 形）。命題が
// 成り立たなければタイル長という**性能ノブが出力を変える**ので、そこは tolerance で緩めず
// Uint32 の完全一致で縛る。
//
// 入力は実 z（`outputs/series/irodori-v4-small/pipeline/case.full.safetensors` の `z`・S = 161）。
// 合成乱数ではなく実運用の値域を使うのは、Snake の `sin(αx)` が入力の値域に強く効くため
// （`packages/runtime/tests/e2e_dacvae_test.ts` の tolerance 導出と同じ理由）。
//
// 資産が欠けた環境と GPU 無し環境は生成コマンド付きで**明示 SKIP**する（ADR 0005）。

import { assertEquals } from "@std/assert";
import {
  acquireGpu,
  createSession,
  openModel,
  parseSafetensors,
  type Tensor,
} from "@karume/runtime";
import { decodeTiles, planCodecTiles } from "../src/irodori/codec.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/** 実重み v4-small の運用値（`pipelineConfig` が運ぶ数と同じ）。 */
const LATENT_DIM = 32;
const HOP_LENGTH = 1920;
const HALO_FRAMES = 8;

/**
 * 門で使う小さいタイル長。既定（182）より**十分小さく**取るのは、S = 161 で複数枚に割れる
 * 必要があるから（既定のままだと 1 枚に縮退して命題を一度も試さない）。
 */
const GATE_TILE_FRAMES = 64;

const DECODER_URL = new URL(
  "../../../outputs/series/dacvae-32dim/decoder/model.safetensors",
  import.meta.url,
);
const LATENT_URL = new URL(
  "../../../outputs/series/irodori-v4-small/pipeline/case.full.safetensors",
  import.meta.url,
);

const DECODER_COMMAND =
  "cd tools/exporter && uv run --with descript-audiotools --with einops python export_dacvae.py";
const LATENT_COMMAND =
  "cd tools/exporter && uv run --with 'transformers==5.14.1' python irodori_pipeline.py";

const readFile = async (url: URL): Promise<ArrayBuffer | undefined> => {
  const bytes = await Deno.readFile(url).catch(() => undefined);
  return bytes === undefined
    ? undefined
    : bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};

const decoderBytes = await readFile(DECODER_URL);
const latentBytes = await readFile(LATENT_URL);
const ASSETS_AVAILABLE = decoderBytes !== undefined && latentBytes !== undefined;
if (!ASSETS_AVAILABLE) {
  console.warn(
    "[karume] codec タイル同値門を SKIP する（decoder 資産と full-loop golden の両方が要る）。" +
      `生成: ${decoderBytes === undefined ? DECODER_COMMAND : LATENT_COMMAND}`,
  );
}

const RUNNABLE = GPU_AVAILABLE && ASSETS_AVAILABLE;

/** golden の `z`（`[1,S,32]`）を読む。 */
const readLatent = (): Float32Array<ArrayBuffer> => {
  const file = parseSafetensors(latentBytes as ArrayBuffer);
  const spec = file.tensors.get("z");
  if (spec === undefined) throw new Error("full-loop golden に 'z' が無い");
  return new Float32Array(
    file.buffer.slice(spec.byteOffset, spec.byteOffset + spec.byteLength),
  ) as Float32Array<ArrayBuffer>;
};

Deno.test({
  name: "e2e(実GPU): codec のタイル decode は単発 decode と全サンプルでビット一致する",
  ignore: !RUNNABLE,
  fn: async () => {
    const latent = readLatent();
    const frames = latent.length / LATENT_DIM;
    const model = openModel(decoderBytes as ArrayBuffer);
    const gpu = await acquireGpu();
    try {
      const session = await createSession(gpu, model, {});
      try {
        const run = async (
          slice: Float32Array<ArrayBuffer>,
          tileFrames: number,
        ): Promise<Float32Array> => {
          const input: Tensor = {
            dtype: "f32",
            shape: [1, tileFrames, LATENT_DIM],
            data: slice,
          };
          const outputs = await session.run({ latent: input });
          const tensor = outputs[model.graph.outputs[0]];
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
});
