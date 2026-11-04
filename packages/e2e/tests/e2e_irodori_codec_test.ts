// The **tile-equivalence check** of the Irodori codec decoder (real GPU).
//
// Tiling rests on the proposition "the kept span with the halo discarded is **bit-identical** to a
// full-length decode" (the module doc of `src/irodori/codec.ts` — zero causal layers, every layer
// either symmetric pad or a convT of exactly `L·stride`, and karume's conv is a gather form that
// reduces in a fixed order per output element). If the proposition failed, the tile length, a
// **performance knob, would change the output**, so it is pinned by an exact Uint32 match rather
// than loosened with a tolerance.
//
// The input is a real z (`z` of the golden
// `outputs/test-assets/goldens/irodori-v4-small/pipeline/case.full.safetensors`, S = 161); the
// decoder is `codec_decoder` at f32 of the mirror `models/karume-irodori-v4-small/`.
// The real operating value range is used instead of synthetic random numbers because Snake's
// `sin(αx)` depends strongly on the input's value range (the same reason as the tolerance
// derivation of `e2e_dacvae_test.ts` (removed in the 0.15 triage)).
//
// An environment missing the assets or without a GPU **SKIPs explicitly** with the generation
// command (ADR 0005).

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

/** The operating values of real-weight v4-small (the same numbers `pipelineConfig` carries). */
const LATENT_DIM = 32;
const HOP_LENGTH = 1920;
const HALO_FRAMES = 8;

/**
 * The small tile length the check uses. It is taken **well below** the default (182) because S =
 * 161 has to split into several tiles (with the default it degenerates to one tile and never tests
 * the proposition).
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

/** Reads the golden's `z` (`[1,S,32]`). */
const readLatent = (): Float32Array<ArrayBuffer> => {
  if (latentBytes === undefined) {
    throw new Error(
      `${LATENT_URL.pathname} absent, but outputs/test-assets/ holds a complete fetch of the ` +
        "pinned revision: the test-asset dataset at the pin lacks it",
    );
  }
  const file = parseSafetensors(latentBytes);
  const spec = file.tensors.get("z");
  if (spec === undefined) throw new Error("the full-loop golden has no 'z'");
  return new Float32Array(
    file.buffer.slice(spec.byteOffset, spec.byteOffset + spec.byteLength),
  ) as Float32Array<ArrayBuffer>;
};

test.skipIf(!RUNNABLE)(
  "e2e (real GPU): the codec's tiled decode is bit-identical to a single decode on every sample",
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
          if (tensor.dtype !== "f32") throw new Error(`the decoder output is ${tensor.dtype}`);
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
          `[e2e] irodori codec: S ${frames} / single 1 tile vs tiled ${tiles.length} tiles` +
            ` (decode length ${GATE_TILE_FRAMES} / halo ${HALO_FRAMES}) / ${single.length} samples`,
        );
        assertEquals(
          tiles.length > 1,
          true,
          "the tiling degenerated to one tile (the check tests nothing)",
        );
        assertEquals(single.length, frames * HOP_LENGTH, "output length of the single decode");
        // Compare as Uint32 — an f32 equality comparison misses the difference between −0 and +0.
        assertEquals(
          new Uint32Array(tiled.buffer),
          new Uint32Array(single.buffer),
          "the tiled decode differs from the single decode by even one bit (the halo is too short or " +
            "the paste geometry is off — suspect the geometry instead of loosening with a tolerance)",
        );
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);
