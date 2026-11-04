// Behaviour tests of Irodori's codec host layer (tile geometry + tail trim). No GPU needed.
//
// The tile geometry is checked **structurally** ("the taken ranges cover S without gaps, with halo
// context on both sides (or a true boundary)"); bit equality of the values is checked by the
// real-GPU check (`e2e_irodori_codec_test.ts`).
// The tail trim is compared with the 3 golden cases
// (`outputs/test-assets/goldens/dacvae-32dim/host/trim.safetensors`, which records what upstream
// `find_flattening_point` returned). An environment without the golden SKIPs explicitly; a golden
// absent from a complete fetch of the pinned test assets FAILs (ADR 0139).

import { test } from "vite-plus/test";
import { assertEquals, assertRejects, assertStringIncludes, assertThrows } from "@std/assert";
import { parseSafetensors } from "@karume/runtime";
import { goldenDir, testAssetsFetched } from "@karume/test-support/test-assets";
import { ModelInputError } from "../src/errors.ts";
import {
  type CodecTile,
  decodeTiles,
  DEFAULT_CODEC_TILE_FRAMES,
  planCodecTiles,
} from "../src/irodori/codec.ts";
import { findFlatteningPoint, trimmedSampleCount } from "../src/irodori/host/trim.ts";
import { readFileIfPresent, readTextIfPresent } from "./helpers/read-if-present.ts";

// ---- Tile geometry -----------------------------------------------------------

/** The value used with the real v4-small weights (`pipelineConfig.codecHaloFrames`). */
const HALO = 8;

/**
 * Re-checks the geometry's invariants from the outside (written differently from the assertions
 * inside `planCodecTiles`). Returns the total frames of the taken ranges.
 */
const auditTiles = (tiles: readonly CodecTile[], frames: number, halo: number): number => {
  let covered = 0;
  for (const tile of tiles) {
    const takeStart = tile.start + tile.offset;
    assertEquals(takeStart, covered, "the taken ranges are not contiguous");
    const takeEnd = takeStart + tile.take;
    const leftContext = takeStart - tile.start;
    const rightContext = tile.start + tile.length - takeEnd;
    if (tile.start > 0)
      assertEquals(leftContext >= halo, true, "the left context is less than halo");
    if (tile.start + tile.length < frames) {
      assertEquals(rightContext >= halo, true, "the right context is less than halo");
    }
    covered = takeEnd;
  }
  return covered;
};

test("planCodecTiles: S at most the tile length degenerates to 1 tile (= a single decode)", () => {
  for (const frames of [1, 2, 161, DEFAULT_CODEC_TILE_FRAMES]) {
    const tiles = planCodecTiles(frames, {
      tileFrames: DEFAULT_CODEC_TILE_FRAMES,
      haloFrames: HALO,
    });
    assertEquals(tiles, [{ start: 0, length: frames, offset: 0, take: frames }], `S=${frames}`);
  }
});

test("planCodecTiles: the taken ranges cover S without gaps, with halo context on both sides", () => {
  for (const frames of [17, 33, 64, 161, 183, 300, 749, 750]) {
    for (const tileFrames of [17, 24, 64, 100, DEFAULT_CODEC_TILE_FRAMES]) {
      const tiles = planCodecTiles(frames, { tileFrames, haloFrames: HALO });
      assertEquals(auditTiles(tiles, frames, HALO), frames, `S=${frames} tile=${tileFrames}`);
    }
  }
});

test("planCodecTiles: with several tiles, every tile has the same decode length (the prepared plan keeps working)", () => {
  const tiles = planCodecTiles(750, { tileFrames: 64, haloFrames: HALO });
  assertEquals(new Set(tiles.map((tile) => tile.length)), new Set([64]));
  // Taken in steps of 64 − 2×8 = 48 frames (only the last has a remainder).
  assertEquals(tiles.length, Math.ceil(750 / 48));
  assertEquals(tiles[0].offset, 0, "the first tile has no left halo (a true boundary)");
  const last = tiles[tiles.length - 1];
  assertEquals(
    last.start + last.length,
    750,
    "the last tile's right edge snaps to the true boundary",
  );
});

test("planCodecTiles: rejects a tile length of at most 2 halos (no taken range would remain)", () => {
  assertThrows(
    () => planCodecTiles(300, { tileFrames: 16, haloFrames: HALO }),
    ModelInputError,
    "greater than two halos",
  );
});

test("planCodecTiles: rejects an S that is not a positive integer", () => {
  assertThrows(() => planCodecTiles(0, { tileFrames: 64, haloFrames: HALO }), Error, "frames 0");
  assertThrows(
    () => planCodecTiles(1.5, { tileFrames: 64, haloFrames: HALO }),
    Error,
    "frames 1.5",
  );
});

// ---- Stitching (decodeTiles) ---------------------------------------------
//
// `decodeTiles` takes `run` by injection (so the stitching itself can be measured from the outside
// — the module doc of `codec.ts`). The real-GPU check (`e2e_irodori_codec_test.ts`) needs the
// decoder asset and goldens, so on a machine without the assets this is the only check. anima's
// counterpart `decodeTiled` is pinned the same way without a GPU (`anima_tiling_test.ts`).

const TILE_LATENT_DIM = 2;
const TILE_HOP = 4;

/**
 * A latent `[frames, latentDim]` (the first column is the frame number = a ramp whose positions can
 * be read from the values).
 */
const rampLatent = (frames: number): Float32Array<ArrayBuffer> => {
  const latent = new Float32Array(frames * TILE_LATENT_DIM) as Float32Array<ArrayBuffer>;
  for (let frame = 0; frame < frames; frame += 1) latent[frame * TILE_LATENT_DIM] = frame;
  return latent;
};

/**
 * A translation-equivariant fake decoder — an output sample is determined **only by the input
 * frame's value** (`out[i*hop + s] = slice[i*latentDim]*hop + s`). Fed the ramp, a full-length
 * decode gives the identity sequence, so stitching that is off by even one sample shows in the
 * values.
 */
const equivariantDecoder = (slice: Float32Array<ArrayBuffer>, frames: number) => {
  const out = new Float32Array(frames * TILE_HOP);
  for (let index = 0; index < frames; index += 1) {
    for (let sample = 0; sample < TILE_HOP; sample += 1) {
      out[index * TILE_HOP + sample] = slice[index * TILE_LATENT_DIM] * TILE_HOP + sample;
    }
  }
  return Promise.resolve(out);
};

test("decodeTiles: places the taken ranges at the right positions of the full length (several tiles match a single decode)", async () => {
  const frames = 12;
  const tiles = planCodecTiles(frames, { tileFrames: 8, haloFrames: 2 });
  assertEquals(
    tiles.length > 1,
    true,
    "degenerating to 1 tile would skip the stitching's index arithmetic",
  );

  const waveform = await decodeTiles(
    rampLatent(frames),
    { latentDim: TILE_LATENT_DIM, hopLength: TILE_HOP, tiles },
    equivariantDecoder,
  );

  assertEquals(waveform.length, frames * TILE_HOP);
  // If any of the multiplications of offset / take / start is wrong, this identity sequence breaks.
  assertEquals(
    [...waveform],
    Array.from({ length: frames * TILE_HOP }, (_, index) => index),
  );
});

test("decodeTiles: a decoder that ignores positions breaks the match (evidence that the check is not vacuous)", async () => {
  const frames = 12;
  const tiles = planCodecTiles(frames, { tileFrames: 8, haloFrames: 2 });
  // A decoder built from **the index within the tile** rather than the input frame's value (= not
  // translation-equivariant).
  const waveform = await decodeTiles(
    rampLatent(frames),
    { latentDim: TILE_LATENT_DIM, hopLength: TILE_HOP, tiles },
    (_slice, tileFrames) =>
      Promise.resolve(Float32Array.from({ length: tileFrames * TILE_HOP }, (_, index) => index)),
  );
  assertEquals(
    [...waveform].some((value, index) => value !== index),
    true,
    "it matches even with a decoder that ignores positions, so this check pins nothing",
  );
});

test("decodeTiles: what is passed to run is a copy of one tile (not a view)", async () => {
  const frames = 12;
  const tiles = planCodecTiles(frames, { tileFrames: 8, haloFrames: 2 });
  let call = 0;
  await decodeTiles(
    rampLatent(frames),
    { latentDim: TILE_LATENT_DIM, hopLength: TILE_HOP, tiles },
    (slice, tileFrames) => {
      const tile = tiles[call];
      call += 1;
      assertEquals(
        tileFrames,
        tile.length,
        "the frames argument differs from the tile's decode length",
      );
      assertEquals(slice.length, tile.length * TILE_LATENT_DIM);
      // A copy spans its whole buffer exactly (passing a view would break the hand-off contract
      // that assumes the whole buffer is spanned).
      assertEquals(slice.buffer.byteLength, slice.length * 4);
      assertEquals(slice[0], tile.start, "the slice does not start at the tile's start frame");
      return equivariantDecoder(slice, tileFrames);
    },
  );
  assertEquals(call, tiles.length);
});

test("decodeTiles: rejects a latent whose element count is not divisible by latentDim", async () => {
  const tiles = planCodecTiles(12, { tileFrames: 8, haloFrames: 2 });
  await assertRejects(
    () =>
      decodeTiles(
        new Float32Array(5) as Float32Array<ArrayBuffer>,
        { latentDim: TILE_LATENT_DIM, hopLength: TILE_HOP, tiles },
        equivariantDecoder,
      ),
    Error,
    "is not divisible by latentDim 2",
  );
});

test("decodeTiles: rejects a decoder output of the wrong length (with the expected sample count)", async () => {
  const frames = 12;
  const tiles = planCodecTiles(frames, { tileFrames: 8, haloFrames: 2 });
  const error = await assertRejects(
    () =>
      decodeTiles(
        rampLatent(frames),
        { latentDim: TILE_LATENT_DIM, hopLength: TILE_HOP, tiles },
        (_slice, tileFrames) => Promise.resolve(new Float32Array(tileFrames * TILE_HOP - 1)),
      ),
    Error,
    "the decoder output is",
  );
  assertStringIncludes(error.message, `= ${tiles[0].length * TILE_HOP})`);
});

test("decodeTiles: with a plan degenerated to 1 tile, it is exactly a copy of the single decode", async () => {
  const frames = 12;
  const tiles = planCodecTiles(frames, { tileFrames: DEFAULT_CODEC_TILE_FRAMES, haloFrames: HALO });
  assertEquals(tiles.length, 1);
  const latent = rampLatent(frames);
  const waveform = await decodeTiles(
    latent,
    { latentDim: TILE_LATENT_DIM, hopLength: TILE_HOP, tiles },
    equivariantDecoder,
  );
  const single = await equivariantDecoder(latent, frames);
  // Bit equality (compared as f32 bit patterns — value equality lets mixed-up NaN / -0 slip
  // through).
  assertEquals(new Uint32Array(waveform.buffer), new Uint32Array(single.buffer));
});

// ---- Tail trim -----------------------------------------------------------

const LATENT_DIM = 32;
/** Upstream's `hop_length` (`pipelineConfig.hopLength`). */
const HOP = 1920;

const GOLDEN_DIR = new URL("host/", goldenDir("dacvae-32dim"));
const FETCH_HINT =
  "fetch: node tools/test-assets/fetch.ts — the assets gate accepts only a complete fetch; " +
  "--include 'goldens/dacvae-32dim/host/**' with KARUME_ALLOW_NO_ASSETS=1 runs this file alone";

/**
 * The trim section of the golden `meta.json` (the source of truth is
 * `tools/export-recipes/irodori/dacvae/host.py` — this is a reader).
 */
type TrimMeta = {
  readonly trim: Readonly<Record<string, { readonly frames: number; readonly point: number }>>;
};

const TRIM_URL = new URL("trim.safetensors", GOLDEN_DIR);
const META_URL = new URL("meta.json", GOLDEN_DIR);
const trimBytes = await readFileIfPresent(TRIM_URL);
const trimMetaText = await readTextIfPresent(META_URL);
const missingGoldens = [
  ...(trimBytes === undefined ? [TRIM_URL.pathname] : []),
  ...(trimMetaText === undefined ? [META_URL.pathname] : []),
];
// An absent golden SKIPs unless `outputs/test-assets/` holds a complete fetch of the pin (ADR 0139),
// where the dataset at the pin lacks it and the case FAILs.
const GOLDEN_AVAILABLE = missingGoldens.length === 0 || testAssetsFetched();
if (!GOLDEN_AVAILABLE) {
  console.warn(
    `[karume] The tail-trim golden comparison SKIPs: missing ${missingGoldens.join(" / ")} ` +
      `(${FETCH_HINT})`,
  );
}

test.skipIf(!GOLDEN_AVAILABLE)(
  "findFlatteningPoint: the 3 golden cases return the same position as upstream's find_flattening_point",
  () => {
    if (trimBytes === undefined || trimMetaText === undefined) {
      throw new Error(
        `${missingGoldens.join(" / ")} absent, but outputs/test-assets/ holds a complete fetch of ` +
          "the pinned revision: the test-asset dataset at the pin lacks it",
      );
    }
    const bytes = trimBytes;
    const file = parseSafetensors(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
    );
    const meta = JSON.parse(trimMetaText) as TrimMeta;
    // The 2 real z are on the "do not cut" side, the 1 synthetic one on the "returns exactly 121"
    // side (guarding against a vacuous check).
    assertEquals(Object.keys(meta.trim).sort(), ["z-full", "z-no-ref", "z-silent-tail"]);
    for (const [name, expected] of Object.entries(meta.trim)) {
      const spec = file.tensors.get(name);
      if (spec === undefined) throw new Error(`golden trim has no '${name}'`);
      const latent = new Float32Array(
        file.buffer.slice(spec.byteOffset, spec.byteOffset + spec.byteLength),
      );
      assertEquals(spec.shape, [1, expected.frames, LATENT_DIM], `shape of ${name}`);
      assertEquals(
        findFlatteningPoint(latent, expected.frames, LATENT_DIM),
        expected.point,
        `${name}: returned a position different from upstream`,
      );
    }
  },
);

test("findFlatteningPoint: an all-zero latent is flat from the start (every window equals the pad)", () => {
  assertEquals(findFlatteningPoint(new Float32Array(10 * LATENT_DIM), 10, LATENT_DIM), 0);
});

test("findFlatteningPoint: a constant value that is not near 0 is not flat (the window includes the pad)", () => {
  // Every element 1.0. The tail window's mean is diluted by the zero pad (mean 0.05 < 0.1 at i=9),
  // but the difference between real rows and the pad gives std 0.218, so it does not pass — nothing
  // is found and `frames` is returned.
  const flat = new Float32Array(10 * LATENT_DIM).fill(1);
  assertEquals(findFlatteningPoint(flat, 10, LATENT_DIM), 10);
});

test("findFlatteningPoint: returns the position from which the tail was zeroed (a reduced version of the golden synthetic case)", () => {
  const frames = 40;
  const silentFrom = 25;
  const latent = new Float32Array(frames * LATENT_DIM);
  // Alternating signs with mean 0 and variance 1 (a shape that the mean threshold does not catch,
  // only std).
  for (let index = 0; index < silentFrom * LATENT_DIM; index += 1) {
    latent[index] = index % 2 === 0 ? 1 : -1;
  }
  assertEquals(findFlatteningPoint(latent, frames, LATENT_DIM), silentFrom);
});

test("findFlatteningPoint: rejects an element count different from frames × latentDim", () => {
  assertThrows(
    () => findFlatteningPoint(new Float32Array(31), 1, LATENT_DIM),
    Error,
    "element count 31",
  );
});

test("trimmedSampleCount: takes the shorter one (upstream's max_samples)", () => {
  assertEquals(trimmedSampleCount(161 * HOP, 121, HOP), 121 * HOP, "the trim applies");
  assertEquals(trimmedSampleCount(161 * HOP, 161, HOP), 161 * HOP, "not cut");
  // When a seconds setting requests a cut shorter than the full frame length, that wins.
  assertEquals(trimmedSampleCount(48_192, 26, HOP), 48_192);
});

test("trimmedSampleCount: does not cut only when it is flat (0) from the start (never returns 0 samples)", () => {
  assertEquals(trimmedSampleCount(161 * HOP, 0, HOP), 161 * HOP);
});
