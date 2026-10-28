/**
 * Turning a relative depth map into pixels to look at. The pipeline returns raw f32 relative depth
 * (larger = nearer, no unit, no origin); normalizing and colouring are visualization decisions the
 * pipeline leaves to its user, so they live here. Pure, so `vp test` covers it.
 */

/** How the page draws the depth map. */
export type DepthColoring = "inferno" | "grayscale";

/**
 * matplotlib's `inferno` (Nathaniel Smith and Stéfan van der Walt, CC0), 256 entries as RGB8 hex,
 * rounded from `matplotlib/_cm_listed.py`. Perceptually uniform: equal steps in depth read as equal
 * steps in lightness, and the hue moves too, so a band of crowded far values still shows structure
 * where plain gray would sink into black. Dark = far, light yellow = near.
 */
const INFERNO_HEX = [
  "00000401000501010601010802010a02020c02020e03021004031204031405041706041907051b08051d09061f0a0722",
  "0b07240c08260d08290e092b10092d110a30120a32140b34150b37160b39180c3c190c3e1b0c411c0c431e0c451f0c48",
  "210c4a230c4c240c4f260c51280b53290b552b0b572d0b592f0a5b310a5c320a5e340a5f3609613809623909633b0964",
  "3d09653e0966400a67420a68440a68450a69470b6a490b6a4a0c6b4c0c6b4d0d6c4f0d6c510e6c520e6d540f6d550f6d",
  "57106e59106e5a116e5c126e5d126e5f136e61136e62146e64156e65156e67166e69166e6a176e6c186e6d186e6f196e",
  "71196e721a6e741a6e751b6e771c6d781c6d7a1d6d7c1d6d7d1e6d7f1e6c801f6c82206c84206b85216b87216b88226a",
  "8a226a8c23698d23698f24699025689225689326679526679727669827669a28659b29649d29649f2a63a02a63a22b62",
  "a32c61a52c60a62d60a82e5fa92e5eab2f5ead305dae305cb0315bb1325ab3325ab43359b63458b73557b93556ba3655",
  "bc3754bd3853bf3952c03a51c13a50c33b4fc43c4ec63d4dc73e4cc83f4bca404acb4149cc4248ce4347cf4446d04545",
  "d24644d34743d44842d54a41d74b3fd84c3ed94d3dda4e3cdb503bdd513ade5238df5337e05536e15635e25734e35933",
  "e45a31e55c30e65d2fe75e2ee8602de9612bea632aeb6429eb6628ec6726ed6925ee6a24ef6c23ef6e21f06f20f1711f",
  "f1731df2741cf3761bf37819f47918f57b17f57d15f67e14f68013f78212f78410f8850ff8870ef8890cf98b0bf98c0a",
  "f98e09fa9008fa9207fa9407fb9606fb9706fb9906fb9b06fb9d07fc9f07fca108fca309fca50afca60cfca80dfcaa0f",
  "fcac11fcae12fcb014fcb216fcb418fbb61afbb81dfbba1ffbbc21fbbe23fac026fac228fac42afac62df9c72ff9c932",
  "f9cb35f8cd37f8cf3af7d13df7d340f6d543f6d746f5d949f5db4cf4dd4ff4df53f4e156f3e35af3e55df2e661f2e865",
  "f2ea69f1ec6df1ed71f1ef75f1f179f2f27df2f482f3f586f3f68af4f88ef5f992f6fa96f8fb9af9fc9dfafda1fcffa4",
].join("");

/** The 256 × RGB lookup table of {@link INFERNO_HEX}. */
const inferno = (): Uint8Array<ArrayBuffer> => {
  const table = new Uint8Array(256 * 3);
  for (let index = 0; index < table.length; index += 1) {
    table[index] = Number.parseInt(INFERNO_HEX.slice(index * 2, index * 2 + 2), 16);
  }
  return table;
};

/** The colour of the 256 levels for each coloring (level 0 = far, 255 = near). */
const colorTable = (coloring: DepthColoring): Uint8Array<ArrayBuffer> => {
  if (coloring === "inferno") return inferno();
  const table = new Uint8Array(256 * 3);
  for (let level = 0; level < 256; level += 1) table.fill(level, level * 3, level * 3 + 3);
  return table;
};

/** The range of a depth map. */
export type DepthRange = { readonly min: number; readonly max: number };

/**
 * min and max of a depth map. The values are finite: `estimate` rejects a non-finite depth value
 * (it throws with the pixel's coordinates), so this does not look for NaN / ±Infinity.
 */
export const depthRange = (depth: Float32Array): DepthRange => {
  if (depth.length === 0) throw new RangeError("Depth map is empty");
  let min = Infinity;
  let max = -Infinity;
  for (const value of depth) {
    if (value < min) min = value;
    if (value > max) max = value;
  }
  return { min, max };
};

/**
 * Opaque RGBA8 of a depth map: min-max normalized (relative depth has no fixed scale, so maps of
 * different images are not comparable by colour), then one of 256 levels. A flat map (min = max)
 * draws as level 0 everywhere — one colour, so no gradient appears that the data does not have.
 */
export const depthToRgba = (
  depth: Float32Array,
  width: number,
  height: number,
  coloring: DepthColoring,
): Uint8ClampedArray<ArrayBuffer> => {
  if (depth.length !== width * height) {
    throw new RangeError(`Depth map has ${depth.length} values, expected ${width}×${height}`);
  }
  const { min, max } = depthRange(depth);
  const span = max - min;
  const table = colorTable(coloring);
  const rgba = new Uint8ClampedArray(depth.length * 4);
  for (let pixel = 0; pixel < depth.length; pixel += 1) {
    const level = span > 0 ? Math.round(((depth[pixel] - min) / span) * 255) : 0;
    const out = pixel * 4;
    rgba[out] = table[level * 3];
    rgba[out + 1] = table[level * 3 + 1];
    rgba[out + 2] = table[level * 3 + 2];
    rgba[out + 3] = 255;
  }
  return rgba;
};
