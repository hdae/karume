/**
 * Reference comparison of Wan2.2 TI2V-5B's VAE **encoder chunk 0** (one I2V conditioning image) (real GPU — the
 * acceptance of ADR 0121 stage 9a).
 *
 * The encoder's three graphs of the distribution mirror `models/karume-wan2.2/` (f16 — written by the recipe
 * `tools/export-recipes/wan/export_vae_encoder.py`; opened through the manifest — ADR 0139 decision 1; the goldens are
 * in the test-asset root — decision 2) run on the input the TS preprocessing (`[-1, 1]` → patchify) makes from the
 * golden's RGB8; the output mu goes through the TS normalization (`normalizeWanLatents`) and is compared with the
 * normalized mu of upstream's untiled encode (diffusers, CPU f32, weights rounded to f16).
 *
 * - pre `[12,1,8h,8w]` → `[640,1,h,w]` (symbols h, w) / attn `[640,S]` → `[640,S]` (symbol S = h·w) / post
 *   `[640,1,h,w]` → mu `[48,1,h,w]`. h, w are the latent height and width. A dimension expression of the IR is linear
 *   in one symbol, so the graphs are split around the mid attention, which makes h·w one axis (ADR 0121 decision 11).
 * - It also checks that the golden's encoder input (upstream's `patchify`) and the output of the TS preprocessing are
 *   bit-identical (what goes to the GPU is the output of the TS preprocessing — the same input as the product path).
 *
 * ## Hand-off (resident tensors, without a host round trip)
 *
 * The three Sessions sit on the same GpuContext, and pre → attn → post are `enqueue`d in one batch. The outputs of pre
 * / attn are copied to resident tensors with `copyOutputs`, and the next graph's input is bound to that resident as is.
 * `[640,1,h,w]` and `[640,S]` are reinterpretations of the same bytes, so no re-copy is needed (a resident carries only
 * its size — the only check is that the byte count matches the declared shape). A resident input is not a source of
 * symbol bindings, so attn gets `S` and post gets `h` / `w` through `bindings`. mu is copied to a resident too and
 * read once with `finishAndRead` (one fence: the batch's settlement).
 *
 * ## Cases and band
 *
 * - The goldens are 3 test images × 3 sizes (1280×704, 704×1280 (a portrait size given explicitly for a landscape
 *   image), 256×160 (the downscale path outside the accepted set)). The band-setting images are boxing-cats /
 *   cat-dog-baking, the acceptance image is ferret (the recipe's `BAND_IMAGES` / `ACCEPT_IMAGES` — matched against
 *   the golden metadata `role`).
 * - **All 9 run in the default lane** (measured on 2026-10-09: comparing all 9 takes about 3 s — only about 1 s more
 *   than the two 1280×704 ones). The same Session runs 1280×704 → 704×1280 (same S, h and w swapped) → 256×160 and is
 *   compared with a golden per binding, so the default lane checks that **the assets do not depend on the size** (same
 *   container, same Session, only the bindings differ) and that a plan does not carry the previous binding's shapes
 *   over when the bindings change (every node, including pre's reshape (space-to-depth by a symbolic division),
 *   slice / cat and the sum reduction, and the f16 weights). The encoder sizes specific to I2V in portrait are
 *   covered here.
 * - The band is an absolute value: the worst normalized maxAbs over all 6 band-setting cases (2 images × 3 sizes) × 5,
 *   rounded up to 2 significant digits (`rtol = 0` — derived the same way as the e2e of the decoder's chunk
 *   sequence). The acceptance check is that the acceptance cases fall inside the band. `undefined` = not derived yet:
 *   each case records its maxAbs and stops red, and the last step prints a candidate.
 *
 * ## Fault injection (the check is that the acceptance 1280×704 falls outside the band)
 *
 * - Swapping the height and width sub-indices of patchify (host — an input with the 1st and 2nd of each group of 4
 *   channels swapped).
 * - Mixing up std and 1/std (host — the statistics' std inverted and passed through the same normalization = mu ·
 *   std).
 * - Swapping h and w in post's bindings (hand-off — the resident's byte count is the same, so it passes the size check,
 *   and post reads the bytes of `[640,1,h,w]` as `[640,1,w,h]`).
 * - The 48 logvar channels instead of mu (post's output slice moved to the second half — a replaced declaration).
 * - Mixing up the parity of AvgDown3D (in the two blocks with temporal factor 2, the shortcut's mean is added to the
 *   even channels instead of the odd ones — replacing the two slices of pre's declaration and the order of the cat
 *   inputs).
 *
 * NOTE: the std / 1/std mix-up does not run the GPU (it only renormalizes on the host the mu read in the comparison
 * step). All it shows is "the band is narrower than a normalization error"; it does not prove the detection power of
 * the GPU check (the host test `wan_ti2v_i2v_preprocess_test.ts` pins the formula itself). The binding swap is the one
 * aimed at the GPU hand-off path.
 *
 * The last two are nodes inside a graph, so they cannot be injected from the graph's inputs and outputs (the shortcut
 * has no weights, and the mu slice is the graph's output itself). The assets are not rewritten: the Session is built
 * from a container whose **declaration alone** is replaced in memory (the weight blocks are read from the original
 * container — {@link withRewrittenNodes}). An injection Session is disposed right after use (the second pre holds
 * about 0.9 GiB in weights and the backing of the 1280×704 plan).
 *
 * ## The mid attention (D = 640, S = h·w)
 *
 * Right after each case, the check is that the most recent run's plan of attn holds one or more of each of the three
 * stages of fused attention (`attention_qk` / `attention_stats` / `attention_pv`) (the same check as the decoder's D
 * = 1024 — the band checks the values, but it cannot tell whether the fused path ran). The row-statistics variant
 * changes with S (the key's `rc` — 14 at S = 3,520, 1 at S = 160), so it is checked per size.
 *
 * ## Records (not checks — copied into the addendum of ADR 0121 stage 9a)
 *
 * The build time of the three Sessions, the encode wall time per case (the first case of each size includes deriving
 * the plan and creating the pipelines), the Sessions' VRAM breakdown (weights, plan backing), the resident size, the
 * mu difference, and the attention pipeline keys.
 *
 * NOTE: the number of backings kept when the size changes is decided by the budgeted LRU of ADR 0095 decision 1
 * (`planBackingBudgetBytes` — 256 MiB by default). It does not depend on whether the input is a resident or a host
 * tensor. Measured on 2026-10-09: pre's single 1280×704 backing (801.5 MiB) exceeds the budget alone, so it is
 * replaced at 256×160 (1 kept). attn (81.7 + 2.4 MiB) and post (17.2 MiB × 2 + 0.8 MiB) fit in the budget, so one is
 * kept per size (2 / 3). What a Session keeps never exceeds `max(budget, the largest single backing)`.
 *
 * Environments without any of the assets (3 containers and 9 goldens) and environments without a GPU SKIP explicitly.
 * An environment with only some of them FAILs (the first test, which needs no GPU).
 */

import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import {
  type GpuContext,
  parseSafetensors,
  prepareContainer,
  type PreparedModel,
  type ResidentTensor,
  type SafetensorsFile,
  type Session,
  type SessionDiagnostics,
} from "@karume/runtime";
import { WAN22_TI2V_GENERATION } from "../../models/src/wan/descriptor.ts";
import { patchifyWanImage, wanI2vPixels } from "../../models/src/wan/i2v-preprocess.ts";
import { normalizeWanLatents } from "../../models/src/wan/latents.ts";
import { disposeSteps } from "../../models/src/session/dispose-steps.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { settleReleases } from "../../models/tests/helpers/settle-releases.ts";
import { filePresent, firstBitMismatch } from "../../models/tests/helpers/wan-ti2v-dit.ts";
import { readBuffer, viewOf } from "../../models/tests/helpers/safetensors.ts";
import {
  WAN_I2V_ENCODER_BAND,
  WAN_I2V_GOLDEN_CASES,
  WAN_I2V_GOLDEN_KEYS,
  WAN_I2V_GOLDEN_METADATA,
  WAN_I2V_IMAGE_SHA256,
  type WanI2vGoldenCase,
  wanI2vGoldenName,
  wanI2vGoldenUrl,
} from "../../models/tests/helpers/wan-i2v-image.ts";
import { WAN_TI2V_VAE_GOLDEN_DIR as GOLDEN_DIR } from "../../models/tests/helpers/wan-ti2v-vae.ts";
import { mirrorPresent } from "../../models/tests/helpers/mirror.ts";
import { TEST_ASSETS, testAssetsFetched } from "@karume/test-support/test-assets";
import { TEST_ASSETS_FETCH as FETCH } from "../../models/tests/helpers/wan-ti2v-dit.ts";
import {
  openWanTi2vComponent,
  WAN_TI2V_ASSEMBLE_COMMAND as ASSEMBLE,
  WAN_TI2V_DIST_ROOT as DIST_ROOT,
} from "../../models/tests/helpers/wan-ti2v-pipeline.ts";
import { allclose, type Tolerance } from "../../runtime/src/reference/allclose.ts";
import type { BoundContainer } from "../../runtime/src/format/container/bind.ts";
import type { IrNode } from "../../runtime/src/format/ir.ts";
import { verifyRunningGpu } from "../../runtime/tests/helpers/environment.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";
import { UPSTREAM_PARITY, UPSTREAM_PARITY_OPT_IN } from "./helpers/upstream-parity.ts";

/**
 * The band (absolute) of the normalized latent. The value, its derivation and the fault-injection record are in
 * {@link WAN_I2V_ENCODER_BAND} (the end-to-end I2V e2e uses the same value for its conditioning-latent check, so it
 * lives in one place in the helper). `undefined` = not derived yet (module doc "Cases and band" — a run that
 * re-derives it sets this to `undefined`).
 */
const BAND: number | undefined = WAN_I2V_ENCODER_BAND;
/**
 * The band for the verdict and the record (a run without a derived band records an infinite band and fails the
 * verdict).
 */
const TOLERANCE: Tolerance = { atol: BAND ?? Number.POSITIVE_INFINITY, rtol: 0 };

/** Component names: the manifest's weights keys, which are also the containers' graph names (container-v1 §2.1). */
const PRE = "vae_encoder_pre";
const ATTENTION = "vae_encoder_attn";
const POST = "vae_encoder_post";
const PARTS = [PRE, ATTENTION, POST] as const;

/** The graph input spellings (the forward argument names of the recipe's modules). */
const INPUT = { pre: "image", attn: "tokens", post: "hidden" } as const;

/** The mid block's channel count (`base_dim 160 × dim_mult[-1] 4`). */
const HIDDEN_CHANNELS = 640;
/** The latent channel count (upstream config's `z_dim` — the number of mu channels). */
const LATENT_CHANNELS = 48;
/** The patchify factor, and the image edge per latent element (the encoder's 3 down stages 8 × patchify 2). */
const PATCH = WAN22_TI2V_GENERATION.vaePatchSize;
const SPATIAL_COMPRESSION = 8 * PATCH;
/** Bytes per element (every semantic dtype is 4 bytes — ADR 0009). */
const BYTES_PER_ELEMENT = 4;

/** The band-setting images (the recipe's `BAND_IMAGES`; matched against the golden metadata `role`). */
const BAND_IMAGES: readonly string[] = ["boxing-cats", "cat-dog-baking"];

/** The fault-injection input (the acceptance 1280×704). */
const FAULT_SIZE = { width: 1280, height: 704 } as const;
const FAULT_IMAGE = "ferret";

/** The three recorded stages of fused attention (ADR 0023 — pipeline-key prefixes of ①QK, ② row statistics, ③PV). */
const ATTENTION_KERNELS = ["attention_qk", "attention_stats", "attention_pv"] as const;

type Role = "band" | "accept";
type EncoderCase = WanI2vGoldenCase & {
  readonly name: string;
  readonly role: Role;
};

/**
 * The encoder goldens (the 9 with fit = crop — in the golden table's order = 3 images per size. They also run in this
 * order = bindings 1280×704 → 704×1280 → 256×160).
 */
const CASES: readonly EncoderCase[] = WAN_I2V_GOLDEN_CASES.filter(({ fit }) => fit === "crop").map(
  (testCase) => ({
    ...testCase,
    name: wanI2vGoldenName(testCase),
    role: BAND_IMAGES.includes(testCase.image) ? "band" : "accept",
  }),
);
const FAULT_CASE = CASES.find(
  ({ image, width, height }) =>
    image === FAULT_IMAGE && width === FAULT_SIZE.width && height === FAULT_SIZE.height,
);

const MIRROR_PRESENT = mirrorPresent(DIST_ROOT);
const ASSETS = CASES.map((testCase) => {
  const url = wanI2vGoldenUrl(testCase);
  return { path: url.pathname, present: filePresent(url) };
});
const ANY_ASSETS = ASSETS.some(({ present }) => present);
/** The GPU checks read the mirror's three graphs and the nine goldens. */
const ALL_ASSETS = MIRROR_PRESENT && ASSETS.every(({ present }) => present);
/**
 * A complete fetch of the pinned test-asset dataset makes a missing golden a FAIL (the completeness test below): the
 * dataset at the pin lacks a file the cases read. Without one, the cases SKIP (the assets gate is red then).
 */
const FETCHED = testAssetsFetched();
const MISSING_GOLDENS = FETCHED
  ? `the test-asset dataset at revision ${TEST_ASSETS.revision} lacks them`
  : `fetch: ${FETCH}`;
if (!MIRROR_PRESENT) {
  console.warn(
    `[karume] the mirror ${DIST_ROOT.pathname} is absent (or not karume/5, or a sibling it references is ` +
      `absent), so the VAE encoder checks of Wan2.2 TI2V are SKIPPED. Assemble it: ${ASSEMBLE}`,
  );
}
if (!ANY_ASSETS) {
  console.warn(
    `[karume] no VAE encoder golden under ${GOLDEN_DIR.pathname}, so the encoder checks are SKIPPED ` +
      `(${MISSING_GOLDENS})`,
  );
}

const results = openResults("wan-ti2v-vae-encoder");

// ---- golden ---------------------------------------------------------------------------------

type EncoderGolden = {
  readonly rgb8: Uint8Array<ArrayBuffer>;
  readonly encoderInput: Float32Array<ArrayBuffer>;
  readonly mu: Float32Array<ArrayBuffer>;
  readonly latent: Float32Array<ArrayBuffer>;
};

const metadataOf = (file: SafetensorsFile, key: string, where: string): string => {
  const value = file.metadata.get(key);
  if (value === undefined) throw new Error(`${where}: __metadata__.${key} is missing`);
  return value;
};

/**
 * Copies a tensor, checking its dtype and shape (with U8 placed before it, the F32 start is not guaranteed to fall on a
 * 4-byte boundary).
 */
const tensorBytes = (
  file: SafetensorsFile,
  key: string,
  dtype: "U8" | "F32",
  shape: readonly number[],
  where: string,
): ArrayBuffer => {
  const view = viewOf(file, key, where);
  assertEquals(view.dtype, dtype, `${where}: ${key} dtype`);
  assertEquals([...view.shape], [...shape], `${where}: ${key} shape`);
  return file.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength);
};

const readGolden = async (testCase: EncoderCase): Promise<EncoderGolden> => {
  const where = testCase.name;
  const file = parseSafetensors(await readBuffer(wanI2vGoldenUrl(testCase)));
  const meta = WAN_I2V_GOLDEN_METADATA;
  const { width, height } = testCase;
  assertEquals(metadataOf(file, meta.image, where), testCase.image, `${where}: image`);
  assertEquals(
    metadataOf(file, meta.imageSha256, where),
    WAN_I2V_IMAGE_SHA256[testCase.image],
    `${where}: image sha256 differs from the table`,
  );
  assertEquals(metadataOf(file, meta.fit, where), "crop", `${where}: fit`);
  assertEquals(
    [metadataOf(file, meta.width, where), metadataOf(file, meta.height, where)],
    [String(width), String(height)],
    `${where}: output size`,
  );
  assertEquals(metadataOf(file, meta.role, where), testCase.role, `${where}: role (band / accept)`);
  const latentShape = [
    LATENT_CHANNELS,
    1,
    height / SPATIAL_COMPRESSION,
    width / SPATIAL_COMPRESSION,
  ];
  const keys = WAN_I2V_GOLDEN_KEYS;
  return {
    rgb8: new Uint8Array(tensorBytes(file, keys.rgb8, "U8", [height, width, 3], where)),
    encoderInput: new Float32Array(
      tensorBytes(
        file,
        keys.encoderInput,
        "F32",
        [3 * PATCH * PATCH, 1, height / PATCH, width / PATCH],
        where,
      ),
    ),
    mu: new Float32Array(tensorBytes(file, keys.mu, "F32", latentShape, where)),
    latent: new Float32Array(tensorBytes(file, keys.latent, "F32", latentShape, where)),
  };
};

/** The TS preprocessing (the golden's RGB8 → `[-1, 1]` → patchify). */
const encoderInputOf = (testCase: EncoderCase, rgb8: Uint8Array<ArrayBuffer>) =>
  patchifyWanImage(
    wanI2vPixels({ data: rgb8, width: testCase.width, height: testCase.height }),
    [3, testCase.height, testCase.width],
    PATCH,
  );

// ---- containers and replaced declarations ----------------------------------------------------

type EncoderParts<T> = { readonly pre: T; readonly attn: T; readonly post: T };

const openParts = async (): Promise<EncoderParts<BoundContainer>> => {
  const [pre, attn, post] = await Promise.all(
    PARTS.map((part) => openWanTi2vComponent(part, "f16")),
  );
  return { pre, attn, post };
};

const prepareParts = (opened: EncoderParts<BoundContainer>): EncoderParts<PreparedModel> => ({
  pre: prepareContainer(opened.pre, PRE),
  attn: prepareContainer(opened.attn, ATTENTION),
  post: prepareContainer(opened.post, POST),
});

/** The name of the graph's output (read from the declaration — the export decides the spelling). */
const outputOf = (model: PreparedModel, part: string): string => {
  const [output, ...rest] = model.graph.outputs;
  if (output === undefined || rest.length > 0) {
    throw new Error(`${part}: not exactly one output (${model.graph.outputs.join(" / ")})`);
  }
  return output;
};

/**
 * A container with only the declaration's nodes replaced (for fault injection). The weight blocks and the supply plan
 * stay those of the original container — the assets are not rewritten.
 */
const withRewrittenNodes = (
  opened: BoundContainer,
  graphName: string,
  rewrite: (nodes: readonly IrNode[]) => readonly IrNode[],
): BoundContainer => {
  const bound = opened.graphs[graphName];
  if (bound === undefined) throw new Error(`the container has no graph '${graphName}'`);
  return {
    graphs: {
      [graphName]: {
        declaration: { ...bound.declaration, nodes: rewrite(bound.declaration.nodes) },
        supplies: bound.supplies,
      },
    },
    readBlock: (id) => opened.readBlock(id),
  };
};

/** Value name → the index of the node that produces it. */
const producers = (nodes: readonly IrNode[]): ReadonlyMap<string, number> =>
  new Map(nodes.flatMap((node, index) => node.outs.map((name) => [name, index] as const)));

const sliceRange = (node: IrNode | undefined): string =>
  node === undefined || node.op !== "slice"
    ? `${node?.op ?? "none"}`
    : `slice dim ${node.attrs["dim"]} [${node.attrs["start"]}, ${node.attrs["end"]})`;

/**
 * Fault: moves post's output slice to the second half of quant_conv's output (the 48 logvar channels). Throws when the
 * output slice (dim 0, [0, 48)) is missing from the declaration (so the injection is never vacuous).
 */
const takeLogvarHalf =
  (output: string) =>
  (nodes: readonly IrNode[]): readonly IrNode[] => {
    const index = producers(nodes).get(output);
    const node = index === undefined ? undefined : nodes[index];
    if (
      index === undefined ||
      node?.op !== "slice" ||
      node.attrs["dim"] !== 0 ||
      node.attrs["start"] !== 0 ||
      node.attrs["end"] !== LATENT_CHANNELS
    ) {
      throw new Error(`post's output '${output}' is not the mu slice (${sliceRange(node)})`);
    }
    const logvar = {
      ...node,
      attrs: { ...node.attrs, start: LATENT_CHANNELS, end: 2 * LATENT_CHANNELS },
    };
    return nodes.map((current, at) => (at === index ? logvar : current));
  };

/**
 * Fault: in the blocks of AvgDown3D with temporal factor 2 (two — down_blocks.1 / .2), adds the shortcut's spatial
 * 2×2 mean to the even channels (`2c`) instead of the odd ones (`2c+1`).
 *
 * The recipe's closed form is `pairs = residual.reshape(C, 2, h, w)` → `cat([pairs[:, 0:1], pairs[:, 1:2] + mean], 1)`.
 * Swapping the ranges of the two slices (0:1 ↔ 1:2) and the order of the cat inputs gives
 * `[pairs[:, 0:1] + mean, pairs[:, 1:2]]`. Throws when the shape is not the expected one (so the injection is never
 * vacuous).
 */
const swapShortcutParity = (nodes: readonly IrNode[]): readonly IrNode[] => {
  const producedBy = producers(nodes);
  const nodeOf = (name: string): { readonly index: number; readonly node: IrNode } | undefined => {
    const index = producedBy.get(name);
    return index === undefined ? undefined : { index, node: nodes[index] };
  };
  const isSlice = (node: IrNode, start: number, end: number): boolean =>
    node.op === "slice" &&
    node.attrs["dim"] === 1 &&
    node.attrs["start"] === start &&
    node.attrs["end"] === end;
  const replaced = new Map<number, IrNode>();
  const cats = nodes.flatMap((node, index) => (node.op === "cat" ? [index] : []));
  assertEquals(cats.length, 2, "pre's cats are not the two shortcuts with temporal factor 2");
  for (const catIndex of cats) {
    const cat = nodes[catIndex];
    const [zeroName, sumName] = cat.ins;
    const zero = nodeOf(zeroName ?? "");
    const sum = nodeOf(sumName ?? "");
    if (
      cat.attrs["dim"] !== 1 ||
      cat.ins.length !== 2 ||
      zero === undefined ||
      sum === undefined ||
      !isSlice(zero.node, 0, 1) ||
      sum.node.op !== "add"
    ) {
      throw new Error(`cat '${cat.outs[0]}' is not the shortcut's closed form`);
    }
    const data = sum.node.ins
      .map(nodeOf)
      .find(
        (entry) =>
          entry !== undefined &&
          isSlice(entry.node, 1, 2) &&
          entry.node.ins[0] === zero.node.ins[0],
      );
    if (data === undefined) throw new Error(`cat '${cat.outs[0]}' has no odd-channel slice`);
    replaced.set(zero.index, { ...zero.node, attrs: { ...zero.node.attrs, start: 1, end: 2 } });
    replaced.set(data.index, { ...data.node, attrs: { ...data.node.attrs, start: 0, end: 1 } });
    replaced.set(catIndex, { ...cat, ins: [sumName, zeroName] });
  }
  return nodes.map((node, index) => replaced.get(index) ?? node);
};

/** Swaps patchify's height and width sub-indices (the 1st and 2nd of each group of 4 channels `c·4 + dx·2 + dy`). */
const swapSubIndices = (input: Float32Array, plane: number): Float32Array<ArrayBuffer> => {
  const out = new Float32Array(input);
  for (let group = 0; group < input.length / (4 * plane); group += 1) {
    const one = (group * 4 + 1) * plane;
    const two = (group * 4 + 2) * plane;
    out.set(input.subarray(one, one + plane), two);
    out.set(input.subarray(two, two + plane), one);
  }
  return out;
};

// ---- encode (pre → attn → post in one batch) --------------------------------------------------

/**
 * The residents per size (pre's output / attn's output / mu). Looked up by size (1280×704 and 704×1280 have the same
 * size).
 */
class EncoderResidents {
  readonly #gpu: GpuContext;
  readonly #byArea = new Map<number, EncoderParts<ResidentTensor>>();

  constructor(gpu: GpuContext) {
    this.#gpu = gpu;
  }

  async forArea(area: number): Promise<EncoderParts<ResidentTensor>> {
    const known = this.#byArea.get(area);
    if (known !== undefined) return known;
    const hidden = HIDDEN_CHANNELS * area * BYTES_PER_ELEMENT;
    const created = {
      pre: await this.#gpu.createResident(hidden, `encoder pre ${area}`),
      attn: await this.#gpu.createResident(hidden, `encoder attn ${area}`),
      post: await this.#gpu.createResident(
        LATENT_CHANNELS * area * BYTES_PER_ELEMENT,
        `encoder mu ${area}`,
      ),
    };
    this.#byArea.set(area, created);
    return created;
  }

  get bytes(): number {
    return [...this.#byArea.values()].reduce(
      (total, { pre, attn, post }) => total + pre.byteLength + attn.byteLength + post.byteLength,
      0,
    );
  }

  /** MUST: called after the Sessions are disposed (disposal is refused while a baked-in reference remains). */
  dispose(): void {
    for (const { pre, attn, post } of this.#byArea.values()) {
      pre.dispose();
      attn.dispose();
      post.dispose();
    }
    this.#byArea.clear();
  }
}

/**
 * Runs the three graphs in one batch and reads mu (module doc "Hand-off").
 *
 * MUST: do not issue another batch or run on the same device concurrently (a batch holds the device's section lock).
 */
const encodeOnGpu = async (
  gpu: GpuContext,
  sessions: EncoderParts<Session>,
  outputs: EncoderParts<string>,
  residents: EncoderResidents,
  input: Float32Array<ArrayBuffer>,
  size: { readonly width: number; readonly height: number },
  /** Replaced bindings for post (only fault injection passes them — by default h, w are derived from size). */
  postBindings?: { readonly h: number; readonly w: number },
): Promise<Float32Array<ArrayBuffer>> => {
  const h = size.height / SPATIAL_COMPRESSION;
  const w = size.width / SPATIAL_COMPRESSION;
  const resident = await residents.forArea(h * w);
  const batch = await gpu.beginBatch();
  try {
    await sessions.pre.enqueue(
      { [INPUT.pre]: { dtype: "f32", shape: [3 * PATCH * PATCH, 1, 8 * h, 8 * w], data: input } },
      { batch, copyOutputs: { [outputs.pre]: resident.pre } },
    );
    await sessions.attn.enqueue(
      { [INPUT.attn]: resident.pre },
      { batch, bindings: { S: h * w }, copyOutputs: { [outputs.attn]: resident.attn } },
    );
    await sessions.post.enqueue(
      { [INPUT.post]: resident.attn },
      { batch, bindings: postBindings ?? { h, w }, copyOutputs: { [outputs.post]: resident.post } },
    );
  } catch (cause) {
    // MUST: close the section and return the lock (otherwise the next batch / run on the same device waits forever).
    await batch.finish().catch(() => undefined);
    throw cause;
  }
  const read = await batch.finishAndRead({ mu: resident.post });
  return new Float32Array(read["mu"]);
};

// ---- records -----------------------------------------------------------------------------------

/** Rounding up to 2 significant digits (the same rule as the band candidate of the decoder chunk-sequence e2e). */
const roundUpTwoDigits = (value: number): number => {
  const unit = 10 ** (Math.floor(Math.log10(value)) - 1);
  return Number((Math.ceil(value / unit) * unit).toPrecision(2));
};

/**
 * The dispatch counts of the three stages of fused attention and the list of pipeline keys starting with `attention`
 * (from the plan of the most recent run).
 *
 * MUST: a run without a census is not counted as 0 (the fusion check would silently become vacuous) — it throws
 * instead.
 */
const attentionPipelines = (diagnostics: SessionDiagnostics) => {
  const pipelines = diagnostics.lastRunPipelines;
  if (pipelines === undefined) throw new Error("attn: no pipeline census for the most recent run");
  const dispatches = (kernel: (typeof ATTENTION_KERNELS)[number]): number =>
    pipelines
      .filter(({ key }) => key.split(":")[0] === kernel)
      .reduce((total, { dispatchCount }) => total + dispatchCount, 0);
  return {
    dispatches: Object.fromEntries(ATTENTION_KERNELS.map((kernel) => [kernel, dispatches(kernel)])),
    keys: pipelines.filter(({ key }) => key.startsWith("attention")).map(({ key }) => key),
  };
};

/**
 * Each Session's VRAM breakdown (weights, plan backing — the intermediates of the enqueue path live in the backing) and
 * the resident size.
 */
const vramNote = (sessions: EncoderParts<Session>, residents: EncoderResidents): string => {
  const mib = (bytes: number): string => `${(bytes / 2 ** 20).toFixed(1)} MiB`;
  const parts = Object.entries(sessions).map(([part, session]) => {
    const stats = session.diagnostics();
    return `${part} weights ${mib(stats.weights.allocatedBytes)}, backing ${mib(
      stats.planBacking.residentBytes + stats.planBacking.inputBytes,
    )} (${stats.planBacking.retainedCount} kept)`;
  });
  return [...parts, `residents ${mib(residents.bytes)}`].join(" / ");
};

const maxAbsOf = (values: Float32Array): number =>
  values.reduce((max, value) => Math.max(max, Math.abs(value)), 0);

// ---- GPU setup shared by the default case and the opt-in case --------------------------------

/** The opened graphs, the three product Sessions and the residents one GPU case runs on. */
type EncoderRun = {
  readonly gpu: GpuContext;
  readonly opened: EncoderParts<BoundContainer>;
  readonly models: EncoderParts<PreparedModel>;
  readonly live: EncoderParts<Session>;
  readonly outputs: EncoderParts<string>;
  readonly residents: EncoderResidents;
};

/**
 * Opens the three graphs and builds the three product Sessions on one device (a describe's
 * beforeAll calls `open`, its afterAll calls `close`), so the default case and the opt-in parity
 * case share one lifecycle and one teardown order. `close` releases what `open` created, also when
 * `open` failed partway.
 */
const encoderLifecycle = (): {
  readonly open: () => Promise<EncoderRun>;
  readonly close: () => Promise<void>;
} => {
  let gpu: GpuContext | undefined;
  let residents: EncoderResidents | undefined;
  /**
   * The three product Sessions (disposed in the teardown — the fault-injection Sessions are disposed on the spot by
   * {@link encodeWithRewritten}).
   */
  const created: Session[] = [];
  return {
    open: async () => {
      const opened = await openParts();
      const models = prepareParts(opened);
      const outputs = {
        pre: outputOf(models.pre, PRE),
        attn: outputOf(models.attn, ATTENTION),
        post: outputOf(models.post, POST),
      };

      const device = verifyRunningGpu(await acquireTestGpu());
      gpu = device;
      const owned = new EncoderResidents(device);
      residents = owned;
      const createSession = async (model: PreparedModel): Promise<Session> => {
        const session = await model.createContainerSession(device);
        created.push(session);
        return session;
      };
      const building = performance.now();
      const live: EncoderParts<Session> = {
        pre: await createSession(models.pre),
        attn: await createSession(models.attn),
        post: await createSession(models.post),
      };
      console.log(
        `[wan-ti2v-vae-encoder] building the three Sessions ${(performance.now() - building).toFixed(0)} ms`,
      );
      return { gpu: device, opened, models, live, outputs, residents: owned };
    },
    close: async () => {
      const device = gpu;
      if (device === undefined) return;
      const owned = residents;
      // MUST: dispose in the order Session → residents → device (the residents are referenced by what the Sessions
      // baked in, so the Sessions go first; when one step fails, the remaining steps still run). A teardown failure is
      // reported by Vitest next to the failures of the its, so it does not hide them.
      await disposeSteps([
        ...created.map((session) => () => session.dispose()),
        () => owned?.dispose(),
        () => settleReleases(device),
        () => device.destroy(),
      ]);
    },
  };
};

/** One golden case encoded through the product Sessions. */
type EncodedCase = {
  readonly golden: EncoderGolden;
  readonly mu: Float32Array<ArrayBuffer>;
  readonly latent: Float32Array;
  readonly encodeMs: number;
};

/**
 * Reads the golden, asserts that the TS preprocessing is bit-identical to its encoder input (the GPU
 * gets the TS output, as in the product path), encodes that input and normalizes the mu.
 */
const encodeCase = async (run: EncoderRun, testCase: EncoderCase): Promise<EncodedCase> => {
  const golden = await readGolden(testCase);
  const input = encoderInputOf(testCase, golden.rgb8);
  const mismatch = firstBitMismatch(input, golden.encoderInput);
  assertEquals(
    mismatch,
    -1,
    `${testCase.name}: the TS preprocessing differs from the golden's encoder input`,
  );
  const started = performance.now();
  const mu = await encodeOnGpu(run.gpu, run.live, run.outputs, run.residents, input, testCase);
  const encodeMs = performance.now() - started;
  return {
    golden,
    mu,
    latent: normalizeWanLatents(mu, WAN22_TI2V_GENERATION.latents),
    encodeMs,
  };
};

const nonFiniteCountOf = (values: Float32Array): number =>
  values.reduce((count, value) => (Number.isFinite(value) ? count : count + 1), 0);

// ---- tests -------------------------------------------------------------------------------------

test.skipIf(!ANY_ASSETS && !FETCHED)(
  "Wan2.2 TI2V VAE encoder: the 9 goldens are present (some or a complete fetch → FAIL, none → SKIP)",
  () => {
    const missing = ASSETS.filter(({ present }) => !present).map(({ path }) => path);
    assertEquals(missing, [], `missing encoder goldens (${MISSING_GOLDENS})`);
  },
);

// Default run: the I/O contract, the bit identity of the TS preprocessing with the golden encoder
// input, a finite mu and the fused-attention census, for all 9 cases through one Session per graph
// (so a binding change that carries over the previous plan still fails or goes non-finite here).
// It owns the per-case results.json ids; the band comparison with the diffusers mu is the opt-in
// case below.
describe.skipIf(!ALL_ASSETS || !GPU_AVAILABLE)(
  "Wan2.2 TI2V VAE encoder: pre → attn → post (resident hand-off) contract, bit-identical TS preprocessing, " +
    "finite mu, the three stages of fused attention (real GPU)",
  () => {
    const lifecycle = encoderLifecycle();
    // Assigned in beforeAll; the its below run sequentially against these Sessions.
    let run: EncoderRun;
    beforeAll(async () => {
      run = await lifecycle.open();
    });
    afterAll(() => lifecycle.close());

    it("contract: input / output names and shapes, symbols (pre [h,w], attn [S], post [h,w])", () => {
      const { models } = run;
      const io = (model: PreparedModel, part: string) => ({
        symbols: [...model.graph.symbols],
        inputs: model.graph.inputs.map(({ name, shape }) => [name, [...shape]]),
        output: [...(model.graph.values[outputOf(model, part)]?.shape ?? [])],
      });
      assertEquals(io(models.pre, PRE), {
        symbols: ["h", "w"],
        inputs: [[INPUT.pre, [3 * PATCH * PATCH, 1, "8h", "8w"]]],
        output: [HIDDEN_CHANNELS, 1, "h", "w"],
      });
      assertEquals(io(models.attn, ATTENTION), {
        symbols: ["S"],
        inputs: [[INPUT.attn, [HIDDEN_CHANNELS, "S"]]],
        output: [HIDDEN_CHANNELS, "S"],
      });
      assertEquals(io(models.post, POST), {
        symbols: ["h", "w"],
        inputs: [[INPUT.post, [HIDDEN_CHANNELS, 1, "h", "w"]]],
        output: [LATENT_CHANNELS, 1, "h", "w"],
      });
    });

    /** Whether a case is the first of its size (its wall time includes deriving the plan and building pipelines). */
    const seenSizes = new Set<string>();
    for (const testCase of CASES) {
      it(
        `${testCase.name} (${testCase.role}): the TS preprocessing is bit-identical to the golden's encoder input, ` +
          "mu is finite and goes through the three stages of fused attention",
        async () => {
          const { live, residents } = run;
          await runRecordedCase(results, { id: testCase.name }, async () => {
            const sizeKey = `${testCase.width}x${testCase.height}`;
            const cold = !seenSizes.has(sizeKey);
            seenSizes.add(sizeKey);
            const { latent, encodeMs } = await encodeCase(run, testCase);
            const nonFinite = nonFiniteCountOf(latent);
            const attention = attentionPipelines(live.attn.diagnostics());
            const line = `${testCase.name}: nonFinite=${nonFinite} encodeMs=${encodeMs.toFixed(0)}${
              cold ? " (first of its size)" : ""
            } / ${vramNote(live, residents)} / attention ${JSON.stringify(
              attention.dispatches,
            )} ${attention.keys.join(" ")}`;
            console.log(`[wan-ti2v-vae-encoder] ${line}`);
            assertEquals(nonFinite, 0, `${testCase.name}: non-finite values`);
            // The mid attention (D = 640, S = h·w) goes through the three stages of fused attention (module doc). The
            // row-statistics variant changes with S, so it is checked per size.
            for (const kernel of ATTENTION_KERNELS) {
              assert(
                attention.dispatches[kernel] > 0,
                `${testCase.name}: 0 dispatches of ${kernel} (${JSON.stringify(
                  attention.dispatches,
                )})`,
              );
            }
            return { status: "pass", note: line };
          });
        },
      );
    }
  },
);

// Upstream parity (the normalized mu against the diffusers non-tiled encode under the absolute
// band), so it is opt-in, together with the fault injections that only prove the band is live and
// the band-candidate step. It writes the fault/* and band-candidate results.json ids; the per-case
// ids belong to the default case above, and writing them here would duplicate its settlements.
describe.skipIf(!ALL_ASSETS || !GPU_AVAILABLE || !UPSTREAM_PARITY)(
  "Wan2.2 TI2V VAE encoder: the normalized mu of pre → attn → post (resident hand-off) matches upstream's untiled " +
    `encode within the band, and fault injection falls outside it (real GPU)${UPSTREAM_PARITY_OPT_IN}`,
  () => {
    const lifecycle = encoderLifecycle();
    // Assigned in beforeAll; the its below run sequentially against these Sessions.
    let run: EncoderRun;
    let faultCase: EncoderCase;
    beforeAll(async () => {
      assert(FAULT_CASE !== undefined, "the fault-injection case is not in the table");
      faultCase = FAULT_CASE;
      run = await lifecycle.open();
    });
    afterAll(() => lifecycle.close());

    /**
     * maxAbs per case (the material for the band candidate — stored before the verdict, so a run outside the band is
     * kept too).
     */
    const observed = new Map<string, number>();
    /** The correct mu of the acceptance case (the fault-injection input; used by the std / 1/std injection). */
    let faultMu: Float32Array<ArrayBuffer> | undefined;

    for (const testCase of CASES) {
      it(
        `${testCase.name} (${testCase.role}): the TS preprocessing is bit-identical to the golden's encoder input, ` +
          "and the normalized mu is inside the band",
        async () => {
          const { golden, mu, latent } = await encodeCase(run, testCase);
          const report = allclose(latent, golden.latent, TOLERANCE);
          const muReport = allclose(mu, golden.mu, { atol: Number.POSITIVE_INFINITY, rtol: 0 });
          const line = `${testCase.name}: latent maxAbs=${report.maxAbsError} refMax=${maxAbsOf(
            golden.latent,
          )} / mu maxAbs=${muReport.maxAbsError} refMax=${maxAbsOf(
            golden.mu,
          )} nonFinite=${report.nonFiniteCount}`;
          console.log(`[wan-ti2v-vae-encoder] ${line}`);
          assertEquals(report.nonFiniteCount, 0, `${testCase.name}: non-finite values`);
          observed.set(testCase.name, report.maxAbsError);
          if (testCase === FAULT_CASE) faultMu = mu;
          if (BAND === undefined) {
            throw new Error(
              `${testCase.name}: band not derived yet (maxAbs ${report.maxAbsError}) — write the worst band-setting ` +
                "case × 5 into the band constant (the last step prints the band candidate)",
            );
          }
          assert(
            report.pass,
            `${testCase.name}: maxAbs ${report.maxAbsError} is outside the band ${BAND}`,
          );
        },
      );
    }

    /** Encodes the faulty input and checks that it falls outside the band against the acceptance golden. */
    const faultIt = (
      id: string,
      label: string,
      latentOf: (golden: EncoderGolden, faultCase: EncoderCase) => Promise<Float32Array>,
    ): void => {
      it(`fault injection: ${label} → ${FAULT_CASE?.name ?? FAULT_IMAGE} is outside the band`, async () => {
        await runRecordedCase(results, { id: `fault/${id}` }, async ({ measurements }) => {
          const golden = await readGolden(faultCase);
          const latent = await latentOf(golden, faultCase);
          const report = allclose(latent, golden.latent, TOLERANCE);
          measurements.push({
            output: `latent(fault ${id})`,
            maxAbs: report.maxAbsError,
            maxRel: report.maxRelError,
            tolerance: TOLERANCE,
            stage: "karume",
          });
          const line = `fault ${id}: maxAbs=${report.maxAbsError} nonFinite=${report.nonFiniteCount}${
            BAND === undefined ? "" : ` band×${(report.maxAbsError / BAND).toFixed(0)}`
          }`;
          console.log(`[wan-ti2v-vae-encoder] ${line}`);
          if (BAND === undefined) {
            throw new Error(
              `${label}: the band is not derived yet (fault-injection maxAbs ${report.maxAbsError})`,
            );
          }
          assert(
            !report.pass,
            `${label}: fell inside the band ${BAND} (a sign that the band is too wide)`,
          );
          return { status: "pass", note: line };
        });
      });
    };

    faultIt(
      "patchify-swap",
      "patchify height and width sub-indices swapped",
      async (golden, faultCase) => {
        const { gpu, live, outputs, residents } = run;
        const plane = (faultCase.height / PATCH) * (faultCase.width / PATCH);
        const input = swapSubIndices(encoderInputOf(faultCase, golden.rgb8), plane);
        const mu = await encodeOnGpu(gpu, live, outputs, residents, input, faultCase);
        return normalizeWanLatents(mu, WAN22_TI2V_GENERATION.latents);
      },
    );

    faultIt(
      "std-inverse",
      "std and 1/std mixed up (mu · std — host normalization only, the GPU is not run)",
      () => {
        if (faultMu === undefined) {
          throw new Error(`no mu for ${faultCase.name} (the comparison step failed)`);
        }
        const { mean, std } = WAN22_TI2V_GENERATION.latents;
        return Promise.resolve(
          normalizeWanLatents(faultMu, { mean, std: std.map((value) => 1 / value) }),
        );
      },
    );

    faultIt(
      "post-bindings-swap",
      "h and w swapped in post's bindings (the resident byte count is the same — it passes the hand-off check)",
      async (golden, faultCase) => {
        const { gpu, live, outputs, residents } = run;
        const h = faultCase.height / SPATIAL_COMPRESSION;
        const w = faultCase.width / SPATIAL_COMPRESSION;
        assert(h !== w, `${faultCase.name}: with h = w the swap is vacuous`);
        const mu = await encodeOnGpu(
          gpu,
          live,
          outputs,
          residents,
          encoderInputOf(faultCase, golden.rgb8),
          faultCase,
          { h: w, w: h },
        );
        return normalizeWanLatents(mu, WAN22_TI2V_GENERATION.latents);
      },
    );

    /**
     * Encodes with a Session of one part whose declaration is replaced (the other two parts use the product Sessions).
     *
     * The injection Session is disposed right after use and its release is awaited (the second pre holds about 0.9 GiB
     * in weights and the backing of the 1280×704 plan — carried over to the remaining steps and the teardown, the
     * injection step alone could OOM on a machine with little VRAM). A disposal failure does not overwrite the main
     * failure.
     */
    const encodeWithRewritten = async (
      part: "pre" | "post",
      rewrite: (nodes: readonly IrNode[]) => readonly IrNode[],
      golden: EncoderGolden,
      faultCase: EncoderCase,
    ): Promise<Float32Array> => {
      const { gpu, opened, live, outputs, residents } = run;
      const name = part === "pre" ? PRE : POST;
      const graphName = name;
      const faulty = await prepareContainer(
        withRewrittenNodes(opened[part], graphName, rewrite),
        graphName,
      ).createContainerSession(gpu);
      let injected: { readonly error: unknown } | undefined;
      try {
        const mu = await encodeOnGpu(
          gpu,
          { ...live, [part]: faulty },
          outputs,
          residents,
          encoderInputOf(faultCase, golden.rgb8),
          faultCase,
        );
        return normalizeWanLatents(mu, WAN22_TI2V_GENERATION.latents);
      } catch (error) {
        injected = { error };
        throw error;
      } finally {
        await disposeSteps([
          () => {
            if (injected !== undefined) throw injected.error;
          },
          () => faulty.dispose(),
          () => settleReleases(gpu),
        ]);
      }
    };

    faultIt(
      "logvar",
      "the 48 logvar channels instead of mu (post's output slice moved to the second half)",
      (golden, faultCase) =>
        encodeWithRewritten("post", takeLogvarHalf(run.outputs.post), golden, faultCase),
    );

    faultIt(
      "avgdown-parity",
      "AvgDown3D parity mixed up (the mean added to the even channels — the two blocks with temporal factor 2)",
      (golden, faultCase) => encodeWithRewritten("pre", swapShortcutParity, golden, faultCase),
    );

    if (BAND === undefined) {
      it("band candidate (not derived yet)", async () => {
        const bandCases = CASES.filter(({ role }) => role === "band");
        const missing = bandCases.filter(({ name }) => !observed.has(name)).map(({ name }) => name);
        const worst = Math.max(...bandCases.map(({ name }) => observed.get(name) ?? 0));
        const candidate =
          missing.length > 0
            ? `band-setting ${missing.join(" / ")} failed before reporting a maxAbs — no candidate`
            : worst > 0
              ? `worst band-setting case ${worst.toExponential(3)} × 5 = ${(worst * 5).toExponential(3)}` +
                ` (rounded up to 2 significant digits: ${roundUpTwoDigits(worst * 5)})`
              : "the band-setting maxAbs is 0 — × 5 cannot derive a band (investigate the comparison path first)";
        const accepted = CASES.filter(({ role }) => role === "accept").map(
          ({ name }) => `${name} ${observed.get(name)?.toExponential(3) ?? "—"}`,
        );
        const message = `the band is not derived yet — ${candidate}. Acceptance maxAbs (not used to set the band): ${accepted.join(
          " / ",
        )}`;
        await runRecordedCase(results, { id: "band-candidate", failureNote: () => message }, () => {
          console.log(`[wan-ti2v-vae-encoder] ${message}`);
          throw new Error(message);
        });
      });
    }
  },
);
