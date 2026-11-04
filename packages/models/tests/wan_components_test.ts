/**
 * Exercises the **component replacement** of Wan2.1's umT5 (`text_encoder`) (`components` — ADR 0122
 * decision 7 item 3, ADR 0108 decision 19) through the `Pipeline.fromPretrained` entry. Needs neither
 * a GPU nor real assets (it stands up mock HF repositories and in-memory directories, and observes the
 * (repository, path) pairs requested).
 *
 * The distribution skeleton is the same as the real one (`models/karume-wan2.1`): the `text_encoder`
 * of the Wan repository is a **cross-repository reference** to the umT5 distribution repository
 * (`repo` + `revision` per part), and the replacement is the same role in another repository. Four
 * things are pinned:
 *
 * (1) **Only the replaced `text_encoder` comes from the replacement repository** — not one byte is
 *     fetched from the original reference target (the umT5 distribution repository) (control: without
 *     the replacement, the target's parts are fetched).
 * (2) **Replacing a cross-referenced slot in a local distribution (a local source) works without the
 *     original target's `crossRepo` mapping** (control: without the replacement it fails for lack of
 *     the mapping). A replaced slot is read from the replacement's source and does not follow the
 *     original slot's cross-repository reference (where a slot comes from in `src/hub/components.ts`).
 *     A local source has no fetch phase and reads the weights while building the Session, so the
 *     weight fetch itself is observed with an HTTP replacement.
 * (3) **An i4 replacement is rejected by Wan's family check (`umt5Contract`) before any weight part is
 *     fetched, even with the same graph descriptor** (the component replacement only checks the graph
 *     descriptor's sha256, so the storage check lives on the family side).
 * (4) **Fault injection: changing one character of the replacement's graph-descriptor sha256 makes
 *     admission reject it** (no container is fetched).
 *
 * Observation setup: the embedding asset (`text_embeds`) is 8 bytes that cannot be read as
 * safetensors. The fetch surface is built in the order admission → fetching the weight parts →
 * fetching the assets → parsing the assets, so the happy path stops at "parsing the embedding asset"
 * (= it got through admission and the weight fetch of every component — the same "observe passage by
 * the next step's message" form as `irodori_admission_test.ts`). The shared mock GPU is passed, and
 * the adapter is not touched either.
 */

import { readFile } from "@karume/test-support/fs";
import {
  assertEquals,
  assertInstanceOf,
  assertNotEquals,
  assertRejects,
  assertStringIncludes,
} from "@std/assert";
import { describe, it } from "vite-plus/test";
import { SafetensorsError } from "@karume/runtime";
import {
  type DistributionSource,
  type FileRef,
  type HubRepoRef,
  localDirectory,
} from "@karume/hub";
import { type FromPretrainedOptions, Pipeline } from "../src/wan/pipeline.ts";
import type { ModelInput, TensorInput } from "../../runtime/tests/helpers/container-write.ts";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";
import { parseIrDeclarationValue } from "./helpers/container-fixture.ts";
import {
  fileRef,
  HUB_URL,
  MANIFEST_PATH,
  manifestBytes,
  type MockFetch,
  serveContainer,
  type ServedContainer,
  serveRepos,
  SHA,
} from "./helpers/container-loading-fixture.ts";
import { MemoryCacheStorage } from "./helpers/memory-cache.ts";

/** The mock repository of the Wan distribution. */
const WAN_REPO = "karume-test/wan";
/** The umT5 distribution repository that Wan's `text_encoder` references (the slot of the real `karume-umt5-xxl`). */
const UMT5_REPO = "karume-test/umt5-xxl";
/** The revision the cross-repository reference pins (a value different from the source session's SHA, to catch a mix-up). */
const UMT5_REVISION = "fedcba9876543210fedcba9876543210fedcba98";
/** The replacement (another repository holding a compatible umT5). */
const REPLACEMENT_REPO = "karume-test/umt5-compatible";

/**
 * The message showing it stopped while parsing the embedding asset (the observation setup — module
 * doc). The thrower is `SafetensorsError`, and the only other safetensors this skeleton reads is
 * `rope_base` (a valid asset), so the happy path pairs it with the assertion that the assets were
 * fetched to pin the stop at the embedding asset.
 */
const STOPPED_AT_EMBEDS = "Cannot parse the header JSON";

/** The raw RoPE table (the asset `rope_base` of the `transformer` container — the same widths 22 / 21 / 21 as the real 1.3B). */
const ROPE_BASE = await readFile(
  new URL("./fixtures/wan-dit/rope_base.safetensors", import.meta.url),
);

/** The DiT declaration (the same as the valid form of `ditContract` in `wan_pipeline_test.ts` — never run). */
const transformerInput = (): ModelInput => ({
  graphs: {
    transformer: parseIrDeclarationValue({
      format: "karume-ir",
      version: 2,
      requires: { ops: ["mul"] },
      symbols: ["S"],
      inputs: [
        { name: "tokens", dtype: "f32", shape: [1, "S", 64] },
        { name: "timesteps_proj", dtype: "f32", shape: [1, 256] },
        { name: "encoder_hidden_states", dtype: "f32", shape: [1, 512, 4096] },
        { name: "rope_cos", dtype: "f32", shape: [1, "S", 1, 128] },
        { name: "rope_sin", dtype: "f32", shape: [1, "S", 1, 128] },
      ],
      outputs: ["out"],
      initializers: {},
      values: { out: { dtype: "f32", shape: [1, "S", 64] } },
      states: {},
      nodes: [{ op: "mul", ins: ["tokens", "tokens"], outs: ["out"], attrs: {} }],
    }),
  },
  consts: [],
  weights: [],
  assets: [{ name: "rope_base", role: "rope-base", bytes: ROPE_BASE, dedicatedPart: true }],
  provenance: { license: "test" },
});

/** The declaration of the VAE chunk graph (latent tile 32, scale 8, one cache — the same geometry as the distribution). */
const vaeInput = (graph: string, frames: number): ModelInput => ({
  graphs: {
    [graph]: parseIrDeclarationValue({
      format: "karume-ir",
      version: 2,
      requires: { ops: ["mul"] },
      symbols: [],
      inputs: [
        { name: "latent", dtype: "f32", shape: [16, 1, 32, 32] },
        { name: "cache_00", dtype: "f32", shape: [4, 2, 8, 8] },
      ],
      outputs: ["frame", "cache_out_00"],
      initializers: {},
      values: {
        frame: { dtype: "f32", shape: [3, frames, 256, 256] },
        cache_out_00: { dtype: "f32", shape: [4, 2, 8, 8] },
      },
      states: {},
      nodes: [
        { op: "mul", ins: ["latent", "latent"], outs: ["frame"], attrs: {} },
        { op: "mul", ins: ["cache_00", "cache_00"], outs: ["cache_out_00"], attrs: {} },
      ],
    }),
  },
  consts: [],
  weights: [],
  assets: [],
  provenance: { license: "test" },
});

/** The shape of the umT5 vocabulary embedding (2 vocabulary rows × width 4096 — the width is the DiT's context width). */
const EMBED_ROWS = 2;
const EMBED_WIDTH = 4096;

const f32Bytes = (values: readonly number[]): Uint8Array<ArrayBuffer> =>
  new Uint8Array(Float32Array.from(values).buffer);

/** The storage of the vocabulary embedding (i8 = per-channel, i4 = a group per 32 elements). `fill` is the weight byte value. */
const embedWeight = (storage: "i8" | "i4", fill: number): TensorInput => {
  const elements = EMBED_ROWS * EMBED_WIDTH;
  if (storage === "i8") {
    return {
      graph: "text_encoder",
      initializer: "embed",
      bytes: new Uint8Array(elements).fill(fill),
      encoding: {
        codec: "int8-sym",
        groupSize: EMBED_WIDTH,
        scale: { bytes: f32Bytes(Array(EMBED_ROWS).fill(0.5)), dtype: "f32" },
      },
    };
  }
  const groupSize = 32;
  return {
    graph: "text_encoder",
    initializer: "embed",
    bytes: new Uint8Array(elements / 2).fill(fill),
    encoding: {
      codec: "int4-sym-g",
      groupSize,
      scale: {
        bytes: f32Bytes(Array(EMBED_ROWS * (EMBED_WIDTH / groupSize)).fill(0.5)),
        dtype: "f32",
      },
    },
  };
};

/**
 * The umT5 declaration (the valid form of `umt5Contract` — two i32 inputs, symbol L, output
 * `[1, L, 4096]`). The graph declaration does not depend on the storage, so the graph descriptor is
 * byte-identical for i8 and i4 (the storage goes into the model descriptor's binding table).
 */
const umt5Input = (storage: "i8" | "i4", fill: number): ModelInput => ({
  graphs: {
    text_encoder: parseIrDeclarationValue({
      format: "karume-ir",
      version: 2,
      requires: { ops: ["embedding"] },
      symbols: ["L"],
      inputs: [
        { name: "input_ids", dtype: "i32", shape: [1, "L"] },
        { name: "relative_position_buckets", dtype: "i32", shape: ["L", "L"] },
      ],
      outputs: ["out"],
      initializers: { embed: {} },
      values: {
        embed: { dtype: "f32", shape: [EMBED_ROWS, EMBED_WIDTH] },
        out: { dtype: "f32", shape: [1, "L", EMBED_WIDTH] },
      },
      states: {},
      nodes: [
        {
          op: "embedding",
          ins: ["embed", "input_ids"],
          outs: ["out"],
          attrs: { padding_idx: -1 },
        },
      ],
    }),
  },
  consts: [],
  weights: [embedWeight(storage, fill)],
  assets: [],
  provenance: { license: "test" },
});

/** A sha256 with one character changed (the last hex digit set to another value). */
const flipLastHex = (sha256: string): string =>
  sha256.slice(0, -1) + (sha256.endsWith("0") ? "1" : "0");

/** Replaces only the graph-descriptor sha256 of one distribution entry (`ServedContainer.entry`). */
const withGraphSha = (served: ServedContainer, sha256: string): Record<string, unknown> => {
  const { descriptor } = served.written;
  return {
    container: {
      descriptor: { ...descriptor, graph: { ...descriptor.graph, sha256 } },
      parts: served.parts,
    },
  };
};

/** A cross-repository distribution entry (`repo` + `revision` on every part — the same spelling as the real Wan manifest). */
const crossRepoEntry = (served: ServedContainer): Record<string, unknown> => ({
  container: {
    descriptor: served.written.descriptor,
    parts: served.parts.map((part) => ({ repo: UMT5_REPO, revision: UMT5_REVISION, ...part })),
  },
});

/** One mock repository (the input of `serveRepos`, and the same contents a local source reads). */
type RepoFiles = {
  readonly repo: string;
  readonly models: Record<string, unknown>;
  readonly files: readonly (readonly [string, Uint8Array<ArrayBuffer>])[];
};

/** Uses an in-memory directory as the source (records the paths read — the slot of a local distribution). */
const memoryDirectory = (spec: RepoFiles) => {
  const files = new Map<string, Uint8Array<ArrayBuffer>>([
    [MANIFEST_PATH, manifestBytes(spec.models)],
    ...spec.files,
  ]);
  const reads: string[] = [];
  const source: DistributionSource = localDirectory(
    {
      readFile: (path) => {
        reads.push(path);
        const bytes = files.get(path);
        if (bytes === undefined) return Promise.reject(new Error(`${spec.repo}: no ${path}`));
        // MUST: return a view spanning the whole buffer (the contract of `DirectoryAdapter.readFile`).
        return Promise.resolve(bytes.slice());
      },
    },
    { label: spec.repo },
  );
  return { source, reads };
};

type WanRigOptions = {
  /** Makes the replacement claim a graph-descriptor sha256 with one character changed (fault injection (4)). */
  readonly corruptReplacementGraphSha?: boolean;
};

/**
 * Builds the 3 repositories (Wan, the umT5 reference target, the replacement). The replacement's i8 is
 * placed at **the same path** as the original umT5 with only the weight values changed — so which one
 * was fetched can only be told by the repository coordinates.
 */
const prepareWanRepos = async (options: WanRigOptions = {}) => {
  const transformer = await serveContainer("test/transformer/model.f32", transformerInput(), {
    partBytes: 64 * 1024,
    blockBytes: 64 * 1024,
  });
  const first = await serveContainer(
    "test/vae_decoder_first/model.f32",
    vaeInput("vae_decoder_first", 1),
  );
  const next = await serveContainer(
    "test/vae_decoder_next/model.f32",
    vaeInput("vae_decoder_next", 4),
  );
  // The block is sized to hold one vocabulary-embedding row (4096 bytes in i8).
  const umt5Write = { partBytes: 16 * 1024, blockBytes: 4096 };
  const umt5 = await serveContainer("xxl/text_encoder/model.i8", umt5Input("i8", 1), umt5Write);
  const replacement = await serveContainer(
    "xxl/text_encoder/model.i8",
    umt5Input("i8", 2),
    umt5Write,
  );
  const replacementI4 = await serveContainer(
    "xxl/text_encoder/model.i4",
    umt5Input("i4", 3),
    umt5Write,
  );
  // Precondition: all three share the graph descriptor (the condition for accepting a replacement)
  // and differ in their weight parts.
  for (const other of [replacement, replacementI4]) {
    assertEquals(other.written.descriptor.graph.sha256, umt5.written.descriptor.graph.sha256);
  }
  assertNotEquals(replacement.parts.at(-1)?.sha256, umt5.parts.at(-1)?.sha256);

  const embeds = new Uint8Array(8); // unreadable as safetensors (the observation setup — module doc)
  const tokenizer = new TextEncoder().encode("{}");
  const embedsRef = await fileRef("test/text_embeds/text_embeds.safetensors", embeds);
  const tokenizerRef = await fileRef("test/umt5_tokenizer/tokenizer.json", tokenizer);
  const wan: RepoFiles = {
    repo: WAN_REPO,
    models: {
      test: {
        pipeline: "wan/1",
        weights: {
          transformer: { f32: transformer.entry },
          vae_decoder_first: { f32: first.entry },
          vae_decoder_next: { f32: next.entry },
          text_encoder: { i8: crossRepoEntry(umt5) },
        },
        assets: { text_embeds: embedsRef, umt5_tokenizer: tokenizerRef },
        quants: {
          f32: {
            weights: {
              transformer: "f32",
              vae_decoder_first: "f32",
              vae_decoder_next: "f32",
              text_encoder: "i8",
            },
            session: {},
          },
        },
        defaultQuant: "f32",
        pipelineConfig: { scheduler: { shift: 3 }, defaults: { steps: 50, guidance: 5 } },
      },
    },
    files: [
      ...transformer.files,
      ...first.files,
      ...next.files,
      [embedsRef.path, embeds],
      [tokenizerRef.path, tokenizer],
    ],
  };
  const umt5Repo: RepoFiles = {
    repo: UMT5_REPO,
    models: {
      test: {
        pipeline: "umt5-encoder/1",
        weights: { text_encoder: { i8: umt5.entry } },
        assets: {},
        quants: { i8: { weights: { text_encoder: "i8" }, session: {} } },
        defaultQuant: "i8",
        pipelineConfig: {},
      },
    },
    files: umt5.files,
  };
  const replacementI8Entry =
    options.corruptReplacementGraphSha === true
      ? withGraphSha(replacement, flipLastHex(replacement.written.descriptor.graph.sha256))
      : replacement.entry;
  const replacementRepo: RepoFiles = {
    repo: REPLACEMENT_REPO,
    models: {
      test: {
        pipeline: "umt5-encoder/1",
        weights: { text_encoder: { i8: replacementI8Entry, i4: replacementI4.entry } },
        assets: {},
        quants: {
          i8: { weights: { text_encoder: "i8" }, session: {} },
          i4: { weights: { text_encoder: "i4" }, session: {} },
        },
        defaultQuant: "i8",
        pipelineConfig: {},
      },
    },
    files: [...replacement.files, ...replacementI4.files],
  };
  return {
    repos: { wan, umt5: umt5Repo, replacement: replacementRepo },
    parts: {
      transformer: transformer.parts,
      vae: [...first.parts, ...next.parts],
      umt5: umt5.parts,
      replacement: replacement.parts,
      replacementI4: replacementI4.parts,
    },
    assets: [embedsRef, tokenizerRef],
  };
};

type WanRig = Awaited<ReturnType<typeof prepareWanRepos>>;

/** Parts with content (a part of length 0 is not fetched — the hub skips it too). */
const nonEmpty = (parts: readonly FileRef[]): readonly FileRef[] =>
  parts.filter((part) => part.size > 0);

/** The shared build options (the shared mock GPU — the adapter's limits are not read). */
const BASE_OPTIONS: FromPretrainedOptions = { gpu: fakeGpuContext(fakeDevice()) };

/** Stands up the 3 mock HF repositories and provides `fromPretrained` from the Wan repository. */
const overHttp = (rig: WanRig) => {
  const mock: MockFetch = serveRepos([rig.repos.wan, rig.repos.umt5, rig.repos.replacement]);
  const hubOptions = { fetch: mock.fetch, caches: new MemoryCacheStorage() };
  const replacementSource: HubRepoRef = {
    repo: REPLACEMENT_REPO,
    revision: SHA,
    hubUrl: HUB_URL,
  };
  const load = (options: FromPretrainedOptions = {}) =>
    Pipeline.fromPretrained(
      { repo: WAN_REPO, revision: SHA, hubUrl: HUB_URL },
      { ...BASE_OPTIONS, ...hubOptions, ...options },
    );
  /** The paths fetched from that repository (in request order). */
  const asked = (repo: string): readonly string[] =>
    mock.requests.filter((request) => request.repo === repo).map((request) => request.path);
  return { load, asked, replacementSource };
};

describe("replacing Wan's text_encoder (mock HF — a cross-referenced slot)", () => {
  it("control: without a replacement, text_encoder is fetched from the referenced umT5 repository", async () => {
    const rig = await prepareWanRepos();
    const http = overHttp(rig);

    await assertRejects(() => http.load(), SafetensorsError, STOPPED_AT_EMBEDS);

    for (const part of nonEmpty(rig.parts.umt5)) {
      assertEquals(http.asked(UMT5_REPO).includes(part.path), true, `${part.path} was not fetched`);
    }
    for (const ref of rig.assets) {
      assertEquals(http.asked(WAN_REPO).includes(ref.path), true, `${ref.path} was not fetched`);
    }
    assertEquals(http.asked(REPLACEMENT_REPO), []);
  });

  it("only the replaced text_encoder comes from the replacement, and not one byte is fetched from the original target", async () => {
    const rig = await prepareWanRepos();
    const http = overHttp(rig);

    // It stops at parsing the embedding asset = after getting through admission and the weight fetch
    // of every component.
    await assertRejects(
      () => http.load({ components: { text_encoder: { source: http.replacementSource } } }),
      SafetensorsError,
      STOPPED_AT_EMBEDS,
    );

    // Nothing was ever fetched from the original target (including the descriptor in part 0).
    assertEquals(http.asked(UMT5_REPO), []);
    // Only the manifest and the text_encoder parts came from the replacement (neither the i4 slot nor
    // other roles are fetched).
    assertEquals(
      [...http.asked(REPLACEMENT_REPO)].sort(),
      [MANIFEST_PATH, ...nonEmpty(rig.parts.replacement).map((part) => part.path)].sort(),
    );
    // The roles not replaced and the assets are still fetched from the Wan repository (the
    // text_encoder paths are not in the Wan repository).
    for (const ref of [...nonEmpty(rig.parts.transformer), ...nonEmpty(rig.parts.vae)]) {
      assertEquals(http.asked(WAN_REPO).includes(ref.path), true, `${ref.path} was not fetched`);
    }
    for (const ref of rig.assets) {
      assertEquals(http.asked(WAN_REPO).includes(ref.path), true, `${ref.path} was not fetched`);
    }
    for (const part of rig.parts.replacement) {
      assertEquals(
        http.asked(WAN_REPO).includes(part.path),
        false,
        `${part.path} was fetched from Wan`,
      );
    }
  });

  it("an i4 replacement is rejected by umt5Contract before any weight part is fetched (even with the same graph descriptor)", async () => {
    const rig = await prepareWanRepos();
    const http = overHttp(rig);

    const error = await assertRejects(
      () =>
        http.load({
          components: { text_encoder: { source: http.replacementSource, quant: "i4" } },
        }),
      Error,
      "text_encoder initializer 'embed' has storage int4-sym-g, which is not accepted",
    );
    assertStringIncludes(error.message, "ADR 0119 decision 5");

    // The descriptor (part 0) was fetched for admission, but no weight part was fetched.
    const [descriptor, ...weightParts] = rig.parts.replacementI4;
    assertEquals(http.asked(REPLACEMENT_REPO).includes(descriptor.path), true);
    for (const part of nonEmpty(weightParts)) {
      assertEquals(
        http.asked(REPLACEMENT_REPO).includes(part.path),
        false,
        `${part.path} was fetched`,
      );
    }
    assertEquals(
      nonEmpty(weightParts).length > 0,
      true,
      "not observed with an i4 that has weight parts",
    );
    // It reached neither the asset step nor the original target.
    for (const ref of rig.assets) {
      assertEquals(http.asked(WAN_REPO).includes(ref.path), false, `${ref.path} was fetched`);
    }
    assertEquals(http.asked(UMT5_REPO), []);
  });

  it("fault injection: changing one character of the replacement's graph-descriptor sha256 makes admission reject it, and no container is fetched", async () => {
    const rig = await prepareWanRepos({ corruptReplacementGraphSha: true });
    const http = overHttp(rig);

    const error = await assertRejects(
      () => http.load({ components: { text_encoder: { source: http.replacementSource } } }),
      Error,
      "the graph descriptor does not match the manifest's declaration",
    );
    assertStringIncludes(error.message, "components['text_encoder']");
    // The declaration alone decides it — no `.krm` of any repository was fetched (only the manifest
    // of the replacement).
    assertEquals(http.asked(REPLACEMENT_REPO), [MANIFEST_PATH]);
    assertEquals(http.asked(UMT5_REPO), []);
    assertEquals(http.asked(WAN_REPO), [MANIFEST_PATH]);
  });
});

describe("Wan2.1's components carry no 32-latent decoder pair (ADR 0134 — its distribution has no _32 roles)", () => {
  it("replacing 'vae_decoder_first_32' is rejected as not a component of this family, before any container is fetched", async () => {
    const rig = await prepareWanRepos();
    const http = overHttp(rig);
    const error = await assertRejects(
      () => http.load({ components: { vae_decoder_first_32: { source: http.replacementSource } } }),
      Error,
      "'vae_decoder_first_32' in components is not a component of this family" +
        " (replaceable roles: transformer / vae_decoder_first / vae_decoder_next / text_encoder)",
    );
    assertStringIncludes(error.message, "wan.Pipeline.fromPretrained");
    assertEquals(http.asked(WAN_REPO), [MANIFEST_PATH]);
    assertEquals(http.asked(REPLACEMENT_REPO), []);
  });
});

describe("replacing Wan's text_encoder (a local distribution — no mapping for the reference target)", () => {
  it("control: without a replacement it fails for lack of a crossRepo mapping for the target", async () => {
    const rig = await prepareWanRepos();
    const wan = memoryDirectory(rig.repos.wan);

    const error = await assertRejects(
      () => Pipeline.fromPretrained(wan.source, BASE_OPTIONS),
      Error,
      `${rig.parts.umt5[0].path}: prefetch failed (repo ${UMT5_REPO} @ ${UMT5_REVISION})`,
    );
    // The fetch layer keeps the source's guidance in cause (the hub's local.ts — it does not guess a
    // neighbouring directory).
    assertInstanceOf(error.cause, Error);
    assertStringIncludes(error.cause.message, `no cross-repo target for repo '${UMT5_REPO}'`);
    assertStringIncludes(error.cause.message, "crossRepo");
    // The target's parts were not read from the Wan source (no guessing of a neighbour).
    for (const part of rig.parts.umt5) {
      assertEquals(wan.reads.includes(part.path), false, `${part.path} was read from Wan`);
    }
  });

  // The replacement is either a local directory (the real use — an experimental mirror) or mock HF. A
  // local source has no fetch phase (prefetch) and reads the weights while building the Session, so
  // only part 0 is read before the build. The weight fetch can only be observed with the HTTP
  // replacement, which checks that every part was fetched.
  it("replacing a cross-referenced slot gets through admission and the fetch without the original target's mapping (local replacement)", async () => {
    const rig = await prepareWanRepos();
    // No crossRepo at all is passed to the Wan source (the umT5 target is not available locally).
    const wan = memoryDirectory(rig.repos.wan);
    const replacement = memoryDirectory(rig.repos.replacement);

    await assertRejects(
      () =>
        Pipeline.fromPretrained(wan.source, {
          ...BASE_OPTIONS,
          components: { text_encoder: { source: replacement.source } },
        }),
      SafetensorsError,
      STOPPED_AT_EMBEDS,
    );

    // Admission passes by reading the replacement's descriptor, and the Wan source never looked up a
    // path of the target.
    assertEquals(replacement.reads.includes(rig.parts.replacement[0].path), true);
    for (const part of rig.parts.umt5) {
      assertEquals(wan.reads.includes(part.path), false, `${part.path} was read from Wan`);
    }
    for (const ref of rig.assets) {
      assertEquals(wan.reads.includes(ref.path), true, `${ref.path} was not read`);
    }
  });

  it("replacing a cross-referenced slot gets as far as the weight fetch without the original target's mapping (HF replacement)", async () => {
    const rig = await prepareWanRepos();
    const wan = memoryDirectory(rig.repos.wan);
    const http = serveRepos([rig.repos.replacement]);

    await assertRejects(
      () =>
        Pipeline.fromPretrained(wan.source, {
          ...BASE_OPTIONS,
          fetch: http.fetch,
          caches: new MemoryCacheStorage(),
          components: {
            text_encoder: { source: { repo: REPLACEMENT_REPO, revision: SHA, hubUrl: HUB_URL } },
          },
        }),
      SafetensorsError,
      STOPPED_AT_EMBEDS,
    );

    for (const part of nonEmpty(rig.parts.replacement)) {
      assertEquals(http.paths.includes(part.path), true, `${part.path} was not fetched`);
    }
    for (const part of rig.parts.umt5) {
      assertEquals(wan.reads.includes(part.path), false, `${part.path} was read from Wan`);
    }
    for (const ref of rig.assets) {
      assertEquals(wan.reads.includes(ref.path), true, `${ref.path} was not read`);
    }
  });
});
