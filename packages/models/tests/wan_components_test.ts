/**
 * Wan2.1 の umT5（`text_encoder`）の**部品差し替え**（`components` — ADR 0122 決定 7 の 3・ADR 0108
 * 決定 19）を `WanPipeline.fromPretrained` の入口から踏む。GPU も実資産も要らない（疑似 HF のリポと
 * メモリ上のディレクトリを立て、叩かれた (リポ, path) を観測する）。
 *
 * 配布形の骨格は実物（`models/karume-wan2.1`）と同じ: Wan のリポの `text_encoder` は umT5 の配布リポへの
 * **越境参照**（part ごとに `repo` + `revision`）で、差し替え先は別のリポの同じ役割。押さえるのは 4 点:
 *
 * ① **差した `text_encoder` だけが差し替え先のリポから来る** — 元の越境先（umT5 の配布リポ）は
 *    1 バイトも取らない（対照: 差さなければ越境先の part を取る）。
 * ② **手元の配布形（ローカルの取得元）で越境参照の席を差すと、元の越境先の `crossRepo` mapping が
 *    無くても通る**（対照: 差さなければ mapping が無いことで落ちる）。差し替えた席は差し替え先の
 *    取得元から読まれ、元の席の越境参照を引かない（`src/hub/components.ts` の席の出所）。手元の取得元は
 *    取得の相を持たず重みを Session の構築で読むので、重みの取得まで観測するのは差し替え先が HTTP の形。
 * ③ **i4 の差し替え先は、グラフ記述が同じでも Wan の家族の門（`umt5Contract`）が重みの part を取る前に
 *    落とす**（部品差し替え席はグラフ記述の sha256 しか見ないので、格納の門は家族の側にある）。
 * ④ **故障注入: 差し替え先のグラフ記述の sha256 を 1 文字変えると admission が拒む**（容器は 1 本も
 *    取らない）。
 *
 * 観測の仕掛け: 埋め込み資産（`text_embeds`）は safetensors として読めない 8 バイトにしてある。取得面の
 * 構築は admission → 重みの part の取得 → 資産の取得 → 資産の解析の順なので、正の経路は「埋め込み資産の
 * 解析」で止まる（= admission と全部品の重みの取得を通り終えた — `irodori_admission_test.ts` と同じ
 * 「次の段の文言で通過を見る」形）。GPU は共有の模擬 GPU を渡し、アダプタにも触らない。
 */

import {
  assertEquals,
  assertInstanceOf,
  assertNotEquals,
  assertRejects,
  assertStringIncludes,
} from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { SafetensorsError } from "@karume/runtime";
import {
  type DistributionSource,
  type FileRef,
  type HubRepoRef,
  localDirectory,
} from "@karume/hub";
import { type WanFromPretrainedOptions, WanPipeline } from "../src/wan/pipeline.ts";
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

/** Wan の配布形の疑似リポ。 */
const WAN_REPO = "karume-test/wan";
/** Wan の `text_encoder` が越境参照する umT5 の配布リポ（実物の `karume-umt5-xxl` の席）。 */
const UMT5_REPO = "karume-test/umt5-xxl";
/** 越境参照が pin する revision（取得元のセッションの SHA と別の値にして、取り違えを見分ける）。 */
const UMT5_REVISION = "fedcba9876543210fedcba9876543210fedcba98";
/** 差し替え先（互換の umT5 を持つ別リポ）。 */
const REPLACEMENT_REPO = "karume-test/umt5-compatible";

/**
 * 埋め込み資産の解析で止まったことを示す文言（観測の仕掛け — モジュール doc）。投げるのは `SafetensorsError`
 * で、この骨格でほかに safetensors を読むのは `rope_base`（正しい資産）だけなので、正の経路は資産を
 * 取り終えたことのアサーションと組にして埋め込み資産での停止を特定する。
 */
const STOPPED_AT_EMBEDS = "ヘッダ JSON を解析できない";

/** RoPE の素表（`transformer` の容器の資産 `rope_base` — 実物の 1.3B と同じ幅 22 / 21 / 21）。 */
const ROPE_BASE = await Deno.readFile(
  new URL("./fixtures/wan-dit/rope_base.safetensors", import.meta.url),
);

/** DiT の宣言（`wan_pipeline_test.ts` の `ditContract` の正常形と同じ — 実行はしない）。 */
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

/** VAE の chunk グラフの宣言（潜在タイル 32・縮尺 8・cache 1 本 — 配布形と同じ幾何）。 */
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

/** umT5 の語彙埋め込みの形（語彙 2 行 × 幅 4096 — 幅は DiT の文脈の幅）。 */
const EMBED_ROWS = 2;
const EMBED_WIDTH = 4096;

const f32Bytes = (values: readonly number[]): Uint8Array<ArrayBuffer> =>
  new Uint8Array(Float32Array.from(values).buffer);

/** 語彙埋め込みの格納（i8 = per-channel・i4 = 32 要素ごとの group）。`fill` は重みのバイト値。 */
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
 * umT5 の宣言（`umt5Contract` の正常形 — 入力 2 本の i32・記号 L・出力 `[1, L, 4096]`）。グラフの宣言は
 * 格納によらず同じなので、i8 と i4 でグラフ記述はバイト同一になる（格納はモデル記述の束縛表に入る）。
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
      nodes: [{
        op: "embedding",
        ins: ["embed", "input_ids"],
        outs: ["out"],
        attrs: { padding_idx: -1 },
      }],
    }),
  },
  consts: [],
  weights: [embedWeight(storage, fill)],
  assets: [],
  provenance: { license: "test" },
});

/** 1 文字だけ変えた sha256（末尾の 16 進 1 桁を別の値へ）。 */
const flipLastHex = (sha256: string): string =>
  sha256.slice(0, -1) + (sha256.endsWith("0") ? "1" : "0");

/** 配布形の 1 エントリ（`ServedContainer.entry`）のグラフ記述の sha256 だけを差し替える。 */
const withGraphSha = (served: ServedContainer, sha256: string): Record<string, unknown> => {
  const { descriptor } = served.written;
  return {
    container: {
      descriptor: { ...descriptor, graph: { ...descriptor.graph, sha256 } },
      parts: served.parts,
    },
  };
};

/** 越境参照の配布形エントリ（全 part に `repo` + `revision` — 実物の Wan の manifest と同じ綴り）。 */
const crossRepoEntry = (served: ServedContainer): Record<string, unknown> => ({
  container: {
    descriptor: served.written.descriptor,
    parts: served.parts.map((part) => ({ repo: UMT5_REPO, revision: UMT5_REVISION, ...part })),
  },
});

/** 疑似リポ 1 つぶん（`serveRepos` の入力と、ローカルの取得元が読む同じ中身）。 */
type RepoFiles = {
  readonly repo: string;
  readonly models: Record<string, unknown>;
  readonly files: readonly (readonly [string, Uint8Array<ArrayBuffer>])[];
};

/** メモリ上のディレクトリを取得元にする（読んだ path を記録する — ローカルの配布形の席）。 */
const memoryDirectory = (spec: RepoFiles) => {
  const files = new Map<string, Uint8Array<ArrayBuffer>>([
    [MANIFEST_PATH, manifestBytes(spec.models)],
    ...spec.files,
  ]);
  const reads: string[] = [];
  const source: DistributionSource = localDirectory({
    readFile: (path) => {
      reads.push(path);
      const bytes = files.get(path);
      if (bytes === undefined) return Promise.reject(new Error(`${spec.repo}: ${path} が無い`));
      // MUST: buffer 全体を占める view を返す（`DirectoryAdapter.readFile` の契約）。
      return Promise.resolve(bytes.slice());
    },
  }, { label: spec.repo });
  return { source, reads };
};

type WanRigOptions = {
  /** 差し替え先のグラフ記述の sha256 を 1 文字変えて名乗らせる（故障注入 ④）。 */
  readonly corruptReplacementGraphSha?: boolean;
};

/**
 * 3 リポ（Wan・umT5 の越境先・差し替え先）を組む。差し替え先の i8 は元の umT5 と**同じ path** に置き、
 * 重みの値だけを変える — どちらから取ったかはリポの座標でしか見分けられない形にする。
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
  // block は語彙埋め込みの 1 行（i8 で 4096 バイト）が収まる大きさにする。
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
  // 前提: 3 本ともグラフ記述が同じ（差し替え席の受理の条件）で、重みの part は違う。
  for (const other of [replacement, replacementI4]) {
    assertEquals(other.written.descriptor.graph.sha256, umt5.written.descriptor.graph.sha256);
  }
  assertNotEquals(replacement.parts.at(-1)?.sha256, umt5.parts.at(-1)?.sha256);

  const embeds = new Uint8Array(8); // safetensors として読めない（観測の仕掛け — モジュール doc）
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
  const replacementI8Entry = options.corruptReplacementGraphSha === true
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

/** 実体を持つ part（長さ 0 の part は取得の対象外 — hub も取らない）。 */
const nonEmpty = (parts: readonly FileRef[]): readonly FileRef[] =>
  parts.filter((part) => part.size > 0);

/** 構築の共通ノブ（共有の模擬 GPU — アダプタの limits を読みに行かない）。 */
const BASE_OPTIONS: WanFromPretrainedOptions = { gpu: fakeGpuContext(fakeDevice()) };

/** 疑似 HF の 3 リポを立て、Wan のリポから `fromPretrained` する口。 */
const overHttp = (rig: WanRig) => {
  const mock: MockFetch = serveRepos([rig.repos.wan, rig.repos.umt5, rig.repos.replacement]);
  const hubOptions = { fetch: mock.fetch, caches: new MemoryCacheStorage() };
  const replacementSource: HubRepoRef = {
    repo: REPLACEMENT_REPO,
    revision: SHA,
    hubUrl: HUB_URL,
  };
  const load = (options: WanFromPretrainedOptions = {}) =>
    WanPipeline.fromPretrained(
      { repo: WAN_REPO, revision: SHA, hubUrl: HUB_URL },
      { ...BASE_OPTIONS, ...hubOptions, ...options },
    );
  /** そのリポから取った path（宣言順）。 */
  const asked = (repo: string): readonly string[] =>
    mock.requests.filter((request) => request.repo === repo).map((request) => request.path);
  return { load, asked, replacementSource };
};

describe("Wan の text_encoder の差し替え（疑似 HF — 越境参照の席）", () => {
  it("対照: 差さなければ text_encoder は越境先の umT5 のリポから取る", async () => {
    const rig = await prepareWanRepos();
    const http = overHttp(rig);

    await assertRejects(() => http.load(), SafetensorsError, STOPPED_AT_EMBEDS);

    for (const part of nonEmpty(rig.parts.umt5)) {
      assertEquals(http.asked(UMT5_REPO).includes(part.path), true, `${part.path} を取っていない`);
    }
    for (const ref of rig.assets) {
      assertEquals(http.asked(WAN_REPO).includes(ref.path), true, `${ref.path} を取っていない`);
    }
    assertEquals(http.asked(REPLACEMENT_REPO), []);
  });

  it("差した text_encoder だけが差し替え先から来て、元の越境先は 1 バイトも取らない", async () => {
    const rig = await prepareWanRepos();
    const http = overHttp(rig);

    // 止まるのは埋め込み資産の解析 = admission と全部品の重みの取得を通り終えた後。
    await assertRejects(
      () => http.load({ components: { text_encoder: { source: http.replacementSource } } }),
      SafetensorsError,
      STOPPED_AT_EMBEDS,
    );

    // 元の越境先には 1 度も取りに行っていない（part 0 の descriptor も含めて）。
    assertEquals(http.asked(UMT5_REPO), []);
    // 差し替え先から来たのは manifest と text_encoder の part だけ（i4 の席や他の役割は取らない）。
    assertEquals(
      [...http.asked(REPLACEMENT_REPO)].sort(),
      [MANIFEST_PATH, ...nonEmpty(rig.parts.replacement).map((part) => part.path)].sort(),
    );
    // 差していない役割と資産は Wan のリポのまま取っている（text_encoder の path は Wan のリポに無い）。
    for (const ref of [...nonEmpty(rig.parts.transformer), ...nonEmpty(rig.parts.vae)]) {
      assertEquals(http.asked(WAN_REPO).includes(ref.path), true, `${ref.path} を取っていない`);
    }
    for (const ref of rig.assets) {
      assertEquals(http.asked(WAN_REPO).includes(ref.path), true, `${ref.path} を取っていない`);
    }
    for (const part of rig.parts.replacement) {
      assertEquals(
        http.asked(WAN_REPO).includes(part.path),
        false,
        `${part.path} を Wan から取った`,
      );
    }
  });

  it("i4 の差し替え先は umt5Contract が重みの part を取る前に落とす（グラフ記述は同じでも）", async () => {
    const rig = await prepareWanRepos();
    const http = overHttp(rig);

    const error = await assertRejects(
      () =>
        http.load({
          components: { text_encoder: { source: http.replacementSource, quant: "i4" } },
        }),
      Error,
      "text_encoder の initializer 'embed' の格納 int4-sym-g は受けない",
    );
    assertStringIncludes(error.message, "ADR 0119 決定 5");

    // descriptor（part 0）は admission のために取ったが、重みの part は 1 本も取っていない。
    const [descriptor, ...weightParts] = rig.parts.replacementI4;
    assertEquals(http.asked(REPLACEMENT_REPO).includes(descriptor.path), true);
    for (const part of nonEmpty(weightParts)) {
      assertEquals(
        http.asked(REPLACEMENT_REPO).includes(part.path),
        false,
        `${part.path} を取った`,
      );
    }
    assertEquals(nonEmpty(weightParts).length > 0, true, "重みの part を持つ i4 で観測していない");
    // 資産の段にも進んでいない・元の越境先にも行っていない。
    for (const ref of rig.assets) {
      assertEquals(http.asked(WAN_REPO).includes(ref.path), false, `${ref.path} を取った`);
    }
    assertEquals(http.asked(UMT5_REPO), []);
  });

  it("故障注入: 差し替え先のグラフ記述の sha256 を 1 文字変えると admission が拒み、容器を 1 本も取らない", async () => {
    const rig = await prepareWanRepos({ corruptReplacementGraphSha: true });
    const http = overHttp(rig);

    const error = await assertRejects(
      () => http.load({ components: { text_encoder: { source: http.replacementSource } } }),
      Error,
      "グラフ記述が manifest の宣言と違う",
    );
    assertStringIncludes(error.message, "components['text_encoder']");
    // 宣言だけで判る — どのリポの `.krm` も取っていない（差し替え先は manifest だけ）。
    assertEquals(http.asked(REPLACEMENT_REPO), [MANIFEST_PATH]);
    assertEquals(http.asked(UMT5_REPO), []);
    assertEquals(http.asked(WAN_REPO), [MANIFEST_PATH]);
  });
});

describe("Wan の text_encoder の差し替え（手元の配布形 — 越境先の mapping なし）", () => {
  it("対照: 差さなければ越境先の crossRepo mapping が無いことで落ちる", async () => {
    const rig = await prepareWanRepos();
    const wan = memoryDirectory(rig.repos.wan);

    const error = await assertRejects(
      () => WanPipeline.fromPretrained(wan.source, BASE_OPTIONS),
      Error,
      `${rig.parts.umt5[0].path} の事前取得に失敗した（repo ${UMT5_REPO} @ ${UMT5_REVISION}）`,
    );
    // 取得層は取得元の案内を cause に残す（hub の local.ts — 隣のディレクトリを推測しない）。
    assertInstanceOf(error.cause, Error);
    assertStringIncludes(error.cause.message, `repo '${UMT5_REPO}' の越境先が無い`);
    assertStringIncludes(error.cause.message, "crossRepo");
    // 越境先の part は Wan の取得元から読もうとしていない（隣を推測しない）。
    for (const part of rig.parts.umt5) {
      assertEquals(wan.reads.includes(part.path), false, `${part.path} を Wan から読んだ`);
    }
  });

  // 差し替え先は手元のディレクトリ（実際の使い方 — 実験用ミラー）と疑似 HF の 2 通り。手元の取得元は
  // 取得の相（prefetch）を持たず重みを Session の構築で読むので、構築の前に読まれるのは part 0 だけ。
  // 重みの取得まで観測できるのは HTTP の差し替え先で、そちらは全 part を取ったことまで見る。
  it("越境参照の席を差すと、元の越境先の mapping 無しで admission と取得を通る（差し替え先が手元）", async () => {
    const rig = await prepareWanRepos();
    // Wan の取得元に crossRepo を 1 本も渡さない（umT5 の越境先は手元に無い）。
    const wan = memoryDirectory(rig.repos.wan);
    const replacement = memoryDirectory(rig.repos.replacement);

    await assertRejects(
      () =>
        WanPipeline.fromPretrained(wan.source, {
          ...BASE_OPTIONS,
          components: { text_encoder: { source: replacement.source } },
        }),
      SafetensorsError,
      STOPPED_AT_EMBEDS,
    );

    // 差し替え先の descriptor を読んで admission を通し、Wan の取得元は越境先の path を 1 度も引いていない。
    assertEquals(replacement.reads.includes(rig.parts.replacement[0].path), true);
    for (const part of rig.parts.umt5) {
      assertEquals(wan.reads.includes(part.path), false, `${part.path} を Wan から読んだ`);
    }
    for (const ref of rig.assets) {
      assertEquals(wan.reads.includes(ref.path), true, `${ref.path} を読んでいない`);
    }
  });

  it("越境参照の席を差すと、元の越境先の mapping 無しで重みの取得まで通る（差し替え先が HF）", async () => {
    const rig = await prepareWanRepos();
    const wan = memoryDirectory(rig.repos.wan);
    const http = serveRepos([rig.repos.replacement]);

    await assertRejects(
      () =>
        WanPipeline.fromPretrained(wan.source, {
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
      assertEquals(http.paths.includes(part.path), true, `${part.path} を取っていない`);
    }
    for (const part of rig.parts.umt5) {
      assertEquals(wan.reads.includes(part.path), false, `${part.path} を Wan から読んだ`);
    }
    for (const ref of rig.assets) {
      assertEquals(wan.reads.includes(ref.path), true, `${ref.path} を読んでいない`);
    }
  });
});
