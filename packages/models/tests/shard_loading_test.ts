/**
 * shard 面のロード経路（`src/hub/components.ts`）の門 — 取得層だけを差し替えて 4 点を固定する:
 *
 * ① **進捗はモデル全体で 1 本のまま**（取得が「グラフ shard の逐次面 + 残り資産の全量面」へ
 *    割れても `loaded` は単調増加で全ファイルの size 合計に着地し、`complete` はファイル数ぶん）。
 * ② **admission は重み shard を取る前に落ちる**（グラフ shard だけで capability 違反が決まり、
 *    重み shard の URL は 1 度も叩かれない — ADR 0070 決定 5 の存在理由そのもの）。この門は
 *    prefetch が admission の**後**に置かれていることの門でもある。
 * ③ **重み shard はロード時に落ち切る**（Session を 1 本も張らないうちに DL 済み — 遅延構築の
 *    家族で「初回実行まで DL が遅れ、ロード進捗にも現れない」形を無くす）。
 * ④ **Session 構築は進捗を動かさない**（prefetch 済みキャッシュの読み直しで `complete` が
 *    もう一度飛ぶと、集約 `loaded` が二重計上になる）。④ だけは実 GPU が要る。
 * ⑤ **ロード時の `signal` は Session 構築へ持ち越さない**（`AbortSignal.timeout` や画面の
 *    アンマウントで「ロードは成功したのに以後の生成が全部落ちる」形を作らない）。対で
 *    「abort 済みで始めたロードは落ちる」も置き、ロード中の中断契約が消えていないことを見る。
 * ⑥ **グラフ shard のバイト列を握らない**（コンポーネントの供給口を保持したままでも常駐
 *    ゼロ）。gc を強制する必要があるので別プロセスの台本
 *    （`helpers/graph-shard-retention.ts`）へ出し、ここでは終了コードだけを見る。
 * ⑦ **家族 admission の違反でも重み shard は取得されない**（②の家族版）。② は runtime の
 *    capability 門だけを踏むので、「実行できないモデルの重みは 1 バイトも落とさない」の
 *    うち家族側（pipeline major / `pipelineConfig`）が前段に居ることは縛れない。実家族
 *    （siglip2 = コンポーネント 1 本の最小形）を `fromPretrained` で通して同じ観測法で見る。
 * ⑧ **`requiredLimits` 超過でも重み shard は取得されない（共有 GPU）**（⑦の limits 版 —
 *    ADR 0089 決定 5）。突き合わせ相手は渡された `GpuContext.limits`。
 * ⑨ **同じことが自前 GPU 取得の経路でも成り立つ**（突き合わせ相手が `readAdapterLimits()` の
 *    アダプタ実測値に変わる側）。⑨ だけは実 GPU が要る。
 * ⑩ **取得キーの失敗診断と受理集合**（コンポーネント欠落 / 未取得キー / 素キーと `[i]` の混在 /
 *    添字が `[0]` から始まらない）。受理集合は全量面と同じ 1 本から来る。
 * ⑪ **遅延資産**（`eagerAssets` / `deferred` / `readCachedAsset`）— gemma4 の PLE sidecar の
 *    経路で、実行者が実重み e2e しか無いと CI では 1 度も踏まれない。
 *
 * NOTE: hub / runtime のテスト helper は import しない（向こうの都合がこちらへ漏れる —
 * `helpers/memory-cache.ts` と同じ規律）。モックはこのファイル内で最小限だけ組む。**唯一の
 * 例外が⑧の `fake-gpu.ts`** — `GpuContext` は runtime が値として公開しない（`acquireGpu` が
 * 唯一の入口 — ADR 0008）ので、共有 GPU 経路を GPU 無し環境で踏むには向こうの実物を包む
 * helper が要る。ここで自前に偽物を組むと、検査対象そのもの（limits を持つ GpuContext）が
 * 偽物になる。
 */

import { assertEquals, assertRejects, assertStringIncludes, assertThrows } from "@std/assert";
import { type AssetProgress, loadManifest, resolveFiles } from "@karume/hub";
import { acquireGpu } from "@karume/runtime";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";
import { loadShardComponents, readCachedAsset } from "../src/hub/components.ts";
import { Siglip2Pipeline } from "../src/siglip2/pipeline.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { MemoryCacheStorage } from "./helpers/memory-cache.ts";
import { type DumpTensor, writeSafetensors } from "./helpers/safetensors-write.ts";

/**
 * 家族 admission の席（この機構だけを見るテストは家族の門を 1 つも置かない）。実家族の門が
 * 席に載っていることは ⑦ が別に縛る。
 */
const NO_FAMILY_GATE = (): undefined => undefined;

const HUB_URL = "https://hub.test";
const REPO = "karume-test/shards";
const SHA = "0123456789abcdef0123456789abcdef01234567";
const MANIFEST_PATH = "karume.json";

const REVISION_RE = /^\/api\/models\/(.+)\/revision\/(.+)$/;
const RESOLVE_RE = /^\/(.+?)\/resolve\/([^/]+)\/(.+)$/;

/** HF の 2 経路（revision 解決 API・resolve URL）だけを喋る `fetch`。叩かれた path を記録する。 */
const createMockFetch = (
  files: ReadonlyMap<string, Uint8Array<ArrayBuffer>>,
): { fetch: typeof globalThis.fetch; paths: string[] } => {
  const paths: string[] = [];
  const fetch: typeof globalThis.fetch = (input) => {
    const href = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const { pathname } = new URL(href);
    if (REVISION_RE.test(pathname)) return Promise.resolve(Response.json({ sha: SHA }));
    const resolved = RESOLVE_RE.exec(pathname);
    if (resolved === null) return Promise.resolve(new Response("not found", { status: 404 }));
    const path = decodeURIComponent(resolved[3]);
    paths.push(path);
    const bytes = files.get(path);
    if (bytes === undefined) return Promise.resolve(new Response("not found", { status: 404 }));
    return Promise.resolve(
      new Response(bytes, { headers: { "content-length": String(bytes.byteLength) } }),
    );
  };
  return { fetch, paths };
};

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

/** manifest の 3 点セット（path / size / sha256）を現物から作る。 */
const fileRef = async (
  path: string,
  bytes: Uint8Array<ArrayBuffer>,
): Promise<{ path: string; size: number; sha256: string }> => ({
  path,
  size: bytes.byteLength,
  sha256: await sha256Hex(bytes),
});

const f32Tensor = (shape: readonly number[], value: number): DumpTensor => ({
  dtype: "F32",
  shape: [...shape],
  data: new Float32Array(shape.reduce((product, dim) => product * dim, 1)).fill(value),
});

/**
 * 最小の IR グラフ 1 本（`linear` 1 段）。`op` を差し替えると「実行できないグラフ」になる
 * （IR パーサは op 名の綴りを見ないので、落ちるのは capability 門 = admission 相）。
 */
const miniGraph = (op: string): unknown => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: [op] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [2, 2] }],
  outputs: ["y"],
  initializers: {
    w: { tensor: "m.w", storage: { dtype: "f32" } },
    b: { tensor: "m.b", storage: { dtype: "f32" } },
  },
  values: {
    w: { dtype: "f32", shape: [2, 2] },
    b: { dtype: "f32", shape: [2] },
    y: { dtype: "f32", shape: [2, 2] },
  },
  states: {},
  nodes: [{ op, ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
});

/** グラフ shard（`karume_ir` + 同居テンソル）1 本を焼く。 */
const graphShardBytes = (
  op: string,
  tensors: readonly (readonly [string, DumpTensor])[],
): Uint8Array<ArrayBuffer> =>
  writeSafetensors(new Map(tensors), { karume_ir: JSON.stringify(miniGraph(op)) });

/** 重み shard（`karume_ir` を持たない）1 本を焼く。 */
const weightShardBytes = (
  tensors: readonly (readonly [string, DumpTensor])[],
): Uint8Array<ArrayBuffer> => writeSafetensors(new Map(tensors), {});

const manifestBytes = (models: unknown): Uint8Array<ArrayBuffer> =>
  new TextEncoder().encode(
    JSON.stringify({
      format: "karume/4",
      generator: "karume-test/0",
      defaultModel: "test",
      models,
    }),
  );

Deno.test(
  "loadShardComponents: 複数コンポーネントでも進捗はモデル全体で 1 本（単調増加・合計へ着地）",
  async () => {
    // 1 shard のコンポーネント 2 本 + 非 safetensors の資産 1 本（= 残りは全量面で取る）。
    const front = graphShardBytes("linear", [
      ["m.w", f32Tensor([2, 2], 0.5)],
      ["m.b", f32Tensor([2], 0.25)],
    ]);
    const voice = graphShardBytes("linear", [
      ["m.w", f32Tensor([2, 2], 1.5)],
      ["m.b", f32Tensor([2], 0.75)],
    ]);
    const tokenizer = new TextEncoder().encode(JSON.stringify({ vocab: ["a", "b"] }));

    const refs = {
      front: await fileRef("front/model.safetensors", front),
      voice: await fileRef("voice/model.safetensors", voice),
      tokenizer: await fileRef("tokenizer/tokenizer.json", tokenizer),
    };
    const manifest = manifestBytes({
      test: {
        pipeline: "test/1",
        weights: {
          front: { f32: { shards: [refs.front] } },
          voice: { f32: { shards: [refs.voice] } },
        },
        assets: { tokenizer: refs.tokenizer },
        quants: { f32: { weights: { front: "f32", voice: "f32" }, session: {} } },
        defaultQuant: "f32",
        pipelineConfig: {},
      },
    });

    const mock = createMockFetch(
      new Map([
        [MANIFEST_PATH, manifest],
        [refs.front.path, front],
        [refs.voice.path, voice],
        [refs.tokenizer.path, tokenizer],
      ]),
    );
    const caches = new MemoryCacheStorage();
    const hubOptions = { fetch: mock.fetch, caches };
    const loaded = await loadManifest({ repo: REPO, revision: SHA, hubUrl: HUB_URL }, hubOptions);
    const files = resolveFiles(loaded.manifest);

    const events: AssetProgress[] = [];
    const { open, assets } = await loadShardComponents(
      "test.fromPretrained",
      loaded,
      files,
      ["front", "voice"],
      NO_FAMILY_GATE,
      { ...hubOptions, onProgress: (progress) => events.push(progress) },
    );

    // admission を通ったこと（グラフ宣言が読める = prepareModel が成功した）。
    assertEquals(open("front").graph.outputs, ["y"]);
    assertEquals(open("voice").graph.outputs, ["y"]);
    // 残り資産は全量面のまま届く。
    assertEquals(Object.keys(assets), ["tokenizer"]);

    const total = refs.front.size + refs.voice.size + refs.tokenizer.size;
    // total は全イベントで**モデル全体の合計**の 1 値（取得の割れ方に依らない）。
    assertEquals(new Set(events.map((event) => event.total)), new Set([total]));
    // loaded は単調増加で、最後は合計に一致する。
    let previous = 0;
    for (const event of events) {
      if (event.loaded < previous) {
        throw new Error(
          `loaded が巻き戻った: ${previous} → ${event.loaded}（${event.phase} ${event.path}）`,
        );
      }
      previous = event.loaded;
    }
    assertEquals(events[events.length - 1].loaded, total);
    // complete はファイル 1 本につき 1 回（per-file 欄も取得層の契約どおり fileTotal で揃う）。
    const completes = events.filter((event) => event.phase === "complete");
    assertEquals(completes.length, 3);
    assertEquals(
      new Set(completes.map((event) => event.path)),
      new Set([refs.front.path, refs.voice.path, refs.tokenizer.path]),
    );
    for (const event of completes) assertEquals(event.fileLoaded, event.fileTotal);
  },
);

Deno.test(
  "loadShardComponents: capability 違反はグラフ shard だけで落ち、重み shard は取得されない",
  async () => {
    // グラフ shard（非対応 op）+ 重み shard の 2 本構成。
    const graph = graphShardBytes("karume_test_unsupported_op", [["m.b", f32Tensor([2], 0.25)]]);
    const weights = weightShardBytes([["m.w", f32Tensor([2, 2], 0.5)]]);
    const refs = {
      graph: await fileRef("dit/model-00000.safetensors", graph),
      weights: await fileRef("dit/model-00001.safetensors", weights),
    };
    const manifest = manifestBytes({
      test: {
        pipeline: "test/1",
        weights: { dit: { f32: { shards: [refs.graph, refs.weights] } } },
        assets: {},
        quants: { f32: { weights: { dit: "f32" }, session: {} } },
        defaultQuant: "f32",
        pipelineConfig: {},
      },
    });

    const mock = createMockFetch(
      new Map([
        [MANIFEST_PATH, manifest],
        [refs.graph.path, graph],
        [refs.weights.path, weights],
      ]),
    );
    const caches = new MemoryCacheStorage();
    const hubOptions = { fetch: mock.fetch, caches };
    const loaded = await loadManifest({ repo: REPO, revision: SHA, hubUrl: HUB_URL }, hubOptions);
    const files = resolveFiles(loaded.manifest);

    const error = await assertRejects(
      () =>
        loadShardComponents(
          "test.fromPretrained",
          loaded,
          files,
          ["dit"],
          NO_FAMILY_GATE,
          hubOptions,
        ),
      Error,
    );
    // 落ちた理由が capability 門であること（別の失敗で「重みを取らなかった」が成立しない）。
    if (!error.message.includes("karume_test_unsupported_op")) {
      throw new Error(`capability 門の文言でない: ${error.message}`);
    }
    // 重み shard の URL は 1 度も叩かれていない（= 重み DL 前 admission。ロード時 prefetch が
    // admission の後に置かれていることの門でもある — 前に出ると落ちるモデルの重みまで落ちる）。
    assertEquals(mock.paths.includes(refs.weights.path), false);
    assertEquals(mock.paths.includes(refs.graph.path), true);
  },
);

/**
 * 実行可能な 1 コンポーネント（`linear` 1 段）を **グラフ shard + 重み shard の 2 本**へ割った
 * 配布形を組み、manifest まで解決して返す。重み `m.w` は 2 本目にしか無いので、Session が
 * 張れること自体が「重み shard が届いている」ことの証拠になる。
 */
const prepareTwoShard = async () => {
  const graph = graphShardBytes("linear", [["m.b", f32Tensor([2], 0.25)]]);
  const weights = weightShardBytes([["m.w", f32Tensor([2, 2], 0.5)]]);
  const refs = {
    graph: await fileRef("dit/model-00000.safetensors", graph),
    weights: await fileRef("dit/model-00001.safetensors", weights),
  };
  const manifest = manifestBytes({
    test: {
      pipeline: "test/1",
      weights: { dit: { f32: { shards: [refs.graph, refs.weights] } } },
      assets: {},
      quants: { f32: { weights: { dit: "f32" }, session: {} } },
      defaultQuant: "f32",
      pipelineConfig: {},
    },
  });
  const mock = createMockFetch(
    new Map([
      [MANIFEST_PATH, manifest],
      [refs.graph.path, graph],
      [refs.weights.path, weights],
    ]),
  );
  const caches = new MemoryCacheStorage();
  const hubOptions = { fetch: mock.fetch, caches };
  const loaded = await loadManifest({ repo: REPO, revision: SHA, hubUrl: HUB_URL }, hubOptions);
  return { loaded, files: resolveFiles(loaded.manifest), refs, mock, hubOptions };
};

Deno.test(
  "loadShardComponents: 重み shard は Session を張る前（ロード時）に落ち切り、進捗にも現れる",
  async () => {
    const { loaded, files, refs, mock, hubOptions } = await prepareTwoShard();

    const events: AssetProgress[] = [];
    await loadShardComponents("test.fromPretrained", loaded, files, ["dit"], NO_FAMILY_GATE, {
      ...hubOptions,
      onProgress: (progress) => events.push(progress),
    });

    // Session は 1 本も張っていない。それでも重み shard の URL は叩かれている。
    assertEquals(mock.paths.includes(refs.weights.path), true, "重み shard がロード時に落ちない");
    assertEquals(mock.paths.includes(refs.graph.path), true);
    // 進捗にも 2 本ぶんが乗る（遅延構築の家族でも「ロード = 全 DL」の表示が成立する）。
    const completes = events.filter((event) => event.phase === "complete");
    assertEquals(
      new Set(completes.map((event) => event.path)),
      new Set([refs.graph.path, refs.weights.path]),
    );
    assertEquals(events[events.length - 1].loaded, refs.graph.size + refs.weights.size);
  },
);

Deno.test({
  name: "loadShardComponents: Session 構築は進捗を 1 イベントも動かさない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const { loaded, files, mock, hubOptions } = await prepareTwoShard();

    const events: AssetProgress[] = [];
    const { open } = await loadShardComponents(
      "test.fromPretrained",
      loaded,
      files,
      ["dit"],
      NO_FAMILY_GATE,
      { ...hubOptions, onProgress: (progress) => events.push(progress) },
    );
    const afterLoad = events.length;
    const calls = mock.paths.length;
    // complete はファイル数ぶんちょうど（ロードを抜けた時点で全ファイルが終端に達している）。
    assertEquals(events.filter((event) => event.phase === "complete").length, 2);

    const gpu = await acquireGpu();
    try {
      // 重み shard は 2 本目にしかないので、Session が張れた時点で shard 列は流れている。
      const session = await open("dit").createSession(gpu);
      await session.dispose();
    } finally {
      gpu.destroy();
    }

    assertEquals(events.length, afterLoad, "Session 構築が進捗イベントを追加している");
    assertEquals(mock.paths.length, calls, "Session 構築が network へ出ている");
  },
});

Deno.test({
  name: "loadShardComponents: 同じ供給口から Session を 2 本続けて張れる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const { loaded, files, hubOptions } = await prepareTwoShard();
    const { open } = await loadShardComponents(
      "test.fromPretrained",
      loaded,
      files,
      ["dit"],
      NO_FAMILY_GATE,
      hubOptions,
    );

    const gpu = await acquireGpu();
    try {
      // 2 本目が張れるのは、shard 列を**呼ぶたびに**新しく作っているとき（使い切った列を
      // 使い回すと 2 本目が空の列を受けて「重みが足りない」で落ちる）。グラフ shard も列に
      // 含むようになった後も同じ規律が要る。
      for (let index = 0; index < 2; index += 1) {
        const session = await open("dit").createSession(gpu);
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "loadShardComponents: ロード時の signal は Session 構築へ持ち越さない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const { loaded, files, hubOptions } = await prepareTwoShard();
    const controller = new AbortController();
    const { open } = await loadShardComponents(
      "test.fromPretrained",
      loaded,
      files,
      ["dit"],
      NO_FAMILY_GATE,
      { ...hubOptions, signal: controller.signal },
    );

    // 呼び手の中断ノブは「このロード 1 回」の寿命のもの — ロード成功の後に発火するのは
    // `AbortSignal.timeout` でも画面のアンマウントでもごく普通の綴り。
    controller.abort();

    const gpu = await acquireGpu();
    try {
      // 持ち越していると相 2 の `throwIfAborted()` で落ちる（キャッシュ完備でも確実に）。
      const session = await open("dit").createSession(gpu);
      await session.dispose();
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test(
  "loadShardComponents: abort 済みの signal で始めたロードは落ちる（中断契約の維持）",
  async () => {
    const { loaded, files, hubOptions } = await prepareTwoShard();
    const reason = new Error("test: ロード開始前に中断済み");

    const error = await assertRejects(
      () =>
        loadShardComponents("test.fromPretrained", loaded, files, ["dit"], NO_FAMILY_GATE, {
          ...hubOptions,
          signal: AbortSignal.abort(reason),
        }),
    );
    assertEquals(error, reason);
  },
);

/** `models/karume-siglip2-base/karume.json` の `pipelineConfig` 実物（6 欄）。 */
const SIGLIP2_CONFIG: Record<string, unknown> = {
  imageWidth: 224,
  imageHeight: 224,
  imageMean: [0.5, 0.5, 0.5],
  imageStd: [0.5, 0.5, 0.5],
  hiddenDim: 768,
  interpolation: "bilinear",
};

const CHANNELS = 3;
const IMAGE_SIZE = 224;
const HIDDEN_DIM = 768;

/**
 * siglip2 の家族 admission が読むグラフ**宣言**（入出力の名前と形）まで再現した最小の IR。
 *
 * MUST: `admitSiglip2` の構造検査（`pixel_values` の 4 軸 / 出力の `hiddenDim`）を**通る**形で
 * 綴る — ここで落ちると、その後段に居る門（quant / GPU 前提）を踏んだことにならない。実行は
 * しない（Session を張るテストはこの rig を使わない）ので、`linear` の内側の整合は問わない。
 */
const siglip2Graph = (): unknown => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["linear"] },
  symbols: [],
  inputs: [{ name: "pixel_values", dtype: "f32", shape: [1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE] }],
  outputs: ["pooler_output"],
  initializers: {
    w: { tensor: "m.w", storage: { dtype: "f32" } },
    b: { tensor: "m.b", storage: { dtype: "f32" } },
  },
  values: {
    w: { dtype: "f32", shape: [HIDDEN_DIM, IMAGE_SIZE] },
    b: { dtype: "f32", shape: [HIDDEN_DIM] },
    pooler_output: { dtype: "f32", shape: [1, HIDDEN_DIM] },
  },
  states: {},
  nodes: [{ op: "linear", ins: ["pixel_values", "w", "b"], outs: ["pooler_output"], attrs: {} }],
});

/**
 * siglip2 の配布形（グラフ shard + 重み shard の 2 本）を疑似 HF に載せる。`patch` で
 * `models["test"]` の欄を差し替えて**家族 admission だけ**が落ちる形を作る。
 *
 * MUST: グラフは実行可能な `linear` 1 段のまま — 非対応 op にすると runtime の capability 門
 * （②が既に縛る側）で落ちてしまい、家族の門を通ったことの証明にならない。
 */
const prepareSiglip2 = async (patch: Record<string, unknown>) => {
  const graph = writeSafetensors(
    new Map([["m.b", f32Tensor([HIDDEN_DIM], 0.25)]]),
    { karume_ir: JSON.stringify(siglip2Graph()) },
  );
  const weights = weightShardBytes([["m.w", f32Tensor([HIDDEN_DIM, IMAGE_SIZE], 0.5)]]);
  const refs = {
    graph: await fileRef("vision/model-00000.safetensors", graph),
    weights: await fileRef("vision/model-00001.safetensors", weights),
  };
  const manifest = manifestBytes({
    test: {
      pipeline: "siglip2/1",
      weights: { vision: { f32: { shards: [refs.graph, refs.weights] } } },
      assets: {},
      quants: { f32: { weights: { vision: "f32" }, session: {} } },
      defaultQuant: "f32",
      pipelineConfig: SIGLIP2_CONFIG,
      ...patch,
    },
  });
  const mock = createMockFetch(
    new Map([
      [MANIFEST_PATH, manifest],
      [refs.graph.path, graph],
      [refs.weights.path, weights],
    ]),
  );
  return { refs, mock, caches: new MemoryCacheStorage() };
};

Deno.test(
  "家族 admission（pipeline major 不一致）はグラフ shard だけで落ち、重み shard は取得されない",
  async () => {
    const { refs, mock, caches } = await prepareSiglip2({ pipeline: "siglip2/99" });

    const error = await assertRejects(
      () =>
        Siglip2Pipeline.fromPretrained(
          { repo: REPO, revision: SHA, hubUrl: HUB_URL },
          { fetch: mock.fetch, caches },
        ),
      Error,
    );
    // 落ちた理由が家族の major 門であること（別の失敗で「重みを取らなかった」が成立しない）。
    if (!error.message.includes("major に未対応")) {
      throw new Error(`家族 admission の文言でない: ${error.message}`);
    }
    // グラフ shard は取りに行き、重み shard は 1 度も叩いていない（ADR 0070 決定 5 の文面
    // 「実行できないモデルの重みは 1 バイトも落とさない」が家族の門にも及んでいる）。
    assertEquals(mock.paths.includes(refs.graph.path), true);
    assertEquals(mock.paths.includes(refs.weights.path), false);
  },
);

Deno.test(
  "家族 admission（pipelineConfig の schema 違反）でも重み shard は取得されない",
  async () => {
    const { refs, mock, caches } = await prepareSiglip2({
      pipelineConfig: { ...SIGLIP2_CONFIG, karumeUnknownKey: 1 },
    });

    const error = await assertRejects(
      () =>
        Siglip2Pipeline.fromPretrained(
          { repo: REPO, revision: SHA, hubUrl: HUB_URL },
          { fetch: mock.fetch, caches },
        ),
      Error,
    );
    if (!error.message.includes("karumeUnknownKey")) {
      throw new Error(`pipelineConfig の門の文言でない: ${error.message}`);
    }
    assertEquals(mock.paths.includes(refs.graph.path), true);
    assertEquals(mock.paths.includes(refs.weights.path), false);
  },
);

/** ⑧⑨で使う quant 欄（`requiredLimits` だけが違う 2 通り）。 */
const quantsRequiring = (maxBufferSize: number): Record<string, unknown> => ({
  f32: { weights: { vision: "f32" }, session: {}, requiredLimits: { maxBufferSize } },
});

Deno.test(
  "家族 admission（requiredLimits 超過）でも重み shard は取得されない（共有 GPU）",
  async () => {
    // 全 limit が 0 の GpuContext（`fake-gpu.ts` の ZERO_LIMITS）へ、1 バイトでも要求する
    // 配布形を渡す。共有 GPU は取り直せない（feature も limits も device 生成時の話）ので、
    // 落とせる唯一の場所が admission 席になる。
    const { refs, mock, caches } = await prepareSiglip2({ quants: quantsRequiring(1) });

    const error = await assertRejects(
      () =>
        Siglip2Pipeline.fromPretrained(
          { repo: REPO, revision: SHA, hubUrl: HUB_URL },
          { fetch: mock.fetch, caches, gpu: fakeGpuContext(fakeDevice()) },
        ),
      Error,
    );
    // 落ちた理由が limits 検査であること（別の失敗で「重みを取らなかった」が成立しない）。
    if (!error.message.includes("maxBufferSize")) {
      throw new Error(`limits 門の文言でない: ${error.message}`);
    }
    assertEquals(mock.paths.includes(refs.graph.path), true);
    assertEquals(mock.paths.includes(refs.weights.path), false);
  },
);

Deno.test({
  name: "requiredLimits 超過は自前 GPU 取得の経路でも重み shard を取らない（アダプタ実測）",
  // MUST: アダプタ無し環境は明示 SKIP（ADR 0005）。この経路の突き合わせ相手は
  // `readAdapterLimits()` なので、アダプタが無いと GpuUnavailableError に化けて門が見えない。
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // どの実機も満たせない要求（`Number.MAX_SAFE_INTEGER` — manifest が受ける最大値）。
    const { refs, mock, caches } = await prepareSiglip2({
      quants: quantsRequiring(Number.MAX_SAFE_INTEGER),
    });

    const error = await assertRejects(
      () =>
        Siglip2Pipeline.fromPretrained(
          { repo: REPO, revision: SHA, hubUrl: HUB_URL },
          { fetch: mock.fetch, caches },
        ),
      Error,
    );
    if (!error.message.includes("maxBufferSize")) {
      throw new Error(`limits 門の文言でない: ${error.message}`);
    }
    assertEquals(mock.paths.includes(refs.graph.path), true);
    assertEquals(mock.paths.includes(refs.weights.path), false);
  },
});

Deno.test(
  "loadShardComponents: グラフ shard のバイト列を握らない（別プロセスで gc 観測）",
  async () => {
    // MUST: 別プロセス — 到達不能なだけの状態と握られた状態を区別するには gc の強制が要り、
    // `deno test` に `--v8-flags` を渡す口が無い。
    const script = new URL("./helpers/graph-shard-retention.ts", import.meta.url);
    const { code, stdout, stderr } = await new Deno.Command(Deno.execPath(), {
      args: ["run", "-A", "--v8-flags=--expose-gc", script.href],
      stdout: "piped",
      stderr: "piped",
    }).output();
    const decoder = new TextDecoder();
    assertEquals(
      code,
      0,
      `${decoder.decode(stdout)}${decoder.decode(stderr)}`,
    );
  },
);

// ---- 取得キーの失敗診断と受理集合（⑩）--------------------------------------
//
// `componentShards` / `open()` の 2 本の診断と、shard 面の受理集合が全量面
// （`asset_shard_components_test.ts`）と**同じ 1 本**（`planComponentKeys`）から来ていること。
// 素キー優先で `[i]` を読み飛ばす形だと、残ったキーは `consumed` に入らず資産として全量取得
// され、Session の shard 列からは消える（= 重みが黙って 1 本欠ける）。

/** `prepareTwoShard` の manifest を、weights / assets のキーだけ差し替えて組み直す。 */
const prepareKeyed = async (
  weights: (refs: { graph: unknown; weights: unknown }) => Record<string, unknown>,
  assets: (refs: { graph: unknown; weights: unknown }) => Record<string, unknown> = () => ({}),
) => {
  const graph = graphShardBytes("linear", [["m.b", f32Tensor([2], 0.25)]]);
  const weightBytes = weightShardBytes([["m.w", f32Tensor([2, 2], 0.5)]]);
  const refs = {
    graph: await fileRef("dit/model-00000.safetensors", graph),
    weights: await fileRef("dit/model-00001.safetensors", weightBytes),
  };
  const declared = weights(refs);
  const manifest = manifestBytes({
    test: {
      pipeline: "test/1",
      weights: declared,
      assets: assets(refs),
      // quant の weights 写像は宣言したコンポーネント名から導く（名前を差し替えるたびに
      // ここを直すと、テストが見たい形と manifest の整合が別々にずれる）。
      quants: {
        f32: {
          weights: Object.fromEntries(Object.keys(declared).map((key) => [key, "f32"])),
          session: {},
        },
      },
      defaultQuant: "f32",
      pipelineConfig: {},
    },
  });
  const mock = createMockFetch(
    new Map([
      [MANIFEST_PATH, manifest],
      [refs.graph.path, graph],
      [refs.weights.path, weightBytes],
    ]),
  );
  const hubOptions = { fetch: mock.fetch, caches: new MemoryCacheStorage() };
  const loaded = await loadManifest({ repo: REPO, revision: SHA, hubUrl: HUB_URL }, hubOptions);
  return { loaded, files: resolveFiles(loaded.manifest), refs, mock, hubOptions };
};

Deno.test(
  "loadShardComponents: manifest が持たないコンポーネント名は取得キー一覧つきで落ちる",
  async () => {
    const { loaded, files, hubOptions } = await prepareTwoShard();
    const error = await assertRejects(
      () =>
        loadShardComponents("test.fromPretrained", loaded, files, ["dit", "vae"], NO_FAMILY_GATE, {
          ...hubOptions,
        }),
      Error,
      "コンポーネント 'vae' のファイルが manifest に無い",
    );
    // 読み手が現物と突き合わせられる形（既存の資産診断の流儀）。
    assertStringIncludes(error.message, "dit[0] / dit[1]");
  },
);

Deno.test("loadShardComponents: 取得していないキーで open() すると取得済み一覧つきで落ちる", async () => {
  const { loaded, files, hubOptions } = await prepareTwoShard();
  const { open } = await loadShardComponents(
    "test.fromPretrained",
    loaded,
    files,
    ["dit"],
    NO_FAMILY_GATE,
    hubOptions,
  );
  const error = assertThrows(() => open("vae"), Error, "コンポーネント 'vae' は取得していない");
  assertStringIncludes(error.message, "取得済み: dit");
});

Deno.test("loadShardComponents: 素キーと shard 分割キーの混在は shard 面でも落ちる", async () => {
  // weights が 1 本（= 素キー `dit`）なのに、assets が `dit[0]` の綴りで届く形。以前は素キーを
  // 優先して `dit[0]` を黙って読み飛ばし、資産として全量取得していた。
  const { loaded, files, hubOptions } = await prepareKeyed(
    (refs) => ({ dit: { f32: { shards: [refs.graph] } } }),
    (refs) => ({ "dit[0]": refs.weights }),
  );
  const error = await assertRejects(
    () =>
      loadShardComponents("test.fromPretrained", loaded, files, ["dit"], NO_FAMILY_GATE, {
        ...hubOptions,
      }),
    Error,
    "素のキーと shard 分割キー",
  );
  assertStringIncludes(error.message, "dit / dit[0]");
});

Deno.test("loadShardComponents: 添字が [0] から始まらない取得キーは shard 面でも落ちる", async () => {
  // 重みは別名（`vae`）で持ち、`dit` の綴りでは添字つきの資産だけが届く形。以前は
  // 「素キーも `[0]` も無い」= コンポーネント欠落として、始点がずれていることを言わずに落ちた。
  const { loaded, files, hubOptions } = await prepareKeyed(
    (refs) => ({ vae: { f32: { shards: [refs.graph] } } }),
    (refs) => ({ "dit[1]": refs.graph, "dit[2]": refs.weights }),
  );
  await assertRejects(
    () =>
      loadShardComponents("test.fromPretrained", loaded, files, ["dit"], NO_FAMILY_GATE, {
        ...hubOptions,
      }),
    Error,
    "shard 添字が [0] から始まっていない",
  );
});

// ---- 遅延資産（`eagerAssets` / `deferred` / `readCachedAsset`）⑪ ------------
//
// gemma4 の PLE sidecar（1 本 758MB × 3）の経路。実行者が実重み e2e しか無いと CI では 1 度も
// 踏まれないので、疑似 HF リグで割り振り・prefetch・読み直しの 3 点を縛る。

/** グラフ shard 1 本 + 資産 3 本（`tokenizer` / `ple.0` / `ple.1`）の配布形。 */
const prepareDeferred = async () => {
  const graph = graphShardBytes("linear", [
    ["m.w", f32Tensor([2, 2], 0.5)],
    ["m.b", f32Tensor([2], 0.25)],
  ]);
  const encoder = new TextEncoder();
  const tokenizer = encoder.encode(JSON.stringify({ vocab: ["a", "b"] }));
  const ple0 = encoder.encode("ple shard 0 payload");
  const ple1 = encoder.encode("ple shard 1 payload");
  const refs = {
    graph: await fileRef("dit/model.safetensors", graph),
    tokenizer: await fileRef("tokenizer/tokenizer.json", tokenizer),
    ple0: await fileRef("ple/ple-00000.bin", ple0),
    ple1: await fileRef("ple/ple-00001.bin", ple1),
  };
  const manifest = manifestBytes({
    test: {
      pipeline: "test/1",
      weights: { dit: { f32: { shards: [refs.graph] } } },
      assets: { tokenizer: refs.tokenizer, "ple.0": refs.ple0, "ple.1": refs.ple1 },
      quants: { f32: { weights: { dit: "f32" }, session: {} } },
      defaultQuant: "f32",
      pipelineConfig: {},
    },
  });
  const mock = createMockFetch(
    new Map([
      [MANIFEST_PATH, manifest],
      [refs.graph.path, graph],
      [refs.tokenizer.path, tokenizer],
      [refs.ple0.path, ple0],
      [refs.ple1.path, ple1],
    ]),
  );
  const hubOptions = { fetch: mock.fetch, caches: new MemoryCacheStorage() };
  const loaded = await loadManifest({ repo: REPO, revision: SHA, hubUrl: HUB_URL }, hubOptions);
  return {
    loaded,
    files: resolveFiles(loaded.manifest),
    refs,
    mock,
    hubOptions,
    bytes: { ple0, ple1 },
  };
};

Deno.test(
  "loadShardComponents: eagerAssets が全量常駐と参照のままを割り振り、遅延側も同じ 1 回で落ちる",
  async () => {
    const { loaded, files, refs, mock, hubOptions } = await prepareDeferred();
    const events: AssetProgress[] = [];
    const { assets, deferred } = await loadShardComponents(
      "test.fromPretrained",
      loaded,
      files,
      ["dit"],
      NO_FAMILY_GATE,
      { ...hubOptions, eagerAssets: ["tokenizer"], onProgress: (event) => events.push(event) },
    );

    // 並べたキーだけが全量、残りは参照のまま。
    assertEquals(Object.keys(assets), ["tokenizer"]);
    assertEquals(Object.keys(deferred).toSorted(), ["ple.0", "ple.1"]);

    // 遅延側も**同じ prefetch 1 回**に載る（後から無進捗の DL が始まらない）。
    assertEquals(mock.paths.includes(refs.ple0.path), true, "遅延資産がロード時に落ちない");
    assertEquals(mock.paths.includes(refs.ple1.path), true, "遅延資産がロード時に落ちない");
    const completes = events.filter((event) => event.phase === "complete");
    assertEquals(completes.length, 4);
    const total = refs.graph.size + refs.tokenizer.size + refs.ple0.size + refs.ple1.size;
    assertEquals(events[events.length - 1].loaded, total);
  },
);

Deno.test("loadShardComponents: eagerAssets 未指定なら deferred は空（従来どおり全量）", async () => {
  const { loaded, files, hubOptions } = await prepareDeferred();
  const { assets, deferred } = await loadShardComponents(
    "test.fromPretrained",
    loaded,
    files,
    ["dit"],
    NO_FAMILY_GATE,
    hubOptions,
  );
  assertEquals(Object.keys(deferred), []);
  assertEquals(Object.keys(assets).toSorted(), ["ple.0", "ple.1", "tokenizer"]);
});

Deno.test("readCachedAsset: 遅延資産をキャッシュから読み直す（network へ出ない）", async () => {
  const { loaded, files, mock, hubOptions, bytes } = await prepareDeferred();
  const { deferred } = await loadShardComponents(
    "test.fromPretrained",
    loaded,
    files,
    ["dit"],
    NO_FAMILY_GATE,
    { ...hubOptions, eagerAssets: ["tokenizer"] },
  );

  const before = mock.paths.length;
  const buffer = await readCachedAsset("test.readPle", loaded, deferred["ple.0"], hubOptions);
  assertEquals(new Uint8Array(buffer), bytes.ple0);
  // キャッシュヒット = 取得層は 1 度も叩かれない（prefetch が効いていることの対偶）。
  assertEquals(mock.paths.length, before);
});

Deno.test("readCachedAsset: 届かない参照は「1 本も届かなかった」で落ちる", async () => {
  const { loaded, files, hubOptions } = await prepareDeferred();
  const { deferred } = await loadShardComponents(
    "test.fromPretrained",
    loaded,
    files,
    ["dit"],
    NO_FAMILY_GATE,
    { ...hubOptions, eagerAssets: ["tokenizer"] },
  );
  const missing = { ...deferred["ple.0"], path: "ple/does-not-exist.bin" };
  await assertRejects(() => readCachedAsset("test.readPle", loaded, missing, hubOptions), Error);
});
