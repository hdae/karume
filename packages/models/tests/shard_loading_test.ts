/**
 * shard 面のロード経路（`src/hub/components.ts`）の門 — GPU も実資産も使わず、取得層だけを
 * 差し替えて 2 点を固定する:
 *
 * ① **進捗はモデル全体で 1 本のまま**（取得が「グラフ shard の逐次面 + 残り資産の全量面」へ
 *    割れても `loaded` は単調増加で全ファイルの size 合計に着地し、`complete` はファイル数ぶん）。
 * ② **admission は重み shard を取る前に落ちる**（グラフ shard だけで capability 違反が決まり、
 *    重み shard の URL は 1 度も叩かれない — ADR 0070 決定 5 の存在理由そのもの）。
 *
 * NOTE: hub / runtime のテスト helper は import しない（向こうの都合がこちらへ漏れる —
 * `helpers/memory-cache.ts` と同じ規律）。モックはこのファイル内で最小限だけ組む。
 */

import { assertEquals, assertRejects } from "@std/assert";
import { type AssetProgress, loadManifest, resolveFiles } from "@karume/hub";
import { loadShardComponents } from "../src/hub/components.ts";
import { MemoryCacheStorage } from "./helpers/memory-cache.ts";
import { type DumpTensor, writeSafetensors } from "./helpers/safetensors-write.ts";

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
      () => loadShardComponents("test.fromPretrained", loaded, files, ["dit"], hubOptions),
      Error,
    );
    // 落ちた理由が capability 門であること（別の失敗で「重みを取らなかった」が成立しない）。
    if (!error.message.includes("karume_test_unsupported_op")) {
      throw new Error(`capability 門の文言でない: ${error.message}`);
    }
    // 重み shard の URL は 1 度も叩かれていない（= 重み DL 前 admission）。
    assertEquals(mock.paths.includes(refs.weights.path), false);
    assertEquals(mock.paths.includes(refs.graph.path), true);
  },
);
