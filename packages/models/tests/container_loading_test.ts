/**
 * コンテナ経路のロード（`src/hub/components.ts`）の門のうち、GPU を取らずに回る側 —
 * 取得層だけを差し替えて固定する（実 GPU が要る 3 点は gpu_container_loading_test.ts）:
 *
 * ① **進捗はモデル全体で 1 本のまま**（取得が「descriptor の温め + 重みの part + 資産の
 *    全量面」へ割れても `loaded` は単調増加で全ファイルの size 合計に着地し、`complete` は
 *    ファイル数ぶん）。越境参照が自リポと同じ path を持っても、合計は 2 本ぶんのまま。
 * ② **admission は重みの part を取る前に落ちる**（part 0 の descriptor だけで capability 違反が
 *    決まり、重みの part の URL は 1 度も叩かれない — ADR 0070 決定 5 の存在理由そのもの）。
 *    この門は prefetch が admission の**後**に置かれていることの門でもある。
 * ③ **重みの part はロード時に落ち切る**（Session を 1 本も張らないうちに DL 済み — 遅延構築の
 *    家族で「初回実行まで DL が遅れ、ロード進捗にも現れない」形を無くす）。
 * ⑤ **abort 済みの `signal` で始めたロードは落ちる**（ロード中の中断契約が消えていないこと。
 *    対の「ロード時の signal を Session 構築へ持ち越さない」は GPU 側）。
 * ⑥ **descriptor のバイト列を握らない**（部品の供給口を保持したままでも常駐ゼロ）。gc を強制
 *    する必要があるので別プロセスの台本（`helpers/descriptor-retention.ts`）へ出し、ここでは
 *    終了コードだけを見る。
 * ⑦ **家族 admission の違反でも重みの part は取得されない**（②の家族版）。② は runtime の
 *    capability 門だけを踏むので、「実行できないモデルの重みは 1 バイトも落とさない」の
 *    うち家族側（pipeline major / `pipelineConfig`）が前段に居ることは縛れない。実家族
 *    （siglip2 = 部品 1 本の最小形）を `fromPretrained` で通して同じ観測法で見る。
 * ⑧ **`requiredLimits` 超過でも重みの part は取得されない（共有 GPU）**（⑦の limits 版 —
 *    ADR 0089 決定 5）。突き合わせ相手は渡された `GpuContext.limits`。
 * ⑩ **選択と供給口の失敗診断**（manifest が持たない部品名 / 開いていない役割）。
 *
 * NOTE: hub / runtime の**テストの都合**は import しない（`helpers/memory-cache.ts` と同じ
 * 規律）。モックは `helpers/container-loading-fixture.ts` に最小限だけ組んである。例外は 2 つ
 * — 容器の**書き手**（形式の道具。fixture の NOTE）と、⑧の `fake-gpu.ts`（`GpuContext` は
 * runtime が値として公開しない〈`acquireGpu` が唯一の入口 — ADR 0008〉ので、共有 GPU 経路を
 * GPU 無し環境で踏むには向こうの実物を包む helper が要る。ここで自前に偽物を組むと、検査対象
 * そのもの（limits を持つ GpuContext）が偽物になる）。
 */

import { assertEquals, assertRejects, assertStringIncludes, assertThrows } from "@std/assert";
import type { AssetProgress } from "@karume/hub";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";
import { loadContainerComponents } from "../src/hub/components.ts";
import { Siglip2Pipeline } from "../src/siglip2/pipeline.ts";
import { linearComponent } from "./helpers/container-fixture.ts";
import {
  HUB_URL,
  NO_FAMILY_GATE,
  prepareComponent,
  prepareSiglip2,
  quantsRequiring,
  REPO,
  serveContainer,
  serveRepo,
  serveRepos,
  SHA,
  SIGLIP2_CONFIG,
} from "./helpers/container-loading-fixture.ts";
import { MemoryCacheStorage } from "./helpers/memory-cache.ts";
import { loadManifest, resolveSelection } from "@karume/hub";

/** 長さ 0 でない part（= 取得層が実際に取りに行く列）。 */
const fetched = (parts: readonly { path: string; size: number }[]): readonly string[] =>
  parts.filter((part) => part.size > 0).map((part) => part.path);

Deno.test(
  "loadContainerComponents: 複数部品でも進捗はモデル全体で 1 本（単調増加・合計へ着地）",
  async () => {
    const front = await serveContainer("front/model.f32", linearComponent("front"));
    const voice = await serveContainer("voice/model.f32", linearComponent("voice"));
    const tokenizer = new TextEncoder().encode(JSON.stringify({ vocab: ["a", "b"] }));
    const tokenizerRef = {
      path: "tokenizer/tokenizer.json",
      size: tokenizer.byteLength,
      sha256: [...new Uint8Array(await crypto.subtle.digest("SHA-256", tokenizer))]
        .map((byte) => byte.toString(16).padStart(2, "0")).join(""),
    };
    const rig = await serveRepo(
      {
        test: {
          pipeline: "test/1",
          weights: { front: { f32: front.entry }, voice: { f32: voice.entry } },
          assets: { tokenizer: tokenizerRef },
          quants: { f32: { weights: { front: "f32", voice: "f32" }, session: {} } },
          defaultQuant: "f32",
          pipelineConfig: {},
        },
      },
      new Map([
        ...front.files,
        ...voice.files,
        [tokenizerRef.path, tokenizer as Uint8Array<ArrayBuffer>],
      ]),
      { front: front.parts, voice: voice.parts },
    );

    const events: AssetProgress[] = [];
    const { open, assets } = await loadContainerComponents(
      "test.fromPretrained",
      rig.loaded,
      rig.selection,
      ["front", "voice"],
      NO_FAMILY_GATE,
      { ...rig.hubOptions, onProgress: (progress) => events.push(progress) },
    );

    // admission を通ったこと（グラフ宣言が読める = 合流まで済んでいる）。
    assertEquals(open("front").graph.outputs, ["y"]);
    assertEquals(open("voice").graph.outputs, ["y"]);
    // 残り資産は全量面のまま届く。
    assertEquals(Object.keys(assets), ["tokenizer"]);

    const paths = [...fetched(front.parts), ...fetched(voice.parts), tokenizerRef.path];
    const total = [...front.parts, ...voice.parts].filter((part) => part.size > 0)
      .reduce((sum, part) => sum + part.size, 0) + tokenizerRef.size;
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
    assertEquals(completes.length, paths.length);
    assertEquals(new Set(completes.map((event) => event.path)), new Set(paths));
    for (const event of completes) assertEquals(event.fileLoaded, event.fileTotal);
    // 長さ 0 の part は取得にも進捗にも現れない（ADR 0109 決定 3）。
    const empty = [...front.parts, ...voice.parts].filter((part) => part.size === 0);
    assertEquals(empty.length > 0, true, "長さ 0 の part を持つ容器で観測していない");
    for (const part of empty) assertEquals(rig.mock.paths.includes(part.path), false);
  },
);

Deno.test(
  "loadContainerComponents: 越境参照が自リポと同じ path でも、進捗は別の 1 本として数える",
  async () => {
    // `voice` は越境先のリポから借りる容器（容器単位の越境 — 全 part が越境先の座標を名乗る）。
    // 書き手の規約どおりの part 名は stem だけで決まるので、同じ stem の 2 本は同じ path を持つ。
    const CROSS_REPO = "karume-test/borrowed";
    const own = await serveContainer("shared/model.f32", linearComponent("front"));
    const borrowed = await serveContainer("shared/model.f32", linearComponent("voice"));
    const crossParts = borrowed.parts.map((part) => ({
      ...part,
      repo: CROSS_REPO,
      revision: SHA,
    }));
    const models = {
      test: {
        pipeline: "test/1",
        weights: {
          front: { f32: own.entry },
          voice: {
            f32: { container: { descriptor: borrowed.written.descriptor, parts: crossParts } },
          },
        },
        assets: {},
        quants: { f32: { weights: { front: "f32", voice: "f32" }, session: {} } },
        defaultQuant: "f32",
        pipelineConfig: {},
      },
    };
    const mock = serveRepos([
      { repo: REPO, models, files: own.files },
      { repo: CROSS_REPO, models: {}, files: borrowed.files },
    ]);
    const hubOptions = { fetch: mock.fetch, caches: new MemoryCacheStorage() };
    const loaded = await loadManifest({ repo: REPO, revision: SHA, hubUrl: HUB_URL }, hubOptions);
    const shared = fetched(own.parts).filter((path) => fetched(borrowed.parts).includes(path));
    assertEquals(shared.length > 0, true, "同じ path を持つ 2 本で観測していない");

    const events: AssetProgress[] = [];
    await loadContainerComponents(
      "test.fromPretrained",
      loaded,
      resolveSelection(loaded.manifest),
      ["front", "voice"],
      NO_FAMILY_GATE,
      { ...hubOptions, onProgress: (progress) => events.push(progress) },
    );

    // path で畳むと `total` が 1 本ぶん小さくなり、`loaded` も同じだけ手前で止まる。
    const total = [...own.parts, ...borrowed.parts].reduce((sum, part) => sum + part.size, 0);
    assertEquals(new Set(events.map((event) => event.total)), new Set([total]));
    assertEquals(events[events.length - 1].loaded, total);
    // 取得先の欄は素通しされ、同じ path の 2 本を消費側でも見分けられる。
    for (const path of shared) {
      const origins = events
        .filter((event) => event.phase === "complete" && event.path === path)
        .map((event) => event.repo)
        .sort();
      assertEquals(origins, [CROSS_REPO, REPO].sort(), `${path} の 2 本が見分けられない`);
    }
  },
);

Deno.test(
  "loadContainerComponents: capability 違反は descriptor だけで落ち、重みの part は取得されない",
  async () => {
    const rig = await prepareComponent({ op: "karume_test_unsupported_op" });
    const parts = rig.parts["dit"];

    const error = await assertRejects(
      () =>
        loadContainerComponents(
          "test.fromPretrained",
          rig.loaded,
          rig.selection,
          ["dit"],
          NO_FAMILY_GATE,
          rig.hubOptions,
        ),
      Error,
    );
    // 落ちた理由が capability 門であること（別の失敗で「重みを取らなかった」が成立しない）。
    if (!error.message.includes("karume_test_unsupported_op")) {
      throw new Error(`capability 門の文言でない: ${error.message}`);
    }
    // part 0（descriptor）は取りに行き、重みの part は 1 度も叩いていない（= 重み DL 前
    // admission。ロード時 prefetch が admission の後に置かれていることの門でもある）。
    assertEquals(rig.mock.paths.includes(parts[0].path), true);
    for (const part of parts.slice(1)) {
      assertEquals(rig.mock.paths.includes(part.path), false, `${part.path} を取っている`);
    }
  },
);

Deno.test(
  "loadContainerComponents: 重みの part は Session を張る前（ロード時）に落ち切り、進捗にも現れる",
  async () => {
    const rig = await prepareComponent();
    const parts = rig.parts["dit"];

    const events: AssetProgress[] = [];
    await loadContainerComponents(
      "test.fromPretrained",
      rig.loaded,
      rig.selection,
      ["dit"],
      NO_FAMILY_GATE,
      { ...rig.hubOptions, onProgress: (progress) => events.push(progress) },
    );

    // Session は 1 本も張っていない。それでも重みの part の URL は叩かれている。
    for (const path of fetched(parts)) {
      assertEquals(rig.mock.paths.includes(path), true, `${path} がロード時に落ちない`);
    }
    // 進捗にも取得した part ぶんが乗る（遅延構築の家族でも「ロード = 全 DL」の表示が成立する）。
    const completes = events.filter((event) => event.phase === "complete");
    assertEquals(new Set(completes.map((event) => event.path)), new Set(fetched(parts)));
    assertEquals(
      events[events.length - 1].loaded,
      parts.reduce((sum, part) => sum + part.size, 0),
    );
  },
);

Deno.test(
  "loadContainerComponents: abort 済みの signal で始めたロードは落ちる（中断契約の維持）",
  async () => {
    const rig = await prepareComponent();
    const reason = new Error("test: ロード開始前に中断済み");

    const error = await assertRejects(
      () =>
        loadContainerComponents(
          "test.fromPretrained",
          rig.loaded,
          rig.selection,
          ["dit"],
          NO_FAMILY_GATE,
          { ...rig.hubOptions, signal: AbortSignal.abort(reason) },
        ),
    );
    assertEquals(error, reason);
  },
);

Deno.test(
  "家族 admission（pipeline major 不一致）は descriptor だけで落ち、重みの part は取得されない",
  async () => {
    const rig = await prepareSiglip2({ pipeline: "siglip2/99" });
    const parts = rig.parts["vision"];

    const error = await assertRejects(
      () =>
        Siglip2Pipeline.fromPretrained(
          { repo: REPO, revision: SHA, hubUrl: HUB_URL },
          { fetch: rig.mock.fetch, caches: rig.hubOptions.caches },
        ),
      Error,
    );
    // 落ちた理由が家族の major 門であること（別の失敗で「重みを取らなかった」が成立しない）。
    if (!error.message.includes("major に未対応")) {
      throw new Error(`家族 admission の文言でない: ${error.message}`);
    }
    // descriptor は取りに行き、重みの part は 1 度も叩いていない（ADR 0070 決定 5 の文面
    // 「実行できないモデルの重みは 1 バイトも落とさない」が家族の門にも及んでいる）。
    assertEquals(rig.mock.paths.includes(parts[0].path), true);
    for (const part of parts.slice(1)) {
      assertEquals(rig.mock.paths.includes(part.path), false, `${part.path} を取っている`);
    }
  },
);

Deno.test(
  "家族 admission（pipelineConfig の schema 違反）でも重みの part は取得されない",
  async () => {
    const rig = await prepareSiglip2({
      pipelineConfig: { ...SIGLIP2_CONFIG, karumeUnknownKey: 1 },
    });
    const parts = rig.parts["vision"];

    const error = await assertRejects(
      () =>
        Siglip2Pipeline.fromPretrained(
          { repo: REPO, revision: SHA, hubUrl: HUB_URL },
          { fetch: rig.mock.fetch, caches: rig.hubOptions.caches },
        ),
      Error,
    );
    if (!error.message.includes("karumeUnknownKey")) {
      throw new Error(`pipelineConfig の門の文言でない: ${error.message}`);
    }
    assertEquals(rig.mock.paths.includes(parts[0].path), true);
    for (const part of parts.slice(1)) {
      assertEquals(rig.mock.paths.includes(part.path), false, `${part.path} を取っている`);
    }
  },
);

Deno.test(
  "家族 admission（requiredLimits 超過）でも重みの part は取得されない（共有 GPU）",
  async () => {
    // 全 limit が 0 の GpuContext（`fake-gpu.ts` の ZERO_LIMITS）へ、1 バイトでも要求する
    // 配布形を渡す。共有 GPU は取り直せない（feature も limits も device 生成時の話）ので、
    // 落とせる唯一の場所が admission 席になる。
    const rig = await prepareSiglip2({ quants: quantsRequiring(1) });
    const parts = rig.parts["vision"];

    const error = await assertRejects(
      () =>
        Siglip2Pipeline.fromPretrained(
          { repo: REPO, revision: SHA, hubUrl: HUB_URL },
          {
            fetch: rig.mock.fetch,
            caches: rig.hubOptions.caches,
            gpu: fakeGpuContext(fakeDevice()),
          },
        ),
      Error,
    );
    // 落ちた理由が limits 検査であること（別の失敗で「重みを取らなかった」が成立しない）。
    if (!error.message.includes("maxBufferSize")) {
      throw new Error(`limits 門の文言でない: ${error.message}`);
    }
    assertEquals(rig.mock.paths.includes(parts[0].path), true);
    for (const part of parts.slice(1)) {
      assertEquals(rig.mock.paths.includes(part.path), false, `${part.path} を取っている`);
    }
  },
);

Deno.test(
  "loadContainerComponents: descriptor のバイト列を握らない（別プロセスで gc 観測）",
  async () => {
    // MUST: 別プロセス — 到達不能なだけの状態と握られた状態を区別するには gc の強制が要り、
    // `deno test` に `--v8-flags` を渡す口が無い。
    const script = new URL("./helpers/descriptor-retention.ts", import.meta.url);
    const { code, stdout, stderr } = await new Deno.Command(Deno.execPath(), {
      args: ["run", "-A", "--v8-flags=--expose-gc", script.href],
      stdout: "piped",
      stderr: "piped",
    }).output();
    const decoder = new TextDecoder();
    assertEquals(code, 0, `${decoder.decode(stdout)}${decoder.decode(stderr)}`);
  },
);

// ---- 選択と供給口の失敗診断（⑩）--------------------------------------------

Deno.test(
  "loadContainerComponents: manifest が持たない部品名は選択の部品一覧つきで落ちる",
  async () => {
    const rig = await prepareComponent();
    const error = await assertRejects(
      () =>
        loadContainerComponents(
          "test.fromPretrained",
          rig.loaded,
          rig.selection,
          ["dit", "vae"],
          NO_FAMILY_GATE,
          rig.hubOptions,
        ),
      Error,
      "部品 'vae' の容器が manifest に無い",
    );
    // 読み手が現物と突き合わせられる形（既存の資産診断の流儀）。
    assertStringIncludes(error.message, "持つ部品: dit");
  },
);

Deno.test("loadContainerComponents: 開いていない役割で open() すると開いた一覧つきで落ちる", async () => {
  const rig = await prepareComponent();
  const { open } = await loadContainerComponents(
    "test.fromPretrained",
    rig.loaded,
    rig.selection,
    ["dit"],
    NO_FAMILY_GATE,
    rig.hubOptions,
  );
  const error = assertThrows(() => open("vae"), Error, "部品 'vae' は開いていない");
  assertStringIncludes(error.message, "開いた部品: dit");
});
