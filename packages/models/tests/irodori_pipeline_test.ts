// `IrodoriPipeline` の**構築ガード**。GPU も実資産も要らない範囲だけを見る（実 GPU の突合は
// `e2e_irodori_*_test.ts` 群が持つ）。
//
// ここで押さえるのは 2 点:
//  ① `fromAssets` は **manifest の契約違反を GPU を取りに行く前**に落とす
//     （`src/irodori/admission.ts` の `admitIrodori` が掲げる MUST）。順序がずれると、GPU の
//     無い環境では別の例外に化けて「何が悪かったのか」が読み手に伝わらない。
//  ② 構築の `signal` が**入口でも実行開始後でも**効く（DL 完了後の組み立てが中断不能だと、
//     UI の中止ボタンが無反応になる窓ができる）。後者は「最初の段境界」までを空資産で見る —
//     それより先の境界は実資産と GPU が要るのでここでは見られない。
//
// 観測の仕掛け: **全ケースで容器 8 本は揃えて**おく（中身は宣言だけ）。したがって
//  - manifest 契約の違反ケースが「その違反の文言」で落ちる = 資産が揃っていても manifest の
//    門が先（容器を開くのは admission の前 — `assetComponentOpener` は同期の供給口を返すため
//    先に全部品を開く）
//  - 空の Record は `部品 'backbone' の容器が無い` で落ちる（受け口の診断）
// の 2 つが噛み合って、門の順序そのものを縛る。グラフ宣言との 12 点突合は
// `irodori_admission_test.ts` が実物と同じ宣言で踏むので、ここでは扱わない。
// 資産 JSON の decode 門（`assetJson`）も同じ理由で `fromAssets` からは届かないので、末尾の 2 本
// だけは**門を直接叩く**。`denoise-step` の `copyLatents`（`latentSnapshot`）も GPU 無しで
// 縛れる純関数なので、最後の 1 本で同じく直接叩く — ここが壊れると購読側へ**別 step の潜在が
// 黙って**届き、実 GPU の WAV 門は観測席を通らないので緑のままになる。

import {
  assertEquals,
  assertNotStrictEquals,
  assertRejects,
  assertStrictEquals,
  assertThrows,
} from "@std/assert";
import { parseManifest } from "@karume/hub";
import { parseIrodoriPipelineConfig } from "../src/irodori/config.ts";
import { assetJson } from "../src/irodori/admission.ts";
import { latentSnapshot } from "../src/irodori/dit-loop.ts";
import { assertIrodoriRequest, IrodoriPipeline } from "../src/irodori/pipeline.ts";
import { ModelInputError } from "../src/errors.ts";
import { declaredContainer, partAssets, tensorlessContainer } from "./helpers/container-fixture.ts";

const FILE = {
  path: "tokenizer.json",
  size: 16,
  sha256: "a".repeat(64),
};

/** グラフ資産の名前（`openIrodoriState` が `assetBuffer` で引く順に並べる）。 */
const WEIGHT_NAMES = [
  "backbone",
  "text_proj",
  "caption_proj",
  "speaker",
  "duration",
  "dit",
  "codec_decoder",
  "codec_encoder",
] as const;

/** `models/karume-irodori-v4-small/karume.json` の `pipelineConfig` 実物（23 欄）。 */
const PIPELINE_CONFIG: Record<string, unknown> = {
  maxTextLen: 256,
  maxCaptionLen: 512,
  speakerRows: 751,
  ditSymMax: 750,
  frameRate: 25,
  sampleRate: 48000,
  hopLength: 1920,
  codecHaloFrames: 8,
  latentDim: 32,
  speakerPatchSize: 4,
  speakerDim: 768,
  textDim: 512,
  captionDim: 512,
  timestepEmbedDim: 512,
  steps: 40,
  initScale: 0.999,
  cfgMinT: 0.5,
  cfgMaxT: 1,
  cfgScales: { text: 3, speaker: 5, caption: 3 },
  minSeconds: 0.5,
  maxSeconds: 30,
  speakerUncondMode: "mask",
  cfgGuidanceMode: "independent",
};

/** 配布形の骨格（検査に要る欄だけ）。`patch` は `models["v4-small"]` の中身を上書きする。 */
const manifestText = (patch: Record<string, unknown> = {}): string => {
  let weights: Record<string, unknown> = {};
  let mapping: Record<string, string> = {};
  for (const name of WEIGHT_NAMES) {
    weights = { ...weights, [name]: { f32: declaredContainer(`${name}/model.f32`) } };
    mapping = { ...mapping, [name]: "f32" };
  }
  return JSON.stringify({
    format: "karume/5",
    generator: "karume/0.1.0",
    defaultModel: "v4-small",
    models: {
      "v4-small": {
        pipeline: "irodori/1",
        weights,
        assets: { tokenizer: { ...FILE, path: "tokenizer.json" } },
        quants: { f32: { weights: mapping, session: {} } },
        defaultQuant: "f32",
        pipelineConfig: PIPELINE_CONFIG,
        ...patch,
      },
    },
  });
};

const emptyAssets = {} as Record<string, Uint8Array<ArrayBuffer>>;

/**
 * 開ける容器 8 本（宣言は最小 — グラフ突合そのものは `irodori_admission_test.ts` が実物と同じ
 * 宣言で踏む）。`tokenizer` は入れない: 門を全部通った先で落ちる 1 本として残す。
 */
const COMPONENTS: Record<string, Uint8Array<ArrayBuffer>> = {};
for (const name of WEIGHT_NAMES) {
  Object.assign(
    COMPONENTS,
    partAssets(
      name,
      await tensorlessContainer(name, {
        inputs: [{ name: "x", shape: [1, 4] }],
        output: { name: "y", shape: [1, 4] },
      }),
    ),
  );
}

/** `pipelineConfig` だけを差し替えた manifest（残りは骨格のまま）。 */
const withConfig = (config: Record<string, unknown>): string =>
  manifestText({ pipelineConfig: config });

Deno.test("fromAssets: 存在しない model は利用可能な一覧を添えて落とす", async () => {
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => IrodoriPipeline.fromAssets({ manifest, assets: COMPONENTS }, { model: "nope" }),
    Error,
    "model 'nope' は manifest に無い",
  );
});

Deno.test("fromAssets: pipeline の契約名が irodori でない manifest を落とす", async () => {
  const manifest = parseManifest(manifestText({ pipeline: "sbv2/1" }));
  await assertRejects(
    () => IrodoriPipeline.fromAssets({ manifest, assets: COMPONENTS }),
    Error,
    "manifest の pipeline が 'sbv2/1'",
  );
});

Deno.test("fromAssets: 未知 major は fail loudly（検査責務は models 側 — ADR 0038 §1）", async () => {
  // 「古い実装 × 新しいリポ」の沈黙劣化を止める唯一の門。hub は major を検査しない。
  const manifest = parseManifest(manifestText({ pipeline: "irodori/2" }));
  await assertRejects(
    () => IrodoriPipeline.fromAssets({ manifest, assets: COMPONENTS }),
    Error,
    "major に未対応",
  );
});

Deno.test("fromAssets: 存在しない quant は利用可能な一覧を添えて落とす", async () => {
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => IrodoriPipeline.fromAssets({ manifest, assets: COMPONENTS }, { quant: "nope" }),
    Error,
    "quant 'nope' は manifest に無い",
  );
});

Deno.test("fromAssets: pipelineConfig の未知キーは構築時に落ちる", async () => {
  // 綴り違い（`steps` に対する `step`）が黙って既定へ縮退する経路を作らない（config.ts の MUST）。
  const manifest = parseManifest(withConfig({ ...PIPELINE_CONFIG, step: 40 }));
  await assertRejects(
    () => IrodoriPipeline.fromAssets({ manifest, assets: COMPONENTS }),
    Error,
    "pipelineConfig: 未知キー 'step'",
  );
});

Deno.test("fromAssets: pipelineConfig の欄が欠けていれば構築時に落ちる", async () => {
  // 欠けた欄が既定で埋まると、ホストだけが別の数を持ったまま **shape は合う**形で沈黙誤値になる
  // （config.ts 冒頭の MUST — モデル固有の数は manifest が正本）。
  const { hopLength: _dropped, ...missing } = PIPELINE_CONFIG;
  const manifest = parseManifest(withConfig(missing));
  await assertRejects(
    () => IrodoriPipeline.fromAssets({ manifest, assets: COMPONENTS }),
    Error,
    "pipelineConfig.hopLength: 無い",
  );
});

Deno.test("fromAssets: 部品の容器が無ければ 2 形の綴りつきで落ちる（受け口の診断）", async () => {
  // 上の 6 ケースの裏返し。容器が揃っていない Record では**部品の不在**で落ちる（manifest の
  // 文言では落ちない）= 上のケースが資産の不在に巻き添えられていないことの対偶。
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => IrodoriPipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "部品 'backbone' の容器が無い",
  );
});

// ---- 構築の中断（`options.signal`）----------------------------------------
//
// UI の中止ボタンは DL 完了後の組み立て（資産解析 → acquireGpu）にも効かなければならない。
// 上の門と同じ manifest + 空資産で見る: signal 無しなら「資産 'backbone' が無い」で落ちる形が、
// 中断済み / 実行中の中断では `signal.reason` で落ちる。

Deno.test("fromAssets: 中断済み signal は資産へ触る前に reason そのままで reject する", async () => {
  const manifest = parseManifest(manifestText());
  const controller = new AbortController();
  const reason = new Error("中止ボタン");
  controller.abort(reason);
  const error = await assertRejects(() =>
    IrodoriPipeline.fromAssets({ manifest, assets: COMPONENTS }, { signal: controller.signal })
  );
  // 包まない（消費側が `error === controller.signal.reason` で自分の中断を識別できる）。
  assertStrictEquals(error, reason);
});

Deno.test("fromAssets: 実行開始後に届いた中断も最初の段境界で効く", async () => {
  // 上の門は「呼ぶ前に中断済み」だけを見る。中止ボタンは**組み立てが走っている最中**に
  // 押されるので、段境界の検査はイベントループへ譲ってからでなければ死文になる
  // （abort() の届き方はタスク配送 — 同期解析中は 1 度も観測されない）。
  // 仕掛けてから呼ぶと、資産エラー（「資産 'backbone' が無い」）ではなく reason で落ちる。
  const manifest = parseManifest(manifestText());
  const controller = new AbortController();
  const reason = new Error("中止ボタン（実行中）");
  setTimeout(() => controller.abort(reason), 0);
  const error = await assertRejects(() =>
    IrodoriPipeline.fromAssets({ manifest, assets: COMPONENTS }, { signal: controller.signal })
  );
  assertStrictEquals(error, reason);
});

// ---- 資産 JSON の decode（不正 UTF-8 を黙って置換しない）------------------

/** JSON としては閉じているが、文字列値に不正 UTF-8（0xff）を 1 バイト混ぜた資産。 */
const brokenUtf8Asset = (): Uint8Array<ArrayBuffer> =>
  Uint8Array.from([
    ...new TextEncoder().encode('{"unkId":'),
    0x22,
    0xff,
    0x22,
    ...new TextEncoder().encode("}"),
  ]);

Deno.test("assetJson: 不正 UTF-8 の資産は decode 段の文言で落ちる（JSON 段まで進まない）", () => {
  const broken = brokenUtf8Asset();
  // 前提の固定: 既定の TextDecoder は 0xff を U+FFFD へ置換するので、置換して読むと
  // **内容の違う valid JSON** が黙って通ってしまう。塞いだのはこの経路。
  assertEquals(
    JSON.parse(new TextDecoder().decode(broken)),
    { unkId: "�" },
    "置換 decode が valid JSON にならない前提が崩れた（このテストの主題が消える）",
  );
  assertThrows(
    () => assetJson({ tokenizer: broken }, "tokenizer"),
    Error,
    "irodori: 資産 'tokenizer' が UTF-8 として読めない",
  );
});

Deno.test("assetJson: JSON 構文違反は decode とは別の文言で落ちる（正常域は不変）", () => {
  const truncated = new TextEncoder().encode('{"unkId":');
  assertThrows(
    () => assetJson({ tokenizer: truncated }, "tokenizer"),
    Error,
    "irodori: 資産 'tokenizer' が JSON として読めない",
  );
  // 正しい UTF-8 は多バイト文字を含んでもそのまま通る。
  const valid = new TextEncoder().encode('{"unkId":0,"space":"あ"}');
  assertEquals(assetJson({ tokenizer: valid }, "tokenizer"), { unkId: 0, space: "あ" });
});

Deno.test("assertIrodoriRequest: 要求ノブの綴り違いは重い計算に入る前に落ちる", () => {
  // これらの門は本来 latent 生成の**後**（`planCodecTiles` は decode 直前・`Randn` は段 ⑦）に
  // しかなく、重み 1.26GB のロードと DiT の全 step を消費してから落ちていた。入口で同じ
  // 受理集合を借りることで、綴り違いの代償が計算時間にならない。
  const config = parseIrodoriPipelineConfig(PIPELINE_CONFIG);
  const text = "テスト";

  // codecTileFrames は「halo 2 枚ぶんより大きい整数」。既定 halo 8 に対し 10 は採用区間が
  // 残らない値（`10 <= 16`）で、素通しすると decode 直前まで走ってから落ちる。
  for (const codecTileFrames of [10, 0, 1.5]) {
    assertThrows(
      () => assertIrodoriRequest({ text, codecTileFrames }, config),
      ModelInputError,
      "より大きい整数でない",
    );
  }
  for (const seed of [-1, 1.5]) {
    assertThrows(
      () => assertIrodoriRequest({ text, seed }, config),
      ModelInputError,
      "非負の安全整数",
    );
  }
  assertThrows(
    () => assertIrodoriRequest({ text, durationSeconds: NaN }, config),
    ModelInputError,
    "durationSeconds",
  );

  // 正常値は通る（門が恒真に落ちていない証拠は上の 6 ケース側が持つ）。
  assertIrodoriRequest({ text, codecTileFrames: 182, seed: 0, durationSeconds: 3 }, config);
  assertIrodoriRequest({ text }, config);
});

Deno.test("assertIrodoriRequest: codecTileFrames を渡さない要求でも既定タイルを検査する", () => {
  // 呼び手は `codecTileFrames` を渡す義務を負っていないので、「渡さなければ通る」形にすると
  // 既定タイルと `codecHaloFrames` の関係が壊れた配布形が**必ず** DiT ループ後に落ちる。
  const config = parseIrodoriPipelineConfig({ ...PIPELINE_CONFIG, codecHaloFrames: 91 });
  assertThrows(
    () => assertIrodoriRequest({ text: "テスト" }, config),
    ModelInputError,
    "tileFrames 182 が halo 2 枚ぶん（182）より大きい整数でない",
  );
});

Deno.test("latentSnapshot: 束縛した時点の潜在を写す（step を進めても写しは変わらない）", () => {
  const shape = [2, 2];
  // DiT ループの再現: `eulerStep` は純関数なので `x` は step ごとに**新しい配列**へ
  // 差し替わる。イベントの口は step ごとに作って渡す。
  const step1 = Float32Array.from([1, 2, 3, 4]);
  const first = latentSnapshot(step1, shape);
  const step2 = Float32Array.from([5, 6, 7, 8]);
  const second = latentSnapshot(step2, shape);

  // 生成が全部終わってから呼んでも「作った時点」の値が返る。ループ変数を閉じ込める実装だと
  // ここが両方 [5, 6, 7, 8] になる（購読側からは検出できない取り違え）。
  assertEquals(Array.from(first().data), [1, 2, 3, 4]);
  assertEquals(Array.from(second().data), [5, 6, 7, 8]);
  assertEquals(first().shape, shape);

  // 返すのは毎回**別の写し**。購読側が受け取った配列を書き換えても、次の写しにも
  // パイプライン側の配列にも波及しない（参照を握られる事故の構造的な排除）。
  const copy = first();
  assertNotStrictEquals(copy.data, step1);
  copy.data[0] = 99;
  assertEquals(Array.from(first().data), [1, 2, 3, 4]);
  assertEquals(Array.from(step1), [1, 2, 3, 4]);
  assertNotStrictEquals(copy.data, first().data);

  // `shape` も同じ扱い（anima 側 `latentSnapshot` と同じ MUST）。参照のまま返すと、同じ step の
  // `copyLatents()` を 2 回呼んだ写しが同一の配列を共有し、購読側が 1 回目の `shape` を
  // 書き換えると 2 回目の写しが黙って別の形を名乗る。
  assertNotStrictEquals(copy.shape, shape);
  // 2 回呼んだ写しが `shape` を共有していない（W-Q6-2 の破れはここに出る — 参照のままだと
  // 1 回目の写しの書き換えが 2 回目の写しへ波及する）。
  assertNotStrictEquals(copy.shape, first().shape);
  shape[0] = 99;
  assertEquals(copy.shape, [2, 2]);
});
