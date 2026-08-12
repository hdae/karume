// `AnimaPipeline` の**構築ガード**と session 写像。GPU も実資産も要らない範囲だけを見る
// （実 GPU の E2E は P3 波 2）。
//
// ここで押さえるのは 3 つ:
//  ① `fromAssets` は GPU を取りに行く**前**に manifest の契約違反を落とす（未知 model /
//     pipeline 名 / 未知 major / 未知 quant）。落とす位置がずれると、GPU の無い環境では
//     別の例外に化けて「何が悪かったのか」が読み手に伝わらない。
//  ② manifest の `session`（3 キー固定）→ runtime `SessionOptions` の写像が 1 キーずつ通る。
//     ADR 0038 §3 の綴りの契約そのもので、抜けは**沈黙劣化**（未知キーは runtime が黙って
//     無視する）になる。
//  ③ `generate` の入口ガード（`resolveNegativePrompt`）が「効かないノブを黙って受けない」。
//     回帰しても実 GPU の PNG 門は緑のままなので、純関数として直接縛る。

import { assertEquals, assertRejects, assertThrows } from "@std/assert";
import { parseManifest } from "@karume/hub";
import { AnimaPipeline, resolveNegativePrompt, toSessionOptions } from "../src/anima/pipeline.ts";

const FILE = {
  path: "transformer/model.f16.safetensors",
  size: 16,
  sha256: "a".repeat(64),
};

/** `models/karume-anima-turbo/karume.json` の骨格（検査に要る欄だけ）。 */
const manifestText = (patch: Record<string, unknown> = {}): string =>
  JSON.stringify({
    format: "karume/2",
    generator: "karume/0.1.0",
    defaultModel: "anima-turbo",
    models: {
      "anima-turbo": {
        pipeline: "anima/1",
        weights: { transformer: { f16: { file: FILE } } },
        assets: {},
        quants: { "w8a8-s16": { weights: { transformer: "f16" }, session: {} } },
        defaultQuant: "w8a8-s16",
        pipelineConfig: {
          scheduler: { shift: 3, numTrainTimesteps: 1000 },
          defaults: {
            steps: 10,
            guidanceScale: 1,
            resolution: { width: 1024, height: 1024 },
          },
        },
        ...patch,
      },
    },
  });

const emptyAssets = {} as Record<string, Uint8Array<ArrayBuffer>>;

Deno.test("fromAssets: pipeline の契約名が anima でない manifest を落とす", async () => {
  const manifest = parseManifest(manifestText({ pipeline: "sbv2/1" }));
  await assertRejects(
    () => AnimaPipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "'sbv2/1'",
  );
});

Deno.test("fromAssets: 未知 major は fail loudly（検査責務は models 側 — ADR 0038 §1）", async () => {
  const manifest = parseManifest(manifestText({ pipeline: "anima/2" }));
  // hub は `pipeline` の major を検査しない（読めるかどうかはパイプライン実装しか知らない）。
  assertEquals(manifest.models["anima-turbo"].pipeline, { name: "anima", major: 2 });
  await assertRejects(
    () => AnimaPipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "major に未対応",
  );
});

Deno.test("fromAssets: pipelineConfig のスキーマ違反は構築時に落ちる", async () => {
  const manifest = parseManifest(manifestText({ pipelineConfig: {} }));
  await assertRejects(
    () => AnimaPipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "pipelineConfig.scheduler: 無い",
  );
});

Deno.test("fromAssets: 存在しない quant は利用可能な一覧を添えて落とす", async () => {
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => AnimaPipeline.fromAssets({ manifest, assets: emptyAssets }, { quant: "w8a8" }),
    Error,
    "利用可能: w8a8-s16",
  );
});

Deno.test("fromAssets: 存在しない model は利用可能な一覧を添えて落とす", async () => {
  // v2 で増えた軸。モデル名を打ち間違えたときに「では何があるのか」を一次情報で返す
  // （ADR 0041 §8）。
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => AnimaPipeline.fromAssets({ manifest, assets: emptyAssets }, { model: "anima-xl" }),
    Error,
    "利用可能: anima-turbo",
  );
});

Deno.test("resolveNegativePrompt: guidanceScale 1 で negativePrompt を渡したら落とす", () => {
  // 効かないノブを黙って受けると、ユーザーは指定したつもりで実際は 1 文字も使われない
  // （guidance=1 は uncond 分岐を丸ごと計算しない — `sampler.ts` の `needsUncond`）。
  assertThrows(
    () => resolveNegativePrompt("worst quality", undefined, 1),
    Error,
    "negativePrompt は効かない",
  );
  // 落とすのは **request での明示指定**だけ。manifest の既定が埋まっているだけの状態は
  // 「効かないノブを渡した」ことにならないので、guidance=1 でも通る。
  assertThrows(
    () => resolveNegativePrompt("worst quality", "既定のネガ", 1),
    Error,
    "negativePrompt は効かない",
  );
  assertEquals(resolveNegativePrompt(undefined, undefined, 1), undefined);
});

Deno.test("resolveNegativePrompt: uncond を計算する設定で negativePrompt が無ければ落とす", () => {
  // request にも manifest の defaults にも無い形。ここで落とさないと uncond 側の綴りが
  // 空のまま GPU 経路へ入る。
  assertThrows(
    () => resolveNegativePrompt(undefined, undefined, 7),
    Error,
    "negativePrompt が要る",
  );
  // 供給元は request > defaults の順。どちらかがあれば通る。
  assertEquals(resolveNegativePrompt(undefined, "既定のネガ", 7), "既定のネガ");
  assertEquals(resolveNegativePrompt("要求のネガ", "既定のネガ", 7), "要求のネガ");
});

Deno.test("toSessionOptions: 3 キーを 1 つずつ写す（未指定は欄ごと作らない）", () => {
  assertEquals(toSessionOptions({}), {});
  assertEquals(toSessionOptions({ linearCompute: "i8a8" }), { linearCompute: "i8a8" });
  assertEquals(toSessionOptions({ attentionCompute: "f16" }), { attentionCompute: "f16" });
  assertEquals(toSessionOptions({ attentionScoreStorage: "f16" }), {
    attentionScoreStorage: "f16",
  });
  // 配布物の既定 quant（w8a8-s16）の 3 キーが全て通ること。1 キーでも落とすと
  // 「名前だけ s16」の沈黙劣化になる。
  assertEquals(
    toSessionOptions({
      linearCompute: "i8a8",
      attentionCompute: "i8a8",
      attentionScoreStorage: "f16",
    }),
    {
      linearCompute: "i8a8",
      attentionCompute: "i8a8",
      attentionScoreStorage: "f16",
    },
  );
});

Deno.test("toSessionOptions: manifest 側に無いノブ（submitPolicy）は写さない", () => {
  // `SessionOptions.submitPolicy` は TDR 予算 = **ホスト政策**なので配布者に書かせない
  // （ADR 0038 §3 の理由 ③）。スプレッド素通しに書き換えるとここが素通りしうる。
  const mapped = toSessionOptions({ linearCompute: "i8a8" }) as Record<string, unknown>;
  assertEquals(Object.hasOwn(mapped, "submitPolicy"), false);
  assertEquals(Object.keys(mapped), ["linearCompute"]);
});
