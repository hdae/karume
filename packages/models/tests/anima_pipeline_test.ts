// `AnimaPipeline` の**構築ガード**。GPU も実資産も要らない範囲だけを見る
// （実 GPU の E2E は P3 波 2）。
//
// ここで押さえるのは 4 つ:
//  ① `fromAssets` は GPU を取りに行く**前**に manifest の契約違反と資産の解析を落とす
//     （未知 model / pipeline 名 / 未知 major / 未知 quant / 資産の不在）。落とす位置が
//     ずれると、GPU の無い環境では別の例外に化けて「何が悪かったのか」が読み手に伝わらない。
//  ② `generate` の入口ガード（`resolveNegativePrompt`）が「効かないノブを黙って受けない」。
//     回帰しても実 GPU の PNG 門は緑のままなので、純関数として直接縛る。
//  ③ `denoise-step` の `copyLatents`（`latentSnapshot`）が「作った時点の latent の写し」を
//     返す。ここが壊れると購読側には**別 step の latent が黙って**届く（実 GPU の PNG 門は
//     観測席を通らないので緑のまま）。
//  ④ 構築の `signal` が**入口でも実行開始後でも**効く（DL 完了後の組み立てが中断不能だと、
//     UI の中止ボタンが無反応になる窓ができる）。後者は「最初の段境界」までを空資産で見る
//     — それより先の境界は実資産と GPU が要るのでここでは見られない。
//
// NOTE: manifest の `session` → `SessionOptions` の写像は 7 家族共有になったので、門は
// `session_options_test.ts` にある。

import {
  assertEquals,
  assertNotStrictEquals,
  assertRejects,
  assertStrictEquals,
  assertThrows,
} from "@std/assert";
import { parseManifest } from "@karume/hub";
import { AnimaPipeline, latentSnapshot, resolveNegativePrompt } from "../src/anima/pipeline.ts";

const FILE = {
  path: "transformer/model.f16.safetensors",
  size: 16,
  sha256: "a".repeat(64),
};

/** `models/karume-anima-turbo/karume.json` の骨格（検査に要る欄だけ）。 */
const manifestText = (patch: Record<string, unknown> = {}): string =>
  JSON.stringify({
    format: "karume/3",
    generator: "karume/0.1.0",
    defaultModel: "anima-turbo",
    models: {
      "anima-turbo": {
        pipeline: "anima/1",
        weights: { transformer: { f16: { shards: [FILE] } } },
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

Deno.test("fromAssets: manifest 契約を全て満たして初めて資産へ触る（門の順序の対偶）", async () => {
  // 上のケースが「資産が空でも manifest の文言で落ちる」ことの裏返し。正しい manifest なら
  // 検査は資産まで進み、`tokenizer` の不在で落ちる（= 契約検査も資産の解析も GPU より前）。
  // GPU の無い環境で `acquireGpu` の失敗に化けたらこの門が赤くなる。
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => AnimaPipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "資産 'tokenizer' が無い",
  );
});

Deno.test("fromAssets: 中断済み signal は資産へ触る前に reason そのままで reject する", async () => {
  // UI の中止ボタンは DL 完了後の組み立て（資産解析 → acquireGpu）にも効かなければならない。
  // 入口の検査が資産解析より**先**にあることを、上の門と同じ manifest + 空資産で見る:
  // signal 無しなら「資産 'tokenizer' が無い」で落ちる形が、中断済みなら reason で落ちる。
  const manifest = parseManifest(manifestText());
  const controller = new AbortController();
  const reason = new Error("中止ボタン");
  controller.abort(reason);
  const error = await assertRejects(() =>
    AnimaPipeline.fromAssets({ manifest, assets: emptyAssets }, { signal: controller.signal })
  );
  // 包まない（消費側が `error === controller.signal.reason` で自分の中断を識別できる）。
  assertStrictEquals(error, reason);
});

Deno.test("fromAssets: 実行開始後に届いた中断も最初の段境界で効く", async () => {
  // 上の門は「呼ぶ前に中断済み」だけを見る。中止ボタンは**組み立てが走っている最中**に
  // 押されるので、段境界の検査はイベントループへ譲ってからでなければ死文になる
  // （abort() の届き方はタスク配送 — 同期解析中は 1 度も観測されない）。
  // 仕掛けてから呼ぶと、資産エラー（「資産 'tokenizer' が無い」）ではなく reason で落ちる。
  const manifest = parseManifest(manifestText());
  const controller = new AbortController();
  const reason = new Error("中止ボタン（実行中）");
  setTimeout(() => controller.abort(reason), 0);
  const error = await assertRejects(() =>
    AnimaPipeline.fromAssets({ manifest, assets: emptyAssets }, { signal: controller.signal })
  );
  assertStrictEquals(error, reason);
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
  // 「効かないノブを渡した」ことにならないので、guidance=1 でも通る。これは配布形
  // `anima-turbo` の**既定経路そのもの**（defaults.guidanceScale = 1 + defaults.negativePrompt）
  // で、ここを締めるリファクタが入ると turbo の既定生成が丸ごと throw する。
  assertEquals(resolveNegativePrompt(undefined, "既定のネガ", 1), "既定のネガ");
  // fallback があっても requested があれば落ちる = 落とす / 落とさないの分岐は
  // **requested の有無だけ**で決まり、fallback の有無では変わらない。
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

Deno.test("latentSnapshot: 束縛した時点の latent を写す（step を進めても写しは変わらない）", () => {
  const shape = [1, 2, 1, 1];
  // denoise ループの再現: `cfgEulerStep` は純関数なので `current` は step ごとに**新しい
  // 配列**へ差し替わる。イベントの口は step ごとに作って渡す。
  const step1 = Float32Array.from([1, 2]);
  const first = latentSnapshot(step1, shape);
  const step2 = Float32Array.from([3, 4]);
  const second = latentSnapshot(step2, shape);

  // 生成が全部終わってから呼んでも「作った時点」の値が返る。ループ変数を閉じ込める実装だと
  // ここが両方 [3, 4] になる（購読側からは検出できない取り違え）。
  assertEquals(Array.from(first().data), [1, 2]);
  assertEquals(Array.from(second().data), [3, 4]);
  assertEquals(first().shape, shape);

  // 返すのは毎回**別の写し**。購読側が受け取った配列を書き換えても、次の写しにも
  // パイプライン側の配列にも波及しない（参照を握られる事故の構造的な排除）。
  const copy = first();
  assertNotStrictEquals(copy.data, step1);
  copy.data[0] = 99;
  assertEquals(Array.from(first().data), [1, 2]);
  assertEquals(Array.from(step1), [1, 2]);
  assertNotStrictEquals(copy.data, first().data);
});
