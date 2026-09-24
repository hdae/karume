// RAM ピーク harness の門（ADR 0108 段階分解表 段 2 の検収②③の縮図）。
//
// 合成の小さな `krm` 配布形 1 本を作り、`measure.ts` を **cold → warm → local** の順に通して
// container-v1 §7 の表（digest の回数）とキャッシュの挙動を観測する:
//
// | 取得元                                  | この門が固定する事実                                   |
// | --------------------------------------- | ------------------------------------------------------ |
// | cold（疑似 HF + 空のキャッシュ）        | キャッシュへ**書く**（`cache.puts > 0`）               |
// | warm（同じ疑似 HF + 温まったキャッシュ）| **1 本も書かない**・ヒットだけで済む                   |
// | local（手元のディレクトリ）             | **未検証の取得元**なので block ごとに digest が掛かる  |
//
// ## digest の計数で「cold > 0 / warm == 0」にならないこと（実装の事実）
//
// 取得層（`@hdae/fetch-cache` 0.8.0）が cold の part に掛ける逐次 sha256 は**純 TS 実装**
// （`src/sha256.ts` — `crypto.subtle.digest` が一括専用でストリームに使えないため）であり、
// `crypto.subtle.digest` の包みには 1 度も現れない。したがって**容器だけの配布形では cold の
// payload 側 digest も 0** になる。cold / warm の対比はキャッシュの書込本数で取り、digest の側は
// 「未検証の取得元だけが block ごとに digest を掛ける」（§7 の 3 行目）を固定する。
//
// GPU が要る（Session を 1 本張る）ので、アダプタが無い環境は明示 SKIP（ADR 0005）。

import { assert, assertEquals } from "@std/assert";
// NOTE: 容器 → 配布形の組み立ては models のテスト helper を借りる（`writeModelContainer` の
// 薄い包み + 実行できる最小の部品）。**形式の道具**であって向こうのテストの都合ではないので、
// tools から使っても「テストが他パッケージのテスト内部に依存する」形にはならない
// （models 自身も runtime の `container-write.ts` を同じ理由で借りている）。
import {
  linearComponent,
  sha256Hex,
  writeContainer,
} from "../../packages/models/tests/helpers/container-fixture.ts";
import { measure, type MeasureReport, type MeasureState } from "./measure.ts";

const detectAdapter = async (): Promise<boolean> => {
  const gpu: GPU | undefined = navigator.gpu;
  if (gpu === undefined) return false;
  return (await gpu.requestAdapter()) !== null;
};

const GPU_AVAILABLE: boolean = await detectAdapter();

if (!GPU_AVAILABLE) {
  console.warn(
    "[karume] GPUAdapter が無いため tools/ram-peak の実 GPU テストを SKIP する" +
      "（リリース判定は実 GPU 緑が必須 — ADR 0005）",
  );
}

/** 合成配布形の部品名（= グラフ名 — container-v1 §2.1）。 */
const COMPONENT = "dit";

/**
 * `external` の最大値に許す「定数」ぶん（MiB）。
 *
 * container-v1 §11 の見積りは「最大 part 長 + 重ね合わせ」だが、実測の `external` には
 * Deno / wgpu が抱える ArrayBuffer（アダプタ情報・パイプラインキャッシュ等）も乗る。合成
 * fixture の part は 4KiB なので、ここが見ているのは実質「定数の側が跳ねていないこと」である。
 * 2026-09-23 の開発機（Linux / Arc B570）では baseline 1MiB に対しピークも 1MiB（増分 1MiB 未満）。
 */
const EXTERNAL_ALLOWANCE_MIB = 64;

/** 合成の `karume/5` 配布形を `dir` に書く（1 モデル / 1 部品 / 1 quant）。 */
const writeFixtureDist = async (dir: string): Promise<void> => {
  const written = await writeContainer(linearComponent(COMPONENT));
  const total = String(written.parts.length).padStart(5, "0");
  await Deno.mkdir(`${dir}/${COMPONENT}`, { recursive: true });
  const parts = [];
  for (const [index, bytes] of written.parts.entries()) {
    const path = `${COMPONENT}/model.f32-${String(index + 1).padStart(5, "0")}-of-${total}.krm`;
    await Deno.writeFile(`${dir}/${path}`, bytes);
    parts.push({ path, size: bytes.byteLength, sha256: await sha256Hex(bytes) });
  }
  const manifest = {
    format: "karume/5",
    generator: "karume-ram-peak-test/0",
    defaultModel: "test",
    models: {
      test: {
        pipeline: "test/1",
        weights: { [COMPONENT]: { f32: { container: { descriptor: written.descriptor, parts } } } },
        assets: {},
        quants: { f32: { weights: { [COMPONENT]: "f32" }, session: {} } },
        defaultQuant: "f32",
        pipelineConfig: {},
      },
    },
  };
  await Deno.writeTextFile(`${dir}/karume.json`, JSON.stringify(manifest));
};

const measureState = (
  state: MeasureState,
  source: string,
  cacheDir: string,
): Promise<MeasureReport> =>
  measure({
    mode: "component",
    state,
    // 部品面では使わないが、条件を 1 つの型で運ぶために既定を渡す。
    family: "anima",
    source,
    cacheDir,
    component: COMPONENT,
    steps: 2,
    size: 512,
    maxNewTokens: 8,
    explicitGc: false,
  });

Deno.test({
  name: "ram-peak: 合成 krm 配布形を cold → warm → local で測る",
  ignore: !GPU_AVAILABLE,
  fn: async (t) => {
    const root = await Deno.makeTempDir({ prefix: "karume-ram-peak-" });
    const dist = `${root}/dist`;
    const cacheDir = `${root}/cache`;
    await Deno.mkdir(dist, { recursive: true });
    try {
      await writeFixtureDist(dist);

      const cold = await measureState("cold", dist, cacheDir);
      const warm = await measureState("warm", dist, cacheDir);
      const local = await measureState("local", dist, cacheDir);

      await t.step("cold は空のキャッシュへ書き、warm は 1 本も書かない", () => {
        assert(cold.cache !== null, "cold のキャッシュ実績が無い");
        assert(warm.cache !== null, "warm のキャッシュ実績が無い");
        assert(cold.cache.puts > 0, `cold が書いていない（puts=${cold.cache.puts}）`);
        assert(cold.cache.putBytes > 0, "cold の書込バイト数が 0");
        assertEquals(warm.cache.puts, 0, "warm がキャッシュへ書いている（cold が温めていない）");
        assert(warm.cache.hits > 0, `warm がヒットしていない（hits=${warm.cache.hits}）`);
      });

      await t.step("descriptor 2 文書の突合はどの取得元でも 2 回掛かる", () => {
        for (const report of [cold, warm, local]) {
          assertEquals(
            report.digest.total.descriptorCalls,
            2,
            `${report.state}: descriptor の digest 回数`,
          );
        }
      });

      await t.step("未検証の取得元（local）だけが block ごとに digest を掛ける", () => {
        // container-v1 §7 の表: HF 経由は取得層が検証済みを名乗るので block の digest は 0 回。
        assertEquals(cold.digest.total.payloadCalls, 0, "cold で block の digest が掛かっている");
        assertEquals(warm.digest.total.payloadCalls, 0, "warm で block の digest が掛かっている");
        assert(
          local.digest.total.payloadCalls > 0,
          "local で block の digest が掛かっていない（未検証の取得元の検証が抜けている）",
        );
      });

      await t.step("external の最大値が「最大 part 長 + 定数」以内", () => {
        for (const report of [cold, warm, local]) {
          const ceiling = report.externalBaselineMiB +
            Math.ceil(report.fetch.maxPartBytes / (1024 * 1024)) + EXTERNAL_ALLOWANCE_MIB;
          assert(
            report.peaks.total.externalMaxMiB <= ceiling,
            `${report.state}: external 最大 ${report.peaks.total.externalMaxMiB}MiB が` +
              ` 上限 ${ceiling}MiB を超えた`,
          );
        }
      });

      await t.step("取得の内訳が part を役割で分ける", () => {
        assertEquals(cold.fetch.containerCount, 1);
        assertEquals(cold.fetch.components[0].component, COMPONENT);
        assertEquals(cold.fetch.components[0].partRoles[0], "descriptor");
        assert(cold.fetch.descriptorPartBytes > 0, "part 0 のバイト数が 0");
        assert(cold.fetch.weightPartBytes > 0, "重み part のバイト数が 0");
        assertEquals(cold.fetch.assetPartBytes, 0, "資産を持たない容器に資産 part がある");
        assertEquals(cold.fetch.mixedPartBytes, 0, "重みと資産が同居した part がある");
        assert(cold.fetch.blockBytes.weight > 0, "重み block のバイト数が 0");
      });

      await t.step("持越し scale は「runtime に席が無い」と名乗る", () => {
        assert(
          cold.missingFromRuntime.some((note) => note.includes("carriedScaleBytes")),
          "持越し scale の欠落が出力に出ていない",
        );
      });
    } finally {
      await Deno.remove(root, { recursive: true });
    }
  },
});
