/**
 * Irodori の **`i8-a8` 席**の門（実 GPU）。重みは `i8` 席と**同じ i8 バイト**で、違いは `dit` と
 * `dit_context` の Session に降りる `linearCompute: "a8"` だけ — DiT の linear（`dit` 245 本 +
 * 条件側 K/V 射影の `dit_context` 72 本 — ADR 0114）が活性まで整数内積で走る。
 *
 * ## なぜ latent 門（`e2e_irodori_latent_test.ts`）に席を足さないのか
 *
 * **活性量子化は数値パリティ網にできない**（ADR 0025 決定 6 / ADR 0026 の検出限界）。活性の
 * 丸めは不連続関数で、上流 1e-5 級の差が丸め境界の ±1 段飛びを起こし数層で飽和する — GPU と
 * torch 鏡像は「同じ分布の別標本」になり、素直な「実測の 5〜10 倍」を閾値に採ると**恒真化する**
 * （どんな実装でも通る）。そこで検出力を 3 本に置き直す:
 *
 * 1. **整数の判断の完全一致** — S / forward 数 / latentDim。活性量子化は `dit` / `dit_context` の
 *    Session の内側にしか掛からず、S を決める `duration` グラフは**その外**（quant の `session` は
 *    `dit` と `dit_context` にだけ渡る — `pipeline.ts` のモジュール doc）。ここが割れたら席の配線が違う。
 * 2. **判別帯**（{@link MEASURED.zBand}）— `i8` golden との z maxAbs が \[下限, 上限\] に入る。
 *    **下限が要る**のが肝で、`linearCompute` が黙って f32 経路へ落ちると差は**小さくなる**
 *    （ADR 0028 決定 6 — 沈黙フォールバックは誤差が小さい側に出る）。上限だけの門は
 *    「i8a8 が一度も走っていない」を緑で通す。
 * 3. **パイプラインキーの census**（{@link MEASURED.ditKeys} と束の census 表
 *    `helpers/census-table.ts`）— `dit` と `dit_context` の run が実際に i8a8 GEMM と
 *    `quantize_rows` を回し、f32 の linear カーネルを **1 回も**回していないこと。2 の下限が
 *    分布の話であるのに対し、こちらは実行そのものの直接観測。
 *
 * `i8` 席（重みだけ i8）の数値パリティは latent 門が持つ。**この門は `i8` 席の門を置き換え
 * ない**（併存 — ADR 0049 決定 5 の形をもう 1 軸伸ばしたもの）。
 *
 * ## golden は `i8` のもの（`outputs/series/irodori-v4-small-i8/pipeline/`）
 *
 * 活性側に対応する torch 鏡像の golden は焼かない（焼いても「別標本」なので網にならない —
 * 上の 1 段目）。突き合わせる相手が `i8` golden であることが、判別帯の下限に
 * 「重みだけ i8 のときの差」という具体的な意味を与えている。
 *
 * MUST: 資産は `models/karume-irodori-v4-small/` の `i8-a8` 席（untracked・実 GPU 機のローカル
 * 資産）と上の golden。欠けた環境と GPU 無し環境は生成コマンド付きで**明示 SKIP** する
 * （テストを消して無音で緑にしない — ADR 0005）。
 */

import { assert, assertEquals } from "@std/assert";
import { acquireGpu } from "@karume/runtime";
import type { SessionDiagnostics } from "@karume/runtime";
import { IrodoriPipeline, type IrodoriRunComponent } from "../mod.ts";
import { requireCensus, SEAT_SIGNATURES } from "../../runtime/tests/helpers/pipeline-census.ts";
import { assertRowCensus, censusRowOf, effectiveSessionOptions } from "./helpers/census-table.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import {
  ASSETS_DIR,
  DIST_COMMAND,
  GOLDEN_COMMAND,
  type GoldenCase,
  goldenDir,
  type GoldenMeta,
  hasQuantSeat,
  loadLocalAssets,
  manifestText,
  MODEL,
  modelEntry,
  readCase,
  readManifest,
  worstDifference,
} from "./helpers/irodori-assets.ts";
import { readTextIfPresent } from "./helpers/read-if-present.ts";

/** 配布形の席の綴りと、その席が指す格納系列（`i8` と 1 組のバイトを共有する）。 */
const QUANT = "i8-a8";
const GOLDEN_DIR = goldenDir("irodori-v4-small-i8");
const GENERATE_GOLDEN = `${GOLDEN_COMMAND} --dtype i8`;

/**
 * **実測から導出する 2 席**（どちらも `undefined` = 未導出。埋めるまでこの門は赤で止まる —
 * ADR 0050 決定 4 の形で、未導出のまま「とりあえず通る値」を置いて検出力の無い門を増やさない）。
 *
 * 1 つのオブジェクトに束ねてあるのは、**同じ 1 回の実走で両方の実測が出る**ため（判別帯の
 * maxAbs とキー本数は、どちらもこの門のログに並ぶ）。
 */
const MEASURED: {
  /**
   * `i8` golden との z maxAbs 差が入るべき**判別帯** `[下限, 上限]`。
   *
   * 実測（`atol = rtol = 0` の素の突合 — この門が毎回ログに出す値。2026-08-12 / 実 GPU）:
   *
   * | ケース   | S    | i8 席の maxAbs | i8-a8 席の maxAbs | \|z\| 上端 |
   * | -------- | ---- | -------------- | ---------------- | ---------- |
   * | full     | 161  | 1.8429e-3      | **2.9717**       | 5.11886    |
   * | no-ref   | 116  | 1.6475e-4      | **1.4349**       | 4.33821    |
   *
   * 採った帯 [0.1, 6]: 下限 0.1 は i8 席の実測最悪 1.8429e-3 の **54 倍**（f32 フォールバックは
   * この帯に届かない）かつ i8-a8 実測最小 1.4349 の 1/14。上限 6 は i8-a8 実測最大 2.9717 の
   * **2.0 倍**（\|z\| 上端 5.12 と同じ桁だが、崩壊は NaN / 発散で桁ごと超える）。
   *
   * 導出（ADR 0026 決定 3 と同じ採り方 — 「実測の 5〜10 倍」ではない）:
   *
   * - **下限** = `i8` 席の実測 maxAbs（latent 門の `W8_Z_ATOL` の素の実測）より**大きく**採る。
   *   ここを 0 や小さい値にすると、`linearCompute` が効かずに f32 経路で走った run が緑で
   *   通る（活性量子化を外すと差は `i8` 席の値まで**縮む** — ADR 0028 決定 6）。
   * - **上限** = i8-a8 の実測 maxAbs の 1.5〜2 倍程度。崩壊（NaN / 発散 / 別の声）の検出で
   *   あって数値パリティではないが、\|z\| 上端と同じ桁まで開いたら意味を失う。
   */
  readonly zBand: readonly [number, number] | undefined;
  /**
   * 部品ごとの run 1 回あたりに走る `quantize_rows` の dispatch 数（`dit` は 1 forward・
   * `dit_context` は生成 1 回の 1 run）。
   *
   * DiT の linear は分割前の 1 グラフで 317 本（k ∈ {32, 192, 512, 768, 1280, 3680} — 全て 4 の
   * 倍数で i8-a8 適格・量子化 recon の sizeBreakdown）。ADR 0114 で条件側 K/V 射影 72 本〈12
   * ブロック × text / speaker / caption × K / V〉を `dit_context` へ割り出したので、`dit` 245 本 +
   * `dit_context` 72 本になった — **グラフが変わったための期待値の変更**で、本数を緩めたのでは
   * ない。合計 317 の検査は置かない（部品ごとにちょうどで縛れば足り、合計は片方の取り違えを
   * 打ち消しうる）。i8a8 の linear は「活性を per-token i8 へ落とす `quantize_rows` → 整数内積の
   * GEMM」の対で走るので、期待は本数の関数として書ける。GEMM 側（席 `linearCompute: "a8"` の
   * 変種キー）の本数は束の census 表（`helpers/census-table.ts`）が持つ — 席の本数の正本を
   * 1 か所にするため。
   *
   * 導出: `undefined` のままこの門を走らせ、ログに出る実測本数を書き写す（源は計測に依らない
   * census = `lastRunPipelines`）。
   */
  readonly ditKeys:
    | { readonly quantizeRows: Readonly<Record<CensusComponent, number>> }
    | undefined;
} = {
  zBand: [0.1, 6],
  ditKeys: { quantizeRows: { dit: 245, "dit-context": 72 } },
};

/** census の生成で使う発話長（秒）。`duration` を回さず S を小さく固定する。 */
const CENSUS_SECONDS = 1;

/** census を採る部品（quant の `session` が降りる 2 本 — 観測席の綴り）。 */
type CensusComponent = Extract<IrodoriRunComponent, "dit" | "dit-context">;
const CENSUS_COMPONENTS: readonly CensusComponent[] = ["dit", "dit-context"];
const isCensusComponent = (component: IrodoriRunComponent): component is CensusComponent =>
  (CENSUS_COMPONENTS as readonly IrodoriRunComponent[]).includes(component);
/** census 表の相（`dit` は denoise の step ごと・`dit_context` は生成 1 回の run）。 */
const PHASE_OF: Readonly<Record<CensusComponent, string>> = { dit: "step", "dit-context": "run" };

/** パイプラインキーの分類（綴りの正本は `src/kernels/linear{,-i8a8}.ts` / `quantize-rows.ts`）。 */
const isLinearKey = (key: string): boolean => key.startsWith("linear:");
const isI8a8 = SEAT_SIGNATURES.linearCompute.a8;
const isQuantizeRows = (key: string): boolean => key.startsWith("quantize_rows:");

/** 1 run ぶんの分類済み dispatch 数。 */
type KeyCensus = {
  readonly i8a8Linear: number;
  readonly quantizeRows: number;
  readonly plainLinear: number;
};

const censusOf = (diagnostics: SessionDiagnostics, component: CensusComponent): KeyCensus => {
  // MUST: census の無い run を「0 本」として数えない — キー検査が黙って空振りする
  //（requireCensus が undefined / 空を落とす）。
  const entries = requireCensus(diagnostics.lastRunPipelines, component);
  let i8a8Linear = 0;
  let quantizeRows = 0;
  let plainLinear = 0;
  for (const entry of entries) {
    if (isLinearKey(entry.key)) {
      if (isI8a8(entry.key)) i8a8Linear += entry.dispatchCount;
      else plainLinear += entry.dispatchCount;
    } else if (isQuantizeRows(entry.key)) {
      quantizeRows += entry.dispatchCount;
    }
  }
  return { i8a8Linear, quantizeRows, plainLinear };
};

const metaText = await readTextIfPresent(new URL("meta.json", GOLDEN_DIR));
const readMeta = (): GoldenMeta => JSON.parse(metaText as string) as GoldenMeta;

const seated = hasQuantSeat(QUANT);
const AVAILABLE = manifestText !== undefined && metaText !== undefined && seated;

if (!AVAILABLE) {
  console.warn(
    `[karume] Irodori の e2e（${QUANT} 席）を SKIP する（配布形 ${ASSETS_DIR.pathname} の ` +
      `quant 席 '${QUANT}' と i8 の full-loop golden ${GOLDEN_DIR.pathname} の両方が要る）。` +
      `生成: ${manifestText === undefined || !seated ? DIST_COMMAND : GENERATE_GOLDEN}`,
  );
}

const RUNNABLE = GPU_AVAILABLE && AVAILABLE;

const openPipeline = async (): Promise<IrodoriPipeline> => {
  const manifest = readManifest();
  const assets = await loadLocalAssets(manifest, QUANT);
  return await IrodoriPipeline.fromAssets({ manifest, assets }, { model: MODEL, quant: QUANT });
};

/**
 * golden の 1 ケースを `i8-a8` 席で再現し、整数の判断の一致と判別帯を見る。
 *
 * 入力の与え方は latent 門と同じ（参照 latent と初期ノイズだけ golden 由来・S は `duration`
 * グラフに決めさせる）。違うのは**突合の性格**だけ — こちらは帯であって tolerance ではない。
 */
const runCase = async (
  pipeline: IrodoriPipeline,
  name: string,
  expected: GoldenCase,
): Promise<void> => {
  const golden = await readCase(GOLDEN_DIR, name);
  const started = performance.now();
  const latent = await pipeline.generateLatent({
    text: expected.text,
    ...(expected.caption === "" ? {} : { caption: expected.caption }),
    ...(expected.reference === null ? {} : { speaker: { latent: golden("reference_latent") } }),
    initialNoise: golden("noise"),
  });
  const elapsed = ((performance.now() - started) / 1000).toFixed(1);
  const z = golden("z");
  const { maxAbs, at } = worstDifference(latent.data, z);
  console.log(
    `[e2e] irodori ${MODEL}/${QUANT} ${name}: ${elapsed}s / S ${latent.frames} / ` +
      `forwards ${latent.forwards} / i8 golden との z maxAbs ${maxAbs.toExponential(4)} ` +
      `(判別帯 ${MEASURED.zBand?.join(" 〜 ") ?? "未導出"} / |z| 上端 ${expected.zAbsMax}）`,
  );
  // 活性量子化は整数の判断を 1 つも変えない（`duration` は `dit` の Session の外）。
  assertEquals(latent.frames, expected.S, `${name}: S が i8 golden と違う`);
  assertEquals(latent.forwards, expected.forwards, `${name}: dit の forward 数が違う`);
  assertEquals(latent.latentDim, z.length / expected.S, `${name}: latentDim が違う`);
  if (MEASURED.zBand === undefined) {
    throw new Error(
      `${QUANT} 席の判別帯が未導出（実測 maxAbs ${maxAbs.toExponential(4)} / ` +
        `|z| 上端 ${expected.zAbsMax}）— 上のログの実測と latent 門の i8 席の実測を並べて ` +
        "`MEASURED.zBand` を埋める（下限は i8 席の実測より大きく採る）",
    );
  }
  const [floor, ceiling] = MEASURED.zBand;
  if (maxAbs < floor) {
    throw new Error(
      `${name}: i8 golden との差 ${maxAbs.toExponential(4)} が判別帯の下限 ${floor} を` +
        '下回った — 活性量子化が走っていない疑い（`linearCompute: "a8"` が席から降りて' +
        "いないか、適格判定が外れて f32 経路へ落ちている）。差が**小さい**のは沈黙" +
        "フォールバックの兆候で、良化ではない（ADR 0028 決定 6）",
    );
  }
  if (maxAbs > ceiling) {
    const frame = Math.floor(at / latent.latentDim);
    throw new Error(
      `${name}: i8 golden との差 ${maxAbs.toExponential(4)} が判別帯の上限 ${ceiling} を` +
        `超えた（崩壊の検出）\n  最悪は フレーム ${frame} / 列 ${at % latent.latentDim}: ` +
        `golden ${z[at]} → 実測 ${latent.data[at]}\n` +
        "  帯を広げるのではなく、i8a8 の scale / accumulator / 適格判定のどれが動いたかを" +
        "先に確かめる（数値契約の正本は packages/runtime/tests/gpu_i8a8_test.ts の atol=0）",
    );
  }
};

Deno.test({
  name: `e2e(実GPU): i8-a8 席の latent が i8 golden の判別帯に入る（参照 + caption あり）`,
  ignore: !RUNNABLE,
  fn: async () => {
    await using pipeline = await openPipeline();
    await runCase(pipeline, "full", readMeta().cases["full"]);
  },
});

Deno.test({
  name: `e2e(実GPU): 参照なし / caption 空でも i8-a8 席が判別帯に入る`,
  ignore: !RUNNABLE,
  fn: async () => {
    await using pipeline = await openPipeline();
    await runCase(pipeline, "no-ref", readMeta().cases["no-ref"]);
  },
});

Deno.test({
  name:
    `e2e(実GPU): dit / dit_context の run が i8a8 GEMM と quantize_rows だけで回る（キー census）`,
  ignore: !RUNNABLE,
  fn: async () => {
    // ここで `gpu` を渡すので破棄もこちらの責任になる（渡した側が所有権を持つ —
    // `IrodoriPipelineOptions.gpu`）。計測を要求しない device なので DiT ループは常駐経路
    // （src/irodori/pipeline.ts — `enqueue` ごとに観測席へ届き、census はその run のもの）。
    const gpu = await acquireGpu();
    try {
      const manifest = readManifest();
      const assets = await loadLocalAssets(manifest, QUANT);
      // 束の census 表の行（鍵は quant 席の宣言を家族の合成に通した実効設定）。この門は `i8-a8`
      // 席そのものの門なので、行が無いのは表の欠落として落とす。
      const where = `irodori ${MODEL}/${QUANT}`;
      const row = censusRowOf(
        "irodori",
        MODEL,
        effectiveSessionOptions("irodori", modelEntry(manifest).quants[QUANT].session, {}, where),
      );
      assert(row !== undefined, `${where}: 束の census 表（helpers/census-table.ts）に行が無い`);
      const observed: Record<CensusComponent, KeyCensus[]> = { dit: [], "dit-context": [] };
      const runs: Record<CensusComponent, SessionDiagnostics["lastRunPipelines"][]> = {
        dit: [],
        "dit-context": [],
      };
      await using pipeline = await IrodoriPipeline.fromAssets({ manifest, assets }, {
        gpu,
        model: MODEL,
        quant: QUANT,
        onRunDiagnostics: (component, diagnostics) => {
          if (!isCensusComponent(component)) return;
          observed[component].push(censusOf(diagnostics, component));
          runs[component].push(diagnostics.lastRunPipelines);
        },
      });
      // `durationSeconds` で S を固定するのは `duration` を回さず短く保つため（見たいのは
      // 長さではなく「何が走ったか」）。参照も caption も無い最小構成で足りる。
      await pipeline.generateLatent({
        text: readMeta().cases["no-ref"].text,
        seed: 7,
        durationSeconds: CENSUS_SECONDS,
      });

      if (observed.dit.length === 0) throw new Error("dit の run が 1 回も観測されなかった");
      // 条件側 K/V は生成 1 回に 1 run（forward ごとに回っていたら ADR 0114 のホイストが外れている）。
      assertEquals(
        observed["dit-context"].length,
        1,
        "dit_context の run が生成 1 回に 1 回でない",
      );
      for (const component of CENSUS_COMPONENTS) {
        const first = observed[component][0];
        const count = observed[component].length;
        console.log(
          `[e2e] irodori ${MODEL}/${QUANT} census: ${component} run ${count} 回 / ` +
            `i8a8 linear ${first.i8a8Linear} 本 / quantize_rows ${first.quantizeRows} 本 / ` +
            `f32 linear ${first.plainLinear} 本`,
        );
        // 全 forward が同じ計画で走る（uncond はマスク還元 — ADR 0047 決定 1）ので、本数は
        // run ごとに揺れない。揺れたら「一部の run だけ別の経路へ落ちた」の直接の証拠になる。
        for (const [index, census] of observed[component].entries()) {
          assertEquals(census, first, `${component} の run ${index} だけ dispatch の内訳が違う`);
        }
        assertEquals(
          first.plainLinear,
          0,
          `${component}: f32 骨格の linear が走っている（適格判定が外れて一部が黙って f32 経路へ落ちた）`,
        );
      }
      // 席の変種（i8a8 GEMM）と参照経路に残った linear を run ごとにちょうどの本数で見る
      // （相 → 部品は census 表の軸の綴り — `dit` は step・`dit_context` は run）。
      const checked = assertRowCensus(
        row,
        new Map(CENSUS_COMPONENTS.map((component) => [PHASE_OF[component], runs[component]])),
        (phase) => phase === PHASE_OF.dit ? "dit" : "dit-context",
        where,
      );
      assert(checked > 0, `${where}: 表の期待と突き合わせた run が 1 本も無い`);
      if (MEASURED.ditKeys === undefined) {
        throw new Error(
          `i8-a8 のキー本数が未導出（実測 quantize_rows dit ${observed.dit[0].quantizeRows} / ` +
            `dit-context ${observed["dit-context"][0].quantizeRows}）— 上のログの実測を` +
            " `MEASURED.ditKeys` へ書く",
        );
      }
      for (const component of CENSUS_COMPONENTS) {
        assertEquals(
          observed[component][0].quantizeRows,
          MEASURED.ditKeys.quantizeRows[component],
          `${component} の quantize_rows の dispatch 数が期待と違う（DiT の linear 本数か融合が動いた）`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
});
