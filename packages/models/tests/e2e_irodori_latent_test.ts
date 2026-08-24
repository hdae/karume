/**
 * Irodori の**移植の門**（実 GPU）。配布形 → `IrodoriPipeline.fromAssets` → `generateLatent`
 * まで通し、出た latent を **full-loop golden の `z` と全要素突合**する（SBV2 の WAV sha256 門
 * `e2e_sbv2_wav_test.ts` の latent 版 — こちらは codec がまだ無いので波形では閉じられない）。
 *
 * golden は `tools/exporter/irodori_pipeline.py` が書く
 * `outputs/series/irodori-v4-small/pipeline/`（`case.full` / `case.no-ref` / `meta.json`）。
 * あちらは**上流の `sample_euler_rf_cfg` との突合を毎回通ってから**書かれるので、この門が閉じる
 * ことは「TS のホスト実装が上流と同じ latent を出す」の 2 段目にあたる。
 *
 * ## 何をビット一致で縛り、何を tolerance で見るか
 *
 * - **S（latent 長）と forward 数は完全一致**を要求する。どちらも整数の判断（expm1 → 銀行家
 *   丸め → clamp / CFG を掛ける t の区間）で、1 でもずれたら式が違う。
 * - **`z` は tolerance 付き**。グラフ経路は golden 側（torch のグラフラッパ）と「条件 KV を
 *   毎 forward 再計算するか」「縮約の順序」「WGSL と ATen の丸め」が構造的に違うので、
 *   ビット一致は原理的に出ない。閾値の導出は {@link Z_ATOL} を参照。
 * - **初期ノイズは golden のものを注入**する（`initialNoise`）。TS の `Randn` と torch の
 *   Philox は別系列なので、乱数まで一致させると門が「乱数の写経」を要求してしまう。乱数列
 *   そのものの決定性は別のテスト（同一 seed で 2 回生成 → バイト同一）で見る。
 *
 * ## 実効値 drift（門の前提）
 *
 * golden と配布形は**別々に焼かれる**（`irodori_pipeline.py` と `karume dist`）ので、
 * サンプラのノブがどちらか片方で動くと「z が合わない」という形でしか出てこない。前提が
 * 崩れたことを先に言うために、`pipelineConfig` と `meta.json` の突合を独立した門に置く。
 *
 * ## 格納 dtype の系列でパラメタ化する（ADR 0018 / 0027 の型）
 *
 * 配布形の quant 席（`f32` / `f16` / `i8`）と full-loop golden の系列は **1 対 1** で、系列ごとに
 * ①golden を fake-quant 済みの重みで焼き直し ②tolerance を**素の実測から独立導出**する
 * （系列間で流用しない — 片方の再導出がもう片方を黙って動かす）。圧縮系列の golden も
 * 丸めた重みで採ってあるので、どの系列でも見ているのは「実装差だけ」で、量子化そのものの
 * 質は聴感ゲート（ユーザー裁定）の領分。
 *
 * NOTE: 席名と系列 root の綴りは**必ずしも同じではない**（`i8+dit4` 席 ↔ `-i4` 系列 — 席名は
 * 実行構成の名前、系列 root は格納 dtype の名前で、`i8` / `i8-a8` は同じバイトを共有する）。
 * この門が見るのは格納 dtype までの席で、活性量子化の `i8-a8` 席は数値パリティ網に**できない**
 * ため別の門が持つ（`e2e_irodori_w8a8_test.ts` — ADR 0025 決定 6 / 0026）。
 *
 * **S と forward 数の完全一致は系列に依らず要求する** — 格納 dtype を落としても duration の
 * 出力を動かす軸は無い（重み格納だけの変更で、S を決める式もホストのまま）ので、ここが
 * 割れたら tolerance ではなく経路を疑う。**i8 で S が動くかどうかは別軸の実測**
 * （`tools/exporter/measure_quant_irodori.py` の S ドリフト表）で、割れたらそちらを先に読む。
 *
 * MUST: 資産は `models/karume-irodori-v4-small/`（untracked・実 GPU 機のローカル資産）と
 * 上の golden。どちらか欠けた環境と GPU 無し環境は生成コマンド付きで**明示 SKIP** する
 * （テストを消して無音で緑にしない — ADR 0005）。
 *
 * NOTE: ローカル読みに `Deno.readFile` を使うのはテストだけ。パッケージ本体は Web 標準 API
 * のみ（横断不変条件）で、`fromAssets` はバイト列を受け取るだけの面。
 */

import { assertEquals } from "@std/assert";
import { IrodoriPipeline } from "../mod.ts";
import { parseIrodoriPipelineConfig } from "../src/irodori/config.ts";
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

/**
 * `z` の全要素突合に使う許容誤差（**f32 系列**）。
 *
 * 実測（`atol = rtol = 0` の素の突合 — この門が毎回ログに出す値。2026-08-12 / 実 GPU）:
 *
 * | ケース   | S   | 要素数 | z の maxAbs 差 | \|z\| 上端 | 比        |
 * | -------- | --- | ------ | -------------- | ---------- | --------- |
 * | full     | 161 | 5,152  | **1.9050e-4**  | 5.10       | 値域の 1/26,800 |
 * | no-ref   | 116 | 3,712  | **7.8994e-4**  | 4.33       | 値域の 1/5,480  |
 *
 * atol 5e-3 は実測最悪 7.8994e-4 の **6.3 倍**で、値域（\|z\| 上端 4.33）の 1/870。差は
 * **構造的**なもので、golden 側（torch のグラフラッパ）と WGSL とで縮約の順序と丸めが違い、
 * それを 40 step の Euler が積み上げた結果（S も forward 数も完全一致しているので、経路の
 * 違いはここしか残らない）。**式の取り違え**（CFG の符号やスケール・t スケジュール・マスクの
 * 区間割り）は値域と同じ O(1) で出る — golden 側の実測で、CFG の有無だけで z は 5.56 / 3.23
 * 動く（`meta.json` の `cfgEffectMaxAbs`）ので、この閾値との間には 3 桁以上の隔たりがある。
 *
 * MUST: 合わないときにここを緩めない。上流突合の閾値（`irodori_pipeline.py` の
 * `EULER_REFERENCE_ATOL` = 1e-3）と同じ性格の数で、動かすと「移植できた」の意味が消える。
 */
const Z_ATOL = 5e-3;

/**
 * **f16 系列**（`--dtype f16` — 適格な重みスロットだけ f16 格納・計算は f32）の `z` の許容誤差。
 *
 * 実測（`atol = rtol = 0` の素の突合 — この門が毎回ログに出す値。2026-08-12 / 実 GPU）:
 *
 * | ケース   | S   | 要素数 | z の maxAbs 差 | \|z\| 上端 | 比       |
 * | -------- | --- | ------ | -------------- | ---------- | -------- |
 * | full     | 161 | 5,152  | 2.1321e-4      | 5.10272    | 1/23,933 |
 * | no-ref   | 116 | 3,712  | 1.8516e-4      | 4.33354    | 1/23,404 |
 *
 * 1.5e-3 = 実測最悪 2.1321e-4 の **7.0 倍**（f32 の 6.3 倍と同じ採り方）・\|z\| 上端の約
 * 1/3,400。golden は fake-quant 済みの重みで採ってあるので、この差も f32 系列と同じ
 * 「実装差だけ」— 実測が f32（7.9e-4）より小さいのは偶然の範囲で、系列間に順序関係は無い。
 *
 * MUST: **f32 の値を流用しない**（ADR 0027 の型 — 系列ごとに素の実測から独立導出する。
 * 流用すると、片方の再導出がもう片方を黙って動かす）。再導出の手順は f32 と同じ:
 * ①f16 系列の資産を焼く（`--dtype f16` の export 3 本 + `karume dist`）②この定数を
 * `undefined` に戻してこの門を走らせ、ログの素の `maxAbs` を表へ書く ③実測最悪の
 * 5〜10 倍を閾値に採り、値域との比を添える。
 *
 * `undefined` のまま資産だけが揃った環境では、この門は**緑にならない**（実測を出してから
 * 赤で止まる）— 未導出の tolerance を「とりあえず f32 と同じ」で埋めて緑にすると、
 * 検出力の無い門が黙って増えるため。
 */
const F16_Z_ATOL: number | undefined = 1.5e-3;

/**
 * **i8 席**（`--dtype i8` — 適格な重みスロットだけ per-channel i8 格納・計算は f32）の `z` の
 * 許容誤差。
 *
 * 実測（`atol = rtol = 0` の素の突合 — この門が毎回ログに出す値。2026-08-12 / 実 GPU）:
 *
 * | ケース   | S   | 要素数 | z の maxAbs 差 | \|z\| 上端 | 比      |
 * | -------- | --- | ------ | -------------- | ---------- | ------- |
 * | full     | 161 | 5,152  | **1.8429e-3**  | 5.11886    | 1/2,777 |
 * | no-ref   | 116 | 3,712  | 1.6475e-4      | 4.33821    | 1/26,331 |
 *
 * 1e-2 = 実測最悪 1.8429e-3 の **5.4 倍**・\|z\| 上端の約 1/512。f16 より 1 桁大きい差が
 * 出るのは i8 格子の粗さがそのまま（golden も同じ丸めた重みなので「実装差」に載る係数が
 * 大きくなる）— S / forwards は完全一致しているので経路は同一。
 *
 * MUST: **f32 / f16 の値を流用しない**（ADR 0027 / 0029 の型 — 系列ごとに素の実測から独立
 * 導出する）。再導出の手順は f16 と同じ: ①i8 系列の資産を焼く（`--dtype i8` の export 3 本 +
 * `karume dist`）②この定数を `undefined` に戻してこの門を走らせ、ログの素の `maxAbs` を上の表へ
 * 書く ③実測最悪の 5〜10 倍を閾値に採り、値域との比を添える。
 *
 * `undefined` のまま資産だけが揃った環境では、この門は**緑にならない**（実測を出してから
 * 赤で止まる — ADR 0050 決定 4）。
 *
 * NOTE: **S / forwards が割れた場合は tolerance の問題ではない** ので、閾値を触る前に
 * `measure_quant_irodori.py` の S ドリフト表を読む。
 */
const W8_Z_ATOL: number | undefined = 1e-2;

/**
 * **i8+dit4 席**（`--dtype i4` — `dit` だけ i4 g32・GPTQ 校正付き。他 7 役は `i8` と同じ i8 バイトを
 * 共有する唯一の混成席）の `z` の許容誤差。
 *
 * golden は**出荷バイトから**焼かれる（`irodori.pipeline_ref` が export 済み i4 コンテナを
 * 読み戻す — 校正を 2 度走らせない）ので、この差も他系列と同じ「実装差だけ」。
 *
 * 実測（`atol = rtol = 0` の素の突合 — この門が毎回ログに出す値。2026-08-23 / 実 GPU）:
 *
 * | ケース   | S   | 要素数 | z の maxAbs 差 | \|z\| 上端 | 比       |
 * | -------- | --- | ------ | -------------- | ---------- | -------- |
 * | full     | 161 | 5,152  | **1.1522e-3**  | 5.03657    | 1/4,371  |
 * | no-ref   | 116 | 3,712  | 2.8908e-5      | 4.99839    | 1/172,907 |
 *
 * 6e-3 = 実測最悪 1.1522e-3 の **5.2 倍**（i8 の 5.4 倍と同じ採り方）・\|z\| 上端の約 1/839。
 * 実測が i8（1.8429e-3）より小さいのは GPTQ 校正の帰結ではなく偶然の範囲 — 系列間に順序
 * 関係は無い（i8 の同注記と同文）。
 *
 * MUST: **f32 / f16 / i8 の値を流用しない**（ADR 0027 / 0050 の型 — 系列ごとに素の実測から
 * 独立導出する）。再導出の手順は i8 と同じ: ①i4 系列の資産を焼く（`--dtype i4` の export +
 * `pipeline_ref --dtype i4` + `karume dist`）②この定数を `undefined` に戻してこの門を走らせ、
 * ログの素の `maxAbs` を表へ書く ③実測最悪の 5〜10 倍を閾値に採り、値域との比を添える。
 *
 * NOTE: **S / forwards が割れた場合は tolerance の問題ではない**（i8 と同文)。i4 で S が
 * 動かないことは J-2 第 2 段の実測（gptq 3 構成とも S 予測完全一致 —
 * research 2026-08-20 §6）が根拠で、上の実測でも S / forwards は完全一致だった。
 */
const W4_Z_ATOL: number | undefined = 6e-3;

/** 決定性の門で使う発話長（秒）。`duration` を回さず S を小さく固定して 2 回生成する。 */
const DETERMINISM_SECONDS = 1;

/**
 * 格納 dtype ごとの系列（配布形の quant 席 × full-loop golden の系列 — 1 対 1）。
 *
 * MUST: golden の置き場を系列ごとに分ける。f32 の golden で f16 資産を突き合わせると、差に
 * 量子化誤差そのものが混ざって tolerance の意味が消える（しかも緩める方向にしか作用しない
 * ので、緑のまま検出力だけが落ちる — `karume.quantize` のモジュール docstring と同じ罠）。
 */
type IrodoriSeries = {
  /** テスト名に出る名前（= 配布形の quant 席の綴り — 系列 root の綴りとは別軸）。 */
  readonly name: string;
  /** full-loop golden の置き場（`irodori_pipeline.py --dtype <name>` の既定の出力先）。 */
  readonly goldenDir: URL;
  /** `z` の全要素突合に使う許容誤差（**系列ごとに実測導出** — `undefined` は未導出）。 */
  readonly zAtol: number | undefined;
  /** SKIP 時にそのまま貼れる golden の生成コマンド。 */
  readonly generate: string;
};

const SERIES: readonly IrodoriSeries[] = [
  {
    name: "f32",
    goldenDir: goldenDir("irodori-v4-small"),
    zAtol: Z_ATOL,
    generate: GOLDEN_COMMAND,
  },
  {
    name: "f16",
    goldenDir: goldenDir("irodori-v4-small-f16"),
    zAtol: F16_Z_ATOL,
    generate: `${GOLDEN_COMMAND} --dtype f16`,
  },
  {
    // 席の綴りは `i8`・系列 root は `-i8`（`i8-a8` 席も同じバイトを指すが、あちらは活性量子化が
    // 効くので数値パリティ網にならず、別の門が持つ）。
    name: "i8",
    goldenDir: goldenDir("irodori-v4-small-i8"),
    zAtol: W8_Z_ATOL,
    generate: `${GOLDEN_COMMAND} --dtype i8`,
  },
  {
    // 席の綴りは `i8+dit4`・系列 root は `-i4`。`dit` だけ i4 で他 7 役は `i8` の i8 バイトを共有する
    // 唯一の混成席（席表の裁定 2026-08-23）。golden の生成は export 済み系列が前提
    // （`pipeline_ref --dtype i4` が出荷バイトを読み戻す）。
    name: "i8+dit4",
    goldenDir: goldenDir("irodori-v4-small-i4"),
    zAtol: W4_Z_ATOL,
    generate: `${GOLDEN_COMMAND} --dtype i4`,
  },
];

/**
 * golden の 1 ケースを配布形で再現して突き合わせる。
 *
 * ノブは 1 つも渡さない（`pipelineConfig` の宣言がそのまま実効値）。参照 latent と初期ノイズ
 * だけが golden 由来で、S は `duration` グラフに決めさせる — S が違えば `initialNoise` の長さで
 * 落ちるので、ここは「S も含めて同じ経路を通った」ことの検査になっている。
 */
const runCase = async (
  series: IrodoriSeries,
  pipeline: IrodoriPipeline,
  name: string,
  expected: GoldenCase,
): Promise<void> => {
  const golden = await readCase(series.goldenDir, name);
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
    `[e2e] irodori ${MODEL}/${series.name} ${name}: ${elapsed}s / S ${latent.frames} / ` +
      `forwards ${latent.forwards} / z maxAbs ${maxAbs.toExponential(4)} ` +
      `(atol ${series.zAtol ?? "未導出"} / |z| 上端 ${expected.zAbsMax}）`,
  );
  // 整数の判断（S の決定式・CFG を掛ける t の区間）は系列に依らず完全一致を要求する。
  assertEquals(latent.frames, expected.S, `${name}: S が golden と違う`);
  assertEquals(latent.forwards, expected.forwards, `${name}: dit の forward 数が golden と違う`);
  assertEquals(latent.latentDim, z.length / expected.S, `${name}: latentDim が golden と違う`);
  if (series.zAtol === undefined) {
    throw new Error(
      `${series.name} 席の tolerance が未導出（実測 maxAbs ${maxAbs.toExponential(4)} / ` +
        `|z| 上端 ${expected.zAbsMax}）— 上のログの実測を全ケースぶん集めて、この席の ` +
        "`*_Z_ATOL` 定数の表を埋める（他の席の値を流用しない — ADR 0027 の型）",
    );
  }
  if (maxAbs > series.zAtol) {
    const frame = Math.floor(at / latent.latentDim);
    throw new Error(
      `${name}: latent が golden と ${maxAbs.toExponential(4)} 違う（許容 ${series.zAtol}）\n` +
        `  最悪は フレーム ${frame} / 列 ${at % latent.latentDim}: ` +
        `golden ${z[at]} → 実測 ${latent.data[at]}\n` +
        "  tolerance を上げるのではなく、CFG 合成式 / t スケジュール / マスク区間 / 条件の " +
        "右 pad のどれが動いたかを先に確かめる",
    );
  }
};

for (const series of SERIES) {
  /** この系列の golden。無い環境は「焼いていない」なので系列ごと SKIP（部分欠けは FAIL 側）。 */
  const metaText = await Deno.readTextFile(new URL("meta.json", series.goldenDir)).catch(
    () => undefined,
  );
  const readMeta = (): GoldenMeta => JSON.parse(metaText as string) as GoldenMeta;
  /** 配布形にこの席があるか（席の綴りと golden の系列は 1 対 1 だが、綴りは別軸）。 */
  const seated = hasQuantSeat(series.name);
  const available = manifestText !== undefined && metaText !== undefined && seated;

  if (!available) {
    console.warn(
      `[karume] Irodori の e2e（${series.name} 系列）を SKIP する（配布形 ` +
        `${ASSETS_DIR.pathname} の quant 席 '${series.name}' と full-loop golden ` +
        `${series.goldenDir.pathname} の両方が要る）。生成: ` +
        `${manifestText === undefined || !seated ? DIST_COMMAND : series.generate}`,
    );
  }

  const runnable = GPU_AVAILABLE && available;

  Deno.test({
    name:
      `e2e(実GPU): 配布形の latent が full-loop golden と一致する（${series.name} / 参照 + caption あり）`,
    ignore: !runnable,
    fn: async () => {
      const manifest = readManifest();
      const assets = await loadLocalAssets(manifest, series.name);
      // `await using` は [Symbol.asyncDispose] 経由の解放をこの実 GPU 経路で検査する意図込み。
      await using pipeline = await IrodoriPipeline.fromAssets({ manifest, assets }, {
        model: MODEL,
        quant: series.name,
      });
      await runCase(series, pipeline, "full", readMeta().cases["full"]);
    },
  });

  Deno.test({
    name:
      `e2e(実GPU): 参照なし / caption 空でも latent が full-loop golden と一致する（${series.name}）`,
    ignore: !runnable,
    fn: async () => {
      const manifest = readManifest();
      const assets = await loadLocalAssets(manifest, series.name);
      await using pipeline = await IrodoriPipeline.fromAssets({ manifest, assets }, {
        model: MODEL,
        quant: series.name,
      });
      // speaker のゼロ短絡と caption 空の CFG off が効く経路（forward が 100 → 60 に落ちる）。
      await runCase(series, pipeline, "no-ref", readMeta().cases["no-ref"]);
    },
  });

  Deno.test({
    name:
      `e2e(実GPU): 同じ seed の生成は 2 回ともバイト同一（${series.name} / 乱数もホストグルーも決定的）`,
    ignore: !runnable,
    fn: async () => {
      const manifest = readManifest();
      const assets = await loadLocalAssets(manifest, series.name);
      await using pipeline = await IrodoriPipeline.fromAssets({ manifest, assets }, {
        model: MODEL,
        quant: series.name,
      });
      // `initialNoise` を渡さない = 内部の `Randn` を通る唯一の経路。`durationSeconds` で S を
      // 固定するのは `duration` を回さず短く保つため（見たいのは長さではなく再現性）。
      const request = {
        text: readMeta().cases["full"].text,
        seed: 7,
        durationSeconds: DETERMINISM_SECONDS,
      };
      const first = await pipeline.generateLatent(request);
      const second = await pipeline.generateLatent(request);
      assertEquals(first.seed, 7);
      assertEquals(first.frames, DETERMINISM_SECONDS * readMeta().frameRate);
      assertEquals(
        new Uint8Array(second.data.buffer),
        new Uint8Array(first.data.buffer),
        "同じ seed・同じ要求で latent が変わった（乱数かホストグルーに非決定性がある）",
      );
    },
  });

  Deno.test({
    name: `e2e(資産): 配布形の pipelineConfig が golden の生成条件と一致する（${series.name}）`,
    ignore: !available,
    fn: () => {
      // GPU は要らない — 門の**前提**（golden 再生成と dist 再生成がずれていないこと）だけを見る。
      const meta = readMeta();
      const config = parseIrodoriPipelineConfig(modelEntry(readManifest()).pipelineConfig);
      const drifted: string[] = [];
      const check = (key: string, expected: number, actual: number): void => {
        if (expected !== actual) drifted.push(`${key} golden ${expected} → 配布形 ${actual}`);
      };
      check("steps", meta.steps, config.steps);
      check("initScale", meta.initScale, config.initScale);
      check("cfgMinT", meta.cfgRange[0], config.cfgMinT);
      check("cfgMaxT", meta.cfgRange[1], config.cfgMaxT);
      check("cfgScales.text", meta.cfgScales.text, config.cfgScales.text);
      check("cfgScales.speaker", meta.cfgScales.speaker, config.cfgScales.speaker);
      check("cfgScales.caption", meta.cfgScales.caption, config.cfgScales.caption);
      check("frameRate", meta.frameRate, config.frameRate);
      check("timestepEmbedDim", meta.tEmbedDim, config.timestepEmbedDim);
      check("maxTextLen", meta.caps.text, config.maxTextLen);
      check("speakerRows", meta.caps.speaker, config.speakerRows);
      check("maxCaptionLen", meta.caps.caption, config.maxCaptionLen);
      if (drifted.length > 0) {
        throw new Error(
          `${series.name}: 配布形の pipelineConfig が golden の生成条件と違う: ` +
            `${drifted.join(" / ")}\n` +
            "（門の前提が崩れている — z の突合を疑う前に、どちらを焼き直したのかを確かめる）",
        );
      }
    },
  });
}
