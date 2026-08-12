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
 * MUST: 資産は `models/karume-irodori-v4-small/`（untracked・実 GPU 機のローカル資産）と
 * 上の golden。どちらか欠けた環境と GPU 無し環境は生成コマンド付きで**明示 SKIP** する
 * （テストを消して無音で緑にしない — ADR 0005）。
 *
 * NOTE: ローカル読みに `Deno.readFile` を使うのはテストだけ。パッケージ本体は Web 標準 API
 * のみ（横断不変条件）で、`fromAssets` はバイト列を受け取るだけの面。
 */

import { assertEquals } from "@std/assert";
import { parseManifest, resolveFiles } from "@karume/hub";
import type { Manifest, ModelEntry } from "@karume/hub";
import { parseSafetensors } from "@karume/runtime";
import { IrodoriPipeline } from "../mod.ts";
import { parseIrodoriPipelineConfig } from "../src/irodori/config.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/** 配布形の置き場（`karume dist --pipeline irodori` の既定の出力先）。 */
const ASSETS_DIR = new URL("../../../models/karume-irodori-v4-small/", import.meta.url);
/** full-loop golden の置き場（`irodori_pipeline.py` の既定の出力先）。 */
const GOLDEN_DIR = new URL("../../../outputs/series/irodori-v4-small/pipeline/", import.meta.url);

/** SKIP 時にそのまま貼れる生成コマンド。 */
const DIST_COMMAND = "cd tools/exporter && uv run karume dist --pipeline irodori";
const GOLDEN_COMMAND =
  "cd tools/exporter && uv run --with 'transformers==5.14.1' python irodori_pipeline.py";

const MODEL = "v4-small";
const QUANT = "f32";

/**
 * `z` の全要素突合に使う許容誤差。
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

/** 決定性の門で使う発話長（秒）。`duration` を回さず S を小さく固定して 2 回生成する。 */
const DETERMINISM_SECONDS = 1;

const manifestText = await Deno.readTextFile(new URL("karume.json", ASSETS_DIR)).catch(
  () => undefined,
);
const metaText = await Deno.readTextFile(new URL("meta.json", GOLDEN_DIR)).catch(() => undefined);
const ASSETS_AVAILABLE = manifestText !== undefined && metaText !== undefined;
if (!ASSETS_AVAILABLE) {
  console.warn(
    `[karume] Irodori の e2e を SKIP する（配布形 ${ASSETS_DIR.pathname} と full-loop golden ` +
      `${GOLDEN_DIR.pathname} の両方が要る）。` +
      `生成: ${manifestText === undefined ? DIST_COMMAND : GOLDEN_COMMAND}`,
  );
}

const RUNNABLE = GPU_AVAILABLE && ASSETS_AVAILABLE;

/** golden `meta.json` のうち、この門が読む欄だけ（形は exporter が持つ — ここは読み口）。 */
type GoldenCase = {
  readonly text: string;
  readonly caption: string;
  readonly S: number;
  readonly forwards: number;
  readonly zAbsMax: number;
  readonly reference: { readonly frames: number } | null;
};
type GoldenMeta = {
  readonly steps: number;
  readonly initScale: number;
  readonly cfgRange: readonly [number, number];
  readonly cfgScales: { readonly text: number; readonly speaker: number; readonly caption: number };
  readonly frameRate: number;
  readonly tEmbedDim: number;
  readonly caps: { readonly text: number; readonly speaker: number; readonly caption: number };
  readonly cases: Readonly<Record<string, GoldenCase>>;
};

const readMeta = (): GoldenMeta => JSON.parse(metaText as string) as GoldenMeta;
const readManifest = (): Manifest => parseManifest(manifestText as string);

const modelEntry = (manifest: Manifest): ModelEntry => {
  if (!Object.hasOwn(manifest.models, MODEL)) {
    throw new Error(
      `配布形に model '${MODEL}' が無い（あるもの: ${manifest.available.models.join(" / ")}）`,
    );
  }
  return manifest.models[MODEL];
};

/** 配布形が要求する資産をローカルから読む（`fetchAssets` のローカル版 — 取得層を通さない）。 */
const loadLocalAssets = async (
  manifest: Manifest,
): Promise<Record<string, Uint8Array<ArrayBuffer>>> => {
  const files = resolveFiles(manifest, { model: MODEL, quant: QUANT });
  const byPath = new Map<string, Uint8Array<ArrayBuffer>>();
  let assets: Record<string, Uint8Array<ArrayBuffer>> = {};
  for (const key of Object.keys(files)) {
    const { path } = files[key];
    const cached = byPath.get(path);
    const bytes = cached ?? await Deno.readFile(new URL(path, ASSETS_DIR));
    if (cached === undefined) byPath.set(path, bytes);
    assets = { ...assets, [key]: bytes };
  }
  return assets;
};

/** golden の 1 ケース（`case.<name>.safetensors`）から f32 テンソルを引く。 */
const readCase = async (
  name: string,
): Promise<(tensor: string) => Float32Array<ArrayBuffer>> => {
  const bytes = await Deno.readFile(new URL(`case.${name}.safetensors`, GOLDEN_DIR));
  const file = parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  return (tensor: string): Float32Array<ArrayBuffer> => {
    const spec = file.tensors.get(tensor);
    if (spec === undefined) throw new Error(`golden case.${name} に '${tensor}' が無い`);
    if (spec.dtype !== "F32") throw new Error(`golden '${tensor}' の dtype が ${spec.dtype}`);
    // MUST: 写して返す（golden の buffer は 1 本を全テンソルで共有しており、`initialNoise` は
    // パイプラインが**そのまま書き換える**配列になる — 借りたままだと 2 回目の突合が壊れる）。
    return new Float32Array(
      file.buffer.slice(spec.byteOffset, spec.byteOffset + spec.byteLength),
    ) as Float32Array<ArrayBuffer>;
  };
};

/** 全要素の最大絶対差と、その位置（差分の報告用）。 */
const worstDifference = (
  actual: Float32Array,
  expected: Float32Array,
): { readonly maxAbs: number; readonly at: number } => {
  if (actual.length !== expected.length) {
    throw new Error(`要素数が違う（実測 ${actual.length} / golden ${expected.length}）`);
  }
  let maxAbs = 0;
  let at = 0;
  for (let index = 0; index < actual.length; index += 1) {
    const difference = Math.abs(actual[index] - expected[index]);
    if (difference > maxAbs) {
      maxAbs = difference;
      at = index;
    }
  }
  return { maxAbs, at };
};

/**
 * golden の 1 ケースを配布形で再現して突き合わせる。
 *
 * ノブは 1 つも渡さない（`pipelineConfig` の宣言がそのまま実効値）。参照 latent と初期ノイズ
 * だけが golden 由来で、S は `duration` グラフに決めさせる — S が違えば `initialNoise` の長さで
 * 落ちるので、ここは「S も含めて同じ経路を通った」ことの検査になっている。
 */
const runCase = async (
  pipeline: IrodoriPipeline,
  name: string,
  expected: GoldenCase,
): Promise<void> => {
  const golden = await readCase(name);
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
      `forwards ${latent.forwards} / z maxAbs ${maxAbs.toExponential(4)} ` +
      `(atol ${Z_ATOL} / |z| 上端 ${expected.zAbsMax}）`,
  );
  assertEquals(latent.frames, expected.S, `${name}: S が golden と違う`);
  assertEquals(latent.forwards, expected.forwards, `${name}: dit の forward 数が golden と違う`);
  assertEquals(latent.latentDim, z.length / expected.S, `${name}: latentDim が golden と違う`);
  if (maxAbs > Z_ATOL) {
    const frame = Math.floor(at / latent.latentDim);
    throw new Error(
      `${name}: latent が golden と ${maxAbs.toExponential(4)} 違う（許容 ${Z_ATOL}）\n` +
        `  最悪は フレーム ${frame} / 列 ${at % latent.latentDim}: ` +
        `golden ${z[at]} → 実測 ${latent.data[at]}\n` +
        "  tolerance を上げるのではなく、CFG 合成式 / t スケジュール / マスク区間 / 条件の " +
        "右 pad のどれが動いたかを先に確かめる",
    );
  }
};

Deno.test({
  name: "e2e(実GPU): 配布形の latent が full-loop golden と一致する（参照 + caption あり）",
  ignore: !RUNNABLE,
  fn: async () => {
    const manifest = readManifest();
    const assets = await loadLocalAssets(manifest);
    // `using` は [Symbol.dispose] 経由の解放をこの実 GPU 経路で検査する意図込み。
    using pipeline = await IrodoriPipeline.fromAssets({ manifest, assets }, {
      model: MODEL,
      quant: QUANT,
    });
    await runCase(pipeline, "full", readMeta().cases["full"]);
  },
});

Deno.test({
  name: "e2e(実GPU): 参照なし / caption 空でも latent が full-loop golden と一致する",
  ignore: !RUNNABLE,
  fn: async () => {
    const manifest = readManifest();
    const assets = await loadLocalAssets(manifest);
    using pipeline = await IrodoriPipeline.fromAssets({ manifest, assets }, {
      model: MODEL,
      quant: QUANT,
    });
    // speaker のゼロ短絡と caption 空の CFG off が効く経路（forward が 100 → 60 に落ちる）。
    await runCase(pipeline, "no-ref", readMeta().cases["no-ref"]);
  },
});

Deno.test({
  name: "e2e(実GPU): 同じ seed の生成は 2 回ともバイト同一（乱数もホストグルーも決定的）",
  ignore: !RUNNABLE,
  fn: async () => {
    const manifest = readManifest();
    const assets = await loadLocalAssets(manifest);
    using pipeline = await IrodoriPipeline.fromAssets({ manifest, assets }, {
      model: MODEL,
      quant: QUANT,
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
  name: "e2e(資産): 配布形の pipelineConfig が golden の生成条件と一致する",
  ignore: !ASSETS_AVAILABLE,
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
        `配布形の pipelineConfig が golden の生成条件と違う: ${drifted.join(" / ")}\n` +
          "（門の前提が崩れている — z の突合を疑う前に、どちらを焼き直したのかを確かめる）",
      );
    }
  },
});
