/**
 * SBV2 の**移植の門**（実 GPU）。配布形 → `Sbv2Pipeline.fromAssets` → `generate` →
 * `encodeWav` まで通し、出力 WAV の sha256 が参照値と**ビット一致**するかだけを見る
 * （Anima の PNG 門 `e2e_anima_test.ts` と同じ哲学の音声版）。
 *
 * 参照値は ADR [0039](../../../docs/decisions/0039-sbv2-distribution.md) の Consequences 節が
 * 記録した実測 — 「配布形 + `w8` で出した WAV と、`outputs/series/` を直接読む段 1 経路
 * （i8 構成）で出した WAV が sha256 完全一致」。比較経路そのものはコミット 0bbfc65
 * （example を `Sbv2Pipeline` の 1 画面へ縮退）で消えており、digest は散文にしか残っていな
 * かった。ここが自動の門としての置き場になる。
 *
 * **数値が 1 bit でも動いたら移植のどこかが変わっている — tolerance 化も参照値の差し替えも
 * 禁止**で、赤のまま止めて差分の内容（WAV バイト長 / サンプル数 / 実効ノブ / 実物の WAV）を
 * 出す。ここを緩めると「移植できた」の意味が消える。
 *
 * ## 参照値の生成条件（git 履歴から復元 — 推測で埋めない）
 *
 * - **モデル FN4 / quant `w8`**（front i8 + voice i8 + text_encoder i8）。ADR 0039 の
 *   Consequences 節の記述そのもの。
 * - **テキストは `examples/sbv2/main.ts` の `DEFAULT_TEXT`・seed 0** — 0bbfc65 が入れた
 *   既定値（`args.get("text") ?? DEFAULT_TEXT` / `integer("seed") ?? 0`）で、現在も同値。
 * - **style / speaker / styleWeight / 4 ノブは manifest の `pipelineConfig.defaults`**。
 *   example は未指定のノブを `generate` へ渡さないため、実効値は配布形の既定がそのまま。
 *   段 1 側（旧 example）も同じ値で、スタイルと話者は `sbv2_demo.py assets` が
 *   `style_bert_vits2.constants` の `DEFAULT_STYLE` / `DEFAULT_STYLE_WEIGHT` から焼いた
 *   `assets.safetensors` を読んでいた — `karume dist` が `pipelineConfig.defaults` を書く
 *   ときの出所と同じ定数（`karume/dist.py`）。
 * - 実効値は {@link REFERENCE_KNOBS} に写してあり、**配布形の既定が動いたら門より先に**
 *   落ちる（「数値の回帰」と「資産の差し替え」を混同しないため）。
 *
 * ## 2 本目の門 — `w8-bert4`（BERT の linear を i4 混成で焼いた席）
 *
 * 条件は `w8` と同一（FN4 / seed 0 / 配布形の既定ノブ）で、差し替えるのは quant だけ。参照値は
 * 2026-08-20 にこの参照環境で採取して凍結した（perf-ledger Q-1 の配布配線）。`w8` と**別の値に
 * なるのが正**なので、`w8` の digest と一致した場合も落とす — 一致は「i4 席が効いていない」
 * （quant 解決が既定へ落ちた / i4 席に i8 資産が入った）ことの証拠で、数値の網では捕まらない。
 *
 * ## 3 本目の門 — `w4`（3 席とも i4 混成で焼いた配布形）
 *
 * 条件はやはり `w8` と同一で、差し替えるのは quant だけ。参照値は 2026-08-20 に採取して凍結した
 * （perf-ledger Q-1 の full-w4 側）。`w8` とも `w8-bert4` とも**別の値になるのが正**で、どちらかと
 * 一致したら落とす — `w8-bert4` との一致は「net_g 側の i4 席が効いていない」ことの証拠で、
 * net_g の適格 linear は 6 本しかない（front 2 / voice 4）ぶん**資産サイズの差でも気づけない**。
 *
 * ## 参照 digest はこの参照環境専用（クロスデバイスのビット同一は保証しない）
 *
 * 参照値は参照環境（RTX 3080 Ti / Linux / Vulkan (wgpu)）で焼いたもので、他バックエンド
 * （Metal 等）では一致しないのが仕様。その機序と、別バックエンドでの健全性検証の作法
 * （自己 A/B）は [limitations](../../../docs/limitations.md) の「sha256 参照門は参照環境専用」
 * 節にある。
 *
 * MUST: 資産は `models/karume-sbv2-fn/`（untracked・実 GPU 機のローカル資産）。無い環境と
 * GPU 無し環境は理由を出して**明示 SKIP** する（テストを消して無音で緑にしない — ADR 0005）。
 *
 * NOTE: ローカル読みに `Deno.readFile` を使うのはテストだけ。パッケージ本体は Web 標準 API
 * のみ（横断不変条件）で、`fromAssets` はバイト列を受け取るだけの面。日本語辞書だけは
 * パイプラインが自分で取りに出る（hub 非経由の唯一の経路 — `pipeline.ts` のモジュール doc）
 * ので、初回だけネットワークが要る（以降は Cache API から返る）。辞書の版も波形を決める
 * 入力なので、突合が割れたときのために `@hdae/yomi` の版をログに出す。
 */

import { assertEquals, assertRejects } from "@std/assert";
import { parseManifest, resolveFiles } from "@karume/hub";
import type { Manifest, ModelEntry } from "@karume/hub";
import { VERSION as YOMI_VERSION } from "@hdae/yomi";
import { encodeWav, type GeneratedAudio, Sbv2Pipeline } from "../mod.ts";
import { parseSbv2PipelineConfig, type Sbv2Defaults } from "../src/sbv2/config.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { buildSafetensors, f32Bytes } from "./helpers/safetensors.ts";

/** 資産の置き場（リポ直下 `models/karume-sbv2-fn/`）。 */
const ASSETS_DIR = new URL("../../../models/karume-sbv2-fn/", import.meta.url);
const OUTPUTS_DIR = new URL("../../../outputs/demo/", import.meta.url);
/**
 * 参照 WAV の**実体**があれば置かれている場所（旧 example の dump 先・ホスト資産）。
 * 門は sha256 だけで閉じており、ここは**差分位置を出すためだけ**の任意の材料 —
 * 無くても、sha256 が {@link REFERENCE_SHA256} でなくても、門の判定は変わらない。
 */
const REFERENCE_WAV = new URL("../../../outputs/sbv2-demo/out/out.wav", import.meta.url);

const MODEL = "FN4";
const QUANT = "w8";
const TEXT = "こんにちは、これはテストです。";
const SEED = 0;

/**
 * `w8` と**同構成で `text_encoder` だけ i4 混成**（BERT の linear と語彙表が group32 の i4・
 * conv と相対位置表は i8）の quant。配布形の差分は 1 席だけで、session ノブは `w8` と同じ
 * （`sbv2/distribution.py` の `SBV2_QUANTS` がそれを固定する）。
 */
const BERT4_QUANT = "w8-bert4";

/** 参照値（ADR 0039 の実測 — **変更禁止**）。 */
const REFERENCE_SHA256 = "a82f72e2c18956ec725a3f692182e8c9a7dad4011e760dab9fb3d051653db2f4";

/**
 * `w8-bert4` の参照値（2026-08-20 に参照環境で採取して凍結 — **変更禁止**）。
 *
 * `w8` と**別の値になるのが正**（BERT の linear が i4 に落ちれば特徴量が動き、波形も動く）。
 * 一致してしまったら i4 席が効いていない（i8 資産が i4 席に入った / quant 解決が既定へ落ちた）
 * ことを意味するので、そこも門に含める。実体は {@link REFERENCE_WAV} のような手元の材料が
 * 無いので、食い違ったときは WAV を落として人が聴ける形にするだけにする。
 *
 * NOTE: 2026-08-20 の embedding i4 追補で `text_encoder` 席の中身（語彙表が i8 → i4）が変わった
 * ため、参照を採り直した（旧値 `aa9671ad…`）。同日、encoder linear の丸めが GPTQ 校正付きに
 * なり（perf-ledger Q-6 — 格納形は不変・丸め値だけが動く）再度採り直した（旧値 `c2e2f567…`）。
 */
const BERT4_REFERENCE_SHA256 = "e1aabe02821962a4eff869a3aa1565d892b8b0eed235487741ffb485cb797ef4";

/**
 * `w8-bert4` から**さらに `front` / `voice` も i4 混成**へ替えた quant（3 席とも i4）。session
 * ノブは `w8` と同じで、動かす軸は格納形だけ（`sbv2/distribution.py` の `SBV2_QUANTS`）。
 */
const W4_QUANT = "w4";

/**
 * `w4` の参照値（2026-08-20 に参照環境で採取して凍結 — **変更禁止**）。
 *
 * `w8` とも `w8-bert4` とも**別の値になるのが正**。`w8-bert4` と一致したら net_g 側の i4 席が
 * 効いていない（i4 席に i8 資産が入った / quant 解決が既定へ落ちた）ことを意味し、net_g の適格
 * linear は 6 本だけで配布バイトもほぼ変わらないため、**この門以外に検出手段が無い**。
 *
 * NOTE: 2026-08-20 の embedding i4 追補で `text_encoder` 席の中身（語彙表が i8 → i4）が変わった
 * ため、参照を採り直した（旧値 `c0bc803e…`）。同日、encoder linear の丸めが GPTQ 校正付きに
 * なり（perf-ledger Q-6）再度採り直した（旧値 `054bfca7…`）。さらに同日、conv1d の i4 追補
 * （ADR 0069 追記 7 — `front` / `voice` 席の conv1d が i4 g32 へ）で三度採り直した
 * （旧値 `2d69fc80…`）。
 */
const W4_REFERENCE_SHA256 = "869f1cfd7e3c0633a5f97f2e61fa7dee90861deb241350aa3e746d617189517b";

/** 参照 WAV を焼いた時点の実効ノブ（配布形の `pipelineConfig.defaults` と一致するはず）。 */
const REFERENCE_KNOBS: Sbv2Defaults = {
  speaker: "FN4",
  style: "Neutral",
  styleWeight: 1,
  sdpRatio: 0.2,
  noiseScale: 0.6,
  noiseScaleW: 0.8,
  lengthScale: 1,
};
const KNOB_KEYS = [
  "speaker",
  "style",
  "styleWeight",
  "sdpRatio",
  "noiseScale",
  "noiseScaleW",
  "lengthScale",
] as const satisfies readonly (keyof Sbv2Defaults)[];

const manifestText = await Deno.readTextFile(new URL("karume.json", ASSETS_DIR)).catch(
  () => undefined,
);
const ASSETS_AVAILABLE = manifestText !== undefined;
if (!ASSETS_AVAILABLE) {
  console.warn(
    `[karume] ${ASSETS_DIR.pathname} に karume.json が無いため SBV2 の e2e を SKIP する` +
      "（移植の門は実資産でしか閉じない — exporter の dist.py で焼く）",
  );
}

const RUNNABLE = GPU_AVAILABLE && ASSETS_AVAILABLE;

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

const readManifest = (): Manifest => parseManifest(manifestText as string);

const modelEntry = (manifest: Manifest): ModelEntry => {
  if (!Object.hasOwn(manifest.models, MODEL)) {
    throw new Error(
      `配布形に model '${MODEL}' が無い（あるもの: ${manifest.available.models.join(" / ")}）`,
    );
  }
  return manifest.models[MODEL];
};

/**
 * FN4 / w8 が要求する資産をローカルから読む（`fetchAssets` のローカル版 — 取得層を通さない
 * 経路で、パイプライン単体を門に掛ける）。同じ path を指すキーは 1 度だけ読む。
 */
const loadLocalAssets = async (
  manifest: Manifest,
  quant: string = QUANT,
): Promise<Record<string, Uint8Array<ArrayBuffer>>> => {
  const files = resolveFiles(manifest, { model: MODEL, quant });
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

/**
 * 門の**前提**を検査する（門そのものではない）。参照 WAV は配布形の既定ノブで焼かれている
 * ので、既定が動いた配布形に対しては「sha256 が違う」ではなく「条件が違う」と言って落とす。
 */
const assertReferenceKnobs = (entry: ModelEntry): Sbv2Defaults => {
  const { defaults } = parseSbv2PipelineConfig(entry.pipelineConfig);
  const drifted = KNOB_KEYS.filter((key) => defaults[key] !== REFERENCE_KNOBS[key]);
  if (drifted.length > 0) {
    throw new Error(
      `配布形の pipelineConfig.defaults が参照 WAV の生成条件と違う: ` +
        drifted.map((key) => `${key} ${REFERENCE_KNOBS[key]} → ${defaults[key]}`).join(" / ") +
        "（門の前提が崩れている — 参照 sha256 を差し替えるのではなく、条件が変わった理由を先に確かめる）",
    );
  }
  return defaults;
};

/**
 * 実効ノブの 1 行表示（突合が割れたときに「何を焼いたか」を報告へ残す）。門は `generate` へ
 * ノブを 1 つも渡さないので、配布形の既定＝実効値。
 */
const describeKnobs = (defaults: Sbv2Defaults, quant: string = QUANT): string =>
  `model ${MODEL} / quant ${quant} / seed ${SEED} / text ${JSON.stringify(TEXT)} / ` +
  KNOB_KEYS.map((key) => `${key} ${defaults[key]}`).join(" / ") +
  ` / yomi ${YOMI_VERSION}`;

/**
 * 参照 WAV の実体が手元にあれば**先頭差分位置**まで出す。
 *
 * MUST: 実体の sha256 を確かめてから使う — 別の条件で焼かれた WAV を参照として差分を出すと、
 * 診断そのものが嘘になる。
 */
const describeFirstDifference = async (actual: Uint8Array<ArrayBuffer>): Promise<string> => {
  const reference = await Deno.readFile(REFERENCE_WAV).catch(() => undefined);
  if (reference === undefined) {
    return `参照 WAV の実体が無い（${REFERENCE_WAV.pathname}）— 先頭差分位置は出せない`;
  }
  const sha = await sha256Hex(reference);
  if (sha !== REFERENCE_SHA256) {
    return `${REFERENCE_WAV.pathname} は参照 WAV ではない（sha256 ${sha}）— 先頭差分位置は出せない`;
  }
  const shared = Math.min(reference.length, actual.length);
  let at = 0;
  while (at < shared && reference[at] === actual[at]) at += 1;
  if (at === shared) {
    return `先頭 ${shared} バイトは一致（長さだけが違う: 参照 ${reference.length} / 実測 ${actual.length}）`;
  }
  const header = 44;
  const where = at < header
    ? `RIFF ヘッダ内（オフセット ${at}）`
    : `サンプル ${Math.floor((at - header) / 2)}（オフセット ${at}）`;
  const view = (bytes: Uint8Array): number =>
    new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
      .getInt16(header + Math.floor((at - header) / 2) * 2, true);
  const values = at < header
    ? `バイト ${reference[at]} → ${actual[at]}`
    : `i16 ${view(reference)} → ${view(actual)}`;
  return `先頭差分は ${where}: ${values}`;
};

/**
 * 参照 sha と食い違ったときの報告。
 *
 * MUST: ここで tolerance に逃げない。実物を `outputs/demo/` へ落として、人が聴き比べ・
 * 突き合わせできる形にする（`outputs/demo/` は「消して安全」な置き場 — docs/assets-layout.md）。
 */
const mismatchReport = async (
  audio: GeneratedAudio,
  wav: Uint8Array<ArrayBuffer>,
  actual: string,
  defaults: Sbv2Defaults,
): Promise<string> => {
  await Deno.mkdir(OUTPUTS_DIR, { recursive: true });
  const dumped = new URL("e2e-sbv2-mismatch.wav", OUTPUTS_DIR);
  await Deno.writeFile(dumped, wav);
  return `出力 WAV の sha256 が参照と一致しない\n` +
    `  期待 ${REFERENCE_SHA256}\n  実際 ${actual}\n` +
    `  WAV ${wav.length} バイト / サンプル ${audio.data.length} / ${audio.sampleRate}Hz / ` +
    `${(audio.data.length / audio.sampleRate).toFixed(3)}s\n` +
    `  実効ノブ ${describeKnobs(defaults)}\n` +
    `  ${await describeFirstDifference(wav)}\n` +
    `  実物 ${dumped.pathname}`;
};

Deno.test({
  name:
    `e2e(実GPU): 配布形 ${MODEL} / quant ${QUANT} / seed ${SEED} の WAV が参照 sha256 と一致する`,
  ignore: !RUNNABLE,
  fn: async () => {
    const manifest = readManifest();
    const defaults = assertReferenceKnobs(modelEntry(manifest));
    const assets = await loadLocalAssets(manifest);
    const started = performance.now();
    // `await using` は [Symbol.asyncDispose] 経由の解放をこの実 GPU 経路で検査する意図込み。
    await using pipeline = await Sbv2Pipeline.fromAssets({ manifest, assets }, {
      model: MODEL,
      quant: QUANT,
    });
    // ノブは 1 つも渡さない — 参照 WAV と同じく `pipelineConfig.defaults` に埋めさせる。
    const audio = await pipeline.generate({ text: TEXT, seed: SEED });
    const wav = encodeWav(audio.data, audio.sampleRate);
    const actual = await sha256Hex(wav);
    const elapsed = ((performance.now() - started) / 1000).toFixed(1);
    console.log(
      `[e2e] sbv2 ${MODEL}/${QUANT}: ${elapsed}s / WAV ${wav.length}B / ` +
        `${(audio.data.length / audio.sampleRate).toFixed(2)}s / yomi ${YOMI_VERSION} / ` +
        `sha256 ${actual}`,
    );
    if (actual !== REFERENCE_SHA256) {
      throw new Error(await mismatchReport(audio, wav, actual, defaults));
    }
  },
});

Deno.test({
  name:
    `e2e(実GPU): 配布形 ${MODEL} / quant ${BERT4_QUANT} / seed ${SEED} の WAV が参照 sha256 と一致する`,
  ignore: !RUNNABLE,
  fn: async () => {
    const manifest = readManifest();
    const defaults = assertReferenceKnobs(modelEntry(manifest));
    const assets = await loadLocalAssets(manifest, BERT4_QUANT);
    const started = performance.now();
    await using pipeline = await Sbv2Pipeline.fromAssets({ manifest, assets }, {
      model: MODEL,
      quant: BERT4_QUANT,
    });
    // ノブは 1 つも渡さない — `w8` の門と同じ条件（差は text_encoder の格納形だけ）。
    const audio = await pipeline.generate({ text: TEXT, seed: SEED });
    const wav = encodeWav(audio.data, audio.sampleRate);
    const actual = await sha256Hex(wav);
    const elapsed = ((performance.now() - started) / 1000).toFixed(1);
    console.log(
      `[e2e] sbv2 ${MODEL}/${BERT4_QUANT}: ${elapsed}s / WAV ${wav.length}B / ` +
        `${(audio.data.length / audio.sampleRate).toFixed(2)}s / yomi ${YOMI_VERSION} / ` +
        `sha256 ${actual}`,
    );
    if (actual === REFERENCE_SHA256) {
      throw new Error(
        `${BERT4_QUANT} の WAV が ${QUANT} と sha256 完全一致した（${actual}）— ` +
          "i4 席が効いていない（i4 quant の解決が既定へ落ちた / i4 席に i8 資産が入っている）",
      );
    }
    if (actual !== BERT4_REFERENCE_SHA256) {
      await Deno.mkdir(OUTPUTS_DIR, { recursive: true });
      const dumped = new URL("e2e-sbv2-w8-bert4-mismatch.wav", OUTPUTS_DIR);
      await Deno.writeFile(dumped, wav);
      throw new Error(
        `出力 WAV の sha256 が参照と一致しない\n` +
          `  期待 ${BERT4_REFERENCE_SHA256}\n  実際 ${actual}\n` +
          `  WAV ${wav.length} バイト / サンプル ${audio.data.length} / ${audio.sampleRate}Hz / ` +
          `${(audio.data.length / audio.sampleRate).toFixed(3)}s\n` +
          `  実効ノブ ${describeKnobs(defaults, BERT4_QUANT)}\n` +
          `  実物 ${dumped.pathname}`,
      );
    }
  },
});

Deno.test({
  name:
    `e2e(実GPU): 配布形 ${MODEL} / quant ${W4_QUANT} / seed ${SEED} の WAV が参照 sha256 と一致する`,
  ignore: !RUNNABLE,
  fn: async () => {
    const manifest = readManifest();
    const defaults = assertReferenceKnobs(modelEntry(manifest));
    const assets = await loadLocalAssets(manifest, W4_QUANT);
    const started = performance.now();
    await using pipeline = await Sbv2Pipeline.fromAssets({ manifest, assets }, {
      model: MODEL,
      quant: W4_QUANT,
    });
    // ノブは 1 つも渡さない — `w8` の門と同じ条件（差は 3 席の格納形だけ）。
    const audio = await pipeline.generate({ text: TEXT, seed: SEED });
    const wav = encodeWav(audio.data, audio.sampleRate);
    const actual = await sha256Hex(wav);
    const elapsed = ((performance.now() - started) / 1000).toFixed(1);
    console.log(
      `[e2e] sbv2 ${MODEL}/${W4_QUANT}: ${elapsed}s / WAV ${wav.length}B / ` +
        `${(audio.data.length / audio.sampleRate).toFixed(2)}s / yomi ${YOMI_VERSION} / ` +
        `sha256 ${actual}`,
    );
    // 席の不発は 2 通りある（3 席とも既定へ落ちた / net_g の 2 席だけ i8 のまま）ので、
    // 「先行する 2 つの quant のどちらかと一致したら落とす」を 1 本の判定で持つ。
    const collided = [[QUANT, REFERENCE_SHA256], [BERT4_QUANT, BERT4_REFERENCE_SHA256]]
      .find(([, digest]) => digest === actual);
    if (collided !== undefined) {
      throw new Error(
        `${W4_QUANT} の WAV が ${collided[0]} と sha256 完全一致した（${actual}）— ` +
          "i4 席が効いていない（i4 quant の解決が既定へ落ちた / i4 席に i8 資産が入っている）",
      );
    }
    if (actual !== W4_REFERENCE_SHA256) {
      await Deno.mkdir(OUTPUTS_DIR, { recursive: true });
      const dumped = new URL("e2e-sbv2-w4-mismatch.wav", OUTPUTS_DIR);
      await Deno.writeFile(dumped, wav);
      throw new Error(
        `出力 WAV の sha256 が参照と一致しない\n` +
          `  期待 ${W4_REFERENCE_SHA256}\n  実際 ${actual}\n` +
          `  WAV ${wav.length} バイト / サンプル ${audio.data.length} / ${audio.sampleRate}Hz / ` +
          `${(audio.data.length / audio.sampleRate).toFixed(3)}s\n` +
          `  実効ノブ ${describeKnobs(defaults, W4_QUANT)}\n` +
          `  実物 ${dumped.pathname}`,
      );
    }
  },
});

// --- スタイル / 話者の門（実資産が要る側）------------------------------------
//
// 表の行数・列数の突合（`assertTableFits`）は `front` のグラフ入力の静的次元を要求するので、
// 実資産と GPU が無いと届かない（合成フィクスチャで書ける範囲は `sbv2_style_test.ts`）。
// ここが落ちる形は全て**沈黙誤値クラス** — 通してしまうと shape は合ったまま別のスタイル・
// 別の話者の声が出る（ADR 0039 決定 4）。

Deno.test({
  name: "e2e(実GPU): スタイル / 話者は実資産の表から解決され、食い違いは fail loudly",
  ignore: !RUNNABLE,
  fn: async (t) => {
    const manifest = readManifest();
    const assets = await loadLocalAssets(manifest);
    const open = (
      patch: Record<string, Uint8Array<ArrayBuffer>> = {},
    ): Promise<Sbv2Pipeline> =>
      Sbv2Pipeline.fromAssets({ manifest, assets: { ...assets, ...patch } }, {
        model: MODEL,
        quant: QUANT,
      });
    const table = (name: string, rows: number, cols: number): Uint8Array<ArrayBuffer> =>
      new Uint8Array(
        buildSafetensors([{
          name,
          dtype: "F32",
          shape: [rows, cols],
          data: f32Bytes(new Float32Array(rows * cols)),
        }]),
      );

    await using pipeline = await open();

    await t.step(
      "未知のスタイルは利用可能な一覧を添えて落ちる（既定へ黙って落ちない）",
      async () => {
        await assertRejects(
          () => pipeline.generate({ text: TEXT, style: "Angry" }),
          Error,
          "スタイル 'Angry' は manifest に無い",
        );
      },
    );

    await t.step("未知の話者も同じく落ちる", async () => {
      await assertRejects(
        () => pipeline.generate({ text: TEXT, speaker: "FN1" }),
        Error,
        "話者 'FN1' は manifest に無い",
      );
    });

    await t.step("style_vectors の行数が pipelineConfig の件数と違えば構築時に落ちる", async () => {
      // FN4 は 4 スタイル。3 行の表を渡すと、行番号 3 が表の外を指す（= 沈黙誤値の入口）。
      await assertRejects(
        () => open({ style_vectors: table("style_vectors", 3, 256) }),
        Error,
        "の行数 3 が pipelineConfig の 4 件",
      );
    });

    await t.step("speaker_embeddings の列数がグラフ入力と違えば構築時に落ちる", async () => {
      await assertRejects(
        () => open({ speaker_embeddings: table("speaker_embeddings", 1, 256) }),
        Error,
        "の列数 256 がグラフ入力の",
      );
    });
  },
});

// --- 観測席（onRunDiagnostics）------------------------------------------------
//
// 1 合成 = text_encoder → front → voice の 3 run が**この順で** 1 回ずつ観測され、診断が
// 実行実績（dispatch 数・直近 run のアリーナ実績）を運ぶことを固定する。gpuTiming を
// 要求しない GPU では `lastRunTiming` が undefined のまま（ADR 0021/0032 — 黙って近似
// しない）ことも門に含める。

Deno.test({
  name: "e2e(実GPU): onRunDiagnostics は run 1 回ごとに component 名付きで診断を運ぶ",
  ignore: !RUNNABLE,
  fn: async () => {
    const manifest = readManifest();
    const assets = await loadLocalAssets(manifest);
    const seen: { component: string; dispatches: number; ran: boolean; timed: boolean }[] = [];
    await using pipeline = await Sbv2Pipeline.fromAssets({ manifest, assets }, {
      model: MODEL,
      quant: QUANT,
      onRunDiagnostics: (component, diagnostics) =>
        seen.push({
          component,
          dispatches: diagnostics.submit.dispatchCount,
          ran: diagnostics.lastRun !== undefined,
          timed: diagnostics.lastRunTiming !== undefined,
        }),
    });
    await pipeline.generate({ text: TEXT, seed: SEED });
    assertEquals(seen.map((run) => run.component), ["text_encoder", "front", "voice"]);
    for (const run of seen) {
      if (run.dispatches <= 0 || !run.ran) {
        throw new Error(
          `${run.component} の診断が実行実績を運んでいない` +
            `（dispatch ${run.dispatches} / lastRun ${run.ran}）`,
        );
      }
      // gpuTiming を要求していない GPU なので op 別時間は無いのが正
      // （有効化は acquireGpu({ gpuTiming: true }) を options.gpu へ渡す側の責務）。
      if (run.timed) {
        throw new Error(`${run.component} が計測なしの GPU で lastRunTiming を返した`);
      }
    }
  },
});
