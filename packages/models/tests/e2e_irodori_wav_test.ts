/**
 * Irodori の **WAV sha256 門**（実 GPU）。配布形 → `IrodoriPipeline.fromAssets` → `generate` →
 * `encodeWav` まで通し、出力 WAV の sha256 が参照値と**ビット一致**するかだけを見る
 * （SBV2 の `e2e_sbv2_wav_test.ts` / Anima の PNG 門と同じ哲学の Irodori 版）。
 *
 * ## 既存の Irodori 門との分担（役割が違うので併存する）
 *
 * - latent 門（`e2e_irodori_latent_test.ts`）— **上流との一致**を見る。golden の初期ノイズを
 *   注入して z を tolerance 付きで突き合わせ、S と forward 数だけを完全一致で縛る。torch と
 *   WGSL の縮約順序差が構造的に残るので、そもそもビット一致は原理的に出ない。
 * - タイル同値門（`e2e_irodori_codec_test.ts`）— decode の**性能ノブが出力を変えない**こと。
 * - 参照前処理 parity 門（`e2e_irodori_reference_test.ts`）— 参照音声 → latent の鎖と上流の一致。
 * - **この門** — 上流は一切見ない。テキストから波形までの鎖ぜんぶ（内部 `Randn` の乱数列 /
 *   duration の S 決定 / Euler + CFG / 末尾トリム / codec タイル decode / 秒切り出し /
 *   `encodeWav` の量子化）を、**この実行環境での回帰検出器として 1 ビットで**縛る。上の 3 門が
 *   全部緑でも、ホストのグルー 1 行が変われば波形は動く — それを掴む最終門。
 *
 * ## MUST: 割れたら tolerance 化せず、原因を特定してから digest を更新する
 *
 * 数値が 1 bit でも動いたら鎖のどこかが変わっている。**tolerance 化も参照値の差し替えも禁止**
 * で、赤のまま止めて差分（WAV バイト長 / サンプル数 / S / forward 数 / 実効ノブ / 先頭差分位置 /
 * 実物の WAV）を出す。digest を書き換えてよいのは「何が変わったか」を先に言えたときだけ。
 *
 * 更新の手順（原因を特定した後で）: ①この門を赤のまま走らせ、ログの `sha256 <hex>` を
 * {@link CASES} の該当ケースへ書き戻す ②**別プロセスで 2 回**走らせて一致を確認する
 * ③何が変わって焼き直したのかをコミットメッセージに書く。
 *
 * ## 参照 digest はこの参照環境専用（クロスデバイスのビット同一は保証しない）
 *
 * 参照値は 2026-08-12 に RTX 3080 Ti / Linux / Vulkan (wgpu) で実測し、**別プロセスで 2 回**
 * 焼いて一致を確認したもの。他バックエンド（Metal 等）では一致しないのが仕様で、その機序と
 * 別バックエンドでの健全性検証の作法（自己 A/B）は [limitations](../../../docs/limitations.md)
 * の「sha256 参照門は参照環境専用」節にある。
 *
 * ## ケース 2 本
 *
 * - **voice-clone** — 参照音声 + caption 付き。`{ audio }` 経路（120 秒切り詰め → LUFS −16 →
 *   reflect pad → `codec_encoder`）を含む正本ケースで、CFG 3 本が立つ（100 forward / S 170 /
 *   6.80 秒の波形）。生成 + WAV 化に **13.5 秒**（参照環境・2026-08-12）。
 * - **no-ref** — 参照なし / caption なし。speaker のゼロ短絡と caption の CFG off が効く経路
 *   （60 forward / S 116 / 4.64 秒の波形）。**7.6 秒**。
 *
 * テスト 1 本ぶんの実時間は約 25 秒 — 上の 21 秒に、3.3GB の配布形を読んで `openModel` する
 * ぶんが乗る（2 ケースは 1 本のパイプラインを共有してこれを 1 回に抑えている）。
 *
 * どちらも `initialNoise` は渡さない — 内部 `Randn` の決定論ごと縛るのがこの門の役目
 * （golden ノイズを注入する latent 門との一番の違い）。ノブも 1 つも渡さないので、実効値は
 * 配布形の `pipelineConfig` がそのまま（{@link REFERENCE_KNOBS} で drift を先に検査する）。
 *
 * MUST: 資産は配布形 `models/karume-irodori-v4-small/` と参照音声
 * `inputs/irodori/v4-small/samples/clone_ref1.wav`（どちらも untracked・実 GPU 機のローカル
 * 資産）。欠けた環境と GPU 無し環境は理由を出して**明示 SKIP** する（テストを消して無音で
 * 緑にしない — ADR 0005）。
 *
 * NOTE: ローカル読みに `Deno.readFile` を使うのはテストだけ。パッケージ本体は Web 標準 API
 * のみ（横断不変条件）で、`fromAssets` はバイト列を受け取るだけの面。
 */

import { assertEquals, assertFalse, assertRejects, assertStrictEquals } from "@std/assert";
import { parseManifest, resolveFiles } from "@karume/hub";
import type { Manifest, ModelEntry } from "@karume/hub";
import type { SessionDiagnostics } from "@karume/runtime";
import {
  type DecodedWav,
  decodeWav,
  encodeWav,
  type IrodoriGeneratedAudio,
  type IrodoriGenerateEvent,
  type IrodoriGenerateRequest,
  IrodoriPipeline,
  type IrodoriRunComponent,
} from "../mod.ts";
import { type IrodoriPipelineConfig, parseIrodoriPipelineConfig } from "../src/irodori/config.ts";
import { tSchedule } from "../src/irodori/host/sampler.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/** 配布形の置き場（`karume dist --pipeline irodori` の既定の出力先）。 */
const ASSETS_DIR = new URL("../../../models/karume-irodori-v4-small/", import.meta.url);
/** 参照音声（手置きの実資産 — 48kHz mono PCM16）。 */
const REFERENCE_AUDIO = new URL(
  "../../../inputs/irodori/v4-small/samples/clone_ref1.wav",
  import.meta.url,
);
/** 実行日（モジュールロード時に 1 回だけ確定 — ダンプ先の日付ディレクトリに使う）。 */
const TODAY = new Date().toISOString().slice(0, 10);
/** ミスマッチ時の実物ダンプ先（`rm -rf outputs/bench` で常に安全に消せる — docs/assets-layout.md）。 */
const OUTPUTS_DIR = new URL(
  `../../../outputs/bench/karume-irodori-v4-small/${TODAY}_e2e-mismatch/`,
  import.meta.url,
);

/** SKIP 時にそのまま貼れる生成コマンド。 */
const DIST_COMMAND = "cd tools/exporter && uv run karume dist --pipeline irodori";

const MODEL = "v4-small";
const QUANT = "f32";

/** `encodeWav` が書く RIFF ヘッダ長（先頭差分の位置をサンプル番号へ直すのに要る）。 */
const HEADER_BYTES = 44;

/** 門 1 本ぶんの条件と参照値。 */
type WavCase = {
  readonly name: string;
  /** ステップ名に出す「この 1 本で何を通しているか」。 */
  readonly why: string;
  readonly text: string;
  readonly caption?: string;
  /** 参照音声を `{ audio }` で渡すか（wav 経路 = LUFS 正規化 + `codec_encoder` を含めるか）。 */
  readonly withReference: boolean;
  readonly seed: number;
  /** 参照 digest（**変更禁止** — 上の MUST）。 */
  readonly sha256: string;
};

/**
 * ケース 2 本。テキストと caption は full-loop golden の `meta.json` から借りている（同じ文で
 * 比べられるほうが、割れたときに latent 門の実測と突き合わせやすい）。ただし**この門は golden
 * を読まない** — 参照 digest は波形まで通した実測で、golden の生成条件とは独立に閉じる。
 */
const CASES: readonly WavCase[] = [
  {
    name: "voice-clone",
    why: "参照音声 + caption あり（wav 経路 = LUFS 正規化 + codec_encoder 込み・CFG 3 本）",
    text: "今日は近くの店まで歩いて行きました。とても良い天気でしたね。",
    caption:
      "若く元気な女性の声。カフェの店員のように、明るくハキハキとした少し高めのトーンで話している。",
    withReference: true,
    seed: 1234,
    sha256: "05f82a9cd8bbb8055fee47b4599840097f62d3d07252c50e1a7ef7fe21631e5e",
  },
  {
    name: "no-ref",
    why: "参照なし / caption なし（speaker のゼロ短絡と caption の CFG off が効く経路）",
    text: "本日はお越しいただき、誠にありがとうございます。",
    withReference: false,
    seed: 1235,
    sha256: "5cc43d7fe9ae0a8733e0b06088948e33f947cdabe0934fea3988622530da3e8d",
  },
];

/**
 * 参照 digest を焼いたときの実効ノブ（配布形の `pipelineConfig` と一致するはず）。
 *
 * 波形を決める数だけを写す — サンプラ（`steps` / `initScale` / CFG の強さと区間）と、latent →
 * 波形の幾何（`frameRate` / `sampleRate` / `hopLength` / `codecHaloFrames`）。
 */
const REFERENCE_KNOBS = {
  steps: 40,
  initScale: 0.999,
  cfgMinT: 0.5,
  cfgMaxT: 1,
  cfgText: 3,
  cfgSpeaker: 5,
  cfgCaption: 3,
  frameRate: 25,
  sampleRate: 48000,
  hopLength: 1920,
  codecHaloFrames: 8,
} as const;

const manifestText = await Deno.readTextFile(new URL("karume.json", ASSETS_DIR)).catch(
  () => undefined,
);
const referenceBytes = await Deno.readFile(REFERENCE_AUDIO).catch(() => undefined);
const ASSETS_AVAILABLE = manifestText !== undefined && referenceBytes !== undefined;
if (!ASSETS_AVAILABLE) {
  console.warn(
    `[karume] Irodori の WAV sha256 門を SKIP する（配布形 ${ASSETS_DIR.pathname} と参照音声 ` +
      `${REFERENCE_AUDIO.pathname} の両方が要る）。` +
      (manifestText === undefined
        ? `生成: ${DIST_COMMAND}`
        : "参照音声は手置きの実資産（docs/assets-layout.md の inputs 根）"),
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

/**
 * 門の**前提**を検査する（門そのものではない）。参照 digest は配布形の宣言どおりのノブで
 * 焼かれているので、宣言が動いた配布形に対しては「sha256 が違う」ではなく「条件が違う」と
 * 言って落とす（配布形を焼き直して digest が割れたとき、原因が資産側か実装側かを先に言う）。
 */
const assertReferenceKnobs = (entry: ModelEntry): IrodoriPipelineConfig => {
  const config = parseIrodoriPipelineConfig(entry.pipelineConfig);
  const drifted: string[] = [];
  const check = (key: string, expected: number, actual: number): void => {
    if (expected !== actual) drifted.push(`${key} ${expected} → ${actual}`);
  };
  check("steps", REFERENCE_KNOBS.steps, config.steps);
  check("initScale", REFERENCE_KNOBS.initScale, config.initScale);
  check("cfgMinT", REFERENCE_KNOBS.cfgMinT, config.cfgMinT);
  check("cfgMaxT", REFERENCE_KNOBS.cfgMaxT, config.cfgMaxT);
  check("cfgScales.text", REFERENCE_KNOBS.cfgText, config.cfgScales.text);
  check("cfgScales.speaker", REFERENCE_KNOBS.cfgSpeaker, config.cfgScales.speaker);
  check("cfgScales.caption", REFERENCE_KNOBS.cfgCaption, config.cfgScales.caption);
  check("frameRate", REFERENCE_KNOBS.frameRate, config.frameRate);
  check("sampleRate", REFERENCE_KNOBS.sampleRate, config.sampleRate);
  check("hopLength", REFERENCE_KNOBS.hopLength, config.hopLength);
  check("codecHaloFrames", REFERENCE_KNOBS.codecHaloFrames, config.codecHaloFrames);
  if (drifted.length > 0) {
    throw new Error(
      `配布形の pipelineConfig が参照 digest の生成条件と違う: ${drifted.join(" / ")}\n` +
        "（門の前提が崩れている — 参照 sha256 を差し替えるのではなく、条件が変わった理由を先に確かめる）",
    );
  }
  return config;
};

/** 実効ノブの 1 行表示（突合が割れたときに「何を焼いたか」を報告へ残す）。 */
const describeKnobs = (item: WavCase, config: IrodoriPipelineConfig): string =>
  `model ${MODEL} / quant ${QUANT} / seed ${item.seed} / text ${JSON.stringify(item.text)} / ` +
  `caption ${JSON.stringify(item.caption ?? "")} / 参照音声 ${
    item.withReference ? "あり" : "なし"
  } / ` +
  `steps ${config.steps} / initScale ${config.initScale} / ` +
  `cfg ${config.cfgScales.text}/${config.cfgScales.speaker}/${config.cfgScales.caption} ` +
  `on [${config.cfgMinT}, ${config.cfgMaxT}] / ${config.frameRate}fps / ` +
  `${config.sampleRate}Hz / hop ${config.hopLength} / halo ${config.codecHaloFrames}`;

/**
 * 参照 WAV の**実体**があれば置かれている場所（任意 — 参照環境で焼いた実物を人がここへ置く）。
 * 門は sha256 だけで閉じており、ここは**先頭差分位置を出すためだけ**の材料。
 */
const referenceWavPath = (item: WavCase): URL =>
  new URL(`e2e-irodori-${item.name}-reference.wav`, OUTPUTS_DIR);

/**
 * 参照 WAV の実体が手元にあれば**先頭差分位置と近傍値**まで出す。
 *
 * MUST: 実体の sha256 を確かめてから使う — 別の条件で焼かれた WAV を参照として差分を出すと、
 * 診断そのものが嘘になる。
 */
const describeFirstDifference = async (
  item: WavCase,
  actual: Uint8Array<ArrayBuffer>,
): Promise<string> => {
  const path = referenceWavPath(item);
  const reference = await Deno.readFile(path).catch(() => undefined);
  if (reference === undefined) {
    return `参照 WAV の実体が無い（${path.pathname}）— 先頭差分位置は出せない`;
  }
  const sha = await sha256Hex(reference);
  if (sha !== item.sha256) {
    return `${path.pathname} は参照 WAV ではない（sha256 ${sha}）— 先頭差分位置は出せない`;
  }
  const shared = Math.min(reference.length, actual.length);
  let at = 0;
  while (at < shared && reference[at] === actual[at]) at += 1;
  if (at === shared) {
    return `先頭 ${shared} バイトは一致（長さだけが違う: 参照 ${reference.length} / 実測 ${actual.length}）`;
  }
  if (at < HEADER_BYTES) {
    return `先頭差分は RIFF ヘッダ内（オフセット ${at}）: バイト ${reference[at]} → ${actual[at]}`;
  }
  const i16 = (bytes: Uint8Array<ArrayBuffer>, sample: number): number =>
    new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
      .getInt16(HEADER_BYTES + sample * 2, true);
  const sample = Math.floor((at - HEADER_BYTES) / 2);
  const last = Math.floor((shared - HEADER_BYTES) / 2) - 1;
  const neighbours: string[] = [];
  for (let index = Math.max(0, sample - 2); index <= Math.min(last, sample + 2); index += 1) {
    neighbours.push(
      `${index === sample ? "*" : ""}${index} ${i16(reference, index)} → ${i16(actual, index)}`,
    );
  }
  return `先頭差分は サンプル ${sample}（オフセット ${at}）/ 近傍 i16 ${neighbours.join(" , ")}`;
};

/**
 * 参照 sha と食い違ったときの報告。
 *
 * MUST: ここで tolerance に逃げない。実物を {@link OUTPUTS_DIR} へ落として、人が聴き比べ・
 * 突き合わせできる形にする。
 */
const mismatchReport = async (
  item: WavCase,
  config: IrodoriPipelineConfig,
  audio: IrodoriGeneratedAudio,
  wav: Uint8Array<ArrayBuffer>,
  actual: string,
): Promise<string> => {
  await Deno.mkdir(OUTPUTS_DIR, { recursive: true });
  const dumped = new URL(`e2e-irodori-${item.name}-mismatch.wav`, OUTPUTS_DIR);
  await Deno.writeFile(dumped, wav);
  return `${item.name}: 出力 WAV の sha256 が参照と一致しない\n` +
    `  期待 ${item.sha256}\n  実際 ${actual}\n` +
    `  WAV ${wav.length} バイト / サンプル ${audio.data.length} / ${audio.sampleRate}Hz / ` +
    `${(audio.data.length / audio.sampleRate).toFixed(3)}s / S ${audio.frames} / ` +
    `forwards ${audio.forwards}\n` +
    `  実効ノブ ${describeKnobs(item, config)}\n` +
    `  ${await describeFirstDifference(item, wav)}\n` +
    `  実物 ${dumped.pathname}`;
};

/**
 * 1 ケースを生成して digest を突き合わせる。
 *
 * `onEvent` を渡した呼びも**同じ参照値**で突き合わせる — 観測席が数値に 1 ビットも触って
 * いないこと、そして常駐経路 / ホスト経路の出力が同一であることの直接証拠になる
 * （下の onEvent 門）。
 */
const runCase = async (
  pipeline: IrodoriPipeline,
  config: IrodoriPipelineConfig,
  item: WavCase,
  reference: DecodedWav,
  onEvent?: (event: IrodoriGenerateEvent) => void,
): Promise<IrodoriGeneratedAudio> => {
  // `initialNoise` もノブも渡さない — 内部 `Randn` と `pipelineConfig` の既定ごと縛る。
  const request: IrodoriGenerateRequest = {
    text: item.text,
    ...(item.caption === undefined ? {} : { caption: item.caption }),
    ...(item.withReference ? { speaker: { audio: reference } } : {}),
    seed: item.seed,
    ...(onEvent === undefined ? {} : { onEvent }),
  };
  const started = performance.now();
  const audio = await pipeline.generate(request);
  const wav = encodeWav(audio.data, audio.sampleRate);
  const actual = await sha256Hex(wav);
  const elapsed = ((performance.now() - started) / 1000).toFixed(1);
  console.log(
    `[e2e] irodori ${MODEL}/${QUANT} ${item.name}${
      onEvent === undefined ? "" : "(onEvent)"
    }: ${elapsed}s / S ${audio.frames} / ` +
      `forwards ${audio.forwards} / WAV ${wav.length}B / ` +
      `${(audio.data.length / audio.sampleRate).toFixed(2)}s / sha256 ${actual}`,
  );
  if (actual !== item.sha256) {
    throw new Error(await mismatchReport(item, config, audio, wav, actual));
  }
  return audio;
};

Deno.test({
  name: `e2e(実GPU): 配布形 ${MODEL} / quant ${QUANT} の WAV が参照 sha256 と一致する`,
  ignore: !RUNNABLE,
  fn: async (t) => {
    const manifest = readManifest();
    // 門の前提を先に見る（実効ノブが動いていたら、生成する前に「条件が違う」と言う）。
    const config = assertReferenceKnobs(modelEntry(manifest));
    const assets = await loadLocalAssets(manifest);
    // 参照音声は 48kHz mono PCM16。周波数が配布形と違えば `generate` が fail loudly（リサンプル
    // は持たない）ので、ここでは読むだけにする。
    const reference = decodeWav(referenceBytes as Uint8Array<ArrayBuffer>);
    // run 1 回ごとの観測。`lastRun`（アリーナ実績）の有無が「ホスト経路か常駐経路か」の
    // **独立な証拠**になる — 常駐経路の `enqueue` はアリーナも計測窓も作らないので undefined。
    const runs: { component: IrodoriRunComponent; hasArena: boolean }[] = [];
    // `await using` は [Symbol.asyncDispose] 経由の解放をこの実 GPU 経路で検査する意図込み。
    // 3.3GB の資産を 2 度読まないよう、全ケースは 1 本のパイプラインを共有する。
    await using pipeline = await IrodoriPipeline.fromAssets({ manifest, assets }, {
      model: MODEL,
      quant: QUANT,
      onRunDiagnostics: (component: IrodoriRunComponent, diagnostics: SessionDiagnostics) =>
        runs.push({ component, hasArena: diagnostics.lastRun !== undefined }),
    });
    for (const item of CASES) {
      await t.step(`${item.name}: ${item.why}`, async () => {
        await runCase(pipeline, config, item, reference);
      });
    }

    // --- onEvent（生成イベントの観測席）------------------------------------
    //
    // 押さえるのは 4 点:
    //  ① `onEvent` を渡すと DiT ループが**ホスト経路**へ切り替わる（常駐経路は 1 batch +
    //     単一フェンスで途中を観測できない — ADR 0054）。証拠は診断側の `lastRun` の有無で、
    //     イベントが届いたこと自体とは独立に採る。
    //  ② それでも **WAV の sha256 は上の voice-clone と同じ値**（2 経路ビット同一の MUST が
    //     公開 API から観測できる唯一の場所）。購読側は毎 step 写しを NaN で壊してすらいる。
    //  ③ `denoise-step` は **step 単位**で `config.steps` 回（CFG の内側の forward 数ではない）。
    //  ④ コールバックの throw が生成ごと落とす（step 粒度の中断手段）。
    const withEvents = CASES[0];
    await t.step(
      `onEvent: ホスト経路でも WAV sha が同一（step ${REFERENCE_KNOBS.steps} 回の発火）`,
      async () => {
        const log: string[] = [];
        const shapes = new Set<string>();
        const lengths = new Set<number>();
        let firstStep: (() => { data: Float32Array<ArrayBuffer> }) | undefined;
        let firstStepHead: number[] = [];
        const before = runs.length;
        const audio = await runCase(pipeline, config, withEvents, reference, (event) => {
          if (event.kind === "stage") {
            log.push(`stage:${event.component}:${event.at}`);
            return;
          }
          log.push(`step:${event.step}/${event.steps}@${event.t}`);
          const snapshot = event.copyLatents();
          shapes.add(snapshot.shape.join("x"));
          lengths.add(snapshot.data.length);
          // 写しを壊す。内部配列を渡していたら以後の step が NaN 汚染され、WAV sha が割れる
          // （波形の非有限検査が先に落とす）。
          snapshot.data.fill(Number.NaN);
          if (event.step === 1) {
            firstStep = event.copyLatents;
            firstStepHead = Array.from(event.copyLatents().data.slice(0, 8));
          }
        });

        // ① ホスト経路の証拠（dit の run が全てアリーナを通っている = `enqueue` ではない）。
        const ditRuns = runs.slice(before).filter((run) => run.component === "dit");
        assertEquals(ditRuns.length, audio.forwards, "dit の run 数と forwards が合わない");
        assertFalse(
          ditRuns.some((run) => !run.hasArena),
          "dit の run に lastRun が無い = 常駐経路のまま（onEvent でホスト経路へ切り替わっていない）",
        );

        // ③ イベント列が実行構造と完全一致する（段の前後 + step 単位の発火）。
        const schedule = tSchedule(config.steps, config.initScale);
        assertEquals(
          log,
          [
            "stage:backbone:start",
            "stage:backbone:end",
            "stage:text-proj:start",
            "stage:text-proj:end",
            "stage:caption-proj:start",
            "stage:caption-proj:end",
            "stage:codec-encoder:start",
            "stage:codec-encoder:end",
            "stage:speaker:start",
            "stage:speaker:end",
            "stage:duration:start",
            "stage:duration:end",
            "stage:dit:start",
            ...Array.from(
              { length: config.steps },
              (_, at) => `step:${at + 1}/${config.steps}@${schedule[at]}`,
            ),
            "stage:dit:end",
            "stage:codec-decoder:start",
            "stage:codec-decoder:end",
          ],
          "イベント列が実行構造と合わない（段の前後 / step 数 / t の値）",
        );
        // CFG の内側は 1 step で 3 forward 走る — 発火は step 単位であって forward 単位ではない。
        assertFalse(
          audio.forwards <= config.steps,
          `forwards ${audio.forwards} が steps ${config.steps} 以下（CFG が立っていない条件では` +
            "「step 単位である」ことの証明にならない)",
        );

        // ② 途中潜在の形は `[frames, latentDim]`。
        assertEquals(shapes, new Set([`${audio.frames}x${config.latentDim}`]));
        assertEquals(lengths, new Set([audio.frames * config.latentDim]));
        // 生成が終わった後に呼んでも step 1 の値が返る（lazy だが束縛は作った時点）。
        assertEquals(
          Array.from((firstStep as () => { data: Float32Array<ArrayBuffer> })().data.slice(0, 8)),
          firstStepHead,
          "生成後に呼んだ copyLatents が step 1 の値を返さない",
        );
      },
    );

    await t.step("onEvent: コールバックの throw は生成ごと落とす（step 粒度の中断）", async () => {
      const seen: string[] = [];
      const abortAt = 2;
      await assertRejects(
        () =>
          pipeline.generate({
            text: CASES[1].text,
            seed: CASES[1].seed,
            onEvent: (event) => {
              seen.push(
                event.kind === "stage"
                  ? `stage:${event.component}:${event.at}`
                  : `step:${event.step}`,
              );
              if (event.kind === "denoise-step" && event.step === abortAt) {
                throw new Error("購読側で中断");
              }
            },
          }),
        Error,
        "購読側で中断",
      );
      // 例外を握らないので DiT の段で止まる — codec は 1 度も開かない。
      assertEquals(seen.at(-1), `step:${abortAt}`, "中断した step が最後のイベントでない");
      assertFalse(seen.includes("stage:dit:end"), "中断したのに段が正常終了している");
      assertFalse(
        seen.some((entry) => entry.startsWith("stage:codec-decoder")),
        "中断したのに codec decode へ入っている",
      );
      // この後の `await using` の解放が、中断した生成の後でも正常に効くことの検査になる。
    });

    // --- dispose の直列化（flush-before-destroy）------------------------------
    //
    // `concurrency/serial.ts` は `serial_test.ts` が独立に縛っているが、`IrodoriPipeline` が
    // その鎖へ正しく載っていることは公開面からしか見えない。**この step は最後に置く**
    // （ここでパイプラインを解放するため）。
    await t.step("dispose は in-flight の完了を待ち、2 度目も同じ完了を返す", async () => {
      const order: string[] = [];
      const running = pipeline.generate({ text: CASES[1].text, seed: CASES[1].seed })
        .then(() => {
          order.push("generate");
        });
      const disposal = pipeline.dispose();
      assertStrictEquals(pipeline.dispose(), disposal, "2 度目の dispose が別の完了を返す");
      await Promise.all([running, disposal.then(() => order.push("dispose"))]);
      // 破棄が先に解決すると、消費側は「解放済み」と見なして次へ進む（GPU はまだ生成中）。
      assertEquals(order, ["generate", "dispose"]);

      // dispose 済みの判定は**鎖に入る前**（呼び出し時点）— 先行生成の決着を待たずに落ちる。
      // `async` メソッドなので観測は reject（同期 throw にはならない）。
      await assertRejects(
        () => pipeline.generate({ text: CASES[1].text }),
        Error,
        "dispose 済みでは生成できない",
      );
      await assertRejects(
        () => pipeline.generateLatent({ text: CASES[1].text }),
        Error,
        "dispose 済みでは生成できない",
      );
    });
  },
});
