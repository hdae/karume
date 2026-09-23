/**
 * 8 系列ぶんの「**最小の生成 1 回**」（RAM ピーク harness のパイプライン面）。
 *
 * 測りたいのはロードのピークなので、生成は**出力の質を問わない最短の 1 回**でよい。ただし
 * 0 回では測れない — anima / irodori / sbv2 は Session を生成時に張る（`fromPretrained` の
 * 決着では重みが GPU に載っていない部品がある）ので、最小の 1 回まで回して初めて「準備完了の
 * ピーク」が出る。
 *
 * 入力は**合成**する（`examples/` の実サンプルは画像 2 系列・音声 2 系列とも配布形と別の
 * ホスト資産で、機によって在ったり無かったりする）。ピークは入力の中身に依らず寸法だけで
 * 決まるので、合成で足りる — 代わりに寸法は実サンプルと同じ桁に合わせてある。
 *
 * MUST: 家族ごとの呼び分けはここ 1 箇所に閉じる（`measure.ts` は家族を知らない）。
 */

import type { DistributionSource, HubRepoRef } from "../../packages/hub/mod.ts";
import {
  AnimaPipeline,
  BirefnetPipeline,
  DepthAnythingPipeline,
  Gemma4Pipeline,
  IrodoriPipeline,
  type Rgb8Image,
  Sbv2Pipeline,
  type Sbv2Utterance,
  Siglip2Pipeline,
  toSbv2Utterance,
  VowelDetectorPipeline,
} from "../../packages/models/mod.ts";
import type { SessionDiagnostics } from "../../packages/runtime/mod.ts";

/** 対応系列（`--family` の受理集合）。 */
export const FAMILIES = [
  "anima",
  "gemma4",
  "irodori",
  "sbv2",
  "birefnet",
  "depth",
  "siglip2",
  "vowel",
] as const;

export type FamilyName = typeof FAMILIES[number];

export const isFamilyName = (value: string): value is FamilyName =>
  (FAMILIES as readonly string[]).includes(value);

/** 生成 1 回のノブ（家族ごとに効くものだけを読む）。 */
export type FamilyRunKnobs = {
  /** anima の denoise 段数。 */
  readonly steps: number;
  /** anima の解像度（正方）。 */
  readonly size: number;
  /** gemma4 の生成 token 数。 */
  readonly maxNewTokens: number;
};

export type FamilyLoadOptions = {
  readonly model?: string;
  readonly quant?: string;
  /** cold / warm で渡すディレクトリ固定の CacheStorage（local では渡さない）。 */
  readonly caches?: CacheStorage;
  /** 疑似 HF の実ポートを隠す `fetch`（cold / warm のみ）。 */
  readonly fetch?: typeof globalThis.fetch;
  /** 部品ごとの Session 診断（構築費と低精度格納の内訳）。 */
  readonly onDiagnostics: (component: string, diagnostics: SessionDiagnostics) => void;
  readonly knobs: FamilyRunKnobs;
};

/** ロード済みのパイプライン（`run` が最小の生成 1 回・`dispose` が後始末）。 */
export type LoadedFamily = {
  readonly run: () => Promise<void>;
  readonly dispose: () => Promise<void>;
};

/** `fromPretrained` へ渡す取得元。 */
export type FamilySource = string | HubRepoRef | DistributionSource;

/**
 * 合成画像（横方向のグラデーション）。寸法は実サンプルと同じ桁（前処理は配布形の宣言寸法へ
 * 伸縮するので、ここでの寸法は前処理の費用にしか効かない）。
 */
const syntheticImage = (width: number, height: number): Rgb8Image => {
  const data = new Uint8Array(width * height * 3);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const at = (y * width + x) * 3;
      data[at] = (x * 255 / Math.max(1, width - 1)) | 0;
      data[at + 1] = (y * 255 / Math.max(1, height - 1)) | 0;
      data[at + 2] = 128;
    }
  }
  return { data, width, height };
};

/** 合成音声（440Hz の正弦波・16kHz モノラル）。10ms フレームで 200 本ぶん。 */
const syntheticAudio = (sampleRate: number, seconds: number): Float32Array => {
  const samples = new Float32Array(Math.round(sampleRate * seconds));
  for (let at = 0; at < samples.length; at += 1) {
    samples[at] = Math.sin(2 * Math.PI * 440 * at / sampleRate) * 0.25;
  }
  return samples;
};

/**
 * 合成の発話（「テスト」3 モーラ）。
 *
 * 形態素解析（`@hdae/yomi`）を通さないのは、辞書が別途 network / Cache API を踏むためである —
 * harness が測っているホスト RAM と digest の計数に、測る対象でないものを混ぜない。`words` の
 * 音素列とモーラの音素列は一致させてある（`sum(word2ph) === given_phone 長` の門）。
 */
const syntheticUtterance = (): Sbv2Utterance =>
  toSbv2Utterance({
    result: {
      leadingPunctuations: [],
      accentPhrases: [{
        moras: [
          { kana: "テ", consonant: "t", vowel: "e" },
          { kana: "ス", consonant: "s", vowel: "u" },
          { kana: "ト", consonant: "t", vowel: "o" },
        ],
        accentNucleus: 1,
        punctuations: [],
      }],
    },
    words: [{ surface: "テスト", phones: ["t", "e", "s", "u", "t", "o"] }],
  });

/** anima / irodori の生成に渡す固定 seed（比較可能にするためだけ — 値そのものに意味は無い）。 */
const SEED = 1;

/** 画像 3 系列に渡す合成画像の寸法。 */
const IMAGE_SIZE = 512;

/** vowel-detector に渡す合成音声（16kHz・2.0 秒 = 200 フレーム）。 */
const AUDIO_SECONDS = 2;

/**
 * 系列 1 つを読み込む。返る {@link LoadedFamily.run} を呼ぶまで生成は起きない
 * （`load` 区間と `run` 区間を呼び手が分けられるようにするため）。
 */
export const loadFamily = async (
  family: FamilyName,
  from: FamilySource,
  options: FamilyLoadOptions,
): Promise<LoadedFamily> => {
  const selection = {
    ...(options.model === undefined ? {} : { model: options.model }),
    ...(options.quant === undefined ? {} : { quant: options.quant }),
    ...(options.caches === undefined ? {} : { caches: options.caches }),
    ...(options.fetch === undefined ? {} : { fetch: options.fetch }),
  };
  const { onDiagnostics, knobs } = options;

  if (family === "anima") {
    const pipeline = await AnimaPipeline.fromPretrained(from, {
      ...selection,
      onRunDiagnostics: onDiagnostics,
    });
    return {
      run: async () => {
        await pipeline.generate({
          prompt: "1girl, solo, upper body",
          steps: knobs.steps,
          resolution: { width: knobs.size, height: knobs.size },
          seed: SEED,
        });
      },
      dispose: () => pipeline.dispose(),
    };
  }

  if (family === "gemma4") {
    const pipeline = await Gemma4Pipeline.fromPretrained(from, {
      ...selection,
      // gemma4 の区間は prefill / decode の 2 つ（`kind` が部品名の代わり）。
      onRunDiagnostics: (diagnostics, phase) => onDiagnostics(phase.kind, diagnostics),
    });
    return {
      run: async () => {
        await pipeline.chat([{ role: "user", content: "こんにちは" }], {
          maxNewTokens: knobs.maxNewTokens,
        }).text();
      },
      dispose: () => pipeline.dispose(),
    };
  }

  if (family === "irodori") {
    const pipeline = await IrodoriPipeline.fromPretrained(from, {
      ...selection,
      onRunDiagnostics: onDiagnostics,
    });
    return {
      run: async () => {
        await pipeline.generate({ text: "これはテストです。", seed: SEED });
      },
      dispose: () => pipeline.dispose(),
    };
  }

  if (family === "sbv2") {
    const pipeline = await Sbv2Pipeline.fromPretrained(from, {
      ...selection,
      onRunDiagnostics: onDiagnostics,
    });
    const utterance = syntheticUtterance();
    return {
      run: async () => {
        await pipeline.generate(utterance, { seed: SEED });
      },
      dispose: () => pipeline.dispose(),
    };
  }

  if (family === "birefnet") {
    const pipeline = await BirefnetPipeline.fromPretrained(from, {
      ...selection,
      onRunDiagnostics: (diagnostics) => onDiagnostics("birefnet", diagnostics),
    });
    const image = syntheticImage(IMAGE_SIZE, IMAGE_SIZE);
    return {
      run: async () => {
        await pipeline.segment(image);
      },
      dispose: () => pipeline.dispose(),
    };
  }

  if (family === "depth") {
    const pipeline = await DepthAnythingPipeline.fromPretrained(from, {
      ...selection,
      onRunDiagnostics: (diagnostics) => onDiagnostics("depth-anything", diagnostics),
    });
    const image = syntheticImage(IMAGE_SIZE, IMAGE_SIZE);
    return {
      run: async () => {
        await pipeline.estimate(image);
      },
      dispose: () => pipeline.dispose(),
    };
  }

  if (family === "siglip2") {
    const pipeline = await Siglip2Pipeline.fromPretrained(from, {
      ...selection,
      onRunDiagnostics: (diagnostics) => onDiagnostics("siglip2", diagnostics),
    });
    const image = syntheticImage(IMAGE_SIZE, IMAGE_SIZE);
    return {
      run: async () => {
        await pipeline.embed(image);
      },
      dispose: () => pipeline.dispose(),
    };
  }

  const pipeline = await VowelDetectorPipeline.fromPretrained(from, {
    ...selection,
    onRunDiagnostics: (diagnostics) => onDiagnostics("vowel-detector", diagnostics),
  });
  const audio = syntheticAudio(pipeline.sampleRate, AUDIO_SECONDS);
  return {
    run: async () => {
      await pipeline.detect(audio);
    },
    dispose: () => pipeline.dispose(),
  };
};
