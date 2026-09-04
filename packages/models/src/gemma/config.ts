/**
 * gemma4 の `pipelineConfig` スキーマ検証（ADR 0038 §1 — スキーマは各パイプライン実装が所有・
 * 検証）。
 *
 * hub は `pipelineConfig` を素通しする（禁止キーの一掃と規模上限だけを見る）ので、**形の正本は
 * このモジュール**で、手書きの検査を全て parse 時に走らせる。焼く側の正本は
 * `tools/export-recipes/gemma4/distribution.py`（`gemma4_pipeline_config` / `gemma4_sampler`）。
 *
 * MUST: 未知キーは fail loudly（`{ capacity: 640, capasity: 640 }` のような綴り違いが黙って
 * 既定へ縮退すると、配布者の意図した宣言と実行が食い違ったまま気づけない）。
 * MUST: マップは `Object.hasOwn` 経由でのみ引く（横断不変条件）。
 *
 * NOTE: 公開配布リポの対応表（{@link GEMMA4_SOURCES}）もここに置く。manifest から導ける値では
 * なく「どの manifest を取りに行くか」の側なので、配布形が持てない（ADR 0073）。
 */

import type { HubRepoRef } from "@karume/hub";

import {
  assertGemma4RopeSpec,
  GEMMA4_ROPE_LAYER_TYPES,
  type Gemma4RopeLayerSpec,
  type Gemma4RopeLayerType,
  type Gemma4RopeSpec,
} from "./rope.ts";

/** `pipeline` の契約名と、この実装が受け付ける major（ADR 0038 §1）。 */
export const GEMMA4_PIPELINE_NAME = "gemma4";
export const GEMMA4_PIPELINE_MAJOR = 1;

/**
 * Gemma 4 ファミリの**公開配布リポ対応表**（ADR 0092 — 家族 1 つにつき 1 表・**既定の席は
 * 無い**）。値は**このパッケージ版が検証した取得元**（pin 済み commit SHA — ADR 0073）。
 *
 * キーは HF リポ名の basename から `karume-` を落とした綴り（`"karume-" + key` がリポ名の
 * basename に戻る — この不変条件は `tests/sources_test.ts` の門が見る）。同一家族 = 1 リポなので、
 * 将来の E4B / 12B もこの 1 本（`"gemma4"`）に同居する（既定以外は `{ model }` で選ぶ）。
 *
 * **パッケージ版に合わせて自動追従したい場合のオプトイン**として渡す — 再現性を自分で
 * 固定したい場合は、この表ではなく自分の `{ repo, revision }` を書く（`fromPretrained` に
 * 既定は無い）。
 *
 * MUST: revision は commit SHA で固定する — ブランチ・タグは配布側で付け替えられるので、
 * 公開済みのこのパッケージが読むバイト列がネットワーク側の都合で黙って変わる（回復不能側の
 * 事故）。SHA 指定は revision 解決要求そのものを消すため、完全キャッシュ時のオフライン起動も
 * 同時に成立する（ADR 0038）。main 追従が要る利用者は
 * `{ ...GEMMA4_SOURCES["gemma4"], revision: "main" }` を明示的に選ぶ。
 */
// NOTE: revision はリリース手順書（docs/release-runbook.md）§3 で、アップロード後の main の
// SHA に更新する（ADR 0073 決定 3 の維持義務を継承 — 手書き + 手順書ゲート）。
export const GEMMA4_SOURCES = {
  "gemma4": {
    repo: "hdae/karume-gemma4",
    revision: "94d6222bb96ab1b84ede787dd93083bc7e0261dc",
  },
} as const satisfies Record<string, HubRepoRef>;

/**
 * 配布形が宣言できる sampler の既定（`SamplerSpec` の**部分集合**）。
 *
 * MUST: `SamplerSpec` 全体を開けない — `logitBias` / `seed` /
 * `repetitionPenalty` は「配布者が推奨する」性質の値ではない（上流 `generation_config.json` が
 * 持つのもこの 3 つだけ）。型を `SamplerSpec` のままにすると**受理集合より広い型**になり、
 * 「型は通るのにパーサが未知キーで落とす」欄が公開面に生える。欄を増やすのは実需が出てから。
 *
 * MUST: 3 欄とも必須（{@link parseGemma4PipelineConfig} は欄の欠落を落とす）。部分宣言を
 * 許すと「温度だけ推奨・top-k は低層の既定」という**上流のどこにも無い**組み合わせが生まれる。
 */
export type Gemma4DefaultSampler = {
  /** softmax の前に logits を割る温度（0 以上の有限数）。 */
  readonly temperature: number;
  /** 候補を確率上位 `topK` 件に絞る（1 以上の整数）。 */
  readonly topK: number;
  /** 累積確率が `topP` に達するまでの最小の候補集合に絞る（0 < topP ≤ 1）。 */
  readonly topP: number;
};

/**
 * 静的配線のうち**資産世代ごとに動く数**（配布形の `pipelineConfig` が宣言する）。
 *
 * NOTE: 記号（full スロットの容量記号）はここに置かない — グラフから導出できるものを宣言に
 * 二重持ちすると、片方だけ古びる（`pipeline.ts` の `capacitySymbolOf`）。
 */
export type Gemma4PipelineConfig = {
  /**
   * 固定長 prefill chunk の行数の**既定**（ADR 0066 決定 4 — context の計画時定数）。
   *
   * 実行時ノブでもある（`Gemma4PipelineOptions.chunkLength`）— グラフの chunk 行は記号 `M` なので、
   * 資産を焼き直さずに選べる（上限は {@link maxChunkLength}）。
   */
  readonly chunkLength: number;
  /**
   * `chunkLength` に選べる値の**上限**（記号 `M` を焼いた trace 範囲の上端）。
   *
   * MUST: 必須。IR の `symbols` は名前の列だけで記号の上限を持たない（`format/ir.ts`）ので、
   * これは**資産から導けない唯一の数**である。宣言が無いと、trace 範囲の外の `chunkLength` は
   * 例外なく素通りして走る（値が壊れるとは限らないが保証の外 — 横断不変条件「想定外は
   * fail loudly」に反する）。焼く側の正本は `gemma4.export.SYM_MAX`（配布 recipe が写しを持つ）。
   */
  readonly maxChunkLength: number;
  /**
   * この**モデルが宣言する**絶対位置の排他的上限（上流 `max_position_embeddings`）。
   *
   * NOTE: 以前は「焼き込んだ RoPE 表の行数」だった。表は配布形から外れ、cos / sin は chunk ぶんだけ
   * ホストが作る（`./rope.ts`）ので、この数はもう資産の物理的な形ではなく**モデルの宣言**である。
   * 読み手にとっての意味（`capacity` はこれを超えられない）は変わらない。
   */
  readonly maxPosition: number;
  /**
   * full スロットの容量の**既定**（会話が使える最大の論理長）。
   *
   * 実行時ノブでもある（`createGenerationSequence` / `Gemma4Pipeline.sequence` の `capacity`）—
   * 容量は state スロットの物理確保量そのものなので、長い会話と VRAM の交換を呼び手が選ぶ。
   */
  readonly capacity: number;
  /**
   * RoPE の cos / sin をホストで作るためのパラメータ（層種別 2 本 — ADR 0067 決定 4 の改訂）。
   *
   * MUST: 必須。表がグラフから外れた配布形では、この宣言が**位置エンコーディングの唯一の出どころ**
   * である（欠けたまま動けば、回転しない attention が例外なしで走る）。式と値域の正本は
   * `./rope.ts`。
   */
  readonly rope: Gemma4RopeSpec;
  /**
   * 配布形が宣言する sampler の既定（ADR 0083 決定 7）。
   *
   * リクエストが `sampler` を省略したときに使われる。宣言が無ければ低層の既定（温度 0 =
   * greedy）のままで、これは「sampler 未宣言 = 温度 0 の縮退形」という位置づけである
   * （parity 門はこの経路で生き続ける）。
   */
  readonly sampler?: Gemma4DefaultSampler;
};

const ROOT_KEYS: readonly string[] = [
  "chunkLength",
  "maxChunkLength",
  "maxPosition",
  "capacity",
  "rope",
  "sampler",
];

/** 受理する sampler の欄（型の正本は {@link Gemma4DefaultSampler} — 増やす理由も同 doc）。 */
const SAMPLER_KEYS: readonly string[] = ["temperature", "topK", "topP"];

/** 受理する rope 層種別の欄（型の正本は `./rope.ts` の {@link Gemma4RopeLayerSpec}）。 */
const ROPE_LAYER_KEYS: readonly string[] = ["theta", "headDim", "rotaryDim"];

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const assertAllowedKeys = (
  value: Record<string, unknown>,
  allowed: readonly string[],
  where: string,
): void => {
  for (const key of Object.keys(value)) {
    if (!allowed.includes(key)) {
      throw new Error(`${where}: 未知キー '${key}'（許可: ${allowed.join(" / ")}）`);
    }
  }
};

/** 欄が実在して数であることだけを見る（値域は呼び手が足す）。 */
const readRawNumber = (raw: Record<string, unknown>, key: string, where: string): number => {
  if (!Object.hasOwn(raw, key)) throw new Error(`${where}.${key}: 無い`);
  const value = raw[key];
  if (typeof value !== "number") throw new Error(`${where}.${key}: 数でない（${String(value)}）`);
  return value;
};

const readNumber = (
  raw: Record<string, unknown>,
  key: string,
  where: string,
  check: (value: number) => boolean,
  requirement: string,
): number => {
  const value = readRawNumber(raw, key, where);
  if (!check(value)) throw new Error(`${where}.${key}: ${requirement}（${String(value)}）`);
  return value;
};

/**
 * `sampler` を読む（欄ごと無ければ `undefined` = 低層の既定 = greedy）。
 *
 * 値域は `generation/sampler.ts` の `assertSpec` と同じ — 配布形を読んだ時点で落とし、
 * 「1 回生成させてから初めて分かる」形を作らない。
 */
const parseSampler = (raw: unknown, where: string): Gemma4DefaultSampler => {
  const record = isRecord(raw) ? raw : undefined;
  if (record === undefined) throw new Error(`${where}: 無い / オブジェクトでない`);
  assertAllowedKeys(record, SAMPLER_KEYS, where);
  return {
    temperature: readNumber(
      record,
      "temperature",
      where,
      (value) => Number.isFinite(value) && value >= 0,
      "0 以上の有限数でない",
    ),
    topK: readNumber(
      record,
      "topK",
      where,
      (value) => Number.isSafeInteger(value) && value >= 1,
      "1 以上の整数でない",
    ),
    topP: readNumber(
      record,
      "topP",
      where,
      (value) => Number.isFinite(value) && value > 0 && value <= 1,
      "0 < topP ≤ 1 の範囲にない",
    ),
  };
};

/**
 * 層種別 1 つぶんの rope パラメータを読む。
 *
 * MUST: 値域はここで見ない — 数の受理集合（正の theta・偶数の headDim・`2 ≤ rotaryDim ≤ headDim`）は
 * `./rope.ts` の `assertGemma4RopeLayerSpec` が正本で、同じ式を 2 実装持つと片方だけが古びる。
 * ここが持つのは**宣言の形**（未知キー・欠落・型）だけである。
 */
const parseRopeLayer = (raw: unknown, where: string): Gemma4RopeLayerSpec => {
  if (!isRecord(raw)) throw new Error(`${where}: オブジェクトでない（${String(raw)}）`);
  assertAllowedKeys(raw, ROPE_LAYER_KEYS, where);
  return {
    theta: readRawNumber(raw, "theta", where),
    headDim: readRawNumber(raw, "headDim", where),
    rotaryDim: readRawNumber(raw, "rotaryDim", where),
  };
};

/** 層種別 2 本ぶんを読み、値域は `./rope.ts` の門へ委ねる。 */
const parseRope = (raw: unknown, where: string): Gemma4RopeSpec => {
  if (!isRecord(raw)) throw new Error(`${where}: オブジェクトでない（${String(raw)}）`);
  assertAllowedKeys(raw, GEMMA4_ROPE_LAYER_TYPES, where);
  const layer = (layerType: Gemma4RopeLayerType): Gemma4RopeLayerSpec => {
    if (!Object.hasOwn(raw, layerType)) throw new Error(`${where}.${layerType}: 無い`);
    return parseRopeLayer(raw[layerType], `${where}.${layerType}`);
  };
  const spec: Gemma4RopeSpec = {
    sliding_attention: layer("sliding_attention"),
    full_attention: layer("full_attention"),
  };
  assertGemma4RopeSpec(where, spec);
  return spec;
};

/**
 * `unknown`（hub が素通しした生の宣言）を検査して {@link Gemma4PipelineConfig} にする。
 *
 * MUST: 数の関係まで見る — `chunkLength ≤ maxChunkLength` と `chunkLength ≤ capacity ≤ maxPosition`。
 * 容量いっぱいの会話は位置 `capacity - 1` まで進むので、モデルが宣言した位置上限を超える既定は
 * **長い会話でだけ**実行時に落ちる（焼く側の `gemma4_pipeline_config` が同じ関係を見るが、他人の
 * 配布形は検査を通っていない前提で読む）。実行時ノブ（sequence の `capacity`・pipeline の
 * `chunkLength`）にも同じ関係が掛かる（`generation/sequence.ts` / `./pipeline.ts`）。
 *
 * ## 公開している理由 — `fromAssets` を使う消費者のための口
 *
 * `Gemma4Pipeline.fromPretrained` は内部でこれを通すので呼ぶ必要は無い。要るのは
 * `Gemma4Pipeline.fromAssets` に自分でバイト列を渡す側で、`karume.json` を自前で読むと
 * 手元にあるのは hub の `ModelEntry.pipelineConfig`（`Record<string, unknown>`）である。
 * この口が無いと消費者は `as Gemma4PipelineConfig` で被せるか、3 つの数を配布形と自分の
 * コードに二重持ちするしかない（どちらも配布形が動いたときに黙って食い違う）。
 */
export const parseGemma4PipelineConfig = (raw: unknown): Gemma4PipelineConfig => {
  const where = "pipelineConfig";
  if (!isRecord(raw)) throw new Error(`${where}: オブジェクトでない（${String(raw)}）`);
  assertAllowedKeys(raw, ROOT_KEYS, where);
  const positiveInteger = (value: number): boolean => Number.isSafeInteger(value) && value >= 1;
  const chunkLength = readNumber(raw, "chunkLength", where, positiveInteger, "1 以上の整数でない");
  const maxChunkLength = readNumber(
    raw,
    "maxChunkLength",
    where,
    positiveInteger,
    "1 以上の整数でない",
  );
  const maxPosition = readNumber(raw, "maxPosition", where, positiveInteger, "1 以上の整数でない");
  const capacity = readNumber(raw, "capacity", where, positiveInteger, "1 以上の整数でない");
  if (chunkLength > maxChunkLength) {
    throw new Error(
      `${where}: chunkLength ${chunkLength} が maxChunkLength ${maxChunkLength} を超えた` +
        `（記号 M を焼いた trace 範囲の外）`,
    );
  }
  if (chunkLength > capacity) {
    throw new Error(
      `${where}: chunkLength ${chunkLength} が capacity ${capacity} を超えた` +
        `（1 chunk すら入らない容量）`,
    );
  }
  if (capacity > maxPosition) {
    throw new Error(
      `${where}: capacity ${capacity} が maxPosition ${maxPosition} を超えた` +
        `（容量いっぱいの会話が位置表の外を引く）`,
    );
  }
  if (!Object.hasOwn(raw, "rope")) throw new Error(`${where}.rope: 無い`);
  const rope = parseRope(raw["rope"], `${where}.rope`);
  if (!Object.hasOwn(raw, "sampler")) {
    return { chunkLength, maxChunkLength, maxPosition, capacity, rope };
  }
  return {
    chunkLength,
    maxChunkLength,
    maxPosition,
    capacity,
    rope,
    sampler: parseSampler(raw["sampler"], `${where}.sampler`),
  };
};
