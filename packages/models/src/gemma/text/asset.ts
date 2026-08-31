/**
 * Gemma 系トークナイザ資産（compile 済み）の受理と解釈。
 *
 * 配布形は上流の `tokenizer.json`（32.2MB）ではなく、export recipe が畳んだ小さな JSON
 * （`tools/export-recipes/_shared/gemma_tokenizer.py` — ADR 0084 決定 1）。**取得は hub の
 * 責務**なので、ここはバイト列を受けるだけで I/O を持たない。
 *
 * ここが**ファミリ側**にあるのは、受理 schema と特殊トークン集合が資産ごとに別だから
 * （ADR 0084 決定 2）。実装（`src/text/bpe.ts` / `src/text/detokenizer.ts`）は gemma4 と
 * EmbeddingGemma で共用できるが、**資産は共用できない**: merges はビット同一でも語彙は
 * 6,206 スロットで綴りが違い、追加語彙は 24 本 vs 6,415 本、post_processor も別
 * （ADR 0084 Context 1）。
 *
 * MUST: `fromPretrained(ref)` は任意の repo を指せるので、資産 JSON は**信用できない外部
 * 入力**として扱う。畳んだ資産は上流 JSON を持たないため、compile が確かめた構成を
 * {@link GemmaTokenizerSpec} として**宣言**し、ここで exact-match する（宣言が違う資産は
 * 別のトークナイザなので、読まずに落とす）。
 */

import { assertUniqueLines, setUnique } from "../../text/asset-gates.ts";
import { type BpeModel, createBpeModel } from "../../text/bpe.ts";

/** 資産 schema の版（知らない版は読まない）。 */
const ASSET_FORMAT = "karume-gemma-tokenizer/1";

/** compile が確かめた上流構成の宣言（綴りは compile 台本と 1 対 1）。 */
const SPEC_NORMALIZER = "replace-space-with-metaspace";
const SPEC_PRE_TOKENIZER = "split-space-merged-with-previous";
const SPEC_DECODER = "metaspace-byte-fallback-fuse";

/** SentencePiece の空白置換文字（U+2581）。normalizer と decoder の両方が使う。 */
export const METASPACE = "▁";

/**
 * post_processor 相当（実資産の 2 形）。
 *
 * - `none` — 何も足さない（gemma4。`<bos>` を付けるのは chat 関数だけ — ADR 0084 決定 5）
 * - `bos-eos` — `<bos>` … `<eos>` で挟む（EmbeddingGemma）
 */
export type GemmaPostProcessor = "none" | "bos-eos";

/** 資産が宣言する上流構成。 */
export type GemmaTokenizerSpec = {
  readonly postProcessor: GemmaPostProcessor;
};

/** 解釈済みの資産（{@link parseGemmaTokenizerAsset} の出力・テストは直接組む）。 */
export type GemmaTokenizerAssets = {
  readonly spec: GemmaTokenizerSpec;
  readonly model: BpeModel;
  /** 追加語彙（正規化の**前**に leftmost-longest で切り出される）。 */
  readonly addedTokens: ReadonlyMap<string, number>;
  /** `special: true` の追加語彙の id（復号時の skip 対象）。 */
  readonly specialIds: ReadonlySet<number>;
  readonly unkId: number;
  readonly bosId: number;
  readonly eosId: number;
};

const asRecord = (value: unknown, label: string): Record<string, unknown> => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label}: オブジェクトでない`);
  }
  return value as Record<string, unknown>;
};

const asString = (value: unknown, label: string): string => {
  if (typeof value !== "string") throw new Error(`${label}: 文字列でない`);
  return value;
};

/** i32 の上限（id は最終的に `Int32Array` としてグラフへ渡る — ADR 0009）。 */
const MAX_ID = 2147483647;

const asId = (value: unknown, label: string): number => {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 0 || value > MAX_ID) {
    throw new Error(`${label}: トークン id が 0..${MAX_ID} の整数でない（${String(value)}）`);
  }
  return value;
};

const asArray = (value: unknown, label: string): unknown[] => {
  if (!Array.isArray(value)) throw new Error(`${label}: 配列でない`);
  return value;
};

/** 宣言された構成を exact-match する（1 欄でも違えば別のトークナイザ）。 */
const parseSpec = (raw: unknown, label: string): GemmaTokenizerSpec => {
  const spec = asRecord(raw, label);
  for (
    const [field, want] of [
      ["normalizer", SPEC_NORMALIZER],
      ["preTokenizer", SPEC_PRE_TOKENIZER],
      ["decoder", SPEC_DECODER],
    ] as const
  ) {
    const got = asString(spec[field], `${label}.${field}`);
    if (got !== want) {
      throw new Error(
        `${label}.${field}: ${JSON.stringify(want)} でない（${JSON.stringify(got)}）`,
      );
    }
  }
  const post = asString(spec["postProcessor"], `${label}.postProcessor`);
  if (post !== "none" && post !== "bos-eos") {
    throw new Error(`${label}.postProcessor: 未知の形（${JSON.stringify(post)}）`);
  }
  return { postProcessor: post };
};

/** `左 右` の行を id 対 + rank（行番号）へ。 */
const parseMerges = (
  raw: unknown,
  count: unknown,
  label: string,
): [number, number, number][] => {
  const text = asString(raw, label);
  // MUST: 件数を突き合わせる。merges が欠けても BPE は「見つからないので結合しない」だけで
  // **落ちない** — 分割が変わって id 列が静かに別物になる。
  const want = asId(count, `${label}Count`);
  const lines = text === "" ? [] : text.split("\n");
  if (lines.length !== want) {
    throw new Error(`${label}: 行数 ${lines.length} が申告の ${want} と違う（資産が壊れている）`);
  }
  return lines.map((line, rank) => {
    const parts = line.split(" ");
    if (parts.length !== 2) throw new Error(`${label}: 行 ${rank} が「左 右」でない`);
    return [
      asId(Number(parts[0]), `${label}[${rank}].左`),
      asId(
        Number(parts[1]),
        `${label}[${rank}].右`,
      ),
      rank,
    ];
  });
};

const parseAddedTokens = (raw: unknown, label: string): Map<string, number> => {
  const out = new Map<string, number>();
  for (const [index, entry] of asArray(raw, label).entries()) {
    const pair = asArray(entry, `${label}[${index}]`);
    if (pair.length !== 2) throw new Error(`${label}[${index}]: [綴り, id] でない`);
    // NOTE: 語彙の行数では縛らない — 追加語彙は語彙表の**外**へ採番されうる
    // （EmbeddingGemma の `<image_soft_token>` は id 262144 / 語彙 262,144 行）。
    setUnique(
      out,
      asString(pair[0], `${label}[${index}]`),
      asId(pair[1], `${label}[${index}]`),
      label,
    );
  }
  return out;
};

/** 資産 JSON（解析済み）を資産表へ。 */
const interpretGemmaTokenizerAsset = (raw: unknown): GemmaTokenizerAssets => {
  const asset = asRecord(raw, "tokenizer");
  const format = asString(asset["format"], "tokenizer.format");
  if (format !== ASSET_FORMAT) {
    throw new Error(`tokenizer.format: ${JSON.stringify(ASSET_FORMAT)} でない（${format}）`);
  }
  const spec = parseSpec(asset["spec"], "tokenizer.spec");

  const vocabLines = asArray(asset["vocab"], "tokenizer.vocab").map((token, index) =>
    asString(token, `tokenizer.vocab[${index}]`)
  );
  // 行番号 = id なので、重複は「例外にならない配布破損」（先の id が引けなくなる）。
  assertUniqueLines(vocabLines, "tokenizer.vocab");

  const byteIds = asArray(asset["byteIds"], "tokenizer.byteIds").map((id, index) =>
    asId(id, `tokenizer.byteIds[${index}]`)
  );
  const model = createBpeModel({
    vocab: vocabLines.map((token, id) => [id, token] as const),
    merges: parseMerges(asset["mergesText"], asset["mergesCount"], "tokenizer.mergesText"),
    byteIds,
  });

  const addedTokens = parseAddedTokens(asset["addedTokens"], "tokenizer.addedTokens");
  const addedIds = new Set(addedTokens.values());
  const specialIds = new Set<number>();
  for (const [index, id] of asArray(asset["specialIds"], "tokenizer.specialIds").entries()) {
    const specialId = asId(id, `tokenizer.specialIds[${index}]`);
    // 特殊トークンは追加語彙の部分集合（正本の `is_special_token` も追加語彙側を見る）。
    // 外れた id を通すと「skip されるはずのトークンが本文に出る / 出ないはずが消える」。
    if (!addedIds.has(specialId)) {
      throw new Error(`tokenizer.specialIds[${index}]: id ${specialId} が追加語彙に無い`);
    }
    specialIds.add(specialId);
  }

  const vocabId = (field: string): number => {
    const id = asId(asset[field], `tokenizer.${field}`);
    if (!model.tokenOf.has(id)) throw new Error(`tokenizer.${field}: id ${id} が語彙に無い`);
    return id;
  };
  return {
    spec,
    model,
    addedTokens,
    specialIds,
    unkId: vocabId("unkId"),
    bosId: vocabId("bosId"),
    eosId: vocabId("eosId"),
  };
};

/** 取得済みバイト列（manifest の `tokenizer`）から資産表を組む。 */
export const parseGemmaTokenizerAsset = (bytes: Uint8Array): GemmaTokenizerAssets => {
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch (cause) {
    throw new Error("tokenizer: UTF-8 として読めない", { cause });
  }
  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch (cause) {
    throw new Error("tokenizer: JSON として読めない", { cause });
  }
  return interpretGemmaTokenizerAsset(json);
};
