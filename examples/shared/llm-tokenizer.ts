/** 実験 CLI 用の byte-level BPE。語彙は手元の公式 tokenizer.json、Unicode 表は検証済みデータ。 */
import { splitAddedTokens } from "../../packages/models/src/text/added-tokens.ts";
import { bpeEncode, type BpeMerge, type BpeModel } from "../../packages/models/src/text/bpe.ts";
import { assertCodePoint, setUnique } from "../../packages/models/src/text/asset-gates.ts";
import { toCodePoints } from "../../packages/models/src/text/code-points.ts";
import { inCodeRanges, parseCodeRanges } from "../../packages/models/src/anima/text/code-ranges.ts";
import {
  normalizeNfc,
  qwen2PreTokenize,
} from "../../packages/models/src/anima/text/qwen2-tokenizer.ts";

export type LlmFamily = "qwen3" | "minicpm5";

export const record = (raw: unknown, label: string): Record<string, unknown> => {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    throw new Error(`${label} がオブジェクトでない`);
  }
  return raw as Record<string, unknown>;
};
const array = (raw: unknown, label: string): unknown[] => {
  if (!Array.isArray(raw)) throw new Error(`${label} が配列でない`);
  return raw;
};
const string = (raw: unknown, label: string): string => {
  if (typeof raw !== "string" || raw === "") throw new Error(`${label} が空でない文字列でない`);
  return raw;
};
const id = (raw: unknown, label: string): number => {
  if (typeof raw !== "number" || !Number.isSafeInteger(raw) || raw < 0) {
    throw new Error(`${label} が非負整数でない`);
  }
  return raw;
};
const expect = (actual: unknown, expected: unknown, label: string): void => {
  // JSON のキー順には依存せず、宣言された処理を再帰的に照合する。
  if (typeof expected === "object" && expected !== null) {
    if (Array.isArray(expected)) {
      const values = array(actual, label);
      if (values.length !== expected.length) throw new Error(`${label} の長さが未対応`);
      expected.forEach((value, at) => expect(values[at], value, `${label}[${at}]`));
    } else {
      const values = record(actual, label);
      for (const [key, value] of Object.entries(expected)) {
        expect(values[key], value, `${label}.${key}`);
      }
    }
  } else if (actual !== expected) throw new Error(`${label} が未対応（${JSON.stringify(actual)}）`);
};

/** ByteLevel の全 256 バイトを可視文字へ写す（Unicode バージョンには依存しない）。 */
const byteAlphabet = (): string[] => {
  let extra = 0;
  return Array.from(
    { length: 256 },
    (_, byte) =>
      String.fromCodePoint(
        (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || byte >= 174
          ? byte
          : 256 + extra++,
      ),
  );
};

export type LlmTokenizer = {
  readonly stopTokens: readonly number[];
  encode(text: string, completion?: boolean): number[];
  chat(prompt: string, system?: string): number[];
  decoder(): { push(token: number): string; finish(): string };
};

export const createLlmTokenizer = (
  family: LlmFamily,
  raw: unknown,
  unicodeRaw: unknown,
): LlmTokenizer => {
  const data = record(raw, "tokenizer");
  expect(data.version, "1.0", "version");
  expect(data.truncation, null, "truncation");
  expect(data.padding, null, "padding");
  const mini = family === "minicpm5";
  expect(data.normalizer, mini ? null : { type: "NFC" }, "normalizer");
  const split = (pattern: string): unknown => ({
    type: "Split",
    pattern: { Regex: pattern },
    behavior: "Isolated",
    invert: false,
  });
  const pattern = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}" +
    (mini ? "+" : "") + "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
  expect(data.pre_tokenizer, {
    type: "Sequence",
    pretokenizers: [
      ...(mini ? [split("\\p{N}{1,3}")] : []),
      split(pattern),
      { type: "ByteLevel", add_prefix_space: false, trim_offsets: mini, use_regex: false },
    ],
  }, "pre_tokenizer");
  expect(data.decoder, {
    type: "ByteLevel",
    add_prefix_space: mini,
    trim_offsets: mini,
    use_regex: mini,
  }, "decoder");
  expect(
    data.post_processor,
    mini
      ? {
        type: "TemplateProcessing",
        single: [{ SpecialToken: { id: "<s>", type_id: 0 } }, {
          Sequence: { id: "A", type_id: 0 },
        }],
        special_tokens: { "<s>": { id: "<s>", ids: [0], tokens: ["<s>"] } },
      }
      : { type: "ByteLevel", add_prefix_space: false, trim_offsets: false, use_regex: false },
    "post_processor",
  );
  const model = record(data.model, "model");
  expect(model, {
    type: "BPE",
    dropout: null,
    unk_token: mini ? "<unk>" : null,
    continuing_subword_prefix: mini ? null : "",
    end_of_word_suffix: mini ? null : "",
    fuse_unk: false,
    byte_fallback: false,
    ignore_merges: false,
  }, "model");
  const tokenOf = new Map<number, string>();
  const idOf = new Map<string, number>();
  for (const [token, value] of Object.entries(record(model.vocab, "vocab"))) {
    const tokenId = id(value, `vocab.${token}`);
    if (token === "" || tokenId >= 0x7fffffff) throw new Error("語彙の綴りまたは id が不正");
    setUnique(tokenOf, tokenId, token, "vocab id");
    idOf.set(token, tokenId);
  }
  const getId = (token: string): number => {
    const found = idOf.get(token);
    if (found === undefined) throw new Error(`語彙に ${JSON.stringify(token)} が無い`);
    return found;
  };
  const added = new Map<string, number>();
  const special = new Set<number>();
  for (const rawEntry of array(data.added_tokens, "added_tokens")) {
    const entry = record(rawEntry, "added token");
    expect(entry, { single_word: false, lstrip: false, rstrip: false }, "added token");
    if (typeof entry.normalized !== "boolean" || (!mini && entry.normalized)) {
      throw new Error("追加語彙の正規化が未対応");
    }
    const text = string(entry.content, "added token content");
    const value = id(entry.id, "added token id");
    if (typeof entry.special !== "boolean") {
      throw new Error("added token special が boolean でない");
    }
    const existing = tokenOf.get(value);
    if (
      (existing !== undefined && existing !== text) || (idOf.has(text) && idOf.get(text) !== value)
    ) throw new Error("追加語彙と BPE 語彙が食い違う");
    tokenOf.set(value, text);
    idOf.set(text, value);
    setUnique(added, text, value, "added token");
    if (entry.special) special.add(value);
  }
  const alphabet = byteAlphabet();
  const byteIds = alphabet.map(getId);
  const byteOf = new Map(alphabet.map((text, byte) => [text, byte]));
  const pairStride = [...tokenOf.keys()].reduce((maximum, value) => Math.max(maximum, value), 0) +
    1;
  if (!Number.isSafeInteger(pairStride * pairStride)) throw new Error("語彙の id が大きすぎる");
  const merges = new Map<number, BpeMerge>();
  array(model.merges, "merges").forEach((rawPair, rank) => {
    const pair = array(rawPair, "merge");
    if (pair.length !== 2) throw new Error("merge が対でない");
    const left = string(pair[0], "merge left"), right = string(pair[1], "merge right");
    setUnique(
      merges,
      getId(left) * pairStride + getId(right),
      { rank, newId: getId(left + right) },
      "merge",
    );
  });
  // 入力を必ず ByteLevel の 256 文字へ写してから渡すので、byte_fallback 分岐には入らない。
  const bpe: BpeModel = { tokenOf, idOf, merges, pairStride, byteIds, byteOf: new Map() };
  const unicode = record(unicodeRaw, "unicode");
  const rawClasses = record(unicode.classes, "classes");
  const classes = {
    letter: parseCodeRanges(rawClasses.letter, "letter"),
    number: parseCodeRanges(rawClasses.number, "number"),
    space: parseCodeRanges(rawClasses.space, "space"),
  };
  const caseFold = new Map<number, number>();
  for (const rawPair of array(unicode.caseFold, "caseFold")) {
    const pair = array(rawPair, "caseFold entry");
    if (pair.length !== 2) throw new Error("caseFold が対でない");
    const source = id(pair[0], "caseFold source");
    const target = id(pair[1], "caseFold target");
    assertCodePoint(source, "caseFold source");
    assertCodePoint(target, "caseFold target");
    setUnique(caseFold, source, target, "caseFold");
  }
  const nfc = parseCodeRanges(unicode.nfcSegments, "nfcSegments");
  const encoder = new TextEncoder();
  const addedNames = [...added.keys()];
  const encode = (text: string, completion = false): number[] => {
    const ids = mini && completion ? [getId("<s>")] : [];
    for (const chunk of splitAddedTokens(text, addedNames)) {
      if (chunk.added) {
        ids.push(getId(chunk.text));
        continue;
      }
      const cps = toCodePoints(mini ? chunk.text : normalizeNfc(chunk.text, nfc));
      const pieces: string[] = [];
      if (mini) {
        // MiniCPM の第 1 Split は数字を最大 3 文字で隔離する。残りには数字が無い。
        for (let at = 0; at < cps.length;) {
          const start = at;
          if (inCodeRanges(classes.number, cps[at])) {
            while (at < cps.length && at - start < 3 && inCodeRanges(classes.number, cps[at])) at++;
            pieces.push(cps.slice(start, at).map((cp) => String.fromCodePoint(cp)).join(""));
          } else {
            while (at < cps.length && !inCodeRanges(classes.number, cps[at])) at++;
            pieces.push(...qwen2PreTokenize(cps.slice(start, at), classes, caseFold));
          }
        }
      } else pieces.push(...qwen2PreTokenize(cps, classes, caseFold));
      for (const piece of pieces) {
        let encoded = "";
        for (const byte of encoder.encode(piece)) encoded += alphabet[byte];
        for (const token of bpeEncode(bpe, encoded)) ids.push(token);
      }
    }
    return ids;
  };
  return {
    stopTokens: [getId("<|im_end|>"), getId(mini ? "</s>" : "<|endoftext|>")],
    encode,
    chat(prompt, system): number[] {
      let text = mini ? "<s>" : "";
      if (system !== undefined) text += `<|im_start|>system\n${system}<|im_end|>\n`;
      text +=
        `<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;
      return encode(text);
    },
    decoder() {
      const decoder = new TextDecoder("utf-8", { ignoreBOM: true });
      return {
        push(token): string {
          if (special.has(token)) return "";
          const text = tokenOf.get(token);
          if (text === undefined) throw new Error(`未知の出力 token ${token}`);
          const bytes: number[] = [];
          for (const char of text) {
            const byte = byteOf.get(char);
            if (byte === undefined) {
              throw new Error(`ByteLevel の外の出力文字 ${JSON.stringify(char)}`);
            }
            bytes.push(byte);
          }
          return decoder.decode(Uint8Array.from(bytes), { stream: true });
        },
        finish: () => decoder.decode(),
      };
    },
  };
};
