/**
 * Wan2.1 のテキスト埋め込み資産（第 1 段 = 事前計算した埋め込み — ADR 0118 決定 4）の読み口。
 *
 * 資産は recipe `tools/export-recipes/wan/text_embeds.py` が書く safetensors 1 本:
 *
 * - テンソル: プロンプトごとに 1 本（名前 = プロンプト名・`F32`・`[L_valid, width]` — umT5 の出力を
 *   有効長で切った行。`width` は DiT の `encoder_hidden_states` の最終次元）。
 * - メタ: キー {@link WAN_TEXT_EMBEDS_METADATA_KEY} の 1 つだけに JSON（`prompts` に各プロンプトの
 *   名前・役割・原文・正規化後の文字列・トークン数。他の欄〈出所・版〉は来歴で、ここでは読まない）。
 *
 * 受理集合はメタのプロンプト文そのもの: `generate({ prompt })` は原文か正規化後の文字列のどちらかに
 * 完全一致した行だけを受け、集合の外は fail loudly（決定 4 — 第 2 段〈umT5 i8〉で集合が広がるだけで
 * API は割れない）。
 *
 * MUST: モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 */

import { parseSafetensors } from "@karume/runtime";

/** メタの唯一のキー（recipe の `METADATA_KEY`）。 */
export const WAN_TEXT_EMBEDS_METADATA_KEY = "karume.wan.text_embeds";

/** プロンプトの役割（`negative` は CFG の uncond 側の既定 — 公式の `sample_neg_prompt`）。 */
export type WanPromptRole = "positive" | "negative";

/** 受理集合の 1 行（資産のメタ — 埋め込みの値は持たない）。 */
export type WanPrompt = {
  /** 資産のテンソル名（`boxing-cats` など）。 */
  readonly name: string;
  readonly role: WanPromptRole;
  /** 出所の原文（改行や全角の約物を含めて逐語）。 */
  readonly prompt: string;
  /** 上流の `prompt_clean`（ftfy → html の unescape → 空白の畳み込み）の後の文字列。 */
  readonly normalized: string;
};

/** 1 プロンプトの埋め込み（`[tokens, width]` の行優先・f32）。 */
export type WanTextEmbedding = WanPrompt & {
  /** 有効長 `L_valid`（トークナイザのマスクが 1 の長さ）。 */
  readonly tokens: number;
  readonly data: Float32Array;
};

/** 埋め込み資産の全体（検査済み）。 */
export type WanTextEmbeds = {
  /** 1 行の幅（umT5 の出力次元）。 */
  readonly width: number;
  /** メタの並びのまま。 */
  readonly entries: readonly WanTextEmbedding[];
};

/** 埋め込み資産がこのモジュールの取り決めから外れた（資産の齟齬 — 入力起因ではない）。 */
export class WanTextEmbedsError extends Error {
  override readonly name = "WanTextEmbedsError";
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const textField = (entry: Record<string, unknown>, key: string, where: string): string => {
  const value = entry[key];
  if (typeof value !== "string") throw new WanTextEmbedsError(`${where}: '${key}' が文字列でない`);
  return value;
};

/**
 * 埋め込み資産（safetensors）を読んで検査する。
 *
 * MUST: メタとテンソルの対応を全部見る（名前の集合が一致・行数がメタのトークン数と一致・幅が全行で
 * 同じ）。1 つでも崩れていると「別のプロンプトの埋め込みで生成する」形が黙って通る。
 * MUST: 原文 / 正規化後の文字列が 2 つの行に当たる資産は拒む（どちらの埋め込みを使うかが決まらない）。
 */
export const parseWanTextEmbeds = (buffer: ArrayBuffer): WanTextEmbeds => {
  const file = parseSafetensors(buffer);
  const keys = [...file.metadata.keys()];
  if (keys.length !== 1 || keys[0] !== WAN_TEXT_EMBEDS_METADATA_KEY) {
    throw new WanTextEmbedsError(
      `メタのキー [${keys.join(", ")}] が '${WAN_TEXT_EMBEDS_METADATA_KEY}' 1 つでない`,
    );
  }
  let meta: unknown;
  try {
    meta = JSON.parse(file.metadata.get(WAN_TEXT_EMBEDS_METADATA_KEY) as string);
  } catch (cause) {
    throw new WanTextEmbedsError("メタが JSON として読めない", { cause });
  }
  if (!isRecord(meta) || !Array.isArray(meta.prompts) || meta.prompts.length === 0) {
    throw new WanTextEmbedsError("メタに 'prompts' の配列（1 行以上）が無い");
  }
  const names = new Set<string>();
  const texts = new Map<string, string>();
  let width: number | undefined;
  const entries = meta.prompts.map((raw, index): WanTextEmbedding => {
    const where = `prompts[${index}]`;
    if (!isRecord(raw)) throw new WanTextEmbedsError(`${where} がオブジェクトでない`);
    const name = textField(raw, "name", where);
    const role = textField(raw, "role", where);
    if (role !== "positive" && role !== "negative") {
      throw new WanTextEmbedsError(
        `${where}: 役割 '${role}' は positive / negative のどちらでもない`,
      );
    }
    const prompt = textField(raw, "prompt", where);
    const normalized = textField(raw, "normalized", where);
    const tokens = raw.tokens;
    if (typeof tokens !== "number" || !Number.isInteger(tokens) || tokens < 1) {
      throw new WanTextEmbedsError(`${where}: 'tokens' が正の整数でない（${String(tokens)}）`);
    }
    if (names.has(name)) throw new WanTextEmbedsError(`${where}: 名前 '${name}' が重複`);
    names.add(name);
    for (const text of new Set([prompt, normalized])) {
      const owner = texts.get(text);
      if (owner !== undefined) {
        throw new WanTextEmbedsError(
          `${where}: '${name}' の文字列が '${owner}' と同じ（行が決まらない）`,
        );
      }
      texts.set(text, name);
    }
    const view = file.tensors.get(name);
    if (view === undefined) throw new WanTextEmbedsError(`${where}: テンソル '${name}' が無い`);
    if (view.dtype !== "F32" || view.shape.length !== 2 || view.shape[0] !== tokens) {
      throw new WanTextEmbedsError(
        `テンソル '${name}' が F32 [${tokens}, width] でない（${view.dtype} [${
          view.shape.join(",")
        }]）`,
      );
    }
    width ??= view.shape[1];
    if (view.shape[1] !== width || width < 1) {
      throw new WanTextEmbedsError(
        `テンソル '${name}' の幅 ${view.shape[1]} が他の行の ${width} と違う`,
      );
    }
    return {
      name,
      role,
      prompt,
      normalized,
      tokens,
      data: new Float32Array(file.buffer, view.byteOffset, tokens * width),
    };
  });
  const extra = [...file.tensors.keys()].filter((name) => !names.has(name));
  if (extra.length > 0) {
    throw new WanTextEmbedsError(`メタに無いテンソル [${extra.join(", ")}] がある`);
  }
  return { width: width as number, entries };
};

/** 原文か正規化後の文字列に完全一致する行（無ければ undefined）。 */
export const findWanTextEmbedding = (
  embeds: WanTextEmbeds,
  text: string,
): WanTextEmbedding | undefined =>
  embeds.entries.find((entry) => entry.prompt === text || entry.normalized === text);

/**
 * DiT の `encoder_hidden_states [1, rows, width]` へ有効長の後ろをゼロで埋める（上流の
 * `_get_t5_prompt_embeds` の `new_zeros` と同じ — ホストが埋める。決定 4）。
 */
export const padWanTextEmbedding = (
  entry: WanTextEmbedding,
  rows: number,
  width: number,
): Float32Array<ArrayBuffer> => {
  if (entry.data.length !== entry.tokens * width) {
    throw new WanTextEmbedsError(`'${entry.name}' の幅が DiT の文脈の幅 ${width} と違う`);
  }
  if (entry.tokens > rows) {
    throw new WanTextEmbedsError(
      `'${entry.name}' の有効長 ${entry.tokens} が文脈の行数 ${rows} を超える`,
    );
  }
  const padded = new Float32Array(rows * width);
  padded.set(entry.data);
  return padded;
};
