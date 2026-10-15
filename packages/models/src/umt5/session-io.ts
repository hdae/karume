/**
 * umT5-XXL のグラフ（ADR 0119 決定 3・4）の入出力をホストで組む純関数（単体の `Umt5Encoder`〈`./encoder.ts`〉と
 * Wan の text 段〈`src/wan/text-stage.ts` の `encodeWanPrompts`〉が使う）。
 *
 * - 入力: トークナイザ（`./tokenizer.ts` の `Umt5Tokenizer.tokenize`）の id 列 `[L]` と
 *   バケット表 `[L, L]`（`./relative-position.ts`）を、グラフ入力 token id `[1, L]` と
 *   バケット表に束ねる。L は 1 シンボルの動的次元で、2 本の入力が同じ L であることをここで見る。
 * - 出力: グラフの出力 `[1, L, width]` を検査して `[L, width]`（行優先・f32）として読む
 *   （{@link umt5HiddenStates}）。Wan の DiT へ渡すときは、文脈入力 `encoder_hidden_states [1, rows, width]` の
 *   中身へ写して有効長の後ろをゼロで詰める（{@link padUmt5Context} — 上流 `_get_t5_prompt_embeds` の
 *   `new_zeros` と同じ・事前計算の資産の経路の `padWanTextEmbedding` と同じ規則）。マスクは持たない（決定 4）。
 *
 * どちらの食い違いも入力起因ではなく組み立ての誤り（素の `Error` — ADR 0107 決定 2 の区分）。
 *
 * MUST: モジュール副作用ゼロ（CLAUDE.md）。
 */

import type { Tensor } from "@karume/runtime";
import { type I32Tensor, UMT5_MAX_LENGTH, UMT5_MIN_TOKENS } from "./relative-position.ts";

/** グラフ入力の名前（recipe `wan/umt5_patch.py` の `INPUT_NAMES` — fixture のテストが一致を見る）。 */
export const UMT5_INPUT_IDS = "input_ids";
export const UMT5_RELATIVE_POSITION_BUCKETS = "relative_position_buckets";

/** Session に渡す入力の束（2 本とも i32）。 */
export type Umt5SessionInputs = {
  readonly [UMT5_INPUT_IDS]: I32Tensor;
  readonly [UMT5_RELATIVE_POSITION_BUCKETS]: I32Tensor;
};

/**
 * id 列とバケット表をグラフ入力に束ねる。
 *
 * `ids` と `buckets.data` は写さずにそのまま渡す（Session の入力は borrowed — 実行が settle するまで
 * 書き換えない。契約は `@karume/runtime` の `Tensor` の doc）。
 */
export const umt5SessionInputs = (
  ids: Int32Array<ArrayBuffer>,
  buckets: I32Tensor,
): Umt5SessionInputs => {
  const length = ids.length;
  if (length < UMT5_MIN_TOKENS || length > UMT5_MAX_LENGTH) {
    throw new Error(
      `umT5 の id 列の長さ ${length} が ${UMT5_MIN_TOKENS}〜${UMT5_MAX_LENGTH} の外`,
    );
  }
  const [rows, columns] = buckets.shape;
  if (buckets.shape.length !== 2 || rows !== length || columns !== length) {
    const shape = buckets.shape.join(",");
    throw new Error(`umT5 のバケット表 [${shape}] が id 列の長さ ${length} の正方形でない`);
  }
  if (buckets.data.length !== length * length) {
    throw new Error(
      `umT5 のバケット表の要素数 ${buckets.data.length} が [${length},${length}] と合わない`,
    );
  }
  return {
    [UMT5_INPUT_IDS]: { dtype: "i32", shape: [1, length], data: ids },
    [UMT5_RELATIVE_POSITION_BUCKETS]: buckets,
  };
};

/**
 * グラフの出力 `[1, tokens, width]` を検査し、中身を `[tokens, width]`（行優先・f32）として返す（写さない —
 * Session の出力は呼び手の所有）。
 *
 * `tokens` は入力に渡した id 列の長さ（出力の行数がそれと一致することを見る — 別の run の出力の取り違えを
 * 形で拾う）。`width` はグラフ宣言の出力の幅（`./graph-contract.ts` の `hiddenSize`）。
 */
export const umt5HiddenStates = (
  output: Tensor,
  tokens: number,
  width: number,
): Float32Array<ArrayBuffer> => {
  if (output.dtype !== "f32") {
    throw new Error(`umT5 の出力の dtype ${output.dtype} が f32 でない`);
  }
  const [batch, length, channels] = output.shape;
  if (output.shape.length !== 3 || batch !== 1 || length !== tokens || channels !== width) {
    throw new Error(
      `umT5 の出力 [${output.shape.join(",")}] が [1,${tokens},${width}] でない`,
    );
  }
  if (output.data.length !== tokens * width) {
    throw new Error(
      `umT5 の出力の要素数 ${output.data.length} が [1,${tokens},${width}] と合わない`,
    );
  }
  return output.data;
};

/**
 * グラフの出力 `[1, tokens, width]` を DiT の文脈 `[rows, width]`（行優先・f32）へゼロで詰める（検査は
 * {@link umt5HiddenStates}）。
 *
 * `rows` / `width` は DiT のグラフの `encoder_hidden_states` の宣言から取る（`src/wan/text-stage.ts` と
 * 同じ — 512 / 4,096 を写経しない）。
 */
export const padUmt5Context = (
  output: Tensor,
  tokens: number,
  rows: number,
  width: number,
): Float32Array<ArrayBuffer> => {
  const states = umt5HiddenStates(output, tokens, width);
  if (tokens > rows) {
    throw new Error(`umT5 の出力の行数 ${tokens} が DiT の文脈の行数 ${rows} を超える`);
  }
  const padded = new Float32Array(rows * width);
  padded.set(states);
  return padded;
};
