/**
 * Gemma 系トークナイザ（SPM-BPE + byte_fallback）— 文字列 ⇄ id 列。
 *
 * 正本の経路（両資産で同一構成 — ADR 0084 Context 2）:
 *   AddedVocabulary（追加語彙を leftmost-longest で切り出す・正規化の**前**）
 *   → Replace(" " → "▁") → Split(" ", MergedWithPrevious) → BPE（byte_fallback）
 *   → TemplateProcessing（gemma4 = 素通し / EmbeddingGemma = `<bos>` … `<eos>`）
 * 復号は Sequence[Replace("▁" → " "), ByteFallback, Fuse]。
 *
 * MUST: **pre_tokenizer は正規化後に 1 度も切らない**。normalizer が U+0020 を 1 つ残らず
 * U+2581 へ置換した後の文字列に対して `Split(" ")` が走るので、区切りは原理的に見つからない
 * （実測でも `pre_tokenize_str("hello▁world▁foo")` は 1 断片）。**断片は入力全長**になり、
 * 日本語だけでなく英文でも BPE の探索が長くなる — だから `bpeEncode` は merge queue で
 * 書いてある（ADR 0084 決定 3）。走査そのものを写さないのは、写しても常に恒等だから。
 *
 * MUST: {@link GemmaTokenizer.encode} は `<bos>` を付けない（gemma4 の post_processor の
 * 実測と一致）。付けるのは chat 関数だけ — 分けないと chat 導入時に double-BOS になる
 * （ADR 0084 決定 5）。EmbeddingGemma のように資産が付与を宣言している場合は
 * {@link GemmaTokenizer.encodeWithSpecialTokens} が付ける。
 *
 * MUST: **モジュールスコープで表を組み立てない**（横断不変条件「全モジュール副作用ゼロ」）。
 * 追加語彙の並びも復号の口も、インスタンス 1 個につき 1 回だけ組む。
 */

import { splitAddedTokens } from "../../text/added-tokens.ts";
import { bpeEncode } from "../../text/bpe.ts";
import {
  detokenize,
  type DetokenizerSource,
  StreamingDetokenizer,
} from "../../text/detokenizer.ts";
import { type GemmaTokenizerAssets, METASPACE } from "./asset.ts";

/** 復号の振る舞い。 */
export type GemmaDecodeOptions = {
  /**
   * 特殊トークン（`<bos>` / `<eos>` / `<|turn>` …）を出力から落とすか。既定 `true`
   * （正本 `Tokenizer.decode` の既定と同じ・生成の出力をそのまま人へ見せる形）。
   *
   * NOTE: 落としたトークンは **byte run を切らない** — 正本も skip を綴り列を作る段で
   * 行うので、挟まれた byte 列は隣接したものとして 1 文字に畳まれる。
   */
  readonly skipSpecialTokens?: boolean;
};

export class GemmaTokenizer {
  readonly #assets: GemmaTokenizerAssets;
  readonly #added: string[];
  readonly #tokenOfAdded: Map<number, string>;

  constructor(assets: GemmaTokenizerAssets) {
    this.#assets = assets;
    this.#added = [...assets.addedTokens.keys()];
    this.#tokenOfAdded = new Map(
      [...assets.addedTokens].map(([token, id]) => [id, token] as const),
    );
  }

  /** 資産が宣言している post_processor の形。 */
  get postProcessor(): GemmaTokenizerAssets["spec"]["postProcessor"] {
    return this.#assets.spec.postProcessor;
  }

  /** 資産が宣言している `<eos>` の id（停止条件の起点 — ADR 0083 決定 8）。 */
  get eosId(): number {
    return this.#assets.eosId;
  }

  /**
   * この資産が生成しうる**最大の token id**（id 空間の相互照合 — ADR 0085 決定 5）。
   *
   * 追加語彙は語彙表の**外**へ採番されうる（EmbeddingGemma の `<image_soft_token>` は
   * id 262144 / 語彙 262,144 行）ので、語彙行数だけでは上限にならない。ここが主 embedding の
   * 行数を超える資産の組み合わせは、OOB ではなく**別 token の有効な行**を引く。
   */
  get maxTokenId(): number {
    let max = this.#assets.model.pairStride - 1;
    for (const id of this.#assets.addedTokens.values()) if (id > max) max = id;
    return max;
  }

  /**
   * 追加語彙の綴り → id（無ければ `undefined`）。
   *
   * chat の綴り（`<|turn>` / `<turn|>`）も停止 token（`<|tool_response>`）も**同じ資産**から
   * 引くための口である（ADR 0084 決定 5 の「同一 digest set」— 別々の場所から拾うと片方だけ
   * 古くなる）。欠けている綴りを名指しで落とすのは呼び手の責務なので、ここは `undefined` を
   * 返すだけにする（この面自体は「知らない綴り」を異常と見なせない）。
   */
  addedTokenId(token: string): number | undefined {
    return this.#assets.addedTokens.get(token);
  }

  /**
   * `tokenizer.encode(text, add_special_tokens=False)` と同じ id 列。
   *
   * post_processor は通さない — `<bos>` の所有者は chat 関数（ADR 0084 決定 5）。
   */
  encode(text: string): number[] {
    const ids: number[] = [];
    for (const chunk of splitAddedTokens(text, this.#added)) {
      if (chunk.added) {
        ids.push(this.#assets.addedTokens.get(chunk.text) as number);
        continue;
      }
      // normalizer は `Replace(" " → "▁")` 1 本きり（Unicode 分類にも NFC にも触らない）。
      ids.push(...bpeEncode(this.#assets.model, chunk.text.replaceAll(" ", METASPACE)));
    }
    return ids;
  }

  /**
   * `tokenizer.encode(text, add_special_tokens=True)` と同じ id 列。
   *
   * 足すものは資産の宣言が決める（gemma4 は何も足さない / EmbeddingGemma は
   * `<bos>` … `<eos>`）。呼び手が家族を場合分けしないための口。
   */
  encodeWithSpecialTokens(text: string): number[] {
    const ids = this.encode(text);
    if (this.#assets.spec.postProcessor === "none") return ids;
    return [this.#assets.bosId, ...ids, this.#assets.eosId];
  }

  /** `tokenizer.decode(ids, …)` と同じ文字列（逐次復号と同じ状態機械を通る）。 */
  decode(ids: Iterable<number>, options: GemmaDecodeOptions = {}): string {
    return detokenize(this.#decoderSource(options), ids);
  }

  /**
   * `push(id) → 確定した文字列` の逐次復号器を組む（ADR 0083 決定 2 の 1 反復 = 1 push）。
   */
  createDetokenizer(options: GemmaDecodeOptions = {}): StreamingDetokenizer {
    return new StreamingDetokenizer(this.#decoderSource(options));
  }

  #decoderSource(options: GemmaDecodeOptions): DetokenizerSource {
    const skipSpecialTokens = options.skipSpecialTokens ?? true;
    const { model, specialIds } = this.#assets;
    return {
      byteOf: (id) => model.byteOf.get(id),
      textOf: (id) => {
        if (skipSpecialTokens && specialIds.has(id)) return undefined;
        // 追加語彙が先（正本の `id_to_token` と同じ順 — 語彙表の外の id はこちらにしかない）。
        const token = this.#tokenOfAdded.get(id) ?? model.tokenOf.get(id);
        if (token === undefined) {
          // 正本は未知 id を黙って読み飛ばすが、こちらは落とす（横断不変条件の fail loudly）。
          throw new Error(`復号: id ${id} が語彙にも追加語彙にも無い`);
        }
        // デコーダ鎖の 1 段目 `Replace("▁" → " ")`。byte_fallback の綴りは metaspace を
        // 含まないので、この置換と byte 判定の順序は結果を変えない。
        return token.replaceAll(METASPACE, " ");
      },
    };
  }
}
