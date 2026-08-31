/**
 * BPE（merge rank）の記号列生成と併合、および byte_fallback の展開。
 *
 * **ファミリ非依存の共通処理**（BPE を使う tokenizer は総じてこの経路を通る）なので、
 * ファミリのディレクトリではなく `src/text/` に置く（`unigram.ts` と同じ位置づけ理由）。
 * 正規化・pre-tokenize・特殊トークン・post_processor はファミリ側の責務で、ここは
 * 「1 断片 → id 列」だけを見る。
 *
 * 正本は `tokenizers`（Rust）の `BPE::merge_word` + `Word::merge_all` を写したもの:
 *
 *   ① 記号はコードポイント単位で始まる。語彙に無い 1 文字は byte_fallback で UTF-8 バイト
 *      1 個ずつの記号へ割れる（本移植は byteIds 256 本が揃っている資産だけを受けるので、
 *      `<unk>` へ落ちる枝は**原理的に到達しない** — {@link createBpeModel} が門になる）
 *   ② 併合は **rank の最小 → 位置の小さい方**の優先度つきキューで進める。取り出した項目は
 *      「その位置の記号が生きているか」「同じ対がまだそこに在るか」で無効化する
 *
 * MUST: **最初から merge queue で書く**（ADR 0084 決定 3）。「全隣接ペアを走査 → 最小 rank を
 * splice」の O(n²) を写してはならない — Gemma の pre_tokenizer は正規化後の文字列を 1 つも
 * 切らないので、**1 断片が入力全長になる**（`qwen2-tokenizer.ts` の `#bpe` が実害を出さないのは
 * あちらの pre_tokenizer が細かく切るからで、前提が違う）。
 */

import { setUnique } from "./asset-gates.ts";
import { toCodePoints } from "./code-points.ts";

/** 1 本の併合規則（`rank` は上流の並び順・`newId` は連結した綴りの id）。 */
export type BpeMerge = {
  readonly rank: number;
  readonly newId: number;
};

/** 引くだけの形へ畳んだ BPE モデル。 */
export type BpeModel = {
  /** id → 綴り。 */
  readonly tokenOf: ReadonlyMap<number, string>;
  /** 綴り → id。 */
  readonly idOf: ReadonlyMap<string, number>;
  /** {@link pairKey} → 併合規則。 */
  readonly merges: ReadonlyMap<number, BpeMerge>;
  /** 対の合成鍵の刻み（= 語彙の最大 id + 1）。 */
  readonly pairStride: number;
  /** byte 値 → id（256 本）。 */
  readonly byteIds: readonly number[];
  /** id → byte 値（復号側が byte run を判定する向き）。 */
  readonly byteOf: ReadonlyMap<number, number>;
};

/** {@link createBpeModel} が受ける素材（密な語彙表でも部分集合でも同じ口）。 */
export type BpeSource = {
  /** `[id, 綴り]`。 */
  readonly vocab: readonly (readonly [number, string])[];
  /** `[左 id, 右 id, rank]`。 */
  readonly merges: readonly (readonly [number, number, number])[];
  /** byte 0..255 に対応する id（**256 本ちょうど**）。 */
  readonly byteIds: readonly number[];
};

/** byte_fallback 語彙の綴り（`tokenizers` の `format!("<{:#04X}>")`）。 */
const byteTokenSpelling = (byte: number): string =>
  `<0x${byte.toString(16).toUpperCase().padStart(2, "0")}>`;

/** 隣接 2 記号の合成鍵。 */
const pairKey = (model: BpeModel, left: number, right: number): number =>
  left * model.pairStride + right;

/**
 * 資産の素材から BPE モデルを組む（外部境界なので値域と整合をここで見る）。
 *
 * MUST: **`newId` を資産に持たせず、ここで綴りの連結から引き直す**。上流も
 * `vocab[a + b]` を引いて作るので、資産へ書いて持ち回ると「語彙と併合先がずれた資産」が
 * 例外にならずに通る（分割だけが静かに変わる）。
 */
export const createBpeModel = (source: BpeSource): BpeModel => {
  const tokenOf = new Map<number, string>();
  const idOf = new Map<string, number>();
  let maxId = 0;
  for (const [id, token] of source.vocab) {
    if (!Number.isSafeInteger(id) || id < 0) {
      throw new Error(`語彙の id が非負の安全整数でない（${id}）`);
    }
    setUnique(tokenOf, id, token, "語彙の id");
    setUnique(idOf, token, id, "語彙の綴り");
    maxId = Math.max(maxId, id);
  }
  // 対の鍵は `左 * 刻み + 右` の 1 つの数。刻みを語彙から導くので、資産が大きくなれば
  // 鍵が安全整数の外へ出る — その時点で **fail loudly**（黙って別の対に当たらせない）。
  const pairStride = maxId + 1;
  if (pairStride * pairStride > Number.MAX_SAFE_INTEGER) {
    throw new Error(`語彙が大きすぎて対の合成鍵が安全整数に収まらない（最大 id ${maxId}）`);
  }

  if (source.byteIds.length !== 256) {
    throw new Error(`byteIds が 256 本でない（${source.byteIds.length} 本）`);
  }
  const byteOf = new Map<number, number>();
  for (const [byte, id] of source.byteIds.entries()) {
    const token = tokenOf.get(id);
    // MUST: 綴りまで検める。`base + byte` の連番は**実資産の実測事実**であって schema の
    // 保証ではない（ADR 0084 決定 1）— ずれた表を通すと byte 展開が別の行を指す。
    if (token !== byteTokenSpelling(byte)) {
      throw new Error(
        `byteIds[${byte}] の id ${id} の綴りが ${byteTokenSpelling(byte)} でない（${token}）`,
      );
    }
    setUnique(byteOf, id, byte, "byteIds");
  }

  const merges = new Map<number, BpeMerge>();
  const model: BpeModel = {
    tokenOf,
    idOf,
    merges,
    pairStride,
    byteIds: [...source.byteIds],
    byteOf,
  };
  for (const [left, right, rank] of source.merges) {
    const leftToken = tokenOf.get(left);
    const rightToken = tokenOf.get(right);
    if (leftToken === undefined || rightToken === undefined) {
      throw new Error(`merge の対 (${left}, ${right}) に語彙外の id がある`);
    }
    const newId = idOf.get(leftToken + rightToken);
    if (newId === undefined) {
      throw new Error(`merge (${left}, ${right}) の連結した綴りが語彙に無い`);
    }
    if (!Number.isSafeInteger(rank) || rank < 0) {
      throw new Error(`merge の rank が非負の安全整数でない（${rank}）`);
    }
    // MUST: 後勝ちを禁じる。上流も HashMap なので重複対は片方が黙って消え、分割規則だけが
    // 別物になる（`asset-gates.ts` の `setUnique` を置いた理由そのもの）。
    setUnique(merges, pairKey(model, left, right), { rank, newId }, "merge の対");
  }
  return model;
};

/** 優先度つきキューの項目（正本の `Merge` — 順序は rank 昇順・同点は位置の昇順）。 */
type QueuedMerge = {
  readonly pos: number;
  readonly rank: number;
  /** 積んだ時点の規則。取り出し時に**同一性で**照合して古い項目を捨てる。 */
  readonly merge: BpeMerge;
};

/** 二分ヒープ（最小取り出し）。単一用途なので {@link bpeEncode} の中だけで使う。 */
const isBefore = (a: QueuedMerge, b: QueuedMerge): boolean =>
  a.rank !== b.rank ? a.rank < b.rank : a.pos < b.pos;

const heapPush = (heap: QueuedMerge[], item: QueuedMerge): void => {
  heap.push(item);
  let index = heap.length - 1;
  while (index > 0) {
    const parent = (index - 1) >> 1;
    if (!isBefore(heap[index], heap[parent])) break;
    [heap[parent], heap[index]] = [heap[index], heap[parent]];
    index = parent;
  }
};

const heapPop = (heap: QueuedMerge[]): QueuedMerge | undefined => {
  const top = heap[0];
  if (top === undefined) return undefined;
  const last = heap.pop() as QueuedMerge;
  if (heap.length === 0) return top;
  heap[0] = last;
  let index = 0;
  for (;;) {
    const left = index * 2 + 1;
    const right = left + 1;
    let best = index;
    if (left < heap.length && isBefore(heap[left], heap[best])) best = left;
    if (right < heap.length && isBefore(heap[right], heap[best])) best = right;
    if (best === index) break;
    [heap[best], heap[index]] = [heap[index], heap[best]];
    index = best;
  }
  return top;
};

/**
 * 1 断片（pre-token）を id 列へ。
 *
 * 断片は正規化まで済んだ文字列で受ける（正規化・pre-tokenize はファミリ側の責務）。
 * 対にならないサロゲートは {@link toCodePoints} が落とす — 正本（Rust の `str`）に載らない
 * 値で、黙って U+FFFD へ潰すと id 列が静かに変わるため。
 */
export const bpeEncode = (model: BpeModel, piece: string): number[] => {
  const symbols = initialSymbols(model, piece);
  const count = symbols.length;
  if (count < 2) return symbols;

  const prev = new Int32Array(count);
  const next = new Int32Array(count);
  const alive = new Uint8Array(count).fill(1);
  for (let index = 0; index < count; index++) {
    prev[index] = index - 1;
    next[index] = index + 1 < count ? index + 1 : -1;
  }

  const heap: QueuedMerge[] = [];
  const enqueue = (pos: number, right: number): void => {
    const merge = model.merges.get(pairKey(model, symbols[pos], symbols[right]));
    if (merge !== undefined) heapPush(heap, { pos, rank: merge.rank, merge });
  };
  for (let index = 0; index + 1 < count; index++) enqueue(index, index + 1);

  for (;;) {
    const top = heapPop(heap);
    if (top === undefined) break;
    if (alive[top.pos] === 0) continue;
    const right = next[top.pos];
    if (right === -1) continue;
    // MUST: **同一性で**照合する。位置の記号は他の併合で入れ替わりうるので、rank だけの
    // 比較では「別の対がたまたま同じ rank」を通してしまう（正本は `new_id` で見ている）。
    if (model.merges.get(pairKey(model, symbols[top.pos], symbols[right])) !== top.merge) continue;

    symbols[top.pos] = top.merge.newId;
    alive[right] = 0;
    next[top.pos] = next[right];
    if (next[right] !== -1) prev[next[right]] = top.pos;
    if (prev[top.pos] !== -1) enqueue(prev[top.pos], top.pos);
    if (next[top.pos] !== -1) enqueue(top.pos, next[top.pos]);
  }
  return symbols.filter((_, index) => alive[index] === 1);
};

/**
 * 記号列の初期状態（コードポイント 1 個 = 記号 1 個・語彙に無ければ byte_fallback）。
 *
 * NOTE: `<unk>` へ落ちる枝は写していない。{@link createBpeModel} が byteIds 256 本の存在を
 * 門にしているので、正本の「バイトが 1 つでも語彙に無ければ unk」は到達しない。
 */
const initialSymbols = (model: BpeModel, piece: string): number[] => {
  const symbols: number[] = [];
  const encoder = new TextEncoder();
  for (const cp of toCodePoints(piece)) {
    const char = String.fromCodePoint(cp);
    const id = model.idOf.get(char);
    if (id !== undefined) {
      symbols.push(id);
      continue;
    }
    for (const byte of encoder.encode(char)) symbols.push(model.byteIds[byte]);
  }
  return symbols;
};
