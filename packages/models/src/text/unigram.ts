/**
 * Unigram の格子探索（Viterbi）と id 化（fuse_unk / byte_fallback）。
 *
 * **ファミリ非依存の共通処理**（Unigram を使う tokenizer は総じてこの経路を通る）なので、
 * ファミリのディレクトリではなく `src/text/` に置く（`src/audio/wav.ts` / `src/image/png.ts`
 * と同じ位置づけ）。正規化・pre-tokenize・特殊トークン・末尾の後処理はファミリ側の責務で、
 * ここは「1 断片 → id 列」だけを見る。
 *
 * 正本は `tokenizers`（Rust）の `Lattice::viterbi` をコードポイント位置で写したもの。同点処理
 * まで正本に合わせる: 同じ位置で終わるノードは「開始位置の昇順」に積まれ、同点なら先に積まれた
 * 方（= より長い断片）が勝つ。ここがずれると、稀にだが別の分割になり id 列が静かに変わる。
 *
 * 入力を文字列でなくコードポイント列で受けるのは、対にならないサロゲートを弾く文字列 → cp 変換
 * がファミリ側（`anima/text/code-ranges.ts` の `toCodePoints`）にあり、共通層からファミリへの
 * 依存を作らないため。呼び手は必ずその変換を通してから渡す。
 */

/** 未知ノードのスコアは「語彙全体の最小スコア − 10」（正本の `K_UNK_PENALTY`）。 */
const UNK_PENALTY = 10;

export type UnigramVocabEntry = { readonly id: number; readonly score: number };

export type UnigramModel = {
  readonly vocab: ReadonlyMap<string, UnigramVocabEntry>;
  /** 語彙**全体**の最小スコア。部分集合を渡す場合も全体の値を渡す（未知ノードの重み）。 */
  readonly minScore: number;
  /** 語彙**全体**の最長トークンのコードポイント数（前方一致の探索幅）。 */
  readonly maxTokenLength: number;
  readonly unkId: number;
  /**
   * byte_fallback の `<0x00>` に当たる id。指定すると未知の断片は UTF-8 バイト列へ展開され、
   * 各バイト `b` が `byteBaseId + b` になる（未指定なら未知は {@link UnigramModel.unkId} 1 個）。
   * 語彙が `<0x00>`..`<0xFF>` を連番で持つことが前提（`tokenizers` の byte_fallback 語彙の形）。
   */
  readonly byteBaseId?: number;
};

/** Viterbi 経路のノード（開始位置とそこから始まるノードのスロット）。 */
type Slot = { readonly pos: number; readonly slot: number };

/**
 * 1 断片を id 列へ。連続する未知ノードは 1 つに融合される（`fuse_unk`）。
 *
 * NOTE: `tokenizer.json` の `fuse_unk` が `null` でも融合する（`tokenizers` の Unigram は
 * Rust 側の既定で融合する）。融合しないと日本語のように語彙に無い文字が続く入力で未知が
 * 1 文字ずつ並び、正本との突合が落ちる。
 */
export const unigramTokenize = (model: UnigramModel, cps: readonly number[]): number[] => {
  const ids: number[] = [];
  // 連続する未知ノードを溜める緩衝。空でない = 直前まで未知が続いていた。
  let pending = "";
  for (const span of viterbi(model, cps)) {
    let text = "";
    for (let k = span.start; k < span.start + span.length; k++) {
      text += String.fromCodePoint(cps[k]);
    }
    const entry = model.vocab.get(text);
    if (entry === undefined) {
      pending += text;
      continue;
    }
    if (pending !== "") {
      ids.push(...expandUnknown(model, pending));
      pending = "";
    }
    ids.push(entry.id);
  }
  if (pending !== "") ids.push(...expandUnknown(model, pending));
  return ids;
};

/**
 * 融合済みの未知断片 → id 列。
 *
 * NOTE: 「融合してから断片**全体**をバイト分解する」という境界は暫定。正本（HF）の実挙動は
 * golden で確定させる予定で、もし「ノード単位に分解してから融合しない」が正なら**ここだけ**を
 * 差し替えれば済むよう、展開点を 1 関数に閉じてある（{@link unigramTokenize} 側の融合ループは
 * byte_fallback の有無に依らず同じ形）。
 */
const expandUnknown = (model: UnigramModel, text: string): number[] => {
  const base = model.byteBaseId;
  if (base === undefined) return [model.unkId];
  const bytes = new TextEncoder().encode(text);
  return Array.from(bytes, (byte) => base + byte);
};

/** 1 断片の最適分割（正本の `Lattice::viterbi` をコードポイント位置で写す）。 */
const viterbi = (
  model: UnigramModel,
  cps: readonly number[],
): { start: number; length: number }[] => {
  const n = cps.length;
  const unkScore = model.minScore - UNK_PENALTY;
  // begin[pos] = そこから始まるノードの長さ（挿入順 = 長さ昇順）
  const begin: number[][] = [];
  const scores: number[][] = [];
  for (let pos = 0; pos < n; pos++) {
    const lengths: number[] = [];
    const nodeScores: number[] = [];
    let text = "";
    const limit = Math.min(model.maxTokenLength, n - pos);
    for (let length = 1; length <= limit; length++) {
      text += String.fromCodePoint(cps[pos + length - 1]);
      const entry = model.vocab.get(text);
      if (entry !== undefined) {
        lengths.push(length);
        nodeScores.push(entry.score);
      }
    }
    if (lengths[0] !== 1) {
      lengths.unshift(1);
      nodeScores.unshift(unkScore);
    }
    begin.push(lengths);
    scores.push(nodeScores);
  }
  // end[e] = そこで終わるノード。begin 昇順に積む（= 同じ e なら長い方が先）。
  const end: Slot[][] = Array.from({ length: n + 1 }, () => []);
  for (let pos = 0; pos < n; pos++) {
    for (const [slot, length] of begin[pos].entries()) end[pos + length].push({ pos, slot });
  }
  const best: number[][] = begin.map((lengths) => Array.from(lengths, () => 0));
  const prev: (Slot | undefined)[][] = begin.map((lengths) =>
    Array.from(lengths, (): Slot | undefined => undefined)
  );

  /** pos で終わる最良ノード。pos=0 は bos（スコア 0）。到達不能なら undefined。 */
  const bestLeft = (pos: number): { score: number; node: Slot | undefined } | undefined => {
    if (pos === 0) return { score: 0, node: undefined };
    let found: { score: number; node: Slot } | undefined;
    for (const node of end[pos]) {
      const candidate = best[node.pos][node.slot];
      // MUST: 厳密な `>` — 同点は先に積まれた方（より長い断片）が残る。
      if (found === undefined || candidate > found.score) found = { score: candidate, node };
    }
    return found;
  };

  for (let pos = 0; pos < n; pos++) {
    const left = bestLeft(pos);
    if (left === undefined) return [];
    for (let slot = 0; slot < begin[pos].length; slot++) {
      best[pos][slot] = left.score + scores[pos][slot];
      prev[pos][slot] = left.node;
    }
  }
  const tail = bestLeft(n);
  if (tail === undefined) return [];
  const path: { start: number; length: number }[] = [];
  let node = tail.node;
  while (node !== undefined) {
    path.push({ start: node.pos, length: begin[node.pos][node.slot] });
    node = prev[node.pos][node.slot];
  }
  return path.reverse();
};
