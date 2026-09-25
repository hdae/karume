/**
 * states 形 attention（ADR 0067 決定 4 / 6）と `state_append`（決定 5）の CPU 参照。
 *
 * GPU 側（src/kernels/state-attention.ts）と**構造を揃えない**のがオラクルの条件:
 *
 * - GPU は 3 dispatch（S 実体化 → 行統計 → 非実体化 PV）だが、参照は**素直な 3 段**
 *   （S → P を実体化 → PV）で書く。融合の段の誤りが両側で相殺しない
 *   （`referenceAttention` の MUST と同じ規律）。
 * - 縮約は **f64 で積んで格納時に 1 度だけ f32 へ丸める**（GPU の f32 逐次累積とはビット一致
 *   しないのが正しい — だから allclose で判定する）。
 * - 論理 col → 物理行の写像・両側述語・空行 → 0 は**意味論そのもの**なので同じ式になる。
 *   ここが同じでも突合は恒真化しない（GPU 側は 3 dispatch とバッファ跨ぎ、参照は 1 ループで、
 *   一致するのは値だけ）。
 *
 * MUST: 空行（述語を満たす col が 1 本も無い行）と述語外の寄与は**厳密 0**（tolerance ではなく
 * 厳密比較で見る門の相手 — ADR 0067 決定 6）。
 * MUST: pad 行（`row ≥ queryLength`）は**厳密 0 を書く**（出力は full-write されるが、
 * ADR 0066 追記 6 の「値が契約上無意味」を 0 で固定した — GPU 側は pad 行の live 走査を
 * 丸ごと省く形でこれを満たす。空行 ⊂ pad 行なので、空行 → 0 はこの規約が包含する）。逆に
 * `state_append` は先頭 `queryLength` 行しか書かない（スロットは full-write 対象外）。
 */

import { stateColumnBase, stateLiveColumns, stateSliding } from "../kernels/state-attention.ts";
import { ReferenceOpError, type RefTensor } from "./ops.ts";

/** states 形の形状と論理長（記号は ADR 0067 決定 4 — src/kernels/state-attention.ts の表）。 */
export type StateAttentionRefShape = {
  /** `B`。 */
  readonly batch: number;
  /** `H`（q の head 数）。 */
  readonly heads: number;
  /** `Hkv`（k / v の head 数 — `H % Hkv == 0`）。 */
  readonly kvHeads: number;
  /** `M`（物理 chunk 行数）。 */
  readonly chunkRows: number;
  /** `D`。 */
  readonly depth: number;
  /** `C`（スロットの行容量）。 */
  readonly capacity: number;
  /** `W`（`0` = full）。 */
  readonly window: number;
  /** `P`（pastLength）。 */
  readonly past: number;
  /** `Q`（queryLength — `1 ≤ Q ≤ M`）。 */
  readonly query: number;
};

export type StateAttentionRefInput = StateAttentionRefShape & {
  /** `[B,H,M,D]`。 */
  readonly q: Float32Array<ArrayBuffer>;
  /** 今 step の k `[B,Hkv,M,D]`（有効データは先頭 `Q` 行）。 */
  readonly insK: Float32Array<ArrayBuffer>;
  /** 今 step の v `[B,Hkv,M,D]`。 */
  readonly insV: Float32Array<ArrayBuffer>;
  /** 過去の k スロット `[B,Hkv,C,D]`。 */
  readonly slotK: Float32Array<ArrayBuffer>;
  /** 過去の v スロット `[B,Hkv,C,D]`。 */
  readonly slotV: Float32Array<ArrayBuffer>;
  /** 半スケール（q 側と k 側の**両方**へ掛ける値 — 片側だけの実装はこのオラクルが検出する）。 */
  readonly scale: number;
};

/**
 * 論理長 `past` と窓 `window` の値域（どちらも非負の整数）。
 *
 * MUST: 容量との比較より先に見る。負の `window` は `stateSliding` が false なので黙って full
 * 扱いになり、負の `past` は live 列を 0 本へ潰して厳密 0 を返す — どちらも「オラクルが黙って
 * 別の数を返す」形で、突合の赤ではなく誤った期待値になる。
 */
const assertPastAndWindow = (where: string, past: number, window: number): void => {
  if (!Number.isSafeInteger(past) || past < 0) {
    throw new ReferenceOpError(`${where}: pastLength ${past} が非負の整数でない`);
  }
  if (!Number.isSafeInteger(window) || window < 0) {
    throw new ReferenceOpError(`${where}: window ${window} が非負の整数でない（0 = full）`);
  }
};

/** 形と論理長の整合（取り違えを突合の前で落とす）。 */
const assertShape = (shape: StateAttentionRefShape): void => {
  const { batch, heads, kvHeads, chunkRows, depth, capacity, window, past, query } = shape;
  assertPastAndWindow("states 形", past, window);
  if (batch < 1 || heads < 1 || kvHeads < 1 || chunkRows < 1 || depth < 1 || capacity < 1) {
    throw new ReferenceOpError(
      `states 形の形が正でない（B=${batch} H=${heads} Hkv=${kvHeads} M=${chunkRows} D=${depth} C=${capacity}）`,
    );
  }
  if (heads % kvHeads !== 0) {
    throw new ReferenceOpError(`H=${heads} が Hkv=${kvHeads} で割り切れない（ADR 0067 決定 1）`);
  }
  if (query < 1 || query > chunkRows) {
    throw new ReferenceOpError(`queryLength ${query} が 1..${chunkRows}（= M）に入らない`);
  }
  if (stateSliding(window)) {
    if (window > capacity) {
      throw new ReferenceOpError(`window ${window} が容量 ${capacity} を超える`);
    }
  } else if (past + query > capacity) {
    throw new ReferenceOpError(
      `full スロットで pastLength ${past} + queryLength ${query} が容量 ${capacity} を超える（ADR 0067 決定 4 ④）`,
    );
  }
};

/** 与えられた配列の長さが期待どおりか（引数の取り違え = shape 不一致で落とす）。 */
const assertLength = (name: string, data: Float32Array<ArrayBuffer>, expected: number): void => {
  if (data.length !== expected) {
    throw new ReferenceOpError(`${name}: 要素数 ${data.length} が期待 ${expected} と合わない`);
  }
};

/**
 * 論理 col → スロット物理行（**読み書き同式** — GPU 側 `slot_row` の写し）。
 *
 * MUST: `state_append` の書き（{@link referenceStateAppend}）と同じ関数を使う。参照側で
 * 読みと書きの式が割れていると、GPU 側の同じ誤りをオラクルが再現して突合が恒真になる。
 * MUST: 剰余の法は**容量 `C`**（窓 `W` ではない — `src/kernels/state-attention.ts` の
 * `stateSlotRowWgsl` の doc。`C > W` の余裕が「論理長より先に書かれた行」の置き場になる）。
 */
const slotRow = (window: number, capacity: number, col: number): number =>
  stateSliding(window) ? col % capacity : col;

/**
 * states 形 attention。出力は `[B,H,M,D]`（pad 行を含む全 M 行）。
 *
 * 述語は**両側**（causal `col ≤ P + row` AND sliding `col ≥ max(0, P + row − W + 1)`）。
 * 空行は出力が厳密 0（行 max が −inf のときの構造的な 0 — ADR 0067 決定 6）。
 * pad 行（`row ≥ query`）も厳密 0（`out` の 0 初期化のまま — モジュール doc の値契約）。
 */
export const referenceStateAttention = (input: StateAttentionRefInput): RefTensor => {
  assertShape(input);
  const { batch, heads, kvHeads, chunkRows, depth, capacity, window, past, query } = input;
  const repeat = heads / kvHeads;
  assertLength("q", input.q, batch * heads * chunkRows * depth);
  assertLength("insK", input.insK, batch * kvHeads * chunkRows * depth);
  assertLength("insV", input.insV, batch * kvHeads * chunkRows * depth);
  assertLength("slotK", input.slotK, batch * kvHeads * capacity * depth);
  assertLength("slotV", input.slotV, batch * kvHeads * capacity * depth);
  const scale = Math.fround(input.scale);
  const base = stateColumnBase(window, past);
  const live = stateLiveColumns(window, past, query);
  const shape = [batch, heads, chunkRows, depth];
  const out = new Float32Array(batch * heads * chunkRows * depth);
  const scores = new Float32Array(live);
  const weights = new Float32Array(live);
  for (let plane = 0; plane < batch * heads; plane += 1) {
    // kv 平面は**整数除算**（`z / r = b·Hkv + h/r` — GPU の `wid.z / r` と同じ恒等式）
    const kvPlane = Math.floor(plane / repeat);
    // MUST: 走るのは有効行だけ（pad 行は 0 初期化のまま = GPU の ③ が書く厳密 0 と同じ値）。
    for (let row = 0; row < query; row += 1) {
      const qBase = (plane * chunkRows + row) * depth;
      const limit = past + row;
      // ① S（述語外は −inf。live の全列を埋める = GPU が S へ書く範囲と同じ）
      for (let cl = 0; cl < live; cl += 1) {
        const col = base + cl;
        // 下限は GPU 側（`in_window`）と同じ `limit − col < W` の形（u32 の巻き戻り安全形）
        const inWindow = col <= limit &&
          (!stateSliding(window) || limit - col < window);
        if (!inWindow) {
          scores[cl] = Number.NEGATIVE_INFINITY;
          continue;
        }
        const kBase = col < past
          ? (kvPlane * capacity + slotRow(window, capacity, col)) * depth
          : (kvPlane * chunkRows + (col - past)) * depth;
        const source = col < past ? input.slotK : input.insK;
        let acc = 0;
        for (let d = 0; d < depth; d += 1) {
          acc += (input.q[qBase + d] * scale) * (source[kBase + d] * scale);
        }
        scores[cl] = Math.fround(acc);
      }
      // ② safe-softmax（identity −inf・空行は全 0）
      let amax = Number.NEGATIVE_INFINITY;
      for (let cl = 0; cl < live; cl += 1) amax = Math.max(amax, scores[cl]);
      if (amax === Number.NEGATIVE_INFINITY) {
        // 空行は出力**厳密 0**（out は 0 初期化なので書かずに次の行へ）
        continue;
      }
      let total = 0;
      for (let cl = 0; cl < live; cl += 1) total += Math.exp(scores[cl] - amax);
      for (let cl = 0; cl < live; cl += 1) {
        weights[cl] = Math.fround(Math.exp(scores[cl] - amax) / total);
      }
      // ③ O[row, d] = Σ_col P[row,col] · V(col, d)
      for (let d = 0; d < depth; d += 1) {
        let acc = 0;
        for (let cl = 0; cl < live; cl += 1) {
          const col = base + cl;
          const vBase = col < past
            ? (kvPlane * capacity + slotRow(window, capacity, col)) * depth
            : (kvPlane * chunkRows + (col - past)) * depth;
          const source = col < past ? input.slotV : input.insV;
          acc += weights[cl] * source[vBase + d];
        }
        out[qBase + d] = Math.fround(acc);
      }
    }
  }
  return { dtype: "f32", shape, data: out };
};

export type StateAppendRefInput = {
  /** `B·Hkv`（スロットの平面数）。 */
  readonly kvPlanes: number;
  /** `M`（入力の行ストライド）。 */
  readonly chunkRows: number;
  /** `D`。 */
  readonly depth: number;
  /** `C`（スロットの行容量）。 */
  readonly capacity: number;
  /** `W`（`0` = full）。 */
  readonly window: number;
  /** `P`（書き込み先の論理位置の起点）。 */
  readonly past: number;
  /** `Q`（**書くのはこの行数だけ**）。 */
  readonly query: number;
  /** 今 step の k か v `[B,Hkv,M,D]`。 */
  readonly x: Float32Array<ArrayBuffer>;
  /** 書き込み前のスロット `[B,Hkv,C,D]`（**破壊しない** — 複製を返す）。 */
  readonly slot: Float32Array<ArrayBuffer>;
};

/**
 * `state_append`。返すのは**書き込み後のスロットの複製**（`[B,Hkv,C,D]`）。
 *
 * MUST: 入力の `slot` を破壊しない（同じ配列を複数のケースで使い回すテストで、前のケースの
 * 書き込みが次のケースの「書き込み前」に混ざる）。
 * MUST: 書くのは先頭 `Q` 行だけ。pad 行を書かないことが値で見える形（残骸が残る）で出る。
 */
export const referenceStateAppend = (input: StateAppendRefInput): RefTensor => {
  const { kvPlanes, chunkRows, depth, capacity, window, past, query } = input;
  assertPastAndWindow("state_append", past, window);
  if (kvPlanes < 1 || chunkRows < 1 || depth < 1 || capacity < 1) {
    throw new ReferenceOpError(
      `state_append の形が正でない（B·Hkv=${kvPlanes} M=${chunkRows} D=${depth} C=${capacity}）`,
    );
  }
  if (query < 1 || query > chunkRows) {
    throw new ReferenceOpError(`queryLength ${query} が 1..${chunkRows}（= M）に入らない`);
  }
  if (stateSliding(window)) {
    if (window > capacity) {
      throw new ReferenceOpError(`window ${window} が容量 ${capacity} を超える`);
    }
  } else if (past + query > capacity) {
    throw new ReferenceOpError(
      `full スロットで pastLength ${past} + queryLength ${query} が容量 ${capacity} を超える`,
    );
  }
  assertLength("x", input.x, kvPlanes * chunkRows * depth);
  assertLength("slot", input.slot, kvPlanes * capacity * depth);
  const out = input.slot.slice();
  for (let plane = 0; plane < kvPlanes; plane += 1) {
    for (let row = 0; row < query; row += 1) {
      const src = (plane * chunkRows + row) * depth;
      const dst = (plane * capacity + slotRow(window, capacity, past + row)) * depth;
      for (let d = 0; d < depth; d += 1) out[dst + d] = input.x[src + d];
    }
  }
  return { dtype: "f32", shape: [kvPlanes, capacity, depth], data: out };
};

/** readonly（past だけを読む — ADR 0096 決定 1）の形と論理長。`M = Q = 1` 固定なので欄は持たない。 */
export type StateAttentionReadonlyRefInput = {
  readonly batch: number;
  readonly heads: number;
  readonly kvHeads: number;
  readonly depth: number;
  /** `C`（貸し手スロットの行容量）。 */
  readonly capacity: number;
  /** `W`（`0` = full）。 */
  readonly window: number;
  /** `P`（貸し手の pastLength — query は論理位置 `P−1`）。 */
  readonly past: number;
  /** `[B,H,1,D]`。 */
  readonly q: Float32Array<ArrayBuffer>;
  /** 貸し手の k スロット `[B,Hkv,C,D]`。 */
  readonly slotK: Float32Array<ArrayBuffer>;
  /** 貸し手の v スロット `[B,Hkv,C,D]`。 */
  readonly slotV: Float32Array<ArrayBuffer>;
  readonly scale: number;
};

/**
 * readonly attention（drafter の cross-attention — ADR 0096 決定 1）。出力は `[B,H,1,D]`。
 *
 * 位置 `P−1` の query 1 行が、スロットの列 `[P − min(P, W), P)`（full は `[0, P)`）を読む。
 * 今 step の ins は**無い**。述語は `col < P`（棄却行 = 論理長より先に書かれた行を読まない）AND
 * sliding `P−1−col < W`。`P = 0` は空行 → 厳密 0。数値契約（半スケール・−inf identity・f64 で
 * 積んで格納時に 1 度だけ f32 へ）は {@link referenceStateAttention} と同じ。
 */
export const referenceStateAttentionReadonly = (
  input: StateAttentionReadonlyRefInput,
): RefTensor => {
  const { batch, heads, kvHeads, depth, capacity, window, past } = input;
  assertPastAndWindow("readonly", past, window);
  if (batch < 1 || heads < 1 || kvHeads < 1 || depth < 1 || capacity < 1) {
    throw new ReferenceOpError(
      `readonly の形が正でない（B=${batch} H=${heads} Hkv=${kvHeads} D=${depth} C=${capacity}）`,
    );
  }
  if (heads % kvHeads !== 0) {
    throw new ReferenceOpError(`H=${heads} が Hkv=${kvHeads} で割り切れない（ADR 0067 決定 1）`);
  }
  if (stateSliding(window) ? window > capacity : past > capacity) {
    throw new ReferenceOpError(
      `readonly: window ${window} / pastLength ${past} が容量 ${capacity} に収まらない`,
    );
  }
  const repeat = heads / kvHeads;
  assertLength("q", input.q, batch * heads * depth);
  assertLength("slotK", input.slotK, batch * kvHeads * capacity * depth);
  assertLength("slotV", input.slotV, batch * kvHeads * capacity * depth);
  const scale = Math.fround(input.scale);
  const base = stateSliding(window) ? past - Math.min(past, window) : 0;
  const live = stateSliding(window) ? Math.min(past, window) : past;
  const out = new Float32Array(batch * heads * depth);
  const scores = new Float32Array(live);
  const weights = new Float32Array(live);
  for (let plane = 0; plane < batch * heads; plane += 1) {
    const kvPlane = Math.floor(plane / repeat);
    const qBase = plane * depth;
    for (let cl = 0; cl < live; cl += 1) {
      const col = base + cl;
      const inWindow = col < past && (!stateSliding(window) || past - 1 - col < window);
      if (!inWindow) {
        scores[cl] = Number.NEGATIVE_INFINITY;
        continue;
      }
      const kBase = (kvPlane * capacity + slotRow(window, capacity, col)) * depth;
      let acc = 0;
      for (let d = 0; d < depth; d += 1) {
        acc += (input.q[qBase + d] * scale) * (input.slotK[kBase + d] * scale);
      }
      scores[cl] = Math.fround(acc);
    }
    let amax = Number.NEGATIVE_INFINITY;
    for (let cl = 0; cl < live; cl += 1) amax = Math.max(amax, scores[cl]);
    if (amax === Number.NEGATIVE_INFINITY) continue;
    let total = 0;
    for (let cl = 0; cl < live; cl += 1) total += Math.exp(scores[cl] - amax);
    for (let cl = 0; cl < live; cl += 1) {
      weights[cl] = Math.fround(Math.exp(scores[cl] - amax) / total);
    }
    for (let d = 0; d < depth; d += 1) {
      let acc = 0;
      for (let cl = 0; cl < live; cl += 1) {
        const col = base + cl;
        acc += weights[cl] *
          input.slotV[(kvPlane * capacity + slotRow(window, capacity, col)) * depth + d];
      }
      out[qBase + d] = Math.fround(acc);
    }
  }
  return { dtype: "f32", shape: [batch, heads, 1, depth], data: out };
};
