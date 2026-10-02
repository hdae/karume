/**
 * 決定的な標準正規乱数（Wan の初期ノイズ — ADR 0118 決定 5）。
 *
 * 乱数はグラフに焼かない（ADR 0013 — 実行時ノブと乱数はホスト側）。seed 付きの生成器を持つのは
 * **同じ seed なら同じ動画が出る**ことを再現性の前提にするため。
 *
 * 実装は splitmix64 + Box–Muller。`anima/random.ts` / `sbv2/host/random.ts` /
 * `irodori/host/random.ts` の `Randn` と**同じ核の意図的な複製**で、互いに import しない
 * （`generation/random.ts` の doc の前例 — 乱数列はその消費者の出力を決める入力なので、共有すると
 * 片方の都合〈消費順・生成量〉がもう片方の出力を静かに動かす）。
 *
 * ## MUST: torch の `randn` とは別物である
 *
 * 同じ seed を与えても torch の列にはならない。参照との照合はこの生成器で作った潜在ではなく、
 * fixture の `latents_init` を `WanGenerateRequest.latents` で注入して行う。
 */

// seed の受理集合（非負の安全整数）の所有者は家族横断の `request-gates.ts` 1 本。
import { assertAcceptableSeed } from "../request-gates.ts";

const GOLDEN_GAMMA = 0x9e3779b97f4a7c15n;
const MIX_1 = 0xbf58476d1ce4e5b9n;
const MIX_2 = 0x94d049bb133111ebn;
const MASK_64 = (1n << 64n) - 1n;
/** [0,1) の一様乱数を作るときの分母（53bit 仮数ぶん）。 */
const DENOMINATOR = 1n << 53n;

/** splitmix64 の 1 ステップ。 */
const nextUint64 = (state: bigint): { readonly state: bigint; readonly value: bigint } => {
  const next = (state + GOLDEN_GAMMA) & MASK_64;
  let z = next;
  z = ((z ^ (z >> 30n)) * MIX_1) & MASK_64;
  z = ((z ^ (z >> 27n)) * MIX_2) & MASK_64;
  return { state: next, value: (z ^ (z >> 31n)) & MASK_64 };
};

/** 決定的な標準正規列の生成器（`seed` が同じなら同じ列）。 */
export class WanRandn {
  #state: bigint;

  constructor(seed: number) {
    assertAcceptableSeed(seed);
    this.#state = BigInt(seed) & MASK_64;
  }

  /** [0, 1) の一様乱数。 */
  #uniform(): number {
    const { state, value } = nextUint64(this.#state);
    this.#state = state;
    return Number(value >> 11n) / Number(DENOMINATOR);
  }

  /**
   * 標準正規列を `count` 要素ぶん引いて f32 で返す。
   *
   * MUST: 奇数長でも Box–Muller の対を丸ごと消費する（余った sin 側を次の呼び出しへ持ち越さない）。
   */
  normals(count: number): Float32Array<ArrayBuffer> {
    if (!Number.isInteger(count) || count < 0) {
      throw new RangeError(`要素数 ${count} が 0 以上の整数でない`);
    }
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 2) {
      // u1 = 0 は log(0) = -Inf を生む。53-bit 格子の中点 0.5/2^53 へ寄せる（anima と同じ扱い）。
      const u1 = this.#uniform() || 0.5 / 2 ** 53;
      const u2 = this.#uniform();
      const radius = Math.sqrt(-2 * Math.log(u1));
      out[i] = radius * Math.cos(2 * Math.PI * u2);
      if (i + 1 < count) out[i + 1] = radius * Math.sin(2 * Math.PI * u2);
    }
    return out;
  }
}
