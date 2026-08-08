/**
 * 決定的な標準正規乱数（sdp の `z_noise` と z_p のノイズに使う）。
 *
 * 乱数はグラフに焼かない（ADR 0013 — 実行時ノブと乱数はホスト側）。デモが seed 付きの
 * 生成器を持つのは、**同じ seed なら同じ WAV が出る**ことを再現性の前提にするため
 * （`Math.random` だと torch 参照との突合が「毎回別の入力」になり、dump した乱数列を
 * 読ませる意味も薄れる）。
 *
 * 実装は splitmix64（状態遷移）+ Box–Muller（正規化）。splitmix64 は 64bit 定数を使うので
 * `BigInt` で回す — 生成量は 2·P + C·T 程度（数万〜数十万要素）で、ホスト側の他の処理に
 * 比べて無視できる。
 */

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
export class Randn {
  #state: bigint;

  constructor(seed: number) {
    if (!Number.isInteger(seed) || seed < 0 || seed > Number.MAX_SAFE_INTEGER) {
      throw new RangeError(`seed ${seed} が非負の安全整数でない`);
    }
    this.#state = BigInt(seed) & MASK_64;
  }

  /** [0, 1) の一様乱数。 */
  #uniform(): number {
    const { state, value } = nextUint64(this.#state);
    this.#state = state;
    return Number(value >> 11n) / Number(DENOMINATOR);
  }

  /**
   * 標準正規列を `count` 要素ぶん引き、`scale` を掛けて f32 で返す。
   *
   * MUST: 奇数長でも Box–Muller の対を丸ごと消費する（余った sin 側を次の呼び出しへ
   * 持ち越さない）。持ち越すと「同じ seed でも呼び出し順で列が変わる」形になり、
   * dump した乱数列との対応が読めなくなる。
   */
  normals(count: number, scale = 1): Float32Array<ArrayBuffer> {
    if (!Number.isInteger(count) || count < 0) {
      throw new RangeError(`要素数 ${count} が 0 以上の整数でない`);
    }
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 2) {
      // u1 = 0 は log(0) = -Inf を生む。最小の正規化数へ寄せる（分布への影響は 2^-53 未満）。
      const u1 = this.#uniform() || Number.MIN_VALUE;
      const u2 = this.#uniform();
      const radius = Math.sqrt(-2 * Math.log(u1));
      out[i] = radius * Math.cos(2 * Math.PI * u2) * scale;
      if (i + 1 < count) out[i + 1] = radius * Math.sin(2 * Math.PI * u2) * scale;
    }
    return out;
  }
}
