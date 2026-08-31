/**
 * 決定的な [0,1) 一様乱数（sampling の抽選に使う）。
 *
 * 乱数はグラフに焼かない（ADR 0013 — 実行時ノブと乱数はホスト側）。seed 付きの生成器を持つのは
 * **同じ seed なら同じ token 列が出る**ことを再現性の前提にするため。
 *
 * 実装は splitmix64（状態遷移）。`anima/random.ts` / `sbv2/host/random.ts` /
 * `irodori/host/random.ts` の `Randn` と**同じ核の意図的な複製**で、互いに import しない
 * （3 者の間に既に同じ複製があり、そちらが前例）— 乱数列はその消費者の出力を決める入力なので、
 * 共有すると片方の都合（消費順・生成量）がもう片方の画像／波形／token 列を静かに動かす。
 * 違いは正規化の段だけで、こちらは Box–Muller を通さず一様のまま出す（多項分布の抽選に
 * 要るのは [0,1) の 1 本だけ）。
 *
 * MUST: torch の sampling とはビット一致しない（ADR 0083 決定 7）。HF の `generate` と同じ
 * token 列が要る突合は**温度 0（= greedy 縮退）**で採る — 抽選が走る経路の parity は取れない。
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

/** 決定的な [0,1) 一様列の生成器（`seed` が同じなら同じ列）。 */
export class Randu {
  #state: bigint;

  constructor(seed: number) {
    if (!Number.isInteger(seed) || seed < 0 || seed > Number.MAX_SAFE_INTEGER) {
      throw new RangeError(`seed ${seed} が非負の安全整数でない`);
    }
    this.#state = BigInt(seed) & MASK_64;
  }

  /** 次の [0, 1) 一様乱数。 */
  uniform(): number {
    const { state, value } = nextUint64(this.#state);
    this.#state = state;
    return Number(value >> 11n) / Number(DENOMINATOR);
  }
}
