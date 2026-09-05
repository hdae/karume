/**
 * 決定的な標準正規乱数（Euler の初期ノイズ `x_t` に使う）。
 *
 * 乱数はグラフに焼かない（ADR 0013 — 実行時ノブと乱数はホスト側）。seed 付きの生成器を持つのは
 * **同じ seed なら同じ latent が出る**ことを再現性の前提にするため。
 *
 * 実装は splitmix64（状態遷移）+ Box–Muller（正規化）。
 *
 * NOTE: `sbv2/host/random.ts` の**意図的な複製**。ファミリ間で import し合わない（sbv2 と
 * anima の間にも同じ複製があり、そちらが前例）— 乱数列はそのファミリの出力を決める入力なので、
 * 共有すると片方の都合（消費順・生成量・分布）がもう片方の波形／latent を静かに動かす。
 *
 * MUST: torch の `randn` とはビット一致しない。上流と同じ列が要る突合（golden 相当）は
 * **ノイズを外から注入**して行う（`IrodoriGenerateRequest.initialNoise`）。
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

/**
 * seed の受理集合（非負の安全整数 — `BigInt` へ落とすので端数も負も表せない）。
 *
 * NOTE: `export` は**生成の入口**（`pipeline.ts` の要求検査）が同じ集合で先に落とすため。
 * 生成器を作るのは DiT の段に入った後なので、ここだけに検査があると GB 級のロードと 5 グラフを
 * 回し終えた末に落ちる。受理集合は 1 本しか持たない（両側に条件を書くと必ず割れる）。
 */
export const assertAcceptableSeed = (seed: number): void => {
  if (!Number.isInteger(seed) || seed < 0 || seed > Number.MAX_SAFE_INTEGER) {
    throw new RangeError(`seed ${seed} が非負の安全整数でない`);
  }
};

/** 決定的な標準正規列の生成器（`seed` が同じなら同じ列）。 */
export class Randn {
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
   * 標準正規列を `count` 要素ぶん引き、`scale` を掛けて f32 で返す。
   *
   * MUST: 奇数長でも Box–Muller の対を丸ごと消費する（余った sin 側を次の呼び出しへ
   * 持ち越さない）。持ち越すと「同じ seed でも呼び出し順で列が変わる」形になる。
   */
  normals(count: number, scale = 1): Float32Array<ArrayBuffer> {
    if (!Number.isInteger(count) || count < 0) {
      throw new RangeError(`要素数 ${count} が 0 以上の整数でない`);
    }
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 2) {
      // u1 = 0 は log(0) = -Inf を生む。53-bit 格子の中点 0.5/2^53 へ寄せる（半径 ≈8.65σ =
      // 格子の自然な最大 ≈8.57σ と同程度。旧 Number.MIN_VALUE は最小の**非正規化数**で
      // ≈38.6σ の異常値を作る形だった）。発生は 1 ドローあたり厳密に 2^-53 なので、この置換で
      // 既存 seed の出力ビットは実質動かない（凍結 sha の生成量では踏まない）。
      const u1 = this.#uniform() || 0.5 / 2 ** 53;
      const u2 = this.#uniform();
      const radius = Math.sqrt(-2 * Math.log(u1));
      out[i] = radius * Math.cos(2 * Math.PI * u2) * scale;
      if (i + 1 < count) out[i + 1] = radius * Math.sin(2 * Math.PI * u2) * scale;
    }
    return out;
  }
}
