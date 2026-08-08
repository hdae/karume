/**
 * （キー → compute pipeline）のキャッシュ。
 *
 * MUST: 同一キーには常にバイト単位で同一の WGSL が対応すること（codegen 決定性）。
 * ブラウザ側の暗黙パイプラインキャッシュは WGSL 文字列そのものをキーにしており、明示的な
 * 制御 API は存在しない。決定性が崩れると初回コンパイル代を毎回払うだけでなく、同じキーで
 * 別のカーネルが走る（キャッシュヒット時に古い WGSL が使われる）ため、下の実行時ガードで
 * 不一致を即座に落とす。
 *
 * device と同じ寿命で作り直す。device 消失後に持ち越したパイプラインは使えない。
 */

import { withValidationScope } from "./device.ts";

/** 同一キーに異なる WGSL が渡された（決定性の破れ）。 */
export class PipelineKeyConflictError extends Error {
  override readonly name = "PipelineKeyConflictError";
}

type CacheEntry = {
  readonly wgsl: string;
  /**
   * **未決着の生成も込みで**共有する（解決済みの値ではなく Promise を持つ）。同じキーの
   * 並行 get() がそれぞれシェーダモジュールとパイプラインを作ると、コンパイル代が実測で
   * 二重に乗るうえ validation errorScope も入れ子で張られる。
   */
  readonly pipeline: Promise<GPUComputePipeline>;
};

export class PipelineCache {
  readonly #device: GPUDevice;
  readonly #entries = new Map<string, CacheEntry>();

  constructor(device: GPUDevice) {
    this.#device = device;
  }

  /** 登録済みエントリ数（生成中のものを含み、失敗したものは含まない）。 */
  get size(): number {
    return this.#entries.size;
  }

  /**
   * キーに対応する compute pipeline を返す（未生成なら生成する）。**生成中のものも同じ
   * エントリで待ち合わせる** — 登録は await の前に行う（await 後に登録すると、その間に来た
   * 同一キーの get() が全て取りこぼして重複生成になる）。
   *
   * WGSL 不一致の判定は生成の完了を待たない（キャッシュに載った瞬間から効く）。
   *
   * 生成は必ず validation errorScope の中で行う。無効なシェーダ / パイプラインは同期例外に
   * ならず、生成成功に見えたまま dispatch が no-op 化して出力が全て 0 になる。
   */
  async get(key: string, wgsl: string): Promise<GPUComputePipeline> {
    const cached = this.#entries.get(key);
    if (cached !== undefined) {
      if (cached.wgsl !== wgsl) {
        throw new PipelineKeyConflictError(
          `パイプラインキー "${key}" に異なる WGSL が渡された（codegen 決定性の破れ）`,
        );
      }
      return await cached.pipeline;
    }
    // MUST: 失敗したエントリは残さない（残すと同じ拒否 Promise を全員が受け取り続け、
    // 再試行の余地が消える）。この handler は生成時に登録済みなので、待ち手が拒否を
    // 観測するより先に走る = 削除の後に来た get() だけが作り直す。
    const pipeline = withValidationScope(
      this.#device,
      `createComputePipeline(${key})`,
      () => {
        const module = this.#device.createShaderModule({ label: key, code: wgsl });
        return this.#device.createComputePipeline({
          label: key,
          layout: "auto",
          compute: { module },
        });
      },
    ).catch((cause: unknown) => {
      this.#entries.delete(key);
      throw cause;
    });
    this.#entries.set(key, { wgsl, pipeline });
    return await pipeline;
  }
}
