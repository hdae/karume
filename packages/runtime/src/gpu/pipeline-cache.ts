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
  readonly pipeline: GPUComputePipeline;
};

export class PipelineCache {
  readonly #device: GPUDevice;
  readonly #entries = new Map<string, CacheEntry>();

  constructor(device: GPUDevice) {
    this.#device = device;
  }

  get size(): number {
    return this.#entries.size;
  }

  /**
   * キーに対応する compute pipeline を返す（未生成なら生成する）。
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
      return cached.pipeline;
    }
    const pipeline = await withValidationScope(
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
    );
    this.#entries.set(key, { wgsl, pipeline });
    return pipeline;
  }
}
