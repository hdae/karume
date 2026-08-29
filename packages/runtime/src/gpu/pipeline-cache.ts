/**
 * （キー → compute pipeline）のキャッシュ。
 *
 * MUST: 同一キーには常にバイト単位で同一の WGSL が対応すること（codegen 決定性）。
 * ブラウザ側の暗黙パイプラインキャッシュは WGSL 文字列そのものをキーにしており、明示的な
 * 制御 API は存在しない。決定性が崩れると初回コンパイル代を毎回払うだけでなく、同じキーで
 * 別のカーネルが走る（キャッシュヒット時に古い WGSL が使われる）ため、下の実行時ガードで
 * 不一致を即座に落とす。
 *
 * **device 1 個につき 1 個**（所有者は GpuContext — `GpuContextInternals.pipelines`）。同一
 * device 上の Session はこの 1 本を共有し、Session ごとに割り直さない — パイプラインの再利用
 * 可能性は device 単位で決まるので、Session ごとに持つと同じ WGSL のコンパイルと
 * `getBindGroupLayout` の解決を Session の本数だけ払い直すことになる。device 消失後に持ち越した
 * パイプラインは使えないので、作り直しの入口は `acquireGpu()` からの GpuContext 再構築だけ。
 */

import { withPipelineScope } from "./error-scope.ts";

/** 同一キーに異なる WGSL が渡された（決定性の破れ）。 */
export class PipelineKeyConflictError extends Error {
  override readonly name = "PipelineKeyConflictError";
}

/**
 * {@link PipelineCache.get} が返す解決済みの組。layout はパイプラインごとに不変なので、
 * パイプライン生成解決時に `getBindGroupLayout(0)` を 1 回だけ呼んで保持する（毎 dispatch の
 * bind group 生成で呼び直さない）。
 */
type CachedPipeline = {
  readonly pipeline: GPUComputePipeline;
  readonly layout: GPUBindGroupLayout;
};

type CacheEntry = {
  readonly wgsl: string;
  /**
   * **未決着の生成も込みで**共有する（解決済みの値ではなく Promise を持つ）。同じキーの
   * 並行 get() がそれぞれシェーダモジュールとパイプラインを作ると、コンパイル代が実測で
   * 二重に乗るうえ validation errorScope も入れ子で張られる。
   */
  readonly resolved: Promise<CachedPipeline>;
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
   * キーに対応する compute pipeline + bind group layout を返す（未生成なら生成する）。
   * **生成中のものも同じエントリで待ち合わせる** — 登録は await の前に行う（await 後に登録
   * すると、その間に来た同一キーの get() が全て取りこぼして重複生成になる）。
   *
   * WGSL 不一致の判定は生成の完了を待たない（キャッシュに載った瞬間から効く）。
   *
   * 生成は必ず errorScope（internal + validation の 2 本 = {@link withPipelineScope}）の中で
   * 行う。無効なシェーダ / パイプラインは同期例外にならず、生成成功に見えたまま dispatch が
   * no-op 化して出力が全て 0 になる。layout はパイプライン生成解決時に `getBindGroupLayout(0)`
   * を 1 回だけ呼んで一緒に保持する（呼び出し側が毎 dispatch で呼び直さずに済む）。
   */
  async get(key: string, wgsl: string): Promise<CachedPipeline> {
    const cached = this.#entries.get(key);
    if (cached !== undefined) {
      if (cached.wgsl !== wgsl) {
        throw new PipelineKeyConflictError(
          `パイプラインキー "${key}" に異なる WGSL が渡された（codegen 決定性の破れ）`,
        );
      }
      return await cached.resolved;
    }
    // MUST: 失敗したエントリは残さない（残すと同じ拒否 Promise を全員が受け取り続け、
    // 再試行の余地が消える）。この handler は生成時に登録済みなので、待ち手が拒否を
    // 観測するより先に走る = 削除の後に来た get() だけが作り直す。
    const resolved = withPipelineScope(
      this.#device,
      `createComputePipeline(${key})`,
      () => {
        const module = this.#device.createShaderModule({ label: key, code: wgsl });
        const pipeline = this.#device.createComputePipeline({
          label: key,
          layout: "auto",
          compute: { module },
        });
        // MUST: `getBindGroupLayout(0)` はスコープの**内側**で呼ぶ。層の取得は本来ただの
        // 付随処理だが、無効なパイプラインに対して呼ぶと派生の validation エラーが立つため、
        // 生成の検出網を二重化する役目を兼ねている（例えば internal エラーで無効化された
        // パイプラインは、ここで立つ派生エラーによっても捕捉され、下の eviction 経路へ落ちる）。
        // スコープの外へ出すとこの偶然の防御が黙って消える。
        return { pipeline, layout: pipeline.getBindGroupLayout(0) };
      },
    ).catch((cause: unknown) => {
      this.#entries.delete(key);
      throw cause;
    });
    this.#entries.set(key, { wgsl, resolved });
    return await resolved;
  }
}

/**
 * 1 Session ぶんの使用記録を付けた {@link PipelineCache} への薄い委譲。
 *
 * キャッシュが device 寿命になったことで「この device に載っているパイプラインの本数」と
 * 「この Session が使ったパイプラインの本数」が別の量になった。後者は診断
 * （`SessionDiagnostics.pipelineCount`）が答える問い — グラフと opt-in の組み合わせに対して
 * 何本のカーネルが立ったかは Session ごとの事実で、同一 device に別の Session が居るかどうかで
 * 変わってはいけない。前者は `SessionDiagnostics.devicePipelineCount` が答える。
 *
 * MUST: 本数は使用キー集合から導出する（独立に更新するカウンタを持たない）。同じキーを
 * 複数のステップが引くのが通常形なので、加算カウンタでは「使ったキーの本数」ではなく
 * 「引いた回数」になり、しかもズレても例外も警告も出ない。
 */
export class SessionPipelines {
  readonly #cache: PipelineCache;
  readonly #usedKeys = new Set<string>();

  constructor(cache: PipelineCache) {
    this.#cache = cache;
  }

  /** この Session が使ったパイプラインキーの本数（集合から導出）。 */
  get usedCount(): number {
    return this.#usedKeys.size;
  }

  /** 委譲先が抱える本数 = この device 上の全 Session の合計。 */
  get deviceCount(): number {
    return this.#cache.size;
  }

  /**
   * {@link PipelineCache.get} へ委譲し、**解決したキーだけ**を使用集合へ記録する。
   *
   * MUST: 記録は解決後（失敗したキーは数えない）。委譲先は失敗したエントリを残さないので、
   * 発行時点で記録すると「device には無いのに Session は使ったことになっている」形が残る。
   */
  async get(key: string, wgsl: string): Promise<CachedPipeline> {
    const resolved = await this.#cache.get(key, wgsl);
    this.#usedKeys.add(key);
    return resolved;
  }
}
