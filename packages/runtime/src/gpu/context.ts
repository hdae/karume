/**
 * 取得済み device の器（{@link GpuContext}）と、それが所有する常駐テンソル / バッチ区間。
 *
 * MUST: このファイルから `acquire.ts` を**実体として** import しない。層の依存は
 * acquire → context の一方向で、{@link GpuContext} を構築するのは `acquireGpu` 側だけ
 * （逆向きの参照は `import type` のみ — 消去されるので実行時の import グラフは一方向のまま）。
 * 層の入口はこの 2 ファイルの公開名を再 export する `device.ts`。
 *
 * この層はさらに 3 つ、**device と同じ寿命で GpuContext が所有する**器を持つ:
 * {@link ResidentTensor}（Session を跨いで共有できる第 4 の寿命クラス）と
 * {@link BatchScope}（フェンス 1 本で閉じる enqueue 区間）、そして
 * {@link GpuContextInternals.pipelines}（コンパイル済み compute pipeline のキャッシュ）。
 * 前 2 者は errorScope 区間と消失レースの規律がそのまま効くため、GPU バッファの器でありながら
 * ここに置いてある。
 */

// MUST: 型だけを取る（実体を import するとこの層（context.ts）→ カナリア → kernels / reference の
// 依存が生まれ、「この層から kernels / reference への import を作らない」規律が崩れる —
// {@link GpuContextInternals.attentionI8a8Dot}）。`import type` は消去されるので、
// 実行時の import グラフは今までどおり一方向のまま。
import type { AttentionI8a8Decision } from "./attention-dp4a-canary.ts";
// MUST: acquire.ts からも型だけを取る（値を取ると冒頭の「acquire → context の一方向」が
// 壊れる）。消去される参照なので、実行時の import グラフは acquire → context のまま。
import type { DeviceLostHandler, RequiredLimits } from "./acquire.ts";
import { STORAGE_USAGE } from "./arena.ts";
import { discardFailureScopes, popFailureScopes, pushFailureScopes } from "./error-scope.ts";
import { PipelineCache } from "./pipeline-cache.ts";
import { BUFFER_USAGE, MAP_MODE } from "./webgpu-constants.ts";

/** device が失われた状態での操作。待ち続ける代わりに必ずこれを投げる。 */
export class GpuDeviceLostError extends Error {
  override readonly name = "GpuDeviceLostError";
}

/**
 * 消失理由を {@link GpuDeviceLostError} の文言へ足す接尾（未消失・情報無しなら空文字）。
 *
 * MUST: 消失を例外へ変える経路（`#raceDeviceLost` の待ち・{@link assertDeviceUsable} の同期判定・
 * shader-f16 カナリアの `raceCanaryDeviceLost`・`runtime/generation-context.ts` の同名判定）は
 * 全てこれを通す。`GPUDeviceLostInfo.message` は
 * **バックエンドが入れた真因の唯一の生き残り**で（Metal の
 * `Device::create_query_set: ...` 相当）、errorScope には入らない — 消失済み device の
 * エラーは Deno の error handler の入口で捨てられるため、ここで落とすと真因は karume の
 * どこにも残らない（`docs/limitations.md` の Metal timestamp 節）。
 * 接**頭**は変えない: 「どこで失われたか」を先に読ませる形は既存の呼び手と doc が前提に
 * している（理由は後置の追加情報）。
 */
export const describeDeviceLoss = (info: GPUDeviceLostInfo | undefined): string => {
  if (info === undefined) return "";
  const message = info.message.trim();
  return ` — reason: ${info.reason}${message === "" ? "" : ` / ${message}`}`;
};

/** 常駐テンソル（{@link ResidentTensor}）の寿命規律の破れ（破棄後利用・参照中の破棄）。 */
export class ResidentTensorError extends Error {
  override readonly name = "ResidentTensorError";
}

/** バッチ区間（{@link BatchScope}）の使い方の破れ（決着後の enqueue・計測との併用など）。 */
export class BatchScopeError extends Error {
  override readonly name = "BatchScopeError";
}

/** GPU 側時間計測（pass 境界の timestamp）に要る feature — ADR 0021。 */
export const TIMESTAMP_QUERY_FEATURE: GPUFeatureName = "timestamp-query";

/**
 * f16 **計算**変種（共有タイルを f16 にする GEMM — `enable f16`）に要る feature。
 *
 * MUST: 重み**格納** f16（ADR 0018）と混同しない。格納側は core WGSL の `unpack2x16float`
 * だけで動き、この feature を一切要求しない。こちらは WGSL の `f16` 型そのものを使うので
 * feature が無ければシェーダのコンパイルが通らない。
 */
export const SHADER_F16_FEATURE: GPUFeatureName = "shader-f16";

/**
 * **ランタイム内部だけが触る面**の鍵（mod.ts からは輸出しない — ADR 0008 の「薄い面」を
 * 汚さない）。{@link GpuContext} / {@link ResidentTensor} / {@link BatchScope} は利用者向けの
 * 数メソッドだけを素の名前で持ち、executor が要る実体（直列化プリミティブ・GPUBuffer・
 * 焼き込み参照計数・メンバ登録）はこの鍵の下に畳む。テスト専用ノブ（`I8A8_DOT`）と同じ流儀。
 *
 * MUST: 宣言は {@link GpuContext} より**前**に置く。クラス本体の計算プロパティ名は
 * クラス定義時に評価されるので、後ろに置くと TDZ で落ちる。
 */
export const RUNTIME_INTERNAL: unique symbol = Symbol("karume.runtimeInternal");

/** {@link GpuContext} のランタイム内部面（利用者ストーリーに対応しない直列化プリミティブ）。 */
type GpuContextInternals = {
  /**
   * `work` の決着と device 消失を競わせ、消失が先なら {@link GpuDeviceLostError} にする。
   *
   * MUST: 待ちは失敗ではなくハングになりうる（`onSubmittedWorkDone` / `mapAsync` が消失後に
   * 解決するかは実装差のある面 — {@link "./acquire.ts"} の `raceCanaryDeviceLost` の doc に実測を記録）。
   * 決着時は購読を必ず解除する（reaction を積み残さない）。
   */
  raceDeviceLost<T>(work: Promise<T>, where: string): Promise<T>;
  /**
   * `await` を跨いで errorScope を張る区間を、device 単位で直列化して実行する。
   * 規則の全体は {@link GpuContext} 冒頭「errorScope 区間の不変条件」を参照。
   *
   * トレードオフ: 保持区間は run 1 本の GPU 操作全体（エンコード〜`flush()` の完了待ち〜
   * readback〜アリーナ破棄）に及ぶため、同一 device 上の複数 Session の run は丸ごと
   * 直列化される。区間を狭めない理由は「run が GPU 操作を発行するのは自分のロック区間内
   * のみ」という不変条件を単純に保つため（executor の #runOnce を参照）。それでもこの設計を
   * 採るのは、誤帰属の帰結が「無関係な run が落ち、本来落ちるべき run が全 0 を静かに返す」
   * ことだから — 沈黙した誤値は検出手段が無く、失うスループットとは釣り合わない。
   *
   * MUST: 利用者の面に出さない。素の名前で公開すると
   * `gpu.withScopeLock(() => session.run(...))` が書けてしまい、run が同じロックを取りに行って
   * **診断も例外も出ないまま自己デッドロック**する（再入検出器は置けない — 下記 doc）。
   */
  withScopeLock<T>(body: () => Promise<T>): Promise<T>;
  /**
   * 融合 attention の整数内積変種を **device 単位で 1 度だけ**決める（遅延・メモ化）。
   *
   * `run` は判定の実体（{@link "./attention-dp4a-canary.ts"} の `decideAttentionI8a8Dot`）を
   * 呼び手が渡す形にしてある — この層から kernels / reference 層への import を作らないため
   * （逆向きの import は canary 側が張る）。MUST: 呼び出し点は Session 構築の 1 箇所だけ
   * （複数の実体を渡せる形にすると、メモが「最初に渡された判定」を意味するだけの席になる）。
   *
   * MUST: メモするのは **Promise そのもの**（値ではない）。attentionCompute "a8" の Session を
   * 並行構築すると、値でメモする形ではカナリアが 2 本走って device 単位 1 回の契約が崩れる。
   * 失敗（両腕とも sanity 帯を外した / device 消失）も同じ Promise のまま配る — device の性質は
   * 走らせ直しても変わらないので、再試行は同じ結論を得るためだけに 1 submit を払う。
   *
   * 席が持つのは変種 1 値ではなく**判定まるごと**（{@link AttentionI8a8Decision}）。「既知解と
   * 厳密一致ではなかったが帯内なので通した」という事実は判定と同じ寿命で、値に潰すと呼び手が
   * 警告を出せなくなる。
   */
  attentionI8a8Dot(run: () => Promise<AttentionI8a8Decision>): Promise<AttentionI8a8Decision>;
  /**
   * コンパイル済み compute pipeline のキャッシュ（**device 1 個につき 1 個** — 初回参照で生成）。
   *
   * 寿命が device 単位なのは、パイプラインの再利用可能性が device 単位だから — 同一 device 上の
   * Session はブラウザ側の暗黙キャッシュ（WGSL 文字列がキー）に当たるだけで、明示キャッシュを
   * Session ごとに割ると `createShaderModule` / `createComputePipeline` の呼び出しと
   * `getBindGroupLayout` の解決を Session の本数だけ払い直すことになる。
   *
   * MUST: 破棄・消失で明示的に捨てない（**GpuContext インスタンスと心中する** — ADR 0004 の
   * 再構築規律）。lazy 生成なので、destroy で undefined に戻すと次の参照が「死んだ device の
   * キャッシュ」を新品として作り直してしまう。作り直しの唯一の入口は `acquireGpu()` からの
   * GpuContext 再構築で、その時点で新しい空のキャッシュになる。
   * MUST: 呼び手（executor）は **Session の構築相からは触らない** — 構築相はスコープロックの
   * 外にあり、そこでパイプラインを生成すると並行構築の errorScope が誤帰属する
   * （{@link GpuContext} 冒頭「errorScope 区間の不変条件」）。
   */
  pipelines(): PipelineCache;
};

/**
 * device が使える状態かの同期判定（常駐テンソル経路の受付門 — ADR 0054）。
 *
 * MUST: `lost` だけでなく `destroyRequested` も見る。`destroy()` はフラグを同期に立てるのに
 * `device.lost` の reaction が走るのは以後のタスクなので、`lost` だけだとその窓で操作が通る。
 * 通したときの現れ方が沈黙なのがここを置く理由 — {@link ResidentTensor.write} は破棄済み
 * バッファへの**沈黙 no-op**（警告すら出ない）になり、{@link GpuContext.createResident} は
 * 消失後の `popErrorScope` が null で resolve する（`docs/research/2026-08-16-device-lost-wait-settlement.md`）
 * ため**無効なバッファを掴んだまま成功として返る**。どちらも loud になるのは次のフェンスで、
 * その間の診断は誤導的になる。
 * MUST: 型は {@link GpuDeviceLostError}（`runtime/generation-context.ts` の `assertDeviceUsable`
 * と同じ規律 — lost device 由来の GPU 資源は WebGPU 仕様上回復不能で、意図的な破棄と予期しない
 * 消失で復旧手段は変わらないので型は分けない）。
 */
const assertDeviceUsable = (gpu: GpuContext, where: string): void => {
  if (gpu.destroyRequested || gpu.lost !== undefined) {
    throw new GpuDeviceLostError(
      `${where}: device が失われた（device を取り直して作り直すこと）` +
        describeDeviceLoss(gpu.lost),
    );
  }
};

/**
 * 取得済み device と正規化済み能力の束。
 *
 * `device.lost` の購読はコンストラクタで**無条件に、かつ 1 回だけ**行う。消失を未処理の
 * まま放置すると `mapAsync` / `onSubmittedWorkDone` が永久に解決せず、失敗ではなくハングと
 * して現れる。購読を任意にしないことで、消失は必ず {@link GpuContext.lost} に記録され、
 * この層の待機は例外に変換される。
 *
 * ## errorScope 区間の不変条件（device 単位・この層が守らせる）
 *
 * errorScope は **device 単位の LIFO スタック**で、`popErrorScope()` は呼んだ時点のスタック
 * 先頭を無条件に取る。つまり「誰のスコープか」という概念が無い。ここから 3 つの規則が出る:
 *
 * - MUST: `await` を跨いで errorScope を張る区間は `withScopeLock`
 *   （{@link GpuContextInternals}）の中で実行する。重なると①自分のエラーが他人のスコープに
 *   入り②自分の pop が他人のスコープを取るため、失敗が無関係な呼び出しに帰属し、本来
 *   落ちるべき呼び出しは沈黙のまま全 0 を返す。
 * - MUST NOT: `withScopeLock` の中で `withScopeLock` を再取得する（自己デッドロック）。
 *   ロックは再入可能ではない。検出器は置かない — 「保持中」フラグでは**正当な待ち行列**
 *   （別 Session が先行区間の完了を待って並ぶ形）と再入が区別できず、区別するには async
 *   呼び出しを跨ぐ実行コンテキスト追跡が要る。Web 標準 API のみという制約（ADR 0002）の下で
 *   純粋な手段が無いため、取得点を executor の 1 箇所に限定し、その内側の層（PipelineCache /
 *   SubmitScheduler / RunArena）は同期区間で完結するスコープしか使わない、という層規約で守る。
 * - 同期区間だけで完結するスコープ（{@link "./error-scope.ts"} の `withPipelineScope` /
 *   {@link pushFailureScopes}〜{@link popFailureScopes}）はロック不要。他のタスクが割り込む
 *   余地が無く、LIFO の入れ子が必ず均衡するため。ロック内から呼ばれる層（PipelineCache 等）
 *   はこの形でなければならない。
 */
export class GpuContext {
  /** ランタイム内部面（利用者が触る面ではない）。 */
  readonly [RUNTIME_INTERNAL]: GpuContextInternals;
  /**
   * 生の device（ランタイムの管理外で触る面）。ここから `destroy()` を呼ぶと
   * {@link "./acquire.ts"} の `AcquireGpuOptions.onDeviceLost` の抑止が効かず予期しない消失として通知され、
   * `pushErrorScope` / `popErrorScope` を張ると errorScope 区間ロック（LIFO の排他）に参加しない。
   * 後始末は {@link GpuContext.destroy} を使うこと。
   */
  readonly device: GPUDevice;
  readonly adapterInfo: GPUAdapterInfo;
  /** device が実際に有効化した feature（アダプタが持つだけの feature は含まない）。 */
  readonly features: ReadonlySet<string>;
  /** 参考情報。未実装環境では空集合。機能検出には使わない。 */
  readonly wgslLanguageFeatures: ReadonlySet<string>;
  readonly limits: RequiredLimits;
  #lost: GPUDeviceLostInfo | undefined;
  #destroyRequested = false;
  readonly #lostListeners = new Set<DeviceLostHandler>();
  /**
   * errorScope 区間の直列化チェーン。決着（成功・失敗）だけを次に渡すため自身は決して
   * reject しない（1 本の失敗で以後の全区間を巻き添えにしない）。
   */
  #scopeChain: Promise<void> = Promise.resolve();
  /**
   * 融合 attention の内積変種カナリアの結果（{@link GpuContextInternals.attentionI8a8Dot}）。
   * 未要求なら undefined のまま = 1 dispatch も出ない（a8 を使わない利用者はコストを払わない）。
   */
  #attentionI8a8Dot: Promise<AttentionI8a8Decision> | undefined;
  /**
   * {@link ResidentTensor} の識別子の発番。**モジュールスコープに置かない**（副作用ゼロの
   * 不変条件）— GpuContext ごとに別空間で足りる（resident は device を跨がない）。
   */
  #residentSeq = 0;
  /**
   * device 寿命のパイプラインキャッシュ（{@link GpuContextInternals.pipelines}）。
   * 未参照なら undefined のまま = 1 本も作らない（Session を作らない利用者はコストを払わない）。
   */
  #pipelines: PipelineCache | undefined;

  constructor(
    device: GPUDevice,
    adapterInfo: GPUAdapterInfo,
    limits: RequiredLimits,
    wgslLanguageFeatures: ReadonlySet<string>,
    onDeviceLost?: DeviceLostHandler,
  ) {
    this.device = device;
    this.adapterInfo = adapterInfo;
    this.limits = limits;
    this.features = new Set(device.features);
    this.wgslLanguageFeatures = wgslLanguageFeatures;
    this[RUNTIME_INTERNAL] = {
      raceDeviceLost: <T>(work: Promise<T>, where: string): Promise<T> =>
        this.#raceDeviceLost(work, where),
      withScopeLock: <T>(body: () => Promise<T>): Promise<T> => this.#withScopeLock(body),
      attentionI8a8Dot: (
        run: () => Promise<AttentionI8a8Decision>,
      ): Promise<AttentionI8a8Decision> => {
        this.#attentionI8a8Dot ??= run();
        return this.#attentionI8a8Dot;
      },
      pipelines: (): PipelineCache => {
        this.#pipelines ??= new PipelineCache(this.device);
        return this.#pipelines;
      },
    };
    if (onDeviceLost !== undefined) {
      this.onLost((info) => {
        if (!this.#destroyRequested) {
          onDeviceLost(info);
        }
      });
    }
    void device.lost.then((info) => {
      this.#lost = info;
      // 通知中の解除で反復が壊れないよう複製してから呼ぶ。消失は 1 度きりなので、通知後は
      // 購読を空にして以後の onLost を即時通知経路に落とす。
      const listeners = [...this.#lostListeners];
      this.#lostListeners.clear();
      for (const listener of listeners) {
        // MUST: listener の例外は通知の fan-out を止めない。購読は上で clear 済みで再通知が
        // 無いため、公開 onDeviceLost（挿入順で先）の throw で後続の内部購読
        // （`raceDeviceLost`）へ通知が届かないと、消失が例外ではなくハングになる。捕えた例外は
        // 握り潰さず、消失の制御経路から切り離して非同期に再 throw する。
        try {
          listener(info);
        } catch (cause) {
          queueMicrotask(() => {
            throw cause;
          });
        }
      }
    });
  }

  /** 消失済みならその情報。未消失は undefined。`destroy()` 後も記録される。 */
  get lost(): GPUDeviceLostInfo | undefined {
    return this.#lost;
  }

  /** 意図的な破棄を要求済みか（予期しない消失と区別するため）。 */
  get destroyRequested(): boolean {
    return this.#destroyRequested;
  }

  /**
   * GPU 側時間計測が有効か（ADR 0021）。
   *
   * MUST: 要求値を別フィールドに複製せず、**実際に有効化された feature** から導く。
   * device の feature 集合は要求したものそのものなので、複製すると「要求したのに載らなかった」
   * ときに診断だけが有効を主張する形になる（内訳が空のまま「計測中」に見える）。
   */
  get gpuTimingEnabled(): boolean {
    return this.features.has(TIMESTAMP_QUERY_FEATURE);
  }

  /**
   * f16 計算変種が使えるか（ADR 0028）。
   *
   * MUST: {@link GpuContext.gpuTimingEnabled} と同じ規律で、要求値の複製ではなく**実際に
   * 有効化された feature** から導く。ここが true を返すのは `acquireGpu({shaderF16: true})`
   * が feature の要求とカナリアの実走の両方を通ったときだけ。
   */
  get shaderF16Enabled(): boolean {
    return this.features.has(SHADER_F16_FEATURE);
  }

  /**
   * 未解除の消失購読の本数（診断）。
   *
   * `raceDeviceLost`（{@link GpuContextInternals}）は決着時に必ず解除するので、待機が全て
   * 決着していれば 0 に戻る。0 に戻らないことが「flush / readback ごとに reaction が積み残る」
   * リークの姿で、挙動からは観測できない（ハングも誤値も起こさず、長寿命 Session で単調増加
   * するだけ）。
   * 見えない残留を正直に数値で出すためだけの面。
   */
  get pendingLostListeners(): number {
    return this.#lostListeners.size;
  }

  /**
   * device 消失を購読する。戻り値の関数で解除する。購読時点で既に消失していれば**即時
   * （同期）**に通知して取りこぼしを作らない。
   *
   * MUST: 待機のたびに `device.lost.then(...)` を新しく張らない。`lost` は device の寿命の間
   * 未解決のままなので、`.then` は解除手段の無い reaction として promise に積まれ続け、
   * flush / readback ごとに単調増加する（長寿命 Session でのリーク）。購読はここに一本化し、
   * 待機側は `raceDeviceLost`（{@link GpuContextInternals}）を使って決着時に必ず解除する。
   */
  onLost(listener: DeviceLostHandler): () => void {
    const info = this.#lost;
    if (info !== undefined) {
      listener(info);
      return () => {};
    }
    this.#lostListeners.add(listener);
    return () => {
      this.#lostListeners.delete(listener);
    };
  }

  /** {@link GpuContextInternals.raceDeviceLost} の実体（面は `RUNTIME_INTERNAL` の下だけ）。 */
  async #raceDeviceLost<T>(work: Promise<T>, where: string): Promise<T> {
    let unsubscribe: () => void = () => {};
    const lost = new Promise<never>((_resolve, reject) => {
      unsubscribe = this.onLost((info) => {
        reject(
          new GpuDeviceLostError(
            `${where} 中に device が失われた（再構築が必要）${describeDeviceLoss(info)}`,
          ),
        );
      });
    });
    try {
      return await Promise.race([work, lost]);
    } finally {
      unsubscribe();
    }
  }

  /** {@link GpuContextInternals.withScopeLock} の実体（面は `RUNTIME_INTERNAL` の下だけ）。 */
  #withScopeLock<T>(body: () => Promise<T>): Promise<T> {
    const result = this.#scopeChain.then(body);
    this.#scopeChain = result.then(() => undefined, () => undefined);
    return result;
  }

  /**
   * Session を跨いで共有できる常駐バッファを作る（**第 4 の寿命クラス** — 重みアリーナ /
   * slot backing / run アリーナのどれにも属さない）。
   *
   * 用途は「生成ループの間ずっと GPU に置いたままにしたい値」— 条件テンソル（ループ前に
   * {@link ResidentTensor.write} で 1 度だけ投入）と、ステップ間で受け渡す潜在
   * （{@link Session.enqueue} の `copyOutputs` で書き、次の enqueue の入力にする）。どちらも
   * ホストを 1 度も経由しないので、run 境界のフェンスを消せる。
   *
   * MUST: async なのは errorScope で囲むため。上限超過 / 余力切れの `createBuffer` は同期
   * 例外を投げず**無効なバッファを返す**ので、囲まないと以後の `writeBuffer` が警告すら
   * 出さない no-op になり、空の常駐テンソルのまま生成ループが回る。
   * MUST: `byteLength` は 4 の倍数（要素は全型 4 バイト — ADR 0009 の意味論 dtype）。
   * MUST: 消失済み device では受け付けない（{@link assertDeviceUsable} — errorScope は
   * 消失を捕らえないので、囲んでいても無効なバッファが成功として返る）。
   */
  async createResident(byteLength: number, label = "resident"): Promise<ResidentTensor> {
    assertDeviceUsable(this, `resident '${label}' の確保`);
    if (!Number.isInteger(byteLength) || byteLength <= 0 || byteLength % 4 !== 0) {
      throw new ResidentTensorError(
        `resident '${label}': byteLength は 4 の倍数の正の整数である必要がある: ${byteLength}`,
      );
    }
    // MUST: push から pop の発行までに await を挟まない（device 単位 LIFO の交錯を防ぐ根拠 —
    // クラス冒頭「errorScope 区間の不変条件」の 3 つ目）。
    pushFailureScopes(this.device);
    let buffer: GPUBuffer;
    try {
      buffer = this.device.createBuffer({ label, size: byteLength, usage: STORAGE_USAGE });
    } catch (cause) {
      await discardFailureScopes(this.device);
      throw cause;
    }
    const failure = await popFailureScopes(this.device, `resident '${label}' の確保`);
    if (failure !== undefined) {
      buffer.destroy();
      throw failure;
    }
    this.#residentSeq += 1;
    return new ResidentTensor(this, this.#residentSeq, buffer, byteLength, label);
  }

  /**
   * フェンス無しの enqueue を束ねる区間を開く（{@link Session.enqueue} — H-5）。
   *
   * 区間の間 **device 単位の errorScope 区間ロックを保持し続ける**
   * （`withScopeLock` — {@link GpuContextInternals}）。これが「batch の内側で出した GPU 操作は
   * 全て batch の errorScope に帰属する」ことと、「pop がスタック先頭を取り違えない」ことの根拠。
   *
   * MUST: 区間中に同一 device の {@link Session.run} / `dispose` を**発行しない**（await の
   * 有無に依らない）。run は区間ロックを取りに行くので {@link BatchScope.finish} まで決着せず、
   * その run が同一 Session の `enqueue` より前に居ると「finish → in-flight リース → enqueue 本体
   * → 先行 run → 区間ロック」の 4 辺が閉じて確定的な自己デッドロックになる（区間を開く**前**に
   * 発行した未 await の run も同じ — `beginBatch` はコンストラクタで同期にロックを先取りするので、
   * 同一 tick なら常に batch が先）。この形は `Session.enqueue` が {@link BatchScopeError} へ
   * 変換する。
   * NOTE: 閉路にならない形でも、**区間中に await する Session 操作全般**（`session.dispose()` /
   * `context.dispose()`）は先行に未決着 run があると `finish()` まで返らず、利用者からは
   * ハングに見える。batch 中は `enqueue` と `finish` だけを使うこと。**Session の構築も同じ**
   * — `attentionCompute: "a8"` の初回構築はカナリア（{@link GpuContextInternals.attentionI8a8Dot}）
   * で区間ロックを取りに行くため、区間中に構築すると `finish()` まで返らない。
   * MUST: 計測が有効な device では開けない。1 dispatch = 1 pass に開いた timestamp は
   * flush でしか回収されないため、batch の間 N run 分が未回収で溜まる（内訳を取るなら
   * 通常の run で計測すること — ADR 0021）。
   * NOTE: 同一 device で 2 本目を開こうとすると、1 本目が {@link BatchScope.finish} で
   * ロックを返すまでここで待つ（区間は device 単位で排他 — 入れ子にはならない）。
   */
  async beginBatch(): Promise<BatchScope> {
    if (this.gpuTimingEnabled) {
      throw new BatchScopeError(
        "gpuTiming が有効な device では batch を開けない（1 dispatch = 1 pass に開いた " +
          "timestamp が flush まで回収されず、batch の間 run 数ぶん溜まる）。" +
          "GPU 時間内訳は通常の run で計測すること",
      );
    }
    const batch = new BatchScope(this);
    await batch[RUNTIME_INTERNAL].entered;
    return batch;
  }

  /**
   * device を破棄する。VRAM を返すのはこの経路のみ。
   *
   * MUST: 未 submit のエンコードと生存中のバッファを持つ層（RunArena）を先に flush /
   * destroy してから呼ぶこと。破棄後の続行は `acquireGpu()` での再構築になる。
   */
  destroy(): void {
    this.#destroyRequested = true;
    this.device.destroy();
  }
}

/** {@link ResidentTensor} のランタイム内部面。 */
type ResidentInternals = {
  /** 焼き込み / 別名の対象になる実体。 */
  readonly buffer: GPUBuffer;
  /**
   * GpuContext 内で一意な識別子。**導出済み計画のキーと backing signature に載る**ので、
   * resident を差し替えれば別 signature（= 焼き直し）になり、戻せば元の backing に当たる。
   */
  readonly id: number;
  /**
   * この実体を確保した GpuContext。
   *
   * MUST: 束縛する側（executor）が「自分の device の resident か」を照合するために載せる。
   * {@link id} は GpuContext ごとの独立採番なので、別 context の同 id・同サイズな resident は
   * 導出済み計画のキーが衝突する — ヒット run（焼き込み済み backing）では渡された実体が
   * 一切参照されないため、**例外も警告も無く前の context の古い値を読む**。ミス経路だけは
   * device 不一致の validation で偶然落ちるが、キャッシュが当たった瞬間に検出が消える。
   * {@link BatchScope} が owner を検査しているのと同じ門をここにも置く。
   */
  readonly owner: GpuContext;
  /** 焼き込み bind group からの参照を 1 本積む（backing の構築時）。 */
  retainBaked(): void;
  /** 焼き込み参照を 1 本返す（backing の退役時）。 */
  releaseBaked(): void;
  /**
   * **進行中の run が入力として束ねた**ことを 1 本積む（焼き込み参照とは別枠）。
   *
   * 焼き込み参照は backing が生きている間の静的な参照だが、こちらはミス run の
   * 「env へ生バッファを束縛 → パイプライン生成を await → エンコード」という窓を塞ぐための
   * 予約。この窓では焼き込みがまだ 1 本も無いため、参照計数だけでは dispose が素通りする。
   */
  retainBound(): void;
  /** 進行中 run の束縛予約を 1 本返す（run のエンコードと submit が済んだ時点）。 */
  releaseBound(): void;
  /**
   * **発行済み（未決着）の run / enqueue が使う**ことを 1 本積む（焼き込み・束縛とは別枠）。
   *
   * MUST: 取るのは `Session.run` / `Session.enqueue` の**発行の同期区間**。実行本体は
   * マイクロタスクを 1 段挟むので、本体（束縛予約 {@link retainBound}）で取ると、焼き込みも
   * 束縛もまだ 1 本も立っていない `bakedRefs === 0` の resident では「API が受理した run が
   * 居るのに同じ tick の `dispose()` が通る」窓が開く。その run は後で fail loudly になるが、
   * 失敗地点が dispose の呼び出し点から run の決着へずれ、{@link ResidentTensor.dispose} の
   * doc が謳う「誤りは dispose の呼び出し点で真因のまま落ちる」が受理済み run に対して
   * 成り立たなくなる。入力だけでなく `copyOutputs` の写し先も対象。
   */
  retainUse(): void;
  /** 使用予約を 1 本返す（run / enqueue の決着時 — 成功・失敗のどちらの経路でも必ず）。 */
  releaseUse(): void;
};

/** ホストから常駐テンソルへ書ける配列（要素は全型 4 バイト — ADR 0009 と同じ規約）。 */
export type ResidentData =
  | Float32Array<ArrayBuffer>
  | Int32Array<ArrayBuffer>
  | Uint32Array<ArrayBuffer>;

/**
 * GpuContext が所有する常駐バッファ（**第 4 の寿命クラス**）。
 *
 * 既存 3 クラス（重みアリーナ / slot backing / run アリーナ）はどれも Session の内側に閉じて
 * いて、Session を跨いだ受け渡しは必ずホスト経由（readback → writeBuffer）になる。生成ループ
 * のようにグラフ間で値を回す形では、その 1 往復ごとにフェンスが 2 本立つ。ここはその往復を
 * 消すためだけの器で、**dtype も shape も持たない**（バイト列と大きさだけ）。
 *
 * MUST: 破棄は「参照されていないこと」を確かめてから（{@link ResidentTensor.dispose}）。
 * flush-before-destroy（ADR 0004）は次の 2 つで満たしている — ①焼き込み bind group からの参照が
 * 1 本でもある間は破棄を拒む ②`enqueue` は末尾で必ず eager submit するので、戻った時点で
 * この実体を参照する**未 submit の**エンコードは存在しない（submit 済みのコマンドが参照する
 * バッファの破棄は WebGPU 的に安全 — 実解放は完了まで実装が遅延する）。
 */
export class ResidentTensor {
  /** 確保したバイト数（要求値そのもの — 4 の倍数）。 */
  readonly byteLength: number;
  /** 診断用の名前（GPUBuffer のラベルと同じ）。 */
  readonly label: string;
  /** ランタイム内部面（利用者が触る面ではない）。 */
  readonly [RUNTIME_INTERNAL]: ResidentInternals;
  readonly #gpu: GpuContext;
  readonly #buffer: GPUBuffer;
  #disposed = false;
  #bakedRefs = 0;
  #boundRefs = 0;
  #useRefs = 0;

  /** MUST: 構築の入口は {@link GpuContext.createResident} だけ（errorScope の門を迂回させない）。 */
  constructor(gpu: GpuContext, id: number, buffer: GPUBuffer, byteLength: number, label: string) {
    this.#gpu = gpu;
    this.#buffer = buffer;
    this.byteLength = byteLength;
    this.label = label;
    this[RUNTIME_INTERNAL] = {
      buffer,
      id,
      owner: gpu,
      retainBaked: () => {
        this.#assertUsable("焼き込み");
        this.#bakedRefs += 1;
      },
      releaseBaked: () => {
        if (this.#bakedRefs === 0) {
          throw new ResidentTensorError(
            `resident '${this.label}': 焼き込み参照の解放が過多（ランタイム内部の簿記の破れ）`,
          );
        }
        this.#bakedRefs -= 1;
      },
      retainBound: () => {
        this.#assertUsable("束縛");
        this.#boundRefs += 1;
      },
      releaseBound: () => {
        if (this.#boundRefs === 0) {
          throw new ResidentTensorError(
            `resident '${this.label}': 束縛予約の解放が過多（ランタイム内部の簿記の破れ）`,
          );
        }
        this.#boundRefs -= 1;
      },
      retainUse: () => {
        this.#assertUsable("使用");
        this.#useRefs += 1;
      },
      releaseUse: () => {
        if (this.#useRefs === 0) {
          throw new ResidentTensorError(
            `resident '${this.label}': 使用予約の解放が過多（ランタイム内部の簿記の破れ）`,
          );
        }
        this.#useRefs -= 1;
      },
    };
  }

  /** 破棄済みか。 */
  get disposed(): boolean {
    return this.#disposed;
  }

  /** 焼き込み bind group から参照されている本数（0 でなければ破棄できない）。 */
  get bakedReferences(): number {
    return this.#bakedRefs;
  }

  /** 進行中の run が入力として束ねている本数（0 でなければ破棄できない）。 */
  get boundReferences(): number {
    return this.#boundRefs;
  }

  /**
   * 発行済み（未決着）の run / enqueue が入力または `copyOutputs` の写し先として使っている
   * 本数（0 でなければ破棄できない）。
   */
  get useReferences(): number {
    return this.#useRefs;
  }

  /**
   * ホストから全域を書く（`queue.writeBuffer`）。生成ループに入る**前**の条件テンソル投入用。
   *
   * MUST: 大きさは厳密一致。部分書きを許すと残りのバイトが前の内容のまま残り、例外も警告も
   * 出ないまま古い条件で回る（full-write — ADR 0014 と同じ思想）。
   * MUST NOT: **発行済み（未決着）の run / enqueue がこの実体を入力に取っている間は書かない**。
   * 実行本体は 1 マイクロタスク以降に走るので、発行直後の同一 tick の write はその dispatch より
   * 先に queue へ載り、**発行時に意図した値ではなく後から書いた値**を dispatch に読ませる
   * （例外も警告も出ない）。書いてよいのは戻り Promise が settle した後 — 非 await で回す
   * `enqueue` なら {@link BatchScope.finish} の後。ホスト入力 `Tensor.data`（`Session.run` の
   * 「入力の寿命」節の MUST NOT）と**同じ borrowed 側の契約**で、機構の門は置かない（値の
   * 複製を毎回払わないための線引きも同じ）。
   * MUST: この `writeBuffer` は issue 順で queue timeline に載るので、**先に submit 済みの
   * dispatch を追い越さない**。追い越すのは未 submit のエンコードだけ（ADR 0004 不変条件④）
   * で、`enqueue` はその末尾で必ず submit してから戻るため、**決着済みの** enqueue とは
   * 競合しない。これは queue timeline の順序の主張であって、上の MUST NOT が言う「発行したが
   * 本体がまだ走っていない窓」を守るものではない（そちらは呼び出し側の契約）。
   * MUST: 消失済み device では書かない（{@link assertDeviceUsable}）。`queue.writeBuffer` は
   * 破棄済みバッファに対して例外も警告も出さない no-op なので、ここで止めないと空の条件の
   * まま生成ループが回る。
   */
  write(data: ResidentData): void {
    this.#assertUsable("write");
    assertDeviceUsable(this.#gpu, `resident '${this.label}' の write`);
    if (data.byteLength !== this.byteLength) {
      throw new ResidentTensorError(
        `resident '${this.label}': write のバイト数 ${data.byteLength} が確保 ${this.byteLength} と合わない`,
      );
    }
    this.#gpu.device.queue.writeBuffer(this.#buffer, 0, data);
  }

  /**
   * 全域をホストへ読み戻す（staging へ copy → `mapAsync`）。**フェンスはこの 1 本だけ**で、
   * 生成ループの終端で潜在を 1 度取り出すために置いてある。
   *
   * MUST: 呼ぶのは {@link BatchScope.finish} の**後**。queue の順序は保たれるので値としては
   * 正しいが、batch の内側で呼ぶとフェンスが 1 本増え、しかもこの submit の失敗が batch の
   * errorScope に帰属して原因の切り分けができなくなる。
   *
   * MUST: 消失済み device では読まない（{@link assertDeviceUsable} — `write` /
   * {@link GpuContext.createResident} と同じ門）。`destroy()` 直後の窓（フラグは同期に立つが
   * `device.lost` の reaction は次のタスク）では破棄済み device 上で staging を確保して copy を
   * submit してしまい、消失後の `popErrorScope` は null で resolve するので失敗は捕まらない。
   */
  async read(): Promise<ArrayBuffer> {
    this.#assertUsable("read");
    const device = this.#gpu.device;
    const where = `resident '${this.label}' の読み戻し`;
    assertDeviceUsable(this.#gpu, where);
    // MUST: copy → submit も errorScope の両建てで囲む。COPY_SRC 欠落等の validation 失敗も
    // staging の確保失敗も例外にならず、読み戻しが全 0 のまま静かに返る（#readOutputs と同じ）。
    // MUST NOT: push から pop の発行までに await を挟まない（同期区間で完結するのでロック不要）。
    pushFailureScopes(device);
    let popped = false;
    let staging: GPUBuffer | undefined;
    try {
      staging = device.createBuffer({
        size: this.byteLength,
        usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ,
      });
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(this.#buffer, 0, staging, 0, this.byteLength);
      device.queue.submit([encoder.finish()]);
      const pending = popFailureScopes(device, where);
      popped = true;
      const failure = await pending;
      if (failure !== undefined) throw failure;
      // MUST: 消失後の mapAsync が解決しない実装がありうる（実測は raceCanaryDeviceLost の
      // doc）ため競わせる — ハングを失敗に変換する保険。
      await this.#gpu[RUNTIME_INTERNAL].raceDeviceLost(staging.mapAsync(MAP_MODE.READ), where);
      const copy = staging.getMappedRange().slice(0);
      staging.unmap();
      return copy;
    } finally {
      if (!popped) await discardFailureScopes(device);
      // MUST: staging は成否によらず必ず返す（destroy は暗黙 unmap を含む）。
      staging?.destroy();
    }
  }

  /**
   * 破棄する（2 度目以降は no-op）。
   *
   * MUST: 焼き込み bind group から参照されている間は **fail loudly**。黙って破棄すると、
   * その backing の dispatch が破棄済みバッファを束ねたまま submit され、コマンドバッファ
   * ごと失敗して**無関係な dispatch まで実行されないまま誤った値が静かに残る**（ADR 0004）。
   * 参照を外すには、その Session を dispose するか、別 signature の run / enqueue で backing を
   * 切り替える。
   * MUST: **進行中の run が入力として束ねている間**も同じく fail loudly。焼き込みが立つのは
   * backing 構築（= ヒット run 以降）なので、ミス run の「束縛 → パイプライン生成の await →
   * エンコード」の窓は焼き込み参照だけでは守れない（その窓で破棄すると、再開したエンコードが
   * 破棄済みバッファを掴んで run が validation で落ちる）。この予約があると、誤りは
   * dispose の呼び出し点で真因のまま落ちる。
   * MUST: **発行済み（未決着）の run / enqueue が入力または `copyOutputs` の写し先として
   * 使っている間**も同じく fail loudly。束縛予約が立つのは実行本体（マイクロタスク以降）なので、
   * 「API が受理した直後の同じ tick」は焼き込みも束縛も 0 本 — その窓を塞ぐのは発行の同期区間で
   * 取る使用予約（{@link ResidentInternals.retainUse}）だけで、上の保証が受理済みの run / enqueue
   * に対して成り立つ根拠もそこにある。
   */
  dispose(): void {
    if (this.#disposed) return;
    if (this.#bakedRefs > 0) {
      throw new ResidentTensorError(
        `resident '${this.label}': 焼き込み bind group から参照中（${this.#bakedRefs} 本）のため破棄できない。` +
          "参照している Session を dispose するか、別 signature の run / enqueue で backing を切り替えること",
      );
    }
    if (this.#boundRefs > 0) {
      throw new ResidentTensorError(
        `resident '${this.label}': 進行中の run が入力として束縛中（${this.#boundRefs} 本）のため破棄できない。` +
          "その run の完了を await してから破棄すること",
      );
    }
    // 焼き込み・束縛より後に見る（どちらもこれより狭い窓を名指しするので、診断としては
    // そちらが先に出る方が近い）。
    if (this.#useRefs > 0) {
      throw new ResidentTensorError(
        `resident '${this.label}': 発行済みの run / enqueue が入力または写し先として使用中` +
          `（${this.#useRefs} 本）のため破棄できない。` +
          "その run / enqueue の決着を await してから破棄すること",
      );
    }
    this.#disposed = true;
    this.#buffer.destroy();
  }

  [Symbol.dispose](): void {
    this.dispose();
  }

  #assertUsable(where: string): void {
    if (this.#disposed) {
      throw new ResidentTensorError(
        `resident '${this.label}': 破棄済みの常駐テンソルは使えない（${where}）`,
      );
    }
  }
}

/**
 * batch が決着時に取りまとめる相手（実体は {@link SubmitScheduler}）。
 * 構造型にしてあるのは、device 層が submit 層へ依存しないため。
 */
export type BatchMember = {
  /** 未 submit のエンコードを submit する（**フェンスは張らない**）。 */
  submitPending(): void;
  /** batch のフェンス完了後に計測窓を閉じる。 */
  closeMeasurementWindowAfterFence(): void;
  /** フェンスに到達しなかった batch の計測窓を記録せずに捨てる。 */
  discardMeasurementWindow(): void;
};

/** バッチの成功判定後に状態を確定し、成否によらず使用予約を返す。 */
type BatchFinalizer = {
  complete(): void;
  fail(cause: unknown): void;
  release(): void;
};

/** 区間の決着時に 1 本の staging へ連結コピーして読み戻す写し元（{@link BatchInternals.readAtFinish}）。 */
export type BatchReadSource = {
  readonly buffer: GPUBuffer;
  readonly offset: number;
  readonly size: number;
};

/** {@link BatchInternals.readAtFinish} で登録したグラフ出力の読み戻し 1 件（決着まで保持）。 */
type GraphRead = {
  readonly sources: readonly BatchReadSource[];
  readonly resolve: (buffers: readonly ArrayBuffer[]) => void;
  readonly reject: (cause: unknown) => void;
  /** 決着の読み戻しで埋まる（区間の成否が決まるまで呼び手へは返さない）。 */
  result?: readonly ArrayBuffer[];
};

/** {@link BatchScope} のランタイム内部面。 */
type BatchInternals = {
  /** errorScope 区間が実際に開くまでの待ち（{@link GpuContext.beginBatch} が await する）。 */
  readonly entered: Promise<void>;
  /** 決着時に取りまとめる相手を登録する（同じ相手の重複登録は無害）。 */
  join(member: BatchMember): void;
  /**
   * enqueue の受け口として使える状態か検査し、**in-flight リースを 1 本取る**
   * （決着済み / 別 device は fail loudly で、リースは取らない）。
   *
   * MUST: 呼ぶのは `Session.enqueue` の**同期区間**（`#chain` に積む前）。enqueue の本体は
   * マイクロタスクを 1 段挟むので、本体でリースを取ると「未 await の enqueue を積んだ直後に
   * `finish()`」で finish が先に決着し、積んだ enqueue が 1 本も dispatch されないまま区間が
   * 正常終了に見える（沈黙の空振り）。
   */
  enter(owner: GpuContext): void;
  /**
   * in-flight リースを 1 本返す（enqueue 本体の成功・失敗どちらの経路でも必ず）。
   *
   * `failure` を渡すと、その区間の**最初の 1 件だけ**をホスト側の失敗として記録し、
   * {@link BatchScope.finish} が帰属先になる（渡さなければ従来どおり何も記録しない）。
   * 記録が要るのは、非 await の `enqueue` が本体で落ちたときに「区間は 0 dispatch 少ないまま
   * 成功」で決着してしまうため — errorScope に載るのは GPU 側の失敗だけで、ホスト側の throw は
   * 戻り Promise にしか出ない（握っていなければ未処理拒否として抜ける）。
   * MUST: 2 件目以降は捨てる。1 件目に引きずられた派生失敗が並ぶと根因が読めなくなる
   * （errorScope が internal / out-of-memory を優先するのと同じ判断）。
   */
  leave(failure?: { readonly cause: unknown }): void;
  onSettled(finalizer: BatchFinalizer): void;
  /**
   * グラフ出力の slot を区間の決着時に読み戻す相手として登録する（`Session.enqueueRead`）。
   * 常駐テンソルの読み戻し（{@link BatchScope.finishAndRead}）と同じ 1 本の staging に連結し、
   * その map が唯一のフェンスになる。戻りは区間が例外なく決着した後にだけ解決し、失敗
   * （GPU 側・ホスト側とも）では拒否する。
   *
   * MUST: 呼ぶのは enqueue 本体（in-flight リースを持つ間）。`finish` は全リースの返却を待って
   * から読み戻しへ進むので、リースの中で登録した相手は取りこぼさない。読み戻しが始まった後の
   * 登録は fail loudly（解決しない Promise を返さない）。
   */
  readAtFinish(sources: readonly BatchReadSource[]): Promise<readonly ArrayBuffer[]>;
};

/**
 * フェンス無しの enqueue を束ねる区間（{@link GpuContext.beginBatch}）。
 *
 * 区間の間 device の errorScope 区間ロックを保持し、`out-of-memory` + `validation` の 2 本を
 * 張り続ける。{@link BatchScope.finish} が ①全メンバの未 submit を出し切り ②
 * `onSubmittedWorkDone` を**1 回だけ**待ち ③スコープを pop して失敗を型付き例外にする。
 *
 * トレードオフ（設計上の受容）:
 *
 * - **失敗の帰属は batch 単位**。1 区間に N 本の enqueue が相乗りするので、validation /
 *   out-of-memory が出ても「どの enqueue か」までは絞れない。切り分けが要るときは通常の
 *   `run` で 1 本ずつ回す（run は従来どおり run 単位で帰属する）。
 * - **device 消失の検出は finish まで遅延する**。区間中の待ちが 1 本も無いのだから当然で、
 *   消失は finish の `raceDeviceLost` が例外へ変換する（enqueue 側はハングしない — 待たない
 *   から）。
 */
export class BatchScope {
  /** ランタイム内部面（利用者が触る面ではない）。 */
  readonly [RUNTIME_INTERNAL]: BatchInternals;
  readonly #gpu: GpuContext;
  readonly #members = new Set<BatchMember>();
  readonly #finalizers: BatchFinalizer[] = [];
  readonly #completion: Promise<Readonly<Record<string, ArrayBuffer>> | undefined>;
  readonly #release: () => void;
  /** 未返却の in-flight リースが全て返ったことの通知（{@link BatchInternals.enter}）。 */
  readonly #drained = Promise.withResolvers<void>();
  #leases = 0;
  #readback: readonly (readonly [string, ResidentTensor])[] | undefined;
  /** 決着時に読み戻すグラフ出力（{@link BatchInternals.readAtFinish}）。 */
  readonly #graphReads: GraphRead[] = [];
  /** 決着の読み戻しへ進んだか（以後の登録は fail loudly）。 */
  #readbackStarted = false;
  #finished = false;
  /** {@link BatchScope.settle} の途中か（この間の enqueue は拒否する — 窓の帰属を守る）。 */
  #settling = false;
  /** 区間で最初に起きたホスト側の失敗（{@link BatchInternals.leave}）。2 件目以降は捨てる。 */
  #failure: { readonly cause: unknown } | undefined;
  /** {@link BatchScope.finish} が返す決着（errorScope の結果と {@link #failure} の合流）。 */
  #settled: Promise<void> | undefined;

  /** MUST: 構築の入口は {@link GpuContext.beginBatch} だけ（計測との併用の門をここに置く）。 */
  constructor(gpu: GpuContext) {
    this.#gpu = gpu;
    const hold = Promise.withResolvers<void>();
    const entered = Promise.withResolvers<void>();
    this.#release = hold.resolve;
    this.#completion = gpu[RUNTIME_INTERNAL].withScopeLock(async () => {
      try {
        pushFailureScopes(gpu.device);
      } catch (cause) {
        // MUST: 開けなかったことを beginBatch の待ちへ必ず伝える（伝えないとハングになる）。
        entered.reject(cause);
        throw cause;
      }
      entered.resolve();
      await hold.promise;
      // MUST: フェンスを張る前に in-flight の enqueue を全て決着させる。`finish()` は同期で
      // `#finished` を立てるだけなので、これが無いと「未 await の enqueue を積んだ直後の
      // finish()」で ①まだ本体が走っていない enqueue が全て reject し、区間は 0 dispatch の
      // まま**成功で決着する** ②走り出していた enqueue はフェンスと pop の後に submit し、
      // 未完了の GPU 実行を残したまま finish が返る（直後の read が古い値を返す）。
      await this.#drained.promise;
      const device = gpu.device;
      let outputs: Readonly<Record<string, ArrayBuffer>> | undefined;
      let popped = false;
      /** 完了フェンス（読み戻しの `mapAsync` か `onSubmittedWorkDone`）に到達したか。 */
      let fenced = false;
      const checkFailureScopes = async (): Promise<void> => {
        const pending = popFailureScopes(
          device,
          this.#readbackBytes() === 0 ? "batch のエンコード" : "batch のエンコードと読み戻し",
        );
        popped = true;
        const failure = await pending;
        if (failure !== undefined) throw failure;
      };
      try {
        // enqueue は末尾で必ず submit するので通常ここは空振りする。それでも出し切るのは、
        // 「batch が閉じた時点で未 submit のエンコードは 1 つも無い」を区間側の責務として
        // 閉じるため（破棄経路の担い手を増やさない — ADR 0004）。
        for (const member of this.#members) member.submitPending();
        // MUST: 消失後の onSubmittedWorkDone が解決しない実装がありうる（実測は
        // raceCanaryDeviceLost の doc）ため競わせる。
        // MUST: ここから先の登録は受けない（全リースが返った後なので来ないはずだが、来たら
        // 解決しない Promise を返すよりは fail loudly）。
        this.#readbackStarted = true;
        if (this.#readbackBytes() > 0) {
          outputs = await this.#readOutputs(
            this.#readback ?? [],
            this.#graphReads,
            checkFailureScopes,
            () => {
              fenced = true;
            },
          );
        } else {
          // 写し元が 1 本も無い登録（グラフ出力 0 本）は従来の完了フェンスで閉じる。
          for (const read of this.#graphReads) read.result = [];
          await gpu[RUNTIME_INTERNAL].raceDeviceLost(
            device.queue.onSubmittedWorkDone(),
            "batch の完了",
          );
          fenced = true;
        }
      } catch (cause) {
        if (!popped) await discardFailureScopes(device);
        throw cause;
      } finally {
        // 計測窓は batch のフェンス 1 回で閉じる。窓に N 本の enqueue が入るぶん推定は粗く
        // （過大に）出るが、過大 = チャンクが小さくなる向き = TDR に対して安全側
        // （src/gpu/submit.ts の「計測の帰属」）。
        // MUST: 閉じるのはフェンスに到達した経路だけ。読み戻しが `mapAsync` より前の errorScope
        // 検査で落ちた窓は GPU 実行の途中にあり、閉じると過小な実測が推定に入ってチャンク上限が
        // 恒久的に開く（危険側）。その窓は捨てて既存の推定を据え置く。
        for (const member of this.#members) {
          if (fenced) member.closeMeasurementWindowAfterFence();
          else member.discardMeasurementWindow();
        }
      }
      if (!popped) {
        await checkFailureScopes();
        // MUST: pop 待ちの間の消失はフェンスの競合に掛からない（`raceDeviceLost` は待ちを抜けた
        // 時点で購読を外す）。消失後の pop は null で決着する（docs/research の
        // 2026-08-16-device-lost-wait-settlement）ので、ここで見ないと消失を跨いだ finish が
        // 成功で返り「区間は無事に閉じた」と誤読される。
        assertDeviceUsable(gpu, "batch の完了");
      }
      return outputs;
    });
    // 決着を finish が受け取るまで未処理拒否にしない（拒否の中身は finish がそのまま返す）。
    void this.#completion.catch(() => undefined);
    this[RUNTIME_INTERNAL] = {
      entered: entered.promise,
      join: (member) => {
        this.#members.add(member);
      },
      onSettled: (finalizer) => {
        if (this.#finished) {
          throw new BatchScopeError("batch finalizer must be registered at admission");
        }
        this.#finalizers.push(finalizer);
      },
      readAtFinish: (sources) => {
        if (this.#readbackStarted) {
          throw new BatchScopeError("決着の読み戻しが始まった batch にはグラフ出力を登録できない");
        }
        const total = sources.reduce((sum, source) => sum + source.size, 0);
        this.#assertReadbackBytes(this.#readbackBytes() + total, "グラフ出力の読み戻し");
        const { promise, resolve, reject } = Promise.withResolvers<readonly ArrayBuffer[]>();
        // 決着を呼び手が受け取るまで未処理拒否にしない（拒否の中身は finish がそのまま返す）。
        void promise.catch(() => undefined);
        this.#graphReads.push({ sources, resolve, reject });
        return promise;
      },
      enter: (owner) => {
        if (owner !== this.#gpu) {
          throw new BatchScopeError("別の GpuContext で開いた batch には enqueue できない");
        }
        if (this.#finished) {
          throw new BatchScopeError("finish() 済みの batch には enqueue できない");
        }
        if (this.#settling) {
          throw new BatchScopeError(
            "settle() の途中の batch には enqueue できない（窓を閉じる前に submit が混ざると、" +
              "フェンスが待っていない dispatch まで実測に入って推定が過小 = チャンクが TDR 域へ膨らむ）",
          );
        }
        this.#leases += 1;
      },
      leave: (failure) => {
        if (this.#leases === 0) {
          throw new BatchScopeError("batch の in-flight リースの返却が過多（内部の簿記の破れ）");
        }
        // MUST: 記録は `#drained` を解決する前（区間本体はその解決の先で pop と合流する）。
        this.#failure ??= failure;
        this.#leases -= 1;
        if (this.#leases === 0 && this.#finished) this.#drained.resolve();
      },
    };
  }

  /** 決着済みか（{@link BatchScope.finish} を 1 度でも呼んだか）。 */
  get finished(): boolean {
    return this.#finished;
  }

  /**
   * 区間の**途中**でフェンスを 1 本だけ張り、ここまでに出した dispatch の実測で
   * {@link SubmitScheduler} の推定を裏付ける（perf-ledger P-2）。
   *
   * 区間の計測窓は {@link BatchScope.finish} の 1 回でしか閉じないので、区間の間ずっと
   * 「実測 0」= チャンクは `initialChunkSize` に据え置かれる（submit.ts の不変条件 1 —
   * 実測 0 は成長の根拠にならない）。数万 dispatch の区間（irodori の DiT ループ）では
   * submit が数千回になり、1 回 ≈0.5 ms のホスト固定費が壁時計に積む。最初の 1 実行を
   * await した直後にここを 1 度呼ぶと、残りの区間は裏付けのある予算でチャンクが伸びる。
   *
   * やることは finish の ①未 submit を出し切る ②`onSubmittedWorkDone` を 1 回待つ ③窓を
   * 閉じる、だけ（リースの決着待ちと errorScope の pop はしない — 区間は続く）。
   *
   * MUST: 呼ぶのは in-flight の enqueue が無い時点（enqueue を await した直後）。未 await の
   * enqueue が残っていれば fail loudly — その enqueue の submit がフェンスの後・窓を閉じる前に
   * 混ざると、フェンスが待っていない dispatch まで実測に入って推定が過小に出る（過小 =
   * チャンクが膨らむ = TDR 域へ向かう危険側）。待っている間の新規 enqueue も同じ理由で拒否する。
   * MUST: finish 済みの区間では fail loudly。device 消失は finish と同じく例外へ変換する。
   */
  async settle(): Promise<void> {
    if (this.#finished) throw new BatchScopeError("finish() 済みの batch では settle() できない");
    if (this.#settling) throw new BatchScopeError("settle() を重ねて呼べない");
    if (this.#leases > 0) {
      throw new BatchScopeError(
        `in-flight の enqueue が ${this.#leases} 本ある batch では settle() できない` +
          "（enqueue を await してから呼ぶこと）",
      );
    }
    this.#settling = true;
    try {
      for (const member of this.#members) member.submitPending();
      await this.#gpu[RUNTIME_INTERNAL].raceDeviceLost(
        this.#gpu.device.queue.onSubmittedWorkDone(),
        "batch の途中決着",
      );
      for (const member of this.#members) member.closeMeasurementWindowAfterFence();
    } finally {
      this.#settling = false;
    }
  }

  /**
   * 区間を閉じる。**新規 enqueue を拒否 → in-flight の enqueue が全て決着するのを待つ →
   * 未 submit を出し切る → フェンス 1 本で全 enqueue の完了を待つ → errorScope を pop して
   * 失敗を型付き例外にする**、の順で進む。
   * `finishAndRead`指定時はcopy後にerrorScopeを検査し、mapを唯一の完了フェンスにする。
   *
   * MUST: 2 度目以降も同じ完了を返す（先に返すと呼び出し側が破棄へ進み、ロックと errorScope が
   * 開いたまま残る）。
   * MUST: 区間の中で起きた**最初のホスト側失敗**もここへ帰属する（{@link BatchInternals.leave}）
   * — errorScope が何も捕らえていなければその失敗を、両方あるときは errorScope 側を投げて
   * 記録を `cause` に載せる。**破壊的な挙動変更**（0.7.0 まではホスト側の失敗が
   * `ExecutionError` 等として `finish()` から出ることは無かった）だが、これが無いと「非 await の
   * `enqueue` が本体で落ちた区間が、dispatch を 1 本落としたまま成功で決着する」— しかも
   * 戻り Promise を握っていなければ未処理拒否として抜けるだけになる。同じ失敗は enqueue の
   * 戻り Promise 側にも従来どおり出る（1 つの事実が 2 経路で見えるのは `run` と同じ）。
   * MUST: {@link BatchScope.settle} の途中では閉じられない（fail loudly）。区間の遷移は
   * open → settling → open → finished の 1 方向で、飛び越えを許すと settle 側のフェンスが
   * errorScope の pop より後に解け、未 await の `settle()` が未処理拒否として抜ける
   * （finish は既に決着を返しているので事実が 1 つ消える）。拒否は**同期 throw ではなく
   * 拒否済み Promise** で返す — finish は従来から同期には投げない規約で、`await using` /
   * `Symbol.asyncDispose` 経路の後始末で例外が置き換わるのを避ける。
   * NOTE: in-flight を待つので、`enqueue()` の戻り Promise を await せずに `finish()` を
   * 呼んでも積んだぶんは必ず区間に入って完了する（リースの機構は {@link BatchInternals.enter}）。
   */
  finish(): Promise<void> {
    if (this.#settling) {
      // MUST: 決着は積まない（`#settled` を拒否で埋めると settle を await した後の正当な
      // finish() まで同じ拒否を返し続ける）。
      return Promise.reject(
        new BatchScopeError(
          "settle() の途中の batch は finish() できない（settle() を await してから閉じること）",
        ),
      );
    }
    if (this.#settled === undefined) {
      this.#finished = true;
      // in-flight が 1 本も無ければここで決着させる（enqueue を 1 本も出していない区間・
      // 全て await 済みの区間は従来どおり待ちが増えない）。
      if (this.#leases === 0) this.#drained.resolve();
      this.#release();
      this.#settled = this.#resolveFinish();
      // 決着を呼び出し側が受け取るまで未処理拒否にしない（中身は finish がそのまま返す）。
      void this.#settled.catch(() => undefined);
    }
    return this.#settled;
  }

  /**
   * 指定した常駐出力をまとめて読み戻し、そのmapをバッチの完了フェンスにする。
   * 既存finishの後にreadする二重待ちを避ける。空の指定はfinishと同じフェンスで閉じる。
   *
   * 構成は同期区間で固定し、全出力を決着まで使用予約する。データは借用であり、
   * 呼び出しから決着まではwriteしない。返したArrayBufferは呼び手の所有物。
   * 指定できるのは未終了・settle中でないbatchへ1回だけ。以後のfinishは同じ決着を待つ。
   * 出力の合計はdeviceのmaxBufferSize以下とし、別device・破棄済みは受け付けない。
   * DECIDED: docs/decisions/0054-resident-loop-and-fence.md#バッチ終端の一括読み戻し2026-09-12
   */
  finishAndRead(
    outputs: Readonly<Record<string, ResidentTensor>>,
  ): Promise<Readonly<Record<string, ArrayBuffer>>> {
    const retained: ResidentTensor[] = [];
    try {
      this.#assertReadbackAcceptable();
      assertDeviceUsable(this.#gpu, "batch の読み戻し");
      const entries = Object.entries(outputs);
      // MUST: 写しの後にもう一度検査する。`Object.entries` は利用者の getter を同期で走らせるので、
      // その中で同じ batch の `finishAndRead` / `finish` が先に決着させられる。ここで弾かないと
      // 外側が `#readback` を上書きし、内側が予約した常駐の使用予約が永久に返らない（以後
      // dispose できない）。
      this.#assertReadbackAcceptable();
      let totalBytes = 0;
      for (const [name, resident] of entries) {
        if (
          !(resident instanceof ResidentTensor) || resident[RUNTIME_INTERNAL].owner !== this.#gpu
        ) {
          throw new BatchScopeError(
            `finishAndRead '${name}': 同じ GpuContext の ResidentTensor が必要`,
          );
        }
        resident[RUNTIME_INTERNAL].retainUse();
        retained.push(resident);
        totalBytes += resident.byteLength;
      }
      this.#assertReadbackBytes(totalBytes + this.#readbackBytes(), "finishAndRead");
      this.#readback = entries;
    } catch (cause) {
      for (const resident of retained) resident[RUNTIME_INTERNAL].releaseUse();
      return Promise.reject(cause);
    }
    return this.finish().then(async () => (await this.#completion) ?? {});
  }

  /** `finishAndRead` の受け口（未終了・settle 中でない batch へ 1 回だけ）。 */
  #assertReadbackAcceptable(): void {
    if (this.#finished || this.#settling) {
      throw new BatchScopeError(
        "finishAndRead は未終了・settle中でない batch に1回だけ指定できる",
      );
    }
  }

  /** 決着時の staging に載る合計バイト数（常駐の指定 + 登録済みグラフ出力）。 */
  #readbackBytes(): number {
    const residents = (this.#readback ?? []).reduce(
      (sum, [, resident]) => sum + resident.byteLength,
      0,
    );
    return this.#graphReads.reduce(
      (sum, read) => read.sources.reduce((inner, source) => inner + source.size, sum),
      residents,
    );
  }

  /** 1 本の staging に収まるか（分割 staging や複数 map へ黙って切り替えない — ADR 0054）。 */
  #assertReadbackBytes(total: number, where: string): void {
    if (!Number.isSafeInteger(total) || total > this.#gpu.limits.maxBufferSize) {
      throw new BatchScopeError(
        `${where}: 合計 ${total} bytes が maxBufferSize ${this.#gpu.limits.maxBufferSize} を超える`,
      );
    }
  }

  /**
   * 常駐テンソルの指定とグラフ出力の登録を 1 本の staging へ連結コピーして読み戻す。
   * 常駐の対応表を返し、グラフ出力は登録側の `result` に置く（呼び手へ返すのは区間の成否が
   * 決まった後 — {@link BatchScope.#resolveFinish}）。
   */
  async #readOutputs(
    residents: readonly (readonly [string, ResidentTensor])[],
    graphReads: readonly GraphRead[],
    checkFailureScopes: () => Promise<void>,
    markFenced: () => void,
  ): Promise<Readonly<Record<string, ArrayBuffer>>> {
    const device = this.#gpu.device;
    const sources: readonly BatchReadSource[] = [
      ...residents.map(([, resident]) => ({
        buffer: resident[RUNTIME_INTERNAL].buffer,
        offset: 0,
        size: resident.byteLength,
      })),
      ...graphReads.flatMap((read) => read.sources),
    ];
    const staging = device.createBuffer({
      label: "batch-readback",
      size: sources.reduce((sum, source) => sum + source.size, 0),
      usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ,
    });
    try {
      const encoder = device.createCommandEncoder();
      let offset = 0;
      for (const source of sources) {
        encoder.copyBufferToBuffer(source.buffer, source.offset, staging, offset, source.size);
        offset += source.size;
      }
      device.queue.submit([encoder.finish()]);
      await checkFailureScopes();
      await this.#gpu[RUNTIME_INTERNAL].raceDeviceLost(
        staging.mapAsync(MAP_MODE.READ),
        "batch の読み戻し",
      );
      markFenced();
      const mapped = staging.getMappedRange();
      offset = 0;
      const take = (size: number): ArrayBuffer => {
        const copy = mapped.slice(offset, offset + size);
        offset += size;
        return copy;
      };
      const outputs = Object.fromEntries(
        residents.map(([name, resident]) => [name, take(resident.byteLength)]),
      );
      for (const read of graphReads) read.result = read.sources.map((source) => take(source.size));
      return outputs;
    } finally {
      staging.destroy();
    }
  }

  /**
   * `await using` 対応（Explicit Resource Management）— {@link BatchScope.finish} の別名。
   *
   * 区間は device 単位の errorScope 区間ロックを握り続けるので、`finish()` を通らずに抜けると
   * ロックが恒久保持され、以後その device の {@link Session.run} が例外も診断も出さずに待ち
   * 続ける。この面は「その取りこぼしを起こしにくい書き方（`await using`）を可能にする」もので、
   * 平文の `try`/`finally` を書き忘れた場合の無診断ハングそのものは塞がない。
   */
  [Symbol.asyncDispose](): Promise<void> {
    return this.finish();
  }

  /**
   * errorScope の pop 結果と記録したホスト側失敗を 1 本の決着にまとめる
   * （{@link BatchScope.finish} の中身 — 区間の**途中**決着である公開
   * {@link BatchScope.settle} とは別物なので、同名を避けてこの名前にしている）。
   */
  async #resolveFinish(): Promise<void> {
    let failure: { readonly cause: unknown } | undefined;
    try {
      try {
        await this.#completion;
      } catch (cause) {
        if (this.#failure !== undefined && cause instanceof Error && cause.cause === undefined) {
          cause.cause = this.#failure.cause;
        }
        failure = { cause };
      }
      failure ??= this.#failure;
      if (failure === undefined) {
        try {
          for (const finalizer of this.#finalizers) finalizer.complete();
        } catch (cause) {
          failure = { cause };
        }
      }
      const notifyFailure = (): void => {
        if (failure === undefined) return;
        for (const finalizer of this.#finalizers) {
          try {
            finalizer.fail(failure.cause);
          } catch { /* 元の失敗を保持し、残る使用予約も返す。 */ }
        }
      };
      if (failure !== undefined) notifyFailure();
      const hadFailure = failure !== undefined;
      for (const finalizer of this.#finalizers) {
        try {
          finalizer.release();
        } catch (cause) {
          failure ??= { cause };
        }
      }
      if (!hadFailure && failure !== undefined) notifyFailure();
      if (failure !== undefined) throw failure.cause;
    } finally {
      this.#finalizers.length = 0;
      for (const [, resident] of this.#readback ?? []) resident[RUNTIME_INTERNAL].releaseUse();
      this.#readback = undefined;
      // グラフ出力の読み戻しは区間の成否が決まった後にだけ返す（finishAndRead と同じ「合流の後」
      // — 成功データを持ったまま finish が失敗する形を作らない）。
      for (const read of this.#graphReads) {
        if (failure === undefined && read.result !== undefined) read.resolve(read.result);
        else {
          read.reject(
            failure?.cause ?? new BatchScopeError("batch がグラフ出力を読み戻さずに決着した"),
          );
        }
      }
      this.#graphReads.length = 0;
    }
  }
}
