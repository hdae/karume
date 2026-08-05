/**
 * 時間予算ベースの submit 分割。
 *
 * MUST: dispatch 列を 1 つの command buffer に積み切らない。長時間の submit は Windows TDR
 * （既定 2 秒）と Chromium の watchdog に掛かり、device lost として現れる。チャンク単位で
 * submit し、**実測から導いた 1 workgroup あたりの推定時間**でチャンクの切れ目を決める。
 *
 * ## 不変条件（ADR 0004 の実行モデル）
 *
 * 1. **成長は実測の裏付けにのみ基づく**。実測 0（タイマ分解能未満）は「速い」ではなく
 *    「情報が無い」— 推定を更新しない。裏付けが無い間の上限は
 *    {@link SubmitPolicy.initialChunkSize} で据え置く。
 * 2. **1 チャンクの推定時間が時間予算を超えない**。積む前に「この dispatch を足すと超える」を
 *    判定して先にチャンクを閉じる（超えてから縮めるのではない）。単独で予算を超える
 *    dispatch は分割できないので例外（{@link SubmitPolicy.minChunkSize} も下限として効く）。
 * 3. **フィードフォワード**。推定の単位は dispatch 数ではなく **workgroup 数**で、入力長 T が
 *    変われば dispatch あたりの workgroup 数も変わる。dispatch 数で推定すると、短い入力で
 *    測った値のまま長い入力を積んで一度は予算を大きく踏み越える（記号次元を持つグラフでは
 *    同じ Session で必ず起きる形）。
 *
 * ## 計測の帰属（なぜ flush の窓 1 本で推定するか）
 *
 * MUST: `onSubmittedWorkDone` を **submit ごとに呼ばない**。実装によってはこの呼び出しの
 * 同期部分が「そこまでに submit した全作業の完了」までホストをブロックし、CPU エンコードと
 * GPU 実行が完全に直列化する（壁時計が `max(GPU, ホスト)` ではなく **GPU + ホストの足し算**
 * になる。実測と素の WebGPU での単離実験は
 * `docs/research/2026-08-04-host-overhead-recon.md` §4.1）。呼ぶのは
 * {@link SubmitScheduler.flush} の 1 回だけにする — そこの待ちは flush-before-destroy の
 * ために元から要るもので、計測のための追加コストはゼロになる。
 *
 * 推定はその 1 回で閉じる**窓**から取る。窓 = 「窓で最初に submit した時刻 → flush の
 * `onSubmittedWorkDone` が解決した時刻」、仕事量 = 窓に積んだ全チャンクの workgroup 合計で、
 * 推定は **実測 ÷ 合計 workgroup 数**。チャンク単位の帰属は元々信用できない（完了通知は
 * 重なった submit だとほぼ同時に届き、先頭 1 本に全時間が乗って残りは 0 になる）ので、
 * 粒度を窓へ落として失う情報は無い。
 *
 * NOTE: この窓はホスト側のエンコード時間も内側に含むため、推定は GPU 実時間より**過大**に
 * 出る。過大 = 同じ予算で積める workgroup が減る = **チャンクが小さくなる**向きで、TDR に
 * 対しては安全側。細りすぎの歯止めは {@link SubmitPolicy.minChunkSize}（予算で切るのは
 * これ以上積んでから）と {@link SubmitPolicy.maxChunkSize}（硬い上限）の 2 つ。
 *
 * ## GPU 実時間の内訳（timestamp-query — ADR 0021）
 *
 * 上の壁時計計測はチャンク粒度で、しかも帰属が信用できない（適応制御の材料にしかならない）。
 * op 別の内訳は **pass 境界の timestamp** でしか取れないため、計測が有効な device
 * （{@link GpuContext.gpuTimingEnabled}）でだけ **1 dispatch = 1 pass** に開いて
 * `timestampWrites` を書く。同一 pass 内の連続 dispatch には元々 storage の可視性保証による
 * 暗黙の依存順序があるので、pass 分割で実行意味論は変わらない。
 *
 * MUST: 計測が無効なときはこの経路のコードを 1 行も通らない（1 チャンク = 1 pass のまま）。
 */

import type { GpuContext } from "./device.ts";

/** SubmitPolicy の値が実行不能（ハングを含む）な構成である。 */
export class SubmitPolicyError extends Error {
  override readonly name = "SubmitPolicyError";
}

export type SubmitPolicy = {
  /** 1 チャンクの GPU 実行がこの時間を超えないよう、推定に基づいて切れ目を決める。 */
  readonly timeBudgetMs: number;
  /**
   * **実測の裏付けが無い間**の 1 チャンクの dispatch 数上限（ブートストラップ）。
   * MUST: 実測 0 が続いてもここから成長しない（不変条件 1）。
   */
  readonly initialChunkSize: number;
  /**
   * 時間予算による切り上げがこの dispatch 数を下回らないようにする下限（submit 回数の歯止め）。
   * 単独で予算を超える dispatch があるときだけ、推定時間が予算を超えたチャンクが出る。
   */
  readonly minChunkSize: number;
  /** 実測の裏付けがあっても超えない dispatch 数の硬い上限。 */
  readonly maxChunkSize: number;
};

export const DEFAULT_SUBMIT_POLICY: SubmitPolicy = {
  timeBudgetMs: 100,
  initialChunkSize: 16,
  minChunkSize: 1,
  maxChunkSize: 1024,
};

/**
 * 推定時間に掛ける安全率。チャンクは grid-stride カーネルの workgroup 数で測った
 * **平均コスト**で見積もるため、チャンク内に重い dispatch が偏ったぶんの余裕をここで持つ
 * （2 倍の偏りまで予算内に収まる）。
 */
export const CHUNK_TIME_SAFETY = 0.5;

/**
 * 構築時の政策検査。
 *
 * MUST: チャンクサイズは 1 以上。0 だとチャンク切り出しが空配列を返し続け、`flush()` の
 * ループが永久に抜けない（誤値ではなくハングになるため、実行時ではなく構築時に落とす）。
 */
const assertPolicy = (policy: SubmitPolicy): void => {
  for (const key of ["initialChunkSize", "minChunkSize", "maxChunkSize"] as const) {
    const value = policy[key];
    if (!Number.isInteger(value) || value < 1) {
      throw new SubmitPolicyError(`SubmitPolicy.${key} は 1 以上の整数である必要がある: ${value}`);
    }
  }
  if (policy.minChunkSize > policy.maxChunkSize) {
    throw new SubmitPolicyError(
      `SubmitPolicy.minChunkSize (${policy.minChunkSize}) が maxChunkSize (${policy.maxChunkSize}) を超えている`,
    );
  }
  if (
    policy.initialChunkSize < policy.minChunkSize || policy.initialChunkSize > policy.maxChunkSize
  ) {
    throw new SubmitPolicyError(
      `SubmitPolicy.initialChunkSize (${policy.initialChunkSize}) が [${policy.minChunkSize}, ${policy.maxChunkSize}] の外にある`,
    );
  }
  if (!(policy.timeBudgetMs > 0) || !Number.isFinite(policy.timeBudgetMs)) {
    throw new SubmitPolicyError(
      `SubmitPolicy.timeBudgetMs は正の有限値である必要がある: ${policy.timeBudgetMs}`,
    );
  }
};

/**
 * `measuredMs` に保持する直近計測の件数。
 *
 * MUST: 有界であること。適応制御が使うのは直近に閉じた窓 1 本だけで、履歴は診断用にすぎない。
 * 無制限に伸ばすと長時間の推論で単調にメモリを食い、`stats` の複製代も flush 回数に比例
 * して増える（診断を読むほど遅くなる）。適応の傾向を見るには 32 件で足りる。
 */
export const MEASURED_HISTORY = 32;

export type SubmitStats = {
  readonly submitCount: number;
  readonly dispatchCount: number;
  /**
   * 閉じた**計測窓**の実測時間（ms）を**直近 {@link MEASURED_HISTORY} 件まで**。
   * 全期間の件数は {@link SubmitStats.measuredCount} を見る。
   *
   * NOTE: 1 件 = 1 窓 = {@link SubmitScheduler.flush} 1 回ぶん（多くの場合 run 1 回ぶん）で、
   * チャンク単位の値ではない。窓はホスト側のエンコード時間も含む（モジュール doc）。
   */
  readonly measuredMs: readonly number[];
  /** 全期間で閉じた計測窓の件数（`measuredMs` の履歴から溢れたぶんを含む）。 */
  readonly measuredCount: number;
  /**
   * {@link SubmitScheduler.discard} が submit せずに捨てた dispatch の累計件数。
   * 失敗した run の残骸なので通常は 0。0 でないことは「その回の run が途中で落ちた」の記録。
   */
  readonly discardedDispatches: number;
  /**
   * 現在の 1 チャンクの **dispatch 数上限**。実測の裏付けが無い間は
   * {@link SubmitPolicy.initialChunkSize}、裏付けがあれば {@link SubmitPolicy.maxChunkSize}。
   *
   * NOTE: 適応の主役ではなくなった（M1-P2 で意味変更）。実際の切れ目は
   * {@link SubmitStats.msPerWorkgroup} から決まる時間予算のほうが先に効くことが多く、この値は
   * 「これ以上は積まない」硬い上限を表す。
   */
  readonly currentChunkSize: number;
  /**
   * 直近に閉じた窓から導いた 1 workgroup あたりの推定時間（ms）。**裏付けがまだ無ければ
   * undefined**（実測が 0 の間はここが埋まらない = 成長しない、が不変条件 1 の観測点）。
   * 1 チャンクの workgroup 予算は `timeBudgetMs * CHUNK_TIME_SAFETY / msPerWorkgroup`。
   */
  readonly msPerWorkgroup?: number;
};

/** パイプラインキー 1 本ぶんの GPU 実時間の内訳（ADR 0021）。 */
export type GpuTimingEntry = {
  /** 集計の単位は**パイプラインキー**（op 種 + 変種を既に一意識別する語彙）。 */
  readonly key: string;
  /** GPU 実時間の合計（ナノ秒）。pass の begin/end の差分の総和。 */
  readonly ns: number;
  readonly dispatchCount: number;
  readonly workgroupCount: number;
};

/**
 * 直近 run の GPU 実時間内訳。計測が**無効**な device では丸ごと undefined になる
 * （空表を返すと「計測したが全部 0」と区別できない）。
 * NOTE: 有効なら run 前は空表（1 度も走っていないことは `lastRun === undefined` が表す）。
 */
export type GpuTimingStats = {
  /** `ns` の降順（同値はキーの辞書順）。 */
  readonly entries: readonly GpuTimingEntry[];
  readonly totalNs: number;
  /** 内訳に載った dispatch の総数。{@link SubmitStats.dispatchCount} との相互検算に使う。 */
  readonly dispatchCount: number;
  /**
   * begin > end（ドライバの timestamp が非単調）で **0 に丸めた**サンプル数。
   * MUST: 黙って捨てない — 0 でないなら内訳の読みそのものが疑わしい、と気づける形にする。
   */
  readonly clampedNegativeSamples: number;
};

type PendingDispatch = {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroup: GPUBindGroup;
  readonly workgroups: readonly [number, number, number];
  /** この dispatch の仕事量（workgroup 数）。時間予算の比例配分はこの単位で行う。 */
  readonly work: number;
  /** 内訳の帰属先（{@link PipelineCache} のキーそのもの）。 */
  readonly key: string;
};

/** timestamp 1 件のバイト数（u64 ns）。 */
const TIMESTAMP_BYTES = 8;

/**
 * 1 チャンクで作る querySet の容量上限。
 *
 * WebGPU の `maxQueryCount` は 4,096 で、必要数は「チャンクの dispatch 数 × 2」。既定政策の
 * `maxChunkSize` 1,024 なら 2,048 で収まるが、政策で上限を上げると**静かに** validation で
 * 落ちる（= その submit の全 pass が実行されない）ため、エンコード前に明示的に落とす。
 */
export const MAX_TIMESTAMP_QUERIES = 2048;

/** submit 済み・未回収の timestamp 資源（1 チャンク 1 件）。 */
type PendingTiming = {
  readonly querySet: GPUQuerySet;
  readonly resolveBuffer: GPUBuffer;
  readonly readBuffer: GPUBuffer;
  /** query 対（begin, end）の並びと 1:1 で対応する帰属先。 */
  readonly entries: readonly { readonly key: string; readonly work: number }[];
};

/** キー別の累計（{@link GpuTimingEntry} の可変版）。 */
type TimingTotal = { ns: number; dispatchCount: number; workgroupCount: number };

/**
 * timestamp 資源の解放。回収の成否によらず**必ず** 1 回通す（診断のために VRAM を積み残さない）。
 * 対象は既に submit 済みのコマンドだけが参照するので、実行中の破棄も WebGPU 的に安全
 * （実解放は実装が完了まで遅延する）。
 */
const destroyTiming = (timing: PendingTiming): void => {
  timing.readBuffer.destroy();
  timing.resolveBuffer.destroy();
  timing.querySet.destroy();
};

export class SubmitScheduler {
  readonly #gpu: GpuContext;
  readonly #device: GPUDevice;
  readonly #policy: SubmitPolicy;
  /** 計測用の時刻源（ms 単位・単調）。既定は `performance.now()`。 */
  readonly #now: () => number;
  #pending: PendingDispatch[] = [];
  /** 未 submit のエンコードの仕事量合計（`#pending` の `work` の和）。 */
  #pendingWork = 0;
  #submitCount = 0;
  #dispatchCount = 0;
  /** 直近 {@link MEASURED_HISTORY} 件のリング（古いものから捨てる）。 */
  readonly #measuredMs: number[] = [];
  #measuredCount = 0;
  #discardedDispatches = 0;
  /**
   * 開いている計測窓の起点（窓で最初に submit した時刻）。undefined = 窓が開いていない
   * （前回の flush 以降 1 度も submit していない）。
   */
  #windowStartedAt: number | undefined;
  /** 開いている窓に積んだ workgroup の合計。 */
  #windowWork = 0;
  /** 実測に裏付けられた 1 workgroup あたりの推定時間。undefined = 裏付けがまだ無い。 */
  #msPerWorkgroup: number | undefined;
  /** GPU 側時間計測が有効か（device の feature で決まる — ADR 0021）。 */
  readonly #timingEnabled: boolean;
  /** submit 済み・未回収の timestamp 資源。 */
  #pendingTimings: PendingTiming[] = [];
  /** パイプラインキー別の累計（寿命は直近 run — {@link SubmitScheduler.resetTiming}）。 */
  readonly #timingTotals = new Map<string, TimingTotal>();
  #clampedNegativeSamples = 0;

  /**
   * @param now 時刻源（ms）。差分計測の規則をテストで固定できるよう注入可能にしてある。
   */
  constructor(
    gpu: GpuContext,
    policy: SubmitPolicy = DEFAULT_SUBMIT_POLICY,
    now: () => number = () => performance.now(),
  ) {
    assertPolicy(policy);
    this.#gpu = gpu;
    this.#device = gpu.device;
    this.#policy = policy;
    this.#now = now;
    this.#timingEnabled = gpu.gpuTimingEnabled;
  }

  /**
   * dispatch を 1 件積む。時間予算か dispatch 数上限に達したらその場で submit する。
   *
   * MUST: 予算判定は**積む前**に行う（不変条件 2）。積んでから「超えたので次から縮める」に
   * すると、超過したチャンクは既に submit 済みで、予算超過を毎回 1 度は踏む形が定常になる。
   *
   * @param key この dispatch のパイプラインキー（GPU 時間内訳の帰属先 — ADR 0021）。
   *   MUST: `pipeline` を引いたときのキーそのものを渡す。別に組み立てると内訳だけが
   *   静かに別の op へ寄る。
   */
  dispatch(
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    workgroups: readonly [number, number, number],
    key: string,
  ): void {
    const work = workgroups[0] * workgroups[1] * workgroups[2];
    if (this.#exceedsTimeBudget(this.#pendingWork + work)) {
      this.#submitChunk();
    }
    this.#pending.push({ pipeline, bindGroup, workgroups, work, key });
    this.#pendingWork += work;
    this.#dispatchCount += 1;
    if (this.#pending.length >= this.#chunkSizeLimit()) {
      this.#submitChunk();
    }
  }

  /**
   * 未 submit のエンコードを submit せずに捨てる。
   *
   * MUST: 失敗した run の後始末はこの経路を通す。残骸を実行する理由が無いうえ、失敗経路の
   * submit は errorScope の外（呼び出し側が既にスコープを畳んだ後）で起きるため、その
   * validation エラーは device 単位 LIFO の先頭にある**他人のスコープ**に帰属する。
   * 捨てることで {@link SubmitScheduler.flush} と併せて「discard-or-flush before destroy」
   * — 破棄済みバッファを参照するエンコードが submit に残らない — も同時に満たす。
   *
   * NOTE: 開いている計測窓には触らない。捨てたぶんは仕事量に入っていない（submit 済みの
   * チャンクだけが積んである）ので、後始末の flush が閉じる窓は失敗の巻き戻し時間ぶん
   * **過大**に出る — チャンクが縮む向き = 安全側で、次の窓が正しい値へ戻す。
   */
  discard(): void {
    this.#discardedDispatches += this.#pending.length;
    this.#pending.length = 0;
    this.#pendingWork = 0;
    // MUST: 回収待ちの timestamp 資源も同じ経路で捨てる（discard-or-flush before destroy の
    // query 版）。残すと、後続の後始末 flush が「失敗した run の残骸」を map しに行く。
    for (const timing of this.#pendingTimings) destroyTiming(timing);
    this.#pendingTimings.length = 0;
  }

  /**
   * 未 submit のエンコードを全て submit し、GPU 側の完了まで待つ。
   *
   * MUST: バッファを destroy する前に必ず flush する（成功経路）。未 submit のエンコードが
   * 破棄済みのバッファを参照すると submit がコマンドバッファ丸ごと失敗し、同じスケジューラに
   * 相乗りしている無関係な dispatch まで実行されないまま、誤った値が静かに残る。失敗経路で
   * 残骸を出し切らずに済ませるのが {@link SubmitScheduler.discard}。
   *
   * MUST: `onSubmittedWorkDone` を呼ぶのは**ここだけ**（モジュール doc の計測の帰属）。
   * 適応制御の観測点も同じ待ちに相乗りする — 窓を閉じるのは待ちの直後で、timestamp の回収
   * （`mapAsync`）は窓の外に置く（診断の代を推定へ混ぜない）。
   */
  async flush(): Promise<void> {
    // 残りは常に予算と上限の内側（切れ目は dispatch 時に決まっている）なので 1 回で出し切る。
    this.#submitChunk();
    // MUST: device 消失時 onSubmittedWorkDone は解決しない。待ち続けるとハングになるため
    // 消失を競わせて例外に変換する（購読は決着時に必ず解除される — GpuContext 側の責務）。
    await this.#gpu.raceDeviceLost(this.#device.queue.onSubmittedWorkDone(), "flush");
    this.#closeMeasurementWindow();
    await this.#collectTimings();
  }

  /**
   * 直近 run の GPU 時間内訳。計測が無効な device では undefined。
   *
   * NOTE: 表を組むのは読むたび（集計の実体はキー別の累計 Map だけ）。合計欄も derive する —
   * 独立に更新する非正規化フィールドを持つと、丸め件数だけ増えて合計が動かない類の
   * 食い違いが起きる。
   */
  get timing(): GpuTimingStats | undefined {
    if (!this.#timingEnabled) return undefined;
    const entries: GpuTimingEntry[] = [...this.#timingTotals]
      .map(([key, total]) => ({ key, ...total }))
      // 降順で読む表。同値のときはキー順で並びを決定的にする（ロケール非依存の比較）。
      .sort((a, b) => b.ns - a.ns || (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
    return {
      entries,
      totalNs: entries.reduce((total, entry) => total + entry.ns, 0),
      dispatchCount: entries.reduce((total, entry) => total + entry.dispatchCount, 0),
      clampedNegativeSamples: this.#clampedNegativeSamples,
    };
  }

  /**
   * 内訳の集計を空にする。**寿命は直近 run**（ADR 0021）なので run の開始で呼ぶ。
   * 未回収の timestamp 資源には触らない（前の run が flush / discard で必ず出し切っている）。
   */
  resetTiming(): void {
    this.#timingTotals.clear();
    this.#clampedNegativeSamples = 0;
  }

  get stats(): SubmitStats {
    return {
      submitCount: this.#submitCount,
      dispatchCount: this.#dispatchCount,
      measuredMs: [...this.#measuredMs],
      measuredCount: this.#measuredCount,
      discardedDispatches: this.#discardedDispatches,
      currentChunkSize: this.#chunkSizeLimit(),
      msPerWorkgroup: this.#msPerWorkgroup,
    };
  }

  /**
   * 現在の dispatch 数上限。実測の裏付けが無い間はブートストラップ値のまま据え置く
   * （不変条件 1 — 実測 0 は成長の根拠にならない）。
   */
  #chunkSizeLimit(): number {
    return this.#msPerWorkgroup === undefined
      ? this.#policy.initialChunkSize
      : this.#policy.maxChunkSize;
  }

  /**
   * 仕事量 `work` のチャンクが時間予算を超える見込みか。裏付けが無ければ判定しない
   * （false = 予算では切らない — 上限は dispatch 数側が持つ）。
   */
  #exceedsTimeBudget(work: number): boolean {
    const perWorkgroup = this.#msPerWorkgroup;
    if (perWorkgroup === undefined) return false;
    // 予算で切るのは minChunkSize 以上を積んでからにする（submit 回数の歯止め）。単独で
    // 予算を超える dispatch はここで分割できない — チャンク 1 本ぶんとして出すしかない。
    if (this.#pending.length < this.#policy.minChunkSize) return false;
    return work * perWorkgroup > this.#policy.timeBudgetMs * CHUNK_TIME_SAFETY;
  }

  #submitChunk(): void {
    const chunk = this.#pending;
    if (chunk.length === 0) {
      return;
    }
    this.#pending = [];
    const chunkWork = this.#pendingWork;
    this.#pendingWork = 0;
    const encoder = this.#device.createCommandEncoder();
    if (this.#timingEnabled) {
      this.#encodeTimedChunk(encoder, chunk);
    } else {
      const pass = encoder.beginComputePass();
      for (const item of chunk) {
        pass.setPipeline(item.pipeline);
        pass.setBindGroup(0, item.bindGroup);
        pass.dispatchWorkgroups(...item.workgroups);
      }
      pass.end();
    }
    const submittedAt = this.#now();
    this.#device.queue.submit([encoder.finish()]);
    this.#submitCount += 1;
    // MUST NOT: ここで onSubmittedWorkDone を呼ばない（モジュール doc — ホストと GPU の
    // 直列化）。計測の起点だけを残し、閉じるのは flush 側。
    this.#windowStartedAt ??= submittedAt;
    this.#windowWork += chunkWork;
  }

  /**
   * 計測モードのエンコード（**1 dispatch = 1 pass**）。pass の begin/end を chunk 1 本ぶんの
   * querySet に書き、`resolveQuerySet` → COPY_SRC バッファ → MAP_READ バッファまでを同じ
   * command buffer に積む（回収は完了後の {@link SubmitScheduler.flush} で行う）。
   *
   * MUST: `writeTimestamp` は使わない（標準の WebGPU に無い — pass 境界だけが移植可能な計測点）。
   */
  #encodeTimedChunk(encoder: GPUCommandEncoder, chunk: readonly PendingDispatch[]): void {
    const count = chunk.length * 2;
    if (count > MAX_TIMESTAMP_QUERIES) {
      throw new SubmitPolicyError(
        `GPU 時間計測の querySet 容量を超えた（dispatch ${chunk.length} 件 = query ${count} 件 > ` +
          `${MAX_TIMESTAMP_QUERIES}）。SubmitPolicy.maxChunkSize を ${
            MAX_TIMESTAMP_QUERIES / 2
          } 以下にするか、gpuTiming: false で計測を切ること`,
      );
    }
    const byteLength = count * TIMESTAMP_BYTES;
    const querySet = this.#device.createQuerySet({ type: "timestamp", count });
    const resolveBuffer = this.#device.createBuffer({
      size: byteLength,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const readBuffer = this.#device.createBuffer({
      size: byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    for (const [index, item] of chunk.entries()) {
      const pass = encoder.beginComputePass({
        timestampWrites: {
          querySet,
          beginningOfPassWriteIndex: index * 2,
          endOfPassWriteIndex: index * 2 + 1,
        },
      });
      pass.setPipeline(item.pipeline);
      pass.setBindGroup(0, item.bindGroup);
      pass.dispatchWorkgroups(...item.workgroups);
      pass.end();
    }
    encoder.resolveQuerySet(querySet, 0, count, resolveBuffer, 0);
    encoder.copyBufferToBuffer(resolveBuffer, 0, readBuffer, 0, byteLength);
    this.#pendingTimings.push({
      querySet,
      resolveBuffer,
      readBuffer,
      entries: chunk.map((item) => ({ key: item.key, work: item.work })),
    });
  }

  /**
   * 完了済みチャンクの timestamp を回収してキー別に集計する。
   *
   * MUST: 呼び出しは `onSubmittedWorkDone` の**後**（該当チャンクの GPU 実行完了後）に限る。
   * MUST: `mapAsync` の待ちを完了コールバック（fire-and-forget）側に載せない。待ちが run の
   * ロック区間の外へ抜け、errorScope 直列化の外で GPU 操作を出すことになる（device 単位 LIFO
   * の誤帰属）。flush の内側で待てば、run の GPU 操作は全てロック区間内に収まる。
   */
  async #collectTimings(): Promise<void> {
    if (this.#pendingTimings.length === 0) return;
    const pending = this.#pendingTimings;
    this.#pendingTimings = [];
    try {
      // MUST: device 消失時 mapAsync は解決しない（readback と同じ理由で競わせる）。
      await this.#gpu.raceDeviceLost(
        Promise.all(pending.map((item) => item.readBuffer.mapAsync(GPUMapMode.READ))),
        "timestamp の回収",
      );
      for (const item of pending) {
        // 同期区間で読み切るのでコピーは要らない（unmap まで view は有効）。
        this.#accumulate(item.entries, new BigUint64Array(item.readBuffer.getMappedRange()));
        item.readBuffer.unmap();
      }
    } finally {
      // MUST: 成否によらず解放する。失敗しても資源は残さない（destroy は暗黙 unmap を含む）。
      for (const item of pending) destroyTiming(item);
    }
  }

  /** query 対の差分をキー別累計へ足す（並びは {@link PendingTiming.entries} と 1:1）。 */
  #accumulate(
    entries: readonly { readonly key: string; readonly work: number }[],
    stamps: BigUint64Array,
  ): void {
    for (const [index, entry] of entries.entries()) {
      const delta = stamps[index * 2 + 1] - stamps[index * 2];
      // MUST: 負値は 0 に丸めたうえで**件数を残す**（黙って捨てない — ADR 0021）。
      if (delta < 0n) this.#clampedNegativeSamples += 1;
      const total = this.#timingTotals.get(entry.key) ??
        { ns: 0, dispatchCount: 0, workgroupCount: 0 };
      total.ns += delta < 0n ? 0 : Number(delta);
      total.dispatchCount += 1;
      total.workgroupCount += entry.work;
      this.#timingTotals.set(entry.key, total);
    }
  }

  /**
   * 計測窓を閉じて推定を更新する（呼び出しは flush の `onSubmittedWorkDone` 待ちの直後）。
   *
   * MUST: 実測が 0（窓全体がタイマ分解能未満）の窓では**更新しない**。0 は「速い」ではなく
   * 「情報が無い」で、これを成長の根拠にすると分解能の粗い環境で無条件にチャンクが膨らみ、
   * 1 submit 全積み（= TDR 域）へ行き着く。仕事量 0 の窓も同じ理由で情報が無い
   * （0 除算で Infinity を掴むと、今度は永久に minChunkSize へ張り付く）。
   */
  #closeMeasurementWindow(): void {
    const startedAt = this.#windowStartedAt;
    // 窓が開いていない = 前回の flush 以降 1 度も submit していない。記録するものが無い
    // （空の flush を「実測 0 の窓」として数えると、診断の窓が中身の無い件で埋まる）。
    if (startedAt === undefined) return;
    const measured = this.#now() - startedAt;
    const work = this.#windowWork;
    this.#windowStartedAt = undefined;
    this.#windowWork = 0;
    this.#measuredMs.push(measured);
    if (this.#measuredMs.length > MEASURED_HISTORY) {
      this.#measuredMs.shift();
    }
    this.#measuredCount += 1;
    if (measured > 0 && work > 0) {
      this.#msPerWorkgroup = measured / work;
    }
  }
}
