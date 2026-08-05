/**
 * 実行単位（1 run）のバッファアリーナ。確保した全バッファの所有と、最終消費を過ぎた出力
 * ストレージのサイズクラス別プール再利用を行う。素朴な全実体化は中間値のピークが実行不能な
 * 大きさになるため、寿命解放が前提。
 *
 * 再利用の安全性（WebGPU 仕様に依拠）:
 * - compute pass の usage scope は dispatch 単位で、同一キューでは先行 dispatch の書き込みが
 *   後続 dispatch から可視・実行順序も submit 順。解放は「読む dispatch のエンコード後」に
 *   のみ起き、再利用先の書き込みは必ずそれより後のエンコードなので read→write の追い越しは
 *   起きない。
 * - MUST: `queue.writeBuffer` で書くバッファ（uniform / 入力 / 重み）はプールに入れない。
 *   writeBuffer はキュー順で未 submit の先行エンコードを追い越す。プール対象は dispatch が
 *   書く出力ストレージのみ（{@link RunArena.allocStorage}）。
 * - MUST: プール再利用バッファはゼロ初期化されていない（ゼロ保証は新規 createBuffer のみ）。
 *   これに依存しないための不変条件が **full-write**（ADR 0014）: **全ノードは出力バッファの
 *   全バイトを書く**。1 ノードが複数 dispatch を出す形（cat）でも、ノード単位で全域が覆われる
 *   ことを呼び出し側が担保する。
 *   MUST: 「出力を部分的にしか書かない op のためにプールを迂回して新品を配る」経路は**足さない**
 *   （ADR 0014 で却下）。不変条件がカーネル実装の知識に依存し、アリーナ側が op 種別を知る必要が
 *   生じるため。ゼロ埋めが要る op（pad）は、範囲外にも 0 を**書く**カーネルとして実装する。
 *   固定はフォールト注入で行う（tests/gpu_full_write_test.ts — 再利用バッファに毒値を書いてから
 *   cat / pad を実行し、出力に毒値が 1 語も残らないことを実 GPU で確かめる）。
 *
 * エイリアス（ADR 0011 の reshape）:
 * - 1 つの実バッファが**複数の論理値**の実体になりうる。アリーナはバッファ単位でしか物を
 *   見ないので、参照計数は論理値ごとの {@link RunArena.retain} が同じバッファに積み上がる。
 *   実バッファの参照数 = 各論理値の「定義ぶんの 1 + 消費回数」の総和。
 * - **確保していないバッファは所有しない**。別名化は `#owned` を増やさないので、
 *   discard-or-flush before destroy と実際の `destroy()` は実バッファごとにちょうど 1 回に
 *   なる（論理値の本数に依らない）。この不変条件は「別名は alloc 経路を通らない」ことだけで
 *   成立しており、別名の数を数える状態を持たない。
 */

/** アリーナの不変条件違反（参照計数の破れ・破棄後利用など）。 */
export class ArenaError extends Error {
  override readonly name = "ArenaError";
}

/** 出力ストレージの usage。dispatch が書き、必要なら readback のため COPY_SRC を持つ。 */
const STORAGE_USAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

/**
 * サイズクラスは「4 バイト整列した実バイト数そのもの」とする。
 *
 * MUST: 要求より大きいバッファを配らない。runtime-sized array を束縛したときの
 * `arrayLength()` は束縛範囲のバイト数から決まるため、切り上げた大きさを配ると要素数が
 * 静かに変わり、誤った値が例外なしで出る。
 */
const toSizeClass = (bytes: number): number => {
  if (!Number.isInteger(bytes) || bytes < 0) {
    throw new ArenaError(`確保サイズは 0 以上の整数である必要がある: ${bytes}`);
  }
  return Math.max(4, Math.ceil(bytes / 4) * 4);
};

export type ArenaStats = {
  readonly allocCount: number;
  readonly reuseCount: number;
  /** 実際に createBuffer したバイト数の合計（再利用ぶんは含まない）。 */
  readonly allocatedBytes: number;
  /**
   * プール管理下（出力ストレージ）で生存中のバイト数。ホスト書き込み / readback staging の
   * ぶんは含まない（どちらもプール外）。
   */
  readonly transientBytes: number;
  readonly peakTransientBytes: number;
};

export type RetainOptions = {
  /** グラフ出力。参照が尽きてもプールに返さず、readback 可能なまま保つ。 */
  readonly pinned?: boolean;
};

/** 未 submit のエンコードを submit し、GPU 完了まで待つ手段（SubmitScheduler.flush 等）。 */
export type ArenaFlush = () => Promise<void>;

export class RunArena {
  readonly #device: GPUDevice;
  readonly #flush: ArenaFlush;
  readonly #owned: GPUBuffer[] = [];
  readonly #pool = new Map<number, GPUBuffer[]>();
  readonly #poolable = new Set<GPUBuffer>();
  readonly #refs = new Map<GPUBuffer, number>();
  readonly #pinned = new Set<GPUBuffer>();
  #transientBytes = 0;
  #peakTransientBytes = 0;
  #allocCount = 0;
  #reuseCount = 0;
  #allocatedBytes = 0;
  #destroying: Promise<void> | undefined;

  constructor(device: GPUDevice, flush: ArenaFlush) {
    this.#device = device;
    this.#flush = flush;
  }

  /**
   * dispatch が書く出力ストレージを確保する。プール適格（最終消費後に再利用される）。
   * MUST: 出力 alloc は当該 dispatch のエンコードより前に行う。
   */
  allocStorage(bytes: number): GPUBuffer {
    this.#assertUsable();
    const sizeClass = toSizeClass(bytes);
    const reused = this.#pool.get(sizeClass)?.pop();
    if (reused !== undefined) {
      this.#reuseCount += 1;
      this.#addTransient(sizeClass);
      return reused;
    }
    const buffer = this.#device.createBuffer({ size: sizeClass, usage: STORAGE_USAGE });
    this.#owned.push(buffer);
    this.#poolable.add(buffer);
    this.#allocCount += 1;
    this.#allocatedBytes += sizeClass;
    this.#addTransient(sizeClass);
    return buffer;
  }

  /**
   * ホストが `queue.writeBuffer` で書くバッファ（uniform / 入力 / 重み）を確保する。
   * MUST: この経路のバッファはプール対象外（冒頭の writeBuffer 追い越しの理由）。
   */
  allocHostWritten(bytes: number, usage: number): GPUBuffer {
    this.#assertUsable();
    if ((usage & GPUBufferUsage.COPY_DST) === 0) {
      throw new ArenaError("writeBuffer で書くバッファには COPY_DST が必要");
    }
    const buffer = this.#device.createBuffer({ size: toSizeClass(bytes), usage });
    this.#owned.push(buffer);
    this.#allocCount += 1;
    this.#allocatedBytes += buffer.size;
    return buffer;
  }

  /**
   * ホストが `mapAsync` で読む staging（`COPY_DST | MAP_READ`）を確保する。
   * MUST: この経路のバッファもプール対象外 — MAP_READ は STORAGE と併用できず、マップ状態が
   * 寿命に絡むため使い回せない（1 回の readback で使い捨て）。それでもアリーナが所有するのは
   * 「確保と破棄を 1 箇所へ」（ADR 0004）— 破棄が flush-before-destroy に自動で乗る。
   */
  allocHostRead(bytes: number): GPUBuffer {
    this.#assertUsable();
    const buffer = this.#device.createBuffer({
      size: toSizeClass(bytes),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.#owned.push(buffer);
    this.#allocCount += 1;
    this.#allocatedBytes += buffer.size;
    return buffer;
  }

  /**
   * 値の定義を登録する。参照数は「**定義ぶんの 1** + `uses`」で、`uses` はその値の将来の
   * 消費回数 ＝ `node.ins` の**厳密な延べ計数**でなければならない（同じ値を 2 回入力に取る
   * ノードは 2 と数える）。過少なら解放が早すぎて生きている値が上書きされ、過多なら解放
   * されずピークが落ちない。
   *
   * MUST: 定義ぶんの 1 は**定義したノードの境界で** {@link release} して返す。これがあるので
   * 消費者が 1 つも無い値（到達不能な中間出力）も他と同じ経路でプールへ戻る — `uses` が 0 の
   * 値に release が 1 度も来ないと、再利用から外れたまま居座って peak が過大に出る。
   *
   * 同じバッファへの複数回 retain は**加算**する（別名 — 冒頭のエイリアス節）。上書きにすると
   * 別名を作った時点で入力側の残り消費が消え、生きている値がプールへ戻る。
   */
  retain(buffer: GPUBuffer, uses: number, options: RetainOptions = {}): void {
    if (!Number.isInteger(uses) || uses < 0) {
      throw new ArenaError(`retain の uses は 0 以上の整数である必要がある: ${uses}`);
    }
    if (!this.#poolable.has(buffer)) {
      return;
    }
    if (options.pinned === true) {
      this.#pinned.add(buffer);
    }
    this.#refs.set(buffer, (this.#refs.get(buffer) ?? 0) + uses + 1);
  }

  /**
   * 参照 1 つ分の解放（消費 1 回ぶん、または {@link retain} が積んだ定義ぶんの 1）。
   * 参照が尽きたらプールへ返す（ピン留めは除外）。
   * MUST: 呼ぶのはノード境界（当該ノードの全 dispatch をエンコードし終えた後）のみ。
   * プール対象外のバッファに対しては何もしない（消費側が種別を意識せず呼べるようにする）。
   */
  release(buffer: GPUBuffer): void {
    if (!this.#poolable.has(buffer)) {
      return;
    }
    const left = (this.#refs.get(buffer) ?? 0) - 1;
    if (left < 0) {
      throw new ArenaError("参照カウントが負（解放過多 — 消費計数の誤り）");
    }
    this.#refs.set(buffer, left);
    if (left === 0 && !this.#pinned.has(buffer)) {
      const bucket = this.#pool.get(buffer.size);
      if (bucket === undefined) {
        this.#pool.set(buffer.size, [buffer]);
      } else {
        bucket.push(buffer);
      }
      this.#transientBytes -= buffer.size;
    }
  }

  /**
   * readback を許すか。プール対象外（ホスト書き込み側）またはピン留め（グラフ出力）のみ。
   * 中間値は再利用で内容が入れ替わるため readback を拒否する。
   */
  isReadable(buffer: GPUBuffer): boolean {
    return !this.#poolable.has(buffer) || this.#pinned.has(buffer);
  }

  /** 非ピン留めのプール対象に未解放の参照が残っていないことを検査する（計数漏れの検出）。 */
  assertDrained(): void {
    for (const [buffer, refs] of this.#refs) {
      if (refs > 0 && !this.#pinned.has(buffer)) {
        throw new ArenaError(`未解放の参照が残存（refs=${refs}, size=${buffer.size}）`);
      }
    }
  }

  get stats(): ArenaStats {
    return {
      allocCount: this.#allocCount,
      reuseCount: this.#reuseCount,
      allocatedBytes: this.#allocatedBytes,
      transientBytes: this.#transientBytes,
      peakTransientBytes: this.#peakTransientBytes,
    };
  }

  /**
   * 所有する全バッファを破棄する。
   *
   * MUST: 未 submit のエンコードを片付けてから破棄する（discard-or-flush before destroy）。
   * ここが待つのは flush の完了だけなので、**捨てる側（失敗経路の
   * `SubmitScheduler.discard`）は呼び出し側の責務**。破棄済みバッファを参照するエンコードを
   * submit するとコマンドバッファ丸ごと失敗し、同じスケジューラに相乗りしている無関係な
   * dispatch まで実行されないまま、誤った値が静かに残る。
   *
   * NOTE: VRAM 自体が返るのは `device.destroy()` のみ。ここでの破棄は所有と再利用の終了。
   */
  destroy(): Promise<void> {
    // MUST: 2 度目の destroy も同じ完了を待つ。先行する flush の完了前に返すと、呼び出し側が
    // 「破棄済み」と見なして device.destroy() まで進み、flush-before-destroy が崩れる。
    this.#destroying ??= this.#destroyOnce();
    return this.#destroying;
  }

  async #destroyOnce(): Promise<void> {
    // MUST: flush が失敗（主因は device 消失）してもバッファ破棄と状態クリアは必ず行う。
    // ここを飛ばすと 1 本も破棄されないまま `#destroying` が rejected で居座り、再試行も
    // できなくなる。flush の失敗自体は握り潰さず、後始末を終えてから伝播させる。
    try {
      await this.#flush();
    } finally {
      for (const buffer of this.#owned) {
        buffer.destroy();
      }
      this.#owned.length = 0;
      this.#pool.clear();
      this.#poolable.clear();
      this.#refs.clear();
      this.#pinned.clear();
      this.#transientBytes = 0;
    }
  }

  #assertUsable(): void {
    // 破棄の flush 待ち中に確保されたバッファは誰にも破棄されないため、開始時点で閉じる。
    if (this.#destroying !== undefined) {
      throw new ArenaError("破棄済みの RunArena は使えない（run ごとに作り直す）");
    }
  }

  #addTransient(bytes: number): void {
    this.#transientBytes += bytes;
    this.#peakTransientBytes = Math.max(this.#peakTransientBytes, this.#transientBytes);
  }
}
