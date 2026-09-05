/**
 * 実行単位（1 run）のバッファアリーナ。確保した全バッファの**所有**と flush-before-destroy を担う。
 *
 * 中間バッファ（ノード出力・ノード内一時）の配置はここでは決めない — 計画（src/runtime/transient-plan.ts
 * の静的 liveness パッキング・ADR 0093）が生存区間から領域バッファの offset を決め、アリーナは
 * その**領域**（{@link RunArena.allocRegion}）を run 寿命で所有するだけ。旧「サイズクラス別プール +
 * 最終消費者解放」（ADR 0004）は ADR 0093 決定 8 で退役した。
 *
 * 配り直しの安全性（WebGPU 仕様に依拠）:
 * - compute pass の usage scope は dispatch 単位で、同一キューでは先行 dispatch の書き込みが
 *   後続 dispatch から可視・実行順序も submit 順。計画は「読む dispatch のエンコード後」に生存を
 *   終えた区間のバイトだけを後続へ配るので、read→write の追い越しは起きない。
 * - MUST: `queue.writeBuffer` で書くバッファ（uniform / 入力 / 重み）は領域に入れない。
 *   writeBuffer はキュー順で未 submit の先行エンコードを追い越す。領域に載るのは dispatch が
 *   書く出力ストレージと一時のみ。
 * - MUST: 配り直されたバイトはゼロ初期化されていない（ゼロ保証は新規 createBuffer のみ）。
 *   これに依存しないための不変条件が **full-write**（ADR 0014）: **全ノードは自分の束縛範囲の
 *   全バイトを書く**。1 ノードが複数 dispatch を出す形（cat）でも、ノード単位で全域が覆われる
 *   ことを呼び出し側が担保する。
 *   MUST: 「出力を部分的にしか書かない op のために領域を迂回して新品を配る」経路は**足さない**
 *   （ADR 0014 で却下）。不変条件がカーネル実装の知識に依存し、アリーナ側が op 種別を知る必要が
 *   生じるため。ゼロ埋めが要る op（pad）は、範囲外にも 0 を**書く**カーネルとして実装する。
 *   固定はフォールト注入で行う（tests/gpu_full_write_test.ts — 配り直されたバイトに毒値を書いてから
 *   cat / pad を実行し、出力に毒値が 1 語も残らないことを実 GPU で確かめる）。
 */

import { BUFFER_USAGE } from "./webgpu-constants.ts";

/** アリーナの使い方の誤り（破棄後の確保・不正サイズ・usage 不足）。 */
export class ArenaError extends Error {
  override readonly name = "ArenaError";
}

/**
 * 出力ストレージの usage。dispatch が書き、必要なら readback のため COPY_SRC を持つ。
 * MUST: transient の領域バッファ（run 寿命も Session 常駐の backing も）と
 * {@link ResidentTensor}（src/gpu/device.ts）も**この定数**で作る。別立てにすると、同じ役割の
 * バッファが 2 つの usage を持つ形になる。
 */
export const STORAGE_USAGE = BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST |
  BUFFER_USAGE.COPY_SRC;

/**
 * サイズクラスは「4 バイト整列した実バイト数そのもの」とする。
 *
 * MUST: 束縛範囲は要求より大きくしない。runtime-sized array を束縛したときの
 * `arrayLength()` は束縛範囲のバイト数から決まるため、切り上げた大きさを束ねると要素数が
 * 静かに変わり、誤った値が例外なしで出る（領域バッファ自体は大きくてよい — 束縛 `size` を
 * この値で切る。ADR 0093 決定 2）。
 * MUST: 計画（`transient-plan.ts`）も見積り（`estimate.ts`）も**この関数**を使う。サイズクラスの
 * 定義が 2 つに分かれると、slot の大きさが実行時の束縛とずれ、同じ理由で沈黙誤値になる。
 * MUST: 値域は**安全整数**（`Number.isInteger` では足りない）。2^53 を超えた要求は
 * `Math.ceil(bytes / 4) * 4` が端数を落とした別の値を返し、要求より小さいバッファを
 * 「切り上げた」顔で配ってしまう（上の MUST が例外なしに破れる）。
 */
export const toSizeClass = (bytes: number): number => {
  if (!Number.isSafeInteger(bytes) || bytes < 0) {
    throw new ArenaError(`確保サイズは 0 以上の安全整数である必要がある: ${bytes}`);
  }
  return Math.max(4, Math.ceil(bytes / 4) * 4);
};

/**
 * 1 run の GPU バッファ実績。`allocCount` / `allocatedBytes` はアリーナが createBuffer した実数
 * （領域 + ホスト書き込み + readback staging）。残り 3 欄は計画（ADR 0093）から executor が写す:
 * `reuseCount` = 先住の slot とバイト範囲を共有した slot の本数（`TransientPlan.sharedSlots`）、
 * `peakTransientBytes` = 中間の同時生存バイトの最大（`TransientPlan.peakLiveBytes`）、
 * `transientBytes` = run 末尾に生存している中間 = pin されたグラフ出力の総バイト数。
 * slot backing に乗った run（中間がアリーナを通らない）では 3 欄とも 0（session-types の NOTE）。
 */
export type ArenaStats = {
  readonly allocCount: number;
  readonly reuseCount: number;
  /** 実際に createBuffer したバイト数の合計。 */
  readonly allocatedBytes: number;
  readonly transientBytes: number;
  readonly peakTransientBytes: number;
};

/** executor が計画から写す 3 欄（{@link ArenaStats} の残り）。 */
export type TransientStats = Pick<
  ArenaStats,
  "reuseCount" | "transientBytes" | "peakTransientBytes"
>;

/**
 * 未 submit のエンコードを**出し切る**手段（`SubmitScheduler.flush` / `submitPending`）。
 *
 * MUST: 戻った時点で未 submit のエンコードが 1 つも残っていないこと。**GPU 完了まで待つかは
 * 注入側の裁量** — {@link RunArena.destroy} が要求するのは「破棄済みバッファを参照する
 * エンコードが submit に残らない」ことだけで、submit 済みコマンドからの参照は WebGPU 的に
 * 安全（実解放は完了まで実装が遅延する）。既定は待つ側（`flush`）で、待ちを別のフェンスへ
 * 集約した経路だけが待たない側を渡す。
 */
export type ArenaFlush = () => Promise<void>;

export class RunArena {
  readonly #device: GPUDevice;
  readonly #flush: ArenaFlush;
  readonly #owned: GPUBuffer[] = [];
  #allocCount = 0;
  #allocatedBytes = 0;
  #transients: TransientStats = { reuseCount: 0, transientBytes: 0, peakTransientBytes: 0 };
  #destroying: Promise<void> | undefined;

  constructor(device: GPUDevice, flush: ArenaFlush) {
    this.#device = device;
    this.#flush = flush;
  }

  /**
   * 計画が決めた領域バッファ（中間バッファの実体）を確保する。offset / size で切って束ねるのは
   * 計画側（src/runtime/recipe.ts）で、アリーナは所有と破棄だけを持つ。
   * MUST: 確保は当該 run のどの dispatch のエンコードよりも前（ADR 0004 の不変条件 — 計画は
   * run の先頭で全領域を確保する）。
   */
  allocRegion(bytes: number): GPUBuffer {
    this.#assertUsable();
    const buffer = this.#device.createBuffer({ size: toSizeClass(bytes), usage: STORAGE_USAGE });
    this.#owned.push(buffer);
    this.#allocCount += 1;
    this.#allocatedBytes += buffer.size;
    return buffer;
  }

  /**
   * ホストが `queue.writeBuffer` で書くバッファ（uniform / 入力 / 重み）を確保する。
   * MUST: この経路のバッファは領域に入れない（冒頭の writeBuffer 追い越しの理由）。
   */
  allocHostWritten(bytes: number, usage: number): GPUBuffer {
    this.#assertUsable();
    if ((usage & BUFFER_USAGE.COPY_DST) === 0) {
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
   * MUST: この経路のバッファも領域外 — MAP_READ は STORAGE と併用できない。
   * 使い**回さない**（1 回の readback で使い捨て）のは設計判断で、仕様上の不能ではない
   * （`unmap()` 後の再利用は可能 — 2026-08-29 実 GPU で確認済み）。使い捨てにしておくと
   * アリーナの一括破棄が後始末の唯一の口になり、失敗経路で `mapAsync` が pending のまま
   * 残っても次 run に汚染が持ち越されない。それでもアリーナが所有するのは「確保と破棄を
   * 1 箇所へ」（ADR 0004）— 破棄が flush-before-destroy に自動で乗る。
   * NOTE: 共有 staging 化（perf-ledger H-9）を採る場合は、失敗経路で pending の staging を
   * 破棄して作り直す復帰規律が必須（無いと以後の全 readback が OperationError の恒久故障）。
   */
  allocHostRead(bytes: number): GPUBuffer {
    this.#assertUsable();
    const buffer = this.#device.createBuffer({
      size: toSizeClass(bytes),
      usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ,
    });
    this.#owned.push(buffer);
    this.#allocCount += 1;
    this.#allocatedBytes += buffer.size;
    return buffer;
  }

  /** 計画から写す診断 3 欄を載せる（{@link ArenaStats} の doc — run の中間を領域で回した後に 1 度）。 */
  recordTransients(stats: TransientStats): void {
    this.#transients = stats;
  }

  get stats(): ArenaStats {
    return {
      allocCount: this.#allocCount,
      allocatedBytes: this.#allocatedBytes,
      ...this.#transients,
    };
  }

  /**
   * 所有する全バッファを破棄する。
   *
   * MUST: 未 submit のエンコードを片付けてから破棄する（discard-or-flush before destroy）。
   * ここが待つのは注入された {@link ArenaFlush} の決着だけなので、**捨てる側（失敗経路の
   * `SubmitScheduler.discard`）は呼び出し側の責務**。破棄済みバッファを参照するエンコードを
   * submit するとコマンドバッファ丸ごと失敗し、同じスケジューラに相乗りしている無関係な
   * dispatch まで実行されないまま、誤った値が静かに残る。
   *
   * NOTE: VRAM 自体が返るのは `device.destroy()` のみ。ここでの破棄は所有の終了。
   */
  destroy(): Promise<void> {
    // MUST: 2 度目の destroy も同じ完了を待つ。先行する flush の完了前に返すと、呼び出し側が
    // 「破棄済み」と誤認したまま次の確保へ進む。
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
    }
  }

  #assertUsable(): void {
    // 破棄後の確保は「破棄されないバッファ」を生む（所有リストから漏れる）ので拒否する。
    if (this.#destroying !== undefined) {
      throw new ArenaError("破棄済みの RunArena は使えない（run ごとに作り直す）");
    }
  }
}
