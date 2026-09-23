/**
 * 計測プロセスに仕掛ける**観測器 2 本** — ホスト RAM の標本化と `crypto.subtle.digest` の計数。
 *
 * どちらも「区間（{@link ProbePhase}）ごとに分ける」のが要点である。ADR 0108 段階分解表 段 2 の
 * 検収②は *Session 準備完了まで* の external 最大値を要求しており、最小の生成 1 回まで含めた
 * 値と混ぜると「読み込みのピーク」と「実行のピーク」が同じ数字に化ける。
 */

/** 計測の区間。`load` = `fromPretrained` の決着まで / `run` = 最小の生成 1 回。 */
export type ProbePhase = "load" | "run";

const PHASES: readonly ProbePhase[] = ["load", "run"];

// ---------------------------------------------------------------------------
// ① ホスト RAM の標本化
// ---------------------------------------------------------------------------

/** 1 区間ぶんのピーク。 */
export type PhasePeak = {
  /** `Deno.memoryUsage().rss` の最大値。 */
  readonly rssMaxBytes: number;
  /**
   * `Deno.memoryUsage().external` の最大値（V8 の外部メモリ = ArrayBuffer の実勢）。
   * 取得バッファ・展開 scratch はここに出る（JS ヒープではない）。
   */
  readonly externalMaxBytes: number;
  /** この区間で取れた標本数（0 なら区間が標本間隔より短かった＝境界の 2 点だけで見ている）。 */
  readonly samples: number;
};

export type MemorySampler = {
  /** 以後の標本を別の区間へ付け替える（切替の瞬間にも 1 点取る）。 */
  readonly phase: (phase: ProbePhase) => void;
  /** 標本化を止める（止める瞬間にも 1 点取る）。 */
  readonly stop: () => void;
  /** 仕掛けた瞬間の値（ピークから引けば「この計測が増やしたぶん」になる）。 */
  readonly baseline: PhasePeak;
  /** 区間 1 つのピーク。 */
  readonly peak: (phase: ProbePhase) => PhasePeak;
  /** 全区間を通したピーク。 */
  readonly total: () => PhasePeak;
};

/**
 * 標本間隔の既定。50ms は既存の `rss` 標本化の流儀をそのまま継ぐ。
 *
 * NOTE: 標本化である以上、間隔より短い尖りは取りこぼす（下限の観測であって上限の証明ではない）。
 * 宣言だけで閉じる見積り（container-v1 §11）と突き合わせて読むこと。
 */
const SAMPLE_INTERVAL_MS = 50;

/** ホスト RAM の標本化を始める。MUST: 呼び手は必ず {@link MemorySampler.stop} を通す。 */
export const sampleMemory = (intervalMs: number = SAMPLE_INTERVAL_MS): MemorySampler => {
  const peaks = new Map<ProbePhase, { rss: number; external: number; samples: number }>(
    PHASES.map((phase) => [phase, { rss: 0, external: 0, samples: 0 }]),
  );
  let current: ProbePhase = "load";
  const take = (counted: boolean): void => {
    const { rss, external } = Deno.memoryUsage();
    const slot = peaks.get(current);
    if (slot === undefined) return;
    slot.rss = Math.max(slot.rss, rss);
    slot.external = Math.max(slot.external, external);
    if (counted) slot.samples += 1;
  };
  const usage = Deno.memoryUsage();
  const baseline: PhasePeak = {
    rssMaxBytes: usage.rss,
    externalMaxBytes: usage.external,
    samples: 1,
  };
  const timer = setInterval(() => take(true), intervalMs);
  const peakOf = (phase: ProbePhase): PhasePeak => {
    const slot = peaks.get(phase);
    if (slot === undefined) throw new Error(`ram-peak: 区間 ${phase} は無い`);
    return { rssMaxBytes: slot.rss, externalMaxBytes: slot.external, samples: slot.samples };
  };
  return {
    phase: (phase) => {
      // 切替の前後で 1 点ずつ取る（間隔より短い区間でも両端の値は残る）。
      take(false);
      current = phase;
      take(false);
    },
    stop: () => {
      take(false);
      clearInterval(timer);
    },
    baseline,
    peak: peakOf,
    total: () => {
      const all = PHASES.map(peakOf);
      return {
        rssMaxBytes: Math.max(...all.map((peak) => peak.rssMaxBytes)),
        externalMaxBytes: Math.max(...all.map((peak) => peak.externalMaxBytes)),
        samples: all.reduce((sum, peak) => sum + peak.samples, 0),
      };
    },
  };
};

// ---------------------------------------------------------------------------
// ② digest の計数
// ---------------------------------------------------------------------------

/** 1 区間ぶんの digest 実績（バイト長ごとの回数まで残す — 後から役割別に仕分けるため）。 */
export type DigestTally = {
  readonly calls: number;
  readonly bytes: number;
  /** バイト長 → 回数。 */
  readonly sizes: ReadonlyMap<number, number>;
};

export type DigestCounter = {
  /** 以後の呼びを別の区間へ付け替える。 */
  readonly phase: (phase: ProbePhase) => void;
  /** 包みを外して元の `digest` に戻す（MUST: 計測が終わったら必ず外す）。 */
  readonly stop: () => void;
  readonly tally: (phase: ProbePhase) => DigestTally;
};

/**
 * `crypto.subtle.digest` を包んで**呼び出し回数と総バイト数**を数える。
 *
 * NOTE（重要・実測で判明）: これで数えられるのは「materialize 済みバイト列の一括 digest」だけ
 * である。取得層（`@hdae/fetch-cache` 0.8.0）の cold 経路 — `prefetchUrl` がファイルを流しながら
 * 掛ける逐次 sha256 — は**純 TS の自前実装**（`src/sha256.ts`。`crypto.subtle.digest` が一括
 * 専用でストリームに使えないため）なので、ここには 1 度も現れない。したがって計数に出るのは
 *
 * - `openContainer` が 2 文書（グラフ記述 / モデル記述）を期待値と突合する digest（常に掛かる）
 * - `fetchBytes` 経由で取る資産（manifest の `assets`）の全量 digest（cold のみ）
 * - 未検証の取得元（ローカルディレクトリ）で `readBlock` が block ごとに掛ける digest
 *
 * の 3 種で、**cold の part 全量 digest は見えない**。検収③（warm で digest 0 回）の判定には
 * 「宣言された descriptor 文書長と一致する呼び」を除いた**払い出し側の digest**
 * （`payload`）を使うこと — 仕分けは {@link splitDigest}。
 */
export const countDigests = (): DigestCounter => {
  const tallies = new Map<ProbePhase, { calls: number; bytes: number; sizes: Map<number, number> }>(
    PHASES.map((phase) => [phase, { calls: 0, bytes: 0, sizes: new Map<number, number>() }]),
  );
  let current: ProbePhase = "load";
  const subtle = crypto.subtle;
  const original = subtle.digest.bind(subtle);
  const wrapped: SubtleCrypto["digest"] = (algorithm, data) => {
    const slot = tallies.get(current);
    if (slot !== undefined) {
      const bytes = data.byteLength;
      slot.calls += 1;
      slot.bytes += bytes;
      slot.sizes.set(bytes, (slot.sizes.get(bytes) ?? 0) + 1);
    }
    return original(algorithm, data);
  };
  // MUST: `defineProperty` で差し替える（`SubtleCrypto.prototype` の getter 経由ではなく実体の
  // 欄として置く）— 取得層と runtime のどちらも `crypto.subtle.digest(...)` という同じ綴りで
  // 呼ぶので、この 1 点で両方を通せる。
  Object.defineProperty(subtle, "digest", {
    value: wrapped,
    configurable: true,
    writable: true,
  });
  return {
    phase: (phase) => {
      current = phase;
    },
    stop: () => {
      Object.defineProperty(subtle, "digest", {
        value: original,
        configurable: true,
        writable: true,
      });
    },
    tally: (phase) => {
      const slot = tallies.get(phase);
      if (slot === undefined) throw new Error(`ram-peak: 区間 ${phase} は無い`);
      return { calls: slot.calls, bytes: slot.bytes, sizes: slot.sizes };
    },
  };
};

/** 役割別に仕分けた digest 実績。 */
export type DigestSplit = {
  readonly calls: number;
  readonly bytes: number;
  /**
   * descriptor 2 文書の突合ぶん（container-v1 §7 の①）。**cold でも warm でも必ず掛かる**ので、
   * 検収③の判定からは外す。
   */
  readonly descriptorCalls: number;
  readonly descriptorBytes: number;
  /** それ以外（資産の全量検証・block ごとの digest）= **検収③が 0 を要求する側**。 */
  readonly payloadCalls: number;
  readonly payloadBytes: number;
  /** 1 回の digest に渡った最大バイト数（part 級の digest が混じっていないかの傍証）。 */
  readonly maxBytes: number;
};

/**
 * バイト長で digest を仕分ける。`descriptorLengths` は manifest が宣言した 2 文書の長さ
 * （`container.descriptor.graph.length` / `.model.length`）の集合で、**宣言だけで閉じる**
 * 仕分けである（runtime に digest の出所を聞ける席は無い）。
 *
 * NOTE: 払い出し側の digest がたまたま descriptor 文書と同じ長さなら descriptor 側に数えられる。
 * 判定の傍証であって証明ではない（検収③の判定は人が読む — ADR 0108 段階分解表 段 2）。
 */
export const splitDigest = (
  tally: DigestTally,
  descriptorLengths: ReadonlySet<number>,
): DigestSplit => {
  let descriptorCalls = 0;
  let descriptorBytes = 0;
  let maxBytes = 0;
  for (const [bytes, count] of tally.sizes) {
    maxBytes = Math.max(maxBytes, bytes);
    if (!descriptorLengths.has(bytes)) continue;
    descriptorCalls += count;
    descriptorBytes += bytes * count;
  }
  return {
    calls: tally.calls,
    bytes: tally.bytes,
    descriptorCalls,
    descriptorBytes,
    payloadCalls: tally.calls - descriptorCalls,
    payloadBytes: tally.bytes - descriptorBytes,
    maxBytes,
  };
};

/** 2 区間の実績を足す（出力の合計欄）。 */
export const mergeTallies = (tallies: readonly DigestTally[]): DigestTally => {
  const sizes = new Map<number, number>();
  let calls = 0;
  let bytes = 0;
  for (const tally of tallies) {
    calls += tally.calls;
    bytes += tally.bytes;
    for (const [size, count] of tally.sizes) sizes.set(size, (sizes.get(size) ?? 0) + count);
  }
  return { calls, bytes, sizes };
};
