/**
 * 投機の**自己採算ゲート**（ADR 0096 段 4-B ④）— 「投機が decode に勝っているか」を実行時に
 * 測り、負ける文脈では decode 形（M=1）の 1 step へ落ちる。**パイプライン非依存・GPU 非依存・
 * 時計も RNG も持たない**純関数モジュールなので `speculation.ts` と同じ `src/generation/` に置く。
 *
 * ## なぜ実行時に測るのか（固定の閾値を焼けない）
 *
 * 投機の取り分は課題と host で決まる（`docs/research/2026-09-09-mtp-stage4.md` の実測 —
 * RTX 3080 Ti は抽出 1.81× / 要約 1.41× / 対話 1.18× / 自由文 **0.96×**、Apple M2 は
 * 抽出 1.27× / 要約 1.03× / 対話 0.93× / 自由文 **0.76×**）。採算が合う受理数の閾値
 * `A* = cycle の壁 ÷ decode の壁` も RTX 1.72〜1.88・M2 2.17〜2.6 と host で 0.5 以上動くので、
 * 配布形にも定数にも焼けない。ゲートが測るのは 3 本の量だけで、比
 *
 * ```text
 * (ΣWc / Σdelivered) / W1   … 投機の 1 token あたりの壁 ÷ decode の 1 token あたりの壁
 * ```
 *
 * が 1 を跨いだ側へ倒す（1 未満なら投機が速い）。`Wc` = 投機 cycle の壁・`delivered` = その cycle が
 * 確定させた token 数・`W1` = plain step（M=1）の壁で、どれも呼び手が観測した実測値である。
 *
 * ## 推定器は**ブロック集計**（受理数に移動平均を使わない）
 *
 * 分母の `delivered` は cycle ごとに大きく散る — 実測の受理数分布（RTX 対話）は平均 2.03 に対して
 * **標準偏差 0.88**（43%）である。1 サンプルや数サンプルの平均では「投機が負けている」ことを
 * 判定できない。よって speculate 側の判定は `window`（既定 16）本の cycle を**非重複ブロック**に
 * まとめ、ブロックが満ちるごとに `ΣWc / Σdelivered` を 1 度だけ評価する（ブロックは評価の後に
 * 空にする）。`msPerTokenSpec > W1 × leave` が `confirm`（既定 2）ブロック**連続**で成り立った
 * ときだけ plain へ倒す。
 *
 * 例外は**強い負け**である。比が `leave + strong`（既定 1.01 + 0.15 = 1.16）を超えたブロックは
 * 1 本で plain へ倒す（{@link SpeculationGateOptions.strong}）。ノイズで 1.16 に届くのは
 * 勝っている文脈ではまず起きないので（下の机上の根拠）、大きい負けが「抜けるまでに払う費用」を
 * 1 ブロックぶん削れる。
 *
 * **EWMA を使わないのは種付けの欠陥のため**である。`ewma(undefined, sample) = sample` なので、
 * 受理数の EWMA は**最初の 1 サンプルで種付けされる** — 最初に観測した cycle が受理 0
 * （配送 1 個）だと `A = 1` になり、比 = `Wc / 1 / W1 ≈ 1.7 > leave` で即座に plain へ抜ける。
 * plain 側の探索が 1 cycle 単発だと同じノイズを引くので戻れず、バックオフだけが伸びる。実走で
 * 観測した形がこれで、`200 token / 7 cycle / plain step 190 / 切替 1` — 7 cycle 目で plain に
 * 落ちて 190 step 戻らず、`always`（ゲート無し）の 7,066 ms に対して 9,262 ms（**31% 遅い**）に
 * なった。ブロック集計は満ちるまで判定しないので、この種付けが起きる余地が無い。
 *
 * ## 不感帯は**非対称**（守るのは「大きい負け」だけ）
 *
 * plain へ抜けるのは比 > {@link SpeculationGateOptions.leave}（既定 1.01）、speculate へ戻るのは
 * 比 < {@link SpeculationGateOptions.enter}（既定 0.97）である。対称な 5% の不感帯にすると、
 * 守りたい最大の負けである RTX 自由文（比 1.037）が不感帯に入って**一度も発火しない**。ゲートが
 * 実際に救うのは M2 自由文 −24% / M2 対話 −7% 級の負けで、検収基準もその粒度で読む。
 *
 * ## 机上の根拠（既定のノブがこの値である理由）
 *
 * 受理数の cycle あたりの標準偏差を 0.88（RTX 自由文の実測分布 — 対話は 1.14 でさらに散る）として:
 *
 * - **対話級（比 0.85・A 2.03）**: 16 cycle ブロックの `A` の sd ≈ 0.22 なので、比の sd ≈ 0.09。
 *   1 ブロックが 1.01 を超える確率 ≈ 5%・2 ブロック連続で ≈ 0.3%。200 token のターンは
 *   ≈ 6 ブロックなので、**誤って抜けるのはターンあたり ≈ 1.5%** である。EWMA の 1 サンプル
 *   種付け（上）だと同じ条件でほぼ確実に抜けるので、ここが `confirm = 2` の効き所である。
 * - **M2 自由文（比 1.31）**: 1 ブロックが 1.01 を下回る確率 ≈ 0.1%。よって 2 ブロック =
 *   32 cycle（≈ 3.8 秒）で確実に抜ける — 比 1.31 は「強い負け」なので、実際はその半分の
 *   1 ブロックで抜ける（次の項）。sequence 寿命で 1 回払えばよい費用である。
 * - **戻る側**: `burst`（既定 8）cycle のバーストは比の sd ≈ 0.13。M2 自由文で 0.97 を下回る確率
 *   ≈ 0.5% なので、誤って戻らない。負ける側が払うバーストの費用は M2 自由文で +227 ms/バースト
 *   で、間隔が 16 → 256 と伸びるぶん長い会話では消える。
 * - **強い負けの早抜け（`strong` = 0.15）**: 対話級（比 0.85）のブロックの比の sd は上の 0.09
 *   なので、1.16 を超える確率は ≈ 0.03%（`confirm` の 2 本連続を待たずに倒しても、勝っている
 *   文脈を誤って落とす確率はこの桁である）。損益分岐（比 1.0）のブロックなら ≈ 4% で発火するが、
 *   **そこで抜けても損は 0** なので許容する。
 * - **バーストの早期打ち切り（`burstMin` = 4）**: 比 0.85 のとき 4 cycle の比（sd ≈ 0.19）が
 *   1.16 を超える確率は ≈ 5%。この誤発火は「誤って抜けた後のバースト」でしか起こりえず、代償は
 *   戻りがバースト 1 回ぶん遅れることだけである。効き側は比 1.27（M2 自由文級）で 4 cycle の比が
 *   1.16 を超える確率 ≈ 0.72 なので、負ける文脈のバーストの平均長は 8 cycle → **≈ 4.8 cycle**（実測分布でゲート本体を回した値・打ち切りの 7 割は 4 本目）
 *   に縮む。
 *
 * ## 探索（負けている側にも「戻る道」を残す）
 *
 * - **plain 中**は幾何バックオフ（16 → 32 → 64 → … → 上限 256 step）で投機を `burst` cycle
 *   **連続**で試す。1 cycle 単発の探索では上の sd 0.88 に埋もれて判定できない（EWMA 設計が
 *   戻れなかった第 2 の理由がこれである）。バーストの `ΣWc / Σdelivered < W1 × enter` なら
 *   speculate へ戻し、間隔は `exploreBase` へ戻す。外れたら間隔を倍にする。
 * - **外れが確定したバーストは回し切らない**。`burstMin`（既定 4）本目以降の各観測で比が
 *   `leave + strong` を超えていたら、残りの cycle を回さずにそのバーストを「外れ」として畳む
 *   （{@link SpeculationGateOptions.burstMin}）。負ける文脈が払う探索費はここが主で、`burst`
 *   そのものを縮めるのと違って**戻れる側の判定精度は落ちない**（当たりの判定は満ちたバーストの
 *   集計のままである）。
 * - **speculate 中**は {@link SpeculationGateOptions.exploreBase} 観測ごとに 1 回 plain を測る
 *   （`W1` の鮮度が「抜ける」判断の材料そのもの）。既定は判定ブロックと同じ 16 で、費用は
 *   16 cycle につき plain 価格の 1 token（実測の壁では always 比 1% 前後）である。この
 *   plain step は**ブロックの cycle 数に数えない**（投機 cycle の集計だけがブロックである）。
 * - **`W1` の初回サンプルは 2 回目の decide**（1 cycle 回した直後）。未観測のまま 16 cycle 回すと
 *   M2 自由文で 2% を先に失う。`W1` は plain step の壁なので host で安定しており、ここだけは
 *   EWMA（係数 {@link SpeculationGateOptions.alpha}）でよい。
 *
 * ## 呼び出し規約
 *
 * MUST: 1 cycle につき {@link SpeculationGate.decide} を 1 回呼び、その cycle の観測を
 * {@link SpeculationGate.observeCycle} / {@link SpeculationGate.observePlain} で 1 回返す。
 * `decide()` は**状態を一切動かさない読み取り**である（モードの切替も `switches` の加算も観測の
 * 側で起きる）ので、同じ cycle で 2 度読んでも決定は割れない。
 *
 * MUST: 呼び手が「壁の観測に混ぜない」と決めた cycle（各ターンの最初の cycle と予算末尾の強制
 * plain）では、観測の代わりに {@link SpeculationGate.skip} を 1 回呼ぶ。何も呼ばずに次の cycle へ
 * 進むとゲートはその cycle が無かったものとして決定を出すので、探索の周期が止まり、上の
 * 「`W1` の初回サンプルは 2 回目の decide」も「観測に混ぜる最初の cycle」まで遅れる。
 */

/** 次の 1 cycle を投機で回すか、decode 形（M=1）の 1 step で回すか。 */
export type SpeculationDecision = "speculate" | "plain";

/** ゲートのノブ（すべて既定値を持つ — 公開面には出さない内部のノブである）。 */
export type SpeculationGateOptions = {
  /** `W1`（plain step の壁）の EWMA 係数（既定 0.2・`0 < alpha ≤ 1`）。 */
  readonly alpha?: number;
  /** speculate 側の判定ブロックの cycle 数（既定 16・2 以上の整数）。 */
  readonly window?: number;
  /** plain へ抜けるのに要る「負けブロック」の連続数（既定 2・1 以上の整数）。 */
  readonly confirm?: number;
  /** plain 側の探索バーストの cycle 数（既定 8・1 以上の整数）。 */
  readonly burst?: number;
  /**
   * バーストを早期に打ち切れる最小の cycle 数（既定 `min(4, burst)`・`1 ≤ burstMin ≤ burst` の
   * 整数）。ここから `burst - 1` 本目までの各観測で比が `leave + strong` を超えていたら、
   * 残りを回さずに「外れたバースト」として畳む。
   */
  readonly burstMin?: number;
  /** 探索の基本間隔（既定 16・cycle / step 単位）。 */
  readonly exploreBase?: number;
  /** plain 側バックオフの上限（既定 256）。 */
  readonly exploreMax?: number;
  /** speculate へ戻る条件 `ΣWc/Σdelivered < W1 × enter`（既定 0.97・`enter < 1`）。 */
  readonly enter?: number;
  /** plain へ抜ける条件 `ΣWc/Σdelivered > W1 × leave`（既定 1.01・`1 < leave`）。 */
  readonly leave?: number;
  /**
   * 「強い負け」の上乗せ（既定 0.15・正の有限数）。比が `leave + strong` を超えたブロックは
   * `confirm` の連続を待たずに 1 本で plain へ倒し、進行中の探索バーストは `burstMin` 本目以降で
   * 打ち切る。
   */
  readonly strong?: number;
};

export type SpeculationGate = {
  /**
   * 次に回すのはどちらか（観測と内部 counter だけの決定的な**読み取り** — 時計は observe* が
   * 受け取る）。
   *
   * MUST: 1 cycle に 1 回呼ぶ。状態は一切動かさないので、同じ cycle で何度読んでも同じ値で、
   * 決定を持ち越す必要も無い（モードの切替と `switches` の加算はブロック / バーストが満ちた
   * 観測の側で起きる）。
   */
  decide(): SpeculationDecision;
  /**
   * 投機 cycle の観測（壁 ms と、その cycle が配送した token 数 = `confirmed.length`）。
   *
   * speculate 中はブロックへ、plain 中は探索バーストへ積む。判定が動くのは満ちた時と、
   * バーストが `burstMin` 本目以降で強い負けを出した時（そのバーストの打ち切り）だけである。
   */
  observeCycle(wallMs: number, delivered: number): void;
  /** plain step（M=1）の観測（壁 ms）— `W1` の EWMA だけを動かす。 */
  observePlain(wallMs: number): void;
  /**
   * 観測を返さない cycle（呼び手が「壁の観測に混ぜない」と決めた cycle — 決定を諮らずに形が
   * 決まった cycle も含む）。
   *
   * 動くのは探索の周期（次の探索までの残り）だけで、mode も `W1` もブロックも動かさない。混ぜ
   * ない cycle でも GPU の仕事は 1 本走っているので、周期はそのぶん進めるのが正しい。
   */
  skip(): void;
  /** speculate ↔ plain の切替回数（探索バーストそのものは数えない）。 */
  readonly switches: number;
  /** 現在の定常モード（探索バーストで逆側を回している間も変わらない）。 */
  readonly mode: SpeculationDecision;
};

/** 正の有限数の門（比・係数のノブ）。 */
const assertPositive = (name: string, value: number): void => {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`speculation gate: ${name} ${value} が正の有限数でない`);
  }
};

/** cycle 数の門 — 端数の cycle 数は「何 cycle ごと」を意味できない。 */
const assertInterval = (name: string, value: number, least: number): void => {
  if (!Number.isSafeInteger(value) || value < least) {
    throw new Error(`speculation gate: ${name} ${value} が ${least} 以上の整数でない`);
  }
};

export const createSpeculationGate = (
  options: SpeculationGateOptions = {},
): SpeculationGate => {
  const alpha = options.alpha ?? 0.2;
  const window = options.window ?? 16;
  const confirm = options.confirm ?? 2;
  const burst = options.burst ?? 8;
  // 既定を `burst` で抑えるのは、`burst` を縮めた席（バースト 1 cycle など）で「打ち切る余地が
  // 無い」のが正しい形であり、既定の組み合わせを門で弾く理由が無いためである。
  const burstMin = options.burstMin ?? Math.min(4, burst);
  const exploreBase = options.exploreBase ?? 16;
  const exploreMax = options.exploreMax ?? 256;
  const enter = options.enter ?? 0.97;
  const leave = options.leave ?? 1.01;
  const strong = options.strong ?? 0.15;
  assertPositive("alpha", alpha);
  // α > 1 の EWMA は過去へ負の重みを載せる（平均ではなくなる）ので上も閉じる。
  if (alpha > 1) throw new Error(`speculation gate: alpha ${alpha} が 0 < α ≤ 1 の外`);
  // ブロックが 1 cycle だと「ブロック集計」が 1 サンプルの判定に戻る（種付けの欠陥そのもの）。
  assertInterval("window", window, 2);
  assertInterval("confirm", confirm, 1);
  assertInterval("burst", burst, 1);
  assertInterval("burstMin", burstMin, 1);
  // `burstMin > burst` は「打ち切りが一度も発火しない」死んだノブなので、黙って通さない。
  if (burstMin > burst) {
    throw new Error(`speculation gate: burstMin ${burstMin} が burst ${burst} を超えている`);
  }
  assertInterval("exploreBase", exploreBase, 1);
  assertInterval("exploreMax", exploreMax, exploreBase);
  assertPositive("enter", enter);
  assertPositive("leave", leave);
  assertPositive("strong", strong);
  // 不感帯は 1 を挟む（`enter ≥ 1` だと負けている最中に戻り、`leave ≤ 1` だと勝っていても抜ける）。
  if (!(enter < 1 && 1 < leave)) {
    throw new Error(`speculation gate: enter ${enter} < 1 < leave ${leave} でない`);
  }

  let mode: SpeculationDecision = "speculate";
  /** plain step の壁 ms の EWMA（未観測は `undefined`）。 */
  let plainWall: number | undefined;
  /** 現モードで観測した回数（モード切替と plain 側のバースト完了で 0 へ戻る）。 */
  let sinceProbe = 0;
  /** plain 側の探索間隔（外れるたびに倍・当たれば `exploreBase` へ戻る）。 */
  let backoff = exploreBase;
  let switches = 0;
  /**
   * 集計中の一群（speculate 中は `window` 本のブロック・plain 中は `burst` 本の探索バースト）。
   *
   * 2 つを 1 組の変数で持てるのは、モードの切替が必ずここを空にするためである（ブロックと
   * バーストが同時に生きることは無い）。
   */
  let batchCycles = 0;
  let batchWall = 0;
  let batchDelivered = 0;
  /** `leave` を連続で超えたブロックの数（`confirm` に届いたら抜ける）。 */
  let overRuns = 0;

  const ewma = (value: number | undefined, sample: number): number =>
    value === undefined ? sample : value + alpha * (sample - value);

  const resetBatch = (): void => {
    batchCycles = 0;
    batchWall = 0;
    batchDelivered = 0;
  };

  /**
   * 一群の 1 token あたりの壁 ms（`ΣWc / Σdelivered`）と `W1` の比。
   *
   * `delivered` は cycle あたり 1 個以上確定するので 0 にはならないが、割り算の前に見る
   * （`W1` が未観測の状態で倒すと、根拠の無い切替になる）。
   */
  const batchRatio = (): number | undefined =>
    batchDelivered <= 0 || plainWall === undefined || plainWall <= 0
      ? undefined
      : (batchWall / batchDelivered) / plainWall;

  /** plain へ倒す（ブロックが `confirm` 本連続で負けた / 1 本が強い負けだったとき）。 */
  const leaveSpeculation = (): void => {
    mode = "plain";
    switches += 1;
    backoff = exploreBase;
    sinceProbe = 0;
    overRuns = 0;
    resetBatch();
  };

  /** 外れたバースト（満ちても `enter` を下回らない / 途中で強い負けが出た）を畳む。 */
  const missBurst = (): void => {
    resetBatch();
    // 次の探索を遠ざけ、間隔を数え直す。
    sinceProbe = 0;
    backoff = Math.min(backoff * 2, exploreMax);
  };

  /** speculate へ戻す（探索バーストが `enter` を下回ったとき）。 */
  const enterSpeculation = (): void => {
    mode = "speculate";
    switches += 1;
    backoff = exploreBase;
    sinceProbe = 0;
    overRuns = 0;
    resetBatch();
  };

  return {
    /**
     * 今の状態から決定を出す（読み取りだけ — {@link SpeculationGate.decide} の MUST）。
     *
     * plain 側は「バースト中は続ける / 間隔ぶん回したらバーストを始める」、speculate 側は
     * 「`W1` が未観測なら 1 cycle 回した直後に 1 step / 以後は `exploreBase` ごとに 1 step」。
     */
    decide: (): SpeculationDecision => {
      if (mode === "plain") {
        // `batchCycles > 0` = 探索バーストが進行中（満ちた時と、`burstMin` 以降の打ち切りで
        // 0 へ戻る）。打ち切りが即座に plain へ返るのはこの読みだけで成り立つ。
        return batchCycles > 0 || sinceProbe >= backoff ? "speculate" : "plain";
      }
      // `W1` を 1 度も測っていないなら、1 cycle 回した直後に 1 step だけ plain を挟む。
      if (plainWall === undefined) return sinceProbe >= 1 ? "plain" : "speculate";
      return sinceProbe % exploreBase === exploreBase - 1 ? "plain" : "speculate";
    },
    observeCycle: (wallMs: number, delivered: number): void => {
      batchCycles += 1;
      batchWall += wallMs;
      batchDelivered += delivered;
      if (mode === "plain") {
        // plain 中の探索バースト — 満ちるまでは「戻る」判定も周期の数え直しもしない。ただし
        // `burstMin` 本目からは強い負けだけを見て、外れが確定したバーストを畳む。
        if (batchCycles < burst) {
          if (batchCycles < burstMin) return;
          const probe = batchRatio();
          if (probe !== undefined && probe > leave + strong) missBurst();
          return;
        }
        const measured = batchRatio();
        if (measured !== undefined && measured < enter) {
          enterSpeculation();
          return;
        }
        missBurst();
        return;
      }
      sinceProbe += 1;
      if (batchCycles < window) return;
      // 満ちたブロック 1 本ぶんの判定（`W1` 未観測なら判定を持たない = 連続を切る）。
      const measured = batchRatio();
      // 強い負けは `confirm` の連続を待たない（1 本で倒す）。
      if (measured !== undefined && measured > leave + strong) {
        leaveSpeculation();
        return;
      }
      overRuns = measured !== undefined && measured > leave ? overRuns + 1 : 0;
      if (overRuns >= confirm) leaveSpeculation();
      else resetBatch();
    },
    observePlain: (wallMs: number): void => {
      plainWall = ewma(plainWall, wallMs);
      sinceProbe += 1;
    },
    skip: (): void => {
      sinceProbe += 1;
    },
    get switches(): number {
      return switches;
    },
    get mode(): SpeculationDecision {
      return mode;
    },
  };
};
