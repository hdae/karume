/**
 * ホスト側 sampling（温度 / top-k / top-p / repetition penalty / logit bias / seed）と、
 * EOS **集合**での停止判定。**パイプライン非依存の共通処理**なので `greedy.ts` と同じ
 * `src/generation/` に置く（ADR 0083 決定 7 — 置き場の裁定そのもの）。
 *
 * 入口は「最終行 logits `[1,1,V]` の生データ」（ADR 0083 決定 6）。GPU 側は argmax も topk も
 * 持たないので、ここは**全語彙を見られる**前提で書ける — top-k に実装上限（ADR 0068 追記 2 の
 * `k ≤ 63`）は無く、repetition penalty も full-vocab nucleus もそのまま表現できる。これが
 * 「topk 製品グラフを採らない」根拠 3 の裏返しである。
 *
 * ## MUST: HF の token 列 parity は取れない
 *
 * RNG は splitmix64（`random.ts`）で torch の Philox とは別物 — **同じ seed でも同じ列に
 * ならない**（ADR 0083 決定 7 の MUST）。参照突合は温度 0（= greedy 縮退）で採り、抽選が走る
 * 経路は自前 fixture と分布門で縛る。
 *
 * ## この層の既定は greedy
 *
 * 既定は `temperature = 0` = argmax の縮退形で、推奨値（gemma-4-E2B-it = temperature 1.0 /
 * top_k 64 / top_p 0.95）を注入するのは**配布形の責務**（ADR 0083 決定 7 — 前例は anima の
 * `scheduler.type`）。「sampler 未宣言 = 決定的」にしておくと、既存 golden との parity 門が
 * この経路で生き続ける。
 *
 * ## 加工の順序（HF の `LogitsProcessorList` と同じ）
 *
 * ① repetition penalty → ② logit bias → ③ 温度で除算 → ④ top-k → ⑤ top-p → ⑥ 多項抽選。
 * ①②は温度の**前**（HF は processor と warper で段が分かれており、penalty / bias は前段）。
 * 順序を入れ替えると同じノブでも別の分布になるので、ここが契約である。
 */

import { Randu } from "./random.ts";

/** sampling の指定（省略時は greedy = 温度 0 の縮退形）。 */
export type SamplerSpec = {
  /**
   * softmax の前に logits を割る温度（0 以上・既定 0）。
   *
   * `0` は **greedy**（argmax）で、RNG を消費せず top-k / top-p も効かない（候補を削っても
   * 最大値は残るので結果が変わらない）。repetition penalty と logit bias は温度 0 でも効く。
   */
  readonly temperature?: number;
  /**
   * 候補を確率上位 `topK` 件に絞る（1 以上の整数・**上限なし**）。語彙数を超える指定は語彙数へ
   * 丸める（HF の `min(top_k, vocab)` と同じ）。
   */
  readonly topK?: number;
  /**
   * 累積確率が `topP` に達するまでの**最小の候補集合**に絞る（0 < topP ≤ 1）。`1` は絞らない。
   * 最上位 1 件は必ず残る（HF の `min_tokens_to_keep = 1`）。
   */
  readonly topP?: number;
  /**
   * 既出 token の logit を弱める係数（正の有限数・`1` は無効化）。向きは HF の
   * `RepetitionPenaltyLogitsProcessor` と同じで、**負の logit は掛け・非負の logit は割る**
   * （どちらも係数 > 1 なら値が下がる）。同じ token が履歴に何度出ても**適用は 1 回**。
   */
  readonly repetitionPenalty?: number;
  /**
   * token id → logit への加算（値は有限数、または `-Infinity` = その token の禁止）。
   * キーが語彙の外なら fail loudly（黙って無視すると「効かない bias」が静かに残る）。
   */
  readonly logitBias?: ReadonlyMap<number, number>;
  /** 抽選列の seed（非負の安全整数・既定 0 — 同じ seed なら同じ token 列）。 */
  readonly seed?: number;
};

/**
 * 抽選対象（除外済み・確率は候補集合の中で正規化済み。`tokens[i]` が `probabilities[i]` で
 * 選ばれる）。温度 0 では長さ 1（argmax の 1 点分布）になる。
 *
 * 並びは **`topK` / `topP` のどちらかが指定されていれば確率降順**（top-p が「上位からの累積」で
 * 定義されているため必要）、どちらも無ければ token id 昇順 — 絞り込みが無い指定で語彙全体
 * （V = 262,144）を毎 step 並べ替えないため。**並びは抽選結果に影響しない**（多項抽選は
 * 順序に依らない）ので、この差は観測の便宜だけに効く。
 */
export type SamplerDistribution = {
  readonly tokens: Int32Array<ArrayBuffer>;
  readonly probabilities: Float64Array<ArrayBuffer>;
};

/** 1 本の生成に張り付く抽選器（RNG 状態を持つので step をまたいで使い回す）。 */
export type Sampler = {
  /**
   * 最終行 logits から次の token を 1 個選ぶ。
   *
   * `history` は repetition penalty が見る**それまでの token 列**（prompt を含む — HF が
   * `input_ids` 全体に掛けるのと同じ）。penalty を使わない指定では読まれない。
   */
  next(logits: Float32Array<ArrayBuffer>, history: readonly number[]): number;
};

const assertFiniteAtLeast = (value: number, minimum: number, where: string): void => {
  if (!Number.isFinite(value) || value < minimum) {
    throw new RangeError(`${where} ${value} が ${minimum} 以上の有限数でない`);
  }
};

/**
 * 指定の受理集合（fail loudly の唯一の位置）。
 *
 * MUST: 黙って丸めない。`temperature: -1` や `topP: 0` を既定へ畳むと「効いていないノブ」が
 * 静かに残り、出力の違いからしか気づけない。
 */
const assertSpec = (spec: SamplerSpec): void => {
  if (spec.temperature !== undefined) assertFiniteAtLeast(spec.temperature, 0, "temperature");
  if (spec.topK !== undefined && (!Number.isSafeInteger(spec.topK) || spec.topK < 1)) {
    throw new RangeError(`topK ${spec.topK} が 1 以上の整数でない`);
  }
  if (spec.topP !== undefined) {
    if (!Number.isFinite(spec.topP) || spec.topP <= 0 || spec.topP > 1) {
      throw new RangeError(`topP ${spec.topP} が 0 < topP ≤ 1 の範囲にない`);
    }
  }
  if (spec.repetitionPenalty !== undefined) {
    if (!Number.isFinite(spec.repetitionPenalty) || spec.repetitionPenalty <= 0) {
      throw new RangeError(`repetitionPenalty ${spec.repetitionPenalty} が正の有限数でない`);
    }
  }
  for (const [token, bias] of spec.logitBias ?? []) {
    if (!Number.isSafeInteger(token) || token < 0) {
      throw new RangeError(`logitBias のキー ${token} が 0 以上の整数でない`);
    }
    // `-Infinity` は「その token を禁止する」慣用（HF の `SequenceBias` と同じ）。`+Infinity` と
    // `NaN` は softmax を NaN にするだけなので受けない。
    if (Number.isNaN(bias) || bias === Number.POSITIVE_INFINITY) {
      throw new RangeError(`logitBias[${token}] ${bias} が有限数でも -Infinity でもない`);
    }
  }
};

/**
 * ①repetition penalty + ②logit bias + ③温度 を掛けた logits。
 *
 * 何も掛からない指定では**入力をそのまま返す**（V = 262,144 の複製を毎 step 作らない）。
 * 返り値は読むだけで、書き換えてはならない。
 */
const processLogits = (
  logits: Float32Array<ArrayBuffer>,
  spec: SamplerSpec,
  history: readonly number[],
): Float32Array<ArrayBuffer> => {
  const temperature = spec.temperature ?? 0;
  const penalty = spec.repetitionPenalty ?? 1;
  const bias = spec.logitBias;
  const scaled = temperature !== 0 && temperature !== 1;
  if (penalty === 1 && (bias === undefined || bias.size === 0) && !scaled) return logits;

  const processed = new Float32Array(logits.length);
  processed.set(logits);
  if (penalty !== 1) {
    // MUST: 同じ token が履歴に何度出ても 1 回だけ（HF は `gather` した**元の値**を書き戻すので
    // 重複は冪等）。素直に履歴を舐めると重複ぶん累乗され、別のノブになる。
    for (const token of new Set(history)) {
      if (!Number.isSafeInteger(token) || token < 0 || token >= processed.length) {
        throw new RangeError(`履歴の token id ${token} が語彙 0..${processed.length - 1} の外`);
      }
      const value = processed[token];
      // HF `RepetitionPenaltyLogitsProcessor` の向き: 負なら掛け・非負なら割る。
      processed[token] = value < 0 ? value * penalty : value / penalty;
    }
  }
  if (bias !== undefined) {
    for (const [token, amount] of bias) {
      if (token >= processed.length) {
        throw new RangeError(`logitBias のキー ${token} が語彙 0..${processed.length - 1} の外`);
      }
      processed[token] += amount;
    }
  }
  if (scaled) {
    for (let token = 0; token < processed.length; token += 1) processed[token] /= temperature;
  }
  return processed;
};

/**
 * NaN の混入を入口で落とす（fail loudly）。
 *
 * MUST: 走査してでも落とす。NaN は**あらゆる比較が false になる**ので、下流の 3 箇所が黙って
 * 壊れる — argmax は素通りして別 token を返し、top-k の閾値は一度 NaN になると以降の候補を
 * 全部弾き、softmax の分母は NaN になって確率が全滅する。範囲外添字の gather は行ごと NaN 汚染
 * するので（[known-issues] / [limitations]）、これは実際に起きる形である。
 *
 * 代償は V = 262,144 の比較 1 周（0.2ms 級）で、decode の壁（ADR 0082）に対して 1% 未満。
 */
const assertNoNaN = (logits: Float32Array<ArrayBuffer>): void => {
  for (let token = 0; token < logits.length; token += 1) {
    // MUST: `Number.isNaN` を使わない（呼び出しが V 回入る）— 自分自身と等しくないのが NaN。
    if (logits[token] !== logits[token]) {
      throw new Error(`logits[${token}] が NaN（非有限） — token id へ畳まずここで落とす`);
    }
  }
};

/**
 * 最大値の添字（同値は**先に出た方** — torch の `argmax` と同じ tie-break）。
 *
 * MUST: 最大値が非有限なら落とす。位置表の外の gather や壊れた重みは logits を非有限にするが、
 * argmax はそれを**もっともらしい token id に畳む**ので、畳まれる前がここしかない。
 * NaN は {@link assertNoNaN} が先に落としているので、ここで見るのは ±Infinity だけ。
 */
const argmax = (logits: Float32Array<ArrayBuffer>): number => {
  let best = 0;
  let bestValue = logits[0];
  for (let token = 1; token < logits.length; token += 1) {
    if (logits[token] > bestValue) {
      bestValue = logits[token];
      best = token;
    }
  }
  if (!Number.isFinite(bestValue)) {
    throw new Error(`logits の最大値が非有限（${bestValue}） — token id へ畳まずここで落とす`);
  }
  return best;
};

/**
 * 上位 `k` 件の token id を**降順**で選ぶ（同値は id の小さい方が上 — {@link argmax} と同じ
 * tie-break）。
 *
 * 全体ソートを避けるのは V = 262,144 に対して毎 step 走るため。長さ ≤ k の降順配列へ挿入する
 * だけなので、走査 V 回 + 改善が起きたときだけ O(k) の押し出しで済む。
 *
 * NOTE: HF の `TopKLogitsWarper` は「k 番目の値**未満**を落とす」閾値形なので、同値が並ぶと
 * k 件より多く残る。こちらは id で決着させて厳密に k 件に絞る — HF との token 列 parity は
 * どのみち取れない（RNG が別物）ので、`topK: 1` が greedy と厳密に一致する側を採る。
 */
const selectTopK = (logits: Float32Array<ArrayBuffer>, k: number): number[] => {
  const selected: number[] = [];
  let floor = Number.NEGATIVE_INFINITY;
  for (let token = 0; token < logits.length; token += 1) {
    const value = logits[token];
    if (selected.length === k && !(value > floor)) continue;
    let at = selected.length;
    // 同値では止まる（`<` が厳密）ので、先に入った小さい id が上に残る。
    while (at > 0 && logits[selected[at - 1]] < value) at -= 1;
    selected.splice(at, 0, token);
    if (selected.length > k) selected.pop();
    floor = logits[selected[selected.length - 1]];
  }
  return selected;
};

/** 全 token id を logit の降順（同値は id 昇順 — {@link selectTopK} と同じ決着）に並べる。 */
const sortAllDescending = (logits: Float32Array<ArrayBuffer>): number[] => {
  const order = Array.from({ length: logits.length }, (_unused, token) => token);
  order.sort((left, right) => logits[right] - logits[left] || left - right);
  return order;
};

/**
 * {@link selectTopK} を使う `k` の上限。押し出しが k² 級になるので、これを超える k は全体ソート
 * （O(V log V)）へ倒す。`topK` に上限を設けない（ADR 0083 決定 7）以上、大きな k でも
 * 語彙サイズに対して線形近くに留まる必要がある。
 */
const SELECTION_LIMIT = 1024;

/**
 * 加工後 logits → 抽選対象（正規化済み。並びの契約は {@link SamplerDistribution}）。
 *
 * 温度 0 は argmax の 1 点分布へ縮退する（`greedy.ts` の parity 門と同じ token を出す面）。
 *
 * NOTE: `topP` を `topK` 無しで指定すると語彙全体のソート（O(V log V)）が要る。配布形が宣言する
 * 推奨値は top_k を伴う（gemma-4-E2B-it = top_k 64 / top_p 0.95）ので製品経路はこれを踏まない。
 */
export const samplerDistribution = (
  logits: Float32Array<ArrayBuffer>,
  spec: SamplerSpec,
  history: readonly number[],
): SamplerDistribution => {
  if (logits.length < 1) throw new Error("logits が空");
  assertSpec(spec);
  // 加工の**後**で見る（生の logits だけでなく、`+Infinity` の席へ `-Infinity` の bias を
  // 足したような組み合わせもここで捕まる）。
  const processed = processLogits(logits, spec, history);
  assertNoNaN(processed);
  const temperature = spec.temperature ?? 0;
  if (temperature === 0) {
    return {
      tokens: Int32Array.of(argmax(processed)),
      probabilities: Float64Array.of(1),
    };
  }

  // 語彙数を超える topK は語彙数へ丸める（HF の `min(top_k, vocab)`）。
  const topK = spec.topK === undefined ? undefined : Math.min(spec.topK, processed.length);
  const topP = spec.topP;
  let candidates: readonly number[];
  if (topK !== undefined && topK <= SELECTION_LIMIT) candidates = selectTopK(processed, topK);
  else if (topK !== undefined) candidates = sortAllDescending(processed).slice(0, topK);
  else if (topP !== undefined && topP < 1) candidates = sortAllDescending(processed);
  else candidates = Array.from({ length: processed.length }, (_unused, token) => token);

  // softmax は候補集合の中だけで正規化する（top-k で落とした質量は分母から消える — HF が
  // 落とした席を −inf にして `softmax` へ渡すのと同値）。
  let peak = Number.NEGATIVE_INFINITY;
  for (const token of candidates) {
    if (processed[token] > peak) peak = processed[token];
  }
  if (!Number.isFinite(peak)) {
    throw new Error(`候補の最大 logit が非有限（${peak}） — 全 token が禁止されている`);
  }
  const weights = new Float64Array(candidates.length);
  let total = 0;
  for (let index = 0; index < candidates.length; index += 1) {
    const weight = Math.exp(processed[candidates[index]] - peak);
    weights[index] = weight;
    total += weight;
  }

  // ⑤top-p: 累積が topP に**達した 1 件を含めて**残す（HF `TopPLogitsWarper` の昇順 cumsum と
  // 同値の降順形）。最上位 1 件は必ず残る。`topP = 1` は丸め次第で末尾を落とすので絞らない。
  let kept = candidates.length;
  if (topP !== undefined && topP < 1) {
    let cumulative = 0;
    for (let index = 0; index < candidates.length; index += 1) {
      cumulative += weights[index] / total;
      if (cumulative >= topP) {
        kept = index + 1;
        break;
      }
    }
  }

  const tokens = new Int32Array(kept);
  const probabilities = new Float64Array(kept);
  let keptTotal = 0;
  for (let index = 0; index < kept; index += 1) keptTotal += weights[index];
  for (let index = 0; index < kept; index += 1) {
    tokens[index] = candidates[index];
    probabilities[index] = weights[index] / keptTotal;
  }
  return { tokens, probabilities };
};

/**
 * 抽選器を組む（RNG 状態を持つので、1 本の生成では**同じ実体を使い回す**）。
 *
 * 指定を省略すると greedy（温度 0）— この層が既定を持たないことの表現である（ADR 0083 決定 7）。
 */
export const createSampler = (spec: SamplerSpec = {}): Sampler => {
  assertSpec(spec);
  // seed の受理集合検査もここで済ませる（抽選が走るのは decode の途中なので、生成器を作るのを
  // 遅らせると不正な seed が GB 級のロードの末に落ちる — anima の `assertAcceptableSeed` と同趣旨）。
  const random = new Randu(spec.seed ?? 0);
  return {
    next(logits: Float32Array<ArrayBuffer>, history: readonly number[]): number {
      const { tokens, probabilities } = samplerDistribution(logits, spec, history);
      // 候補 1 件（温度 0 / topK 1 / 語彙 1）は抽選が自明なので RNG を回さない。
      if (tokens.length === 1) return tokens[0];
      const draw = random.uniform();
      let cumulative = 0;
      for (let index = 0; index < tokens.length; index += 1) {
        cumulative += probabilities[index];
        if (draw < cumulative) return tokens[index];
      }
      // 累積は丸めで 1 を僅かに下回りうる（`draw` は [0,1) なので、そのときだけここへ来る）。
      return tokens[tokens.length - 1];
    },
  };
};

/**
 * 停止 token 集合の判定（ADR 0083 決定 8）。
 *
 * 停止条件は単数の EOS ではなく**集合**である（gemma-4-E2B-it の `generation_config.json` は
 * `eos_token_id = [1, 106, 50]`）。判定するのは **sampling の結果**で、argmax でも抽選でも
 * 同じ段に置く — logits の側で見ると「禁止された EOS が停止に化ける」類の食い違いが出る。
 *
 * MUST: 集合の出どころは chat 形式と**同じ配布 digest set**（ADR 0084 決定 5）。別々の場所から
 * 拾うと片方だけ古くなる。
 */
export const isStopToken = (token: number, stopTokens: readonly number[]): boolean =>
  stopTokens.includes(token);
