// ホスト側 sampling（`src/generation/sampler.ts`）と EOS 集合の停止判定の挙動テスト。
// GPU も実資産も要らない。
//
// ここで縛るのは「logits 1 本 → token 1 個」の**加工の順序と境界**— 温度 0 の greedy 縮退・
// top-k / top-p の残す集合・repetition penalty の符号規約・logit bias・seed の決定性。実 GPU で
// 回すと「文章が少し変」という形でしか出ず（sampling は例外を出さない）、境界を 1 個間違えても
// 検収 e2e は温度 0 経路しか踏まないので**永久に緑のまま**になる位置にある。
//
// MUST: HF の sampling 出力との token 列 parity は取れない（RNG が splitmix64 で torch の
// Philox と別物 — ADR 0083 決定 7）。したがって期待値はすべて**この門が手で書き下した理論値**で、
// 実装から採った fixture は 1 つも無い。分布そのものは χ² 門（下段）で縛る。

import { assert, assertAlmostEquals, assertEquals, assertThrows } from "@std/assert";
import { ModelInputError } from "../src/errors.ts";
import {
  createSampler,
  isStopToken,
  samplerDistribution,
  type SamplerSpec,
} from "../src/generation/sampler.ts";

const f32 = (values: readonly number[]): Float32Array<ArrayBuffer> => {
  const data = new Float32Array(values.length);
  data.set(values);
  return data;
};

/**
 * 期待値側の softmax（**実装と別の形**で書く — 実装は最大値を引いてから exp を取るので、
 * 素の式で書いておくと「同じ関数を両側から呼んで一致を見る」恒真化を避けられる）。
 */
const softmax = (logits: readonly number[]): number[] => {
  const weights = logits.map((logit) => Math.exp(logit));
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  return weights.map((weight) => weight / total);
};

const assertProbabilities = (
  actual: Float64Array<ArrayBuffer>,
  expected: readonly number[],
  where: string,
): void => {
  assertEquals(actual.length, expected.length, `${where}: 候補数`);
  expected.forEach((want, index) => {
    assertAlmostEquals(actual[index], want, 1e-6, `${where}: 確率[${index}]`);
  });
};

/** 分布門の対象（`logits = ln(p)` なので温度 1 の softmax がそのまま `p` に戻る）。 */
const BASE_PROBABILITIES = [0.5, 0.3, 0.15, 0.05] as const;
const BASE_LOGITS = f32(BASE_PROBABILITIES.map((probability) => Math.log(probability)));

// ---------------------------------------------------------------------------
// 温度 0 = greedy 縮退（ADR 0083 決定 7 — parity 門が生き続ける経路）
// ---------------------------------------------------------------------------

Deno.test("sampler: 既定（指定なし）は温度 0 の greedy で argmax と厳密一致", () => {
  const logits = f32([-1, 4.5, 0, 4.25, -3]);
  assertEquals(createSampler().next(logits, []), 1);
  assertEquals(createSampler({}).next(logits, []), 1);
  const distribution = samplerDistribution(logits, {}, []);
  assertEquals([...distribution.tokens], [1], "温度 0 は 1 点分布へ縮退する");
  assertEquals([...distribution.probabilities], [1]);
});

Deno.test("sampler: 温度 0 の tie-break は先に出た方（torch の argmax と同じ）", () => {
  // 同値が並ぶ列で `>=` 更新の実装（後勝ち）を落とす。
  assertEquals(createSampler().next(f32([1, 3, 3, 2]), []), 1);
  assertEquals(createSampler().next(f32([3, 3, 3]), []), 0);
});

Deno.test("sampler: 温度 0 は seed に依らず同じ列（抽選が走らない）", () => {
  const logits = f32([0.1, 0.9, 0.4]);
  const first = createSampler({ seed: 1 });
  const second = createSampler({ seed: 999 });
  for (let step = 0; step < 8; step += 1) {
    assertEquals(first.next(logits, []), second.next(logits, []), `step ${step}`);
  }
});

Deno.test("sampler: NaN は最大値でなくても落ちる（比較が全部 false になる値を通さない）", () => {
  // NaN はあらゆる比較が false なので、素通しすると①argmax が別 token を返し、②top-k の閾値が
  // 一度 NaN になった後の候補を全部弾き、③softmax の分母が NaN になる。どれも例外を出さない。
  const poisoned = f32([1, NaN, 3, 2]);
  assertThrows(() => createSampler().next(poisoned, []), Error, "NaN");
  assertThrows(
    () => samplerDistribution(poisoned, { temperature: 1, topK: 2 }, []),
    Error,
    "NaN",
  );
  assertThrows(
    () => samplerDistribution(poisoned, { temperature: 1, topP: 0.9 }, []),
    Error,
    "NaN",
  );
});

Deno.test("sampler: greedy は範囲内の最初の NaN を最大値より優先して拒否する", () => {
  for (const values of [[NaN, 3, 2], [Infinity, 3, NaN], [NaN, NaN, Infinity]]) {
    // 外側の NaN は view の対象外。最大値を見つけても末尾まで検査する。
    const backing = f32([NaN, ...values, NaN]);
    const logits = backing.subarray(1, backing.length - 1);
    const first = values.findIndex(Number.isNaN);
    const error = assertThrows(() => createSampler().next(logits, []), Error);
    assertEquals(
      error.message,
      `logits[${first}] が NaN（非有限） — token id へ畳まずここで落とす`,
    );
  }
  assertEquals(createSampler().next(f32([-Infinity, -0, 0, -Infinity]), []), 1);
  assertThrows(() => createSampler().next(f32([0, Infinity, -1]), []), Error, "最大値が非有限");
});

Deno.test("sampler: topK は候補の範囲外でも最初の NaN を拒否する", () => {
  for (const topK of [1, 2, 4, 8]) {
    for (const values of [[NaN, 3, 2, 1], [3, NaN, 2, NaN], [Infinity, 3, 2, NaN]]) {
      const backing = f32([NaN, ...values, NaN]);
      const logits = backing.subarray(1, backing.length - 1);
      const first = values.findIndex(Number.isNaN);
      const error = assertThrows(
        () => samplerDistribution(logits, { temperature: 1, topK }, []),
        Error,
      );
      assertEquals(
        error.message,
        `logits[${first}] が NaN（非有限） — token id へ畳まずここで落とす`,
      );
    }
  }
  // 加工で生じた NaN も、上位候補から外れる位置にあっても拒否する。
  assertThrows(
    () =>
      samplerDistribution(f32([3, 2, Infinity]), {
        temperature: 1,
        topK: 1,
        logitBias: [[2, -Infinity]],
      }, []),
    Error,
    "logits[2] が NaN",
  );
});

Deno.test("sampler: 最大 logit が非有限なら token id へ畳まずに落ちる", () => {
  // 位置表の外の gather は行ごと NaN 汚染する（known-issues）ので、最終行 logits は**丸ごと**
  // 非有限になる。argmax はそれを「もっともらしい token id」（NaN 比較が常に false なので
  // 先頭 = token 0）へ畳むため、畳まれる前がここしかない。
  assertThrows(() => createSampler().next(f32([NaN, NaN, NaN]), []), Error, "非有限");
  assertThrows(() => createSampler().next(f32([-Infinity, -Infinity]), []), Error, "非有限");
  // 温度 > 0 側も同じ段で落ちる（softmax が全 0 になって抽選が壊れる前）。
  assertThrows(
    () => createSampler({ temperature: 1 }).next(f32([NaN, NaN]), []),
    Error,
    "非有限",
  );
});

// ---------------------------------------------------------------------------
// top-k（上限なし — 最終行 logits 出口だから。ADR 0083 決定 6 の裏返し）
// ---------------------------------------------------------------------------

Deno.test("sampler: topK = 1 は温度に依らず greedy と一致する（候補 1 件で抽選が自明）", () => {
  const spec = { temperature: 1, topK: 1, seed: 3 } satisfies SamplerSpec;
  const distribution = samplerDistribution(BASE_LOGITS, spec, []);
  assertEquals([...distribution.tokens], [0]);
  assertEquals([...distribution.probabilities], [1]);
  // seed を変えても動かない（絞り込みが最上位 1 件へ縮退している）。
  const first = createSampler(spec);
  const second = createSampler({ ...spec, seed: 4242 });
  for (let step = 0; step < 16; step += 1) {
    assertEquals(first.next(BASE_LOGITS, []), 0, `step ${step}`);
    assertEquals(second.next(BASE_LOGITS, []), 0, `step ${step}`);
  }
});

Deno.test("sampler: topK は上位 k 件だけを残し、確率をその中で正規化する", () => {
  const distribution = samplerDistribution(BASE_LOGITS, { temperature: 1, topK: 2 }, []);
  assertEquals([...distribution.tokens], [0, 1], "上位 2 件（確率降順）");
  // 落とした 0.15 + 0.05 の質量は分母から消える。
  assertProbabilities(distribution.probabilities, [0.5 / 0.8, 0.3 / 0.8], "topK 2");
});

Deno.test("sampler: topK の同値は token id の小さい方を残す", () => {
  const distribution = samplerDistribution(f32([2, 2, 1]), { temperature: 1, topK: 1 }, []);
  assertEquals([...distribution.tokens], [0]);
});

Deno.test("sampler: topK が語彙数以上なら全件（HF の min(top_k, vocab)）", () => {
  for (const topK of [4, 5, 1_000_000]) {
    const distribution = samplerDistribution(BASE_LOGITS, { temperature: 1, topK }, []);
    assertEquals([...distribution.tokens], [0, 1, 2, 3], `topK ${topK}`);
    assertProbabilities(distribution.probabilities, [...BASE_PROBABILITIES], `topK ${topK}`);
  }
});

/**
 * 参照の上位 k 件 — **契約の言葉どおり**「全件を値の降順（同値は token id の昇順）に並べて
 * 先頭 k」。実装（有界 heap）とは別の形で書く（全件ソートなので恒真化しない）。
 */
const referenceTopK = (logits: Float32Array<ArrayBuffer>, k: number): number[] =>
  Array.from({ length: logits.length }, (_unused, token) => token)
    .sort((left, right) => {
      if (logits[left] === logits[right]) return left - right;
      return logits[left] > logits[right] ? -1 : 1;
    })
    .slice(0, k);

/**
 * 選択の形を変える 3 分布（語彙 8,192）。**昇順**は「値が改善し続ける」最悪形で、**同値**は
 * 足切りが効かないと候補の入れ替えが毎回起きる形 — どちらも上位 k 件の中身より先に並びが
 * 壊れる入力である。
 */
const SELECTION_LOGITS = (() => {
  const size = 8_192;
  let state = 20_260_907;
  const random = (): number => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
  const build = (value: (token: number) => number): Float32Array<ArrayBuffer> => {
    const logits = new Float32Array(size);
    for (let token = 0; token < size; token += 1) logits[token] = value(token);
    return logits;
  };
  return {
    uniform: build(() => random() * 16 - 8),
    ascending: build((token) => token / size),
    ties: build(() => Math.floor(random() * 8)),
  };
})();

Deno.test("sampler: topK はどの k でも参照（全件を並べた先頭 k）と中身も並びも一致する", () => {
  // 選び方が k で切り替わる形（k の境界の前後で並びや tie-break が変わりうる）への逆戻りを縛る
  // ため、離れた k と隣接する 2 値（1024 / 1025）を並べて同じ参照と突き合わせる。
  for (const [shape, logits] of Object.entries(SELECTION_LOGITS)) {
    for (const topK of [64, 1024, 1025, 4096]) {
      const distribution = samplerDistribution(logits, { temperature: 1, topK }, []);
      assertEquals(
        [...distribution.tokens],
        referenceTopK(logits, topK),
        `${shape} / topK ${topK}`,
      );
    }
  }
});

Deno.test("sampler: 大きな topK の確率は参照の上位 k 件だけで正規化される", () => {
  const logits = SELECTION_LOGITS.uniform;
  const topK = 1025;
  const distribution = samplerDistribution(logits, { temperature: 1, topK }, []);
  const expected = softmax(referenceTopK(logits, topK).map((token) => logits[token]));
  assertProbabilities(distribution.probabilities, expected, `topK ${topK}`);
});

// ---------------------------------------------------------------------------
// top-p（nucleus — 累積が p に達する最小の集合・最上位 1 件は必ず残す）
// ---------------------------------------------------------------------------

Deno.test("sampler: topP = 1 は絞らない", () => {
  const distribution = samplerDistribution(BASE_LOGITS, { temperature: 1, topP: 1 }, []);
  assertEquals([...distribution.tokens], [0, 1, 2, 3]);
  assertProbabilities(distribution.probabilities, [...BASE_PROBABILITIES], "topP 1");
});

Deno.test("sampler: topP は累積が p に達した 1 件までを残す", () => {
  // 累積は 0.5 / 0.8 / 0.95 / 1.0。境界の読み方（「達した件を含む」）をここで固定する。
  // NOTE: 累積とちょうど等しい p（ここでは 0.5 / 0.8 / 0.95）は f32 の丸めで両側に振れるので
  // 門にしない — 固定するのは「達した件を含む」の読み方であって、境界での丸めではない。
  const cases = [
    { topP: 0.4, tokens: [0] },
    { topP: 0.75, tokens: [0, 1] },
    { topP: 0.85, tokens: [0, 1, 2] },
    { topP: 0.99, tokens: [0, 1, 2, 3] },
  ] as const;
  for (const { topP, tokens } of cases) {
    const distribution = samplerDistribution(BASE_LOGITS, { temperature: 1, topP }, []);
    assertEquals([...distribution.tokens], [...tokens], `topP ${topP}`);
    const mass = tokens.reduce<number>((sum, token) => sum + BASE_PROBABILITIES[token], 0);
    assertProbabilities(
      distribution.probabilities,
      tokens.map((token) => BASE_PROBABILITIES[token] / mass),
      `topP ${topP}`,
    );
  }
});

Deno.test("sampler: topP は topK の後に効く（残った集合の中で累積を取る）", () => {
  // topK 3 で {0.5, 0.3, 0.15} が残り、その中の正規化確率は 0.5263 / 0.3158 / 0.1579。
  // 最上位の累積 0.5263 は 0.52 に達するので 1 件だけ残る — 語彙全体（0.5 < 0.52）で累積を
  // 取る実装なら 2 件残って落ちる。
  const distribution = samplerDistribution(
    BASE_LOGITS,
    { temperature: 1, topK: 3, topP: 0.52 },
    [],
  );
  assertEquals([...distribution.tokens], [0]);
});

// ---------------------------------------------------------------------------
// repetition penalty（HF `RepetitionPenaltyLogitsProcessor` の符号規約）
// ---------------------------------------------------------------------------

Deno.test("sampler: repetition penalty は負 logit に掛け・非負 logit で割る", () => {
  const logits = f32([2, -2, 0, 1]);
  const distribution = samplerDistribution(
    logits,
    { temperature: 1, repetitionPenalty: 2 },
    [0, 1, 2],
  );
  // 2 → 2/2 = 1 / −2 → −2×2 = −4 / 0 → 0/2 = 0（非負側）/ 履歴外の 1 は素通し。
  assertEquals([...distribution.tokens], [0, 1, 2, 3]);
  assertProbabilities(distribution.probabilities, softmax([1, -4, 0, 1]), "penalty 2");
});

Deno.test("sampler: repetition penalty < 1 は逆向きに効く（既出を持ち上げる）", () => {
  const distribution = samplerDistribution(
    f32([2, -2, 0]),
    { temperature: 1, repetitionPenalty: 0.5 },
    [0, 1],
  );
  assertProbabilities(distribution.probabilities, softmax([4, -1, 0]), "penalty 0.5");
});

Deno.test("sampler: repetition penalty は履歴に何度出ても 1 回だけ効く", () => {
  const once = samplerDistribution(f32([2, 0]), { temperature: 1, repetitionPenalty: 2 }, [0]);
  const many = samplerDistribution(
    f32([2, 0]),
    { temperature: 1, repetitionPenalty: 2 },
    [0, 0, 0, 0],
  );
  assertProbabilities(many.probabilities, [...once.probabilities], "重複した履歴");
  assertProbabilities(once.probabilities, softmax([1, 0]), "1 回ぶんの適用");
});

Deno.test("sampler: repetitionPenalty = 1（既定）は履歴を読まない", () => {
  const plain = samplerDistribution(BASE_LOGITS, { temperature: 1 }, []);
  const withHistory = samplerDistribution(BASE_LOGITS, { temperature: 1 }, [0, 1, 2, 3]);
  assertProbabilities(withHistory.probabilities, [...plain.probabilities], "履歴あり");
});

Deno.test("sampler: 履歴の token id が語彙の外なら fail loudly", () => {
  assertThrows(
    () => samplerDistribution(f32([1, 2]), { temperature: 1, repetitionPenalty: 2 }, [5]),
    RangeError,
    "語彙",
  );
});

// ---------------------------------------------------------------------------
// logit bias
// ---------------------------------------------------------------------------

Deno.test("sampler: logit bias は温度の前に加算される", () => {
  const logits = f32([2, 1, 0]);
  const distribution = samplerDistribution(
    logits,
    { temperature: 2, logitBias: [[2, 6]] },
    [],
  );
  // 加算 → 除算の順（[2,1,6] / 2 = [1,0.5,3]）。温度の後に足す実装なら [1,0.5,3] にならない。
  assertProbabilities(distribution.probabilities, softmax([1, 0.5, 3]), "bias then temperature");
  assertEquals(
    createSampler({ logitBias: [[2, 6]] }).next(logits, []),
    2,
    "温度 0 でも効く",
  );
});

Deno.test("sampler: logit bias の -Infinity はその token を候補から外す", () => {
  const logits = f32([5, 1, 2]);
  const spec = { logitBias: [[0, Number.NEGATIVE_INFINITY]] } satisfies SamplerSpec;
  assertEquals(createSampler(spec).next(logits, []), 2, "温度 0");
  const distribution = samplerDistribution(logits, { ...spec, temperature: 1 }, []);
  assertEquals([...distribution.tokens], [0, 1, 2]);
  assertEquals(distribution.probabilities[0], 0, "禁止した token の確率");
});

Deno.test("sampler: 全 token を禁止したら fail loudly（黙って 0 を返さない）", () => {
  const banned: SamplerSpec["logitBias"] = [[0, -Infinity], [1, -Infinity]];
  assertThrows(
    () => samplerDistribution(f32([1, 2]), { temperature: 1, logitBias: banned }, []),
    Error,
    "非有限",
  );
});

Deno.test("sampler: logit bias の token が語彙の外なら fail loudly", () => {
  assertThrows(
    () => createSampler({ logitBias: [[9, 1]] }).next(f32([1, 2]), []),
    ModelInputError,
    "語彙",
  );
});

Deno.test("sampler: 同じ token の logit bias を 2 度書いたら fail loudly（後勝ちで畳まない）", () => {
  // `Map` なら黙って後勝ちで畳んでいた形（`[[2,6],[2,-1]]`）— 配列では「2 つ書いた bias の
  // 片方だけが効く」ことになるので落とす。値が同じでも落とす（畳み方の疑義は同じ）。
  assertThrows(
    () => createSampler({ logitBias: [[2, 6], [2, -1]] }),
    ModelInputError,
    "token 2 が 2 度出る",
  );
  assertThrows(
    () => samplerDistribution(f32([1, 2, 3]), { temperature: 1, logitBias: [[0, 1], [0, 1]] }, []),
    ModelInputError,
    "token 0 が 2 度出る",
  );
});

Deno.test("sampler: 発行後に logitBias 配列を書き換えても抽選は変わらない（防御コピー）", () => {
  const logits = f32([5, 1, 2]);
  // 呼び手が握ったままの可変配列（外側も内側のタプルも書き換えられる形）で発行する。
  const bias: [number, number][] = [[0, Number.NEGATIVE_INFINITY]];
  const sampler = createSampler({ logitBias: bias });
  assertEquals(sampler.next(logits, []), 2, "発行時の指定（token 0 を禁止）");

  bias[0][1] = 0; // 禁止を解く
  assertEquals(sampler.next(logits, []), 2, "タプルの書き換えは効かない");
  bias.push([2, Number.NEGATIVE_INFINITY]); // 勝った token を禁止する
  assertEquals(sampler.next(logits, []), 2, "配列への追加も効かない");
  bias.length = 0;
  assertEquals(sampler.next(logits, []), 2, "空にしても効かない");
});

// ---------------------------------------------------------------------------
// 指定の受理集合（黙って丸めない）
// ---------------------------------------------------------------------------

Deno.test("sampler: 受理できない指定は構築時に落ちる", () => {
  const rejected: readonly SamplerSpec[] = [
    { temperature: -1 },
    { temperature: Number.NaN },
    { topK: 0 },
    { topK: 1.5 },
    { topP: 0 },
    { topP: 1.5 },
    { repetitionPenalty: 0 },
    { repetitionPenalty: -1 },
    { seed: -1 },
    { seed: 0.5 },
    { logitBias: [[-1, 1]] },
    { logitBias: [[0, Number.POSITIVE_INFINITY]] },
    { logitBias: [[0, Number.NaN]] },
    { logitBias: [[0, 1], [0, 2]] },
  ];
  for (const spec of rejected) {
    assertThrows(
      () => createSampler(spec),
      ModelInputError,
      undefined,
      JSON.stringify(Object.keys(spec)),
    );
  }
});

// ---------------------------------------------------------------------------
// 失敗の分類（ADR 0107 — 呼び手の分岐先で型を割る）
// ---------------------------------------------------------------------------

Deno.test("sampler: 失敗の分類は呼び手の分岐先で決まる", async (t) => {
  await t.step("指定の値域は入力起因（呼び手が値を直せば通る）", () => {
    assertThrows(() => createSampler({ topP: 0 }), ModelInputError, "topP 0");
    // seed の受理集合の所有者は `request-gates.ts` の 1 本（ADR 0107 決定 6）。ここが
    // `ModelInputError` で落ちること自体が、`Randu` が自前の条件を持っていないことの対。
    assertThrows(() => createSampler({ seed: -1 }), ModelInputError, "seed -1");
  });

  await t.step("語彙の外の logitBias も入力起因（語彙が判るのが最初の抽選なだけ）", () => {
    assertThrows(
      () => createSampler({ logitBias: [[9, 1]] }).next(f32([1, 2]), []),
      ModelInputError,
      "logitBias の token 9 が語彙 0..1 の外",
    );
  });

  await t.step("logits の NaN は入力起因ではない（呼び手が要求を直しても直らない）", () => {
    // 重みか配線の破れで、HTTP サーバーなら 500。同じ型で捕まると「入力を直せ」と返す形になる。
    const error = assertThrows(
      () => createSampler().next(f32([Number.NaN, 1]), []),
      Error,
      "NaN",
    );
    assert(!(error instanceof ModelInputError), "logits の NaN が入力起因に分類されている");
  });

  await t.step("履歴の token id が語彙の外も入力起因ではない（唯一の呼び元は検査済み）", () => {
    // `createSampler` は公開面に出ておらず、`history` を渡すのは `GenerationSequence.generate`
    // だけ（検査済み prompt + 自分が選んだ token）。ここが落ちるのは配線の破れである。
    const error = assertThrows(
      () => samplerDistribution(f32([1, 2]), { temperature: 1, repetitionPenalty: 2 }, [5]),
      RangeError,
      "語彙",
    );
    assert(!(error instanceof ModelInputError), "履歴の語彙外が入力起因に分類されている");
  });
});

// ---------------------------------------------------------------------------
// seed の決定性と分布門
//
// 分布門の設計: 語彙 4 の既知分布 p =(0.5, 0.3, 0.15, 0.05) から N = 20,000 回引き、Pearson の
// χ² 統計量（自由度 3）を**理論値**（この門が手で書いた p / p²）と突き合わせる。閾値
// 21.108 は片側 α = 1e-4 の分位点で、門は 3 本（温度 1 × seed 2 本 + 温度 0.5）なので、
// 実装が正しいのに落ちる確率は union bound で ≤ 3e-4。seed は固定なので、いったん緑なら
// 以後は決定的に緑（「たまたま落ちる CI」にはならない）。
//
// 恒真でないことの根拠: 一様に引く実装なら χ² ≈ 20,000、温度を無視する（p のまま引く）実装なら
// 温度 0.5 の門で χ² ≈ 9,200 になり、どちらも閾値の 2〜3 桁上を通る。
// ---------------------------------------------------------------------------

/** 自由度 3・片側 α = 1e-4 の χ² 分位点。 */
const CHI_SQUARE_DF3_1E4 = 21.108;
const DRAWS = 20_000;

const drawCounts = (spec: SamplerSpec, draws: number): number[] => {
  const sampler = createSampler(spec);
  const counts = [0, 0, 0, 0];
  for (let draw = 0; draw < draws; draw += 1) counts[sampler.next(BASE_LOGITS, [])] += 1;
  return counts;
};

const chiSquare = (counts: readonly number[], probabilities: readonly number[]): number => {
  const total = counts.reduce((sum, count) => sum + count, 0);
  return counts.reduce((statistic, observed, index) => {
    const expected = probabilities[index] * total;
    return statistic + ((observed - expected) ** 2) / expected;
  }, 0);
};

Deno.test("sampler: 温度 1 の抽選が理論分布に従う（χ² 門・自由度 3・α = 1e-4）", () => {
  for (const seed of [20_260_831, 7]) {
    const counts = drawCounts({ temperature: 1, seed }, DRAWS);
    assertEquals(counts.reduce((sum, count) => sum + count, 0), DRAWS, `seed ${seed}: 総数`);
    const statistic = chiSquare(counts, BASE_PROBABILITIES);
    assert(
      statistic < CHI_SQUARE_DF3_1E4,
      `seed ${seed}: χ² = ${statistic.toFixed(3)} が ${CHI_SQUARE_DF3_1E4} 以上` +
        `（観測 ${counts.join("/")} / 期待 ${BASE_PROBABILITIES.map((p) => p * DRAWS).join("/")}）`,
    );
  }
});

Deno.test("sampler: 温度が分布を鋭くする（温度 0.5 の理論分布と一致）", () => {
  // 温度 T の理論分布は p^(1/T) を正規化したもの。T = 0.5 なら p² / Σp²。
  const squared = BASE_PROBABILITIES.map((probability) => probability ** 2);
  const total = squared.reduce((sum, weight) => sum + weight, 0);
  const expected = squared.map((weight) => weight / total);
  const counts = drawCounts({ temperature: 0.5, seed: 11 }, DRAWS);
  const statistic = chiSquare(counts, expected);
  assert(
    statistic < CHI_SQUARE_DF3_1E4,
    `χ² = ${statistic.toFixed(3)} が ${CHI_SQUARE_DF3_1E4} 以上` +
      `（観測 ${counts.join("/")} / 期待 ${
        expected.map((p) => (p * DRAWS).toFixed(1)).join("/")
      }）`,
  );
});

Deno.test("sampler: 同じ seed は同じ列・違う seed は違う列", () => {
  const draw = (seed: number): number[] => {
    const sampler = createSampler({ temperature: 1, seed });
    return Array.from({ length: 64 }, () => sampler.next(BASE_LOGITS, []));
  };
  assertEquals(draw(5), draw(5), "同じ seed");
  assert(draw(5).join() !== draw(6).join(), "違う seed で同じ列が出た");
});

Deno.test("sampler: topK / topP で落とした token は一度も引かれない", () => {
  // 分布門の対（頻度ではなく**集合**を見る）。境界の実装が緩むと必ず落ちる。
  const sampler = createSampler({ temperature: 1, topK: 2, seed: 31 });
  const seen = new Set<number>();
  for (let draw = 0; draw < 5_000; draw += 1) seen.add(sampler.next(BASE_LOGITS, []));
  assertEquals([...seen].sort(), [0, 1], "topK 2 で観測された token");
});

// ---------------------------------------------------------------------------
// 停止判定（ADR 0083 決定 8 — EOS は単数ではなく集合）
// ---------------------------------------------------------------------------

Deno.test("isStopToken: 集合のどれでも止まり、集合外では止まらない", () => {
  // 実資産の集合（gemma-4-E2B-it の `generation_config.json` は eos_token_id = [1, 106, 50]）。
  const stopTokens = [1, 106, 50] as const;
  for (const token of stopTokens) {
    assert(isStopToken(token, stopTokens), `token ${token} で止まらない`);
  }
  for (const token of [0, 2, 49, 51, 105, 107, 262_143]) {
    assert(!isStopToken(token, stopTokens), `token ${token} で止まった`);
  }
});

Deno.test("isStopToken: 空の集合では止まらない（EOS を宣言しない資産）", () => {
  for (const token of [0, 1, 106]) assert(!isStopToken(token, []));
});
