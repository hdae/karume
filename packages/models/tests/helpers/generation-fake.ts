/**
 * 生成面（`src/generation/sequence.ts`）の GPU 無しテストが共有する fake（静的配線・Session・
 * context）。非投機の契約（`generation_sequence_test.ts`）と投機経路の契約
 * （`generation_speculation_test.ts`）が**同じ実体**を差すための置き場である。
 *
 * 生成ループは narrow interface（`GenerationSession` / `GenerationContextFace`）で受けるので、
 * fake は素の object 1 個で足りる。ここが持つのは 4 つ:
 *
 * - **論理長の進行**: `run` の成功で `pastLength` が進む。`commit: "deferred"` の run は進めずに
 *   保留を立て、`commit(rows)` が受理行数ぶんだけ進める（実 `GenerationContext` と同じ順序）。
 *   保留がある間の run と、保留の無い `commit`、sliding ring の余裕を超える deferred run は、
 *   実 context / 実 executor と同じく fail loudly（門の文言まで写す）。
 * - **行ごとの logits**: {@link FakeOptions.successor} を渡すと「その行の入力 token → その行の
 *   argmax」の写像になる（= 直前 token だけで決まる決定的な target のモデル）。verify の複数行を
 *   1 本の後続関数で表せるので、投機の受理・棄却を配列 1 本で組める。
 * - **行ごとの hidden**: 行ごとに違う値（{@link hiddenMark}）を書く。生成ループが「どの行の
 *   hidden を写したか」がここでしか観測できない。
 * - **位置列の記録**: 位置は run の入力ではなく派生入力の席（`derivedInputs`）へ降りるので、
 *   「どの run にどの位置が渡ったか」を見られる唯一の場所がこの fake である。
 */

import type { GenerationContextSpec, RunInputs, RunOutputs, SymbolBindings } from "@karume/runtime";
import {
  createGenerationProgram,
  type DerivedRunInputs,
  type GenerationGraph,
  type GenerationProgramSpec,
  type GenerationWiring,
} from "../../src/generation/program.ts";
import type {
  GenerationEvent,
  GenerationSession,
  GenerationStop,
  GenerationStream,
} from "../../src/generation/sequence.ts";

export const VOCAB = 16;
export const HIDDEN_SIZE = 4;
export const IDS = "input_ids";
export const LAST_ROW = "last_row";
export const LOGITS = "logits";
export const HIDDEN = "hidden";
export const DERIVED = "per_layer_inputs";
export const CHUNK_LENGTH = 4;

/** sliding ring の余裕の既定（門に掛からない十分大きい値 — 小さくするのは門のテストだけ）。 */
export const SLIDING_SLACK = 64;

export const graphOf = (): GenerationGraph => ({
  symbols: ["C", "M", "R"],
  inputs: [
    { name: IDS, dtype: "i32", shape: [1, "M"] },
    { name: DERIVED, dtype: "f32", shape: [1, "M", 2] },
    // 行選択は添字**列**（`R` = その run で選ぶ行数）。非投機は常に 1 本・verify は k+1 本。
    { name: LAST_ROW, dtype: "i32", shape: ["R"] },
  ],
  outputs: [LOGITS, HIDDEN],
  values: {
    [LOGITS]: { dtype: "f32", shape: [1, "R", VOCAB] },
    [HIDDEN]: { dtype: "f32", shape: [1, "R", HIDDEN_SIZE] },
  },
});

/**
 * 静的配線。**派生入力の席は fake が持つ**（既定は位置列を記録する実装）。
 *
 * 位置は run の入力ではなくなった（`position_ids` はグラフから消え、位置に依存するホスト入力は
 * 派生入力の席が受ける）ので、「どの run にどの位置が渡ったか」を見られる唯一の場所がここである。
 */
export const programOf = (
  fake: FakeSession,
  override: Partial<GenerationProgramSpec> = {},
): GenerationWiring =>
  createGenerationProgram({
    graph: graphOf(),
    inputIds: IDS,
    lastRow: LAST_ROW,
    logits: LOGITS,
    hidden: HIDDEN,
    chunkLength: CHUNK_LENGTH,
    maxPosition: 128,
    capacity: 64,
    vocabSize: VOCAB,
    stopTokens: [],
    capacitySymbol: "C",
    derivedInputs: fake.derivedInputs,
    ...override,
  });

/** run 1 回ぶんの記録（呼び出し列だけで step の形が全部読める粒度）。 */
export type RunCall = {
  readonly ids: readonly number[];
  readonly idsShape: readonly number[];
  readonly positions: readonly number[];
  readonly lastRow: number;
  /** `last_row` に渡った添字の**全部**（verify は k+1 本 — 末尾は pad されうる）。 */
  readonly lastRows: readonly number[];
  readonly lastRowShape: readonly number[];
  readonly queryLength: number;
  /** `"deferred"` = 論理長を保留する run（投機の verify）・`undefined` = その場で進める run。 */
  readonly commit: "deferred" | undefined;
  readonly pastBefore: number;
  readonly bindings: SymbolBindings | undefined;
  readonly extra: readonly string[];
  readonly sameContext: boolean;
};

const readRow = (
  inputs: RunInputs,
  name: string,
): { readonly shape: readonly number[]; readonly values: readonly number[] } => {
  if (!Object.hasOwn(inputs, name)) throw new Error(`fake: 入力 '${name}' が渡っていない`);
  const tensor = inputs[name];
  if (!("data" in tensor)) throw new Error(`fake: 入力 '${name}' がホストテンソルでない`);
  if (tensor.dtype !== "i32") throw new Error(`fake: 入力 '${name}' が i32 でない`);
  return { shape: tensor.shape, values: [...tensor.data] };
};

/**
 * hidden の行ごとの目印（`run` の番号と**入力行の添字**で決まる — 行の取り違えの検出線）。
 *
 * 生成ループが読むのは「出力の行 n」だが、その中身は `last_row[n]` が指した入力行のものである。
 * 目印を入力行の添字で作ると、出力行と入力行の対応が壊れた実装がそのまま値の違いに出る。
 */
export const hiddenMark = (call: number, inputRow: number): number => call * 100 + inputRow + 1;

export type FakeOptions = {
  /** run ごとに argmax が指すべき token id（call 番号で引く — {@link FakeOptions.successor} が優先）。 */
  readonly tokens?: readonly number[];
  /**
   * 「その行の入力 token → その行の argmax」の写像（長さ `VOCAB`）。
   *
   * 直前 token だけで次が決まる決定的な target のモデルである。verify の行 `j` は token
   * `[b, d₁..][j]` を食う行なので、この写像 1 本で「draft が当たっていれば受理・外れていれば
   * 棄却」が表現でき、投機と非投機が**同じ token 列**を出すこと自体を fake の側で保証しない
   * （実装が行を取り違えれば列が割れる）。
   */
  readonly successor?: readonly number[];
  /** この回数目（0 始まり）の run を失敗させる。 */
  readonly failAt?: number;
  /**
   * この回数目（0 始まり）の run の logits に NaN を混ぜる（**run は成功し、抽選が落ちる**）。
   *
   * {@link FakeOptions.failAt} との違いがこの席の意味である — run は通っているので論理長は進み、
   * 落ちるのは `sampler.next`（`assertNoNaN`）である。範囲外 gather の行ごと NaN 汚染で実際に
   * 起きる形で、「run は成功したが token が選べなかった」ときの後始末の唯一の検出線になる。
   */
  readonly nanAt?: number;
  /**
   * 対抗馬（第 1 候補より低い logit を持つ id）。
   *
   * 既定の logits は「狙った id だけ 10・他は全部 0」なので、正値を割る repetition penalty では
   * 順位が動かず（10/penalty > 0）、温度 0 の argmax では効きが**原理的に観測できない**。
   * 2 番手を置くと、penalty が第 1 候補を 2 番手の下へ落としたかを決定論的に見られる。
   */
  readonly runnerUp?: { readonly id: number; readonly logit: number };
  /**
   * logits / hidden を**この行数**で返す（既定は渡った `last_row` の本数 = 正しい実装）。
   *
   * 故障注入用の席である — 「1 行頼んだのに R 行返る」形はグラフの宣言としては正しい
   * （R は記号）ので、`readLogits` の行数検査だけが検出線になる。
   */
  readonly logitsRows?: number;
  /** {@link FakeOptions.logitsRows} を効かせる run の番号（0 始まり・省略時は全 run）。 */
  readonly logitsRowsAt?: number;
  /**
   * sliding ring の余裕（省略時は {@link SLIDING_SLACK}）。
   *
   * `undefined` を**明示**すると sliding スロットの無い context になる（欄の有無で区別するので、
   * 省略と明示は別物である）。
   */
  readonly slidingSlack?: number | undefined;
  /** dispose の順序を見るための記録先（context は `"context"` を積む）。 */
  readonly disposeLog?: string[];
};

export type FakeSession = ReturnType<typeof fakeSession>;

export const fakeSession = (options: FakeOptions = {}) => {
  const calls: RunCall[] = [];
  const specs: GenerationContextSpec[] = [];
  /** `commit(rows)` に渡った受理行数（cycle ごとに 1 件 — 投機の門の観測口）。 */
  const commits: number[] = [];
  let pastLength = 0;
  let pending: { readonly pastLength: number; readonly queryLength: number } | undefined;
  let disposals = 0;
  /** 直前の `derive` が受けた位置列（run の記録へ合流させる — 位置は run の入力ではない）。 */
  let derivedPositions: readonly number[] = [];
  /** 既定の派生入力の席（`[1,M,2]` の f32 を返しつつ、渡った位置列を記録する）。 */
  const derivedInputs: DerivedRunInputs = {
    names: [DERIVED],
    derive: (ids, positions) => {
      if (ids.length !== positions.length) {
        throw new Error(`fake: ids ${ids.length} と positions ${positions.length} の長さが違う`);
      }
      derivedPositions = [...positions];
      return Promise.resolve(
        {
          [DERIVED]: {
            dtype: "f32",
            shape: [1, ids.length, 2],
            data: new Float32Array(ids.length * 2),
          },
        } satisfies RunInputs,
      );
    },
  };
  const context = {
    get pastLength(): number {
      return pastLength;
    },
    get pendingCommit(): { readonly pastLength: number; readonly queryLength: number } | undefined {
      return pending;
    },
    // 宣言しなければ十分大きい値（門に掛からない）。`undefined` の明示は sliding スロット無し。
    slidingSlack: Object.hasOwn(options, "slidingSlack") ? options.slidingSlack : SLIDING_SLACK,
    /**
     * 保留中の deferred run の先頭 `rows` 行を確定させる（実 `GenerationContext.commit` と
     * 同じ受理集合 — 保留が無い呼びと値域外は fail loudly）。
     */
    commit: (rows: number): void => {
      if (pending === undefined) {
        throw new Error(
          "fake: 確定待ちの generation run が無い（deferred で発行した run だけが commit を要る）",
        );
      }
      if (!Number.isSafeInteger(rows) || rows < 0 || rows > pending.queryLength) {
        throw new Error(`fake: 受理行数 ${rows} が 0..${pending.queryLength} の外`);
      }
      pastLength = pending.pastLength + rows;
      pending = undefined;
      commits.push(rows);
    },
    dispose: (): Promise<void> => {
      disposals += 1;
      options.disposeLog?.push("context");
      return Promise.resolve();
    },
  };
  const session: GenerationSession<typeof context> = {
    createGenerationContext: (spec) => {
      specs.push(spec);
      return Promise.resolve(context);
    },
    // deno-lint-ignore require-await
    run: async (inputs, bindings, generation): Promise<RunOutputs> => {
      // MUST: 実 context と同じ拒否（保留がある間の 2 本目は「どこまでが確定か」が決まらない
      // まま論理長を捕捉する）。記録もしない = 発行の同期区間で落ちる run と同じ扱い。
      if (pending !== undefined) {
        throw new Error(
          `fake: commit 待ちの generation run がある（queryLength ${pending.queryLength}）`,
        );
      }
      // MUST: deferred run の行数は sliding ring の余裕まで（実 runtime の門を**同じ文言で**
      // 写す — `executor.ts` の deferred 節）。棄却行は物理 ring 上で live な過去 KV を潰すので、
      // ここを持たない fake だと `k + 1 ≤ slidingSlack` を外す退行が GPU 無しでは緑のまま、
      // 実 GPU の最初の verify（GB 級ロードの末）で初めて落ちる。
      if (generation.commit === "deferred") {
        const slack = context.slidingSlack;
        if (slack !== undefined && generation.queryLength > slack) {
          throw new Error(
            `run: deferred な generation run の queryLength ${generation.queryLength} が ` +
              `sliding ring の余裕 ${slack} を超える（棄却行が live な過去 KV を潰す）`,
          );
        }
      }
      const call = calls.length;
      const ids = readRow(inputs, IDS);
      const lastRow = readRow(inputs, LAST_ROW);
      calls.push({
        ids: ids.values,
        idsShape: ids.shape,
        // この run の直前に `derive` が受けた位置列（run の入力には無い）。
        positions: derivedPositions,
        lastRow: lastRow.values[0],
        lastRows: lastRow.values,
        lastRowShape: lastRow.shape,
        queryLength: generation.queryLength,
        commit: generation.commit,
        pastBefore: pastLength,
        bindings,
        extra: Object.keys(inputs).filter((name) => name !== IDS && name !== LAST_ROW),
        sameContext: generation.context === context,
      });
      if (options.failAt === call) throw new Error("run が落ちた");
      // 論理長の進行は run の成功で起きる（deferred は保留して commit を待つ — 実 context と同じ）。
      if (generation.commit === "deferred") {
        pending = { pastLength, queryLength: generation.queryLength };
      } else {
        pastLength += generation.queryLength;
      }
      // 返す行数は渡った `last_row` の本数（実グラフと同じ = R は入力が束縛する）。
      const rows = options.logitsRows !== undefined &&
          (options.logitsRowsAt === undefined || options.logitsRowsAt === call)
        ? options.logitsRows
        : lastRow.values.length;
      const logits = new Float32Array(rows * VOCAB);
      const hidden = new Float32Array(rows * HIDDEN_SIZE);
      for (let row = 0; row < rows; row += 1) {
        // 出力の行 n が運ぶのは `last_row[n]` が指した**入力行**のものである（行数を偽る故障
        // 注入では添字が尽きるので、その分は入力行 0 に畳む）。
        const inputRow = lastRow.values[row] ?? 0;
        const id = options.successor === undefined
          ? options.tokens?.[call] ?? (call + 1) % VOCAB
          : options.successor[ids.values[inputRow]];
        logits[row * VOCAB + id] = 10;
        if (options.runnerUp !== undefined) {
          logits[row * VOCAB + options.runnerUp.id] = options.runnerUp.logit;
        }
        // 汚染は行の**別の** id に置く（狙った id を潰すと「行の argmax」の写像が壊れる）。
        if (options.nanAt === call) logits[row * VOCAB + (id + 1) % VOCAB] = Number.NaN;
        hidden[row * HIDDEN_SIZE] = hiddenMark(call, inputRow);
        hidden[row * HIDDEN_SIZE + 1] = ids.values[inputRow];
      }
      return {
        [LOGITS]: { dtype: "f32", shape: [1, rows, VOCAB], data: logits },
        // hidden は投機経路だけが読む（非投機で読む実装へ退行すれば、行ごとに違うこの値が
        // logits の形検査に掛かる）。
        [HIDDEN]: { dtype: "f32", shape: [1, rows, HIDDEN_SIZE], data: hidden },
      };
    },
  };
  return {
    session,
    derivedInputs,
    /** `createGenerationContext` が返す唯一の実体（借り手の `open` に渡る面と同一であること）。 */
    context,
    calls,
    specs,
    commits,
    disposals: (): number => disposals,
    pastLength: (): number => pastLength,
    pendingCommit: (): { readonly pastLength: number; readonly queryLength: number } | undefined =>
      pending,
  };
};

/** イベントを全部汲む（`done` も一緒に返す）。 */
export const drain = async (
  stream: GenerationStream,
): Promise<{ readonly events: GenerationEvent[]; readonly stop: GenerationStop }> => {
  const events: GenerationEvent[] = [];
  for await (const event of stream) events.push(event);
  return { events, stop: await stream.done };
};

export const tokenIds = (events: readonly GenerationEvent[]): number[] =>
  events.filter((event) => event.kind === "token").map((event) => event.id);
