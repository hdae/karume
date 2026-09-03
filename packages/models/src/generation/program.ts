/**
 * 生成の**静的配線**（ADR 0083 決定 1 の `GenerationProgram`）。**パイプライン非依存の共通処理**
 * なので `greedy.ts` / `sampler.ts` と同じ `src/generation/` に置く。
 *
 * ## 2 層に割る理由
 *
 * 現行の `GreedySpec`（`greedy.ts`）は静的配線（グラフ入力 / 出力の名前・chunk 長・位置上限）と
 * リクエスト（prompt・生成数）が 1 つの型に同居している。多ターンの会話は「同じ配線で別の
 * リクエストを何度も出す」形なので、ここを割らないと**毎回すべての配線を書き直す**面になる。
 * program は不変で、可変な寿命は {@link GenerationSequence}（`sequence.ts`）だけが持つ。
 *
 * 配線の型はさらに 2 つに割れている — 生成ループが読む全欄が {@link GenerationWiring}（内部）、
 * 消費者が `sequence()` を回すときに読む数だけが {@link GenerationProgram}（公開・凍結）。
 *
 * ## MUST: setup 時に全結線を検証する
 *
 * 名前の取り違えは**例外も警告も出ない**（`greedy.ts` の `readToken` が同じ理由で dtype と形を
 * 見ている）。出力名が別の出力を指していれば、形が合う限り「もっともらしい token 列」が出る。
 * グラフ入力が 1 本結線されないまま run へ行けば診断は真因から遠い場所で出る。よって
 * {@link createGenerationProgram} は**グラフと突き合わせて**次を全部見る:
 *
 * - 名前の実在（入力 2 本 + 派生入力の名前 + 出力 1 本）
 * - 形と dtype（`[1,M]` の i32・`[1]` の i32・`[1,1,V]` の f32）
 * - **グラフ入力の完全被覆**（program が結線しない入力が 1 本も残らない・余分な名前も無い）
 * - 記号（入力 shape から決まらない記号は容量記号ちょうど 1 本であること）
 *
 * MUST: program は可変状態を持たない（ADR 0083 決定 1）。
 */

import type { RunInputs } from "@karume/runtime";

/**
 * program が結線検証に使うグラフの面（`PreparedModel["graph"]` の部分集合）。
 *
 * MUST: `PreparedModel["graph"]` をそのまま要求しない。IR の全体（nodes / initializers /
 * states）は結線検証に要らないうえ、要求すると単体テストが**実 IR コンテナを組む**羽目になる
 * （`greedy.ts` の `GreedySession` が narrow interface である理由と同じ）。実 `IrGraph` が
 * この面を満たすことは型門（テスト側）で固定する。
 */
export type GenerationGraph = {
  readonly symbols: readonly string[];
  readonly inputs: readonly {
    readonly name: string;
    readonly dtype: string;
    readonly shape: readonly (number | string)[];
  }[];
  readonly outputs: readonly string[];
  readonly values: Readonly<
    Record<string, { readonly dtype: string; readonly shape: readonly (number | string)[] }>
  >;
};

/**
 * {@link DerivedRunInputs.derive} が受ける実行時のノブ（静的配線ではなく**その run 1 回**の事情）。
 *
 * MUST: **best-effort** の契約である — 実装が `signal` を無視しても壊れない（無視すれば中断が
 * 「今の派生入力を作り終えてから」効くだけ）。生成ループは `derive` の `await` 明けに自分でも
 * `signal` を見て run の発行を止めるので、中断の正しさをこの席へ委ねていない。
 */
export type DeriveInputsOptions = {
  /**
   * この生成の中断（`GenerationRequest.signal` がそのまま降りてくる）。
   *
   * 派生入力の材料が GB 級の遅延ロードになる配布形（gemma4 の PLE sidecar — ADR 0085）では、
   * ここを見ないと「停止を押しても 758MB の読みが終わるまで返らない」形になる。
   */
  readonly signal?: AbortSignal;
};

/**
 * ホスト由来の per-chunk 入力の席（ADR 0083 決定 1 の「モデル固有の入力の作り方は models 側の
 * 知識」— gemma4 の PLE `per_layer_inputs[1,M,35,256]` がこの一実装。ADR 0085）。
 *
 * 前例は SBV2 の相対位置表 / Anima の rope 素表で、どちらも「グラフ入力を作るのはホスト」。
 * ここは**その作り手を program に差し込む席**で、生成ループは token id 列を渡すだけになる。
 */
export type DerivedRunInputs = {
  /**
   * この席が供給するグラフ入力の名前。
   *
   * MUST: 省略可能にしない。名前を宣言させることが「グラフ入力の完全被覆」を setup で見られる
   * 唯一の手段で（関数の返り値は呼ばないと分からない）、宣言と実際の返り値の食い違いは
   * 実行時に落とす（{@link DerivedRunInputs.derive} の契約）。
   */
  readonly names: readonly string[];
  /**
   * 物理 chunk 1 本ぶんの token id 列と**絶対位置列** → 追加入力。
   *
   * 渡るのはどちらも**物理行数ぶん**（prefill は pad 行を含む・decode は 1 行）で、pad 行には
   * `input_ids` と同じ 0 が入る — グラフ内で引いていたときと同じ値にするため（PLE の
   * `gather` doc と ADR 0066 追記 6 の値契約）。位置も同じ規約で、pad 行は 0 である。
   *
   * `positions` を渡すのは、位置に依存するホスト由来入力（gemma4 の RoPE cos / sin —
   * `gemma/rope.ts`）がこの席を使うため。`position_ids` を**グラフ入力として**渡す形は
   * もう無いので、位置の唯一の行き先がここである。
   *
   * MUST: **純関数席**（同じ `(ids, positions)` に同じ値。呼ぶ順序に依らない）。返り値のキーは
   * {@link DerivedRunInputs.names} と過不足なく一致すること。`options` は値に影響しない
   * 事情（中断）だけを運ぶので、この MUST とは両立する。
   */
  readonly derive: (
    ids: readonly number[],
    positions: readonly number[],
    options?: DeriveInputsOptions,
  ) => Promise<RunInputs>;
};

/** {@link createGenerationProgram} の指定（グラフを伴う — 検証はここで全部済ませる）。 */
export type GenerationProgramSpec = {
  /** 結線を突き合わせるグラフ（`PreparedModel.graph` をそのまま渡せる）。 */
  readonly graph: GenerationGraph;
  /** token id 列を受けるグラフ入力の名前（`[1,M]` の i32）。 */
  readonly inputIds: string;
  /** 最終有効行の添字を受けるグラフ入力の名前（`[1]` の i32 — ADR 0068 決定 4 の行選択）。 */
  readonly lastRow: string;
  /** 最終行 logits を出すグラフ出力の名前（`[1,1,V]` の f32 — ADR 0083 決定 6）。 */
  readonly logits: string;
  /** 固定長 prefill chunk の行数（ADR 0066 決定 4 — context の計画時定数）。 */
  readonly chunkLength: number;
  /**
   * 引ける絶対位置の**排他的上限**（位置は `0..maxPosition-1` — モデルが宣言する位置上限）。
   *
   * MUST: 省略可能にしない（`greedy.ts` の `maxPosition` と同じ理由 — 上限の外の位置は例外を
   * 出さず、学習していない位置の attention が「もっともらしい token id」に畳まれる）。
   */
  readonly maxPosition: number;
  /**
   * full スロットの容量の**既定**（`pastLength + queryLength ≤ capacity` — ADR 0067 決定 4 ④）。
   *
   * MUST: 省略可能にしない。超過はランタイムも拒否するが、それは **run のエンコード直前**で、
   * 「会話が入り切らない」という**ホストが判断すべき事実**が汎用メッセージに埋もれる
   * （ADR 0083 決定 10）。sequence はこの値で run の**前**に見て専用型で落とす。
   *
   * NOTE: 実際に使う容量は sequence（= context）ごとに選べる（`createGenerationSequence` の
   * `capacity`）。program が持つのは**既定**で、context の物理確保はその sequence の値で決まる。
   */
  readonly capacity: number;
  /** 語彙数（logits 出口の最終軸 — グラフと相互照合する）。 */
  readonly vocabSize: number;
  /** 停止 token の集合（ADR 0083 決定 8 — 空なら EOS 停止をしない）。 */
  readonly stopTokens: readonly number[];
  /**
   * full スロット容量の**記号名**（`createGenerationContext` の束縛点で使う綴り）。
   *
   * MUST: 束縛**値**は持たない。容量は sequence ごとに選べるので、値を配線側にも持つと
   * 「program の `capacity` と `bindings` のどちらが本当の容量か」という独立に更新される
   * 二重持ちになる（CLAUDE.md の派生状態の禁止）。記号は資産の綴りで不変、値は実行時ノブ。
   *
   * MUST: 入力 shape から決まる記号を指してはならない（{@link createGenerationProgram} が見る）—
   * その記号は run の入力から決まるので、context 側の束縛と分裂して run が拒否する。
   */
  readonly capacitySymbol: string;
  /** ホスト由来の per-chunk 入力（無い配布形は省略）。 */
  readonly derivedInputs?: DerivedRunInputs;
};

/**
 * 検証済みの静的配線（{@link createGenerationProgram} だけが作る）。**内部の型**で、公開面には
 * 出さない（出す面は {@link GenerationProgram}）。
 *
 * MUST: フィールドを足すときは {@link createGenerationProgram} の検証も同時に足す — 検証されない
 * 配線欄は「setup 時に全結線を検証する」という本型の存在理由を静かに壊す。
 */
export type GenerationWiring = Omit<GenerationProgramSpec, "graph">;

/**
 * 検証済み静的配線の**読み口**（公開面 — `Gemma4Pipeline.program`）。
 *
 * 出すのは「自分で `sequence()` を回すときに読む必要がある数」だけである。グラフ入力 / 出力の
 * 名前・容量記号・`derivedInputs` は**内部配線**（{@link GenerationWiring}）で、公開すると
 * ①消費者が読んでも使い道が無い（配線の相手である Session は公開面に出ていない）
 * ②`derive` の差し替えや記号の改変が公開面から書ける — 検証済みであることが
 * `GenerationProgram` の意味そのものなので、書ける口は意味を壊す。
 *
 * MUST: {@link generationProgramFace} が凍結して返す（`stopTokens` は凍結コピー）。
 */
export type GenerationProgram = {
  /** 固定長 prefill chunk の行数（ADR 0066 決定 4 — この pipeline が使う値）。 */
  readonly chunkLength: number;
  /** 引ける絶対位置の排他的上限（位置は `0..maxPosition-1` — モデルの宣言）。 */
  readonly maxPosition: number;
  /**
   * full スロットの容量の**既定**（`pastLength + queryLength ≤ capacity`）。
   *
   * 実際に使う容量は sequence ごとに選べる（`GenerationSequence.capacity` が**その会話の**値）。
   */
  readonly capacity: number;
  /** 語彙数（`prompt` の token id の値域はここで決まる）。 */
  readonly vocabSize: number;
  /** 停止 token の集合（ADR 0083 決定 8 — 空なら EOS 停止をしない）。 */
  readonly stopTokens: readonly number[];
};

/**
 * 内部配線 → 公開の読み口（**凍結**）。
 *
 * MUST: `stopTokens` は凍結**コピー**にする。同じ配列を出すと、消費者側の `sort()` /
 * `length = 0` が生成ループの停止集合そのものを書き換える（例外にならない沈黙劣化で、
 * 「EOS で止まらない生成」として現れる）。
 */
export const generationProgramFace = (wiring: GenerationWiring): GenerationProgram =>
  Object.freeze({
    chunkLength: wiring.chunkLength,
    maxPosition: wiring.maxPosition,
    capacity: wiring.capacity,
    vocabSize: wiring.vocabSize,
    stopTokens: Object.freeze([...wiring.stopTokens]),
  });

const assertPositiveInteger = (value: number, where: string): void => {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new Error(`${where} ${value} が 1 以上の整数でない`);
  }
};

/** 形を診断文へ落とす（記号次元は綴りのまま出す）。 */
const showShape = (shape: readonly (number | string)[]): string => `[${shape.join(",")}]`;

const findInput = (graph: GenerationGraph, name: string, role: string) => {
  const spec = graph.inputs.find((input) => input.name === name);
  if (spec === undefined) {
    throw new Error(
      `${role} '${name}' がグラフ入力に無い（実在するのは ` +
        `${graph.inputs.map((input) => input.name).join(" / ")}）`,
    );
  }
  return spec;
};

const assertDtype = (actual: string, expected: string, where: string): void => {
  if (actual !== expected) throw new Error(`${where} の dtype が ${actual}（${expected} でない）`);
};

/**
 * `[1, M]` の i32 入力（M は記号）であることを見る。
 *
 * MUST: 2 次元目が**記号**であることまで見る。固定数だと prefill 形（`M = chunkLength`）と
 * decode 形（`M = 1`）の 2 本を同じグラフで回せず（ADR 0066 決定 4）、その食い違いは
 * 「decode の 1 回目で形が合わない」という真因から遠い診断で出る。
 */
const assertRowInput = (
  graph: GenerationGraph,
  name: string,
  role: string,
): void => {
  const spec = findInput(graph, name, role);
  assertDtype(spec.dtype, "i32", `${role} '${name}'`);
  if (spec.shape.length !== 2 || spec.shape[0] !== 1 || typeof spec.shape[1] !== "string") {
    throw new Error(
      `${role} '${name}' の shape ${showShape(spec.shape)} が [1,<記号>] でない` +
        `（prefill 形と decode 形を同じグラフで回せない）`,
    );
  }
};

const assertLastRowInput = (graph: GenerationGraph, name: string): void => {
  const spec = findInput(graph, name, "last_row 入力");
  assertDtype(spec.dtype, "i32", `last_row 入力 '${name}'`);
  if (spec.shape.length !== 1 || spec.shape[0] !== 1) {
    throw new Error(`last_row 入力 '${name}' の shape ${showShape(spec.shape)} が [1] でない`);
  }
};

/**
 * logits 出口（`[1,1,V]` の f32）であることを見る。
 *
 * MUST: **グラフ出力に載っていること**まで見る。ノード出力として存在するだけの名前は run から
 * 返ってこないので、「出力 '…' が無い」という真因から遠い実行時例外になる。
 */
const assertLogitsOutput = (graph: GenerationGraph, name: string, vocabSize: number): void => {
  if (!graph.outputs.includes(name)) {
    throw new Error(
      `logits 出口 '${name}' がグラフ出力に無い（実在するのは ${graph.outputs.join(" / ")}）`,
    );
  }
  if (!Object.hasOwn(graph.values, name)) {
    throw new Error(`logits 出口 '${name}' の値情報がグラフに無い`);
  }
  const info = graph.values[name];
  assertDtype(info.dtype, "f32", `logits 出口 '${name}'`);
  if (
    info.shape.length !== 3 || info.shape[0] !== 1 || info.shape[1] !== 1 ||
    info.shape[2] !== vocabSize
  ) {
    throw new Error(
      `logits 出口 '${name}' の shape ${showShape(info.shape)} が [1,1,${vocabSize}] でない` +
        `（最終**行**のみの出口であること — ADR 0083 決定 6）`,
    );
  }
};

/**
 * グラフ入力が過不足なく結線されていることを見る。
 *
 * MUST: 両方向を見る。**欠け**は run が「バッファが無い」で落ちる（真因から遠い）し、
 * **余り**（`derivedInputs.names` に居ない名前を宣言した形）は毎 run 無視される入力を作る。
 */
const assertInputCoverage = (graph: GenerationGraph, wired: readonly string[]): void => {
  const declared = new Set(wired);
  if (declared.size !== wired.length) {
    throw new Error(`結線した入力名に重複がある（${wired.join(" / ")}）`);
  }
  const actual = new Set(graph.inputs.map((input) => input.name));
  const missing = [...actual].filter((name) => !declared.has(name));
  if (missing.length > 0) {
    throw new Error(
      `グラフ入力 ${missing.join(" / ")} が結線されていない` +
        `（ホスト由来の入力は derivedInputs で供給する）`,
    );
  }
  const extra = wired.filter((name) => !actual.has(name));
  if (extra.length > 0) {
    throw new Error(`結線した ${extra.join(" / ")} がグラフ入力に無い`);
  }
};

/**
 * 記号が全部決まることを見る。
 *
 * 入力 shape に現れる記号は run の入力から決まり、残り（states の容量記号）は
 * `createGenerationContext` の束縛が与える唯一の源である（`resolveBindings` の MUST — states は
 * 束縛源にならない）。両方から漏れた記号は context 生成まで気づけないので、ここで落とす。
 */
const assertSymbols = (graph: GenerationGraph, capacitySymbol: string): void => {
  const fromInputs = new Set<string>();
  for (const input of graph.inputs) {
    for (const dim of input.shape) if (typeof dim === "string") fromInputs.add(dim);
  }
  if (!graph.symbols.includes(capacitySymbol)) {
    throw new Error(
      `容量記号 ${capacitySymbol} がグラフの symbols [${graph.symbols.join(", ")}] に無い`,
    );
  }
  // 入力由来の記号を容量記号に選ぶと、run の入力と context の束縛が同じ記号を別の値で決める
  // （runtime が分裂として拒否する）— 綴りの取り違えなので、配線を組む時点で落とす。
  if (fromInputs.has(capacitySymbol)) {
    throw new Error(
      `容量記号 ${capacitySymbol} は入力 shape から決まる記号である` +
        `（state スロットの容量記号は入力に現れない 1 本 — ADR 0066 追記 7）`,
    );
  }
  const unresolved = graph.symbols.filter(
    (symbol) => !fromInputs.has(symbol) && symbol !== capacitySymbol,
  );
  if (unresolved.length > 0) {
    throw new Error(
      `記号 ${unresolved.join(" / ")} が入力 shape からも容量記号からも決まらない` +
        `（state スロットの容量記号は 1 本だけ — ADR 0066 追記 7）`,
    );
  }
};

/**
 * 静的配線をグラフと突き合わせて確定する（**唯一の入口** — 検証を迂回した program を作らせない）。
 *
 * MUST: GPU に触る前に落ちる（引数はグラフと数値だけ）。配線の誤りが 3.7GiB のロードの末に
 * 出るのと、`prepareModel` の直後に出るのとでは診断の価値が違う。
 */
export const createGenerationProgram = (spec: GenerationProgramSpec): GenerationWiring => {
  const { graph } = spec;
  assertPositiveInteger(spec.chunkLength, "chunkLength");
  assertPositiveInteger(spec.maxPosition, "maxPosition");
  assertPositiveInteger(spec.capacity, "capacity");
  assertPositiveInteger(spec.vocabSize, "vocabSize");
  spec.stopTokens.forEach((token, index) => {
    if (!Number.isSafeInteger(token) || token < 0 || token >= spec.vocabSize) {
      throw new Error(`stopTokens[${index}] ${token} が語彙 0..${spec.vocabSize - 1} の外`);
    }
  });

  assertRowInput(graph, spec.inputIds, "token id 入力");
  assertLastRowInput(graph, spec.lastRow);
  assertLogitsOutput(graph, spec.logits, spec.vocabSize);
  assertInputCoverage(graph, [
    spec.inputIds,
    spec.lastRow,
    ...(spec.derivedInputs?.names ?? []),
  ]);
  assertSymbols(graph, spec.capacitySymbol);

  return {
    inputIds: spec.inputIds,
    lastRow: spec.lastRow,
    logits: spec.logits,
    chunkLength: spec.chunkLength,
    maxPosition: spec.maxPosition,
    capacity: spec.capacity,
    vocabSize: spec.vocabSize,
    stopTokens: [...spec.stopTokens],
    capacitySymbol: spec.capacitySymbol,
    ...(spec.derivedInputs === undefined ? {} : { derivedInputs: spec.derivedInputs }),
  };
};
