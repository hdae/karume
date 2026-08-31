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
 * ## MUST: setup 時に全結線を検証する
 *
 * 名前の取り違えは**例外も警告も出ない**（`greedy.ts` の `readToken` が同じ理由で dtype と形を
 * 見ている）。出力名が別の出力を指していれば、形が合う限り「もっともらしい token 列」が出る。
 * グラフ入力が 1 本結線されないまま run へ行けば診断は真因から遠い場所で出る。よって
 * {@link createGenerationProgram} は**グラフと突き合わせて**次を全部見る:
 *
 * - 名前の実在（入力 3 本 + 派生入力の名前 + 出力 1 本）
 * - 形と dtype（`[1,M]` の i32・`[1]` の i32・`[1,1,V]` の f32）
 * - **グラフ入力の完全被覆**（program が結線しない入力が 1 本も残らない・余分な名前も無い）
 * - 記号（入力 shape から決まらない記号は `bindings` で与えられていること）
 *
 * MUST: program は可変状態を持たない（ADR 0083 決定 1）。
 */

import type { RunInputs, SymbolBindings } from "@karume/runtime";

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
   * 物理 chunk 1 本ぶんの token id 列 → 追加入力。
   *
   * 渡るのは**物理行数ぶん**（prefill は pad 行を含む・decode は 1 行）で、pad 行にも
   * `input_ids` と同じ 0 が入る — グラフ内で引いていたときと同じ値にするため（PLE の
   * `gather` doc と ADR 0066 追記 6 の値契約）。
   *
   * MUST: **純関数席**（同じ id 列に同じ値。呼ぶ順序に依らない）。返り値のキーは
   * {@link DerivedRunInputs.names} と過不足なく一致すること。`options` は値に影響しない
   * 事情（中断）だけを運ぶので、この MUST とは両立する。
   */
  readonly derive: (
    ids: readonly number[],
    options?: DeriveInputsOptions,
  ) => Promise<RunInputs>;
};

/** {@link createGenerationProgram} の指定（グラフを伴う — 検証はここで全部済ませる）。 */
export type GenerationProgramSpec = {
  /** 結線を突き合わせるグラフ（`PreparedModel.graph` をそのまま渡せる）。 */
  readonly graph: GenerationGraph;
  /** token id 列を受けるグラフ入力の名前（`[1,M]` の i32）。 */
  readonly inputIds: string;
  /** 絶対位置を受けるグラフ入力の名前（`[1,M]` の i32）。 */
  readonly positionIds: string;
  /** 最終有効行の添字を受けるグラフ入力の名前（`[1]` の i32 — ADR 0068 決定 4 の行選択）。 */
  readonly lastRow: string;
  /** 最終行 logits を出すグラフ出力の名前（`[1,1,V]` の f32 — ADR 0083 決定 6）。 */
  readonly logits: string;
  /** 固定長 prefill chunk の行数（ADR 0066 決定 4 — context の計画時定数）。 */
  readonly chunkLength: number;
  /**
   * 資産が引ける絶対位置の**排他的上限**（位置は `0..maxPosition-1` — RoPE 表を焼いた系列なら
   * 表の行数）。
   *
   * MUST: 省略可能にしない（`greedy.ts` の `maxPosition` と同じ理由 — 表の外の gather は例外を
   * 出さず、非有限 logits が「もっともらしい token id」に畳まれる）。
   */
  readonly maxPosition: number;
  /**
   * full スロットの容量（`pastLength + queryLength ≤ capacity` — ADR 0067 決定 4 ④）。
   *
   * MUST: 省略可能にしない。超過はランタイムも拒否するが、それは **run のエンコード直前**で、
   * 「会話が入り切らない」という**ホストが判断すべき事実**が汎用メッセージに埋もれる
   * （ADR 0083 決定 10）。sequence はこの値で run の**前**に見て専用型で落とす。
   */
  readonly capacity: number;
  /** 語彙数（logits 出口の最終軸 — グラフと相互照合する）。 */
  readonly vocabSize: number;
  /** 停止 token の集合（ADR 0083 決定 8 — 空なら EOS 停止をしない）。 */
  readonly stopTokens: readonly number[];
  /** state スロット容量の記号束縛（`createGenerationContext` へ素通し）。 */
  readonly bindings?: SymbolBindings;
  /** ホスト由来の per-chunk 入力（無い配布形は省略）。 */
  readonly derivedInputs?: DerivedRunInputs;
};

/**
 * 検証済みの静的配線（{@link createGenerationProgram} だけが作る）。
 *
 * MUST: フィールドを足すときは {@link createGenerationProgram} の検証も同時に足す — 検証されない
 * 配線欄は「setup 時に全結線を検証する」という本型の存在理由を静かに壊す。
 */
export type GenerationProgram = Omit<GenerationProgramSpec, "graph">;

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
 * 入力 shape に現れる記号は run の入力から決まり、残り（states の容量記号など）は `bindings` が
 * 与える唯一の源である（`resolveBindings` の MUST — states は束縛源にならない）。両方から
 * 漏れた記号は context 生成まで気づけないので、ここで落とす。
 */
const assertSymbols = (graph: GenerationGraph, bindings: SymbolBindings | undefined): void => {
  const fromInputs = new Set<string>();
  for (const input of graph.inputs) {
    for (const dim of input.shape) if (typeof dim === "string") fromInputs.add(dim);
  }
  const bound = new Set(Object.keys(bindings ?? {}));
  const unknown = [...bound].filter((symbol) => !graph.symbols.includes(symbol));
  if (unknown.length > 0) {
    throw new Error(
      `束縛 ${unknown.join(" / ")} がグラフの symbols [${graph.symbols.join(", ")}] に無い`,
    );
  }
  const unresolved = graph.symbols.filter(
    (symbol) => !fromInputs.has(symbol) && !bound.has(symbol),
  );
  if (unresolved.length > 0) {
    throw new Error(
      `記号 ${unresolved.join(" / ")} が入力 shape からも bindings からも決まらない` +
        `（state スロットの容量記号は bindings で与える — ADR 0066 追記 7）`,
    );
  }
};

/**
 * 静的配線をグラフと突き合わせて確定する（**唯一の入口** — 検証を迂回した program を作らせない）。
 *
 * MUST: GPU に触る前に落ちる（引数はグラフと数値だけ）。配線の誤りが 3.7GiB のロードの末に
 * 出るのと、`prepareModel` の直後に出るのとでは診断の価値が違う。
 */
export const createGenerationProgram = (spec: GenerationProgramSpec): GenerationProgram => {
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
  assertRowInput(graph, spec.positionIds, "位置入力");
  assertLastRowInput(graph, spec.lastRow);
  assertLogitsOutput(graph, spec.logits, spec.vocabSize);
  assertInputCoverage(graph, [
    spec.inputIds,
    spec.positionIds,
    spec.lastRow,
    ...(spec.derivedInputs?.names ?? []),
  ]);
  assertSymbols(graph, spec.bindings);

  return {
    inputIds: spec.inputIds,
    positionIds: spec.positionIds,
    lastRow: spec.lastRow,
    logits: spec.logits,
    chunkLength: spec.chunkLength,
    maxPosition: spec.maxPosition,
    capacity: spec.capacity,
    vocabSize: spec.vocabSize,
    stopTokens: [...spec.stopTokens],
    ...(spec.bindings === undefined ? {} : { bindings: spec.bindings }),
    ...(spec.derivedInputs === undefined ? {} : { derivedInputs: spec.derivedInputs }),
  };
};
