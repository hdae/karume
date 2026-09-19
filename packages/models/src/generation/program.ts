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
 * - 名前の実在（入力 2 本 + 派生入力の名前 + 出力 2 本）
 * - 形と dtype（`[1,M]` の i32・`[R]` の i32・`[1,R,V]` の f32・`[1,R,H]` の f32）
 * - **グラフ入力の完全被覆**（program が結線しない入力が 1 本も残らない・余分な名前も無い）
 * - 記号（入力 shape から決まらない記号は容量記号ちょうど 1 本であること）
 *
 * MUST: program は可変状態を持たない（ADR 0083 決定 1）。
 */

import { assertChunkBuckets, type RunInputs } from "@karume/runtime";

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
  /**
   * 選ぶ行の添字**列**を受けるグラフ入力の名前（`[R]` の i32 — ADR 0068 決定 4 の行選択）。
   *
   * R は記号で、その run で選ぶ行数そのものである（この入力の要素数が R を束縛する唯一の源）。
   * 通常の prefill / decode は R=1（最終有効行 1 本）で、値も token 列も従来と同一である。
   * R を固定数 1 で焼くと、投機の verify（draft 行を一度に検証する run）が同じグラフで回せない。
   */
  readonly lastRow: string;
  /** 選んだ行の logits を出すグラフ出力の名前（`[1,R,V]` の f32 — ADR 0083 決定 6）。 */
  readonly logits: string;
  /**
   * 選んだ行の**最終 norm 後 hidden** を出すグラフ出力の名前（`[1,R,H]` の f32）。
   *
   * MUST: 省略可能にしない。drafter は「本体が実際に置いた行の hidden」を入力に取るので、
   * ここが欠けた配布形では投機が組めない。省略可能にすると「hidden の無いグラフでも program は
   * 組める」形になり、欠けは drafter を繋ぐ段まで落ちない（配布形の焼き直しが要る所まで）。
   */
  readonly hidden: string;
  /** 固定長 prefill chunk の行数（ADR 0066 決定 4 — context の計画時定数）。 */
  readonly chunkLength: number;
  /**
   * prefill 形として `chunkLength` に**加えて**許す物理 chunk 行数（ADR 0066 決定 4 /
   * 追記〈バケット〉）。省略 / 空 = 追加なし（prefill 形 1 本 + decode 形の従来どおり）。
   *
   * 短い prompt を `chunkLength` 行へ pad すると、pad 行ぶんの仕事（行局所な linear /
   * pointwise / norm は物理行数に比例する）がそのまま無駄になる。宣言しておくと chunk ごとに
   * `queryLength` 以上の最小バケットを物理行数に選べる（`sequence.ts` の `physicalChunkRows`）。
   *
   * MUST: 受理集合（2 以上 `chunkLength` 未満の整数・狭義昇順）の検査は runtime の
   * `assertChunkBuckets` 1 本に任せる — 同じ規則をここで写すと、context が許す集合と
   * 呼び出し側が選ぶ集合が別々に育つ。
   */
  readonly chunkBuckets?: readonly number[];
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
  /**
   * **Session 側が常駐入力として差す**グラフ入力の名前（gemma4 の PLE GPU 常駐席 — ADR 0085
   * 追記〈GPU 常駐席〉）。
   *
   * 宣言するのは名前だけで、値は作らない（作り手は run を発行する側 = `Session` の包み）。
   * それでも席が要るのは「グラフ入力の完全被覆」を setup で見るためで、ここに書かないと
   * **ホストが作らない入力**が「結線されていない」として落ちる。
   *
   * MUST: {@link DerivedRunInputs.names} と重ねない（同じ入力を 2 つの作り手が名乗る形）。
   * 重複は {@link assertInputCoverage} が落とす。
   */
  readonly residentInputs?: readonly string[];
};

/**
 * 検証済みの静的配線（{@link createGenerationProgram} だけが作る）。**内部の型**で、公開面には
 * 出さない（出す面は {@link GenerationProgram}）。
 *
 * MUST: フィールドを足すときは {@link createGenerationProgram} の検証も同時に足す — 検証されない
 * 配線欄は「setup 時に全結線を検証する」という本型の存在理由を静かに壊す。
 */
export type GenerationWiring =
  & Omit<GenerationProgramSpec, "graph" | "chunkBuckets">
  & {
    /**
     * 検証済みの prefill バケット（**省略できない** — 宣言の無い spec は空配列へ畳んである）。
     *
     * 省略可能なまま持ち回すと、物理行数を選ぶ側（`sequence.ts`）が毎回 `?? []` を書くことに
     * なり、その 1 つが欠けても「バケットが黙って効かない」だけで例外は出ない。
     */
    readonly chunkBuckets: readonly number[];
    /**
     * hidden 出口の最終軸 H（{@link GenerationProgramSpec.hidden} の宣言形から**導出**した値）。
     *
     * MUST: 呼び手に宣言させない（`vocabSize` と違い、突き合わせる相手が資産側に無い）。
     * グラフが唯一の源なので、宣言を受けると「配線の H」と「グラフの H」が独立に更新される
     * 二重持ちになる。
     */
    readonly hiddenSize: number;
    /**
     * 行数記号 R の名前（`last_row` 入力の宣言形 `[R]` から**導出**）。run では `last_row` の要素数が
     * これを束縛するが、run を伴わない見積り（`estimateGraphMemory` の `bindings`）は入力 shape を
     * 持たないので、この名前で R = 1 を明示して渡す。
     */
    readonly rowSymbol: string;
  };

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
  /**
   * prefill 形として `chunkLength` に加えて使う物理 chunk 行数（狭義昇順・空なら無し）。
   *
   * 短い chunk はこの中から `queryLength` 以上の最小値を物理行数に選ぶ（無ければ
   * `chunkLength`）。自分で `sequence()` を回す側が「この prompt 長は何行に載るか」を
   * 読める唯一の値である。
   */
  readonly chunkBuckets: readonly number[];
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
    // `stopTokens` と同じ理由で凍結コピー（消費者の `sort()` / `length = 0` が物理行数の
    // 選び方そのものを書き換えないようにする）。
    chunkBuckets: Object.freeze([...wiring.chunkBuckets]),
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

/**
 * 行選択入力（`[R]` の i32）であることを見て、その**記号名 R** を返す。
 *
 * MUST: 次元が**記号**であることまで見る。固定数 1 だと「1 run = 1 行」しか焼かれておらず、
 * 投機の verify（draft 行をまとめて検証する run）が同じグラフで回せない。R=1 の run
 * （通常の prefill / decode）は要素 1 本の入力で従来と同じ形に畳まれる。
 *
 * NOTE: R が入力 shape に現れることが {@link assertSymbols} の前提でもある — 行数は run の
 * 入力（この列の長さ）から決まり、容量記号のように context の束縛を要らない。
 */
const assertLastRowInput = (graph: GenerationGraph, name: string): string => {
  const spec = findInput(graph, name, "last_row 入力");
  assertDtype(spec.dtype, "i32", `last_row 入力 '${name}'`);
  const rowSymbol = spec.shape[0];
  if (spec.shape.length !== 1 || typeof rowSymbol !== "string") {
    throw new Error(
      `last_row 入力 '${name}' の shape ${showShape(spec.shape)} が [<記号>] でない` +
        `（R=1 の prefill / decode と R>1 の verify を同じグラフで回せない）`,
    );
  }
  return rowSymbol;
};

/**
 * グラフ**出力**に載っている名前の値情報を引く。
 *
 * MUST: **グラフ出力に載っていること**まで見る。ノード出力として存在するだけの名前は run から
 * 返ってこないので、「出力 '…' が無い」という真因から遠い実行時例外になる。
 */
const findOutputValue = (graph: GenerationGraph, name: string, role: string) => {
  if (!graph.outputs.includes(name)) {
    throw new Error(
      `${role} '${name}' がグラフ出力に無い（実在するのは ${graph.outputs.join(" / ")}）`,
    );
  }
  if (!Object.hasOwn(graph.values, name)) {
    throw new Error(`${role} '${name}' の値情報がグラフに無い`);
  }
  return graph.values[name];
};

/**
 * logits 出口（`[1,R,V]` の f32）であることを見る。
 *
 * MUST: 2 次元目が **`last_row` と同じ記号**であることまで見る。別記号なら「選んだ行数」と
 * 「返る行数」が別々に決まる形で、run は形が合う限り通ってしまう（verify で `R` 行渡したのに
 * 1 行しか返らない、が例外なしで起きる）。
 */
const assertLogitsOutput = (
  graph: GenerationGraph,
  name: string,
  vocabSize: number,
  rowSymbol: string,
): void => {
  const info = findOutputValue(graph, name, "logits 出口");
  assertDtype(info.dtype, "f32", `logits 出口 '${name}'`);
  if (
    info.shape.length !== 3 || info.shape[0] !== 1 || info.shape[1] !== rowSymbol ||
    info.shape[2] !== vocabSize
  ) {
    throw new Error(
      `logits 出口 '${name}' の shape ${showShape(info.shape)} が ` +
        `[1,${rowSymbol},${vocabSize}] でない` +
        `（選んだ**行**だけの出口であること — ADR 0083 決定 6）`,
    );
  }
};

/**
 * hidden 出口（`[1,R,H]` の f32）であることを見て、H を返す。
 *
 * `vocabSize` に当たる宣言を受けないのは、H を突き合わせる相手が資産側に無いためである
 * （語彙数は tokenizer と PLE sidecar の相互照合の基準になるが、hidden 幅はグラフだけが持つ）。
 * よってここが見るのは「正整数の固定次元であること」まで — 記号のままなら run ごとに幅が
 * 変わる形で、drafter 側の重みと繋がらない。
 */
const assertHiddenOutput = (graph: GenerationGraph, name: string, rowSymbol: string): number => {
  const info = findOutputValue(graph, name, "hidden 出口");
  assertDtype(info.dtype, "f32", `hidden 出口 '${name}'`);
  const hiddenSize = info.shape[2];
  if (
    info.shape.length !== 3 || info.shape[0] !== 1 || info.shape[1] !== rowSymbol ||
    typeof hiddenSize !== "number"
  ) {
    throw new Error(
      `hidden 出口 '${name}' の shape ${showShape(info.shape)} が [1,${rowSymbol},<H>] でない` +
        `（選んだ行の最終 norm 後 hidden であること）`,
    );
  }
  assertPositiveInteger(hiddenSize, `hidden 出口 '${name}' の H`);
  return hiddenSize;
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
  // 受理集合の正本は runtime（`createGenerationContext` が同じ関数で拒否する）。ここで通すのは
  // 「context を作る前に落とす」ためで、規則そのものは持たない。
  assertChunkBuckets(spec.chunkBuckets, spec.chunkLength);
  assertPositiveInteger(spec.maxPosition, "maxPosition");
  assertPositiveInteger(spec.capacity, "capacity");
  assertPositiveInteger(spec.vocabSize, "vocabSize");
  spec.stopTokens.forEach((token, index) => {
    if (!Number.isSafeInteger(token) || token < 0 || token >= spec.vocabSize) {
      throw new Error(`stopTokens[${index}] ${token} が語彙 0..${spec.vocabSize - 1} の外`);
    }
  });

  assertRowInput(graph, spec.inputIds, "token id 入力");
  const rowSymbol = assertLastRowInput(graph, spec.lastRow);
  // 同じ名前を 2 本の出口に結線した形は、V = H のときだけ両方の形検査を通ってしまう
  // （drafter が logits を hidden として食う = 例外の出ない取り違え）。
  if (spec.logits === spec.hidden) {
    throw new Error(`logits 出口と hidden 出口が同じ名前 '${spec.logits}' を指している`);
  }
  assertLogitsOutput(graph, spec.logits, spec.vocabSize, rowSymbol);
  const hiddenSize = assertHiddenOutput(graph, spec.hidden, rowSymbol);
  assertInputCoverage(graph, [
    spec.inputIds,
    spec.lastRow,
    ...(spec.derivedInputs?.names ?? []),
    ...(spec.residentInputs ?? []),
  ]);
  assertSymbols(graph, spec.capacitySymbol);

  return {
    inputIds: spec.inputIds,
    lastRow: spec.lastRow,
    logits: spec.logits,
    hidden: spec.hidden,
    hiddenSize,
    rowSymbol,
    chunkLength: spec.chunkLength,
    // 凍結コピー: 配線は不変オブジェクトなので、呼び手の配列を後から書き換えられると
    // 「context が許す集合」と「物理行数を選ぶ集合」が実行中に割れる。
    chunkBuckets: Object.freeze([...(spec.chunkBuckets ?? [])]),
    maxPosition: spec.maxPosition,
    capacity: spec.capacity,
    vocabSize: spec.vocabSize,
    stopTokens: [...spec.stopTokens],
    capacitySymbol: spec.capacitySymbol,
    ...(spec.derivedInputs === undefined ? {} : { derivedInputs: spec.derivedInputs }),
    ...(spec.residentInputs === undefined ? {} : { residentInputs: [...spec.residentInputs] }),
  };
};
