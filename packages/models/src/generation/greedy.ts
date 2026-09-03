/**
 * 固定長 greedy 生成ループ（**パイプライン非依存の共通処理** — autoregressive な言語モデルは
 * 総じて「プロンプトを固定長 chunk で流し込む → 1 token ずつ回す」を通る）なので、ファミリの
 * ディレクトリではなく `src/generation/` に置く（`src/image/` / `src/audio/` と同じ位置づけ）。
 *
 * 実行形は ADR 0066 決定 4 の **2 本だけ**: prefill は物理 chunk 行数 `M = chunkLength` 固定で
 * 末尾 chunk を pad し、decode は `M = queryLength = 1`。切るのは `queryLength` で、論理長
 * （`pastLength`）はホストが持たない — 進行は run の成功で context が進める（決定 6 の
 * 二重簿記の禁止）。したがってこのループが覚えているのは「次に食わせる token と絶対位置」だけ。
 *
 * MUST: sampling は載せない。次 token はグラフ側の argmax 出力（`[1,M,1]` の i32）をそのまま
 * 読むだけで、温度も top-k も RNG もホストに置かない（sampling / RNG は op-vocabulary の裁定
 * どおりホスト維持だが、この関数の目的は**固定 token id 列での検収**なので決定的な経路 1 本に
 * 閉じる）。
 */

import type {
  GenerationContext,
  GenerationContextSpec,
  RunInputs,
  RunOutputs,
  SymbolBindings,
  Tensor,
} from "@karume/runtime";

/** 生成ループが context に要求する面 — 実体は不透明で、寿命の返却だけを呼ぶ。 */
export type GenerationDisposable = { dispose(): Promise<void> };

/**
 * 生成ループが Session に要求する面（narrow interface — DI で fake を差せる）。
 *
 * MUST: `Pick<Session, "run" | "createGenerationContext">` にはしない。`GenerationContext` は
 * `#` private を持つ**名前的な型**で、GPU 無しの fake がどう書いても満たせないため、Pick では
 * 単体テストが実 GPU を要求する形になる。context の型を型引数で通してあるのは「create が返した
 * 実体だけが run へ戻る」ことを型で縛るためで、`{ dispose }` に潰すと（メソッドの双変性で）
 * 別の context を実 Session へ渡す形が型検査を通ってしまう。
 *
 * MUST: 実 `Session` がこの面を満たすことはテスト側の型門で固定する（綴りのドリフト検出）。
 */
export type GreedySession<C extends GenerationDisposable = GenerationContext> = {
  createGenerationContext(spec: GenerationContextSpec): Promise<C>;
  run(
    inputs: RunInputs,
    bindings: SymbolBindings | undefined,
    generation: { readonly context: C; readonly queryLength: number },
  ): Promise<RunOutputs>;
};

/** {@link generateGreedy} の指定。 */
export type GreedySpec<C extends GenerationDisposable = GenerationContext> = {
  readonly session: GreedySession<C>;
  /** token id 列を受けるグラフ入力の名前（`[1,M]` の i32）。 */
  readonly inputIds: string;
  /**
   * ホスト由来の派生入力の作り手（`program.ts` の `DerivedRunInputs` と同じ席の最小形）。
   *
   * 渡るのは**物理行数ぶん**の token id 列と絶対位置列（prefill は pad 行込み・decode は 1 行）で、
   * pad 行にはどちらも 0 が入る（ADR 0066 追記 6 の値契約）。返り値はそのまま run の入力へ
   * 展開するので、位置に依存するグラフ入力（gemma4 の RoPE cos / sin — `gemma/rope.ts`・
   * MiniCPM5 decode 系列の `position_ids`）はここが唯一の供給口である。
   *
   * MUST: **同期の純関数**（同じ `(ids, positions)` に同じ値）。`sequence.ts` 側が非同期なのは
   * GB 級の遅延ロード（PLE sidecar）を待つ席が要るためで、この関数の目的は固定 token id 列での
   * 検収なので、待ちの入る派生入力は通さない。
   */
  readonly derive?: (
    ids: Int32Array<ArrayBuffer>,
    positions: Int32Array<ArrayBuffer>,
  ) => RunInputs;
  /** argmax token を出すグラフ出力の名前（`[1,M,1]` の i32 — token-only 形は `[1,1,1]`）。 */
  readonly token: string;
  /**
   * token-only 出口（ADR 0068 決定 4 の既定形）の **last_row 入力**の名前（`[1]` の i32）。
   * 指定すると各 run に最終有効行の添字（prefill = `queryLength − 1` / decode = `0`）を供給し、
   * `token` 出力を `[1,1,1]`（選ばれた 1 行ぶん）として読む。省略時は現行の logits opt-in 形
   * （`[1,M,1]` の全行 argmax から最終有効行を読む）。
   */
  readonly lastRow?: string;
  /** 固定長 prefill chunk の行数（ADR 0066 決定 4 — context の計画時定数）。 */
  readonly chunkLength: number;
  /**
   * 引ける絶対位置の**排他的上限**（位置は `0..maxPosition-1` — モデルが宣言する位置上限。
   * RoPE 表を焼いたまま出す系列ではその表の行数がそのまま上限で、MiniCPM5 decode 系列は 512）。
   *
   * MUST: 省略可能にしない。上限の外の位置は例外を出さず（表を焼いた系列の OOB gather は
   * by-design で非有限 — limitations。表を持たない系列でも学習していない位置の attention に
   * なる）、argmax がその logits を**もっともらしい token id に畳む**ので、上限を知らないまま
   * 回すと生成の後半が沈黙誤 token になる（2026-08-18 実測 — 位置 512 で全 logits 非有限 →
   * token 0）。ここで落とすのが唯一の fail loudly の位置。
   */
  readonly maxPosition: number;
  /** state スロット容量の記号束縛（`createGenerationContext` へ素通し）。 */
  readonly bindings?: SymbolBindings;
  /** プロンプトの token id 列（長さ 1 以上）。 */
  readonly prompt: readonly number[];
  /** 生成する token 数（1 以上）。 */
  readonly maxNewTokens: number;
};

/** prefill 1 回ぶんの割り当て（{@link planPrefillChunks}）。 */
export type PrefillChunk = {
  /** 先頭有効行の絶対位置（= それまでに流し込んだ token 数）。 */
  readonly position: number;
  /** 有効行数（`1..chunkLength` — 末尾 chunk だけが chunkLength 未満になりうる）。 */
  readonly queryLength: number;
};

const assertPositiveInteger = (value: number, where: string): void => {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new Error(`${where} ${value} が 1 以上の整数でない`);
  }
};

/**
 * プロンプト長を固定長 chunk へ割る（ADR 0066 決定 4 — 可変長 chunk も全量一括も採らない）。
 *
 * 末尾 chunk だけが `queryLength < chunkLength` になり、その差が pad 行。**純関数**にしてある
 * のは、割り方の誤り（位置の飛び / 重なり / 末尾の取りこぼし）が GPU を回さずに落ちる位置に
 * あるべきだからで、`generateGreedy` の入口検査もここ 1 本に相乗りしている。
 */
export const planPrefillChunks = (
  promptLength: number,
  chunkLength: number,
): readonly PrefillChunk[] => {
  assertPositiveInteger(promptLength, "prompt の長さ");
  assertPositiveInteger(chunkLength, "chunkLength");
  const chunks: PrefillChunk[] = [];
  for (let position = 0; position < promptLength; position += chunkLength) {
    chunks.push({ position, queryLength: Math.min(chunkLength, promptLength - position) });
  }
  return chunks;
};

/** i32 の入力テンソル 1 本（token id 列も絶対位置列も `[1, rows]`）。 */
const i32Row = (rows: number, data: Int32Array<ArrayBuffer>): Tensor => ({
  dtype: "i32",
  shape: [1, rows],
  data,
});

/** token-only 形の last_row 入力（`[1]` の i32 — 選ぶ行の添字）。 */
const lastRowInput = (row: number): Tensor => ({
  dtype: "i32",
  shape: [1],
  data: Int32Array.of(row),
});

const prefillInputs = <C extends GenerationDisposable>(
  spec: GreedySpec<C>,
  chunk: PrefillChunk,
): RunInputs => {
  // MUST: pad 行は 0 のまま（ADR 0066 追記 6 の値契約 — pad 行の値が非有限だと「−inf 加算 +
  // exp」経路で valid 行へ漏れるので、ホストが 0 で埋める）。`Int32Array` の初期値がそのまま
  // その契約なので、埋めるのは先頭 queryLength 行だけでよい。
  const ids = new Int32Array(spec.chunkLength);
  const positions = new Int32Array(spec.chunkLength);
  for (let row = 0; row < chunk.queryLength; row += 1) {
    ids[row] = spec.prompt[chunk.position + row];
    positions[row] = chunk.position + row;
  }
  return {
    [spec.inputIds]: i32Row(spec.chunkLength, ids),
    // token-only 形は最終**有効**行を添字で選ぶ（pad 行の lm_head は走らない — ADR 0068 決定 4）。
    ...(spec.lastRow === undefined ? {} : { [spec.lastRow]: lastRowInput(chunk.queryLength - 1) }),
    ...(spec.derive === undefined ? {} : spec.derive(ids, positions)),
  };
};

const decodeInputs = <C extends GenerationDisposable>(
  spec: GreedySpec<C>,
  token: number,
  position: number,
): RunInputs => {
  const ids = Int32Array.of(token);
  const positions = Int32Array.of(position);
  return {
    [spec.inputIds]: i32Row(1, ids),
    ...(spec.lastRow === undefined ? {} : { [spec.lastRow]: lastRowInput(0) }),
    ...(spec.derive === undefined ? {} : spec.derive(ids, positions)),
  };
};

/**
 * argmax 出力の 1 行から token id を読む。
 *
 * MUST: dtype と形をここで検査する（fail loudly）。**出力名の取り違え**は、たまたま形の合う
 * 別の出力（logits など）を掴むと例外も警告も出ないまま誤った token 列を返すので、名前の
 * 引き当てと同時に「argmax token 列の形」であることを見る。
 */
const readToken = (outputs: RunOutputs, name: string, rows: number, row: number): number => {
  if (!Object.hasOwn(outputs, name)) throw new Error(`グラフ出力 '${name}' が無い`);
  const tensor = outputs[name];
  if (tensor.dtype !== "i32") {
    throw new Error(
      `グラフ出力 '${name}' が i32 でない（${tensor.dtype}） — argmax token 列を指していない`,
    );
  }
  if (
    tensor.shape.length !== 3 || tensor.shape[0] !== 1 || tensor.shape[1] !== rows ||
    tensor.shape[2] !== 1
  ) {
    throw new Error(
      `グラフ出力 '${name}' の形 [${tensor.shape.join(",")}] が [1,${rows},1] でない`,
    );
  }
  return tensor.data[row];
};

/**
 * プロンプトから `maxNewTokens` 個の token を greedy に生成する（ADR 0066 決定 4 の 2 形）。
 *
 * NOTE: **EOS 停止は載せない** — 固定 token id 列での parity 検収が目的で、停止条件を持つと
 * 「どこで止まったか」が比較対象に混ざる（実用の停止はこの返り値を見る呼び出し側の仕事）。
 */
export const generateGreedy = async <C extends GenerationDisposable>(
  spec: GreedySpec<C>,
): Promise<number[]> => {
  const { session, prompt, chunkLength, maxNewTokens, maxPosition } = spec;
  const chunks = planPrefillChunks(prompt.length, chunkLength);
  assertPositiveInteger(maxNewTokens, "maxNewTokens");
  assertPositiveInteger(maxPosition, "maxPosition");
  prompt.forEach((id, index) => {
    // `Int32Array` への書き込みは非整数の切り詰めも i32 値域外の wrap も**黙って**行う
    // （例: 2^32+1 → 1 — 別の有効 token id に化けて例外にならない）ので、入口で落とす。
    if (!Number.isSafeInteger(id) || id < 0 || id > 0x7fffffff) {
      throw new Error(`prompt[${index}] ${id} が 0..2147483647 の整数でない`);
    }
  });
  // 踏む最大の絶対位置 = 最終 decode step の T + maxNewTokens - 2（decode 0 回なら prefill の
  // T - 1 で、同じ式に畳まれる）。表の外は沈黙誤 token（maxPosition の JSDoc）なのでここで落とす。
  const lastPosition = prompt.length + maxNewTokens - 2;
  if (lastPosition >= maxPosition) {
    throw new Error(
      `prompt ${prompt.length} + maxNewTokens ${maxNewTokens} は最終位置 ${lastPosition} を踏む` +
        `（maxPosition ${maxPosition} の外 — この資産では位置 0..${maxPosition - 1} しか引けない）`,
    );
  }

  // MUST: 容量記号（states 専用記号）の束縛点は context 生成**だけ**で、run の bindings へは
  // 渡さない（ADR 0066 追記 7 — run の記号解決に効くのは入力由来の束縛だけ）。
  const context = await session.createGenerationContext({
    bindings: spec.bindings,
    chunkLength,
  });
  try {
    // token-only 形は出力が選ばれた 1 行ぶん（`[1,1,1]`）なので、読む位置は形も添字も固定。
    const tokenOnly = spec.lastRow !== undefined;
    let token = 0;
    for (const chunk of chunks) {
      const outputs = await session.run(prefillInputs(spec, chunk), undefined, {
        context,
        queryLength: chunk.queryLength,
      });
      // 生成の起点になるのは**最終 chunk の最終有効行**だけ（chunks は 1 本以上なので、この
      // 初期値は必ず上書きされる）。途中 chunk の出力は捨てる。
      token = tokenOnly
        ? readToken(outputs, spec.token, 1, 0)
        : readToken(outputs, spec.token, chunkLength, chunk.queryLength - 1);
    }

    const generated = [token];
    // decode は「位置 `T + i` に `g_i` を置くと `g_{i+1}` が出る」形。前 step の返り値を
    // そのまま次の入力にするのが greedy の feedback で、ここが唯一の状態。
    for (let index = 0; index + 1 < maxNewTokens; index += 1) {
      const outputs = await session.run(
        decodeInputs(spec, generated[index], prompt.length + index),
        undefined,
        { context, queryLength: 1 },
      );
      generated.push(readToken(outputs, spec.token, 1, 0));
    }
    return generated;
  } finally {
    // MUST: 途中で落ちても返す（失敗した context は poison 化していて再利用できない —
    // ADR 0066 追記 3。抱えたままにすると KV 容量ぶんの VRAM が生成 1 本ごとに積み上がる）。
    await context.dispose();
  }
};
