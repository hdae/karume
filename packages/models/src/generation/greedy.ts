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
  /** 絶対位置を受けるグラフ入力の名前（`[1,M]` の i32）。 */
  readonly positionIds: string;
  /** argmax token を出すグラフ出力の名前（`[1,M,1]` の i32）。 */
  readonly token: string;
  /** 固定長 prefill chunk の行数（ADR 0066 決定 4 — context の計画時定数）。 */
  readonly chunkLength: number;
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
    [spec.positionIds]: i32Row(spec.chunkLength, positions),
  };
};

const decodeInputs = <C extends GenerationDisposable>(
  spec: GreedySpec<C>,
  token: number,
  position: number,
): RunInputs => ({
  [spec.inputIds]: i32Row(1, Int32Array.of(token)),
  [spec.positionIds]: i32Row(1, Int32Array.of(position)),
});

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
  const { session, prompt, chunkLength, maxNewTokens } = spec;
  const chunks = planPrefillChunks(prompt.length, chunkLength);
  assertPositiveInteger(maxNewTokens, "maxNewTokens");
  prompt.forEach((id, index) => {
    // `Int32Array` への書き込みは非整数を黙って切り詰めるので、入口で落とす。
    if (!Number.isSafeInteger(id)) throw new Error(`prompt[${index}] ${id} が整数でない`);
  });

  // MUST: 容量記号（states 専用記号）の束縛点は context 生成**だけ**で、run の bindings へは
  // 渡さない（ADR 0066 追記 7 — run の記号解決に効くのは入力由来の束縛だけ）。
  const context = await session.createGenerationContext({
    bindings: spec.bindings,
    chunkLength,
  });
  try {
    let token = 0;
    for (const chunk of chunks) {
      const outputs = await session.run(prefillInputs(spec, chunk), undefined, {
        context,
        queryLength: chunk.queryLength,
      });
      // 生成の起点になるのは**最終 chunk の最終有効行**だけ（chunks は 1 本以上なので、この
      // 初期値は必ず上書きされる）。途中 chunk の出力は捨てる。
      token = readToken(outputs, spec.token, chunkLength, chunk.queryLength - 1);
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
