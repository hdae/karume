/**
 * WebGPU errorScope の張り方の規律と、そこから出る型付きエラー。
 *
 * WebGPU の失敗の大半は**同期例外にならない** — 無効なパイプライン / 無効なバッファが返り、
 * それを使う dispatch や writeBuffer が警告すら出さない no-op になる。この層の責務は
 * 「その沈黙を errorScope で捕まえて loud な例外に変換する」ことだけで、device の取得・能力の
 * 正規化（`device.ts`）とは独立している。
 *
 * MUST: この層は `device.ts` を import しない。errorScope の規律は device 単位の LIFO
 * スタックという WebGPU の性質だけで閉じており、GpuContext を知る必要が無い。分離してあるのは
 * それに加えて**実行時の循環を構造的に消す**ため — `PipelineCache` はここだけに依存するので、
 * GpuContext が PipelineCache を所有しても device.ts ⇄ pipeline-cache.ts の輪ができない。
 *
 * NOTE: `GpuDeviceLostError` 等の device 固有のエラーは `device.ts` に残る（errorScope は device
 * 消失を捕らえない — 消失後の `popErrorScope` は null で resolve する）。
 */

/** errorScope が捕捉した validation エラー（無効パイプライン等）。 */
export class GpuValidationError extends Error {
  override readonly name = "GpuValidationError";
}

/**
 * errorScope が捕捉した out-of-memory エラー（確保要求が device の余力を超えた）。
 * validation と違い利用者のモデルサイズと実行環境で決まるため、別型で分岐可能にする。
 */
export class GpuOutOfMemoryError extends Error {
  override readonly name = "GpuOutOfMemoryError";
}

/**
 * errorScope が捕捉した internal エラー（実装側の都合で操作が失敗した — シェーダが複雑すぎて
 * コンパイルできない等）。WGSL 自体は妥当なので validation とは分岐先が違う: 利用者にとって
 * 「記述が不正」ではなく「この環境ではこの形のカーネルが通らない」であり、別型で区別できないと
 * codegen のバグと環境の限界が同じ報告に潰れる。
 */
export class GpuInternalError extends Error {
  override readonly name = "GpuInternalError";
}

/**
 * internal + validation の 2 本を張って `body`（パイプライン生成）を実行し、捕捉したエラーを
 * 例外に変換する。**パイプライン生成経路はこちらを使う**（validation 1 本では internal
 * エラーが素通りする）。
 *
 * 両方が要るのは失敗の出方が違うため — 上限超過や型不整合は validation、実装側の都合
 * （シェーダが複雑すぎてコンパイルできない等）は internal で、どちらも同期例外にならず
 * **無効なパイプラインが返るだけ**。囲まないと dispatch が no-op 化して出力が全て 0 になる。
 *
 * MUST NOT: `body` の中で非同期処理を待たない。errorScope はスタックで、pop は body の同期
 * 実行直後に起きるため、await 後にエンコードした操作は別のスコープに吸われる。
 */
export const withPipelineScope = async <T>(
  device: GPUDevice,
  label: string,
  body: () => T,
): Promise<T> => {
  device.pushErrorScope("internal");
  device.pushErrorScope("validation");
  let value: T;
  try {
    value = body();
  } catch (cause) {
    // MUST: body が throw しても 2 本とも pop する。片方でも積み残すと後続の検証結果が誤った
    // スコープに吸われ、以後のエラーが恒久的に見えなくなる。pop 自体の失敗は握り潰す
    // （後始末で本体の例外を上書きしない — discardFailureScopes と同じ規律）。
    const validation = device.popErrorScope().catch(() => null);
    const internal = device.popErrorScope().catch(() => null);
    await Promise.all([validation, internal]);
    throw cause;
  }
  // MUST: 2 本の pop は**同一同期区間で発行**する（await するのは発行済みの promise だけ）。
  // pop はスタック先頭を無条件に取るため、発行の間に await を挟むと、その隙に他所が push した
  // スコープを 2 本目が取り、失敗が誤帰属する（popFailureScopes と同じ規律）。
  const validation = device.popErrorScope();
  const internal = device.popErrorScope();
  const [validationError, internalError] = await Promise.all([validation, internal]);
  // MUST: 両方が捕捉されたときは internal を返す。internal で無効化されたパイプラインを触る
  // 後続の操作（`getBindGroupLayout` 等）は**派生の** validation エラーを立てるため、
  // validation を先に返すと根因の internal が捨てられ、原因の分からない「無効パイプライン」
  // として報告される。OOM を validation より先に返すのと同型の判断
  // （docs/research/2026-08-08-vram-oom-misreport.md）。
  if (internalError !== null) {
    throw new GpuInternalError(`${label}: ${internalError.message}`);
  }
  if (validationError !== null) {
    throw new GpuValidationError(`${label}: ${validationError.message}`);
  }
  return value;
};

/**
 * out-of-memory + validation の 2 本を張る。確保を伴う区間はこの両建てで囲む。
 *
 * 両方が要るのは失敗の出方が違うため — 上限超過の `createBuffer` は validation、device の
 * 余力切れは out-of-memory で、どちらも同期例外にはならず**無効なバッファが返るだけ**。
 * 囲まないと、そこへの `writeBuffer` が警告すら出さない no-op になり、空の重みや空の中間
 * バッファのまま処理が続く。対応する pop は必ず {@link popFailureScopes} /
 * {@link discardFailureScopes} を使う（pop の順序と同期性がここに閉じている）。
 */
export const pushFailureScopes = (device: GPUDevice): void => {
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
};

/**
 * {@link pushFailureScopes} の 2 本を pop し、捕捉した失敗を型付き例外にして返す
 * （何も捕捉しなければ undefined）。
 *
 * MUST: 2 本の pop は**同一同期区間で発行**する（await するのは発行済みの promise だけ）。
 * pop はスタック先頭を無条件に取るため、発行の間に await を挟むと、その隙に他所が push した
 * スコープを 2 本目が取り、失敗が誤帰属する。
 *
 * MUST: 両方が捕捉されたときは out-of-memory を返す。確保が余力切れで失敗すると `createBuffer`
 * は無効なバッファを返し、それを使う後続の `createBindGroup` / `writeBuffer` が
 * `Buffer with '' label is invalid` という**派生の** validation エラーを立てる。validation を
 * 先に返すと根因の OOM が捨てられ、区別のつかない「無効バッファ」として報告される
 * （docs/research/2026-08-08-vram-oom-misreport.md）。
 */
export const popFailureScopes = async (
  device: GPUDevice,
  label: string,
): Promise<GpuValidationError | GpuOutOfMemoryError | undefined> => {
  const validation = device.popErrorScope();
  const outOfMemory = device.popErrorScope();
  const [validationError, outOfMemoryError] = await Promise.all([validation, outOfMemory]);
  if (outOfMemoryError !== null) {
    return new GpuOutOfMemoryError(`${label}: ${outOfMemoryError.message}`);
  }
  if (validationError !== null) {
    return new GpuValidationError(`${label}: ${validationError.message}`);
  }
  return undefined;
};

/**
 * 失敗経路の後始末。{@link pushFailureScopes} の 2 本を pop して結果を捨てる。
 *
 * MUST: 本体が例外で抜けてもスコープは必ず 2 本とも pop する。積み残すと以後の検証結果が
 * 誤ったスコープに吸われ、エラーが恒久的に見えなくなる。pop 自体の失敗も握り潰す
 * （後始末で本体の例外を上書きしない）。
 */
export const discardFailureScopes = async (device: GPUDevice): Promise<void> => {
  const validation = device.popErrorScope().catch(() => null);
  const outOfMemory = device.popErrorScope().catch(() => null);
  await Promise.all([validation, outOfMemory]);
};
