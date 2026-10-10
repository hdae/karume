// errorScope の pop に「非致命 OOM」を注入する（実 GPU の device を包む）。
//
// 非致命 OOM = `createBuffer` が例外を投げずに無効なバッファを返し、失敗は後の
// `popErrorScope`（out-of-memory スコープ）で初めて判明する状態。実 GPU ではこの状況を
// 狙って作れないので、out-of-memory スコープの pop 結果だけを差し替える（生産コードに
// 注入面を開けないための代替 — `gpu_plan_backing_test.ts` の `injectStorageFault` と同型）。
//
// MUST: 実 pop は必ず**同期に**発行する。握り潰すと push した分が device の LIFO に残り、
// 以後の失敗が誤ったスコープに吸われる。`popFailureScopes` は 2 本の pop を同一同期区間で
// 発行する規約なので、包んだ側も最初の await より前に実 pop を出す。
// MUST: push を包んで filter のスタックを持つ。pop は filter を受け取らないので、どの pop が
// out-of-memory スコープかは push 側の記録からしか分からない（パイプライン生成の
// validation / internal スコープを撃たないため）。

/** {@link injectOutOfMemory} の操作面。 */
export type OutOfMemoryInjection = {
  /** 次に pop される out-of-memory スコープ 1 本に OOM を載せる。 */
  readonly arm: () => void;
  /** 注入が実際に効いた回数（門が空振っていないことの確認に使う）。 */
  readonly fired: () => number;
  readonly restore: () => void;
};

export const injectOutOfMemory = (device: GPUDevice): OutOfMemoryInjection => {
  const originalPush = device.pushErrorScope;
  const originalPop = device.popErrorScope;
  const push = originalPush.bind(device);
  const pop = originalPop.bind(device);
  // 注入を張る前に push されたスコープは記録に無い（`undefined` で pop される）。
  const filters: GPUErrorFilter[] = [];
  let armed = false;
  let fired = 0;
  device.pushErrorScope = (filter: GPUErrorFilter): undefined => {
    filters.push(filter);
    return push(filter);
  };
  device.popErrorScope = (): Promise<GPUError | null> => {
    const filter = filters.pop();
    const real = pop();
    if (!armed || filter !== "out-of-memory") return real;
    armed = false;
    fired += 1;
    return real.then((error) => error ?? new GPUOutOfMemoryError("注入: 非致命 OOM"));
  };
  return {
    arm: (): void => {
      armed = true;
    },
    fired: (): number => fired,
    restore: (): void => {
      device.pushErrorScope = originalPush;
      device.popErrorScope = originalPop;
    },
  };
};
