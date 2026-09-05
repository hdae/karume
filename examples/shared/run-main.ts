/**
 * デモ台本の最上位の包み — 本体を関数の中に閉じ、落ちた理由を 1 つ残らず stderr へ出す。
 *
 * ## なぜ包む必要があるのか
 *
 * `using` / `await using` は、本体が投げたうえに解放も投げると両方を `SuppressedError` に畳む
 * （`.error` = 解放側 / `.suppressed` = 本体側）。Deno の未捕捉ハンドラはその**外皮しか印字
 * しない**ため、モジュール本体（トップレベル）で `await using` を掴んだ台本は、device 消失の
 * ように「本体と解放が同じ理由で落ちる」故障で型も文言も画面に残さない。トップレベルの
 * `using` は自分では捕まえられない（モジュール本体の外に catch を置けない）ので、
 * **`await using` を必ず関数の中へ移して** {@link runMain} に渡すのが唯一の確実な形である。
 */

const encoder = new TextEncoder();
const note = (text: string): void => {
  Deno.stderr.writeSync(encoder.encode(text));
};

/**
 * 例外を 1 つ残らず stderr へ展開する（型名 + 文言 + 入れ子の深さ）。
 *
 * `SuppressedError` / `AggregateError` / `cause` 連鎖の 3 経路を辿る。
 * MUST: 中身を 1 つも落とさない。ここは台本の最後の出力口で、握り潰せば以後どこにも出ない。
 */
export const printError = (error: unknown, depth = 0): void => {
  const indent = "  ".repeat(depth);
  if (!(error instanceof Error)) {
    note(`${indent}[${depth}] ${typeof error}: ${String(error)}\n`);
    return;
  }
  note(`${indent}[${depth}] ${error.name}: ${error.message}\n`);
  // 外皮のスタックだけは残す（Deno の既定の未捕捉出力と同じ情報量を下回らないため）。
  // 1 行目は「名前: 文言」の写しなので落とす — 直前の行と重複する。
  if (depth === 0 && error.stack !== undefined) {
    const frames = error.stack.slice(error.stack.indexOf("\n") + 1);
    if (frames !== error.stack) note(`${frames}\n`);
  }
  if (error instanceof SuppressedError) {
    note(`${indent}  ↳ error（解放側）\n`);
    printError(error.error, depth + 1);
    note(`${indent}  ↳ suppressed（本体側）\n`);
    printError(error.suppressed, depth + 1);
  }
  if (error instanceof AggregateError) {
    for (const [at, inner] of error.errors.entries()) {
      note(`${indent}  ↳ errors[${at}]\n`);
      printError(inner, depth + 1);
    }
  }
  if (error.cause !== undefined) {
    note(`${indent}  ↳ cause\n`);
    printError(error.cause, depth + 1);
  }
};

/**
 * 台本の本体を走らせ、落ちたら {@link printError} で展開して終了コード 1 で終わる。
 *
 * MUST: 呼び手は `using` / `await using` を **`main` の中**に置く（外に残すと畳まれた
 * `SuppressedError` はここまで届かない）。
 */
export const runMain = async (main: () => Promise<void>): Promise<void> => {
  try {
    await main();
  } catch (error) {
    printError(error);
    Deno.exit(1);
  }
};
