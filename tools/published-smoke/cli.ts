/**
 * published-smoke の純ロジック（引数の門と pin エントリの型ガード）。
 *
 * `main.ts` は起動と同時に registry から `jsr:@karume/*` を動的 import するので、テストから
 * import できない（ネットワークと公開物が要る）。門だけをここへ置いて `main_test.ts` が見る。
 */

/** 取得元対応表の 1 エントリ（`{repo, revision}` の pin — ADR 0092）。 */
export type PinRef = { readonly repo: string; readonly revision: string };

export const isPinRef = (value: unknown): value is PinRef =>
  typeof value === "object" && value !== null &&
  typeof (value as { repo?: unknown }).repo === "string" &&
  typeof (value as { revision?: unknown }).revision === "string";

/** 起動の 2 通り（README の起動例と同じ綴り）。 */
export const USAGE = "deno task smoke:published [--manifests-only]";

/** 受ける引数はこれだけ（`--manifests-only` = GPU の要る fromPretrained を省く）。 */
const KNOWN: ReadonlySet<string> = new Set(["--manifests-only"]);

/**
 * 引数を読む。
 *
 * MUST: 未知の引数は落とす（opbench / fusion-hints / inductor と同じ規律）。
 * `--manifest-only`（単数形）のような打ち間違いを黙って捨てると、省いたつもりの GPU 込みの
 * 完全経路へ静かに落ちる。GPU の無い機体では sbv2 の `fromPretrained` が別の理由で落ちるので、
 * 公開直後の疎通が「公開物が壊れている」という**偽の赤**として読まれる。
 */
export const parseSmokeArgs = (
  argv: readonly string[],
): { readonly manifestsOnly: boolean } => {
  for (const arg of argv) {
    if (!KNOWN.has(arg)) throw new Error(`未知の引数 ${arg}（使い方: ${USAGE}）`);
  }
  return { manifestsOnly: argv.includes("--manifests-only") };
};
