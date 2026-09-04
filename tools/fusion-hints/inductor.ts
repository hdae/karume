/**
 * fusion-hints `inductor` の Deno 側 — CUDA venv の python で inductor_probe.py を走らせる。
 * 引数解析は純関数（テストで見る）。実体と方法は inductor_probe.py の冒頭。
 */

import { defaultVenv, runVenvPython } from "../_shared/python.ts";

export type InductorOptions = {
  readonly out: string;
  readonly candidates: readonly string[];
  readonly venv: string;
};

const INDUCTOR_USAGE =
  `使い方: deno run -A tools/fusion-hints/main.ts inductor --out <dir> [--candidates <candidates.jsonl>]…
  --out <dir>                 inductor.jsonl / comparison.jsonl / summary.json の書き出し先
  --candidates <file>         enumerate の candidates.jsonl（繰り返し可 — 資産ごとに 1 本）
  --venv <dir>                CUDA venv（既定 = KARUME_CUDA_VENV か ~/workspace/karume-cuda-venv）`;

/** `inductor` サブコマンドの引数（argv[0] は "inductor" 済み）。未知のオプションは落とす。 */
export const parseInductorArgs = (argv: readonly string[]): InductorOptions => {
  let out: string | undefined;
  let venv: string | undefined;
  const candidates: string[] = [];
  for (let at = 0; at < argv.length; at += 2) {
    const key = argv[at];
    const value = argv[at + 1];
    if (!key.startsWith("--") || value === undefined || value.startsWith("--")) {
      throw new Error(`引数 ${key} が '--キー 値' の対になっていない。\n${INDUCTOR_USAGE}`);
    }
    switch (key) {
      case "--out":
        if (out !== undefined) throw new Error("--out は 1 度しか指定できない");
        out = value;
        break;
      case "--venv":
        if (venv !== undefined) throw new Error("--venv は 1 度しか指定できない");
        venv = value;
        break;
      case "--candidates":
        candidates.push(value);
        break;
      default:
        throw new Error(`未知のオプション ${key}。\n${INDUCTOR_USAGE}`);
    }
  }
  if (out === undefined) throw new Error(`--out は必須。\n${INDUCTOR_USAGE}`);
  return { out, candidates, venv: venv ?? defaultVenv() };
};

/** python の argv（純関数）。 */
export const inductorArgs = (options: InductorOptions): readonly string[] => [
  "--out",
  options.out,
  ...options.candidates.flatMap((path) => ["--candidates", path]),
];

export const runInductor = async (options: InductorOptions): Promise<void> => {
  await runVenvPython(
    options.venv,
    new URL("./inductor_probe.py", import.meta.url),
    inductorArgs(options),
  );
};
