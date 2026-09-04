/**
 * CUDA venv の python を subprocess として呼ぶ共通口（opbench `torch` / fusion-hints `inductor`）。
 *
 * venv は引数 → 環境変数 KARUME_CUDA_VENV → 既定 ~/workspace/karume-cuda-venv（GPU 校正が使う隔離
 * venv — research 2026-08-28-cuda-calibration）。python が無ければ fail loudly — 別の python へ
 * 黙って落ちない（torch の版が違えば数値も融合判定も別物になる）。
 */

export const defaultVenv = (): string =>
  Deno.env.get("KARUME_CUDA_VENV") ??
    `${Deno.env.get("HOME") ?? "/home/developer"}/workspace/karume-cuda-venv`;

/** `<venv>/bin/python <script> <args…>` を走らせる（stdout / stderr はそのまま流す）。 */
export const runVenvPython = async (
  venv: string,
  script: URL,
  args: readonly string[],
): Promise<void> => {
  const python = `${venv}/bin/python`;
  try {
    await Deno.stat(python);
  } catch {
    throw new Error(
      `${python} が無い — --venv か KARUME_CUDA_VENV で CUDA venv を指す（既定 ~/workspace/karume-cuda-venv）`,
    );
  }
  const command = new Deno.Command(python, {
    args: [script.pathname, ...args],
    stdout: "inherit",
    stderr: "inherit",
  });
  const status = await command.output();
  if (!status.success) {
    throw new Error(`${script.pathname} が code ${status.code} で終了した（上の stderr が理由）`);
  }
};
