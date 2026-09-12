/** デモの壁時計計測。停止 token を除き、最初の token の到着と以降の速度を分ける。 */
export type GenerationTiming = {
  readonly elapsedMs: number;
  readonly generatedTokens: number;
  readonly ttftMs: number | null;
  readonly decodeTokensPerSecond: number | null;
};

/** 時計は生成開始直前に作る。トークンの復号や端末への出力より先に onToken を呼ぶ。 */
export const generationTimer = (now: () => number = () => performance.now()): {
  readonly onToken: () => void;
  readonly finish: () => GenerationTiming;
} => {
  const started = now();
  let first: number | undefined;
  let count = 0;
  return {
    onToken: () => {
      first ??= now();
      count += 1;
    },
    finish: () => {
      const ended = now();
      return {
        elapsedMs: ended - started,
        generatedTokens: count,
        ttftMs: first === undefined ? null : first - started,
        decodeTokensPerSecond: first === undefined || count <= 1 || ended <= first
          ? null
          : (count - 1) * 1000 / (ended - first),
      };
    },
  };
};

export const formatGenerationTiming = (timing: GenerationTiming): string =>
  `TTFT ${timing.ttftMs === null ? "—" : `${timing.ttftMs.toFixed(0)} ms`}` +
  ` · decode ${
    timing.decodeTokensPerSecond === null ? "—" : `${timing.decodeTokensPerSecond.toFixed(1)} tok/s`
  }` +
  ` · total ${(timing.elapsedMs / 1000).toFixed(2)}s`;
