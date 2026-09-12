import { assertEquals } from "@std/assert";
import { formatGenerationTiming, generationTimer } from "./generation-timing.ts";

Deno.test("生成計測: TTFT を除く token 数と時間で decode 速度を算出する", () => {
  let at = 100;
  const timer = generationTimer(() => at);
  at = 600;
  timer.onToken();
  at = 800;
  timer.onToken();
  at = 1000;
  timer.onToken();
  at = 1100;
  assertEquals(timer.finish(), {
    elapsedMs: 1000,
    generatedTokens: 3,
    ttftMs: 500,
    decodeTokensPerSecond: 4,
  });
});
Deno.test("生成計測: 即 EOS・中断・1 token は未測定の値を数値にしない", () => {
  let at = 0;
  const timer = generationTimer(() => at);
  at = 100;
  assertEquals(timer.finish(), {
    elapsedMs: 100,
    generatedTokens: 0,
    ttftMs: null,
    decodeTokensPerSecond: null,
  });
  timer.onToken();
  at = 200;
  assertEquals(formatGenerationTiming(timer.finish()), "TTFT 100 ms · decode — · total 0.20s");
});
Deno.test("生成計測: 同時刻の token 群は無限大にならず、時刻0も失わない", () => {
  const timer = generationTimer(() => 0);
  timer.onToken();
  timer.onToken();
  assertEquals(timer.finish(), {
    elapsedMs: 0,
    generatedTokens: 2,
    ttftMs: 0,
    decodeTokensPerSecond: null,
  });
});
