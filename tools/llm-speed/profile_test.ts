// `llm-speed` が受ける profile の門（GPU 不要）。
//
// この比較器は**配布形の Gemma 4 だけ**を測る。系列出力（`outputs/series/`）を直に読む枝は
// 2026-09-23 に落とした — 旧形のまま残す研究記録の系列（`-probe`）は容器（krm）に変換せず、
// 読める profile も `tools/llm-baseline/data.py` から外したので、呼び手のいない枝だった。
//
// 枝を消したこと自体は型検査では守れない（profile は JSON = `unknown` 境界から来る）。
// ここは「読めない family を渡したら、黙って落ちるのではなく名指しで止まる」ことを固定する。

import { assertEquals, assertThrows } from "@std/assert";
import { profileFrom } from "./main.ts";

const inputs = (profile: Record<string, unknown>): unknown => ({
  model: "gemma4-e2b",
  profile,
});

Deno.test("profileFrom は配布形の gemma4 / gemma4-qat を読む", () => {
  const parsed = profileFrom(inputs({
    family: "gemma4-qat",
    model: "e4b",
    checkpoint: "/inputs/gemma4/qat-e4b/",
    distribution: "/models/karume-gemma4-qat",
  }));
  assertEquals(parsed.family, "gemma4-qat");
  assertEquals(parsed.model, "e4b");
  assertEquals(parsed.source, "/models/karume-gemma4-qat");
  // checkpoint の末尾 `/` を畳んで tokenizer を導く。
  assertEquals(parsed.tokenizer, "/inputs/gemma4/qat-e4b/tokenizer.json");
});

Deno.test("profileFrom は系列出力の profile を名指しで拒む（退役した枝の復活を許さない）", () => {
  const error = assertThrows(
    () =>
      profileFrom(inputs({
        family: "minicpm5",
        checkpoint: "/inputs/minicpm5/MiniCPM5-2B/",
        series: "outputs/series/minicpm5-2b-gptq-i4-2026-09-10-probe",
      })),
    Error,
    "minicpm5",
  );
  assertEquals(error.message.includes("data.py"), true, error.message);
});

Deno.test("profileFrom は配布形の置き場と model の綴りを欠いた profile を落とす", () => {
  assertThrows(
    () => profileFrom(inputs({ family: "gemma4", model: "e2b", checkpoint: "/c/" })),
    Error,
    "distribution",
  );
  assertThrows(
    () =>
      profileFrom(inputs({
        family: "gemma4",
        model: "e8b",
        checkpoint: "/c/",
        distribution: "/models/karume-gemma4",
      })),
    Error,
    "e2b / e4b",
  );
});
