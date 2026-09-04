// fusion-hints inductor の引数解析（GPU・venv 不要）。

import { assertEquals, assertThrows } from "@std/assert";
import { inductorArgs, parseInductorArgs } from "./inductor.ts";

Deno.test("parseInductorArgs: --candidates は繰り返し、--out は必須、未知オプションは落ちる", () => {
  const options = parseInductorArgs([
    "--out",
    "o",
    "--candidates",
    "a.jsonl",
    "--candidates",
    "b.jsonl",
    "--venv",
    "/v",
  ]);
  assertEquals(options.out, "o");
  assertEquals(options.candidates, ["a.jsonl", "b.jsonl"]);
  assertEquals(options.venv, "/v");
  assertEquals(inductorArgs(options), [
    "--out",
    "o",
    "--candidates",
    "a.jsonl",
    "--candidates",
    "b.jsonl",
  ]);
  assertThrows(() => parseInductorArgs(["--candidates", "a.jsonl"]), Error, "--out は必須");
  assertThrows(
    () => parseInductorArgs(["--out", "o", "--candidate", "a"]),
    Error,
    "未知のオプション",
  );
  assertThrows(() => parseInductorArgs(["--out", "--candidates"]), Error, "対になっていない");
});
