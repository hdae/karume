// opbench CLI の引数の門（GPU 不要）。fusion-hints 側は同等を enumerate_test.ts が押さえている。
//
// 「打ち間違いを黙って通さない」のがこの道具の設計目的なので、対の崩れ・値の書き忘れ・
// 未知オプション・同キーの 2 度渡し・未知サブコマンドの 5 形を固定する。

import { assertEquals, assertThrows } from "@std/assert";
import { CENSUS_OPTIONS, parseArgs, single } from "./main.ts";

Deno.test("parseArgs: '--キー 値' の対になっていない並びは落ちる", () => {
  assertThrows(
    () => parseArgs(["--source"], CENSUS_OPTIONS),
    Error,
    "'--キー 値' の対になっていない",
  );
});

Deno.test("parseArgs: 値の書き忘れ（次がオプション）は落ちる", () => {
  assertThrows(
    () => parseArgs(["--source", "--out"], CENSUS_OPTIONS),
    Error,
    "値が無い（'--out' はオプション）",
  );
});

Deno.test("parseArgs: 未知のオプションは USAGE つきで落ちる", () => {
  const error = assertThrows(
    () => parseArgs(["--scenarios", "x"], CENSUS_OPTIONS),
    Error,
    "未知のオプション --scenarios",
  );
  assertEquals(error.message.includes("使い方: deno run -A tools/opbench/main.ts"), true);
});

Deno.test("parseArgs: 繰り返してよいキー（--scenario）は全ての値が残る", () => {
  const args = parseArgs(["--scenario", "a=M:1", "--scenario", "b=M:2"], CENSUS_OPTIONS);
  assertEquals(args.get("scenario"), ["a=M:1", "b=M:2"]);
});

Deno.test("single: 1 度しか指定できないキーを 2 度渡すと落ちる", () => {
  const args = parseArgs(["--source", "a", "--source", "b"], CENSUS_OPTIONS);
  assertThrows(() => single(args, "source"), Error, "--source は 1 度しか指定できない");
});

Deno.test("CLI: 未知のサブコマンドは USAGE を stderr へ出して終了コード 2", async () => {
  const entry = new URL("./main.ts", import.meta.url);
  const { code, stderr } = await new Deno.Command(Deno.execPath(), {
    args: ["run", "-A", entry.href, "cencus"],
    stdout: "null",
    stderr: "piped",
  }).output();
  assertEquals(code, 2);
  assertEquals(new TextDecoder().decode(stderr).includes("使い方: deno run -A"), true);
});
