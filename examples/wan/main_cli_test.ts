// examples/wan の台本（`main.ts`）の I2V のノブの拒否の経路。台本はトップレベルで走るので、サブプロセスで起動して
// 終了コードと出力を見る。拒否は配布形や画像を読む前に起きる — 画像のパスは存在しないものを渡し、読みに行けば
// NotFound で別の文言になることで「読む前に落ちた」を縛る。GPU も実資産も要らない。

import { assert, assertNotEquals, assertStringIncludes } from "@std/assert";
import { describe, it } from "@std/testing/bdd";

const ENTRY = new URL("./main.ts", import.meta.url);
const DECODER = new TextDecoder();
/** 存在しない画像のパス（読みに行けば NotFound になる）。 */
const MISSING_IMAGE = "does-not-exist-9c.png";

const run = async (args: readonly string[]) => {
  const output = await new Deno.Command(Deno.execPath(), {
    args: ["run", "-A", ENTRY.href, ...args],
    stdout: "piped",
    stderr: "piped",
  }).output();
  return {
    code: output.code,
    stdout: DECODER.decode(output.stdout),
    stderr: DECODER.decode(output.stderr),
  };
};

/** 拒否されたこと・その文言・画像も配布形も読んでいないこと。 */
const assertRejectedBeforeLoading = async (args: readonly string[], message: string) => {
  const { code, stdout, stderr } = await run(args);
  assertNotEquals(code, 0);
  assertStringIncludes(stderr, message);
  assert(!stderr.includes("NotFound"), `画像を読みに行った: ${stderr}`);
  assert(!stdout.includes("[wan] image") && !stdout.includes("[wan] source"), stdout);
};

describe("main.ts — I2V のノブの拒否（読み込みの前）", () => {
  it("--generation wan2.1 の --image を拒む", async () => {
    await assertRejectedBeforeLoading(
      ["--generation", "wan2.1", "--image", MISSING_IMAGE],
      "--image は --generation wan2.1 では効かない",
    );
  });

  it("--generation wan2.1 の --fit を拒む", async () => {
    await assertRejectedBeforeLoading(
      ["--generation", "wan2.1", "--fit", "crop"],
      "--fit は --generation wan2.1 では効かない",
    );
  });

  it("--image 無しの --fit を拒む", async () => {
    await assertRejectedBeforeLoading(
      ["--generation", "wan2.2", "--fit", "stretch"],
      "--fit stretch は --image 無しでは効かない",
    );
  });

  it("--fit の綴り違いを拒む", async () => {
    await assertRejectedBeforeLoading(
      ["--generation", "wan2.2", "--image", MISSING_IMAGE, "--fit", "fill"],
      "--fit fill が crop / stretch のどちらでもない",
    );
  });
});
