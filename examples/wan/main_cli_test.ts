// examples/wan の台本（`main.ts`）の I2V のノブの拒否の経路。台本はトップレベルで走るので、サブプロセスで起動して
// 終了コードと出力を見る。拒否は配布形や画像を読む前に起きる — 画像のパスは存在しないものを渡し、読みに行けば
// NotFound で別の文言になることで「読む前に落ちた」を縛る。GPU も実資産も要らない。
//
// 条件画像の復号は配布形の読み込みより先（`main.ts` の main の冒頭 — 壊れた・対応外の画像を重みの読み込みの後で
// 知らない）。これは空の一時ディレクトリを cwd にして縛る: 既定の配布形ミラー（cwd からの相対パス）がそこには
// 無いので、読み込みへ進めば「ミラーが無い」で落ちる — 手元にミラーがある機でも GPU と実資産に触らない。

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

/**
 * file URL を実パスへ（`pathname` は空白・非 ASCII を %エンコードしたままなので、読み込みと文言の照合が
 * 別の理由で落ちる — リポの他の台本と同じ書き方）。
 */
const filePath = (url: URL): string => decodeURIComponent(url.pathname);

/** 対応外の画像（GIF）の fixture。 */
const GIF = new URL("./fixtures/gif.gif", import.meta.url);
/** 7×5 の RGB の PNG の fixture。 */
const RGB_PNG = new URL("./fixtures/rgb.png", import.meta.url);
/** リポの設定（import map — cwd を移すと台本から見つからないので明示する）。 */
const CONFIG = new URL("../../deno.json", import.meta.url);

/** 空の一時ディレクトリを cwd にして起動する（既定の配布形ミラーが見つからない場所）。 */
const runInEmptyDirectory = async (args: readonly string[]) => {
  const cwd = await Deno.makeTempDir({ prefix: "karume-wan-cli-" });
  try {
    const output = await new Deno.Command(Deno.execPath(), {
      args: ["run", "-A", "--config", filePath(CONFIG), ENTRY.href, ...args],
      cwd,
      stdout: "piped",
      stderr: "piped",
    }).output();
    return {
      code: output.code,
      stdout: DECODER.decode(output.stdout),
      stderr: DECODER.decode(output.stderr),
    };
  } finally {
    await Deno.remove(cwd, { recursive: true });
  }
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

describe("main.ts — 条件画像の復号は配布形の読み込みより先", () => {
  it("対応外の画像（GIF）は、配布形の解決の前に対応形式を名指しして落ちる", async () => {
    const { code, stdout, stderr } = await runInEmptyDirectory([
      "--generation",
      "wan2.2",
      "--image",
      filePath(GIF),
    ]);
    assertNotEquals(code, 0);
    assertStringIncludes(stderr, "対応形式は PNG");
    assert(!stderr.includes("配布形ミラー"), `配布形を解決してから画像を復号した: ${stderr}`);
    assert(!stdout.includes("[wan] image") && !stdout.includes("[wan] source"), stdout);
  });

  it("正常な画像は復号して寸法を見せてから、配布形の解決へ進む（ここではミラーが無くて落ちる）", async () => {
    const { code, stdout, stderr } = await runInEmptyDirectory([
      "--generation",
      "wan2.2",
      "--image",
      filePath(RGB_PNG),
    ]);
    assertNotEquals(code, 0);
    assertStringIncludes(
      stdout,
      `[wan] image: ${filePath(RGB_PNG)}（7×5・fit: パイプラインの既定）`,
    );
    assertStringIncludes(stderr, "配布形ミラー models/karume-wan2.2 に karume.json が無い");
    assert(!stdout.includes("[wan] source"), stdout);
  });
});
