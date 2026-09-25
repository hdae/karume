// hf-upload.zsh の断片化表の門（TG3）: 検証できなかった part を健全な行に化けさせない。
//
// 台本を一時ディレクトリへ `tools/release/` の形で写し（台本は自分の位置から 2 段上をリポ根と
// みなす）、`models/<name>/` に part を置いて `check` を走らせる。network へは出ない — PATH の
// 先頭に置いた stub の `curl` が HF / CAS の応答を演じる（`deno eval` は本物を使う）。
//
// zsh が無い環境（CI の ubuntu ランナーなど）は明示 SKIP する（ADR 0005）。

import { assert, assertEquals, assertNotEquals } from "@std/assert";

const hasZsh = (): boolean => {
  try {
    return new Deno.Command("zsh", { args: ["-c", "true"] }).outputSync().success;
  } catch (error) {
    if (error instanceof Deno.errors.NotFound) return false;
    throw error;
  }
};

const ZSH_AVAILABLE = hasZsh();
if (!ZSH_AVAILABLE) {
  console.warn("[karume] zsh が無いため tools/release/hf-upload.zsh の門を SKIP する");
}

const SCRIPT = new URL("./hf-upload.zsh", import.meta.url);
const NAME = "karume-stub";

/**
 * HF / CAS を演じる curl。part の名前で応答を決める:
 * `missing.krm` = 404（`-f` で終了コード 22）/ `small.krm`・`large-plain.krm` = xet ハッシュの無い
 * 200 / それ以外 = 名前（拡張子抜き）をハッシュに名乗る xet の part。reconstruction は
 * ハッシュ `empty` だけ term 0 本、他は 2 term（1 xorb）。`STUB_TOKEN=fail` で token が 401。
 */
const STUB_CURL = `#!/bin/sh
url=""
for arg in "$@"; do
  case "$arg" in
    http*) url="$arg" ;;
  esac
done
case "$url" in
  *xet-read-token*)
    if [ "\${STUB_TOKEN:-ok}" != ok ]; then echo "curl: (22) 401" >&2; exit 22; fi
    echo '{"casUrl":"https://cas.invalid","accessToken":"token"}' ;;
  */resolve/main/missing.krm) echo "curl: (22) 404" >&2; exit 22 ;;
  */resolve/main/small.krm|*/resolve/main/large-plain.krm) printf 'HTTP/2 200\\r\\n\\r\\n' ;;
  */resolve/main/*)
    name=\${url##*/}
    printf 'HTTP/2 302\\r\\nx-xet-hash: %s\\r\\n\\r\\nHTTP/2 200\\r\\n\\r\\n' "\${name%.krm}" ;;
  */v1/reconstructions/empty) echo '{"terms":[]}' ;;
  */v1/reconstructions/*) echo '{"terms":[{"hash":"a"},{"hash":"a"}]}' ;;
  *) echo "stub curl: 想定外の URL $url" >&2; exit 99 ;;
esac
`;

type Run = { readonly code: number; readonly stdout: string };

/** part（名前 → バイト数）を置いた一時リポで `check` を走らせる。 */
const runCheck = async (
  parts: Readonly<Record<string, number>>,
  env: Readonly<Record<string, string>> = {},
): Promise<Run> => {
  const root = await Deno.makeTempDir({ prefix: "karume-hf-upload-test-" });
  try {
    await Deno.mkdir(`${root}/tools/release`, { recursive: true });
    await Deno.copyFile(SCRIPT, `${root}/tools/release/hf-upload.zsh`);
    await Deno.mkdir(`${root}/bin`);
    await Deno.writeTextFile(`${root}/bin/curl`, STUB_CURL, { mode: 0o755 });
    await Deno.mkdir(`${root}/models/${NAME}`, { recursive: true });
    for (const [name, bytes] of Object.entries(parts)) {
      // 疎ファイルで大きさだけ作る（中身は台本が読まない — 見るのは stat の長さ）。
      await Deno.writeFile(`${root}/models/${NAME}/${name}`, new Uint8Array(0));
      await Deno.truncate(`${root}/models/${NAME}/${name}`, bytes);
    }
    const output = await new Deno.Command("zsh", {
      args: [`${root}/tools/release/hf-upload.zsh`, "check", NAME],
      env: { ...env, PATH: `${root}/bin:${Deno.env.get("PATH") ?? ""}` },
      stdout: "piped",
      stderr: "piped",
    }).output();
    return { code: output.code, stdout: new TextDecoder().decode(output.stdout) };
  } finally {
    await Deno.remove(root, { recursive: true });
  }
};

const MIB = 1048576;

Deno.test({
  name:
    "hf-upload check: 全 part を検証できたら表を出して 0 で終わる（xet に載らない小さいファイルは SKIP 行）",
  ignore: !ZSH_AVAILABLE,
  fn: async () => {
    const run = await runCheck({ "good.krm": 20 * MIB, "small.krm": 1024 });
    assertEquals(run.code, 0, run.stdout);
    assert(run.stdout.includes("### fragmentation good.krm"), run.stdout);
    assert(run.stdout.includes("### SKIP small.krm"), run.stdout);
    assert(!run.stdout.includes("FAILED"), run.stdout);
  },
});

Deno.test({
  name: "hf-upload check: HF に無い part は FAILED 行で非 0 になり、健全な行に化けない",
  ignore: !ZSH_AVAILABLE,
  fn: async () => {
    const run = await runCheck({ "good.krm": 20 * MIB, "missing.krm": 20 * MIB });
    assertNotEquals(run.code, 0, run.stdout);
    assert(run.stdout.includes("### FAILED missing.krm"), run.stdout);
    assert(!run.stdout.includes("### fragmentation missing.krm"), run.stdout);
    // 残りの part の行は最後まで出す（落ちる場合でも実測を残す）。
    assert(run.stdout.includes("### fragmentation good.krm"), run.stdout);
  },
});

Deno.test({
  name:
    "hf-upload check: term が 0 本の reconstruction と、大きいのに xet ハッシュの無い part は FAILED",
  ignore: !ZSH_AVAILABLE,
  fn: async () => {
    const run = await runCheck({ "empty.krm": 20 * MIB, "large-plain.krm": 20 * MIB });
    assertNotEquals(run.code, 0, run.stdout);
    assert(run.stdout.includes("### FAILED empty.krm"), run.stdout);
    assert(run.stdout.includes("### FAILED large-plain.krm"), run.stdout);
    assert(!run.stdout.includes("### fragmentation"), run.stdout);
  },
});

Deno.test({
  name: "hf-upload check: xet-read-token を取れなければ表を始めずに非 0 で落ちる",
  ignore: !ZSH_AVAILABLE,
  fn: async () => {
    const run = await runCheck({ "good.krm": 20 * MIB }, { STUB_TOKEN: "fail" });
    assertNotEquals(run.code, 0, run.stdout);
    assert(run.stdout.includes("### FAILED xet-read-token"), run.stdout);
    assert(!run.stdout.includes("### fragmentation"), run.stdout);
    assert(!run.stdout.includes("### SKIP"), run.stdout);
  },
});
