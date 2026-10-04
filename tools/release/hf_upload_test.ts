// hf-upload.zsh の門: 断片化表（TG3 — 検証できなかった part を健全な行に化けさせない）と、
// 公開前の門（ADR 0122 決定 6 の 3 — ライセンス未宣言の印の容器を上げない。末尾の節）。
//
// 台本を一時ディレクトリへ `tools/release/` の形で写し（台本は自分の位置から 2 段上をリポ根と
// みなす）、`models/<name>/` に part を置いて `check` を走らせる。network へは出ない — PATH の
// 先頭に置いた stub の `curl` が HF / CAS の応答を演じる（`deno eval` は本物を使う）。
//
// zsh が無い環境（CI の ubuntu ランナーなど）は明示 SKIP する（ADR 0005）。

import { assert, assertEquals, assertNotEquals, assertRejects } from "@std/assert";
import {
  ContainerLicenseError,
  isUnidentifiedLicense,
  readContainerLicenses,
  UNDECLARED_LICENSE,
} from "./container_license.ts";

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

// ---------------------------------------------------------------------------
// 公開前の門（ADR 0122 決定 6 の 3）: 上げる前に全容器の provenance.license を読み、ライセンス未宣言の
// 印（NOASSERTION）があれば 1 バイトも上げない。hf は stub（呼ばれたら印のファイルを置き、終了コード 3）
// で、呼ばれたかどうかを見る — 本物の HF へは出ない。
// ---------------------------------------------------------------------------

const encoder = new TextEncoder();

/** part 0（container-v1 §1 — ヘッダ + グラフ記述 + モデル記述）のバイト列。 */
const part0 = (license: string, model: Record<string, unknown> = {}): Uint8Array => {
  const graph = encoder.encode(JSON.stringify({ format: "karume-container", version: 1 }));
  const descriptor = encoder.encode(
    JSON.stringify({
      format: "karume-model",
      version: 1,
      provenance: { license, notice: "NOTICE.md" },
      ...model,
    }),
  );
  const bytes = new Uint8Array(24 + graph.length + descriptor.length);
  const view = new DataView(bytes.buffer);
  bytes.set(encoder.encode("KRMC"), 0);
  view.setUint32(4, 1, true);
  view.setBigUint64(8, BigInt(graph.length), true);
  view.setBigUint64(16, BigInt(descriptor.length), true);
  bytes.set(graph, 24);
  bytes.set(descriptor, 24 + graph.length);
  return bytes;
};

/** part 1 以降の偽物（magic を持たない重みのバイト列）。 */
const weights = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);

const STUB_HF = `#!/bin/sh
if [ "$1" = upload ]; then : > "$HF_CALLED"; exit 3; fi
echo "stub hf 1.0"
`;

type Upload = Run & { readonly hfCalled: boolean };

/**
 * `files`（配布ディレクトリからの相対 path → 中身）を置いた一時リポで `upload` を走らせる。
 * `linkMirror` なら配布ディレクトリを `models/` の外に置き、`models/<name>` をそこへの symlink にする
 * （手で写した / 張ったミラーの形）。
 */
const runUpload = async (
  files: Readonly<Record<string, Uint8Array>>,
  { linkMirror = false }: { readonly linkMirror?: boolean } = {},
): Promise<Upload> => {
  const root = await Deno.makeTempDir({ prefix: "karume-hf-upload-gate-" });
  try {
    await Deno.mkdir(`${root}/tools/release`, { recursive: true });
    await Deno.copyFile(SCRIPT, `${root}/tools/release/hf-upload.zsh`);
    await Deno.copyFile(
      new URL("./container_license.ts", import.meta.url),
      `${root}/tools/release/container_license.ts`,
    );
    await Deno.mkdir(`${root}/tools/.venv/bin`, { recursive: true });
    await Deno.writeTextFile(`${root}/tools/.venv/bin/hf`, STUB_HF, { mode: 0o755 });
    await Deno.mkdir(`${root}/home`);
    await Deno.mkdir(`${root}/models`);
    const mirror = linkMirror ? `${root}/elsewhere/${NAME}` : `${root}/models/${NAME}`;
    await Deno.mkdir(mirror, { recursive: true });
    if (linkMirror) await Deno.symlink(mirror, `${root}/models/${NAME}`);
    for (const [path, bytes] of Object.entries(files)) {
      const target = `${mirror}/${path}`;
      await Deno.mkdir(target.slice(0, target.lastIndexOf("/")), { recursive: true });
      await Deno.writeFile(target, bytes);
    }
    const marker = `${root}/hf-called`;
    const output = await new Deno.Command("zsh", {
      args: [`${root}/tools/release/hf-upload.zsh`, "upload", NAME],
      // HOME を一時ディレクトリへ向ける（台本は ~/.cache/huggingface の shard-cache を退避する）。
      env: { HOME: `${root}/home`, HF_CALLED: marker },
      stdout: "piped",
      stderr: "piped",
    }).output();
    const hfCalled = await Deno.stat(marker).then(() => true, () => false);
    return { code: output.code, stdout: new TextDecoder().decode(output.stdout), hfCalled };
  } finally {
    await Deno.remove(root, { recursive: true });
  }
};

const ENCODER_PARTS = "xxl/text_encoder/model.i8";

Deno.test({
  name: "hf-upload upload: ライセンス未宣言の印の容器があれば hf を呼ばずに非 0 で落ちる",
  ignore: !ZSH_AVAILABLE,
  fn: async () => {
    const run = await runUpload({
      [`${ENCODER_PARTS}-00001-of-00003.krm`]: part0(UNDECLARED_LICENSE),
      [`${ENCODER_PARTS}-00002-of-00003.krm`]: new Uint8Array(0),
      [`${ENCODER_PARTS}-00003-of-00003.krm`]: weights,
    });
    assertNotEquals(run.code, 0, run.stdout);
    assert(!run.hfCalled, run.stdout);
    assert(
      run.stdout.includes(`### provenance ${ENCODER_PARTS}-00001-of-00003.krm license=NOASSERTION`),
      run.stdout,
    );
    assert(run.stdout.includes("### FAILED ライセンス未宣言の印"), run.stdout);
  },
});

Deno.test({
  name: "hf-upload upload: models/ の下へ symlink で張った実験用ミラーも中身で止まる",
  ignore: !ZSH_AVAILABLE,
  fn: async () => {
    const run = await runUpload(
      {
        [`${ENCODER_PARTS}-00001-of-00002.krm`]: part0(UNDECLARED_LICENSE),
        [`${ENCODER_PARTS}-00002-of-00002.krm`]: weights,
      },
      { linkMirror: true },
    );
    assertNotEquals(run.code, 0, run.stdout);
    assert(!run.hfCalled, run.stdout);
    assert(run.stdout.includes("license=NOASSERTION"), run.stdout);
  },
});

Deno.test({
  name: "hf-upload upload: 宣言済みのライセンスの容器だけなら門を通り hf を呼ぶ",
  ignore: !ZSH_AVAILABLE,
  fn: async () => {
    const run = await runUpload({
      [`${ENCODER_PARTS}-00001-of-00002.krm`]: part0("apache-2.0"),
      [`${ENCODER_PARTS}-00002-of-00002.krm`]: weights,
      "other/model.f16.krm": part0("mit"),
    });
    // stub の hf は終了コード 3 — 門を通って upload まで進んだ印。
    assertEquals(run.code, 3, run.stdout);
    assert(run.hfCalled, run.stdout);
    assert(run.stdout.includes("license=apache-2.0"), run.stdout);
    assert(run.stdout.includes("### provenance other/model.f16.krm license=mit"), run.stdout);
  },
});

for (const license of ["unknown", "noassertion"]) {
  Deno.test({
    name: `hf-upload upload: 再配布の条件を識別しない値（${license}）の容器も hf を呼ばずに落ちる`,
    ignore: !ZSH_AVAILABLE,
    fn: async () => {
      const run = await runUpload({
        [`${ENCODER_PARTS}-00001-of-00002.krm`]: part0(license),
        [`${ENCODER_PARTS}-00002-of-00002.krm`]: weights,
      });
      assertNotEquals(run.code, 0, run.stdout);
      assert(!run.hfCalled, run.stdout);
      assert(run.stdout.includes(`license=${license}`), run.stdout);
      assert(run.stdout.includes("### FAILED ライセンス未宣言の印"), run.stdout);
    },
  });
}

Deno.test({
  name:
    "hf-upload upload: 容器が 1 本も無いディレクトリ（生の safetensors だけ）は hf を呼ばずに落ちる",
  ignore: !ZSH_AVAILABLE,
  fn: async () => {
    const run = await runUpload({
      "fixture/model.safetensors": weights,
      "README.md": encoder.encode("# fixture\n"),
    });
    assertNotEquals(run.code, 0, run.stdout);
    assert(!run.hfCalled, run.stdout);
    assert(run.stdout.includes("容器（.krm）が 1 本も無い"), run.stdout);
  },
});

Deno.test({
  name: "hf-upload upload: 読めない part 0 は合格に見せず hf を呼ばずに落ちる",
  ignore: !ZSH_AVAILABLE,
  fn: async () => {
    const broken = part0("apache-2.0");
    broken.set(encoder.encode("{"), broken.length - 1);
    const run = await runUpload({
      [`${ENCODER_PARTS}.krm`]: broken.subarray(0, broken.length - 3),
    });
    assertNotEquals(run.code, 0, run.stdout);
    assert(!run.hfCalled, run.stdout);
    assert(run.stdout.includes("### FAILED 容器の出所を読めない"), run.stdout);
  },
});

Deno.test("container_license: part 0 だけを読み、part 1 以降と krg は読まない", async () => {
  const root = await Deno.makeTempDir({ prefix: "karume-container-license-" });
  try {
    await Deno.mkdir(`${root}/a`);
    await Deno.writeFile(`${root}/a/model-00001-of-00002.krm`, part0("apache-2.0"));
    await Deno.writeFile(`${root}/a/model-00002-of-00002.krm`, weights);
    const krg = part0("ignored");
    krg.set(encoder.encode("KRGC"), 0);
    await Deno.writeFile(`${root}/a/model.krg`, krg);
    await Deno.writeFile(`${root}/b.krm`, part0(UNDECLARED_LICENSE));

    assertEquals(await readContainerLicenses(root), [
      { path: "a/model-00001-of-00002.krm", license: "apache-2.0" },
      { path: "b.krm", license: UNDECLARED_LICENSE },
    ]);
  } finally {
    await Deno.remove(root, { recursive: true });
  }
});

Deno.test("container_license: part 列があるのに part 0 が無ければ読めなかったとして落ちる", async () => {
  const root = await Deno.makeTempDir({ prefix: "karume-container-license-" });
  try {
    await Deno.writeFile(`${root}/model-00002-of-00002.krm`, weights);
    await assertRejects(
      () => readContainerLicenses(root),
      ContainerLicenseError,
      "part 0 が 1 本も無い",
    );
  } finally {
    await Deno.remove(root, { recursive: true });
  }
});

Deno.test("container_license: 容器が 1 本も無いディレクトリは読めなかったとして落ちる", async () => {
  const root = await Deno.makeTempDir({ prefix: "karume-container-license-" });
  try {
    await Deno.writeFile(`${root}/model.safetensors`, weights);
    await assertRejects(
      () => readContainerLicenses(root),
      ContainerLicenseError,
      "容器（.krm）が 1 本も無い",
    );
  } finally {
    await Deno.remove(root, { recursive: true });
  }
});

Deno.test("container_license: 印の綴りの違いと再配布の条件を識別しない値を未宣言として扱う", () => {
  for (
    const license of [
      UNDECLARED_LICENSE,
      "noassertion",
      " NoAssertion ",
      "unknown",
      "UNKNOWN",
      "none",
    ]
  ) {
    assert(isUnidentifiedLicense(license), license);
  }
  // 対: 識別子として名乗る値は通す（判定がどれでも落ちる形でない）。`other` は既存の配布形（anima）が
  // 名乗る値（条件は NOTICE）なので通す。
  for (const license of ["apache-2.0", "mit", "cc-by-sa-4.0", "other", "other-vendor-license"]) {
    assert(!isUnidentifiedLicense(license), license);
  }
});

Deno.test("container_license: provenance.license の無いモデル記述は落ちる", async () => {
  const root = await Deno.makeTempDir({ prefix: "karume-container-license-" });
  try {
    await Deno.writeFile(`${root}/model.krm`, part0("apache-2.0", { provenance: {} }));
    await assertRejects(
      () => readContainerLicenses(root),
      ContainerLicenseError,
      "provenance.license が無い",
    );
  } finally {
    await Deno.remove(root, { recursive: true });
  }
});
