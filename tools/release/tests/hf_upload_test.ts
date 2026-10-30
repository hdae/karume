// Checks of hf-upload.zsh: the fragmentation table (TG3 — a part that could not be checked must not pass as a
// healthy row) and the pre-publication check (ADR 0122 decision 6, guard 3 — a container with the undeclared-licence
// marker is not uploaded; the last section).
//
// Copies the script into a temporary directory in the shape of `tools/release/` (the script takes the directory
// two levels above its own as the repository root), places parts in `models/<name>/` and runs `check`. It does not
// go to the network — a stub `curl` placed first on PATH plays the HF / CAS responses (`node -e` is the real one).
//
// An environment without zsh (the CI ubuntu runner and the like) is an explicit SKIP (ADR 0005).

import { spawn, spawnSync } from "node:child_process";
import { copyFile, truncate, writeFile as writeFileWithMode } from "node:fs/promises";
import {
  isNotFound,
  makeTempDir,
  mkdir,
  remove,
  stat,
  symlink,
  writeFile,
} from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assert, assertEquals, assertNotEquals, assertRejects } from "@std/assert";
import {
  ContainerLicenseError,
  isUnidentifiedLicense,
  readContainerLicenses,
  UNDECLARED_LICENSE,
} from "../container_license.ts";

const hasZsh = (): boolean => {
  const result = spawnSync("zsh", ["-c", "true"], { stdio: "ignore" });
  if (result.error !== undefined) {
    if (isNotFound(result.error)) return false;
    throw result.error;
  }
  return result.status === 0;
};

const ZSH_AVAILABLE = hasZsh();
if (!ZSH_AVAILABLE) {
  console.warn("[karume] zsh is not available; skipping the checks of tools/release/hf-upload.zsh");
}

const SCRIPT = new URL("../hf-upload.zsh", import.meta.url);
// The script runs `node` from the repository root, where `.node-version` pins the Node line; the
// temporary repositories below carry the same file so `node` resolves as it does in the real one
// (without it a version manager may pick a Node that does not start on the machine).
const NODE_VERSION = new URL("../../../.node-version", import.meta.url);
const NAME = "karume-stub";

/**
 * Runs the script with zsh and returns the exit code and stdout (stderr is discarded). env is layered over the parent's environment.
 * A non-zero exit is returned as a result too (the tests look at the script's exit code, so it does not reject).
 */
const runZsh = (args: readonly string[], env: Readonly<Record<string, string>>): Promise<Run> =>
  new Promise((resolve, reject) => {
    const child = spawn("zsh", args, {
      env: { ...process.env, ...env },
      stdio: ["ignore", "pipe", "pipe"],
    });
    const stdout: Buffer[] = [];
    child.stdout.on("data", (chunk: Buffer) => stdout.push(chunk));
    child.stderr.resume();
    child.on("error", reject);
    child.on("close", (code, signal) => {
      if (code === null) {
        reject(new Error(`zsh ${args.join(" ")} ended with signal ${signal}`));
        return;
      }
      resolve({ code, stdout: Buffer.concat(stdout).toString("utf8") });
    });
  });

/**
 * A curl that plays HF / CAS. The part's name decides the response:
 * `missing.krm` = 404 (exit code 22 with `-f`) / `small.krm`, `large-plain.krm` = 200 without an xet
 * hash / anything else = an xet part whose hash is its name (without the extension). For the reconstruction,
 * only the hash `empty` has 0 terms, `nofetch` has 2 terms without fetch_info, `spread` has 3 terms pointing at
 * 3 ranges of 2 xorbs, and the rest have 2 terms pointing at the same single range of 1 xorb (the same shape as
 * repeated references to an all-zero chunk — terms=2 but one fetch range). With `STUB_TOKEN=fail` the token is a 401.
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
  */v1/reconstructions/empty) echo '{"terms":[],"fetch_info":{}}' ;;
  */v1/reconstructions/nofetch) echo '{"terms":[{"hash":"a"},{"hash":"a"}]}' ;;
  */v1/reconstructions/spread)
    echo '{"terms":[{"hash":"a"},{"hash":"b"},{"hash":"a"}],"fetch_info":{"a":[{"range":{"start":0,"end":4}},{"range":{"start":9,"end":12}}],"b":[{"range":{"start":0,"end":2}}]}}' ;;
  */v1/reconstructions/*)
    echo '{"terms":[{"hash":"a"},{"hash":"a"}],"fetch_info":{"a":[{"range":{"start":0,"end":1}}]}}' ;;
  *) echo "stub curl: unexpected URL $url" >&2; exit 99 ;;
esac
`;

type Run = { readonly code: number; readonly stdout: string };

/** Runs `check` in a temporary repository holding the parts (name → byte count). */
const runCheck = async (
  parts: Readonly<Record<string, number>>,
  env: Readonly<Record<string, string>> = {},
): Promise<Run> => {
  const root = await makeTempDir({ prefix: "karume-hf-upload-test-" });
  try {
    await mkdir(`${root}/tools/release`, { recursive: true });
    await copyFile(SCRIPT, `${root}/tools/release/hf-upload.zsh`);
    await copyFile(NODE_VERSION, `${root}/.node-version`);
    await mkdir(`${root}/bin`);
    await writeFileWithMode(`${root}/bin/curl`, STUB_CURL, { mode: 0o755 });
    await mkdir(`${root}/models/${NAME}`, { recursive: true });
    for (const [name, bytes] of Object.entries(parts)) {
      // Create only the size with a sparse file (the script does not read the content — it looks at the stat length).
      await writeFile(`${root}/models/${NAME}/${name}`, new Uint8Array(0));
      await truncate(`${root}/models/${NAME}/${name}`, bytes);
    }
    return await runZsh([`${root}/tools/release/hf-upload.zsh`, "check", NAME], {
      ...env,
      PATH: `${root}/bin:${process.env.PATH ?? ""}`,
    });
  } finally {
    await remove(root, { recursive: true });
  }
};

const MIB = 1048576;

test.skipIf(!ZSH_AVAILABLE)(
  "hf-upload check: prints the table and exits 0 when every part is checked (a small file not on xet gets a SKIP row)",
  async () => {
    const run = await runCheck({ "good.krm": 20 * MIB, "small.krm": 1024 });
    assertEquals(run.code, 0, run.stdout);
    assert(run.stdout.includes("### fragmentation good.krm"), run.stdout);
    assert(run.stdout.includes("### SKIP small.krm"), run.stdout);
    assert(!run.stdout.includes("FAILED"), run.stdout);
  },
);

test.skipIf(!ZSH_AVAILABLE)(
  "hf-upload check: a part not on HF gets a FAILED row and a non-zero exit, and does not pass as a healthy row",
  async () => {
    const run = await runCheck({ "good.krm": 20 * MIB, "missing.krm": 20 * MIB });
    assertNotEquals(run.code, 0, run.stdout);
    assert(run.stdout.includes("### FAILED missing.krm"), run.stdout);
    assert(!run.stdout.includes("### fragmentation missing.krm"), run.stdout);
    // The rows of the remaining parts are printed to the end (the observations are kept even when failing).
    assert(run.stdout.includes("### fragmentation good.krm"), run.stdout);
  },
);

test.skipIf(!ZSH_AVAILABLE)(
  "hf-upload check: a reconstruction with 0 terms and a large part without an xet hash are FAILED",
  async () => {
    const run = await runCheck({ "empty.krm": 20 * MIB, "large-plain.krm": 20 * MIB });
    assertNotEquals(run.code, 0, run.stdout);
    assert(run.stdout.includes("### FAILED empty.krm"), run.stdout);
    assert(run.stdout.includes("### FAILED large-plain.krm"), run.stdout);
    assert(!run.stdout.includes("### fragmentation"), run.stdout);
  },
);

test.skipIf(!ZSH_AVAILABLE)(
  "hf-upload check: the table prints the number of fetch ranges of fetch_info next to terms, and MiB/term stays divided by terms",
  async () => {
    const run = await runCheck({ "good.krm": 20 * MIB, "spread.krm": 30 * MIB });
    assertEquals(run.code, 0, run.stdout);
    // Two terms pointing at the same range twice give ranges=1 (repeated references do not add fetch ranges).
    assert(
      /### fragmentation good\.krm +20 MiB terms= +2 ranges= +1 xorbs= +1 MiB\/term=10\.0$/m.test(
        run.stdout,
      ),
      run.stdout,
    );
    // Three ranges across 2 xorbs (2 in xorb a, 1 in b) are counted as the total over all xorbs.
    assert(
      /### fragmentation spread\.krm +30 MiB terms= +3 ranges= +3 xorbs= +2 MiB\/term=10\.0$/m.test(
        run.stdout,
      ),
      run.stdout,
    );
  },
);

test.skipIf(!ZSH_AVAILABLE)(
  "hf-upload check: a reconstruction without fetch_info gets a FAILED row and a non-zero exit, and does not pass as a healthy row",
  async () => {
    const run = await runCheck({ "good.krm": 20 * MIB, "nofetch.krm": 20 * MIB });
    assertNotEquals(run.code, 0, run.stdout);
    assert(
      run.stdout.includes("### FAILED nofetch.krm reconstruction response has no fetch_info"),
      run.stdout,
    );
    assert(!run.stdout.includes("### fragmentation nofetch.krm"), run.stdout);
    assert(run.stdout.includes("### fragmentation good.krm"), run.stdout);
  },
);

test.skipIf(!ZSH_AVAILABLE)(
  "hf-upload check: fails non-zero without starting the table when xet-read-token cannot be obtained",
  async () => {
    const run = await runCheck({ "good.krm": 20 * MIB }, { STUB_TOKEN: "fail" });
    assertNotEquals(run.code, 0, run.stdout);
    assert(run.stdout.includes("### FAILED xet-read-token"), run.stdout);
    assert(!run.stdout.includes("### fragmentation"), run.stdout);
    assert(!run.stdout.includes("### SKIP"), run.stdout);
  },
);

// ---------------------------------------------------------------------------
// Pre-publication check (ADR 0122 decision 6, guard 3): before uploading, read provenance.license of every container,
// and with the undeclared-licence marker (NOASSERTION) upload not a single byte. hf is a stub (when called it creates
// the marker file and exits with code 3), and the tests look at whether it was called — nothing goes to the real HF.
// ---------------------------------------------------------------------------

const encoder = new TextEncoder();

/** The bytes of part 0 (container-v1 §1 — header + graph descriptor + model descriptor). */
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

/** A fake part 1 or later (weight bytes without a magic). */
const weights = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);

const STUB_HF = `#!/bin/sh
if [ "$1" = upload ]; then : > "$HF_CALLED"; exit 3; fi
echo "stub hf 1.0"
`;

type Upload = Run & { readonly hfCalled: boolean };

/**
 * Runs `upload` in a temporary repository holding `files` (path relative to the distribution directory → content).
 * With `linkMirror` the distribution directory is placed outside `models/` and `models/<name>` is a symlink to it
 * (the shape of a mirror copied / linked by hand).
 */
const runUpload = async (
  files: Readonly<Record<string, Uint8Array>>,
  { linkMirror = false }: { readonly linkMirror?: boolean } = {},
): Promise<Upload> => {
  const root = await makeTempDir({ prefix: "karume-hf-upload-gate-" });
  try {
    await mkdir(`${root}/tools/release`, { recursive: true });
    await copyFile(SCRIPT, `${root}/tools/release/hf-upload.zsh`);
    await copyFile(NODE_VERSION, `${root}/.node-version`);
    await copyFile(
      new URL("../container_license.ts", import.meta.url),
      `${root}/tools/release/container_license.ts`,
    );
    await mkdir(`${root}/tools/.venv/bin`, { recursive: true });
    await writeFileWithMode(`${root}/tools/.venv/bin/hf`, STUB_HF, { mode: 0o755 });
    await mkdir(`${root}/home`);
    await mkdir(`${root}/models`);
    const mirror = linkMirror ? `${root}/elsewhere/${NAME}` : `${root}/models/${NAME}`;
    await mkdir(mirror, { recursive: true });
    if (linkMirror) await symlink(mirror, `${root}/models/${NAME}`);
    for (const [path, bytes] of Object.entries(files)) {
      const target = `${mirror}/${path}`;
      await mkdir(target.slice(0, target.lastIndexOf("/")), { recursive: true });
      await writeFile(target, bytes);
    }
    const marker = `${root}/hf-called`;
    // Point HOME at the temporary directory (the script moves the shard-cache of ~/.cache/huggingface aside).
    const output = await runZsh([`${root}/tools/release/hf-upload.zsh`, "upload", NAME], {
      HOME: `${root}/home`,
      HF_CALLED: marker,
    });
    const hfCalled = await stat(marker).then(
      () => true,
      () => false,
    );
    return { ...output, hfCalled };
  } finally {
    await remove(root, { recursive: true });
  }
};

const ENCODER_PARTS = "xxl/text_encoder/model.i8";

test.skipIf(!ZSH_AVAILABLE)(
  "hf-upload upload: fails non-zero without calling hf when a container carries the undeclared-licence marker",
  async () => {
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
    assert(run.stdout.includes("### FAILED undeclared-licence marker"), run.stdout);
  },
);

test.skipIf(!ZSH_AVAILABLE)(
  "hf-upload upload: an experimental mirror symlinked under models/ is stopped by its content too",
  async () => {
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
);

test.skipIf(!ZSH_AVAILABLE)(
  "hf-upload upload: with only containers of declared licences it passes the check and calls hf",
  async () => {
    const run = await runUpload({
      [`${ENCODER_PARTS}-00001-of-00002.krm`]: part0("apache-2.0"),
      [`${ENCODER_PARTS}-00002-of-00002.krm`]: weights,
      "other/model.f16.krm": part0("mit"),
    });
    // The stub hf exits with code 3 — the sign that it passed the check and reached the upload.
    assertEquals(run.code, 3, run.stdout);
    assert(run.hfCalled, run.stdout);
    assert(run.stdout.includes("license=apache-2.0"), run.stdout);
    assert(run.stdout.includes("### provenance other/model.f16.krm license=mit"), run.stdout);
  },
);

for (const license of ["unknown", "noassertion"]) {
  test.skipIf(!ZSH_AVAILABLE)(
    `hf-upload upload: a container with a value that does not identify the redistribution terms (${license}) also fails without calling hf`,
    async () => {
      const run = await runUpload({
        [`${ENCODER_PARTS}-00001-of-00002.krm`]: part0(license),
        [`${ENCODER_PARTS}-00002-of-00002.krm`]: weights,
      });
      assertNotEquals(run.code, 0, run.stdout);
      assert(!run.hfCalled, run.stdout);
      assert(run.stdout.includes(`license=${license}`), run.stdout);
      assert(run.stdout.includes("### FAILED undeclared-licence marker"), run.stdout);
    },
  );
}

test.skipIf(!ZSH_AVAILABLE)(
  "hf-upload upload: a directory without a single container (raw safetensors only) fails without calling hf",
  async () => {
    const run = await runUpload({
      "fixture/model.safetensors": weights,
      "README.md": encoder.encode("# fixture\n"),
    });
    assertNotEquals(run.code, 0, run.stdout);
    assert(!run.hfCalled, run.stdout);
    assert(run.stdout.includes("not a single container (.krm)"), run.stdout);
  },
);

test.skipIf(!ZSH_AVAILABLE)(
  "hf-upload upload: an unreadable part 0 does not look like a pass and fails without calling hf",
  async () => {
    const broken = part0("apache-2.0");
    broken.set(encoder.encode("{"), broken.length - 1);
    const run = await runUpload({
      [`${ENCODER_PARTS}.krm`]: broken.subarray(0, broken.length - 3),
    });
    assertNotEquals(run.code, 0, run.stdout);
    assert(!run.hfCalled, run.stdout);
    assert(run.stdout.includes("### FAILED cannot read the containers' provenance"), run.stdout);
  },
);

test("container_license: reads only part 0, not parts 1 and later nor krg", async () => {
  const root = await makeTempDir({ prefix: "karume-container-license-" });
  try {
    await mkdir(`${root}/a`);
    await writeFile(`${root}/a/model-00001-of-00002.krm`, part0("apache-2.0"));
    await writeFile(`${root}/a/model-00002-of-00002.krm`, weights);
    const krg = part0("ignored");
    krg.set(encoder.encode("KRGC"), 0);
    await writeFile(`${root}/a/model.krg`, krg);
    await writeFile(`${root}/b.krm`, part0(UNDECLARED_LICENSE));

    assertEquals(await readContainerLicenses(root), [
      { path: "a/model-00001-of-00002.krm", license: "apache-2.0" },
      { path: "b.krm", license: UNDECLARED_LICENSE },
    ]);
  } finally {
    await remove(root, { recursive: true });
  }
});

test("container_license: fails as unreadable when parts are present but no part 0", async () => {
  const root = await makeTempDir({ prefix: "karume-container-license-" });
  try {
    await writeFile(`${root}/model-00002-of-00002.krm`, weights);
    await assertRejects(
      () => readContainerLicenses(root),
      ContainerLicenseError,
      "not a single part 0",
    );
  } finally {
    await remove(root, { recursive: true });
  }
});

test("container_license: a directory without a single container fails as unreadable", async () => {
  const root = await makeTempDir({ prefix: "karume-container-license-" });
  try {
    await writeFile(`${root}/model.safetensors`, weights);
    await assertRejects(
      () => readContainerLicenses(root),
      ContainerLicenseError,
      "not a single container (.krm)",
    );
  } finally {
    await remove(root, { recursive: true });
  }
});

test("container_license: treats spelling variants of the marker and values that do not identify the redistribution terms as undeclared", () => {
  for (const license of [
    UNDECLARED_LICENSE,
    "noassertion",
    " NoAssertion ",
    "unknown",
    "UNKNOWN",
    "none",
  ]) {
    assert(isUnidentifiedLicense(license), license);
  }
  // Counterpart: values that declare an identifier pass (the predicate does not fail on everything). `other` passes
  // because an existing distribution (anima) declares it (terms in the NOTICE).
  for (const license of ["apache-2.0", "mit", "cc-by-sa-4.0", "other", "other-vendor-license"]) {
    assert(!isUnidentifiedLicense(license), license);
  }
});

test("container_license: a model descriptor without provenance.license fails", async () => {
  const root = await makeTempDir({ prefix: "karume-container-license-" });
  try {
    await writeFile(`${root}/model.krm`, part0("apache-2.0", { provenance: {} }));
    await assertRejects(
      () => readContainerLicenses(root),
      ContainerLicenseError,
      "provenance.license is missing",
    );
  } finally {
    await remove(root, { recursive: true });
  }
});
