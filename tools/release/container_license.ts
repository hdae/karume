// 公開前の門（ADR 0122 決定 6 の 3）: 配布ディレクトリの全容器（`.krm` の part 0）が名乗る
// `provenance.license` を読み、ライセンス未宣言の印（`NOASSERTION`）か再配布の条件を識別しない値
// （`unknown` など・大文字小文字を問わない — `UNIDENTIFIED_LICENSES`）があれば拒む。容器が 1 本も無い
// ディレクトリも拒む（読む出所が無いものを合格に見せない）。
//
//     deno run --no-config --allow-read tools/release/container_license.ts models/<repo>
//
// `hf-upload.zsh upload` が上げる前に呼ぶ（置き場ではなく容器の中身で閉じる — 手で `models/` の下へ
// 写したミラーや symlink も、中身で止まる）。容器ごとに `### provenance <相対 path> license=<識別子>`
// を出し、印があれば `### FAILED …` を出して終了コード 1。読めない part 0 も 1（検証できなかった
// 容器を合格に見せない）。
//
// part 0 の見分け方: container-v1 §1 — ファイル先頭が magic `KRMC`（単一形の `krm` も先頭が part 0）。
// part 1 以降（const 領域・重みの block）は magic を持たない。`KRGC`（`krg`）は provenance を引き継がない
// （§9）ので読まない。偶然 `KRMC` で始まる part 1 以降は descriptor として読めずに落ちる（安全側）。
//
// MUST: import を持たない（台本と一緒に一時ディレクトリへ写して試す — `hf_upload_test.ts`）。

/** ライセンス未宣言の印（`wan.umt5_intake.UNDECLARED_LICENSE` と同じ綴り）。 */
export const UNDECLARED_LICENSE = "NOASSERTION";

/**
 * 再配布の条件を識別しない値（前後の空白と大文字小文字を無視して比べる —
 * `wan.umt5_intake.UNIDENTIFIED_LICENSES` と同じ集合）。印の綴りの完全一致だけを拒むと、手で書き換えた
 * 記録（`noassertion`）や HF の `unknown` から書いた容器が門を通る。
 *
 * NOTE: `other` は入れない — 既存の配布形（anima）の容器が `other` を名乗り、条件を NOTICE に書く形で
 * 公開している（取り込みは素の `other` を記録しない — `wan.umt5_intake.load_intake`）。
 */
export const UNIDENTIFIED_LICENSES: ReadonlySet<string> = new Set([
  "",
  "noassertion",
  "unknown",
  "none",
]);

/** `license` が再配布の条件を識別しない値か。 */
export const isUnidentifiedLicense = (license: string): boolean =>
  UNIDENTIFIED_LICENSES.has(license.trim().toLowerCase());

/** part 0 のヘッダ長（container-v1 §1 — magic 4・version 4・グラフ記述長 8・モデル記述長 8）。 */
const HEADER_BYTES = 24;

/** 読む descriptor の長さの上限（container-v1 §10 の天井より十分大きい — 壊れた長さで巨大な確保をしない）。 */
const MAX_DESCRIPTOR_BYTES = 256 * 1024 * 1024;

export class ContainerLicenseError extends Error {
  override name = "ContainerLicenseError";
}

export type ContainerLicense = {
  /** 走査の根からの相対 path。 */
  readonly path: string;
  readonly license: string;
};

const readExactly = async (
  file: Deno.FsFile,
  length: number,
  where: string,
): Promise<Uint8Array> => {
  const buffer = new Uint8Array(length);
  let offset = 0;
  while (offset < length) {
    const read = await file.read(buffer.subarray(offset));
    if (read === null) {
      throw new ContainerLicenseError(`${where}: ${length} バイトを読む前にファイルが終わった`);
    }
    offset += read;
  }
  return buffer;
};

const descriptorLength = (view: DataView, offset: number, where: string): number => {
  const value = view.getBigUint64(offset, true);
  if (value < 1n || value > BigInt(MAX_DESCRIPTOR_BYTES)) {
    throw new ContainerLicenseError(`${where}: descriptor の長さ ${value} が範囲外`);
  }
  return Number(value);
};

/** `.krm` 1 本の provenance.license（part 0 でなければ `undefined`）。 */
export const readPart0License = async (path: string): Promise<string | undefined> => {
  const file = await Deno.open(path, { read: true });
  try {
    const head = new Uint8Array(4);
    const got = await file.read(head);
    if (got !== 4 || new TextDecoder().decode(head) !== "KRMC") return undefined;
    await file.seek(0, Deno.SeekMode.Start);
    const header = await readExactly(file, HEADER_BYTES, path);
    const view = new DataView(header.buffer);
    const version = view.getUint32(4, true);
    if (version !== 1) {
      throw new ContainerLicenseError(`${path}: container version ${version} は読めない`);
    }
    const graphBytes = descriptorLength(view, 8, path);
    const modelBytes = descriptorLength(view, 16, path);
    await file.seek(HEADER_BYTES + graphBytes, Deno.SeekMode.Start);
    const text = new TextDecoder("utf-8", { fatal: true }).decode(
      await readExactly(file, modelBytes, path),
    );
    let model: unknown;
    try {
      model = JSON.parse(text);
    } catch (cause) {
      throw new ContainerLicenseError(`${path}: モデル記述が JSON として読めない`, { cause });
    }
    if (
      typeof model !== "object" || model === null ||
      (model as { format?: unknown }).format !== "karume-model"
    ) {
      throw new ContainerLicenseError(`${path}: モデル記述の format が karume-model でない`);
    }
    const provenance = (model as { provenance?: unknown }).provenance;
    const license = typeof provenance === "object" && provenance !== null
      ? (provenance as { license?: unknown }).license
      : undefined;
    if (typeof license !== "string" || license === "") {
      throw new ContainerLicenseError(`${path}: provenance.license が無い`);
    }
    return license;
  } finally {
    file.close();
  }
};

/** `root` の下の全 `.krm`（symlink は辿る — 同じ実体は 1 度だけ）の part 0 のライセンス。 */
export const readContainerLicenses = async (root: string): Promise<ContainerLicense[]> => {
  const found: ContainerLicense[] = [];
  const visited = new Set<string>();
  let containerFiles = 0;
  const walk = async (directory: string, relative: string): Promise<void> => {
    const real = await Deno.realPath(directory);
    if (visited.has(real)) return;
    visited.add(real);
    const entries: Deno.DirEntry[] = [];
    for await (const entry of Deno.readDir(directory)) entries.push(entry);
    entries.sort((a, b) => a.name.localeCompare(b.name));
    for (const entry of entries) {
      const path = `${directory}/${entry.name}`;
      const rel = relative === "" ? entry.name : `${relative}/${entry.name}`;
      const info = await Deno.stat(path);
      if (info.isDirectory) {
        await walk(path, rel);
      } else if (info.isFile && entry.name.endsWith(".krm")) {
        containerFiles += 1;
        const license = await readPart0License(path);
        if (license !== undefined) found.push({ path: rel, license });
      }
    }
  };
  await walk(root, "");
  if (containerFiles === 0) {
    // 容器が無い = 出所を 1 本も読んでいない（今の配布形は全て容器を持つ — 生の safetensors だけの
    // ディレクトリを「未宣言の印が無い」として通さない）。
    throw new ContainerLicenseError(`${root}: 容器（.krm）が 1 本も無い — 読む出所が無い`);
  }
  if (found.length === 0) {
    // part 列はあるのに part 0 が無い = 出所を読めていない（合格に見せない）。
    throw new ContainerLicenseError(
      `${root}: .krm が ${containerFiles} 本あるのに part 0 が 1 本も無い`,
    );
  }
  return found;
};

if (import.meta.main) {
  const root = Deno.args[0];
  if (root === undefined || Deno.args.length !== 1) {
    console.log("### FAILED 使い方: container_license.ts <配布ディレクトリ>");
    Deno.exit(2);
  }
  try {
    const licenses = await readContainerLicenses(root);
    for (const { path, license } of licenses) {
      console.log(`### provenance ${path} license=${license}`);
    }
    const undeclared = licenses.filter(({ license }) => isUnidentifiedLicense(license));
    if (undeclared.length > 0) {
      console.log(
        `### FAILED ライセンス未宣言の印（${UNDECLARED_LICENSE} — 再配布の条件を識別しない値を含む）の容器が ` +
          `${undeclared.length} 本 — 手元の実験用ミラーは公開しない（ADR 0122 決定 6）: ` +
          undeclared.map(({ path, license }) => `${path}（${license}）`).join(", "),
      );
      Deno.exit(1);
    }
  } catch (error) {
    console.log(
      `### FAILED 容器の出所を読めない — ${error instanceof Error ? error.message : String(error)}`,
    );
    Deno.exit(1);
  }
}
