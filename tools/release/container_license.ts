// Pre-publication check (ADR 0122 decision 6, guard 3): reads the `provenance.license` that every container
// (part 0 of `.krm`) of a distribution directory declares, and refuses when an undeclared-licence marker
// (`NOASSERTION`) or a value that does not identify the redistribution terms (`unknown` and the like,
// case-insensitive — `UNIDENTIFIED_LICENSES`) is present. A directory without a single container is refused
// too (what has no provenance to read must not look like a pass).
//
//     node tools/release/container_license.ts models/<repo>
//
// `hf-upload.zsh upload` calls it before uploading (closes on the containers' content, not on the location —
// a mirror copied by hand under `models/` or a symlink is stopped by its content too). Prints
// `### provenance <relative path> license=<identifier>` per container, and with a marker prints
// `### FAILED …` and exits with code 1. An unreadable part 0 also exits with 1 (a container that could not be
// checked must not look like a pass).
//
// How part 0 is recognised: container-v1 §1 — the file starts with the magic `KRMC` (a single-form `krm` also
// starts with part 0). Parts 1 and later (const region, weight blocks) carry no magic. `KRGC` (`krg`) does not
// carry provenance over (§9), so it is not read. A part 1 or later that happens to start with `KRMC` fails as an
// unreadable descriptor (the safe side).
//
// MUST: import nothing but `node:` built-ins (the test copies this file with the script into a temporary
// directory — `tests/hf_upload_test.ts`).

import { type FileHandle, open, readdir, realpath, stat } from "node:fs/promises";

/** The undeclared-licence marker (the same spelling as `wan.umt5_intake.UNDECLARED_LICENSE`). */
export const UNDECLARED_LICENSE = "NOASSERTION";

/**
 * Values that do not identify the redistribution terms (compared ignoring surrounding whitespace and case —
 * the same set as `wan.umt5_intake.UNIDENTIFIED_LICENSES`). Refusing only the exact spelling of the marker
 * would let through a hand-edited record (`noassertion`) or a container written from HF's `unknown`.
 *
 * NOTE: `other` is not included — the containers of an existing distribution (anima) declare `other` and
 * are published with the terms in the NOTICE (intake never records a bare `other` — `wan.umt5_intake.load_intake`).
 */
export const UNIDENTIFIED_LICENSES: ReadonlySet<string> = new Set([
  "",
  "noassertion",
  "unknown",
  "none",
]);

/** Whether `license` is a value that does not identify the redistribution terms. */
export const isUnidentifiedLicense = (license: string): boolean =>
  UNIDENTIFIED_LICENSES.has(license.trim().toLowerCase());

/** Header length of part 0 (container-v1 §1 — magic 4, version 4, graph descriptor length 8, model descriptor length 8). */
const HEADER_BYTES = 24;

/** Upper bound on the descriptor length read (well above the ceiling of container-v1 §10 — no huge allocation from a corrupt length). */
const MAX_DESCRIPTOR_BYTES = 256 * 1024 * 1024;

export class ContainerLicenseError extends Error {
  override name = "ContainerLicenseError";
}

export type ContainerLicense = {
  /** Path relative to the root of the walk. */
  readonly path: string;
  readonly license: string;
};

const readExactly = async (
  file: FileHandle,
  position: number,
  length: number,
  where: string,
): Promise<Uint8Array<ArrayBuffer>> => {
  const buffer = new Uint8Array(length);
  let offset = 0;
  while (offset < length) {
    const { bytesRead } = await file.read(buffer, offset, length - offset, position + offset);
    if (bytesRead === 0) {
      throw new ContainerLicenseError(`${where}: file ended before ${length} bytes were read`);
    }
    offset += bytesRead;
  }
  return buffer;
};

const descriptorLength = (view: DataView, offset: number, where: string): number => {
  const value = view.getBigUint64(offset, true);
  if (value < 1n || value > BigInt(MAX_DESCRIPTOR_BYTES)) {
    throw new ContainerLicenseError(`${where}: descriptor length ${value} is out of range`);
  }
  return Number(value);
};

/** provenance.license of one `.krm` (`undefined` when it is not a part 0). */
export const readPart0License = async (path: string): Promise<string | undefined> => {
  const file = await open(path, "r");
  try {
    const head = new Uint8Array(4);
    const { bytesRead } = await file.read(head, 0, 4, 0);
    if (bytesRead !== 4 || new TextDecoder().decode(head) !== "KRMC") return undefined;
    const header = await readExactly(file, 0, HEADER_BYTES, path);
    const view = new DataView(header.buffer);
    const version = view.getUint32(4, true);
    if (version !== 1) {
      throw new ContainerLicenseError(`${path}: container version ${version} cannot be read`);
    }
    const graphBytes = descriptorLength(view, 8, path);
    const modelBytes = descriptorLength(view, 16, path);
    const text = new TextDecoder("utf-8", { fatal: true }).decode(
      await readExactly(file, HEADER_BYTES + graphBytes, modelBytes, path),
    );
    let model: unknown;
    try {
      model = JSON.parse(text);
    } catch (cause) {
      throw new ContainerLicenseError(`${path}: model descriptor cannot be read as JSON`, {
        cause,
      });
    }
    if (
      typeof model !== "object" ||
      model === null ||
      (model as { format?: unknown }).format !== "karume-model"
    ) {
      throw new ContainerLicenseError(`${path}: model descriptor format is not karume-model`);
    }
    const provenance = (model as { provenance?: unknown }).provenance;
    const license =
      typeof provenance === "object" && provenance !== null
        ? (provenance as { license?: unknown }).license
        : undefined;
    if (typeof license !== "string" || license === "") {
      throw new ContainerLicenseError(`${path}: provenance.license is missing`);
    }
    return license;
  } finally {
    await file.close();
  }
};

/** The part 0 licences of every `.krm` under `root` (symlinks are followed — the same target only once). */
export const readContainerLicenses = async (root: string): Promise<ContainerLicense[]> => {
  const found: ContainerLicense[] = [];
  const visited = new Set<string>();
  let containerFiles = 0;
  const walk = async (directory: string, relative: string): Promise<void> => {
    const real = await realpath(directory);
    if (visited.has(real)) return;
    visited.add(real);
    const names = await readdir(directory);
    names.sort((a, b) => a.localeCompare(b));
    for (const name of names) {
      const path = `${directory}/${name}`;
      const rel = relative === "" ? name : `${relative}/${name}`;
      // stat follows symlinks (a linked mirror is read by its content too).
      const info = await stat(path);
      if (info.isDirectory()) {
        await walk(path, rel);
      } else if (info.isFile() && name.endsWith(".krm")) {
        containerFiles += 1;
        const license = await readPart0License(path);
        if (license !== undefined) found.push({ path: rel, license });
      }
    }
  };
  await walk(root, "");
  if (containerFiles === 0) {
    // No container = not a single provenance was read (every current distribution has containers — a directory
    // of raw safetensors only must not pass as "no undeclared marker").
    throw new ContainerLicenseError(
      `${root}: not a single container (.krm) — no provenance to read`,
    );
  }
  if (found.length === 0) {
    // Parts present but no part 0 = the provenance was not read (must not look like a pass).
    throw new ContainerLicenseError(
      `${root}: ${containerFiles} .krm files but not a single part 0`,
    );
  }
  return found;
};

if (import.meta.main) {
  const args = process.argv.slice(2);
  const root = args[0];
  if (root === undefined || args.length !== 1) {
    console.log("### FAILED usage: container_license.ts <distribution-directory>");
    process.exit(2);
  }
  try {
    const licenses = await readContainerLicenses(root);
    for (const { path, license } of licenses) {
      console.log(`### provenance ${path} license=${license}`);
    }
    const undeclared = licenses.filter(({ license }) => isUnidentifiedLicense(license));
    if (undeclared.length > 0) {
      console.log(
        `### FAILED undeclared-licence marker (${UNDECLARED_LICENSE} — including values that do not identify the redistribution terms) in ` +
          `${undeclared.length} container(s) — a local experimental mirror is not published (ADR 0122 decision 6): ` +
          undeclared.map(({ path, license }) => `${path}（${license}）`).join(", "),
      );
      process.exit(1);
    }
  } catch (error) {
    console.log(
      `### FAILED cannot read the containers' provenance — ${error instanceof Error ? error.message : String(error)}`,
    );
    process.exit(1);
  }
}
