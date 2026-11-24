/**
 * The header comment of a generated table: what the table is, which op families (and whether the
 * GEMV choices of linear) this generation derived and from which records (and how each record was
 * made), which it carried over from the registered table and from which commit, and the adoption
 * rules with the derivations' verdict lines. A shared table's header also says which devices it
 * predicts from and that it is matched without a description, and lists the records per device.
 * Pure over the derivation's inputs and outputs: no date, machine name or absolute path enters it,
 * so the same inputs give the same bytes.
 */

import type { GeometryProfile } from "@karume/runtime";
import type { LoadedRecord } from "./records.ts";

/** A header line would end the comment early. */
export class HeaderError extends Error {
  override name = "HeaderError";
}

export type HeaderInput = {
  readonly profile: GeometryProfile;
  readonly verdicts: readonly string[];
  readonly records: readonly LoadedRecord[];
  readonly attentionBench: readonly LoadedRecord[];
  /** The adoption threshold the derivation was given. */
  readonly minSpeedup: number;
  /**
   * The GEMV sweep records the GEMV choices of linear were derived from, with the rule's threshold
   * and the factor within which a second record repeats a reference's fastest time; absent when
   * this generation did not derive them (their verdict lines are among `verdicts`).
   */
  readonly gemv?: {
    readonly records: readonly LoadedRecord[];
    readonly minSpeedup: number;
    readonly referenceRepeat: number;
  };
  /**
   * The registered table the derivation took as its base: the last commit that touched its file
   * and the file's repository-relative path. Absent for a first table.
   */
  readonly base?: { readonly commit: string; readonly path: string };
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const field = (value: unknown, name: string): unknown =>
  isRecord(value) ? value[name] : undefined;

/** `Chrome 157 on Windows` from a browser's user agent string. */
export const browserOf = (userAgent: string): string => {
  const version = (pattern: RegExp): string | undefined => pattern.exec(userAgent)?.[1];
  const edge = version(/Edg\/(\d+)/);
  const firefox = version(/Firefox\/(\d+)/);
  const chrome = version(/Chrome\/(\d+)/);
  const safari = /Safari\//.test(userAgent) ? version(/Version\/(\d+)/) : undefined;
  const browser =
    edge !== undefined
      ? `Edge ${edge}`
      : firefox !== undefined
        ? `Firefox ${firefox}`
        : chrome !== undefined
          ? `Chrome ${chrome}`
          : safari !== undefined
            ? `Safari ${safari}`
            : "an unrecognised browser";
  const system = /Windows/.test(userAgent)
    ? "Windows"
    : /Android/.test(userAgent)
      ? "Android"
      : /iPhone|iPad/.test(userAgent)
        ? "iOS"
        : /Mac OS X|Macintosh/.test(userAgent)
          ? "macOS"
          : /Linux/.test(userAgent)
            ? "Linux"
            : "an unrecognised OS";
  return `${browser} on ${system}`;
};

/** A user agent as the record carries it: a browser's string, `Node.js/24`, or the Deno CLI's object. */
const userAgentOf = (value: unknown): string => {
  if (typeof value === "string") return value.startsWith("Mozilla/") ? browserOf(value) : value;
  const deno = field(value, "deno");
  return typeof deno === "string" ? `Deno/${deno}` : "user agent not recorded";
};

/** What the runtime record says about its timing (a sweep's gpuTiming, a bench's timing unit). */
const timingOf = (report: unknown): string => {
  const quantized = field(field(report, "gpuTiming"), "quantized");
  if (typeof quantized === "boolean") {
    return quantized ? "timestamps quantised" : "timestamps not quantised";
  }
  const unit = field(field(report, "timing"), "unit");
  return typeof unit === "string" ? `timing unit ${unit}` : "timing not recorded";
};

/** How one record was made, from the workbench envelope or from the bare record itself. */
const madeOf = (record: LoadedRecord): string => {
  const timing = timingOf(record.report);
  const { origin } = record;
  if (origin.kind === "bare") {
    return `bare record; ${userAgentOf(field(record.report, "userAgent"))}; ${timing}; checkout not recorded`;
  }
  const where =
    origin.host === "node"
      ? "Node"
      : origin.userAgent === undefined
        ? `host ${origin.host}`
        : browserOf(origin.userAgent);
  const backend =
    origin.backend === undefined
      ? "backend not recorded"
      : `backend ${origin.backend.name}${origin.backend.source === undefined ? "" : ` (${origin.backend.source})`}`;
  const checkout = `checkout ${origin.checkout.commit}${origin.checkout.dirty ? " (dirty)" : ""}`;
  return `workbench; ${where}; ${backend}; ${timing}; ${checkout}`;
};

/** An adapter as a shared table's header names it: its description and its device id. */
const deviceOf = (adapter: unknown): string => {
  const description = field(adapter, "description");
  const device = field(adapter, "device");
  return `${typeof description === "string" && description !== "" ? description : "no description"} (device ${typeof device === "string" && device !== "" ? device : "not reported"})`;
};

/** `a`, `a and b`, `a, b and c`. */
const listOf = (items: readonly string[]): string =>
  items.length < 2 ? items.join("") : `${items.slice(0, -1).join(", ")} and ${items.at(-1)}`;

/** The four fields of an adapter, for telling a record's adapter apart. */
const adapterKey = (adapter: unknown): string =>
  JSON.stringify(
    ["vendor", "architecture", "device", "description"].map((name) => field(adapter, name)),
  );

const recordLines = (record: LoadedRecord): string[] => {
  const date = field(record.report, "date");
  return [
    `- ${record.path} (sha256 ${record.sha256}, ${typeof date === "string" ? date : "date not recorded"})`,
    `  ${madeOf(record)}`,
  ];
};

const matchOf = (match: GeometryProfile["match"]): string =>
  match === undefined
    ? "no adapter (the table has no match and is used only when injected)"
    : [
        ...(match.vendor === undefined ? [] : [`vendor ${match.vendor}`]),
        ...(match.architecture === undefined ? [] : [`architecture ${match.architecture}`]),
        ...(match.description === undefined ? [] : [`description ${match.description}`]),
      ].join(", ");

const ratio = (value: number): string => `×${value.toFixed(3)}`;

/**
 * The sweep records of a shared table grouped by device, in the order of the table's adapters
 * (`provenance.adapters`): a heading per device, then its records.
 */
const sharedRecordLines = (
  adapters: readonly unknown[],
  records: readonly LoadedRecord[],
): string[] =>
  adapters.flatMap((adapter, index) => [
    ...(index === 0 ? [] : [""]),
    `Records of ${deviceOf(adapter)}:`,
    ...records
      .filter((record) => adapterKey(field(record.report, "adapter")) === adapterKey(adapter))
      .flatMap(recordLines),
  ]);

/**
 * The op families this generation derived: a derived family's provenance entry cites the records
 * given (each record is cited by every family it has rows of), a carried one cites the base's.
 */
const derivedFamilies = (
  profile: GeometryProfile,
  records: readonly LoadedRecord[],
): { readonly derived: string[]; readonly carried: string[] } => {
  const given = new Set(records.map((record) => record.sha256));
  const derived: string[] = [];
  const carried: string[] = [];
  const families = profile.provenance?.families ?? {};
  for (const [family, entry] of Object.entries(families)) {
    if (entry.sha256.split(", ").some((sha256) => given.has(sha256))) derived.push(family);
    else carried.push(family);
  }
  return { derived, carried };
};

const ONLINE = "the online attention's key axis (attention-bench records)";

const GEMV = "the GEMV choices of linear (GEMV sweep records)";

/** The header comment (with its closing line and a final newline). */
export const profileHeader = (input: HeaderInput): string => {
  const { profile, records, attentionBench, base, minSpeedup, gemv } = input;
  const { derived, carried } = derivedFamilies(profile, records);
  const adapters = profile.provenance?.adapters;
  const benchGiven = attentionBench.length > 0;
  const benchCarried = !benchGiven && profile.provenance?.attentionBench !== undefined;
  const gemvCarried = gemv === undefined && profile.linearGemv !== undefined;
  const derivedNames = [
    ...derived,
    ...(benchGiven ? [ONLINE] : []),
    ...(gemv === undefined ? [] : [GEMV]),
  ];
  const carriedNames = [
    ...carried,
    ...(benchCarried ? [ONLINE] : []),
    ...(gemvCarried ? ["the GEMV choices of linear"] : []),
  ];
  const lines = [
    `Geometry profile \`${profile.id}\` (**generated — do not edit by hand**).`,
    "",
    `A static table of tile geometries for the adapter match: ${matchOf(profile.match)}.`,
    ...(adapters === undefined
      ? []
      : [
          `A shared table: derived from the sweep records of ${adapters.length} devices of that vendor and architecture,`,
          `${listOf(adapters.map(deviceOf))}.`,
          "It is matched without a description, so it predicts the geometries for every adapter of that",
          "vendor and architecture no table names by its description (Chrome without its",
          "developer-features flag reports none).",
        ]),
    "The runtime only selects the table by the adapter's match; nothing is measured at run time.",
    "To change a value, take a sweep with the workbench's `geometry-sweep` entry and regenerate the",
    adapters === undefined
      ? `table with \`node tools/geometry-profiles/generate.ts ${profile.id} --records=<path>[,<path>…]\`.`
      : `table with \`node tools/geometry-profiles/generate.ts ${profile.id} --shared --records=<path>,<path>[,…]\`.`,
    ...(gemv === undefined && profile.linearGemv === undefined
      ? []
      : [
          "The GEMV choices of linear (`linearGemv`) come from records of the `gemv-sweep` entry",
          "(`--gemv-records=<path>,<path>[,…]`).",
        ]),
    "",
    `Derived in this generation: ${derivedNames.length === 0 ? "none" : derivedNames.join(", ")}.`,
    "",
    ...(adapters === undefined
      ? [...records, ...attentionBench, ...(gemv?.records ?? [])].flatMap(recordLines)
      : sharedRecordLines(adapters, records)),
    ...(base === undefined
      ? []
      : [
          "",
          `Carried over from the registered table at commit ${base.commit}: ${
            carriedNames.length === 0 ? "none" : carriedNames.join(", ")
          }.`,
          `Their adoption lines are in \`git show ${base.commit}:${base.path}\`.`,
        ]),
    ...(records.length === 0 || adapters === undefined
      ? []
      : [
          "",
          "Adoption (shared): a geometry replaces the default of a field only when, on every device, on",
          "every case of the field that device's records hold, its output is identical to the default",
          `geometry's and it is at least ${ratio(minSpeedup)} faster than the default on that device; among those,`,
          "the one with the largest geometric mean over the devices of each device's geometric mean is",
          "taken, else the default stays. A field some device has no case of keeps the default. No timing",
          "of one device is compared with another's.",
          ...(base === undefined
            ? []
            : [
                "Regenerated on the registered table, a family some device covers only in part (a case of its",
                "case set with no usable row of the default geometry there) keeps the default in every field;",
                "its lines below name the device and the cases.",
              ]),
          "Within one device, a case whose default re-measurement ratio lies outside 0.9–1.1 (or whose",
          "re-measurement failed or is missing) is dropped from that record's ratios, as is an observation",
          "whose timestamp rounding could move its ratio by more than 1%; output mismatches and failures",
          "still count.",
        ]),
    ...(records.length === 0 || adapters !== undefined
      ? []
      : [
          "",
          "Adoption: a geometry replaces the default of a field only when, on every case of the field,",
          `its output is identical to the default geometry's and it is at least ${ratio(minSpeedup)} faster;`,
          "among those, the one with the largest geometric mean of the ratios is taken, else the default",
          "stays.",
          "A case whose default re-measurement ratio lies outside 0.9–1.1 (or whose re-measurement failed",
          "or is missing) is dropped from that record's ratios, as is an observation whose timestamp",
          "rounding could move its ratio by more than 1%; output mismatches and failures still count.",
          ...(benchGiven
            ? [
                "The online attention's tiled key axis is adopted only when every shape of every",
                `attention-bench record is identical and every shape gains at least ${ratio(minSpeedup)}.`,
              ]
            : []),
        ]),
    ...(gemv === undefined
      ? []
      : [
          "",
          "GEMV choices: a candidate replaces the default profile's choice for a shape only when it is at",
          `least ${ratio(gemv.minSpeedup)} faster than that choice in every GEMV sweep record, with its words equal to`,
          "the baseline's (identical class: the sequential variant's per-shape exceptions) or inside the",
          "band (reorder class: the parallel rows, measured at every row count of the parallel range and",
          "not slower at any) in every record; among those, the one whose smallest gain is largest. The",
          "choice's time is its fastest over the records, and only when a second record repeats it within",
          `${ratio(gemv.referenceRepeat)}: a shape whose choice has no such time keeps the default, and so do the shared`,
          "sequential keys whose cases disagree and the rows with packed activations.",
        ]),
    "",
    // A verdict can quote a record's error text, which may span lines: a line break inside it would
    // leave the following lines outside the comment's ` * ` prefix.
    ...input.verdicts.map((verdict) => verdict.replace(/\s*\n\s*/g, " ")),
  ];
  const broken = lines.find((line) => line.includes("*/"));
  if (broken !== undefined) {
    throw new HeaderError(`A header line would close the comment early: ${broken}`);
  }
  return `/**\n${lines.map((line) => (line === "" ? " *" : ` * ${line}`.trimEnd())).join("\n")}\n */\n`;
};
