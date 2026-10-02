/**
 * 保存した表の JSON を読む unknown 境界（DECIDED: ADR 0117 決定 7）。書き手は `geometryProfileJson`
 * （derive.ts — 末尾の `maxRows` の Infinity を `1e999` と書く）。
 *
 * 未知の欄は全ての階層で拒む。表は注入して実行に効く値で、形の版が違う表を黙って読むと欄の意味の取り違えが
 * 通る — 読めない表は捨てて掃引し直すのがアプリの正しい手（ADR 0117 決定 10）。掃引の記録（追記型のログ
 * — 読み手は自分が読む欄だけを使う）とは方針が逆になる。
 */
import { CodegenError } from "../codegen/errors.ts";
import type { GemmGeometry } from "../kernels/gemm-geometry.ts";
import {
  assertGeometryProfile,
  type GemmRowsRule,
  type GeometryProfile,
} from "../kernels/geometry-profile.ts";
import type { I8a8Geometry } from "../kernels/i8a8-geometry.ts";

/** 保存した表の JSON を読めない（文言は欄の path を名指す — {@link parseGeometryProfileJson}）。 */
export class GeometryProfileParseError extends Error {
  override readonly name = "GeometryProfileParseError";
}

type Match = NonNullable<GeometryProfile["match"]>;
type Provenance = NonNullable<GeometryProfile["provenance"]>;

/** 表そのもの（最上位）の path の綴り。 */
const ROOT = "最上位";

const fail = (path: string, reason: string): never => {
  throw new GeometryProfileParseError(`幾何プロファイルの JSON: ${path} ${reason}`);
};

const show = (value: unknown): string => {
  if (value === undefined) return "無い";
  // JSON.stringify は Infinity を null と綴るので、数は String で出す
  return typeof value === "number" ? String(value) : JSON.stringify(value);
};

const child = (path: string, key: string): string => path === ROOT ? key : `${path}.${key}`;

/** `path` の値がオブジェクトで、欄が `required` ∪ `optional` に収まり、`required` が全てあることを見る。 */
const readRecord = (
  value: unknown,
  path: string,
  required: readonly string[],
  optional: readonly string[] = [],
): Record<string, unknown> => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return fail(path, `がオブジェクトでない（${show(value)}）`);
  }
  const fields: Record<string, unknown> = { ...value };
  const unknown = Object.keys(fields).filter((key) =>
    !required.includes(key) && !optional.includes(key)
  );
  if (unknown.length > 0) {
    return fail(
      path,
      `に未知の欄 ${unknown.join(", ")}（形の版が違う表は読まない — 掃引し直して作る）`,
    );
  }
  const missing = required.filter((key) => fields[key] === undefined);
  if (missing.length > 0) return fail(path, `に欄 ${missing.join(", ")} が無い`);
  return fields;
};

const readString = (value: unknown, path: string): string =>
  typeof value === "string" ? value : fail(path, `が文字列でない（${show(value)}）`);

/**
 * `provenance.userAgent`（文字列の配列 — 現れた順・重複除去は書き手の担当）。空の配列は書き手が書かない形なので拒む。
 */
const readUserAgents = (value: unknown, path: string): string[] => {
  if (!Array.isArray(value)) return fail(path, `が配列でない（${show(value)}）`);
  const agents: readonly unknown[] = value;
  if (agents.length === 0) return fail(path, "が空の配列（userAgent の無い表は欄ごと書かない）");
  return agents.map((agent, index) => readString(agent, `${path}[${index}]`));
};

const readInteger = (value: unknown, path: string): number =>
  typeof value === "number" && Number.isSafeInteger(value)
    ? value
    : fail(path, `が整数でない（${show(value)}）`);

const GEMM_FIELDS = ["regM", "regN", "wgX", "wgY"] as const;

const readGemmGeometry = (value: unknown, path: string): GemmGeometry => {
  const fields = readRecord(value, path, GEMM_FIELDS);
  return {
    regM: readInteger(fields.regM, child(path, "regM")),
    regN: readInteger(fields.regN, child(path, "regN")),
    wgX: readInteger(fields.wgX, child(path, "wgX")),
    wgY: readInteger(fields.wgY, child(path, "wgY")),
  };
};

const readI8a8Geometry = (value: unknown, path: string): I8a8Geometry => {
  const fields = readRecord(value, path, [...GEMM_FIELDS, "tileK"]);
  return {
    regM: readInteger(fields.regM, child(path, "regM")),
    regN: readInteger(fields.regN, child(path, "regN")),
    wgX: readInteger(fields.wgX, child(path, "wgX")),
    wgY: readInteger(fields.wgY, child(path, "wgY")),
    tileK: readInteger(fields.tileK, child(path, "tileK")),
  };
};

/** `match`（3 欄とも任意 — 組の条件は {@link assertGeometryProfile} の担当）。 */
const readMatch = (value: unknown, path: string): Match => {
  const fields = readRecord(value, path, [], ["vendor", "architecture", "description"]);
  const optional = (key: keyof Match): string | undefined =>
    fields[key] === undefined ? undefined : readString(fields[key], child(path, key));
  const vendor = optional("vendor");
  const architecture = optional("architecture");
  const description = optional("description");
  return {
    ...(vendor === undefined ? {} : { vendor }),
    ...(architecture === undefined ? {} : { architecture }),
    ...(description === undefined ? {} : { description }),
  };
};

/**
 * `gemmRows`。末尾の `maxRows` は Infinity（JSON では `1e999` — `JSON.parse` が Infinity に読む）。末尾以外の
 * 値域・昇順は {@link assertGeometryProfile} の担当。
 */
const readGemmRows = (value: unknown, path: string): GemmRowsRule[] => {
  if (!Array.isArray(value)) return fail(path, `が配列でない（${show(value)}）`);
  const rules: readonly unknown[] = value;
  return rules.map((entry, index) => {
    const where = `${path}[${index}]`;
    const fields = readRecord(entry, where, ["maxRows", "geometry"]);
    const maxRows = fields.maxRows;
    const last = index === rules.length - 1;
    if (last && maxRows === null) {
      return fail(
        child(where, "maxRows"),
        "が null（Infinity の欠落 — 素の JSON.stringify は Infinity を null にする。表は " +
          "geometryProfileJson で書く）",
      );
    }
    if (typeof maxRows !== "number") {
      return fail(child(where, "maxRows"), `が数でない（${show(maxRows)}）`);
    }
    if (last && maxRows !== Number.POSITIVE_INFINITY) {
      return fail(
        child(where, "maxRows"),
        `が Infinity でない（${maxRows} — 末尾の規則はどの行数にも当たる 1e999）`,
      );
    }
    return { maxRows, geometry: readGemmGeometry(fields.geometry, child(where, "geometry")) };
  });
};

const readProvenance = (value: unknown, path: string): Provenance => {
  // userAgent は参考の任意欄（照合に使わない — ADR 0117 追記 2026-10-02）。未知の欄の拒否は保つ
  const fields = readRecord(value, path, [
    "sweep",
    "sha256",
    "date",
    "candidateSet",
    "adapter",
    "kernels",
    "caseSet",
  ], ["userAgent"]);
  const adapterPath = child(path, "adapter");
  const adapter = readRecord(fields.adapter, adapterPath, [
    "vendor",
    "architecture",
    "device",
    "description",
  ]);
  return {
    sweep: readString(fields.sweep, child(path, "sweep")),
    sha256: readString(fields.sha256, child(path, "sha256")),
    date: readString(fields.date, child(path, "date")),
    candidateSet: readString(fields.candidateSet, child(path, "candidateSet")),
    ...(fields.userAgent === undefined
      ? {}
      : { userAgent: readUserAgents(fields.userAgent, child(path, "userAgent")) }),
    adapter: {
      vendor: readString(adapter.vendor, child(adapterPath, "vendor")),
      architecture: readString(adapter.architecture, child(adapterPath, "architecture")),
      device: readString(adapter.device, child(adapterPath, "device")),
      description: readString(adapter.description, child(adapterPath, "description")),
    },
    kernels: readString(fields.kernels, child(path, "kernels")),
    caseSet: readString(fields.caseSet, child(path, "caseSet")),
  };
};

/**
 * 保存した表の JSON（`geometryProfileJson` の文字列）を読み、`acquireGpu({ geometryProfile })` に渡せる
 * 表に戻す。失敗は {@link GeometryProfileParseError}（文言は欄の path を名指す）。
 *
 * 検査: JSON として読めること・全欄の型（`id` と `provenance` の欄は文字列・幾何は整数の欄だけ）・
 * 全ての階層の未知の欄の拒否・`gemmRows` の末尾の `maxRows` が Infinity（`1e999` の復元 — 末尾が `null`
 * なら Infinity の欠落と名指す）・最後に runtime の注入の門と同じ `assertGeometryProfile`。
 */
export const parseGeometryProfileJson = (text: string): GeometryProfile => {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (cause) {
    throw new GeometryProfileParseError(
      `幾何プロファイルの JSON: JSON として読めない（${
        cause instanceof Error ? cause.message : String(cause)
      }）`,
      { cause },
    );
  }
  const fields = readRecord(parsed, ROOT, ["id", "gemmRows", "attention", "conv2d", "i8a8"], [
    "match",
    "provenance",
  ]);
  const attention = readRecord(fields.attention, "attention", ["qk", "pv"]);
  const conv2d = readRecord(fields.conv2d, "conv2d", ["rows64", "rows32"]);
  const i8a8 = readRecord(fields.i8a8, "i8a8", ["linear", "attentionQk", "attentionPv"]);
  const profile: GeometryProfile = {
    id: readString(fields.id, "id"),
    ...(fields.match === undefined ? {} : { match: readMatch(fields.match, "match") }),
    gemmRows: readGemmRows(fields.gemmRows, "gemmRows"),
    attention: {
      qk: readGemmGeometry(attention.qk, "attention.qk"),
      pv: readGemmGeometry(attention.pv, "attention.pv"),
    },
    conv2d: {
      rows64: readGemmGeometry(conv2d.rows64, "conv2d.rows64"),
      rows32: readGemmGeometry(conv2d.rows32, "conv2d.rows32"),
    },
    i8a8: {
      linear: readI8a8Geometry(i8a8.linear, "i8a8.linear"),
      attentionQk: readI8a8Geometry(i8a8.attentionQk, "i8a8.attentionQk"),
      attentionPv: readI8a8Geometry(i8a8.attentionPv, "i8a8.attentionPv"),
    },
    ...(fields.provenance === undefined
      ? {}
      : { provenance: readProvenance(fields.provenance, "provenance") }),
  };
  try {
    // runtime の注入の門と同じ 1 本（規則列の昇順・幾何の整除条件・match の組）
    assertGeometryProfile(profile);
  } catch (cause) {
    if (cause instanceof CodegenError) {
      throw new GeometryProfileParseError(`幾何プロファイルの JSON: ${cause.message}`, { cause });
    }
    throw cause;
  }
  return profile;
};
