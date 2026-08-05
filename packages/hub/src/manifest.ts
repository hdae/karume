/**
 * `karume.json`（配布 manifest v1）の parse と全構造検査 — ADR 0038 §1〜§3 の正本実装。
 *
 * MUST: 手書き parse・Web 標準 API のみ・未対応と想定外は fail loudly（黙って正規化しない）。
 * MUST: manifest 由来のマップは `Object.hasOwn` 経由でのみ引き、合成はスプレッドのみ
 * （`Object.assign` 禁止 — CLAUDE.md 横断不変条件 / ADR 0038 §1）。
 */

import {
  type AvailableLabels,
  ManifestFormatError,
  ManifestPathError,
  ManifestReferenceError,
  NO_LABELS,
} from "./errors.ts";

/** manifest のファイル名（リポジトリ直下の固定名 — ADR 0038 §1）。 */
export const MANIFEST_FILENAME = "karume.json";

/** manifest 本体の上限（DoS 防波堤 — ADR 0038 §1。取得中にも同じ値で abort する）。 */
export const MAX_MANIFEST_BYTES = 1024 * 1024;

const MAX_COMPONENTS = 64;
const MAX_PRESETS = 32;
const MAX_PIPELINE_CONFIG_BYTES = 256 * 1024;
/** 1 ファイルの上限バイト数（ADR 0038 §2）。 */
const MAX_FILE_BYTES = 16 * 2 ** 30;
/** hub が理解する `format` の major。未知 major は fail loudly（ADR 0038 §1）。 */
const FORMAT_MAJOR = 1;

const SHA256_RE = /^[0-9a-f]{64}$/;
const SEGMENT_RE = /^[A-Za-z0-9._-]+$/;
const FORMAT_RE = /^karume\/([1-9][0-9]*)$/;
const PIPELINE_RE = /^([A-Za-z0-9_-]+)\/([1-9][0-9]*)$/;

/**
 * プロトタイプ汚染に使われるキー。JSON.parse は `__proto__` を**自前プロパティ**として
 * 置くため汚染そのものは起きないが、下流が素朴な代入で読み直す事故を構造的に断つため
 * manifest 全域で拒否する（ADR 0038 §1）。
 */
const FORBIDDEN_KEYS: readonly string[] = ["__proto__", "constructor", "prototype"];

const ENVELOPE_KEYS: readonly string[] = [
  "format",
  "generator",
  "pipeline",
  "components",
  "presets",
  "defaultPreset",
  "pipelineConfig",
];
const COMPONENT_BASE_KEYS: readonly string[] = ["file", "extras"];
const FILE_REF_KEYS: readonly string[] = ["path", "size", "sha256"];
const PRESET_KEYS: readonly string[] = ["weights", "session", "gpuFeatures"];
const GPU_FEATURE_KEYS: readonly string[] = ["shaderF16"];

/** `session` の allowlist（キーも値も — ADR 0038 §3）。runtime 型の素通しではなく manifest 所有。 */
const LINEAR_COMPUTE: readonly LinearCompute[] = ["f32", "i8a8", "f16"];
const ATTENTION_COMPUTE: readonly AttentionCompute[] = ["f32", "f16", "i8a8"];
const SCORE_STORAGE: readonly ScoreStorage[] = ["f32", "f16"];
const SESSION_KEYS: readonly string[] = [
  "linearCompute",
  "attentionCompute",
  "attentionScoreStorage",
];

export type LinearCompute = "f32" | "i8a8" | "f16";
export type AttentionCompute = "f32" | "f16" | "i8a8";
export type ScoreStorage = "f32" | "f16";

/** ファイル参照の 3 点セット（ADR 0038 §2）。3 点全ての存在と形式が parse 時の必須検査。 */
export type FileRef = {
  readonly path: string;
  /** Hub 上の保存形 raw のバイト数。 */
  readonly size: number;
  /** 小文字 hex 64 桁。 */
  readonly sha256: string;
};

/** コンポーネントの基本形 `{file, extras?}`（ADR 0038 §2）。extras は無ければ空マップ。 */
export type ComponentFiles = {
  readonly file: FileRef;
  readonly extras: Readonly<Record<string, FileRef>>;
};

/** `file` と `variants` の排他を型で持つ（ADR 0038 §2）。 */
export type ManifestComponent =
  | { readonly kind: "single"; readonly files: ComponentFiles }
  | { readonly kind: "variants"; readonly variants: Readonly<Record<string, ComponentFiles>> };

/** manifest 所有の実行ノブ語彙（v1 は 3 キー固定 — ADR 0038 §3）。 */
export type SessionSpec = {
  readonly linearCompute?: LinearCompute;
  readonly attentionCompute?: AttentionCompute;
  readonly attentionScoreStorage?: ScoreStorage;
};

/** device 生成前に要る GPU feature（v1 は `shaderF16` のみ — ADR 0038 §3）。 */
export type GpuFeaturesSpec = {
  readonly shaderF16?: boolean;
};

export type Preset = {
  /** `variants` を持つ全コンポーネントへの完全写像。 */
  readonly weights: Readonly<Record<string, string>>;
  readonly session: SessionSpec;
  readonly gpuFeatures?: GpuFeaturesSpec;
};

/** パイプライン実装の契約名 + major（`anima/1`）。未知 major の判定は models 側の責務。 */
export type PipelineId = {
  readonly name: string;
  readonly major: number;
};

export type Manifest = {
  /** `karume/<major>`。major は hub が検査済み。 */
  readonly format: string;
  /** 焼いたツールの版（障害報告の照合用・実行意味論なし）。 */
  readonly generator: string;
  readonly pipeline: PipelineId;
  readonly components: Readonly<Record<string, ManifestComponent>>;
  readonly presets: Readonly<Record<string, Preset>>;
  readonly defaultPreset: string;
  /** パイプライン所有 — hub は素通し（スキーマ検証は `@karume/models` の各実装）。 */
  readonly pipelineConfig: Readonly<Record<string, unknown>>;
  /** 利用可能な preset / variant ラベル（エラー提示にも使う）。 */
  readonly available: AvailableLabels;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/** 検査失敗の生成器。`available` を全エラーへ確実に載せるため、parse 中は必ずこれを通す。 */
type Fail = {
  format: (message: string, cause?: unknown) => ManifestFormatError;
  reference: (message: string) => ManifestReferenceError;
  path: (message: string, path: string) => ManifestPathError;
};

const createFail = (available: AvailableLabels): Fail => ({
  format: (message, cause) => new ManifestFormatError(message, { available, cause }),
  reference: (message) => new ManifestReferenceError(message, { available }),
  path: (message, path) => new ManifestPathError(message, { available, path }),
});

/**
 * manifest 全域から禁止キーを一掃する（`pipelineConfig` の内側も含む）。素通し先の実装が
 * 素朴な代入で読み直しても事故らないことを、hub の入口で一度だけ保証する。
 */
const assertNoForbiddenKeys = (value: unknown, where: string): void => {
  if (Array.isArray(value)) {
    value.forEach((entry, index) => assertNoForbiddenKeys(entry, `${where}[${index}]`));
    return;
  }
  if (!isRecord(value)) return;
  for (const key of Object.keys(value)) {
    if (FORBIDDEN_KEYS.includes(key)) {
      throw new ManifestFormatError(`${where}: 禁止キー '${key}' が現れた`);
    }
    assertNoForbiddenKeys(value[key], `${where}.${key}`);
  }
};

/**
 * 検査を通す前に「利用可能なもの」を最善努力で拾う。壊れた manifest でも
 * 「では何なら動くのか」をエラーに載せられるようにするためだけの走査で、絶対に throw しない。
 */
const surveyLabels = (root: Record<string, unknown>): AvailableLabels => {
  const presetsRaw = root["presets"];
  const presets = isRecord(presetsRaw) ? Object.keys(presetsRaw) : [];
  const componentsRaw = root["components"];
  let variants: Readonly<Record<string, readonly string[]>> = {};
  if (isRecord(componentsRaw)) {
    for (const name of Object.keys(componentsRaw)) {
      const component = componentsRaw[name];
      if (!isRecord(component)) continue;
      const labels = component["variants"];
      if (!isRecord(labels)) continue;
      variants = { ...variants, [name]: Object.keys(labels) };
    }
  }
  return { presets, variants };
};

const assertAllowedKeys = (
  fail: Fail,
  value: Record<string, unknown>,
  allowed: readonly string[],
  where: string,
): void => {
  for (const key of Object.keys(value)) {
    if (!allowed.includes(key)) {
      throw fail.reference(`${where}: 未知キー '${key}'（許可: ${allowed.join(" / ")}）`);
    }
  }
};

/**
 * path の**許可リスト**検査（ADR 0038 §2）。取得層はセグメントを percent-encode してもドットを
 * 透過するため、禁止列挙では抜けがそのまま SHA ピン外への traversal になる。
 */
const assertPath = (fail: Fail, path: string, where: string): void => {
  for (const segment of path.split("/")) {
    if (segment === "") {
      throw fail.path(
        `${where}: path '${path}' に空セグメントがある（先頭 / 末尾 / 連続スラッシュ）`,
        path,
      );
    }
    if (segment.startsWith(".")) {
      throw fail.path(
        `${where}: path '${path}' に先頭ドットのセグメント '${segment}' がある`,
        path,
      );
    }
    if (!SEGMENT_RE.test(segment)) {
      throw fail.path(
        `${where}: path '${path}' のセグメント '${segment}' が許可文字 [A-Za-z0-9._-] に一致しない`,
        path,
      );
    }
  }
};

/**
 * 3 点セットを検査して読む。同一 path の重複参照は合法だが `{size, sha256}` の完全一致を
 * 要求する（矛盾 manifest は self-heal を振動させ、正しいキャッシュを evict し続ける）。
 */
const parseFileRef = (
  fail: Fail,
  raw: unknown,
  where: string,
  seen: Map<string, FileRef>,
): FileRef => {
  if (!isRecord(raw)) throw fail.format(`${where}: ファイル参照がオブジェクトでない`);
  assertAllowedKeys(fail, raw, FILE_REF_KEYS, where);
  const path = raw["path"];
  const size = raw["size"];
  const sha256 = raw["sha256"];
  if (typeof path !== "string") throw fail.format(`${where}: 'path' が無い / 文字列でない`);
  if (typeof size !== "number") throw fail.format(`${where}: 'size' が無い / 数値でない`);
  if (typeof sha256 !== "string") throw fail.format(`${where}: 'sha256' が無い / 文字列でない`);
  if (!SHA256_RE.test(sha256)) {
    throw fail.format(`${where}: 'sha256' は小文字 hex 64 桁でなければならない: '${sha256}'`);
  }
  if (!Number.isSafeInteger(size) || size <= 0 || size > MAX_FILE_BYTES) {
    throw fail.format(
      `${where}: 'size' は 0 < size <= ${MAX_FILE_BYTES} の安全整数でなければならない: ${size}`,
    );
  }
  assertPath(fail, path, where);
  const previous = seen.get(path);
  if (previous !== undefined) {
    if (previous.size !== size || previous.sha256 !== sha256) {
      throw fail.reference(
        `${where}: 重複 path '${path}' の {size, sha256} が食い違う ` +
          `({${previous.size}, ${previous.sha256}} と {${size}, ${sha256}})`,
      );
    }
    return previous;
  }
  const ref: FileRef = { path, size, sha256 };
  seen.set(path, ref);
  return ref;
};

const parseComponentFiles = (
  fail: Fail,
  raw: Record<string, unknown>,
  where: string,
  seen: Map<string, FileRef>,
): ComponentFiles => {
  const file = parseFileRef(fail, raw["file"], `${where}.file`, seen);
  const extrasRaw = raw["extras"];
  if (extrasRaw === undefined) return { file, extras: {} };
  if (!isRecord(extrasRaw)) throw fail.format(`${where}.extras: オブジェクトでない`);
  let extras: Readonly<Record<string, FileRef>> = {};
  for (const name of Object.keys(extrasRaw)) {
    const ref = parseFileRef(fail, extrasRaw[name], `${where}.extras.${name}`, seen);
    extras = { ...extras, [name]: ref };
  }
  return { file, extras };
};

const parseComponent = (
  fail: Fail,
  raw: unknown,
  where: string,
  seen: Map<string, FileRef>,
): ManifestComponent => {
  if (!isRecord(raw)) throw fail.format(`${where}: コンポーネントがオブジェクトでない`);
  const hasFile = Object.hasOwn(raw, "file");
  const hasVariants = Object.hasOwn(raw, "variants");
  if (hasFile === hasVariants) {
    throw fail.format(
      `${where}: 'file' と 'variants' は排他必須（${hasFile ? "両方ある" : "両方ない"}）`,
    );
  }
  if (!hasVariants) {
    assertAllowedKeys(fail, raw, COMPONENT_BASE_KEYS, where);
    return { kind: "single", files: parseComponentFiles(fail, raw, where, seen) };
  }
  assertAllowedKeys(fail, raw, ["variants"], where);
  const variantsRaw = raw["variants"];
  if (!isRecord(variantsRaw)) throw fail.format(`${where}.variants: オブジェクトでない`);
  const labels = Object.keys(variantsRaw);
  if (labels.length === 0) throw fail.format(`${where}.variants: 空（1 つ以上のラベルが要る）`);
  let variants: Readonly<Record<string, ComponentFiles>> = {};
  for (const label of labels) {
    const entry = variantsRaw[label];
    const at = `${where}.variants.${label}`;
    if (!isRecord(entry)) throw fail.format(`${at}: オブジェクトでない`);
    assertAllowedKeys(fail, entry, COMPONENT_BASE_KEYS, at);
    if (!Object.hasOwn(entry, "file")) throw fail.format(`${at}: 'file' が無い`);
    variants = { ...variants, [label]: parseComponentFiles(fail, entry, at, seen) };
  }
  return { kind: "variants", variants };
};

const readEnum = <T extends string>(
  fail: Fail,
  raw: Record<string, unknown>,
  key: string,
  values: readonly T[],
  where: string,
): T | undefined => {
  if (!Object.hasOwn(raw, key)) return undefined;
  const value = raw[key];
  if (typeof value !== "string") throw fail.format(`${where}.${key}: 文字列でない`);
  const found = values.find((candidate) => candidate === value);
  if (found === undefined) {
    throw fail.reference(`${where}.${key}: 未知の値 '${value}'（許可: ${values.join(" / ")}）`);
  }
  return found;
};

const parseSession = (fail: Fail, raw: unknown, where: string): SessionSpec => {
  if (raw === undefined) return {};
  if (!isRecord(raw)) throw fail.format(`${where}.session: オブジェクトでない`);
  assertAllowedKeys(fail, raw, SESSION_KEYS, `${where}.session`);
  const at = `${where}.session`;
  const linearCompute = readEnum(fail, raw, "linearCompute", LINEAR_COMPUTE, at);
  const attentionCompute = readEnum(fail, raw, "attentionCompute", ATTENTION_COMPUTE, at);
  const attentionScoreStorage = readEnum(fail, raw, "attentionScoreStorage", SCORE_STORAGE, at);
  return {
    ...(linearCompute === undefined ? {} : { linearCompute }),
    ...(attentionCompute === undefined ? {} : { attentionCompute }),
    ...(attentionScoreStorage === undefined ? {} : { attentionScoreStorage }),
  };
};

const parseGpuFeatures = (
  fail: Fail,
  raw: unknown,
  where: string,
): GpuFeaturesSpec | undefined => {
  if (raw === undefined) return undefined;
  if (!isRecord(raw)) throw fail.format(`${where}.gpuFeatures: オブジェクトでない`);
  assertAllowedKeys(fail, raw, GPU_FEATURE_KEYS, `${where}.gpuFeatures`);
  if (!Object.hasOwn(raw, "shaderF16")) return {};
  const shaderF16 = raw["shaderF16"];
  if (typeof shaderF16 !== "boolean") {
    throw fail.format(`${where}.gpuFeatures.shaderF16: 真偽値でない`);
  }
  return { shaderF16 };
};

const parseWeights = (
  fail: Fail,
  raw: unknown,
  where: string,
  components: Readonly<Record<string, ManifestComponent>>,
): Readonly<Record<string, string>> => {
  if (!isRecord(raw)) throw fail.format(`${where}.weights: 無い / オブジェクトでない`);
  let weights: Readonly<Record<string, string>> = {};
  for (const name of Object.keys(raw)) {
    if (!Object.hasOwn(components, name)) {
      throw fail.reference(`${where}.weights: 未知のコンポーネント '${name}'`);
    }
    const component = components[name];
    if (component.kind !== "variants") {
      throw fail.reference(
        `${where}.weights: コンポーネント '${name}' は variants を持たない（{file} 形）`,
      );
    }
    const label = raw[name];
    if (typeof label !== "string") {
      throw fail.format(`${where}.weights.${name}: ラベルが文字列でない`);
    }
    if (!Object.hasOwn(component.variants, label)) {
      throw fail.reference(
        `${where}.weights: '${name}' に variant '${label}' が無い` +
          `（利用可能: ${Object.keys(component.variants).join(" / ")}）`,
      );
    }
    weights = { ...weights, [name]: label };
  }
  for (const name of Object.keys(components)) {
    if (components[name].kind === "variants" && !Object.hasOwn(weights, name)) {
      throw fail.reference(
        `${where}.weights: variants を持つ '${name}' の指定が無い（完全写像が必要）`,
      );
    }
  }
  return weights;
};

const parsePreset = (
  fail: Fail,
  raw: unknown,
  where: string,
  components: Readonly<Record<string, ManifestComponent>>,
): Preset => {
  if (!isRecord(raw)) throw fail.format(`${where}: preset がオブジェクトでない`);
  assertAllowedKeys(fail, raw, PRESET_KEYS, where);
  const weights = parseWeights(fail, raw["weights"], where, components);
  const session = parseSession(fail, raw["session"], where);
  const gpuFeatures = parseGpuFeatures(fail, raw["gpuFeatures"], where);
  return {
    weights,
    session,
    ...(gpuFeatures === undefined ? {} : { gpuFeatures }),
  };
};

const parsePipeline = (fail: Fail, raw: unknown): PipelineId => {
  if (typeof raw !== "string") throw fail.format("pipeline: 無い / 文字列でない");
  const matched = PIPELINE_RE.exec(raw);
  if (matched === null) {
    throw fail.format(`pipeline: '<name>/<major>' の形でない: '${raw}'`);
  }
  return { name: matched[1], major: Number(matched[2]) };
};

const parseFormat = (fail: Fail, raw: unknown): string => {
  if (typeof raw !== "string") throw fail.format("format: 無い / 文字列でない");
  const matched = FORMAT_RE.exec(raw);
  if (matched === null) {
    throw fail.format(`format: 'karume/<major>' でない（取り違え）: '${raw}'`);
  }
  const major = Number(matched[1]);
  if (major !== FORMAT_MAJOR) {
    throw fail.format(
      `format: 未対応の major '${raw}'（この hub が読めるのは karume/${FORMAT_MAJOR}）`,
    );
  }
  return raw;
};

const parsePipelineConfig = (fail: Fail, raw: unknown): Readonly<Record<string, unknown>> => {
  if (!isRecord(raw)) {
    throw fail.format("pipelineConfig: 無い / オブジェクトでない（空でも {} を明示）");
  }
  const bytes = new TextEncoder().encode(JSON.stringify(raw)).length;
  if (bytes > MAX_PIPELINE_CONFIG_BYTES) {
    throw fail.format(
      `pipelineConfig: ${bytes} バイトが上限 ${MAX_PIPELINE_CONFIG_BYTES} を超えた`,
    );
  }
  return raw;
};

const parseMap = <T>(
  fail: Fail,
  raw: unknown,
  where: string,
  max: number,
  parseEntry: (name: string, value: unknown) => T,
): Readonly<Record<string, T>> => {
  if (!isRecord(raw)) throw fail.format(`${where}: 無い / オブジェクトでない`);
  const names = Object.keys(raw);
  if (names.length === 0) throw fail.format(`${where}: 空（1 つ以上が要る）`);
  if (names.length > max) {
    throw fail.format(`${where}: ${names.length} 件が上限 ${max} を超えた`);
  }
  let parsed: Readonly<Record<string, T>> = {};
  for (const name of names) {
    parsed = { ...parsed, [name]: parseEntry(name, raw[name]) };
  }
  return parsed;
};

/**
 * `karume.json` のテキストを検査して読む。ADR 0038 §1〜§3 の検査を**全て** parse 時に走らせ、
 * 1 つでも破れたら fetch を開始せずに throw する（「DL 開始後に初めて欠けが分かる」を許さない
 * のが manifest 導入の目的そのもの）。
 */
export const parseManifest = (text: string): Manifest => {
  const bytes = new TextEncoder().encode(text).length;
  if (bytes > MAX_MANIFEST_BYTES) {
    throw new ManifestFormatError(
      `manifest: ${bytes} バイトが上限 ${MAX_MANIFEST_BYTES} を超えた`,
      { available: NO_LABELS },
    );
  }
  let root: unknown;
  try {
    root = JSON.parse(text);
  } catch (error) {
    throw new ManifestFormatError(`manifest: JSON として読めない`, { cause: error });
  }
  if (!isRecord(root)) throw new ManifestFormatError("manifest: 最上位がオブジェクトでない");
  assertNoForbiddenKeys(root, "manifest");

  const fail = createFail(surveyLabels(root));
  assertAllowedKeys(fail, root, ENVELOPE_KEYS, "manifest");
  const format = parseFormat(fail, root["format"]);
  const generator = root["generator"];
  if (typeof generator !== "string" || generator === "") {
    throw fail.format("generator: 無い / 非空文字列でない");
  }
  const pipeline = parsePipeline(fail, root["pipeline"]);

  const seenPaths = new Map<string, FileRef>();
  const components = parseMap(
    fail,
    root["components"],
    "components",
    MAX_COMPONENTS,
    (name, value) => parseComponent(fail, value, `components.${name}`, seenPaths),
  );
  const presets = parseMap(
    fail,
    root["presets"],
    "presets",
    MAX_PRESETS,
    (name, value) => parsePreset(fail, value, `presets.${name}`, components),
  );

  const defaultPreset = root["defaultPreset"];
  if (typeof defaultPreset !== "string") throw fail.format("defaultPreset: 無い / 文字列でない");
  if (!Object.hasOwn(presets, defaultPreset)) {
    throw fail.reference(
      `defaultPreset: '${defaultPreset}' は presets に無い` +
        `（利用可能: ${Object.keys(presets).join(" / ")}）`,
    );
  }
  const pipelineConfig = parsePipelineConfig(fail, root["pipelineConfig"]);

  let variants: Readonly<Record<string, readonly string[]>> = {};
  for (const name of Object.keys(components)) {
    const component = components[name];
    if (component.kind === "variants") {
      variants = { ...variants, [name]: Object.keys(component.variants) };
    }
  }
  return {
    format,
    generator,
    pipeline,
    components,
    presets,
    defaultPreset,
    pipelineConfig,
    available: { presets: Object.keys(presets), variants },
  };
};
