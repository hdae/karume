/**
 * verify-diff の純関数部 — 読み込んだ `results.json` の並び → 「ケース × 環境キー」の行列。
 *
 * FS も GPU も触らない（読み込みは main.ts の仕事）。スキーマの正本は書く側の
 * `packages/runtime/tests/helpers/results.ts` で、ここは**読む側**である。そのため検査するのは
 * 道具が実際に読む欄だけだが、欠け・型違い・`schema` 違いは黙って空にせず throw する
 * （空の行列は「差異なし」と区別が付かず、割れているのに緑に見える形になる）。
 *
 * sha256 のクロスデバイス一致は仕様として保証しない（ADR 0106 / docs/limitations.md）ので、
 * 実物の sha が環境間で食い違っても**失敗ではなく差異として挙げるだけ**である。
 */

/** ケース 1 件の決着（results.ts の `ResultStatus` と同じ語彙）。 */
export type ResultStatus = "pass" | "fail" | "written" | "rewritten";

/** 受理に使った許容差の帯。 */
export type Tolerance = { readonly atol: number; readonly rtol: number };

/** 許容差判定の実測（golden 系列のケースが出力ごとに積む）。 */
export type Measurement = {
  /** グラフの出力名。 */
  readonly output: string;
  /** JSON では非有限が null になる。 */
  readonly maxAbs: number | null;
  readonly maxRel: number | null;
  /** 受理に使った帯。 */
  readonly tolerance: Tolerance;
  /** どの段で受理したか（fail のときは最後に測った段）。 */
  readonly stage: "karume" | "spec";
};

/** `results.json` の `cases` 1 件。 */
export type ResultEntry = {
  readonly id: string;
  readonly status: ResultStatus;
  readonly expected?: string;
  readonly actual?: string;
  readonly artifact?: string;
  readonly elapsedMs: number;
  readonly note?: string;
  readonly measurements?: readonly Measurement[];
};

/** 結果を採ったチェックアウト（`git` が無い機で採られた結果は持たない = null）。 */
export type Checkout = { readonly sha: string; readonly dirty: boolean };

/**
 * `results.json` 1 本。
 *
 * `environment.key` を必須にするのは、書き手が環境キーを持たない機では席を決められず throw
 * するためで、実在する `results.json` は必ずキーを持つ。持たない文書は結果を「どの環境の列か」
 * 決められないので、読んだ側でも通さない。
 */
export type ResultsDocument = {
  readonly schema: 1;
  readonly family: string;
  readonly environment: { readonly key: string };
  readonly checkout: Checkout | null;
  readonly startedAt: string;
  readonly cases: readonly ResultEntry[];
};

/** 読み込んだ `results.json` 1 本と、その席の日付。 */
export type LoadedResults = {
  /** 出どころ（警告と失敗の文面に出す）。 */
  readonly path: string;
  /** 席のディレクトリ名から採った日付（`YYYY-MM-DD`）。 */
  readonly date: string;
  readonly document: ResultsDocument;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const requireRecord = (value: unknown, where: string): Record<string, unknown> => {
  if (!isRecord(value)) throw new Error(`${where}: オブジェクトでない`);
  return value;
};

const requireString = (record: Record<string, unknown>, key: string, where: string): string => {
  const value = record[key];
  if (typeof value !== "string") throw new Error(`${where}: ${key} が文字列でない`);
  return value;
};

const optionalString = (
  record: Record<string, unknown>,
  key: string,
  where: string,
): string | undefined => {
  if (!Object.hasOwn(record, key)) return undefined;
  return requireString(record, key, where);
};

const requireNumber = (record: Record<string, unknown>, key: string, where: string): number => {
  const value = record[key];
  if (typeof value !== "number") throw new Error(`${where}: ${key} が数値でない`);
  return value;
};

/** JSON では非有限が null になるので、`null` はそのまま「測れなかった」として通す。 */
const nullableNumber = (
  record: Record<string, unknown>,
  key: string,
  where: string,
): number | null => {
  if (record[key] === null) return null;
  return requireNumber(record, key, where);
};

const STATUSES: readonly ResultStatus[] = ["pass", "fail", "written", "rewritten"];
const STAGES: readonly Measurement["stage"][] = ["karume", "spec"];

const requireStatus = (record: Record<string, unknown>, where: string): ResultStatus => {
  const status = requireString(record, "status", where);
  // 未知の綴りを黙って通すと、行列に「知らない決着」が並んで差異の判定に混ざる。
  const known = STATUSES.find((candidate) => candidate === status);
  if (known === undefined) throw new Error(`${where}: 未知の status '${status}'`);
  return known;
};

const parseMeasurement = (value: unknown, where: string): Measurement => {
  const record = requireRecord(value, where);
  const stage = requireString(record, "stage", where);
  const known = STAGES.find((candidate) => candidate === stage);
  if (known === undefined) throw new Error(`${where}: 未知の stage '${stage}'`);
  const tolerance = requireRecord(record.tolerance, `${where}.tolerance`);
  return {
    output: requireString(record, "output", where),
    maxAbs: nullableNumber(record, "maxAbs", where),
    maxRel: nullableNumber(record, "maxRel", where),
    tolerance: {
      atol: requireNumber(tolerance, "atol", `${where}.tolerance`),
      rtol: requireNumber(tolerance, "rtol", `${where}.tolerance`),
    },
    stage: known,
  };
};

const parseEntry = (value: unknown, where: string): ResultEntry => {
  const record = requireRecord(value, where);
  const id = requireString(record, "id", where);
  const at = `${where} (${id})`;
  const expected = optionalString(record, "expected", at);
  const actual = optionalString(record, "actual", at);
  const artifact = optionalString(record, "artifact", at);
  const note = optionalString(record, "note", at);
  const measurements = record.measurements;
  if (measurements !== undefined && !Array.isArray(measurements)) {
    throw new Error(`${at}: measurements が配列でない`);
  }
  return {
    id,
    status: requireStatus(record, at),
    ...(expected === undefined ? {} : { expected }),
    ...(actual === undefined ? {} : { actual }),
    ...(artifact === undefined ? {} : { artifact }),
    elapsedMs: requireNumber(record, "elapsedMs", at),
    ...(note === undefined ? {} : { note }),
    ...(measurements === undefined ? {} : {
      measurements: measurements.map((one, index) =>
        parseMeasurement(one, `${at}.measurements[${index}]`)
      ),
    }),
  };
};

const parseCheckout = (value: unknown, where: string): Checkout | null => {
  if (value === null) return null;
  const record = requireRecord(value, `${where}.checkout`);
  const dirty = record.dirty;
  if (typeof dirty !== "boolean") throw new Error(`${where}.checkout: dirty が真偽値でない`);
  return { sha: requireString(record, "sha", `${where}.checkout`), dirty };
};

/**
 * `results.json` の中身（`unknown`）を検査して型を付ける。
 *
 * MUST: 検査に落ちたら throw する（この道具の値打ちは「割れている所を指す」ことなので、
 * 読めない文書を空として飲み込むと嘘の「差異なし」になる）。
 */
export const parseResultsDocument = (value: unknown, source = "results.json"): ResultsDocument => {
  const record = requireRecord(value, source);
  const schema = record.schema;
  if (schema !== 1) {
    throw new Error(`${source}: schema が 1 でない（${JSON.stringify(schema)}）`);
  }
  const environment = requireRecord(record.environment, `${source}.environment`);
  const cases = record.cases;
  if (!Array.isArray(cases)) throw new Error(`${source}: cases が配列でない`);
  return {
    schema: 1,
    family: requireString(record, "family", source),
    environment: { key: requireString(environment, "key", `${source}.environment`) },
    checkout: parseCheckout(record.checkout, source),
    startedAt: requireString(record, "startedAt", source),
    cases: cases.map((one, index) => parseEntry(one, `${source}.cases[${index}]`)),
  };
};

/** 行列の絞り込み。 */
export type DiffOptions = {
  /** この系列だけ（未指定 = 全系列）。 */
  readonly families?: readonly string[];
  /** この日付だけ（未指定 = 環境ごとに最新の日付）。 */
  readonly date?: string;
};

/** 行列の 1 マス（その環境がそのケースについて持っている決着）。 */
export type Cell = {
  readonly status: ResultStatus;
  readonly actual?: string;
  readonly elapsedMs: number;
  readonly note?: string;
  readonly measurements?: readonly Measurement[];
};

/** 系列 1 本について、その環境から採った文書。 */
export type SelectedDocument = {
  readonly environment: string;
  readonly date: string;
  readonly path: string;
  readonly checkout: Checkout | null;
};

/** 差異の種類（実物の sha 違いは失敗ではない — 冒頭の注記）。 */
export type DifferenceKind = "status" | "actual" | "missing";

export type Difference = {
  readonly caseId: string;
  readonly kind: DifferenceKind;
  /** 環境キー → その環境の値（status か実物の sha）。値を持たない環境は含まない。 */
  readonly values: Readonly<Record<string, string>>;
  /** そのケースを持たない環境（`kind` が `missing` のときだけ）。 */
  readonly missing?: readonly string[];
};

export type MatrixRow = {
  readonly id: string;
  /** 環境キー → マス。結果が無い環境は欄ごと持たない。 */
  readonly cells: Readonly<Record<string, Cell>>;
};

export type FamilyMatrix = {
  readonly family: string;
  /** 列（辞書順）。 */
  readonly environments: readonly string[];
  readonly selected: readonly SelectedDocument[];
  /** 根には居るが、この系列（と `--date`）の結果を持たない環境。 */
  readonly absent: readonly string[];
  /** 行（ケース ID の和集合・辞書順）。 */
  readonly rows: readonly MatrixRow[];
  readonly differences: readonly Difference[];
  readonly warnings: readonly string[];
};

export type DiffReport = {
  readonly families: readonly FamilyMatrix[];
  readonly warnings: readonly string[];
};

const toCell = (entry: ResultEntry): Cell => ({
  status: entry.status,
  ...(entry.actual === undefined ? {} : { actual: entry.actual }),
  elapsedMs: entry.elapsedMs,
  ...(entry.note === undefined ? {} : { note: entry.note }),
  ...(entry.measurements === undefined ? {} : { measurements: entry.measurements }),
});

const shortSha = (sha: string): string => sha.slice(0, 7);

/** 選んだ文書のチェックアウトを見る（別のコミットの結果を並べていないか）。 */
const checkoutWarnings = (selected: readonly SelectedDocument[]): string[] => {
  const warnings: string[] = [];
  const known = selected.flatMap((one) =>
    one.checkout === null ? [] : [{ environment: one.environment, checkout: one.checkout }]
  );
  if (new Set(known.map((one) => one.checkout.sha)).size > 1) {
    const columns = known
      .map((one) => `${one.environment}=${shortSha(one.checkout.sha)}`)
      .join(" / ");
    warnings.push(
      `checkout の sha が環境間で食い違う（別のコミットの結果を並べている）: ${columns}`,
    );
  }
  for (const one of selected) {
    if (one.checkout === null) {
      warnings.push(`${one.environment}: checkout が無い（git の無い機で採られた結果）`);
      continue;
    }
    if (one.checkout.dirty) {
      warnings.push(
        `${one.environment}: 作業木が dirty なチェックアウトで採られた結果` +
          `（${shortSha(one.checkout.sha)}）`,
      );
    }
  }
  return warnings;
};

const differencesOf = (
  environments: readonly string[],
  rows: readonly MatrixRow[],
): Difference[] => {
  const differences: Difference[] = [];
  for (const row of rows) {
    const statuses: Record<string, string> = {};
    const actuals: Record<string, string> = {};
    const missing: string[] = [];
    for (const environment of environments) {
      const cell = Object.hasOwn(row.cells, environment) ? row.cells[environment] : undefined;
      if (cell === undefined) {
        missing.push(environment);
        continue;
      }
      statuses[environment] = cell.status;
      if (cell.actual !== undefined) actuals[environment] = cell.actual;
    }
    // 片方にしか無いケース（環境が 1 本しか無いときは「割れよう」が無いので挙げない）。
    if (environments.length > 1 && missing.length > 0) {
      differences.push({ caseId: row.id, kind: "missing", values: statuses, missing });
    }
    if (new Set(Object.values(statuses)).size > 1) {
      differences.push({ caseId: row.id, kind: "status", values: statuses });
    }
    // sha を持つ環境どうしだけを見る（片方が sha を採らないケースは「割れた」ではない）。
    if (new Set(Object.values(actuals)).size > 1) {
      differences.push({ caseId: row.id, kind: "actual", values: actuals });
    }
  }
  return differences;
};

/**
 * 読み込んだ結果を系列ごとの行列にする。
 *
 * 同じ（系列, 環境キー）に複数の日付があれば**最新の日付**を採る（`date` 指定があればその
 * 日付だけ）。席は日付までで分かれるので、これが「その環境の最後の走行」になる。
 */
export const buildDiff = (
  loaded: readonly LoadedResults[],
  options: DiffOptions = {},
): DiffReport => {
  const warnings: string[] = [];
  const wanted = options.families === undefined ? undefined : new Set(options.families);
  if (wanted !== undefined) {
    const present = new Set(loaded.map((one) => one.document.family));
    for (const family of [...wanted].sort()) {
      // 系列名の打ち間違いが「差異なし」に化けないよう、当たらなかった指定を必ず言う。
      if (!present.has(family)) warnings.push(`--family ${family} に当たる結果が無い`);
    }
  }

  const allEnvironments = [...new Set(loaded.map((one) => one.document.environment.key))].sort();
  const byFamily = new Map<string, Map<string, LoadedResults>>();
  for (const entry of loaded) {
    const { family, environment } = entry.document;
    if (wanted !== undefined && !wanted.has(family)) continue;
    if (options.date !== undefined && entry.date !== options.date) continue;
    const perEnvironment = byFamily.get(family) ?? new Map<string, LoadedResults>();
    byFamily.set(family, perEnvironment);
    const held = perEnvironment.get(environment.key);
    if (held === undefined || held.date < entry.date) perEnvironment.set(environment.key, entry);
  }

  const families = [...byFamily.entries()]
    .sort(([left], [right]) => (left < right ? -1 : left > right ? 1 : 0))
    .map(([family, perEnvironment]): FamilyMatrix => {
      const chosen = [...perEnvironment.entries()].sort(([left], [right]) =>
        left < right ? -1 : left > right ? 1 : 0
      );
      const environments = chosen.map(([environment]) => environment);
      const selected = chosen.map(([environment, entry]): SelectedDocument => ({
        environment,
        date: entry.date,
        path: entry.path,
        checkout: entry.document.checkout,
      }));
      const caseIds = new Set<string>();
      for (const [, entry] of chosen) for (const one of entry.document.cases) caseIds.add(one.id);
      const rows = [...caseIds].sort().map((id): MatrixRow => {
        const cells: Record<string, Cell> = {};
        for (const [environment, entry] of chosen) {
          const found = entry.document.cases.find((one) => one.id === id);
          if (found !== undefined) cells[environment] = toCell(found);
        }
        return { id, cells };
      });
      return {
        family,
        environments,
        selected,
        absent: allEnvironments.filter((environment) => !perEnvironment.has(environment)),
        rows,
        differences: differencesOf(environments, rows),
        warnings: checkoutWarnings(selected),
      };
    });

  return { families, warnings };
};
