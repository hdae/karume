/**
 * `main.ts profile` の CLI の殻 — 掃引の記録（`karume-geometry-sweep/2`）を読み、adapter 1 種ぶんの
 * **幾何プロファイル**の生成物（TS）を書く / `--check` で再生成とのバイト同一を見る（perf-ledger K-71）。
 *
 * 規則の導出と生成物の文字列は Deno に依らない純関数（`derive.ts` — ブラウザのページと共有）。ここは
 * 引数・ファイルの読み書き・`deno fmt` による整形・終了コードだけを持つ。
 */
import {
  DEFAULT_MIN_SPEEDUP,
  deriveProfile,
  parseSweepReport,
  PROFILE_ID,
  profileConstName,
  type ProfileSpec,
  renderProfileSource,
  type SweepSource,
  verdictLines,
} from "./derive.ts";

/** 整形に使う設定（cwd に依らず `deno fmt --check` と同じ lineWidth で整形する）。 */
const REPO_CONFIG = new URL("../../deno.json", import.meta.url);

const DECODER = new TextDecoder();

export type ProfileFlags = ProfileSpec & { readonly check: boolean };

/**
 * `--check` と `--opt-in` 以外は `--key value` の対（未知のキーは落とす — 綴り違いが既定で走らない）。
 *
 * 表の相手は `--vendor` [`--architecture`] [`--description`]（生成物の `match`）か、`--opt-in`
 * （`match` を省いた注入専用の表）のどちらか一方。両方を渡したら落とす — 注入専用のつもりの表が
 * 自動選択に載る / その逆を黙って起こさない。
 */
export const parseProfileFlags = (argv: readonly string[]): ProfileFlags => {
  const from: string[] = [];
  let id: string | undefined;
  let vendor: string | undefined;
  let architecture: string | undefined;
  let description: string | undefined;
  let optIn = false;
  let out: string | undefined;
  let minSpeedup = DEFAULT_MIN_SPEEDUP;
  let check = false;
  for (let at = 0; at < argv.length; at += 1) {
    const key = argv[at];
    if (key === "--check") {
      check = true;
      continue;
    }
    if (key === "--opt-in") {
      optIn = true;
      continue;
    }
    const value = argv[at + 1];
    if (value === undefined || value.startsWith("--")) {
      throw new Error(`引数を読めない: ${key}`);
    }
    at += 1;
    switch (key) {
      case "--from":
        from.push(value);
        break;
      case "--id":
        id = value;
        break;
      case "--vendor":
        vendor = value;
        break;
      case "--architecture":
        architecture = value;
        break;
      case "--description":
        description = value;
        break;
      case "--out":
        out = value;
        break;
      case "--min-speedup":
        minSpeedup = Number(value);
        // 1 未満を許すと既定より遅い幾何を採りうる（比の門の意味が消える）
        if (!Number.isFinite(minSpeedup) || minSpeedup < 1) {
          throw new Error(`--min-speedup は 1 以上の数（${value}）`);
        }
        break;
      default:
        throw new Error(`引数を読めない: ${key} ${value}`);
    }
  }
  if (from.length === 0) throw new Error("--from <掃引の JSON> が要る（複数可）");
  if (id === undefined || !PROFILE_ID.test(id)) {
    throw new Error(`--id は英小文字始まりの kebab-case（${id ?? "無し"}）`);
  }
  if (out === undefined) throw new Error("--out <生成物の .ts> が要る");
  // MUST: ファイル名 = id（index.ts の一覧と runtime の診断が id からファイルを辿れる形を保つ）
  const basename = out.slice(out.lastIndexOf("/") + 1);
  if (basename !== `${id}.ts`) {
    throw new Error(`--out のファイル名は ${id}.ts（${basename}）`);
  }
  const common = { from, id, out, minSpeedup, check };
  if (optIn) {
    if (vendor !== undefined || architecture !== undefined || description !== undefined) {
      throw new Error(
        "--opt-in（match を省いた注入専用の表）は --vendor / --architecture / --description と同時に指定しない",
      );
    }
    return { ...common, optIn: true };
  }
  if (vendor === undefined || vendor === "") {
    throw new Error("--vendor が要る（自動選択しない注入専用の表は --opt-in）");
  }
  if (description === "") throw new Error("--description は空文字にしない（未指定は省く）");
  return {
    ...common,
    vendor,
    ...(architecture === undefined ? {} : { architecture }),
    ...(description === undefined ? {} : { description }),
  };
};

/**
 * `deno fmt` を通す（リポの設定で — 生成物が `deno fmt --check` を通り、生成 → 再生成でバイト同一に
 * なる形）。
 */
export const formatTypeScript = async (source: string): Promise<string> => {
  const child = new Deno.Command(Deno.execPath(), {
    args: ["fmt", "--config", decodeURIComponent(REPO_CONFIG.pathname), "--ext=ts", "-"],
    stdin: "piped",
    stdout: "piped",
    stderr: "piped",
  }).spawn();
  const writer = child.stdin.getWriter();
  await writer.write(new TextEncoder().encode(source));
  await writer.close();
  const output = await child.output();
  if (!output.success) {
    throw new Error(`deno fmt が生成物を整形できない: ${DECODER.decode(output.stderr)}`);
  }
  return DECODER.decode(output.stdout);
};

/**
 * 表示と再生成コマンドに載せる path。cwd（リポ直下）の下の絶対 path は相対にし、先頭の `./` を
 * 落とす — 同じファイルを別の綴りで渡しても生成物が同じバイトになるように。
 */
export const displayPath = (path: string, cwd: string): string => {
  const prefix = cwd.endsWith("/") ? cwd : `${cwd}/`;
  const relative = path.startsWith(prefix) ? path.slice(prefix.length) : path;
  return relative.replace(/^(\.\/)+/, "");
};

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(
    new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
    (byte) => byte.toString(16).padStart(2, "0"),
  ).join("");

/** 掃引の記録を読む（path は {@link displayPath} で正規化したもの）。 */
const readSweepSource = async (path: string): Promise<SweepSource> => {
  const bytes = await Deno.readFile(path);
  let parsed: unknown;
  try {
    parsed = JSON.parse(DECODER.decode(bytes));
  } catch (cause) {
    const reason = cause instanceof Error ? cause.message : String(cause);
    throw new Error(`${path}: JSON として読めない（${reason}）`);
  }
  return parseSweepReport(parsed, { path, sha256: await sha256Hex(bytes) });
};

/**
 * 既存の生成物と再生成の差の要約（共通の先頭行と末尾行を除いた 1 塊を `-` / `+` で出す —
 * 採否の変化はコメントの採否の行か値の行に現れる）。
 */
const summarizeDifference = (
  before: string,
  after: string,
  limit = 20,
): string[] => {
  const left = before.split("\n");
  const right = after.split("\n");
  let head = 0;
  while (head < left.length && head < right.length && left[head] === right[head]) head += 1;
  let tail = 0;
  while (
    tail < left.length - head && tail < right.length - head &&
    left[left.length - 1 - tail] === right[right.length - 1 - tail]
  ) {
    tail += 1;
  }
  const removed = left.slice(head, left.length - tail);
  const added = right.slice(head, right.length - tail);
  const clip = (lines: readonly string[], mark: string): string[] =>
    lines.length > limit
      ? [
        ...lines.slice(0, limit).map((line) => `${mark} ${line}`),
        `${mark} …（ほか ${lines.length - limit} 行）`,
      ]
      : lines.map((line) => `${mark} ${line}`);
  return [
    `@@ ${head + 1} 行目から（既存 ${removed.length} 行 → 再生成 ${added.length} 行）`,
    ...clip(removed, "-"),
    ...clip(added, "+"),
  ];
};

const readTextOrUndefined = async (path: string): Promise<string | undefined> => {
  try {
    return await Deno.readTextFile(path);
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return undefined;
    throw cause;
  }
};

/**
 * `geometry-profiles/index.ts` の一覧（別の担当の手書き — 生成器は書き換えない）に載っているかの
 * 案内。載っていなければ足す行を出す。一覧は 1 本で、`match` を省いた注入専用の表も同じ一覧に足す
 * （自動選択されないだけで、公開面の `BUILTIN_GEOMETRY_PROFILES` から id で引ける）。
 */
const indexHint = async (flags: ProfileFlags): Promise<string[]> => {
  const directory = flags.out.slice(0, Math.max(0, flags.out.lastIndexOf("/"))) || ".";
  const indexPath = `${directory}/index.ts`;
  const index = await readTextOrUndefined(indexPath);
  const name = profileConstName(flags.id);
  if (index?.includes(`"./${flags.id}.ts"`) === true) {
    return [`[geometry-profile] ${indexPath} には ${name} が既に載っている`];
  }
  return [
    `[geometry-profile] ${indexPath} の BUILTIN_GEOMETRY_PROFILES にまだ載っていない — 次を足す:`,
    `  import { ${name} } from "./${flags.id}.ts";`,
    `  （BUILTIN_GEOMETRY_PROFILES の配列に ${name} を加える${
      flags.optIn === true ? " — 注入専用の表も同じ一覧（match が無いので自動選択はされない）" : ""
    }）`,
  ];
};

/** `main.ts profile …` の本体。終了コードを返す（`--check` の不一致は 1）。 */
export const runProfileCommand = async (argv: readonly string[]): Promise<number> => {
  const parsed = parseProfileFlags(argv);
  const cwd = Deno.cwd();
  const flags: ProfileFlags = {
    ...parsed,
    from: parsed.from.map((path) => displayPath(path, cwd)),
    out: displayPath(parsed.out, cwd),
  };
  const sources = await Promise.all(flags.from.map(readSweepSource));
  const verdicts = deriveProfile(sources, flags);
  const generated = await formatTypeScript(renderProfileSource(flags, sources, verdicts));
  if (flags.check) {
    const existing = await readTextOrUndefined(flags.out);
    if (existing === generated) {
      console.log(`[geometry-profile] ${flags.out} は再生成とバイト同一`);
      return 0;
    }
    console.log(
      existing === undefined
        ? `[geometry-profile] ${flags.out} が無い`
        : `[geometry-profile] ${flags.out} が再生成と違う:\n${
          summarizeDifference(existing, generated).join("\n")
        }`,
    );
    return 1;
  }
  for (const line of verdictLines(verdicts)) console.log(line);
  await Deno.mkdir(flags.out.slice(0, Math.max(0, flags.out.lastIndexOf("/"))) || ".", {
    recursive: true,
  });
  await Deno.writeTextFile(flags.out, generated);
  console.log(`\n[geometry-profile] ${flags.out}`);
  for (const line of await indexHint(flags)) console.log(line);
  return 0;
};
