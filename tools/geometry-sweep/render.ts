/**
 * 幾何プロファイルの**生成物の TS** と再生成コマンドの描画（純関数 — Deno の API に依らない）。CLI
 * （`profile.ts` = `main.ts profile`）とブラウザのページ（`tools/gpu-lab` のプロファイルタブ）が同じ文字列を
 * 作る。
 *
 * 規則の導出と表の値は runtime の生成器（`packages/runtime/src/tune/derive.ts` — 公開面は
 * `@karume/runtime/tune` の `deriveGeometryProfile`）が持ち、ここはリポへ登録する生成物とアプリ用の TS の
 * 綴りだけを持つ（ADR 0117 決定 1 — 道具だけが使う物は道具側に残す）。
 */
import {
  assertGemmGeometry,
  type GemmGeometry,
} from "../../packages/runtime/src/kernels/gemm-geometry.ts";
import {
  assertI8a8Geometry,
  type I8a8Geometry,
} from "../../packages/runtime/src/kernels/i8a8-geometry.ts";
import { DEFAULT_DRIFT_RANGE } from "../../packages/runtime/src/tune/report.ts";
import {
  buildGeometryProfile,
  formatRatio,
  type GeneratedProfile,
  type ProfileSlot,
  type ProfileTarget,
  ROUNDING_ERROR_LIMIT,
  rowsRange,
  type SlotVerdict,
  type SweepSource,
  targetLabel,
  verdictLines,
} from "../../packages/runtime/src/tune/derive.ts";

type Geometry = GemmGeometry | I8a8Geometry;

/**
 * 表の `provenance.adapter`（4 欄）の表示（空の欄は落とす — Deno は architecture を、フラグ無しの Chrome は
 * device と description を空で返す）。表は 4 欄を値のまま持ち、ラベルは描画のときに導く（同じ情報を
 * 2 か所に持たない — ADR 0117 決定 4）。
 */
const adapterLabel = (adapter: GeneratedProfile["provenance"]["adapter"]): string =>
  [adapter.vendor, adapter.architecture, adapter.device, adapter.description]
    .filter((part) => part !== "").join(" / ");

/** プロファイル id（ファイル名 `<id>.ts` と export 名 `<ID を大文字 snake>` の元）。 */
export const PROFILE_ID = /^[a-z][a-z0-9]*(-[a-z0-9]+)*$/;

/**
 * 生成の入力のうち掃引の記録以外（CLI の `--from` / `--id` / `--vendor` / `--architecture` /
 * `--description` / `--opt-in` / `--out` / `--min-speedup`）。`from` と `out` は生成物のコメントの
 * 再生成コマンドに載る path。
 */
export type ProfileSpec = ProfileTarget & {
  readonly from: readonly string[];
  readonly id: string;
  readonly out: string;
  readonly minSpeedup: number;
};

/** `apple-metal-3` → `APPLE_METAL_3`。 */
export const profileConstName = (id: string): string => id.toUpperCase().replaceAll("-", "_");

/** シェルにそのまま貼れる形（空白・記号を含む語だけ単引用符で包む）。 */
const shellWord = (word: string): string =>
  /^[A-Za-z0-9_./:=@%+-]+$/.test(word) ? word : `'${word.replaceAll("'", `'\\''`)}'`;

/** 表の相手を指す CLI の引数（`--opt-in` か `--vendor` [`--architecture`] [`--description`]）。 */
const targetFlags = (target: ProfileTarget): string[] =>
  target.optIn === true ? ["--opt-in"] : [
    "--vendor",
    shellWord(target.vendor),
    ...(target.architecture === undefined
      ? []
      : ["--architecture", shellWord(target.architecture)]),
    ...(target.description === undefined ? [] : ["--description", shellWord(target.description)]),
  ];

/** 再生成コマンド（`--check` 抜き・行継続つきの複数行）。 */
export const regenerateCommand = (flags: ProfileSpec): string[] => [
  "deno run -A tools/geometry-sweep/main.ts profile \\",
  ...flags.from.map((path) => `  --from ${shellWord(path)} \\`),
  `  --id ${shellWord(flags.id)} ${targetFlags(flags).join(" ")} \\`,
  `  --out ${shellWord(flags.out)} --min-speedup ${flags.minSpeedup}`,
];

const renderGeometry = (geometry: Geometry, slot: ProfileSlot): string => {
  if ("tileK" in geometry) {
    assertI8a8Geometry(geometry, slot);
    return `{ regM: ${geometry.regM}, regN: ${geometry.regN}, wgX: ${geometry.wgX}, wgY: ${geometry.wgY}, tileK: ${geometry.tileK} }`;
  }
  assertGemmGeometry(geometry, slot);
  return `{ regM: ${geometry.regM}, regN: ${geometry.regN}, wgX: ${geometry.wgX}, wgY: ${geometry.wgY} }`;
};

/**
 * 生成物の TS（整形前）。同じ入力からは常に同じ文字列（時刻・環境を読まない — 日付は掃引の記録の
 * `date`）。整形は {@link formatTypeScript} が担う。
 */
export const renderProfileSource = (
  flags: ProfileSpec,
  sources: readonly SweepSource[],
  verdicts: readonly SlotVerdict[],
): string => {
  // 値は注入の表（{@link buildGeometryProfile}）から書く — TS の生成物と注入の JSON を 1 本の経路で作る
  const profile = buildGeometryProfile(flags, sources, verdicts);
  const adapter = adapterLabel(profile.provenance.adapter);
  // 行には掃引の記録由来の文字列（失敗の error 文）が入るので、コメントを閉じる綴りと改行を潰す
  const comment = (lines: readonly string[]): string[] =>
    lines.map((line) =>
      line === "" ? " *" : ` * ${line.replaceAll("*/", "* /").replaceAll(/\r?\n/g, " ")}`
    );
  return [
    "/**",
    ...comment([
      `幾何プロファイル \`${flags.id}\`（**生成物 — 手で編集しない**）。`,
      "",
      ...(flags.optIn === true
        ? [
          "tools/geometry-sweep の `profile` が掃引の記録から書いた、タイル幾何の静的な表（perf-ledger K-71）。",
          "**注入専用**（`match` を省いた表）: runtime は自動では選ばない — アプリが `BUILTIN_GEOMETRY_PROFILES`",
          "から id で引いて `acquireGpu({ geometryProfile })` に渡す（ADR 0115 追記決定 7）。",
        ]
        : [
          `tools/geometry-sweep の \`profile\` が掃引の記録から書いた、adapter \`${
            targetLabel(flags)
          }\` 用のタイル幾何の`,
          "静的な表（perf-ledger K-71）。runtime は adapter の `match`（vendor / architecture / description の",
          "完全一致）でこの表を選ぶだけ。",
        ]),
      "実行時には測らない（オートチューン禁止 — ADR 0022 決定 3）。値を変えるときは掃引を取り直して",
      "下のコマンドで再生成する。",
      "",
      "再生成（リポ直下から・`--check` を足すと再生成とバイト同一かだけを見る）:",
      "",
      ...regenerateCommand(flags).map((line) => `  ${line}`),
      "",
      `掃引（adapter ${adapter}）:`,
      "",
      ...sources.map((source) => `- ${source.path}（sha256 ${source.sha256}・${source.date}）`),
      "",
      "採否の基準: クラスの全ケースで出力が既定と一致し、既定比が " +
      `${formatRatio(flags.minSpeedup)} 以上の幾何のうち、`,
      "ケース間の幾何平均が最大のもの。無ければ既定（掃引の既定の行の幾何）。同じケースを複数の掃引が",
      "測っていれば、比はその観測の幾何平均。gemmRows は掃引にある linear / matmul / bmm のケースで決め、3 経路に同じ表が効く。",
      `材料の門: 掃引ごとに、既定の再測定比（cases[].defaultRepeat.driftRatio）が ${DEFAULT_DRIFT_RANGE.min}〜${DEFAULT_DRIFT_RANGE.max} の外か、`,
      "再測定が失敗 / 無いケースはその掃引の比の材料から外す（出力の一致と失敗は見る — 外した掃引で不一致 /",
      "失敗の幾何は採らない。比は同じケースを他の掃引が測っていればそちらで判定し、どの掃引にも残らなければ",
      "測っていない扱い）。外したケースは採否の欄ごとに「掃引 …」の行で示す。",
      "丸めの門: timestamp が丸められた掃引（Chrome のフラグ無しの 100 µs）では、観測（掃引 1 本の中の 1 行）",
      "ごとに比の丸め誤差の上界 E = e(行) + e(既定の行)（e = 刻み ÷ 最小の round）を出し、E が " +
      `${ROUNDING_ERROR_LIMIT * 100}% を超える観測を`,
      "その掃引の比の材料から外す（出力の一致と失敗は見る）。既定の行の e が超える（か出せない）ケースは全観測を",
      "外す。外した観測は「掃引 … の <幾何> は丸め誤差の上界 E …」の行で示す。",
      "",
      "採否:",
      "",
      ...verdictLines(verdicts),
    ]),
    " */",
    'import type { GeometryProfile } from "../geometry-profile.ts";',
    "",
    ...profileDeclaration(profile),
    "",
  ].join("\n");
};

/**
 * 表の値の宣言（`export const <ID>: GeometryProfile = { … };` の行 — 整形前）。リポへ登録する生成物
 * （{@link renderProfileSource}）とアプリ用の TS（{@link renderAppProfileSource}）が同じ行を書く。
 */
const profileDeclaration = (profile: GeneratedProfile): string[] => {
  const { provenance, match } = profile;
  // match を省いた表（注入専用）は欄ごと書かない — `match: {}` は runtime の門が落とす別物
  const matchLine = match === undefined ? [] : [
    `match: { ${
      (["vendor", "architecture", "description"] as const)
        .flatMap((key) => match[key] === undefined ? [] : [`${key}: ${JSON.stringify(match[key])}`])
        .join(", ")
    } },`,
  ];
  const maxRows = (value: number): string =>
    value === Number.POSITIVE_INFINITY ? "Number.POSITIVE_INFINITY" : String(value);
  // 検査の綴りは欄名と同じ範囲（表自身の maxRows から導く）
  const bounds = profile.gemmRows.map((rule) => rule.maxRows);
  const gemmRows = profile.gemmRows.map((rule, index) =>
    `{ maxRows: ${maxRows(rule.maxRows)}, geometry: ${
      renderGeometry(rule.geometry, `gemmRows ${rowsRange(bounds, index)}`)
    } },`
  );
  return [
    `export const ${profileConstName(profile.id)}: GeometryProfile = {`,
    `id: ${JSON.stringify(profile.id)},`,
    ...matchLine,
    "gemmRows: [",
    ...gemmRows,
    "],",
    `attention: { qk: ${renderGeometry(profile.attention.qk, "attention.qk")}, pv: ${
      renderGeometry(profile.attention.pv, "attention.pv")
    } },`,
    `conv2d: { rows64: ${renderGeometry(profile.conv2d.rows64, "conv2d.rows64")}, rows32: ${
      renderGeometry(profile.conv2d.rows32, "conv2d.rows32")
    } },`,
    `i8a8: { linear: ${renderGeometry(profile.i8a8.linear, "i8a8.linear")}, attentionQk: ${
      renderGeometry(profile.i8a8.attentionQk, "i8a8.attentionQk")
    }, attentionPv: ${renderGeometry(profile.i8a8.attentionPv, "i8a8.attentionPv")} },`,
    `provenance: { sweep: ${JSON.stringify(provenance.sweep)}, sha256: ${
      JSON.stringify(provenance.sha256)
    }, date: ${JSON.stringify(provenance.date)}, candidateSet: ${
      JSON.stringify(provenance.candidateSet)
    }, ${
      // 任意の欄（userAgent の無い記録が混ざると生成器が書かない）は値が無ければ欄ごと書かない。値は文字列の配列
      provenance.userAgent === undefined
        ? ""
        : `userAgent: ${JSON.stringify(provenance.userAgent)}, `}adapter: { ${
      (["vendor", "architecture", "device", "description"] as const)
        .map((key) => `${key}: ${JSON.stringify(provenance.adapter[key])}`)
        .join(", ")
    } }, kernels: ${JSON.stringify(provenance.kernels)}, caseSet: ${
      JSON.stringify(provenance.caseSet)
    } },`,
    "};",
  ];
};

/**
 * アプリに置く TS（整形前）: 公開 API の型（`@karume/runtime` の `GeometryProfile`）で表の値を
 * 定数として書く。アプリはこの定数を `acquireGpu({ geometryProfile })` へ渡して注入する（リポへ
 * 登録しない使い方 — ADR 0115 追記決定 6）。末尾の `maxRows` は `Number.POSITIVE_INFINITY`。
 */
export const renderAppProfileSource = (profile: GeneratedProfile): string => {
  const constName = profileConstName(profile.id);
  return [
    "/**",
    ` * 幾何プロファイル \`${profile.id}\`（GPU lab のプロファイルタブが掃引から作った表）。`,
    ` * acquireGpu({ geometryProfile: ${constName} }) に渡す。`,
    " */",
    'import type { GeometryProfile } from "@karume/runtime";',
    "",
    ...profileDeclaration(profile),
    "",
  ].join("\n");
};
