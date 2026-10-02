/**
 * 保存した表を当ててよいかの照合キーのうち、runtime から導く 2 つ（DECIDED: ADR 0117 決定 4）—
 * **カーネルの指紋**（{@link geometryProfileKernelsId}）と**ケース集合の版**（{@link sweepCaseSetId}）。
 *
 * どちらも GPU も時刻も読まない同期の純関数で、同じ runtime なら常に同じ値になる（codegen 決定性 —
 * 同一キー → バイト単位同一 WGSL）。runtime の版の文字列を使わないのは、runtime に版の定数が無く、版は
 * GEMM のカーネルに触れないリリースでも表を捨てさせ、未リリースの checkout ではカーネルが変わっても同じ値の
 * ままだから。指紋は、表が指すカーネルか既定のカーネルが変わったときだけ変わる。
 *
 * ハッシュは FNV-1a 64 bit（{@link fnv1a64Hex}）。同期にするのは、照合の純関数と注入口のコールバック
 * （ADR 0117 決定 6）を同期に保つため（`crypto.subtle` は非同期しか無い）。偶発の衝突だけを想定し、
 * 敵対的な偽装は射程外（表は利用者自身の保存物）。
 *
 * MUST: import 時に計算しない（定数ではなく関数 — 全モジュール副作用ゼロ・ADR 0117 決定 1）。
 */
import { CodegenError, DispatchLimitError } from "../codegen/errors.ts";
import { conv2dIgemmMTile } from "../kernels/conv2d.ts";
import {
  conv2dProfileGeometry,
  gemmRowsGeometry,
  type GeometryProfile,
} from "../kernels/geometry-profile.ts";
import { PROFILE_GEMM_ROWS_BOUNDS, SWEEP_CASES, type SweepCase } from "./cases.ts";
import {
  conv2dCandidate,
  gemmCandidate,
  type GeometryCandidate,
  i8a8Candidate,
} from "./geometries.ts";
import { type CasePlan, casePlan } from "./harness.ts";

/** FNV-1a 64 bit の初期値の上位 / 下位 32 bit（0xcbf29ce484222325）。 */
const FNV_OFFSET_HIGH = 0xcbf29ce4;
const FNV_OFFSET_LOW = 0x84222325;
/** FNV 64 bit の素数 2^40 + 0x1b3 の下位側（2^40 の側は 8 bit の左シフトとして足す）。 */
const FNV_PRIME_LOW = 0x1b3;
const TWO_POW_32 = 0x1_0000_0000;

/**
 * 文字列の UTF-8 バイト列の FNV-1a 64 bit（16 進 16 桁）。
 *
 * 64 bit の状態を 32 bit 2 語で持つ（BigInt はバイトごとの確保が重い — 指紋の入力は約 1 MB）。積
 * `h · (2^40 + 0x1b3)` mod 2^64 は、下位語 × 0x1b3（< 2^41 で double に正確に載る）の繰り上がりと、
 * 上位語 × 0x1b3 と、下位語の 2^40 倍（上位語へ 8 bit 左シフト）の和で組む。
 */
export const fnv1a64Hex = (text: string): string => {
  const bytes = new TextEncoder().encode(text);
  let high = FNV_OFFSET_HIGH;
  let low = FNV_OFFSET_LOW;
  // 添字のループにするのは、Uint8Array の for-of が V8（Deno 2.9.6）で約 2 倍遅いため（約 1 MB で 11 → 5 ms）
  for (let index = 0; index < bytes.length; index += 1) {
    low = (low ^ bytes[index]) >>> 0;
    const product = low * FNV_PRIME_LOW;
    high = (Math.imul(high, FNV_PRIME_LOW) + Math.floor(product / TWO_POW_32) + (low << 8)) >>> 0;
    low = product >>> 0;
  }
  return `${high.toString(16).padStart(8, "0")}${low.toString(16).padStart(8, "0")}`;
};

/**
 * JSON に書く `Infinity` の綴り（`JSON.parse` は範囲外の数を Infinity に読む — 素の `JSON.stringify` は
 * null にする）。表の JSON（derive.ts の `geometryProfileJson`）と正規の JSON（{@link canonicalJson}）が
 * この 1 つを使う — 綴りを別々に持つと、片方だけ変えたときに保存した表と照合キーの導出が食い違う。
 */
export const INFINITY_JSON = "1e999";

/**
 * 正規の JSON（キーを UTF-16 の符号単位順に並べ・空白なし・undefined の欄は落とす）。`Infinity` は
 * {@link INFINITY_JSON} と書く（素の `JSON.stringify` は null にし、境界の末尾が別の値に変わっても同じ
 * 綴りになる）。
 */
export const canonicalJson = (value: unknown): string => {
  if (typeof value === "number") {
    if (Number.isFinite(value)) return JSON.stringify(value);
    if (value === Number.POSITIVE_INFINITY) return INFINITY_JSON;
    if (value === Number.NEGATIVE_INFINITY) return `-${INFINITY_JSON}`;
    throw new Error("正規の JSON: NaN は書けない");
  }
  if (value === null || typeof value === "string" || typeof value === "boolean") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (typeof value === "object") {
    const entries: [string, unknown][] = Object.entries(value);
    return `{${
      entries
        .filter(([, entry]) => entry !== undefined)
        .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
        .map(([key, entry]) => `${JSON.stringify(key)}:${canonicalJson(entry)}`)
        .join(",")
    }}`;
  }
  throw new Error(`正規の JSON: ${typeof value} の値は書けない`);
};

/**
 * 指紋の dispatch 数の上限 = WebGPU が全 device に保証する `maxComputeWorkgroupsPerDimension`（既定値
 * 65535）。workgroup 数の値は上限に依らず（上限は超えたときに投げるだけ）、指紋を device から独立に保つ。
 */
const FINGERPRINT_WORKGROUP_LIMIT = 65535;

/**
 * 指紋の導出で、ケースの dispatch を組めなかった（codegen の門に落ちる・dispatch 数が
 * {@link FINGERPRINT_WORKGROUP_LIMIT} を超える）。文言はケース・表の欄・幾何を名指し、`cause` は codegen の
 * 例外。照合（mismatch.ts）は不一致の文言として返し、生成器（derive.ts）は表を作らずに投げる。
 */
export class KernelsIdError extends Error {
  override readonly name = "KernelsIdError";
}

/** dispatch 数の上限に当たったときに添える説明（device の上限で掃引を通った幾何でも落ちうる理由）。 */
const LIMIT_NOTE = `指紋は device に依らないよう、WebGPU が全 device に保証する ` +
  `maxComputeWorkgroupsPerDimension の既定値 ${FINGERPRINT_WORKGROUP_LIMIT} で dispatch を組む — ` +
  "上限の大きい device の掃引を通った幾何でも、この数を超える表は指紋を導けない";

/** `build` の codegen の例外を、`where` を名指す {@link KernelsIdError} にする（他の例外はそのまま）。 */
const naming = <T>(where: () => string, build: () => T): T => {
  try {
    return build();
  } catch (cause) {
    if (cause instanceof DispatchLimitError) {
      throw new KernelsIdError(`${where()}: ${cause.message}（${LIMIT_NOTE}）`, { cause });
    }
    if (cause instanceof CodegenError) {
      throw new KernelsIdError(`${where()}: ${cause.message}`, { cause });
    }
    throw cause;
  }
};

/** ケースに表が当てる欄（文言の名指し用）と幾何。 */
type ProfileSlot = { readonly field: string; readonly candidate: GeometryCandidate };

/**
 * ケースに表が当てる幾何（Session の経路と同じ選択関数 — src/runtime/recipe-builders/linear.ts の
 * `gemmRowsGeometry(profile, m)`・conv.ts の `conv2dProfileGeometry`・attention.ts の各欄）。bmm の M は
 * 行列 1 枚の行数（buildBmm と同じ）。
 */
const profileSlot = (profile: GeometryProfile, sweepCase: SweepCase): ProfileSlot => {
  switch (sweepCase.op) {
    case "linear":
    case "matmul":
    case "bmm":
      return {
        field: `gemmRows の M ${sweepCase.m} の段`,
        candidate: gemmCandidate(gemmRowsGeometry(profile, sweepCase.m)),
      };
    case "attention":
      return {
        field: `attention.${sweepCase.stage}`,
        candidate: gemmCandidate(
          sweepCase.stage === "qk" ? profile.attention.qk : profile.attention.pv,
        ),
      };
    case "conv2d": {
      const mTile = conv2dIgemmMTile(sweepCase.channelsOut);
      return {
        field: `conv2d.rows${mTile}`,
        candidate: conv2dCandidate(conv2dProfileGeometry(profile, mTile)),
      };
    }
    case "i8a8-linear":
      return { field: "i8a8.linear", candidate: i8a8Candidate(profile.i8a8.linear) };
    case "i8a8-attention":
      return sweepCase.stage === "qk"
        ? { field: "i8a8.attentionQk", candidate: i8a8Candidate(profile.i8a8.attentionQk) }
        : { field: "i8a8.attentionPv", candidate: i8a8Candidate(profile.i8a8.attentionPv) };
  }
};

/**
 * dp4a（WGSL 言語機能 `packed_4x8_integer_dot_product`）の変種。i8a8 の WGSL はこの列挙で 2 つに割れ
 * （数値は同一 — linear-i8a8.ts の `dp4aAvailable`）、表を当てる device がどちらを使うかは adapter の
 * 情報から決まらないので、i8a8 のケースは両方の変種を指紋に入れる。他の op の WGSL は dp4a に依らない。
 */
const dp4aVariants = (sweepCase: SweepCase): readonly boolean[] =>
  sweepCase.op === "i8a8-linear" || sweepCase.op === "i8a8-attention" ? [false, true] : [false];

/** 掃引の case plan（harness.ts の `casePlan` — テストは既定の差し替えを注入する）。 */
export type CasePlanner = (sweepCase: SweepCase, limit: number, dp4a: boolean) => CasePlan;

/**
 * カーネルの指紋の本体（{@link geometryProfileKernelsId} — ケースと case plan を注入できる形）。
 *
 * ケースの順（`cases` の並び）× dp4a の変種（false → true）ごとに、表の幾何と既定の幾何（case plan の
 * `defaultCandidate` = 欄の fallback — 生成器が掃引の既定の行と突き合わせる値）の dispatch を組み、
 * パイプラインキー・params・WGSL・workgroups を 1 行の JSON にして改行で連結した文字列のハッシュを取る
 * （JSON は改行を含まないので、連結が曖昧にならない）。
 *
 * MUST: dispatch は掃引と同じ case plan で組む — 掃引で測ったカーネルと、指紋が指すカーネルを別の経路で
 * 組まない。`id`・`match`・`provenance` は読まない（表の幾何だけが指紋に効く）。
 *
 * 組めない dispatch（codegen の門・dispatch 数の上限）は、ケース・欄・幾何を名指す {@link KernelsIdError}
 * で投げる。
 */
export const profileKernelsId = (
  profile: GeometryProfile,
  cases: readonly SweepCase[],
  planner: CasePlanner,
): string => {
  const lines: string[] = [];
  for (const sweepCase of cases) {
    const slot = naming(
      () => `表 '${profile.id}' のケース ${sweepCase.id} に当てる欄`,
      () => profileSlot(profile, sweepCase),
    );
    for (const dp4a of dp4aVariants(sweepCase)) {
      const plan = planner(sweepCase, FINGERPRINT_WORKGROUP_LIMIT, dp4a);
      const roles = [
        { field: `表 '${profile.id}' の ${slot.field}`, candidate: slot.candidate },
        { field: "既定の幾何（case plan の defaultCandidate）", candidate: plan.defaultCandidate },
      ];
      for (const { field, candidate } of roles) {
        const launch = naming(
          () =>
            `ケース ${sweepCase.id} で ${field} の幾何 ${candidate.name} ${
              JSON.stringify(candidate.geometry)
            } の dispatch を組めない`,
          () => plan.launch(candidate),
        );
        lines.push(
          JSON.stringify([launch.key, [...launch.params], launch.wgsl, launch.workgroups]),
        );
      }
    }
  }
  return fnv1a64Hex(lines.join("\n"));
};

/**
 * 表のカーネルの指紋（`provenance.kernels` — ADR 0117 決定 4）: 掃引の全ケースの shape で、表の幾何と
 * 既定の幾何が組むカーネルのハッシュ（16 進 16 桁）。GPU も時刻も読まない。
 *
 * GEMM のカーネル（キー・params・WGSL・dispatch 数）か既定の幾何が変わると値が変わる。キー・params・
 * WGSL・workgroups の外の変更（recipe の束縛の並びなど）は指紋に出ない — そこは runtime 自身の検査
 * （codegen スナップショット・既定の幾何での GPU テスト）が受け持つ。壊れた幾何（codegen の門に落ちる・
 * dispatch 数が 65535 を超える）の表は {@link KernelsIdError} を投げる。
 */
export const geometryProfileKernelsId = (profile: GeometryProfile): string =>
  profileKernelsId(profile, SWEEP_CASES, casePlan);

/** ケースの型のどれかが持つ欄の名前（ケースの型の和の各要素の keyof の和）。 */
type SweepCaseField = SweepCase extends infer Case ? Case extends SweepCase ? keyof Case : never
  : never;

/**
 * ケースの欄ごとに、ケース集合の版に入れるか（`true` = dispatch を決める欄）。入れる欄は harness.ts の
 * `casePlan` が読む欄 — `id`（ケースの名前で、掃引の記録の行と表の採否の結び目）・`op`・shape（`m` / `n` /
 * `k` / `batch`・attention の `stage` / `batchHeads` / `d`・conv2d の `channelsIn` / `channelsOut` /
 * `height` / `width`）・dispatch に効く attr（attention の `scale` は params に・`score` は S の格納形として
 * WGSL に入る）。外す欄は人向けの情報（`source` = 説明文・`censusCount` = census の本数・`mirrorOf` = 鏡像元）
 * — 説明文を直しただけで全ての保存した表を不一致にしない。
 *
 * MUST: ケースの型に欄を足したら、ここで入れるか外すかを決める（`satisfies` が全ての欄の分類を要求する —
 * 分類し忘れた欄を黙って版の外に置かない）。
 */
const CASE_SET_FIELDS = {
  id: true,
  op: true,
  m: true,
  n: true,
  k: true,
  batch: true,
  stage: true,
  batchHeads: true,
  d: true,
  scale: true,
  score: true,
  channelsIn: true,
  channelsOut: true,
  height: true,
  width: true,
  censusCount: false,
  source: false,
  mirrorOf: false,
} as const satisfies Record<SweepCaseField, boolean>;

const isSweepCaseField = (field: string): field is SweepCaseField =>
  Object.hasOwn(CASE_SET_FIELDS, field);

/** ケースのうち、ケース集合の版に入れる欄だけ（{@link CASE_SET_FIELDS}）。分類に無い欄は投げる。 */
const caseSetFields = (sweepCase: SweepCase): Record<string, unknown> =>
  Object.fromEntries(
    Object.entries(sweepCase).filter(([field]) => {
      if (!isSweepCaseField(field)) {
        throw new Error(`ケース ${sweepCase.id}: 欄 ${field} の分類が無い（CASE_SET_FIELDS）`);
      }
      return CASE_SET_FIELDS[field];
    }),
  );

/** ケース集合の版の本体（{@link sweepCaseSetId} — ケースと境界を注入できる形）。 */
export const caseSetId = (
  cases: readonly SweepCase[],
  bounds: readonly number[],
): string => fnv1a64Hex(canonicalJson({ cases: cases.map(caseSetFields), bounds }));

/**
 * ケース集合の版（`provenance.caseSet` — ADR 0117 決定 4）: 掃引のケース（`SWEEP_CASES`）の dispatch を
 * 決める欄（{@link CASE_SET_FIELDS}）と gemmRows の段の境界（`PROFILE_GEMM_ROWS_BOUNDS`）を正規の JSON に
 * したもののハッシュ（16 進 16 桁）。
 *
 * 手で上げる版番号は置かない（ケースを足して版を上げ忘れる形を作らない）。ケースの増減・並び、dispatch を
 * 決める欄、境界のどれかが変わると値が変わり、保存した表は照合で不一致になる（再掃引）。説明文などの情報の
 * 欄だけの変更では変わらない。
 */
export const sweepCaseSetId = (): string => caseSetId(SWEEP_CASES, PROFILE_GEMM_ROWS_BOUNDS);
