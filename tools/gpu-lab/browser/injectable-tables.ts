/**
 * GPU 設定の「注入する表」のうち、ページの中で増える選択肢（プロファイルのタブが生成した表と、
 * localStorage に保存した最後の生成表）の一覧と、その保存・復元。
 *
 * DOM と Storage は呼び手が渡す（ここは選択肢の値・文言・適用中の印と、保存の綴り・復元の門だけを
 * 決める純関数）。生成した表はページの寿命の間すべて残す — 表 A を適用中に表 B を作っても A を
 * 選び直せるように。最後に生成した 1 本だけを localStorage に残すのは、Chrome の reload が要る場面で
 * 生成した表を失わないため。
 */
import type { GeometryProfile } from "../../../packages/runtime/mod.ts";
import { assertGeometryProfile } from "../../../packages/runtime/src/kernels/geometry-profile.ts";
import { infinityJson } from "../../geometry-sweep/derive.ts";

/** 保存の置き場（版つき — 形を変えたら版を上げ、古い値は門で落として消す）。 */
export const LAST_GENERATED_KEY = "karume-gpu-lab/last-generated-profile/1";

/** select の値: 保存した表。 */
export const SAVED_VALUE = "saved";
/** select の値の前置: 生成した表（`generated:<連番>`）。 */
export const GENERATED_PREFIX = "generated:";

/** 生成に成功した表（連番は 1 から・生成成功ごとに増える）。 */
export type GeneratedEntry = { readonly serial: number; readonly profile: GeometryProfile };

/** localStorage に置く「最後に生成した表」。 */
export type SavedProfile = {
  /** 保存した時刻（ISO 8601）。 */
  readonly savedAt: string;
  /** 保存したページの adapter。 */
  readonly adapter: {
    readonly vendor: string;
    readonly architecture: string;
    readonly description: string;
  };
  /** 保存したページの checkout。 */
  readonly checkout: { readonly revision: string; readonly dirty: boolean };
  readonly profile: GeometryProfile;
};

export type InjectableTables = {
  readonly saved?: SavedProfile;
  readonly generated: readonly GeneratedEntry[];
};

export type TableOption = { readonly value: string; readonly text: string };

export type ProfileStorage = Pick<Storage, "getItem" | "setItem" | "removeItem">;

const appliedMark = (profile: GeometryProfile, applied: GeometryProfile | undefined): string =>
  profile === applied ? "（適用中）" : "";

const adapterLabel = ({ vendor, architecture, description }: SavedProfile["adapter"]): string =>
  description !== "" ? description : [vendor, architecture].filter((part) => part !== "").join("/");

/**
 * 選択肢の並び（保存した表 → 生成した表の連番順）。`applied` は適用中の注入表（同じ id でも表そのもの
 * で比べる — 作り直した表は id が同じで中身が違う）。
 */
export const tableOptions = (
  tables: InjectableTables,
  applied: GeometryProfile | undefined,
): TableOption[] => [
  ...(tables.saved === undefined ? [] : [{
    value: SAVED_VALUE,
    text: `保存した表: ${tables.saved.profile.id}（${
      new Date(tables.saved.savedAt).toLocaleString()
    }・${adapterLabel(tables.saved.adapter)}）（注入）${
      appliedMark(tables.saved.profile, applied)
    }`,
  }]),
  ...[...tables.generated].sort((a, b) => a.serial - b.serial).map(({ serial, profile }) => ({
    value: `${GENERATED_PREFIX}${serial}`,
    text: `生成した表 #${serial}: ${profile.id}（注入）${appliedMark(profile, applied)}`,
  })),
];

/** select の値 → 表（この一覧の値でなければ undefined）。 */
export const tableForValue = (
  tables: InjectableTables,
  value: string,
): GeometryProfile | undefined => {
  if (value === SAVED_VALUE) return tables.saved?.profile;
  if (!value.startsWith(GENERATED_PREFIX)) return undefined;
  const serial = value.slice(GENERATED_PREFIX.length);
  return tables.generated.find((entry) => String(entry.serial) === serial)?.profile;
};

/** 表 → select の値（この一覧に無い表なら undefined）。 */
export const valueForTable = (
  tables: InjectableTables,
  profile: GeometryProfile,
): string | undefined => {
  if (tables.saved?.profile === profile) return SAVED_VALUE;
  const entry = tables.generated.find((candidate) => candidate.profile === profile);
  return entry === undefined ? undefined : `${GENERATED_PREFIX}${entry.serial}`;
};

/**
 * 最後に生成した表を保存する（上書き）。`Infinity`（最後の `maxRows`）は `1e999` と書く — 素の
 * `JSON.stringify` は null に落とし、復元の門で落ちる。Storage の例外はそのまま投げる。
 */
export const writeLastGenerated = (storage: ProfileStorage, saved: SavedProfile): void =>
  storage.setItem(LAST_GENERATED_KEY, infinityJson(saved));

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null;

/** 復元の門（形が違えば理由つきで投げる）。表は runtime の `assertGeometryProfile` を通す。 */
const parseSaved = (text: string): SavedProfile => {
  const value: unknown = JSON.parse(text);
  if (!isRecord(value)) throw Error("オブジェクトでない");
  const { savedAt, adapter, checkout, profile } = value;
  if (typeof savedAt !== "string" || Number.isNaN(Date.parse(savedAt))) {
    throw Error("savedAt が日時の文字列でない");
  }
  if (
    !isRecord(adapter) || typeof adapter.vendor !== "string" ||
    typeof adapter.architecture !== "string" || typeof adapter.description !== "string"
  ) throw Error("adapter の vendor / architecture / description が文字列でない");
  if (
    !isRecord(checkout) || typeof checkout.revision !== "string" ||
    typeof checkout.dirty !== "boolean"
  ) {
    throw Error("checkout の revision / dirty の形が違う");
  }
  if (!isRecord(profile) || typeof profile.id !== "string") {
    throw Error("profile の id が文字列でない");
  }
  // 門の本体は runtime と同じ（規則列・幾何の整除条件 — 欄が欠けた表もここで落ちる）
  const geometryProfile = profile as GeometryProfile;
  assertGeometryProfile(geometryProfile);
  return {
    savedAt,
    adapter: {
      vendor: adapter.vendor,
      architecture: adapter.architecture,
      description: adapter.description,
    },
    checkout: { revision: checkout.revision, dirty: checkout.dirty },
    profile: geometryProfile,
  };
};

/**
 * 保存した表を読む。無ければ undefined。読めない・形が違うときはキーを消してから理由つきで投げる
 * （壊れた値を次の起動でも読み続けない）。Storage の例外はそのまま投げる。
 */
export const readLastGenerated = (storage: ProfileStorage): SavedProfile | undefined => {
  const text = storage.getItem(LAST_GENERATED_KEY);
  if (text === null) return undefined;
  try {
    return parseSaved(text);
  } catch (cause) {
    storage.removeItem(LAST_GENERATED_KEY);
    throw Error(
      `${LAST_GENERATED_KEY} の保存した表を読めないので消した（${
        cause instanceof Error ? cause.message : String(cause)
      }）`,
      { cause },
    );
  }
};
