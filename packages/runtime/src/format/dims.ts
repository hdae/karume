// 次元言語: shape の記号次元を表す一次式 `coeff·sym+offset`（1 次元 1 シンボル）。
// 文法の正本は tests/fixtures/dim-grammar.json（valid / invalid / eval の 3 節）で、
// TS 実装と将来の Python 実装は同じ表で検証する（実装を正本にすると同期が人手の規律になる）。

/** `coeff·sym + offset`。coeff ≥ 1・offset ≥ 0 の非負一次式のみ。 */
export type DimExpr = {
  readonly coeff: number;
  readonly sym: string;
  readonly offset: number;
};

export class DimError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DimError";
  }
}

const SYMBOL_PATTERN = "[A-Za-z_][A-Za-z0-9_]*";
const SYMBOL_RE = new RegExp(`^${SYMBOL_PATTERN}$`);
const DIM_RE = new RegExp(`^(\\d+)?(${SYMBOL_PATTERN})(?:\\+(\\d+))?$`);

export const isSymbolName = (name: string): boolean => SYMBOL_RE.test(name);

/**
 * 省略可能部（coeff=1 / offset=0）の明示表記と先頭ゼロは非正準として拒否する。
 * 同じ次元に 2 通り以上の表記を許すと、束縛表・キャッシュキー・shape 同一性判定が
 * 表記ゆれで割れる MUST NOT。
 */
const parsePart = (text: string | undefined, min: number, omitted: number): number | undefined => {
  if (text === undefined) return omitted;
  if (text.startsWith("0")) return undefined;
  const value = Number(text);
  if (!Number.isSafeInteger(value) || value < min) return undefined;
  return value;
};

/** 正準表記なら分解、非正準・非該当は undefined（判定と分解を兼ねる版）。 */
export const tryParseDim = (text: string): DimExpr | undefined => {
  const matched = DIM_RE.exec(text);
  if (matched === null) return undefined;
  const sym = matched[2];
  if (sym === undefined) return undefined;
  const coeff = parsePart(matched[1], 2, 1);
  const offset = parsePart(matched[3], 1, 0);
  if (coeff === undefined || offset === undefined) return undefined;
  return { coeff, sym, offset };
};

export const parseDim = (text: string): DimExpr => {
  const expr = tryParseDim(text);
  if (expr === undefined) {
    throw new DimError(`次元式 '${text}' が正準文法 coeff·sym+offset に適合しない`);
  }
  return expr;
};

/** 正準表記へ戻す。parseDim との往復は文字列同一（適合ケース表で固定）。 */
export const formatDim = (expr: DimExpr): string => {
  const { coeff, sym, offset } = expr;
  if (!isSymbolName(sym)) throw new DimError(`シンボル名 '${sym}' が不正`);
  if (!Number.isSafeInteger(coeff) || coeff < 1) throw new DimError(`係数 ${coeff} が不正`);
  if (!Number.isSafeInteger(offset) || offset < 0) {
    throw new DimError(`オフセット ${offset} が不正`);
  }
  return `${coeff === 1 ? "" : coeff}${sym}${offset === 0 ? "" : `+${offset}`}`;
};

/**
 * 束縛表を当てて具体次元にする。
 * 束縛の有無は Object.hasOwn のみで判定する MUST — `bindings[sym] !== undefined` は
 * Object.prototype 由来の toString 等が素通りし、以後の算術が黙って NaN 化する。
 */
export const evalDim = (expr: DimExpr, bindings: Readonly<Record<string, number>>): number => {
  if (!Object.hasOwn(bindings, expr.sym)) {
    throw new DimError(`シンボル '${expr.sym}' が束縛されていない`);
  }
  const bound = bindings[expr.sym];
  if (typeof bound !== "number" || !Number.isSafeInteger(bound) || bound < 0) {
    throw new DimError(`シンボル '${expr.sym}' の束縛 ${String(bound)} が非負整数でない`);
  }
  const size = expr.coeff * bound + expr.offset;
  if (!Number.isSafeInteger(size)) {
    throw new DimError(`次元 ${formatDim(expr)} の評価結果が安全整数を超える`);
  }
  return size;
};

/**
 * {@link evalDim} の逆 — 実寸から `sym` の束縛を解く。**解は一意**（coeff ≥ 1・offset ≥ 0 の
 * 一次式なので `(size − offset) / coeff` の 1 つきり）で、割り切れない実寸は「その宣言の形を
 * していない」= 束縛が存在しないので `undefined` を返す。
 *
 * MUST: 割り切れない実寸を四捨五入や切り捨てで受けない。派生次元（`2T`）の入力に半端な長さが
 * 来たとき、丸めて通すと**入力の末尾 1 要素が黙って別の意味になる**（呼び手はグラフが
 * 受理したと見なす）。呼び手は `undefined` を fail loudly に変換する責務を負う。
 */
export const solveDim = (expr: DimExpr, size: number): number | undefined => {
  if (!Number.isSafeInteger(size) || size < 0) return undefined;
  const rest = size - expr.offset;
  if (rest < 0 || rest % expr.coeff !== 0) return undefined;
  return rest / expr.coeff;
};
