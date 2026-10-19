// Dimension language: a linear expression `coeff·sym+offset` for a symbolic dimension of a shape
// (one symbol per dimension). The source of truth for the grammar is
// tests/fixtures/dim-grammar.json (three sections: valid / invalid / eval), and the TS
// implementation and the Python one (tools/exporter/src/karume/dims.py) are both checked against
// the same table (making an implementation the source of truth would leave syncing to manual
// discipline).

/** `coeff·sym + offset`. Only non-negative linear expressions with coeff ≥ 1 and offset ≥ 0. */
type DimExpr = {
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
 * Spelling out the optional parts (coeff=1 / offset=0) and leading zeros are rejected as
 * non-canonical. Two or more spellings for the same dimension MUST NOT be allowed: the binding
 * table, cache keys and shape-identity checks would split on spelling variants.
 */
const parsePart = (text: string | undefined, min: number, omitted: number): number | undefined => {
  if (text === undefined) return omitted;
  if (text.startsWith("0")) return undefined;
  const value = Number(text);
  if (!Number.isSafeInteger(value) || value < min) return undefined;
  return value;
};

/**
 * Decomposes a canonical spelling; non-canonical or non-matching text gives undefined (check and
 * parse in one).
 */
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
    throw new DimError(
      `Dimension expression '${text}' does not match the canonical grammar coeff·sym+offset`,
    );
  }
  return expr;
};

/**
 * Back to the canonical spelling. A round trip with parseDim is string-identical (pinned by the
 * conformance table).
 */
export const formatDim = (expr: DimExpr): string => {
  const { coeff, sym, offset } = expr;
  if (!isSymbolName(sym)) throw new DimError(`Invalid symbol name '${sym}'`);
  if (!Number.isSafeInteger(coeff) || coeff < 1) throw new DimError(`Invalid coefficient ${coeff}`);
  if (!Number.isSafeInteger(offset) || offset < 0) {
    throw new DimError(`Invalid offset ${offset}`);
  }
  return `${coeff === 1 ? "" : coeff}${sym}${offset === 0 ? "" : `+${offset}`}`;
};

/**
 * Applies the binding table to get a concrete dimension.
 * Whether a symbol is bound MUST be decided by Object.hasOwn only — `bindings[sym] !== undefined`
 * lets toString and the like from Object.prototype through, and the arithmetic after it silently
 * turns into NaN.
 */
export const evalDim = (expr: DimExpr, bindings: Readonly<Record<string, number>>): number => {
  if (!Object.hasOwn(bindings, expr.sym)) {
    throw new DimError(`Symbol '${expr.sym}' is not bound`);
  }
  const bound = bindings[expr.sym];
  if (typeof bound !== "number" || !Number.isSafeInteger(bound) || bound < 0) {
    throw new DimError(
      `Symbol '${expr.sym}' is bound to ${String(bound)}, which is not a non-negative integer`,
    );
  }
  const size = expr.coeff * bound + expr.offset;
  if (!Number.isSafeInteger(size)) {
    throw new DimError(`Value of dimension ${formatDim(expr)} exceeds the safe integer range`);
  }
  return size;
};

/**
 * The inverse of {@link evalDim} — solves the binding of `sym` from an actual size. **The solution
 * is unique** (a linear expression with coeff ≥ 1 and offset ≥ 0 has only `(size − offset) / coeff`),
 * and a size that does not divide evenly "does not have the declared shape" = no binding exists, so
 * `undefined` is returned.
 *
 * MUST: do not accept a size that does not divide evenly by rounding or truncating it. When an input
 * of a derived dimension (`2T`) has an odd length, rounding it through **silently changes the meaning
 * of the input's last element** (the caller assumes the graph accepted it). The caller is
 * responsible for turning `undefined` into a loud failure.
 */
export const solveDim = (expr: DimExpr, size: number): number | undefined => {
  if (!Number.isSafeInteger(size) || size < 0) return undefined;
  const rest = size - expr.offset;
  if (rest < 0 || rest % expr.coeff !== 0) return undefined;
  return rest / expr.coeff;
};
