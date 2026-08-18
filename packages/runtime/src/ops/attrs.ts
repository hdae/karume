import { isSymbolName } from "../format/dims.ts";
import { type IrDtype, isSemanticDtype, SEMANTIC_DTYPES } from "../format/ir.ts";
import { OpContractError } from "./names.ts";

/**
 * attrs スキーマ = attr キー → 値の検査（ADR 0012）。
 *
 * MUST: 宣言したキーは**全て必須**、宣言外のキーは fail loudly。省略可能な attr を持つ op
 * （融合 op の bias 有無など）が出た時点で表現を広げる — 先回りして機構だけ増やさない。
 */
export type AttrSchema = Readonly<Record<string, (value: unknown, where: string) => void>>;
/**
 * cast の変換先を検査する。IR の意味論 dtype 語彙そのものを受理する（同型 cast は恒等コピー）。
 *
 * MUST: f32 → i32 は torch 準拠の **truncate（0 方向切り捨て）**、x → bool は **x != 0**。
 * bool の実表現は u32 の 0 / 1（ADR 0009）。丸め規約を契約に明記しないと、GPU 側の
 * `i32(x)`（WGSL も truncate）と CPU 参照が静かにずれる。
 */
const assertCastTarget = (value: unknown, where: string): IrDtype => {
  if (!isSemanticDtype(value)) {
    throw new OpContractError(
      `${where}: cast の変換先が意味論 dtype でない（${SEMANTIC_DTYPES.join(" / ")}）: ${
        JSON.stringify(value)
      }`,
    );
  }
  return value;
};

/** cast ノードの変換先 dtype（attrs の検査は {@link assertNodeContract} が済ませている前提）。 */
export const castTargetDtype = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): IrDtype =>
  assertCastTarget(Object.hasOwn(attrs, "to") ? attrs["to"] : undefined, `${where} の attrs.to`);

export const CAST_ATTRS: AttrSchema = {
  to: (value, where) => {
    assertCastTarget(value, where);
  },
};

/**
 * permute の軸並べ替え表を検査する。`dims[d]` = 出力の次元 d が取る入力の次元番号。
 *
 * MUST: 負の軸番号を受理しない。torch の `-1` 表記はエクスポータ境界で正規化する規約で、
 * ランタイム側で両表記を受けると同じ並べ替えに 2 通りの IR ができる。
 * MUST: 重複を拒否する（並べ替えは全単射）。重複を許すと同じ入力軸を 2 度読む「複製」に
 * なり、要素数が合わないまま stride 計算だけが通る。入力 rank との突合は shape 計算側
 * （rank は束縛解決後にしか分からない）。
 */
const assertPermuteDims = (value: unknown, where: string): readonly number[] => {
  if (!Array.isArray(value) || value.length === 0) {
    throw new OpContractError(
      `${where}: permute の dims が非空の配列でない: ${JSON.stringify(value)}`,
    );
  }
  const dims: number[] = value.map((dim) => {
    if (typeof dim !== "number" || !Number.isSafeInteger(dim) || dim < 0) {
      throw new OpContractError(
        `${where}: permute の dims に非負整数でない要素がある: ${JSON.stringify(value)}`,
      );
    }
    return dim;
  });
  if (new Set(dims).size !== dims.length) {
    throw new OpContractError(`${where}: permute の dims [${dims.join(",")}] に重複がある`);
  }
  return dims;
};

/** permute ノードの軸並べ替え表（attrs の検査は {@link assertNodeContract} が済ませている前提）。 */
export const permuteDims = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): readonly number[] =>
  assertPermuteDims(
    Object.hasOwn(attrs, "dims") ? attrs["dims"] : undefined,
    `${where} の attrs.dims`,
  );

export const PERMUTE_ATTRS: AttrSchema = {
  dims: (value, where) => {
    assertPermuteDims(value, where);
  },
};

/** attrs の値を prototype 汚染に触れずに引く（`Object.hasOwn` のみ — 横断の不変条件）。 */
export const attrValue = (attrs: Readonly<Record<string, unknown>>, key: string): unknown =>
  Object.hasOwn(attrs, key) ? attrs[key] : undefined;

/**
 * `min` 以上の整数 attr。
 * MUST: `typeof value === "number"` を先に見る（`Number.isSafeInteger` は非数値に false を
 * 返すので通るが、真偽値は数値化されないまま比較を素通りする形が JSON 由来で入りうる）。
 */
const assertIntegerAttr = (value: unknown, where: string, min: number): number => {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < min) {
    throw new OpContractError(`${where}: ${min} 以上の整数でない: ${JSON.stringify(value)}`);
  }
  return value;
};

/**
 * slice の切り出し指定（ADR 0014）。`dim` 軸を `[start, end)` に縮める。
 *
 * MUST: 3 つとも**非負整数**（負の軸表記・負の添字表記はエクスポータ境界で正規化する規約 —
 * permute の dims と同じ）。torch の `x[..., -1]` や既定の巨大 end をそのまま持ち込むと、
 * 同じ切り出しに 2 通りの IR ができる。
 * MUST: `start <= end` と `end <= 入力の軸長` は shape 計算側が見る（キーを跨ぐ不変条件と、
 * 入力 shape が要る規則は attrs スキーマでは表せない — clamp の min/max と同じ分担）。
 */
export type SliceAttrs = {
  readonly dim: number;
  readonly start: number;
  readonly end: number;
};

export const SLICE_ATTRS: AttrSchema = {
  dim: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
  start: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
  end: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
};

/** slice ノードの attrs（検査は {@link assertNodeContract} が済ませている前提）。 */
export const sliceAttrs = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): SliceAttrs => ({
  dim: assertIntegerAttr(attrValue(attrs, "dim"), `${where} の attrs.dim`, 0),
  start: assertIntegerAttr(attrValue(attrs, "start"), `${where} の attrs.start`, 0),
  end: assertIntegerAttr(attrValue(attrs, "end"), `${where} の attrs.end`, 0),
});

/** cat / flip の軸（非負の軸番号。入力 rank との突合は shape 計算側）。 */
export const AXIS_ATTRS: AttrSchema = {
  dim: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
};

export const catDim = (attrs: Readonly<Record<string, unknown>>, where: string): number =>
  assertIntegerAttr(attrValue(attrs, "dim"), `${where} の attrs.dim`, 0);

export const flipDim = (attrs: Readonly<Record<string, unknown>>, where: string): number =>
  assertIntegerAttr(attrValue(attrs, "dim"), `${where} の attrs.dim`, 0);

/**
 * pad の左右パディング幅（**最終次元・定数 0 のみ** — ADR 0014）。
 *
 * MUST: 負の幅（torch の `constant_pad_nd` は負で切り詰めができる）を受理しない。切り詰めは
 * pad ではなく slice の意味で、通すと「同じ形を 2 つの op で書ける」うえに専用カーネルの
 * 出力長計算が負になる。
 * MUST: 埋め値を attrs に持たない。実測は 0 のみで、欄を作らないことが「0 以外を黙って 0 で
 * 実行する」経路を構造的に潰す（conv1d の groups と同じ絞り方）。
 */
export type PadAttrs = {
  readonly left: number;
  readonly right: number;
};

export const PAD_ATTRS: AttrSchema = {
  left: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
  right: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
};

export const padAttrs = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): PadAttrs => ({
  left: assertIntegerAttr(attrValue(attrs, "left"), `${where} の attrs.left`, 0),
  right: assertIntegerAttr(attrValue(attrs, "right"), `${where} の attrs.right`, 0),
});

/** sym_prefix_slice の 1 軸ぶんの切り出し指定（`dim` を長さ `coeff·sym+offset` に縮める）。 */
export type PrefixSlice = {
  readonly dim: number;
  readonly coeff: number;
  readonly offset: number;
};

const PREFIX_SLICE_KEYS = ["dim", "coeff", "offset"] as const;

/**
 * sym_prefix_slice の `slices` を検査する。
 *
 * MUST: 軸の重複を拒否する。同じ軸に 2 つの指定があると「後勝ち」で片方が黙って消え、
 * 宣言 shape との照合だけが通る形が作れる。
 * MUST: 係数は 1 以上・オフセットは 0 以上（次元言語 `coeff·sym+offset` と同じ値域 —
 * 負を許すと prefix 長が負になり、要素数計算だけが 0 で通る）。
 * MUST: キーは `Object.hasOwn` のみで見る（JSON 由来のオブジェクトは prototype 経由で
 * `toString` 等を「持っている」ように見える — 横断の不変条件）。
 */
const assertPrefixSlices = (value: unknown, where: string): readonly PrefixSlice[] => {
  if (!Array.isArray(value) || value.length === 0) {
    throw new OpContractError(
      `${where}: sym_prefix_slice の slices が非空の配列でない: ${JSON.stringify(value)}`,
    );
  }
  const slices: PrefixSlice[] = value.map((raw, index) => {
    const at = `${where}[${index}]`;
    if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
      throw new OpContractError(`${at}: オブジェクトでない: ${JSON.stringify(raw)}`);
    }
    const entry: Record<string, unknown> = raw as Record<string, unknown>;
    const unknown = Object.keys(entry).filter((key) =>
      !PREFIX_SLICE_KEYS.some((known) => known === key)
    );
    if (unknown.length > 0) {
      throw new OpContractError(`${at}: 未知のキー [${unknown.sort().join(", ")}]`);
    }
    for (const key of PREFIX_SLICE_KEYS) {
      if (!Object.hasOwn(entry, key)) throw new OpContractError(`${at}: キー '${key}' が無い`);
    }
    return {
      dim: assertIntegerAttr(entry["dim"], `${at}.dim`, 0),
      coeff: assertIntegerAttr(entry["coeff"], `${at}.coeff`, 1),
      offset: assertIntegerAttr(entry["offset"], `${at}.offset`, 0),
    };
  });
  const dims = slices.map((slice) => slice.dim);
  if (new Set(dims).size !== dims.length) {
    throw new OpContractError(`${where}: sym_prefix_slice の slices に同じ dim が 2 度ある`);
  }
  return slices;
};

/** sym_prefix_slice の `sym`（次元言語のシンボル名 — 束縛済みかはグラフ側の検査）。 */
const assertPrefixSym = (value: unknown, where: string): string => {
  if (typeof value !== "string" || !isSymbolName(value)) {
    throw new OpContractError(
      `${where}: sym_prefix_slice の sym がシンボル名でない: ${JSON.stringify(value)}`,
    );
  }
  return value;
};

export const SYM_PREFIX_SLICE_ATTRS: AttrSchema = {
  sym: (value, where) => {
    assertPrefixSym(value, where);
  },
  slices: (value, where) => {
    assertPrefixSlices(value, where);
  },
};

/** sym_prefix_slice ノードの attrs（検査は {@link assertNodeContract} が済ませている前提）。 */
export const symPrefixSliceAttrs = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): { readonly sym: string; readonly slices: readonly PrefixSlice[] } => ({
  sym: assertPrefixSym(attrValue(attrs, "sym"), `${where} の attrs.sym`),
  slices: assertPrefixSlices(attrValue(attrs, "slices"), `${where} の attrs.slices`),
});

/**
 * layer_norm の正規化軸。**長さ 1（= 最終次元）のみ**受理する。
 *
 * MUST: 多軸正規化を「対称性のため」受け入れない（ADR 0007 の語彙 allowlist 凍結）。実測は
 * 全 7 本が `[1024]`（recon §5）で、行カーネルは最終次元の連続並びを前提に組んである。
 * 軸との突合（`x.shape` の末尾と一致するか）は束縛解決後の shape 計算側。
 */
const assertNormalizedShape = (value: unknown, where: string): readonly number[] => {
  if (!Array.isArray(value) || value.length !== 1) {
    throw new OpContractError(
      `${where}: layer_norm の normalized_shape は長さ 1 の配列のみ（最終次元の正規化だけを実行できる）: ${
        JSON.stringify(value)
      }`,
    );
  }
  return [assertIntegerAttr(value[0], `${where}[0]`, 1)];
};

/**
 * 正規化 op（layer_norm / rms_norm）の eps。**有限の正数**のみ（0 を許すと分散 0・
 * 全要素 0 の行で `1/sqrt(0)` が inf になり、「定数行の正規化」が黙って NaN を吐く）。
 */
const assertEps = (value: unknown, where: string, what: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new OpContractError(
      `${where}: ${what} の eps は有限の正数でない: ${JSON.stringify(value)}`,
    );
  }
  return value;
};

/**
 * params の f32 語で運ぶスカラ attr。**有限の f32 スカラ**（IR v1 は非有限値を JSON
 * リテラルでも値レベルでも拒否する）。
 *
 * NOTE: f32 に厳密表現できる値だけに絞りはしない — 適用時に f32 へ丸める規約（GPU は
 * params の f32 語、CPU 参照は `Math.fround`）で両側が一致する。
 */
export const assertFiniteAttr = (value: unknown, where: string, what: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new OpContractError(`${where}: ${what} は有限の数値でない: ${JSON.stringify(value)}`);
  }
  return value;
};

/**
 * masked_fill の埋め値。実測値 −3.4028234663852886e+38 は f32 の最小有限値ちょうどで、
 * JSON 往復でも f32 丸めでも ulp が動かない（tests/ops_contract_test.ts が固定）。
 */
const assertFillValue = (value: unknown, where: string): number =>
  assertFiniteAttr(value, where, "masked_fill の value");

export const LAYER_NORM_ATTRS: AttrSchema = {
  eps: (value, where) => {
    assertEps(value, where, "layer_norm");
  },
  normalized_shape: (value, where) => {
    assertNormalizedShape(value, where);
  },
};

/**
 * rms_norm の attrs（ADR 0017）。
 *
 * MUST: `normalized_shape` の欄を作らない。正規化軸は最終次元 1 本に固定で、長さは
 * **weight の長さ**が持つ（layer_norm は attrs と weight で同じ事実を二重に持っていて、
 * shape 計算がその一致を毎回検査している）。欄を作ると同じ二重管理をもう 1 op ぶん増やす。
 */
export const RMS_NORM_ATTRS: AttrSchema = {
  eps: (value, where) => {
    assertEps(value, where, "rms_norm");
  },
};

/** rms_norm ノードの eps（検査は {@link assertNodeContract} が済ませている前提）。 */
export const rmsNormEps = (attrs: Readonly<Record<string, unknown>>, where: string): number =>
  assertEps(attrValue(attrs, "eps"), `${where} の attrs.eps`, "rms_norm");

export const SOFTMAX_ATTRS: AttrSchema = {
  // 負の軸表記はエクスポータ境界で正規化する規約（permute の dims と同じ）。最終次元との
  // 突合は rank が分かる shape 計算側。
  dim: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
};

/**
 * attention の attrs（ADR 0023）。
 *
 * MUST: `scale` は**宣言必須で既定値補完をしない**。エクスポータが SDPA の `scale` 引数
 * （省略時 `1/√D`）から `f32(√scale_factor)` を計算して載せる規約で、ランタイム側が
 * 「無ければ 1/√D」を補うと、`scale` を明示した SDPA の IR と補完した IR が同じ形になり
 * 取り違えが値にしか出なくなる（conv1d の dilation / groups と同じ規律 — ADR 0015）。
 * MUST: mask / causal / dropout の欄を作らない。欄が無いこと自体が「その形は語彙に無い」を
 * 構造で表す（pad の埋め値と同じ絞り方）。
 */
export const ATTENTION_ATTRS: AttrSchema = {
  scale: (value, where) => {
    assertFiniteAttr(value, where, "attention の scale");
  },
};

/** attention ノードの半スケール（検査は {@link assertNodeContract} が済ませている前提）。 */
export const attentionScale = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): number =>
  assertFiniteAttr(attrValue(attrs, "scale"), `${where} の attrs.scale`, "attention の scale");

/**
 * state 参照ノードの sliding window（ADR 0067 決定 4 / 5）。**省略可能な attr** で、
 * 欄の不存在が「全 context」を意味する。宣言されたら 1 以上の整数。
 *
 * MUST: 既定値で補完しない。「欄が無い = 全 context」と「window = 容量」は別の宣言で、
 * 補完すると sliding 層（Gemma 4 E2B の 28 層）と full 層（7 層）の取り違えが値にしか出ない。
 * MUST: 同一スロットに触れる全ノードで**存在有無も値も一致**する（読み書き同式 MUST —
 * 検査は `runtime/plan.ts` の `validateGraphContracts`。読み側だけ別式にすると沈黙誤読）。
 */
export const STATE_WINDOW_ATTRS: AttrSchema = {
  window: (value, where) => {
    assertIntegerAttr(value, where, 1);
  },
};

/** state 参照ノードの `window`（宣言が無ければ `undefined` = 全 context）。 */
export const stateWindow = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): number | undefined =>
  Object.hasOwn(attrs, "window")
    ? assertIntegerAttr(attrs["window"], `${where} の attrs.window`, 1)
    : undefined;

export const EMBEDDING_ATTRS: AttrSchema = {
  /**
   * torch の `padding_idx`。**受理するが forward には効かない**（勾配で padding 行を更新
   * しないための欄で、順伝播は素の行 gather と完全に同じ — torch の `F.embedding` も
   * forward では参照しない）。したがってカーネルにも CPU 参照にも渡さない。
   *
   * MUST: それでも契約表に載せる。attrs を落として無視すると「未知 attr は fail loudly」の
   * 規律に穴が開き、次に forward へ効く欄が来たときに同じ理由で素通りする。
   * 値域は torch 準拠で `-1`（未指定を表す番兵）以上。
   */
  padding_idx: (value, where) => {
    assertIntegerAttr(value, where, -1);
  },
};

export const MASKED_FILL_ATTRS: AttrSchema = {
  value: (value, where) => {
    assertFillValue(value, where);
  },
};

export const CLAMP_ATTRS: AttrSchema = {
  min: (value, where) => {
    assertFiniteAttr(value, where, "clamp の min");
  },
  max: (value, where) => {
    assertFiniteAttr(value, where, "clamp の max");
  },
};

/**
 * clamp_min の attrs（ADR 0017 — チャネル L2 正規化の `clamp(min=eps)` 30 本）。
 *
 * MUST: `max` の欄を作らない。「欠けた側を f32 の最大有限値で補って clamp へ流す」は
 * 上限を持たない意味論を「上限が飽和するほど大きい」で置き換える近似で、ADR 0017 が
 * 名指しで却下している。欄が無いこと自体が両側必須の clamp との住み分けを構造で表す。
 */
export const CLAMP_MIN_ATTRS: AttrSchema = {
  min: (value, where) => {
    assertFiniteAttr(value, where, "clamp_min の min");
  },
};

export const LEAKY_RELU_ATTRS: AttrSchema = {
  /**
   * torch の `negative_slope`。**必須で既定値補完はしない**（ADR 0015）— dec は 0.1
   * （ups / ResBlock）と 0.01（最終段・位置引数ごと省略）が混在し、既定に頼ると片方が
   * 黙って誤る。torch 側の既定はエクスポータが読み取って attrs に載せる（境界で明示化する）。
   */
  negative_slope: (value, where) => {
    assertFiniteAttr(value, where, "leaky_relu の negative_slope");
  },
};

/** ge_scalar / le_scalar / gt_scalar の比較相手（有限の f32 スカラ）。 */
export const SCALAR_COMPARE_ATTRS: AttrSchema = {
  value: (value, where) => {
    assertFiniteAttr(value, where, "比較 op の value");
  },
};

export const CUMSUM_ATTRS: AttrSchema = {
  // 負の軸表記はエクスポータ境界で正規化する規約（softmax の dim と同じ）。最終次元との
  // 突合は rank が分かる shape 計算側。
  dim: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
};

/**
 * reduce 族（sum / amax / amin）の縮約軸。
 *
 * MUST: **宣言必須**（既定値補完をしない — conv1d の dilation / groups と同じ理由）。
 * 「欄が無い = 最終次元」を許すと、チャネル軸の縮約を書いたつもりの IR が黙って最終次元を
 * 畳んだ別の計算として実行される（形が合ってしまう組み合わせが実在する）。
 * 負の軸表記はエクスポータ境界で正規化する規約で、rank との突合は shape 計算側。
 */
export const REDUCE_ATTRS: AttrSchema = {
  dim: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
};

/**
 * topk の本数 `k`（ADR 0068 決定 3）。
 *
 * MUST: **宣言必須の正整数**（= static-k）。k=0 は torch なら受理される形（空の出力）だが、
 * 「大きい順の 0 本」を返すノードは値を定義しないのと同じなので語彙に入れない。記号 k
 * （次元式の文字列）も `typeof value !== "number"` でここが弾く — 実行時に決まる k は静的形状
 * の前提（ADR 0004）に載らず、出力 shape も確保バイト数も計画時に決まらなくなる。
 * MUST: 上限は**ここでは見ない**。`k ≤ 最終次元` は入力 rank が要るので shape 計算側、
 * **実装上限**（workgroup storage 由来）は device limit が要るのでレシピ組み立て側
 * （src/kernels/topk.ts の `assertTopkK`）— 層の分担は clamp の min/max と同じ。
 */
export const TOPK_ATTRS: AttrSchema = {
  k: (value, where) => {
    assertIntegerAttr(value, where, 1);
  },
};

/**
 * conv1d の attrs（ADR 0015 で `dilation` / `groups` を追加）。
 *
 * MUST: 4 つとも**宣言必須**（{@link assertNodeContract} が全キーの存在を要求する）。
 * 「欄が無い = 1 固定」で担保していた「1 以外を黙って 1 で実行する経路が無い」性質は、
 * 欄を作った後は「既定値補完をしない」ことだけが担保している — `dilation` / `groups` を
 * 省略可能にした瞬間に、depthwise の IR が黙って通常畳み込みとして実行される。
 */
export const CONV1D_ATTRS: AttrSchema = {
  dilation: (value, where) => {
    assertIntegerAttr(value, where, 1);
  },
  groups: (value, where) => {
    assertIntegerAttr(value, where, 1);
  },
  padding: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
  stride: (value, where) => {
    assertIntegerAttr(value, where, 1);
  },
};

/**
 * conv_transpose1d の attrs（ADR 0015）。
 *
 * MUST: `stride >= 1`。stride 0 は「入力 1 点が出力の 1 点にしか寄与しない」形にすら
 * ならず、実装によってはループが進まず **GPU ハング**（例外にならない）— 契約検査と
 * params 検査の両方で遮断する（recon §4）。
 * MUST: `output_padding` / `dilation` / `groups` の欄を作らない。実測は全て 0 / 1 / 1 で、
 * 欄を持たないことが「0 / 1 以外を黙って既定値で実行する」経路を構造的に潰す
 * （pad の埋め値と同じ絞り方）。広げるのは実測が出てから。
 */
export const CONV_TRANSPOSE1D_ATTRS: AttrSchema = {
  padding: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
  stride: (value, where) => {
    assertIntegerAttr(value, where, 1);
  },
};

/** layer_norm ノードの attrs（検査は {@link assertNodeContract} が済ませている前提）。 */
export const layerNormAttrs = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): { readonly normalizedShape: readonly number[]; readonly eps: number } => ({
  normalizedShape: assertNormalizedShape(
    attrValue(attrs, "normalized_shape"),
    `${where} の attrs.normalized_shape`,
  ),
  eps: assertEps(attrValue(attrs, "eps"), `${where} の attrs.eps`, "layer_norm"),
});

/** softmax ノードの縮約軸（非負の軸番号）。 */
export const softmaxDim = (attrs: Readonly<Record<string, unknown>>, where: string): number =>
  assertIntegerAttr(attrValue(attrs, "dim"), `${where} の attrs.dim`, 0);

/** masked_fill ノードの埋め値。 */
export const maskedFillValue = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): number => assertFillValue(attrValue(attrs, "value"), `${where} の attrs.value`);

/** cumsum ノードの累積軸（非負の軸番号）。 */
export const cumsumDim = (attrs: Readonly<Record<string, unknown>>, where: string): number =>
  assertIntegerAttr(attrValue(attrs, "dim"), `${where} の attrs.dim`, 0);

/** reduce 族ノードの縮約軸（非負の軸番号 — 既定値補完はしない）。 */
export const reduceDim = (attrs: Readonly<Record<string, unknown>>, where: string): number =>
  assertIntegerAttr(attrValue(attrs, "dim"), `${where} の attrs.dim`, 0);

/** topk ノードの本数 k（正整数 — 既定値補完はしない）。 */
export const topkK = (attrs: Readonly<Record<string, unknown>>, where: string): number =>
  assertIntegerAttr(attrValue(attrs, "k"), `${where} の attrs.k`, 1);

/**
 * elementwise カーネルへ **params の末尾で** f32 として渡す attr（並びがそのまま params の
 * レイアウト）。
 *
 * MUST: 値を WGSL に焼かない。焼くと値の種類だけパイプラインが増える（masked_fill の
 * 埋め値と同じ理由 — src/kernels/masked-fill.ts）。
 * MUST: 並びの正本はここ 1 箇所。attrs スキーマのキー順（オブジェクトの挿入順）に頼ると、
 * 宣言の並べ替えが codegen の添字と params の書き込み順を黙ってずらす。
 */
export const SCALAR_PARAM_ATTRS: ReadonlyMap<string, readonly string[]> = new Map([
  ["clamp", ["min", "max"]],
  ["clamp_min", ["min"]],
  ["leaky_relu", ["negative_slope"]],
  ["ge_scalar", ["value"]],
  ["le_scalar", ["value"]],
  ["gt_scalar", ["value"]],
]);

/** op が params 末尾に載せる f32 スカラの本数（codegen が添字を決めるのに使う）。 */
export const scalarParamCount = (op: string): number => SCALAR_PARAM_ATTRS.get(op)?.length ?? 0;

/**
 * conv2d の空間 attr（`[H, W]` の 2 成分）。
 *
 * MUST: **長さちょうど 2 の配列**のみ受理する。スカラ表記（torch の `stride=1` が両軸に
 * 効く形）を併せて許すと同じ畳み込みに 2 通りの IR ができ、CSE も適合表の突合も割れる —
 * 正規化はエクスポータ境界の仕事（permute の負の軸表記と同じ分担）。
 * MUST: H と W を別のキーに割らない（`stride_h` / `stride_w`）。2 軸で 3 つの attr なので
 * 6 キーになり、「片方だけ書き忘れた IR」の見え方が「必須キー欠落」から「値が既定に
 * 見える」へ落ちる。組であることを型で表す。
 */
const assertIntPair = (
  value: unknown,
  where: string,
  min: number,
  what: string,
): readonly [number, number] => {
  if (!Array.isArray(value) || value.length !== 2) {
    throw new OpContractError(
      `${where}: ${what} は [H, W] の長さ 2 の配列でない: ${JSON.stringify(value)}`,
    );
  }
  return [
    assertIntegerAttr(value[0], `${where}[0]`, min),
    assertIntegerAttr(value[1], `${where}[1]`, min),
  ];
};

/**
 * conv2d の attrs（ADR 0017）。空間 3 つは H/W の 2 成分、`groups` はスカラ。
 *
 * MUST: 4 つとも**宣言必須・既定値補完なし**（conv1d と同じ規律 — ADR 0015）。
 * depthwise（groups = Cin = Cout）と非対称 stride/padding が実測に出るので、省略を許すと
 * 黙って通常畳み込み・対称パディングとして実行される。
 */
export const CONV2D_ATTRS: AttrSchema = {
  dilation: (value, where) => {
    assertIntPair(value, where, 1, "conv2d の dilation");
  },
  groups: (value, where) => {
    assertIntegerAttr(value, where, 1);
  },
  padding: (value, where) => {
    assertIntPair(value, where, 0, "conv2d の padding");
  },
  stride: (value, where) => {
    assertIntPair(value, where, 1, "conv2d の stride");
  },
};

/** conv2d ノードの stride / padding / dilation（H/W の組）と groups。 */
export type Conv2dAttrs = {
  readonly stride: readonly [number, number];
  readonly padding: readonly [number, number];
  readonly dilation: readonly [number, number];
  readonly groups: number;
};

export const conv2dAttrs = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): Conv2dAttrs => ({
  stride: assertIntPair(
    attrValue(attrs, "stride"),
    `${where} の attrs.stride`,
    1,
    "conv2d の stride",
  ),
  padding: assertIntPair(
    attrValue(attrs, "padding"),
    `${where} の attrs.padding`,
    0,
    "conv2d の padding",
  ),
  dilation: assertIntPair(
    attrValue(attrs, "dilation"),
    `${where} の attrs.dilation`,
    1,
    "conv2d の dilation",
  ),
  groups: assertIntegerAttr(attrValue(attrs, "groups"), `${where} の attrs.groups`, 1),
});

/**
 * deform_conv2d の attrs（第 1' 層・ADR 0055）。**`padding`（`[H, W]`）の 1 キーだけ**。
 *
 * MUST: `stride` / `dilation` / `groups` / `offset_groups` のキーを**足さない**。実測が
 * 全て 1 で、欄の不存在が「その形は語彙に無い」を構造で表す（エクスポータ境界の
 * fail loudly と対）。値の形は conv2d と同じ `[H, W]` の長さ 2 の配列。
 */
export const DEFORM_CONV2D_ATTRS: AttrSchema = {
  padding: (value, where) => {
    assertIntPair(value, where, 0, "deform_conv2d の padding");
  },
};

/** deform_conv2d ノードの空間パディング（`[H, W]`）。 */
export type DeformConv2dAttrs = {
  readonly padding: readonly [number, number];
};

export const deformConv2dAttrs = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): DeformConv2dAttrs => ({
  padding: assertIntPair(
    attrValue(attrs, "padding"),
    `${where} の attrs.padding`,
    0,
    "deform_conv2d の padding",
  ),
});

/**
 * upsample_bilinear2d の attrs（第 1 層）。**`output_size`（`[Hout, Wout]`）の 1 キーだけ**。
 *
 * MUST: conv2d と同じ `[H, W]` の長さ 2 の配列で、スカラ表記は受理しない（同じ resample に
 * 2 通りの IR ができる — 正規化はエクスポータ境界の仕事）。
 * MUST: `align_corners` / `mode` / `scale_factor` の欄を**足さない**。欄の不存在がそのまま
 * 「その形は語彙に無い」の宣言で、エクスポータ側の fail loudly と対になっている。
 */
export const UPSAMPLE_BILINEAR2D_ATTRS: AttrSchema = {
  output_size: (value, where) => {
    assertIntPair(value, where, 1, "upsample_bilinear2d の output_size");
  },
};

/** upsample_bilinear2d ノードの出力空間長（`[Hout, Wout]`）。 */
export type UpsampleBilinear2dAttrs = {
  readonly outputSize: readonly [number, number];
};

export const upsampleBilinear2dAttrs = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): UpsampleBilinear2dAttrs => ({
  outputSize: assertIntPair(
    attrValue(attrs, "output_size"),
    `${where} の attrs.output_size`,
    1,
    "upsample_bilinear2d の output_size",
  ),
});

/** conv1d ノードの stride / padding / dilation / groups。 */
export type Conv1dAttrs = {
  readonly stride: number;
  readonly padding: number;
  readonly dilation: number;
  readonly groups: number;
};

export const conv1dAttrs = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): Conv1dAttrs => ({
  stride: assertIntegerAttr(attrValue(attrs, "stride"), `${where} の attrs.stride`, 1),
  padding: assertIntegerAttr(attrValue(attrs, "padding"), `${where} の attrs.padding`, 0),
  dilation: assertIntegerAttr(attrValue(attrs, "dilation"), `${where} の attrs.dilation`, 1),
  groups: assertIntegerAttr(attrValue(attrs, "groups"), `${where} の attrs.groups`, 1),
});

/** conv_transpose1d ノードの stride / padding。 */
export const convTranspose1dAttrs = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): { readonly stride: number; readonly padding: number } => ({
  stride: assertIntegerAttr(attrValue(attrs, "stride"), `${where} の attrs.stride`, 1),
  padding: assertIntegerAttr(attrValue(attrs, "padding"), `${where} の attrs.padding`, 0),
});
