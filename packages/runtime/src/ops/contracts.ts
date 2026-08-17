import type { OpSupport, RuntimeSupport } from "../format/container.ts";
import { type IrDtype, type IrNode, SEMANTIC_DTYPES } from "../format/ir.ts";
import {
  assertFiniteAttr,
  ATTENTION_ATTRS,
  type AttrSchema,
  attrValue,
  AXIS_ATTRS,
  CAST_ATTRS,
  castTargetDtype,
  CLAMP_ATTRS,
  CLAMP_MIN_ATTRS,
  CONV1D_ATTRS,
  CONV2D_ATTRS,
  CONV_TRANSPOSE1D_ATTRS,
  CUMSUM_ATTRS,
  DEFORM_CONV2D_ATTRS,
  EMBEDDING_ATTRS,
  LAYER_NORM_ATTRS,
  LEAKY_RELU_ATTRS,
  MASKED_FILL_ATTRS,
  PAD_ATTRS,
  PERMUTE_ATTRS,
  REDUCE_ATTRS,
  RMS_NORM_ATTRS,
  SCALAR_COMPARE_ATTRS,
  SCALAR_PARAM_ATTRS,
  SLICE_ATTRS,
  SOFTMAX_ATTRS,
  SYM_PREFIX_SLICE_ATTRS,
  UPSAMPLE_BILINEAR2D_ATTRS,
} from "./attrs.ts";
import {
  ARGMAX_OP,
  ATTENTION_OP,
  BINARY_OPS,
  type BinaryOpName,
  BMM_OP,
  CAST_OP,
  CAT_OP,
  CONV1D_OP,
  CONV2D_OP,
  CONV_TRANSPOSE1D_OP,
  CUMSUM_OP,
  DEFORM_CONV2D_OP,
  EMBEDDING_OP,
  EXPAND_OP,
  FLIP_OP,
  GATHER_OP,
  GRU_SCAN_OPS,
  type GruScanOpName,
  LAYER_NORM_OP,
  LINEAR_OP,
  MASKED_FILL_OP,
  MATMUL_OP,
  OpContractError,
  PAD_OP,
  PERMUTE_OP,
  REDUCE_OPS,
  type ReduceOpName,
  RESHAPE_OP,
  RMS_NORM_OP,
  SAFE_SOFTMAX_OP,
  SLICE_OP,
  SOFTMAX_OP,
  SYM_PREFIX_SLICE_OP,
  UNARY_OPS,
  type UnaryOpName,
  UPSAMPLE_BILINEAR2D_OP,
  WHERE_OP,
} from "./names.ts";

export type OpKind =
  | "unary"
  | "binary"
  | "where"
  | "cumsum"
  | "matmul"
  | "bmm"
  | "gather"
  | "rowReduce"
  | "argmax"
  | "cast"
  | "reshape"
  | "permute"
  | "expand"
  | "slice"
  | "cat"
  | "pad"
  | "flip"
  | "symPrefixSlice"
  | "linear"
  | "layerNorm"
  | "rmsNorm"
  | "softmax"
  | "safeSoftmax"
  | "attention"
  | "embedding"
  | "maskedFill"
  | "conv1d"
  | "conv2d"
  | "convTranspose1d"
  | "deformConv2d"
  | "upsampleBilinear2d"
  | "gruScan";

/**
 * 入力スロットの dtype 契約（ADR 0012 の拡張）。
 *
 * - `uniform` — 全スロットが同じ受理集合を持ち、かつ**互いに同型**であることまで要求する
 *   （出力も同型）。elementwise / matmul / bmm / 行 reduce / レイアウトはこれ。
 * - `perSlot` — スロットごとに受理集合が違い、スロット間の同型は要求しない。出力は
 *   **スロット 0（値の側）と同型**。gather（src=f32 / index=i32 → out=f32）が最初の例。
 *
 * MUST: 受理集合が同じ op を `perSlot` で書かない。`uniform` は `mul(f32, i32)` のような
 * 混合を拒否する規則も担っており、perSlot に潰すとその拒否が黙って消える。
 * MUST: 出力 dtype は**スロット 0 の dtype を写像に通して**導く（{@link ContractBase.outputDtypes}）。
 * op ごとに出力元スロットを指定できる欄は作らない — `where(cond, a, b)` のように値の側が
 * 先頭でない op も、写像（bool → f32）で表せる。欄を増やすと「どのスロットから取るか」と
 * 「何に写すか」の 2 つの自由度が同じ事実を二重に持つ。
 */
export type SlotDtypes =
  | { readonly kind: "uniform"; readonly accept: readonly IrDtype[] }
  | { readonly kind: "perSlot"; readonly slots: readonly (readonly IrDtype[])[] };

type ContractBase = {
  /**
   * 受理する**入力の**意味論 dtype の**和**。capability 射影（{@link RUNTIME_SUPPORT}）と
   * 列挙門の診断が使う射影で、スロット別契約の正本は {@link slotDtypes}。
   *
   * MUST: 手書きせず {@link slotDtypes} から導出する（{@link unionDtypes}）— 二重管理すると
   * 「対応表では実行可、契約検査で落ちる」が生える。
   */
  readonly dtypes: readonly IrDtype[];
  /** スロット別の受理集合と出力導出の正本。 */
  readonly slotDtypes: SlotDtypes;
  /**
   * **出力 slot 別**の「スロット 0 の入力 dtype → その出力の dtype」写像の**列**（ADR 0068
   * 決定 1）。列の長さがその op の出力数で、現状の op は全て長さ 1（= 単一出力の明示化）。
   * 各写像の既定は恒等（入力と同型）で、違うのは実測に出た 3 系統だけ: 比較（f32 → bool）/
   * bool 入力の `sum`（→ i32 のカウント）/ `where`（条件 bool → 値の f32）。
   *
   * MUST: 出力数はこの列の長さから導く（{@link outputCountOf}）— 出力数の欄を別に持たせると
   * 「dtype 列は 2 本なのに宣言出力数は 1 本」という食い違いが書けてしまう。
   * MUST: 各写像の定義域はスロット 0 の受理集合と**完全一致**させる（{@link slotContract} が
   * 恒等で埋める）。部分写像にすると、スロット検査を通った dtype の出力が決まらない穴ができる。
   * NOTE: `cast` だけはこの写像を使わない（出力は attrs.to — {@link resolveNodeDtypes}）。
   */
  readonly outputDtypes: readonly ReadonlyMap<IrDtype, IrDtype>[];
  readonly attrs: AttrSchema;
  /**
   * 入力数が可変の op（現状 `cat` のみ）。true のとき {@link OpContract} の `arity` は
   * **下限**として読む（{@link assertArity}）。
   *
   * MUST: 「アリティ検査を緩める」ためにこの欄を立てない。可変なのは cat の入力本数だけで、
   * 他の op は本数そのものが契約（bias 常時ありのアリティ 3 固定など）。
   * NOTE: capability 射影（{@link slotDtypesOf}）は下限ぶんのスロットしか作らない。uniform
   * 契約では全スロットが同じ受理集合なので、余ったスロットを和で見る列挙門（container.ts）と
   * 結論が一致する。
   */
  readonly variadic?: true;
  /**
   * **末尾に省略可能な入力**を持つ op の入力数の上限（現状 `attention` の mask だけ）。
   * 指定時は `arity` が下限・これが上限で、受理するのはその閉区間の本数だけ。
   *
   * MUST: `variadic` の代わりに使わない。`variadic` は「何本でも」（cat の連結）で、こちらは
   * 「決まったスロットが 1 つ増えるだけ」— 上限を持たない表現に潰すと、余分な入力が
   * 黙って無視される形（カーネルが読まないスロット）を契約が受理してしまう。
   * NOTE: capability 射影（{@link slotDtypesOf}）は**上限ぶん**のスロットを作る（省略可能な
   * スロットも dtype 契約を持つ — uniform なら本体と同じ受理集合）。
   */
  readonly maxArity?: number;
};

/**
 * kind と op 名を判別可能ユニオンで結ぶ。消費側（codegen 選択・CPU 参照）が
 * `op as UnaryOpName` のような取り違えの効かないキャストを書かずに済むようにするため。
 * `arity` は入力の個数で、出力の個数は {@link ContractBase.outputDtypes} の列長
 * （現状の op は全て 1 本）。
 */
export type OpContract =
  | (ContractBase & { readonly kind: "unary"; readonly name: UnaryOpName; readonly arity: 1 })
  | (ContractBase & { readonly kind: "binary"; readonly name: BinaryOpName; readonly arity: 2 })
  | (ContractBase & { readonly kind: "where"; readonly name: typeof WHERE_OP; readonly arity: 3 })
  | (ContractBase & { readonly kind: "cumsum"; readonly name: typeof CUMSUM_OP; readonly arity: 1 })
  | (ContractBase & { readonly kind: "matmul"; readonly name: "matmul"; readonly arity: 2 })
  | (ContractBase & { readonly kind: "bmm"; readonly name: typeof BMM_OP; readonly arity: 2 })
  | (ContractBase & {
    readonly kind: "gather";
    readonly name: typeof GATHER_OP;
    readonly arity: 2;
  })
  | (ContractBase & { readonly kind: "rowReduce"; readonly name: ReduceOpName; readonly arity: 1 })
  // 最終次元の argmax（ADR 0068 決定 2）。reduce 族と kind を分けるのは、attrs（無し）・
  // 出力 dtype（i32）・rank の扱い（保存）・カーネル族（identity −inf + index 追跡）の
  // 4 面が全て違い、消費側の網羅 switch に「どちらの意味論で実行するか」を必ず宣言させる
  // ため（safe_softmax を softmax と別 kind にしたのと同じ理由）。
  | (ContractBase & { readonly kind: "argmax"; readonly name: typeof ARGMAX_OP; readonly arity: 1 })
  | (ContractBase & { readonly kind: "cast"; readonly name: typeof CAST_OP; readonly arity: 1 })
  | (ContractBase & {
    readonly kind: "reshape";
    readonly name: typeof RESHAPE_OP;
    readonly arity: 1;
  })
  | (ContractBase & {
    readonly kind: "permute";
    readonly name: typeof PERMUTE_OP;
    readonly arity: 1;
  })
  | (ContractBase & {
    readonly kind: "expand";
    readonly name: typeof EXPAND_OP;
    readonly arity: 1;
  })
  | (ContractBase & { readonly kind: "slice"; readonly name: typeof SLICE_OP; readonly arity: 1 })
  // 唯一の可変アリティ op。`arity` は**下限** 2（1 本だけの cat は恒等コピーで、実測にも
  // 出ない — 表現を持たないことがそのまま fail loudly になる）。
  | (ContractBase & {
    readonly kind: "cat";
    readonly name: typeof CAT_OP;
    readonly arity: 2;
    readonly variadic: true;
  })
  | (ContractBase & { readonly kind: "pad"; readonly name: typeof PAD_OP; readonly arity: 1 })
  | (ContractBase & { readonly kind: "flip"; readonly name: typeof FLIP_OP; readonly arity: 1 })
  | (ContractBase & {
    readonly kind: "symPrefixSlice";
    readonly name: typeof SYM_PREFIX_SLICE_OP;
    readonly arity: 1;
  })
  | (ContractBase & { readonly kind: "linear"; readonly name: typeof LINEAR_OP; readonly arity: 3 })
  | (ContractBase & {
    readonly kind: "layerNorm";
    readonly name: typeof LAYER_NORM_OP;
    readonly arity: 3;
  })
  // affine が weight だけなのでアリティ 2（bias の欄を作らない — ADR 0017）。
  | (ContractBase & {
    readonly kind: "rmsNorm";
    readonly name: typeof RMS_NORM_OP;
    readonly arity: 2;
  })
  | (ContractBase & {
    readonly kind: "softmax";
    readonly name: typeof SOFTMAX_OP;
    readonly arity: 1;
  })
  // softmax と同一契約の別 op（空行 → 全 0。ADR 0044）。kind を分けるのは、executor / CPU
  // 参照の網羅 switch に「どちらの意味論で実行するか」を必ず宣言させるため。
  | (ContractBase & {
    readonly kind: "safeSoftmax";
    readonly name: typeof SAFE_SOFTMAX_OP;
    readonly arity: 1;
  })
  // 融合 attention（ADR 0023）。arity 3〜4（4 本目は省略可能な加算 mask）・attrs `scale`
  // 宣言必須・rank-4 head-first。
  | (ContractBase & {
    readonly kind: "attention";
    readonly name: typeof ATTENTION_OP;
    readonly arity: 3;
    readonly maxArity: 4;
  })
  | (ContractBase & {
    readonly kind: "embedding";
    readonly name: typeof EMBEDDING_OP;
    readonly arity: 2;
  })
  | (ContractBase & {
    readonly kind: "maskedFill";
    readonly name: typeof MASKED_FILL_OP;
    readonly arity: 2;
  })
  | (ContractBase & {
    readonly kind: "conv1d";
    readonly name: typeof CONV1D_OP;
    readonly arity: 3;
  })
  | (ContractBase & {
    readonly kind: "conv2d";
    readonly name: typeof CONV2D_OP;
    readonly arity: 3;
  })
  | (ContractBase & {
    readonly kind: "convTranspose1d";
    readonly name: typeof CONV_TRANSPOSE1D_OP;
    readonly arity: 3;
  })
  // DCNv2（ADR 0055）。x / weight / offset / mask / bias の 5 本固定 — mask がスロットとして
  // 必須であることが「DCNv1 は語彙に無い」を構造で表す。
  | (ContractBase & {
    readonly kind: "deformConv2d";
    readonly name: typeof DEFORM_CONV2D_OP;
    readonly arity: 5;
  })
  // 双線形 resample（第 1 層）。入力は x 1 本だけ（重みも bias も無い）で、出力空間は
  // attrs `output_size` が宣言する。
  | (ContractBase & {
    readonly kind: "upsampleBilinear2d";
    readonly name: typeof UPSAMPLE_BILINEAR2D_OP;
    readonly arity: 1;
  })
  // GRU の隠れ側スキャン（ADR 0056）。走査方向だけが違う 2 op を 1 kind で持つ（reduce 族や
  // gelu / gelu_tanh と同じ扱い — 契約面は完全に同一で、消費側は `name` で方向を引く）。
  | (ContractBase & {
    readonly kind: "gruScan";
    readonly name: GruScanOpName;
    readonly arity: 4;
  });

/** f32 専業の op（実測グラフに i32 / bool 形が現れていない — 対称性のためには解禁しない）。 */
const F32: readonly IrDtype[] = ["f32"];
/**
 * f32 と i32 の両方を実行できる op。根拠は DeBERTa front の実測（recon §3-8）:
 * mask 外積 `mul(i64, i64)` と `1 - attention_mask` の `sub`。いずれも入力値依存で
 * 定数畳み込みできないため実行系に必要。
 */
const F32_I32: readonly IrDtype[] = ["f32", "i32"];
/** bool 専業（否定・論理積は真偽値にしか意味が無い）。 */
const BOOL: readonly IrDtype[] = ["bool"];
/**
 * f32 と bool の両方を実行できる op。根拠は sdp の spline（recon §2 の searchsorted 行）:
 * `sum(inputs[…,None] >= bl, dim=-1)` が **bool の行 sum**（= 真の個数）で、出力は i32
 * （{@link OUTPUT_DTYPES}）。f32 の総和と併存する。
 */
const F32_BOOL: readonly IrDtype[] = ["f32", "bool"];
/** 整数添字専業（gather の index スロット — 実測は i32[16,T,T]）。 */
const I32: readonly IrDtype[] = ["i32"];
/** cast の**入力**側（出力は attrs.to）と、要素値に触れない reshape。 */
const ANY_DTYPE: readonly IrDtype[] = SEMANTIC_DTYPES;

const NO_ATTRS: AttrSchema = {};

/** op ごとの dtype 解禁表。ここに無い op は f32 のまま。 */
const DTYPES: ReadonlyMap<string, readonly IrDtype[]> = new Map([
  ["mul", F32_I32],
  ["sub", F32_I32],
  ["bitwise_not", BOOL],
  // spline の `inside = (x >= -b) & (x <= b)`（recon §2）— bool の論理積。整数の bitwise_and は
  // 意味が違うので解禁しない（bitwise_not と同じ絞り方）。
  ["bitwise_and", BOOL],
  ["sum", F32_BOOL],
  // reshape は要素を 1 つも読み書きしない（別名化のみ）ので全語彙。実測も f32 の view と
  // i32 mask の squeeze/unsqueeze、bool マスクの unsqueeze が全て出る（recon §2）。
  [RESHAPE_OP, ANY_DTYPE],
  // gather 添字（i32）・conv 経路の bool マスクに加え、相対位置埋め込みの 4D 化
  // `ek.expand(...)` / `ev.expand(...)`（f32 — recon §2 の enc_p / flow 行）で f32 も解禁。
  // strided コピー族は dtype パラメトリックなのでカーネルは共用のまま。
  [EXPAND_OP, ANY_DTYPE],
  // 焼いた定数は相対位置バケット表（i32）と位置テーブル（f32）の 2 系統。bool の
  // initializer は語彙に無い（ir.ts の INITIALIZER_STORAGE）ので解禁しない。
  [SYM_PREFIX_SLICE_OP, F32_I32],
]);

const dtypesOf = (name: string): readonly IrDtype[] => DTYPES.get(name) ?? F32;

/**
 * 出力 slot 別の dtype 写像の**列**を宣言する表。ここに無い op は「出力 1 本・恒等（入力と
 * 同型）」（{@link SINGLE_IDENTITY_OUTPUT}）。列の要素は「スロット 0 の入力 dtype → その
 * 出力の dtype」で、恒等な dtype は書かなくてよい（{@link slotContract} が埋める）。
 *
 * MUST: 写像の値域は実測に出た形だけ（ADR 0009 の「op ごと実測ベース」）。
 * - 比較 4 本 — spline の `inside` 判定と searchsorted（recon §2）。真偽値なので bool。
 * - `sum` の bool 入力 — `sum(x >= bl, dim=-1)` は**真の個数**なので i32（f32 で数えると
 *   2^24 を超えた時点で静かに丸まる）。f32 入力は従来どおり f32。
 * - `where` — 条件が先頭スロットなので、写像だけが「出力は値の側」を表せる。
 * - `argmax` — 出力は**添字**なので i32（ADR 0068 決定 2）。f32 で返すと語彙 2^24 を超えた
 *   時点で隣の token に丸まる（bool の `sum` と同型の理由）。
 */
const OUTPUT_DTYPES: ReadonlyMap<string, readonly ReadonlyMap<IrDtype, IrDtype>[]> = new Map([
  ["ge", [new Map<IrDtype, IrDtype>([["f32", "bool"]])]],
  ["ge_scalar", [new Map<IrDtype, IrDtype>([["f32", "bool"]])]],
  ["le_scalar", [new Map<IrDtype, IrDtype>([["f32", "bool"]])]],
  ["gt_scalar", [new Map<IrDtype, IrDtype>([["f32", "bool"]])]],
  ["sum", [new Map<IrDtype, IrDtype>([["f32", "f32"], ["bool", "i32"]])]],
  [WHERE_OP, [new Map<IrDtype, IrDtype>([["bool", "f32"]])]],
  [ARGMAX_OP, [new Map<IrDtype, IrDtype>([["f32", "i32"]])]],
]);

/** 宣言が無い op の既定 = **出力 1 本・恒等**（空の写像は定義域全体が恒等で埋まる）。 */
const SINGLE_IDENTITY_OUTPUT: readonly ReadonlyMap<IrDtype, IrDtype>[] = [
  new Map<IrDtype, IrDtype>(),
];

/**
 * スカラ attr を params の並び順で取り出す（検査は {@link assertNodeContract} が済ませている
 * 前提だが、CPU 参照から直接呼ばれる経路もあるので値域はここでも見る）。
 *
 * MUST: clamp の `min <= max` はここでしか見られない（attrs スキーマはキー単位の検査なので、
 * 2 つのキーに跨る不変条件を表せない）。逆転を許すと WGSL 側の分岐順に依存した「片方だけ
 * 効く clamp」になり、CPU 参照との差が入力の位置によって出たり出なかったりする。
 */
export const scalarParamValues = (
  found: OpContract,
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): readonly number[] => {
  const keys = SCALAR_PARAM_ATTRS.get(found.name);
  if (keys === undefined) return [];
  const values = keys.map((key) =>
    assertFiniteAttr(
      attrValue(attrs, key),
      `${where} の attrs.${key}`,
      `op '${found.name}' の ${key}`,
    )
  );
  if (found.name === "clamp" && values[0] > values[1]) {
    throw new OpContractError(`${where}: clamp の min ${values[0]} が max ${values[1]} を超える`);
  }
  return values;
};

/**
 * スロット別の受理集合を 1 本の和へ畳む。順序は語彙の宣言順（{@link SEMANTIC_DTYPES}）で
 * 固定する — 表示と突合に使う射影なので、スロットの並びで順序が変わってはいけない。
 */
const unionDtypes = (slotDtypes: SlotDtypes): readonly IrDtype[] =>
  slotDtypes.kind === "uniform"
    ? slotDtypes.accept
    : SEMANTIC_DTYPES.filter((dtype) => slotDtypes.slots.some((accept) => accept.includes(dtype)));

const contract = (name: string, attrs: AttrSchema = NO_ATTRS): ContractBase =>
  slotContract(name, { kind: "uniform", accept: dtypesOf(name) }, attrs);

/** スロット 0 の受理集合（uniform は全スロット共通）。出力 dtype 写像の定義域になる。 */
const firstSlotAccept = (slotDtypes: SlotDtypes): readonly IrDtype[] =>
  slotDtypes.kind === "uniform" ? slotDtypes.accept : slotDtypes.slots[0];

const slotContract = (
  name: string,
  slotDtypes: SlotDtypes,
  attrs: AttrSchema = NO_ATTRS,
): ContractBase => {
  const domain = firstSlotAccept(slotDtypes);
  // 宣言が無い dtype は恒等写像で埋める（定義域 = スロット 0 の受理集合という不変条件を
  // 出力 slot ごとに構造で満たす — 表の書き忘れが「出力が決まらない dtype」にならない）。
  const outputDtypes = (OUTPUT_DTYPES.get(name) ?? SINGLE_IDENTITY_OUTPUT).map((slot) =>
    new Map<IrDtype, IrDtype>(domain.map((dtype) => [dtype, slot.get(dtype) ?? dtype]))
  );
  return { dtypes: unionDtypes(slotDtypes), slotDtypes, outputDtypes, attrs };
};

/** attrs を持つ単項 op のスキーマ（無い op は attrs 空）。 */
const UNARY_ATTRS: ReadonlyMap<string, AttrSchema> = new Map([
  ["clamp", CLAMP_ATTRS],
  ["clamp_min", CLAMP_MIN_ATTRS],
  ["leaky_relu", LEAKY_RELU_ATTRS],
  ["ge_scalar", SCALAR_COMPARE_ATTRS],
  ["le_scalar", SCALAR_COMPARE_ATTRS],
  ["gt_scalar", SCALAR_COMPARE_ATTRS],
]);

export const OP_CONTRACTS: ReadonlyMap<string, OpContract> = new Map<string, OpContract>([
  ...UNARY_OPS.map((name): [string, OpContract] => [
    name,
    { ...contract(name, UNARY_ATTRS.get(name) ?? NO_ATTRS), kind: "unary", name, arity: 1 },
  ]),
  ...BINARY_OPS.map((name): [string, OpContract] => [
    name,
    { ...contract(name), kind: "binary", name, arity: 2 },
  ]),
  ...REDUCE_OPS.map((name): [string, OpContract] => [
    name,
    { ...contract(name, REDUCE_ATTRS), kind: "rowReduce", name, arity: 1 },
  ]),
  // 条件 bool と値 f32 のスロット別契約。出力は**値の側**（写像 bool → f32 — 出力元
  // スロットを指す欄は作らない規律）。
  [WHERE_OP, {
    ...slotContract(WHERE_OP, { kind: "perSlot", slots: [BOOL, F32, F32] }),
    kind: "where",
    name: WHERE_OP,
    arity: 3,
  }],
  [CUMSUM_OP, {
    ...contract(CUMSUM_OP, CUMSUM_ATTRS),
    kind: "cumsum",
    name: CUMSUM_OP,
    arity: 1,
  }],
  [MATMUL_OP, { ...contract(MATMUL_OP), kind: "matmul", name: MATMUL_OP, arity: 2 }],
  [BMM_OP, { ...contract(BMM_OP), kind: "bmm", name: BMM_OP, arity: 2 }],
  // 最初のスロット別 dtype 契約: 値 f32 と添字 i32 が混在し、出力は値の側と同型。
  [GATHER_OP, {
    ...slotContract(GATHER_OP, { kind: "perSlot", slots: [F32, I32] }),
    kind: "gather",
    name: GATHER_OP,
    arity: 2,
  }],
  // 最終次元の argmax（ADR 0068 決定 2）。attrs 空・入力 f32 → 出力 i32（写像は
  // OUTPUT_DTYPES）で、rank 保存は shape 規則が持つ。
  [ARGMAX_OP, { ...contract(ARGMAX_OP), kind: "argmax", name: ARGMAX_OP, arity: 1 }],
  [CAST_OP, {
    ...slotContract(CAST_OP, { kind: "uniform", accept: ANY_DTYPE }, CAST_ATTRS),
    kind: "cast",
    name: CAST_OP,
    arity: 1,
  }],
  [RESHAPE_OP, { ...contract(RESHAPE_OP), kind: "reshape", name: RESHAPE_OP, arity: 1 }],
  [PERMUTE_OP, {
    ...contract(PERMUTE_OP, PERMUTE_ATTRS),
    kind: "permute",
    name: PERMUTE_OP,
    arity: 1,
  }],
  [EXPAND_OP, { ...contract(EXPAND_OP), kind: "expand", name: EXPAND_OP, arity: 1 }],
  // レイアウト第 2 群（ADR 0014）。実測は全て f32（enc_p の m_p/logs_p 分割・coupling の
  // 96/96 分割と cat・相対位置 value 側の pad・flow/sdp の Flip — recon §2）なので、
  // dtype は解禁表に載せない = f32 専業。
  [SLICE_OP, { ...contract(SLICE_OP, SLICE_ATTRS), kind: "slice", name: SLICE_OP, arity: 1 }],
  [CAT_OP, {
    ...contract(CAT_OP, AXIS_ATTRS),
    kind: "cat",
    name: CAT_OP,
    arity: 2,
    variadic: true,
  }],
  [PAD_OP, { ...contract(PAD_OP, PAD_ATTRS), kind: "pad", name: PAD_OP, arity: 1 }],
  [FLIP_OP, { ...contract(FLIP_OP, AXIS_ATTRS), kind: "flip", name: FLIP_OP, arity: 1 }],
  [SYM_PREFIX_SLICE_OP, {
    ...contract(SYM_PREFIX_SLICE_OP, SYM_PREFIX_SLICE_ATTRS),
    kind: "symPrefixSlice",
    name: SYM_PREFIX_SLICE_OP,
    arity: 1,
  }],
  // 融合 op（ADR 0012）。bias / affine を持つ 3 本はアリティ 3 固定 — 実測が bias 常時ありで、
  // 「bias 無し」を表す欄を作らないことがそのまま fail loudly になる。
  [LINEAR_OP, { ...contract(LINEAR_OP), kind: "linear", name: LINEAR_OP, arity: 3 }],
  [LAYER_NORM_OP, {
    ...contract(LAYER_NORM_OP, LAYER_NORM_ATTRS),
    kind: "layerNorm",
    name: LAYER_NORM_OP,
    arity: 3,
  }],
  // bias が無いのでアリティ 2（ADR 0017）。weight 無しの形はエクスポータが ones 合成で
  // アリティ 2 へ正規化する — ゼロ bias 合成（ADR 0015）と同じ手筋。
  [RMS_NORM_OP, {
    ...contract(RMS_NORM_OP, RMS_NORM_ATTRS),
    kind: "rmsNorm",
    name: RMS_NORM_OP,
    arity: 2,
  }],
  [SOFTMAX_OP, {
    ...contract(SOFTMAX_OP, SOFTMAX_ATTRS),
    kind: "softmax",
    name: SOFTMAX_OP,
    arity: 1,
  }],
  // ADR 0044。attrs スキーマは softmax と**同じ 1 本を共有**する（複製すると片方だけ
  // 絞りが緩む形が作れる）。
  [SAFE_SOFTMAX_OP, {
    ...contract(SAFE_SOFTMAX_OP, SOFTMAX_ATTRS),
    kind: "safeSoftmax",
    name: SAFE_SOFTMAX_OP,
    arity: 1,
  }],
  // 融合 attention（ADR 0023）。q / k / v と省略可能な mask の 4 本とも f32 で同型
  // （uniform 契約）。mask は加算型なので値の側と同じ dtype で、bool は受理しない。
  [ATTENTION_OP, {
    ...contract(ATTENTION_OP, ATTENTION_ATTRS),
    kind: "attention",
    name: ATTENTION_OP,
    arity: 3,
    maxArity: 4,
  }],
  // 値 f32 と添字 i32 のスロット別契約（gather と同型 — 出力は値の側と同型）。
  [EMBEDDING_OP, {
    ...slotContract(EMBEDDING_OP, { kind: "perSlot", slots: [F32, I32] }, EMBEDDING_ATTRS),
    kind: "embedding",
    name: EMBEDDING_OP,
    arity: 2,
  }],
  // 値 f32 と条件 bool のスロット別契約。出力はスロット 0（値の側）と同型。
  [MASKED_FILL_OP, {
    ...slotContract(MASKED_FILL_OP, { kind: "perSlot", slots: [F32, BOOL] }, MASKED_FILL_ATTRS),
    kind: "maskedFill",
    name: MASKED_FILL_OP,
    arity: 2,
  }],
  [CONV1D_OP, {
    ...contract(CONV1D_OP, CONV1D_ATTRS),
    kind: "conv1d",
    name: CONV1D_OP,
    arity: 3,
  }],
  [CONV2D_OP, {
    ...contract(CONV2D_OP, CONV2D_ATTRS),
    kind: "conv2d",
    name: CONV2D_OP,
    arity: 3,
  }],
  // bias 無し conv はエクスポータのゼロ bias 合成でアリティ 3 に正規化される（ADR 0015）—
  // カーネルにも契約にも arity 分岐を持ち込まない。
  [CONV_TRANSPOSE1D_OP, {
    ...contract(CONV_TRANSPOSE1D_OP, CONV_TRANSPOSE1D_ATTRS),
    kind: "convTranspose1d",
    name: CONV_TRANSPOSE1D_OP,
    arity: 3,
  }],
  // DCNv2（第 1' 層・ADR 0055）。5 スロットとも f32 で同型（offset / mask も値の側と
  // 同じ dtype）。重みは持つが**低精度格納の適格外**なので WEIGHT_SLOTS には載せない。
  [DEFORM_CONV2D_OP, {
    ...contract(DEFORM_CONV2D_OP, DEFORM_CONV2D_ATTRS),
    kind: "deformConv2d",
    name: DEFORM_CONV2D_OP,
    arity: 5,
  }],
  // 双線形 resample（第 1 層）。重みを持たないので WEIGHT_SLOTS には載らない。
  [UPSAMPLE_BILINEAR2D_OP, {
    ...contract(UPSAMPLE_BILINEAR2D_OP, UPSAMPLE_BILINEAR2D_ATTRS),
    kind: "upsampleBilinear2d",
    name: UPSAMPLE_BILINEAR2D_OP,
    arity: 1,
  }],
  // GRU 隠れ側スキャン 2 方向（第 2 層・ADR 0056）。4 スロットとも f32 で同型・attrs 空。
  // `w_hh` は op 内スロットなので**低精度格納の適格外**（WEIGHT_SLOTS に載せない）— 入力側の
  // 重みは呼び手の `linear` が持つので、そちらは従来どおり f16 / i8 が効く。
  ...GRU_SCAN_OPS.map((name): [string, OpContract] => [
    name,
    { ...contract(name), kind: "gruScan", name, arity: 4 },
  ]),
]);

/** 契約の attrs スキーマが宣言するキー（capability 射影と診断で使う）。 */
export const attrKeysOf = (found: OpContract): readonly string[] => Object.keys(found.attrs);

/**
 * 契約が宣言する**出力の本数**。出力 dtype 写像の列長そのもの（ADR 0068 決定 1）。
 *
 * MUST: 本数を独立した欄で持たない — 派生できる事実を二重に持つと、dtype 列だけ増やした
 * 契約が「宣言 1 本・写像 2 本」で通り、2 本目の出力の dtype が誰にも照合されなくなる。
 */
export const outputCountOf = (found: OpContract): number => found.outputDtypes.length;

/**
 * 入力スロット別の受理集合（capability 射影）。uniform 契約は**受理しうるスロット数ぶん
 * 複製**する（省略可能な末尾スロットを含む — {@link ContractBase.maxArity}）。
 * 消費側（列挙門）がスロット番号で引けるようにするため。
 */
export const slotDtypesOf = (found: OpContract): readonly (readonly IrDtype[])[] => {
  const slots = found.slotDtypes;
  return slots.kind === "uniform"
    ? Array.from({ length: found.maxArity ?? found.arity }, () => slots.accept)
    : slots.slots;
};

/**
 * グラフ入力として転送できる意味論 dtype。要素は全型 4 バイトで、bool は u32 の 0/1
 * として運ぶ（ADR 0009）。op の dtype 集合とは別軸 — 実行器はどのノードも消費しない入力も
 * 含めて転送するため。
 */
export const IO_DTYPES: readonly IrDtype[] = SEMANTIC_DTYPES;

/**
 * `assertRuntimeSupport` に渡す対応表。契約表からのみ導出する MUST — 手書きの集合は
 * 表と乖離し、「宣言では対応、実行で落ちる」を作る。op 名だけでなく意味論 dtype と attrs
 * まで運ぶ（名前だけの突合が見逃す差は recon §3-9 の実バグ）。
 */
export const RUNTIME_SUPPORT: RuntimeSupport = {
  ops: new Map<string, OpSupport>(
    [...OP_CONTRACTS].map(([name, found]) => [name, {
      dtypes: new Set(found.dtypes),
      slotDtypes: slotDtypesOf(found).map((accept) => new Set(accept)),
      // cast の出力は attrs.to で決まる（写像は恒等で埋まっているが値域が狭すぎる）ので、
      // 列挙門には語彙全体を渡す。それ以外は出力 slot ごとの写像の値域そのもの。
      outDtypes: found.kind === "cast"
        ? [new Set(SEMANTIC_DTYPES)]
        : found.outputDtypes.map((slot) => new Set(slot.values())),
      attrKeys: new Set(attrKeysOf(found)),
    }]),
  ),
  // 生の int32 格納（ADR 0010）は記号依存定数の焼き込み先として実行対象。f16（ADR 0018）と
  // i8（ADR 0019）は実行経路が入った（適格な重みスロットは圧縮のまま GPU 常駐・適格外は
  // ロード時に CPU で f32 展開）ので、**どの initializer に付いていても実行できる**。
  // bf16 だけが宣言としては valid で実行できない（capability 不足として列挙で落ちる）。
  storage: new Set(["f32", "f16", "i8", "i32"]),
  io: new Set(IO_DTYPES),
};

/** ランタイムが実行できるものの照会用（公開 API 向けに安定した形へ整えた射影）。 */
export type RuntimeCapabilities = {
  readonly ops: readonly string[];
  readonly storage: readonly string[];
};

export const capabilities = (): RuntimeCapabilities => ({
  ops: [...RUNTIME_SUPPORT.ops.keys()].sort(),
  storage: [...RUNTIME_SUPPORT.storage].sort(),
});

/**
 * 入力の本数が契約に合うか（可変アリティ op では `arity` を**下限**として読む）。
 *
 * MUST: 判定を呼び出し側に散らさない。「固定なら ===、可変なら >=」を 3 箇所（契約検査 /
 * shape 計算 / CPU 参照）で書くと、cat を足したときのように 1 箇所だけ古い判定が残る。
 * 例外の**型**は層ごとに違う（契約層は OpContractError / CPU 参照は ReferenceOpError）ので、
 * 共有するのは述語と表示だけにする。
 */
export const arityFits = (found: OpContract, count: number): boolean =>
  found.variadic === true
    ? count >= found.arity
    : count >= found.arity && count <= (found.maxArity ?? found.arity);

/** 契約のアリティの表示形（可変なら「N 本以上」・省略可能な末尾があれば「N か M」）。 */
export const describeArity = (found: OpContract): string => {
  if (found.variadic === true) return `${found.arity} 本以上`;
  return found.maxArity === undefined ? `${found.arity}` : `${found.arity} か ${found.maxArity}`;
};

export const assertArity = (
  found: OpContract,
  count: number,
  what: string,
  where: string,
): void => {
  if (arityFits(found, count)) return;
  throw new OpContractError(
    `${where}: op '${found.name}' の${what}が ${count}（契約は ${describeArity(found)}）`,
  );
};

export const resolveOpContract = (op: string): OpContract => {
  const found = OP_CONTRACTS.get(op);
  if (found === undefined) {
    throw new OpContractError(`op '${op}' は契約表に無い`);
  }
  return found;
};

/**
 * ノードが契約に適合することを検査して契約を返す（shape は含まない — 束縛前に決まる規則だけ）。
 *
 * MUST: attrs の照合は `Object.hasOwn` のみで行う。スキーマは素のオブジェクトリテラルなので、
 * IR 由来のキー（`toString` 等）が prototype 経由で「既知キー」に化ける。
 */
export const assertNodeContract = (node: IrNode, where: string): OpContract => {
  const found = resolveOpContract(node.op);
  assertArity(found, node.ins.length, "入力数", where);
  if (node.outs.length !== outputCountOf(found)) {
    throw new OpContractError(
      `${where}: op '${node.op}' の出力数が ${node.outs.length}（契約は ${outputCountOf(found)}）`,
    );
  }
  const unknown = Object.keys(node.attrs).filter((key) => !Object.hasOwn(found.attrs, key));
  if (unknown.length > 0) {
    throw new OpContractError(
      `${where}: op '${node.op}' の契約外 attrs [${unknown.sort().join(", ")}]`,
    );
  }
  for (const key of attrKeysOf(found)) {
    if (!Object.hasOwn(node.attrs, key)) {
      throw new OpContractError(`${where}: op '${node.op}' の必須 attr '${key}' が無い`);
    }
    found.attrs[key](node.attrs[key], `${where} の attrs.${key}`);
  }
  return found;
};

/**
 * スロット 0 の入力 dtype から**出力 slot `slot` の** dtype を導く
 * （{@link ContractBase.outputDtypes}）。
 *
 * 写像の定義域はスロット 0 の受理集合と一致する（{@link slotContract} が恒等で埋める）ので、
 * スロット検査を通った dtype は必ず引ける — 引けないのはランタイム内部の不変条件破れ。
 */
export const outputDtypeOf = (
  found: OpContract,
  slot: number,
  inputDtype: IrDtype,
  where: string,
): IrDtype => {
  const column = found.outputDtypes[slot];
  if (column === undefined) {
    throw new OpContractError(`${where}: op '${found.name}' に出力スロット ${slot} は無い`);
  }
  const mapped = column.get(inputDtype);
  if (mapped === undefined) {
    throw new OpContractError(
      `${where}: op '${found.name}' の出力 dtype 写像に入力 '${inputDtype}' が無い`,
    );
  }
  return mapped;
};

/**
 * 受理集合の**和**との突合。スロットを跨いだ緩い検査なので、スロット別契約の op では
 * {@link assertSlotDtype} を使う（この関数は「その op が触れる dtype か」しか見ない）。
 */
export const assertDtype = (found: OpContract, dtype: IrDtype, where: string): void => {
  if (!found.dtypes.includes(dtype)) {
    throw new OpContractError(
      `${where}: op '${found.name}' は意味論 dtype '${dtype}' を実行できない（対応: ${
        found.dtypes.join(", ")
      }）`,
    );
  }
};

/**
 * 入力スロット単位の dtype 検査。uniform 契約では和との突合と同義（診断文も同じ）で、
 * perSlot 契約でだけスロット番号まで見る。
 */
export const assertSlotDtype = (
  found: OpContract,
  slot: number,
  dtype: IrDtype,
  where: string,
): void => {
  if (found.slotDtypes.kind === "uniform") {
    assertDtype(found, dtype, where);
    return;
  }
  const accept = found.slotDtypes.slots[slot];
  if (accept === undefined) {
    throw new OpContractError(`${where}: op '${found.name}' に入力スロット ${slot} は無い`);
  }
  if (!accept.includes(dtype)) {
    throw new OpContractError(
      `${where}: op '${found.name}' の入力スロット ${slot} は意味論 dtype '${dtype}' を実行できない（対応: ${
        accept.join(", ")
      }）`,
    );
  }
};

/**
 * ノードの意味論 dtype を検査して**出力 dtype の列を返す**（出力 slot 順）。
 *
 * MUST: 出力 dtype は宣言を鵜呑みにせず契約から導き、宣言と突き合わせる。宣言をそのまま
 * 信じると、`cast` の attrs.to と values{} の宣言が食い違ったグラフが「宣言どおり」に
 * 通ってしまい、readback で別の TypedArray として読まれる沈黙誤値になる。
 * 出力の導出は 2 通りだけ: cast は attrs.to、それ以外は**スロット 0 の dtype を出力 slot
 * ごとの写像に通す**（{@link ContractBase.outputDtypes} — 既定は恒等）。uniform 契約では
 * 加えてスロット間の同型も要求する（混合型の演算は語彙に無い）。
 */
export const resolveNodeDtypes = (
  found: OpContract,
  node: IrNode,
  inputDtypes: readonly IrDtype[],
  declaredOutputs: readonly IrDtype[],
  where: string,
): readonly IrDtype[] => {
  inputDtypes.forEach((dtype, index) => {
    assertSlotDtype(found, index, dtype, `${where} の入力 '${node.ins[index]}'`);
  });
  let expected: readonly IrDtype[];
  if (found.kind === "cast") {
    expected = [castTargetDtype(node.attrs, where)];
  } else {
    // スロットごとに受理集合が違う op は、スロット間の同型を要求しない（それが perSlot の
    // 意味）。uniform 契約だけが混在を拒否する。
    if (found.slotDtypes.kind === "uniform") {
      const mixed = inputDtypes.findIndex((dtype) => dtype !== inputDtypes[0]);
      if (mixed >= 0) {
        throw new OpContractError(
          `${where}: op '${found.name}' の入力 dtype が混在（${inputDtypes.join(", ")}）`,
        );
      }
    }
    expected = found.outputDtypes.map((_, slot) =>
      outputDtypeOf(found, slot, inputDtypes[0], where)
    );
  }
  if (declaredOutputs.length !== expected.length) {
    throw new OpContractError(
      `${where}: op '${found.name}' の出力の宣言が ${declaredOutputs.length} 本（契約は ${expected.length} 本）`,
    );
  }
  expected.forEach((dtype, slot) => {
    if (declaredOutputs[slot] !== dtype) {
      throw new OpContractError(
        `${where}: 出力 '${node.outs[slot]}' の宣言 dtype '${
          declaredOutputs[slot]
        }' が契約の '${dtype}' と違う`,
      );
    }
  });
  return expected;
};
