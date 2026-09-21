/**
 * 融合ルールの共通面: 実行ステップの型・**全ルール共通の適格条件**・`defineRule`。
 *
 * ルールの実体は `fusion-rules/` 配下（1 ルール = 1 ファイル）、宣言表と走査は
 * {@link "./fusion.ts"}。MUST: 実体の依存は fusion-rule → fusion-rules → fusion の
 * **一方向**に保つ（ここから `fusion.ts` を import しない — 宣言表がルールを import するので
 * 循環になる）。
 */

import { outputCountOf } from "../ops.ts";
import type { LinearGemvReduce } from "./session-types.ts";
import { ExecutionError, type NodePlan } from "./plan.ts";

/** 融合ルールの識別子（{@link "./fusion.ts"} の `FUSION_RULES` の宣言順と 1 対 1）。 */
type FusionRuleName =
  | "linearStaticQuantize"
  | "rmsNormAdd"
  | "silu"
  | "upsample2x"
  | "rope"
  | "adaln"
  | "rowBlockAttention";

/**
 * 診断カウンタの見出し。融合ルールに加えて、0 dispatch の別名化のうち**条件付きで外れうる**
 * 恒等 expand（{@link ExecStep} の `aliasesInput`）と、packed int8 活性で受け渡す
 * 固定 SRQ の本数（ADR 0105）を数える。reshape の別名化は無条件なので数えない
 * （外れようがない = 観測する意味がない）。
 */
export type FusionCounterName = FusionRuleName | "identityExpand" | "packedStaticQuantize";

/** ルール別の適用回数。「融合が黙って外れて性能だけ落ちる」事故の唯一の観測点。 */
export type FusionCounts = Readonly<Record<FusionCounterName, number>>;

/**
 * 融合ステップ内で束縛しうる実体。
 *
 * MUST: ステップ内一時（{@link FusedStep.temps}）を指せるのはここだけで、外部入力は
 * {@link FusedStep.binds} の**添字**で指す（値名の再解決を executor に持たせない）。
 */
export type FusedOperand =
  | { readonly kind: "bind"; readonly index: number }
  | { readonly kind: "temp"; readonly id: number }
  | { readonly kind: "weightScale"; readonly index: number }
  | { readonly kind: "output" };

/**
 * dispatch の workgroup 数の決め方。
 *
 * - `gridStride` = 要素 / 行の被覆数を割る形（elementwise・reduce・1 dispatchの融合）。上限を
 *   超えたら縮退し、カーネル側の grid-stride が残りを回す。
 * - `tiled` = **1 workgroup = 1 出力タイル**の GEMM 族。grid-stride で縮退できないので、
 *   上限超過は宣言側（`tiledWorkgroups`）が fail loudly にする。
 */
type FusedWorkgroups =
  | { readonly kind: "gridStride"; readonly items: number; readonly size: number }
  | { readonly kind: "tiled"; readonly counts: readonly [number, number, number] };

/** 融合カーネル 1 dispatch ぶんの生成入力（全ルール共通の形）。 */
export type FusedDispatch = {
  readonly key: string;
  /**
   * MUST: 同一キーには常にバイト単位で同一の WGSL（PipelineCache の決定性契約）。
   * thunk なのは、キャッシュ済みでも本文の突き合わせが必要な一方、計画時に全ステップぶんを
   * 文字列で持つ必要が無いため。
   */
  readonly wgsl: () => string;
  /** binding 0 の Params。既定は uniform（16 バイト整列の MUST は各カーネル側）。 */
  readonly params: Uint32Array<ArrayBuffer>;
  /**
   * params を **storage** で束ねる（可変長 params を持つ elementwise 族だけ — 素のノードの
   * `#buildElementwise` と同じ）。省略時は uniform。
   */
  readonly paramsStorage?: boolean;
  /**
   * binding 1 以降のオペランド列。**省略できるのは 1 dispatch のルールだけ**で、そのときは
   * 「{@link FusedStep.binds} を宣言順 → 末尾に出力」（1 dispatchの融合に共通の形）になる。
   */
  readonly operands?: readonly FusedOperand[];
  readonly workgroups: FusedWorkgroups;
};

/**
 * ステップ内一時の確保仕様。形も意味も recipe.ts の `TempRecipe` と同じで、**寿命は
 * dispatch 境界の添字**で表す。
 *
 * MUST: 宣言した一時には必ず解放境界がある（`releaseAfter` が確保より前だと宣言の受け口
 * （`validateStepRecipe`）が落とす）。
 */
export type FusedTemp = {
  readonly byteLength: number;
  /** {@link FusedStep.dispatches} のこの添字の**直前**に確保する。 */
  readonly allocBefore: number;
  /** {@link FusedStep.dispatches} のこの添字の**直後**に解放する。 */
  readonly releaseAfter: number;
};

/**
 * 融合ステップ（元の連続ノード列 1 本を private カーネルの dispatch 列へ置換したもの）。
 *
 * dispatch は**複数取りうる**。畳んだ結果が 1 dispatch にならないルール
 * （{@link "./fusion-rules/row-block-attention.ts"} の `ROW_BLOCK_ATTENTION_RULE`）は、
 * ステップ内で閉じた一時（{@link FusedStep.temps}）を
 * 挟んで数本を並べる。MUST: それでも解放簿記の根拠は {@link FusedStep.ins} の延べ列 1 本の
 * ままで、ステップ境界の外へは一時が 1 本も漏れない。
 */
export type FusedStep = {
  readonly kind: "fused";
  readonly rule: FusionRuleName;
  /**
   * 元ノード列が消費した**外部**入力の延べ列（重複込み）。内部値は 1 本も実体化しないので
   * 含めない。MUST: 元の延べ回数と 1 つも違えてはならない（アリーナの参照計数の唯一の根拠）。
   */
  readonly ins: readonly string[];
  /** bind group のオペランド順（binding 1 から。重複無し）。 */
  readonly binds: readonly string[];
  /**
   * 畳んだノードの本数。**走査幅ではない** — 窓内 passthrough を持つルールでは
   * 窓幅（{@link FusionHit.advance}）の方が大きい。
   */
  readonly nodeCount: number;
  readonly outputName: string;
  readonly outputShape: readonly number[];
  /** ステップ内で閉じた一時（{@link FusedStep.dispatches} の添字で寿命を表す）。 */
  readonly temps: readonly FusedTemp[];
  readonly dispatches: readonly FusedDispatch[];
};

/**
 * packed int8 活性の受け渡し（ADR 0105）— 固定 SRQ の出力を u32 1 語 = int8 コード 4 個で
 * 渡す対の宣言。`role` は**このノードがどちら側か**で、`scale` はどちらの側でも
 * **生産側 SRQ の f32 scale**（消費側が `f32(code) * scale` で復元するのに要る）。
 *
 * MUST: 対の受理は {@link "./fusion.ts"} の `planFusions` の 1 箇所で決まる。
 * 生産側だけ / 消費側だけを立てると、
 * `vec4<u32>` 束縛に f32 の語が流れる形（例外なしの沈黙誤値）になる。
 */
export type PackedActivations = {
  readonly role: "write" | "read";
  readonly scale: number;
};

type NodeStep = {
  readonly kind: "node";
  readonly plan: NodePlan;
  /**
   * 出力を入力バッファの別名にする（0 dispatch — ADR 0011）。reshapeは常に真、permuteは
   * 要素順が変わらず、実体が内部で確保されている場合だけ。expand は
   * 束縛後の入出力 shape が rank を含め完全一致するとき（= 複製軸を持たない恒等写像）だけ真。
   */
  readonly aliasesInput: boolean;
  /** packed int8 活性の対（ADR 0105）。`undefined` = 従来どおり f32 で受け渡す。 */
  readonly packedActivations?: PackedActivations;
};

export type ExecStep = NodeStep | FusedStep;

export type FusionPlan = {
  readonly steps: readonly ExecStep[];
  readonly counts: FusionCounts;
};

/**
 * 判定に要る device の能力（**granted limit の値そのもの**）。
 *
 * MUST: ここに載るのは「計画を決める入力」であって計測値ではない。実行時に測って選び直す形
 * （オートチューン）は ADR 0022 で禁じている — 同じ device・同じ束縛なら常に同じ計画が出る、
 * が prepared plan キャッシュとキーの意味の前提。
 */
export type FusionLimits = {
  /** ストレージ束縛 1 本の上限。行ブロック枚数を決める唯一の device 側入力。 */
  readonly maxStorageBufferBindingSize: number;
  /** 1 軸あたりの workgroup 数の上限（タイル型 dispatch の fail loudly 用）。 */
  readonly maxComputeWorkgroupsPerDimension: number;
};

/** GPU実体を持たない常駐格納情報。scaleの寿命と所有権はSession側に残す。 */
export type FusionWeightLayout =
  | { readonly storage: "f16" | "i2" | "i8" }
  | { readonly storage: "i4"; readonly groupSize: number };

/** 判定に要るグラフ全体の事実（executor の Session 状態から渡す）。 */
export type FusionContext = {
  /** 値名 → グラフ内の消費回数（plan.ts の countUses）。 */
  readonly useCounts: ReadonlyMap<string, number>;
  readonly outputNames: ReadonlySet<string>;
  readonly limits: FusionLimits;
  /**
   * 行ブロック枚数の強制（**テスト専用** — executor の `ROW_BLOCK_SPLIT`）。上限に収まる
   * 最小枚数の代わりにこの枚数で割る。上限に収まらない枚数は fail loudly。
   */
  readonly rowBlockSplit?: number;
  /** 明示指定時だけRMS→addを融合する（ADR 0099）。 */
  readonly fuseRmsNormAdd?: boolean;
  readonly fuseLinearStaticQuantize?: boolean;
  /** 固定 SRQ の活性を packed int8 で並列 GEMV へ渡す（ADR 0105）。 */
  readonly packedStaticQuantize?: boolean;
  readonly linearGemvReduce?: LinearGemvReduce;
  readonly linearCompute?: "f32" | "f16" | "a8";
  readonly weightLayouts?: ReadonlyMap<string, FusionWeightLayout>;
  readonly rmsNormReduce?: "workgroup" | "subgroup32";
};

/**
 * ルールが見る文脈 = 呼び手の {@link FusionContext} + **走査の内側で決まる** packed 活性の対
 * （値名 → 生産側 SRQ の scale）。
 *
 * MUST: `packedValues` は呼び手が宣言するものではない（{@link "./fusion.ts"} の `planFusions` が
 * 1 度目の走査の
 * 結果から導く — 素のノードとして残った SRQ だけが対象なので、走査の前には決まらない）。
 */
export type FusionScanContext = FusionContext & {
  readonly packedValues: ReadonlyMap<string, number>;
};

export const sameShape = (a: readonly number[], b: readonly number[]): boolean =>
  a.length === b.length && a.every((dim, index) => dim === b[index]);

/**
 * 融合が扱えるのは**契約の出力数が 1 本**のノードだけ（ADR 0068 決定 1）。
 *
 * MUST: 判定は窓の全ノードで行い、1 本でも外れたら窓ごと諦める（掴めなければ既存の素の
 * ノード列が必ず正しい）。{@link FusedStep} は単一出力で、内部値の畳み込み
 * （{@link externalIns} / {@link internalsArePrivate}）も末尾出力の引き継ぎ
 * （`build` の `outputName`）も「ノード 1 本 = 値 1 本」から導いている。多出力ノードが鎖に
 * 入ると畳まれなかった出力を誰も定義しないまま消え、窓内 passthrough に入ると素のノードとして
 * 前へ動かす対象になるので、どちらも matcher の綴りだけでは防げない。
 * MUST: ここが唯一の判定点（{@link defineRule} が全ルールを通す）。ルール側の match に
 * 書かせると、将来 topk / argmax の綴りに一致する窓を持つルールだけが黙って誤融合する。
 */
export const windowIsSingleOutput = (window: readonly NodePlan[]): boolean =>
  window.every((step) => outputCountOf(step.contract) === 1);

/**
 * 窓に **state を触るノード**が 1 本でも入っていないか（ADR 0067 決定 5b）。
 *
 * MUST: state 参照はテンソルのデータ辺を張らないため、融合が窓内 passthrough を前へ動かすと
 * `nodes` 配列順が崩れても shape 検査も参照計数も何も落ちない — 「今 step の k/v を過去として
 * 二重に読む」形が例外なしに出る。
 * MUST: ここが唯一の判定点（{@link windowIsSingleOutput} と同じ位置）。ルール側の match に
 * 書かせると、将来のルールが 1 本書き忘れただけで沈黙誤値になる。
 */
export const windowTouchesState = (window: readonly NodePlan[]): boolean =>
  window.some((step) => Object.keys(step.node.states).length > 0);

/** 鎖の全ノードが f32 専業か（融合カーネルは全て f32 固定）。 */
export const allF32 = (chain: readonly NodePlan[]): boolean =>
  chain.every((step) =>
    step.outputs.every((out) => out.dtype === "f32") &&
    step.inputDtypes.every((dtype) => dtype === "f32")
  );

/**
 * 鎖の内部値（最終ノード以外の出力）が全て「消費者ちょうど 1 本・graph output でない」か。
 *
 * MUST: 融合後は内部値のバッファを 1 本も作らないので、外部 consumer や readback が 1 つでも
 * あれば値が消える。ここが全ルール共通の適格条件。
 * MUST: ノードの**全出力**を見る（{@link windowIsSingleOutput} が多出力を先に落とすかどうかに
 * 依らせない — 判定の順序を変えても結論が動かない形にしておく）。
 */
export const internalsArePrivate = (chain: readonly NodePlan[], context: FusionContext): boolean =>
  chain.slice(0, -1).every((step) =>
    step.outputs.every((out) =>
      (context.useCounts.get(out.name) ?? 0) === 1 && !context.outputNames.has(out.name)
    )
  );

/**
 * 鎖が消費した外部入力の延べ列（重複込み・元の node.ins の並び順）。
 * 内部値は畳まれて消えるので除く（ノードの全出力が内部値）。
 */
const externalIns = (chain: readonly NodePlan[]): readonly string[] => {
  const internal = new Set(
    chain.slice(0, -1).flatMap((step) => step.outputs.map((out) => out.name)),
  );
  return chain.flatMap((step) => step.node.ins.filter((name) => !internal.has(name)));
};

/**
 * 窓内 passthrough を融合ステップより**前**へ動かしてよいか。
 *
 * MUST: passthrough は鎖が定義する値（内部値も**最終出力も**）を 1 つも消費してはならない。
 * 消費していれば「まだ計算されていない値を読むノード」を先に置くことになり、順序の
 * 入れ替えが非合法になる（executor は steps を順に encode するだけなので、ここで弾かないと
 * `値 'x' のバッファが無い` か、名前が使い回されていれば沈黙誤値になる）。
 *
 * NOTE: 窓内 passthrough を持つ現行 2 ルール（adaln / rope）に限れば、passthrough が読める
 * 鎖の値は**内部値だけ**（最終出力は passthrough より後に定義されるので読めない）で、それは
 * {@link internalsArePrivate}（consumer ちょうど 1 本）が先に落とす — **反例を
 * 単独では構成できない**（tests/runtime_fusion_test.ts のフォールト注入で確認済み）。
 * ここが独立に効くのは「鎖の**最終**ノードが passthrough より前に来る」窓を持つ将来の
 * ルールで、internalsArePrivate は最終出力を見ないので代替にならない。窓の仕組み側の
 * 不変条件として `defineRule` に置く。
 */
export const passthroughIsIndependent = (
  chain: readonly NodePlan[],
  passthrough: readonly NodePlan[],
): boolean => {
  if (passthrough.length === 0) return true;
  const defined = new Set(chain.flatMap((step) => step.outputs.map((out) => out.name)));
  return passthrough.every((step) => step.node.ins.every((name) => !defined.has(name)));
};

/** 融合ルール 1 件の適用結果（融合ステップ + 窓内 passthrough + 走査幅）。 */
export type FusionHit = {
  /** 融合ステップより**前**に素のまま並べる窓内ノード（元のノード順）。 */
  readonly passthrough: readonly NodePlan[];
  readonly step: FusedStep;
  /** 走査を進める幅（= 窓のノード数 = passthrough + 畳んだ鎖）。 */
  readonly advance: number;
};

/** ルールの本体（match で掴み、build で宣言的にステップへ落とす）。 */
export type FusionRule = {
  readonly name: FusionRuleName;
  /** このルールが掴みうる先頭 op（適用順の互いに素性を機械検査するための宣言）。 */
  readonly heads: readonly string[];
  readonly apply: (
    nodes: readonly NodePlan[],
    index: number,
    context: FusionScanContext,
  ) => FusionHit | undefined;
};

/**
 * ルールが掴んだ窓。`window` は連続ノード列、`chain` はそのうち**畳む**部分列
 * （実際のノード順）で、差分が窓内 passthrough になる。
 */
export type FusionMatch = {
  /** 連続窓（走査幅の唯一の根拠）。 */
  readonly window: readonly NodePlan[];
  /** 畳むノード列（`window` の部分列）。 */
  readonly chain: readonly NodePlan[];
};

/**
 * ステップ内一時の寿命宣言が dispatch 列の内側で閉じているか。
 *
 * MUST: `allocBefore ≤ releaseAfter < dispatch 数`。外れた宣言は executor の replay で
 * 「未確保の一時を束ねる」か「解放されない一時が残る」になり、前者は bind 面が組めず、
 * 後者は計画の閉包検査まで気づけない。ルールの本数だけ手書きさせず、
 * 宣言の受け口 1 箇所で落とす。
 */
const assertTempLifetimes = (
  name: FusionRuleName,
  temps: readonly FusedTemp[],
  dispatchCount: number,
): void => {
  temps.forEach((temp, id) => {
    if (
      !Number.isSafeInteger(temp.byteLength) || temp.byteLength < 1 ||
      temp.allocBefore < 0 || temp.releaseAfter < temp.allocBefore ||
      temp.releaseAfter >= dispatchCount
    ) {
      throw new ExecutionError(
        `融合ルール '${name}': 一時 ${id} の寿命宣言 [${temp.allocBefore}, ${temp.releaseAfter}] が` +
          ` dispatch ${dispatchCount} 本の内側で閉じていない（${temp.byteLength}B）`,
      );
    }
  });
};

/**
 * match（掴む）と build（binds / kernel key / params を宣言する）の分離を型で強制する。
 *
 * MUST: 解放簿記の根拠（`ins` の延べ列）・畳んだ本数（`nodeCount`）・走査幅（`advance`）・
 * passthrough は**掴んだ窓と鎖から導く**。ルール側に宣言させると、同じ事実がルールの
 * 本数だけ複製され、1 本ずれても例外は出ない。
 * MUST: 多出力ノードを含む窓は全ルールでここが落とす（{@link windowIsSingleOutput}）。
 * MUST: state を触るノードを含む窓も同じ位置で落とす（{@link windowTouchesState}）。
 */
export const defineRule = <Matched extends FusionMatch>(rule: {
  readonly name: FusionRuleName;
  readonly heads: readonly string[];
  readonly match: (
    nodes: readonly NodePlan[],
    index: number,
    context: FusionScanContext,
  ) => Matched | undefined;
  readonly build: (
    matched: Matched,
  ) => Omit<FusedStep, "kind" | "rule" | "ins" | "nodeCount">;
}): FusionRule => ({
  name: rule.name,
  heads: rule.heads,
  apply: (nodes, index, context) => {
    const matched = rule.match(nodes, index, context);
    if (matched === undefined) return undefined;
    // 多出力ノード（ADR 0068 決定 1）は融合候補にしない。matcher の op 名の綴りは受理集合を
    // 狭めるだけで「出力が 1 本」を保証しないので、窓の全ノードを契約から見る 1 箇所を置く。
    if (!windowIsSingleOutput(matched.window)) return undefined;
    // state を触るノード（ADR 0067 決定 5b）は融合候補にしない — 並べ替えの禁止は窓の
    // 仕組み側の不変条件で、matcher の op 名の綴りでは表せない。
    if (windowTouchesState(matched.window)) return undefined;
    const folded = new Set(matched.chain);
    const passthrough = matched.window.filter((step) => !folded.has(step));
    // MUST: 鎖は窓の部分列。外れていれば窓外のノードを畳んだ（= 走査幅が足りず二重実行）か
    // 窓内を取りこぼした（= 未実行）ことになり、どちらも例外なしの沈黙誤値になる。
    if (passthrough.length + matched.chain.length !== matched.window.length) {
      throw new ExecutionError(
        `融合ルール '${rule.name}': 畳んだ鎖 ${matched.chain.length} 本が窓 ${matched.window.length} 本の部分列でない`,
      );
    }
    if (!passthroughIsIndependent(matched.chain, passthrough)) return undefined;
    const built = rule.build(matched);
    assertTempLifetimes(rule.name, built.temps, built.dispatches.length);
    return {
      passthrough,
      advance: matched.window.length,
      step: {
        kind: "fused",
        rule: rule.name,
        ins: externalIns(matched.chain),
        nodeCount: matched.chain.length,
        ...built,
      },
    };
  },
});
