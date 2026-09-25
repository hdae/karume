/**
 * 融合パス（純関数）: 計画済みノード列（plan.ts の {@link NodePlan}）→ 実行ステップ列。
 *
 * IR の公開語彙は 1 つも増やさない。エクスポータが出す**隣接ノードの決まった並び**を、実行時
 * にだけ private カーネル 1 本へ畳む peephole で、掴めなかった形は素のノード列にそのまま
 * 落ちる（fallback は常に正しい既存経路）。GPU に触れないのでアダプタ無し環境で検証できる。
 *
 * ## 置き場（融合パスの入口）
 *
 * **このファイルが融合パスの入口**で、実体は 3 つに分かれている: 共通型・全ルール共通の
 * 適格条件・`defineRule` が {@link "./fusion-rule.ts"}、7 ルールの実体が `fusion-rules/` 配下
 * （1 ルール = 1 ファイル）、そしてここが宣言表 {@link FUSION_RULES}・別名化
 * （{@link planAliases}）・走査（{@link planFusions}）・候補列挙
 * （{@link enumerateUnfusedWindows}）を持つ。
 * MUST: 消費側は**この綴り（`fusion.ts`）で import する** — ADR 0040 決定 1 が融合パスの
 * 置き場を「`src/runtime/fusion.ts` の純関数パス」と名指しており、3 ファイルへの分割は実装の
 * 都合にすぎない（公開型と {@link planRowBlocks} はここが再 export する）。
 * MUST: 実体の依存は **fusion-rule → fusion-rules → fusion の一方向**。ルール側から
 * `fusion.ts` を import しない（宣言表がルールを import するので循環になる）。
 *
 * ## 設計の約束
 *
 * - MUST: matcher を executor の走査ループへ直書きしない。ここが唯一の判定点で、反例
 *   （use-count 2 / graph output / near-shape / dtype 違い / 順序違い）を GPU 無しで網羅できる。
 * - MUST: 融合ステップは外部入力の**延べ列**（{@link FusedStep.ins}）を宣言し、解放簿記は
 *   executor の既存経路（素のノードと同一）に合流させる。融合ごとに手書きの retain/release を
 *   置くと、アリーナの参照計数が融合の本数だけ別実装になり、1 本ずれても例外は出ずに
 *   沈黙誤値になる。
 * - MUST: 融合は演算列を潰すが**値は変えない**。丸め位置の保存はカーネル側の責務で、その
 *   手段（workgroup memory 往復による丸め障壁）が仕様保証でないことは各カーネルの docstring に
 *   書いてある。任意指定のrmsNormAddはADR 0099の検収範囲に限り、未検証GPUでは参照経路を残す。
 * - MUST: 掴むのは**契約の出力数が 1 本**のノードだけ（{@link windowIsSingleOutput} —
 *   ADR 0068 決定 1）。{@link FusedStep} は単一出力のままで、鎖のノードの出力は常に
 *   `outputs[0]`（多出力 op の融合は語彙に入れていない）。
 *
 * ## 畳む先は 1 dispatch とは限らない
 *
 * linearStaticQuantize / rmsNormAdd / silu / upsample2x / rope / adaln は「N ノード → private
 * カーネル 1 dispatch」（{@link LINEAR_STATIC_QUANTIZE_RULE} は並列 GEMV と固定再量子化を
 * 1 dispatch へ畳む）だが、
 * {@link ROW_BLOCK_ATTENTION_RULE} は**演算ではなく中間の実体化幅**を畳むので、ステップ内で
 * 閉じた一時（{@link FusedStep.temps}）を挟んだ dispatch 列になる。どちらも
 * {@link FusedStep} 1 つ = 実行ステップ 1 つで、解放簿記の合流点は変わらない。
 *
 * ## 適用順
 *
 * {@link FUSION_RULES} の**宣言順**（linearStaticQuantize → rmsNormAdd → silu → upsample2x → rope → adaln → rowBlockAttention）。
 * 7 ルールの先頭 op は `linear` / `rms_norm` / `sigmoid` / `reshape` / `mul|slice` / `layer_norm` / `bmm` で互いに素
 * なので、この順序は結果に効かない（順序が意味を持つのは先頭 op が重なったときだけ —
 * 重なりが生じていないことは tests/runtime_fusion_test.ts が {@link FusionRule.heads} から
 * 機械的に検査する）。窓の**内側**に他ルールの先頭 op が現れる形（rowBlockAttention の窓は
 * `reshape` / `expand` を 5 本含む）は、掴めた時点で走査が窓幅ぶん進むので発火しえない。
 * MUST: {@link FUSION_RULES} は**手書きの 1 本**に保つ（ルール側の import 時自動登録へ
 * 置き換えない）。全モジュール副作用ゼロの規約が import 時実行を禁じており、先頭 op の
 * 互いに素性も tests/runtime_fusion_test.ts が**この配列**を読んで機械検査するため。
 *
 * ## 窓内 passthrough
 *
 * 鎖は**隣接しているとは限らない**。adaLN は `layer_norm` と `mul` の間にエクスポータが
 * 変調ベクトルの `reshape` を 2〜3 本挟む（実測 85 鎖すべて）。RoPE は `cat` と続く `mul` の
 * 間に cos / sin 表の `sym_prefix_slice` が挟まる形が**表の初出 1 箇所だけ**にある。
 * ルールは連続窓
 * （{@link "./fusion-rule.ts"} の `FusionMatch.window`）を宣言し、そのうち畳むノード（`chain`）以外を
 * **passthrough** として融合ステップの**前**に素のノードのまま並べる。並べ替えが合法なのは
 * passthrough が鎖の定義する値を 1 つも消費しない場合だけで、そこは
 * {@link passthroughIsIndependent} が機械的に見る。
 */

import { linearGemvPackedEligible } from "../kernels/linear-gemv.ts";
import { numel, permuteDims, STATIC_QUANTIZE_OP, staticQuantizeScale } from "../ops.ts";
import { ADALN_RULE } from "./fusion-rules/adaln.ts";
import { LINEAR_STATIC_QUANTIZE_RULE } from "./fusion-rules/linear-static-quantize.ts";
import { RMS_NORM_ADD_RULE } from "./fusion-rules/rms-norm-add.ts";
import { ROPE_RULE } from "./fusion-rules/rope.ts";
import { ROW_BLOCK_ATTENTION_RULE } from "./fusion-rules/row-block-attention.ts";
import { SILU_RULE } from "./fusion-rules/silu.ts";
import { UPSAMPLE_2X_RULE } from "./fusion-rules/upsample2x.ts";
import {
  allF32,
  type ExecStep,
  type FusionContext,
  type FusionCounterName,
  type FusionHit,
  type FusionPlan,
  type FusionRule,
  type FusionScanContext,
  internalsArePrivate,
  type PackedActivations,
  passthroughIsIndependent,
  sameShape,
  windowIsSingleOutput,
  windowTouchesState,
} from "./fusion-rule.ts";
import { ExecutionError, type NodePlan } from "./plan.ts";

/**
 * 融合パスの公開型（実体は {@link "./fusion-rule.ts"}）。MUST: 消費側は `fusion.ts` の綴りで
 * import する — 入口を 1 つに保つための再 export で、3 ファイルへの分割は実装の都合。
 */
export type {
  ExecStep,
  FusedOperand,
  FusedStep,
  FusionCounts,
  FusionLimits,
  FusionPlan,
  FusionWeightLayout,
  PackedActivations,
} from "./fusion-rule.ts";
/** 行ブロック分割（実体は {@link "./fusion-rules/row-block-attention.ts"}）。 */
export { planRowBlocks } from "./fusion-rules/row-block-attention.ts";

/** MUST: この配列の順が適用順（冒頭「適用順」節）。 */
export const FUSION_RULES: readonly FusionRule[] = [
  LINEAR_STATIC_QUANTIZE_RULE,
  RMS_NORM_ADD_RULE,
  SILU_RULE,
  UPSAMPLE_2X_RULE,
  ROPE_RULE,
  ADALN_RULE,
  ROW_BLOCK_ATTENTION_RULE,
];

/**
 * 長さ1の軸は座標が常に0なので、移動しても平坦な要素順へ影響しない。
 * それ以外の軸は元の順を保つ必要がある。同じ長さの軸同士の交換も許さない。
 * 空テンソルは従来経路に残す。DECIDED: docs/decisions/0011-layout-strategy.md#要素順を保つpermute2026-09-14
 */
const permuteKeepsElementOrder = (plan: NodePlan): boolean => {
  const shape = plan.inputShapes[0];
  if (shape.some((dim) => dim === 0)) return false;
  let previous = -1;
  for (const axis of permuteDims(plan.node.attrs, "permute alias")) {
    if (shape[axis] === 1) continue;
    if (axis <= previous) return false;
    previous = axis;
  }
  return true;
};

/**
 * 別名規則の唯一の判定点。新しく省くpermuteのコピーは内部で確保した実体に限る。
 * 入力や重みにまで別名を伸ばすと、従来のcopyOutputsが自己コピーとして拒否される。
 */
const aliasesInput = (plan: NodePlan, internal: ReadonlySet<string>): boolean =>
  plan.contract.kind === "reshape" ||
  (plan.contract.kind === "permute" && internal.has(plan.node.ins[0]) &&
    permuteKeepsElementOrder(plan)) ||
  (plan.contract.kind === "expand" && sameShape(plan.inputShapes[0], plan.outputs[0].shape));

/**
 * 宣言順に別名と実体の由来を導出する。入力・重みは集合の外、確保した値とその別名だけが内部値。
 * 実行相と見積り相が共有し、形状と所有権の判定を別々に書き写さない（ADR 0011）。
 */
export const planAliases = (nodes: readonly NodePlan[]): ReadonlySet<NodePlan> => {
  const internal = new Set<string>();
  const aliases = new Set<NodePlan>();
  for (const plan of nodes) {
    const alias = aliasesInput(plan, internal);
    if (alias) aliases.add(plan);
    if (!alias || internal.has(plan.node.ins[0])) {
      for (const output of plan.outputs) internal.add(output.name);
    }
  }
  return aliases;
};

const NO_PACKED: ReadonlyMap<string, number> = new Map();

/**
 * 素のノード 1 本に packed 活性の役割（ADR 0105）を付ける。生産側は固定 SRQ、消費側は
 * その値を**活性スロット**に取る linear で、どちらも `packedValues` の 1 つの決定から従う。
 */
const packedRole = (
  plan: NodePlan,
  packed: ReadonlyMap<string, number>,
): { readonly packedActivations: PackedActivations } | Record<never, never> => {
  if (packed.size === 0) return {};
  if (plan.node.op === STATIC_QUANTIZE_OP) {
    const scale = packed.get(plan.outputs[0].name);
    return scale === undefined ? {} : { packedActivations: { role: "write", scale } };
  }
  if (plan.node.op !== "linear") return {};
  const scale = packed.get(plan.node.ins[0]);
  return scale === undefined ? {} : { packedActivations: { role: "read", scale } };
};

/** ノード列を 1 度走査して融合ステップへ畳む。掴めなかったノードは素のまま並ぶ。 */
const scanFusions = (
  nodes: readonly NodePlan[],
  context: FusionContext,
  packed: ReadonlyMap<string, number>,
): FusionPlan => {
  const steps: ExecStep[] = [];
  const aliases = planAliases(nodes);
  const scanContext: FusionScanContext = { ...context, packedValues: packed };
  const counts: Record<FusionCounterName, number> = {
    linearStaticQuantize: 0,
    rmsNormAdd: 0,
    silu: 0,
    upsample2x: 0,
    rope: 0,
    adaln: 0,
    rowBlockAttention: 0,
    identityExpand: 0,
    packedStaticQuantize: packed.size,
  };
  const pushNode = (plan: NodePlan): void => {
    const alias = aliases.has(plan);
    if (alias && plan.contract.kind === "expand") counts.identityExpand += 1;
    steps.push({ kind: "node", plan, aliasesInput: alias, ...packedRole(plan, packed) });
  };
  for (let index = 0; index < nodes.length;) {
    const hit = FUSION_RULES.reduce<FusionHit | undefined>(
      (found, rule) => found ?? rule.apply(nodes, index, scanContext),
      undefined,
    );
    if (hit !== undefined) {
      // MUST: passthrough が先（融合ステップは passthrough の出力を入力に取りうる）。
      for (const plan of hit.passthrough) pushNode(plan);
      steps.push(hit.step);
      counts[hit.step.rule] += 1;
      index += hit.advance;
      continue;
    }
    pushNode(nodes[index]);
    index += 1;
  }
  return { steps, counts };
};

/**
 * 値名 → その値を消費するノード（延べ — 同じノードが 2 度取れば 2 回入る）。
 */
const consumersByValue = (
  nodes: readonly NodePlan[],
): ReadonlyMap<string, readonly NodePlan[]> => {
  const consumers = new Map<string, NodePlan[]>();
  for (const plan of nodes) {
    for (const name of plan.node.ins) {
      const found = consumers.get(name);
      if (found === undefined) consumers.set(name, [plan]);
      else found.push(plan);
    }
  }
  return consumers;
};

/**
 * この消費先が「packed 活性を受け取れる linear」か（ADR 0105 の対付けの消費側条件）。
 *
 * MUST: 活性スロット（`ins[0]`）でだけ取ること。重み / bias に同じ値が来る形は f32 の語を
 * 期待する束縛なので、packed へ切り替えると黙って誤値になる。
 * MUST: 判定は {@link linearGemvPackedEligible} 1 本（recipe-builder の門と同じ述語）。
 * 並列 GEMV へ落ちるだけでは足りず、**実測で packed が効いた行**（ADR 0105 追記 1）に
 * 限る — 効かない形まで packed にすると、生産側 SRQ ごと丸損になる。
 */
const readsPackedActivation = (
  plan: NodePlan,
  name: string,
  context: FusionContext,
): boolean => {
  if (plan.node.op !== "linear" || plan.node.ins.length !== 3) return false;
  if (plan.node.ins[0] !== name || plan.node.ins[1] === name || plan.node.ins[2] === name) {
    return false;
  }
  if (!allF32([plan])) return false;
  const weight = context.weightLayouts?.get(plan.node.ins[1]);
  if (weight === undefined || weight.storage === "f16") return false;
  const m = numel(plan.inputShapes[0].slice(0, -1));
  const [n, k] = plan.inputShapes[1];
  const group = weight.storage === "i4" ? weight.groupSize : undefined;
  return linearGemvPackedEligible(weight.storage, m, n, k, group) !== undefined;
};

/**
 * packed int8 活性で受け渡す固定 SRQ（ADR 0105）— 値名 → その SRQ の f32 scale。
 *
 * 受理は 5 条件:
 *
 * 1. 席（`packedStaticQuantize`）が立っており、`linearGemvReduce: parallel` /
 *    `linearCompute: f32`（packed 変種を持つのは並列 GEMV 族だけ）。
 * 2. **素のノードとして残った** `static_quantize`。linear→SRQ 融合（ADR 0103）に飲まれた
 *    SRQ は出力側エピローグで、packed 出力は本段の範囲外なので対象にしない — だから
 *    この判定は 1 度目の走査結果（`steps`）を入力に取る。
 * 3. 出力が graph output でなく、最終次元 k が 16 の倍数（束縛は `vec4<u32>` = 16 要素単位で
 *    読む）、scale が正・有限（scale 0 は恒等 SRQ で int8 コードに落とせない）。
 * 4. 消費先が 1 本以上あり、**その全てが**packed が効くと実測された形の並列 GEMV linear の
 *    活性スロット（{@link readsPackedActivation}）。1 本でも GEMM / 逐次 GEMV / 他 op /
 *    実測で効かなかった形が混ざれば f32 のまま。
 * 5. その値を消費する融合ステップが {@link LINEAR_STATIC_QUANTIZE_RULE} 以外に無いこと。
 *
 * MUST: 5 は**将来のルール**向けの不変条件。packed で読む綴りを持つのは
 * linearStaticQuantize の融合だけなので、別のルールが linear を窓へ入れた瞬間、その
 * 融合カーネルは packed の語を f32 として読む（例外なしの沈黙誤値）。現行 7 ルールの窓に
 * `linear` は 1 本も無いので今は発火しないが、受理側に置かないと追加したルールだけが
 * 黙って壊れる。
 */
const packedStaticQuantizeValues = (
  nodes: readonly NodePlan[],
  steps: readonly ExecStep[],
  context: FusionContext,
): ReadonlyMap<string, number> => {
  if (
    context.packedStaticQuantize !== true || context.linearGemvReduce !== "parallel" ||
    context.linearCompute !== "f32"
  ) return NO_PACKED;
  const consumers = consumersByValue(nodes);
  const foldedElsewhere = new Set<string>();
  for (const step of steps) {
    if (step.kind !== "fused" || step.rule === "linearStaticQuantize") continue;
    for (const name of step.ins) foldedElsewhere.add(name);
  }
  const packed = new Map<string, number>();
  for (const step of steps) {
    if (step.kind !== "node" || step.plan.node.op !== STATIC_QUANTIZE_OP) continue;
    const [output] = step.plan.outputs;
    if (context.outputNames.has(output.name) || foldedElsewhere.has(output.name)) continue;
    if (!allF32([step.plan])) continue;
    const k = output.shape[output.shape.length - 1];
    if (k === undefined || k <= 0 || k % 16 !== 0) continue;
    const scale = staticQuantizeScale(step.plan.node.attrs, "packed static_quantize");
    if (!Number.isFinite(scale) || scale <= 0) continue;
    const uses = consumers.get(output.name) ?? [];
    if (uses.length === 0) continue;
    if (!uses.every((use) => readsPackedActivation(use, output.name, context))) continue;
    packed.set(output.name, scale);
  }
  return packed;
};

/**
 * ノード列を走査して融合ステップへ畳む。掴めなかったノードは素のまま並ぶ。
 *
 * packed 活性の席（ADR 0105）が立っているときだけ**走査を 2 度**回す。1 度目は対付けの入力
 * （どの SRQ が素のノードとして残るか）を得るためで、2 度目が結果。
 * MUST: packed の有無はどのルールの `match` 条件にも入らない（変わるのは掴んだ
 * linear→SRQ 融合が組む dispatch のキー・WGSL・params だけ）ので、2 度の走査は**同じ窓を
 * 同じルールが掴む** — 対付けの入力が 2 度目で動くことはない。
 */
export const planFusions = (
  nodes: readonly NodePlan[],
  context: FusionContext,
): FusionPlan => {
  const first = scanFusions(nodes, context, NO_PACKED);
  const packed = packedStaticQuantizeValues(nodes, first.steps, context);
  return packed.size === 0 ? first : scanFusions(nodes, context, packed);
};

/**
 * 候補窓 1 本（{@link enumerateUnfusedWindows} の返り値）。
 *
 * NOTE: これは**候補であって受理ではない**。ADR 0040 決定 2 の受理集合（各ルールの綴り・
 * shape 条件・丸め位置の保存）を 1 つも代行していない — 共通の適格条件だけを通した「ここに
 * ルールを書けば畳めるかもしれない」印。
 */
export type UnfusedWindow = {
  /** 畳む対象になる鎖の op 名列（窓内 passthrough を除いた並び）。 */
  readonly ops: readonly string[];
  /** 連続窓のノード数。`ops.length` との差が窓内 passthrough の本数。 */
  readonly windowSize: number;
  /** 窓の先頭ノードの位置（ステップ列を展開したノード順の添字）。 */
  readonly nodeIndex: number;
  /** 鎖の最終ノードの出力名（IR 側で窓を引き当てる手掛かり）。 */
  readonly outputName: string;
};

/**
 * 窓内のノード 1 本を鎖に入れてよいか（= その出力が「消費者ちょうど 1 本・graph output で
 * ない」か）。
 *
 * MUST: privacy の綴りをここに書き写さず、{@link internalsArePrivate} にそのまま判定させる
 * — そのノードを先頭に置いた 2 本の鎖と見なせば、判定対象は 1 本目の出力だけになる
 * （2 本目は鎖の末尾なので見られない）。別に持つと、受理集合が変わったとき候補表だけが
 * 古い基準で「畳める」と言い続ける。
 */
const outputStaysInternal = (
  step: NodePlan,
  tail: NodePlan,
  context: FusionContext,
): boolean => internalsArePrivate([step, tail], context);

/**
 * 連続窓のうち**畳む部分列**（鎖）を求める。窓の最終ノードへ流れ込むノードのうち、出力が
 * 窓の内側で閉じているものだけが鎖で、残りが窓内 passthrough
 * （{@link passthroughIsIndependent} の対象）。
 *
 * この分け方は窓内 passthrough を持つ現行 2 ルールの実測形と一致する:
 *
 * - RoPE の cos / sin 表の `sym_prefix_slice` は鎖の `mul` へ流れ込むが、表は他の層でも
 *   読まれる（消費者が 2 本以上）ので鎖に入らない。
 * - adaLN の gate の `reshape` は鎖の外で消費される（窓の最終ノードへ流れ込まない）ので
 *   鎖に入らない。
 *
 * 鎖の先頭が窓の先頭でない窓は `undefined` — 同じ鎖を「前に無関係なノードを足した窓」から
 * 何度も数えないための足切りで、これで鎖 1 本は必ず窓 1 つ（先頭 = 鎖の先頭・末尾 = 鎖の
 * 末尾）に対応する。
 */
const chainWithinWindow = (
  window: readonly NodePlan[],
  context: FusionContext,
): readonly NodePlan[] | undefined => {
  const last = window[window.length - 1];
  const needed = new Set(last.node.ins);
  const chain = [last];
  for (let at = window.length - 2; at >= 0; at -= 1) {
    const step = window[at];
    if (!step.outputs.some((out) => needed.has(out.name))) continue;
    // 出力が窓の外へ出るノードは鎖に入れない（畳むと値が消える）。その入力を needed へ
    // 足さないので、このノード経由でしか最終ノードへ届かない上流も自動的に鎖から外れる。
    if (!outputStaysInternal(step, last, context)) continue;
    chain.push(step);
    for (const name of step.node.ins) needed.add(name);
  }
  chain.reverse();
  return chain[0] === window[0] ? chain : undefined;
};

/**
 * **融合が掴まなかった連続窓の候補列挙**（tools/fusion-hints の 1 段目）。
 *
 * `planFusions` が `kind: "node"` のまま残した素のノードの連続列を、長さ 2〜`maxWindow` の窓で
 * 走り、全ルール共通の適格条件を通る窓だけを返す。掛ける述語は融合と同じ 5 本:
 * 窓に {@link windowIsSingleOutput} と {@link windowTouchesState}、鎖に {@link allF32}、
 * 鎖と窓内 passthrough の分け方そのものに {@link internalsArePrivate}（{@link chainWithinWindow}
 * が 1 ノードずつ掛ける）と {@link passthroughIsIndependent}。
 *
 * MUST: 述語を融合パス層（`fusion-rule.ts` / `fusion-rules/` / ここ）の外へ持ち出さない —
 * ADR 0040 決定 1 の「唯一の判定点」は融合の判定だけでなく適格条件の綴りにも掛かる。列挙器を
 * 融合パス層の外に置いて述語を持ち出すと、受理集合の変更が候補表に追随しない（表だけが古い
 * 基準で「畳める」と言い続ける）。
 * MUST: 融合ステップを跨いだ窓は作らない（既に掴めている鎖を候補として二重に数えない）。
 * 走査は素のノードの**連続する走り**ごとに閉じる。
 *
 * NOTE: 返るのは候補であって設計ではない。長い窓の接頭辞も別の候補として出る（n-gram 列挙の
 * 性質）ので、読むときは同じ `nodeIndex` の最長窓を見る。
 */
export const enumerateUnfusedWindows = (
  steps: readonly ExecStep[],
  context: FusionContext,
  maxWindow: number,
): readonly UnfusedWindow[] => {
  if (!Number.isSafeInteger(maxWindow) || maxWindow < 2) {
    throw new ExecutionError(`窓幅の上限 ${maxWindow} は 2 以上の整数でなければならない`);
  }
  const found: UnfusedWindow[] = [];
  let run: NodePlan[] = [];
  let runStart = 0;
  let nodeIndex = 0;
  const sweepRun = (): void => {
    for (let start = 0; start < run.length; start += 1) {
      for (let size = 2; size <= maxWindow && start + size <= run.length; size += 1) {
        const window = run.slice(start, start + size);
        if (!windowIsSingleOutput(window) || windowTouchesState(window)) continue;
        const chain = chainWithinWindow(window, context);
        if (chain === undefined || !allF32(chain)) continue;
        const folded = new Set(chain);
        const passthrough = window.filter((step) => !folded.has(step));
        if (!passthroughIsIndependent(chain, passthrough)) continue;
        found.push({
          ops: chain.map((step) => step.node.op),
          windowSize: size,
          nodeIndex: runStart + start,
          outputName: chain[chain.length - 1].outputs[0].name,
        });
      }
    }
    run = [];
  };
  for (const step of steps) {
    if (step.kind === "node") {
      if (run.length === 0) runStart = nodeIndex;
      run.push(step.plan);
      nodeIndex += 1;
      continue;
    }
    sweepRun();
    nodeIndex += step.nodeCount;
  }
  sweepRun();
  return found;
};
