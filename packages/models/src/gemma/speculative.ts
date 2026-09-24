/**
 * Gemma 4 の MTP drafter を 1 回回す**内部 API**（ADR
 * [0096](../../../../docs/decisions/0096-speculative-decoding.md) 決定 1〜3・段 2 §5）。
 *
 * 部品は 3 本 — 借り手 context を開く {@link openDrafterContext}、1 サイクルぶんの draft を採る
 * {@link draftOnce}、その 2 本を 1 つの `DraftFace` に束ねる {@link openGemma4DraftFace}。投機
 * ループ（`src/generation/sequence.ts`）が見るのは束ねた面だけである（生成面は gemma4 も Session
 * も知らない — ADR 0083）。
 *
 * MUST: `mod.ts` / `./gemma` サブパスには出さない（ADR 0008 の薄い面）。GenerationContext は
 * sequence ごとの内部実体で公開面に無い（ADR 0083 決定 3）ので、この 2 本を公開面へ出すと
 * 「利用者が握れない値を引数に取る関数」が並ぶ。
 *
 * ## drafter が回るための前提（貸し手 = target との結線）
 *
 * - drafter Session は貸し手 Session の embedding 表 1 本を**借りている**（`sharedWeights` —
 *   バイトは 1 つも複製されない）。束ねるのは `pipeline.ts` の構築で、ここは走らせるだけ。
 * - 借り手 context は貸し手 context の KV スロット（l13 sliding / l14 full）を**名前で**束ねる。
 *   draft の query は論理位置 `P`（= `context.pastLength`）— 貸し手が最後に確定させた token は
 *   まだ KV に入っていない frontier で、その位置に居る。ホストが渡す RoPE の 1 行も同じ位置。
 * - 貸し手の run / commit と直列化されるのは runtime の仕事（借り手 run が貸し手の run リースを
 *   取る）。**貸し手に未 commit の run が残っていると draft は拒否される** — draft は commit の
 *   後に採る。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 */

import type { GenerationContext, PreparedModel, Session, Tensor } from "@karume/runtime";

import type { DraftCycle, DraftFace } from "../generation/speculation.ts";
import {
  GEMMA4_ROPE_LAYER_TYPES,
  GEMMA4_ROPE_PARTS,
  gemma4RopeInputName,
  gemma4RopeInputNames,
  gemma4RopeInputs,
  type Gemma4RopeSpec,
} from "./rope.ts";

/**
 * drafter が 1 サイクルで出す draft token の本数（= MTP head の段数）。
 *
 * 受けるのは**この値で焼いた資産だけ**である（グラフ出口の本数が段数そのもの）。投機ループが
 * 1 cycle で使う本数 `k` はこの範囲内で選べる（`1..GEMMA4_DRAFT_STEPS` — 予算末尾ではさらに
 * 縮む）が、資産側の段数は固定である。
 */
export const GEMMA4_DRAFT_STEPS = 3;

/**
 * この層が読むグラフ宣言（`PreparedModel["graph"]` = IR そのもの）。
 *
 * 生成面の `GenerationGraph`（`../generation/program.ts`）の部分集合では足りない — 見るのは
 * states の `external` と initializers の `shared` で、どちらも IR 本体にしか無い欄である。
 */
type DrafterGraph = PreparedModel["graph"];

/** drafter グラフの入力名（正本は `tools/export-recipes/gemma4/export_drafter.py`）。 */
const TOKEN_INPUT = "token";
const HIDDEN_INPUT = "hidden";

/**
 * drafter グラフの入力名（**順序が契約** — 確定した token / それを出した行の hidden / RoPE 4 本）。
 *
 * 名前ではなく順序まで見るのは、焼く側（`DrafterWrapper.forward` の引数順）と読む側が
 * 同じ列を主張していることの検出器にするため。順序だけずれた資産は名前で引く限り黙って通る。
 */
const DRAFTER_INPUTS: readonly string[] = [TOKEN_INPUT, HIDDEN_INPUT, ...gemma4RopeInputNames()];

/**
 * drafter 1 本ぶんの実行に要るもの。
 *
 * MUST: Session だけでは足りない — 出口の**名前**（段順）も RoPE の式も Session からは引けない
 * （`Session` はグラフを公開しない）ので、admission が確定させたものをここに束ねて持ち回る。
 * 席を分けて持つと「別の資産の出口名で読む」形が書ける。
 */
export type Gemma4Drafter = {
  readonly session: Session;
  /**
   * drafter コンテナのグラフ宣言（**見積り専用** — `Gemma4Pipeline.estimateSessionMemory` が
   * drafter の常駐重みと借り手 context の state をこれから引く）。
   *
   * MUST: `PreparedModel` ではなくグラフだけを持つ（`Gemma4State.graph` と同じ MUST — 全量の
   * バイト列を掴んだままにしない）。
   */
  readonly graph: DrafterGraph;
  /** draft 出口 3 本の名前（**段順** = グラフの宣言順）。 */
  readonly outputs: readonly string[];
  /** 位置 1 行ぶんの cos / sin を作る宣言（貸し手と同じ `pipelineConfig.rope`）。 */
  readonly rope: Gemma4RopeSpec;
  /** 入力 `hidden` の幅（= 貸し手の最終 norm 後 hidden の幅）。 */
  readonly hiddenSize: number;
};

/** {@link admitGemma4Drafter} が確定させる材料（Session を組む前に決まるもの）。 */
export type Gemma4DrafterAdmission = {
  readonly outputs: readonly string[];
  readonly hiddenSize: number;
  /**
   * `createSession` の `sharedWeights` に渡す対応表 — **借り手の initializer 名 → 貸し手の
   * initializer 名**。値の側で `lender.exportWeight(...)` を引く。
   */
  readonly sharedWeights: Readonly<Record<string, string>>;
};

/** 形の突合（記号次元は綴りのまま比べる — 貸し手と借り手で同じ記号名 MUST）。 */
const sameShape = (
  left: readonly (number | string)[],
  right: readonly (number | string)[],
): boolean => left.length === right.length && left.every((dim, index) => dim === right[index]);

const showShape = (shape: readonly (number | string)[]): string => `[${shape.join(",")}]`;

/**
 * drafter グラフの入力 6 本（名前・順序・dtype・形）を見る。
 *
 * `hidden` の幅は**貸し手から**渡す（drafter 側の宣言を正としない — 別世代の target と組んだ
 * 資産はここで落ちる）。RoPE 4 本の幅は `pipelineConfig.rope.<層種>.headDim` と突き合わせる
 * （sliding 256 と full 512 の引き違いは、ホストが表を渡す初 run まで落ちない誤りである —
 * `pipeline.ts` の `assertRopeInputShapes` と同じ理由）。
 */
const assertDrafterInputs = (
  where: string,
  graph: DrafterGraph,
  rope: Gemma4RopeSpec,
  hiddenSize: number,
): void => {
  const names = graph.inputs.map((input) => input.name);
  if (names.length !== DRAFTER_INPUTS.length || !names.every((n, i) => n === DRAFTER_INPUTS[i])) {
    throw new Error(
      `${where}: drafter グラフの入力が [${names.join(", ")}]` +
        `（順序も含めて [${DRAFTER_INPUTS.join(", ")}] であること）`,
    );
  }
  const expected = new Map<string, { dtype: string; shape: readonly (number | string)[] }>([
    [TOKEN_INPUT, { dtype: "i32", shape: [1, 1] }],
    [HIDDEN_INPUT, { dtype: "f32", shape: [1, hiddenSize] }],
  ]);
  for (const layerType of GEMMA4_ROPE_LAYER_TYPES) {
    for (const part of GEMMA4_ROPE_PARTS) {
      expected.set(gemma4RopeInputName(layerType, part), {
        dtype: "f32",
        shape: [1, 1, rope[layerType].headDim],
      });
    }
  }
  for (const input of graph.inputs) {
    const want = expected.get(input.name);
    if (want === undefined) throw new Error(`${where}: 想定外の drafter 入力 '${input.name}'`);
    if (input.dtype !== want.dtype || !sameShape(input.shape, want.shape)) {
      throw new Error(
        `${where}: drafter 入力 '${input.name}' が ${input.dtype} ${showShape(input.shape)}` +
          `（${want.dtype} ${showShape(want.shape)} が要る）`,
      );
    }
  }
};

/** draft 出口 3 本（段順・`[1,1,1]` の i32）。 */
const assertDrafterOutputs = (where: string, graph: DrafterGraph): readonly string[] => {
  if (graph.outputs.length !== GEMMA4_DRAFT_STEPS) {
    throw new Error(
      `${where}: drafter グラフの出口が ${graph.outputs.length} 本` +
        `（配布形の drafter は ${GEMMA4_DRAFT_STEPS} 段で焼かれている — 段数を変えるのは再 export）`,
    );
  }
  for (const name of graph.outputs) {
    const value = graph.values[name];
    if (value === undefined) throw new Error(`${where}: drafter 出口 '${name}' の値情報が無い`);
    if (value.dtype !== "i32" || !sameShape(value.shape, [1, 1, 1])) {
      throw new Error(
        `${where}: drafter 出口 '${name}' が ${value.dtype} ${showShape(value.shape)}` +
          `（i32 [1,1,1] の draft token が要る）`,
      );
    }
  }
  return [...graph.outputs];
};

/**
 * 借り物スロットが**貸し手と同名・同形**であることを見る（借り手 context の束ねが名前で行われる
 * ので、綴りが違えば runtime が落とすが、その時点では既に Session を 2 本組んでいる）。
 *
 * MUST: 全スロットが external であること。1 本でも自前スロットがあると、借り手は「空の過去」を
 * 自分で確保して読む（値は出るが中身が無い = 沈黙誤値）。
 */
const assertDrafterStates = (where: string, graph: DrafterGraph, target: DrafterGraph): void => {
  const names = Object.keys(graph.states);
  if (names.length === 0) {
    throw new Error(`${where}: drafter グラフに state スロットが無い（target の KV を読まない）`);
  }
  for (const name of names) {
    const slot = graph.states[name];
    if (!slot.external) {
      throw new Error(
        `${where}: drafter の state スロット '${name}' が external でない` +
          `（借り物の実体は貸し手 context にある — 自前確保は空の過去を読む）`,
      );
    }
    const lender = target.states[name];
    if (lender === undefined) {
      throw new Error(
        `${where}: drafter が読む external スロット '${name}' が target グラフに無い` +
          `（target のスロット: ${Object.keys(target.states).join(" / ")}）`,
      );
    }
    if (slot.dtype !== lender.dtype || !sameShape(slot.shape, lender.shape)) {
      throw new Error(
        `${where}: external スロット '${name}' が ${slot.dtype} ${showShape(slot.shape)}` +
          `（target 側は ${lender.dtype} ${showShape(lender.shape)}）`,
      );
    }
  }
};

/**
 * 共有 initializer（バイトを持たない宣言）を貸し手の initializer 名へ解決する。
 *
 * 借り手の名前は**貸し手の initializer 名と同じ**（docs/ir-v2.md「共有 initializer」— 名前が
 * 実体の鍵なので、対応表は恒等写像）。貸し手グラフに同名の実体が無ければ fail loudly — 通すと
 * `createSession` が「shared 宣言に対して重みが不足」と落ちるだけで、**どの表を借りそこねたか**が
 * 残らない。格納の互換（貸し手の codec を借り手の消費席で実行できるか）は借り手 Session 構築時の
 * 門（runtime の `resolveSharedWeights`）が見る。
 */
const resolveSharedWeights = (
  where: string,
  graph: DrafterGraph,
  target: DrafterGraph,
): Readonly<Record<string, string>> => {
  let shared: Record<string, string> = {};
  for (const name of Object.keys(graph.initializers)) {
    if (graph.initializers[name].shared === undefined) continue;
    const lender = target.initializers[name];
    if (lender === undefined || lender.shared !== undefined) {
      throw new Error(
        `${where}: drafter の共有 initializer '${name}' に対応する実体が target グラフに無い` +
          `（借り手の名前は貸し手の initializer 名と同じであること）`,
      );
    }
    shared = { ...shared, [name]: name };
  }
  if (Object.keys(shared).length === 0) {
    throw new Error(
      `${where}: drafter グラフに共有 initializer が 1 本も無い` +
        `（target の埋め込み表を借りる宣言が焼かれていない）`,
    );
  }
  return shared;
};

/**
 * drafter コンテナを「この target と組める MTP head か」で見る門（**重みの part を取る前**）。
 *
 * 見るのは 5 つ — ①入力 6 本（順序込み）②出口 3 本 ③全スロットが external で貸し手と同名同形
 * ④共有 initializer が貸し手の initializer へ解決できる ⑤記号は貸し手の容量記号 1 本だけ。
 * どれも「別世代の target と組んだ drafter」を、GB 級の重みを落とす前に落とすためのものである。
 *
 * NOTE: `requiredLimits` は見ない — drafter は target と**同じ quant 席**に居る（manifest の
 * `quants.<名>.weights` が両方を指す）ので、宣言は 1 つしかない（その 1 つを DL 前に見るのは
 * `Gemma4Pipeline.fromPretrained` の admission 閉包）。
 */
export const admitGemma4Drafter = (
  where: string,
  graph: DrafterGraph,
  target: {
    readonly graph: DrafterGraph;
    readonly rope: Gemma4RopeSpec;
    readonly hiddenSize: number;
    readonly capacitySymbol: string;
  },
): Gemma4DrafterAdmission => {
  assertDrafterInputs(where, graph, target.rope, target.hiddenSize);
  const outputs = assertDrafterOutputs(where, graph);
  assertDrafterStates(where, graph, target.graph);
  const sharedWeights = resolveSharedWeights(where, graph, target.graph);
  if (graph.symbols.length !== 1 || graph.symbols[0] !== target.capacitySymbol) {
    throw new Error(
      `${where}: drafter グラフの記号が [${graph.symbols.join(", ")}]` +
        `（貸し手から継承する容量記号 '${target.capacitySymbol}' 1 本であること）`,
    );
  }
  return { outputs, hiddenSize: target.hiddenSize, sharedWeights };
};

/**
 * 貸し手（target）の生成 context を借りる drafter 用 context を開く。
 *
 * 借り手は自前の KV を 1 バイトも確保しない（束ねるのは貸し手のスロットバッファ）。
 * `chunkLength` は 1 ちょうど・`bindings` は貸し手のものを継承する（どちらも runtime の門）。
 *
 * MUST: **借り手を先に畳む**。借り手が生きている間、貸し手 context の `dispose()` は
 * fail loudly になる（借り手の bind group が貸し手のスロットを掴んでいる）。
 */
export const openDrafterContext = (
  drafter: Gemma4Drafter,
  target: GenerationContext,
): Promise<GenerationContext> =>
  drafter.session.createGenerationContext({ chunkLength: 1, borrow: target });

/**
 * 1 サイクルぶんの入力（`P = context.pastLength` = KV に入っている行数）。
 *
 * 組は「最後に確定した token と、**それを出した行**の hidden」— drafter 自身の段間再帰
 * （token と、その token を出した段の hidden を次段へ送る）と同じ組み方で、torch drafter の
 * golden もこの組で採っている。
 *
 * MUST: 生成面の `DraftCycle`（`../generation/speculation.ts`）**そのもの**である。写した型を
 * 持つと、欄の意味が片方だけ改まったときに型検査が通り続ける。
 */
export type Gemma4DraftCycle = DraftCycle;

/**
 * 貸し手 context 1 本ぶんの `DraftFace` を開く（投機ループ〈`createGenerationSequence` の
 * `speculative.open`〉が sequence 生成時に 1 度だけ呼ぶ席）。
 *
 * 束ねるのは 3 つ — 段数（配布形に焼かれた出口の本数）・{@link draftOnce}・借り手 context の
 * 返却である。生成面は gemma4 も Session も知らないので、借り手 context の寿命はこの閉包が持つ
 * （`dispose` を呼ぶ順序 = 借り手 → 貸し手 は sequence 側の MUST）。
 */
export const openGemma4DraftFace = async (
  drafter: Gemma4Drafter,
  target: GenerationContext,
): Promise<DraftFace> => {
  const borrowed = await openDrafterContext(drafter, target);
  return {
    steps: drafter.outputs.length,
    draft: (cycle: DraftCycle): Promise<Int32Array<ArrayBuffer>> =>
      draftOnce(drafter, borrowed, cycle),
    dispose: (): Promise<void> => borrowed.dispose(),
  };
};

/**
 * draft を 1 サイクル採る（k = 3 段ぶんの token を 1 run で出す）。
 *
 * 借り手 run は貸し手の run リースを取るので、貸し手の run / commit / rewind と直列化される。
 * **貸し手に未 commit の run が残っていれば拒否される**（draft は commit の後）。借り手は論理長を
 * 1 つも動かさない（`state_append` が 0 本 = KV に 1 行も書かない）ので、棄却されても巻き戻す
 * ものが無い。
 *
 * MUST: `hidden` は**`token` を出した行**（位置 `P−1`）であること — 別の行を渡しても形は合うので
 * 例外にならず、draft の質だけが静かに落ちる（この層で検出できるのは幅だけ）。
 */
export const draftOnce = async (
  drafter: Gemma4Drafter,
  borrowed: GenerationContext,
  cycle: Gemma4DraftCycle,
): Promise<Int32Array<ArrayBuffer>> => {
  const where = "gemma4 draftOnce";
  const { token, hidden, position } = cycle;
  if (!Number.isSafeInteger(token) || token < 0) {
    throw new Error(`${where}: 直前 token ${token} が非負整数でない`);
  }
  if (!Number.isSafeInteger(position) || position < 0) {
    throw new Error(`${where}: 位置 ${position} が非負整数でない`);
  }
  if (hidden.length !== drafter.hiddenSize) {
    throw new Error(
      `${where}: hidden の要素数 ${hidden.length} が幅 ${drafter.hiddenSize} と違う`,
    );
  }
  const inputs: Record<string, Tensor> = {
    [TOKEN_INPUT]: { dtype: "i32", shape: [1, 1], data: Int32Array.of(token) },
    [HIDDEN_INPUT]: { dtype: "f32", shape: [1, drafter.hiddenSize], data: hidden },
    // 位置 P の 1 行（PLE は要らない — drafter は主表も per-layer 表も引かない）。
    ...gemma4RopeInputs(drafter.rope, [position]),
  };
  const outputs = await drafter.session.run(inputs, {}, { context: borrowed, queryLength: 1 });
  const draft = new Int32Array(new ArrayBuffer(drafter.outputs.length * 4));
  drafter.outputs.forEach((name, step) => {
    const tensor = outputs[name];
    if (tensor === undefined) throw new Error(`${where}: 出口 '${name}' が run の結果に無い`);
    if (tensor.dtype !== "i32" || tensor.data.length !== 1) {
      throw new Error(
        `${where}: 出口 '${name}' が ${tensor.dtype} ${tensor.data.length} 要素` +
          `（i32 1 要素の draft token が要る）`,
      );
    }
    draft[step] = tensor.data[0];
  });
  return draft;
};
