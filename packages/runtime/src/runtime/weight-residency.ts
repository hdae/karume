/**
 * 重みの**常駐分類**（席）— グラフ宣言だけで決まる純関数プランナと、その結果を GPU 実体へ
 * 結び付けた Session 側の判別 union。
 *
 * MUST: 分類の正本はここ 1 本（{@link planWeightResidency}）。Session 構築（session-build.ts）と
 * 見積り（estimate.ts）が別々に「適格判定 + 格納 dtype の分岐」を書くと、片方だけ直された
 * ときに **見積りと実ロードが別のモデルを説明する**（例外も警告も出ない）。
 * MUST: 実テンソルを見ない。バイト数は宣言 shape と格納メタデータから導き、実バイトとの一致は
 * 容器の突合門（合流層の payload 長検査と、構築相の「宣言由来 N バイト」検査）が保証する。
 *
 * 席から「実際に確保される GPU バッファ」への写像（{@link planWeightBuffers}）と、その寸法を
 * device の絶対上限と突き合わせる門（{@link assertWeightsWithinLimits}）も同じ理由でここに置く
 * — 確保する側（session-build.ts）と数える側（estimate.ts）が別々に席を展開すると、片方だけ直された
 * ときに検査・見積り・実ロードが別の寸法を主張する。
 */

import {
  codecLayout,
  type CodecName,
  groupScaleShape,
  payloadBytes,
  scaleBytes,
} from "../format/container/codecs.ts";
import type { IrGraph } from "../format/ir.ts";
import { toSizeClass } from "../gpu/arena.ts";
import type { GpuContext } from "../gpu/device.ts";
import { RUNTIME_INTERNAL } from "../gpu/device.ts";
import { numel } from "../ops.ts";
import {
  eligibleCompressedInitializers,
  ExecutionError,
  i2EligibleInitializers,
  i4EligibleInitializers,
  weightChannelAxes,
} from "./plan.ts";

/**
 * 圧縮のまま GPU 常駐した重み 1 本（Session の重み台帳と対の索引 — カーネル変種の選択と
 * 追加束縛がここだけで決まる）。
 *
 * MUST: payload の `GPUBuffer` は持たない（重み台帳 `weightBuffers` が所有 — 二重保持にすると
 * 「どちらが本物か」が生まれる）。scale だけは重み台帳に載らない実体なのでここが所有する。
 * MUST: 席ごとの付随情報を**型で**要求する（i8 / i2 / i4 は scale 必須・i4 は group 長必須）。
 * 3 本の並列 Map で持つと「i4 なのに group 長が無い」形が型の上では作れてしまい、実行時検査
 * だけが最後の砦になる。
 */
export type ResidentWeight =
  | { readonly storage: "f16" }
  | { readonly storage: "i8" | "i2"; readonly scale: GPUBuffer }
  | { readonly storage: "i4"; readonly scale: GPUBuffer; readonly groupSize: number };

/**
 * initializer 1 本の常駐分類（{@link planWeightResidency} の値）。
 *
 * 席は 7 つ:
 * - `raw` — 圧縮しない格納（f32 / i32 / bf16）を生バイトのまま GPU 常駐（session-build.ts の構築相）
 * - `f16` / `i8` / `i2` / `i4` — 圧縮のまま常駐し dequant はカーネル内（ADR 0018 / 0019 / 0097 / 0069）
 * - `expanded` — 適格外でロード時に CPU で f32 展開（正しさは保たれ VRAM 削減はゼロ）
 * - `shared` — 借り物の重み（ADR 0096 段 2 §1.3 — 貸し手 Session の GPU バッファを束ねる）
 *
 * `payloadBytes` は**格納バイト列そのものの長さ**（整列詰め物もバッファ床も含まない）。
 * 整列は転送側（`alignF16Payload` / `alignI8Payload`）とアリーナ（`toSizeClass`）が持つ。
 */
export type WeightResidency =
  | { readonly seat: "raw"; readonly payloadBytes: number }
  | { readonly seat: "f16"; readonly payloadBytes: number }
  | {
    readonly seat: "i8" | "i2";
    readonly payloadBytes: number;
    readonly scaleBytes: number;
    /** 行（per-channel scale が掛かる）の軸 — 宣言 `storage.rowAxis`（消費側 op の軸と一致することは検査済み）。 */
    readonly rowAxis: 0 | 1;
  }
  | {
    readonly seat: "i4";
    readonly payloadBytes: number;
    readonly scaleBytes: number;
    readonly groupSize: number;
  }
  | {
    readonly seat: "expanded";
    readonly payloadBytes: number;
    /** CPU で f32 へ展開した後のバイト数（常駐するのはこちら）。 */
    readonly expandedBytes: number;
  }
  | {
    /**
     * **借り物の重み**（ADR 0096 段 2 §1.3 の共有 initializer）。この Session は 1 バイトも
     * 確保・転送せず、実体は貸し手 Session の GPU バッファをそのまま束ねる
     * （`SessionOptions.sharedWeights`）。
     */
    readonly seat: "shared";
    /**
     * 借り手側の消費（適格判定）。借り手は格納を宣言しない（正本は貸し手の常駐重み）ので、
     * 期待する席は貸し手の codec が分かる構築時に {@link resolveSharedWeights} が導く。
     */
    readonly eligible: boolean;
    readonly i4Eligible: boolean;
    readonly i2Eligible: boolean;
    /** 消費側 op から決まる per-channel scale の軸（重みスロットで消費されないときは無い）。 */
    readonly consumerAxis?: 0 | 1;
  };

/**
 * グラフだけから initializer ごとの常駐分類を決める（GPU も safetensors も要らない純関数）。
 *
 * 分類は 3 点 — 適格判定 {@link eligibleCompressedInitializers}、i4 だけ
 * {@link i4EligibleInitializers} との積（ADR 0069 決定 5）、格納 dtype ごとの分岐。
 *
 * MUST: `graph.initializers` の**全件**を返す（宣言順）。欠けを許すと、消費側が「表に無い =
 * f32 で読む」と解釈する既定と区別できなくなる。
 */
export const planWeightResidency = (graph: IrGraph): ReadonlyMap<string, WeightResidency> => {
  // 圧縮格納のまま上げてよい initializer（消費が重みスロットだけ — ADR 0018）。
  const eligible = eligibleCompressedInitializers(graph);
  // i4 の適格はさらに狭く「重みスロットでの消費が linear / embedding / conv1d(groups==1) だけ」
  // （ADR 0069 決定 5 とその追補 — 展開経路を持つカーネルはこの 3 つ）。
  const i4Eligible = i4EligibleInitializers(graph);
  const i2Eligible = i2EligibleInitializers(graph);
  // per-channel scale の軸は**宣言**（`storage.rowAxis` — container-v1 §6.1）が正本。消費側 op から
  // 決まる軸（ADR 0019）は突合点として残す — 宣言と食い違うと GPU 常駐経路が scale を別の軸に
  // 当てる沈黙誤値になる。
  const consumerAxes = weightChannelAxes(graph);
  const plan = new Map<string, WeightResidency>();
  for (const [name, initializer] of Object.entries(graph.initializers)) {
    const where = `initializer '${name}'`;
    // initializer の宣言 shape は数値のみ（パーサが保証 — 記号次元は拒否）。
    const shape = graph.values[name].shape.map(Number);
    const count = numel(shape);
    // 共有 initializer は席だけを決める（バイト数は 1 つも数えない — 確保するのは貸し手）。
    // 借り手は格納を宣言しないので、期待する席は貸し手の codec が分かる借り手構築時に
    // 導く（{@link resolveSharedWeights}）— ここは借り手側の消費（適格判定）だけを持つ。
    if (initializer.shared !== undefined) {
      const consumerAxis = consumerAxes.get(name);
      plan.set(name, {
        seat: "shared",
        eligible: eligible.has(name),
        i4Eligible: i4Eligible.has(name),
        i2Eligible: i2Eligible.has(name),
        ...(consumerAxis === 0 || consumerAxis === 1 ? { consumerAxis } : {}),
      });
      continue;
    }
    const { codec, groupSize, rowAxis } = initializer.storage;
    const layout = codecLayout(codec);
    let bytes: number;
    try {
      bytes = payloadBytes(codec, count, where);
    } catch (cause) {
      throw new ExecutionError(cause instanceof Error ? cause.message : String(cause));
    }
    if (layout === "f32" || layout === "i32" || layout === "bf16") {
      // 圧縮しない格納は生バイトがそのまま GPU 表現。
      plan.set(name, { seat: "raw", payloadBytes: bytes });
      continue;
    }
    const resident = layout === "i4"
      ? eligible.has(name) && i4Eligible.has(name)
      : layout === "i2"
      ? eligible.has(name) && i2Eligible.has(name)
      : eligible.has(name);
    if (!resident) {
      plan.set(name, { seat: "expanded", payloadBytes: bytes, expandedBytes: count * 4 });
      continue;
    }
    if (layout === "f16") {
      plan.set(name, { seat: "f16", payloadBytes: bytes });
      continue;
    }
    // 量子化 codec: rowAxis / groupSize の存在と値域は合流層が保証済み。存在は型の上で
    // だけ optional なので、黙って読み飛ばさず言い直す。
    const axis = rowAxis ?? 0;
    if (groupSize === undefined) {
      throw new ExecutionError(`${where}: 量子化 codec '${codec}' なのに groupSize が無い`);
    }
    const consumerAxis = consumerAxes.get(name);
    if (consumerAxis !== undefined && consumerAxis !== axis) {
      throw new ExecutionError(
        `${where}: 宣言の rowAxis ${axis} が消費側 op のチャネル軸 ${consumerAxis} と違う`,
      );
    }
    const rows = shape[axis];
    if (rows === undefined) {
      throw new ExecutionError(`${where}: 重み [${shape.join(",")}] に行の軸 ${axis} が無い`);
    }
    const scale = scaleBytes(numel(groupScaleShape(shape, axis, groupSize)), where);
    if (layout === "i8" || layout === "i2") {
      plan.set(name, { seat: layout, payloadBytes: bytes, scaleBytes: scale, rowAxis: axis });
      continue;
    }
    // i4 の group 形は先頭次元を行とする（展開カーネルと `decodeI4` の前提 — ADR 0069 決定 3）。
    if (axis !== 0) {
      throw new ExecutionError(
        `${where}: group 量子化（${codec}）の rowAxis は 0 だけ（宣言は ${axis}）`,
      );
    }
    plan.set(name, { seat: "i4", payloadBytes: bytes, scaleBytes: scale, groupSize });
  }
  return plan;
};

/**
 * 借り手側の消費（適格判定）と貸し手の codec から、借りる実体に**期待する席**を導く。
 *
 * MUST: 借り手側の消費から**独立に**導き直す。貸し手が i4 常駐でも借り手の消費に展開経路が
 * 無ければ席は `expanded` になり、同じバッファが「packed i4 のバイト列」と「f32 の値」の
 * 2 通りに読まれる — 例外は 1 つも出ない。
 */
const expectedSharedSeat = (
  codec: CodecName,
  seat: Extract<WeightResidency, { readonly seat: "shared" }>,
): Exclude<WeightResidency["seat"], "shared"> => {
  const layout = codecLayout(codec);
  if (layout === "f32" || layout === "i32" || layout === "bf16") return "raw";
  const resident = layout === "i4"
    ? seat.eligible && seat.i4Eligible
    : layout === "i2"
    ? seat.eligible && seat.i2Eligible
    : seat.eligible;
  return resident ? layout : "expanded";
};

/**
 * {@link SharedWeight} のランタイム内部面（利用者が触る面ではない）。
 *
 * MUST: 貸し手の実体（バッファ・`ResidentWeight`）は**参照だけ**を持つ。写しを取ると
 * 「どちらが本物か」が生まれ、貸し手が f16 常駐から展開席へ変わったときに借り手だけが
 * 古い席で走る。
 */
export type SharedWeightInternals = {
  /** 貸し手の initializer 名（診断用）。 */
  readonly initializer: string;
  /** 貸し手の device（借り手と同一 MUST — 別 device のバッファは束縛できない）。 */
  readonly gpu: GpuContext;
  /** 貸し手が確保した重み本体のバッファ（借り手は所有しない）。 */
  readonly buffer: GPUBuffer;
  /** 圧縮のまま常駐している場合の席と付随実体（f32 / 生バイト席では undefined）。 */
  readonly resident: ResidentWeight | undefined;
  /** 貸し手の常駐席（借り手の期待席と一致 MUST）。 */
  readonly seat: Exclude<WeightResidency["seat"], "shared">;
  /** i8 / i2 席の行（per-channel scale）の軸（それ以外は undefined）。 */
  readonly rowAxis: 0 | 1 | undefined;
  /** 貸し手の格納 codec（借り手の期待席はこれと借り手側の消費から導く）。 */
  readonly codec: CodecName;
  /** 貸し手の宣言 shape。 */
  readonly shape: readonly number[];
  /** 借用を 1 本積む（借り手 Session の構築が成功したとき）。 */
  retain(): void;
  /** 借用を 1 本返す（借り手 Session の dispose / 構築の失敗）。 */
  release(): void;
};

/**
 * 貸し手 Session が GPU へ載せた重み 1 本への**不透明な参照**（ADR 0096 段 2 §2.2）。
 *
 * `Session.exportWeight(initializerName)` だけが作り、`SessionOptions.sharedWeights` で
 * 借り手 Session へ渡す。借り手はバイトを 1 つも持たず、貸し手のバッファをそのまま束ねる。
 *
 * MUST: 構築の入口は `Session.exportWeight` だけ（`ResidentTensor` と同じ流儀 — 直接
 * 構築すると席の突合と借用計数を迂回できる）。
 */
export class SharedWeight {
  /** ランタイム内部面（利用者が触る面ではない）。 */
  readonly [RUNTIME_INTERNAL]: SharedWeightInternals;

  constructor(internals: SharedWeightInternals) {
    this[RUNTIME_INTERNAL] = internals;
  }
}

/**
 * 借り手グラフの共有 initializer 宣言と、渡された {@link SharedWeight} を突き合わせる
 * （ADR 0096 段 2 §2.2 の門）。返すのは注入すべき組（宣言順）。
 *
 * 見るのは 5 点（借り手は格納を宣言しないので「格納 dtype 一致」の門は無く、代わりに 4 で期待席を導く）:
 * 1. **過不足なし** — 宣言 1 本につき 1 つ（欠けは「バイトの無い重みで走る」、余りは
 *    「渡したつもりの重みが誰にも使われない」）
 * 2. **同一 device** — 別 device のバッファを束ねる bind group は validation で落ちるが、
 *    診断は真因から遠い
 * 3. **宣言 shape 一致** — バイト数だけでは `[2,3]` と `[3,2]` の取り違えが通る
 * 4. **期待席の導出** — 借り手は格納を宣言しないので、貸し手の codec と借り手側の消費（適格判定）から
 *    期待席を導く（`expectedSharedSeat`）。貸し手の実際の席と違えば同じバッファが別の読み方をされる
 * 5. **席の一致**（i8 は per-channel scale の軸まで） — 貸し手が i4 常駐でも借り手の消費に
 *    展開経路が無ければ席は `expanded` で、同じバッファが packed バイトと f32 の 2 通りに
 *    読まれる。i8 の軸違い（embedding と linear）も同じ機序で沈黙誤値になる
 *
 * MUST: 全て fail loudly。5 点とも破れは例外ではなく**別の値**として出る。
 */
export const resolveSharedWeights = (
  graph: IrGraph,
  residency: ReadonlyMap<string, WeightResidency>,
  gpu: GpuContext,
  provided: Readonly<Record<string, SharedWeight>> | undefined,
): readonly { readonly name: string; readonly shared: SharedWeight }[] => {
  const declared = Object.entries(graph.initializers)
    .filter(([, initializer]) => initializer.shared !== undefined)
    .map(([name]) => name);
  const given = Object.keys(provided ?? {});
  const missing = declared.filter((name) => !Object.hasOwn(provided ?? {}, name));
  const surplus = given.filter((name) => !declared.includes(name));
  if (missing.length > 0 || surplus.length > 0) {
    throw new ExecutionError(
      `options.sharedWeights がグラフの shared 宣言と一致しない: 不足 [${
        missing.join(", ")
      }] / 余剰 [${surplus.join(", ")}]` +
        "（shared 宣言 1 本につき 1 つ・過不足なく渡す MUST — ADR 0096 段 2 §2.2）",
    );
  }
  return declared.map((name) => {
    const weight = (provided ?? {})[name];
    // MUST: 型の外から来た値（JS の呼び手）は内部面を読む前に落とす — 読むと `ExecutionError` では
    // なく `TypeError`（`shared.initializer` の参照）や利用者の getter の例外が抜ける。
    if (!(weight instanceof SharedWeight)) {
      throw new ExecutionError(
        `sharedWeights['${name}']: SharedWeight でない値（${typeof weight}）— 貸し手 Session の ` +
          "exportWeight が返したものを渡すこと",
      );
    }
    const shared = weight[RUNTIME_INTERNAL];
    const where = `sharedWeights['${name}'（貸し手の initializer '${shared.initializer}'）`;
    if (shared.gpu !== gpu) {
      throw new ExecutionError(`${where}: 貸し手と借り手の GpuContext（device）が別`);
    }
    const shape = graph.values[name].shape.map(Number);
    if (shape.length !== shared.shape.length || shape.some((d, i) => d !== shared.shape[i])) {
      throw new ExecutionError(
        `${where}: 宣言 shape [${shape.join(",")}] が貸し手の [${shared.shape.join(",")}] と違う`,
      );
    }
    const seat = residency.get(name);
    if (seat === undefined || seat.seat !== "shared") {
      throw new ExecutionError(`${where}: 常駐分類が shared でない（簿記の破れ）`);
    }
    // 借り手は格納を宣言しない — 貸し手の codec と借り手側の消費から期待席を導き、貸し手の
    // 実際の席（と i8 / i2 の行の軸）と突き合わせる。
    const expected = expectedSharedSeat(shared.codec, seat);
    const expectedAxis = expected === "i8" || expected === "i2"
      ? seat.consumerAxis ?? 0
      : undefined;
    if (expected !== shared.seat || expectedAxis !== shared.rowAxis) {
      throw new ExecutionError(
        `${where}: 消費席が貸し手と互換でない（借り手は席 '${expected}'${
          expectedAxis === undefined ? "" : `・行の軸 ${expectedAxis}`
        }・貸し手は席 '${shared.seat}'${
          shared.rowAxis === undefined ? "" : `・行の軸 ${shared.rowAxis}`
        }）— 同じバッファが別の読み方をされる`,
      );
    }
    return { name, shared: weight };
  });
};

/**
 * 席 1 つが GPU に確保させるバッファ 1 本（{@link planWeightBuffers} の要素）。
 *
 * MUST: 「席のどのバイト数が GPU バッファになるか」の分岐はここ 1 本 — 上限検査
 * （{@link assertWeightsWithinLimits}）と見積り（estimate.ts の `weightEstimate`）が席の分岐を
 * 別々に書くと、適格判定が動いたときに片方だけが別の寸法を主張する。
 */
export type WeightBuffer = {
  readonly name: string;
  /**
   * この確保を出した席。**借り物（`shared`）は現れない** — 型でそれを言うことで、席ごとの
   * 網羅 switch（見積りの 3 欄振り分け）が「数えない席」を数える形にならない。
   */
  readonly seat: Exclude<WeightResidency["seat"], "shared">;
  /** `payload` = 重み本体（`expanded` 席は f32 展開後）/ `scale` = companion scale。 */
  readonly kind: "payload" | "scale";
  /** `createBuffer` に渡るバイト数（`toSizeClass` = 4 バイト整列 + 4 バイト床）。 */
  readonly byteLength: number;
  /** 整列前の宣言由来バイト数（見積りの「厳密」欄が数えるのはこちら）。 */
  readonly declaredBytes: number;
};

/**
 * 常駐計画が GPU に確保させるバッファを宣言順に並べる（GPU も device も要らない純関数）。
 *
 * 適格席（f16 / i8 / i4）と生バイト席は payload をそのまま上げ、適格外席（`expanded`）は CPU で
 * f32 展開した**後**のバイト列を上げる（配布形の圧縮バイト数は GPU に載らない）。i8 / i4 は
 * companion scale が payload とは別に**もう 1 本**確保される（session-build.ts の
 * `timedAlloc(Math.max(4, scale.bytes.byteLength))`）。
 */
export const planWeightBuffers = (
  residency: ReadonlyMap<string, WeightResidency>,
): readonly WeightBuffer[] => {
  const buffers: WeightBuffer[] = [];
  const add = (
    name: string,
    seat: WeightBuffer["seat"],
    kind: WeightBuffer["kind"],
    declaredBytes: number,
  ): void => {
    buffers.push({ name, seat, kind, byteLength: toSizeClass(declaredBytes), declaredBytes });
  };
  for (const [name, seat] of residency) {
    // 借り物の席は 1 本も確保しない（実体は貸し手の Session が抱えている）。上限検査からも
    // 見積りからも外れるのはこの 1 行が唯一の分岐点。
    if (seat.seat === "shared") continue;
    add(
      name,
      seat.seat,
      "payload",
      seat.seat === "expanded" ? seat.expandedBytes : seat.payloadBytes,
    );
    if (seat.seat === "i8" || seat.seat === "i4" || seat.seat === "i2") {
      add(name, seat.seat, "scale", seat.scaleBytes);
    }
  }
  return buffers;
};

/**
 * 上限検査が見る device limits（`GpuContext.limits` の部分集合）。
 *
 * MUST: 2 本とも見る。`maxStorageBufferBindingSize ≤ maxBufferSize` は device を計画する側
 * （gpu/device.ts の `planRequiredLimits`）が保っている関係であって、外から渡された
 * `GpuContext` にまで効く保証ではない — 片方だけ見る形にすると関係が崩れた device で沈黙する。
 */
export type WeightLimits = {
  readonly maxStorageBufferBindingSize: number;
  readonly maxBufferSize: number;
};

/** エラー文言の主語（席が `expanded` のときだけ「確保されるのは展開後」を明示する）。 */
const bufferLabel = (buffer: WeightBuffer): string =>
  buffer.kind === "scale"
    ? "scale"
    : buffer.seat === "expanded"
    ? "payload（f32 展開後）"
    : "payload";

/**
 * 重み 1 本ずつの確保寸法を device の絶対上限と突き合わせ、超過があれば**確保の前に**落とす。
 *
 * 動機は「確保失敗の検出は block（item）単位 errorScope に全面依存」（ADR 0108 決定 9）の弱点 —
 * docs/known-issues.md「Metal で out-of-memory errorScope が沈黙する」が名指しした修正候補
 * （重み経路への明示サイズ門）そのもの。errorScope の網では 3 点足りない:
 * ①**実装依存** — 上限超過そのものは validation で捕まる実装が普通だが、同じ経路の
 * out-of-memory scope が黙る device は実在する（M2 実測）。網の成立を実装の報告品質に賭ける形が
 * 残るかぎり「確保失敗 = 無効バッファへの no-op writeBuffer = ゴミを読む」が通り得る。
 * ②**遅い** — 検出は重みを上げ始めた後で、数 GiB 転送してからになる。
 * ③**粒度が粗い** — 名乗れるのは失敗した block までで、どの重みが何バイト超えたのかは出ない。
 * 寸法は確保より前に宣言だけで確定している（常駐計画は prepare 相の純関数）ので、決定論的に
 * 落とせるぶんはここで落とす。
 *
 * MUST: 見るのは**絶対上限との比較だけ**。空き VRAM とは比べない（ADR 0070 決定 5 — WebGPU は
 * 総 / 空き VRAM を露出しないので、比較の形にした瞬間に当て推量になる）。合計サイズも見ない
 * （ここが見ているのは 1 バッファ単位の device 制約で、総量の可否は最終門 = errorScope の担当）。
 * MUST: 超過は**全件列挙して 1 回で落とす**。1 本ずつ落とすと、export をやり直すたびに次の 1 本が
 * 現れる形になり、何本直せば載るのかが最後まで分からない。
 */
export const assertWeightsWithinLimits = (
  residency: ReadonlyMap<string, WeightResidency>,
  limits: WeightLimits,
): void => {
  // 文言が「どの上限か」を必ず名乗るための組（state 側のゲートと同じ形 —
  // generation-context.ts の `limits`）。
  const entries = [
    ["maxStorageBufferBindingSize", limits.maxStorageBufferBindingSize],
    ["maxBufferSize", limits.maxBufferSize],
  ] as const;
  const violations: string[] = [];
  for (const buffer of planWeightBuffers(residency)) {
    const exceeded = entries
      .filter(([, limit]) => buffer.byteLength > limit)
      .map(([key, limit]) => `${key} ${limit} バイトを ${buffer.byteLength - limit} バイト超える`);
    if (exceeded.length === 0) continue;
    violations.push(
      `  - initializer '${buffer.name}' の ${bufferLabel(buffer)}（席 ${buffer.seat}・確保 ` +
        `${buffer.byteLength} バイト）: ${exceeded.join("・")}`,
    );
  }
  if (violations.length === 0) return;
  throw new ExecutionError(
    `重みバッファ ${violations.length} 本が device の上限を超える（確保の前に検出）:\n` +
      `${violations.join("\n")}\n` +
      "1 バッファ単位の上限なので block を分けても（piece に割っても）解消しない — " +
      "より小さい格納 dtype で export し" +
      "直すか、重みを分割してグラフを組み直すこと",
  );
};
